#!/usr/bin/env python3
"""
bench_refit_train.py  (SM8.6 RTX3090 Linux)  v3 — 双缓冲异步 Refit
════════════════════════════════════════════════════════════════════════════
TRT REFIT 训练循环基准测试

A. FP32 基线
B. INT8+K-Refit：每 K 轮 refit 一次（摊薄开销）
C. INT8+Async 双缓冲：refit 在后台线程运行，隐藏在下一轮 rollout 中
   + 在线 scale 更新：PPO 前向 hook 收集激活统计，每轮更新 QDQ scale

运行：python3 bench_refit_train.py 2>&1 | tee refit_bench.log
"""
import sys, os, gc, time, types, warnings, threading
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── Linux mock packages ───────────────────────────────────────────────────
_git = types.ModuleType("git")
class _Repo:
    def __init__(self, *a, **kw): pass
_git.Repo = _Repo; _git.InvalidGitRepositoryError = Exception
sys.modules.setdefault("git", _git)
_td = types.ModuleType("tensordict")
class _TensorDict(dict): pass
_td.TensorDict = _TensorDict
sys.modules.setdefault("tensordict", _td)

import torch
import torch.nn.functional as F
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import tensorrt as trt

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
sys.path.insert(0, os.path.join(WORKDIR, "rsl_rl"))
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ── Config ────────────────────────────────────────────────────────────────
B, PA, MAP, NA = 4096, 96, 33*21*3, 29
DEVICE = torch.device("cuda")

BASE_ONNX    = os.path.join(WORKDIR, "engines", "glad_actor_base.onnx")
REFIT_ONNX   = os.path.join(WORKDIR, "engines", "_refit_int8.onnx")
REFIT_ENG    = os.path.join(WORKDIR, "engines", "_refit_int8.trt")
TIMING_CACHE = os.path.join(WORKDIR, "engines", "trt_timing.cache")

# 当前最优子集（穷举搜索第 1 名）
BEST_SUBSET = {"node_MatMul_69", "node_conv2d", "node_conv2d_1", "node_linear_10"}

# TRT refittable 权重名 → (node_name, param_kind) — 使用 LET 折叠 closure 的层
# 格式：trt_name: (node_name, "weight"/"bias")
# Conv:  W shape [C_out, C_in, kH, kW],  W' = W / α.reshape(1,-1,1,1)
# Gemm:  W shape [C_out, C_in],           W' = W / α.reshape(1,-1)
TRT_TO_LET_NODE: dict = {
    "m.actor.6.weight":   ("node_linear_10", "weight"),
    "m.actor.6.bias":     ("node_linear_10", "bias"),
    "m.map_cnn.0.weight": ("node_conv2d",    "weight"),
    "m.map_cnn.0.bias":   ("node_conv2d",    "bias"),
    "m.map_cnn.3.weight": ("node_conv2d_1",  "weight"),
    "m.map_cnn.3.bias":   ("node_conv2d_1",  "bias"),
}

INT8_MATMUL_OK = {"node_MatMul_50","node_MatMul_59","node_MatMul_69","node_MatMul_71"}
EXCLUDE_ALWAYS = {
    "node_native_batch_norm__0","node_native_batch_norm_1__0",
    "node_softmax","node_softmax_1","node_topk__1",
    "node_select","node_select_1","node_bmm","node_bmm_1","node_bmm_2",
}

# 训练仿真参数
N_ITERS       = 8     # 模拟轮数（含 warmup）
N_STEPS       = 24    # 每轮 rollout 步数（标准 PPO）
N_MINIBATCHES = 4     # PPO minibatch 数
MINIBATCH_SZ  = B // N_MINIBATCHES
N_WARMUP      = 2     # 热身轮数（不计入统计）


# ════════════════════════════════════════════════════════════════════════════
# 模型构建
# ════════════════════════════════════════════════════════════════════════════

def build_model():
    dummy = {"policy_obs": torch.zeros(1, PA+MAP), "critic_obs": torch.zeros(1, PA+3+MAP)}
    grps  = {"policy": ["policy_obs"], "critic": ["critic_obs"]}
    return ActorCriticEncoder(
        obs=dummy, obs_groups=grps, num_actions=NA,
        map_scan_dim=(33,21,3), mha_dim=64, num_heads=16,
        cnn_downsample=True, use_global_context=True, topk=32,
    ).to(DEVICE)


# ════════════════════════════════════════════════════════════════════════════
# 在线激活统计追踪（PPO 前向时 hook 收集，EMA 平滑）
# ════════════════════════════════════════════════════════════════════════════

class ActivationTracker:
    """
    在 PPO update 前向 pass 中收集各层激活统计，用于在线更新 QDQ scale 和 LET α, β。

    支持的层（BEST_SUBSET）：
      - node_conv2d    : model.map_cnn[0]   输入 per-channel [C_in=3]
      - node_conv2d_1  : model.map_cnn[3]   输入 per-channel [C_in=16]
      - node_linear_10 : model.actor[6]     输入 per-channel [C_in=128]
      - node_MatMul_69 : query_projector 输出（≈ Q-proj 的输入）per-channel [64]

    EMA 公式: stat = decay * stat + (1-decay) * new_val
    update_node_data() 将统计量写回 node_data：
      - scale_z_x/y_x : absmax EMA → QDQ scale 在线更新
      - alpha, beta   : per-channel 1/σ, -μ/σ → LET 折叠 refit closure 使用
    """

    def __init__(self, model, node_data, include_nodes=None, ema_decay=0.95):
        self.node_data = node_data
        self.ema_decay = ema_decay
        self._stats    = {}   # node_name → {"z_max": float, "y_max": float}
        self._ch_mu    = {}   # node_name → Tensor[C_in]  EMA 均值
        self._ch_sigma = {}   # node_name → Tensor[C_in]  EMA 标准差
        self._handles  = []
        self._lock     = threading.Lock()

        if include_nodes is None:
            include_nodes = BEST_SUBSET
        self._register(model, include_nodes)

    # ── EMA 更新（标量 / 张量）───────────────────────────────────────────────

    def _ema(self, node_name, key, new_val):
        with self._lock:
            d = self._stats.setdefault(node_name, {})
            prev = d.get(key)
            d[key] = new_val if prev is None else (
                self.ema_decay * prev + (1 - self.ema_decay) * new_val)

    def _ema_ch(self, name_mu, name_sigma, x_flat, max_rows=2048):
        """x_flat: [N, C_in]。EMA 更新 per-channel 均值和标准差。
        为降低大 Conv 输入（~712K 行）的 hook 开销，最多采样 max_rows 行。"""
        if x_flat.shape[0] > max_rows:
            idx = torch.randint(0, x_flat.shape[0],
                                (max_rows,), device=x_flat.device)
            x_flat = x_flat[idx]
        mu    = x_flat.mean(dim=0)
        sigma = x_flat.std(dim=0).clamp_min(1e-8)
        with self._lock:
            prev_mu = self._ch_mu.get(name_mu)
            if prev_mu is None:
                self._ch_mu[name_mu]    = mu.clone()
                self._ch_sigma[name_mu] = sigma.clone()
            else:
                self._ch_mu[name_mu].mul_(self.ema_decay).add_(mu    * (1 - self.ema_decay))
                self._ch_sigma[name_mu].mul_(self.ema_decay).add_(sigma * (1 - self.ema_decay))

    # ── hook 工厂 ─────────────────────────────────────────────────────────────

    def _inp_hook(self, node_name, is_conv=False):
        """输入 absmax + per-channel μ/σ。is_conv=True 时 x:[B,C,H,W]。"""
        def fn(m, inp, out):
            x = inp[0].detach()
            if is_conv:
                xf = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # [N*H*W, C_in]
            else:
                xf = x.reshape(-1, x.shape[-1])                       # [N, C_in]
            self._ema(node_name, "z_max", float(x.abs().amax().item()))
            self._ema_ch(node_name, node_name, xf)
        return fn

    def _out_hook(self, node_name):
        """输出 absmax → y_max。"""
        def fn(m, inp, out):
            y = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()
            self._ema(node_name, "y_max", float(y.abs().amax().item()))
        return fn

    def _out_as_inp_hook(self, node_name):
        """前驱层输出 = 本层输入（MHA Q-proj）：absmax + per-channel。"""
        def fn(m, inp, out):
            y = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()
            yf = y.reshape(-1, y.shape[-1])
            self._ema(node_name, "z_max", float(y.abs().amax().item()))
            self._ema_ch(node_name, node_name, yf)
        return fn

    # ── 注册 hook ─────────────────────────────────────────────────────────────

    def _register(self, model, include_nodes):
        def add(hook_fn, module):
            self._handles.append(module.register_forward_hook(hook_fn))

        if "node_conv2d" in include_nodes and hasattr(model, "map_cnn"):
            add(self._inp_hook("node_conv2d",   is_conv=True), model.map_cnn[0])
            add(self._out_hook("node_conv2d"),                 model.map_cnn[0])

        if "node_conv2d_1" in include_nodes and hasattr(model, "map_cnn"):
            add(self._inp_hook("node_conv2d_1", is_conv=True), model.map_cnn[3])
            add(self._out_hook("node_conv2d_1"),               model.map_cnn[3])

        if "node_linear_10" in include_nodes and hasattr(model, "actor"):
            add(self._inp_hook("node_linear_10"), model.actor[6])
            add(self._out_hook("node_linear_10"), model.actor[6])

        if "node_MatMul_69" in include_nodes and hasattr(model, "query_projector"):
            add(self._out_as_inp_hook("node_MatMul_69"), model.query_projector)

    # ── 写回 node_data ────────────────────────────────────────────────────────

    def update_node_data(self):
        """将 EMA 统计量写回 node_data（in-place）。每轮 PPO update 后调用一次。"""
        with self._lock:
            # 1. QDQ scale（absmax EMA）
            for node_name, stats in self._stats.items():
                nd = self.node_data.get(node_name)
                if nd is None:
                    continue
                if "z_max" in stats and stats["z_max"] > 1e-8:
                    nd["scale_z_x"] = max(stats["z_max"] / 127.0, 1e-8)
                    nd["scale_z"]   = nd["scale_z_x"]
                if "y_max" in stats and stats["y_max"] > 1e-8:
                    nd["scale_y_x"] = max(stats["y_max"] / 127.0, 1e-8)
                    nd["scale_y"]   = nd["scale_y_x"]
            # 2. LET α, β（per-channel 1/σ, -μ/σ）
            for node_name in list(self._ch_mu.keys()):
                nd = self.node_data.get(node_name)
                if nd is None:
                    continue
                mu    = self._ch_mu[node_name]
                sigma = self._ch_sigma[node_name]
                nd["alpha"] = (1.0 / sigma).cpu().numpy().astype(np.float32)
                nd["beta"]  = (-mu / sigma).cpu().numpy().astype(np.float32)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ════════════════════════════════════════════════════════════════════════════
# 快速 LET 校准（仅用于生成合法 QDQ scale，不追求最优精度）
# ════════════════════════════════════════════════════════════════════════════

def quick_calibrate(model, n_calib=5, n_steps=30):
    """快速校准，~1-2 分钟，scale 合法即可（基准测试不测精度）。"""
    model.eval()
    MAX_S = 2048
    bufs = {k: [] for k in ["prop_emb","gps","qproj","topk",
                              "mlp_l1","mlp_l2","mlp_l3","mlp_l4",
                              "conv0_in","conv1_in","mha_q_in","mha_kv_in"]}
    # Conv 输入 per-channel 采集（正确 channel_dim=1）
    conv_ch_bufs: dict = {"conv0": [], "conv1": []}

    def collect(name):
        def hook(m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            cur = sum(t.shape[0] for t in bufs[name])
            if cur >= MAX_S: return
            bufs[name].append(x[:min(MAX_S-cur, x.shape[0])].cpu())
        return hook
    def collect_out(name, C_out):
        def hook(m, inp, out):
            x = out[0].detach() if isinstance(out, tuple) else out.detach()
            if x.ndim == 4: x = x.permute(0,2,3,1).reshape(-1, C_out)
            else: x = x.reshape(-1, C_out)
            cur = sum(t.shape[0] for t in bufs[name])
            if cur >= MAX_S: return
            bufs[name].append(x[:min(MAX_S-cur, x.shape[0])].cpu())
        return hook
    def collect_conv_ch(name):
        """Conv 输入 per-channel 采集: x [B,C,H,W] → reshape [B*H*W, C]"""
        def hook(m, inp, out):
            x  = inp[0].detach()  # [B, C_in, H, W]
            xf = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # [N, C_in]
            cur = sum(t.shape[0] for t in conv_ch_bufs[name])
            if cur >= MAX_S: return
            conv_ch_bufs[name].append(xf[:min(MAX_S-cur, xf.shape[0])].cpu())
        return hook

    hooks = [
        model.actor_proprio_embedding.register_forward_hook(collect("prop_emb")),
        model.global_pool_selector   .register_forward_hook(collect("gps")),
        model.query_projector        .register_forward_hook(collect("qproj")),
        model.topk_scorer            .register_forward_hook(collect("topk")),
        model.actor[0]               .register_forward_hook(collect("mlp_l1")),
        model.actor[2]               .register_forward_hook(collect("mlp_l2")),
        model.actor[4]               .register_forward_hook(collect("mlp_l3")),
        model.actor[6]               .register_forward_hook(collect("mlp_l4")),
        model.map_cnn[0]             .register_forward_hook(collect("conv0_in")),
        model.map_cnn[3]             .register_forward_hook(collect("conv1_in")),
        model.query_projector.register_forward_hook(collect_out("mha_q_in", 64)),
        model.map_cnn        .register_forward_hook(collect_out("mha_kv_in", 64)),
        model.map_cnn[0]     .register_forward_hook(collect_conv_ch("conv0")),
        model.map_cnn[3]     .register_forward_hook(collect_conv_ch("conv1")),
    ]
    print(f"  采集激活 ({n_calib} 批)...")
    with torch.no_grad():
        for _ in range(n_calib):
            obs = torch.randn(B, PA+MAP, device=DEVICE)
            model.act_inference({"policy_obs": obs,
                                 "critic_obs": torch.randn(B, PA+3+MAP, device=DEVICE)})
    for h in hooks: h.remove()
    act = {k: torch.cat(v, 0).to(DEVICE) for k, v in bufs.items() if v}

    # ORT 捕获 Gemm_139 输入（快速版：只跑 2 批）
    try:
        import onnx.helper as _oh
        from onnx import TensorProto as _tp
        _m2 = onnx.load(BASE_ONNX)
        _m2.graph.output.append(_oh.make_tensor_value_info("view_6", _tp.FLOAT, None))
        _sess2 = ort.InferenceSession(_m2.SerializeToString(),
                                      providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        _in2 = _sess2.get_inputs()[0].name
        _g139 = []
        for _ in range(2):
            _x2 = np.random.randn(B, PA+MAP).astype(np.float32)
            _all = _sess2.run(None, {_in2: _x2})
            _v6_idx = [o.name for o in _sess2.get_outputs()].index("view_6")
            _v6 = _all[_v6_idx]
            if _v6.ndim == 3: _v6 = _v6.squeeze(1)
            _g139.append(_v6)
        act["gemm139_in"] = torch.from_numpy(np.concatenate(_g139)[:MAX_S]).to(DEVICE)
        del _m2, _sess2, _g139
    except Exception as e:
        print(f"  [warn] view_6 ORT 失败: {e}，用 mha_kv_in 近似")
        act["gemm139_in"] = act.get("mha_kv_in", torch.randn(MAX_S, 64, device=DEVICE))

    import torch.nn as nn
    from quant.let import LETQuantizer
    _linear_q  = nn.Linear(64, 64,    bias=True).to(DEVICE)
    _linear_kv = nn.Linear(64, 2*64,  bias=True).to(DEVICE)
    with torch.no_grad():
        _linear_q.weight.copy_(model.mha.in_proj_weight[:64, :])
        _linear_kv.weight.copy_(model.mha.in_proj_weight[64:, :])
        if model.mha.in_proj_bias is not None:
            _linear_q.bias.copy_(model.mha.in_proj_bias[:64])
            _linear_kv.bias.copy_(model.mha.in_proj_bias[64:])

    LINEAR_MAP = {
        "node_linear":    ("prop_emb",    model.actor_proprio_embedding, PA,  "Gemm"),
        "node_MatMul_50": ("gps",         model.global_pool_selector,    64,  "MatMul"),
        "node_linear_2":  ("qproj",       model.query_projector,         128, "Gemm"),
        "node_MatMul_59": ("topk",        model.topk_scorer,             128, "MatMul"),
        "node_MatMul_69": ("mha_q_in",    _linear_q,                     64,  "MatMul"),
        "node_MatMul_71": ("mha_kv_in",   _linear_kv,                    64,  "MatMul"),
        "node_Gemm_139":  ("gemm139_in",  model.mha.out_proj,            64,  "Gemm"),
        "node_linear_7":  ("mlp_l1",      model.actor[0],                224, "Gemm"),
        "node_linear_8":  ("mlp_l2",      model.actor[2],                512, "Gemm"),
        "node_linear_9":  ("mlp_l3",      model.actor[4],                256, "Gemm"),
        "node_linear_10": ("mlp_l4",      model.actor[6],                128, "Gemm"),
    }

    node_data = {}
    print(f"  LET 校准 ({n_steps} 步/层，仅 BEST_SUBSET)...")
    for nname, (akey, layer, C_in, op_type) in LINEAR_MAP.items():
        if nname not in BEST_SUBSET: continue
        x_all  = act.get(akey)
        if x_all is None: continue
        x_flat = x_all.reshape(-1, C_in)[:min(B*4, x_all.reshape(-1,C_in).shape[0])].to(DEVICE)
        let = LETQuantizer(n_channels=C_in).to(DEVICE)
        let.initialize_from_activation(x_flat)
        opt = torch.optim.Adam(let.parameters(), lr=8e-3)
        for _ in range(n_steps):
            idx = torch.randperm(x_flat.shape[0], device=DEVICE)[:min(2048, x_flat.shape[0])]
            _, L = let.block_reconstruction_loss(x_flat[idx], layer)
            opt.zero_grad(); L.backward(); opt.step()
        scale_z_x = max(float(x_flat.abs().amax().item()) / 127.0, 1e-8)
        # s_y：在 LET 折叠后的权重 W'=W/α 上计算（可靠工具 mode，无 Mul+Add）
        alpha_np = let.alpha.detach().cpu().numpy()  # [C_in]
        beta_np  = let.beta.detach().cpu().numpy()   # [C_in]
        W_prime_np = layer.weight.detach().cpu().numpy() / alpha_np.reshape(1, -1)
        b_prime_np = (layer.bias.detach().cpu().numpy()
                      - (W_prime_np * beta_np.reshape(1, -1)).sum(axis=1))
        xq_np = np.clip(np.round(x_flat.cpu().numpy() / scale_z_x), -127, 127) * scale_z_x
        y_folded_np = xq_np @ W_prime_np.T + b_prime_np
        scale_y_x = max(float(np.abs(y_folded_np).max()) / 127.0, 1e-8)
        node_data[nname] = {
            "let": let.cpu(), "alpha": alpha_np, "beta": beta_np,
            "scale_z": scale_z_x, "scale_z_x": scale_z_x,
            "scale_y": scale_y_x, "scale_y_x": scale_y_x,
            "layer_op": op_type,
        }
        del opt; gc.collect()

    with torch.no_grad():
        ti = torch.randn(min(B,256), 3, 21, 33, device=DEVICE)
        c0 = model.map_cnn[0](ti)
        c1 = model.map_cnn[3](c0)
    def mm_sc(t): return max(float(t.abs().max().item())/127.0, 1e-8)

    # Conv LET 初始化：从 per-channel 激活统计计算 α=1/σ, β=-μ/σ
    def conv_let_init(ch_buf_key, layer_conv, layer_output_tensor):
        """返回 (alpha, beta, s_z, s_y)。采集数据不足时 alpha/beta=None。"""
        s_y_fallback = mm_sc(layer_output_tensor)
        s_z_fallback = mm_sc(act.get("conv0_in" if ch_buf_key=="conv0" else "conv1_in",
                                     layer_output_tensor))
        if not conv_ch_bufs.get(ch_buf_key):
            return None, None, s_z_fallback, s_y_fallback
        xc = torch.cat(conv_ch_bufs[ch_buf_key], 0).to(DEVICE)  # [N, C_in]
        mu    = xc.mean(dim=0).cpu().numpy().astype(np.float32)
        sigma = xc.std(dim=0).clamp_min(1e-8).cpu().numpy().astype(np.float32)
        alpha_c = (1.0 / sigma).astype(np.float32)
        beta_c  = (-mu / sigma).astype(np.float32)
        s_z = max(float(np.abs(xc.cpu().numpy()).max()) / 127.0, 1e-8)
        s_y = s_y_fallback  # 用原始输出作 s_y（折叠输出范围更小，保守估计）
        return alpha_c, beta_c, s_z, s_y

    a0, b0, sz0, sy0 = conv_let_init("conv0", model.map_cnn[0], c0)
    a1, b1, sz1, sy1 = conv_let_init("conv1", model.map_cnn[3], c1)

    node_data["node_conv2d"]   = {
        "let": None, "alpha": a0, "beta": b0,
        "scale_z": sz0, "scale_z_x": sz0,
        "scale_y": sy0, "scale_y_x": sy0, "layer_op": "Conv"}
    node_data["node_conv2d_1"] = {
        "let": None, "alpha": a1, "beta": b1,
        "scale_z": sz1, "scale_z_x": sz1,
        "scale_y": sy1, "scale_y_x": sy1, "layer_op": "Conv"}

    gc.collect(); torch.cuda.empty_cache()
    return node_data


# ════════════════════════════════════════════════════════════════════════════
# ONNX 构建（原样复制 build_let_onnx，不折叠权重）
# ════════════════════════════════════════════════════════════════════════════

def build_let_onnx(include_nodes, node_data, out_path):
    graph = gs.import_onnx(onnx.load(BASE_ONNX))
    new_nodes = []; inserted = 0
    for node in graph.nodes:
        is_target = (node.op in {"Conv","Gemm"} or
                     (node.op=="MatMul" and node.name in INT8_MATMUL_OK))
        if not is_target or node.name in EXCLUDE_ALWAYS: continue
        if node.name not in include_nodes: continue
        nd = node_data.get(node.name)
        if nd is None: continue
        if not isinstance(node.inputs[0], gs.Variable): continue
        uid = node.name.replace("/","_")
        inp = node.inputs[0]
        s_z   = float(nd.get("scale_z_x", nd["scale_z"]))
        qs_i  = gs.Constant(f"qs_i_{uid}",  np.float32(s_z).reshape(1))
        qzp_i = gs.Constant(f"qzp_i_{uid}", np.array([0], dtype=np.int8))
        dqs_i = gs.Constant(f"dqs_i_{uid}", np.float32(s_z).reshape(1))
        dqzp_i= gs.Constant(f"dqzp_i_{uid}",np.array([0], dtype=np.int8))
        qi_v  = gs.Variable(f"qi_{uid}",  dtype=np.int8)
        dqi_v = gs.Variable(f"dqi_{uid}", dtype=np.float32)
        new_nodes += [
            gs.Node("QuantizeLinear",   inputs=[inp, qs_i, qzp_i],   outputs=[qi_v]),
            gs.Node("DequantizeLinear", inputs=[qi_v, dqs_i, dqzp_i], outputs=[dqi_v]),
        ]
        node.inputs[0] = dqi_v
        orig_out  = node.outputs[0]
        orig_name = orig_out.name
        orig_out.name = f"_raw_{uid}"
        s_y   = float(nd.get("scale_y_x", nd["scale_y"]))
        qs_o  = gs.Constant(f"qs_o_{uid}",   np.float32(s_y).reshape(1))
        qzp_o = gs.Constant(f"qzp_o_{uid}",  np.array([0], dtype=np.int8))
        dqs_o = gs.Constant(f"dqs_o_{uid}",  np.float32(s_y).reshape(1))
        dqzp_o= gs.Constant(f"dqzp_o_{uid}", np.array([0], dtype=np.int8))
        qo_v  = gs.Variable(f"qo_{uid}",  dtype=np.int8)
        dqo_v = gs.Variable(orig_name,    dtype=np.float32)
        new_nodes += [
            gs.Node("QuantizeLinear",   inputs=[orig_out, qs_o, qzp_o], outputs=[qo_v]),
            gs.Node("DequantizeLinear", inputs=[qo_v, dqs_o, dqzp_o],  outputs=[dqo_v]),
        ]
        for other in graph.nodes:
            if other is node: continue
            other.inputs = [dqo_v if v is orig_out else v for v in other.inputs]
        graph.outputs = [dqo_v if v is orig_out else v for v in graph.outputs]

        # ── LET 权重折叠（可靠工具 mode：无 Mul+Add，折叠 W'=W/α, b'=b-W'β）──
        # 数学：(W/α)·Q(x)+b' ≈ W·x+b（α≈1 近似成立）
        # 好处：W' 列分布更均匀，INT8 权重量化误差更小
        alpha_nd = nd.get("alpha")
        beta_nd  = nd.get("beta")
        if (alpha_nd is not None and beta_nd is not None
                and isinstance(node.inputs[1], gs.Constant)):
            alpha_np = np.asarray(alpha_nd, dtype=np.float32)
            beta_np  = np.asarray(beta_nd,  dtype=np.float32)
            op = nd.get("layer_op", node.op)
            if op in ("Gemm", "MatMul"):
                W = node.inputs[1].values.copy()
                if W.shape[1] == len(alpha_np):           # [C_out, C_in]
                    W_p = (W / alpha_np[np.newaxis, :]).astype(np.float32)
                    node.inputs[1].values = W_p
                    if (op == "Gemm" and len(node.inputs) > 2
                            and isinstance(node.inputs[2], gs.Constant)):
                        b = node.inputs[2].values.copy()
                        b_p = (b - (W_p * beta_np[np.newaxis, :]).sum(axis=1)).astype(np.float32)
                        node.inputs[2].values = b_p
                elif W.shape[0] == len(alpha_np):         # [C_in, C_out] 转置
                    W_p = (W / alpha_np[:, np.newaxis]).astype(np.float32)
                    node.inputs[1].values = W_p
            elif op == "Conv":
                W = node.inputs[1].values.copy()          # [C_out, C_in, kH, kW]
                if W.shape[1] == len(alpha_np):
                    W_p = (W / alpha_np.reshape(1, -1, 1, 1)).astype(np.float32)
                    node.inputs[1].values = W_p
                    if (len(node.inputs) > 2
                            and isinstance(node.inputs[2], gs.Constant)):
                        b = node.inputs[2].values.copy()
                        b_p = (b - (W_p * beta_np.reshape(1, -1, 1, 1)).sum(axis=(1,2,3))).astype(np.float32)
                        node.inputs[2].values = b_p

        inserted += 1
    graph.nodes.extend(new_nodes)
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_path)
    print(f"  LET-INT8 GS: {inserted} 层 → {out_path}  ({os.path.getsize(out_path)//1024} KB)")
    return inserted


# ════════════════════════════════════════════════════════════════════════════
# TRT 构建（加 REFIT flag）
# ════════════════════════════════════════════════════════════════════════════

def build_refit_engine(onnx_path, eng_path):
    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    par = trt.OnnxParser(net, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not par.parse(f.read()):
            for i in range(par.num_errors): print(f"  [parse] {par.get_error(i)}")
            return False
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.REFIT)  # ← 关键：允许 refit
    cfg.avg_timing_iterations = 8
    tc_data = open(TIMING_CACHE,"rb").read() if os.path.exists(TIMING_CACHE) else b""
    tc = cfg.create_timing_cache(tc_data)
    cfg.set_timing_cache(tc, ignore_mismatch=True)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        print("  BUILD FAILED"); return False
    with open(eng_path, "wb") as f: f.write(bytes(ser))
    with open(TIMING_CACHE, "wb") as f: f.write(memoryview(cfg.get_timing_cache().serialize()))
    print(f"  Refit engine: {os.path.getsize(eng_path)//1024}KB  {time.time()-t0:.1f}s")
    del builder, net, cfg; gc.collect()
    return True


# ════════════════════════════════════════════════════════════════════════════
# Refit 权重名发现与映射
# ════════════════════════════════════════════════════════════════════════════

def discover_and_map(engine, model, node_data=None):
    """
    TRT 10.x: get_all_weights() 返回字符串列表。
    模型权重用 'm.' 前缀（ONNX export 时变量名为 m）。

    node_data（可选）：若传入，QDQ scale 名称也加入 refit_map，
    闭包读取 node_data[node_name]["scale_z_x"/"scale_y_x"] 的当前值。
    ActivationTracker.update_node_data() 会更新这些值，从而实现在线 scale refit。

    返回 refit_map: list of (trt_name, value_fn)，value_fn() → np.ndarray
    """
    refitter = trt.Refitter(engine, TRT_LOGGER)
    try:
        all_names = list(refitter.get_all_weights())
    except Exception as e:
        print(f"  [warn] get_all_weights() 失败: {e}")
        return []

    # 模型参数字典
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict['m.' + name] = param
    for name, buf in model.named_buffers():
        param_dict['m.' + name] = buf

    # QDQ scale 查找表：trt_name → (nd_dict, scale_key)
    # uid = node_name.replace("/","_") 与 build_let_onnx 保持一致
    qdq_lookup = {}
    if node_data:
        for node_name, nd in node_data.items():
            uid = node_name.replace("/", "_")
            qdq_lookup[f"qs_i_{uid}"]  = (nd, "scale_z_x")
            qdq_lookup[f"dqs_i_{uid}"] = (nd, "scale_z_x")
            qdq_lookup[f"qs_o_{uid}"]  = (nd, "scale_y_x")
            qdq_lookup[f"dqs_o_{uid}"] = (nd, "scale_y_x")

    print(f"\n  Refittable weights ({len(all_names)}):")
    refit_map = []
    n_model = n_scale = n_let = n_fixed = 0

    for trt_name in sorted(all_names):
        if trt_name.startswith('m.') and trt_name in param_dict:
            p = param_dict[trt_name]
            # ── LET 折叠 closure（若该层有 alpha/beta）────────────────────────
            if node_data and trt_name in TRT_TO_LET_NODE:
                node_name, kind = TRT_TO_LET_NODE[trt_name]
                nd_ref = node_data.get(node_name, {})
                alpha = nd_ref.get("alpha")
                if alpha is not None:
                    op = nd_ref.get("layer_op", "Gemm")
                    if kind == "weight":
                        if op == "Conv":
                            fn = (lambda p=p, nd=nd_ref:
                                (p.detach().float().cpu().numpy()
                                 / nd["alpha"].reshape(1, -1, 1, 1))
                                if nd.get("alpha") is not None
                                else p.detach().float().cpu().numpy())
                        else:  # Gemm / MatMul
                            fn = (lambda p=p, nd=nd_ref:
                                (p.detach().float().cpu().numpy()
                                 / nd["alpha"].reshape(1, -1))
                                if nd.get("alpha") is not None
                                else p.detach().float().cpu().numpy())
                    else:  # bias
                        w_trt = trt_name.replace(".bias", ".weight")
                        p_w   = param_dict.get(w_trt, p)
                        if op == "Conv":
                            fn = (lambda p_w=p_w, p_b=p, nd=nd_ref:
                                (p_b.detach().float().cpu().numpy()
                                 - (p_w.detach().float().cpu().numpy()
                                    / nd["alpha"].reshape(1, -1, 1, 1)
                                    * nd["beta"].reshape(1, -1, 1, 1)).sum(axis=(1, 2, 3)))
                                if nd.get("alpha") is not None
                                else p_b.detach().float().cpu().numpy())
                        else:  # Gemm
                            fn = (lambda p_w=p_w, p_b=p, nd=nd_ref:
                                (p_b.detach().float().cpu().numpy()
                                 - (p_w.detach().float().cpu().numpy()
                                    / nd["alpha"].reshape(1, -1)
                                    * nd["beta"].reshape(1, -1)).sum(axis=1))
                                if nd.get("alpha") is not None
                                else p_b.detach().float().cpu().numpy())
                    refit_map.append((trt_name, fn))
                    print(f"    [let]   {trt_name}  {tuple(p.shape)}  ({kind})")
                    n_let += 1
                    continue
            # ── 普通参数 closure ──────────────────────────────────────────────
            fn = (lambda p=p: p.detach().float().cpu().numpy())
            refit_map.append((trt_name, fn))
            print(f"    [model] {trt_name}  {tuple(p.shape)}")
            n_model += 1
        elif node_data and trt_name in qdq_lookup:
            nd_ref, sk = qdq_lookup[trt_name]
            fn = (lambda nd=nd_ref, k=sk: np.float32(nd[k]).reshape(1))
            refit_map.append((trt_name, fn))
            print(f"    [scale] {trt_name}  ({sk}={nd_ref[sk]:.5f})")
            n_scale += 1
        else:
            print(f"    [fixed] {trt_name}")
            n_fixed += 1

    print(f"\n  Mapped {n_model} model + {n_let} LET-folded + {n_scale} QDQ scales"
          f"  ({n_fixed} fixed)")
    return refit_map


def do_refit(engine, refit_map):
    """用当前 model 权重 refit TRT 引擎，返回耗时(ms)。"""
    if not refit_map:
        return 0.0
    t0 = time.time()
    refitter = trt.Refitter(engine, TRT_LOGGER)
    for trt_name, fn in refit_map:
        val = fn()
        if val is None: continue
        try:
            refitter.set_named_weights(trt_name, np.ascontiguousarray(val.astype(np.float32)))
        except Exception:
            pass
    refitter.refit_cuda_engine()
    return (time.time() - t0) * 1000


# ════════════════════════════════════════════════════════════════════════════
# 双缓冲引擎：一个用于 rollout，另一个在后台线程被 refit
# ════════════════════════════════════════════════════════════════════════════

class DoubleBufferedEngine:
    """
    两个 TRT 引擎交替使用：
      - active  引擎：当前 rollout
      - inactive 引擎：后台线程正在 refit

    核心时序（refit ~62ms < rollout ~106ms）：
      iter t:   rollout(A) → update → snapshot_weights → start_refit(B)
      iter t+1: rollout(A)   ← 此时 B 正在后台 refit，62ms 内完成
                wait_swap()  ← ~0ms（refit 已完成）
      iter t+2: rollout(B) → update → start_refit(A)
      ...

    refit 开销完全隐藏在 rollout 中，净加速接近理论上限（~1.54×）。
    代价：1 iteration 权重延迟（与 K=2 相当，但开销为零）。
    """

    def __init__(self, engine_a, engine_b):
        self._engines = [engine_a, engine_b]
        self._active  = 0   # 当前 rollout 使用的引擎下标
        self._thread  = None

        # 为每个引擎预建 ctx / stream / output buffer
        self._ctxs    = []
        self._streams = []
        self._obs_bufs = []
        for eng in self._engines:
            ctx, stream, ob = make_trt_ctx(eng)
            self._ctxs.append(ctx)
            self._streams.append(stream)
            self._obs_bufs.append(ob)

    # ── active 引擎属性 ────────────────────────────────────────────────────

    @property
    def ctx(self):    return self._ctxs[self._active]
    @property
    def stream(self): return self._streams[self._active]
    @property
    def ob(self):     return self._obs_bufs[self._active]

    # ── 异步 refit ─────────────────────────────────────────────────────────

    def start_async_refit(self, refit_map):
        """
        立即对 refit_map 求值（快照模型权重和 scale），
        然后在后台线程对 inactive 引擎做 refit。

        快照在主线程同步完成，避免与下一轮 optimizer.step() 产生数据竞争。
        """
        inactive = 1 - self._active
        eng = self._engines[inactive]

        # 同步快照所有权重和 scale（主线程，~2ms）
        snapshots = []
        for trt_name, fn in refit_map:
            try:
                val = fn()
                if val is not None:
                    snapshots.append((trt_name,
                                      np.ascontiguousarray(val.astype(np.float32))))
            except Exception:
                pass

        def _worker():
            refitter = trt.Refitter(eng, TRT_LOGGER)
            for trt_name, arr in snapshots:
                try:
                    refitter.set_named_weights(trt_name, arr)
                except Exception:
                    pass
            refitter.refit_cuda_engine()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def wait_and_swap(self):
        """等待后台 refit 完成，然后将 inactive 引擎切换为 active。"""
        t0 = time.time()
        if self._thread is not None:
            self._thread.join(timeout=300)
            self._thread = None
        self._active = 1 - self._active
        return (time.time() - t0) * 1000   # 返回等待耗时 ms

    def __del__(self):
        if self._thread is not None:
            try: self._thread.join(timeout=5)
            except Exception: pass
        for ctx in self._ctxs:
            try: del ctx
            except Exception: pass


# ════════════════════════════════════════════════════════════════════════════
# TRT 单步推理（rollout 中调用）
# ════════════════════════════════════════════════════════════════════════════

def make_trt_ctx(engine):
    """Phase B 用：ob 地址预绑定，obs 地址每步通过 set_tensor_address 传入。"""
    ctx    = engine.create_execution_context()
    stream = torch.cuda.Stream()
    ob = torch.empty(B, NA, device=DEVICE, dtype=torch.float32).contiguous()
    ctx.set_tensor_address("action", int(ob.data_ptr()))
    return ctx, stream, ob


def trt_step(ctx, stream, ob, obs_tensor):
    """每步更新 obs 地址后执行推理（Phase B）。"""
    ctx.set_tensor_address("obs", int(obs_tensor.data_ptr()))
    ctx.execute_async_v3(stream.cuda_stream)


# ════════════════════════════════════════════════════════════════════════════
# 模拟 PPO 更新（dummy loss，让权重实际改变）
# ════════════════════════════════════════════════════════════════════════════

def ppo_update_step(model, optimizer):
    """每次调用改变模型权重，模拟一轮 PPO 更新（4 个 minibatch）。"""
    model.train()
    for _ in range(N_MINIBATCHES):
        obs_dict = {"policy_obs": torch.randn(MINIBATCH_SZ, PA+MAP, device=DEVICE),
                    "critic_obs": torch.randn(MINIBATCH_SZ, PA+3+MAP, device=DEVICE)}
        # 走 actor 路径（update_distribution → actor MLP），通过 .mean 做 backprop
        actor_obs = model.actor_obs_normalizer(model.get_actor_obs(obs_dict))
        model.update_distribution(actor_obs)
        loss = model.distribution.mean.pow(2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()


# ════════════════════════════════════════════════════════════════════════════
# 基准测试主循环
# ════════════════════════════════════════════════════════════════════════════

def bench_fp32(model, optimizer, n_iters, n_steps=N_STEPS):
    """FP32 基线：rollout 用 PyTorch FP32 前向。obs 预生成，排除在计时外。"""
    print(f"\n{'─'*65}")
    print(f"  A. FP32 基线  (n_steps={n_steps}, N_ENVS={B})")
    print(f"{'─'*65}")
    times_roll, times_upd = [], []

    obs_pool = [{"policy_obs": torch.randn(B, PA+MAP,   device=DEVICE),
                 "critic_obs": torch.randn(B, PA+3+MAP, device=DEVICE)}
                for _ in range(n_steps)]
    torch.cuda.synchronize()

    for it in range(n_iters + N_WARMUP):
        model.eval()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for i in range(n_steps):
                model.act_inference(obs_pool[i])
        torch.cuda.synchronize()
        t_roll = (time.time() - t0) * 1000

        torch.cuda.synchronize()
        t0 = time.time()
        ppo_update_step(model, optimizer)
        torch.cuda.synchronize()
        t_upd = (time.time() - t0) * 1000

        if it < N_WARMUP:
            print(f"  [warmup {it+1}] rollout={t_roll:.0f}ms  update={t_upd:.0f}ms")
            continue

        times_roll.append(t_roll); times_upd.append(t_upd)
        print(f"  iter {it-N_WARMUP+1:02d}:"
              f"  rollout={t_roll:6.0f}ms"
              f"  update={t_upd:5.0f}ms"
              f"  total={t_roll+t_upd:6.0f}ms")

    r = np.mean(times_roll); u = np.mean(times_upd)
    print(f"\n  均值: rollout={r:.0f}ms  update={u:.0f}ms  total={r+u:.0f}ms")
    return r, u


def bench_refit(model, optimizer, engine, refit_map, n_iters,
                n_steps=N_STEPS, refit_every_k=1, tracker=None):
    """INT8+Refit：每 K 次 rollout 才 refit 一次，将固定开销摊薄 K 倍。
    tracker（可选）：ActivationTracker，若传入则每轮 PPO update 后更新 node_data
                      中的 α, β 和 QDQ scale，供 LET 折叠 closure 使用。
    """
    print(f"\n{'─'*65}")
    print(f"  B. INT8+Refit  (n_steps={n_steps}, K={refit_every_k},"
          f" LET={'on' if tracker else 'static'})")
    print(f"{'─'*65}")

    obs_pool = [torch.randn(B, PA+MAP, device=DEVICE).contiguous()
                for _ in range(n_steps)]
    torch.cuda.synchronize()

    ctx, stream, ob = make_trt_ctx(engine)
    times_refit, times_roll, times_upd = [], [], []

    for it in range(n_iters + N_WARMUP):
        # ── refit（每 K 轮触发一次）────────────────────────────────────────
        if it % refit_every_k == 0:
            t_refit = do_refit(engine, refit_map)
        else:
            t_refit = 0.0

        # ── rollout (INT8 TRT) ─────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.stream(stream):
            for i in range(n_steps):
                trt_step(ctx, stream, ob, obs_pool[i])
        stream.synchronize()
        t_roll = (time.time() - t0) * 1000

        # ── PPO update ─────────────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.time()
        ppo_update_step(model, optimizer)
        torch.cuda.synchronize()
        t_upd = (time.time() - t0) * 1000

        # 在线 LET α/β 更新（若有 tracker）
        if tracker is not None:
            tracker.update_node_data()

        if it < N_WARMUP:
            print(f"  [warmup {it+1}] refit={t_refit:.0f}ms  "
                  f"rollout={t_roll:.0f}ms  update={t_upd:.0f}ms")
            continue

        times_refit.append(t_refit)
        times_roll.append(t_roll)
        times_upd.append(t_upd)
        tag = " ←refit" if t_refit > 0 else ""
        print(f"  iter {it-N_WARMUP+1:02d}:"
              f"  refit={t_refit:5.0f}ms"
              f"  rollout={t_roll:6.0f}ms"
              f"  update={t_upd:5.0f}ms"
              f"  total={t_refit+t_roll+t_upd:6.0f}ms{tag}")

    del ctx; gc.collect()
    rf = np.mean(times_refit)
    r  = np.mean(times_roll)
    u  = np.mean(times_upd)
    n_refit = sum(1 for x in times_refit if x > 0)
    print(f"\n  均值(摊薄): refit={rf:.1f}ms  rollout={r:.0f}ms  update={u:.0f}ms  "
          f"total={rf+r+u:.0f}ms  ({n_refit}/{n_iters} iters refitted)")
    return rf, r, u


# ════════════════════════════════════════════════════════════════════════════
# C. 双缓冲异步 Refit + 在线 scale 更新
# ════════════════════════════════════════════════════════════════════════════

def bench_async_refit(model, optimizer, engine_a, engine_b, refit_map,
                      node_data, n_iters, n_steps=N_STEPS, refit_every_k=1):
    """
    双缓冲异步 Refit 基准。

    时序（K=1）：
      rollout(active)            ← 用当前 active engine
      ppo_update (hook 采集统计)
      tracker.update_node_data() ← EMA scale 写入 node_data
      wait_and_swap()            ← 等待上轮 refit，swap active↔inactive
      start_async_refit()        ← 快照权重/scale，启动后台 refit

    refit_every_k=K：每 K 轮触发一次 swap+refit，可用窗口 = K×(r+u)。
      K=1: 窗口=r+u，refit 可能溢出 → wait>0
      K=4: 窗口=4×(r+u)>>refit 耗时 → wait≈0，开销完全分摊
    tracker.update_node_data() 每轮仍执行（EMA 持续积累）。
    """
    print(f"\n{'─'*65}")
    print(f"  C. INT8+Async Refit  双缓冲 + 在线 scale"
          f" (n_steps={n_steps}, K={refit_every_k})")
    print(f"{'─'*65}")

    tracker = ActivationTracker(model, node_data, BEST_SUBSET)
    db      = DoubleBufferedEngine(engine_a, engine_b)

    obs_pool = [torch.randn(B, PA+MAP, device=DEVICE).contiguous()
                for _ in range(n_steps)]
    torch.cuda.synchronize()

    times_roll, times_upd, times_wait = [], [], []

    for it in range(n_iters + N_WARMUP):
        # ── rollout on active engine ──────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.stream(db.stream):
            for i in range(n_steps):
                trt_step(db.ctx, db.stream, db.ob, obs_pool[i])
        db.stream.synchronize()
        t_roll = (time.time() - t0) * 1000

        # ── PPO update（hook 在前向中采集激活统计）───────────────────────
        torch.cuda.synchronize()
        t0 = time.time()
        ppo_update_step(model, optimizer)
        torch.cuda.synchronize()
        t_upd = (time.time() - t0) * 1000

        # 将 EMA 统计写回 node_data（供 refit 闭包读取）
        tracker.update_node_data()

        # ── swap + refit（每 K 轮触发一次）────────────────────────────────
        if (it + 1) % refit_every_k == 0:
            t_wait = db.wait_and_swap()      # 等待上轮 refit，swap active↔inactive
            db.start_async_refit(refit_map)  # 快照 + 启动后台线程
        else:
            t_wait = 0.0

        if it < N_WARMUP:
            print(f"  [warmup {it+1}] rollout={t_roll:.0f}ms  "
                  f"update={t_upd:.0f}ms  wait={t_wait:.0f}ms")
            continue

        times_roll.append(t_roll)
        times_upd.append(t_upd)
        times_wait.append(t_wait)
        print(f"  iter {it-N_WARMUP+1:02d}:"
              f"  rollout={t_roll:5.0f}ms"
              f"  update={t_upd:5.0f}ms"
              f"  wait={t_wait:4.0f}ms"
              f"  total={t_roll+t_upd+t_wait:6.0f}ms")

    # 收尾：等待最后一轮 refit（若有）
    if db._thread is not None:
        db.wait_and_swap()
    tracker.remove_hooks()

    r = np.mean(times_roll)
    u = np.mean(times_upd)
    w = np.mean(times_wait)
    print(f"\n  均值: rollout={r:.0f}ms  update={u:.0f}ms  "
          f"wait={w:.1f}ms  total={r+u+w:.0f}ms")
    return r, u, w


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print(f"  TRT REFIT 训练循环基准  GPU: {torch.cuda.get_device_name(0)}")
    print("="*65)

    # ── 构建模型 ──────────────────────────────────────────────────────────
    print("\n[0] 构建模型...")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # ── 校准 ─────────────────────────────────────────────────────────────
    print("\n[0] 快速 LET 校准...")
    node_data = quick_calibrate(model)

    # ── 构建 INT8 ONNX + TRT (REFIT, LET 折叠权重) ───────────────────────
    print("\n[1] 构建 LET-INT8 ONNX（权重折叠：W'=W/α, b'=b-W'β）...")
    build_let_onnx(BEST_SUBSET, node_data, REFIT_ONNX)

    print("\n[1] 构建 TRT 引擎 (REFIT flag)...")
    # 每次重建 ONNX 后必须重建引擎（折叠权重已更新）
    for stale in [REFIT_ENG,
                  os.path.join(WORKDIR, "engines", "_refit_int8_b.trt")]:
        if os.path.exists(stale):
            os.remove(stale)
            print(f"  删除旧引擎: {os.path.basename(stale)}")
    if not build_refit_engine(REFIT_ONNX, REFIT_ENG):
        print("引擎构建失败，退出"); return

    # ── 发现 refit 权重名映射（LET 模式：传入 node_data）──────────────────
    print("\n[2] 发现 refit 权重名映射（LET 折叠 closure）...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(open(REFIT_ENG, "rb").read())
    # 用 node_data 发现映射，打印一次（后续各 bench 自行创建 model 对应的 map）
    _m_tmp = build_model()
    discover_and_map(engine, _m_tmp, node_data)
    del _m_tmp; gc.collect()

    # ── FP32 基线（只跑一次）─────────────────────────────────────────────
    m_fp  = build_model()
    opt_fp = torch.optim.Adam(m_fp.parameters(), lr=3e-4)
    r_fp, u_fp = bench_fp32(m_fp, opt_fp, N_ITERS)
    del m_fp, opt_fp; gc.collect()
    fp_total = r_fp + u_fp

    # ── K sweep：K=1,2,4  (LET 静态折叠，α/β 固定于校准值) ──────────────
    K_LIST = [1, 2, 4]
    results_k = {}

    for k in K_LIST:
        m   = build_model()
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        # 传 node_data → 使用 LET 折叠 closure（W_new/α_calib）
        rm  = discover_and_map(engine, m, node_data)
        rf, r, u = bench_refit(m, opt, engine, rm, N_ITERS, refit_every_k=k)
        del m, opt; gc.collect()
        results_k[k] = (rf, r, u)

    # ── K=4 + 在线 LET（α/β 随 PPO 前向 EMA 更新）──────────────────────
    print(f"\n{'─'*65}")
    print(f"  B2. K=4 + 在线 LET (ActivationTracker online α/β update)")
    print(f"{'─'*65}")
    m_let  = build_model()
    opt_let = torch.optim.Adam(m_let.parameters(), lr=3e-4)
    rm_let  = discover_and_map(engine, m_let, node_data)
    trk_let = ActivationTracker(m_let, node_data, BEST_SUBSET)
    rf_ol, r_ol, u_ol = bench_refit(
        m_let, opt_let, engine, rm_let, N_ITERS,
        refit_every_k=4, tracker=trk_let)
    trk_let.remove_hooks()
    del m_let, opt_let; gc.collect()
    total_ol = rf_ol + r_ol + u_ol

    # ── K sweep 汇总 ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  K sweep 汇总  (N_STEPS={N_STEPS}, FP32={fp_total:.0f}ms/iter, LET折叠)")
    print(f"  {'方案':>22}  {'refit(amort)':>13}  {'rollout':>8}  {'update':>7}"
          f"  {'total':>7}  {'speedup':>8}  {'roll×':>6}")
    print("="*65)
    for k, (rf, r, u) in results_k.items():
        total   = rf + r + u
        speedup = fp_total / total
        print(f"  {'LET K='+str(k):>22}  {rf:>13.1f}ms  {r:>8.0f}ms  {u:>7.0f}ms"
              f"  {total:>7.0f}ms  {speedup:>7.3f}×  {r_fp/r:>5.2f}×")
    print(f"  {'K=4 + online α/β':>22}  {rf_ol:>13.1f}ms  {r_ol:>8.0f}ms  {u_ol:>7.0f}ms"
          f"  {total_ol:>7.0f}ms  {fp_total/total_ol:>7.3f}×  {r_fp/r_ol:>5.2f}×")
    print("="*65)

    r_int8_base = results_k[1][1]
    u_int8_base = results_k[1][2]
    theoretical = fp_total / (r_int8_base + u_int8_base)
    print(f"  理论上限(refit=0ms):  {theoretical:.3f}×"
          f"  (INT8 r+u={r_int8_base+u_int8_base:.0f}ms)")

    # ── C. 双缓冲异步 Refit（K=1 参考 + K=4 最优）───────────────────────
    print("\n[3] 复制引擎 B（双缓冲）...")
    import shutil
    REFIT_ENG_B = os.path.join(WORKDIR, "engines", "_refit_int8_b.trt")
    shutil.copy(REFIT_ENG, REFIT_ENG_B)
    print(f"  engine_b: {os.path.getsize(REFIT_ENG_B)//1024}KB")

    results_async = {}
    for ak in [1, 4]:
        print(f"\n[4] 双缓冲异步 Refit K={ak}...")
        engine_b = runtime.deserialize_cuda_engine(open(REFIT_ENG_B, "rb").read())
        m_async   = build_model()
        opt_async = torch.optim.Adam(m_async.parameters(), lr=3e-4)
        rm_async  = discover_and_map(engine, m_async, node_data)
        r_a, u_a, w_a = bench_async_refit(
            m_async, opt_async, engine, engine_b, rm_async, node_data, N_ITERS,
            refit_every_k=ak)
        results_async[ak] = (r_a, u_a, w_a)
        del m_async, opt_async, engine_b; gc.collect()

    # ── 完整对比 ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  完整对比  (FP32={fp_total:.0f}ms/iter)")
    print(f"  {'方案':<32}  {'total':>7}  {'speedup':>8}  {'roll×':>6}")
    print("─"*65)
    print(f"  {'FP32 基线':<32}  {fp_total:>7.0f}ms  {'1.000×':>8}")
    for k, (rf, r, u) in results_k.items():
        t = rf + r + u
        print(f"  {f'LET K={k}':<32}  {t:>7.0f}ms  {fp_total/t:>7.3f}×  {r_fp/r:>5.2f}×")
    print(f"  {'LET K=4 + online α/β':<32}  {total_ol:>7.0f}ms  "
          f"{fp_total/total_ol:>7.3f}×  {r_fp/r_ol:>5.2f}×")
    for ak, (r_a, u_a, w_a) in results_async.items():
        t = r_a + u_a + w_a
        tag = "  ★" if ak == 4 else ""
        print(f"  {f'Async K={ak} 双缓冲+LET':<32}  {t:>7.0f}ms  "
              f"{fp_total/t:>7.3f}×  {r_fp/r_a:>5.2f}×{tag}")
    print("─"*65)
    print(f"  理论上限(refit=0ms):  {theoretical:.3f}×")
    print("="*65)


if __name__ == "__main__":
    main()

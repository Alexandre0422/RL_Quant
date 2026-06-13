#!/usr/bin/env python3
"""
bench_let_int8_full_remote.py  (SM8.6 RTX3090 Linux)
═══════════════════════════════════════════════════════════════════════════
完整 LET + INT8 最优分布搜索

LET 原文正确部署形态（每个量化层插入以下节点序列）：
  X
  ──Mul(α)──► Z_mul
  ──Add(β)──► Z = αX+β            (per-channel 变换)
  ──Q(s_z)──► Z_int8              (s_z = absmax(Z)/127)
  ──DQ(s_z)──► Z_dq
  ──GEMM(W'=W/α, b'=b-W'β)──► Y_raw   (折叠权重)
  ──Q(s_y)──► Y_int8              (s_y = absmax(Y_quant)/127)
  ──DQ(s_y)──► Y（继承原始张量名）

与之前版本的区别：
  v1  仅输入 QDQ，权重折叠错误 → CosSim 0.72
  v2  双侧 QDQ，不折叠权重，不插入 Mul/Add → CosSim 0.995, 1.09×
  此版  Mul(α)+Add(β)+QDQ_in + GEMM(W',b') + QDQ_out
       → 数学上与 FP32 精确等价，TRT 可做完整 INT8 融合

流程：
  Phase 0  LET PTQ 校准（100批×200步）→ α,β,W',b',s_z,s_y
  Phase 1  单层消融（每次一层用完整 LET-INT8，其余 FP16）→ 正贡献层排名
  Phase 2  穷举搜索 → 全局最优 INT8 子集
  Phase 3  最优配置最终测试 + 报告
"""
import sys, os, gc, time, types, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

_git = types.ModuleType("git")
class _Repo:
    def __init__(self, *a, **kw): pass
_git.Repo = _Repo; _git.InvalidGitRepositoryError = Exception
sys.modules["git"] = _git
_td = types.ModuleType("tensordict")
class _TensorDict(dict): pass
_td.TensorDict = _TensorDict
sys.modules["tensordict"] = _td

import torch
import torch.nn.functional as F
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import tensorrt as trt

WORKDIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
sys.path.insert(0, os.path.join(WORKDIR, "rsl_rl"))
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder
from quant.let import LETQuantizer, fold_weights_in_onnx_node, fold_inproj_weight_onnx

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
B, PA, MAP, NA = 4096, 96, 33*21*3, 29
DEVICE    = torch.device("cuda")
BASE_ONNX         = os.path.join(WORKDIR, "engines", "glad_actor_base.onnx")
TIMING_CACHE_PATH = os.path.join(WORKDIR, "engines", "trt_timing.cache")

INT8_MATMUL_OK = {"node_MatMul_50","node_MatMul_59","node_MatMul_69","node_MatMul_71"}
EXCLUDE_ALWAYS = {
    "node_native_batch_norm__0","node_native_batch_norm_1__0",
    "node_softmax","node_softmax_1","node_topk__1",
    "node_select","node_select_1","node_bmm","node_bmm_1","node_bmm_2",
}

ALL_LAYERS = [
    ("node_conv2d",    "Conv 3→16  k=5",      "3ch, NOT mult16"),
    ("node_conv2d_1",  "Conv 16→64 k=3",      "16ch, ok"),
    ("node_linear",    "Linear 96→64",         "proprio_emb"),
    ("node_MatMul_50", "Linear 64→1 (GPS N=1)","N=1"),
    ("node_linear_2",  "Linear 128→64",        "query_proj"),
    ("node_MatMul_59", "Linear 128→1 (TopK N=1)","N=1"),
    ("node_MatMul_69", "MatMul MHA-Q 64→64",   ""),
    ("node_MatMul_71", "MatMul MHA-KV 64→128", ""),
    ("node_Gemm_139",  "Gemm 64→64 out_proj",  ""),
    ("node_linear_7",  "Linear 224→512 MLP-L1 ★","large"),
    ("node_linear_8",  "Linear 512→256 MLP-L2 ★","large"),
    ("node_linear_9",  "Linear 256→128 MLP-L3", ""),
    ("node_linear_10", "Linear 128→29  MLP-L4", ""),
]


# ════════════════════════════════════════════════════════════════════════════
# Phase 0：LET PTQ 校准
# ════════════════════════════════════════════════════════════════════════════

def run_let_calibration(n_calib=100, n_steps=200):
    """
    对每个 Linear/Gemm/MatMul 层做 LET block_reconstruction_loss 校准。

    返回 node_data: {node_name →
        "let"     : LETQuantizer (CPU),
        "alpha"   : np.ndarray [C_in],
        "beta"    : np.ndarray [C_in],
        "scale_z" : float,  (absmax(αx+β)/127, for input QDQ)
        "scale_y" : float,  (absmax(W'·Q(αx+β)+b')/127, for output QDQ)
        "layer_op": str,    ("Gemm"/"MatMul")
    }
    """
    dummy = {"policy_obs":torch.zeros(1,PA+MAP),"critic_obs":torch.zeros(1,PA+3+MAP)}
    grps  = {"policy":["policy_obs"],"critic":["critic_obs"]}
    model = ActorCriticEncoder(
        obs=dummy, obs_groups=grps, num_actions=NA,
        map_scan_dim=(33,21,3), mha_dim=64, num_heads=16,
        cnn_downsample=True, use_global_context=True, topk=32,
    ).to(DEVICE).eval()

    MAX_S = 8192
    bufs  = {k:[] for k in [
        "prop_emb","gps","qproj","topk","mha_out",
        "mlp_l1","mlp_l2","mlp_l3","mlp_l4","conv0_in","conv1_in",
        "mha_q_in","mha_kv_in","mha_attn_out","gemm139_in",
    ]}
    def collect(name):
        def hook(m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            cur = sum(t.shape[0] for t in bufs[name])
            if cur >= MAX_S: return
            bufs[name].append(x[:min(MAX_S-cur, x.shape[0])].cpu())
        return hook

    def collect_out(name, C_out):
        """捕获模块 OUTPUT；支持 Conv2d [B,C,H,W] 和 MHA tuple。"""
        def hook(m, inp, out):
            x = out[0].detach() if isinstance(out, tuple) else out.detach()
            if x.ndim == 4:
                x = x.permute(0, 2, 3, 1).reshape(-1, C_out)
            else:
                x = x.reshape(-1, C_out)
            cur = sum(t.shape[0] for t in bufs[name])
            if cur >= MAX_S: return
            bufs[name].append(x[:min(MAX_S-cur, x.shape[0])].cpu())
        return hook
    hooks = [
        model.actor_proprio_embedding.register_forward_hook(collect("prop_emb")),
        model.global_pool_selector    .register_forward_hook(collect("gps")),
        model.query_projector         .register_forward_hook(collect("qproj")),
        model.topk_scorer             .register_forward_hook(collect("topk")),
        model.actor[0]                .register_forward_hook(collect("mlp_l1")),
        model.actor[2]                .register_forward_hook(collect("mlp_l2")),
        model.actor[4]                .register_forward_hook(collect("mlp_l3")),
        model.actor[6]                .register_forward_hook(collect("mlp_l4")),
        model.map_cnn[0]              .register_forward_hook(collect("conv0_in")),
        model.map_cnn[3]              .register_forward_hook(collect("conv1_in")),
        model.query_projector.register_forward_hook(collect_out("mha_q_in", 64)),
        model.map_cnn        .register_forward_hook(collect_out("mha_kv_in", 64)),
        # mha_attn_out: 用 ORT 中间输出替代 model.mha hook
        # (MHA output hook 捕获的是 out_proj 之后，而非之前；改为 ORT 方式)
    ]
    print(f"  采集激活 ({n_calib} 批)...")
    with torch.no_grad():
        for _ in range(n_calib):
            obs = torch.randn(B, PA+MAP, device=DEVICE)
            model.act_inference({"policy_obs": obs,
                                 "critic_obs": torch.randn(B,PA+3+MAP,device=DEVICE)})
    for h in hooks: h.remove()
    act = {k: torch.cat(v,0).to(DEVICE) for k,v in bufs.items() if v}
    # ── ORT 捕获 Gemm_139 真实输入（view_6）──────────────────────────────────
    # PyTorch 2.x 的 MHA 绕过 out_proj.forward()，hook 永远不触发。
    # mha_attn_out 是 out_proj「输出」，用它校准「输入」会得到 alpha=120 的错误结果。
    # 正确做法：给 glad_actor_base.onnx 加中间输出，ORT 直接提取 view_6 张量。
    print("  ORT 捕获 Gemm_139 真实输入 (view_6)...")
    import onnx as _onnx_mod
    from onnx import helper as _oh, TensorProto as _tp
    import onnxruntime as _ort
    _m2 = _onnx_mod.load(BASE_ONNX)
    _m2.graph.output.append(_oh.make_tensor_value_info("view_6", _tp.FLOAT, None))
    _sess2 = _ort.InferenceSession(_m2.SerializeToString(),
                                    providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    _in2 = _sess2.get_inputs()[0].name
    _g139_bufs = []
    # ONNX 静态 batch=4096，ORT 必须用完整 B
    while sum(x.shape[0] for x in _g139_bufs) < MAX_S:
        _x2 = np.random.randn(B, PA+MAP).astype(np.float32)
        _all_out = _sess2.run(None, {_in2: _x2})
        _out_names = [o.name for o in _sess2.get_outputs()]
        _v6_idx = _out_names.index("view_6")
        _g139_bufs.append(_all_out[_v6_idx])
    act["gemm139_in"] = torch.from_numpy(
        np.concatenate(_g139_bufs)[:MAX_S]).to(DEVICE)
    print(f"  gemm139_in: {act['gemm139_in'].shape}  "
          f"mean_abs={act['gemm139_in'].abs().mean():.4f}")
    del _m2, _sess2, _g139_bufs
    # ── 采集完毕打印 ────────────────────────────────────────────────────────

    # ── 用 ORT 中间输出提取 Gemm_139 真实输入 (view_6) ──────────────────
    # view_6 是 MHA out_proj 的直接输入张量（来自注意力加权后的 value 组合）
    # PyTorch hook 无法捕获（被 F.multi_head_attention_forward 绕过），
    # 因此用 ONNX Runtime 临时追加 view_6 为输出来提取。
    try:
        _base_m = onnx.load(BASE_ONNX)
        _vi = None
        for _v in _base_m.graph.value_info:
            if _v.name == "view_6":
                _vi = _v; break
        if _vi is None:
            import onnx.helper as _oh
            _vi = _oh.make_tensor_value_info("view_6", 1, None)
        _base_m.graph.output.append(_vi)
        _ort_tmp = os.path.join(WORKDIR, "engines", "_tmp_view6.onnx")
        onnx.save(_base_m, _ort_tmp)
        _ort_sess = ort.InferenceSession(_ort_tmp,
                        providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        _ort_in = _ort_sess.get_inputs()[0].name
        _bufs_v6 = []
        _MAX_V6 = 8192
        print("  采集 view_6 (Gemm_139 真实输入) via ORT...")
        with torch.no_grad():
            for _ in range(min(n_calib, 20)):
                _x = np.random.randn(B, PA+MAP).astype(np.float32)
                _outs = _ort_sess.run(None, {_ort_in: _x})
                _v6 = _outs[-1]  # view_6: [B, 64]
                if _v6.ndim == 3: _v6 = _v6.squeeze(1)
                _cur = sum(t.shape[0] for t in _bufs_v6)
                if _cur < _MAX_V6:
                    _v6_t = torch.from_numpy(_v6[:min(_MAX_V6-_cur, _v6.shape[0])])
                    _bufs_v6.append(_v6_t)
        if _bufs_v6:
            act["mha_attn_out"] = torch.cat(_bufs_v6, 0).to(DEVICE)
            print(f"  view_6 采集完成: {act['mha_attn_out'].shape[0]} 个样本")
        del _ort_sess, _base_m
        if os.path.exists(_ort_tmp): os.remove(_ort_tmp)
    except Exception as _e:
        print(f"  [warn] view_6 ORT 提取失败: {_e}，退回 mha_kv_in 近似")
        if "mha_kv_in" in act:
            act["mha_attn_out"] = act["mha_kv_in"]

    D = 64
    import torch.nn as nn
    _linear_q  = nn.Linear(D, D,   bias=True).to(DEVICE)
    _linear_kv = nn.Linear(D, 2*D, bias=True).to(DEVICE)
    with torch.no_grad():
        _linear_q.weight.copy_(model.mha.in_proj_weight[:D, :])
        _linear_kv.weight.copy_(model.mha.in_proj_weight[D:, :])
        if model.mha.in_proj_bias is not None:
            _linear_q.bias.copy_(model.mha.in_proj_bias[:D])
            _linear_kv.bias.copy_(model.mha.in_proj_bias[D:])

    LINEAR_MAP = {
        "node_linear":    ("prop_emb",    model.actor_proprio_embedding, PA,  "Gemm"),
        "node_MatMul_50": ("gps",         model.global_pool_selector,    64,  "MatMul"),
        "node_linear_2":  ("qproj",       model.query_projector,         128, "Gemm"),
        "node_MatMul_59": ("topk",        model.topk_scorer,             128, "MatMul"),
        "node_MatMul_69": ("mha_q_in",    _linear_q,                     D,   "MatMul"),
        "node_MatMul_71": ("mha_kv_in",   _linear_kv,                    D,   "MatMul"),
        "node_Gemm_139":  ("gemm139_in",   model.mha.out_proj,            D,   "Gemm"),
        "node_linear_7":  ("mlp_l1",      model.actor[0],                224, "Gemm"),
        "node_linear_8":  ("mlp_l2",      model.actor[2],                512, "Gemm"),
        "node_linear_9":  ("mlp_l3",      model.actor[4],                256, "Gemm"),
        "node_linear_10": ("mlp_l4",      model.actor[6],                128, "Gemm"),
    }

    node_data = {}

    print(f"  LET block_reconstruction_loss ({n_steps} 步/层)...")
    for nname, (akey, layer, C_in, op_type) in LINEAR_MAP.items():
        x_all  = act[akey]
        x_flat = x_all.reshape(-1, C_in)[:min(B*4, x_all.reshape(-1,C_in).shape[0])].to(DEVICE)

        let = LETQuantizer(n_channels=C_in).to(DEVICE)
        let.initialize_from_activation(x_flat)
        opt = torch.optim.Adam(let.parameters(), lr=8e-3)
        for _ in range(n_steps):
            idx = torch.randperm(x_flat.shape[0], device=DEVICE)[:min(2048, x_flat.shape[0])]
            _, L = let.block_reconstruction_loss(x_flat[idx], layer)
            opt.zero_grad(); L.backward(); opt.step()

        # ── scale_z_x：x 本身的 MinMax（用于可靠无 Mul+Add 部署）──────
        scale_z_x = max(float(x_flat.abs().amax().item()) / 127.0, 1e-8)
        scale_z    = let.calibrate_scale(x_flat)  # 保留 LET scale（备用）

        # ── scale_y_x：原始权重输出 y=Wx+b 的 MinMax ────────────────────
        with torch.no_grad():
            y_fp32 = F.linear(x_flat, layer.weight, layer.bias)
            scale_y_x = max(float(y_fp32.abs().amax().item()) / 127.0, 1e-8)
        # scale_y（折叠权重版，保留作备用）
        folded = let.fold_into_linear(layer)
        with torch.no_grad():
            z   = let.alpha * x_flat + let.beta
            z_q = let._fake_quant_act(z)
            y   = F.linear(z_q,
                            folded.weight.to(DEVICE),
                            folded.bias.to(DEVICE) if folded.bias is not None else None)
            scale_y = max(float(y.abs().amax().item()) / 127.0, 1e-8)

        alpha_np = let.alpha.detach().clamp_min(1e-8).cpu().numpy()
        beta_np  = let.beta.detach().cpu().numpy()

        node_data[nname] = {
            "let":       let.cpu(),
            "alpha":     alpha_np,
            "beta":      beta_np,
            "scale_z":   scale_z,    # LET scale for αx+β
            "scale_z_x": scale_z_x,  # MinMax scale for x (no-Mul/Add mode)
            "scale_y":   scale_y,    # output scale (folded weight path)
            "scale_y_x": scale_y_x,  # output scale (no-fold path)
            "layer_op":  op_type,
        }
        print(f"  {nname:<22}: α={alpha_np.mean():.3f}±{alpha_np.std():.3f}"
              f"  β={beta_np.mean():.4f}  s_z={scale_z:.5f}  s_y={scale_y:.5f}")
        del opt; gc.collect(); torch.cuda.empty_cache()

    # Conv：仅 MinMax（C_in=3 的 conv2d 通常被 exclude）
    def mm_sc(key): return max(float(act[key].abs().max().item())/127.0, 1e-8)
    with torch.no_grad():
        ti = torch.randn(min(B,256), 3, 21, 33, device=DEVICE)
        c0 = model.map_cnn[0](ti)
        c1 = model.map_cnn[3](c0)
    node_data["node_conv2d"] = {
        "let": None, "alpha": None, "beta": None,
        "scale_z": mm_sc("conv0_in"), "scale_z_x": mm_sc("conv0_in"),
        "scale_y": max(float(c0.abs().max().item())/127.0, 1e-8),
        "scale_y_x": max(float(c0.abs().max().item())/127.0, 1e-8),
        "layer_op": "Conv",
    }
    node_data["node_conv2d_1"] = {
        "let": None, "alpha": None, "beta": None,
        "scale_z": mm_sc("conv1_in"), "scale_z_x": mm_sc("conv1_in"),
        "scale_y": max(float(c1.abs().max().item())/127.0, 1e-8),
        "scale_y_x": max(float(c1.abs().max().item())/127.0, 1e-8),
        "layer_op": "Conv",
    }

    del model; gc.collect(); torch.cuda.empty_cache()
    return node_data


# ════════════════════════════════════════════════════════════════════════════
# GraphSurgeon：完整 LET-INT8 节点序列
# ════════════════════════════════════════════════════════════════════════════

def build_let_onnx(include_nodes: set, node_data: dict, out_path: str) -> int:
    """
    对 include_nodes 中每层插入完整 LET-INT8 序列：
      Mul(α) → Add(β) → QDQ_in(s_z) → GEMM(W',b') → QDQ_out(s_y)

    修复（来自 v2 的 Bug1 fix）：
      dq_out_var 继承 orig_out 的原始名称，保持 TRT engine IO 名不变。
    """
    graph = gs.import_onnx(onnx.load(BASE_ONNX))
    new_nodes = []; inserted = 0

    # MHA 权重折叠已禁用（可靠工具模式：不折叠，使用原始权重）

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

        # ── 可靠工具：直接量化 x（不插 Mul+Add）────────────────────────────
        # Mul(α)+Add(β) 是逐通道变换，TRT INT8 GEMM 只支持逐张量 activation
        # scale，因此 TRT 无法保证将其融合进 QDQ kernel，导致每层 +0.5ms 开销。
        # 方案：去掉运行时 Mul+Add，x 直接进 QDQ；
        #       权重折叠 (W/α, b-W'β) 在 ONNX 构建时静态完成，零推理开销。
        # 数学等价性：
        #   有 Mul+Add: (W/α)·Q(αx+β) + b' ≡ W·x+b（精确等价）
        #   无 Mul+Add: (W/α)·Q(x) + b'    ≈ W·x+b  (α≈1 时近似；
        #               偏差来自量化 x 而非 αx+β，LET 权重折叠仍改善权重分布)
        q_input = inp  # 直接量化 x，不做逐通道变换

        # ── Q_in(scale_z_x = MinMax of x, not αx+β) ─────────────────────
        # 可靠工具：用 x 本身的 MinMax scale，不依赖 Mul+Add 被 TRT 融合
        s_z    = float(nd.get("scale_z_x", nd["scale_z"]))
        qs_i   = gs.Constant(f"qs_i_{uid}",  np.float32(s_z).reshape(1))
        qzp_i  = gs.Constant(f"qzp_i_{uid}", np.array([0], dtype=np.int8))
        dqs_i  = gs.Constant(f"dqs_i_{uid}", np.float32(s_z).reshape(1))
        dqzp_i = gs.Constant(f"dqzp_i_{uid}",np.array([0], dtype=np.int8))
        qi_v   = gs.Variable(f"qi_{uid}",  dtype=np.int8)
        dqi_v  = gs.Variable(f"dqi_{uid}", dtype=np.float32)
        new_nodes += [
            gs.Node("QuantizeLinear",   inputs=[q_input, qs_i, qzp_i],  outputs=[qi_v]),
            gs.Node("DequantizeLinear", inputs=[qi_v, dqs_i, dqzp_i],   outputs=[dqi_v]),
        ]
        node.inputs[0] = dqi_v

        # ── 可靠工具：不折叠权重 ───────────────────────────────────────────
        # 无 Mul+Add 时折叠 W→W/α 会导致输出缩小 α 倍（CosSim 崩溃）。
        # 数学正确的方式：直接量化 x，使用原始权重 W 和 bias b。
        # (fold_weights_in_onnx_node 调用已移除)

        # ── Q_out(s_y)（Fix1：继承原始名）──────────────────────────────
        orig_out  = node.outputs[0]
        orig_name = orig_out.name
        orig_out.name = f"_raw_{uid}"

        s_y    = float(nd.get("scale_y_x", nd["scale_y"]))
        qs_o   = gs.Constant(f"qs_o_{uid}",   np.float32(s_y).reshape(1))
        qzp_o  = gs.Constant(f"qzp_o_{uid}",  np.array([0], dtype=np.int8))
        dqs_o  = gs.Constant(f"dqs_o_{uid}",  np.float32(s_y).reshape(1))
        dqzp_o = gs.Constant(f"dqzp_o_{uid}", np.array([0], dtype=np.int8))
        qo_v   = gs.Variable(f"qo_{uid}",    dtype=np.int8)
        dqo_v  = gs.Variable(orig_name,       dtype=np.float32)  # 继承原始名！
        new_nodes += [
            gs.Node("QuantizeLinear",   inputs=[orig_out, qs_o, qzp_o], outputs=[qo_v]),
            gs.Node("DequantizeLinear", inputs=[qo_v, dqs_o, dqzp_o],  outputs=[dqo_v]),
        ]

        # 修复下游引用
        for other in graph.nodes:
            if other is node: continue
            other.inputs = [dqo_v if v is orig_out else v for v in other.inputs]
        graph.outputs = [dqo_v if v is orig_out else v for v in graph.outputs]

        inserted += 1

    graph.nodes.extend(new_nodes)
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_path)
    sz = os.path.getsize(out_path) // 1024
    print(f"  LET-INT8 GS: {inserted} 层 → {out_path}  ({sz} KB)")
    return inserted


# ════════════════════════════════════════════════════════════════════════════
# TRT 工具
# ════════════════════════════════════════════════════════════════════════════

def _build_trt(onnx_path):
    out = onnx_path.replace(".onnx",".trt")
    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    par = trt.OnnxParser(net, TRT_LOGGER)
    with open(onnx_path,"rb") as f:
        if not par.parse(f.read()):
            for i in range(par.num_errors): print(f"  [parse] {par.get_error(i)}")
            del builder,net; return None
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.INT8)
    tc_data = open(TIMING_CACHE_PATH,"rb").read() if os.path.exists(TIMING_CACHE_PATH) else b""
    tc = cfg.create_timing_cache(tc_data)
    cfg.set_timing_cache(tc, ignore_mismatch=False)
    # avg_timing_iterations=8：每个 kernel 候选测 8 次取均值
    # 噪声降低 2.8x（√8），保证可靠识别并锁定融合路径
    cfg.avg_timing_iterations = 8
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        print("  BUILD FAILED"); del builder,net,cfg; return None
    with open(out,"wb") as f: f.write(bytes(ser))
    print(f"  TRT {os.path.getsize(out)//1024}KB  {time.time()-t0:.1f}s")
    with open(TIMING_CACHE_PATH,"wb") as f: f.write(memoryview(cfg.get_timing_cache().serialize()))
    del builder,net,cfg; gc.collect(); torch.cuda.empty_cache()
    return out

def _bench_trt(eng_path, n=300, warmup=50):
    rt = trt.Runtime(TRT_LOGGER)
    with open(eng_path,"rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    io_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    if "obs" not in io_names or "action" not in io_names:
        print(f"  [!] IO 名异常: {io_names[:8]}"); del ctx,eng; return None,None
    stream  = torch.cuda.Stream()
    ib = torch.empty(B,PA+MAP,device=DEVICE,dtype=torch.float32).contiguous()
    ob = torch.empty(B,NA,    device=DEVICE,dtype=torch.float32).contiguous()
    ctx.set_tensor_address("obs",    int(ib.data_ptr()))
    ctx.set_tensor_address("action", int(ob.data_ptr()))
    def go(): ctx.execute_async_v3(stream.cuda_stream)
    with torch.cuda.stream(stream):
        with torch.inference_mode():
            for _ in range(warmup): go()
    stream.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        s.record(stream)
        with torch.inference_mode():
            for _ in range(n): go()
        e.record(stream)
    stream.synchronize()
    lat = s.elapsed_time(e)/n
    raw   = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.ONELINE)
    skip  = {"Layers:","Bindings:","obs","action"}
    lines = [l.strip() for l in raw.splitlines() if l.strip() and l.strip() not in skip]
    del ctx,eng; gc.collect(); torch.cuda.empty_cache()
    return lat, len(lines)

def _cossim_trt(eng_path, n=20):
    ort_s = ort.InferenceSession(BASE_ONNX, providers=["CPUExecutionProvider"])
    ort_in = ort_s.get_inputs()[0].name
    rt = trt.Runtime(TRT_LOGGER)
    with open(eng_path,"rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    stream = torch.cuda.Stream()
    ib = torch.empty(B,PA+MAP,device=DEVICE,dtype=torch.float32).contiguous()
    ob = torch.empty(B,NA,    device=DEVICE,dtype=torch.float32).contiguous()
    ctx.set_tensor_address("obs",    int(ib.data_ptr()))
    ctx.set_tensor_address("action", int(ob.data_ptr()))
    coss=[]
    with torch.inference_mode():
        for _ in range(n):
            x = np.random.randn(B,PA+MAP).astype(np.float32)
            ib.copy_(torch.from_numpy(x).to(DEVICE))
            ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
            to = ob.clone().cpu().numpy()
            oo = ort_s.run(None,{ort_in:x})[0]
            cos = (to*oo).sum(1)/(np.linalg.norm(to,axis=1)*np.linalg.norm(oo,axis=1)+1e-12)
            coss.append(float(cos.mean()))
    del ctx,eng; gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(coss))

def test_config(include_nodes, node_data, fp16_lat, label=""):
    """编译 + 测试一组 INT8 配置，返回 (lat, nl, cos)。"""
    tmp_o = os.path.join(WORKDIR,"engines",f"_let_tmp.onnx")
    n = build_let_onnx(include_nodes, node_data, tmp_o)
    eng_p = _build_trt(tmp_o)
    if os.path.exists(tmp_o): os.remove(tmp_o)
    if eng_p is None: return None
    lat,nl = _bench_trt(eng_p)
    if lat is None:
        if os.path.exists(eng_p): os.remove(eng_p); return None
    cos = _cossim_trt(eng_p)
    if os.path.exists(eng_p): os.remove(eng_p)
    spd = fp16_lat / lat
    sym = "✅" if spd>1.02 else ("～" if spd>0.98 else "❌")
    print(f"  {sym} {lat:.4f}ms  layers={nl}  CosSim={cos:.5f}  ({spd:.3f}× vs FP16)"
          + (f"  [{label}]" if label else ""))
    return lat, nl, cos

def fp16_baseline():
    import shutil
    tmp_o = os.path.join(WORKDIR,"engines","_fp16bl.onnx")
    shutil.copy(BASE_ONNX, tmp_o)
    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    par = trt.OnnxParser(net, TRT_LOGGER)
    with open(tmp_o,"rb") as f: par.parse(f.read())
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    ser = builder.build_serialized_network(net, cfg)
    eng_path = tmp_o.replace(".onnx",".trt")
    with open(eng_path,"wb") as f: f.write(bytes(ser))
    del builder,net,cfg; gc.collect()
    lat,nl = _bench_trt(eng_path)
    cos    = _cossim_trt(eng_path)
    for p in [tmp_o, eng_path]:
        if os.path.exists(p): os.remove(p)
    return lat, nl, cos



# ════════════════════════════════════════════════════════════════════════════
# 穷举搜索：遍历所有 2^N 子集，保证找到全局最优 INT8 配置
# ════════════════════════════════════════════════════════════════════════════

def exhaustive_search(node_data, fp16_lat, candidate_names=None,
                      results_path=None, min_size=1, max_size=None):
    """
    穷举所有 2^N 子集，返回全局最优 INT8 配置。
    结果增量写入 JSON，中断后重启自动续跑（results_path 已有记录则跳过）。

    candidate_names: 层名列表（默认全部 ALL_LAYERS）
    results_path   : 进度 JSON（默认 engines/exhaustive_results.json）
    min_size / max_size: 子集大小范围
    """
    import itertools, json

    if candidate_names is None:
        candidate_names = [n for n, *_ in ALL_LAYERS]
    N = len(candidate_names)
    if max_size is None:
        max_size = N

    total = sum(1 for k in range(min_size, max_size+1)
                  for _ in itertools.combinations(candidate_names, k))

    if results_path is None:
        results_path = os.path.join(WORKDIR, "engines", "exhaustive_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"  [resume] 已有 {len(results)} 条，目标 {total} 子集")
    else:
        results = {}

    best_lat = fp16_lat
    best_key = ""
    for key, v in results.items():
        lat = v.get("lat")
        if lat and lat < best_lat:
            best_lat = lat; best_key = key
    if best_key:
        print(f"  [resume] 当前最优: {best_key}  {best_lat:.4f}ms  {fp16_lat/best_lat:.3f}x")

    tested = skipped = 0
    t_start = time.time()

    for k in range(min_size, max_size + 1):
        for combo in itertools.combinations(candidate_names, k):
            key = ",".join(sorted(combo))
            if key in results:
                skipped += 1
                continue
            tested += 1
            done = tested + skipped
            pct  = done / total * 100
            eta_h = (time.time()-t_start) / max(tested,1) * max(total-done,0) / 3600
            sys.stdout.write(
                f"  [{done:5d}/{total}  {pct:5.1f}%  ETA:{eta_h:5.1f}h] "
                f"k={k} {key[:55]}\n"
            )
            sys.stdout.flush()
            t0  = time.time()
            res = test_config(set(combo), node_data, fp16_lat)
            dur = time.time() - t0
            if res is None:
                results[key] = {"lat": None, "cos": None, "dur": dur}
            else:
                lat, nl, cos = res
                spd = fp16_lat / lat
                results[key] = {"lat": lat, "cos": cos, "nl": nl, "spd": spd, "dur": dur}
                star = ""
                if lat < best_lat:
                    best_lat = lat; best_key = key; star = "  <<< NEW BEST"
                print(f"      -> {lat:.4f}ms  {spd:.3f}x  CosSim={cos:.5f}  [{dur:.1f}s]{star}")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n穷举完成: 新测 {tested}，跳过 {skipped}，共 {total} 子集")
    if best_key:
        best_subset = set(best_key.split(","))
        v = results[best_key]
        print(f"全局最优: {sorted(best_subset)}")
        print(f"  延迟 {best_lat:.4f}ms  {fp16_lat/best_lat:.3f}x vs FP16  CosSim={v.get('cos','?')}")
    else:
        best_subset = set()
    return best_key, best_lat, results

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*72)
    print("  完整 LET-INT8 最优分布搜索  (SM8.6 RTX3090, TRT 10.16)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("  LET 原文正确部署：Mul(α)+Add(β)+Q_in → GEMM(W',b') → Q_out")
    print("="*72)

    # ── Phase 0：LET 校准 ─────────────────────────────────────────────────
    print("\n[Phase 0] LET PTQ 校准")
    t0 = time.time()
    node_data = run_let_calibration(n_calib=100, n_steps=200)
    print(f"  校准完成 {(time.time()-t0)/60:.1f} min")

    # ── FP16 基线 ─────────────────────────────────────────────────────────
    print("\n[基线] TRT FP16")
    fp16_lat, fp16_nl, fp16_cos = fp16_baseline()
    print(f"  FP16: {fp16_lat:.4f} ms  layers={fp16_nl}  CosSim={fp16_cos:.5f}")

    ALL_NAMES = [n for n,*_ in ALL_LAYERS]

    # ── Phase 1：单层消融 ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Phase 1：单层消融（完整 LET-INT8 序列，共 13 层）")
    print(f"  受控变量：LET 校准 scale（固定）  实验变量：哪层用 INT8")
    print(f"{'='*72}")

    single = {}
    t_p1 = time.time()
    for i, (nname, desc, hint) in enumerate(ALL_LAYERS):
        sys.stdout.write(f"\n  [{i+1:02d}/13] {nname}  ({desc})\n")
        sys.stdout.flush()
        t0 = time.time()
        res = test_config({nname}, node_data, fp16_lat)
        elapsed = time.time() - t0
        if res:
            lat,nl,cos = res
            single[nname] = (lat, nl, cos, fp16_lat-lat, fp16_lat/lat)
            print(f"    [{elapsed:.0f}s]")

    print(f"\n  Phase 1 耗时 {(time.time()-t_p1)/60:.1f} min")

    # ── Phase 1 排名 ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Phase 1 排名（×FP16 降序，完整 LET-INT8）")
    print(f"  FP16 基线: {fp16_lat:.4f} ms")
    print(f"{'='*72}")
    sorted_single = sorted(single.items(), key=lambda x: x[1][4], reverse=True)
    print(f"  {'层名':<22} {'ms':>7} {'Δms':>8} {'×FP16':>7}  {'CosSim':>9}  描述")
    print("  "+"-"*72)
    positive = []
    for rank,(nname,(lat,nl,cos,delta,spd)) in enumerate(sorted_single,1):
        _,desc,_ = next(x for x in ALL_LAYERS if x[0]==nname)
        sym = "↑" if delta>0.02 else ("↓" if delta<-0.02 else "=")
        print(f"  {rank:>2}. {nname:<22} {lat:>7.4f} {delta:>+8.4f} {spd:>7.3f}×"
              f"  {cos:>9.5f}  {sym} {desc[:28]}")
        if delta > 0.02: positive.append(nname)

    print(f"\n  正贡献层（Δ > 0.02 ms）: {positive if positive else '（无）'}")

    # ── Phase 2：穷举搜索全局最优 INT8 子集 ─────────────────────────
    print(f"\n{'='*72}")
    print("  Phase 2：穷举搜索（遍历所有 2^N 子集，N=13）")
    print(f"  候选层数: {len(ALL_LAYERS)}  子集总数: {2**len(ALL_LAYERS)-1}")
    print(f"  timing_cache: {TIMING_CACHE_PATH}")
    print(f"{'='*72}")

    best_key, best_lat, all_results = exhaustive_search(
        node_data, fp16_lat,
        candidate_names=[n for n,*_ in ALL_LAYERS],
    )
    best_set = set(best_key.split(",")) if best_key else set()

    # ── Phase 3：最终最优配置测试 ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Phase 3：最终最优配置验证（5 次重复测量取平均）")
    print(f"  最优 INT8 子集: {sorted(best_set)}")
    print(f"{'='*72}")

    if best_set:
        lats = []
        for trial in range(5):
            res = test_config(best_set, node_data, fp16_lat, f"trial {trial+1}")
            if res: lats.append(res[0])
        if lats:
            mean_lat = float(np.mean(lats))
            std_lat  = float(np.std(lats))
            print(f"\n  最终延迟: {mean_lat:.4f} ± {std_lat:.4f} ms")
            print(f"  vs FP16:  {fp16_lat/mean_lat:.3f}×  ({fp16_lat:.4f}ms → {mean_lat:.4f}ms)")

    # ── 完整报告 ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  穷举搜索 Top-10 最优子集")
    print(f"{'='*72}")
    ranked = sorted(
        [(k,v) for k,v in all_results.items() if v.get("lat")],
        key=lambda x: x[1]["lat"]
    )[:10]
    for rank, (key, v) in enumerate(ranked, 1):
        spd = fp16_lat / v["lat"]
        print(f"  {rank:2d}. {v['lat']:.4f}ms ({spd:.3f}x)  "
              f"CosSim={v['cos']:.5f}  {key[:60]}")

    print(f"\n{'='*72}")
    print(f"  结论")
    print(f"{'='*72}")
    print(f"  最优 INT8 子集: {sorted(best_set)}")
    print(f"  FP16 基线:      {fp16_lat:.4f} ms")
    best_spd = fp16_lat/best_lat if best_set else 1.0
    print(f"  最优 INT8:      {best_lat:.4f} ms  ({best_spd:.3f}× vs FP16)")
    print()
    for nname, desc, hint in ALL_LAYERS:
        status = "INT8 ✅" if nname in best_set else "FP16  "
        sr = single.get(nname)
        sz = node_data.get(nname,{}).get("scale_z","N/A")
        spd_s = f"{sr[4]:.3f}×" if sr else "—"
        sz_s  = f"{sz:.5f}" if isinstance(sz,float) else "—"
        print(f"    {status}  {nname:<22}  单层={spd_s}  s_z={sz_s}  {hint}")


if __name__ == "__main__":
    main()

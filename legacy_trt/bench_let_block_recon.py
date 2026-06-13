# -*- coding: utf-8 -*-
"""
bench_let_block_recon.py
========================
用 block_reconstruction_loss（正确 LET 目标）逐层校准，
对比三种精度配置的加速效果和输出精度：

  FP16 TRT          — 参照基准
  INT8 GS scale=1.0 — 旧实现（z-space 代理）
  INT8 GS + block 重建校准 — 本脚本（正确 LET）

校准数据来源：100 batch rollout obs（随机，代表真实分布）
校准方式：每个 Linear 层独立做 block_reconstruction_loss 校准
          每个 Conv 层用 hook MinMax（Conv block-recon 留待后续）
精度参照：ORT FP32 同权重运行（glad_actor_base.onnx）
"""
import sys, os, ctypes, gc, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

_trt_libs = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\Lib\site-packages\tensorrt_libs"
_torch_lib = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\lib\site-packages\torch\lib"
os.add_dll_directory(_trt_libs); os.add_dll_directory(_torch_lib)
for d in ["nvinfer_10.dll","nvinfer_plugin_10.dll","nvonnxparser_10.dll"]:
    try: ctypes.CDLL(os.path.join(_trt_libs,d))
    except: pass

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, onnx
import tensorrt as trt

sys.path.insert(0,".")
sys.path.insert(0,"rsl_rl")
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder
from quant.let import LETQuantizer

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
B, PA, MAP, NA = 4096, 96, 33*21*3, 29
DEVICE = torch.device("cuda")

INT8_MATMUL = {"node_MatMul_50","node_MatMul_59","node_MatMul_69","node_MatMul_71"}
EXCLUDE     = {
    "node_native_batch_norm__0","node_native_batch_norm_1__0",
    "node_softmax","node_softmax_1","node_topk__1",
    "node_select","node_select_1",
    "node_bmm","node_bmm_1","node_bmm_2",
}

# ══════════════════════════════════════════════════════════════════════════════
# 校准：block 重建损失（Linear 层）+ MinMax（Conv 层）
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_block_recon(n_calib=100, n_steps=200):
    """
    逐层 block 重建校准。

    对每个 GLAD 里的 Linear 层：
      1. 用 hook 采集 n_calib 批输入激活
      2. 用 block_reconstruction_loss 优化 α, β, scale（200步）
      3. 提取 trt_scale = scale / α_mean

    对 Conv 层：MinMax 估计。
    """
    dummy = {"policy_obs":torch.zeros(1,PA+MAP),"critic_obs":torch.zeros(1,PA+3+MAP)}
    grps  = {"policy":["policy_obs"],"critic":["critic_obs"]}
    model = ActorCriticEncoder(
        obs=dummy,obs_groups=grps,num_actions=NA,
        map_scan_dim=(33,21,3),mha_dim=64,num_heads=16,
        cnn_downsample=True,use_global_context=True,topk=32,
    ).to(DEVICE).eval()

    # ── 1. 用 hook 采集各层输入激活样本 ──────────────────────────────────────
    # 注意：topk/gps 激活形状为 [B, N=187, C=128/64]，B=4096 时很大
    # 每层最多存 MAX_SAMPLES 个样本（展平后），避免 OOM
    MAX_SAMPLES = 8192   # 足够覆盖激活分布
    act_buffers = {k: [] for k in [
        "prop_emb","gps","qproj","topk","mha_out",
        "mlp_l1","mlp_l2","mlp_l3","mlp_l4",
        "conv0_in","conv1_in",
        # MHA 修复：捕获各投影层的真实激活输入
        "mha_q_in",    # query_projector 输出 = Q proj 输入 [B, D=64]
        "mha_kv_in",   # map_cnn 输出（local_features）= KV proj 输入近似 [B*N, D]
        "mha_attn_out",# MHA 模块输出 = out_proj 输入 [B, D=64]
    ]}

    def collect(name, max_batches=n_calib):
        """捕获模块 INPUT[0]（原有逻辑，用于 Linear 层）。"""
        def hook(m, inp, out):
            x = inp[0].detach()
            x_flat = x.reshape(-1, x.shape[-1])
            total  = act_buffers[name]
            cur_n  = sum(t.shape[0] for t in total)
            if cur_n >= MAX_SAMPLES:
                return
            keep = min(MAX_SAMPLES - cur_n, x_flat.shape[0])
            idx  = torch.randperm(x_flat.shape[0], device=x.device)[:keep]
            act_buffers[name].append(x_flat[idx].cpu())
        return hook

    def collect_out(name, C_out):
        """
        捕获模块 OUTPUT，用于 Q/KV/out_proj 激活。
        支持：
          - Linear   输出 [B, C]        → reshape(-1, C)
          - Conv2d   输出 [B, C, H, W]  → permute+reshape → [B*H*W, C]
          - MHA      输出 tuple(out, _)  → out[0] reshape(-1, C)
        """
        def hook(m, inp, out):
            x = out[0].detach() if isinstance(out, tuple) else out.detach()
            if x.ndim == 4:          # Conv2d: [B, C, H, W] → [B*H*W, C]
                x = x.permute(0, 2, 3, 1).reshape(-1, C_out)
            else:
                x = x.reshape(-1, C_out)
            total = act_buffers[name]
            cur_n = sum(t.shape[0] for t in total)
            if cur_n >= MAX_SAMPLES:
                return
            keep = min(MAX_SAMPLES - cur_n, x.shape[0])
            idx  = torch.randperm(x.shape[0], device=x.device)[:keep]
            act_buffers[name].append(x[idx].cpu())
        return hook

    hooks = [
        model.actor_proprio_embedding.register_forward_hook(collect("prop_emb")),
        model.global_pool_selector    .register_forward_hook(collect("gps")),
        model.query_projector         .register_forward_hook(collect("qproj")),
        model.topk_scorer             .register_forward_hook(collect("topk")),
        model.mha.out_proj            .register_forward_hook(collect("mha_out")),
        model.actor[0]                .register_forward_hook(collect("mlp_l1")),
        model.actor[2]                .register_forward_hook(collect("mlp_l2")),
        model.actor[4]                .register_forward_hook(collect("mlp_l3")),
        model.actor[6]                .register_forward_hook(collect("mlp_l4")),
        model.map_cnn[0]              .register_forward_hook(collect("conv0_in")),
        model.map_cnn[3]              .register_forward_hook(collect("conv1_in")),
        # MHA 输出侧 hook（捕获真实激活，而非近似）
        model.query_projector.register_forward_hook(collect_out("mha_q_in", 64)),
        model.map_cnn        .register_forward_hook(collect_out("mha_kv_in", 64)),
        model.mha            .register_forward_hook(collect_out("mha_attn_out", 64)),
    ]

    print(f"  采集激活样本 ({n_calib} 批)...")
    with torch.no_grad():
        for _ in range(n_calib):
            obs = torch.randn(B, PA+MAP, device=DEVICE)
            ao  = {"policy_obs": obs, "critic_obs": torch.randn(B,PA+3+MAP,device=DEVICE)}
            model.act_inference(ao)
    for h in hooks: h.remove()

    # 拼接采集结果
    act_data = {k: torch.cat(v, dim=0).to(DEVICE) for k,v in act_buffers.items() if v}
    print(f"  激活采集完毕：" + ", ".join(f"{k}={v.shape[0]}" for k,v in act_data.items()))

    # ── 2. 为 MHA Q/KV 投影构造临时 Linear（从 in_proj_weight 切片）──────────
    # PyTorch MHA 用合并权重 in_proj_weight [3D, D]，没有独立 nn.Linear 对象，
    # 需要手动构造才能传入 block_reconstruction_loss。
    D = 64
    _linear_q  = nn.Linear(D, D,   bias=True).to(DEVICE)   # Q  proj: [D, D]
    _linear_kv = nn.Linear(D, 2*D, bias=True).to(DEVICE)   # KV proj: [2D, D]
    with torch.no_grad():
        _linear_q.weight.copy_(model.mha.in_proj_weight[:D, :])
        _linear_kv.weight.copy_(model.mha.in_proj_weight[D:, :])
        if model.mha.in_proj_bias is not None:
            _linear_q.bias.copy_(model.mha.in_proj_bias[:D])
            _linear_kv.bias.copy_(model.mha.in_proj_bias[D:])

    # ── 3. 逐层 block 重建校准（Linear 层）──────────────────────────────────
    # 节点名 → (激活 key, Linear 层对象, C_in)
    # MatMul_69: Q proj，输入 = query_projector 输出（mha_q_in）[B, D]
    # MatMul_71: KV proj，输入 = local_features（mha_kv_in）[B*N, D]（local_sparse 近似）
    # Gemm_139:  out_proj，输入 = MHA 模块输出（mha_attn_out）[B, D]（hook 现已正确触发）
    LINEAR_MAP = {
        "node_linear":    ("prop_emb",    model.actor_proprio_embedding, PA),
        "node_MatMul_50": ("gps",         model.global_pool_selector,    64),
        "node_linear_2":  ("qproj",       model.query_projector,         128),
        "node_MatMul_59": ("topk",        model.topk_scorer,             128),
        "node_MatMul_69": ("mha_q_in",    _linear_q,                     D),
        "node_MatMul_71": ("mha_kv_in",   _linear_kv,                    D),
        "node_Gemm_139":  ("mha_attn_out",model.mha.out_proj,            D),
        "node_linear_7":  ("mlp_l1",      model.actor[0],                224),
        "node_linear_8":  ("mlp_l2",      model.actor[2],                512),
        "node_linear_9":  ("mlp_l3",      model.actor[4],                256),
        "node_linear_10": ("mlp_l4",      model.actor[6],                128),
    }

    node_scales = {}
    # MHA Q/KV 的 LET 校准结果需保留，供 GS 阶段折叠权重
    node_lets: dict[str, LETQuantizer] = {}

    print(f"\n  逐层 block 重建校准 ({n_steps} 步/层)...")

    for node_name, (act_key, layer, C_in) in LINEAR_MAP.items():
        if act_key not in act_data:
            print(f"  {node_name:<22}: 激活 '{act_key}' 未采集，跳过")
            continue
        x_all = act_data[act_key]
        x_flat = x_all.reshape(-1, C_in)[:min(B*4, x_all.reshape(-1,C_in).shape[0])].to(DEVICE)

        let = LETQuantizer(n_channels=C_in).to(DEVICE)
        let.initialize_from_activation(x_flat)
        opt = torch.optim.Adam(let.parameters(), lr=8e-3)

        for step in range(n_steps):
            idx = torch.randperm(x_flat.shape[0], device=DEVICE)[:min(2048,x_flat.shape[0])]
            x_b = x_flat[idx]
            _, L = let.block_reconstruction_loss(x_b, layer)
            opt.zero_grad(); L.backward(); opt.step()

        scale_x = let.calibrate_scale(x_flat)
        node_scales[node_name] = scale_x
        print(f"  {node_name:<22}: α_mean={let.alpha.abs().mean().item():.4f}  "
              f"β_mean={let.beta.abs().mean().item():.4f}  "
              f"→ trt_scale={scale_x:.5f}")

        # 保留 MHA Q/KV LET 对象（用于 fold_inproj_weight_onnx）
        if node_name in ("node_MatMul_69", "node_MatMul_71"):
            let.cpu()
            node_lets[node_name] = let
            del opt
        else:
            del let, opt
        gc.collect(); torch.cuda.empty_cache()

    # ── 3. Conv 层：MinMax 估计 ───────────────────────────────────────────────
    def minmax_scale(key):
        x = act_data[key]
        return float(x.abs().max().item()) / 127.0

    node_scales["node_conv2d"]   = minmax_scale("conv0_in")
    node_scales["node_conv2d_1"] = minmax_scale("conv1_in")
    print(f"  {'node_conv2d':<22}: trt_scale={node_scales['node_conv2d']:.5f}  (MinMax)")
    print(f"  {'node_conv2d_1':<22}: trt_scale={node_scales['node_conv2d_1']:.5f}  (MinMax)")

    del model; gc.collect(); torch.cuda.empty_cache()
    return node_scales, node_lets


# ══════════════════════════════════════════════════════════════════════════════
# GraphSurgeon
# ══════════════════════════════════════════════════════════════════════════════
def run_graphsurgeon(node_scales, node_lets=None,
                     out_onnx="engines/glad_actor_int8_block_recon.onnx"):
    import onnx_graphsurgeon as gs
    from quant.let import fold_inproj_weight_onnx

    graph = gs.import_onnx(onnx.load("engines/glad_actor_base.onnx"))

    # ── 折叠 MHA in_proj_weight（修复 MatMul_69/71 Variable 权重问题）────────
    # 必须在插入 QDQ 之前完成，避免 QDQ 节点干扰权重搜索。
    if node_lets:
        let_q  = node_lets.get("node_MatMul_69")
        let_kv = node_lets.get("node_MatMul_71")
        if let_q is not None:
            ok = fold_inproj_weight_onnx(graph, let_q, let_kv, D=64)
            if not ok:
                print("  [warn] fold_inproj_weight_onnx 失败，MatMul_69/71 权重未折叠")

    inserted = 0; new_nodes = []
    for node in graph.nodes:
        is_target = (node.op in {"Conv","Gemm"} or
                     (node.op=="MatMul" and node.name in INT8_MATMUL))
        if not is_target or node.name in EXCLUDE: continue
        inp = node.inputs[0]
        if not isinstance(inp, gs.Variable): continue
        sc  = float(node_scales.get(node.name, 0.05))
        uid = node.name.replace("/","_")
        q_s  = gs.Constant(f"qs_{uid}",  np.float32(sc).reshape(1))
        q_zp = gs.Constant(f"qz_{uid}",  np.array([0],dtype=np.int8))
        d_s  = gs.Constant(f"dqs_{uid}", np.float32(sc).reshape(1))
        d_zp = gs.Constant(f"dqz_{uid}", np.array([0],dtype=np.int8))
        q_out  = gs.Variable(f"q_{uid}",  dtype=np.int8)
        dq_out = gs.Variable(f"dq_{uid}", dtype=np.float32)
        q_node  = gs.Node("QuantizeLinear",  inputs=[inp,q_s,q_zp],    outputs=[q_out])
        dq_node = gs.Node("DequantizeLinear",inputs=[q_out,d_s,d_zp],  outputs=[dq_out])
        node.inputs[0] = dq_out
        new_nodes.extend([q_node,dq_node]); inserted+=1
    graph.nodes.extend(new_nodes); graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_onnx)
    print(f"  GS 完成: {inserted} 组 QDQ → {out_onnx}  ({os.path.getsize(out_onnx)//1024} KB)")
    return out_onnx


# ══════════════════════════════════════════════════════════════════════════════
# TRT 编译 + bench
# ══════════════════════════════════════════════════════════════════════════════
def build_trt(onnx_path, engine_path):
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    with open(onnx_path,"rb") as f: ok = parser.parse(f.read())
    if not ok: print("PARSE FAILED"); return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None: print("BUILD FAILED"); return False
    with open(engine_path,"wb") as f: f.write(bytes(ser))
    print(f"  TRT 编译: {os.path.getsize(engine_path)//1024} KB  {time.time()-t0:.1f}s")
    del builder, network, config; gc.collect(); torch.cuda.empty_cache()
    return True


def bench_engine(path, in_name, out_name, static=True, n=300, warmup=50):
    rt  = trt.Runtime(TRT_LOGGER)
    with open(path,"rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    if not static:
        try: ctx.set_input_shape(in_name,(B,PA+MAP))
        except: pass
    stream  = torch.cuda.Stream()
    in_buf  = torch.randn(B,PA+MAP,device=DEVICE).contiguous()
    out_buf = torch.empty(B,NA,   device=DEVICE).contiguous()
    ctx.set_tensor_address(in_name,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(out_name, int(out_buf.data_ptr()))
    def run(): ctx.execute_async_v3(stream.cuda_stream)
    with torch.cuda.stream(stream):
        with torch.inference_mode():
            for _ in range(warmup): run()
    stream.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        s.record(stream)
        with torch.inference_mode():
            for _ in range(n): run()
        e.record(stream)
    stream.synchronize()
    t = s.elapsed_time(e)/n
    # 层数
    insp  = eng.create_engine_inspector()
    raw   = insp.get_engine_information(trt.LayerInformationFormat.ONELINE)
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and l.strip() not in ("Layers:","Bindings:",in_name,out_name)]
    del ctx, eng; gc.collect(); torch.cuda.empty_cache()
    return t, len(lines), sum(1 for l in lines if any(k in l for k in ["MinMax","Roun"]))


def measure_precision(engine_path, in_name, out_name, ort_sess, ort_in, static, n_batches=50):
    import onnxruntime as ort
    rt  = trt.Runtime(TRT_LOGGER)
    with open(engine_path,"rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    if not static:
        try: ctx.set_input_shape(in_name,(B,PA+MAP))
        except: pass
    stream  = torch.cuda.Stream()
    in_buf  = torch.empty(B,PA+MAP,device=DEVICE,dtype=torch.float32).contiguous()
    out_buf = torch.empty(B,NA,   device=DEVICE,dtype=torch.float32).contiguous()
    ctx.set_tensor_address(in_name,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(out_name, int(out_buf.data_ptr()))
    maes, coss = [], []
    with torch.inference_mode():
        for _ in range(n_batches):
            x_np = np.random.randn(B,PA+MAP).astype(np.float32)
            in_buf.copy_(torch.from_numpy(x_np).to(DEVICE))
            ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
            trt_out = out_buf.clone().cpu().numpy()
            ort_out = ort_sess.run(None,{ort_in: x_np})[0]
            maes.append(np.abs(trt_out-ort_out).mean())
            cos = (trt_out*ort_out).sum(1)/(
                  np.linalg.norm(trt_out,axis=1)*np.linalg.norm(ort_out,axis=1)+1e-12)
            coss.append(cos.mean())
    del ctx,eng; gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(maes)), float(np.mean(coss))


# ══════════════════════════════════════════════════════════════════════════════
# FP32 PyTorch 延迟
# ══════════════════════════════════════════════════════════════════════════════
def measure_fp32():
    dummy = {"policy_obs":torch.zeros(1,PA+MAP),"critic_obs":torch.zeros(1,PA+3+MAP)}
    grps  = {"policy":["policy_obs"],"critic":["critic_obs"]}
    m = ActorCriticEncoder(obs=dummy,obs_groups=grps,num_actions=NA,
            map_scan_dim=(33,21,3),mha_dim=64,num_heads=16,
            cnn_downsample=True,use_global_context=True,topk=32).to(DEVICE).eval()
    class AW(nn.Module):
        def __init__(self,m): super().__init__(); self.m=m
        def forward(self,x): e,_,_,_=self.m._encode_terrain(m.actor_obs_normalizer(x)); return m.actor(e)
    w = AW(m); obs = torch.randn(B,PA+MAP,device=DEVICE)
    with torch.inference_mode():
        for _ in range(30): w(obs)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        for _ in range(200): w(obs)
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e)/200
    del m,w; gc.collect(); torch.cuda.empty_cache()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*68)
    print("  LET block_reconstruction_loss 逐层校准 → GS → TRT INT8")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  TRT: {trt.__version__}")
    print("="*68)

    # ── 校准 ──────────────────────────────────────────────────────────────────
    print("\n[1] 逐层 block 重建校准")
    node_scales, node_lets = calibrate_block_recon(n_calib=100, n_steps=200)

    # ── GS ────────────────────────────────────────────────────────────────────
    print("\n[2] GraphSurgeon")
    run_graphsurgeon(node_scales, node_lets, "engines/glad_actor_int8_block_recon.onnx")

    # ── TRT 编译 ───────────────────────────────────────────────────────────────
    print("\n[3] TRT INT8 编译")
    if not build_trt("engines/glad_actor_int8_block_recon.onnx",
                     "engines/glad_actor_int8_block_recon.trt"):
        return

    # ── ORT 参照 ──────────────────────────────────────────────────────────────
    import onnxruntime as ort
    ort_sess = ort.InferenceSession("engines/glad_actor_base.onnx",
                                    providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    ort_in = ort_sess.get_inputs()[0].name

    # ── 延迟测量 ──────────────────────────────────────────────────────────────
    print("\n[4] 延迟测量")
    t_fp32 = measure_fp32()
    t_fp16, nl16, nq16   = bench_engine("engines/glad_actor_fp16.trt",             "actor_obs","action_mean",False)
    t_gs1,  nl_g1, nq_g1 = bench_engine("engines/glad_actor_int8_gs.trt",          "obs","action",            True)
    t_br,   nl_br, nq_br = bench_engine("engines/glad_actor_int8_block_recon.trt", "obs","action",             True)

    # ── 精度测量 ──────────────────────────────────────────────────────────────
    print("[5] 精度测量（vs ORT FP32，50 batch）")
    mae_gs1, cos_gs1 = measure_precision("engines/glad_actor_int8_gs.trt",          "obs","action",ort_sess,ort_in,True)
    mae_br,  cos_br  = measure_precision("engines/glad_actor_int8_block_recon.trt", "obs","action",ort_sess,ort_in,True)

    # ── 报告 ─────────────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  完整报告")
    print("="*68)

    print(f"""
┌─ 加速效果（B=4096 rollout，专用 stream）──────────────────────┐
│  {'配置':<32} {'ms':>8} {'层数':>6} {'vs FP32':>8} {'vs FP16':>8} │
│  {'─'*62} │
│  {'FP32 PyTorch':<32} {t_fp32:>8.3f}    {'—':>6} {'1.00×':>8}    {'—':>8} │
│  {'FP16 TRT':<32} {t_fp16:>8.3f} {nl16:>6} {t_fp32/t_fp16:>7.2f}×    {'1.00×':>8} │
│  {'INT8 GS (scale=1.0)':<32} {t_gs1:>8.3f} {nl_g1:>6} {t_fp32/t_gs1:>7.2f}× {t_fp16/t_gs1:>7.2f}× │
│  {'INT8 LET block 重建校准':<32} {t_br:>8.3f} {nl_br:>6} {t_fp32/t_br:>7.2f}× {t_fp16/t_br:>7.2f}× │
└──────────────────────────────────────────────────────────────┘

┌─ 精度（vs ORT FP32，同权重）──────────────────────────────────┐
│  {'配置':<32} {'CosSim':>10} {'MAE':>12}              │
│  {'─'*56}              │
│  {'INT8 GS scale=1.0（未校准）':<32} {cos_gs1:>10.6f} {mae_gs1:>12.6f}              │
│  {'INT8 LET block 重建校准':<32} {cos_br:>10.6f} {mae_br:>12.6f}              │
└──────────────────────────────────────────────────────────────┘

  结论：
  • 加速：LET block 重建校准 {t_fp16/t_br:.2f}× vs TRT FP16，{t_fp32/t_br:.2f}× vs FP32
  • 精度：LET block 重建校准 CosSim={cos_br:.4f}，
          vs scale=1.0 未校准 CosSim={cos_gs1:.4f}（提升={cos_br-cos_gs1:+.4f}）
  • 校准成本：一次性（100 批采样 + 每层 200 步，含 α/β 联合优化）
""")


if __name__ == "__main__":
    main()

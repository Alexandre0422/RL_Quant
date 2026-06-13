# -*- coding: utf-8 -*-
"""
bench_precision_ablation.py
===========================
精度消融实验：量化精度对 GLAD encoder 效率的独立影响

控制变量设计
────────────
所有 TRT 引擎编译自同一 ONNX（glad_actor_base.onnx 或其 QDQ 变体），
相同静态 batch B=4096，相同 IO 名称（obs / action），相同 workspace 2GB。
唯一受控变量：TRT builder 精度 flag。

四个测试点
──────────
  [0] PyTorch FP32          — 绝对基线（无 TRT 图优化，无精度变换）
  [A] TRT FP32              — 仅 TRT 图优化（kernel fusion/layout），精度不变
  [B] TRT FP16              — 在 A 基础上开启 FP16（Tensor Core + 带宽减半）
  [C] TRT FP16 + INT8       — 在 B 基础上叠加 INT8 量化（关键层带宽再减半）
                              使用 LET block 重建校准的 GS-QDQ ONNX

速度分解
────────
  [0]→[A]  TRT 图优化本身的贡献（与精度无关）
  [A]→[B]  FP16 精度的贡献
  [B]→[C]  INT8 量化的贡献

临时文件：A 和 B 的引擎编译到 engines/_abl_fp32.trt / engines/_abl_fp16.trt，
          实验结束后自动删除，不污染 engines/ 目录。
"""

import sys, os, ctypes, gc, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

_trt_libs  = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\Lib\site-packages\tensorrt_libs"
_torch_lib = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\lib\site-packages\torch\lib"
os.add_dll_directory(_trt_libs); os.add_dll_directory(_torch_lib)
for _d in ["nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"]:
    try: ctypes.CDLL(os.path.join(_trt_libs, _d))
    except: pass

import torch
import numpy as np
import tensorrt as trt
import onnxruntime as ort

sys.path.insert(0, "."); sys.path.insert(0, "rsl_rl")
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = torch.device("cuda")

# ── 模型常量（与 ONNX 静态 batch 一致）──────────────────────────────────────
B, PA, MAP, NA = 4096, 96, 33 * 21 * 3, 29
IO_IN, IO_OUT  = "obs", "action"          # 所有受控引擎共用同一 IO 名

# ── 文件路径 ─────────────────────────────────────────────────────────────────
BASE_ONNX = "engines/glad_actor_base.onnx"
INT8_ONNX = "engines/glad_actor_int8_block_recon.onnx"  # 带 QDQ 的 INT8 源
INT8_TRT  = "engines/glad_actor_int8_block_recon.trt"   # 已有生产引擎

# 临时引擎：实验结束后自动清理
_FP32_TRT = "engines/_abl_fp32.trt"
_FP16_TRT = "engines/_abl_fp16.trt"


# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════

def build_engine(onnx_path: str, out_path: str, fp16: bool, int8: bool) -> bool:
    """
    从 ONNX 编译 TRT 引擎。
    fp16=False, int8=False → TRT FP32（只做图优化，不改精度）
    fp16=True,  int8=False → TRT FP16
    fp16=True,  int8=True  → TRT FP16+INT8（需要 QDQ 注释）
    """
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [parse error] {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    t0  = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        print("  [!] build failed")
        return False

    with open(out_path, "wb") as f:
        f.write(bytes(ser))
    kb = os.path.getsize(out_path) // 1024
    print(f"    {os.path.basename(out_path)}  {kb} KB  {time.time()-t0:.1f}s")

    del builder, network, config
    gc.collect(); torch.cuda.empty_cache()
    return True


def _load_engine(path: str):
    rt = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return rt.deserialize_cuda_engine(f.read())


def _engine_layers(eng) -> tuple[int, int]:
    """返回 (总层数, INT8 层数)。"""
    insp = eng.create_engine_inspector()
    raw  = insp.get_engine_information(trt.LayerInformationFormat.ONELINE)
    skip = {"Layers:", "Bindings:", IO_IN, IO_OUT}
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and l.strip() not in skip]
    total = len(lines)
    n_int8 = sum(1 for l in lines if "Int8" in l or "INT8" in l)
    return total, n_int8


def measure_latency(engine_path: str,
                    n: int = 300, warmup: int = 50) -> tuple[float, int, int]:
    """返回 (延迟ms, 总层数, INT8层数)。"""
    eng    = _load_engine(engine_path)
    ctx    = eng.create_execution_context()
    stream = torch.cuda.Stream()
    in_buf  = torch.randn(B, PA + MAP, device=DEVICE).contiguous()
    out_buf = torch.empty(B, NA,       device=DEVICE).contiguous()
    ctx.set_tensor_address(IO_IN,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(IO_OUT, int(out_buf.data_ptr()))

    def run(): ctx.execute_async_v3(stream.cuda_stream)

    with torch.cuda.stream(stream):
        with torch.inference_mode():
            for _ in range(warmup): run()
    stream.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        s.record(stream)
        with torch.inference_mode():
            for _ in range(n): run()
        e.record(stream)
    stream.synchronize()
    lat = s.elapsed_time(e) / n

    total, n_int8 = _engine_layers(eng)
    del ctx, eng; gc.collect(); torch.cuda.empty_cache()
    return lat, total, n_int8


def measure_cossim(engine_path: str, ort_sess, ort_in_name: str,
                   n_batches: int = 50) -> tuple[float, float]:
    """返回 (均值CosSim, 均值MAE)。"""
    eng    = _load_engine(engine_path)
    ctx    = eng.create_execution_context()
    stream = torch.cuda.Stream()
    in_buf  = torch.empty(B, PA + MAP, device=DEVICE, dtype=torch.float32).contiguous()
    out_buf = torch.empty(B, NA,       device=DEVICE, dtype=torch.float32).contiguous()
    ctx.set_tensor_address(IO_IN,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(IO_OUT, int(out_buf.data_ptr()))

    coss, maes = [], []
    with torch.inference_mode():
        for _ in range(n_batches):
            x_np = np.random.randn(B, PA + MAP).astype(np.float32)
            in_buf.copy_(torch.from_numpy(x_np).to(DEVICE))
            ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
            trt_out = out_buf.clone().cpu().numpy()
            ort_out = ort_sess.run(None, {ort_in_name: x_np})[0]
            maes.append(float(np.abs(trt_out - ort_out).mean()))
            cos = ((trt_out * ort_out).sum(1) /
                   (np.linalg.norm(trt_out, axis=1) *
                    np.linalg.norm(ort_out, axis=1) + 1e-12))
            coss.append(float(cos.mean()))

    del ctx, eng; gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(coss)), float(np.mean(maes))


def measure_pytorch_fp32(n: int = 300, warmup: int = 50) -> float:
    """返回 PyTorch FP32 rollout 延迟 ms。"""
    dummy = {"policy_obs": torch.zeros(1, PA + MAP),
             "critic_obs": torch.zeros(1, PA + 3 + MAP)}
    grps  = {"policy": ["policy_obs"], "critic": ["critic_obs"]}
    m = ActorCriticEncoder(
        obs=dummy, obs_groups=grps, num_actions=NA,
        map_scan_dim=(33, 21, 3), mha_dim=64, num_heads=16,
        cnn_downsample=True, use_global_context=True, topk=32
    ).to(DEVICE).eval()

    obs = torch.randn(B, PA + MAP, device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            enc, _, _, _ = m._encode_terrain(m.actor_obs_normalizer(obs))
            _ = m.actor(enc)

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        for _ in range(n):
            enc, _, _, _ = m._encode_terrain(m.actor_obs_normalizer(obs))
            _ = m.actor(enc)
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e) / n

    del m; gc.collect(); torch.cuda.empty_cache()
    return t


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  精度消融实验：FP32 → FP16 → FP16+INT8 逐步效果分解")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  TRT : {trt.__version__}   PyTorch : {torch.__version__}")
    print()
    print("  控制变量：相同 ONNX 权重 | 相同静态 B=4096 | 相同 IO(obs/action)")
    print("  单一变量：TRT builder 精度 flag")
    print("=" * 72)

    # ── 步骤 1：编译两个受控临时引擎 ─────────────────────────────────────────
    print("\n[1] 编译受控引擎")
    print("  [A] TRT FP32  ← glad_actor_base.onnx，无精度 flag（纯图优化）")
    if not build_engine(BASE_ONNX, _FP32_TRT, fp16=False, int8=False):
        return

    print("  [B] TRT FP16  ← glad_actor_base.onnx，FP16 flag")
    if not build_engine(BASE_ONNX, _FP16_TRT, fp16=True, int8=False):
        return

    print("  [C] TRT FP16+INT8 ← glad_actor_int8_block_recon.onnx（已有，复用）")

    # ── 步骤 2：ORT FP32 精度参照 ────────────────────────────────────────────
    ort_sess = ort.InferenceSession(BASE_ONNX, providers=["CPUExecutionProvider"])
    ort_in   = ort_sess.get_inputs()[0].name  # "obs"

    # ── 步骤 3：延迟测量（300 次，专用 CUDA stream）──────────────────────────
    print("\n[2] 延迟测量  (B=4096, 300次, warmup=50, 专用 CUDA stream)")
    t_pt              = measure_pytorch_fp32()
    t_a, nl_a, ni_a   = measure_latency(_FP32_TRT)
    t_b, nl_b, ni_b   = measure_latency(_FP16_TRT)
    t_c, nl_c, ni_c   = measure_latency(INT8_TRT)
    print(f"  [0] PyTorch FP32          : {t_pt:7.3f} ms")
    print(f"  [A] TRT FP32              : {t_a:7.3f} ms  layers={nl_a}  INT8={ni_a}")
    print(f"  [B] TRT FP16              : {t_b:7.3f} ms  layers={nl_b}  INT8={ni_b}")
    print(f"  [C] TRT FP16+INT8         : {t_c:7.3f} ms  layers={nl_c}  INT8={ni_c}")

    # ── 步骤 4：精度测量（50 batch，vs ORT FP32）────────────────────────────
    print("\n[3] 精度测量  (CosSim vs ORT FP32，50 batch × B=4096)")
    cos_a, mae_a = measure_cossim(_FP32_TRT, ort_sess, ort_in)
    cos_b, mae_b = measure_cossim(_FP16_TRT, ort_sess, ort_in)
    cos_c, mae_c = measure_cossim(INT8_TRT,  ort_sess, ort_in)
    print(f"  [A] TRT FP32              : CosSim={cos_a:.6f}  MAE={mae_a:.6f}")
    print(f"  [B] TRT FP16              : CosSim={cos_b:.6f}  MAE={mae_b:.6f}")
    print(f"  [C] TRT FP16+INT8         : CosSim={cos_c:.6f}  MAE={mae_c:.6f}")

    # ── 步骤 5：汇总报告 ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  完整结果表")
    print("=" * 72)

    hdr = f"  {'配置':<24} {'ms':>7} {'层数':>5} {'INT8层':>6}  {'CosSim':>10}  {'MAE':>10}  {'vs[0]':>7}  {'vs[A]':>7}"
    sep = "  " + "─" * 70
    print(hdr); print(sep)

    def row(tag, t, nl, ni, cos, mae, vs0=None, vsa=None):
        nl_s  = f"{nl}"   if nl  is not None else "—"
        ni_s  = f"{ni}"   if ni  is not None else "—"
        cos_s = f"{cos:.6f}" if cos is not None else "—"
        mae_s = f"{mae:.6f}" if mae is not None else "—"
        vs0_s = f"{vs0:.2f}×" if vs0 is not None else "—"
        vsa_s = f"{vsa:.2f}×" if vsa is not None else "—"
        print(f"  {tag:<24} {t:>7.3f} {nl_s:>5} {ni_s:>6}  {cos_s:>10}  {mae_s:>10}  {vs0_s:>7}  {vsa_s:>7}")

    row("[0] PyTorch FP32",  t_pt, None, None, None,  None)
    row("[A] TRT FP32",      t_a,  nl_a, ni_a, cos_a, mae_a, t_pt/t_a, None)
    row("[B] TRT FP16",      t_b,  nl_b, ni_b, cos_b, mae_b, t_pt/t_b, t_a/t_b)
    row("[C] TRT FP16+INT8", t_c,  nl_c, ni_c, cos_c, mae_c, t_pt/t_c, t_a/t_c)

    # 速度分解
    d_trt  = t_pt - t_a            # TRT 图优化贡献
    d_fp16 = t_a  - t_b            # FP16 精度贡献
    d_int8 = t_b  - t_c            # INT8 量化贡献
    d_tot  = t_pt - t_c            # 总加速

    print(f"""
  速度分解（绝对贡献 / 占总加速比例）
  ────────────────────────────────────────────────────────
  [0]→[A]  TRT 图优化       : {d_trt:+.3f} ms  ({d_trt/d_tot*100:5.1f}% of total speedup)
  [A]→[B]  FP16 精度        : {d_fp16:+.3f} ms  ({d_fp16/d_tot*100:5.1f}% of total speedup)
  [B]→[C]  INT8 量化        : {d_int8:+.3f} ms  ({d_int8/d_tot*100:5.1f}% of total speedup)
  ────────────────────────────────────────────────────────
  总加速                    : {d_tot:+.3f} ms  = {t_pt/t_c:.2f}× vs PyTorch FP32

  精度损失（ΔCosSim vs FP32 参照）
  ────────────────────────────────────────────────────────
  [A] TRT FP32  精度损失    : {cos_a-1:.6f}  （TRT 图优化自身引入）
  [B] TRT FP16  额外损失    : {cos_b-cos_a:+.6f}  （FP16 引入，累计 {cos_b-1:.6f}）
  [C] TRT INT8  额外损失    : {cos_c-cos_b:+.6f}  （INT8 引入，累计 {cos_c-1:.6f}）
""")

    # ── 清理临时引擎 ──────────────────────────────────────────────────────────
    for p in [_FP32_TRT, _FP16_TRT]:
        if os.path.exists(p):
            os.remove(p)
    print("  [done] 临时引擎已清理")


if __name__ == "__main__":
    main()

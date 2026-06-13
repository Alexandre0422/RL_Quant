# -*- coding: utf-8 -*-
"""
run_bench_report.py
仅做延迟 + 精度测量，不重新校准/编译。
依赖已存在的引擎文件：
  glad_actor_fp16.trt             (FP16 参照)
  glad_actor_int8_gs.trt          (INT8 scale=1.0 未校准基线)
  glad_actor_int8_block_recon.trt (INT8 LET block重建校准，新版)
精度参照：glad_actor_base.onnx 通过 ORT CPU Provider 运行
"""
import sys, os, ctypes, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

_trt_libs  = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\Lib\site-packages\tensorrt_libs"
_torch_lib = r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\lib\site-packages\torch\lib"
os.add_dll_directory(_trt_libs); os.add_dll_directory(_torch_lib)
for d in ["nvinfer_10.dll","nvinfer_plugin_10.dll","nvonnxparser_10.dll"]:
    try: ctypes.CDLL(os.path.join(_trt_libs, d))
    except: pass

import torch, torch.nn as nn
import numpy as np
import tensorrt as trt
import onnxruntime as ort

sys.path.insert(0, "."); sys.path.insert(0, "rsl_rl")
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
B, PA, MAP, NA = 4096, 96, 33*21*3, 29
DEVICE = torch.device("cuda")


# ─── FP32 PyTorch 延迟 ────────────────────────────────────────────────────────
def measure_fp32_latency(n=300, warmup=50):
    dummy = {"policy_obs": torch.zeros(1, PA+MAP),
             "critic_obs": torch.zeros(1, PA+3+MAP)}
    grps  = {"policy": ["policy_obs"], "critic": ["critic_obs"]}
    m = ActorCriticEncoder(obs=dummy, obs_groups=grps, num_actions=NA,
            map_scan_dim=(33,21,3), mha_dim=64, num_heads=16,
            cnn_downsample=True, use_global_context=True, topk=32
        ).to(DEVICE).eval()

    obs = torch.randn(B, PA+MAP, device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            e, _, _, _ = m._encode_terrain(m.actor_obs_normalizer(obs))
            _ = m.actor(e)

    s = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        for _ in range(n):
            enc, _, _, _ = m._encode_terrain(m.actor_obs_normalizer(obs))
            _ = m.actor(enc)
    e_ev.record(); torch.cuda.synchronize()
    return s.elapsed_time(e_ev) / n


# ─── TRT 引擎延迟 ─────────────────────────────────────────────────────────────
def measure_trt_latency(engine_path, in_name, out_name, static=True,
                        n=300, warmup=50):
    rt = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx    = eng.create_execution_context()
    if not static:
        try: ctx.set_input_shape(in_name, (B, PA+MAP))
        except: pass
    stream = torch.cuda.Stream()
    in_buf  = torch.randn(B, PA+MAP, device=DEVICE).contiguous()
    out_buf = torch.empty(B, NA,     device=DEVICE).contiguous()
    ctx.set_tensor_address(in_name,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(out_name, int(out_buf.data_ptr()))

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
    t = s.elapsed_time(e) / n

    # 层数
    insp  = eng.create_engine_inspector()
    raw   = insp.get_engine_information(trt.LayerInformationFormat.ONELINE)
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and l.strip() not in ("Layers:", "Bindings:", in_name, out_name)]
    return t, len(lines)


# ─── 精度（vs ORT FP32）─────────────────────────────────────────────────────
def measure_precision(engine_path, in_name, out_name, ort_sess, ort_in_name,
                      static=True, n_batches=50):
    rt = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx    = eng.create_execution_context()
    if not static:
        try: ctx.set_input_shape(in_name, (B, PA+MAP))
        except: pass
    stream  = torch.cuda.Stream()
    in_buf  = torch.empty(B, PA+MAP, device=DEVICE, dtype=torch.float32).contiguous()
    out_buf = torch.empty(B, NA,     device=DEVICE, dtype=torch.float32).contiguous()
    ctx.set_tensor_address(in_name,  int(in_buf.data_ptr()))
    ctx.set_tensor_address(out_name, int(out_buf.data_ptr()))

    maes, coss = [], []
    with torch.inference_mode():
        for _ in range(n_batches):
            x_np = np.random.randn(B, PA+MAP).astype(np.float32)
            in_buf.copy_(torch.from_numpy(x_np).to(DEVICE))
            ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
            trt_out = out_buf.clone().cpu().numpy()
            ort_out = ort_sess.run(None, {ort_in_name: x_np})[0]
            maes.append(float(np.abs(trt_out - ort_out).mean()))
            cos = (trt_out * ort_out).sum(1) / (
                  np.linalg.norm(trt_out, axis=1) *
                  np.linalg.norm(ort_out, axis=1) + 1e-12)
            coss.append(float(cos.mean()))
    return float(np.mean(maes)), float(np.mean(coss))


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  GLAD 量化效果报告（LET block 重建校准 vs 基线）")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  TRT: {trt.__version__}")
    print("=" * 68)

    # 检查引擎文件
    engines = {
        "FP16":        ("engines/glad_actor_fp16.trt",             "actor_obs", "action_mean", False),
        "INT8 未校准": ("engines/glad_actor_int8_gs.trt",          "obs",       "action",      True),
        "INT8 LET":    ("engines/glad_actor_int8_block_recon.trt", "obs",       "action",      True),
    }
    missing = [k for k, (p,*_) in engines.items() if not os.path.exists(p)]
    if missing:
        print(f"[!] 以下引擎文件不存在，跳过：{missing}")

    # ORT 精度参照
    ort_path = "engines/glad_actor_base.onnx"
    has_ort = os.path.exists(ort_path)
    if has_ort:
        ort_sess = ort.InferenceSession(ort_path, providers=["CPUExecutionProvider"])
        ort_in   = ort_sess.get_inputs()[0].name
        print(f"  精度参照: {ort_path} via ORT CPU\n")
    else:
        print(f"  [!] {ort_path} 不存在，跳过精度测量\n")

    # ── 延迟 ─────────────────────────────────────────────────────────────────
    print("[1] 延迟测量  (B=4096, 300 次, 专用 stream)")
    t_fp32 = measure_fp32_latency()
    print(f"    FP32 PyTorch          : {t_fp32:7.3f} ms  (1.00×)")

    latency = {}
    for label, (path, in_n, out_n, static) in engines.items():
        if not os.path.exists(path):
            print(f"    {label:<22}: 文件不存在，跳过")
            continue
        t, nlayers = measure_trt_latency(path, in_n, out_n, static)
        latency[label] = t
        print(f"    {label:<22}: {t:7.3f} ms  ({t_fp32/t:.2f}× vs FP32)"
              f"  layers={nlayers}")

    # ── 精度 ─────────────────────────────────────────────────────────────────
    precision = {}
    if has_ort:
        print("\n[2] 精度测量  (vs ORT FP32, 50 batch × B=4096)")
        for label, (path, in_n, out_n, static) in engines.items():
            if not os.path.exists(path):
                continue
            mae, cos = measure_precision(path, in_n, out_n, ort_sess, ort_in, static)
            precision[label] = (mae, cos)
            print(f"    {label:<22}: CosSim={cos:.6f}  MAE={mae:.6f}")

    # ── 汇总表 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  汇总")
    print("=" * 68)
    t_fp16 = latency.get("FP16", None)

    rows = [("FP32 PyTorch", t_fp32, None, None, None)]
    for label in ["FP16", "INT8 未校准", "INT8 LET"]:
        if label not in latency:
            continue
        t = latency[label]
        mae, cos = precision.get(label, (None, None))
        rows.append((label, t, t_fp32/t, t_fp16/t if t_fp16 else None, cos))

    print(f"  {'配置':<22} {'ms':>7} {'vs FP32':>9} {'vs FP16':>9} {'CosSim':>10}")
    print("  " + "─" * 60)
    for name, t, r32, r16, cos in rows:
        r32_s = f"{r32:.2f}×" if r32 else "1.00×"
        r16_s = f"{r16:.2f}×" if r16 else "—"
        cos_s = f"{cos:.6f}"  if cos  else "—"
        print(f"  {name:<22} {t:>7.3f} {r32_s:>9} {r16_s:>9} {cos_s:>10}")


if __name__ == "__main__":
    main()

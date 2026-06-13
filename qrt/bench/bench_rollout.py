#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_rollout.py — 单步推理矩阵: 延迟 + 精度(CosSim vs FP32 eager)
══════════════════════════════════════════════════════════════════════════════
配置(全部同一 state_dict,公平精度对比):
  A  FP32 eager                     基线/精度参照
  B  FP32 + Runner(max-autotune)    与 TRT 协议严格同 dtype 的编译版
  C  FP16 eager
  D  FP16 + Runner(max-autotune)    P0 主形态
  E  FP16 + Runner(manual-graph)    P0 可控形态
  F  FP16 + naive-INT8(6层) + Runner    无 LET 对照
  G  FP16 + LET-INT8(6层) + Runner      P1 主形态

INT8 目标层: actor.0/2/4/6 + actor_proprio_embedding + query_projector
(MHA in_proj 非 nn.Linear 子模块,P1 暂不覆盖,见 PLAN §四)

运行(远程): python3 qrt/bench/bench_rollout.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B, DEVICE, build_model, make_obs, cuda_time, cossim)

import time
import torch
import torch._dynamo as dynamo

from qrt import InferenceRunner, swap, calib

INT8_PATHS = ["actor.0", "actor.2", "actor.4", "actor.6",
              "actor_proprio_embedding", "query_projector"]

results = []   # (name, latency_ms, cossim, note)


def record(name, lat, cos, note=""):
    results.append((name, lat, cos, note))
    print(f"    => {lat:.3f}ms   CosSim={cos:.6f}  {note}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"B={B}  INT8 层: {INT8_PATHS}")

    obs16 = make_obs(torch.float16)
    obs32 = {k: v.float() for k, v in obs16.items()}
    po16, po32 = obs16["policy_obs"], obs32["policy_obs"]

    # ── A. FP32 eager(参照)─────────────────────────────────────────────────
    print("\n[A] FP32 eager")
    m32 = build_model(torch.float32)
    sd = {k: v.clone() for k, v in m32.state_dict().items()}
    with torch.inference_mode():
        t = cuda_time(lambda: m32.act_inference(obs32), n_repeat=50)
        ref = m32.act_inference(obs32)[0].clone().float()
    record("A FP32 eager", t, 1.0)

    # ── B. FP32 + Runner(max-autotune)───────────────────────────────────────
    print("\n[B] FP32 + Runner(max-autotune)")
    r = InferenceRunner(m32, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po32)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po32))
    record("B FP32 compile", t, cossim(r.act(po32), ref))
    del m32, r
    dynamo.reset()
    torch.cuda.empty_cache()

    # ── C. FP16 eager ─────────────────────────────────────────────────────────
    print("\n[C] FP16 eager")
    m = build_model(torch.float16)
    m.load_state_dict(sd)
    with torch.inference_mode():
        t = cuda_time(lambda: m.act_inference(obs16))
        record("C FP16 eager", t, cossim(m.act_inference(obs16)[0], ref))

    # ── D. FP16 + Runner(max-autotune)────────────────────────────────────────
    print("\n[D] FP16 + Runner(max-autotune)")
    r = InferenceRunner(m, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("D FP16 compile", t, cossim(r.act(po16), ref))
    del r
    dynamo.reset()

    # ── E. FP16 + Runner(manual-graph)────────────────────────────────────────
    print("\n[E] FP16 + Runner(manual-graph)")
    m_e = build_model(torch.float16)
    m_e.load_state_dict(sd)
    r = InferenceRunner(m_e, mode="manual-graph")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译+捕获) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("E FP16 manual-graph", t, cossim(r.act(po16).clone(), ref))
    del m_e, r
    dynamo.reset()
    torch.cuda.empty_cache()

    # ── F. naive INT8 对照 ───────────────────────────────────────────────────
    print("\n[F] FP16 + naive-INT8 + Runner(max-autotune)")
    m_f = build_model(torch.float16)
    m_f.load_state_dict(sd)
    plan_naive = calib.make_plan(m_f, INT8_PATHS, obs_fn=lambda: make_obs(torch.float16),
                                 use_let=False)
    swap.apply(m_f, plan_naive)
    r = InferenceRunner(m_f, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("F naive-INT8 compile", t, cossim(r.act(po16), ref))
    del m_f, r
    dynamo.reset()
    torch.cuda.empty_cache()

    # ── G. LET-INT8(P1 主形态)──────────────────────────────────────────────
    print("\n[G] FP16 + LET-INT8 + Runner(max-autotune)")
    m_g = build_model(torch.float16)
    m_g.load_state_dict(sd)
    t0 = time.perf_counter()
    plan_let = calib.make_plan(m_g, INT8_PATHS, obs_fn=lambda: make_obs(torch.float16),
                               use_let=True, let_steps=200)
    print(f"    LET 校准 {time.perf_counter()-t0:.1f}s")
    swapped = swap.apply(m_g, plan_let)
    r = InferenceRunner(m_g, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("G LET-INT8 compile", t, cossim(r.act(po16), ref))

    # ── H. FP16 + QuantCNN(P2: conv 定制 kernel)────────────────────────────
    print("\n[H] FP16 + QuantCNN(Triton INT8 conv)+ Runner(max-autotune)")
    del r
    dynamo.reset()
    torch.cuda.empty_cache()
    from qrt import swap_cnn, calib_cnn_scale
    m_h = build_model(torch.float16)
    m_h.load_state_dict(sd)
    s_cnn = calib_cnn_scale(m_h, lambda: make_obs(torch.float16))
    swap_cnn(m_h, s_cnn)
    r = InferenceRunner(m_h, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("H QuantCNN compile", t, cossim(r.act(po16), ref))

    # ── I. 全家桶: LET-INT8 linear + QuantCNN ────────────────────────────────
    print("\n[I] FP16 + LET-INT8(6层) + QuantCNN + Runner(max-autotune)")
    del r
    dynamo.reset()
    torch.cuda.empty_cache()
    m_i = build_model(torch.float16)
    m_i.load_state_dict(sd)
    plan_i = calib.make_plan(m_i, INT8_PATHS, obs_fn=lambda: make_obs(torch.float16),
                             use_let=True, let_steps=200, verbose=False)
    s_cnn_i = calib_cnn_scale(m_i, lambda: make_obs(torch.float16))
    swap.apply(m_i, plan_i)
    swap_cnn(m_i, s_cnn_i)
    r = InferenceRunner(m_i, mode="max-autotune")
    t0 = time.perf_counter()
    r.warmup(po16)
    print(f"    warmup(编译) {time.perf_counter()-t0:.1f}s")
    t = cuda_time(lambda: r.act(po16))
    record("I LET+QuantCNN compile", t, cossim(r.act(po16), ref))
    t_refresh = cuda_time(lambda: r.refresh_quant(), n_warmup=5, n_repeat=50)
    print(f"    refresh(6 Linear + QuantCNN, CUDA Graph): {t_refresh:.3f}ms")
    # 热更新冒烟(I 全家桶: conv 权重扰动 → refresh → 输出跟随)
    with torch.no_grad():
        m_i.map_cnn._modules["3"].weight.mul_(1.02)
    r.refresh_quant()
    o1 = r.act(po16).clone()
    drift = (o1 - r.act(po16)).abs().max().item()
    print(f"    conv 权重热更新后两次调用一致性 diff={drift:.2e}(应=0)")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    base = results[0][1]
    print("\n" + "=" * 76)
    print(f"{'配置':<26} {'单步':>9} {'×FP32':>7} {'CosSim(vs FP32)':>16}")
    print("-" * 76)
    for name, lat, cos, note in results:
        print(f"{name:<26} {lat:>8.3f}ms {base/lat:>6.2f}x {cos:>16.6f}")
    print("-" * 76)
    print("参考(同卡历史): TRT FP16=6.298ms  TRT INT8最优=3.274ms")
    print("=" * 76)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_matrix.py — 全版本统一横评(延迟 + 精度,不含仿真)
══════════════════════════════════════════════════════════════════════════════
单一进程、单一 GPU 状态、同一 state_dict、GLAD 主配置(gc+topk)、B=4096。
逐档叠加,看每层技术的边际贡献。精度 = 动作 mean 对 FP32 eager 的 CosSim
(随机权重 → 反映各档引入的数值保真;真实收敛权重的精度见 P1/P5/P6 表)。

Part 1  单步推理延迟矩阵(纯 act 路径,各档 vs FP32):
  A FP32 eager           基线
  B FP16 eager
  C FP16 compile         P0(全图融合)
  D + encoder 重写       P3
  E + QuantCNN(INT8 conv) P2
  F + LET-INT8 linear     P1/P5(动态 s_z,推理完全体)
  G naive-INT8 对照(无 LET)

Part 2  真实 rsl_rl 协议端到端 per-iter(扣除 mock env;另脚本 bench_real_loop
  已测 base/full,此处汇总引用 + 标注口径)。

运行(远程,空闲 GPU): python3 qrt/bench/bench_matrix.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import B, DEVICE, build_model, make_obs, make_terrain_obs, cuda_time, cossim

import time
import torch
import torch._dynamo as dynamo

from qrt import swap, calib, swap_cnn, calib_cnn_scale, patch_encoder

rows = []   # (tag, lat_ms, cos, note)


def bench_one(tag, build_fn, ref_mean, po, note="", n_rep=100):
    """build_fn() → callable act(po);测延迟 + CosSim vs ref_mean。"""
    dynamo.reset()
    torch.cuda.empty_cache()
    act, t_compile = build_fn()
    with torch.inference_mode():
        for _ in range(3):
            act(po)
        torch.cuda.synchronize()
        lat = cuda_time(lambda: act(po), n_warmup=15, n_repeat=n_rep)
        out = act(po).float()
    cos = cossim(out, ref_mean) if ref_mean is not None else 1.0
    rows.append((tag, lat, cos, note))
    print(f"  {tag:<28} {lat:7.3f}ms   CosSim={cos:.6f}   编译{t_compile:4.1f}s {note}")
    return out


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}  B={B}")
    print("GLAD 主配置(gc+topk),结构化地形 obs,逐档叠加\n")

    obs = make_terrain_obs(torch.float16)
    po16 = obs["policy_obs"]
    po32 = po16.float()

    # 共享权重起点
    m32 = build_model(torch.float32)
    sd = {k: v.clone() for k, v in m32.state_dict().items()}

    INT8 = ["actor.0", "actor.2", "actor.4", "actor.6",
            "actor_proprio_embedding", "query_projector"]

    print("Part 1  单步推理延迟矩阵")
    print("-" * 78)

    # A. FP32 eager(参照)
    def build_A():
        m = build_model(torch.float32); m.load_state_dict(sd)
        return (lambda p: m.act_inference({"policy_obs": p})[0]), 0.0
    with torch.inference_mode():
        mA = build_model(torch.float32); mA.load_state_dict(sd)
        ref = mA.act_inference({"policy_obs": po32})[0].float().clone()
        t = cuda_time(lambda: mA.act_inference({"policy_obs": po32}), n_repeat=50)
    rows.append(("A FP32 eager", t, 1.0, "参照"))
    print(f"  {'A FP32 eager':<28} {t:7.3f}ms   CosSim=1.000000   (参照)")
    del mA

    # B. FP16 eager
    def build_B():
        m = build_model(torch.float16); m.load_state_dict(sd)
        return (lambda p: m.act_inference({"policy_obs": p})[0]), 0.0
    bench_B = build_B()
    with torch.inference_mode():
        mB = build_model(torch.float16); mB.load_state_dict(sd)
        t = cuda_time(lambda: mB.act_inference({"policy_obs": po16}))
        cB = cossim(mB.act_inference({"policy_obs": po16})[0], ref)
    rows.append(("B FP16 eager", t, cB, ""))
    print(f"  {'B FP16 eager':<28} {t:7.3f}ms   CosSim={cB:.6f}")
    del mB

    # C. FP16 compile
    def build_C():
        m = build_model(torch.float16); m.load_state_dict(sd)
        def f(p): return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3): cf(po16)
            torch.cuda.synchronize()
        return cf, time.perf_counter() - t0
    bench_one("C FP16 compile (P0)", build_C, ref, po16)

    # D. + encoder 重写
    def build_D():
        m = build_model(torch.float16); m.load_state_dict(sd); patch_encoder(m)
        def f(p): return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3): cf(po16)
            torch.cuda.synchronize()
        return cf, time.perf_counter() - t0
    bench_one("D + encoder 重写 (P3)", build_D, ref, po16)

    # E. + QuantCNN
    def build_E():
        m = build_model(torch.float16); m.load_state_dict(sd)
        s = calib_cnn_scale(m, lambda: make_obs(torch.float16))
        swap_cnn(m, s); patch_encoder(m)
        def f(p): return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3): cf(po16)
            torch.cuda.synchronize()
        return cf, time.perf_counter() - t0
    bench_one("E + QuantCNN (P2)", build_E, ref, po16)

    # F. + LET-INT8 linear(推理完全体)
    def build_F():
        m = build_model(torch.float16); m.load_state_dict(sd)
        plan = calib.make_plan(m, INT8, obs_fn=lambda: make_obs(torch.float16),
                               use_let=True, let_steps=100, act_mode="dynamic",
                               verbose=False)
        swap.apply(m, plan)
        s = calib_cnn_scale(m, lambda: make_obs(torch.float16))
        swap_cnn(m, s); patch_encoder(m)
        def f(p): return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3): cf(po16)
            torch.cuda.synchronize()
        return cf, time.perf_counter() - t0
    bench_one("F + LET-INT8 (完全体)", build_F, ref, po16, "P1/P5")

    # G. naive-INT8 对照
    def build_G():
        m = build_model(torch.float16); m.load_state_dict(sd)
        plan = calib.make_plan(m, INT8, obs_fn=lambda: make_obs(torch.float16),
                               use_let=False, act_mode="dynamic", verbose=False)
        swap.apply(m, plan)
        s = calib_cnn_scale(m, lambda: make_obs(torch.float16))
        swap_cnn(m, s); patch_encoder(m)
        def f(p): return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3): cf(po16)
            torch.cuda.synchronize()
        return cf, time.perf_counter() - t0
    bench_one("G naive-INT8 对照", build_G, ref, po16, "无 LET")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    base = rows[0][1]
    print("\n" + "=" * 78)
    print(f"{'配置':<28} {'单步':>9} {'×FP32':>7} {'×FP16e':>7} {'CosSim':>10}")
    print("-" * 78)
    fp16e = rows[1][1]
    for tag, lat, cos, note in rows:
        print(f"{tag:<28} {lat:>7.3f}ms {base/lat:>6.2f}x {fp16e/lat:>6.2f}x {cos:>10.6f}")
    print("=" * 78)
    print("注: CosSim 为随机权重下的数值保真;真实收敛权重精度见 P1/P5/P6")
    print("    (LET 全 5 层 0.99977 > naive 0.99948;TRT 同卡 INT8 最优 3.274ms)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_int8linear_net.py — INT8 linear 净效果隔离 + quant/dequant 融合诊断
══════════════════════════════════════════════════════════════════════════════
最干净对照(空闲 GPU,B=4096,GLAD 主配置,只改"linear 是否 INT8"一个变量,
不带 QuantCNN/encoder 重写,排除一切混淆):
  X  FP16 compile                  (纯基线)
  Y  FP16 compile + LET-INT8(6 层) (dynamic s_z)
延迟 + kernel 级分解(看 quant round/clamp、dequant mul 是否独立成 kernel,
还是被 inductor 融进 _int_mm 模板的 epilogue / 相邻 elementwise)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import B, build_model, make_obs, make_terrain_obs, cuda_time, cossim

import time
import torch
import torch._dynamo as dynamo
from torch.profiler import profile, ProfilerActivity

from qrt import swap, calib

INT8 = ["actor.0", "actor.2", "actor.4", "actor.6",
        "actor_proprio_embedding", "query_projector"]


def build_compiled(m, po):
    def f(p):
        return m.act_inference({"policy_obs": p})[0]
    cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            cf(po)
        torch.cuda.synchronize()
    return cf, time.perf_counter() - t0


def kernel_breakdown(cf, po, tag):
    with torch.inference_mode():
        for _ in range(10):
            cf(po)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                cf(po)
            torch.cuda.synchronize()
    evs = [(ev.self_device_time_total / 20, ev.key) for ev in prof.key_averages()
           if ev.self_device_time_total > 0]
    evs.sort(reverse=True)
    tot = sum(t for t, _ in evs)
    # 标记疑似 quant/dequant 独立 kernel(round/clamp/mul/convert,且非 _int_mm/_mm)
    print(f"\n[{tag}] 单步 GPU 合计 ~{tot/1000:.3f}ms,top kernel:")
    int8_k, quant_k = 0.0, 0.0
    for t, k in evs[:16]:
        kl = k.lower()
        mark = ""
        if "_int_mm" in kl or "s8s8" in kl or "imma" in kl or "igemm" in kl:
            mark = " ←INT8 GEMM"; int8_k += t
        elif any(w in kl for w in ("round", "clamp", "convert", "_to_copy")) and "mm" not in kl:
            mark = " ←疑似独立 quant/cvt"; quant_k += t
        print(f"   {t/1000:7.4f}ms  {k[:80]}{mark}")
    return tot / 1000


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}  B={B}")
    obs = make_terrain_obs(torch.float16)
    po16 = obs["policy_obs"]
    po32 = po16.float()

    m32 = build_model(torch.float32)
    sd = {k: v.clone() for k, v in m32.state_dict().items()}
    with torch.inference_mode():
        ref = m32.act_inference({"policy_obs": po32})[0].float().clone()
    del m32

    # ── X: FP16 compile ──────────────────────────────────────────────────────
    dynamo.reset(); torch.cuda.empty_cache()
    mX = build_model(torch.float16); mX.load_state_dict(sd)
    cfX, tcX = build_compiled(mX, po16)
    with torch.inference_mode():
        latX = cuda_time(lambda: cfX(po16), n_warmup=20, n_repeat=200)
        cosX = cossim(cfX(po16), ref)
    print(f"\nX FP16 compile        : {latX:.4f}ms   CosSim={cosX:.6f}  (编译 {tcX:.0f}s)")

    # ── Y: FP16 compile + LET-INT8 ───────────────────────────────────────────
    dynamo.reset(); torch.cuda.empty_cache()
    mY = build_model(torch.float16); mY.load_state_dict(sd)
    plan = calib.make_plan(mY, INT8, obs_fn=lambda: make_obs(torch.float16),
                           use_let=True, let_steps=100, act_mode="dynamic", verbose=False)
    swap.apply(mY, plan)
    cfY, tcY = build_compiled(mY, po16)
    with torch.inference_mode():
        latY = cuda_time(lambda: cfY(po16), n_warmup=20, n_repeat=200)
        cosY = cossim(cfY(po16), ref)
    print(f"Y FP16 + LET-INT8(6)  : {latY:.4f}ms   CosSim={cosY:.6f}  (编译 {tcY:.0f}s)")

    print(f"\n净效果 (Y − X) = {latY - latX:+.4f}ms  "
          f"({'INT8 linear 变慢' if latY > latX else 'INT8 linear 变快'})")

    # ── kernel 级分解(看 quant/dequant 融合质量)─────────────────────────────
    print("\n" + "=" * 70)
    print("kernel 级分解(quant/dequant 是否独立成 kernel?)")
    print("=" * 70)
    kernel_breakdown(cfX, po16, "X FP16")
    kernel_breakdown(cfY, po16, "Y INT8")


if __name__ == "__main__":
    main()

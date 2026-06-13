#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时检查: QuantLinear INT8 路径在编译图内是否真实生效。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_model, make_obs

import torch
from torch.profiler import profile, ProfilerActivity

from qrt import InferenceRunner, swap, calib

m = build_model(torch.float16)
obs = make_obs(torch.float16)
plan = calib.make_plan(m, ["actor.0", "actor.2"],
                       obs_fn=lambda: make_obs(torch.float16),
                       use_let=True, let_steps=50, verbose=False)
swapped = swap.apply(m, plan)
ql = swapped["actor.0"]
x = torch.randn(4096, 224, device="cuda", dtype=torch.float16)
with torch.inference_mode():
    y_int8 = ql(x)
    y_fp = torch.nn.functional.linear(x, ql.weight, ql.bias)
d = (y_int8.float() - y_fp.float()).abs()
rel = d.mean().item() / y_fp.float().abs().mean().item()
print(f"QuantLinear eval vs FP16: max|diff|={d.max().item():.4f}  "
      f"mean={d.mean().item():.5f}  rel={rel:.2e}  (应>0,量化误差存在)")
print(f"w8 buffer 非零占比: {(ql.w8 != 0).float().mean().item():.3f}")

r = InferenceRunner(m, mode="max-autotune")
r.warmup(obs["policy_obs"])
with torch.inference_mode():
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            r.act(obs["policy_obs"])
        torch.cuda.synchronize()

print("图内 INT8 相关 kernel:")
found = False
for ev in prof.key_averages():
    n = ev.key.lower()
    if any(k in n for k in ("int8", "i8", "imma", "s8", "_int_mm", "igemm")) \
            and ev.self_device_time_total > 0:
        print(f"   {ev.key[:100]}   {ev.self_device_time_total/10:.1f}us/步")
        found = True
if not found:
    print("   (未发现 —— INT8 可能未生效,需排查!)")

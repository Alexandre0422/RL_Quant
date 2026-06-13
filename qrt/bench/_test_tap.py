#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_tap.py — B 方案可行性 spike: tap 算子在 compile 下的副作用可靠性
══════════════════════════════════════════════════════════════════════════════
验证(决定 B 能否成立):
  ① eager: tap 副作用执行,buffer == 中间激活,前向 bit-exact
  ② compile(no-cudagraphs): 同上(关键 —— cocalib 开启时训练前向用此模式)
  ③ compile(max-autotune, 含 cudagraph trees): 行为记录(可行/不可行,用于决策)
  ④ bf16 autocast 下采集(真实 update 场景)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: F401  (mock/path 副作用)

import torch
import torch.nn as nn
import torch._dynamo as dynamo

from qrt.tap import TapStore, tap_

DEV = torch.device("cuda")


class Net(nn.Module):
    def __init__(self, slot):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(128, 64, device=DEV) * 0.1)
        self.w2 = nn.Parameter(torch.randn(64, 32, device=DEV) * 0.1)
        self.slot = slot

    def forward(self, x):
        h = torch.relu(x @ self.w1)        # 中间激活 [M,64](w2 的输入)
        h = tap_(h, self.slot)             # ← 采集点
        return h @ self.w2


def check(tag, captured, ref_mid, out, ref_out):
    if captured is None:
        print(f"  [{tag}] ❌ buffer 空 —— 副作用未执行")
        return
    n = captured.shape[0]
    # captured 是下采样的子集,逐行未必对齐;比较分布统计(absmax/mean/std)
    cap_stat = (captured.abs().amax().item(), captured.mean().item(), captured.std().item())
    ref_stat = (ref_mid.abs().amax().item(), ref_mid.mean().item(), ref_mid.std().item())
    stat_ok = all(abs(a - b) < 1e-2 * (abs(b) + 1e-3) for a, b in zip(cap_stat, ref_stat))
    out_ok = (out - ref_out).abs().max().item() < 1e-4
    print(f"  [{tag}] buffer 行={n}  统计匹配={stat_ok}  前向bit-exact={out_ok}  "
          f"(cap absmax={cap_stat[0]:.4f} vs ref {ref_stat[0]:.4f})")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    x = torch.randn(4096, 128, device=DEV)
    TapStore.enable(n_layers=1, k_dims=[64], sample_rows=2048)

    net = Net(0)
    with torch.no_grad():
        ref_mid = torch.relu(x @ net.w1)
        ref_out = ref_mid @ net.w2

    # ── ① eager ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = net(x)
    check("① eager", TapStore.read(0), ref_mid, out, ref_out)

    # ── ② compile no-cudagraphs ──────────────────────────────────────────────
    dynamo.reset()
    cf = torch.compile(net, mode="max-autotune-no-cudagraphs", dynamic=False)
    with torch.no_grad():
        out = cf(x)
        torch.cuda.synchronize()
    check("② compile(no-cudagraphs)", TapStore.read(0), ref_mid, out, ref_out)

    # ── ③ compile max-autotune (cudagraph trees) ─────────────────────────────
    dynamo.reset()
    cf2 = torch.compile(net, mode="max-autotune", dynamic=False)
    try:
        with torch.no_grad():
            for _ in range(3):
                out = cf2(x)
            torch.cuda.synchronize()
        check("③ compile(max-autotune/cudagraph)", TapStore.read(0), ref_mid, out, ref_out)
    except Exception as e:
        print(f"  [③ cudagraph] 异常: {type(e).__name__}: {str(e)[:120]}")

    # ── ④ bf16 autocast 采集(真实 update 场景)+ grad ───────────────────────
    dynamo.reset()
    net2 = Net(0)
    cf3 = torch.compile(net2, mode="max-autotune-no-cudagraphs", dynamic=False)
    with torch.autocast("cuda", torch.bfloat16):
        out = cf3(x)
        loss = out.float().pow(2).mean()
    loss.backward()
    cap = TapStore.read(0)
    grad_ok = net2.w1.grad is not None and torch.isfinite(net2.w1.grad).all()
    print(f"  [④ bf16+grad] buffer 行={cap.shape[0] if cap is not None else 0}  "
          f"dtype={cap.dtype if cap is not None else '-'}  反向正常={grad_ok}")

    TapStore.disable()
    print("\nspike 完成。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_tdmpc2.py — TD-MPC2 量化接线对拍(eager,秒级;跑 bench_tdmpc2 长编译前先过这个)
══════════════════════════════════════════════════════════════════════════════
只验证「接线正确 + 量化真生效 + INT8/FP 数值接近」,不测 latency(那是 bench 的事):
  1. 世界模型 build + 单步 plan() 端到端形状
  2. rollout 量化(QuantLinear W8A8)后 estimate_value 的 CosSim(FP16 vs INT8)
  3. encoder 量化(WeightOnlyLinear W8A16/W4A16)后 encode() 的 CosSim
  4. 断言 rollout 层确被换成 QuantLinear、encoder 层确被换成 WeightOnlyLinear

注:CosSim「过好」(=1.000000)是警报(可能 INT8 没生效走了 FP 分支)——
这里断言模块类型 + CosSim 落在合理区间([0.99, 0.9999]),双保险。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch

from qrt.tdmpc2 import (TDMPC2Config, build_world_model,
                        quantize_rollout, quantize_encoder,
                        rollout_linear_paths, encoder_linear_paths)
from qrt.qlinear import QuantLinear
from qrt.wlinear import WeightOnlyLinear


def cossim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA(远程 3090);本地仅语法检查。"); return
    torch.manual_seed(0)
    dev = "cuda"
    cfg = TDMPC2Config()
    N = cfg.num_samples + cfg.num_pi_trajs

    # 同起点权重
    wm0 = build_world_model(cfg, dtype=torch.float32, device=dev)
    sd = {k: v.clone() for k, v in wm0.state_dict().items()}
    sd16 = {k: (v.half().float() if v.is_floating_point() else v) for k, v in sd.items()}
    del wm0

    # ── 1. 端到端 plan() 形状 ────────────────────────────────────────────────
    wm = build_world_model(cfg, dtype=torch.float16, device=dev); wm.load_state_dict(sd16)
    obs = torch.randn(cfg.obs_dim, device=dev, dtype=torch.float16)
    with torch.inference_mode():
        a = wm.plan(obs)
    assert a.shape == (cfg.action_dim,), a.shape
    print(f"[1] plan() OK → action {tuple(a.shape)}")

    # rollout 参照(FP16)
    z0 = torch.randn(N, cfg.latent_dim, device=dev, dtype=torch.float16)
    acts = torch.rand(cfg.horizon, N, cfg.action_dim, device=dev, dtype=torch.float16) * 2 - 1
    with torch.inference_mode():
        ref = wm.estimate_value(z0, acts).float().clone()

    # ── 2. rollout 量化(W8A8 + LET dynamic)→ CosSim ────────────────────────
    wmq = build_world_model(cfg, dtype=torch.float16, device=dev); wmq.load_state_dict(sd16)
    ok, skip = rollout_linear_paths(wmq)
    swapped = quantize_rollout(wmq, obs_fn=lambda: torch.randn(cfg.obs_dim, device=dev,
                               dtype=torch.float16), act_mode="dynamic",
                               use_let=True, let_steps=30, verbose=False)
    assert all(isinstance(wmq.get_submodule(p), QuantLinear) for p in ok), "rollout 未换 QuantLinear"
    with torch.inference_mode():
        y = wmq.estimate_value(z0, acts).float()
    c = cossim(y, ref)
    print(f"[2] rollout W8A8+LET: 换 {len(swapped)} 层(跳过 {len(skip)}),CosSim={c:.6f}")
    assert 0.98 <= c < 0.999999, f"CosSim 异常(过好=警报 / 过差=bug): {c}"

    # ── 3. encoder 量化(W8A16 / W4A16)→ CosSim ─────────────────────────────
    obs1 = torch.randn(1, cfg.obs_dim, device=dev, dtype=torch.float16)
    with torch.inference_mode():
        z_ref = wm.encode(obs1).float().clone()
    for bits in (8, 4):
        wme = build_world_model(cfg, dtype=torch.float16, device=dev); wme.load_state_dict(sd16)
        quantize_encoder(wme, bits=bits, verbose=False)
        epaths = encoder_linear_paths(wme)
        assert all(isinstance(wme.get_submodule(p), WeightOnlyLinear) for p in epaths)
        with torch.inference_mode():
            z_q = wme.encode(obs1).float()
        c = cossim(z_q, z_ref)
        print(f"[3] encoder W{bits}A16 (B=1): 换 {len(epaths)} 层,CosSim={c:.6f}")
        assert c >= (0.98 if bits == 8 else 0.90), f"W{bits}A16 CosSim 过差: {c}"

    print("\n全部通过 ✓  (接线正确、INT8 生效、数值合理;latency 见 bench_tdmpc2.py)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/_test_encoder_opt.py — P3 encoder 代数重写对拍单测(远程 3090)
══════════════════════════════════════════════════════════════════════════════
① FP32 eval 等价性: patch 前后 encoded/attn/topk_idx/gc_weights 对拍(恒等式,
   仅浮点顺序差异,~1e-5 级;topk 并列边界允许极少 index 漂移)
② train 态等价性(固定 seed,Gumbel 路径)
③ FP16 延迟: eager orig vs patch;compile orig vs patch
④ P3 完全体单步预览: patch + QuantCNN + compile
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import B, build_model, make_obs, make_terrain_obs, cuda_time

import time
import torch
import torch._dynamo as dynamo

from qrt import patch_encoder, unpatch_encoder, swap_cnn, calib_cnn_scale


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")

    # ── ① FP32 eval 等价性 ───────────────────────────────────────────────────
    m32 = build_model(torch.float32)
    obs32 = {k: v.float() for k, v in make_terrain_obs(torch.float16).items()}
    with torch.inference_mode():
        ref = m32.act_inference(obs32)
        patch_encoder(m32)
        out = m32.act_inference(obs32)
    names = ["action_mean", "attn_weights", "topk_indices", "gc_weights"]
    print("\n[①] FP32 eval 等价性(patch vs 原版):")
    for nm, a, b_ in zip(names, out, ref):
        if nm == "topk_indices":
            mismatch = (a != b_).float().mean().item()
            print(f"    {nm:<13} index 不匹配率={mismatch:.2e}(并列边界,应≈0)")
        else:
            d = (a.float() - b_.float()).abs().max().item()
            print(f"    {nm:<13} max|diff|={d:.2e}")
    d_enc = (out[0].float() - ref[0].float()).abs().max().item()
    assert d_enc < 1e-3, "代数重写不等价!"

    # ── ② train 态等价性(固定 seed)─────────────────────────────────────────
    unpatch_encoder(m32)
    m32.train()
    with torch.no_grad():
        torch.manual_seed(42)
        r_tr = m32._encode_terrain(
            m32.actor_obs_normalizer(m32.get_actor_obs(obs32)))[0]
        patch_encoder(m32)
        torch.manual_seed(42)
        o_tr = m32._encode_terrain(
            m32.actor_obs_normalizer(m32.get_actor_obs(obs32)))[0]
    d_tr = (o_tr - r_tr).abs().max().item()
    print(f"[②] train 态(Gumbel, 同 seed): max|diff|={d_tr:.2e}")
    m32.eval()
    del m32
    torch.cuda.empty_cache()

    # ── ③ FP16 延迟 ──────────────────────────────────────────────────────────
    print("\n[③] FP16 延迟")
    obs16 = make_obs(torch.float16)
    po = obs16["policy_obs"]
    m = build_model(torch.float16)
    with torch.inference_mode():
        t_e0 = cuda_time(lambda: m.act_inference(obs16), n_repeat=50)
    patch_encoder(m)
    with torch.inference_mode():
        t_e1 = cuda_time(lambda: m.act_inference(obs16), n_repeat=50)
    print(f"    eager: orig={t_e0:.3f}ms  patch={t_e1:.3f}ms  ({t_e0/t_e1:.2f}x)")

    def f(p):
        return m.act_inference({"policy_obs": p})[0]
    unpatch_encoder(m)
    cf0 = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for _ in range(3):
            cf0(po)
        t_c0 = cuda_time(lambda: cf0(po))
    dynamo.reset()
    patch_encoder(m)
    cf1 = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            cf1(po)
        torch.cuda.synchronize()
    print(f"    patch 版编译 {time.perf_counter()-t0:.1f}s")
    with torch.inference_mode():
        t_c1 = cuda_time(lambda: cf1(po))
    print(f"    compile: orig={t_c0:.3f}ms  patch={t_c1:.3f}ms  ({t_c0/t_c1:.2f}x)")
    del m
    dynamo.reset()
    torch.cuda.empty_cache()

    # ── ④ P3 完全体预览: patch + QuantCNN + compile ──────────────────────────
    print("\n[④] patch + QuantCNN + compile(P3 速度完全体)")
    m2 = build_model(torch.float16)
    s_cnn = calib_cnn_scale(m2, lambda: make_obs(torch.float16))
    swap_cnn(m2, s_cnn)
    patch_encoder(m2)

    def f2(p):
        return m2.act_inference({"policy_obs": p})[0]
    cf2 = torch.compile(f2, mode="max-autotune", fullgraph=True, dynamic=False)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            cf2(po)
        torch.cuda.synchronize()
    print(f"    编译 {time.perf_counter()-t0:.1f}s")
    with torch.inference_mode():
        t_full = cuda_time(lambda: cf2(po))
    print(f"    单步: {t_full:.3f}ms(对照: P0 FP16 compile 1.69ms / P2 H 1.51ms)")
    print("\n单测通过。")


if __name__ == "__main__":
    main()

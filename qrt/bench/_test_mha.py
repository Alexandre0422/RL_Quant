#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_mha.py — MHA 单独剖析:nn.MHA 原版 vs P3 手写,延迟+精度+kernel 分解
══════════════════════════════════════════════════════════════════════════════
GLAD 的 MHA: single-query cross-attention, D=64, H=16, head_dim=4, K=32, B=4096.
nn.MultiheadAttention 在 need_weights=True 时走 Python 分解慢路径(GLAD 需 attn
权重)。P3 改为紧凑张量手写。本脚本隔离测两者:
  Part A  纯 MHA 子模块延迟(eager + compile),数值等价
  Part B  整模型 profiler 里 MHA 相关 kernel 的占比(原版 vs 重写)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import B, DEVICE, build_model, make_terrain_obs, cuda_time, cossim

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch.profiler import profile, ProfilerActivity

D, H, K = 64, 16, 32
hd = D // H


def mha_ref(mha, q1, kv):
    """nn.MHA 原版调用(need_weights=True,GLAD 用法)。"""
    out, w = mha(query=q1, key=kv, value=kv)        # q1:[B,1,D] kv:[B,K,D]
    return out.squeeze(1), w


def mha_rewrite(mha, query_vec, local_sparse):
    """P3 手写(encoder_opt 同款)。"""
    Bn = query_vec.shape[0]
    W_in, b_in = mha.in_proj_weight, mha.in_proj_bias
    q = query_vec @ W_in[:D].t() + b_in[:D]
    k = local_sparse @ W_in[D:2 * D].t() + b_in[D:2 * D]
    v = local_sparse @ W_in[2 * D:].t() + b_in[2 * D:]
    qh = q.view(Bn, H, 1, hd)
    kh = k.view(Bn, K, H, hd).transpose(1, 2)
    vh = v.view(Bn, K, H, hd).transpose(1, 2)
    scores = (qh @ kh.transpose(-1, -2)) * (1.0 / hd ** 0.5)
    attn = F.softmax(scores, dim=-1)
    oh = attn @ vh
    out = oh.reshape(Bn, D) @ mha.out_proj.weight.t() + mha.out_proj.bias
    return out, attn.squeeze(2).mean(dim=1, keepdim=True)


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  B={B}  D={D} H={H} head_dim={hd} K={K}")

    mha = nn.MultiheadAttention(D, H, batch_first=True).to(DEVICE).half().eval()
    query_vec = torch.randn(B, D, device=DEVICE, dtype=torch.float16)
    local_sparse = torch.randn(B, K, D, device=DEVICE, dtype=torch.float16)
    q1 = query_vec.unsqueeze(1)

    # ── Part A: 数值等价 + 子模块延迟 ────────────────────────────────────────
    print("\n" + "=" * 64 + "\nPart A  MHA 子模块(B=4096)\n" + "=" * 64)
    with torch.inference_mode():
        o_ref, w_ref = mha_ref(mha, q1, local_sparse)
        o_rw, w_rw = mha_rewrite(mha, query_vec, local_sparse)
    print(f"数值等价: out CosSim={cossim(o_rw, o_ref):.6f}  "
          f"max|diff|={(o_rw - o_ref).abs().max().item():.2e}  "
          f"attn CosSim={cossim(w_rw, w_ref):.6f}")

    with torch.inference_mode():
        t_ref = cuda_time(lambda: mha_ref(mha, q1, local_sparse), n_warmup=20, n_repeat=200)
        t_rw = cuda_time(lambda: mha_rewrite(mha, query_vec, local_sparse),
                         n_warmup=20, n_repeat=200)
    print(f"eager 延迟: nn.MHA 原版={t_ref:.4f}ms   P3 手写={t_rw:.4f}ms   "
          f"({t_ref/t_rw:.2f}×)")

    # compile 各自
    dynamo.reset()
    cf_ref = torch.compile(lambda: mha_ref(mha, q1, local_sparse),
                           mode="max-autotune", dynamic=False)
    cf_rw = torch.compile(lambda: mha_rewrite(mha, query_vec, local_sparse),
                          mode="max-autotune", dynamic=False)
    with torch.inference_mode():
        for _ in range(3):
            cf_ref(); cf_rw()
        torch.cuda.synchronize()
        t_ref_c = cuda_time(lambda: cf_ref(), n_warmup=20, n_repeat=200)
        t_rw_c = cuda_time(lambda: cf_rw(), n_warmup=20, n_repeat=200)
    print(f"compile 延迟: nn.MHA 原版={t_ref_c:.4f}ms   P3 手写={t_rw_c:.4f}ms   "
          f"({t_ref_c/t_rw_c:.2f}×)")

    # ── Part B: 整模型 profiler 中 MHA kernel 占比 ───────────────────────────
    print("\n" + "=" * 64 + "\nPart B  整模型 FP16 eager 中 MHA 相关 kernel\n" + "=" * 64)
    m = build_model(torch.float16)
    obs = make_terrain_obs(torch.float16)
    with torch.inference_mode():
        for _ in range(10):
            m.act_inference(obs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                m.act_inference(obs)
            torch.cuda.synchronize()
    mha_ops = ("bmm", "baddbmm", "softmax", "_scaled_dot", "addmm")  # MHA 痕迹粗筛
    print("  含 bmm/softmax/addmm 的 kernel(MHA 内部 Q·Kᵀ / attn·V / proj):")
    tot_mha = 0.0
    for ev in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:30]:
        kl = ev.key.lower()
        if ("bmm" in kl or "baddbmm" in kl) and ev.self_device_time_total > 0:
            t = ev.self_device_time_total / 20
            tot_mha += t
            print(f"    {t/1000:7.4f}ms  {ev.key[:78]}")
    print(f"  bmm 类合计 ≈ {tot_mha/1000:.4f}ms/步(MHA 注意力核心,attn 重写主要省这里)")


if __name__ == "__main__":
    main()

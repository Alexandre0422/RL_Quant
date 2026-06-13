#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/_test_qconv.py — P2 QuantCNN 对拍单测(远程 3090)
══════════════════════════════════════════════════════════════════════════════
① kernel1 索引/边界正确性: z8·s_z vs BN1(relu(conv1(x))) fp32(误差应≈量化步)
② 整链精度: QuantCNN vs 原 map_cnn fp32(CosSim / max|diff| / rel)
③ 延迟: fp16 eager map_cnn vs QuantCNN(两 kernel)
④ compile 冒烟: 数值一致 + 不报错
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import B, DEVICE, build_model, make_terrain_obs, cuda_time

import torch
import torch._dynamo as dynamo

from qrt.qconv import QuantCNN, swap_cnn, calib_cnn_scale

L_, W_ = 33, 21


def get_height_map(obs):
    po = obs["policy_obs"]
    ms = po[:, -L_ * W_ * 3:].reshape(-1, W_, L_, 3)
    return ms.permute(0, 3, 1, 2)        # [B,3,21,33]


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    import triton
    print(f"Triton: {triton.__version__}")

    # FP32 参照模型 + FP16 工作模型(同权重)
    m32 = build_model(torch.float32)
    sd = m32.state_dict()
    m16 = build_model(torch.float16)
    m16.load_state_dict(sd)

    obs16 = make_terrain_obs(torch.float16)
    x16 = get_height_map(obs16).contiguous()
    x32 = x16.float()

    with torch.inference_mode():
        ref_full = m32.map_cnn(x32)                                  # [B,64,11,17]
        conv1_, bn1_ = m32.map_cnn[0], m32.map_cnn[2]
        ref_z = bn1_(torch.relu(conv1_(x32)))                        # [B,16,11,17]

    # 校准 + swap
    s_z = calib_cnn_scale(m16, lambda: make_terrain_obs(torch.float16), n_batches=8)
    print(f"\ns_z(conv2 输入 per-tensor)= {s_z:.5f}")
    qcnn = swap_cnn(m16, s_z)
    print(qcnn)

    with torch.inference_mode():
        # ── ① kernel1 对拍 ───────────────────────────────────────────────────
        x_nhwc = x16.permute(0, 2, 3, 1).contiguous()
        z8 = torch.ops.qrt.conv1_fused_q(x_nhwc, qcnn.w1p, qcnn.b1f, qcnn.qa, qcnn.qc)
        z_deq = z8.float().reshape(B, 11, 17, 16).permute(0, 3, 1, 2) * s_z
        d1 = (z_deq - ref_z).abs()
        # clamp 区域(|ref|>127·s_z)以外的误差应 ≤ ~1 量化步 + fp16 舍入
        in_range = ref_z.abs() < (126.0 * s_z)
        err_in = d1[in_range].max().item() if in_range.any() else 0.0
        print(f"\n[①] kernel1: max|err|(范围内)={err_in:.5f}  "
              f"(≈量化步 s_z={s_z:.5f} 的 {err_in/s_z:.2f} 倍,应≲1.5)")
        assert err_in < 2.0 * s_z + 1e-3, "kernel1 索引/边界可能有错!"

        # ── ② 整链对拍 ──────────────────────────────────────────────────────
        out_q = qcnn(x16).float()
        d2 = (out_q - ref_full).abs()
        cos = torch.nn.functional.cosine_similarity(
            out_q.flatten(), ref_full.flatten(), dim=0).item()
        rel = (d2.pow(2).mean().sqrt() / ref_full.pow(2).mean().sqrt()).item()
        print(f"[②] 整链: CosSim={cos:.6f}  max|diff|={d2.max().item():.4f}  "
              f"relRMSE={rel:.2e}")

        # FP16 原链对照(量化误差 vs 半精度误差的量级感)
        out_fp16 = m32.map_cnn.half()(x16).float()
        d3 = (out_fp16 - ref_full).abs()
        rel3 = (d3.pow(2).mean().sqrt() / ref_full.pow(2).mean().sqrt()).item()
        print(f"     fp16 原链对照: relRMSE={rel3:.2e}")

        # ── ③ 延迟 ──────────────────────────────────────────────────────────
        m_ref16 = m32.map_cnn        # 已 half
        t_ref = cuda_time(lambda: m_ref16(x16), n_warmup=20, n_repeat=100)
        t_q = cuda_time(lambda: qcnn(x16), n_warmup=20, n_repeat=100)
        print(f"\n[③] eager 延迟: fp16 map_cnn={t_ref:.3f}ms  "
              f"QuantCNN={t_q:.3f}ms  ({t_ref/t_q:.2f}x)")

        # ── ④ compile 冒烟 ──────────────────────────────────────────────────
        cf = torch.compile(lambda t: qcnn(t), mode="max-autotune", dynamic=False)
        for _ in range(3):
            out_c = cf(x16)
        torch.cuda.synchronize()
        dc = (out_c.float() - out_q).abs().max().item()
        t_c = cuda_time(lambda: cf(x16), n_warmup=10, n_repeat=100)
        print(f"[④] compile: 数值 vs eager max|diff|={dc:.2e}  延迟={t_c:.3f}ms")

    dynamo.reset()
    print("\n单测通过。")


if __name__ == "__main__":
    main()

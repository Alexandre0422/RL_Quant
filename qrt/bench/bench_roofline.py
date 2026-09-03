#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_roofline.py — INT8 roofline 交叉点扫描(batch 维度)
══════════════════════════════════════════════════════════════════════════════
动机:项目核心论点是「INT8 的价值由逐算子的 roofline 位置决定,而非一刀切」。
bench_matrix 在 B=4096 给出一个反直觉结论——INT8 **linear** 是负项(+0.10ms),
因为大 batch 下这些 GEMM 已是激活/计算主导,量化的访存节省被 quant/dequant 开销
盖过。PLAN/README 进一步**预测**:在小 batch(B→1,板端单机部署)下,linear 退化
为**权重访存主导**,INT8 把权重字节减半 → 应翻转为净正项。但这个预测此前**未实测**。

本脚本在**同一张 3090**上扫 batch,把同一个算子从 roofline 的计算受限端推到访存
受限端,定位 ΔINT8_linear(B) = lat(F) − lat(E) 的**变号点 B***(以及 INT8 conv 的
对应曲线)。这把「INT8 价值随 roofline 位置变化」从纸面预测变成一条实测曲线。

口径(与 bench_matrix 一致,逐档叠加):
  C  FP16 compile(P0)              图融合基线
  D  + encoder 重写(P3)
  E  + QuantCNN(INT8 conv,P2)     —— linear 仍 FP16
  F  + LET-INT8 linear(P1/P5)      —— 推理完全体
边际:ΔconvE−D = lat(E)−lat(D) < 0 表示 INT8 conv 有收益;
      ΔlinF−E = lat(F)−lat(E)  < 0 表示 INT8 linear 有收益(预期仅小 B 成立)。

注意:
  - dynamic=False ⇒ 每个 batch 形状各自冷编译(~35s–数分钟/档),全扫程较长,
    建议远程后台 + tmux(见 CLAUDE.md「长任务」)。可用 QRT_BATCHES 缩小扫描。
  - 同一张 3090 扫 batch 是「固定硬件、移动算子在 roofline 上的位置」的科学控制;
    真实板端(Orin 等)硬件不同,B* 的绝对值会平移,但变号现象本身是硬件无关的
    roofline 性质。本脚本同时打印一阶 roofline 解析预测作对照。

运行(远程,空闲 GPU):
  CUDA_VISIBLE_DEVICES=2 python3 qrt/bench/bench_roofline.py
  QRT_BATCHES=1,16,256,4096 python3 qrt/bench/bench_roofline.py   # 缩小扫描
  QRT_CFGS=D,E,F python3 qrt/bench/bench_roofline.py              # 只跑关心的档
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (build_model, make_obs, make_terrain_obs,
                     cuda_time, cossim)

import time
import torch
import torch._dynamo as dynamo

from qrt import swap, calib, swap_cnn, calib_cnn_scale, patch_encoder

INT8_LIN = ["actor.0", "actor.2", "actor.4", "actor.6",
            "actor_proprio_embedding", "query_projector"]

BATCHES = [int(x) for x in os.environ.get(
    "QRT_BATCHES", "1,4,16,64,256,1024,4096").split(",")]
CFGS = os.environ.get("QRT_CFGS", "C,D,E,F").split(",")


# ── 配置构建器 ────────────────────────────────────────────────────────────────
def _compile(m, n_warm):
    def f(p):
        return m.act_inference({"policy_obs": p})[0]
    cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            cf(n_warm)
        torch.cuda.synchronize()
    return cf, time.perf_counter() - t0


def build_cfg(tag, sd, B):
    """按档构建 compiled act(po);返回 (act, compile_s)。各档逐层叠加。"""
    m = build_model(torch.float16)
    m.load_state_dict(sd)
    obs_fn = lambda: make_obs(torch.float16, batch=B)   # noqa: E731
    po16 = make_terrain_obs(torch.float16, batch=B)["policy_obs"]

    if tag in ("E", "F"):                                # INT8 conv
        if tag == "F":                                   # INT8 linear(LET)先 swap
            plan = calib.make_plan(m, INT8_LIN, obs_fn=obs_fn, use_let=True,
                                   let_steps=50, act_mode="dynamic", verbose=False)
            swap.apply(m, plan)
        s = calib_cnn_scale(m, obs_fn)
        swap_cnn(m, s)
    if tag in ("D", "E", "F"):                           # encoder 重写
        patch_encoder(m)
    return _compile(m, po16)


# ── 主扫描 ────────────────────────────────────────────────────────────────────
def roofline_predict(batches):
    """一阶 roofline 解析预测(actor MLP + emb/query 的 INT8 vs FP16 linear 总时间)。

    GEMM [B,in]→[B,out],权重 W[out,in]。时间 ≈ max(计算, 访存):
      计算_fp16 = 2·B·in·out / F16      计算_int8 = 2·B·in·out / I8
      访存_fp16 = (2·in·out + 2·B·in + 2·B·out)/BW
      访存_int8 = (1·in·out + 1·B·in + 2·B·out)/BW   (权重/激活 8bit,输出 fp16)
    小 B:权重项 in·out 主导 → INT8 访存减半 → 净正;
    大 B:计算/激活主导 → INT8 计算更快但本模型访存受限,量化开销反成净负。
    标称值(RTX 3090,非稀疏):F16≈71 TFLOP/s,I8≈284 TOP/s,HBM≈936 GB/s。
    """
    F16, I8, BW = 71e12, 284e12, 936e9
    # GLAD actor 主干 + 量化的两个 emb/proj(近似 in→out)
    layers = [(224, 512), (512, 256), (256, 128), (128, 29),  # actor.0/2/4/6
              (96, 64), (128, 64)]                            # proprio_emb, query_proj
    QUANT_BYTES = 2  # quant/dequant epilogue 的额外 elementwise 估算系数(每元素)
    out = {}
    for B in batches:
        t16 = t8 = 0.0
        for i, o in layers:
            comp16 = 2 * B * i * o / F16
            comp8 = 2 * B * i * o / I8
            mem16 = (2 * i * o + 2 * B * i + 2 * B * o) / BW
            mem8 = (1 * i * o + 1 * B * i + 2 * B * o
                    + QUANT_BYTES * B * i) / BW            # +量化 elementwise 访存
            t16 += max(comp16, mem16)
            t8 += max(comp8, mem8)
        out[B] = (t16 * 1e3, t8 * 1e3, (t8 - t16) * 1e3)    # ms, ms, Δms
    return out


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"扫描 batch: {BATCHES}   配置档: {CFGS}")
    print("GLAD 主配置(gc+topk),逐档叠加;ΔlinF−E<0 = INT8 linear 净正(roofline 翻转)\n")

    m32 = build_model(torch.float32)
    sd16 = {k: v.half().float() for k, v in m32.state_dict().items()}  # 同起点
    sd = {k: v.clone() for k, v in m32.state_dict().items()}
    del m32

    # results[B][cfg] = (lat_ms, cos)
    results = {B: {} for B in BATCHES}
    for B in BATCHES:
        print(f"━━━ B = {B} " + "━" * 60)
        # FP32 eager 参照(latency baseline + cos 参照)
        with torch.inference_mode():
            mA = build_model(torch.float32); mA.load_state_dict(sd)
            po32 = make_terrain_obs(torch.float32, batch=B)["policy_obs"]
            ref = mA.act_inference({"policy_obs": po32})[0].float().clone()
            tA = cuda_time(lambda: mA.act_inference({"policy_obs": po32}),
                           n_repeat=50)
            del mA
        results[B]["A"] = (tA, 1.0)
        print(f"  {'A FP32 eager':<24} {tA:8.3f}ms")

        po16 = make_terrain_obs(torch.float16, batch=B)["policy_obs"]
        for tag in CFGS:
            # INT8 linear 经 torch._int_mm 后端,要求 GEMM 的 M>16(=batch);
            # B≤16 无法跑(smoke 实测 RuntimeError),正是 roofline 预测 INT8 linear
            # 该有收益的小 batch 区——记录此「primitive 不可达」事实(EXP-001 结论)。
            if tag == "F" and B < 17:
                results[B][tag] = (float("nan"), float("nan"))
                print(f"  {tag:<24} {'skip':>8}     (_int_mm 需 M>16,B={B} INT8 linear 不可达)")
                continue
            dynamo.reset()
            torch.cuda.empty_cache()
            try:
                act, tc = build_cfg(tag, sd, B)
                with torch.inference_mode():
                    for _ in range(3):
                        act(po16)
                    torch.cuda.synchronize()
                    nrep = 300 if B <= 64 else 100   # 小 B 单步极快,加重复压噪声
                    lat = cuda_time(lambda: act(po16), n_warmup=20, n_repeat=nrep)
                    cos = cossim(act(po16).float(), ref)
                results[B][tag] = (lat, cos)
                print(f"  {tag:<24} {lat:8.3f}ms   ×FP32={tA/lat:5.2f}  "
                      f"CosSim={cos:.6f}   编译{tc:4.1f}s")
                del act
            except Exception as e:
                results[B][tag] = (float("nan"), float("nan"))
                print(f"  {tag:<24} {'FAIL':>8}     {type(e).__name__}: {str(e)[:80]}")
        print()

    # ── 汇总:逐档延迟 + INT8 边际(变号点)──────────────────────────────────
    print("=" * 84)
    print("延迟矩阵(ms),逐档叠加")
    hdr = f"{'B':>6} " + "".join(f"{c:>10}" for c in ["A"] + CFGS)
    print(hdr); print("-" * 84)
    for B in BATCHES:
        line = f"{B:>6} "
        for c in ["A"] + CFGS:
            line += f"{results[B].get(c, (float('nan'),))[0]:>10.3f}"
        print(line)

    print("\n" + "=" * 84)
    print("INT8 边际贡献 Δ(ms) = 加该档后 − 加之前;负 = 有收益")
    print(f"{'B':>6} {'ΔconvE−D':>12} {'ΔlinF−E':>12}   roofline 解析预测 Δlin(ms)")
    print("-" * 84)
    pred = roofline_predict(BATCHES)
    cross_meas = cross_pred = None
    prev_meas = prev_pred = None
    for B in BATCHES:
        r = results[B]
        d_conv = (r["E"][0] - r["D"][0]) if ("E" in r and "D" in r) else float("nan")
        d_lin = (r["F"][0] - r["E"][0]) if ("F" in r and "E" in r) else float("nan")
        p_lin = pred[B][2]
        print(f"{B:>6} {d_conv:>12.3f} {d_lin:>12.3f}   {p_lin:>+10.3f}")
        # 线性内插变号点(实测)
        if prev_meas is not None and not (d_lin != d_lin):
            if prev_meas[1] * d_lin < 0:
                b0, d0 = prev_meas; b1, d1 = B, d_lin
                cross_meas = b0 + (b1 - b0) * (0 - d0) / (d1 - d0)
        if not (d_lin != d_lin):
            prev_meas = (B, d_lin)
        if prev_pred is not None and prev_pred[1] * p_lin < 0:
            b0, d0 = prev_pred; b1, d1 = B, p_lin
            cross_pred = b0 + (b1 - b0) * (0 - d0) / (d1 - d0)
        prev_pred = (B, p_lin)

    print("=" * 84)
    print("结论:")
    print(f"  INT8 linear 实测变号点 B* ≈ {cross_meas if cross_meas else 'N/A(扫描区间内未变号)'}")
    print(f"  INT8 linear 解析预测 B* ≈ {cross_pred if cross_pred else 'N/A'}")
    print("  (Δlin 由正转负的 B 即 INT8 linear 从负项翻转为收益的边界;")
    print("   验证『INT8 价值由 roofline 位置决定』——同硬件、移动 batch 即可翻转。)")
    print("  注:量化是否真生效请用 _check_int8.py 验证 _int_mm kernel(精度过好是警报)。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_tdmpc2.py — TD-MPC2 roofline 镜像实验(EXP-002)
══════════════════════════════════════════════════════════════════════════════
论点闭环:GLAD 部署是 B=1(EXP-001 → weight-only),TD-MPC2 在线 MPPI 规划天然
把世界模型 rollout 推到 batch≈num_samples+num_pi_trajs(≈536,计算/激活主导)——
正是 `_int_mm`/IMMA(QuantLinear W8A8)该赢、GLAD 单步永远够不到的 roofline 端。

沿用 EXP-001 的方法学教训:**隔离带 GEMM 的计算单元**(estimate_value 的 batched
rollout),而非计整条 plan()(topk/softmax 更新是噪声)。扫 num_samples(TD-MPC2
独有的「batch 是超参」),画量化收益随 roofline 位置移动的曲线。

配置档(逐档叠加,只在 estimate_value 上):
  A  FP16 eager                      纯基线
  C  FP16 compile(max-autotune)      图融合 + CUDA Graph(qrt runtime 同款)
  E  C + QuantLinear(W8A8)+LET       rollout INT8(dynamic s_z)
边际:ΔlinE−C = lat(E)−lat(C);< 0 = INT8 linear 在该 batch 净正
(GLAD 在 B=4096 测出为正=负项;TD-MPC2 的 B≈536 是否翻正 → 本实验回答)。

encoder(B=1,WeightOnlyLinear)另行单测(open_encoder_bench),因其与 rollout 分属
roofline 两端、量级差两个数量级,混在一起会被 rollout 淹没。

运行(远程,空闲卡,长任务 tmux 后台;每个 num_samples 各自冷编译):
  CUDA_VISIBLE_DEVICES=2 python3 qrt/bench/bench_tdmpc2.py
  QRT_SAMPLES=64,256,512 QRT_CFGS=C,E python3 qrt/bench/bench_tdmpc2.py   # 缩小扫描
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # 项目根,供 import qrt
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import time
import torch
import torch._dynamo as dynamo

from qrt.tdmpc2 import (TDMPC2Config, build_world_model,
                        quantize_rollout, quantize_encoder, rollout_linear_paths)

SAMPLES = [int(x) for x in os.environ.get("QRT_SAMPLES",
                                          "64,128,256,512,1024").split(",")]
CFGS = os.environ.get("QRT_CFGS", "A,C,E").split(",")
DEV = "cuda"


def cuda_time(fn, n_warmup=20, n_repeat=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_repeat):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_repeat


def cossim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def make_rollout_inputs(cfg: TDMPC2Config, N: int, dtype):
    """isolate estimate_value:z0[N,L] + actions[H,N,A]。"""
    z0 = torch.randn(N, cfg.latent_dim, device=DEV, dtype=dtype)
    acts = torch.rand(cfg.horizon, N, cfg.action_dim, device=DEV, dtype=dtype) * 2 - 1
    return z0, acts


def build_estimate_value(cfg, sd, N, tag, dtype=torch.float16):
    """按档构建 compiled estimate_value(z0,acts);返回 (fn, compile_s)。"""
    wm = build_world_model(cfg, dtype=dtype, device=DEV)
    wm.load_state_dict(sd)
    if tag == "E":
        obs_fn = lambda: torch.randn(cfg.obs_dim, device=DEV, dtype=dtype)  # noqa: E731
        quantize_rollout(wm, obs_fn=obs_fn, act_mode="dynamic",
                         use_let=True, let_steps=50, verbose=False)

    def ev(z0, acts):
        return wm.estimate_value(z0, acts)

    if tag == "A":
        return ev, 0.0
    cf = torch.compile(ev, mode="max-autotune", fullgraph=True, dynamic=False)
    z0, acts = make_rollout_inputs(cfg, N, dtype)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            cf(z0, acts)
        torch.cuda.synchronize()
    return cf, time.perf_counter() - t0


def roofline_predict(cfg: TDMPC2Config, samples):
    """一阶 roofline 解析预测:estimate_value 内所有 rollout GEMM 的 FP16 vs INT8 总时间。
    标称 RTX 3090:F16≈71 TFLOP/s,I8≈284 TOP/s,HBM≈936 GB/s。
    时间 ≈ max(计算, 访存);INT8 权重/激活 8bit,+quant elementwise 访存。"""
    F16, I8, BW, QB = 71e12, 284e12, 936e9, 2
    L, M, A, NB, H = (cfg.latent_dim, cfg.mlp_dim, cfg.action_dim,
                      cfg.num_bins, cfg.horizon)
    # (in, out, 调用次数/estimate_value)
    per_step = [(L + A, M), (M, M), (M, L),          # dynamics ×H
                (L + A, M), (M, M), (M, NB)]         # reward   ×H
    terminal = [(L, M), (M, M), (M, 2 * A)]          # pi ×1
    terminal += [(L + A, M), (M, M), (M, NB)] * cfg.num_q   # Q 集成 ×1
    out = {}
    for ns in samples:
        B = ns + cfg.num_pi_trajs
        layers = [(i, o, H) for (i, o) in per_step] + [(i, o, 1) for (i, o) in terminal]
        t16 = t8 = 0.0
        for i, o, rep in layers:
            comp16 = 2 * B * i * o / F16
            comp8 = 2 * B * i * o / I8
            mem16 = (2 * i * o + 2 * B * i + 2 * B * o) / BW
            mem8 = (1 * i * o + 1 * B * i + 2 * B * o + QB * B * i) / BW
            t16 += rep * max(comp16, mem16)
            t8 += rep * max(comp8, mem8)
        out[ns] = (t16 * 1e3, t8 * 1e3, (t8 - t16) * 1e3)
    return out


def open_encoder_bench(cfg, sd):
    """encoder B=1 单测:WeightOnlyLinear(W8A16/W4A16)vs FP16。roofline 另一端。"""
    print("\n" + "=" * 84)
    print("encoder(B=1)weight-only 单测 —— roofline 权重访存主导端")
    print("-" * 84)
    obs = torch.randn(1, cfg.obs_dim, device=DEV, dtype=torch.float16)
    for bits, tag in [(None, "FP16"), (8, "W8A16"), (4, "W4A16")]:
        wm = build_world_model(cfg, dtype=torch.float16, device=DEV)
        wm.load_state_dict(sd)
        if bits is not None:
            if cfg.enc_dim % 2 or cfg.obs_dim % 2:
                if bits == 4:
                    print(f"  {tag:<8} skip (W4 需偶数 in_features)"); continue
            quantize_encoder(wm, bits=bits, verbose=False)
        with torch.inference_mode():
            enc = lambda: wm.encode(obs)   # noqa: E731
            for _ in range(20):
                enc()
            torch.cuda.synchronize()
            lat = cuda_time(enc, n_warmup=50, n_repeat=500)
        print(f"  {tag:<8} {lat*1e3:8.2f}µs")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA(远程 3090 运行);本地仅语法检查。"); return
    torch.manual_seed(0)
    cfg = TDMPC2Config()
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"TD-MPC2 默认 config: L={cfg.latent_dim} M={cfg.mlp_dim} A={cfg.action_dim} "
          f"H={cfg.horizon} num_pi_trajs={cfg.num_pi_trajs} num_q={cfg.num_q}")
    ok, skip = rollout_linear_paths(build_world_model(cfg, device=DEV))
    print(f"rollout 可量化 Linear {len(ok)},跳过(K%8!=0){len(skip)}: {skip}")
    print(f"扫描 num_samples: {SAMPLES}(→ rollout batch = num_samples+{cfg.num_pi_trajs})")
    print(f"配置档: {CFGS}   ΔlinE−C<0 = INT8 linear 净正(roofline 翻转)\n")

    # 同起点权重
    wm0 = build_world_model(cfg, dtype=torch.float32, device=DEV)
    sd = {k: v.clone() for k, v in wm0.state_dict().items()}
    sd16 = {k: (v.half().float() if v.is_floating_point() else v)
            for k, v in sd.items()}
    del wm0

    results = {ns: {} for ns in SAMPLES}
    for ns in SAMPLES:
        N = ns + cfg.num_pi_trajs
        print(f"━━━ num_samples={ns}  (batch N={N}) " + "━" * 40)
        z0_16, acts_16 = make_rollout_inputs(cfg, N, torch.float16)
        # FP32 eager 参照(cos 基准)
        with torch.inference_mode():
            wmref = build_world_model(cfg, dtype=torch.float32, device=DEV)
            wmref.load_state_dict(sd)
            z0_32, acts_32 = z0_16.float(), acts_16.float()
            ref = wmref.estimate_value(z0_32, acts_32).float().clone()
            del wmref

        for tag in CFGS:
            dynamo.reset(); torch.cuda.empty_cache()
            try:
                fn, tc = build_estimate_value(cfg, sd16, N, tag)
                with torch.inference_mode():
                    for _ in range(3):
                        fn(z0_16, acts_16)
                    torch.cuda.synchronize()
                    nrep = 200 if ns <= 256 else 100
                    lat = cuda_time(lambda: fn(z0_16, acts_16), n_repeat=nrep)
                    cos = cossim(fn(z0_16, acts_16), ref)
                results[ns][tag] = (lat, cos)
                print(f"  {tag:<20} {lat:8.4f}ms  CosSim={cos:.6f}  编译{tc:5.1f}s")
                del fn
            except Exception as ex:
                results[ns][tag] = (float("nan"), float("nan"))
                print(f"  {tag:<20} {'FAIL':>8}   {type(ex).__name__}: {str(ex)[:80]}")
        print()

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("=" * 84)
    print("estimate_value 延迟(ms)+ INT8 边际;逐档叠加")
    print(f"{'num_samp':>9} {'N':>6}" + "".join(f"{c:>12}" for c in CFGS) +
          f"{'ΔlinE−C':>12}   roofline预测Δ")
    print("-" * 84)
    pred = roofline_predict(cfg, SAMPLES)
    cross = prev = None
    for ns in SAMPLES:
        r = results[ns]
        line = f"{ns:>9} {ns+cfg.num_pi_trajs:>6}"
        for c in CFGS:
            line += f"{r.get(c, (float('nan'),))[0]:>12.4f}"
        d = (r["E"][0] - r["C"][0]) if ("E" in r and "C" in r) else float("nan")
        line += f"{d:>12.4f}   {pred[ns][2]:>+8.4f}"
        print(line)
        if prev is not None and not (d != d) and prev[1] * d < 0:
            b0, d0 = prev; cross = b0 + (ns - b0) * (0 - d0) / (d - d0)
        if not (d != d):
            prev = (ns, d)
    print("=" * 84)
    print(f"结论:INT8 linear 实测变号 num_samples* ≈ "
          f"{cross if cross else 'N/A(区间内未变号)'}")
    print("  (ΔlinE−C 由正转负处 = TD-MPC2 规划批量下 INT8 linear 从负项翻为收益的边界;")
    print("   同硬件、移动 num_samples 即翻转 → 验证「INT8 价值由 roofline 位置决定」。)")
    print("  注:量化是否真生效用 qrt/bench/_check_int8.py grep _int_mm(精度过好是警报)。")

    open_encoder_bench(cfg, sd16)


if __name__ == "__main__":
    main()

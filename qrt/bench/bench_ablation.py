#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_ablation.py — P1: 真实权重下的 LET-INT8 精度消融 + 训练漂移实验
══════════════════════════════════════════════════════════════════════════════
权重: AME_Locomotion-main/pretrained/ame1.pt(iter 21000 真实收敛 checkpoint,
      variant="ame": use_global_context=False, topk=None,与 GLAD 共享
      CNN/MHA/MLP 主干 —— 量化精度结论可迁移)
obs : make_terrain_obs 结构化地形(xy 近常量网格 + 平滑台阶高度场,
      接近真实 rollout 分布形态;校准集与测试集 seed 分离)

Part 1  逐层/全量消融: naive vs LET,CosSim/max|Δaction|(eager 测精度,
        compile 与 eager 数值一致已验证)
Part 2  训练漂移(30 iter dummy PPO update,权重持续变化):
        a) LET 静态 s_z   b) LET + s_z 在线 EMA(update 前向 hook,零额外成本)
        c) naive 静态     —— 每 iter refresh 后测精度演化(论文在线校准章节素材)
Part 3  refresh_quant CUDA Graph 化前后开销对比

运行(远程): python3 qrt/bench/bench_ablation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B, PA, MAP, DEVICE, build_model, load_ame_ckpt,
                     make_terrain_obs, cuda_time)

import copy
import time
import torch
import torch._dynamo as dynamo

from qrt import InferenceRunner, swap, calib

PATHS = ["actor.0", "actor.2", "actor.4", "actor.6", "actor_proprio_embedding"]
N_TEST = 4          # 测试 batch 数(seed 与校准集分离)
N_CALIB = 8
DRIFT_ITERS = 30
MB_SZ, N_MB = 1024, 4


def gen_obs(seed):
    g = torch.Generator(device=DEVICE.type)
    g.manual_seed(seed)
    return make_terrain_obs(torch.float16, gen=g)


def metrics(out, ref):
    """返回 (CosSim, max|Δ|, relRMSE)。out/ref: [B, NA]"""
    o, r = out.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    mad = (o - r).abs().max().item()
    rel = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
    return cos, mad, rel


@torch.inference_mode()
def eval_model(model, test_obs, ref_outs):
    cs, ms, rs = [], [], []
    for obs, ref in zip(test_obs, ref_outs):
        out = model.act_inference(obs)[0]
        c, m, r = metrics(out, ref)
        cs.append(c); ms.append(m); rs.append(r)
    n = len(cs)
    return sum(cs) / n, max(ms), sum(rs) / n


def sync_weights(dst, src_sd):
    dst.load_state_dict(src_sd, strict=False)   # 量化 buffer 键自动忽略


def main():
    torch.manual_seed(0)        # 固定 LET 校准的 minibatch 采样,保证可复现
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"权重: ame1.pt(真实收敛)  obs: 结构化地形  量化层: {PATHS}")

    # ── 数据与参照 ────────────────────────────────────────────────────────────
    test_obs = [gen_obs(1000 + i) for i in range(N_TEST)]
    test_obs32 = [{k: v.float() for k, v in o.items()} for o in test_obs]

    ref32 = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    with torch.inference_mode():
        ref_outs = [ref32.act_inference(o)[0].clone() for o in test_obs32]

    m16 = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    sd16 = {k: v.clone() for k, v in m16.state_dict().items()}
    c, mx, r = eval_model(m16, test_obs, ref_outs)
    print(f"\nFP16 eager 基线: CosSim={c:.8f}  max|Δa|={mx:.4f}  relRMSE={r:.2e}")

    # ── 校准(同一份激活样本喂 naive 与 LET,严格可比)───────────────────────
    _seed = [0]
    def calib_obs():
        _seed[0] += 1
        return gen_obs(_seed[0])

    print(f"\n[校准] 结构化 obs ×{N_CALIB} batch")
    t0 = time.perf_counter()
    plan_let = calib.make_plan(m16, PATHS, obs_fn=calib_obs, use_let=True,
                               n_calib=N_CALIB, let_steps=200, verbose=False)
    plan_naive = calib.make_plan(m16, PATHS, obs_fn=calib_obs, use_let=False,
                                 n_calib=N_CALIB, verbose=False)
    print(f"  校准耗时 {time.perf_counter()-t0:.1f}s(LET 200步×5层 + naive)")

    # ── Part 1: 逐层 / 全量消融 ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Part 1  逐层/全量消融(真实权重,eager 精度)")
    print("=" * 78)
    print(f"{'配置':<34} {'CosSim':>12} {'max|Δa|':>9} {'relRMSE':>9}")
    print("-" * 78)

    def ablate(tag, plan_subset):
        sw = swap.apply(m16, plan_subset)
        c, mx, r = eval_model(m16, test_obs, ref_outs)
        swap.restore(m16, sw)
        print(f"{tag:<34} {c:>12.8f} {mx:>9.4f} {r:>9.2e}")
        return c

    for p in PATHS:
        ablate(f"naive [{p}]", {p: plan_naive[p]})
    for p in PATHS:
        ablate(f"LET   [{p}]", {p: plan_let[p]})
    c_naive = ablate("naive [全部 5 层]", plan_naive)
    c_let = ablate("LET   [全部 5 层]", plan_let)
    print("-" * 78)
    print(f"全量化精度: LET {'优于' if c_let > c_naive else '不及'} naive "
          f"(ΔCosSim={c_let-c_naive:+.2e})")

    # ── Part 2: 训练漂移实验 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"Part 2  训练漂移({DRIFT_ITERS} iter dummy PPO update,lr=1e-4)")
    print("  a) LET 静态s_z   b) LET+在线EMA(update前向hook)   c) naive 静态")
    print("=" * 78)

    # 训练器(纯 Linear,FP32——标准实践: 训练全精度,rollout 量化)产生权重轨迹;
    # 三个 FP16 量化模型 + FP32 参照每 iter 同步权重(load_state_dict 自动 cast)
    trainer = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-4)

    def fresh_quant(plan):
        m = load_ame_ckpt(build_model(torch.float16, variant="ame"))
        m.load_state_dict(sd16)
        return m, swap.apply(m, copy.deepcopy(plan))

    qa, sw_a = fresh_quant(plan_let)     # a: LET 静态
    qb, sw_b = fresh_quant(plan_let)     # b: LET + 在线 EMA s_z
    qc, sw_c = fresh_quant(plan_naive)   # c: naive 静态

    # b 的在线 s_z: 在 qb 的 train 态前向上挂 hook 收集 **LET 变换后 z=αx+β** 的
    # absmax(EMA)。注意 s_z 的语义是 z 的 scale 而非 x 的 —— 直接收集 x 会把
    # s_z 设到错误尺度(实测立即劣化,已修)。
    ema = {p: float(sw_b[p].s_z) * 127.0 for p in PATHS}    # 以 absmax 形式维护
    hooks = []
    def mk_hook(p):
        ql = sw_b[p]
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            z = x * ql.alpha + ql.beta if ql.has_let else x
            ema[p] = 0.9 * ema[p] + 0.1 * z.abs().amax().item()
        return fn
    for p in PATHS:
        hooks.append(sw_b[p].register_forward_hook(mk_hook(p)))

    mb_pool32 = [
        {k: v.float() for k, v in make_terrain_obs(torch.float16, batch=MB_SZ).items()}
        for _ in range(N_MB)
    ]
    mb16 = {k: v.half() for k, v in mb_pool32[0].items()}   # qb 在线校准前向用

    def update(model, optimizer):
        model.train()
        for mb in mb_pool32:
            actor_obs = model.actor_obs_normalizer(model.get_actor_obs(mb))
            model.update_distribution(actor_obs)
            loss = model.distribution.mean.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

    print(f"{'iter':>4} {'a) LET静态':>14} {'b) LET+EMA':>14} {'c) naive静态':>14}")
    log = []
    for it in range(DRIFT_ITERS + 1):
        if it > 0:
            update(trainer, opt)
            sd = trainer.state_dict()
            for m_ in (qa, qc, ref32):
                sync_weights(m_, sd)
            sync_weights(qb, sd)
            # b: 一个 minibatch 的 train 前向触发 hook(真实训练中由 PPO 前向免费提供)
            qb.train()
            with torch.no_grad():
                actor_obs = qb.actor_obs_normalizer(qb.get_actor_obs(mb16))
                qb.update_distribution(actor_obs)
            qb.eval()
            for p in PATHS:
                sw_b[p].set_act_scale_(ema[p] / 127.0)
            for sw_ in (sw_a, sw_b, sw_c):
                swap.refresh(sw_)
            with torch.inference_mode():
                ref_outs = [ref32.act_inference(o)[0].clone() for o in test_obs32]
        ca, _, _ = eval_model(qa, test_obs, ref_outs)
        cb, _, _ = eval_model(qb, test_obs, ref_outs)
        cc, _, _ = eval_model(qc, test_obs, ref_outs)
        log.append((it, ca, cb, cc))
        if it % 5 == 0 or it == 1:
            print(f"{it:>4} {ca:>14.8f} {cb:>14.8f} {cc:>14.8f}")
    for h in hooks:
        h.remove()
    a_end, b_end, c_end = log[-1][1], log[-1][2], log[-1][3]
    print("-" * 78)
    print(f"漂移 {DRIFT_ITERS} iter 后: LET静态={a_end:.8f}  LET+EMA={b_end:.8f}  "
          f"naive={c_end:.8f}")
    print(f"在线 EMA 增益: {b_end - a_end:+.2e}   LET vs naive(静态): {a_end - c_end:+.2e}")

    # ── Part 3: refresh graph 化 + AME 配置延迟参考 ──────────────────────────
    print("\n" + "=" * 78)
    print("Part 3  refresh_quant 开销(graph 化前后)+ AME 配置延迟")
    print("=" * 78)
    runner = InferenceRunner(qa, mode="max-autotune")
    t_eager_ref = cuda_time(lambda: runner.refresh_quant(use_graph=False),
                            n_warmup=5, n_repeat=50)
    t_graph_ref = cuda_time(lambda: runner.refresh_quant(use_graph=True),
                            n_warmup=5, n_repeat=50)
    print(f"refresh(逐层 eager): {t_eager_ref:.3f}ms   refresh(CUDA Graph): "
          f"{t_graph_ref:.3f}ms   ({t_eager_ref/t_graph_ref:.1f}x)")
    # graph 化后正确性: 改权重 → graph refresh → 与重新逐层 requantize 一致
    with torch.no_grad():
        qa.actor[0].weight.mul_(1.02)
    runner.refresh_quant(use_graph=True)
    w8_g = sw_a["actor.0"].w8.clone()
    sw_a["actor.0"].requantize_()
    diff = (w8_g.int() - sw_a["actor.0"].w8.int()).abs().max().item()
    print(f"graph refresh 正确性: w8 与逐层重算 max|diff|={diff}(应=0)")

    po = test_obs[0]["policy_obs"]
    t0 = time.perf_counter()
    runner.warmup(po)
    print(f"\nAME 配置(187 token MHA)LET-INT8 compile: warmup {time.perf_counter()-t0:.0f}s")
    t_let = cuda_time(lambda: runner.act(po))
    del runner
    dynamo.reset()
    m_fp = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    r2 = InferenceRunner(m_fp, mode="max-autotune")
    t0 = time.perf_counter()
    r2.warmup(po)
    print(f"AME 配置 FP16 compile: warmup {time.perf_counter()-t0:.0f}s")
    t_fp = cuda_time(lambda: r2.act(po))
    with torch.inference_mode():
        t_fp_eager = cuda_time(lambda: m_fp.act_inference(test_obs[0]), n_repeat=50)
    print(f"AME 单步: FP16 eager={t_fp_eager:.3f}ms  FP16 compile={t_fp:.3f}ms  "
          f"LET-INT8 compile={t_let:.3f}ms")
    print("\n完成。")


if __name__ == "__main__":
    main()

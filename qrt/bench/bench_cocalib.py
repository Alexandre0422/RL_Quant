#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_cocalib.py — P6: 在线伴随校准 五方漂移对照(论文图数据)
══════════════════════════════════════════════════════════════════════════════
设置同 P1/P5(ame1 真实收敛权重 + 结构化地形 obs,50 iter dummy update 漂移),
全部配置使用 act_mode="dynamic"(P5 默认,隔离 α/β 效应)。

对照组(精度 vs 累计校准步数):
  A frozen        α/β 冻结(P5 dynamic 组,基线)                成本 0
  B periodic-10   每 10 iter 全量重校准(fresh 200 步/层)        成本 5 层×200×5 次
  C fixed-1       每 iter 每层固定 1 步(warm-start,无触发)     成本 5 层×50
  D adaptive      CoCalibrator(τ=1.25, k_max=8,误差驱动)      成本 实测
预期: D 以远低于 B 的步数贴住 B 的精度上界;A 持续退化;C 介于其间。

运行(远程): python3 qrt/bench/bench_cocalib.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_model, load_ame_ckpt, make_terrain_obs

import copy
import time
import torch

from qrt import swap, calib
from qrt.cocalib import CoCalibrator
from qrt.tap import install_taps

PATHS = ["actor.0", "actor.2", "actor.4", "actor.6", "actor_proprio_embedding"]
N_TEST, N_CALIB, ITERS = 4, 8, 50
MB_SZ, N_MB = 1024, 4
DEVICE = torch.device("cuda")


def gen_obs(seed, batch=4096):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return make_terrain_obs(torch.float16, batch=batch, gen=g)


@torch.inference_mode()
def eval_cos(model, test_obs, ref_outs):
    cs = []
    for obs, ref in zip(test_obs, ref_outs):
        out = model.act_inference(obs)[0].float()
        cs.append(torch.nn.functional.cosine_similarity(
            out.flatten(), ref.float().flatten(), dim=0).item())
    return sum(cs) / len(cs)


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"协议: {ITERS} iter 漂移,act_mode=dynamic,量化层 {PATHS}")

    test_obs = [gen_obs(1000 + i) for i in range(N_TEST)]
    test_obs32 = [{k: v.float() for k, v in o.items()} for o in test_obs]
    ref32 = load_ame_ckpt(build_model(torch.float32, variant="ame"))

    m0 = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    sd16 = {k: v.clone() for k, v in m0.state_dict().items()}
    _seed = [0]

    def calib_obs():
        _seed[0] += 1
        return gen_obs(_seed[0])

    t0 = time.perf_counter()
    plan = calib.make_plan(m0, PATHS, obs_fn=calib_obs, use_let=True,
                           n_calib=N_CALIB, let_steps=200,
                           act_mode="dynamic", verbose=False)
    print(f"初始 LET 校准 {time.perf_counter()-t0:.1f}s(各配置共享同一起点)")
    del m0

    # ── 训练器(共享权重轨迹)与四个量化模型 ──────────────────────────────────
    trainer = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-4)

    def fresh():
        m = load_ame_ckpt(build_model(torch.float16, variant="ame"))
        m.load_state_dict(sd16)
        sw = swap.apply(m, copy.deepcopy(plan))
        return m, sw

    mA, swA = fresh()                                   # frozen
    mB, swB = fresh()                                   # periodic-10
    mC, swC = fresh()                                   # fixed-1 (eager 独立前向)
    mD, swD = fresh()                                   # adaptive
    mE, swE = fresh()                                   # fixed-1 (tap, 0 额外前向)

    # trainer 装 tap(纯采集,数值恒等);E 组从 update 前向便车读激活
    tap_slots = install_taps(trainer, PATHS, sample_rows=2048)

    ccC = CoCalibrator(trainer, swC, tau=0.0, k_max=1)  # τ=0 → 永远触发,1 步
    ccD = CoCalibrator(trainer, swD, tau=1.25, k_max=8)
    ccE = CoCalibrator(trainer, swE, tau=0.0, k_max=1,
                       source="tap", tap_slots=tap_slots)
    stepsB = [0]

    mb32 = [{k: v.float() for k, v in
             make_terrain_obs(torch.float16, batch=MB_SZ).items()}
            for _ in range(N_MB)]

    def update():
        trainer.train()
        for mb in mb32:
            ao = trainer.actor_obs_normalizer(trainer.get_actor_obs(mb))
            trainer.update_distribution(ao)
            loss = trainer.distribution.mean.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        trainer.eval()

    def periodic_recalib():
        """B: fresh 重校准(重新初始化 + 200 步/层),计步。"""
        def obs32_fn():                       # FP32 训练模型需 fp32 obs
            o = gen_obs(9000 + stepsB[0])
            return {k: v.float() for k, v in o.items()}
        acts = calib.collect_inputs(trainer, PATHS, obs_fn=obs32_fn, n_batches=4)
        for p in PATHS:
            lin = trainer.get_submodule(p)
            alpha, beta, _ = calib.fit_let(acts[p], lin, n_steps=200, verbose=False)
            with torch.no_grad():
                swB[p].alpha.copy_(alpha.clamp_min(1e-8).to(DEVICE))
                swB[p].beta.copy_(beta.to(DEVICE))
            stepsB[0] += 200

    def sync_to(m, sw):
        m.load_state_dict(trainer.state_dict(), strict=False)
        swap.refresh(sw)

    print(f"\n{'iter':>4} {'A frozen':>12} {'B periodic':>12} {'C fix1-eager':>12} "
          f"{'D adaptive':>12} {'E fix1-tap':>12}")
    for it in range(ITERS + 1):
        if it > 0:
            update()                       # ← tap 在此前向中已写 buffer(E 用)
            # C/D: eager 独立前向采集;E: tap 便车(读 update 已写的 buffer)
            ccC.step(mb32[it % N_MB])
            ccD.step(mb32[it % N_MB])
            ccE.step(None)                 # source=tap,obs 参数不用
            if it % 10 == 0:
                periodic_recalib()
            ref32.load_state_dict(trainer.state_dict(), strict=False)
            for m_, sw_ in ((mA, swA), (mB, swB), (mC, swC), (mD, swD), (mE, swE)):
                sync_to(m_, sw_)
        with torch.inference_mode():
            ref_outs = [ref32.act_inference(o)[0].clone() for o in test_obs32]
        if it % 5 == 0 or it == 1:
            row = [eval_cos(m_, test_obs, ref_outs) for m_ in (mA, mB, mC, mD, mE)]
            print(f"{it:>4} " + " ".join(f"{c:>12.8f}" for c in row))

    print("-" * 92)
    print(f"最终累计校准步数: B periodic={stepsB[0]}  C/E fixed-1={ccC.total_steps}  "
          f"D adaptive={ccD.total_steps}")
    print(f"C(eager独立前向) vs E(tap 0额外前向) 末值对照 —— 应吻合(数值等效验证)")
    print(ccD.summary())
    print("完成。")


if __name__ == "__main__":
    main()

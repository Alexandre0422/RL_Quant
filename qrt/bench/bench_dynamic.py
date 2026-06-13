#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_dynamic.py — P5: 动态 scale 融编译图(四方对照)
══════════════════════════════════════════════════════════════════════════════
借鉴 torchao float8 的 dynamic-scaling 经验(delayed→dynamic 演化):
  static        校准 s_z(P1 现状,有漂移问题,EMA 补丁)
  dynamic       s_z = absmax(z)/127 图内现算;α/β 仍 LET 校准 → 漂移源消失
  dynamic_full  α/β=batch 统计、权重折叠+量化全图内 → **零校准**(LET→动态标准化)

Part A  精度(ame1 真实收敛权重 + 结构化 obs,eager,与 P1 同设置可比)
Part B  训练漂移 30 iter(预期: dynamic 两组曲线平坦,static 持续退化)
Part C  compile 延迟(GLAD 配置,B=4096,max-autotune,vs FP16/static 基线)

运行(远程): python3 qrt/bench/bench_dynamic.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B, DEVICE, build_model, load_ame_ckpt, make_terrain_obs,
                     make_obs, cuda_time)

import copy
import time
import torch
import torch._dynamo as dynamo

from qrt import swap, calib

PATHS_AME = ["actor.0", "actor.2", "actor.4", "actor.6", "actor_proprio_embedding"]
PATHS_GLAD = PATHS_AME + ["query_projector"]
N_TEST, N_CALIB, DRIFT_ITERS = 4, 8, 30
MB_SZ, N_MB = 1024, 4


def gen_obs(seed):
    g = torch.Generator(device=DEVICE.type)
    g.manual_seed(seed)
    return make_terrain_obs(torch.float16, gen=g)


@torch.inference_mode()
def eval_model(model, test_obs, ref_outs):
    cs, rs = [], []
    for obs, ref in zip(test_obs, ref_outs):
        out = model.act_inference(obs)[0].float()
        r = ref.float()
        cs.append(torch.nn.functional.cosine_similarity(
            out.flatten(), r.flatten(), dim=0).item())
        rs.append(((out - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item())
    return sum(cs) / len(cs), sum(rs) / len(rs)


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")

    # ════════ Part A: 精度(ame1 真实权重)════════════════════════════════════
    print("\n" + "=" * 76)
    print("Part A  精度(ame1 真实权重,eager,CosSim/relRMSE vs FP32)")
    print("=" * 76)
    test_obs = [gen_obs(1000 + i) for i in range(N_TEST)]
    test_obs32 = [{k: v.float() for k, v in o.items()} for o in test_obs]
    ref32 = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    with torch.inference_mode():
        ref_outs = [ref32.act_inference(o)[0].clone() for o in test_obs32]

    m16 = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    sd16 = {k: v.clone() for k, v in m16.state_dict().items()}
    c, r = eval_model(m16, test_obs, ref_outs)
    print(f"{'FP16 基线':<26} CosSim={c:.8f}  relRMSE={r:.2e}")

    _seed = [0]
    def calib_obs():
        _seed[0] += 1
        return gen_obs(_seed[0])

    plan_let = calib.make_plan(m16, PATHS_AME, obs_fn=calib_obs, use_let=True,
                               n_calib=N_CALIB, let_steps=200, verbose=False)

    def with_mode(plan, mode):
        p2 = {k: dict(v) for k, v in plan.items()}
        for v in p2.values():
            v["act_mode"] = mode
        return p2

    configs = [
        ("static (LET, P1 现状)", with_mode(plan_let, "static")),
        ("dynamic (LET + 图内s_z)", with_mode(plan_let, "dynamic")),
        ("dynamic_full (零校准)", calib.make_plan(m16, PATHS_AME, act_mode="dynamic_full")),
    ]
    plans = {}
    for tag, plan in configs:
        sw = swap.apply(m16, copy.deepcopy(plan))
        c, r = eval_model(m16, test_obs, ref_outs)
        swap.restore(m16, sw)
        plans[tag] = plan
        print(f"{tag:<26} CosSim={c:.8f}  relRMSE={r:.2e}")

    # ════════ Part B: 训练漂移 ════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print(f"Part B  训练漂移({DRIFT_ITERS} iter update;dynamic 应平坦)")
    print("=" * 76)
    trainer = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-4)

    def fresh(plan):
        m = load_ame_ckpt(build_model(torch.float16, variant="ame"))
        m.load_state_dict(sd16)
        return m, swap.apply(m, copy.deepcopy(plan))

    qs, sws, tags = [], [], []
    for tag, plan in plans.items():
        m, sw = fresh(plan)
        qs.append(m); sws.append(sw); tags.append(tag.split(" ")[0])

    mb32 = [{k: v.float() for k, v in make_terrain_obs(torch.float16, batch=MB_SZ).items()}
            for _ in range(N_MB)]

    def update():
        trainer.train()
        for mb in mb32:
            ao = trainer.actor_obs_normalizer(trainer.get_actor_obs(mb))
            trainer.update_distribution(ao)
            loss = trainer.distribution.mean.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        trainer.eval()

    hdr = "".join(f"{t:>16}" for t in tags)
    print(f"{'iter':>4}{hdr}")
    for it in range(DRIFT_ITERS + 1):
        if it > 0:
            update()
            sd = trainer.state_dict()
            ref32.load_state_dict(sd, strict=False)
            with torch.inference_mode():
                ref_outs = [ref32.act_inference(o)[0].clone() for o in test_obs32]
            for m, sw in zip(qs, sws):
                m.load_state_dict(sd, strict=False)
                swap.refresh(sw)                       # dynamic_full 内为 no-op
        if it % 5 == 0 or it == 1:
            row = ""
            for m in qs:
                c, _ = eval_model(m, test_obs, ref_outs)
                row += f"{c:>16.8f}"
            print(f"{it:>4}{row}")

    # ════════ Part C: compile 延迟(GLAD 配置)═══════════════════════════════
    print("\n" + "=" * 76)
    print("Part C  compile 延迟(GLAD 配置,B=4096,max-autotune)")
    print("=" * 76)
    obs16 = make_obs(torch.float16)
    po = obs16["policy_obs"]
    results = []
    for tag, mode in [("FP16(无INT8)", None), ("static", "static"),
                      ("dynamic", "dynamic"), ("dynamic_full", "dynamic_full")]:
        dynamo.reset()
        torch.cuda.empty_cache()
        m = build_model(torch.float16)
        if mode is not None:
            if mode == "dynamic_full":
                plan = calib.make_plan(m, PATHS_GLAD, act_mode="dynamic_full")
            else:
                plan = calib.make_plan(m, PATHS_GLAD,
                                       obs_fn=lambda: make_obs(torch.float16),
                                       use_let=True, let_steps=100,
                                       act_mode=mode, verbose=False)
            swap.apply(m, plan)

        def f(p):
            return m.act_inference({"policy_obs": p})[0]
        cf = torch.compile(f, mode="max-autotune", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):
                cf(po)
            torch.cuda.synchronize()
            t = cuda_time(lambda: cf(po))
        print(f"{tag:<16} {t:.3f}ms   (编译 {time.perf_counter()-t0:.0f}s)")
        results.append((tag, t))
        del m, cf

    print("\n完成。")


if __name__ == "__main__":
    main()

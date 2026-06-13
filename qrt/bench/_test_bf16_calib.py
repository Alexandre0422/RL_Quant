#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_test_bf16_calib.py — 思路 C 验证: bf16 update 前向激活 vs FP32 激活做 LET 校准
══════════════════════════════════════════════════════════════════════════════
问题: 若从 bf16 autocast 的 update 前向采集激活做 LET 校准,~1e-3 bf16 噪声
      是否影响 α/β 及端到端量化精度?

设置: ame1 真实权重。同一批结构化 obs,两种激活源:
  fp32  : FP32 模型普通前向(现状,独立 eager FP32)
  bf16  : FP32 模型在 autocast(bf16) 下前向(模拟真实 update 前向,全链 bf16 传播)
各自 fit_let 200 步 → 比较 (a) α/β 余弦/相对差 (b) swap 后端到端 CosSim vs FP32。

Part B: 漂移版 —— 30 iter fixed-1 伴随校准,采集源 fp32 vs bf16,比末值。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_model, load_ame_ckpt, make_terrain_obs

import copy
import torch

from qrt import swap, calib
from qrt.cocalib import CoCalibrator

PATHS = ["actor.0", "actor.2", "actor.4", "actor.6", "actor_proprio_embedding"]
DEVICE = torch.device("cuda")


def gen_obs(seed, batch=4096, dtype=torch.float32):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    o = make_terrain_obs(torch.float16, batch=batch, gen=g)
    return {k: v.to(dtype) for k, v in o.items()}


@torch.no_grad()
def collect_acts(model, obs, amp):
    feats = {}
    hooks = []
    def mk(p):
        def fn(mod, inp, out):
            feats[p] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        return fn
    for p in PATHS:
        hooks.append(model.get_submodule(p).register_forward_hook(mk(p)))
    ctx = torch.autocast("cuda", torch.bfloat16) if amp else torch.autocast("cuda", enabled=False)
    model.eval()
    with ctx:
        model.act_inference(obs)
    for h in hooks:
        h.remove()
    return feats


@torch.inference_mode()
def eval_cos(model, test_obs, refs):
    dt = next(model.parameters()).dtype
    cs = []
    for o, r in zip(test_obs, refs):
        o = {k: v.to(dt) for k, v in o.items()}
        out = model.act_inference(o)[0].float()
        cs.append(torch.nn.functional.cosine_similarity(
            out.flatten(), r.float().flatten(), dim=0).item())
    return sum(cs) / len(cs)


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")

    ref32 = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    test_obs = [gen_obs(1000 + i) for i in range(4)]
    with torch.inference_mode():
        refs = [ref32.act_inference(o)[0].clone() for o in test_obs]

    calib_obs = gen_obs(7)
    m_src = load_ame_ckpt(build_model(torch.float32, variant="ame"))

    # 检查 bf16 采集是否真的拿到 bf16 量级噪声(打印激活 dtype 与 fp32 的差)
    acts_fp32 = collect_acts(m_src, calib_obs, amp=False)
    acts_bf16 = collect_acts(m_src, calib_obs, amp=True)
    print("\n[激活源差异] (bf16 autocast 前向 vs fp32 前向,同 obs):")
    for p in PATHS:
        d = (acts_fp32[p] - acts_bf16[p]).abs()
        rel = d.mean().item() / acts_fp32[p].abs().mean().clamp_min(1e-9).item()
        print(f"  {p:<26} relΔ={rel:.2e}  (确认 bf16 噪声量级 ~1e-3)")

    # ── Part A: 静态校准对照 ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Part A  静态: fp32 激活 vs bf16 激活 各自 LET 校准")
    print("=" * 72)
    plan_fp32, plan_bf16 = {}, {}
    for p in PATHS:
        lin = m_src.get_submodule(p)
        torch.manual_seed(0)
        a_f, b_f, sz_f = calib.fit_let(acts_fp32[p], lin, n_steps=200)
        torch.manual_seed(0)
        a_b, b_b, sz_b = calib.fit_let(acts_bf16[p], lin, n_steps=200)
        cos = torch.nn.functional.cosine_similarity(
            torch.cat([a_f, b_f]).flatten(), torch.cat([a_b, b_b]).flatten(), dim=0).item()
        rel_a = (a_f - a_b).abs().mean().item() / a_f.abs().mean().item()
        print(f"  {p:<26} α/β cos={cos:.6f}  α relΔ={rel_a:.2e}")
        plan_fp32[p] = {"s_z": None, "alpha": a_f, "beta": b_f, "act_mode": "dynamic"}
        plan_bf16[p] = {"s_z": None, "alpha": a_b, "beta": b_b, "act_mode": "dynamic"}

    m_f = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    swap.apply(m_f, copy.deepcopy(plan_fp32))
    c_f = eval_cos(m_f, test_obs, refs)
    m_b = load_ame_ckpt(build_model(torch.float16, variant="ame"))
    swap.apply(m_b, copy.deepcopy(plan_bf16))
    c_b = eval_cos(m_b, test_obs, refs)
    print(f"\n  端到端 CosSim:  fp32 激活校准={c_f:.8f}   bf16 激活校准={c_b:.8f}")
    print(f"  差异 ΔCosSim = {abs(c_f - c_b):.2e}  "
          f"({'无害,bf16 可用' if abs(c_f - c_b) < 1e-4 else '有影响,需注意'})")

    # ── Part B: 漂移版 fixed-1,采集源 fp32 vs bf16 ───────────────────────────
    print("\n" + "=" * 72)
    print("Part B  漂移 30 iter,fixed-1 伴随校准,采集源 fp32 vs bf16")
    print("=" * 72)
    sd16 = {k: v.clone() for k, v in
            load_ame_ckpt(build_model(torch.float16, variant="ame")).state_dict().items()}
    trainer = load_ame_ckpt(build_model(torch.float32, variant="ame"))
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-4)
    base_plan = calib.make_plan(load_ame_ckpt(build_model(torch.float16, variant="ame")),
                                PATHS, obs_fn=lambda: gen_obs(99, dtype=torch.float16),
                                use_let=True, let_steps=200, act_mode="dynamic", verbose=False)

    def fresh():
        m = load_ame_ckpt(build_model(torch.float16, variant="ame"))
        m.load_state_dict(sd16)
        return m, swap.apply(m, copy.deepcopy(base_plan))

    mF, swF = fresh()
    mB, swB = fresh()
    ccF = CoCalibrator(trainer, swF, tau=0.0, k_max=1); ccF.collect_amp = False
    ccB = CoCalibrator(trainer, swB, tau=0.0, k_max=1); ccB.collect_amp = True

    mb = [gen_obs(200 + i, batch=1024) for i in range(4)]
    def update():
        trainer.train()
        for m_ in mb:
            ao = trainer.actor_obs_normalizer(trainer.get_actor_obs(m_))
            trainer.update_distribution(ao)
            loss = trainer.distribution.mean.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        trainer.eval()

    print(f"{'iter':>4} {'fp32 采集':>14} {'bf16 采集':>14}")
    for it in range(31):
        if it > 0:
            update()
            ccF.step(mb[it % 4]); ccB.step(mb[it % 4])
            ref32.load_state_dict(trainer.state_dict(), strict=False)
            for m_, sw_ in ((mF, swF), (mB, swB)):
                m_.load_state_dict(trainer.state_dict(), strict=False)
                swap.refresh(sw_)
        with torch.inference_mode():
            refs2 = [ref32.act_inference(o)[0].clone() for o in test_obs]
        if it % 10 == 0 or it == 30:
            print(f"{it:>4} {eval_cos(mF, test_obs, refs2):>14.8f} "
                  f"{eval_cos(mB, test_obs, refs2):>14.8f}")
    print("完成。")


if __name__ == "__main__":
    main()

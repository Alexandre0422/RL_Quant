# -*- coding: utf-8 -*-
"""
qrt/tdmpc2/quantize.py — 用 qrt 方法量化 TD-MPC2 世界模型(严格复用主线组件)
══════════════════════════════════════════════════════════════════════════════
把 GLAD 主线的量化工具**原样**搬到 TD-MPC2,按 roofline 两端分而治之:

  rollout(dynamics/reward/Q/π,推理时 batch≈536,计算/激活主导)
      → qrt.QuantLinear(W8A8,_int_mm/IMMA)+ LET(quant.let.LETQuantizer)
      → 经 qrt.swap.apply 替换;正是 GLAD 单步(B=1)永远够不到、
        而 TD-MPC2 在线规划天然命中的 roofline 计算主导端(见 EXP-002)

  encoder(推理时 batch=1,权重访存主导)
      → qrt.wlinear.WeightOnlyLinear(W8A16/W4A16,任意 M 可跑)
      → 经 swap_wlinear 替换;因 _int_mm 在 M≤16 不可用(EXP-001 硬发现),
        B=1 只能走 weight-only

**本文件不重写任何量化逻辑**——LET 用 calib.fit_let,INT8 GEMM 用 QuantLinear,
weight-only 用 WeightOnlyLinear;此处只做「TD-MPC2 特有的层枚举 + 激活采集驱动」
(TD-MPC2 无 GLAD 的 act_inference,故不用 calib.make_plan,自带 plan() 驱动采集)。

约束沿用主线:QuantLinear 要求 in_features%8==0 且 M>16;不满足的层自动跳过并记录
(方法学诚实:不满足的层不假装量化)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qrt import swap, calib, QuantLinear
from qrt.wlinear import swap_wlinear


# ── 层枚举 ────────────────────────────────────────────────────────────────────
def _linear_paths(model: nn.Module, roots: list[str]) -> list[str]:
    """枚举 roots 各子树下的 nn.Linear 点分路径。"""
    paths: list[str] = []
    for root in roots:
        sub = model.get_submodule(root)
        for name, mod in sub.named_modules():
            if isinstance(mod, nn.Linear):
                paths.append(f"{root}.{name}" if name else root)
    return paths


def rollout_linear_paths(wm) -> tuple[list[str], list[str]]:
    """rollout 世界模型里可量化(K%8==0)与跳过的 Linear 路径。
    roots = dynamics / reward / Qs / pi(推理时 batch≈num_samples+num_pi_trajs)。"""
    roots = ["dynamics", "reward", "pi"] + [f"Qs.{i}" for i in range(len(wm.Qs))]
    ok, skip = [], []
    for p in _linear_paths(wm, roots):
        (ok if wm.get_submodule(p).in_features % 8 == 0 else skip).append(p)
    return ok, skip


def encoder_linear_paths(wm) -> list[str]:
    return _linear_paths(wm, ["encoder"])


# ── 激活采集(TD-MPC2 版:由 plan() 驱动,替代 calib.collect_inputs 的 act_inference）──
@torch.no_grad()
def collect_rollout_inputs(wm, obs_fn, paths: list[str],
                           n_calib: int = 8, max_rows: int = 16384) -> dict:
    """在真实 MPPI 规划中 forward-hook 采集各 rollout Linear 的输入(batch≈536)。
    obs_fn() → [obs_dim] 或 [1,obs_dim]。返回 {path: [S,K] fp32}。"""
    feats: dict[str, list[torch.Tensor]] = {p: [] for p in paths}
    cap = max(1, max_rows // n_calib)

    def mk(p):
        def fn(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            if x.shape[0] > cap:
                idx = torch.randint(0, x.shape[0], (cap,), device=x.device)
                x = x[idx]
            feats[p].append(x.float())
        return fn

    hooks = [wm.get_submodule(p).register_forward_hook(mk(p)) for p in paths]
    was_training = wm.training
    wm.eval()
    with torch.inference_mode():
        for _ in range(n_calib):
            wm.plan(obs_fn())
    if was_training:
        wm.train()
    for h in hooks:
        h.remove()
    return {p: torch.cat(v, dim=0) for p, v in feats.items() if v}


# ── rollout 量化:QuantLinear(A8W8)+ LET ─────────────────────────────────────
def quantize_rollout(wm, obs_fn=None, act_mode: str = "dynamic",
                     use_let: bool = True, let_steps: int = 100,
                     verbose: bool = True) -> dict[str, QuantLinear]:
    """把 rollout 的 Linear 换成 QuantLinear(W8A8)。

    act_mode(直接透传 QuantLinear,语义见 qlinear.py):
      "dynamic"       s_z 图内现算、LET α/β 校准得来(推荐:无校准漂移)
      "static"        校准静态 s_z(需 obs_fn)
      "dynamic_full"  零校准(α/β/s_w 全图内);obs_fn 可为 None,LET 退化为动态标准化

    返回 swapped(供 swap.refresh 在权重更新后热刷 INT8 buffer)。
    """
    ok, skip = rollout_linear_paths(wm)
    if verbose:
        print(f"[rollout] 可量化 Linear {len(ok)} 个,跳过(K%8!=0){len(skip)} 个: {skip}")

    if act_mode == "dynamic_full":
        plan = {p: {"s_z": None, "alpha": None, "beta": None,
                    "act_mode": "dynamic_full"} for p in ok}
    else:
        assert obs_fn is not None, f"act_mode={act_mode} 需 obs_fn 采集激活"
        acts = collect_rollout_inputs(wm, obs_fn, ok)
        missing = [p for p in ok if p not in acts]
        if missing:                                    # 采集时未触发 → 不假装量化
            if verbose:
                print(f"  [warn] 无激活样本,跳过: {missing}")
            ok = [p for p in ok if p in acts]
        plan = {}
        for p in ok:
            lin = wm.get_submodule(p)
            if use_let:
                if verbose:
                    print(f"  [LET] {p}  K={lin.in_features} samples={acts[p].shape[0]}")
                alpha, beta, s_z = calib.fit_let(acts[p], lin, n_steps=let_steps)
                plan[p] = {"s_z": s_z, "alpha": alpha, "beta": beta, "act_mode": act_mode}
            else:
                s_z = acts[p].abs().amax().item() / 127.0
                plan[p] = {"s_z": s_z, "alpha": None, "beta": None, "act_mode": act_mode}

    return swap.apply(wm, plan)


# ── encoder 量化:WeightOnlyLinear(W8A16/W4A16,B=1)─────────────────────────
def quantize_encoder(wm, bits: int = 8, verbose: bool = True) -> int:
    """把 encoder 的 Linear 换成 WeightOnlyLinear(权重量化,激活 FP16,任意 M 可跑)。
    B=1 推理下 _int_mm 不可用(EXP-001),weight-only 是唯一兑现权重访存收益的路径。"""
    paths = encoder_linear_paths(wm)
    n = swap_wlinear(wm, paths, bits=bits)
    if verbose:
        print(f"[encoder] WeightOnlyLinear(W{bits}A16) 替换 {n} 个: {paths}")
    return n


def refresh_rollout(swapped: dict[str, QuantLinear]) -> None:
    """权重更新后热刷 rollout INT8 buffer(替代 refit,<0.1ms,graph 安全)。"""
    swap.refresh(swapped)

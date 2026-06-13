# -*- coding: utf-8 -*-
"""
calib.py — 校准管线: 激活采集 → LET 优化 → swap plan
══════════════════════════════════════════════════════════════════════════════
复用主线 quant/let.py 的 LETQuantizer(零修改,block_reconstruction_loss 校准),
输出 swap.apply 可直接消费的 plan。

    plan = make_plan(model, ["actor.0", "actor.2", "actor.4", "actor.6"],
                     use_let=True, obs_fn=my_real_rollout_obs)   # 默认随机 obs
    swapped = swap.apply(model, plan)

obs_fn 接真实 rollout obs 时即解决 TRT 路线已知问题 3(随机校准)。
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from quant.let import LETQuantizer   # 主线工具,只读复用


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_inputs(
    model: nn.Module,
    paths: list[str],
    obs_fn,
    n_batches: int = 8,
    max_rows: int = 16384,
) -> dict[str, torch.Tensor]:
    """forward hook 采集各层输入(行下采样到 max_rows),返回 {path: [S, K] fp32}。"""
    feats: dict[str, list[torch.Tensor]] = {p: [] for p in paths}
    hooks = []

    def mk(p):
        def fn(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            if x.shape[0] > max_rows // n_batches:
                idx = torch.randint(0, x.shape[0],
                                    (max_rows // n_batches,), device=x.device)
                x = x[idx]
            feats[p].append(x.float())
        return fn

    for p in paths:
        hooks.append(model.get_submodule(p).register_forward_hook(mk(p)))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(n_batches):
            model.act_inference(obs_fn())
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return {p: torch.cat(v, dim=0) for p, v in feats.items()}


# ─────────────────────────────────────────────────────────────────────────────
def fit_let(
    x: torch.Tensor,
    linear: nn.Linear,
    n_steps: int = 200,
    lr: float = 5e-3,
    batch_rows: int = 4096,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """LET 校准单层: 返回 (alpha, beta, s_z)。在 fp32 副本上优化。"""
    lin32 = copy.deepcopy(linear).float()
    for p in lin32.parameters():
        p.requires_grad_(False)
    x = x.float()

    let = LETQuantizer(n_channels=lin32.in_features).to(x.device)
    let.initialize_from_activation(x)
    opt = torch.optim.Adam(let.parameters(), lr=lr)
    for i in range(n_steps):
        idx = torch.randint(0, x.shape[0], (min(batch_rows, x.shape[0]),),
                            device=x.device)
        _, loss = let.block_reconstruction_loss(x[idx], lin32)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (i + 1) % 50 == 0:
            print(f"      step {i+1:4d}  recon_loss={loss.item():.3e}")
    s_z = let.calibrate_scale(x)
    return let.alpha.detach().clone(), let.beta.detach().clone(), s_z


# ─────────────────────────────────────────────────────────────────────────────
def make_plan(
    model: nn.Module,
    paths: list[str],
    obs_fn=None,
    use_let: bool = True,
    n_calib: int = 8,
    let_steps: int = 200,
    act_mode: str = "static",
    verbose: bool = True,
) -> dict[str, dict]:
    """一条龙: 采集 → (LET 优化) → s_z 估计 → swap plan。

    use_let=False 时为 naive PTQ 对照(alpha/beta=None,s_z=absmax/127)。
    act_mode(P5): "static"(默认)/"dynamic"(s_z 图内现算,仍需 LET 校准)
                  /"dynamic_full"(α/β/s_w 全图内,**零校准**,obs_fn 可为 None)。
    """
    if act_mode == "dynamic_full":
        return {p: {"s_z": None, "alpha": None, "beta": None,
                    "act_mode": act_mode} for p in paths}

    acts = collect_inputs(model, paths, obs_fn, n_batches=n_calib)
    plan: dict[str, dict] = {}
    for p in paths:
        lin = model.get_submodule(p)
        x = acts[p]
        if use_let:
            if verbose:
                print(f"  [calib/LET] {p}  (K={lin.in_features}, "
                      f"samples={x.shape[0]})")
            alpha, beta, s_z = fit_let(x, lin, n_steps=let_steps, verbose=verbose)
            plan[p] = {"s_z": s_z, "alpha": alpha, "beta": beta, "act_mode": act_mode}
        else:
            s_z = x.abs().amax().item() / 127.0
            if verbose:
                print(f"  [calib/naive] {p}  s_z={s_z:.4g}")
            plan[p] = {"s_z": s_z, "alpha": None, "beta": None, "act_mode": act_mode}
    return plan

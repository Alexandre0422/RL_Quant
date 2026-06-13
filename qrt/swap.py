# -*- coding: utf-8 -*-
"""
swap.py — 模型手术: nn.Linear ↔ QuantLinear 替换/还原/批量刷新
══════════════════════════════════════════════════════════════════════════════
用于层子集消融(对齐 TRT 路线 Phase1/2 的方法学,每配置重编译仅 ~8s 缓存命中):

    plan = calib.make_plan(model, ["actor.0", "actor.2"], use_let=True)
    swapped = swap.apply(model, plan)        # 替换(参数对象转移,optimizer 安全)
    ... rollout / 训练 ...
    swap.refresh(swapped)                    # PPO update 后批量 requantize_
    swap.restore(model, swapped)             # 还原为原 nn.Linear(消融下一组)

注意: 替换不复制参数 —— QuantLinear 持有源 Linear 的同一 weight/bias Parameter,
因此 swap 前后创建的 optimizer 均有效;restore 把同一参数对象放回新 nn.Linear 壳。
"""
from __future__ import annotations

import torch.nn as nn

from .qlinear import QuantLinear


def get_module(model: nn.Module, path: str) -> nn.Module:
    return model.get_submodule(path)


def set_module(model: nn.Module, path: str, mod: nn.Module) -> None:
    if "." in path:
        parent_path, name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
    else:
        parent, name = model, path
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = mod
    else:
        setattr(parent, name, mod)


def apply(model: nn.Module, plan: dict[str, dict]) -> dict[str, QuantLinear]:
    """按 plan 替换。plan: {path: {"s_z": float, "alpha": Tensor|None, "beta": Tensor|None}}"""
    swapped: dict[str, QuantLinear] = {}
    for path, cfg in plan.items():
        lin = get_module(model, path)
        assert isinstance(lin, nn.Linear), f"{path} 不是 nn.Linear: {type(lin)}"
        ql = QuantLinear(lin, s_z=cfg.get("s_z"),
                         alpha=cfg.get("alpha"), beta=cfg.get("beta"),
                         act_mode=cfg.get("act_mode", "static"))
        # 关键: 新建 nn.Module 默认 training=True,必须继承源层状态,
        # 否则 eval 模型上 swap 后 forward 静默走 FP 训练分支(INT8 不生效)
        ql.train(lin.training)
        set_module(model, path, ql)
        swapped[path] = ql
    return swapped


def restore(model: nn.Module, swapped: dict[str, QuantLinear]) -> None:
    """还原为 nn.Linear(参数对象原样放回,权重保持当前训练值)。"""
    for path, ql in swapped.items():
        lin = nn.Linear(ql.in_features, ql.out_features,
                        bias=ql.bias is not None, device=ql.weight.device,
                        dtype=ql.weight.dtype)
        lin.weight = ql.weight
        if ql.bias is not None:
            lin.bias = ql.bias
        # 同样继承 train/eval 状态: 否则 restore 出的 Linear 默认 train=True,
        # 下一轮 swap.apply 会把错误状态传给新 QuantLinear(INT8 静默失效)
        lin.train(ql.training)
        set_module(model, path, lin)


def refresh(swapped: dict[str, QuantLinear]) -> None:
    """PPO update 后批量刷新 INT8 buffer(替代 TRT refit,<0.1ms)。"""
    for ql in swapped.values():
        ql.requantize_()

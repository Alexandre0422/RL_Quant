# -*- coding: utf-8 -*-
"""
qrt/tdmpc2/qnormed.py — QuantNormedLinear:量化 TD-MPC2 的 NormedLinear(保留 ln/act)
══════════════════════════════════════════════════════════════════════════════
TD-MPC2 的 _dynamics/_reward/_pi 是 `nn.Sequential(NormedLinear...)`,末层可能是
plain `nn.Linear`。NormedLinear 是 `nn.Linear` 子类,forward = Linear→dropout→LayerNorm→act。
若直接用 GLAD 的 swap 把它换成 QuantLinear,会**丢掉 LayerNorm+act**(等于改了网络)。

故本模块提供 `QuantNormedLinear(QuantLinear)`:复用 QuantLinear 的 W8A8/_int_mm/LET
matmul,再补回 NormedLinear 的 dropout→ln→act epilogue。plain 末层仍用原 QuantLinear。

**这是「双网络」里施加在推理副本上的模块级零件**——只在冻结的 fp16 副本上、只走 INT8
分支(推理);master 侧仍是原 NormedLinear。参见 integration.accelerate_tdmpc2。

命名:QuantNormedLinear 直接持有 weight/bias(经 QuantLinear)与 .ln/.act(从 NormedLinear
转移),与 master 的 NormedLinear **同名**(`<seq>.<i>.weight` / `<seq>.<i>.ln.weight`),
故每 iter sync 时 named_parameters 可按名配对拷贝。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qrt.qlinear import QuantLinear


def _is_normed_linear(mod: nn.Module) -> bool:
    """鸭子判定 TD-MPC2 NormedLinear(不 import tdmpc2):nn.Linear 子类且带 .ln/.act。"""
    return (isinstance(mod, nn.Linear) and mod.__class__ is not nn.Linear
            and hasattr(mod, "ln") and hasattr(mod, "act"))


class QuantNormedLinear(QuantLinear):
    """NormedLinear 的量化替身:量化 matmul(W8A8+LET),保留 dropout→ln→act。

    继承 QuantLinear 拿到:weight/bias 转移、_int_mm 前向、LET(α/β)、requantize_、
    act_mode(static/dynamic/dynamic_full)。仅重写 forward 补 epilogue。
    """

    def __init__(self, normed: nn.Linear, **kw):
        super().__init__(normed, **kw)                 # 转移 weight/bias,建 INT8 buffer
        self.ln = normed.ln                            # 同名转移(sync 配对)
        self.act = normed.act
        self.dropout = getattr(normed, "dropout", None)

    def forward(self, x):
        y = super().forward(x)                         # 量化 matmul(推理)/ F.linear(训练)
        if self.dropout is not None and torch.is_grad_enabled():
            y = self.dropout(y)                        # dropout 仅训练态(推理恒等)
        return self.act(self.ln(y))

    def extra_repr(self):
        return "NormedLinear+" + super().extra_repr()


def swap_rollout_int8(model: nn.Module,
                      roots=("_dynamics", "_reward", "_pi"),
                      act_mode: str = "dynamic_full",
                      let_plan: dict | None = None,
                      verbose: bool = True) -> tuple[dict, list]:
    """把 rollout 各 Sequential 里的 NormedLinear/plain-Linear 原地换成量化版。

    只动 rollout 世界模型(_dynamics/_reward/_pi);encoder 与 vmap 的 _Qs 不在此(见
    integration docstring 的 TODO)。约束沿用 QuantLinear:in_features%8==0(否则跳过并记录,
    不假装量化);推理时 batch=N≈536>16 满足 _int_mm 的 M>16。

    act_mode:
      "dynamic_full" 零校准(α/β/s_w 全图内现算,requantize_ 为空操作)——默认,patch 免校准数据
      "dynamic"/"static" 需 let_plan[path]={"s_z","alpha","beta"}(由 calib.fit_let 得),
                         对 rollout 激活做真 LET;见 integration 的 calib_obs 选项
    返回 (swapped{path: module}, skipped[path])。
    """
    swapped, skipped = {}, []
    for root in roots:
        if not hasattr(model, root):
            continue
        seq = getattr(model, root)
        for idx in range(len(seq)):
            child = seq[idx]
            path = f"{root}.{idx}"
            is_normed = _is_normed_linear(child)
            is_plain = isinstance(child, nn.Linear) and not is_normed
            if not (is_normed or is_plain):
                continue                               # SimNorm/Dropout/... 跳过
            if child.in_features % 8 != 0:
                skipped.append(path)
                continue
            cfg = (let_plan or {}).get(path, {})
            kw = dict(act_mode=act_mode, s_z=cfg.get("s_z"),
                      alpha=cfg.get("alpha"), beta=cfg.get("beta"))
            q = (QuantNormedLinear(child, **kw) if is_normed
                 else QuantLinear(child, **kw))
            q.train(child.training)                    # 继承 train/eval 态(INT8 生效关键)
            seq[idx] = q
            swapped[path] = q
    if verbose:
        print(f"[rollout-int8] 换 {len(swapped)} 层"
              f"({sum(isinstance(m, QuantNormedLinear) for m in swapped.values())} NormedLinear),"
              f"跳过(K%8!=0){len(skipped)}: {skipped}")
    return swapped, skipped

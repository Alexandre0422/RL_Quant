"""
observer.py — 激活范围统计观测器

用于 QAT 前的校准阶段，统计激活值的动态范围以初始化步长。
也用于 PTQ 静态校准。

三种观测器：
    MinMaxObserver      : 记录全局 min/max（对 outlier 敏感，简单）
    EMAObserver         : 指数移动平均 min/max（更稳定）
    PercentileObserver  : 用百分位数 clip outlier（推荐用于激活量化）
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MinMaxObserver(nn.Module):
    """全局 min/max 观测器。简单直接，对 outlier 敏感。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.num_batches = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.min_val = torch.min(self.min_val, x.detach().min())
        self.max_val = torch.max(self.max_val, x.detach().max())
        self.num_batches += 1
        return x   # pass-through，不修改数据

    def compute_scale(self, nbits: int = 8, symmetric: bool = True) -> torch.Tensor:
        """根据统计范围计算量化步长。"""
        if symmetric:
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            return abs_max / (2 ** (nbits - 1) - 1)
        else:
            return (self.max_val - self.min_val) / (2 ** nbits - 1)

    def reset(self):
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches = 0


class EMAObserver(nn.Module):
    """
    指数移动平均 min/max 观测器。
    比 MinMaxObserver 更稳定，减少单 batch outlier 的影响。

    Args:
        momentum: EMA 动量（0.01 = 慢速更新，更稳定）
    """

    def __init__(self, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_min = x.detach().min()
        batch_max = x.detach().max()
        if not self.initialized:
            self.min_val.copy_(batch_min)
            self.max_val.copy_(batch_max)
            self.initialized = True
        else:
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * batch_min
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * batch_max
        return x

    def compute_scale(self, nbits: int = 8, symmetric: bool = True) -> torch.Tensor:
        if symmetric:
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            return abs_max / (2 ** (nbits - 1) - 1)
        else:
            return (self.max_val - self.min_val) / (2 ** nbits - 1)


class PercentileObserver(nn.Module):
    """
    百分位数 clip 观测器。
    clip outlier 后再统计范围，适合激活分布有长尾的场景（常见于 MHA）。

    Args:
        lower_pct: 下分位数（默认 0.1%）
        upper_pct: 上分位数（默认 99.9%）
    """

    def __init__(self, lower_pct: float = 0.001, upper_pct: float = 0.999):
        super().__init__()
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self._buffer: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._buffer.append(x.detach().flatten().cpu())
        return x

    def compute_scale(self, nbits: int = 8, symmetric: bool = True) -> torch.Tensor:
        all_vals = torch.cat(self._buffer)
        lo = torch.quantile(all_vals, self.lower_pct)
        hi = torch.quantile(all_vals, self.upper_pct)
        if symmetric:
            abs_max = max(lo.abs().item(), hi.abs().item())
            return torch.tensor(abs_max / (2 ** (nbits - 1) - 1))
        else:
            return (hi - lo) / (2 ** nbits - 1)

    def reset(self):
        self._buffer.clear()

"""
fake_quant.py — 核心量化算子

FakeQuantize：在前向时把浮点值量化再反量化（模拟 INT8 舍入误差），
              反向时用 STE（Straight-Through Estimator）直接传递梯度。

数学定义（对称均匀量化）：
    x_q = clamp(round(x / s), -2^(b-1), 2^(b-1)-1)   ← INT b-bit integer
    x̂   = x_q * s                                       ← dequantize to float

    ∂L/∂x  ≈  ∂L/∂x̂  （STE：直接透传，忽略 round 的不可导性）
    ∂L/∂s      见 lsq.py（LSQ 中步长 s 有解析梯度）

使用示例：
    q = FakeQuantize(nbits=8, symmetric=True)
    x_hat = q(x)    # 前向：模拟量化；反向：STE
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 底层 STE round 函数（被 LSQQuantizer 等复用）
# ---------------------------------------------------------------------------
class _STERound(torch.autograd.Function):
    """round() 的 STE 包装：前向取整，反向恒等。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad   # straight-through


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


# ---------------------------------------------------------------------------
# FakeQuantize：固定步长的 fake-quant 模块
# ---------------------------------------------------------------------------
class FakeQuantize(nn.Module):
    """
    带 STE 的 fake-quant 算子。步长 s 固定（不可学习）。
    若需要可学习步长，使用 LSQQuantizer。

    Args:
        nbits      : 量化位宽（默认 8）
        symmetric  : True = 对称量化（zero_point=0），False = 非对称
        per_channel: True = 每个 output channel 独立步长（仅权重量化用）
        ch_axis    : per_channel 时对应的 tensor 轴（Conv weight 通常为 0）
    """

    def __init__(
        self,
        nbits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        ch_axis: int = 0,
    ):
        super().__init__()
        self.nbits = nbits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis

        # 量化范围
        if symmetric:
            self.q_min = -(2 ** (nbits - 1))
            self.q_max =  (2 ** (nbits - 1)) - 1
        else:
            self.q_min = 0
            self.q_max = (2 ** nbits) - 1

        # 由 observer 或 LSQ 写入
        self.register_buffer("scale",      torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.enabled = True   # 可在推理 benchmark 时临时关闭

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        s  = self.scale
        zp = self.zero_point
        # 前向：量化 + 反量化（模拟 INT8 舍入）
        x_q  = ste_round(x / s + zp).clamp(self.q_min, self.q_max)
        x_hat = (x_q - zp) * s
        return x_hat

    def extra_repr(self) -> str:
        return (f"nbits={self.nbits}, symmetric={self.symmetric}, "
                f"per_channel={self.per_channel}, enabled={self.enabled}")

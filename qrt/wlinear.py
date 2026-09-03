# -*- coding: utf-8 -*-
"""
wlinear.py — WeightOnlyLinear: W8A16 / W4A16 权重量化 Linear(小 batch 可用)
══════════════════════════════════════════════════════════════════════════════
动机(EXP-001 硬发现): `torch._int_mm(q, W8ᵀ)`(QuantLinear 的 INT8 GEMM 后端,
cuBLASLt IMMA)要求 GEMM 的 **M(=batch)> 16**,B≤16 直接抛 RuntimeError。而
roofline **预测** INT8 linear 恰恰只在**小 batch**(板端 B=1,权重访存主导)才
翻转为收益 —— 那个区间 `_int_mm` 原语**不可达**。要兑现小 batch 的访存收益,
必须绕开 `_int_mm`,换 **weight-only / GEMV** 路线。

本模块即该路线:
  • 权重存 INT8(W8A16)或 packed-INT4(W4A16),**激活保持 FP16**;
  • 自写 Triton tiled matmul:int8/int4 权重在寄存器内反量化为 fp16,再做
    fp16 `tl.dot` —— 这是普通 MMA(非 IMMA),**任意 M 都可跑(含 B=1)**;
  • 收益来源是**权重字节减半/减到 1/4**(DRAM 流量),正是小 batch 权重访存
    主导区的 roofline 杠杆。

与 QuantLinear(W8A8)的分工:
  QuantLinear   激活也量化(A8),靠 IMMA 吃**算力**,只在大 M、算力主导时划算;
  WeightOnlyLin 只量化权重(A16),靠**访存**省字节,小 M、访存主导时划算。
两者覆盖 roofline 的两端,是「INT8 价值由逐算子 roofline 位置决定」论点的两个工具。

注:weight-only 不引入 LET。LET/SmoothQuant 是把激活离群值搬进权重以便**激活
量化**;此处激活全精度,per-output-channel 的权重 scale 已吸收每通道幅度,LET
无额外增益,故不实现(避免暗示虚假价值)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════════════════
# Triton kernels — int8/int4 权重在寄存器反量化 → fp16 tl.dot(任意 M 可跑)
# ════════════════════════════════════════════════════════════════════════════

_AUTOTUNE = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE, key=["M", "N", "K"])
@triton.jit
def _k_w8a16(
    x_ptr,        # [M,K] fp16
    w_ptr,        # [N,K] int8(per-output-channel 对称量化)
    s_ptr,        # [N] fp32  s_w
    b_ptr,        # [N] fp16  bias
    y_ptr,        # [M,N] fp16
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = (offs_k[None, :] + k0) < K
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & km, other=0.0)      # [BM,BK] fp16
        wm = (offs_k[:, None] + k0) < K
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & wm, other=0)        # [BK,BN] int8
        acc += tl.dot(a, w.to(tl.float16), out_dtype=tl.float32)            # fp16 MMA(任意 M)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)                  # [BN] fp32
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = acc * s[None, :] + b[None, :]                                        # per-channel dequant + bias
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, y.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(configs=_AUTOTUNE, key=["M", "N", "K"])
@triton.jit
def _k_w4a16(
    x_ptr,        # [M,K] fp16
    w_ptr,        # [N,K//2] uint8  每字节低/高 nibble = 偶/奇 k 的有符号 int4
    s_ptr,        # [N] fp32
    b_ptr,        # [N] fp16
    y_ptr,        # [M,N] fp16
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wkp,                                                   # 沿 packed-K 的 stride
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        gk = offs_k + k0                                                     # [BK] 全局 k
        km = gk < K
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & km[None, :], other=0.0)
        # packed-int4:byte = w[n, gk//2];偶 k 取低 nibble,奇 k 取高 nibble
        byte_col = gk // 2
        shift = (gk % 2) * 4
        wp_ptrs = (w_ptr + offs_n[None, :] * stride_wn
                   + byte_col[:, None] * stride_wkp)                          # [BK,BN]
        packed = tl.load(wp_ptrs, mask=(offs_n[None, :] < N) & km[:, None], other=0)
        nib = (packed >> shift[:, None]) & 0xF                               # [BK,BN] uint8 [0,15]
        w = tl.where(nib >= 8, nib.to(tl.int32) - 16, nib.to(tl.int32))      # 还原符号 [-8,7]
        acc += tl.dot(a, w.to(tl.float16), out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk

    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = acc * s[None, :] + b[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, y.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ════════════════════════════════════════════════════════════════════════════
# custom op 注册(dynamo/inductor 可入图,cudagraph 安全)
# ════════════════════════════════════════════════════════════════════════════

@torch.library.custom_op("qrt::w8a16_linear", mutates_args=(), device_types="cuda")
def _op_w8a16(x: torch.Tensor, w8: torch.Tensor,
              s_w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w8.shape[0]
    y = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    _k_w8a16[grid](x, w8, s_w, bias, y, M, N, K,
                   x.stride(0), x.stride(1), w8.stride(0), w8.stride(1))
    return y


@_op_w8a16.register_fake
def _(x, w8, s_w, bias):
    return x.new_empty((x.shape[0], w8.shape[0]), dtype=torch.float16)


@torch.library.custom_op("qrt::w4a16_linear", mutates_args=(), device_types="cuda")
def _op_w4a16(x: torch.Tensor, w4: torch.Tensor, s_w: torch.Tensor,
              bias: torch.Tensor, K: int) -> torch.Tensor:
    M = x.shape[0]
    N = w4.shape[0]
    y = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    _k_w4a16[grid](x, w4, s_w, bias, y, M, N, K,
                   x.stride(0), x.stride(1), w4.stride(0), w4.stride(1))
    return y


@_op_w4a16.register_fake
def _(x, w4, s_w, bias, K):
    return x.new_empty((x.shape[0], w4.shape[0]), dtype=torch.float16)


# ════════════════════════════════════════════════════════════════════════════
# WeightOnlyLinear 模块(nn.Linear 的权重量化替身)
# ════════════════════════════════════════════════════════════════════════════

class WeightOnlyLinear(nn.Module):
    """W8A16 / W4A16:权重 per-output-channel 对称量化,激活 FP16。

    - train(梯度态)走 `F.linear` 全精度(weight/bias 是从源 Linear 转移的同一
      Parameter,optimizer 引用不失效);
    - eval(无梯度)走 Triton weight-only kernel,**任意 batch 可跑(含 B=1)**;
    - requantize_() 从当前 weight 重算量化 buffer(GPU,graph 安全,热更可用)。
    """

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        if bits not in (8, 4):
            raise ValueError(f"bits 仅支持 8 或 4,收到 {bits}")
        N, K = linear.out_features, linear.in_features
        if bits == 4 and K % 2 != 0:
            raise ValueError(f"INT4 packing 要求 in_features 为偶数,收到 {K}")
        self.bits = bits
        self.in_features = K
        self.out_features = N
        self.qmax = 127 if bits == 8 else 7
        self.qmin = -128 if bits == 8 else -8
        dev = linear.weight.device

        # 训练路径:转移同一 Parameter(optimizer 引用不失效)
        self.weight = linear.weight
        self.bias = linear.bias

        self.register_buffer("s_w", torch.ones(N, dtype=torch.float32, device=dev))
        self.register_buffer("bias_f", torch.zeros(N, dtype=torch.float16, device=dev))
        if bits == 8:
            self.register_buffer("qw", torch.zeros(N, K, dtype=torch.int8, device=dev))
        else:
            self.register_buffer("qw", torch.zeros(N, K // 2, dtype=torch.uint8, device=dev))
        self.requantize_()

    @torch.no_grad()
    def requantize_(self):
        W = self.weight.detach().float()                                    # [N,K]
        s_w = W.abs().amax(dim=1).clamp_min(1e-8) / self.qmax               # [N] 对称 scale
        q = torch.clamp(torch.round(W / s_w[:, None]), self.qmin, self.qmax)
        if self.bits == 8:
            self.qw.copy_(q.to(torch.int8))
        else:
            qi = q.to(torch.int8)
            u = (qi & 0xF).to(torch.uint8)                                  # 二补码低 nibble
            packed = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)       # 偶→低,奇→高
            self.qw.copy_(packed)
        self.s_w.copy_(s_w)
        if self.bias is not None:
            self.bias_f.copy_(self.bias.detach().half())

    def forward(self, x):
        if torch.is_grad_enabled():
            return F.linear(x, self.weight, self.bias)
        orig = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.float16).contiguous()
        if self.bits == 8:
            y = torch.ops.qrt.w8a16_linear(x2, self.qw, self.s_w, self.bias_f)
        else:
            y = torch.ops.qrt.w4a16_linear(x2, self.qw, self.s_w, self.bias_f,
                                           self.in_features)
        return y.reshape(*orig[:-1], self.out_features).to(x.dtype)

    @torch.no_grad()
    def forward_ref(self, x):
        """纯 PyTorch 参照(反量化权重 → F.linear),用于隔离 kernel bug 与量化方案 bug。"""
        if self.bits == 8:
            w = self.qw.float() * self.s_w[:, None]
        else:
            lo = (self.qw & 0xF).to(torch.int16)
            hi = (self.qw >> 4).to(torch.int16)
            lo = torch.where(lo >= 8, lo - 16, lo)
            hi = torch.where(hi >= 8, hi - 16, hi)
            qi = torch.empty(self.out_features, self.in_features,
                             dtype=torch.float32, device=self.qw.device)
            qi[:, 0::2] = lo.float()
            qi[:, 1::2] = hi.float()
            w = qi * self.s_w[:, None]
        return F.linear(x, w.to(x.dtype),
                        self.bias_f.to(x.dtype) if self.bias is not None else None)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.bits}, W{self.bits}A16")


def swap_wlinear(model, names, bits: int = 8):
    """把 model 中给定点分路径的 nn.Linear 原地替换为 WeightOnlyLinear,返回替换数。"""
    swapped = 0
    for name in names:
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        lin = getattr(parent, leaf)
        wo = WeightOnlyLinear(lin, bits=bits)
        wo.train(lin.training)
        setattr(parent, leaf, wo)
        swapped += 1
    return swapped

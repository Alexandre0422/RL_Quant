# -*- coding: utf-8 -*-
"""
qlinear.py — QuantLinear: W8A8 静态量化 Linear(LET 前置,训练/推理双路径)
══════════════════════════════════════════════════════════════════════════════
推理路径(eval, inductor 图内全融合):

    z  = α⊙x + β                    LET 等价变换(α=None 时跳过,编译期特化)
    q  = clamp(round(z / s_z))       per-tensor 静态 scale,INT8
    Y' = _int_mm(q, W8ᵀ)            aten 原生 INT8 GEMM(cuBLASLt IMMA)
    y  = Y'·(s_w·s_z) + b'           fp32 dequant(防 int32→fp16 溢出,实测教训)
    其中 W8 = quant(W/α),b' = b − (W/α)β   —— 与 TRT 路线的
    Mul(α)→Add(β)→Q→DQ→GEMM(W') ONNX 插桩在数学上逐项等价(OmniQuant §3)

训练路径(train):
    F.linear(x, self.weight, self.bias) —— weight/bias 是从源 nn.Linear
    **转移来的同一 Parameter 对象**,optimizer 引用保持有效;
    P3 在线 QAT 在此分支插 LETQuantizer fake-quant 即可。

热更新(替代 TRT refit):
    requantize_() 从当前 weight/bias 重算 W8/b'/scale 并 copy_ 进 buffer,
    data_ptr 不变 → CUDA Graph / cudagraph trees 下 replay 立即生效。

约束: in_features % 8 == 0(_int_mm 硬约束,GLAD 全部满足);
      out_features 任意(内部 pad 到 8 的倍数,输出切片);
      输入须为 2D [M, K] 且 M > 16(B=4096 / minibatch=1024 均满足)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):

    def __init__(self, linear: nn.Linear, s_z: float | None = None,
                 alpha: torch.Tensor | None = None,
                 beta: torch.Tensor | None = None,
                 act_mode: str = "static"):
        """act_mode(P5,激活 scale 策略;torchao float8 的 dynamic 经验移植):
        "static"       校准的 s_z(0-d buffer)+ EMA/refresh 维护(默认,原行为)
        "dynamic"      s_z = absmax(z)/127 图内现算(inductor 融合);α/β 仍为
                       LET 校准值,权重静态 INT8 —— 校准漂移问题消失
        "dynamic_full" α=1/σ、β=−μ/σ 亦由 batch 统计图内现算,权重折叠与量化
                       同样进图(每步重做)—— 零校准;LET 退化为动态标准化,
                       科研对照用("学习的 α" vs "统计的 α")
        """
        super().__init__()
        if act_mode not in ("static", "dynamic", "dynamic_full"):
            raise ValueError(f"unknown act_mode: {act_mode}")
        N, K = linear.out_features, linear.in_features
        if K % 8 != 0:
            raise ValueError(f"in_features={K} 不是 8 的倍数,_int_mm 不支持")
        self.in_features = K
        self.out_features = N
        self.n_pad = (-N) % 8                      # pad 到 8 的倍数
        self.act_mode = act_mode
        dev = linear.weight.device

        # 训练路径参数: 转移同一 Parameter 对象(optimizer 引用不失效)
        self.weight = linear.weight
        self.bias = linear.bias

        self.has_let = alpha is not None and act_mode != "dynamic_full"
        if self.has_let:
            self.register_buffer("alpha", alpha.detach().float().clamp_min(1e-8).to(dev))
            self.register_buffer("beta",
                                 beta.detach().float().to(dev) if beta is not None
                                 else torch.zeros(K, device=dev))
        # 静态激活 scale(0-d buffer → 可热更,不被 inductor 烘成常量)
        if s_z is None:
            s_z = 1.0                              # dynamic 模式下不参与计算
        self.register_buffer("s_z", torch.tensor(float(max(s_z, 1e-8)), device=dev))
        self.register_buffer("inv_sz", torch.tensor(1.0 / float(max(s_z, 1e-8)), device=dev))

        Np = N + self.n_pad
        self.register_buffer("w8", torch.zeros(Np, K, dtype=torch.int8, device=dev))
        self.register_buffer("dq_scale", torch.ones(Np, dtype=torch.float32, device=dev))   # static: s_w·s_z
        self.register_buffer("s_w", torch.ones(Np, dtype=torch.float32, device=dev))        # dynamic: 单独存 s_w
        self.register_buffer("b_prime", torch.zeros(Np, dtype=torch.float32, device=dev))
        self.requantize_()

    # ── 离线/在线 权重量化(LET 折叠 → per-channel 对称 INT8)─────────────────
    @torch.no_grad()
    def requantize_(self):
        """从当前 self.weight/self.bias 刷新 INT8 buffer(全 GPU,<0.1ms,graph 安全)。
        dynamic_full 模式下权重量化在前向图内进行,此处为空操作。"""
        if self.act_mode == "dynamic_full":
            return
        W = self.weight.detach().float()                       # [N, K]
        b = (self.bias.detach().float() if self.bias is not None
             else torch.zeros(self.out_features, device=W.device))
        if self.has_let:
            W = W / self.alpha[None, :]                        # W' = W/α
            b = b - (W * self.beta[None, :]).sum(dim=1)        # b' = b − W'β
        s_w = W.abs().amax(dim=1).clamp_min(1e-8) / 127.0      # [N]
        w8 = torch.clamp(torch.round(W / s_w[:, None]), -128, 127).to(torch.int8)
        N = self.out_features
        self.w8[:N].copy_(w8)
        self.dq_scale[:N].copy_(s_w * self.s_z)
        self.s_w[:N].copy_(s_w)
        self.b_prime[:N].copy_(b)

    @torch.no_grad()
    def set_act_scale_(self, s_z: float):
        """更新静态激活 scale(在线校准用),随后需 requantize_()。"""
        self.s_z.fill_(float(max(s_z, 1e-8)))
        self.inv_sz.fill_(1.0 / float(max(s_z, 1e-8)))

    # ── 前向 ─────────────────────────────────────────────────────────────────
    def forward(self, x):
        # 分支按梯度态而非 training: rsl_rl 的 rollout 是 train 态(Gumbel 探索)
        # + inference_mode —— 应走 INT8;PPO update 是梯度态 —— 走可导的 FP 路径
        if torch.is_grad_enabled():
            # 训练路径: 全精度(P3 QAT: 在此插 fake-quant,与推理共享 α/β/s_z)
            return F.linear(x, self.weight, self.bias)

        if self.act_mode == "dynamic_full":
            y = self._forward_dynamic_full(x)
            return y

        z = x
        if self.has_let:
            z = z * self.alpha + self.beta            # fp16×fp32 → fp32,融进 quant kernel
        if self.act_mode == "dynamic":
            # P5: s_z 图内现算(absmax 归约被 inductor 融合;统计对象是 z,
            # 与 LET 语义一致)。校准 s_z / EMA / 漂移问题全部消失。
            s_z = z.detach().abs().amax().clamp_min(1e-8) * (1.0 / 127.0)
            q = torch.clamp(torch.round(z / s_z), -128.0, 127.0).to(torch.int8)
            y = torch._int_mm(q, self.w8.t())
            y = y.float() * (self.s_w * s_z) + self.b_prime
        else:                                         # static(原行为)
            q = torch.clamp(torch.round(z * self.inv_sz), -128.0, 127.0).to(torch.int8)
            y = torch._int_mm(q, self.w8.t())         # [M, Np] int32; w8.t() 列主序
            y = y.float() * self.dq_scale + self.b_prime   # fp32 dequant(防溢出)
        if self.n_pad:
            y = y[:, : self.out_features]
        return y.to(x.dtype)

    def _forward_dynamic_full(self, x):
        """P5 dynamic_full: α/β 由 batch 统计现算,权重折叠+量化亦在图内 —— 零校准。
        LET 退化为动态标准化(只优化激活侧);权重每步重量化(~10μs/层,仅 rollout)。

        恒等性: α=1/σ, β=−μ/σ ⇒ z=(x−μ)/σ, W'=W·σ,
                b' = b − W'β = b + Σₖ W'[n,k]·μ[k]/σ[k] = b + Σₖ W[n,k]·μ[k]
        验证:  W'z+b' = Wσ·(x−μ)/σ + b + Wμ = Wx + b ✓
        """
        xf = x.float()
        mu = xf.mean(dim=0)                           # [K]
        sigma = xf.std(dim=0).clamp_min(1e-5)         # [K]
        z = (xf - mu) / sigma
        s_z = z.detach().abs().amax().clamp_min(1e-8) * (1.0 / 127.0)
        q = torch.clamp(torch.round(z / s_z), -128.0, 127.0).to(torch.int8)

        Wp = self.weight.detach().float() * sigma[None, :]     # W' = W·σ  [N,K]
        s_w = Wp.abs().amax(dim=1).clamp_min(1e-8) * (1.0 / 127.0)
        w8 = torch.clamp(torch.round(Wp / s_w[:, None]), -128.0, 127.0).to(torch.int8)
        b = self.bias.detach().float() if self.bias is not None else 0.0
        b_prime = b + (Wp * (mu / sigma)[None, :]).sum(dim=1)  # = b + W@μ
        if self.n_pad:                                         # _int_mm 要求 N%8==0
            w8 = F.pad(w8, (0, 0, 0, self.n_pad))
            s_w = F.pad(s_w, (0, self.n_pad), value=1.0)
            b_prime = F.pad(b_prime, (0, self.n_pad))

        y = torch._int_mm(q, w8.t())
        y = y.float() * (s_w * s_z) + b_prime
        if self.n_pad:
            y = y[:, : self.out_features]
        return y.to(x.dtype)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"let={self.has_let}, s_z={self.s_z.item():.4g}, pad={self.n_pad}")

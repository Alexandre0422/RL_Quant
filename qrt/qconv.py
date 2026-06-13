# -*- coding: utf-8 -*-
"""
qconv.py — P2: GLAD CNN 链的定制 Triton kernel(INT8 implicit-GEMM conv2)
══════════════════════════════════════════════════════════════════════════════
背景(PLAN §P2): compile 后 conv+BN 链仍占单步 ~0.8ms(46%),但理论访存下限
仅 ~0.2ms —— inductor 通用 conv template 对 GLAD 的怪异形状(空间 11×17 极小、
通道 3→16→64 极小、B=4096 巨大)效率低。本模块用两个形状特化 kernel 替换整条链:

  kernel1  conv1(5×5,s2,p2,fp16 tl.dot)+ bias + ReLU + BN1 + quantize
           [B,21,33,3] fp16 → z8 [B*187,16] int8       (中间激活访存减半)
  kernel2  conv2(3×3,s1,p1,implicit-GEMM,INT8 tl.dot/IMMA)+ dequant
           + bias + ReLU + BN2 → [B*187,64] fp16
           直接产出 local_features 行主序布局,permute/reshape 变为零拷贝视图

融合语义(与 map_cnn = [Conv,ReLU,BN,Conv,ReLU,BN] 逐项等价,eval 态):
  z  = quant( BN1(relu(conv1(x)+b1)) ; s_z )          ← qa=a1/s_z, qc=c1/s_z
  y  = BN2( relu( deq(int8gemm(z,W8)) + b2 ) )         ← deq=s_w·s_z per-channel

LET 扩展(P2.5,公式已就位): conv2 输入的 per-channel α/β 直接并入 BN1 仿射
(a1←a1·α, c1←c1·α+β),W2 ← W2/α(requantize_ 中折叠),零额外运行时成本。

形状硬编码为 GLAD 配置(33×21 地形图,D=64);其它形状需改 constexpr。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# GLAD 形状常量
H1, W1, C1 = 21, 33, 3          # conv1 输入 (NHWC)
H2, W2, C2 = 11, 17, 16         # conv1 输出 / conv2 输入
P = H2 * W2                     # 187 token
N2 = 64                         # conv2 输出通道
# tl.arange 必须为 2 的幂 → K 维 pad(无效位 mask=0,权重 pad 0,算力过剩无碍)
K1 = 128                        # conv1 packed K: 8(kh,前5有效) × 16(kw*3+c,前15有效)
K2S = 64                        # conv2 每个 kh 步的 K: kw(3)*16c=48 → pad 64,共 3 步


# ════════════════════════════════════════════════════════════════════════════
# Triton kernels
# ════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _k_conv1_fused_q(
    x_ptr,       # [B,21,33,3] fp16 contiguous
    w_ptr,       # [128,16] fp16 packed(k=kh*16+r; kh<5 且 r<15 有效: kw=r//3,c=r%3)
    b_ptr,       # [16] fp32 conv1 bias
    qa_ptr,      # [16] fp32 = a1/s_z(BN1 scale / 激活 scale)
    qc_ptr,      # [16] fp32 = c1/s_z
    out_ptr,     # [B*187,16] int8
    M,           # B*187
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)            # [M_]
    m_mask = m < M
    b = m // 187
    p_ = m % 187
    oh = p_ // 17
    ow = p_ % 17
    ih0 = oh * 2 - 2                                     # stride 2, pad 2
    iw0 = ow * 2 - 2

    k = tl.arange(0, 128)                                # [K1] pad 至 2 的幂
    kh = k // 16
    r = k % 16
    kw = r // 3
    c = r % 3
    k_valid = (kh < 5) & (r < 15)

    ih = ih0[:, None] + kh[None, :]                      # [M_,K1]
    iw = iw0[:, None] + kw[None, :]
    in_bounds = (ih >= 0) & (ih < 21) & (iw >= 0) & (iw < 33)
    addr = (b[:, None] * (21 * 33 * 3)
            + ih * (33 * 3) + iw * 3 + c[None, :])
    a_mask = m_mask[:, None] & (in_bounds & k_valid[None, :])
    A = tl.load(x_ptr + addr, mask=a_mask, other=0.0)    # [M_,128] fp16

    n = tl.arange(0, 16)
    W = tl.load(w_ptr + k[:, None] * 16 + n[None, :])    # [128,16] fp16(pad 行=0)

    acc = tl.dot(A, W)                                   # [M_,16] fp32
    bias = tl.load(b_ptr + n)
    qa = tl.load(qa_ptr + n)
    qc = tl.load(qc_ptr + n)
    z = tl.maximum(acc + bias[None, :], 0.0)             # conv+b → ReLU
    z = z * qa[None, :] + qc[None, :]                    # BN1 / s_z(融合)
    # round-half-away-from-zero(避开 libdevice 版本差异;与 torch.round 仅在
    # 恰好 .5 处相差 1 个量化步,可忽略)
    q = tl.where(z >= 0, tl.floor(z + 0.5), tl.ceil(z - 0.5))
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + m[:, None] * 16 + n[None, :],
             q.to(tl.int8), mask=m_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _k_conv2_int8_fused(
    z_ptr,       # [B*187,16] int8(= [B,11,17,16] NHWC)
    w_ptr,       # [3,64,64] int8 packed(kh; ks=kw*16+c, ks<48 有效)
    dq_ptr,      # [64] fp32 = s_w·s_z per-out-channel
    b_ptr,       # [64] fp32 conv2 bias
    a2_ptr,      # [64] fp32 BN2 scale
    c2_ptr,      # [64] fp32 BN2 shift
    out_ptr,     # [B*187,64] fp16(= local_features [B,187,64])
    M,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M
    b = m // 187
    p_ = m % 187
    oh = p_ // 17
    ow = p_ % 17

    ks = tl.arange(0, 64)                                # 每个 kh 步: kw*16+c(<48 有效)
    kw = ks // 16
    c = ks % 16
    k_valid = ks < 48
    n = tl.arange(0, 64)

    acc = tl.zeros((BLOCK_M, 64), dtype=tl.int32)
    for kh in tl.static_range(3):                        # k=3, pad=1
        ih = oh - 1 + kh                                 # [M_]
        iw = (ow - 1)[:, None] + kw[None, :]             # [M_,64]
        in_bounds = ((ih >= 0) & (ih < 11))[:, None] & (iw >= 0) & (iw < 17)
        addr = (b[:, None] * (187 * 16)
                + (ih[:, None] * 17 + iw) * 16 + c[None, :])
        a_mask = m_mask[:, None] & (in_bounds & k_valid[None, :])
        A = tl.load(z_ptr + addr, mask=a_mask, other=0)  # [M_,64] int8(pad=0 ✓)
        W = tl.load(w_ptr + kh * (64 * 64)
                    + ks[:, None] * 64 + n[None, :])     # [64,64] int8(pad 行=0)
        # Triton 3.6: int8 dot 须用 out_dtype 形式(位置参数 acc 形式编译失败)
        acc += tl.dot(A, W, out_dtype=tl.int32)          # int8×int8 → int32(IMMA)

    dq = tl.load(dq_ptr + n)
    bias = tl.load(b_ptr + n)
    a2 = tl.load(a2_ptr + n)
    c2 = tl.load(c2_ptr + n)
    y = acc.to(tl.float32) * dq[None, :] + bias[None, :]  # dequant + conv bias
    y = tl.maximum(y, 0.0)                                # ReLU
    y = y * a2[None, :] + c2[None, :]                     # BN2
    tl.store(out_ptr + m[:, None] * 64 + n[None, :],
             y.to(tl.float16), mask=m_mask[:, None])


# ════════════════════════════════════════════════════════════════════════════
# custom op 注册(dynamo/inductor 可入图,cudagraph 安全)
# ════════════════════════════════════════════════════════════════════════════

@torch.library.custom_op("qrt::conv1_fused_q", mutates_args=(), device_types="cuda")
def _op_conv1(x_nhwc: torch.Tensor, w1p: torch.Tensor, b1: torch.Tensor,
              qa: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
    B = x_nhwc.shape[0]
    M = B * P
    out = torch.empty(M, C2, dtype=torch.int8, device=x_nhwc.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _k_conv1_fused_q[grid](x_nhwc, w1p, b1, qa, qc, out, M)
    return out


@_op_conv1.register_fake
def _(x_nhwc, w1p, b1, qa, qc):
    return x_nhwc.new_empty((x_nhwc.shape[0] * P, C2), dtype=torch.int8)


@torch.library.custom_op("qrt::conv2_int8_fused", mutates_args=(), device_types="cuda")
def _op_conv2(z8: torch.Tensor, w8: torch.Tensor, dq: torch.Tensor,
              b2: torch.Tensor, a2: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    M = z8.shape[0]
    out = torch.empty(M, N2, dtype=torch.float16, device=z8.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _k_conv2_int8_fused[grid](z8, w8, dq, b2, a2, c2, out, M)
    return out


@_op_conv2.register_fake
def _(z8, w8, dq, b2, a2, c2):
    return z8.new_empty((z8.shape[0], N2), dtype=torch.float16)


# ════════════════════════════════════════════════════════════════════════════
# QuantCNN 模块(替换 model.map_cnn,参数路径保持 map_cnn.0/2/3/5 不变)
# ════════════════════════════════════════════════════════════════════════════

class QuantCNN(nn.Module):
    """GLAD map_cnn 的量化替身。

    - 子模块沿用原 Sequential 的索引名("0"=conv1,"2"=BN1,"3"=conv2,"5"=BN2),
      named_parameters 路径不变 → sync_weights_from / optimizer 引用均兼容;
    - eval 前向走两个 fused kernel,输出以零拷贝视图伪装回 [B,64,11,17]
      (下游的 permute+reshape 在 compile 图内被消除);
    - train 前向走原始算子(单模型形态可训练);
    - requantize_(): 从当前 conv/BN 参数重算全部 packed buffer(GPU,graph 安全);
      LET 时传入 alpha/beta([16],conv2 输入通道)自动并入 BN1 仿射与 W2 折叠。
    """

    def __init__(self, map_cnn: nn.Sequential, s_z: float,
                 alpha: torch.Tensor | None = None,
                 beta: torch.Tensor | None = None):
        super().__init__()
        # 索引名与原 Sequential 对齐(1/4 是无参数的 ReLU,不注册)
        self.add_module("0", map_cnn[0])     # Conv2d(3,16,5,s2,p2)
        self.add_module("2", map_cnn[2])     # BatchNorm2d(16)
        self.add_module("3", map_cnn[3])     # Conv2d(16,64,3,p1)
        self.add_module("5", map_cnn[5])     # BatchNorm2d(64)
        dev = map_cnn[0].weight.device

        self.has_let = alpha is not None
        if self.has_let:
            self.register_buffer("alpha", alpha.detach().float().clamp_min(1e-8).to(dev))
            self.register_buffer("beta",
                                 beta.detach().float().to(dev) if beta is not None
                                 else torch.zeros(C2, device=dev))
        self.register_buffer("s_z", torch.tensor(float(max(s_z, 1e-8)), device=dev))

        self.register_buffer("w1p", torch.zeros(K1, C2, dtype=torch.float16, device=dev))
        self.register_buffer("b1f", torch.zeros(C2, dtype=torch.float32, device=dev))
        self.register_buffer("qa", torch.zeros(C2, dtype=torch.float32, device=dev))
        self.register_buffer("qc", torch.zeros(C2, dtype=torch.float32, device=dev))
        self.register_buffer("w8", torch.zeros(3, K2S, N2, dtype=torch.int8, device=dev))
        self.register_buffer("dq", torch.zeros(N2, dtype=torch.float32, device=dev))
        self.register_buffer("b2f", torch.zeros(N2, dtype=torch.float32, device=dev))
        self.register_buffer("a2", torch.zeros(N2, dtype=torch.float32, device=dev))
        self.register_buffer("c2", torch.zeros(N2, dtype=torch.float32, device=dev))
        self.requantize_()

    @torch.no_grad()
    def requantize_(self):
        conv1, bn1 = self._modules["0"], self._modules["2"]
        conv2, bn2 = self._modules["3"], self._modules["5"]
        # BN 仿射(eval): y = a·x + c
        a1 = (bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)).float()
        c1 = bn1.bias.float() - a1 * bn1.running_mean.float()
        a2 = (bn2.weight / torch.sqrt(bn2.running_var + bn2.eps)).float()
        c2 = bn2.bias.float() - a2 * bn2.running_mean.float()
        if self.has_let:                       # LET 并入 BN1 仿射(z=α·BN1(·)+β)
            a1 = a1 * self.alpha
            c1 = c1 * self.alpha + self.beta

        # conv1 packed [128,16]: k=kh*16+r(kh<5 且 r<15 有效: kw=r//3,c=r%3)
        W1 = conv1.weight.detach().float()                     # [16,3,5,5]
        w1p = torch.zeros(K1, C2, device=W1.device)
        w1p.view(8, 16, C2)[:5, :15, :] = W1.permute(2, 3, 1, 0).reshape(5, 15, C2)
        self.w1p.copy_(w1p.half())
        self.b1f.copy_(conv1.bias.detach().float())
        self.qa.copy_(a1 / self.s_z)
        self.qc.copy_(c1 / self.s_z)

        # conv2 packed [3,64,64]: (kh; ks=kw*16+c, ks<48 有效)
        W2 = conv2.weight.detach().float()                     # [64,16,3,3]
        if self.has_let:
            W2 = W2 / self.alpha[None, :, None, None]          # W' = W/α
        Wp = W2.permute(2, 3, 1, 0).reshape(3, 48, N2)         # [kh, kw*16+c, n]
        s_w = Wp.abs().amax(dim=(0, 1)).clamp_min(1e-8) / 127.0    # [64]
        w8 = torch.zeros(3, K2S, N2, device=W2.device)
        w8[:, :48, :] = torch.clamp(torch.round(Wp / s_w[None, None, :]), -128, 127)
        self.w8.copy_(w8.to(torch.int8))
        self.dq.copy_(s_w * self.s_z)
        b2 = conv2.bias.detach().float()
        if self.has_let:                                       # b' = b − Σ W'·β
            b2 = b2 - (W2 * self.beta[None, :, None, None]).sum(dim=(1, 2, 3))
        self.b2f.copy_(b2)
        self.a2.copy_(a2)
        self.c2.copy_(c2)

    @torch.no_grad()
    def set_act_scale_(self, s_z: float):
        self.s_z.fill_(float(max(s_z, 1e-8)))

    def forward(self, x):                       # x: [B,3,21,33]
        # 同 QuantLinear: 按梯度态分支(train 态 rollout 也走 INT8 kernel;
        # 注意 INT8 路径的 BN 烘的是 running stats —— eval 语义)
        if torch.is_grad_enabled():
            m = self._modules
            h = m["2"](torch.relu(m["0"](x)))
            return m["5"](torch.relu(m["3"](h)))
        B = x.shape[0]
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()            # 多为零拷贝还原
        z8 = torch.ops.qrt.conv1_fused_q(x_nhwc, self.w1p, self.b1f, self.qa, self.qc)
        feat = torch.ops.qrt.conv2_int8_fused(z8, self.w8, self.dq,
                                              self.b2f, self.a2, self.c2)
        # 伪装回 [B,64,11,17](零拷贝视图;下游 permute+reshape 在图内抵消)
        return feat.view(B, H2, W2, N2).permute(0, 3, 1, 2)

    def extra_repr(self):
        return f"GLAD CNN fused, let={self.has_let}, s_z={self.s_z.item():.4g}"


def swap_cnn(model, s_z: float, alpha=None, beta=None) -> QuantCNN:
    """把 model.map_cnn 替换为 QuantCNN(继承 train/eval 状态),返回新模块。"""
    qcnn = QuantCNN(model.map_cnn, s_z, alpha=alpha, beta=beta)
    qcnn.train(model.map_cnn.training)
    model.map_cnn = qcnn
    return qcnn


@torch.no_grad()
def calib_cnn_scale(model, obs_fn, n_batches: int = 8) -> float:
    """校准 conv2 输入(BN1 输出)的 per-tensor scale s_z = absmax/127。
    在原始 map_cnn(swap 前)上 hook;LET 时应改为统计 α·BN1+β 的 absmax。"""
    amax = [0.0]
    bn1 = model.map_cnn[2]
    h = bn1.register_forward_hook(
        lambda m, i, o: amax.__setitem__(0, max(amax[0], o.detach().abs().amax().item())))
    was = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(n_batches):
            model.act_inference(obs_fn())
    if was:
        model.train()
    h.remove()
    return max(amax[0], 1e-6) / 127.0

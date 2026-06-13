# -*- coding: utf-8 -*-
"""
let.py — LET (Learnable Equivalent Transformation)

来源: OmniQuant §3, Shao et al., arXiv:2308.13137

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
等价变换（对应论文 Eq. 3）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  原始:   Y = X·W + B
  等价:   Y = [(X−δ)⊘s]·[s⊙W] + [B+δW]
            = Qa(X̃)·Qw(W̃) + B̃       （加入量化后的目标形式）

  论文参数:  s ∈ ℝ^Cin（channel-wise 缩放），δ ∈ ℝ^Cin（channel-wise 偏移）
  本文映射:  α = 1/s，β = −δ/s    →   z = αx+β = (x−δ)/s = X̃
  可学习集: Θ₂ = {δ, s}，即本文的 {α, β} **两者均可学习**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
β 的梯度分析（为什么 β 必须是可学习参数）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

block_reconstruction_loss 计算：

    y_quant = Qw(W') · Q(z) + b'
    其中：
        W' = W/α         (精确浮点)
        Qw(W') = fake_quant(W')   (量化后的 W')
        b' = b − W'·β    (用精确 W'，不是 Qw(W'))
        z  = αx + β

    ∂y_quant/∂β_k  展开为两路：
        ① 通过 Q(z) (STE≈1)：  Qw(W')_{:,k}
        ② 通过 b'：            −W'_{:,k}

    由于 Qw(W') ≠ W'（量化误差 ε_W = Qw(W')−W' ≠ 0），两路不等价：
        ∂y/∂β_k = Qw(W')_{:,k} − W'_{:,k} = ε_W_{:,k}  ≠ 0

    梯度由权重量化误差驱动：误差大的 channel 给 β 更强的校正信号。
    这正是 OmniQuant 中 δ（对应 β）可以被有效优化的根本原因。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
旧实现的错误（已修复）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Bug 1: beta 是 register_buffer（不可学习），应为 nn.Parameter
  Bug 2: block_reconstruction_loss 里 F.linear 用精确浮点 W'，而 b' 也用精确 W'，
         导致两路完全抵消，β 梯度恒为零。正确做法：F.linear 用 Qw(W')，
         b' 仍用精确 W'（与 OmniQuant B̃ = B+δW 对应）。
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LETQuantizer(nn.Module):
    """
    LET 量化器。

    可学习参数（两个）：
        alpha: [n_channels]  per-channel 缩放（对应论文 1/s）
        beta:  [n_channels]  per-channel 平移（对应论文 −δ/s）

    典型用法（PTQ 离线校准）：
    ─────────────────────────────────────────────────────────────────
        let = LETQuantizer(n_channels=64)
        let.initialize_from_activation(x_calib)   # 可选热启动

        opt = torch.optim.Adam(let.parameters(), lr=5e-3)
        for x_batch in calib_batches:
            _, loss = let.block_reconstruction_loss(x_batch, linear_layer)
            opt.zero_grad(); loss.backward(); opt.step()

        # 校准收敛后提取部署参数
        trt_scale = let.calibrate_scale(x_calib)   # → TRT QDQ scale
        new_linear = let.fold_into_linear(linear_layer)
    ─────────────────────────────────────────────────────────────────

    Args:
        n_channels: 激活最后维大小（C_in）
        nbits:      量化位宽，默认 8（INT8）
    """

    def __init__(self, n_channels: int, nbits: int = 8):
        super().__init__()
        self.n_channels = n_channels
        self.nbits  = nbits
        self.q_max  =  2 ** (nbits - 1) - 1   # INT8:  127
        self.q_min  = -(2 ** (nbits - 1))       # INT8: -128

        # 两个可学习参数（对应论文 Θ₂ = {s, δ}）
        self.alpha = nn.Parameter(torch.ones(n_channels))
        self.beta  = nn.Parameter(torch.zeros(n_channels))

    # ── 私有：STE 量化 ────────────────────────────────────────────────────────

    @staticmethod
    def _ste_round(x: torch.Tensor) -> torch.Tensor:
        return x + (x.round() - x).detach()

    def _fake_quant_act(self, z: torch.Tensor) -> torch.Tensor:
        """激活 fake-quant：per-tensor absmax 对称 INT8，STE。"""
        scale = z.detach().abs().amax().clamp_min(1e-8) / self.q_max
        z_scaled  = (z / scale).clamp(self.q_min, self.q_max)
        z_rounded = self._ste_round(z_scaled)
        return z_rounded * scale

    def _fake_quant_weight(self, W: torch.Tensor) -> torch.Tensor:
        """
        权重 fake-quant：per-output-channel absmax 对称 INT8，STE。

        W: [C_out, C_in]，每行（每个 output channel）独立缩放。
        这与 TRT INT8 权重量化方式一致。
        """
        scale = W.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.q_max
        W_scaled  = (W / scale).clamp(self.q_min, self.q_max)
        W_rounded = self._ste_round(W_scaled)
        return W_rounded * scale

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        训练时 inline 前向（嵌入模型 forward 时使用）：
            z     = α⊙x + β
            z_hat = Qa(z)         fake-quant, STE
            x_hat = (z_hat − β)/α  逆变换还原 x 空间

        下游层继续用原始权重 W·x_hat+b。
        等价于用精确权重做 W'·Qa(z)+b'（见推导），但下游层无需修改。
        """
        z     = self.alpha * x + self.beta
        z_hat = self._fake_quant_act(z)
        x_hat = (z_hat - self.beta) / self.alpha.clamp_min(1e-8)
        return x_hat

    def block_reconstruction_loss(
        self,
        x: torch.Tensor,
        linear: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LET 校准损失（OmniQuant §3 block 输出重建，Eq. 4）：

            L = MSE( Qw(W')·Qa(z) + b',   W·x+b )

        其中：
            z    = α⊙x + β          (LET 变换激活)
            W'   = W/α               (等价权重，精确浮点)
            Qw(W') = fake_quant(W')  (INT8 量化后的等价权重)
            b'   = b − W'·β          (偏置校正，用精确 W' 而非 Qw(W'))

        关键：b' 用精确 W'，F.linear 用 Qw(W')，两路不等价，
        β 的梯度 = Qw(W')−W' = ε_W（权重量化误差），非零。
        α 的梯度来自 W'=W/α 的导数路径（更强信号）。

        Args:
            x:      校准激活 [..., C_in]
            linear: 目标 nn.Linear（W, b 不更新）
        Returns:
            (y_quant, loss)
        """
        with torch.no_grad():
            y_fp32 = F.linear(x, linear.weight, linear.bias)

        # LET 变换 + 激活量化
        z     = self.alpha * x + self.beta
        z_hat = self._fake_quant_act(z)                  # Qa(z), STE

        # 等价权重（精确浮点，用于 b' 计算）
        # linear.weight: [C_out, C_in], alpha: [C_in]
        W_prime = linear.weight.detach() / self.alpha.unsqueeze(0)   # [C_out, C_in]

        # 偏置校正：用精确 W'（对应论文 B̃ = B + δW，其中 δ = −β/α·s = −β）
        if linear.bias is not None:
            b_prime = linear.bias.detach() \
                      - (W_prime * self.beta.unsqueeze(0)).sum(dim=1)   # [C_out]
        else:
            b_prime = -(W_prime * self.beta.unsqueeze(0)).sum(dim=1)

        # 前向用量化后的 W'（对应论文 Qw(W̃)），b' 仍用精确 W'
        # 两路不对称 → β 梯度 = ε_W = Qw(W') − W' ≠ 0
        W_prime_q = self._fake_quant_weight(W_prime)     # Qw(W'), STE

        y_quant = F.linear(z_hat, W_prime_q, b_prime)
        return y_quant, F.mse_loss(y_quant, y_fp32)

    # ── 初始化工具 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize_from_activation(self, x: torch.Tensor) -> None:
        """
        从激活统计量热启动（可选，比默认 α=1, β=0 收敛更快）：
            α = 1/σ_c   → z = αx+β 各通道方差 ≈ 1（等方差，INT8 友好）
            β = −μ_c/σ_c → z 各通道均值 ≈ 0（对应 OmniQuant 初始化 from OS+）

        Args:
            x: 激活样本 [..., C_in]
        """
        x_flat = x.detach().reshape(-1, self.n_channels)
        mu     = x_flat.mean(dim=0)
        sigma  = x_flat.std(dim=0).clamp_min(1e-8)
        self.alpha.data.copy_(1.0 / sigma)
        self.beta.data.copy_(-mu / sigma)

    @torch.no_grad()
    def set_beta_from_mean(self, x: torch.Tensor) -> None:
        """仅固定 β = −mean(x)（中心化激活），不改变 α。"""
        mu = x.detach().reshape(-1, self.n_channels).mean(dim=0)
        self.beta.data.copy_(-mu)

    # ── 部署工具 ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def calibrate_scale(self, x_samples: torch.Tensor) -> float:
        """
        在校准样本上估计 z = αx+β 的量化 scale，填入 TRT QDQ 节点。

        TRT_QDQ.scale = absmax(αx+β) / 127

        调用时机：block_reconstruction_loss 收敛后，fold_into_linear 之前。

        Args:
            x_samples: 代表性激活样本 [..., C_in]
        Returns:
            scale (float)
        """
        z     = self.alpha * x_samples + self.beta
        scale = z.abs().amax().item() / self.q_max
        return max(scale, 1e-8)

    @torch.no_grad()
    def fold_into_linear(self, linear: nn.Linear) -> nn.Linear:
        """
        将 α, β 折叠进 nn.Linear，返回新 Linear（不修改原层）。

        折叠公式：
            W_new = W / α    (per-column，对应 W'=W/α=s⊙W)
            b_new = b − W_new·β

        折叠后推理路径：
            y = W_new · Qa(αx+β) + b_new  ≡ W·x+b（无量化误差时）

        Returns:
            folded: 新的 nn.Linear
        """
        α = self.alpha.clamp_min(1e-8)
        β = self.beta

        folded = copy.deepcopy(linear)

        # W_new = W / α（per-column 广播）
        folded.weight.data.div_(α.unsqueeze(0))

        # b_new = b − W_new·β
        bias_corr = (folded.weight.data * β.unsqueeze(0)).sum(dim=1)
        if folded.bias is not None:
            folded.bias.data.sub_(bias_corr)
        else:
            folded.bias = nn.Parameter(-bias_corr.clone(), requires_grad=False)

        return folded

    def extra_repr(self) -> str:
        return (
            f"n_ch={self.n_channels}, nbits={self.nbits}, "
            f"α_mean={self.alpha.data.abs().mean().item():.3f}, "
            f"β_mean={self.beta.data.abs().mean().item():.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONNX 图权重修改工具
# ══════════════════════════════════════════════════════════════════════════════
def fold_weights_in_onnx_node(
    node,           # onnx_graphsurgeon.Node
    let: LETQuantizer,
    node_type: str  # "MatMul" 或 "Gemm"
) -> None:
    """
    将折叠后的 W' = W/α, b' = b − W'·β 直接写入 ONNX GraphSurgeon 节点。

    调用时机：LET 校准收敛后，GS 插入 QDQ 之前。

    ONNX 节点输入槽位约定：
        MatMul: inputs[0]=激活, inputs[1]=权重 W
        Gemm:   inputs[0]=激活, inputs[1]=权重 W, inputs[2]=偏置 b（可选）

    注意：MatMul 权重可能是 [C_in, C_out]（列向量），Gemm 是 [C_out, C_in]。
    函数通过 C_in = len(α) 自动判断并处理转置。
    """
    import onnx_graphsurgeon as gs
    import numpy as np

    α = let.alpha.detach().clamp_min(1e-8).cpu().numpy()
    β = let.beta.detach().cpu().numpy()

    w_idx = 1
    assert isinstance(node.inputs[w_idx], gs.Constant), \
        f"inputs[{w_idx}] of '{node.name}' is not a weight Constant"

    W = node.inputs[w_idx].values.copy()

    if W.shape[0] == len(α):
        W = W.T
        transposed = True
    else:
        transposed = False

    assert W.shape[1] == len(α), \
        f"W.shape={W.shape}，α.shape={α.shape}，C_in 不匹配"

    W_prime = W / α[np.newaxis, :]

    if (node_type == "Gemm"
            and len(node.inputs) > 2
            and isinstance(node.inputs[2], gs.Constant)):
        b = node.inputs[2].values.copy()
        bias_corr = (W_prime * β[np.newaxis, :]).sum(axis=1)
        node.inputs[2].values = (b - bias_corr).astype(np.float32)

    if transposed:
        W_prime = W_prime.T
    node.inputs[w_idx].values = W_prime.astype(np.float32)


def fold_inproj_weight_onnx(
    graph,
    let_q: "LETQuantizer | None" = None,
    let_kv: "LETQuantizer | None" = None,
    D: int = 64,
    weight_name: str = "m.mha.in_proj_weight",
    bias_name:   str = "m.mha.in_proj_bias",
) -> bool:
    """
    将 LET alpha, beta 折叠进 MHA 的 in_proj_weight Constant，
    修复 MatMul_69/71 因 inputs[1] 是 Variable 无法直接折叠的问题。

    ONNX 图结构（Split_124 + Transpose，不能直接改 MatMul inputs[1]）：
        in_proj_weight [3D, D]  Constant [C_out, C_in]
          |
          Split([D, 2D])
          |         |
        Q [D,D]   KV [2D,D]
          |         |
        Transpose  Transpose
          |         |
        MatMul_69  MatMul_71

    直接修改 in_proj_weight Constant，Split/Transpose 自动传播新值。
    同步修正 in_proj_bias。

    let_q 和 let_kv 均可为 None：
      - let_q=None  : 只折叠 KV 部分（仅 MatMul_71 在 include_nodes 时）
      - let_kv=None : 只折叠 Q  部分（仅 MatMul_69 在 include_nodes 时）
      - 两者均 None : 无操作，直接返回 True

    折叠公式（[C_out, C_in] 约定）：
        W_new[i, j] = W[i, j] / alpha[j]
        b_new[i]    = b[i] - sum_j(W_new[i,j] * beta[j])
    """
    import onnx_graphsurgeon as gs
    import numpy as np

    if let_q is None and let_kv is None:
        return True  # 无需折叠

    tensors = graph.tensors()

    # 按名称找权重，找不到时按形状兜底
    inproj_w = tensors.get(weight_name)
    if not isinstance(inproj_w, gs.Constant):
        inproj_w = next(
            (t for t in tensors.values()
             if isinstance(t, gs.Constant) and tuple(t.shape) == (3 * D, D)),
            None,
        )
    if inproj_w is None:
        print(f"  [warn] fold_inproj_weight_onnx: '{weight_name}' not found, "
              f"tried shape {(3*D, D)}")
        return False

    inproj_b = tensors.get(bias_name)
    if not isinstance(inproj_b, gs.Constant):
        inproj_b = next(
            (t for t in tensors.values()
             if isinstance(t, gs.Constant) and tuple(t.shape) == (3 * D,)),
            None,
        )

    W = inproj_w.values.copy()   # [3D, D]
    b = inproj_b.values.copy() if inproj_b is not None else None

    # 折叠 Q（rows 0:D）
    if let_q is not None:
        alpha_q = let_q.alpha.detach().clamp_min(1e-8).cpu().numpy()
        beta_q  = let_q.beta.detach().cpu().numpy()
        W[:D] /= alpha_q
        if b is not None:
            b[:D] -= (W[:D] * beta_q[np.newaxis, :]).sum(axis=1)
        print(f"  [in_proj Q ] folded: a_mean={alpha_q.mean():.4f}"
              f"  |b_mean|={np.abs(beta_q).mean():.4f}")

    # 折叠 KV（rows D:3D）
    if let_kv is not None:
        alpha_kv = let_kv.alpha.detach().clamp_min(1e-8).cpu().numpy()
        beta_kv  = let_kv.beta.detach().cpu().numpy()
        W[D:] /= alpha_kv
        if b is not None:
            b[D:] -= (W[D:] * beta_kv[np.newaxis, :]).sum(axis=1)
        print(f"  [in_proj KV] folded: a_mean={alpha_kv.mean():.4f}"
              f"  |b_mean|={np.abs(beta_kv).mean():.4f}")

    inproj_w.values = W.astype(np.float32)
    if b is not None and inproj_b is not None:
        inproj_b.values = b.astype(np.float32)

    return True

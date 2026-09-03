# -*- coding: utf-8 -*-
"""
qrt/tdmpc2/world_model.py — 自包含的 TD-MPC2 世界模型 + MPPI 规划器
══════════════════════════════════════════════════════════════════════════════
目的:给 qrt 量化方法(QuantLinear+LET / WeightOnlyLinear / compile+CUDA Graph)
提供一个**结构忠实、可独立跑**的 TD-MPC2 推理载体,无需 clone 官方 tdmpc2 及其
mujoco/dm_control 重依赖(远程 GPU 机装包要走代理别名,见 CLAUDE.md)。

与官方 TD-MPC2(Hansen et al., 2024, arXiv:2310.16828)对齐处:
  • 组件:Encoder / Latent dynamics / Reward / Terminal value(Q 集成)/ Policy prior π
  • 结构:MLP + LayerNorm + Mish;隐状态过 SimNorm;reward/value 走 num_bins 二热回归
  • 官方默认 config.yaml 维度:latent_dim=512, mlp_dim=512, enc_dim=256, num_q=5,
    horizon=3, num_samples=512, num_pi_trajs=24, iterations=6, num_elites=64
  • 推理即在线 MPPI 规划:每环境步 encoder 跑 1 次(B=1),世界模型 rollout 跑
    iterations×horizon 次、每次 batch=num_samples+num_pi_trajs(≈536)

**这正是 roofline 的镜像**:encoder=B1(权重访存主导 → WeightOnlyLinear),
rollout=B536(计算/激活主导 → QuantLinear A8W8 IMMA)。见 EXP-002。

注意:本文件只求**结构/形状/算子分布忠实**以支撑 latency 与量化研究,不追求 RL
训练可复现(权重随机初始化)。要跑真实任务精度时,把官方 checkpoint 的权重
load_state_dict 进来即可(命名对齐见下方各模块 docstring)。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 配置(官方 config.yaml 默认值)────────────────────────────────────────────
@dataclass
class TDMPC2Config:
    obs_dim: int = 64            # 状态观测维(state-based;像素版另走 conv encoder)
    action_dim: int = 24        # 动作维(取 8 的倍数,使 dynamics/reward 首层 K%8==0)
    latent_dim: int = 512
    mlp_dim: int = 512
    enc_dim: int = 256
    num_q: int = 5
    num_bins: int = 101         # reward/value 二热回归 bin 数
    vmin: float = -10.0
    vmax: float = 10.0
    simnorm_dim: int = 8        # SimNorm 分组维
    horizon: int = 3
    num_samples: int = 512
    num_pi_trajs: int = 24
    num_elites: int = 64
    iterations: int = 6
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    gamma: float = 0.99


# ── SimNorm(TD-MPC2 隐状态归一:分组 softmax)────────────────────────────────
class SimNorm(nn.Module):
    def __init__(self, group_dim: int):
        super().__init__()
        self.group_dim = group_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.group_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def extra_repr(self):
        return f"group_dim={self.group_dim}"


def _mlp(in_dim: int, hidden: list[int], out_dim: int, act_out: nn.Module | None):
    """TD-MPC2 风格 MLP:每隐层 Linear→LayerNorm→Mish;输出层可选 act(SimNorm/None)。
    命名为 Sequential 数字索引(与 swap 的点分路径 '<mod>.0/.2/...' 对齐)。"""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.Mish(inplace=False)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    if act_out is not None:
        layers += [act_out]
    return nn.Sequential(*layers)


# ── 世界模型 ──────────────────────────────────────────────────────────────────
class TDMPC2WorldModel(nn.Module):
    """自包含 TD-MPC2 世界模型。子模块命名对齐官方,便于日后 load 真实权重。

    量化点分路径(供 qrt.swap / swap_wlinear 消费):
      encoder(B=1,WeightOnlyLinear):  encoder.0 / encoder.3
      rollout(B≈536,QuantLinear A8W8): dynamics.* / reward.* / Qs.<i>.* / pi.*
    """

    def __init__(self, cfg: TDMPC2Config):
        super().__init__()
        self.cfg = cfg
        L, M, A = cfg.latent_dim, cfg.mlp_dim, cfg.action_dim
        sn = SimNorm(cfg.simnorm_dim)

        # Encoder(state):obs → enc_dim → latent,输出 SimNorm。推理时 B=1。
        self.encoder = _mlp(cfg.obs_dim, [cfg.enc_dim], L, act_out=sn)
        # Latent dynamics:[z, a] → latent,输出 SimNorm。rollout 主干,B≈536。
        self.dynamics = _mlp(L + A, [M, M], L, act_out=SimNorm(cfg.simnorm_dim))
        # Reward:[z, a] → num_bins(二热回归)。
        self.reward = _mlp(L + A, [M, M], cfg.num_bins, act_out=None)
        # Terminal value:num_q 个 Q 头,[z, a] → num_bins。
        self.Qs = nn.ModuleList(
            [_mlp(L + A, [M, M], cfg.num_bins, act_out=None) for _ in range(cfg.num_q)]
        )
        # Policy prior π:z → 2*action_dim(mean, log_std)。
        self.pi = _mlp(L, [M, M], 2 * A, act_out=None)

        # 二热回归 bin 中心值(vmin..vmax)
        self.register_buffer(
            "bins", torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins), persistent=False
        )
        self.register_buffer("log_std_min", torch.tensor(-10.0), persistent=False)
        self.register_buffer("log_std_max", torch.tensor(2.0), persistent=False)

    # ── 基础前向 ──────────────────────────────────────────────────────────────
    def encode(self, obs):
        return self.encoder(obs)

    def next(self, z, a):
        return self.dynamics(torch.cat([z, a], dim=-1))

    def _decode_bins(self, logits):
        """二热 logits → 期望标量值(softmax·bin 中心)。"""
        p = F.softmax(logits, dim=-1)
        return (p * self.bins).sum(dim=-1, keepdim=True)

    def reward_val(self, z, a):
        return self._decode_bins(self.reward(torch.cat([z, a], dim=-1)))

    def pi_action(self, z, std_scale: float = 0.0):
        """从策略先验采样动作(std_scale=0 → 取均值,规划内 bootstrap 用)。"""
        mu, log_std = self.pi(z).chunk(2, dim=-1)
        mu = torch.tanh(mu)
        if std_scale <= 0.0:
            return mu
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return torch.tanh(mu + std_scale * log_std.exp() * torch.randn_like(mu))

    def q_min(self, z, a):
        """Q 集成 → 期望值,取两成员随机子采样的最小(TD-MPC2 保守估计的简化)。"""
        vals = torch.stack([self._decode_bins(q(torch.cat([z, a], dim=-1)))
                            for q in self.Qs], dim=0)          # [num_q, B, 1]
        # 简化:取全体最小(保守);真实实现随机抽 2 个取 min,latency 同量级
        return vals.min(dim=0).values

    # ── MPPI 规划(推理主循环;固定形状 → 适合 compile + CUDA Graph 捕获）────
    @torch.no_grad()
    def estimate_value(self, z0, actions):
        """batched rollout:z0[N,L],actions[H,N,A] → 折扣回报 G[N,1]。
        这是计算主导的核心(每步 dynamics/reward 在 batch=N 上跑,H 步串行)。"""
        G = torch.zeros(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        discount = 1.0
        z = z0
        for t in range(actions.shape[0]):
            a = actions[t]
            G = G + discount * self.reward_val(z, a)
            z = self.next(z, a)
            discount *= self.cfg.gamma
        # 终点 bootstrap:π(z) 的 Q
        a_last = self.pi_action(z, std_scale=0.0)
        G = G + discount * self.q_min(z, a_last)
        return G

    @torch.no_grad()
    def plan(self, obs):
        """单环境步 MPPI 规划,返回选定动作 [action_dim]。obs:[obs_dim] 或 [1,obs_dim]。"""
        cfg = self.cfg
        dev = obs.device
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z0 = self.encode(obs)                                  # [1, L]  ← encoder B=1

        N = cfg.num_samples + cfg.num_pi_trajs
        H, A = cfg.horizon, cfg.action_dim

        # π 轨迹(bootstrap 采样分布)
        pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=dev, dtype=z0.dtype)
        z = z0.repeat(cfg.num_pi_trajs, 1)
        for t in range(H):
            a = self.pi_action(z, std_scale=1.0)
            pi_actions[t] = a
            z = self.next(z, a)

        mean = torch.zeros(H, A, device=dev, dtype=z0.dtype)
        std = torch.full((H, A), cfg.max_std, device=dev, dtype=z0.dtype)
        z0N = z0.repeat(N, 1)                                  # [N, L]  ← rollout B=N

        for _ in range(cfg.iterations):
            # 采样 num_samples 条 + 拼接 num_pi_trajs 条
            noise = torch.randn(H, cfg.num_samples, A, device=dev, dtype=z0.dtype)
            samp = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            actions = torch.cat([samp, pi_actions], dim=1)     # [H, N, A]

            value = self.estimate_value(z0N, actions).squeeze(-1)          # [N]
            elite_val, elite_idx = torch.topk(value, cfg.num_elites)
            elite_actions = actions[:, elite_idx]                          # [H, E, A]

            # softmax(温度) 加权更新分布
            w = torch.softmax(cfg.temperature * (elite_val - elite_val.max()), dim=0)
            w = w.view(1, -1, 1)
            mean = (w * elite_actions).sum(dim=1)
            var = (w * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        return mean[0]                                         # 首步动作均值


def build_world_model(cfg: TDMPC2Config | None = None,
                      dtype: torch.dtype = torch.float16,
                      device: str = "cuda") -> TDMPC2WorldModel:
    cfg = cfg or TDMPC2Config()
    m = TDMPC2WorldModel(cfg).to(device)
    if dtype == torch.float16:
        m = m.half()
    return m.eval()

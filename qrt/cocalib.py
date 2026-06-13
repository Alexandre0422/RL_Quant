# -*- coding: utf-8 -*-
"""
cocalib.py — P6: 在线伴随校准器(误差驱动的自适应 LET α/β 保鲜)
══════════════════════════════════════════════════════════════════════════════
动机链(全部有实验背书):
  P5  学习的 α 平衡激活/权重两侧误差,优于统计 α(dynamic_full 漂移最快)
  P1/P5  但学习的 α 会过时 —— 消除 s_z 失配后,α/β 静态失配是唯一剩余漂移源
  P6(本模块)  以最小代价在线保鲜 α/β,补完三时间尺度的量化参数生命周期:
      s_z     每次前向(图内动态,P5)
      W8/s_w/b'  每个 iteration(sync_weights_from 重折叠,P1.5)
      α/β     误差驱动伴随优化(本模块,多数 iteration 为 0 步)

设计要点:
  - **与策略优化完全解耦**: LETQuantizer 的重建损失天然 detach 权重,只有 α/β
    收到梯度;独立 Adam,不进 L_PPO,无 λ 超参,策略梯度零污染;
  - **对齐 FP32 真值**: 重建参照 = FP32 master 权重的精确前向(激活与计算全 fp32);
  - **warm-start**: α/β(及 Adam 动量)永不重置,跟踪缓慢移动的最优点;
  - **误差驱动 + 早停**: 每次 step() 先在小批激活上评估各层重建损失 L,
    仅 L > τ·baseline 的层进入优化,步进至达标或 k_max;baseline 以 EMA 缓慢
    适应"新常态"(避免分布整体变难时永久打满预算);
  - **激活采集**: 触发检查时对训练模型做一次小批 eager FP32 前向(hook 采集,
    不碰编译路径 —— dynamo 对 module hook 会重编译/降级,实测教训)。

用法(每个 iteration,在 PPO update 之后、sync_weights_from 之前):
    cc = CoCalibrator(train_policy, swapped)        # swapped: {path: QuantLinear}
    ...
    stats = cc.step(obs_minibatch)                  # 0~k_max 步/层
    runner.sync_weights_from(train_policy)          # 新 α/β 随权重一起进推理副本
"""
from __future__ import annotations

import torch

from quant.let import LETQuantizer


class CoCalibrator:

    def __init__(
        self,
        train_model,
        swapped: dict,                  # {path: QuantLinear},须 has_let=True
        lr: float = 2e-3,
        tau: float = 1.25,              # 触发阈值: L > τ·baseline 才优化
        k_max: int = 8,                 # 单层单次最大步数(早停预算)
        sample_rows: int = 2048,        # 每次评估/优化用的激活行数
        baseline_ema: float = 0.8,      # baseline 跟踪系数
        source: str = "eager",          # 激活采集源:
                                        #   "eager" 独立小批 FP32 前向(默认,稳)
                                        #   "tap"   搭 update 前向便车,0 额外前向
                                        #           (需先 tap.install_taps 装好)
        tap_slots: dict | None = None,  # source="tap" 时 {path: slot}
    ):
        self.train_model = train_model
        self.swapped = {p: q for p, q in swapped.items() if q.has_let}
        self.paths = list(self.swapped.keys())
        self.tau, self.k_max = float(tau), int(k_max)
        self.sample_rows = sample_rows
        self.ema = baseline_ema
        self.baseline: dict[str, float | None] = {p: None for p in self.paths}
        self.total_steps = 0            # 累计优化步数(成本指标)
        self.collect_amp = False        # True: 激活采集在 bf16 autocast 下(实验用)
        self.source = source
        self.tap_slots = tap_slots or {}

        self.lets: dict[str, LETQuantizer] = {}
        self.opts: dict[str, torch.optim.Adam] = {}
        for p, ql in self.swapped.items():
            let = LETQuantizer(n_channels=ql.in_features).to(ql.weight.device)
            let.alpha.data.copy_(ql.alpha)          # warm-start 自当前校准值
            let.beta.data.copy_(ql.beta)
            self.lets[p] = let
            self.opts[p] = torch.optim.Adam(let.parameters(), lr=lr)

    # ── 激活采集: 小批 eager FP32 前向(独立于编译路径)────────────────────────
    @torch.no_grad()
    def _collect(self, obs) -> dict[str, torch.Tensor]:
        feats: dict[str, torch.Tensor] = {}
        hooks = []

        def mk(p):
            def fn(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                if x.shape[0] > self.sample_rows:
                    idx = torch.randint(0, x.shape[0], (self.sample_rows,),
                                        device=x.device)
                    x = x[idx]
                feats[p] = x
            return fn

        for p in self.paths:
            hooks.append(self.train_model.get_submodule(p).register_forward_hook(mk(p)))
        was = self.train_model.training
        self.train_model.eval()
        ctx = (torch.autocast("cuda", torch.bfloat16) if self.collect_amp
               else torch.autocast("cuda", enabled=False))
        with ctx:
            self.train_model.act_inference(obs)
        self.train_model.train(was)
        for h in hooks:
            h.remove()
        return feats

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def step(self, obs) -> dict[str, tuple[float, float, int]]:
        """评估各层重建损失,误差驱动地优化 α/β 并写回推理副本。

        返回 {path: (L_before, L_after, k_steps)}。
        注意: 不可在 inference_mode 上下文内调用(优化需要 autograd)。
        """
        assert not torch.is_inference_mode_enabled(), \
            "CoCalibrator.step 需要 autograd,勿在 inference_mode 内调用"
        if self.source == "tap":
            from .tap import TapStore     # 0 额外前向: 读 update 前向已写入的 buffer
            feats = {p: TapStore.read(self.tap_slots[p]) for p in self.paths}
            feats = {p: v for p, v in feats.items() if v is not None}
        else:
            feats = self._collect(obs)
        stats: dict[str, tuple[float, float, int]] = {}

        for p in self.paths:
            x = feats[p]
            let = self.lets[p]
            lin = self.train_model.get_submodule(p)       # FP32 master 层

            with torch.no_grad():
                _, L0t = let.block_reconstruction_loss(x, lin)
            L0 = float(L0t)
            if self.baseline[p] is None:                  # 首次: 以当前为基线
                self.baseline[p] = L0
            L, k = L0, 0

            if L0 > self.tau * self.baseline[p]:
                opt = self.opts[p]
                with torch.enable_grad():
                    while k < self.k_max:
                        _, loss = let.block_reconstruction_loss(x, lin)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        k += 1
                        L = float(loss.detach())
                        if L <= self.tau * self.baseline[p]:
                            break
                ql = self.swapped[p]                      # 写回推理副本(buffer,
                with torch.no_grad():                     #   graph 安全;随后 sync
                    ql.alpha.copy_(let.alpha.clamp_min(1e-8))   # 的 requantize 生效)
                    ql.beta.copy_(let.beta)
                self.total_steps += k
                # baseline 仅在优化后锚定到收敛水平。若未优化时也跟踪 L,
                # 阈值会随漂移水涨船高而永不触发(实测教训:温水煮青蛙失明)
                self.baseline[p] = self.ema * self.baseline[p] + (1 - self.ema) * L
            stats[p] = (L0, L, k)
        return stats

    def summary(self) -> str:
        return (f"CoCalibrator(layers={len(self.paths)}, "
                f"total_steps={self.total_steps}, tau={self.tau}, k_max={self.k_max})")

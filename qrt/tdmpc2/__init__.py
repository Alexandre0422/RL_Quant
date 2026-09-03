# -*- coding: utf-8 -*-
"""
qrt.tdmpc2 — 用 qrt 量化方法加速 TD-MPC2 世界模型(EXP-002:roofline 镜像)
══════════════════════════════════════════════════════════════════════════════
TD-MPC2 是 GLAD 在 roofline 上的镜像工作负载:
  encoder    B=1   权重访存主导 → WeightOnlyLinear(W8A16/W4A16)  ← GLAD 部署端同款
  rollout    B≈536 计算/激活主导 → QuantLinear(W8A8/IMMA)+ LET   ← GLAD 单步够不到的端

于是「INT8 价值由逐算子 roofline 位置 + 可用 kernel 原语决定」在两个真实 RL 系统上
各证一端。用法见 qrt/bench/bench_tdmpc2.py。
"""
from .world_model import (TDMPC2Config, TDMPC2WorldModel, SimNorm,
                          build_world_model)
from .quantize import (quantize_rollout, quantize_encoder, refresh_rollout,
                       rollout_linear_paths, encoder_linear_paths)

__all__ = ["TDMPC2Config", "TDMPC2WorldModel", "SimNorm", "build_world_model",
           "quantize_rollout", "quantize_encoder", "refresh_rollout",
           "rollout_linear_paths", "encoder_linear_paths"]

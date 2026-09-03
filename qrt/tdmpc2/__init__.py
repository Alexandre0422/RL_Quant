# -*- coding: utf-8 -*-
"""
qrt.tdmpc2 — 用 qrt 双网络量化 patch 加速 TD-MPC2(EXP-002:roofline 镜像)
══════════════════════════════════════════════════════════════════════════════
patch 真实 `tdmpc2.TDMPC2`(不重写网络)。双网络形态:FP32 master(`agent.model`,训练
不动)+ FP16 推理副本(rollout 世界模型量化)+ 每 iter 同步。只量化 rollout
(_dynamics/_reward/_pi);encoder 与 vmap _Qs 未量化(TODO)。

TD-MPC2 在线 MPPI 规划把 rollout 推到 batch≈536(计算/激活主导)——正是 _int_mm/IMMA
该赢、GLAD 单步(B=1)够不到的 roofline 端;与 GLAD(B=1→weight-only)镜像互补。

用法:
    from qrt.tdmpc2 import accelerate_tdmpc2
    acc = accelerate_tdmpc2(agent)      # 一行接入;acc.detach() 还原
见 qrt/bench/bench_tdmpc2.py。
"""
from .integration import accelerate_tdmpc2, TDMPC2Accelerator
from .qnormed import QuantNormedLinear, swap_rollout_int8

__all__ = ["accelerate_tdmpc2", "TDMPC2Accelerator",
           "QuantNormedLinear", "swap_rollout_int8"]

# -*- coding: utf-8 -*-
"""
qrt/tdmpc2/integration.py — accelerate_tdmpc2:TD-MPC2 双网络量化 patch(一行接入)
══════════════════════════════════════════════════════════════════════════════
把 GLAD 主线的**双网络形态**(qrt/integration.py 的 accelerate_runner)搬到 TD-MPC2,
**patch 真实 `tdmpc2.TDMPC2`,不重写网络**:

    from qrt.tdmpc2 import accelerate_tdmpc2
    agent = TDMPC2(cfg)                 # 官方 agent(建议 cfg.compile=False,我们自带加速)
    acc = accelerate_tdmpc2(agent)      # 即插即用;acc.detach() 可完全还原
    agent.act(obs) / agent.update(buf)  # 训练/推理代码零修改

双网络(与单模型双 forward 的区别见 README/记忆):
  • master = `agent.model`(FP32):训练/PPO/optimizer 完全不动,梯度保持全精度;
  • infer  = `deepcopy(agent.model).half()`(FP16,requires_grad=False):**独立权重**,
    rollout 世界模型(_dynamics/_reward/_pi)换成 QuantNormedLinear/QuantLinear(W8A8+_int_mm);
  • 路由:wrap `agent.act` —— 规划期间把 `agent.model` 临时指向 infer(TD-MPC2 的
    act/_plan/_estimate_value 全 @no_grad 且都经 self.model 调世界模型,故 rebind 即
    整条 rollout 走量化副本);act 返回后还原,`update()` 训练仍用 master;
  • 同步:wrap `agent.update` —— 每次 update 末尾 master→infer 按名 cast copy + requantize
    (替代 refit,每 iter 同步,严格 on-policy)。

**只量化 rollout**(用户明确):encoder(B=1,非 rollout)与 _Qs(vmap Ensemble,params
是 stacked TensorDict、非可 swap 的 nn.Linear)**不在本 patch**,列为 TODO:
  - _Qs:vmap 的批量 int8 GEMM 需自写/改造 kernel(_int_mm 不支持 batched ensemble 维);
  - encoder:rollout-only 口径下不量化。

roofline 意义:TD-MPC2 在线规划把 rollout 推到 batch=num_samples+num_pi_trajs(≈536),
计算/激活主导端 —— 正是 _int_mm/IMMA 该赢、GLAD 单步(B=1)够不到的区间(EXP-002)。

**本文件不 import tdmpc2**(鸭子接口:只依赖 agent.model / agent.act / agent.update /
world model 的 _dynamics/_reward/_pi)。仅 py_compile 验证;真实数值/latency 待 ps2
装 tdmpc2 后复跑(见 qrt/bench/bench_tdmpc2.py)。
"""
from __future__ import annotations

import copy
import warnings

import torch

from .qnormed import swap_rollout_int8


class TDMPC2Accelerator:
    """持有 master / infer 双网络与 swap 结果;detach() 还原 agent(消融对照)。"""

    def __init__(self, agent, act_mode: str = "dynamic_full",
                 roots=("_dynamics", "_reward", "_pi"), int8: bool = True,
                 calib_obs=None, let_steps: int = 50, verbose: bool = True):
        self.agent = agent
        self.master = agent.model                      # FP32 master(不动)

        if getattr(getattr(agent, "cfg", None), "compile", False):
            warnings.warn("[qrt-tdmpc2] agent.cfg.compile=True:官方编译的 _plan 可能"
                          "绕过 model rebind 路由;建议 cfg.compile=False,由本 patch 加速。")

        # ── 1. 推理副本(FP16,独立权重,冻结)────────────────────────────────
        infer = copy.deepcopy(self.master).to(next(self.master.parameters()).device).half()
        for p in infer.parameters():
            p.requires_grad_(False)
        infer.eval()
        self.infer = infer

        # ── 2. (可选)LET 校准:在未 swap 的 infer 上采集 rollout 激活 → fit_let ──
        let_plan = None
        if int8 and act_mode in ("dynamic", "static"):
            assert calib_obs is not None, f"act_mode={act_mode} 需 calib_obs(真实 obs 列表)"
            let_plan = self._calibrate_let(infer, roots, calib_obs, let_steps, verbose)

        # ── 3. 量化 rollout(推理副本上;master 不动)。int8=False → 纯 FP16 双网络
        #        (bench 的干净 baseline:与 +INT8 只差量化一项,隔离 fp32→fp16 混淆)。
        if int8:
            self.swapped, self.skipped = swap_rollout_int8(
                infer, roots=roots, act_mode=act_mode, let_plan=let_plan, verbose=verbose)
        else:
            self.swapped, self.skipped = {}, []

        # ── 4. 路由:wrap agent.act —— 规划期间 agent.model 指向 infer ──────────
        self._orig_act = agent.act
        _orig_act = self._orig_act
        infer_ref = self.infer

        def act_qrt(*args, **kwargs):
            saved = agent.model
            agent.model = infer_ref
            try:
                return _orig_act(*args, **kwargs)
            finally:
                agent.model = saved
        agent.act = act_qrt

        # ── 5. 同步:wrap agent.update —— 每 iter master→infer + requantize ──────
        self._orig_update = agent.update
        _orig_update = self._orig_update
        self._pairs = None

        def update_qrt(*args, **kwargs):
            out = _orig_update(*args, **kwargs)
            self.sync()
            return out
        agent.update = update_qrt

        self.sync()                                    # 建 pair + 首次对齐(幂等)
        if verbose:
            print(f"[qrt-tdmpc2] 就绪:双网络 patch,act_mode={act_mode},"
                  f"rollout 量化 {len(self.swapped)} 层(跳过 {len(self.skipped)});"
                  f"encoder/_Qs 未量化(rollout-only + vmap TODO)。")

    # ── 权重同步(替代 refit;每 iter)────────────────────────────────────────
    @torch.inference_mode()
    def sync(self):
        """master(FP32)→ infer(FP16)按名 cast copy + 全量化层 requantize。"""
        if self._pairs is None:
            src = dict(self.master.named_parameters())
            src.update(dict(self.master.named_buffers()))
            pairs = []
            for name, dst in (list(self.infer.named_parameters())
                              + list(self.infer.named_buffers())):
                s = src.get(name)
                if s is not None and s.shape == dst.shape:
                    pairs.append((dst, s))             # 量化 buffer(w8/...)无 master 对应,自动略过
            self._pairs = pairs
        for dst, s in self._pairs:
            dst.copy_(s)                               # fp32→fp16 cast
        for q in self.swapped.values():
            q.requantize_()                            # dynamic_full 下为空操作(权重量化在前向图内)

    # ── LET 校准(可选)────────────────────────────────────────────────────────
    @torch.no_grad()
    def _calibrate_let(self, infer, roots, calib_obs, let_steps, verbose):
        """在未 swap 的 infer 上跑 agent.act(真实 obs)采集 rollout 激活,逐层 fit_let。"""
        from qrt import calib
        import torch.nn as nn
        targets = {}                                   # path -> Linear
        for root in roots:
            if not hasattr(infer, root):
                continue
            seq = getattr(infer, root)
            for i in range(len(seq)):
                if isinstance(seq[i], nn.Linear) and seq[i].in_features % 8 == 0:
                    targets[f"{root}.{i}"] = seq[i]
        feats = {p: [] for p in targets}

        def mk(p):
            def fn(m, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                feats[p].append(x.float())
            return fn
        hooks = [targets[p].register_forward_hook(mk(p)) for p in targets]

        saved = self.agent.model
        self.agent.model = infer
        try:
            for obs in calib_obs:
                self.agent.act(obs)                    # 触发 rollout,hook 采激活(batch≈536)
        finally:
            self.agent.model = saved
        for h in hooks:
            h.remove()

        plan = {}
        for p, lin in targets.items():
            if not feats[p]:
                continue
            x = torch.cat(feats[p], dim=0)
            alpha, beta, s_z = calib.fit_let(x, lin, n_steps=let_steps)
            plan[p] = {"s_z": s_z, "alpha": alpha, "beta": beta}
            if verbose:
                print(f"  [LET] {p}  K={lin.in_features}  samples={x.shape[0]}")
        return plan

    def detach(self):
        """还原 agent(移除 act/update 包装,回到原始路径;master 从未被改)。"""
        self.agent.act = self._orig_act
        self.agent.update = self._orig_update


def accelerate_tdmpc2(agent, **kwargs) -> TDMPC2Accelerator:
    """一行接入入口(参数见 TDMPC2Accelerator)。返回值保留以便 detach/inspect。

    默认 act_mode='dynamic_full'(零校准,免真实 obs);要真 LET 传
    act_mode='dynamic', calib_obs=[obs, ...](真实 rollout obs)。
    """
    return TDMPC2Accelerator(agent, **kwargs)

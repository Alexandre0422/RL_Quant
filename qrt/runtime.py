# -*- coding: utf-8 -*-
"""
runtime.py — InferenceRunner: GLAD rollout 推理加速器
══════════════════════════════════════════════════════════════════════════════
三种形态(实测数据见 PLAN_compile_quant.md §1):

  "max-autotune"  inductor 全图融合 + 内置 CUDAGraph Trees。
                  FP16 实测 1.723ms/步(3.60× vs FP32 eager)。默认。
  "manual-graph"  max-autotune-no-cudagraphs 编译 + 手动 torch.cuda.CUDAGraph。
                  FP16 实测 1.640ms/步。静态 IO buffer,完全可控,
                  适合嵌入 Isaac Lab 等已有 stream/graph 的复杂环境。
  "eager"         不编译(调试/对照)。

权重热更新(替代 TRT refit):
  - FP16/FP32 全精度层: 无需任何操作。compile/graph 持有的是参数 tensor 本身,
    optimizer 的 in-place 更新立即生效(实测更新后首调 0.6ms,无重编译)。
  - INT8 层(QuantLinear): PPO update 后调一次 refresh_quant(),
    GPU 上 re-quantize + copy_ 进 buffer(<0.1ms),graph 安全(data_ptr 不变)。

已知约束(实测,勿改):
  - 必须 mode="max-autotune"。inductor 默认模式与 reduce-overhead 在本模型
    触发 CUDA invalid configuration;freezing 触发 view/stride 错误。
  - rollout 前 model.eval();PPO update 的 train 态前向走 eager(或另行编译),
    dynamo 对 train/eval 两态各缓存一图,不反复重编译。
  - manual-graph 输出是静态 buffer,下一次 act() 前必须消费(或 clone_outputs=True)。
"""
from __future__ import annotations
import warnings

import torch

from .qlinear import QuantLinear

_COMPILE_MODES = {
    "max-autotune": "max-autotune",
    "manual-graph": "max-autotune-no-cudagraphs",
}

_rng_keepalive_graph = None


def _ensure_graph_rng_state():
    """在普通(非 inference_mode)上下文预注册 CUDA RNG 的 graph-safe state,
    并**保持一个哨兵 graph 永久存活**。

    背景: CUDAGeneratorState 的 seed/offset extragraph tensor 在"首个 graph
    注册时分配、全部 graph 注销后释放重建"。若(重)分配发生在 inference_mode
    内的 capture(成为 inference tensor),之后任何 grad 模式下的 capture(如
    编译训练图的 cudagraph trees record)都会报 'Inplace update to inference
    tensor outside InferenceMode';反之普通 tensor 在两种上下文均合法。
    哨兵 graph 不析构 → registered_graphs 永不清空 → state 锁定为普通 tensor。
    """
    global _rng_keepalive_graph
    if _rng_keepalive_graph is not None:
        return
    with torch.no_grad():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pass
        _rng_keepalive_graph = g      # 永久持有,勿删


class InferenceRunner:
    """
    用法(rollout):
        runner = InferenceRunner(model, mode="max-autotune", with_critic=True)
        runner.warmup(example_policy_obs, example_critic_obs)   # 触发编译/捕获
        mean  = runner.act(policy_obs)        # [B, num_actions] 动作均值
        value = runner.evaluate(critic_obs)   # [B, 1]
        # 探索采样在图外: Normal(mean, model.std.expand_as(mean)).sample()

    PPO update 后:
        runner.refresh_quant()    # 仅当模型含 QuantLinear;全精度模型无需调用
    """

    def __init__(
        self,
        model,
        mode: str = "max-autotune",
        with_critic: bool = False,
        clone_outputs: bool = False,
    ):
        if mode not in ("max-autotune", "manual-graph", "eager"):
            raise ValueError(f"unknown mode: {mode}")
        if model.training:
            warnings.warn("[qrt] model 处于 train 态;rollout 推理应先 model.eval()")
        _ensure_graph_rng_state()

        self.model = model
        self.mode = mode
        self.with_critic = with_critic
        self.clone_outputs = clone_outputs
        # 收集所有可热刷新的量化模块(QuantLinear / QuantCNN 等,鸭子类型)
        self._quant_layers = [m for m in model.modules()
                              if m is not model and hasattr(m, "requantize_")]

        # actor 路径: act_inference 只读 obs["policy_obs"](dynamo 实测 0 break)
        def f_act(po):
            return model.act_inference({"policy_obs": po})[0]

        # critic 路径: 重组 evaluate,绕开其中 isnan().any() 的
        # data-dependent 分支(graph break / capture 失败源),不修改原模型文件
        def f_val(co):
            c = model.critic_obs_normalizer(co)
            enc, _, _, _ = model._encode_terrain(c)
            return model.critic(enc)

        if mode == "eager":
            self._act_fn = f_act
            self._val_fn = f_val if with_critic else None
        else:
            cmode = _COMPILE_MODES[mode]
            self._act_fn = torch.compile(f_act, mode=cmode, fullgraph=True, dynamic=False)
            self._val_fn = (torch.compile(f_val, mode=cmode, fullgraph=True, dynamic=False)
                            if with_critic else None)

        # manual-graph 状态(warmup 时填充)
        self._g_act = self._g_val = None
        self._static_po = self._static_co = None
        self._out_act = self._out_val = None
        # refresh_quant / sync_weights_from 的 CUDA Graph 缓存(首次调用时捕获)
        self._g_refresh = None
        self._g_sync = None
        self._sync_pairs = None

    # ── 编译触发 / 手动捕获 ───────────────────────────────────────────────────

    @torch.inference_mode()
    def warmup(self, example_po, example_co=None, n_warmup: int = 3):
        """触发编译(冷 ~35s/缓存 ~8s,一次性);manual-graph 模式同时完成捕获。"""
        for _ in range(n_warmup):
            self._act_fn(example_po)
        if self._val_fn is not None:
            if example_co is None:
                raise ValueError("with_critic=True 时 warmup 需提供 example_co")
            for _ in range(n_warmup):
                self._val_fn(example_co)
        torch.cuda.synchronize()

        if self.mode == "manual-graph":
            self._static_po = example_po.clone()
            self._g_act, self._out_act = self._capture(self._act_fn, self._static_po)
            if self._val_fn is not None:
                self._static_co = example_co.clone()
                self._g_val, self._out_val = self._capture(self._val_fn, self._static_co)
        return self

    @staticmethod
    def _capture(fn, static_in):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(static_in)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(static_in)
        return g, out

    # ── 热路径 ────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def act(self, policy_obs):
        """actor 前向 → 动作均值 [B, num_actions]。"""
        if self.mode == "manual-graph":
            if self._g_act is None:
                self.warmup(policy_obs)
            self._static_po.copy_(policy_obs)
            self._g_act.replay()
            return self._out_act.clone() if self.clone_outputs else self._out_act
        return self._act_fn(policy_obs)

    @torch.inference_mode()
    def evaluate(self, critic_obs):
        """critic 前向 → value [B, 1]。需 with_critic=True。"""
        if self._val_fn is None and self._g_val is None:
            raise RuntimeError("InferenceRunner(with_critic=True) 才能调用 evaluate")
        if self.mode == "manual-graph":
            self._static_co.copy_(critic_obs)
            self._g_val.replay()
            return self._out_val.clone() if self.clone_outputs else self._out_val
        return self._val_fn(critic_obs)

    @torch.inference_mode()
    def act_sample(self, policy_obs):
        """便捷方法: mean + Normal 采样(采样 kernel 在图外,~μs 级)。"""
        mean = self.act(policy_obs)
        std = self.model.std.expand_as(mean) if hasattr(self.model, "std") \
            else torch.exp(self.model.log_std).expand_as(mean)
        return torch.normal(mean, std)

    # ── 权重同步(替代 TRT refit)──────────────────────────────────────────────

    @torch.inference_mode()
    def refresh_quant(self, use_graph: bool = True):
        """PPO update 后刷新所有 QuantLinear 的 INT8 buffer(替代 TRT refit)。

        use_graph=True: 首次调用时把全部层的 requantize 序列捕获为一张
        CUDA Graph(读 weight 指针/写 buffer 指针均固定),后续 replay,
        消除 ~50 个微 kernel 的 launch 开销(实测 2.7ms → 亚毫秒级)。
        """
        if not self._quant_layers:
            return
        if not use_graph:
            for q in self._quant_layers:
                q.requantize_()
            return
        if self._g_refresh is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    for q in self._quant_layers:
                        q.requantize_()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for q in self._quant_layers:
                    q.requantize_()
            self._g_refresh = g
        self._g_refresh.replay()

    @torch.inference_mode()
    def sync_weights_from(self, src_model, use_graph: bool = True):
        """从 FP32 训练模型(master weights)同步参数与 buffer 到本推理模型,
        并刷新全部 QuantLinear 的 INT8 buffer —— 一次调用完成整个权重交接。

        - 同名 cast copy(fp32→fp16),含 BN running stats(train 态会更新,必须同步);
        - 推理模型独有的量化 buffer(w8/dq_scale/...)由 requantize 重算,不参与配对;
        - 本推理模型被视为纯推理副本:首次调用时参数 requires_grad 关闭
          (训练发生在 src_model 上;inference_mode 内 in-place 写参数也因此合法);
        - use_graph=True 时整个序列(copy×N + requantize)捕获为一张 CUDA Graph,
          src/dst 指针均固定(optimizer in-place 更新),replay 安全。
          建议在 warmup() 编译之前先调用一次(标准顺序: 构建→swap→sync→warmup)。
        """
        if self._sync_pairs is None:
            for p in self.model.parameters():
                p.requires_grad_(False)          # 推理副本不参与 autograd
            dst_p = dict(self.model.named_parameters())
            dst_b = dict(self.model.named_buffers())
            pairs = []
            for name, s_ in list(src_model.named_parameters()) + list(src_model.named_buffers()):
                d = dst_p.get(name)
                if d is None:
                    d = dst_b.get(name)
                if d is not None and d.shape == s_.shape:
                    pairs.append((d, s_))
            self._sync_pairs = pairs

        def _do_sync():
            for d, s_ in self._sync_pairs:
                d.copy_(s_)
            for q in self._quant_layers:
                q.requantize_()

        if not use_graph:
            _do_sync()
            return
        if self._g_sync is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    _do_sync()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _do_sync()
            self._g_sync = g
        self._g_sync.replay()

    def __repr__(self):
        return (f"InferenceRunner(mode={self.mode}, with_critic={self.with_critic}, "
                f"quant_layers={len(self._quant_layers)})")


class TrainStepRunner:
    """
    PPO update 加速器(P1.5): bf16 autocast + torch.compile 训练前向
    (AOTAutograd 自动生成并编译反向图)+ fused Adam。

    要求 model 参数为 FP32(master weights,标准实践 —— 纯 FP16 参数直接
    训练会 nan,漂移实验已证);rollout 用独立的 FP16/INT8 推理模型,每次
    update 后调 InferenceRunner.sync_weights_from(train_model) 交接权重。

    用法(bench 协议的 dummy loss;真实 PPO 用 forward_mean 自组 loss):
        tsr = TrainStepRunner(train_model, lr=1e-4)
        train_model.train()
        for mb_po in minibatches:
            tsr.dummy_step(mb_po)
        train_model.eval()
        infer_runner.sync_weights_from(train_model)
    """

    def __init__(self, model, lr: float = 1e-4, mode: str = "max-autotune",
                 amp_dtype=torch.bfloat16, optimizer=None):
        _ensure_graph_rng_state()
        self.model = model
        self.amp_dtype = amp_dtype
        if optimizer is None:
            try:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
            except (RuntimeError, ValueError):
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.optimizer = optimizer

        def f_train(po):
            obs = {"policy_obs": po}
            actor_obs = model.actor_obs_normalizer(model.get_actor_obs(obs))
            enc, _, _, _ = model._encode_terrain(actor_obs)
            return model.actor(enc)

        self._fwd = torch.compile(f_train, mode=mode, dynamic=False)

    def forward_mean(self, po):
        """train 态前向(bf16 autocast,编译)→ 动作均值。可 backward。"""
        with torch.autocast("cuda", self.amp_dtype):
            return self._fwd(po)

    def dummy_step(self, po):
        """bench 协议的 dummy update(与 bench_refit_train.ppo_update_step 同构)。"""
        mean = self.forward_mean(po)
        loss = mean.float().pow(2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss

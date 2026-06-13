# -*- coding: utf-8 -*-
"""
integration.py — rsl_rl / Isaac Lab 一行接入层
══════════════════════════════════════════════════════════════════════════════
在 OnPolicyRunner 创建之后、runner.learn() 之前调用:

    from qrt.integration import accelerate_runner
    acc = accelerate_runner(runner, int8_linear=True, int8_cnn=True,
                            graph_rewrite=True)        # 即插即用
    runner.learn(...)                                   # 训练代码零修改

设计(双模型形态,P1.5–P3 已验证):
  - 训练侧: runner.alg.policy(FP32 master)与 optimizer/PPO 完全不动;
  - 推理侧: deepcopy 出 FP16 副本 → 可选 LET-INT8(Linear)/QuantCNN(conv)
    /encoder 代数重写 → torch.compile(max-autotune);
  - 路由: monkeypatch policy.update_distribution / policy.evaluate ——
    rollout(inference_mode 内,含 train 态 Gumbel 探索)走编译副本,
    PPO update(梯度态)走原始路径。Normal 分布构造/采样/log_prob 原样;
  - 同步: wrap alg.update,每次 update 末尾自动 sync_weights_from
    (参数+BN buffer cast copy + INT8 requantize,单张 CUDA Graph,
    **每个 iteration 都同步,无 K-step 滞后,严格 on-policy**);
  - BN 语义: 推理副本的 BN 固定 eval(running stats,由训练侧 update 前向
    持续更新并经 sync 传入);Gumbel/topk 探索保留(train 态 trace)。
    这是相对原版(rollout 用 batch stats)唯一的语义近似,真实训练建议 A/B。

量化校准: 默认用 runner 当前 env 的真实 obs(collect_calib_obs 滚动收集),
即真实分布校准 —— TRT 路线已知问题 3 在此闭环。
"""
from __future__ import annotations

import copy
import time
import types

import torch
from torch.distributions import Normal

from .runtime import InferenceRunner, _ensure_graph_rng_state
from . import swap as _swap
from . import calib as _calib

INT8_PATHS_DEFAULT = ["actor.0", "actor.2", "actor.4", "actor.6",
                      "actor_proprio_embedding", "query_projector"]


@torch.inference_mode()
def collect_calib_obs(runner, n_steps: int = 8):
    """用当前策略在真实 env 上滚动 n 步,收集 obs 批次(校准数据=真实分布)。
    注: 会推进 env 状态;在 learn() 之前调用,对训练无实质影响。"""
    obs_list = []
    obs = runner.env.get_observations().to(runner.device)
    policy = runner.alg.policy
    was_training = policy.training
    policy.eval()
    for _ in range(n_steps):
        obs_list.append({k: v.clone() for k, v in obs.items()})
        actions = policy.act(obs)
        obs, _, _, _ = runner.env.step(actions.to(runner.env.device))
        obs = obs.to(runner.device)
    policy.train(was_training)
    return obs_list


class QRTAccelerator:
    """持有推理副本与编译产物;detach() 可完全还原 runner(消融对照)。"""

    def __init__(
        self,
        runner,
        int8_linear: bool = True,
        int8_cnn: bool = True,
        graph_rewrite: bool = True,
        accelerate_update: bool = True,     # PPO update 的梯度前向也编译(AOT 反向)
        rollout_training: bool = True,      # 保留 Gumbel topk 探索(原版语义)
        update_amp: str | None = "bf16",    # update 前向 autocast: "bf16"/"fp32"/None
        act_mode: str = "dynamic",          # P5: INT8 激活 scale 策略(dynamic=图内
                                            # 现算,免校准漂移;static=P1 校准+EMA)
        cocalib: bool = False,              # P6: 每 iter 对 α/β 做 1 步伴随跟踪
                                            # (fixed-1,~5-10ms/iter;实验结论:
                                            # 以 periodic 重校准 5% 的成本取得更优
                                            # 更平滑的精度保持;默认关,Isaac A/B
                                            # 确认量化漂移影响 reward 后开启)
        cocalib_source: str = "eager",      # "eager" 独立小批前向(稳,默认)/
                                            # "tap" 搭 update 前向便车 0 额外前向
                                            # (需 update_amp 路径用 no-cudagraphs)
        int8_paths: list[str] | None = None,
        calib_obs: list | None = None,      # None → collect_calib_obs(真实 obs)
        let_steps: int = 200,
        mode: str = "max-autotune",
        verbose: bool = True,
    ):
        _ensure_graph_rng_state()
        self.runner = runner
        self.alg = runner.alg
        self.policy = runner.alg.policy     # FP32 master(不动)
        dev = runner.device

        # ── 1. 推理副本(FP16)────────────────────────────────────────────────
        t0 = time.perf_counter()
        infer = copy.deepcopy(self.policy).half().to(dev)
        self.infer = infer
        self.rollout_training = rollout_training
        self.set_infer_training(rollout_training)

        # ── 2. 量化校准(真实 obs)+ swap ────────────────────────────────────
        if int8_linear or int8_cnn:
            if calib_obs is None:
                calib_obs = collect_calib_obs(runner)
            calib_obs16 = [{k: v.half() for k, v in o.items()} for o in calib_obs]
            _it = iter(calib_obs16 * 64)
            obs_fn = lambda: next(_it)      # noqa: E731
        if int8_linear:
            paths = int8_paths or INT8_PATHS_DEFAULT
            paths = [p for p in paths if self._has_linear(infer, p)]
            plan = _calib.make_plan(infer, paths, obs_fn=obs_fn,
                                    n_calib=len(calib_obs), use_let=True,
                                    let_steps=let_steps, act_mode=act_mode,
                                    verbose=False)
            self.swapped = _swap.apply(infer, plan)
        if int8_cnn:
            from .qconv import swap_cnn, calib_cnn_scale
            s_cnn = calib_cnn_scale(infer, obs_fn, n_batches=len(calib_obs))
            swap_cnn(infer, s_cnn)
        if graph_rewrite:
            from .encoder_opt import patch_encoder
            patch_encoder(infer)

        # ── 3. 编译 actor / critic 路径(底层函数,绕开 normalizer 重复)─────
        def f_act(actor_obs):
            enc, _, _, _ = infer._encode_terrain(actor_obs)
            return infer.actor(enc)

        def f_val(critic_obs):
            enc, _, _, _ = infer._encode_terrain(critic_obs)
            return infer.critic(enc)

        self._cf_act = torch.compile(f_act, mode=mode, fullgraph=True, dynamic=False)
        self._cf_val = torch.compile(f_val, mode=mode, fullgraph=True, dynamic=False)

        # InferenceRunner 仅复用其 sync/refresh 设施(不用其 act 包装)
        self._sync_runner = InferenceRunner(infer, mode="eager")

        # ── 3b. (可选)PPO update 的梯度前向编译(FP32 master 模型上,
        #         AOTAutograd 自动生成反向;PPO loss/optimizer 代码不动)。
        #         update_amp="bf16" 时前向在 bf16 autocast 内(图内 GEMM bf16,
        #         AOT 反向同样 bf16;参数与梯度仍 fp32),mean/value 出图即
        #         .float() —— ratio/KL/entropy/loss 全部保持 fp32 数值路径。──
        self._cf_act_train = self._cf_val_train = None
        self._amp_dtype = {"bf16": torch.bfloat16, "fp32": None,
                           None: None}[update_amp]
        # cocalib="tap": 在 trainer 目标层装 TapLinear(零额外前向采集);须在
        # 编译 f_act_t 之前(tap 才能进图)且训练前向用 no-cudagraphs(cudagraph
        # 不重放 custom-op 副作用,spike 实测)
        self._tap_slots = None
        train_mode = mode
        if cocalib and cocalib_source == "tap" and int8_linear:
            from .tap import install_taps
            paths_t = [p for p in (int8_paths or INT8_PATHS_DEFAULT)
                       if self._has_linear(self.policy, p)]
            self._tap_slots = install_taps(self.policy, paths_t)
            train_mode = "max-autotune-no-cudagraphs"

        if accelerate_update:
            policy_ = self.policy
            if graph_rewrite:
                from .encoder_opt import patch_encoder
                patch_encoder(policy_)            # train 分支等价性已验证

            def f_act_t(actor_obs):
                enc, _, _, _ = policy_._encode_terrain(actor_obs)
                return policy_.actor(enc)

            def f_val_t(critic_obs):
                enc, _, _, _ = policy_._encode_terrain(critic_obs)
                return policy_.critic(enc)

            self._cf_act_train = torch.compile(f_act_t, mode=train_mode,
                                               fullgraph=True, dynamic=False)
            self._cf_val_train = torch.compile(f_val_t, mode=train_mode,
                                               fullgraph=True, dynamic=False)

        # ── 4. 路由: monkeypatch update_distribution / evaluate ────────────────
        policy = self.policy
        self._orig_ud = policy.update_distribution
        self._orig_eval = policy.evaluate
        cf_act, cf_val = self._cf_act, self._cf_val
        cf_act_t, cf_val_t = self._cf_act_train, self._cf_val_train
        amp = self._amp_dtype

        def _make_dist(self_p, mean):
            if self_p.noise_std_type == "scalar":
                std = self_p.std.expand_as(mean)
            else:
                std = torch.exp(self_p.log_std).expand_as(mean)
            self_p.distribution = Normal(mean, std)

        def _train_fwd(cf, x):
            if amp is not None:
                with torch.autocast("cuda", amp):
                    out = cf(x)
                return out.float()          # PPO 数值路径回 fp32
            return cf(x)

        def update_distribution_qrt(self_p, actor_obs):
            if torch.is_inference_mode_enabled():
                _make_dist(self_p, cf_act(actor_obs.half()).float())
            elif cf_act_t is not None:
                _make_dist(self_p, _train_fwd(cf_act_t, actor_obs))
            else:
                self._orig_ud(actor_obs)

        def evaluate_qrt(self_p, obs, **kwargs):
            if torch.is_inference_mode_enabled():
                critic_obs = self_p.critic_obs_normalizer(self_p.get_critic_obs(obs))
                return cf_val(critic_obs.half()).float()
            if cf_val_t is not None:
                critic_obs = self_p.critic_obs_normalizer(self_p.get_critic_obs(obs))
                return _train_fwd(cf_val_t, critic_obs)
            return self._orig_eval(obs, **kwargs)

        policy.update_distribution = types.MethodType(update_distribution_qrt, policy)
        policy.evaluate = types.MethodType(evaluate_qrt, policy)

        # ── 4b. (可选)P6 伴随校准器: fixed-1 连续跟踪 α/β ─────────────────
        self._cc = None
        self._cc_obs = None
        if cocalib and int8_linear and getattr(self, "swapped", None):
            from .cocalib import CoCalibrator
            self._cc = CoCalibrator(self.policy, self.swapped, tau=0.0, k_max=1,
                                    source=cocalib_source,
                                    tap_slots=self._tap_slots)
            # eager 源: 复用初始校准 obs(tap 源不需要 obs,读 update 前向 buffer)
            self._cc_obs = [{k: v.float() for k, v in o.items()} for o in calib_obs]

        # ── 5. wrap alg.update: 每 iter 末尾自动 (cocalib→) sync ───────────────
        self._orig_update = self.alg.update
        self._cc_it = 0

        def update_qrt():
            out = self._orig_update()
            if self._cc is not None:
                self._cc.step(self._cc_obs[self._cc_it % len(self._cc_obs)])
                self._cc_it += 1
            self._sync_runner.sync_weights_from(self.policy)
            return out

        self.alg.update = update_qrt
        if verbose:
            n_q = len(self._sync_runner._quant_layers)
            print(f"[qrt] accelerate_runner 就绪({time.perf_counter()-t0:.1f}s): "
                  f"int8_linear={int8_linear} int8_cnn={int8_cnn} "
                  f"rewrite={graph_rewrite} quant_modules={n_q} "
                  f"rollout_training={rollout_training}")

    def set_infer_training(self, flag: bool):
        """切换推理副本的 train/eval(Gumbel 探索开关);BN 始终保持 eval
        (running stats 语义,见模块 docstring)。"""
        self.infer.train(flag)
        for mod in self.infer.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()

    @staticmethod
    def _has_linear(model, path):
        try:
            return isinstance(model.get_submodule(path), torch.nn.Linear)
        except AttributeError:
            return False

    @torch.inference_mode()
    def warmup(self, obs=None):
        """触发编译(可选;不调用则首个 rollout 步承担编译时间)。"""
        if obs is None:
            obs = self.runner.env.get_observations().to(self.runner.device)
        p = self.policy
        actor_obs = p.actor_obs_normalizer(p.get_actor_obs(obs))
        critic_obs = p.critic_obs_normalizer(p.get_critic_obs(obs))
        t0 = time.perf_counter()
        for _ in range(3):
            self._cf_act(actor_obs.half())
            self._cf_val(critic_obs.half())
        torch.cuda.synchronize()
        self._sync_runner.sync_weights_from(self.policy)
        print(f"[qrt] warmup 完成({time.perf_counter()-t0:.1f}s,一次性)")
        return self

    def detach(self):
        """还原 runner(移除全部 monkeypatch,训练回到原始路径)。"""
        self.policy.update_distribution = self._orig_ud
        self.policy.evaluate = self._orig_eval
        self.alg.update = self._orig_update


def accelerate_runner(runner, **kwargs) -> QRTAccelerator:
    """一行接入入口(参数见 QRTAccelerator)。返回值保留以便 detach/调试。"""
    return QRTAccelerator(runner, **kwargs)

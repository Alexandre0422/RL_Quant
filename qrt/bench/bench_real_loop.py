#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_real_loop.py — 真实 rsl_rl OnPolicyRunner 全链路验证(mock env)
══════════════════════════════════════════════════════════════════════════════
与 Isaac Lab 训练唯一的差别是 env.step 由 mock 地形 obs 生成器代替物理仿真
(网络栈代码路径 100% 真实: OnPolicyRunner.learn / PPO act-storage-GAE-update /
TensorDict storage / Gumbel train 态 rollout / adaptive-KL 5epoch×4minibatch)。

配置(QRT_MODE 环境变量):
  base       原始 FP32 训练(基线)
  rollout    只加速 rollout(int8+cnn+rewrite,update 原始)
  full       完全体(+update 梯度前向编译,bf16 autocast;PPO loss 仍 fp32)
  full_fp32  update 编译但 fp32(对照)

验证项: ① 接入后推理路径 vs 原始路径的 action_mean 等价性
        ② N iter 训练后参数健全 + 学习信号(|action| 下降,与基线同向)
        ③ 每 iter 耗时(扣除 mock env 时间 = 纯网络栈)
运行(远程): QRT_MODE=base python3 qrt/bench/bench_real_loop.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import WORKDIR, make_terrain_obs   # noqa: F401(mock/path 副作用)

import time
import torch

from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from qrt.integration import accelerate_runner

MODE = os.environ.get("QRT_MODE", "base")
# 验证规模默认 2048: 共享 GPU 上他人占 ~6GB,真实协议 update(B*24/4 样本
# FP32 反向)在 B=4096 时峰值 ~15GB 会 OOM;独占 24GB 卡可设 QRT_B=4096
B = int(os.environ.get("QRT_B", "2048"))
N_WARM, N_MEAS = 2, 8


class MockG1Env:
    """接口对齐 rsl_rl VecEnv;物理仿真用结构化地形 obs 生成器代替。"""

    def __init__(self, num_envs=B, device="cuda"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_actions = 29
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_time = 0.0
        self._obs = self._gen_obs()

    def _gen_obs(self):
        o = make_terrain_obs(torch.float32, batch=self.num_envs)
        return TensorDict(
            {"policy_obs": o["policy_obs"], "critic_obs": o["critic_obs"]},
            batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self._obs

    def step(self, actions):
        t0 = time.perf_counter()
        self.episode_length_buf += 1
        # 可学信号: 惩罚动作幅度 → 策略应学会缩小 |action|
        rew = (-0.05 * actions.pow(2).mean(dim=-1)
               + 0.02 * torch.randn(self.num_envs, device=self.device))
        dones = ((torch.rand(self.num_envs, device=self.device) < 0.002)
                 | (self.episode_length_buf >= self.max_episode_length))
        self.episode_length_buf[dones] = 0
        self._obs = self._gen_obs()
        torch.cuda.synchronize()
        self.env_time += time.perf_counter() - t0
        return self._obs, rew, dones, {}


def make_cfg():
    return {
        "num_steps_per_env": 24,
        "save_interval": 10 ** 9,
        "seed": 1,
        "obs_groups": {"policy": ["policy_obs"], "critic": ["critic_obs"]},
        "policy": {
            "class_name": "ActorCriticEncoder",
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "map_scan_dim": (33, 21, 3),
            "mha_dim": 64,
            "num_heads": 16,
            "cnn_downsample": True,
            "use_global_context": True,
            "topk": 32,
        },
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1.0e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
        },
    }


@torch.inference_mode()
def mean_action_norm(policy, env):
    obs = env.get_observations()
    was = policy.training
    policy.eval()
    a = policy.act_inference(obs)[0]      # act_inference 接收 obs dict
    policy.train(was)
    return a.abs().mean().item()


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    import tensordict as td
    print(f"tensordict: {td.__version__}  MODE={MODE}  B={B}  "
          f"协议: 24 步 rollout + 5 epoch × 4 minibatch update")

    env = MockG1Env()
    runner = OnPolicyRunner(env, make_cfg(), log_dir=None, device="cuda")
    policy = runner.alg.policy
    # log_dir=None 垫片: rsl_rl 的 store_code_state/logger_type 在无日志时缺防护
    import rsl_rl.runners.on_policy_runner as _opr
    _opr.store_code_state = lambda *a, **k: []
    runner.logger_type = "tensorboard"

    acc = None
    if MODE != "base":
        # MODE: rollout / full_fp32 / full / full_cc(eager 伴随校准) /
        #       full_cc_tap(tap 0 额外前向伴随校准)
        cc = MODE.startswith("full_cc")
        src = "tap" if MODE == "full_cc_tap" else "eager"
        acc = accelerate_runner(
            runner,
            int8_linear=True, int8_cnn=True, graph_rewrite=True,
            accelerate_update=MODE.startswith("full"),
            update_amp=("fp32" if MODE == "full_fp32" else "bf16"),
            cocalib=cc, cocalib_source=src,
        )
        acc.warmup()
        # ① 等价性单点对拍 —— 必须在 eval 态做(train 态两条路径各自采样
        # Gumbel,topk 随机不同,差异是探索噪声而非量化误差)
        obs = env.get_observations()
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
        acc.set_infer_training(False)
        was_training = policy.training
        policy.eval()
        with torch.inference_mode():
            policy.update_distribution(actor_obs)      # qrt 编译路径(eval 图)
            mean_fast = policy.distribution.mean.clone()
        with torch.no_grad():
            acc._orig_ud(actor_obs.float())            # 原始 FP32 路径
            mean_ref = policy.distribution.mean
        policy.train(was_training)
        acc.set_infer_training(acc.rollout_training)
        cos = torch.nn.functional.cosine_similarity(
            mean_fast.flatten().float(), mean_ref.flatten().float(), dim=0).item()
        print(f"[①] 接入等价性(eval 态,纯量化+fp16 误差): "
              f"CosSim={cos:.6f}  max|diff|={(mean_fast - mean_ref).abs().max().item():.4f}")

    if MODE.startswith("full"):
        # ①b 梯度保真度: 同数据同 loss,qrt update 路径 vs 原始 FP32 eager,
        # 参数梯度方向 cosine(eval 态,消除 Gumbel 随机性)
        obs = env.get_observations()
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs)).detach()
        critic_obs = policy.critic_obs_normalizer(policy.get_critic_obs(obs)).detach()
        policy.eval()

        def grad_of(use_qrt):
            policy.zero_grad()
            if use_qrt:
                policy.update_distribution(actor_obs)      # 编译(bf16/fp32)分支
                v = policy.evaluate(obs)
            else:
                acc._orig_ud(actor_obs)
                v = acc._orig_eval(obs)
            loss = policy.distribution.mean.pow(2).mean() + v.pow(2).mean()
            loss.backward()
            return torch.cat([p.grad.flatten().float()
                              for p in policy.parameters() if p.grad is not None]).clone()

        g_ref = grad_of(False)
        g_qrt = grad_of(True)
        gcos = torch.nn.functional.cosine_similarity(g_ref, g_qrt, dim=0).item()
        policy.zero_grad()
        policy.train()
        print(f"[①b] 梯度保真度(update 路径 vs FP32 eager): cos={gcos:.6f}")

    a0 = mean_action_norm(policy, env)

    # ── warmup learn(吸收编译: rollout 图 + update 图首次触发)──────────────
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=N_WARM)
    torch.cuda.synchronize()
    print(f"warmup {N_WARM} iter(含编译): {time.perf_counter()-t0:.1f}s")

    # ── 计时 learn ───────────────────────────────────────────────────────────
    env.env_time = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=N_MEAS)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    per_iter = wall / N_MEAS * 1000
    env_per_iter = env.env_time / N_MEAS * 1000

    # ── ② 健全性 + 学习信号 ──────────────────────────────────────────────────
    bad = [n for n, p in policy.named_parameters() if not torch.isfinite(p).all()]
    a1 = mean_action_norm(policy, env)
    print(f"\n[②] 参数健全: {'全部有限 ✓' if not bad else f'❌ {bad}'}")
    print(f"    学习信号: mean|action| {a0:.4f} → {a1:.4f}"
          f"(动作幅度惩罚下应下降)")
    print(f"\n[③] 每 iter: {per_iter:.1f}ms(其中 mock env {env_per_iter:.1f}ms,"
          f"纯网络栈 ≈ {per_iter - env_per_iter:.1f}ms)")
    print(f"RESULT {MODE} per_iter={per_iter:.1f} env={env_per_iter:.1f} "
          f"net={per_iter - env_per_iter:.1f} a0={a0:.4f} a1={a1:.4f}")


if __name__ == "__main__":
    main()

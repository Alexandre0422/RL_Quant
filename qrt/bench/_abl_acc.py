#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接入等价性消融: 定位 int8_linear / int8_cnn / rewrite 各自的精度贡献。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make_terrain_obs   # noqa: F401

import torch
import torch._dynamo as dynamo
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from qrt.integration import accelerate_runner
from bench_real_loop import MockG1Env, make_cfg

COMBOS = [
    ("fp16+compile only", dict(int8_linear=False, int8_cnn=False, graph_rewrite=False)),
    ("+rewrite",          dict(int8_linear=False, int8_cnn=False, graph_rewrite=True)),
    ("+int8_linear",      dict(int8_linear=True,  int8_cnn=False, graph_rewrite=False)),
    ("+int8_cnn",         dict(int8_linear=False, int8_cnn=True,  graph_rewrite=False)),
    ("全家桶",             dict(int8_linear=True,  int8_cnn=True,  graph_rewrite=True)),
]


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for tag, kw in COMBOS:
        dynamo.reset()
        torch.manual_seed(0)
        env = MockG1Env()
        runner = OnPolicyRunner(env, make_cfg(), log_dir=None, device="cuda")
        policy = runner.alg.policy
        acc = accelerate_runner(runner, accelerate_update=False, verbose=False, **kw)
        obs = env.get_observations()
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
        acc.set_infer_training(False)
        policy.eval()
        with torch.inference_mode():
            policy.update_distribution(actor_obs)
            mean_fast = policy.distribution.mean.clone()
        with torch.no_grad():
            acc._orig_ud(actor_obs.float())
            mean_ref = policy.distribution.mean
        cos = torch.nn.functional.cosine_similarity(
            mean_fast.flatten().float(), mean_ref.flatten().float(), dim=0).item()
        mad = (mean_fast - mean_ref).abs().max().item()
        print(f"{tag:<22} CosSim={cos:.6f}  max|diff|={mad:.4f}")
        del env, runner, acc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

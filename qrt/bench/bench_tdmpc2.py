#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_tdmpc2.py — TD-MPC2 双网络量化 patch 的 roofline 实验(EXP-002)
══════════════════════════════════════════════════════════════════════════════
在**真实 `tdmpc2.TDMPC2`** 上量 `accelerate_tdmpc2` 的效果:规划(agent.act)单步延迟
baseline(FP16 rollout)vs +INT8 rollout,扫 num_samples(TD-MPC2 的 batch 是超参 →
把 rollout 在 roofline 上从算力不足推到计算主导)。

论点:GLAD 部署 B=1(EXP-001 → weight-only);TD-MPC2 在线 MPPI 规划把 rollout 推到
batch=num_samples+num_pi_trajs(≈536,计算/激活主导)—— 正是 _int_mm/IMMA 该赢、GLAD
单步够不到的 roofline 端。ΔINT8<0 = INT8 rollout linear 在该 batch 净正。

前置(远程 ps2,需装 tdmpc2 及其依赖,走代理别名):
  ssh yishan_3090-7897-proxy '~/.conda/envs/glad_quant/bin/pip install <tdmpc2 依赖>'
运行:
  CUDA_VISIBLE_DEVICES=2 python3 qrt/bench/bench_tdmpc2.py
  QRT_SAMPLES=64,256,512 python3 qrt/bench/bench_tdmpc2.py

注意:本 bench 只能在装好 tdmpc2 的机器跑。`build_agent()` 是**接线缝**——TD-MPC2 的
cfg 由 hydra/OmegaConf 构建,依安装布局而异;若自动构建失败,按提示在 build_agent 里
补上你环境的 cfg 构造(state 任务、cfg.compile=False),测量协议部分无需改。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import time
import torch

SAMPLES = [int(x) for x in os.environ.get("QRT_SAMPLES",
                                          "64,128,256,512,1024").split(",")]
DEV = "cuda"


def cuda_time(fn, n_warmup=10, n_repeat=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_repeat):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_repeat


def build_agent():
    """构建真实 TD-MPC2 agent(state 任务,cfg.compile=False)+ 一个 obs 采样器。
    返回 (agent, cfg, make_obs)。这是接线缝:依你 tdmpc2 安装布局调整 cfg 构造。"""
    try:
        # 常见布局:tdmpc2 仓库根在 PYTHONPATH,包内模块直接 import
        from tdmpc2 import TDMPC2                       # noqa: F401
        from common.parser import parse_cfg
        from omegaconf import OmegaConf
    except Exception as ex:
        raise RuntimeError(
            "无法 import tdmpc2/common。请在装好官方 tdmpc2 的机器运行,并把其仓库根\n"
            "加入 PYTHONPATH(export PYTHONPATH=/path/to/tdmpc2/tdmpc2:$PYTHONPATH),\n"
            f"或在 build_agent() 里改成你环境的 cfg 构造。原始错误:{ex}")

    # 默认 config.yaml + state 任务 + 关闭官方编译(由本 patch 加速)
    cfg_path = os.environ.get("TDMPC2_CFG", "config.yaml")
    cfg = parse_cfg(OmegaConf.load(cfg_path))
    cfg.task = os.environ.get("TDMPC2_TASK", "dog-run")   # state 任务
    cfg.compile = False
    cfg.mpc = True
    agent = TDMPC2(cfg)
    agent.model.eval()

    # state obs:一个 [obs_dim] 张量(dict-obs 任务请在此改成 dict)
    obs_dim = cfg.obs_shape[cfg.obs][0] if hasattr(cfg, "obs_shape") else 64

    def make_obs():
        return torch.randn(obs_dim, device=DEV, dtype=torch.float32)
    return agent, cfg, make_obs


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA(远程 3090);本地仅语法检查。"); return
    from qrt.tdmpc2 import accelerate_tdmpc2

    torch.manual_seed(0)
    try:
        agent, cfg, make_obs = build_agent()
    except RuntimeError as ex:
        print(ex); return
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"task={cfg.task}  latent={cfg.latent_dim} mlp={cfg.mlp_dim} "
          f"num_pi_trajs={cfg.num_pi_trajs} horizon={cfg.horizon}")
    print(f"扫描 num_samples: {SAMPLES}(→ rollout batch = num_samples+{cfg.num_pi_trajs})\n")

    obs = make_obs()
    results = {}
    for ns in SAMPLES:
        cfg.num_samples = ns
        print(f"━━━ num_samples={ns}  (batch≈{ns+cfg.num_pi_trajs}) " + "━" * 40)
        # 干净对照:baseline 与 +INT8 都走**同一个 FP16 双网络副本**,只差 rollout 是否 INT8
        # (隔离 fp32→fp16 混淆;ΔINT8 = 纯 INT8 边际)。
        acc0 = accelerate_tdmpc2(agent, int8=False, verbose=False)   # 纯 FP16 副本
        with torch.no_grad():
            for _ in range(3):
                agent.act(obs)
            torch.cuda.synchronize()
            lat_base = cuda_time(lambda: agent.act(obs))
        acc0.detach()

        acc = accelerate_tdmpc2(agent, int8=True, act_mode="dynamic_full", verbose=False)
        with torch.no_grad():
            for _ in range(3):
                agent.act(obs)
            torch.cuda.synchronize()
            lat_int8 = cuda_time(lambda: agent.act(obs))
        n_q, n_skip = len(acc.swapped), len(acc.skipped)
        acc.detach()

        d = lat_int8 - lat_base
        results[ns] = (lat_base, lat_int8, d, n_q, n_skip)
        print(f"  baseline(FP16)     {lat_base:8.3f}ms")
        print(f"  +INT8 rollout      {lat_int8:8.3f}ms   ΔINT8={d:+.3f}ms   "
              f"(量化 {n_q} 层, 跳过 {n_skip})\n")

    print("=" * 78)
    print("agent.act 单步规划延迟;ΔINT8<0 = INT8 rollout linear 在该 batch 净正")
    print(f"{'num_samp':>9} {'baseline':>10} {'+INT8':>10} {'ΔINT8':>10}  {'量化层':>6}")
    print("-" * 78)
    cross = prev = None
    for ns in SAMPLES:
        b, q, d, nq, nsk = results[ns]
        print(f"{ns:>9} {b:>10.3f} {q:>10.3f} {d:>+10.3f}  {nq:>6}")
        if prev is not None and prev[1] * d < 0:
            cross = prev[0] + (ns - prev[0]) * (0 - prev[1]) / (d - prev[1])
        prev = (ns, d)
    print("=" * 78)
    print(f"INT8 rollout 变号 num_samples* ≈ {cross if cross else 'N/A(区间内未变号)'}")
    print("  (由正转负处 = TD-MPC2 规划批量下 INT8 rollout linear 从负项翻为收益的边界;")
    print("   同硬件、移动 num_samples 即翻转 → 验证「INT8 价值由 roofline 位置决定」。)")
    print("  注:量化生效核验用 _check_int8.py 思路 grep _int_mm(精度过好是警报);")
    print("      encoder 与 vmap _Qs 未量化(rollout-only + TODO),故 ΔINT8 只反映"
          "rollout backbone。")


if __name__ == "__main__":
    main()

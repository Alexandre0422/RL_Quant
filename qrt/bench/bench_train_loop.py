#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qrt/bench/bench_train_loop.py — 端到端训练循环基准(P0/P1/P1.5 系统级数据)
══════════════════════════════════════════════════════════════════════════════
协议与 TRT 路线 bench_refit_train.py 严格对齐(可直接对比其历史数据):
  N_ITERS=8(另 N_WARMUP=2)  N_STEPS=24  N_MINIBATCHES=4  MINIBATCH_SZ=1024
  rollout: eval + inference_mode + 24× actor 前向(obs 预生成,不计时)
  update : 4× minibatch dummy loss(actor 路径,randn 现场生成) + Adam

配置(QRT_CONFIGS 环境变量选择,默认 A,C,E,F,G):
  A  FP32 eager(基线,复刻 TRT 协议口径)
  B  FP32 + Runner(max-autotune)          单模型,update 与 A 完全一致
  C  FP16 + Runner(max-autotune)          单模型;⚠ 纯 FP16 训练会 nan,
                                           仅作计时对照,非可用训练形态
  D  FP16+LET-INT8 单模型                  同上
  ── P1.5 双模型形态(FP32 master 训练器 + 独立推理模型 + 每 iter 同步)──
  E  trainer FP32 eager update + FP16 compile 推理 + sync
  F  E 中 update 换 TrainStepRunner(bf16 autocast compile 前向+AOT反向+fused Adam)
  G  F + 推理模型 LET-INT8(sync 内含 requantize,整体 CUDA Graph)

TRT 历史对照(同协议): FP32=175ms/iter,INT8+Async refit 最优=122ms(1.436×)
运行(远程): python3 qrt/bench/bench_train_loop.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B, PA, MAP, DEVICE, build_model, make_obs, cuda_time)

import time
import numpy as np
import torch
import torch._dynamo as dynamo

from qrt import InferenceRunner, TrainStepRunner, swap, calib

N_ITERS, N_WARMUP = 8, 2
N_STEPS = 24
N_MINIBATCHES = 4
MINIBATCH_SZ = B // N_MINIBATCHES
INT8_PATHS = ["actor.0", "actor.2", "actor.4", "actor.6",
              "actor_proprio_embedding", "query_projector"]
CONFIGS = os.environ.get("QRT_CONFIGS", "A,C,E,F,G").split(",")


def mb_obs(dtype):
    """协议:update 的 minibatch obs 现场 randn 生成(计时包含,与 TRT 协议一致)。"""
    return {
        "policy_obs": torch.randn(MINIBATCH_SZ, PA + MAP, device=DEVICE, dtype=dtype),
        "critic_obs": torch.randn(MINIBATCH_SZ, PA + 3 + MAP, device=DEVICE, dtype=dtype),
    }


def ppo_update_step(model, optimizer, dtype):
    """eager dummy update(与 bench_refit_train.ppo_update_step 同构)。"""
    model.train()
    for _ in range(N_MINIBATCHES):
        obs_dict = mb_obs(dtype)
        actor_obs = model.actor_obs_normalizer(model.get_actor_obs(obs_dict))
        model.update_distribution(actor_obs)
        loss = model.distribution.mean.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def run_loop(tag, act_fn, update_fn, obs_dtype, sync_fn=None):
    """N_WARMUP+N_ITERS 轮,返回 (rollout, update, sync) 均值 ms。"""
    obs_pool = [make_obs(obs_dtype) for _ in range(N_STEPS)]
    po_pool = [o["policy_obs"] for o in obs_pool]
    torch.cuda.synchronize()

    rolls, upds, syncs = [], [], []
    for it in range(N_ITERS + N_WARMUP):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for i in range(N_STEPS):
                act_fn(po_pool[i])
        torch.cuda.synchronize()
        t_roll = (time.time() - t0) * 1000

        torch.cuda.synchronize()
        t0 = time.time()
        update_fn()
        torch.cuda.synchronize()
        t_upd = (time.time() - t0) * 1000

        t_sync = 0.0
        if sync_fn is not None:
            torch.cuda.synchronize()
            t0 = time.time()
            sync_fn()
            torch.cuda.synchronize()
            t_sync = (time.time() - t0) * 1000

        if it < N_WARMUP:
            print(f"  [warmup {it+1}] rollout={t_roll:.0f}ms update={t_upd:.0f}ms"
                  + (f" sync={t_sync:.2f}ms" if sync_fn else ""))
            continue
        rolls.append(t_roll); upds.append(t_upd); syncs.append(t_sync)
        print(f"  iter {it-N_WARMUP+1:02d}: rollout={t_roll:6.1f}ms"
              f"  update={t_upd:5.1f}ms"
              + (f"  sync={t_sync:.2f}ms" if sync_fn else "")
              + f"  total={t_roll+t_upd+t_sync:6.1f}ms")

    r, u, s = np.mean(rolls), np.mean(upds), np.mean(syncs)
    print(f"  均值: rollout={r:.1f}ms  update={u:.1f}ms"
          + (f"  sync={s:.2f}ms" if sync_fn else "") + f"  total={r+u+s:.1f}ms")
    return r, u, s


def banner(s):
    print(f"\n{'─'*70}\n  {s}\n{'─'*70}")


def make_dual_setup(int8: bool, patch: bool = False):
    """双模型: FP32 trainer + FP16(/INT8/重写) 推理模型 + runner(已 warmup)。"""
    trainer = build_model(torch.float32)
    infer = build_model(torch.float16)
    infer.load_state_dict(trainer.state_dict())     # 同起点(load 自动 cast)
    if int8:
        from qrt import swap_cnn, calib_cnn_scale
        t0 = time.perf_counter()
        plan = calib.make_plan(infer, INT8_PATHS,
                               obs_fn=lambda: make_obs(torch.float16),
                               use_let=True, let_steps=200, verbose=False)
        s_cnn = calib_cnn_scale(infer, lambda: make_obs(torch.float16))  # swap_cnn 前
        swap.apply(infer, plan)
        swap_cnn(infer, s_cnn)                       # P2: Triton INT8 conv
        print(f"  LET 校准+swap(6 Linear + QuantCNN){time.perf_counter()-t0:.1f}s(一次性)")
    if patch:
        from qrt import patch_encoder
        patch_encoder(infer)                         # P3: encoder 代数重写(等价)
    runner = InferenceRunner(infer, mode="max-autotune")
    runner.sync_weights_from(trainer)     # 标准顺序: sync(graph 捕获)先于编译
    t0 = time.perf_counter()
    runner.warmup(make_obs(torch.float16)["policy_obs"])
    print(f"  rollout warmup(编译) {time.perf_counter()-t0:.1f}s(一次性)")
    return trainer, runner


def check_health(model, tag):
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    print(f"  [{tag}] 参数健全性: " + ("全部有限 ✓" if not bad else f"❌ 非有限: {bad}"))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
    print(f"协议: N_ITERS={N_ITERS}(+{N_WARMUP} warmup) N_STEPS={N_STEPS} "
          f"MINIBATCH={N_MINIBATCHES}x{MINIBATCH_SZ}  CONFIGS={CONFIGS}")
    summary = []
    base_total = None

    # ── A. FP32 eager 基线 ───────────────────────────────────────────────────
    if "A" in CONFIGS:
        banner("A. FP32 eager(基线,对照 TRT 协议 175ms)")
        m = build_model(torch.float32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        r, u, s = run_loop("A", lambda po: m.act_inference({"policy_obs": po}),
                           lambda: ppo_update_step(m, opt, torch.float32),
                           torch.float32)
        summary.append(("A FP32 eager", r, u, s))
        base_total = r + u + s
        del m, opt
        dynamo.reset(); torch.cuda.empty_cache()

    # ── B. FP32 + Runner(单模型)─────────────────────────────────────────────
    if "B" in CONFIGS:
        banner("B. FP32 + Runner(单模型,update 与 A 一致)")
        m = build_model(torch.float32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        runner = InferenceRunner(m, mode="max-autotune")
        runner.warmup(make_obs(torch.float32)["policy_obs"])
        r, u, s = run_loop("B", runner.act,
                           lambda: ppo_update_step(m, opt, torch.float32),
                           torch.float32)
        summary.append(("B FP32 compile", r, u, s))
        del m, opt, runner
        dynamo.reset(); torch.cuda.empty_cache()

    # ── C. FP16 单模型(⚠ 非可用训练形态,仅计时对照)────────────────────────
    if "C" in CONFIGS:
        banner("C. FP16 单模型(⚠ 纯FP16训练会nan,仅计时对照)")
        m = build_model(torch.float16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        runner = InferenceRunner(m, mode="max-autotune")
        runner.warmup(make_obs(torch.float16)["policy_obs"])
        r, u, s = run_loop("C", runner.act,
                           lambda: ppo_update_step(m, opt, torch.float16),
                           torch.float16)
        summary.append(("C FP16 单模型(对照)", r, u, s))
        del m, opt, runner
        dynamo.reset(); torch.cuda.empty_cache()

    # ── E. 双模型: FP32 eager update + FP16 compile rollout + sync ──────────
    if "E" in CONFIGS:
        banner("E. 双模型: FP32 eager update + FP16 compile rollout + sync")
        trainer, runner = make_dual_setup(int8=False)
        opt = torch.optim.Adam(trainer.parameters(), lr=1e-4)
        runner.sync_weights_from(trainer)        # 触发配对+graph 捕获
        r, u, s = run_loop("E", runner.act,
                           lambda: ppo_update_step(trainer, opt, torch.float32),
                           torch.float16,
                           sync_fn=lambda: runner.sync_weights_from(trainer))
        summary.append(("E 双模型 eager-upd", r, u, s))
        check_health(trainer, "E")
        del trainer, runner, opt
        dynamo.reset(); torch.cuda.empty_cache()

    # ── F. 双模型: TrainStepRunner(bf16+compile) + sync ─────────────────────
    if "F" in CONFIGS:
        banner("F. 双模型: compile(bf16 autocast)update + FP16 compile rollout")
        trainer, runner = make_dual_setup(int8=False)
        tsr = TrainStepRunner(trainer, lr=1e-4)
        t0 = time.perf_counter()
        trainer.train()
        tsr.dummy_step(mb_obs(torch.float32)["policy_obs"])   # 触发 train 图编译
        trainer.eval()
        torch.cuda.synchronize()
        print(f"  train 前向+反向编译 {time.perf_counter()-t0:.1f}s(一次性)")
        runner.sync_weights_from(trainer)

        def upd_f():
            trainer.train()
            for _ in range(N_MINIBATCHES):
                obs_dict = mb_obs(torch.float32)               # 协议开销保持一致
                tsr.dummy_step(obs_dict["policy_obs"])
            trainer.eval()

        r, u, s = run_loop("F", runner.act, upd_f, torch.float16,
                           sync_fn=lambda: runner.sync_weights_from(trainer))
        summary.append(("F +compile update", r, u, s))
        check_health(trainer, "F")
        del trainer, runner, tsr
        dynamo.reset(); torch.cuda.empty_cache()

    # ── G. P3 完全体: LET-INT8 + QuantCNN + encoder 重写(双侧 patch)─────────
    if "G" in CONFIGS:
        banner("G. 完全体: compile update(patch) + LET+QuantCNN+patch rollout")
        trainer, runner = make_dual_setup(int8=True, patch=True)
        from qrt import patch_encoder
        patch_encoder(trainer)        # 训练器也 patch(train 分支等价已验证)
        tsr = TrainStepRunner(trainer, lr=1e-4)
        t0 = time.perf_counter()
        trainer.train()
        tsr.dummy_step(mb_obs(torch.float32)["policy_obs"])
        trainer.eval()
        torch.cuda.synchronize()
        print(f"  train 前向+反向编译 {time.perf_counter()-t0:.1f}s(一次性)")
        runner.sync_weights_from(trainer)

        def upd_g():
            trainer.train()
            for _ in range(N_MINIBATCHES):
                obs_dict = mb_obs(torch.float32)
                tsr.dummy_step(obs_dict["policy_obs"])
            trainer.eval()

        r, u, s = run_loop("G", runner.act, upd_g, torch.float16,
                           sync_fn=lambda: runner.sync_weights_from(trainer))
        summary.append(("G +LET-INT8 rollout", r, u, s))
        check_health(trainer, "G")
        del trainer, runner, tsr
        dynamo.reset(); torch.cuda.empty_cache()

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"{'配置':<24} {'rollout':>9} {'update':>8} {'sync':>7} "
          f"{'total':>9} {'×FP32':>7} {'vs TRT122':>9}")
    print("-" * 88)
    for name, r, u, s in summary:
        tot = r + u + s
        ratio = f"{base_total/tot:>6.2f}x" if base_total else "   n/a"
        print(f"{name:<24} {r:>8.1f}ms {u:>7.1f}ms {s:>6.2f}ms "
              f"{tot:>8.1f}ms {ratio} {122.0/tot:>8.2f}x")
    print("-" * 88)
    print("TRT 历史(同协议): FP32=175ms  INT8+Async refit 最优=122ms(1.436×,上限 1.814×)")
    print("=" * 88)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""qrt/bench/_common.py — bench 公共工具(mock/路径/模型构建/计时)"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # 远程 GPU0 常被占用

import sys
import types


def _mock_missing():
    """真包存在则用真包(bench_real_loop 需要真 tensordict),缺失才 mock。"""
    try:
        import git  # noqa: F401
    except ImportError:
        _git = types.ModuleType("git")

        class _Repo:
            def __init__(self, *a, **kw):
                pass

        _git.Repo = _Repo
        _git.InvalidGitRepositoryError = Exception
        sys.modules.setdefault("git", _git)
    try:
        import tensordict  # noqa: F401
    except ImportError:
        _td = types.ModuleType("tensordict")

        class _TensorDict(dict):
            pass

        _td.TensorDict = _TensorDict
        sys.modules.setdefault("tensordict", _td)


_mock_missing()

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))      # .../GLAD_Quant/qrt/bench
WORKDIR = os.path.dirname(os.path.dirname(BENCH_DIR))       # .../GLAD_Quant
sys.path.insert(0, WORKDIR)
sys.path.insert(0, os.path.join(WORKDIR, "rsl_rl"))

import torch                                   # noqa: E402
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder   # noqa: E402

B, PA, MAP, NA = 4096, 96, 33 * 21 * 3, 29
DEVICE = torch.device("cuda")


def build_model(dtype=torch.float16, variant="glad"):
    """variant:
    "glad" — 完整 GLAD(GPS+TopK,论文主配置,速度主线用)
    "ame"  — AME 基线(use_global_context=False, topk=None),
             与 AME_Locomotion-main/pretrained/ame1.pt 真实 checkpoint 结构一致
    """
    use_gc, topk = (True, 32) if variant == "glad" else (False, None)
    dummy = {"policy_obs": torch.zeros(1, PA + MAP),
             "critic_obs": torch.zeros(1, PA + 3 + MAP)}
    grps = {"policy": ["policy_obs"], "critic": ["critic_obs"]}
    m = ActorCriticEncoder(
        obs=dummy, obs_groups=grps, num_actions=NA,
        map_scan_dim=(33, 21, 3), mha_dim=64, num_heads=16,
        cnn_downsample=True, use_global_context=use_gc, topk=topk,
    ).to(DEVICE)
    if dtype == torch.float16:
        m = m.half()
    return m.eval()


def load_ame_ckpt(model, path="ame1.pt"):
    """加载 AME 真实 checkpoint(iter 21000 收敛权重)到 variant='ame' 模型。"""
    import os
    for cand in (path, os.path.join(WORKDIR, path), os.path.join(WORKDIR, "qrt", path)):
        if os.path.exists(cand):
            ckpt = torch.load(cand, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"]
            # ActorCriticEncoder.load_state_dict 被重写为返回 bool,自行校验键集合
            model_keys = set(model.state_dict().keys())
            unexpected = [k for k in sd if k not in model_keys]
            assert not unexpected, f"unexpected keys: {unexpected}"
            model.load_state_dict(sd, strict=False)
            return model
    raise FileNotFoundError(f"ame checkpoint not found: {path}")


def make_obs(dtype=torch.float16, batch=B):
    """纯随机 obs(快速基准用;分布与真实差异大,精度实验勿用)。"""
    return {
        "policy_obs": torch.randn(batch, PA + MAP, device=DEVICE, dtype=dtype),
        "critic_obs": torch.randn(batch, PA + 3 + MAP, device=DEVICE, dtype=dtype),
    }


def make_terrain_obs(dtype=torch.float16, batch=B, gen=None):
    """结构化地形 obs(精度实验用,接近真实 rollout 分布形态):
    - map 部分 [33,21,3]: xy = 机器人系固定扫描网格(近常量,决定 conv1 前两通道分布),
      z = 平滑随机高度场(低频上采样 + 台阶离散化 + 传感噪声,~±0.4m)
    - proprio 96 维: 站立姿态附近的有界随机量
    注: 仍是合成数据;接真实 Isaac Lab rollout obs 时,直接替换本函数即可
    (calib.make_plan(obs_fn=...) / bench 均通过 obs_fn 注入)。
    """
    import torch.nn.functional as F
    g = {"generator": gen} if gen is not None else {}
    W_, L_ = 21, 33                                     # 模型 reshape(-1, W, L, 3)
    ys = torch.linspace(-1.0, 1.0, W_, device=DEVICE)
    xs = torch.linspace(-1.6, 1.6, L_, device=DEVICE)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")      # [W, L]
    xy = torch.stack([xx, yy], dim=-1).expand(batch, W_, L_, 2)

    coarse = torch.randn(batch, 1, 5, 4, device=DEVICE, **g) * 0.25
    z = F.interpolate(coarse, size=(W_, L_), mode="bilinear", align_corners=False)
    z = torch.round(z / 0.08) * 0.08                    # 8cm 台阶离散化
    z = z + torch.randn(batch, 1, W_, L_, device=DEVICE, **g) * 0.02   # 传感噪声
    z = z.squeeze(1).unsqueeze(-1)                      # [B, W, L, 1]

    map_scan = torch.cat([xy, z], dim=-1).reshape(batch, -1)           # [B, MAP]
    proprio = torch.randn(batch, PA, device=DEVICE, **g) * 0.8
    po = torch.cat([proprio, map_scan], dim=-1).to(dtype)
    proprio_c = torch.randn(batch, PA + 3, device=DEVICE, **g) * 0.8
    co = torch.cat([proprio_c, map_scan], dim=-1).to(dtype)
    return {"policy_obs": po, "critic_obs": co}


def cuda_time(fn, n_warmup=15, n_repeat=100):
    """CUDA Event 计时,返回均值 ms。"""
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


def cossim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()

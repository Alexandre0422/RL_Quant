# -*- coding: utf-8 -*-
"""
tap.py — P6.B: 编译图内激活旁路(零额外前向的校准数据采集)
══════════════════════════════════════════════════════════════════════════════
问题: LET 伴随校准需要各量化层的输入激活,但它们是编译训练前向(bf16 autocast)
      内部的中间张量 —— forward hook 会触发 dynamo 重编译/降级;图边界(思路 A)
      只暴露 2/6 层(其余在 actor/encode 图内部,见数据流分析)。

方案 B: 注册一个 custom op `qrt::tap`,前向恒等(返回输入),副作用是把输入
      (下采样后)copy_ 进预分配的静态 buffer。custom op 是 dynamo 认可的算子
      (同 P2 的 conv kernel),可插在编译图内部任意位置、不 break、不重编译;
      副作用写固定 buffer → CUDA Graph 安全。

数值/语义保证:
  - 前向 = 输入原样返回,bit-exact,不改训练计算;
  - 旁路写的是 detach 的下采样副本,不进反向图(register_fake 无副作用);
  - bf16 激活做 LET 校准端到端无害(_test_bf16_calib.py 实测 ΔCosSim 7.7e-6);
  - 行采样在算子内完成(固定 sample_rows),buffer 体积与 batch 无关。

用法(install_taps 装在 FP32 trainer 上,cocalib source="tap" 读 buffer):
    slots = install_taps(trainer, paths)            # 包 TapLinear + 分配 buffer
    ... 编译的 update 前向(经过 TapLinear 时旁路写 buffer)...
    x = TapStore.read(slot)                          # 取回该层最近激活 [≤rows, K]

实测验证(_test_tap.py + bench_cocalib.py E 组):
  - 前向 bit-exact(tap 恒等);
  - tap 采集(E)与 eager 独立前向(C)的 50-iter 漂移曲线吻合(末值差 1.4e-5),
    证明数值等效且省一次前向;
  - **必须 register_autograd**(反向恒等直通)—— 否则编译训练前向 backward 时
    报 "no autograd formula"(spike ④ 暴露);
  - **cocalib 开启时训练前向须用 no-cudagraphs**: cudagraph replay 不重放
    custom-op 的 Python 副作用,buffer 采集失真(spike ③);
  - TapLinear 转移 weight/bias Parameter(不作子模块)→ named_parameters 路径
    保持 actor.0.weight,sync_weights_from 配对不破。
"""
from __future__ import annotations

import torch


class TapStore:
    """进程级单例: 持有各 slot 的静态采集 buffer 与有效行数。"""
    _bufs: list[torch.Tensor] | None = None     # [n_layers] 各 [sample_rows, max_K]
    _valid: list[int] | None = None             # 各 slot 实际写入行数(≤rows)
    _kdim: list[int] | None = None              # 各 slot 的真实 K
    rows: int = 2048
    enabled: bool = False
    _device = None

    @classmethod
    def enable(cls, n_layers: int, k_dims: list[int], sample_rows: int = 2048,
               device="cuda"):
        cls.rows = sample_rows
        cls._device = torch.device(device)
        cls._bufs = [torch.zeros(sample_rows, k, device=device) for k in k_dims]
        cls._valid = [0] * n_layers
        cls._kdim = list(k_dims)
        cls.enabled = True

    @classmethod
    def disable(cls):
        cls.enabled = False
        cls._bufs = cls._valid = cls._kdim = None

    @classmethod
    def write(cls, slot: int, x2d: torch.Tensor):
        """x2d: [M, K] fp32。下采样到 ≤rows 行写入 slot buffer。"""
        if not cls.enabled or cls._bufs is None:
            return
        m = x2d.shape[0]
        buf = cls._bufs[slot]
        if m >= cls.rows:
            idx = torch.randint(0, m, (cls.rows,), device=x2d.device)
            buf.copy_(x2d[idx])
            cls._valid[slot] = cls.rows
        else:
            buf[:m].copy_(x2d)
            cls._valid[slot] = m

    @classmethod
    def read(cls, slot: int) -> torch.Tensor | None:
        if not cls.enabled or cls._bufs is None or cls._valid[slot] == 0:
            return None
        return cls._bufs[slot][: cls._valid[slot]].clone()


# ── custom op: 前向恒等 + 旁路写 buffer ──────────────────────────────────────
@torch.library.custom_op("qrt::tap", mutates_args=(), device_types="cuda")
def tap(x: torch.Tensor, slot: int) -> torch.Tensor:
    # 副作用: 写静态 buffer(detach,fp32,行采样);返回输入的别名(恒等)
    if TapStore.enabled:
        TapStore.write(int(slot), x.detach().float().reshape(-1, x.shape[-1]))
    return x.clone()        # clone 而非原样返回: 避免别名干扰 inductor 的 buffer 复用


@tap.register_fake
def _(x, slot):
    return torch.empty_like(x)


# 反向恒等直通: tap 不改前向数值,故梯度原样透传(slot 无梯度)。
# 没有这一条,编译训练前向(需 backward)会报 "no autograd formula"(spike ④)。
def _tap_setup_ctx(ctx, inputs, output):
    pass


def _tap_backward(ctx, grad):
    return grad, None


tap.register_autograd(_tap_backward, setup_context=_tap_setup_ctx)


def tap_(x: torch.Tensor, slot: int) -> torch.Tensor:
    """便捷封装: 仅在采集开启时入图,否则零开销直通。"""
    if TapStore.enabled:
        return torch.ops.qrt.tap(x, slot)
    return x


# ── 训练模型侧的零侵入采集包装 ────────────────────────────────────────────────
class TapLinear(torch.nn.Module):
    """包装 nn.Linear,grad 态前向时旁路采集输入激活(数值恒等,不量化)。
    装在 FP32 trainer 的目标层上 —— 训练前向(编译)经过时顺便采集,零额外前向。

    转移同一 weight/bias Parameter 对象(不作子模块)→ named_parameters 路径
    保持 `actor.0.weight`,与 infer 侧 QuantLinear 一致,sync 配对/optimizer 不破。"""

    def __init__(self, linear: torch.nn.Linear, slot: int):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.slot = slot
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        if torch.is_grad_enabled():
            x = tap_(x, self.slot)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def install_taps(model, paths: list[str], sample_rows: int = 2048,
                 device="cuda") -> dict[str, int]:
    """把 model 的 paths 各层包成 TapLinear 并 enable TapStore。
    返回 {path: slot}。注意: 须在编译训练前向**之前**调用(tap 才能进图)。"""
    import torch.nn as nn

    k_dims, slots = [], {}
    for i, p in enumerate(paths):
        lin = model.get_submodule(p)
        assert isinstance(lin, nn.Linear), f"{p} 不是 nn.Linear"
        k_dims.append(lin.in_features)
        slots[p] = i
        # 替换为 TapLinear(参数对象不动,optimizer/sync 路径不变)
        if "." in p:
            par, name = p.rsplit(".", 1)
            parent = model.get_submodule(par)
        else:
            parent, name = model, p
        tl = TapLinear(lin, i)
        if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(name)] = tl
        else:
            setattr(parent, name, tl)
    TapStore.enable(len(paths), k_dims, sample_rows=sample_rows, device=device)
    return slots

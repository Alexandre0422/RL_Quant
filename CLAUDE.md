# CLAUDE.md — GLAD 量化项目工作指引

> 面向人的完整项目说明见 **[README.md](README.md)**(架构/实现/测试结果/用法)。
> 完整方案与逐阶段实验记录见 **[PLAN_compile_quant.md](PLAN_compile_quant.md)**。
> 本文件只放 Claude 在此代码库工作时需要的约定与高频信息。

---

## 一句话现状

主线是 **`qrt/` 包**(torch.compile + CUDA Graph + 选择性 INT8 + LET + 图重写 + 双模型训练)。
旧 TensorRT 路线已归档到 **`legacy_trt/`**(不再维护,保留作对比基线)。
当前进度:P0~P6 全部完成,待用户上 Isaac Lab 训练机跑真实 reward A/B(清单见 README §10)。

核心数字:单步推理 1.054ms(5.90× vs FP32);训练系统 990ms/iter(2.85×,B=2048);梯度保真 cos=0.999998。

---

## 环境

- **本地**(开发):RTX 5060 Laptop(SM 12.0),Windows,conda `pytorch_gpu`,PyTorch 2.10+cu130。
  **SM 12.0 无传统 INT8 IMMA,所有 INT8/性能实验必须在远程跑。** 本地仅编辑代码 + 语法检查。
  本地 Python:`C:\Users\LiZixuan\.conda\envs\pytorch_gpu\python.exe`。
- **远程**(实验):RTX 3090 ×2(SM 8.6),Ubuntu,PyTorch 2.10+cu126,Triton 3.6,tensordict 0.13。
  SSH 见 `.env`,`tools/ssh_helper.py` 封装(`run` / `run_in_project` / `put` / `put_text`),项目路径 `/tmp/glad_quant_test/`。
- 所有脚本从 `GLAD_Quant/` 目录运行;`qrt/bench/_common.py` 已处理 mock(git/tensordict)与 path。

## 远程运行惯例

```python
# 从 GLAD_Quant/ 目录,本地 python:
import sys; sys.path.insert(0, 'tools')
from ssh_helper import put, run_in_project
put('qrt/xxx.py', '/tmp/glad_quant_test/qrt/xxx.py')
out, err, code = run_in_project('python3 qrt/bench/bench_xxx.py', timeout=560)
```
- PowerShell 内联 Python 易踩转义坑(f-string 的 `:`、`\`)→ 用 here-string `@'...'@` 包脚本。
- 远程 stdout 用 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`(GBK 默认会炸 emoji/✓)。
- 长任务(编 update 图 ~300s)用后台 + 轮询 ALLDONE,勿干等;跑前 `nvidia-smi` 确认 GPU 空闲(共享卡,他人负载会污染基准)。

---

## 工作区约定

- **新代码进 `qrt/`**;LET 工具在 `quant/`(两路线共用);模型在 `rsl_rl/`(勿改 `actor_critic_encoder.py` 结构)。
- **不碰 `legacy_trt/`**(归档,只读);`engines/glad_actor_base.onnx` 是唯一起点 ONNX,勿删。
- 临时调试脚本(`_*.py`、`smoke_*`)用完即删,有价值的逻辑并进 `qrt/` 对应模块;根目录不留新 `.py`。
- `qrt/bench/` 下 `bench_*` 是正式套件、`_test_*` 是对拍单测,都保留可复跑。

## 改了代码后

- 同步到远程(`put`)再跑;改 `qrt/` 或 `quant/` 或 `rsl_rl/` 后远程需重传。
- 量化是否真生效:`qrt/bench/_check_int8.py` grep `_int_mm` kernel(**精度指标"过好"是警报,要 kernel 级证据**)。
- 有重大结论/坑:更新 README + PLAN + 记忆三处。

---

## 高频坑(详见 README §7)

1. 必须 `mode="max-autotune"`(默认/reduce-overhead/freezing 各有崩法)。
2. INT8 反量化 int32→fp32→fp16(直接 .half() 溢出 inf;compile 版会掩盖)。
3. swap 后须继承 train/eval 状态(否则 INT8 静默失效,CosSim=1 假象)。
4. 量化分支按 `torch.is_grad_enabled()` 不是 `self.training`。
5. CUDA RNG 哨兵 graph(`runtime._ensure_graph_rng_state`,Runner 构造自动调)。
6. 等价性对拍必须 eval 态(train 态 Gumbel 噪声 = 0.984 假象)。
7. Triton:`tl.arange` 2 的幂;int8 dot 用 `acc += tl.dot(...,out_dtype=tl.int32)`。
8. cocalib+tap 训练前向须 no-cudagraphs;但实测 tap 端到端反而比 eager 慢(默认 eager)。

---

## 关键事实(避免重走弯路)

- INT8 **linear** 是速度负项(+0.1ms,小 GEMM 访存受限);INT8 **conv** 才有效(输入流量大)。INT8 按算子 roofline 逐个部署,不是一刀切。
- 加速大头是图融合(−1.9ms)与 INT8 conv(−0.36ms),不是 INT8 linear。生产部署推荐 `int8_linear=False`。
- INT8 linear 的价值在精度/科研维度(LET 算法对照平台)与未来 B=1 板端(roofline 翻转)。
- 真实 PPO 协议下 update 占 ~95%,bf16 编译是系统加速关键(rollout 的 5× 被稀释)。

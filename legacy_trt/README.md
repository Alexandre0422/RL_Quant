# legacy_trt/ — TensorRT 路线归档(已被 qrt 主线取代)

> 本目录是 GLAD 量化的**第一代实现(TensorRT INT8 + Refit)**,自 2026-06-11 起
> 不再维护。新主线见上级目录 `../qrt/` 与 `../README.md`。
> 保留目的:作为性能对比基线的**可复现来源**(README 中"qrt vs TRT"对比数字源自此)。

## 这条路线做了什么

将 GLAD actor 导出 ONNX → onnx_graphsurgeon 插入 LET 变换(Mul/Add)与 QDQ 节点 →
编译 TensorRT INT8 引擎 → 训练中用 `trt.Refitter` 更新权重。

## 为什么被取代(三个结构性问题)

1. **编译成本**:每个量化配置重建引擎 ~5 分钟;2^13 子集穷举极慢;timing cache 非确定。
2. **权重同步**:RL 每 iter 更新权重必须 refit,refit 太慢 → 被迫 K=4 摊销 + 异步双缓冲
   → rollout 用滞后 1–4 iter 的权重,**破坏 on-policy 语义**;即便如此系统加速仅 1.436×。
3. **基线错位**:TRT FP16(6.298ms)比 PyTorch FP16 eager(3.66ms)慢 72%——TRT 把
   GLAD 的 TopK/Gather/MHA 分解成大量小 kernel 时调度低效;TRT INT8 的"加速"是相对
   这个劣化基线的。

## 关键历史数据(qrt 的对比基线,RTX 3090 SM8.6)

| 指标 | 数值 |
|---|---|
| TRT FP16 单步 | 6.298 ms |
| TRT INT8 最优单步(node_linear_10,完整 LET) | 3.274 ms |
| 训练系统(24步 rollout + update,Async refit 最优) | 122 ms/iter(1.436× vs FP32,理论上限 1.814×) |
| Phase1 单层消融 | 12/13 层正贡献,node_MatMul_50(N=1)负贡献 |

(对比:qrt 完全体单步 1.054ms、训练系统 990ms/iter = 2.85× vs FP32,见 `../README.md` §测试结果。)

## 文件

| 文件 | 作用 |
|---|---|
| `bench_let_int8_full_remote.py` | TRT 主流程 Linux 版(LET 校准 + GS + Phase1/2 搜索) |
| `bench_refit_train.py` | TRT Refit 训练系统(K-refit + 异步双缓冲;qrt bench_train_loop 对齐其协议) |
| `bench_let_block_recon.py` | 本地 TRT 参考实现(含 MHA in_proj 折叠修复) |
| `bench_precision_ablation.py` | FP32/FP16/INT8 精度消融 |
| `run_bench_report.py` | 已有引擎延迟+精度报告 |
| `engines/*.trt /.onnx` | TRT FP16 / INT8 引擎产物(SM8.6) |
| `tools/*` | TRT 穷举进度 / TRT 环境检查 / ONNX 结构检查 / 一键运行 |

## 如何复现(若需)

依赖 TensorRT 10.16 + onnx_graphsurgeon 0.6.1 + onnxruntime,远程 RTX 3090。
`engines/glad_actor_base.onnx`(唯一起点 ONNX)在上级 `../engines/` 保留,两条路线共用。
LET 校准器 `../quant/let.py` 两条路线共用。

# GLAD 非 TRT 量化路径方案（2026-06-10 调研定稿）

**结论先行：** 放弃 TRT 图编译路线,主线切换为
**`torch.compile(max-autotune)` 图融合 + CUDA Graph + 选择性 INT8（`torch._int_mm`）+ LET 校准注入**。
FP16 编译基线实测 **1.723ms（3.60× vs FP32 eager）**,已比 TRT INT8 历史最优（3.274ms）快 1.9×,
权重热更新零成本（替代 refit）,零新增依赖,LET 科研线完整保留。

---

## 一、调研实测数据（RTX 3090 SM8.6, B=4096, torch 2.10.0+cu126, 2026-06-10）

### 1.1 单步延迟全景

| 方案 | 单步 | ×FP32 | CosSim | 构建成本 | 权重更新成本 |
|---|---|---|---|---|---|
| PyTorch FP32 eager | 6.203ms | 1.00× | 参照 | 0 | 0 |
| PyTorch FP16 eager | 3.658ms | 1.70× | 1.000000 | 0 | 0 |
| TRT FP16（历史） | 6.298ms | ≈0.98× | 0.9999 | ~5min | — |
| TRT INT8 最优（历史） | 3.274ms | ≈1.89× | 0.9999 | ~5min | refit,系统 1.436× |
| **FP16 compile(max-autotune)** | **1.723ms** | **3.60×** | **1.000000** | 35s 冷 / 8.4s 缓存 | **0（in-place 即生效）** |
| compile(no-cudagraphs)+手动 Graph | 1.640ms | 3.78× | 同上 | 同上 | 0 |
| INT8-MLP(naive) + compile | 1.732ms | 3.58× | 0.999813 | 同上 | re-quantize <0.1ms |

### 1.2 四个被实验推翻/确立的关键事实

1. **TRT 路线的基线错位**：TRT FP16（6.3ms）比 PyTorch FP16 eager（3.66ms）慢 72%,
   TRT INT8 的"1.924×"是相对这个劣化基线的。与 compile 路线相比 TRT 全面落后。
2. **瓶颈是图结构,不是算力**：eager 3.66ms 分解（profiler 实测）：
   `cat` 物化 0.71ms（scorer 输入 [B,187,128]）+ N=1 gemv 0.80ms（GPS/scorer 退化为巨型矩阵向量乘）
   + 碎 elementwise（add 0.45 / BN 0.39 / ReLU 0.30）+ conv 0.61 + addmm 0.59 + bmm 0.32。
   inductor 全模型融合（dynamo 0 graph break）直接消解,无需手工重写。
3. **`torch._int_mm` raw GEMM 在本模型形状下无算力收益**（0.5–1.2×,访存受限）；
   im2col INT8 路线死刑（仅 quantize 就 2.08ms）。INT8 的价值=访存减半,
   只在融合语境下有意义,**主靶是 conv（compile 后仍占 ~0.8ms / 46%）**。
4. **CUDA Graph 权重热更新成立**：in-place `mul_/copy_` 后 graph replay 立即生效
   （diff 4.96e-04 = fp16 舍入水平）,compile 路径更新后首调 0.6ms 无重编译。
   **TRT refit 的全部复杂度（双缓冲/异步/K-step 摊销）被一个 `copy_` 取代。**

### 1.3 历史探索结论修正

- ~~"非 TRT 路径无法匹配 TRT,CNN 瓶颈无解"~~（2026-06-09 结论,基于 torchao/eager/单子模块 compile）
  → **作废**。当时 compile 只编译了 actor 子模块（占时 4%）；全模型编译 2.12× vs eager。
- torchao 版本死锁依旧成立,但不再相关：我们直接用其底层原语 `torch._int_mm`（aten 自带）。
- SageAttention 不适用依旧成立（head_dim=4）；MHA 优化改走 inductor 融合 + P3 手写 kernel。

### 1.4 证据脚本（no_trt/,均可在远程直接复跑）

| 脚本 | 内容 |
|---|---|
| `bench_feasibility.py` | `_int_mm` 全形状扫描 + 手动 CUDA Graph 冒烟 + 热更新验证 |
| `bench_profile_compile.py` | eager kernel 级分解 + `dynamo.explain`（0 break）+ compile 各模式 |
| `bench_compile_stack.py` | FP32/FP16/compile/graph/INT8 组合矩阵 |
| `bench_hotswap.py` | 热更新传导验证 + QuantLinear 修复版（int32→fp32 dequant）+ compile 后分解 |

---

## 二、路径决策与排除理由

**主线**：`torch.compile`（inductor 图融合）+ CUDA Graph（内置 trees 或手动）
+ 选择性 INT8（QuantLinear/`_int_mm`）+ LET 校准注入。

| 排除项 | 理由 |
|---|---|
| TRT（现状） | 编译 5min/配置；refit 稀释系统加速（1.436×,上限 1.814×）；timing_cache 非确定性；被 compile 1.9× 超越 |
| torchao | torch 2.10 处于 0.15/0.17 版本死锁；其底层 `_int_mm` 原语 aten 自带,直接用 |
| 纯手写 Triton 全家桶 | inductor 已自动拿到 ~80% 融合收益；手写保留给定点突破（P2 conv,P3 attention） |
| ORT INT8 | CUDA EP 融合弱,引入第二运行时 |

**依赖面**：torch 2.10.0（现有）+ Triton（torch 自带）。**零新增依赖,无版本冲突。**

---

## 三、架构设计（科研可扩展性）

```
GLAD_Quant/qrt/                     # Quantized RunTime（新主线包）
├── qlinear.py        # QuantLinear: W8A8, _int_mm 后端, LET(α,β) 前置
│                     #   y = DQ( int_mm( Q(α⊙x+β; s_z), W8' ), s_w·s_z ) + b'
│                     #   from_float(linear, let=None, s_x=...) 一行替换
│                     #   requantize_(src_linear)  ← PPO update 后热刷新(<0.1ms)
├── qconv.py          # P2: Triton implicit-GEMM INT8 conv (NHWC, fused BN+ReLU+requant)
├── swap.py           # 模型手术: 按层名替换/还原, 任意子集消融
├── runtime.py        # InferenceRunner(model, mode='max-autotune'|'manual-graph')
│                     #   .step(obs)→(mean,value)  .refresh_weights()
├── calib/pipeline.py # 激活采集(hook/真实rollout) → LETQuantizer 优化 → 注入
│                     #   quant/let.py 的 LETQuantizer 零修改复用
└── bench/            # bench_rollout.py / bench_train_loop.py(对齐 bench_refit_train 协议)
```

**LET 接入（与 TRT 路线数学逐项等价）**：
ONNX 流程的 `Mul(α)→Add(β)→Q(s_z)→DQ→GEMM(W'=W/α, b'=b−W'β)` 1:1 映射为
QuantLinear.forward 内的 `round((α*x+β)*inv_s_z)` → `_int_mm` → dequant,
inductor 把 LET 变换+quantize 融为单 kernel——**LET 从"ONNX 图插桩"升级为"模块内原生算子"**。
`LETQuantizer.fold_into_linear` / 校准代码直接复用。

**科研扩展点**：
1. **保精度算法族**：LET / SmoothQuant / AWQ / OmniQuant-LWC 全是 per-channel scale+shift 范式
   → 统一 `from_float(linear, transform=(α,β))` 接口,换算法不动 runtime。
2. **在线 QAT**：训练前向插 `LETQuantizer.forward`（fake-quant,已有）,rollout 用 QuantLinear（真 INT8）,
   共享 α/β → `L_total = L_PPO + λ·L_LET` 直接可做（TRT 路线做不到训练/推理共享参数对象）。
3. **kernel 后端可换**：`_int_mm` 一行换自写 Triton GEMM（per-token 动态量化 / INT4 packed / FP8 模拟）。
4. **消融自动化**：每配置重编译 ~8s（缓存命中）vs TRT 每配置分钟级重建——
   Phase1/2 的单层消融与子集搜索方法学可整体平移且提速一个量级。

---

## 四、分阶段实施

### P0 ✅ 已完成（2026-06-10,qrt/ 包落地 + 实测）

**qrt/ 包**（与 TRT 主线并行,零修改原文件）：`runtime.InferenceRunner`（max-autotune /
manual-graph / eager 三形态,actor+critic 双编译接口）、`qlinear.QuantLinear`、
`swap`、`calib`、`bench/`（bench_rollout / bench_train_loop / _check_int8）。

**单步实测（bench_rollout,RTX 3090,B=4096,同一 state_dict）：**

| 配置 | 单步 | ×FP32 | CosSim(vs FP32) |
|---|---|---|---|
| A FP32 eager | 6.256ms | 1.00× | — |
| B FP32 compile | 3.797ms | 1.65× | 1.000000 |
| C FP16 eager | 3.663ms | 1.71× | 1.000000 |
| **D FP16 compile（P0 主形态）** | **1.698ms** | **3.68×** | 1.000000 |
| E FP16 manual-graph | 1.694ms | 3.69× | 1.000000 |
| F naive-INT8(6层) compile | 1.758ms | 3.56× | 0.999724 |
| **G LET-INT8(6层) compile（P1 形态）** | 1.780ms | 3.51× | **0.999796** |

**训练循环实测（bench_train_loop,协议=bench_refit_train：24 步 rollout + 4×1024 update）：**

| 配置 | rollout | update | refresh | total | ×FP32 | vs TRT 最优 122ms |
|---|---|---|---|---|---|---|
| A FP32 eager | 148.5ms | 39.8ms | — | 188.3ms | 1.00× | 0.65× |
| B FP32 compile | 91.0ms | 38.2ms | — | 129.2ms | 1.46× | 0.94× |
| **C FP16 compile** | **40.2ms** | 48.6ms | — | **88.8ms** | **2.12×** | **1.37×** |
| **D FP16+LET-INT8** | 40.4ms | 44.2ms | 2.72ms | **87.3ms** | **2.16×** | **1.40×** |

（A 的 rollout 148.5ms 与 TRT 路线历史基线 149ms 一致,协议对齐可信。）

**结论：** 系统 2.12–2.16×,超 TRT async 最优（1.436×）与其理论上限（1.814×）。
LET 精度 > naive（0.999796 vs 0.999724）——LET 保精度价值在 qrt 内首次验证。
inductor 把 `_int_mm` lower 为 Triton INT8 GEMM 模板并自动融合
quantize(round/clamp)+dequant+ELU 进单 kernel（`triton_tem_fused__int_mm_*_elu_mul_round`）。

**实施教训（已修复,勿回退）：**
1. swap 后新建 QuantLinear 默认 `training=True`,必须 `ql.train(lin.training)` 继承状态,
   否则 eval 模型上静默走 FP 分支（INT8 不生效,且 CosSim=1 极具迷惑性）。
   验证手段：`qrt/bench/_check_int8.py`（kernel 名 grep `_int_mm`）。
2. INT8 dequant 必须 int32→fp32→half；int32 直接 `.half()` 溢出 inf（量级 K·127²≈8M > 65504）。

### P1 ✅ 已完成（2026-06-10,bench_ablation.py,真实权重）

**实验设置**：权重 = `AME_Locomotion-main/pretrained/ame1.pt`（iter 21000 真实收敛
checkpoint,variant="ame"：use_global_context=False/topk=None,与 GLAD 共享
CNN/MHA/MLP 主干）；obs = `make_terrain_obs` 结构化地形（xy 近常量网格 + 8cm 台阶
平滑高度场,校准/测试 seed 分离）。g1_locomotion 的 ckpt 是 git-lfs 指针不可用；
真实 Isaac Lab rollout obs 接口已留（`calib.make_plan(obs_fn=...)`,接入即换）。

**① 逐层/全量消融（eager 精度,FP16 基线 CosSim=0.99999961 / relRMSE=8.7e-4）**

| 配置 | CosSim | relRMSE | 备注 |
|---|---|---|---|
| naive 全 5 层 | 0.99948 | 3.21e-2 | |
| **LET 全 5 层** | **0.99977** | **2.14e-2** | relRMSE 改善 33% |
| naive [actor.0]（最敏感层） | 0.99977 | 2.12e-2 | |
| **LET [actor.0]** | **0.99997** | **7.50e-3** | 误差降近一个量级 |

逐层均 LET≥naive；敏感度排序 actor.0 ≫ actor.6 > actor.4 > emb > actor.2。

**② 训练漂移实验（30 iter dummy PPO update lr=1e-4,FP32 训练器产生权重轨迹,
每 iter refresh 后测试,论文在线校准章节核心数据）**

| iter | LET+在线EMA | LET 静态 | naive 静态 |
|---|---|---|---|
| 0 | 0.99977 | 0.99977 | 0.99948 |
| 10 | 0.99905 | 0.99873 | 0.99692 |
| 30 | **0.99825** | 0.99697 | 0.99359 |

结论：LET 静态全程优于 naive（+3.4e-3 @30）；静态 s_z 失配随权重漂移累积；
**在线 EMA s_z（hook 挂 PPO update 前向,零额外推理成本）把退化斜率压平**
（iter20→30：EMA 掉 3e-4,静态掉 7e-4,naive 掉 1.3e-3）。
实现要点（坑,已修）：EMA 统计的对象必须是 **LET 变换后 z=αx+β** 的 absmax
（s_z 的语义）,直接收集 x 会把 scale 设错尺度,立即劣化。

**③ refresh CUDA Graph 化**：1.42ms → **0.146ms（9.7×）**,与逐层重算 bit-exact
（`InferenceRunner.refresh_quant(use_graph=True)`,默认开启）。

**④ AME 配置延迟（187 token 全量 MHA,同 GPU 同次测量）**：
FP16 eager 6.16ms → FP16 compile 2.02ms → **LET-INT8 compile 1.79ms**——
该配置下 INT8 比 FP16 快 11%（MHA KV/MLP 占比上升,INT8 访存收益显现）,
佐证 P2 conv INT8 的方向。

**新增修复（勿回退）**：`swap.restore` 创建的 nn.Linear 也必须继承 train/eval
状态（否则下一轮 swap 静默失效——消融循环必踩）;漂移实验训练器必须 FP32
（纯 FP16 参数直接上 Adam 会 nan,标准实践=FP32/AMP 训练+FP16 rollout）。

### P1.5 ✅ 已完成（2026-06-10,update 编译 + 双模型架构,超预期）

**架构定型（双模型形态,可直接迁移 Isaac Lab）：**
FP32 master 训练器（`TrainStepRunner`：bf16 autocast + compile 训练前向,
AOTAutograd 自动编译反向,fused Adam）+ 独立 FP16/INT8 推理模型
（`InferenceRunner`）+ 每 iter `sync_weights_from()`（参数/BN-buffer cast copy
+ requantize,整体单张 CUDA Graph）。标准顺序：构建→swap→**sync→warmup**→循环。

**同进程完整对照（bench_train_loop,协议同 TRT）：**

| 配置 | rollout | update | sync | total | ×FP32 | vs TRT 最优 122ms |
|---|---|---|---|---|---|---|
| A FP32 eager | 148.2ms | 49.8ms | — | 198.0ms | 1.00× | 0.62× |
| E 双模型(update eager) | 41.0ms | 52.0ms | 0.10ms | 93.1ms | 2.13× | 1.31× |
| **F +compile update** | 41.7ms | **14.5ms** | 0.09ms | **56.3ms** | **3.52×** | 2.17× |
| **G +LET-INT8 rollout（完全体）** | 41.7ms | 14.2ms | 0.27ms | **56.2ms** | **3.53×** | **2.17×** |

- update 49.8→14.5ms（3.4×,bf16 tensor core + inductor 前向/反向融合双重贡献;
  bench 为 dummy loss,真实 PPO loss 经 `forward_mean()` 接口同样可编译）
- **sync 0.09–0.27ms**：权重交接（含 INT8 requantize）≈ 免费,TRT refit 体系
  （快照线程+双缓冲+K-step 摊销）彻底归零
- 编译成本（inductor 缓存命中）：rollout 1–4s + train 图 3–7s,一次性
- 训练数值安全：bf16 master-weights 形态参数全程有限（纯 FP16 训练会 nan,
  C/D 单模型配置仅作计时对照,勿用于真实训练）

**实施教训（已修,勿回退）：**
1. CUDA RNG 的 graph-safe state（seed/offset extragraph tensor)是 per-device
   单例,且"全部注册 graph 析构后会释放重建"。若(重)建发生在 inference_mode
   capture 内 → 成为 inference tensor → 之后任何 grad 模式 capture(如编译
   训练图的 cudagraph trees record)报 'Inplace update to inference tensor'。
   解法：`runtime._ensure_graph_rng_state()` 在普通上下文空捕获并**永久持有
   哨兵 graph**,两个 Runner 构造时自动调用。
2. `sync_weights_from` 把推理副本参数 `requires_grad_(False)`(纯推理语义,
   亦使 inference_mode 内 in-place 合法);refresh/sync 统一 inference_mode 装饰。
3. `dict.get(a) or dict.get(b)` 在值为 Tensor 时触发布尔歧义,需显式 None 判断。

### P2 ✅ 已完成（2026-06-11,qconv.py: GLAD 形状特化 Triton kernel）

**实现**：`qrt/qconv.py` —— 两个形状特化 kernel 替换整条 CNN 链
（瓶颈本质是 inductor 通用 conv template 对"空间 11×17/通道 3→16→64 极小、
B=4096 巨大"形状效率仅 ~25%,定制 kernel + INT8 双管齐下）：
- kernel1：conv1(5×5,s2,fp16 `tl.dot`) + bias + ReLU + BN1 + quantize
  → z8 [B·187,16] int8（中间激活减半,NHWC,LET-α/β 可并入 BN1 仿射）
- kernel2：conv2(3×3,implicit-GEMM,int8 `tl.dot`/IMMA,kh 三步循环×K64) + dequant
  + bias + ReLU + BN2 → 直接产出 local_features [B·187,64] 行主序,
  下游 permute/reshape 化为零拷贝视图（profiler 证实无物化 copy）
- `QuantCNN` 子模块沿用原 Sequential 索引名(map_cnn.0/2/3/5) → sync/optimizer
  路径全兼容;train 态走原始算子;requantize_() 全 GPU 重打包(graph 安全)
- custom op(`torch.library.custom_op`, register_fake)入图,cudagraph trees 兼容

**单测（_test_qconv.py,真实分布 obs）**：kernel1 误差 0.52 量化步（索引/边界/融合
全对）;整链 CosSim 0.999935;eager 下 vs fp16 map_cnn **6.2×**(2.275→0.367ms);
compile 集成 bit-exact。

**单步（bench_rollout,同一 state_dict）**：

| 配置 | 单步 | ×FP32 | CosSim |
|---|---|---|---|
| D FP16 compile（P0） | 1.693ms | 3.68× | 1.000000 |
| **H FP16 + QuantCNN（速度最优）** | **1.508ms** | **4.12×** | 1.000000 |
| **I LET-INT8(6层) + QuantCNN（全家桶）** | **1.619ms** | **3.84×** | 0.999785 |

conv 链 0.8 → 0.375ms(conv1 0.123 + conv2 0.210 + 输入准备 0.042)。
refresh 全家桶(6 Linear + QuantCNN, 单 CUDA Graph)0.240ms,conv 热更新一致性 0。

**系统级（bench_train_loop 双模型,同进程对照）**：

| 配置 | rollout | update | sync | total | ×FP32 | vs TRT 122ms |
|---|---|---|---|---|---|---|
| A FP32 eager | 148.5ms | 32.7ms | — | 181.2ms | 1.00× | 0.67× |
| F 双模型(无 INT8) | 43.0ms | 17.1ms | 0.12ms | 60.3ms | 3.01× | 2.02× |
| **G 完全体(LET+QuantCNN)** | **38.6ms** | 15.0ms | 0.34ms | **54.0ms** | **3.36×** | **2.26×** |

G 的 rollout 首次反超 F —— INT8 conv 收益盖过 LET-linear 微亏,
INT8 在系统层面转为净正贡献。

**实施教训（已修,勿回退）**：
1. `tl.arange` 必须 2 的幂 → conv1 K pad 80→128 单 dot;conv2 改 kh 三步循环×K64(48 有效+mask)。
2. Triton 3.6 的 int8 dot 不接受位置参数 acc 形式,必须 `acc += tl.dot(A,B,out_dtype=tl.int32)`。
3. CUDA 上 `int32 @ int32` 无实现,对拍参照用 `(a.float()@b.float()).int()`(K 小时 fp32 精确)。

**剩余空间（移交 P3）**：profile 显示当前最大块是 GPS/TopK scorer 链 ~0.73ms
（softmax+bmm+cat 物化+两个 N=1 GEMM）—— 代数重写 `scorer(cat(a,b))=a@W₁+b@W₂`
+ 合并 N=1 投影,预期再省 ~0.4ms;MHA 碎块 ~0.2ms 手写 single-query attention。

### P3 ✅ 已完成（2026-06-11,encoder_opt.py: 数学等价图重写）

**实现**：`qrt/encoder_opt.py` —— monkeypatch 数学等价版 `_encode_terrain`
（不改 actor_critic_encoder.py,原实现保留为参照,patch/unpatch 可逆）：
1. **GPS/scorer 代数重写**（消除当时最大块 ~0.73ms）：
   `scorer(cat([local,query])) ≡ local@Wₛ[:,:D]ᵀ + (query@Wₛ[:,D:]ᵀ+b)`,
   且 scorer-local 与 GPS selector 合并为单个 [D→2] GEMM——
   cat 物化 [B,187,128] 消失,98MB 的 local_features 只读一次；
2. **MHA 显式重写**：nn.MultiheadAttention(need_weights 慢路径) → 手写
   single-query cross-attention(H=16,head_dim=4),权重取同一参数对象
   (训练更新自动生效),attn_weights 语义一致(头平均 [B,1,K])。

**等价性（_test_encoder_opt.py）**：FP32 eval `action_mean` max|diff|=5.96e-8、
topk index 不匹配率 0、gc/attn 1e-9 级;train 态(Gumbel 同 seed)4.77e-7
—— 精确恒等,训练/推理两路径均可 patch。

**单步**：patch 单独 1.11×;**patch+QuantCNN+compile = 1.213ms**
(对照 P0 1.69 / P2-H 1.51;vs FP32 eager ≈5.1×)。

**系统级（bench_train_loop,G=LET+QuantCNN+双侧patch;注:本两次 run 共享
GPU 有他人负载 34-38%,绝对值为劣化条件下限,两次独立 run 一致）**：

| 配置 | rollout | update | sync | total | ×FP32(同run) | vs TRT 122ms |
|---|---|---|---|---|---|---|
| A FP32 eager | 178.6ms | 40.3ms | — | 218.9ms | 1.00× | 0.56× |
| F 双模型(P1.5 形态) | 49.2ms | 16.1ms | 0.20ms | 65.5ms | 3.34× | 1.86× |
| **G P3 完全体** | **30.2ms** | 13.5ms | 0.48ms | **44.2ms** | **4.96×** | **2.76×** |

（按历史干净基线 181–198ms 保守折算:G ≈ **4.1–4.5× vs FP32**。）

**剩余可选**（移交后续研究,优先级让位于真实训练接入）：
- 在线 QAT（L_PPO+λ·L_LET,QuantLinear train 分支已留接口）；
- GPS softmax+bmm 链 Triton 单 kernel（当前 ~0.25ms）;conv2 kernel 带宽利用 50%→70%；
- INT4 / 混合精度逐层搜索（消融框架直接复用）。

---

### P4 ✅ 真实训练接入层（2026-06-11,integration.py + rsl_rl 全链路验证）

**一行接入（Isaac Lab 训练代码零修改）**：
```python
# scripts/rsl_rl/train.py 中 runner 创建之后、runner.learn() 之前:
from qrt.integration import accelerate_runner
acc = accelerate_runner(runner)          # 默认全家桶+update 编译
runner.learn(...)
```

**机制**：deepcopy FP16 推理副本(LET-INT8 + QuantCNN + encoder 重写) →
monkeypatch `policy.update_distribution/evaluate`(inference_mode → 编译副本,
梯度态 → 编译的 FP32 master 前向/AOT 反向) → wrap `alg.update` 末尾自动
`sync_weights_from`。**每 iteration 同步,无 K-step 滞后,严格 on-policy**;
校准数据 = `collect_calib_obs` 真实 env 滚动采集(TRT 已知问题 3 闭环)。
量化模块分支条件已改为 `torch.is_grad_enabled()`(rsl_rl rollout 是
train 态+inference_mode —— Gumbel 探索保留并走 INT8;唯一语义近似:
推理副本 BN 固定 eval/running-stats,训练侧 BN 照常 train 并经 sync 闭环)。

**全链路验证（bench_real_loop.py: mock 地形 env + 真实 OnPolicyRunner.learn,
B=2048,24 步 rollout + 5 epoch×4 minibatch adaptive-KL update,共享 GPU 条件）**：

| 模式 | 每 iter(纯网络栈) | vs base | 说明 |
|---|---|---|---|
| base(原始 FP32) | 3527ms | 1.00× | **update 占 ~95%**(5ep×4mb×12288 样本) |
| rollout(只加速推理) | 3360ms | 1.05× | rollout 部分 ~150→~25ms |
| full_fp32(+update 编译 fp32) | 2195ms | 1.60× | update ~3370→~2050ms |
| **full(+update 编译 bf16)★默认** | **1227ms** | **2.87×** | update →~1080ms(≈3.1×) |

三/四模式 10 iter 训练参数全程健全,学习信号同向。
**真实协议画像与 bench 协议截然不同**: update 才是大头(95%),
bf16 autocast 把它打 3.1× 后系统达 2.87×。

**bf16 update 的数值安全设计与验证**：autocast 只包网络前向(图内 GEMM bf16,
AOT 反向 bf16,参数/梯度 fp32),mean/value 出图即 `.float()` ——
PPO 的 ratio/KL/entropy/loss 全程 fp32 与原版逐字一致。
**梯度保真度实测 cos=0.999998**(同数据同 loss,bf16 编译路径 vs FP32 eager,
eval 态消除 Gumbel 随机性)。bf16 无需 GradScaler。
Amdahl 重算(网络栈 2.87×): 仿真占比 10%→端到端 2.42×;30%→1.93×;50%→1.58×。

**接入等价性消融（eval 态,随机初始化权重的最坏情况）**：
fp16+compile / +rewrite / +int8_cnn 均 CosSim=1.000000;
int8_linear 为唯一偏差源(0.9926,max|diff| 0.04)——随机 init 网络输出幅度
仅 ~0.06,静态量化噪声相对放大;真实收敛权重下同配置为 0.9998(P1),
且偏差 ≪ 探索 std=1.0。**建议训练策略**: 初期 `int8_linear=False`,
收敛趋稳后开启(或全程开启 + reward 曲线 A/B)。

**实施教训（已修,勿回退）**：
1. 等价性对拍必须 eval 态——train 态两路径各自采样 Gumbel,差异是探索噪声
   (首测 0.984 即此假象);
2. rsl_rl `log_dir=None` 时 `store_code_state/logger_type` 缺防护(bench 已加垫片);
3. 真实协议 update(B·24/4 样本 FP32 反向)B=4096 时峰值 ~15GB,共享 GPU 需注意;
4. update 梯度图(12288 batch)冷编译 ~301s,一次性(inductor 磁盘缓存)。

### P5 ✅ 动态 scale 融编译图(2026-06-11,bench_dynamic.py,借鉴 torchao float8 经验)

QuantLinear 新增 `act_mode`,三方对照(ame1 真实权重,与 P1 同设置):

| 模式 | 机制 | 精度(静态) | 漂移@30iter | 单步延迟 |
|---|---|---|---|---|
| static(P1 现状) | 校准 s_z + EMA 补丁 | 0.99977 / 2.14e-2 | 0.99700 | 1.721ms |
| **dynamic(新默认)** | s_z=absmax(z)/127 **图内现算**,α/β 仍 LET 校准 | **0.99979 / 2.06e-2(最优)** | **0.99815** | 1.836ms(+0.115) |
| dynamic_full | α/β=batch 统计、权重量化也进图,**零校准** | 0.99969 / 2.48e-2 | 0.99553(最差) | 1.978ms |

**结论与科研发现**：
1. **dynamic 成为新默认**(integration `act_mode="dynamic"`)：精度三者最优(每 batch
   scale 精确匹配)、漂移退化减半(≈EMA 效果但无 hook/超参,EMA 方案退役)、
   延迟代价 +0.115ms/步(系统层面 ~2.8ms/iter,可忽略);剩余漂移源=α/β 静态失配,
   解药为定期重校准(秒级)或在线 LET(QAT 接口)。
2. **"学习的 α vs 统计的 α"(反直觉发现,论文素材)**：dynamic_full 的 α/β 永远
   "新鲜"却漂移最快——纯统计 α=1/σ 只优化激活侧,**权重侧 W'=W·σ 的量化误差
   不受控**;LET 学习的 α 在校准时平衡了激活/权重两侧的误差,即使过时仍占优。
   这是 LET"学习"属性价值的直接实验证据(W4 下预计差距更大)。
3. dynamic_full 的"零校准"特性保留为可选(原型/快速实验场景)。
4. torchao 的 delayed→dynamic 演化路径在 INT8+RL 场景复现验证。

### P6 ✅ 在线伴随校准(2026-06-11,cocalib.py + bench_cocalib.py)

**机制(与策略优化完全解耦,零 λ)**：LET 重建损失天然 detach 权重(仅 α/β 有
梯度)+ 双模型架构下训练前向不经量化路径 ⇒ α/β 优化是独立回归问题
("让推理副本逼近 FP32 真值"),独立 Adam、warm-start(参数与动量永不重置)、
挂在 update 之后 sync 之前;激活用小批 eager FP32 前向采集(不碰编译路径——
dynamo 对 module hook 会重编译,实测教训)。**校准对齐对象=FP32 master 前向**
(bf16 仅为训练内部计算精度,不定义策略真值)。

**四方对照(ame1+结构化 obs,50 iter 漂移,全部 act_mode=dynamic)**：

| 配置 | CosSim@50 | 累计步数 | 特征 |
|---|---|---|---|
| A frozen(α/β 冻结) | 0.99759 | 0 | 持续退化 |
| B periodic-10(每10 iter fresh 200步) | 0.99785 | 5000 | 中段最优(0.9985-0.9993)但有重校准方差尖刺 |
| **C fixed-1(每 iter 每层 1 步连续跟踪)★推荐** | **0.99790** | **250(=B 的 5%)** | 全程平滑,末值最优 |
| D adaptive(τ=1.25 误差驱动) | 0.99748 | 40 | 触发失明(见下) |

**结论**：**连续小步跟踪(fixed-1)> 周期性重校准 > 误差驱动间歇**。
fixed-1 以 periodic 5% 的成本取得更优末值与平滑曲线;每 iter 开销 ~5-10ms
(占真实 iter <1%),warm-start 使有效步长自适应漂移速度(漂移大→梯度大)。
k 是平滑的成本-精度旋钮(fixed-2/4 可进一步贴近 B 的中段水平)。

**adaptive 的两个失败模式(论文 discussion 素材)**：
1. baseline 若在未优化时也 EMA 跟踪 L → 阈值随漂移水涨船高,温水煮青蛙式失明(已修为仅优化后锚定);
2. 修复后仍迟钝——**单层重建损失与端到端误差弱耦合**(每层 L 涨<25% 时
   5 层累积+下游放大已使端到端掉 2e-3),相对阈值天然不敏感。
   ⇒ 既然 fixed-1 成本已可忽略,触发机制属于过度工程。

**接线**:`accelerate_runner(runner, cocalib=True)`(默认关;Isaac A/B 确认
量化漂移影响 reward 后开启)。至此量化参数生命周期三时间尺度全部闭环:
s_z 每步图内动态(P5)/ W8·s_w·b' 每 iter 重折叠(P1.5)/ α/β 每 iter 1 步
伴随跟踪(P6,可选)。

### P6.B ✅ 零额外前向的激活采集(tap.py,2026-06-11)

**动机**:P6 的 eager 源每 iter 多一次小批 FP32 前向(~5-10ms);追求架构自洽,
让校准激活搭 update 前向便车,0 额外前向。**先验证两个前提再实现**:

1. **bf16 激活可否用于 LET 校准**(_test_bf16_calib.py):bf16 autocast 前向激活
   relΔ~7e-3、深层 α/β 偏移到 cos 0.985,但**端到端 ΔCosSim=7.7e-6,漂移 30iter
   全程吻合 1e-4**。原因:LET 恒等变换性质 ⇒ α/β 误差被权重折叠精确吸收,
   最优解附近存在平坦等效盆地(鲁棒性观察,论文素材)。⇒ bf16 采集无害。
2. **激活源对齐对象 = FP32 master**(非 bf16);校准内部 `.float()`+FP32 权重
   参照,采集端 dtype 不影响损失空间。

**实现**:`qrt::tap` custom op(前向恒等返回 + 旁路 detach 写静态 buffer,
register_fake + register_autograd 恒等直通)+ `TapLinear`(转移 Parameter 包装,
named_parameters 路径不变)+ `install_taps` 装在 FP32 trainer 目标层。
cocalib `source="tap"` 从 buffer 读,不做独立前向。

**数值等效验证**(bench_cocalib E vs C):tap(0 前向)与 eager(独立前向)
50-iter 漂移曲线吻合,末值 0.99790 vs 0.99791(差 1.4e-5),步数同为 250。

**三个工程坑(spike 抓到,已固化)**:
1. custom-op 须 `register_autograd`(恒等直通)否则编译训练前向 backward 报
   "no autograd formula";
2. cocalib+tap 时训练前向须 `max-autotune-no-cudagraphs`——cudagraph replay 不
   重放 custom-op Python 副作用,buffer 失真(integration 已自动切换);
3. TapLinear 转移 weight/bias Parameter(不作子模块),否则 named_parameters 变
   `actor.0.lin.weight`,sync_weights_from 配对失败。

**接线**:`accelerate_runner(runner, cocalib=True, cocalib_source="tap")`。
eager 源仍为默认(更稳、不强制 no-cudagraphs);tap 源用于追求零额外前向。

---

## 七、全版本统一横评（2026-06-11,空闲双 3090,单进程同口径）

### 7.1 单步推理（bench_matrix.py,GLAD 主配置,B=4096,逐档叠加）

| 配置 | 单步 | ×FP32 | ×FP16e | 边际 |
|---|---|---|---|---|
| A FP32 eager | 6.214ms | 1.00× | 0.58× | — |
| B FP16 eager | 3.607ms | 1.72× | 1.00× | 半精度 |
| C FP16 compile(P0) | 1.692ms | 3.67× | 2.13× | **图融合 −1.9ms** |
| D + encoder 重写(P3) | 1.409ms | 4.41× | 2.56× | cat/gemv 消除 −0.28ms |
| **E + QuantCNN(P2,速度最优)** | **1.054ms** | **5.90×** | **3.42×** | **INT8 conv −0.36ms** |
| F + LET-INT8 linear(完全体) | 1.157ms | 5.37× | 3.12× | **−0.10ms(变慢)** |
| G naive-INT8 linear | 1.117ms | 5.56× | 3.23× | 同上 |

**关键**:推理速度最优是 **E(5.90×),不是全 INT8 的 F**。给 6 个小 Linear 上
INT8 反而 +0.1ms(quantize/dequant kernel 开销 > 访存节省,roofline 早有预言)。
加速边际:图融合(−1.9ms)≫ INT8 conv(−0.36)> 图重写(−0.28)。
**INT8 linear 是精度/科研维度,纯论速度为微负项**;生产部署推荐 E,INT8 linear
留给保精度算法研究或 B=1 板端(roofline 翻转、权重访存主导)。

### 7.2 训练系统 per-iter（bench_real_loop.py,真实 rsl_rl 协议,B=2048,扣 env）

| 配置 | net/iter | ×FP32 | 梯度保真 | 备注 |
|---|---|---|---|---|
| base(FP32 eager) | 2825ms | 1.00× | — | update 占 ~95% |
| **full(完全体,bf16 update)** | **990ms** | **2.85×** | cos 0.999998 | 默认推荐 |
| full_cc(+ 伴随校准 eager) | 1014ms | 2.79× | 0.999998 | cocalib +23ms/iter |
| full_cc_tap(+ 伴随校准 tap) | 1058ms | 2.67× | 0.999997 | 见下 |

**反直觉发现(诚实记录)**:`full_cc_tap`(tap 零额外前向)反而比 `full_cc`
(eager 独立前向)**慢 45ms**。原因:tap 强制训练前向用 no-cudagraphs(cudagraph
不重放 custom-op 副作用),损失的 cudagraph 收益(~45ms)超过省下的那次采集前向
(~5-10ms)。⇒ **在本模型 update-cudagraph 收益显著的场景,eager 源端到端更优**;
tap 的"0 额外前向"优势要在采集前向成本占比更高(更大 batch / 更多量化层)时才
兑现。又一个"实测推翻直觉"的案例。**默认 cocalib_source="eager"。**

**口径说明**:7.1(B=4096)与 7.2(B=2048,受显存限制)不同 batch;7.2 空闲 GPU
本批 base=2825ms,与早期共享 GPU 批次(3527ms)不可直接比,但同批次内相对比有效。
接入等价性本批 CosSim=0.944(B=2048 随机初始化权重 + 动作幅度仅 0.06,量化噪声相对
放大;真实收敛权重为 0.9998,见 P1)——梯度保真 cos=0.999998、学习信号正常。

**Isaac Lab 训练机运行清单**：
1. 拷贝 `GLAD_Quant/qrt/` + `GLAD_Quant/quant/` 到训练机,`sys.path` 加入 GLAD_Quant;
2. 依赖: torch≥2.8(实测 2.10)+ triton(torch 自带)+ tensordict(rsl_rl 本就依赖);
3. train.py 加两行(见上);首次运行编译 ~5-8 分钟(缓存后秒-分钟级);
4. 监控: `TORCH_LOGS=recompiles` 确认无反复重编译;前 2 iter 慢为编译,属正常;
5. A/B 实验建议: base / full(int8_linear=False) / full 三组 reward 曲线对照。

---

## 五、风险与回退

| 风险 | 缓解 |
|---|---|
| inductor 默认/reduce-overhead 模式触发 CUDA invalid configuration（实测 bug） | 固定 `mode=max-autotune`（已验证）；或 no-cudagraphs+手动 graph |
| freezing 模式 view/stride bug（实测） | 不用 freezing；BN fold 在 P2 手动做 |
| 真实训练循环 guard 误触发重编译 | `cache_size_limit` 调大；`TORch_LOGS=recompiles` 监控；P0 即验证 |
| 内置 cudagraph trees 与 Isaac Lab 流冲突 | 切 manual-graph 形态（已验证等价,1.640ms） |
| INT8 影响 PPO 收敛 | LET+真实 obs 校准；逐层消融控制；QAT 兜底 |
| 共享 GPU 干扰基准 | 沿用 `CUDA_VISIBLE_DEVICES=1` 约定 |

---

## 六、对用户三痛点的逐项回应

| 痛点 | 本方案 |
|---|---|
| TRT 编译过长稀释加速 | 冷编译 35s/缓存 8.4s（一次性）；权重更新 0 成本,无 refit 概念 |
| 模型量级不够、预处理开销大 | 不再依赖 INT8 算力收益；收益来自图融合（2.12×）；quantize 融入图内无独立开销 |
| 环境依赖冲突 | 零新增依赖（torch 2.10 自带全部原语）,不碰 torchao/TRT/ORT |

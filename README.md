# GLAD 量化加速 — qrt 主线

> **一句话**:用 `torch.compile` 图融合 + CUDA Graph + 选择性 INT8(自写 Triton kernel)+ LET 校准 + encoder 图重写 + 双模型训练架构,把 GLAD 地形编码器在 PPO 训练中的推理与训练加速 **2.85–5.9×**,替代编译成本过高、权重同步笨重、依赖沉重的 TensorRT 路线。
>
> 平台:RTX 3090(SM 8.6)· PyTorch 2.10 + Triton 3.6 · **零新增依赖**(不用 TensorRT/torchao/ORT)
> 旧 TensorRT 路线归档于 [`legacy_trt/`](legacy_trt/README.md),保留作对比基线。
> 完整方案与逐阶段实验记录见 [`PLAN_compile_quant.md`](PLAN_compile_quant.md)。

---

## 0. 核心成果

| 指标 | TRT 旧路线 | **qrt 新主线** | 提升 |
|---|---|---|---|
| 单步推理(B=4096) | INT8 最优 3.274 ms | **1.054 ms(5.90× vs FP32 / 3.42× vs FP16)** | 2.2× 于 TRT |
| 训练系统(真实 PPO 协议) | Async refit 122 ms/iter(1.436×) | **990 ms/iter（B=2048）= 2.85× vs FP32** | — |
| 权重同步 | refit + K=4 摊销 + 异步双缓冲(滞后 1–4 iter) | **每 iter 同步,0.2–0.5 ms,严格 on-policy** | ~免费 |
| 构建成本 | ~5 min/配置 | 冷编译 35 s–5 min(一次性,磁盘缓存后秒级) | 量级降 |
| 依赖 | TensorRT + onnx + graphsurgeon + ORT | **仅 PyTorch(Triton 随附)** | 零冲突 |
| 梯度保真(训练) | — | bf16 update vs FP32 eager **cos = 0.999998** | — |
| LET 科研线 | ONNX 图插桩 | 模块内原生算子,QAT 可共享参数,在线伴随校准 | 增强 |

> 口径:推理 B=4096、系统 B=2048(显存所限);各表为同进程同 GPU 状态测得,**表内相对加速比可靠**,跨表绝对值因 batch / GPU 负载不同不可直接比。

---

## 1. 背景与动机

**任务**:GLAD(Global-Local Attention Decomposition)是地形编码器,用于 Isaac Lab + Unitree G1 的 PPO 感知运动训练(B=4096 并行环境,每步需 actor/critic forward)。目标是加速这个网络的推理(rollout)与训练(update)。

**为什么放弃 TensorRT**(详见 [legacy_trt/](legacy_trt/README.md)):①每配置重编译 ~5 min;②RL 每 iter 更新权重必须 refit,refit 太慢被迫 K=4 摊销+异步双缓冲,**破坏 on-policy 语义**,系统加速仅 1.436×;③TRT FP16 反而比 PyTorch FP16 eager 慢 72%(小 kernel 调度低效)。

**瓶颈的真相**(kernel 级 profiler 实测,推翻四个流行假设):

| 假设 | 实测结论 |
|---|---|
| "INT8 算力能加速" | ❌ GLAD 的 GEMM 全部**访存受限**,INT8 算力无用;价值只在"访存减半",且必须融合 |
| "小模型慢是 CPU launch 开销,CUDA Graph 能救" | ❌ replay 仅快 1%,3.66ms 全是 GPU 上大量小 kernel 真实执行时间(但发现:权重 in-place 更新后 graph replay 立即生效 → 取代 refit) |
| "torch.compile 对此模型无效" | ❌ 旧实验只编了 4% 的子模块;整模型 0 graph break,max-autotune 直接 2.12× |
| "瓶颈在卷积/矩阵乘" | ❌ 瓶颈是**图结构病灶**:cat 物化 0.71ms + N=1 退化 gemv 0.80ms + 碎 elementwise 1.1ms |

**技术路线排序由此确定**:编译器融合 > 代数重写 > 形状特化 kernel > INT8 访存减半。INT8 排最后,且按算子 roofline 逐个部署而非一刀切。

---

## 2. 快速开始

### 接入真实训练(Isaac Lab / rsl_rl,训练代码零修改)

```python
# scripts/rsl_rl/train.py 中,runner 创建之后、runner.learn() 之前:
from qrt.integration import accelerate_runner
acc = accelerate_runner(runner)        # 默认: INT8-conv + LET-INT8-linear + 图重写 + bf16 update
runner.learn(...)                      # 原样,无需改动

# 可选档位:
# accelerate_runner(runner, int8_linear=False)        # 推理纯 FP16+INT8conv(速度最优,见 §6)
# accelerate_runner(runner, cocalib=True)             # 开启 α/β 在线伴随校准(抗训练漂移)
# accelerate_runner(runner, update_amp="fp32")        # update 不用 bf16(数值对照)
# acc.detach()                                        # 完全还原 runner
```

### 复现实验(远程 RTX 3090)

```bash
# 从 GLAD_Quant/ 目录,所有脚本自带 mock/path 设置
python qrt/bench/bench_matrix.py        # 单步推理横评(逐档叠加,§6.1)
python qrt/bench/bench_real_loop.py     # 真实 rsl_rl 协议端到端(QRT_MODE=base/full/...)
python qrt/bench/bench_ablation.py      # LET vs naive 精度 + 训练漂移(需 ame1.pt)
python qrt/bench/bench_cocalib.py       # 伴随校准四方对照
```

---

## 3. 架构

```
qrt/
├── runtime.py       InferenceRunner   推理加速器(compile / manual-graph / eager 三形态)
│                                      refresh_quant / sync_weights_from 权重同步原语(CUDA Graph 化)
│                    TrainStepRunner   bf16 autocast + compile 训练步(AOTAutograd 自动编译反向)
│                    _ensure_graph_rng_state  RNG 哨兵 graph(见 §7 坑 6)
├── qlinear.py       QuantLinear       W8A8 Linear,LET 前置,训练/推理双路径,静态/动态 s_z
├── qconv.py         QuantCNN          GLAD CNN 链的两个形状特化 Triton kernel(INT8 implicit-GEMM)
├── encoder_opt.py   patch_encoder     _encode_terrain 数学等价重写(scorer 代数化 + MHA 手写)
├── swap.py          模型手术          nn.Linear ↔ QuantLinear 替换/还原/批量刷新(消融框架)
├── calib.py         校准管线          激活采集 → LETQuantizer(复用 ../quant/let.py)→ swap plan
├── cocalib.py       CoCalibrator      α/β 在线伴随校准(解耦优化,误差驱动 + 早停)
├── tap.py           TapStore/TapLinear 编译图内零额外前向激活采集(custom op)
├── integration.py   accelerate_runner rsl_rl / Isaac Lab 一行接入(路由 + 生命周期)
└── bench/           bench_* 性能/精度套件 + _test_* 单元对拍

quant/               LET 量化工具库(两条路线共用):let.py(LETQuantizer)/ fake_quant / observer
rsl_rl/              模型与训练框架:modules/actor_critic_encoder.py(GLAD 模型,勿改结构)
engines/             glad_actor_base.onnx(唯一起点 ONNX,两条路线共用)
tools/ssh_helper.py  远程 RTX 3090 操作封装
```

**分层逻辑**:底层算子(qlinear/qconv)→ 图层(encoder_opt + torch.compile)→ 运行时(runtime:graph 化权重同步)→ 训练集成(integration)。每层可独立开关(消融),全部以"运行时改造"实现——**`actor_critic_encoder.py` 与 PPO/Runner 源码一行未改**。

---

## 4. GLAD 模型结构(被加速对象)

```
输入 obs [B, 96 + 33×21×3]
  1. CNN: Conv(3→16,k5,s2)+ReLU+BN → Conv(16→64,k3)+ReLU+BN → local_features [B,187,64]
  2. GPS 全局注意力: Linear(64→1)→softmax → gc;  query_projector Linear(128→64) → query
  3. TopK(K=32): topk_scorer Linear(128→1) + top-32 → local_sparse [B,32,64]
  4. MHA: single-query cross-attention(D=64,H=16,head_dim=4)→ foothold [B,64]
  5. Actor MLP: cat(foothold,gc,proprio)[B,224] → 512→256→128→29
```

---

## 5. 实现详解(逐阶段)

### P0 · 全模型编译(图融合)

`torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)` 编译 actor/critic 两条纯函数路径。inductor 自动完成:BN/ReLU 融入相邻 kernel、elementwise 链合并、GEMM 用 Triton 模板生成并把 ELU 融进 epilogue、view/permute 链消除。max-autotune 自带 CUDAGraph Trees;另提供 manual-graph 形态(no-cudagraphs 编译 + 手动 `torch.cuda.CUDAGraph`)。
**FP16 eager 3.66ms → compile 1.69ms,CosSim=1.000000。** 必须用 max-autotune(默认/reduce-overhead 触发 CUDA config 错误,freezing 触发 view/stride 错误,见 §7)。

### P1 · LET-INT8 Linear（`qlinear.py`）

对 6 个 Linear(actor.0/2/4/6 + proprio_emb + query_proj)做 W8A8。**LET(Learnable Equivalent Transformation,源自 OmniQuant)**:

```
y = Wx+b ≡ (W/α)·(α⊙x+β) + (b−(W/α)β)  =  W'·z+b'
```

恒等变换,但 z = αx+β 的通道分布被 α、β 整形为 INT8 友好。α/β 用 Adam 最小化块重建损失 `MSE(Q_w(W')·Q_a(z)+b', Wx+b)` 校准 200 步(复用 `quant/let.py`,零修改)。前向(推理路径,inductor 把 quant/dequant/ELU 全融进 `_int_mm` 模板):

```python
z = x*alpha + beta;  q = round(z*inv_s_z).clamp(-128,127).int8
y = (_int_mm(q, w8.t()).float() * dq_scale + b_prime).to(x.dtype)   # int32→fp32 防溢出
```

**精度(真实收敛权重)**:LET 全 5 层 CosSim 0.99977 / relRMSE 2.14e-2 > naive 0.99948 / 3.21e-2(改善 33%);最敏感层 actor.0 上 LET 误差降近一个量级。**注意:速度上 INT8-linear 是中性偏负项(见 §6.3),它是精度/科研维度,不是提速器。**

### P1.5 · 双模型训练架构 + update 编译

纯 FP16 参数直接 Adam 会 nan,正确形态是 **FP32 master 训练器 + FP16/INT8 推理副本 + 每 iter 同步**:

```
FP32 master(原 policy)──每 iter sync_weights_from(单 CUDA Graph,0.2–0.5ms)──► FP16/INT8 推理副本
       ▲                                                                          │
       │ PPO update(梯度态,bf16 autocast 编译前向 + AOT 反向,fused Adam)        │ rollout(inference_mode
       └────────── optimizer.step() in-place ◄────────────────────────────────────┘  + Gumbel,编译)
```

- `TrainStepRunner`:bf16 autocast 只包网络前向,`mean/value` 出图即 `.float()` → **PPO 的 ratio/KL/entropy/loss 全程 fp32**,与原版逐字一致;参数/梯度 fp32(bf16 无需 GradScaler)。**梯度保真度 cos=0.999998**。
- `sync_weights_from`:同名参数 + BN buffer cast copy + 量化 buffer 重算,整序列捕获为一张 CUDA Graph(显存指针固定,optimizer in-place 不改指针)。**每 iter 同步,无 K-step 滞后,严格 on-policy。**

### P2 · INT8 卷积（`qconv.py`)

compile 后 conv+BN 链仍占 ~46%,但理论访存下限仅 ~25% 效率——inductor 通用 conv 模板对 GLAD 的病态形状(空间 11×17、通道 3→16→64、batch 4096)效率低。两个形状特化 Triton kernel 替换整条链:

- **kernel1**(fp16):conv1 + bias + ReLU + BN1 + 量化 → z8 `[B·187,16]` int8(中间激活减半);
- **kernel2**(INT8 IMMA):conv2 implicit-GEMM(`tl.dot` int8×int8→int32)+ 反量化 + bias + ReLU + BN2 → **直接产出 local_features `[B·187,64]` 行主序**,下游 permute/reshape 退化为零拷贝视图。

LET 在 conv 上的 α/β 直接并入 BN1 仿射(零额外算子)。`QuantCNN` 沿用 `map_cnn.0/2/3/5` 索引名,sync/optimizer/checkpoint 全兼容。**CNN 链 6.2×(eager 2.275→0.367ms);整模型 1.69→1.51ms;整链 CosSim 0.999935。**

### P3 · Encoder 图重写（`encoder_opt.py`,数学恒等）

monkeypatch 数学等价版 `_encode_terrain`(不改模型文件,可逆):

1. **GPS/scorer 代数化**:`scorer(cat([local,query])) ≡ local@Wₛ[:,:D]ᵀ + (query@Wₛ[:,D:]ᵀ+b)`,且 scorer-local 与 GPS selector 合并为单 [D→2] GEMM ——**消除 cat 物化([B,187,128],392MB 读写)+ 两次 N=1 gemv 合并**;
2. **MHA 手写**:`nn.MultiheadAttention`(need_weights=True 走 Python 分解慢路径)→ 紧凑张量式 single-query attention,权重取同一参数对象。

**等价性(FP32 eval)**:action_mean max|diff| 5.96e-8、topk 不匹配 0、train 态(Gumbel 同种子)4.77e-7。**单步 1.51→1.21ms。**(收益拆分:scorer 代数化 ~0.23ms 大头,MHA 重写 ~0.05ms 小头——见 §6.4。)

### P5 · 动态 s_z 融编译图(`act_mode`,借鉴 torchao float8)

激活 scale 改为图内现算 `absmax(z)/127`(inductor 融合,边际成本 +0.1ms/步),**校准 s_z / EMA / 漂移问题一次性消失**。三模式:`static`(P1 校准+EMA)/ `dynamic`(默认,s_z 图内,α/β 仍 LET)/ `dynamic_full`(α/β 也统计,零校准)。

**科研发现**:`dynamic_full` 虽 α/β 永远"新鲜"却漂移最快——纯统计 α=1/σ 只优化激活侧,**权重侧 W'=W·σ 的量化误差不受控**;LET 学习的 α 平衡两侧误差,过时仍占优。**这是 LET"学习"属性不可被统计替代的直接证据。**

### P6 · α/β 在线伴随校准(`cocalib.py` + `tap.py`)

冻结的 α/β 会随权重漂移失配(P5 消除 s_z 失配后唯一剩余漂移源)。CoCalibrator 在线保鲜:

- **与策略优化完全解耦,零 λ**:LET 重建损失天然 detach 权重(只 α/β 有梯度)+ 双模型下训练前向不经量化路径 → α/β 优化是独立回归问题("让推理副本逼近 FP32 真值"),独立 Adam、warm-start、挂在 update 后 sync 前;
- **对齐 FP32 master**(bf16 仅训练内部计算精度,实测 bf16 激活做校准端到端 ΔCosSim 7.7e-6 无害);
- **两种采集后端**:`eager`(独立小批 FP32 前向,稳,默认)/ `tap`(custom op 搭 update 前向便车,零额外前向)。

**对照结论**:连续小步跟踪(fixed-1,每 iter 每层 1 步)以 periodic 重校准 **5% 的成本**取得更优更平滑的精度保持(50 iter 0.99790 vs frozen 0.99760);误差驱动触发因"单层损失与端到端误差弱耦合"而过度工程。**量化参数生命周期三时间尺度闭环**:s_z 每步(P5)/ W8·s_w·b' 每 iter(P1.5)/ α/β 每 iter 1 步(P6,可选)。

---

## 6. 完整测试结果（RTX 3090,空闲,单进程同口径）

### 6.1 单步推理横评（`bench_matrix.py`,GLAD 配置,B=4096,逐档叠加）

| 配置 | 单步 | ×FP32 | ×FP16e | 边际贡献 |
|---|---|---|---|---|
| A FP32 eager | 6.214ms | 1.00× | 0.58× | — |
| B FP16 eager | 3.607ms | 1.72× | 1.00× | 半精度 |
| C FP16 compile(P0) | 1.692ms | 3.67× | 2.13× | **图融合 −1.9ms** |
| D + encoder 重写(P3) | 1.409ms | 4.41× | 2.56× | cat/gemv 消除 −0.28ms |
| **E + QuantCNN(P2,速度最优)** | **1.054ms** | **5.90×** | **3.42×** | **INT8 conv −0.36ms** |
| F + LET-INT8 linear(完全体) | 1.157ms | 5.37× | 3.12× | **−0.10ms(变慢)** |
| G naive-INT8 linear | 1.117ms | 5.56× | 3.23× | 同上 |

### 6.2 训练系统 per-iter（`bench_real_loop.py`,真实 rsl_rl 协议,B=2048,扣 mock env）

| 模式 | net/iter | ×FP32 | 梯度保真 | 备注 |
|---|---|---|---|---|
| base(FP32 eager) | 2825ms | 1.00× | — | update 占 ~95% |
| **full(完全体,bf16 update)** | **990ms** | **2.85×** | cos 0.999998 | 默认推荐 |
| full_cc(+ 伴随校准 eager) | 1014ms | 2.79× | 0.999998 | cocalib +23ms/iter |
| full_cc_tap(+ 伴随校准 tap) | 1058ms | 2.67× | 0.999997 | 见 §7 反直觉发现 |

### 6.3 INT8-Linear 是负项的微观证据（`_test_int8linear_net.py`)

纯隔离(只改 linear 是否 INT8):FP16 compile 1.697ms → +LET-INT8 1.806ms（**+0.108ms**）。kernel 分解显示 quant/dequant **大部分已融进 `_int_mm` epilogue**（`triton_tem_fused__int_mm__to_copy_add_clamp_div_elu_mul_round`），但:① INT8 GEMM 在这些小形状上不比 FP16 快（访存受限）；② 融进 epilogue 的量化算术反而给访存受限 kernel 添了活。**融合质量好 ≠ 有收益——收益由算子 roofline 位置决定**：conv 该上 INT8（输入流量大），小 linear 不该上。

### 6.4 MHA 处理（`_test_mha.py`)

P3 手写 vs `nn.MHA`：数值 bit-exact（CosSim 1.000000）；**eager 0.99×（无收益），compile 1.23×（0.277→0.224ms，省 0.05ms）**。价值在"把 inductor 无法融合的 nn.MHA 黑盒变成可融合的纯张量"，但 head_dim=4/K=32 极小 → MHA 本就不是大头，压缩空间有限。

### 6.5 精度消融与训练漂移（`bench_ablation.py` / `bench_dynamic.py`，ame1 真实权重）

- LET vs naive：见 §5-P1（LET 全 5 层 0.99977 > naive 0.99948）。
- 训练漂移 30 iter：LET+在线 EMA 0.99825 > LET 静态 0.99697 > naive 0.99359。
- 动态 s_z：dynamic 0.99979（精度最优）> static 0.99977；漂移退化减半。
- 伴随校准 50 iter（§5-P6）：fixed-1 0.99790（250 步）≈ periodic（5000 步）> frozen 0.99760。

---

## 7. 关键设计决策与踩坑（每条均已修复并固化）

1. **必须 `mode="max-autotune"`**:默认/reduce-overhead 触发 CUDA invalid configuration,freezing 触发 view/stride 错误。
2. **INT8 反量化 int32→fp32→fp16**:int32(~8M)直接 `.half()` 溢出 inf;compile 版因 inductor 自动 fp32 中转"碰巧正常",掩盖问题。
3. **swap 后 INT8 静默失效**:新建 nn.Module 默认 `training=True`,eval 模型上走了 FP 分支;`swap.apply/restore` 均继承源状态;用 `_check_int8.py` grep `_int_mm` kernel 验证。**量化是否生效要用 kernel 级证据,精度"过好"反而是警报。**
4. **量化分支按 `torch.is_grad_enabled()` 而非 `self.training`**:rsl_rl rollout 是 train 态(Gumbel)+ inference_mode,应走 INT8;PPO update 是梯度态,走可导 FP 路径。
5. **纯 FP16 训练 nan**:用 FP32 master + FP16 推理副本(master weights 范式)。
6. **CUDA RNG graph-safe state 单例陷阱**:它在全部 graph 析构后释放重建;若重建发生在 inference_mode capture 内 → inference tensor → 之后梯度态 capture(训练图 cudagraph trees)报 "Inplace update to inference tensor"。解法:`_ensure_graph_rng_state()` 普通上下文空捕获并**永久持有哨兵 graph**。
7. **`dict.get(a) or dict.get(b)`** 对 Tensor 触发布尔歧义,显式判 None。
8. **Triton**:`tl.arange` 须 2 的幂(K pad);int8 dot 须 `acc += tl.dot(A,B,out_dtype=tl.int32)`;CUDA 无 int32 matmul(对拍走 float 中转)。
9. **等价性对拍必须 eval 态**:train 态两路径各自采样 Gumbel,差异是探索噪声(0.984 假象)。
10. **custom-op tap 须 `register_autograd`**(恒等直通)否则编译训练前向 backward 报错;cocalib+tap 时训练前向须 `no-cudagraphs`(cudagraph 不重放 custom-op 副作用)。
11. **反直觉发现:`full_cc_tap`(tap 0 额外前向)比 `full_cc`(eager)慢 45ms**——tap 强制训练前向 no-cudagraphs,损失的 cudagraph 收益超过省下的采集前向。**eager 源端到端更优,默认 eager**;tap 优势要 batch 更大/量化层更多时才兑现。

---

## 8. 环境与远程

- **本地**(开发):RTX 5060 Laptop(SM 12.0),Windows,conda `pytorch_gpu`,PyTorch 2.10+cu130。注:SM 12.0 无传统 INT8 IMMA,**所有 INT8 实验必须在远程**。
- **远程**(实验):RTX 3090 ×2(SM 8.6),Ubuntu,conda `pytorch_gpu`,PyTorch 2.10+cu126,Triton 3.6,tensordict 0.13。SSH 见 `.env`(`tools/ssh_helper.py` 封装),项目路径 `/tmp/glad_quant_test/`。
- **数据**:`AME_Locomotion/pretrained/ame1.pt`(真实收敛权重,精度实验用);`engines/glad_actor_base.onnx`(基础 ONNX)。

---

## 9. 局限与后续

1. **未测**:含 Isaac Lab 物理仿真的端到端加速(本地/远程无 Isaac 环境)与 reward 收敛 A/B——接入清单见 [PLAN §P4](PLAN_compile_quant.md)。Amdahl 预测(网络栈 2.85×):仿真占 30% → 端到端 1.9×。
2. **B=4096 真实协议**未实测(显存所限,验证用 B=2048;结构相同)。
3. **BN batch-stats 语义近似**:推理副本 BN 固定 eval(running stats),唯一不严格等价点,需 reward A/B 确认。
4. **科研延伸(接口已留)**:在线 QAT(`L_PPO + λ·L_LET`)、INT4/混合精度逐层搜索、SmoothQuant/AWQ 等同范式算法(只需换 `make_plan` 的 α/β 来源)。INT4 经 roofline 分析速度无收益(权重访存非瓶颈),W4A8 精度灰区需补 LWC + block 重建;FP8 需 SM 8.9+ 硬件。

---

## 10. 一键接入真实训练（Isaac Lab 运行清单）

1. 拷贝 `qrt/` + `quant/` 到训练机,`sys.path` 加入 `GLAD_Quant`;依赖仅 torch≥2.8 + tensordict(rsl_rl 本有);
2. `train.py` 加两行（§2）；首次运行编译 ~5–8 min（缓存后秒-分钟级），`TORCH_LOGS=recompiles` 监控无反复重编译；
3. B=4096 真实 update 峰值显存 ~15GB，需独占卡；
4. **建议 A/B 三组 reward 曲线**：`base` / `full(int8_linear=False)` / `full` —— 论文实验的最后一块数据。

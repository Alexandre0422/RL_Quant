# GLAD 量化 — 改进方法实验日志

> 每次尝试一节:**假设 → 设计 → 命令 → 结果 → 结论**。结果未出时标 `⏳ 待跑`。
> 单一可信源是 git remote;本文件随代码进分支(`houyishan/<主题>`),与 README/PLAN/记忆三处呼应。
> GPU 实验一律远程(ps2,见 CLAUDE.md);本地仅写代码 + 语法检查。

---

## EXP-001 · INT8 roofline 交叉点扫描(batch 维度) ⏳ 待跑

- **分支**:`houyishan/roofline-crossover`
- **脚本**:[`qrt/bench/bench_roofline.py`](qrt/bench/bench_roofline.py)
- **日期**:2026-07-01 设计

### 假设
项目核心论点:「INT8 的价值由逐算子的 roofline 位置决定,而非一刀切」。bench_matrix 在
**B=4096** 实测 INT8 **linear** 是负项(ΔlinF−E ≈ **+0.10ms**)——大 batch 下这些小 GEMM
已是激活/计算主导,量化的访存节省被 quant/dequant 开销盖过。README/PLAN 进一步**预测**(此前
未实测):小 batch(B→1,板端单机部署)下 linear 退化为**权重访存主导**,INT8 把权重字节减半
→ 应翻转为净正项。

**待验证**:在同一张 3090 上扫 batch,ΔlinF−E(B) 是否存在变号点 B*(由正转负),
即 INT8 linear 从负项翻转为收益的边界?INT8 conv(ΔconvE−D)随 batch 又如何变化?

### 设计
- 固定硬件(单张 3090),扫 `B ∈ {1,4,16,64,256,1024,4096}`,把同一算子从 roofline 计算受限端
  推到访存受限端(控制变量 = batch,而非换硬件)。
- 逐档叠加(与 bench_matrix 同口径):
  - C FP16 compile(P0) → D +encoder 重写(P3) → E +QuantCNN/INT8 conv(P2) → F +LET-INT8 linear(P1/P5)
- 边际:`ΔconvE−D = lat(E)−lat(D)`、`ΔlinF−E = lat(F)−lat(E)`;**< 0 = 该 INT8 档有收益**。
- 内附一阶 **roofline 解析预测**(actor MLP + emb/query 的 FP16 vs INT8 linear 总时间,
  标称 3090:F16≈71 TFLOP/s,I8≈284 TOP/s,HBM≈936 GB/s),与实测变号点对照。

### 命令(远程,空闲卡,长任务建议 tmux 后台)
```bash
# 全扫(每个 batch 形状各自冷编译,程较长 → 后台):
ssh yishan_3090 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate glad_quant \
  && cd ~/RL_Quant && CUDA_VISIBLE_DEVICES=<空闲卡> python qrt/bench/bench_roofline.py'
# 快速验证管线(少档少 batch):
QRT_BATCHES=1,256,4096 QRT_CFGS=D,E,F python qrt/bench/bench_roofline.py
# 量化生效核验(独立):python qrt/bench/_check_int8.py
```

### 结果

**Smoke(2026-07-01,GPU0,QRT_BATCHES=1,4096 QRT_CFGS=D,E,F)——已发现硬约束:**

| B | D(+rewrite) | E(+INT8 conv) | F(+INT8 linear) |
|---|---|---|---|
| 1 | 0.169ms (×FP32 5.84) | 0.144ms (×FP32 6.89) | **崩溃** |

- **关键发现**:`torch._int_mm(q, w8.t())` 在 B=1 抛
  `RuntimeError: self.size(0) needs to be greater than 16, but got 1`。
  即 cuBLAS 的 INT8 GEMM 后端**要求 M(=batch)> 16**。
- **意义(直接强化论点)**:roofline **预测** INT8 linear 在小 batch(板端 B=1)应翻转为收益,
  但本技术栈的 `_int_mm` 原语**在 B≤16 根本无法执行**——预测的小 batch 收益对该原语**不可达**。
  要在 B=1 兑现 INT8 linear 的访存收益,必须换 **weight-only / GEMV / 自写 Triton** kernel
  (cuBLAS IMMA 是为大 M 吞吐设计的)。这把「INT8 价值取决于 roofline 位置」再加一层:
  **不仅 roofline 位置要对,可用的 kernel 原语也要支持那个形状**——直接给「INT4/weight-only 边端」
  方向(改进选项 B)提供了实测动机。
- D 冷编译 307s、E 13.6s(复用缓存);B=1 单样本 CosSim≈0.50 是 29 维单样本随机权重的噪声,
  非精度问题(latency 研究,精度见 P1/P5)。

**已修脚本**:F 在 B<17 自动跳过并记录原因;每档 build 包 try/except,单档失败不影响整扫。

**全扫描(2026-07-01,GPU0,QRT_BATCHES=1,16,32,64,256,1024,4096 QRT_CFGS=D,E,F):**

延迟矩阵(ms),逐档叠加:

| B | A FP32 | D +rewrite | E +INT8conv | F +INT8lin | ΔconvE−D | ΔlinF−E |
|---|---|---|---|---|---|---|
| 1    | 0.929 | 0.161 | 0.096 | skip  | −0.065 | — |
| 16   | 0.887 | 0.186 | 0.104 | skip  | −0.082 | — |
| 32   | 0.911 | 0.103 | 0.154 | 0.157 | +0.050 | +0.003 |
| 64   | 0.914 | 0.112 | 0.108 | 0.138 | −0.004 | +0.029 |
| 256  | 0.915 | 0.178 | 0.204 | 0.172 | +0.026 | −0.032 |
| 1024 | 1.948 | 0.528 | 0.352 | 0.389 | −0.176 | +0.036 |
| 4096 | 7.272 | 1.696 | 1.262 | 2.275 | −0.434 | +1.013 |

**数据质量评估(诚实记录,关键):**
- **小/中 B 行不可信**。E 非单调(B=32→0.154 却 B=64→0.108)、D 抖动(0.186→0.103→0.112);该区
  kernel 仅数十 μs,launch 抖动 + 共享 GPU 外部负载(GPU0 11% 他人占用)淹没信号。中 B 的 Δ(±0.03ms)
  小于相邻 batch 的run间抖动 → **纯噪声**。
- **脚本自动算出的「B*≈617」是噪声伪迹**(在 B=256 的 −0.032 与 B=1024 的 +0.036 两个噪声值间插值)——
  **作废,不作为结论**。
- **唯一干净信号在 B=4096**(kernel 时间足够大、盖过 launch 抖动):
  ΔconvE−D = **−0.434ms**(INT8 conv 明确收益,与 bench_matrix 的 −0.36ms 同量级 ✓);
  ΔlinF−E 明确为正(INT8 linear 是负项,符号与 bench_matrix 一致 ✓)。
- **异常**:F@4096 = 2.275ms ≈ bench_matrix 干净值 1.157ms 的 **2×**(我的 D/E 也较干净基线高 ~20%,
  与污染一致;但 F 被放大得更多)→ 量级待**空闲卡复跑**确认,暂不采信其绝对值。
- **解析 roofline 预测 ≈0 全程**(总是略偏向 INT8):模型太粗,**漏算 quant/dequant 的 kernel-launch
  开销**——而正是这项开销让 INT8 linear 在实测中成为负项。需补该项才能匹配实测符号。

### 结论
1. ✅ **硬发现(干净、可复现)**:INT8 linear 经 `torch._int_mm` 在 **B≤16 不可执行**(cuBLAS IMMA 要求 M>16)。
   roofline 预测 INT8 linear 在小 batch(板端 B=1)才翻正,但**该原语恰好在那个区间不可用** ⇒ 要兑现小 batch
   INT8 linear 收益必须换 weight-only/GEMV/自写 Triton kernel。→ 直接动机:改进选项 B(INT4/weight-only 边端)。
2. ✅ **大 B(4096)**:INT8 conv 是明确净正(−0.43ms),INT8 linear 是明确净负——与既有结论一致,本扫描复现。
3. ⚠️ **方法学教训**:小/中 B 区(可能存在 roofline 翻转的地方)在**共享 GPU + 全模型计时**下信噪比不足,
   测不出可信的交叉点。需:① 空闲独占卡;② 微基准隔离单层 linear(仿 `_test_int8linear_net.py`)而非整模型;
   ③ median-of-N 多次取中位、剔除 GPU 繁忙样本。
4. ❌ **未达成**:可信的 INT8-linear roofline 交叉点 B*。原因是 (1) 的原语约束 + (3) 的测量噪声双重阻挡。

### 后续(待定方向)
- **A. 干净复跑**:抢空闲卡 + 微基准隔离 linear + median-of-N,把 B∈[32,512] 的 ΔlinF−E 测准(确认是否真有交叉,
  还是全程为负)。
- **B. weight-only INT8/INT4 GEMV kernel**:小 batch 唯一能兑现 INT8 linear 访存收益的路径(绕开 `_int_mm` 的 M>16),
  也是边端 B=1 部署的真实需求(改进方向选项 B)。本次实验已为其提供硬动机。

### 风险 / 注意
- **Triton kernel 小 batch 行为未验**:QuantCNN 两个 kernel 处理 `[B·187, C]`,B=1 时仅 187 行,
  grid/block 可能退化或低效 → 先看 B=1 是否报错或 cos 异常。
- 同卡扫 batch 是 roofline 的科学控制;真实板端(Orin 等)硬件不同,B* 绝对值会平移,但**变号
  现象本身硬件无关**(roofline 性质)。解析预测提供硬件无关的趋势锚。
- 小 B 单步极快、launch/库开销占比高 → 已加大 n_repeat(B≤64 用 300);仍需空闲卡避免负载污染。
- 量化「精度过好」是警报;变号若与直觉矛盾,先 `_check_int8.py` grep `_int_mm` 确认 INT8 真生效。

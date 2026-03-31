# Vibesing（高音觉醒）Taxonomy V1.0 — 工程级深度评审报告

> **文档性质**：针对 Vibesing 7 维声学特征标签矩阵的 DSP/DL/端侧工程可行性逐条拆解  
> **评审视角**：物理声学 × 深度学习架构 × CoreML 端侧约束 × MVP 数据策略  
> **日期**：2026-03-30

---

## 目录

1. [物理声学特征陷阱与特征重叠](#1-物理声学特征陷阱与特征重叠)
2. [iOS 端侧实时流式模型架构设计](#2-ios-端侧实时流式模型架构设计)
3. [帧级抖动与实时打分平滑算法](#3-帧级抖动与实时打分平滑算法)
4. [MVP 阶段破局与数据清洗策略](#4-mvp-阶段破局与数据清洗策略)
5. [附录：Taxonomy V1.0 各维度工程可行性速查表](#附录)

---

## 1. 物理声学特征陷阱与特征重叠

### 1.1 大白嗓 / 挤卡声 / 喉位高 — "三重混淆区"的物理本质

这三者在 Mel-Spectrogram 上的确存在高度重叠——它们**都**会导致高频能量异常集中、第一共振峰 (F1) 偏高、整体频谱"亮"且"紧"。但它们的物理发声机制完全不同，可以从以下特征维度拆开：

| 病灶 | 声门层机制 | 声道层机制 | 关键区分特征 |
|------|-----------|-----------|-------------|
| **大白嗓 (Yelling)** | 声带全长振动、厚重接触，声门闭合相(CQ)极高 | 咽腔/口腔完全打开，无任何"管道窄化" | ① **H1-H2 极负**（声门闭合过强导致第一谐波被压制）；② **3-5kHz 歌手共振峰(Singer's Formant)完全缺失**——这是核心标志，因为大白嗓完全没有咽腔窄化来产生这个共振峰；③ **F3/F4 无聚拢**，频谱能量在 1-4kHz 均匀"摊平" |
| **挤卡声 (Squeezed)** | 声带过度内收 + 假声带(室带)参与内收，声门上压极高 | 咽腔被压缩但**非自然窄化**，喉室(Morgagni's ventricle)塌陷 | ① **CPP (Cepstral Peak Prominence) 骤降**——挤卡的核心声学签名是谐波结构被噪声淹没，因为假声带的参与引入大量非周期成分；② **HNR 显著低于大白嗓**（大白嗓虽然粗暴但谐波纯度反而高）；③ **频谱斜率(Spectral Slope)在 1-3kHz 段出现异常"鼓包"**，这是室带振动叠加的特征 |
| **喉位高 (High Larynx)** | 声门层可正常（不一定过度闭合） | **喉头整体上移**导致声道缩短 → 所有共振峰系统性上移 | ① **F1/F2/F3 全部上移 10-20%**，且**彼此间距(formant spacing)保持相对稳定**——这是区别于挤卡的关键；② CPP 和 HNR 可正常（因为声门层没病）；③ **相同元音的共振峰偏离标准值的方向一致且均匀** |

#### 1.1.1 决策边界的数学表达

用特征向量 $\mathbf{x} = [\text{CPP}, \text{HNR}, H_1\!-\!H_2, S_{3\text{kHz}}, \Delta F_1, \Delta F_2, \Delta F_3]$ 构建判别空间：

- **大白嗓 vs 挤卡**：分界面主要由 $\text{CPP}$ 和 $\text{HNR}$ 主导。当 $\text{CPP} < 6\text{dB}$ 且 $\text{HNR} < 12\text{dB}$ 时，判挤卡；当 $\text{CPP} > 8\text{dB}$ 且 $H_1\!-\!H_2 < -6\text{dB}$ 时，判大白嗓。
- **挤卡 vs 喉位高**：分界面主要由共振峰偏移模式 $(\Delta F_1, \Delta F_2, \Delta F_3)$ 的**一致性**主导。定义一致性指标：

$$
\text{FormantShiftConsistency} = 1 - \frac{\sigma(\Delta F_1 / F_1^{\text{ref}},\; \Delta F_2 / F_2^{\text{ref}},\; \Delta F_3 / F_3^{\text{ref}})}{\mu(\Delta F_1 / F_1^{\text{ref}},\; \Delta F_2 / F_2^{\text{ref}},\; \Delta F_3 / F_3^{\text{ref}})}
$$

当 $\text{FormantShiftConsistency} > 0.7$ 且 $\text{CPP}$ 正常时，判喉位高；低一致性+低 CPP 判挤卡。

- **大白嗓 vs 喉位高**：$S_{3\text{kHz}}$（3kHz 附近的歌手共振峰能量）是决定性特征。喉位高仍然可能存在一定 Singer's Formant（如果用户有基本的声乐训练），而大白嗓几乎为零。

### 1.2 必须引入的专业声学特征（频谱之外）

**结论：是的，裸 Mel-Spectrogram / MFCC 绝对不够。** 以下特征是硬性输入需求：

| 特征 | 计算复杂度 | 在端侧的实现方式 | 用途 |
|------|-----------|-----------------|------|
| **CPP (Cepstral Peak Prominence)** | 低（一次 IFFT + argmax） | 在音频前端 DSP pipeline 中逐帧计算，作为标量特征 concat 到 Mel 后 | 挤卡/闭合不实的核心判据 |
| **HNR (Harmonics-to-Noise Ratio)** | 中（需自相关或倒谱法） | 同上，逐帧标量 | 漏气 / 闭合不严 / 挤卡噪声检测 |
| **H1-H2 (第一/第二谐波差)** | 低（F0 追踪后取频谱采样） | 依赖 F0 估计，可复用 CREPE-tiny 的 F0 输出 | 声门接触程度（Open Quotient 代理指标） |
| **F1/F2/F3 共振峰轨迹** | 高（LPC 或 deep formant tracker） | **建议不做独立模块**，而是让 CNN 隐式学习——但在训练标注中引入共振峰信息做 auxiliary loss | 喉位高 / 舌根后压 / 鼻音过重 |
| **F0 + F0 变异率** | 低 | CREPE-tiny 或 FCPE（轻量 F0 估计器） | 破音检测(F0突变)、颤音调制分析 |

**关键工程决策**：共振峰追踪（F1/F2/F3）在端侧做精确 LPC 既贵又不稳定。推荐方案是：

1. **训练时**：使用 Praat/STRAIGHT 离线提取精确共振峰作为辅助监督信号（Auxiliary Target），训练一个联合网络同时预测病灶标签 + 共振峰位置。
2. **推理时**：不单独跑共振峰追踪模块，CNN 的中间层已隐式编码了共振峰信息，直接输出到多任务头。

### 1.3 50-100ms 窗口的特征完备性评估

在 44.1kHz 采样率下：

- **50ms = 2205 samples**：对于 F0 > 100Hz 的成人声完全足够（至少 5 个基频周期）。但对于男低音在 80Hz 以下时，只有 4 个周期，CPP 和 HNR 的估计方差会增大。
- **100ms = 4410 samples**：对所有人声频段都足够。

**建议**：特征提取层用 **93ms (4096 samples @ 44.1kHz)** 窗口、**23ms (1024 samples) 帧移**。这给出约 75% 重叠率，既保证了泛音特征完整性（低 F0 也有足够周期数），又保证了 ~43fps 的帧率供 UI 刷新。

---

## 2. iOS 端侧实时流式模型架构设计

### 2.1 推荐网络拓扑：Causal Depthwise-Separable CNN + Streaming Squeeze-and-Excitation

**核心原则：在 CoreML Neural Engine 上，Conv 吞吐量远高于 Attention。**

不推荐 Conformer/MobileViT 的原因：
- **Self-Attention 的动态 shape**：CoreML 的 Neural Engine 对 Attention 的 K/V 缓存支持有限，需要固定 sequence length，否则 fallback 到 CPU/GPU。
- **内存带宽**：Streaming Conformer 需要维护较大的 KV-cache，在 Neural Engine 上争抢有限的 on-chip SRAM。
- **收益不成正比**：对于单通道语音（非 ASR），Conformer 对比 TCN 的精度提升通常 < 2%，但推理延迟翻倍。

**推荐的具体架构**：

```
┌─────────────────────────────────────────────────────┐
│                 Audio Frontend (DSP)                  │
│  44.1kHz → 16kHz resample → STFT(n_fft=1024,        │
│  hop=256, win=1024) → 80-Mel Filterbank              │
│  + CPP + HNR + H1-H2 (per-frame scalars)             │
│  → Feature: [83 × T] (83 = 80 mel + 3 scalar)       │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           Causal Feature Encoder (4 blocks)           │
│                                                       │
│  Each block:                                          │
│    CausalDepthwiseConv1D(kernel=7, dilation=2^i)     │
│    → BatchNorm → SiLU                                 │
│    → PointwiseConv1D(expand=4x) → SiLU               │
│    → PointwiseConv1D(squeeze) → SE-Block              │
│    → Residual Connection                              │
│                                                       │
│  Receptive field: 7 × (1+2+4+8) = 105 frames         │
│  @ 16ms/frame = ~1.68s temporal context               │
│  Causal padding → zero future dependency              │
│  Channel progression: 83→128→128→256→256              │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           Multi-Task Prediction Heads                 │
│                                                       │
│  Head A: 基础机能分类 (Softmax, 4 classes)             │
│    Conv1D(256→4) → Softmax                            │
│                                                       │
│  Head B: 混声比例回归 (Sigmoid, continuous 0-1)        │
│    Conv1D(256→1) → Sigmoid                            │
│                                                       │
│  Head C: 核心病灶多标签 (Sigmoid × 5)                  │
│    Conv1D(256→5) → Sigmoid (per-label)                │
│                                                       │
│  Head D: 共鸣位置分类 (Softmax, 4 classes)             │
│    Conv1D(256→4) → Softmax                            │
│                                                       │
│  Head E: 时序事件检测 (Sigmoid × 4)                    │
│    Conv1D(256→4) → Sigmoid                            │
│                                                       │
│  [后期追加] Head F: 隐蔽代偿 (Sigmoid × 2)            │
│  [后期追加] Head G: 训练态评估 (task-specific)         │
└─────────────────────────────────────────────────────┘
```

**参数量估算**：~800K 参数（对比 CREPE-tiny 的 21M，这非常轻量）。在 A15+ Neural Engine 上，单帧推理约 **0.3-0.5ms**。

### 2.2 端侧重采样决策

**强烈建议将 44.1kHz 降采样到 16kHz**：

1. 人声有效信息在 8kHz 以下（奈奎斯特定理 → 16kHz 足够）。
2. 80-Mel Filterbank 在 16kHz 上的分辨率足以覆盖所有相关声学特征。
3. STFT 计算量减少 $44100/16000 \approx 2.76\times$。
4. 模型输入 tensor 缩小，Neural Engine cache 命中率显著提升。

### 2.3 Buffer / Overlap / Frame 设计参数

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| 采样率 | **16kHz** | 覆盖人声全频段，计算量最优 |
| STFT window | **1024 samples (64ms)** | ≥5 个基频周期 @ 80Hz |
| STFT hop | **256 samples (16ms)** | 帧分辨率 ~62.5fps，足够捕捉瞬态 |
| 重叠率 | **75% (= 1 - 256/1024)** | 频谱平滑，泛音特征完整 |
| 模型输入块 | **32 帧 = 512ms** | 每次推理消耗 32 帧；配合 causal 设计可每 1 帧(16ms)触发一次推理 |
| Audio ring buffer | **2048 samples (128ms)** | 每 16ms 喂 256 新 sample，维持 1024 sample 的 STFT 窗 |
| 端到端延迟预算 | **≤ 48ms** | Audio buffer (16ms) + DSP (2ms) + Model (~0.5ms) + UI render (16ms @60fps) + margin |

### 2.4 CoreML 特化优化

```
关键 CoreML 编译策略：
1. 使用 coremltools 的 ct.convert() 时指定 compute_units=ct.ComputeUnit.ALL
   让 CoreML runtime 自动调度 NE/GPU/CPU
2. 所有 Conv1D 使用 groups=channels (DepthwiseSeparable) 确保 NE 加速
3. BatchNorm 在导出时 fuse 进 Conv weights（减少 NE 指令数）
4. 输入 shape 固定为 [1, 83, 32]（batch=1, features=83, frames=32）
   避免 dynamic shape fallback
5. 使用 Float16 精度（NE 原生支持，精度损失可忽略）
```

### 2.5 iOS 端音频采集 Pipeline

```
AVAudioEngine (44.1kHz, mono)
    │
    ▼ installTap(bufferSize: 1024, ...)
    │
    ▼ vDSP.Downsample → 16kHz ring buffer
    │
    ▼ Every 256 new samples (16ms):
    │   ├── vDSP.FFT → Mel Filterbank → [80-dim]
    │   ├── Cepstrum → CPP → [1-dim]
    │   ├── Autocorrelation → HNR → [1-dim]
    │   └── F0-indexed spectrum → H1-H2 → [1-dim]
    │   → Concat → [83-dim feature frame]
    │   → Push to feature ring buffer (32 frames)
    │
    ▼ CoreML predict(feature_buffer) → Multi-head outputs
    │
    ▼ Post-processing (smoothing, scoring) → UI update @ 60fps
```

---

## 3. 帧级抖动与实时打分平滑算法

### 3.1 分层平滑策略

帧级输出的抖动有两个层次，需要分别处理：

#### 层次 1：标签概率平滑（信号级）

**推荐：一阶 IIR 低通滤波（即指数移动平均 EMA），但对不同标签使用不同时间常数。**

$$
\hat{p}_t^{(k)} = \alpha_k \cdot p_t^{(k)} + (1 - \alpha_k) \cdot \hat{p}_{t-1}^{(k)}
$$

其中 $p_t^{(k)}$ 是第 $k$ 个标签在第 $t$ 帧的原始模型输出，$\hat{p}_t^{(k)}$ 是滤波后值。

**关键洞察：不同标签需要不同的 $\alpha$ 值：**

| 标签类别 | $\alpha$ (越大越灵敏) | 时间常数 $\tau$ | 理由 |
|----------|---------------------|-----------------|------|
| 破音/换声断层 | **0.8 - 0.9** | ~30ms | 瞬态事件，必须快速响应 |
| 挤卡/闭合不严 | **0.3 - 0.5** | ~100ms | 持续性病灶，不应对单帧脉冲响应 |
| 混声比例 | **0.15 - 0.25** | ~200ms | 连续渐变指标，需平滑 |
| 基础机能分类 | **0.1 - 0.2** | ~250ms | 通常在乐句内稳定，不应快速跳变 |

**为什么不用卡尔曼滤波？**  
Kalman Filter 在这里是过度设计。它假设状态转移满足线性高斯模型，但声乐标签的时序动态远非线性高斯。EMA 的简单性和可调节性在端侧更实用。如果后续发现 EMA 不够，可以升级到 **粒子滤波** 或直接在网络末端加 **CausalConv1D + Softmax 时序头**（在训练阶段解决，而非后处理）。

#### 层次 2：状态跳变过滤（语义级）

在 EMA 之上，加一层 **最小持续时间约束（Minimum Duration Constraint）** 状态机：

```python
class StateFilter:
    """
    防止标签在极短时间内反复跳变。
    例如：混声 → 挤卡 → 混声 这样的 <200ms 跳变会被过滤。
    """
    def __init__(self, min_hold_frames: int = 12):  # 12 frames @ 16ms = 192ms
        self.current_state = None
        self.candidate_state = None
        self.candidate_count = 0
        self.min_hold = min_hold_frames

    def update(self, new_state: str) -> str:
        if new_state == self.current_state:
            self.candidate_state = None
            self.candidate_count = 0
            return self.current_state
        
        if new_state == self.candidate_state:
            self.candidate_count += 1
            if self.candidate_count >= self.min_hold:
                self.current_state = new_state
                self.candidate_state = None
                self.candidate_count = 0
                return self.current_state
        else:
            self.candidate_state = new_state
            self.candidate_count = 1
        
        return self.current_state  # 未确认切换前维持原状态
```

### 3.2 实时动态打分函数：VocalHealthScore

设计一个兼顾"瞬时惩罚"和"持续奖励"的动态打分系统：

#### 3.2.1 核心数学框架

定义**帧级原始健康度** $h_t \in [0, 1]$：

$$
h_t = w_{\text{mix}} \cdot Q_{\text{mix}}(t) + w_{\text{fault}} \cdot (1 - P_{\text{fault}}(t)) + w_{\text{stable}} \cdot S_{\text{stable}}(t)
$$

其中：
- $Q_{\text{mix}}(t)$：混声质量分（0=全胸声白嗓，1=完美平衡混声），来自 Head B 的 sigmoid 输出经 EMA 平滑
- $P_{\text{fault}}(t)$：综合病灶概率，$P_{\text{fault}} = \max(p_{\text{squz}}, p_{\text{hlarynx}}, p_{\text{leak}}, p_{\text{yell}})$，取最严重病灶的概率
- $S_{\text{stable}}(t)$：短期稳定性指标，$S_{\text{stable}} = 1 - \text{std}(h_{t-N:t}) / 0.3$（过去 N 帧的标准差归一化）
- 权重建议：$w_{\text{mix}} = 0.5, \; w_{\text{fault}} = 0.35, \; w_{\text{stable}} = 0.15$

#### 3.2.2 瞬时惩罚机制（Instant Penalty）

当检测到严重瞬态事件（破音、挤卡爆发）时，$h_t$ 的 EMA 平滑会导致分数下降太慢。引入**惩罚脉冲**：

$$
\text{Score}_t = \begin{cases}
\text{EMA}(h_t, \alpha_{\text{normal}}) & \text{if no critical event} \\
\text{EMA}(h_t, \alpha_{\text{normal}}) - \Delta_{\text{penalty}} \cdot e^{-\lambda(t - t_{\text{event}})} & \text{if critical event at } t_{\text{event}}
\end{cases}
$$

- $\Delta_{\text{penalty}}$：惩罚幅度，破音 = 30分，挤卡爆发 = 15分
- $\lambda$：衰减速率，建议 $\lambda = 3.0$（即惩罚在 ~1s 内衰减到 $e^{-3} \approx 5\%$）

#### 3.2.3 持续奖励机制（Sustained Reward）

当用户连续 N 秒维持良好混声（$Q_{\text{mix}} > 0.6$ 且 $P_{\text{fault}} < 0.2$）时，给予加速奖励：

```python
def compute_sustained_bonus(streak_seconds: float) -> float:
    """
    连续良好发声的奖励函数。
    前 2 秒无奖励（避免闪烁），2-5 秒线性增长，5 秒后饱和。
    """
    if streak_seconds < 2.0:
        return 0.0
    elif streak_seconds < 5.0:
        return (streak_seconds - 2.0) / 3.0 * 10.0  # 最高 +10 分
    else:
        return 10.0
```

#### 3.2.4 最终分数映射

```python
def frame_to_display_score(raw_score: float, penalty: float, bonus: float) -> int:
    """
    将原始分数映射为用户友好的 0-100 分。
    使用 S 型映射避免分数聚集在某个窄区间。
    """
    base = raw_score * 100.0  # [0, 100]
    adjusted = base - penalty + bonus
    clamped = max(0.0, min(100.0, adjusted))
    
    # 非线性映射：让中间区间有更大的区分度
    # 使用 logistic 映射将 [20, 80] 区间展开
    normalized = clamped / 100.0
    display = 100.0 / (1.0 + math.exp(-10.0 * (normalized - 0.5)))
    
    return int(round(display))
```

### 3.3 UI 绑定策略

| UI 元素 | 数据源 | 刷新率 | 平滑策略 |
|---------|--------|--------|---------|
| 胸/头声比例进度条 | Head B (sigmoid) | 60fps | EMA α=0.2, 插值动画 |
| 发声健康度数字 | VocalHealthScore | 10fps | 上述完整 pipeline + 数字跳动阻尼 |
| 病灶红灯警告 | Head C (argmax) | 事件驱动 | StateFilter(min_hold=8帧) |
| 实时频谱/声纹可视化 | Mel-Spectrogram | 30fps | 原始值，无需平滑 |

---

## 4. MVP 阶段破局与数据清洗策略

### 4.1 MVP 绝对优先死磕的 2 个维度

**答案：Dim 2（混声比例）+ Dim 3（核心病灶，聚焦"挤卡"单一标签）。**

理由从产品—技术—数据三角度锁死：

#### 产品角度
- "高音觉醒"的核心体验是**让用户看到自己的混声比例**在实时变动——这是竞品完全没有的维度。仅凭这一个功能就足以让声乐老师和学生"哇"出来。
- "挤卡"是高音路上第一大杀手。90% 的业余歌手在 E4 以上都会挤卡。如果 App 能实时告诉用户"你挤了"，立刻就有不可替代的价值。

#### 技术角度
- **混声比例**是回归任务，对标注噪音的容忍度天然高于分类任务（连续值 ±0.1 的误差影响远小于二分类标签翻转）。
- **挤卡检测**有明确的物理信号锚点（CPP 骤降 + HNR 下降），不依赖主观标注就能建立弱监督信号。

#### 数据角度
- 混声比例可以通过**合成数据增强**：将纯胸声和纯头声录音按不同比例叠加（Mixup），自动生成带标签的训练数据。这是其他维度做不到的。
- 挤卡数据可以通过**物理规则预标注**：先用 CPP + HNR 阈值自动打标，再用少量人工标注修正。

### 4.2 其余维度的优先级排序

| 优先级 | 维度 | 进入时机 | 理由 |
|--------|------|---------|------|
| **P0 (MVP)** | Dim 2: 混声比例 | Day 1 | 核心差异化体验 |
| **P0 (MVP)** | Dim 3: 挤卡检测 | Day 1 | 最高频病灶，物理信号清晰 |
| P1 | Dim 1: 基础机能 | V1.1 | 依赖混声比例模型作为 backbone 特征 |
| P1 | Dim 3: 漏气/闭合不严 | V1.1 | HNR 专项检测，工程简单 |
| P2 | Dim 5: 破音检测 | V1.2 | F0 跳变检测较独立，可用规则引擎 |
| P2 | Dim 4: 共鸣位置 | V1.2 | 需要更多标注数据 |
| P3 | Dim 3: 大白嗓/喉位高 | V1.3 | 依赖三分类解混淆（§1.1） |
| P3 | Dim 6: 隐蔽代偿 | V1.3 | 需要精确共振峰或专门子模型 |
| P4 | Dim 7: 训练态 | V2.0 | 需要专项训练数据集 |

### 4.3 标注噪音处理策略

#### 4.3.1 混声比例（回归任务）

**策略：Multi-Annotator Regression + Evidential Deep Learning**

1. **多人标注取分布**：让 3-5 位声乐老师独立标注同一段音频的混声比例（0-1）。不取平均值，而是保留**标注分布**。
   
2. **训练时使用 Distribution Loss**：

$$
\mathcal{L}_{\text{mix}} = \text{KL}\left(\mathcal{N}(\hat{\mu}_t, \hat{\sigma}_t^2) \;\|\; \mathcal{N}(\mu_{\text{anno}}, \sigma_{\text{anno}}^2)\right)
$$

让模型同时输出预测均值 $\hat{\mu}$ 和预测不确定性 $\hat{\sigma}$。当标注者分歧大时（$\sigma_{\text{anno}}$ 大），模型自动降低该样本的梯度贡献。

3. **物理一致性正则化**（关键创新）：

$$
\mathcal{L}_{\text{phys}} = \max\left(0, \; \hat{Q}_{\text{mix}}(t) - f_{\text{upper}}(\text{F0}_t)\right) + \max\left(0, \; f_{\text{lower}}(\text{F0}_t) - \hat{Q}_{\text{mix}}(t)\right)
$$

其中 $f_{\text{upper/lower}}(\text{F0})$ 是基于发声物理学的混声比例上下界函数——例如在 C3 (130Hz) 附近，胸声比例不可能低于 0.7；在 C6 (1046Hz)，头声比例不可能低于 0.9。这个物理约束可以在**完全无标注**的情况下提供监督信号。

#### 4.3.2 挤卡检测（二分类任务）

**策略：Noise-Aware Training + Co-Teaching + 物理规则预滤波**

1. **Phase 1 — 物理规则预标注**：
   ```python
   def rule_based_squeeze_label(cpp, hnr, h1_h2, f0):
       """基于物理阈值的弱标签生成器"""
       if f0 < 200:  # 低音区不太可能挤卡
           return 0.1  # soft label, low confidence
       if cpp < 5.0 and hnr < 10.0 and h1_h2 > -2.0:
           return 0.9  # high confidence squeeze
       if cpp > 10.0 and hnr > 20.0:
           return 0.05  # high confidence non-squeeze
       return 0.5  # uncertain zone → let human annotate
   ```
   
   先用此规则扫描全部数据，只把 0.5（不确定区）的样本交给人工标注，大幅降低标注成本。

2. **Phase 2 — Label Smoothing + Mixup**：
   - Label Smoothing：$y_{\text{smooth}} = (1 - \epsilon) \cdot y + \epsilon / 2$，$\epsilon = 0.1$
   - 对挤卡/非挤卡的音频做 Mixup（时域混合 + 标签线性插值），生成大量"轻微挤卡"的渐变样本

3. **Phase 3 — Co-Teaching**：
   训练两个独立网络 A 和 B，每个 epoch 中，A 选取自己 loss 最小的 80% 样本喂给 B 训练，反之亦然。这样可以自动过滤标注噪音最大的样本。

4. **Phase 4 — 温度校准（Temperature Scaling）**：
   训练完成后，在干净的验证集上做温度校准 $T$：
   $$p_{\text{cal}} = \sigma(z / T)$$
   确保模型输出概率对应真实的频率（calibrated probability），避免对 UI 端报告误导性的高置信度。

### 4.4 数据采集的实操建议

| 数据来源 | 数量级 | 用途 | 优先级 |
|---------|--------|------|--------|
| **自录数据**：团队成员+3-5 位声乐老师，每人录 50 段不同音高的长音 | 500-1000 段 | 核心有标签训练集 | P0 |
| **VocalSet 数据集**（公开学术数据集，含不同技巧） | ~3500 段 | 基础机能分类预训练 | P0 |
| **Mixup 合成数据**：胸声+头声叠加 | 10000+ 段 | 混声比例的半监督信号 | P0 |
| **YouTube 翻唱视频提取**（注意版权） | 5000+ 段 | 带噪标签 + 预训练特征空间 | P1 |
| **EGG (电声门图) 配对数据**（如有条件） | 200-500 段 | 精确声门闭合度标注 → 挤卡 Ground Truth | P1 |

---

## 附录

### Taxonomy V1.0 各维度工程可行性速查表

| 维度 | 标签 | 任务类型 | 核心 DSP 特征 | 模型可行性 | 端侧实时性 | MVP 纳入? |
|------|------|---------|--------------|-----------|-----------|----------|
| D1 基础机能 | 胸声/头声/假声/咽音 | 4-class | Mel + H1-H2 + 频谱斜率 | ★★★★☆ | ★★★★★ | ❌ (V1.1) |
| D2 混声比例 | 连续值 0-1 | 回归 | Mel + Singer's Formant 3kHz 能量 | ★★★★☆ | ★★★★★ | ✅ P0 |
| D3 挤卡 | 二分类 | 分类 | CPP + HNR + Mel | ★★★★☆ | ★★★★★ | ✅ P0 |
| D3 喉位高 | 二分类 | 分类 | F1/F2/F3 偏移一致性 | ★★★☆☆ | ★★★☆☆ | ❌ (V1.3) |
| D3 大白嗓 | 二分类 | 分类 | H1-H2 + Singer's Formant | ★★★☆☆ | ★★★★☆ | ❌ (V1.3) |
| D3 漏气 | 二分类 | 分类 | HNR + CPP | ★★★★★ | ★★★★★ | ❌ (V1.1) |
| D3 气息不稳 | 二分类 | 分类 | Amplitude 变异系数 | ★★★★☆ | ★★★★★ | ❌ (V1.2) |
| D4 面罩共鸣 | 分类 | 分类 | F3/F4/F5 能量比 | ★★★☆☆ | ★★★★☆ | ❌ (V1.2) |
| D4 鼻音过重 | 二分类 | 分类 | 鼻音零点(anti-formant) | ★★☆☆☆ | ★★★☆☆ | ❌ (V1.3) |
| D5 破音 | 事件检 | 检测 | F0 帧间差 > 阈值 | ★★★★★ | ★★★★★ | ❌ (V1.2, 可用规则) |
| D5 颤音检测 | 回归 | 检测 | F0 调制频率/深度 (AM/FM) | ★★★★☆ | ★★★★★ | ❌ (V1.3) |
| D6 舌根后压 | 二分类 | 分类 | F2 异常降低 | ★★☆☆☆ | ★★☆☆☆ | ❌ (V1.3) |
| D6 下巴僵硬 | 二分类 | 分类 | 元辅音过渡时长 | ★★☆☆☆ | ★★☆☆☆ | ❌ (V2.0) |
| D7 唇颤音/哼鸣 | 分类 | 分类 | 频谱模式匹配 | ★★★★☆ | ★★★★★ | ❌ (V2.0) |

### Taxonomy 设计修订建议

1. **D1 "咽音" 应从基础机能中移除**：咽音(Twang/Pharyngeal)在严格声学定义上不是独立的发声类型（register），而是一种共鸣增强策略（Resonance Strategy）。建议将其移入 D4（共鸣与位置），定义为"咽腔窄化共鸣"，用 Singer's Formant ~3kHz 的相对强度作为检测指标。

2. **D2 "艺术性气声" 应独立为 D2 的子标签**：艺术性气声（Breathy/Airy Voice）本质上是"有意为之的闭合不全"，在频谱上和 D3 的"漏气"几乎完全重合。区分二者需要**上下文信息**（是否在乐句末尾、音量是否主动降低）。建议在模型层用"漏气概率 + 音量动态 + 乐句位置"三信号做后处理判别，而非在声学模型中硬编码区分。

3. **D5 "气泡音(Vocal Fry)" 可用规则引擎检测**：气泡音有极端明确的物理签名——F0 骤降到 40-80Hz + 极不规则脉冲间隔。这不需要深度学习，用基于 F0 + Jitter 的规则完全能搞定，节省模型容量。

---

---

## 5. 三方交叉评审补充：隐藏工程陷阱（Cross-Model Consensus Addendum）

> 以下内容综合 Claude 4.6 / ChatGPT / Gemini 3.1 Pro 三方独立评审的交叉验证结论。三方在核心路径上高度收敛，以下重点补充**仅 ChatGPT 明确指出的隐蔽工程陷阱**。

### 5.1 定理级结论（三方一致，不再讨论）

| 结论 | 状态 |
|------|------|
| MVP 只做 D2（混声比例）+ D3（挤卡） | **锁定** |
| 裸 Mel 必死，CPP/HNR/H1-H2 是硬性输入 | **锁定** |
| TCN (Causal Depthwise-Separable CNN) 是 iOS 端当前 Pareto 最优 | **锁定** |
| 16kHz 是唯一合理采样率 | **锁定** |
| EMA（信号层）+ 状态机（语义层）= 最优平滑组合 | **锁定** |
| 气泡音/破音用规则引擎，不浪费模型容量 | **锁定** |
| 咽音从 D1 降级到 D4（共鸣策略而非发声类型） | **锁定** |

### 5.2 隐蔽陷阱 #1：CPP 在端侧实时场景会"崩"

CPP 虽是挤卡检测的核心锚点，但在 <64ms 窗口 + 非静音室环境下极不稳定：

**问题矩阵**：
- 窗口 < 50ms → 倒谱分辨率不足，假峰频发
- 环境噪声 → 倒谱峰被淹没，CPP 虚高/虚低
- 低 F0 (< 100Hz) → 基频周期不足，CPP 估计方差剧增

**工程级解决方案**：

```python
# 1. 中值滤波取代原始 CPP（消除单帧崩溃）
CPP_stable = np.median(CPP[t-2 : t+3])  # 5帧中值窗

# 2. Voicing Mask：无声/非周期段不采信 CPP
if f0_confidence < 0.6:
    CPP_valid = False  # 丢弃本帧 CPP，保持上一有效值

# 3. 倒谱搜索范围锁定（防止错误峰值）
# 仅在 quefrency = 1/F0_max ~ 1/F0_min 范围内搜索倒谱峰
cepstrum_search_range = (1.0 / F0_MAX_HZ, 1.0 / F0_MIN_HZ)
```

### 5.3 隐蔽陷阱 #2：混声比例是"伪连续变量"

**致命洞察**：混声比例在物理上**不是**线性连续量。0.4 和 0.5 之间的声学差异可能远小于 0.7 和 0.8 之间的差异。直接做回归会导致：

- Loss 不收敛（标注者在中间区域分歧巨大）
- UI 上数值疯狂抖动（因为 0.4~0.6 区间模型毫无区分度）

**修正方案：分类 + Soft Mapping（替代直接回归）**

```python
# 将连续回归替换为 5-class 有序分类
MIX_BINS = ["full_chest", "chest_mix", "balanced", "head_mix", "full_head"]
# 对应中心值：     [0.1,         0.3,        0.5,       0.7,       0.9]

# 模型输出 softmax → 期望值映射为连续 UI 显示
bin_centers = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
probs = F.softmax(logits, dim=-1)  # [B, 5]
mix_ratio = (probs * bin_centers).sum(dim=-1)  # 期望值作为连续输出
```

> **注**：此方案比直接回归稳定 ~3 倍（经验值），因为 softmax 天然产生平滑的概率分布，即使标注噪音也不会导致输出剧烈跳变。

### 5.4 隐蔽陷阱 #3：挤卡检测被音量混淆（Volume Confounding）

模型极容易学到捷径：`大声 ≈ 挤卡`。因为训练数据中挤卡样本往往音量更大。

**三层防御**：

```python
# 防御层 1：输入时 per-utterance energy normalization
mel_normalized = mel - mel.mean(dim=-1, keepdim=True)

# 防御层 2：训练时随机增益增强（打破音量-标签相关性）
gain_db = random.uniform(-12, +6)
audio *= 10 ** (gain_db / 20)

# 防御层 3：将 frame energy 作为显式输入特征（强制 disentangle）
energy = mel.logsumexp(dim=0)  # scalar per frame
features = concat(mel_normalized, cpp, hnr, h1_h2, energy)
# 模型被迫将 energy 信息与挤卡信息解耦
```

### 5.5 隐蔽陷阱 #4：UI 延迟不等于模型延迟

端到端延迟的真实分解：

```
Audio buffer fill:    16ms  (256 samples @ 16kHz)
STFT window:          64ms  (1024 samples, 但只需等最后 256 个新 sample)
DSP feature extract:  ~2ms
CoreML inference:     ~0.5ms
Main thread dispatch: ~1ms
UI render (next frame): 0-16ms (60fps vsync)
─────────────────────────
理论最小延迟:          ~20ms (16 + 2 + 0.5 + 1)
典型感知延迟:          ~35ms (含 vsync 等待)
最坏延迟:              ~52ms (所有环节都踩在 worst case)
```

**关键优化**：不要等完整 STFT window 填满才触发推理。使用**滑动更新**——每进 256 个新 sample，用 ring buffer 中已有的 768 old + 256 new 组成 1024-sample window，立刻算 STFT。

### 5.6 双流融合架构修正（Late Fusion 替代 Early Concat）

三方共识中 ChatGPT 明确指出：简单 concat 物理特征到 Mel 后面，模型大概率忽略低维的 CPP/HNR。

**修正后的模型拓扑**：

```
Stream A (频谱流):                  Stream B (物理流):
  80-Mel → Causal CNN (4 blocks)     [CPP, HNR, H1-H2, energy] → MLP(4→64→128)
  → 256-dim                          → 128-dim
              ↘                    ↙
               Late Fusion: concat → 384-dim
                        ↓
              Multi-Task Prediction Heads
```

这确保物理特征有独立的梯度通路，不会被 80 维 Mel 的梯度淹没。

---

## 6. MVP 第一周行动指南（The First Red Light）

> **唯一目标：让红灯亮起来。**

三方的终极共识指向同一个工程起点：不是训练模型，不是搭 iOS 工程，而是——

### 先用 Python 让自己"看到"挤卡的物理信号

```
Week 1 Day 1-3:
  ✅ Python 实现 CPP + HNR + F0 逐帧提取
  ✅ 对 1 段"正常长音" + 1 段"挤卡长音"画出 4 条时序曲线
  ✅ 肉眼确认：CPP 和 HNR 在挤卡时是否真的骤降

Week 1 Day 4-5:
  ✅ 写 rule-based 挤卡检测器 (CPP<6 && HNR<10)
  ✅ 在 10 段测试音频上验证准确率
  ✅ 如果 rule-based 已经 >70% 准确 → MVP 可行性确认

Week 2:
  ✅ PyTorch 搭 Dual-Stream Causal TCN
  ✅ 用 rule-based 标签做第一轮伪标签训练
  ✅ 对比 rule-based vs 模型 → 确认模型有增益
```

> 如果连 rule-based 的红灯都点不亮——说明物理特征方案需要修正，此时再训练模型是浪费时间。**先验证物理假设，再上深度学习。**

---

> **文档完毕。** 此评审覆盖了从物理声学特征辨析、端侧流式架构设计到实时打分算法、MVP 数据策略、三方交叉验证隐蔽陷阱到第一周行动指南的全部内容。所有推荐方案均锚定 CoreML Neural Engine 的实际算力边界和 <100ms 端到端延迟约束。

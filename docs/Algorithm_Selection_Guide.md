# Vibesing — 传统算法 vs 深度学习算法：诊断维度技术选型完整指南

> **目标**：对 Vibesing 涉及的每一个诊断维度，详细说明应该用传统 DSP 算法还是深度学习，以及为什么。  
> **原则**：能用规则解决的绝不上模型；模型只用在模式复杂、阈值难定、特征耦合高的场景。  
> **日期**：2026-03-30

---

## 目录

- [第一章：总体技术选型哲学](#第一章总体技术选型哲学)
- [第二章：传统算法详解](#第二章传统算法详解)
- [第三章：深度学习算法详解](#第三章深度学习算法详解)
- [第四章：逐维度技术选型](#第四章逐维度技术选型)
- [第五章：混合架构设计](#第五章混合架构设计)
- [第六章：端侧部署约束下的选型策略](#第六章端侧部署约束下的选型策略)
- [第七章：从 V0.1 到 V2.0 的算法演进路线](#第七章从v01到v20的算法演进路线)

---

## 第一章：总体技术选型哲学

### 1.1 核心决策矩阵

```
                    特征工程难度
                    低 ←───────→ 高
                    │              │
    数据需求  低    │  传统规则     │  传统+手工特征+ML
                    │  (直接阈值)   │  (SVM/XGBoost)
                    │              │
              高    │  ——          │  深度学习
                    │  (不存在)     │  (CNN/RNN)
                    │              │
```

### 1.2 选型原则

| 原则 | 解释 |
|------|------|
| **可解释性优先** | 声乐教学需要告诉用户"为什么"，规则算法天然可解释 |
| **数据量决定上限** | < 100 样本 → 只能规则；100-1000 → ML；> 1000 → DL |
| **延迟决定下限** | 端侧 < 10ms/frame → 规则或轻量 CNN；不能用大模型 |
| **维护成本** | 规则可以一个人维护；DL 需要训练管线、标注系统、版本管理 |
| **渐进式升级** | 先规则跑通，用规则标注数据，然后用数据训练 DL 替换规则 |

### 1.3 Vibesing 的实际约束

| 约束 | 当前状态 |
|------|---------|
| 标注数据量 | **0**（MVP 刚开始，TestFlight 还没上线） |
| 设备算力 | iPhone A14+，Neural Engine 11 TOPS |
| 延迟要求 | < 100ms 端到端 |
| 团队规模 | 极小（1-2人） |
| 结论 | **V0.1-V0.2 必须全部使用传统算法** |

---

## 第二章：传统算法详解

### 2.1 可用的传统 DSP 算法库

| 算法 | 功能 | 复杂度 | 端侧延迟 | 实现 |
|------|------|--------|---------|------|
| **FFT / STFT** | 时频分析基础 | O(N log N) | < 0.1ms | vDSP (iOS) / numpy (Python) |
| **Mel Filterbank** | 频谱→听觉尺度 | 矩阵乘法 | < 0.1ms | 手写 (iOS) / librosa (Python) |
| **自相关法 (ACF)** | F0 估计 | O(N²) | < 0.5ms | 手写 |
| **RAPT / YIN** | 鲁棒 F0 估计 | O(N²) | < 1ms | Praat (Python) / 手写 (iOS) |
| **HNR (谐噪比)** | 声质评估 | O(N²) | < 0.5ms | 自相关法 |
| **CPP (倒谱峰突出度)** | 声质评估 | O(N log N) | < 0.3ms | 倒频谱分析 |
| **LPC (线性预测编码)** | 共振峰提取 | O(P²N) | < 1ms | Levinson-Durbin |
| **H1-H2** | 声门状态估计 | 需要精确 F0 | < 0.2ms | 谐波提取 |
| **Jitter / Shimmer** | 声质微扰 | 需要 F0 | < 0.1ms | 逐周期分析 |
| **Zero Crossing Rate** | 浊/清音检测 | O(N) | < 0.01ms | 简单计数 |
| **RMS Energy** | 能量估计 | O(N) | < 0.01ms | 均方根 |
| **Spectral Centroid** | 频谱重心 | O(N) | < 0.01ms | 加权平均 |
| **Spectral Slope** | 频谱倾斜度 | O(N) | < 0.01ms | 线性回归 |
| **Spectral Kurtosis** | 频谱尖峰度 | O(N) | < 0.01ms | 四阶矩 |

### 2.2 传统规则引擎架构

```
raw_audio (16kHz, 1024 samples)
        │
        ↓ FFT
  frequency_spectrum (513 bins)
        │
        ├──→ Mel Filterbank → mel_spectrum (80 bands)
        │
        ├──→ Autocorrelation → F0 (Hz) + HNR (dB)
        │
        ├──→ RMS → Energy (dB)
        │
        ├──→ Cepstrum → CPP (dB)
        │
        ├──→ Harmonic Extraction → H1, H2 → H1-H2 (dB)
        │
        ├──→ LPC → F1, F2, F3 (Hz)
        │
        └──→ Spectral Moments → Centroid, Slope, Kurtosis
                    │
                    ↓
        ┌──────────────────────┐
        │   Rule Engine         │
        │                       │
        │   if HNR < 12         │
        │     → squeeze = high  │
        │   if H1-H2 > 0       │
        │     → breathy = true  │
        │   if F1_shift > 0.15  │
        │     → high_larynx     │
        │   ...                 │
        └──────────────────────┘
                    │
                    ↓
        ┌──────────────────────┐
        │   Post-processing    │
        │   EMA + StateFilter  │
        └──────────────────────┘
                    │
                    ↓
              UI Feedback
```

### 2.3 传统算法的优势与局限

| ✅ 优势 | ❌ 局限 |
|--------|--------|
| 零数据即可启动 | 阈值需要手动调，对个体差异鲁棒性差 |
| 可解释（"HNR = 8dB，低于 12dB 阈值 → 挤卡"） | 特征之间的交互模式难以捕捉 |
| 延迟极低（< 1ms/frame） | 频谱微妙模式（如假声带次谐波）无法用规则描述 |
| 易于调试（每个特征都有物理含义） | 随维度增多，规则维护成指数增长 |
| 不需要 GPU/NPU | 精度上限有限（~75-85%） |

---

## 第三章：深度学习算法详解

### 3.1 可选的 DL 架构

| 架构 | 适用性 | 参数量 | 端侧延迟 (A14) | 选用理由 |
|------|--------|--------|----------------|---------|
| **Causal 1D CNN** | ⭐⭐⭐⭐⭐ | 50-300K | 0.5-2ms | 因果卷积，低延迟，适合实时 |
| **DWSep CNN (TCN)** | ⭐⭐⭐⭐⭐ | 100-300K | 0.5-2ms | 深度可分离卷积，更低参数量 |
| **LSTM / GRU** | ⭐⭐⭐ | 200K-1M | 2-5ms | 长程依赖好，但序列化不能并行 |
| **Transformer** | ⭐⭐ | > 1M | > 10ms | 注意力机制强但参数多、延迟高 |
| **Conformer** | ⭐ | > 2M | > 15ms | 精度最高但完全不适合端侧实时 |
| **MLP-Mixer** | ⭐⭐⭐ | 100-500K | 1-3ms | 无注意力的混合器，值得探索 |

**Vibesing 选型**：**Causal DWSep CNN (TCN)**，理由：
1. 因果（causal）：只用当前帧和过去帧，不看未来，适合实时
2. 深度可分离（depthwise-separable）：参数量只有标准卷积的 1/8
3. 感受野可控：堆叠膨胀卷积（dilated conv）扩大感受野而不增加延迟

### 3.2 模型架构详细设计

```
Input Tensor: [batch, 82, 32]
  82 = 80 Mel bands + HNR + Energy
  32 = 32 frames × 16ms = 512ms context

Layer 1: DWSep Conv1D (82→128, kernel=3, dilation=1)
  + BatchNorm + ReLU + Dropout(0.1)
  感受野: 3 frames = 48ms

Layer 2: DWSep Conv1D (128→128, kernel=3, dilation=2)
  + BatchNorm + ReLU + Dropout(0.1)
  感受野: 7 frames = 112ms

Layer 3: DWSep Conv1D (128→128, kernel=3, dilation=4)
  + BatchNorm + ReLU + Dropout(0.1)
  感受野: 15 frames = 240ms

Layer 4: DWSep Conv1D (128→64, kernel=3, dilation=8)
  + BatchNorm + ReLU + Dropout(0.1)
  感受野: 31 frames = 496ms ← 覆盖完整发声周期

Head: Conv1D (64→N_outputs, kernel=1)
  → squeeze_prob: sigmoid     # 挤卡概率
  → mix_class: softmax(5)     # 混声5分类 (V0.2+)
  → vibrato_params: linear(3) # 颤音频率/深度/规整度 (V1.0+)

Total params: ~280K
CoreML Float16 size: ~560KB
A14 Neural Engine latency: ~1.5ms / frame
```

### 3.3 DWSep Conv1D 详解

```python
class DWSepConv1D(nn.Module):
    """
    深度可分离卷积 = Depthwise Conv + Pointwise Conv
    参数量对比：
      标准 Conv(128→128, k=3): 128×128×3 = 49,152
      DWSep Conv(128→128, k=3): 128×3 + 128×128 = 384 + 16,384 = 16,768
      节省: 66%
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        # Step 1: 每个通道独立卷积（不混合通道信息）
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=(kernel_size - 1) * dilation,  # causal padding
            dilation=dilation,
            groups=in_ch  # ← 关键：groups=in_channels
        )
        # Step 2: 1×1 卷积混合通道信息
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = x[:, :, :-self.depthwise.padding[0]]  # causal: 去掉未来帧
        x = self.pointwise(x)
        return x
```

### 3.4 多任务输出头设计

```
               ┌──────────────┐
               │  Shared CNN   │
               │  Backbone     │
               │  (Layer 1-4)  │
               └──────┬───────┘
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Squeeze  │ │ Mix Voice│ │ Vibrato  │
    │ Head     │ │ Head     │ │ Head     │
    │ sigmoid  │ │ softmax5 │ │ linear3  │
    └──────────┘ └──────────┘ └──────────┘
    
    V0.2: 只有 Squeeze Head
    V0.3: + Mix Voice Head
    V1.0: + Vibrato Head + 更多
```

### 3.5 DL 的优势与局限

| ✅ 优势 | ❌ 局限 |
|--------|--------|
| 自动学习特征交互 | 需要标注数据（> 1000 clips） |
| 精度上限高（90-95%+） | 黑箱，难以解释"为什么判定为挤卡" |
| 一个模型搞定多个维度 | 需要训练管线维护 |
| 对个体差异更鲁棒 | 端侧部署需要模型优化（量化、剪枝） |
| 可以不断迭代提升 | 有过拟合风险（数据少时） |

---

## 第四章：逐维度技术选型

### 4.1 挤卡检测 (Squeeze Detection)

| 阶段 | 方法 | 核心逻辑 | 精度预期 | 数据需求 |
|------|------|---------|---------|---------|
| **V0.1** | **传统规则** | HNR < 12dB → sigmoid → EMA → 状态机 | 65-75% | 0 |
| V0.2 | **DL (CNN)** | Mel[80×32] + HNR + Energy → CNN → sigmoid | 80-90% | 500+ clips |
| V1.0 | **DL (多任务)** | + 多头输出，与其他维度联合训练 | 90-95% | 2000+ clips |

**详细算法路线**：

```
V0.1 传统算法：
────────────────
输入：F0, HNR, Energy (3个标量/帧)
算法：
  1. 门控：if energy < -40dB OR F0 < 50Hz → prob = 0
  2. 核心判别：prob = sigmoid(0.5 × (12 - HNR))
     - HNR = 12dB → prob = 0.5
     - HNR = 6dB  → prob = 0.95
     - HNR = 18dB → prob = 0.05
  3. 音高补偿：if F0 > 220Hz → prob × (1.0 + 0.3 × min((F0-220)/300, 1))
     [高音区更容易误挤卡，加权提高灵敏度]
  4. 平滑：ema_prob = α × prob + (1-α) × ema_prev  (α=0.4)
  5. 状态机：
     - 连续 6 帧 ema_prob > 0.55 → 触发红灯
     - 连续 10 帧 ema_prob < 0.30 → 恢复绿灯

优势：
  - 即刻可用，不需要数据
  - 每个决策都有明确的物理含义
  
局限：
  - HNR 单一特征容易误报（漏气也会导致 HNR 低）
  - 阈值固定，不适应不同用户的嗓音差异
  - 无法检测"轻度挤卡"（HNR 只是略低但 Mel 频谱已经有异常）

V0.2 DL 替换：
────────────────
输入：Mel[80×32] + [HNR, Energy] = [82×32]
模型：4层 Causal DWSep CNN (~280K params)
输出：squeeze_prob (sigmoid)
训练数据来源：
  - V0.1 TestFlight 用户的反馈（"准/不准"）
  - 用规则引擎的输出作为弱标签
  - 声乐教师的人工标注（金标准）

新增检测能力：
  - 从 Mel 频谱学习 1-3kHz 鼓包模式
  - 学习谐波结构的完整度（不只是 HNR 一个数字）
  - 对不同音高、不同元音的自动适应
```

---

### 4.2 漏气检测 (Breathiness Detection)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V0.3** | **传统规则** | H1-H2 + HNR 联合判别 | 60-70% |
| V1.0 | **DL** | CNN 同模型多任务头 | 85-90% |

**详细算法路线**：

```
V0.3 传统算法：
────────────────
关键特征：H1-H2（第一谐波与第二谐波的幅度差）

提取 H1-H2：
  1. 从 FFT 频谱中找到 F0 对应的频率 bin → H1 = amplitude_at_F0
  2. 找到 2×F0 对应的频率 bin → H2 = amplitude_at_2F0
  3. H1_H2_dB = 20 × log10(H1/H2)

判别规则：
  if HNR < 12 AND H1_H2 > 0dB → 漏气（声门闭合不全）
  if HNR < 12 AND H1_H2 < -6dB → 挤卡（声门闭合过紧）
  
  物理解释：
  - H1-H2 > 0：基频分量比倍频强 → 声门开放相长 → 漏气
  - H1-H2 < -6：倍频比基频强 → 声门闭合相长 → 过紧
  - H1-H2 在 -2 ~ -4：正常闭合

V1.0 DL 替换：
────────────────
CNN 多任务输出头直接输出 breathiness_prob：
  共享 backbone → breathiness_head (sigmoid)
  训练时与 squeeze_head 联合训练
  好处：模型自动学习"挤卡 vs 漏气"的边界（不需要手工设计 H1-H2 阈值）
```

---

### 4.3 喉位检测 (Larynx Height)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 + ML** | LPC → 共振峰 → 偏移检测 | 60-75% |
| V1.0 | **DL** | CNN 隐式学习共振峰位置 | 80-90% |

**详细算法路线**：

```
传统算法方案：
────────────────
Step 1: LPC 共振峰提取
  - 对每帧信号做 LPC 分析（14 阶）
  - LPC 系数 → 求根 → 提取共振峰 F1, F2, F3
  
Step 2: 共振峰基准值
  需要知道用户唱的是什么元音（不同元音 F1/F2 不同）
  简化方案：不去识别元音，而是看共振峰的"相对变化"
  
Step 3: 一致性偏移检测
  计算 F1_shift, F2_shift, F3_shift（相对于该用户的历史均值）
  if shift_consistency > 0.7（三个共振峰同方向偏移）
    if direction = UP → 喉位偏高
    if direction = DOWN → 喉位偏低

问题：
  ① LPC 共振峰提取在端侧不太稳定（需要精确的预加重、窗函数）
  ② 需要用户"基准值"（历史数据或标准值）
  ③ 元音变化时共振峰自然变化 → 容易误报

DL 方案（更优）：
────────────────
CNN 从 Mel 频谱中隐式学习共振峰位置：
  - Mel 频谱本身就编码了频谱包络信息
  - 共振峰位置决定了 Mel 频谱的"峰谷"分布
  - CNN 可以学习到"这个 Mel 模式对应高喉位"
  
训练策略：
  - 用 Praat 提取精确 F1/F2/F3 作为辅助标签（auxiliary loss）
  - 主标签：声乐教师标注的喉位评级（1-5）
  
  Loss = BCE(squeeze) + CE(mix_class) + MSE(formant_aux) + CE(larynx_height)
```

---

### 4.4 混声比例检测 (Mix Voice Ratio)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V0.2** | **DL (CNN)** | 5 分类 ordinal | 75-85% |
| V1.0 | **DL (多任务)** | 联合训练 | 85-90% |

**为什么不用传统算法？**

```
混声是多个维度的综合表现：
  ① 声带振动模式（全振动 vs 边缘振动）→ 影响 Mel 低频
  ② 声门闭合程度 → 影响 H1-H2
  ③ 共鸣空间 → 影响共振峰
  ④ 气息分配 → 影响能量分布

传统规则需要手动组合这些特征的阈值：
  if H1_H2 < -4 AND F0 > passaggio AND some_formant_condition...
  → 组合爆炸，不同嗓音条件需要不同阈值

结论：混声检测天然适合 DL，因为：
  1. 特征之间有复杂的非线性交互
  2. 没有单一指标能代表混声比例
  3. CNN 从 Mel 中同时学习多个相关模式
```

**为什么是 5 分类而不是回归？**

```
回归（输出 0.0-1.0）的问题：
  - 混声比例不是连续可分的（声乐教师也只能粗略判断）
  - 标注一致性差（教师 A 说 0.4，教师 B 说 0.6）
  - 回归 loss 对噪声标签不鲁棒

5 分类 Ordinal：
  Class 0：纯胸声（Pure Chest）
  Class 1：偏胸混声（Chest-dominant Mix）
  Class 2：平衡混声（Balanced Mix）
  Class 3：偏头混声（Head-dominant Mix）
  Class 4：纯头声/假声（Pure Head/Falsetto）

Ordinal 编码：标签 = [1,1,1,0,0] 表示 Class 2
  → 自然保持了"2 比 0 更接近 3 而不是 0"的顺序关系
  → 标注一致性更好（教师对粗粒度分类更一致）
```

---

### 4.5 破音检测 (Voice Break)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统规则** | F0 帧间差 > 50Hz | **95%+** |

**为什么永远不需要 DL？**

```
破音是一个完美的规则检测场景：
  ① 物理定义极其清晰：基频在 1-2 帧内跳变 > 50Hz
  ② 单一特征就足够：F0 帧间差
  ③ 没有灰度：破了就是破了，没有"半破"
  ④ 极低延迟需求：需要即时检测

算法：
  if abs(F0[t] - F0[t-1]) > 50 
     AND F0[t] > 50 AND F0[t-1] > 50     # 都是有效 voiced
     AND energy[t] > -40 AND energy[t-1] > -40  # 不是静音
    → VOICE BREAK at frame t

不需要 EMA 平滑（事件是瞬态的）
不需要状态机（不需要持续）
不需要 DL（特征完全可设计）
```

---

### 4.6 颤音分析 (Vibrato Analysis)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 DSP** | F0 轨迹的短时 FFT | 90%+ |
| V2.0 | DL (可选) | 对复杂颤音模式分类 | 95%+ |

**为什么传统算法就够了？**

```
颤音有非常明确的物理定义：
  - 频率：F0 轨迹上 4.5-6.5Hz 的周期性调制
  - 深度：调制幅度 0.5-2 semitones
  - 规整度：调制的周期一致性

检测算法（传统 DSP）：
  1. 提取 F0 轨迹（已有，不需要额外计算）
  2. 取 1 秒窗（~62 帧 @ 16ms/帧）
  3. 对 F0 轨迹做 FFT
  4. 在 2-10Hz 范围内找峰值
  5. 峰值频率 = 颤音频率
  6. 峰值幅度 = 颤音深度
  7. 峰宽 = 规整度的倒数

判别：
  if peak_freq ∈ [4.5, 6.5] AND peak_depth ∈ [0.5, 2.0] st
    → 正常颤音
  if peak_freq < 4.5 → Wobble（过慢）
  if peak_freq > 6.5 → Tremolo（过快）
  if peak_depth > 2.0 → 幅度过大
  if no_clear_peak → 直声（无颤音）
```

---

### 4.7 气息稳定性 (Breath Stability)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 DSP** | 能量 + F0 变异系数 | 80-85% |

```
传统算法：
  window = 0.5 秒 (31 帧)
  energy_cv = std(energy[window]) / mean(energy[window])
  f0_cv = std(f0[window]) / mean(f0[window])
  
  if energy_cv > 0.10 → 气息不稳
  if f0_cv > 0.03     → 音高不稳（可能是气息引起）
  
  排除颤音：如果 F0 调制在 4.5-6.5Hz → 这是正常颤音，不是不稳
```

---

### 4.8 大白嗓检测 (Yelling Detection)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 + ML** | H1-H2 + Singer's Formant + Energy | 70-80% |
| V1.0 | **DL** | CNN 多任务头 | 85-90% |

```
传统算法方案：
────────────────
大白嗓的三个核心特征：
  ① 声门闭合过强但不是挤卡 → H1-H2 < -10dB
  ② Singer's Formant 缺失 → 3kHz 能量占比低
  ③ 音量极大 → Energy > -10dB

singer_formant_ratio = energy_in_2500_3500Hz / total_energy
if H1_H2 < -10 AND singer_formant_ratio < 0.05 AND energy > -10
  → 大白嗓

与挤卡的区别：
  挤卡：HNR 低（噪声多）
  大白嗓：HNR 可以正常（谐波纯度好，只是没有共鸣聚焦）

DL 优势：
  CNN 可以从 Mel 频谱中直接学习"有没有 3kHz 聚焦"这个模式
  比手工提取 singer_formant_ratio 更鲁棒
```

---

### 4.9 鼻音检测 (Nasality Detection)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **DL (CNN)** | Mel 频谱中的反共振峰 | 70-80% |

**为什么不用传统算法？**

```
问题：反共振峰（anti-formant）不像共振峰那样容易提取
  ① LPC 主要建模共振峰（极点），反共振峰（零点）需要 ARMA 模型
  ② ARMA 模型在端侧不稳定
  ③ 反共振峰的频率位置随元音变化很大

DL 方案更好：
  CNN 从 Mel 频谱中学习"800-1200Hz 凹陷"模式
  不需要显式提取反共振峰
  但需要排除鼻辅音（/m/, /n/）→ 可能需要简单 ASR
```

---

### 4.10 面罩共鸣检测 (Singer's Formant)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 DSP** | 频段能量比 | 80%+ |

```
传统算法（足够且简单）：
────────────────
singer_formant_energy = sum(mel_band[band_2500:band_3500])
total_energy = sum(mel_band[all])
ratio = singer_formant_energy / total_energy

这就是一个简单的频段比值 → 不需要 DL
但可以作为 DL 模型的辅助输入特征
```

---

### 4.11 硬起音检测 (Hard Onset)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统规则** | 能量上升斜率 | 85%+ |

```
传统算法：
────────────────
检测语音起始点（onset detection）：
  onset_frames = where(energy[t] > -40 AND energy[t-1] < -40)

对每个 onset：
  rise_rate = (energy[onset+2] - energy[onset]) / (2 × frame_duration)
  # 单位：dB/ms
  
  if rise_rate > 3.0 dB/ms → 硬起音
  if rise_rate < 1.0 dB/ms → 软起音（正常或气息起音）
```

---

### 4.12 力度控制检测 (Dynamics Control)

| 阶段 | 方法 | 核心逻辑 | 精度预期 |
|------|------|---------|---------|
| **V1.0** | **传统 DSP** | 帧间能量变化的统计特征 | 75%+ |

```
energy_diff = abs(energy[t] - energy[t-1]) for all voiced frames
energy_jitter = std(energy_diff)

if energy_jitter > 3.0 dB → 力度控制较差
if energy_jitter < 1.0 dB → 力度控制良好（或太平淡）
```

---

## 第五章：混合架构设计

### 5.1 V0.1 纯规则架构

```
audio → DSP Features → Rule Engine → EMA + State → UI
         (all trad)     (all trad)    (all trad)
         
特征：F0, HNR, Energy
维度：仅挤卡
参数：0
延迟：< 5ms
```

### 5.2 V0.2 规则 + 单任务 DL

```
audio → DSP Features → ┬──→ CNN Model → squeeze_prob
         (trad)        │                      ↓
                       └──→ Rule Engine → mix_voice_class
                                               ↓
                                        ┬──────┘
                                        ↓
                                   EMA + State → UI

CNN 负责：挤卡（替代规则引擎）
规则负责：混声 (初步，基于 H1-H2 简单版)
```

### 5.3 V1.0 多任务 DL + 辅助规则

```
audio → DSP Features → ┬──→ Multi-task CNN → squeeze_prob
         (trad)        │                   → mix_class
                       │                   → vibrato_params
                       │                   → larynx_score
                       │                   → breathiness_prob
                       │                        ↓
                       └──→ Rule Engine → voice_break (F0 jump)
                                       → hard_onset (energy rise)
                                       → breath_stability (CV)
                                       → dynamics_control (jitter)
                                       → singer_formant (band ratio)
                                               ↓
                                        ┬──────┘
                                        ↓
                                   Post-processing → UI

DL 负责：需要学习复杂模式的维度
  - 挤卡（频谱微妙模式）
  - 混声（多特征交互）
  - 颤音（F0 调制的复杂模式）
  - 喉位（共振峰偏移）
  - 漏气（与挤卡的精细区分）

规则负责：物理定义清晰的维度
  - 破音（F0 跳变）
  - 硬起音（能量上升率）
  - 气息稳定性（变异系数）
  - 力度控制（能量 jitter）
  - 面罩共鸣（频段比值）
```

### 5.4 DL 与规则的协作模式

```
模式 1: DL 输出 + 规则校验
────────────────
CNN → squeeze_prob = 0.8
Rule → HNR = 15dB (正常范围)
校验：CNN 说挤卡但 HNR 正常 → 降低置信度
final_prob = squeeze_prob × 0.5 (因为被规则"否决"了)

模式 2: 规则输出，DL 精化
────────────────
Rule → 粗略判断"可能挤卡"(HNR < 12)
CNN → 精确概率 0.65
final → 0.65 (DL 精化了规则的粗略判断)

模式 3: DL 全权，规则备份
────────────────
CNN → squeeze_prob (正常模式)
if CNN_uncertainty > threshold → 回退到 Rule 引擎
(模型不确定时使用规则作为 fallback)
```

---

## 第六章：端侧部署约束下的选型策略

### 6.1 iPhone A14 算力预算

| 处理器 | 算力 | 适合 |
|--------|------|------|
| **CPU (2+4 核)** | ~3 GFLOPS | DSP 特征提取、规则引擎 |
| **GPU (4 核)** | ~1.4 TFLOPS | 通用计算（但不适合小模型） |
| **Neural Engine** | **11 TOPS** | **CNN 推理（最适合）** |

### 6.2 每帧时间预算分配

```
总预算：< 16ms / 帧（16kHz, hop=256）

DSP 特征提取：< 3ms (CPU)
  ├── FFT:          0.1ms
  ├── Mel:          0.1ms
  ├── F0 (ACF):     0.5ms
  ├── HNR:          0.3ms
  ├── H1-H2:        0.1ms
  ├── LPC (if used): 0.5ms
  └── Others:       0.3ms

CNN 推理：< 2ms (Neural Engine)
  └── 280K params Float16

规则引擎：< 0.1ms (CPU)
  └── 简单比较和分支

后处理：< 0.5ms (CPU)
  ├── EMA:          < 0.01ms
  └── State filter: < 0.01ms

UI 更新：< 1ms (Main thread)
  └── SwiftUI @Published 更新

总计：< 7ms ← 远低于 16ms 预算
```

### 6.3 CoreML 部署考量

| 项目 | 考量 |
|------|------|
| 量化 | Float16 足够（A14 Neural Engine 原生 Float16） |
| 模型大小 | 280K params × 2 bytes = 560KB |
| 输入形状 | [1, 82, 32] — 固定维度 |
| 输出形状 | [1, 1] (squeeze_prob) + [1, 5] (mix_class) |
| 转换工具 | coremltools 7+ (PyTorch → CoreML .mlpackage) |
| 预测 API | MLModel.prediction(from:) |
| 批处理 | 不需要（实时逐帧） |

### 6.4 模型更新策略

```
V0.2: 模型打包在 App 内（随 App 更新）
V1.0: On-Device Fine-tuning
  - 在用户设备上用用户反馈数据微调最后 1-2 层
  - 个性化适配用户嗓音
  - 需要 CoreML 的 MLUpdateTask API
V2.0: 后台下载新模型
  - 服务器端训练 → 推送新 .mlpackage → 后台下载替换
```

---

## 第七章：从 V0.1 到 V2.0 的算法演进路线

### 7.1 完整演进时间线

```
V0.1 (Week 1-2)    纯规则 
────────────────
  传统算法：HNR → sigmoid → EMA → StateFilter
  维度：挤卡 only
  精度：65-75%
  数据：0

        ↓ TestFlight 上线，收集用户反馈数据

V0.2 (Week 5-8)    引入第一个 DL 模型
────────────────
  DL：挤卡 CNN (单任务)
  传统：混声 (H1-H2 简单版)
  维度：挤卡 + 混声（粗略）
  精度：80-90% 挤卡，70% 混声
  数据需求：500+ clips

        ↓ 持续收集数据，声乐教师标注

V0.3 (Week 9-12)   DL 多任务
────────────────
  DL：挤卡 + 混声 + 漏气 (3头)
  传统：破音、硬起音
  维度：5
  精度：85%+ 挤卡
  数据需求：1000+ clips

        ↓ 数据量充足，开始增加维度

V1.0 (Week 13-20)  全面 DL + 辅助规则
────────────────
  DL：7 维度 (挤卡+混声+漏气+喉位+鼻音+颤音+大白嗓)
  传统：5 维度 (破音+硬起音+气息+力度+面罩)
  总维度：12
  精度：90%+ 核心维度
  数据需求：2000+ clips

V2.0 (Week 20-30)  轻量 ASR 辅助
────────────────
  新增：舌根后压、下巴僵硬（需要音素级分析）
  DL：主模型 + 轻量 ASR (WhisperTiny ~40M params)
  总维度：15+
```

### 7.2 "规则引导 DL"的渐进策略

```
Phase 1: 规则运行，积累数据
  规则引擎在线运行 → 用户反馈"准/不准" → 标注数据入库

Phase 2: 规则输出作为弱标签
  rule_label = rule_engine(features)  # 弱标签（有噪声）
  user_feedback = user_click()         # 半标签（只有部分帧有）
  teacher_label = teacher_annotation() # 强标签（最少量）
  
  训练 DL 时：
  loss = 0.3 × BCE(model_out, rule_label)      # 弱标签大量
       + 0.2 × BCE(model_out, user_feedback)    # 半标签中量
       + 0.5 × BCE(model_out, teacher_label)    # 强标签少量

Phase 3: DL 替换规则
  DL 精度 > 规则精度 → 在线切换
  保留规则引擎作为 fallback 和 sanity check

Phase 4: DL 反哺规则
  DL 发现的新模式 → 提炼为规则 → 用于可解释性
  "模型认为你在挤卡，因为 HNR 偏低且 1-3kHz 有异常能量"
```

### 7.3 最终总结表

| 维度 | V0.1 | V0.2 | V0.3 | V1.0 | V2.0 | 最终方法 |
|------|------|------|------|------|------|---------|
| 挤卡 | 📏 规则 | 🧠 DL | 🧠 DL | 🧠 DL | 🧠 DL | DL |
| 混声 | — | 📏 规则 | 🧠 DL | 🧠 DL | 🧠 DL | DL |
| 漏气 | — | — | 📏 规则 | 🧠 DL | 🧠 DL | DL |
| 喉位 | — | — | — | 🧠 DL | 🧠 DL | DL |
| 鼻音 | — | — | — | 🧠 DL | 🧠 DL | DL |
| 颤音 | — | — | — | 📏 DSP | 📏 DSP | 传统 DSP |
| 大白嗓 | — | — | — | 🧠 DL | 🧠 DL | DL |
| 破音 | — | — | — | 📏 规则 | 📏 规则 | **永远规则** |
| 硬起音 | — | — | — | 📏 规则 | 📏 规则 | **永远规则** |
| 气息稳定 | — | — | — | 📏 DSP | 📏 DSP | 传统 DSP |
| 力度控制 | — | — | — | 📏 DSP | 📏 DSP | 传统 DSP |
| 面罩共鸣 | — | — | — | 📏 DSP | 📏 DSP | 传统 DSP |
| 舌根后压 | — | — | — | — | 🧠 DL+ASR | DL+ASR |
| 下巴僵硬 | — | — | — | — | 🧠 DL+ASR | DL+ASR |

> **原则重申**：物理定义清晰、单一特征可决定的 → 传统规则/DSP。特征交互复杂、模式微妙、个体差异大的 → DL。两者互补，缺一不可。

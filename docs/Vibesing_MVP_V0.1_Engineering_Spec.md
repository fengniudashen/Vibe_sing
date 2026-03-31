# Vibesing MVP V0.1 — 极简工程规格书

> **原则**：如果某个设计不能在 2 周内让用户"明显感知价值"，就不属于 MVP。  
> **本规格书经过 Claude 4.6 / ChatGPT / Gemini 3.1 Pro 三轮交叉审计后，砍掉 70% 复杂度，只保留最小可验证路径。**  
> **日期**：2026-03-30

---

## 一句话定义 MVP

> 用户对着手机高音唱一句 → 屏幕**立刻亮红灯说"你挤了"** → 用户放松后红灯灭掉。  
> 仅此一个闭环。做到这一点，产品已经秒杀所有竞品。

---

## 1. 砍到骨头的特征输入

### ❌ 被砍掉的（V0.1 不做）

| 特征 | 砍掉理由 |
|------|---------|
| CPP（倒谱峰值突变） | 移动端麦克风差异 + 房间混响 + 蓝牙降噪 → 实际场景不可靠。降级为 Debug/可视化用，不进推理 |
| F1/F2/F3 共振峰追踪 | 端侧 LPC 又贵又不稳定，MVP 不需要 |
| H1-H2（谐波差） | 依赖精确 F0，增加 pipeline 复杂度，收益在 MVP 阶段不值 |
| Singer's Formant 3kHz 能量 | 需要精确频段分析，V1.1 再加 |

### ✅ V0.1 只用这 4 个特征

| 特征 | 维度 | 计算方式 | 用途 |
|------|------|---------|------|
| **Mel-Spectrogram** | 80 dim/frame | `librosa.feature.melspectrogram` @ 16kHz | 主要频谱特征 |
| **F0（基频）** | 1 dim/frame | CREPE-tiny 或 `parselmouth (Praat)` | voicing mask + 音域判断 |
| **HNR（谐噪比）** | 1 dim/frame | `parselmouth` 自相关法 | **挤卡核心判据**（抗噪性远好于 CPP） |
| **Frame Energy** | 1 dim/frame | `mel.logsumexp(dim=0)` | 防音量混淆的显式解耦特征 |

**输入 tensor shape**: `[82 × T]`（80 mel + HNR + energy）, F0 用于 voicing mask 不直接进模型。

---

## 2. 模型：只有一个头

### ❌ 被砍掉的

- ~~7 个多任务头~~ → 1 个头
- ~~双流 Late Fusion~~ → 单流 concat（数据 < 5k 时双流没优势）
- ~~混声比例回归/分类~~ → V0.1 不做（先 fake 一个 UI 动画即可）
- ~~Evidential DL / Co-Teaching~~ → V0.1 不做

### ✅ V0.1 模型定义

```
任务：挤卡二分类（Squeeze Detection, Binary）
输入：[82 × 32] (82 features × 32 frames = 512ms context)
输出：squeeze_prob ∈ [0, 1]（单个 Sigmoid 标量）

网络：4-layer Causal Depthwise-Separable CNN
参数量：< 300K
端侧推理：< 1ms on A15+ Neural Engine
```

### 架构图

```
Input [82 × 32]
    │
    ▼
CausalDWConv1D(k=7, d=1) → BN → SiLU → PW(82→128)
    │ + Residual
    ▼
CausalDWConv1D(k=7, d=2) → BN → SiLU → PW(128→128)
    │ + Residual
    ▼
CausalDWConv1D(k=7, d=4) → BN → SiLU → PW(128→256)
    │ + Residual
    ▼
CausalDWConv1D(k=7, d=8) → BN → SiLU → PW(256→256)
    │ + Residual
    ▼
AdaptiveAvgPool1D(1) → Linear(256→1) → Sigmoid
    │
    ▼
squeeze_prob ∈ [0, 1]
```

**感受野**：7 × (1+2+4+8) = 105 frames × 16ms = 1.68s — 足够覆盖一个完整的高音延长音。

---

## 3. 后处理：极简版

### ❌ 被砍掉的

- ~~Kalman Filter~~
- ~~Penalty decay 指数衰减~~
- ~~Sustained reward 计算~~
- ~~Logistic mapping~~
- ~~VocalHealthScore 复杂打分~~

### ✅ V0.1 只保留

```python
# 1. EMA 平滑（防 UI 闪烁）
ALPHA = 0.4  # 偏灵敏，让红灯响应快
smooth_prob = ALPHA * raw_prob + (1 - ALPHA) * prev_smooth_prob

# 2. 状态机（防假触发）
class SqueezeStateFilter:
    MIN_FRAMES_TO_TRIGGER = 6   # 连续 6 帧 (~96ms) 才亮红灯
    MIN_FRAMES_TO_CLEAR = 10    # 连续 10 帧 (~160ms) 才灭红灯
    THRESHOLD_ON = 0.6
    THRESHOLD_OFF = 0.3
    
    def update(self, prob: float) -> bool:
        """返回 True = 红灯亮"""
        ...

# 3. 分数（极简版）
score = int(round(100 * (1 - smooth_prob)))
# 就是这么简单。
```

---

## 4. 数据策略：从 0 到 1000 条

### Phase 1：规则预标注（Week 1，零人工成本）

```python
def auto_label_squeeze(hnr_mean, energy_mean, f0_mean):
    """基于 HNR 的弱标签生成器"""
    if f0_mean < 200:        # 低音区不太挤卡
        return 0.1
    if hnr_mean < 12 and energy_mean > -20:  # HNR 低 + 音量不小
        return 0.85           # 大概率挤卡
    if hnr_mean > 20:
        return 0.05           # 大概率正常
    return 0.5                # 不确定 → 人工标注
```

### Phase 2：数据来源

| 来源 | 数量 | 方式 |
|------|------|------|
| 自录：团队 + 3 位声乐老师各录 30 段高音（正常+挤卡） | ~200 段 | 手动录制 |
| VocalSet 学术数据集（已有标注） | ~500 段（筛选高音段） | 下载 + 自动筛选 |
| Phase 1 规则预标注的 YouTube 翻唱 | ~300 段（只用高置信度） | 爬取 + 自动标注 |

### Phase 3：训练时的抗噪措施（只保留最简单的）

```python
# 1. Label Smoothing (必须)
label_smooth = 0.1
y = (1 - label_smooth) * y + label_smooth / 2

# 2. 随机增益增强 (必须，防音量混淆)  
gain_db = random.uniform(-12, +6)
audio *= 10 ** (gain_db / 20)

# 3. 输入能量归一化 (必须)
mel = mel - mel.mean(dim=-1, keepdim=True)
```

---

## 5. 端侧 Audio Pipeline（iOS）

```
AVAudioEngine (44.1kHz mono)
    │ installTap(bufferSize: 1024)
    ▼
vDSP.Downsample → 16kHz ring buffer (2048 samples)
    │ Every 256 new samples (16ms):
    ▼
┌── STFT(n_fft=1024, hop=256) → 80-Mel
├── Parselmouth-port or simple autocorrelation → HNR
├── log energy
└── (F0 for voicing mask only, not model input)
    │ concat → [82-dim frame]
    │ push to 32-frame ring buffer
    ▼
CoreML.predict([82 × 32]) → squeeze_prob
    │
    ▼
EMA smooth → StateFilter → Red/Green light @ 60fps
```

**端到端延迟**：~35ms typical, ~52ms worst case。

---

## 6. 两周执行计划

### Week 1：让红灯亮起来（纯 Python，不碰 iOS）

| Day | 任务 | 交付物 |
|-----|------|--------|
| Day 1 | 用 `librosa` + `parselmouth` 实现逐帧 Mel + HNR + F0 + Energy 提取 | `dsp_features.py` |
| Day 2 | 对 2 段干音（1正常 + 1挤卡）画 4 条时序曲线 | `visualize_features.py` + 截图确认 HNR 在挤卡时骤降 |
| Day 3 | 写 rule-based 挤卡检测器 (`HNR < 12 && energy > thresh`) | `rule_detector.py` |
| Day 4-5 | 在 10-20 段测试音频上验证 rule-based 准确率；调阈值 | 准确率报告。**如果 > 70%，MVP 可行性确认** |

### Week 2：最小 DL 模型（PyTorch）

| Day | 任务 | 交付物 |
|-----|------|--------|
| Day 1 | 搭 4-layer Causal DWSep CNN（上述架构） | `model.py` |
| Day 2 | 写 Dataset + DataLoader（Mel+HNR+Energy 输入） | `dataset.py` |
| Day 3 | 用 rule-based 标签做第一轮伪标签训练 | `train.py` + 训练 log |
| Day 4 | 对比 rule-based vs 模型准确率 | 消融实验结果 |
| Day 5 | 导出 CoreML `.mlpackage`（Float16） | `squeeze_detector.mlpackage` |

### Week 3+（仅当 Week 1-2 跑通后）

- 加 3-class 混声分类头（胸声主导 / 混声 / 头声主导）
- iOS 端 AVAudioEngine pipeline
- 实时 UI（红绿灯 + 简易 Slider）

---

## 7. MVP 成功判据

| 指标 | 阈值 | 意义 |
|------|------|------|
| Rule-based 挤卡检测准确率 | > 70% | 物理假设成立 |
| DL 模型 vs Rule-based 准确率增益 | > 5% | 证明模型有价值 |
| 端到端延迟（iOS） | < 60ms | 用户感知"实时" |
| 用户测试反馈："红灯在我挤的时候亮了" | ≥ 4/5 人确认 | **产品成立** |

---

## 8. TestFlight 即数据工厂（关键战略）

> 三方终极共识：**不要等模型完美再上线。Rule-based 能亮红灯就发 TestFlight。**

### 为什么 2 周内必须上 TestFlight

1. **免费数据采集**：50 个 beta 用户会在各种真实场景（浴室混响、蓝牙耳机、嘈杂房间）里疯狂唱歌。这些数据比你自己录 1000 遍有价值得多。
2. **免费人工标注**：在红灯亮起时弹出一个极简反馈按钮：`"刚才红灯亮了，你觉得准吗？ [准 👍] [不准 👎]"`。这就是你的标注系统。
3. **验证 PMF**：如果用户试玩一次就删了 → 方向错误，趁早转弯。如果用户自发玩第二次 → 产品成立。
4. **心理锚点**：用户不需要 99% 准确率。只要红灯在他们"感觉挤了"的时候**经常**亮，60-70% 主观正确率就足以产生"这个 App 在听我唱歌"的魔法感。

### TestFlight 上线判据

```
✅ 红灯功能不崩溃（crash-free）
✅ 延迟 < 100ms（用户感知"实时"）
✅ UI 不丑到离谱（一个红绿灯 + 一个能量条）
→ 发布 TestFlight
```

### TestFlight V0.1 极简功能清单

| 功能 | 实现方式 | 需要 DL？ |
|------|---------|----------|
| 实时录音 | AVAudioEngine | ❌ |
| 挤卡红灯 | Rule-based (HNR < 阈值 && Energy > 阈值 && F0 > 高音线) | ❌ |
| 能量跳动条（伪装成"发声状态"） | 实时 RMS Energy | ❌ |
| 简易分数 | `100 - 100 * squeeze_binary` + EMA | ❌ |
| "准吗？"反馈按钮 | 本地 SQLite 记录 + 音频片段缓存 | ❌ |

> **注意**：V0.1 完全不需要 CoreML / 深度学习。纯 DSP + 规则引擎 + 极简 UI 就够了。

### 数据回收 Pipeline

```
用户唱歌 → 红灯亮起 → 弹出"准吗?"按钮
    │
    ▼ 用户点击 [准 👍] 或 [不准 👎]
    │
    ▼ 本地保存：{audio_clip_2s.wav, hnr_values, energy, f0, user_label}
    │
    ▼ 下次联网时批量上传（用户授权后）
    │
    ▼ 服务端汇总 → 产生带人工标注的真实训练数据
    │      （50用户 × 每日20次触发 × 7天 = ~7000 条标注样本）
    │
    ▼ Week 3-4：用这些数据训练第一版 DL 模型
```

---

## 9. 演进路线图

```
Week 1-2  TestFlight V0.1：Rule-based 红灯 + "准吗?"反馈按钮
    │     （纯 DSP，零 DL，验证 PMF + 采集数据）
    ▼
Week 3-4  V0.2：用回收数据训练极简 CNN 替换 Rule-based
    │     （CoreML 上线，准确率应显著提升）
    ▼
Week 5-6  V0.3：+ 3-class 混声 Slider（胸/混/头）
    │
    ▼
Month 3   V1.0：Multi-Task Dual-Stream + 完整打分系统
    │     （此时再用 Taxonomy V1.0 的完整蓝图）
    ▼
Month 6+  V2.0：隐蔽代偿 + 训练态评估 + App Store 正式上架
```

---

## 10. 这份规格书的定位

| 文档 | 用途 |
|------|------|
| [Vibesing_Taxonomy_V1_Technical_Review.md](Vibesing_Taxonomy_V1_Technical_Review.md) | 完整技术蓝图（V1.0-V2.0 的理论上限） |
| **本文档（MVP V0.1 Engineering Spec）** | **2 周内必须交付的最小可验证产品** |

---

> **结论**：你手里已经有了从 V0.1 到 V2.0 的完整路线图。V0.1 只需要做到"红灯亮起来 + 用户能反馈准不准"这两件事。**不要等模型完美。先让 50 个人觉得"这个 App 在听我唱歌"，一切就开始了。**

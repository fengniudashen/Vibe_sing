# Vibesing — AI Agent 工程指令文档

> **目标读者**：任何 AI 编码 Agent（GitHub Copilot, Claude, ChatGPT, Cursor Agent 等）  
> **用途**：在后续对话中加载此文档作为项目上下文，让 Agent 立刻理解项目状态并高效执行任务。

---

## 项目一句话概述

**Vibesing（高音觉醒）** 是一款 iOS 声乐辅助 App，通过端侧实时 DSP + AI 检测用户高音时的"挤卡"发声病灶，以红绿灯形式即时警告。当前处于 MVP V0.1 阶段，使用纯 Rule-based 检测（无深度学习）。

---

## 项目结构

```
Vibesing/
├── docs/
│   ├── Vibesing_Taxonomy_V1_Technical_Review.md   # 完整 7 维技术蓝图
│   ├── Vibesing_MVP_V0.1_Engineering_Spec.md      # MVP 工程规格书
│   ├── PROJECT_PLAN.md                             # 计划表与进度
│   ├── AI_AGENT_GUIDE.md                           # 本文档
│   └── USER_GUIDE.md                               # 用户指南
├── python/                                         # Week 1 DSP 验证代码
│   ├── dsp_features.py                             # 特征提取 (Mel/F0/HNR/Energy)
│   ├── visualize_features.py                       # 可视化
│   ├── rule_detector.py                            # Rule-based 挤卡检测器
│   └── requirements.txt
└── ios/Vibesing/                                   # iOS TestFlight App
    ├── VibesingApp.swift                           # App 入口
    ├── Audio/
    │   ├── AudioCaptureEngine.swift                # AVAudioEngine 录音 → 16kHz
    │   ├── DSPProcessor.swift                      # 端侧 Mel/HNR/F0/Energy
    │   └── SqueezeDetector.swift                   # Rule-based 检测 + EMA + 状态机
    ├── UI/
    │   ├── ContentView.swift                       # 主界面
    │   ├── SqueezeIndicator.swift                  # 红绿灯动画
    │   └── EnergyBar.swift                         # 能量条
    └── Data/
        └── FeedbackStore.swift                     # SQLite 用户反馈存储
```

---

## 核心技术栈

| 层级 | 技术 | 版本要求 |
|------|------|---------|
| Python DSP | librosa, parselmouth (Praat), numpy, matplotlib | Python 3.10+ |
| iOS | Swift 5.9+, SwiftUI, AVFoundation, Accelerate (vDSP) | iOS 17+, Xcode 15+ |
| 数据存储 | SQLite3 (iOS 内置) | — |
| 未来 DL | PyTorch → CoreML (.mlpackage, Float16) | PyTorch 2.x, coremltools 7+ |

---

## 关键技术决策（不可更改）

这些决策经过三方 AI 交叉验证后锁定，后续 Agent 不应质疑或修改：

1. **MVP 只做挤卡检测（Squeeze Detection, Binary）** — 不做混声比例、共鸣位置等。
2. **采样率 16kHz** — 所有音频处理统一 16kHz，不保留 44.1kHz pipeline。
3. **输入特征：Mel(80) + HNR + Energy** — CPP 不进推理（端侧不稳定），F0 仅用于 voicing mask。
4. **模型架构：Causal Depthwise-Separable CNN (TCN)** — 不用 Transformer/Conformer。
5. **平滑：EMA（信号层）+ 状态机（语义层）** — 不用 Kalman Filter。
6. **混声比例（V0.2）用 5-class 有序分类** — 不用直接回归。

---

## 挤卡检测算法核心逻辑

Python 和 iOS 代码使用完全一致的参数：

```
输入：每帧 HNR(dB), Energy(dB), F0(Hz)

1. 静音门控：if Energy < -40dB OR F0 < 50Hz → prob = 0
2. HNR → 概率映射：prob = sigmoid(0.5 * (HNR_THRESHOLD - HNR))
   - HNR_THRESHOLD = 12 dB
   - HNR = 6 dB → prob ≈ 0.95（高度挤卡）
   - HNR = 20 dB → prob ≈ 0.02（健康）
3. 高音增益：if F0 > 220Hz → boost × 1.0~1.3
4. EMA 平滑：α = 0.4
5. 状态机：连续 6 帧(96ms) > 0.55 才亮红灯，连续 10 帧(160ms) < 0.30 才灭灯
6. 分数：score = 100 × (1 - smooth_prob)
```

---

## 常见任务模板

### 如果被要求"调整挤卡阈值"

修改以下两个文件中的对应常量，保持 Python 和 iOS 一致：
- `python/rule_detector.py` → `HNR_SQUEEZE_THRESHOLD`
- `ios/Vibesing/Audio/SqueezeDetector.swift` → `hnrSqueezeThreshold`

### 如果被要求"添加新的检测维度"

1. 在 `DSPProcessor.swift` 的 `FrameResult` 中添加新字段
2. 在 `SqueezeDetector.swift` 中添加对应的处理逻辑
3. 同步更新 Python `dsp_features.py` 的 `FrameFeatures`
4. 确保 Python 和 iOS 逻辑完全一致

### 如果被要求"训练 DL 模型"

参考 `docs/Vibesing_MVP_V0.1_Engineering_Spec.md` 第 2-4 节：
- 架构：4-layer Causal DWSep CNN, <300K 参数
- 输入：[82 × 32] tensor (80 mel + HNR + energy, 32 帧)
- 输出：squeeze_prob sigmoid
- 训练技巧：Label Smoothing ε=0.1, 随机增益 [-12, +6] dB, 能量归一化

### 如果被要求"导出 CoreML"

```python
import coremltools as ct
model_traced = torch.jit.trace(model, dummy_input)  # [1, 82, 32]
mlmodel = ct.convert(
    model_traced,
    inputs=[ct.TensorType(name="features", shape=(1, 82, 32))],
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save("squeeze_detector.mlpackage")
```

---

## 阈值参数速查表

| 参数 | Python 变量 | iOS 变量 | 默认值 | 含义 |
|------|------------|---------|--------|------|
| HNR 挤卡阈值 | `HNR_SQUEEZE_THRESHOLD` | `hnrSqueezeThreshold` | 12.0 dB | 低于此值 → 疑似挤卡 |
| 能量底噪 | `ENERGY_MIN_THRESHOLD` | `energyMinThreshold` | -40.0 dB | 低于此值 → 静音忽略 |
| F0 高音线 | `F0_HIGH_VOICE_THRESHOLD` | `f0HighVoiceThreshold` | 220 Hz | 高于此线增强检测灵敏度 |
| EMA α | `ema.alpha` (init) | `emaAlpha` | 0.4 | 越大越灵敏（快响应） |
| 红灯触发帧数 | `MIN_FRAMES_TO_TRIGGER` | `minFramesToTrigger` | 6 帧 (96ms) | 连续帧确认避免闪烁 |
| 红灯消除帧数 | `MIN_FRAMES_TO_CLEAR` | `minFramesToClear` | 10 帧 (160ms) | 持续确认避免误灭 |
| 触发阈值 | `THRESHOLD_ON` | `thresholdOn` | 0.55 | 平滑概率 > 此值开始计数 |
| 消除阈值 | `THRESHOLD_OFF` | `thresholdOff` | 0.30 | 平滑概率 < 此值开始计数 |

---

## 注意事项

- **Python 和 iOS 的检测逻辑必须始终保持一致**。任何阈值修改都要同步两端。
- 所有音频处理均在 **16kHz 单声道** 下进行。
- iOS 端的 DSP 在 `DispatchQueue.global(qos: .userInteractive)` 上运行，UI 更新通过 `@Published` 自动在主线程。
- TestFlight 用户反馈数据存储在 App 沙盒的 `Documents/vibesing_feedback.sqlite3`，可通过 `FeedbackStore.exportCSV()` 导出。

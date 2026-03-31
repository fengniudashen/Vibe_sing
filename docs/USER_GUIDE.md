# Vibesing（高音觉醒）— 快速上手指南

> 面向开发者的项目入门文档。读完本指南后，你可以在 10 分钟内跑通 Python DSP 验证，并了解如何构建 iOS App。

---

## 一、项目简介

Vibesing 是一款 iOS 声乐辅助 App，能在你唱高音时**实时检测是否挤卡（vocal squeeze）**，并用红绿灯即时警告。

- 🔴 红灯 = 你在挤卡（喉咙紧了，声带过度压迫）
- 🟢 绿灯 = 发声健康

当前版本（V0.1）使用纯物理规则检测，不依赖深度学习模型。

---

## 二、Python 快速验证（5 分钟）

### 2.1 安装依赖

```bash
cd python
pip install -r requirements.txt
```

所需库：`librosa`, `parselmouth`, `numpy`, `matplotlib`, `soundfile`

### 2.2 准备测试音频

录制两段干音（无伴奏、无混响，手机录音即可）：

1. **正常高音**：选一首歌的高音段（比如 C4-E5），用放松的声音唱一个长音（如 "啊~~" 持续 5 秒）
2. **挤卡高音**：故意用力挤、扯着嗓子喊同一个音（你会感到喉咙紧绷）

保存为 `normal.wav` 和 `squeezed.wav`。

### 2.3 查看声学特征

```bash
# 单文件分析（4 面板：频谱 + F0 + HNR + 能量）
python visualize_features.py normal.wav

# 两文件对比
python visualize_features.py normal.wav squeezed.wav --compare

# 保存到图片
python visualize_features.py normal.wav squeezed.wav --compare --save comparison.png
```

**你应该看到**：挤卡音频的 HNR（谐噪比）明显低于正常音频的 HNR。如果 HNR 差异小于 5dB，你的"挤卡"可能录得不够典型——试着更用力地挤。

### 2.4 运行挤卡检测

```bash
# 分析并输出报告
python rule_detector.py squeezed.wav

# 分析 + 可视化（5 面板：频谱+HNR+概率+分数+红绿灯时间线）
python rule_detector.py squeezed.wav --plot

# 保存可视化
python rule_detector.py squeezed.wav --save result.png
```

输出示例：
```
==================================================
  SQUEEZE DETECTION REPORT
==================================================
  Duration:          5.23s
  Voiced frames:     298/326
  Squeeze ratio:     67.4%
  Mean health score: 32.6/100
==================================================

  [!] HIGH SQUEEZE DETECTED — 挤卡严重！
      Your voice shows significant constriction.
      Try relaxing your throat and supporting with breath.
```

### 2.5 调整阈值

如果检测结果不符合预期，编辑 `rule_detector.py` 顶部的三个阈值：

```python
HNR_SQUEEZE_THRESHOLD = 12.0     # 降低 → 更宽松（少报挤卡）
ENERGY_MIN_THRESHOLD = -40.0     # 提高 → 忽略更多低音量段
F0_HIGH_VOICE_THRESHOLD = 220.0  # 降低 → 对中低音也检测挤卡
```

---

## 三、iOS App 构建

### 3.1 前置条件

- macOS + Xcode 15+
- iPhone（真机，模拟器无麦克风）
- Apple Developer 账号（TestFlight 需要）

### 3.2 创建 Xcode 工程

1. 打开 Xcode → File → New → Project → iOS → App
2. Product Name: `Vibesing`
3. Interface: SwiftUI
4. Language: Swift
5. 创建后，将 `ios/Vibesing/` 下所有 `.swift` 文件拖入工程

### 3.3 配置权限

在 `Info.plist` 或 Target → Info → Custom iOS Target Properties 中添加：

| Key | Value |
|-----|-------|
| `NSMicrophoneUsageDescription` | Vibesing 需要麦克风来实时分析你的声音，帮助你改善发声技巧 |

### 3.4 运行

1. 连接 iPhone → 选择真机作为 Target
2. `Cmd + R` 运行
3. 授权麦克风 → 点击"开始演唱"
4. 对着手机唱一个高音 → 观察红绿灯

### 3.5 App 使用流程

```
打开 App
   ↓
点击「开始演唱」（蓝色按钮）
   ↓
授权麦克风（首次）
   ↓
开始唱歌
   ↓
正常发声 → 绿灯 🟢 + 分数接近 100
挤卡/喊 → 红灯 🔴 + 分数骤降 + 红灯振动
   ↓
红灯消失后弹出反馈：「准吗？👍 / 👎」
   ↓
点击反馈（数据自动存储到本地）
   ↓
点击「停止」结束
```

---

## 四、文件说明

| 文件 | 说明 |
|------|------|
| `docs/Vibesing_Taxonomy_V1_Technical_Review.md` | 完整技术蓝图（深度技术人员阅读） |
| `docs/Vibesing_MVP_V0.1_Engineering_Spec.md` | MVP 工程规格（2 周落地计划） |
| `docs/PROJECT_PLAN.md` | 项目进度与里程碑 |
| `docs/AI_AGENT_GUIDE.md` | AI Agent 接手项目的快速上下文 |
| `python/dsp_features.py` | 声学特征提取核心模块 |
| `python/visualize_features.py` | 特征可视化工具 |
| `python/rule_detector.py` | 挤卡检测器（Rule-based）|
| `ios/Vibesing/Audio/*.swift` | iOS 音频采集 + DSP + 检测 |
| `ios/Vibesing/UI/*.swift` | SwiftUI 界面组件 |
| `ios/Vibesing/Data/*.swift` | 用户反馈数据存储 |

---

## 五、FAQ

**Q: 为什么不用 AI 模型？**  
A: V0.1 先用物理规则验证核心假设。如果 HNR 阈值在真实场景下准确率 >70%，再训练深度学习模型。先验证物理，再上 AI——这样不会浪费时间在错误的方向上。

**Q: "挤卡"是什么？**  
A: 声乐术语，指唱高音时喉部肌肉过度紧张、声带过度内收（甚至假声带参与闭合），导致声音变得"挤"、"卡"、"憋"。是 90% 业余歌手唱高音时的最大问题。

**Q: 检测不准怎么办？**  
A: 调整 `HNR_SQUEEZE_THRESHOLD`。默认 12 dB 是保守值。如果你的"正常高音"也被误判为挤卡，把阈值降到 10 dB 试试。反之，如果明显挤卡没被检测到，把阈值提高到 14 dB。

**Q: 需要什么样的环境录音？**  
A: 越安静越好。避免：打开空调的房间、有回声的浴室、蓝牙耳机（其降噪算法会干扰 HNR 计算）。最理想的是近距离对着有线耳机的麦克风唱。

**Q: 下一步是什么？**  
A: 见 `docs/PROJECT_PLAN.md`。核心路线：TestFlight 上线 → 收集用户反馈数据 → 训练 DL 模型 → 添加混声检测 → App Store。

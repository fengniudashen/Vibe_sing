# Vibesing（高音觉醒）— 项目计划表与完成情况

> **最后更新**：2026-03-30  
> **项目状态**：🟡 MVP V0.1 开发中

---

## 一、里程碑总览

| 里程碑 | 目标 | 状态 | 预期时间 |
|--------|------|------|---------|
| M0: 技术蓝图 | 完成 Taxonomy 评审 + MVP 工程规格 | ✅ 已完成 | Week 0 |
| M1: Python DSP 验证 | 离线特征提取 + 可视化 + Rule-based 检测器 | ✅ 代码已交付 | Week 1 Day 1-3 |
| M2: Rule-based 准确率验证 | 在 10-20 段测试音频上验证 >70% 准确率 | 🔲 待执行 | Week 1 Day 4-5 |
| M3: iOS App 骨架 | AVAudioEngine + DSP + Rule-based + UI | ✅ 代码已交付 | Week 1-2 |
| M4: TestFlight V0.1 | 最小可用版本上线 TestFlight | 🔲 待执行 | Week 2 |
| M5: DL 模型训练 | PyTorch Causal CNN + 伪标签训练 | 🔲 未开始 | Week 3-4 |
| M6: CoreML 集成 | 模型导出 + iOS 端集成 | 🔲 未开始 | Week 4 |
| M7: V0.2 混声检测 | 3-class 混声 Slider | 🔲 未开始 | Week 5-6 |
| M8: V1.0 多任务模型 | Dual-Stream + 完整打分系统 | 🔲 未开始 | Month 3 |

---

## 二、已完成交付物清单

### 文档

| # | 文件 | 描述 | 状态 |
|---|------|------|------|
| D1 | `docs/Vibesing_Taxonomy_V1_Technical_Review.md` | 7 维标签矩阵完整技术评审（含三方交叉验证） | ✅ |
| D2 | `docs/Vibesing_MVP_V0.1_Engineering_Spec.md` | 极简 MVP 工程规格书（2 周 TestFlight 路线图） | ✅ |
| D3 | `docs/PROJECT_PLAN.md` | 本文档 — 计划表与完成情况 | ✅ |
| D4 | `docs/AI_AGENT_GUIDE.md` | 面向 AI Agent 的项目指令文档 | ✅ |
| D5 | `docs/USER_GUIDE.md` | 面向用户的 Quick Start 指南 | ✅ |

### Python 代码（DSP 验证 Pipeline）

| # | 文件 | 功能 | 状态 |
|---|------|------|------|
| P1 | `python/dsp_features.py` | Mel + F0 + HNR + Energy 逐帧特征提取 | ✅ |
| P2 | `python/visualize_features.py` | 4 面板时序曲线可视化 + 对比模式 | ✅ |
| P3 | `python/rule_detector.py` | Rule-based 挤卡检测 + EMA + 状态机 + 打分 | ✅ |
| P4 | `python/requirements.txt` | Python 依赖清单 | ✅ |

### iOS 代码（TestFlight V0.1）

| # | 文件 | 功能 | 状态 |
|---|------|------|------|
| S1 | `ios/Vibesing/Audio/AudioCaptureEngine.swift` | AVAudioEngine → 16kHz 降采样 → 帧发布器 | ✅ |
| S2 | `ios/Vibesing/Audio/DSPProcessor.swift` | 端侧 Mel + HNR + F0 + Energy (vDSP/Accelerate) | ✅ |
| S3 | `ios/Vibesing/Audio/SqueezeDetector.swift` | Rule-based 挤卡检测 (与 Python 逻辑一致) | ✅ |
| S4 | `ios/Vibesing/UI/ContentView.swift` | 主界面：红绿灯 + 分数 + 反馈弹窗 | ✅ |
| S5 | `ios/Vibesing/UI/SqueezeIndicator.swift` | 动画红绿灯组件 | ✅ |
| S6 | `ios/Vibesing/UI/EnergyBar.swift` | 实时能量条 + F0/HNR 统计 | ✅ |
| S7 | `ios/Vibesing/Data/FeedbackStore.swift` | SQLite 反馈存储 + CSV 导出 | ✅ |

---

## 三、未完成 / 待执行任务

### Week 1 剩余（你需要手动执行）

| 任务 | 描述 | 前置条件 |
|------|------|---------|
| **录制测试音频** | 自录 2 段干音：1 段正常高音 + 1 段挤卡高音（C4-E5 区间） | 麦克风/手机 |
| **跑 Python 可视化** | `python visualize_features.py normal.wav squeezed.wav --compare` | P1-P3 |
| **确认 HNR 骤降** | 肉眼检查：挤卡音频的 HNR 是否明显低于正常音频 | 可视化结果 |
| **调阈值** | 根据可视化调整 `HNR_SQUEEZE_THRESHOLD` (默认 12 dB) | 观察结果 |
| **跑 Rule-based 检测** | `python rule_detector.py squeezed.wav --plot` | P3 + 音频 |
| **验证准确率 >70%** | 在 10-20 段音频上统计 TP/FP/FN | 更多测试音频 |

### Week 2 剩余

| 任务 | 描述 | 前置条件 |
|------|------|---------|
| **创建 Xcode 工程** | 在 Xcode 中新建 iOS App 项目，导入已交付的 Swift 文件 | Xcode 15+ |
| **配置权限** | Info.plist 添加 `NSMicrophoneUsageDescription` | Xcode 工程 |
| **真机调试** | 在 iPhone 上跑通完整 pipeline (录音→检测→红灯) | 开发者账号 |
| **上线 TestFlight** | Archive + Upload + 邀请 50 beta 用户 | 真机验证通过 |

### Week 3-4（DL 模型）

| 任务 | 描述 | 前置条件 |
|------|------|---------|
| **搭 PyTorch 模型** | 4-layer Causal DWSep CNN (model.py) | 未开始 |
| **准备训练数据** | VocalSet 下载 + 自录数据 + Rule-based 伪标签 | 测试音频 |
| **训练 + 消融** | 对比 Rule-based vs DL 准确率 | 模型 + 数据 |
| **CoreML 导出** | `coremltools` 转换 .mlpackage (Float16) | 训练完成 |
| **iOS 集成 CoreML** | 替换 Rule-based 为 CoreML 推理 | .mlpackage |

---

## 四、关键决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-30 | MVP 只做 D2(混声)+D3(挤卡)，砍掉其他 5 个维度 | 三方 AI 一致建议 |
| 2026-03-30 | TestFlight V0.1 不用 DL，纯 Rule-based | ChatGPT: 先验证物理假设 |
| 2026-03-30 | CPP 降级为 Debug 特征，不进推理 | 端侧不稳定（麦克风/混响/蓝牙） |
| 2026-03-30 | 混声比例用 5-class 分类替代直接回归 | 伪连续变量 → 回归不收敛 |
| 2026-03-30 | 16kHz 采样率、Causal CNN (非 Conformer) | 三方锁定的定理级结论 |
| 2026-03-30 | TestFlight 用户 = 免费数据标注工厂 | "准吗?"按钮 + 音频回收 |

---

## 五、风险与阻塞项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 测试音频不足（<20 段） | 无法验证 Rule-based 准确率 | VocalSet 数据集 + 团队自录 |
| HNR 阈值在真实场景不稳定 | 红灯误报/漏报 | TestFlight 反馈数据快速迭代阈值 |
| 苹果审核拒绝（麦克风用途说明） | TestFlight 上线延迟 | 提前写好 privacy description |
| DL 模型比 Rule-based 没有显著提升 | Week 3-4 白费功夫 | 如果增益 <5%，保留 Rule-based，投入精力做 UX |

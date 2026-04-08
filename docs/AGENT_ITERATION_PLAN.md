# Vibesing — 自我迭代 Agent 开发计划

> **文档性质**：Agent 自治开发蓝图。每个 Sprint 结束时 Agent 自行评估、调整、进入下一轮。
> **启动条件**：将此文档 + `AI_AGENT_GUIDE.md` + `copilot-instructions.md` 喂给 Agent 即可冷启动。
> **最后更新**：2025-07-15

---

## 0. Agent 自治循环协议

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ITERATION LOOP                     │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ 1. SCAN  │───▶│ 2. PLAN  │───▶│ 3. BUILD │             │
│   │ 扫描现状  │    │ 规划任务  │    │ 编码实现  │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        ▲                               │                    │
│        │          ┌──────────┐         │                    │
│        │          │ 5. ADAPT │         ▼                    │
│        └──────────│ 自我调整  │◀───┌──────────┐             │
│                   └──────────┘    │ 4. TEST  │             │
│                                   │ 验证测试  │             │
│                                   └──────────┘             │
│                                                             │
│   每轮输出：代码 + 测试报告 + 下轮计划更新                    │
│   终止条件：用户命令 STOP 或 全部 Sprint 完成                 │
└─────────────────────────────────────────────────────────────┘
```

### Agent 行为规则

1. **每轮开始**：读取本文档 → 找到当前 Sprint → 检查前置条件是否满足
2. **前置条件未满足**：向用户报告阻塞项，提供解决方案（不空等）
3. **编码时**：Python 和 iOS 逻辑必须同步更新（不可只改一端）
4. **每轮结束**：更新本文档的进度标记（✅/🔲/❌），写出自评 + 下轮调整
5. **异常处理**：遇到不确定的技术决策 → 列出 2-3 个选项 + 推荐 → 问用户
6. **不可违反**：`copilot-instructions.md` 中的 6 条锁定技术决策

---

## 1. 当前项目状态快照

### 已交付 ✅

| 类别 | 文件数 | 状态 |
|------|--------|------|
| Python DSP Pipeline | 4 | ✅ dsp_features / visualize / rule_detector / requirements |
| iOS App 代码 | 8 | ✅ AudioCapture / DSPProcessor / SqueezeDetector / UI × 3 / FeedbackStore / App |
| 技术文档 | 14 | ✅ PRD / 工程规格 / 路线图 / Taxonomy / Agent指南 等 |
| Web 展示页 | 9 | ✅ 首页 / 原型 / 架构 / App UI / 生理 / 练习 / 误区 / 计划 / 解剖 |

### 未完成 🔲

| 项目 | 阻塞原因 | 优先级 |
|------|---------|--------|
| 录制测试音频 (≥10段) | 需要人工录制（挤卡+正常高音） | P0 |
| Python Pipeline 验证 | 依赖测试音频 | P0 |
| Xcode 工程创建 + 真机调试 | 需要 Mac + Apple Developer 账号 | P0 |
| TestFlight V0.1 上线 | 依赖 Xcode 工程 + 真机验证 | P1 |
| DL 模型训练 | 依赖数据集 (≥500条) | P2 |

---

## 2. Sprint 路线图

```
Sprint 0  ──▶  Sprint 1  ──▶  Sprint 2  ──▶  Sprint 3  ──▶  Sprint 4  ──▶  Sprint 5  ──▶  Sprint 6
验证红灯       补全工程       数据引擎       DL模型V1      混声检测       多维诊断       训练引导
(3天)         (3天)         (5天)         (5天)         (5天)         (2周)         (3周)
```

---

## Sprint 0 — 验证红灯可行性 🔲

> **目标**：证明 Rule-based 挤卡检测在真实音频上准确率 >70%
> **前置条件**：≥10 段测试音频（5 正常 + 5 挤卡）
> **自治度**：⚠️ 需要用户提供音频文件

### 任务清单

| # | 任务 | Agent 可自治 | 产出 |
|---|------|-------------|------|
| 0.1 | 创建 `test_audio/` 目录 + README（说明录音要求） | ✅ | 目录 + 录音指南 |
| 0.2 | 编写自动化验证脚本 `python/validate_pipeline.py` | ✅ | 批量跑 rule_detector + 生成准确率报告 |
| 0.3 | 编写音频预处理脚本（16kHz 重采样 + 归一化 + 切段） | ✅ | `python/preprocess.py` |
| 0.4 | 在用户提供的音频上跑 pipeline | ⚠️ 需要音频 | 4-panel 可视化 × N |
| 0.5 | 统计准确率 (TP/FP/FN/TN) + 生成报告 | ✅ | `reports/sprint0_accuracy.md` |
| 0.6 | 根据结果调整阈值（HNR/Energy/F0） | ✅ | 更新 Python + iOS 双端 |

### 验收标准

- [ ] Rule-based 准确率 ≥ 70%（帧级 F1-Score）
- [ ] 误报率 < 20%（正常唱歌时红灯亮的比例）
- [ ] 延迟等效测试：从输入到判决 < 100ms（Pipeline latency sim）

### 自评模板

```markdown
## Sprint 0 自评
- 准确率：___% (目标 70%)
- 主要错误模式：___
- 阈值调整记录：HNR ___ → ___，Energy ___ → ___
- 下轮调整：___
```

---

## Sprint 1 — 补全工程基础设施 🔲

> **目标**：iOS 工程可编译 + CI/CD + 自动化测试框架
> **前置条件**：Sprint 0 通过（准确率 ≥ 70%）
> **自治度**：✅ 大部分可自治

### 任务清单

| # | 任务 | 产出 |
|---|------|------|
| 1.1 | 创建 Xcode 项目配置文件（.xcodeproj 或 Package.swift） | iOS 构建配置 |
| 1.2 | 添加 Python 单元测试 `python/tests/test_dsp.py` | pytest 测试：特征提取维度/范围验证 |
| 1.3 | 添加 Python 单元测试 `python/tests/test_detector.py` | pytest 测试：已知信号→已知输出 |
| 1.4 | 编写 Swift 单元测试 `ios/Tests/DSPTests.swift` | XCTest：DSP 输出与 Python 一致性 |
| 1.5 | 创建 GitHub Actions CI workflow `.github/workflows/ci.yml` | PR 自动测试 |
| 1.6 | 添加 `CHANGELOG.md` + 语义版本化 | 版本管理 |
| 1.7 | 创建 `.env.example` + 配置管理（阈值外部化） | 配置可调 |

### 验收标准

- [ ] `pytest python/tests/ -v` 全绿
- [ ] GitHub Actions 在 PR 时自动跑测试
- [ ] iOS 工程可通过 `xcodebuild` 命令行编译（模拟器）
- [ ] 所有阈值参数有配置化入口（不再硬编码在代码中间）

---

## Sprint 2 — 数据引擎 🔲

> **目标**：建立半自动化数据标注 + 增强 pipeline，积累 ≥ 500 条标注数据
> **前置条件**：Sprint 1 通过
> **自治度**：✅ 大部分可自治（标注需人工审核）

### 任务清单

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | VocalSet 数据集下载 + 自动筛选高音段 | `data/vocalset/` (~500 clips) |
| 2.2 | 编写 Rule-based 弱标签生成器 `python/auto_labeler.py` | 自动预标注 |
| 2.3 | 编写数据增强 pipeline `python/augment.py` | 随机增益 / 噪声注入 / Pitch Shift / Time Stretch |
| 2.4 | 创建 Streamlit 标注审核 UI `python/label_ui.py` | 可视化审核+修正弱标签 |
| 2.5 | 编写 PyTorch Dataset + DataLoader `python/dataset.py` | 训练就绪的数据管道 |
| 2.6 | 数据统计报告：标签分布 / 特征分布 / 质量评估 | `reports/data_quality.md` |

### 数据增强策略（已锁定）

```python
AUGMENT_CONFIG = {
    "gain_db": [-12, +6],       # 随机增益 (必须)
    "noise_snr": [15, 40],      # 加噪 SNR 范围
    "pitch_shift": [-2, +2],    # 半音 (不改变挤卡标签)
    "time_stretch": [0.9, 1.1], # 时间拉伸 (不改变挤卡标签)
}
```

### 验收标准

- [ ] 数据集 ≥ 500 条（含增强后）
- [ ] 正负样本比例 40%-60% 范围内
- [ ] 弱标签的人工审核修正率 < 30%（即 Rule-based 预标注正确率 > 70%）
- [ ] DataLoader 可正确输出 `[82 × 32]` tensor

---

## Sprint 3 — DL 模型 V1 🔲

> **目标**：训练 Causal DWSep CNN，在测试集上超过 Rule-based 准确率
> **前置条件**：Sprint 2 通过（数据 ≥ 500 条）
> **自治度**：✅ 完全可自治

### 任务清单

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | PyTorch 模型定义 `python/model.py` | CausalTCN 4-layer, <300K params |
| 3.2 | 训练脚本 `python/train.py` | 含 Label Smoothing / 学习率调度 / 早停 |
| 3.3 | 评估脚本 `python/evaluate.py` | F1 / AUC / 混淆矩阵 / 错误分析 |
| 3.4 | CoreML 导出脚本 `python/export_coreml.py` | `.mlpackage` (Float16) |
| 3.5 | iOS CoreML 推理集成 `ios/Vibesing/Audio/SqueezeModel.swift` | 替换/并行 Rule-based |
| 3.6 | A/B 对比：DL vs Rule-based 在相同测试集上的表现 | 对比报告 |

### 模型超参数（初始值，Agent 可在自评中调整）

```python
MODEL_CONFIG = {
    "input_dim": 82,           # 80 mel + HNR + energy
    "context_frames": 32,      # 32 frames × 16ms = 512ms
    "channels": [82, 128, 128, 256, 256],
    "kernel_size": 7,
    "dilations": [1, 2, 4, 8],
    "dropout": 0.1,
    "label_smooth": 0.1,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "early_stop_patience": 10,
}
```

### 验收标准

- [ ] 测试集 F1-Score > Rule-based + 5%
- [ ] 模型参数 < 300K
- [ ] CoreML 导出成功 + iOS 端推理 < 2ms (A14+)
- [ ] 无过拟合（train/val loss 收敛且差距 < 15%）

### Agent 自评要点

```markdown
## Sprint 3 自评
- 最终测试 F1：___% (Rule-based: ___%)
- 参数量：___K
- CoreML 推理延迟：___ms
- 过拟合程度：train_loss=___ / val_loss=___
- 错误分析：主要错误模式 → ___
- 超参数调整日志：___
- 下轮建议：是否需要更多数据 / 架构调整
```

---

## Sprint 4 — 混声检测 V0.2 🔲

> **目标**：在挤卡检测之上添加胸声/混声/头声 3 级 Slider
> **前置条件**：Sprint 3 通过 (DL 模型上线)
> **自治度**：✅ 大部分可自治

### 任务清单

| # | 任务 | 产出 |
|---|------|------|
| 4.1 | 混声标签定义 + 标注指南 | 5-class 有序分类标注协议 |
| 4.2 | VocalSet + 新数据的混声标注 | ≥ 300 条混声标签 |
| 4.3 | 模型升级：2 头输出 (squeeze + mix_class) | `model.py` 更新 |
| 4.4 | 期望值 Slider 映射逻辑 | `bin_centers × softmax → 连续值` |
| 4.5 | iOS UI：胸/混/头 Slider 组件 | `MixVoiceSlider.swift` |
| 4.6 | Python 端同步：`rule_detector.py` 添加混声估计 | 基于频谱特征的 Rule-based 混声近似 |
| 4.7 | 端到端测试：录一段从胸声渐变到头声的音频，验证 Slider 变化 | 演示视频/截图 |

### 5-class 混声分类定义

```
Class 0: Full Chest   (全胸声) → slider 0.1
Class 1: Chest-Mix    (胸声偏混) → slider 0.3
Class 2: Balanced     (均衡混声) → slider 0.5
Class 3: Head-Mix     (头声偏混) → slider 0.7
Class 4: Full Head    (全头声) → slider 0.9
```

### 验收标准

- [ ] 混声 3-class F1 ≥ 0.65（合并为胸/混/头 三大类）
- [ ] Slider 在同一段音频中随声区变化平滑过渡
- [ ] 挤卡检测准确率不因多任务训练而下降（回归测试）

---

## Sprint 5 — 多维诊断引擎 V1.0 🔲

> **目标**：实现 Taxonomy 前 5 个维度的完整诊断
> **前置条件**：Sprint 4 通过 + 数据 ≥ 2000 条
> **自治度**：✅ 大部分可自治（漏气/喊叫等新标签需人工审核）

### 新增检测维度

| 维度 | 任务类型 | 输出 |
|------|---------|------|
| D1: 发声机能 | 4-class (胸声/头声/假声/混声) | Softmax |
| D2: 混声比例 | 5-class 有序 (Sprint 4 延续) | Slider |
| D3-挤卡 | 二分类 (已有) | Sigmoid |
| D3-漏气 | 二分类 | Sigmoid |
| D3-大白嗓 | 二分类 | Sigmoid |
| D3-喉位高 | 二分类 | Sigmoid |
| D4: 共鸣位置 | 4-class (喉/咽/口/头) | Softmax |
| D5: 声区断裂事件 | 二分类 (时序事件) | Sigmoid |

### 关键任务

| # | 任务 | 产出 |
|---|------|------|
| 5.1 | 双流架构实现：Mel Stream + Physics Stream | `model_v2.py` |
| 5.2 | 多任务损失权重自动调节 (Uncertainty Weighting) | 训练稳定性 |
| 5.3 | 新增特征：H1-H2, CPP (后端稳定版), 频谱斜率 | `dsp_features.py` 更新 |
| 5.4 | iOS 诊断报告页面 | `DiagnosisView.swift` |
| 5.5 | 雷达图组件 (7 维得分可视化) | `RadarChart.swift` |
| 5.6 | 病灶时间线回放 | `TimelineView.swift` |

### 验收标准

- [ ] 全维度 macro-F1 ≥ 0.65
- [ ] 模型参数 < 800K
- [ ] 诊断报告页面完整可用
- [ ] 端到端延迟仍 < 100ms

---

## Sprint 6 — 训练引导模式 🔲

> **目标**：从"诊断"升级为"教学"——App 引导用户做练声练习并实时反馈
> **前置条件**：Sprint 5 通过 + ≥ 100 TestFlight 用户反馈
> **自治度**：⚠️ 练声内容需声乐专家审核

### 关键任务

| # | 任务 | 产出 |
|---|------|------|
| 6.1 | 练习库数据结构设计 | `ExerciseLibrary.swift` |
| 6.2 | 8 个基础练习定义 (同 `vocal-exercises.html` 内容对齐) | 练习元数据 JSON |
| 6.3 | 实时引导 UI：目标音高线 + 当前音高追踪 | `PracticeView.swift` |
| 6.4 | 练习评分系统：准确度 / 稳定性 / 问题出现率 | `PracticeScorer.swift` |
| 6.5 | 训练计划生成器 (同 `plan-generator.html` 逻辑) | `PlanGenerator.swift` |
| 6.6 | 进度追踪：每日/每周报告 | `ProgressTracker.swift` |
| 6.7 | 本地通知：练习提醒 | UNUserNotificationCenter |

### 验收标准

- [ ] 用户可完成一次完整的 10 分钟练习
- [ ] 练习中实时反馈延迟 < 100ms
- [ ] 练习评分与人工评估相关系数 > 0.7
- [ ] 训练计划根据用户问题自动调整

---

## 3. Agent 自我调整协议

### 何时调整计划

| 信号 | 动作 |
|------|------|
| 准确率低于目标 5%+ | 增加 Sprint 天数 + 数据收集 |
| 准确率高于目标 10%+ | 跳过优化，提前进入下一 Sprint |
| 新的阻塞项出现 | 插入 0.5 天的 spike task |
| 用户反馈新需求 | 评估是否影响当前 Sprint，否则加入 backlog |
| 技术方案失败 | 回退到上一可用方案 + 记录失败原因 |

### 自评维度 (每轮 Sprint 结束必填)

```markdown
## Sprint N 自评表

### 定量指标
| 指标 | 目标 | 实际 | 差距 |
|------|------|------|------|
| 准确率 (F1) | ___% | ___% | ___% |
| 延迟 (ms) | <___ms | ___ms | ___ms |
| 参数量 (K) | <___K | ___K | ___K |
| 数据量 (条) | ≥___ | ___ | ___ |

### 定性评估
- 最大收获：___
- 最大风险：___
- 技术债务：___
- 用户反馈关键词：___

### 下轮调整
- 增加：___
- 减少：___
- 改变：___
- 保持：___
```

---

## 4. 技术决策备忘录

### 不可更改 (来自 copilot-instructions.md)

| # | 决策 | 理由 |
|---|------|------|
| 1 | MVP 只做挤卡 (binary) | 最小可验证闭环 |
| 2 | 16kHz everywhere | 移动端效率 + 语音频带足够 |
| 3 | Mel(80)+HNR+Energy 作为输入 | CPP 端侧不稳定, F0 仅做 mask |
| 4 | Causal DWSep CNN (TCN) | 低延迟 + 少参数 + 因果性保证实时 |
| 5 | EMA + 状态机 | 简单有效, 不过度工程 |
| 6 | 混声 5-class 有序分类 | 比回归更稳定, 有序约束提升泛化 |

### 可调整 (Agent 有权在范围内优化)

| 参数 | 当前值 | 允许范围 | 调整条件 |
|------|--------|---------|---------|
| HNR 阈值 | 12 dB | 8-16 dB | Sprint 0 验证结果 |
| EMA α | 0.4 | 0.2-0.6 | UI 闪烁 vs 响应速度 |
| 触发帧数 | 6 (96ms) | 4-10 | 误报率 vs 延迟 |
| 清除帧数 | 10 (160ms) | 6-16 | 跟踪灵敏度 |
| 开启阈值 | 0.55 | 0.45-0.65 | Precision/Recall 平衡 |
| 关闭阈值 | 0.30 | 0.20-0.40 | 同上 |
| 模型通道数 | [128,128,256,256] | [64-512] | 模型容量 vs 延迟 |
| 学习率 | 1e-3 | 1e-4 to 5e-3 | 收敛情况 |
| Label Smoothing ε | 0.1 | 0.05-0.2 | 标签噪声水平 |

---

## 5. 风险登记表

| ID | 风险 | 概率 | 影响 | 缓解策略 |
|----|------|------|------|---------|
| R1 | 测试音频不足，准确率无法验证 | 高 | 高 | 提供录音指南 + 用 VocalSet 替代验证 |
| R2 | Rule-based 准确率 < 60% | 中 | 高 | 调阈值 → 如仍不够则直接进 DL (跳到 Sprint 3) |
| R3 | 没有 Mac 无法编译 iOS | 高 | 中 | Swift 代码通过 Linux Swift 语法检查; iOS 部分延后 |
| R4 | DL 模型过拟合 (数据太少) | 高 | 高 | 激进数据增强 + 小模型 + DropOut + 早停 |
| R5 | CoreML 导出失败 | 低 | 中 | 保留 ONNX 备选路径 |
| R6 | 端侧推理延迟超标 | 低 | 高 | 减通道数 / 减层数 / 量化到 Int8 |
| R7 | 混声标签标注不一致 | 高 | 中 | 制定严格标注协议 + 多人标注取多数 |

---

## 6. 文件产出清单 (累积)

```
Sprint 0:
  python/preprocess.py           # 音频预处理
  python/validate_pipeline.py    # 批量验证脚本
  test_audio/README.md           # 录音指南
  reports/sprint0_accuracy.md    # 准确率报告

Sprint 1:
  python/tests/test_dsp.py       # DSP 单元测试
  python/tests/test_detector.py  # 检测器单元测试
  ios/Tests/DSPTests.swift       # iOS 单元测试
  .github/workflows/ci.yml      # CI pipeline
  CHANGELOG.md                   # 版本变更日志

Sprint 2:
  python/auto_labeler.py         # 弱标签生成器
  python/augment.py              # 数据增强
  python/label_ui.py             # Streamlit 标注 UI
  python/dataset.py              # PyTorch Dataset
  reports/data_quality.md        # 数据质量报告

Sprint 3:
  python/model.py                # Causal TCN 模型
  python/train.py                # 训练脚本
  python/evaluate.py             # 评估脚本
  python/export_coreml.py        # CoreML 导出
  ios/.../SqueezeModel.swift     # iOS CoreML 推理

Sprint 4:
  ios/.../MixVoiceSlider.swift   # 混声 Slider UI
  python/mix_labels.py           # 混声标注工具

Sprint 5:
  python/model_v2.py             # 双流多任务模型
  ios/.../DiagnosisView.swift    # 诊断报告页
  ios/.../RadarChart.swift       # 雷达图
  ios/.../TimelineView.swift     # 时间线回放

Sprint 6:
  ios/.../PracticeView.swift     # 练习引导 UI
  ios/.../PracticeScorer.swift   # 练习评分
  ios/.../PlanGenerator.swift    # 计划生成
  ios/.../ProgressTracker.swift  # 进度追踪
  ios/.../ExerciseLibrary.swift  # 练习库
```

---

## 7. Agent 冷启动指令

如果 Agent 是全新会话（无历史上下文），按以下步骤冷启动：

```
1. 读取文件：
   - docs/AGENT_ITERATION_PLAN.md (本文档)
   - .github/copilot-instructions.md (技术约束)
   - docs/AI_AGENT_GUIDE.md (项目结构)

2. 扫描当前状态：
   - ls python/ ios/ test_audio/ reports/ → 确认已有文件
   - 检查本文档中各 Sprint 的 ✅/🔲/❌ 标记
   - git log --oneline -10 → 确认最近活动

3. 确定当前 Sprint：
   - 找到第一个状态为 🔲 的 Sprint
   - 检查其前置条件是否满足
   - 如果满足 → 开始执行
   - 如果不满足 → 向用户报告阻塞项

4. 执行当前 Sprint 的任务清单

5. Sprint 结束后：
   - 填写自评表
   - 更新本文档的进度标记
   - 提出下轮调整建议
   - 询问用户确认后进入下一 Sprint
```

---

## 8. 用户交互节点

Agent 在以下节点**必须**暂停并征求用户意见：

| 节点 | 原因 |
|------|------|
| Sprint 0 开始前 | 需要用户提供测试音频 |
| Sprint 0 结束后 | 需要用户确认准确率是否可接受 |
| 阈值调整后 | 需要用户主观评估"红灯时间感受" |
| DL 模型训练完成后 | 需要用户在真机上体验 DL vs Rule-based |
| 每个 Sprint 结束 | 需要用户确认进入下一 Sprint |
| 混声标注阶段 | 需要声乐专家审核标签 |
| 训练引导内容 | 需要声乐专家审核练习正确性 |
| 任何破坏性操作 | 删除代码/重写架构/合并分支 |

---

## 9. 终态愿景

当 Sprint 6 全部完成后，Vibesing 将达到：

```
✅ 实时 7 维发声诊断 (< 100ms 延迟)
✅ 全离线端侧推理 (< 1M 参数)
✅ 交互式练声引导 + 实时反馈
✅ 个性化训练计划自动生成
✅ 可量化进步追踪 (日报/周报)
✅ TestFlight 公开测试 (500+ 用户)
```

**Agent 的使命**：每一轮迭代让 Vibesing 离"每个人都有高音，只是没有觉醒"更近一步。

---

*本文档由 Agent 维护。每次 Sprint 结束后自动更新状态标记和自评。*

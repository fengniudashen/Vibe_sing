# Vibesing — 自动剪片技术路线 (Automated Demonstration-Clip Mining)

> 从 YouTube/B 站等海量声乐教学视频中，**全自动、零幻觉** 地提取特定发声技巧（强混 / 弱混 / 咽音 / 哼鸣 / 假声 …）的 3 秒标准示范，用于 Vibesing 声学模型训练集。

---

## 1. 业务问题与边界

| 维度 | 说明 |
|---|---|
| 输入 | 任意时长的 mp4/mkv 视频（可含画面、字幕、教师讲解、示范片段） |
| 输出 | 按技巧分类的 **3 秒** mp4 + 16kHz mono WAV 切片，附 metadata |
| 准确率目标 | 误召率 < 1%（训练集容不下噪声样本） |
| 召回率目标 | ≥ 60%（漏召可以靠扩大视频源补） |
| 硬件 | 本地 RTX 5060 8GB；云端只有大文本 LLM（无 VLM 额度） |
| 离线性 | 视觉/音频/ASR/VAD **全部本地**；只把极简 JSON 发云端裁决 |

### 三大典型陷阱

1. **隐性引导**：老师不开口说"下面是强混"，只在画面打 0.5s 花字。
2. **非词汇盲区**：哼鸣、长拖音、纯气流——Whisper 抓不到，时间轴会断。
3. **VAD 欺骗**：咳嗽/呼吸/清嗓子也是高能量，传统 VAD 会误判。

---

## 2. 总体架构（云端 - 本地混合双打）

```
┌──────────────────────────── 本地 RTX 5060 ────────────────────────────┐
│                                                                      │
│   ┌──────── Layer 1: 多模态特征提取 (并行) ────────┐                 │
│   │                                                │                 │
│   │  视觉轨 OCR ─┐                                 │                 │
│   │             ├─→ 100ms Tick 统一时间轴 → 候选 │                 │
│   │  语音轨 ASR ─┤                                 │                 │
│   │             │                                  │                 │
│   │  声学轨 VAD+f0+HNR+flatness ─┘                 │                 │
│   └────────────────────────────────────────────────┘                 │
│                          │                                          │
│                          ▼                                          │
│   ┌────── Stage 1: 候选构造 + Tick-IoU 去重 ──────┐                 │
│   │   双触发器 (OCR ∪ ASR) + 声学闸门预筛         │                 │
│   └────────────────────────────────────────────────┘                 │
│                          │                                          │
│                          ▼ (极简 JSON, < 2KB)                       │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌────────────── 云端 LLM (DeepSeek / MiniMax …) ──────────────────────┐
│                                                                      │
│   Stage 2: "无情法官" 文本裁决                                       │
│     • Evidence Binding：时间戳必须原样复制                            │
│     • Pydantic 强校验：越界即丢弃，绝不重试                           │
│     • 输出 ACCEPT/REJECT + chosen_center_tick                         │
│                                                                      │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────── 本地 RTX 5060 ────────────────────────────┐
│   Stage 3: ffmpeg + h264_nvenc 极速切片                            │
│     • 中心点 ± 1.5s → 3s mp4 (NVENC) + 16kHz mono wav              │
│     • 命名：<technique>_<video_id>_<idx>.{mp4,wav}                 │
│     • metadata.jsonl：时间戳、quality_score、verdict                │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键技术决策

### 3.1 时间量化：100ms Tick
- 所有模态（OCR/ASR word/VAD）的时间戳统一量化为 `tick = round(sec / 0.1)`
- 优点：跨模态 join 不会浮点漂移；LLM 输入用 int 不用 float，token 更省、更不易幻觉
- 损失：对边界精度损失 < 50ms，对 4s 切片完全可忽略

### 3.2 双触发器（OCR ∪ ASR）
| 触发源 | 解决什么 | 阈值 |
|---|---|---|
| OCR 花字关键词 | "老师只打字不开口" | 模糊匹配 ≥ 85，conf ≥ 0.4 |
| ASR 引导词 + 技巧词组合 | "老师只口播不打字" | 引导词后 3s 内出现技巧词 |

### 3.3 反"伪发声"声学闸门
对每个 VAD block 同时检查 5 个指标，**全部通过**才算 `is_valid_demo`：

| 指标 | 阈值 | 排除什么 |
|---|---|---|
| `avg_rms` | ≥ 0.05 | 静音/极弱信号 |
| `avg_pitch_hz` | 80 ~ 1100 Hz | 噪音/超出人声范围 |
| `pitch_stability` | ≥ 0.6 (=1−CV) | 不稳定的"喊叫"、咳嗽 |
| `avg_hnr_db` | ≥ 8 dB | 喘气、清嗓子（无谐波） |
| `spectral_flatness` | ≤ 0.4 | 白噪/呼吸 |

> **HNR 的实现与 Vibesing 主项目 `python/dsp_features.py` 共享同一套定义**——保证训练集特征分布与 iOS 推理端 100% 一致。

### 3.4 LLM 防幻觉的 5 道闸
1. **Evidence Binding**：所有时间戳必须从输入 JSON 原样复制，不允许生成。
2. **Constraint Field**：`must_pick_center_within` 列出唯一合法区间。
3. **Pydantic Schema**：本地强类型校验，结构错误直接 `None`。
4. **Range Check**：`chosen_center_tick` 必须落在某个 valid VAD block 内，越界丢弃。
5. **No Retry**：失败永不重试——重试只会让模型"取悦你"，反而引入新幻觉。

### 3.5 IoU 去重
同一花字会持续 1-2 秒（30-60 OCR 帧），如不去重会触发数十次 LLM 调用：

```
按 technique 分桶 → 按窗口起点排序 → 相邻 IoU ≥ 0.5 合并 → 保留 quality_hint 高者
```

效果：LLM 调用量减少 ~80%。

### 3.6 质量分（用于 top-K 筛选）
```
quality_score = 0.40 * HNR_norm
              + 0.30 * pitch_stability
              + 0.20 * duration_norm
              + 0.10 * RMS_norm
```
训练时按 `(technique, quality_score)` 排序，每个技巧取 top-K 即得"极品示范集"。

### 3.7 转声示范处理（passaggio detection）

**问题**：老师做"胸声 → 假声"示范时，3 秒窗口若以稳定假声段为中心，会**完整漏掉胸声段**和**最有价值的换声瞬间**。

**解决方案**——三步走：

#### Step A：声学层自动检测转声点
对每个 VAD block 内做滑窗 f0 比较：
```
对每帧 i：
  before_median_f0 = median(f0[i-W-pre : i-W])  # 前 0.5s 中位数
  after_median_f0  = median(f0[i+W   : i+W+post]) # 后 0.5s 中位数
  semitones = |12 * log2(after / before)|
  if semitones >= 5 半音: → 标记为 transition (passaggio)
```
每个 transition 记录：`{tick, semitone_jump, direction: "up"|"down", f0_before, f0_after}`

#### Step B：识别转声示范类型
当 OCR/ASR 命中 `TRANSITION_KEYWORDS`（"胸转假"、"换声"、"passaggio" 等），生成 **`transition_demo`** 类型候选，**每个 passaggio 一个候选**。其 `must_pick_center_within` = `[transition_tick - 5, transition_tick + 5]`（即 ±0.5s 紧带）。

#### Step C：LLM 强约束 + 3s 切片对齐
LLM 必须从 `transitions[]` 中选一个 tick 作为 `chosen_center_tick`，本地切片器以此为中心 ±1.5s：

```
   |←—— 1.5s ——→|←—— 1.5s ——→|
   ─────胸声─────●─────假声─────
                ↑
          passaggio (chosen_center)
```
3 秒窗口同时包含**转前稳定段、换声瞬间、转后稳定段**——这正是训练"模式切换检测"模型最需要的样本。

#### 转声技巧词表（`config.TRANSITION_KEYWORDS`）
| 技巧名 | 关键词 |
|---|---|
| 胸转假 | 胸声转假声、胸转假、由胸到头、chest to falsetto |
| 头转胸 | 头声转胸声、头转胸、falsetto to chest |
| 转混 | 转混声、换声、换声点、passaggio、break、flip |
| 强转弱 | 强混转弱混、belt to mix |

---

## 4. 数据流 Schema

### 4.1 LLM 输入（Stage-2 payload）
```json
{
  "task": "adjudicate_vocal_demonstration",
  "candidate_id": "a3f1b9c2",
  "target_technique": "强混",
  "trigger_source": "ocr",
  "ocr_trigger": {
    "time_tick": 1250, "text": "强混示范",
    "matched_keyword": "强混", "technique": "强混", "confidence": 0.91
  },
  "asr_trigger": null,
  "subsequent_speech": [
    {"start_tick": 1255, "end_tick": 1268, "text": "听这个"}
  ],
  "subsequent_vad_blocks": [
    {
      "start_tick": 1270, "end_tick": 1305,
      "acoustic_features": {
        "avg_rms": 0.21, "avg_pitch_hz": 412.3,
        "pitch_stability": 0.78, "avg_hnr_db": 14.2,
        "avg_spectral_flatness": 0.12, "is_valid_demo": true
      }
    }
  ],
  "constraints": {
    "must_pick_center_within": [[1270, 1305]]
  }
}
```

### 4.2 LLM 输出（Stage-2 verdict）
```json
{
  "candidate_id": "a3f1b9c2",
  "decision": "ACCEPT",
  "reject_reason": null,
  "chosen_center_tick": 1287,
  "chosen_vad_block": [1270, 1305],
  "confidence": 0.82,
  "notes": "高HNR稳定混声示范"
}
```

---

## 5. 输出目录布局（按技巧自动建子目录）

```
pipeline_out/
├── _shared/                       # 跨技巧复用，跑一次
│   ├── audio_16k.wav
│   ├── ocr.json
│   ├── asr.json
│   └── vad.json                   # 含声学特征
├── 强混/
│   ├── candidates.jsonl           # Stage-1 输出
│   ├── verdicts.jsonl             # Stage-2 输出
│   └── slices/
│       ├── 强混_<vid>_001.mp4
│       ├── 强混_<vid>_001.wav
│       └── metadata.jsonl
├── 弱混/
│   └── ...
└── 咽音/
    └── ...
```

> 用户在 `--techniques 强混 弱混 咽音` 指定哪些技巧，就自动创建哪些子目录。

---

## 6. 模块清单

| 模块 | 文件 | 职责 |
|---|---|---|
| 全局配置 | [`config.py`](../python/dataset_pipeline/config.py) | 阈值、关键词表、Tick 量化 |
| 数据契约 | [`schemas.py`](../python/dataset_pipeline/schemas.py) | 所有跨模块 dataclass |
| 音频 IO | [`audio_io.py`](../python/dataset_pipeline/audio_io.py) | NVDEC 抽 16kHz wav |
| 视觉轨 | [`visual_track.py`](../python/dataset_pipeline/visual_track.py) | EasyOCR(GPU) + PySceneDetect + 模糊匹配 |
| 语音轨 | [`speech_track.py`](../python/dataset_pipeline/speech_track.py) | Faster-Whisper(fp16) + ASR 双触发 |
| 声学轨 | [`acoustic_track.py`](../python/dataset_pipeline/acoustic_track.py) | Silero VAD + librosa + 声学闸门 |
| 候选构造 | [`candidate_builder.py`](../python/dataset_pipeline/candidate_builder.py) | Stage-1 + IoU 去重 |
| Orchestrator | [`run.py`](../python/dataset_pipeline/run.py) | CLI 串联 |
| LLM Prompt | [`STAGE2_LLM_PROMPT.md`](../python/dataset_pipeline/STAGE2_LLM_PROMPT.md) | 0 幻觉裁决器 + Pydantic 校验 |

---

## 7. 性能预算（单视频 60 分钟，RTX 5060 8GB）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 抽音频 | ~30s | NVDEC + ffmpeg |
| OCR (3fps + scene-detect) | ~6 min | EasyOCR GPU，预筛后实际 OCR ~30% 帧 |
| ASR Whisper-small fp16 | ~3 min | RTF ≈ 0.05 |
| Silero VAD + librosa 特征 | ~1 min | CPU 主导 |
| Stage-1 候选构造 | < 5s | 纯 Python |
| Stage-2 LLM 裁决 | ~30s / 100 候选 | 50 TPS |
| Stage-3 NVENC 切片 | ~2s / 切片 | 硬编码 |
| **合计** | **~10-12 min** | 端到端 |

---

## 8. 命令行用法

```bash
# 安装依赖（在 ../requirements.txt 之上）
pip install -r python/dataset_pipeline/requirements.txt

# 模式 A：自主发现（推荐）——老师示范什么，就建什么文件夹
python -m dataset_pipeline.run \
    --video data/teach_001.mp4 \
    --workdir pipeline_out/teach_001

# 模式 B：手动指定目标技巧（只建指定的文件夹）
python -m dataset_pipeline.run \
    --video data/teach_001.mp4 \
    --techniques 强混 弱混 胸转假 \
    --workdir pipeline_out/teach_001
```

**自主发现模式**扫描所有 OCR + ASR 命中的技巧关键词（`TECHNIQUE_KEYWORDS` ∪ `TRANSITION_KEYWORDS`），取并集作为 `target_techniques`。视频里老师示范了什么就自动建什么子目录，不需要预先声明。

---

## 9. 路线图

| 阶段 | 状态 | 内容 |
|---|---|---|
| **M1 — Layer 1 + Stage 1 + Prompt** | ✅ Done | 本文档对应代码 |
| M2 — Stage 2 LLM 调用层 (`llm_judge.py`) | ⏳ Next | MiniMax 并发 + Pydantic 校验 |
| M3 — Stage 3 NVENC 切片层 (`slicer.py`) | ⏳ Next | ffmpeg-python h264_nvenc |
| M4 — 端到端集成 + 单元测试 | 📋 Plan | mock 视频走通全链路 |
| M5 — Web 仪表盘 (人工 review 兜底) | 📋 Plan | 可视化候选 + ACCEPT/REJECT |
| M6 — 多视频批跑 + 训练集导出 | 📋 Plan | manifest.csv，按 quality top-K 筛 |

---

## 10. 与 Vibesing 主项目的关系

| 共享点 | 说明 |
|---|---|
| **采样率** | 全链路 16kHz mono，与 iOS `AudioCaptureEngine.swift` 一致 |
| **特征定义** | HNR / F0 / Energy 沿用 `python/dsp_features.py`，模型推理时无 train-serve gap |
| **数据消费方** | 本流水线产物 → 喂给 V1.0 的 Causal DWSep CNN 训练（`docs/Vibesing_MVP_V0.1_Engineering_Spec.md`） |
| **不复用** | 检测算法（`rule_detector.py` 是推理时用的二分类器，本流水线是采集训练数据） |

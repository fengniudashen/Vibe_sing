"""
Vibesing Dataset Pipeline - Global Config
========================================
所有时间统一用 100ms Tick (1 Tick = 0.1s)。
所有阈值集中在此，避免散落在各处。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# =========================================================
# 时间量化
# =========================================================
TICK_MS = 100                       # 1 Tick = 100ms
TICK_SEC = TICK_MS / 1000.0


def sec_to_tick(sec: float) -> int:
    return int(round(sec / TICK_SEC))


def tick_to_sec(tick: int) -> float:
    return tick * TICK_SEC


# =========================================================
# 目标技巧关键词表 (OCR + ASR 共用，双触发器)
# =========================================================
TECHNIQUE_KEYWORDS: Dict[str, List[str]] = {
    "强混": ["强混", "强混声", "belting mix", "chest mix"],
    "弱混": ["弱混", "弱混声", "head mix", "soft mix"],
    "咽音": ["咽音", "pharyngeal", "yan yin"],
    "哼鸣": ["哼鸣", "humming", "lip trill"],
    "假声": ["假声", "falsetto"],
}

# ASR 引导词：与技巧词组合时也触发 (e.g. "听这个" + "强混")
ASR_DEMO_TRIGGERS = ["听这个", "像这样", "示范", "demo", "听好"]


# =========================================================
# 视频/音频
# =========================================================
AUDIO_SAMPLE_RATE = 16000           # 与 Vibesing iOS/主算法对齐
AUDIO_CHANNELS = 1
OCR_FPS = 3                         # 每秒抽 3 帧做 OCR
SCENE_CHANGE_THRESHOLD = 0.30       # PySceneDetect content-detector


# =========================================================
# Whisper
# =========================================================
@dataclass
class WhisperCfg:
    model_size: str = "small"        # 8GB 显存够 small/base，medium 紧张
    device: str = "cuda"
    compute_type: str = "float16"    # 5060 支持
    language: str = "zh"
    vad_filter: bool = True          # faster-whisper 自带 vad
    word_timestamps: bool = True


# =========================================================
# Silero VAD
# =========================================================
@dataclass
class VADCfg:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 50


# =========================================================
# 声学有效性过滤 (反"伪发声")
# =========================================================
@dataclass
class AcousticGate:
    min_avg_rms: float = 0.05            # 0~1 归一化后
    min_avg_pitch_hz: float = 80.0
    max_avg_pitch_hz: float = 1100.0
    min_pitch_stability: float = 0.6     # 1 - cv(f0)，越大越稳
    min_hnr_db: float = 8.0              # < 8 视为噪音/挤卡 (非标准示范)
    max_spectral_flatness: float = 0.4   # > 0.4 偏白噪


# =========================================================
# 候选事件窗口
# =========================================================
@dataclass
class CandidateCfg:
    lookahead_ticks: int = 30            # OCR 触发后向后看 3 秒 (30 ticks)
    lookback_ticks: int = 5              # 也回看 0.5 秒 (老师可能先说后打字)
    iou_merge_threshold: float = 0.5     # 候选 IoU 去重阈值
    min_vad_blocks: int = 1              # 候选必须至少包含 1 个 VAD 块
    min_demo_duration_ticks: int = 10    # 示范本身 ≥ 1 秒才有意义


# =========================================================
# 切片输出
# =========================================================
@dataclass
class SliceCfg:
    pad_ticks: int = 20                  # 中心点前后各 2.0 秒 → 共 4s 切片
    out_video_codec: str = "h264_nvenc"  # 5060 NVENC
    out_audio_codec: str = "pcm_s16le"   # WAV
    out_audio_sr: int = 16000


# =========================================================
# 路径
# =========================================================
@dataclass
class Paths:
    """目录布局（按技巧自动建子目录）：
        pipeline_out/
        ├── _shared/                # 与具体技巧无关的中间产物
        │   ├── audio_16k.wav
        │   ├── ocr.json
        │   ├── asr.json
        │   └── vad.json
        ├── 强混/
        │   ├── candidates.jsonl
        │   ├── verdicts.jsonl
        │   └── slices/
        │       ├── 强混_<vid>_<idx>.mp4
        │       └── 强混_<vid>_<idx>.wav
        ├── 弱混/
        │   └── ...
    """
    work_dir: Path = field(default_factory=lambda: Path("./pipeline_out"))

    # ---- 共享（多技巧复用同一份）----
    @property
    def shared_dir(self) -> Path:
        return self.work_dir / "_shared"

    @property
    def audio_wav(self) -> Path:
        return self.shared_dir / "audio_16k.wav"

    @property
    def ocr_json(self) -> Path:
        return self.shared_dir / "ocr.json"

    @property
    def asr_json(self) -> Path:
        return self.shared_dir / "asr.json"

    @property
    def vad_json(self) -> Path:
        return self.shared_dir / "vad.json"

    # ---- 按技巧分目录 ----
    def technique_dir(self, technique: str) -> Path:
        return self.work_dir / technique

    def candidates_jsonl(self, technique: str) -> Path:
        return self.technique_dir(technique) / "candidates.jsonl"

    def verdicts_jsonl(self, technique: str) -> Path:
        return self.technique_dir(technique) / "verdicts.jsonl"

    def slices_dir(self, technique: str) -> Path:
        return self.technique_dir(technique) / "slices"


# =========================================================
# 顶层配置
# =========================================================
@dataclass
class PipelineConfig:
    target_techniques: List[str] = field(default_factory=lambda: ["强混"])
    whisper: WhisperCfg = field(default_factory=WhisperCfg)
    vad: VADCfg = field(default_factory=VADCfg)
    gate: AcousticGate = field(default_factory=AcousticGate)
    candidate: CandidateCfg = field(default_factory=CandidateCfg)
    slicing: SliceCfg = field(default_factory=SliceCfg)
    paths: Paths = field(default_factory=Paths)

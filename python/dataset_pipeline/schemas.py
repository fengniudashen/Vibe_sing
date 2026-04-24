"""
Schema definitions — 所有跨模块/落盘 JSON 结构都在这里。
统一时间单位：tick (100ms)。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any


@dataclass
class OCRHit:
    time_tick: int
    text: str
    matched_keyword: str
    technique: str
    confidence: float
    bbox: Optional[List[int]] = None     # [x1,y1,x2,y2]


@dataclass
class ASRWord:
    start_tick: int
    end_tick: int
    text: str


@dataclass
class ASRHit:
    """ASR 命中触发词的事件 (e.g. '听这个强混')"""
    time_tick: int
    text: str
    technique: str
    matched_keyword: str


@dataclass
class VADBlock:
    start_tick: int
    end_tick: int
    # 由 acoustic_features.py 填充
    avg_rms: float = 0.0
    avg_pitch_hz: float = 0.0
    pitch_stability: float = 0.0         # 1 - cv(f0_voiced)
    avg_hnr_db: float = 0.0
    avg_spectral_flatness: float = 0.0
    is_valid_demo: bool = False          # 通过 AcousticGate
    quality_score: float = 0.0
    # 转声点：块内检测到的 passaggio tick 列表 (可能有多个)
    # 每个元素: {"tick": int, "semitone_jump": float, "direction": "up"|"down"}
    transitions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CandidateEvent:
    """发给 LLM 裁决的最小单元。"""
    candidate_id: str
    target_technique: str
    trigger_source: str                  # "ocr" | "asr" | "ocr+asr"
    candidate_type: str = "steady_demo"   # "steady_demo" | "transition_demo"
    # 转声示范专用：chosen_center_tick 必须 ±tol 范围内
    transition_anchor_tick: Optional[int] = None
    transition_tolerance_ticks: int = 5
    ocr_trigger: Optional[OCRHit] = None
    asr_trigger: Optional[ASRHit] = None
    subsequent_speech: List[ASRWord] = field(default_factory=list)
    subsequent_vad_blocks: List[VADBlock] = field(default_factory=list)
    # 本地预筛后给 LLM 的 hint，但 LLM 不能直接抄
    local_quality_hint: float = 0.0
    window_start_tick: int = 0
    window_end_tick: int = 0

    def to_llm_payload(self) -> Dict[str, Any]:
        """精简版给 LLM，去掉冗余字段。"""
        # 构造 must_pick_center_within 约束
        if self.candidate_type == "transition_demo" and self.transition_anchor_tick is not None:
            tol = self.transition_tolerance_ticks
            must_pick = [[self.transition_anchor_tick - tol,
                          self.transition_anchor_tick + tol]]
        else:
            must_pick = [
                [b.start_tick, b.end_tick]
                for b in self.subsequent_vad_blocks if b.is_valid_demo
            ]
        return {
            "task": "adjudicate_vocal_demonstration",
            "candidate_id": self.candidate_id,
            "target_technique": self.target_technique,
            "candidate_type": self.candidate_type,
            "trigger_source": self.trigger_source,
            "transition_anchor_tick": self.transition_anchor_tick,
            "ocr_trigger": asdict(self.ocr_trigger) if self.ocr_trigger else None,
            "asr_trigger": asdict(self.asr_trigger) if self.asr_trigger else None,
            "subsequent_speech": [asdict(w) for w in self.subsequent_speech],
            "subsequent_vad_blocks": [
                {
                    "start_tick": b.start_tick,
                    "end_tick": b.end_tick,
                    "acoustic_features": {
                        "avg_rms": round(b.avg_rms, 3),
                        "avg_pitch_hz": round(b.avg_pitch_hz, 1),
                        "pitch_stability": round(b.pitch_stability, 3),
                        "avg_hnr_db": round(b.avg_hnr_db, 2),
                        "avg_spectral_flatness": round(b.avg_spectral_flatness, 3),
                        "is_valid_demo": b.is_valid_demo,
                    },
                    "transitions": b.transitions,
                }
                for b in self.subsequent_vad_blocks
            ],
            "constraints": {
                "must_pick_center_within": must_pick,
                "slice_total_seconds": 3.0,
                "slice_pad_ticks_each_side": 15,
            },
        }


@dataclass
class LLMVerdict:
    """LLM 返回的裁决结果（也是 Stage-3 切片的输入）。"""
    candidate_id: str
    decision: str                        # "ACCEPT" | "REJECT"
    reject_reason: Optional[str] = None  # 枚举
    chosen_center_tick: Optional[int] = None
    chosen_vad_block: Optional[List[int]] = None  # [start_tick, end_tick]
    confidence: float = 0.0
    notes: str = ""

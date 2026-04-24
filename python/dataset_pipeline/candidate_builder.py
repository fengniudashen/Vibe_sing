"""
Stage 1 — 候选事件构造 + IoU 去重。
输入：OCRHits, ASRHits, ASRWords, VADBlocks
输出：CandidateEvent[]  (准备发给 Stage-2 LLM 裁决)
"""
from __future__ import annotations
from typing import List, Iterable
import logging
import uuid

from .config import CandidateCfg
from .schemas import OCRHit, ASRHit, ASRWord, VADBlock, CandidateEvent

log = logging.getLogger(__name__)


def _slice_words_in(words: List[ASRWord], a: int, b: int) -> List[ASRWord]:
    return [w for w in words if w.end_tick >= a and w.start_tick <= b]


def _slice_vad_in(blocks: List[VADBlock], a: int, b: int) -> List[VADBlock]:
    return [vb for vb in blocks if vb.end_tick >= a and vb.start_tick <= b]


def _iou(a: tuple, b: tuple) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def build_candidates(
    ocr_hits: List[OCRHit],
    asr_hits: List[ASRHit],
    asr_words: List[ASRWord],
    vad_blocks: List[VADBlock],
    target_techniques: Iterable[str],
    cfg: CandidateCfg,
) -> List[CandidateEvent]:
    target_set = set(target_techniques)
    candidates: List[CandidateEvent] = []

    # ---- 1. OCR 触发 ----
    for hit in ocr_hits:
        if hit.technique not in target_set:
            continue
        c = _make_candidate_from_anchor(
            anchor_tick=hit.time_tick,
            tech=hit.technique,
            cfg=cfg,
            asr_words=asr_words,
            vad_blocks=vad_blocks,
            ocr=hit, asr=None, source="ocr",
        )
        if c is not None:
            candidates.append(c)

    # ---- 2. ASR 触发 ----
    for hit in asr_hits:
        if hit.technique not in target_set:
            continue
        c = _make_candidate_from_anchor(
            anchor_tick=hit.time_tick,
            tech=hit.technique,
            cfg=cfg,
            asr_words=asr_words,
            vad_blocks=vad_blocks,
            ocr=None, asr=hit, source="asr",
        )
        if c is not None:
            candidates.append(c)

    # ---- 3. IoU 去重 (按 technique 分桶) ----
    deduped: List[CandidateEvent] = []
    by_tech = {}
    for c in candidates:
        by_tech.setdefault(c.target_technique, []).append(c)
    for tech, group in by_tech.items():
        # 按窗口起点排序
        group.sort(key=lambda x: x.window_start_tick)
        merged: List[CandidateEvent] = []
        for c in group:
            if merged and _iou(
                (merged[-1].window_start_tick, merged[-1].window_end_tick),
                (c.window_start_tick, c.window_end_tick),
            ) >= cfg.iou_merge_threshold:
                # 合并：保留 quality_hint 高者，但合并 trigger_source
                prev = merged[-1]
                if c.local_quality_hint > prev.local_quality_hint:
                    c.trigger_source = "+".join(sorted({prev.trigger_source, c.trigger_source}))
                    merged[-1] = c
                else:
                    prev.trigger_source = "+".join(sorted({prev.trigger_source, c.trigger_source}))
            else:
                merged.append(c)
        deduped.extend(merged)

    log.info("candidates: raw=%d, after IoU-dedupe=%d", len(candidates), len(deduped))
    return deduped


def _make_candidate_from_anchor(
    anchor_tick: int, tech: str, cfg: CandidateCfg,
    asr_words: List[ASRWord], vad_blocks: List[VADBlock],
    ocr, asr, source: str,
):
    win_a = anchor_tick - cfg.lookback_ticks
    win_b = anchor_tick + cfg.lookahead_ticks

    sub_words = _slice_words_in(asr_words, win_a, win_b)
    sub_vads = _slice_vad_in(vad_blocks, win_a, win_b)

    # 必须至少有 1 个 valid demo VAD block
    valid = [b for b in sub_vads if b.is_valid_demo
             and (b.end_tick - b.start_tick) >= cfg.min_demo_duration_ticks]
    if len(valid) < cfg.min_vad_blocks:
        return None

    quality_hint = max((b.quality_score for b in valid), default=0.0)
    return CandidateEvent(
        candidate_id=str(uuid.uuid4())[:8],
        target_technique=tech,
        trigger_source=source,
        ocr_trigger=ocr,
        asr_trigger=asr,
        subsequent_speech=sub_words,
        subsequent_vad_blocks=sub_vads,
        local_quality_hint=quality_hint,
        window_start_tick=win_a,
        window_end_tick=win_b,
    )

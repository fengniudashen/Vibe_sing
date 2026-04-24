"""
Speech track — Faster-Whisper word-level timestamps + ASR 触发器。
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import json
import logging

from .config import WhisperCfg, TECHNIQUE_KEYWORDS, TRANSITION_KEYWORDS, ASR_DEMO_TRIGGERS, sec_to_tick
from .schemas import ASRWord, ASRHit

log = logging.getLogger(__name__)


def transcribe(audio_wav: Path, cfg: WhisperCfg, out_json: Path) -> List[ASRWord]:
    if out_json.exists():
        log.info("asr cache hit: %s", out_json)
        return [ASRWord(**d) for d in json.loads(out_json.read_text("utf-8"))]

    from faster_whisper import WhisperModel  # type: ignore
    log.info("loading whisper %s on %s/%s", cfg.model_size, cfg.device, cfg.compute_type)
    model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)

    segments, info = model.transcribe(
        str(audio_wav),
        language=cfg.language,
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        beam_size=5,
    )
    words: List[ASRWord] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.word is None or w.start is None or w.end is None:
                continue
            words.append(ASRWord(
                start_tick=sec_to_tick(w.start),
                end_tick=sec_to_tick(w.end),
                text=w.word.strip(),
            ))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([w.__dict__ for w in words], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("ASR done: %d words → %s", len(words), out_json)
    return words


def find_asr_triggers(words: List[ASRWord], window_ticks: int = 30) -> List[ASRHit]:
    """在 ASR 文本中找"引导词 + 技巧词"组合触发。
    e.g. '听这个' 后面 3 秒内出现 '强混' → 触发。
    """
    hits: List[ASRHit] = []
    if not words:
        return hits

    # 拼成 (tick, text) 的滚动窗口
    for i, w in enumerate(words):
        if not any(t in w.text for t in ASR_DEMO_TRIGGERS):
            continue
        # 在 i 之后 window_ticks 内寻找技巧词 (含转声词)
        anchor_tick = w.start_tick
        all_kw_groups = list(TECHNIQUE_KEYWORDS.items()) + list(TRANSITION_KEYWORDS.items())
        for j in range(i + 1, len(words)):
            if words[j].start_tick - anchor_tick > window_ticks:
                break
            for tech, kws in all_kw_groups:
                for kw in kws:
                    if kw in words[j].text:
                        hits.append(ASRHit(
                            time_tick=words[j].start_tick,
                            text=f"{w.text}...{words[j].text}",
                            technique=tech,
                            matched_keyword=kw,
                        ))
    log.info("ASR triggers found: %d", len(hits))
    return hits

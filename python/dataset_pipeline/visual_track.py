"""
Visual track — OCR 花字提取。
- 用 PySceneDetect 找"画面变化点"，只对变化帧做 OCR (节省 3-5x 时间)。
- 兜底：保证最低 OCR_FPS 的均匀采样，避免长静态镜头漏掉持续花字。
- 关键词匹配支持中英混合 + 模糊 (RapidFuzz)。
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Set, Dict
import json
import logging

import cv2
import numpy as np
from rapidfuzz import fuzz

from .config import (
    OCR_FPS, SCENE_CHANGE_THRESHOLD, TECHNIQUE_KEYWORDS, TRANSITION_KEYWORDS, sec_to_tick,
)
from .schemas import OCRHit

log = logging.getLogger(__name__)


# -------- OCR backend (lazy-load EasyOCR with GPU) --------
_reader = None
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # type: ignore
        _reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
        log.info("EasyOCR loaded on GPU")
    return _reader


# -------- 关键词倒排表 (含转声词) --------
def _build_keyword_index() -> Dict[str, str]:
    """返回 keyword_lower -> technique_name"""
    idx = {}
    for tech, kws in TECHNIQUE_KEYWORDS.items():
        for kw in kws:
            idx[kw.lower()] = tech
    for tech, kws in TRANSITION_KEYWORDS.items():
        for kw in kws:
            idx[kw.lower()] = tech
    return idx
_KW_INDEX = _build_keyword_index()


def _match_keywords(text: str, fuzzy_threshold: int = 85):
    """返回 [(matched_kw, technique, score)]"""
    text_l = text.lower()
    hits = []
    for kw, tech in _KW_INDEX.items():
        if kw in text_l:
            hits.append((kw, tech, 100))
            continue
        # 模糊匹配 (容忍 OCR 错字)
        score = fuzz.partial_ratio(kw, text_l)
        if score >= fuzzy_threshold:
            hits.append((kw, tech, score))
    return hits


# -------- 帧采样 --------
def _sample_frame_indices(total_frames: int, fps: float,
                          ocr_fps: float = OCR_FPS) -> List[int]:
    """按 ocr_fps 均匀采样的帧号列表 (兜底)。"""
    if fps <= 0:
        return []
    step = max(1, int(round(fps / ocr_fps)))
    return list(range(0, total_frames, step))


def _scene_change_frames(video_path: Path) -> Set[int]:
    """用 PySceneDetect 找场景切换点 (不是必需，失败不影响主流程)。"""
    try:
        from scenedetect import open_video, SceneManager  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
        video = open_video(str(video_path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=SCENE_CHANGE_THRESHOLD * 100))
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        return {s[0].get_frames() for s in scenes}
    except Exception as e:
        log.warning("scene detect skipped: %s", e)
        return set()


# -------- 主入口 --------
def extract_ocr_hits(video_path: Path, out_json: Path) -> List[OCRHit]:
    if out_json.exists():
        log.info("ocr cache hit: %s", out_json)
        return [OCRHit(**d) for d in json.loads(out_json.read_text("utf-8"))]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log.info("video fps=%.2f total_frames=%d", fps, total)

    target_frames = set(_sample_frame_indices(total, fps))
    target_frames |= _scene_change_frames(video_path)
    target_frames = sorted(target_frames)

    reader = _get_reader()
    hits: List[OCRHit] = []
    last_seen_text_per_tech: Dict[str, int] = {}   # 防止同一花字连续重复

    for fidx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # 下采样加速 OCR (花字一般够大)
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
        try:
            results = reader.readtext(frame, detail=1, paragraph=False)
        except Exception as e:
            log.warning("ocr fail @frame=%d: %s", fidx, e)
            continue

        t_sec = fidx / fps
        tick = sec_to_tick(t_sec)

        for box, text, conf in results:
            if not text or conf < 0.4:
                continue
            for kw, tech, score in _match_keywords(text):
                # 同一技巧 1 秒内只记一次 (花字会持续多帧)
                last_tick = last_seen_text_per_tech.get(tech, -999)
                if tick - last_tick < 10:
                    continue
                last_seen_text_per_tech[tech] = tick

                bbox = [int(min(p[0] for p in box)), int(min(p[1] for p in box)),
                        int(max(p[0] for p in box)), int(max(p[1] for p in box))]
                hits.append(OCRHit(
                    time_tick=tick,
                    text=text,
                    matched_keyword=kw,
                    technique=tech,
                    confidence=float(conf) * (score / 100),
                    bbox=bbox,
                ))
    cap.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("OCR done: %d hits → %s", len(hits), out_json)
    return hits

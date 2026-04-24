"""
Stage 3 — ffmpeg + h264_nvenc 3s 切片。
输入：verdicts.jsonl (ACCEPT only)
输出：<technique>_<video_id>_<idx>.mp4 + .wav + 追加 metadata.jsonl
"""
from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import ffmpeg  # type: ignore

from .config import SliceCfg, tick_to_sec

log = logging.getLogger(__name__)


def _nvenc_available() -> bool:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT,
        )
        return "h264_nvenc" in out
    except Exception:
        return False


def slice_one(video_path: Path, out_mp4: Path, out_wav: Path,
              center_tick: int, cfg: SliceCfg) -> bool:
    """切一段。3s = pad_ticks * 2 (默认 ±15 = 3.0s)。"""
    start = max(0.0, tick_to_sec(center_tick - cfg.pad_ticks))
    duration = tick_to_sec(cfg.pad_ticks * 2)

    video_codec = cfg.out_video_codec if _nvenc_available() else "libx264"

    try:
        # 视频：NVENC 硬编码，尽量 copy 音频做最终 wav
        (
            ffmpeg
            .input(str(video_path), ss=start, hwaccel="cuda")
            .output(
                str(out_mp4),
                t=duration,
                vcodec=video_codec,
                preset="p5",
                acodec="aac",
                loglevel="error",
            )
            .overwrite_output()
            .run()
        )
        # 音频：单独抽 16kHz mono wav（训练用）
        (
            ffmpeg
            .input(str(video_path), ss=start)
            .output(
                str(out_wav),
                t=duration,
                ac=1,
                ar=cfg.out_audio_sr,
                acodec=cfg.out_audio_codec,
                vn=None,
                loglevel="error",
            )
            .overwrite_output()
            .run()
        )
        return True
    except Exception as e:
        log.error("slice fail center=%d: %s", center_tick, e)
        return False


def slice_technique(video_path: Path, work_dir: Path, technique: str,
                    cfg: SliceCfg, video_id: Optional[str] = None) -> int:
    """读 verdicts.jsonl，对 ACCEPT 逐条切片。返回成功切片数。"""
    vid = video_id or video_path.stem
    v_dir = work_dir / technique
    verdicts_path = v_dir / "verdicts.jsonl"
    slices_dir = v_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    meta_path = slices_dir / "metadata.jsonl"

    if not verdicts_path.exists():
        log.warning("no verdicts for %s", technique)
        return 0

    accepted: List[dict] = []
    for ln in verdicts_path.read_text("utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        if d.get("decision") == "ACCEPT" and d.get("chosen_center_tick") is not None:
            accepted.append(d)

    ok = 0
    with meta_path.open("a", encoding="utf-8") as meta_f:
        for idx, v in enumerate(accepted, start=1):
            stem = f"{technique}_{vid}_{idx:03d}"
            out_mp4 = slices_dir / f"{stem}.mp4"
            out_wav = slices_dir / f"{stem}.wav"
            if out_mp4.exists() and out_wav.exists():
                log.info("skip existing: %s", stem)
                ok += 1
                continue
            if slice_one(video_path, out_mp4, out_wav, v["chosen_center_tick"], cfg):
                ok += 1
                meta_f.write(json.dumps({
                    "stem": stem,
                    "technique": technique,
                    "video_id": vid,
                    "source_video": str(video_path),
                    "center_tick": v["chosen_center_tick"],
                    "center_sec": round(tick_to_sec(v["chosen_center_tick"]), 2),
                    "duration_sec": round(tick_to_sec(cfg.pad_ticks * 2), 2),
                    "confidence": v.get("confidence", 0.0),
                    "notes": v.get("notes", ""),
                    "candidate_id": v["candidate_id"],
                }, ensure_ascii=False) + "\n")
    log.info("[%s] sliced %d / %d → %s", technique, ok, len(accepted), slices_dir)
    return ok

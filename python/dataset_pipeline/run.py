"""
Pipeline orchestrator — 串起 Layer 1 + Stage 1 + (可选) Stage 2 + Stage 3。
用法：
    # 只跑到 Stage 1（候选 JSON）
    python -m dataset_pipeline.run --video teach.mp4
    # 端到端（需设置 MINIMAX_API_KEY）
    python -m dataset_pipeline.run --video teach.mp4 --end-to-end
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path

from .config import PipelineConfig
from .audio_io import extract_audio_16k, check_ffmpeg
from .visual_track import extract_ocr_hits
from .speech_track import transcribe, find_asr_triggers
from .acoustic_track import run_vad, apply_acoustic_gate
from .candidate_builder import build_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")


def _auto_discover_techniques(ocr_hits, asr_hits) -> list[str]:
    """自主发现模式：扫 OCR + ASR 所有命中的技巧，全部作为 target。"""
    found = {h.technique for h in ocr_hits} | {h.technique for h in asr_hits}
    return sorted(found)


def run(video_path: Path, cfg: PipelineConfig) -> dict[str, Path]:
    """跑完整 Layer 1 + Stage 1。返回 {technique: candidates_jsonl_path}."""
    check_ffmpeg()
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.shared_dir.mkdir(parents=True, exist_ok=True)

    # ---- Layer 1: 多模态特征提取（与技巧无关，跑一次给所有技巧用）----
    audio = extract_audio_16k(video_path, cfg.paths.audio_wav)

    ocr_hits = extract_ocr_hits(video_path, cfg.paths.ocr_json)
    asr_words = transcribe(audio, cfg.whisper, cfg.paths.asr_json)
    asr_hits = find_asr_triggers(asr_words)

    vad_blocks = run_vad(audio, cfg.vad, cfg.paths.vad_json, cfg.transition)
    apply_acoustic_gate(vad_blocks, cfg.gate)

    # ---- 自主发现模式：老师示范什么，就建什么文件夹 ----
    techs = cfg.target_techniques
    if not techs or techs == ["auto"]:
        techs = _auto_discover_techniques(ocr_hits, asr_hits)
        log.info("🔍 auto-discover mode: found techniques = %s", techs)
        if not techs:
            log.warning("No technique mentioned anywhere in video; nothing to do.")
            return {}

    # ---- Stage 1: 按技巧分别构造候选 + 落到独立子目录 ----
    outputs: dict[str, Path] = {}
    for tech in techs:
        tech_dir = cfg.paths.technique_dir(tech)
        tech_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.slices_dir(tech).mkdir(parents=True, exist_ok=True)

        candidates = build_candidates(
            ocr_hits=ocr_hits,
            asr_hits=asr_hits,
            asr_words=asr_words,
            vad_blocks=vad_blocks,
            target_techniques=[tech],
            cfg=cfg.candidate,
        )

        out = cfg.paths.candidates_jsonl(tech)
        with out.open("w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c.to_llm_payload(), ensure_ascii=False) + "\n")
        log.info("[%s] %d candidates → %s", tech, len(candidates), out)
        outputs[tech] = out

    log.info("ALL DONE. techniques=%s", list(outputs.keys()))
    return outputs


def run_end_to_end(video_path: Path, cfg: PipelineConfig) -> None:
    """Layer1 → Stage1 → Stage2 (LLM) → Stage3 (NVENC slice)."""
    outputs = run(video_path, cfg)
    if not outputs:
        return

    # Stage 2
    from .llm_judge import MiniMaxClient, judge_technique_folder
    client = MiniMaxClient()
    for tech in outputs:
        judge_technique_folder(cfg.paths.work_dir, tech, client)

    # Stage 3
    from .slicer import slice_technique
    for tech in outputs:
        slice_technique(video_path, cfg.paths.work_dir, tech, cfg.slicing)


def main():
    p = argparse.ArgumentParser(
        description="Vibesing dataset pipeline. "
                    "不传 --techniques 即自主发现模式。"
    )
    p.add_argument("--video", required=True, type=Path)
    p.add_argument(
        "--techniques", nargs="*", default=["auto"],
        help="目标技巧列表。默认 'auto' = 自主发现（老师示范什么就建什么文件夹）。",
    )
    p.add_argument("--workdir", type=Path, default=Path("./pipeline_out"))
    p.add_argument("--end-to-end", action="store_true",
                   help="跑完 Stage1 后继续 LLM 裁决 + NVENC 切片。需设置 MINIMAX_API_KEY。")
    args = p.parse_args()

    cfg = PipelineConfig()
    cfg.target_techniques = args.techniques
    cfg.paths.work_dir = args.workdir

    if args.end_to_end:
        if not os.getenv("MINIMAX_API_KEY"):
            raise SystemExit("MINIMAX_API_KEY not set; cannot run --end-to-end")
        run_end_to_end(args.video, cfg)
    else:
        run(args.video, cfg)


if __name__ == "__main__":
    main()

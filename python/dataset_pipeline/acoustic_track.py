"""
Acoustic track — Silero VAD + Librosa 声学特征 (RMS / f0 / HNR / flatness)。
关键：用 HNR + spectral flatness + pitch stability 联合过滤"伪发声"。
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import logging

import numpy as np
import librosa
import soundfile as sf

from .config import (
    AUDIO_SAMPLE_RATE, VADCfg, AcousticGate, sec_to_tick,
)
from .schemas import VADBlock

log = logging.getLogger(__name__)


# =========================================================
# Silero VAD
# =========================================================
def run_vad(audio_wav: Path, cfg: VADCfg, out_json: Path) -> List[VADBlock]:
    if out_json.exists():
        log.info("vad cache hit: %s", out_json)
        return [VADBlock(**d) for d in json.loads(out_json.read_text("utf-8"))]

    import torch  # type: ignore
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, *_) = utils

    wav = read_audio(str(audio_wav), sampling_rate=AUDIO_SAMPLE_RATE)
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=AUDIO_SAMPLE_RATE,
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_duration_ms,
        min_silence_duration_ms=cfg.min_silence_duration_ms,
        speech_pad_ms=cfg.speech_pad_ms,
    )
    blocks = [
        VADBlock(
            start_tick=sec_to_tick(t["start"] / AUDIO_SAMPLE_RATE),
            end_tick=sec_to_tick(t["end"] / AUDIO_SAMPLE_RATE),
        )
        for t in ts
    ]

    # 提取声学特征 (一次性 load 完整 wav)
    audio, sr = librosa.load(str(audio_wav), sr=AUDIO_SAMPLE_RATE, mono=True)
    for b in blocks:
        _enrich_acoustic(b, audio, sr)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([b.__dict__ for b in blocks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("VAD+acoustic done: %d blocks → %s", len(blocks), out_json)
    return blocks


# =========================================================
# 声学特征
# =========================================================
def _enrich_acoustic(b: VADBlock, audio: np.ndarray, sr: int) -> None:
    s = int(b.start_tick * sr * 0.1)
    e = int(b.end_tick * sr * 0.1)
    seg = audio[s:e]
    if seg.size < sr // 10:               # < 100ms 跳过
        return

    # RMS
    rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=160)[0]
    b.avg_rms = float(np.mean(rms))

    # f0 via pyin (鲁棒，自带 voicing mask)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            seg, fmin=65.0, fmax=1100.0, sr=sr, frame_length=2048,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        if f0_voiced.size > 5:
            b.avg_pitch_hz = float(np.mean(f0_voiced))
            cv = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-9))
            b.pitch_stability = float(max(0.0, 1.0 - cv))
    except Exception as exc:
        log.debug("pyin fail: %s", exc)

    # HNR (用 librosa harmonic/percussive 比近似；更准可用 parselmouth)
    try:
        b.avg_hnr_db = _hnr_db(seg, sr)
    except Exception as exc:
        log.debug("hnr fail: %s", exc)

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=seg, n_fft=1024, hop_length=256)[0]
    b.avg_spectral_flatness = float(np.mean(flatness))


def _hnr_db(y: np.ndarray, sr: int) -> float:
    """近似 HNR：harmonic 能量 / (residual 能量)。
    若安装了 parselmouth，建议替换为 Praat 的 To Harmonicity。
    """
    try:
        import parselmouth  # type: ignore
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        harm = snd.to_harmonicity_cc()
        vals = harm.values[harm.values != -200]   # -200 = undef
        return float(np.mean(vals)) if vals.size else 0.0
    except Exception:
        # fallback: harmonic-percussive 分离
        h, p = librosa.effects.hpss(y)
        eh = float(np.sum(h ** 2))
        ep = float(np.sum(p ** 2)) + 1e-9
        return float(10.0 * np.log10(eh / ep))


# =========================================================
# 声学闸门
# =========================================================
def apply_acoustic_gate(blocks: List[VADBlock], gate: AcousticGate) -> None:
    """就地标记 is_valid_demo & quality_score。"""
    for b in blocks:
        ok = (
            b.avg_rms >= gate.min_avg_rms
            and gate.min_avg_pitch_hz <= b.avg_pitch_hz <= gate.max_avg_pitch_hz
            and b.pitch_stability >= gate.min_pitch_stability
            and b.avg_hnr_db >= gate.min_hnr_db
            and b.avg_spectral_flatness <= gate.max_spectral_flatness
        )
        b.is_valid_demo = bool(ok)

        # 质量分：HNR 占 40%，pitch_stability 30%，时长 20%，能量 10%
        dur_ticks = max(0, b.end_tick - b.start_tick)
        dur_score = min(1.0, dur_ticks / 30.0)             # 3s 满分
        hnr_norm = max(0.0, min(1.0, (b.avg_hnr_db - 5) / 20.0))
        rms_norm = min(1.0, b.avg_rms / 0.3)
        b.quality_score = float(
            0.40 * hnr_norm + 0.30 * b.pitch_stability
            + 0.20 * dur_score + 0.10 * rms_norm
        )

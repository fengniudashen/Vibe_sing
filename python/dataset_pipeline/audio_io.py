"""
Audio I/O — 用 ffmpeg-python 抽 16kHz mono WAV，硬件加速解码 (CUDA)。
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import shutil
import logging

import ffmpeg

log = logging.getLogger(__name__)


def extract_audio_16k(video_path: Path, out_wav: Path,
                      sample_rate: int = 16000) -> Path:
    """从视频抽取 16kHz mono PCM WAV。
    NVDEC 用于解码加速 (RTX 5060 支持 H.264/H.265/AV1)。
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        log.info("audio cache hit: %s", out_wav)
        return out_wav

    # 注：纯音频抽取，NVDEC 收益有限，但对带宽大的源(>4K)仍有提速。
    stream = (
        ffmpeg
        .input(str(video_path), hwaccel="cuda")
        .output(
            str(out_wav),
            ac=1,
            ar=sample_rate,
            acodec="pcm_s16le",
            vn=None,
            loglevel="error",
        )
        .overwrite_output()
    )
    stream.run()
    log.info("extracted audio → %s", out_wav)
    return out_wav


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")
    # 检查 nvenc 可用性
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT
        )
        if "h264_nvenc" not in out:
            log.warning("h264_nvenc not available, slicing will fall back to libx264")
    except Exception as e:
        log.warning("nvenc probe failed: %s", e)

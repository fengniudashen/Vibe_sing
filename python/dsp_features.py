"""
Vibesing MVP V0.1 — DSP Feature Extraction Module
===================================================
Extracts frame-level acoustic features from audio files:
  - 80-band Mel Spectrogram
  - F0 (fundamental frequency) via Parselmouth/Praat
  - HNR (Harmonics-to-Noise Ratio)
  - Frame Energy (RMS)

All features aligned to a common time grid (16ms hop @ 16kHz).

Dependencies:
    pip install librosa parselmouth numpy soundfile
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from dataclasses import dataclass
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SR = 16000       # Downsample everything to 16kHz
N_FFT = 1024            # 64ms window @ 16kHz
HOP_LENGTH = 256        # 16ms hop → ~62.5 fps
N_MELS = 80             # Mel filterbank bands
F_MIN = 50.0            # Hz – lowest fundamental we care about
F_MAX = 8000.0          # Hz – Nyquist @ 16kHz


@dataclass
class FrameFeatures:
    """Container for aligned frame-level features."""
    mel_spectrogram: np.ndarray   # (n_mels, n_frames) – log-power Mel
    f0: np.ndarray                # (n_frames,) – Hz, 0 = unvoiced
    hnr: np.ndarray               # (n_frames,) – dB
    energy: np.ndarray            # (n_frames,) – dB RMS
    times: np.ndarray             # (n_frames,) – seconds
    sr: int                       # sample rate used
    hop_length: int


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file, convert to mono, resample to target_sr."""
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y


def extract_mel(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract log-power Mel spectrogram.
    Returns: (n_mels, n_frames) array in dB.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
        power=2.0,
    )
    # Convert to log scale (dB), clamp floor at -80 dB
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    return S_db


def extract_f0_and_hnr(
    y: np.ndarray,
    sr: int = TARGET_SR,
    hop: int = HOP_LENGTH,
    n_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 and HNR using Parselmouth (Praat backend).
    
    Returns:
        f0: (n_frames,) array in Hz, 0 = unvoiced
        hnr: (n_frames,) array in dB
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    time_step = hop / sr  # align to our hop grid

    # ── F0 via autocorrelation (Praat's gold-standard method) ──
    pitch = call(
        snd, "To Pitch (ac)",
        time_step,          # time step
        75.0,               # pitch floor (Hz)
        15,                 # max candidates
        "no",               # very accurate (silence threshold)
        0.03,               # silence threshold
        0.45,               # voicing threshold
        0.01,               # octave cost
        0.35,               # octave-jump cost
        0.14,               # voiced/unvoiced cost
        600.0,              # pitch ceiling (Hz)
    )

    # ── HNR ──
    harmonicity = call(snd, "To Harmonicity (cc)", time_step, 75.0, 0.1, 1.0)

    # ── Sample onto frame grid ──
    if n_frames is None:
        n_frames = 1 + len(y) // hop

    frame_times = np.arange(n_frames) * time_step
    f0 = np.zeros(n_frames, dtype=np.float32)
    hnr = np.zeros(n_frames, dtype=np.float32)

    for i, t in enumerate(frame_times):
        f0_val = call(pitch, "Get value at time", t, "Hertz", "Linear")
        f0[i] = f0_val if not np.isnan(f0_val) else 0.0

        hnr_val = call(harmonicity, "Get value at time", t, "Cubic")
        hnr[i] = hnr_val if not np.isnan(hnr_val) else -10.0  # floor for unvoiced

    return f0, hnr


def extract_energy(y: np.ndarray, hop: int = HOP_LENGTH, n_fft: int = N_FFT) -> np.ndarray:
    """
    Extract frame-level RMS energy in dB.
    Returns: (n_frames,) array.
    """
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    # Convert to dB, floor at -80
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return rms_db


def extract_all(path: str | Path) -> FrameFeatures:
    """
    One-shot extraction of all features from an audio file.
    All features are time-aligned to the same frame grid.
    """
    y = load_audio(path)

    mel = extract_mel(y)
    n_frames = mel.shape[1]

    f0, hnr = extract_f0_and_hnr(y, n_frames=n_frames)
    energy = extract_energy(y)

    # Align lengths (tiny rounding differences possible)
    min_len = min(n_frames, len(f0), len(hnr), len(energy))
    mel = mel[:, :min_len]
    f0 = f0[:min_len]
    hnr = hnr[:min_len]
    energy = energy[:min_len]
    times = np.arange(min_len) * (HOP_LENGTH / TARGET_SR)

    return FrameFeatures(
        mel_spectrogram=mel,
        f0=f0,
        hnr=hnr,
        energy=energy,
        times=times,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dsp_features.py <audio_file>")
        sys.exit(1)

    features = extract_all(sys.argv[1])
    print(f"Extracted features from: {sys.argv[1]}")
    print(f"  Sample rate:  {features.sr} Hz")
    print(f"  Hop length:   {features.hop_length} samples ({features.hop_length/features.sr*1000:.1f} ms)")
    print(f"  Mel shape:    {features.mel_spectrogram.shape}")
    print(f"  Frames:       {len(features.times)}")
    print(f"  Duration:     {features.times[-1]:.2f} s")
    print(f"  F0 range:     {features.f0[features.f0 > 0].min():.1f} - {features.f0[features.f0 > 0].max():.1f} Hz (voiced frames)")
    print(f"  HNR range:    {features.hnr.min():.1f} - {features.hnr.max():.1f} dB")
    print(f"  Energy range: {features.energy.min():.1f} - {features.energy.max():.1f} dB")

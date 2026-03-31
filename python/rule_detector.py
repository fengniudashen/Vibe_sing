"""
Vibesing MVP V0.1 — Rule-Based Squeeze Detector
=================================================
Detects vocal squeeze (挤卡) using simple HNR + Energy + F0 thresholds.
No deep learning required. This is the Week 1 baseline.

The detector runs in two modes:
  1. Offline: Analyze a full audio file and output frame-level squeeze labels
  2. Simulated streaming: Process frames sequentially with EMA + state filter

Usage:
    python rule_detector.py <audio_file> [--plot] [--save output.png]
    python rule_detector.py <audio_file> --batch-test     # test on directory

Dependencies:
    pip install librosa parselmouth numpy matplotlib soundfile
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from dsp_features import extract_all, FrameFeatures


# ─── Thresholds (tune these on your first 10 test recordings) ─────────────────
HNR_SQUEEZE_THRESHOLD = 12.0     # dB — below this → suspect squeeze
ENERGY_MIN_THRESHOLD = -40.0     # dB — ignore silence (below this = not singing)
F0_HIGH_VOICE_THRESHOLD = 220.0  # Hz — squeeze mainly happens in upper register
# Note: F0 threshold is a soft gate. We still flag squeeze below this F0 but
# with reduced confidence.


@dataclass
class SqueezeResult:
    """Frame-level squeeze detection results."""
    raw_prob: np.ndarray          # (n_frames,) – raw squeeze probability [0, 1]
    smooth_prob: np.ndarray       # (n_frames,) – EMA-smoothed probability
    is_squeeze: np.ndarray        # (n_frames,) – binary after state filter (bool)
    score: np.ndarray             # (n_frames,) – vocal health score 0-100
    times: np.ndarray             # (n_frames,) – seconds
    squeeze_ratio: float          # fraction of voiced frames flagged as squeeze
    mean_score: float             # average score over voiced frames


def compute_raw_squeeze_probability(
    hnr: np.ndarray,
    energy: np.ndarray,
    f0: np.ndarray,
) -> np.ndarray:
    """
    Rule-based squeeze probability for each frame.
    
    Logic:
      - Silence (low energy) → prob = 0 (not singing)
      - Unvoiced (f0 ≈ 0) → prob = 0 (can't judge)
      - HNR < threshold AND energy is present → high prob
      - F0 in upper register → boost probability (squeeze is more likely)
    
    Returns: (n_frames,) array of probabilities in [0, 1]
    """
    n = len(hnr)
    prob = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Gate: skip silence and unvoiced
        if energy[i] < ENERGY_MIN_THRESHOLD or f0[i] < 50:
            prob[i] = 0.0
            continue

        # Core: HNR-based squeeze signal
        # Map HNR to probability via sigmoid-like function
        # HNR = 12 → prob ≈ 0.5; HNR = 6 → prob ≈ 0.9; HNR = 20 → prob ≈ 0.1
        hnr_signal = 1.0 / (1.0 + np.exp(0.5 * (hnr[i] - HNR_SQUEEZE_THRESHOLD)))

        # Boost for high pitch (squeeze is more common in upper register)
        if f0[i] > F0_HIGH_VOICE_THRESHOLD:
            pitch_boost = min(1.3, 1.0 + (f0[i] - F0_HIGH_VOICE_THRESHOLD) / 500.0)
        else:
            pitch_boost = 0.7  # still possible but less likely

        prob[i] = np.clip(hnr_signal * pitch_boost, 0.0, 1.0)

    return prob


class EMAFilter:
    """Exponential Moving Average filter for smoothing frame-level outputs."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = 0.0
        self.initialized = False


class SqueezeStateFilter:
    """
    Prevents rapid red-light flickering.
    Requires N consecutive frames above threshold to trigger,
    and M consecutive frames below to clear.
    """
    MIN_FRAMES_TO_TRIGGER = 6    # ~96ms @ 16ms/frame
    MIN_FRAMES_TO_CLEAR = 10     # ~160ms
    THRESHOLD_ON = 0.55
    THRESHOLD_OFF = 0.30

    def __init__(self):
        self.is_active = False
        self.consecutive_on = 0
        self.consecutive_off = 0

    def update(self, smooth_prob: float) -> bool:
        if self.is_active:
            if smooth_prob < self.THRESHOLD_OFF:
                self.consecutive_off += 1
                self.consecutive_on = 0
                if self.consecutive_off >= self.MIN_FRAMES_TO_CLEAR:
                    self.is_active = False
            else:
                self.consecutive_off = 0
        else:
            if smooth_prob > self.THRESHOLD_ON:
                self.consecutive_on += 1
                self.consecutive_off = 0
                if self.consecutive_on >= self.MIN_FRAMES_TO_TRIGGER:
                    self.is_active = True
            else:
                self.consecutive_on = 0

        return self.is_active

    def reset(self):
        self.is_active = False
        self.consecutive_on = 0
        self.consecutive_off = 0


def detect_squeeze(features: FrameFeatures) -> SqueezeResult:
    """
    Full squeeze detection pipeline:
      1. Compute raw frame-level squeeze probability (rule-based)
      2. Apply EMA smoothing
      3. Apply state filter (debounce)
      4. Compute vocal health score
    """
    n = len(features.times)

    # Step 1: Raw probability
    raw_prob = compute_raw_squeeze_probability(
        features.hnr, features.energy, features.f0
    )

    # Step 2: EMA smoothing (simulated streaming)
    ema = EMAFilter(alpha=0.4)
    smooth_prob = np.zeros(n, dtype=np.float32)
    for i in range(n):
        smooth_prob[i] = ema.update(raw_prob[i])

    # Step 3: State filter
    state_filter = SqueezeStateFilter()
    is_squeeze = np.zeros(n, dtype=bool)
    for i in range(n):
        is_squeeze[i] = state_filter.update(smooth_prob[i])

    # Step 4: Vocal health score (simple version)
    score = np.clip(100.0 * (1.0 - smooth_prob), 0, 100).astype(np.float32)

    # Stats
    voiced_mask = features.f0 > 50
    if voiced_mask.any():
        squeeze_ratio = is_squeeze[voiced_mask].mean()
        mean_score = score[voiced_mask].mean()
    else:
        squeeze_ratio = 0.0
        mean_score = 100.0

    return SqueezeResult(
        raw_prob=raw_prob,
        smooth_prob=smooth_prob,
        is_squeeze=is_squeeze,
        score=score,
        times=features.times,
        squeeze_ratio=squeeze_ratio,
        mean_score=mean_score,
    )


def plot_detection(features: FrameFeatures, result: SqueezeResult, title: str = "", save_path: str | None = None):
    """Visualize squeeze detection results."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 0.5], hspace=0.4)
    t = features.times

    # Panel 1: Mel with squeeze overlay
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(
        features.mel_spectrogram, aspect="auto", origin="lower",
        extent=[t[0], t[-1], 0, 80], cmap="magma", interpolation="nearest",
    )
    # Red overlay where squeeze detected
    squeeze_mask = result.is_squeeze.astype(float)
    for i in range(len(t) - 1):
        if squeeze_mask[i]:
            ax1.axvspan(t[i], t[i + 1], color="red", alpha=0.25)
    ax1.set_ylabel("Mel Band")
    ax1.set_title(f"Mel Spectrogram + Squeeze Zones (red) — {title}")

    # Panel 2: HNR
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, features.hnr, color="forestgreen", linewidth=0.8)
    ax2.axhline(y=HNR_SQUEEZE_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.fill_between(t, features.hnr, HNR_SQUEEZE_THRESHOLD,
                     where=(features.hnr < HNR_SQUEEZE_THRESHOLD), color="red", alpha=0.15)
    ax2.set_ylabel("HNR (dB)")
    ax2.set_title("HNR (red zone = below squeeze threshold)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Squeeze probability (raw vs smooth)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, result.raw_prob, color="lightcoral", linewidth=0.5, alpha=0.6, label="Raw prob")
    ax3.plot(t, result.smooth_prob, color="red", linewidth=1.2, label="EMA smooth")
    ax3.axhline(y=SqueezeStateFilter.THRESHOLD_ON, color="orange", linestyle=":", alpha=0.5)
    ax3.set_ylabel("Squeeze Prob")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_title("Squeeze Probability (raw vs smoothed)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Score
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, result.score, color="dodgerblue", linewidth=1)
    ax4.fill_between(t, 0, result.score, alpha=0.15, color="dodgerblue")
    ax4.set_ylabel("Score (0-100)")
    ax4.set_ylim(-5, 105)
    ax4.set_title(f"Vocal Health Score (mean: {result.mean_score:.1f})")
    ax4.grid(True, alpha=0.3)

    # Panel 5: Red/Green light timeline
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    colors = ["red" if s else "green" for s in result.is_squeeze]
    for i in range(len(t) - 1):
        ax5.axvspan(t[i], t[i + 1], color=colors[i], alpha=0.8)
    ax5.set_yticks([])
    ax5.set_xlabel("Time (s)")
    ax5.set_title(f"Red Light Timeline (squeeze ratio: {result.squeeze_ratio:.1%})")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vibesing Rule-Based Squeeze Detector")
    parser.add_argument("audio_file", help="Audio file to analyze")
    parser.add_argument("--plot", action="store_true", help="Show visualization")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    print(f"Analyzing: {args.audio_file}")
    features = extract_all(args.audio_file)
    result = detect_squeeze(features)

    print(f"\n{'='*50}")
    print(f"  SQUEEZE DETECTION REPORT")
    print(f"{'='*50}")
    print(f"  Duration:          {features.times[-1]:.2f}s")
    print(f"  Voiced frames:     {(features.f0 > 50).sum()}/{len(features.f0)}")
    print(f"  Squeeze ratio:     {result.squeeze_ratio:.1%}")
    print(f"  Mean health score: {result.mean_score:.1f}/100")
    print(f"{'='*50}")

    if result.squeeze_ratio > 0.3:
        print("\n  [!] HIGH SQUEEZE DETECTED — 挤卡严重！")
        print("      Your voice shows significant constriction.")
        print("      Try relaxing your throat and supporting with breath.")
    elif result.squeeze_ratio > 0.1:
        print("\n  [~] MODERATE SQUEEZE — 轻度挤卡。")
        print("      Some constriction detected in high notes.")
    else:
        print("\n  [OK] HEALTHY VOICE — 发声健康！")

    if args.plot or args.save:
        plot_detection(
            features, result,
            title=Path(args.audio_file).stem,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()

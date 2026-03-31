"""
Vibesing MVP V0.1 — Feature Visualization
==========================================
Plots 4 time-aligned curves for any audio file:
  1. Mel Spectrogram (heatmap)
  2. F0 contour
  3. HNR contour
  4. Frame Energy contour

Usage:
    python visualize_features.py <audio_file> [--save output.png]
    python visualize_features.py normal.wav squeezed.wav --compare

Dependencies:
    pip install matplotlib librosa parselmouth numpy soundfile
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from dsp_features import extract_all, FrameFeatures


def plot_single(features: FrameFeatures, title: str = "", save_path: str | None = None):
    """Plot 4-panel feature visualization for a single audio file."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.35)

    t = features.times

    # ── Panel 1: Mel Spectrogram ──
    ax1 = fig.add_subplot(gs[0])
    img = ax1.imshow(
        features.mel_spectrogram,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, 80],
        cmap="magma",
        interpolation="nearest",
    )
    ax1.set_ylabel("Mel Band")
    ax1.set_title(f"Mel Spectrogram — {title}" if title else "Mel Spectrogram")
    plt.colorbar(img, ax=ax1, label="dB", pad=0.01)

    # ── Panel 2: F0 Contour ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    voiced_mask = features.f0 > 0
    ax2.scatter(t[voiced_mask], features.f0[voiced_mask], s=2, c="dodgerblue", alpha=0.8)
    ax2.set_ylabel("F0 (Hz)")
    ax2.set_ylim(50, min(800, features.f0[voiced_mask].max() * 1.2) if voiced_mask.any() else 600)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Fundamental Frequency (F0)")

    # ── Panel 3: HNR Contour ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, features.hnr, color="forestgreen", linewidth=0.8, alpha=0.9)
    ax3.axhline(y=12, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Squeeze threshold (12 dB)")
    ax3.fill_between(t, features.hnr, 12, where=(features.hnr < 12), color="red", alpha=0.15)
    ax3.set_ylabel("HNR (dB)")
    ax3.set_ylim(-10, 40)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Harmonics-to-Noise Ratio (HNR) — Red zone = potential squeeze")

    # ── Panel 4: Energy Contour ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, features.energy, color="darkorange", linewidth=0.8, alpha=0.9)
    ax4.set_ylabel("Energy (dB)")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Frame Energy (RMS)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_comparison(
    feat_a: FrameFeatures,
    feat_b: FrameFeatures,
    label_a: str = "Normal",
    label_b: str = "Squeezed",
    save_path: str | None = None,
):
    """Side-by-side comparison of HNR/F0/Energy for two audio files."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")

    for col, (feat, label) in enumerate([(feat_a, label_a), (feat_b, label_b)]):
        t = feat.times
        voiced = feat.f0 > 0

        # F0
        axes[0, col].scatter(t[voiced], feat.f0[voiced], s=2, c="dodgerblue", alpha=0.8)
        axes[0, col].set_ylabel("F0 (Hz)")
        axes[0, col].set_title(f"{label} — F0")
        axes[0, col].set_ylim(50, 800)
        axes[0, col].grid(True, alpha=0.3)

        # HNR
        axes[1, col].plot(t, feat.hnr, color="forestgreen", linewidth=0.8)
        axes[1, col].axhline(y=12, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes[1, col].fill_between(t, feat.hnr, 12, where=(feat.hnr < 12), color="red", alpha=0.15)
        axes[1, col].set_ylabel("HNR (dB)")
        axes[1, col].set_title(f"{label} — HNR (red zone = squeeze)")
        axes[1, col].set_ylim(-10, 40)
        axes[1, col].grid(True, alpha=0.3)

        # Energy
        axes[2, col].plot(t, feat.energy, color="darkorange", linewidth=0.8)
        axes[2, col].set_ylabel("Energy (dB)")
        axes[2, col].set_xlabel("Time (s)")
        axes[2, col].set_title(f"{label} — Energy")
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle("Normal vs Squeezed Voice Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Vibesing Feature Visualization")
    parser.add_argument("audio_files", nargs="+", help="Audio file(s) to analyze")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    parser.add_argument("--compare", action="store_true", help="Side-by-side comparison mode (requires 2 files)")
    args = parser.parse_args()

    if args.compare:
        if len(args.audio_files) != 2:
            print("Error: --compare requires exactly 2 audio files")
            sys.exit(1)
        print(f"Extracting features from: {args.audio_files[0]}")
        feat_a = extract_all(args.audio_files[0])
        print(f"Extracting features from: {args.audio_files[1]}")
        feat_b = extract_all(args.audio_files[1])
        plot_comparison(
            feat_a, feat_b,
            label_a=Path(args.audio_files[0]).stem,
            label_b=Path(args.audio_files[1]).stem,
            save_path=args.save,
        )
    else:
        for audio_path in args.audio_files:
            print(f"Extracting features from: {audio_path}")
            features = extract_all(audio_path)
            save = args.save if len(args.audio_files) == 1 else None
            plot_single(features, title=Path(audio_path).stem, save_path=save)


if __name__ == "__main__":
    main()

---
applyTo: "**"
---

# Vibesing（高音觉醒）— Copilot Instructions

## Project Overview

Vibesing is an iOS vocal coaching app that detects "vocal squeeze" (挤卡) in real-time using on-device DSP + AI. Current stage: MVP V0.1 with rule-based detection (no deep learning yet).

## Project Structure

```
Vibesing/
├── docs/                    # Technical docs, PRD, roadmap
├── python/                  # DSP validation pipeline (Week 1)
│   ├── dsp_features.py      # Mel + F0 + HNR + Energy extraction
│   ├── visualize_features.py # 4-panel visualization
│   ├── rule_detector.py     # Rule-based squeeze detection
│   └── requirements.txt
└── ios/Vibesing/            # iOS TestFlight App
    ├── Audio/
    │   ├── AudioCaptureEngine.swift  # AVAudioEngine → 16kHz
    │   ├── DSPProcessor.swift        # On-device Mel/HNR/F0/Energy
    │   └── SqueezeDetector.swift     # Rule-based detection + EMA + state filter
    ├── UI/
    │   ├── ContentView.swift         # Main screen
    │   ├── SqueezeIndicator.swift    # Red/green light
    │   └── EnergyBar.swift           # Energy bar + stats
    └── Data/
        └── FeedbackStore.swift       # SQLite feedback storage
```

## Locked Technical Decisions (DO NOT change)

1. **MVP is squeeze detection only** (binary). No mix voice, no resonance, no other dimensions.
2. **Sample rate: 16kHz** everywhere. No 44.1kHz pipeline.
3. **Input features: Mel(80) + HNR + Energy**. CPP is debug-only (unstable on mobile). F0 is voicing mask only, not model input.
4. **Model architecture: Causal Depthwise-Separable CNN (TCN)**. No Transformer/Conformer.
5. **Smoothing: EMA (signal layer) + State machine (semantic layer)**. No Kalman Filter.
6. **Mix voice ratio (V0.2+): 5-class ordinal classification**, not regression.

## Squeeze Detection Algorithm

Python and iOS implementations use identical parameters. When modifying thresholds, **always update both files**:

| Parameter | Python file | iOS file | Default |
|-----------|------------|----------|---------|
| HNR threshold | `rule_detector.py: HNR_SQUEEZE_THRESHOLD` | `SqueezeDetector.swift: hnrSqueezeThreshold` | 12.0 dB |
| Energy floor | `rule_detector.py: ENERGY_MIN_THRESHOLD` | `SqueezeDetector.swift: energyMinThreshold` | -40.0 dB |
| F0 high voice | `rule_detector.py: F0_HIGH_VOICE_THRESHOLD` | `SqueezeDetector.swift: f0HighVoiceThreshold` | 220 Hz |
| EMA alpha | `EMAFilter.__init__` | `SqueezeDetector: emaAlpha` | 0.4 |
| Trigger frames | `SqueezeStateFilter.MIN_FRAMES_TO_TRIGGER` | `SqueezeDetector: minFramesToTrigger` | 6 (96ms) |
| Clear frames | `SqueezeStateFilter.MIN_FRAMES_TO_CLEAR` | `SqueezeDetector: minFramesToClear` | 10 (160ms) |
| Threshold on | `SqueezeStateFilter.THRESHOLD_ON` | `SqueezeDetector: thresholdOn` | 0.55 |
| Threshold off | `SqueezeStateFilter.THRESHOLD_OFF` | `SqueezeDetector: thresholdOff` | 0.30 |

## Detection Logic (pseudocode)

```
1. Gate: if energy < -40dB OR f0 < 50Hz → prob = 0 (silence/unvoiced)
2. HNR → probability: sigmoid(0.5 * (12 - HNR))
3. Pitch boost: if f0 > 220Hz → multiply by 1.0-1.3
4. EMA smooth: α = 0.4
5. State filter: 6+ frames > 0.55 → RED; 10+ frames < 0.30 → GREEN
6. Score: 100 × (1 - smoothed_prob)
```

## Key Technical Constraints

- End-to-end latency: < 100ms
- Core features work fully offline (on-device)
- iOS 17+, iPhone 12+ (A14 chip minimum)
- Model params (V1.0+): < 1M
- All audio processing at 16kHz mono

## Common Tasks

### Adjusting squeeze sensitivity
Edit both `HNR_SQUEEZE_THRESHOLD` in Python and `hnrSqueezeThreshold` in Swift. Lower = less sensitive, higher = more sensitive.

### Adding a new detection dimension
1. Add field to `DSPProcessor.FrameResult` (Swift) and `FrameFeatures` (Python)
2. Add processing logic in `SqueezeDetector` / `rule_detector.py`
3. Add UI element in `ContentView.swift`
4. Keep Python and iOS logic identical

### Training a DL model (future)
- Architecture: 4-layer Causal DWSep CNN, <300K params
- Input tensor: [82 × 32] (80 mel + HNR + energy, 32 frames @ 16ms)  
- Output: squeeze_prob (sigmoid)
- Export: `coremltools` → `.mlpackage` (Float16)
- Training tricks: Label Smoothing ε=0.1, random gain [-12, +6] dB, energy normalization

## Code Style

- **Python**: Type hints, dataclasses, numpy-first (avoid loops where vectorizable)
- **Swift**: SwiftUI, Combine publishers, `@Published` for UI state, `@MainActor` for UI updates
- **Both**: Keep detection logic in sync between Python and iOS at all times

## Vocabulary

- 挤卡 = Vocal Squeeze / Constriction
- 混声 = Mix Voice (chest+head resonance blend)
- 大白嗓 = Yelling / Untrained belting
- 喉位高 = High Larynx
- HNR = Harmonics-to-Noise Ratio
- CPP = Cepstral Peak Prominence
- EMA = Exponential Moving Average

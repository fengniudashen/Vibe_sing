# Vibesing 高音觉醒 🎤

Real-time vocal squeeze (挤卡) detection for iOS — on-device DSP + AI coaching.

## What is Vibesing?

Vibesing detects **vocal squeeze** (vocal constriction / 挤卡) in real-time while you sing, giving you instant visual feedback to improve your technique. No internet required — all processing happens on-device.

## Project Structure

```
Vibesing/
├── docs/                    # Technical docs, PRD, roadmap
├── python/                  # DSP validation pipeline
│   ├── dsp_features.py      # Mel + F0 + HNR + Energy extraction
│   ├── visualize_features.py # 4-panel visualization
│   ├── rule_detector.py     # Rule-based squeeze detection
│   └── requirements.txt
├── ios/Vibesing/            # iOS TestFlight App
│   ├── Audio/               # AudioEngine, DSP, SqueezeDetector
│   ├── UI/                  # SwiftUI views
│   └── Data/                # SQLite feedback storage
└── web/                     # Interactive demos & prototypes
    ├── index.html           # Project dashboard
    ├── prototype.html       # V0.1 onboarding prototype
    └── tech-architecture.html # Technical pipeline visualization
```

## MVP V0.1 — Squeeze Detection

| Feature | Detail |
|---------|--------|
| Detection | Vocal squeeze (挤卡) — binary red/green |
| Method | Rule-based: HNR → sigmoid → EMA → State machine |
| Latency | < 100ms end-to-end |
| Audio | 16kHz mono, 1024 FFT, 256 hop |
| Platform | iOS 17+, iPhone 12+ (A14) |

## Detection Algorithm

```
1. Gate: energy < -40dB OR f0 < 50Hz → silence
2. HNR → probability: sigmoid(0.5 × (12 - HNR))
3. Pitch boost: f0 > 220Hz → ×1.0–1.3
4. EMA smooth: α = 0.4
5. State filter: 6 frames > 0.55 → RED; 10 frames < 0.30 → GREEN
6. Score: 100 × (1 - smoothed_prob)
```

## Quick Start

### Python DSP Pipeline

```bash
cd python
pip install -r requirements.txt
python rule_detector.py your_audio.wav
```

### iOS App

Open `ios/Vibesing/` in Xcode, build and run on a physical device (microphone required).

## Roadmap

| Version | Focus | Timeline |
|---------|-------|----------|
| V0.1 | Squeeze detection (rule-based) | Week 1-2 |
| V0.2 | DL model + Mix voice | Month 2 |
| V1.0 | 5 core dimensions | Month 4 |
| V2.0 | Social features | Month 6 |

## Documentation

- [MVP Engineering Spec](docs/Vibesing_MVP_V0.1_Engineering_Spec.md)
- [Full Product Roadmap](docs/FULL_PRODUCT_ROADMAP.md)
- [Product Requirements (PRD)](docs/PRD_Product_Requirements.md)
- [Algorithm Selection Guide](docs/Algorithm_Selection_Guide.md)
- [Vocal Science ↔ Product](docs/Vocal_Science_Product_Connection.md)
- [Vocal Diagnosis Handbook](docs/Vocal_Problems_Diagnosis_Handbook.md)

## Tech Stack

- **iOS**: Swift 5.9, SwiftUI, AVAudioEngine, Accelerate/vDSP
- **Python**: librosa, parselmouth (Praat), numpy, matplotlib
- **Future**: PyTorch → CoreML (Causal DWSep CNN, <300K params)

## License

All rights reserved. © 2024 Vibesing

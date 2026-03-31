// !$*UTF8*$!
// Vibesing iOS MVP V0.1
// This is a minimal Xcode project reference.
// Open Vibesing.xcodeproj in Xcode to build.
//
// Structure:
//   Vibesing/
//     VibesingApp.swift          — App entry point
//     Audio/
//       AudioCaptureEngine.swift — AVAudioEngine real-time capture
//       DSPProcessor.swift       — Mel + HNR + F0 + Energy extraction
//       SqueezeDetector.swift    — Rule-based squeeze detection + EMA + state filter
//     UI/
//       ContentView.swift        — Main screen with red/green light
//       SqueezeIndicator.swift   — Animated red/green light component
//       EnergyBar.swift          — Real-time energy level bar
//     Data/
//       FeedbackStore.swift      — Local storage for "Was this accurate?" feedback
//       AudioClipRecorder.swift  — Saves 2s audio clips around squeeze events

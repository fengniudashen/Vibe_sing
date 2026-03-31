import Foundation

/// Rule-based squeeze detector with EMA smoothing and state filter.
/// Mirrors the Python `rule_detector.py` logic for consistency.
final class SqueezeDetector: ObservableObject {
    
    // MARK: - Thresholds (match Python rule_detector.py)
    static let hnrSqueezeThreshold: Float = 12.0   // dB
    static let energyMinThreshold: Float = -40.0    // dB (silence gate)
    static let f0HighVoiceThreshold: Float = 220.0  // Hz
    
    // MARK: - Published State (drives UI)
    @Published var squeezeProbability: Float = 0.0   // smoothed, 0-1
    @Published var isSqueezeActive: Bool = false      // after state filter
    @Published var healthScore: Int = 100             // 0-100
    @Published var currentF0: Float = 0.0             // Hz
    @Published var currentHNR: Float = 0.0            // dB
    @Published var currentEnergy: Float = -80.0       // dB
    @Published var isSinging: Bool = false             // true if energy + f0 present
    
    // MARK: - Private State
    private var emaValue: Float = 0.0
    private var emaInitialized = false
    private let emaAlpha: Float = 0.4
    
    // State filter
    private var stateActive = false
    private var consecutiveOn = 0
    private var consecutiveOff = 0
    private let minFramesToTrigger = 6   // ~96ms
    private let minFramesToClear = 10    // ~160ms
    private let thresholdOn: Float = 0.55
    private let thresholdOff: Float = 0.30
    
    // MARK: - Process Frame
    
    /// Call this for every DSP frame result (~62.5 fps).
    /// Updates all published properties on the main thread actor.
    @MainActor
    func processFrame(_ frame: DSPProcessor.FrameResult) {
        currentF0 = frame.f0
        currentHNR = frame.hnr
        currentEnergy = frame.energy
        
        // Gate: detect if user is actually singing
        let singing = frame.energy > Self.energyMinThreshold && frame.f0 > 50
        isSinging = singing
        
        // Compute raw squeeze probability
        let rawProb: Float
        if !singing {
            rawProb = 0
        } else {
            rawProb = computeRawProbability(hnr: frame.hnr, energy: frame.energy, f0: frame.f0)
        }
        
        // EMA smooth
        let smoothed = applyEMA(rawProb)
        squeezeProbability = smoothed
        
        // State filter
        isSqueezeActive = applyStateFilter(smoothed)
        
        // Score
        healthScore = max(0, min(100, Int(round(100.0 * (1.0 - smoothed)))))
    }
    
    /// Reset all state (new session)
    func reset() {
        emaValue = 0
        emaInitialized = false
        stateActive = false
        consecutiveOn = 0
        consecutiveOff = 0
        squeezeProbability = 0
        isSqueezeActive = false
        healthScore = 100
        currentF0 = 0
        currentHNR = 0
        currentEnergy = -80
        isSinging = false
    }
    
    // MARK: - Private
    
    private func computeRawProbability(hnr: Float, energy: Float, f0: Float) -> Float {
        // Sigmoid mapping: HNR = 12 → 0.5, HNR = 6 → ~0.9, HNR = 20 → ~0.1
        let hnrSignal = 1.0 / (1.0 + exp(0.5 * (hnr - Self.hnrSqueezeThreshold)))
        
        // Pitch boost
        let pitchBoost: Float
        if f0 > Self.f0HighVoiceThreshold {
            pitchBoost = min(1.3, 1.0 + (f0 - Self.f0HighVoiceThreshold) / 500.0)
        } else {
            pitchBoost = 0.7
        }
        
        return min(max(hnrSignal * pitchBoost, 0), 1)
    }
    
    private func applyEMA(_ value: Float) -> Float {
        if !emaInitialized {
            emaValue = value
            emaInitialized = true
        } else {
            emaValue = emaAlpha * value + (1.0 - emaAlpha) * emaValue
        }
        return emaValue
    }
    
    private func applyStateFilter(_ smoothProb: Float) -> Bool {
        if stateActive {
            if smoothProb < thresholdOff {
                consecutiveOff += 1
                consecutiveOn = 0
                if consecutiveOff >= minFramesToClear {
                    stateActive = false
                }
            } else {
                consecutiveOff = 0
            }
        } else {
            if smoothProb > thresholdOn {
                consecutiveOn += 1
                consecutiveOff = 0
                if consecutiveOn >= minFramesToTrigger {
                    stateActive = true
                }
            } else {
                consecutiveOn = 0
            }
        }
        return stateActive
    }
}

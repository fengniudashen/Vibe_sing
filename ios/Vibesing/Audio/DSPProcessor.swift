import Accelerate
import Foundation

/// Lightweight DSP processor that extracts frame-level features from 16kHz audio.
/// Features: Mel Spectrogram, HNR, RMS Energy.
/// Designed for real-time streaming — processes one hop (256 samples) at a time.
final class DSPProcessor {
    
    // MARK: - Constants
    static let sampleRate: Float = 16000
    static let nFFT = 1024               // 64ms window
    static let hopLength = 256            // 16ms hop
    static let nMels = 80
    static let fMin: Float = 50
    static let fMax: Float = 8000
    
    // MARK: - State
    
    /// Ring buffer holding the last nFFT samples for STFT
    private var ringBuffer: [Float]
    private var ringWriteIndex = 0
    private var ringFilled = false // true once we have nFFT samples
    
    /// Precomputed Mel filterbank matrix [nMels × (nFFT/2 + 1)]
    private let melFilterbank: [[Float]]
    
    /// Hann window
    private let window: [Float]
    
    /// FFT setup (vDSP)
    private let fftSetup: vDSP.FFT<DSPSplitComplex>?
    private let log2n: vDSP_Length
    
    // MARK: - Output
    
    struct FrameResult {
        let melSpectrum: [Float]   // 80-dim log-mel, dB
        let energy: Float          // RMS energy, dB
        let hnr: Float             // Harmonics-to-Noise Ratio, dB
        let f0: Float              // Fundamental frequency, Hz (0 = unvoiced)
    }
    
    // MARK: - Init
    
    init() {
        ringBuffer = [Float](repeating: 0, count: Self.nFFT)
        
        // Hann window
        window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: Self.nFFT, isHalfWindow: false)
        
        // FFT setup
        log2n = vDSP_Length(log2(Double(Self.nFFT)))
        fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)
        
        // Build Mel filterbank
        melFilterbank = Self.buildMelFilterbank(
            nMels: Self.nMels,
            nFFT: Self.nFFT,
            sampleRate: Self.sampleRate,
            fMin: Self.fMin,
            fMax: Self.fMax
        )
    }
    
    // MARK: - Streaming Interface
    
    /// Feed 256 new samples (one hop). Returns a FrameResult if we have enough context.
    func pushSamples(_ samples: [Float]) -> FrameResult? {
        guard samples.count == Self.hopLength else { return nil }
        
        // Write into ring buffer
        for sample in samples {
            ringBuffer[ringWriteIndex] = sample
            ringWriteIndex = (ringWriteIndex + 1) % Self.nFFT
            if ringWriteIndex == 0 { ringFilled = true }
        }
        
        // Need at least nFFT samples before first frame
        guard ringFilled || ringWriteIndex >= Self.nFFT else { return nil }
        
        // Extract current window from ring buffer (linearized)
        let frame = linearizeRingBuffer()
        
        // Compute features
        let melSpectrum = computeMelSpectrum(frame)
        let energy = computeEnergy(frame)
        let (f0, hnr) = computeF0AndHNR(frame)
        
        return FrameResult(
            melSpectrum: melSpectrum,
            energy: energy,
            hnr: hnr,
            f0: f0
        )
    }
    
    /// Reset all state (e.g., when starting a new recording session)
    func reset() {
        ringBuffer = [Float](repeating: 0, count: Self.nFFT)
        ringWriteIndex = 0
        ringFilled = false
    }
    
    // MARK: - Feature Computation
    
    private func linearizeRingBuffer() -> [Float] {
        var frame = [Float](repeating: 0, count: Self.nFFT)
        for i in 0..<Self.nFFT {
            frame[i] = ringBuffer[(ringWriteIndex + i) % Self.nFFT]
        }
        return frame
    }
    
    private func computeMelSpectrum(_ frame: [Float]) -> [Float] {
        // Apply window
        var windowed = [Float](repeating: 0, count: Self.nFFT)
        vDSP_vmul(frame, 1, window, 1, &windowed, 1, vDSP_Length(Self.nFFT))
        
        // FFT
        let halfN = Self.nFFT / 2
        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)
        
        realPart.withUnsafeMutableBufferPointer { realBuf in
            imagPart.withUnsafeMutableBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                windowed.withUnsafeBufferPointer { inputBuf in
                    inputBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }
                }
                fftSetup?.forward(input: splitComplex, output: &splitComplex)
            }
        }
        
        // Power spectrum
        var powerSpectrum = [Float](repeating: 0, count: halfN + 1)
        for i in 0..<halfN {
            powerSpectrum[i] = realPart[i] * realPart[i] + imagPart[i] * imagPart[i]
        }
        
        // Apply Mel filterbank
        var melSpectrum = [Float](repeating: 0, count: Self.nMels)
        for m in 0..<Self.nMels {
            var sum: Float = 0
            let filterLen = min(melFilterbank[m].count, powerSpectrum.count)
            for k in 0..<filterLen {
                sum += melFilterbank[m][k] * powerSpectrum[k]
            }
            // Log scale (dB), floor at 1e-10
            melSpectrum[m] = 10.0 * log10(max(sum, 1e-10))
        }
        
        return melSpectrum
    }
    
    private func computeEnergy(_ frame: [Float]) -> Float {
        var rms: Float = 0
        vDSP_rmsqv(frame, 1, &rms, vDSP_Length(frame.count))
        return 20.0 * log10(max(rms, 1e-10))
    }
    
    /// Simple autocorrelation-based F0 and HNR estimation.
    /// F0: peak of normalized autocorrelation in the pitch period range.
    /// HNR: ratio of autocorrelation peak to (1 - peak), converted to dB.
    private func computeF0AndHNR(_ frame: [Float]) -> (f0: Float, hnr: Float) {
        let n = frame.count
        
        // Pitch period search range (in samples)
        let minPeriod = Int(Self.sampleRate / 600)  // 600 Hz ceiling
        let maxPeriod = Int(Self.sampleRate / 75)    // 75 Hz floor
        
        guard maxPeriod < n else { return (0, -10) }
        
        // Normalized autocorrelation
        var autocorr = [Float](repeating: 0, count: maxPeriod + 1)
        
        // Energy of the frame
        var frameEnergy: Float = 0
        vDSP_dotpr(frame, 1, frame, 1, &frameEnergy, vDSP_Length(n))
        
        guard frameEnergy > 1e-8 else { return (0, -10) }
        
        for lag in minPeriod...maxPeriod {
            var sum: Float = 0
            let overlapLength = n - lag
            vDSP_dotpr(frame, 1, Array(frame[lag...]), 1, &sum, vDSP_Length(overlapLength))
            
            // Normalize by geometric mean of energies
            var lagEnergy: Float = 0
            let lagSlice = Array(frame[lag..<n])
            vDSP_dotpr(lagSlice, 1, lagSlice, 1, &lagEnergy, vDSP_Length(overlapLength))
            
            let headSlice = Array(frame[0..<overlapLength])
            var headEnergy: Float = 0
            vDSP_dotpr(headSlice, 1, headSlice, 1, &headEnergy, vDSP_Length(overlapLength))
            
            let norm = sqrt(headEnergy * lagEnergy)
            autocorr[lag] = norm > 1e-8 ? sum / norm : 0
        }
        
        // Find peak
        var peakValue: Float = 0
        var peakLag = 0
        for lag in minPeriod...maxPeriod {
            if autocorr[lag] > peakValue {
                peakValue = autocorr[lag]
                peakLag = lag
            }
        }
        
        // Voicing decision
        let voicingThreshold: Float = 0.4
        guard peakValue > voicingThreshold, peakLag > 0 else {
            return (0, -10)  // unvoiced
        }
        
        let f0 = Self.sampleRate / Float(peakLag)
        
        // HNR in dB: HNR = 10 * log10(r / (1 - r)) where r is autocorrelation peak
        let clampedPeak = min(peakValue, 0.9999)
        let hnr = 10.0 * log10(clampedPeak / (1.0 - clampedPeak))
        
        return (f0, hnr)
    }
    
    // MARK: - Mel Filterbank Construction
    
    private static func hzToMel(_ hz: Float) -> Float {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }
    
    private static func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
    
    private static func buildMelFilterbank(
        nMels: Int, nFFT: Int, sampleRate: Float, fMin: Float, fMax: Float
    ) -> [[Float]] {
        let nBins = nFFT / 2 + 1
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        
        // nMels + 2 equally spaced points in Mel scale
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }
        
        // Convert to Hz, then to FFT bin indices
        let hzPoints = melPoints.map { melToHz($0) }
        let binPoints = hzPoints.map { Int(round($0 * Float(nFFT) / sampleRate)) }
        
        // Build triangular filters
        var filterbank = [[Float]]()
        for m in 0..<nMels {
            var filter = [Float](repeating: 0, count: nBins)
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]
            
            // Rising slope
            if center > left {
                for k in left..<center where k < nBins {
                    filter[k] = Float(k - left) / Float(center - left)
                }
            }
            // Falling slope
            if right > center {
                for k in center...right where k < nBins {
                    filter[k] = Float(right - k) / Float(right - center)
                }
            }
            
            filterbank.append(filter)
        }
        
        return filterbank
    }
}

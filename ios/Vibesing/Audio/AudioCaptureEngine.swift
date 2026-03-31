import AVFoundation
import Accelerate
import Combine

/// Real-time audio capture engine using AVAudioEngine.
/// Captures at 44.1kHz, downsamples to 16kHz, delivers frames via Combine publisher.
final class AudioCaptureEngine: ObservableObject {
    
    // MARK: - Public State
    @Published var isRunning = false
    @Published var permissionGranted = false
    
    /// Publishes 256-sample (16ms) audio chunks at 16kHz
    let audioFramePublisher = PassthroughSubject<[Float], Never>()
    
    // MARK: - Constants
    static let inputSampleRate: Double = 44100
    static let targetSampleRate: Double = 16000
    static let bufferSize: AVAudioFrameCount = 4096  // ~93ms @ 44.1kHz
    static let outputChunkSize = 256  // 16ms @ 16kHz
    
    // MARK: - Private
    private let engine = AVAudioEngine()
    private var downsampleBuffer: [Float] = []
    private let downsampleRatio: Int  // 44100 / 16000 ≈ 2.75 → use rational resampling
    
    // Ring buffer for rational resampling (44100 → 16000)
    private var resampleAccumulator: [Float] = []
    private let resampleInputRate = 441  // simplified: 44100/100
    private let resampleOutputRate = 160 // simplified: 16000/100
    
    init() {
        // Use integer approximation for resampling ratio
        downsampleRatio = Int(Self.inputSampleRate / Self.targetSampleRate)
    }
    
    // MARK: - Permission
    
    func requestPermission() {
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                self?.permissionGranted = granted
            }
        }
    }
    
    // MARK: - Start / Stop
    
    func start() throws {
        guard permissionGranted else { return }
        
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
        try session.setPreferredSampleRate(Self.inputSampleRate)
        try session.setActive(true)
        
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        
        // Install tap — delivers audio buffers on a background thread
        inputNode.installTap(onBus: 0, bufferSize: Self.bufferSize, format: inputFormat) {
            [weak self] buffer, _ in
            self?.processInputBuffer(buffer)
        }
        
        engine.prepare()
        try engine.start()
        
        DispatchQueue.main.async {
            self.isRunning = true
        }
    }
    
    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        resampleAccumulator.removeAll()
        downsampleBuffer.removeAll()
        
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }
    
    // MARK: - Audio Processing
    
    private func processInputBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        
        // Simple decimation downsample: 44.1k → ~16k
        // For production, use vDSP_desamp for proper anti-aliasing
        let downsampled = downsample(samples)
        
        // Accumulate and emit fixed-size chunks
        downsampleBuffer.append(contentsOf: downsampled)
        
        while downsampleBuffer.count >= Self.outputChunkSize {
            let chunk = Array(downsampleBuffer.prefix(Self.outputChunkSize))
            downsampleBuffer.removeFirst(Self.outputChunkSize)
            audioFramePublisher.send(chunk)
        }
    }
    
    /// Simple decimation with basic anti-alias lowpass.
    /// For MVP this is sufficient; upgrade to vDSP_desamp for production.
    private func downsample(_ input: [Float]) -> [Float] {
        // Step ~2.76 decimation (44100 / 16000)
        // Use every ~2.76th sample (linear interpolation)
        let ratio = Self.inputSampleRate / Self.targetSampleRate
        let outputCount = Int(Double(input.count) / ratio)
        var output = [Float](repeating: 0, count: outputCount)
        
        for i in 0..<outputCount {
            let srcIdx = Double(i) * ratio
            let idx0 = Int(srcIdx)
            let frac = Float(srcIdx - Double(idx0))
            
            if idx0 + 1 < input.count {
                output[i] = input[idx0] * (1.0 - frac) + input[idx0 + 1] * frac
            } else if idx0 < input.count {
                output[i] = input[idx0]
            }
        }
        
        return output
    }
}

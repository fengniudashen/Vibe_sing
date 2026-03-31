import SwiftUI
import Combine

/// Main screen: red/green light + energy bar + score + feedback button.
struct ContentView: View {
    
    @StateObject private var audioEngine = AudioCaptureEngine()
    @StateObject private var squeezeDetector = SqueezeDetector()
    @StateObject private var feedbackStore = FeedbackStore()
    
    private let dspProcessor = DSPProcessor()
    
    @State private var cancellables = Set<AnyCancellable>()
    @State private var showFeedbackPrompt = false
    @State private var lastSqueezeEndTime: Date? = nil
    
    var body: some View {
        ZStack {
            // Background
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 4) {
                    Text("高音觉醒")
                        .font(.largeTitle.bold())
                        .foregroundColor(.white)
                    Text("Vibesing MVP V0.1")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .padding(.top, 20)
                
                Spacer()
                
                // Main squeeze indicator
                SqueezeIndicator(
                    isSqueezing: squeezeDetector.isSqueezeActive,
                    probability: squeezeDetector.squeezeProbability,
                    isSinging: squeezeDetector.isSinging
                )
                
                // Score
                VStack(spacing: 4) {
                    Text("\(squeezeDetector.healthScore)")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(scoreColor)
                        .contentTransition(.numericText())
                        .animation(.easeInOut(duration: 0.2), value: squeezeDetector.healthScore)
                    
                    Text("发声健康度")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                
                Spacer()
                
                // Energy bar + stats
                EnergyBar(
                    energy: squeezeDetector.currentEnergy,
                    hnr: squeezeDetector.currentHNR,
                    f0: squeezeDetector.currentF0,
                    isSinging: squeezeDetector.isSinging
                )
                .padding(.horizontal, 24)
                
                // Start/Stop button
                Button(action: toggleRecording) {
                    HStack(spacing: 8) {
                        Image(systemName: audioEngine.isRunning ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.title2)
                        Text(audioEngine.isRunning ? "停止" : "开始演唱")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(audioEngine.isRunning ? Color.red.opacity(0.8) : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
            }
            
            // Feedback overlay
            if showFeedbackPrompt {
                FeedbackOverlay(
                    onAccurate: {
                        feedbackStore.recordFeedback(accurate: true)
                        showFeedbackPrompt = false
                    },
                    onInaccurate: {
                        feedbackStore.recordFeedback(accurate: false)
                        showFeedbackPrompt = false
                    },
                    onDismiss: {
                        showFeedbackPrompt = false
                    }
                )
                .transition(.opacity.combined(with: .scale))
            }
        }
        .onAppear {
            audioEngine.requestPermission()
        }
        .onChange(of: squeezeDetector.isSqueezeActive) { _, isActive in
            // Show feedback prompt when squeeze ends
            if !isActive && lastSqueezeEndTime == nil {
                lastSqueezeEndTime = Date()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    showFeedbackPrompt = true
                    // Auto-dismiss after 5 seconds
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                        showFeedbackPrompt = false
                    }
                }
            }
            if isActive {
                lastSqueezeEndTime = nil
                showFeedbackPrompt = false
            }
        }
    }
    
    private var scoreColor: Color {
        let score = squeezeDetector.healthScore
        if score >= 80 { return .green }
        if score >= 50 { return .yellow }
        return .red
    }
    
    private func toggleRecording() {
        if audioEngine.isRunning {
            audioEngine.stop()
            dspProcessor.reset()
            squeezeDetector.reset()
        } else {
            do {
                try audioEngine.start()
                setupAudioPipeline()
            } catch {
                print("Failed to start audio engine: \(error)")
            }
        }
    }
    
    private func setupAudioPipeline() {
        // Subscribe to audio frames from the capture engine
        audioEngine.audioFramePublisher
            .receive(on: DispatchQueue.global(qos: .userInteractive))
            .compactMap { [dspProcessor] samples in
                dspProcessor.pushSamples(samples)
            }
            .receive(on: DispatchQueue.main)
            .sink { [squeezeDetector] frame in
                squeezeDetector.processFrame(frame)
            }
            .store(in: &cancellables)
    }
}

// MARK: - Feedback Overlay

struct FeedbackOverlay: View {
    let onAccurate: () -> Void
    let onInaccurate: () -> Void
    let onDismiss: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Text("刚才红灯亮了，你觉得准吗？")
                .font(.headline)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
            
            Text("Was the squeeze detection accurate?")
                .font(.caption)
                .foregroundColor(.gray)
            
            HStack(spacing: 20) {
                Button(action: onAccurate) {
                    VStack(spacing: 4) {
                        Text("👍")
                            .font(.title)
                        Text("准")
                            .font(.caption.bold())
                    }
                    .frame(width: 80, height: 70)
                    .background(Color.green.opacity(0.3))
                    .cornerRadius(12)
                }
                
                Button(action: onInaccurate) {
                    VStack(spacing: 4) {
                        Text("👎")
                            .font(.title)
                        Text("不准")
                            .font(.caption.bold())
                    }
                    .frame(width: 80, height: 70)
                    .background(Color.red.opacity(0.3))
                    .cornerRadius(12)
                }
            }
            .foregroundColor(.white)
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(.ultraThinMaterial)
        )
        .padding(.horizontal, 40)
        .onTapGesture { } // absorb taps
        .background(Color.black.opacity(0.3).ignoresSafeArea().onTapGesture(perform: onDismiss))
    }
}

#Preview {
    ContentView()
}

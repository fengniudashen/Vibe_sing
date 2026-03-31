import SwiftUI

/// Animated squeeze indicator — the core "red light / green light" component.
struct SqueezeIndicator: View {
    let isSqueezing: Bool
    let probability: Float
    let isSinging: Bool
    
    @State private var pulseScale: CGFloat = 1.0
    
    private var indicatorColor: Color {
        if !isSinging {
            return .gray.opacity(0.4)
        }
        return isSqueezing ? .red : .green
    }
    
    private var glowRadius: CGFloat {
        if !isSinging { return 0 }
        return isSqueezing ? 20 : 8
    }
    
    private var statusText: String {
        if !isSinging { return "等待演唱..." }
        return isSqueezing ? "挤卡警告！" : "发声健康"
    }
    
    private var statusSubtext: String {
        if !isSinging { return "Start singing" }
        return isSqueezing ? "SQUEEZE DETECTED — Relax your throat" : "GOOD — Keep it up!"
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Main indicator circle
            ZStack {
                // Outer glow
                Circle()
                    .fill(indicatorColor.opacity(0.2))
                    .frame(width: 160, height: 160)
                    .scaleEffect(pulseScale)
                
                // Inner solid circle
                Circle()
                    .fill(indicatorColor)
                    .frame(width: 120, height: 120)
                    .shadow(color: indicatorColor.opacity(0.6), radius: glowRadius)
                
                // Center icon
                Image(systemName: isSqueezing ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.system(size: 44, weight: .bold))
                    .foregroundColor(.white)
            }
            .animation(.easeInOut(duration: 0.3), value: isSqueezing)
            .onChange(of: isSqueezing) { _, newValue in
                if newValue {
                    // Pulse animation when squeeze triggers
                    withAnimation(.easeInOut(duration: 0.15).repeatCount(3, autoreverses: true)) {
                        pulseScale = 1.15
                    }
                    // Reset after animation
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        withAnimation { pulseScale = 1.0 }
                    }
                }
            }
            
            // Status text
            Text(statusText)
                .font(.title2.bold())
                .foregroundColor(indicatorColor)
            
            Text(statusSubtext)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

#Preview {
    VStack(spacing: 40) {
        SqueezeIndicator(isSqueezing: false, probability: 0.1, isSinging: true)
        SqueezeIndicator(isSqueezing: true, probability: 0.8, isSinging: true)
        SqueezeIndicator(isSqueezing: false, probability: 0.0, isSinging: false)
    }
    .padding()
}

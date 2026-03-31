import SwiftUI

/// Real-time energy level bar with smooth animation.
struct EnergyBar: View {
    let energy: Float        // dB, typically -80 to 0
    let hnr: Float           // dB
    let f0: Float            // Hz
    let isSinging: Bool
    
    /// Normalized energy level 0-1
    private var normalizedEnergy: CGFloat {
        // Map -60dB..0dB to 0..1
        let clamped = max(-60, min(0, energy))
        return CGFloat((clamped + 60) / 60)
    }
    
    var body: some View {
        VStack(spacing: 12) {
            // Energy bar
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("能量 Energy")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.0f dB", energy))
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                }
                
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.gray.opacity(0.2))
                        
                        RoundedRectangle(cornerRadius: 4)
                            .fill(
                                LinearGradient(
                                    colors: [.green, .yellow, .orange, .red],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * normalizedEnergy)
                    }
                }
                .frame(height: 8)
                .animation(.linear(duration: 0.05), value: normalizedEnergy)
            }
            
            // Stats row
            HStack(spacing: 20) {
                StatBadge(label: "F0", value: f0 > 0 ? String(format: "%.0f Hz", f0) : "—", color: .blue)
                StatBadge(label: "HNR", value: String(format: "%.1f dB", hnr), color: hnr < 12 ? .red : .green)
                StatBadge(label: "状态", value: isSinging ? "演唱中" : "静音", color: isSinging ? .blue : .gray)
            }
        }
    }
}

struct StatBadge: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption.monospacedDigit().bold())
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    EnergyBar(energy: -20, hnr: 15.3, f0: 440, isSinging: true)
        .padding()
}

import SwiftUI

/// Overlay legend showing detected interaction types with color coding.
/// Only shows types that are present in the current interaction set.
struct InteractionLegendView: View {
    let interactions: [MolecularInteraction]

    private var presentTypes: [MolecularInteraction.InteractionType] {
        let types = Set(interactions.map(\.type))
        return MolecularInteraction.InteractionType.allCases.filter { types.contains($0) }
    }

    var body: some View {
        if !presentTypes.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text("Interactions")
                    .font(.footnote.weight(.bold))
                    .foregroundStyle(.secondary)

                ForEach(presentTypes, id: \.rawValue) { type in
                    let count = interactions.filter { $0.type == type }.count
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(Color(
                                red: Double(type.color.x),
                                green: Double(type.color.y),
                                blue: Double(type.color.z)
                            ))
                            .frame(width: 16, height: 4)

                        Text(type.label)
                            .font(.footnote)
                            .foregroundStyle(.primary)

                        Text("(\(count))")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

struct SurfaceLegendView: View {
    let mode: SurfaceColorMode
    let legend: SurfaceLegend

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(legend.title)
                .font(.footnote.weight(.bold))
                .foregroundStyle(.secondary)

            switch legend.kind {
            case .gradient(let minLabel, let midLabel, let maxLabel, let colors):
                VStack(alignment: .leading, spacing: 4) {
                    LinearGradient(
                        colors: colors.map(Self.color(from:)),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(width: 112, height: 10)
                    .clipShape(Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5)
                    )

                    HStack(spacing: 0) {
                        Text(minLabel)
                        Spacer()
                        if let midLabel {
                            Text(midLabel)
                            Spacer()
                        }
                        Text(maxLabel)
                    }
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(width: 112)
                }
            case .categorical(let entries):
                ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Self.color(from: entry.color))
                            .frame(width: 14, height: 8)

                        Text(entry.label)
                            .font(.footnote)
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
        .padding(10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityIdentifier(accessibilityID)
    }

    private var accessibilityID: String {
        switch mode {
        case .uniform: return "surface_legend_uniform"
        case .esp: return "surface_legend_esp"
        case .hydrophobicity: return "surface_legend_hydrophobicity"
        case .pharmacophore: return "surface_legend_pharmacophore"
        }
    }

    private static func color(from simd: SIMD4<Float>) -> Color {
        Color(
            red: Double(simd.x),
            green: Double(simd.y),
            blue: Double(simd.z),
            opacity: Double(simd.w)
        )
    }
}

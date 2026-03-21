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
            VStack(alignment: .leading, spacing: 3) {
                Text("Interactions")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)

                ForEach(presentTypes, id: \.rawValue) { type in
                    let count = interactions.filter { $0.type == type }.count
                    HStack(spacing: 5) {
                        RoundedRectangle(cornerRadius: 1)
                            .fill(Color(
                                red: Double(type.color.x),
                                green: Double(type.color.y),
                                blue: Double(type.color.z)
                            ))
                            .frame(width: 14, height: 3)

                        Text(type.label)
                            .font(.system(size: 9))
                            .foregroundStyle(.primary)

                        Text("(\(count))")
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .padding(8)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

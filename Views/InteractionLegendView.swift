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

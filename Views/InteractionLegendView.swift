// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Interaction palette

extension MolecularInteraction.InteractionType {
    /// Theme color for 2D diagrams, lists and charts. Each hue matches the
    /// type's 3D viewport line (`color`) so the 2D diagram reads like the 3D
    /// view; the light variant is darkened for contrast on white paper, the dark
    /// variant stays close to the viewport color.
    var themeColor: Color {
        switch self {
        case .hbond:       DS.dynamic(light: 0x0A78B4, dark: 0x4FD1FF)   // cyan
        case .hydrophobic: DS.dynamic(light: 0x6E6E73, dark: 0xA8A8AE)   // gray
        case .saltBridge:  DS.dynamic(light: 0xC85A00, dark: 0xFF9A4D)   // orange
        case .piStack:     DS.dynamic(light: 0x8A3FD1, dark: 0xC38BFF)   // purple
        case .piCation:    DS.dynamic(light: 0xC2257F, dark: 0xFF79C6)   // magenta
        case .halogen:     DS.dynamic(light: 0x1E8F4A, dark: 0x4BE38A)   // green
        case .metalCoord:  DS.dynamic(light: 0xA67C00, dark: 0xFFD43B)   // gold
        case .chPi:        DS.dynamic(light: 0x7563A8, dark: 0xB4A2E0)   // light purple
        case .amideStack:  DS.dynamic(light: 0xA8651C, dark: 0xF0AE6A)   // amber
        case .chalcogen:   DS.dynamic(light: 0x6B7A00, dark: 0xD2E65A)   // yellow-green
        }
    }

    /// Color of this interaction's line in the 3D viewport (renderer data palette).
    /// Canvas legends use it so their swatches match the lines drawn in 3D.
    var viewportColor: Color {
        let c = color
        return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }

    /// Line pattern of this interaction in the 2D diagram and its legends. Every
    /// type gets a distinct pattern so the figure stays readable in grayscale:
    /// polar contacts are dashed, π contacts dotted, metal coordination solid.
    var diagramStroke: StrokeStyle {
        let dots: [CGFloat] = [0.01, 4.2]
        switch self {
        case .hbond:       return StrokeStyle(lineWidth: 1.8, lineCap: .butt, dash: [5, 3])
        case .saltBridge:  return StrokeStyle(lineWidth: 2.0, lineCap: .butt, dash: [8, 3, 2, 3])
        case .halogen:     return StrokeStyle(lineWidth: 1.8, lineCap: .butt, dash: [9, 4])
        case .chalcogen:   return StrokeStyle(lineWidth: 1.8, lineCap: .butt, dash: [8, 3, 2, 3, 2, 3])
        case .metalCoord:  return StrokeStyle(lineWidth: 2.2, lineCap: .round)
        case .piStack, .piCation, .chPi, .amideStack:
            return StrokeStyle(lineWidth: 2.6, lineCap: .round, dash: dots)
        case .hydrophobic: return StrokeStyle(lineWidth: 1.2, lineCap: .butt, dash: [3, 3])
        }
    }
}

// MARK: - Canvas legends

/// Floating canvas legend listing the interaction types present in the current pose.
/// Collapses to a pill so it never hides the structure for long.
struct InteractionLegendView: View {
    let interactions: [MolecularInteraction]
    @State private var isExpanded = true

    private var presentTypes: [MolecularInteraction.InteractionType] {
        let types = Set(interactions.map(\.type))
        return MolecularInteraction.InteractionType.allCases.filter { types.contains($0) }
    }

    var body: some View {
        if !presentTypes.isEmpty {
            if isExpanded {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    LegendHeader(title: "Interactions", isExpanded: $isExpanded)
                    Grid(alignment: .leading, horizontalSpacing: DS.Space.m, verticalSpacing: DS.Space.xs) {
                        ForEach(presentTypes, id: \.rawValue) { type in
                            GridRow {
                                Capsule()
                                    .fill(type.viewportColor)
                                    .frame(width: 16, height: 4)
                                Text(type.label)
                                    .font(DS.Font.caption)
                                    .foregroundStyle(.primary)
                                Text("\(interactions.filter { $0.type == type }.count)")
                                    .font(DS.Font.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(.secondary)
                                    .gridColumnAlignment(.trailing)
                            }
                        }
                    }
                }
                .padding(DS.Space.m)
                .frame(minWidth: 150, alignment: .leading)
                .dsFloating()
            } else {
                LegendPill(title: "Interactions", count: interactions.count, isExpanded: $isExpanded)
            }
        }
    }
}

struct SurfaceLegendView: View {
    let mode: SurfaceColorMode
    let legend: SurfaceLegend
    @State private var isExpanded = true

    var body: some View {
        Group {
            if isExpanded {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    LegendHeader(title: legend.title, isExpanded: $isExpanded)
                    content
                }
                .padding(DS.Space.m)
                .frame(minWidth: 150, alignment: .leading)
                .dsFloating()
            } else {
                LegendPill(title: legend.title, count: nil, isExpanded: $isExpanded)
            }
        }
        .accessibilityIdentifier(accessibilityID)
    }

    @ViewBuilder
    private var content: some View {
        switch legend.kind {
        case .gradient(let minLabel, let midLabel, let maxLabel, let colors):
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                LinearGradient(
                    colors: colors.map(Self.color(from:)),
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(width: 132, height: 8)
                .clipShape(Capsule())
                .overlay(Capsule().strokeBorder(DS.Color.hairline, lineWidth: 0.5))

                HStack(spacing: 0) {
                    Text(minLabel)
                    Spacer(minLength: DS.Space.xs)
                    if let midLabel {
                        Text(midLabel)
                        Spacer(minLength: DS.Space.xs)
                    }
                    Text(maxLabel)
                }
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 132)
            }
        case .categorical(let entries):
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                    HStack(spacing: DS.Space.m) {
                        RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                            .fill(Self.color(from: entry.color))
                            .frame(width: 14, height: 8)
                        Text(entry.label)
                            .font(DS.Font.caption)
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
    }

    private var accessibilityID: String {
        switch mode {
        case .uniform: return "surface_legend_uniform"
        case .esp: return "surface_legend_esp"
        case .hydrophobicity: return "surface_legend_hydrophobicity"
        case .pharmacophore: return "surface_legend_pharmacophore"
        }
    }

    /// Surface colors come from the renderer's data palette.
    private static func color(from simd: SIMD4<Float>) -> Color {
        Color(
            red: Double(simd.x),
            green: Double(simd.y),
            blue: Double(simd.z),
            opacity: Double(simd.w)
        )
    }
}

// MARK: - Legend chrome

private struct LegendHeader: View {
    let title: String
    @Binding var isExpanded: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: DS.Space.s) {
            Text(title)
                .font(DS.Font.eyebrow)
                .textCase(.uppercase)
                .tracking(0.6)
                .foregroundStyle(.secondary)
            Spacer(minLength: DS.Space.m)
            Button {
                withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) { isExpanded = false }
            } label: {
                Image(systemName: "chevron.up")
                    .font(DS.Font.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 16, height: 16)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Collapse legend")
            .accessibilityLabel("Collapse \(title) legend")
        }
    }
}

private struct LegendPill: View {
    let title: String
    let count: Int?
    @Binding var isExpanded: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Button {
            withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) { isExpanded = true }
        } label: {
            HStack(spacing: DS.Space.xs) {
                Image(systemName: "list.bullet.rectangle")
                Text(title)
                if let count {
                    Text("\(count)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            .font(DS.Font.caption.weight(.medium))
            .padding(.horizontal, DS.Space.m)
            .padding(.vertical, DS.Space.xs)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .dsFloatingCapsule()
        .help("Show \(title) legend")
    }
}

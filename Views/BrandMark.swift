// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import SwiftUI

enum DruseMarkStyle: Sendable { case color, monochrome }

/// Square app mark ("crystal pocket"): teal crystals lining a pocket with an amber aromatic ligand docked inside.
/// Same geometry as the app icon (`Tools/render_icon.swift`) without the rounded body; `size` is the side length in points.
struct DruseMark: View {
    var size: CGFloat = 64
    var style: DruseMarkStyle = .color

    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        let compact = size < 24
        let crystals = DruseMarkGeometry.crystals(compact: compact)
        ZStack {
            ForEach(crystals.indices, id: \.self) { index in
                crystal(crystals[index], compact: compact)
            }
            ligand(compact: compact)
        }
        .frame(width: size, height: size)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Druse")
    }

    private func crystal(_ crystal: DruseMarkGeometry.Crystal, compact: Bool) -> some View {
        ZStack {
            if compact {
                MarkPolygon(points: crystal.outline).fill(litFill)
            } else {
                MarkPolygon(points: crystal.leftFacet).fill(crystal.leftFacetIsLit ? litFill : shadedFill)
                MarkPolygon(points: crystal.rightFacet).fill(crystal.leftFacetIsLit ? shadedFill : litFill)
            }
        }
    }

    private func ligand(compact: Bool) -> some View {
        let unit = size / DruseMarkGeometry.extent
        let hexagon = MarkPolygon(points: DruseMarkGeometry.hexagon(scale: compact ? 1.25 : 1))
        let ringDiameter = 2 * DruseMarkGeometry.ringRadius * unit
        let ringWidth = max(DruseMarkGeometry.ringWidth * unit, 1)
        return ZStack {
            switch style {
            case .color:
                hexagon.fill(DS.Color.ligand)
                if !compact {
                    Circle()
                        .stroke(BrandPalette.ring.opacity(0.85), lineWidth: ringWidth)
                        .frame(width: ringDiameter, height: ringDiameter)
                }
            case .monochrome:
                hexagon.fill(.secondary)
                if !compact {
                    Circle()
                        .stroke(lineWidth: ringWidth)
                        .frame(width: ringDiameter, height: ringDiameter)
                        .blendMode(.destinationOut)
                }
            }
        }
        .compositingGroup()
    }

    private var litFill: AnyShapeStyle {
        switch style {
        case .color: AnyShapeStyle(colorScheme == .dark ? BrandPalette.crystalLitDark : BrandPalette.crystalLitLight)
        case .monochrome: AnyShapeStyle(.primary)
        }
    }

    private var shadedFill: AnyShapeStyle {
        switch style {
        case .color: AnyShapeStyle(colorScheme == .dark ? BrandPalette.crystalShadedDark : BrandPalette.crystalShadedLight)
        case .monochrome: AnyShapeStyle(.primary.opacity(0.55))
        }
    }
}

/// Mark + "Druse" lockup. The mark is centred on the cap height of the title text.
struct DruseWordmark: View {
    /// Sized so the visible crystal ring (720 of the mark's 824 units) is 1.25× the title's cap height.
    var markSize: CGFloat = (DruseWordmark.titleCapHeight * 1.25 * 824 / 720).rounded()

    private static let titleCapHeight = NSFont.preferredFont(forTextStyle: .title1).capHeight

    var body: some View {
        let capHeight = Self.titleCapHeight
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
            DruseMark(size: markSize)
                .alignmentGuide(.firstTextBaseline) { d in d.height / 2 + capHeight / 2 }
            Text("Druse")
                .font(.system(.title, design: .default).weight(.semibold))
                .tracking(-0.4)
                .foregroundStyle(.primary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Druse")
    }
}

// MARK: - Geometry (shared with Tools/render_icon.swift)

/// Brand-art colours (exempt from UI tokens, like the CPK palette). The ligand uses `DS.Color.ligand`.
private enum BrandPalette {
    static let crystalLitDark = rgb(0x5FD3DF)
    static let crystalShadedDark = rgb(0x2A9BB0)
    static let crystalLitLight = rgb(0x0A7C8C)
    static let crystalShadedLight = rgb(0x0B5563)
    static let ring = rgb(0xFFF1D6)

    private static func rgb(_ hex: UInt32) -> Color {
        Color(.sRGB,
              red: Double((hex >> 16) & 0xFF) / 255,
              green: Double((hex >> 8) & 0xFF) / 255,
              blue: Double(hex & 0xFF) / 255)
    }
}

/// Icon geometry normalised to the 824-unit body, centred on the pocket. Angles run clockwise from 12 o'clock.
private enum DruseMarkGeometry {
    static let extent: CGFloat = 824
    static let center = CGPoint(x: 412, y: 412)
    static let ringRadius: CGFloat = 44
    static let ringWidth: CGFloat = 10

    /// Tip radius per primary slot; 45° is the pocket mouth.
    private static let tipRadius: [(slot: CGFloat, radius: CGFloat)] = [
        (270, 158), (315, 172), (0, 160), (90, 176), (135, 164), (180, 150), (225, 170),
    ]
    private static let compactSlots: Set<CGFloat> = [0, 90, 135, 180, 270]
    /// Light falls from the upper left.
    private static let toLight = CGPoint(x: -0.6, y: -0.8)

    struct Crystal {
        let outline: [CGPoint]
        let leftFacet: [CGPoint]
        let rightFacet: [CGPoint]
        let leftFacetIsLit: Bool
    }

    /// The seven pocket-lining crystals, or the five-crystal simplification for small sizes.
    static func crystals(compact: Bool) -> [Crystal] {
        tipRadius
            .filter { !compact || compactSlots.contains($0.slot) }
            .map { crystal(theta: $0.slot, tipRadius: $0.radius) }
    }

    static func hexagon(scale: CGFloat) -> [CGPoint] {
        stride(from: 30, to: 360, by: 60).map { polar(CGFloat($0), 86 * scale) }
    }

    /// Kite: tip towards the pocket, back 184 units further out, shoulders at ±19° and 34 units in from the back.
    private static func crystal(theta: CGFloat, tipRadius tipR: CGFloat) -> Crystal {
        let backR = tipR + 184
        let tip = polar(theta, tipR)
        let back = polar(theta, backR)
        let left = polar(theta - 19, backR - 34)
        let right = polar(theta + 19, backR - 34)
        let side = direction(theta - 90)
        return Crystal(outline: [back, left, tip, right],
                       leftFacet: [back, left, tip],
                       rightFacet: [back, tip, right],
                       leftFacetIsLit: side.x * toLight.x + side.y * toLight.y > 0)
    }

    private static func direction(_ degrees: CGFloat) -> CGPoint {
        let radians = degrees * .pi / 180
        return CGPoint(x: sin(radians), y: -cos(radians))
    }

    private static func polar(_ degrees: CGFloat, _ radius: CGFloat) -> CGPoint {
        let d = direction(degrees)
        return CGPoint(x: center.x + d.x * radius, y: center.y + d.y * radius)
    }
}

/// Closed polygon given in 824-unit mark space, scaled to the frame it is drawn in.
private struct MarkPolygon: Shape {
    let points: [CGPoint]

    func path(in rect: CGRect) -> Path {
        let unit = min(rect.width, rect.height) / DruseMarkGeometry.extent
        var path = Path()
        path.addLines(points.map { CGPoint(x: rect.minX + $0.x * unit, y: rect.minY + $0.y * unit) })
        path.closeSubpath()
        return path
    }
}

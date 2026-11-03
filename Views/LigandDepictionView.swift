// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

// MARK: - LigandDepictionView
//
// Self-contained 2D structure depiction for a SMILES string. Primary path is
// RDKit's SVG drawer rasterized once into a bitmap (cached by SMILES); if the
// SVG pipeline fails, falls back to drawing RDKit 2D coordinates on a Canvas.
//
// The SVG canvas follows the view's aspect ratio (quantized), so long
// molecules fill a wide box instead of being letterboxed inside a fixed 5:4
// drawing at half the available width.
//
// RDKit draws black ink on white paper. In Light mode the bitmap is multiplied
// onto `diagramPaper`; in Dark mode it is inverted (hue-rotated so heteroatom
// colours keep their identity) and screened onto the dark paper.

@MainActor
private enum LigandDepictionCache {
    static var images: [String: NSImage] = [:]
    static let limit = 128

    static func store(_ image: NSImage, for smiles: String) {
        if images.count >= limit { images.removeAll(keepingCapacity: true) }
        images[smiles] = image
    }
}

struct LigandDepictionView: View {
    let smiles: String

    @Environment(\.colorScheme) private var colorScheme
    @State private var image: NSImage?
    @State private var coords: RDKitBridge.Coords2D?
    @State private var isLoading = false

    /// What a render depends on: the molecule and the drawing size in points.
    private struct RenderKey: Hashable {
        let smiles: String
        let size: CGSize

        func hash(into hasher: inout Hasher) {
            hasher.combine(smiles)
            hasher.combine(size.width)
            hasher.combine(size.height)
        }

        var cacheKey: String { "\(smiles)|\(Int(size.width))x\(Int(size.height))" }
    }

    /// Snap the available drawing area to a 20 pt grid so live resizing does
    /// not re-render on every pixel, with a floor that keeps RDKit's labels legible.
    private static func drawingSize(for size: CGSize) -> CGSize {
        let inset = DS.Space.xs * 2
        func snap(_ v: CGFloat) -> CGFloat { max(80, ((v - inset) / 20).rounded() * 20) }
        return CGSize(width: snap(size.width), height: snap(size.height))
    }

    var body: some View {
        GeometryReader { geo in
            content
                .task(id: RenderKey(smiles: smiles, size: Self.drawingSize(for: geo.size))) {
                    await load(RenderKey(smiles: smiles, size: Self.drawingSize(for: geo.size)))
                }
        }
    }

    private var content: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        return ZStack {
            if let image {
                depiction(image)
            } else if let coords {
                Canvas { context, size in
                    Self.draw(context: context, size: size, coords: coords)
                }
            } else if isLoading {
                ProgressView().controlSize(.small)
            } else {
                VStack(spacing: DS.Space.xs) {
                    Image(systemName: "hexagon")
                        .font(.title2)
                        .symbolRenderingMode(.hierarchical)
                    Text("No 2D structure")
                        .font(DS.Font.caption)
                }
                .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(DS.Color.diagramPaper)
        .compositingGroup()
        .clipShape(shape)
        .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("2D structure")
    }

    @ViewBuilder
    private func depiction(_ image: NSImage) -> some View {
        let base = Image(nsImage: image)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .padding(DS.Space.xs)
        if colorScheme == .dark {
            base
                .colorInvert()
                .hueRotation(.degrees(180))
                .blendMode(.screen)
        } else {
            base.blendMode(.multiply)
        }
    }

    // MARK: Loading

    private func load(_ key: RenderKey) async {
        let smi = key.smiles
        guard !smi.isEmpty else {
            image = nil
            coords = nil
            isLoading = false
            return
        }
        if let cached = LigandDepictionCache.images[key.cacheKey] {
            image = cached
            coords = nil
            isLoading = false
            return
        }

        isLoading = true
        // Debounce rapid selection changes (task(id:) cancels the previous run).
        try? await Task.sleep(for: .milliseconds(80))
        guard !Task.isCancelled else { return }

        // RDKit line widths and fonts are in canvas pixels: a canvas 1.5× the
        // point size gives ~1.3 pt bonds once fitted, a typical 2D-sketch weight.
        let canvas = CGSize(width: key.size.width * 1.5, height: key.size.height * 1.5)
        let svg = await RDKitBridge.moleculeToSVG(smiles: smi, width: Int(canvas.width), height: Int(canvas.height))
        guard !Task.isCancelled else { return }

        if let svg, let raster = Self.rasterize(svg: svg, canvas: canvas, pointSize: key.size) {
            LigandDepictionCache.store(raster, for: key.cacheKey)
            image = raster
            coords = nil
            isLoading = false
            return
        }

        let fallback = await Task.detached { @Sendable in
            RDKitBridge.compute2DCoords(smiles: smi)
        }.value
        guard !Task.isCancelled else { return }
        image = nil
        coords = fallback
        isLoading = false
    }

    /// Rasterize once to a 2× bitmap of `pointSize`. Without this, SwiftUI
    /// re-parses the SVG and re-rasterizes CoreSVG on every redraw.
    private static func rasterize(svg: String, canvas: CGSize, pointSize: CGSize) -> NSImage? {
        guard let data = svg.data(using: .utf8), let raw = NSImage(data: data) else { return nil }
        raw.size = canvas
        let size = NSSize(width: pointSize.width, height: pointSize.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(pointSize.width * 2), pixelsHigh: Int(pointSize.height * 2),
            bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0
        ) else { return raw }
        rep.size = size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        raw.draw(in: NSRect(origin: .zero, size: size))
        NSGraphicsContext.restoreGraphicsState()
        let bitmap = NSImage(size: size)
        bitmap.addRepresentation(rep)
        return bitmap
    }

    // MARK: Canvas fallback

    private static func draw(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let positions = coords.positions
        guard !positions.isEmpty else { return }

        let xs = positions.map(\.x), ys = positions.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return }
        let rangeX = max(maxX - minX, 1), rangeY = max(maxY - minY, 1)
        let padding: CGFloat = 24
        let scale = min((size.width - padding * 2) / rangeX, (size.height - padding * 2) / rangeY) * 0.85
        let cx = size.width / 2, cy = size.height / 2
        let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale, y: cy - (p.y - midY) * scale)
        }

        let ink = DS.Color.diagramInk
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1]), p2 = project(positions[a2])
            if order == 2 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = sqrt(dx * dx + dy * dy)
                guard len > 0 else { continue }
                let nx = -dy / len * 2, ny = dx / len * 2
                for sign in [-1.0, 1.0] {
                    var path = Path()
                    path.move(to: CGPoint(x: p1.x + nx * sign, y: p1.y + ny * sign))
                    path.addLine(to: CGPoint(x: p2.x + nx * sign, y: p2.y + ny * sign))
                    context.stroke(path, with: .color(ink.opacity(0.8)), lineWidth: 1.5)
                }
            } else {
                var path = Path()
                path.move(to: p1)
                path.addLine(to: p2)
                context.stroke(path, with: .color(ink.opacity(0.8)), lineWidth: 1.8)
            }
        }

        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            let z = coords.atomicNums[i]
            guard z != 6 else { continue }
            let r: CGFloat = 9
            let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
            let (symbol, color) = atomLabel(z)
            context.fill(Path(ellipseIn: rect), with: .color(DS.Color.diagramPaper))
            let text = Text(symbol).font(DS.Font.monoSmall.weight(.bold)).foregroundColor(color)
            context.draw(context.resolve(text), at: p, anchor: .center)
        }
    }

    /// CPK-style heteroatom label colours (data palette).
    private static func atomLabel(_ atomicNum: Int) -> (String, Color) {
        switch atomicNum {
        case 7:  ("N", DS.Color.protein)
        case 8:  ("O", DS.Color.danger)
        case 9, 17: (atomicNum == 9 ? "F" : "Cl", DS.Color.success)
        case 15: ("P", DS.Color.scorePoor)
        case 16: ("S", DS.Color.warning)
        case 35: ("Br", DS.Color.metal)
        case 53: ("I", DS.Color.pocket)
        default: (Element(rawValue: atomicNum)?.symbol ?? "?", DS.Color.diagramInk)
        }
    }
}

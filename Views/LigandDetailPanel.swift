import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Detail Panel Extension
//
// Bottom detail area: top header bar with name/badges/SMILES, then a row with
// the 2D RDKit depiction on the left and the LigandMini3DView (self-contained
// 3D viewport with conformer navigation) on the right. Plus the 2D preview
// computation and rasterization helpers.

extension LigandDatabaseWindow {

    // MARK: - Bottom Detail Area (2D depiction | 3D viewport, side by side)

    @ViewBuilder
    func bottomDetailArea(_ entry: LigandEntry) -> some View {
        VStack(spacing: 0) {
            detailHeaderBar(entry)
            Divider()

            HStack(spacing: 0) {
                structure2DPanel(entry)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                Divider()

                LigandMini3DView(
                    atoms: entry.atoms,
                    bonds: entry.bonds,
                    conformers: conformers,
                    selectedConformerIndex: $selectedConformerIndex
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - Header Bar

    @ViewBuilder
    private func detailHeaderBar(_ entry: LigandEntry) -> some View {
        HStack(spacing: 8) {
            Text(entry.name)
                .font(.headline)
            if entry.isPrepared {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.subheadline)
            }

            // Enumeration form badge (when this entry came from enumeration)
            if entry.isEnumerated, let kind = entry.formKind {
                let kindColor: Color = switch kind {
                case .parent: .green
                case .tautomer: .cyan
                case .protomer: .orange
                case .tautomerProtomer: .purple
                }
                Text(kind.symbol)
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 16, height: 14)
                    .background(RoundedRectangle(cornerRadius: 4).fill(kindColor))
                Text(kind.label)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                if let pn = entry.parentName {
                    Text("from \(pn)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if let pop = entry.populationWeight {
                    Text(String(format: "%.1f%%", pop * 100))
                        .font(.footnote.monospaced())
                        .foregroundStyle(pop > 0.3 ? .green : pop > 0.1 ? .yellow : .secondary)
                }
            }

            Spacer()

            Text(entry.smiles.prefix(40) + (entry.smiles.count > 40 ? "..." : ""))
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .help(entry.smiles)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - 2D Structure Panel

    @ViewBuilder
    private func structure2DPanel(_ entry: LigandEntry) -> some View {
        VStack(spacing: 0) {
            if isComputing2D {
                ProgressView("Computing 2D layout...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let nsImage = ligand2DImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else if let coords = ligand2DCoords {
                Canvas { context, size in
                    draw2DStructure(context: context, size: size, coords: coords)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor))
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "hexagon")
                        .font(.title)
                        .foregroundStyle(.secondary)
                    Text("No 2D structure")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - 2D Structure Drawing

    /// Debounced 2D preview — cancels previous computation when a new row is selected.
    private static var preview2DTask: Task<Void, Never>?

    /// LRU-ish cache of pre-rasterized 2D previews keyed by SMILES. Avoids re-running
    /// the RDKit layout + SVG rasterization pipeline when revisiting a ligand.
    private static var previewImageCache: [String: NSImage] = [:]
    private static let previewImageCacheLimit = 128

    func compute2DPreview(smiles: String) {
        guard !smiles.isEmpty else { ligand2DCoords = nil; ligand2DImage = nil; return }

        // Cache hit — display instantly, no RDKit work
        if let cached = Self.previewImageCache[smiles] {
            Self.preview2DTask?.cancel()
            ligand2DImage = cached
            ligand2DCoords = nil
            isComputing2D = false
            return
        }

        // Cancel any in-flight computation
        Self.preview2DTask?.cancel()
        isComputing2D = true

        Self.preview2DTask = Task {
            // Small delay to debounce rapid clicks
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled else { return }

            let smi = smiles
            // Single RDKit call: SVG is the primary path. Coords are only needed
            // as a fallback if SVG generation fails, so skip them on the happy path.
            let svg: String? = await Task.detached { @Sendable in
                RDKitBridge.moleculeToSVG(smiles: smi, width: 500, height: 400)
            }.value
            guard !Task.isCancelled else { return }

            // Rasterize once to a bitmap NSImage. Without this, SwiftUI re-parses
            // the SVG string and re-rasterizes CoreSVG on every view redraw, which
            // is the dominant cost for this preview.
            var image: NSImage? = nil
            if let svg, let data = svg.data(using: .utf8), let raw = NSImage(data: data) {
                let size = NSSize(width: 500, height: 400)
                raw.size = size
                if let rep = NSBitmapImageRep(
                    bitmapDataPlanes: nil,
                    pixelsWide: 1000, pixelsHigh: 800,
                    bitsPerSample: 8, samplesPerPixel: 4,
                    hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB,
                    bytesPerRow: 0, bitsPerPixel: 0
                ) {
                    rep.size = size
                    NSGraphicsContext.saveGraphicsState()
                    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
                    raw.draw(in: NSRect(origin: .zero, size: size))
                    NSGraphicsContext.restoreGraphicsState()
                    let bitmap = NSImage(size: size)
                    bitmap.addRepresentation(rep)
                    image = bitmap
                } else {
                    image = raw
                }
            }
            guard !Task.isCancelled else { return }

            // Fallback to Canvas-drawn coords only if SVG pipeline failed
            var fallbackCoords: RDKitBridge.Coords2D? = nil
            if image == nil {
                fallbackCoords = await Task.detached { @Sendable in
                    RDKitBridge.compute2DCoords(smiles: smi)
                }.value
                guard !Task.isCancelled else { return }
            }

            if let image {
                if Self.previewImageCache.count >= Self.previewImageCacheLimit {
                    Self.previewImageCache.removeAll(keepingCapacity: true)
                }
                Self.previewImageCache[smi] = image
            }
            ligand2DImage = image
            ligand2DCoords = fallbackCoords
            isComputing2D = false
        }
    }

    func draw2DStructure(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let positions = coords.positions
        guard !positions.isEmpty else { return }

        let xs = positions.map(\.x), ys = positions.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 1), rangeY = max(maxY - minY, 1)
        let padding: CGFloat = 30
        let scale = min((size.width - padding * 2) / rangeX, (size.height - padding * 2) / rangeY) * 0.85
        let cx = size.width / 2, cy = size.height / 2
        let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale, y: cy - (p.y - midY) * scale)
        }

        // Bonds
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1]), p2 = project(positions[a2])
            let bondColor = Color.primary.opacity(0.7)
            if order == 2 {
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = sqrt(dx*dx + dy*dy)
                guard len > 0 else { continue }
                let nx = -dy/len * 2, ny = dx/len * 2
                for sign in [-1.0, 1.0] {
                    var path = Path()
                    path.move(to: CGPoint(x: p1.x + nx*sign, y: p1.y + ny*sign))
                    path.addLine(to: CGPoint(x: p2.x + nx*sign, y: p2.y + ny*sign))
                    context.stroke(path, with: .color(bondColor), lineWidth: 1.5)
                }
            } else {
                var path = Path()
                path.move(to: p1)
                path.addLine(to: p2)
                context.stroke(path, with: .color(bondColor), lineWidth: order == 4 ? 2 : 1.8)
                if order == 4 {
                    let dx = p2.x - p1.x, dy = p2.y - p1.y
                    let len = sqrt(dx*dx + dy*dy)
                    if len > 0 {
                        let nx = -dy/len * 3, ny = dx/len * 3
                        var dashed = Path()
                        dashed.move(to: CGPoint(x: p1.x + nx, y: p1.y + ny))
                        dashed.addLine(to: CGPoint(x: p2.x + nx, y: p2.y + ny))
                        context.stroke(dashed, with: .color(bondColor.opacity(0.4)),
                                       style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                    }
                }
            }
        }

        // Atoms
        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            let z = coords.atomicNums[i]
            if z == 6 {
                let r: CGFloat = 3
                context.fill(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                             with: .color(.primary.opacity(0.5)))
            } else {
                let r: CGFloat = 10
                let (sym, col) = atomSymbolColor(z)
                context.fill(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                             with: .color(Color(nsColor: .textBackgroundColor)))
                context.stroke(Path(ellipseIn: CGRect(x: p.x-r, y: p.y-r, width: r*2, height: r*2)),
                               with: .color(col), lineWidth: 1.5)
                let text = Text(sym).font(.subheadline.monospaced().weight(.bold)).foregroundColor(col)
                context.draw(context.resolve(text), at: p, anchor: .center)
            }
        }
    }

    func atomSymbolColor(_ atomicNum: Int) -> (String, Color) {
        switch atomicNum {
        case 7:  return ("N", .blue)
        case 8:  return ("O", .red)
        case 9:  return ("F", .green)
        case 15: return ("P", .orange)
        case 16: return ("S", .yellow)
        case 17: return ("Cl", .green)
        case 35: return ("Br", Color(red: 0.6, green: 0.2, blue: 0.2))
        case 53: return ("I", .purple)
        default: return (Element(rawValue: atomicNum)?.symbol ?? "?", .gray)
        }
    }


}

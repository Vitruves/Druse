import SwiftUI
import UniformTypeIdentifiers

/// 2D Protein-Ligand Interaction Diagram (ProLig2D) — MOE-style.
/// Renders a flat schematic with the ligand centered and interacting residues
/// arranged radially. Supports zoom/pan for exploration, and export to PNG.
/// Residues colored by chemical property (acidic/basic/polar/greasy).
struct InteractionDiagramView: View {
    let interactions: [MolecularInteraction]
    let ligandAtoms: [Atom]
    let ligandBonds: [Bond]
    let proteinAtoms: [Atom]
    let ligandSmiles: String?
    let poseEnergy: Float
    let poseIndex: Int

    @Environment(\.dismiss) private var dismiss
    @State private var coords2D: RDKitBridge.Coords2D?
    @State private var isLoading = true

    // Zoom & pan state
    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastPanOffset: CGSize = .zero
    @State private var lastZoomScale: CGFloat = 1.0

    // Export state
    @State private var exportSize: CGSize = .zero

    // Hover state
    @State private var hoveredResidue: String? = nil
    @State private var hoverPoint: CGPoint = .zero
    @State private var residuePositions: [(name: String, center: CGPoint, interactions: [MolecularInteraction])] = []

    // Aromatic rings detected from bond connectivity
    @State private var aromaticRings: [[Int]] = []

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if isLoading {
                ProgressView("Generating 2D layout...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let coords = coords2D {
                diagramCanvas(coords: coords)
            } else {
                fallbackDiagram
            }
            Divider()
            legend
        }
        .frame(minWidth: 1000, minHeight: 750)
        .task {
            if let smiles = ligandSmiles, !smiles.isEmpty {
                let result = await Task.detached {
                    RDKitBridge.compute2DCoords(smiles: smiles)
                }.value
                coords2D = result
                if let c = result {
                    aromaticRings = detectAromaticRings(bonds: c.bonds, atomCount: c.positions.count)
                }
            }
            isLoading = false
        }
    }

    // MARK: - Zoomable Diagram Canvas

    @ViewBuilder
    private func diagramCanvas(coords: RDKitBridge.Coords2D) -> some View {
        GeometryReader { geo in
            Canvas { context, size in
                // Apply zoom + pan transform
                var ctx = context
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                ctx.translateBy(x: center.x + panOffset.width, y: center.y + panOffset.height)
                ctx.scaleBy(x: zoomScale, y: zoomScale)
                ctx.translateBy(x: -center.x, y: -center.y)
                drawDiagram(context: ctx, size: size, coords: coords)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .controlBackgroundColor))
            .gesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoomScale = max(0.3, min(5.0, lastZoomScale * value.magnification))
                    }
                    .onEnded { value in
                        lastZoomScale = zoomScale
                    }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { value in
                        panOffset = CGSize(
                            width: lastPanOffset.width + value.translation.width,
                            height: lastPanOffset.height + value.translation.height
                        )
                    }
                    .onEnded { value in
                        lastPanOffset = panOffset
                    }
            )
            .onTapGesture(count: 2) {
                // Double-tap to reset zoom
                withAnimation(.easeInOut(duration: 0.3)) {
                    zoomScale = 1.0
                    panOffset = .zero
                    lastZoomScale = 1.0
                    lastPanOffset = .zero
                }
            }
            .overlay {
                Color.clear
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            hoverPoint = location
                            hoveredResidue = findResidueAtPoint(location, in: geo.size)
                        case .ended:
                            hoveredResidue = nil
                        }
                    }
            }
            .overlay(alignment: .topLeading) {
                if let res = hoveredResidue {
                    let resInteractions = residueInteractions(for: res)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(res).font(.subheadline.weight(.bold))
                        ForEach(resInteractions, id: \.id) { inter in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(interactionColor(inter.type))
                                    .frame(width: 6, height: 6)
                                Text(inter.type.label)
                                    .font(.footnote)
                                Text(String(format: "%.1f \u{00C5}", inter.distance))
                                    .font(.footnote.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(.ultraThickMaterial))
                    .shadow(radius: 4)
                    .position(x: hoverPoint.x + 60, y: hoverPoint.y - 20)
                    .allowsHitTesting(false)
                }
            }
            .overlay(alignment: .topTrailing) {
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.escape, modifiers: [])
                .padding(12)
                .help("Close (Esc)")
            }
            .onChange(of: geo.size) { _, newSize in
                exportSize = newSize
            }
            .onAppear {
                exportSize = geo.size
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Label("Interaction Diagram", systemImage: "circle.hexagongrid")
                .font(.body.weight(.semibold))
            Spacer()
            Text("Pose #\(poseIndex + 1)")
                .font(.callout.monospaced().weight(.medium))
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f kcal/mol", poseEnergy))
                .font(.callout.monospaced().weight(.semibold))
                .foregroundStyle(poseEnergy < -6 ? .green : poseEnergy < 0 ? .orange : .red)

            // Zoom controls
            HStack(spacing: 4) {
                Button(action: { withAnimation { zoomScale = max(0.3, zoomScale - 0.2); lastZoomScale = zoomScale } }) {
                    Image(systemName: "minus.magnifyingglass").font(.callout)
                }
                .buttonStyle(.plain)
                .help("Zoom out")
                Text(String(format: "%.0f%%", zoomScale * 100))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(width: 36)
                Button(action: { withAnimation { zoomScale = min(5.0, zoomScale + 0.2); lastZoomScale = zoomScale } }) {
                    Image(systemName: "plus.magnifyingglass").font(.callout)
                }
                .buttonStyle(.plain)
                .help("Zoom in")
                Button(action: {
                    withAnimation { zoomScale = 1.0; panOffset = .zero; lastZoomScale = 1.0; lastPanOffset = .zero }
                }) {
                    Image(systemName: "arrow.counterclockwise").font(.subheadline)
                }
                .buttonStyle(.plain)
                .help("Reset zoom (or double-click)")
            }
            .padding(.horizontal, 8)

            Button(action: { exportPNG() }) {
                Image(systemName: "square.and.arrow.down")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .help("Save as PNG")

            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Close diagram")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Legend

    private var legend: some View {
        let presentTypes = Set(interactions.map(\.type))
        let types = MolecularInteraction.InteractionType.allCases.filter { presentTypes.contains($0) }

        return VStack(spacing: 8) {
            // Top row: Interaction type legend (wraps if needed)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(types, id: \.rawValue) { type in
                        let count = interactions.filter { $0.type == type }.count
                        HStack(spacing: 4) {
                            interactionSymbol(type)
                                .frame(width: 24, height: 14)
                            Text("\(type.label) (\(count))")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .fixedSize()
                        }
                    }
                }
            }

            // Bottom row: Residue types + total count
            HStack(spacing: 0) {
                HStack(spacing: 12) {
                    residueLegendDot("polar", Color(red: 0.2, green: 0.6, blue: 0.7))
                    residueLegendDot("acidic", Color(red: 0.8, green: 0.2, blue: 0.2))
                    residueLegendDot("basic", Color(red: 0.3, green: 0.3, blue: 0.9))
                    residueLegendDot("greasy", Color(red: 0.4, green: 0.6, blue: 0.3))
                }

                Spacer()

                Text("\(interactions.count) interactions")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func residueLegendDot(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 4)
                .fill(color.opacity(0.3))
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(color, lineWidth: 1))
                .frame(width: 14, height: 14)
            Text(label)
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func interactionSymbol(_ type: MolecularInteraction.InteractionType) -> some View {
        let c = interactionColor(type)
        Canvas { ctx, size in
            let y = size.height / 2
            switch type {
            case .hbond:
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y), to: CGPoint(x: size.width, y: y),
                               color: c, dashLen: 3, lineWidth: 2)
            case .hydrophobic:
                var path = Path()
                path.move(to: CGPoint(x: 2, y: size.height))
                path.addQuadCurve(to: CGPoint(x: size.width - 2, y: size.height),
                                  control: CGPoint(x: size.width / 2, y: -2))
                ctx.stroke(path, with: .color(c), lineWidth: 2)
            case .saltBridge:
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y - 2), to: CGPoint(x: size.width, y: y - 2),
                               color: c, dashLen: 2, lineWidth: 1.5)
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y + 2), to: CGPoint(x: size.width, y: y + 2),
                               color: c, dashLen: 2, lineWidth: 1.5)
            default:
                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(path, with: .color(c), lineWidth: 2)
            }
        }
    }

    // MARK: - Fallback

    private var fallbackDiagram: some View {
        VStack(spacing: 8) {
            Image(systemName: "atom")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Could not generate 2D layout")
                .font(.body)
                .foregroundStyle(.secondary)
            Text("SMILES required for 2D depiction")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Drawing (scaled up ~50% for legibility)

    private func drawDiagram(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let padding: CGFloat = 160  // generous space for residue bubbles

        let positions = coords.positions
        guard !positions.isEmpty else { return }

        let xs = positions.map(\.x)
        let ys = positions.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 1)
        let rangeY = max(maxY - minY, 1)

        let ligandAreaW = size.width - padding * 2
        let ligandAreaH = size.height - padding * 2
        let scale = min(ligandAreaW / rangeX, ligandAreaH / rangeY) * 0.75
        let cx = size.width / 2
        let cy = size.height / 2
        let midX = (minX + maxX) / 2
        let midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale,
                    y: cy - (p.y - midY) * scale)
        }

        // 1. Draw ligand bonds (thicker)
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1])
            let p2 = project(positions[a2])
            drawBond(context: context, from: p1, to: p2, order: order)
        }

        // 2. Draw aromatic ring inscribed circles
        for ring in aromaticRings {
            guard ring.count >= 5 else { continue }
            let ringPts = ring.compactMap { $0 < positions.count ? project(positions[$0]) : nil }
            guard ringPts.count == ring.count else { continue }
            let centX = ringPts.map(\.x).reduce(0, +) / CGFloat(ringPts.count)
            let centY = ringPts.map(\.y).reduce(0, +) / CGFloat(ringPts.count)
            let avgR = ringPts.map { hypot($0.x - centX, $0.y - centY) }.reduce(0, +) / CGFloat(ringPts.count)
            let r = avgR * 0.55
            context.stroke(Path(ellipseIn: CGRect(x: centX - r, y: centY - r, width: r * 2, height: r * 2)),
                           with: .color(Color.primary.opacity(0.3)), lineWidth: 1.5)
        }

        // 3. Draw ligand atoms (larger)
        var projectedPositions: [CGPoint] = []
        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            projectedPositions.append(p)
            let atomicNum = coords.atomicNums[i]
            drawLigandAtom(context: context, at: p, atomicNum: atomicNum, index: i)
        }

        // Compute ligand bounding box for line routing
        let projXs = projectedPositions.map(\.x)
        let projYs = projectedPositions.map(\.y)
        let ligandAABB = CGRect(
            x: (projXs.min() ?? cx) - 20, y: (projYs.min() ?? cy) - 20,
            width: ((projXs.max() ?? cx) - (projXs.min() ?? cx)) + 40,
            height: ((projYs.max() ?? cy) - (projYs.min() ?? cy)) + 40
        )

        // 4. Group interactions by residue
        let residueGroups = groupInteractionsByResidue()
        guard !residueGroups.isEmpty else { return }

        let ligandCenter = CGPoint(x: cx, y: cy)
        var residuePlacements: [(key: ResidueKey, center: CGPoint, interactions: [MolecularInteraction])] = []

        for (key, ixns) in residueGroups {
            var sumX: CGFloat = 0, sumY: CGFloat = 0
            var count = 0
            for ixn in ixns {
                let ligIdx = ixn.ligandAtomIndex
                if ligIdx < positions.count {
                    let p = project(positions[ligIdx])
                    sumX += p.x
                    sumY += p.y
                    count += 1
                }
            }
            guard count > 0 else { continue }
            let avgLigandPos = CGPoint(x: sumX / CGFloat(count), y: sumY / CGFloat(count))

            let dx = avgLigandPos.x - ligandCenter.x
            let dy = avgLigandPos.y - ligandCenter.y
            let dist = max(sqrt(dx * dx + dy * dy), 1)
            let pushDist: CGFloat = min(size.width, size.height) * 0.44
            let residueCenter = CGPoint(
                x: ligandCenter.x + dx / dist * pushDist,
                y: ligandCenter.y + dy / dist * pushDist
            )
            residuePlacements.append((key, residueCenter, ixns))
        }

        // Resolve overlaps with tighter angular constraints
        resolveOverlaps(&residuePlacements, around: ligandCenter,
                        radius: min(size.width, size.height) * 0.44)

        // 4. Store positions for hover detection (in pre-transform coordinates)
        DispatchQueue.main.async {
            self.residuePositions = residuePlacements.map { (key, center, ixns) in
                (name: key.label, center: center, interactions: ixns)
            }
        }

        // 5. Pre-compute side chain positions once per residue (for consistent line + drawing)
        struct ResidueDrawData {
            let key: ResidueKey
            let center: CGPoint
            let interactions: [MolecularInteraction]
            let avgLigandContact: CGPoint
            let scPositions: [CGPoint]  // transformed side chain atom positions
            let template: SideChainTemplate?
        }

        var drawData: [ResidueDrawData] = []
        for (key, residueCenter, ixns) in residuePlacements {
            // Average ligand contact position for this residue
            var sX: CGFloat = 0, sY: CGFloat = 0, n: CGFloat = 0
            for ixn in ixns {
                if ixn.ligandAtomIndex < positions.count {
                    let p = project(positions[ixn.ligandAtomIndex])
                    sX += p.x; sY += p.y; n += 1
                }
            }
            let avgContact = n > 0 ? CGPoint(x: sX / n, y: sY / n) : ligandCenter

            // Compute side chain once — pivot around first matching side chain atom
            var scPos: [CGPoint] = []
            let template = sideChainTemplates[key.name]
            if let tmpl = template, !tmpl.atoms.isEmpty {
                // Find best pivot: first interacting atom that exists in template
                var pivotName = tmpl.atoms.last!.name // default: outermost
                for ixn in ixns {
                    guard ixn.proteinAtomIndex < proteinAtoms.count else { continue }
                    let aName = proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
                    if tmpl.atoms.contains(where: { $0.name == aName }) {
                        pivotName = aName
                        break
                    }
                }
                scPos = transformSideChain(template: tmpl, bubbleCenter: residueCenter,
                                            interactingAtomName: pivotName, towardLigand: avgContact)
            }

            drawData.append(ResidueDrawData(key: key, center: residueCenter, interactions: ixns,
                                             avgLigandContact: avgContact, scPositions: scPos, template: template))
        }

        // 6. Draw interaction lines (behind side chains and bubbles)
        for dd in drawData {
            let hydrophobic = dd.interactions.filter { $0.type == .hydrophobic }
            let directional = dd.interactions.filter { $0.type != .hydrophobic }

            // Hydrophobic: green arc at contact zone + dashed connector
            if !hydrophobic.isEmpty {
                let arcColor = interactionColor(.hydrophobic)
                let avgLig = dd.avgLigandContact
                let dirAngle = atan2(dd.center.y - avgLig.y, dd.center.x - avgLig.x)
                let arcRadius: CGFloat = 18
                let arcSpan: CGFloat = max(0.8, min(CGFloat(hydrophobic.count) * 0.35, 1.6))

                var arcPath = Path()
                arcPath.addArc(center: avgLig, radius: arcRadius,
                               startAngle: .radians(dirAngle - arcSpan / 2),
                               endAngle: .radians(dirAngle + arcSpan / 2), clockwise: false)
                context.stroke(arcPath, with: .color(arcColor),
                               style: StrokeStyle(lineWidth: 3.5, lineCap: .round))

                // Spokes
                let nSpokes = min(hydrophobic.count, 5)
                for i in 0..<nSpokes {
                    let t = nSpokes == 1 ? 0.5 : CGFloat(i) / CGFloat(nSpokes - 1)
                    let a = dirAngle - arcSpan / 2 + arcSpan * t
                    let inner = CGPoint(x: avgLig.x + (arcRadius - 4) * cos(a), y: avgLig.y + (arcRadius - 4) * sin(a))
                    let outer = CGPoint(x: avgLig.x + (arcRadius + 4) * cos(a), y: avgLig.y + (arcRadius + 4) * sin(a))
                    var sp = Path(); sp.move(to: inner); sp.addLine(to: outer)
                    context.stroke(sp, with: .color(arcColor.opacity(0.6)), lineWidth: 1.5)
                }

                // Dashed connector to residue
                let arcTip = CGPoint(x: avgLig.x + (arcRadius + 6) * cos(dirAngle),
                                     y: avgLig.y + (arcRadius + 6) * sin(dirAngle))
                let conn = shorten(from: arcTip, to: dd.center, fromInset: 0, toInset: 40)
                drawDashedLine(ctx: context, from: conn.0, to: conn.1,
                               color: arcColor.opacity(0.5), dashLen: 4, lineWidth: 1.2)
            }

            // Directional interactions: connect to specific side chain atom,
            // route around ligand AABB if line would cross through it
            for ixn in directional {
                let ligIdx = ixn.ligandAtomIndex
                guard ligIdx < positions.count else { continue }
                let ligPos = project(positions[ligIdx])

                var targetPos = dd.center
                var toInset: CGFloat = 40
                if let tmpl = dd.template, !dd.scPositions.isEmpty,
                   ixn.proteinAtomIndex < proteinAtoms.count {
                    let atomName = proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
                    if let idx = tmpl.atoms.firstIndex(where: { $0.name == atomName }),
                       idx < dd.scPositions.count {
                        targetPos = dd.scPositions[idx]
                        toInset = 7
                    }
                }

                let fromInset: CGFloat = 8
                let shortened = shorten(from: ligPos, to: targetPos, fromInset: fromInset, toInset: toInset)

                // Check if the straight line crosses the ligand AABB
                // (only if the line endpoint is outside the AABB — i.e. it would cross through)
                if lineSegmentCrossesRect(from: shortened.0, to: shortened.1, rect: ligandAABB) &&
                   !ligandAABB.contains(shortened.1) {
                    // Route around: use a bezier curve through the nearest AABB corner
                    let corners = [
                        CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.minY - 12),
                        CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.minY - 12),
                        CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.maxY + 12),
                        CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.maxY + 12)
                    ]
                    let midPt = CGPoint(x: (shortened.0.x + shortened.1.x) / 2,
                                        y: (shortened.0.y + shortened.1.y) / 2)
                    let ctrl = corners.min(by: {
                        hypot($0.x - midPt.x, $0.y - midPt.y) < hypot($1.x - midPt.x, $1.y - midPt.y)
                    })!
                    let color = interactionColor(ixn.type)
                    var path = Path(); path.move(to: shortened.0)
                    path.addQuadCurve(to: shortened.1, control: ctrl)
                    context.stroke(path, with: .color(color),
                                   style: ixn.type == .hbond || ixn.type == .saltBridge
                                       ? StrokeStyle(lineWidth: 2, dash: [6, 4.2])
                                       : StrokeStyle(lineWidth: 2))
                } else {
                    drawInteractionLine(context: context, from: shortened.0, to: shortened.1,
                                        type: ixn.type, distance: ixn.distance)
                }
            }
        }

        // 7. Draw side chains + residue bubbles (on top of lines)
        for dd in drawData {
            drawResidueBubble(context: context, at: dd.center, key: dd.key, interactions: dd.interactions,
                              scPositions: dd.scPositions, template: dd.template)
        }
    }

    // MARK: - PNG Export

    private func exportPNG() {
        let scale: CGFloat = 2.0
        let width = max(exportSize.width, 900) * scale
        let height = max(exportSize.height, 700) * scale

        guard let coords = coords2D else { return }

        let renderer = ImageRenderer(content:
            Canvas { context, size in
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.white))
                self.drawDiagram(context: context, size: size, coords: coords)
            }
            .frame(width: width / scale, height: height / scale)
        )
        renderer.scale = scale

        guard let image = renderer.nsImage else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "interaction_diagram_pose\(poseIndex + 1).png"
        if panel.runModal() == .OK, let url = panel.url {
            if let tiff = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiff),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        }
    }

    // MARK: - Hover Detection

    private func findResidueAtPoint(_ point: CGPoint, in size: CGSize) -> String? {
        // Transform the point from view space back to diagram space
        // accounting for zoom and pan
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        // Reverse the transform: translate, scale, translate
        let px = (point.x - center.x - panOffset.width) / zoomScale + center.x
        let py = (point.y - center.y - panOffset.height) / zoomScale + center.y
        let transformed = CGPoint(x: px, y: py)

        let hitRadius: CGFloat = 42.0  // bubble is ~72 wide x 40 tall, generous hit area
        for entry in residuePositions {
            let dx = transformed.x - entry.center.x
            let dy = transformed.y - entry.center.y
            if dx * dx + dy * dy <= hitRadius * hitRadius {
                return entry.name
            }
        }
        return nil
    }

    private func residueInteractions(for residueLabel: String) -> [MolecularInteraction] {
        interactions.filter { ixn in
            let pIdx = ixn.proteinAtomIndex
            guard pIdx < proteinAtoms.count else { return false }
            let atom = proteinAtoms[pIdx]
            return "\(atom.residueName)\(atom.residueSeq)" == residueLabel
        }
    }

    // MARK: - Residue Grouping

    private struct ResidueKey: Hashable {
        let name: String
        let seq: Int
        let chain: String
        var label: String { "\(name)\(seq)" }
    }

    private func groupInteractionsByResidue() -> [ResidueKey: [MolecularInteraction]] {
        var groups: [ResidueKey: [MolecularInteraction]] = [:]
        for ixn in interactions {
            let pIdx = ixn.proteinAtomIndex
            guard pIdx < proteinAtoms.count else { continue }
            let atom = proteinAtoms[pIdx]
            let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
            groups[key, default: []].append(ixn)
        }
        return groups
    }

    // MARK: - Drawing Primitives (scaled up for legibility)

    private func drawBond(context: GraphicsContext, from p1: CGPoint, to p2: CGPoint, order: Int) {
        let bondColor = Color.primary.opacity(0.7)
        let lineW: CGFloat = 2.0  // thicker than before

        if order == 1 || order == 4 {
            // Single or aromatic — draw as single line; aromatic ring circles drawn separately
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(bondColor), lineWidth: lineW)
        } else if order == 2 {
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2, ny = dx / len * 2
            for sign in [-1.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * sign, y: p1.y + ny * sign))
                path.addLine(to: CGPoint(x: p2.x + nx * sign, y: p2.y + ny * sign))
                context.stroke(path, with: .color(bondColor), lineWidth: lineW - 0.3)
            }
        } else if order == 3 {
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2.5, ny = dx / len * 2.5
            for offset in [-1.0, 0.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * offset, y: p1.y + ny * offset))
                path.addLine(to: CGPoint(x: p2.x + nx * offset, y: p2.y + ny * offset))
                context.stroke(path, with: .color(bondColor), lineWidth: lineW - 0.5)
            }
        }
    }

    private func drawLigandAtom(context: GraphicsContext, at point: CGPoint, atomicNum: Int, index: Int) {
        let pharmaColor = pharmacophoreColor(forLigandAtomIndex: index)

        if atomicNum == 6 {
            // Carbon: slightly larger dot with optional halo
            let r: CGFloat = 3.5
            if let haloColor = pharmaColor {
                let hr: CGFloat = 10
                context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                             with: .color(haloColor.opacity(0.2)))
                context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                               with: .color(haloColor.opacity(0.6)), lineWidth: 1.5)
            }
            context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                         with: .color(.primary.opacity(0.6)))
            return
        }

        let (symbol, color) = atomDisplay(atomicNum)
        let r: CGFloat = 13  // larger circle for heteroatoms

        if let haloColor = pharmaColor {
            let hr: CGFloat = 18
            context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                         with: .color(haloColor.opacity(0.15)))
            context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                           with: .color(haloColor.opacity(0.5)), lineWidth: 2)
        }

        context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                     with: .color(Color(nsColor: .controlBackgroundColor)))
        context.stroke(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                       with: .color(color), lineWidth: 2)
        let text = Text(symbol).font(.body.monospaced().weight(.bold)).foregroundColor(color)
        context.draw(context.resolve(text), at: point, anchor: .center)
    }

    private func pharmacophoreColor(forLigandAtomIndex index: Int) -> Color? {
        let types = interactions.filter { $0.ligandAtomIndex == index }.map(\.type)
        guard !types.isEmpty else { return nil }

        if types.contains(.hbond) || types.contains(.halogen) {
            return Color(red: 0.2, green: 0.6, blue: 1.0)
        }
        if types.contains(.saltBridge) {
            return Color(red: 1.0, green: 0.5, blue: 0.1)
        }
        if types.contains(.piStack) || types.contains(.piCation) || types.contains(.amideStack) {
            return Color(red: 0.6, green: 0.3, blue: 0.9)
        }
        if types.contains(.metalCoord) {
            return Color(red: 1.0, green: 0.85, blue: 0.0)
        }
        if types.contains(.chalcogen) {
            return Color(red: 0.5, green: 0.8, blue: 0.2)
        }
        if types.contains(.hydrophobic) || types.contains(.chPi) {
            return Color(red: 0.9, green: 0.8, blue: 0.2)
        }
        return Color.gray.opacity(0.5)
    }

    private func drawResidueBubble(context: GraphicsContext, at center: CGPoint,
                                    key: ResidueKey, interactions: [MolecularInteraction],
                                    scPositions: [CGPoint], template: SideChainTemplate?) {
        let dominantType = dominantInteractionType(interactions)
        let (fillColor, borderColor) = residuePropertyColors(key.name)

        // Interacting atom names
        let atomNames = Set(interactions.compactMap { ixn -> String? in
            guard ixn.proteinAtomIndex < proteinAtoms.count else { return nil }
            return proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
        })

        // Measure text for proper sizing
        let nameSize = NSAttributedString(string: key.label,
                                          attributes: [.font: NSFont.systemFont(ofSize: 12, weight: .bold)]).size()
        let bubbleW: CGFloat = max(nameSize.width + 24, 70)
        let bubbleH: CGFloat = 36
        let rect = CGRect(x: center.x - bubbleW / 2, y: center.y - bubbleH / 2,
                          width: bubbleW, height: bubbleH)

        // --- Draw side chain (behind the core box) ---
        if let tmpl = template, !scPositions.isEmpty, scPositions.count == tmpl.atoms.count {
            // Bonds
            for bond in tmpl.bonds {
                guard bond.from < scPositions.count, bond.to < scPositions.count else { continue }
                let p1 = scPositions[bond.from], p2 = scPositions[bond.to]
                if bond.order == 2 {
                    let dx = p2.x - p1.x, dy = p2.y - p1.y, len = max(hypot(dx, dy), 1)
                    let nx = -dy / len * 1.5, ny = dx / len * 1.5
                    for sign in [-1.0, 1.0] as [CGFloat] {
                        var path = Path()
                        path.move(to: CGPoint(x: p1.x + nx * sign, y: p1.y + ny * sign))
                        path.addLine(to: CGPoint(x: p2.x + nx * sign, y: p2.y + ny * sign))
                        context.stroke(path, with: .color(borderColor.opacity(0.5)), lineWidth: 1.2)
                    }
                } else {
                    var path = Path()
                    path.move(to: p1); path.addLine(to: p2)
                    context.stroke(path, with: .color(borderColor.opacity(0.5)), lineWidth: 1.3)
                }
            }

            // Atoms
            for (i, atom) in tmpl.atoms.enumerated() {
                guard i < scPositions.count else { continue }
                let pos = scPositions[i]
                let isInteracting = atomNames.contains(atom.name)

                if atom.element == .C {
                    let r: CGFloat = 2.5
                    context.fill(Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                                 with: .color(borderColor.opacity(0.4)))
                } else {
                    let r: CGFloat = 7.5
                    let elemColor: Color = atom.element == .N ? .blue : atom.element == .O ? .red :
                        atom.element == .S ? .yellow : .gray
                    context.fill(Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                                 with: .color(Color(nsColor: .controlBackgroundColor)))
                    context.stroke(Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                                   with: .color(elemColor), lineWidth: 1.5)
                    context.draw(context.resolve(
                        Text(atom.element.symbol).font(.footnote.monospaced().weight(.bold))
                            .foregroundColor(elemColor)), at: pos, anchor: .center)
                }

                // Highlight interacting atoms with colored ring
                if isInteracting {
                    let hr: CGFloat = atom.element == .C ? 7 : 12
                    context.stroke(
                        Path(ellipseIn: CGRect(x: pos.x - hr, y: pos.y - hr, width: hr * 2, height: hr * 2)),
                        with: .color(interactionColor(dominantType)), lineWidth: 2)
                }
            }
        }

        // --- Core box (residue name) — drawn on top of side chain ---
        context.fill(Path(roundedRect: rect.offsetBy(dx: 1.5, dy: 1.5), cornerRadius: 8),
                     with: .color(Color(nsColor: .shadowColor).opacity(0.12)))
        context.fill(Path(roundedRect: rect, cornerRadius: 8), with: .color(fillColor))
        context.stroke(Path(roundedRect: rect, cornerRadius: 8), with: .color(borderColor), lineWidth: 2)

        // Residue label
        context.draw(context.resolve(
            Text(key.label).font(.callout.weight(.bold)).foregroundColor(.primary)),
            at: CGPoint(x: center.x, y: center.y - 3), anchor: .center)

        // Dominant interaction type
        context.draw(context.resolve(
            Text(dominantType.label).font(.caption2.weight(.medium)).foregroundColor(borderColor)),
            at: CGPoint(x: center.x, y: center.y + 11), anchor: .center)
    }

    /// MOE-style: color by amino acid chemical property.
    private func residuePropertyColors(_ residueName: String) -> (fill: Color, border: Color) {
        let acidic = Set(["ASP", "GLU"])
        let basic = Set(["ARG", "LYS", "HIS"])
        let polar = Set(["SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP"])

        if acidic.contains(residueName) {
            return (Color(red: 0.95, green: 0.85, blue: 0.85), Color(red: 0.8, green: 0.2, blue: 0.2))
        }
        if basic.contains(residueName) {
            return (Color(red: 0.85, green: 0.85, blue: 0.95), Color(red: 0.3, green: 0.3, blue: 0.9))
        }
        if polar.contains(residueName) {
            return (Color(red: 0.85, green: 0.92, blue: 0.95), Color(red: 0.2, green: 0.6, blue: 0.7))
        }
        return (Color(red: 0.88, green: 0.92, blue: 0.85), Color(red: 0.4, green: 0.6, blue: 0.3))
    }

    private func dominantInteractionType(_ interactions: [MolecularInteraction]) -> MolecularInteraction.InteractionType {
        let types = Set(interactions.map(\.type))
        if types.contains(.hbond) { return .hbond }
        if types.contains(.saltBridge) { return .saltBridge }
        if types.contains(.metalCoord) { return .metalCoord }
        if types.contains(.piStack) { return .piStack }
        if types.contains(.piCation) { return .piCation }
        if types.contains(.amideStack) { return .amideStack }
        if types.contains(.halogen) { return .halogen }
        if types.contains(.chalcogen) { return .chalcogen }
        if types.contains(.chPi) { return .chPi }
        return .hydrophobic
    }

    private func drawInteractionLine(context: GraphicsContext, from: CGPoint, to: CGPoint,
                                      type: MolecularInteraction.InteractionType, distance: Float) {
        let color = interactionColor(type)
        let lineW: CGFloat = 2.0

        switch type {
        case .hbond:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 6, lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 7)
        case .hydrophobic, .chPi:
            let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            let nx = -dy / max(len, 1) * 15
            let ny = dx / max(len, 1) * 15
            let ctrl = CGPoint(x: mid.x + nx, y: mid.y + ny)
            var path = Path()
            path.move(to: from)
            path.addQuadCurve(to: to, control: ctrl)
            context.stroke(path, with: .color(color), lineWidth: 1.5)
        case .saltBridge:
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2.5, ny = dx / len * 2.5
            for sign in [-1.0, 1.0] {
                let f = CGPoint(x: from.x + nx * sign, y: from.y + ny * sign)
                let t = CGPoint(x: to.x + nx * sign, y: to.y + ny * sign)
                drawDashedLine(ctx: context, from: f, to: t, color: color, dashLen: 5, lineWidth: 1.5)
            }
        case .piStack:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
        case .piCation:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 5, lineWidth: lineW)
        case .halogen:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 6)
        case .metalCoord:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: 2.5)
        case .amideStack:
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 6, lineWidth: lineW)
        case .chalcogen:
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: lineW)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 6)
        }

        // Distance label — offset perpendicular to line with background pill
        if type != .hydrophobic && type != .chPi {
            let lineLen = hypot(to.x - from.x, to.y - from.y)
            guard lineLen > 30 else { return } // skip labels on very short lines
            let t: CGFloat = 0.40
            let lp = CGPoint(x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t)
            let dx = to.x - from.x, dy = to.y - from.y
            let len = max(lineLen, 1)
            // Alternate offset side based on line angle to reduce overlaps
            let side: CGFloat = (from.x + from.y).truncatingRemainder(dividingBy: 2) < 1 ? 1 : -1
            let ox = -dy / len * 10 * side, oy = dx / len * 10 * side
            let labelPt = CGPoint(x: lp.x + ox, y: lp.y + oy)
            let distStr = String(format: "%.1f", distance)
            let distText = Text(distStr)
                .font(.footnote.monospaced().weight(.medium))
                .foregroundColor(.secondary)
            // Background pill for readability
            let sz = NSAttributedString(string: distStr,
                                        attributes: [.font: NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)]).size()
            let pill = CGRect(x: labelPt.x - sz.width / 2 - 3, y: labelPt.y - sz.height / 2 - 1,
                              width: sz.width + 6, height: sz.height + 2)
            context.fill(Path(roundedRect: pill, cornerRadius: 4),
                         with: .color(Color(nsColor: .controlBackgroundColor).opacity(0.85)))
            context.draw(context.resolve(distText), at: labelPt, anchor: .center)
        }
    }

    // MARK: - Geometry Utilities

    /// Check if a line segment crosses through a rectangle (both endpoints outside, line crosses edges).
    private func lineSegmentCrossesRect(from p1: CGPoint, to p2: CGPoint, rect: CGRect) -> Bool {
        // If both endpoints inside, no crossing (contained)
        if rect.contains(p1) && rect.contains(p2) { return false }
        // If either endpoint inside, the line enters the rect
        if rect.contains(p1) || rect.contains(p2) { return false } // not "crossing through"
        // Check intersection with each of the 4 edges
        let edges: [(CGPoint, CGPoint)] = [
            (CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY)),
            (CGPoint(x: rect.maxX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.maxY)),
            (CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.maxY)),
            (CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.minY))
        ]
        var crossCount = 0
        for (e1, e2) in edges {
            if segmentsIntersect(p1, p2, e1, e2) { crossCount += 1 }
        }
        return crossCount >= 2 // enters and exits = crosses through
    }

    private func segmentsIntersect(_ a1: CGPoint, _ a2: CGPoint, _ b1: CGPoint, _ b2: CGPoint) -> Bool {
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        let d1 = cross(b1, b2, a1), d2 = cross(b1, b2, a2)
        let d3 = cross(a1, a2, b1), d4 = cross(a1, a2, b2)
        return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
               ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
    }

    // MARK: - Drawing Utilities

    private func drawDashedLine(ctx: GraphicsContext, from: CGPoint, to: CGPoint,
                                 color: Color, dashLen: CGFloat, lineWidth: CGFloat) {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        ctx.stroke(path, with: .color(color),
                   style: StrokeStyle(lineWidth: lineWidth, dash: [dashLen, dashLen * 0.7]))
    }

    private func drawArrowhead(context: GraphicsContext, at tip: CGPoint, from: CGPoint,
                                color: Color, size: CGFloat) {
        let dx = tip.x - from.x
        let dy = tip.y - from.y
        let len = sqrt(dx * dx + dy * dy)
        guard len > 0 else { return }
        let ux = dx / len, uy = dy / len
        let px = -uy, py = ux

        var arrow = Path()
        arrow.move(to: tip)
        arrow.addLine(to: CGPoint(x: tip.x - ux * size + px * size * 0.5,
                                  y: tip.y - uy * size + py * size * 0.5))
        arrow.addLine(to: CGPoint(x: tip.x - ux * size - px * size * 0.5,
                                  y: tip.y - uy * size - py * size * 0.5))
        arrow.closeSubpath()
        context.fill(arrow, with: .color(color))
    }

    private func interactionColor(_ type: MolecularInteraction.InteractionType) -> Color {
        let c = type.color
        return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
    }

    private func atomDisplay(_ atomicNum: Int) -> (String, Color) {
        switch atomicNum {
        case 7:  return ("N", .blue)
        case 8:  return ("O", .red)
        case 9:  return ("F", .green)
        case 15: return ("P", .orange)
        case 16: return ("S", .yellow)
        case 17: return ("Cl", .green)
        case 35: return ("Br", Color(red: 0.6, green: 0.2, blue: 0.2))
        case 53: return ("I", .purple)
        default:
            let elem = Element(rawValue: atomicNum)
            return (elem?.symbol ?? "?", .gray)
        }
    }

    private func shorten(from: CGPoint, to: CGPoint,
                         fromInset: CGFloat, toInset: CGFloat) -> (CGPoint, CGPoint) {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let len = sqrt(dx * dx + dy * dy)
        guard len > fromInset + toInset else { return (from, to) }
        let ux = dx / len, uy = dy / len
        return (CGPoint(x: from.x + ux * fromInset, y: from.y + uy * fromInset),
                CGPoint(x: to.x - ux * toInset, y: to.y - uy * toInset))
    }

    /// Resolve overlapping residue bubbles by angular redistribution.
    private func resolveOverlaps(
        _ placements: inout [(key: ResidueKey, center: CGPoint, interactions: [MolecularInteraction])],
        around origin: CGPoint,
        radius: CGFloat
    ) {
        guard placements.count > 1 else { return }

        struct PolarEntry {
            var angle: CGFloat
            var originalIndex: Int
        }

        var polar: [PolarEntry] = placements.enumerated().map { i, p in
            let dx = p.center.x - origin.x
            let dy = p.center.y - origin.y
            return PolarEntry(angle: atan2(dy, dx), originalIndex: i)
        }
        polar.sort { $0.angle < $1.angle }

        let minSpacing: CGFloat = 0.50  // ~29 degrees between bubbles

        // 5-pass relaxation for better distribution
        for _ in 0..<5 {
            for i in 1..<polar.count {
                let gap = polar[i].angle - polar[i - 1].angle
                if gap < minSpacing {
                    polar[i].angle = polar[i - 1].angle + minSpacing
                }
            }
            if polar.count > 1 {
                let wrapGap = (polar[0].angle + 2 * .pi) - polar[polar.count - 1].angle
                if wrapGap < minSpacing {
                    polar[polar.count - 1].angle = (polar[0].angle + 2 * .pi) - minSpacing
                }
            }
        }

        // Scale radius outward when many residues
        let totalSpan = polar.last!.angle - polar.first!.angle
        let effectiveRadius: CGFloat
        if placements.count > 12 {
            effectiveRadius = radius * 1.25
        } else if totalSpan > 1.8 * .pi {
            effectiveRadius = radius * 1.15
        } else {
            effectiveRadius = radius
        }

        for entry in polar {
            let idx = entry.originalIndex
            placements[idx].center = CGPoint(
                x: origin.x + cos(entry.angle) * effectiveRadius,
                y: origin.y + sin(entry.angle) * effectiveRadius
            )
        }
    }
}

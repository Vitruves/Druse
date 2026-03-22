import SwiftUI

/// 2D Protein-Ligand Interaction Diagram (ProLig2D).
/// Renders a flat schematic with the ligand centered and interacting residues
/// arranged radially. H-bonds shown as dashed arrows, hydrophobic contacts as
/// arcs, salt bridges, pi-stacking, halogen bonds, and metal coordination.
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

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if isLoading {
                ProgressView("Generating 2D layout...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let coords = coords2D {
                Canvas { context, size in
                    drawDiagram(context: context, size: size, coords: coords)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .controlBackgroundColor))
            } else {
                fallbackDiagram
            }
            Divider()
            legend
        }
        .frame(minWidth: 600, minHeight: 500)
        .task {
            if let smiles = ligandSmiles, !smiles.isEmpty {
                let result = await Task.detached {
                    RDKitBridge.compute2DCoords(smiles: smiles)
                }.value
                coords2D = result
            }
            isLoading = false
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Label("Interaction Diagram", systemImage: "circle.hexagongrid")
                .font(.system(size: 13, weight: .semibold))
            Spacer()
            Text("Pose #\(poseIndex + 1)")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f kcal/mol", poseEnergy))
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(poseEnergy < -6 ? .green : poseEnergy < 0 ? .orange : .red)
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
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

        return HStack(spacing: 16) {
            ForEach(types, id: \.rawValue) { type in
                let count = interactions.filter { $0.type == type }.count
                HStack(spacing: 4) {
                    interactionSymbol(type)
                        .frame(width: 20, height: 12)
                    Text("\(type.label) (\(count))")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text("\(interactions.count) interactions")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func interactionSymbol(_ type: MolecularInteraction.InteractionType) -> some View {
        let c = interactionColor(type)
        Canvas { ctx, size in
            let y = size.height / 2
            switch type {
            case .hbond:
                // Dashed line
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y), to: CGPoint(x: size.width, y: y),
                               color: c, dashLen: 3, lineWidth: 1.5)
            case .hydrophobic:
                // Gray arc
                var path = Path()
                path.move(to: CGPoint(x: 2, y: size.height))
                path.addQuadCurve(to: CGPoint(x: size.width - 2, y: size.height),
                                  control: CGPoint(x: size.width / 2, y: -2))
                ctx.stroke(path, with: .color(c), lineWidth: 1.5)
            case .saltBridge:
                // Double dashed
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y - 1.5), to: CGPoint(x: size.width, y: y - 1.5),
                               color: c, dashLen: 2, lineWidth: 1)
                drawDashedLine(ctx: ctx, from: CGPoint(x: 0, y: y + 1.5), to: CGPoint(x: size.width, y: y + 1.5),
                               color: c, dashLen: 2, lineWidth: 1)
            default:
                // Solid colored line
                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(path, with: .color(c), lineWidth: 1.5)
            }
        }
    }

    // MARK: - Fallback (no SMILES / no 2D coords)

    private var fallbackDiagram: some View {
        VStack(spacing: 8) {
            Image(systemName: "atom")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text("Could not generate 2D layout")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Text("SMILES required for 2D depiction")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Main Drawing

    private func drawDiagram(context: GraphicsContext, size: CGSize, coords: RDKitBridge.Coords2D) {
        let padding: CGFloat = 120  // space around ligand for residue bubbles

        // 1. Compute ligand 2D bounding box and scale
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
        let scale = min(ligandAreaW / rangeX, ligandAreaH / rangeY) * 0.8
        let cx = size.width / 2
        let cy = size.height / 2
        let midX = (minX + maxX) / 2
        let midY = (minY + maxY) / 2

        func project(_ p: CGPoint) -> CGPoint {
            CGPoint(x: cx + (p.x - midX) * scale,
                    y: cy - (p.y - midY) * scale)
        }

        // 2. Draw ligand bonds
        for (a1, a2, order) in coords.bonds {
            guard a1 < positions.count, a2 < positions.count else { continue }
            let p1 = project(positions[a1])
            let p2 = project(positions[a2])
            drawBond(context: context, from: p1, to: p2, order: order)
        }

        // 3. Draw ligand atoms
        for (i, pos) in positions.enumerated() {
            let p = project(pos)
            let atomicNum = coords.atomicNums[i]
            drawLigandAtom(context: context, at: p, atomicNum: atomicNum, index: i)
        }

        // 4. Group interactions by residue
        let residueGroups = groupInteractionsByResidue()
        guard !residueGroups.isEmpty else { return }

        // 5. Place residue bubbles around the ligand perimeter
        // For each residue, find the average ligand atom position it interacts with,
        // then push the residue outward from the ligand center
        let ligandCenter = CGPoint(x: cx, y: cy)

        // Build mapping from 3D ligand heavy atom index → 2D atom index
        // The 2D coords are for heavy atoms only in canonical order.
        // The ligandAtoms in interactions reference the heavy atom array from docking.
        // We need to map ligandAtomIndex → 2D position.
        // Since both are heavy-atoms-only from the same SMILES, indices should align.

        var residuePlacements: [(key: ResidueKey, center: CGPoint, interactions: [MolecularInteraction])] = []

        for (key, ixns) in residueGroups {
            // Average 2D position of interacting ligand atoms
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

            // Direction from ligand center to the average contact point, then push outward
            let dx = avgLigandPos.x - ligandCenter.x
            let dy = avgLigandPos.y - ligandCenter.y
            let dist = max(sqrt(dx * dx + dy * dy), 1)
            let pushDist: CGFloat = min(size.width, size.height) * 0.40
            let residueCenter = CGPoint(
                x: ligandCenter.x + dx / dist * pushDist,
                y: ligandCenter.y + dy / dist * pushDist
            )

            residuePlacements.append((key, residueCenter, ixns))
        }

        // Resolve overlapping residue bubbles by angular redistribution
        resolveOverlaps(&residuePlacements, around: ligandCenter,
                        radius: min(size.width, size.height) * 0.40)

        // 6. Draw interactions and residue bubbles
        for (key, residueCenter, ixns) in residuePlacements {
            // Draw interaction lines from ligand atom to residue bubble
            for ixn in ixns {
                let ligIdx = ixn.ligandAtomIndex
                guard ligIdx < positions.count else { continue }
                let ligPos = project(positions[ligIdx])

                // Shorten line so it stops at bubble edge and atom edge
                let toResidue = shorten(from: ligPos, to: residueCenter, fromInset: 6, toInset: 36)
                drawInteractionLine(context: context, from: toResidue.0, to: toResidue.1,
                                    type: ixn.type, distance: ixn.distance)
            }

            // Draw residue bubble
            drawResidueBubble(context: context, at: residueCenter, key: key,
                              interactions: ixns)
        }
    }

    // MARK: - Residue Grouping

    private struct ResidueKey: Hashable {
        let name: String
        let seq: Int
        let chain: String

        var label: String {
            "\(name)\(seq)"
        }
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

    // MARK: - Drawing Primitives

    private func drawBond(context: GraphicsContext, from p1: CGPoint, to p2: CGPoint, order: Int) {
        let bondColor = Color.primary.opacity(0.7)

        if order == 1 || order == 4 {
            // Single or aromatic
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(bondColor), lineWidth: order == 4 ? 1.5 : 1.2)

            if order == 4 {
                // Aromatic: draw inner dashed line
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let len = sqrt(dx * dx + dy * dy)
                guard len > 0 else { return }
                let nx = -dy / len * 2.5, ny = dx / len * 2.5
                drawDashedLine(ctx: context,
                               from: CGPoint(x: p1.x + nx, y: p1.y + ny),
                               to: CGPoint(x: p2.x + nx, y: p2.y + ny),
                               color: bondColor.opacity(0.5), dashLen: 3, lineWidth: 0.8)
            }
        } else if order == 2 {
            // Double bond: two parallel lines
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 1.5, ny = dx / len * 1.5
            for sign in [-1.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * sign, y: p1.y + ny * sign))
                path.addLine(to: CGPoint(x: p2.x + nx * sign, y: p2.y + ny * sign))
                context.stroke(path, with: .color(bondColor), lineWidth: 1)
            }
        } else if order == 3 {
            // Triple bond
            let dx = p2.x - p1.x, dy = p2.y - p1.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2, ny = dx / len * 2
            for offset in [-1.0, 0.0, 1.0] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * offset, y: p1.y + ny * offset))
                path.addLine(to: CGPoint(x: p2.x + nx * offset, y: p2.y + ny * offset))
                context.stroke(path, with: .color(bondColor), lineWidth: 0.8)
            }
        }
    }

    private func drawLigandAtom(context: GraphicsContext, at point: CGPoint, atomicNum: Int, index: Int) {
        // Pharmacophoric halo: check if this atom is involved in any interaction
        let pharmaColor = pharmacophoreColor(forLigandAtomIndex: index)

        if atomicNum == 6 {
            // Carbon: small dot, optionally with pharmacophoric halo
            let r: CGFloat = 2
            if let haloColor = pharmaColor {
                let hr: CGFloat = 7
                context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                             with: .color(haloColor.opacity(0.2)))
                context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                               with: .color(haloColor.opacity(0.6)), lineWidth: 1)
            }
            context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                         with: .color(.primary.opacity(0.6)))
            return
        }

        let (symbol, color) = atomDisplay(atomicNum)
        let r: CGFloat = 9

        // Pharmacophoric halo (larger ring behind the atom)
        if let haloColor = pharmaColor {
            let hr: CGFloat = 13
            context.fill(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                         with: .color(haloColor.opacity(0.15)))
            context.stroke(Path(ellipseIn: CGRect(x: point.x - hr, y: point.y - hr, width: hr * 2, height: hr * 2)),
                           with: .color(haloColor.opacity(0.5)), lineWidth: 1.5)
        }

        // Background circle
        context.fill(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                     with: .color(Color(nsColor: .controlBackgroundColor)))
        // Colored border
        context.stroke(Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2)),
                       with: .color(color), lineWidth: 1.5)
        // Label
        let text = Text(symbol).font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundColor(color)
        context.draw(context.resolve(text), at: point, anchor: .center)
    }

    /// Get pharmacophoric halo color for a ligand atom based on its interaction types.
    private func pharmacophoreColor(forLigandAtomIndex index: Int) -> Color? {
        let types = interactions.filter { $0.ligandAtomIndex == index }.map(\.type)
        guard !types.isEmpty else { return nil }

        // Priority: H-bond > salt bridge > pi-stack > hydrophobic
        if types.contains(.hbond) || types.contains(.halogen) {
            return Color(red: 0.2, green: 0.6, blue: 1.0)       // H-bond: blue
        }
        if types.contains(.saltBridge) {
            return Color(red: 1.0, green: 0.5, blue: 0.1)       // Salt bridge: orange
        }
        if types.contains(.piStack) || types.contains(.piCation) {
            return Color(red: 0.6, green: 0.3, blue: 0.9)       // Aromatic: purple
        }
        if types.contains(.metalCoord) {
            return Color(red: 1.0, green: 0.85, blue: 0.0)      // Metal: gold
        }
        if types.contains(.hydrophobic) || types.contains(.chPi) {
            return Color(red: 0.9, green: 0.8, blue: 0.2)       // Hydrophobic: yellow
        }
        return Color.gray.opacity(0.5)
    }

    private func drawResidueBubble(context: GraphicsContext, at center: CGPoint,
                                    key: ResidueKey, interactions: [MolecularInteraction]) {
        let bubbleW: CGFloat = 66
        let bubbleH: CGFloat = 44

        let rect = CGRect(x: center.x - bubbleW / 2, y: center.y - bubbleH / 2,
                          width: bubbleW, height: bubbleH)

        // Pharmacophoric coloring based on dominant interaction type
        let dominantType = dominantInteractionType(interactions)
        let (fillColor, borderColor) = residueBubbleColors(dominantType)

        context.fill(Path(roundedRect: rect, cornerRadius: 8), with: .color(fillColor))
        context.stroke(Path(roundedRect: rect, cornerRadius: 8), with: .color(borderColor), lineWidth: 1.5)

        // Small pharmacophoric indicator dot (top-right corner)
        let dotR: CGFloat = 4
        let dotCenter = CGPoint(x: center.x + bubbleW / 2 - 8, y: center.y - bubbleH / 2 + 8)
        context.fill(Path(ellipseIn: CGRect(x: dotCenter.x - dotR, y: dotCenter.y - dotR,
                                            width: dotR * 2, height: dotR * 2)),
                     with: .color(borderColor))

        // Residue label (e.g., "ASP25")
        let resLabel = Text(key.label)
            .font(.system(size: 10, weight: .bold))
            .foregroundColor(.primary)
        context.draw(context.resolve(resLabel), at: CGPoint(x: center.x, y: center.y - 8), anchor: .center)

        // Interacting protein atom names (e.g., "OD1, N") with chain
        let atomNames = uniqueProteinAtomNames(for: interactions)
        let detailText = atomNames.isEmpty ? key.chain : "\(key.chain): \(atomNames)"
        let detailLabel = Text(detailText)
            .font(.system(size: 7, weight: .medium, design: .monospaced))
            .foregroundColor(borderColor)
        context.draw(context.resolve(detailLabel), at: CGPoint(x: center.x, y: center.y + 6), anchor: .center)

        // Interaction type label (e.g., "H-bond", "Hydrophobic")
        let typeLabel = Text(dominantType.label)
            .font(.system(size: 6, weight: .medium))
            .foregroundColor(.secondary)
        context.draw(context.resolve(typeLabel), at: CGPoint(x: center.x, y: center.y + 15), anchor: .center)
    }

    /// Find the dominant interaction type for a set of interactions.
    private func dominantInteractionType(_ interactions: [MolecularInteraction]) -> MolecularInteraction.InteractionType {
        // Priority: H-bond > salt bridge > metal > pi-stack > halogen > hydrophobic
        let types = Set(interactions.map(\.type))
        if types.contains(.hbond) { return .hbond }
        if types.contains(.saltBridge) { return .saltBridge }
        if types.contains(.metalCoord) { return .metalCoord }
        if types.contains(.piStack) { return .piStack }
        if types.contains(.piCation) { return .piCation }
        if types.contains(.halogen) { return .halogen }
        if types.contains(.chPi) { return .chPi }
        return .hydrophobic
    }

    /// Get fill and border colors for a residue bubble based on pharmacophoric type.
    private func residueBubbleColors(_ type: MolecularInteraction.InteractionType) -> (fill: Color, border: Color) {
        switch type {
        case .hbond:
            return (Color.cyan.opacity(0.1), Color.cyan.opacity(0.6))
        case .hydrophobic, .chPi:
            return (Color.green.opacity(0.1), Color.green.opacity(0.5))
        case .saltBridge:
            return (Color.orange.opacity(0.1), Color.orange.opacity(0.5))
        case .piStack, .piCation:
            return (Color.purple.opacity(0.1), Color.purple.opacity(0.5))
        case .halogen:
            return (Color.green.opacity(0.08), Color.green.opacity(0.4))
        case .metalCoord:
            return (Color.yellow.opacity(0.1), Color.yellow.opacity(0.6))
        }
    }

    /// Extract unique protein atom names involved in interactions for a residue bubble.
    private func uniqueProteinAtomNames(for interactions: [MolecularInteraction]) -> String {
        var names: [String] = []
        var seen = Set<Int>()
        for ixn in interactions {
            let pIdx = ixn.proteinAtomIndex
            guard pIdx < proteinAtoms.count, !seen.contains(pIdx) else { continue }
            seen.insert(pIdx)
            let name = proteinAtoms[pIdx].name.trimmingCharacters(in: .whitespaces)
            if !name.isEmpty { names.append(name) }
        }
        // Limit to 3 atom names to keep bubble readable
        if names.count > 3 {
            return names.prefix(3).joined(separator: ",") + "..."
        }
        return names.joined(separator: ", ")
    }

    private func drawInteractionLine(context: GraphicsContext, from: CGPoint, to: CGPoint,
                                      type: MolecularInteraction.InteractionType, distance: Float) {
        let color = interactionColor(type)

        switch type {
        case .hbond:
            // Dashed arrow
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 5, lineWidth: 1.5)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 5)

        case .hydrophobic, .chPi:
            // Curved arc
            let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            let nx = -dy / max(len, 1) * 12
            let ny = dx / max(len, 1) * 12
            let ctrl = CGPoint(x: mid.x + nx, y: mid.y + ny)
            var path = Path()
            path.move(to: from)
            path.addQuadCurve(to: to, control: ctrl)
            context.stroke(path, with: .color(color), lineWidth: 1.2)

        case .saltBridge:
            // Double dashed line
            let dx = to.x - from.x, dy = to.y - from.y
            let len = sqrt(dx * dx + dy * dy)
            guard len > 0 else { return }
            let nx = -dy / len * 2, ny = dx / len * 2
            for sign in [-1.0, 1.0] {
                let f = CGPoint(x: from.x + nx * sign, y: from.y + ny * sign)
                let t = CGPoint(x: to.x + nx * sign, y: to.y + ny * sign)
                drawDashedLine(ctx: context, from: f, to: t, color: color, dashLen: 4, lineWidth: 1.2)
            }

        case .piStack:
            // Solid colored line with pi symbol at midpoint
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: 1.5)

        case .piCation:
            // Dashed colored line
            drawDashedLine(ctx: context, from: from, to: to, color: color, dashLen: 4, lineWidth: 1.5)

        case .halogen:
            // Solid green with arrow
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: 1.5)
            drawArrowhead(context: context, at: to, from: from, color: color, size: 4)

        case .metalCoord:
            // Solid gold line
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: .color(color), lineWidth: 2)
        }

        // Distance label at midpoint
        let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
        let distText = Text(String(format: "%.1f", distance))
            .font(.system(size: 7, design: .monospaced))
            .foregroundColor(.secondary)
        context.draw(context.resolve(distText), at: CGPoint(x: mid.x, y: mid.y - 6), anchor: .center)
    }

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

    // MARK: - Helpers

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

    /// Resolve overlapping residue bubbles by redistributing them angularly.
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

        // Larger bubbles need more angular spacing (~25-30 degrees)
        let minSpacing: CGFloat = 0.45

        // Multi-pass relaxation for better distribution
        for _ in 0..<3 {
            // Forward pass
            for i in 1..<polar.count {
                let gap = polar[i].angle - polar[i - 1].angle
                if gap < minSpacing {
                    polar[i].angle = polar[i - 1].angle + minSpacing
                }
            }
            // Wrap-around check: last vs first
            if polar.count > 1 {
                let wrapGap = (polar[0].angle + 2 * .pi) - polar[polar.count - 1].angle
                if wrapGap < minSpacing {
                    // Push the last element back slightly
                    polar[polar.count - 1].angle = (polar[0].angle + 2 * .pi) - minSpacing
                }
            }
        }

        // If total angular span exceeds 2*pi, scale radius outward
        let totalSpan = polar.last!.angle - polar.first!.angle
        let effectiveRadius = totalSpan > 1.8 * .pi ? radius * 1.15 : radius

        // Apply corrected positions
        for entry in polar {
            let idx = entry.originalIndex
            placements[idx].center = CGPoint(
                x: origin.x + cos(entry.angle) * effectiveRadius,
                y: origin.y + sin(entry.angle) * effectiveRadius
            )
        }
    }
}

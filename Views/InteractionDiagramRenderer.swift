// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Residue Key

struct ResidueKey: Hashable {
    let name: String
    let seq: Int
    let chain: String
    var label: String { "\(name)\(seq)" }
}

// MARK: - DiagramScene
//
// All geometry needed to draw the interaction diagram, computed once from
// the input coordinates and interactions. Drawing is a pure function of this
// scene plus a target canvas size (the auto-fit transform is applied at draw
// time so the scene is canvas-size independent).

struct DiagramScene {
    // --- Ligand ---
    let ligandPositions: [CGPoint]                            // 2D, centered at origin
    let ligandBonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)]
    let ligandAtomicNums: [Int]
    let ligandFormalCharges: [Int]
    let ligandRings: [[Int]]                                  // aromatic rings (atom indices)
    let ligandAABB: CGRect
    let ligandCenter: CGPoint                                 // geometric center for outward normals
    let ligandHalos: [Int: DiagramHalo]                       // ligand atom index → halo kind (non-π)

    // --- Residues ---
    let residues: [ResidueDraw]
    struct ResidueDraw {
        let key: ResidueKey
        let center: CGPoint
        let template: SideChainTemplate?
        let scPositions: [CGPoint]
        let interactingAtomNames: Set<String>
        let residueClass: DiagramResidueClass
        let dominantType: MolecularInteraction.InteractionType
        let interactions: [MolecularInteraction]
    }

    // --- Connectors ---
    let directionalLines: [DirectionalLine]
    struct DirectionalLine {
        let from: CGPoint                                     // ligand contact
        let to: CGPoint                                       // residue side-chain atom or ring centroid
        let routedVia: CGPoint?                               // bezier control if routed around ligand
        let type: MolecularInteraction.InteractionType
        let distance: Float
    }
    let hydrophobicMarks: [HydrophobicMark]
    struct HydrophobicMark {
        let anchor: CGPoint                                   // group centroid (or ring centroid)
        let baseline: CGPoint                                 // eyelash arc center, outside the ligand
        let outwardNormal: CGPoint                            // unit vector anchor → baseline
        let span: CGFloat                                     // tangent span across the eyelash
        let residueCenter: CGPoint                            // single residue this mark connects to
    }
}

// MARK: - Constants

private let kBaseScale: CGFloat = 30
private let kPiTypes: Set<MolecularInteraction.InteractionType> = [.piStack, .piCation, .chPi, .amideStack]
private let kDiagramHydroExcluded: Set<String> = ["ASP", "GLU", "ASN", "GLN", "SER", "THR", "CYS"]
private let kDiagramHydroAllowed: [String: Set<String>] = [
    "ARG": ["CB", "CG"],
    "LYS": ["CB", "CG", "CD"],
]

// MARK: - Scene Builder

/// Re-index a 2D depiction (atoms in SMILES parse order) onto the ligand's heavy-atom order,
/// which is the order interaction `ligandAtomIndex` values refer to. A SMILES derived from an
/// SDF/PDB ligand is RDKit-canonical, so its atom order differs from the file's; matching the
/// two molecular graphs (element + bonds) puts every contact on the right depicted atom.
/// Returns nil when the graphs are not the same molecule.
func alignDepiction(
    _ coords: RDKitBridge.Coords2D,
    toHeavyAtoms atoms: [Atom],
    bonds: [Bond]
) -> RDKitBridge.Coords2D? {
    let n = atoms.count
    guard coords.positions.count == n, coords.bonds.count == bonds.count else { return nil }
    let ligandElement = atoms.map { $0.element.rawValue }
    var depictionAdj = Array(repeating: Set<Int>(), count: n)
    for (a, b, _) in coords.bonds where a < n && b < n {
        depictionAdj[a].insert(b); depictionAdj[b].insert(a)
    }
    var ligandAdj = Array(repeating: Set<Int>(), count: n)
    for bond in bonds where bond.atomIndex1 < n && bond.atomIndex2 < n {
        ligandAdj[bond.atomIndex1].insert(bond.atomIndex2); ligandAdj[bond.atomIndex2].insert(bond.atomIndex1)
    }

    // ligand atom → depiction atom
    var map = Array(repeating: -1, count: n)
    let identity = (0..<n).allSatisfy { coords.atomicNums[$0] == ligandElement[$0] && depictionAdj[$0] == ligandAdj[$0] }
    if identity {
        map = Array(0..<n)
    } else {
        // Visit ligand atoms breadth-first so each one (after a component's first) has an
        // already-mapped neighbour that narrows its candidates to that neighbour's neighbours.
        var order: [Int] = []
        var queued = Array(repeating: false, count: n)
        for root in 0..<n where !queued[root] {
            queued[root] = true
            var head = order.count
            order.append(root)
            while head < order.count {
                for next in ligandAdj[order[head]].sorted() where !queued[next] {
                    queued[next] = true
                    order.append(next)
                }
                head += 1
            }
        }
        var used = Array(repeating: false, count: n)
        var budget = 200_000 // backtracking steps; symmetric molecules resolve far sooner
        func assign(_ k: Int) -> Bool {
            if k == order.count { return true }
            budget -= 1
            if budget < 0 { return false }
            let atom = order[k]
            let mappedNeighbors = ligandAdj[atom].filter { map[$0] >= 0 }
            let pool = mappedNeighbors.first.map { depictionAdj[map[$0]] } ?? Set(0..<n)
            for candidate in pool.sorted() where !used[candidate]
                && coords.atomicNums[candidate] == ligandElement[atom]
                && depictionAdj[candidate].count == ligandAdj[atom].count
                && mappedNeighbors.allSatisfy({ depictionAdj[candidate].contains(map[$0]) }) {
                map[atom] = candidate
                used[candidate] = true
                if assign(k + 1) { return true }
                map[atom] = -1
                used[candidate] = false
            }
            return false
        }
        guard assign(0) else { return nil }
    }

    var depictionToLigand = Array(repeating: 0, count: n)
    for (ligandIndex, depictionIndex) in map.enumerated() { depictionToLigand[depictionIndex] = ligandIndex }
    return RDKitBridge.Coords2D(
        positions: map.map { coords.positions[$0] },
        atomicNums: map.map { coords.atomicNums[$0] },
        bonds: coords.bonds.map { (depictionToLigand[$0.0], depictionToLigand[$0.1], $0.2) },
        isAromatic: map.map { coords.isAromatic[$0] }
    )
}

/// Compute a complete diagram scene from raw input data. Pure function.
func buildDiagramScene(
    coords: RDKitBridge.Coords2D,
    interactions: [MolecularInteraction],
    ligandAtoms: [Atom],
    proteinAtoms: [Atom]
) -> DiagramScene? {
    let positions = coords.positions
    guard !positions.isEmpty else { return nil }

    // --- Project ligand to abstract space centered at (0,0) ---
    let xs = positions.map(\.x), ys = positions.map(\.y)
    let minX = xs.min()!, maxX = xs.max()!
    let minY = ys.min()!, maxY = ys.max()!
    let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2

    let projected = positions.map {
        CGPoint(x: ($0.x - midX) * kBaseScale, y: -($0.y - midY) * kBaseScale)
    }
    let ligandCenter = CGPoint.zero

    let projXs = projected.map(\.x), projYs = projected.map(\.y)
    let ligandAABB = CGRect(
        x: (projXs.min() ?? 0) - 20, y: (projYs.min() ?? 0) - 20,
        width: ((projXs.max() ?? 0) - (projXs.min() ?? 0)) + 40,
        height: ((projYs.max() ?? 0) - (projYs.min() ?? 0)) + 40
    )

    // --- Aromatic rings + ring centroid lookup ---
    let rings = detectAromaticRings(bonds: coords.bonds,
                                     atomCount: projected.count,
                                     isAromatic: coords.isAromatic)
    var ringCentroidForAtom: [Int: CGPoint] = [:]
    for ring in rings {
        let pts = ring.compactMap { $0 < projected.count ? projected[$0] : nil }
        guard pts.count == ring.count, !pts.isEmpty else { continue }
        let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
        let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
        for idx in ring { ringCentroidForAtom[idx] = CGPoint(x: cx, y: cy) }
    }

    func ligandContactPoint(for ixn: MolecularInteraction) -> CGPoint? {
        let idx = ixn.ligandAtomIndex
        guard idx < projected.count else { return nil }
        if kPiTypes.contains(ixn.type) {
            if let centroid = ringCentroidForAtom[idx] { return centroid }
            // Fallback: nearest aromatic ring centroid
            let atomPos = projected[idx]
            var bestDist: CGFloat = .greatestFiniteMagnitude
            var bestCentroid: CGPoint?
            for ring in rings {
                let pts = ring.compactMap { $0 < projected.count ? projected[$0] : nil }
                guard pts.count == ring.count, !pts.isEmpty else { continue }
                let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
                let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
                let d = hypot(atomPos.x - cx, atomPos.y - cy)
                if d < bestDist { bestDist = d; bestCentroid = CGPoint(x: cx, y: cy) }
            }
            if let centroid = bestCentroid, bestDist < 40 { return centroid }
        }
        return projected[idx]
    }

    // --- Group interactions by residue (with hydrophobic display filter) ---
    var residueGroups: [ResidueKey: [MolecularInteraction]] = [:]
    for ixn in interactions {
        let pIdx = ixn.proteinAtomIndex
        guard pIdx < proteinAtoms.count else { continue }
        let atom = proteinAtoms[pIdx]
        if ixn.type == .hydrophobic {
            let resName = atom.residueName
            let atomName = atom.name.trimmingCharacters(in: .whitespaces)
            if kDiagramHydroExcluded.contains(resName) { continue }
            if let allowed = kDiagramHydroAllowed[resName], !allowed.contains(atomName) { continue }
        }
        let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
        residueGroups[key, default: []].append(ixn)
    }
    guard !residueGroups.isEmpty else { return nil }

    // --- Place residues on an orbit around the ligand ---
    let ligandHalfDiag = hypot(ligandAABB.width, ligandAABB.height) / 2
    let orbitRadius = ligandHalfDiag + 180   // CB at edge-50, accommodates long chains

    var placements: [(key: ResidueKey, angle: CGFloat,
                      center: CGPoint, ixns: [MolecularInteraction])] = []
    for (key, ixns) in residueGroups {
        // Weight contact points: closer-to-ligand-center contacts get more weight
        var sumX: CGFloat = 0, sumY: CGFloat = 0, wTotal: CGFloat = 0
        for ixn in ixns {
            guard let p = ligandContactPoint(for: ixn) else { continue }
            // Inverse-distance weighting biases toward the strongest direction
            // when contacts span > 90°.
            let w: CGFloat = 1.0 / (1.0 + CGFloat(ixn.distance))
            sumX += p.x * w; sumY += p.y * w; wTotal += w
        }
        guard wTotal > 0 else { continue }
        let avgPos = CGPoint(x: sumX / wTotal, y: sumY / wTotal)
        let angle = atan2(avgPos.y - ligandCenter.y, avgPos.x - ligandCenter.x)
        let center = CGPoint(x: ligandCenter.x + cos(angle) * orbitRadius,
                             y: ligandCenter.y + sin(angle) * orbitRadius)
        placements.append((key, angle, center, ixns))
    }

    // Sort by angle then resolve overlaps via angular relaxation (preserves order)
    placements.sort { $0.angle < $1.angle }
    relaxAnglesAndPlace(&placements, origin: ligandCenter, radius: orbitRadius)

    // --- Build per-residue draw data with side chain transforms ---
    var residues: [DiagramScene.ResidueDraw] = []
    for placement in placements {
        let (key, _, center, ixns) = placement

        // Average ligand contact direction (for side-chain orientation)
        var sX: CGFloat = 0, sY: CGFloat = 0, n: CGFloat = 0
        for ixn in ixns {
            if let p = ligandContactPoint(for: ixn) {
                sX += p.x; sY += p.y; n += 1
            }
        }
        let avgContact = n > 0 ? CGPoint(x: sX / n, y: sY / n) : ligandCenter

        let template = sideChainTemplates[key.name]
        var scPositions: [CGPoint] = []

        if let tmpl = template, !tmpl.atoms.isEmpty {
            // Centroid-pivot transform: every branch of the side chain stays
            // visible (no atoms hidden behind the bubble).
            scPositions = transformSideChain(template: tmpl,
                                              bubbleCenter: center,
                                              towardLigand: avgContact)
        }

        // Per-atom highlights are for directional interactions only —
        // hydrophobic contacts are visualized as eyelashes on the ligand side,
        // not as colored dots on side chain carbons.
        let interactingAtomNames = Set(ixns.compactMap { ixn -> String? in
            guard ixn.type != .hydrophobic else { return nil }
            guard ixn.proteinAtomIndex < proteinAtoms.count else { return nil }
            return proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
        })
        let dominant = dominantInteractionType(ixns)

        residues.append(DiagramScene.ResidueDraw(
            key: key, center: center, template: template,
            scPositions: scPositions,
            interactingAtomNames: interactingAtomNames,
            residueClass: DiagramResidueClass(residueName: key.name),
            dominantType: dominant, interactions: ixns))
    }

    // --- Build directional (non-hydrophobic) lines ---
    var directionalLines: [DiagramScene.DirectionalLine] = []
    for residue in residues {
        for ixn in residue.interactions where ixn.type != .hydrophobic {
            guard let ligPos = ligandContactPoint(for: ixn) else { continue }

            // Directional: target = side-chain atom or ring centroid
            var targetPos = residue.center
            if let tmpl = residue.template, !residue.scPositions.isEmpty,
               ixn.proteinAtomIndex < proteinAtoms.count {
                if kPiTypes.contains(ixn.type) {
                    let aName = proteinAtoms[ixn.proteinAtomIndex].name
                        .trimmingCharacters(in: .whitespaces)
                    if let centroid = sideChainRingCentroid(template: tmpl,
                                                            scPositions: residue.scPositions,
                                                            matchingAtomName: aName) {
                        targetPos = centroid
                    }
                } else {
                    let aName = proteinAtoms[ixn.proteinAtomIndex].name
                        .trimmingCharacters(in: .whitespaces)
                    if let idx = tmpl.atoms.firstIndex(where: { $0.name == aName }),
                       idx < residue.scPositions.count {
                        targetPos = residue.scPositions[idx]
                    }
                }
            }

            let toInset: CGFloat = (targetPos == residue.center) ? 40 : 8
            let shortened = shortenSegment(from: ligPos, to: targetPos,
                                            fromInset: 8, toInset: toInset)

            var routed: CGPoint? = nil
            if lineCrossesRect(from: shortened.0, to: shortened.1, rect: ligandAABB),
               !ligandAABB.contains(shortened.1) {
                let corners = [
                    CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.minY - 12),
                    CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.minY - 12),
                    CGPoint(x: ligandAABB.maxX + 12, y: ligandAABB.maxY + 12),
                    CGPoint(x: ligandAABB.minX - 12, y: ligandAABB.maxY + 12),
                ]
                let mid = CGPoint(x: (shortened.0.x + shortened.1.x) / 2,
                                  y: (shortened.0.y + shortened.1.y) / 2)
                routed = corners.min(by: {
                    hypot($0.x - mid.x, $0.y - mid.y) < hypot($1.x - mid.x, $1.y - mid.y)
                })
            }

            directionalLines.append(DiagramScene.DirectionalLine(
                from: shortened.0, to: shortened.1, routedVia: routed,
                type: ixn.type, distance: ixn.distance))
        }
    }

    // --- Hydrophobic clustering ---
    //
    // A residue's hydrophobic contacts are grouped by ligand connectivity:
    // contiguous contact atoms become one cluster (so an entire phenyl ring or
    // an alkyl segment becomes ONE eyelash, not one per atom). When every
    // cluster atom belongs to the same aromatic ring, the anchor is the ring
    // centroid; otherwise the cluster centroid. Each (residue, cluster) pair
    // produces its own eyelash, oriented toward that residue and pushed past
    // the ligand AABB so it never overlaps the molecule.
    var ligandAdj: [[Int]] = Array(repeating: [], count: projected.count)
    for bond in coords.bonds where bond.0 < projected.count && bond.1 < projected.count {
        ligandAdj[bond.0].append(bond.1)
        ligandAdj[bond.1].append(bond.0)
    }
    var ringIdxForAtom: [Int: Int] = [:]
    for (rIdx, ring) in rings.enumerated() {
        for a in ring { ringIdxForAtom[a] = rIdx }
    }

    var hydrophobicMarks: [DiagramScene.HydrophobicMark] = []
    for residue in residues {
        let contactAtoms = Set(residue.interactions
            .filter { $0.type == .hydrophobic }
            .map(\.ligandAtomIndex)
            .filter { $0 < projected.count })
        guard !contactAtoms.isEmpty else { continue }

        var visited = Set<Int>()
        for start in contactAtoms where !visited.contains(start) {
            var component: [Int] = []
            var queue = [start]
            visited.insert(start)
            while !queue.isEmpty {
                let n = queue.removeFirst()
                component.append(n)
                for nb in ligandAdj[n]
                where contactAtoms.contains(nb) && !visited.contains(nb) {
                    visited.insert(nb)
                    queue.append(nb)
                }
            }

            // Anchor: full ring centroid if all atoms share one aromatic ring,
            // otherwise centroid of the contact atoms themselves.
            let anchor: CGPoint
            let radius: CGFloat
            let ringIdxs = Set(component.compactMap { ringIdxForAtom[$0] })
            if ringIdxs.count == 1, let rIdx = ringIdxs.first {
                let ringPts = rings[rIdx].compactMap {
                    $0 < projected.count ? projected[$0] : nil
                }
                let cx = ringPts.map(\.x).reduce(0, +) / CGFloat(ringPts.count)
                let cy = ringPts.map(\.y).reduce(0, +) / CGFloat(ringPts.count)
                anchor = CGPoint(x: cx, y: cy)
                radius = ringPts.map { hypot($0.x - cx, $0.y - cy) }.max() ?? 0
            } else {
                let pts = component.map { projected[$0] }
                let cx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
                let cy = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
                anchor = CGPoint(x: cx, y: cy)
                radius = pts.map { hypot($0.x - cx, $0.y - cy) }.max() ?? 0
            }

            // Direction toward the residue (not the ligand center) — guarantees
            // the eyelash sits on the residue side of the cluster.
            let rdx = residue.center.x - anchor.x
            let rdy = residue.center.y - anchor.y
            let rlen = max(hypot(rdx, rdy), 1)
            let normal = CGPoint(x: rdx / rlen, y: rdy / rlen)

            // Push the baseline past whichever is further: cluster radius or
            // the ligand AABB exit point along the normal.
            let aabbExit = rayBoxExitDistance(origin: anchor,
                                               direction: normal, box: ligandAABB)
            // Just outside the cluster (radius + 14) OR just past the AABB
            // exit (+ 10), whichever is larger. Capped to prevent runaway
            // distances when the residue is very far from the ligand.
            let baseDist = min(max(radius + 14, aabbExit + 10), radius + 30)
            let baseline = CGPoint(x: anchor.x + normal.x * baseDist,
                                   y: anchor.y + normal.y * baseDist)
            let span = max(18, radius * 1.4)

            hydrophobicMarks.append(DiagramScene.HydrophobicMark(
                anchor: anchor, baseline: baseline,
                outwardNormal: normal, span: span,
                residueCenter: residue.center))
        }
    }

    // --- Pharmacophore halos per ligand atom (excluding π types) ---
    // π interactions are visualized by the line targeting the ring centroid,
    // so we don't add per-atom halos for them.
    var halos: [Int: DiagramHalo] = [:]
    for i in 0..<projected.count {
        if let halo = pharmacophoreHalo(forLigandAtomIndex: i, interactions: interactions) {
            halos[i] = halo
        }
    }

    // Formal charges
    var charges: [Int] = Array(repeating: 0, count: projected.count)
    for i in 0..<min(projected.count, ligandAtoms.count) {
        charges[i] = ligandAtoms[i].formalCharge
    }

    let typedBonds = coords.bonds.map { (a, b, order) -> (a: Int, b: Int, order: Int, isAromatic: Bool) in
        let aro = a < coords.isAromatic.count && b < coords.isAromatic.count
            && coords.isAromatic[a] && coords.isAromatic[b]
        return (a: a, b: b, order: order, isAromatic: aro)
    }

    return DiagramScene(
        ligandPositions: projected,
        ligandBonds: typedBonds,
        ligandAtomicNums: coords.atomicNums,
        ligandFormalCharges: charges,
        ligandRings: rings,
        ligandAABB: ligandAABB,
        ligandCenter: ligandCenter,
        ligandHalos: halos,
        residues: residues,
        directionalLines: directionalLines,
        hydrophobicMarks: hydrophobicMarks)
}

// MARK: - Layout helpers

private func relaxAnglesAndPlace(
    _ placements: inout [(key: ResidueKey, angle: CGFloat,
                          center: CGPoint, ixns: [MolecularInteraction])],
    origin: CGPoint,
    radius: CGFloat
) {
    guard placements.count > 1 else { return }
    var angles = placements.map(\.angle)
    let minSpacing: CGFloat = 0.56  // ~32°

    for _ in 0..<12 {
        for i in 1..<angles.count {
            let gap = angles[i] - angles[i - 1]
            if gap < minSpacing { angles[i] = angles[i - 1] + minSpacing }
        }
        for i in stride(from: angles.count - 2, through: 0, by: -1) {
            let gap = angles[i + 1] - angles[i]
            if gap < minSpacing { angles[i] = angles[i + 1] - minSpacing }
        }
        if angles.count > 1 {
            let wrap = (angles[0] + 2 * .pi) - angles[angles.count - 1]
            if wrap < minSpacing {
                let deficit = minSpacing - wrap
                angles[angles.count - 1] -= deficit / 2
                angles[0] += deficit / 2
            }
        }
    }

    let totalSpan = angles.last! - angles.first!
    let effRadius: CGFloat
    if placements.count > 14 {
        effRadius = radius * 1.3
    } else if placements.count > 10 {
        effRadius = radius * 1.2
    } else if totalSpan > 1.8 * .pi {
        effRadius = radius * 1.15
    } else {
        effRadius = radius
    }

    for i in 0..<placements.count {
        placements[i].angle = angles[i]
        placements[i].center = CGPoint(x: origin.x + cos(angles[i]) * effRadius,
                                       y: origin.y + sin(angles[i]) * effRadius)
    }
}

/// Centroid of a side-chain aromatic ring in screen space. For TRP (two
/// fused rings) picks whichever ring contains the named protein atom; falls
/// back to the first ring otherwise.
private func sideChainRingCentroid(template: SideChainTemplate,
                                   scPositions: [CGPoint],
                                   matchingAtomName: String) -> CGPoint? {
    guard !template.aromaticRings.isEmpty else { return nil }
    let ring = template.aromaticRings.first { ring in
        ring.contains(where: { idx in
            idx < template.atoms.count && template.atoms[idx].name == matchingAtomName
        })
    } ?? template.aromaticRings[0]
    var rx: CGFloat = 0, ry: CGFloat = 0, rn: CGFloat = 0
    for idx in ring where idx < scPositions.count {
        rx += scPositions[idx].x; ry += scPositions[idx].y; rn += 1
    }
    guard rn > 0 else { return nil }
    return CGPoint(x: rx / rn, y: ry / rn)
}

// MARK: - Skeletal Renderer
//
// Single shared function used for both ligand and side chains. Carbons are
// bare bond vertices (no marker). Heteroatoms render as element labels with
// a canvas-color background pill so the bond appears to terminate at the
// letter. Aromatic rings get an inscribed circle. Double bonds render as
// parallel lines (inner line shortened, offset toward ring centroid for ring
// bonds).

/// Unified RDKit-style skeletal geometry used for both the ligand and the
/// side chains so they look visually consistent. Colors come from `DiagramPalette`.
private enum Skeletal {
    static let bondWidth: CGFloat = 2.2
    static let heteroatomFontSize: CGFloat = NSFont.systemFontSize   // 13 pt, matches `.body`
    static let aromaticCircleScale: CGFloat = 0.55                   // of average vertex radius
    static let heteroatomLabelInset: CGFloat = 9                     // bond inset around heteroatom labels
}

// MARK: - Palette
//
// Every color the diagram draws, resolved once for a concrete appearance.
// Tokens are dynamic (light/dark); resolving them explicitly against an
// NSAppearance keeps on-screen drawing in sync with the window and lets
// exports always use the light ("white paper") variant.

/// Ligand-atom halo kinds (directional pharmacophore contacts).
enum DiagramHalo: Sendable {
    case hbond, saltBridge, metal, chalcogen, other
}

/// Residue property class driving the residue bubble colors.
enum DiagramResidueClass: Sendable, CaseIterable {
    case polar, acidic, basic, hydrophobic

    init(residueName: String) {
        switch residueName {
        case "ASP", "GLU":                                     self = .acidic
        case "ARG", "LYS", "HIS":                              self = .basic
        case "SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP":  self = .polar
        default:                                               self = .hydrophobic
        }
    }

    var label: String {
        switch self {
        case .polar:       "polar"
        case .acidic:      "acidic"
        case .basic:       "basic"
        case .hydrophobic: "hydrophobic"
        }
    }

    /// Dynamic theme color (for SwiftUI chrome such as legends).
    var themeColor: Color {
        switch self {
        case .polar:       DS.Color.water
        case .acidic:      DS.Color.danger
        case .basic:       DS.Color.protein
        case .hydrophobic: DS.Color.success
        }
    }
}

struct DiagramPalette: Sendable {
    let paper: Color
    let ink: Color
    let secondaryInk: Color
    let shadow: Color
    private let interactions: [MolecularInteraction.InteractionType: Color]
    private let residues: [DiagramResidueClass: Color]
    private let halos: [DiagramHalo: Color]
    private let elements: [Int: Color]
    let chargePositive: Color
    let chargeNegative: Color

    func interaction(_ type: MolecularInteraction.InteractionType) -> Color { interactions[type] ?? ink }
    func residue(_ cls: DiagramResidueClass) -> Color { residues[cls] ?? ink }
    func halo(_ kind: DiagramHalo) -> Color { halos[kind] ?? secondaryInk }
    func element(_ atomicNum: Int) -> Color { elements[atomicNum] ?? secondaryInk }

    /// Element colors (CPK-derived data palette) tuned for contrast on diagram paper.
    private static let elementTokens: [Int: Color] = [
        7:  DS.dynamic(light: 0x2F5BD1, dark: 0x7AA2FF),   // N
        8:  DS.dynamic(light: 0xCF222E, dark: 0xFF7B72),   // O
        9:  DS.dynamic(light: 0x1A7F37, dark: 0x56D364),   // F
        15: DS.dynamic(light: 0xBC4C00, dark: 0xF0883E),   // P
        16: DS.dynamic(light: 0x9A6700, dark: 0xE3B341),   // S
        17: DS.dynamic(light: 0x1A7F37, dark: 0x56D364),   // Cl
        35: DS.dynamic(light: 0x8B2E2E, dark: 0xE08A8A),   // Br
        53: DS.dynamic(light: 0x7A4FD6, dark: 0xB392F0),   // I
    ]

    @MainActor
    static func resolved(dark: Bool) -> DiagramPalette {
        let appearance = NSAppearance(named: dark ? .darkAqua : .aqua) ?? NSAppearance.currentDrawing()
        func r(_ color: Color) -> Color {
            var out = NSColor(color)
            appearance.performAsCurrentDrawingAppearance {
                out = NSColor(color).usingColorSpace(.sRGB) ?? out
            }
            return Color(nsColor: out)
        }
        let ink = r(DS.Color.diagramInk)
        let hbond = r(DS.Color.interactionHBond)
        return DiagramPalette(
            paper: r(DS.Color.diagramPaper),
            ink: ink,
            secondaryInk: ink.opacity(0.6),
            shadow: Color(nsColor: .shadowColor).opacity(dark ? 0.35 : 0.12),
            interactions: Dictionary(uniqueKeysWithValues:
                MolecularInteraction.InteractionType.allCases.map { ($0, r($0.themeColor)) }),
            residues: Dictionary(uniqueKeysWithValues:
                DiagramResidueClass.allCases.map { ($0, r($0.themeColor)) }),
            halos: [
                .hbond: hbond,
                .saltBridge: r(DS.Color.interactionSaltBridge),
                .metal: r(DS.Color.interactionMetal),
                .chalcogen: r(MolecularInteraction.InteractionType.chalcogen.themeColor),
                .other: ink.opacity(0.45),
            ],
            elements: elementTokens.mapValues(r),
            chargePositive: hbond,
            chargeNegative: r(DS.Color.danger))
    }
}

func drawSkeletal(
    context: GraphicsContext,
    positions: [CGPoint],
    bonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)],
    atomicNums: [Int],
    rings: [[Int]],
    formalCharges: [Int],
    palette: DiagramPalette,
    halos: [Int: DiagramHalo] = [:],
    labelOverrides: [Int: String] = [:]    // explicit labels (e.g. "NH", "OH")
) {
    // Per-ring info, kept 1:1 with `rings` so callers can index by position.
    struct RingInfo {
        let atoms: Set<Int>
        let centroid: CGPoint
        let avgRadius: CGFloat
    }
    let ringInfos: [RingInfo?] = rings.map { ring in
        var cx: CGFloat = 0, cy: CGFloat = 0, n: CGFloat = 0
        for idx in ring where idx < positions.count {
            cx += positions[idx].x; cy += positions[idx].y; n += 1
        }
        guard n > 0 else { return nil }
        let centroid = CGPoint(x: cx / n, y: cy / n)
        let avgR = ring
            .compactMap { $0 < positions.count ? positions[$0] : nil }
            .map { hypot($0.x - centroid.x, $0.y - centroid.y) }
            .reduce(0, +) / n
        return RingInfo(atoms: Set(ring), centroid: centroid, avgRadius: avgR)
    }

    // Helper: shrink a bond's endpoints if either atom is a heteroatom (so
    // the bond visually stops at the letter, not behind it)
    func bondEndpoints(_ a: Int, _ b: Int) -> (CGPoint, CGPoint) {
        let p1 = positions[a], p2 = positions[b]
        let isHetA = a < atomicNums.count && atomicNums[a] != 6
        let isHetB = b < atomicNums.count && atomicNums[b] != 6
        guard isHetA || isHetB else { return (p1, p2) }
        let dx = p2.x - p1.x, dy = p2.y - p1.y
        let len = hypot(dx, dy)
        guard len > 0 else { return (p1, p2) }
        let ux = dx / len, uy = dy / len
        let inset = Skeletal.heteroatomLabelInset
        let q1 = isHetA ? CGPoint(x: p1.x + ux * inset, y: p1.y + uy * inset) : p1
        let q2 = isHetB ? CGPoint(x: p2.x - ux * inset, y: p2.y - uy * inset) : p2
        return (q1, q2)
    }

    let bondColor = palette.ink.opacity(0.85)

    // --- Halos (drawn first, behind everything) ---
    for (idx, kind) in halos where idx < positions.count {
        let color = palette.halo(kind)
        let p = positions[idx]
        let isHet = idx < atomicNums.count && atomicNums[idx] != 6
        let r: CGFloat = isHet ? 18 : 10
        context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r,
                                             width: r * 2, height: r * 2)),
                     with: .color(color.opacity(0.18)))
        context.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r,
                                               width: r * 2, height: r * 2)),
                       with: .color(color.opacity(0.55)), lineWidth: 1.8)
    }

    // --- Bonds ---
    for bond in bonds {
        guard bond.a < positions.count, bond.b < positions.count else { continue }
        let (p1, p2) = bondEndpoints(bond.a, bond.b)
        let dx = p2.x - p1.x, dy = p2.y - p1.y
        let len = hypot(dx, dy)
        guard len > 0 else { continue }

        if bond.order == 2 && !bond.isAromatic {
            // Double bond: outer line full length, inner line 60% length
            // Inner offset direction: toward ring centroid if both atoms in
            // a ring, else perpendicular toward +y.
            var nx = -dy / len, ny = dx / len
            if let entry = ringInfos.compactMap({ $0 }).first(where: { $0.atoms.contains(bond.a) && $0.atoms.contains(bond.b) }) {
                let mid = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
                let toC = CGPoint(x: entry.centroid.x - mid.x, y: entry.centroid.y - mid.y)
                if nx * toC.x + ny * toC.y < 0 { nx = -nx; ny = -ny }
            }
            // Outer line on the original bond axis
            var outer = Path(); outer.move(to: p1); outer.addLine(to: p2)
            context.stroke(outer, with: .color(bondColor), lineWidth: Skeletal.bondWidth)
            // Inner line offset by ~3 px
            let off: CGFloat = 3.5
            let shrink: CGFloat = 0.18
            let i1 = CGPoint(x: p1.x + dx * shrink + nx * off,
                             y: p1.y + dy * shrink + ny * off)
            let i2 = CGPoint(x: p2.x - dx * shrink + nx * off,
                             y: p2.y - dy * shrink + ny * off)
            var inner = Path(); inner.move(to: i1); inner.addLine(to: i2)
            context.stroke(inner, with: .color(bondColor),
                           lineWidth: Skeletal.bondWidth * 0.85)
        } else if bond.order == 3 {
            // Triple bond: three parallel lines
            let nx = -dy / len, ny = dx / len
            let off: CGFloat = 2.6
            for sign in [-1.0, 0.0, 1.0] as [CGFloat] {
                var path = Path()
                path.move(to: CGPoint(x: p1.x + nx * off * sign, y: p1.y + ny * off * sign))
                path.addLine(to: CGPoint(x: p2.x + nx * off * sign, y: p2.y + ny * off * sign))
                context.stroke(path, with: .color(bondColor),
                               lineWidth: Skeletal.bondWidth * 0.85)
            }
        } else {
            // Single bond (or aromatic rendered as single line + ring circle)
            var path = Path(); path.move(to: p1); path.addLine(to: p2)
            context.stroke(path, with: .color(bondColor), lineWidth: Skeletal.bondWidth)
        }
    }

    // --- Aromatic ring inscribed circles ---
    for info in ringInfos {
        guard let info else { continue }
        let r = info.avgRadius * Skeletal.aromaticCircleScale
        let rect = CGRect(x: info.centroid.x - r, y: info.centroid.y - r,
                          width: r * 2, height: r * 2)
        context.stroke(Path(ellipseIn: rect),
                       with: .color(bondColor.opacity(0.6)),
                       lineWidth: Skeletal.bondWidth * 0.7)
    }

    // --- Atoms ---
    for (i, pos) in positions.enumerated() {
        let z = i < atomicNums.count ? atomicNums[i] : 6
        // Carbon: bare vertex (skeletal style)
        if z == 6 { continue }
        let symbol = labelOverrides[i] ?? elementSymbol(z)
        let color = palette.element(z)
        // Background pill sized to the actual rendered text
        let nsFont = NSFont.monospacedSystemFont(ofSize: Skeletal.heteroatomFontSize, weight: .bold)
        let textSize = NSAttributedString(string: symbol, attributes: [.font: nsFont]).size()
        let labelW = textSize.width + 6
        let labelH = textSize.height + 2
        context.fill(Path(roundedRect: CGRect(x: pos.x - labelW / 2,
                                              y: pos.y - labelH / 2,
                                              width: labelW, height: labelH),
                                       cornerRadius: labelH / 2),
                     with: .color(palette.paper))
        let text = Text(symbol)
            .font(.system(.body, design: .monospaced).weight(.bold))
            .foregroundColor(color)
        context.draw(context.resolve(text), at: pos, anchor: .center)

        // Formal charge
        if i < formalCharges.count, formalCharges[i] != 0 {
            let label = formalCharges[i] > 0 ? "+" : "−"
            let chargeColor = formalCharges[i] > 0 ? palette.chargePositive : palette.chargeNegative
            context.draw(context.resolve(
                Text(label).font(.system(.callout, design: .rounded).weight(.heavy))
                    .foregroundColor(chargeColor)),
                at: CGPoint(x: pos.x + 9, y: pos.y - 9), anchor: .center)
        }
    }
}

// MARK: - Scene Drawer

/// Draw a complete diagram scene with auto-fit. Returns the screen-space
/// residue centers (after the auto-fit transform) for hover hit-testing.
@discardableResult
func drawScene(_ scene: DiagramScene, in context: GraphicsContext, size: CGSize,
               palette: DiagramPalette)
    -> [(label: String, center: CGPoint, interactions: [MolecularInteraction])]
{
    // --- Auto-fit ---
    var allPoints = scene.ligandPositions
    for r in scene.residues {
        allPoints.append(CGPoint(x: r.center.x - 50, y: r.center.y - 24))
        allPoints.append(CGPoint(x: r.center.x + 50, y: r.center.y + 24))
        for p in r.scPositions {
            allPoints.append(CGPoint(x: p.x - 14, y: p.y - 14))
            allPoints.append(CGPoint(x: p.x + 14, y: p.y + 14))
        }
    }
    for mark in scene.hydrophobicMarks {
        let outDist: CGFloat = mark.span * 0.12 + 12      // bulge + tooth
        let tipX = mark.baseline.x + mark.outwardNormal.x * outDist
        let tipY = mark.baseline.y + mark.outwardNormal.y * outDist
        let tx = -mark.outwardNormal.y * mark.span / 2
        let ty = mark.outwardNormal.x * mark.span / 2
        allPoints.append(CGPoint(x: tipX + tx, y: tipY + ty))
        allPoints.append(CGPoint(x: tipX - tx, y: tipY - ty))
    }

    let xs = allPoints.map(\.x), ys = allPoints.map(\.y)
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    let w = max(maxX - minX, 1), h = max(maxY - minY, 1)
    let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2

    let margin: CGFloat = 24
    let fitScale = min((size.width - margin * 2) / w,
                       (size.height - margin * 2) / h, 1.3)
    let fitOffsetX = size.width / 2 - cx * fitScale
    let fitOffsetY = size.height / 2 - cy * fitScale

    var ctx = context
    ctx.translateBy(x: fitOffsetX, y: fitOffsetY)
    ctx.scaleBy(x: fitScale, y: fitScale)

    // --- Layer 1: ligand skeleton ---
    drawSkeletal(context: ctx,
                 positions: scene.ligandPositions,
                 bonds: scene.ligandBonds,
                 atomicNums: scene.ligandAtomicNums,
                 rings: scene.ligandRings,
                 formalCharges: scene.ligandFormalCharges,
                 palette: palette,
                 halos: scene.ligandHalos)

    // --- Layer 2: hydrophobic eyelashes (LigPlot+ style) ---
    let hydroColor = palette.interaction(.hydrophobic)
    for mark in scene.hydrophobicMarks {
        drawHydrophobicEyelash(context: ctx, mark: mark, color: hydroColor)
    }

    // --- Layer 3: directional interaction lines ---
    for line in scene.directionalLines {
        drawDirectionalLine(context: ctx, line: line, palette: palette)
    }

    // --- Layer 4: residue side chains + bubbles (on top of lines) ---
    for residue in scene.residues {
        drawResidue(context: ctx, residue: residue, palette: palette)
    }

    // --- Compute screen-space residue centers for hover ---
    return scene.residues.map { r in
        (label: r.key.label,
         center: CGPoint(x: r.center.x * fitScale + fitOffsetX,
                         y: r.center.y * fitScale + fitOffsetY),
         interactions: r.interactions)
    }
}

// MARK: - Hydrophobic eyelash

private func drawHydrophobicEyelash(
    context: GraphicsContext, mark: DiagramScene.HydrophobicMark, color: Color
) {
    // Tangent direction (perpendicular to outward normal toward the residue)
    let tx = -mark.outwardNormal.y
    let ty = mark.outwardNormal.x

    let span = mark.span
    let teethCount = max(3, min(7, Int(span / 6)))
    let toothLen: CGFloat = 6
    let bulge: CGFloat = max(4, span * 0.12)

    // Quad-curve baseline cradling the cluster, opening toward the residue
    let arcLeft = CGPoint(x: mark.baseline.x - tx * span / 2,
                          y: mark.baseline.y - ty * span / 2)
    let arcRight = CGPoint(x: mark.baseline.x + tx * span / 2,
                           y: mark.baseline.y + ty * span / 2)
    let ctrl = CGPoint(x: mark.baseline.x + mark.outwardNormal.x * bulge,
                       y: mark.baseline.y + mark.outwardNormal.y * bulge)
    var arc = Path()
    arc.move(to: arcLeft)
    arc.addQuadCurve(to: arcRight, control: ctrl)
    context.stroke(arc, with: .color(color),
                   style: StrokeStyle(lineWidth: 2.2, lineCap: .round))

    // Teeth pointing outward from the baseline along the normal
    for i in 0..<teethCount {
        let t = teethCount == 1 ? CGFloat(0.5) : CGFloat(i) / CGFloat(teethCount - 1)
        let mt = 1 - t
        let bx = mt * mt * arcLeft.x + 2 * mt * t * ctrl.x + t * t * arcRight.x
        let by = mt * mt * arcLeft.y + 2 * mt * t * ctrl.y + t * t * arcRight.y
        let outer = CGPoint(x: bx + mark.outwardNormal.x * toothLen,
                            y: by + mark.outwardNormal.y * toothLen)
        var p = Path(); p.move(to: CGPoint(x: bx, y: by)); p.addLine(to: outer)
        context.stroke(p, with: .color(color),
                       style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
    }

    // Dashed connector from the outer tip of the eyelash bulge to the residue
    let tip = CGPoint(x: mark.baseline.x + mark.outwardNormal.x * (bulge + toothLen),
                      y: mark.baseline.y + mark.outwardNormal.y * (bulge + toothLen))
    let shortened = shortenSegment(from: tip, to: mark.residueCenter,
                                    fromInset: 0, toInset: 42)
    var path = Path(); path.move(to: shortened.0); path.addLine(to: shortened.1)
    context.stroke(path, with: .color(color.opacity(0.55)),
                   style: StrokeStyle(lineWidth: 1.4, dash: [4, 3]))
}

// MARK: - Directional line

private func drawDirectionalLine(context: GraphicsContext, line: DiagramScene.DirectionalLine,
                                 palette: DiagramPalette) {
    let color = palette.interaction(line.type)
    let lw: CGFloat = 2.0

    var path = Path()
    path.move(to: line.from)
    if let ctrl = line.routedVia {
        path.addQuadCurve(to: line.to, control: ctrl)
    } else {
        path.addLine(to: line.to)
    }

    let stroke: StrokeStyle
    switch line.type {
    case .hbond, .saltBridge, .piCation, .amideStack:
        stroke = StrokeStyle(lineWidth: lw, dash: [6, 4.2])
    case .metalCoord:
        stroke = StrokeStyle(lineWidth: 2.5)
    default:
        stroke = StrokeStyle(lineWidth: lw)
    }
    context.stroke(path, with: .color(color), style: stroke)

    // Distance label at the midpoint of the line (or the bezier midpoint)
    let labelPt: CGPoint
    if let ctrl = line.routedVia {
        labelPt = CGPoint(x: 0.25 * line.from.x + 0.5 * ctrl.x + 0.25 * line.to.x,
                          y: 0.25 * line.from.y + 0.5 * ctrl.y + 0.25 * line.to.y)
    } else {
        labelPt = CGPoint(x: (line.from.x + line.to.x) / 2,
                          y: (line.from.y + line.to.y) / 2)
    }
    let distStr = String(format: "%.1f", line.distance)
    let sz = NSAttributedString(string: distStr,
        attributes: [.font: NSFont.monospacedSystemFont(ofSize: NSFont.smallSystemFontSize - 1, weight: .medium)]).size()
    let pill = CGRect(x: labelPt.x - sz.width / 2 - 4,
                      y: labelPt.y - sz.height / 2 - 2,
                      width: sz.width + 8, height: sz.height + 4)
    context.fill(Path(roundedRect: pill, cornerRadius: DS.Radius.xs),
                 with: .color(palette.paper.opacity(0.92)))
    context.draw(context.resolve(
        Text(distStr).font(DS.Font.monoSmall.weight(.medium))
            .foregroundColor(palette.secondaryInk)), at: labelPt, anchor: .center)
}

// MARK: - Residue (side chain + bubble)

private func drawResidue(context: GraphicsContext, residue: DiagramScene.ResidueDraw,
                         palette: DiagramPalette) {
    // Side chain via the shared skeletal renderer
    if let tmpl = residue.template, !residue.scPositions.isEmpty,
       residue.scPositions.count == tmpl.atoms.count {
        let bonds: [(a: Int, b: Int, order: Int, isAromatic: Bool)] =
            tmpl.bonds.map { (a: $0.from, b: $0.to, order: $0.order, isAromatic: $0.isAromatic) }
        let atomicNums = tmpl.atoms.map { $0.element.rawValue }
        // Carboxylate: only the *deprotonated* oxygen carries the formal
        // charge. The double-bonded oxygen (OD1 / OE1 in our kekulized
        // template) stays neutral. This matches Druse's protonation model in
        // [Chemistry/Protonation.swift](Chemistry/Protonation.swift) which
        // assigns -1 to a single carboxylate oxygen at pH 7.4.
        let formalCharges: [Int] = tmpl.atoms.map { atom in
            switch (residue.key.name, atom.name) {
            case ("ASP", "OD2"), ("GLU", "OE2"):
                return -1
            case ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"):
                return +1
            default: return 0
            }
        }

        // Explicit hydrogen-count labels for heteroatoms (RDKit convention).
        // We list every donor/charged-N/S-H/O-H atom by name; everything else
        // falls back to the bare element symbol.
        var labelOverrides: [Int: String] = [:]
        for (i, atom) in tmpl.atoms.enumerated() {
            let label: String?
            switch (residue.key.name, atom.name) {
            case ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"):    label = "OH"
            case ("CYS", "SG"):                                    label = "SH"
            case ("LYS", "NZ"):                                    label = "NH\u{2083}"      // NH3 (charged)
            case ("ARG", "NH1"), ("ARG", "NH2"):                   label = "NH\u{2082}"      // NH2 (charged)
            case ("ARG", "NE"):                                    label = "NH"
            case ("ASN", "ND2"), ("GLN", "NE2"):                   label = "NH\u{2082}"      // amide
            case ("HIS", "ND1"), ("HIS", "NE2"):                   label = "NH"              // tautomer
            case ("TRP", "NE1"):                                   label = "NH"
            default:                                                label = nil
            }
            if let label { labelOverrides[i] = label }
        }

        // Atom-index set covered by any aromatic ring (used to skip per-atom
        // highlight rings on aromatic carbons — the π line already targets
        // the ring centroid, no need for a per-atom dot).
        let aromaticAtomIdx: Set<Int> = Set(tmpl.aromaticRings.flatMap { $0 })

        drawSkeletal(context: context,
                     positions: residue.scPositions,
                     bonds: bonds,
                     atomicNums: atomicNums,
                     rings: tmpl.aromaticRings,
                     formalCharges: formalCharges,
                     palette: palette,
                     labelOverrides: labelOverrides)

        // Per-atom highlight ring for interacting atoms — but skip aromatic
        // ring carbons (the π line shows the ring interaction by itself).
        let dominantColor = palette.interaction(residue.dominantType)
        for (i, atom) in tmpl.atoms.enumerated() where i < residue.scPositions.count {
            guard residue.interactingAtomNames.contains(atom.name) else { continue }
            if aromaticAtomIdx.contains(i) { continue }
            let pos = residue.scPositions[i]
            let r: CGFloat = atom.element == .C ? 7 : 11
            context.stroke(
                Path(ellipseIn: CGRect(x: pos.x - r, y: pos.y - r, width: r * 2, height: r * 2)),
                with: .color(dominantColor), lineWidth: 1.8)
        }

        // Connector line: bubble edge → CB. The bubble itself is drawn next
        // and will cover the inner end of this line.
        let cb = residue.scPositions[0]
        let cdx = cb.x - residue.center.x
        let cdy = cb.y - residue.center.y
        let clen = max(hypot(cdx, cdy), 1)
        var connector = Path()
        connector.move(to: residue.center)
        connector.addLine(to: CGPoint(x: residue.center.x + cdx / clen * (clen - 2),
                                      y: residue.center.y + cdy / clen * (clen - 2)))
        context.stroke(connector, with: .color(palette.ink.opacity(0.85)), lineWidth: Skeletal.bondWidth)
    }

    // Core bubble (residue label) drawn on top of side chain + connector
    let nameSize = NSAttributedString(string: residue.key.label,
        attributes: [.font: NSFont.systemFont(ofSize: 12, weight: .bold)]).size()
    let bubbleW: CGFloat = max(nameSize.width + 24, 72)
    let bubbleH: CGFloat = 38
    let rect = CGRect(x: residue.center.x - bubbleW / 2,
                      y: residue.center.y - bubbleH / 2,
                      width: bubbleW, height: bubbleH)

    let border = palette.residue(residue.residueClass)
    let bubble = Path(roundedRect: rect, cornerRadius: DS.Radius.m, style: .continuous)
    context.fill(Path(roundedRect: rect.offsetBy(dx: 1.5, dy: 1.5), cornerRadius: DS.Radius.m, style: .continuous),
                 with: .color(palette.shadow))
    // Opaque paper first so lines underneath are hidden, then the class tint.
    context.fill(bubble, with: .color(palette.paper))
    context.fill(bubble, with: .color(border.opacity(0.14)))
    context.stroke(bubble, with: .color(border), lineWidth: 2)

    context.draw(context.resolve(
        Text(residue.key.label).font(.callout.weight(.bold)).foregroundColor(palette.ink)),
        at: CGPoint(x: residue.center.x, y: residue.center.y - 4), anchor: .center)
    context.draw(context.resolve(
        Text(residue.dominantType.label).font(DS.Font.caption.weight(.medium))
            .foregroundColor(border)),
        at: CGPoint(x: residue.center.x, y: residue.center.y + 11), anchor: .center)
}

// MARK: - Pharmacophore halo

private func pharmacophoreHalo(forLigandAtomIndex index: Int,
                               interactions: [MolecularInteraction]) -> DiagramHalo? {
    // Per-atom halos for directional interactions only.
    // - π types are visualized via the line targeting the ring centroid.
    // - hydrophobic is visualized via the eyelash on the ligand side.
    let excluded = kPiTypes.union([.hydrophobic, .chPi])
    let types = interactions
        .filter { $0.ligandAtomIndex == index && !excluded.contains($0.type) }
        .map(\.type)
    guard !types.isEmpty else { return nil }
    if types.contains(.hbond) || types.contains(.halogen) { return .hbond }
    if types.contains(.saltBridge) { return .saltBridge }
    if types.contains(.metalCoord) { return .metal }
    if types.contains(.chalcogen) { return .chalcogen }
    return .other
}

// MARK: - Shared helpers

private func elementSymbol(_ atomicNum: Int) -> String {
    switch atomicNum {
    case 7:  "N"
    case 8:  "O"
    case 9:  "F"
    case 15: "P"
    case 16: "S"
    case 17: "Cl"
    case 35: "Br"
    case 53: "I"
    default: Element(rawValue: atomicNum)?.symbol ?? "?"
    }
}

func dominantInteractionType(_ interactions: [MolecularInteraction]) -> MolecularInteraction.InteractionType {
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

func shortenSegment(from p1: CGPoint, to p2: CGPoint,
                    fromInset: CGFloat, toInset: CGFloat) -> (CGPoint, CGPoint) {
    let dx = p2.x - p1.x, dy = p2.y - p1.y
    let len = sqrt(dx * dx + dy * dy)
    guard len > fromInset + toInset else { return (p1, p2) }
    let ux = dx / len, uy = dy / len
    return (CGPoint(x: p1.x + ux * fromInset, y: p1.y + uy * fromInset),
            CGPoint(x: p2.x - ux * toInset, y: p2.y - uy * toInset))
}

/// Distance from `origin` along unit `direction` at which the ray first exits
/// `box`. If origin is inside the box, this is the distance to the closest
/// boundary along the direction. Returns 0 if the direction is degenerate.
func rayBoxExitDistance(origin: CGPoint, direction: CGPoint, box: CGRect) -> CGFloat {
    var tx = CGFloat.greatestFiniteMagnitude
    var ty = CGFloat.greatestFiniteMagnitude
    if direction.x > 1e-6 {
        tx = (box.maxX - origin.x) / direction.x
    } else if direction.x < -1e-6 {
        tx = (box.minX - origin.x) / direction.x
    }
    if direction.y > 1e-6 {
        ty = (box.maxY - origin.y) / direction.y
    } else if direction.y < -1e-6 {
        ty = (box.minY - origin.y) / direction.y
    }
    return max(0, min(tx, ty))
}

func lineCrossesRect(from p1: CGPoint, to p2: CGPoint, rect: CGRect) -> Bool {
    if rect.contains(p1) && rect.contains(p2) { return false }
    if rect.contains(p1) || rect.contains(p2) { return false }
    let edges: [(CGPoint, CGPoint)] = [
        (CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY)),
        (CGPoint(x: rect.maxX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.maxY)),
        (CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.maxY)),
        (CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.minY)),
    ]
    var crossCount = 0
    for (e1, e2) in edges where segmentsIntersect(p1, p2, e1, e2) { crossCount += 1 }
    return crossCount >= 2
}

private func segmentsIntersect(_ a1: CGPoint, _ a2: CGPoint,
                                _ b1: CGPoint, _ b2: CGPoint) -> Bool {
    func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    let d1 = cross(b1, b2, a1), d2 = cross(b1, b2, a2)
    let d3 = cross(a1, a2, b1), d4 = cross(a1, a2, b2)
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
           ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
}

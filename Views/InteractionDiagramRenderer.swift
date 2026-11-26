// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import simd

// MARK: - Residue Key

struct ResidueKey: Hashable, Sendable {
    let name: String
    let seq: Int
    let chain: String

    /// Publication-style label: "Asp86" for amino acids, the raw name otherwise
    /// ("HOH301"), with the chain appended ("Asp86(A)") when several chains are shown.
    func displayLabel(withChain: Bool) -> String {
        let base = sideChainTemplates[canonicalResidueName(name)] != nil
            ? String(name.prefix(1)) + name.dropFirst().lowercased()
            : name
        let chainSuffix = withChain && !chain.isEmpty ? "(\(chain))" : ""
        return "\(base)\(seq)\(chainSuffix)"
    }
}

// MARK: - DiagramScene
//
// All geometry needed to draw the interaction diagram, computed once (off the
// main thread) from the 2D depiction and the interactions. Coordinates are in
// an abstract diagram space; drawing applies a fit transform for the canvas or
// a fixed offset for exports, so the scene is canvas-size independent.

struct DiagramBond: Sendable {
    let a: Int
    let b: Int
    let order: Int
    let isAromatic: Bool
}

/// A skeletal 2D molecule (the ligand or one residue side chain).
struct DiagramMolecule: Sendable {
    let positions: [CGPoint]
    let bonds: [DiagramBond]
    let atomicNums: [Int]
    let formalCharges: [Int]
    /// Implicit hydrogens drawn next to heteroatom labels ("OH", "NH₂").
    let hydrogens: [Int]
    /// Every 5/6-membered ring: places the inner line of ring double bonds.
    let rings: [[Int]]
    /// Aromatic rings, drawn with an inscribed circle.
    let aromaticRings: [[Int]]
}

struct DiagramScene: Sendable {
    let ligand: DiagramMolecule
    /// Ligand atom → interaction type of its directional contact (soft halo).
    let ligandHalos: [Int: MolecularInteraction.InteractionType]
    let residues: [Residue]
    let lines: [Line]
    let hydrophobicMarks: [HydrophobicMark]
    /// Extent of everything drawn, in diagram space.
    let bounds: CGRect

    enum ResidueKind: Sendable {
        /// Residue label with its side chain drawn toward the ligand.
        case sideChain
        /// Residue label only (backbone or hydrophobic contacts, non-standard residues).
        case label
        /// Metal ion, drawn as a sphere with its element symbol.
        case metal(symbol: String)
    }

    struct Residue: Sendable {
        let key: ResidueKey
        let label: String
        let center: CGPoint
        let labelSize: CGSize
        let kind: ResidueKind
        let residueClass: DiagramResidueClass
        let sideChain: DiagramMolecule?
        /// Side chain atom index → type of the contact it makes (highlight ring).
        let contactAtoms: [Int: MolecularInteraction.InteractionType]
        let backboneTags: [BackboneTag]
        /// Every interaction of this residue, including those not drawn as lines.
        let interactions: [MolecularInteraction]

        var bubbleRect: CGRect {
            CGRect(x: center.x - labelSize.width / 2, y: center.y - labelSize.height / 2,
                   width: labelSize.width, height: labelSize.height)
        }

        func contains(_ p: CGPoint) -> Bool {
            if bubbleRect.insetBy(dx: -4, dy: -4).contains(p) { return true }
            return sideChain?.positions.contains { hypot($0.x - p.x, $0.y - p.y) < 12 } ?? false
        }
    }

    /// Backbone atom (amide N, carbonyl O) attached to a residue label.
    struct BackboneTag: Sendable {
        let edge: CGPoint
        let position: CGPoint
        let atomicNum: Int
    }

    struct Line: Sendable {
        let from: CGPoint
        let to: CGPoint
        /// Quadratic control point when the line bends around the ligand.
        let control: CGPoint?
        let type: MolecularInteraction.InteractionType
        let label: String
        let labelCenter: CGPoint
        let labelSize: CGSize
        let residue: Int
        let fromCentroid: Bool
        let toCentroid: Bool
    }

    /// LigPlot-style "eyelash": an arc with teeth facing the contact partner.
    struct HydrophobicMark: Sendable {
        let baseline: CGPoint
        let normal: CGPoint
        let span: CGFloat
        let residue: Int
        /// Where the dashed connector from this mark's tip ends (nil: no connector).
        let connectorEnd: CGPoint?

        static let toothLength: CGFloat = 6
        var bulge: CGFloat { max(4, span * 0.12) }
        var tip: CGPoint {
            let d = bulge + Self.toothLength
            return CGPoint(x: baseline.x + normal.x * d, y: baseline.y + normal.y * d)
        }
        var boundingRect: CGRect {
            let r = span / 2 + Self.toothLength
            return CGRect(x: baseline.x - r, y: baseline.y - r, width: r * 2, height: r * 2)
        }
    }

    /// Interaction types drawn in the scene, in canonical order.
    var drawnTypes: [MolecularInteraction.InteractionType] {
        var present = Set(lines.map(\.type))
        if !hydrophobicMarks.isEmpty { present.insert(.hydrophobic) }
        return MolecularInteraction.InteractionType.allCases.filter { present.contains($0) }
    }

    /// Number of drawn contacts per type: lines, or residues for hydrophobic contacts.
    func drawnCount(_ type: MolecularInteraction.InteractionType) -> Int {
        type == .hydrophobic
            ? Set(hydrophobicMarks.map(\.residue)).count
            : lines.filter { $0.type == type }.count
    }

    var residueClasses: [DiagramResidueClass] {
        let present = Set(residues.compactMap { r -> DiagramResidueClass? in
            if case .metal = r.kind { return nil }
            return r.residueClass
        })
        return DiagramResidueClass.allCases.filter { present.contains($0) }
    }
}

// MARK: - Constants

private let kPiTypes: Set<MolecularInteraction.InteractionType> = [.piStack, .piCation, .chPi, .amideStack]
private let kDiagramHydroExcluded: Set<String> = ["ASP", "GLU", "ASN", "GLN", "SER", "THR", "CYS"]
private let kDiagramHydroAllowed: [String: Set<String>] = [
    "ARG": ["CB", "CG"],
    "LYS": ["CB", "CG", "CD"],
]
private let kBackboneAtoms: Set<String> = ["N", "CA", "C", "O", "OXT"]
private let kMetalAtomicNums: Set<Int> = [3, 11, 12, 19, 20, 25, 26, 27, 28, 29, 30]

/// Drawing precedence when one atom pair carries several types.
private let kInteractionPriority: [MolecularInteraction.InteractionType] =
    [.saltBridge, .metalCoord, .hbond, .halogen, .chalcogen, .piCation, .piStack, .amideStack, .chPi, .hydrophobic]

private func priority(_ type: MolecularInteraction.InteractionType) -> Int {
    kInteractionPriority.firstIndex(of: type) ?? kInteractionPriority.count
}

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

/// Implicit hydrogens on N, O and S from the default valence, bond orders and
/// formal charge (other elements are drawn without hydrogens).
private func implicitHydrogens(atomicNum: Int, bondOrderSum: Int, charge: Int) -> Int {
    switch atomicNum {
    case 7:  return max(0, 3 + charge - bondOrderSum)
    case 8:  return max(0, 2 + charge - bondOrderSum)
    case 16: return bondOrderSum <= 2 ? max(0, 2 + charge - bondOrderSum) : 0
    default: return 0
    }
}

/// Hydrogens and formal charge of side chain heteroatoms by protonation state.
/// Plain "HIS" carries no tautomer information, so neither ring N gets an H.
private func sideChainProtonation(residue: String, atom: String) -> (hydrogens: Int, charge: Int) {
    switch (residue, atom) {
    case ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
         ("ASH", "OD2"), ("GLH", "OE2"), ("ARG", "NE"), ("TRP", "NE1"),
         ("HID", "ND1"), ("HSD", "ND1"), ("HIE", "NE2"), ("HSE", "NE2"),
         ("HIP", "ND1"), ("HSP", "ND1"):
        return (1, 0)
    case ("ASP", "OD2"), ("GLU", "OE2"):
        return (0, -1)
    case ("LYS", "NZ"):
        return (3, 1)
    case ("ARG", "NH1"), ("HIP", "NE2"), ("HSP", "NE2"):
        return (atom == "NH1" ? 2 : 1, 1)
    case ("LYN", "NZ"), ("ARG", "NH2"), ("ASN", "ND2"), ("GLN", "NE2"):
        return (2, 0)
    default:
        return (0, 0)
    }
}

private func centroid(_ points: [CGPoint]) -> CGPoint {
    guard !points.isEmpty else { return .zero }
    let n = CGFloat(points.count)
    return CGPoint(x: points.map(\.x).reduce(0, +) / n, y: points.map(\.y).reduce(0, +) / n)
}

private func centroid3(_ points: [SIMD3<Float>]) -> SIMD3<Float>? {
    guard !points.isEmpty else { return nil }
    return points.reduce(.zero, +) / Float(points.count)
}

private func rect(around p: CGPoint, size: CGSize) -> CGRect {
    CGRect(x: p.x - size.width / 2, y: p.y - size.height / 2, width: size.width, height: size.height)
}

/// Compute a complete diagram scene. Pure function, safe off the main thread.
/// - Parameter aligned: the depiction's atom order matches `ligandAtoms` (see
///   `alignDepiction`); only then are formal charges and 3D positions trusted.
func buildDiagramScene(
    coords: RDKitBridge.Coords2D,
    aligned: Bool,
    interactions: [MolecularInteraction],
    ligandAtoms: [Atom],
    proteinAtoms: [Atom]
) -> DiagramScene? {
    let n = coords.positions.count
    guard n > 0, coords.atomicNums.count == n, coords.isAromatic.count == n else { return nil }
    let useLigand3D = aligned && ligandAtoms.count >= n

    // --- Ligand: normalize RDKit coordinates to the shared bond length ---
    let validBonds = coords.bonds.filter { $0.0 < n && $0.1 < n }
    let bondLengths = validBonds
        .map { hypot(coords.positions[$0.0].x - coords.positions[$0.1].x,
                     coords.positions[$0.0].y - coords.positions[$0.1].y) }
        .filter { $0 > 1e-3 }
        .sorted()
    let medianBond = bondLengths.isEmpty ? 1.5 : bondLengths[bondLengths.count / 2]
    let scale = kDiagramBondLength / medianBond
    let xs = coords.positions.map(\.x), ys = coords.positions.map(\.y)
    let midX = ((xs.min() ?? 0) + (xs.max() ?? 0)) / 2
    let midY = ((ys.min() ?? 0) + (ys.max() ?? 0)) / 2
    let positions = coords.positions.map {
        CGPoint(x: ($0.x - midX) * scale, y: -($0.y - midY) * scale)
    }
    let ligandCenter = centroid(positions)

    let aromaticRings = detectAromaticRings(bonds: validBonds, atomCount: n, isAromatic: coords.isAromatic)
    let allRings = detectAromaticRings(bonds: validBonds, atomCount: n,
                                       isAromatic: Array(repeating: true, count: n))
    var bondOrderSum = Array(repeating: 0, count: n)
    for (a, b, order) in validBonds { bondOrderSum[a] += order; bondOrderSum[b] += order }
    let charges = (0..<n).map { useLigand3D ? ligandAtoms[$0].formalCharge : 0 }
    let ligand = DiagramMolecule(
        positions: positions,
        bonds: validBonds.map { DiagramBond(a: $0.0, b: $0.1, order: $0.2,
                                            isAromatic: coords.isAromatic[$0.0] && coords.isAromatic[$0.1]) },
        atomicNums: coords.atomicNums,
        formalCharges: charges,
        hydrogens: (0..<n).map { implicitHydrogens(atomicNum: coords.atomicNums[$0],
                                                   bondOrderSum: bondOrderSum[$0], charge: charges[$0]) },
        rings: allRings,
        aromaticRings: aromaticRings)

    let ring2D = aromaticRings.map { centroid($0.map { positions[$0] }) }
    let ring3D: [SIMD3<Float>?] = aromaticRings.map { ring in
        useLigand3D ? centroid3(ring.map { ligandAtoms[$0].position }) : nil
    }

    /// Ligand end of a contact: the atom itself, or a ring centroid when the
    /// detector reported one (matched in 3D; π types fall back to the atom's ring).
    func ligandContact(_ ixn: MolecularInteraction) -> (point: CGPoint, centroid: Bool)? {
        let idx = ixn.ligandAtomIndex
        guard idx < n else { return nil }
        if useLigand3D {
            if simd_distance(ligandAtoms[idx].position, ixn.ligandPosition) < 0.3 {
                return (positions[idx], false)
            }
            var best: (index: Int, distance: Float)?
            for (r, c) in ring3D.enumerated() {
                guard let c else { continue }
                let d = simd_distance(c, ixn.ligandPosition)
                if d < (best?.distance ?? 0.6) { best = (r, d) }
            }
            if let best { return (ring2D[best.index], true) }
        }
        if kPiTypes.contains(ixn.type), let r = aromaticRings.firstIndex(where: { $0.contains(idx) }) {
            return (ring2D[r], true)
        }
        return (positions[idx], false)
    }

    // --- Group interactions by residue (with the hydrophobic display filter) ---
    var groups: [ResidueKey: [MolecularInteraction]] = [:]
    var residueOrder: [ResidueKey] = []
    for ixn in interactions {
        guard ixn.proteinAtomIndex < proteinAtoms.count, ixn.ligandAtomIndex < n else { continue }
        let atom = proteinAtoms[ixn.proteinAtomIndex]
        if ixn.type == .hydrophobic {
            let resName = canonicalResidueName(atom.residueName)
            let atomName = atom.name.trimmingCharacters(in: .whitespaces)
            if kDiagramHydroExcluded.contains(resName) { continue }
            if let allowed = kDiagramHydroAllowed[resName], !allowed.contains(atomName) { continue }
        }
        let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
        if groups[key] == nil { residueOrder.append(key) }
        groups[key, default: []].append(ixn)
    }
    let showsChains = Set(residueOrder.map(\.chain)).count > 1

    // Atom positions of every contacting residue, by atom name (ring matching in 3D).
    var residueAtomPositions: [ResidueKey: [String: SIMD3<Float>]] = [:]
    for atom in proteinAtoms {
        let key = ResidueKey(name: atom.residueName, seq: atom.residueSeq, chain: atom.chainID)
        guard groups[key] != nil else { continue }
        residueAtomPositions[key, default: [:]][atom.name.trimmingCharacters(in: .whitespaces)] = atom.position
    }

    func atomName(_ ixn: MolecularInteraction) -> String {
        proteinAtoms[ixn.proteinAtomIndex].name.trimmingCharacters(in: .whitespaces)
    }

    /// Template ring the protein end of a contact refers to: matched against the
    /// reported 3D position, else (π types) the ring containing the atom.
    func proteinRing(_ ixn: MolecularInteraction, key: ResidueKey, template: SideChainTemplate) -> Int? {
        if let named = residueAtomPositions[key] {
            var best: (index: Int, distance: Float)?
            for (r, ring) in template.aromaticRings.enumerated() {
                guard let c = centroid3(ring.compactMap { named[template.atoms[$0].name] }) else { continue }
                let d = simd_distance(c, ixn.proteinPosition)
                if d < (best?.distance ?? 0.6) { best = (r, d) }
            }
            if let best { return best.index }
        }
        guard kPiTypes.contains(ixn.type) else { return nil }
        let name = atomName(ixn)
        return template.aromaticRings.firstIndex { ring in ring.contains { template.atoms[$0].name == name } }
    }

    // --- Residue placement inputs ---
    struct Pending {
        let key: ResidueKey
        let ixns: [MolecularInteraction]
        let kind: DiagramScene.ResidueKind
        let template: SideChainTemplate?
        let input: DiagramPlacementInput
        let label: String
        let labelSize: CGSize
    }
    var pending: [Pending] = []
    for key in residueOrder {
        guard let ixns = groups[key] else { continue }
        let directional = ixns.filter { $0.type != .hydrophobic }
        let template = sideChainTemplates[canonicalResidueName(key.name)]
        let label = key.displayLabel(withChain: showsChains)

        let metalAtom = template == nil
            ? ixns.lazy.map { proteinAtoms[$0.proteinAtomIndex] }.first { kMetalAtomicNums.contains($0.element.rawValue) }
            : nil
        let sideChainContacts = directional.filter { ixn in
            template?.atoms.contains { $0.name == atomName(ixn) } ?? false
        }
        let kind: DiagramScene.ResidueKind
        if let metalAtom {
            kind = .metal(symbol: metalAtom.element.symbol)
        } else if let template, !template.atoms.isEmpty, !sideChainContacts.isEmpty {
            kind = .sideChain
        } else {
            kind = .label
        }

        // Orient the side chain so its most important contact atom faces the ligand.
        var referenceAtom: Int?
        if case .sideChain = kind, let template,
           let best = sideChainContacts.min(by: { priority($0.type) < priority($1.type) }) {
            if let r = proteinRing(best, key: key, template: template) {
                let cb = template.atoms[0].offset
                referenceAtom = template.aromaticRings[r].max { a, b in
                    hypot(template.atoms[a].offset.x - cb.x, template.atoms[a].offset.y - cb.y)
                        < hypot(template.atoms[b].offset.x - cb.x, template.atoms[b].offset.y - cb.y)
                }
            } else {
                referenceAtom = template.atoms.firstIndex { $0.name == atomName(best) }
            }
        }

        // Anchor: inverse-distance weighted ligand contact point.
        let anchorSource = directional.isEmpty ? ixns : directional
        var sx: CGFloat = 0, sy: CGFloat = 0, sw: CGFloat = 0
        for ixn in anchorSource {
            guard let c = ligandContact(ixn) else { continue }
            let w = 1 / (1 + CGFloat(ixn.distance))
            sx += c.point.x * w; sy += c.point.y * w; sw += w
        }
        let anchor = sw > 0 ? CGPoint(x: sx / sw, y: sy / sw) : ligandCenter

        let labelSize: CGSize
        let labelRadius: CGFloat
        let reach: CGFloat
        switch kind {
        case .metal:
            labelSize = CGSize(width: 30, height: 30)
            labelRadius = 16
            reach = 15 + 56
        case .sideChain:
            let text = DiagramFont.residue.measure(label)
            labelSize = CGSize(width: ceil(text.width) + 16, height: ceil(text.height) + 6)
            labelRadius = labelSize.width / 2 + 2
            let cb = template?.atoms[0].offset ?? .zero
            let extent = (template?.atoms ?? []).enumerated()
                .filter { referenceAtom == nil || $0.offset == referenceAtom }
                .map { hypot($0.element.offset.x - cb.x, $0.element.offset.y - cb.y) }
                .max() ?? 0
            reach = kCBAnchorDistance + extent + 60
        case .label:
            let text = DiagramFont.residue.measure(label)
            labelSize = CGSize(width: ceil(text.width) + 16, height: ceil(text.height) + 6)
            labelRadius = labelSize.width / 2 + 2
            reach = labelSize.width / 2 + (directional.isEmpty ? 54 : 66)
        }
        let isSideChain: Bool = { if case .sideChain = kind { return true }; return false }()
        pending.append(Pending(
            key: key, ixns: ixns, kind: kind, template: template,
            input: DiagramPlacementInput(anchor: anchor,
                                         template: isSideChain ? template : nil,
                                         referenceAtom: referenceAtom,
                                         labelRadius: labelRadius, reach: reach),
            label: label, labelSize: labelSize))
    }

    let centers = placeResidues(pending.map(\.input), ligand: positions, ligandCenter: ligandCenter)

    // --- Residues, lines, halos ---
    struct RawLine {
        let from: CGPoint, to: CGPoint
        let type: MolecularInteraction.InteractionType
        let distance: Float
        let residue: Int
        let fromCentroid: Bool, toCentroid: Bool
    }
    var residues: [DiagramScene.Residue] = []
    var rawLines: [RawLine] = []
    var halos: [Int: MolecularInteraction.InteractionType] = [:]

    for (i, p) in pending.enumerated() {
        let center = centers[i]
        let bubble = rect(around: center, size: p.labelSize)
        func edgePoint(toward target: CGPoint) -> (point: CGPoint, dir: CGPoint) {
            var dir = unit(CGPoint(x: target.x - center.x, y: target.y - center.y))
            if dir == .zero { dir = CGPoint(x: 0, y: -1) }
            let d: CGFloat
            if case .metal = p.kind { d = 15 } else { d = rayBoxExitDistance(origin: center, direction: dir, box: bubble) }
            return (CGPoint(x: center.x + dir.x * d, y: center.y + dir.y * d), dir)
        }

        var sideChain: DiagramMolecule?
        var scPositions: [CGPoint] = []
        if case .sideChain = p.kind, let template = p.template {
            scPositions = transformSideChain(template: template, bubbleCenter: center,
                                             towardLigand: p.input.anchor,
                                             referenceAtom: p.input.referenceAtom)
            let protonation = template.atoms.map { sideChainProtonation(residue: p.key.name, atom: $0.name) }
            sideChain = DiagramMolecule(
                positions: scPositions,
                bonds: template.bonds.map { DiagramBond(a: $0.from, b: $0.to, order: $0.order, isAromatic: $0.isAromatic) },
                atomicNums: template.atoms.map { $0.element.rawValue },
                formalCharges: protonation.map(\.charge),
                hydrogens: protonation.map(\.hydrogens),
                rings: template.aromaticRings,
                aromaticRings: template.aromaticRings)
        }

        let directional = p.ixns.filter { $0.type != .hydrophobic }

        // Backbone N/O tags: one per atom, facing the mean of its ligand contacts.
        var backboneContacts: [String: [CGPoint]] = [:]
        var backboneElement: [String: Int] = [:]
        for ixn in directional {
            if case .metal = p.kind { break }
            let name = atomName(ixn)
            if sideChain != nil, p.template?.atoms.contains(where: { $0.name == name }) == true { continue }
            let element = proteinAtoms[ixn.proteinAtomIndex].element.rawValue
            guard kBackboneAtoms.contains(name), element == 7 || element == 8,
                  let c = ligandContact(ixn) else { continue }
            backboneContacts[name, default: []].append(c.point)
            backboneElement[name] = element
        }
        var tagPosition: [String: CGPoint] = [:]
        var tags: [DiagramScene.BackboneTag] = []
        for name in backboneContacts.keys.sorted() {
            let e = edgePoint(toward: centroid(backboneContacts[name] ?? []))
            let pos = CGPoint(x: e.point.x + e.dir.x * 14, y: e.point.y + e.dir.y * 14)
            tagPosition[name] = pos
            tags.append(.init(edge: e.point, position: pos, atomicNum: backboneElement[name] ?? 8))
        }

        var contactAtoms: [Int: MolecularInteraction.InteractionType] = [:]
        for ixn in directional {
            guard let lig = ligandContact(ixn) else { continue }
            let name = atomName(ixn)
            var to: CGPoint
            var toCentroid = false
            var toInset: CGFloat = 0
            if case .metal = p.kind {
                to = center; toInset = 17
            } else if let template = p.template, sideChain != nil,
                      let r = proteinRing(ixn, key: p.key, template: template) {
                to = centroid(template.aromaticRings[r].map { scPositions[$0] })
                toCentroid = true
            } else if let template = p.template, sideChain != nil,
                      let t = template.atoms.firstIndex(where: { $0.name == name }) {
                to = scPositions[t]
                toInset = template.atoms[t].element == .C ? 5 : 9
                if contactAtoms[t].map({ priority(ixn.type) < priority($0) }) ?? true {
                    contactAtoms[t] = ixn.type
                }
            } else if let tag = tagPosition[name] {
                to = tag; toInset = 8
            } else {
                to = edgePoint(toward: lig.point).point; toInset = 2
            }
            let fromInset: CGFloat = lig.centroid ? 0 : (coords.atomicNums[ixn.ligandAtomIndex] == 6 ? 4 : 9)
            let s = shortenSegment(from: lig.point, to: to, fromInset: fromInset, toInset: toInset)
            rawLines.append(RawLine(from: s.0, to: s.1, type: ixn.type, distance: ixn.distance,
                                    residue: i, fromCentroid: lig.centroid, toCentroid: toCentroid))
            if !lig.centroid {
                let idx = ixn.ligandAtomIndex
                if halos[idx].map({ priority(ixn.type) < priority($0) }) ?? true {
                    halos[idx] = ixn.type
                }
            }
        }

        residues.append(DiagramScene.Residue(
            key: p.key, label: p.label, center: center, labelSize: p.labelSize, kind: p.kind,
            residueClass: DiagramResidueClass(residueName: p.key.name),
            sideChain: sideChain, contactAtoms: contactAtoms, backboneTags: tags,
            interactions: p.ixns))
    }

    // One line per atom pair: when several types join the same two points
    // (e.g. a salt bridge that is also an H-bond) keep the highest-priority one.
    var lineIndex: [String: Int] = [:]
    var dedupedLines: [RawLine] = []
    for line in rawLines {
        let key = "\(Int(line.from.x.rounded()))_\(Int(line.from.y.rounded()))_\(Int(line.to.x.rounded()))_\(Int(line.to.y.rounded()))"
        if let existing = lineIndex[key] {
            let old = dedupedLines[existing]
            if priority(line.type) < priority(old.type)
                || (line.type == old.type && line.distance < old.distance) {
                dedupedLines[existing] = line
            }
        } else {
            lineIndex[key] = dedupedLines.count
            dedupedLines.append(line)
        }
    }

    // --- Hydrophobic eyelashes ---
    //
    // A residue's hydrophobic contacts are grouped by ligand connectivity, so a
    // whole ring or alkyl segment becomes one eyelash facing the residue. A
    // label-only residue gets a matching eyelash on its label (LigPlot style);
    // a thin dashed connector links the two.
    var ligandAdj: [[Int]] = Array(repeating: [], count: n)
    for (a, b, _) in validBonds { ligandAdj[a].append(b); ligandAdj[b].append(a) }
    var ringIndexForAtom: [Int: Int] = [:]
    for (r, ring) in aromaticRings.enumerated() { for a in ring { ringIndexForAtom[a] = r } }

    var marks: [DiagramScene.HydrophobicMark] = []
    for (i, residue) in residues.enumerated() {
        let contacts = Set(residue.interactions.filter { $0.type == .hydrophobic }.map(\.ligandAtomIndex))
        guard !contacts.isEmpty else { continue }

        var residueMarkTip: CGPoint?
        if case .label = residue.kind, residue.interactions.allSatisfy({ $0.type == .hydrophobic }) {
            let anchor = pending[i].input.anchor
            var dir = unit(CGPoint(x: anchor.x - residue.center.x, y: anchor.y - residue.center.y))
            if dir == .zero { dir = CGPoint(x: 0, y: -1) }
            let d = rayBoxExitDistance(origin: residue.center, direction: dir, box: residue.bubbleRect) + 3
            let mark = DiagramScene.HydrophobicMark(
                baseline: CGPoint(x: residue.center.x + dir.x * d, y: residue.center.y + dir.y * d),
                normal: dir, span: min(residue.labelSize.width * 0.7, 36), residue: i, connectorEnd: nil)
            marks.append(mark)
            residueMarkTip = mark.tip
        }

        var visited = Set<Int>()
        for start in contacts.sorted() where !visited.contains(start) {
            var component: [Int] = []
            var queue = [start]
            visited.insert(start)
            while let atom = queue.popLast() {
                component.append(atom)
                for nb in ligandAdj[atom] where contacts.contains(nb) && !visited.contains(nb) {
                    visited.insert(nb)
                    queue.append(nb)
                }
            }

            let ringIndices = Set(component.compactMap { ringIndexForAtom[$0] })
            let members = (ringIndices.count == 1 ? aromaticRings[ringIndices.first!] : component).map { positions[$0] }
            let anchor = centroid(members)
            let radius = members.map { hypot($0.x - anchor.x, $0.y - anchor.y) }.max() ?? 0

            let normal = unit(CGPoint(x: residue.center.x - anchor.x, y: residue.center.y - anchor.y))
            var baseDist = radius + 13
            func clear(_ d: CGFloat) -> Bool {
                let b = CGPoint(x: anchor.x + normal.x * d, y: anchor.y + normal.y * d)
                return !positions.contains { hypot($0.x - b.x, $0.y - b.y) < 12 }
            }
            while !clear(baseDist) && baseDist < radius + 40 { baseDist += 4 }
            let baseline = CGPoint(x: anchor.x + normal.x * baseDist, y: anchor.y + normal.y * baseDist)
            let tipGuess = CGPoint(x: baseline.x + normal.x * 12, y: baseline.y + normal.y * 12)
            let end = residueMarkTip ?? {
                let dir = unit(CGPoint(x: tipGuess.x - residue.center.x, y: tipGuess.y - residue.center.y))
                if case .metal = residue.kind {
                    return CGPoint(x: residue.center.x + dir.x * 16, y: residue.center.y + dir.y * 16)
                }
                let d = rayBoxExitDistance(origin: residue.center, direction: dir, box: residue.bubbleRect) + 2
                return CGPoint(x: residue.center.x + dir.x * d, y: residue.center.y + dir.y * d)
            }()
            marks.append(DiagramScene.HydrophobicMark(
                baseline: baseline, normal: normal, span: max(18, radius * 1.4),
                residue: i, connectorEnd: end))
        }
    }

    // --- Distance labels, placed clear of atoms, labels and marks ---
    var obstacles: [CGRect] = []
    for (idx, p) in positions.enumerated() {
        let s: CGFloat = coords.atomicNums[idx] == 6 ? 10 : 18
        obstacles.append(rect(around: p, size: CGSize(width: s, height: s)))
    }
    for residue in residues {
        obstacles.append(residue.bubbleRect)
        if let sc = residue.sideChain {
            for (idx, p) in sc.positions.enumerated() {
                let s: CGFloat = sc.atomicNums[idx] == 6 ? 10 : 18
                obstacles.append(rect(around: p, size: CGSize(width: s, height: s)))
            }
        }
        for tag in residue.backboneTags { obstacles.append(rect(around: tag.position, size: CGSize(width: 16, height: 16))) }
    }
    obstacles += marks.map(\.boundingRect)

    let controls = dedupedLines.map { connectorBend(from: $0.from, to: $0.to, ligand: positions) }
    let labelTexts = dedupedLines.map { String(format: "%.1f Å", $0.distance) }
    let labelSizes = labelTexts.map {
        let s = DiagramFont.distance.measure($0)
        return CGSize(width: ceil(s.width) + 6, height: ceil(s.height) + 2)
    }
    let labelCenters = placeConnectorLabels(
        paths: dedupedLines.indices.map { (dedupedLines[$0].from, dedupedLines[$0].to, controls[$0]) },
        sizes: labelSizes, obstacles: obstacles)

    let lines = dedupedLines.indices.map { k in
        let l = dedupedLines[k]
        return DiagramScene.Line(
            from: l.from, to: l.to, control: controls[k], type: l.type,
            label: labelTexts[k], labelCenter: labelCenters[k], labelSize: labelSizes[k],
            residue: l.residue, fromCentroid: l.fromCentroid, toCentroid: l.toCentroid)
    }

    // --- Bounds ---
    var bounds = obstacles.reduce(CGRect.null) { $0.union($1) }
    for line in lines {
        bounds = bounds.union(rect(around: line.labelCenter, size: line.labelSize))
        if line.control != nil {
            let mid = segmentPoint(from: line.from, to: line.to, control: line.control, t: 0.5)
            bounds = bounds.union(rect(around: mid, size: CGSize(width: 4, height: 4)))
        }
    }
    bounds = bounds.insetBy(dx: -10, dy: -10)

    return DiagramScene(ligand: ligand, ligandHalos: halos, residues: residues, lines: lines,
                        hydrophobicMarks: marks, bounds: bounds)
}

// MARK: - Fonts

/// A font usable both on the Canvas (SwiftUI) and in SVG export, with AppKit
/// metrics for layout.
struct DiagramFont: Sendable {
    enum Weight: Sendable { case regular, medium, semibold, bold }
    let size: CGFloat
    let weight: Weight

    static let atom = DiagramFont(size: 13, weight: .semibold)
    static let charge = DiagramFont(size: 10, weight: .bold)
    static let residue = DiagramFont(size: 12, weight: .semibold)
    static let distance = DiagramFont(size: 10, weight: .medium)
    static let legend = DiagramFont(size: 11, weight: .regular)

    var nsFont: NSFont {
        let w: NSFont.Weight = switch weight {
        case .regular: .regular
        case .medium: .medium
        case .semibold: .semibold
        case .bold: .bold
        }
        return NSFont.systemFont(ofSize: size, weight: w)
    }

    var font: Font {
        let w: Font.Weight = switch weight {
        case .regular: .regular
        case .medium: .medium
        case .semibold: .semibold
        case .bold: .bold
        }
        return .system(size: size, weight: w)
    }

    var cssWeight: Int {
        switch weight {
        case .regular: 400
        case .medium: 500
        case .semibold: 600
        case .bold: 700
        }
    }

    func measure(_ string: String) -> CGSize {
        NSAttributedString(string: string, attributes: [.font: nsFont]).size()
    }
}

// MARK: - Drawing surfaces
//
// The diagram is drawn through a tiny vector API so the same code renders on
// screen (Canvas), to PNG/PDF (ImageRenderer of a Canvas) and to SVG.

protocol DiagramSurface: AnyObject {
    func concatenate(_ transform: CGAffineTransform)
    func fill(_ path: Path, _ color: Color)
    func stroke(_ path: Path, _ color: Color, _ style: StrokeStyle)
    func text(_ string: String, font: DiagramFont, color: Color, at point: CGPoint, anchor: UnitPoint)
}

final class CanvasDiagramSurface: DiagramSurface {
    private var context: GraphicsContext

    init(_ context: GraphicsContext) { self.context = context }

    func concatenate(_ transform: CGAffineTransform) { context.concatenate(transform) }
    func fill(_ path: Path, _ color: Color) { context.fill(path, with: .color(color)) }
    func stroke(_ path: Path, _ color: Color, _ style: StrokeStyle) {
        context.stroke(path, with: .color(color), style: style)
    }
    func text(_ string: String, font: DiagramFont, color: Color, at point: CGPoint, anchor: UnitPoint) {
        context.draw(Text(string).font(font.font).foregroundStyle(color), at: point, anchor: anchor)
    }
}

/// Writes the diagram as a standalone SVG document.
final class SVGDiagramSurface: DiagramSurface {
    private let size: CGSize
    private var body = ""
    private var openGroups = 0

    init(size: CGSize) { self.size = size }

    var document: String {
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="\(Self.number(size.width))" height="\(Self.number(size.height))" viewBox="0 0 \(Self.number(size.width)) \(Self.number(size.height))">

        """ + body + String(repeating: "</g>\n", count: openGroups) + "</svg>\n"
    }

    func concatenate(_ t: CGAffineTransform) {
        body += "<g transform=\"matrix(\([t.a, t.b, t.c, t.d, t.tx, t.ty].map(Self.number).joined(separator: " ")))\">\n"
        openGroups += 1
    }

    func fill(_ path: Path, _ color: Color) {
        let paint = Self.paint(color)
        body += "<path d=\"\(Self.pathData(path))\" fill=\"\(paint.hex)\"\(Self.opacity("fill-opacity", paint.opacity))/>\n"
    }

    func stroke(_ path: Path, _ color: Color, _ style: StrokeStyle) {
        let paint = Self.paint(color)
        let cap = switch style.lineCap {
        case .round: "round"
        case .square: "square"
        default: "butt"
        }
        var attrs = "fill=\"none\" stroke=\"\(paint.hex)\" stroke-width=\"\(Self.number(style.lineWidth))\" stroke-linecap=\"\(cap)\""
        if !style.dash.isEmpty {
            attrs += " stroke-dasharray=\"\(style.dash.map(Self.number).joined(separator: " "))\""
        }
        body += "<path d=\"\(Self.pathData(path))\" \(attrs)\(Self.opacity("stroke-opacity", paint.opacity))/>\n"
    }

    func text(_ string: String, font: DiagramFont, color: Color, at point: CGPoint, anchor: UnitPoint) {
        let paint = Self.paint(color)
        let ns = font.nsFont
        let size = font.measure(string)
        // Baseline from AppKit metrics so text sits where the Canvas draws it.
        let top = point.y - anchor.y * size.height
        let baseline = top + (size.height - (ns.ascender - ns.descender)) / 2 + ns.ascender
        let textAnchor = anchor.x < 0.25 ? "start" : (anchor.x > 0.75 ? "end" : "middle")
        let escaped = string
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        body += "<text x=\"\(Self.number(point.x))\" y=\"\(Self.number(baseline))\" text-anchor=\"\(textAnchor)\" font-family=\"Helvetica Neue, Helvetica, Arial, sans-serif\" font-size=\"\(Self.number(font.size))\" font-weight=\"\(font.cssWeight)\" fill=\"\(paint.hex)\"\(Self.opacity("fill-opacity", paint.opacity))>\(escaped)</text>\n"
    }

    private static func number(_ v: CGFloat) -> String { String(format: "%.2f", v) }

    private static func opacity(_ attribute: String, _ value: Double) -> String {
        value < 0.999 ? " \(attribute)=\"\(String(format: "%.3f", value))\"" : ""
    }

    private static func paint(_ color: Color) -> (hex: String, opacity: Double) {
        let r = color.resolve(in: EnvironmentValues())
        func byte(_ c: Float) -> Int { Int((min(max(c, 0), 1) * 255).rounded()) }
        return (String(format: "#%02X%02X%02X", byte(r.red), byte(r.green), byte(r.blue)), Double(r.opacity))
    }

    private static func pathData(_ path: Path) -> String {
        var d = ""
        func pt(_ p: CGPoint) -> String { "\(number(p.x)) \(number(p.y))" }
        path.forEach { element in
            switch element {
            case .move(let to): d += "M\(pt(to))"
            case .line(let to): d += "L\(pt(to))"
            case .quadCurve(let to, let control): d += "Q\(pt(control)) \(pt(to))"
            case .curve(let to, let c1, let c2): d += "C\(pt(c1)) \(pt(c2)) \(pt(to))"
            case .closeSubpath: d += "Z"
            }
        }
        return d
    }
}

// MARK: - Palette
//
// Every color the diagram draws, resolved once for a concrete appearance.
// Tokens are dynamic (light/dark); resolving them explicitly against an
// NSAppearance keeps on-screen drawing in sync with the window and lets
// exports always use the light ("white paper") variant.

/// Residue property class driving the residue label colors.
enum DiagramResidueClass: Sendable, CaseIterable {
    case polar, acidic, basic, hydrophobic

    init(residueName: String) {
        switch canonicalResidueName(residueName) {
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
    private let interactions: [MolecularInteraction.InteractionType: Color]
    private let residues: [DiagramResidueClass: Color]
    private let elements: [Int: Color]

    func interaction(_ type: MolecularInteraction.InteractionType) -> Color { interactions[type] ?? ink }
    func residue(_ cls: DiagramResidueClass) -> Color { residues[cls] ?? ink }
    func element(_ atomicNum: Int) -> Color { elements[atomicNum] ?? ink.opacity(0.7) }

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
        return DiagramPalette(
            paper: r(DS.Color.diagramPaper),
            ink: r(DS.Color.diagramInk),
            interactions: Dictionary(uniqueKeysWithValues:
                MolecularInteraction.InteractionType.allCases.map { ($0, r($0.themeColor)) }),
            residues: Dictionary(uniqueKeysWithValues:
                DiagramResidueClass.allCases.map { ($0, r($0.themeColor)) }),
            elements: elementTokens.mapValues(r))
    }
}

// MARK: - Skeletal renderer
//
// Shared by the ligand and the side chains. Carbons are bare bond vertices.
// Heteroatoms render as element labels (with implicit H and charge) on a
// paper-colored pill so bonds appear to end at the letter. Aromatic rings get
// an inscribed circle; double bonds follow RDKit conventions (inner line
// toward the ring center, centered for terminal atoms).

private enum Skeletal {
    static let bondWidth: CGFloat = 1.8
    static let doubleOffset: CGFloat = 4
    static let heteroInset: CGFloat = 8
}

private func elementSymbol(_ atomicNum: Int) -> String {
    Element(rawValue: atomicNum)?.symbol ?? "?"
}

private func subscriptDigits(_ n: Int) -> String {
    let digits: [Character] = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    return String(String(n).compactMap { $0.wholeNumberValue.map { digits[$0] } })
}

private func line(_ a: CGPoint, _ b: CGPoint) -> Path {
    var p = Path(); p.move(to: a); p.addLine(to: b); return p
}

private func drawSkeletal(_ mol: DiagramMolecule, on surface: some DiagramSurface,
                          palette: DiagramPalette, opacity: Double = 1) {
    let pos = mol.positions
    let count = pos.count
    var adj: [[Int]] = Array(repeating: [], count: count)
    for b in mol.bonds where b.a < count && b.b < count { adj[b.a].append(b.b); adj[b.b].append(b.a) }
    let bondColor = palette.ink.opacity(0.88 * opacity)
    let style = StrokeStyle(lineWidth: Skeletal.bondWidth, lineCap: .round)
    func isHetero(_ i: Int) -> Bool { mol.atomicNums[i] != 6 }

    // --- Bonds ---
    for bond in mol.bonds where bond.a < count && bond.b < count {
        var p1 = pos[bond.a], p2 = pos[bond.b]
        let u = unit(CGPoint(x: p2.x - p1.x, y: p2.y - p1.y))
        guard u != .zero else { continue }
        if isHetero(bond.a) { p1 = CGPoint(x: p1.x + u.x * Skeletal.heteroInset, y: p1.y + u.y * Skeletal.heteroInset) }
        if isHetero(bond.b) { p2 = CGPoint(x: p2.x - u.x * Skeletal.heteroInset, y: p2.y - u.y * Skeletal.heteroInset) }
        var nrm = CGPoint(x: -u.y, y: u.x)
        let off = Skeletal.doubleOffset

        if bond.order == 2 && !bond.isAromatic {
            let ring = mol.rings.first { $0.contains(bond.a) && $0.contains(bond.b) }
            let terminal = adj[bond.a].count == 1 || adj[bond.b].count == 1
            if ring == nil && terminal {
                // Centered pair (C=O, C=CH2).
                for s: CGFloat in [-0.5, 0.5] {
                    surface.stroke(line(CGPoint(x: p1.x + nrm.x * off * s, y: p1.y + nrm.y * off * s),
                                        CGPoint(x: p2.x + nrm.x * off * s, y: p2.y + nrm.y * off * s)),
                                   bondColor, style)
                }
                continue
            }
            // Inner line toward the ring center, or toward the substituent side.
            let mid = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            var side: CGFloat = 0
            if let ring {
                let c = centroid(ring.map { pos[$0] })
                side = (c.x - mid.x) * nrm.x + (c.y - mid.y) * nrm.y
            } else {
                for (atom, other) in [(bond.a, bond.b), (bond.b, bond.a)] {
                    for nb in adj[atom] where nb != other {
                        side += (pos[nb].x - mid.x) * nrm.x + (pos[nb].y - mid.y) * nrm.y
                    }
                }
            }
            if side < 0 { nrm = CGPoint(x: -nrm.x, y: -nrm.y) }
            surface.stroke(line(p1, p2), bondColor, style)
            let len = hypot(p2.x - p1.x, p2.y - p1.y)
            let shrink = min(len * 0.15, 6)
            surface.stroke(line(CGPoint(x: p1.x + u.x * shrink + nrm.x * off, y: p1.y + u.y * shrink + nrm.y * off),
                                CGPoint(x: p2.x - u.x * shrink + nrm.x * off, y: p2.y - u.y * shrink + nrm.y * off)),
                           bondColor, style)
        } else if bond.order == 3 {
            for s: CGFloat in [-1, 0, 1] {
                surface.stroke(line(CGPoint(x: p1.x + nrm.x * off * s, y: p1.y + nrm.y * off * s),
                                    CGPoint(x: p2.x + nrm.x * off * s, y: p2.y + nrm.y * off * s)),
                               bondColor, style)
            }
        } else {
            surface.stroke(line(p1, p2), bondColor, style)
        }
    }

    // --- Aromatic ring circles ---
    for ring in mol.aromaticRings where ring.allSatisfy({ $0 < count }) {
        let c = centroid(ring.map { pos[$0] })
        let r = ring.map { hypot(pos[$0].x - c.x, pos[$0].y - c.y) }.reduce(0, +) / CGFloat(ring.count) * 0.58
        surface.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2)),
                       bondColor.opacity(0.75), StrokeStyle(lineWidth: Skeletal.bondWidth * 0.8))
    }

    // --- Heteroatom labels ---
    for i in 0..<count where isHetero(i) {
        let p = pos[i]
        let color = palette.element(mol.atomicNums[i]).opacity(opacity)
        let symbol = elementSymbol(mol.atomicNums[i])
        let symbolSize = DiagramFont.atom.measure(symbol)
        let h = i < mol.hydrogens.count ? mol.hydrogens[i] : 0
        let hText = h > 0 ? "H" + (h > 1 ? subscriptDigits(h) : "") : ""
        let hSize = hText.isEmpty ? .zero : DiagramFont.atom.measure(hText)
        // Hydrogens go on the side away from the bonds.
        let meanDx = adj[i].isEmpty ? 0 : adj[i].map { pos[$0].x - p.x }.reduce(0, +) / CGFloat(adj[i].count)
        let hOnLeft = meanDx > 1
        let left = p.x - symbolSize.width / 2 - (hOnLeft ? hSize.width : 0)
        let right = p.x + symbolSize.width / 2 + (hOnLeft ? 0 : hSize.width)
        let height = symbolSize.height
        surface.fill(Path(roundedRect: CGRect(x: left - 2, y: p.y - height / 2, width: right - left + 4, height: height),
                          cornerRadius: height / 2),
                     palette.paper)
        surface.text(symbol, font: .atom, color: color, at: p, anchor: .center)
        if !hText.isEmpty {
            surface.text(hText, font: .atom, color: color,
                         at: CGPoint(x: hOnLeft ? p.x - symbolSize.width / 2 : p.x + symbolSize.width / 2, y: p.y),
                         anchor: hOnLeft ? .trailing : .leading)
        }
        let charge = i < mol.formalCharges.count ? mol.formalCharges[i] : 0
        if charge != 0 {
            let magnitude = abs(charge) > 1 ? "\(abs(charge))" : ""
            let label = magnitude + (charge > 0 ? "+" : "−")
            surface.text(label, font: .charge, color: palette.element(charge > 0 ? 7 : 8).opacity(opacity),
                         at: CGPoint(x: right + 1, y: p.y - height * 0.32), anchor: .leading)
        }
    }
}

// MARK: - Scene drawing

/// Transform that fits `bounds` into a canvas of `size`, centered.
func diagramFitTransform(bounds: CGRect, in size: CGSize, margin: CGFloat = 24,
                         maxScale: CGFloat = 1.4) -> CGAffineTransform {
    let w = max(bounds.width, 1), h = max(bounds.height, 1)
    let s = max(min((size.width - margin * 2) / w, (size.height - margin * 2) / h, maxScale), 0.05)
    return CGAffineTransform(translationX: size.width / 2, y: size.height / 2)
        .scaledBy(x: s, y: s)
        .translatedBy(x: -bounds.midX, y: -bounds.midY)
}

/// Draw the scene in diagram coordinates (the caller sets the transform).
/// `emphasis` dims everything that does not belong to that residue index.
func drawScene(_ scene: DiagramScene, on surface: some DiagramSurface,
               palette: DiagramPalette, emphasis: Int? = nil) {
    func alpha(_ residue: Int) -> Double { emphasis == nil || emphasis == residue ? 1 : 0.2 }
    let hydroColor = palette.interaction(.hydrophobic)

    // Connectors behind everything else.
    for mark in scene.hydrophobicMarks {
        guard let end = mark.connectorEnd else { continue }
        surface.stroke(line(mark.tip, end), hydroColor.opacity(0.8 * alpha(mark.residue)),
                       MolecularInteraction.InteractionType.hydrophobic.diagramStroke)
    }
    for l in scene.lines {
        var path = Path()
        path.move(to: l.from)
        if let c = l.control { path.addQuadCurve(to: l.to, control: c) } else { path.addLine(to: l.to) }
        surface.stroke(path, palette.interaction(l.type).opacity(alpha(l.residue)), l.type.diagramStroke)
    }

    // Ligand: halos, then skeleton.
    let ligandAlpha = emphasis == nil ? 1.0 : 0.55
    for (idx, type) in scene.ligandHalos where idx < scene.ligand.positions.count {
        let p = scene.ligand.positions[idx]
        let r: CGFloat = scene.ligand.atomicNums[idx] == 6 ? 10 : 13
        let color = palette.interaction(type)
        let disc = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
        surface.fill(disc, color.opacity(0.16 * ligandAlpha))
        surface.stroke(disc, color.opacity(0.55 * ligandAlpha), StrokeStyle(lineWidth: 1.2))
    }
    drawSkeletal(scene.ligand, on: surface, palette: palette, opacity: ligandAlpha)

    for mark in scene.hydrophobicMarks {
        drawEyelash(mark, on: surface, color: hydroColor.opacity(alpha(mark.residue)))
    }

    for (i, residue) in scene.residues.enumerated() {
        drawResidue(residue, on: surface, palette: palette, opacity: alpha(i))
    }

    // Ring centroid dots where π contacts terminate, then distance labels on top.
    for l in scene.lines {
        let color = palette.interaction(l.type).opacity(alpha(l.residue))
        for (point, isCentroid) in [(l.from, l.fromCentroid), (l.to, l.toCentroid)] where isCentroid {
            surface.fill(Path(ellipseIn: CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)), color)
        }
    }
    for l in scene.lines {
        let a = alpha(l.residue)
        surface.fill(Path(roundedRect: rect(around: l.labelCenter, size: l.labelSize), cornerRadius: 3),
                     palette.paper.opacity(0.92 * a))
        surface.text(l.label, font: .distance, color: palette.interaction(l.type).opacity(a),
                     at: l.labelCenter, anchor: .center)
    }
}

private func drawEyelash(_ mark: DiagramScene.HydrophobicMark, on surface: some DiagramSurface, color: Color) {
    let tx = -mark.normal.y, ty = mark.normal.x
    let half = mark.span / 2
    let left = CGPoint(x: mark.baseline.x - tx * half, y: mark.baseline.y - ty * half)
    let right = CGPoint(x: mark.baseline.x + tx * half, y: mark.baseline.y + ty * half)
    let control = CGPoint(x: mark.baseline.x + mark.normal.x * mark.bulge * 2,
                          y: mark.baseline.y + mark.normal.y * mark.bulge * 2)
    var arc = Path()
    arc.move(to: left)
    arc.addQuadCurve(to: right, control: control)
    surface.stroke(arc, color, StrokeStyle(lineWidth: 1.8, lineCap: .round))
    let teeth = max(3, min(7, Int(mark.span / 6)))
    for k in 0..<teeth {
        let base = segmentPoint(from: left, to: right, control: control, t: CGFloat(k) / CGFloat(teeth - 1))
        let tip = CGPoint(x: base.x + mark.normal.x * DiagramScene.HydrophobicMark.toothLength,
                          y: base.y + mark.normal.y * DiagramScene.HydrophobicMark.toothLength)
        surface.stroke(line(base, tip), color, StrokeStyle(lineWidth: 1.4, lineCap: .round))
    }
}

private func drawResidue(_ residue: DiagramScene.Residue, on surface: some DiagramSurface,
                         palette: DiagramPalette, opacity: Double) {
    let ink = palette.ink.opacity(opacity)
    let stub = StrokeStyle(lineWidth: Skeletal.bondWidth, lineCap: .round)

    if let sc = residue.sideChain, let cb = sc.positions.first {
        surface.stroke(line(residue.center, cb), ink.opacity(0.88), stub)
        drawSkeletal(sc, on: surface, palette: palette, opacity: opacity)
        for (idx, type) in residue.contactAtoms where idx < sc.positions.count {
            let p = sc.positions[idx]
            let r: CGFloat = sc.atomicNums[idx] == 6 ? 7 : 11
            surface.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)),
                           palette.interaction(type).opacity(opacity), StrokeStyle(lineWidth: 1.5))
        }
    }

    for tag in residue.backboneTags {
        let s = shortenSegment(from: tag.edge, to: tag.position, fromInset: 0, toInset: 7)
        surface.stroke(line(s.0, s.1), ink.opacity(0.88), stub)
        let symbol = elementSymbol(tag.atomicNum)
        let size = DiagramFont.atom.measure(symbol)
        surface.fill(Path(roundedRect: rect(around: tag.position, size: CGSize(width: size.width + 4, height: size.height)),
                          cornerRadius: size.height / 2), palette.paper)
        surface.text(symbol, font: .atom, color: palette.element(tag.atomicNum).opacity(opacity),
                     at: tag.position, anchor: .center)
    }

    switch residue.kind {
    case .metal(let symbol):
        let color = palette.interaction(.metalCoord)
        let disc = Path(ellipseIn: CGRect(x: residue.center.x - 15, y: residue.center.y - 15, width: 30, height: 30))
        surface.fill(disc, palette.paper)
        surface.fill(disc, color.opacity(0.25 * opacity))
        surface.stroke(disc, color.opacity(opacity), StrokeStyle(lineWidth: 1.6))
        surface.text(symbol, font: .residue, color: ink, at: residue.center, anchor: .center)
    case .sideChain, .label:
        let color = palette.residue(residue.residueClass)
        let rect = residue.bubbleRect
        let bubble = Path(roundedRect: rect, cornerRadius: rect.height / 2, style: .continuous)
        surface.fill(bubble, palette.paper)
        surface.fill(bubble, color.opacity(0.13 * opacity))
        surface.stroke(bubble, color.opacity(opacity), StrokeStyle(lineWidth: 1.3))
        surface.text(residue.label, font: .residue, color: ink, at: residue.center, anchor: .center)
    }
}

// MARK: - Exported figure (diagram + legend)

enum DiagramLegendItem: Sendable {
    case interaction(MolecularInteraction.InteractionType)
    case residueClass(DiagramResidueClass)

    var title: String {
        switch self {
        case .interaction(let type): type.label
        case .residueClass(let cls): "\(cls.label.capitalized) residue"
        }
    }
}

struct DiagramFigureLayout: Sendable {
    let size: CGSize
    let diagramTransform: CGAffineTransform
    let legend: [(item: DiagramLegendItem, origin: CGPoint)]

    private static let margin: CGFloat = 28
    private static let sampleWidth: CGFloat = 26
    private static let rowHeight: CGFloat = 20

    init(scene: DiagramScene) {
        let m = Self.margin
        let width = max(scene.bounds.width + m * 2, 560)
        let items = scene.drawnTypes.map(DiagramLegendItem.interaction)
            + scene.residueClasses.map(DiagramLegendItem.residueClass)
        var placed: [(item: DiagramLegendItem, origin: CGPoint)] = []
        let legendTop = m + scene.bounds.height + 20
        var x = m, row = 0
        for item in items {
            let w = Self.sampleWidth + 6 + ceil(DiagramFont.legend.measure(item.title).width)
            if x + w > width - m && x > m { x = m; row += 1 }
            placed.append((item, CGPoint(x: x, y: legendTop + CGFloat(row) * Self.rowHeight)))
            x += w + 20
        }
        let height = items.isEmpty
            ? scene.bounds.height + m * 2
            : legendTop + CGFloat(row + 1) * Self.rowHeight + m - Self.rowHeight / 2
        size = CGSize(width: ceil(width), height: ceil(height))
        diagramTransform = CGAffineTransform(
            translationX: (width - scene.bounds.width) / 2 - scene.bounds.minX,
            y: m - scene.bounds.minY)
        legend = placed
    }

    /// Legend sample for one item; `origin` is its left end, vertically centered.
    static func drawSample(_ item: DiagramLegendItem, at origin: CGPoint,
                           on surface: some DiagramSurface, palette: DiagramPalette) {
        let y = origin.y, x = origin.x
        switch item {
        case .interaction(.hydrophobic):
            drawEyelash(DiagramScene.HydrophobicMark(
                baseline: CGPoint(x: x + sampleWidth / 2, y: y + 4), normal: CGPoint(x: 0, y: -1),
                span: 20, residue: 0, connectorEnd: nil),
                on: surface, color: palette.interaction(.hydrophobic))
        case .interaction(let type):
            surface.stroke(line(CGPoint(x: x, y: y), CGPoint(x: x + sampleWidth, y: y)),
                           palette.interaction(type), type.diagramStroke)
        case .residueClass(let cls):
            let r = CGRect(x: x + 3, y: y - 6, width: sampleWidth - 6, height: 12)
            let bubble = Path(roundedRect: r, cornerRadius: 6, style: .continuous)
            surface.fill(bubble, palette.residue(cls).opacity(0.13))
            surface.stroke(bubble, palette.residue(cls), StrokeStyle(lineWidth: 1.3))
        }
    }

    static let sampleSize = CGSize(width: sampleWidth, height: 14)
}

/// Draw the publication figure: paper, the diagram at 1:1 scale, and a legend
/// of the interaction types and residue classes actually drawn.
func drawFigure(_ scene: DiagramScene, layout: DiagramFigureLayout,
                on surface: some DiagramSurface, palette: DiagramPalette) {
    surface.fill(Path(CGRect(origin: .zero, size: layout.size)), palette.paper)
    for entry in layout.legend {
        DiagramFigureLayout.drawSample(entry.item, at: entry.origin, on: surface, palette: palette)
        surface.text(entry.item.title, font: .legend, color: palette.ink,
                     at: CGPoint(x: entry.origin.x + DiagramFigureLayout.sampleSize.width + 6, y: entry.origin.y),
                     anchor: .leading)
    }
    surface.concatenate(layout.diagramTransform)
    drawScene(scene, on: surface, palette: palette)
}

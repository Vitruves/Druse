// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Side Chain Templates
//
// Each residue is drawn as: a colored core box (residue name + seq number)
// with a side chain rendered in classical skeletal style next to it. The
// interaction line connects to the specific interacting atom in the side
// chain (or the ring centroid for π-interactions), not the box center.
//
// Template coordinates are in a local frame: (0,0) = Cα attachment point,
// chain extends rightward. Rotated at render time so the interacting atom
// points toward the ligand.

/// Bond length (in diagram points) shared by the ligand depiction and the
/// residue side chains, so both are drawn at the same scale.
let kDiagramBondLength: CGFloat = 28

struct SideChainAtom: Sendable {
    let name: String
    let offset: CGPoint     // local coords relative to Cα
    let element: Element
}

struct SideChainBond: Sendable {
    let from: Int
    let to: Int
    let order: Int          // 1=single, 2=double (kekulized)
    let isAromatic: Bool    // true for ring bonds in PHE/TYR/HIS/TRP
}

struct SideChainTemplate: Sendable {
    let atoms: [SideChainAtom]
    let bonds: [SideChainBond]
    let aromaticRings: [[Int]]   // atom-index sets per aromatic ring
}

/// Side chain templates for the 20 standard amino acids.
/// Only functional group atoms are included (not Cα/backbone).
let sideChainTemplates: [String: SideChainTemplate] = {
    typealias A = SideChainAtom
    typealias B = SideChainBond
    let s = kDiagramBondLength
    let h: CGFloat = s * 0.866  // sin(60°) — used for both hexagons and zigzag y-offsets

    var t: [String: SideChainTemplate] = [:]

    // No side chain
    t["GLY"] = SideChainTemplate(atoms: [], bonds: [], aromaticRings: [])

    // Simple
    t["ALA"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C)],
        bonds: [],
        aromaticRings: [])

    // Branching aliphatics — proper Y at CB with 120° bond angles.
    t["VAL"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["LEU"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s, y: -h * 2), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    // ILE — Y at CB (CG1 upper, CG2 lower); CG1-CD1 zigzag right.
    t["ILE"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["PRO"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.7, y: s * 0.7), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 0.7, y: s * 1.2), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Hydroxyl / thiol — slight zigzag for visual variety.
    t["SER"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG", offset: CGPoint(x: s * 1.5, y: -h), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["THR"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG1", offset: CGPoint(x: s * 1.5, y: -h), element: .O),
                A(name: "CG2", offset: CGPoint(x: s * 1.5, y:  h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 0, to: 2, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["CYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "SG", offset: CGPoint(x: s * 1.5, y: -h), element: .S)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false)],
        aromaticRings: [])

    // MET — proper zigzag CB-CG-SD-CE with 120° angles at every vertex.
    t["MET"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "SD", offset: CGPoint(x: s * 2.5, y: -h), element: .S),
                A(name: "CE", offset: CGPoint(x: s * 3, y: 0), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Carboxylate / amide — sp2 Y at CG (or CD) with both heteroatoms at 120°.
    t["ASP"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 2.5, y: -h), element: .O),
                A(name: "OD2", offset: CGPoint(x: s * 2.5, y:  h), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLU"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 3, y: 0), element: .O),
                A(name: "OE2", offset: CGPoint(x: s * 3, y: -h * 2), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ASN"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s * 2.5, y: -h), element: .O),
                A(name: "ND2", offset: CGPoint(x: s * 2.5, y:  h), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 2, isAromatic: false),
                B(from: 1, to: 3, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["GLN"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "OE1", offset: CGPoint(x: s * 3, y: 0), element: .O),
                A(name: "NE2", offset: CGPoint(x: s * 3, y: -h * 2), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 2, isAromatic: false),
                B(from: 2, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Charged — long zigzag chains.
    t["LYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE", offset: CGPoint(x: s * 3, y: 0), element: .C),
                A(name: "NZ", offset: CGPoint(x: s * 4, y: 0), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false),
                B(from: 3, to: 4, order: 1, isAromatic: false)],
        aromaticRings: [])

    t["ARG"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 1.5, y: -h), element: .C),
                A(name: "CD",  offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "NE",  offset: CGPoint(x: s * 3, y: 0), element: .N),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "NH1", offset: CGPoint(x: s * 4.5, y: -h), element: .N),
                A(name: "NH2", offset: CGPoint(x: s * 4.5, y:  h), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: false),
                B(from: 2, to: 3, order: 1, isAromatic: false),
                B(from: 3, to: 4, order: 1, isAromatic: false),
                B(from: 4, to: 5, order: 2, isAromatic: false),
                B(from: 4, to: 6, order: 1, isAromatic: false)],
        aromaticRings: [])

    // Aromatic — proper hexagonal geometry, bond length s
    t["PHE"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE1", offset: CGPoint(x: s * 3.5, y: -h), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s * 3.5, y: h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s * 2.5, y: h), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: true),
                B(from: 2, to: 3, order: 1, isAromatic: true),
                B(from: 3, to: 4, order: 1, isAromatic: true),
                B(from: 4, to: 5, order: 1, isAromatic: true),
                B(from: 5, to: 6, order: 1, isAromatic: true),
                B(from: 6, to: 1, order: 1, isAromatic: true)],
        aromaticRings: [[1, 2, 3, 4, 5, 6]])

    t["TYR"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s * 2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s * 2.5, y: -h), element: .C),
                A(name: "CE1", offset: CGPoint(x: s * 3.5, y: -h), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s * 4, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s * 3.5, y: h), element: .C),
                A(name: "CD2", offset: CGPoint(x: s * 2.5, y: h), element: .C),
                A(name: "OH",  offset: CGPoint(x: s * 5, y: 0), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                B(from: 1, to: 2, order: 1, isAromatic: true),
                B(from: 2, to: 3, order: 1, isAromatic: true),
                B(from: 3, to: 4, order: 1, isAromatic: true),
                B(from: 4, to: 5, order: 1, isAromatic: true),
                B(from: 5, to: 6, order: 1, isAromatic: true),
                B(from: 6, to: 1, order: 1, isAromatic: true),
                B(from: 4, to: 7, order: 1, isAromatic: false)],
        aromaticRings: [[1, 2, 3, 4, 5, 6]])

    // HIS — 5-membered imidazole, regular pentagon (bond length s)
    // Pentagon center at (2.85s, 0), circumradius r = s / (2*sin(36°)) ≈ 0.851s
    t["HIS"] = {
        let r = s / (2 * sin(.pi / 5))           // ≈ 0.851 s
        let cx: CGFloat = s * 2 + r              // pentagon center
        // vertices at 180°, 108°, 36°, -36°, -108° (CW from CG, using screen y-down)
        func v(_ deg: CGFloat) -> CGPoint {
            CGPoint(x: cx + r * cos(deg * .pi / 180),
                    y: -r * sin(deg * .pi / 180))   // screen y inverted
        }
        return SideChainTemplate(
            atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                    A(name: "CG",  offset: v(180), element: .C),
                    A(name: "ND1", offset: v(108), element: .N),
                    A(name: "CE1", offset: v(36),  element: .C),
                    A(name: "NE2", offset: v(-36), element: .N),
                    A(name: "CD2", offset: v(-108), element: .C)],
            bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),
                    B(from: 1, to: 2, order: 1, isAromatic: true),
                    B(from: 2, to: 3, order: 1, isAromatic: true),
                    B(from: 3, to: 4, order: 1, isAromatic: true),
                    B(from: 4, to: 5, order: 1, isAromatic: true),
                    B(from: 5, to: 1, order: 1, isAromatic: true)],
            aromaticRings: [[1, 2, 3, 4, 5]])
    }()

    // TRP — proper indole: pyrrole (5-ring) fused to benzene (6-ring) along
    // the CD2-CE2 edge. CG attaches to CB on the left of the pyrrole; the
    // benzene extends to the lower right. All bonds length s, all angles 108°
    // (pentagon) or 120° (hexagon).
    t["TRP"] = {
        // Pyrrole pentagon: CG at (2s, 0), pentagon vertices derived from the
        // regular pentagon centered at (2s + r5, 0) where r5 = s/(2 sin 36°).
        // Coefficients precomputed for clarity.
        let CG  = CGPoint(x: s * 2, y: 0)
        let CD1 = CGPoint(x: s * 2.588, y: -s * 0.809)
        let NE1 = CGPoint(x: s * 3.539, y: -s * 0.500)
        let CE2 = CGPoint(x: s * 3.539, y:  s * 0.500)
        let CD2 = CGPoint(x: s * 2.588, y:  s * 0.809)

        // Benzene hexagon: shares the CD2-CE2 edge. Hexagon center is offset
        // from the edge midpoint by hexagon apothem (s*sqrt(3)/2) toward +y
        // (away from the pentagon center which is on -y from the edge).
        // Vertices in cycle order: CD2 → CE3 → CZ3 → CH2 → CZ2 → CE2 → CD2.
        let CE3 = CGPoint(x: s * 2.380, y: s * 1.785)
        let CZ3 = CGPoint(x: s * 3.123, y: s * 2.455)
        let CH2 = CGPoint(x: s * 4.073, y: s * 2.146)
        let CZ2 = CGPoint(x: s * 4.281, y: s * 1.169)

        return SideChainTemplate(
            atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                    A(name: "CG",  offset: CG, element: .C),
                    A(name: "CD1", offset: CD1, element: .C),
                    A(name: "NE1", offset: NE1, element: .N),
                    A(name: "CE2", offset: CE2, element: .C),
                    A(name: "CD2", offset: CD2, element: .C),
                    A(name: "CE3", offset: CE3, element: .C),
                    A(name: "CZ3", offset: CZ3, element: .C),
                    A(name: "CH2", offset: CH2, element: .C),
                    A(name: "CZ2", offset: CZ2, element: .C)],
            bonds: [B(from: 0, to: 1, order: 1, isAromatic: false),  // CB-CG
                    // Pyrrole
                    B(from: 1, to: 2, order: 1, isAromatic: true),   // CG-CD1
                    B(from: 2, to: 3, order: 1, isAromatic: true),   // CD1-NE1
                    B(from: 3, to: 4, order: 1, isAromatic: true),   // NE1-CE2
                    B(from: 4, to: 5, order: 1, isAromatic: true),   // CE2-CD2 (fused)
                    B(from: 5, to: 1, order: 1, isAromatic: true),   // CD2-CG
                    // Benzene (CD2 → CE3 → CZ3 → CH2 → CZ2 → CE2)
                    B(from: 5, to: 6, order: 1, isAromatic: true),   // CD2-CE3
                    B(from: 6, to: 7, order: 1, isAromatic: true),   // CE3-CZ3
                    B(from: 7, to: 8, order: 1, isAromatic: true),   // CZ3-CH2
                    B(from: 8, to: 9, order: 1, isAromatic: true),   // CH2-CZ2
                    B(from: 9, to: 4, order: 1, isAromatic: true)],  // CZ2-CE2
            aromaticRings: [[1, 2, 3, 4, 5],          // pyrrole
                            [4, 5, 6, 7, 8, 9]])      // benzene
    }()

    return t
}()

// MARK: - Aromatic Ring Detection (ligand)

/// Find all aromatic 5- and 6-membered rings in the kekulized bond graph.
/// A ring is "aromatic" iff every atom in it has `isAromatic[i] == true`
/// (per-atom aromaticity is captured before kekulization in the C++ side).
func detectAromaticRings(bonds: [(Int, Int, Int)],
                          atomCount: Int,
                          isAromatic: [Bool]) -> [[Int]] {
    var adj: [[Int]] = Array(repeating: [], count: atomCount)
    for (a1, a2, _) in bonds {
        guard a1 < atomCount, a2 < atomCount else { continue }
        adj[a1].append(a2); adj[a2].append(a1)
    }
    var allRings: [[Int]] = []
    var globalSeen = Set<Int>()
    for start in 0..<atomCount where !adj[start].isEmpty && !globalSeen.contains(start) {
        var comp: [Int] = []; var queue = [start]; var seen = Set([start])
        while !queue.isEmpty {
            let n = queue.removeFirst(); comp.append(n)
            for nb in adj[n] where !seen.contains(nb) { seen.insert(nb); queue.append(nb) }
        }
        globalSeen.formUnion(seen)
        let compSet = Set(comp)
        for ringSize in [5, 6] {
            for s in comp {
                var path = [s]
                findRingDFS(cur: s, start: s, depth: 1, maxDepth: ringSize,
                            path: &path, adj: adj, nodes: compSet, rings: &allRings)
            }
        }
    }
    var unique = Set<[Int]>()
    return allRings
        .filter { ring in ring.allSatisfy { idx in idx < isAromatic.count && isAromatic[idx] } }
        .filter { unique.insert(canonRing($0)).inserted }
}

private func findRingDFS(cur: Int, start: Int, depth: Int, maxDepth: Int,
                          path: inout [Int], adj: [[Int]], nodes: Set<Int>, rings: inout [[Int]]) {
    if depth == maxDepth {
        if adj[cur].contains(start) { rings.append(Array(path)) }
        return
    }
    for next in adj[cur] {
        guard nodes.contains(next), !path.contains(next) else { continue }
        if next == start && depth > 2 { continue }
        path.append(next)
        findRingDFS(cur: next, start: start, depth: depth + 1, maxDepth: maxDepth,
                    path: &path, adj: adj, nodes: nodes, rings: &rings)
        path.removeLast()
    }
}

private func canonRing(_ ring: [Int]) -> [Int] {
    guard let minVal = ring.min(), let minIdx = ring.firstIndex(of: minVal) else { return ring }
    let rot = Array(ring[minIdx...]) + Array(ring[..<minIdx])
    let rev = [rot[0]] + rot[1...].reversed()
    return rot.lexicographicallyPrecedes(rev) ? rot : rev
}

/// Template key for a residue name: protonation-state and force-field
/// variants (HID/HIE/HIP, ASH, GLH, LYN, CYX, …) share the standard template.
func canonicalResidueName(_ name: String) -> String {
    switch name {
    case "HID", "HIE", "HIP", "HSD", "HSE", "HSP": "HIS"
    case "ASH": "ASP"
    case "GLH": "GLU"
    case "LYN": "LYS"
    case "CYX", "CYM": "CYS"
    case "TYM": "TYR"
    default: name
    }
}

// MARK: - Side Chain Transform

/// Distance from the residue label center at which CB (the side chain
/// attachment point) sits — places CB just outside the label.
let kCBAnchorDistance: CGFloat = 40

/// Extra rotation applied when the side chain is oriented by its centroid, so
/// that CB→CG is never parallel to the label→ligand axis. For any sp2/sp3 atom
/// with three 120°-spaced substituents the centroid of those substituents lies
/// on the CB→CG line, which would otherwise draw Y-branch residues with a
/// straight CB→CG bond. −30° restores the conventional zigzag.
let kSideChainZigzagTilt: CGFloat = -.pi / 6

/// Transform a side chain template into diagram space so that:
///   - CB (atoms[0]) sits at `kCBAnchorDistance` from the label center,
///     on the label→ligand axis;
///   - the chain extends toward the ligand. With `referenceAtom` (the atom
///     that makes a directional contact) that atom points straight at the
///     ligand contact; otherwise the centroid of the side chain does (with the
///     zigzag tilt), keeping every branch visible.
func transformSideChain(template: SideChainTemplate,
                        bubbleCenter: CGPoint,
                        towardLigand: CGPoint,
                        referenceAtom: Int? = nil) -> [CGPoint] {
    guard !template.atoms.isEmpty else { return [] }

    let cb = template.atoms[0].offset
    var refX: CGFloat = 0, refY: CGFloat = 0
    var tilt: CGFloat = 0
    if let ref = referenceAtom, ref > 0, ref < template.atoms.count {
        refX = template.atoms[ref].offset.x - cb.x
        refY = template.atoms[ref].offset.y - cb.y
    } else {
        for atom in template.atoms.dropFirst() {
            refX += atom.offset.x - cb.x
            refY += atom.offset.y - cb.y
        }
        tilt = kSideChainZigzagTilt
    }
    if hypot(refX, refY) < 1e-3 { refX = 1; refY = 0; tilt = 0 }
    let templateAngle = atan2(refY, refX)

    let dx = towardLigand.x - bubbleCenter.x
    let dy = towardLigand.y - bubbleCenter.y
    let len = max(hypot(dx, dy), 1)
    let ux = dx / len, uy = dy / len

    let rotateBy = atan2(uy, ux) - templateAngle + tilt
    let cosA = cos(rotateBy), sinA = sin(rotateBy)
    let cbX = bubbleCenter.x + ux * kCBAnchorDistance
    let cbY = bubbleCenter.y + uy * kCBAnchorDistance

    return template.atoms.map { atom in
        let lx = atom.offset.x - cb.x
        let ly = atom.offset.y - cb.y
        return CGPoint(x: cbX + lx * cosA - ly * sinA,
                       y: cbY + lx * sinA + ly * cosA)
    }
}

// MARK: - Residue placement
//
// Residues start at an ideal position: outward from the ligand along the local
// surface normal at their (weighted) contact point, far enough that the side
// chain and the interaction line fit. A force-directed relaxation then removes
// overlaps — between residues (label + side chain atoms) and between residues
// and the ligand — while a spring keeps each residue near its ideal spot.
// Finally crossing connectors are untangled by swapping residue positions.

/// One residue to place.
struct DiagramPlacementInput {
    /// Weighted ligand contact point the residue points at.
    let anchor: CGPoint
    /// Side chain drawn toward the ligand, or nil for a label-only residue.
    let template: SideChainTemplate?
    /// Template atom oriented straight at the ligand (directional contact).
    let referenceAtom: Int?
    /// Radius of the label disc used for collision tests.
    let labelRadius: CGFloat
    /// Preferred distance from the anchor to the label center.
    let reach: CGFloat
}

/// Collision discs of a residue at `center`: the label plus every side chain atom.
func residueFootprint(_ input: DiagramPlacementInput, center: CGPoint) -> [(p: CGPoint, r: CGFloat)] {
    var discs: [(p: CGPoint, r: CGFloat)] = [(center, input.labelRadius)]
    if let template = input.template, !template.atoms.isEmpty {
        let positions = transformSideChain(template: template, bubbleCenter: center,
                                           towardLigand: input.anchor,
                                           referenceAtom: input.referenceAtom)
        for (atom, p) in zip(template.atoms, positions) {
            discs.append((p, atom.element == .C ? 10 : 13))
        }
    }
    return discs
}

/// Unit vector pointing away from the ligand at `point`: a blend of the local
/// surface normal (away from nearby atoms) and the direction from the ligand center.
func ligandOutwardDirection(at point: CGPoint, ligand: [CGPoint], center: CGPoint) -> CGPoint {
    var lx: CGFloat = 0, ly: CGFloat = 0, n: CGFloat = 0
    for q in ligand where hypot(q.x - point.x, q.y - point.y) < kDiagramBondLength * 2.2 {
        lx += q.x; ly += q.y; n += 1
    }
    let local = n > 0 ? unit(CGPoint(x: point.x - lx / n, y: point.y - ly / n)) : .zero
    let global = unit(CGPoint(x: point.x - center.x, y: point.y - center.y))
    let blended = unit(CGPoint(x: local.x + global.x * 0.6, y: local.y + global.y * 0.6))
    if blended != .zero { return blended }
    return global != .zero ? global : CGPoint(x: 0, y: -1)
}

/// Place every residue; returns the label centers in input order.
func placeResidues(_ inputs: [DiagramPlacementInput], ligand: [CGPoint], ligandCenter: CGPoint) -> [CGPoint] {
    guard !inputs.isEmpty else { return [] }
    let ligandRadius: CGFloat = 16
    let residueGap: CGFloat = 8

    func ligandOverlap(_ discs: [(p: CGPoint, r: CGFloat)]) -> Bool {
        discs.contains { d in ligand.contains { hypot($0.x - d.p.x, $0.y - d.p.y) < d.r + ligandRadius } }
    }

    // Ideal positions, pushed outward until the residue clears the ligand.
    var ideal: [CGPoint] = inputs.map { input in
        let dir = ligandOutwardDirection(at: input.anchor, ligand: ligand, center: ligandCenter)
        var p = CGPoint(x: input.anchor.x + dir.x * input.reach, y: input.anchor.y + dir.y * input.reach)
        var steps = 0
        while ligandOverlap(residueFootprint(input, center: p)) && steps < 60 {
            p.x += dir.x * 6; p.y += dir.y * 6
            steps += 1
        }
        return p
    }
    var pos = ideal

    func relax(iterations: Int) {
        for _ in 0..<iterations {
            let footprints = inputs.indices.map { residueFootprint(inputs[$0], center: pos[$0]) }
            var force = Array(repeating: CGPoint.zero, count: inputs.count)
            for i in inputs.indices {
                force[i].x += (ideal[i].x - pos[i].x) * 0.04
                force[i].y += (ideal[i].y - pos[i].y) * 0.04
                for disc in footprints[i] {
                    for q in ligand {
                        let dx = disc.p.x - q.x, dy = disc.p.y - q.y
                        let d = max(hypot(dx, dy), 0.01)
                        let minD = disc.r + ligandRadius
                        if d < minD {
                            force[i].x += dx / d * (minD - d) * 0.5
                            force[i].y += dy / d * (minD - d) * 0.5
                        }
                    }
                }
            }
            for i in inputs.indices {
                for j in (i + 1)..<inputs.count {
                    for a in footprints[i] {
                        for b in footprints[j] {
                            var dx = a.p.x - b.p.x, dy = a.p.y - b.p.y
                            var d = hypot(dx, dy)
                            let minD = a.r + b.r + residueGap
                            guard d < minD else { continue }
                            if d < 0.01 { dx = CGFloat(i - j); dy = 1; d = hypot(dx, dy) }
                            let push = (minD - d) * 0.25
                            force[i].x += dx / d * push; force[i].y += dy / d * push
                            force[j].x -= dx / d * push; force[j].y -= dy / d * push
                        }
                    }
                }
            }
            var moved: CGFloat = 0
            for i in inputs.indices {
                let len = hypot(force[i].x, force[i].y)
                let step = min(len, 10)
                guard len > 1e-4 else { continue }
                pos[i].x += force[i].x / len * step
                pos[i].y += force[i].y / len * step
                moved = max(moved, step)
            }
            if moved < 0.05 { break }
        }
    }

    relax(iterations: 300)

    // Untangle: swap residues whose anchor→label connectors cross.
    func crossings(_ p: [CGPoint]) -> Int {
        var count = 0
        for i in p.indices {
            for j in (i + 1)..<p.count
            where segmentsIntersect(inputs[i].anchor, p[i], inputs[j].anchor, p[j]) {
                count += 1
            }
        }
        return count
    }
    var current = crossings(pos)
    var pass = 0
    while current > 0 && pass < 4 {
        var improved = false
        for i in pos.indices {
            for j in (i + 1)..<pos.count
            where segmentsIntersect(inputs[i].anchor, pos[i], inputs[j].anchor, pos[j]) {
                var trial = pos
                trial.swapAt(i, j)
                let c = crossings(trial)
                if c < current {
                    pos = trial
                    ideal.swapAt(i, j)
                    current = c
                    improved = true
                }
            }
        }
        if !improved { break }
        relax(iterations: 120)
        current = crossings(pos)
        pass += 1
    }
    return pos
}

// MARK: - Line routing and label placement

/// Point on a straight or quadratic segment at parameter `t`.
func segmentPoint(from: CGPoint, to: CGPoint, control: CGPoint?, t: CGFloat) -> CGPoint {
    guard let c = control else {
        return CGPoint(x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t)
    }
    let mt = 1 - t
    return CGPoint(x: mt * mt * from.x + 2 * mt * t * c.x + t * t * to.x,
                   y: mt * mt * from.y + 2 * mt * t * c.y + t * t * to.y)
}

/// Quadratic control point that bends a connector around the ligand when the
/// straight line would cut through it, or nil when the straight line is clear.
/// Atoms bonded to the start atom (within one bond length) are ignored — the
/// line legitimately leaves from among them.
func connectorBend(from: CGPoint, to: CGPoint, ligand: [CGPoint]) -> CGPoint? {
    let obstacles = ligand.filter { hypot($0.x - from.x, $0.y - from.y) > kDiagramBondLength * 1.1 }
    func hits(_ control: CGPoint?) -> Int {
        var count = 0
        for q in obstacles {
            for k in 1..<20 {
                let p = segmentPoint(from: from, to: to, control: control, t: CGFloat(k) / 20)
                if hypot(p.x - q.x, p.y - q.y) < 11 { count += 1; break }
            }
        }
        return count
    }
    var best = hits(nil)
    guard best > 0 else { return nil }
    let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
    let perp = unit(CGPoint(x: -(to.y - from.y), y: to.x - from.x))
    var bestControl: CGPoint?
    for k: CGFloat in [40, -40, 80, -80, 120, -120, 170, -170] {
        let control = CGPoint(x: mid.x + perp.x * k, y: mid.y + perp.y * k)
        let h = hits(control)
        if h < best { best = h; bestControl = control }
        if h == 0 { break }
    }
    return bestControl
}

/// Greedy label placement along connectors: each label takes the first
/// candidate position (midpoint first, then sliding toward the ends and
/// sideways) that overlaps no obstacle or previously placed label; when every
/// candidate collides, the one with the least overlap wins.
func placeConnectorLabels(paths: [(from: CGPoint, to: CGPoint, control: CGPoint?)],
                          sizes: [CGSize], obstacles: [CGRect]) -> [CGPoint] {
    var placed: [CGRect] = []
    var result: [CGPoint] = []
    let ts: [CGFloat] = [0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74]
    for (path, size) in zip(paths, sizes) {
        let dir = unit(CGPoint(x: path.to.x - path.from.x, y: path.to.y - path.from.y))
        let perp = CGPoint(x: -dir.y, y: dir.x)
        let sideOffset = max(size.width, size.height) / 2 + 3
        var best = CGPoint.zero
        var bestOverlap = CGFloat.greatestFiniteMagnitude
        search: for side: CGFloat in [0, 1, -1] {
            for t in ts {
                let base = segmentPoint(from: path.from, to: path.to, control: path.control, t: t)
                let c = CGPoint(x: base.x + perp.x * sideOffset * side, y: base.y + perp.y * sideOffset * side)
                let rect = CGRect(x: c.x - size.width / 2, y: c.y - size.height / 2,
                                  width: size.width, height: size.height)
                var overlap: CGFloat = 0
                for o in obstacles + placed {
                    let i = rect.intersection(o)
                    if !i.isNull { overlap += i.width * i.height }
                }
                if overlap < bestOverlap { bestOverlap = overlap; best = c }
                if overlap == 0 { break search }
            }
        }
        placed.append(CGRect(x: best.x - size.width / 2, y: best.y - size.height / 2,
                             width: size.width, height: size.height))
        result.append(best)
    }
    return result
}

// MARK: - Geometry helpers

func unit(_ v: CGPoint) -> CGPoint {
    let len = hypot(v.x, v.y)
    return len > 1e-6 ? CGPoint(x: v.x / len, y: v.y / len) : .zero
}

func shortenSegment(from p1: CGPoint, to p2: CGPoint,
                    fromInset: CGFloat, toInset: CGFloat) -> (CGPoint, CGPoint) {
    let dx = p2.x - p1.x, dy = p2.y - p1.y
    let len = hypot(dx, dy)
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
    let t = min(tx, ty)
    return t == .greatestFiniteMagnitude ? 0 : max(0, t)
}

func segmentsIntersect(_ a1: CGPoint, _ a2: CGPoint, _ b1: CGPoint, _ b2: CGPoint) -> Bool {
    func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    let d1 = cross(b1, b2, a1), d2 = cross(b1, b2, a2)
    let d3 = cross(a1, a2, b1), d4 = cross(a1, a2, b2)
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
           ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
}

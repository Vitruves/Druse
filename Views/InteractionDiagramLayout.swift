import SwiftUI

// MARK: - Side Chain Templates
//
// Each residue is drawn as: a colored core box (residue name + seq number)
// with side chain atoms extending outward. The interaction line connects
// to the specific interacting atom in the side chain, not the box center.
//
// Template coordinates are in a local frame: (0,0) = Cα attachment point,
// chain extends rightward. Rotated at render time so the interacting atom
// points toward the ligand.

struct SideChainAtom {
    let name: String
    let offset: CGPoint     // local coords relative to Cα
    let element: Element
}

struct SideChainBond {
    let from: Int
    let to: Int
    let order: Int          // 1=single, 2=double
}

struct SideChainTemplate {
    let atoms: [SideChainAtom]
    let bonds: [SideChainBond]
}

/// Side chain templates for the 20 standard amino acids.
/// Only functional group atoms are included (not Cα/backbone).
let sideChainTemplates: [String: SideChainTemplate] = {
    typealias A = SideChainAtom
    typealias B = SideChainBond
    let s: CGFloat = 14  // bond length in local coords

    var t: [String: SideChainTemplate] = [:]

    // No side chain
    t["GLY"] = SideChainTemplate(atoms: [], bonds: [])

    // Simple
    t["ALA"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C)],
        bonds: [])

    // Branching aliphatics
    t["VAL"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s*2, y: -s*0.6), element: .C),
                A(name: "CG2", offset: CGPoint(x: s*2, y: s*0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 0, to: 2, order: 1)])

    t["LEU"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s*3, y: -s*0.6), element: .C),
                A(name: "CD2", offset: CGPoint(x: s*3, y: s*0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 1, to: 3, order: 1)])

    t["ILE"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG1", offset: CGPoint(x: s*2, y: -s*0.5), element: .C),
                A(name: "CG2", offset: CGPoint(x: s, y: s), element: .C),
                A(name: "CD1", offset: CGPoint(x: s*3, y: -s*0.5), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 0, to: 2, order: 1), B(from: 1, to: 3, order: 1)])

    t["PRO"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*1.7, y: s*0.7), element: .C),
                A(name: "CD", offset: CGPoint(x: s*0.7, y: s*1.2), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1)])

    // Hydroxyl / thiol
    t["SER"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG", offset: CGPoint(x: s*2, y: 0), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1)])

    t["THR"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "OG1", offset: CGPoint(x: s*2, y: -s*0.6), element: .O),
                A(name: "CG2", offset: CGPoint(x: s*2, y: s*0.6), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 0, to: 2, order: 1)])

    t["CYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "SG", offset: CGPoint(x: s*2, y: 0), element: .S)],
        bonds: [B(from: 0, to: 1, order: 1)])

    t["MET"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "SD", offset: CGPoint(x: s*3, y: 0), element: .S),
                A(name: "CE", offset: CGPoint(x: s*4, y: 0), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 2, to: 3, order: 1)])

    // Carboxylate / amide
    t["ASP"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s*3, y: -s*0.6), element: .O),
                A(name: "OD2", offset: CGPoint(x: s*3, y: s*0.6), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 2), B(from: 1, to: 3, order: 1)])

    t["GLU"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s*3, y: 0), element: .C),
                A(name: "OE1", offset: CGPoint(x: s*4, y: -s*0.6), element: .O),
                A(name: "OE2", offset: CGPoint(x: s*4, y: s*0.6), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1),
                B(from: 2, to: 3, order: 2), B(from: 2, to: 4, order: 1)])

    t["ASN"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "OD1", offset: CGPoint(x: s*3, y: -s*0.6), element: .O),
                A(name: "ND2", offset: CGPoint(x: s*3, y: s*0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 2), B(from: 1, to: 3, order: 1)])

    t["GLN"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s*3, y: 0), element: .C),
                A(name: "OE1", offset: CGPoint(x: s*4, y: -s*0.6), element: .O),
                A(name: "NE2", offset: CGPoint(x: s*4, y: s*0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1),
                B(from: 2, to: 3, order: 2), B(from: 2, to: 4, order: 1)])

    // Charged
    t["LYS"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s*3, y: 0), element: .C),
                A(name: "CE", offset: CGPoint(x: s*4, y: 0), element: .C),
                A(name: "NZ", offset: CGPoint(x: s*5, y: 0), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1),
                B(from: 2, to: 3, order: 1), B(from: 3, to: 4, order: 1)])

    t["ARG"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD", offset: CGPoint(x: s*3, y: 0), element: .C),
                A(name: "NE", offset: CGPoint(x: s*4, y: 0), element: .N),
                A(name: "CZ", offset: CGPoint(x: s*5, y: 0), element: .C),
                A(name: "NH1", offset: CGPoint(x: s*6, y: -s*0.6), element: .N),
                A(name: "NH2", offset: CGPoint(x: s*6, y: s*0.6), element: .N)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1),
                B(from: 2, to: 3, order: 1), B(from: 3, to: 4, order: 1),
                B(from: 4, to: 5, order: 2), B(from: 4, to: 6, order: 1)])

    // Aromatic — simplified: ring drawn as hexagon
    t["PHE"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s*2.7, y: -s*0.7), element: .C),
                A(name: "CE1", offset: CGPoint(x: s*3.7, y: -s*0.7), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s*4.2, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s*3.7, y: s*0.7), element: .C),
                A(name: "CD2", offset: CGPoint(x: s*2.7, y: s*0.7), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 2, to: 3, order: 1),
                B(from: 3, to: 4, order: 1), B(from: 4, to: 5, order: 1), B(from: 5, to: 6, order: 1),
                B(from: 6, to: 1, order: 1)])

    t["TYR"] = SideChainTemplate(
        atoms: [A(name: "CB", offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG", offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s*2.7, y: -s*0.7), element: .C),
                A(name: "CE1", offset: CGPoint(x: s*3.7, y: -s*0.7), element: .C),
                A(name: "CZ",  offset: CGPoint(x: s*4.2, y: 0), element: .C),
                A(name: "CE2", offset: CGPoint(x: s*3.7, y: s*0.7), element: .C),
                A(name: "CD2", offset: CGPoint(x: s*2.7, y: s*0.7), element: .C),
                A(name: "OH",  offset: CGPoint(x: s*5.2, y: 0), element: .O)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 2, to: 3, order: 1),
                B(from: 3, to: 4, order: 1), B(from: 4, to: 5, order: 1), B(from: 5, to: 6, order: 1),
                B(from: 6, to: 1, order: 1), B(from: 4, to: 7, order: 1)])

    // HIS — 5-membered imidazole
    t["HIS"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "ND1", offset: CGPoint(x: s*2.8, y: -s*0.6), element: .N),
                A(name: "CE1", offset: CGPoint(x: s*3.6, y: 0), element: .C),
                A(name: "NE2", offset: CGPoint(x: s*3.2, y: s*0.7), element: .N),
                A(name: "CD2", offset: CGPoint(x: s*2.3, y: s*0.7), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 2, to: 3, order: 1),
                B(from: 3, to: 4, order: 1), B(from: 4, to: 5, order: 1), B(from: 5, to: 1, order: 1)])

    // TRP — simplified: 5-ring + 6-ring fused
    t["TRP"] = SideChainTemplate(
        atoms: [A(name: "CB",  offset: CGPoint(x: s, y: 0), element: .C),
                A(name: "CG",  offset: CGPoint(x: s*2, y: 0), element: .C),
                A(name: "CD1", offset: CGPoint(x: s*2.6, y: -s*0.7), element: .C),
                A(name: "NE1", offset: CGPoint(x: s*3.5, y: -s*0.4), element: .N),
                A(name: "CE2", offset: CGPoint(x: s*3.5, y: s*0.4), element: .C),
                A(name: "CD2", offset: CGPoint(x: s*2.6, y: s*0.7), element: .C),
                A(name: "CE3", offset: CGPoint(x: s*2.8, y: s*1.5), element: .C),
                A(name: "CZ3", offset: CGPoint(x: s*3.8, y: s*1.5), element: .C),
                A(name: "CH2", offset: CGPoint(x: s*4.3, y: s*0.8), element: .C),
                A(name: "CZ2", offset: CGPoint(x: s*4.1, y: s*0.1), element: .C)],
        bonds: [B(from: 0, to: 1, order: 1), B(from: 1, to: 2, order: 1), B(from: 2, to: 3, order: 1),
                B(from: 3, to: 4, order: 1), B(from: 4, to: 5, order: 1), B(from: 5, to: 1, order: 1),
                B(from: 5, to: 6, order: 1), B(from: 6, to: 7, order: 1), B(from: 7, to: 8, order: 1),
                B(from: 8, to: 9, order: 1), B(from: 9, to: 4, order: 1)])

    return t
}()

// MARK: - Aromatic Ring Detection

func detectAromaticRings(bonds: [(Int, Int, Int)], atomCount: Int) -> [[Int]] {
    var adj: [[Int]] = Array(repeating: [], count: atomCount)
    for (a1, a2, order) in bonds {
        guard order == 4, a1 < atomCount, a2 < atomCount else { continue }
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
    return allRings.filter { unique.insert(canonRing($0)).inserted }
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

// MARK: - Side Chain Transform

/// Transform a side chain template so the interacting atom faces toward the ligand.
/// Returns screen-space positions for each template atom.
func transformSideChain(template: SideChainTemplate, bubbleCenter: CGPoint,
                         interactingAtomName: String, towardLigand: CGPoint) -> [CGPoint] {
    guard !template.atoms.isEmpty else { return [] }

    // Find the interacting atom in the template
    let targetIdx = template.atoms.firstIndex(where: { $0.name == interactingAtomName })
        ?? template.atoms.count - 1  // default: outermost atom

    // Direction from bubble center toward ligand contact
    let dx = towardLigand.x - bubbleCenter.x
    let dy = towardLigand.y - bubbleCenter.y
    let angle = atan2(dy, dx)

    // Template atoms: rotate so chain points toward ligand
    let cosA = cos(angle), sinA = sin(angle)

    // Center the template on the target atom's offset so it aligns with the bubble edge
    let pivot = template.atoms[targetIdx].offset

    let dist = max(hypot(dx, dy), 1)
    let ux = dx / dist, uy = dy / dist
    // Place interacting atom 50px from bubble center (well outside the ~35px bubble half-width)
    let anchorDist: CGFloat = 50

    return template.atoms.map { atom in
        let lx = atom.offset.x - pivot.x
        let ly = atom.offset.y - pivot.y
        let rx = lx * cosA - ly * sinA
        let ry = lx * sinA + ly * cosA
        return CGPoint(x: bubbleCenter.x + ux * anchorDist + rx,
                       y: bubbleCenter.y + uy * anchorDist + ry)
    }
}

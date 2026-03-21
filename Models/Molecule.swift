import Foundation
import simd

@Observable
@MainActor
final class Molecule: Identifiable {
    let id = UUID()
    var name: String
    var atoms: [Atom]
    var bonds: [Bond]
    var residues: [Residue]
    var chains: [Chain]
    var title: String
    /// Canonical SMILES string (set when molecule originates from SMILES input).
    /// Used for torsion tree construction. Separate from `title` which is for display.
    var smiles: String?

    // Secondary structure assignments from PDB HELIX/SHEET records
    var secondaryStructureAssignments: [(start: Int, end: Int, type: SecondaryStructure, chain: String)] = []

    // Cached geometry
    private(set) var center: SIMD3<Float> = .zero
    private(set) var radius: Float = 5.0

    // Adjacency list: atomIndex → [atomIndex]
    private var adjacency: [[Int]] = []

    init(name: String, atoms: [Atom] = [], bonds: [Bond] = [], title: String = "", smiles: String? = nil) {
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
        self.residues = []
        self.chains = []
        self.title = title
        self.smiles = smiles
        rebuildDerivedData()
    }

    // MARK: - Computed Properties

    var atomCount: Int { atoms.count }
    var bondCount: Int { bonds.count }
    var heavyAtomCount: Int { atoms.filter { $0.element != .H }.count }

    var molecularWeight: Float {
        atoms.reduce(0) { $0 + $1.element.mass }
    }

    var positions: [SIMD3<Float>] {
        atoms.map(\.position)
    }

    var chainIDs: [String] {
        chains.map(\.id)
    }

    // MARK: - Rebuild

    func rebuildDerivedData() {
        rebuildGeometry()
        rebuildAdjacency()
        rebuildResiduesAndChains()
    }

    private func rebuildGeometry() {
        let pts = positions
        guard !pts.isEmpty else { return }
        center = centroid(pts)
        radius = boundingRadius(positions: pts, center: center)
    }

    private func rebuildAdjacency() {
        adjacency = Array(repeating: [], count: atoms.count)
        for bond in bonds {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard a < atoms.count, b < atoms.count else { continue }
            adjacency[a].append(b)
            adjacency[b].append(a)
        }
    }

    /// Sanitize a chain ID: keep only printable ASCII, limit to 3 chars.
    /// Returns "X" for empty or fully-garbled IDs.
    private static func sanitizeChainID(_ raw: String) -> String {
        let cleaned = String(raw.unicodeScalars.filter { $0.isASCII && !$0.properties.isWhitespace && $0.value >= 0x21 }.prefix(3))
        return cleaned.isEmpty ? "X" : cleaned
    }

    private func rebuildResiduesAndChains() {
        // Sanitize atom chain IDs in-place to prevent garbled chains
        for i in 0..<atoms.count {
            let raw = atoms[i].chainID
            let clean = Molecule.sanitizeChainID(raw)
            if clean != raw {
                atoms[i].chainID = clean
            }
        }

        // Group atoms by (chainID, residueSeq, residueName)
        struct ResKey: Hashable {
            let chainID: String
            let seq: Int
            let name: String
        }

        var resMap: [ResKey: [Int]] = [:]
        var resOrder: [ResKey] = []

        for (i, atom) in atoms.enumerated() {
            let key = ResKey(chainID: atom.chainID, seq: atom.residueSeq, name: atom.residueName)
            if resMap[key] == nil {
                resOrder.append(key)
            }
            resMap[key, default: []].append(i)
        }

        residues = resOrder.enumerated().map { idx, key in
            let isWater = key.name == "HOH" || key.name == "WAT"
            let standardAA = Set([
                "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
            ])
            return Residue(
                id: idx,
                name: key.name,
                sequenceNumber: key.seq,
                chainID: key.chainID,
                atomIndices: resMap[key] ?? [],
                isStandard: standardAA.contains(key.name),
                isWater: isWater
            )
        }

        // Group residues by chainID
        var chainMap: [String: [Int]] = [:]
        var chainOrder: [String] = []
        for (i, res) in residues.enumerated() {
            if chainMap[res.chainID] == nil {
                chainOrder.append(res.chainID)
            }
            chainMap[res.chainID, default: []].append(i)
        }

        chains = chainOrder.map { cid in
            let resIndices = chainMap[cid] ?? []
            let type: ChainType = {
                guard let firstRes = resIndices.first else { return .unknown }
                let res = residues[firstRes]
                if res.isWater { return .water }
                if res.isStandard { return .protein }
                if res.name.count <= 3 && atoms[res.atomIndices.first ?? 0].isHetAtom { return .ligand }
                return .unknown
            }()
            return Chain(id: cid, residueIndices: resIndices, type: type)
        }
    }

    // MARK: - Queries

    func neighbors(of atomIndex: Int) -> [Int] {
        guard atomIndex < adjacency.count else { return [] }
        return adjacency[atomIndex]
    }

    func residue(forAtom atomIndex: Int) -> Residue? {
        let atom = atoms[atomIndex]
        return residues.first { $0.chainID == atom.chainID && $0.sequenceNumber == atom.residueSeq }
    }

    func residueIndex(forAtom atomIndex: Int) -> Int? {
        let atom = atoms[atomIndex]
        return residues.firstIndex { $0.chainID == atom.chainID && $0.sequenceNumber == atom.residueSeq }
    }

    func atomIndices(forResidueIndex resIdx: Int) -> [Int] {
        guard resIdx < residues.count else { return [] }
        return residues[resIdx].atomIndices
    }

    func atomIndices(forChainID chainID: String) -> [Int] {
        guard let chain = chains.first(where: { $0.id == chainID }) else { return [] }
        return chain.residueIndices.flatMap { residues[$0].atomIndices }
    }

    // MARK: - Editing

    /// Remove residues at the given indices, along with their atoms and associated bonds.
    /// Rebuilds all derived data (residues, chains, adjacency, geometry) afterward.
    func removeResidues(at residueIndices: Set<Int>) {
        guard !residueIndices.isEmpty else { return }

        // Collect atom indices to remove
        var atomsToRemove = Set<Int>()
        for resIdx in residueIndices {
            guard resIdx < residues.count else { continue }
            atomsToRemove.formUnion(residues[resIdx].atomIndices)
        }
        guard !atomsToRemove.isEmpty else { return }

        // Build old → new atom index mapping
        var indexMap = [Int: Int]()
        var newIdx = 0
        for i in 0..<atoms.count {
            if !atomsToRemove.contains(i) {
                indexMap[i] = newIdx
                newIdx += 1
            }
        }

        // Remove atoms (reverse order to preserve indices)
        for idx in atomsToRemove.sorted().reversed() {
            atoms.remove(at: idx)
        }

        // Remap bonds, dropping any that reference removed atoms
        bonds = bonds.compactMap { bond in
            guard let newA = indexMap[bond.atomIndex1],
                  let newB = indexMap[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: newA, atomIndex2: newB,
                        order: bond.order, isRotatable: bond.isRotatable)
        }

        // Update secondary structure assignments: remove those whose chain's residues are gone
        secondaryStructureAssignments = secondaryStructureAssignments.filter { ssa in
            // Keep if any residue in this range still exists after removal
            atoms.contains { $0.chainID == ssa.chain && $0.residueSeq >= ssa.start && $0.residueSeq <= ssa.end }
        }

        rebuildDerivedData()
    }

    /// Rename a chain: updates all atoms' chainID and rebuilds.
    func renameChain(from oldID: String, to newID: String) {
        guard oldID != newID, !newID.isEmpty else { return }
        let sanitized = Molecule.sanitizeChainID(newID)
        for i in 0..<atoms.count where atoms[i].chainID == oldID {
            atoms[i].chainID = sanitized
        }
        // Update SS assignments
        for i in 0..<secondaryStructureAssignments.count {
            if secondaryStructureAssignments[i].chain == oldID {
                secondaryStructureAssignments[i].chain = sanitized
            }
        }
        rebuildDerivedData()
    }

    /// Merge source chain into target chain: reassigns all atoms from source to target chainID.
    func mergeChains(from sourceID: String, into targetID: String) {
        guard sourceID != targetID else { return }
        for i in 0..<atoms.count where atoms[i].chainID == sourceID {
            atoms[i].chainID = targetID
        }
        for i in 0..<secondaryStructureAssignments.count {
            if secondaryStructureAssignments[i].chain == sourceID {
                secondaryStructureAssignments[i].chain = targetID
            }
        }
        rebuildDerivedData()
    }
}

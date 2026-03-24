import Foundation
import simd

enum Protonation {

    enum ChemicalKind: Sendable {
        case acid
        case base
        case histidine
        case nTerminus
        case cTerminus
    }

    enum StateKind: String, Sendable {
        case protonated
        case deprotonated
        case histidineDelta
        case histidineEpsilon
        case histidineDoublyProtonated
    }

    struct ResiduePrediction: Sendable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let chemicalKind: ChemicalKind
        let state: StateKind
        let modelPKa: Float
        let shiftedPKa: Float
        let burialContribution: Float
        let hydrogenBondContribution: Float
        let chargeContribution: Float
        let managedAtomNames: Set<String>
        let atomFormalCharges: [String: Int]
        let protonatedAtoms: Set<String>

        var summary: String {
            "\(residueName) \(residueSeq) \(state.rawValue) pKa \(String(format: "%.2f", shiftedPKa))"
        }
    }

    private struct ResidueKey: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
    }

    private struct SiteDefinition {
        let residueName: String
        let kind: ChemicalKind
        let groupType: String
        let modelPKa: Float
        let chargeSign: Int
        let siteAtoms: [String]
        let managedAtoms: Set<String>
    }

    private struct SiteSnapshot {
        let key: ResidueKey
        let definition: SiteDefinition
        let center: SIMD3<Float>
    }

    /// Spatial grid for O(1) neighbor lookups instead of O(N) all-atom scans.
    private struct SpatialGrid {
        let cellSize: Float
        let origin: SIMD3<Float>
        let dims: SIMD3<Int32>
        let cells: [[Int]]  // cell index → atom indices

        init(atoms: [Atom], cellSize: Float) {
            self.cellSize = cellSize
            guard !atoms.isEmpty else {
                self.origin = .zero
                self.dims = .zero
                self.cells = []
                return
            }

            var minPos = atoms[0].position
            var maxPos = atoms[0].position
            for atom in atoms {
                minPos = simd_min(minPos, atom.position)
                maxPos = simd_max(maxPos, atom.position)
            }
            // Pad by one cell to avoid boundary issues.
            minPos -= SIMD3<Float>(repeating: cellSize)
            let range = maxPos - minPos + SIMD3<Float>(repeating: cellSize)
            let d = SIMD3<Int32>(
                max(1, Int32(ceilf(range.x / cellSize))),
                max(1, Int32(ceilf(range.y / cellSize))),
                max(1, Int32(ceilf(range.z / cellSize)))
            )

            self.origin = minPos
            self.dims = d
            let totalCells = Int(d.x) * Int(d.y) * Int(d.z)
            var grid = [[Int]](repeating: [], count: totalCells)

            for (index, atom) in atoms.enumerated() {
                guard atom.element != .H else { continue }
                let rel = atom.position - minPos
                let cx = min(Int(d.x) - 1, max(0, Int(rel.x / cellSize)))
                let cy = min(Int(d.y) - 1, max(0, Int(rel.y / cellSize)))
                let cz = min(Int(d.z) - 1, max(0, Int(rel.z / cellSize)))
                let ci = cx + Int(d.x) * (cy + Int(d.y) * cz)
                grid[ci].append(index)
            }
            self.cells = grid
        }

        /// Returns indices of heavy atoms within `radius` of `center`.
        func query(center: SIMD3<Float>, radius: Float, atoms: [Atom]) -> [Int] {
            let radiusSq = radius * radius
            let rel = center - origin
            let cellRadius = Int(ceilf(radius / cellSize))
            let cx0 = max(0, Int(rel.x / cellSize) - cellRadius)
            let cy0 = max(0, Int(rel.y / cellSize) - cellRadius)
            let cz0 = max(0, Int(rel.z / cellSize) - cellRadius)
            let cx1 = min(Int(dims.x) - 1, Int(rel.x / cellSize) + cellRadius)
            let cy1 = min(Int(dims.y) - 1, Int(rel.y / cellSize) + cellRadius)
            let cz1 = min(Int(dims.z) - 1, Int(rel.z / cellSize) + cellRadius)

            var result: [Int] = []
            for cz in cz0...cz1 {
                for cy in cy0...cy1 {
                    for cx in cx0...cx1 {
                        let ci = cx + Int(dims.x) * (cy + Int(dims.y) * cz)
                        for idx in cells[ci] {
                            if simd_distance_squared(atoms[idx].position, center) <= radiusSq {
                                result.append(idx)
                            }
                        }
                    }
                }
            }
            return result
        }
    }

    /// Bond adjacency list for O(1) neighbor lookups instead of O(B) bond scans.
    private struct BondAdjacency {
        let neighbors: [[Int]]

        init(bonds: [Bond], atomCount: Int) {
            var adj = [[Int]](repeating: [], count: atomCount)
            for bond in bonds {
                adj[bond.atomIndex1].append(bond.atomIndex2)
                adj[bond.atomIndex2].append(bond.atomIndex1)
            }
            self.neighbors = adj
        }

        func bonded(to atomIndex: Int) -> [Int] {
            atomIndex < neighbors.count ? neighbors[atomIndex] : []
        }
    }

    // Parameters below are taken from temp/propka/propka/propka.cfg and
    // temp/propka/propka/energy.py.
    private enum PropkaConfig {
        static let nTermPKa: Float = 8.00
        static let cTermPKa: Float = 3.20
        static let sidechainInteraction: Float = 0.85
        static let desolvationSurfaceScalingFactor: Float = 0.25
        static let desolvationPrefactor: Float = -13.0
        static let desolvationAllowance: Float = 0.0
        static let desolvationCutoff: Float = 20.0
        static let buriedCutoff: Float = 15.0
        static let nmin: Int = 280
        static let nmax: Int = 560
        static let minDistance: Float = 2.75
        static let coulombCutoff1: Float = 4.0
        static let coulombCutoff2: Float = 10.0
        static let dielectric1: Float = 160.0
        static let dielectric2: Float = 30.0
        static let pKaScaling1: Float = 244.12
        static let combinedBuriedMax: Int = 900
        static let separateBuriedMax: Int = 400
        static let cooHisException: Float = 1.60
        static let ocoHisException: Float = 1.60
        static let cysHisException: Float = 1.60
        static let cysCysException: Float = 3.60
        static let angularFactorMinimum: Float = 0.001

        // Exact proton bond lengths from temp/propka/propka/protonate.py.
        static let protonBondLengths: [String: Float] = [
            "N": 1.01,
            "O": 0.96,
            "S": 1.35,
        ]

        static let vanDerWaalsVolume: [String: Float] = [
            "C": 1.40,
            "C4": 2.64,
            "N": 1.06,
            "O": 1.00,
            "S": 1.66,
            "F": 0.90,
            "CL": 1.53,
            "P": 1.66,
        ]

        static let hydrogenBondCutoffs: [String: SIMD2<Float>] = [
            pairKey("COO", "COO"): SIMD2(2.5, 3.5),
            pairKey("COO", "ARG"): SIMD2(1.85, 2.85),
            pairKey("COO", "LYS"): SIMD2(2.85, 3.85),
            pairKey("COO", "HIS"): SIMD2(2.0, 3.0),
            pairKey("COO", "AMD"): SIMD2(2.0, 3.0),
            pairKey("COO", "TRP"): SIMD2(2.0, 3.0),
            pairKey("COO", "ROH"): SIMD2(2.65, 3.65),
            pairKey("COO", "TYR"): SIMD2(2.65, 3.65),
            pairKey("COO", "N+"): SIMD2(2.85, 3.85),
            pairKey("ARG", "CYS"): SIMD2(2.5, 4.0),
            pairKey("ARG", "TYR"): SIMD2(2.5, 4.0),
            pairKey("HIS", "AMD"): SIMD2(2.0, 3.0),
            pairKey("HIS", "TYR"): SIMD2(2.0, 3.0),
            pairKey("CYS", "CYS"): SIMD2(3.0, 5.0),
            pairKey("CYS", "TRP"): SIMD2(2.5, 3.5),
            pairKey("CYS", "ROH"): SIMD2(3.5, 4.5),
            pairKey("CYS", "AMD"): SIMD2(2.5, 3.5),
            pairKey("CYS", "TYR"): SIMD2(3.5, 4.5),
            pairKey("CYS", "N+"): SIMD2(3.0, 4.5),
            pairKey("TYR", "TYR"): SIMD2(3.5, 4.5),
            pairKey("TYR", "TRP"): SIMD2(2.5, 3.5),
            pairKey("TYR", "ROH"): SIMD2(3.5, 4.5),
            pairKey("TYR", "AMD"): SIMD2(2.5, 3.5),
            pairKey("TYR", "N+"): SIMD2(3.0, 4.5),
        ]

        // Exact backbone hydrogen-bond parameters from temp/propka/propka/propka.cfg.
        static let backboneNHHydrogenBondParameters: [String: (maxShift: Float, cutoffs: SIMD2<Float>)] = [
            "COO": (0.85, SIMD2(2.0, 3.0)),
            "CYS": (0.85, SIMD2(3.0, 4.0)),
            "TYR": (0.85, SIMD2(2.2, 3.2)),
        ]

        static let backboneCOHydrogenBondParameters: [String: (maxShift: Float, cutoffs: SIMD2<Float>)] = [
            "HIS": (0.85, SIMD2(2.0, 3.0)),
        ]

        // Protein-relevant subset of temp/propka/propka/propka.cfg base_list.
        static let baseGroupTypes: Set<String> = ["ARG", "LYS", "HIS", "N+"]

        static func pairKey(_ lhs: String, _ rhs: String) -> String {
            lhs < rhs ? "\(lhs)|\(rhs)" : "\(rhs)|\(lhs)"
        }
    }

    private static let siteDefinitions: [String: SiteDefinition] = [
        "ASP": SiteDefinition(residueName: "ASP", kind: .acid, groupType: "COO", modelPKa: 3.80, chargeSign: -1, siteAtoms: ["OD1", "OD2"], managedAtoms: ["OD1", "OD2"]),
        "GLU": SiteDefinition(residueName: "GLU", kind: .acid, groupType: "COO", modelPKa: 4.50, chargeSign: -1, siteAtoms: ["OE1", "OE2"], managedAtoms: ["OE1", "OE2"]),
        "CYS": SiteDefinition(residueName: "CYS", kind: .acid, groupType: "CYS", modelPKa: 9.00, chargeSign: -1, siteAtoms: ["SG"], managedAtoms: ["SG"]),
        "TYR": SiteDefinition(residueName: "TYR", kind: .acid, groupType: "TYR", modelPKa: 10.00, chargeSign: -1, siteAtoms: ["OH"], managedAtoms: ["OH"]),
        "HIS": SiteDefinition(residueName: "HIS", kind: .histidine, groupType: "HIS", modelPKa: 6.50, chargeSign: 1, siteAtoms: ["ND1", "NE2"], managedAtoms: ["ND1", "NE2"]),
        "LYS": SiteDefinition(residueName: "LYS", kind: .base, groupType: "LYS", modelPKa: 10.50, chargeSign: 1, siteAtoms: ["NZ"], managedAtoms: ["NZ"]),
        "ARG": SiteDefinition(residueName: "ARG", kind: .base, groupType: "ARG", modelPKa: 12.50, chargeSign: 1, siteAtoms: ["NE", "NH1", "NH2"], managedAtoms: ["NH1"]),
    ]

    static func fractionProtonated(pH: Float, pKa: Float) -> Float {
        1.0 / (1.0 + powf(10.0, pH - pKa))
    }

    static func predictResidueStates(
        atoms: [Atom],
        bonds: [Bond] = [],
        pH: Float = 7.4
    ) -> [ResiduePrediction] {
        guard !atoms.isEmpty else { return [] }

        let residues = groupedResidues(from: atoms)
        let chainBounds = chainTerminalBounds(from: residues)

        var snapshots: [SiteSnapshot] = []
        snapshots.reserveCapacity(residues.count + chainBounds.count * 2)

        for (key, indices) in residues {
            guard let definition = siteDefinitions[key.residueName],
                  let center = siteCenter(atomNames: definition.siteAtoms, residueIndices: indices, atoms: atoms) else {
                continue
            }
            snapshots.append(SiteSnapshot(
                key: key,
                definition: definition,
                center: center
            ))
        }

        for (chainID, bounds) in chainBounds {
            if let nCenter = terminusCenter(chainID: chainID, residueSeq: bounds.minResidueSeq, atomNames: ["N"], residues: residues, atoms: atoms) {
                let key = ResidueKey(chainID: chainID, residueSeq: bounds.minResidueSeq, residueName: "*N-term*")
                snapshots.append(SiteSnapshot(
                    key: key,
                    definition: SiteDefinition(
                        residueName: "*N-term*",
                        kind: .nTerminus,
                        groupType: "N+",
                        modelPKa: PropkaConfig.nTermPKa,
                        chargeSign: 1,
                        siteAtoms: ["N"],
                        managedAtoms: ["N"]
                    ),
                    center: nCenter
                ))
            }

            if let cAtomName = preferredCTerminalAtomName(
                chainID: chainID,
                residueSeq: bounds.maxResidueSeq,
                residues: residues,
                atoms: atoms
            ), let cCenter = terminusCenter(
                chainID: chainID,
                residueSeq: bounds.maxResidueSeq,
                atomNames: [cAtomName],
                residues: residues,
                atoms: atoms
            ) {
                let key = ResidueKey(chainID: chainID, residueSeq: bounds.maxResidueSeq, residueName: "*C-term*")
                snapshots.append(SiteSnapshot(
                    key: key,
                    definition: SiteDefinition(
                        residueName: "*C-term*",
                        kind: .cTerminus,
                        groupType: "COO",
                        modelPKa: PropkaConfig.cTermPKa,
                        chargeSign: -1,
                        siteAtoms: [cAtomName],
                        managedAtoms: ["O", "OXT"]
                    ),
                    center: cCenter
                ))
            }
        }

        // Precompute spatial grid and bond adjacency for O(1) lookups.
        let grid = SpatialGrid(atoms: atoms, cellSize: PropkaConfig.desolvationCutoff)
        let adjacency = BondAdjacency(bonds: bonds, atomCount: atoms.count)

        // Precompute per-snapshot residue indices and numVolume (avoids redundant O(N) scans).
        var snapshotResidueIndicesCache: [Int: [Int]] = [:]
        var snapshotNumVolumeCache: [Int: Int] = [:]
        for (i, snapshot) in snapshots.enumerated() {
            let indices = residueIndices(for: snapshot.key, residues: residues)
            snapshotResidueIndicesCache[i] = indices
            snapshotNumVolumeCache[i] = numVolume(for: snapshot.center, residueIndices: indices, atoms: atoms, grid: grid)
        }

        var hydrogenBondContributions: [ResidueKey: Float] = [:]
        var chargeContributions: [ResidueKey: Float] = [:]

        for i in snapshots.indices {
            for j in snapshots.indices where j < i {
                let lhs = snapshots[i]
                let rhs = snapshots[j]

                let lhsResidueIndices = snapshotResidueIndicesCache[i]!
                let rhsResidueIndices = snapshotResidueIndicesCache[j]!
                let lhsNumVolume = snapshotNumVolumeCache[i]!
                let rhsNumVolume = snapshotNumVolumeCache[j]!

                if let hbond = pairwiseHydrogenBondContribution(
                    lhs: lhs,
                    lhsResidueIndices: lhsResidueIndices,
                    rhs: rhs,
                    rhsResidueIndices: rhsResidueIndices,
                    lhsNumVolume: lhsNumVolume,
                    rhsNumVolume: rhsNumVolume,
                    atoms: atoms,
                    bonds: bonds,
                    adjacency: adjacency
                ) {
                    hydrogenBondContributions[lhs.key, default: 0] += hbond.lhs
                    hydrogenBondContributions[rhs.key, default: 0] += hbond.rhs
                }

                if let electrostatic = pairwiseChargeContribution(
                    lhs: lhs,
                    rhs: rhs,
                    lhsNumVolume: lhsNumVolume,
                    rhsNumVolume: rhsNumVolume
                ) {
                    chargeContributions[lhs.key, default: 0] += electrostatic.lhs
                    chargeContributions[rhs.key, default: 0] += electrostatic.rhs
                }
            }
        }

        if !bonds.isEmpty {
            for (i, snapshot) in snapshots.enumerated() {
                let snapshotResidueIndices = snapshotResidueIndicesCache[i]!
                let contribution = environmentalHydrogenBondContribution(
                    for: snapshot,
                    residueIndices: snapshotResidueIndices,
                    residues: residues,
                    atoms: atoms,
                    bonds: bonds,
                    adjacency: adjacency
                )
                if abs(contribution) > 0.0001 {
                    hydrogenBondContributions[snapshot.key, default: 0] += contribution
                }
            }
        }

        var predictions: [ResiduePrediction] = []
        predictions.reserveCapacity(snapshots.count)

        for snapshot in snapshots.sorted(by: residueOrder) {
            let residueIndices = residueIndices(for: snapshot.key, residues: residues)
            let burial = desolvationContribution(
                for: snapshot.center,
                chargeSign: snapshot.definition.chargeSign,
                kind: snapshot.definition.kind,
                residueIndices: residueIndices,
                atoms: atoms,
                grid: grid
            )
            let hbond = hydrogenBondContributions[snapshot.key, default: 0]
            let charge = chargeContributions[snapshot.key, default: 0]
            let shiftedPKa = snapshot.definition.modelPKa + burial + hbond + charge
            let state = determineState(
                for: snapshot,
                shiftedPKa: shiftedPKa,
                pH: pH,
                residueIndices: residueIndices,
                atoms: atoms
            )
            let atomFormalCharges = formalCharges(
                for: snapshot,
                state: state,
                residueIndices: residueIndices,
                atoms: atoms
            )
            let protonatedAtoms = protonatedAtomNames(
                for: snapshot,
                state: state,
                residueIndices: residueIndices,
                atoms: atoms
            )

            predictions.append(ResiduePrediction(
                chainID: snapshot.key.chainID,
                residueSeq: snapshot.key.residueSeq,
                residueName: snapshot.key.residueName,
                chemicalKind: snapshot.definition.kind,
                state: state,
                modelPKa: snapshot.definition.modelPKa,
                shiftedPKa: shiftedPKa,
                burialContribution: burial,
                hydrogenBondContribution: hbond,
                chargeContribution: charge,
                managedAtomNames: snapshot.definition.managedAtoms,
                atomFormalCharges: atomFormalCharges,
                protonatedAtoms: protonatedAtoms
            ))
        }

        return predictions
    }

    static func predictCharges(
        atoms: [Atom],
        bonds: [Bond] = [],
        pH: Float = 7.4
    ) -> [(atomIndex: Int, charge: Int)] {
        let predictions = predictResidueStates(atoms: atoms, bonds: bonds, pH: pH)
        let keyedPredictions = Dictionary(uniqueKeysWithValues: predictions.map {
            (ResidueKey(chainID: $0.chainID, residueSeq: $0.residueSeq, residueName: $0.residueName), $0)
        })

        var results: [(atomIndex: Int, charge: Int)] = []
        results.reserveCapacity(atoms.count)

        for (index, atom) in atoms.enumerated() {
            let atomName = normalize(atom.name)
            let directKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: normalize(atom.residueName))
            let nTermKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: "*N-term*")
            let cTermKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: "*C-term*")

            for key in [directKey, nTermKey, cTermKey] {
                guard let prediction = keyedPredictions[key], prediction.managedAtomNames.contains(atomName) else { continue }
                results.append((atomIndex: index, charge: prediction.atomFormalCharges[atomName] ?? 0))
                break
            }
        }

        return results
    }

    static func applyProtonation(
        atoms: [Atom],
        bonds: [Bond] = [],
        pH: Float = 7.4
    ) -> [Atom] {
        applyProtonation(atoms: atoms, predictions: predictResidueStates(atoms: atoms, bonds: bonds, pH: pH))
    }

    static func applyProtonation(atoms: [Atom], predictions: [ResiduePrediction]) -> [Atom] {
        let keyedPredictions = Dictionary(uniqueKeysWithValues: predictions.map {
            (ResidueKey(chainID: $0.chainID, residueSeq: $0.residueSeq, residueName: $0.residueName), $0)
        })

        return atoms.map { atom in
            let atomName = normalize(atom.name)
            let directKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: normalize(atom.residueName))
            let nTermKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: "*N-term*")
            let cTermKey = ResidueKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: "*C-term*")

            for key in [directKey, nTermKey, cTermKey] {
                guard let prediction = keyedPredictions[key],
                      prediction.managedAtomNames.contains(atomName) else {
                    continue
                }

                let formalCharge = prediction.atomFormalCharges[atomName] ?? 0
                var updated = atom
                updated.formalCharge = formalCharge
                updated.charge = Float(formalCharge)
                return updated
            }

            return atom
        }
    }

    private static func groupedResidues(from atoms: [Atom]) -> [ResidueKey: [Int]] {
        var grouped: [ResidueKey: [Int]] = [:]
        for index in atoms.indices {
            let atom = atoms[index]
            guard !atom.isHetAtom || atom.residueName == "ACE" || atom.residueName == "NME" else { continue }
            let key = ResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: normalize(atom.residueName)
            )
            grouped[key, default: []].append(index)
        }
        return grouped
    }

    private static func chainTerminalBounds(from residues: [ResidueKey: [Int]]) -> [String: (minResidueSeq: Int, maxResidueSeq: Int)] {
        var bounds: [String: (minResidueSeq: Int, maxResidueSeq: Int)] = [:]
        for key in residues.keys where !key.residueName.hasPrefix("*") && key.residueName != "ACE" && key.residueName != "NME" {
            if let current = bounds[key.chainID] {
                bounds[key.chainID] = (
                    min(current.minResidueSeq, key.residueSeq),
                    max(current.maxResidueSeq, key.residueSeq)
                )
            } else {
                bounds[key.chainID] = (key.residueSeq, key.residueSeq)
            }
        }
        return bounds
    }

    private static func residueIndices(
        for key: ResidueKey,
        residues: [ResidueKey: [Int]]
    ) -> [Int] {
        if let indices = residues[key] {
            return indices
        }
        if key.residueName.hasPrefix("*") {
            return residues.first {
                $0.key.chainID == key.chainID &&
                $0.key.residueSeq == key.residueSeq &&
                !$0.key.residueName.hasPrefix("*")
            }?.value ?? []
        }
        return []
    }

    private static func siteCenter(
        atomNames: [String],
        residueIndices: [Int],
        atoms: [Atom]
    ) -> SIMD3<Float>? {
        let wanted = Set(atomNames.map(normalize))
        let positions = residueIndices.compactMap { index -> SIMD3<Float>? in
            let atom = atoms[index]
            return wanted.contains(normalize(atom.name)) ? atom.position : nil
        }
        guard !positions.isEmpty else { return nil }
        return positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
    }

    private static func terminusCenter(
        chainID: String,
        residueSeq: Int,
        atomNames: [String],
        residues: [ResidueKey: [Int]],
        atoms: [Atom]
    ) -> SIMD3<Float>? {
        for (key, indices) in residues where key.chainID == chainID && key.residueSeq == residueSeq {
            if let center = siteCenter(atomNames: atomNames, residueIndices: indices, atoms: atoms) {
                return center
            }
        }
        return nil
    }

    private static func preferredCTerminalAtomName(
        chainID: String,
        residueSeq: Int,
        residues: [ResidueKey: [Int]],
        atoms: [Atom]
    ) -> String? {
        let residueIndices = residues.first { $0.key.chainID == chainID && $0.key.residueSeq == residueSeq }?.value ?? []
        let atomNames = Set(residueIndices.map { normalize(atoms[$0].name) })
        if atomNames.contains("OXT") { return "OXT" }
        if atomNames.contains("O") { return "O" }
        return nil
    }

    private static func desolvationContribution(
        for center: SIMD3<Float>,
        chargeSign: Int,
        kind: ChemicalKind,
        residueIndices: [Int],
        atoms: [Atom],
        grid: SpatialGrid
    ) -> Float {
        let residueSet = Set(residueIndices)
        let minDistanceFourth = powf(PropkaConfig.minDistance, 4)

        var volume: Float = 0
        var buriedHeavyAtomCount = 0

        // Use spatial grid: query at the larger radius, then filter for both cutoffs.
        let nearby = grid.query(center: center, radius: PropkaConfig.desolvationCutoff, atoms: atoms)
        for index in nearby {
            guard !residueSet.contains(index) else { continue }
            let atom = atoms[index]
            let distance = simd_distance(atom.position, center)
            if distance < PropkaConfig.desolvationCutoff {
                let symbol = atom.element.symbol.uppercased()
                let atomName = normalize(atom.name)
                let volumeKey = (symbol == "C" && atomName != "CA" && atomName != "C") ? "C4" : symbol
                let dvol = PropkaConfig.vanDerWaalsVolume[volumeKey, default: 1.0]
                let distanceFourth = max(minDistanceFourth, powf(distance, 4))
                volume += dvol / distanceFourth
            }
            if distance < PropkaConfig.buriedCutoff {
                buriedHeavyAtomCount += 1
            }
        }

        let weight = calculateWeight(numVolume: buriedHeavyAtomCount)
        let scaleFactor = 1.0 - (1.0 - PropkaConfig.desolvationSurfaceScalingFactor) * (1.0 - weight)
        let volumeAfterAllowance = max(0, volume - PropkaConfig.desolvationAllowance)
        let contribution = Float(chargeSign) * PropkaConfig.desolvationPrefactor * volumeAfterAllowance * scaleFactor

        switch kind {
        case .acid, .base, .histidine, .nTerminus, .cTerminus:
            return contribution
        }
    }

    private static func numVolume(
        for center: SIMD3<Float>,
        residueIndices: [Int],
        atoms: [Atom],
        grid: SpatialGrid
    ) -> Int {
        let residueSet = Set(residueIndices)
        let nearby = grid.query(center: center, radius: PropkaConfig.buriedCutoff, atoms: atoms)
        return nearby.reduce(into: 0) { count, index in
            if !residueSet.contains(index) {
                count += 1
            }
        }
    }

    private static func calculateWeight(numVolume: Int) -> Float {
        let raw = Float(numVolume - PropkaConfig.nmin) / Float(PropkaConfig.nmax - PropkaConfig.nmin)
        return min(1, max(0, raw))
    }

    private static func calculatePairWeight(lhsNumVolume: Int, rhsNumVolume: Int) -> Float {
        let raw = Float(lhsNumVolume + rhsNumVolume - 2 * PropkaConfig.nmin) / Float(2 * (PropkaConfig.nmax - PropkaConfig.nmin))
        return min(1, max(0, raw))
    }

    private static func pairwiseHydrogenBondContribution(
        lhs: SiteSnapshot,
        lhsResidueIndices: [Int],
        rhs: SiteSnapshot,
        rhsResidueIndices: [Int],
        lhsNumVolume: Int,
        rhsNumVolume: Int,
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> (lhs: Float, rhs: Float)? {
        guard let cutoff = hydrogenBondCutoff(lhsType: lhs.definition.groupType, rhsType: rhs.definition.groupType),
              let value = pairwiseHydrogenBondValue(
                lhs: lhs,
                lhsResidueIndices: lhsResidueIndices,
                rhs: rhs,
                rhsResidueIndices: rhsResidueIndices,
                lhsNumVolume: lhsNumVolume,
                rhsNumVolume: rhsNumVolume,
                cutoffs: cutoff,
                atoms: atoms,
                bonds: bonds,
                adjacency: adjacency
              ),
              value > 0 else {
            return nil
        }

        let lhsCharge = lhs.definition.chargeSign
        let rhsCharge = rhs.definition.chargeSign

        if lhsCharge == rhsCharge {
            if lhs.definition.modelPKa < rhs.definition.modelPKa {
                return (-value, value)
            }
            return (value, -value)
        }

        return (value * Float(lhsCharge), value * Float(rhsCharge))
    }

    private static func pairwiseChargeContribution(
        lhs: SiteSnapshot,
        rhs: SiteSnapshot,
        lhsNumVolume: Int,
        rhsNumVolume: Int
    ) -> (lhs: Float, rhs: Float)? {
        guard lhsNumVolume + rhsNumVolume >= PropkaConfig.nmin else { return nil }

        let distance = simd_distance(lhs.center, rhs.center)
        guard distance <= PropkaConfig.coulombCutoff2 else { return nil }

        let value = coulombEnergy(distance: distance, weight: calculatePairWeight(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume))
        let lhsCharge = lhs.definition.chargeSign
        let rhsCharge = rhs.definition.chargeSign

        if lhsCharge < 0 && rhsCharge < 0 {
            if lhs.definition.modelPKa > rhs.definition.modelPKa {
                return (value, 0)
            }
            return (0, value)
        }

        if lhsCharge > 0 && rhsCharge > 0 {
            if lhs.definition.modelPKa < rhs.definition.modelPKa {
                return (-value, 0)
            }
            return (0, -value)
        }

        return (value * Float(lhsCharge), value * Float(rhsCharge))
    }

    private static func hydrogenBondCutoff(lhsType: String, rhsType: String) -> SIMD2<Float>? {
        PropkaConfig.hydrogenBondCutoffs[PropkaConfig.pairKey(lhsType, rhsType)]
    }

    private struct NeutralHydrogenBondGroup {
        let groupType: String
        let acidHeavyAtoms: [String]
        let acidHydrogenParents: [String]
        let baseHeavyAtoms: [String]
    }

    private struct InteractionAtom {
        let identifier: String
        let position: SIMD3<Float>
        let element: Element
        let bondedHeavyPosition: SIMD3<Float>?
    }

    private struct InteractionPair {
        let lhs: InteractionAtom
        let rhs: InteractionAtom
        let distance: Float
        let angleFactor: Float
    }

    private static func environmentalHydrogenBondContribution(
        for snapshot: SiteSnapshot,
        residueIndices: [Int],
        residues: [ResidueKey: [Int]],
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> Float {
        var contribution: Float = 0

        // Precompute snapshot interaction atoms per unique group type to avoid recomputation.
        var snapshotAtomsCache: [String: [InteractionAtom]] = [:]

        for (key, environmentResidueIndices) in residues {
            guard key != snapshot.key,
                  let environmentGroup = neutralHydrogenBondGroup(for: key.residueName),
                  let cutoffs = hydrogenBondCutoff(
                    lhsType: snapshot.definition.groupType,
                    rhsType: environmentGroup.groupType
                  ) else {
                continue
            }

            // Early distance pruning: skip residues whose centers are too far apart.
            // Max cutoff across all pair types is ~5 Å; residue atoms are typically within ~5 Å of center.
            let envCenter = siteCenter(atomNames: environmentGroup.acidHeavyAtoms + environmentGroup.baseHeavyAtoms,
                                       residueIndices: environmentResidueIndices, atoms: atoms)
            if let ec = envCenter, simd_distance(snapshot.center, ec) > cutoffs.y + 10.0 {
                continue
            }

            let snapshotAtoms = snapshotAtomsCache[environmentGroup.groupType] ?? {
                let result = interactionAtoms(
                    for: snapshot,
                    residueIndices: residueIndices,
                    atoms: atoms,
                    bonds: bonds,
                    interactingWith: environmentGroup.groupType,
                    adjacency: adjacency
                )
                snapshotAtomsCache[environmentGroup.groupType] = result
                return result
            }()
            let environmentAtoms = interactionAtoms(
                for: environmentGroup,
                residueIndices: environmentResidueIndices,
                atoms: atoms,
                bonds: bonds,
                interactingWith: snapshot.definition.groupType,
                adjacency: adjacency
            )

            if let interaction = bestInteraction(
                lhsAtoms: snapshotAtoms,
                lhsGroupType: snapshot.definition.groupType,
                rhsAtoms: environmentAtoms,
                rhsGroupType: environmentGroup.groupType
            ), interaction.distance < cutoffs.y {
                contribution += Float(snapshot.definition.chargeSign) * hydrogenBondEnergy(
                    distance: interaction.distance,
                    maxShift: PropkaConfig.sidechainInteraction,
                    cutoffs: cutoffs,
                    angleFactor: interaction.angleFactor
                )
            }
        }

        contribution += backboneHydrogenBondContribution(
            for: snapshot,
            residueIndices: residueIndices,
            residues: residues,
            atoms: atoms,
            bonds: bonds,
            adjacency: adjacency
        )

        return contribution
    }

    private static func neutralHydrogenBondGroup(for residueName: String) -> NeutralHydrogenBondGroup? {
        switch residueName {
        case "SER":
            return NeutralHydrogenBondGroup(
                groupType: "ROH",
                acidHeavyAtoms: ["OG"],
                acidHydrogenParents: [],
                baseHeavyAtoms: ["OG"]
            )
        case "THR":
            return NeutralHydrogenBondGroup(
                groupType: "ROH",
                acidHeavyAtoms: ["OG1"],
                acidHydrogenParents: [],
                baseHeavyAtoms: ["OG1"]
            )
        case "ASN":
            return NeutralHydrogenBondGroup(
                groupType: "AMD",
                acidHeavyAtoms: ["ND2"],
                acidHydrogenParents: ["ND2"],
                baseHeavyAtoms: ["OD1"]
            )
        case "GLN":
            return NeutralHydrogenBondGroup(
                groupType: "AMD",
                acidHeavyAtoms: ["NE2"],
                acidHydrogenParents: ["NE2"],
                baseHeavyAtoms: ["OE1"]
            )
        case "TRP":
            return NeutralHydrogenBondGroup(
                groupType: "TRP",
                acidHeavyAtoms: ["NE1"],
                acidHydrogenParents: ["NE1"],
                baseHeavyAtoms: ["NE1"]
            )
        default:
            return nil
        }
    }

    private static func backboneHydrogenBondContribution(
        for snapshot: SiteSnapshot,
        residueIndices: [Int],
        residues: [ResidueKey: [Int]],
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> Float {
        var contribution: Float = 0

        if let parameters = PropkaConfig.backboneNHHydrogenBondParameters[snapshot.definition.groupType] {
            let backboneGroup = NeutralHydrogenBondGroup(
                groupType: "BBN",
                acidHeavyAtoms: ["N"],
                acidHydrogenParents: ["N"],
                baseHeavyAtoms: ["N"]
            )

            // Precompute snapshot atoms once outside the loop (same for all backbone residues).
            let snapshotAtoms = interactionAtoms(
                for: snapshot,
                residueIndices: residueIndices,
                atoms: atoms,
                bonds: bonds,
                interactingWith: backboneGroup.groupType,
                adjacency: adjacency
            )

            // Distance threshold for early pruning: cutoff + margin for atom offset from center.
            let bbNHPruneDistance = parameters.cutoffs.y + 8.0

            for (key, backboneResidueIndices) in residues {
                guard key != snapshot.key,
                      isBackboneResidue(key.residueName),
                      key.residueName != "PRO" else {
                    continue
                }

                // Early distance pruning using backbone N position.
                if let nIdx = backboneResidueIndices.first(where: { normalize(atoms[$0].name) == "N" }),
                   simd_distance(snapshot.center, atoms[nIdx].position) > bbNHPruneDistance {
                    continue
                }

                let backboneAtoms = interactionAtoms(
                    for: backboneGroup,
                    residueIndices: backboneResidueIndices,
                    atoms: atoms,
                    bonds: bonds,
                    interactingWith: snapshot.definition.groupType,
                    adjacency: adjacency
                )

                guard let interaction = bestInteraction(
                    lhsAtoms: snapshotAtoms,
                    lhsGroupType: snapshot.definition.groupType,
                    rhsAtoms: backboneAtoms,
                    rhsGroupType: backboneGroup.groupType
                ), interaction.distance < parameters.cutoffs.y else {
                    continue
                }

                contribution += Float(snapshot.definition.chargeSign) * hydrogenBondEnergy(
                    distance: interaction.distance,
                    maxShift: parameters.maxShift,
                    cutoffs: parameters.cutoffs,
                    angleFactor: interaction.angleFactor
                )
            }
        }

        if snapshot.definition.groupType == "HIS",
           let parameters = PropkaConfig.backboneCOHydrogenBondParameters[snapshot.definition.groupType] {
            let backboneGroup = NeutralHydrogenBondGroup(
                groupType: "BBC",
                acidHeavyAtoms: ["O"],
                acidHydrogenParents: [],
                baseHeavyAtoms: ["O"]
            )

            // Precompute snapshot atoms once outside the loop.
            let snapshotAtoms = interactionAtoms(
                for: snapshot,
                residueIndices: residueIndices,
                atoms: atoms,
                bonds: bonds,
                interactingWith: backboneGroup.groupType,
                adjacency: adjacency
            )

            let bbCOPruneDistance = parameters.cutoffs.y + 8.0

            for (key, backboneResidueIndices) in residues {
                guard key != snapshot.key,
                      isBackboneResidue(key.residueName) else {
                    continue
                }

                // Early distance pruning using backbone O position.
                if let oIdx = backboneResidueIndices.first(where: { normalize(atoms[$0].name) == "O" }),
                   simd_distance(snapshot.center, atoms[oIdx].position) > bbCOPruneDistance {
                    continue
                }

                let backboneAtoms = interactionAtoms(
                    for: backboneGroup,
                    residueIndices: backboneResidueIndices,
                    atoms: atoms,
                    bonds: bonds,
                    interactingWith: snapshot.definition.groupType,
                    adjacency: adjacency
                )

                guard let interaction = bestInteraction(
                    lhsAtoms: snapshotAtoms,
                    lhsGroupType: snapshot.definition.groupType,
                    rhsAtoms: backboneAtoms,
                    rhsGroupType: backboneGroup.groupType
                ), interaction.distance < parameters.cutoffs.y else {
                    continue
                }

                contribution += Float(snapshot.definition.chargeSign) * hydrogenBondEnergy(
                    distance: interaction.distance,
                    maxShift: parameters.maxShift,
                    cutoffs: parameters.cutoffs,
                    angleFactor: interaction.angleFactor
                )
            }
        }

        return contribution
    }

    private static func pairwiseHydrogenBondValue(
        lhs: SiteSnapshot,
        lhsResidueIndices: [Int],
        rhs: SiteSnapshot,
        rhsResidueIndices: [Int],
        lhsNumVolume: Int,
        rhsNumVolume: Int,
        cutoffs: SIMD2<Float>,
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> Float? {
        if let exceptionValue = exceptionalHydrogenBondValue(
            lhs: lhs,
            lhsResidueIndices: lhsResidueIndices,
            rhs: rhs,
            rhsResidueIndices: rhsResidueIndices,
            lhsNumVolume: lhsNumVolume,
            rhsNumVolume: rhsNumVolume,
            cutoffs: cutoffs,
            atoms: atoms,
            bonds: bonds,
            adjacency: adjacency
        ) {
            return exceptionValue
        }

        let lhsAtoms = interactionAtoms(
            for: lhs,
            residueIndices: lhsResidueIndices,
            atoms: atoms,
            bonds: bonds,
            interactingWith: rhs.definition.groupType,
            adjacency: adjacency
        )
        let rhsAtoms = interactionAtoms(
            for: rhs,
            residueIndices: rhsResidueIndices,
            atoms: atoms,
            bonds: bonds,
            interactingWith: lhs.definition.groupType,
            adjacency: adjacency
        )

        guard let interaction = bestInteraction(
            lhsAtoms: lhsAtoms,
            lhsGroupType: lhs.definition.groupType,
            rhsAtoms: rhsAtoms,
            rhsGroupType: rhs.definition.groupType
        ), interaction.distance < cutoffs.y else {
            return nil
        }

        return hydrogenBondEnergy(
            distance: interaction.distance,
            maxShift: PropkaConfig.sidechainInteraction,
            cutoffs: cutoffs,
            angleFactor: interaction.angleFactor
        )
    }

    private static func exceptionalHydrogenBondValue(
        lhs: SiteSnapshot,
        lhsResidueIndices: [Int],
        rhs: SiteSnapshot,
        rhsResidueIndices: [Int],
        lhsNumVolume: Int,
        rhsNumVolume: Int,
        cutoffs: SIMD2<Float>,
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> Float? {
        let pairTypes = Set([lhs.definition.groupType, rhs.definition.groupType])
        let pairWeight = calculatePairWeight(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume)

        if pairTypes == Set(["COO", "ARG"]) {
            let coo: (snapshot: SiteSnapshot, residueIndices: [Int]) = lhs.definition.groupType == "COO"
                ? (lhs, lhsResidueIndices)
                : (rhs, rhsResidueIndices)
            let arg: (snapshot: SiteSnapshot, residueIndices: [Int]) = lhs.definition.groupType == "ARG"
                ? (lhs, lhsResidueIndices)
                : (rhs, rhsResidueIndices)
            let value = cooArgExceptionValue(
                coo: coo.snapshot,
                cooResidueIndices: coo.residueIndices,
                arg: arg.snapshot,
                argResidueIndices: arg.residueIndices,
                cutoffs: cutoffs,
                atoms: atoms,
                bonds: bonds,
                adjacency: adjacency
            )
            return value > 0 ? value : nil
        }

        if pairTypes == Set(["COO", "COO"]) {
            let lhsAtoms = interactionAtoms(
                for: lhs,
                residueIndices: lhsResidueIndices,
                atoms: atoms,
                bonds: bonds,
                interactingWith: rhs.definition.groupType,
                adjacency: adjacency
            )
            let rhsAtoms = interactionAtoms(
                for: rhs,
                residueIndices: rhsResidueIndices,
                atoms: atoms,
                bonds: bonds,
                interactingWith: lhs.definition.groupType,
                adjacency: adjacency
            )

            guard let interaction = bestInteraction(
                lhsAtoms: lhsAtoms,
                lhsGroupType: lhs.definition.groupType,
                rhsAtoms: rhsAtoms,
                rhsGroupType: rhs.definition.groupType
            ), interaction.distance < cutoffs.y else {
                return nil
            }

            return hydrogenBondEnergy(
                distance: interaction.distance,
                maxShift: PropkaConfig.sidechainInteraction,
                cutoffs: cutoffs,
                angleFactor: interaction.angleFactor
            ) * (1 + pairWeight)
        }

        if pairTypes == Set(["COO", "HIS"]),
           isBuriedInteraction(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume) {
            return PropkaConfig.cooHisException
        }

        if pairTypes == Set(["CYS", "HIS"]),
           isBuriedInteraction(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume) {
            return PropkaConfig.cysHisException
        }

        if pairTypes == Set(["CYS", "CYS"]),
           isBuriedInteraction(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume) {
            return PropkaConfig.cysCysException
        }

        return nil
    }

    private static func cooArgExceptionValue(
        coo: SiteSnapshot,
        cooResidueIndices: [Int],
        arg: SiteSnapshot,
        argResidueIndices: [Int],
        cutoffs: SIMD2<Float>,
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency
    ) -> Float {
        var cooAtoms = interactionAtoms(
            for: coo,
            residueIndices: cooResidueIndices,
            atoms: atoms,
            bonds: bonds,
            interactingWith: arg.definition.groupType,
            adjacency: adjacency
        )
        var argAtoms = interactionAtoms(
            for: arg,
            residueIndices: argResidueIndices,
            atoms: atoms,
            bonds: bonds,
            interactingWith: coo.definition.groupType,
            adjacency: adjacency
        )

        var total: Float = 0
        for _ in 0..<2 {
            guard let interaction = bestInteraction(
                lhsAtoms: cooAtoms,
                lhsGroupType: coo.definition.groupType,
                rhsAtoms: argAtoms,
                rhsGroupType: arg.definition.groupType
            ) else {
                break
            }

            total += hydrogenBondEnergy(
                distance: interaction.distance,
                maxShift: PropkaConfig.sidechainInteraction,
                cutoffs: cutoffs,
                angleFactor: interaction.angleFactor
            )
            cooAtoms.removeAll { $0.identifier == interaction.lhs.identifier }
            argAtoms.removeAll { $0.identifier == interaction.rhs.identifier }
        }

        return total
    }

    private static func interactionAtoms(
        for snapshot: SiteSnapshot,
        residueIndices: [Int],
        atoms: [Atom],
        bonds: [Bond],
        interactingWith otherGroupType: String,
        adjacency: BondAdjacency? = nil
    ) -> [InteractionAtom] {
        switch snapshot.definition.groupType {
        case "ARG", "HIS":
            return interactionAtoms(
                residueIndices: residueIndices,
                acidHeavyAtoms: snapshot.definition.siteAtoms,
                acidHydrogenParents: snapshot.definition.siteAtoms,
                baseHeavyAtoms: snapshot.definition.siteAtoms,
                atoms: atoms,
                bonds: bonds,
                interactingWith: otherGroupType,
                adjacency: adjacency
            )
        default:
            return interactionAtoms(
                residueIndices: residueIndices,
                acidHeavyAtoms: snapshot.definition.siteAtoms,
                acidHydrogenParents: [],
                baseHeavyAtoms: snapshot.definition.siteAtoms,
                atoms: atoms,
                bonds: bonds,
                interactingWith: otherGroupType,
                adjacency: adjacency
            )
        }
    }

    private static func interactionAtoms(
        for group: NeutralHydrogenBondGroup,
        residueIndices: [Int],
        atoms: [Atom],
        bonds: [Bond],
        interactingWith otherGroupType: String,
        adjacency: BondAdjacency? = nil
    ) -> [InteractionAtom] {
        interactionAtoms(
            residueIndices: residueIndices,
            acidHeavyAtoms: group.acidHeavyAtoms,
            acidHydrogenParents: group.acidHydrogenParents,
            baseHeavyAtoms: group.baseHeavyAtoms,
            atoms: atoms,
            bonds: bonds,
            interactingWith: otherGroupType,
            adjacency: adjacency
        )
    }

    private static func interactionAtoms(
        residueIndices: [Int],
        acidHeavyAtoms: [String],
        acidHydrogenParents: [String],
        baseHeavyAtoms: [String],
        atoms: [Atom],
        bonds: [Bond],
        interactingWith otherGroupType: String,
        adjacency: BondAdjacency? = nil
    ) -> [InteractionAtom] {
        let useBaseAtoms = PropkaConfig.baseGroupTypes.contains(otherGroupType)
        let heavyAtomNames = Set((useBaseAtoms ? baseHeavyAtoms : acidHeavyAtoms).map(normalize))
        let hydrogenParentNames = useBaseAtoms ? Set<String>() : Set(acidHydrogenParents.map(normalize))

        var result: [InteractionAtom] = []
        for residueIndex in residueIndices {
            let atomName = normalize(atoms[residueIndex].name)
            guard heavyAtomNames.contains(atomName) else { continue }

            result.append(InteractionAtom(
                identifier: "atom:\(residueIndex)",
                position: atoms[residueIndex].position,
                element: atoms[residueIndex].element,
                bondedHeavyPosition: nil
            ))

            guard hydrogenParentNames.contains(atomName) else { continue }
            result.append(contentsOf: donorHydrogenInteractionAtoms(
                for: residueIndex,
                atoms: atoms,
                bonds: bonds,
                adjacency: adjacency
            ))
        }

        return result
    }

    private static func bestInteraction(
        lhsAtoms: [InteractionAtom],
        lhsGroupType: String,
        rhsAtoms: [InteractionAtom],
        rhsGroupType: String
    ) -> InteractionPair? {
        guard !lhsAtoms.isEmpty, !rhsAtoms.isEmpty else { return nil }

        var bestPair: (lhs: InteractionAtom, rhs: InteractionAtom, distance: Float)?
        for lhsAtom in lhsAtoms {
            for rhsAtom in rhsAtoms {
                let distance = simd_distance(lhsAtom.position, rhsAtom.position)
                guard distance > 0.1 else { continue }
                if let current = bestPair {
                    if distance < current.distance {
                        bestPair = (lhsAtom, rhsAtom, distance)
                    }
                } else {
                    bestPair = (lhsAtom, rhsAtom, distance)
                }
            }
        }

        guard let bestPair else { return nil }
        let angleFactor = interactionAngleFactor(
            lhsAtom: bestPair.lhs,
            lhsGroupType: lhsGroupType,
            rhsAtom: bestPair.rhs,
            rhsGroupType: rhsGroupType
        )

        if requiresAngularFactor(groupType: lhsGroupType) || requiresAngularFactor(groupType: rhsGroupType) {
            guard angleFactor > PropkaConfig.angularFactorMinimum else {
                return nil
            }
        }

        return InteractionPair(
            lhs: bestPair.lhs,
            rhs: bestPair.rhs,
            distance: bestPair.distance,
            angleFactor: angleFactor
        )
    }

    private static func donorHydrogenInteractionAtoms(
        for donorAtomIndex: Int,
        atoms: [Atom],
        bonds: [Bond],
        adjacency: BondAdjacency? = nil
    ) -> [InteractionAtom] {
        let bondedIndices = adjacency?.bonded(to: donorAtomIndex) ?? bondedAtomIndices(of: donorAtomIndex, bonds: bonds)
        let explicitHydrogens = bondedIndices.compactMap { bondedIndex -> InteractionAtom? in
            guard atoms[bondedIndex].element == .H else { return nil }
            return InteractionAtom(
                identifier: "hydrogen:\(bondedIndex)",
                position: atoms[bondedIndex].position,
                element: .H,
                bondedHeavyPosition: atoms[donorAtomIndex].position
            )
        }
        if !explicitHydrogens.isEmpty {
            return explicitHydrogens
        }

        let donorAtom = atoms[donorAtomIndex]
        let neighborPositions = bondedIndices.compactMap { bondedIndex -> SIMD3<Float>? in
            atoms[bondedIndex].element == .H ? nil : atoms[bondedIndex].position
        }
        let bondLength = PropkaConfig.protonBondLengths[donorAtom.element.symbol.uppercased(), default: 1.01]
        let position = placeVirtualHydrogen(on: donorAtom.position, awayFrom: neighborPositions, distance: bondLength)
        return [InteractionAtom(
            identifier: "virtual-hydrogen:\(donorAtomIndex)",
            position: position,
            element: .H,
            bondedHeavyPosition: donorAtom.position
        )]
    }

    private static func bondedAtomIndices(of atomIndex: Int, bonds: [Bond]) -> [Int] {
        bonds.compactMap { bond in
            if bond.atomIndex1 == atomIndex {
                return bond.atomIndex2
            }
            if bond.atomIndex2 == atomIndex {
                return bond.atomIndex1
            }
            return nil
        }
    }

    private static func placeVirtualHydrogen(
        on center: SIMD3<Float>,
        awayFrom neighbors: [SIMD3<Float>],
        distance: Float
    ) -> SIMD3<Float> {
        if neighbors.isEmpty {
            return center + SIMD3<Float>(0, distance, 0)
        }

        let average = neighbors.reduce(SIMD3<Float>.zero) { partial, neighbor in
            partial + (neighbor - center)
        } / Float(neighbors.count)

        let length = simd_length(average)
        if length < 1e-6 {
            return center + SIMD3<Float>(0, distance, 0)
        }

        return center + (-average / length) * distance
    }

    private static func interactionAngleFactor(
        lhsAtom: InteractionAtom,
        lhsGroupType: String,
        rhsAtom: InteractionAtom,
        rhsGroupType: String
    ) -> Float {
        if let factor = angleFactorIfAvailable(
            donorAtom: rhsAtom,
            donorGroupType: rhsGroupType,
            acceptorPosition: lhsAtom.position
        ) {
            return factor
        }
        if let factor = angleFactorIfAvailable(
            donorAtom: lhsAtom,
            donorGroupType: lhsGroupType,
            acceptorPosition: rhsAtom.position
        ) {
            return factor
        }
        if requiresAngularFactor(groupType: lhsGroupType) || requiresAngularFactor(groupType: rhsGroupType) {
            return 0
        }
        return 1
    }

    private static func angleFactorIfAvailable(
        donorAtom: InteractionAtom,
        donorGroupType: String,
        acceptorPosition: SIMD3<Float>
    ) -> Float? {
        guard requiresAngularFactor(groupType: donorGroupType),
              donorAtom.element == .H,
              let donorPosition = donorAtom.bondedHeavyPosition else {
            return nil
        }

        return donorAngleFactor(
            acceptorPosition: acceptorPosition,
            hydrogenPosition: donorAtom.position,
            donorPosition: donorPosition
        )
    }

    private static func donorAngleFactor(
        acceptorPosition: SIMD3<Float>,
        hydrogenPosition: SIMD3<Float>,
        donorPosition: SIMD3<Float>
    ) -> Float {
        let acceptorVector = simd_normalize(acceptorPosition - hydrogenPosition)
        // Match PROPKA's angle_distance_factors(): the donor vector points
        // from the donor heavy atom toward the hydrogen.
        let donorVector = simd_normalize(hydrogenPosition - donorPosition)
        let factor = simd_dot(acceptorVector, donorVector)
        return max(0, factor)
    }

    private static func requiresAngularFactor(groupType: String) -> Bool {
        switch groupType {
        case "HIS", "ARG", "AMD", "TRP", "BBN":
            return true
        default:
            return false
        }
    }

    private static func isBackboneResidue(_ residueName: String) -> Bool {
        guard ProteinResidueReferenceTemplateStore.template(for: residueName) != nil else {
            return false
        }
        return residueName != "ACE" && residueName != "NME"
    }

    private static func hydrogenBondEnergy(
        distance: Float,
        pairWeight: Float,
        lhsType: String,
        rhsType: String,
        cutoffs: SIMD2<Float>,
        lhsNumVolume: Int,
        rhsNumVolume: Int
    ) -> Float {
        if isBuriedInteraction(lhsNumVolume: lhsNumVolume, rhsNumVolume: rhsNumVolume) {
            switch Set([lhsType, rhsType]) {
            case Set(["COO", "HIS"]):
                return PropkaConfig.cooHisException
            case Set(["CYS", "HIS"]):
                return PropkaConfig.cysHisException
            case Set(["CYS", "CYS"]):
                return PropkaConfig.cysCysException
            default:
                break
            }
        }

        let raw = hydrogenBondEnergy(distance: distance, maxShift: PropkaConfig.sidechainInteraction, cutoffs: cutoffs)
        if lhsType == "COO" && rhsType == "COO" {
            return raw * (1.0 + pairWeight)
        }
        return raw
    }

    private static func hydrogenBondEnergy(
        distance: Float,
        maxShift: Float,
        cutoffs: SIMD2<Float>,
        angleFactor: Float = 1.0
    ) -> Float {
        let value: Float
        if distance < cutoffs.x {
            value = 1
        } else if distance > cutoffs.y {
            value = 0
        } else {
            value = 1 - (distance - cutoffs.x) / (cutoffs.y - cutoffs.x)
        }
        return abs(maxShift * value * angleFactor)
    }

    private static func isBuriedInteraction(lhsNumVolume: Int, rhsNumVolume: Int) -> Bool {
        if lhsNumVolume + rhsNumVolume <= PropkaConfig.combinedBuriedMax &&
            (lhsNumVolume <= PropkaConfig.separateBuriedMax || rhsNumVolume <= PropkaConfig.separateBuriedMax) {
            return false
        }
        return true
    }

    private static func coulombEnergy(distance: Float, weight: Float) -> Float {
        let dielectric = PropkaConfig.dielectric1 - (PropkaConfig.dielectric1 - PropkaConfig.dielectric2) * weight
        let adjustedDistance = max(distance, PropkaConfig.coulombCutoff1)
        let scale = min(1, max(0, (adjustedDistance - PropkaConfig.coulombCutoff2) / (PropkaConfig.coulombCutoff1 - PropkaConfig.coulombCutoff2)))
        return abs(PropkaConfig.pKaScaling1 / (dielectric * adjustedDistance) * scale)
    }

    private static func determineState(
        for snapshot: SiteSnapshot,
        shiftedPKa: Float,
        pH: Float,
        residueIndices: [Int],
        atoms: [Atom]
    ) -> StateKind {
        switch snapshot.definition.kind {
        case .acid, .cTerminus:
            return shiftedPKa > pH ? .protonated : .deprotonated
        case .base, .nTerminus:
            return shiftedPKa > pH ? .protonated : .deprotonated
        case .histidine:
            if shiftedPKa > pH {
                return .histidineDoublyProtonated
            }
            let nd1Score = histidineTautomerScore(
                atomName: "ND1",
                residueIndices: residueIndices,
                atoms: atoms
            )
            let ne2Score = histidineTautomerScore(
                atomName: "NE2",
                residueIndices: residueIndices,
                atoms: atoms
            )
            return nd1Score >= ne2Score ? .histidineDelta : .histidineEpsilon
        }
    }

    /// Ring neighbor atom names for each histidine imidazole nitrogen.
    private static let hisRingNeighbors: [String: [String]] = [
        "ND1": ["CG", "CE1"],
        "NE2": ["CE1", "CD2"]
    ]

    private static func histidineTautomerScore(
        atomName: String,
        residueIndices: [Int],
        atoms: [Atom]
    ) -> Float {
        guard let siteIndex = residueIndices.first(where: { normalize(atoms[$0].name) == atomName }) else {
            return 0
        }
        let site = atoms[siteIndex]
        let residueSet = Set(residueIndices)

        // Compute the approximate N-H direction (away from the ring center).
        // This is: normalize(N_pos - midpoint(neighbor1, neighbor2))
        let neighborNames = hisRingNeighbors[atomName] ?? []
        let neighborPositions: [SIMD3<Float>] = neighborNames.compactMap { name in
            guard let idx = residueIndices.first(where: { normalize(atoms[$0].name) == name }) else {
                return nil
            }
            return atoms[idx].position
        }

        // If we can't find both ring neighbors, fall back to distance-only scoring.
        let hDirection: SIMD3<Float>?
        if neighborPositions.count == 2 {
            let midpoint = (neighborPositions[0] + neighborPositions[1]) * 0.5
            let raw = site.position - midpoint
            let len = simd_length(raw)
            hDirection = len > 1e-6 ? raw / len : nil
        } else {
            hDirection = nil
        }

        return atoms.enumerated().reduce(Float.zero) { partial, entry in
            let (otherIndex, other) = entry
            guard !residueSet.contains(otherIndex), other.element != .H else { return partial }
            let distance = simd_distance(site.position, other.position)
            guard distance > 0.1 && distance < 3.5 else { return partial }

            // Angle filter: candidate must be within 120° of the N-H direction.
            if let hDir = hDirection {
                let toCandidate = simd_normalize(other.position - site.position)
                let cosAngle = simd_dot(toCandidate, hDir)
                // cos(120°) = -0.5; reject atoms behind the imidazole plane
                guard cosAngle > -0.5 else { return partial }
            }

            if isAcceptorLike(other) {
                return partial + max(0.0, 3.5 - distance) / 3.5
            }
            if isDonorLike(other) {
                return partial - 0.15 * max(0.0, 3.5 - distance) / 3.5
            }
            return partial
        }
    }

    private static func formalCharges(
        for snapshot: SiteSnapshot,
        state: StateKind,
        residueIndices: [Int],
        atoms: [Atom]
    ) -> [String: Int] {
        let atomNames = Set(residueIndices.map { normalize(atoms[$0].name) })

        switch snapshot.definition.residueName {
        case "ASP":
            switch state {
            case .protonated:
                return [:]
            case .deprotonated:
                return [selectAvailableAtom(["OD2", "OD1"], available: atomNames): -1]
            default:
                return [:]
            }
        case "GLU":
            switch state {
            case .protonated:
                return [:]
            case .deprotonated:
                return [selectAvailableAtom(["OE2", "OE1"], available: atomNames): -1]
            default:
                return [:]
            }
        case "CYS":
            return state == .deprotonated ? ["SG": -1] : [:]
        case "TYR":
            return state == .deprotonated ? ["OH": -1] : [:]
        case "HIS":
            return state == .histidineDoublyProtonated ? ["ND1": 1] : [:]
        case "LYS":
            return state == .protonated ? ["NZ": 1] : [:]
        case "ARG":
            return state == .protonated ? ["NH1": 1] : [:]
        case "*N-term*":
            return state == .protonated ? ["N": 1] : [:]
        case "*C-term*":
            return state == .deprotonated ? [selectAvailableAtom(["OXT", "O"], available: atomNames): -1] : [:]
        default:
            return [:]
        }
    }

    private static func protonatedAtomNames(
        for snapshot: SiteSnapshot,
        state: StateKind,
        residueIndices: [Int],
        atoms: [Atom]
    ) -> Set<String> {
        let atomNames = Set(residueIndices.map { normalize(atoms[$0].name) })

        switch snapshot.definition.residueName {
        case "ASP":
            return state == .protonated ? [selectAvailableAtom(["OD2", "OD1"], available: atomNames)] : []
        case "GLU":
            return state == .protonated ? [selectAvailableAtom(["OE2", "OE1"], available: atomNames)] : []
        case "CYS":
            return state == .protonated ? ["SG"] : []
        case "TYR":
            return state == .protonated ? ["OH"] : []
        case "HIS":
            switch state {
            case .histidineDelta:
                return ["ND1"]
            case .histidineEpsilon:
                return ["NE2"]
            case .histidineDoublyProtonated:
                return ["ND1", "NE2"]
            default:
                return []
            }
        case "LYS":
            return state == .protonated ? ["NZ"] : []
        case "ARG":
            return state == .protonated ? ["NE", "NH1", "NH2"] : []
        case "*N-term*":
            return state == .protonated ? ["N"] : []
        case "*C-term*":
            return state == .protonated ? [selectAvailableAtom(["OXT", "O"], available: atomNames)] : []
        default:
            return []
        }
    }

    private static func residueOrder(lhs: SiteSnapshot, rhs: SiteSnapshot) -> Bool {
        if lhs.key.chainID != rhs.key.chainID {
            return lhs.key.chainID < rhs.key.chainID
        }
        if lhs.key.residueSeq != rhs.key.residueSeq {
            return lhs.key.residueSeq < rhs.key.residueSeq
        }
        return lhs.key.residueName < rhs.key.residueName
    }

    private static func normalize(_ atomName: String) -> String {
        ProteinResidueTemplateStore.normalizeAtomName(atomName)
    }

    private static func selectAvailableAtom(_ preferred: [String], available: Set<String>) -> String {
        preferred.first(where: { available.contains($0) }) ?? preferred[0]
    }

    private static func isDonorLike(_ atom: Atom) -> Bool {
        let atomName = normalize(atom.name)
        if atom.element == .N {
            return true
        }
        if atom.element == .O || atom.element == .S {
            return atomName == "OG" || atomName == "OG1" || atomName == "OH" || atomName == "SG"
        }
        return false
    }

    private static func isAcceptorLike(_ atom: Atom) -> Bool {
        let atomName = normalize(atom.name)
        if atom.element == .O || atom.element == .S {
            return true
        }
        if atom.element == .N {
            return atomName == "ND1" || atomName == "NE2"
        }
        return false
    }
}

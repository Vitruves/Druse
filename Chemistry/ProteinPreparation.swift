import Foundation
import simd

enum ProteinPreparation {

    struct ChainBreak: Sendable {
        let chainID: String
        let previousResidueSeq: Int
        let nextResidueSeq: Int
        let carbonNitrogenDistance: Float
        let missingResidueCount: Int
        let isCapped: Bool
    }

    enum WaterPolicy: Sendable {
        case keepAll
        case removeAll
        case keepNearby(centers: [SIMD3<Float>], radius: Float)
    }

    struct CleanupOptions: Sendable {
        var removeNonStandardResidues: Bool = true
        var waterPolicy: WaterPolicy = .keepAll
        var capChainBreaks: Bool = true
        var keepExistingCaps: Bool = true
    }

    struct CleanupReport: Sendable {
        var removedAltConformerAtoms: Int = 0
        var removedHeterogenResidues: Int = 0
        var removedHeterogenAtoms: Int = 0
        var removedWaterResidues: Int = 0
        var addedCappingResidues: Int = 0
        var chainBreaks: [ChainBreak] = []
    }

    struct DockingPreparationReport: Sendable {
        var altConformerAtomsRemoved: Int = 0
        var heterogenResiduesRemoved: Int = 0
        var cappingResiduesAdded: Int = 0
        var chainBreaksDetected: Int = 0
        var chainBreaksCapped: Int = 0
        var missingHeavyAtomsAdded: Int = 0
        var protonationUpdates: Int = 0
        var hydrogensAdded: Int = 0
        var hydrogenMethod: String = "none"
        var hbondNetworkReport: HBondNetworkOptimizer.NetworkReport?
        var rdkitChargeMatches: Int = 0
        var fallbackChargeAtoms: Int = 0
        var nonZeroChargeAtoms: Int = 0
    }

    struct PreparationReport: Sendable {
        var waterCount: Int = 0
        var altConfsRemoved: Int = 0
        var missingResidues: [(chainID: String, gapStart: Int, gapEnd: Int)] = []
        var chainBreaks: [ChainBreak] = []
        var residueCompleteness: [ProteinResidueCompleteness] = []
        var chainSummary: [(chainID: String, residueCount: Int, type: String)] = []
        var hetGroups: [(name: String, count: Int)] = []
        var clashCount: Int = 0
        var totalAtoms: Int = 0
        var heavyAtoms: Int = 0
    }

    private static let waterResidueNames: Set<String> = ["HOH", "WAT", "DOD", "H2O"]
    private static let standardResidueNames: Set<String> = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    ]
    private static let cappingResidueNames: Set<String> = ["ACE", "NME"]

    private struct ResidueKey: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let isHetAtom: Bool
    }

    private struct WaterResidueKey: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
    }

    private struct ResidueRecord {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let isHetAtom: Bool
        let atomIndices: [Int]
    }

    // MARK: - Altloc Selection

    static func selectPreferredAltConformers(
        atoms: [Atom],
        bonds: [Bond]
    ) -> (atoms: [Atom], bonds: [Bond], removedAtomCount: Int) {
        var residueIndices: [ResidueKey: [Int]] = [:]
        var residueOrder: [ResidueKey] = []

        for index in atoms.indices {
            let atom = atoms[index]
            let key = ResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: atom.residueName,
                isHetAtom: atom.isHetAtom
            )
            if residueIndices[key] == nil {
                residueOrder.append(key)
            }
            residueIndices[key, default: []].append(index)
        }

        var keptIndices = Set<Int>()

        for key in residueOrder {
            guard let indices = residueIndices[key] else { continue }

            struct AltScore {
                var occupancyTotal: Float = 0
                var occupancyCount = 0
                var bFactorTotal: Float = 0
                var atomCount = 0

                var averageOccupancy: Float {
                    occupancyTotal / Float(max(occupancyCount, 1))
                }

                var averageBFactor: Float {
                    bFactorTotal / Float(max(atomCount, 1))
                }
            }

            var scores: [String: AltScore] = [:]
            var hasAlternateAtoms = false

            for index in indices {
                let altLoc = atoms[index].altLoc.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !altLoc.isEmpty else { continue }
                hasAlternateAtoms = true
                var score = scores[altLoc, default: AltScore()]
                score.occupancyTotal += atoms[index].occupancy
                score.occupancyCount += 1
                score.bFactorTotal += atoms[index].tempFactor
                score.atomCount += 1
                scores[altLoc] = score
            }

            if !hasAlternateAtoms {
                keptIndices.formUnion(indices)
                continue
            }

            guard let preferredAltLoc = scores.keys.sorted(by: { lhs, rhs in
                let left = scores[lhs] ?? AltScore()
                let right = scores[rhs] ?? AltScore()
                if abs(left.averageOccupancy - right.averageOccupancy) > 0.0001 {
                    return left.averageOccupancy > right.averageOccupancy
                }
                if abs(left.averageBFactor - right.averageBFactor) > 0.0001 {
                    return left.averageBFactor < right.averageBFactor
                }
                return lhs < rhs
            }).first else {
                keptIndices.formUnion(indices)
                continue
            }

            for index in indices {
                let altLoc = atoms[index].altLoc.trimmingCharacters(in: .whitespacesAndNewlines)
                if altLoc.isEmpty || altLoc == preferredAltLoc {
                    keptIndices.insert(index)
                }
            }
        }

        let selectedIndices = atoms.indices.filter { keptIndices.contains($0) }
        let remapped = remapSubstructure(atoms: atoms, bonds: bonds, selectedIndices: selectedIndices)
        let cleanedAtoms = remapped.atoms.map { atom -> Atom in
            var atom = atom
            if !atom.altLoc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                atom.altLoc = ""
            }
            return atom
        }

        return (cleanedAtoms, remapped.bonds, atoms.count - cleanedAtoms.count)
    }

    // MARK: - Remove Waters

    static func removeWaters(
        atoms: [Atom],
        bonds: [Bond],
        keepingNearby centers: [SIMD3<Float>] = [],
        within retentionRadius: Float? = nil
    ) -> (atoms: [Atom], bonds: [Bond], removedCount: Int) {
        let effectiveRadius = retentionRadius ?? 0
        var residueIndices: [WaterResidueKey: [Int]] = [:]
        var residueOrder: [WaterResidueKey] = []

        for index in atoms.indices {
            let atom = atoms[index]
            let key = WaterResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: atom.residueName
            )
            if residueIndices[key] == nil {
                residueOrder.append(key)
            }
            residueIndices[key, default: []].append(index)
        }

        var keptIndices: [Int] = []
        var removedResidues = 0

        for key in residueOrder {
            guard let indices = residueIndices[key] else { continue }
            let isWater = waterResidueNames.contains(key.residueName)
            if !isWater {
                keptIndices.append(contentsOf: indices)
                continue
            }

            let shouldKeep = !centers.isEmpty && effectiveRadius > 0 && indices.contains { index in
                centers.contains { center in
                    simd_distance(atoms[index].position, center) <= effectiveRadius
                }
            }

            if shouldKeep {
                keptIndices.append(contentsOf: indices)
            } else {
                removedResidues += 1
            }
        }

        let remapped = remapSubstructure(atoms: atoms, bonds: bonds, selectedIndices: keptIndices)
        return (remapped.atoms, remapped.bonds, removedResidues)
    }

    // MARK: - Remove Non-standard Residues

    static func removeNonStandardResidues(
        atoms: [Atom],
        bonds: [Bond],
        keepingWaters: Bool = true,
        keepingExistingCaps: Bool = true
    ) -> (atoms: [Atom], bonds: [Bond], removedResidueCount: Int, removedAtomCount: Int) {
        let residueRecords = buildResidueRecords(from: atoms)
        var keptIndices: [Int] = []
        var removedResidues = 0
        var removedAtoms = 0

        for residue in residueRecords.values.sorted(by: residueSortOrder) {
            let residueName = residue.residueName.uppercased()
            let shouldKeep: Bool
            if standardResidueNames.contains(residueName) {
                shouldKeep = true
            } else if waterResidueNames.contains(residueName) {
                shouldKeep = keepingWaters
            } else if cappingResidueNames.contains(residueName) {
                shouldKeep = keepingExistingCaps
            } else {
                shouldKeep = false
            }

            if shouldKeep {
                keptIndices.append(contentsOf: residue.atomIndices)
            } else {
                removedResidues += 1
                removedAtoms += residue.atomIndices.count
            }
        }

        let remapped = remapSubstructure(atoms: atoms, bonds: bonds, selectedIndices: keptIndices.sorted())
        return (remapped.atoms, remapped.bonds, removedResidues, removedAtoms)
    }

    // MARK: - Cleanup Pipeline

    static func cleanupStructure(
        atoms: [Atom],
        bonds: [Bond],
        options: CleanupOptions = CleanupOptions()
    ) -> (atoms: [Atom], bonds: [Bond], report: CleanupReport) {
        var report = CleanupReport()

        let altConformerSelection = selectPreferredAltConformers(atoms: atoms, bonds: bonds)
        var workingAtoms = altConformerSelection.atoms
        var workingBonds = altConformerSelection.bonds
        report.removedAltConformerAtoms = altConformerSelection.removedAtomCount
        if altConformerSelection.removedAtomCount > 0 {
            Task { @MainActor in ActivityLog.shared.info("[Prep] Alt-conf selection: kept \(altConformerSelection.atoms.count) atoms, removed \(altConformerSelection.removedAtomCount)", category: .prep) }
        }

        if options.removeNonStandardResidues {
            let filtered = removeNonStandardResidues(
                atoms: workingAtoms,
                bonds: workingBonds,
                keepingWaters: true,
                keepingExistingCaps: options.keepExistingCaps
            )
            workingAtoms = filtered.atoms
            workingBonds = filtered.bonds
            report.removedHeterogenResidues = filtered.removedResidueCount
            report.removedHeterogenAtoms = filtered.removedAtomCount
            if filtered.removedResidueCount > 0 {
                Task { @MainActor in ActivityLog.shared.info("[Prep] Removed \(filtered.removedResidueCount) non-standard residues (\(filtered.removedAtomCount) atoms)", category: .prep) }
            }
        }

        switch options.waterPolicy {
        case .keepAll:
            break
        case .removeAll:
            let stripped = removeWaters(atoms: workingAtoms, bonds: workingBonds)
            workingAtoms = stripped.atoms
            workingBonds = stripped.bonds
            report.removedWaterResidues = stripped.removedCount
        case .keepNearby(let centers, let radius):
            let stripped = removeWaters(
                atoms: workingAtoms,
                bonds: workingBonds,
                keepingNearby: centers,
                within: radius
            )
            workingAtoms = stripped.atoms
            workingBonds = stripped.bonds
            report.removedWaterResidues = stripped.removedCount
        }

        if report.removedWaterResidues > 0 {
            Task { @MainActor in ActivityLog.shared.info("[Prep] Removed \(report.removedWaterResidues) water residues", category: .prep) }
        }

        report.chainBreaks = detectChainBreaks(in: workingAtoms, bonds: workingBonds)

        if options.capChainBreaks {
            let uncappedBreaks = report.chainBreaks.filter { !$0.isCapped }
            if !uncappedBreaks.isEmpty {
                let capped = capChainBreaks(atoms: workingAtoms, bonds: workingBonds, breaks: uncappedBreaks)
                workingAtoms = capped.atoms
                workingBonds = capped.bonds
                report.addedCappingResidues = capped.addedResidueCount
                report.chainBreaks = detectChainBreaks(in: workingAtoms, bonds: workingBonds)
                Task { @MainActor in ActivityLog.shared.info("[Prep] Capped \(capped.addedResidueCount) chain break termini", category: .prep) }
            }
        }

        Task { @MainActor in ActivityLog.shared.info("[Prep] Cleanup complete: \(workingAtoms.count) atoms, \(report.chainBreaks.count) chain breaks", category: .prep) }
        return (workingAtoms, workingBonds, report)
    }

    // MARK: - Detect Missing Residues

    static func detectMissingResidues(in atoms: [Atom]) -> [(chainID: String, gapStart: Int, gapEnd: Int)] {
        // Group residue sequence numbers by chain
        var chainResidues: [String: Set<Int>] = [:]
        for atom in atoms {
            guard !atom.isHetAtom else { continue }
            chainResidues[atom.chainID, default: []].insert(atom.residueSeq)
        }

        var gaps: [(chainID: String, gapStart: Int, gapEnd: Int)] = []

        for (chainID, seqNums) in chainResidues.sorted(by: { $0.key < $1.key }) {
            let sorted = seqNums.sorted()
            guard sorted.count > 1 else { continue }

            for i in 0..<(sorted.count - 1) {
                let diff = sorted[i + 1] - sorted[i]
                if diff > 1 {
                    gaps.append((chainID: chainID, gapStart: sorted[i] + 1, gapEnd: sorted[i + 1] - 1))
                }
            }
        }

        return gaps
    }

    // MARK: - Analyze Structure

    static func analyze(atoms: [Atom], bonds: [Bond], waterCount: Int = 0) -> PreparationReport {
        var report = PreparationReport()
        report.waterCount = waterCount
        report.totalAtoms = atoms.count
        report.heavyAtoms = atoms.filter { $0.element != .H }.count

        // Chain summary
        var chainResCount: [String: Set<Int>] = [:]
        var chainTypes: [String: String] = [:]
        for atom in atoms {
            chainResCount[atom.chainID, default: []].insert(atom.residueSeq)
            if chainTypes[atom.chainID] == nil {
                chainTypes[atom.chainID] = atom.isHetAtom ? "Ligand" : "Protein"
            }
        }
        report.chainSummary = chainResCount.keys.sorted().map { chainID in
            (chainID: chainID,
             residueCount: chainResCount[chainID]?.count ?? 0,
             type: chainTypes[chainID] ?? "Unknown")
        }

        // HETATM groups
        var hetGroupCounts: [String: Int] = [:]
        for atom in atoms where atom.isHetAtom {
            hetGroupCounts[atom.residueName, default: 0] += 1
        }
        report.hetGroups = hetGroupCounts.map { (name: $0.key, count: $0.value) }
            .sorted { $0.count > $1.count }

        // Missing residues
        report.missingResidues = detectMissingResidues(in: atoms)
        report.chainBreaks = detectChainBreaks(in: atoms, bonds: bonds)
        report.residueCompleteness = analyzeResidueCompleteness(atoms: atoms, bonds: bonds)

        return report
    }

    // MARK: - Disulfide Bond Detection

    struct DisulfideBond: Sendable {
        let cys1ChainID: String
        let cys1ResSeq: Int
        let cys2ChainID: String
        let cys2ResSeq: Int
        let sgDistance: Float
    }

    /// Detect disulfide bonds between CYS residues (SG-SG distance < 2.5 Å).
    /// Returns detected bonds and adds them to the molecule's bond array.
    static func detectDisulfideBonds(atoms: [Atom]) -> [DisulfideBond] {
        let sgAtoms = atoms.enumerated().filter { _, atom in
            atom.residueName == "CYS"
                && atom.name.trimmingCharacters(in: .whitespaces) == "SG"
                && !atom.isHetAtom
        }
        var bonds: [DisulfideBond] = []
        for i in 0..<sgAtoms.count {
            for j in (i + 1)..<sgAtoms.count {
                let (_, a1) = sgAtoms[i]
                let (_, a2) = sgAtoms[j]
                let dist = simd_distance(a1.position, a2.position)
                if dist < 2.5 {
                    bonds.append(DisulfideBond(
                        cys1ChainID: a1.chainID, cys1ResSeq: a1.residueSeq,
                        cys2ChainID: a2.chainID, cys2ResSeq: a2.residueSeq,
                        sgDistance: dist
                    ))
                }
            }
        }
        return bonds
    }

    /// Add disulfide bonds to the bond array and return indices of bonded SG atoms
    /// (so protonation can skip adding HG to these cysteines).
    static func applyDisulfideBonds(
        atoms: [Atom],
        bonds: inout [Bond],
        disulfides: [DisulfideBond]
    ) -> Set<Int> {
        var bondedSGIndices = Set<Int>()
        for ss in disulfides {
            let sg1 = atoms.firstIndex(where: {
                $0.chainID == ss.cys1ChainID && $0.residueSeq == ss.cys1ResSeq
                    && $0.name.trimmingCharacters(in: .whitespaces) == "SG"
            })
            let sg2 = atoms.firstIndex(where: {
                $0.chainID == ss.cys2ChainID && $0.residueSeq == ss.cys2ResSeq
                    && $0.name.trimmingCharacters(in: .whitespaces) == "SG"
            })
            if let i1 = sg1, let i2 = sg2 {
                // Check if bond already exists
                let exists = bonds.contains {
                    ($0.atomIndex1 == i1 && $0.atomIndex2 == i2)
                        || ($0.atomIndex1 == i2 && $0.atomIndex2 == i1)
                }
                if !exists {
                    bonds.append(Bond(id: bonds.count, atomIndex1: i1, atomIndex2: i2, order: .single))
                }
                bondedSGIndices.insert(i1)
                bondedSGIndices.insert(i2)
            }
        }
        return bondedSGIndices
    }

    // MARK: - Detect Chain Breaks

    static func detectChainBreaks(in atoms: [Atom]) -> [ChainBreak] {
        detectChainBreaks(in: atoms, bonds: [])
    }

    static func detectChainBreaks(in atoms: [Atom], bonds: [Bond]) -> [ChainBreak] {
        let residueRecords = buildResidueRecords(from: atoms)
        var residuesByChain: [String: [ResidueRecord]] = [:]
        for residue in residueRecords.values where standardResidueNames.contains(residue.residueName.uppercased()) {
            residuesByChain[residue.chainID, default: []].append(residue)
        }

        var breaks: [ChainBreak] = []
        for (chainID, residues) in residuesByChain {
            let sortedResidues = residues.sorted { lhs, rhs in
                if lhs.residueSeq != rhs.residueSeq {
                    return lhs.residueSeq < rhs.residueSeq
                }
                return lhs.residueName < rhs.residueName
            }

            guard sortedResidues.count > 1 else { continue }

            for index in 0..<(sortedResidues.count - 1) {
                let previous = sortedResidues[index]
                let next = sortedResidues[index + 1]
                let missingResidueCount = max(next.residueSeq - previous.residueSeq - 1, 0)

                let carbonIndex = atomIndex(named: "C", element: .C, in: previous, atoms: atoms)
                let nitrogenIndex = atomIndex(named: "N", element: .N, in: next, atoms: atoms)

                let distance: Float = if let carbonIndex, let nitrogenIndex {
                    simd_distance(atoms[carbonIndex].position, atoms[nitrogenIndex].position)
                } else {
                    .infinity
                }

                if missingResidueCount > 0 || distance > 1.8 {
                    let isCapped = hasCapBond(
                        anchorAtomIndex: carbonIndex,
                        capResidueName: "NME",
                        capAtomName: "N",
                        atoms: atoms,
                        bonds: bonds
                    ) && hasCapBond(
                        anchorAtomIndex: nitrogenIndex,
                        capResidueName: "ACE",
                        capAtomName: "C",
                        atoms: atoms,
                        bonds: bonds
                    )
                    breaks.append(.init(
                        chainID: chainID,
                        previousResidueSeq: previous.residueSeq,
                        nextResidueSeq: next.residueSeq,
                        carbonNitrogenDistance: distance,
                        missingResidueCount: missingResidueCount,
                        isCapped: isCapped
                    ))
                }
            }
        }

        return breaks.sorted { lhs, rhs in
            if lhs.chainID != rhs.chainID {
                return lhs.chainID < rhs.chainID
            }
            return lhs.previousResidueSeq < rhs.previousResidueSeq
        }
    }

    // MARK: - Shared Docking Preparation

    /// Prepare a receptor copy for docking without mutating the displayed model.
    /// Full native protein-prep pass used before docking:
    /// Phase 1 cleanup, Phase 2 residue completion, and Phase 3 protonation-aware
    /// hydrogen/charge assignment, with RDKit partial charges merged afterward when
    /// the raw PDB is available.
    static func prepareForDocking(
        atoms: [Atom],
        bonds: [Bond],
        rawPDBContent: String? = nil,
        pH: Float = 7.4,
        chargeMethod: ChargeMethod = .gasteiger
    ) -> (atoms: [Atom], bonds: [Bond], report: DockingPreparationReport) {
        var report = DockingPreparationReport()

        let cleanup = cleanupStructure(atoms: atoms, bonds: bonds)
        report.altConformerAtomsRemoved = cleanup.report.removedAltConformerAtoms
        report.heterogenResiduesRemoved = cleanup.report.removedHeterogenResidues
        report.cappingResiduesAdded = cleanup.report.addedCappingResidues
        report.chainBreaksDetected = cleanup.report.chainBreaks.count
        report.chainBreaksCapped = cleanup.report.chainBreaks.filter(\.isCapped).count

        let phase23 = completePhase23(atoms: cleanup.atoms, bonds: cleanup.bonds, pH: pH)
        var workingAtoms = phase23.atoms
        var workingBonds = phase23.bonds

        report.missingHeavyAtomsAdded = phase23.report.heavyAtomsAdded
        report.hydrogensAdded = phase23.report.hydrogensAdded
        report.protonationUpdates = phase23.protonation.reduce(into: 0) { partial, prediction in
            partial += prediction.atomFormalCharges.count
        }
        if phase23.report.hydrogensAdded > 0 {
            report.hydrogenMethod = "template-driven"
        } else if cleanup.atoms.contains(where: { $0.element == .H }) {
            report.hydrogenMethod = "existing"
        }

        // Phase 4: H-bond network optimization (enumerate, score, optimize, apply)
        let network = HBondNetworkOptimizer.optimizeNetwork(
            atoms: workingAtoms,
            bonds: workingBonds,
            predictions: phase23.protonation
        )
        workingAtoms = network.atoms
        workingBonds = network.bonds
        report.hbondNetworkReport = network.report

        // Phase 5: Charge assignment — dispatch based on selected method
        switch chargeMethod {
        case .gasteiger:
            // RDKit's PDB parser can crash on very large proteins (>8000 atoms).
            // For large structures, skip RDKit Gasteiger and rely on the
            // electrostatic fallback table (residue-based partial charges).
            // RDKit's PDB-based Gasteiger charges can crash on large structures.
            // The fragment-based approach in RDKit parses each residue independently
            // which is O(n_residues) but the internal graph operations are expensive
            // for large molecules. Above this threshold, we use the residue-based
            // fallback charge table instead (accurate for standard amino acids).
            let maxAtomsForRDKitCharges = 6000
            if let pdbContent = rawPDBContent, workingAtoms.count <= maxAtomsForRDKitCharges,
               let chargeData = RDKitBridge.computeChargesPDB(pdbContent: pdbContent) {
                let merged = mergeProteinAtoms(currentAtoms: workingAtoms, sourceAtoms: chargeData.atoms) { current, source in
                    current.charge = source.charge
                }
                workingAtoms = merged.atoms
                report.rdkitChargeMatches = merged.matchedCount
            } else if workingAtoms.count > maxAtomsForRDKitCharges {
                report.hydrogenMethod = (report.hydrogenMethod ?? "") + " (charges: fallback, too large for RDKit)"
            }

        case .eem, .qeq, .xtb:
            // Use the new charge calculators (EEM/QEq run on GPU, xTB on CPU)
            let calculator: ChargeCalculator
            switch chargeMethod {
            case .eem: calculator = EEMChargeCalculator(device: nil)
            case .qeq: calculator = QEqChargeCalculator(device: nil)
            case .xtb: calculator = XTBChargeCalculator()
            default:   calculator = GasteigerChargeCalculator()
            }
            // Synchronous wrapper — prepareForDocking is called from a Task context
            let semaphore = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var computedCharges: [Float]?
            let calcAtoms = workingAtoms
            let calcBonds = workingBonds
            let calc = calculator
            Task.detached { @Sendable in
                computedCharges = try? await calc.computeCharges(
                    atoms: calcAtoms, bonds: calcBonds, totalCharge: 0
                )
                semaphore.signal()
            }
            semaphore.wait()
            if let charges = computedCharges {
                for i in 0..<min(charges.count, workingAtoms.count) {
                    workingAtoms[i].charge = charges[i]
                }
            }
        }

        let beforeFallback = workingAtoms
        workingAtoms = applyElectrostaticFallback(to: workingAtoms)
        report.fallbackChargeAtoms = zip(beforeFallback, workingAtoms).filter {
            abs($0.charge) <= 0.0001 && abs($1.charge) > 0.0001
        }.count
        report.nonZeroChargeAtoms = workingAtoms.filter { abs($0.charge) > 0.001 }.count

        Task { @MainActor in ActivityLog.shared.info("[Prep] Docking prep done: \(workingAtoms.count) atoms, \(report.hydrogensAdded) H added, \(report.nonZeroChargeAtoms) charged atoms (\(chargeMethod))", category: .prep) }
        return (workingAtoms, workingBonds, report)
    }

    // MARK: - Add Polar Hydrogens (preserves all PDB residue info)

    /// Add polar hydrogens (N-H, O-H) to protein backbone and sidechains.
    /// Places H atoms at standard bond distances along estimated bond vectors.
    /// Preserves all existing atom properties (residueName, chainID, residueSeq, etc).
    static func addPolarHydrogens(atoms: [Atom], bonds: [Bond]) -> (atoms: [Atom], bonds: [Bond]) {
        // Build adjacency map
        var neighbors: [Int: [Int]] = [:]
        for bond in bonds {
            neighbors[bond.atomIndex1, default: []].append(bond.atomIndex2)
            neighbors[bond.atomIndex2, default: []].append(bond.atomIndex1)
        }

        var newAtoms = atoms
        var newBonds = bonds
        var addedCount = 0

        // Standard N-H bond distance: 1.01 Å, O-H: 0.96 Å
        let nhDist: Float = 1.01
        let ohDist: Float = 0.96

        for (idx, atom) in atoms.enumerated() {
            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)

            // Backbone amide N-H (every residue except PRO N-terminus)
            if trimmedName == "N" && atom.element == .N && atom.residueName != "PRO" {
                let nNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                // Backbone N has 2 heavy neighbors (CA and C of previous residue) or 1 if N-terminus
                if nNeighbors.count <= 2 {
                    let hPos = placeHydrogen(on: atom.position, awayFrom: nNeighbors.map { atoms[$0].position }, distance: nhDist)
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "H", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }

            // Sidechain N-H: NZ (Lys), NE/NH1/NH2 (Arg), ND2 (Asn), NE2 (Gln), ND1/NE2 (His)
            if atom.element == .N && !["N", ""].contains(trimmedName) {
                let donors: Set<String> = ["NZ", "NE", "NH1", "NH2", "ND2", "NE2", "ND1", "NE1"]
                if donors.contains(trimmedName) {
                    let nNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                    let hPos = placeHydrogen(on: atom.position, awayFrom: nNeighbors.map { atoms[$0].position }, distance: nhDist)
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "HN", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }

            // Hydroxyl O-H: OG (Ser), OG1 (Thr), OH (Tyr)
            // These are sp3 oxygens — H should be placed at ~109.5° from the C-O bond,
            // not 180° opposite (which would be linear / sp geometry).
            if atom.element == .O {
                let hydroxyls: Set<String> = ["OG", "OG1", "OH"]
                if hydroxyls.contains(trimmedName) {
                    let oNeighbors = (neighbors[idx] ?? []).filter { atoms[$0].element != .H }
                    let hPos = placeTetrahedralHydrogen(
                        on: atom.position,
                        neighbors: oNeighbors.map { atoms[$0].position },
                        distance: ohDist
                    )
                    let hAtom = Atom(
                        id: newAtoms.count, element: .H, position: hPos,
                        name: "HO", residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: 0, formalCharge: 0, isHetAtom: atom.isHetAtom
                    )
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: idx, atomIndex2: newAtoms.count, order: .single))
                    newAtoms.append(hAtom)
                    addedCount += 1
                }
            }
        }

        return (newAtoms, newBonds)
    }

    /// Place a hydrogen atom at `distance` from `center`, in the direction opposite to the average
    /// of neighbor positions (so it points away from existing bonds).
    private static func placeHydrogen(on center: SIMD3<Float>, awayFrom neighbors: [SIMD3<Float>], distance: Float) -> SIMD3<Float> {
        if neighbors.isEmpty {
            // No neighbors — place along +Y
            return center + SIMD3<Float>(0, distance, 0)
        }

        // Average direction of neighbors from center
        let avg = neighbors.reduce(SIMD3<Float>.zero) { $0 + ($1 - center) } / Float(neighbors.count)
        let len = simd_length(avg)
        if len < 1e-6 {
            return center + SIMD3<Float>(0, distance, 0)
        }

        // Place H opposite to average neighbor direction
        let dir = -simd_normalize(avg)
        return center + dir * distance
    }

    /// Place a hydrogen at tetrahedral (sp3) geometry from `center`.
    ///
    /// - 1 neighbor: place H at 109.5° from the neighbor-center bond, with an arbitrary
    ///   dihedral rotation. The exact dihedral will be refined later by H-bond network
    ///   optimization, so any consistent choice is fine.
    /// - 2 neighbors: place H in the direction that makes ~109.5° with both existing bonds
    ///   (i.e. between the two lone-pair positions of a tetrahedral center).
    /// - 0 or 3+ neighbors: falls back to the linear (opposite) placement.
    private static func placeTetrahedralHydrogen(
        on center: SIMD3<Float>,
        neighbors: [SIMD3<Float>],
        distance: Float
    ) -> SIMD3<Float> {
        let tetrahedralAngle: Float = 109.5 * .pi / 180.0  // radians

        if neighbors.count == 1 {
            // Single neighbor (e.g. C-O-H): place H at 109.5° from the C-O bond direction
            let bondDir = simd_normalize(center - neighbors[0])  // O←C direction, i.e. away from C

            // Find a vector perpendicular to bondDir
            let candidate: SIMD3<Float> = abs(bondDir.x) < 0.9
                ? SIMD3<Float>(1, 0, 0)
                : SIMD3<Float>(0, 1, 0)
            let perp = simd_normalize(simd_cross(bondDir, candidate))

            // Rotate bondDir by (180° - 109.5°) = 70.5° around the perpendicular axis
            // This gives the H direction at 109.5° from the neighbor bond (measuring C-O-H angle)
            let rotAngle = Float.pi - tetrahedralAngle  // 70.5°
            let hDir = rotateVector(bondDir, around: perp, by: rotAngle)

            return center + hDir * distance

        } else if neighbors.count == 2 {
            // Two neighbors: place H so it makes ~109.5° with both existing bonds.
            // The H goes roughly opposite to the average neighbor direction, but tilted
            // to maintain tetrahedral angles.
            let dir1 = simd_normalize(neighbors[0] - center)
            let dir2 = simd_normalize(neighbors[1] - center)
            let avgNeighborDir = simd_normalize(dir1 + dir2)

            // The H direction is opposite to the average, which for ideal tetrahedral
            // geometry already gives the correct angle. For two neighbors in a tetrahedron,
            // the third position is in the plane defined by the two bonds, pointing away.
            let hDir = -avgNeighborDir

            // Verify: for a perfect tetrahedral with 2 neighbors, placing opposite to
            // their average naturally yields ~109.5° angles when the neighbor-center-neighbor
            // angle is also ~109.5°. This is the standard approach.
            return center + hDir * distance

        } else {
            // Fallback to linear placement for 0 or 3+ neighbors
            return placeHydrogen(on: center, awayFrom: neighbors, distance: distance)
        }
    }

    /// Rotate a vector around an axis by a given angle (Rodrigues' rotation formula).
    private static func rotateVector(
        _ v: SIMD3<Float>,
        around axis: SIMD3<Float>,
        by angle: Float
    ) -> SIMD3<Float> {
        let k = simd_normalize(axis)
        let cosA = cos(angle)
        let sinA = sin(angle)
        return v * cosA + simd_cross(k, v) * sinA + k * simd_dot(k, v) * (1 - cosA)
    }

    private struct ProteinAtomIdentity: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let atomName: String
        let atomicNumber: Int
        let isHetAtom: Bool
    }

    private static func proteinAtomIdentity(for atom: Atom) -> ProteinAtomIdentity {
        ProteinAtomIdentity(
            chainID: atom.chainID,
            residueSeq: atom.residueSeq,
            residueName: atom.residueName,
            atomName: atom.name.trimmingCharacters(in: .whitespaces),
            atomicNumber: atom.element.rawValue,
            isHetAtom: atom.isHetAtom
        )
    }

    private static func mergeProteinAtoms(
        currentAtoms: [Atom],
        sourceAtoms: [Atom],
        update: (inout Atom, Atom) -> Void
    ) -> (atoms: [Atom], matchedCount: Int) {
        var buckets: [ProteinAtomIdentity: [Atom]] = [:]
        for atom in sourceAtoms {
            buckets[proteinAtomIdentity(for: atom), default: []].append(atom)
        }

        var offsets: [ProteinAtomIdentity: Int] = [:]
        var updated = currentAtoms
        var matchedCount = 0

        for index in updated.indices {
            let identity = proteinAtomIdentity(for: updated[index])
            let offset = offsets[identity, default: 0]
            guard let bucket = buckets[identity], offset < bucket.count else { continue }
            update(&updated[index], bucket[offset])
            offsets[identity] = offset + 1
            matchedCount += 1
        }

        return (updated, matchedCount)
    }

    private static func buildResidueRecords(from atoms: [Atom]) -> [ResidueKey: ResidueRecord] {
        var residueIndices: [ResidueKey: [Int]] = [:]
        for index in atoms.indices {
            let atom = atoms[index]
            let key = ResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: atom.residueName,
                isHetAtom: atom.isHetAtom
            )
            residueIndices[key, default: []].append(index)
        }

        var records: [ResidueKey: ResidueRecord] = [:]
        for (key, indices) in residueIndices {
            records[key] = ResidueRecord(
                chainID: key.chainID,
                residueSeq: key.residueSeq,
                residueName: key.residueName,
                isHetAtom: key.isHetAtom,
                atomIndices: indices.sorted()
            )
        }
        return records
    }

    private static func residueSortOrder(lhs: ResidueRecord, rhs: ResidueRecord) -> Bool {
        if lhs.chainID != rhs.chainID {
            return lhs.chainID < rhs.chainID
        }
        if lhs.residueSeq != rhs.residueSeq {
            return lhs.residueSeq < rhs.residueSeq
        }
        if lhs.residueName != rhs.residueName {
            return lhs.residueName < rhs.residueName
        }
        return lhs.isHetAtom && !rhs.isHetAtom
    }

    private static func atomIndex(
        named atomName: String,
        element: Element? = nil,
        in residue: ResidueRecord,
        atoms: [Atom]
    ) -> Int? {
        residue.atomIndices.first { index in
            let atom = atoms[index]
            guard atom.name.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() == atomName.uppercased() else {
                return false
            }
            if let element {
                return atom.element == element
            }
            return true
        }
    }

    private static func hasCapBond(
        anchorAtomIndex: Int?,
        capResidueName: String,
        capAtomName: String,
        atoms: [Atom],
        bonds: [Bond]
    ) -> Bool {
        guard let anchorAtomIndex else { return false }
        for bond in bonds {
            let neighborIndex: Int?
            if bond.atomIndex1 == anchorAtomIndex {
                neighborIndex = bond.atomIndex2
            } else if bond.atomIndex2 == anchorAtomIndex {
                neighborIndex = bond.atomIndex1
            } else {
                neighborIndex = nil
            }

            guard let neighborIndex else { continue }
            let neighbor = atoms[neighborIndex]
            if neighbor.residueName.uppercased() == capResidueName &&
                neighbor.name.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() == capAtomName.uppercased() {
                return true
            }
        }
        return false
    }

    private static func capChainBreaks(
        atoms: [Atom],
        bonds: [Bond],
        breaks: [ChainBreak]
    ) -> (atoms: [Atom], bonds: [Bond], addedResidueCount: Int) {
        let residueRecords = buildResidueRecords(from: atoms)
        var workingAtoms = atoms
        var workingBonds = bonds
        var addedResidues = 0

        for chainBreak in breaks {
            let previousResidue = residueRecords[ResidueKey(
                chainID: chainBreak.chainID,
                residueSeq: chainBreak.previousResidueSeq,
                residueName: residueName(
                    chainID: chainBreak.chainID,
                    residueSeq: chainBreak.previousResidueSeq,
                    from: residueRecords
                ),
                isHetAtom: false
            )]
            let nextResidue = residueRecords[ResidueKey(
                chainID: chainBreak.chainID,
                residueSeq: chainBreak.nextResidueSeq,
                residueName: residueName(
                    chainID: chainBreak.chainID,
                    residueSeq: chainBreak.nextResidueSeq,
                    from: residueRecords
                ),
                isHetAtom: false
            )]

            guard let previousResidue, let nextResidue else {
                Task { @MainActor in ActivityLog.shared.debug("[Prep] Skipped capping chain break at \(chainBreak.chainID):\(chainBreak.previousResidueSeq)-\(chainBreak.nextResidueSeq) — residue not found", category: .prep) }
                continue
            }

            if appendNMECap(
                to: &workingAtoms,
                bonds: &workingBonds,
                after: previousResidue,
                toward: nextResidue
            ) {
                addedResidues += 1
            }

            if appendACECap(
                to: &workingAtoms,
                bonds: &workingBonds,
                before: nextResidue,
                toward: previousResidue
            ) {
                addedResidues += 1
            }
        }

        return (workingAtoms, workingBonds, addedResidues)
    }

    private static func residueName(
        chainID: String,
        residueSeq: Int,
        from residueRecords: [ResidueKey: ResidueRecord]
    ) -> String {
        residueRecords.keys.first {
            $0.chainID == chainID &&
            $0.residueSeq == residueSeq &&
            !$0.isHetAtom &&
            standardResidueNames.contains($0.residueName.uppercased())
        }?.residueName ?? ""
    }

    private static func appendNMECap(
        to atoms: inout [Atom],
        bonds: inout [Bond],
        after residue: ResidueRecord,
        toward nextResidue: ResidueRecord
    ) -> Bool {
        guard let carbonIndex = atomIndex(named: "C", element: .C, in: residue, atoms: atoms) else {
            Task { @MainActor in ActivityLog.shared.debug("[Prep] NME cap skipped: no backbone C in residue \(residue.chainID):\(residue.residueSeq)", category: .prep) }
            return false
        }
        if hasCapBond(anchorAtomIndex: carbonIndex, capResidueName: "NME", capAtomName: "N", atoms: atoms, bonds: bonds) {
            return false
        }

        let carbonPosition = atoms[carbonIndex].position
        let oxygenPosition = atomIndex(named: "O", element: .O, in: residue, atoms: atoms).map { atoms[$0].position }
            ?? atomIndex(named: "OXT", element: .O, in: residue, atoms: atoms).map { atoms[$0].position }
        let alphaCarbonPosition = atomIndex(named: "CA", element: .C, in: residue, atoms: atoms).map { atoms[$0].position }
        let nextNitrogenPosition = atomIndex(named: "N", element: .N, in: nextResidue, atoms: atoms).map { atoms[$0].position }

        let bondDirection = missingBondDirection(
            origin: carbonPosition,
            existingNeighbors: [oxygenPosition, alphaCarbonPosition].compactMap { $0 },
            fallbackTarget: nextNitrogenPosition
        )
        let nitrogenPosition = carbonPosition + bondDirection * 1.33

        let nitrogenIndex = atoms.count
        atoms.append(Atom(
            id: nitrogenIndex,
            element: .N,
            position: nitrogenPosition,
            name: "N",
            residueName: "NME",
            residueSeq: residue.residueSeq,
            chainID: residue.chainID,
            isHetAtom: true
        ))
        bonds.append(Bond(id: bonds.count, atomIndex1: carbonIndex, atomIndex2: nitrogenIndex, order: .single))

        let methylDirection = normalized(nitrogenPosition - carbonPosition, fallback: SIMD3<Float>(0, 1, 0))
        let methylIndex = atoms.count
        atoms.append(Atom(
            id: methylIndex,
            element: .C,
            position: nitrogenPosition + methylDirection * 1.46,
            name: "CH3",
            residueName: "NME",
            residueSeq: residue.residueSeq,
            chainID: residue.chainID,
            isHetAtom: true
        ))
        bonds.append(Bond(id: bonds.count, atomIndex1: nitrogenIndex, atomIndex2: methylIndex, order: .single))

        return true
    }

    private static func appendACECap(
        to atoms: inout [Atom],
        bonds: inout [Bond],
        before residue: ResidueRecord,
        toward previousResidue: ResidueRecord
    ) -> Bool {
        guard let nitrogenIndex = atomIndex(named: "N", element: .N, in: residue, atoms: atoms) else {
            Task { @MainActor in ActivityLog.shared.debug("[Prep] ACE cap skipped: no backbone N in residue \(residue.chainID):\(residue.residueSeq)", category: .prep) }
            return false
        }
        if hasCapBond(anchorAtomIndex: nitrogenIndex, capResidueName: "ACE", capAtomName: "C", atoms: atoms, bonds: bonds) {
            return false
        }

        let nitrogenPosition = atoms[nitrogenIndex].position
        let alphaCarbonPosition = atomIndex(named: "CA", element: .C, in: residue, atoms: atoms).map { atoms[$0].position }
        let amideHydrogenPosition = atomIndex(named: "H", element: .H, in: residue, atoms: atoms).map { atoms[$0].position }
        let previousCarbonPosition = atomIndex(named: "C", element: .C, in: previousResidue, atoms: atoms).map { atoms[$0].position }

        let bondDirection = missingBondDirection(
            origin: nitrogenPosition,
            existingNeighbors: [alphaCarbonPosition, amideHydrogenPosition].compactMap { $0 },
            fallbackTarget: previousCarbonPosition
        )
        let carbonylCarbonPosition = nitrogenPosition + bondDirection * 1.33
        let axisToNitrogen = normalized(nitrogenPosition - carbonylCarbonPosition, fallback: SIMD3<Float>(1, 0, 0))
        let preferredReference = (alphaCarbonPosition ?? previousCarbonPosition).map { $0 - carbonylCarbonPosition }
        let planeReference = planarReference(axis: axisToNitrogen, preferred: preferredReference)
        let oxygenDirection = normalized((-0.5 * axisToNitrogen) + (0.8660254 * planeReference), fallback: planeReference)
        let methylDirection = normalized((-0.5 * axisToNitrogen) - (0.8660254 * planeReference), fallback: -planeReference)

        let carbonylCarbonIndex = atoms.count
        atoms.append(Atom(
            id: carbonylCarbonIndex,
            element: .C,
            position: carbonylCarbonPosition,
            name: "C",
            residueName: "ACE",
            residueSeq: residue.residueSeq,
            chainID: residue.chainID,
            isHetAtom: true
        ))
        bonds.append(Bond(id: bonds.count, atomIndex1: carbonylCarbonIndex, atomIndex2: nitrogenIndex, order: .single))

        let oxygenIndex = atoms.count
        atoms.append(Atom(
            id: oxygenIndex,
            element: .O,
            position: carbonylCarbonPosition + oxygenDirection * 1.24,
            name: "O",
            residueName: "ACE",
            residueSeq: residue.residueSeq,
            chainID: residue.chainID,
            isHetAtom: true
        ))
        bonds.append(Bond(id: bonds.count, atomIndex1: carbonylCarbonIndex, atomIndex2: oxygenIndex, order: .double))

        let methylIndex = atoms.count
        atoms.append(Atom(
            id: methylIndex,
            element: .C,
            position: carbonylCarbonPosition + methylDirection * 1.51,
            name: "CH3",
            residueName: "ACE",
            residueSeq: residue.residueSeq,
            chainID: residue.chainID,
            isHetAtom: true
        ))
        bonds.append(Bond(id: bonds.count, atomIndex1: carbonylCarbonIndex, atomIndex2: methylIndex, order: .single))

        return true
    }

    private static func missingBondDirection(
        origin: SIMD3<Float>,
        existingNeighbors: [SIMD3<Float>],
        fallbackTarget: SIMD3<Float>?
    ) -> SIMD3<Float> {
        let neighborSum = existingNeighbors.reduce(SIMD3<Float>.zero) { partial, position in
            partial + normalized(position - origin, fallback: SIMD3<Float>.zero)
        }

        if simd_length_squared(neighborSum) > 1e-6 {
            return normalized(-neighborSum, fallback: SIMD3<Float>(1, 0, 0))
        }

        if let fallbackTarget {
            return normalized(fallbackTarget - origin, fallback: SIMD3<Float>(1, 0, 0))
        }

        return SIMD3<Float>(1, 0, 0)
    }

    private static func normalized(
        _ vector: SIMD3<Float>,
        fallback: SIMD3<Float>
    ) -> SIMD3<Float> {
        let lengthSquared = simd_length_squared(vector)
        guard lengthSquared > 1e-8 else { return fallback }
        return vector / sqrt(lengthSquared)
    }

    private static func planarReference(
        axis: SIMD3<Float>,
        preferred: SIMD3<Float>?
    ) -> SIMD3<Float> {
        if let preferred {
            let projected = preferred - simd_dot(preferred, axis) * axis
            if simd_length_squared(projected) > 1e-8 {
                return normalized(projected, fallback: arbitraryPerpendicular(to: axis))
            }
        }
        return arbitraryPerpendicular(to: axis)
    }

    private static func arbitraryPerpendicular(to axis: SIMD3<Float>) -> SIMD3<Float> {
        let seed = abs(axis.x) < 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        let perpendicular = simd_cross(axis, seed)
        if simd_length_squared(perpendicular) > 1e-8 {
            return normalized(perpendicular, fallback: SIMD3<Float>(0, 0, 1))
        }
        return SIMD3<Float>(0, 0, 1)
    }

    static func remapSubstructure(
        atoms: [Atom],
        bonds: [Bond],
        selectedIndices: [Int]
    ) -> (atoms: [Atom], bonds: [Bond]) {
        let selectedSet = Set(selectedIndices)
        var oldToNew: [Int: Int] = [:]
        var remappedAtoms: [Atom] = []
        remappedAtoms.reserveCapacity(selectedIndices.count)

        for oldIndex in selectedIndices {
            oldToNew[oldIndex] = remappedAtoms.count
            var atom = atoms[oldIndex]
            atom = Atom(
                id: remappedAtoms.count,
                element: atom.element,
                position: atom.position,
                name: atom.name,
                residueName: atom.residueName,
                residueSeq: atom.residueSeq,
                chainID: atom.chainID,
                charge: atom.charge,
                formalCharge: atom.formalCharge,
                isHetAtom: atom.isHetAtom,
                occupancy: atom.occupancy,
                tempFactor: atom.tempFactor,
                altLoc: atom.altLoc
            )
            remappedAtoms.append(atom)
        }

        var remappedBonds: [Bond] = []
        remappedBonds.reserveCapacity(bonds.count)
        for bond in bonds where selectedSet.contains(bond.atomIndex1) && selectedSet.contains(bond.atomIndex2) {
            guard let atom1 = oldToNew[bond.atomIndex1], let atom2 = oldToNew[bond.atomIndex2] else { continue }
            remappedBonds.append(Bond(
                id: remappedBonds.count,
                atomIndex1: atom1,
                atomIndex2: atom2,
                order: bond.order,
                isRotatable: bond.isRotatable
            ))
        }

        return (remappedAtoms, remappedBonds)
    }

    private static func applyElectrostaticFallback(to atoms: [Atom]) -> [Atom] {
        atoms.map { atom in
            var atom = atom
            if abs(atom.charge) <= 0.0001 {
                atom.charge = Float(atom.formalCharge)
            }
            return atom
        }
    }
}

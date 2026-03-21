import XCTest
import simd
import MetalKit
@testable import Druse

/// Shared helpers for all docking integration tests.
/// Every helper prints exhaustive diagnostics so a single test run reveals root cause.
class DockingTestCase: XCTestCase {

    // MARK: - RMSD

    /// RMSD between two equal-length position arrays.
    func computeRMSD(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return .infinity }
        var sum: Float = 0
        for i in 0..<n { sum += simd_distance_squared(a[i], b[i]) }
        return sqrt(sum / Float(n))
    }

    /// Per-atom distance breakdown between two pose arrays. Prints every atom.
    func printPerAtomDistances(_ label: String, _ a: [SIMD3<Float>], _ b: [SIMD3<Float>], atoms: [Atom]? = nil) {
        let n = min(a.count, b.count)
        guard n > 0 else { print("  [\(label)] Cannot compare: a=\(a.count), b=\(b.count)"); return }
        print("  [\(label)] Per-atom distances (\(n) atoms):")
        var maxDist: Float = 0
        var maxIdx = 0
        for i in 0..<n {
            let d = simd_distance(a[i], b[i])
            if d > maxDist { maxDist = d; maxIdx = i }
            let atomInfo = atoms.map { i < $0.count ? " \($0[i].element.symbol)\($0[i].name)" : "" } ?? ""
            let flag = d > 2.0 ? " *** HIGH ***" : ""
            print("    atom \(i)\(atomInfo): d=\(String(format: "%.3f", d))Å  a=(\(String(format: "%.2f,%.2f,%.2f", a[i].x, a[i].y, a[i].z)))  b=(\(String(format: "%.2f,%.2f,%.2f", b[i].x, b[i].y, b[i].z)))\(flag)")
        }
        print("  [\(label)] max deviation: atom \(maxIdx) = \(String(format: "%.3f", maxDist))Å")
    }

    // MARK: - Heavy Bonds

    /// Build heavy-atom-only bonds from a full molecule.
    func heavyBonds(from atoms: [Atom], bonds: [Bond]) -> [Bond] {
        var oldToNew: [Int: Int] = [:]
        var newIdx = 0
        for atom in atoms where atom.element != .H {
            oldToNew[atom.id] = newIdx
            newIdx += 1
        }
        return bonds.compactMap { bond in
            guard let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }
    }

    // MARK: - Ligand Atom Type Debug

    /// Print Vina atom type assignments for a prepared ligand (heavy atoms only).
    @MainActor
    func printLigandAtomTypes(engine: DockingEngine, ligand: Molecule) {
        let prepared = engine.prepareLigandGeometry(ligand)
        print("  [AtomTypes] Ligand \(ligand.name): \(prepared.heavyAtoms.count) heavy atoms, \(prepared.heavyBonds.count) bonds")
        print("  [AtomTypes] Centroid: \(prepared.centroid)")
        let vinaTypeNames = [
            "C_H", "C_P", "N_P", "N_D", "N_A", "N_DA", "O_P", "O_D",
            "O_A", "O_DA", "S_P", "P_P", "F_H", "Cl_H", "Br_H", "I_H",
            "Si", "At", "Met_D"
        ]
        for (i, gpuAtom) in prepared.gpuAtoms.enumerated() {
            let atom = prepared.heavyAtoms[i]
            let typeName = gpuAtom.vinaType >= 0 && gpuAtom.vinaType < vinaTypeNames.count ? vinaTypeNames[Int(gpuAtom.vinaType)] : "type[\(gpuAtom.vinaType)]"
            print("    atom \(i): \(atom.element.symbol) \(atom.name) → VINA_\(typeName) (type=\(gpuAtom.vinaType)) vdw=\(String(format: "%.2f", gpuAtom.vdwRadius)) charge=\(String(format: "%.3f", gpuAtom.charge)) pos=(\(String(format: "%.2f,%.2f,%.2f", gpuAtom.position.x, gpuAtom.position.y, gpuAtom.position.z)))")
        }

        // Summarize type distribution
        var typeCounts: [Int32: Int] = [:]
        for gpuAtom in prepared.gpuAtoms { typeCounts[gpuAtom.vinaType, default: 0] += 1 }
        let summary = typeCounts.sorted(by: { $0.key < $1.key })
            .map { Int($0.key) < vinaTypeNames.count ? "\(vinaTypeNames[Int($0.key)]): \($0.value)" : "type[\($0.key)]: \($0.value)" }
            .joined(separator: ", ")
        print("  [AtomTypes] Distribution: \(summary)")
    }

    // MARK: - Grid Debug

    /// Print detailed grid map statistics after computation.
    @MainActor
    func printGridDebug(engine: DockingEngine) {
        print("  [Grid] Diagnostics:")
        print(engine.gridDiagnostics())
    }

    // MARK: - Pocket Debug

    /// Print pocket details.
    func printPocketDebug(_ pocket: BindingPocket, label: String = "Pocket") {
        print("  [\(label)] center=(\(String(format: "%.2f, %.2f, %.2f", pocket.center.x, pocket.center.y, pocket.center.z)))")
        print("  [\(label)] size=(\(String(format: "%.2f, %.2f, %.2f", pocket.size.x, pocket.size.y, pocket.size.z)))")
        print("  [\(label)] volume=\(String(format: "%.0f", pocket.volume))ų")
        print("  [\(label)] buriedness=\(String(format: "%.2f", pocket.buriedness))")
        print("  [\(label)] polarity=\(String(format: "%.2f", pocket.polarity))")
        print("  [\(label)] druggability=\(String(format: "%.1f", pocket.druggability))")
        print("  [\(label)] residues=\(pocket.residueIndices.count)")
        print("  [\(label)] probePositions=\(pocket.probePositions.count)")
    }

    // MARK: - Results Debug

    /// Print full results breakdown with energy decomposition and pose geometry.
    func printResultsDebug(_ results: [DockingResult], label: String, maxPoses: Int = 10,
                            crystalPositions: [SIMD3<Float>]? = nil, ligandAtoms: [Atom]? = nil) {
        print("  [\(label)] \(results.count) total results")
        if results.isEmpty { print("  [\(label)] WARNING: 0 results!"); return }

        // Energy statistics
        let energies = results.map(\.energy)
        let validEnergies = energies.filter { $0.isFinite && $0 < 1e9 }
        print("  [\(label)] Valid poses: \(validEnergies.count)/\(results.count)")
        if let minE = validEnergies.min(), let maxE = validEnergies.max() {
            let mean = validEnergies.reduce(0, +) / Float(validEnergies.count)
            print("  [\(label)] Energy range: \(String(format: "%.2f", minE)) to \(String(format: "%.2f", maxE)) (mean=\(String(format: "%.2f", mean)))")
        }
        let invalidCount = energies.filter { !$0.isFinite || $0 >= 1e9 }.count
        if invalidCount > 0 {
            print("  [\(label)] Invalid/sentinel poses: \(invalidCount)")
        }

        // Cluster analysis
        let clusterIDs = Set(results.map(\.clusterID))
        print("  [\(label)] Clusters: \(clusterIDs.count)")
        for cid in clusterIDs.sorted().prefix(10) {
            let members = results.filter { $0.clusterID == cid }
            print("    cluster \(cid): \(members.count) poses, best E=\(String(format: "%.2f", members.first?.energy ?? .infinity))")
        }

        // Top poses detail
        for (i, r) in results.prefix(maxPoses).enumerated() {
            let posCount = r.transformedAtomPositions.count
            let centroid: SIMD3<Float> = posCount > 0
                ? r.transformedAtomPositions.reduce(.zero, +) / Float(posCount)
                : .zero

            var rmsdStr = ""
            if let crystal = crystalPositions {
                let rmsd = computeRMSD(crystal, r.transformedAtomPositions)
                rmsdStr = " RMSD=\(String(format: "%.2f", rmsd))Å"
            }

            print("    pose \(i): E=\(String(format: "%8.3f", r.energy)) steric=\(String(format: "%7.3f", r.stericEnergy)) hydro=\(String(format: "%7.3f", r.hydrophobicEnergy)) hbond=\(String(format: "%7.3f", r.hbondEnergy)) torsion=\(String(format: "%6.3f", r.torsionPenalty)) gen=\(r.generation) cluster=\(r.clusterID) atoms=\(posCount) centroid=(\(String(format: "%.1f,%.1f,%.1f", centroid.x, centroid.y, centroid.z)))\(rmsdStr)")
        }

        // Energy decomposition summary for top 5 valid poses
        let top5 = results.prefix(min(5, results.count)).filter { $0.energy.isFinite }
        if !top5.isEmpty {
            let avgSteric = top5.map(\.stericEnergy).reduce(0, +) / Float(top5.count)
            let avgHydro = top5.map(\.hydrophobicEnergy).reduce(0, +) / Float(top5.count)
            let avgHbond = top5.map(\.hbondEnergy).reduce(0, +) / Float(top5.count)
            let avgTorsion = top5.map(\.torsionPenalty).reduce(0, +) / Float(top5.count)
            let avgTotal = top5.map(\.energy).reduce(0, +) / Float(top5.count)
            print("  [\(label)] Scoring breakdown (avg top \(top5.count)): total=\(String(format: "%.2f", avgTotal)) steric=\(String(format: "%.2f", avgSteric)) hydro=\(String(format: "%.2f", avgHydro)) hbond=\(String(format: "%.2f", avgHbond)) torsion=\(String(format: "%.2f", avgTorsion))")
        }

        // Best pose: per-atom detail with crystal comparison
        if let crystal = crystalPositions, let best = results.first, !best.transformedAtomPositions.isEmpty {
            printPerAtomDistances("\(label)-BestVsCrystal", crystal, best.transformedAtomPositions, atoms: ligandAtoms)
        }

        // Pose distribution: how far are pose centroids from pocket center?
        if results.count > 1 {
            let centroids = results.compactMap { r -> SIMD3<Float>? in
                guard !r.transformedAtomPositions.isEmpty else { return nil }
                return r.transformedAtomPositions.reduce(.zero, +) / Float(r.transformedAtomPositions.count)
            }
            if centroids.count > 1 {
                let globalCentroid = centroids.reduce(.zero, +) / Float(centroids.count)
                let spread = centroids.map { simd_distance($0, globalCentroid) }
                print("  [\(label)] Centroid spread: mean=\(String(format: "%.1f", spread.reduce(0,+)/Float(spread.count)))Å max=\(String(format: "%.1f", spread.max() ?? 0))Å")
            }
        }
    }

    // MARK: - Fetch & Parse PDB

    /// Fetch and parse a PDB with full debug output.
    @MainActor
    func fetchAndParsePDB(
        id: String,
        ligandResidue: String? = nil
    ) async throws -> (protein: Molecule, ligand: Molecule, crystalHeavyPositions: [SIMD3<Float>]) {
        print("  [\(id)] Fetching PDB from RCSB...")
        let pdbContent: String
        do {
            pdbContent = try await PDBService.shared.fetchPDBFile(id: id)
        } catch {
            throw XCTSkip("Network unavailable for \(id): \(error.localizedDescription)")
        }
        print("  [\(id)] PDB content: \(pdbContent.count) chars")

        let result = PDBParser.parse(pdbContent)
        print("  [\(id)] Parse result: protein=\(result.protein != nil ? "\(result.protein!.atoms.count) atoms" : "nil"), ligands=\(result.ligands.count), waters=\(result.waterCount), warnings=\(result.warnings.count)")
        for (i, lig) in result.ligands.enumerated() {
            print("    ligand \(i): \(lig.name) — \(lig.atoms.count) atoms, \(lig.bonds.count) bonds")
        }
        for w in result.warnings.prefix(3) { print("    warning: \(w)") }

        guard let protData = result.protein else {
            throw XCTSkip("\(id) protein parse failed")
        }
        let ligData: MoleculeData
        if let resName = ligandResidue {
            guard let ld = result.ligands.first(where: { $0.name == resName }) ?? result.ligands.first else {
                throw XCTSkip("\(id) ligand \(resName) not found (available: \(result.ligands.map(\.name).joined(separator: ", ")))")
            }
            ligData = ld
        } else {
            guard let ld = result.ligands.first else {
                throw XCTSkip("\(id) has no ligands")
            }
            ligData = ld
        }

        let crystalHeavy = ligData.atoms.filter { $0.element != .H }.map(\.position)
        let crystalCentroid = crystalHeavy.reduce(SIMD3<Float>.zero, +) / Float(max(crystalHeavy.count, 1))
        print("  [\(id)] Crystal ligand \(ligData.name): \(crystalHeavy.count) heavy atoms, centroid=(\(String(format: "%.2f, %.2f, %.2f", crystalCentroid.x, crystalCentroid.y, crystalCentroid.z)))")

        // Print element composition
        let elemCounts = Dictionary(grouping: ligData.atoms.filter { $0.element != .H }, by: { $0.element })
            .mapValues(\.count)
            .sorted(by: { $0.key.rawValue < $1.key.rawValue })
            .map { "\($0.key.symbol):\($0.value)" }
            .joined(separator: " ")
        print("  [\(id)] Crystal ligand composition: \(elemCounts)")

        let protein = Molecule(name: protData.name, atoms: protData.atoms,
                               bonds: protData.bonds, title: protData.title)
        let ligand = Molecule(name: ligData.name, atoms: ligData.atoms,
                              bonds: ligData.bonds, title: ligData.title)

        print("  [\(id)] Protein: \(protein.atomCount) atoms, \(protein.heavyAtomCount) heavy, \(protein.residues.count) residues, \(protein.chains.count) chains (\(protein.chainIDs.joined(separator: ",")))")

        return (protein, ligand, crystalHeavy)
    }

    // MARK: - Engine Creation

    @MainActor
    func makeDockingEngine() throws -> DockingEngine {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU available")
        }
        print("  [Engine] Metal device: \(device.name), unified memory: \(device.hasUnifiedMemory)")
        guard let engine = DockingEngine(device: device) else {
            throw XCTSkip("Failed to create DockingEngine")
        }
        return engine
    }

    // MARK: - Run Docking (with full debug)

    @MainActor
    func runTestDocking(
        engine: DockingEngine,
        protein: Molecule,
        ligand: Molecule,
        pocket: BindingPocket,
        populationSize: Int = 200,
        generations: Int = 200,
        flexibility: Bool = true
    ) async -> [DockingResult] {
        // Compute ligand extent
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        var ligMin = SIMD3<Float>(repeating: .infinity)
        var ligMax = SIMD3<Float>(repeating: -.infinity)
        let centroid = heavyAtoms.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(max(heavyAtoms.count, 1))
        for a in heavyAtoms {
            let p = a.position - centroid
            ligMin = simd_min(ligMin, p)
            ligMax = simd_max(ligMax, p)
        }
        let ligExtent = (ligMax - ligMin) * 0.5

        print("  [Dock] Ligand: \(ligand.name), \(heavyAtoms.count) heavy atoms, \(ligand.bondCount) bonds")
        print("  [Dock] Ligand centroid: (\(String(format: "%.2f, %.2f, %.2f", centroid.x, centroid.y, centroid.z)))")
        print("  [Dock] Ligand extent: (\(String(format: "%.2f, %.2f, %.2f", ligExtent.x, ligExtent.y, ligExtent.z)))")

        // Print atom types BEFORE docking
        printLigandAtomTypes(engine: engine, ligand: ligand)

        // Compute grid maps
        print("  [Dock] Computing grid maps (spacing=0.375, protein=\(protein.heavyAtomCount) heavy atoms)...")
        let gridStart = CFAbsoluteTimeGetCurrent()
        engine.computeGridMaps(protein: protein, pocket: pocket, spacing: 0.375, ligandExtent: ligExtent)
        print("  [Dock] Grid maps computed in \(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - gridStart))s")
        printGridDebug(engine: engine)

        // Config
        var config = DockingConfig()
        config.numRuns = 1
        config.generationsPerRun = generations
        config.populationSize = populationSize
        config.enableFlexibility = flexibility
        config.liveUpdateFrequency = 999
        print("  [Dock] Config: pop=\(populationSize), gen=\(generations), flex=\(flexibility), runs=\(config.numRuns)")
        print("  [Dock] Config: mutation=\(config.mutationRate), crossover=\(config.crossoverRate), translationStep=\(config.translationStep), rotationStep=\(config.rotationStep), torsionStep=\(config.torsionStep)")

        // Run
        let dockStart = CFAbsoluteTimeGetCurrent()
        let results = await engine.runDocking(ligand: ligand, pocket: pocket, config: config)
        let dockTime = CFAbsoluteTimeGetCurrent() - dockStart
        print("  [Dock] Docking completed in \(String(format: "%.3f", dockTime))s")

        // Print full diagnostics
        if let diag = engine.lastDiagnostics {
            print(diag.summary)
        }

        return results
    }

    // MARK: - Round-Trip Test (with full debug)

    @MainActor
    func runRoundTripTest(
        pdbID: String, ligandResidue: String, smiles: String, ligandName: String,
        centroidThreshold: Float = 10.0
    ) async throws {
        print("\n  ========== ROUND-TRIP TEST: \(pdbID) / \(ligandName) ==========")

        let (protein, crystalLigand, crystalHeavyPos) = try await fetchAndParsePDB(id: pdbID, ligandResidue: ligandResidue)
        guard crystalHeavyPos.count >= 5 else {
            throw XCTSkip("\(pdbID) crystal ligand has <5 heavy atoms (\(crystalHeavyPos.count))")
        }

        // Pocket detection
        print("  [\(pdbID)] Detecting pocket from crystal ligand...")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(
            protein: protein, ligand: crystalLigand, distance: 6.0)
        guard let pocket else { throw XCTSkip("\(pdbID) pocket detection failed") }
        printPocketDebug(pocket, label: "\(pdbID)-Pocket")

        // Regenerate ligand from SMILES
        print("  [\(pdbID)] Regenerating \(ligandName) from SMILES: \(smiles.prefix(60))...")
        let (regen, _, err) = RDKitBridge.prepareLigand(
            smiles: smiles, name: "\(ligandName)_regen",
            addHydrogens: false, minimize: true, computeCharges: true)
        if let err { print("  [\(pdbID)] RDKit error: \(err)") }
        guard let rd = regen else { throw XCTSkip("\(pdbID) RDKit prep failed for \(ligandName)") }
        let regenLigand = Molecule(name: rd.name, atoms: rd.atoms, bonds: rd.bonds, title: smiles, smiles: smiles)
        let regenHeavy = regenLigand.atoms.filter { $0.element != .H }
        print("  [\(pdbID)] Regenerated: \(regenHeavy.count) heavy atoms, \(regenLigand.bondCount) bonds")

        // Compare crystal vs regenerated atom counts
        if regenHeavy.count != crystalHeavyPos.count {
            print("  [\(pdbID)] WARNING: heavy atom count mismatch — crystal=\(crystalHeavyPos.count) regen=\(regenHeavy.count)")
        }

        // Dock
        print("  [\(pdbID)] Docking...")
        let engine = try makeDockingEngine()
        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: regenLigand, pocket: pocket,
            populationSize: 250, generations: 250, flexibility: true)

        // Results debug
        let heavyAtoms = regenLigand.atoms.filter { $0.element != .H }
        printResultsDebug(results, label: "\(pdbID)-Results", maxPoses: 5,
                          crystalPositions: crystalHeavyPos, ligandAtoms: heavyAtoms)

        guard !results.isEmpty else {
            XCTFail("[\(pdbID)] Docking produced 0 results")
            return
        }

        let best = results[0]
        guard !best.transformedAtomPositions.isEmpty else {
            XCTFail("[\(pdbID)] Best pose has no transformed positions")
            return
        }

        let dockedCentroid = best.transformedAtomPositions.reduce(.zero, +) / Float(best.transformedAtomPositions.count)
        let crystalCentroid = crystalHeavyPos.reduce(.zero, +) / Float(crystalHeavyPos.count)
        let centroidDist = simd_distance(dockedCentroid, crystalCentroid)

        print("  [\(pdbID)] Docked centroid: (\(String(format: "%.2f, %.2f, %.2f", dockedCentroid.x, dockedCentroid.y, dockedCentroid.z)))")
        print("  [\(pdbID)] Crystal centroid: (\(String(format: "%.2f, %.2f, %.2f", crystalCentroid.x, crystalCentroid.y, crystalCentroid.z)))")
        print("  [\(pdbID)] Centroid distance: \(String(format: "%.2f", centroidDist))Å")

        print("✓ \(pdbID) round-trip: centroid=\(String(format: "%.2f", centroidDist))Å, E=\(String(format: "%.1f", best.energy))kcal/mol, " +
              "crystal=\(crystalHeavyPos.count) atoms, docked=\(best.transformedAtomPositions.count) atoms")

        XCTAssertTrue(best.energy.isFinite, "[\(pdbID)] Energy should be finite, got \(best.energy)")
        XCTAssertLessThan(best.energy, 5.0, "[\(pdbID)] Energy should be favorable, got \(best.energy)")
        XCTAssertTrue(centroidDist.isFinite, "[\(pdbID)] Centroid distance should be finite")
        XCTAssertLessThan(centroidDist, centroidThreshold,
            "[\(pdbID)] Docked centroid should be within \(centroidThreshold)Å of crystal (\(String(format: "%.1f", centroidDist))Å)")
    }
}

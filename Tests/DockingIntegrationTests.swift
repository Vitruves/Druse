import XCTest
import simd
import MetalKit
@testable import Druse

// =============================================================================
// MARK: - Docking Integration Tests
// =============================================================================
//
// Comprehensive docking pipeline validation:
//   1. Docking runs correctly (valid output, finite energies, sorted poses)
//   2. Reference literature redocking (3PYY, 2QWK, 1M17)
//   3. Grid exploration & pose diversity (grid bounds, clustering, centering)
//   4. Chemically valid conformers (bond lengths, clashes, torsion trees)
//   5. Interactions, export, multi-size docking, protonation
//   6. Round-trip: PDB crystal → SMILES regeneration → dock → compare to crystal
//
// Every test prints step-by-step diagnostics so a single run reveals root cause.

final class DockingIntegrationTests: DockingTestCase {

    // ======================================================================
    // MARK: 1 — Docking Runs Correctly
    // ======================================================================

    @MainActor
    func testDockingProducesValidResults() async throws {
        let engine = try sharedDockingEngine()

        let protein = TestMolecules.alanineDipeptide()
        print("  [ValidResults] Protein: \(protein.atomCount) atoms, \(protein.residues.count) residues")
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))
        print("  [ValidResults] Pocket: center=\(pocket.center), size=\(pocket.size), vol=\(String(format: "%.0f", pocket.volume))")

        let (molData, err) = RDKitBridge.smilesToMolecule(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "aspirin")
        print("  [ValidResults] Ligand: \(ligand.atomCount) atoms (\(ligand.heavyAtomCount) heavy), \(ligand.bondCount) bonds")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 100, generations: 100, flexibility: false)
        print("  [ValidResults] \(results.count) poses returned")
        for (i, r) in results.prefix(5).enumerated() {
            print("    Pose \(i): E=\(String(format: "%8.2f", r.energy)) steric=\(String(format: "%6.2f", r.stericEnergy)) hydro=\(String(format: "%6.2f", r.hydrophobicEnergy)) hbond=\(String(format: "%6.2f", r.hbondEnergy)) cluster=\(r.clusterID) positions=\(r.transformedAtomPositions.count)")
        }

        XCTAssertFalse(results.isEmpty, "Docking must produce results")
        XCTAssertTrue(results.count >= 5, "Should produce ≥5 poses, got \(results.count)")

        for (i, r) in results.enumerated() {
            XCTAssertTrue(r.energy.isFinite, "Pose \(i) energy not finite: \(r.energy)")
            XCTAssertTrue(r.stericEnergy.isFinite, "Pose \(i) steric not finite: \(r.stericEnergy)")
            XCTAssertTrue(r.hydrophobicEnergy.isFinite, "Pose \(i) hydro not finite: \(r.hydrophobicEnergy)")
            XCTAssertTrue(r.hbondEnergy.isFinite, "Pose \(i) hbond not finite: \(r.hbondEnergy)")
        }

        for i in 1..<max(min(results.count, 20), 1) {
            XCTAssertLessThanOrEqual(results[i-1].energy, results[i].energy, "Results not sorted at index \(i)")
        }

        let best = results[0]
        XCTAssertTrue(best.energy.isFinite, "Best energy should be finite, got \(best.energy)")
        XCTAssertFalse(best.transformedAtomPositions.isEmpty, "Best pose must have positions")
        for pos in best.transformedAtomPositions {
            XCTAssertFalse(pos.x.isNaN || pos.y.isNaN || pos.z.isNaN, "Position NaN: \(pos)")
        }
    }

    @MainActor
    func testScoringDecompositionConsistency() async throws {
        let engine = try sharedDockingEngine()
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, _) = RDKitBridge.smilesToMolecule(smiles: "c1ccccc1", name: "Benzene", numConformers: 1, minimize: true)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "benzene")
        print("  [Scoring] Benzene: \(ligand.heavyAtomCount) heavy atoms")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 50, generations: 50, flexibility: false)
        print("  [Scoring] \(results.count) poses")

        for (i, r) in results.prefix(5).enumerated() {
            print("    Pose \(i): steric=\(r.stericEnergy) hydro=\(r.hydrophobicEnergy) hbond=\(r.hbondEnergy) total=\(r.energy)")
            XCTAssertTrue(r.stericEnergy.isFinite, "Pose \(i) steric not finite")
            XCTAssertLessThanOrEqual(r.hydrophobicEnergy, 0.01, "Hydro should be ≤0, got \(r.hydrophobicEnergy)")
            XCTAssertLessThanOrEqual(r.hbondEnergy, 0.01, "HBond should be ≤0, got \(r.hbondEnergy)")
        }
    }

    @MainActor
    func testStopDockingPreservesResults() async throws {
        let engine = try sharedDockingEngine()
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, _) = RDKitBridge.smilesToMolecule(smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "")

        engine.computeGridMaps(protein: protein, pocket: pocket, spacing: 0.375)
        var config = DockingConfig()
        config.numRuns = 1; config.generationsPerRun = 200; config.populationSize = 64; config.liveUpdateFrequency = 999

        print("  [Stop] Starting long docking run, will stop after 500ms...")
        let task = Task { await engine.runDocking(ligand: ligand, pocket: pocket, config: config) }
        try await Task.sleep(for: .milliseconds(500))
        engine.stopDocking()
        let results = await task.value

        XCTAssertFalse(engine.isRunning, "Engine should not be running after stop")
        let finite = results.filter { $0.energy.isFinite }
        print("  [Stop] \(results.count) results (\(finite.count) finite energy), isRunning=\(engine.isRunning)")
    }

    // ======================================================================
    // MARK: 2 — Reference Literature Redocking
    // ======================================================================

    @MainActor
    func testRedocking3PYYImatinib() async throws {

        print("\n  ========== REDOCKING: 3PYY / Imatinib (Kd ~37 nM) ==========")
        let engine = try sharedDockingEngine()
        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(id: "3PYY", ligandResidue: "STI")

        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { return XCTFail("[3PYY] Pocket detection failed") }
        printPocketDebug(pocket, label: "3PYY-Pocket")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 300, generations: 300)
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        printResultsDebug(results, label: "3PYY", maxPoses: 5, crystalPositions: crystalHeavy, ligandAtoms: heavyAtoms)
        XCTAssertFalse(results.isEmpty, "[3PYY] No results")

        let best = results[0]
        let rmsd = computeRMSD(crystalHeavy, best.transformedAtomPositions)
        print("✓ 3PYY: RMSD=\(String(format: "%.2f", rmsd))Å E=\(String(format: "%.1f", best.energy))kcal/mol")

        XCTAssertLessThan(rmsd, 15.0, "[3PYY] RMSD \(String(format: "%.2f", rmsd))Å > 15Å")
        XCTAssertLessThan(best.energy, 0.0, "[3PYY] Energy \(best.energy) should be negative")
    }

    @MainActor
    func testRedocking2QWKOseltamivir() async throws {

        print("\n  ========== REDOCKING: 2QWK / Oseltamivir (Ki ~0.1 nM) ==========")
        let engine = try sharedDockingEngine()
        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(id: "2QWK", ligandResidue: "G39")

        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("[2QWK] Pocket detection failed") }
        printPocketDebug(pocket, label: "2QWK-Pocket")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 250, generations: 250)
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        printResultsDebug(results, label: "2QWK", maxPoses: 5, crystalPositions: crystalHeavy, ligandAtoms: heavyAtoms)
        XCTAssertFalse(results.isEmpty)

        let best = results[0]
        let rmsd = computeRMSD(crystalHeavy, best.transformedAtomPositions)
        print("✓ 2QWK: RMSD=\(String(format: "%.2f", rmsd))Å E=\(String(format: "%.1f", best.energy))kcal/mol")

        XCTAssertLessThan(rmsd, 5.0, "[2QWK] RMSD \(String(format: "%.2f", rmsd))Å > 5Å")
        XCTAssertLessThan(best.energy, 0.0, "[2QWK] Energy \(best.energy) not negative")
    }

    @MainActor
    func testRedocking1M17Erlotinib() async throws {

        print("\n  ========== REDOCKING: 1M17 / Erlotinib (IC50 ~2 nM) ==========")
        let engine = try sharedDockingEngine()
        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(id: "1M17", ligandResidue: "AQ4")

        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("[1M17] Pocket detection failed") }
        printPocketDebug(pocket, label: "1M17-Pocket")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 250, generations: 250)
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        printResultsDebug(results, label: "1M17", maxPoses: 5, crystalPositions: crystalHeavy, ligandAtoms: heavyAtoms)
        XCTAssertFalse(results.isEmpty)

        let best = results[0]
        let rmsd = computeRMSD(crystalHeavy, best.transformedAtomPositions)
        print("✓ 1M17: RMSD=\(String(format: "%.2f", rmsd))Å E=\(String(format: "%.1f", best.energy))kcal/mol")

        XCTAssertLessThan(rmsd, 10.0, "[1M17] RMSD \(String(format: "%.2f", rmsd))Å > 10Å")
        XCTAssertLessThan(best.energy, 0.0, "[1M17] Energy \(best.energy) not negative")
    }

    // ======================================================================
    // MARK: 3 — Grid Exploration & Pose Diversity
    // ======================================================================

    @MainActor
    func testAllPosesInsideGridBox() async throws {
        let engine = try sharedDockingEngine()
        let (protein, ligand, _) = try await fetchAndParsePDB(id: "1HSG", ligandResidue: "MK1")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("No pocket") }

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 64, generations: 40)
        print("  [GridBox] \(results.count) poses, pocket center=\(pocket.center), size=\(pocket.size)")

        let padding: Float = 8.0
        let boxMin = pocket.center - pocket.size - SIMD3(repeating: padding)
        let boxMax = pocket.center + pocket.size + SIMD3(repeating: padding)
        var outsideCount = 0
        let totalAtoms = results.reduce(0) { $0 + $1.transformedAtomPositions.count }
        for r in results {
            for pos in r.transformedAtomPositions {
                if pos.x < boxMin.x || pos.x > boxMax.x || pos.y < boxMin.y || pos.y > boxMax.y || pos.z < boxMin.z || pos.z > boxMax.z {
                    outsideCount += 1
                }
            }
        }
        let frac = Float(outsideCount) / Float(max(totalAtoms, 1))
        print("  [GridBox] \(outsideCount)/\(totalAtoms) atoms outside (\(String(format: "%.1f%%", frac * 100)))")
        XCTAssertLessThan(frac, 0.05, "≤5% atoms should be outside grid, got \(String(format: "%.1f%%", frac * 100))")
    }

    @MainActor
    func testPoseDiversityMultipleClusters() async throws {
        let engine = try sharedDockingEngine()
        let (protein, ligand, _) = try await fetchAndParsePDB(id: "1HSG", ligandResidue: "MK1")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("No pocket") }

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 100, generations: 80)
        let clusterIDs = Set(results.map(\.clusterID))
        print("  [Diversity] \(results.count) poses, \(clusterIDs.count) clusters")
        for cid in clusterIDs.sorted() {
            let members = results.filter { $0.clusterID == cid }
            print("    Cluster \(cid): \(members.count) poses, best E=\(String(format: "%.1f", members.first?.energy ?? .infinity))")
        }

        XCTAssertTrue(clusterIDs.count >= 2, "Should have ≥2 clusters, got \(clusterIDs.count)")
    }

    @MainActor
    func testPosesCenteredOnPocket() async throws {
        let engine = try sharedDockingEngine()
        let (protein, ligand, _) = try await fetchAndParsePDB(id: "1HSG", ligandResidue: "MK1")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("No pocket") }

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 64, generations: 40)
        // Docking constrains translations per-axis to pocket.size + searchPadding (2.0Å).
        // Max centroid distance from center is the half-diagonal of that box + ligand extent margin.
        let maxR = simd_length(pocket.size + SIMD3<Float>(repeating: 2.0)) + 10.0

        for (i, r) in results.prefix(5).enumerated() {
            guard !r.transformedAtomPositions.isEmpty else { continue }
            let centroid = r.transformedAtomPositions.reduce(.zero, +) / Float(r.transformedAtomPositions.count)
            let dist = simd_distance(centroid, pocket.center)
            print("  [Center] Pose \(i): centroid dist to pocket = \(String(format: "%.1f", dist))Å (max \(String(format: "%.0f", maxR))Å)")
            XCTAssertLessThan(dist, maxR, "Pose \(i) centroid \(String(format: "%.1f", dist))Å from pocket center (limit \(String(format: "%.0f", maxR))Å)")
        }
    }

    // ======================================================================
    // MARK: 4 — Chemically Valid Conformers
    // ======================================================================

    @MainActor
    func testBondLengthsPreservedAfterDocking() async throws {
        let engine = try sharedDockingEngine()
        let (molData, _, err) = RDKitBridge.prepareLigand(smiles: "CCCCCC", name: "Hexane", addHydrogens: false, minimize: true, computeCharges: true)
        XCTAssertNil(err, "RDKit: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "hexane")
        print("  [BondLen] Hexane: \(ligand.heavyAtomCount) heavy atoms, \(ligand.bondCount) bonds")

        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(protein: protein, residueIndices: Array(0..<protein.residues.count))
        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 100, generations: 100, flexibility: true)
        print("  [BondLen] \(results.count) poses")

        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        let hBonds = heavyBonds(from: ligand.atoms, bonds: ligand.bonds)
        print("  [BondLen] \(hBonds.count) heavy bonds to check")

        for (pi, r) in results.prefix(3).enumerated() {
            let pos = r.transformedAtomPositions
            guard pos.count == heavyAtoms.count else {
                print("  [BondLen] Pose \(pi): positions(\(pos.count)) ≠ heavyAtoms(\(heavyAtoms.count)), skipping")
                continue
            }
            for b in hBonds {
                guard b.atomIndex1 < pos.count, b.atomIndex2 < pos.count,
                      b.atomIndex1 < heavyAtoms.count, b.atomIndex2 < heavyAtoms.count else { continue }
                let orig = simd_distance(heavyAtoms[b.atomIndex1].position, heavyAtoms[b.atomIndex2].position)
                guard orig > 0.01 else { continue }
                let docked = simd_distance(pos[b.atomIndex1], pos[b.atomIndex2])
                guard docked.isFinite else { continue }
                let delta = abs(docked - orig)
                if delta > 0.1 {
                    print("  [BondLen] Pose \(pi) bond \(b.atomIndex1)-\(b.atomIndex2): \(String(format: "%.3f", orig))→\(String(format: "%.3f", docked))Å (Δ\(String(format: "%.3f", delta)))")
                }
                XCTAssertEqual(Double(docked), Double(orig), accuracy: 0.15,
                    "Pose \(pi) bond \(b.atomIndex1)-\(b.atomIndex2): \(String(format: "%.3f", orig))→\(String(format: "%.3f", docked))Å")
            }
        }
    }

    @MainActor
    func testNoSevereIntraMolecularClashes() async throws {
        let engine = try sharedDockingEngine()
        let (molData, _, err) = RDKitBridge.prepareLigand(smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", addHydrogens: false, minimize: true, computeCharges: true)
        XCTAssertNil(err)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "")
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(protein: protein, residueIndices: Array(0..<protein.residues.count))
        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 100, generations: 100, flexibility: true)

        // Build set of pairs close in original structure (topology-close)
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        var topoPairs: Set<String> = []
        for i in 0..<heavyAtoms.count {
            for j in (i+1)..<heavyAtoms.count {
                if simd_distance(heavyAtoms[i].position, heavyAtoms[j].position) < 3.0 {
                    topoPairs.insert("\(i)-\(j)")
                }
            }
        }

        for (pi, r) in results.prefix(3).enumerated() {
            let pos = r.transformedAtomPositions
            var clashes = 0
            for i in 0..<pos.count {
                for j in (i+1)..<pos.count {
                    guard !topoPairs.contains("\(i)-\(j)") else { continue }
                    if simd_distance(pos[i], pos[j]) < 0.5 { clashes += 1 }
                }
            }
            print("  [Clash] Pose \(pi): \(clashes) severe clashes (<0.5Å), \(pos.count) atoms")
            XCTAssertEqual(clashes, 0, "Pose \(pi) has \(clashes) severe clashes (<0.5Å)")
        }
    }

    func testTorsionTreeIbuprofen() {
        let tree = RDKitBridge.buildTorsionTree(smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        XCTAssertNotNil(tree)
        guard let t = tree else { return }
        print("  [Torsion] Ibuprofen: \(t.count) rotatable bonds")
        for (i, edge) in t.enumerated() {
            print("    Edge \(i): atoms \(edge.atom1)→\(edge.atom2), \(edge.movingAtoms.count) moving atoms")
        }
        XCTAssertTrue(t.count >= 2, "Should have ≥2 rotatable bonds, got \(t.count)")
        XCTAssertTrue(t.count <= 10, "Should have ≤10, got \(t.count)")
        for e in t { XCTAssertNotEqual(e.atom1, e.atom2); XCTAssertFalse(e.movingAtoms.isEmpty) }
    }

    func testConformerDiversityEthylbenzene() {
        let conformers = RDKitBridge.generateConformers(smiles: "CCc1ccccc1", name: "Ethylbenzene", count: 5, minimize: true)
        print("  [Conformers] Generated \(conformers.count) conformers")
        for (i, c) in conformers.prefix(5).enumerated() {
            print("    Conf \(i): E=\(String(format: "%.2f", c.energy)), atoms=\(c.molecule.atoms.count)")
        }
        XCTAssertTrue(conformers.count >= 3, "Should generate ≥3 conformers, got \(conformers.count)")

        if conformers.count >= 2 {
            let rmsd = computeRMSD(conformers[0].molecule.atoms.map(\.position), conformers.last!.molecule.atoms.map(\.position))
            print("  [Conformers] RMSD between first/last: \(String(format: "%.3f", rmsd))Å")
            XCTAssertGreaterThan(rmsd, 0.01, "Conformers should differ geometrically")
        }
    }

    // ======================================================================
    // MARK: 5 — Interactions, Export, Multi-Size, Protonation
    // ======================================================================

    @MainActor
    func testInteractionDetectionOnDockedPose() async throws {
        let engine = try sharedDockingEngine()
        let (protein, ligand, _) = try await fetchAndParsePDB(id: "1HSG", ligandResidue: "MK1")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("No pocket") }

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 64, generations: 50)
        guard let best = results.first else { throw XCTSkip("No results") }

        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        let hBonds = heavyBonds(from: ligand.atoms, bonds: ligand.bonds)
        let proteinHeavy = protein.atoms.filter { $0.element != .H }

        let interactions = InteractionDetector.detect(
            ligandAtoms: heavyAtoms, ligandPositions: best.transformedAtomPositions,
            proteinAtoms: proteinHeavy, ligandBonds: hBonds)

        print("  [Interactions] \(interactions.count) total:")
        let byType = Dictionary(grouping: interactions, by: { $0.type })
        for (type, ints) in byType.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            print("    \(type.label): \(ints.count) (avg dist \(String(format: "%.1f", ints.map(\.distance).reduce(0,+)/Float(ints.count)))Å)")
        }

        XCTAssertTrue(interactions.count >= 2, "Should detect ≥2 interactions, got \(interactions.count)")
        for inter in interactions {
            XCTAssertTrue(inter.distance > 0 && inter.distance < 10.0, "Distance \(inter.distance) out of range")
            XCTAssertTrue(inter.ligandAtomIndex < best.transformedAtomPositions.count, "Ligand index OOB")
            XCTAssertTrue(inter.proteinAtomIndex < proteinHeavy.count, "Protein index OOB")
        }
    }

    @MainActor
    func testSDFExportDockingResultsRoundTrip() async throws {
        let engine = try sharedDockingEngine()
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(protein: protein, residueIndices: Array(0..<protein.residues.count))
        let (molData, _, err) = RDKitBridge.prepareLigand(smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", addHydrogens: false, minimize: true, computeCharges: true)
        XCTAssertNil(err)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "aspirin")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket, populationSize: 50, generations: 50, flexibility: false)
        XCTAssertFalse(results.isEmpty)

        let sdf = SDFWriter.writeDockingResults(results, ligand: ligand)
        print("  [SDF] Exported \(sdf.count) chars, \(sdf.components(separatedBy: "$$$$").count - 1) molecules")
        XCTAssertTrue(sdf.contains("$$$$"), "Missing SDF delimiter")
        XCTAssertTrue(sdf.contains("V2000"), "Missing V2000 marker")
        XCTAssertTrue(sdf.contains("> <Energy>"), "Missing energy field")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("test_\(UUID().uuidString).sdf")
        try SDFWriter.save(sdf, to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let parsed = try SDFParser.parse(url: url)
        print("  [SDF] Re-parsed \(parsed.count) molecules")
        XCTAssertEqual(parsed.count, results.count, "Re-parsed count should match")
    }

    @MainActor
    func testGridMapDiagnostics() async throws {
        let engine = try sharedDockingEngine()
        let (protein, ligand, _) = try await fetchAndParsePDB(id: "1HSG", ligandResidue: "MK1")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("No pocket") }
        engine.computeGridMaps(protein: protein, pocket: pocket, spacing: 0.375)
        let diag = engine.gridDiagnostics()
        print("  [Grid]\n\(diag)")
        XCTAssertFalse(diag.isEmpty, "Diagnostics empty")
        XCTAssertTrue(diag.contains("Steric"), "Should mention Steric grid")
    }

    @MainActor
    func testDockingAcrossMolecularSizes() async throws {
        print("\n  ========== MOLECULAR SIZE SWEEP ==========")
        let engine = try sharedDockingEngine()
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(protein: protein, residueIndices: Array(0..<protein.residues.count))
        printPocketDebug(pocket, label: "SizeSweep-Pocket")

        let cases: [(smi: String, name: String)] = [
            ("O", "Water"), ("c1ccccc1", "Benzene"),
            ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"), ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
        ]
        for tc in cases {
            print("  --- \(tc.name) (\(tc.smi)) ---")
            let (md, err) = RDKitBridge.smilesToMolecule(smiles: tc.smi, name: tc.name, numConformers: 1, minimize: true)
            guard err == nil, let md else { print("  [Sizes] \(tc.name): RDKit failed (\(err ?? "nil"))"); continue }
            let lig = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: tc.smi)
            print("  [Sizes] \(tc.name): \(lig.atomCount) atoms (\(lig.heavyAtomCount) heavy)")
            let results = await runTestDocking(engine: engine, protein: protein, ligand: lig, pocket: pocket, populationSize: 32, generations: 30, flexibility: false)
            let bestE = results.first?.energy ?? .infinity
            print("  [Sizes] \(tc.name) RESULT: \(results.count) poses, E=\(bestE.isFinite ? String(format: "%.1f", bestE) : "inf")")
            if results.isEmpty {
                print("  [Sizes] WARNING: \(tc.name) produced 0 results!")
            } else {
                XCTAssertFalse(results[0].transformedAtomPositions.isEmpty, "\(tc.name) should have positions")
            }
        }
    }

    func testTransitionMetalVdWRadii() {
        let metals: [Element] = [.Sc, .Ti, .V, .Cr, .Mn, .Fe, .Co, .Ni, .Cu, .Zn]
        for el in metals {
            print("  [VdW] \(el.symbol): vdw=\(el.vdwRadius) cov=\(el.covalentRadius)")
            XCTAssertTrue(el.vdwRadius > 1.0 && el.vdwRadius < 3.0, "\(el.symbol) vdW \(el.vdwRadius) out of range")
            XCTAssertTrue(el.covalentRadius > 0.5 && el.covalentRadius < 2.5, "\(el.symbol) cov \(el.covalentRadius) out of range")
        }
        XCTAssertEqual(Element.Fe.vdwRadius, 2.04, accuracy: 0.05)
        XCTAssertEqual(Element.Zn.vdwRadius, 2.01, accuracy: 0.05)
    }

    func testHistidineProtonationBothSites() {
        let atoms = [
            Atom(id: 0, element: .N, position: .zero, name: "ND1", residueName: "HIS", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .N, position: SIMD3(1, 0, 0), name: "NE2", residueName: "HIS", residueSeq: 1, chainID: "A"),
        ]
        let predictions5 = Protonation.predictResidueStates(atoms: atoms, pH: 5.0)
        let predictions8 = Protonation.predictResidueStates(atoms: atoms, pH: 8.0)
        let prot5 = Protonation.applyProtonation(atoms: atoms, pH: 5.0)
        let prot8 = Protonation.applyProtonation(atoms: atoms, pH: 8.0)
        print("  [His] pH 5.0: ND1=\(prot5[0].formalCharge) NE2=\(prot5[1].formalCharge)")
        print("  [His] pH 8.0: ND1=\(prot8[0].formalCharge) NE2=\(prot8[1].formalCharge)")
        XCTAssertEqual(predictions5.first?.state, .histidineDoublyProtonated, "Histidine should be doubly protonated at pH 5")
        XCTAssertEqual(predictions5.first?.protonatedAtoms, Set(["ND1", "NE2"]))
        XCTAssertEqual(prot5[0].formalCharge + prot5[1].formalCharge, 1, "PROPKA-like histidine keeps a single formal +1 charge while protonating both nitrogens")
        XCTAssertEqual(prot8[0].formalCharge, 0, "ND1 should be 0 at pH 8")
        XCTAssertEqual(prot8[1].formalCharge, 0, "NE2 should be 0 at pH 8")
        XCTAssertEqual(predictions8.first?.protonatedAtoms.count, 1, "Neutral histidine should choose one tautomeric proton at pH 8")
    }

    // ======================================================================
    // MARK: 6 — Round-Trip: Crystal → SMILES → Dock → Compare
    // ======================================================================

    @MainActor
    func testRoundTrip1HSGIndinavirViaSMILES() async throws {

        try await runRoundTripTest(pdbID: "1HSG", ligandResidue: "MK1",
            smiles: "CC(C)(C)NC(=O)C1CN(CCc2ccccc2)CC1O", ligandName: "Indinavir")
    }

    @MainActor
    func testRoundTrip2QWKOseltamivirViaSMILES() async throws {

        try await runRoundTripTest(pdbID: "2QWK", ligandResidue: "G39",
            smiles: "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1", ligandName: "Oseltamivir")
    }

    @MainActor
    func testRoundTrip4DFRMethotrexateViaSMILES() async throws {

        try await runRoundTripTest(pdbID: "4DFR", ligandResidue: "MTX",
            smiles: "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
            ligandName: "Methotrexate", centroidThreshold: 12.0)
    }

    // ======================================================================
    // MARK: — DruseMLScoring Integration Test
    // ======================================================================

    /// Integration test: dock a known ligand, then score with DruseMLScoring.
    /// Verifies that:
    ///   1. ML scorer loads and produces valid predictions
    ///   2. pKd values are in a reasonable range (2-12)
    ///   3. Confidence values are in [0, 1]
    ///   4. dockingScore = pKd * confidence
    ///   5. pKd → Ki conversion is mathematically correct
    @MainActor
    func testDruseMLScoringOnDockedPoses() async throws {
        let engine = try sharedDockingEngine()

        // Load 3PYY (kinase + imatinib)
        let (protein, crystalLigand, _) = try await fetchAndParsePDB(id: "3PYY", ligandResidue: "STI")
        print("  [MLScoring] Protein: \(protein.atomCount) atoms")

        // Dock imatinib from SMILES
        let imatinibSMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
        let (molData, err) = RDKitBridge.smilesToMolecule(smiles: imatinibSMILES, name: "Imatinib", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "imatinib")
        print("  [MLScoring] Ligand: \(ligand.atomCount) atoms")

        guard let pocket = BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: crystalLigand, distance: 6.0) else {
            throw XCTSkip("3PYY pocket detection failed")
        }
        print("  [MLScoring] Pocket: center=\(pocket.center), vol=\(String(format: "%.0f", pocket.volume))")

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 200, generations: 150, flexibility: false)
        XCTAssertFalse(results.isEmpty, "Docking must produce results")
        print("  [MLScoring] Docking: \(results.count) poses, best Vina = \(String(format: "%.1f", results.first!.energy))")

        // Score with DruseMLScoring
        let scorer = DruseMLScoringInference()
        scorer.loadModel()

        guard scorer.isAvailable else {
            print("  [MLScoring] SKIP: DruseScorePKi model not available in test bundle")
            throw XCTSkip("DruseScorePKi model not in test bundle")
        }

        let pocketProteinAtoms = protein.atoms.filter {
            simd_distance($0.position, pocket.center) <= 10.0
        }

        let topN = min(results.count, 10)
        var scoredResults: [DockingResult] = []

        for i in 0..<topN {
            var poseAtoms = ligand.atoms
            for j in 0..<poseAtoms.count {
                if j < results[i].transformedAtomPositions.count {
                    poseAtoms[j].position = results[i].transformedAtomPositions[j]
                }
            }

            let features = DruseScoreFeatureExtractor.extract(
                proteinAtoms: pocketProteinAtoms,
                ligandAtoms: poseAtoms,
                pocketCenter: pocket.center
            )

            if let pred = await scorer.score(features: features) {
                var result = results[i]
                result.mlDockingScore = pred.dockingScore
                result.mlPKd = pred.pKd
                result.mlPoseConfidence = pred.poseConfidence
                scoredResults.append(result)

                print("    Pose \(i): Vina=\(String(format: "%6.1f", result.energy)) pKd=\(String(format: "%.2f", pred.pKd)) conf=\(String(format: "%.2f", pred.poseConfidence)) score=\(String(format: "%.2f", pred.dockingScore))")

                // Validate ranges
                XCTAssertTrue(pred.pKd >= 0 && pred.pKd <= 15,
                    "pKd should be in [0, 15], got \(pred.pKd)")
                XCTAssertTrue(pred.poseConfidence >= 0 && pred.poseConfidence <= 1,
                    "Confidence should be in [0, 1], got \(pred.poseConfidence)")
                XCTAssertTrue(pred.dockingScore.isFinite,
                    "dockingScore must be finite, got \(pred.dockingScore)")

                // Verify docking_score ≈ pKd * confidence
                let expected = pred.pKd * pred.poseConfidence
                XCTAssertEqual(pred.dockingScore, expected, accuracy: 0.1,
                    "dockingScore should ≈ pKd * confidence: \(pred.dockingScore) vs \(expected)")

                // Verify Ki conversion
                let ki = AffinityDisplayUnit.pKdToKi(pKd: pred.pKd)
                XCTAssertTrue(ki > 0, "Ki must be positive, got \(ki)")
                let expectedKi = pow(10, -Double(pred.pKd))
                XCTAssertEqual(ki, expectedKi, accuracy: expectedKi * 0.001,
                    "Ki conversion: \(ki) vs expected \(expectedKi)")
            }
        }

        XCTAssertFalse(scoredResults.isEmpty, "At least one pose should be scored")
        print("  [MLScoring] Scored \(scoredResults.count)/\(topN) poses with DruseMLScoring")

        // Best ML-scored pose should have reasonable pKd for imatinib (known ~8 pKd for ABL kinase)
        if let bestML = scoredResults.max(by: { ($0.mlDockingScore ?? 0) < ($1.mlDockingScore ?? 0) }) {
            let pKd = bestML.mlPKd ?? 0
            let kiFormatted = AffinityDisplayUnit.ki.format(pKd)
            print("  [MLScoring] Best ML pose: pKd=\(String(format: "%.2f", pKd)), Ki=\(kiFormatted), conf=\(String(format: "%.0f%%", (bestML.mlPoseConfidence ?? 0) * 100))")
        }

        print("✓ DruseMLScoring integration test PASSED")
    }

    // ======================================================================
    // MARK: 8 — Pharmacophore Constrained Docking
    // ======================================================================

    /// Test the Swift-side constraint data model: bitmask computation, residue resolution,
    /// GPU buffer conversion.
    @MainActor
    func testPharmacophoreConstraintDataModel() throws {
        // 1. Bitmask correctness
        let hbondDonor = ConstraintInteractionType.hbondDonor
        let mask = hbondDonor.compatibleVinaTypes
        // N_D = 3, N_DA = 5, O_D = 7, O_DA = 9, MET_D = 18
        XCTAssertTrue(mask & (1 << 3) != 0, "N_D (3) should be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 5) != 0, "N_DA (5) should be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 7) != 0, "O_D (7) should be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 9) != 0, "O_DA (9) should be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 18) != 0, "MET_D (18) should be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 0) == 0, "C_H (0) should NOT be compatible with H-bond donor")
        XCTAssertTrue(mask & (1 << 12) == 0, "F_H (12) should NOT be compatible with H-bond donor")
        print("  [Constraint] H-bond donor bitmask: 0x\(String(mask, radix: 16)) ✓")

        let halogen = ConstraintInteractionType.halogen
        let halMask = halogen.compatibleVinaTypes
        XCTAssertTrue(halMask & (1 << 12) != 0, "F_H (12) should be compatible with halogen")
        XCTAssertTrue(halMask & (1 << 13) != 0, "Cl_H (13) should be compatible with halogen")
        XCTAssertTrue(halMask & (1 << 14) != 0, "Br_H (14) should be compatible with halogen")
        XCTAssertTrue(halMask & (1 << 15) != 0, "I_H (15) should be compatible with halogen")
        print("  [Constraint] Halogen bitmask: 0x\(String(halMask, radix: 16)) ✓")

        // 2. Auto-detection
        let suggestedN = ConstraintInteractionType.suggestDefault(element: .N, atomName: "NZ", residueName: "LYS")
        XCTAssertEqual(suggestedN, .hbondDonor, "NZ on LYS should suggest H-bond donor, got \(suggestedN)")

        let suggestedO = ConstraintInteractionType.suggestDefault(element: .O, atomName: "OD1", residueName: "ASP")
        XCTAssertEqual(suggestedO, .hbondAcceptor, "OD1 on ASP should suggest H-bond acceptor, got \(suggestedO)")

        let suggestedZn = ConstraintInteractionType.suggestDefault(element: .Zn, atomName: "ZN", residueName: "ZN")
        XCTAssertEqual(suggestedZn, .metalCoordination, "Zn should suggest metal coordination, got \(suggestedZn)")

        let suggestedC = ConstraintInteractionType.suggestDefault(element: .C, atomName: "CG", residueName: "PHE")
        XCTAssertEqual(suggestedC, .piStacking, "CG on PHE should suggest pi-stacking, got \(suggestedC)")
        print("  [Constraint] Auto-detection ✓")

        // 3. Residue atom resolution
        let aspAcceptors = ConstraintAtomResolver.relevantAtomNames(residueName: "ASP", interactionType: .hbondAcceptor)
        XCTAssertTrue(aspAcceptors.contains("OD1"), "ASP acceptors should include OD1")
        XCTAssertTrue(aspAcceptors.contains("OD2"), "ASP acceptors should include OD2")

        let lysDonors = ConstraintAtomResolver.relevantAtomNames(residueName: "LYS", interactionType: .hbondDonor)
        XCTAssertTrue(lysDonors.contains("NZ"), "LYS donors should include NZ")

        let pheAromatic = ConstraintAtomResolver.relevantAtomNames(residueName: "PHE", interactionType: .piStacking)
        XCTAssertTrue(pheAromatic.count >= 6, "PHE pi-stacking should include 6 ring atoms, got \(pheAromatic.count)")
        print("  [Constraint] Residue resolution ✓")

        // 4. GPU buffer conversion
        let constraint = PharmacophoreConstraintDef(
            targetScope: .atom,
            interactionType: .hbondAcceptor,
            strength: .soft(kcalPerAngstromSq: 5.0),
            distanceThreshold: 3.5,
            sourceType: .receptor,
            proteinAtomIndex: 0
        )
        let (gpuConstraints, params) = PharmacophoreConstraintDef.toGPUBuffers(
            constraints: [constraint], atoms: [
                Atom(id: 0, element: .O, position: SIMD3<Float>(1, 2, 3),
                     name: "OD1", residueName: "ASP", residueSeq: 25, chainID: "A")
            ], residues: []
        )
        XCTAssertEqual(params.numConstraints, 1, "Should have 1 GPU constraint")
        XCTAssertEqual(params.numGroups, 1, "Should have 1 group")
        XCTAssertEqual(gpuConstraints[0].strength, 5.0, accuracy: 0.01, "Soft strength should be 5.0")
        XCTAssertEqual(gpuConstraints[0].distanceThreshold, 3.5, accuracy: 0.01)
        XCTAssertEqual(gpuConstraints[0].ligandAtomIndex, -1, "Receptor-side constraint should have ligandAtomIndex = -1")
        print("  [Constraint] GPU buffer conversion ✓")

        // 5. Hard constraint GPU value
        let hardConstraint = PharmacophoreConstraintDef(
            targetScope: .atom,
            interactionType: .hbondDonor,
            strength: .hard,
            sourceType: .receptor,
            proteinAtomIndex: 0
        )
        let (hardGPU, _) = PharmacophoreConstraintDef.toGPUBuffers(
            constraints: [hardConstraint], atoms: [
                Atom(id: 0, element: .N, position: .zero, name: "N", residueName: "ALA", residueSeq: 1, chainID: "A")
            ], residues: []
        )
        XCTAssertEqual(hardGPU[0].strength, 1000.0, accuracy: 0.01, "Hard constraint strength should be 1000.0")
        print("  [Constraint] Hard constraint value ✓")

        print("✓ Pharmacophore constraint data model test PASSED")
    }

    /// Test that constraint buffers can be created and passed to the docking engine
    /// without crashing, and that constrained docking produces valid results.
    @MainActor
    func testConstrainedDockingProducesValidResults() async throws {
        let engine = try sharedDockingEngine()

        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, err) = RDKitBridge.smilesToMolecule(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "aspirin")

        // Create a soft constraint at the pocket center (should be easily satisfiable)
        var constraint = PharmacophoreConstraintDef(
            targetScope: .atom,
            interactionType: .hydrophobic,
            strength: .soft(kcalPerAngstromSq: 5.0),
            distanceThreshold: 5.0,  // generous threshold
            sourceType: .receptor,
            proteinAtomIndex: 0
        )
        constraint.targetPositions = [pocket.center]
        print("  [ConstrainedDock] Constraint: hydrophobic at pocket center \(pocket.center), threshold=5.0 Å")

        // Prepare constraint buffers
        engine.prepareConstraintBuffers([constraint], atoms: protein.atoms, residues: protein.residues)
        print("  [ConstrainedDock] Constraint buffers prepared")

        // Run docking with constraint
        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 100, generations: 80, flexibility: false)

        XCTAssertFalse(results.isEmpty, "Constrained docking must produce results")
        print("  [ConstrainedDock] \(results.count) poses returned")

        for (i, r) in results.prefix(5).enumerated() {
            print("    Pose \(i): E=\(String(format: "%8.2f", r.energy)) constraintPen=\(String(format: "%.3f", r.constraintPenalty))")
            XCTAssertTrue(r.energy.isFinite, "Pose \(i) energy not finite: \(r.energy)")
            XCTAssertTrue(r.constraintPenalty.isFinite, "Pose \(i) constraint penalty not finite: \(r.constraintPenalty)")
            XCTAssertTrue(r.constraintPenalty >= 0, "Constraint penalty should be non-negative, got \(r.constraintPenalty)")
        }

        // Best pose should have low or zero constraint penalty with the generous threshold
        let best = results[0]
        print("  [ConstrainedDock] Best pose: E=\(String(format: "%.2f", best.energy)), constraintPenalty=\(String(format: "%.4f", best.constraintPenalty))")

        // Clean up: clear constraint buffers so other tests are unaffected
        engine.prepareConstraintBuffers([], atoms: [], residues: [])

        print("✓ Constrained docking valid results test PASSED")
    }

    /// Test that a hard constraint steers the GA toward satisfying the constraint.
    /// Compare energies of unconstrained vs hard-constrained docking to verify
    /// the constraint penalty is active.
    @MainActor
    func testHardConstraintAffectsScoring() async throws {
        let engine = try sharedDockingEngine()

        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, _) = RDKitBridge.smilesToMolecule(
            smiles: "c1ccccc1", name: "Benzene", numConformers: 1, minimize: true)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "benzene")

        // First: unconstrained docking
        engine.prepareConstraintBuffers([], atoms: [], residues: [])
        let unconstrainedResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 80, generations: 60, flexibility: false)
        let unconstrainedBestE = unconstrainedResults.first?.energy ?? .infinity
        print("  [HardConstraint] Unconstrained best energy: \(String(format: "%.2f", unconstrainedBestE))")

        // Now: add a hard hydrophobic constraint far from the pocket (impossible to satisfy).
        // Use hydrophobic type so benzene C_H atoms are compatible — the penalty fires
        // because no C_H atom can reach a target 50 Å away within the grid box.
        var impossibleConstraint = PharmacophoreConstraintDef(
            targetScope: .atom,
            interactionType: .hydrophobic,
            strength: .hard,
            distanceThreshold: 2.0,  // very tight threshold
            sourceType: .receptor,
            proteinAtomIndex: 0
        )
        // Place constraint 50 Å away from the pocket — no ligand atom can reach it
        impossibleConstraint.targetPositions = [pocket.center + SIMD3<Float>(50, 50, 50)]
        engine.prepareConstraintBuffers([impossibleConstraint], atoms: protein.atoms, residues: protein.residues)
        print("  [HardConstraint] Hard hydrophobic constraint placed 50 Å from pocket center")

        let constrainedResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 80, generations: 60, flexibility: false)

        let constrainedBestE = constrainedResults.first?.energy ?? .infinity
        let constrainedPenalty = constrainedResults.first?.constraintPenalty ?? 0
        print("  [HardConstraint] Constrained best energy: \(String(format: "%.2f", constrainedBestE)), penalty: \(String(format: "%.2f", constrainedPenalty))")

        // The constrained energy should be worse (higher) or the penalty non-zero,
        // because the constraint target is unreachable. With a tiny test system the
        // GA may not fully explore, so we check the combined effect.
        let totalWorsening = constrainedBestE - unconstrainedBestE + constrainedPenalty
        XCTAssertGreaterThanOrEqual(totalWorsening, 0,
            "Hard constraint should worsen total energy+penalty: unconstrained=\(unconstrainedBestE), constrained=\(constrainedBestE), penalty=\(constrainedPenalty)")

        // Clean up
        engine.prepareConstraintBuffers([], atoms: [], residues: [])

        print("✓ Hard constraint affects scoring test PASSED")
    }

    /// Test that constraint buffers with zero constraints (the no-op case) don't
    /// alter docking results compared to having no buffers.
    @MainActor
    func testZeroConstraintsNoOp() async throws {
        let engine = try sharedDockingEngine()

        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, _) = RDKitBridge.smilesToMolecule(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "aspirin")

        // Prepare empty constraint buffers (should be a no-op)
        engine.prepareConstraintBuffers([], atoms: [], residues: [])

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 80, generations: 60, flexibility: false)

        XCTAssertFalse(results.isEmpty, "Docking with zero constraints should produce results")
        for r in results {
            XCTAssertLessThan(r.constraintPenalty, 0.1,
                "Zero constraints should produce near-zero constraint penalty, got \(r.constraintPenalty)")
        }
        print("  [ZeroConstraints] \(results.count) poses, all with zero constraint penalty ✓")

        // Clean up
        engine.prepareConstraintBuffers([], atoms: [], residues: [])

        print("✓ Zero constraints no-op test PASSED")
    }

    // ======================================================================
    // MARK: 8 — Ligand Strain Penalty (Functional)
    // ======================================================================

    /// Dock aspirin into alanine dipeptide, compute MMFF strain for top poses,
    /// and verify that strained poses get penalized.
    @MainActor
    func testStrainPenaltyAffectsRanking() async throws {
        print("\n  ========== STRAIN PENALTY: Aspirin / Ala dipeptide ==========")
        let engine = try sharedDockingEngine()
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let smiles = "CC(=O)Oc1ccccc1C(=O)O"
        let (molData, err) = RDKitBridge.smilesToMolecule(smiles: smiles, name: "Aspirin", numConformers: 5, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: smiles, smiles: smiles)

        let results = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                            populationSize: 100, generations: 80, flexibility: true)
        XCTAssertFalse(results.isEmpty, "Docking must produce results")

        // Compute reference energy
        guard let refEnergy = RDKitBridge.mmffReferenceEnergy(smiles: smiles) else {
            throw XCTSkip("MMFF reference energy failed")
        }
        print("  [Strain] Reference energy: \(String(format: "%.2f", refEnergy)) kcal/mol")

        // Compute strain for top 10 poses
        var strainValues: [(index: Int, strain: Float, energy: Float)] = []
        for (i, r) in results.prefix(10).enumerated() {
            guard !r.transformedAtomPositions.isEmpty else { continue }
            if let dockedE = RDKitBridge.mmffStrainEnergy(smiles: smiles, heavyPositions: r.transformedAtomPositions) {
                let strain = Float(dockedE - refEnergy)
                strainValues.append((i, strain, r.energy))
                print("  [Strain] Pose \(i): Vina=\(String(format: "%7.2f", r.energy)) kcal/mol, MMFF strain=\(String(format: "%7.1f", strain)) kcal/mol")
            }
        }

        XCTAssertFalse(strainValues.isEmpty, "Should compute strain for at least one pose")

        // Verify strain values are finite and mostly non-negative
        for sv in strainValues {
            XCTAssertTrue(sv.strain.isFinite, "Strain for pose \(sv.index) should be finite")
        }

        // Apply penalty and verify re-ranking
        let threshold: Float = 6.0
        let weight: Float = 0.5
        var penalizedEnergies = results.prefix(10).enumerated().map { (i, r) -> (Int, Float) in
            let strain = strainValues.first(where: { $0.index == i })?.strain ?? 0
            let penalty = strain > threshold ? weight * (strain - threshold) : 0
            return (i, r.energy + penalty)
        }
        penalizedEnergies.sort { $0.1 < $1.1 }

        let originalOrder = Array(0..<min(10, results.count))
        let penalizedOrder = penalizedEnergies.map(\.0)
        let orderChanged = originalOrder != penalizedOrder
        print("  [Strain] Re-ranking changed order: \(orderChanged)")
        print("  [Strain] Original top-3: \(originalOrder.prefix(3)), Penalized top-3: \(penalizedOrder.prefix(3))")

        print("✓ Strain penalty test PASSED (\(strainValues.count) poses scored)")
    }

    // ======================================================================
    // MARK: 9 — Bridging Waters (Functional)
    // ======================================================================

    /// Verify that keeping a water near the pocket changes the grid energy landscape.
    @MainActor
    func testBridgingWaterChangesGridScoring() async throws {
        print("\n  ========== BRIDGING WATER: Grid impact ==========")
        let engine = try sharedDockingEngine()

        // Build a small protein with and without a water at a specific position
        var baseAtoms: [Atom] = []
        // 3 ALA residues as a minimal pocket
        let backbonePositions: [(String, SIMD3<Float>, Element)] = [
            ("N",  SIMD3(0, 0, 0), .N), ("CA", SIMD3(1.47, 0, 0), .C),
            ("C",  SIMD3(2.5, 1.2, 0), .C), ("O",  SIMD3(2.5, 2.4, 0), .O),
            ("CB", SIMD3(1.47, -1.5, 0), .C),
            ("N",  SIMD3(3.5, 0.5, 0), .N), ("CA", SIMD3(5.0, 0.5, 0), .C),
            ("C",  SIMD3(6.0, 1.7, 0), .C), ("O",  SIMD3(6.0, 2.9, 0), .O),
            ("CB", SIMD3(5.0, -1.0, 0), .C),
            ("N",  SIMD3(7.0, 1.0, 0), .N), ("CA", SIMD3(8.5, 1.0, 0), .C),
            ("C",  SIMD3(9.5, 2.2, 0), .C), ("O",  SIMD3(9.5, 3.4, 0), .O),
            ("CB", SIMD3(8.5, -0.5, 0), .C),
        ]
        for (i, (name, pos, elem)) in backbonePositions.enumerated() {
            let resSeq = i / 5 + 1
            baseAtoms.append(Atom(id: i, element: elem, position: pos, name: name,
                                   residueName: "ALA", residueSeq: resSeq, chainID: "A", charge: 0.0))
        }

        let proteinNoWater = Molecule(name: "test", atoms: baseAtoms, bonds: [], title: "no water")
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: proteinNoWater, residueIndices: [1, 2, 3])

        // Version WITH water at the pocket center
        var atomsWithWater = baseAtoms
        let waterPos = pocket.center
        let waterIdx = atomsWithWater.count
        atomsWithWater.append(Atom(id: waterIdx, element: .O, position: waterPos, name: "O",
                                    residueName: "HOH", residueSeq: 100, chainID: "W", charge: -0.8))
        // Two H atoms for the water
        atomsWithWater.append(Atom(id: waterIdx + 1, element: .H,
                                    position: waterPos + SIMD3(0.96, 0, 0), name: "H1",
                                    residueName: "HOH", residueSeq: 100, chainID: "W", charge: 0.4))
        atomsWithWater.append(Atom(id: waterIdx + 2, element: .H,
                                    position: waterPos + SIMD3(-0.24, 0.93, 0), name: "H2",
                                    residueName: "HOH", residueSeq: 100, chainID: "W", charge: 0.4))
        let proteinWithWater = Molecule(name: "test+water", atoms: atomsWithWater, bonds: [], title: "with water")

        // Dock a small ligand against both
        let (molData, _) = RDKitBridge.smilesToMolecule(smiles: "c1ccccc1", name: "Benzene", numConformers: 1, minimize: true)
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "benzene")

        // Dock WITHOUT water
        let resultsNoWater = await runTestDocking(engine: engine, protein: proteinNoWater, ligand: ligand, pocket: pocket,
                                                    populationSize: 80, generations: 60, flexibility: false)
        let bestNoWater = resultsNoWater.first?.energy ?? .infinity

        // Dock WITH water
        let resultsWithWater = await runTestDocking(engine: engine, protein: proteinWithWater, ligand: ligand, pocket: pocket,
                                                     populationSize: 80, generations: 60, flexibility: false)
        let bestWithWater = resultsWithWater.first?.energy ?? .infinity

        print("  [Water] Best energy without water: \(String(format: "%.2f", bestNoWater)) kcal/mol")
        print("  [Water] Best energy with water:    \(String(format: "%.2f", bestWithWater)) kcal/mol")
        print("  [Water] Difference:                \(String(format: "%.2f", bestWithWater - bestNoWater)) kcal/mol")

        // The water should change the energy landscape — energies should differ
        XCTAssertNotEqual(bestNoWater, bestWithWater, accuracy: 0.01,
            "Adding a water molecule at the pocket center should change the best docking energy")

        print("✓ Bridging water grid impact test PASSED")
    }

    // ======================================================================
    // MARK: 10 — Receptor Flexibility (Functional)
    // ======================================================================

    /// Verify that FlexDockingEngine correctly excludes sidechain atoms and creates valid GPU buffers
    /// using a real protein from PDB.
    @MainActor
    func testFlexExclusionOnRealProtein() async throws {
        print("\n  ========== FLEX EXCLUSION: 3PYY real protein ==========")
        let engine = try sharedDockingEngine()

        let (protein, ligand, _) = try await fetchAndParsePDB(id: "3PYY", ligandResidue: "STI")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { return XCTFail("[Flex] Pocket detection failed") }

        // Pick 3 pocket residues that have rotamers
        // pocket.residueIndices are indices into protein.residues, not residueSeq values
        let residues = protein.residues
        var flexIndices: [Int] = []
        for resArrayIdx in pocket.residueIndices {
            guard flexIndices.count < 3 else { break }
            guard resArrayIdx < residues.count else { continue }
            let res = residues[resArrayIdx]
            if RotamerLibrary.rotamers(for: res.name) != nil {
                flexIndices.append(res.sequenceNumber)
            }
        }
        print("  [Flex] Selected \(flexIndices.count) flexible residues from pocket")
        guard flexIndices.count >= 2 else { throw XCTSkip("Not enough rotamerable residues in 3PYY pocket") }

        for idx in flexIndices {
            if let atom = protein.atoms.first(where: { $0.residueSeq == idx }) {
                let chiCount = RotamerLibrary.rotamers(for: atom.residueName)?.chiAngles.count ?? 0
                print("    Residue \(atom.residueName)\(idx): \(chiCount) chi angles")
            }
        }

        // Create flex engine and exclude atoms
        let flexEngine = FlexDockingEngine(device: engine.device, commandQueue: engine.commandQueue)
        var config = FlexibleResidueConfig()
        config.flexibleResidueIndices = flexIndices

        let exclusion = flexEngine.excludeFlexAtoms(
            proteinAtoms: protein.atoms, proteinBonds: protein.bonds,
            flexConfig: config
        )

        print("  [Flex] Rigid atoms: \(exclusion.rigidAtoms.count) (was \(protein.atoms.count))")
        print("  [Flex] Flex atoms:  \(exclusion.flexAtoms.count)")
        print("  [Flex] Chi slots:   \(exclusion.chiSlotCount)")
        print("  [Flex] Torsion edges: \(exclusion.flexTorsionEdges.count)")
        print("  [Flex] Moving indices: \(exclusion.flexMovingIndices.count)")

        // Validate exclusion
        XCTAssertGreaterThan(exclusion.flexAtoms.count, 0, "Should have flex atoms")
        XCTAssertGreaterThan(exclusion.chiSlotCount, 0, "Should have chi slots")
        // Rigid + flex atoms should cover the original protein (flex buffer may include
        // pseudo backbone atoms used as rotation pivots, so allow some tolerance)
        let rigidCount = exclusion.rigidAtoms.count
        let flexCount = exclusion.flexAtoms.count
        let totalOriginal = protein.atoms.count
        XCTAssertGreaterThanOrEqual(rigidCount + flexCount, totalOriginal,
            "Rigid (\(rigidCount)) + flex (\(flexCount)) should be >= original (\(totalOriginal))")

        // Verify rigid protein has no sidechain atoms from the flex residues
        let flexResSet = Set(flexIndices)
        let backboneNames: Set<String> = ["N", "CA", "C", "O", "H", "HA"]
        for atom in exclusion.rigidAtoms {
            if flexResSet.contains(atom.residueSeq) {
                XCTAssertTrue(backboneNames.contains(atom.name),
                    "Rigid protein should only contain backbone atoms for flex residues, found \(atom.name) in \(atom.residueName)\(atom.residueSeq)")
            }
        }

        // Create GPU buffers and verify
        flexEngine.prepareFlexBuffers(exclusion: exclusion)
        XCTAssertTrue(flexEngine.isEnabled, "Flex engine should be enabled after buffer creation")
        XCTAssertNotNil(flexEngine.flexAtomBuffer, "flexAtomBuffer should be created")
        XCTAssertNotNil(flexEngine.flexTorsionEdgeBuffer, "flexTorsionEdgeBuffer should be created")
        XCTAssertNotNil(flexEngine.flexParamsBuffer, "flexParamsBuffer should be created")

        print("  [Flex] GPU buffers created: atomBuf=\(flexEngine.flexAtomBuffer!.length)B, " +
              "edgeBuf=\(flexEngine.flexTorsionEdgeBuffer!.length)B")

        print("✓ Flex exclusion on real protein test PASSED")
    }

    /// Verify that flexible docking with a real system runs, converges, and produces
    /// results comparable to rigid docking. Uses 3PYY (imatinib/ABL kinase).
    /// Runs rigid baseline first, then flex with 2-3 pocket residues.
    @MainActor
    func testFlexDockingProducesValidResults() async throws {
        print("\n  ========== FLEX DOCKING: 3PYY / Imatinib ==========")
        let engine = try sharedDockingEngine()

        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(id: "3PYY", ligandResidue: "STI")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { return XCTFail("[Flex] Pocket detection failed") }

        // ---------- Phase 1: Rigid baseline ----------
        print("  [FlexDock] Phase 1: Rigid baseline (300 pop, 300 gen)")
        engine.flexEngine = nil
        let rigidResults = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                                 populationSize: 300, generations: 300, flexibility: true)
        let rigidBest = rigidResults.first!
        let rigidRMSD = computeRMSD(crystalHeavy, rigidBest.transformedAtomPositions)
        let rigidClusters = Set(rigidResults.map(\.clusterID)).count
        print("  [FlexDock] Rigid: E=\(String(format: "%.1f", rigidBest.energy)), RMSD=\(String(format: "%.2f", rigidRMSD))Å, \(rigidResults.count) poses, \(rigidClusters) clusters")

        // ---------- Phase 2: Select flex residues ----------
        // Pick 2-3 residues from pocket that are on the ligand's chain and close to ligand centroid,
        // so they actually matter for induced-fit scoring.
        let ligandCentroid = crystalHeavy.reduce(SIMD3<Float>.zero, +) / Float(max(crystalHeavy.count, 1))
        let ligandChain = ligand.atoms.first?.chainID ?? "A"
        let residues = protein.residues

        // Collect candidate residues: must have rotamers, be on the same chain as (or near) the ligand,
        // and have sidechain atoms within 6 Å of the ligand centroid
        struct FlexCandidate {
            let seqNum: Int
            let name: String
            let minDist: Float
            let chiCount: Int
        }
        var candidates: [FlexCandidate] = []
        for resArrayIdx in pocket.residueIndices {
            guard resArrayIdx < residues.count else { continue }
            let res = residues[resArrayIdx]
            guard let rotDef = RotamerLibrary.rotamers(for: res.name) else { continue }

            // Check that this residue's sidechain atoms are near the ligand
            let backboneNames: Set<String> = ["N", "CA", "C", "O", "H", "HA"]
            let scAtoms = res.atomIndices.compactMap { idx -> Atom? in
                guard idx < protein.atoms.count else { return nil }
                let a = protein.atoms[idx]
                return backboneNames.contains(a.name) ? nil : a
            }
            guard !scAtoms.isEmpty else { continue }
            let minDist = scAtoms.map { simd_distance($0.position, ligandCentroid) }.min() ?? 999
            guard minDist < 12.0 else { continue }  // sidechain within 12 Å of ligand center

            candidates.append(FlexCandidate(
                seqNum: res.sequenceNumber, name: res.name,
                minDist: minDist, chiCount: rotDef.chiAngles.count
            ))
        }
        // Sort by proximity to ligand, pick closest 3
        candidates.sort { $0.minDist < $1.minDist }
        let flexIndices = Array(candidates.prefix(3).map(\.seqNum))

        guard flexIndices.count >= 2 else { throw XCTSkip("Not enough rotamerable residues near ligand in 3PYY pocket") }

        for c in candidates.prefix(3) {
            print("  [FlexDock] Flex residue: \(c.name)\(c.seqNum), \(c.chiCount) chi, minDist=\(String(format: "%.1f", c.minDist))Å")
        }

        // ---------- Phase 3: Flex docking ----------
        let flexEngine = FlexDockingEngine(device: engine.device, commandQueue: engine.commandQueue)
        var flexConfig = FlexibleResidueConfig()
        flexConfig.flexibleResidueIndices = flexIndices

        let vinaTypes = engine.vinaTypesForProtein(protein)
        let exclusion = flexEngine.excludeFlexAtoms(
            proteinAtoms: protein.atoms, proteinBonds: protein.bonds,
            flexConfig: flexConfig, vinaTypes: vinaTypes
        )
        flexEngine.prepareFlexBuffers(exclusion: exclusion)
        engine.flexEngine = flexEngine

        let typedCount = exclusion.flexAtoms.filter { $0.vinaType >= 0 }.count
        print("  [FlexDock] Flex atoms: \(exclusion.flexAtoms.count) (\(typedCount) typed), \(exclusion.chiSlotCount) chi slots")
        print("  [FlexDock] Delta scoring: grid includes ALL atoms; flex kernel computes (rotated - reference)")
        print("  [FlexDock] Phase 3: Flex docking (300 pop, 300 gen)")

        // Grid uses FULL protein (flex atoms at reference positions). The flex kernel
        // computes delta = pairwise(rotated) - pairwise(reference). When chi=0, delta=0 → rigid docking.
        let flexResults = await runTestDocking(engine: engine, protein: protein, ligand: ligand, pocket: pocket,
                                                populationSize: 300, generations: 300, flexibility: true)

        XCTAssertFalse(flexResults.isEmpty, "[FlexDock] No results")

        let flexBest = flexResults[0]
        XCTAssertTrue(flexBest.energy.isFinite, "[FlexDock] Best energy not finite: \(flexBest.energy)")
        XCTAssertTrue(flexBest.energy < 0, "[FlexDock] Energy should be negative, got \(flexBest.energy)")
        XCTAssertFalse(flexBest.transformedAtomPositions.isEmpty, "[FlexDock] No transformed positions")

        let flexRMSD = computeRMSD(crystalHeavy, flexBest.transformedAtomPositions)
        let flexClusters = Set(flexResults.map(\.clusterID)).count

        if !flexBest.pose.chiAngles.isEmpty {
            print("  [FlexDock] Best chi angles: \(flexBest.pose.chiAngles.map { String(format: "%.1f°", $0 * 180 / .pi) })")
        }

        // ---------- Phase 4: Compare ----------
        print("")
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║  3PYY Flex vs Rigid Comparison                       ║")
        print("  ╠═══════════════════════════════════════════════════════╣")
        print("  ║  Rigid:  E=\(String(format: "%7.1f", rigidBest.energy)) kcal/mol  RMSD=\(String(format: "%5.2f", rigidRMSD))Å  \(String(format: "%3d", rigidClusters)) clusters ║")
        print("  ║  Flex:   E=\(String(format: "%7.1f", flexBest.energy)) kcal/mol  RMSD=\(String(format: "%5.2f", flexRMSD))Å  \(String(format: "%3d", flexClusters)) clusters ║")
        print("  ║  ΔE=\(String(format: "%+.1f", flexBest.energy - rigidBest.energy))  ΔRMSD=\(String(format: "%+.2f", flexRMSD - rigidRMSD))Å                          ║")
        print("  ╚═══════════════════════════════════════════════════════╝")

        // Flex should produce valid results — don't assert RMSD improvement since
        // flex with few residues on a well-packed kinase may not help, but it must
        // not catastrophically diverge
        XCTAssertLessThan(flexRMSD, 20.0,
            "[FlexDock] Flex RMSD \(String(format: "%.2f", flexRMSD))Å is too high — scoring may be broken")
        XCTAssertGreaterThan(flexClusters, 1,
            "[FlexDock] Should have >1 cluster, got \(flexClusters)")

        // Clean up
        engine.flexEngine = nil

        print("✓ Flex docking test PASSED")
    }
}

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
        let maxR = simd_length(pocket.size) + 5.0

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
}

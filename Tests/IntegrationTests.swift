import XCTest
import simd
import MetalKit
@testable import Druse

/// Slow integration tests — requires GPU + network + PDB downloads.
/// Gated behind DRUSE_INTEGRATION_TESTS=1 environment variable.
/// Validates the full pipeline against real protein-ligand complexes from the PDB.
/// Expected runtime: ~120s with all tests enabled.
final class IntegrationTests: DockingTestCase {

    // MARK: - Round-Trip: PDB Crystal → SMILES Regen → Dock → Compare to Crystal

    /// 1HSG: HIV-1 protease + indinavir (MK1).
    /// Tests basic round-trip pipeline with a well-studied inhibitor.
    @MainActor
    func testRoundTrip1HSGIndinavir() async throws {
        try await runRoundTripTest(
            pdbID: "1HSG", ligandResidue: "MK1",
            smiles: "CC(C)(C)NC(=O)C1CN(CCc2ccccc2)CC1O",
            ligandName: "Indinavir")
    }

    /// 2QWK: neuraminidase + oseltamivir (G39).
    /// Tests handling of 3 stereocenters — correct stereochemistry is essential
    /// for proper binding. Without @/@@ SMILES, RDKit generates wrong enantiomer.
    @MainActor
    func testRoundTrip2QWKOseltamivir() async throws {
        try await runRoundTripTest(
            pdbID: "2QWK", ligandResidue: "G39",
            smiles: "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1",
            ligandName: "Oseltamivir", centroidThreshold: 12.0)
    }

    /// 4DFR: dihydrofolate reductase + methotrexate (MTX).
    /// Large ligand with multiple aromatic rings and carboxylic acid groups.
    @MainActor
    func testRoundTrip4DFRMethotrexate() async throws {
        try await runRoundTripTest(
            pdbID: "4DFR", ligandResidue: "MTX",
            smiles: "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
            ligandName: "Methotrexate", centroidThreshold: 12.0)
    }

    // MARK: - Drusina vs Vina on Salt Bridge Complex

    /// 3ERT: estrogen receptor α + 4-hydroxytamoxifen (OHT).
    /// The ligand's dimethylamine (protonated at pH 7.4) forms a salt bridge
    /// with Asp351. Drusina should capture this interaction vs classical Vina.
    @MainActor
    func testDrusinaVsVinaOn3ERT() async throws {
        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(
            id: "3ERT", ligandResidue: "OHT")
        guard crystalHeavy.count >= 5 else {
            throw XCTSkip("3ERT crystal ligand too small")
        }

        guard let pocket = BindingSiteDetector.ligandGuidedPocket(
            protein: protein, ligand: ligand, distance: 6.0) else {
            throw XCTSkip("3ERT pocket detection failed")
        }
        printPocketDebug(pocket, label: "3ERT")

        let engine = try makeDockingEngine()

        // Vina
        let vinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 150, generations: 150, scoringMethod: .vina)
        // Drusina
        let drusinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 150, generations: 150, scoringMethod: .drusina)

        XCTAssertFalse(vinaResults.isEmpty, "Vina must produce results for 3ERT")
        XCTAssertFalse(drusinaResults.isEmpty, "Drusina must produce results for 3ERT")

        let vinaE = vinaResults[0].energy
        let drusinaE = drusinaResults[0].energy
        XCTAssertTrue(vinaE.isFinite && vinaE < 0,
            "Vina energy should be negative, got \(vinaE)")
        XCTAssertTrue(drusinaE.isFinite && drusinaE < 0,
            "Drusina energy should be negative, got \(drusinaE)")

        // RMSD should be finite
        if let best = vinaResults.first, !best.transformedAtomPositions.isEmpty {
            XCTAssertTrue(computeRMSD(crystalHeavy, best.transformedAtomPositions).isFinite)
        }
        if let best = drusinaResults.first, !best.transformedAtomPositions.isEmpty {
            XCTAssertTrue(computeRMSD(crystalHeavy, best.transformedAtomPositions).isFinite)
        }

        print("  [3ERT] Vina: \(String(format: "%.2f", vinaE)) kcal/mol, " +
              "Drusina: \(String(format: "%.2f", drusinaE)) kcal/mol, " +
              "correction: \(String(format: "%.3f", drusinaResults[0].drusinaCorrection))")
    }

    // MARK: - ML Scoring Pipeline

    /// 3PYY: ABL kinase + imatinib (STI).
    /// Dock imatinib from SMILES, then score with DruseScorePKi CoreML model.
    /// Validates pKd range, confidence, and Ki conversion.
    @MainActor
    func testMLScoringOn3PYY() async throws {
        let (protein, crystalLigand, _) = try await fetchAndParsePDB(
            id: "3PYY", ligandResidue: "STI")

        let imatinibSMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
        let (molData, err) = RDKitBridge.smilesToMolecule(
            smiles: imatinibSMILES, name: "Imatinib", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(
            name: md.name, atoms: md.atoms, bonds: md.bonds, title: "imatinib")

        guard let pocket = BindingSiteDetector.ligandGuidedPocket(
            protein: protein, ligand: crystalLigand, distance: 6.0) else {
            throw XCTSkip("3PYY pocket detection failed")
        }

        let engine = try makeDockingEngine()
        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 200, generations: 150, flexibility: false)
        XCTAssertFalse(results.isEmpty, "Docking must produce results")

        // Score with DruseMLScoring
        let scorer = DruseMLScoringInference()
        scorer.loadModel()
        guard scorer.isAvailable else {
            throw XCTSkip("DruseScorePKi model not available in test bundle")
        }

        let pocketAtoms = protein.atoms.filter {
            simd_distance($0.position, pocket.center) <= 10.0
        }

        var scoredCount = 0
        for i in 0..<min(results.count, 10) {
            var poseAtoms = ligand.atoms
            for j in 0..<poseAtoms.count {
                if j < results[i].transformedAtomPositions.count {
                    poseAtoms[j].position = results[i].transformedAtomPositions[j]
                }
            }

            let features = DruseScoreFeatureExtractor.extract(
                proteinAtoms: pocketAtoms, ligandAtoms: poseAtoms,
                pocketCenter: pocket.center)

            if let pred = await scorer.score(features: features) {
                XCTAssertTrue(pred.pKd >= 0 && pred.pKd <= 15,
                    "pKd should be in [0,15], got \(pred.pKd)")
                XCTAssertTrue(pred.poseConfidence >= 0 && pred.poseConfidence <= 1,
                    "Confidence should be in [0,1], got \(pred.poseConfidence)")
                XCTAssertTrue(pred.dockingScore.isFinite,
                    "dockingScore must be finite")
                XCTAssertEqual(pred.dockingScore, pred.pKd * pred.poseConfidence,
                    accuracy: 0.1, "dockingScore ≈ pKd × confidence")

                let ki = AffinityDisplayUnit.pKdToKi(pKd: pred.pKd)
                XCTAssertTrue(ki > 0, "Ki must be positive")
                let expectedKi = pow(10, -Double(pred.pKd))
                XCTAssertEqual(ki, expectedKi, accuracy: expectedKi * 0.001)

                scoredCount += 1
            }
        }
        XCTAssertGreaterThan(scoredCount, 0, "At least one pose should be scored")
        print("  [MLScoring] Scored \(scoredCount) poses with DruseScorePKi")
    }
}

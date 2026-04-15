// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

}

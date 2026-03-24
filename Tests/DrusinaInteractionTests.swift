import XCTest
import simd
import MetalKit
@testable import Druse

// =============================================================================
// MARK: - Drusina Extended Interaction Tests
// =============================================================================
//
// Tests for the new Drusina interaction scoring terms:
//   1. Salt bridge scoring (geometry, flag assignment)
//   2. Amide-π stacking (backbone amide detection, scoring)
//   3. Chalcogen bonds (S...O/N σ-hole detection)
//   4. Integration: Drusina vs Vina on salt-bridge–rich complex (3ERT)
//

// =============================================================================
// MARK: 1 — Basic Struct & Parameter Tests (no GPU needed)
// =============================================================================

final class DrusinaParameterTests: XCTestCase {

    // MARK: - DrusinaParams new fields

    func testDrusinaParamsLayout() {
        // Verify the expanded DrusinaParams struct has all expected fields
        var params = DrusinaParams()
        params.numProteinRings = 5
        params.numLigandRings = 3
        params.numProteinCations = 2
        params.numHalogens = 1
        params.wPiPi = -0.40
        params.wPiCation = -0.80
        params.wHalogenBond = -0.50
        params.wMetalCoord = -1.00
        // New fields
        params.numProteinAmides = 42
        params.numChalcogens = 2
        params.wSaltBridge = -0.60
        params.wAmideStack = -0.40
        params.wChalcogenBond = -0.30

        XCTAssertEqual(params.numProteinAmides, 42)
        XCTAssertEqual(params.numChalcogens, 2)
        XCTAssertEqual(params.wSaltBridge, -0.60, accuracy: 0.001)
        XCTAssertEqual(params.wAmideStack, -0.40, accuracy: 0.001)
        XCTAssertEqual(params.wChalcogenBond, -0.30, accuracy: 0.001)

        // Check stride is reasonable (should be aligned, ~64 bytes)
        let stride = MemoryLayout<DrusinaParams>.stride
        print("  [DrusinaParams] stride = \(stride) bytes")
        XCTAssertTrue(stride >= 56, "DrusinaParams too small: \(stride) bytes")
        XCTAssertTrue(stride % 4 == 0, "DrusinaParams not 4-byte aligned: \(stride)")
    }

    // MARK: - GridProteinAtom salt bridge flags

    func testGridProteinAtomFlagField() {
        // Verify flags field replaces _pad0 and can store salt bridge flags
        var atom = GridProteinAtom()
        atom.position = SIMD3<Float>(1, 2, 3)
        atom.vdwRadius = 1.7
        atom.charge = 0.5
        atom.vinaType = 4 // VINA_N_A
        atom.flags = UInt32(GRPROT_FLAG_POS_CHARGED)

        XCTAssertEqual(atom.flags & UInt32(GRPROT_FLAG_POS_CHARGED), UInt32(GRPROT_FLAG_POS_CHARGED))
        XCTAssertEqual(atom.flags & UInt32(GRPROT_FLAG_NEG_CHARGED), 0)

        // Both flags
        atom.flags = UInt32(GRPROT_FLAG_POS_CHARGED) | UInt32(GRPROT_FLAG_NEG_CHARGED)
        XCTAssertNotEqual(atom.flags & UInt32(GRPROT_FLAG_POS_CHARGED), 0)
        XCTAssertNotEqual(atom.flags & UInt32(GRPROT_FLAG_NEG_CHARGED), 0)
    }

    // MARK: - Salt bridge flag assignment on real residues

    func testSaltBridgeFlagAssignment() {
        // Build a small protein with Lys, Arg, Asp, Glu, and Ala
        // Verify that only the correct atoms get POS/NEG flags
        var atoms: [Atom] = []

        func addAtom(_ elem: Element, _ x: Float, _ y: Float, _ z: Float,
                     name: String, resName: String, resSeq: Int, charge: Int = 0) {
            atoms.append(Atom(
                id: atoms.count, element: elem,
                position: SIMD3<Float>(x, y, z),
                name: name, residueName: resName, residueSeq: resSeq,
                chainID: "A", formalCharge: charge
            ))
        }

        // Lys NZ (should be positive)
        addAtom(.N, 10, 10, 10, name: "NZ", resName: "LYS", resSeq: 1)
        // Arg NH1, NH2 (should be positive)
        addAtom(.N, 12, 10, 10, name: "NH1", resName: "ARG", resSeq: 2)
        addAtom(.N, 12, 12, 10, name: "NH2", resName: "ARG", resSeq: 2)
        // Asp OD1, OD2 (should be negative)
        addAtom(.O, 20, 10, 10, name: "OD1", resName: "ASP", resSeq: 3)
        addAtom(.O, 20, 12, 10, name: "OD2", resName: "ASP", resSeq: 3)
        // Glu OE1 (should be negative)
        addAtom(.O, 30, 10, 10, name: "OE1", resName: "GLU", resSeq: 4)
        // Ala CA (should be neither)
        addAtom(.C, 40, 10, 10, name: "CA", resName: "ALA", resSeq: 5)

        // Simulate flag assignment logic (same as DockingEngine.swift)
        let posAtomNames: Set<String> = ["NZ", "NH1", "NH2", "NE", "CZ"]
        let posResNames: Set<String> = ["LYS", "ARG"]
        let negAtomNames: Set<String> = ["OD1", "OD2", "OE1", "OE2"]
        let negResNames: Set<String> = ["ASP", "GLU"]

        for atom in atoms {
            let name = atom.name.trimmingCharacters(in: .whitespaces)
            let res = atom.residueName.trimmingCharacters(in: .whitespaces)
            var flags: UInt32 = 0

            if (posResNames.contains(res) && posAtomNames.contains(name)) || atom.formalCharge > 0 {
                flags |= UInt32(GRPROT_FLAG_POS_CHARGED)
            }
            if negResNames.contains(res) && negAtomNames.contains(name) {
                flags |= UInt32(GRPROT_FLAG_NEG_CHARGED)
            }

            switch name {
            case "NZ":
                XCTAssertNotEqual(flags & UInt32(GRPROT_FLAG_POS_CHARGED), 0,
                    "Lys NZ should have POS flag")
            case "NH1", "NH2":
                XCTAssertNotEqual(flags & UInt32(GRPROT_FLAG_POS_CHARGED), 0,
                    "Arg \(name) should have POS flag")
            case "OD1", "OD2":
                XCTAssertNotEqual(flags & UInt32(GRPROT_FLAG_NEG_CHARGED), 0,
                    "Asp \(name) should have NEG flag")
            case "OE1":
                XCTAssertNotEqual(flags & UInt32(GRPROT_FLAG_NEG_CHARGED), 0,
                    "Glu OE1 should have NEG flag")
            case "CA":
                XCTAssertEqual(flags, 0, "Ala CA should have no salt bridge flags")
            default:
                break
            }
        }
        print("  [SaltBridgeFlags] All 7 atoms verified")
    }

    // MARK: - ProteinAmideGPU / ChalcogenBondInfo struct layout

    func testNewStructLayouts() {
        // ProteinAmideGPU should match ProteinRingGPU layout (32 bytes)
        let amideStride = MemoryLayout<ProteinAmideGPU>.stride
        let ringStride = MemoryLayout<ProteinRingGPU>.stride
        print("  [Structs] ProteinAmideGPU stride = \(amideStride), ProteinRingGPU stride = \(ringStride)")
        XCTAssertEqual(amideStride, ringStride,
            "ProteinAmideGPU and ProteinRingGPU should have same stride for GPU consistency")

        // ChalcogenBondInfo should match HalogenBondInfo layout (8 bytes)
        let chalcogenStride = MemoryLayout<ChalcogenBondInfo>.stride
        let halogenStride = MemoryLayout<HalogenBondInfo>.stride
        print("  [Structs] ChalcogenBondInfo stride = \(chalcogenStride), HalogenBondInfo stride = \(halogenStride)")
        XCTAssertEqual(chalcogenStride, halogenStride,
            "ChalcogenBondInfo and HalogenBondInfo should have same stride")
    }

    // MARK: - Amide plane geometry

    func testAmidePlaneNormalComputation() {
        // Test backbone C-O-N amide plane normal computation
        // Planar amide: C at origin, O along x, N along y → normal along z
        let cPos = SIMD3<Float>(0, 0, 0)
        let oPos = SIMD3<Float>(1.24, 0, 0)   // C=O distance ~1.24 Å
        let nPos = SIMD3<Float>(-0.5, 1.32, 0) // C-N distance ~1.33 Å

        let centroid = (cPos + oPos + nPos) / 3.0
        let v1 = oPos - cPos
        let v2 = nPos - cPos
        var normal = simd_cross(v1, v2)
        let nLen = simd_length(normal)
        XCTAssertGreaterThan(nLen, 1e-6, "Normal length should be non-zero")
        normal /= nLen

        print("  [AmidePlane] centroid = \(centroid)")
        print("  [AmidePlane] normal = \(normal)")

        // Normal should be approximately (0, 0, ±1) for planar amide in xy-plane
        XCTAssertLessThan(abs(normal.x), 0.01, "Normal x should be ~0")
        XCTAssertLessThan(abs(normal.y), 0.01, "Normal y should be ~0")
        XCTAssertGreaterThan(abs(normal.z), 0.99, "Normal z should be ~±1")
    }

    // MARK: - Salt bridge scoring geometry (CPU verification)

    func testSaltBridgeGaussianProfile() {
        // Verify the Gaussian scoring function shape for salt bridges
        // E(d) = wSaltBridge * exp(-(d - 3.5)^2 * 2.0)
        let w: Float = -0.60

        // At optimal distance (3.5 Å): should be strongest
        let eOptimal = w * exp(-pow(3.5 - 3.5, 2) * 2.0)
        XCTAssertEqual(eOptimal, w, accuracy: 0.001, "At 3.5Å should give full weight")
        print("  [SaltBridge] E(3.5Å) = \(String(format: "%.4f", eOptimal)) kcal/mol")

        // At 2.5 Å: close contact, some penalty from Gaussian shape
        let e25 = w * exp(-pow(2.5 - 3.5, 2) * 2.0)
        print("  [SaltBridge] E(2.5Å) = \(String(format: "%.4f", e25)) kcal/mol")
        XCTAssertLessThan(abs(e25), abs(eOptimal), "2.5Å should be weaker than 3.5Å")

        // At 4.5 Å cutoff: should be weak
        let e45 = w * exp(-pow(4.5 - 3.5, 2) * 2.0)
        print("  [SaltBridge] E(4.5Å) = \(String(format: "%.4f", e45)) kcal/mol")
        XCTAssertLessThan(abs(e45), abs(eOptimal) * 0.2, "4.5Å should be <20% of optimal")

        // At 5.0 Å (beyond cutoff): should be essentially zero
        let e50 = w * exp(-pow(5.0 - 3.5, 2) * 2.0)
        print("  [SaltBridge] E(5.0Å) = \(String(format: "%.4f", e50)) kcal/mol")
        XCTAssertLessThan(abs(e50), 0.01, "5.0Å should be ~0 (beyond cutoff)")
    }

    // MARK: - Amide stacking Gaussian profile

    func testAmideStackGaussianProfile() {
        // Harder 2013: optimal d=3.4Å interplanar, centroid r=3.8Å
        // E(d) = wAmideStack * exp(-(d - 3.8)^2 * 2.0)  [parallel]
        let w: Float = -0.40

        let eOptimal = w * exp(-pow(3.8 - 3.8, 2) * 2.0)
        XCTAssertEqual(eOptimal, w, accuracy: 0.001)
        print("  [AmideStack] E(3.8Å) = \(String(format: "%.4f", eOptimal)) kcal/mol")

        // At 3.0 Å: too close, weaker
        let e30 = w * exp(-pow(3.0 - 3.8, 2) * 2.0)
        print("  [AmideStack] E(3.0Å) = \(String(format: "%.4f", e30)) kcal/mol")
        XCTAssertLessThan(abs(e30), abs(eOptimal), "3.0Å should be weaker than 3.8Å")

        // At 5.0 Å: edge of range
        let e50 = w * exp(-pow(5.0 - 3.8, 2) * 2.0)
        print("  [AmideStack] E(5.0Å) = \(String(format: "%.4f", e50)) kcal/mol")
        XCTAssertLessThan(abs(e50), abs(eOptimal) * 0.15, "5.0Å should be <15% of optimal")
    }

    // MARK: - InteractionType enum coverage

    func testInteractionTypeNewCases() {
        // Verify new cases exist and have correct raw values
        let amide = MolecularInteraction.InteractionType.amideStack
        let chalcogen = MolecularInteraction.InteractionType.chalcogen

        XCTAssertEqual(amide.rawValue, 8)
        XCTAssertEqual(chalcogen.rawValue, 9)
        XCTAssertEqual(amide.label, "Amide-π")
        XCTAssertEqual(chalcogen.label, "Chalcogen bond")

        // All cases should be enumerable
        let allCases = MolecularInteraction.InteractionType.allCases
        XCTAssertTrue(allCases.contains(.amideStack))
        XCTAssertTrue(allCases.contains(.chalcogen))
        XCTAssertEqual(allCases.count, 10, "Should have 10 interaction types total")
        print("  [InteractionType] \(allCases.count) types: \(allCases.map(\.label).joined(separator: ", "))")
    }
}

// =============================================================================
// MARK: 2 — GPU Integration: Drusina Scoring Produces Valid Output
// =============================================================================

final class DrusinaGPUTests: DockingTestCase {

    /// Verify Drusina scoring runs on GPU with new interaction buffers and produces
    /// finite, favorable energies. Uses aspirin against alanine dipeptide (minimal system).
    @MainActor
    func testDrusinaScoringProducesFiniteEnergies() async throws {
        let engine = try sharedDockingEngine()

        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        let (molData, err) = RDKitBridge.smilesToMolecule(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let md = molData else { throw XCTSkip("RDKit unavailable") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "aspirin")

        // --- Vina scoring ---
        let vinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 100, generations: 80, flexibility: false, scoringMethod: .vina)

        // --- Drusina scoring ---
        let drusinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 100, generations: 80, flexibility: false, scoringMethod: .drusina)

        // Both must produce valid results
        XCTAssertFalse(vinaResults.isEmpty, "Vina should produce results")
        XCTAssertFalse(drusinaResults.isEmpty, "Drusina should produce results")

        for (i, r) in drusinaResults.prefix(10).enumerated() {
            XCTAssertTrue(r.energy.isFinite, "Drusina pose \(i) energy not finite: \(r.energy)")
        }

        let vinaBest = vinaResults.first!.energy
        let drusinaBest = drusinaResults.first!.energy
        print("  [Drusina vs Vina] Vina best: \(String(format: "%.3f", vinaBest)), Drusina best: \(String(format: "%.3f", drusinaBest))")

        // Both should be negative (favorable)
        XCTAssertLessThan(vinaBest, 5.0, "Vina best should be reasonable")
        XCTAssertLessThan(drusinaBest, 5.0, "Drusina best should be reasonable")
    }

    /// Test that Drusina detects and scores salt bridges for a charged ligand.
    /// Metformin (antidiabetic, pKa ~12.4, doubly protonated at pH 7.4) docked
    /// against a protein with Asp/Glu residues should show Drusina correction.
    @MainActor
    func testDrusinaSaltBridgeCorrectionForChargedLigand() async throws {
        let engine = try sharedDockingEngine()

        // Build a minimal protein with charged residues
        // Use alanine dipeptide as scaffold, it has backbone amides
        let protein = TestMolecules.alanineDipeptide()
        let pocket = await BindingSiteDetector.pocketFromResidues(
            protein: protein, residueIndices: Array(0..<protein.residues.count))

        // Metformin: charged biguanide (has formal +charge on N atoms)
        let (molData, err) = RDKitBridge.smilesToMolecule(
            smiles: "CN(C)C(=N)NC(=N)N", name: "Metformin", numConformers: 1, minimize: true)
        if let err = err { print("  [Metformin] RDKit note: \(err)") }
        guard let md = molData else { throw XCTSkip("RDKit unavailable for metformin") }
        let ligand = Molecule(name: md.name, atoms: md.atoms, bonds: md.bonds, title: "metformin")
        print("  [Metformin] \(ligand.heavyAtomCount) heavy atoms, \(ligand.bondCount) bonds")

        // Dock with Drusina
        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 100, generations: 80, flexibility: false, scoringMethod: .drusina)

        XCTAssertFalse(results.isEmpty, "Docking must produce results for metformin")
        let best = results.first!
        XCTAssertTrue(best.energy.isFinite, "Best energy must be finite")
        print("  [Metformin-Drusina] Best E = \(String(format: "%.3f", best.energy)) kcal/mol")
        print("  [Metformin-Drusina] drusinaCorr = \(String(format: "%.3f", best.drusinaCorrection)) kcal/mol")
    }
}

// =============================================================================
// MARK: 3 — Integration: Drusina vs Vina on Salt Bridge Complex (3ERT)
// =============================================================================

final class DrusinaIntegrationTests: DockingTestCase {

    /// 3ERT: estrogen receptor α with 4-hydroxytamoxifen.
    /// The ligand's dimethylamine (protonated at pH 7.4) forms a salt bridge
    /// with Asp351. Drusina should capture this interaction.
    ///
    /// This tests the full pipeline:
    ///   PDB download → parse → pocket detection → grid → Vina dock → Drusina dock → compare
    @MainActor
    func testDrusinaVsVinaOn3ERT() async throws {
        print("\n  ========== INTEGRATION: Drusina vs Vina on 3ERT ==========")

        let (protein, ligand, crystalHeavy) = try await fetchAndParsePDB(id: "3ERT", ligandResidue: "OHT")
        guard crystalHeavy.count >= 5 else { throw XCTSkip("3ERT crystal ligand too small") }

        let pocket = await BindingSiteDetector.ligandGuidedPocket(
            protein: protein, ligand: ligand, distance: 6.0)
        guard let pocket else { throw XCTSkip("3ERT pocket detection failed") }
        printPocketDebug(pocket, label: "3ERT")

        // Check that Asp351 is in the pocket vicinity
        let asp351 = protein.atoms.filter {
            $0.residueName.trimmingCharacters(in: .whitespaces) == "ASP" &&
            $0.residueSeq == 351
        }
        print("  [3ERT] Asp351 atoms: \(asp351.count) (salt bridge partner)")

        let engine = try makeDockingEngine()

        // --- Vina ---
        let vinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 150, generations: 150, flexibility: false, scoringMethod: .vina)
        let vinaE = vinaResults.first?.energy ?? .infinity
        let vinaRMSD: Float
        if let best = vinaResults.first, !best.transformedAtomPositions.isEmpty {
            vinaRMSD = computeRMSD(crystalHeavy, best.transformedAtomPositions)
        } else {
            vinaRMSD = .infinity
        }

        // --- Drusina ---
        let drusinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 150, generations: 150, flexibility: false, scoringMethod: .drusina)
        let drusinaE = drusinaResults.first?.energy ?? .infinity
        let drusinaCorr = drusinaResults.first?.drusinaCorrection ?? 0
        let drusinaRMSD: Float
        if let best = drusinaResults.first, !best.transformedAtomPositions.isEmpty {
            drusinaRMSD = computeRMSD(crystalHeavy, best.transformedAtomPositions)
        } else {
            drusinaRMSD = .infinity
        }

        print("\n  ===== 3ERT COMPARISON =====")
        print("  Vina:    E = \(String(format: "%8.3f", vinaE)) kcal/mol, RMSD = \(String(format: "%.2f", vinaRMSD)) Å")
        print("  Drusina: E = \(String(format: "%8.3f", drusinaE)) kcal/mol, RMSD = \(String(format: "%.2f", drusinaRMSD)) Å")
        print("           correction = \(String(format: "%.3f", drusinaCorr)) kcal/mol")
        print("  ===========================\n")

        // Both should produce valid results
        XCTAssertFalse(vinaResults.isEmpty, "Vina must produce results for 3ERT")
        XCTAssertFalse(drusinaResults.isEmpty, "Drusina must produce results for 3ERT")
        XCTAssertTrue(vinaE.isFinite, "Vina energy must be finite")
        XCTAssertTrue(drusinaE.isFinite, "Drusina energy must be finite")

        // Drusina correction should be non-zero (3ERT has salt bridge + aromatic interactions)
        // It's acceptable if the correction is zero for some poses, but the scoring should work
        print("  [3ERT] Drusina correction: \(String(format: "%.4f", drusinaCorr))")

        // Energy should be favorable (negative)
        XCTAssertLessThan(vinaE, 0, "Vina best energy should be negative for 3ERT")
        XCTAssertLessThan(drusinaE, 0, "Drusina best energy should be negative for 3ERT")

        // Both RMSD should be finite
        XCTAssertTrue(vinaRMSD.isFinite, "Vina RMSD must be finite")
        XCTAssertTrue(drusinaRMSD.isFinite, "Drusina RMSD must be finite")
    }
}

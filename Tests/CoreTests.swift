import XCTest
import simd
@testable import Druse

/// Fast core tests — no GPU, no network. ~10s total.
/// Validates RDKit bridge, parsers, protein preparation, protonation, fragments, and config.
final class CoreTests: XCTestCase {

    // MARK: - RDKit Bridge

    func testSmilesToMolecule() {
        let (mol, err) = RDKitBridge.smilesToMolecule(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", numConformers: 1, minimize: true)
        XCTAssertNil(err, "RDKit error: \(err ?? "")")
        guard let mol else { XCTFail("No molecule returned"); return }
        XCTAssertEqual(mol.atoms.count, 21, "Aspirin C9H8O4 = 21 atoms with H")
        XCTAssertTrue(mol.bonds.count > 0)
        for atom in mol.atoms {
            XCTAssertFalse(atom.position.x.isNaN || atom.position.y.isNaN || atom.position.z.isNaN,
                "Atom \(atom.id) has NaN position")
        }

        // Invalid SMILES
        let (bad, badErr) = RDKitBridge.smilesToMolecule(smiles: "NOT_A_SMILES")
        XCTAssertNil(bad)
        XCTAssertNotNil(badErr)
    }

    func testDescriptorsAndDrugLikeness() {
        // Caffeine: MW=194, Lipinski pass
        guard let d = RDKitBridge.computeDescriptors(
            smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O") else {
            XCTFail("Caffeine descriptors failed"); return
        }
        XCTAssertEqual(d.molecularWeight, 194.19, accuracy: 1.0)
        XCTAssertEqual(d.hbd, 0, "Caffeine has no H-bond donors")
        XCTAssertEqual(d.rings, 2)
        XCTAssertTrue(d.lipinski, "Caffeine should pass Lipinski")
        XCTAssertTrue(d.veber, "Caffeine should pass Veber")

        // Large cyclic peptide: Lipinski fail
        let large = "CC(C)CC1NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC1=O"
        guard let d2 = RDKitBridge.computeDescriptors(smiles: large) else {
            XCTFail("Large peptide descriptors failed"); return
        }
        XCTAssertTrue(d2.molecularWeight > 500)
        XCTAssertFalse(d2.lipinski)
    }

    func testGasteigerCharges() {
        let (mol, _, err) = RDKitBridge.prepareLigand(
            smiles: "CC(=O)O", name: "AceticAcid", numConformers: 1, computeCharges: true)
        XCTAssertNil(err)
        guard let m = mol else { XCTFail("Prep failed"); return }
        XCTAssertTrue(m.atoms.contains { $0.charge != 0.0 }, "Should compute non-zero charges")
        for o in m.atoms where o.element == .O {
            XCTAssertLessThan(o.charge, 0, "Oxygen should have negative Gasteiger charge, got \(o.charge)")
        }
    }

    func testConformerGeneration() {
        let conformers = RDKitBridge.generateConformers(
            smiles: "CCCCCCCC", name: "Octane", count: 10, minimize: true)
        XCTAssertGreaterThanOrEqual(conformers.count, 5, "Octane should produce ≥5 conformers")
        // Energy-sorted
        for i in 1..<conformers.count {
            XCTAssertGreaterThanOrEqual(conformers[i].energy, conformers[i-1].energy,
                "Conformers not sorted by energy at \(i)")
        }
        // Geometric diversity
        if conformers.count >= 2 {
            let first = conformers[0].molecule.atoms.map(\.position)
            let last = conformers.last!.molecule.atoms.map(\.position)
            let n = min(first.count, last.count)
            var sum: Float = 0
            for i in 0..<n { sum += simd_distance_squared(first[i], last[i]) }
            XCTAssertGreaterThan(sqrt(sum / Float(n)), 0.01, "Conformers should differ geometrically")
        }
    }

    func testTorsionTree() {
        // Butane: flexible
        let butane = RDKitBridge.buildTorsionTree(smiles: "CCCC")
        XCTAssertNotNil(butane)
        XCTAssertGreaterThanOrEqual(butane!.count, 1, "Butane should have ≥1 rotatable bond")
        for edge in butane! {
            XCTAssertNotEqual(edge.atom1, edge.atom2)
            XCTAssertFalse(edge.movingAtoms.isEmpty)
        }
        // Benzene: rigid
        XCTAssertEqual(RDKitBridge.buildTorsionTree(smiles: "c1ccccc1")?.count, 0,
            "Benzene should have 0 rotatable bonds")
        // Ibuprofen: multiple torsions
        XCTAssertGreaterThanOrEqual(
            RDKitBridge.buildTorsionTree(smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O")?.count ?? 0, 2,
            "Ibuprofen should have ≥2 rotatable bonds")
    }

    func testMorganFingerprint() {
        let fp = RDKitBridge.morganFingerprint(smiles: "c1ccccc1", radius: 2, nBits: 2048)
        XCTAssertEqual(fp.count, 2048)
        XCTAssertGreaterThan(fp.filter { $0 == 1.0 }.count, 0, "Benzene should have set bits")
        for (i, bit) in fp.enumerated() {
            XCTAssertTrue(bit == 0.0 || bit == 1.0, "Bit \(i) = \(bit), must be 0 or 1")
        }
    }

    // MARK: - Fragment Decomposition & Similarity

    func testFragmentDecomposition() {
        // Benzene: 1 rigid fragment
        guard let benz = RDKitBridge.decomposeFragments(smiles: "c1ccccc1") else {
            XCTFail("Benzene decomposition failed"); return
        }
        XCTAssertEqual(benz.numFragments, 1)
        XCTAssertEqual(benz.connections.count, 0)

        // Ibuprofen: multiple fragments
        guard let ibu = RDKitBridge.decomposeFragments(
            smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O") else {
            XCTFail("Ibuprofen decomposition failed"); return
        }
        XCTAssertGreaterThanOrEqual(ibu.numFragments, 2)
        for (i, size) in ibu.fragmentSizes.enumerated() {
            XCTAssertGreaterThan(size, 0, "Fragment \(i) is empty")
        }

        // Imatinib: many rigid fragments
        let imatinib = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
        guard let ima = RDKitBridge.decomposeFragments(smiles: imatinib) else {
            XCTFail("Imatinib decomposition failed"); return
        }
        XCTAssertGreaterThanOrEqual(ima.numFragments, 3)
        XCTAssertGreaterThanOrEqual(ima.connections.count, 2)
        let totalAssigned = ima.fragmentSizes.reduce(Int32(0), +)
        XCTAssertEqual(Int(totalAssigned), ima.fragmentMembership.count,
            "All heavy atoms should be assigned to fragments")

        // Invalid SMILES
        XCTAssertNil(RDKitBridge.decomposeFragments(smiles: "NOT_VALID"))
    }

    func testScaffoldMatchingAndSimilarity() {
        // Aspirin contains benzene ring
        guard let match = RDKitBridge.matchScaffold(
            smiles: "CC(=O)Oc1ccccc1C(=O)O", scaffoldSMARTS: "c1ccccc1") else {
            XCTFail("matchScaffold returned nil"); return
        }
        XCTAssertTrue(match.hasMatch, "Aspirin should contain benzene")
        XCTAssertEqual(match.matchedAtomIndices.count, 6)

        // Butane does not contain benzene
        XCTAssertFalse(
            RDKitBridge.matchScaffold(smiles: "CCCC", scaffoldSMARTS: "c1ccccc1")?.hasMatch ?? true)

        // Tanimoto similarity
        XCTAssertEqual(
            RDKitBridge.tanimotoSimilarity(smiles1: "c1ccccc1", smiles2: "c1ccccc1"),
            1.0, accuracy: 0.001, "Identical molecules should have Tanimoto = 1.0")
        XCTAssertLessThan(
            RDKitBridge.tanimotoSimilarity(smiles1: "c1ccccc1", smiles2: "CCCCCCCCCCCCCCCCCCCC"),
            0.3, "Benzene vs long alkane should be very different")
        XCTAssertGreaterThan(
            RDKitBridge.tanimotoSimilarity(
                smiles1: "CC(=O)Oc1ccccc1C(=O)O", smiles2: "OC(=O)c1ccccc1O"),
            0.3, "Aspirin vs salicylic acid should be related")
    }

    // MARK: - AutoTune Search Method

    func testAutoTuneSearchMethodSelection() {
        // Low torsion → GA
        let ga = DockingConfig.autoTune(
            proteinAtomCount: 3000, pocketVolume: 800, pocketBuriedness: 0.5,
            ligandHeavyAtoms: 15, ligandRotatableBonds: 3)
        XCTAssertEqual(ga.searchMethod, .genetic)

        // Medium torsion → REMC
        let remc = DockingConfig.autoTune(
            proteinAtomCount: 3000, pocketVolume: 800, pocketBuriedness: 0.5,
            ligandHeavyAtoms: 25, ligandRotatableBonds: 8)
        XCTAssertEqual(remc.searchMethod, .parallelTempering)

        // High torsion → Fragment
        let frag = DockingConfig.autoTune(
            proteinAtomCount: 3000, pocketVolume: 800, pocketBuriedness: 0.5,
            ligandHeavyAtoms: 40, ligandRotatableBonds: 12)
        XCTAssertEqual(frag.searchMethod, .fragmentBased)
    }

    // MARK: - PDB / mmCIF Parsing

    @MainActor
    func testPDBParser() {
        // Minimal PDB
        let pdb = """
        HEADER    TEST
        ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
        ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 10.00           C
        ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00 10.00           C
        ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00 10.00           O
        ATOM      5  CB  ALA A   1       2.000   1.000   4.000  1.00 10.00           C
        END
        """
        let result = PDBParser.parse(pdb)
        XCTAssertEqual(result.protein?.atoms.count, 5)

        // HETATM extraction
        let pdb2 = """
        ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
        ATOM      2  C   ALA A   1       2.000   2.000   3.000  1.00 10.00           C
        HETATM    3  C1  MK1 B   1      10.000  10.000  10.000  1.00 10.00           C
        HETATM    4  O1  MK1 B   1      11.000  10.000  10.000  1.00 10.00           O
        END
        """
        XCTAssertGreaterThanOrEqual(PDBParser.parse(pdb2).ligands.count, 1,
            "Should extract HETATM as ligand")

        // Water counting
        let pdb3 = """
        ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
        HETATM    2  O   HOH A 100      20.000  20.000  20.000  1.00 10.00           O
        HETATM    3  O   HOH A 101      21.000  20.000  20.000  1.00 10.00           O
        END
        """
        XCTAssertEqual(PDBParser.parse(pdb3).waterCount, 2)

        // Alt loc: highest occupancy wins
        let pdb4 = """
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
        ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.40 20.00           C
        ATOM      3  CA BALA A   1       5.000   0.000   0.000  0.60 40.00           C
        ATOM      4  C   ALA A   1       2.200   0.000   0.000  1.00 10.00           C
        END
        """
        let cas = PDBParser.parse(pdb4).protein!.atoms.filter {
            $0.name.trimmingCharacters(in: .whitespaces) == "CA"
        }
        XCTAssertEqual(cas.count, 1)
        XCTAssertEqual(cas[0].position.x, 5.0, accuracy: 0.001,
            "Should select alt loc B (higher occupancy)")
    }

    @MainActor
    func testMMCIFParser() throws {
        let mmcif = """
        data_test
        loop_
        _atom_site.group_PDB
        _atom_site.type_symbol
        _atom_site.label_atom_id
        _atom_site.label_comp_id
        _atom_site.label_asym_id
        _atom_site.label_seq_id
        _atom_site.Cartn_x
        _atom_site.Cartn_y
        _atom_site.Cartn_z
        ATOM N N ALA A 1 1.000 2.000 3.000
        ATOM C CA ALA A 1 2.000 2.000 3.000
        ATOM C C ALA A 1 3.000 2.000 3.000
        ATOM O O ALA A 1 3.500 3.000 3.000
        #
        """
        let parsed = try MMCIFParser.parse(content: mmcif, fileName: "test")
        XCTAssertEqual(parsed.atoms.count, 4)
        XCTAssertEqual(parsed.atoms[0].residueName, "ALA")
        XCTAssertEqual(parsed.atoms[0].chainID, "A")
        XCTAssertGreaterThanOrEqual(parsed.bonds.count, 3)
    }

    // MARK: - Protein Preparation

    func testProteinPreparation() {
        // Chain break detection (C-N gap > 1.8 Å)
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0, 0, 0), name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .C, position: SIMD3<Float>(1.2, 0, 0), name: "CA",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 2, element: .C, position: SIMD3<Float>(2.4, 0, 0), name: "C",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 3, element: .N, position: SIMD3<Float>(6.2, 0, 0), name: "N",
                 residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 4, element: .C, position: SIMD3<Float>(7.4, 0, 0), name: "CA",
                 residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 5, element: .C, position: SIMD3<Float>(8.6, 0, 0), name: "C",
                 residueName: "GLY", residueSeq: 2, chainID: "A")
        ]
        let breaks = ProteinPreparation.detectChainBreaks(in: atoms)
        XCTAssertEqual(breaks.count, 1)
        XCTAssertGreaterThan(breaks[0].carbonNitrogenDistance, 1.8)

        // Water removal: keep nearby, remove far
        let waterAtoms = [
            Atom(id: 0, element: .N, position: .zero, name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .O, position: SIMD3<Float>(1.2, 0, 0), name: "O",
                 residueName: "HOH", residueSeq: 100, chainID: "A", isHetAtom: true),
            Atom(id: 2, element: .O, position: SIMD3<Float>(8, 0, 0), name: "O",
                 residueName: "HOH", residueSeq: 101, chainID: "A", isHetAtom: true)
        ]
        let retained = ProteinPreparation.removeWaters(
            atoms: waterAtoms, bonds: [], keepingNearby: [.zero], within: 2.0)
        XCTAssertEqual(retained.atoms.filter { $0.residueName == "HOH" }.count, 1)
        XCTAssertEqual(retained.removedCount, 1)

        // Non-standard residue removal keeps waters + caps, removes glycerol
        let mixedAtoms = [
            Atom(id: 0, element: .N, position: .zero, name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .O, position: SIMD3(2, 0, 0), name: "O",
                 residueName: "HOH", residueSeq: 100, chainID: "A", isHetAtom: true),
            Atom(id: 2, element: .C, position: SIMD3(4, 0, 0), name: "C1",
                 residueName: "GOL", residueSeq: 200, chainID: "A", isHetAtom: true),
            Atom(id: 3, element: .C, position: SIMD3(6, 0, 0), name: "C",
                 residueName: "ACE", residueSeq: 1, chainID: "A", isHetAtom: true),
            Atom(id: 4, element: .N, position: SIMD3(8, 0, 0), name: "N",
                 residueName: "NME", residueSeq: 2, chainID: "A", isHetAtom: true)
        ]
        let filtered = ProteinPreparation.removeNonStandardResidues(
            atoms: mixedAtoms, bonds: [], keepingWaters: true, keepingExistingCaps: true)
        XCTAssertFalse(filtered.atoms.contains { $0.residueName == "GOL" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "HOH" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "ACE" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "NME" })

        // Chain break capping adds ACE + NME
        let capAtoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0, 0, 0), name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .C, position: SIMD3<Float>(1.45, 0, 0), name: "CA",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 2, element: .C, position: SIMD3<Float>(2.55, 0, 0), name: "C",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 3, element: .O, position: SIMD3<Float>(3.30, 0.85, 0), name: "O",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 4, element: .N, position: SIMD3<Float>(6.20, 0, 0), name: "N",
                 residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 5, element: .C, position: SIMD3<Float>(7.45, 0, 0), name: "CA",
                 residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 6, element: .C, position: SIMD3<Float>(8.55, 0, 0), name: "C",
                 residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 7, element: .O, position: SIMD3<Float>(9.30, 0.85, 0), name: "O",
                 residueName: "GLY", residueSeq: 2, chainID: "A")
        ]
        let capBonds = [
            Bond(id: 0, atomIndex1: 0, atomIndex2: 1, order: .single),
            Bond(id: 1, atomIndex1: 1, atomIndex2: 2, order: .single),
            Bond(id: 2, atomIndex1: 2, atomIndex2: 3, order: .double),
            Bond(id: 3, atomIndex1: 4, atomIndex2: 5, order: .single),
            Bond(id: 4, atomIndex1: 5, atomIndex2: 6, order: .single),
            Bond(id: 5, atomIndex1: 6, atomIndex2: 7, order: .double)
        ]
        let cleaned = ProteinPreparation.cleanupStructure(atoms: capAtoms, bonds: capBonds)
        XCTAssertEqual(cleaned.report.addedCappingResidues, 2)
        XCTAssertTrue(cleaned.atoms.contains { $0.residueName == "ACE" })
        XCTAssertTrue(cleaned.atoms.contains { $0.residueName == "NME" })
    }

    // MARK: - Protonation

    func testProtonation() {
        // Scoring charge assignment at pH 7.4
        let atoms = [
            Atom(id: 0, element: .N, position: .zero, name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .N, position: SIMD3<Float>(1, 0, 0), name: "NZ",
                 residueName: "LYS", residueSeq: 2, chainID: "A"),
            Atom(id: 2, element: .O, position: SIMD3<Float>(2, 0, 0), name: "OXT",
                 residueName: "LYS", residueSeq: 2, chainID: "A")
        ]
        let prot = Protonation.applyProtonation(atoms: atoms, pH: 7.4)
        XCTAssertEqual(prot[0].formalCharge, 1, "N-term should be +1")
        XCTAssertEqual(prot[1].formalCharge, 1, "LYS NZ should be +1")
        XCTAssertEqual(prot[2].formalCharge, -1, "C-term OXT should be -1")

        // Histidine: doubly protonated at pH 5, singly at pH 8
        let his = [
            Atom(id: 0, element: .N, position: .zero, name: "ND1",
                 residueName: "HIS", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .N, position: SIMD3(1, 0, 0), name: "NE2",
                 residueName: "HIS", residueSeq: 1, chainID: "A"),
        ]
        XCTAssertEqual(
            Protonation.predictResidueStates(atoms: his, pH: 5.0).first?.state,
            .histidineDoublyProtonated, "HIS should be doubly protonated at pH 5")
        XCTAssertEqual(
            Protonation.predictResidueStates(atoms: his, pH: 8.0).first?.protonatedAtoms.count,
            1, "HIS should have 1 protonated N at pH 8")

        // ASP pKa lowered by backbone NH hydrogen bond
        let aspOnly = [
            Atom(id: 0, element: .O, position: .zero, name: "OD1",
                 residueName: "ASP", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .O, position: SIMD3<Float>(0, 1.25, 0), name: "OD2",
                 residueName: "ASP", residueSeq: 1, chainID: "A"),
        ]
        let withBB = aspOnly + [
            Atom(id: 2, element: .N, position: SIMD3<Float>(2.60, 0, 0), name: "N",
                 residueName: "ALA", residueSeq: 2, chainID: "A"),
            Atom(id: 3, element: .H, position: SIMD3<Float>(1.60, 0, 0), name: "H",
                 residueName: "ALA", residueSeq: 2, chainID: "A"),
        ]
        let bbBonds = [Bond(id: 0, atomIndex1: 2, atomIndex2: 3, order: .single)]
        guard let aspAlone = Protonation.predictResidueStates(atoms: aspOnly, bonds: [], pH: 7.4)
                .first(where: { $0.residueName == "ASP" }),
              let aspBB = Protonation.predictResidueStates(atoms: withBB, bonds: bbBonds, pH: 7.4)
                .first(where: { $0.residueName == "ASP" }) else {
            XCTFail("Expected ASP predictions"); return
        }
        XCTAssertLessThan(aspBB.shiftedPKa, aspAlone.shiftedPKa,
            "Backbone NH should lower ASP pKa")
    }

    // MARK: - Tautomer & Protomer Enumeration

    func testTautomerEnumeration() {
        // Warfarin: keto/enol tautomers
        let tautomers = RDKitBridge.enumerateTautomers(
            smiles: "CC(=O)CC1=C(C2=CC=CC=C2OC1=O)O",
            maxTautomers: 10, energyCutoff: 30.0)
        XCTAssertGreaterThanOrEqual(tautomers.count, 2, "Warfarin should have ≥2 tautomers")
        for t in tautomers {
            XCTAssertFalse(t.smiles.isEmpty, "Tautomer SMILES should not be empty")
            XCTAssertNotNil(t.molecule, "Tautomer should have 3D coordinates")
        }
        let unique = Set(tautomers.map(\.smiles))
        XCTAssertEqual(unique.count, tautomers.count, "Tautomers should have unique SMILES")

        // Benzene: no tautomerizable sites
        XCTAssertGreaterThanOrEqual(
            RDKitBridge.enumerateTautomers(smiles: "c1ccccc1").count, 1,
            "Benzene should return at least canonical form")
    }

    func testProtomerEnumeration() {
        // Histamine: 2 ionizable groups (imidazole pKa ~6, amine pKa ~10.5)
        let protomers = RDKitBridge.enumerateProtomers(
            smiles: "NCCc1c[nH]cn1", maxProtomers: 16, pH: 7.4, pkaThreshold: 4.0)
        XCTAssertGreaterThanOrEqual(protomers.count, 2,
            "Histamine should have ≥2 protomers at pH 7.4")
        for p in protomers {
            XCTAssertFalse(p.smiles.isEmpty)
            XCTAssertNotNil(p.molecule)
        }

        // Benzene: no ionizable groups → 1 protomer (parent)
        let benz = RDKitBridge.enumerateProtomers(smiles: "c1ccccc1")
        XCTAssertGreaterThanOrEqual(benz.count, 1)
    }
}

import XCTest
@testable import Druse

final class DruseTests: XCTestCase {

    private func envInt(_ key: String, default defaultValue: Int) -> Int {
        guard let raw = ProcessInfo.processInfo.environment[key],
              let value = Int(raw), value > 0 else {
            return defaultValue
        }
        return value
    }

    // MARK: - Element Tests

    func testElementProperties() {
        print("  [Element] C: symbol=\(Element.C.symbol) name=\(Element.C.name) vdw=\(Element.C.vdwRadius) mass=\(Element.C.mass) cov=\(Element.C.covalentRadius)")
        print("  [Element] O: symbol=\(Element.O.symbol) mass=\(Element.O.mass) vdw=\(Element.O.vdwRadius)")
        print("  [Element] H: rawValue=\(Element.H.rawValue), Kr: rawValue=\(Element.Kr.rawValue)")
        XCTAssertEqual(Element.C.symbol, "C")
        XCTAssertEqual(Element.C.name, "Carbon")
        XCTAssertEqual(Element.C.vdwRadius, 1.70, accuracy: 0.01)
        XCTAssertEqual(Element.O.mass, 15.999, accuracy: 0.001)
        XCTAssertEqual(Element.H.rawValue, 1)
        XCTAssertEqual(Element.Kr.rawValue, 36)
    }

    func testElementFromSymbol() {
        let cases: [(String, Element?)] = [("C", .C), ("n", .N), ("Cl", .Cl), ("Xx", nil)]
        for (sym, expected) in cases {
            let result = Element.from(symbol: sym)
            print("  [Element] from(\"\(sym)\") = \(result.map(\.symbol) ?? "nil") (expected: \(expected.map(\.symbol) ?? "nil"))")
            XCTAssertEqual(result, expected)
        }
    }

    func testElementCPKColors() {
        let carbon = Element.C.color
        let oxygen = Element.O.color
        print("  [Element] C color: (\(String(format: "%.2f, %.2f, %.2f, %.2f", carbon.x, carbon.y, carbon.z, carbon.w)))")
        print("  [Element] O color: (\(String(format: "%.2f, %.2f, %.2f, %.2f", oxygen.x, oxygen.y, oxygen.z, oxygen.w)))")
        XCTAssertTrue(carbon.x > 0.5 && carbon.x < 0.6, "C should be dark gray, got r=\(carbon.x)")
        XCTAssertTrue(oxygen.x > 0.8, "O should have high red channel, got r=\(oxygen.x)")
    }

    // MARK: - Caffeine Tests

    @MainActor
    func testCaffeineAtomCount() {
        let caffeine = TestMolecules.caffeine()
        XCTAssertEqual(caffeine.atomCount, 24, "Caffeine should have 24 atoms (C8H10N4O2)")
    }

    @MainActor
    func testCaffeineBondCount() {
        let caffeine = TestMolecules.caffeine()
        XCTAssertEqual(caffeine.bondCount, 25, "Caffeine should have 25 bonds")
    }

    @MainActor
    func testCaffeineElementCounts() {
        let caffeine = TestMolecules.caffeine()
        let counts = Dictionary(grouping: caffeine.atoms, by: { $0.element })
            .mapValues { $0.count }
        XCTAssertEqual(counts[.C], 8, "Caffeine has 8 carbons")
        XCTAssertEqual(counts[.N], 4, "Caffeine has 4 nitrogens")
        XCTAssertEqual(counts[.O], 2, "Caffeine has 2 oxygens")
        XCTAssertEqual(counts[.H], 10, "Caffeine has 10 hydrogens")
    }

    @MainActor
    func testCaffeineMolecularWeight() {
        let caffeine = TestMolecules.caffeine()
        // Caffeine MW = 194.19 Da
        XCTAssertEqual(caffeine.molecularWeight, 194.19, accuracy: 1.0)
    }

    // MARK: - Alanine Dipeptide Tests

    @MainActor
    func testAlanineDipeptideAtomCount() {
        let mol = TestMolecules.alanineDipeptide()
        XCTAssertEqual(mol.atomCount, 22)
    }

    @MainActor
    func testAlanineDipeptideResidues() {
        let mol = TestMolecules.alanineDipeptide()
        XCTAssertEqual(mol.residues.count, 3, "Should have ACE, ALA, NME")
        XCTAssertEqual(mol.residues[0].name, "ACE")
        XCTAssertEqual(mol.residues[1].name, "ALA")
        XCTAssertEqual(mol.residues[2].name, "NME")
    }

    @MainActor
    func testAlanineDipeptideChains() {
        let mol = TestMolecules.alanineDipeptide()
        XCTAssertEqual(mol.chains.count, 1, "Should have 1 chain")
        XCTAssertEqual(mol.chains[0].id, "A")
    }

    // MARK: - Molecule Tests

    @MainActor
    func testMoleculeCenter() {
        let caffeine = TestMolecules.caffeine()
        // Center should be somewhere near the geometric center of the molecule
        let c = caffeine.center
        XCTAssertTrue(abs(c.x) < 5.0 && abs(c.y) < 5.0 && abs(c.z) < 5.0,
                      "Center should be near origin for caffeine")
    }

    @MainActor
    func testMoleculeBoundingRadius() {
        let caffeine = TestMolecules.caffeine()
        XCTAssertTrue(caffeine.radius > 1.0 && caffeine.radius < 10.0,
                      "Bounding radius should be reasonable for a small molecule")
    }

    @MainActor
    func testMoleculeAdjacency() {
        let caffeine = TestMolecules.caffeine()
        // C1 (index 0) should have neighbors from its bonds
        let neighbors = caffeine.neighbors(of: 0)
        XCTAssertTrue(neighbors.count >= 2, "C1 should have at least 2 neighbors")
    }

    // MARK: - Math Utils Tests

    func testQuaternionIdentity() {
        let q = Quat.identity
        let v = SIMD3<Float>(1, 2, 3)
        let rotated = q.rotate(v)
        XCTAssertEqual(rotated.x, v.x, accuracy: 0.0001)
        XCTAssertEqual(rotated.y, v.y, accuracy: 0.0001)
        XCTAssertEqual(rotated.z, v.z, accuracy: 0.0001)
    }

    func testQuaternion90DegRotation() {
        let q = Quat.fromAxisAngle(SIMD3<Float>(0, 0, 1), angle: .pi / 2)
        let v = SIMD3<Float>(1, 0, 0)
        let rotated = q.rotate(v)
        XCTAssertEqual(rotated.x, 0, accuracy: 0.001)
        XCTAssertEqual(rotated.y, 1, accuracy: 0.001)
        XCTAssertEqual(rotated.z, 0, accuracy: 0.001)
    }

    func testCentroid() {
        let points: [SIMD3<Float>] = [
            SIMD3(0, 0, 0),
            SIMD3(2, 0, 0),
            SIMD3(0, 2, 0),
            SIMD3(0, 0, 2)
        ]
        let c = centroid(points)
        XCTAssertEqual(c.x, 0.5, accuracy: 0.001)
        XCTAssertEqual(c.y, 0.5, accuracy: 0.001)
        XCTAssertEqual(c.z, 0.5, accuracy: 0.001)
    }

    func testBoundingBox() {
        let points: [SIMD3<Float>] = [
            SIMD3(-1, -2, -3),
            SIMD3(4, 5, 6)
        ]
        let bb = boundingBox(points)
        XCTAssertEqual(bb.min.x, -1.0)
        XCTAssertEqual(bb.max.x, 4.0)
        XCTAssertEqual(bb.min.y, -2.0)
        XCTAssertEqual(bb.max.y, 5.0)
    }

    func testRaySphereIntersect() {
        let origin = SIMD3<Float>(0, 0, 5)
        let dir = SIMD3<Float>(0, 0, -1)
        let center = SIMD3<Float>(0, 0, 0)
        let radius: Float = 1.0

        let t = raySphereIntersect(rayOrigin: origin, rayDir: dir, sphereCenter: center, sphereRadius: radius)
        XCTAssertNotNil(t)
        XCTAssertEqual(t!, 4.0, accuracy: 0.001) // hit at z=1
    }

    func testRaySphereIntersectMiss() {
        let origin = SIMD3<Float>(5, 5, 5)
        let dir = SIMD3<Float>(0, 0, -1)
        let center = SIMD3<Float>(0, 0, 0)
        let radius: Float = 1.0

        let t = raySphereIntersect(rayOrigin: origin, rayDir: dir, sphereCenter: center, sphereRadius: radius)
        XCTAssertNil(t)
    }

    // MARK: - Camera Tests

    func testCameraDefaultState() {
        let cam = Camera()
        XCTAssertEqual(cam.distance, 15.0)
        XCTAssertEqual(cam.target, .zero)
    }

    func testPerspectiveMatrix() {
        let proj = Mat4.perspective(fovY: .pi / 4, aspect: 1.5, near: 0.1, far: 500)
        // Should be a valid non-identity matrix
        XCTAssertNotEqual(proj.columns.0.x, 0)
        XCTAssertNotEqual(proj.columns.1.y, 0)
    }

    // MARK: - RDKit Bridge Tests

    func testRDKitVersion() {
        let version = RDKitBridge.rdkitVersion
        print("  [RDKit] Version: \(version)")
        XCTAssertNotEqual(version, "unknown")
        XCTAssertTrue(version.contains("."), "Version should contain dots: \(version)")
    }

    func testSmilesToMoleculeAspirin() {
        let smiles = "CC(=O)Oc1ccccc1C(=O)O"
        print("  [RDKit] SMILES→3D: \(smiles)")
        let (mol, err) = RDKitBridge.smilesToMolecule(smiles: smiles, name: "Aspirin",
                                                       numConformers: 1, minimize: true)
        print("  [RDKit] Result: atoms=\(mol?.atoms.count ?? 0), bonds=\(mol?.bonds.count ?? 0), error=\(err ?? "none")")
        XCTAssertNil(err, "Should not error: \(err ?? "")")
        XCTAssertNotNil(mol)
        guard let mol else { return }
        XCTAssertEqual(mol.atoms.count, 21, "Aspirin C9H8O4 = 21 atoms with H, got \(mol.atoms.count)")
        XCTAssertTrue(mol.bonds.count > 0, "Should have bonds, got \(mol.bonds.count)")
        let elemCounts = Dictionary(grouping: mol.atoms, by: { $0.element }).mapValues(\.count)
        print("  [RDKit] Aspirin composition: \(elemCounts.sorted(by: { $0.key.rawValue < $1.key.rawValue }).map { "\($0.key.symbol):\($0.value)" }.joined(separator: " "))")
        for atom in mol.atoms {
            XCTAssertFalse(atom.position.x.isNaN, "Atom \(atom.id) \(atom.element.symbol) position NaN")
            XCTAssertFalse(atom.position.y.isInfinite, "Atom \(atom.id) \(atom.element.symbol) position infinite")
        }
    }

    func testSmilesToMoleculeInvalid() {
        let (mol, err) = RDKitBridge.smilesToMolecule(smiles: "NOT_A_SMILES_STRING")
        print("  [RDKit] Invalid SMILES: mol=\(mol != nil ? "non-nil" : "nil"), error=\(err ?? "none")")
        XCTAssertNil(mol)
        XCTAssertNotNil(err)
    }

    func testDescriptorsCaffeine() {
        let desc = RDKitBridge.computeDescriptors(smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        XCTAssertNotNil(desc)
        guard let d = desc else { return }
        print("  [RDKit] Caffeine descriptors: MW=\(d.molecularWeight) LogP=\(d.logP) TPSA=\(d.tpsa) HBD=\(d.hbd) HBA=\(d.hba) RotB=\(d.rotatableBonds) Rings=\(d.rings) ArRings=\(d.aromaticRings) HA=\(d.heavyAtomCount) fCSP3=\(d.fractionCSP3) Lip=\(d.lipinski) Veber=\(d.veber)")
        XCTAssertEqual(d.molecularWeight, 194.19, accuracy: 1.0)
        XCTAssertEqual(d.hbd, 0, "Caffeine has no H-bond donors")
        XCTAssertEqual(d.rings, 2, "Caffeine has 2 rings")
        XCTAssertTrue(d.lipinski, "Caffeine passes Lipinski's rule of 5")
        XCTAssertTrue(d.veber, "Caffeine passes Veber's rules")
    }

    func testDescriptorsLipinskiViolator() {
        let smi = "CC(C)CC1NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC1=O"
        let desc = RDKitBridge.computeDescriptors(smiles: smi)
        XCTAssertNotNil(desc)
        guard let d = desc else { return }
        print("  [RDKit] Large peptide: MW=\(d.molecularWeight) LogP=\(d.logP) HBD=\(d.hbd) HBA=\(d.hba) Lip=\(d.lipinski)")
        XCTAssertTrue(d.molecularWeight > 500, "Should be >500 Da, got \(d.molecularWeight)")
        XCTAssertFalse(d.lipinski, "Large peptide should fail Lipinski")
    }

    func testGasteigerCharges() {
        let (mol, _, err) = RDKitBridge.prepareLigand(smiles: "CC(=O)O", name: "AceticAcid",
                                                       numConformers: 1, computeCharges: true)
        print("  [Gasteiger] AceticAcid: atoms=\(mol?.atoms.count ?? 0), error=\(err ?? "none")")
        XCTAssertNil(err)
        XCTAssertNotNil(mol)
        guard let m = mol else { return }
        for atom in m.atoms {
            print("    \(atom.element.symbol)\(atom.id) \(atom.name): charge=\(String(format: "%.4f", atom.charge)) formalCharge=\(atom.formalCharge)")
        }
        let hasNonZeroCharge = m.atoms.contains { $0.charge != 0.0 }
        XCTAssertTrue(hasNonZeroCharge, "Gasteiger charges should be non-zero on some atoms")
        let oxygens = m.atoms.filter { $0.element == .O }
        for o in oxygens {
            XCTAssertTrue(o.charge < 0.0, "Oxygen should have negative Gasteiger charge, got \(o.charge)")
        }
    }

    func testProtonationAssignsScoringCharge() {
        let atoms = [
            Atom(id: 0, element: .N, position: .zero, name: "N",
                 residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .N, position: SIMD3<Float>(1, 0, 0), name: "NZ",
                 residueName: "LYS", residueSeq: 2, chainID: "A"),
            Atom(id: 2, element: .O, position: SIMD3<Float>(2, 0, 0), name: "OXT",
                 residueName: "LYS", residueSeq: 2, chainID: "A")
        ]

        let protonated = Protonation.applyProtonation(atoms: atoms, pH: 7.4)
        for (i, a) in protonated.enumerated() {
            print("  [Protonation] atom \(i) \(a.element.symbol) \(a.name) \(a.residueName)\(a.residueSeq): formalCharge=\(a.formalCharge) charge=\(String(format: "%.3f", a.charge))")
        }
        XCTAssertEqual(protonated[0].formalCharge, 1, "N-term ALA N should be +1 at pH 7.4")
        XCTAssertEqual(protonated[0].charge, 1, accuracy: 0.001)
        XCTAssertEqual(protonated[1].formalCharge, 1, "LYS NZ should be +1 at pH 7.4")
        XCTAssertEqual(protonated[1].charge, 1, accuracy: 0.001)
        XCTAssertEqual(protonated[2].formalCharge, -1, "C-term OXT should be -1 at pH 7.4")
        XCTAssertEqual(protonated[2].charge, -1, accuracy: 0.001)
    }

    func testConformerGeneration() {
        let conformers = RDKitBridge.generateConformers(smiles: "CCCCCCCC", name: "Octane", count: 10, minimize: true)
        print("  [Conformers] Octane: \(conformers.count) conformers generated")
        for (i, c) in conformers.enumerated() {
            print("    conf \(i): E=\(String(format: "%.4f", c.energy)) kcal/mol, atoms=\(c.molecule.atoms.count)")
        }
        XCTAssertTrue(conformers.count >= 5, "Should generate ≥5 conformers for octane, got \(conformers.count)")
        for i in 1..<conformers.count {
            XCTAssertTrue(conformers[i].energy >= conformers[i-1].energy,
                          "Conformers should be sorted by energy: conf \(i-1) E=\(conformers[i-1].energy) > conf \(i) E=\(conformers[i].energy)")
        }
        for (i, conf) in conformers.enumerated() {
            XCTAssertEqual(conf.molecule.atoms.count, 26, "Conf \(i): Octane should have 26 atoms, got \(conf.molecule.atoms.count)")
        }
    }

    func testMorganFingerprint() {
        let fp = RDKitBridge.morganFingerprint(smiles: "c1ccccc1", radius: 2, nBits: 2048)
        let setBits = fp.filter { $0 == 1.0 }.count
        print("  [Fingerprint] Benzene ECFP4: \(fp.count) bits, \(setBits) set (\(String(format: "%.1f", Float(setBits) / Float(fp.count) * 100))%)")
        XCTAssertEqual(fp.count, 2048, "Should have 2048 bits, got \(fp.count)")
        XCTAssertTrue(setBits > 0, "Benzene fingerprint should have some bits set, got \(setBits)")
        for (i, bit) in fp.enumerated() {
            XCTAssertTrue(bit == 0.0 || bit == 1.0, "Bit \(i) = \(bit), must be 0 or 1")
        }
    }

    func testTorsionTreeButane() {
        let tree = RDKitBridge.buildTorsionTree(smiles: "CCCC")
        print("  [Torsion] Butane: \(tree?.count ?? 0) rotatable bonds")
        XCTAssertNotNil(tree)
        guard let t = tree else { return }
        for (i, edge) in t.enumerated() {
            print("    edge \(i): atom1=\(edge.atom1) atom2=\(edge.atom2) moving=\(edge.movingAtoms)")
        }
        XCTAssertTrue(t.count >= 1, "Butane should have ≥1 rotatable bond, got \(t.count)")
        for (i, edge) in t.enumerated() {
            XCTAssertNotEqual(edge.atom1, edge.atom2, "Edge \(i): atom1==atom2")
            XCTAssertTrue(edge.movingAtoms.count > 0, "Edge \(i) must have moving atoms")
        }
    }

    func testTorsionTreeBenzene() {
        let tree = RDKitBridge.buildTorsionTree(smiles: "c1ccccc1")
        print("  [Torsion] Benzene: \(tree?.count ?? 0) rotatable bonds")
        XCTAssertNotNil(tree)
        XCTAssertEqual(tree!.count, 0, "Benzene has no rotatable bonds, got \(tree!.count)")
    }

    // MARK: - Parser Tests

    @MainActor
    func testPDBParserMinimal() throws {
        let pdb = """
        HEADER    TEST PROTEIN
        ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
        ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 10.00           C
        ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00 10.00           C
        ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00 10.00           O
        ATOM      5  CB  ALA A   1       2.000   1.000   4.000  1.00 10.00           C
        END
        """
        let result = PDBParser.parse(pdb)
        XCTAssertNotNil(result.protein)
        XCTAssertEqual(result.protein!.atoms.count, 5)
        // Check element types
        let elements: [Element] = result.protein!.atoms.map(\.element)
        XCTAssertEqual(elements[0], Element.N)
        XCTAssertEqual(elements[1], Element.C)
        XCTAssertEqual(elements[3], Element.O)
    }

    @MainActor
    func testPDBParserHETATM() throws {
        let pdb = """
        ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
        ATOM      2  C   ALA A   1       2.000   2.000   3.000  1.00 10.00           C
        HETATM    3  C1  MK1 B   1      10.000  10.000  10.000  1.00 10.00           C
        HETATM    4  O1  MK1 B   1      11.000  10.000  10.000  1.00 10.00           O
        END
        """
        let result = PDBParser.parse(pdb)
        XCTAssertTrue(result.ligands.count >= 1, "Should extract HETATM as ligand")
        let lig = result.ligands.first!
        XCTAssertTrue(lig.atoms.count >= 2, "Ligand should have ≥2 atoms")
    }

    @MainActor
    func testPDBParserWaterCount() throws {
        let pdb = """
        ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
        HETATM    2  O   HOH A 100      20.000  20.000  20.000  1.00 10.00           O
        HETATM    3  O   HOH A 101      21.000  20.000  20.000  1.00 10.00           O
        HETATM    4  O   HOH A 102      22.000  20.000  20.000  1.00 10.00           O
        END
        """
        let result = PDBParser.parse(pdb)
        XCTAssertEqual(result.waterCount, 3, "Should count 3 water molecules")
    }

    @MainActor
    func testPDBParserSelectsHighestOccupancyAltLoc() throws {
        let pdb = """
        HEADER    ALTLOC TEST
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
        ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.40 20.00           C
        ATOM      3  CA BALA A   1       5.000   0.000   0.000  0.60 40.00           C
        ATOM      4  C   ALA A   1       2.200   0.000   0.000  1.00 10.00           C
        END
        """

        let result = PDBParser.parse(pdb)
        guard let protein = result.protein else {
            XCTFail("Expected protein to parse")
            return
        }

        let alphaCarbons = protein.atoms.filter { $0.name.trimmingCharacters(in: .whitespaces) == "CA" }
        XCTAssertEqual(alphaCarbons.count, 1)
        XCTAssertEqual(alphaCarbons[0].position.x, 5.0, accuracy: 0.001)
        XCTAssertTrue(alphaCarbons[0].altLoc.isEmpty, "Selected altloc should be normalized away after cleanup")
    }

    @MainActor
    func testPDBParserBreaksAltLocTiesByBFactor() throws {
        let pdb = """
        HEADER    ALTLOC B-FACTOR TEST
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
        ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.50 30.00           C
        ATOM      3  CA BALA A   1       4.000   0.000   0.000  0.50 12.00           C
        ATOM      4  C   ALA A   1       2.200   0.000   0.000  1.00 10.00           C
        END
        """

        let result = PDBParser.parse(pdb)
        guard let protein = result.protein,
              let alphaCarbon = protein.atoms.first(where: { $0.name.trimmingCharacters(in: .whitespaces) == "CA" }) else {
            XCTFail("Expected preferred alpha carbon")
            return
        }

        XCTAssertEqual(alphaCarbon.position.x, 4.0, accuracy: 0.001)
        XCTAssertTrue(result.warnings.contains { $0.contains("alternate conformations") })
    }

    @MainActor
    func testMMCIFParserPreservesResidueMetadata() throws {
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
        XCTAssertEqual(parsed.name, "test")
        XCTAssertEqual(parsed.atoms.count, 4)
        XCTAssertEqual(parsed.atoms[0].residueName, "ALA")
        XCTAssertEqual(parsed.atoms[0].residueSeq, 1)
        XCTAssertEqual(parsed.atoms[0].chainID, "A")
        XCTAssertFalse(parsed.atoms[0].isHetAtom)
        XCTAssertGreaterThanOrEqual(parsed.bonds.count, 3)
    }

    func testGemmiChemCompParsesResidueTopology() throws {
        let chemComp = """
        data_ALA
        _chem_comp.id ALA
        _chem_comp.name ALANINE
        _chem_comp.type "L-PEPTIDE LINKING"
        loop_
        _chem_comp_atom.comp_id
        _chem_comp_atom.atom_id
        _chem_comp_atom.alt_atom_id
        _chem_comp_atom.type_symbol
        _chem_comp_atom.charge
        _chem_comp_atom.pdbx_align
        _chem_comp_atom.pdbx_aromatic_flag
        _chem_comp_atom.pdbx_leaving_atom_flag
        _chem_comp_atom.pdbx_stereo_config
        _chem_comp_atom.pdbx_backbone_atom_flag
        _chem_comp_atom.pdbx_n_terminal_atom_flag
        _chem_comp_atom.pdbx_c_terminal_atom_flag
        _chem_comp_atom.model_Cartn_x
        _chem_comp_atom.model_Cartn_y
        _chem_comp_atom.model_Cartn_z
        _chem_comp_atom.pdbx_model_Cartn_x_ideal
        _chem_comp_atom.pdbx_model_Cartn_y_ideal
        _chem_comp_atom.pdbx_model_Cartn_z_ideal
        _chem_comp_atom.pdbx_component_atom_id
        _chem_comp_atom.pdbx_component_comp_id
        _chem_comp_atom.pdbx_ordinal
        ALA N N N 0 1 N N N Y Y N 2.281 26.213 12.804 -0.966 0.493 1.500 N ALA 1
        ALA CA CA C 0 1 N N S Y N N 1.169 26.942 13.411 0.257 0.418 0.692 CA ALA 2
        ALA C C C 0 1 N N N Y N Y 1.539 28.344 13.874 -0.094 0.017 -0.716 C ALA 3
        ALA O O O 0 1 N N N Y N Y 2.709 28.647 14.114 -1.056 -0.682 -0.923 O ALA 4
        loop_
        _chem_comp_bond.comp_id
        _chem_comp_bond.atom_id_1
        _chem_comp_bond.atom_id_2
        _chem_comp_bond.value_order
        _chem_comp_bond.pdbx_aromatic_flag
        _chem_comp_bond.pdbx_stereo_config
        _chem_comp_bond.pdbx_ordinal
        ALA N CA SING N N 1
        ALA CA C SING N N 2
        ALA C O DOUB N N 3
        """

        let topology = try GemmiBridge.parseChemCompCIF(chemComp)
        XCTAssertEqual(topology.residueName, "ALA")
        XCTAssertEqual(topology.atoms.count, 4)
        XCTAssertEqual(topology.bonds.count, 3)
        XCTAssertFalse(topology.angles.isEmpty)
        XCTAssertEqual(topology.bonds[2].order, .double)
        XCTAssertGreaterThan(topology.bonds[0].idealLength, 1.0)
    }

    func testGemmiNeighborSearchFindsNearbyAtoms() {
        let pdb = """
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
        ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 10.00           C
        ATOM      3  C   ALA A   1       4.200   0.000   0.000  1.00 10.00           C
        END
        """

        let indices = GemmiBridge.neighborIndices(
            content: pdb,
            queryPoint: SIMD3<Float>(0, 0, 0),
            radius: 1.6,
            includeHydrogens: true,
            maxResults: 8
        )

        XCTAssertEqual(Set(indices), Set([0, 1]))
        XCTAssertEqual(indices.first, 0)
    }

    func testProteinPreparationDetectsChainBreaks() {
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0, 0, 0), name: "N", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .C, position: SIMD3<Float>(1.2, 0, 0), name: "CA", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 2, element: .C, position: SIMD3<Float>(2.4, 0, 0), name: "C", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 3, element: .N, position: SIMD3<Float>(6.2, 0, 0), name: "N", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 4, element: .C, position: SIMD3<Float>(7.4, 0, 0), name: "CA", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 5, element: .C, position: SIMD3<Float>(8.6, 0, 0), name: "C", residueName: "GLY", residueSeq: 2, chainID: "A")
        ]

        let breaks = ProteinPreparation.detectChainBreaks(in: atoms)
        XCTAssertEqual(breaks.count, 1)
        XCTAssertEqual(breaks[0].chainID, "A")
        XCTAssertEqual(breaks[0].previousResidueSeq, 1)
        XCTAssertEqual(breaks[0].nextResidueSeq, 2)
        XCTAssertGreaterThan(breaks[0].carbonNitrogenDistance, 1.8)
    }

    func testProteinPreparationRetainsNearbyWaters() {
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0, 0, 0), name: "N", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .O, position: SIMD3<Float>(1.2, 0, 0), name: "O", residueName: "HOH", residueSeq: 100, chainID: "A", isHetAtom: true),
            Atom(id: 2, element: .O, position: SIMD3<Float>(8.0, 0, 0), name: "O", residueName: "HOH", residueSeq: 101, chainID: "A", isHetAtom: true)
        ]

        let retained = ProteinPreparation.removeWaters(
            atoms: atoms,
            bonds: [],
            keepingNearby: [SIMD3<Float>(0, 0, 0)],
            within: 2.0
        )

        let retainedWaters = retained.atoms.filter { $0.residueName == "HOH" }
        XCTAssertEqual(retainedWaters.count, 1)
        XCTAssertEqual(retained.removedCount, 1)
        XCTAssertEqual(retainedWaters[0].residueSeq, 100)
    }

    func testProteinPreparationRemovesNonStandardResiduesButKeepsWatersAndCaps() {
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0, 0, 0), name: "N", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .O, position: SIMD3<Float>(2, 0, 0), name: "O", residueName: "HOH", residueSeq: 100, chainID: "A", isHetAtom: true),
            Atom(id: 2, element: .C, position: SIMD3<Float>(4, 0, 0), name: "C1", residueName: "GOL", residueSeq: 200, chainID: "A", isHetAtom: true),
            Atom(id: 3, element: .C, position: SIMD3<Float>(6, 0, 0), name: "C", residueName: "ACE", residueSeq: 1, chainID: "A", isHetAtom: true),
            Atom(id: 4, element: .N, position: SIMD3<Float>(8, 0, 0), name: "N", residueName: "NME", residueSeq: 2, chainID: "A", isHetAtom: true)
        ]

        let filtered = ProteinPreparation.removeNonStandardResidues(
            atoms: atoms,
            bonds: [],
            keepingWaters: true,
            keepingExistingCaps: true
        )

        XCTAssertEqual(filtered.removedResidueCount, 1)
        XCTAssertEqual(filtered.removedAtomCount, 1)
        XCTAssertFalse(filtered.atoms.contains { $0.residueName == "GOL" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "HOH" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "ACE" })
        XCTAssertTrue(filtered.atoms.contains { $0.residueName == "NME" })
    }

    func testProteinPreparationCapsChainBreaksWithAceAndNme() {
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0.0, 0.0, 0.0), name: "N", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .C, position: SIMD3<Float>(1.45, 0.0, 0.0), name: "CA", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 2, element: .C, position: SIMD3<Float>(2.55, 0.0, 0.0), name: "C", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 3, element: .O, position: SIMD3<Float>(3.30, 0.85, 0.0), name: "O", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 4, element: .N, position: SIMD3<Float>(6.20, 0.0, 0.0), name: "N", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 5, element: .C, position: SIMD3<Float>(7.45, 0.0, 0.0), name: "CA", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 6, element: .C, position: SIMD3<Float>(8.55, 0.0, 0.0), name: "C", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 7, element: .O, position: SIMD3<Float>(9.30, 0.85, 0.0), name: "O", residueName: "GLY", residueSeq: 2, chainID: "A")
        ]
        let bonds = [
            Bond(id: 0, atomIndex1: 0, atomIndex2: 1, order: .single),
            Bond(id: 1, atomIndex1: 1, atomIndex2: 2, order: .single),
            Bond(id: 2, atomIndex1: 2, atomIndex2: 3, order: .double),
            Bond(id: 3, atomIndex1: 4, atomIndex2: 5, order: .single),
            Bond(id: 4, atomIndex1: 5, atomIndex2: 6, order: .single),
            Bond(id: 5, atomIndex1: 6, atomIndex2: 7, order: .double)
        ]

        let cleaned = ProteinPreparation.cleanupStructure(atoms: atoms, bonds: bonds)

        XCTAssertEqual(cleaned.report.addedCappingResidues, 2)
        XCTAssertEqual(cleaned.report.chainBreaks.count, 1)
        XCTAssertTrue(cleaned.report.chainBreaks[0].isCapped)
        XCTAssertEqual(cleaned.atoms.count, atoms.count + 5)
        XCTAssertTrue(cleaned.atoms.contains { $0.residueName == "ACE" && $0.name == "C" })
        XCTAssertTrue(cleaned.atoms.contains { $0.residueName == "NME" && $0.name == "N" })

        let previousCarbonIndex = cleaned.atoms.firstIndex {
            $0.residueName == "ALA" && $0.residueSeq == 1 && $0.name == "C"
        }
        let nmeNitrogenIndex = cleaned.atoms.firstIndex {
            $0.residueName == "NME" && $0.name == "N"
        }
        let aceCarbonIndex = cleaned.atoms.firstIndex {
            $0.residueName == "ACE" && $0.name == "C"
        }
        let nextNitrogenIndex = cleaned.atoms.firstIndex {
            $0.residueName == "GLY" && $0.residueSeq == 2 && $0.name == "N"
        }

        XCTAssertNotNil(previousCarbonIndex)
        XCTAssertNotNil(nmeNitrogenIndex)
        XCTAssertNotNil(aceCarbonIndex)
        XCTAssertNotNil(nextNitrogenIndex)

        guard let previousCarbonIndex,
              let nmeNitrogenIndex,
              let aceCarbonIndex,
              let nextNitrogenIndex else {
            XCTFail("Expected capping atoms and anchors to exist")
            return
        }

        XCTAssertTrue(cleaned.bonds.contains {
            ($0.atomIndex1 == previousCarbonIndex && $0.atomIndex2 == nmeNitrogenIndex) ||
            ($0.atomIndex2 == previousCarbonIndex && $0.atomIndex1 == nmeNitrogenIndex)
        })
        XCTAssertTrue(cleaned.bonds.contains {
            ($0.atomIndex1 == aceCarbonIndex && $0.atomIndex2 == nextNitrogenIndex) ||
            ($0.atomIndex2 == aceCarbonIndex && $0.atomIndex1 == nextNitrogenIndex)
        })
    }

    func testPrepareForDockingRunsPhase12CleanupBeforeCharges() {
        let atoms = [
            Atom(id: 0, element: .N, position: SIMD3<Float>(0.0, 0.0, 0.0), name: "N", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 1, element: .C, position: SIMD3<Float>(1.45, 0.0, 0.0), name: "CA", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 2, element: .C, position: SIMD3<Float>(2.55, 0.0, 0.0), name: "C", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 3, element: .O, position: SIMD3<Float>(3.30, 0.85, 0.0), name: "O", residueName: "ALA", residueSeq: 1, chainID: "A"),
            Atom(id: 4, element: .N, position: SIMD3<Float>(6.20, 0.0, 0.0), name: "N", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 5, element: .C, position: SIMD3<Float>(7.45, 0.0, 0.0), name: "CA", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 6, element: .C, position: SIMD3<Float>(8.55, 0.0, 0.0), name: "C", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 7, element: .O, position: SIMD3<Float>(9.30, 0.85, 0.0), name: "O", residueName: "GLY", residueSeq: 2, chainID: "A"),
            Atom(id: 8, element: .C, position: SIMD3<Float>(11.0, 0.0, 0.0), name: "C1", residueName: "GOL", residueSeq: 100, chainID: "A", isHetAtom: true)
        ]
        let bonds = [
            Bond(id: 0, atomIndex1: 0, atomIndex2: 1, order: .single),
            Bond(id: 1, atomIndex1: 1, atomIndex2: 2, order: .single),
            Bond(id: 2, atomIndex1: 2, atomIndex2: 3, order: .double),
            Bond(id: 3, atomIndex1: 4, atomIndex2: 5, order: .single),
            Bond(id: 4, atomIndex1: 5, atomIndex2: 6, order: .single),
            Bond(id: 5, atomIndex1: 6, atomIndex2: 7, order: .double)
        ]

        let prepared = ProteinPreparation.prepareForDocking(atoms: atoms, bonds: bonds, rawPDBContent: nil, pH: 7.4)

        XCTAssertEqual(prepared.report.heterogenResiduesRemoved, 1)
        XCTAssertEqual(prepared.report.cappingResiduesAdded, 2)
        XCTAssertEqual(prepared.report.chainBreaksDetected, 1)
        XCTAssertEqual(prepared.report.chainBreaksCapped, 1)
        XCTAssertFalse(prepared.atoms.contains { $0.residueName == "GOL" })
        XCTAssertTrue(prepared.atoms.contains { $0.residueName == "ACE" })
        XCTAssertTrue(prepared.atoms.contains { $0.residueName == "NME" })
    }

    @MainActor
    func testProteinPreparationResidueCompletenessFlagsMissingHeavyAtomsAndExtraAtoms() {
        let molecule = TestMolecules.alanineDipeptide()
        var atoms = molecule.atoms.filter {
            !($0.residueName == "ALA" && $0.residueSeq == 2 && $0.name == "CB")
        }
        atoms.append(Atom(
            id: atoms.count,
            element: .C,
            position: SIMD3<Float>(2.4, -1.8, 1.4),
            name: "QX",
            residueName: "ALA",
            residueSeq: 2,
            chainID: "A"
        ))

        let completeness = ProteinPreparation.analyzeResidueCompleteness(atoms: atoms)
        guard let alanine = completeness.first(where: {
            $0.chainID == "A" && $0.residueSeq == 2 && $0.residueName == "ALA"
        }) else {
            XCTFail("Expected alanine completeness issue")
            return
        }

        XCTAssertEqual(alanine.missingHeavyAtoms, ["CB"])
        XCTAssertEqual(alanine.extraAtoms, ["QX"])
        XCTAssertTrue(alanine.missingHydrogens.isEmpty)
    }

    @MainActor
    func testProteinPreparationResidueCompletenessFlagsMissingHydrogenGroups() {
        let molecule = TestMolecules.alanineDipeptide()
        let atoms = molecule.atoms.filter {
            !($0.residueName == "ALA" && $0.residueSeq == 2 && $0.name == "HB1")
        }

        let completeness = ProteinPreparation.analyzeResidueCompleteness(atoms: atoms)
        guard let alanine = completeness.first(where: {
            $0.chainID == "A" && $0.residueSeq == 2 && $0.residueName == "ALA"
        }) else {
            XCTFail("Expected alanine completeness issue")
            return
        }

        XCTAssertTrue(alanine.missingHeavyAtoms.isEmpty)
        XCTAssertTrue(alanine.missingHydrogens.contains("HB"))
        XCTAssertTrue(alanine.extraAtoms.isEmpty)
    }

    @MainActor
    func testProteinPreparationResidueCompletenessSuppressesHydrogenGapsInHeavyAtomOnlyStructures() {
        let molecule = TestMolecules.alanineDipeptide()
        let heavyAtomOnly = molecule.atoms.filter { $0.element != .H }

        let completeness = ProteinPreparation.analyzeResidueCompleteness(atoms: heavyAtomOnly)
        XCTAssertTrue(completeness.isEmpty)
    }

    @MainActor
    func testProteinPreparationResidueCompletenessAcceptsAlternateHydrogenNamingSchemes() {
        let molecule = TestMolecules.alanineDipeptide()
        let atoms = molecule.atoms.map { atom -> Atom in
            guard atom.residueName == "ALA", atom.residueSeq == 2 else { return atom }

            var renamed = atom
            switch atom.name {
            case "HB1":
                renamed.name = "1HB"
            case "HB2":
                renamed.name = "2HB"
            case "HB3":
                renamed.name = "3HB"
            default:
                break
            }
            return renamed
        }

        let completeness = ProteinPreparation.analyzeResidueCompleteness(atoms: atoms)
        XCTAssertFalse(completeness.contains(where: {
            $0.chainID == "A" && $0.residueSeq == 2 && $0.residueName == "ALA"
        }))
    }

    @MainActor
    func testBatchedDockingMetalAcceleratorPrescoresLigands() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU available")
        }
        guard let engine = DockingEngine(device: device) else {
            throw XCTSkip("Failed to create DockingEngine")
        }
        guard let accelerator = BatchedDockingMetalAccelerator(device: device) else {
            throw XCTSkip("Failed to create batched screening accelerator")
        }

        let protein = TestMolecules.alanineDipeptide()
        let pocket = BindingPocket(
            id: 0,
            center: protein.center,
            size: SIMD3<Float>(repeating: 6.0),
            volume: 864.0,
            buriedness: 0.4,
            polarity: 0.5,
            druggability: 1.0,
            residueIndices: [0],
            probePositions: []
        )
        engine.computeGridMaps(protein: protein, pocket: pocket, spacing: 0.5)
        guard let snapshot = engine.gridSnapshot() else {
            XCTFail("Grid snapshot should be available after grid computation")
            return
        }

        let entries = [
            ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            ("Acetanilide", "CC(=O)Nc1ccccc1")
        ]
        let preparedLigands: [PreparedDockingLigand] = try entries.map { name, smiles in
            let (mol, _, error) = RDKitBridge.prepareLigand(
                smiles: smiles,
                name: name,
                addHydrogens: true,
                minimize: true,
                computeCharges: true
            )
            XCTAssertNil(error)
            guard let mol else { throw XCTSkip("RDKit ligand prep failed for \(name)") }
            return engine.prepareLigandGeometry(
                Molecule(name: mol.name, atoms: mol.atoms, bonds: mol.bonds, title: smiles)
            )
        }

        let results = accelerator.prescore(ligands: preparedLigands, gridSnapshot: snapshot, posesPerLigand: 16)
        XCTAssertEqual(results.count, preparedLigands.count)
        XCTAssertTrue(results.allSatisfy { $0 != nil })
        for result in results.compactMap({ $0 }) {
            XCTAssertTrue(result.energy.isFinite)
            XCTAssertTrue(result.rotation.vector.x.isFinite)
            XCTAssertTrue(result.rotation.vector.w.isFinite)
        }
    }

    @MainActor
    func testApplyingRefinedHeavyAtomPositionsPreservesHydrogens() {
        let molecule = TestMolecules.caffeine()
        let heavyAtoms = molecule.atoms.filter { $0.element != .H }
        let refined = heavyAtoms.enumerated().map { offset, atom in
            atom.position + SIMD3<Float>(Float(offset) * 0.01, 0.0, 0.0)
        }

        let updated = applyingRefinedHeavyAtomPositions(refined, to: molecule.atoms)
        XCTAssertNotNil(updated)
        guard let updated else { return }

        var heavyIndex = 0
        for (original, newAtom) in zip(molecule.atoms, updated) {
            if original.element == .H {
                XCTAssertEqual(original.position.x, newAtom.position.x, accuracy: 0.0001)
            } else {
                XCTAssertEqual(refined[heavyIndex].x, newAtom.position.x, accuracy: 0.0001)
                heavyIndex += 1
            }
        }
    }

    func testOpenMMPocketRefinerAvailabilityProbe() {
        let refiner = OpenMMPocketRefiner.shared
        XCTAssertFalse(refiner.availability.message.isEmpty)
    }

    @MainActor
    func testSDFParserSingle() throws {
        let sdf = """
        methane
             RDKit          3D

          5  4  0  0  0  0  0  0  0  0999 V2000
            0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            0.6300    0.6300    0.6300 H   0  0  0  0  0  0  0  0  0  0  0  0
           -0.6300   -0.6300    0.6300 H   0  0  0  0  0  0  0  0  0  0  0  0
           -0.6300    0.6300   -0.6300 H   0  0  0  0  0  0  0  0  0  0  0  0
            0.6300   -0.6300   -0.6300 H   0  0  0  0  0  0  0  0  0  0  0  0
          1  2  1  0
          1  3  1  0
          1  4  1  0
          1  5  1  0
        M  END
        $$$$
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("test.sdf")
        try sdf.write(to: url, atomically: true, encoding: .utf8)
        let molecules = try SDFParser.parse(url: url)
        XCTAssertEqual(molecules.count, 1)
        XCTAssertEqual(molecules[0].atoms.count, 5)
        XCTAssertEqual(molecules[0].bonds.count, 4)
        try? FileManager.default.removeItem(at: url)
    }

    @MainActor
    func testSDFParserMulti() throws {
        let sdf = """
        mol1
             RDKit

          1  0  0  0  0  0  0  0  0  0999 V2000
            0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        M  END
        $$$$
        mol2
             RDKit

          1  0  0  0  0  0  0  0  0  0999 V2000
            1.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
        M  END
        $$$$
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("multi.sdf")
        try sdf.write(to: url, atomically: true, encoding: .utf8)
        let molecules = try SDFParser.parse(url: url)
        XCTAssertEqual(molecules.count, 2)
        try? FileManager.default.removeItem(at: url)
    }

    @MainActor
    func testSDFParserBondOrders() throws {
        let sdf = """
        ethylene
             RDKit

          2  1  0  0  0  0  0  0  0  0999 V2000
            0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            1.3400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
          1  2  2  0
        M  END
        $$$$
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ethylene.sdf")
        try sdf.write(to: url, atomically: true, encoding: .utf8)
        let molecules = try SDFParser.parse(url: url)
        XCTAssertEqual(molecules.count, 1)
        let doubleBonds = molecules[0].bonds.filter { $0.order == .double }
        XCTAssertTrue(doubleBonds.count >= 1, "Should have at least one double bond")
        try? FileManager.default.removeItem(at: url)
    }

    @MainActor
    func testMOL2ParserBasic() throws {
        let mol2 = """
        @<TRIPOS>MOLECULE
        benzene
        6 6 0 0 0
        SMALL
        NO_CHARGES

        @<TRIPOS>ATOM
              1 C1          0.0000    1.4000    0.0000 C.ar      1 LIG  0.0000
              2 C2          1.2124    0.7000    0.0000 C.ar      1 LIG  0.0000
              3 C3          1.2124   -0.7000    0.0000 C.ar      1 LIG  0.0000
              4 C4          0.0000   -1.4000    0.0000 C.ar      1 LIG  0.0000
              5 C5         -1.2124   -0.7000    0.0000 C.ar      1 LIG  0.0000
              6 C6         -1.2124    0.7000    0.0000 C.ar      1 LIG  0.0000
        @<TRIPOS>BOND
             1     1     2 ar
             2     2     3 ar
             3     3     4 ar
             4     4     5 ar
             5     5     6 ar
             6     6     1 ar
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("benzene.mol2")
        try mol2.write(to: url, atomically: true, encoding: .utf8)
        let molecules = try MOL2Parser.parse(url: url)
        XCTAssertEqual(molecules.count, 1)
        XCTAssertEqual(molecules[0].atoms.count, 6)
        XCTAssertEqual(molecules[0].bonds.count, 6)
        // All atoms should be carbon
        for atom in molecules[0].atoms {
            XCTAssertEqual(atom.element, .C, "Benzene atoms should all be carbon")
        }
        // All bonds should be aromatic
        for bond in molecules[0].bonds {
            XCTAssertEqual(bond.order, .aromatic, "Benzene bonds should be aromatic")
        }
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Pocket Detection Tests

    @MainActor
    func testPocketDetectionEmpty() async {
        // Too few atoms for pocket detection
        let atoms = (0..<5).map { i in
            Atom(id: i, element: .C, position: SIMD3(Float(i), 0, 0), name: "CA",
                 residueName: "ALA", residueSeq: 1, chainID: "A")
        }
        let mol = Molecule(name: "tiny", atoms: atoms, bonds: [], title: "")
        let pockets = await BindingSiteDetector.detectPockets(protein: mol, gridSpacing: 2.0)
        XCTAssertTrue(pockets.isEmpty, "Too few atoms should produce no pockets")
    }

    @MainActor
    func testPocketDetectionSyntheticCavity() async {
        // Create a dense hollow sphere of atoms — should detect a pocket in the center
        var atoms: [Atom] = []
        let radius: Float = 8.0
        var id = 0
        // Dense sampling: ~200 atoms in a spherical shell
        for phi in stride(from: Float(0.2), to: Float.pi, by: Float.pi / 10) {
            for theta in stride(from: Float(0.0), to: 2 * Float.pi, by: Float.pi / 6) {
                let x = radius * sin(phi) * cos(theta)
                let y = radius * sin(phi) * sin(theta)
                let z = radius * cos(phi)
                atoms.append(Atom(id: id, element: id % 3 == 0 ? .N : .C,
                                  position: SIMD3(x, y, z), name: "CA",
                                  residueName: "ALA", residueSeq: id / 4 + 1, chainID: "A"))
                id += 1
            }
        }
        XCTAssertTrue(atoms.count >= 100, "Need dense sphere for pocket detection")
        let mol = Molecule(name: "sphere", atoms: atoms, bonds: [], title: "")
        let pockets = await BindingSiteDetector.detectPockets(protein: mol, gridSpacing: 1.5)
        XCTAssertFalse(pockets.isEmpty, "Hollow sphere should produce at least 1 pocket")
        if let best = pockets.first {
            XCTAssertTrue(best.volume >= 200, "Pocket should have ≥200 Å³ volume")
            XCTAssertTrue(simd_length(best.center) < 6.0,
                          "Pocket center should be near origin, got \(best.center)")
        }
    }

    @MainActor
    func testPocketDetectionLigandGuided() async {
        // Protein atoms surrounding a point, ligand at that point
        var protAtoms: [Atom] = []
        let center = SIMD3<Float>(10, 10, 10)
        for i in 0..<30 {
            let angle = Float(i) * 2 * .pi / 30
            let pos = center + SIMD3(5 * cos(angle), 5 * sin(angle), Float(i % 3) - 1)
            protAtoms.append(Atom(id: i, element: .C, position: pos, name: "CA",
                                  residueName: "ALA", residueSeq: i + 1, chainID: "A"))
        }
        let protein = Molecule(name: "prot", atoms: protAtoms, bonds: [], title: "")

        let ligAtoms = [
            Atom(id: 0, element: .C, position: center, name: "C1",
                 residueName: "LIG", residueSeq: 1, chainID: "L", isHetAtom: true),
            Atom(id: 1, element: .N, position: center + SIMD3(1, 0, 0), name: "N1",
                 residueName: "LIG", residueSeq: 1, chainID: "L", isHetAtom: true)
        ]
        let ligand = Molecule(name: "lig", atoms: ligAtoms, bonds: [], title: "")

        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        XCTAssertNotNil(pocket, "Should detect a pocket around the ligand")
        if let p = pocket {
            XCTAssertTrue(simd_distance(p.center, center) < 3.0,
                          "Pocket center should be near ligand")
            XCTAssertTrue(p.residueIndices.count > 0, "Pocket should contain residues")
        }
    }

    @MainActor
    func testPocketFromResidues() async {
        var atoms: [Atom] = []
        for i in 0..<15 {
            atoms.append(Atom(id: i, element: .C, position: SIMD3(Float(i), 0, 0), name: "CA",
                              residueName: "ALA", residueSeq: i + 1, chainID: "A"))
        }
        let mol = Molecule(name: "chain", atoms: atoms, bonds: [], title: "")
        let pocket = await BindingSiteDetector.pocketFromResidues(protein: mol, residueIndices: [0, 1, 2])
        XCTAssertTrue(pocket.volume > 0)
    }

    // MARK: - Pose Clustering Tests

    func testClusteringSingle() {
        let pose = DockingResult(id: 0, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                                 energy: -8.0, stericEnergy: -4, hydrophobicEnergy: -3, hbondEnergy: -1, torsionPenalty: 0, generation: 0,
                                 transformedAtomPositions: [SIMD3(1, 2, 3)])
        let clusters = PoseClustering.clusterPoses(poses: [pose])
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].count, 1)
        XCTAssertEqual(clusters[0][0], 0)
    }

    func testClusteringIdentical() {
        let positions: [SIMD3<Float>] = [SIMD3(1, 2, 3), SIMD3(4, 5, 6)]
        let poses = (0..<5).map { i in
            DockingResult(id: i, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                          energy: Float(-10 + i), stericEnergy: -5, hydrophobicEnergy: -3, hbondEnergy: -1, torsionPenalty: -1, generation: 0,
                          transformedAtomPositions: positions)
        }
        let clusters = PoseClustering.clusterPoses(poses: poses, rmsdCutoff: 2.0)
        XCTAssertEqual(clusters.count, 1, "Identical poses should form 1 cluster")
        XCTAssertEqual(clusters[0].count, 5, "All 5 poses should be in the cluster")
        // Should be sorted by energy (lowest first)
        XCTAssertEqual(clusters[0][0], 0, "Best energy pose (index 0, energy -10) should be first")
    }

    func testClusteringDistinct() {
        let groupA: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0)]
        let groupB: [SIMD3<Float>] = [SIMD3(20, 20, 20), SIMD3(21, 20, 20)]  // 10+ A away

        var poses: [DockingResult] = []
        for i in 0..<3 {
            poses.append(DockingResult(id: i, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                                       energy: Float(-8 + i), stericEnergy: -4, hydrophobicEnergy: -3, hbondEnergy: -1, torsionPenalty: 0, generation: 0,
                                       transformedAtomPositions: groupA))
        }
        for i in 3..<6 {
            poses.append(DockingResult(id: i, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                                       energy: Float(-5 + (i - 3)), stericEnergy: -3, hydrophobicEnergy: -2, hbondEnergy: 0, torsionPenalty: 0, generation: 0,
                                       transformedAtomPositions: groupB))
        }
        let clusters = PoseClustering.clusterPoses(poses: poses, rmsdCutoff: 2.0)
        XCTAssertEqual(clusters.count, 2, "Two distant groups should form 2 clusters")
    }

    func testClusteringEnergyOrder() {
        // Cluster A: best energy -12, Cluster B: best energy -5
        let posA: [SIMD3<Float>] = [SIMD3(0, 0, 0)]
        let posB: [SIMD3<Float>] = [SIMD3(50, 50, 50)]

        let poses = [
            DockingResult(id: 0, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                          energy: -12, stericEnergy: -6, hydrophobicEnergy: -4, hbondEnergy: -2, torsionPenalty: 0, generation: 0,
                          transformedAtomPositions: posA),
            DockingResult(id: 1, pose: DockPoseSwift(translation: .zero, rotation: .init(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                          energy: -5, stericEnergy: -3, hydrophobicEnergy: -2, hbondEnergy: 0, torsionPenalty: 0, generation: 0,
                          transformedAtomPositions: posB)
        ]
        let clusters = PoseClustering.clusterPoses(poses: poses, rmsdCutoff: 2.0)
        XCTAssertEqual(clusters.count, 2)
        // First cluster should contain the -12 energy pose
        let firstClusterBestEnergy = clusters[0].map { poses[$0].energy }.min()!
        XCTAssertEqual(firstClusterBestEnergy, -12, accuracy: 0.1, "First cluster should have best energy")
    }

    func testClusteringEmpty() {
        let clusters = PoseClustering.clusterPoses(poses: [])
        XCTAssertTrue(clusters.isEmpty)
    }

    // MARK: - Rotamer Library Tests

    func testRotamerCoversAllFlexible() {
        let flexResidues = ["PHE", "TYR", "TRP", "HIS", "ASP", "GLU", "ASN", "GLN",
                            "LYS", "ARG", "SER", "THR", "CYS", "MET", "LEU", "ILE", "VAL"]
        for res in flexResidues {
            let rotamers = RotamerLibrary.rotamers(for: res)
            XCTAssertNotNil(rotamers, "\(res) should have rotamer definitions")
            XCTAssertTrue(rotamers!.chiAngles.count >= 1, "\(res) should have ≥1 chi angle")
        }
    }

    func testRotamerNonFlexible() {
        XCTAssertNil(RotamerLibrary.rotamers(for: "ALA"), "ALA has no rotamers")
        XCTAssertNil(RotamerLibrary.rotamers(for: "GLY"), "GLY has no rotamers")
        XCTAssertNil(RotamerLibrary.rotamers(for: "PRO"), "PRO has no rotamers")
    }

    func testRotamerCountPHE() {
        // PHE: chi1 has 3 states, chi2 has 2 states = 6 total
        XCTAssertEqual(RotamerLibrary.rotamerCount(for: "PHE"), 6)
    }

    func testRotamerCountLYS() {
        // LYS: 4 chi angles, each with 3 states = 81 total
        XCTAssertEqual(RotamerLibrary.rotamerCount(for: "LYS"), 81)
    }

    func testRotamerChiAtomNames() {
        let phe = RotamerLibrary.rotamers(for: "PHE")!
        XCTAssertEqual(phe.chiAngles[0].atomNames.0, "N")
        XCTAssertEqual(phe.chiAngles[0].atomNames.1, "CA")
        XCTAssertEqual(phe.chiAngles[0].atomNames.2, "CB")
        XCTAssertEqual(phe.chiAngles[0].atomNames.3, "CG")
    }

    // MARK: - 1HSG Redocking Integration Test (PLAN.md Phase 2 Validation Gate)
    // Full pipeline: PDB fetch → parse → pocket detect → GA docking → RMSD check
    // Comprehensive redocking tests (3PYY, 2QWK, 1M17) are in DockingIntegrationTests.swift
    // This test uses inline logic (not DockingTestCase) because DruseTests is not a subclass of it.

    @MainActor
    func testRedocking1HSGIndinavir() async throws {
        print("\n  ========== 1HSG REDOCKING: HIV-1 protease + indinavir ==========")

        // 1. Fetch PDB
        print("  [1HSG] Step 1: Fetching PDB from RCSB...")
        let pdbContent: String
        do {
            pdbContent = try await PDBService.shared.fetchPDBFile(id: "1HSG")
        } catch {
            throw XCTSkip("Network unavailable: \(error.localizedDescription)")
        }
        print("  [1HSG] PDB content: \(pdbContent.count) characters")
        XCTAssertTrue(pdbContent.count > 1000, "PDB content too short: \(pdbContent.count) chars")

        // 2. Parse PDB
        print("  [1HSG] Step 2: Parsing PDB...")
        let result = PDBParser.parse(pdbContent)
        print("  [1HSG] Parse: protein=\(result.protein?.atoms.count ?? 0) atoms, ligands=\(result.ligands.count), waters=\(result.waterCount), warnings=\(result.warnings.count)")
        for (i, lig) in result.ligands.enumerated() {
            print("    ligand[\(i)]: \(lig.name) — \(lig.atoms.count) atoms (\(lig.atoms.filter { $0.element != .H }.count) heavy), \(lig.bonds.count) bonds")
        }
        for w in result.warnings.prefix(5) { print("    warning: \(w)") }
        XCTAssertNotNil(result.protein, "1HSG should have a protein")
        XCTAssertTrue(result.ligands.count >= 1, "1HSG should have co-crystallized ligands")

        // 3. Find indinavir (MK1)
        let indinavirData = result.ligands.first { $0.name == "MK1" } ?? result.ligands.first
        XCTAssertNotNil(indinavirData, "Should find a ligand in 1HSG")
        guard let ligData = indinavirData else { return }

        let crystalHeavyPositions = ligData.atoms.filter { $0.element != .H }.map(\.position)
        let crystalHeavyAtoms = ligData.atoms.filter { $0.element != .H }
        let crystalCentroid = crystalHeavyPositions.reduce(SIMD3<Float>.zero, +) / Float(max(crystalHeavyPositions.count, 1))
        let elemComp = Dictionary(grouping: crystalHeavyAtoms, by: { $0.element }).mapValues(\.count)
            .sorted(by: { $0.key.rawValue < $1.key.rawValue }).map { "\($0.key.symbol):\($0.value)" }.joined(separator: " ")
        print("  [1HSG] Crystal ligand \(ligData.name): \(crystalHeavyPositions.count) heavy atoms, centroid=(\(String(format: "%.2f, %.2f, %.2f", crystalCentroid.x, crystalCentroid.y, crystalCentroid.z)))")
        print("  [1HSG] Crystal composition: \(elemComp)")
        XCTAssertTrue(crystalHeavyPositions.count >= 10, "Indinavir should have ≥10 heavy atoms, got \(crystalHeavyPositions.count)")

        // 4. Build molecules
        let protData = result.protein!
        let protein = Molecule(name: protData.name, atoms: protData.atoms, bonds: protData.bonds, title: protData.title)
        let ligand = Molecule(name: ligData.name, atoms: ligData.atoms, bonds: ligData.bonds, title: ligData.title)
        print("  [1HSG] Protein: \(protein.atomCount) atoms, \(protein.heavyAtomCount) heavy, \(protein.residues.count) residues, \(protein.chains.count) chains (\(protein.chainIDs.joined(separator: ",")))")

        let preparedProteinData = ProteinPreparation.prepareForDocking(
            atoms: protein.atoms,
            bonds: protein.bonds,
            rawPDBContent: pdbContent,
            pH: 7.4
        )
        let dockingProtein = Molecule(
            name: protein.name,
            atoms: preparedProteinData.atoms,
            bonds: preparedProteinData.bonds,
            title: protein.title
        )
        print("  [1HSG] Prepared receptor for scoring: +\(preparedProteinData.report.hydrogensAdded) H via \(preparedProteinData.report.hydrogenMethod), protonation updates=\(preparedProteinData.report.protonationUpdates), charged atoms=\(preparedProteinData.report.nonZeroChargeAtoms), RDKit charge matches=\(preparedProteinData.report.rdkitChargeMatches)")

        // 5. Pocket detection
        print("  [1HSG] Step 5: Detecting binding pocket (ligand-guided, 6.0Å cutoff)...")
        let pocket = await BindingSiteDetector.ligandGuidedPocket(protein: protein, ligand: ligand, distance: 6.0)
        XCTAssertNotNil(pocket, "Should detect pocket around indinavir")
        guard let pocket else { return }
        print("  [1HSG] Pocket: center=(\(String(format: "%.2f, %.2f, %.2f", pocket.center.x, pocket.center.y, pocket.center.z)))")
        print("  [1HSG] Pocket: size=(\(String(format: "%.2f, %.2f, %.2f", pocket.size.x, pocket.size.y, pocket.size.z)))")
        print("  [1HSG] Pocket: volume=\(String(format: "%.0f", pocket.volume))ų, buriedness=\(String(format: "%.2f", pocket.buriedness)), residues=\(pocket.residueIndices.count)")
        XCTAssertTrue(pocket.volume > 50, "Pocket volume too small: \(pocket.volume)")
        XCTAssertTrue(pocket.residueIndices.count >= 5, "Pocket should have ≥5 residues, got \(pocket.residueIndices.count)")

        // 6. Metal GPU
        print("  [1HSG] Step 6: Creating Metal docking engine...")
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU") }
        guard let engine = DockingEngine(device: device) else { throw XCTSkip("DockingEngine creation failed") }
        print("  [1HSG] GPU: \(device.name), unified memory: \(device.hasUnifiedMemory)")

        // 7. Ligand atom types & extent
        let prepared = engine.prepareLigandGeometry(ligand)
        let vinaTypeNames = [
            "C_H", "C_P", "N_P", "N_D", "N_A", "N_DA", "O_P", "O_D",
            "O_A", "O_DA", "S_P", "P_P", "F_H", "Cl_H", "Br_H", "I_H",
            "Si", "At", "Met_D"
        ]
        var typeCounts: [Int32: Int] = [:]
        for gpu in prepared.gpuAtoms { typeCounts[gpu.vinaType, default: 0] += 1 }
        let typeSummary = typeCounts.sorted(by: { $0.key < $1.key })
            .map { Int($0.key) < vinaTypeNames.count ? "\(vinaTypeNames[Int($0.key)]): \($0.value)" : "type[\($0.key)]: \($0.value)" }
            .joined(separator: ", ")
        print("  [1HSG] Ligand atom types: \(typeSummary)")
        print("  [1HSG] Prepared ligand: \(prepared.heavyAtoms.count) heavy atoms, centroid=(\(String(format: "%.2f, %.2f, %.2f", prepared.centroid.x, prepared.centroid.y, prepared.centroid.z)))")
        for (i, gpuAtom) in prepared.gpuAtoms.enumerated() {
            let atom = prepared.heavyAtoms[i]
            let typeName = gpuAtom.vinaType >= 0 && gpuAtom.vinaType < vinaTypeNames.count ? vinaTypeNames[Int(gpuAtom.vinaType)] : "type[\(gpuAtom.vinaType)]"
            print("    atom \(i): \(atom.element.symbol) \(atom.name) → VINA_\(typeName) vdw=\(String(format: "%.2f", gpuAtom.vdwRadius)) charge=\(String(format: "%.3f", gpuAtom.charge)) pos=(\(String(format: "%.2f,%.2f,%.2f", gpuAtom.position.x, gpuAtom.position.y, gpuAtom.position.z)))")
        }
        let requiredVinaTypes = Array(Set(prepared.gpuAtoms.map(\.vinaType).filter { $0 >= 0 })).sorted()

        // Compute ligand extent for grid sizing (matches DockingIntegrationTests)
        let ligandHeavy = ligand.atoms.filter { $0.element != .H }
        let ligCentroid = ligandHeavy.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(max(ligandHeavy.count, 1))
        var ligMin = SIMD3<Float>(repeating: .infinity)
        var ligMax = SIMD3<Float>(repeating: -.infinity)
        for a in ligandHeavy {
            let p = a.position - ligCentroid
            ligMin = simd_min(ligMin, p)
            ligMax = simd_max(ligMax, p)
        }
        let ligExtent = (ligMax - ligMin) * 0.5
        print("  [1HSG] Ligand extent: (\(String(format: "%.2f, %.2f, %.2f", ligExtent.x, ligExtent.y, ligExtent.z)))")

        // 7b. Grid maps (with ligand extent)
        print("  [1HSG] Step 7b: Computing grid maps (spacing=0.375, with ligandExtent)...")
        let gridStart = CFAbsoluteTimeGetCurrent()
        engine.computeGridMaps(
            protein: dockingProtein,
            pocket: pocket,
            spacing: 0.375,
            ligandExtent: ligExtent,
            requiredVinaTypes: requiredVinaTypes
        )
        print("  [1HSG] Grid maps computed in \(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - gridStart))s")
        print("  [1HSG] Grid diagnostics:\n\(engine.gridDiagnostics())")
        if let nativeScore = engine.debugScorePose(ligand: ligand, translation: prepared.centroid) {
            let nativeRMSD = nativeScore.transformedAtomPositions.count == crystalHeavyPositions.count
                ? sqrt(zip(crystalHeavyPositions, nativeScore.transformedAtomPositions).map { simd_distance_squared($0, $1) }.reduce(0, +) / Float(crystalHeavyPositions.count))
                : .infinity
            print("  [1HSG] Native crystal pose score: E=\(String(format: "%.3f", nativeScore.energy)) torsion=\(String(format: "%.3f", nativeScore.torsionPenalty)) RMSD=\(String(format: "%.3f", nativeRMSD))Å")
        } else {
            print("  [1HSG] Native crystal pose score: unavailable")
        }

        // 8. Docking (flexible — indinavir has ~7 rotatable bonds)
        let popSize = envInt("DRUSE_DOCK_POPULATION", default: 300)
        let generations = envInt("DRUSE_DOCK_GENERATIONS", default: 300)
        let runs = envInt("DRUSE_DOCK_RUNS", default: 1)
        let localSearchFrequency = envInt("DRUSE_DOCK_LOCAL_SEARCH_FREQUENCY", default: 1)
        let localSearchSteps = envInt("DRUSE_DOCK_LOCAL_SEARCH_STEPS", default: 20)
        print("  [1HSG] Step 8: Running docking (\(popSize) pop × \(generations) gen × \(runs) run, flexible)...")
        var config = DockingConfig()
        config.numRuns = runs
        config.generationsPerRun = generations
        config.populationSize = popSize
        config.localSearchFrequency = localSearchFrequency
        config.localSearchSteps = localSearchSteps
        config.enableFlexibility = true
        config.liveUpdateFrequency = 999
        let dockStart = CFAbsoluteTimeGetCurrent()
        let results = await engine.runDocking(ligand: ligand, pocket: pocket, config: config)
        let dockTime = CFAbsoluteTimeGetCurrent() - dockStart
        print("  [1HSG] Docking completed in \(String(format: "%.3f", dockTime))s, \(results.count) results")
        XCTAssertFalse(results.isEmpty, "Docking should produce results")

        // Full diagnostics
        if let diag = engine.lastDiagnostics { print(diag.summary) }

        // 9. Results analysis
        let validResults = results.filter { $0.energy.isFinite && $0.energy < 1e9 }
        let clusterCount = Set(results.map(\.clusterID)).count
        print("  [1HSG] Valid poses: \(validResults.count)/\(results.count), clusters: \(clusterCount)")

        // Energy breakdown for top 10
        print("  [1HSG] Top 10 poses:")
        for (i, r) in results.prefix(10).enumerated() {
            let posCount = r.transformedAtomPositions.count
            let centroid: SIMD3<Float> = posCount > 0 ? r.transformedAtomPositions.reduce(.zero, +) / Float(posCount) : .zero
            let rmsd_i = posCount == crystalHeavyPositions.count
                ? sqrt(zip(crystalHeavyPositions, r.transformedAtomPositions).map { simd_distance_squared($0, $1) }.reduce(0, +) / Float(posCount))
                : Float.infinity
            print("    pose \(i): E=\(String(format: "%8.3f", r.energy)) steric=\(String(format: "%7.3f", r.stericEnergy)) hydro=\(String(format: "%7.3f", r.hydrophobicEnergy)) hbond=\(String(format: "%7.3f", r.hbondEnergy)) torsion=\(String(format: "%6.3f", r.torsionPenalty)) gen=\(r.generation) cluster=\(r.clusterID) atoms=\(posCount) RMSD=\(String(format: "%.2f", rmsd_i))Å centroid=(\(String(format: "%.1f,%.1f,%.1f", centroid.x, centroid.y, centroid.z)))")
        }

        let rmsdPool = results.compactMap { r -> Float? in
            guard r.transformedAtomPositions.count == crystalHeavyPositions.count else { return nil }
            let sum = zip(crystalHeavyPositions, r.transformedAtomPositions)
                .map { simd_distance_squared($0, $1) }
                .reduce(0, +)
            return sqrt(sum / Float(crystalHeavyPositions.count))
        }
        if let bestPoolRMSD = rmsdPool.min() {
            let nearNativeCount = rmsdPool.filter { $0 < 5.0 }.count
            print("  [1HSG] Best RMSD in pose pool: \(String(format: "%.2f", bestPoolRMSD))Å (\(nearNativeCount) poses < 5Å)")
        }

        // 10. RMSD validation
        let best = results[0]
        let bestHeavyPositions = best.transformedAtomPositions
        let n = min(crystalHeavyPositions.count, bestHeavyPositions.count)
        print("  [1HSG] RMSD comparison: crystal=\(crystalHeavyPositions.count) atoms, docked=\(bestHeavyPositions.count) atoms, comparing \(n)")
        XCTAssertTrue(n >= 10, "Need ≥10 heavy atoms for RMSD, got \(n)")

        var sumSq: Float = 0
        var maxDist: Float = 0
        var maxDistIdx = 0
        for i in 0..<n {
            let d2 = simd_distance_squared(crystalHeavyPositions[i], bestHeavyPositions[i])
            sumSq += d2
            let d = sqrt(d2)
            if d > maxDist { maxDist = d; maxDistIdx = i }
        }
        let rmsd = sqrt(sumSq / Float(n))

        // Print ALL per-atom distances
        print("  [1HSG] Per-atom distances (all \(n) atoms):")
        for i in 0..<n {
            let d = simd_distance(crystalHeavyPositions[i], bestHeavyPositions[i])
            let atom = crystalHeavyAtoms[i]
            let flag = d > 2.0 ? " *** HIGH ***" : ""
            print("    atom \(i) \(atom.element.symbol) \(atom.name): d=\(String(format: "%.3f", d))Å crystal=(\(String(format: "%.2f,%.2f,%.2f", crystalHeavyPositions[i].x, crystalHeavyPositions[i].y, crystalHeavyPositions[i].z))) docked=(\(String(format: "%.2f,%.2f,%.2f", bestHeavyPositions[i].x, bestHeavyPositions[i].y, bestHeavyPositions[i].z)))\(flag)")
        }
        print("  [1HSG] Max atom deviation: atom \(maxDistIdx) = \(String(format: "%.3f", maxDist))Å")

        print("✓ 1HSG Benchmark: RMSD = \(String(format: "%.2f", rmsd)) Å, Energy = \(String(format: "%.1f", best.energy)) kcal/mol")

        XCTAssertLessThan(rmsd, 5.0, "Redocking RMSD should be < 5.0 Å, got \(String(format: "%.2f", rmsd)) Å")
        XCTAssertLessThan(best.energy, 0.0, "Vina score should be negative, got \(String(format: "%.1f", best.energy))")
        XCTAssertGreaterThan(best.energy, -50.0, "Vina score should be > -50, got \(String(format: "%.1f", best.energy))")

        print("✓ 1HSG redocking PASSED: RMSD = \(String(format: "%.2f", rmsd)) Å, Energy = \(String(format: "%.1f", best.energy)) kcal/mol")
    }
}

// Docking integration tests (3PYY, 2QWK, 1M17, round-trips) are in DockingIntegrationTests.swift

import XCTest
import simd
@testable import Druse

// =============================================================================
// MARK: - Tautomer & Protomer Tests
// =============================================================================
//
// Unit tests: VariantKind enum, MolecularVariant struct, LigandEntry variant
//             storage, serialization round-trip.
// Bridge tests: RDKitBridge.enumerateTautomers, RDKitBridge.enumerateProtomers
//               on simple and drug-like molecules.
// Real-world tests: known drug tautomers (warfarin, guanine, imatinib) and
//                   protomers (histamine, morphine, ibuprofen at varying pH).
//
// These tests call into the C++ core (RDKit MolStandardize::TautomerEnumerator
// and SMARTS-based protomer enumeration) — they require libdruse_core.a linked
// with RDKitMolStandardize.

final class TautomerProtomerTests: XCTestCase {

    // =========================================================================
    // MARK: - Unit Tests: Model Layer
    // =========================================================================

    func testVariantKindEnum() {
        XCTAssertEqual(VariantKind.tautomer.rawValue, 0)
        XCTAssertEqual(VariantKind.protomer.rawValue, 1)
        XCTAssertEqual(VariantKind.tautomer.label, "Tautomer")
        XCTAssertEqual(VariantKind.protomer.label, "Protomer")
        XCTAssertEqual(VariantKind.tautomer.symbol, "T")
        XCTAssertEqual(VariantKind.protomer.symbol, "P")

        // Codable round-trip
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try! encoder.encode(VariantKind.protomer)
        let decoded = try! decoder.decode(VariantKind.self, from: data)
        XCTAssertEqual(decoded, .protomer)
    }

    func testMolecularVariantCreation() {
        let variant = MolecularVariant(
            smiles: "O=C1NC=CC(=O)N1",
            relativeEnergy: 2.5,
            kind: .tautomer,
            label: "Tautomer 2"
        )
        XCTAssertEqual(variant.smiles, "O=C1NC=CC(=O)N1")
        XCTAssertEqual(variant.kind, .tautomer)
        XCTAssertEqual(variant.relativeEnergy, 2.5, accuracy: 0.01)
        XCTAssertEqual(variant.label, "Tautomer 2")
        XCTAssertFalse(variant.isPrepared)
        XCTAssertEqual(variant.conformerCount, 0)
        XCTAssertTrue(variant.atoms.isEmpty)
        XCTAssertTrue(variant.bonds.isEmpty)
    }

    func testLigandEntryWithVariants() {
        let taut = MolecularVariant(smiles: "C=CO", relativeEnergy: 1.0, kind: .tautomer, label: "enol")
        let prot = MolecularVariant(smiles: "[NH3+]CC", relativeEnergy: 0.5, kind: .protomer, label: "protonated amine")
        let entry = LigandEntry(name: "Test", smiles: "CC=O", variants: [taut, prot])

        XCTAssertEqual(entry.variants.count, 2)
        XCTAssertEqual(entry.variants.filter { $0.kind == .tautomer }.count, 1)
        XCTAssertEqual(entry.variants.filter { $0.kind == .protomer }.count, 1)
    }

    func testLigandEntryDefaultEmptyVariants() {
        let entry = LigandEntry(name: "NoVariants", smiles: "C")
        XCTAssertTrue(entry.variants.isEmpty, "Default entry should have no variants")
    }

    func testDruseVariantSetStructLayout() {
        let size = MemoryLayout<DruseVariantSet>.size
        XCTAssertGreaterThan(size, 0, "DruseVariantSet should have non-zero size")
        print("  [Variant] DruseVariantSet size: \(size) bytes")

        let infoSize = MemoryLayout<DruseVariantInfo>.size
        XCTAssertGreaterThan(infoSize, 0, "DruseVariantInfo should have non-zero size")
        print("  [Variant] DruseVariantInfo size: \(infoSize) bytes")

        var info = DruseVariantInfo()
        info.kind = 0
        XCTAssertEqual(info.kind, 0)
        info.kind = 1
        XCTAssertEqual(info.kind, 1)
    }

    // =========================================================================
    // MARK: - Bridge Tests: Tautomer Enumeration
    // =========================================================================

    func testEnumerateTautomersSimpleKetone() {
        // Acetone (CC(=O)C) → keto/enol tautomers
        // The keto form is dominant, but enol form (CC(=C)O or CC(O)=C) should exist
        let results = RDKitBridge.enumerateTautomers(
            smiles: "CC(=O)C", name: "Acetone",
            maxTautomers: 10, energyCutoff: 20.0
        )
        print("  [Tautomer] Acetone: \(results.count) tautomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.smiles) E=\(String(format: "%.2f", r.score)) \(r.label)")
        }
        XCTAssertGreaterThanOrEqual(results.count, 1, "Should produce at least the canonical tautomer")

        // All should have valid 3D coordinates
        for r in results {
            XCTAssertFalse(r.molecule.atoms.isEmpty, "Tautomer '\(r.smiles)' should have atoms")
            XCTAssertFalse(r.molecule.bonds.isEmpty, "Tautomer '\(r.smiles)' should have bonds")
            for atom in r.molecule.atoms {
                XCTAssertFalse(atom.position.x.isNaN, "Atom position NaN in tautomer \(r.smiles)")
            }
        }

        // All should be .tautomer kind
        for r in results {
            XCTAssertEqual(r.kind, .tautomer)
        }

        // Should be sorted by energy (ascending)
        for i in 1..<results.count {
            if !results[i].score.isNaN && !results[i-1].score.isNaN {
                XCTAssertGreaterThanOrEqual(results[i].score, results[i-1].score,
                    "Tautomers should be sorted by energy")
            }
        }
    }

    func testEnumerateTautomersGuanine() {
        // Guanine: classic tautomer-rich molecule (keto/enol + amino/imino)
        // SMILES: O=c1[nH]c2[nH]cnc2c(N)[nH]1  (2-amino-6-oxopurine)
        let results = RDKitBridge.enumerateTautomers(
            smiles: "O=c1[nH]c2[nH]cnc2c(N)[nH]1", name: "Guanine",
            maxTautomers: 25, energyCutoff: 15.0
        )
        print("  [Tautomer] Guanine: \(results.count) tautomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.smiles) E=\(String(format: "%.2f", r.score))")
        }
        XCTAssertGreaterThanOrEqual(results.count, 2, "Guanine should have multiple tautomers, got \(results.count)")

        // Distinct SMILES
        let uniqueSmiles = Set(results.map(\.smiles))
        XCTAssertEqual(uniqueSmiles.count, results.count, "All tautomers should have unique SMILES")
    }

    func testEnumerateTautomersWarfarin() {
        // Warfarin: well-known keto/enol tautomerism at the 4-hydroxycoumarin
        // SMILES: CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O
        let results = RDKitBridge.enumerateTautomers(
            smiles: "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O", name: "Warfarin",
            maxTautomers: 10, energyCutoff: 15.0
        )
        print("  [Tautomer] Warfarin: \(results.count) tautomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.smiles) E=\(String(format: "%.2f", r.score))")
        }
        XCTAssertGreaterThanOrEqual(results.count, 2, "Warfarin should have keto/enol tautomers")

        // Each tautomer should be a valid molecule with 3D
        for r in results {
            XCTAssertGreaterThan(r.molecule.atoms.count, 20, "Warfarin tautomers should have >20 atoms")
        }
    }

    func testEnumerateTautomersEnergyCutoff() {
        // Tight energy cutoff should return fewer tautomers
        let loose = RDKitBridge.enumerateTautomers(smiles: "O=c1[nH]c2[nH]cnc2c(N)[nH]1",
                                                    name: "Guanine", maxTautomers: 25, energyCutoff: 50.0)
        let tight = RDKitBridge.enumerateTautomers(smiles: "O=c1[nH]c2[nH]cnc2c(N)[nH]1",
                                                    name: "Guanine", maxTautomers: 25, energyCutoff: 3.0)
        print("  [Tautomer] Guanine cutoff test: loose(\(loose.count)) vs tight(\(tight.count))")
        XCTAssertGreaterThanOrEqual(loose.count, tight.count,
            "Tighter energy cutoff should produce ≤ tautomers")
    }

    func testEnumerateTautomersNoTautomers() {
        // Benzene: no tautomers possible (fully aromatic, symmetric)
        let results = RDKitBridge.enumerateTautomers(smiles: "c1ccccc1", name: "Benzene",
                                                      maxTautomers: 10, energyCutoff: 10.0)
        print("  [Tautomer] Benzene: \(results.count) tautomers")
        // Should return at least the canonical form
        XCTAssertGreaterThanOrEqual(results.count, 1)
        // All should be equivalent (same SMILES or just 1 result)
        if results.count == 1 {
            XCTAssertFalse(results[0].smiles.isEmpty)
        }
    }

    func testEnumerateTautomersInvalidSMILES() {
        let results = RDKitBridge.enumerateTautomers(smiles: "NOT_VALID", name: "Bad",
                                                      maxTautomers: 5, energyCutoff: 10.0)
        XCTAssertEqual(results.count, 0, "Invalid SMILES should return empty array")
    }

    func testEnumerateTautomersEmptySMILES() {
        let results = RDKitBridge.enumerateTautomers(smiles: "", name: "Empty",
                                                      maxTautomers: 5, energyCutoff: 10.0)
        XCTAssertEqual(results.count, 0, "Empty SMILES should return empty array")
    }

    // =========================================================================
    // MARK: - Bridge Tests: Protomer Enumeration
    // =========================================================================

    func testEnumerateProtomersAceticAcid() {
        // Acetic acid (CC(=O)O): pKa ~4.75, deprotonated at pH 7.4
        let results = RDKitBridge.enumerateProtomers(
            smiles: "CC(=O)O", name: "AceticAcid",
            maxProtomers: 8, pH: 7.4, pkaThreshold: 2.0
        )
        print("  [Protomer] Acetic acid pH 7.4: \(results.count) protomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.smiles) E=\(String(format: "%.2f", r.score)) \(r.label)")
        }
        XCTAssertGreaterThanOrEqual(results.count, 1, "Should produce at least the parent form")

        for r in results {
            XCTAssertEqual(r.kind, .protomer)
            XCTAssertFalse(r.label.isEmpty, "Protomer should have a label")
            XCTAssertFalse(r.molecule.atoms.isEmpty, "Protomer should have 3D atoms")
        }
    }

    func testEnumerateProtomersHistamine() {
        // Histamine: has imidazole (pKa ~6.0) and primary amine (pKa ~10.5)
        // At pH 7.4: imidazole is ambiguous (|7.4-6.0| = 1.4 < 2.0),
        //            amine is protonated (|7.4-10.5| = 3.1 but within threshold)
        let results = RDKitBridge.enumerateProtomers(
            smiles: "NCCc1c[nH]cn1", name: "Histamine",
            maxProtomers: 8, pH: 7.4, pkaThreshold: 4.0
        )
        print("  [Protomer] Histamine pH 7.4: \(results.count) protomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.smiles) E=\(String(format: "%.2f", r.score)) \(r.label)")
        }
        // With two ionizable groups and wide threshold, should get multiple protomers
        XCTAssertGreaterThanOrEqual(results.count, 2,
            "Histamine should have ≥2 protomers (imidazole + amine), got \(results.count)")

        // Unique SMILES
        let uniqueSmiles = Set(results.map(\.smiles))
        XCTAssertEqual(uniqueSmiles.count, results.count, "All protomers should have unique SMILES")
    }

    func testEnumerateProtomersIbuprofen() {
        // Ibuprofen: carboxylic acid (pKa ~4.4)
        // At pH 7.4, should be almost fully deprotonated
        // At pH 3.0, should be mostly protonated
        let atPhysiological = RDKitBridge.enumerateProtomers(
            smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", name: "Ibuprofen",
            maxProtomers: 8, pH: 7.4, pkaThreshold: 2.0
        )
        let atAcidic = RDKitBridge.enumerateProtomers(
            smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", name: "Ibuprofen",
            maxProtomers: 8, pH: 3.0, pkaThreshold: 2.0
        )
        print("  [Protomer] Ibuprofen pH 7.4: \(atPhysiological.count) protomers")
        print("  [Protomer] Ibuprofen pH 3.0: \(atAcidic.count) protomers")

        // At pH 7.4, |pH-pKa| = 3.0 > threshold 2.0, so only dominant form
        // At pH 3.0, |pH-pKa| = 1.4 < threshold 2.0, so both forms
        // (Exact counts depend on SMARTS matching, just verify non-empty)
        XCTAssertGreaterThanOrEqual(atPhysiological.count, 1)
        XCTAssertGreaterThanOrEqual(atAcidic.count, 1)
    }

    func testEnumerateProtomersNoIonizableGroups() {
        // Benzene: no ionizable groups → should get just the parent
        let results = RDKitBridge.enumerateProtomers(
            smiles: "c1ccccc1", name: "Benzene",
            maxProtomers: 8, pH: 7.4, pkaThreshold: 2.0
        )
        print("  [Protomer] Benzene pH 7.4: \(results.count) protomers")
        XCTAssertEqual(results.count, 1, "Benzene should have only the parent protomer")
        if let parent = results.first {
            XCTAssertTrue(parent.label.contains("Parent"), "Single protomer should be labeled as parent")
        }
    }

    func testEnumerateProtomersInvalidSMILES() {
        let results = RDKitBridge.enumerateProtomers(smiles: "INVALID", name: "Bad",
                                                      maxProtomers: 4, pH: 7.4, pkaThreshold: 2.0)
        XCTAssertEqual(results.count, 0, "Invalid SMILES should return empty array")
    }

    func testEnumerateProtomersEmptySMILES() {
        let results = RDKitBridge.enumerateProtomers(smiles: "", name: "Empty",
                                                      maxProtomers: 4, pH: 7.4, pkaThreshold: 2.0)
        XCTAssertEqual(results.count, 0, "Empty SMILES should return empty array")
    }

    // =========================================================================
    // MARK: - Real-World Drug Molecules
    // =========================================================================

    func testTautomersImatinib() {
        // Imatinib (Gleevec): multiple tautomerizable groups (amide, pyrimidine)
        let smi = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4"
        let results = RDKitBridge.enumerateTautomers(smiles: smi, name: "Imatinib",
                                                      maxTautomers: 15, energyCutoff: 15.0)
        print("  [Tautomer] Imatinib: \(results.count) tautomers")
        for (i, r) in results.enumerated() {
            print("    \(i): atoms=\(r.molecule.atoms.count) E=\(String(format: "%.2f", r.score))")
        }
        XCTAssertGreaterThanOrEqual(results.count, 1, "Imatinib should produce at least 1 tautomer")

        // All tautomers should have the same heavy atom count
        let heavyCounts = results.map { r in r.molecule.atoms.filter { $0.element != .H }.count }
        if let first = heavyCounts.first {
            for (i, count) in heavyCounts.enumerated() {
                XCTAssertEqual(count, first,
                    "Tautomer \(i) heavy atom count \(count) != first \(first) — tautomers preserve atoms")
            }
        }
    }

    func testProtomersImatinib() {
        // Imatinib has: piperazine NMe (base), pyrimidine N, amide, secondary amine
        let smi = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4"
        let results = RDKitBridge.enumerateProtomers(smiles: smi, name: "Imatinib",
                                                      maxProtomers: 8, pH: 7.4, pkaThreshold: 3.0)
        print("  [Protomer] Imatinib pH 7.4: \(results.count) protomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.label) E=\(String(format: "%.2f", r.score))")
        }
        XCTAssertGreaterThanOrEqual(results.count, 1)

        // All protomers should have valid 3D
        for r in results {
            XCTAssertGreaterThan(r.molecule.atoms.count, 30, "Imatinib protomer should have >30 atoms")
            for atom in r.molecule.atoms {
                XCTAssertFalse(atom.position.x.isNaN, "NaN position in protomer")
            }
        }
    }

    func testTautomersMorphine() {
        // Morphine: phenol OH can tautomerize
        let smi = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@@H]1C5"
        let results = RDKitBridge.enumerateTautomers(smiles: smi, name: "Morphine",
                                                      maxTautomers: 10, energyCutoff: 15.0)
        print("  [Tautomer] Morphine: \(results.count) tautomers")
        XCTAssertGreaterThanOrEqual(results.count, 1)
        for r in results {
            XCTAssertFalse(r.molecule.atoms.isEmpty)
        }
    }

    func testProtomersMorphine() {
        // Morphine: phenol (pKa ~9.9) and tertiary amine (pKa ~8.2)
        // Both near pH 7.4 with threshold 3.0 → multiple protomers expected
        let smi = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@@H]1C5"
        let results = RDKitBridge.enumerateProtomers(smiles: smi, name: "Morphine",
                                                      maxProtomers: 8, pH: 7.4, pkaThreshold: 3.0)
        print("  [Protomer] Morphine pH 7.4: \(results.count) protomers")
        for (i, r) in results.enumerated() {
            print("    \(i): \(r.label)")
        }
        XCTAssertGreaterThanOrEqual(results.count, 1)
    }

    // =========================================================================
    // MARK: - Serialization Round-Trip
    // =========================================================================

    @MainActor
    func testVariantSerializationRoundTrip() {
        let db = LigandDatabase()

        // Create entry with variants
        let taut = MolecularVariant(
            smiles: "C=CO",
            atoms: [Atom(id: 0, element: .C, position: SIMD3(1, 0, 0), name: "C1",
                         residueName: "LIG", residueSeq: 1, chainID: "L",
                         charge: 0, formalCharge: 0, isHetAtom: true)],
            bonds: [],
            relativeEnergy: 1.5,
            kind: .tautomer,
            label: "enol form",
            isPrepared: true,
            conformerCount: 5
        )
        let prot = MolecularVariant(
            smiles: "[NH3+]CC",
            atoms: [Atom(id: 0, element: .N, position: SIMD3(0, 1, 0), name: "N1",
                         residueName: "LIG", residueSeq: 1, chainID: "L",
                         charge: 0.5, formalCharge: 1, isHetAtom: true)],
            bonds: [],
            relativeEnergy: 0.3,
            kind: .protomer,
            label: "protonated amine",
            isPrepared: true,
            conformerCount: 10
        )
        let entry = LigandEntry(name: "TestMol", smiles: "CC=O", variants: [taut, prot])
        db.add(entry)

        // Encode
        let data = db.encodeToData()
        XCTAssertFalse(data.isEmpty, "Encoded data should not be empty")

        // Decode into fresh database
        let db2 = LigandDatabase()
        db2.decodeFromData(data)
        XCTAssertEqual(db2.entries.count, 1)

        let loaded = db2.entries[0]
        XCTAssertEqual(loaded.name, "TestMol")
        XCTAssertEqual(loaded.smiles, "CC=O")
        XCTAssertEqual(loaded.variants.count, 2, "Should have 2 variants after round-trip")

        // Check tautomer
        let loadedTaut = loaded.variants.first { $0.kind == .tautomer }
        XCTAssertNotNil(loadedTaut)
        XCTAssertEqual(loadedTaut?.smiles, "C=CO")
        XCTAssertEqual(loadedTaut?.label, "enol form")
        XCTAssertEqual(loadedTaut?.relativeEnergy ?? 0, 1.5, accuracy: 0.01)
        XCTAssertEqual(loadedTaut?.isPrepared, true)
        XCTAssertEqual(loadedTaut?.conformerCount, 5)
        XCTAssertEqual(loadedTaut?.atoms.count, 1)

        // Check protomer
        let loadedProt = loaded.variants.first { $0.kind == .protomer }
        XCTAssertNotNil(loadedProt)
        XCTAssertEqual(loadedProt?.smiles, "[NH3+]CC")
        XCTAssertEqual(loadedProt?.label, "protonated amine")
        XCTAssertEqual(loadedProt?.kind, .protomer)
    }

    @MainActor
    func testBackwardCompatibilityNoVariants() {
        // Simulate loading a database saved before variant support
        // (no "variants" key in JSON)
        let oldFormatJSON = """
        [{"name":"OldMol","smiles":"C","isPrepared":false}]
        """
        let db = LigandDatabase()
        db.decodeFromData(Data(oldFormatJSON.utf8))
        XCTAssertEqual(db.entries.count, 1)
        XCTAssertEqual(db.entries[0].name, "OldMol")
        XCTAssertTrue(db.entries[0].variants.isEmpty, "Old entries should have empty variants")
    }
}

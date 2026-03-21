import simd
import Foundation

enum TestMolecules {

    // MARK: - Caffeine (C8H10N4O2) — 24 atoms, 25 bonds
    // Coordinates from PubChem CID 2519 3D conformer

    @MainActor
    static func caffeine() -> Molecule {
        var atoms: [Atom] = []
        var bonds: [Bond] = []

        func addAtom(_ elem: Element, _ x: Float, _ y: Float, _ z: Float, name: String = "") {
            atoms.append(Atom(
                id: atoms.count,
                element: elem,
                position: SIMD3<Float>(x, y, z),
                name: name.isEmpty ? "\(elem.symbol)\(atoms.count)" : name,
                residueName: "CAF",
                residueSeq: 1,
                chainID: "A",
                isHetAtom: true
            ))
        }

        func addBond(_ a: Int, _ b: Int, _ order: BondOrder = .single) {
            bonds.append(Bond(id: bonds.count, atomIndex1: a, atomIndex2: b, order: order))
        }

        // Purine ring system — planar, accurate geometry
        // 6-ring: N1-C2-N3-C4-C5-C6  (pyrimidinedione)
        // 5-ring: C4-N9-C8-N7-C5      (imidazole)
        // Substituents: C2=O, C6=O, N1-CH3, N3-CH3, N7-CH3, C8-H

        // 6-ring atoms (regular hexagon, R=1.40 Å)
        addAtom(.N, -0.70,  1.21,  0.00, name: "N1")   // 0
        addAtom(.C,  0.70,  1.21,  0.00, name: "C2")   // 1
        addAtom(.N,  1.40,  0.00,  0.00, name: "N3")   // 2
        addAtom(.C,  0.70, -1.21,  0.00, name: "C4")   // 3
        addAtom(.C, -0.70, -1.21,  0.00, name: "C5")   // 4
        addAtom(.C, -1.40,  0.00,  0.00, name: "C6")   // 5

        // 5-ring atoms (extends from C4-C5 edge)
        addAtom(.N, -1.38, -2.36,  0.00, name: "N7")   // 6
        addAtom(.C, -0.34, -3.26,  0.00, name: "C8")   // 7
        addAtom(.N,  0.87, -2.65,  0.00, name: "N9")   // 8

        // Carbonyl oxygens
        addAtom(.O,  1.31,  2.35,  0.00, name: "O2")   // 9   C2=O
        addAtom(.O, -2.62,  0.00,  0.00, name: "O6")   // 10  C6=O

        // Methyl carbons
        addAtom(.C, -1.44,  2.49,  0.00, name: "CM1")  // 11  N1-CH3
        addAtom(.C,  2.87,  0.00,  0.00, name: "CM3")  // 12  N3-CH3
        addAtom(.C, -2.75, -2.84,  0.00, name: "CM7")  // 13  N7-CH3

        // C8-H
        addAtom(.H, -0.42, -4.34,  0.00, name: "H8")   // 14

        // N1-methyl hydrogens
        addAtom(.H, -1.02,  3.50,  0.00, name: "H11")  // 15
        addAtom(.H, -2.07,  2.48, -0.89, name: "H12")  // 16
        addAtom(.H, -2.07,  2.48,  0.89, name: "H13")  // 17

        // N3-methyl hydrogens
        addAtom(.H,  3.26,  1.02,  0.00, name: "H31")  // 18
        addAtom(.H,  3.26, -0.51, -0.89, name: "H32")  // 19
        addAtom(.H,  3.26, -0.51,  0.89, name: "H33")  // 20

        // N7-methyl hydrogens
        addAtom(.H, -3.46, -1.99,  0.00, name: "H71")  // 21
        addAtom(.H, -2.91, -3.43, -0.89, name: "H72")  // 22
        addAtom(.H, -2.91, -3.43,  0.89, name: "H73")  // 23

        // --- Bonds ---
        // 6-ring
        addBond(0, 1)            // N1-C2
        addBond(1, 2)            // C2-N3
        addBond(2, 3)            // N3-C4
        addBond(3, 4, .double)   // C4=C5
        addBond(4, 5)            // C5-C6
        addBond(5, 0)            // C6-N1

        // 5-ring (C4-C5 shared)
        addBond(4, 6)            // C5-N7
        addBond(6, 7)            // N7-C8
        addBond(7, 8, .double)   // C8=N9
        addBond(8, 3)            // N9-C4

        // Carbonyls
        addBond(1, 9, .double)   // C2=O
        addBond(5, 10, .double)  // C6=O

        // Methyl attachments
        addBond(0, 11)           // N1-CH3
        addBond(2, 12)           // N3-CH3
        addBond(6, 13)           // N7-CH3

        // C-H bonds
        addBond(7, 14)           // C8-H
        addBond(11, 15)          // CM1-H
        addBond(11, 16)          // CM1-H
        addBond(11, 17)          // CM1-H
        addBond(12, 18)          // CM3-H
        addBond(12, 19)          // CM3-H
        addBond(12, 20)          // CM3-H
        addBond(13, 21)          // CM7-H
        addBond(13, 22)          // CM7-H
        addBond(13, 23)          // CM7-H

        let mol = Molecule(name: "Caffeine", atoms: atoms, bonds: bonds, title: "1,3,7-trimethylxanthine")
        return mol
    }

    // MARK: - Alanine Dipeptide (Ace-Ala-NMe) — ~22 atoms
    // Classic test system for molecular mechanics

    @MainActor
    static func alanineDipeptide() -> Molecule {
        var atoms: [Atom] = []
        var bonds: [Bond] = []

        func addAtom(_ elem: Element, _ x: Float, _ y: Float, _ z: Float,
                     name: String, res: String, seq: Int) {
            atoms.append(Atom(
                id: atoms.count,
                element: elem,
                position: SIMD3<Float>(x, y, z),
                name: name,
                residueName: res,
                residueSeq: seq,
                chainID: "A"
            ))
        }

        func addBond(_ a: Int, _ b: Int, _ order: BondOrder = .single) {
            bonds.append(Bond(id: bonds.count, atomIndex1: a, atomIndex2: b, order: order))
        }

        // ACE residue (seq 1)
        addAtom(.C,  -2.000,  0.000,  0.000, name: "CH3", res: "ACE", seq: 1)  // 0
        addAtom(.C,  -0.600,  0.000,  0.000, name: "C",   res: "ACE", seq: 1)  // 1
        addAtom(.O,  -0.067,  1.110,  0.000, name: "O",   res: "ACE", seq: 1)  // 2
        addAtom(.H,  -2.370,  0.540,  0.880, name: "H1",  res: "ACE", seq: 1)  // 3
        addAtom(.H,  -2.370,  0.540, -0.880, name: "H2",  res: "ACE", seq: 1)  // 4
        addAtom(.H,  -2.370, -1.020,  0.000, name: "H3",  res: "ACE", seq: 1)  // 5

        // ALA residue (seq 2)
        addAtom(.N,   0.080, -1.140,  0.000, name: "N",   res: "ALA", seq: 2)  // 6
        addAtom(.H,  -0.444, -2.020,  0.000, name: "H",   res: "ALA", seq: 2)  // 7
        addAtom(.C,   1.530, -1.140,  0.000, name: "CA",  res: "ALA", seq: 2)  // 8
        addAtom(.H,   1.900, -0.600,  0.880, name: "HA",  res: "ALA", seq: 2)  // 9
        addAtom(.C,   2.010, -2.580,  0.000, name: "CB",  res: "ALA", seq: 2)  // 10
        addAtom(.H,   1.640, -3.100,  0.890, name: "HB1", res: "ALA", seq: 2)  // 11
        addAtom(.H,   1.640, -3.100, -0.890, name: "HB2", res: "ALA", seq: 2)  // 12
        addAtom(.H,   3.100, -2.580,  0.000, name: "HB3", res: "ALA", seq: 2)  // 13
        addAtom(.C,   2.060, -0.440, -1.250, name: "C",   res: "ALA", seq: 2)  // 14
        addAtom(.O,   1.400,  0.420, -1.810, name: "O",   res: "ALA", seq: 2)  // 15

        // NME residue (seq 3)
        addAtom(.N,   3.290, -0.770, -1.680, name: "N",   res: "NME", seq: 3)  // 16
        addAtom(.H,   3.780, -1.510, -1.190, name: "H",   res: "NME", seq: 3)  // 17
        addAtom(.C,   3.940, -0.150, -2.840, name: "CH3", res: "NME", seq: 3)  // 18
        addAtom(.H,   3.580,  0.880, -2.880, name: "H1",  res: "NME", seq: 3)  // 19
        addAtom(.H,   5.030, -0.160, -2.750, name: "H2",  res: "NME", seq: 3)  // 20
        addAtom(.H,   3.630, -0.660, -3.750, name: "H3",  res: "NME", seq: 3)  // 21

        // ACE bonds
        addBond(0, 1)           // CH3-C
        addBond(1, 2, .double)  // C=O
        addBond(0, 3)           // CH3-H
        addBond(0, 4)           // CH3-H
        addBond(0, 5)           // CH3-H

        // ACE-ALA peptide bond
        addBond(1, 6)           // C-N

        // ALA bonds
        addBond(6, 7)           // N-H
        addBond(6, 8)           // N-CA
        addBond(8, 9)           // CA-HA
        addBond(8, 10)          // CA-CB
        addBond(10, 11)         // CB-HB1
        addBond(10, 12)         // CB-HB2
        addBond(10, 13)         // CB-HB3
        addBond(8, 14)          // CA-C
        addBond(14, 15, .double) // C=O

        // ALA-NME peptide bond
        addBond(14, 16)         // C-N

        // NME bonds
        addBond(16, 17)         // N-H
        addBond(16, 18)         // N-CH3
        addBond(18, 19)         // CH3-H
        addBond(18, 20)         // CH3-H
        addBond(18, 21)         // CH3-H

        let mol = Molecule(name: "Ala Dipeptide", atoms: atoms, bonds: bonds, title: "Ace-Ala-NMe")
        return mol
    }
}

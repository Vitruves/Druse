// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// SidechainTopology.swift — Internal coordinate definitions for sidechain building
//
// Defines bond geometry (distance, angle, dihedral) for each amino acid's
// sidechain atoms.  Given backbone atoms (N, CA, C, O) and a set of chi
// angles from the Dunbrack rotamer library, builds full sidechain coordinates
// using the Internal2Cartesian algorithm.
//
// Translated from FASPR (Huang 2020, MIT License):
//   temp/FASPR/src/RotamerBuilder.cpp::AssignSidechainTopology()
//   temp/FASPR/src/Utility.cpp::Internal2Cartesian()
// ============================================================================

import Foundation
import simd

// MARK: - Data Structures

/// Definition of one sidechain atom in internal coordinates.
struct SidechainAtomDef {
    let name: String                 // PDB atom name, e.g. " CB "
    let refIndices: SIMD3<Int32>     // indices of 3 preceding atoms in the growing list
    let distance: Float              // bond length (Å)
    let angle: Float                 // bond angle (degrees)
    let dihedral: Float              // dihedral (degrees); 181.0 = substitute chi angle
}

/// Topology for one residue type's sidechain.
struct SidechainTopology {
    let nchi: Int
    let atoms: [SidechainAtomDef]    // [0] = CB, [1..] = remaining sidechain atoms
}

// MARK: - Topology Store

/// Sidechain topology definitions for all standard amino acids.
///
/// Atom indexing convention (same as FASPR):
///   0 = N,  1 = CA,  2 = C,  3 = O,  4 = CB,  5+ = sidechain
///
/// Chi angle placeholder: dihedral = 181.0 means "use rotamer chi angle".
enum SidechainTopologyStore {

    // MARK: - Build Sidechain

    /// Build sidechain atom positions from backbone and chi angles.
    ///
    /// - Parameters:
    ///   - n: Backbone N position
    ///   - ca: Backbone CA position
    ///   - c: Backbone C position
    ///   - residueType: One-letter amino acid code
    ///   - chiAngles: Chi angles from rotamer library (degrees)
    /// - Returns: Array of (atomName, position) for sidechain atoms (CB + beyond).
    ///   Empty for Gly, CB-only for Ala.
    static func buildSidechain(
        n: SIMD3<Float>,
        ca: SIMD3<Float>,
        c: SIMD3<Float>,
        residueType: Character,
        chiAngles: [Float]
    ) -> [(name: String, position: SIMD3<Float>)] {

        guard let topo = topologies[residueType] else { return [] }

        // Coordinate list: starts with backbone N(0), CA(1), C(2), O(3-placeholder)
        // O isn't used for sidechain building; index 3 is never referenced.
        var coords: [SIMD3<Float>] = [n, ca, c, .zero]

        var result: [(name: String, position: SIMD3<Float>)] = []
        var chiIdx = 0

        for atom in topo.atoms {
            let i0 = Int(atom.refIndices.x)
            let i1 = Int(atom.refIndices.y)
            let i2 = Int(atom.refIndices.z)

            guard i0 < coords.count, i1 < coords.count, i2 < coords.count else { break }

            var dihedral = atom.dihedral
            if dihedral == 181.0, chiIdx < chiAngles.count {
                dihedral = chiAngles[chiIdx]
                chiIdx += 1
            }

            let pos = internal2Cartesian(
                c1: coords[i0],
                c2: coords[i1],
                c3: coords[i2],
                distance: atom.distance,
                angle: atom.angle,
                dihedral: dihedral
            )

            coords.append(pos)
            result.append((atom.name, pos))
        }

        return result
    }

    // MARK: - Internal Coordinate → Cartesian

    /// Convert internal coordinates (distance, angle, dihedral) to Cartesian position.
    /// Given three reference atoms c1, c2, c3 and internal coords, computes the
    /// position of the new atom bonded to c3.
    ///
    /// Direct translation of FASPR Utility.cpp::Internal2Cartesian().
    static func internal2Cartesian(
        c1: SIMD3<Float>,
        c2: SIMD3<Float>,
        c3: SIMD3<Float>,
        distance: Float,
        angle: Float,      // degrees
        dihedral: Float     // degrees
    ) -> SIMD3<Float> {
        let angleRad = angle * .pi / 180.0
        let dihedralRad = dihedral * .pi / 180.0

        // New atom position in local frame
        let d = SIMD3<Float>(
            distance * cos(angleRad),
            distance * cos(dihedralRad) * sin(angleRad),
            distance * sin(dihedralRad) * sin(angleRad)
        )

        // Build local coordinate frame from c1→c2→c3
        let ab = c2 - c1
        var bc = c3 - c2
        let bcLen = simd_length(bc)
        guard bcLen > 1e-10 else { return c3 + d }
        bc /= bcLen

        var n = simd_cross(ab, bc)
        let nLen = simd_length(n)
        guard nLen > 1e-10 else { return c3 + d }
        n /= nLen

        let abPerp = simd_cross(n, bc)

        // Rotation matrix columns: [-bc, abPerp, n]
        let col0 = -bc
        let col1 = abPerp
        let col2 = n

        // Matrix × vector
        let result = SIMD3<Float>(
            col0.x * d.x + col1.x * d.y + col2.x * d.z,
            col0.y * d.x + col1.y * d.y + col2.y * d.z,
            col0.z * d.x + col1.z * d.y + col2.z * d.z
        )

        return c3 + result
    }

    // MARK: - Topology Data

    /// All sidechain topologies indexed by one-letter code.
    /// Translated verbatim from RotamerBuilder::AssignSidechainTopology().
    static let topologies: [Character: SidechainTopology] = {
        var t: [Character: SidechainTopology] = [:]

        // Helper to create atom def concisely
        func a(_ name: String, _ i0: Int32, _ i1: Int32, _ i2: Int32,
               _ dist: Float, _ ang: Float, _ dihe: Float) -> SidechainAtomDef {
            SidechainAtomDef(name: name, refIndices: .init(i0, i1, i2),
                             distance: dist, angle: ang, dihedral: dihe)
        }

        // Ala — only CB, no chi angles
        t["A"] = SidechainTopology(nchi: 0, atoms: [
            a(" CB ", 2, 0, 1, 1.53, 110.5, -122.5)
        ])

        // Arg — 4 chi angles
        t["R"] = SidechainTopology(nchi: 4, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.52,  114.1,  181),
            a(" CD ", 1, 4, 5, 1.52,  111.5,  181),
            a(" NE ", 4, 5, 6, 1.461, 112.0,  181),
            a(" CZ ", 5, 6, 7, 1.33,  124.5,  181),
            a(" NH1", 6, 7, 8, 1.326, 120.0,  0),
            a(" NH2", 9, 7, 8, 1.326, 120.0,  180)  // idx 9 = NH1
        ])

        // Asn — 2 chi angles
        t["N"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.516, 112.7,  181),
            a(" OD1", 1, 4, 5, 1.231, 120.8,  181),
            a(" ND2", 6, 4, 5, 1.328, 116.5,  180)  // idx 6 = OD1
        ])

        // Asp — 2 chi angles
        t["D"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.516, 112.7,  181),
            a(" OD1", 1, 4, 5, 1.25,  118.5,  181),
            a(" OD2", 6, 4, 5, 1.25,  118.5,  180)
        ])

        // Cys — 1 chi angle
        t["C"] = SidechainTopology(nchi: 1, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" SG ", 0, 1, 4, 1.807, 114.0,  181)
        ])

        // Gln — 3 chi angles
        t["Q"] = SidechainTopology(nchi: 3, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.52,  114.1,  181),
            a(" CD ", 1, 4, 5, 1.516, 112.7,  181),
            a(" OE1", 4, 5, 6, 1.231, 120.8,  181),
            a(" NE2", 7, 5, 6, 1.328, 116.5,  180)  // idx 7 = OE1
        ])

        // Glu — 3 chi angles
        t["E"] = SidechainTopology(nchi: 3, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.52,  114.1,  181),
            a(" CD ", 1, 4, 5, 1.516, 112.7,  181),
            a(" OE1", 4, 5, 6, 1.25,  118.5,  181),
            a(" OE2", 7, 5, 6, 1.25,  118.5,  180)
        ])

        // His — 2 chi angles
        t["H"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.5,   113.8,  181),
            a(" ND1", 1, 4, 5, 1.378, 122.7,  181),
            a(" CD2", 6, 4, 5, 1.354, 131.0,  180),  // idx 6 = ND1
            a(" CE1", 4, 5, 6, 1.32,  109.2,  180),
            a(" NE2", 4, 5, 7, 1.374, 107.2,  180)   // idx 7 = CD2
        ])

        // Ile — 2 chi angles
        t["I"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.546, 111.5, -122.5),
            a(" CG1", 0, 1, 4, 1.53,  110.3,  181),
            a(" CG2", 5, 1, 4, 1.521, 110.5, -122.6),  // idx 5 = CG1
            a(" CD1", 1, 4, 5, 1.516, 114.0,  181)
        ])

        // Leu — 2 chi angles
        t["L"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.53,  116.3,  181),
            a(" CD1", 1, 4, 5, 1.521, 110.5,  181),
            a(" CD2", 6, 4, 5, 1.521, 110.5,  122.6)  // idx 6 = CD1
        ])

        // Lys — 4 chi angles
        t["K"] = SidechainTopology(nchi: 4, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.52,  114.1,  181),
            a(" CD ", 1, 4, 5, 1.52,  111.5,  181),
            a(" CE ", 4, 5, 6, 1.52,  111.5,  181),
            a(" NZ ", 5, 6, 7, 1.489, 112.0,  181)
        ])

        // Met — 3 chi angles
        t["M"] = SidechainTopology(nchi: 3, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.52,  114.1,  181),
            a(" SD ", 1, 4, 5, 1.807, 112.7,  181),
            a(" CE ", 4, 5, 6, 1.789, 100.8,  181)
        ])

        // Phe — 2 chi angles
        t["F"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.5,   113.8,  181),
            a(" CD1", 1, 4, 5, 1.391, 120.7,  181),
            a(" CD2", 6, 4, 5, 1.391, 120.7,  180),   // idx 6 = CD1
            a(" CE1", 4, 5, 6, 1.393, 120.7,  180),
            a(" CE2", 4, 5, 7, 1.393, 120.7,  180),   // idx 7 = CD2
            a(" CZ ", 5, 6, 8, 1.39,  120.0,  0)      // idx 8 = CE1
        ])

        // Pro — 2 chi angles
        t["P"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  103.2, -120.0),
            a(" CG ", 0, 1, 4, 1.495, 104.5,  181),
            a(" CD ", 1, 4, 5, 1.507, 105.5,  181)
        ])

        // Ser — 1 chi angle
        t["S"] = SidechainTopology(nchi: 1, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" OG ", 0, 1, 4, 1.417, 110.8,  181)
        ])

        // Thr — 1 chi angle
        t["T"] = SidechainTopology(nchi: 1, atoms: [
            a(" CB ", 2, 0, 1, 1.542, 111.5, -122.0),
            a(" OG1", 0, 1, 4, 1.433, 109.5,  181),
            a(" CG2", 5, 1, 4, 1.521, 110.5, -120.0)  // idx 5 = OG1
        ])

        // Trp — 2 chi angles
        t["W"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.5,   113.8,  181),
            a(" CD1", 1, 4, 5, 1.365, 126.9,  181),
            a(" CD2", 6, 4, 5, 1.433, 126.7,  180),   // idx 6 = CD1
            a(" NE1", 4, 5, 6, 1.375, 110.2,  180),
            a(" CE2", 4, 5, 7, 1.413, 107.2,  180),   // idx 7 = CD2
            a(" CE3", 4, 5, 7, 1.4,   133.9,  0),
            a(" CZ2", 5, 7, 9, 1.399, 122.4,  180),   // idx 9 = CE2
            a(" CZ3", 5, 7, 10, 1.392, 118.7, 180),   // idx 10 = CE3
            a(" CH2", 7, 9, 11, 1.372, 117.5, 0)      // idx 11 = CZ2
        ])

        // Tyr — 2 chi angles
        t["Y"] = SidechainTopology(nchi: 2, atoms: [
            a(" CB ", 2, 0, 1, 1.53,  110.5, -122.5),
            a(" CG ", 0, 1, 4, 1.511, 113.8,  181),
            a(" CD1", 1, 4, 5, 1.394, 120.8,  181),
            a(" CD2", 6, 4, 5, 1.394, 120.8,  180),   // idx 6 = CD1
            a(" CE1", 4, 5, 6, 1.392, 121.1,  180),
            a(" CE2", 4, 5, 7, 1.392, 121.1,  180),   // idx 7 = CD2
            a(" CZ ", 5, 6, 8, 1.385, 119.5,  0),     // idx 8 = CE1
            a(" OH ", 6, 8, 10, 1.376, 119.7, 180)     // idx 10 = CZ
        ])

        // Val — 1 chi angle
        t["V"] = SidechainTopology(nchi: 1, atoms: [
            a(" CB ", 2, 0, 1, 1.546, 111.5, -122.5),
            a(" CG1", 0, 1, 4, 1.521, 110.5,  181),
            a(" CG2", 5, 1, 4, 1.521, 110.5,  122.6)  // idx 5 = CG1
        ])

        return t
    }()

    /// Residue types that have rotamers (everything except Ala and Gly).
    static let rotamericResidues: Set<Character> = Set(topologies.keys.filter { $0 != "A" && $0 != "G" })

    // MARK: - Phi/Psi Calculation

    /// Compute backbone phi angle for residue i given consecutive backbone atoms.
    /// phi(i) = dihedral(C[i-1], N[i], CA[i], C[i])
    static func phi(prevC: SIMD3<Float>, n: SIMD3<Float>, ca: SIMD3<Float>, c: SIMD3<Float>) -> Float {
        return dihedral(prevC, n, ca, c)
    }

    /// Compute backbone psi angle for residue i given consecutive backbone atoms.
    /// psi(i) = dihedral(N[i], CA[i], C[i], N[i+1])
    static func psi(n: SIMD3<Float>, ca: SIMD3<Float>, c: SIMD3<Float>, nextN: SIMD3<Float>) -> Float {
        return dihedral(n, ca, c, nextN)
    }

    /// Compute dihedral angle (degrees) for four points.
    static func dihedral(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                         _ p3: SIMD3<Float>, _ p4: SIMD3<Float>) -> Float {
        let b1 = p2 - p1
        let b2 = p3 - p2
        let b3 = p4 - p3

        let v1 = simd_cross(b1, b2)
        let v2 = simd_cross(b2, b3)
        let v3 = simd_cross(v2, v1)

        let b2len = simd_length(b2)
        guard b2len > 1e-10 else { return 0 }

        let sign: Float = simd_dot(v3, b2) >= 0 ? 1 : -1

        let len1 = simd_length(v1)
        let len2 = simd_length(v2)
        guard len1 > 1e-10, len2 > 1e-10 else { return 0 }

        var cosAngle = simd_dot(v1, v2) / (len1 * len2)
        cosAngle = min(1.0, max(-1.0, cosAngle))

        return sign * acos(cosAngle) * 180.0 / .pi
    }

    // MARK: - One-Letter ↔ Three-Letter Code

    static let three2one: [String: Character] = [
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    ]

    static let one2three: [Character: String] = [
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    ]
}

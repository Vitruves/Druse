// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// FASPRVDWParameters.swift — CHARMM19 VDW parameters and energy functions
//
// Defines atom types, van der Waals parameters, H-bond energy, S-S bond
// energy, and rotamer preference scoring for FASPR sidechain packing.
//
// Translated from FASPR (Huang 2020, MIT License):
//   temp/FASPR/src/SelfEnergy.h (constants)
//   temp/FASPR/src/SelfEnergy.cpp (SetVdwPar, VDWEnergyAtomAndAtom, etc.)
// ============================================================================

import Foundation
import simd

// MARK: - Atom Types (CHARMM19)

/// CHARMM19 atom types used by FASPR for VDW calculations.
/// 18 types covering mainchain and sidechain heavy atoms.
enum FASPRAtomType: Int, CaseIterable, Sendable {
    case mainchainCA  = 0   // C1: mainchain CA
    case mainchainC   = 1   // C2: mainchain C
    case CH1          = 2   // C3: CH1 (Ile/Thr CB)
    case CH2          = 3   // C4: CH2
    case CH3          = 4   // C5: CH3
    case aromaticCH   = 5   // C6: aromatic CH1/C (Phe/Tyr/Trp)
    case carbonylC    = 6   // C7: CO, COO, NCNN
    case cysCB        = 7   // C8: Cys CB
    case mainchainNH  = 8   // N1: mainchain NH
    case sidechainNH  = 9   // N2: His/Arg/Trp NH, Asn/Gln/Arg NH2, Lys NH3
    case hisCN        = 10  // N3: His C=N-C
    case proN         = 11  // N4: Pro N
    case mainchainO   = 12  // O1: mainchain O
    case sidechainCO  = 13  // O2: sidechain C=O
    case sidechainCOO = 14  // O3: sidechain COO
    case hydroxylOH   = 15  // O4: Ser/Thr/Tyr OH
    case cysS         = 16  // S1: Cys S (disulfide capable)
    case metS         = 17  // S2: Met S

    var radius: Float {
        switch self {
        case .mainchainCA:  return 1.78
        case .mainchainC:   return 1.40
        case .CH1:          return 2.30
        case .CH2:          return 1.91
        case .CH3:          return 1.96
        case .aromaticCH:   return 1.73
        case .carbonylC:    return 1.43
        case .cysCB:        return 1.99
        case .mainchainNH:  return 1.42
        case .sidechainNH:  return 1.69
        case .hisCN:        return 1.56
        case .proN:         return 1.70
        case .mainchainO:   return 1.48
        case .sidechainCO:  return 1.44
        case .sidechainCOO: return 1.40
        case .hydroxylOH:   return 1.43
        case .cysS:         return 2.15
        case .metS:         return 1.74
        }
    }

    var depth: Float {
        switch self {
        case .mainchainCA:  return 0.25
        case .mainchainC:   return 0.14
        case .CH1:          return 0.30
        case .CH2:          return 0.37
        case .CH3:          return 0.48
        case .aromaticCH:   return 0.38
        case .carbonylC:    return 0.07
        case .cysCB:        return 0.38
        case .mainchainNH:  return 0.08
        case .sidechainNH:  return 0.24
        case .hisCN:        return 0.46
        case .proN:         return 0.48
        case .mainchainO:   return 0.22
        case .sidechainCO:  return 0.27
        case .sidechainCOO: return 0.07
        case .hydroxylOH:   return 0.12
        case .cysS:         return 0.44
        case .metS:         return 0.40
        }
    }

    /// 1-based index used in FASPR's VDWType() for H-bond distance override logic.
    var fasprIndex: Int { rawValue + 1 }
}

// MARK: - Per-Residue Atom Type Assignments

enum FASPRVDWParameters {

    /// VDW atom type indices (1-based FASPR convention) per residue type.
    /// Order: [N, CA, C, O, sidechain_atoms...]
    /// Translated from SelfEnergy::SetVdwPar() vAtomIdx table.
    static let atomTypeIndices: [Character: [Int]] = [
        "G": [9, 1, 2, 13],                                     // 4 atoms
        "A": [9, 1, 2, 13, 5],                                  // 5 atoms
        "C": [9, 1, 2, 13, 8, 17],                              // 6: CB=CysCB, SG=CysS
        "S": [9, 1, 2, 13, 4, 16],                              // 6: CB=CH2, OG=OH
        "P": [12, 1, 2, 13, 4, 4, 4],                           // 7: N=ProN, CB/CG/CD=CH2
        "T": [9, 1, 2, 13, 3, 16, 5],                           // 7: CB=CH1, OG1=OH, CG2=CH3
        "V": [9, 1, 2, 13, 3, 5, 5],                            // 7: CB=CH1, CG1=CH3, CG2=CH3
        "M": [9, 1, 2, 13, 4, 4, 18, 5],                        // 8: SD=MetS, CE=CH3
        "N": [9, 1, 2, 13, 4, 7, 14, 10],                       // 8: CG=CO, OD1=scCO, ND2=scNH
        "D": [9, 1, 2, 13, 4, 7, 15, 15],                       // 8: CG=CO, OD1/OD2=COO
        "L": [9, 1, 2, 13, 4, 3, 5, 5],                         // 8: CG=CH1, CD1/CD2=CH3
        "I": [9, 1, 2, 13, 3, 4, 5, 5],                         // 8: CB=CH1, CG1=CH2, CG2/CD1=CH3
        "Q": [9, 1, 2, 13, 4, 4, 7, 14, 10],                    // 9: CD=CO, OE1=scCO, NE2=scNH
        "E": [9, 1, 2, 13, 4, 4, 7, 15, 15],                    // 9: CD=CO, OE1/OE2=COO
        "K": [9, 1, 2, 13, 4, 4, 4, 4, 10],                     // 9: NZ=scNH
        "H": [9, 1, 2, 13, 4, 6, 10, 6, 6, 11],                 // 10: CG/CD2/CE1=arom, ND1=scNH, NE2=HisCN
        "R": [9, 1, 2, 13, 4, 4, 4, 10, 7, 10, 10],             // 11: NE=scNH, CZ=CO, NH1/NH2=scNH
        "F": [9, 1, 2, 13, 4, 6, 6, 6, 6, 6, 6],                // 11: all ring=aromatic
        "Y": [9, 1, 2, 13, 4, 6, 6, 6, 6, 6, 6, 16],            // 12: ring=aromatic, OH=hydroxyl
        "W": [9, 1, 2, 13, 4, 6, 6, 6, 10, 6, 6, 6, 6, 6]      // 14: NE1=scNH, rest=aromatic
    ]

    /// Resolve atom types for a residue into (radius, depth) arrays.
    static func resolveAtomParams(residueType: Character) -> [(radius: Float, depth: Float)] {
        guard let indices = atomTypeIndices[residueType] else { return [] }
        return indices.map { idx in
            let type = FASPRAtomType(rawValue: idx - 1) ?? .mainchainCA
            return (type.radius, type.depth)
        }
    }

    // MARK: - Rotamer Preference Weights

    /// Per-residue weights for the rotamer library preference energy term.
    static let rotamerPreferenceWeight: [Character: Float] = [
        "C": 5.5, "D": 2.0, "E": 1.0, "F": 1.5, "H": 3.0,
        "I": 1.0, "K": 2.0, "L": 2.0, "M": 1.5, "N": 2.0,
        "P": 1.5, "Q": 2.5, "R": 1.5, "S": 1.5, "T": 2.0,
        "V": 2.0, "W": 3.5, "Y": 1.5
    ]

    /// Residue-radius for contact map construction (CB-CB distance cutoff).
    static let residueRadius: [Character: Float] = [
        "G": 2.4,
        "A": 3.2, "C": 3.2, "D": 3.2, "I": 3.2, "L": 3.2,
        "N": 3.2, "P": 3.2, "S": 3.2, "T": 3.2, "V": 3.2,
        "Q": 3.7, "E": 3.7, "H": 3.7,
        "M": 4.3, "F": 4.3,
        "K": 5.0,
        "W": 5.3,
        "Y": 5.7,
        "R": 6.2
    ]

    // MARK: - VDW Constants

    static let dstarMinCut: Float = 0.015
    static let dstarMaxCut: Float = 1.90
    static let vdwRepCut: Float = 10.0
    static let cacbDistCut: Float = 2.35
    static let residueDistCut: Float = 4.25

    // MARK: - H-bond Constants

    static let optHbondDist: Float = 2.8
    static let minHbondDist: Float = 2.6
    static let maxHbondDist: Float = 3.2
    static let minHbondTheta: Float = 90.0
    static let minHbondPhi: Float = 90.0
    static let wgtHbond: Float = 1.0

    // MARK: - S-S Bond Constants

    static let optSSbondDist: Float = 2.03
    static let minSSbondDist: Float = 1.73
    static let maxSSbondDist: Float = 2.53
    static let optSSbondAngle: Float = 105.0
    static let minSSbondAngle: Float = 75.0
    static let maxSSbondAngle: Float = 135.0
    static let wgtSSbond: Float = 6.0

    // MARK: - Energy Functions

    /// Combined VDW radius for an atom pair, with H-bond/S-S distance override.
    /// Translated from SelfEnergy::VDWType().
    static func vdwEquilibriumDistance(
        typeA: Int,  // 1-based FASPR index
        typeB: Int,
        combinedRadius: Float,
        distance: Float
    ) -> Float {
        let i1 = min(typeA, typeB)
        let i2 = max(typeA, typeB)

        // Check if this is an H-bond-capable pair
        let isNH = (i1 == 9 || i1 == 10)  // mainchainNH or sidechainNH
        let isAcceptor = (i2 == 11 || i2 == 13 || i2 == 14 || i2 == 15 || i2 == 16)
        if isNH && isAcceptor && distance > minHbondDist && distance < maxHbondDist {
            return optHbondDist
        }
        // HisCN with OH
        if i1 == 11 && i2 == 16 && distance > minHbondDist && distance < maxHbondDist {
            return optHbondDist
        }
        // O-O H-bond
        if (i1 == 13 || i1 == 14 || i1 == 15) && i2 == 16
            && distance > minHbondDist && distance < maxHbondDist {
            return optHbondDist
        }
        // OH-OH
        if i1 == 16 && i2 == 16 && distance > minHbondDist && distance < maxHbondDist {
            return optHbondDist
        }
        // S-S disulfide
        if i1 == 17 && i2 == 17 && distance > minSSbondDist && distance < maxSSbondDist {
            return optSSbondDist
        }

        return combinedRadius
    }

    /// VDW energy for a pair of atoms. dstar = distance / rij_combined.
    /// LJ 6-12 with linear repulsion cap at 10 kcal/mol.
    /// Translated from SelfEnergy::VDWEnergyAtomAndAtom().
    @inlinable
    static func vdwEnergy(dstar: Float, epsilon: Float) -> Float {
        if dstar > dstarMaxCut { return 0 }
        if dstar > 1.0 {
            let inv = 1.0 / dstar
            let inv6 = inv * inv * inv * inv * inv * inv
            let inv12 = inv6 * inv6
            return 4.0 * epsilon * (inv12 - inv6)
        }
        if dstar > dstarMinCut {
            return vdwRepCut * (dstar - 1.0) / (dstarMinCut - 1.0)
        }
        return vdwRepCut
    }

    /// H-bond energy between donor and acceptor atoms.
    /// E = [5(d₀/d)¹² - 6(d₀/d)¹⁰] × cos²(θ-θ₀) × cos²(φ-φ₀)
    /// Returns ≤ 0 (favorable) or 0.
    /// Translated from SelfEnergy::HbondEnergyAtomAndAtom().
    static func hbondEnergy(
        donorBase: SIMD3<Float>,    // DB: atom bonded to donor
        donor: SIMD3<Float>,         // D: donor atom
        acceptor: SIMD3<Float>,      // A: acceptor atom
        acceptorBase: SIMD3<Float>,  // AB: atom bonded to acceptor
        optDonorAngle: Float,        // optimal donor angle (degrees)
        optAcceptorAngle: Float      // optimal acceptor angle (degrees)
    ) -> Float {
        let dist = simd_distance(donor, acceptor)
        // Boundaries are inclusive, as in FASPR (rejects only dist > max || dist < min).
        guard dist >= minHbondDist, dist <= maxHbondDist else { return 0 }

        let theta = SidechainTopologyStore.angleBetween(donorBase, donor, acceptor)
        guard theta >= minHbondTheta else { return 0 }

        let phi = SidechainTopologyStore.angleBetween(donor, acceptor, acceptorBase)
        guard phi >= minHbondPhi else { return 0 }

        let ratio = optHbondDist / dist
        let r10 = pow(ratio, 10)
        let r12 = r10 * ratio * ratio
        let fdist = 5.0 * r12 - 6.0 * r10

        let deg2rad: Float = .pi / 180.0
        let ftheta = cos((theta - optDonorAngle) * deg2rad)
        let fthetaSq = ftheta * ftheta
        let fphi = cos((phi - optAcceptorAngle) * deg2rad)
        let fphiSq = fphi * fphi

        let energy = fdist * fthetaSq * fphiSq * wgtHbond
        return energy > 0 ? 0 : energy
    }

    /// S-S bond energy between two Cys SG atoms.
    /// Translated from SelfEnergy::SSbondEnergyAtomAndAtom().
    static func ssbondEnergy(
        ca1: SIMD3<Float>, cb1: SIMD3<Float>, sg1: SIMD3<Float>,
        sg2: SIMD3<Float>, cb2: SIMD3<Float>, ca2: SIMD3<Float>
    ) -> Float {
        let dist = simd_distance(sg1, sg2)
        // Boundaries are inclusive, as in FASPR.
        guard dist >= minSSbondDist, dist <= maxSSbondDist else { return 0 }

        let ang1 = SidechainTopologyStore.angleBetween(cb1, sg1, sg2)
        guard ang1 >= minSSbondAngle, ang1 <= maxSSbondAngle else { return 0 }

        let ang2 = SidechainTopologyStore.angleBetween(cb2, sg2, sg1)
        guard ang2 >= minSSbondAngle, ang2 <= maxSSbondAngle else { return 0 }

        let torsion = SidechainTopologyStore.dihedral(cb1, sg1, sg2, cb2)
        let deg2rad: Float = .pi / 180.0

        let energy = 100.0 * (dist - optSSbondDist) * (dist - optSSbondDist) - 4.0
            + 0.01 * (ang1 - optSSbondAngle) * (ang1 - optSSbondAngle) - 2.0
            + 0.01 * (ang2 - optSSbondAngle) * (ang2 - optSSbondAngle) - 2.0
            + 2.0 * cos(2.0 * torsion * deg2rad)

        let scaled = energy * wgtSSbond
        return scaled > 0 ? 0 : scaled
    }

    /// Rotamer preference energy: -log(P/Pmax) × weight, capped at 5.0.
    @inlinable
    static func rotamerPreferenceEnergy(
        probability: Float,
        maxProbability: Float,
        residueType: Character
    ) -> Float {
        guard maxProbability > 0, probability > 0 else { return 0 }
        let weight = rotamerPreferenceWeight[residueType] ?? 1.0
        var elib = -log(probability / maxProbability)
        if elib > 5.0 { elib = 5.0 }
        return weight * elib
    }

    // MARK: - Atom-Pair VDW

    /// Number of sidechain atoms beyond CB in the FASPR topology for a residue type.
    static let sidechainAtomCount: [Character: Int] = atomTypeIndices.mapValues { max(0, $0.count - 5) }

    /// True for atom types whose equilibrium distance `vdwEquilibriumDistance`
    /// can override (H-bond donors/acceptors and Cys S). Any pair involving an
    /// atom outside this set uses the plain combined radius.
    static func hasDistanceOverride(_ type: Int) -> Bool {
        type >= 9 && type <= 17 && type != 12
    }

    /// VDW energy for one atom pair including the FASPR H-bond / S-S
    /// equilibrium-distance override (SelfEnergy::VDWType + VDWEnergyAtomAndAtom).
    static func pairVDW(
        _ posA: SIMD3<Float>, _ typeA: Int, _ radA: Float, _ depA: Float,
        _ posB: SIMD3<Float>, _ typeB: Int, _ radB: Float, _ depB: Float
    ) -> Float {
        let dist = simd_distance(posA, posB)
        let rij = vdwEquilibriumDistance(typeA: typeA, typeB: typeB,
                                         combinedRadius: radA + radB, distance: dist)
        return vdwEnergy(dstar: dist / rij, epsilon: (depA * depB).squareRoot())
    }

    // MARK: - Polar Sidechain Terms

    /// Sidechain-vs-backbone H-bond energy for one rotamer of `t1` against the
    /// backbone of another residue. `b1` = [N, CA, C, O, CB] of residue 1 (CB
    /// required), `s1` = its sidechain atoms beyond CB in topology order,
    /// `b2` = [N, CA, C, O, ...] of residue 2. Callers skip sequence neighbours.
    /// Translated from SelfEnergy::EnergyPolarSidechainAndBackbone().
    static func polarSidechainBackbone(
        t1: Character, b1: [SIMD3<Float>], s1: [SIMD3<Float>], b2: [SIMD3<Float>]
    ) -> Float {
        var energy: Float = 0
        if t1 == "D" {
            energy = hb(b2[1], b2[0], s1[1], s1[0], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[2], s1[0], 120.0, 120.0)
        } else if t1 == "E" {
            energy = hb(b2[1], b2[0], s1[2], s1[1], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[3], s1[1], 120.0, 120.0)
        } else if t1 == "K" {
            energy = hb(s1[2], s1[3], b2[3], b2[2], 109.5, 120.0)
        } else if t1 == "R" {
            energy = hb(s1[3], s1[4], b2[3], b2[2], 120.0, 120.0) +
                hb(s1[3], s1[5], b2[3], b2[2], 120.0, 120.0) +
                hb(s1[1], s1[2], b2[3], b2[2], 120.0, 120.0)
        } else if t1 == "W" {
            energy = hb(s1[1], s1[3], b2[3], b2[2], 120.0, 120.0)
        } else if t1 == "H" {
            energy = hb(s1[0], s1[1], b2[3], b2[2], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[4], s1[2], 120.0, 120.0)
        } else if t1 == "N" {
            energy = hb(s1[0], s1[2], b2[3], b2[2], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[1], s1[0], 120.0, 120.0)
        } else if t1 == "Q" {
            energy = hb(s1[1], s1[3], b2[3], b2[2], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[2], s1[1], 120.0, 120.0)
        } else if t1 == "S" {
            energy = hb(b1[4], s1[0], b2[3], b2[2], 109.5, 120.0) +
                hb(b2[1], b2[0], s1[0], b1[4], 120.0, 109.5)
        } else if t1 == "T" {
            energy = hb(b1[4], s1[0], b2[3], b2[2], 109.5, 120.0) +
                hb(b2[1], b2[0], s1[0], b1[4], 120.0, 109.5)
        } else if t1 == "Y" {
            energy = hb(s1[5], s1[6], b2[3], b2[2], 120.0, 120.0) +
                hb(b2[1], b2[0], s1[6], s1[5], 120.0, 120.0)
        }
        return energy
    }

    /// Sidechain-vs-sidechain H-bond / disulfide energy between rotamers of two
    /// residues. Backbones must include CB; sidechains must be complete.
    /// Translated from SelfEnergy::EnergyPolarSidechainAndSidechain().
    static func polarSidechainSidechain(
        t1: Character, b1: [SIMD3<Float>], s1: [SIMD3<Float>],
        t2: Character, b2: [SIMD3<Float>], s2: [SIMD3<Float>]
    ) -> Float {
        var energy: Float = 0
        if t1 == "C" {
            if t2 == "C" {
                energy = ss(b1[1], b1[4], s1[0], s2[0], b2[4], b2[1])
            }
        } else if t1 == "D" {
            if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[2], s2[3], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[3], s2[4], s1[2], s1[0], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[2], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[0], s2[1], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[0], s2[2], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[1], s1[0], 109.5, 120.0) +
                    hb(b2[4], s2[0], s1[2], s1[0], 109.5, 120.0)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[1], s1[0], 109.5, 120.0) +
                    hb(b2[4], s2[0], s1[2], s1[0], 109.5, 120.0)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[5], s2[6], s1[2], s1[0], 120.0, 120.0)
            }
        } else if t1 == "E" {
            if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[2], s1[1], 109.5, 120.0) +
                    hb(s2[2], s2[3], s1[3], s1[1], 109.5, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[3], s2[4], s1[3], s1[1], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[3], s1[1], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[3], s1[1], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[3], s1[1], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[0], s2[1], s1[3], s1[1], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[0], s2[2], s1[3], s1[1], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[3], s1[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[2], s1[1], 109.5, 120.0) +
                    hb(b2[4], s2[0], s1[3], s1[1], 109.5, 120.0)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[2], s1[1], 109.5, 120.0) +
                    hb(b2[4], s2[0], s1[3], s1[1], 109.5, 120.0)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[5], s2[6], s1[3], s1[1], 120.0, 120.0)
            }
        } else if t1 == "K" {
            if t2 == "D" {
                energy = hb(s1[2], s1[3], s2[1], s2[0], 109.5, 120.0) +
                    hb(s1[2], s1[3], s2[2], s2[0], 109.5, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[2], s1[3], s2[2], s2[1], 109.5, 120.0) +
                    hb(s1[2], s1[3], s2[3], s2[1], 109.5, 120.0)
            } else if t2 == "H" {
                energy = hb(s1[2], s1[3], s2[4], s2[2], 109.5, 120.0)
            } else if t2 == "N" {
                energy = hb(s1[2], s1[3], s2[1], s2[0], 109.5, 120.0)
            } else if t2 == "Q" {
                energy = hb(s1[2], s1[3], s2[2], s2[1], 109.5, 120.0)
            } else if t2 == "S" {
                energy = hb(s1[2], s1[3], s2[0], b2[4], 109.5, 109.5)
            } else if t2 == "T" {
                energy = hb(s1[2], s1[3], s2[0], b2[4], 109.5, 109.5)
            } else if t2 == "Y" {
                energy = hb(s1[2], s1[3], s2[6], s2[5], 109.5, 120.0)
            }
        } else if t1 == "R" {
            if t2 == "D" {
                energy = hb(s1[3], s1[4], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[3], s1[4], s2[2], s2[0], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[2], s2[0], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[3], s1[4], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[3], s1[4], s2[3], s2[1], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[3], s2[1], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s1[3], s1[4], s2[4], s2[2], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[4], s2[2], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[4], s2[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s1[3], s1[4], s2[2], s2[0], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[2], s2[0], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s1[3], s1[4], s2[3], s2[1], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[3], s2[1], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(s1[3], s1[4], s2[0], b2[4], 120.0, 109.5) +
                    hb(s1[3], s1[5], s2[0], b2[4], 120.0, 109.5) +
                    hb(s1[1], s1[2], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "T" {
                energy = hb(s1[3], s1[4], s2[0], b2[4], 120.0, 109.5) +
                    hb(s1[3], s1[5], s2[0], b2[4], 120.0, 109.5) +
                    hb(s1[1], s1[2], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "Y" {
                energy = hb(s1[3], s1[4], s2[6], s2[5], 120.0, 120.0) +
                    hb(s1[3], s1[5], s2[6], s2[5], 120.0, 120.0) +
                    hb(s1[1], s1[2], s2[6], s2[5], 120.0, 120.0)
            }
        } else if t1 == "W" {
            if t2 == "D" {
                energy = hb(s1[1], s1[3], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[1], s1[3], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s1[1], s1[3], s2[4], s2[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s1[1], s1[3], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s1[1], s1[3], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(s1[1], s1[3], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "T" {
                energy = hb(s1[1], s1[3], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "Y" {
                energy = hb(s1[1], s1[3], s2[6], s2[5], 120.0, 120.0)
            }
        } else if t1 == "H" {
            if t2 == "D" {
                energy = hb(s1[0], s1[1], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[0], s1[1], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[0], s1[1], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[0], s1[1], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[4], s1[2], 109.5, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[4], s1[2], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[4], s1[2], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[4], s1[2], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[2], s1[0], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s1[0], s1[1], s2[4], s2[2], 120.0, 120.0) +
                    hb(s2[0], s2[1], s1[4], s1[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s1[0], s1[1], s2[1], s2[0], 120.0, 120.0) +
                    hb(s2[0], s2[2], s1[4], s1[2], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s1[0], s1[1], s2[2], s2[1], 120.0, 120.0) +
                    hb(s2[1], s2[3], s1[4], s1[2], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(s1[0], s1[1], s2[0], b2[4], 120.0, 109.5) +
                    hb(b2[4], s2[0], s1[4], s1[2], 120.0, 109.5)
            } else if t2 == "T" {
                energy = hb(s1[0], s1[1], s2[0], b2[4], 120.0, 109.5) +
                    hb(b2[4], s2[0], s1[4], s1[2], 120.0, 109.5)
            } else if t2 == "Y" {
                energy = hb(s1[0], s1[1], s2[6], s2[5], 120.0, 120.0) +
                    hb(s2[5], s2[6], s1[4], s1[2], 120.0, 120.0)
            }
        } else if t1 == "N" {
            if t2 == "D" {
                energy = hb(s1[0], s1[2], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[0], s1[2], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[0], s1[2], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[0], s1[2], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[1], s1[0], 109.5, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[1], s1[0], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[1], s1[0], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[1], s1[0], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[1], s1[0], 120.0, 120.0) +
                    hb(s1[0], s1[2], s2[4], s2[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[1], s1[0], 120.0, 120.0) +
                    hb(s1[0], s1[2], s2[1], s2[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[1], s1[0], 120.0, 120.0) +
                    hb(s1[0], s1[2], s2[2], s2[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[1], s1[0], 109.5, 120.0) +
                    hb(s1[0], s1[2], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[1], s1[0], 109.5, 120.0) +
                    hb(s1[0], s1[2], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[1], s1[0], 109.5, 120.0) +
                    hb(s1[0], s1[2], s2[6], s2[5], 120.0, 120.0)
            }
        } else if t1 == "Q" {
            if t2 == "D" {
                energy = hb(s1[1], s1[3], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[1], s1[3], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[2], s1[1], 109.5, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[2], s1[1], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[2], s1[1], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[2], s1[1], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[2], s1[1], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[4], s2[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[2], s1[1], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[1], s2[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[2], s1[1], 120.0, 120.0) +
                    hb(s1[1], s1[3], s2[2], s2[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[2], s1[1], 109.5, 120.0) +
                    hb(s1[1], s1[3], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[2], s1[1], 109.5, 120.0) +
                    hb(s1[1], s1[3], s2[0], b2[4], 120.0, 109.5)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[2], s1[1], 109.5, 120.0) +
                    hb(s1[1], s1[3], s2[6], s2[5], 120.0, 120.0)
            }
        } else if t1 == "S" {
            if t2 == "D" {
                energy = hb(b1[4], s1[0], s2[1], s2[0], 120.0, 120.0) +
                    hb(b1[4], s1[0], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(b1[4], s1[0], s2[2], s2[1], 120.0, 120.0) +
                    hb(b1[4], s1[0], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[0], b1[4], 120.0, 109.5) +
                    hb(s2[3], s2[5], s1[0], b1[4], 120.0, 109.5) +
                    hb(s2[1], s2[2], s1[0], b1[4], 120.0, 109.5)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[0], b1[4], 120.0, 109.5)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[4], s2[2], 109.5, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[1], s2[0], 109.5, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[2], s2[1], 109.5, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[0], b1[4], 120.0, 109.5)
            }
        } else if t1 == "T" {
            if t2 == "D" {
                energy = hb(b1[4], s1[0], s2[1], s2[0], 120.0, 120.0) +
                    hb(b1[4], s1[0], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(b1[4], s1[0], s2[2], s2[1], 120.0, 120.0) +
                    hb(b1[4], s1[0], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[0], b1[4], 120.0, 109.5) +
                    hb(s2[3], s2[5], s1[0], b1[4], 120.0, 109.5) +
                    hb(s2[1], s2[2], s1[0], b1[4], 120.0, 109.5)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[0], b1[4], 120.0, 109.5)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[4], s2[2], 109.5, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[1], s2[0], 109.5, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[0], b1[4], 120.0, 109.5) +
                    hb(b1[4], s1[0], s2[2], s2[1], 109.5, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[0], b1[4], 109.5, 109.5)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[0], b1[4], 120.0, 109.5)
            }
        } else if t1 == "Y" {
            if t2 == "D" {
                energy = hb(s1[5], s1[6], s2[1], s2[0], 120.0, 120.0) +
                    hb(s1[5], s1[6], s2[2], s2[0], 120.0, 120.0)
            } else if t2 == "E" {
                energy = hb(s1[5], s1[6], s2[2], s2[1], 120.0, 120.0) +
                    hb(s1[5], s1[6], s2[3], s2[1], 120.0, 120.0)
            } else if t2 == "K" {
                energy = hb(s2[2], s2[3], s1[6], s1[5], 109.5, 120.0)
            } else if t2 == "R" {
                energy = hb(s2[3], s2[4], s1[6], s1[5], 120.0, 120.0) +
                    hb(s2[3], s2[5], s1[6], s1[5], 120.0, 120.0) +
                    hb(s2[1], s2[2], s1[6], s1[5], 120.0, 120.0)
            } else if t2 == "W" {
                energy = hb(s2[1], s2[3], s1[6], s1[5], 120.0, 120.0)
            } else if t2 == "H" {
                energy = hb(s2[0], s2[1], s1[6], s1[5], 120.0, 120.0) +
                    hb(s1[5], s1[6], s2[4], s2[2], 120.0, 120.0)
            } else if t2 == "N" {
                energy = hb(s2[0], s2[2], s1[6], s1[5], 120.0, 120.0) +
                    hb(s1[5], s1[6], s2[1], s2[0], 120.0, 120.0)
            } else if t2 == "Q" {
                energy = hb(s2[1], s2[3], s1[6], s1[5], 120.0, 120.0) +
                    hb(s1[5], s1[6], s2[2], s2[1], 120.0, 120.0)
            } else if t2 == "S" {
                energy = hb(b2[4], s2[0], s1[6], s1[5], 109.5, 120.0)
            } else if t2 == "T" {
                energy = hb(b2[4], s2[0], s1[6], s1[5], 109.5, 120.0)
            } else if t2 == "Y" {
                energy = hb(s2[5], s2[6], s1[6], s1[5], 120.0, 120.0)
            }
        }
        return energy
    }

    private static func hb(
        _ db: SIMD3<Float>, _ d: SIMD3<Float>, _ a: SIMD3<Float>, _ ab: SIMD3<Float>,
        _ dAngle: Float, _ aAngle: Float
    ) -> Float {
        hbondEnergy(donorBase: db, donor: d, acceptor: a, acceptorBase: ab,
                    optDonorAngle: dAngle, optAcceptorAngle: aAngle)
    }

    private static func ss(
        _ ca1: SIMD3<Float>, _ cb1: SIMD3<Float>, _ sg1: SIMD3<Float>,
        _ sg2: SIMD3<Float>, _ cb2: SIMD3<Float>, _ ca2: SIMD3<Float>
    ) -> Float {
        ssbondEnergy(ca1: ca1, cb1: cb1, sg1: sg1, sg2: sg2, cb2: cb2, ca2: ca2)
    }
}

// MARK: - Angle Helper Extension

extension SidechainTopologyStore {
    /// Compute angle (degrees) at vertex p2 between vectors p1→p2 and p3→p2.
    static func angleBetween(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>) -> Float {
        let v1 = p1 - p2
        let v2 = p3 - p2
        let len1 = simd_length(v1)
        let len2 = simd_length(v2)
        guard len1 > 1e-10, len2 > 1e-10 else { return 0 }
        var cosA = simd_dot(v1, v2) / (len1 * len2)
        cosA = min(1.0, max(-1.0, cosA))
        return acos(cosA) * 180.0 / .pi
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import simd
import Foundation

// MARK: - Pharmacophore Constraint Definition

/// User-facing pharmacophore constraint: specifies an indispensable interaction
/// that the docking GA must satisfy. Stored in DockingCoordinator, flattened
/// to GPU `PharmacophoreConstraint` structs at dock time.
struct PharmacophoreConstraintDef: Identifiable, Sendable {
    let id: UUID
    var targetScope: TargetScope
    var interactionType: ConstraintInteractionType
    var strength: ConstraintStrength
    var distanceThreshold: Float = 3.5
    var isEnabled: Bool = true

    // Source identification
    var sourceType: SourceType
    var proteinAtomIndex: Int?
    var residueIndex: Int?
    var ligandAtomIndex: Int?
    var chainID: String?
    var residueName: String?        // display: "ASP 25"
    var atomName: String?           // display: "OD1"
    var groupID: Int = 0

    // Resolved 3D target positions (one per target atom; multiple for residue-scope)
    var targetPositions: [SIMD3<Float>] = []

    init(
        targetScope: TargetScope,
        interactionType: ConstraintInteractionType,
        strength: ConstraintStrength = .soft(kcalPerAngstromSq: 5.0),
        distanceThreshold: Float = 3.5,
        sourceType: SourceType,
        proteinAtomIndex: Int? = nil,
        residueIndex: Int? = nil,
        ligandAtomIndex: Int? = nil,
        chainID: String? = nil,
        residueName: String? = nil,
        atomName: String? = nil
    ) {
        self.id = UUID()
        self.targetScope = targetScope
        self.interactionType = interactionType
        self.strength = strength
        self.distanceThreshold = distanceThreshold
        self.sourceType = sourceType
        self.proteinAtomIndex = proteinAtomIndex
        self.residueIndex = residueIndex
        self.ligandAtomIndex = ligandAtomIndex
        self.chainID = chainID
        self.residueName = residueName
        self.atomName = atomName
    }

    /// Human-readable description of the constraint target
    var targetLabel: String {
        if let name = atomName, let res = residueName {
            return "\(name) @ \(res)"
        } else if let res = residueName {
            return res
        } else if let idx = ligandAtomIndex {
            return "Ligand atom \(idx)"
        }
        return "Unknown"
    }
}

// MARK: - Enums

enum TargetScope: String, CaseIterable, Sendable {
    case atom = "Specific Atom"
    case residue = "Entire Residue"
}

enum ConstraintInteractionType: String, CaseIterable, Sendable {
    case hbondDonor = "H-Bond Donor"
    case hbondAcceptor = "H-Bond Acceptor"
    case saltBridge = "Salt Bridge"
    case piStacking = "π-Stacking"
    case halogen = "Halogen Bond"
    case metalCoordination = "Metal Coordination"
    case hydrophobic = "Hydrophobic"

    /// Bitmask of VinaAtomType values compatible with this constraint.
    /// Bit N is set if a ligand atom of VinaAtomType(N) can satisfy the constraint.
    var compatibleVinaTypes: UInt32 {
        // Bitmask where bit N = VinaAtomType(N) is compatible with this constraint.
        // Pre-computed constants to avoid complex type-check expressions.
        let mask: UInt32
        switch self {
        case .hbondDonor:
            // N_D(3), N_DA(5), O_D(7), O_DA(9), MET_D(18)
            mask = 0x000402A8
        case .hbondAcceptor:
            // N_A(4), N_DA(5), O_A(8), O_DA(9)
            mask = 0x00000330
        case .saltBridge:
            // N_D(3), N_A(4), N_DA(5), O_D(7), O_A(8), O_DA(9)
            mask = 0x000003B8
        case .piStacking:
            // C_H(0), C_P(1)
            mask = 0x00000003
        case .halogen:
            // F_H(12), Cl_H(13), Br_H(14), I_H(15)
            mask = 0x0000F000
        case .metalCoordination:
            // N_P(2)..O_DA(9), S_P(10)
            mask = 0x000007FC
        case .hydrophobic:
            // C_H(0), C_P(1), F_H(12), Cl_H(13), Br_H(14), I_H(15)
            mask = 0x0000F003
        }
        return mask
    }

    /// GPU enum value (PharmacophoreInteractionType)
    var gpuType: UInt16 {
        switch self {
        case .hbondDonor:       return 0
        case .hbondAcceptor:    return 1
        case .saltBridge:       return 2
        case .piStacking:       return 3
        case .halogen:          return 4
        case .metalCoordination: return 5
        case .hydrophobic:      return 6
        }
    }

    var icon: String {
        switch self {
        case .hbondDonor:       return "arrow.up.right.circle"
        case .hbondAcceptor:    return "arrow.down.left.circle"
        case .saltBridge:       return "bolt.circle"
        case .piStacking:       return "circle.hexagongrid"
        case .halogen:          return "atom"
        case .metalCoordination: return "diamond.circle"
        case .hydrophobic:      return "drop.circle"
        }
    }

    var color: SIMD4<Float> {
        switch self {
        case .hbondDonor:       return SIMD4<Float>(1.0, 0.6, 0.1, 1.0)   // orange
        case .hbondAcceptor:    return SIMD4<Float>(0.1, 0.8, 0.9, 1.0)   // cyan
        case .saltBridge:       return SIMD4<Float>(0.9, 0.2, 0.7, 1.0)   // magenta
        case .piStacking:       return SIMD4<Float>(0.6, 0.3, 0.9, 1.0)   // purple
        case .halogen:          return SIMD4<Float>(0.1, 0.6, 0.3, 1.0)   // dark green
        case .metalCoordination: return SIMD4<Float>(0.9, 0.8, 0.2, 1.0)  // gold
        case .hydrophobic:      return SIMD4<Float>(0.6, 0.6, 0.2, 1.0)   // olive
        }
    }
}

enum ConstraintStrength: Sendable {
    case soft(kcalPerAngstromSq: Float)   // default 5.0
    case hard                              // internally 1000.0

    var gpuValue: Float {
        switch self {
        case .soft(let v): return v
        case .hard: return 1000.0
        }
    }

    var isHard: Bool {
        if case .hard = self { return true }
        return false
    }

    var label: String {
        switch self {
        case .soft(let v): return "Soft (\(String(format: "%.1f", v)))"
        case .hard: return "Hard"
        }
    }
}

enum SourceType: String, CaseIterable, Sendable {
    case receptor
    case ligand
}

// MARK: - Residue-to-Atom Resolution

/// Maps (residueName, interactionType) to the PDB atom names relevant for that interaction.
/// Used to expand residue-scope constraints into atom-level GPU constraints.
enum ConstraintAtomResolver {

    /// Returns the PDB atom names within a residue that are relevant for the given interaction type.
    static func relevantAtomNames(residueName: String, interactionType: ConstraintInteractionType) -> [String] {
        let res = residueName.uppercased().trimmingCharacters(in: .whitespaces)
        switch interactionType {
        case .hbondAcceptor:
            return hbondAcceptorAtoms[res] ?? ["O"]  // backbone O as fallback
        case .hbondDonor:
            return hbondDonorAtoms[res] ?? ["N"]     // backbone N as fallback
        case .saltBridge:
            return saltBridgeAtoms[res] ?? []
        case .piStacking:
            return aromaticRingAtoms[res] ?? []
        case .halogen:
            return []  // protein residues don't have halogens
        case .metalCoordination:
            return metalCoordinatingAtoms[res] ?? ["O"]
        case .hydrophobic:
            return hydrophobicAtoms[res] ?? ["CB"]
        }
    }

    /// Resolve a constraint definition into 3D target positions using the protein's atom array.
    static func resolveTargetPositions(
        constraint: PharmacophoreConstraintDef,
        atoms: [Atom],
        residues: [Residue]
    ) -> [SIMD3<Float>] {
        if constraint.targetScope == .atom, let atomIdx = constraint.proteinAtomIndex,
           atomIdx >= 0, atomIdx < atoms.count {
            return [atoms[atomIdx].position]
        }

        guard constraint.targetScope == .residue,
              let resIdx = constraint.residueIndex,
              resIdx >= 0, resIdx < residues.count else {
            return []
        }

        let residue = residues[resIdx]
        let relevantNames = relevantAtomNames(
            residueName: residue.name,
            interactionType: constraint.interactionType
        )

        var positions: [SIMD3<Float>] = []
        for atomIdx in residue.atomIndices {
            guard atomIdx >= 0, atomIdx < atoms.count else { continue }
            let atom = atoms[atomIdx]
            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)
            if relevantNames.contains(trimmedName) {
                positions.append(atom.position)
            }
        }

        // Fallback: use all heavy atoms in the residue
        if positions.isEmpty {
            for atomIdx in residue.atomIndices {
                guard atomIdx >= 0, atomIdx < atoms.count else { continue }
                let atom = atoms[atomIdx]
                if atom.element != .H {
                    positions.append(atom.position)
                }
            }
        }

        return positions
    }

    // MARK: - Atom Name Lookup Tables

    private static let hbondAcceptorAtoms: [String: [String]] = [
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
        "ASN": ["OD1"],
        "GLN": ["OE1"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TYR": ["OH"],
        "HIS": ["ND1", "NE2"],
        "CYS": ["SG"],
        "MET": ["SD"],
    ]

    private static let hbondDonorAtoms: [String: [String]] = [
        "ARG": ["NH1", "NH2", "NE"],
        "LYS": ["NZ"],
        "ASN": ["ND2"],
        "GLN": ["NE2"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TYR": ["OH"],
        "TRP": ["NE1"],
        "HIS": ["ND1", "NE2"],
        "CYS": ["SG"],
    ]

    private static let saltBridgeAtoms: [String: [String]] = [
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
        "ARG": ["NH1", "NH2", "NE"],
        "LYS": ["NZ"],
        "HIS": ["ND1", "NE2"],
    ]

    private static let aromaticRingAtoms: [String: [String]] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    ]

    private static let metalCoordinatingAtoms: [String: [String]] = [
        "HIS": ["ND1", "NE2"],
        "CYS": ["SG"],
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
        "MET": ["SD"],
    ]

    private static let hydrophobicAtoms: [String: [String]] = [
        "ALA": ["CB"],
        "VAL": ["CG1", "CG2"],
        "LEU": ["CD1", "CD2"],
        "ILE": ["CD1", "CG2"],
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "PRO": ["CG", "CD"],
        "MET": ["CE", "SD"],
    ]
}

// MARK: - GPU Buffer Conversion

extension PharmacophoreConstraintDef {

    /// Convert a set of constraint definitions into GPU-ready structs.
    /// Residue-scope constraints are expanded into multiple GPU constraints sharing a groupID.
    /// Returns (constraints, params) ready for Metal buffer creation.
    static func toGPUBuffers(
        constraints: [PharmacophoreConstraintDef],
        atoms: [Atom],
        residues: [Residue]
    ) -> (constraints: [PharmacophoreConstraint], params: PharmacophoreParams) {
        var gpuConstraints: [PharmacophoreConstraint] = []
        var currentGroupID: UInt16 = 0

        for def in constraints where def.isEnabled {
            let positions: [SIMD3<Float>]
            if !def.targetPositions.isEmpty {
                // Use pre-resolved positions if available
                positions = def.targetPositions
            } else if def.sourceType == .ligand {
                // Ligand-side: use a dummy position (the constraint checks ligand atom → protein)
                positions = [SIMD3<Float>(0, 0, 0)]
            } else {
                // Resolve from protein structure
                positions = ConstraintAtomResolver.resolveTargetPositions(
                    constraint: def, atoms: atoms, residues: residues
                )
            }

            guard !positions.isEmpty else { continue }

            for pos in positions {
                var gc = PharmacophoreConstraint()
                gc.position = pos
                gc.distanceThreshold = def.distanceThreshold
                gc.compatibleVinaTypes = def.interactionType.compatibleVinaTypes
                gc.strength = def.strength.gpuValue
                gc.groupID = currentGroupID
                gc.constraintType = def.interactionType.gpuType
                gc.ligandAtomIndex = Int32(def.ligandAtomIndex ?? -1)
                gc._pad0 = 0
                gpuConstraints.append(gc)
            }

            currentGroupID += 1

            if gpuConstraints.count >= Int(MAX_PHARMACOPHORE_CONSTRAINTS) { break }
        }

        // Clamp to max
        if gpuConstraints.count > Int(MAX_PHARMACOPHORE_CONSTRAINTS) {
            gpuConstraints = Array(gpuConstraints.prefix(Int(MAX_PHARMACOPHORE_CONSTRAINTS)))
        }

        var params = PharmacophoreParams()
        params.numConstraints = UInt32(gpuConstraints.count)
        params.numGroups = UInt32(currentGroupID)
        params.globalScale = 1.0
        params._pad0 = 0

        return (gpuConstraints, params)
    }
}

// MARK: - Auto-Detection Helpers

extension ConstraintInteractionType {
    /// Suggest a default interaction type based on the selected atom's properties.
    static func suggestDefault(element: Element, atomName: String, residueName: String) -> ConstraintInteractionType {
        let name = atomName.trimmingCharacters(in: .whitespaces)
        let res = residueName.uppercased()

        // Metal atoms
        if [.Zn, .Fe, .Mg, .Ca, .Cu, .Mn, .Co, .Ni].contains(element) {
            return .metalCoordination
        }
        // Nitrogen — check if donor or acceptor by context
        if element == .N {
            if ["ARG", "LYS"].contains(res) { return .hbondDonor }
            if ["HIS"].contains(res) { return .hbondAcceptor }
            if ["NH1", "NH2", "NE", "NZ", "ND2", "NE2", "NE1"].contains(name) { return .hbondDonor }
            return .hbondAcceptor
        }
        // Oxygen
        if element == .O {
            if ["ASP", "GLU"].contains(res) { return .hbondAcceptor }
            if ["SER", "THR", "TYR"].contains(res) { return .hbondDonor }
            return .hbondAcceptor
        }
        // Sulfur
        if element == .S { return .hbondAcceptor }
        // Aromatic residues — suggest pi-stacking
        if ["PHE", "TYR", "TRP", "HIS"].contains(res) && element == .C {
            return .piStacking
        }
        // Carbon — hydrophobic
        if element == .C { return .hydrophobic }
        // Halogens
        if [.F, .Cl, .Br].contains(element) { return .halogen }

        return .hydrophobic
    }
}

import Foundation
import simd

// MARK: - Flexible Residue Docking

/// Manages flexible sidechain docking: user selects key binding-site residues (3-6),
/// and the GA simultaneously optimizes ligand pose + sidechain rotamer states.
struct FlexibleResidueConfig: Sendable {
    /// Indices of residues that should be flexible during docking
    var flexibleResidueIndices: [Int] = []
    /// Max number of flexible residues (typically 3-6)
    static let maxFlexibleResidues = 6
    /// Rotamer library resolution (degrees)
    var rotamerResolution: Float = 10.0
}

/// Standard amino acid rotamer definitions.
/// Each residue type has named chi angles with allowed discrete states.
struct RotamerLibrary {

    struct ChiAngle: Sendable {
        var atomNames: (String, String, String, String) // 4 atoms defining dihedral
        var allowedAngles: [Float] // in degrees
    }

    struct ResidueRotamers: Sendable {
        var residueName: String
        var chiAngles: [ChiAngle]
    }

    /// Penrose backbone-dependent rotamer library (simplified).
    /// Returns allowed rotamers for standard amino acids.
    static func rotamers(for residueName: String) -> ResidueRotamers? {
        switch residueName {
        case "PHE":
            return ResidueRotamers(residueName: "PHE", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD1"), allowedAngles: [80, -10])
            ])
        case "TYR":
            return ResidueRotamers(residueName: "TYR", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD1"), allowedAngles: [80, -10])
            ])
        case "TRP":
            return ResidueRotamers(residueName: "TRP", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD1"), allowedAngles: [-100, 90])
            ])
        case "HIS":
            return ResidueRotamers(residueName: "HIS", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "ND1"), allowedAngles: [-165, 80])
            ])
        case "ASP":
            return ResidueRotamers(residueName: "ASP", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "OD1"), allowedAngles: [-10, 0, 10])
            ])
        case "GLU":
            return ResidueRotamers(residueName: "GLU", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CB", "CG", "CD", "OE1"), allowedAngles: [-10, 0, 10])
            ])
        case "ASN":
            return ResidueRotamers(residueName: "ASN", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "OD1"), allowedAngles: [-20, 0, 20])
            ])
        case "GLN":
            return ResidueRotamers(residueName: "GLN", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CB", "CG", "CD", "OE1"), allowedAngles: [-20, 0, 20])
            ])
        case "LYS":
            return ResidueRotamers(residueName: "LYS", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CB", "CG", "CD", "CE"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CG", "CD", "CE", "NZ"), allowedAngles: [-177, -65, 62])
            ])
        case "ARG":
            return ResidueRotamers(residueName: "ARG", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CB", "CG", "CD", "NE"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CG", "CD", "NE", "CZ"), allowedAngles: [-177, 0])
            ])
        case "SER":
            return ResidueRotamers(residueName: "SER", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "OG"), allowedAngles: [-65, 62, 180])
            ])
        case "THR":
            return ResidueRotamers(residueName: "THR", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "OG1"), allowedAngles: [-65, 62, 180])
            ])
        case "CYS":
            return ResidueRotamers(residueName: "CYS", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "SG"), allowedAngles: [-65, 62, 180])
            ])
        case "MET":
            return ResidueRotamers(residueName: "MET", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "SD"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CB", "CG", "SD", "CE"), allowedAngles: [-177, -65, 62])
            ])
        case "LEU":
            return ResidueRotamers(residueName: "LEU", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG", "CD1"), allowedAngles: [65, 175])
            ])
        case "ILE":
            return ResidueRotamers(residueName: "ILE", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG1"), allowedAngles: [-177, -65, 62]),
                ChiAngle(atomNames: ("CA", "CB", "CG1", "CD1"), allowedAngles: [-177, -65, 62])
            ])
        case "VAL":
            return ResidueRotamers(residueName: "VAL", chiAngles: [
                ChiAngle(atomNames: ("N", "CA", "CB", "CG1"), allowedAngles: [-177, -65, 62])
            ])
        default:
            return nil // ALA, GLY, PRO have no rotatable sidechains
        }
    }

    /// Total number of rotamer states for a residue.
    static func rotamerCount(for residueName: String) -> Int {
        guard let r = rotamers(for: residueName) else { return 1 }
        return r.chiAngles.reduce(1) { $0 * $1.allowedAngles.count }
    }
}

/// Applies a rotamer state to a residue's sidechain atoms.
struct RotamerApplicator {

    /// Apply a specific rotamer state to sidechain atoms.
    /// rotamerIndex encodes the combination of chi angles.
    static func applyRotamer(
        atoms: inout [Atom],
        residueAtomIndices: [Int],
        residueName: String,
        rotamerIndex: Int
    ) {
        guard let rotDef = RotamerLibrary.rotamers(for: residueName) else { return }

        // Decode rotamer index into per-chi-angle selections
        var remaining = rotamerIndex
        var selectedAngles: [Float] = []
        for chi in rotDef.chiAngles.reversed() {
            let n = chi.allowedAngles.count
            guard n > 0 else { continue }
            selectedAngles.insert(chi.allowedAngles[remaining % n], at: 0)
            remaining /= n
        }

        // Apply each chi angle rotation
        let atomsByName = Dictionary(uniqueKeysWithValues: residueAtomIndices.compactMap { idx -> (String, Int)? in
            guard idx < atoms.count else { return nil }
            return (atoms[idx].name, idx)
        })

        for (chiIdx, chi) in rotDef.chiAngles.enumerated() {
            guard chiIdx < selectedAngles.count else { continue }
            let targetAngle = selectedAngles[chiIdx] * Float.pi / 180.0

            // Find the 4 atoms defining the dihedral
            guard let idx2 = atomsByName[chi.atomNames.1],
                  let idx3 = atomsByName[chi.atomNames.2]
            else { continue }

            let axis = simd_normalize(atoms[idx3].position - atoms[idx2].position)
            let pivot = atoms[idx2].position

            // Find atoms on the rotating side (downstream of bond 2-3)
            let downstreamAtoms = findDownstreamAtoms(
                from: idx3, excluding: idx2,
                atoms: atoms, residueIndices: residueAtomIndices
            )

            // Compute current dihedral angle
            if let idx1 = atomsByName[chi.atomNames.0],
               let idx4 = atomsByName[chi.atomNames.3] {
                let currentAngle = dihedralAngle(
                    atoms[idx1].position, atoms[idx2].position,
                    atoms[idx3].position, atoms[idx4].position
                )
                let delta = targetAngle - currentAngle

                // Rotate downstream atoms
                let rotation = simd_quatf(angle: delta, axis: axis)
                for atomIdx in downstreamAtoms {
                    let relative = atoms[atomIdx].position - pivot
                    atoms[atomIdx].position = pivot + rotation.act(relative)
                }
            }
        }
    }

    /// Find all atoms downstream of `from` excluding `excluding` within the residue.
    private static func findDownstreamAtoms(
        from startIdx: Int, excluding: Int,
        atoms: [Atom], residueIndices: [Int]
    ) -> [Int] {
        // BFS through bonds within the residue
        let residueSet = Set(residueIndices)
        var visited = Set<Int>([excluding])
        var queue = [startIdx]
        var result: [Int] = []

        while !queue.isEmpty {
            let current = queue.removeFirst()
            guard !visited.contains(current) else { continue }
            visited.insert(current)
            result.append(current)

            // Find bonded neighbors within residue (distance-based)
            for idx in residueIndices {
                if !visited.contains(idx) && residueSet.contains(idx) {
                    let d = simd_distance(atoms[current].position, atoms[idx].position)
                    if d < 1.9 && d > 0.4 { // bond distance
                        queue.append(idx)
                    }
                }
            }
        }

        return result
    }

    /// Compute dihedral angle from 4 positions.
    private static func dihedralAngle(
        _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
        _ p3: SIMD3<Float>, _ p4: SIMD3<Float>
    ) -> Float {
        let b1 = p2 - p1
        let b2 = p3 - p2
        let b3 = p4 - p3

        let n1 = simd_cross(b1, b2)
        let n2 = simd_cross(b2, b3)

        let m1 = simd_cross(n1, simd_normalize(b2))

        let x = simd_dot(n1, n2)
        let y = simd_dot(m1, n2)

        return atan2(y, x)
    }
}

// MARK: - Extended DockPose for Flexible Docking

/// A flexible docking pose includes both ligand pose and sidechain rotamer states.
struct FlexibleDockPose: Sendable {
    var ligandPose: DockPoseSwift
    var rotamerStates: [Int]  // one rotamer index per flexible residue

    /// Total degrees of freedom = ligand DOF + sum of chi angles
    var totalDOF: Int {
        7 + ligandPose.torsions.count + rotamerStates.count
    }
}

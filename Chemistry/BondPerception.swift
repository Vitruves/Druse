// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import simd

enum BondPerception {

    /// Infer bonds from 3D coordinates using covalent radii and spatial hashing.
    /// O(N) expected time for uniformly distributed atoms.
    static func perceiveBonds(
        in atoms: [Atom],
        tolerance: Float = 1.3,
        minDistance: Float = 0.4
    ) -> [Bond] {
        guard atoms.count > 1 else {
            Task { @MainActor in ActivityLog.shared.debug("[Prep] Bond perception skipped: \(atoms.count) atom(s)", category: .prep) }
            return []
        }

        let cellSize: Float = 5.0 // covers max bonded distance
        var grid: [SIMD3<Int32>: [Int]] = [:]

        // Build spatial hash
        for (i, atom) in atoms.enumerated() {
            let cell = cellCoord(atom.position, cellSize: cellSize)
            grid[cell, default: []].append(i)
        }

        var bonds: [Bond] = []
        var bondID = 0

        // Check each atom against neighbors in adjacent cells
        for (i, atomA) in atoms.enumerated() {
            let cellA = cellCoord(atomA.position, cellSize: cellSize)

            for dx: Int32 in -1...1 {
                for dy: Int32 in -1...1 {
                    for dz: Int32 in -1...1 {
                        let neighborCell = SIMD3<Int32>(cellA.x + dx, cellA.y + dy, cellA.z + dz)
                        guard let indices = grid[neighborCell] else { continue }

                        for j in indices {
                            guard j > i else { continue } // avoid duplicates
                            let atomB = atoms[j]

                            let dist = simd_length(atomA.position - atomB.position)

                            // Skip degenerate pairs
                            guard dist > minDistance else { continue }

                            // Check if within bonding distance
                            let threshold = (atomA.element.covalentRadius + atomB.element.covalentRadius) * tolerance
                            guard dist < threshold else { continue }

                            bonds.append(Bond(id: bondID, atomIndex1: i, atomIndex2: j))
                            bondID += 1
                        }
                    }
                }
            }
        }

        Task { @MainActor in ActivityLog.shared.debug("[Prep] Bond perception: \(bonds.count) bonds from \(atoms.count) atoms", category: .prep) }
        return bonds
    }

    private static func cellCoord(_ position: SIMD3<Float>, cellSize: Float) -> SIMD3<Int32> {
        SIMD3<Int32>(
            Int32(floorf(position.x / cellSize)),
            Int32(floorf(position.y / cellSize)),
            Int32(floorf(position.z / cellSize))
        )
    }
}

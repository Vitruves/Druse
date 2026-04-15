// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - Secondary Structure Colors

private let helixColor  = SIMD4<Float>(0.90, 0.20, 0.30, 1.0)
private let sheetColor  = SIMD4<Float>(0.95, 0.85, 0.15, 1.0)
private let coilColor   = SIMD4<Float>(0.55, 0.55, 0.55, 1.0)
private let turnColor   = SIMD4<Float>(0.30, 0.80, 0.90, 1.0)

// MARK: - Catmull-Rom Spline

private func catmullRom(
    p0: SIMD3<Float>, p1: SIMD3<Float>, p2: SIMD3<Float>, p3: SIMD3<Float>, t: Float
) -> SIMD3<Float> {
    let t2: Float = t * t
    let t3: Float = t2 * t
    let a: SIMD3<Float> = 2.0 * p1
    let b: SIMD3<Float> = (p2 - p0) * t
    let c0: SIMD3<Float> = 2.0 * p0 - 5.0 * p1
    let c1: SIMD3<Float> = c0 + 4.0 * p2 - p3
    let c: SIMD3<Float> = c1 * t2
    let d0: SIMD3<Float> = -p0 + 3.0 * p1
    let d1: SIMD3<Float> = d0 - 3.0 * p2 + p3
    let d: SIMD3<Float> = d1 * t3
    let sum: SIMD3<Float> = a + b + c + d
    return 0.5 * sum
}

private func catmullRomTangent(
    p0: SIMD3<Float>, p1: SIMD3<Float>, p2: SIMD3<Float>, p3: SIMD3<Float>, t: Float
) -> SIMD3<Float> {
    let t2: Float = t * t
    let a: SIMD3<Float> = -p0 + p2
    let b0: SIMD3<Float> = 4.0 * p0 - 10.0 * p1
    let b1: SIMD3<Float> = b0 + 8.0 * p2 - 2.0 * p3
    let b: SIMD3<Float> = b1 * t
    let c0: SIMD3<Float> = -3.0 * p0 + 9.0 * p1
    let c1: SIMD3<Float> = c0 - 9.0 * p2 + 3.0 * p3
    let c: SIMD3<Float> = c1 * t2
    let sum: SIMD3<Float> = a + b + c
    let tangent: SIMD3<Float> = 0.5 * sum
    let len: Float = simd_length(tangent)
    return len > 0.0001 ? tangent / len : SIMD3<Float>(0, 0, 1)
}

// MARK: - Ribbon Generator

enum RibbonMeshGenerator {

    static let subdivisionsPerResidue: Int = 10
    static let radialSegments: Int = 8

    struct CrossSection {
        var width: Float
        var height: Float
    }

    static func crossSection(for ss: SecondaryStructure) -> CrossSection {
        switch ss {
        case .helix: CrossSection(width: 2.0, height: 0.4)
        case .sheet: CrossSection(width: 2.0, height: 0.2)
        case .coil:  CrossSection(width: 0.3, height: 0.3)
        case .turn:  CrossSection(width: 0.3, height: 0.3)
        }
    }

    static func color(for ss: SecondaryStructure) -> SIMD4<Float> {
        switch ss {
        case .helix: helixColor
        case .sheet: sheetColor
        case .coil:  coilColor
        case .turn:  turnColor
        }
    }

    /// Generate a proper ribbon mesh from Cα positions and SS assignments.
    static func generate(
        caPositions: [SIMD3<Float>],
        ssAssignments: [SecondaryStructure],
        selectedResidues: [Bool] = []
    ) -> (vertices: [RibbonVertex], indices: [UInt32]) {
        let n = caPositions.count
        guard n >= 4 else { return ([], []) }

        // Step 1: Spline interpolation
        var splinePositions: [SIMD3<Float>] = []
        var splineTangents: [SIMD3<Float>] = []
        var splineSecondary: [SecondaryStructure] = []
        var splineArrowParam: [Float] = []
        var splineFlags: [UInt32] = []

        let subs = subdivisionsPerResidue
        let selected = selectedResidues.count == n ? selectedResidues : Array(repeating: false, count: n)

        for seg in 0..<(n - 1) {
            let i0 = max(seg - 1, 0)
            let i1 = seg
            let i2 = min(seg + 1, n - 1)
            let i3 = min(seg + 2, n - 1)

            let p0 = caPositions[i0]
            let p1 = caPositions[i1]
            let p2 = caPositions[i2]
            let p3 = caPositions[i3]
            let ssType = ssAssignments[i1]
            let isSelected = selected[i1] || selected[i2]

            // Detect C-terminal end of sheet for arrow
            let isLastSheet = (ssType == .sheet) &&
                (seg == n - 2 || ssAssignments[min(seg + 1, n - 1)] != .sheet)

            for sub in 0..<subs {
                let t = Float(sub) / Float(subs)
                splinePositions.append(catmullRom(p0: p0, p1: p1, p2: p2, p3: p3, t: t))
                splineTangents.append(catmullRomTangent(p0: p0, p1: p1, p2: p2, p3: p3, t: t))
                splineSecondary.append(ssType)
                splineArrowParam.append(isLastSheet ? t : 0.0)
                splineFlags.append(isSelected ? 1 : 0)
            }
        }

        // Last point
        splinePositions.append(caPositions[n - 1])
        splineTangents.append(splineTangents.last ?? SIMD3<Float>(0, 0, 1))
        splineSecondary.append(ssAssignments[n - 1])
        splineArrowParam.append(0.0)
        splineFlags.append(selected[n - 1] ? 1 : 0)

        let splineCount = splinePositions.count
        guard splineCount >= 2 else { return ([], []) }

        // Step 2: Parallel transport frames
        var splineNormals = [SIMD3<Float>](repeating: .zero, count: splineCount)

        let t0 = splineTangents[0]
        let up: SIMD3<Float> = abs(t0.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        splineNormals[0] = simd_normalize(simd_cross(t0, up))

        for i in 1..<splineCount {
            let tPrev = splineTangents[i - 1]
            let tCurr = splineTangents[i]
            let b = simd_cross(tPrev, tCurr)
            let bLen = simd_length(b)

            if bLen < 0.0001 {
                splineNormals[i] = splineNormals[i - 1]
            } else {
                let axis = b / bLen
                let angle = acosf(simd_clamp(simd_dot(tPrev, tCurr), -1.0, 1.0))
                let q = simd_quatf(angle: angle, axis: axis)
                splineNormals[i] = q.act(splineNormals[i - 1])
            }

            // Re-orthogonalize
            let corrected = splineNormals[i] - simd_dot(splineNormals[i], tCurr) * tCurr
            let corrLen = simd_length(corrected)
            splineNormals[i] = corrLen > 0.0001 ? corrected / corrLen : splineNormals[max(i - 1, 0)]
        }

        // Step 3: Cross-section extrusion
        let radialSegs = radialSegments
        let ringSize = radialSegs + 1
        var vertices: [RibbonVertex] = []
        vertices.reserveCapacity(splineCount * ringSize)

        for i in 0..<splineCount {
            let pos = splinePositions[i]
            let tang = splineTangents[i]
            let norm = splineNormals[i]
            let binorm = simd_cross(tang, norm)

            let ss = splineSecondary[i]
            let cs = crossSection(for: ss)
            let vertColor = color(for: ss)

            var halfWidth = cs.width * 0.5
            var halfHeight = cs.height * 0.5

            // Sheet arrows
            if ss == .sheet && splineArrowParam[i] > 0.001 {
                let arrowT = splineArrowParam[i]
                halfWidth *= (1.5 * (1.0 - arrowT))
                halfHeight *= max(1.0 - arrowT * 0.5, 0.2)
            }

            let texV = Float(i) / Float(max(splineCount - 1, 1))

            for j in 0...radialSegs {
                let theta = Float(j) / Float(radialSegs) * 2.0 * .pi
                let cosT = cosf(theta)
                let sinT = sinf(theta)

                let localX = cosT * halfWidth
                let localY = sinT * halfHeight
                let surfPos = pos + norm * localX + binorm * localY

                // Elliptical normal
                let nX = cosT / max(halfWidth, 0.001)
                let nY = sinT / max(halfHeight, 0.001)
                let localNormal = simd_normalize(norm * nX + binorm * nY)

                let texU = Float(j) / Float(radialSegs)

                vertices.append(RibbonVertex(
                    position: surfPos,
                    normal: localNormal,
                    color: vertColor,
                    texCoord: SIMD2<Float>(texU, texV),
                    flags: splineFlags[i]
                ))
            }
        }

        // Step 4: Index generation
        var indices: [UInt32] = []
        indices.reserveCapacity((splineCount - 1) * radialSegs * 6)

        for i in 0..<(splineCount - 1) {
            let ringA = UInt32(i * ringSize)
            let ringB = UInt32((i + 1) * ringSize)

            for j in 0..<UInt32(radialSegs) {
                let a0 = ringA + j
                let a1 = ringA + j + 1
                let b0 = ringB + j
                let b1 = ringB + j + 1

                indices.append(a0)
                indices.append(b0)
                indices.append(a1)
                indices.append(a1)
                indices.append(b0)
                indices.append(b1)
            }
        }

        return (vertices, indices)
    }

    /// Generate ribbon mesh for all chains of a molecule using its SS assignments.
    /// When `chainColorMap` is non-empty, each chain is colored by its map entry
    /// instead of using secondary-structure-based colors.
    @MainActor
    static func generateForMolecule(
        _ molecule: Molecule,
        chainColorMap: [String: SIMD3<Float>] = [:],
        selectedResidueKeys: Set<String> = []
    ) -> (vertices: [RibbonVertex], indices: [UInt32]) {
        var allVertices: [RibbonVertex] = []
        var allIndices: [UInt32] = []

        for chain in molecule.chains {
            guard chain.type == .protein else { continue }

            // Extract Cα atoms sorted by residue sequence
            let caAtoms = molecule.atoms
                .filter { $0.chainID == chain.id && $0.name.trimmingCharacters(in: .whitespaces) == "CA" }
                .sorted { $0.residueSeq < $1.residueSeq }

            guard caAtoms.count >= 4 else { continue }

            let caPositions = caAtoms.map(\.position)

            // Build SS assignments per Cα
            let ssAssignments: [SecondaryStructure] = caAtoms.map { atom in
                for ssa in molecule.secondaryStructureAssignments {
                    if ssa.chain == chain.id &&
                       atom.residueSeq >= ssa.start &&
                       atom.residueSeq <= ssa.end {
                        return ssa.type
                    }
                }
                return .coil
            }

            let selectedResidues = caAtoms.map { atom in
                selectedResidueKeys.contains("\(atom.chainID)|\(atom.residueSeq)")
            }

            let (verts, idxs) = generate(
                caPositions: caPositions,
                ssAssignments: ssAssignments,
                selectedResidues: selectedResidues
            )

            // Override vertex colors with per-chain color when chain coloring is active
            var finalVerts = verts
            if let chainColor = chainColorMap[chain.id] {
                let color4 = SIMD4<Float>(chainColor.x, chainColor.y, chainColor.z, 1.0)
                for i in finalVerts.indices {
                    finalVerts[i].color = color4
                }
            }

            // Offset indices for multi-chain merge
            let vertexOffset = UInt32(allVertices.count)
            allVertices.append(contentsOf: finalVerts)
            allIndices.append(contentsOf: idxs.map { $0 + vertexOffset })
        }

        return (allVertices, allIndices)
    }
}

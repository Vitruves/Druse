// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - Loop Builder for Missing Residue Reconstruction

enum LoopBuilder {

    // MARK: - Constants (ideal peptide backbone geometry)

    private static let bondLengthCN: Float  = 1.335   // C-N (peptide bond)
    private static let bondLengthNCA: Float = 1.458   // N-CA
    private static let bondLengthCAC: Float = 1.525   // CA-C
    private static let bondLengthCO: Float  = 1.231   // C=O

    private static let angleNCAC: Float  = 111.2 * .pi / 180  // N-CA-C
    private static let angleCACC: Float  = 116.2 * .pi / 180  // CA-C-N (at C)
    private static let angleCNCA: Float  = 121.7 * .pi / 180  // C-N-CA (at N)
    private static let angleCACO: Float  = 120.8 * .pi / 180  // CA-C=O

    private static let maxGapLength = 15
    private static let caCADistance: Float = 3.8  // approximate CA-CA distance per residue

    // MARK: - Result

    struct LoopBuildResult {
        let atoms: [Atom]
        let bonds: [Bond]
        let gapsBuilt: Int
        let residuesAdded: Int
    }

    // MARK: - Main Entry Point

    static func buildMissingLoops(
        atoms: [Atom],
        bonds: [Bond],
        gaps: [(chainID: String, gapStart: Int, gapEnd: Int)],
        chainSequences: [GemmiBridge.ChainSequence]
    ) -> LoopBuildResult {
        var workingAtoms = atoms
        var workingBonds = bonds
        var gapsBuilt = 0
        var totalResiduesAdded = 0

        // Build sequence lookup: chainID → [residueName]
        var seqLookup: [String: [String]] = [:]
        for seq in chainSequences {
            seqLookup[seq.chainID] = seq.residueNames
        }

        // Process each gap (in reverse order to avoid index invalidation)
        let sortedGaps = gaps.sorted { ($0.chainID, $0.gapStart) > ($1.chainID, $1.gapStart) }

        for gap in sortedGaps {
            let gapLength = gap.gapEnd - gap.gapStart + 1

            if gapLength > maxGapLength {
                continue
            }

            // Determine residue types from SEQRES (fallback to ALA)
            let residueNames = resolveResidueNames(
                chainID: gap.chainID,
                gapStart: gap.gapStart,
                gapEnd: gap.gapEnd,
                seqLookup: seqLookup,
                atoms: workingAtoms
            )

            // Find anchor atoms
            let anchors = findAnchors(
                chainID: gap.chainID,
                gapStart: gap.gapStart,
                gapEnd: gap.gapEnd,
                atoms: workingAtoms
            )

            guard anchors.hasAtLeastOneAnchor else {
                continue
            }

            // Build backbone + sidechain atoms
            let built = buildLoop(
                residueNames: residueNames,
                anchors: anchors,
                chainID: gap.chainID,
                gapStart: gap.gapStart,
                existingAtoms: workingAtoms,
                existingBonds: workingBonds
            )

            guard !built.newAtoms.isEmpty else { continue }

            // Merge built atoms and bonds
            let baseAtomID = workingAtoms.count
            let baseBondID = workingBonds.count

            for (i, atom) in built.newAtoms.enumerated() {
                workingAtoms.append(Atom(
                    id: baseAtomID + i,
                    element: atom.element,
                    position: atom.position,
                    name: atom.name,
                    residueName: atom.residueName,
                    residueSeq: atom.residueSeq,
                    chainID: atom.chainID,
                    charge: atom.charge,
                    formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom,
                    occupancy: atom.occupancy,
                    tempFactor: atom.tempFactor
                ))
            }

            for (i, bond) in built.newBonds.enumerated() {
                workingBonds.append(Bond(
                    id: baseBondID + i,
                    atomIndex1: bond.atomIndex1 + baseAtomID,
                    atomIndex2: bond.atomIndex2 + baseAtomID,
                    order: bond.order
                ))
            }

            // Add peptide bonds connecting loop to anchors
            var nextBondID = baseBondID + built.newBonds.count
            if let anchorBondInfo = built.anchorBonds {
                for (localIdx, existingIdx, order) in anchorBondInfo {
                    workingBonds.append(Bond(
                        id: nextBondID,
                        atomIndex1: existingIdx,
                        atomIndex2: baseAtomID + localIdx,
                        order: order
                    ))
                    nextBondID += 1
                }
            }

            gapsBuilt += 1
            totalResiduesAdded += residueNames.count
        }

        return LoopBuildResult(
            atoms: workingAtoms,
            bonds: workingBonds,
            gapsBuilt: gapsBuilt,
            residuesAdded: totalResiduesAdded
        )
    }

    // MARK: - Resolve Residue Names from SEQRES

    private static func resolveResidueNames(
        chainID: String,
        gapStart: Int,
        gapEnd: Int,
        seqLookup: [String: [String]],
        atoms: [Atom]
    ) -> [String] {
        let gapLength = gapEnd - gapStart + 1

        guard let fullSeq = seqLookup[chainID], !fullSeq.isEmpty else {
            return [String](repeating: "ALA", count: gapLength)
        }

        // Find the offset: the first residue in this chain has some residueSeq
        // and corresponds to index 0 in fullSeq
        let chainAtoms = atoms.filter { $0.chainID == chainID && !$0.isHetAtom }
        guard let firstResSeq = chainAtoms.map(\.residueSeq).min() else {
            return [String](repeating: "ALA", count: gapLength)
        }

        var result: [String] = []
        result.reserveCapacity(gapLength)

        for seq in gapStart...gapEnd {
            let seqIdx = seq - firstResSeq
            if seqIdx >= 0, seqIdx < fullSeq.count {
                result.append(fullSeq[seqIdx])
            } else {
                result.append("ALA")
            }
        }

        return result
    }

    // MARK: - Anchor Atoms

    private struct Anchors {
        // Before gap: the last residue's backbone atoms
        var prevC: (index: Int, position: SIMD3<Float>)?
        var prevCA: (index: Int, position: SIMD3<Float>)?
        var prevO: (index: Int, position: SIMD3<Float>)?

        // After gap: the first residue's backbone atoms
        var nextN: (index: Int, position: SIMD3<Float>)?
        var nextCA: (index: Int, position: SIMD3<Float>)?

        var hasAtLeastOneAnchor: Bool {
            prevC != nil || nextN != nil
        }

        var hasBothAnchors: Bool {
            prevC != nil && nextN != nil
        }
    }

    private static func findAnchors(
        chainID: String,
        gapStart: Int,
        gapEnd: Int,
        atoms: [Atom]
    ) -> Anchors {
        var anchors = Anchors()

        let prevResSeq = gapStart - 1
        let nextResSeq = gapEnd + 1

        for (i, atom) in atoms.enumerated() {
            guard atom.chainID == chainID, !atom.isHetAtom else { continue }

            if atom.residueSeq == prevResSeq {
                let trimmed = atom.name.trimmingCharacters(in: .whitespaces)
                switch trimmed {
                case "C":  anchors.prevC  = (i, atom.position)
                case "CA": anchors.prevCA = (i, atom.position)
                case "O":  anchors.prevO  = (i, atom.position)
                default: break
                }
            } else if atom.residueSeq == nextResSeq {
                let trimmed = atom.name.trimmingCharacters(in: .whitespaces)
                switch trimmed {
                case "N":  anchors.nextN  = (i, atom.position)
                case "CA": anchors.nextCA = (i, atom.position)
                default: break
                }
            }
        }

        return anchors
    }

    // MARK: - Build Loop

    private struct ResidueBackbone {
        var nIdx: Int
        var caIdx: Int
        var cIdx: Int
        var oIdx: Int
    }

    private struct LoopBuildOutput {
        let newAtoms: [Atom]
        let newBonds: [Bond]
        // (localAtomIndex, existingAtomIndex, bondOrder) for connecting to anchors
        let anchorBonds: [(Int, Int, BondOrder)]?
    }

    private static func buildLoop(
        residueNames: [String],
        anchors: Anchors,
        chainID: String,
        gapStart: Int,
        existingAtoms: [Atom],
        existingBonds: [Bond]
    ) -> LoopBuildOutput {
        let n = residueNames.count

        // Step 1: Generate CA positions along a smooth curve between anchors
        let caPositions = generateCAPositions(
            count: n,
            anchors: anchors
        )

        guard caPositions.count == n else {
            return LoopBuildOutput(newAtoms: [], newBonds: [], anchorBonds: nil)
        }

        // Step 2: Build full backbone from CA positions
        var newAtoms: [Atom] = []
        var newBonds: [Bond] = []
        var anchorBondInfo: [(Int, Int, BondOrder)] = []

        var backbones: [ResidueBackbone] = []

        for i in 0..<n {
            let residueSeq = gapStart + i
            let resName = residueNames[i]
            let caPos = caPositions[i]

            // Determine local coordinate frame at this CA
            let forward: SIMD3<Float>
            if i < n - 1 {
                forward = simd_normalize(caPositions[i + 1] - caPos)
            } else if let nextN = anchors.nextN {
                forward = simd_normalize(nextN.position - caPos)
            } else if i > 0 {
                forward = simd_normalize(caPos - caPositions[i - 1])
            } else {
                forward = SIMD3<Float>(1, 0, 0)
            }

            let backward: SIMD3<Float>
            if i > 0 {
                backward = simd_normalize(caPositions[i - 1] - caPos)
            } else if let prevC = anchors.prevC {
                backward = simd_normalize(prevC.position - caPos)
            } else {
                backward = -forward
            }

            // Build a perpendicular vector for placing O atoms
            var up = simd_cross(forward, backward)
            if simd_length(up) < 1e-4 {
                up = arbitraryPerpendicular(to: forward)
            }
            up = simd_normalize(up)

            let baseIdx = newAtoms.count

            // N atom: placed backward from CA
            let nDir = simd_normalize(backward + forward * cos(.pi - angleCNCA))
            let nPos = caPos + simd_normalize(nDir) * bondLengthNCA

            // C atom: placed forward from CA
            let cDir = simd_normalize(forward + backward * cos(.pi - angleNCAC))
            let cPos = caPos + simd_normalize(cDir) * bondLengthCAC

            // O atom: placed perpendicular at C
            let cToCA = simd_normalize(caPos - cPos)
            let oDir = rotateAround(axis: simd_cross(cToCA, up), vector: cToCA, angle: angleCACO)
            let oPos = cPos + simd_normalize(oDir) * bondLengthCO

            // Create backbone atoms
            let nAtom = Atom(
                id: 0, element: .N, position: nPos, name: "N",
                residueName: resName, residueSeq: residueSeq, chainID: chainID,
                charge: 0, formalCharge: 0, isHetAtom: false, occupancy: 1.0, tempFactor: 50.0
            )
            let caAtom = Atom(
                id: 0, element: .C, position: caPos, name: "CA",
                residueName: resName, residueSeq: residueSeq, chainID: chainID,
                charge: 0, formalCharge: 0, isHetAtom: false, occupancy: 1.0, tempFactor: 50.0
            )
            let cAtom = Atom(
                id: 0, element: .C, position: cPos, name: "C",
                residueName: resName, residueSeq: residueSeq, chainID: chainID,
                charge: 0, formalCharge: 0, isHetAtom: false, occupancy: 1.0, tempFactor: 50.0
            )
            let oAtom = Atom(
                id: 0, element: .O, position: oPos, name: "O",
                residueName: resName, residueSeq: residueSeq, chainID: chainID,
                charge: 0, formalCharge: 0, isHetAtom: false, occupancy: 1.0, tempFactor: 50.0
            )

            let nIdx = baseIdx
            let caIdx = baseIdx + 1
            let cIdx = baseIdx + 2
            let oIdx = baseIdx + 3

            newAtoms.append(contentsOf: [nAtom, caAtom, cAtom, oAtom])

            // Intra-residue backbone bonds
            let bondBase = newBonds.count
            newBonds.append(Bond(id: bondBase, atomIndex1: nIdx, atomIndex2: caIdx, order: .single))
            newBonds.append(Bond(id: bondBase + 1, atomIndex1: caIdx, atomIndex2: cIdx, order: .single))
            newBonds.append(Bond(id: bondBase + 2, atomIndex1: cIdx, atomIndex2: oIdx, order: .double))

            backbones.append(ResidueBackbone(nIdx: nIdx, caIdx: caIdx, cIdx: cIdx, oIdx: oIdx))

            // Peptide bond to previous built residue
            if i > 0 {
                let prevCIdx = backbones[i - 1].cIdx
                newBonds.append(Bond(id: newBonds.count, atomIndex1: prevCIdx, atomIndex2: nIdx, order: .single))
            }
        }

        // Anchor bonds: connect first N to prevC, last C to nextN
        if let prevC = anchors.prevC, !backbones.isEmpty {
            anchorBondInfo.append((backbones[0].nIdx, prevC.index, .single))
        }
        if let nextN = anchors.nextN, !backbones.isEmpty {
            anchorBondInfo.append((backbones[n - 1].cIdx, nextN.index, .single))
        }

        // Step 3: Add sidechain atoms from CCD templates
        let sidechainResult = addSidechains(
            residueNames: residueNames,
            backbones: backbones,
            newAtoms: &newAtoms,
            newBonds: &newBonds,
            chainID: chainID,
            gapStart: gapStart
        )
        _ = sidechainResult

        return LoopBuildOutput(
            newAtoms: newAtoms,
            newBonds: newBonds,
            anchorBonds: anchorBondInfo.isEmpty ? nil : anchorBondInfo
        )
    }

    // MARK: - Generate CA Positions (Hermite Interpolation)

    private static func generateCAPositions(
        count n: Int,
        anchors: Anchors
    ) -> [SIMD3<Float>] {
        guard n > 0 else { return [] }

        if anchors.hasBothAnchors {
            return interpolateBetweenAnchors(count: n, anchors: anchors)
        } else if let prevC = anchors.prevC, let prevCA = anchors.prevCA {
            return extendFromPrevious(count: n, cPos: prevC.position, caPos: prevCA.position)
        } else if let nextN = anchors.nextN, let nextCA = anchors.nextCA {
            return extendFromNext(count: n, nPos: nextN.position, caPos: nextCA.position)
        }

        return []
    }

    private static func interpolateBetweenAnchors(
        count n: Int,
        anchors: Anchors
    ) -> [SIMD3<Float>] {
        guard let prevC = anchors.prevC,
              let nextN = anchors.nextN else { return [] }

        let p0 = prevC.position
        let p1 = nextN.position
        let gapDist = simd_length(p1 - p0)
        let extendedLength = Float(n) * caCADistance

        // Check physical feasibility
        if gapDist > extendedLength + 2.0 * caCADistance {
            return []
        }

        // Compute tangent directions
        let t0: SIMD3<Float>
        if let prevCA = anchors.prevCA {
            t0 = simd_normalize(prevC.position - prevCA.position) * extendedLength * 0.5
        } else {
            t0 = simd_normalize(p1 - p0) * extendedLength * 0.5
        }

        let t1: SIMD3<Float>
        if let nextCA = anchors.nextCA {
            t1 = simd_normalize(nextCA.position - nextN.position) * extendedLength * 0.5
        } else {
            t1 = simd_normalize(p1 - p0) * extendedLength * 0.5
        }

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n)

        for i in 0..<n {
            let t = Float(i + 1) / Float(n + 1)
            var pos = hermiteInterpolation(p0: p0, t0: t0, p1: p1, t1: t1, t: t)

            // Add small perpendicular perturbation to avoid collinear backbone
            let perp = arbitraryPerpendicular(to: simd_normalize(p1 - p0))
            let phase = Float(i) * 2.0 * .pi / Float(max(n, 2))
            pos += perp * sin(phase) * 0.3 + simd_cross(perp, simd_normalize(p1 - p0)) * cos(phase) * 0.3

            positions.append(pos)
        }

        return positions
    }

    private static func extendFromPrevious(
        count n: Int,
        cPos: SIMD3<Float>,
        caPos: SIMD3<Float>
    ) -> [SIMD3<Float>] {
        let dir = simd_normalize(cPos - caPos)
        let perp = arbitraryPerpendicular(to: dir)
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n)

        for i in 0..<n {
            let offset: Float = Float(i + 1) * caCADistance
            let phase: Float = Float(i) * 0.7
            let mainOffset: SIMD3<Float> = dir * (offset * 0.85)
            let perpOffset: SIMD3<Float> = perp * sin(phase) * 0.5
            let crossOffset: SIMD3<Float> = simd_cross(perp, dir) * cos(phase) * 0.5
            let pos: SIMD3<Float> = cPos + mainOffset + perpOffset + crossOffset
            positions.append(pos)
        }

        return positions
    }

    private static func extendFromNext(
        count n: Int,
        nPos: SIMD3<Float>,
        caPos: SIMD3<Float>
    ) -> [SIMD3<Float>] {
        let dir = simd_normalize(nPos - caPos)
        let perp = arbitraryPerpendicular(to: dir)
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n)

        for i in 0..<n {
            let offset: Float = Float(n - i) * caCADistance
            let phase: Float = Float(i) * 0.7
            let mainOffset: SIMD3<Float> = dir * (offset * 0.85)
            let perpOffset: SIMD3<Float> = perp * sin(phase) * 0.5
            let crossOffset: SIMD3<Float> = simd_cross(perp, dir) * cos(phase) * 0.5
            let pos: SIMD3<Float> = nPos + mainOffset + perpOffset + crossOffset
            positions.append(pos)
        }

        return positions
    }

    // MARK: - Sidechain Placement from CCD Templates

    private static func addSidechains(
        residueNames: [String],
        backbones: [ResidueBackbone],
        newAtoms: inout [Atom],
        newBonds: inout [Bond],
        chainID: String,
        gapStart: Int
    ) -> Int {
        var sidechainAtomsAdded = 0

        for (i, resName) in residueNames.enumerated() {
            // GLY has no sidechain heavy atoms beyond backbone
            if resName == "GLY" { continue }

            guard let template = ProteinResidueReferenceTemplateStore.template(for: resName) else {
                continue
            }

            let bb = backbones[i]
            let nPos = newAtoms[bb.nIdx].position
            let caPos = newAtoms[bb.caIdx].position
            let cPos = newAtoms[bb.cIdx].position

            // Build overlay transform from placed backbone → template ideal positions
            var refPoints: [SIMD3<Float>] = []
            var movPoints: [SIMD3<Float>] = []

            if let tN = template.atom(named: "N") {
                refPoints.append(nPos)
                movPoints.append(tN.idealPosition)
            }
            if let tCA = template.atom(named: "CA") {
                refPoints.append(caPos)
                movPoints.append(tCA.idealPosition)
            }
            if let tC = template.atom(named: "C") {
                refPoints.append(cPos)
                movPoints.append(tC.idealPosition)
            }

            guard refPoints.count >= 3 else { continue }

            let transform = computeOverlayTransform(referencePoints: refPoints, movingPoints: movPoints)

            // Get sidechain atom names (exclude backbone: N, CA, C, O, and hydrogens and leaving atoms)
            let backboneNames: Set<String> = ["N", "CA", "C", "O", "OXT", "H", "HA"]
            let sidechainAtomNames = template.heavyAtomNames.filter { name in
                !backboneNames.contains(name) &&
                !(template.atom(named: name)?.isLeaving ?? false) &&
                !(template.atom(named: name)?.isCTerminal ?? false) &&
                !(template.atom(named: name)?.isNTerminal ?? false)
            }

            guard !sidechainAtomNames.isEmpty else { continue }

            // Map template atom name → local index for bond creation
            var templateNameToLocalIdx: [String: Int] = [:]
            templateNameToLocalIdx["N"] = bb.nIdx
            templateNameToLocalIdx["CA"] = bb.caIdx
            templateNameToLocalIdx["C"] = bb.cIdx
            templateNameToLocalIdx["O"] = bb.oIdx

            let residueSeq = gapStart + i

            for atomName in sidechainAtomNames {
                guard let atomTemplate = template.atom(named: atomName),
                      !atomTemplate.isHydrogen else { continue }

                let pos = transform.apply(atomTemplate.idealPosition)
                guard pos.x.isFinite && pos.y.isFinite && pos.z.isFinite else { continue }

                let localIdx = newAtoms.count
                let atom = Atom(
                    id: 0, element: atomTemplate.element, position: pos, name: atomName,
                    residueName: resName, residueSeq: residueSeq, chainID: chainID,
                    charge: 0, formalCharge: atomTemplate.formalCharge, isHetAtom: false,
                    occupancy: 1.0, tempFactor: 50.0
                )
                newAtoms.append(atom)
                templateNameToLocalIdx[atomName] = localIdx
                sidechainAtomsAdded += 1
            }

            // Add bonds from template
            for bondTemplate in template.bonds {
                let name1 = template.canonicalAtomName(bondTemplate.atomName1)
                let name2 = template.canonicalAtomName(bondTemplate.atomName2)

                guard let idx1 = templateNameToLocalIdx[name1],
                      let idx2 = templateNameToLocalIdx[name2] else { continue }

                // Don't duplicate backbone bonds
                let isBackboneBond = backboneNames.contains(name1) && backboneNames.contains(name2)
                if isBackboneBond { continue }

                newBonds.append(Bond(
                    id: newBonds.count,
                    atomIndex1: idx1,
                    atomIndex2: idx2,
                    order: bondTemplate.order
                ))
            }
        }

        return sidechainAtomsAdded
    }

    // MARK: - Geometry Utilities

    private static func hermiteInterpolation(
        p0: SIMD3<Float>, t0: SIMD3<Float>,
        p1: SIMD3<Float>, t1: SIMD3<Float>,
        t: Float
    ) -> SIMD3<Float> {
        let t2 = t * t
        let t3 = t2 * t
        let h00 = 2 * t3 - 3 * t2 + 1
        let h10 = t3 - 2 * t2 + t
        let h01 = -2 * t3 + 3 * t2
        let h11 = t3 - t2
        return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
    }

    private static func arbitraryPerpendicular(to v: SIMD3<Float>) -> SIMD3<Float> {
        let absV = SIMD3<Float>(abs(v.x), abs(v.y), abs(v.z))
        let candidate: SIMD3<Float>
        if absV.x <= absV.y && absV.x <= absV.z {
            candidate = SIMD3<Float>(1, 0, 0)
        } else if absV.y <= absV.z {
            candidate = SIMD3<Float>(0, 1, 0)
        } else {
            candidate = SIMD3<Float>(0, 0, 1)
        }
        let perp = simd_cross(v, candidate)
        let len = simd_length(perp)
        return len > 1e-6 ? perp / len : SIMD3<Float>(0, 1, 0)
    }

    private static func rotateAround(axis: SIMD3<Float>, vector v: SIMD3<Float>, angle: Float) -> SIMD3<Float> {
        let k = simd_normalize(axis)
        let cosA = cos(angle)
        let sinA = sin(angle)
        return v * cosA + simd_cross(k, v) * sinA + k * simd_dot(k, v) * (1 - cosA)
    }

    // MARK: - Overlay Transform (Kabsch)

    private struct SimpleOverlayTransform {
        let rotation: simd_float3x3
        let movingCenter: SIMD3<Float>
        let referenceCenter: SIMD3<Float>

        func apply(_ point: SIMD3<Float>) -> SIMD3<Float> {
            (rotation * (point - movingCenter)) + referenceCenter
        }
    }

    private static func computeOverlayTransform(
        referencePoints: [SIMD3<Float>],
        movingPoints: [SIMD3<Float>]
    ) -> SimpleOverlayTransform {
        let refCenter = referencePoints.reduce(.zero, +) / Float(referencePoints.count)
        let movCenter = movingPoints.reduce(.zero, +) / Float(movingPoints.count)

        if referencePoints.count <= 1 {
            return SimpleOverlayTransform(rotation: matrix_identity_float3x3, movingCenter: movCenter, referenceCenter: refCenter)
        }

        var cov = simd_float3x3()
        for (r, m) in zip(referencePoints, movingPoints) {
            let x = r - refCenter
            let y = m - movCenter
            cov[0, 0] += y.x * x.x; cov[0, 1] += y.x * x.y; cov[0, 2] += y.x * x.z
            cov[1, 0] += y.y * x.x; cov[1, 1] += y.y * x.y; cov[1, 2] += y.y * x.z
            cov[2, 0] += y.z * x.x; cov[2, 1] += y.z * x.y; cov[2, 2] += y.z * x.z
        }

        let trace = cov[0, 0] + cov[1, 1] + cov[2, 2]
        let rows: [SIMD4<Float>] = [
            SIMD4(trace, cov[1, 2] - cov[2, 1], cov[2, 0] - cov[0, 2], cov[0, 1] - cov[1, 0]),
            SIMD4(cov[1, 2] - cov[2, 1], cov[0, 0] - cov[1, 1] - cov[2, 2], cov[0, 1] + cov[1, 0], cov[2, 0] + cov[0, 2]),
            SIMD4(cov[2, 0] - cov[0, 2], cov[0, 1] + cov[1, 0], -cov[0, 0] + cov[1, 1] - cov[2, 2], cov[1, 2] + cov[2, 1]),
            SIMD4(cov[0, 1] - cov[1, 0], cov[2, 0] + cov[0, 2], cov[1, 2] + cov[2, 1], -cov[0, 0] - cov[1, 1] + cov[2, 2])
        ]

        var q = SIMD4<Float>(1, 0, 0, 0)
        for _ in 0..<32 {
            let next = SIMD4<Float>(
                simd_dot(rows[0], q), simd_dot(rows[1], q),
                simd_dot(rows[2], q), simd_dot(rows[3], q)
            )
            let len = simd_length(next)
            guard len > 1e-8 else { break }
            q = next / len
        }

        let rot = quaternionToMatrix(q)
        return SimpleOverlayTransform(rotation: rot, movingCenter: movCenter, referenceCenter: refCenter)
    }

    private static func quaternionToMatrix(_ q: SIMD4<Float>) -> simd_float3x3 {
        let nq = q / max(simd_length(q), 1e-8)
        let w = nq.x, x = nq.y, y = nq.z, z = nq.w
        return simd_float3x3(rows: [
            SIMD3(1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
            SIMD3(2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
            SIMD3(2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y))
        ])
    }
}

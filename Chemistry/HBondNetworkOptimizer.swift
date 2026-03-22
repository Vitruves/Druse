import Foundation
import Metal
import simd

/// Phase 4 of protein preparation: hydrogen bond network optimization.
///
/// Identifies moveable hydrogen groups (rotatable OH/SH/NH3+, flippable Asn/Gln/His),
/// enumerates their discrete states, builds a conflict graph of mutually interacting
/// groups, and partitions into independent cliques for scoring (Phase 4.2).
///
/// Algorithm based on: Word et al., J Mol Biol 1999 (Reduce)
enum HBondNetworkOptimizer {

    // MARK: - Public types

    enum MoveableGroupKind: String, Sendable {
        case rotatableOH   // Ser OG-HG, Thr OG1-HG1, Tyr OH-HH
        case rotatableSH   // Cys SG-HG
        case rotatableNH3  // Lys NZ-HZ1/HZ2/HZ3 (3-fold symmetric)
        case flipAmide     // Asn OD1↔ND2, Gln OE1↔NE2
        case flipHis       // His ring: 6 states (3 tautomers × 2 flip orientations)
    }

    struct ResidueID: Hashable, Sendable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
    }

    struct GroupState: Sendable {
        let label: String
        let atomPositions: [String: SIMD3<Float>]
        let penalty: Float
        let isFlipped: Bool
    }

    struct MoveableGroup: Sendable {
        let kind: MoveableGroupKind
        let residueID: ResidueID
        let moveableAtomIndices: [Int]
        let allAtomIndices: [Int]
        let states: [GroupState]
    }

    struct ConflictGraph: Sendable {
        let groups: [MoveableGroup]
        let adjacency: [[Int]]
        let cliques: [[Int]]
        let singletons: [Int]
    }

    struct FlipDecision: Sendable {
        let residueID: ResidueID
        let kind: MoveableGroupKind
        let chosenStateIndex: Int
        let chosenLabel: String
        let isFlipped: Bool
        let score: Float
        let originalScore: Float
        let deltaEnergy: Float
    }

    struct NetworkReport: Sendable {
        var moveableGroups: Int = 0
        var rotatableOH: Int = 0
        var rotatableSH: Int = 0
        var rotatableNH3: Int = 0
        var flipAmide: Int = 0
        var flipHis: Int = 0
        var cliques: Int = 0
        var singletons: Int = 0
        var largestCliqueSize: Int = 0
        var flipsAccepted: Int = 0
        var rotationsOptimized: Int = 0
        var totalEnergyImprovement: Float = 0
        var decisions: [FlipDecision] = []
    }

    // MARK: - Entry point

    static func enumerateNetwork(
        atoms: [Atom],
        bonds: [Bond],
        predictions: [Protonation.ResiduePrediction]
    ) -> (graph: ConflictGraph, report: NetworkReport) {
        let adjacencyList = buildBondAdjacency(atoms: atoms, bonds: bonds)
        let residueGroups = groupAtomsByResidue(atoms: atoms)
        let predictionMap = buildPredictionMap(predictions: predictions)

        var groups: [MoveableGroup] = []

        for (residueID, atomIndices) in residueGroups {
            let detected = detectMoveableGroups(
                residueID: residueID,
                atomIndices: atomIndices,
                atoms: atoms,
                adjacency: adjacencyList,
                predictionMap: predictionMap
            )
            groups.append(contentsOf: detected)
        }

        let graph = buildConflictGraph(groups: groups, atoms: atoms)

        var report = NetworkReport()
        report.moveableGroups = groups.count
        for group in groups {
            switch group.kind {
            case .rotatableOH:  report.rotatableOH += 1
            case .rotatableSH:  report.rotatableSH += 1
            case .rotatableNH3: report.rotatableNH3 += 1
            case .flipAmide:    report.flipAmide += 1
            case .flipHis:      report.flipHis += 1
            }
        }
        report.cliques = graph.cliques.count
        report.singletons = graph.singletons.count
        report.largestCliqueSize = graph.cliques.map(\.count).max() ?? 0

        return (graph, report)
    }

    // MARK: - Residue grouping

    private static func groupAtomsByResidue(
        atoms: [Atom]
    ) -> [(ResidueID, [Int])] {
        var map: [ResidueID: [Int]] = [:]
        var order: [ResidueID] = []
        for (index, atom) in atoms.enumerated() {
            guard !atom.isHetAtom || atom.residueName == "ACE" || atom.residueName == "NME" else { continue }
            let key = ResidueID(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: atom.residueName
            )
            if map[key] == nil { order.append(key) }
            map[key, default: []].append(index)
        }
        return order.map { ($0, map[$0]!) }
    }

    private static func buildBondAdjacency(
        atoms: [Atom],
        bonds: [Bond]
    ) -> [[Int]] {
        var adj = [[Int]](repeating: [], count: atoms.count)
        for bond in bonds {
            adj[bond.atomIndex1].append(bond.atomIndex2)
            adj[bond.atomIndex2].append(bond.atomIndex1)
        }
        return adj
    }

    private static func buildPredictionMap(
        predictions: [Protonation.ResiduePrediction]
    ) -> [ResidueID: Protonation.ResiduePrediction] {
        var map: [ResidueID: Protonation.ResiduePrediction] = [:]
        for p in predictions {
            let key = ResidueID(chainID: p.chainID, residueSeq: p.residueSeq, residueName: p.residueName)
            map[key] = p
        }
        return map
    }

    // MARK: - Moveable group detection

    private static func detectMoveableGroups(
        residueID: ResidueID,
        atomIndices: [Int],
        atoms: [Atom],
        adjacency: [[Int]],
        predictionMap: [ResidueID: Protonation.ResiduePrediction]
    ) -> [MoveableGroup] {
        let nameToIndex = buildNameMap(atomIndices: atomIndices, atoms: atoms)

        switch residueID.residueName {
        case "SER":
            return makeRotatableOH(
                residueID: residueID, atoms: atoms, adjacency: adjacency,
                nameToIndex: nameToIndex,
                parentName: "OG", hydrogenName: "HG",
                axis1Name: "CB", axis2Name: "OG"
            )
        case "THR":
            return makeRotatableOH(
                residueID: residueID, atoms: atoms, adjacency: adjacency,
                nameToIndex: nameToIndex,
                parentName: "OG1", hydrogenName: "HG1",
                axis1Name: "CB", axis2Name: "OG1"
            )
        case "TYR":
            return makeRotatableOH(
                residueID: residueID, atoms: atoms, adjacency: adjacency,
                nameToIndex: nameToIndex,
                parentName: "OH", hydrogenName: "HH",
                axis1Name: "CZ", axis2Name: "OH"
            )
        case "CYS":
            if let prediction = predictionMap[residueID],
               prediction.state == .protonated {
                return makeRotatableSH(
                    residueID: residueID, atoms: atoms, adjacency: adjacency,
                    nameToIndex: nameToIndex
                )
            }
            return []
        case "LYS":
            if let prediction = predictionMap[residueID],
               prediction.state == .protonated {
                return makeRotatableNH3(
                    residueID: residueID, atoms: atoms,
                    nameToIndex: nameToIndex
                )
            }
            return []
        case "ASN":
            return makeFlipAmide(
                residueID: residueID, atoms: atoms,
                nameToIndex: nameToIndex,
                oxygenName: "OD1", nitrogenName: "ND2",
                carbonName: "CG", anchorName: "CB",
                h1Name: "HD21", h2Name: "HD22"
            )
        case "GLN":
            return makeFlipAmide(
                residueID: residueID, atoms: atoms,
                nameToIndex: nameToIndex,
                oxygenName: "OE1", nitrogenName: "NE2",
                carbonName: "CD", anchorName: "CG",
                h1Name: "HE21", h2Name: "HE22"
            )
        case "HIS":
            return makeFlipHis(
                residueID: residueID, atoms: atoms,
                nameToIndex: nameToIndex,
                predictionMap: predictionMap
            )
        default:
            return []
        }
    }

    private static func buildNameMap(
        atomIndices: [Int],
        atoms: [Atom]
    ) -> [String: Int] {
        var map: [String: Int] = [:]
        for index in atomIndices {
            let name = atoms[index].name.trimmingCharacters(in: .whitespaces).uppercased()
            map[name] = index
        }
        return map
    }

    // MARK: - Rotatable OH (Ser, Thr, Tyr)

    private static func makeRotatableOH(
        residueID: ResidueID,
        atoms: [Atom],
        adjacency: [[Int]],
        nameToIndex: [String: Int],
        parentName: String,
        hydrogenName: String,
        axis1Name: String,
        axis2Name: String
    ) -> [MoveableGroup] {
        guard let hIndex = nameToIndex[hydrogenName],
              let parentIndex = nameToIndex[parentName],
              let axis1Index = nameToIndex[axis1Name] else {
            return []
        }
        let axis2Index = nameToIndex[axis2Name] ?? parentIndex

        let axisOrigin = atoms[axis1Index].position
        let axisEnd = atoms[axis2Index].position
        let hPosition = atoms[hIndex].position

        let states = enumerateDonorRotations(
            hydrogenPosition: hPosition,
            axisOrigin: axisOrigin,
            axisEnd: axisEnd,
            hydrogenName: hydrogenName,
            coarseStep: 10,
            nearbyAcceptors: findNearbyAcceptors(
                center: atoms[parentIndex].position,
                radius: 3.5,
                atoms: atoms,
                excludeIndices: Set(nameToIndex.values)
            )
        )

        let allIndices = Array(nameToIndex.values)
        return [MoveableGroup(
            kind: .rotatableOH,
            residueID: residueID,
            moveableAtomIndices: [hIndex],
            allAtomIndices: allIndices,
            states: states
        )]
    }

    // MARK: - Rotatable SH (Cys)

    private static func makeRotatableSH(
        residueID: ResidueID,
        atoms: [Atom],
        adjacency: [[Int]],
        nameToIndex: [String: Int]
    ) -> [MoveableGroup] {
        guard let hIndex = nameToIndex["HG"],
              let sgIndex = nameToIndex["SG"],
              let cbIndex = nameToIndex["CB"] else {
            return []
        }

        let states = enumerateDonorRotations(
            hydrogenPosition: atoms[hIndex].position,
            axisOrigin: atoms[cbIndex].position,
            axisEnd: atoms[sgIndex].position,
            hydrogenName: "HG",
            coarseStep: 10,
            nearbyAcceptors: findNearbyAcceptors(
                center: atoms[sgIndex].position,
                radius: 3.5,
                atoms: atoms,
                excludeIndices: Set(nameToIndex.values)
            )
        )

        let allIndices = Array(nameToIndex.values)
        return [MoveableGroup(
            kind: .rotatableSH,
            residueID: residueID,
            moveableAtomIndices: [hIndex],
            allAtomIndices: allIndices,
            states: states
        )]
    }

    // MARK: - Rotatable NH3+ (Lys)

    private static func makeRotatableNH3(
        residueID: ResidueID,
        atoms: [Atom],
        nameToIndex: [String: Int]
    ) -> [MoveableGroup] {
        guard let nzIndex = nameToIndex["NZ"],
              let ceIndex = nameToIndex["CE"] else {
            return []
        }

        let hNames = ["HZ1", "HZ2", "HZ3"]
        let hIndices = hNames.compactMap { nameToIndex[$0] }
        guard hIndices.count == 3 else { return [] }

        let axisOrigin = atoms[ceIndex].position
        let axisEnd = atoms[nzIndex].position

        let states = enumerate3FoldRotations(
            hydrogenPositions: hIndices.map { atoms[$0].position },
            hydrogenNames: hNames,
            axisOrigin: axisOrigin,
            axisEnd: axisEnd,
            coarseStep: 30
        )

        let allIndices = Array(nameToIndex.values)
        return [MoveableGroup(
            kind: .rotatableNH3,
            residueID: residueID,
            moveableAtomIndices: hIndices,
            allAtomIndices: allIndices,
            states: states
        )]
    }

    // MARK: - Flip amide (Asn, Gln)

    private static func makeFlipAmide(
        residueID: ResidueID,
        atoms: [Atom],
        nameToIndex: [String: Int],
        oxygenName: String,
        nitrogenName: String,
        carbonName: String,
        anchorName: String,
        h1Name: String,
        h2Name: String
    ) -> [MoveableGroup] {
        guard let oIndex = nameToIndex[oxygenName],
              let nIndex = nameToIndex[nitrogenName],
              let cIndex = nameToIndex[carbonName],
              let anchorIndex = nameToIndex[anchorName] else {
            return []
        }
        let h1Index = nameToIndex[h1Name]
        let h2Index = nameToIndex[h2Name]

        let oPos = atoms[oIndex].position
        let nPos = atoms[nIndex].position
        let cPos = atoms[cIndex].position
        let anchorPos = atoms[anchorIndex].position

        // Rotation axis: anchor → carbon (the bond we rotate around)
        let axisOrigin = anchorPos
        let axisEnd = cPos

        // State 0: original (O at oxygenName, N at nitrogenName)
        var originalPositions: [String: SIMD3<Float>] = [
            oxygenName: oPos,
            nitrogenName: nPos,
        ]
        if let h1 = h1Index { originalPositions[h1Name] = atoms[h1].position }
        if let h2 = h2Index { originalPositions[h2Name] = atoms[h2].position }

        // State 1: flipped — 180° rotation of terminal group around anchor-carbon axis
        let flippedPositions = flipTerminalGroup(
            positions: originalPositions,
            axisOrigin: axisOrigin,
            axisEnd: axisEnd,
            oxygenName: oxygenName,
            nitrogenName: nitrogenName,
            carbonPosition: cPos,
            h1Name: h1Name,
            h2Name: h2Name
        )

        let state0 = GroupState(
            label: "original",
            atomPositions: originalPositions,
            penalty: 0.0,
            isFlipped: false
        )
        let state1 = GroupState(
            label: "flipped",
            atomPositions: flippedPositions,
            penalty: 0.5,
            isFlipped: true
        )

        var moveableIndices = [oIndex, nIndex]
        if let h1 = h1Index { moveableIndices.append(h1) }
        if let h2 = h2Index { moveableIndices.append(h2) }
        let allIndices = Array(nameToIndex.values)

        return [MoveableGroup(
            kind: .flipAmide,
            residueID: residueID,
            moveableAtomIndices: moveableIndices,
            allAtomIndices: allIndices,
            states: [state0, state1]
        )]
    }

    // MARK: - Flip His (6 states)

    private static func makeFlipHis(
        residueID: ResidueID,
        atoms: [Atom],
        nameToIndex: [String: Int],
        predictionMap: [ResidueID: Protonation.ResiduePrediction]
    ) -> [MoveableGroup] {
        // Required ring atoms
        guard let cgIndex = nameToIndex["CG"],
              let nd1Index = nameToIndex["ND1"],
              let cd2Index = nameToIndex["CD2"],
              let ce1Index = nameToIndex["CE1"],
              let ne2Index = nameToIndex["NE2"],
              let cbIndex = nameToIndex["CB"] else {
            return []
        }

        let hd1Index = nameToIndex["HD1"]
        let he2Index = nameToIndex["HE2"]
        let he1Index = nameToIndex["HE1"]
        let hd2Index = nameToIndex["HD2"]

        let cgPos = atoms[cgIndex].position
        let nd1Pos = atoms[nd1Index].position
        let cd2Pos = atoms[cd2Index].position
        let ce1Pos = atoms[ce1Index].position
        let ne2Pos = atoms[ne2Index].position
        let cbPos = atoms[cbIndex].position

        // Rotation axis for flip: CB → CG
        let axisOrigin = cbPos
        let axisEnd = cgPos

        // Place protons at ideal geometry on ring nitrogens
        let hd1Pos = hd1Index.map { atoms[$0].position }
            ?? placeRingHydrogen(on: nd1Pos, ring: [cgPos, ce1Pos], distance: 0.86)
        let he2Pos = he2Index.map { atoms[$0].position }
            ?? placeRingHydrogen(on: ne2Pos, ring: [cd2Pos, ce1Pos], distance: 0.86)

        // Build the 6 states:
        // Unflipped: δ-tautomer (HD1 only), ε-tautomer (HE2 only), doubly protonated (both)
        // Flipped:   same 3 but with ring rotated 180°

        var states: [GroupState] = []

        // --- Unflipped states ---
        let unflippedRing: [String: SIMD3<Float>] = [
            "ND1": nd1Pos, "CD2": cd2Pos, "CE1": ce1Pos, "NE2": ne2Pos,
        ]

        // δ-tautomer: HD1 present, no HE2
        var deltaPosU = unflippedRing
        deltaPosU["HD1"] = hd1Pos
        if let hi = he1Index { deltaPosU["HE1"] = atoms[hi].position }
        if let hi = hd2Index { deltaPosU["HD2"] = atoms[hi].position }
        states.append(GroupState(
            label: "δ-tautomer",
            atomPositions: deltaPosU,
            penalty: 0.0,
            isFlipped: false
        ))

        // ε-tautomer: HE2 present, no HD1
        var epsilonPosU = unflippedRing
        epsilonPosU["HE2"] = he2Pos
        if let hi = he1Index { epsilonPosU["HE1"] = atoms[hi].position }
        if let hi = hd2Index { epsilonPosU["HD2"] = atoms[hi].position }
        states.append(GroupState(
            label: "ε-tautomer",
            atomPositions: epsilonPosU,
            penalty: 0.0,
            isFlipped: false
        ))

        // Doubly protonated: both HD1 and HE2
        var doublyPosU = unflippedRing
        doublyPosU["HD1"] = hd1Pos
        doublyPosU["HE2"] = he2Pos
        if let hi = he1Index { doublyPosU["HE1"] = atoms[hi].position }
        if let hi = hd2Index { doublyPosU["HD2"] = atoms[hi].position }
        states.append(GroupState(
            label: "doubly-protonated",
            atomPositions: doublyPosU,
            penalty: 0.05,
            isFlipped: false
        ))

        // --- Flipped states (180° rotation around CB-CG axis) ---
        // Flip swaps ND1↔CD2 and CE1↔NE2 positions
        let flippedRing: [String: SIMD3<Float>] = [
            "ND1": rotatePoint(cd2Pos, around: axisOrigin, axis: axisEnd, angle: .pi),
            "CD2": rotatePoint(nd1Pos, around: axisOrigin, axis: axisEnd, angle: .pi),
            "CE1": rotatePoint(ne2Pos, around: axisOrigin, axis: axisEnd, angle: .pi),
            "NE2": rotatePoint(ce1Pos, around: axisOrigin, axis: axisEnd, angle: .pi),
        ]

        let hd1PosFlipped = placeRingHydrogen(
            on: flippedRing["ND1"]!,
            ring: [flippedRing["CE1"]!, cgPos],
            distance: 0.86
        )
        let he2PosFlipped = placeRingHydrogen(
            on: flippedRing["NE2"]!,
            ring: [flippedRing["CD2"]!, flippedRing["CE1"]!],
            distance: 0.86
        )

        // Flipped carbon H positions
        let he1PosFlipped: SIMD3<Float>? = he1Index != nil ? placeRingHydrogen(
            on: flippedRing["CE1"]!,
            ring: [flippedRing["ND1"]!, flippedRing["NE2"]!],
            distance: 0.93
        ) : nil
        let hd2PosFlipped: SIMD3<Float>? = hd2Index != nil ? placeRingHydrogen(
            on: flippedRing["CD2"]!,
            ring: [flippedRing["NE2"]!, cgPos],
            distance: 0.93
        ) : nil

        // Flipped δ-tautomer
        var deltaPosF = flippedRing
        deltaPosF["HD1"] = hd1PosFlipped
        if let p = he1PosFlipped { deltaPosF["HE1"] = p }
        if let p = hd2PosFlipped { deltaPosF["HD2"] = p }
        states.append(GroupState(
            label: "flipped δ-tautomer",
            atomPositions: deltaPosF,
            penalty: 0.5,
            isFlipped: true
        ))

        // Flipped ε-tautomer
        var epsilonPosF = flippedRing
        epsilonPosF["HE2"] = he2PosFlipped
        if let p = he1PosFlipped { epsilonPosF["HE1"] = p }
        if let p = hd2PosFlipped { epsilonPosF["HD2"] = p }
        states.append(GroupState(
            label: "flipped ε-tautomer",
            atomPositions: epsilonPosF,
            penalty: 0.5,
            isFlipped: true
        ))

        // Flipped doubly protonated
        var doublyPosF = flippedRing
        doublyPosF["HD1"] = hd1PosFlipped
        doublyPosF["HE2"] = he2PosFlipped
        if let p = he1PosFlipped { doublyPosF["HE1"] = p }
        if let p = hd2PosFlipped { doublyPosF["HD2"] = p }
        states.append(GroupState(
            label: "flipped doubly-protonated",
            atomPositions: doublyPosF,
            penalty: 0.55,
            isFlipped: true
        ))

        var moveableIndices = [nd1Index, cd2Index, ce1Index, ne2Index]
        if let i = hd1Index { moveableIndices.append(i) }
        if let i = he2Index { moveableIndices.append(i) }
        if let i = he1Index { moveableIndices.append(i) }
        if let i = hd2Index { moveableIndices.append(i) }
        let allIndices = Array(nameToIndex.values)

        return [MoveableGroup(
            kind: .flipHis,
            residueID: residueID,
            moveableAtomIndices: moveableIndices,
            allAtomIndices: allIndices,
            states: states
        )]
    }

    // MARK: - Rotation enumeration (donor OH/SH)

    /// Enumerates discrete rotational states for a single rotatable hydrogen.
    /// Follows the Reduce RotDonor strategy: sample angles toward nearby acceptors
    /// plus one least-clash angle, merge duplicates within half the coarse step.
    private static func enumerateDonorRotations(
        hydrogenPosition: SIMD3<Float>,
        axisOrigin: SIMD3<Float>,
        axisEnd: SIMD3<Float>,
        hydrogenName: String,
        coarseStep: Int,
        nearbyAcceptors: [SIMD3<Float>]
    ) -> [GroupState] {
        let axis = normalized(axisEnd - axisOrigin)
        let bondVector = hydrogenPosition - axisEnd
        let bondLength = simd_length(bondVector)

        // Project bond vector onto plane perpendicular to axis to get reference angle
        let projected = bondVector - simd_dot(bondVector, axis) * axis
        let projLength = simd_length(projected)
        guard projLength > 1e-6, bondLength > 1e-6 else {
            // H is on the axis — can't rotate meaningfully
            return [GroupState(
                label: "fixed",
                atomPositions: [hydrogenName: hydrogenPosition],
                penalty: 0.0,
                isFlipped: false
            )]
        }

        let stepRad = Float(coarseStep) * .pi / 180.0
        let totalSteps = Int(360.0 / Float(coarseStep))

        // Collect candidate angles
        var candidateAngles: [Float] = []

        // For each nearby acceptor, compute the dihedral angle that points H toward it
        for acceptorPos in nearbyAcceptors {
            let toAcceptor = acceptorPos - axisEnd
            let projAcceptor = toAcceptor - simd_dot(toAcceptor, axis) * axis
            guard simd_length(projAcceptor) > 1e-6 else { continue }

            let cosAngle = simd_dot(
                simd_normalize(projected),
                simd_normalize(projAcceptor)
            )
            let crossVal = simd_dot(
                simd_cross(simd_normalize(projected), simd_normalize(projAcceptor)),
                axis
            )
            let angle = atan2(crossVal, cosAngle)
            candidateAngles.append(angle)
        }

        // If no acceptors found, sample uniformly
        if candidateAngles.isEmpty {
            for step in 0..<totalSteps {
                candidateAngles.append(Float(step) * stepRad)
            }
        } else {
            // Also add one "least-clash" angle: midpoint between acceptor-directed angles
            // (simple heuristic: scan all coarse angles and pick one farthest from any acceptor angle)
            var bestFarAngle: Float = 0
            var bestMinDist: Float = -1
            for step in 0..<totalSteps {
                let testAngle = Float(step) * stepRad
                let minDist = candidateAngles.map { abs(angleDifference(testAngle, $0)) }.min() ?? .pi
                if minDist > bestMinDist {
                    bestMinDist = minDist
                    bestFarAngle = testAngle
                }
            }
            candidateAngles.append(bestFarAngle)
        }

        // Deduplicate within half the coarse step
        let mergeThreshold = stepRad * 0.5
        var uniqueAngles: [Float] = []
        for angle in candidateAngles {
            let normalized = normalizeAngle(angle)
            let isDuplicate = uniqueAngles.contains { existing in
                abs(angleDifference(normalized, existing)) < mergeThreshold
            }
            if !isDuplicate {
                uniqueAngles.append(normalized)
            }
        }

        // Build states
        var states: [GroupState] = []
        for (index, angle) in uniqueAngles.enumerated() {
            let rotatedH = rotatePoint(
                hydrogenPosition,
                around: axisOrigin,
                axis: axisEnd,
                angle: angle
            )
            states.append(GroupState(
                label: index == 0 ? "original" : "rot \(Int(angle * 180.0 / .pi))°",
                atomPositions: [hydrogenName: rotatedH],
                penalty: index == 0 ? 0.0 : 0.0,
                isFlipped: false
            ))
        }

        // Ensure at least the original orientation
        if states.isEmpty {
            states.append(GroupState(
                label: "original",
                atomPositions: [hydrogenName: hydrogenPosition],
                penalty: 0.0,
                isFlipped: false
            ))
        }

        return states
    }

    // MARK: - 3-fold rotation enumeration (NH3+, methyl)

    private static func enumerate3FoldRotations(
        hydrogenPositions: [SIMD3<Float>],
        hydrogenNames: [String],
        axisOrigin: SIMD3<Float>,
        axisEnd: SIMD3<Float>,
        coarseStep: Int
    ) -> [GroupState] {
        // 3-fold symmetry: only need to scan 120° / coarseStep orientations
        let scanRange: Float = 120.0
        let stepDeg = Float(coarseStep)
        let numSteps = max(1, Int(scanRange / stepDeg))

        var states: [GroupState] = []
        for step in 0..<numSteps {
            let angleDeg = Float(step) * stepDeg
            let angleRad = angleDeg * .pi / 180.0

            var positions: [String: SIMD3<Float>] = [:]
            for (i, name) in hydrogenNames.enumerated() {
                positions[name] = rotatePoint(
                    hydrogenPositions[i],
                    around: axisOrigin,
                    axis: axisEnd,
                    angle: angleRad
                )
            }

            states.append(GroupState(
                label: step == 0 ? "original" : "rot \(Int(angleDeg))°",
                atomPositions: positions,
                penalty: 0.0,
                isFlipped: false
            ))
        }

        return states
    }

    // MARK: - Amide flip geometry

    /// Flips the terminal amide group by 180° around the anchor-carbon axis.
    /// After geometric rotation, swaps atom names (O↔N) so the oxygen is now
    /// where the nitrogen was and vice versa. Re-places hydrogens on the new N position.
    private static func flipTerminalGroup(
        positions: [String: SIMD3<Float>],
        axisOrigin: SIMD3<Float>,
        axisEnd: SIMD3<Float>,
        oxygenName: String,
        nitrogenName: String,
        carbonPosition: SIMD3<Float>,
        h1Name: String,
        h2Name: String
    ) -> [String: SIMD3<Float>] {
        guard let oPos = positions[oxygenName],
              let nPos = positions[nitrogenName] else {
            return positions
        }

        // 180° rotation around anchor-carbon axis
        let oRotated = rotatePoint(oPos, around: axisOrigin, axis: axisEnd, angle: .pi)
        let nRotated = rotatePoint(nPos, around: axisOrigin, axis: axisEnd, angle: .pi)

        // After rotation, the O and N have swapped sides. The atom identities swap:
        // what was the O position becomes the N position and vice versa.
        var flipped: [String: SIMD3<Float>] = [:]
        flipped[oxygenName] = nRotated   // N went to where O should be
        flipped[nitrogenName] = oRotated  // O went to where N should be

        // Re-place hydrogens on the new nitrogen position (which is at oRotated)
        let newNPos = oRotated
        let h1Pos = placeAmideHydrogen(
            nitrogenPosition: newNPos,
            carbonPosition: carbonPosition,
            angle: 120.0,
            dihedral: 0.0,
            distance: 0.86
        )
        let h2Pos = placeAmideHydrogen(
            nitrogenPosition: newNPos,
            carbonPosition: carbonPosition,
            angle: 120.0,
            dihedral: 180.0,
            distance: 0.86
        )
        flipped[h1Name] = h1Pos
        flipped[h2Name] = h2Pos

        return flipped
    }

    // MARK: - Conflict graph construction

    /// Two groups conflict if any of their moveable atoms can come within VDW contact
    /// distance across any combination of states.
    private static let interactionCutoff: Float = 4.0

    static func buildConflictGraph(
        groups: [MoveableGroup],
        atoms: [Atom]
    ) -> ConflictGraph {
        let n = groups.count
        var adjacency = [[Int]](repeating: [], count: n)

        // Collect all possible positions for each group's moveable atoms across all states
        var groupBounds: [(center: SIMD3<Float>, radius: Float)] = []
        for group in groups {
            var allPositions: [SIMD3<Float>] = []
            for state in group.states {
                allPositions.append(contentsOf: state.atomPositions.values)
            }
            // Also include current positions of moveable atoms
            for idx in group.moveableAtomIndices {
                allPositions.append(atoms[idx].position)
            }
            let center = allPositions.reduce(.zero, +) / Float(max(1, allPositions.count))
            let radius = allPositions.map { simd_length($0 - center) }.max() ?? 0
            groupBounds.append((center, radius))
        }

        for i in 0..<n {
            for j in (i + 1)..<n {
                // Quick bounding sphere check
                let dist = simd_length(groupBounds[i].center - groupBounds[j].center)
                let maxReach = groupBounds[i].radius + groupBounds[j].radius + interactionCutoff
                guard dist < maxReach else { continue }

                // Detailed check: any atom pair across any state combination within cutoff?
                if groupsInteract(groups[i], groups[j], atoms: atoms) {
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                }
            }
        }

        // Find connected components (cliques)
        let (cliques, singletons) = findConnectedComponents(adjacency: adjacency, count: n)

        return ConflictGraph(
            groups: groups,
            adjacency: adjacency,
            cliques: cliques,
            singletons: singletons
        )
    }

    private static func groupsInteract(
        _ a: MoveableGroup,
        _ b: MoveableGroup,
        atoms: [Atom]
    ) -> Bool {
        // Check if any moveable atom from group A in any state is within
        // interaction distance of any moveable atom from group B in any state.
        // Also check against fixed (non-moveable) atoms of the other group.

        let aPositions = allPositionsForGroup(a, atoms: atoms)
        let bPositions = allPositionsForGroup(b, atoms: atoms)

        for posA in aPositions {
            for posB in bPositions {
                if simd_length(posA - posB) < interactionCutoff {
                    return true
                }
            }
        }
        return false
    }

    private static func allPositionsForGroup(
        _ group: MoveableGroup,
        atoms: [Atom]
    ) -> [SIMD3<Float>] {
        var positions: [SIMD3<Float>] = []
        // All state positions for moveable atoms
        for state in group.states {
            positions.append(contentsOf: state.atomPositions.values)
        }
        // Current positions of all residue atoms (for context)
        for idx in group.allAtomIndices {
            positions.append(atoms[idx].position)
        }
        return positions
    }

    // MARK: - Connected components

    private static func findConnectedComponents(
        adjacency: [[Int]],
        count: Int
    ) -> (cliques: [[Int]], singletons: [Int]) {
        var visited = [Bool](repeating: false, count: count)
        var cliques: [[Int]] = []
        var singletons: [Int] = []

        for i in 0..<count {
            guard !visited[i] else { continue }

            if adjacency[i].isEmpty {
                singletons.append(i)
                visited[i] = true
                continue
            }

            // BFS to find connected component
            var component: [Int] = []
            var queue: [Int] = [i]
            visited[i] = true

            while !queue.isEmpty {
                let node = queue.removeFirst()
                component.append(node)
                for neighbor in adjacency[node] where !visited[neighbor] {
                    visited[neighbor] = true
                    queue.append(neighbor)
                }
            }

            cliques.append(component)
        }

        return (cliques, singletons)
    }

    // MARK: - Geometry helpers

    private static func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let len = simd_length(v)
        guard len > 1e-8 else { return SIMD3<Float>(1, 0, 0) }
        return v / len
    }

    /// Rotates a point around an axis defined by two points (origin, end) by the given angle.
    static func rotatePoint(
        _ point: SIMD3<Float>,
        around axisOrigin: SIMD3<Float>,
        axis axisEnd: SIMD3<Float>,
        angle: Float
    ) -> SIMD3<Float> {
        let axis = normalized(axisEnd - axisOrigin)
        let p = point - axisOrigin

        let cosA = cos(angle)
        let sinA = sin(angle)

        // Rodrigues' rotation formula
        let rotated = p * cosA + simd_cross(axis, p) * sinA + axis * simd_dot(axis, p) * (1 - cosA)
        return rotated + axisOrigin
    }

    /// Places a hydrogen on a ring nitrogen, pointing away from the two bonded ring atoms.
    private static func placeRingHydrogen(
        on nitrogenPos: SIMD3<Float>,
        ring neighborPositions: [SIMD3<Float>],
        distance: Float
    ) -> SIMD3<Float> {
        let avgNeighbor = neighborPositions.reduce(.zero, +) / Float(neighborPositions.count)
        let direction = normalized(nitrogenPos - avgNeighbor)
        return nitrogenPos + direction * distance
    }

    /// Places an amide hydrogen (NH2) with specified angle and dihedral relative to C-N bond.
    private static func placeAmideHydrogen(
        nitrogenPosition: SIMD3<Float>,
        carbonPosition: SIMD3<Float>,
        angle: Float,
        dihedral: Float,
        distance: Float
    ) -> SIMD3<Float> {
        let cnAxis = normalized(nitrogenPosition - carbonPosition)

        // Get a perpendicular reference direction
        let seed = abs(cnAxis.x) < 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        let perp1 = normalized(simd_cross(cnAxis, seed))
        let perp2 = simd_cross(cnAxis, perp1)

        let angleRad = angle * .pi / 180.0
        let dihedralRad = dihedral * .pi / 180.0

        let direction = cnAxis * cos(.pi - angleRad)
            + (perp1 * cos(dihedralRad) + perp2 * sin(dihedralRad)) * sin(.pi - angleRad)

        return nitrogenPosition + normalized(direction) * distance
    }

    /// Finds nearby N/O acceptor atoms within radius, excluding atoms in the same residue.
    private static func findNearbyAcceptors(
        center: SIMD3<Float>,
        radius: Float,
        atoms: [Atom],
        excludeIndices: Set<Int>
    ) -> [SIMD3<Float>] {
        let radiusSq = radius * radius
        var acceptors: [SIMD3<Float>] = []
        for (index, atom) in atoms.enumerated() {
            guard !excludeIndices.contains(index) else { continue }
            guard atom.element == .N || atom.element == .O || atom.element == .S else { continue }
            guard atom.element != .H else { continue }
            let distSq = simd_length_squared(atom.position - center)
            if distSq < radiusSq {
                acceptors.append(atom.position)
            }
        }
        return acceptors
    }

    private static func normalizeAngle(_ angle: Float) -> Float {
        var a = angle
        while a < 0 { a += 2 * .pi }
        while a >= 2 * .pi { a -= 2 * .pi }
        return a
    }

    private static func angleDifference(_ a: Float, _ b: Float) -> Float {
        var diff = a - b
        while diff > .pi { diff -= 2 * .pi }
        while diff < -.pi { diff += 2 * .pi }
        return diff
    }

    // MARK: - Phase 4.2 & 4.3: Scoring, Optimization, and Finalization

    /// Full Phase 4 pipeline: enumerate → score → optimize → apply.
    /// Returns modified atoms/bonds with optimized H-bond network.
    static func optimizeNetwork(
        atoms: [Atom],
        bonds: [Bond],
        predictions: [Protonation.ResiduePrediction],
        device: MTLDevice? = nil
    ) -> (atoms: [Atom], bonds: [Bond], report: NetworkReport) {
        // Phase 4.1: Enumerate
        let enumResult = enumerateNetwork(
            atoms: atoms, bonds: bonds, predictions: predictions
        )
        let graph = enumResult.graph
        var report = enumResult.report

        guard !graph.groups.isEmpty else {
            return (atoms, bonds, report)
        }

        // Phase 4.2: Score and optimize
        let resolvedDevice = device ?? MTLCreateSystemDefaultDevice()
        let optimalStates: [Int]

        if let gpu = resolvedDevice {
            optimalStates = optimizeWithGPU(
                graph: graph, atoms: atoms, bonds: bonds, device: gpu, report: &report
            )
        } else {
            optimalStates = optimizeOnCPU(
                graph: graph, atoms: atoms, bonds: bonds, report: &report
            )
        }

        // Phase 4.3: Apply optimal states
        let result = applyOptimalStates(
            optimalStates: optimalStates,
            graph: graph,
            atoms: atoms,
            bonds: bonds,
            report: &report
        )

        return (result.atoms, result.bonds, report)
    }

    // MARK: - GPU Scoring

    private static func optimizeWithGPU(
        graph: ConflictGraph,
        atoms: [Atom],
        bonds: [Bond],
        device: MTLDevice,
        report: inout NetworkReport
    ) -> [Int] {
        guard let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "scoreHBondCandidates"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            return optimizeOnCPU(graph: graph, atoms: atoms, bonds: bonds, report: &report)
        }

        // Build environment atoms (all non-moveable protein atoms)
        let moveableIndices = Set(graph.groups.flatMap(\.moveableAtomIndices))
        let envAtomData = buildEnvironmentAtoms(atoms: atoms, excludeIndices: moveableIndices)

        // Score each group's states
        var groupStateScores: [[Float]] = []

        for group in graph.groups {
            var stateScores: [Float] = []
            for state in group.states {
                let score = scoreGroupStateGPU(
                    group: group,
                    state: state,
                    envAtoms: envAtomData,
                    atoms: atoms,
                    bonds: bonds,
                    device: device,
                    commandQueue: commandQueue,
                    pipeline: pipeline
                )
                stateScores.append(score - state.penalty)
            }
            groupStateScores.append(stateScores)
        }

        // Score pairwise interactions for clique members
        let pairwiseScores = scorePairwiseInteractions(
            graph: graph, atoms: atoms, groupStateScores: &groupStateScores
        )

        // Optimize: singletons pick their best, cliques use greedy/SA
        return optimizeStates(
            graph: graph,
            groupStateScores: groupStateScores,
            pairwiseScores: pairwiseScores,
            report: &report
        )
    }

    private static func scoreGroupStateGPU(
        group: MoveableGroup,
        state: GroupState,
        envAtoms: [HBondEnvAtom],
        atoms: [Atom],
        bonds: [Bond],
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        pipeline: MTLComputePipelineState
    ) -> Float {
        // Build candidate atoms from state positions
        var candidates: [HBondCandidateAtom] = []
        for (atomName, position) in state.atomPositions {
            // Find the original atom to get its properties
            let atomIndex = group.moveableAtomIndices.first { idx in
                atoms[idx].name.trimmingCharacters(in: .whitespaces).uppercased() == atomName
            }

            let element: Element
            let formalCharge: Int32
            if let idx = atomIndex {
                element = atoms[idx].element
                formalCharge = Int32(atoms[idx].formalCharge)
            } else {
                // Infer from name
                element = atomName.hasPrefix("H") ? .H : (atomName.hasPrefix("N") ? .N : .O)
                formalCharge = 0
            }

            var flags: UInt32 = 0
            if element == .H {
                flags |= UInt32(HBNET_FLAG_HYDROGEN)
                flags |= UInt32(HBNET_FLAG_DONOR)
            } else if element == .N {
                flags |= UInt32(HBNET_FLAG_DONOR)
                flags |= UInt32(HBNET_FLAG_ACCEPTOR)
            } else if element == .O {
                flags |= UInt32(HBNET_FLAG_ACCEPTOR)
            } else if element == .S {
                flags |= UInt32(HBNET_FLAG_ACCEPTOR)
            }
            if formalCharge != 0 {
                flags |= UInt32(HBNET_FLAG_CHARGED)
            }

            candidates.append(HBondCandidateAtom(
                position: position,
                vdwRadius: element.vdwRadius,
                flags: flags,
                formalCharge: formalCharge,
                _pad0: 0, _pad1: 0
            ))
        }

        guard !candidates.isEmpty, !envAtoms.isEmpty else { return 0.0 }

        let numCandidates = candidates.count
        let numEnv = envAtoms.count

        // Build exclusion mask (exclude same-residue env atoms)
        let sameResidueEnvIndices = findSameResidueEnvIndices(
            group: group, atoms: atoms, envAtoms: envAtoms
        )
        let maskStride = (numEnv + 31) / 32
        var excludeMask = [UInt32](repeating: 0, count: numCandidates * maskStride)
        for ci in 0..<numCandidates {
            for ei in sameResidueEnvIndices {
                excludeMask[ci * maskStride + ei / 32] |= (1 << (ei & 31))
            }
        }

        var params = HBondScoringParams(
            numCandidates: UInt32(numCandidates),
            numEnvAtoms: UInt32(numEnv),
            bumpWeight: 10.0,
            hbondWeight: 4.0,
            minRegHBGap: 0.6,
            minChargedHBGap: 0.8,
            badBumpGapCut: 0.4,
            gapScale: 0.25
        )

        guard let candidateBuf = device.makeBuffer(
                bytes: &candidates,
                length: numCandidates * MemoryLayout<HBondCandidateAtom>.stride,
                options: .storageModeShared),
              let envBuf = device.makeBuffer(
                bytes: envAtoms,
                length: numEnv * MemoryLayout<HBondEnvAtom>.stride,
                options: .storageModeShared),
              let scoreBuf = device.makeBuffer(
                length: numCandidates * MemoryLayout<HBondAtomScore>.stride,
                options: .storageModeShared),
              let paramBuf = device.makeBuffer(
                bytes: &params,
                length: MemoryLayout<HBondScoringParams>.stride,
                options: .storageModeShared),
              let maskBuf = device.makeBuffer(
                bytes: &excludeMask,
                length: excludeMask.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return scoreCPUFallback(candidates: candidates, envAtoms: envAtoms, params: params, excludeMask: excludeMask)
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(candidateBuf, offset: 0, index: 0)
        enc.setBuffer(envBuf, offset: 0, index: 1)
        enc.setBuffer(scoreBuf, offset: 0, index: 2)
        enc.setBuffer(paramBuf, offset: 0, index: 3)
        enc.setBuffer(maskBuf, offset: 0, index: 4)

        let tgSize = MTLSize(width: min(numCandidates, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (numCandidates + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = scoreBuf.contents().bindMemory(to: HBondAtomScore.self, capacity: numCandidates)
        var totalScore: Float = 0
        for i in 0..<numCandidates {
            totalScore += ptr[i].totalScore
        }
        return totalScore
    }

    // MARK: - CPU Scoring Fallback

    private static func optimizeOnCPU(
        graph: ConflictGraph,
        atoms: [Atom],
        bonds: [Bond],
        report: inout NetworkReport
    ) -> [Int] {
        let moveableIndices = Set(graph.groups.flatMap(\.moveableAtomIndices))
        let envAtoms = buildEnvironmentAtoms(atoms: atoms, excludeIndices: moveableIndices)

        var groupStateScores: [[Float]] = []
        let params = HBondScoringParams(
            numCandidates: 0, numEnvAtoms: UInt32(envAtoms.count),
            bumpWeight: 10.0, hbondWeight: 4.0,
            minRegHBGap: 0.6, minChargedHBGap: 0.8,
            badBumpGapCut: 0.4, gapScale: 0.25
        )

        for group in graph.groups {
            var stateScores: [Float] = []
            let sameResIndices = findSameResidueEnvIndices(
                group: group, atoms: atoms, envAtoms: envAtoms
            )

            for state in group.states {
                var candidates: [HBondCandidateAtom] = []
                for (atomName, position) in state.atomPositions {
                    let atomIndex = group.moveableAtomIndices.first { idx in
                        atoms[idx].name.trimmingCharacters(in: .whitespaces).uppercased() == atomName
                    }
                    let element: Element
                    let formalCharge: Int32
                    if let idx = atomIndex {
                        element = atoms[idx].element
                        formalCharge = Int32(atoms[idx].formalCharge)
                    } else {
                        element = atomName.hasPrefix("H") ? .H : (atomName.hasPrefix("N") ? .N : .O)
                        formalCharge = 0
                    }

                    var flags: UInt32 = 0
                    if element == .H { flags |= UInt32(HBNET_FLAG_HYDROGEN) | UInt32(HBNET_FLAG_DONOR) }
                    else if element == .N { flags |= UInt32(HBNET_FLAG_DONOR) | UInt32(HBNET_FLAG_ACCEPTOR) }
                    else if element == .O { flags |= UInt32(HBNET_FLAG_ACCEPTOR) }
                    else if element == .S { flags |= UInt32(HBNET_FLAG_ACCEPTOR) }
                    if formalCharge != 0 { flags |= UInt32(HBNET_FLAG_CHARGED) }

                    candidates.append(HBondCandidateAtom(
                        position: position, vdwRadius: element.vdwRadius,
                        flags: flags, formalCharge: formalCharge, _pad0: 0, _pad1: 0
                    ))
                }

                let maskStride = (envAtoms.count + 31) / 32
                var excludeMask = [UInt32](repeating: 0, count: candidates.count * maskStride)
                for ci in 0..<candidates.count {
                    for ei in sameResIndices {
                        excludeMask[ci * maskStride + ei / 32] |= (1 << (ei & 31))
                    }
                }

                let score = scoreCPUFallback(
                    candidates: candidates, envAtoms: envAtoms,
                    params: params, excludeMask: excludeMask
                )
                stateScores.append(score - state.penalty)
            }
            groupStateScores.append(stateScores)
        }

        let pairwiseScores = scorePairwiseInteractions(
            graph: graph, atoms: atoms, groupStateScores: &groupStateScores
        )

        return optimizeStates(
            graph: graph,
            groupStateScores: groupStateScores,
            pairwiseScores: pairwiseScores,
            report: &report
        )
    }

    private static func scoreCPUFallback(
        candidates: [HBondCandidateAtom],
        envAtoms: [HBondEnvAtom],
        params: HBondScoringParams,
        excludeMask: [UInt32]
    ) -> Float {
        let maskStride = (envAtoms.count + 31) / 32
        var totalScore: Float = 0

        for (ci, candidate) in candidates.enumerated() {
            let cPos = candidate.position
            let cRad = candidate.vdwRadius
            let cIsDonor = (candidate.flags & UInt32(HBNET_FLAG_DONOR)) != 0
            let cIsAcceptor = (candidate.flags & UInt32(HBNET_FLAG_ACCEPTOR)) != 0
            let cIsHydrogen = (candidate.flags & UInt32(HBNET_FLAG_HYDROGEN)) != 0
            let cIsCharged = (candidate.flags & UInt32(HBNET_FLAG_CHARGED)) != 0
            let cCharge = candidate.formalCharge

            for (ei, envAtom) in envAtoms.enumerated() {
                let maskWord = excludeMask[ci * maskStride + ei / 32]
                if (maskWord >> (ei & 31)) & 1 != 0 { continue }

                let d = simd_length(cPos - envAtom.position)
                let sumRadii = cRad + envAtom.vdwRadius
                if d > sumRadii + 1.0 { continue }

                let gap = d - sumRadii

                let eIsDonor = (envAtom.flags & UInt32(HBNET_FLAG_DONOR)) != 0
                let eIsAcceptor = (envAtom.flags & UInt32(HBNET_FLAG_ACCEPTOR)) != 0
                let eIsCharged = (envAtom.flags & UInt32(HBNET_FLAG_CHARGED)) != 0
                let eCharge = envAtom.formalCharge

                var isaHB = false
                if (cIsDonor && eIsAcceptor) || (cIsAcceptor && eIsDonor) ||
                   (cIsHydrogen && eIsAcceptor) {
                    let sameSign = (cCharge > 0 && eCharge > 0) || (cCharge < 0 && eCharge < 0)
                    if !sameSign { isaHB = true }
                }

                let bothCharged = cIsCharged && eIsCharged && (cCharge * eCharge < 0)
                let hbMindist = bothCharged ? params.minChargedHBGap : params.minRegHBGap

                if gap > 0 {
                    let scaledGap = gap / params.gapScale
                    totalScore += exp(-scaledGap * scaledGap)
                } else if isaHB {
                    if gap < -hbMindist {
                        let adjustedGap = gap + hbMindist
                        let overlap = -0.5 * adjustedGap
                        totalScore -= params.bumpWeight * overlap
                    } else {
                        let overlap = -0.5 * gap
                        totalScore += params.hbondWeight * overlap
                    }
                } else {
                    let overlap = -0.5 * gap
                    totalScore -= params.bumpWeight * overlap
                }
            }
        }
        return totalScore
    }

    // MARK: - Environment atoms

    private static func buildEnvironmentAtoms(
        atoms: [Atom],
        excludeIndices: Set<Int>
    ) -> [HBondEnvAtom] {
        var envAtoms: [HBondEnvAtom] = []
        envAtoms.reserveCapacity(atoms.count)

        for (index, atom) in atoms.enumerated() {
            if excludeIndices.contains(index) { continue }

            var flags: UInt32 = 0
            switch atom.element {
            case .H:
                flags |= UInt32(HBNET_FLAG_HYDROGEN) | UInt32(HBNET_FLAG_DONOR)
            case .N:
                flags |= UInt32(HBNET_FLAG_DONOR) | UInt32(HBNET_FLAG_ACCEPTOR)
            case .O:
                flags |= UInt32(HBNET_FLAG_ACCEPTOR)
            case .S:
                flags |= UInt32(HBNET_FLAG_ACCEPTOR)
            default:
                break
            }
            if atom.formalCharge != 0 {
                flags |= UInt32(HBNET_FLAG_CHARGED)
            }

            envAtoms.append(HBondEnvAtom(
                position: atom.position,
                vdwRadius: atom.element.vdwRadius,
                flags: flags,
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0
            ))
        }
        return envAtoms
    }

    private static func findSameResidueEnvIndices(
        group: MoveableGroup,
        atoms: [Atom],
        envAtoms: [HBondEnvAtom]
    ) -> [Int] {
        // Find env atom indices that belong to the same residue as the group
        let groupResChain = group.residueID.chainID
        let groupResSeq = group.residueID.residueSeq

        let moveableSet = Set(group.moveableAtomIndices)
        var sameResPositions = Set<SIMD3<Float>>()
        for idx in group.allAtomIndices where !moveableSet.contains(idx) {
            if atoms[idx].chainID == groupResChain && atoms[idx].residueSeq == groupResSeq {
                sameResPositions.insert(atoms[idx].position)
            }
        }

        var indices: [Int] = []
        for (ei, envAtom) in envAtoms.enumerated() {
            if sameResPositions.contains(envAtom.position) {
                indices.append(ei)
            }
        }
        return indices
    }

    // MARK: - Pairwise interaction scoring

    /// Computes interaction scores between pairs of groups in the same clique.
    /// For each pair (i,j) and each state combination (si,sj), estimates the
    /// inter-group interaction by checking VDW overlap between their moveable atoms.
    private static func scorePairwiseInteractions(
        graph: ConflictGraph,
        atoms: [Atom],
        groupStateScores: inout [[Float]]
    ) -> [Int: [Int: [[Float]]]] {
        // pairwiseScores[i][j][si][sj] = interaction score between group i in state si
        // and group j in state sj
        var pairwise: [Int: [Int: [[Float]]]] = [:]

        for clique in graph.cliques {
            for ci in 0..<clique.count {
                let i = clique[ci]
                let groupI = graph.groups[i]
                for cj in (ci + 1)..<clique.count {
                    let j = clique[cj]
                    let groupJ = graph.groups[j]

                    var matrix = [[Float]](
                        repeating: [Float](repeating: 0, count: groupJ.states.count),
                        count: groupI.states.count
                    )

                    for si in 0..<groupI.states.count {
                        for sj in 0..<groupJ.states.count {
                            matrix[si][sj] = pairwiseInteractionScore(
                                groupI.states[si], groupJ.states[sj]
                            )
                        }
                    }

                    pairwise[i, default: [:]][j] = matrix
                    // Symmetric: transpose for j→i lookup
                    var transposed = [[Float]](
                        repeating: [Float](repeating: 0, count: groupI.states.count),
                        count: groupJ.states.count
                    )
                    for si in 0..<groupI.states.count {
                        for sj in 0..<groupJ.states.count {
                            transposed[sj][si] = matrix[si][sj]
                        }
                    }
                    pairwise[j, default: [:]][i] = transposed
                }
            }
        }
        return pairwise
    }

    private static func pairwiseInteractionScore(
        _ stateA: GroupState,
        _ stateB: GroupState
    ) -> Float {
        var score: Float = 0
        for (_, posA) in stateA.atomPositions {
            for (nameB, posB) in stateB.atomPositions {
                let d = simd_length(posA - posB)
                let elementB: Element = nameB.hasPrefix("H") ? .H : (nameB.hasPrefix("N") ? .N : .O)
                let radA: Float = 1.2 // approximate
                let radB = elementB.vdwRadius
                let gap = d - (radA + radB)

                if gap < 0 {
                    // Clash penalty
                    score -= 10.0 * (-0.5 * gap)
                } else if gap < 1.0 {
                    // Weak contact
                    let scaled = gap / 0.25
                    score += exp(-scaled * scaled) * 0.1
                }
            }
        }
        return score
    }

    // MARK: - State optimization

    private static func optimizeStates(
        graph: ConflictGraph,
        groupStateScores: [[Float]],
        pairwiseScores: [Int: [Int: [[Float]]]],
        report: inout NetworkReport
    ) -> [Int] {
        let n = graph.groups.count
        var bestStates = [Int](repeating: 0, count: n)

        // Singletons: just pick the best state
        for singleIdx in graph.singletons {
            let scores = groupStateScores[singleIdx]
            if let bestIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) {
                bestStates[singleIdx] = bestIdx
            }
        }

        // Cliques: greedy optimization with simulated annealing refinement
        for clique in graph.cliques {
            if clique.count <= 4 {
                // Small clique: exhaustive search
                let result = exhaustiveSearch(
                    clique: clique,
                    groupStateScores: groupStateScores,
                    pairwiseScores: pairwiseScores,
                    groups: graph.groups
                )
                for (idx, state) in zip(clique, result) {
                    bestStates[idx] = state
                }
            } else {
                // Larger clique: greedy + simulated annealing
                let result = simulatedAnnealing(
                    clique: clique,
                    groupStateScores: groupStateScores,
                    pairwiseScores: pairwiseScores,
                    groups: graph.groups
                )
                for (idx, state) in zip(clique, result) {
                    bestStates[idx] = state
                }
            }
        }

        // Build decisions for report
        for (groupIdx, group) in graph.groups.enumerated() {
            let chosenState = bestStates[groupIdx]
            let originalScore = groupStateScores[groupIdx][0]
            let chosenScore = groupStateScores[groupIdx][chosenState]
            let delta = chosenScore - originalScore

            let decision = FlipDecision(
                residueID: group.residueID,
                kind: group.kind,
                chosenStateIndex: chosenState,
                chosenLabel: group.states[chosenState].label,
                isFlipped: group.states[chosenState].isFlipped,
                score: chosenScore,
                originalScore: originalScore,
                deltaEnergy: delta
            )
            report.decisions.append(decision)

            if group.states[chosenState].isFlipped {
                report.flipsAccepted += 1
            }
            if chosenState != 0 {
                report.rotationsOptimized += 1
            }
            report.totalEnergyImprovement += delta
        }

        return bestStates
    }

    /// Exhaustive search for small cliques (≤4 groups).
    private static func exhaustiveSearch(
        clique: [Int],
        groupStateScores: [[Float]],
        pairwiseScores: [Int: [Int: [[Float]]]],
        groups: [MoveableGroup]
    ) -> [Int] {
        let numGroups = clique.count
        let stateCounts = clique.map { groups[$0].states.count }

        var bestScore: Float = -.infinity
        var bestAssignment = [Int](repeating: 0, count: numGroups)
        var current = [Int](repeating: 0, count: numGroups)

        func enumerate(depth: Int) {
            if depth == numGroups {
                var score: Float = 0
                // Vertex scores
                for ci in 0..<numGroups {
                    score += groupStateScores[clique[ci]][current[ci]]
                }
                // Pairwise scores
                for ci in 0..<numGroups {
                    for cj in (ci + 1)..<numGroups {
                        let i = clique[ci], j = clique[cj]
                        if let matrix = pairwiseScores[i]?[j] {
                            score += matrix[current[ci]][current[cj]]
                        }
                    }
                }
                if score > bestScore {
                    bestScore = score
                    bestAssignment = current
                }
                return
            }
            for s in 0..<stateCounts[depth] {
                current[depth] = s
                enumerate(depth: depth + 1)
            }
        }

        enumerate(depth: 0)
        return bestAssignment
    }

    /// Simulated annealing for larger cliques.
    private static func simulatedAnnealing(
        clique: [Int],
        groupStateScores: [[Float]],
        pairwiseScores: [Int: [Int: [[Float]]]],
        groups: [MoveableGroup]
    ) -> [Int] {
        let numGroups = clique.count
        let stateCounts = clique.map { groups[$0].states.count }

        // Start with greedy: each group picks its locally best state
        var current = [Int](repeating: 0, count: numGroups)
        for ci in 0..<numGroups {
            let scores = groupStateScores[clique[ci]]
            if let best = scores.indices.max(by: { scores[$0] < scores[$1] }) {
                current[ci] = best
            }
        }

        func totalScore(_ assignment: [Int]) -> Float {
            var score: Float = 0
            for ci in 0..<numGroups {
                score += groupStateScores[clique[ci]][assignment[ci]]
            }
            for ci in 0..<numGroups {
                for cj in (ci + 1)..<numGroups {
                    let i = clique[ci], j = clique[cj]
                    if let matrix = pairwiseScores[i]?[j] {
                        score += matrix[assignment[ci]][assignment[cj]]
                    }
                }
            }
            return score
        }

        var bestAssignment = current
        var bestScore = totalScore(current)
        var currentScore = bestScore

        // SA parameters
        var temperature: Float = 2.0
        let coolingRate: Float = 0.95
        let minTemp: Float = 0.01
        let stepsPerTemp = numGroups * 3

        // Use a simple LCG for deterministic pseudo-randomness
        var rngState: UInt64 = 42

        func nextRandom() -> Float {
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            return Float(rngState >> 33) / Float(UInt32.max >> 1)
        }

        while temperature > minTemp {
            for _ in 0..<stepsPerTemp {
                // Pick a random group and a random alternative state
                let ci = Int(nextRandom() * Float(numGroups)) % numGroups
                let numStates = stateCounts[ci]
                guard numStates > 1 else { continue }

                var newState = Int(nextRandom() * Float(numStates)) % numStates
                if newState == current[ci] {
                    newState = (newState + 1) % numStates
                }

                // Compute delta score
                let oldGroupScore = groupStateScores[clique[ci]][current[ci]]
                let newGroupScore = groupStateScores[clique[ci]][newState]
                var delta = newGroupScore - oldGroupScore

                for cj in 0..<numGroups where cj != ci {
                    let i = clique[ci], j = clique[cj]
                    if let matrix = pairwiseScores[i]?[j] {
                        delta += matrix[newState][current[cj]] - matrix[current[ci]][current[cj]]
                    }
                }

                // Accept or reject
                if delta > 0 || nextRandom() < exp(delta / temperature) {
                    current[ci] = newState
                    currentScore += delta

                    if currentScore > bestScore {
                        bestScore = currentScore
                        bestAssignment = current
                    }
                }
            }
            temperature *= coolingRate
        }

        return bestAssignment
    }

    // MARK: - Phase 4.3: Apply optimal states to atoms

    private static func applyOptimalStates(
        optimalStates: [Int],
        graph: ConflictGraph,
        atoms: [Atom],
        bonds: [Bond],
        report: inout NetworkReport
    ) -> (atoms: [Atom], bonds: [Bond]) {
        var workingAtoms = atoms

        for (groupIdx, group) in graph.groups.enumerated() {
            let stateIdx = optimalStates[groupIdx]
            let state = group.states[stateIdx]

            // Apply position updates for each moveable atom
            for (atomName, newPosition) in state.atomPositions {
                // Find the atom in moveableAtomIndices by name
                if let atomIdx = group.moveableAtomIndices.first(where: { idx in
                    workingAtoms[idx].name.trimmingCharacters(in: .whitespaces).uppercased() == atomName
                }) {
                    workingAtoms[atomIdx].position = newPosition

                    // For His flips, also update donor/acceptor identity via formal charge
                    if group.kind == .flipHis && state.isFlipped {
                        // ND1 and NE2 swap roles — update charges if needed
                        if atomName == "ND1" || atomName == "NE2" ||
                           atomName == "CD2" || atomName == "CE1" {
                            // Positions already swapped; charges handled by protonation state
                        }
                    }
                }
            }

            // For amide flips, swap element types (O↔N) at the swapped positions
            if group.kind == .flipAmide && state.isFlipped {
                let oxygenName: String
                let nitrogenName: String
                switch group.residueID.residueName {
                case "ASN":
                    oxygenName = "OD1"; nitrogenName = "ND2"
                case "GLN":
                    oxygenName = "OE1"; nitrogenName = "NE2"
                default:
                    continue
                }

                if let oIdx = group.moveableAtomIndices.first(where: { idx in
                    workingAtoms[idx].name.trimmingCharacters(in: .whitespaces).uppercased() == oxygenName
                }),
                   let nIdx = group.moveableAtomIndices.first(where: { idx in
                    workingAtoms[idx].name.trimmingCharacters(in: .whitespaces).uppercased() == nitrogenName
                }) {
                    // Swap elements: what was O is now N and vice versa
                    workingAtoms[oIdx].element = .N
                    workingAtoms[oIdx].name = nitrogenName
                    workingAtoms[nIdx].element = .O
                    workingAtoms[nIdx].name = oxygenName
                }
            }

            // For His flips with tautomer changes, handle proton presence
            if group.kind == .flipHis {
                let hasHD1 = state.atomPositions["HD1"] != nil
                let hasHE2 = state.atomPositions["HE2"] != nil

                // Remove or add protons based on tautomer state
                // HD1 present = δ-tautomer or doubly protonated
                // HE2 present = ε-tautomer or doubly protonated
                for atomIdx in group.moveableAtomIndices {
                    let name = workingAtoms[atomIdx].name.trimmingCharacters(in: .whitespaces).uppercased()
                    if name == "HD1" && !hasHD1 {
                        // Move proton far away (effectively remove — actual removal
                        // would require array surgery that breaks indices)
                        workingAtoms[atomIdx].position = SIMD3<Float>(9999, 9999, 9999)
                    }
                    if name == "HE2" && !hasHE2 {
                        workingAtoms[atomIdx].position = SIMD3<Float>(9999, 9999, 9999)
                    }
                }
            }
        }

        return (workingAtoms, bonds)
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Flexible Residue Docking Engine

/// Manages GPU buffers and dispatch for flexible sidechain docking.
/// Works alongside the main DockingEngine — call its methods at the right
/// points in the docking loop rather than replacing the engine.
///
/// Usage:
///   1. Before grid generation: call `excludeFlexAtoms()` to remove flex sidechain atoms from the protein
///   2. After `prepareLigand()`: call `prepareFlexBuffers()` to create GPU buffers
///   3. After each scoring pass: call `dispatchFlexScoring()` to add flex-ligand energy
///   4. After each evolution pass: call `dispatchChiEvolution()` to mutate chi angles
///   5. After each local search: call `dispatchFlexLocalSearch()` to refine chi angles
final class FlexDockingEngine {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Pipeline states
    private var scoreFlexPipeline: MTLComputePipelineState?
    private var evolveChiPipeline: MTLComputePipelineState?
    private var localSearchFlexPipeline: MTLComputePipelineState?

    // GPU buffers
    private(set) var flexAtomBuffer: MTLBuffer?
    private(set) var flexTorsionEdgeBuffer: MTLBuffer?
    private(set) var flexMovingIndicesBuffer: MTLBuffer?
    private(set) var flexParamsBuffer: MTLBuffer?

    // State
    private(set) var flexConfig: FlexibleResidueConfig?
    private(set) var numFlexAtoms: Int = 0
    private(set) var numFlexTorsions: Int = 0
    private(set) var numChiSlots: Int = 0 // total chi angle slots used

    var isEnabled: Bool { numFlexAtoms > 0 && numFlexTorsions > 0 }

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue

        guard let library = device.makeDefaultLibrary() else { return }

        do {
            if let f = library.makeFunction(name: "scoreFlexSidechains") {
                scoreFlexPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "evolveChiAngles") {
                evolveChiPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "localSearchFlex") {
                localSearchFlexPipeline = try device.makeComputePipelineState(function: f)
            }
        } catch {
            print("FlexDockingEngine: failed to create pipeline states: \(error)")
        }
    }

    // MARK: - Exclude Flex Atoms from Protein

    /// Given the protein atoms and selected flexible residue indices, returns the
    /// protein atoms with flexible sidechain atoms removed (for grid generation),
    /// along with the extracted flex sidechain atom data.
    struct FlexExclusion {
        var rigidAtoms: [Atom]      // protein atoms minus flex sidechains
        var rigidBonds: [Bond]
        var flexAtoms: [FlexSidechainAtom]
        var flexTorsionEdges: [FlexTorsionEdge]
        var flexMovingIndices: [Int32]
        var chiSlotCount: Int
        var flexResidueCount: Int
    }

    // Backbone atom names that should NOT be made flexible
    private static let backboneAtomNames: Set<String> = ["N", "CA", "C", "O", "H", "HA", "OXT"]

    /// Capacity of the per-thread arrays in FlexDocking.metal (`flexPositions[64]`,
    /// `DockPose.chiAngles[24]`). Edges whose pivot/axis index exceeds these bounds would
    /// read past the thread-local arrays, so residues that do not fit are kept rigid.
    static let maxKernelFlexAtoms = 64
    static let maxKernelChiSlots = 24

    /// Separate flex sidechain atoms from the rigid protein.
    /// - Parameter vinaTypes: Per-atom Vina XS type for each protein atom (from DockingEngine's typing logic).
    ///   If nil, flex atoms get vinaType = -1 and won't contribute to scoring.
    func excludeFlexAtoms(
        proteinAtoms: [Atom],
        proteinBonds: [Bond],
        flexConfig: FlexibleResidueConfig,
        vinaTypes: [Int32]? = nil
    ) -> FlexExclusion {
        self.flexConfig = flexConfig

        guard !flexConfig.flexibleResidueIndices.isEmpty else {
            return FlexExclusion(
                rigidAtoms: proteinAtoms, rigidBonds: proteinBonds,
                flexAtoms: [], flexTorsionEdges: [], flexMovingIndices: [],
                chiSlotCount: 0, flexResidueCount: 0
            )
        }

        let flexResidueSet = Set(flexConfig.flexibleResidueIndices)

        // Residues are selected by sequence number only, which is ambiguous across chains and
        // with HETATM groups (waters, ligands) sharing the number. Bind each selected number to
        // the first polymer residue carrying it, and keep only one alternate location.
        struct ResidueKey: Hashable { let chainID: String; let insertionCode: String }
        var residueOwner: [Int: ResidueKey] = [:]
        var residueAltLoc: [Int: String] = [:]
        var sidechainIndices: [Int: [Int]] = [:]   // residueSeq → sidechain atom indices (incl. H)
        var otherAltSidechain: [Int: [Int]] = [:]  // sidechain atoms of the non-selected altLocs
        var backboneNameToGlobal: [Int: [String: Int]] = [:]

        for (i, atom) in proteinAtoms.enumerated() {
            guard flexResidueSet.contains(atom.residueSeq), !atom.isHetAtom else { continue }
            let seq = atom.residueSeq
            let key = ResidueKey(chainID: atom.chainID, insertionCode: atom.insertionCode)
            if let owner = residueOwner[seq] {
                guard owner == key else { continue }
            } else {
                residueOwner[seq] = key
            }
            if !atom.altLoc.isEmpty {
                if let alt = residueAltLoc[seq] {
                    guard alt == atom.altLoc else {
                        // Not scored, but must leave the rigid receptor along with the moving copy.
                        if !Self.backboneAtomNames.contains(atom.name) {
                            otherAltSidechain[seq, default: []].append(i)
                        }
                        continue
                    }
                } else {
                    residueAltLoc[seq] = atom.altLoc
                }
            }
            if Self.backboneAtomNames.contains(atom.name) {
                if backboneNameToGlobal[seq]?[atom.name] == nil {
                    backboneNameToGlobal[seq, default: [:]][atom.name] = i
                }
            } else {
                sidechainIndices[seq, default: []].append(i)
            }
        }

        // Build flex sidechain atoms
        var gpuFlexAtoms: [FlexSidechainAtom] = []
        var flexAtomIndices = Set<Int>()

        // Sort residues for deterministic chi slot assignment
        let sortedFlexResidues = flexConfig.flexibleResidueIndices.sorted()

        var chiSlot = 0
        var flexResidueCount = 0
        var allFlexEdges: [FlexTorsionEdge] = []
        var allFlexMoving: [Int32] = []
        var skippedForCapacity: [Int] = []

        for residueSeq in sortedFlexResidues {
            guard let sidechain = sidechainIndices[residueSeq],
                  let firstAtom = sidechain.first,
                  let rotDef = RotamerLibrary.rotamers(for: proteinAtoms[firstAtom].residueName)
            else { continue }

            // Hydrogens carry VINA_OTHER (unsupported by the flex pair potential) and are never
            // scored, so only heavy atoms are uploaded; they would otherwise exhaust the
            // 64-atom kernel budget without contributing energy.
            let heavyIndices = sidechain.filter { proteinAtoms[$0].element != .H }
            let backbone = backboneNameToGlobal[residueSeq] ?? [:]
            let nameToGlobal: [String: Int] = Dictionary(
                heavyIndices.map { (proteinAtoms[$0].name, $0) },
                uniquingKeysWith: { first, _ in first }
            )

            // Resolve chi definitions whose pivot/axis atoms are present.
            let chis = rotDef.chiAngles.filter { chi in
                (nameToGlobal[chi.atomNames.1] != nil || backbone[chi.atomNames.1] != nil) &&
                nameToGlobal[chi.atomNames.2] != nil
            }
            guard !chis.isEmpty else { continue }

            let backbonePivots = Set(chis.compactMap { chi -> String? in
                nameToGlobal[chi.atomNames.1] == nil ? chi.atomNames.1 : nil
            })
            if gpuFlexAtoms.count + heavyIndices.count + backbonePivots.count > Self.maxKernelFlexAtoms ||
               chiSlot + chis.count > Self.maxKernelChiSlots {
                skippedForCapacity.append(residueSeq)
                continue
            }

            let resIdx = Int32(flexResidueCount)
            flexResidueCount += 1
            flexAtomIndices.formUnion(sidechain)
            flexAtomIndices.formUnion(otherAltSidechain[residueSeq] ?? [])

            func appendFlexAtom(_ globalIdx: Int, scored: Bool) -> Int {
                let atom = proteinAtoms[globalIdx]
                var fa = FlexSidechainAtom()
                fa.referencePosition = atom.position
                fa.charge = atom.charge
                fa.vinaType = scored ? (vinaTypes.flatMap { globalIdx < $0.count ? $0[globalIdx] : nil } ?? -1) : -1
                fa.residueIndex = resIdx
                gpuFlexAtoms.append(fa)
                return gpuFlexAtoms.count - 1
            }

            let localStart = gpuFlexAtoms.count
            var nameToLocal: [String: Int] = [:]
            for globalIdx in heavyIndices {
                let localIdx = appendFlexAtom(globalIdx, scored: true)
                if nameToLocal[proteinAtoms[globalIdx].name] == nil {
                    nameToLocal[proteinAtoms[globalIdx].name] = localIdx
                }
            }
            // Backbone pivots (e.g. CA for chi1) never move; they only define the rotation axis
            // and are excluded from pairwise scoring (vinaType = -1).
            for name in backbonePivots.sorted() {
                if let globalIdx = backbone[name] {
                    nameToLocal[name] = appendFlexAtom(globalIdx, scored: false)
                }
            }

            for chi in chis {
                guard let pivotLocal = nameToLocal[chi.atomNames.1],
                      let axisLocal = nameToLocal[chi.atomNames.2] else { continue }

                // Find downstream atoms (everything past the axis atom in the sidechain)
                let downstream = findDownstreamLocalIndices(
                    axisLocal: axisLocal,
                    pivotLocal: pivotLocal,
                    localRange: localStart..<gpuFlexAtoms.count,
                    atoms: gpuFlexAtoms
                )

                var edge = FlexTorsionEdge()
                edge.pivotAtom = Int32(pivotLocal)
                edge.axisAtom = Int32(axisLocal)
                edge.movingStart = Int32(allFlexMoving.count)
                edge.movingCount = Int32(downstream.count)
                edge.chiSlot = Int32(chiSlot)
                allFlexMoving.append(contentsOf: downstream.map { Int32($0) })
                allFlexEdges.append(edge)

                chiSlot += 1
            }
        }

        if !skippedForCapacity.isEmpty {
            let message = "Flex residue(s) \(skippedForCapacity.map(String.init).joined(separator: ", ")) kept rigid: " +
                "GPU flex kernel limit of \(Self.maxKernelFlexAtoms) atoms / \(Self.maxKernelChiSlots) chi angles reached"
            Task { @MainActor in ActivityLog.shared.warn(message, category: .dock) }
        }

        // Build rigid atom list (everything except flex sidechain atoms)
        var oldToNew = [Int](repeating: -1, count: proteinAtoms.count)
        var rigidAtoms: [Atom] = []
        rigidAtoms.reserveCapacity(proteinAtoms.count - flexAtomIndices.count)
        for (i, atom) in proteinAtoms.enumerated() where !flexAtomIndices.contains(i) {
            oldToNew[i] = rigidAtoms.count
            rigidAtoms.append(atom)
        }
        var rigidBonds: [Bond] = []
        for bond in proteinBonds {
            guard bond.atomIndex1 < oldToNew.count, bond.atomIndex2 < oldToNew.count else { continue }
            let a = oldToNew[bond.atomIndex1], b = oldToNew[bond.atomIndex2]
            if a >= 0, b >= 0 {
                rigidBonds.append(Bond(id: rigidBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }

        return FlexExclusion(
            rigidAtoms: rigidAtoms,
            rigidBonds: rigidBonds,
            flexAtoms: gpuFlexAtoms,
            flexTorsionEdges: allFlexEdges,
            flexMovingIndices: allFlexMoving,
            chiSlotCount: chiSlot,
            flexResidueCount: flexResidueCount
        )
    }

    /// Find all local indices downstream of the axis atom (BFS through bond-distance graph).
    private func findDownstreamLocalIndices(
        axisLocal: Int, pivotLocal: Int,
        localRange: Range<Int>,
        atoms: [FlexSidechainAtom]
    ) -> [Int] {
        var visited = Set<Int>([pivotLocal])
        var queue = [axisLocal]
        var result: [Int] = []

        while !queue.isEmpty {
            let current = queue.removeFirst()
            guard !visited.contains(current), localRange.contains(current) else { continue }
            visited.insert(current)
            result.append(current)

            let pos = SIMD3<Float>(atoms[current].referencePosition)
            for idx in localRange {
                if !visited.contains(idx) {
                    let other = SIMD3<Float>(atoms[idx].referencePosition)
                    let d = simd_distance(pos, other)
                    if d < 1.9 && d > 0.4 {
                        queue.append(idx)
                    }
                }
            }
        }

        return result
    }

    // MARK: - Create GPU Buffers

    func prepareFlexBuffers(exclusion: FlexExclusion, flexWeight: Float = 1.0, chiStep: Float = FlexibleResidueConfig.fullChiStep) {
        numFlexAtoms = exclusion.flexAtoms.count
        numFlexTorsions = exclusion.flexTorsionEdges.count
        numChiSlots = exclusion.chiSlotCount

        Task { @MainActor in ActivityLog.shared.info("Flex buffers: \(exclusion.flexAtoms.count) atoms, \(exclusion.flexTorsionEdges.count) torsions, \(exclusion.chiSlotCount) chi slots, weight=\(String(format: "%.2f", flexWeight)), chiStep=\(String(format: "%.2f", chiStep))", category: .dock) }

        guard numFlexAtoms > 0 else {
            flexAtomBuffer = nil
            flexTorsionEdgeBuffer = nil
            flexMovingIndicesBuffer = nil
            flexParamsBuffer = nil
            return
        }

        var atoms = exclusion.flexAtoms
        flexAtomBuffer = device.makeBuffer(
            bytes: &atoms,
            length: MemoryLayout<FlexSidechainAtom>.stride * atoms.count,
            options: .storageModeShared
        )

        var edges = exclusion.flexTorsionEdges
        flexTorsionEdgeBuffer = device.makeBuffer(
            bytes: &edges,
            length: MemoryLayout<FlexTorsionEdge>.stride * max(edges.count, 1),
            options: .storageModeShared
        )

        var moving = exclusion.flexMovingIndices
        if moving.isEmpty { moving = [0] } // Metal requires non-empty buffer
        flexMovingIndicesBuffer = device.makeBuffer(
            bytes: &moving,
            length: MemoryLayout<Int32>.stride * moving.count,
            options: .storageModeShared
        )

        var params = FlexParams()
        params.numFlexAtoms = UInt32(numFlexAtoms)
        params.numFlexTorsions = UInt32(numFlexTorsions)
        params.numFlexResidues = UInt32(exclusion.flexResidueCount)
        params.flexWeight = flexWeight
        params.chiStep = chiStep
        flexParamsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<FlexParams>.stride,
            options: .storageModeShared
        )
    }

    // MARK: - Dispatch Flex Scoring

    /// Dispatch the flex sidechain scoring kernel. Call this after the main scoring pass.
    func dispatchFlexScoring(
        populationBuffer: MTLBuffer,
        ligandAtomBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        torsionEdgeBuffer: MTLBuffer,
        movingIndicesBuffer: MTLBuffer,
        populationSize: Int
    ) {
        guard isEnabled,
              let pipeline = scoreFlexPipeline,
              let fab = flexAtomBuffer,
              let feb = flexTorsionEdgeBuffer,
              let fmb = flexMovingIndicesBuffer,
              let fpb = flexParamsBuffer
        else { return }

        let flexAtomCount = self.numFlexAtoms
        Task { @MainActor in ActivityLog.shared.debug("Dispatching flex scoring: popSize=\(populationSize), flexAtoms=\(flexAtomCount)", category: .dock) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(populationBuffer, offset: 0, index: 0)
        enc.setBuffer(ligandAtomBuffer, offset: 0, index: 1)
        enc.setBuffer(fab, offset: 0, index: 2)
        enc.setBuffer(feb, offset: 0, index: 3)
        enc.setBuffer(fmb, offset: 0, index: 4)
        enc.setBuffer(fpb, offset: 0, index: 5)
        enc.setBuffer(gaParamsBuffer, offset: 0, index: 6)
        enc.setBuffer(torsionEdgeBuffer, offset: 0, index: 7)
        enc.setBuffer(movingIndicesBuffer, offset: 0, index: 8)

        let tgs = MTLSize(width: min(populationSize, 256), height: 1, depth: 1)
        let tgc = MTLSize(width: (populationSize + tgs.width - 1) / tgs.width, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmdBuf.commit()
    }

    /// Dispatch chi angle evolution. Call after main gaEvolve/mcPerturb.
    func dispatchChiEvolution(
        offspringBuffer: MTLBuffer,
        populationBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        guard isEnabled,
              let pipeline = evolveChiPipeline,
              let fpb = flexParamsBuffer
        else { return }

        let chiSlotCount = self.numChiSlots
        Task { @MainActor in ActivityLog.shared.debug("Dispatching chi evolution: popSize=\(populationSize), chiSlots=\(chiSlotCount)", category: .dock) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(offspringBuffer, offset: 0, index: 0)
        enc.setBuffer(populationBuffer, offset: 0, index: 1)
        enc.setBuffer(gaParamsBuffer, offset: 0, index: 2)
        enc.setBuffer(fpb, offset: 0, index: 3)

        let tgs = MTLSize(width: min(populationSize, 256), height: 1, depth: 1)
        let tgc = MTLSize(width: (populationSize + tgs.width - 1) / tgs.width, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmdBuf.commit()
    }

    /// Dispatch flex local search. Call after main local search.
    func dispatchFlexLocalSearch(
        populationBuffer: MTLBuffer,
        ligandAtomBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        torsionEdgeBuffer: MTLBuffer,
        movingIndicesBuffer: MTLBuffer,
        affinityGridBuffer: MTLBuffer,
        typeIndexBuffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        intraPairsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        guard isEnabled,
              let pipeline = localSearchFlexPipeline,
              let fab = flexAtomBuffer,
              let feb = flexTorsionEdgeBuffer,
              let fmb = flexMovingIndicesBuffer,
              let fpb = flexParamsBuffer
        else { return }

        let flexTorsionCount = self.numFlexTorsions
        Task { @MainActor in ActivityLog.shared.debug("Dispatching flex local search: popSize=\(populationSize), torsions=\(flexTorsionCount)", category: .dock) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(populationBuffer, offset: 0, index: 0)
        enc.setBuffer(ligandAtomBuffer, offset: 0, index: 1)
        enc.setBuffer(fab, offset: 0, index: 2)
        enc.setBuffer(feb, offset: 0, index: 3)
        enc.setBuffer(fmb, offset: 0, index: 4)
        enc.setBuffer(fpb, offset: 0, index: 5)
        enc.setBuffer(gaParamsBuffer, offset: 0, index: 6)
        enc.setBuffer(torsionEdgeBuffer, offset: 0, index: 7)
        enc.setBuffer(movingIndicesBuffer, offset: 0, index: 8)
        enc.setBuffer(affinityGridBuffer, offset: 0, index: 9)
        enc.setBuffer(typeIndexBuffer, offset: 0, index: 10)
        enc.setBuffer(gridParamsBuffer, offset: 0, index: 11)
        enc.setBuffer(intraPairsBuffer, offset: 0, index: 12)

        let tgs = MTLSize(width: min(populationSize, 256), height: 1, depth: 1)
        let tgc = MTLSize(width: (populationSize + tgs.width - 1) / tgs.width, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmdBuf.commit()
    }
}

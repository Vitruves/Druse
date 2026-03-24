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
    }

    // Backbone atom names that should NOT be made flexible
    private static let backboneAtomNames: Set<String> = ["N", "CA", "C", "O", "H", "HA"]

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
                flexAtoms: [], flexTorsionEdges: [], flexMovingIndices: [], chiSlotCount: 0
            )
        }

        let flexResidueSet = Set(flexConfig.flexibleResidueIndices)

        // Identify sidechain atom indices for each flexible residue
        var flexAtomIndices = Set<Int>()
        var residueAtomMap: [Int: [Int]] = [:] // residueSeq → [atom indices]

        for (i, atom) in proteinAtoms.enumerated() {
            if flexResidueSet.contains(atom.residueSeq) &&
               !Self.backboneAtomNames.contains(atom.name) {
                flexAtomIndices.insert(i)
                residueAtomMap[atom.residueSeq, default: []].append(i)
            }
        }

        // Log per-residue flex atom counts
        for (residueSeq, atomIndices) in residueAtomMap {
            Task { @MainActor in ActivityLog.shared.debug("Flex residue \(residueSeq): \(atomIndices.count) sidechain atoms excluded", category: .dock) }
        }

        // Build rigid atom list (everything except flex sidechain atoms)
        var rigidIndices: [Int] = []
        for i in proteinAtoms.indices {
            if !flexAtomIndices.contains(i) {
                rigidIndices.append(i)
            }
        }

        let rigidAtoms = rigidIndices.map { proteinAtoms[$0] }

        // Remap bonds
        var oldToNew: [Int: Int] = [:]
        for (newIdx, oldIdx) in rigidIndices.enumerated() {
            oldToNew[oldIdx] = newIdx
        }
        var rigidBonds: [Bond] = []
        for bond in proteinBonds {
            if let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] {
                rigidBonds.append(Bond(id: rigidBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }

        // Build flex sidechain atoms
        var gpuFlexAtoms: [FlexSidechainAtom] = []
        var flexLocalIndexMap: [Int: Int] = [:] // global atom index → local flex index

        // Sort residues for deterministic chi slot assignment
        let sortedFlexResidues = flexConfig.flexibleResidueIndices.sorted()

        var chiSlot = 0
        var allFlexEdges: [FlexTorsionEdge] = []
        var allFlexMoving: [Int32] = []

        for (resIdx, residueSeq) in sortedFlexResidues.enumerated() {
            guard let atomIndices = residueAtomMap[residueSeq] else { continue }

            // Get residue name
            guard let firstAtom = atomIndices.first else { continue }
            let residueName = proteinAtoms[firstAtom].residueName

            // Build local flex atom entries
            let localStart = gpuFlexAtoms.count
            for globalIdx in atomIndices {
                let atom = proteinAtoms[globalIdx]
                let localIdx = gpuFlexAtoms.count
                flexLocalIndexMap[globalIdx] = localIdx

                var fa = FlexSidechainAtom()
                fa.referencePosition = SIMD3<Float>(atom.position.x, atom.position.y, atom.position.z)
                fa.charge = atom.charge
                fa.vinaType = vinaTypes?[globalIdx] ?? -1
                fa.residueIndex = Int32(resIdx)
                gpuFlexAtoms.append(fa)
            }

            // Build chi angle torsion edges
            guard let rotDef = RotamerLibrary.rotamers(for: residueName) else { continue }

            // Map atom names to local flex indices for this residue
            let nameToLocal: [String: Int] = Dictionary(
                atomIndices.compactMap { globalIdx -> (String, Int)? in
                    guard let localIdx = flexLocalIndexMap[globalIdx] else { return nil }
                    return (proteinAtoms[globalIdx].name, localIdx)
                },
                uniquingKeysWith: { first, _ in first }
            )

            // Also include backbone atoms that may be needed as dihedral references
            // (they won't move, but they define the rotation axis)
            let backboneNameToGlobal: [String: Int] = Dictionary(
                proteinAtoms.indices.compactMap { i -> (String, Int)? in
                    let a = proteinAtoms[i]
                    guard a.residueSeq == residueSeq,
                          Self.backboneAtomNames.contains(a.name) else { return nil }
                    return (a.name, i)
                },
                uniquingKeysWith: { first, _ in first }
            )

            for chi in rotDef.chiAngles {
                // Find pivot and axis atoms (atoms 1 and 2 of the dihedral definition = atoms.1, atoms.2)
                let pivotName = chi.atomNames.1
                let axisName = chi.atomNames.2

                // Pivot might be backbone (e.g., "CA") — need to handle that
                let pivotLocal: Int?
                if let p = nameToLocal[pivotName] {
                    pivotLocal = p
                } else if let globalIdx = backboneNameToGlobal[pivotName] {
                    // Add backbone atom as a "pseudo" flex atom (won't actually move, just defines rotation axis)
                    let localIdx = gpuFlexAtoms.count
                    let atom = proteinAtoms[globalIdx]
                    var fa = FlexSidechainAtom()
                    fa.referencePosition = SIMD3<Float>(atom.position.x, atom.position.y, atom.position.z)
                    fa.charge = atom.charge
                    fa.vinaType = -1  // backbone pivot: excluded from pairwise scoring
                    fa.residueIndex = Int32(resIdx)
                    gpuFlexAtoms.append(fa)
                    pivotLocal = localIdx
                } else {
                    continue
                }

                guard let axisLocal = nameToLocal[axisName], let pivotLocal else { continue }

                // Find downstream atoms (everything past the axis atom in the sidechain)
                let downstream = findDownstreamLocalIndices(
                    axisLocal: axisLocal,
                    pivotLocal: pivotLocal,
                    localRange: localStart..<gpuFlexAtoms.count,
                    atoms: gpuFlexAtoms
                )

                let movingStart = Int32(allFlexMoving.count)
                allFlexMoving.append(contentsOf: downstream.map { Int32($0) })

                var edge = FlexTorsionEdge()
                edge.pivotAtom = Int32(pivotLocal)
                edge.axisAtom = Int32(axisLocal)
                edge.movingStart = movingStart
                edge.movingCount = Int32(downstream.count)
                edge.chiSlot = Int32(chiSlot)
                allFlexEdges.append(edge)

                chiSlot += 1
            }
        }

        return FlexExclusion(
            rigidAtoms: rigidAtoms,
            rigidBonds: rigidBonds,
            flexAtoms: gpuFlexAtoms,
            flexTorsionEdges: allFlexEdges,
            flexMovingIndices: allFlexMoving,
            chiSlotCount: chiSlot
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
        params.numFlexResidues = UInt32(flexConfig?.flexibleResidueIndices.count ?? 0)
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

        Task { @MainActor in ActivityLog.shared.debug("Dispatching flex scoring: popSize=\(populationSize), flexAtoms=\(self.numFlexAtoms)", category: .dock) }

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

        Task { @MainActor in ActivityLog.shared.debug("Dispatching chi evolution: popSize=\(populationSize), chiSlots=\(self.numChiSlots)", category: .dock) }

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
        exclusionMaskBuffer: MTLBuffer,
        populationSize: Int
    ) {
        guard isEnabled,
              let pipeline = localSearchFlexPipeline,
              let fab = flexAtomBuffer,
              let feb = flexTorsionEdgeBuffer,
              let fmb = flexMovingIndicesBuffer,
              let fpb = flexParamsBuffer
        else { return }

        Task { @MainActor in ActivityLog.shared.debug("Dispatching flex local search: popSize=\(populationSize), torsions=\(self.numFlexTorsions)", category: .dock) }

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
        enc.setBuffer(exclusionMaskBuffer, offset: 0, index: 12)

        let tgs = MTLSize(width: min(populationSize, 256), height: 1, depth: 1)
        let tgc = MTLSize(width: (populationSize + tgs.width - 1) / tgs.width, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmdBuf.commit()
    }
}

// MARK: - Atom Extension for Vina Type

private extension Atom {
    /// Quick lookup for the Vina XS type assigned during preparation.
    /// Returns the stored vinaType if available, otherwise -1 (untyped).
    var vinaType: Int? {
        // vinaType is typically set during protein preparation on the formalCharge field
        // or via the Vina type assignment pass. Return nil if not assigned.
        return nil
    }
}

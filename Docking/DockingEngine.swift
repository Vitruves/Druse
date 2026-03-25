import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Docking Engine

@MainActor
final class DockingEngine {
    let device: MTLDevice
    private(set) var commandQueue: MTLCommandQueue

    var stericGridPipeline: MTLComputePipelineState!
    var hydrophobicGridPipeline: MTLComputePipelineState!
    var hbondGridPipeline: MTLComputePipelineState!
    var vinaAffinityGridPipeline: MTLComputePipelineState!
    var scorePipeline: MTLComputePipelineState!
    var initPopPipeline: MTLComputePipelineState!
    var evolvePipeline: MTLComputePipelineState!
    var localSearchPipeline: MTLComputePipelineState!
    var mcPerturbPipeline: MTLComputePipelineState!
    var metropolisAcceptPipeline: MTLComputePipelineState!
    var explicitScorePipeline: MTLComputePipelineState!
    var localSearchAnalyticalPipeline: MTLComputePipelineState!
    var localSearchAnalyticalSIMDPipeline: MTLComputePipelineState!
    var drusinaScorePipeline: MTLComputePipelineState?
    var drusinaCorrectionPipeline: MTLComputePipelineState?

    // Parallel Tempering / Replica Exchange pipelines
    var mcPerturbReplicaPipeline: MTLComputePipelineState?
    var metropolisAcceptReplicaPipeline: MTLComputePipelineState?
    var replicaSwapPipeline: MTLComputePipelineState?
    var replicaParamsBuffer: MTLBuffer?

    /// Active local search pipeline based on config.useAnalyticalGradients.
    var activeLocalSearchPipeline: MTLComputePipelineState {
        config.useAnalyticalGradients ? localSearchAnalyticalSIMDPipeline : localSearchPipeline
    }

    /// Whether active local search uses SIMD-cooperative dispatch (32 threads per threadgroup).
    var localSearchIsSIMD: Bool {
        config.useAnalyticalGradients
    }

    private(set) var stericGridBuffer: MTLBuffer?
    var hydrophobicGridBuffer: MTLBuffer?
    var hbondGridBuffer: MTLBuffer?
    private(set) var vinaAffinityGridBuffer: MTLBuffer?
    var vinaTypeIndexBuffer: MTLBuffer?
    var vinaAffinityTypeBuffer: MTLBuffer?
    var proteinAtomBuffer: MTLBuffer?
    private(set) var gridParamsBuffer: MTLBuffer?
    var populationBuffer: MTLBuffer?
    var offspringBuffer: MTLBuffer?
    var bestPopulationBuffer: MTLBuffer?
    var ligandAtomBuffer: MTLBuffer?
    var gaParamsBuffer: MTLBuffer?
    var gaParamsRing: [MTLBuffer] = []
    var pairwiseRMSDPipeline: MTLComputePipelineState?
    var torsionEdgeBuffer: MTLBuffer?
    var movingIndicesBuffer: MTLBuffer?
    var intraPairsBuffer: MTLBuffer?

    // Drusina extended scoring buffers
    var proteinRingBuffer: MTLBuffer?
    var ligandRingBuffer: MTLBuffer?
    var proteinCationBuffer: MTLBuffer?
    var drusinaParamsBuffer: MTLBuffer?
    var halogenInfoBuffer: MTLBuffer?
    var proteinAmideBuffer: MTLBuffer?
    var chalcogenInfoBuffer: MTLBuffer?
    var saltBridgeGroupBuffer: MTLBuffer?

    // Pharmacophore constraint buffers
    var pharmaConstraintBuffer: MTLBuffer?
    var pharmaParamsBuffer: MTLBuffer?

    // DruseAF ML scoring (native Metal neural network)
    var druseAFSetupPipeline: MTLComputePipelineState?
    var druseAFEncodePipeline: MTLComputePipelineState?
    var druseAFScorePipeline: MTLComputePipelineState?
    var druseAFWeights: DruseAFWeightLoader.LoadedWeights?
    var druseAFParamsBuffer: MTLBuffer?
    var druseAFProtFeatBuffer: MTLBuffer?
    var druseAFLigFeatBuffer: MTLBuffer?
    var druseAFProtPosBuffer: MTLBuffer?
    var druseAFSetupBuffer: MTLBuffer?
    var druseAFIntermediateBuffer: MTLBuffer?
    var druseAFIntermediateCapacity: Int = 0

    var isRunning = false
    var currentGeneration = 0
    var bestEnergy: Float = .infinity

    /// Grid dimensions from last computeGridMaps call (for flex grid proxy)
    private(set) var lastGridTotalPoints: Int = 0
    private(set) var lastGridNumAffinityTypes: Int = 0

    /// Optional flex docking engine for receptor flexibility (induced fit).
    var flexEngine: FlexDockingEngine?
    var gridParams = GridParams()
    var config = DockingConfig()
    /// Tracks the last allocated population buffer capacity to avoid redundant reallocation.
    var lastPopulationBufferCapacity: Int = 0
    /// Tracks last allocated ligand buffer capacities (in bytes) to avoid churn during batch docking.
    var lastLigandAtomBufferCapacity: Int = 0
    var lastTorsionEdgeBufferCapacity: Int = 0
    var lastMovingIndicesBufferCapacity: Int = 0
    var lastIntraPairsBufferCapacity: Int = 0

    /// Diagnostics from the last completed docking run.
    var lastDiagnostics: DockingDiagnostics?

    var onPoseUpdate: ((DockingResult, [MolecularInteraction]) -> Void)?
    var onGenerationComplete: ((Int, Float) -> Void)?
    var onDockingComplete: (([DockingResult]) -> Void)?

    // Reference to protein atoms for interaction detection
    var proteinAtoms: [Atom] = []
    var proteinStructure: Molecule?

    /// Set the protein for subsequent docking runs.
    func setProtein(_ atoms: [Atom], _ bonds: [Bond]) {
        let mol = Molecule(name: "protein", atoms: atoms, bonds: bonds, title: "")
        proteinStructure = mol
        proteinAtoms = atoms.filter { $0.element != .H }
    }

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else { return nil }
        self.commandQueue = queue

        do {
            stericGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeStericGrid")!)
            hydrophobicGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHydrophobicGrid")!)
            hbondGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHBondGrid")!)
            vinaAffinityGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeVinaAffinityMaps")!)
            scorePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "scorePoses")!)
            initPopPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "initializePopulation")!)
            evolvePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "gaEvolve")!)
            localSearchPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "localSearch")!)
            localSearchAnalyticalPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "localSearchAnalytical")!)
            localSearchAnalyticalSIMDPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "localSearchAnalyticalSIMD")!)
            mcPerturbPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "mcPerturb")!)
            metropolisAcceptPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "metropolisAccept")!)
            explicitScorePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "scorePosesExplicit")!)
            if let drusinaFunc = library.makeFunction(name: "scorePosesDrusina") {
                drusinaScorePipeline = try device.makeComputePipelineState(function: drusinaFunc)
            }
            if let drusinaCorrFunc = library.makeFunction(name: "applyDrusinaCorrection") {
                drusinaCorrectionPipeline = try device.makeComputePipelineState(function: drusinaCorrFunc)
            }
            if let rmsdFunction = library.makeFunction(name: "computePairwiseRMSD") {
                pairwiseRMSDPipeline = try device.makeComputePipelineState(function: rmsdFunction)
            }
            // Parallel Tempering / Replica Exchange
            if let perturbFunc = library.makeFunction(name: "mcPerturbReplica"),
               let acceptFunc = library.makeFunction(name: "metropolisAcceptReplica"),
               let swapFunc = library.makeFunction(name: "replicaSwap") {
                mcPerturbReplicaPipeline = try device.makeComputePipelineState(function: perturbFunc)
                metropolisAcceptReplicaPipeline = try device.makeComputePipelineState(function: acceptFunc)
                replicaSwapPipeline = try device.makeComputePipelineState(function: swapFunc)
            }
            // DruseAF Metal neural network scoring
            if let setupFunc = library.makeFunction(name: "druseAFSetup"),
               let encodeFunc = library.makeFunction(name: "druseAFEncode"),
               let scoreFunc = library.makeFunction(name: "druseAFScore") {
                druseAFSetupPipeline = try device.makeComputePipelineState(function: setupFunc)
                druseAFEncodePipeline = try device.makeComputePipelineState(function: encodeFunc)
                druseAFScorePipeline = try device.makeComputePipelineState(function: scoreFunc)
                druseAFWeights = DruseAFWeightLoader.loadFromBundle(device: device)
            }
        } catch {
            print("Failed to create docking pipelines: \(error)")
            return nil
        }
    }

    // MARK: - Vina Atom Typing

    func hasAttachedHydrogen(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains { molecule.atoms[$0].element == .H }
    }

    func isBondedToHeteroatom(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains {
            let element = molecule.atoms[$0].element
            return element != .H && element != .C
        }
    }

    func vinaTypeID(_ type: VinaAtomType) -> Int32 {
        Int32(type.rawValue)
    }

    var maxSupportedVinaType: Int32 {
        vinaTypeID(VINA_OTHER)
    }

    func vinaTypesForProtein(_ molecule: Molecule) -> [Int32] {
        molecule.atoms.indices.map { vinaProteinAtomType(for: $0, in: molecule) }
    }

    func vinaProteinAtomType(for atomIndex: Int, in molecule: Molecule) -> Int32 {
        let atom = molecule.atoms[atomIndex]
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        let res = atom.residueName
        let donor = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
            name == "N" || name == "NZ" || name == "NE" || name == "NH1" || name == "NH2" ||
            name == "ND2" || name == "NE2"

        switch atom.element {
        case .C:
            return isBondedToHeteroatom(atomIndex: atomIndex, in: molecule) ? vinaTypeID(VINA_C_P) : vinaTypeID(VINA_C_H)

        case .N:
            let acceptor = res == "HIS" && (name == "ND1" || name == "NE2")
            if donor && acceptor { return vinaTypeID(VINA_N_DA) }
            if acceptor { return vinaTypeID(VINA_N_A) }
            if donor { return vinaTypeID(VINA_N_D) }
            return vinaTypeID(VINA_N_P)

        case .O:
            let donorO = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
                name == "OG" || name == "OG1" || name == "OH"
            let acceptor = atom.formalCharge <= 0 || name == "O" || name.hasPrefix("OD") || name.hasPrefix("OE")
            if donorO && acceptor { return vinaTypeID(VINA_O_DA) }
            if acceptor { return vinaTypeID(VINA_O_A) }
            if donorO { return vinaTypeID(VINA_O_D) }
            return vinaTypeID(VINA_O_P)

        case .S:  return vinaTypeID(VINA_S_P)
        case .P:  return vinaTypeID(VINA_P_P)
        case .F:  return vinaTypeID(VINA_F_H)
        case .Cl: return vinaTypeID(VINA_Cl_H)
        case .Br: return vinaTypeID(VINA_Br_H)
        case .Na, .Mg, .Ca, .Sc, .Ti, .V, .Cr, .Mn, .Fe, .Co, .Ni, .Cu, .Zn:
            return vinaTypeID(VINA_MET_D)
        default:  return vinaTypeID(VINA_OTHER)
        }
    }

    func fallbackLigandVinaAtomType(for atomIndex: Int, in molecule: Molecule) -> Int32 {
        let atom = molecule.atoms[atomIndex]
        let donor = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
            (atom.element == .N && atom.formalCharge > 0)
        let acceptor: Bool

        switch atom.element {
        case .N:
            let heavyNeighborCount = molecule.neighbors(of: atomIndex).filter { molecule.atoms[$0].element != .H }.count
            acceptor = atom.formalCharge <= 0 && !donor && heavyNeighborCount < 4
        case .O:
            acceptor = atom.formalCharge <= 0
        default:
            acceptor = false
        }

        switch atom.element {
        case .C:
            return isBondedToHeteroatom(atomIndex: atomIndex, in: molecule) ? vinaTypeID(VINA_C_P) : vinaTypeID(VINA_C_H)
        case .N:
            if donor && acceptor { return vinaTypeID(VINA_N_DA) }
            if acceptor { return vinaTypeID(VINA_N_A) }
            if donor { return vinaTypeID(VINA_N_D) }
            return vinaTypeID(VINA_N_P)
        case .O:
            if donor && acceptor { return vinaTypeID(VINA_O_DA) }
            if acceptor { return vinaTypeID(VINA_O_A) }
            if donor { return vinaTypeID(VINA_O_D) }
            return vinaTypeID(VINA_O_P)
        case .S:  return vinaTypeID(VINA_S_P)
        case .P:  return vinaTypeID(VINA_P_P)
        case .F:  return vinaTypeID(VINA_F_H)
        case .Cl: return vinaTypeID(VINA_Cl_H)
        case .Br: return vinaTypeID(VINA_Br_H)
        case .Na, .Mg, .Ca, .Sc, .Ti, .V, .Cr, .Mn, .Fe, .Co, .Ni, .Cu, .Zn:
            return vinaTypeID(VINA_MET_D)
        default:  return vinaTypeID(VINA_OTHER)
        }
    }

    func ligandVinaTypes(_ ligand: Molecule) -> [Int32] {
        guard !ligand.atoms.isEmpty else { return [] }

        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        if let rdkitTypes = RDKitBridge.computeVinaTypesMolBlock(molBlock, atomCount: ligand.atoms.count),
           rdkitTypes.count == ligand.atoms.count {
            return rdkitTypes
        }

        return ligand.atoms.indices.map { fallbackLigandVinaAtomType(for: $0, in: ligand) }
    }

    func swiftXSIsHydrophobic(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_C_H) || xsType == vinaTypeID(VINA_F_H) ||
        xsType == vinaTypeID(VINA_Cl_H) || xsType == vinaTypeID(VINA_Br_H) ||
        xsType == vinaTypeID(VINA_I_H)
    }

    func swiftXSIsAcceptor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_A) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_A) || xsType == vinaTypeID(VINA_O_DA)
    }

    func swiftXSIsDonor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_D) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_D) || xsType == vinaTypeID(VINA_O_DA) ||
        xsType == vinaTypeID(VINA_MET_D)
    }

    func swiftXSRadius(_ xsType: Int32) -> Float {
        let radii: [Float] = [
            1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7,
            2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2
        ]
        let index = Int(xsType)
        guard index >= 0, index < radii.count else { return 0 }
        return radii[index]
    }

    func swiftSlopeStep(xBad: Float, xGood: Float, x: Float) -> Float {
        if xBad < xGood {
            if x <= xBad { return 0 }
            if x >= xGood { return 1 }
        } else {
            if x >= xBad { return 0 }
            if x <= xGood { return 1 }
        }
        return (x - xBad) / (xGood - xBad)
    }

    func swiftVinaPairEnergy(_ type1: Int32, _ type2: Int32, distance r: Float) -> Float {
        guard r < 8.0,
              type1 >= 0, type1 <= maxSupportedVinaType,
              type2 >= 0, type2 <= maxSupportedVinaType else {
            return 0
        }

        let d = r - (swiftXSRadius(type1) + swiftXSRadius(type2))
        let gauss1 = -0.035579 * exp(-pow(d * 2.0, 2.0))
        let gauss2 = -0.005156 * exp(-pow((d - 3.0) * 0.5, 2.0))
        let repulsion = d < 0 ? 0.840245 * d * d : 0
        let hydrophobic = (swiftXSIsHydrophobic(type1) && swiftXSIsHydrophobic(type2))
            ? -0.035069 * swiftSlopeStep(xBad: 1.5, xGood: 0.5, x: d)
            : 0
        let hbond = ((swiftXSIsDonor(type1) && swiftXSIsAcceptor(type2)) ||
                     (swiftXSIsDonor(type2) && swiftXSIsAcceptor(type1)))
            ? -0.587439 * swiftSlopeStep(xBad: 0.0, xGood: -0.7, x: d)
            : 0
        return gauss1 + gauss2 + repulsion + hydrophobic + hbond
    }

    func intramolecularReferenceEnergy(
        ligandAtoms: [DockLigandAtom],
        pairList: [UInt32]
    ) -> Float {
        guard ligandAtoms.count > 1 else { return 0 }

        var total: Float = 0
        for packed in pairList {
            let i = Int(packed & 0xFFFF)
            let j = Int(packed >> 16)
            guard i < ligandAtoms.count, j < ligandAtoms.count else { continue }
            let r = simd_distance(ligandAtoms[i].position, ligandAtoms[j].position)
            total += swiftVinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, distance: r)
        }
        return total
    }

    // MARK: - Grid Map Computation

    func computeGridMaps(protein: Molecule, pocket: BindingPocket, spacing: Float = 0.375,
                          ligandExtent: SIMD3<Float>? = nil,
                          requiredVinaTypes: [Int32] = []) {
        proteinStructure = protein
        let heavyAtoms = protein.atoms.filter { $0.element != .H }
        self.proteinAtoms = heavyAtoms

        let gpuAtoms = protein.atoms.enumerated().compactMap { atomIndex, atom -> GridProteinAtom? in
            guard atom.element != .H else { return nil }
            let atomName = atom.name.trimmingCharacters(in: .whitespaces)
            let resName = atom.residueName.trimmingCharacters(in: .whitespaces)
            var atomFlags: UInt32 = 0
            let posAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE", "CZ"]
            let posRes: Set<String> = ["LYS", "ARG"]
            if (posRes.contains(resName) && posAtoms.contains(atomName)) || atom.formalCharge > 0 {
                atomFlags |= UInt32(GRPROT_FLAG_POS_CHARGED)
            }
            if resName == "HIS" && (atomName == "NE2" || atomName == "ND1") && atom.formalCharge > 0 {
                atomFlags |= UInt32(GRPROT_FLAG_POS_CHARGED)
            }
            let negAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]
            let negRes: Set<String> = ["ASP", "GLU"]
            if negRes.contains(resName) && negAtoms.contains(atomName) {
                atomFlags |= UInt32(GRPROT_FLAG_NEG_CHARGED)
            }
            return GridProteinAtom(
                position: atom.position,
                vdwRadius: atom.element.vdwRadius,
                charge: electrostaticCharge(for: atom),
                vinaType: vinaProteinAtomType(for: atomIndex, in: protein),
                flags: atomFlags, _pad1: 0
            )
        }

        let activeVinaTypes = Array(Set(requiredVinaTypes.filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()

        let gridMapCount: UInt64 = activeVinaTypes.isEmpty ? 3 : UInt64(3 + activeVinaTypes.count)

        let searchPadding: Float = 0.0
        let gridPadding: Float = 3.0
        let ligandMargin = ligandExtent ?? SIMD3<Float>(repeating: 4.0)
        let searchCenter = pocket.center
        let searchHalfExtent = pocket.size + SIMD3<Float>(repeating: searchPadding)
        let gridHalfExtent = searchHalfExtent + ligandMargin + SIMD3<Float>(repeating: gridPadding)
        let boxMin = searchCenter - gridHalfExtent
        let boxMax = searchCenter + gridHalfExtent
        let boxSize = boxMax - boxMin

        var effectiveSpacing = spacing
        let maxGridFloatValues: UInt64 = 48_000_000

        func gridDim(_ length: Float, _ sp: Float) -> UInt64 {
            let raw = ceil(length / sp)
            guard raw.isFinite && raw > 0 else { return 1 }
            return UInt64(min(raw, 10000))
        }
        let ex = gridDim(boxSize.x, effectiveSpacing)
        let ey = gridDim(boxSize.y, effectiveSpacing)
        let ez = gridDim(boxSize.z, effectiveSpacing)
        let estimatedPoints = ex * ey * ez
        if estimatedPoints * gridMapCount > maxGridFloatValues {
            let scaleFactor = pow(Float(estimatedPoints * gridMapCount) / Float(maxGridFloatValues), 1.0 / 3.0)
            effectiveSpacing = spacing * max(scaleFactor, 1.001)
        }
        let finalBoxSize = boxMax - boxMin
        var nx = UInt32(gridDim(finalBoxSize.x, effectiveSpacing)) + 1
        var ny = UInt32(gridDim(finalBoxSize.y, effectiveSpacing)) + 1
        var nz = UInt32(gridDim(finalBoxSize.z, effectiveSpacing)) + 1
        while UInt64(nx) * UInt64(ny) * UInt64(nz) * gridMapCount > maxGridFloatValues {
            effectiveSpacing *= 1.2
            nx = UInt32(gridDim(finalBoxSize.x, effectiveSpacing)) + 1
            ny = UInt32(gridDim(finalBoxSize.y, effectiveSpacing)) + 1
            nz = UInt32(gridDim(finalBoxSize.z, effectiveSpacing)) + 1
        }
        let totalPoints = UInt32(UInt64(nx) * UInt64(ny) * UInt64(nz))

        gridParams = GridParams(
            origin: boxMin, spacing: effectiveSpacing,
            dims: SIMD3(nx, ny, nz), _pad0: 0,
            totalPoints: totalPoints,
            numProteinAtoms: UInt32(gpuAtoms.count),
            numAffinityTypes: UInt32(activeVinaTypes.count), _pad2: 0,
            searchCenter: searchCenter, _pad3: 0,
            searchHalfExtent: searchHalfExtent, _pad4: 0
        )

        ActivityLog.shared.info(
            "[Engine] Grid: dims=\(nx)×\(ny)×\(nz) (\(totalPoints) points), spacing=\(String(format: "%.3f", effectiveSpacing)) Å, " +
            "\(activeVinaTypes.count) affinity types, \(gpuAtoms.count) protein atoms, " +
            "box=(\(String(format: "%.1f,%.1f,%.1f", boxMin.x, boxMin.y, boxMin.z)))→(\(String(format: "%.1f,%.1f,%.1f", boxMax.x, boxMax.y, boxMax.z)))",
            category: .dock
        )

        lastGridTotalPoints = Int(totalPoints)
        lastGridNumAffinityTypes = activeVinaTypes.count

        let vramEstimate = estimateVRAMUsage(
            gridDims: SIMD3(nx, ny, nz),
            numAffinityTypes: activeVinaTypes.count,
            populationSize: 300,
            numLigandAtoms: 50,
            numTorsions: 10,
            numProteinAtoms: gpuAtoms.count
        )
        if vramEstimate.usageRatio > 0.85 {
            print("[DockingEngine] WARNING: Estimated VRAM usage \(String(format: "%.0f", vramEstimate.totalMB))MB / \(String(format: "%.0f", vramEstimate.deviceBudgetMB))MB (\(String(format: "%.0f%%", vramEstimate.usageRatio * 100))) — coarsening grid spacing")
        }

        var proteinGPUAtoms = gpuAtoms
        proteinAtomBuffer = device.makeBuffer(bytes: &proteinGPUAtoms, length: proteinGPUAtoms.count * MemoryLayout<GridProteinAtom>.stride, options: .storageModeShared)
        gridParamsBuffer = device.makeBuffer(bytes: &gridParams, length: MemoryLayout<GridParams>.stride, options: .storageModeShared)

        let gridByteSize = Int(totalPoints) * MemoryLayout<UInt16>.stride
        stericGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hydrophobicGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hbondGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)

        if stericGridBuffer == nil || hydrophobicGridBuffer == nil || hbondGridBuffer == nil {
            ActivityLog.shared.error("[Engine] Grid buffer allocation failed: \(gridByteSize) bytes per map (\(totalPoints) points)", category: .dock)
            return
        }

        if activeVinaTypes.isEmpty {
            ActivityLog.shared.warn("[Engine] No valid Vina atom types — affinity grids will be empty (check ligand atom typing)", category: .dock)
            var zeroLookup = [Int32](repeating: -1, count: 32)
            vinaTypeIndexBuffer = device.makeBuffer(bytes: &zeroLookup, length: zeroLookup.count * MemoryLayout<Int32>.stride, options: .storageModeShared)
            vinaAffinityGridBuffer = device.makeBuffer(length: Int(totalPoints) * MemoryLayout<UInt16>.stride, options: .storageModeShared)
            vinaAffinityTypeBuffer = nil
        } else {
            let affinityGridSize = gridByteSize * activeVinaTypes.count
            vinaAffinityGridBuffer = device.makeBuffer(
                length: affinityGridSize,
                options: .storageModeShared
            )
            if vinaAffinityGridBuffer == nil {
                ActivityLog.shared.error("[Engine] Affinity grid allocation failed: \(affinityGridSize) bytes (\(activeVinaTypes.count) types × \(totalPoints) points)", category: .dock)
            }

            var typeLookup = [Int32](repeating: -1, count: 32)
            for (slot, type) in activeVinaTypes.enumerated() where Int(type) < typeLookup.count {
                typeLookup[Int(type)] = Int32(slot)
            }
            vinaTypeIndexBuffer = device.makeBuffer(
                bytes: &typeLookup,
                length: typeLookup.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
            var affinityTypes = activeVinaTypes
            vinaAffinityTypeBuffer = device.makeBuffer(
                bytes: &affinityTypes,
                length: affinityTypes.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        }

        let gridThreads = 128
        let tgSize = MTLSize(width: gridThreads, height: 1, depth: 1)
        let tgCount = MTLSize(width: (Int(totalPoints) + gridThreads - 1) / gridThreads, height: 1, depth: 1)

        guard let stericBuf = stericGridBuffer,
              let hydroBuf = hydrophobicGridBuffer,
              let hbondBuf = hbondGridBuffer,
              let stericPipe = stericGridPipeline,
              let hydroPipe = hydrophobicGridPipeline,
              let hbondPipe = hbondGridPipeline,
              let protAtomBuf = proteinAtomBuffer,
              let gParamsBuf = gridParamsBuffer else { return }

        func dispatchGridKernel(_ pipeline: MTLComputePipelineState, _ gridBuf: MTLBuffer,
                                threadGroups tg: MTLSize, label: String) {
            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gridBuf, offset: 0, index: 0)
            enc.setBuffer(protAtomBuf, offset: 0, index: 1)
            enc.setBuffer(gParamsBuf, offset: 0, index: 2)
            enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status == .error {
                ActivityLog.shared.error("[Engine] Grid \(label) GPU error: \(cb.error?.localizedDescription ?? "unknown")", category: .dock)
            }
        }

        dispatchGridKernel(stericPipe, stericBuf, threadGroups: tgCount, label: "steric")
        dispatchGridKernel(hydroPipe, hydroBuf, threadGroups: tgCount, label: "hydrophobic")
        dispatchGridKernel(hbondPipe, hbondBuf, threadGroups: tgCount, label: "hbond")

        if let affinityBuf = vinaAffinityGridBuffer,
           let affinityTypes = vinaAffinityTypeBuffer {
            let affinityEntryCount = Int(totalPoints) * activeVinaTypes.count
            let affinityTGCount = MTLSize(
                width: (affinityEntryCount + gridThreads - 1) / gridThreads,
                height: 1, depth: 1
            )
            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(vinaAffinityGridPipeline)
            enc.setBuffer(affinityBuf, offset: 0, index: 0)
            enc.setBuffer(protAtomBuf, offset: 0, index: 1)
            enc.setBuffer(gParamsBuf, offset: 0, index: 2)
            enc.setBuffer(affinityTypes, offset: 0, index: 3)
            enc.dispatchThreadgroups(affinityTGCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status == .error {
                ActivityLog.shared.error("[Engine] Affinity grid GPU error: \(cb.error?.localizedDescription ?? "unknown")", category: .dock)
            }
        }
    }

    func gridSnapshot() -> DockingGridSnapshot? {
        guard let stericGridBuffer,
              let hydrophobicGridBuffer,
              let hbondGridBuffer,
              let gridParamsBuffer else {
            return nil
        }
        return DockingGridSnapshot(
            stericGridBuffer: stericGridBuffer,
            hydrophobicGridBuffer: hydrophobicGridBuffer,
            hbondGridBuffer: hbondGridBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gridParams: gridParams,
            vinaAffinityGridBuffer: vinaAffinityGridBuffer,
            vinaTypeIndexBuffer: vinaTypeIndexBuffer
        )
    }

    // MARK: - Ligand Preparation

    func prepareLigandGeometry(_ ligand: Molecule) -> PreparedDockingLigand {
        let chargedLigand = ligandWithDockingCharges(ligand)
        let allVinaTypes = ligandVinaTypes(chargedLigand)
        let heavyEntries = chargedLigand.atoms.enumerated().filter { $0.element.element != .H }
        let validHeavy = heavyEntries.filter { e in
            let p = e.element.position
            return !p.x.isNaN && !p.y.isNaN && !p.z.isNaN
        }
        let heavyAtoms = validHeavy.map(\.element)
        let centroid = heavyAtoms.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(max(heavyAtoms.count, 1))

        var oldToNew: [Int: Int] = [:]
        for (newIdx, entry) in validHeavy.enumerated() {
            oldToNew[entry.offset] = newIdx
        }

        let heavyBonds: [Bond] = chargedLigand.bonds.compactMap { bond in
            guard let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }

        let gpuAtoms: [DockLigandAtom] = validHeavy.map { entry in
            let atom = entry.element
            return DockLigandAtom(
                position: atom.position - centroid,
                vdwRadius: atom.element.vdwRadius,
                charge: electrostaticCharge(for: atom),
                vinaType: allVinaTypes.indices.contains(entry.offset)
                    ? allVinaTypes[entry.offset]
                    : fallbackLigandVinaAtomType(for: entry.offset, in: chargedLigand),
                formalCharge: Int32(atom.formalCharge),
                _pad1: 0, _pad2: 0
            )
        }

        return PreparedDockingLigand(
            heavyAtoms: heavyAtoms,
            heavyBonds: heavyBonds,
            centroid: centroid,
            gpuAtoms: gpuAtoms
        )
    }

    func prepareLigandGA(_ ligand: Molecule) -> LigandGAData? {
        let prepared = prepareLigandGeometry(ligand)
        let gpuAtoms = prepared.gpuAtoms
        guard gpuAtoms.count <= 128, gpuAtoms.count > 0 else { return nil }

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: prepared.heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1), atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count), movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map { Int32($0) })
        }
        if torsionEdges.isEmpty { torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0)) }
        if movingIndices.isEmpty { movingIndices.append(0) }

        var excluded = Set<UInt32>()
        var adj: [[Int]] = Array(repeating: [], count: gpuAtoms.count)
        for bond in prepared.heavyBonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < gpuAtoms.count, b < gpuAtoms.count else { continue }
            adj[a].append(b); adj[b].append(a)
        }
        for i in 0..<gpuAtoms.count {
            for j in adj[i] where j > i {
                excluded.insert(UInt32(i) | (UInt32(j) << 16))
                for k in adj[j] where k > i && k != i {
                    let lo = min(i, k), hi = max(i, k)
                    excluded.insert(UInt32(lo) | (UInt32(hi) << 16))
                }
                for k in adj[i] where k != j {
                    let lo = min(j, k), hi = max(j, k)
                    excluded.insert(UInt32(lo) | (UInt32(hi) << 16))
                }
            }
        }
        var pairList = [UInt32]()
        for i in 0..<gpuAtoms.count {
            for j in (i+1)..<gpuAtoms.count {
                let packed = UInt32(i) | (UInt32(j) << 16)
                if !excluded.contains(packed) { pairList.append(packed) }
            }
        }

        let refIntraE = intramolecularReferenceEnergy(ligandAtoms: gpuAtoms, pairList: pairList)
        let ligRad = max(sqrt(gpuAtoms.map { simd_length_squared($0.position) }.reduce(0, +) / Float(max(gpuAtoms.count, 1))), 1.0)

        return LigandGAData(
            gpuAtoms: gpuAtoms, torsionEdges: torsionEdges, movingIndices: movingIndices,
            pairList: pairList, referenceIntraEnergy: refIntraE, ligandRadius: ligRad,
            centroid: prepared.centroid, heavyAtoms: prepared.heavyAtoms
        )
    }

    // MARK: - Drusina Buffer Preparation

    func prepareDrusinaBuffers(
        ligandAtoms: [Atom],
        ligandBonds: [Bond],
        gpuLigAtoms: [DockLigandAtom],
        centroid: SIMD3<Float>
    ) {
        // --- Protein aromatic rings ---
        let protRings = InteractionDetector.detectAromaticRings(atoms: proteinAtoms)
        var protRingGPU: [ProteinRingGPU] = protRings.map { ring in
            ProteinRingGPU(centroid: ring.centroid, _pad0: 0, normal: ring.normal, _pad1: 0)
        }
        if protRingGPU.isEmpty {
            protRingGPU.append(ProteinRingGPU(centroid: .zero, _pad0: 0, normal: .init(0,1,0), _pad1: 0))
        }
        proteinRingBuffer = device.makeBuffer(
            bytes: &protRingGPU,
            length: protRingGPU.count * MemoryLayout<ProteinRingGPU>.stride,
            options: .storageModeShared)

        // --- Ligand aromatic rings ---
        let ligRings = InteractionDetector.detectAromaticRings(
            atoms: ligandAtoms, bonds: ligandBonds)
        var ligRingGPU: [LigandRingGPU] = ligRings.map { ring in
            var indices: (Int32, Int32, Int32, Int32, Int32, Int32) = (-1, -1, -1, -1, -1, -1)
            let idxArray = ring.atomIndices.prefix(6).map { Int32($0) }
            if idxArray.count > 0 { indices.0 = idxArray[0] }
            if idxArray.count > 1 { indices.1 = idxArray[1] }
            if idxArray.count > 2 { indices.2 = idxArray[2] }
            if idxArray.count > 3 { indices.3 = idxArray[3] }
            if idxArray.count > 4 { indices.4 = idxArray[4] }
            if idxArray.count > 5 { indices.5 = idxArray[5] }
            return LigandRingGPU(
                atomIndices: indices,
                numAtoms: Int32(min(ring.atomIndices.count, 6)),
                _pad: 0)
        }
        if ligRingGPU.isEmpty {
            ligRingGPU.append(LigandRingGPU(atomIndices: (-1,-1,-1,-1,-1,-1), numAtoms: 0, _pad: 0))
        }
        ligandRingBuffer = device.makeBuffer(
            bytes: &ligRingGPU,
            length: ligRingGPU.count * MemoryLayout<LigandRingGPU>.stride,
            options: .storageModeShared)

        // --- Protein cations ---
        let cationNames: Set<String> = ["NZ", "NH1", "NH2"]
        var cations: [SIMD4<Float>] = []
        for atom in proteinAtoms {
            let name = atom.name.trimmingCharacters(in: .whitespaces)
            let isCation = cationNames.contains(name) || atom.formalCharge > 0 ||
                [Element.Zn, .Fe, .Mg, .Ca, .Mn, .Cu, .Co, .Ni].contains(atom.element)
            if isCation {
                cations.append(SIMD4(atom.position, Float(atom.formalCharge)))
            }
        }
        if cations.isEmpty {
            cations.append(.zero)
        }
        proteinCationBuffer = device.makeBuffer(
            bytes: &cations,
            length: cations.count * MemoryLayout<SIMD4<Float>>.stride,
            options: .storageModeShared)

        // --- Halogen bond info ---
        var halogens: [HalogenBondInfo] = []
        for (i, gpuAtom) in gpuLigAtoms.enumerated() {
            let vt = gpuAtom.vinaType
            let isHalogen = (vt == Int32(VINA_F_H.rawValue) || vt == Int32(VINA_Cl_H.rawValue) ||
                             vt == Int32(VINA_Br_H.rawValue) || vt == Int32(VINA_I_H.rawValue))
            guard isHalogen else { continue }
            for bond in ligandBonds {
                let partner: Int?
                if bond.atomIndex1 == i { partner = bond.atomIndex2 }
                else if bond.atomIndex2 == i { partner = bond.atomIndex1 }
                else { partner = nil }
                if let p = partner, p < ligandAtoms.count, ligandAtoms[p].element == .C {
                    halogens.append(HalogenBondInfo(halogenAtomIndex: Int32(i), carbonAtomIndex: Int32(p)))
                    break
                }
            }
        }
        if halogens.isEmpty {
            halogens.append(HalogenBondInfo(halogenAtomIndex: -1, carbonAtomIndex: -1))
        }
        halogenInfoBuffer = device.makeBuffer(
            bytes: &halogens,
            length: halogens.count * MemoryLayout<HalogenBondInfo>.stride,
            options: .storageModeShared)

        // --- Chalcogen bond info ---
        var chalcogens: [ChalcogenBondInfo] = []
        for (i, gpuAtom) in gpuLigAtoms.enumerated() {
            guard gpuAtom.vinaType == Int32(VINA_S_P.rawValue) else { continue }
            for bond in ligandBonds {
                let partner: Int?
                if bond.atomIndex1 == i { partner = bond.atomIndex2 }
                else if bond.atomIndex2 == i { partner = bond.atomIndex1 }
                else { partner = nil }
                if let p = partner, p < ligandAtoms.count, ligandAtoms[p].element == .C {
                    chalcogens.append(ChalcogenBondInfo(sulfurAtomIndex: Int32(i), carbonAtomIndex: Int32(p)))
                    break
                }
            }
        }
        if chalcogens.isEmpty {
            chalcogens.append(ChalcogenBondInfo(sulfurAtomIndex: -1, carbonAtomIndex: -1))
        }
        chalcogenInfoBuffer = device.makeBuffer(
            bytes: &chalcogens,
            length: chalcogens.count * MemoryLayout<ChalcogenBondInfo>.stride,
            options: .storageModeShared)

        // --- Protein backbone amide planes ---
        var amides: [ProteinAmideGPU] = []
        var residueAtoms: [String: [Atom]] = [:]
        for atom in proteinAtoms {
            let key = "\(atom.chainID)_\(atom.residueSeq)"
            residueAtoms[key, default: []].append(atom)
        }
        for (_, atoms) in residueAtoms {
            let bbC = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "C" && !$0.isHetAtom }
            let bbO = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "O" && !$0.isHetAtom }
            let bbN = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "N" && !$0.isHetAtom }
            guard let c = bbC, let o = bbO, let n = bbN else { continue }

            let centroid = (c.position + o.position + n.position) / 3.0
            let v1 = o.position - c.position
            let v2 = n.position - c.position
            var normal = simd_cross(v1, v2)
            let nLen = simd_length(normal)
            guard nLen > 1e-6 else { continue }
            normal /= nLen

            amides.append(ProteinAmideGPU(centroid: centroid, _pad0: 0, normal: normal, _pad1: 0))
        }
        // Sidechain amides (Asn, Gln)
        for (_, atoms) in residueAtoms {
            let resName = atoms.first?.residueName.trimmingCharacters(in: .whitespaces) ?? ""
            if resName == "ASN" {
                let cg = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "CG" }
                let od1 = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "OD1" }
                let nd2 = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "ND2" }
                guard let c = cg, let o = od1, let n = nd2 else { continue }
                let centroid = (c.position + o.position + n.position) / 3.0
                let v1 = o.position - c.position
                let v2 = n.position - c.position
                var normal = simd_cross(v1, v2)
                let nLen = simd_length(normal)
                guard nLen > 1e-6 else { continue }
                normal /= nLen
                amides.append(ProteinAmideGPU(centroid: centroid, _pad0: 0, normal: normal, _pad1: 0))
            } else if resName == "GLN" {
                let cd = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "CD" }
                let oe1 = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "OE1" }
                let ne2 = atoms.first { $0.name.trimmingCharacters(in: .whitespaces) == "NE2" }
                guard let c = cd, let o = oe1, let n = ne2 else { continue }
                let centroid = (c.position + o.position + n.position) / 3.0
                let v1 = o.position - c.position
                let v2 = n.position - c.position
                var normal = simd_cross(v1, v2)
                let nLen = simd_length(normal)
                guard nLen > 1e-6 else { continue }
                normal /= nLen
                amides.append(ProteinAmideGPU(centroid: centroid, _pad0: 0, normal: normal, _pad1: 0))
            }
        }
        if amides.isEmpty {
            amides.append(ProteinAmideGPU(centroid: .zero, _pad0: 0, normal: .init(0, 1, 0), _pad1: 0))
        }
        proteinAmideBuffer = device.makeBuffer(
            bytes: &amides,
            length: amides.count * MemoryLayout<ProteinAmideGPU>.stride,
            options: .storageModeShared)

        // --- Salt bridge groups ---
        var sbGroups: [SaltBridgeGroupGPU] = []
        var residueAtomsSB: [String: [Atom]] = [:]
        for atom in proteinAtoms {
            let key = "\(atom.chainID)_\(atom.residueSeq)"
            residueAtomsSB[key, default: []].append(atom)
        }
        let allPositions = proteinAtoms.map { $0.position }

        for (_, atoms) in residueAtomsSB {
            let resName = atoms.first?.residueName.trimmingCharacters(in: .whitespaces) ?? ""

            if resName == "ARG" {
                let guanAtoms = atoms.filter {
                    let n = $0.name.trimmingCharacters(in: .whitespaces)
                    return n == "NE" || n == "NH1" || n == "NH2" || n == "CZ"
                }
                if guanAtoms.count >= 2 {
                    let centroid = guanAtoms.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(guanAtoms.count)
                    let burial = Self.computeBurialFactor(centroid: centroid, allPositions: allPositions)
                    sbGroups.append(SaltBridgeGroupGPU(
                        centroid: centroid, chargeSign: 1,
                        burialFactor: burial, _pad0: 0, _pad1: 0, _pad2: 0))
                }
            }
            else if resName == "LYS" {
                if let nz = atoms.first(where: { $0.name.trimmingCharacters(in: .whitespaces) == "NZ" }) {
                    let burial = Self.computeBurialFactor(centroid: nz.position, allPositions: allPositions)
                    sbGroups.append(SaltBridgeGroupGPU(
                        centroid: nz.position, chargeSign: 1,
                        burialFactor: burial, _pad0: 0, _pad1: 0, _pad2: 0))
                }
            }
            else if resName == "HIS" {
                let hisN = atoms.filter {
                    let n = $0.name.trimmingCharacters(in: .whitespaces)
                    return (n == "NE2" || n == "ND1") && $0.formalCharge > 0
                }
                if !hisN.isEmpty {
                    let centroid = hisN.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(hisN.count)
                    let burial = Self.computeBurialFactor(centroid: centroid, allPositions: allPositions)
                    sbGroups.append(SaltBridgeGroupGPU(
                        centroid: centroid, chargeSign: 1,
                        burialFactor: burial, _pad0: 0, _pad1: 0, _pad2: 0))
                }
            }
            else if resName == "ASP" {
                let carb = atoms.filter {
                    let n = $0.name.trimmingCharacters(in: .whitespaces)
                    return n == "OD1" || n == "OD2"
                }
                if carb.count == 2 {
                    let centroid = (carb[0].position + carb[1].position) / 2.0
                    let burial = Self.computeBurialFactor(centroid: centroid, allPositions: allPositions)
                    sbGroups.append(SaltBridgeGroupGPU(
                        centroid: centroid, chargeSign: -1,
                        burialFactor: burial, _pad0: 0, _pad1: 0, _pad2: 0))
                }
            }
            else if resName == "GLU" {
                let carb = atoms.filter {
                    let n = $0.name.trimmingCharacters(in: .whitespaces)
                    return n == "OE1" || n == "OE2"
                }
                if carb.count == 2 {
                    let centroid = (carb[0].position + carb[1].position) / 2.0
                    let burial = Self.computeBurialFactor(centroid: centroid, allPositions: allPositions)
                    sbGroups.append(SaltBridgeGroupGPU(
                        centroid: centroid, chargeSign: -1,
                        burialFactor: burial, _pad0: 0, _pad1: 0, _pad2: 0))
                }
            }
        }
        if sbGroups.isEmpty {
            sbGroups.append(SaltBridgeGroupGPU(centroid: .zero, chargeSign: 0,
                                                burialFactor: 0, _pad0: 0, _pad1: 0, _pad2: 0))
        }
        saltBridgeGroupBuffer = device.makeBuffer(
            bytes: &sbGroups,
            length: sbGroups.count * MemoryLayout<SaltBridgeGroupGPU>.stride,
            options: .storageModeShared)

        // --- Drusina parameters ---
        var params = DrusinaParams(
            numProteinRings: UInt32(protRings.count),
            numLigandRings: UInt32(ligRings.count),
            numProteinCations: UInt32(max(cations.count - (cations.first == .zero ? 1 : 0), 0)),
            numHalogens: UInt32(halogens.first?.halogenAtomIndex == -1 ? 0 : halogens.count),
            wPiPi: -0.20,
            wPiCation: -0.40,
            wHalogenBond: -0.30,
            wMetalCoord: -0.60,
            numProteinAmides: UInt32(amides.first?.centroid == .zero && amides.count == 1 ? 0 : amides.count),
            numChalcogens: UInt32(chalcogens.first?.sulfurAtomIndex == -1 ? 0 : chalcogens.count),
            wSaltBridge: -0.45,
            wAmideStack: -0.20,
            wChalcogenBond: -0.15,
            numSaltBridgeGroups: UInt32(sbGroups.first?.chargeSign == 0 ? 0 : sbGroups.count))
        drusinaParamsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<DrusinaParams>.stride,
            options: .storageModeShared)
    }

    static func computeBurialFactor(centroid: SIMD3<Float>, allPositions: [SIMD3<Float>]) -> Float {
        var count: Float = 0
        for pos in allPositions {
            let d = simd_distance(centroid, pos)
            if d > 0.5 && d < 8.0 { count += 1 }
        }
        return max(0.15, min(1.0, count / 25.0))
    }

    // MARK: - DruseAF ML Scoring Buffers

    func prepareDruseAFBuffers(
        ligandAtoms: [Atom],
        gpuLigAtoms: [DockLigandAtom],
        pocket: BindingPocket,
        popSize: Int
    ) {
        guard let afWeights = druseAFWeights,
              let setupPipe = druseAFSetupPipeline else { return }

        let pocketAtoms = proteinAtoms.filter { atom in
            simd_distance(atom.position, pocket.center) <= 10.0
        }
        let P = min(pocketAtoms.count, 256)
        let L = min(ligandAtoms.count, 64)

        let protHybrid = DruseScoreFeatureExtractor.buildHybridizationMap(
            atoms: pocketAtoms, bonds: proteinStructure?.bonds ?? [])
        let ligHybrid = DruseScoreFeatureExtractor.buildHybridizationMap(
            atoms: ligandAtoms, bonds: [])

        var protFeats = [Float](repeating: 0, count: 256 * 20)
        for i in 0..<P {
            let f = DruseScoreFeatureExtractor.atomFeatures(
                pocketAtoms[i], isProtein: true,
                hybridization: protHybrid[pocketAtoms[i].id])
            for j in 0..<20 { protFeats[i * 20 + j] = f[j] }
        }
        druseAFProtFeatBuffer = device.makeBuffer(
            bytes: &protFeats, length: protFeats.count * 4, options: .storageModeShared)

        var ligFeats = [Float](repeating: 0, count: 64 * 20)
        for i in 0..<L {
            let f = DruseScoreFeatureExtractor.atomFeatures(
                ligandAtoms[i], isProtein: false,
                hybridization: ligHybrid[ligandAtoms[i].id])
            for j in 0..<20 { ligFeats[i * 20 + j] = f[j] }
        }
        druseAFLigFeatBuffer = device.makeBuffer(
            bytes: &ligFeats, length: ligFeats.count * 4, options: .storageModeShared)

        var protPos = [SIMD3<Float>](repeating: .zero, count: 256)
        for i in 0..<P { protPos[i] = pocketAtoms[i].position }
        druseAFProtPosBuffer = device.makeBuffer(
            bytes: &protPos, length: protPos.count * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared)

        var params = DruseAFParams(
            numProteinAtoms: UInt32(P),
            numLigandAtoms: UInt32(L),
            hiddenDim: 128,
            numHeads: 4,
            headDim: 32,
            rbfBins: 50,
            rbfGamma: 10.0,
            rbfSpacing: 10.0 / 49.0,
            numCrossAttnLayers: 2,
            numWeightTensors: UInt32(afWeights.numTensors),
            _pad0: 0, _pad1: 0
        )
        druseAFParamsBuffer = device.makeBuffer(
            bytes: &params, length: MemoryLayout<DruseAFParams>.stride,
            options: .storageModeShared)

        let setupFloats = 5 * 256 * 128 + 64 * 128
        druseAFSetupBuffer = device.makeBuffer(
            length: setupFloats * 4, options: .storageModeShared)

        let posFloatsPerPose = 64 * 3
        let requiredBytes = popSize * posFloatsPerPose * 4
        if requiredBytes > druseAFIntermediateCapacity {
            druseAFIntermediateBuffer = device.makeBuffer(
                length: requiredBytes, options: .storageModeShared)
            druseAFIntermediateCapacity = requiredBytes
        }

        guard let afParamsBuf = druseAFParamsBuffer,
              let afProtFeat = druseAFProtFeatBuffer,
              let afLigFeat = druseAFLigFeatBuffer,
              let afSetupBuf = druseAFSetupBuffer else { return }

        let totalAtoms = P + L
        let setupTgSize = MTLSize(width: min(totalAtoms, 256), height: 1, depth: 1)
        let setupTgCount = MTLSize(width: (totalAtoms + 255) / 256, height: 1, depth: 1)

        // Use direct command buffer dispatch (not dispatchCompute which checks isRunning)
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(setupPipe)
        enc.setBuffer(afProtFeat, offset: 0, index: 0)
        enc.setBuffer(afLigFeat, offset: 0, index: 1)
        enc.setBuffer(afWeights.weightBuffer, offset: 0, index: 2)
        enc.setBuffer(afWeights.entryBuffer, offset: 0, index: 3)
        enc.setBuffer(afParamsBuf, offset: 0, index: 4)
        enc.setBuffer(afSetupBuf, offset: 0, index: 5)
        enc.dispatchThreadgroups(setupTgCount, threadsPerThreadgroup: setupTgSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        ActivityLog.shared.info(
            "[DruseAF] Setup complete: \(P) protein atoms, \(L) ligand atoms, " +
            "setup=\(setupFloats * 4 / 1024) KB, intermediates=\(requiredBytes / 1024) KB/pose",
            category: .dock
        )
    }

    // MARK: - Pharmacophore Constraint Buffers

    func prepareConstraintBuffers(_ constraints: [PharmacophoreConstraintDef],
                                   atoms: [Atom], residues: [Residue]) {
        let (gpuConstraints, params) = PharmacophoreConstraintDef.toGPUBuffers(
            constraints: constraints, atoms: atoms, residues: residues
        )

        if gpuConstraints.isEmpty {
            var emptyConstraint = PharmacophoreConstraint()
            pharmaConstraintBuffer = device.makeBuffer(
                bytes: &emptyConstraint,
                length: MemoryLayout<PharmacophoreConstraint>.stride,
                options: .storageModeShared)
        } else {
            var mutableConstraints = gpuConstraints
            pharmaConstraintBuffer = device.makeBuffer(
                bytes: &mutableConstraints,
                length: MemoryLayout<PharmacophoreConstraint>.stride * gpuConstraints.count,
                options: .storageModeShared)
        }

        var mutableParams = params
        pharmaParamsBuffer = device.makeBuffer(
            bytes: &mutableParams,
            length: MemoryLayout<PharmacophoreParams>.stride,
            options: .storageModeShared)
    }

    // MARK: - VRAM Estimation

    var deviceVRAMBudgetMB: Float {
        Float(device.recommendedMaxWorkingSetSize) / (1024 * 1024)
    }

    func estimateVRAMUsage(
        gridDims: SIMD3<UInt32>,
        numAffinityTypes: Int,
        populationSize: Int,
        numLigandAtoms: Int,
        numTorsions: Int,
        numProteinAtoms: Int
    ) -> VRAMEstimate {
        let totalPoints = Int(UInt64(gridDims.x) * UInt64(gridDims.y) * UInt64(gridDims.z))
        let gridMapCount = 3 + numAffinityTypes
        let gridBytes = totalPoints * MemoryLayout<UInt16>.stride * gridMapCount

        let poseStride = 304
        let gaParamsStride = 192
        let populationBytes = 3 * populationSize * poseStride + 3 * gaParamsStride

        let ligandAtomStride = 32
        let torsionEdgeStride = 16
        let ligandBytes = numLigandAtoms * ligandAtomStride
            + numTorsions * torsionEdgeStride
            + numLigandAtoms * MemoryLayout<UInt32>.stride
            + (numLigandAtoms * numLigandAtoms + 31) / 32 * MemoryLayout<UInt32>.stride

        let proteinAtomBytes = numProteinAtoms * 32

        let miscBytes = 32 * MemoryLayout<Int32>.stride + numAffinityTypes * MemoryLayout<Int32>.stride
            + MemoryLayout<GridParams>.stride

        return VRAMEstimate(
            gridBytes: gridBytes,
            populationBytes: populationBytes,
            ligandBytes: ligandBytes,
            proteinAtomBytes: proteinAtomBytes,
            miscBytes: miscBytes,
            deviceBudgetMB: deviceVRAMBudgetMB
        )
    }

    func gridDiagnostics() -> String {
        var lines: [String] = []
        for (name, buf) in [("Steric", stericGridBuffer), ("Hydrophobic", hydrophobicGridBuffer),
                             ("HBond", hbondGridBuffer)] {
            guard let buf else { lines.append("  \(name): nil"); continue }
            let count = buf.length / MemoryLayout<Float16>.stride
            let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
            var minV: Float = .infinity, maxV: Float = -.infinity, nonZero = 0
            var sum: Float = 0
            for i in 0..<count {
                let v = Float(ptr[i])
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                if abs(v) > 1e-6 { nonZero += 1 }
                sum += v
            }
            lines.append("  \(name): \(count) pts, min=\(String(format: "%.3f", minV)) max=\(String(format: "%.3f", maxV)) nonzero=\(nonZero) mean=\(String(format: "%.4f", sum/Float(count)))")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Ligand Charges & Torsion Tree

    func ligandWithDockingCharges(_ ligand: Molecule) -> Molecule {
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        guard !heavyAtoms.isEmpty else { return ligand }

        let hasPartialCharges = heavyAtoms.contains { abs($0.charge) > 1e-4 }
        guard !hasPartialCharges else { return ligand }

        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        guard let charged = RDKitBridge.computeChargesMolBlock(molBlock),
              charged.atoms.count == ligand.atoms.count else {
            return ligand
        }

        var mergedAtoms = ligand.atoms
        for i in mergedAtoms.indices {
            mergedAtoms[i].charge = charged.atoms[i].charge
            if mergedAtoms[i].formalCharge == 0 {
                mergedAtoms[i].formalCharge = charged.atoms[i].formalCharge
            }
        }

        return Molecule(
            name: ligand.name,
            atoms: mergedAtoms,
            bonds: ligand.bonds,
            title: ligand.title,
            smiles: ligand.smiles
        )
    }

    func electrostaticCharge(for atom: Atom) -> Float {
        abs(atom.charge) > 1e-4 ? atom.charge : Float(atom.formalCharge)
    }

    func buildTorsionTree(for ligand: Molecule, heavyBonds: [Bond]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        if let tree = RDKitBridge.buildTorsionTreeMolBlock(molBlock), !tree.isEmpty {
            return tree
        }

        let smilesSource = ligand.smiles
            ?? (ligand.title.contains(where: { $0 == "(" || $0 == ")" || $0.isNumber })
                && !ligand.title.contains(" ")
                ? ligand.title : nil)

        if let smi = smilesSource, !smi.isEmpty,
           let tree = RDKitBridge.buildTorsionTree(smiles: smi), !tree.isEmpty {
            return tree
        }
        return buildGraphTorsionTree(atomCount: ligand.heavyAtomCount, bonds: heavyBonds, atoms: ligand.atoms.filter { $0.element != .H })
    }

    func buildGraphTorsionTree(atomCount: Int, bonds: [Bond], atoms: [Atom]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        guard atomCount > 1 else { return [] }

        var adjacency = Array(repeating: [Int](), count: atomCount)
        for bond in bonds {
            guard bond.atomIndex1 < atomCount, bond.atomIndex2 < atomCount else { continue }
            adjacency[bond.atomIndex1].append(bond.atomIndex2)
            adjacency[bond.atomIndex2].append(bond.atomIndex1)
        }

        func hasAlternatePath(from start: Int, to target: Int, excluding edge: (Int, Int)) -> Bool {
            var visited: Set<Int> = [start]
            var head = 0
            var queue = [start]
            while head < queue.count {
                let current = queue[head]
                head += 1
                for next in adjacency[current] {
                    if (current == edge.0 && next == edge.1) || (current == edge.1 && next == edge.0) {
                        continue
                    }
                    if next == target { return true }
                    if visited.insert(next).inserted {
                        queue.append(next)
                    }
                }
            }
            return false
        }

        func bfsSide(start: Int, excluding: Int) -> Set<Int> {
            var visited: Set<Int> = [start]
            var head = 0
            var queue = [start]
            while head < queue.count {
                let current = queue[head]
                head += 1
                for next in adjacency[current] where next != excluding {
                    if visited.insert(next).inserted {
                        queue.append(next)
                    }
                }
            }
            return visited
        }

        var bfsOrder = Array(repeating: Int.max, count: atomCount)
        var orderCounter = 0
        for root in 0..<atomCount where bfsOrder[root] == Int.max {
            var head = 0
            var queue = [root]
            bfsOrder[root] = orderCounter
            orderCounter += 1
            while head < queue.count {
                let current = queue[head]
                head += 1
                for next in adjacency[current] where bfsOrder[next] == Int.max {
                    bfsOrder[next] = orderCounter
                    orderCounter += 1
                    queue.append(next)
                }
            }
        }

        var bondOrderMap: [Int: [Int: BondOrder]] = [:]
        for bond in bonds {
            bondOrderMap[bond.atomIndex1, default: [:]][bond.atomIndex2] = bond.order
            bondOrderMap[bond.atomIndex2, default: [:]][bond.atomIndex1] = bond.order
        }

        func isAmideBond(_ a: Int, _ b: Int) -> Bool {
            guard a < atoms.count, b < atoms.count else { return false }
            let elemA = atoms[a].element
            let elemB = atoms[b].element
            let carbonIdx: Int
            if elemA == .C && elemB == .N {
                carbonIdx = a
            } else if elemA == .N && elemB == .C {
                carbonIdx = b
            } else {
                return false
            }
            guard let neighbors = bondOrderMap[carbonIdx] else { return false }
            for (neighborIdx, order) in neighbors {
                guard order == .double, neighborIdx < atoms.count else { continue }
                let neighborElem = atoms[neighborIdx].element
                if neighborElem == .O || neighborElem == .S {
                    return true
                }
            }
            return false
        }

        var torsions: [(atom1: Int, atom2: Int, movingAtoms: [Int])] = []
        for bond in bonds where bond.order == .single {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard adjacency[a].count > 1, adjacency[b].count > 1 else { continue }
            guard !hasAlternatePath(from: a, to: b, excluding: (a, b)) else { continue }

            if isAmideBond(a, b) { continue }

            let forward = bfsSide(start: b, excluding: a)
            let backward = bfsSide(start: a, excluding: b)

            let edge: (atom1: Int, atom2: Int, movingAtoms: [Int])
            if forward.count <= backward.count {
                edge = (a, b, forward.sorted())
            } else {
                edge = (b, a, backward.sorted())
            }

            guard !edge.movingAtoms.isEmpty else { continue }
            torsions.append(edge)
        }

        torsions.sort {
            let lhsKey = (bfsOrder[$0.atom1], bfsOrder[$0.atom2], $0.movingAtoms.count)
            let rhsKey = (bfsOrder[$1.atom1], bfsOrder[$1.atom2], $1.movingAtoms.count)
            return lhsKey < rhsKey
        }
        return torsions
    }
}

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
    var electrostaticGridPipeline: MTLComputePipelineState?
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
    var drusinaDecompositionPipeline: MTLComputePipelineState?

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
    var electrostaticGridBuffer: MTLBuffer?
    var proteinChalcogenBuffer: MTLBuffer?
    var torsionStrainBuffer: MTLBuffer?

    // Pharmacophore constraint buffers
    var pharmaConstraintBuffer: MTLBuffer?
    var pharmaParamsBuffer: MTLBuffer?

    // DruseAF ML scoring (native Metal neural network)
    // v3 pipelines (kept for druseAFEncode position transform kernel, shared with v4)
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

    // DruseAF v4 PGN (pairwise geometric network) — scoring + rescoring
    var afv4EncodePipeline: MTLComputePipelineState?
    var afv4MsgTransformPipeline: MTLComputePipelineState?
    var afv4MsgAggregatePipeline: MTLComputePipelineState?
    var afv4PairPrepPipeline: MTLComputePipelineState?
    var afv4ScorePipeline: MTLComputePipelineState?
    var afv4RescorePipeline: MTLComputePipelineState?
    var afv4ProtHiddenBuffer: MTLBuffer?
    var afv4LigHiddenBuffer: MTLBuffer?
    var afv4MsgTempBuffer: MTLBuffer?
    var afv4ProtPairProjBuffer: MTLBuffer?
    var afv4LigPairProjBuffer: MTLBuffer?
    var afv4ParamsBuffer: MTLBuffer?
    var afv4EncodeCompatParamsBuffer: MTLBuffer?  // DruseAFParams shim for druseAFEncode kernel
    /// True when v4 weights are loaded and all v4 pipelines are available.
    var useAFv4: Bool { druseAFWeights?.version == 2 && afv4ScorePipeline != nil }

    // PIGNet2 physics-informed GNN scoring (native Metal)
    var pignet2SetupPipeline: MTLComputePipelineState?
    var pignet2EncodePipeline: MTLComputePipelineState?
    var pignet2ScorePipeline: MTLComputePipelineState?
    var pignet2Weights: PIGNet2WeightLoader.LoadedWeights?
    var pignet2ParamsBuffer: MTLBuffer?
    var pignet2ProtFeatBuffer: MTLBuffer?
    var pignet2LigFeatBuffer: MTLBuffer?
    var pignet2ProtPosBuffer: MTLBuffer?
    var pignet2ProtAuxBuffer: MTLBuffer?
    var pignet2LigAuxBuffer: MTLBuffer?
    var pignet2ProtEdgeBuffer: MTLBuffer?
    var pignet2LigEdgeBuffer: MTLBuffer?
    var pignet2SetupBuffer: MTLBuffer?
    var pignet2ScratchBuffer: MTLBuffer?
    var pignet2IntermediateBuffer: MTLBuffer?
    var pignet2IntermediateCapacity: Int = 0

    var isRunning = false
    var currentGeneration = 0
    var bestEnergy: Float = .infinity

    /// Grid dimensions from last computeGridMaps call (for flex grid proxy)
    private(set) var lastGridTotalPoints: Int = 0
    private(set) var lastGridNumAffinityTypes: Int = 0

    /// Cached Vina types + spacing from last grid computation — skip recompute if unchanged.
    private var lastGridVinaTypes: [Int32] = []
    private var lastGridSpacing: Float = 0
    private var lastGridPocketCenter: SIMD3<Float> = .zero

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
        guard let queue = device.makeCommandQueue() else {
            print("[DockingEngine] Failed to create command queue")
            return nil
        }
        guard let library = device.makeDefaultLibrary() else {
            print("[DockingEngine] Failed to create default Metal library")
            return nil
        }
        self.commandQueue = queue

        do {
            stericGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeStericGrid")!)
            hydrophobicGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHydrophobicGrid")!)
            hbondGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHBondGrid")!)
            vinaAffinityGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeVinaAffinityMaps")!)
            if let elecFunc = library.makeFunction(name: "computeElectrostaticGrid") {
                electrostaticGridPipeline = try device.makeComputePipelineState(function: elecFunc)
            }
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
            if let decompFunc = library.makeFunction(name: "scorePosesDecomposition") {
                drusinaDecompositionPipeline = try device.makeComputePipelineState(function: decompFunc)
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
            // DruseAF Metal neural network scoring (optional — don't fail init if pipeline creation throws)
            do {
                if let setupFunc = library.makeFunction(name: "druseAFSetup"),
                   let encodeFunc = library.makeFunction(name: "druseAFEncode"),
                   let scoreFunc = library.makeFunction(name: "druseAFScore") {
                    druseAFSetupPipeline = try device.makeComputePipelineState(function: setupFunc)
                    druseAFEncodePipeline = try device.makeComputePipelineState(function: encodeFunc)
                    druseAFScorePipeline = try device.makeComputePipelineState(function: scoreFunc)
                }
                druseAFWeights = DruseAFWeightLoader.loadFromBundle(device: device)
                // DruseAF v4 PGN pipelines
                if let v4Enc = library.makeFunction(name: "druseAFv4Encode"),
                   let v4MsgT = library.makeFunction(name: "druseAFv4MsgTransform"),
                   let v4MsgA = library.makeFunction(name: "druseAFv4MsgAggregate"),
                   let v4Prep = library.makeFunction(name: "druseAFv4PairPrep"),
                   let v4Score = library.makeFunction(name: "druseAFv4Score"),
                   let v4Rescore = library.makeFunction(name: "druseAFv4Rescore") {
                    afv4EncodePipeline = try device.makeComputePipelineState(function: v4Enc)
                    afv4MsgTransformPipeline = try device.makeComputePipelineState(function: v4MsgT)
                    afv4MsgAggregatePipeline = try device.makeComputePipelineState(function: v4MsgA)
                    afv4PairPrepPipeline = try device.makeComputePipelineState(function: v4Prep)
                    afv4ScorePipeline = try device.makeComputePipelineState(function: v4Score)
                    afv4RescorePipeline = try device.makeComputePipelineState(function: v4Rescore)
                }
            } catch {
                print("[DockingEngine] DruseAF pipeline creation failed (non-fatal): \(error)")
            }
            // PIGNet2 physics-informed GNN scoring (optional)
            do {
                if let pig2Setup = library.makeFunction(name: "pignet2Setup"),
                   let pig2Encode = library.makeFunction(name: "pignet2Encode"),
                   let pig2Score = library.makeFunction(name: "pignet2Score") {
                    pignet2SetupPipeline = try device.makeComputePipelineState(function: pig2Setup)
                    pignet2EncodePipeline = try device.makeComputePipelineState(function: pig2Encode)
                    pignet2ScorePipeline = try device.makeComputePipelineState(function: pig2Score)
                    pignet2Weights = PIGNet2WeightLoader.loadFromBundle(device: device)
                }
            } catch {
                print("[DockingEngine] PIGNet2 pipeline creation failed (non-fatal): \(error)")
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

        let searchPadding: Float = 0.0  // pocket.size already includes 4A padding from BindingSite
        let gridPadding: Float = 3.0
        let ligandMargin = ligandExtent ?? SIMD3<Float>(repeating: 4.0)
        let searchCenter = pocket.center
        // Ensure search box half-extent is at least ligandMargin + padding (so the ligand fits)
        let minHalfExtent = ligandMargin + SIMD3<Float>(repeating: 2.0)
        let rawSearchHalfExtent = pocket.size + SIMD3<Float>(repeating: searchPadding)
        let searchHalfExtent = max(rawSearchHalfExtent, minHalfExtent)
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

        // Electrostatic potential grid (for Drusina Coulomb term)
        let elecGridSize = Int(totalPoints) * MemoryLayout<UInt16>.stride
        electrostaticGridBuffer = device.makeBuffer(length: elecGridSize, options: .storageModeShared)
        if let elecBuf = electrostaticGridBuffer, let elecPipe = electrostaticGridPipeline {
            dispatchGridKernel(elecPipe, elecBuf, threadGroups: tgCount, label: "electrostatic")
        }

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
                _pad1: 0, flags: 0
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
        gpuLigAtoms: inout [DockLigandAtom],
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

        // --- Halogen bond info (with element-specific types) ---
        var halogens: [HalogenBondInfo] = []
        for (i, gpuAtom) in gpuLigAtoms.enumerated() {
            let vt = gpuAtom.vinaType
            let elementType: Int32
            switch vt {
            case Int32(VINA_F_H.rawValue):  elementType = 0
            case Int32(VINA_Cl_H.rawValue): elementType = 1
            case Int32(VINA_Br_H.rawValue): elementType = 2
            case Int32(VINA_I_H.rawValue):  elementType = 3
            default: continue
            }
            for bond in ligandBonds {
                let partner: Int?
                if bond.atomIndex1 == i { partner = bond.atomIndex2 }
                else if bond.atomIndex2 == i { partner = bond.atomIndex1 }
                else { partner = nil }
                if let p = partner, p < ligandAtoms.count, ligandAtoms[p].element == .C {
                    halogens.append(HalogenBondInfo(halogenAtomIndex: Int32(i), carbonAtomIndex: Int32(p),
                                                     elementType: elementType, _pad: 0))
                    break
                }
            }
        }
        if halogens.isEmpty {
            halogens.append(HalogenBondInfo(halogenAtomIndex: -1, carbonAtomIndex: -1, elementType: -1, _pad: 0))
        }
        halogenInfoBuffer = device.makeBuffer(
            bytes: &halogens,
            length: halogens.count * MemoryLayout<HalogenBondInfo>.stride,
            options: .storageModeShared)

        // --- Chalcogen bond info (dual σ-holes for thioethers) ---
        var chalcogens: [ChalcogenBondInfo] = []
        for (i, gpuAtom) in gpuLigAtoms.enumerated() {
            guard gpuAtom.vinaType == Int32(VINA_S_P.rawValue) else { continue }
            // Collect ALL bonded C atoms (up to 2 for thioethers R-S-R')
            for bond in ligandBonds {
                let partner: Int?
                if bond.atomIndex1 == i { partner = bond.atomIndex2 }
                else if bond.atomIndex2 == i { partner = bond.atomIndex1 }
                else { partner = nil }
                if let p = partner, p < ligandAtoms.count, ligandAtoms[p].element == .C {
                    chalcogens.append(ChalcogenBondInfo(sulfurAtomIndex: Int32(i), carbonAtomIndex: Int32(p)))
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

        // --- Protein chalcogen detection (Met SD, Cys SG → bonded C for σ-hole direction) ---
        var protChalcogens: [ProteinChalcogenGPU] = []
        for atom in proteinAtoms {
            let name = atom.name.trimmingCharacters(in: .whitespaces)
            let resName = atom.residueName.trimmingCharacters(in: .whitespaces)
            guard (resName == "MET" && name == "SD") || (resName == "CYS" && name == "SG") else { continue }
            let sameRes = proteinAtoms.filter { $0.chainID == atom.chainID && $0.residueSeq == atom.residueSeq }
            let bondedCNames: [String] = (resName == "MET") ? ["CG", "CE"] : ["CB"]
            for cName in bondedCNames {
                if let cAtom = sameRes.first(where: { $0.name.trimmingCharacters(in: .whitespaces) == cName }) {
                    let dir = simd_normalize(cAtom.position - atom.position)
                    protChalcogens.append(ProteinChalcogenGPU(
                        position: atom.position, _pad0: 0,
                        bondedCDir: dir, _pad1: 0))
                }
            }
        }
        if protChalcogens.isEmpty {
            protChalcogens.append(ProteinChalcogenGPU(position: .zero, _pad0: 0, bondedCDir: .zero, _pad1: 0))
        }
        proteinChalcogenBuffer = device.makeBuffer(
            bytes: &protChalcogens,
            length: protChalcogens.count * MemoryLayout<ProteinChalcogenGPU>.stride,
            options: .storageModeShared)

        // --- Ligand aromatic flags (for CH-π exclusion) ---
        let aromaticIndices = Set(ligRingGPU.flatMap { ring -> [Int] in
            (0..<Int(ring.numAtoms)).compactMap { i in
                let idx: Int32
                switch i {
                case 0: idx = ring.atomIndices.0
                case 1: idx = ring.atomIndices.1
                case 2: idx = ring.atomIndices.2
                case 3: idx = ring.atomIndices.3
                case 4: idx = ring.atomIndices.4
                case 5: idx = ring.atomIndices.5
                default: idx = -1
                }
                return idx >= 0 ? Int(idx) : nil
            }
        })
        for i in aromaticIndices where i < gpuLigAtoms.count {
            gpuLigAtoms[i].flags = LIGATOM_FLAG_AROMATIC
        }

        // --- Torsion strain detection (amide planarity) ---
        var torsionStrains: [TorsionStrainInfo] = []
        // Build adjacency for quick neighbor lookup
        var adjacency: [Int: [Int]] = [:]
        for bond in ligandBonds {
            adjacency[bond.atomIndex1, default: []].append(bond.atomIndex2)
            adjacency[bond.atomIndex2, default: []].append(bond.atomIndex1)
        }
        // Build bond order map for double bond detection
        var bondOrderMap: [String: Int] = [:]
        for bond in ligandBonds {
            let key = "\(min(bond.atomIndex1, bond.atomIndex2))_\(max(bond.atomIndex1, bond.atomIndex2))"
            bondOrderMap[key] = bond.order.rawValue
        }
        func bondOrder(_ a: Int, _ b: Int) -> Int {
            let key = "\(min(a, b))_\(max(a, b))"
            return bondOrderMap[key] ?? 1
        }
        // Check each rotatable bond for amide pattern (N-C=O)
        // A torsion edge defines a rotatable bond between atom1 and atom2
        // We need torsion edges from the GA data — but prepareDrusinaBuffers doesn't receive them.
        // Instead, detect amide bonds directly from the ligand topology.
        for bond in ligandBonds {
            let a1 = bond.atomIndex1
            let a2 = bond.atomIndex2
            guard a1 < ligandAtoms.count, a2 < ligandAtoms.count else { continue }
            guard bond.order == .single else { continue } // only single bonds can rotate
            let e1 = ligandAtoms[a1].element
            let e2 = ligandAtoms[a2].element
            // Check N-C pattern where C has a double-bonded O neighbor (amide)
            var isAmide = false
            var nIdx = -1, cIdx = -1
            if e1 == .N && e2 == .C {
                nIdx = a1; cIdx = a2
            } else if e1 == .C && e2 == .N {
                cIdx = a1; nIdx = a2
            }
            if nIdx >= 0 && cIdx >= 0 {
                // Check if C has a double-bonded O
                for neighbor in (adjacency[cIdx] ?? []) {
                    if neighbor < ligandAtoms.count && ligandAtoms[neighbor].element == .O
                        && bondOrder(cIdx, neighbor) == 2 {
                        isAmide = true
                        break
                    }
                }
            }
            guard isAmide else { continue }
            // Find dihedral quad: neighbor of N (not C), N, C, neighbor of C (not N, prefer O=)
            let a0 = (adjacency[nIdx] ?? []).first(where: { $0 != cIdx && $0 < ligandAtoms.count })
            let a3 = (adjacency[cIdx] ?? []).first(where: { $0 != nIdx && $0 < ligandAtoms.count })
            guard let atom0 = a0, let atom3 = a3 else { continue }
            torsionStrains.append(TorsionStrainInfo(
                atom0: Int32(atom0), atom1: Int32(nIdx), atom2: Int32(cIdx), atom3: Int32(atom3),
                strainType: 1, forceConstant: 5.0, _pad0: 0, _pad1: 0))
        }
        if torsionStrains.isEmpty {
            torsionStrains.append(TorsionStrainInfo(
                atom0: -1, atom1: -1, atom2: -1, atom3: -1,
                strainType: 0, forceConstant: 0, _pad0: 0, _pad1: 0))
        }
        torsionStrainBuffer = device.makeBuffer(
            bytes: &torsionStrains,
            length: torsionStrains.count * MemoryLayout<TorsionStrainInfo>.stride,
            options: .storageModeShared)

        // --- Drusina parameters ---
        var params = DrusinaParams(
            numProteinRings: UInt32(protRings.count),
            numLigandRings: UInt32(ligRings.count),
            numProteinCations: UInt32(max(cations.count - (cations.first == .zero ? 1 : 0), 0)),
            numHalogens: UInt32(halogens.first?.halogenAtomIndex == -1 ? 0 : halogens.count),
            wPiPi: -0.35,
            wPiCation: -0.40,
            wHalogenBond: -0.40,
            wMetalCoord: -0.95,
            numProteinAmides: UInt32(amides.first?.centroid == .zero && amides.count == 1 ? 0 : amides.count),
            numChalcogens: UInt32(chalcogens.first?.sulfurAtomIndex == -1 ? 0 : chalcogens.count),
            wSaltBridge: -0.35,
            wAmideStack: -0.30,
            wChalcogenBond: -0.20,
            numSaltBridgeGroups: UInt32(sbGroups.first?.chargeSign == 0 ? 0 : sbGroups.count),
            wCoulomb: 0.012,
            wCHPi: -0.08,
            wCooperativity: 0.0,
            wTorsionStrain: 1.5,
            numProteinChalcogens: UInt32(protChalcogens.first?.bondedCDir == .zero ? 0 : protChalcogens.count),
            numTorsionStrains: UInt32(torsionStrains.first?.atom0 == -1 ? 0 : torsionStrains.count),
            _padDP0: 0,
            _padDP1: 0)
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
        ligandBonds: [Bond],
        gpuLigAtoms: [DockLigandAtom],
        pocket: BindingPocket,
        popSize: Int
    ) {
        guard let afWeights = druseAFWeights else { return }
        guard useAFv4 || druseAFSetupPipeline != nil else { return }

        let pocketAtoms = proteinAtoms.filter { atom in
            simd_distance(atom.position, pocket.center) <= 10.0
        }
        let P = min(pocketAtoms.count, 256)
        let L = min(ligandAtoms.count, 64)

        let protHybrid = DruseScoreFeatureExtractor.buildHybridizationMap(
            atoms: pocketAtoms, bonds: proteinStructure?.bonds ?? [])
        let ligHybrid = DruseScoreFeatureExtractor.buildHybridizationMap(
            atoms: ligandAtoms, bonds: ligandBonds)

        var protFeats = [Float](repeating: 0, count: 256 * 20)
        for i in 0..<P {
            let f = DruseScoreFeatureExtractor.atomFeatures(
                pocketAtoms[i], isProtein: true,
                chemInfo: protHybrid[pocketAtoms[i].id])
            for j in 0..<20 { protFeats[i * 20 + j] = f[j] }
        }
        druseAFProtFeatBuffer = device.makeBuffer(
            bytes: &protFeats, length: protFeats.count * 4, options: .storageModeShared)

        var ligFeats = [Float](repeating: 0, count: 64 * 20)
        for i in 0..<L {
            let f = DruseScoreFeatureExtractor.atomFeatures(
                ligandAtoms[i], isProtein: false,
                chemInfo: ligHybrid[ligandAtoms[i].id])
            for j in 0..<20 { ligFeats[i * 20 + j] = f[j] }
        }
        druseAFLigFeatBuffer = device.makeBuffer(
            bytes: &ligFeats, length: ligFeats.count * 4, options: .storageModeShared)

        if useAFv4 {
            // v4: packed float3 positions (3 floats per atom, no SIMD padding)
            var protPosFlat = [Float](repeating: 0, count: 256 * 3)
            for i in 0..<P {
                let p = pocketAtoms[i].position
                protPosFlat[i * 3 + 0] = p.x; protPosFlat[i * 3 + 1] = p.y; protPosFlat[i * 3 + 2] = p.z
            }
            druseAFProtPosBuffer = device.makeBuffer(
                bytes: &protPosFlat, length: protPosFlat.count * 4, options: .storageModeShared)
        } else {
            // v3: SIMD3<Float> (16-byte stride with padding)
            var protPos = [SIMD3<Float>](repeating: .zero, count: 256)
            for i in 0..<P { protPos[i] = pocketAtoms[i].position }
            druseAFProtPosBuffer = device.makeBuffer(
                bytes: &protPos, length: protPos.count * MemoryLayout<SIMD3<Float>>.stride,
                options: .storageModeShared)
        }

        // Intermediate buffer for transformed ligand positions (shared v3/v4)
        let posFloatsPerPose = 64 * 3
        let requiredBytes = popSize * posFloatsPerPose * 4
        if requiredBytes > druseAFIntermediateCapacity {
            druseAFIntermediateBuffer = device.makeBuffer(
                length: requiredBytes, options: .storageModeShared)
            druseAFIntermediateCapacity = requiredBytes
        }

        if useAFv4 {
            // === DruseAF v4 PGN setup ===
            var v4Params = DruseAFv4Params(
                numProteinAtoms: UInt32(P),
                numLigandAtoms: UInt32(L),
                numWeightTensors: UInt32(afWeights.numTensors),
                numPoses: UInt32(popSize)
            )
            afv4ParamsBuffer = device.makeBuffer(
                bytes: &v4Params, length: MemoryLayout<DruseAFv4Params>.stride,
                options: .storageModeShared)

            // Compat DruseAFParams for the shared druseAFEncode kernel (position transform)
            var compatParams = DruseAFParams(
                numProteinAtoms: UInt32(P), numLigandAtoms: UInt32(L),
                hiddenDim: 128, numHeads: 4, headDim: 32, rbfBins: 24,
                rbfGamma: 2.0, rbfSpacing: 8.0 / 23.0, numCrossAttnLayers: 3,
                numWeightTensors: UInt32(afWeights.numTensors), _pad0: 0, _pad1: 0)
            afv4EncodeCompatParamsBuffer = device.makeBuffer(
                bytes: &compatParams, length: MemoryLayout<DruseAFParams>.stride,
                options: .storageModeShared)

            // v4 buffers: hidden states, message passing temp, pair projections
            let H = 128
            let PD = 64
            afv4ProtHiddenBuffer = device.makeBuffer(length: 256 * H * 4, options: .storageModeShared)
            afv4LigHiddenBuffer = device.makeBuffer(length: 64 * H * 4, options: .storageModeShared)
            afv4MsgTempBuffer = device.makeBuffer(length: 256 * H * 4, options: .storageModeShared)
            afv4ProtPairProjBuffer = device.makeBuffer(length: 256 * PD * 4, options: .storageModeShared)
            afv4LigPairProjBuffer = device.makeBuffer(length: 64 * PD * 4, options: .storageModeShared)

            guard let v4ParamsBuf = afv4ParamsBuffer,
                  let protFeat = druseAFProtFeatBuffer,
                  let ligFeat = druseAFLigFeatBuffer,
                  let protHidden = afv4ProtHiddenBuffer,
                  let ligHidden = afv4LigHiddenBuffer,
                  let msgTemp = afv4MsgTempBuffer,
                  let protPP = afv4ProtPairProjBuffer,
                  let ligPP = afv4LigPairProjBuffer,
                  let protPos = druseAFProtPosBuffer,
                  let v4Enc = afv4EncodePipeline,
                  let v4MsgT = afv4MsgTransformPipeline,
                  let v4MsgA = afv4MsgAggregatePipeline,
                  let v4Prep = afv4PairPrepPipeline else { return }

            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return }

            let totalAtoms = P + L
            let tgSize1 = MTLSize(width: min(totalAtoms, 256), height: 1, depth: 1)
            let tgCount1 = MTLSize(width: (totalAtoms + 255) / 256, height: 1, depth: 1)
            let tgSizeP = MTLSize(width: min(P, 256), height: 1, depth: 1)
            let tgCountP = MTLSize(width: (P + 255) / 256, height: 1, depth: 1)

            // Step 1: Encode protein + ligand atoms
            enc.setComputePipelineState(v4Enc)
            enc.setBuffer(protFeat, offset: 0, index: 0)
            enc.setBuffer(ligFeat, offset: 0, index: 1)
            enc.setBuffer(protHidden, offset: 0, index: 2)
            enc.setBuffer(ligHidden, offset: 0, index: 3)
            enc.setBuffer(afWeights.weightBuffer, offset: 0, index: 4)
            enc.setBuffer(afWeights.entryBuffer, offset: 0, index: 5)
            enc.setBuffer(v4ParamsBuf, offset: 0, index: 6)
            enc.dispatchThreadgroups(tgCount1, threadsPerThreadgroup: tgSize1)

            // Step 2: 3 rounds of message passing
            for layer in 0 ..< 3 {
                var layerIdx = UInt32(layer)
                let layerBuf = device.makeBuffer(bytes: &layerIdx, length: 4, options: .storageModeShared)!

                enc.setComputePipelineState(v4MsgT)
                enc.setBuffer(protHidden, offset: 0, index: 0)
                enc.setBuffer(msgTemp, offset: 0, index: 1)
                enc.setBuffer(afWeights.weightBuffer, offset: 0, index: 2)
                enc.setBuffer(afWeights.entryBuffer, offset: 0, index: 3)
                enc.setBuffer(v4ParamsBuf, offset: 0, index: 4)
                enc.setBuffer(layerBuf, offset: 0, index: 5)
                enc.dispatchThreadgroups(tgCountP, threadsPerThreadgroup: tgSizeP)

                enc.setComputePipelineState(v4MsgA)
                enc.setBuffer(protHidden, offset: 0, index: 0)
                enc.setBuffer(msgTemp, offset: 0, index: 1)
                enc.setBuffer(protPos, offset: 0, index: 2)
                enc.setBuffer(afWeights.weightBuffer, offset: 0, index: 3)
                enc.setBuffer(afWeights.entryBuffer, offset: 0, index: 4)
                enc.setBuffer(v4ParamsBuf, offset: 0, index: 5)
                enc.setBuffer(layerBuf, offset: 0, index: 6)
                enc.dispatchThreadgroups(tgCountP, threadsPerThreadgroup: tgSizeP)
            }

            // Step 3: Pair projections
            enc.setComputePipelineState(v4Prep)
            enc.setBuffer(protHidden, offset: 0, index: 0)
            enc.setBuffer(ligHidden, offset: 0, index: 1)
            enc.setBuffer(protPP, offset: 0, index: 2)
            enc.setBuffer(ligPP, offset: 0, index: 3)
            enc.setBuffer(afWeights.weightBuffer, offset: 0, index: 4)
            enc.setBuffer(afWeights.entryBuffer, offset: 0, index: 5)
            enc.setBuffer(v4ParamsBuf, offset: 0, index: 6)
            enc.dispatchThreadgroups(tgCount1, threadsPerThreadgroup: tgSize1)

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            ActivityLog.shared.info(
                "[DruseAF v4] Setup complete: \(P) prot, \(L) lig atoms, 3 msg-pass layers",
                category: .dock)
        } else {
            // === DruseAF v3 cross-attention setup (legacy) ===
            guard let setupPipeV3 = druseAFSetupPipeline else { return }

            var params = DruseAFParams(
                numProteinAtoms: UInt32(P), numLigandAtoms: UInt32(L),
                hiddenDim: 256, numHeads: 4, headDim: 64, rbfBins: 50,
                rbfGamma: 10.0, rbfSpacing: 10.0 / 49.0, numCrossAttnLayers: 3,
                numWeightTensors: UInt32(afWeights.numTensors), _pad0: 0, _pad1: 0)
            druseAFParamsBuffer = device.makeBuffer(
                bytes: &params, length: MemoryLayout<DruseAFParams>.stride,
                options: .storageModeShared)

            let setupFloats = 7 * 256 * 256 + 64 * 256
            druseAFSetupBuffer = device.makeBuffer(
                length: setupFloats * 4, options: .storageModeShared)

            guard let afParamsBuf = druseAFParamsBuffer,
                  let afProtFeat = druseAFProtFeatBuffer,
                  let afLigFeat = druseAFLigFeatBuffer,
                  let afSetupBuf = druseAFSetupBuffer else { return }

            let totalAtoms = P + L
            let setupTgSize = MTLSize(width: min(totalAtoms, 256), height: 1, depth: 1)
            let setupTgCount = MTLSize(width: (totalAtoms + 255) / 256, height: 1, depth: 1)

            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(setupPipeV3)
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
                "[DruseAF v3] Setup complete: \(P) protein atoms, \(L) ligand atoms",
                category: .dock)
        }
    }

    // MARK: - PIGNet2 Physics-Informed GNN Buffers

    /// PIGNet2 atom feature vector (47-dim) matching PIGNet2 data.py atom_to_features().
    /// Symbol one-hot (9+1=10) + Degree one-hot (6) + Hybridization one-hot (7) +
    /// Period one-hot (6) + Group one-hot (18) + Aromaticity (1) = 48? Actually 47.
    /// Period (6 values, 0-5) + Group (18 values, 0-17) = 24, but chem.py uses 6+18=24.
    /// Total: 10 + 6 + 7 + 24 + 1 = 48... PIGNet2 source says 47. Period has 6 slots (0-5),
    /// but the one-hot uses exactly `range(6)` → 6. Group uses `range(18)` → 18.
    /// 10 + 6 + 7 + 6 + 18 + 1 = 48. But the actual model has in_features=47 and
    /// ATOM_SYMBOLS has 9 items (with "X" as last), making 9 one-hot + degree(6) + hyb(7) +
    /// period(6) + group(18) + arom(1) = 47. The "X" is the unknown catch-all.
    private static let pigSymbols: [Element] = [.C, .N, .O, .S, .F, .P, .Cl, .Br]
    // 9th slot is "X" (unknown), used when element not in list

    private static let pigVDWRadii: [Int: Float] = [
        6: 1.90, 7: 1.8, 8: 1.7, 9: 1.5, 12: 1.2, 15: 2.1, 16: 2.0,
        17: 1.8, 20: 1.2, 25: 1.2, 26: 1.2, 27: 1.2, 28: 1.2, 29: 1.2, 30: 1.2,
        35: 2.0, 53: 2.2,
    ]

    // Metal atomic numbers (PIGNet2 chem.py)
    private static let metalAtomicNumbers: Set<Int> = {
        var s = Set<Int>()
        for n in 3...4 { s.insert(n) }
        for n in 11...13 { s.insert(n) }
        for n in 19...31 { s.insert(n) }
        for n in 37...50 { s.insert(n) }
        for n in 55...83 { s.insert(n) }
        for n in 87...116 { s.insert(n) }
        return s
    }()

    private static let hydrophobicElements: Set<Element> = [.C, .S, .F, .Cl, .Br]

    // Period/Group lookup from atomic number (matches PIGNet2 chem.py PERIODIC_TABLE)
    // Period is 0-indexed row (0-5), Group is 0-indexed column (0-17)
    private static let pigPeriodGroup: [Int: (Int, Int)] = {
        // Row 0 (H-He)
        var map = [Int: (Int, Int)]()
        map[1] = (0, 0);  map[2] = (0, 17)
        // Row 1 (Li-Ne)
        map[3] = (1, 0);  map[4] = (1, 1)
        map[5] = (1, 12); map[6] = (1, 13); map[7] = (1, 14); map[8] = (1, 15); map[9] = (1, 16); map[10] = (1, 17)
        // Row 2 (Na-Ar)
        map[11] = (2, 0);  map[12] = (2, 1)
        map[13] = (2, 12); map[14] = (2, 13); map[15] = (2, 14); map[16] = (2, 15); map[17] = (2, 16); map[18] = (2, 17)
        // Row 3 (K-Kr)
        map[19] = (3, 0);  map[20] = (3, 1)
        for (i, z) in (21...30).enumerated() { map[z] = (3, 2 + i) }
        map[31] = (3, 12); map[32] = (3, 13); map[33] = (3, 14); map[34] = (3, 15); map[35] = (3, 16); map[36] = (3, 17)
        // Row 4 (Rb-Xe): extend for completeness
        map[37] = (4, 0);  map[38] = (4, 1)
        for (i, z) in (39...48).enumerated() { map[z] = (4, 2 + i) }
        map[49] = (4, 12); map[50] = (4, 13); map[51] = (4, 14); map[52] = (4, 15); map[53] = (4, 16); map[54] = (4, 17)
        // Row 5 (Cs-Rn): partial
        map[55] = (5, 0);  map[56] = (5, 1)
        // Lanthanides → (5, 2)
        for z in 57...71 { map[z] = (5, 2) }
        for (i, z) in (72...80).enumerated() { map[z] = (5, 3 + i) }
        map[81] = (5, 12); map[82] = (5, 13); map[83] = (5, 14)
        return map
    }()

    func pignet2AtomFeatures(_ atom: Atom, degree: Int, isAromatic: Bool,
                              hybridization: Int) -> [Float] {
        var feat = [Float](repeating: 0, count: 47)

        // Symbol one-hot (9 slots): C=0, N=1, O=2, S=3, F=4, P=5, Cl=6, Br=7, X=8
        if let idx = Self.pigSymbols.firstIndex(of: atom.element) {
            feat[idx] = 1.0
        } else {
            feat[8] = 1.0 // unknown "X"
        }

        // Degree one-hot (6 slots at offset 9): degree 0-5
        let degIdx = min(degree, 5)
        feat[9 + degIdx] = 1.0

        // Hybridization one-hot (7 slots at offset 15):
        // 0=S, 1=SP, 2=SP2, 3=SP3, 4=SP3D, 5=SP3D2, 6=None
        let hybIdx = min(hybridization, 6)
        feat[15 + hybIdx] = 1.0

        // Period one-hot (6 slots at offset 22) + Group one-hot (18 slots at offset 28)
        let atomicNum = atom.element.rawValue
        let (period, group) = Self.pigPeriodGroup[atomicNum] ?? (0, 0)
        feat[22 + min(period, 5)] = 1.0
        feat[28 + min(group, 17)] = 1.0

        // Aromaticity (1 slot at offset 46)
        feat[46] = isAromatic ? 1.0 : 0.0

        return feat
    }

    func pignet2AtomAux(_ atom: Atom, degree: Int, neighbors: [Atom]) -> PIGNet2AtomAux {
        let atomicNum = atom.element.rawValue
        let vdwRadius = Self.pigVDWRadii[atomicNum] ?? atom.element.vdwRadius

        var flags: UInt32 = 0
        // is_metal
        if Self.metalAtomicNumbers.contains(atomicNum) { flags |= 0x1 }
        // is_h_donor: simplified — N, O, S atoms with attached H (or capable)
        if atom.element == .N || atom.element == .O || atom.element == .S {
            flags |= 0x2
        }
        // is_h_acceptor: N, O, S, F
        if atom.element == .N || atom.element == .O || atom.element == .S || atom.element == .F {
            flags |= 0x4
        }
        // is_hydrophobic: atom in {C,S,F,Cl,Br,I} with all neighbors also in that set
        if Self.hydrophobicElements.contains(atom.element) {
            let allNeighborsHydrophobic = neighbors.allSatisfy { Self.hydrophobicElements.contains($0.element) }
            if allNeighborsHydrophobic { flags |= 0x8 }
        }

        return PIGNet2AtomAux(
            vdwRadius: vdwRadius,
            flags: flags,
            formalCharge: Float(atom.formalCharge),
            _pad: 0
        )
    }

    func preparePIGNet2Buffers(
        ligandAtoms: [Atom],
        ligandBonds: [Bond],
        gpuLigAtoms: [DockLigandAtom],
        pocket: BindingPocket,
        popSize: Int
    ) {
        guard let pig2Weights = pignet2Weights,
              let setupPipe = pignet2SetupPipeline else { return }

        // Extract pocket protein atoms
        let pocketAtoms = proteinAtoms.filter { atom in
            simd_distance(atom.position, pocket.center) <= 10.0
        }
        let P = min(pocketAtoms.count, Int(PIG_MAX_PROT))
        let L = min(ligandAtoms.count, Int(PIG_MAX_LIG))

        // Build neighbor maps for degree/hydrophobic computation
        let allProtBonds = proteinStructure?.bonds ?? []
        var protNeighborMap = [Int: [Atom]]()
        for bond in allProtBonds {
            let a1 = bond.atomIndex1, a2 = bond.atomIndex2
            if a1 < proteinAtoms.count && a2 < proteinAtoms.count {
                protNeighborMap[proteinAtoms[a1].id, default: []].append(proteinAtoms[a2])
                protNeighborMap[proteinAtoms[a2].id, default: []].append(proteinAtoms[a1])
            }
        }
        var ligNeighborMap = [Int: [Atom]]()
        for bond in ligandBonds {
            let a1 = bond.atomIndex1, a2 = bond.atomIndex2
            if a1 < ligandAtoms.count && a2 < ligandAtoms.count {
                ligNeighborMap[ligandAtoms[a1].id, default: []].append(ligandAtoms[a2])
                ligNeighborMap[ligandAtoms[a2].id, default: []].append(ligandAtoms[a1])
            }
        }

        // Build protein features (47-dim per atom)
        var protFeats = [Float](repeating: 0, count: Int(PIG_MAX_PROT) * 47)
        var protAuxArray = [PIGNet2AtomAux](repeating: PIGNet2AtomAux(vdwRadius: 0, flags: 0, formalCharge: 0, _pad: 0), count: Int(PIG_MAX_PROT))
        _ = Set(pocketAtoms.prefix(P).map { $0.id })
        let pocketIndexMap = Dictionary(uniqueKeysWithValues: pocketAtoms.prefix(P).enumerated().map { ($1.id, $0) })

        // Aromatic residue atoms set (matches PIGNet2 chem.py)
        let aromaticResAtoms: [String: Set<String>] = [
            "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        ]

        for i in 0..<P {
            let atom = pocketAtoms[i]
            let neighbors = protNeighborMap[atom.id] ?? []
            let degree = neighbors.count
            let isAromatic = aromaticResAtoms[atom.residueName]?.contains(atom.name) ?? false
            let hyb = isAromatic ? 2 : 3
            let f = pignet2AtomFeatures(atom, degree: degree, isAromatic: isAromatic, hybridization: hyb)
            for j in 0..<47 { protFeats[i * 47 + j] = f[j] }
            protAuxArray[i] = pignet2AtomAux(atom, degree: degree, neighbors: neighbors)
        }
        pignet2ProtFeatBuffer = device.makeBuffer(bytes: &protFeats, length: protFeats.count * 4, options: .storageModeShared)
        pignet2ProtAuxBuffer = device.makeBuffer(bytes: &protAuxArray, length: protAuxArray.count * MemoryLayout<PIGNet2AtomAux>.stride, options: .storageModeShared)

        // Build ligand features
        var ligFeats = [Float](repeating: 0, count: Int(PIG_MAX_LIG) * 47)
        var ligAuxArray = [PIGNet2AtomAux](repeating: PIGNet2AtomAux(vdwRadius: 0, flags: 0, formalCharge: 0, _pad: 0), count: Int(PIG_MAX_LIG))
        for i in 0..<L {
            let atom = ligandAtoms[i]
            let neighbors = ligNeighborMap[atom.id] ?? []
            let degree = neighbors.count
            // Ligand aromaticity: approximate from C with 3 bonds (sp2), or het-atom isAromatic
            // For a proper implementation, ring perception from RDKit should be used.
            // As a heuristic: C atoms with exactly 3 bonds that are all to C/N are likely aromatic.
            let isAromatic = (atom.element == .C && degree == 3 &&
                neighbors.allSatisfy { $0.element == .C || $0.element == .N })
            let hyb = isAromatic ? 2 : 3
            let f = pignet2AtomFeatures(atom, degree: degree, isAromatic: isAromatic, hybridization: hyb)
            for j in 0..<47 { ligFeats[i * 47 + j] = f[j] }
            ligAuxArray[i] = pignet2AtomAux(atom, degree: degree, neighbors: neighbors)
        }
        pignet2LigFeatBuffer = device.makeBuffer(bytes: &ligFeats, length: ligFeats.count * 4, options: .storageModeShared)
        pignet2LigAuxBuffer = device.makeBuffer(bytes: &ligAuxArray, length: ligAuxArray.count * MemoryLayout<PIGNet2AtomAux>.stride, options: .storageModeShared)

        // Protein positions (flat float3)
        var protPos = [Float](repeating: 0, count: Int(PIG_MAX_PROT) * 3)
        for i in 0..<P {
            protPos[i * 3 + 0] = pocketAtoms[i].position.x
            protPos[i * 3 + 1] = pocketAtoms[i].position.y
            protPos[i * 3 + 2] = pocketAtoms[i].position.z
        }
        pignet2ProtPosBuffer = device.makeBuffer(bytes: &protPos, length: protPos.count * 4, options: .storageModeShared)

        // Build protein intramolecular edges (bidirectional, from bond adjacency)
        var protEdges = [PIGNet2Edge]()
        for bond in allProtBonds {
            let a1 = bond.atomIndex1, a2 = bond.atomIndex2
            guard a1 < proteinAtoms.count, a2 < proteinAtoms.count else { continue }
            let id1 = proteinAtoms[a1].id, id2 = proteinAtoms[a2].id
            guard let idx1 = pocketIndexMap[id1], let idx2 = pocketIndexMap[id2] else { continue }
            protEdges.append(PIGNet2Edge(src: UInt16(idx1), dst: UInt16(idx2)))
            protEdges.append(PIGNet2Edge(src: UInt16(idx2), dst: UInt16(idx1)))
        }
        if protEdges.isEmpty {
            protEdges.append(PIGNet2Edge(src: 0, dst: 0)) // dummy
        }
        pignet2ProtEdgeBuffer = device.makeBuffer(bytes: &protEdges, length: protEdges.count * MemoryLayout<PIGNet2Edge>.stride, options: .storageModeShared)

        // Build ligand intramolecular edges (bidirectional)
        var ligEdges = [PIGNet2Edge]()
        for bond in ligandBonds {
            let a1 = bond.atomIndex1, a2 = bond.atomIndex2
            guard a1 < L, a2 < L else { continue }
            ligEdges.append(PIGNet2Edge(src: UInt16(a1), dst: UInt16(a2)))
            ligEdges.append(PIGNet2Edge(src: UInt16(a2), dst: UInt16(a1)))
        }
        if ligEdges.isEmpty {
            ligEdges.append(PIGNet2Edge(src: 0, dst: 0))
        }
        pignet2LigEdgeBuffer = device.makeBuffer(bytes: &ligEdges, length: ligEdges.count * MemoryLayout<PIGNet2Edge>.stride, options: .storageModeShared)

        // Compute rotatable bond count
        let numRotBonds = ligandBonds.filter { $0.isRotatable }.count

        // PIGNet2 params
        var pigParams = PIGNet2Params(
            numProteinAtoms: UInt32(P),
            numLigandAtoms: UInt32(L),
            numProtIntraEdges: UInt32(protEdges.count),
            numLigIntraEdges: UInt32(ligEdges.count),
            numRotatableBonds: UInt32(numRotBonds),
            numWeightTensors: UInt32(pig2Weights.numTensors),
            _pad0: 0, _pad1: 0
        )
        pignet2ParamsBuffer = device.makeBuffer(bytes: &pigParams, length: MemoryLayout<PIGNet2Params>.stride, options: .storageModeShared)

        // Setup buffer: 4 sections × P × 128 floats
        let setupFloats = 4 * P * 128  // 4 sections × P atoms × 128 hidden dim
        pignet2SetupBuffer = device.makeBuffer(length: max(setupFloats * 4, 16), options: .storageModeShared)

        // Scratch buffer for GatedGAT ping-pong: P × 128 floats
        let scratchFloats = P * Int(PIG_DIM)
        pignet2ScratchBuffer = device.makeBuffer(length: max(scratchFloats * 4, 16), options: .storageModeShared)

        // Per-pose intermediate buffer
        let posFloatsPerPose = Int(PIG_MAX_LIG) * 3
        let requiredBytes = popSize * posFloatsPerPose * 4
        if requiredBytes > pignet2IntermediateCapacity {
            pignet2IntermediateBuffer = device.makeBuffer(length: requiredBytes, options: .storageModeShared)
            pignet2IntermediateCapacity = requiredBytes
        }

        // Dispatch setup kernel
        guard let pigParamsBuf = pignet2ParamsBuffer,
              let pigProtFeat = pignet2ProtFeatBuffer,
              let pigSetupBuf = pignet2SetupBuffer,
              let pigScratch = pignet2ScratchBuffer,
              let pigProtEdgeBuf = pignet2ProtEdgeBuffer else { return }

        let tgSize = MTLSize(width: min(P, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (P + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)

        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(setupPipe)
        enc.setBuffer(pigProtFeat, offset: 0, index: 0)
        enc.setBuffer(pig2Weights.weightBuffer, offset: 0, index: 1)
        enc.setBuffer(pig2Weights.entryBuffer, offset: 0, index: 2)
        enc.setBuffer(pigParamsBuf, offset: 0, index: 3)
        enc.setBuffer(pigSetupBuf, offset: 0, index: 4)
        enc.setBuffer(pigProtEdgeBuf, offset: 0, index: 5)
        enc.setBuffer(pigScratch, offset: 0, index: 6)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        ActivityLog.shared.info(
            "[PIGNet2] Setup complete: \(P) protein atoms, \(L) ligand atoms, " +
            "\(protEdges.count) prot edges, \(ligEdges.count) lig edges, " +
            "\(numRotBonds) rotatable bonds",
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
        let hasFormalCharges = heavyAtoms.contains { $0.formalCharge != 0 }

        // If both partial and formal charges are already set, nothing to do
        guard !hasPartialCharges || !hasFormalCharges else { return ligand }

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
            if !hasPartialCharges {
                mergedAtoms[i].charge = charged.atoms[i].charge
            }
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

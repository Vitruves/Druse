import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Docking Configuration

struct DockingConfig: Sendable {
    // Population and search
    var populationSize: Int = 300
    var numRuns: Int = 1             // independent Monte Carlo trajectory batches
    var generationsPerRun: Int = 300 // Monte Carlo steps per run
    var gridSpacing: Float = 0.375

    // Search operators
    var mutationRate: Float = 0.08
    var crossoverRate: Float = 0.75
    var translationStep: Float = 2.0 // Angstroms, aligned with Vina mutation amplitude
    var rotationStep: Float = 0.3    // radians (~17°)
    var torsionStep: Float = 0.8     // radians (~46°) — large enough to escape tangled conformers
    var mcTemperature: Float = 1.2   // kcal/mol, matches Vina's default Metropolis temperature
    var explicitRerankTopClusters: Int = 12 // top basin representatives rescored against explicit receptor atoms
    var explicitRerankVariantsPerCluster: Int = 4 // seeded local refinement around each top basin representative
    var explicitRerankLocalSearchSteps: Int = 20 // short second-pass refinement on rerank seeds

    // Local search (Vina-like basin hopping: refine every MC step by default)
    var localSearchFrequency: Int = 1   // every N generations
    var localSearchSteps: Int = 30      // gradient descent steps per refinement
    var liveUpdateFrequency: Int = 10   // visual update every N generations (higher = fewer GPU syncs = faster)

    // Flexibility
    var enableFlexibility: Bool = true  // torsion flexibility during docking
    var flexRefinementSteps: Int = 50   // extra torsion refinement steps after GA

    // Gradient method
    var useAnalyticalGradients: Bool = true  // analytical (~28x fewer evals) vs numerical finite differences

    // Clash handling
    var maxClashOverlap: Float = 0.4    // Angstroms of VdW overlap allowed
    var clashPenaltyScale: Float = 5.0  // kcal/mol per Angstrom of excess overlap

    // Ligand strain penalty: MMFF94-based post-docking filter
    var strainPenaltyEnabled: Bool = true
    var strainPenaltyThreshold: Float = 6.0  // kcal/mol above free energy before penalizing
    var strainPenaltyWeight: Float = 0.5     // score += weight * max(0, strain - threshold)

    // Exploration: broader initial search with higher translation/rotation steps
    // before switching to fine-grained local refinement
    var explorationPhaseRatio: Float = 0.4  // first 40% of generations use broader search
    var explorationTranslationStep: Float = 4.0  // wider initial translation (vs 2.0 during refinement)
    var explorationRotationStep: Float = 0.6     // wider initial rotation (vs 0.3)
    var explorationMutationRate: Float = 0.15    // higher mutation during exploration

    // Auto mode: adapt parameters to protein/ligand/pocket complexity
    var autoMode: Bool = false

    // Legacy flat-generation count (for backward compatibility)
    var numGenerations: Int { numRuns * generationsPerRun }

    /// Compute adaptive docking parameters based on system complexity.
    /// Called once per protein (for pocket/protein features) and can be refined per ligand.
    static func autoTune(
        proteinAtomCount: Int,
        pocketVolume: Float,
        pocketBuriedness: Float,
        ligandHeavyAtoms: Int,
        ligandRotatableBonds: Int
    ) -> DockingConfig {
        var config = DockingConfig()
        config.autoMode = true

        // --- Population scales with ligand flexibility ---
        // More torsions = larger conformational space = need more population diversity
        let torsionFactor = max(1.0, Float(ligandRotatableBonds) / 5.0) // 1.0 for ≤5 torsions, up to ~6 for 30
        let basePop = 150
        config.populationSize = min(500, max(100, Int(Float(basePop) * sqrt(torsionFactor))))

        // --- Generations scale with pocket size and buriedness ---
        // Large/shallow pockets need more exploration; buried pockets converge faster
        let buriednessFactor = max(0.6, 1.5 - pocketBuriedness) // 0.6 for fully buried, 1.5 for exposed
        let volumeFactor = max(0.8, min(2.0, pocketVolume / 800.0)) // normalize around 800 A³
        let baseGen = 150
        config.generationsPerRun = min(400, max(80, Int(Float(baseGen) * buriednessFactor * volumeFactor)))

        // --- Runs scale with overall difficulty ---
        // Large proteins + flexible ligands benefit from independent restarts
        let proteinSizeFactor: Float = proteinAtomCount > 5000 ? 1.5 : (proteinAtomCount > 3000 ? 1.2 : 1.0)
        let difficultyScore = torsionFactor * buriednessFactor * proteinSizeFactor
        config.numRuns = min(8, max(1, Int(difficultyScore)))

        // --- Grid spacing: coarsen for very large proteins to stay in memory ---
        if proteinAtomCount > 10000 {
            config.gridSpacing = 0.5
        } else {
            config.gridSpacing = 0.375
        }

        // --- Exploration phase: extend for flexible ligands ---
        if ligandRotatableBonds > 10 {
            config.explorationPhaseRatio = 0.5 // more exploration for floppy molecules
            config.explorationTranslationStep = 5.0
            config.explorationMutationRate = 0.18
        }

        // --- Local search frequency: increase for rigid ligands (cheap, high payoff) ---
        if ligandRotatableBonds <= 3 {
            config.localSearchFrequency = 1 // every generation
            config.localSearchSteps = 40
        } else if ligandRotatableBonds > 15 {
            config.localSearchFrequency = 3 // less frequent for expensive LS
            config.localSearchSteps = 20
        }

        return config
    }
}

// MARK: - VRAM Estimation

struct VRAMEstimate: Sendable {
    let gridBytes: Int
    let populationBytes: Int
    let ligandBytes: Int
    let proteinAtomBytes: Int
    let miscBytes: Int
    var totalBytes: Int { gridBytes + populationBytes + ligandBytes + proteinAtomBytes + miscBytes }
    var totalMB: Float { Float(totalBytes) / (1024 * 1024) }
    let deviceBudgetMB: Float
    var usageRatio: Float { deviceBudgetMB > 0 ? totalMB / deviceBudgetMB : 0 }
}

// MARK: - Docking Result

struct DockingResult: Identifiable, Sendable {
    let id: Int
    var pose: DockPoseSwift
    var energy: Float           // total Vina score (kcal/mol)
    var stericEnergy: Float     // gauss1 + gauss2 + repulsion
    var hydrophobicEnergy: Float
    var hbondEnergy: Float
    var torsionPenalty: Float   // rotational entropy
    var generation: Int
    var clusterID: Int = -1
    var clusterRank: Int = 0
    var transformedAtomPositions: [SIMD3<Float>] = []
    var refinementEnergy: Float? = nil

    // Druse ML scoring outputs (populated when scoring method is .druseAffinity)
    var mlDockingScore: Float? = nil    // pKd * confidence (primary ranking value)
    var mlPKd: Float? = nil             // predicted -log10(Kd)
    var mlPoseConfidence: Float? = nil  // 0-1, how native-like the pose is

    // Drusina extended scoring correction (populated when scoring method is .drusina)
    var drusinaCorrection: Float = 0

    // Pharmacophore constraint penalty (0 = all constraints satisfied)
    var constraintPenalty: Float = 0

    // MMFF94 ligand strain energy (docked − free, kcal/mol; nil if not computed)
    var strainEnergy: Float? = nil

    /// The display score depending on the active scoring method.
    func displayScore(method: ScoringMethod) -> Float {
        switch method {
        case .vina:          return energy
        case .drusina:       return energy  // energy already includes Drusina corrections
        case .druseAffinity: return mlDockingScore ?? energy
        }
    }

    // Backward-compatible aliases
    var vdwEnergy: Float { stericEnergy }
    var elecEnergy: Float { hydrophobicEnergy }
    var desolvEnergy: Float { torsionPenalty }
}

struct DockPoseSwift: Sendable {
    var translation: SIMD3<Float>
    var rotation: simd_quatf
    var torsions: [Float]
    var chiAngles: [Float] = []
}

struct PreparedDockingLigand {
    var heavyAtoms: [Atom]
    var heavyBonds: [Bond]
    var centroid: SIMD3<Float>
    var gpuAtoms: [DockLigandAtom]
}

struct DockingGridSnapshot {
    let stericGridBuffer: MTLBuffer
    let hydrophobicGridBuffer: MTLBuffer
    let hbondGridBuffer: MTLBuffer
    let gridParamsBuffer: MTLBuffer
    let gridParams: GridParams
}

// MARK: - Interaction Detection

struct MolecularInteraction: Identifiable, Sendable {
    let id: Int
    var ligandAtomIndex: Int
    var proteinAtomIndex: Int
    var type: InteractionType
    var distance: Float
    var ligandPosition: SIMD3<Float>
    var proteinPosition: SIMD3<Float>

    enum InteractionType: Int, CaseIterable, Sendable {
        case hbond = 0        // H-bond: N/O donor ↔ acceptor, 2.2-3.5 Å
        case hydrophobic = 1  // Hydrophobic contact: C/S ↔ C/S, 3.3-4.5 Å (filtered)
        case saltBridge = 2   // Salt bridge: charged group pairs, < 4.0 Å
        case piStack = 3      // π-π stacking: aromatic ring centroids, 3.3-5.5 Å
        case piCation = 4     // π-cation: aromatic ring ↔ cation, < 6.0 Å
        case halogen = 5      // Halogen bond: F/Cl/Br ↔ N/O, 2.5-3.5 Å
        case metalCoord = 6   // Metal coordination: Zn/Fe/Mg ↔ N/O/S, < 2.8 Å
        case chPi = 7         // CH-π: C-H ↔ aromatic ring, 3.5-4.5 Å

        var color: SIMD4<Float> {
            switch self {
            case .hbond:       SIMD4(0.2, 0.8, 1.0, 1.0)    // cyan
            case .hydrophobic: SIMD4(0.5, 0.5, 0.5, 0.6)    // gray, subtle
            case .saltBridge:  SIMD4(1.0, 0.5, 0.1, 1.0)    // orange
            case .piStack:     SIMD4(0.7, 0.3, 1.0, 1.0)    // purple
            case .piCation:    SIMD4(1.0, 0.3, 0.7, 1.0)    // magenta
            case .halogen:     SIMD4(0.2, 1.0, 0.5, 1.0)    // green
            case .metalCoord:  SIMD4(1.0, 0.85, 0.0, 1.0)   // gold
            case .chPi:        SIMD4(0.6, 0.5, 0.8, 0.7)    // light purple
            }
        }

        var label: String {
            switch self {
            case .hbond:       "H-bond"
            case .hydrophobic: "Hydrophobic"
            case .saltBridge:  "Salt bridge"
            case .piStack:     "π-π stack"
            case .piCation:    "π-cation"
            case .halogen:     "Halogen bond"
            case .metalCoord:  "Metal coord."
            case .chPi:        "CH-π"
            }
        }
    }
}

// MARK: - Docking Engine

@MainActor
final class DockingEngine {
    let device: MTLDevice
    private(set) var commandQueue: MTLCommandQueue

    private var stericGridPipeline: MTLComputePipelineState!
    private var hydrophobicGridPipeline: MTLComputePipelineState!
    private var hbondGridPipeline: MTLComputePipelineState!
    private var vinaAffinityGridPipeline: MTLComputePipelineState!
    private var scorePipeline: MTLComputePipelineState!
    private var initPopPipeline: MTLComputePipelineState!
    private var evolvePipeline: MTLComputePipelineState!
    private var localSearchPipeline: MTLComputePipelineState!
    private var mcPerturbPipeline: MTLComputePipelineState!
    private var metropolisAcceptPipeline: MTLComputePipelineState!
    private var explicitScorePipeline: MTLComputePipelineState!
    private var localSearchAnalyticalPipeline: MTLComputePipelineState!
    private var drusinaScorePipeline: MTLComputePipelineState?
    private var drusinaCorrectionPipeline: MTLComputePipelineState?

    /// Active local search pipeline based on config.useAnalyticalGradients.
    private var activeLocalSearchPipeline: MTLComputePipelineState {
        config.useAnalyticalGradients ? localSearchAnalyticalPipeline : localSearchPipeline
    }

    private(set) var stericGridBuffer: MTLBuffer?
    private var hydrophobicGridBuffer: MTLBuffer?
    private var hbondGridBuffer: MTLBuffer?
    private(set) var vinaAffinityGridBuffer: MTLBuffer?
    private var vinaTypeIndexBuffer: MTLBuffer?
    private var vinaAffinityTypeBuffer: MTLBuffer?
    private var proteinAtomBuffer: MTLBuffer?
    private(set) var gridParamsBuffer: MTLBuffer?
    private var populationBuffer: MTLBuffer?
    private var offspringBuffer: MTLBuffer?
    private var bestPopulationBuffer: MTLBuffer?
    private var ligandAtomBuffer: MTLBuffer?
    private var gaParamsBuffer: MTLBuffer?
    private var gaParamsRing: [MTLBuffer] = []  // Triple-buffer for async GA loop
    private var pairwiseRMSDPipeline: MTLComputePipelineState?
    private var torsionEdgeBuffer: MTLBuffer?
    private var movingIndicesBuffer: MTLBuffer?
    private var exclusionMaskBuffer: MTLBuffer?

    // Drusina extended scoring buffers
    private var proteinRingBuffer: MTLBuffer?
    private var ligandRingBuffer: MTLBuffer?
    private var proteinCationBuffer: MTLBuffer?
    private var drusinaParamsBuffer: MTLBuffer?
    private var halogenInfoBuffer: MTLBuffer?

    // Pharmacophore constraint buffers
    private var pharmaConstraintBuffer: MTLBuffer?
    private var pharmaParamsBuffer: MTLBuffer?

    private(set) var isRunning = false
    private(set) var currentGeneration = 0
    private(set) var bestEnergy: Float = .infinity

    /// Grid dimensions from last computeGridMaps call (for flex grid proxy)
    private(set) var lastGridTotalPoints: Int = 0
    private(set) var lastGridNumAffinityTypes: Int = 0

    /// Optional flex docking engine for receptor flexibility (induced fit).
    /// When set, flex scoring/evolution/local-search kernels are dispatched
    /// alongside the standard docking loop.
    var flexEngine: FlexDockingEngine?
    private var gridParams = GridParams()
    private var config = DockingConfig()
    /// Tracks the last allocated population buffer capacity to avoid redundant reallocation.
    private var lastPopulationBufferCapacity: Int = 0
    /// Tracks last allocated ligand buffer capacities (in bytes) to avoid churn during batch docking.
    private var lastLigandAtomBufferCapacity: Int = 0
    private var lastTorsionEdgeBufferCapacity: Int = 0
    private var lastMovingIndicesBufferCapacity: Int = 0
    private var lastExclusionMaskBufferCapacity: Int = 0

    /// Diagnostics from the last completed docking run.
    private(set) var lastDiagnostics: DockingDiagnostics?

    var onPoseUpdate: ((DockingResult, [MolecularInteraction]) -> Void)?
    var onGenerationComplete: ((Int, Float) -> Void)?
    var onDockingComplete: (([DockingResult]) -> Void)?

    // Reference to protein atoms for interaction detection
    var proteinAtoms: [Atom] = []
    private var proteinStructure: Molecule?

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
        } catch {
            print("Failed to create docking pipelines: \(error)")
            return nil
        }
    }

    // MARK: - Grid Map Computation

    // MARK: - Vina Atom Typing

    private func hasAttachedHydrogen(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains { molecule.atoms[$0].element == .H }
    }

    private func isBondedToHeteroatom(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains {
            let element = molecule.atoms[$0].element
            return element != .H && element != .C
        }
    }

    private func vinaTypeID(_ type: VinaAtomType) -> Int32 {
        Int32(type.rawValue)
    }

    private var maxSupportedVinaType: Int32 {
        vinaTypeID(VINA_MET_D)
    }

    /// Approximate protein XS typing in the same space Vina uses upstream.
    /// Donor/acceptor assignment leans on residue chemistry while carbon polarity
    /// uses the actual bond graph when available.
    /// Compute Vina XS types for all atoms in a protein molecule.
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

    private func fallbackLigandVinaAtomType(for atomIndex: Int, in molecule: Molecule) -> Int32 {
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

    private func ligandVinaTypes(_ ligand: Molecule) -> [Int32] {
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

    private func swiftXSIsHydrophobic(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_C_H) || xsType == vinaTypeID(VINA_F_H) ||
        xsType == vinaTypeID(VINA_Cl_H) || xsType == vinaTypeID(VINA_Br_H) ||
        xsType == vinaTypeID(VINA_I_H)
    }

    private func swiftXSIsAcceptor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_A) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_A) || xsType == vinaTypeID(VINA_O_DA)
    }

    private func swiftXSIsDonor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_D) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_D) || xsType == vinaTypeID(VINA_O_DA) ||
        xsType == vinaTypeID(VINA_MET_D)
    }

    private func swiftXSRadius(_ xsType: Int32) -> Float {
        let radii: [Float] = [
            1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7,
            2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2
        ]
        let index = Int(xsType)
        guard index >= 0, index < radii.count else { return 0 }
        return radii[index]
    }

    private func swiftSlopeStep(xBad: Float, xGood: Float, x: Float) -> Float {
        if xBad < xGood {
            if x <= xBad { return 0 }
            if x >= xGood { return 1 }
        } else {
            if x >= xBad { return 0 }
            if x <= xGood { return 1 }
        }
        return (x - xBad) / (xGood - xBad)
    }

    private func swiftVinaPairEnergy(_ type1: Int32, _ type2: Int32, distance r: Float) -> Float {
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

    private func intramolecularReferenceEnergy(
        ligandAtoms: [DockLigandAtom],
        exclusionMask: [UInt32],
        maxAtoms: Int
    ) -> Float {
        guard ligandAtoms.count > 1 else { return 0 }

        var total: Float = 0
        for i in 0..<ligandAtoms.count {
            for j in (i + 1)..<ligandAtoms.count {
                let pairIndex = i * maxAtoms + j
                let word = pairIndex / 32
                let bit = pairIndex % 32
                if exclusionMask[word] & (1 << bit) != 0 { continue }

                let r = simd_distance(ligandAtoms[i].position, ligandAtoms[j].position)
                total += swiftVinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, distance: r)
            }
        }
        return total
    }

    private struct RerankRNG {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed &+ 0x9E3779B97F4A7C15
        }

        mutating func nextUInt32() -> UInt32 {
            state = state &* 2862933555777941757 &+ 3037000493
            return UInt32(truncatingIfNeeded: state >> 16)
        }

        mutating func nextFloat() -> Float {
            Float(nextUInt32()) / Float(UInt32.max)
        }

        mutating func signed(amplitude: Float) -> Float {
            (nextFloat() * 2 - 1) * amplitude
        }

        mutating func vectorInUnitSphere(scale: Float) -> SIMD3<Float> {
            for _ in 0..<16 {
                let v = SIMD3<Float>(
                    signed(amplitude: 1),
                    signed(amplitude: 1),
                    signed(amplitude: 1)
                )
                let len2 = simd_length_squared(v)
                if len2 > 1e-4, len2 <= 1 {
                    return v * scale
                }
            }
            return SIMD3<Float>(scale, 0, 0)
        }
    }

    private func wrappedAngle(_ angle: Float) -> Float {
        var wrapped = angle
        while wrapped > .pi { wrapped -= 2 * .pi }
        while wrapped < -.pi { wrapped += 2 * .pi }
        return wrapped
    }

    private func torsions(from pose: DockPose) -> [Float] {
        let count = max(0, min(Int(pose.numTorsions), 32))
        guard count > 0 else { return [] }
        return withUnsafePointer(to: pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                Array(UnsafeBufferPointer(start: buffer, count: count))
            }
        }
    }

    private func chiAngles(from pose: DockPose) -> [Float] {
        let count = max(0, min(Int(pose.numChiAngles), 24))
        guard count > 0 else { return [] }
        return withUnsafePointer(to: pose.chiAngles) {
            $0.withMemoryRebound(to: Float.self, capacity: 24) { buffer in
                Array(UnsafeBufferPointer(start: buffer, count: count))
            }
        }
    }

    private func makeDockPose(from result: DockingResult) -> DockPose {
        var pose = DockPose()
        pose.translation = result.pose.translation
        pose.energy = result.energy
        pose.rotation = SIMD4<Float>(
            result.pose.rotation.imag.x,
            result.pose.rotation.imag.y,
            result.pose.rotation.imag.z,
            result.pose.rotation.real
        )
        let torsions = result.pose.torsions
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(min(torsions.count, 32))
        pose.generation = Int32(result.generation)
        pose.stericEnergy = result.stericEnergy
        pose.hydrophobicEnergy = result.hydrophobicEnergy
        pose.hbondEnergy = result.hbondEnergy
        pose.torsionPenalty = result.torsionPenalty
        pose.clashPenalty = 0
        pose.drusinaCorrection = 0
        pose.constraintPenalty = 0
        return pose
    }

    private func makeRerankSeedPose(from result: DockingResult, variantIndex: Int) -> DockPose {
        var pose = makeDockPose(from: result)
        guard variantIndex > 0 else { return pose }

        let seed = UInt64(bitPattern: Int64(result.id &* 1_315_423_911
            ^ result.clusterID &* 374_761_393
            ^ variantIndex &* 668_265_263))
        var rng = RerankRNG(seed: seed)

        pose.translation += rng.vectorInUnitSphere(scale: 0.75)

        let axis = simd_normalize(rng.vectorInUnitSphere(scale: 1))
        let deltaRotation = simd_quatf(angle: rng.signed(amplitude: 0.18), axis: axis)
        let currentRotation = simd_quatf(ix: pose.rotation.x, iy: pose.rotation.y, iz: pose.rotation.z, r: pose.rotation.w)
        let updatedRotation = deltaRotation * currentRotation
        pose.rotation = SIMD4<Float>(
            updatedRotation.imag.x,
            updatedRotation.imag.y,
            updatedRotation.imag.z,
            updatedRotation.real
        )

        let torsionCount = max(0, min(Int(pose.numTorsions), 32))
        guard torsionCount > 0 else { return pose }
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for index in 0..<torsionCount {
                    buffer[index] = wrappedAngle(buffer[index] + rng.signed(amplitude: 0.30))
                }
            }
        }
        return pose
    }

    private func scorePopulationExplicit(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        // SIMD-cooperative: each SIMD group (32 threads) handles one pose, so dispatch
        // populationSize * simdWidth total threads. Threadgroup size = 32 (one SIMD group).
        let simdWidth = explicitScorePipeline.threadExecutionWidth
        let totalThreads = max(populationSize, 1) * simdWidth
        let tgSize = MTLSize(width: simdWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalThreads + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1), (proteinAtomBuffer!, 2),
            (gridParamsBuffer, 3), (gaParamsBuffer, 4),
            (torsionEdgeBuffer!, 5), (movingIndicesBuffer!, 6),
            (exclusionMaskBuffer!, 7)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: explicitScorePipeline, buffers: buffers,
                        threadGroups: tgCount, threadGroupSize: tgSize)
    }

    private func localOptimizeGrid(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        let tgSize = MTLSize(width: min(max(populationSize, 1), 64), height: 1, depth: 1)
        let tgCount = MTLSize(width: (max(populationSize, 1) + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer, 4), (gaParamsBuffer, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: activeLocalSearchPipeline, buffers: buffers,
                        threadGroups: tgCount, threadGroupSize: tgSize)
    }

    private func rerankClusterRepresentativesExplicit(
        _ results: [DockingResult],
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>
    ) -> [DockingResult] {
        guard config.explicitRerankTopClusters > 0,
              proteinAtomBuffer != nil,
              let gridParamsBuffer,
              let gaParamsBuffer,
              !results.isEmpty else {
            return results
        }

        let grouped = Dictionary(grouping: results, by: \.clusterID)
        let leaders = results
            .filter { $0.clusterRank == 0 }
            .sorted { $0.energy < $1.energy }
        guard !leaders.isEmpty else { return results }

        let rerankCount = min(config.explicitRerankTopClusters, leaders.count)
        let variantsPerCluster = max(config.explicitRerankVariantsPerCluster, 1)
        let rerankLeaders = Array(leaders.prefix(rerankCount))
        var representativePoses: [DockPose] = []
        var variantClusterIDs: [Int] = []
        representativePoses.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        variantClusterIDs.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        for leader in rerankLeaders {
            for variantIndex in 0..<variantsPerCluster {
                representativePoses.append(makeRerankSeedPose(from: leader, variantIndex: variantIndex))
                variantClusterIDs.append(leader.clusterID)
            }
        }

        let repBuffer = device.makeBuffer(
            bytes: &representativePoses,
            length: representativePoses.count * MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )
        guard let repBuffer else { return results }

        let currentGA = gaParamsBuffer.contents().bindMemory(to: GAParams.self, capacity: 1).pointee
        var rerankGA = currentGA
        rerankGA.populationSize = UInt32(representativePoses.count)
        rerankGA.localSearchSteps = UInt32(max(config.explicitRerankLocalSearchSteps, 1))
        let rerankGABuffer = device.makeBuffer(
            bytes: &rerankGA,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )
        guard let rerankGABuffer else { return results }

        localOptimizeGrid(
            buffer: repBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gaParamsBuffer: rerankGABuffer,
            populationSize: representativePoses.count
        )
        scorePopulationExplicit(
            buffer: repBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gaParamsBuffer: rerankGABuffer,
            populationSize: representativePoses.count
        )

        let rescoredLeaders = extractAllResults(
            from: repBuffer,
            ligandAtoms: ligandAtoms,
            centroid: centroid,
            idOffset: 0,
            sortByEnergy: false
        )
        guard rescoredLeaders.count == representativePoses.count else { return results }

        var representativeByCluster: [Int: DockingResult] = [:]
        for (index, rescored) in rescoredLeaders.enumerated() {
            let sourceClusterID = variantClusterIDs[index]
            var updated = rescored
            updated.clusterID = sourceClusterID
            updated.clusterRank = 0
            if updated.energy < (representativeByCluster[sourceClusterID]?.energy ?? .infinity) {
                representativeByCluster[sourceClusterID] = updated
            }
        }

        let sortedClusterIDs = leaders
            .map(\.clusterID)
            .sorted {
                let lhs = representativeByCluster[$0]?.energy ?? grouped[$0]?.first?.energy ?? .infinity
                let rhs = representativeByCluster[$1]?.energy ?? grouped[$1]?.first?.energy ?? .infinity
                return lhs < rhs
            }

        var reranked: [DockingResult] = []
        reranked.reserveCapacity(results.count)

        for (newClusterID, oldClusterID) in sortedClusterIDs.enumerated() {
            guard let members = grouped[oldClusterID] else { continue }
            let originalLeader = members.first { $0.clusterRank == 0 }
            var leader = representativeByCluster[oldClusterID] ?? originalLeader ?? members[0]
            leader.clusterID = newClusterID
            leader.clusterRank = 0
            reranked.append(leader)

            var rank = 1
            for member in members.sorted(by: { $0.energy < $1.energy }) where member.clusterRank != 0 {
                var updated = member
                updated.clusterID = newClusterID
                updated.clusterRank = rank
                rank += 1
                reranked.append(updated)
            }
        }

        return reranked.sorted {
            if $0.energy != $1.energy { return $0.energy < $1.energy }
            if $0.clusterID != $1.clusterID { return $0.clusterID < $1.clusterID }
            if $0.clusterRank != $1.clusterRank { return $0.clusterRank < $1.clusterRank }
            return $0.id < $1.id
        }
    }

    func computeGridMaps(protein: Molecule, pocket: BindingPocket, spacing: Float = 0.375,
                          ligandExtent: SIMD3<Float>? = nil,
                          requiredVinaTypes: [Int32] = []) {
        proteinStructure = protein
        let heavyAtoms = protein.atoms.filter { $0.element != .H }
        self.proteinAtoms = heavyAtoms

        let gpuAtoms = protein.atoms.enumerated().compactMap { atomIndex, atom -> GridProteinAtom? in
            guard atom.element != .H else { return nil }
            return GridProteinAtom(
                position: atom.position,
                vdwRadius: atom.element.vdwRadius,
                charge: electrostaticCharge(for: atom),
                vinaType: vinaProteinAtomType(for: atomIndex, in: protein),
                _pad0: 0, _pad1: 0
            )
        }

        let activeVinaTypes = Array(Set(requiredVinaTypes.filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()

        let gridMapCount: UInt64 = activeVinaTypes.isEmpty ? 3 : UInt64(3 + activeVinaTypes.count)

        // Keep the search box centered on the selected pocket. Covering the whole
        // protein here turns pocket docking into accidental global surface docking,
        // which allows the optimizer to converge on unrelated basins.
        // Pocket detectors already return padded half-extents, so avoid inflating
        // the translation domain a second time here.
        let searchPadding: Float = 0.0
        let gridPadding: Float = 3.0
        let ligandMargin = ligandExtent ?? SIMD3<Float>(repeating: 4.0)
        let searchCenter = pocket.center
        let searchHalfExtent = pocket.size + SIMD3<Float>(repeating: searchPadding)
        let gridHalfExtent = searchHalfExtent + ligandMargin + SIMD3<Float>(repeating: gridPadding)
        let boxMin = searchCenter - gridHalfExtent
        let boxMax = searchCenter + gridHalfExtent
        let boxSize = boxMax - boxMin

        // Memory guard: if the typed maps would exceed the GPU budget, coarsen spacing.
        var effectiveSpacing = spacing
        // Half-precision grids use 2 bytes per value instead of 4, so we can
        // accommodate 2x the grid points for the same memory budget (~91.5 MB → ~91.5 MB)
        let maxGridFloatValues: UInt64 = 48_000_000

        // Safely compute grid dimensions — clamp to avoid overflow from inf/NaN
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

        // Pre-flight VRAM check
        lastGridTotalPoints = Int(totalPoints)
        lastGridNumAffinityTypes = activeVinaTypes.count

        let vramEstimate = estimateVRAMUsage(
            gridDims: SIMD3(nx, ny, nz),
            numAffinityTypes: activeVinaTypes.count,
            populationSize: 300, // default; actual popSize set later in runDocking
            numLigandAtoms: 50,  // reasonable upper bound for estimation
            numTorsions: 10,
            numProteinAtoms: gpuAtoms.count
        )
        if vramEstimate.usageRatio > 0.85 {
            print("[DockingEngine] WARNING: Estimated VRAM usage \(String(format: "%.0f", vramEstimate.totalMB))MB / \(String(format: "%.0f", vramEstimate.deviceBudgetMB))MB (\(String(format: "%.0f%%", vramEstimate.usageRatio * 100))) — coarsening grid spacing")
            // Already handled by adaptive spacing above, but log the warning
        }

        var proteinGPUAtoms = gpuAtoms
        proteinAtomBuffer = device.makeBuffer(bytes: &proteinGPUAtoms, length: proteinGPUAtoms.count * MemoryLayout<GridProteinAtom>.stride, options: .storageModeShared)
        gridParamsBuffer = device.makeBuffer(bytes: &gridParams, length: MemoryLayout<GridParams>.stride, options: .storageModeShared)

        // Grid maps use half-precision (Float16) on GPU for 2x bandwidth
        let gridByteSize = Int(totalPoints) * MemoryLayout<UInt16>.stride   // half = 2 bytes
        stericGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hydrophobicGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hbondGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)

        if stericGridBuffer == nil || hydrophobicGridBuffer == nil || hbondGridBuffer == nil {
            ActivityLog.shared.error("[Engine] Grid buffer allocation failed: \(gridByteSize) bytes per map (\(totalPoints) points)", category: .dock)
            return
        }

        if activeVinaTypes.isEmpty {
            vinaAffinityGridBuffer = nil
            vinaTypeIndexBuffer = nil
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

        // Dispatch each grid kernel in its own command buffer to avoid GPU timeouts
        // on large proteins (>3000 atoms × >1M grid points can exceed Metal's ~5s limit)
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

        // Affinity maps: one command buffer per dispatch (separate from basic grids)
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
            gridParams: gridParams
        )
    }

    func prepareLigandGeometry(_ ligand: Molecule) -> PreparedDockingLigand {
        let chargedLigand = ligandWithDockingCharges(ligand)
        let allVinaTypes = ligandVinaTypes(chargedLigand)
        let heavyEntries = chargedLigand.atoms.enumerated().filter { $0.element.element != .H }
        let heavyAtoms = heavyEntries.map(\.element)
        let centroid = heavyAtoms.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(max(heavyAtoms.count, 1))

        var oldToNew: [Int: Int] = [:]
        for (newIdx, entry) in heavyEntries.enumerated() {
            oldToNew[entry.offset] = newIdx
        }

        let heavyBonds: [Bond] = chargedLigand.bonds.compactMap { bond in
            guard let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }

        let gpuAtoms: [DockLigandAtom] = heavyEntries.map { entry in
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

    // MARK: - Drusina Buffer Preparation

    /// Prepare GPU buffers for Drusina extended scoring (protein rings, cations, ligand rings, halogens).
    private func prepareDrusinaBuffers(
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

        // --- Ligand aromatic rings (atom indices, centroid/normal computed on GPU from transformed positions) ---
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

        // --- Protein cations (LYS NZ, ARG NH1/NH2/CZ, metals, any +charge) ---
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

        // --- Halogen bond info (ligand halogen → bonded carbon mapping) ---
        var halogens: [HalogenBondInfo] = []
        for (i, gpuAtom) in gpuLigAtoms.enumerated() {
            let vt = gpuAtom.vinaType
            let isHalogen = (vt == Int32(VINA_F_H.rawValue) || vt == Int32(VINA_Cl_H.rawValue) ||
                             vt == Int32(VINA_Br_H.rawValue) || vt == Int32(VINA_I_H.rawValue))
            guard isHalogen else { continue }
            // Find bonded carbon
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

        // --- Drusina parameters ---
        var params = DrusinaParams(
            numProteinRings: UInt32(protRings.count),
            numLigandRings: UInt32(ligRings.count),
            numProteinCations: UInt32(max(cations.count - (cations.first == .zero ? 1 : 0), 0)),
            numHalogens: UInt32(halogens.first?.halogenAtomIndex == -1 ? 0 : halogens.count),
            wPiPi: -0.40,
            wPiCation: -0.80,
            wHalogenBond: -0.50,
            wMetalCoord: -1.00)
        drusinaParamsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<DrusinaParams>.stride,
            options: .storageModeShared)
    }

    // MARK: - Pharmacophore Constraint Buffers

    /// Prepare GPU buffers for pharmacophore constraints.
    /// Always creates buffers (zero-constraint PharmacophoreParams causes early return on GPU).
    func prepareConstraintBuffers(_ constraints: [PharmacophoreConstraintDef],
                                   atoms: [Atom], residues: [Residue]) {
        let (gpuConstraints, params) = PharmacophoreConstraintDef.toGPUBuffers(
            constraints: constraints, atoms: atoms, residues: residues
        )

        if gpuConstraints.isEmpty {
            // Zero-constraint case: still need valid buffers for Metal
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

    /// Device VRAM budget in MB (recommended max working set).
    var deviceVRAMBudgetMB: Float {
        Float(device.recommendedMaxWorkingSetSize) / (1024 * 1024)
    }

    /// Estimate total VRAM usage for a docking run with the given parameters.
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
        let gridBytes = totalPoints * MemoryLayout<UInt16>.stride * gridMapCount  // half-precision grids

        // 3 population buffers (current, offspring, best) + 3 ring buffers for GAParams
        let poseStride = 304 // DockPose stride (from ShaderTypes.h)
        let gaParamsStride = 192
        let populationBytes = 3 * populationSize * poseStride + 3 * gaParamsStride

        // Ligand: atoms + torsion edges + moving indices + exclusion mask
        let ligandAtomStride = 32 // DockLigandAtom stride
        let torsionEdgeStride = 16
        let ligandBytes = numLigandAtoms * ligandAtomStride
            + numTorsions * torsionEdgeStride
            + numLigandAtoms * MemoryLayout<UInt32>.stride // moving indices (upper bound)
            + (numLigandAtoms * numLigandAtoms + 31) / 32 * MemoryLayout<UInt32>.stride // exclusion mask bits

        let proteinAtomBytes = numProteinAtoms * 32 // GridProteinAtom stride

        // Type lookup + affinity type array + grid params
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

    /// Debug: read back grid map statistics (grid maps are stored in half precision)
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

    // MARK: - Run Docking

    func runDocking(
        ligand: Molecule, pocket: BindingPocket, config: DockingConfig = DockingConfig(),
        scoringMethod: ScoringMethod = .vina
    ) async -> [DockingResult] {
        guard !isRunning else { return [] }
        self.config = config
        isRunning = true
        currentGeneration = 0
        bestEnergy = .infinity

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        ActivityLog.shared.info(
            "[Engine] Ligand geometry: \(heavyAtoms.count) heavy atoms, \(heavyBonds.count) heavy bonds, " +
            "\(gpuLigAtoms.count) GPU atoms, centroid=(\(String(format: "%.2f, %.2f, %.2f", centroid.x, centroid.y, centroid.z)))",
            category: .dock
        )

        // Metal shaders use fixed-size stack arrays of 128 atoms; reject oversized ligands
        // to prevent silent truncation and exclusion mask index out-of-bounds
        guard gpuLigAtoms.count <= 128 else {
            ActivityLog.shared.error("[Engine] Ligand has \(gpuLigAtoms.count) heavy atoms, exceeding the 128-atom GPU limit", category: .dock)
            isRunning = false
            return []
        }

        // Compute ligand bounding half-extent (centroid-subtracted coordinates)
        var ligMin = SIMD3<Float>(repeating: .infinity)
        var ligMax = SIMD3<Float>(repeating: -.infinity)
        var ligandRadiusSquared: Float = 0
        for a in gpuLigAtoms {
            ligMin = simd_min(ligMin, a.position)
            ligMax = simd_max(ligMax, a.position)
            ligandRadiusSquared = max(ligandRadiusSquared, simd_length_squared(a.position))
        }
        let ligandHalfExtent = (ligMax - ligMin) * 0.5
        let ligandRadius = max(sqrt(ligandRadiusSquared / Float(max(gpuLigAtoms.count, 1))), 1.0)

        let requiredVinaTypes = Array(Set(gpuLigAtoms.map(\.vinaType).filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()
        if let protein = proteinStructure ?? (!proteinAtoms.isEmpty ? Molecule(name: "cached", atoms: proteinAtoms, bonds: [], title: "") : nil) {
            computeGridMaps(
                protein: protein,
                pocket: pocket,
                spacing: config.gridSpacing,
                ligandExtent: ligandHalfExtent,
                requiredVinaTypes: requiredVinaTypes
            )
        }

        // Reuse ligand atom buffer if large enough, otherwise reallocate
        let ligAtomSize = gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride
        if ligAtomSize > lastLigandAtomBufferCapacity {
            ligandAtomBuffer = device.makeBuffer(bytes: &gpuLigAtoms, length: ligAtomSize, options: .storageModeShared)
            lastLigandAtomBufferCapacity = ligAtomSize
        } else {
            ligandAtomBuffer?.contents().copyMemory(from: &gpuLigAtoms, byteCount: ligAtomSize)
        }

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map { Int32($0) })
        }
        let numTorsions = min(torsionEdges.count, 32)

        // Create torsion Metal buffers (even if empty, need valid pointers)
        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }
        let torsionSize = torsionEdges.count * MemoryLayout<TorsionEdge>.stride
        if torsionSize > lastTorsionEdgeBufferCapacity {
            torsionEdgeBuffer = device.makeBuffer(bytes: &torsionEdges, length: torsionSize, options: .storageModeShared)
            lastTorsionEdgeBufferCapacity = torsionSize
        } else {
            torsionEdgeBuffer?.contents().copyMemory(from: &torsionEdges, byteCount: torsionSize)
        }
        let movingSize = movingIndices.count * MemoryLayout<Int32>.stride
        if movingSize > lastMovingIndicesBufferCapacity {
            movingIndicesBuffer = device.makeBuffer(bytes: &movingIndices, length: movingSize, options: .storageModeShared)
            lastMovingIndicesBufferCapacity = movingSize
        } else {
            movingIndicesBuffer?.contents().copyMemory(from: &movingIndices, byteCount: movingSize)
        }

        // Build exclusion mask for intramolecular clash detection.
        // Marks 1-2 (bonded) and 1-3 (angle) pairs to skip during clash evaluation.
        let maxAtoms = 128
        let maskWords = (maxAtoms * maxAtoms + 31) / 32
        var mask = [UInt32](repeating: 0, count: maskWords)

        // Build adjacency from heavy bonds
        var adj: [[Int]] = Array(repeating: [], count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        // Mark 1-2 and 1-3 pairs in the bitmask
        for i in 0..<gpuLigAtoms.count {
            for j in adj[i] where j > i {
                // 1-2 pair (bonded)
                let idx12 = i * maxAtoms + j
                mask[idx12 / 32] |= 1 << (idx12 % 32)
                // 1-3 pairs (share a bond with i)
                for k in adj[j] where k > i && k != i {
                    let idx13 = min(i, k) * maxAtoms + max(i, k)
                    mask[idx13 / 32] |= 1 << (idx13 % 32)
                }
            }
        }

        let maskSize = mask.count * MemoryLayout<UInt32>.stride
        if maskSize > lastExclusionMaskBufferCapacity {
            exclusionMaskBuffer = device.makeBuffer(bytes: &mask, length: maskSize, options: .storageModeShared)
            lastExclusionMaskBufferCapacity = maskSize
        } else {
            // Mask is fixed-size (128x128 bitmask) but contents differ per ligand — must overwrite
            exclusionMaskBuffer?.contents().copyMemory(from: &mask, byteCount: maskSize)
        }
        let referenceIntraEnergy = intramolecularReferenceEnergy(
            ligandAtoms: gpuLigAtoms,
            exclusionMask: mask,
            maxAtoms: maxAtoms
        )

        ActivityLog.shared.info(
            "[Engine] Torsion tree: \(numTorsions) rotatable bonds, \(movingIndices.count) moving atom indices, refIntraE=\(String(format: "%.3f", referenceIntraEnergy))",
            category: .dock
        )

        // Prepare Drusina buffers if using extended scoring
        let useDrusina = scoringMethod == .drusina && drusinaScorePipeline != nil
        if useDrusina {
            prepareDrusinaBuffers(
                ligandAtoms: heavyAtoms,
                ligandBonds: heavyBonds,
                gpuLigAtoms: gpuLigAtoms,
                centroid: centroid)
        }
        ActivityLog.shared.info("[Engine] Scoring method: \(scoringMethod.rawValue), Drusina: \(useDrusina)", category: .dock)

        let popSize = config.populationSize
        let poseSize = popSize * MemoryLayout<DockPose>.stride
        // Reuse population buffers if they're already large enough
        if poseSize > lastPopulationBufferCapacity {
            populationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            offspringBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            bestPopulationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            lastPopulationBufferCapacity = poseSize
        }

        var gaParams = GAParams(
            populationSize: UInt32(popSize),
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: UInt32(max(config.localSearchSteps, 1)),
            mutationRate: config.mutationRate,
            crossoverRate: config.crossoverRate,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: config.gridSpacing,
            ligandRadius: ligandRadius,
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: referenceIntraEnergy
        )
        gaParamsBuffer = device.makeBuffer(bytes: &gaParams, length: MemoryLayout<GAParams>.stride, options: .storageModeShared)

        // Triple-buffer ring for async GA loop (avoids CPU/GPU sync stalls)
        gaParamsRing = (0..<3).compactMap { _ in
            device.makeBuffer(length: MemoryLayout<GAParams>.stride, options: .storageModeShared)
        }

        let tgSize = MTLSize(width: min(popSize, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (popSize + 255) / 256, height: 1, depth: 1)

        if !config.enableFlexibility {
            gaParams.numTorsions = 0
            gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)
        }

        // Buffer validation logging
        ActivityLog.shared.info(
            "[Engine] GA: pop=\(popSize), torsions=\(gaParams.numTorsions), " +
            "ligAtoms=\(gaParams.numLigandAtoms), lsSteps=\(gaParams.localSearchSteps), " +
            "T=\(String(format: "%.3f", gaParams.mcTemperature)), flex=\(config.enableFlexibility)",
            category: .dock
        )
        ActivityLog.shared.info(
            "[Engine] GPU dispatch: tgSize=\(tgSize.width), tgCount=\(tgCount.width), poseStride=\(MemoryLayout<DockPose>.stride) bytes",
            category: .dock
        )
        ActivityLog.shared.info(
            "[Engine] Buffers: pop=\(populationBuffer != nil), offspring=\(offspringBuffer != nil), " +
            "best=\(bestPopulationBuffer != nil), ligAtom=\(ligandAtomBuffer != nil), " +
            "grid=\(vinaAffinityGridBuffer != nil), gridParams=\(gridParamsBuffer != nil), " +
            "torsion=\(torsionEdgeBuffer != nil), moving=\(movingIndicesBuffer != nil), " +
            "exclusion=\(exclusionMaskBuffer != nil), gaParams=\(gaParamsBuffer != nil)",
            category: .dock
        )
        if let flexEng = flexEngine, flexEng.isEnabled {
            ActivityLog.shared.info(
                "[Engine] Flex: atoms=\(flexEng.numFlexAtoms), torsions=\(flexEng.numFlexTorsions), " +
                "chiSlots=\(flexEng.numChiSlots), buffers=(atom=\(flexEng.flexAtomBuffer != nil), " +
                "edge=\(flexEng.flexTorsionEdgeBuffer != nil), moving=\(flexEng.flexMovingIndicesBuffer != nil), " +
                "params=\(flexEng.flexParamsBuffer != nil))",
                category: .dock
            )
        }

        let totalRuns = max(config.numRuns, 1)
        var aggregatedResults: [DockingResult] = []
        let localSearchFrequency = max(config.localSearchFrequency, 1)
        let liveUpdateFrequency = max(config.liveUpdateFrequency, 1)

        func emitLiveUpdate(generation: Int) {
            if let best = extractBestPose(from: bestPopulationBuffer, ligandAtoms: heavyAtoms, centroid: centroid) {
                bestEnergy = min(bestEnergy, best.energy)
                let interactions = InteractionDetector.detect(
                    ligandAtoms: heavyAtoms,
                    ligandPositions: best.transformedAtomPositions,
                    proteinAtoms: proteinAtoms,
                    ligandBonds: heavyBonds
                )
                onPoseUpdate?(best, interactions)
            }
            onGenerationComplete?(generation, bestEnergy)
        }

        // Validate all critical buffers before entering the GA loop
        guard let popBuf = populationBuffer,
              let offBuf = offspringBuffer,
              let bestBuf = bestPopulationBuffer,
              let ligBuf = ligandAtomBuffer,
              let affinityBuf = vinaAffinityGridBuffer,
              let typeIdxBuf = vinaTypeIndexBuffer,
              let gpBuf = gridParamsBuffer,
              let teBuf = torsionEdgeBuffer,
              let miBuf = movingIndicesBuffer,
              let emBuf = exclusionMaskBuffer,
              let gaBuf = gaParamsBuffer
        else {
            ActivityLog.shared.error("[Engine] Critical buffer nil — cannot start GA. pop=\(populationBuffer != nil) grid=\(gridParamsBuffer != nil) affinity=\(vinaAffinityGridBuffer != nil) typeIdx=\(vinaTypeIndexBuffer != nil) ligand=\(ligandAtomBuffer != nil) torsion=\(torsionEdgeBuffer != nil) moving=\(movingIndicesBuffer != nil) exclusion=\(exclusionMaskBuffer != nil) ga=\(gaParamsBuffer != nil)", category: .dock)
            isRunning = false
            return []
        }

        runLoop: for runIndex in 0..<totalRuns {
            guard isRunning else { break }

            // Pipeline init → local search → score into a single async sequence,
            // syncing only once before the main generation loop starts.
            dispatchCompute(pipeline: initPopPipeline, buffers: [
                (popBuf, 0), (gpBuf, 1), (gaBuf, 2)
            ], threadGroups: tgCount, threadGroupSize: tgSize)
            localOptimize(buffer: popBuf, tg: tgCount, tgs: tgSize)
            if useDrusina {
                scoreDrusina(buffer: popBuf, gaParamsBuffer: gaBuf, tg: tgCount, tgs: tgSize)
            } else {
                scorePopulation(buffer: popBuf, tg: tgCount, tgs: tgSize)
            }
            copyPoseBuffer(from: popBuf, to: bestBuf, poseCount: popSize)

            let generationBase = runIndex * config.generationsPerRun

            let explorationCutoff = Int(Float(config.generationsPerRun) * config.explorationPhaseRatio)

            var lastCmdBuf: (any MTLCommandBuffer)?

            for step in 0..<config.generationsPerRun {
                guard isRunning else {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    aggregatedResults.append(contentsOf: extractAllResults(
                        from: bestBuf,
                        ligandAtoms: heavyAtoms,
                        centroid: centroid,
                        idOffset: aggregatedResults.count
                    ))
                    break runLoop
                }
                let globalGeneration = generationBase + step
                currentGeneration = globalGeneration
                gaParams.generation = UInt32(globalGeneration)

                // Two-phase search: exploration phase uses wider steps for broader grid coverage,
                // refinement phase uses tighter steps for precise local optimization
                if step < explorationCutoff {
                    // Exploration phase: broader search
                    gaParams.translationStep = config.explorationTranslationStep
                    gaParams.rotationStep = config.explorationRotationStep
                    gaParams.mutationRate = config.explorationMutationRate
                } else if step == explorationCutoff {
                    // Switch to refinement phase
                    gaParams.translationStep = config.translationStep
                    gaParams.rotationStep = config.rotationStep
                    gaParams.mutationRate = config.mutationRate
                }

                // Triple-buffer gaParams to avoid CPU/GPU sync stalls:
                // CPU writes to ring buffer N while GPU may still read ring buffer N-1/N-2.
                // Metal command queue serialization ensures correct ordering.
                let ringBuf = gaParamsRing[step % 3]
                ringBuf.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)

                let perturbBuffers: [(MTLBuffer, Int)] = [
                    (offBuf, 0), (popBuf, 1),
                    (ringBuf, 2), (gpBuf, 3)
                ]
                var vinaScoreBuffers: [(MTLBuffer, Int)] = [
                    (offBuf, 0), (ligBuf, 1),
                    (affinityBuf, 2), (typeIdxBuf, 3),
                    (gpBuf, 4), (ringBuf, 5),
                    (teBuf, 6), (miBuf, 7),
                    (emBuf, 8)
                ]
                // Append pharmacophore constraint buffers at indices 15-16
                if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
                    vinaScoreBuffers.append(contentsOf: [
                        (pcBuf, 15), (ppBuf, 16)
                    ])
                }
                let acceptBuffers: [(MTLBuffer, Int)] = [
                    (popBuf, 0), (offBuf, 1),
                    (bestBuf, 2), (ringBuf, 3)
                ]
                // Build dispatch sequence: perturb → [chi evolve] → local search → score → [flex score] → [flex LS] → accept
                // Flex dispatches must happen BEFORE metropolisAccept so the GA sees the full energy.
                var dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])] = [
                    (pipeline: mcPerturbPipeline, buffers: perturbBuffers)
                ]
                if step % localSearchFrequency == 0 {
                    dispatches.append((pipeline: activeLocalSearchPipeline, buffers: vinaScoreBuffers))
                }
                if useDrusina, let drusinaPipe = drusinaScorePipeline {
                    // Drusina: grid Vina + π-π, π-cation, halogen, metal corrections
                    var drusinaBuffers = vinaScoreBuffers
                    drusinaBuffers.append(contentsOf: [
                        (proteinRingBuffer!, 9), (ligandRingBuffer!, 10),
                        (proteinCationBuffer!, 11), (drusinaParamsBuffer!, 12),
                        (proteinAtomBuffer!, 13), (halogenInfoBuffer!, 14)
                    ])
                    dispatches.append((pipeline: drusinaPipe, buffers: drusinaBuffers))
                } else {
                    dispatches.append((pipeline: scorePipeline, buffers: vinaScoreBuffers))
                }
                // Flex scoring BEFORE accept so GA sees the complete energy landscape.
                // Include metropolis accept in the same batch to avoid a GPU sync per generation.
                if let fe = flexEngine, fe.isEnabled {
                    // Flex dispatches need their own command buffers (different pipeline/buffer sets).
                    // Batch the main dispatches first, then flex, then accept — all async.
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                    fe.dispatchChiEvolution(
                        offspringBuffer: offBuf, populationBuffer: popBuf,
                        gaParamsBuffer: ringBuf, populationSize: popSize
                    )
                    fe.dispatchFlexScoring(
                        populationBuffer: offBuf, ligandAtomBuffer: ligBuf,
                        gaParamsBuffer: ringBuf, torsionEdgeBuffer: teBuf,
                        movingIndicesBuffer: miBuf, populationSize: popSize
                    )
                    if step % localSearchFrequency == 0 {
                        fe.dispatchFlexLocalSearch(
                            populationBuffer: offBuf, ligandAtomBuffer: ligBuf,
                            gaParamsBuffer: ringBuf, torsionEdgeBuffer: teBuf,
                            movingIndicesBuffer: miBuf, affinityGridBuffer: affinityBuf,
                            typeIndexBuffer: typeIdxBuf, gridParamsBuffer: gpBuf,
                            exclusionMaskBuffer: emBuf, populationSize: popSize
                        )
                    }
                    // Accept after flex scoring — still needs its own sync since flex used separate CBs
                    lastCmdBuf = dispatchComputeAsync(pipeline: metropolisAcceptPipeline, buffers: acceptBuffers,
                                                       threadGroups: tgCount, threadGroupSize: tgSize)
                } else {
                    // No flex: append accept to the same batch as perturb+score → single command buffer
                    dispatches.append((pipeline: metropolisAcceptPipeline, buffers: acceptBuffers))
                    lastCmdBuf = dispatchBatchAsync(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)
                }

                // Sync with GPU only at live update boundaries — not every generation.
                // This lets the GPU pipeline multiple generations without CPU round-trips.
                let updateFreq = step < explorationCutoff
                    ? max(liveUpdateFrequency * 2 / 3, 1)
                    : liveUpdateFrequency
                if step % updateFreq == 0 || step == config.generationsPerRun - 1 {
                    if let buf = lastCmdBuf {
                        await buf.completed()
                        lastCmdBuf = nil
                    }
                    emitLiveUpdate(generation: globalGeneration)
                }

                await Task.yield()
            }

            // Ensure all GPU work completes before reading results
            if let buf = lastCmdBuf {
                await buf.completed()
                lastCmdBuf = nil
            }

            aggregatedResults.append(contentsOf: extractAllResults(
                from: bestPopulationBuffer,
                ligandAtoms: heavyAtoms,
                centroid: centroid,
                idOffset: aggregatedResults.count
            ))
        }

        let clustered = await clusterPoses(aggregatedResults)
        let reranked = rerankClusterRepresentativesExplicit(
            clustered,
            ligandAtoms: heavyAtoms,
            centroid: centroid
        )

        // Compute and store quality diagnostics
        lastDiagnostics = computeDiagnostics(
            results: reranked,
            ligandAtoms: heavyAtoms,
            heavyBonds: heavyBonds
        )

        isRunning = false
        onDockingComplete?(reranked)
        return reranked
    }

    func stopDocking() { isRunning = false }

    // MARK: - Debug Scoring

    /// Score a user-specified pose against the currently loaded typed Vina maps.
    /// Useful for checking whether the crystal/native pose is actually favorable
    /// under the same Metal kernels used by the GA/ILS search.
    func debugScorePose(
        ligand: Molecule,
        translation: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        torsions: [Float] = []
    ) -> DockingResult? {
        guard vinaAffinityGridBuffer != nil,
              vinaTypeIndexBuffer != nil,
              gridParamsBuffer != nil else {
            return nil
        }

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        guard gpuLigAtoms.count <= 128 else { return nil }

        ligandAtomBuffer = device.makeBuffer(
            bytes: &gpuLigAtoms,
            length: gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride,
            options: .storageModeShared
        )

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map(Int32.init))
        }
        let numTorsions = min(torsionEdges.count, 32)

        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }

        torsionEdgeBuffer = device.makeBuffer(
            bytes: &torsionEdges,
            length: torsionEdges.count * MemoryLayout<TorsionEdge>.stride,
            options: .storageModeShared
        )
        movingIndicesBuffer = device.makeBuffer(
            bytes: &movingIndices,
            length: movingIndices.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        )

        let maxAtoms = 128
        let maskWords = (maxAtoms * maxAtoms + 31) / 32
        var mask = [UInt32](repeating: 0, count: maskWords)
        var adjacency = Array(repeating: [Int](), count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adjacency[a].append(b)
            adjacency[b].append(a)
        }
        for i in 0..<gpuLigAtoms.count {
            for j in adjacency[i] where j > i {
                let idx12 = i * maxAtoms + j
                mask[idx12 / 32] |= 1 << (idx12 % 32)
                for k in adjacency[j] where k > i && k != i {
                    let idx13 = min(i, k) * maxAtoms + max(i, k)
                    mask[idx13 / 32] |= 1 << (idx13 % 32)
                }
            }
        }
        exclusionMaskBuffer = device.makeBuffer(
            bytes: &mask,
            length: mask.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        var pose = DockPose()
        pose.translation = translation
        pose.energy = 1e10
        pose.rotation = SIMD4<Float>(rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(numTorsions)
        pose.generation = 0
        pose.stericEnergy = 0
        pose.hydrophobicEnergy = 0
        pose.hbondEnergy = 0
        pose.torsionPenalty = 0
        pose.clashPenalty = 0
        pose.drusinaCorrection = 0
        pose.constraintPenalty = 0

        populationBuffer = device.makeBuffer(
            bytes: &pose,
            length: MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )

        var gaParams = GAParams(
            populationSize: 1,
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: 1,
            mutationRate: 0,
            crossoverRate: 0,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: gridParams.spacing,
            ligandRadius: max(sqrt(gpuLigAtoms.map { simd_length_squared($0.position) }.reduce(0, +) / Float(max(gpuLigAtoms.count, 1))), 1.0),
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: intramolecularReferenceEnergy(
                ligandAtoms: gpuLigAtoms,
                exclusionMask: mask,
                maxAtoms: maxAtoms
            )
        )
        gaParamsBuffer = device.makeBuffer(
            bytes: &gaParams,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )

        let tgSize = MTLSize(width: 1, height: 1, depth: 1)
        let tgCount = MTLSize(width: 1, height: 1, depth: 1)
        let wasRunning = isRunning
        isRunning = true
        scorePopulation(buffer: populationBuffer!, tg: tgCount, tgs: tgSize)
        isRunning = wasRunning

        return extractBestPose(ligandAtoms: heavyAtoms, centroid: centroid)
    }

    // MARK: - GPU Helpers

    private func localOptimize(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer!, 4), (gaParamsBuffer!, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: activeLocalSearchPipeline, buffers: buffers,
                        threadGroups: tg, threadGroupSize: tgs)
    }

    private func scorePopulation(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        var buffers: [(MTLBuffer, Int)] = [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer!, 4), (gaParamsBuffer!, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ]
        if let pcBuf = pharmaConstraintBuffer, let ppBuf = pharmaParamsBuffer {
            buffers.append(contentsOf: [(pcBuf, 15), (ppBuf, 16)])
        }
        dispatchCompute(pipeline: scorePipeline, buffers: buffers,
                        threadGroups: tg, threadGroupSize: tgs)
    }

    private func scoreDrusina(buffer: MTLBuffer, gaParamsBuffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        guard let pipe = drusinaScorePipeline,
              let prBuf = proteinRingBuffer, let lrBuf = ligandRingBuffer,
              let pcBuf = proteinCationBuffer, let dpBuf = drusinaParamsBuffer,
              let paBuf = proteinAtomBuffer, let hiBuf = halogenInfoBuffer else {
            scorePopulation(buffer: buffer, tg: tg, tgs: tgs)
            return
        }
        dispatchCompute(pipeline: pipe, buffers: [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer!, 4), (gaParamsBuffer, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8),
            (prBuf, 9), (lrBuf, 10), (pcBuf, 11), (dpBuf, 12), (paBuf, 13), (hiBuf, 14)
        ], threadGroups: tg, threadGroupSize: tgs)
    }

    private func dispatchCompute(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize, threadGroupSize: MTLSize
    ) {
        guard isRunning else { return }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else {
            ActivityLog.shared.warn("[Engine] Skipped dispatch: zero threadGroups=\(threadGroups.width)×\(threadGroups.height)×\(threadGroups.depth) or threadGroupSize=\(threadGroupSize.width)×\(threadGroupSize.height)×\(threadGroupSize.depth)", category: .dock)
            return
        }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if cmdBuf.status == .error {
            let err = cmdBuf.error?.localizedDescription ?? "unknown"
            ActivityLog.shared.error("[Engine] GPU command buffer error: \(err)", category: .dock)
            isRunning = false
        }
    }

    /// Async single-dispatch — commits work without blocking CPU.
    /// Returns the command buffer so callers can await completion only when needed.
    @discardableResult
    private func dispatchComputeAsync(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize, threadGroupSize: MTLSize
    ) -> MTLCommandBuffer? {
        guard isRunning else { return nil }
        guard threadGroups.width > 0, threadGroups.height > 0, threadGroups.depth > 0,
              threadGroupSize.width > 0, threadGroupSize.height > 0, threadGroupSize.depth > 0
        else { return nil }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        return cmdBuf
    }

    /// Batch multiple GPU dispatches into a single command buffer (synchronous).
    private func dispatchBatch(_ dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])],
                                threadGroups: MTLSize, threadGroupSize: MTLSize) {
        guard isRunning else { return }
        guard threadGroups.width > 0, threadGroupSize.width > 0 else { return }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        for d in dispatches {
            enc.setComputePipelineState(d.pipeline)
            for (buf, idx) in d.buffers { enc.setBuffer(buf, offset: 0, index: idx) }
            enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if cmdBuf.status == .error {
            let err = cmdBuf.error?.localizedDescription ?? "unknown"
            ActivityLog.shared.error("[Engine] GPU batch error: \(err)", category: .dock)
            isRunning = false
        }
    }

    /// Async batch dispatch — commits work to GPU without blocking CPU.
    /// Returns the command buffer so callers can wait only when they need results.
    @discardableResult
    private func dispatchBatchAsync(_ dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])],
                                     threadGroups: MTLSize, threadGroupSize: MTLSize) -> MTLCommandBuffer? {
        guard isRunning else { return nil }
        guard threadGroups.width > 0, threadGroupSize.width > 0 else { return nil }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }
        for d in dispatches {
            enc.setComputePipelineState(d.pipeline)
            for (buf, idx) in d.buffers { enc.setBuffer(buf, offset: 0, index: idx) }
            enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        enc.endEncoding()
        cmdBuf.commit()
        return cmdBuf
    }

    private func copyPoseBuffer(from source: MTLBuffer, to destination: MTLBuffer, poseCount: Int) {
        let byteCount = poseCount * MemoryLayout<DockPose>.stride
        destination.contents().copyMemory(from: source.contents(), byteCount: byteCount)
    }

    // MARK: - Result Extraction

    // Vina scores are already in kcal/mol — no scaling needed.

    /// Apply rigid-body + torsion transform to recover docked atom positions.
    /// Uses the exact same formulas as the GPU kernels for bit-exact consistency:
    ///   - Rigid body: quatRotate() formula from DockingCompute.metal line 75-79
    ///   - Torsions: Rodrigues rotation from DockingCompute.metal line 114-124
    private func applyPoseTransform(_ pose: DockPose, ligandAtoms: [Atom], centroid: SIMD3<Float>) -> [SIMD3<Float>] {
        let q = SIMD4<Float>(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)
        let u = SIMD3<Float>(q.x, q.y, q.z)
        let s = q.w
        let trans = SIMD3<Float>(pose.translation.x, pose.translation.y, pose.translation.z)

        // Step 1: rigid-body transform — matches GPU quatRotate exactly:
        //   result = 2*dot(u,v)*u + (s*s - dot(u,u))*v + 2*s*cross(u,v)
        var positions = ligandAtoms.map { atom -> SIMD3<Float> in
            let v = atom.position - centroid
            let rotated = 2.0 * simd_dot(u, v) * u + (s * s - simd_dot(u, u)) * v + 2.0 * s * simd_cross(u, v)
            return rotated + trans
        }

        // Step 2: apply torsion rotations — matches GPU Rodrigues exactly:
        //   rotated = v*cos(a) + cross(axis,v)*sin(a) + axis*dot(axis,v)*(1-cos(a))
        if let edgeBuf = torsionEdgeBuffer, let idxBuf = movingIndicesBuffer {
            let edges = edgeBuf.contents().bindMemory(to: TorsionEdge.self, capacity: Int(pose.numTorsions))
            let moving = idxBuf.contents().bindMemory(to: Int32.self, capacity: idxBuf.length / MemoryLayout<Int32>.stride)

            for t in 0..<Int(pose.numTorsions) {
                let angle = withUnsafePointer(to: pose.torsions) {
                    $0.withMemoryRebound(to: Float.self, capacity: 32) { $0[t] }
                }
                if abs(angle) < 1e-6 { continue }

                let edge = edges[t]
                let pivotIdx = Int(edge.atom1)
                let axisIdx = Int(edge.atom2)
                guard pivotIdx < positions.count, axisIdx < positions.count else { continue }

                let pivot = positions[pivotIdx]
                let axis = simd_normalize(positions[axisIdx] - pivot)
                let cosA = cos(angle)
                let sinA = sin(angle)

                for i in 0..<Int(edge.movingCount) {
                    let atomIdx = Int(moving[Int(edge.movingStart) + i])
                    guard atomIdx >= 0, atomIdx < positions.count else { continue }
                    let v = positions[atomIdx] - pivot
                    // Rodrigues rotation (exact GPU formula)
                    let rotated = v * cosA + simd_cross(axis, v) * sinA + axis * simd_dot(axis, v) * (1.0 - cosA)
                    positions[atomIdx] = pivot + rotated
                }
            }
        }

        return positions
    }

    private func extractBestPose(from buffer: MTLBuffer? = nil, ligandAtoms: [Atom], centroid: SIMD3<Float>) -> DockingResult? {
        guard let buffer = buffer ?? populationBuffer else { return nil }
        let poseCount = buffer.length / MemoryLayout<DockPose>.stride
        guard poseCount > 0 else { return nil }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        // Find best VALID pose (finite energy, below sentinel threshold)
        var bestIdx = -1
        var bestE: Float = .infinity
        for i in 0..<poseCount {
            let e = poses[i].energy
            guard e.isFinite, e < 1e9 else { continue }
            if e < bestE { bestE = e; bestIdx = i }
        }
        guard bestIdx >= 0 else { return nil }

        let p = poses[bestIdx]
        let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
        let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)

        let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

        // Vina scores are already in kcal/mol — no kCalScale division needed
        var result = DockingResult(
            id: bestIdx,
            pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p), chiAngles: chiAngles(from: p)),
            energy: p.energy,
            stericEnergy: p.stericEnergy,
            hydrophobicEnergy: p.hydrophobicEnergy,
            hbondEnergy: p.hbondEnergy,
            torsionPenalty: p.torsionPenalty,
            generation: Int(p.generation),
            transformedAtomPositions: transformed
        )
        result.drusinaCorrection = p.drusinaCorrection
        result.constraintPenalty = p.constraintPenalty
        return result
    }

    private func extractAllResults(
        from buffer: MTLBuffer? = nil,
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>,
        idOffset: Int = 0,
        sortByEnergy: Bool = true
    ) -> [DockingResult] {
        guard let buffer = buffer ?? populationBuffer else { return [] }
        let poseCount = buffer.length / MemoryLayout<DockPose>.stride
        guard poseCount > 0 else { return [] }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        var results: [DockingResult] = []
        results.reserveCapacity(poseCount)

        for i in 0..<poseCount {
            let p = poses[i]
            // Skip invalid poses: sentinel energy (1e10 from initialization), NaN, or inf
            guard p.energy.isFinite, p.energy < 1e9 else { continue }

            let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
            let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)
            let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

            // Skip poses with NaN positions (degenerate quaternion/torsion)
            guard transformed.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { continue }

            var r = DockingResult(
                id: results.count + idOffset,
                pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p), chiAngles: chiAngles(from: p)),
                energy: p.energy,
                stericEnergy: p.stericEnergy,
                hydrophobicEnergy: p.hydrophobicEnergy,
                hbondEnergy: p.hbondEnergy,
                torsionPenalty: p.torsionPenalty,
                generation: Int(p.generation),
                transformedAtomPositions: transformed
            )
            r.drusinaCorrection = p.drusinaCorrection
            r.constraintPenalty = p.constraintPenalty
            results.append(r)
        }
        return sortByEnergy ? results.sorted { $0.energy < $1.energy } : results
    }

    // MARK: - RMSD Clustering

    private func clusterPoses(_ results: [DockingResult]) async -> [DockingResult] {
        guard !results.isEmpty else { return [] }
        let threshold: Float = 2.0
        var out = results
        let n = out.count

        // Precompute full pairwise RMSD matrix (GPU if available, CPU fallback)
        let rmsdMatrix = await computeRMSDMatrixGPU(out) ?? computeRMSDMatrixCPU(out)

        // Greedy leader-based clustering using precomputed matrix
        var clusterID = 0
        for i in 0..<n {
            guard out[i].clusterID == -1 else { continue }
            out[i].clusterID = clusterID
            out[i].clusterRank = 0
            var rank = 1
            for j in (i+1)..<n {
                guard out[j].clusterID == -1 else { continue }
                let idx = i * n - i * (i + 1) / 2 + j - i - 1
                if rmsdMatrix[idx] < threshold {
                    out[j].clusterID = clusterID
                    out[j].clusterRank = rank
                    rank += 1
                }
            }
            clusterID += 1
        }
        return out
    }

    private func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        let s = zip(a, b).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
        return sqrt(s / Float(a.count))
    }

    /// Compute pairwise RMSD matrix on GPU using Metal compute.
    private func computeRMSDMatrixGPU(_ results: [DockingResult]) async -> [Float]? {
        guard let pipeline = pairwiseRMSDPipeline,
              let first = results.first,
              !first.transformedAtomPositions.isEmpty else { return nil }

        let n = results.count
        let numAtoms = first.transformedAtomPositions.count
        guard n > 1 else { return [] }

        // Flatten all pose positions into contiguous array
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n * numAtoms)
        for r in results {
            guard r.transformedAtomPositions.count == numAtoms else { return nil }
            positions.append(contentsOf: r.transformedAtomPositions)
        }

        let matrixSize = n * (n - 1) / 2
        var params = RMSDParams(numPoses: UInt32(n), numAtoms: UInt32(numAtoms), _pad0: 0, _pad1: 0)

        guard let posBuffer = device.makeBuffer(
                    bytes: &positions,
                    length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
                    options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(
                    length: matrixSize * MemoryLayout<Float>.stride,
                    options: .storageModeShared),
              let paramsBuffer = device.makeBuffer(
                    bytes: &params,
                    length: MemoryLayout<RMSDParams>.stride,
                    options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else { return nil }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(posBuffer, offset: 0, index: 0)
        enc.setBuffer(matrixBuffer, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)

        // 1D dispatch over upper-triangular pairs only (no wasted threads)
        let totalPairs = n * (n - 1) / 2
        let threadWidth = 256
        let tgSize = MTLSize(width: threadWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalPairs + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        await withCheckedContinuation { continuation in
            cmdBuf.addCompletedHandler { _ in
                continuation.resume()
            }
            cmdBuf.commit()
        }

        let ptr = matrixBuffer.contents().bindMemory(to: Float.self, capacity: matrixSize)
        return Array(UnsafeBufferPointer(start: ptr, count: matrixSize))
    }

    /// CPU fallback for pairwise RMSD matrix.
    private func computeRMSDMatrixCPU(_ results: [DockingResult]) -> [Float] {
        let n = results.count
        let matrixSize = n * (n - 1) / 2
        var matrix = [Float](repeating: 0, count: matrixSize)
        for i in 0..<n {
            for j in (i+1)..<n {
                let idx = i * n - i * (i + 1) / 2 + j - i - 1
                matrix[idx] = rmsd(results[i].transformedAtomPositions, results[j].transformedAtomPositions)
            }
        }
        return matrix
    }

    private func ligandWithDockingCharges(_ ligand: Molecule) -> Molecule {
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

    private func electrostaticCharge(for atom: Atom) -> Float {
        abs(atom.charge) > 1e-4 ? atom.charge : Float(atom.formalCharge)
    }

    private func buildTorsionTree(for ligand: Molecule, heavyBonds: [Bond]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        if let tree = RDKitBridge.buildTorsionTreeMolBlock(molBlock), !tree.isEmpty {
            return tree
        }

        // Prefer the explicit SMILES property (set when molecule comes from SMILES input).
        // Falls back to title only if it looks like a valid SMILES string (contains ring digits
        // or parentheses, and no spaces — distinguishes SMILES from PDB titles).
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

    private func buildGraphTorsionTree(atomCount: Int, bonds: [Bond], atoms: [Atom]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        guard atomCount > 1 else { return [] }

        var adjacency = Array(repeating: [Int](), count: atomCount)
        for bond in bonds {
            guard bond.atomIndex1 < atomCount, bond.atomIndex2 < atomCount else { continue }
            adjacency[bond.atomIndex1].append(bond.atomIndex2)
            adjacency[bond.atomIndex2].append(bond.atomIndex1)
        }

        func hasAlternatePath(from start: Int, to target: Int, excluding edge: (Int, Int)) -> Bool {
            var visited: Set<Int> = [start]
            var queue = [start]
            while !queue.isEmpty {
                let current = queue.removeFirst()
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
            var queue = [start]
            while !queue.isEmpty {
                let current = queue.removeFirst()
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
            var queue = [root]
            bfsOrder[root] = orderCounter
            orderCounter += 1
            while !queue.isEmpty {
                let current = queue.removeFirst()
                for next in adjacency[current] where bfsOrder[next] == Int.max {
                    bfsOrder[next] = orderCounter
                    orderCounter += 1
                    queue.append(next)
                }
            }
        }

        // Build bond-order lookup for amide detection
        var bondOrderMap: [Int: [Int: BondOrder]] = [:]
        for bond in bonds {
            bondOrderMap[bond.atomIndex1, default: [:]][bond.atomIndex2] = bond.order
            bondOrderMap[bond.atomIndex2, default: [:]][bond.atomIndex1] = bond.order
        }

        // Check if a single bond A-B is an amide bond (C-N where C=O or C=S)
        func isAmideBond(_ a: Int, _ b: Int) -> Bool {
            guard a < atoms.count, b < atoms.count else { return false }
            let elemA = atoms[a].element
            let elemB = atoms[b].element
            // Determine which is C and which is N
            let carbonIdx: Int
            if elemA == .C && elemB == .N {
                carbonIdx = a
            } else if elemA == .N && elemB == .C {
                carbonIdx = b
            } else {
                return false
            }
            // Check if the carbon has a double bond to O or S
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

            // Skip amide bonds (C-N where C=O/S) — partial double-bond character
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

    // MARK: - Force Field Parameters

    // MARK: - Docking Diagnostics

    /// Compute quality diagnostics for a set of docking results.
    func computeDiagnostics(
        results: [DockingResult],
        ligandAtoms: [Atom],
        heavyBonds: [Bond]
    ) -> DockingDiagnostics {
        let gridOrigin = SIMD3<Float>(gridParams.origin.x, gridParams.origin.y, gridParams.origin.z)
        let gridEnd = gridOrigin + SIMD3<Float>(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)) * gridParams.spacing
        let gridCenter = (gridOrigin + gridEnd) * 0.5

        let validResults = results.filter { $0.energy.isFinite && $0.energy < 1e9 }
        let energies = validResults.map(\.energy)

        // Energy statistics
        let minE = energies.min() ?? .infinity
        let maxE = energies.max() ?? -.infinity
        let meanE = energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)
        let variance = energies.isEmpty ? 0 : energies.map { ($0 - meanE) * ($0 - meanE) }.reduce(0, +) / Float(energies.count)
        let stddevE = sqrt(variance)

        // Pose location analysis
        var insideGrid = 0
        var outsideGrid = 0
        var centroidDistances: [Float] = []
        var minProteinDistances: [Float] = []  // closest protein atom per pose

        for r in validResults {
            let positions = r.transformedAtomPositions
            let allInside = positions.allSatisfy { p in
                p.x >= gridOrigin.x && p.x <= gridEnd.x &&
                p.y >= gridOrigin.y && p.y <= gridEnd.y &&
                p.z >= gridOrigin.z && p.z <= gridEnd.z
            }
            if allInside { insideGrid += 1 } else { outsideGrid += 1 }

            // Centroid distance to grid center
            if !positions.isEmpty {
                let centroid = positions.reduce(.zero, +) / Float(positions.count)
                centroidDistances.append(simd_distance(centroid, gridCenter))

                // Minimum distance from any ligand atom to any protein atom
                if !proteinAtoms.isEmpty {
                    var minDist: Float = .infinity
                    for lp in positions {
                        for pa in proteinAtoms {
                            let d = simd_distance(lp, pa.position)
                            if d < minDist { minDist = d }
                        }
                    }
                    minProteinDistances.append(minDist)
                }
            }
        }

        let meanCentroidDist = centroidDistances.isEmpty ? 0 : centroidDistances.reduce(0, +) / Float(centroidDistances.count)
        let meanProteinDist = minProteinDistances.isEmpty ? 0 : minProteinDistances.reduce(0, +) / Float(minProteinDistances.count)

        // Poses making protein contact (min distance < 4 Å)
        let contactPoses = minProteinDistances.filter { $0 < 4.0 }.count

        // Bond length analysis (top 10 poses)
        var bondDeviations: [Float] = []
        for r in validResults.prefix(10) {
            let positions = r.transformedAtomPositions
            for b in heavyBonds {
                guard b.atomIndex1 < ligandAtoms.count, b.atomIndex2 < ligandAtoms.count,
                      b.atomIndex1 < positions.count, b.atomIndex2 < positions.count else { continue }
                let orig = simd_distance(ligandAtoms[b.atomIndex1].position, ligandAtoms[b.atomIndex2].position)
                guard orig > 0.01 else { continue }
                let docked = simd_distance(positions[b.atomIndex1], positions[b.atomIndex2])
                guard docked.isFinite else { continue }
                bondDeviations.append(abs(docked - orig))
            }
        }
        let meanBondDev = bondDeviations.isEmpty ? 0 : bondDeviations.reduce(0, +) / Float(bondDeviations.count)
        let maxBondDev = bondDeviations.max() ?? 0

        // Cluster count
        let clusterCount = Set(validResults.map(\.clusterID)).count

        // Spatial exploration: how spread out are the pose centroids?
        let centroidSpread: Float
        if centroidDistances.count >= 2 {
            let meanCD = centroidDistances.reduce(0, +) / Float(centroidDistances.count)
            let cdVar = centroidDistances.map { ($0 - meanCD) * ($0 - meanCD) }.reduce(0, +) / Float(centroidDistances.count)
            centroidSpread = sqrt(cdVar)
        } else {
            centroidSpread = 0
        }

        return DockingDiagnostics(
            totalPopulation: results.count,
            validPoses: validResults.count,
            invalidPoses: results.count - validResults.count,
            posesInsideGrid: insideGrid,
            posesOutsideGrid: outsideGrid,
            posesWithProteinContact: contactPoses,
            meanLigandProteinDistance: meanProteinDist,
            meanCentroidToGridCenter: meanCentroidDist,
            centroidSpread: centroidSpread,
            minEnergy: minE,
            maxEnergy: maxE,
            meanEnergy: meanE,
            energyStdDev: stddevE,
            clusterCount: clusterCount,
            meanBondLengthDeviation: meanBondDev,
            maxBondLengthDeviation: maxBondDev,
            gridDimensions: SIMD3(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)),
            gridSpacing: gridParams.spacing,
            gridBoxSize: gridEnd - gridOrigin
        )
    }
}

/// Docking quality metrics computed after a docking run.
struct DockingDiagnostics: Sendable {
    // Population
    var totalPopulation: Int
    var validPoses: Int         // energy finite and < 1e9
    var invalidPoses: Int       // sentinel, NaN, or inf

    // Spatial quality
    var posesInsideGrid: Int    // all atoms within grid bounds
    var posesOutsideGrid: Int   // any atom outside grid
    var posesWithProteinContact: Int // min ligand-protein distance < 4 Å
    var meanLigandProteinDistance: Float // avg min distance to protein per pose
    var meanCentroidToGridCenter: Float
    var centroidSpread: Float   // stddev of centroid distances (exploration metric)

    // Energy
    var minEnergy: Float
    var maxEnergy: Float
    var meanEnergy: Float
    var energyStdDev: Float

    // Clustering
    var clusterCount: Int

    // Geometry preservation
    var meanBondLengthDeviation: Float  // avg Å deviation from input
    var maxBondLengthDeviation: Float

    // Grid info
    var gridDimensions: SIMD3<Float>    // grid points per axis
    var gridSpacing: Float
    var gridBoxSize: SIMD3<Float>       // Å per axis

    /// Human-readable summary
    var summary: String {
        """
        Docking Diagnostics:
          Population: \(totalPopulation) total, \(validPoses) valid, \(invalidPoses) invalid
          Grid: \(String(format: "%.0f×%.0f×%.0f", gridDimensions.x, gridDimensions.y, gridDimensions.z)) pts (\(String(format: "%.0f×%.0f×%.0f", gridBoxSize.x, gridBoxSize.y, gridBoxSize.z)) Å at \(gridSpacing) spacing)
          Poses inside grid: \(posesInsideGrid)/\(validPoses), outside: \(posesOutsideGrid)
          Protein contacts: \(posesWithProteinContact)/\(validPoses) (mean dist \(String(format: "%.1f", meanLigandProteinDistance)) Å)
          Centroid spread: \(String(format: "%.1f", centroidSpread)) Å (exploration diversity)
          Energy: min=\(String(format: "%.1f", minEnergy)) max=\(String(format: "%.1f", maxEnergy)) mean=\(String(format: "%.1f", meanEnergy)) σ=\(String(format: "%.1f", energyStdDev))
          Clusters: \(clusterCount)
          Bond preservation: mean=\(String(format: "%.3f", meanBondLengthDeviation))Å max=\(String(format: "%.3f", maxBondLengthDeviation))Å
        """
    }
}

// MARK: - Interaction Detection

enum InteractionDetector {

    // MARK: - Aromatic Ring Detection

    /// Detect aromatic rings from protein residues (known aromatic sidechains).
    /// Returns ring centroids and approximate normal vectors.
    struct AromaticRing: Sendable {
        let centroid: SIMD3<Float>
        let normal: SIMD3<Float>
        let atomIndices: [Int]  // indices into the atoms array
    }

    /// Known aromatic sidechain atoms for standard amino acids
    private static let aromaticResidueAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  // both rings
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    ]

    static func detectAromaticRings(
        atoms: [Atom],
        positions: [SIMD3<Float>]? = nil,
        bonds: [Bond]? = nil
    ) -> [AromaticRing] {
        var rings: [AromaticRing] = []

        // 1. Standard protein aromatic residues (PHE, TYR, TRP, HIS)
        var residueAtoms: [String: [(Int, Atom)]] = [:]
        for (i, atom) in atoms.enumerated() {
            let key = "\(atom.chainID)_\(atom.residueSeq)_\(atom.residueName)"
            residueAtoms[key, default: []].append((i, atom))
        }

        for (_, atomList) in residueAtoms {
            guard let resName = atomList.first?.1.residueName,
                  let targetNames = aromaticResidueAtoms[resName] else { continue }

            let ringAtoms = atomList.filter { targetNames.contains($0.1.name.trimmingCharacters(in: .whitespaces)) }
            guard ringAtoms.count >= 5 else { continue }

            if let ring = buildRing(indices: ringAtoms.map(\.0), atoms: atoms, positions: positions) {
                rings.append(ring)
            }
        }

        // 2. Graph-based ring detection for non-standard residues (ligands)
        //    Find 5- and 6-membered rings from bond connectivity, then check planarity.
        if let bonds = bonds {
            let graphRings = detectRingsFromBonds(atoms: atoms, bonds: bonds, positions: positions)
            rings.append(contentsOf: graphRings)
        }

        return rings
    }

    /// Build an AromaticRing from known atom indices.
    private static func buildRing(
        indices: [Int], atoms: [Atom], positions: [SIMD3<Float>]?
    ) -> AromaticRing? {
        let ringPositions = indices.map { positions?[$0] ?? atoms[$0].position }
        let centroid = ringPositions.reduce(.zero, +) / Float(ringPositions.count)

        var normal = SIMD3<Float>(0, 1, 0)
        if ringPositions.count >= 3 {
            let v1 = ringPositions[1] - ringPositions[0]
            let v2 = ringPositions[2] - ringPositions[0]
            let n = simd_cross(v1, v2)
            let len = simd_length(n)
            if len > 1e-6 { normal = n / len }
        }
        return AromaticRing(centroid: centroid, normal: normal, atomIndices: indices)
    }

    /// Detect aromatic rings from bond connectivity (for ligands/non-standard residues).
    /// Finds all 5- and 6-membered rings, then filters by planarity and element composition.
    private static func detectRingsFromBonds(
        atoms: [Atom], bonds: [Bond], positions: [SIMD3<Float>]?
    ) -> [AromaticRing] {
        let n = atoms.count
        guard n > 4 else { return [] }

        // Build adjacency list (only C/N/O/S — skip H and metals)
        let aromaticElements: Set<Element> = [.C, .N, .O, .S]
        var adj: [[Int]] = Array(repeating: [], count: n)
        for bond in bonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < n, b < n else { continue }
            guard aromaticElements.contains(atoms[a].element),
                  aromaticElements.contains(atoms[b].element) else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        // Find all simple cycles of length 5 or 6 using DFS from each atom.
        // To avoid duplicates, only start from the smallest-index atom in the cycle.
        var foundRings: Set<[Int]> = []

        for start in 0..<n {
            guard aromaticElements.contains(atoms[start].element) else { continue }
            // BFS/DFS limited depth search for cycles back to start
            findCycles(start: start, adj: adj, maxLen: 6, found: &foundRings)
        }

        // Filter by planarity: all atoms within 0.5 Å of the best-fit plane
        var result: [AromaticRing] = []
        for ringIndices in foundRings {
            let ringPos = ringIndices.map { positions?[$0] ?? atoms[$0].position }

            // Check planarity: compute normal from first 3 atoms, then check all are near the plane
            guard ringPos.count >= 5 else { continue }
            let v1 = ringPos[1] - ringPos[0]
            let v2 = ringPos[2] - ringPos[0]
            let normal = simd_cross(v1, v2)
            let normalLen = simd_length(normal)
            guard normalLen > 1e-6 else { continue }
            let n = normal / normalLen
            let centroid = ringPos.reduce(.zero, +) / Float(ringPos.count)

            // All atoms should be within 0.5 Å of the plane through centroid
            var planar = true
            for p in ringPos {
                let dist = abs(simd_dot(p - centroid, n))
                if dist > 0.5 { planar = false; break }
            }
            guard planar else { continue }

            // Must contain at least 4 carbon/nitrogen atoms (skip saturated rings)
            let cnCount = ringIndices.filter { atoms[$0].element == .C || atoms[$0].element == .N }.count
            guard cnCount >= ringIndices.count - 1 else { continue }

            result.append(AromaticRing(centroid: centroid, normal: n, atomIndices: ringIndices))
        }
        return result
    }

    /// Find simple cycles of length 5-6 that include `start` as the smallest index.
    private static func findCycles(
        start: Int, adj: [[Int]], maxLen: Int, found: inout Set<[Int]>
    ) {
        // DFS with path tracking, looking for paths that return to start
        var stack: [(node: Int, path: [Int])] = [(start, [start])]

        while !stack.isEmpty {
            let (node, path) = stack.removeLast()
            guard path.count <= maxLen else { continue }

            for neighbor in adj[node] {
                if neighbor == start && (path.count == 5 || path.count == 6) {
                    // Found a cycle — normalize so smallest index is first
                    let ring = path.sorted()
                    if ring[0] == start {  // only record if start is the smallest
                        found.insert(ring)
                    }
                } else if neighbor > start && !path.contains(neighbor) && path.count < maxLen {
                    stack.append((neighbor, path + [neighbor]))
                }
            }
        }
    }

    // MARK: - Full Interaction Detection

    static func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandBonds: [Bond] = []
    ) -> [MolecularInteraction] {
        var result: [MolecularInteraction] = []
        var idCounter = 0

        // Charged residue atoms for salt bridges
        let positiveResAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE"]  // Lys, Arg
        let negativeResAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]  // Asp, Glu
        let metals: Set<Element> = [.Fe, .Zn, .Ca, .Mg, .Mn, .Cu]

        // ---- GPU path: metal coord, salt bridges, H-bonds, halogen, hydrophobic ----
        if let gpu = InteractionDetectorGPU.shared {
            let gpuResults = gpu.detect(
                ligandAtoms: ligandAtoms,
                ligandPositions: ligandPositions,
                proteinAtoms: proteinAtoms,
                positiveResAtoms: positiveResAtoms,
                negativeResAtoms: negativeResAtoms,
                metals: metals
            )
            for gi in gpuResults {
                result.append(MolecularInteraction(
                    id: idCounter,
                    ligandAtomIndex: Int(gi.ligandAtomIndex),
                    proteinAtomIndex: Int(gi.proteinAtomIndex),
                    type: MolecularInteraction.InteractionType(rawValue: Int(gi.type)) ?? .hbond,
                    distance: gi.distance,
                    ligandPosition: gi.ligandPosition,
                    proteinPosition: gi.proteinPosition
                ))
                idCounter += 1
            }
        } else {
            // ---- CPU fallback with spatial grid ----
            var ligandHasStrongInteraction: Set<Int> = []
            let cellSize: Float = 6.0
            let invCell: Float = 1.0 / cellSize
            struct CellKey: Hashable { let x, y, z: Int }
            var proteinGrid: [CellKey: [Int]] = [:]
            proteinGrid.reserveCapacity(proteinAtoms.count / 3)
            for (pi, protAtom) in proteinAtoms.enumerated() {
                let ck = CellKey(x: Int(floor(protAtom.position.x * invCell)),
                                 y: Int(floor(protAtom.position.y * invCell)),
                                 z: Int(floor(protAtom.position.z * invCell)))
                proteinGrid[ck, default: []].append(pi)
            }

            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count else { continue }
                let lp = ligandPositions[li]
                let lcx = Int(floor(lp.x * invCell))
                let lcy = Int(floor(lp.y * invCell))
                let lcz = Int(floor(lp.z * invCell))

                for ndx in -1...1 { for ndy in -1...1 { for ndz in -1...1 {
                let nkey = CellKey(x: lcx + ndx, y: lcy + ndy, z: lcz + ndz)
                guard let cellIndices = proteinGrid[nkey] else { continue }
                for pi in cellIndices {
                    let protAtom = proteinAtoms[pi]
                    let d = simd_distance(lp, protAtom.position)
                    guard d < 6.0 else { continue }
                    let protName = protAtom.name.trimmingCharacters(in: .whitespaces)

                    if d < 2.8 {
                        let ligCoord = ligAtom.element == .N || ligAtom.element == .O || ligAtom.element == .S
                        let protMetal = metals.contains(protAtom.element)
                        let ligMetal = metals.contains(ligAtom.element)
                        let protCoord = protAtom.element == .N || protAtom.element == .O || protAtom.element == .S
                        if (protMetal && ligCoord) || (ligMetal && protCoord) {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .metalCoord, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d < 4.0 {
                        let protPositive = positiveResAtoms.contains(protName)
                        let protNegative = negativeResAtoms.contains(protName)
                        if (protPositive && ligAtom.formalCharge < 0) || (protNegative && ligAtom.formalCharge > 0) {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .saltBridge, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d >= 2.2 && d <= 3.5 {
                        let ligDA = ligAtom.element == .N || ligAtom.element == .O
                        let proDA = protAtom.element == .N || protAtom.element == .O
                        if ligDA && proDA {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .hbond, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li); continue
                        }
                    }
                    if d >= 2.5 && d <= 3.5 {
                        let halogen = ligAtom.element == .F || ligAtom.element == .Cl || ligAtom.element == .Br
                        let acceptor = protAtom.element == .N || protAtom.element == .O
                        if halogen && acceptor {
                            result.append(MolecularInteraction(
                                id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                                type: .halogen, distance: d,
                                ligandPosition: lp, proteinPosition: protAtom.position))
                            idCounter += 1; ligandHasStrongInteraction.insert(li)
                        }
                    }
                }
                }}} // spatial grid
            }

            // CPU hydrophobic contacts
            let invCellH = invCell
            var hydroCount: [Int: Int] = [:]
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count, !ligandHasStrongInteraction.contains(li) else { continue }
                guard ligAtom.element == .C || ligAtom.element == .S else { continue }
                let lp = ligandPositions[li]
                let cx = Int(floor(lp.x * invCellH)), cy = Int(floor(lp.y * invCellH)), cz = Int(floor(lp.z * invCellH))
                for ndx in -1...1 { for ndy in -1...1 { for ndz in -1...1 {
                guard (hydroCount[li, default: 0]) < 3 else { continue }
                let hk = CellKey(x: cx + ndx, y: cy + ndy, z: cz + ndz)
                guard let hci = proteinGrid[hk] else { continue }
                for pi in hci {
                    let pa = proteinAtoms[pi]
                    guard pa.element == .C || pa.element == .S else { continue }
                    let d = simd_distance(lp, pa.position)
                    guard d >= 3.3 && d <= 4.5, (hydroCount[li, default: 0]) < 3 else { continue }
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                        type: .hydrophobic, distance: d,
                        ligandPosition: lp, proteinPosition: pa.position))
                    idCounter += 1; hydroCount[li, default: 0] += 1
                }
                }}}
            }
        }

        // ---- π-π stacking: ring centroid distance 3.3-5.5 Å (always CPU — few rings) ----
        let proteinRings = detectAromaticRings(atoms: proteinAtoms)
        let ligandRings = detectAromaticRings(
            atoms: ligandAtoms, positions: ligandPositions,
            bonds: ligandBonds.isEmpty ? nil : ligandBonds
        )

        for ligRing in ligandRings {
            for protRing in proteinRings {
                let d = simd_distance(ligRing.centroid, protRing.centroid)
                guard d >= 3.3 && d <= 5.5 else { continue }
                let dotN = abs(simd_dot(ligRing.normal, protRing.normal))
                let isFaceToFace = dotN > 0.85 && d < 4.2
                let isEdgeToFace = dotN < 0.5 && d >= 4.0 && d <= 5.5
                if isFaceToFace || isEdgeToFace {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligRing.atomIndices.first ?? 0,
                        proteinAtomIndex: protRing.atomIndices.first ?? 0,
                        type: .piStack, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        // ---- π-cation (always CPU — few rings) ----
        for ligRing in ligandRings {
            for (pi, protAtom) in proteinAtoms.enumerated() {
                let protName = protAtom.name.trimmingCharacters(in: .whitespaces)
                guard positiveResAtoms.contains(protName) || protAtom.formalCharge > 0 else { continue }
                let d = simd_distance(ligRing.centroid, protAtom.position)
                guard d < 6.0 else { continue }
                let toAtom = simd_normalize(protAtom.position - ligRing.centroid)
                if abs(simd_dot(toAtom, ligRing.normal)) > 0.5 {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligRing.atomIndices.first ?? 0,
                        proteinAtomIndex: pi, type: .piCation, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protAtom.position))
                    idCounter += 1
                }
            }
        }
        for protRing in proteinRings {
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count else { continue }
                guard ligAtom.formalCharge > 0 || ligAtom.element == .N else { continue }
                let lp = ligandPositions[li]
                let d = simd_distance(protRing.centroid, lp)
                guard d < 6.0 else { continue }
                let toAtom = simd_normalize(lp - protRing.centroid)
                if abs(simd_dot(toAtom, protRing.normal)) > 0.5 {
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li,
                        proteinAtomIndex: protRing.atomIndices.first ?? 0,
                        type: .piCation, distance: d,
                        ligandPosition: lp, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        return result
    }
}

// MARK: - GPU Interaction Detection Accelerator

/// Metal-accelerated interaction detection. One thread per ligand atom,
/// each checking all protein atoms with atomic append to output buffer.
/// Handles: metal coord, salt bridges, H-bonds, halogen bonds, hydrophobic.
/// π-π stacking and π-cation remain on CPU (few rings, geometry-dependent).
private final class InteractionDetectorGPU {
    nonisolated(unsafe) static let shared: InteractionDetectorGPU? = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "detectInteractions")
        else { return nil }

        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            return InteractionDetectorGPU(device: device, commandQueue: commandQueue, pipeline: pipeline)
        } catch {
            return nil
        }
    }()

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let maxInteractions = 2048

    private init(device: MTLDevice, commandQueue: MTLCommandQueue, pipeline: MTLComputePipelineState) {
        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
    }

    /// Build element/property flags for a single atom.
    private func atomFlags(for atom: Atom, positiveResAtoms: Set<String>, negativeResAtoms: Set<String>, metals: Set<Element>) -> UInt32 {
        var flags: UInt32 = 0
        switch atom.element {
        case .N:  flags |= UInt32(IDET_FLAG_N)
        case .O:  flags |= UInt32(IDET_FLAG_O)
        case .S:  flags |= UInt32(IDET_FLAG_S)
        case .C:  flags |= UInt32(IDET_FLAG_C)
        case .F:  flags |= UInt32(IDET_FLAG_F) | UInt32(IDET_FLAG_HALOGEN)
        case .Cl: flags |= UInt32(IDET_FLAG_CL) | UInt32(IDET_FLAG_HALOGEN)
        case .Br: flags |= UInt32(IDET_FLAG_BR) | UInt32(IDET_FLAG_HALOGEN)
        default:  break
        }
        if metals.contains(atom.element) { flags |= UInt32(IDET_FLAG_METAL) }
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        if positiveResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_POS_RES) }
        if negativeResAtoms.contains(name) { flags |= UInt32(IDET_FLAG_NEG_RES) }
        return flags
    }

    func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        positiveResAtoms: Set<String>,
        negativeResAtoms: Set<String>,
        metals: Set<Element>
    ) -> [GPUInteraction] {
        let nLig = ligandAtoms.count
        let nProt = proteinAtoms.count
        guard nLig > 0, nProt > 0, ligandPositions.count >= nLig else { return [] }

        // Pack protein atoms
        var protGPU = proteinAtoms.map { atom -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom, positiveResAtoms: positiveResAtoms, negativeResAtoms: negativeResAtoms, metals: metals),
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        // Pack ligand atoms
        var ligGPU = ligandAtoms.map { atom -> InteractionAtomGPU in
            InteractionAtomGPU(
                position: atom.position,
                flags: atomFlags(for: atom, positiveResAtoms: positiveResAtoms, negativeResAtoms: negativeResAtoms, metals: metals),
                formalCharge: Int32(atom.formalCharge),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        var positions = Array(ligandPositions.prefix(nLig))
        var counter: UInt32 = 0
        var params = InteractionDetectParams(
            numLigandAtoms: UInt32(nLig),
            numProteinAtoms: UInt32(nProt),
            maxInteractions: UInt32(maxInteractions),
            _pad0: 0
        )

        guard let protBuffer = device.makeBuffer(bytes: &protGPU, length: nProt * MemoryLayout<InteractionAtomGPU>.stride, options: .storageModeShared),
              let ligBuffer = device.makeBuffer(bytes: &ligGPU, length: nLig * MemoryLayout<InteractionAtomGPU>.stride, options: .storageModeShared),
              let posBuffer = device.makeBuffer(bytes: &positions, length: nLig * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: maxInteractions * MemoryLayout<GPUInteraction>.stride, options: .storageModeShared),
              let ctrBuffer = device.makeBuffer(bytes: &counter, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<InteractionDetectParams>.stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else { return [] }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(protBuffer, offset: 0, index: 0)
        enc.setBuffer(ligBuffer, offset: 0, index: 1)
        enc.setBuffer(posBuffer, offset: 0, index: 2)
        enc.setBuffer(outBuffer, offset: 0, index: 3)
        enc.setBuffer(ctrBuffer, offset: 0, index: 4)
        enc.setBuffer(paramBuffer, offset: 0, index: 5)

        let tgSize = MTLSize(width: min(nLig, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (nLig + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let count = min(Int(ctrBuffer.contents().load(as: UInt32.self)), maxInteractions)
        guard count > 0 else { return [] }

        let ptr = outBuffer.contents().bindMemory(to: GPUInteraction.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}

import Foundation
@preconcurrency import MetalKit
import simd

// MARK: - Docking Configuration

struct DockingConfig: Sendable {
    // Population and search
    var populationSize: Int = 300
    var numRuns: Int = 3             // independent Monte Carlo trajectory batches
    var generationsPerRun: Int = 300 // Monte Carlo steps per run
    var gridSpacing: Float = 0.375

    // Search operators
    var mutationRate: Float = 0.10
    var crossoverRate: Float = 0.75
    var translationStep: Float = 2.0 // Angstroms, aligned with Vina mutation amplitude
    var rotationStep: Float = 0.3    // radians (~17°)
    var torsionStep: Float = 0.8     // radians (~46°) — large enough to escape tangled conformers
    var mcTemperature: Float = 1.5   // kcal/mol, matches Vina's default Metropolis temperature
    var explicitRerankTopClusters: Int = 12 // top basin representatives rescored against explicit receptor atoms
    var explicitRerankVariantsPerCluster: Int = 4 // seeded local refinement around each top basin representative
    var explicitRerankLocalSearchSteps: Int = 20 // short second-pass refinement on rerank seeds

    // Local search (Vina-like basin hopping: refine every MC step by default)
    var localSearchFrequency: Int = 3   // every N generations
    var localSearchSteps: Int = 20      // gradient descent steps per refinement
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
    var explorationPhaseRatio: Float = 0.55  // first 55% of generations use broader search
    var explorationTranslationStep: Float = 5.0  // wider initial translation (vs 2.0 during refinement)
    var explorationRotationStep: Float = 0.8     // wider initial rotation (vs 0.3)
    var explorationMutationRate: Float = 0.25    // higher mutation during exploration
    var explorationLocalSearchFrequency: Int = 3  // during exploration, LS every 3rd gen (Lamarckian evolution needs frequent LS)

    // Post-docking refinement
    var gfn2Refinement: GFN2RefinementConfig = GFN2RefinementConfig()
    var useAFv4Rescore: Bool = false  // DruseAF v4 PGN neural rescoring of top poses

    // Search method
    var searchMethod: SearchMethod = .genetic

    // Fragment-Based Docking config
    var fragment: FragmentDockingConfig = FragmentDockingConfig()

    // Diffusion-Guided Docking config
    var diffusion: DiffusionDockingConfig = DiffusionDockingConfig()

    // Parallel Tempering config
    var replicaExchange: ParallelTemperingConfig = ParallelTemperingConfig()

    // Ensemble multi-start docking: dock all qualifying forms × conformers
    var ensemble: EnsembleDockingConfig = EnsembleDockingConfig()

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

        // --- Population scales with ligand size and flexibility ---
        // More atoms + torsions = larger conformational space = need more population diversity
        let torsionFactor = max(1.0, Float(ligandRotatableBonds) / 5.0)
        let sizeFactor = max(1.0, Float(ligandHeavyAtoms) / 20.0)
        let basePop = 200
        config.populationSize = min(600, max(150, Int(Float(basePop) * sqrt(torsionFactor * sizeFactor))))

        // --- Generations scale with pocket size, buriedness, and ligand complexity ---
        let buriednessFactor = max(0.6, 1.5 - pocketBuriedness)
        let volumeFactor = max(0.8, min(2.0, pocketVolume / 800.0))
        let baseGen = 200
        config.generationsPerRun = min(500, max(120, Int(Float(baseGen) * buriednessFactor * volumeFactor * sqrt(sizeFactor))))

        // --- Runs scale with overall difficulty ---
        // Independent restarts are the most effective way to escape local minima
        let proteinSizeFactor: Float = proteinAtomCount > 5000 ? 1.5 : (proteinAtomCount > 3000 ? 1.2 : 1.0)
        let difficultyScore = torsionFactor * buriednessFactor * proteinSizeFactor * sqrt(sizeFactor)
        config.numRuns = min(10, max(2, Int(difficultyScore)))

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

        // --- Search method: auto-select based on flexibility ---
        if ligandRotatableBonds > 10 {
            config.searchMethod = .fragmentBased
        } else if ligandRotatableBonds > 6 {
            config.searchMethod = .parallelTempering
        } else {
            config.searchMethod = .genetic
        }

        return config
    }
}

// MARK: - Ensemble Multi-Start Docking Config

struct EnsembleDockingConfig: Sendable {
    /// Enable multi-start docking across chemical forms and conformers.
    var enabled: Bool = false

    /// Minimum Boltzmann population fraction to include a form (0.0–1.0).
    /// Forms below this cutoff are excluded. E.g., 0.10 = skip forms < 10%.
    var populationCutoff: Double = 0.05

    /// Maximum conformers per qualifying form to use as GA starting geometries.
    /// 0 = use all available conformers.
    var maxConformersPerForm: Int = 3

    /// Whether to weight/rank results by form population.
    /// When true, composite score = docking_energy - RT*ln(population).
    var populationWeighting: Bool = true
}

// MARK: - Fragment-Based Docking Config

struct FragmentDockingConfig: Sendable {
    /// Number of surviving partial poses carried forward at each fragment growth step.
    var beamWidth: Int = 32
    /// Number of initial anchor placements to sample in the binding site.
    var anchorSamplingCount: Int = 256
    /// Local search steps for refining anchor placements before growth.
    var anchorLocalSearchSteps: Int = 20
    /// Energy threshold (kcal/mol) above best partial pose for pruning during growth.
    var growthPruneThreshold: Float = 10.0
    /// Number of torsion angle samples per connecting bond during fragment growth.
    var torsionSamples: Int = 36
    /// SMARTS pattern for enforced scaffold (nil = auto-detect largest fragment).
    var scaffoldSMARTS: String? = nil
    /// Scaffold enforcement mode.
    var scaffoldMode: ScaffoldMode = .auto

    enum ScaffoldMode: String, CaseIterable, Sendable {
        case auto   = "Auto"
        case manual = "Manual"
    }
}

// MARK: - Diffusion-Guided Docking Config

struct DiffusionDockingConfig: Sendable {
    /// Number of reverse diffusion denoising steps.
    var numDenoisingSteps: Int = 50
    /// Number of poses generated in parallel via the diffusion process.
    var numParallelPoses: Int = 128
    /// Noise variance schedule type.
    var noiseSchedule: NoiseSchedule = .cosine
    /// Scale factor for DruseAF attention gradient guidance.
    var guidanceScale: Float = 1.0
    /// Vina gradient local search steps after diffusion completes.
    var refinementSteps: Int = 30

    enum NoiseSchedule: String, CaseIterable, Sendable {
        case linear   = "Linear"
        case cosine   = "Cosine"
        case quadratic = "Quadratic"
    }
}

// MARK: - Parallel Tempering Config

struct ParallelTemperingConfig: Sendable {
    /// Number of temperature replicas.
    var numReplicas: Int = 8
    /// Lowest temperature replica (kcal/mol).
    var minTemperature: Float = 0.6
    /// Highest temperature replica (kcal/mol).
    var maxTemperature: Float = 4.0
    /// Attempt replica swaps every N generations.
    var swapInterval: Int = 5
    /// MC steps per replica before result extraction.
    var stepsPerReplica: Int = 200
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
    var energy: Float           // total score (kcal/mol for Vina/Drusina/PIGNet2; -pKd*conf for DruseAF)
    var stericEnergy: Float     // gauss1 + gauss2 + repulsion
    var hydrophobicEnergy: Float
    var hbondEnergy: Float
    var torsionPenalty: Float   // rotational entropy
    var generation: Int
    var clusterID: Int = -1
    var clusterRank: Int = 0
    var transformedAtomPositions: [SIMD3<Float>] = []
    var refinementEnergy: Float? = nil

    // Drusina extended scoring correction (populated when scoring method is .drusina)
    var drusinaCorrection: Float = 0

    // Pharmacophore constraint penalty (0 = all constraints satisfied)
    var constraintPenalty: Float = 0

    // MMFF94 ligand strain energy (docked − free, kcal/mol; nil if not computed)
    var strainEnergy: Float? = nil

    // GFN2-xTB refinement outputs (populated when gfn2Refinement.enabled)
    var gfn2Energy: Float? = nil            // total GFN2-xTB energy (kcal/mol)
    var gfn2DispersionEnergy: Float? = nil  // D4 dispersion component (kcal/mol)
    var gfn2SolvationEnergy: Float? = nil   // GBSA/ALPB component (kcal/mol)
    var gfn2Converged: Bool? = nil          // optimization converged?
    var gfn2OptSteps: Int? = nil            // optimization steps taken

    // Ensemble multi-start metadata
    var formLabel: String? = nil            // chemical form label (e.g. "Taut2", "prot_Amine+Taut1")
    var formPopulation: Float? = nil        // Boltzmann population fraction of this form

    /// The display score depending on the active scoring method.
    func displayScore(method: ScoringMethod) -> Float {
        switch method {
        case .vina:          return energy
        case .drusina:       return energy  // energy already includes Drusina corrections
        case .druseAffinity: return stericEnergy  // pKd stored in stericEnergy by DruseAF shader
        case .pignet2:       return energy  // physics-decomposed kcal/mol
        }
    }

    /// Confidence score from DruseAF (0–1), stored in hydrophobicEnergy field.
    var afConfidence: Float { hydrophobicEnergy }

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
    // Unified Vina affinity maps for full GA scoring (optional, populated when available)
    var vinaAffinityGridBuffer: MTLBuffer?
    var vinaTypeIndexBuffer: MTLBuffer?
}

/// Per-ligand data prepared for batched GA virtual screening.
struct LigandGAData {
    var gpuAtoms: [DockLigandAtom]
    var torsionEdges: [TorsionEdge]
    var movingIndices: [Int32]
    var pairList: [UInt32]
    var referenceIntraEnergy: Float
    var ligandRadius: Float
    var centroid: SIMD3<Float>
    var heavyAtoms: [Atom]
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
        case amideStack = 8   // Amide-π: backbone amide ↔ aromatic ring, 3.0-5.0 Å
        case chalcogen = 9    // Chalcogen bond: C-S...O/N σ-hole, 2.8-3.8 Å

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
            case .amideStack:  SIMD4(0.9, 0.6, 0.3, 0.9)    // amber
            case .chalcogen:   SIMD4(0.8, 0.9, 0.2, 1.0)    // yellow-green
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
            case .amideStack:  "Amide-π"
            case .chalcogen:   "Chalcogen bond"
            }
        }
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

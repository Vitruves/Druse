import SwiftUI
import MetalKit

/// Docking, pocket detection, batch docking, virtual screening, and ML state
/// extracted from AppViewModel.
/// Value type — stored as a property on @Observable AppViewModel.
struct DockingCoordinator {
    // Pocket detection
    var detectedPockets: [BindingPocket] = []
    var selectedPocket: BindingPocket?

    // Docking state
    var isDocking: Bool = false
    var dockingGeneration: Int = 0
    var dockingTotalGenerations: Int = 100
    var dockingBestEnergy: Float = .infinity
    var dockingBestPKi: Float?
    var dockingResults: [DockingResult] = []
    var dockingConfig = DockingConfig()
    var dockingEngine: DockingEngine?
    var currentInteractions: [MolecularInteraction] = []
    var showInteractionDiagram: Bool = false
    var interactionDiagramPoseIndex: Int = 0

    // Pharmacophore constraints
    var pharmacophoreConstraints: [PharmacophoreConstraintDef] = []

    // Receptor flexibility (induced fit)
    var flexibleResidueConfig = FlexibleResidueConfig()
    var flexDockingEngine: FlexDockingEngine?

    // Grid box state (persisted here so tab switches don't reset it)
    var gridCenter: SIMD3<Float> = .zero
    var gridHalfSize: SIMD3<Float> = SIMD3<Float>(repeating: 10)
    var gridInitialized: Bool = false

    // Docking timing
    var dockingDuration: TimeInterval = 0
    var dockingStartTime: Date?

    // Multi-pose selection
    var selectedPoseIndices: Set<Int> = []

    /// Original ligand preserved before docking mutates the active ligand with pose transforms.
    var originalDockingLigand: Molecule?

    // Batch docking state
    var batchResults: [(ligandName: String, results: [DockingResult])] = []
    var isBatchDocking: Bool = false
    var batchProgress: (current: Int, total: Int) = (0, 0)
    var batchQueue: [LigandEntry] = []
    var batchDockingTask: Task<Void, Never>?

    // Virtual screening state
    var screeningPipeline: VirtualScreeningPipeline?
    var screeningState: VirtualScreeningPipeline.ScreeningState = .idle
    var screeningProgress: Float = 0
    var screeningHits: [VirtualScreeningPipeline.ScreeningHit] = []
    var screeningTask: Task<Void, Never>?

    // Results filter state (persists across tab switches)
    var resultsLipinskiFilter: Bool = false
    var resultsEnergyCutoff: Float = 0
    var resultsMLScoreCutoff: Float = 0
    var resultsHasInitializedCutoffs: Bool = false

    var usePostDockingRefinement: Bool = false

    // Scaffold enforcement UI
    var showScaffoldInput: Bool = false

    // Search method and scoring method preferences
    var searchMethod: SearchMethod = .genetic
    var scoringMethod: ScoringMethod = .vina
    var chargeMethod: ChargeMethod = .gasteiger
    var affinityDisplayUnit: AffinityDisplayUnit = .pKi

    // Live ML scoring: cancellable task prevents stale predictions from blocking updates
    var liveScoringTask: Task<Void, Never>?
}

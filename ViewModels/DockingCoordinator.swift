// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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
    /// Final-population poses the last run scored before output selection (0 = unknown).
    var dockingPosesEvaluated: Int = 0
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

    /// Cached prepared protein for docking scoring. Avoids re-running the full
    /// preparation pipeline (H-bond network, charges) on every docking invocation.
    struct PreparedProteinCache {
        let protein: Molecule
        let report: ProteinPreparation.DockingPreparationReport
        /// Hash of source atoms + pH — invalidated when protein or pH changes.
        let key: Int
    }
    var preparedProteinCache: PreparedProteinCache?

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

    // Pharmacophore editor
    var showPharmacophoreEditor: Bool = false

    // Search method and scoring method preferences
    var searchMethod: SearchMethod = .genetic
    var scoringMethod: ScoringMethod = .vina
    var chargeMethod: ChargeMethod = .gasteiger
    var affinityDisplayUnit: AffinityDisplayUnit = .pKi

    // Live ML scoring: cancellable task prevents stale predictions from blocking updates
    var liveScoringTask: Task<Void, Never>?
}

/// Search-budget presets shared by the Docking step's Preset buttons and the console
/// (`dock preset=…`): population × generations × runs plus a matching torsion policy.
enum DockingPreset: String, CaseIterable, Sendable {
    case auto = "Auto"
    case fast = "Fast"
    case standard = "Standard"
    case thorough = "Thorough"

    /// One-line description (button tooltip and console help).
    var help: String {
        switch self {
        case .auto: "Adapts parameters to protein/ligand complexity at launch"
        case .fast: "Quick exploration: 50 × 30 × 3"
        case .standard: "Balanced: 150 × 80 × 5"
        case .thorough: "Deep search: 300 × 200 × 10"
        }
    }

    /// Writes the preset's budget and torsion policy into `config`.
    func apply(to config: inout DockingConfig) {
        config.autoMode = self == .auto
        switch self {
        case .auto:
            config.populationSize = 200
            config.generationsPerRun = 150
            config.numRuns = 3
            config.applyTorsionSearchPreset(.balanced)
        case .fast:
            config.populationSize = 50
            config.generationsPerRun = 30
            config.numRuns = 3
            config.applyTorsionSearchPreset(.balanced)
        case .standard:
            config.populationSize = 150
            config.generationsPerRun = 80
            config.numRuns = 5
            config.applyTorsionSearchPreset(.balanced)
        case .thorough:
            config.populationSize = 300
            config.generationsPerRun = 200
            config.numRuns = 10
            config.applyTorsionSearchPreset(.exploratory)
        }
    }

    /// Whether `config` currently matches this preset (highlights the button, names the preset in `dock config`).
    func isActive(in config: DockingConfig) -> Bool {
        switch self {
        case .auto:     config.autoMode
        case .fast:     !config.autoMode && config.populationSize == 50 && config.generationsPerRun == 30 && config.numRuns == 3
        case .standard: !config.autoMode && config.populationSize == 150 && config.generationsPerRun == 80 && config.numRuns == 5
        case .thorough: !config.autoMode && config.populationSize == 300 && config.generationsPerRun == 200 && config.numRuns == 10
        }
    }
}

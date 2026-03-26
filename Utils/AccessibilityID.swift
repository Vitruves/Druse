import Foundation
import SwiftUI

// MARK: - View extension for plain-style buttons

extension View {
    /// Apply to `.buttonStyle(.plain)` buttons so XCUITest can find them.
    /// Combines children, adds button trait, and sets the identifier.
    func plainButtonAccessibility(_ id: String) -> some View {
        self
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isButton)
            .accessibilityIdentifier(id)
    }
}

/// Centralized accessibility identifiers for XCUITest automation.
/// Convention: `area_action` in camelCase.
enum AccessibilityID {
    // MARK: - Welcome Screen
    static let welcomeStartProtein = "welcome_startProtein"
    static let welcomeStartLigand = "welcome_startLigand"
    static let welcomeOpenProject = "welcome_openProject"
    static let welcomeShowMe = "welcome_showMe"

    // MARK: - Pipeline Bar (toolbar tabs)
    static func pipelineTab(_ tab: String) -> String { "pipeline_\(tab)" }

    // MARK: - Search Tab
    static let searchImportProtein = "search_importProtein"
    static let searchImportLigand = "search_importLigand"
    static let searchClearProtein = "search_clearProtein"
    static let searchClearLigand = "search_clearLigand"
    static let searchPDBField = "search_pdbField"
    static let searchFetchPDB = "search_fetchPDB"
    static let searchQueryField = "search_queryField"
    static let searchRunSearch = "search_runSearch"
    static let searchLoadStructure = "search_loadStructure"

    // MARK: - Preparation Tab
    static let prepRemoveWaters = "prep_removeWaters"
    static let prepKeepPocketWaters = "prep_keepPocketWaters"
    static let prepRemoveNonStandard = "prep_removeNonStandard"
    static let prepRemoveAltConfs = "prep_removeAltConfs"
    static let prepAddHydrogens = "prep_addHydrogens"
    static let prepRemoveHydrogens = "prep_removeHydrogens"
    static let prepAddPolarH = "prep_addPolarH"
    static let prepAssignCharges = "prep_assignCharges"
    static let prepStructureCleanup = "prep_structureCleanup"
    static let prepFixMissing = "prep_fixMissing"
    static let prepAnalyzeMissing = "prep_analyzeMissing"
    static let prepRepairMissing = "prep_repairMissing"
    static let prepRetainBridgingWaters = "prep_retainBridgingWaters"
    static let prepPHSlider = "prep_phSlider"
    static let prepChargePicker = "prep_chargePicker"

    // MARK: - Sequence Tab
    static let seqClearSelection = "seq_clearSelection"
    static let seqCopySequence = "seq_copySequence"
    static let seqSelectHelices = "seq_selectHelices"
    static let seqSelectSheets = "seq_selectSheets"
    static let seqSelectCoils = "seq_selectCoils"
    static let seqSelectTurns = "seq_selectTurns"
    static let seqDeleteSelected = "seq_deleteSelected"

    // MARK: - Ligand Database Tab
    static let ligSmilesField = "lig_smilesField"
    static let ligAddSmiles = "lig_addSmiles"
    static let ligOpenManager = "lig_openManager"
    static let ligSaveDB = "lig_saveDB"
    static let ligLoadDB = "lig_loadDB"
    static let ligClearLigand = "lig_clearLigand"
    static func ligUseEntry(_ index: Int) -> String { "lig_use_\(index)" }

    // MARK: - Docking Tab
    static let dockOpenLigandDB = "dock_openLigandDB"
    static let dockClearBatch = "dock_clearBatch"
    static let dockRemoveLigand = "dock_removeLigand"
    static let dockDetectAuto = "dock_detectAuto"
    static let dockDetectML = "dock_detectML"
    static let dockDetectLigand = "dock_detectLigand"
    static let dockDetectSelection = "dock_detectSelection"
    static let dockFocusPocket = "dock_focusPocket"
    static let dockApplyGrid = "dock_applyGrid"
    static let dockGridProtein = "dock_gridProtein"
    static let dockGridLigand = "dock_gridLigand"
    static let dockGridSelection = "dock_gridSelection"
    static let dockGridPocket = "dock_gridPocket"
    static let dockPocketViewToggle = "dock_pocketViewToggle"
    static let dockSlabTight = "dock_slabTight"
    static let dockSlabMedium = "dock_slabMedium"
    static let dockSlabWide = "dock_slabWide"
    static let dockStartButton = "dock_startButton"
    static let dockCancelButton = "dock_cancelButton"

    // MARK: - PreDock Sheet
    static let preDockScoringPicker = "preDock_scoringPicker"
    static let preDockUnitPicker = "preDock_unitPicker"
    static let preDockMLToggle = "preDock_mlToggle"
    static let preDockCancel = "preDock_cancel"
    static let preDockStart = "preDock_start"
    static let preDockDismiss = "preDock_dismiss"

    // MARK: - Results Tab
    static let resultsOpenDB = "results_openDB"
    static let resultsViewSelected = "results_viewSelected"
    static let resultsClearSelection = "results_clearSelection"
    static let resultsClearPoses = "results_clearPoses"
    static let resultsExportSDF = "results_exportSDF"
    static let resultsExportCSV = "results_exportCSV"
    static let resultsExportHits = "results_exportHits"
    static let resultsCancelBatch = "results_cancelBatch"
    static func resultsDiagram(_ index: Int) -> String { "results_diagram_\(index)" }
    static func resultsViewPose(_ index: Int) -> String { "results_viewPose_\(index)" }
    static func resultsOptimize(_ index: Int) -> String { "results_optimize_\(index)" }
    static func resultsCheckbox(_ index: Int) -> String { "results_checkbox_\(index)" }

    // MARK: - Lead Optimization Tab
    static let leadGenerate = "lead_generate"
    static let leadClear = "lead_clear"
    static let leadDockAll = "lead_dockAll"
    static let leadStop = "lead_stop"
    static let leadComparison = "lead_comparison"
    static let leadScaffoldToggle = "lead_scaffoldToggle"
    static func leadViewAnalog(_ index: Int) -> String { "lead_view_\(index)" }

    // MARK: - Render Controls (bottom bar)
    static func renderMode(_ mode: String) -> String { "render_\(mode)" }
    static let renderHydrogens = "render_hydrogens"
    static let renderSurface = "render_surface"
    static let renderLighting = "render_lighting"
    static let renderClipping = "render_clipping"
    static let renderFitToView = "render_fitToView"
    static let renderFitToLigand = "render_fitToLigand"
    static let renderResetCamera = "render_resetCamera"
    static let renderColorScheme = "render_colorScheme"
    static let renderSurfaceColor = "render_surfaceColor"

    // MARK: - Status Strip
    static let statusToggleConsole = "status_toggleConsole"
    static let statusStop = "status_stop"
    static let statusSaveProject = "status_saveProject"
    static let statusOpenProject = "status_openProject"
    static let statusCopySelected = "status_copySelected"
    static let statusCopyAll = "status_copyAll"
    static let statusClearLog = "status_clearLog"
    static let statusRevealLog = "status_revealLog"

    // MARK: - Inspector
    static let inspectorClose = "inspector_close"
    static let inspectorToggle = "inspector_toggle"
}

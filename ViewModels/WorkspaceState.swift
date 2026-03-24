import SwiftUI
import MetalKit

/// UI/rendering/selection/surface state extracted from AppViewModel.
/// Value type — stored as a property on @Observable AppViewModel so SwiftUI
/// bindings (`$vm.workspace.showHydrogens`) chain correctly.
struct WorkspaceState {
    // Render state
    var renderMode: RenderMode = .ballAndStick
    var showHydrogens: Bool = true
    var showProtein: Bool = true
    var showLigand: Bool = true
    var hiddenChainIDs: Set<String> = []
    var useDirectionalLighting: Bool = false

    /// Which residue side chains to show as ball-and-stick in ribbon mode.
    var sideChainDisplay: SideChainDisplay = .none

    // Z-slab clipping
    var enableClipping: Bool = false
    var clipNearZ: Float = 0
    var clipFarZ: Float = 100

    // Molecular surface
    var showSurface: Bool = false
    var surfaceType: SurfaceFieldType = .connolly
    var surfaceColorMode: SurfaceColorMode = .uniform
    var surfaceOpacity: Float = 0.85
    var isGeneratingSurface: Bool = false
    var surfaceGenerator: SurfaceGenerator?

    // Background
    var backgroundOpacity: Float = 1.0   // 0 = fully transparent gradient → white, 1 = normal

    // Grid box
    var gridLineWidth: Float = 2.5       // screen-space pixels
    var gridColor: SIMD4<Float> = SIMD4<Float>(0.2, 1.0, 0.4, 0.8)

    // Selection
    var selectedAtomIndex: Int? = nil
    var selectedAtomIndices: Set<Int> = []
    var selectedResidueIndices: Set<Int> = []
    var hiddenAtomIndices: Set<Int> = []

    // Residue subsets (MOE-style user-defined groups)
    var residueSubsets: [ResidueSubset] = []

    // Ligand conformers (synced from database when available)
    struct Conformer {
        var atoms: [Atom]
        var bonds: [Bond]
        var energy: Double
    }
    var ligandConformers: [Conformer] = []
    var activeConformerIndex: Int = 0

    // Pharmacophore constraint sheet
    var showingConstraintSheet: Bool = false
    var constraintSheetContext: ConstraintSheetContext?

    // Status
    var statusMessage: String = "Ready"

    // Loading state
    var isLoading: Bool = false
    var loadingMessage: String = ""

    // Search state
    var searchResults: [PDBSearchResult] = []
    var isSearching: Bool = false
}

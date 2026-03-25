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

    /// Uniform protein coloring (MOE-style): when set, all protein atoms use this color
    /// so the ligand "pops" visually in the binding site.
    var uniformProteinColor: SIMD3<Float>? = nil

    /// Ligand carbon color override. nil = standard CPK gray. Set to yellow for MOE-style.
    /// Only affects carbon atoms in the ligand, not heteroatoms.
    var ligandCarbonColor: SIMD3<Float>? = nil

    /// Predefined color schemes
    enum MoleculeColorScheme: String, CaseIterable {
        case element = "Element (CPK)"
        case ligandFocus = "Ligand Focus"   // yellow carbons, gray protein
        case chainColored = "Chain Colored"

        var proteinColor: SIMD3<Float>? {
            switch self {
            case .element: return nil
            case .ligandFocus: return SIMD3<Float>(0.65, 0.65, 0.65) // light gray
            case .chainColored: return nil
            }
        }
        var ligandCarbon: SIMD3<Float>? {
            switch self {
            case .element: return nil
            case .ligandFocus: return SIMD3<Float>(0.85, 0.78, 0.20) // gold/yellow
            case .chainColored: return nil
            }
        }

        /// 10-color palette for per-chain coloring (wraps for >10 chains).
        static let chainPalette: [SIMD3<Float>] = [
            SIMD3<Float>(0.35, 0.55, 0.85),  // blue
            SIMD3<Float>(0.85, 0.40, 0.40),  // red
            SIMD3<Float>(0.40, 0.75, 0.45),  // green
            SIMD3<Float>(0.80, 0.65, 0.30),  // gold
            SIMD3<Float>(0.65, 0.40, 0.80),  // purple
            SIMD3<Float>(0.30, 0.75, 0.75),  // teal
            SIMD3<Float>(0.85, 0.55, 0.30),  // orange
            SIMD3<Float>(0.65, 0.65, 0.65),  // gray
            SIMD3<Float>(0.80, 0.45, 0.65),  // pink
            SIMD3<Float>(0.50, 0.70, 0.35),  // lime
        ]
    }
    var colorScheme: MoleculeColorScheme = .element

    // Z-slab clipping
    var enableClipping: Bool = false
    var clipNearZ: Float = 0
    var clipFarZ: Float = 100

    // Object-space slab clipping (zoom-invariant)
    var slabThickness: Float = 20
    var slabOffset: Float = 0

    // Molecular surface
    var showSurface: Bool = false
    var surfaceType: SurfaceFieldType = .connolly
    var surfaceColorMode: SurfaceColorMode = .uniform
    var surfaceOpacity: Float = 0.85
    var surfaceProbeRadius: Float = 1.4  // Å — water probe radius (1.4 default, increase for ligand-accessible view)
    var surfaceGridSpacing: Float = 0.5  // Å — finer = more detail but slower
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

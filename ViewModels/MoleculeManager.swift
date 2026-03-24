import SwiftUI

/// Molecule loading, preparation, and editing state extracted from AppViewModel.
/// Value type — stored as a property on @Observable AppViewModel.
struct MoleculeManager {
    // Molecules
    var protein: Molecule?
    var ligand: Molecule?

    // Preparation state
    var preparationReport: ProteinPreparation.PreparationReport?
    var protonationPH: Float = 7.4
    var isMinimizing: Bool = false
    /// Tracks whether the user has actively run any preparation step (not just loaded a protein).
    var proteinPrepared: Bool = false

    // Raw PDB content (cached for C++ protein prep)
    var rawPDBContent: String?

    // Bridging waters: residue keys (chainID_seq) the user chose to retain
    var keptWaterKeys: Set<String> = []
    var pocketWaterRadius: Float = 5.0  // Å around pocket center for auto-retention
}

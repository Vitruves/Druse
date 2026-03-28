import SwiftUI

/// Wrapper that hosts InteractionDiagramView in a standalone macOS window.
/// Reads all required data from the shared AppViewModel.
struct InteractionDiagramWindow: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        Group {
            if let ligand = viewModel.molecules.ligand,
               let protein = viewModel.molecules.protein,
               viewModel.docking.interactionDiagramPoseIndex < viewModel.docking.dockingResults.count {
                let idx = viewModel.docking.interactionDiagramPoseIndex
                let result = viewModel.docking.dockingResults[idx]
                let sm = viewModel.docking.scoringMethod
                InteractionDiagramView(
                    interactions: viewModel.docking.currentInteractions,
                    ligandAtoms: ligand.atoms.filter { $0.element != .H },
                    ligandBonds: ligand.bonds,
                    proteinAtoms: protein.atoms.filter { $0.element != .H },
                    ligandSmiles: ligand.smiles ?? ligand.title,
                    poseEnergy: result.displayScore(method: sm),
                    poseIndex: idx,
                    scoringMethod: sm
                )
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "circle.hexagongrid")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No pose selected")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

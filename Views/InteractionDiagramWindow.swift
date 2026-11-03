// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Wrapper that hosts InteractionDiagramView in a standalone macOS window.
/// Reads all required data from the shared AppViewModel and lets the user
/// switch pose (and batch ligand) from the toolbar.
struct InteractionDiagramWindow: View {
    @Environment(AppViewModel.self) private var viewModel

    private var results: [DockingResult] { viewModel.docking.dockingResults }

    var body: some View {
        Group {
            if let ligand = viewModel.molecules.ligand,
               let protein = viewModel.molecules.protein,
               viewModel.docking.interactionDiagramPoseIndex < results.count {
                let idx = viewModel.docking.interactionDiagramPoseIndex
                let sm = viewModel.docking.scoringMethod
                let heavy = Self.heavyGraph(of: ligand)
                InteractionDiagramView(
                    interactions: viewModel.docking.currentInteractions,
                    ligandName: ligand.name,
                    ligandAtoms: heavy.atoms,
                    ligandBonds: heavy.bonds,
                    proteinAtoms: protein.atoms.filter { $0.element != .H },
                    ligandSmiles: ligand.smiles,
                    poseEnergy: results[idx].displayScore(method: sm),
                    poseIndex: idx,
                    scoringMethod: sm
                )
                // Rebuild the layout whenever the ligand, pose or its contacts change.
                .id(diagramIdentity(ligandName: ligand.name, poseIndex: idx))
            } else {
                DSEmptyState("No Pose Selected",
                             systemImage: "point.3.connected.trianglepath.dotted",
                             description: "Choose a pose in Results to see its 2D interaction map.")
                    .frame(minWidth: 650, maxWidth: .infinity, minHeight: 550, maxHeight: .infinity)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                if !viewModel.docking.batchResults.isEmpty {
                    ligandPicker
                }
                if !results.isEmpty {
                    posePicker
                }
            }
        }
    }

    // MARK: - Pickers

    private var posePicker: some View {
        let sm = viewModel.docking.scoringMethod
        return Picker("Pose", selection: Binding(
            get: { min(viewModel.docking.interactionDiagramPoseIndex, max(results.count - 1, 0)) },
            set: { index in
                viewModel.showDockingPose(at: index)
                viewModel.docking.interactionDiagramPoseIndex = index
            }
        )) {
            ForEach(Array(results.prefix(100).enumerated()), id: \.offset) { index, result in
                Text(String(format: "Pose #%d   %.1f %@", index + 1, result.displayScore(method: sm), sm.unitLabel))
                    .monospacedDigit()
                    .tag(index)
            }
        }
        .pickerStyle(.menu)
        .frame(minWidth: 170)
        .help("Choose which docked pose to diagram")
    }

    private var ligandPicker: some View {
        let names = viewModel.docking.batchResults.map(\.ligandName)
        let current = viewModel.molecules.ligand?.name ?? ""
        return Picker("Ligand", selection: Binding(
            get: { names.contains(current) ? current : (names.first ?? "") },
            set: { name in
                guard let entry = viewModel.docking.batchResults.first(where: { $0.ligandName == name }) else { return }
                ResultsActions.loadBatchLigand(named: name, results: entry.results, viewModel: viewModel)
                viewModel.docking.interactionDiagramPoseIndex = 0
            }
        )) {
            ForEach(names, id: \.self) { name in
                Text(name).tag(name)
            }
        }
        .pickerStyle(.menu)
        .frame(minWidth: 150)
        .help("Choose which batch ligand to diagram")
    }

    /// Heavy atoms with bonds re-indexed into the heavy-atom array (interactions index heavy atoms).
    private static func heavyGraph(of ligand: Molecule) -> (atoms: [Atom], bonds: [Bond]) {
        var heavyIndex: [Int: Int] = [:]
        var atoms: [Atom] = []
        for (i, atom) in ligand.atoms.enumerated() where atom.element != .H {
            heavyIndex[i] = atoms.count
            atoms.append(atom)
        }
        let bonds: [Bond] = ligand.bonds.compactMap { bond in
            guard let a = heavyIndex[bond.atomIndex1], let b = heavyIndex[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }
        return (atoms, bonds)
    }

    private func diagramIdentity(ligandName: String, poseIndex: Int) -> Int {
        var hasher = Hasher()
        hasher.combine(ligandName)
        hasher.combine(poseIndex)
        for ixn in viewModel.docking.currentInteractions {
            hasher.combine(ixn.ligandAtomIndex)
            hasher.combine(ixn.proteinAtomIndex)
            hasher.combine(ixn.type)
        }
        return hasher.finalize()
    }
}

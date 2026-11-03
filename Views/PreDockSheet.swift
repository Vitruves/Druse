// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Confirmation sheet displayed before launching a docking run ("Confirm before docking").
/// Read-only review of protein, ligand, binding site, search and scoring, with Cancel / Start.
struct PreDockSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) private var viewModel
    @AppStorage("ui.confirmBeforeDocking") private var confirmBeforeDocking = true
    /// Torsions the search will sample, from the same RDKit torsion tree the engine builds.
    /// `Bond.isRotatable` is not populated by the parsers, so counting it always gave 0.
    @State private var torsionCount: Int?

    private var canStart: Bool {
        viewModel.docking.selectedPocket != nil
            && viewModel.molecules.ligand != nil
            && viewModel.molecules.protein != nil
    }

    var body: some View {
        DSSheetScaffold(
            title: "Review & Run",
            subtitle: "Check the inputs for this docking run before it starts.",
            systemImage: "play.circle"
        ) {
            Form {
                proteinSection
                ligandSection
                siteSection
                if !viewModel.docking.pharmacophoreConstraints.isEmpty {
                    constraintSection
                }
                searchSection
                scoringSection
                computeSection
            }
            .formStyle(.grouped)
            .task(id: viewModel.molecules.ligand?.smiles) {
                let smiles = viewModel.molecules.ligand?.smiles ?? ""
                torsionCount = smiles.isEmpty ? nil : RDKitBridge.buildTorsionTree(smiles: smiles)?.count
            }
        } footer: {
            Toggle("Don't ask again", isOn: Binding(
                get: { !confirmBeforeDocking },
                set: { confirmBeforeDocking = !$0 }
            ))
            .toggleStyle(.checkbox)
            .font(DS.Font.secondary)
            .help("Start docking directly from the Run button. Re-enable in Settings ▸ Docking.")
            .frame(maxWidth: .infinity, alignment: .leading)
            .layoutPriority(1)

            Button("Cancel", role: .cancel) { dismiss() }
                .keyboardShortcut(.cancelAction)
                .fixedSize()
                .accessibilityIdentifier(AccessibilityID.preDockCancel)

            Button {
                viewModel.runDocking()
                dismiss()
            } label: {
                Label("Start Docking", systemImage: "play.fill")
            }
            .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
            .keyboardShortcut(.defaultAction)
            .fixedSize()
            .disabled(!canStart)
            .accessibilityIdentifier(AccessibilityID.preDockStart)
        }
        .overlay(alignment: .topTrailing) {
            Button { dismiss() } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Close")
            .accessibilityLabel("Close")
            .accessibilityIdentifier(AccessibilityID.preDockDismiss)
            .padding(DS.Space.l)
        }
        .frame(minWidth: 460, idealWidth: 480, minHeight: 480, idealHeight: 600)
    }

    // MARK: - Sections

    @ViewBuilder
    private var proteinSection: some View {
        Section {
            if let prot = viewModel.molecules.protein {
                entityRow(prot.name, tint: DS.Color.protein, subtitle: prot.title.isEmpty ? nil : prot.title)
                value("Atoms", "\(prot.atomCount.formatted()) (\(prot.heavyAtomCount.formatted()) heavy)")
                value("Residues", prot.residues.count.formatted())
                if !prot.chains.isEmpty {
                    value("Chains", prot.chainIDs.joined(separator: ", "))
                }
            } else {
                missing("No protein loaded")
            }
        } header: {
            Text("Protein")
        }
    }

    @ViewBuilder
    private var ligandSection: some View {
        Section {
            if let lig = viewModel.molecules.ligand {
                entityRow(lig.name, tint: DS.Color.ligand, subtitle: nil)
                value("Atoms", "\(lig.atomCount) (\(lig.heavyAtomCount) heavy)")
                value("Bonds", "\(lig.bondCount)")
                value("Rotatable bonds", torsionCount.map(String.init) ?? "—")
                value("Molecular weight", String(format: "%.1f Da", RDKitBridge.parentMolecularWeight(of: lig)))
            } else {
                missing("No ligand loaded")
            }
        } header: {
            Text("Ligand")
        }
    }

    @ViewBuilder
    private var siteSection: some View {
        Section {
            if let pocket = viewModel.docking.selectedPocket {
                value("Volume", String(format: "%.0f Å³", pocket.volume))
                value("Buriedness", String(format: "%.0f%%", pocket.buriedness * 100))
                value("Polarity", String(format: "%.0f%%", pocket.polarity * 100))
                value("Rank score", String(format: "%.1f", pocket.druggability))
                value("Residues", "\(pocket.residueIndices.count)")
                value("Center", String(format: "%.1f, %.1f, %.1f", pocket.center.x, pocket.center.y, pocket.center.z))
                value("Box size", DockingSite.boxDimensions(pocket.size))
            } else {
                missing("No pocket selected — define one in the Binding Site step.")
            }
        } header: {
            Text("Binding Site")
        }
    }

    private var constraintSection: some View {
        let constraints = viewModel.docking.pharmacophoreConstraints.filter(\.isEnabled)
        let hardCount = constraints.filter { $0.strength.isHard }.count
        return Section {
            value("Active", hardCount > 0 ? "\(constraints.count) (\(hardCount) hard)" : "\(constraints.count)")
            ForEach(constraints) { c in
                LabeledContent {
                    DSTag(c.strength.label, tone: c.strength.isHard ? .danger : .neutral)
                } label: {
                    Label {
                        Text("\(c.interactionType.rawValue) → \(c.targetLabel)")
                            .lineLimit(1)
                    } icon: {
                        Image(systemName: c.interactionType.icon)
                            .foregroundStyle(c.interactionType.dsColor)
                    }
                }
            }
        } header: {
            Text("Pharmacophore Constraints")
        }
    }

    private var searchSection: some View {
        let config = viewModel.docking.dockingConfig
        return Section {
            value("Method", config.searchMethod.shortLabel)
            value("Population size", "\(config.populationSize)")
            value("Generations", "\(config.numGenerations)")
            value("Grid spacing", String(format: "%.3f Å", config.gridSpacing))
            value("Mutation rate", String(format: "%.3f", config.mutationRate))
            value("Crossover rate", String(format: "%.1f", config.crossoverRate))
            value("Translation step", String(format: "%.1f Å", config.translationStep))
            value("Rotation step", String(format: "%.2f rad", config.rotationStep))
            value("Total evaluations", "≈ \((config.populationSize * config.numGenerations).formatted())")
        } header: {
            Text("Search")
        }
    }

    @ViewBuilder
    private var scoringSection: some View {
        @Bindable var vm = viewModel
        let method = viewModel.docking.scoringMethod
        Section {
            LabeledContent("Scoring") {
                HStack(spacing: DS.Space.m) {
                    Label(method.shortLabel, systemImage: method.icon)
                    Button("Change…") { dismiss() }
                        .buttonStyle(.borderless)
                        .help("Close and change the scoring function in the Docking step")
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier(AccessibilityID.preDockScoringPicker)

            if method == .druseAffinity {
                Picker("Display unit", selection: $vm.docking.affinityDisplayUnit) {
                    ForEach(AffinityDisplayUnit.allCases, id: \.self) { unit in
                        Text(unit.rawValue).tag(unit)
                    }
                }
                .pickerStyle(.segmented)
                .help("pKi: -log10(Ki). Ki: dissociation constant in nM/µM")
                .accessibilityIdentifier(AccessibilityID.preDockUnitPicker)
            }

            Toggle("DruseAF rescoring", isOn: $vm.docking.dockingConfig.useAFv4Rescore)
                .help("Re-rank top poses with DruseAF v4 after docking")
                .accessibilityIdentifier(AccessibilityID.preDockMLToggle)
        } header: {
            Text("Scoring")
        } footer: {
            Text(method.description)
        }
    }

    private var computeSection: some View {
        Section {
            LabeledContent {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(DS.Color.success)
            } label: {
                Label(viewModel.renderer?.device.name ?? "Metal GPU", systemImage: "cpu")
            }
        } header: {
            Text("Compute Device")
        }
    }

    // MARK: - Rows

    private func entityRow(_ name: String, tint: Color, subtitle: String?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
            Circle()
                .fill(tint)
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text(name)
                    .font(DS.Font.sectionTitle)
                if let subtitle {
                    Text(subtitle)
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
    }

    private func value(_ label: String, _ text: String) -> some View {
        LabeledContent(label) {
            Text(text)
                .monospacedDigit()
                .textSelection(.enabled)
        }
    }

    private func missing(_ text: String) -> some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .foregroundStyle(DS.Color.warning)
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Inspector (single entry)
//
// Native `.inspector` column of the Ligand Database window. Dispatches between
// the multi-selection batch pane (LigandBatchPanel.swift), the single-entry
// detail pane below, and an empty state.

extension LigandDatabaseWindow {

    @ViewBuilder
    var inspectorPane: some View {
        if selectedIDs.count > 1 {
            batchActionPanel
        } else if let entry = inspectedEntry {
            // Stable identity per entry — only recreate when the user picks a
            // different ligand. Conformer/atom mutations within the same entry
            // must NOT tear down the embedded MetalView.
            entryDetailPane(entry)
                .id(entry.id)
        } else {
            DSEmptyState(
                "No Ligand Selected",
                systemImage: "hexagon",
                description: "Select a compound to see its structure, properties and conformers."
            )
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private func entryDetailPane(_ entry: LigandEntry) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                detailHeader(entry)

                LigandDepictionView(smiles: entry.originalSMILES)
                    .frame(height: 200)

                structure3DSection(entry)

                if let d = entry.descriptors {
                    propertiesSection(d)
                }

                if entry.ki != nil || entry.effectivePKi != nil || entry.ic50 != nil {
                    affinitySection(entry)
                }

                if !entry.userProperties.isEmpty {
                    userPropertiesSection(entry)
                }

                if let error = entry.preparationError {
                    PanelHint(text: error, icon: "xmark.octagon.fill", color: DS.Color.danger)
                }

                detailActions(entry)
            }
            .padding(DS.Space.l)
        }
        .scrollIndicators(.automatic)
    }

    @ViewBuilder
    private func detailHeader(_ entry: LigandEntry) -> some View {
        let isActive = viewModel.activeLigandEntryID == entry.id
        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                Text(entry.name)
                    .font(DS.Font.stepTitle)
                    .lineLimit(2)
                    .textSelection(.enabled)
                Spacer(minLength: DS.Space.xs)
                if isActive {
                    DSTag("Active", tone: .ligand, systemImage: "hexagon.fill")
                }
            }

            FlowLayout(spacing: DS.Space.xs, lineSpacing: DS.Space.xs) {
                if entry.preparationError != nil {
                    DSTag("Failed", tone: .danger, systemImage: "xmark")
                } else if entry.isPrepared {
                    DSTag("Prepared", tone: .success, systemImage: "checkmark")
                } else {
                    DSTag("Not prepared")
                }
                if entry.isEnumerated, let kind = entry.formKind {
                    DSTag(kind.label, tint: Self.formKindColor(kind))
                    if let pop = entry.populationWeight {
                        DSTag(String(format: "%.1f%%", pop * 100), tone: Self.populationTone(pop))
                            .help("Boltzmann population at the target pH")
                    }
                }
            }

            if let parent = entry.parentName {
                Text("Form of \(parent)")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .top, spacing: DS.Space.xs) {
                Text(entry.originalSMILES)
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(entry.originalSMILES, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(DS.Font.secondary)
                }
                .buttonStyle(.borderless)
                .help("Copy SMILES")
                .accessibilityLabel("Copy SMILES")
            }
        }
    }

    @ViewBuilder
    private func structure3DSection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSubheader(title: "3D Conformers")
            LigandMini3DView(
                atoms: entry.atoms,
                bonds: entry.bonds,
                conformers: conformers,
                selectedConformerIndex: $selectedConformerIndex
            )
            .frame(height: 240)
            .clipShape(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                    .strokeBorder(DS.Color.hairline, lineWidth: 0.5)
            )
        }
    }

    @ViewBuilder
    private func propertiesSection(_ d: LigandDescriptors) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack {
                PanelSubheader(title: "Properties")
                Spacer()
                DSTag(d.lipinski ? "Lipinski pass" : "Lipinski fail",
                      tone: d.lipinski ? .success : .warning,
                      systemImage: d.lipinski ? "checkmark" : "exclamationmark.triangle")
            }
            DSMetricGrid(columns: 3) {
                DSMetric("MW", value: String(format: "%.1f", d.molecularWeight), unit: "Da")
                DSMetric("LogP", value: d.logPText)
                DSMetric("TPSA", value: String(format: "%.0f", d.tpsa), unit: "Å²")
                DSMetric("HBD", value: "\(d.hbd)")
                DSMetric("HBA", value: "\(d.hba)")
                DSMetric("RotB", value: "\(d.rotatableBonds)")
            }
            FlowLayout(spacing: DS.Space.xs, lineSpacing: DS.Space.xs) {
                DSTag("\(d.heavyAtomCount) heavy atoms")
                DSTag("\(d.rings) \(d.rings == 1 ? "ring" : "rings") · \(d.aromaticRings) aromatic")
                DSTag(String(format: "Fsp3 %.2f", d.fractionCSP3))
                DSTag(d.veber ? "Veber pass" : "Veber fail", tone: d.veber ? .neutral : .warning)
            }
        }
    }

    @ViewBuilder
    private func affinitySection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSubheader(title: "Experimental Affinity")
            HStack(spacing: DS.Space.l) {
                if let pK = entry.effectivePKi {
                    DSScoreBadge(pK: pK)
                }
                if let ki = entry.ki {
                    LabeledContent("Ki", value: String(format: "%g nM", ki))
                }
                if let ic50 = entry.ic50 {
                    LabeledContent("IC50", value: String(format: "%g nM", ic50))
                }
            }
            .font(DS.Font.label)
            .monospacedDigit()
        }
    }

    @ViewBuilder
    private func userPropertiesSection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSubheader(title: "Imported Properties")
            VStack(spacing: DS.Space.xs) {
                ForEach(entry.userProperties.keys.sorted(), id: \.self) { key in
                    LabeledContent {
                        Text(entry.userProperties[key] ?? "")
                            .font(DS.Font.monoSmall)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    } label: {
                        Text(key)
                            .font(DS.Font.label)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func detailActions(_ entry: LigandEntry) -> some View {
        let isParentWithForms = db.hasEnumeratedChildren(entry)
        VStack(spacing: DS.Space.s) {
            Button {
                useEntryForDocking(entry)
            } label: {
                Label("Use for Docking", systemImage: "arrow.right.circle.fill")
            }
            .buttonStyle(.dsPrimary)
            .disabled(entry.atoms.isEmpty || isParentWithForms)
            .help(entry.atoms.isEmpty ? "Prepare this compound first"
                  : isParentWithForms ? "Dock one of the enumerated forms instead"
                  : "Make this the active docking ligand")

            HStack(spacing: DS.Space.s) {
                Button {
                    prepareSingleEntry(entry)
                } label: {
                    Label(entry.isPrepared ? "Re-prepare" : "Prepare", systemImage: "wand.and.stars")
                }
                .buttonStyle(.dsSecondary)
                .disabled(entry.smiles.isEmpty || isProcessing)
                .help("Prepare the dominant form: add H → protonate at pH → minimize → charges → conformers")

                Button {
                    editingEntryID = entry.id
                    showChemDrawer = true
                } label: {
                    Label("Edit…", systemImage: "pencil.and.outline")
                }
                .buttonStyle(.dsSecondary)
                .disabled(entry.smiles.isEmpty)
                .help("Edit the structure in the drawing editor")
            }
        }
    }

    // MARK: - Palette helpers (shared with the table)

    static func formKindColor(_ kind: ChemicalFormKind) -> Color {
        switch kind {
        case .parent:           DS.Color.success
        case .tautomer:         DS.Color.info
        case .protomer:         DS.Color.warning
        case .tautomerProtomer: DS.Color.pocket
        }
    }

    static func populationTone(_ pop: Double) -> DSTone {
        pop > 0.3 ? .success : pop > 0.1 ? .warning : .neutral
    }
}

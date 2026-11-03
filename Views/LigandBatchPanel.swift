// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - LigandDatabaseWindow Inspector (multi-selection)
// Batch actions for a multi-row selection: Prepare, Enumerate, Dock, Delete.

extension LigandDatabaseWindow {

    @ViewBuilder
    var batchActionPanel: some View {
        let selected = db.entries.filter { selectedIDs.contains($0.id) }
        let nPrepared = selected.filter(\.isPrepared).count
        let nFailed = selected.filter { !$0.isPrepared && $0.preparationError != nil }.count
        let nNotPrepared = selected.count - nPrepared - nFailed
        let nConformers = selected.reduce(0) { $0 + $1.conformerCount }
        let nEnumerated = selected.filter(\.isEnumerated).count
        let nPreparable = selected.count - nEnumerated
        // Same rule as useSelectedForDocking(): parents with enumerated forms are skipped.
        let nDockable = selected.filter { $0.isPrepared && !db.hasEnumeratedChildren($0) }.count
        // Once every preparable compound is prepared, docking becomes the
        // primary next step and preparing turns into a secondary re-run.
        let allPrepared = nPreparable > 0 && selected.allSatisfy { $0.isEnumerated || $0.isPrepared }

        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("\(selected.count) Compounds")
                            .font(DS.Font.stepTitle)
                            .monospacedDigit()
                        Spacer()
                        Button("Deselect All") { selectedIDs.removeAll() }
                            .buttonStyle(.borderless)
                            .font(DS.Font.secondary)
                    }
                    if nEnumerated > 0 || nConformers > 0 {
                        FlowLayout(spacing: DS.Space.xs, lineSpacing: DS.Space.xs) {
                            if nEnumerated > 0 {
                                DSTag("\(nEnumerated) enumerated forms", tone: .info)
                            }
                            if nConformers > 0 {
                                DSTag("\(nConformers) conformers")
                            }
                        }
                    }
                }

                DSMetricGrid(columns: 3) {
                    DSMetric("Prepared", value: "\(nPrepared)", tone: nPrepared > 0 ? .success : .neutral)
                    DSMetric("Not prepared", value: "\(nNotPrepared)")
                    DSMetric("Failed", value: "\(nFailed)", tone: nFailed > 0 ? .danger : .neutral)
                }

                if isBatchProcessing {
                    VStack(alignment: .leading, spacing: DS.Space.xs) {
                        ProgressView(value: Double(batchProgress.current),
                                     total: Double(max(batchProgress.total, 1)))
                            .controlSize(.small)
                        Text(processingMessage.isEmpty
                             ? "Processing \(batchProgress.current)/\(batchProgress.total)…"
                             : processingMessage)
                            .font(DS.Font.caption)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .dsCard(padding: DS.Space.m)
                }

                VStack(alignment: .leading, spacing: DS.Space.s) {
                    PanelSubheader(title: "Preparation")
                    if isBatchProcessing {
                        Button(action: cancelPopulateAndPrepare) {
                            Label("Stop Processing", systemImage: "stop.fill")
                        }
                        .buttonStyle(DSPrimaryButtonStyle(tint: DS.Color.danger))
                    } else {
                        batchActionButton(prominent: !allPrepared) {
                            runPopulateAndPrepare(entries: selected.filter { !$0.isEnumerated })
                        } label: {
                            Label(allPrepared ? "Re-prepare \(nPreparable) Compounds" : "Prepare \(nPreparable) Compounds",
                                  systemImage: "wand.and.stars")
                        }
                        .disabled(nPreparable == 0)
                        .help("Add H → adjust protonation → minimize → charges → conformers (re-prepares prepared compounds too)")

                        Button {
                            enumerationSheetEntries = selected.filter { !$0.isEnumerated }
                            showEnumerationSheet = !enumerationSheetEntries.isEmpty
                        } label: {
                            Label("Enumerate \(nPreparable) Compounds…", systemImage: "arrow.triangle.branch")
                        }
                        .buttonStyle(.dsSecondary)
                        .disabled(nPreparable == 0)
                        .help("Expand into tautomers & protomers as separate entries")
                    }
                }

                VStack(alignment: .leading, spacing: DS.Space.s) {
                    PanelSubheader(title: "Docking")
                    batchActionButton(prominent: allPrepared && nDockable > 0, action: useSelectedForDocking) {
                        Label(nDockable > 1 ? "Queue \(nDockable) for Docking"
                              : nDockable == 1 ? "Use for Docking" : "No Prepared Compounds",
                              systemImage: "arrow.right.circle")
                    }
                    .disabled(nDockable == 0)
                    .help("Send the prepared compounds in the selection to docking")
                }

                Divider()

                Button(role: .destructive, action: deleteSelected) {
                    Label("Delete \(selected.count) Compounds", systemImage: "trash")
                }
                .buttonStyle(DSSecondaryButtonStyle(tint: DS.Color.danger))
                .help("Delete the selected compounds and their forms (⌫)")
            }
            .padding(DS.Space.l)
        }
        .scrollIndicators(.automatic)
    }

    /// Full-width action button that takes the primary style when it is the
    /// recommended next step for the selection, the secondary style otherwise.
    @ViewBuilder
    private func batchActionButton<L: View>(
        prominent: Bool,
        action: @escaping () -> Void,
        @ViewBuilder label: () -> L
    ) -> some View {
        if prominent {
            Button(action: action, label: label).buttonStyle(.dsPrimary)
        } else {
            Button(action: action, label: label).buttonStyle(.dsSecondary)
        }
    }
}

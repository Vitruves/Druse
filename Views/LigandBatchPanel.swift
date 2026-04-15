// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Batch Operations Extension
// Batch action panel for multi-selection: Prepare, Enumerate, Dock, Delete.

extension LigandDatabaseWindow {

    // MARK: - Batch Action Panel

    @ViewBuilder
    var batchActionPanel: some View {
        let selected = db.entries.filter { selectedIDs.contains($0.id) }
        let nPrepared = selected.filter(\.isPrepared).count
        let nRaw = selected.filter { !$0.isPrepared }.count
        let nEnumerated = selected.filter(\.isEnumerated).count

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(selected.count) entries selected", systemImage: "checkmark.square.fill")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Deselect All") { selectedIDs.removeAll() }
                    .font(.system(size: 9))
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                if nPrepared > 0 {
                    Text("\(nPrepared) prepared").font(.system(size: 10)).foregroundStyle(.green)
                }
                if nRaw > 0 {
                    Text("\(nRaw) raw").font(.system(size: 10)).foregroundStyle(.secondary)
                }
                if nEnumerated > 0 {
                    Text("\(nEnumerated) enumerated").font(.system(size: 10)).foregroundStyle(.cyan)
                }
            }

            if isBatchProcessing {
                VStack(spacing: 4) {
                    ProgressView(value: Double(batchProgress.current), total: Double(max(batchProgress.total, 1)))
                        .controlSize(.small)
                    Text(processingMessage.isEmpty
                         ? "Processing \(batchProgress.current)/\(batchProgress.total)..."
                         : processingMessage)
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Prepare — dominant form only (no tautomer changes)
            if isBatchProcessing {
                Button(action: { cancelPopulateAndPrepare() }) {
                    Label("Stop Processing", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.regular)
            } else {
                Button(action: {
                    let toPrepare = selected.filter { !$0.isEnumerated }
                    runPopulateAndPrepare(entries: toPrepare)
                }) {
                    let nPreparable = selected.filter { !$0.isEnumerated }.count
                    Label("Prepare \(nPreparable) Entries", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(selected.allSatisfy(\.isEnumerated))
                .help("Add H → minimize → charges → conformers (re-prepares already prepared entries too)")

                Button(action: {
                    let toEnumerate = selected.filter { !$0.isEnumerated }
                    runEnumerate(entries: toEnumerate)
                }) {
                    Label("Enumerate \(selected.filter { !$0.isEnumerated }.count) Entries",
                          systemImage: "arrow.triangle.branch")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .disabled(selected.allSatisfy(\.isEnumerated))
                .help("Expand into all tautomers & protomers as separate entries")
            }

            Divider()

            Button(action: { useSelectedForDocking() }) {
                Label(nPrepared > 1
                      ? "Dock \(nPrepared) Entries"
                      : nPrepared == 1 ? "Use for Docking" : "No Prepared Entries",
                      systemImage: "arrow.right.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(nPrepared == 0)

            Button(action: { deleteSelected() }) {
                Label("Delete Selected (\(selected.count))", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .foregroundStyle(.red)
        }
        .padding()
    }
}

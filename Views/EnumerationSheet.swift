// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Enumeration Sheet
//
// Modal sheet that exposes the full Populate & Prepare pipeline configuration
// (pH, pKa method, max tautomers/protomers, energy cutoff, min population) and
// runs `runEnumerate` on the currently-selected entries. Replaces the bottom
// "Populate & Prepare" tab — config now lives where the user can see and tune
// it without scrolling, and the action button sits next to the sliders that
// drive its output.

struct EnumerationSheet: View {
    @Binding var ph: Double
    @Binding var pkaMethod: PKaMethod
    @Binding var pkaThreshold: Double
    @Binding var numConformers: Int
    @Binding var maxTautomers: Int
    @Binding var maxProtomers: Int
    @Binding var energyCutoff: Double
    @Binding var minPopulation: Double

    let entryCount: Int
    let isProcessing: Bool
    let progressMessage: String
    let progressCurrent: Int
    let progressTotal: Int
    let onRun: () -> Void
    let onCancelRun: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "wand.and.stars")
                    .font(.title2)
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Enumerate")
                        .font(.title3.weight(.semibold))
                    Text("\(entryCount) \(entryCount == 1 ? "molecule" : "molecules") selected")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .disabled(isProcessing)
            }

            Text("Pipeline: add polar H → MMFF94 minimize → Gasteiger charges → enumerate tautomers & protomers at the target pH → generate conformers → filter by Boltzmann population. Each form is inserted as a new entry below the original molecule.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            // Configuration grid
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                GridRow {
                    Text("Target pH").font(.footnote).frame(width: 110, alignment: .leading)
                    Slider(value: $ph, in: 1...14, step: 0.1).controlSize(.small)
                    Text(String(format: "%.1f", ph))
                        .font(.footnote.monospaced()).frame(width: 40, alignment: .trailing)
                }
                GridRow {
                    Text("pKa method").font(.footnote).frame(width: 110, alignment: .leading)
                    Picker("", selection: $pkaMethod) {
                        ForEach(PKaMethod.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .help(pkaMethod.description)
                    Text("").frame(width: 40)
                }
                GridRow {
                    Text("pKa threshold").font(.footnote).frame(width: 110, alignment: .leading)
                    Slider(value: $pkaThreshold, in: 0.5...5.0, step: 0.5).controlSize(.small)
                    Text(String(format: "%.1f", pkaThreshold))
                        .font(.footnote.monospaced()).frame(width: 40, alignment: .trailing)
                }
                GridRow {
                    Text("Conformers/form").font(.footnote).frame(width: 110, alignment: .leading)
                    Stepper("\(numConformers)", value: $numConformers, in: 1...100, step: 5)
                        .controlSize(.small)
                    Text("").frame(width: 40)
                }
                GridRow {
                    Text("Max tautomers").font(.footnote).frame(width: 110, alignment: .leading)
                    Stepper("\(maxTautomers)", value: $maxTautomers, in: 1...50)
                        .controlSize(.small)
                    Text("").frame(width: 40)
                }
                GridRow {
                    Text("Max protomers").font(.footnote).frame(width: 110, alignment: .leading)
                    Stepper("\(maxProtomers)", value: $maxProtomers, in: 1...20)
                        .controlSize(.small)
                    Text("").frame(width: 40)
                }
                GridRow {
                    Text("Energy cutoff").font(.footnote).frame(width: 110, alignment: .leading)
                    Slider(value: $energyCutoff, in: 5...50, step: 1).controlSize(.small)
                    Text(String(format: "%.0f", energyCutoff))
                        .font(.footnote.monospaced()).frame(width: 40, alignment: .trailing)
                }
                GridRow {
                    Text("Min population").font(.footnote).frame(width: 110, alignment: .leading)
                    Slider(value: $minPopulation, in: 0...20, step: 0.5).controlSize(.small)
                    Text(String(format: "%.1f%%", minPopulation))
                        .font(.footnote.monospaced()).frame(width: 50, alignment: .trailing)
                }
            }

            Text(minPopulation <= 0
                 ? "All generated forms are kept; max protomers still limits trace states."
                 : "Forms with Boltzmann population below \(String(format: "%.1f%%", minPopulation)) are discarded.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Divider()

            // Progress (only while running)
            if isProcessing {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(progressMessage.isEmpty
                             ? "Processing \(progressCurrent)/\(progressTotal)..."
                             : progressMessage)
                            .font(.footnote)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    if progressTotal > 0 {
                        ProgressView(value: Double(progressCurrent), total: Double(progressTotal))
                            .controlSize(.small)
                    }
                }
                .padding(8)
                .background(Color.secondary.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            // Action buttons
            HStack {
                Button("Cancel") {
                    if isProcessing {
                        onCancelRun()
                    } else {
                        dismiss()
                    }
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button(action: onRun) {
                    Label(isProcessing ? "Running..." : "Run Enumeration",
                          systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(isProcessing || entryCount == 0)
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}

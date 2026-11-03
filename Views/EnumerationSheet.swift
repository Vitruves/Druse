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
        DSSheetScaffold(
            title: "Enumerate Tautomers & Protomers",
            subtitle: "\(entryCount) \(entryCount == 1 ? "molecule" : "molecules") selected. Each kept form is inserted as a new entry below its parent.",
            systemImage: "arrow.triangle.branch"
        ) {
            Form {
                Section {
                    sliderRow("Target pH", value: $ph, in: 1...14, step: 0.1, format: "%.1f")
                    Picker("pKa method", selection: $pkaMethod) {
                        ForEach(PKaMethod.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    sliderRow("pKa threshold", value: $pkaThreshold, in: 0.5...5.0, step: 0.5, format: "%.1f")
                } header: {
                    Text("Protonation")
                } footer: {
                    Text(pkaMethod.description)
                }

                Section("Enumeration") {
                    stepperRow("Max tautomers", value: $maxTautomers, in: 1...50)
                    stepperRow("Max protomers", value: $maxProtomers, in: 1...20)
                    sliderRow("Energy cutoff", value: $energyCutoff, in: 5...50, step: 1, format: "%.0f kcal/mol")
                }

                Section {
                    stepperRow("Conformers per form", value: $numConformers, in: 1...100, step: 5)
                    sliderRow("Min population", value: $minPopulation, in: 0...20, step: 0.5, format: "%.1f%%")
                } header: {
                    Text("Conformers & Filtering")
                } footer: {
                    Text(minPopulation <= 0
                         ? "All generated forms are kept; max protomers still limits trace states."
                         : "Forms with a Boltzmann population below \(String(format: "%.1f%%", minPopulation)) are discarded (the dominant form is always kept).")
                }

                Section {
                    Text("Add polar H → MMFF94 minimize → Gasteiger charges → enumerate tautomers & protomers at the target pH → generate conformers → filter by Boltzmann population.")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } header: {
                    Text("Pipeline")
                }

                if isProcessing {
                    Section("Progress") {
                        VStack(alignment: .leading, spacing: DS.Space.s) {
                            if progressTotal > 0 {
                                ProgressView(value: Double(progressCurrent), total: Double(progressTotal))
                            } else {
                                ProgressView().controlSize(.small)
                            }
                            Text(progressMessage.isEmpty
                                 ? "Processing \(progressCurrent)/\(progressTotal)…"
                                 : progressMessage)
                                .font(DS.Font.secondary)
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }
            }
            .formStyle(.grouped)
        } footer: {
            if isProcessing {
                Button("Close") { dismiss() }
                    .help("Close this sheet; enumeration keeps running (progress in the status bar)")
            }

            Button(isProcessing ? "Stop" : "Cancel") {
                if isProcessing {
                    onCancelRun()
                } else {
                    dismiss()
                }
            }
            .keyboardShortcut(.cancelAction)

            Button(action: onRun) {
                Label(isProcessing ? "Running…" : "Run Enumeration", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
            .disabled(isProcessing || entryCount == 0)
        }
        .frame(minWidth: 460, idealWidth: 520, minHeight: 520, idealHeight: 660)
    }

    private func sliderRow(_ title: String, value: Binding<Double>, in range: ClosedRange<Double>,
                           step: Double, format: String) -> some View {
        LabeledContent(title) {
            HStack(spacing: DS.Space.m) {
                Slider(value: value, in: range, step: step)
                    .labelsHidden()
                    .controlSize(.small)
                    .frame(minWidth: 140)
                Text(String(format: format, value.wrappedValue))
                    .font(DS.Font.mono)
                    .monospacedDigit()
                    .frame(minWidth: DS.Size.numberField, alignment: .trailing)
            }
        }
    }

    private func stepperRow(_ title: String, value: Binding<Int>, in range: ClosedRange<Int>,
                            step: Int = 1) -> some View {
        LabeledContent(title) {
            Stepper(value: value, in: range, step: step) {
                Text("\(value.wrappedValue)")
                    .font(DS.Font.mono)
                    .monospacedDigit()
            }
        }
    }
}

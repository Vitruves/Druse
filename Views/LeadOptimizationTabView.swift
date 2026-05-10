// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

struct LeadOptimizationTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            if !viewModel.leadOpt.hasReference {
                emptyStateCard
            } else {
                referenceCard
                generationCard
                admetFiltersCard
                generateRunCard
                if !viewModel.leadOpt.analogs.isEmpty {
                    analogsCard
                    dockingControlsCard
                }
                if viewModel.leadOpt.selectedAnalogIndex != nil {
                    comparisonCard
                }
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .sheet(isPresented: Binding(
            get: { viewModel.leadOpt.showComparison },
            set: { viewModel.leadOpt.showComparison = $0 }
        )) {
            if let idx = viewModel.leadOpt.selectedAnalogIndex,
               idx < viewModel.leadOpt.analogs.count {
                LeadOptComparisonView(
                    reference: viewModel.leadOpt,
                    analog: viewModel.leadOpt.analogs[idx]
                )
                .environment(viewModel)
                .frame(minWidth: 500, minHeight: 450)
            }
        }
    }

    // MARK: - Empty state

    @ViewBuilder
    private var emptyStateCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.triangle.branch")
                    .font(.system(size: 18))
                    .foregroundStyle(.secondary)
                Text("No Reference Ligand")
                    .font(PanelStyle.titleFont)
                    .foregroundStyle(.secondary)
            }
            Text("To start lead optimization:")
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) {
                stepRow(number: "1", text: "Run docking in the Docking tab")
                stepRow(number: "2", text: "Open the Results tab")
                stepRow(number: "3", text: "Click the Optimize action on a pose")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(PanelStyle.cardPadding)
        .background(
            RoundedRectangle(cornerRadius: PanelStyle.cardCornerRadius, style: .continuous)
                .fill(Color.primary.opacity(PanelStyle.cardFillOpacity))
        )
    }

    @ViewBuilder
    private func stepRow(number: String, text: String) -> some View {
        HStack(spacing: 6) {
            Text(number)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundStyle(Color.accentColor)
                .frame(width: 18, height: 18)
                .background(Circle().fill(Color.accentColor.opacity(0.15)))
            Text(text)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Reference card

    @ViewBuilder
    private var referenceCard: some View {
        let sm = viewModel.docking.scoringMethod
        PanelCard(
            "Reference",
            icon: "target",
            accessory: {
                if let refResult = viewModel.leadOpt.referenceResult {
                    HStack(alignment: .firstTextBaseline, spacing: 2) {
                        Text(String(format: "%.1f", refResult.displayScore(method: sm)))
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                        Text(sm.unitLabel)
                            .font(PanelStyle.captionFont)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                PanelHighlightRow(color: .accentColor) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(viewModel.leadOpt.referenceName)
                            .font(PanelStyle.bodyFont.weight(.medium))
                            .lineLimit(1)
                        Text(viewModel.leadOpt.referenceSMILES.prefix(60)
                                + (viewModel.leadOpt.referenceSMILES.count > 60 ? "…" : ""))
                            .font(PanelStyle.monoSmall)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                if let desc = viewModel.leadOpt.referenceDescriptors {
                    HStack(spacing: 6) {
                        PanelStat(label: "MW", value: String(format: "%.0f", desc.molecularWeight))
                        PanelStat(label: "LogP", value: String(format: "%.1f", desc.logP))
                        PanelStat(label: "TPSA", value: String(format: "%.0f", desc.tpsa))
                        PanelStat(label: "RotB", value: "\(desc.rotatableBonds)")
                    }
                    if desc.lipinski {
                        HStack {
                            Spacer()
                            Text("✓ Lipinski")
                                .font(.system(size: 10, weight: .semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.green.opacity(0.2)))
                                .foregroundStyle(.green)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Generation card

    @ViewBuilder
    private var generationCard: some View {
        @Bindable var vm = viewModel
        PanelCard("Generation", icon: "wand.and.stars") {
            VStack(alignment: .leading, spacing: 10) {
                PanelLabeledRow("Count") {
                    Picker("", selection: $vm.leadOpt.analogCount) {
                        ForEach([10, 25, 50, 100], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .frame(width: 160)
                }
                PanelSliderRow(
                    label: "Similarity",
                    value: $vm.leadOpt.similarityThreshold,
                    range: 0.3...0.95, step: 0.05,
                    format: { String(format: "%.0f%%", $0 * 100) }
                )
                PanelToggleRow(
                    title: "Keep scaffold",
                    subtitle: "Modify R-groups only",
                    isOn: $vm.leadOpt.keepScaffold
                )

                PanelLabeledDivider(title: "Property Direction", icon: "slider.horizontal.3")
                directionSlider("Polarity",       value: $vm.leadOpt.polarityDirection,       low: "Less",        high: "More")
                directionSlider("Rigidity",       value: $vm.leadOpt.rigidityDirection,       low: "Flexible",    high: "Rigid")
                directionSlider("Lipophilicity",  value: $vm.leadOpt.lipophilicityDirection,  low: "Hydrophilic", high: "Lipophilic")
                directionSlider("Size",           value: $vm.leadOpt.sizeDirection,           low: "Smaller",     high: "Larger")
            }
        }
    }

    @ViewBuilder
    private func directionSlider(_ label: String, value: Binding<Float>, low: String, high: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(directionDescription(value.wrappedValue, low: low, high: high))
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Text(low)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .frame(width: 64, alignment: .leading)
                    .lineLimit(1)
                Slider(value: value, in: -1...1, step: 0.1)
                    .controlSize(.mini)
                Text(high)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .frame(width: 64, alignment: .trailing)
                    .lineLimit(1)
            }
        }
    }

    private func directionDescription(_ value: Float, low: String, high: String) -> String {
        if abs(value) < 0.05 { return "neutral" }
        let pct = Int(abs(value) * 100)
        return value > 0 ? "+\(pct)% \(high.lowercased())" : "+\(pct)% \(low.lowercased())"
    }

    // MARK: - ADMET filters card

    @ViewBuilder
    private var admetFiltersCard: some View {
        @Bindable var vm = viewModel
        PanelCard("ADMET Filters", icon: "shield.checkered") {
            VStack(alignment: .leading, spacing: 8) {
                PanelChoiceGrid(columns: 2) {
                    PanelToggleRow(title: "Lipinski", isOn: $vm.leadOpt.filterLipinski)
                    PanelToggleRow(title: "Veber",    isOn: $vm.leadOpt.filterVeber)
                }
                PanelChoiceGrid(columns: 2) {
                    PanelToggleRow(title: "hERG safe", isOn: $vm.leadOpt.filterHERG)
                    PanelToggleRow(title: "CYP safe",  isOn: $vm.leadOpt.filterCYP)
                }
                PanelSliderRow(
                    label: "Max LogP",
                    value: $vm.leadOpt.maxLogP,
                    range: 1...8, step: 0.5,
                    format: { String(format: "%.1f", $0) }
                )
            }
        }
    }

    // MARK: - Generate run card

    @ViewBuilder
    private var generateRunCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            PanelRunButton(
                title: viewModel.leadOpt.isGenerating ? "Generating…" : "Generate Analogs",
                icon: "sparkles",
                isDisabled: viewModel.leadOpt.isGenerating || !viewModel.leadOpt.hasReference
            ) {
                viewModel.generateOptimizedAnalogs()
            }
            .accessibilityIdentifier(AccessibilityID.leadGenerate)
            if viewModel.leadOpt.isGenerating {
                ProgressView(value: Double(viewModel.leadOpt.generationProgress))
                    .controlSize(.small)
                    .tint(.accentColor)
            }
        }
    }

    // MARK: - Analogs card

    @ViewBuilder
    private var analogsCard: some View {
        let analogs = viewModel.leadOpt.analogs
        let passed = analogs.filter { $0.status != .filtered && $0.status != .failed }.count
        let filtered = analogs.filter { $0.status == .filtered }.count
        let failed = analogs.filter { $0.status == .failed }.count
        PanelCard(
            "Analogs",
            icon: "list.bullet",
            accessory: {
                Button("Clear") {
                    viewModel.leadOpt.analogs.removeAll()
                    viewModel.leadOpt.selectedAnalogIndex = nil
                }
                .font(PanelStyle.smallFont)
                .buttonStyle(.plain)
                .foregroundStyle(.red)
                .accessibilityIdentifier(AccessibilityID.leadClear)
            }
        ) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    PanelStat(label: "Passed",   value: "\(passed)", color: passed > 0 ? .primary : .secondary)
                    PanelStat(label: "Filtered", value: "\(filtered)", color: filtered > 0 ? .orange : .secondary)
                    PanelStat(label: "Failed",   value: "\(failed)",   color: failed > 0 ? .red : .secondary)
                    PanelStat(label: "Total",    value: "\(analogs.count)")
                }
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.primary.opacity(0.04))
                )
                ForEach(Array(analogs.prefix(40).enumerated()), id: \.element.id) { idx, analog in
                    analogRow(index: idx, analog: analog)
                }
                if analogs.count > 40 {
                    Text("+ \(analogs.count - 40) more")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
        }
    }

    @ViewBuilder
    private func analogRow(index: Int, analog: LeadOptAnalog) -> some View {
        let isSelected = viewModel.leadOpt.selectedAnalogIndex == index
        HStack(spacing: 8) {
            Circle()
                .fill(analog.status.color)
                .frame(width: 6, height: 6)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(analog.name)
                        .font(PanelStyle.smallFont.weight(.medium))
                        .lineLimit(1)
                    if let energy = analog.bestEnergy {
                        Text(String(format: "%.1f", energy))
                            .font(PanelStyle.monoSmall.weight(.semibold))
                            .foregroundStyle(energy < -6 ? .green : energy < 0 ? .yellow : .red)
                    }
                }
                HStack(spacing: 6) {
                    if let dMW = analog.deltaMW {
                        deltaLabel("MW", dMW, format: "%+.0f")
                    }
                    if let dLogP = analog.deltaLogP {
                        deltaLabel("LogP", dLogP, format: "%+.1f")
                    }
                    if analog.status == .filtered {
                        Text("filtered")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.orange)
                    }
                }
            }
            Spacer()
            if analog.status == .docked {
                Button(action: {
                    viewModel.leadOpt.selectedAnalogIndex = index
                    viewModel.applyAnalogPose(at: index)
                }) {
                    Text("View")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4, style: .continuous)
                                .fill(Color.accentColor.opacity(0.14))
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.10) : Color.clear)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.leadOpt.selectedAnalogIndex = index
        }
    }

    @ViewBuilder
    private func deltaLabel(_ label: String, _ value: Float, format: String) -> some View {
        let color: Color = value > 0 ? .orange : value < 0 ? .blue : .secondary
        Text("\(label):\(String(format: format, value))")
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(color)
    }

    // MARK: - Docking controls card

    @ViewBuilder
    private var dockingControlsCard: some View {
        let canDockAll = !viewModel.leadOpt.analogs.filter({ $0.status == .generated }).isEmpty
        let prog = viewModel.leadOpt.dockingProgress

        PanelCard("Dock & Deliver", icon: "arrow.triangle.merge") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    PanelSecondaryButton(
                        title: viewModel.leadOpt.isDocking ? "Docking…" : "Dock All Analogs",
                        icon: "arrow.triangle.merge",
                        isDisabled: viewModel.leadOpt.isDocking || !canDockAll,
                        help: "Mini-dock analogs in-place using the reference's grid maps"
                    ) { viewModel.dockAnalogs() }
                    .accessibilityIdentifier(AccessibilityID.leadDockAll)

                    if viewModel.leadOpt.isDocking {
                        PanelSecondaryButton(title: "Stop", icon: "stop.fill", tint: .red) {
                            viewModel.cancelLeadOptimization()
                        }
                        .accessibilityIdentifier(AccessibilityID.leadStop)
                        .frame(width: 76)
                    }
                }
                if viewModel.leadOpt.isDocking {
                    ProgressView(value: Double(prog.current), total: Double(max(prog.total, 1)))
                        .controlSize(.small)
                    Text("Docking \(prog.current)/\(prog.total)")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }
                if viewModel.leadOpt.dockedAnalogCount > 0 {
                    Text("\(viewModel.leadOpt.dockedAnalogCount) analog(s) docked")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }

                PanelLabeledDivider(title: "Send")
                PanelChoiceGrid(columns: 2) {
                    PanelSecondaryButton(
                        title: "To Database",
                        icon: "tray.and.arrow.down",
                        isDisabled: viewModel.leadOpt.analogs.isEmpty || viewModel.leadOpt.isDocking,
                        help: "Copy these analogs into the main Ligand Database"
                    ) { viewModel.addAnalogsToLigandDatabase() }
                    PanelSecondaryButton(
                        title: "To Docking",
                        icon: "arrow.right.circle",
                        isDisabled: viewModel.leadOpt.analogs.isEmpty || viewModel.leadOpt.isDocking,
                        help: "Add to Database and switch to Docking tab"
                    ) { viewModel.sendAnalogsToDocking() }
                }
            }
        }
    }

    // MARK: - Comparison card

    @ViewBuilder
    private var comparisonCard: some View {
        if let idx = viewModel.leadOpt.selectedAnalogIndex,
           idx < viewModel.leadOpt.analogs.count {
            let analog = viewModel.leadOpt.analogs[idx]
            let refEnergy = viewModel.leadOpt.referenceResult?.energy
            let sm = viewModel.docking.scoringMethod

            PanelCard(
                "Comparison",
                icon: "arrow.left.arrow.right",
                accessory: {
                    PanelSecondaryButton(
                        title: "Full",
                        icon: "rectangle.expand.vertical",
                        isDisabled: analog.status != .docked
                    ) {
                        viewModel.leadOpt.showComparison = true
                    }
                    .accessibilityIdentifier(AccessibilityID.leadComparison)
                    .frame(width: 70)
                }
            ) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 12) {
                        comparisonColumn(title: "Reference", value: refEnergy, color: .secondary, unit: sm.unitLabel)
                        Image(systemName: "arrow.right")
                            .foregroundStyle(.secondary)
                        comparisonColumn(
                            title: analog.name,
                            value: analog.bestEnergy,
                            color: comparisonColor(analog.bestEnergy, ref: refEnergy, isAffinity: sm.isAffinityScore),
                            unit: sm.unitLabel
                        )
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.purple.opacity(0.08))
                    )
                    if let rmsd = analog.rmsdToReference {
                        Text(String(format: "RMSD to reference: %.2f Å", rmsd))
                            .font(PanelStyle.smallFont)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func comparisonColumn(title: String, value: Float?, color: Color, unit: String) -> some View {
        VStack(alignment: .center, spacing: 2) {
            Text(title)
                .font(PanelStyle.captionFont.weight(.medium))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            if let v = value {
                Text(String(format: "%.1f", v))
                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                    .foregroundStyle(color)
            } else {
                Text("—")
                    .font(.system(size: 16, weight: .regular, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Text(unit)
                .font(PanelStyle.captionFont)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func comparisonColor(_ value: Float?, ref: Float?, isAffinity: Bool) -> Color {
        guard let v = value, let r = ref else { return .secondary }
        if isAffinity {
            return v > r ? .green : .orange
        }
        return v < r ? .green : .orange
    }
}

// MARK: - Status color

extension LeadOptAnalog.Status {
    var color: Color {
        switch self {
        case .generated: .gray
        case .prepared:  .blue
        case .docked:    .green
        case .filtered:  .orange
        case .failed:    .red
        }
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

/// Step 7 — Lead Optimization: generate analogs of a docked reference pose,
/// filter them on ADMET rules, mini-dock them and hand them back to the pipeline.
struct LeadOptimizationTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow

    /// Rows shown in the analog list until "Show all" is clicked.
    private static let visibleAnalogLimit = 40

    @State private var showAllAnalogs = false

    var body: some View {
        if viewModel.leadOpt.hasReference {
            DSStepScaffold {
                referenceCard
                generationCard
                admetFiltersCard
                // Results follow the settings that produced them.
                if !viewModel.leadOpt.analogs.isEmpty {
                    analogsCard
                    dockAndDeliverCard
                }
            } footer: {
                footer
            }
            .sheet(isPresented: Binding(
                get: { viewModel.leadOpt.showComparison },
                set: { viewModel.leadOpt.showComparison = $0 }
            )) {
                if let analog = selectedAnalog {
                    LeadOptComparisonView(reference: viewModel.leadOpt, analog: analog)
                        .environment(viewModel)
                        .frame(minWidth: 560, idealWidth: 640, minHeight: 480, idealHeight: 620)
                }
            }
        } else {
            // Point at the step that unblocks Lead Opt: Results once poses exist, Docking before that.
            let hasPoses = !viewModel.docking.dockingResults.isEmpty
            let target: SidebarTab = hasPoses ? .results : .dock
            DSStepScaffold {
                DSEmptyState(
                    "No Reference Yet",
                    systemImage: "arrow.triangle.branch",
                    description: hasPoses
                        ? "Pick a docked pose in Results and click its Optimize button to generate and rank analogs."
                        : "Lead optimization starts from a docked pose. Run docking, then pick a pose in Results to optimize."
                ) {
                    Button {
                        target.request(in: viewModel)
                    } label: {
                        Label("Go to \(target.rawValue)", systemImage: target.icon)
                    }
                    .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                }
                .padding(.top, DS.Space.xxl)
            }
        }
    }

    private var selectedAnalog: LeadOptAnalog? {
        guard let idx = viewModel.leadOpt.selectedAnalogIndex,
              idx < viewModel.leadOpt.analogs.count else { return nil }
        return viewModel.leadOpt.analogs[idx]
    }

    // MARK: - Reference

    @ViewBuilder
    private var referenceCard: some View {
        let lead = viewModel.leadOpt
        PanelCard(
            "Reference",
            icon: "target",
            accessory: {
                if let score = lead.referenceScore {
                    DSScoreBadge(score, method: lead.scoringMethod)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                HStack(alignment: .top, spacing: DS.Space.m) {
                    Circle()
                        .fill(DS.Color.ligand)
                        .frame(width: 8, height: 8)
                        .padding(.top, DS.Space.xs)
                    VStack(alignment: .leading, spacing: DS.Space.xxs) {
                        Text(lead.referenceName)
                            .font(DS.Font.label.weight(.medium))
                            .lineLimit(1)
                        Text(lead.referenceSMILES)
                            .font(DS.Font.monoSmall)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                            .help(lead.referenceSMILES)
                    }
                    Spacer(minLength: 0)
                    if !lead.referenceSMILES.isEmpty {
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(lead.referenceSMILES, forType: .string)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .font(DS.Font.secondary.weight(.medium))
                                .frame(width: DS.Size.rowHeight, height: DS.Size.rowHeight)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .help("Copy the reference SMILES")
                        .accessibilityLabel("Copy reference SMILES")
                    }
                }

                if let desc = lead.referenceDescriptors {
                    DSMetricGrid(columns: 4) {
                        DSMetric("MW", value: String(format: "%.1f", desc.molecularWeight))
                        DSMetric("LogP", value: String(format: "%.1f", desc.logP))
                        DSMetric("TPSA", value: String(format: "%.0f", desc.tpsa))
                        DSMetric("RotB", value: "\(desc.rotatableBonds)")
                    }
                    HStack(spacing: DS.Space.xs) {
                        if desc.lipinski {
                            DSTag("Lipinski", tone: .success, systemImage: "checkmark")
                        } else {
                            DSTag("Lipinski violation", tone: .warning, systemImage: "exclamationmark.triangle")
                        }
                        DSTag("HBD \(desc.hbd)", monospaced: true)
                        DSTag("HBA \(desc.hba)", monospaced: true)
                        Spacer(minLength: 0)
                    }
                }
            }
        }
    }

    // MARK: - Generation

    @ViewBuilder
    private var generationCard: some View {
        @Bindable var vm = viewModel
        PanelCard("Generation", icon: "sparkles") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                PanelLabeledRow("Count") {
                    Picker("Count", selection: $vm.leadOpt.analogCount) {
                        ForEach([10, 25, 50, 100], id: \.self) { Text("\($0)").monospacedDigit().tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity)
                }
                PanelSliderRow(
                    label: "Similarity",
                    value: $vm.leadOpt.similarityThreshold,
                    range: 0.3...0.95, step: 0.05,
                    format: { String(format: "%.0f%%", $0 * 100) }
                )
                HStack(alignment: .center, spacing: DS.Space.m) {
                    VStack(alignment: .leading, spacing: DS.Space.xxs) {
                        Text("Keep scaffold")
                            .font(DS.Font.label)
                        Text("Modify R-groups only")
                            .font(DS.Font.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("Keep scaffold", isOn: $vm.leadOpt.keepScaffold)
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                        .labelsHidden()
                        .accessibilityIdentifier(AccessibilityID.leadScaffoldToggle)
                }

                PanelLabeledDivider(title: "Property Direction", icon: "slider.horizontal.3")
                directionSlider("Polarity",      value: $vm.leadOpt.polarityDirection,      low: "Less",        high: "More")
                directionSlider("Rigidity",      value: $vm.leadOpt.rigidityDirection,      low: "Flexible",    high: "Rigid")
                directionSlider("Lipophilicity", value: $vm.leadOpt.lipophilicityDirection, low: "Hydrophilic", high: "Lipophilic")
                directionSlider("Size",          value: $vm.leadOpt.sizeDirection,          low: "Smaller",     high: "Larger")
            }
        }
    }

    @ViewBuilder
    private func directionSlider(_ label: String, value: Binding<Float>, low: String, high: String) -> some View {
        let isNeutral = abs(value.wrappedValue) < 0.05
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack(alignment: .firstTextBaseline) {
                Text(label)
                    .font(DS.Font.label)
                Spacer()
                Text(directionDescription(value.wrappedValue, low: low, high: high))
                    .font(DS.Font.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(isNeutral ? AnyShapeStyle(.tertiary) : AnyShapeStyle(DS.Color.accent))
            }
            Slider(value: snappedSliderBinding(value, range: -1...1, step: 0.1), in: -1...1)
                .controlSize(.small)
                .accessibilityLabel(label)
                .accessibilityValue(directionDescription(value.wrappedValue, low: low, high: high))
            HStack {
                Text(low)
                Spacer()
                Text(high)
            }
            .font(DS.Font.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .help("\(label): drag toward \(low.lowercased()) or \(high.lowercased()); centre is neutral")
    }

    private func directionDescription(_ value: Float, low: String, high: String) -> String {
        if abs(value) < 0.05 { return "Neutral" }
        let pct = Int((abs(value) * 100).rounded())
        return "+\(pct)% \((value > 0 ? high : low).lowercased())"
    }

    // MARK: - ADMET filters

    @ViewBuilder
    private var admetFiltersCard: some View {
        @Bindable var vm = viewModel
        PanelCard("ADMET Filters", icon: "shield.checkered") {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    PanelToggleRow(title: "Lipinski", subtitle: "Rule of five: MW, LogP, HBD, HBA",
                                   isOn: $vm.leadOpt.filterLipinski)
                    PanelToggleRow(title: "Veber", subtitle: "Rotatable bonds and TPSA (oral bioavailability)",
                                   isOn: $vm.leadOpt.filterVeber)
                    PanelToggleRow(title: "hERG safe", subtitle: "Drop analogs with predicted hERG liability",
                                   isOn: $vm.leadOpt.filterHERG)
                    PanelToggleRow(title: "CYP safe", subtitle: "Drop predicted CYP2D6 / CYP3A4 inhibitors",
                                   isOn: $vm.leadOpt.filterCYP)
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

    // MARK: - Analogs

    @ViewBuilder
    private var analogsCard: some View {
        let analogs = viewModel.leadOpt.analogs
        let passed = viewModel.leadOpt.passedFilterCount
        let filtered = analogs.filter { $0.status == .filtered }.count
        let failed = analogs.filter { $0.status == .failed }.count
        let refScore = viewModel.leadOpt.referenceScore

        PanelCard(
            "Analogs",
            icon: "list.bullet",
            accessory: { DSTag("\(analogs.count)", monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                DSMetricGrid(columns: 3) {
                    DSMetric("Passed", value: "\(passed)", tone: passed > 0 ? .success : .neutral)
                    DSMetric("Filtered", value: "\(filtered)", tone: filtered > 0 ? .warning : .neutral)
                    DSMetric("Failed", value: "\(failed)", tone: failed > 0 ? .danger : .neutral)
                }

                let visible = showAllAnalogs ? analogs.count : Self.visibleAnalogLimit
                VStack(spacing: DS.Space.xxs) {
                    ForEach(Array(analogs.prefix(visible).enumerated()), id: \.element.id) { idx, analog in
                        analogRow(index: idx, analog: analog, refScore: refScore)
                    }
                    if analogs.count > Self.visibleAnalogLimit {
                        Button(showAllAnalogs
                               ? "Show top \(Self.visibleAnalogLimit)"
                               : "Show all \(analogs.count) analogs") {
                            showAllAnalogs.toggle()
                        }
                        .buttonStyle(.borderless)
                        .font(DS.Font.secondary.weight(.medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, DS.Space.xs)
                    }
                }

                if let analog = selectedAnalog {
                    selectionSummary(analog: analog, refScore: refScore)
                }
            }
        }
    }

    @ViewBuilder
    private func analogRow(index: Int, analog: LeadOptAnalog, refScore: Float?) -> some View {
        let isSelected = viewModel.leadOpt.selectedAnalogIndex == index
        let method = viewModel.leadOpt.scoringMethod
        HStack(alignment: .center, spacing: DS.Space.m) {
            Circle()
                .fill(analog.status.tone.color)
                .frame(width: 8, height: 8)
                .help(analog.status.label)
                .accessibilityLabel(analog.status.label)

            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text(analog.name)
                    .font(DS.Font.label.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: DS.Space.s) {
                    if let meta = propertyDeltaText(analog) {
                        Text(meta)
                            .font(DS.Font.monoSmall)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .help("Change versus the reference: molecular weight (Da) and LogP")
                    }
                    if analog.status == .filtered || analog.status == .failed {
                        DSTag(analog.status.label, tone: analog.status.tone)
                    }
                }
            }

            Spacer(minLength: DS.Space.xs)

            if let score = analog.bestScore {
                VStack(alignment: .trailing, spacing: DS.Space.xxs) {
                    DSScoreBadge(score, method: method)
                    if let ref = refScore {
                        deltaScoreText(score, reference: ref, method: method)
                    }
                }
            }

            if analog.status != .failed {
                let isDocked = analog.status == .docked
                Button {
                    viewModel.leadOpt.selectedAnalogIndex = index
                    if isDocked { viewModel.applyAnalogPose(at: index) }
                } label: {
                    Image(systemName: "eye")
                        .font(DS.Font.secondary.weight(.medium))
                        .frame(width: DS.Size.rowHeight, height: DS.Size.rowHeight)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .foregroundStyle(isSelected ? AnyShapeStyle(DS.Color.accent) : AnyShapeStyle(.secondary))
                .help(isDocked
                      ? "Show the docked pose of \(analog.name) as an overlay in the viewer"
                      : "Show the structure of \(analog.name) (dock it to see its pose)")
                .accessibilityLabel("View \(analog.name)")
                .accessibilityIdentifier(AccessibilityID.leadViewAnalog(index))
            }
        }
        .padding(.vertical, DS.Space.s)
        .padding(.horizontal, DS.Space.m)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(isSelected ? DS.Color.accentSoft : Color.clear)
        )
        .contentShape(Rectangle())
        .onTapGesture { viewModel.leadOpt.selectedAnalogIndex = index }
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// Δ vs the reference in the method's units: improvement → success, regression → scorePoor.
    @ViewBuilder
    private func deltaScoreText(_ score: Float, reference: Float, method: ScoringMethod) -> some View {
        let delta = score - reference
        let color: Color = abs(delta) <= 0.05 ? .secondary
            : method.ranks(score, before: reference) ? DS.Color.success : DS.Color.scorePoor
        Text("Δ \(String(format: "%+.1f", delta))")
            .font(DS.Font.monoSmall)
            .monospacedDigit()
            .foregroundStyle(color)
            .accessibilityLabel("Change versus reference \(String(format: "%+.1f", delta)) \(method.unitLabel)")
    }

    /// "MW +16 · LogP −0.3": property deltas carry no good/bad meaning, so they stay neutral.
    private func propertyDeltaText(_ analog: LeadOptAnalog) -> String? {
        var parts: [String] = []
        if let dMW = analog.deltaMW { parts.append("MW \(String(format: "%+.0f", dMW))") }
        if let dLogP = analog.deltaLogP { parts.append("LogP \(String(format: "%+.1f", dLogP))") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func selectionSummary(analog: LeadOptAnalog, refScore: Float?) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelLabeledDivider(title: "Selected Analog", icon: "hexagon")
            LigandDepictionView(smiles: analog.smiles)
                .frame(height: 170)
                .frame(maxWidth: .infinity)
            Text(analog.smiles)
                .font(DS.Font.monoSmall)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
                .textSelection(.enabled)
            PanelLabeledDivider(title: "Selected vs Reference", icon: "arrow.left.arrow.right")
            HStack(spacing: DS.Space.m) {
                scoreColumn(title: "Reference", value: refScore, tone: .neutral)
                Image(systemName: "arrow.right")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.tertiary)
                scoreColumn(
                    title: analog.name,
                    value: analog.bestScore,
                    tone: comparisonTone(analog.bestScore, ref: refScore)
                )
            }
        }
    }

    @ViewBuilder
    private func scoreColumn(title: String, value: Float?, tone: DSTone) -> some View {
        DSMetric(
            title,
            value: value.map { String(format: "%.1f", $0) } ?? "—",
            unit: value == nil ? nil : viewModel.leadOpt.scoringMethod.unitLabel,
            tone: tone
        )
    }

    private func comparisonTone(_ value: Float?, ref: Float?) -> DSTone {
        guard let v = value, let r = ref else { return .neutral }
        return viewModel.leadOpt.scoringMethod.ranks(v, before: r) ? .success : .warning
    }

    // MARK: - Dock & Deliver

    @ViewBuilder
    private var dockAndDeliverCard: some View {
        let lead = viewModel.leadOpt
        let canDockAll = lead.analogs.contains { $0.status == .generated }
        let prog = lead.dockingProgress

        PanelCard(
            "Dock & Deliver",
            icon: "paperplane",
            accessory: {
                if lead.dockedAnalogCount > 0 {
                    DSTag("\(lead.dockedAnalogCount) docked", tone: .success, monospaced: true)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if !canDockAll && !lead.isDocking && lead.dockedAnalogCount > 0 {
                    PanelHint(text: "Every analog that passed the filters is docked. Generate new analogs to dock more.",
                              icon: "checkmark.circle.fill", color: DS.Color.success)
                } else {
                HStack(spacing: DS.Space.s) {
                    PanelSecondaryButton(
                        title: lead.isDocking ? "Docking…" : "Dock All Analogs",
                        icon: "arrow.down.right.and.arrow.up.left",
                        isDisabled: lead.isDocking || !canDockAll,
                        help: "Mini-dock analogs into the reference pocket with the reference's scoring method"
                    ) { viewModel.dockAnalogs() }
                    .accessibilityIdentifier(AccessibilityID.leadDockAll)

                    if lead.isDocking {
                        PanelSecondaryButton(
                            title: "Stop",
                            icon: "stop.fill",
                            tint: DS.Color.danger,
                            help: "Stop analog docking"
                        ) { viewModel.cancelLeadOptimization() }
                        .accessibilityIdentifier(AccessibilityID.leadStop)
                        .fixedSize()
                    }
                }
                }

                if lead.isDocking {
                    VStack(alignment: .leading, spacing: DS.Space.xs) {
                        ProgressView(value: Double(prog.current), total: Double(max(prog.total, 1)))
                            .controlSize(.small)
                        Text("Docking \(prog.current) of \(prog.total)")
                            .font(DS.Font.secondary)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }

                PanelLabeledDivider(title: "Send")
                PanelChoiceGrid(columns: 2) {
                    PanelSecondaryButton(
                        title: "Ligand Database",
                        icon: "tray.full",
                        help: "Open the Ligand Database, where generated analogs are added automatically"
                    ) { openWindow(id: "ligand-database") }
                    PanelSecondaryButton(
                        title: "To Docking",
                        icon: "arrow.right.circle",
                        isDisabled: lead.isDocking,
                        help: "Add to the Ligand Database and switch to the Docking step"
                    ) { viewModel.sendAnalogsToDocking() }
                }
            }
        }
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        let lead = viewModel.leadOpt
        PanelRunButton(
            title: lead.isGenerating ? "Generating…" : "Generate Analogs",
            icon: "sparkles",
            isDisabled: lead.isGenerating
        ) {
            viewModel.generateOptimizedAnalogs()
        }
        .accessibilityIdentifier(AccessibilityID.leadGenerate)

        if lead.isGenerating {
            ProgressView(value: Double(lead.generationProgress))
                .controlSize(.small)
        }

        if !lead.analogs.isEmpty {
            HStack(spacing: DS.Space.s) {
                PanelSecondaryButton(
                    title: "Compare with Reference…",
                    icon: "arrow.left.arrow.right",
                    isDisabled: selectedAnalog == nil,
                    help: selectedAnalog == nil
                        ? "Select an analog to compare it with the reference"
                        : "Open the full reference vs analog comparison"
                ) {
                    viewModel.leadOpt.showComparison = true
                }
                .accessibilityIdentifier(AccessibilityID.leadComparison)

                Menu {
                    Button("Clear All Analogs", role: .destructive) {
                        viewModel.leadOpt.analogs.removeAll()
                        viewModel.leadOpt.selectedAnalogIndex = nil
                    }
                    .disabled(lead.isGenerating || lead.isDocking)
                    .accessibilityIdentifier(AccessibilityID.leadClear)
                } label: {
                    Image(systemName: "ellipsis")
                        .font(DS.Font.label.weight(.medium))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .frame(width: DS.Size.buttonHeight, height: DS.Size.buttonHeight)
                .background(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).fill(DS.Color.controlFill))
                .overlay(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).strokeBorder(DS.Color.hairline, lineWidth: 0.5))
                .help("More analog actions")
                .accessibilityLabel("More analog actions")
            }
        }
    }
}

// MARK: - Status presentation

private extension LeadOptAnalog.Status {
    var tone: DSTone {
        switch self {
        case .generated: .neutral
        case .prepared:  .info
        case .docked:    .success
        case .filtered:  .warning
        case .failed:    .danger
        }
    }

    var label: String {
        switch self {
        case .generated: "Generated"
        case .prepared:  "Prepared"
        case .docked:    "Docked"
        case .filtered:  "Filtered"
        case .failed:    "Failed"
        }
    }
}

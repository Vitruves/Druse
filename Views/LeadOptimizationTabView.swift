import SwiftUI

struct LeadOptimizationTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if !viewModel.leadOpt.hasReference {
                    emptyState
                } else {
                    referenceSummary
                    Divider()
                    generationControls
                    Divider()
                    admetFilterSection
                    Divider()
                    generateButton
                    if !viewModel.leadOpt.analogs.isEmpty {
                        Divider()
                        analogList
                        Divider()
                        dockingControls
                    }
                    if viewModel.leadOpt.selectedAnalogIndex != nil {
                        Divider()
                        comparisonCard
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(12)
        }
        .sheet(isPresented: Binding(
            get: { viewModel.leadOpt.showComparison },
            set: { viewModel.leadOpt.showComparison = $0 }
        )) {
            if let idx = viewModel.leadOpt.selectedAnalogIndex, idx < viewModel.leadOpt.analogs.count {
                LeadOptComparisonView(
                    reference: viewModel.leadOpt,
                    analog: viewModel.leadOpt.analogs[idx]
                )
                .environment(viewModel)
                .frame(minWidth: 500, minHeight: 450)
            }
        }
    }

    // MARK: - Empty State

    @ViewBuilder
    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Reference Ligand", systemImage: "arrow.triangle.branch")
        } description: {
            Text("Select a docking pose or screening hit in the Results tab and click \"Optimize\" to start lead optimization.")
        }
    }

    // MARK: - Reference Summary

    @ViewBuilder
    private var referenceSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Reference", systemImage: "target")
                .font(.system(size: 12, weight: .semibold))

            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.leadOpt.referenceName)
                        .font(.system(size: 11, weight: .medium))
                    Text(viewModel.leadOpt.referenceSMILES.prefix(50) + (viewModel.leadOpt.referenceSMILES.count > 50 ? "..." : ""))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let energy = viewModel.leadOpt.referenceResult?.energy {
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(String(format: "%.1f", energy))
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                        Text("kcal/mol")
                            .font(.system(size: 8))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.accentColor.opacity(0.06)))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.accentColor.opacity(0.15), lineWidth: 1))

            if let desc = viewModel.leadOpt.referenceDescriptors {
                HStack(spacing: 8) {
                    propertyBadge("MW", String(format: "%.0f", desc.molecularWeight))
                    propertyBadge("LogP", String(format: "%.1f", desc.logP))
                    propertyBadge("TPSA", String(format: "%.0f", desc.tpsa))
                    propertyBadge("RotB", "\(desc.rotatableBonds)")
                    if desc.lipinski {
                        Text("Lipinski")
                            .font(.system(size: 8, weight: .semibold))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Capsule().fill(Color.green.opacity(0.2)))
                            .foregroundStyle(.green)
                    }
                }
            }
        }
    }

    // MARK: - Generation Controls

    @ViewBuilder
    private var generationControls: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 8) {
            Label("Generation", systemImage: "wand.and.stars")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)

            HStack {
                Text("Count")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                Picker("", selection: $vm.leadOpt.analogCount) {
                    ForEach([10, 25, 50, 100], id: \.self) { n in
                        Text("\(n)").tag(n)
                    }
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
            }

            HStack {
                Text("Similarity")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                Slider(value: $vm.leadOpt.similarityThreshold, in: 0.3...0.95, step: 0.05)
                    .controlSize(.mini)
                Text(String(format: "%.0f%%", viewModel.leadOpt.similarityThreshold * 100))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 30)
            }

            Toggle("Keep scaffold (modify R-groups)", isOn: $vm.leadOpt.keepScaffold)
                .toggleStyle(.switch)
                .controlSize(.small)

            Divider()

            // Property direction sliders
            Label("Property Direction", systemImage: "slider.horizontal.3")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)

            directionSlider("Polarity", value: $vm.leadOpt.polarityDirection, low: "Less", high: "More")
            directionSlider("Rigidity", value: $vm.leadOpt.rigidityDirection, low: "Flexible", high: "Rigid")
            directionSlider("Lipophilicity", value: $vm.leadOpt.lipophilicityDirection, low: "Hydrophilic", high: "Lipophilic")
            directionSlider("Size", value: $vm.leadOpt.sizeDirection, low: "Smaller", high: "Larger")
        }
    }

    @ViewBuilder
    private func directionSlider(_ label: String, value: Binding<Float>, low: String, high: String) -> some View {
        VStack(spacing: 1) {
            HStack {
                Text(label)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .frame(width: 70, alignment: .leading)
                Text(low)
                    .font(.system(size: 8))
                    .foregroundStyle(.tertiary)
                Slider(value: value, in: -1...1, step: 0.1)
                    .controlSize(.mini)
                Text(high)
                    .font(.system(size: 8))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - ADMET Filters

    @ViewBuilder
    private var admetFilterSection: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 6) {
            Label("ADMET Filters", systemImage: "shield.checkered")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Toggle("Lipinski", isOn: $vm.leadOpt.filterLipinski)
                    .toggleStyle(.switch).controlSize(.mini)
                Toggle("Veber", isOn: $vm.leadOpt.filterVeber)
                    .toggleStyle(.switch).controlSize(.mini)
            }
            HStack(spacing: 12) {
                Toggle("hERG safe", isOn: $vm.leadOpt.filterHERG)
                    .toggleStyle(.switch).controlSize(.mini)
                Toggle("CYP safe", isOn: $vm.leadOpt.filterCYP)
                    .toggleStyle(.switch).controlSize(.mini)
            }

            HStack {
                Text("Max LogP")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                Slider(value: $vm.leadOpt.maxLogP, in: 1...8, step: 0.5)
                    .controlSize(.mini)
                Text(String(format: "%.1f", viewModel.leadOpt.maxLogP))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 25)
            }
        }
    }

    // MARK: - Generate Button

    @ViewBuilder
    private var generateButton: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: { viewModel.generateOptimizedAnalogs() }) {
                HStack {
                    Label("Generate Analogs", systemImage: "sparkles")
                    Spacer()
                    if viewModel.leadOpt.isGenerating {
                        ProgressView().controlSize(.mini)
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(viewModel.leadOpt.isGenerating || !viewModel.leadOpt.hasReference)

            if viewModel.leadOpt.isGenerating {
                ProgressView(value: Double(viewModel.leadOpt.generationProgress))
                    .controlSize(.small)
            }
        }
    }

    // MARK: - Analog List

    @ViewBuilder
    private var analogList: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                let active = viewModel.leadOpt.analogs.filter { $0.status != .filtered && $0.status != .failed }.count
                let total = viewModel.leadOpt.analogs.count
                Label("\(active) Analogs", systemImage: "list.bullet")
                    .font(.system(size: 11, weight: .semibold))
                if total != active {
                    Text("(\(total - active) filtered)")
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Button("Clear") {
                    viewModel.leadOpt.analogs.removeAll()
                    viewModel.leadOpt.selectedAnalogIndex = nil
                }
                .font(.system(size: 9))
                .buttonStyle(.plain)
                .foregroundStyle(.red)
            }

            ForEach(Array(viewModel.leadOpt.analogs.prefix(40).enumerated()), id: \.element.id) { idx, analog in
                analogRow(index: idx, analog: analog)
            }
            if viewModel.leadOpt.analogs.count > 40 {
                Text("+ \(viewModel.leadOpt.analogs.count - 40) more")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }

    @ViewBuilder
    private func analogRow(index: Int, analog: LeadOptAnalog) -> some View {
        let isSelected = viewModel.leadOpt.selectedAnalogIndex == index

        HStack(spacing: 6) {
            // Status indicator
            Circle()
                .fill(analog.status.color)
                .frame(width: 6, height: 6)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(analog.name)
                        .font(.system(size: 10, weight: .medium))
                        .lineLimit(1)
                    if let energy = analog.bestEnergy {
                        Text(String(format: "%.1f", energy))
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(energy < -6 ? .green : energy < 0 ? .yellow : .red)
                    }
                }
                HStack(spacing: 4) {
                    if let dMW = analog.deltaMW {
                        deltaLabel("MW", dMW, format: "%+.0f")
                    }
                    if let dLogP = analog.deltaLogP {
                        deltaLabel("LogP", dLogP, format: "%+.1f")
                    }
                    if analog.status == .filtered {
                        Text("filtered")
                            .font(.system(size: 8, weight: .medium))
                            .foregroundStyle(.orange)
                    }
                }
            }

            Spacer()

            if analog.status == .docked {
                Button("View") {
                    viewModel.leadOpt.selectedAnalogIndex = index
                    viewModel.applyAnalogPose(at: index)
                }
                .font(.system(size: 9))
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
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
            .font(.system(size: 8, design: .monospaced))
            .foregroundStyle(color)
    }

    // MARK: - Docking Controls

    @ViewBuilder
    private var dockingControls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(action: { viewModel.dockAnalogs() }) {
                    HStack {
                        Label("Dock All Analogs", systemImage: "arrow.triangle.merge")
                        Spacer()
                        if viewModel.leadOpt.isDocking {
                            ProgressView().controlSize(.mini)
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.leadOpt.isDocking || viewModel.leadOpt.analogs.filter({ $0.status == .generated }).isEmpty)

                if viewModel.leadOpt.isDocking {
                    Button("Stop") { viewModel.cancelLeadOptimization() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .tint(.red)
                }
            }

            if viewModel.leadOpt.isDocking {
                let prog = viewModel.leadOpt.dockingProgress
                ProgressView(value: Double(prog.current), total: Double(max(prog.total, 1)))
                    .controlSize(.small)
                Text("Docking \(prog.current)/\(prog.total)")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }

            if viewModel.leadOpt.dockedAnalogCount > 0 {
                Text("\(viewModel.leadOpt.dockedAnalogCount) analog(s) docked")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Quick Comparison Card

    @ViewBuilder
    private var comparisonCard: some View {
        if let idx = viewModel.leadOpt.selectedAnalogIndex, idx < viewModel.leadOpt.analogs.count {
            let analog = viewModel.leadOpt.analogs[idx]
            let refEnergy = viewModel.leadOpt.referenceResult?.energy

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Label("Comparison", systemImage: "arrow.left.arrow.right")
                        .font(.system(size: 11, weight: .semibold))
                    Spacer()
                    Button("Full Comparison") {
                        viewModel.leadOpt.showComparison = true
                    }
                    .font(.system(size: 9))
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(analog.status != .docked)
                }

                HStack(spacing: 16) {
                    VStack(alignment: .center, spacing: 2) {
                        Text("Reference")
                            .font(.system(size: 8, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(String(format: "%.1f", refEnergy ?? 0))
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                        Text("kcal/mol")
                            .font(.system(size: 7))
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity)

                    Image(systemName: "arrow.right")
                        .foregroundStyle(.secondary)

                    VStack(alignment: .center, spacing: 2) {
                        Text(analog.name)
                            .font(.system(size: 8, weight: .medium))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        if let e = analog.bestEnergy {
                            Text(String(format: "%.1f", e))
                                .font(.system(size: 13, weight: .bold, design: .monospaced))
                                .foregroundStyle(e < (refEnergy ?? 0) ? .green : .orange)
                        } else {
                            Text("—")
                                .font(.system(size: 13))
                                .foregroundStyle(.tertiary)
                        }
                        Text("kcal/mol")
                            .font(.system(size: 7))
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity)
                }
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.purple.opacity(0.06)))

                if let rmsd = analog.rmsdToReference {
                    Text("RMSD to reference: \(String(format: "%.2f", rmsd)) Å")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func propertyBadge(_ label: String, _ value: String) -> some View {
        VStack(spacing: 0) {
            Text(value)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
            Text(label)
                .font(.system(size: 7))
                .foregroundStyle(.tertiary)
        }
        .frame(minWidth: 30)
    }
}

// MARK: - Status Color

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

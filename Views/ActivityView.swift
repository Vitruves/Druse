// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Xcode-style activity capsule on the trailing side of the toolbar (just before the
/// Console / Inspector toggles): what is loaded, what is running, how far along it is,
/// and a stop button. The single home for live progress outside the canvas HUD. Click to
/// see the most recent log lines. The shell hands it a `width` budget; when that is narrow the
/// text drops away first (document name, then the operation), leaving the state dot, progress and stop.
struct ActivityView: View {
    let width: CGFloat
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @AppStorage("ui.consoleOpen") private var showConsole = false
    @State private var showRecentLog = false
    private let log = ActivityLog.shared

    var body: some View {
        HStack(spacing: DS.Space.m) {
            Button { showRecentLog.toggle() } label: {
                HStack(spacing: DS.Space.s) {
                    stateDot
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: DS.Space.s) {
                            documentText
                            Text("·").foregroundStyle(.tertiary)
                            operationLabel
                        }
                        operationLabel
                        Color.clear.frame(width: 0)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("\(documentTitle) — \(operationText)\nClick for recent activity")
            .popover(isPresented: $showRecentLog, arrowEdge: .bottom) { recentLogPopover }

            if isBusy {
                if let progressFraction {
                    ProgressView(value: progressFraction)
                        .progressViewStyle(.linear)
                        .tint(DS.Color.accent)
                        .frame(minWidth: 32, idealWidth: 72, maxWidth: 72)
                } else {
                    ProgressView().controlSize(.mini)
                }

                Button(action: stopCurrentOperation) {
                    Image(systemName: "xmark.circle.fill")
                        .font(DS.Font.body)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help(stopButtonHint)
                .accessibilityLabel(stopButtonHint)
                .accessibilityIdentifier(AccessibilityID.statusStop)
            }
        }
        .padding(.horizontal, DS.Space.m)
        .frame(height: DS.Size.toolButton)
        .frame(width: width)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Activity")
    }

    private var documentText: some View {
        Text(documentTitle)
            .font(DS.Font.label.weight(.medium))
            .lineLimit(1)
            .fixedSize()
    }

    private var operationLabel: some View {
        Text(operationText)
            .font(DS.Font.label)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.tail)
            // A small ideal width lets ViewThatFits keep the text (truncated) rather than drop it.
            .frame(minWidth: 60, idealWidth: 60, maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - State

    private var isScreeningActive: Bool {
        if case .idle = viewModel.docking.screeningState { return false }
        return true
    }

    private var isBusy: Bool {
        viewModel.docking.isDocking || viewModel.docking.isBatchDocking || viewModel.molecules.isMinimizing
            || viewModel.workspace.isGeneratingSurface || viewModel.workspace.isLoading || viewModel.workspace.isSearching
            || isScreeningActive || viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking
            || viewModel.workspace.statusMessage.hasSuffix("...")
    }

    private var hasRecentError: Bool {
        !isBusy && log.entries.last?.level == .error
    }

    private var documentTitle: String {
        let names = [viewModel.molecules.protein?.name, viewModel.molecules.ligand?.name].compactMap { $0 }
        return names.isEmpty ? "Druse" : names.joined(separator: " · ")
    }

    private var operationText: String {
        let docking = viewModel.docking
        if docking.isBatchDocking {
            return "Docking ligand \(docking.batchProgress.current + 1) of \(docking.batchProgress.total)"
        }
        if docking.isDocking {
            return "Docking · \(DockingProgress(docking).percentText)"
        }
        if isScreeningActive { return "Virtual screening" }
        if viewModel.leadOpt.isGenerating { return "Generating analogs" }
        if viewModel.leadOpt.isDocking {
            let p = viewModel.leadOpt.dockingProgress
            return "Docking analogs \(p.current)/\(p.total)"
        }
        if viewModel.molecules.isMinimizing { return "Structure cleanup" }
        if viewModel.workspace.isGeneratingSurface { return "Generating surface" }
        if viewModel.workspace.isLoading {
            let message = viewModel.workspace.loadingMessage
            return message.isEmpty ? "Loading" : message
        }
        if viewModel.workspace.isSearching { return "Searching RCSB" }
        let message = viewModel.workspace.statusMessage
        if message != "Ready" { return message }
        return log.entries.last?.message ?? "Ready"
    }

    /// Determinate progress when the running operation reports it, else nil (spinner).
    private var progressFraction: Double? {
        let docking = viewModel.docking
        if docking.isBatchDocking, docking.batchProgress.total > 0 {
            return Double(docking.batchProgress.current) / Double(docking.batchProgress.total)
        }
        if docking.isDocking {
            return DockingProgress(docking).fraction
        }
        if isScreeningActive, docking.screeningProgress > 0 {
            return Double(min(docking.screeningProgress, 1))
        }
        if viewModel.leadOpt.isGenerating, viewModel.leadOpt.generationProgress > 0 {
            return Double(min(viewModel.leadOpt.generationProgress, 1))
        }
        if viewModel.leadOpt.isDocking, viewModel.leadOpt.dockingProgress.total > 0 {
            let p = viewModel.leadOpt.dockingProgress
            return Double(p.current) / Double(p.total)
        }
        return nil
    }

    @ViewBuilder
    private var stateDot: some View {
        let color: Color = hasRecentError ? DS.Color.danger : DS.Color.accent
        Image(systemName: "circle.fill")
            .font(DS.Font.caption)
            .imageScale(.small)
            .foregroundStyle(color)
            .symbolEffect(.pulse, options: .repeating, isActive: isBusy && !reduceMotion)
            .accessibilityLabel(isBusy ? "Busy" : (hasRecentError ? "Error" : "Idle"))
    }

    // MARK: - Recent log popover

    private var recentLogPopover: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Recent Activity")
                    .font(DS.Font.sectionTitle)
                Spacer()
                Button("Show Console") {
                    showConsole = true
                    showRecentLog = false
                }
                .buttonStyle(.link)
                .font(DS.Font.label)
            }
            .padding(DS.Space.l)

            Divider()

            let entries = Array(log.entries.suffix(8))
            if entries.isEmpty {
                Text("No activity yet")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(DS.Space.xl)
            } else {
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    ForEach(entries) { entry in
                        HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
                            Image(systemName: entry.level.icon)
                                .font(DS.Font.caption)
                                .foregroundStyle(entry.level.dsColor)
                            Text(entry.formattedTime)
                                .font(DS.Font.monoSmall)
                                .foregroundStyle(.secondary)
                            Text(entry.message)
                                .font(DS.Font.secondary)
                                .lineLimit(2)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding(DS.Space.l)
            }
        }
        .frame(width: 440)
    }

    // MARK: - Stop

    private var stopButtonHint: String {
        if viewModel.docking.isBatchDocking { return "Stop batch docking" }
        if viewModel.docking.isDocking { return "Stop docking" }
        if isScreeningActive { return "Stop virtual screening" }
        if viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking { return "Stop lead optimization" }
        if viewModel.molecules.isMinimizing { return "Stop structure cleanup" }
        if viewModel.workspace.isGeneratingSurface { return "Stop surface generation" }
        if viewModel.workspace.isLoading { return "Cancel loading" }
        return "Cancel current operation"
    }

    private func stopCurrentOperation() {
        if viewModel.docking.isBatchDocking {
            viewModel.cancelBatchDocking()
        } else if viewModel.docking.isDocking {
            viewModel.stopDocking()
        } else if isScreeningActive {
            viewModel.cancelScreening()
        } else if viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking {
            viewModel.cancelLeadOptimization()
        } else if viewModel.molecules.isMinimizing {
            viewModel.molecules.isMinimizing = false
        } else if viewModel.workspace.isGeneratingSurface {
            viewModel.workspace.isGeneratingSurface = false
        } else if viewModel.workspace.isLoading {
            viewModel.workspace.isLoading = false
        }
        // Reset status for operations tracked only by statusMessage
        if viewModel.workspace.statusMessage.hasSuffix("...") {
            viewModel.workspace.statusMessage = "Cancelled"
        }
    }
}

// MARK: - Docking progress

/// One reading of a single-ligand docking run's progress, shared by the toolbar capsule, the
/// canvas HUD and the Docking step header so they never disagree. The engine reports a global
/// generation index across all runs (`run * generationsPerRun + step`); for multi-run GA searches
/// it is broken back down into run and per-run generation, matching what the user configured
/// (e.g. 300 × 200 × 10) instead of an opaque "1541/2000".
struct DockingProgress {
    /// 1-based global step, clamped to the total.
    let step: Int
    let total: Int
    let summary: String

    var fraction: Double { Double(step) / Double(total) }
    var percentText: String { "\(Int((fraction * 100).rounded(.down)))%" }

    init(_ docking: DockingCoordinator) {
        let total = max(docking.dockingTotalGenerations, 1)
        let step = min(max(docking.dockingGeneration + 1, 1), total)
        self.total = total
        self.step = step

        let config = docking.dockingConfig
        switch config.searchMethod {
        case .diffusionGuided:
            summary = "Denoising step \(step)/\(total)"
        case .fragmentBased:
            summary = "Fragment step \(step)/\(total)"
        default:
            let perRun = config.generationsPerRun
            if config.numRuns > 1, perRun > 0, total == config.numGenerations {
                let run = (step - 1) / perRun + 1
                let generation = (step - 1) % perRun + 1
                summary = "Run \(run)/\(config.numRuns) · generation \(generation)/\(perRun)"
            } else {
                summary = "Generation \(step)/\(total)"
            }
        }
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Demo Overlay
//
// A narrated glass card shown during the guided demo: the pipeline stages as
// progress dots, the current step and its narration, live docking stats, and
// Stop / Explore. It sits at the bottom of the viewport, above the display
// cluster (the docking HUD is hidden while the demo runs).

struct DemoOverlay: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Vertical room left free for the canvas display cluster below the card.
    private static let clusterClearance = DS.Size.toolButton + DS.Space.xs * 2 + DS.Space.l + DS.Space.m

    var body: some View {
        let step = viewModel.demoStep
        let isRunning = viewModel.isDemoRunning
        let isComplete = step == .complete

        VStack(spacing: 0) {
            Spacer()

            if isRunning || isComplete {
                narrationCard(step: step, isComplete: isComplete)
                    .frame(maxWidth: 420)
                    .padding(.horizontal, DS.Space.xl)
                    .padding(.bottom, Self.clusterClearance)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity)
        .animation(DS.Motion.respecting(reduceMotion), value: step)
    }

    // MARK: - Narration Card

    private func narrationCard(step: AppViewModel.DemoStep, isComplete: Bool) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            HStack(spacing: DS.Space.s) {
                Text(eyebrow(isComplete: isComplete))
                    .font(DS.Font.eyebrow)
                    .tracking(0.6)
                    .foregroundStyle(.tertiary)
                Spacer(minLength: 0)
                demoProgressDots
            }

            HStack(alignment: .top, spacing: DS.Space.l) {
                statusIndicator(isComplete: isComplete)

                VStack(alignment: .leading, spacing: DS.Space.xs) {
                    Text(step.rawValue)
                        .font(DS.Font.sectionTitle)
                        .foregroundStyle(.primary)
                        .fixedSize(horizontal: false, vertical: true)

                    if !viewModel.demoNarration.isEmpty {
                        Text(viewModel.demoNarration)
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if isDockingPhase {
                        dockingLiveStats
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if isComplete {
                    Button("Explore") {
                        withAnimation(DS.Motion.respecting(reduceMotion)) {
                            viewModel.demoStep = .idle
                            viewModel.demoNarration = ""
                        }
                    }
                    .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                    .help("End the demo and explore the results")
                } else {
                    Button("Stop") {
                        viewModel.cancelDemo()
                    }
                    .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    .help("Stop the guided demo")
                }
            }
        }
        .padding(.horizontal, DS.Space.xl)
        .padding(.vertical, DS.Space.l)
        .dsFloating(cornerRadius: DS.Radius.floating)
    }

    private func eyebrow(isComplete: Bool) -> String {
        if isComplete { return "GUIDED DEMO · DONE" }
        guard let stage = currentStage else { return "GUIDED DEMO" }
        return "GUIDED DEMO · \(stage.title.uppercased())"
    }

    @ViewBuilder
    private func statusIndicator(isComplete: Bool) -> some View {
        ZStack {
            Circle()
                .fill(isComplete ? DS.Color.success.opacity(0.14) : DS.Color.accentSoft)
            if isComplete {
                Image(systemName: "checkmark")
                    .font(DS.Font.body.weight(.semibold))
                    .foregroundStyle(DS.Color.success)
            } else {
                Image(systemName: currentStage?.icon ?? "sparkles")
                    .font(DS.Font.body.weight(.medium))
                    .foregroundStyle(DS.Color.accent)
                    .symbolEffect(.pulse, options: .repeating, isActive: !reduceMotion)
            }
        }
        .frame(width: DS.Size.primaryButtonHeight, height: DS.Size.primaryButtonHeight)
        .accessibilityHidden(true)
    }

    // MARK: - Progress Dots

    private var demoProgressDots: some View {
        HStack(spacing: DS.Space.xs) {
            ForEach(DemoStage.allCases, id: \.self) { stage in
                let state = stageState(stage)
                Capsule()
                    .fill(dotFill(state))
                    .overlay(
                        Capsule().strokeBorder(state == .upcoming ? DS.Color.hairline : .clear, lineWidth: 1)
                    )
                    .frame(width: state == .active ? DS.Space.xl : DS.Space.s, height: DS.Space.s)
                    .help(stage.title)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(progressAccessibilityLabel)
    }

    private var progressAccessibilityLabel: String {
        guard let stage = currentStage, let index = DemoStage.allCases.firstIndex(of: stage) else {
            return "Demo progress"
        }
        return "Stage \(index + 1) of \(DemoStage.allCases.count): \(stage.title)"
    }

    private func dotFill(_ state: StageState) -> Color {
        switch state {
        case .completed: DS.Color.accent.opacity(0.55)
        case .active: DS.Color.accent
        case .upcoming: DS.Color.controlFill
        }
    }

    // MARK: - Docking Live Stats

    @ViewBuilder
    private var dockingLiveStats: some View {
        let gen = viewModel.docking.dockingGeneration + 1
        let total = viewModel.docking.dockingTotalGenerations
        let energy = viewModel.docking.dockingBestEnergy

        if viewModel.docking.isDocking {
            HStack(spacing: DS.Space.m) {
                Text("Gen \(gen)/\(total)")
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)

                ProgressView(value: Double(min(gen, max(total, 1))), total: Double(max(total, 1)))
                    .progressViewStyle(.linear)
                    .controlSize(.small)
                    .tint(DS.Color.accent)
                    .frame(maxWidth: 120)

                if energy < .infinity {
                    DSScoreBadge(energy: energy, unit: viewModel.docking.scoringMethod.unitLabel)
                }
            }
            .padding(.top, DS.Space.xxs)
        }
    }

    // MARK: - Demo Stage Mapping

    /// Demo stages, titled after the workflow steps they exercise.
    private enum DemoStage: CaseIterable {
        case load, explore, pocket, dock, results

        var tab: SidebarTab? {
            switch self {
            case .load: .search
            case .explore: nil
            case .pocket: .bindingSite
            case .dock: .dock
            case .results: .results
            }
        }

        var title: String { tab?.rawValue ?? "Explore" }
        var icon: String { tab?.icon ?? "rotate.3d" }
    }

    private enum StageState { case completed, active, upcoming }

    private var currentStage: DemoStage? {
        DemoStage.allCases.first { stageState($0) == .active }
    }

    private func stageState(_ stage: DemoStage) -> StageState {
        let step = viewModel.demoStep
        switch stage {
        case .load:
            if [.fetching, .parsing].contains(step) { return .active }
            return stepsAfterLoad.contains(step) ? .completed : .upcoming
        case .explore:
            if [.overview, .ribbon].contains(step) { return .active }
            return stepsAfterExplore.contains(step) ? .completed : .upcoming
        case .pocket:
            if [.pocketScan, .pocketFound, .gridSetup].contains(step) { return .active }
            return stepsAfterPocket.contains(step) ? .completed : .upcoming
        case .dock:
            if [.dockingStart, .dockingRun, .dockingConverge].contains(step) { return .active }
            return stepsAfterDock.contains(step) ? .completed : .upcoming
        case .results:
            if [.scoring, .bestPose, .interactions].contains(step) { return .active }
            return step == .complete ? .completed : .upcoming
        }
    }

    private var stepsAfterLoad: [AppViewModel.DemoStep] {
        [.overview, .ribbon, .pocketScan, .pocketFound, .gridSetup,
         .dockingStart, .dockingRun, .dockingConverge, .scoring, .bestPose, .interactions, .complete]
    }
    private var stepsAfterExplore: [AppViewModel.DemoStep] {
        [.pocketScan, .pocketFound, .gridSetup,
         .dockingStart, .dockingRun, .dockingConverge, .scoring, .bestPose, .interactions, .complete]
    }
    private var stepsAfterPocket: [AppViewModel.DemoStep] {
        [.dockingStart, .dockingRun, .dockingConverge, .scoring, .bestPose, .interactions, .complete]
    }
    private var stepsAfterDock: [AppViewModel.DemoStep] {
        [.scoring, .bestPose, .interactions, .complete]
    }

    private var isDockingPhase: Bool {
        [.dockingStart, .dockingRun, .dockingConverge].contains(viewModel.demoStep)
    }
}

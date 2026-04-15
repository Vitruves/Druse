// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Demo Overlay
//
// A narrated banner shown during the guided demo, displaying the current step
// description, progress indicator, and pipeline stage. Positioned at the bottom
// of the Metal viewport, above the render controls.

struct DemoOverlay: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        let step = viewModel.demoStep
        let isRunning = viewModel.isDemoRunning
        let isComplete = step == .complete

        VStack(spacing: 0) {
            Spacer()

            if isRunning || isComplete {
                VStack(spacing: 0) {
                    // Progress dots (pipeline stages)
                    if isRunning {
                        demoProgressDots
                            .padding(.bottom, 8)
                    }

                    // Main narration card
                    narrationCard(step: step, isRunning: isRunning, isComplete: isComplete)
                }
                .padding(.horizontal, 30)
                .padding(.bottom, 8)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: step)
    }

    // MARK: - Narration Card

    @ViewBuilder
    private func narrationCard(step: AppViewModel.DemoStep, isRunning: Bool, isComplete: Bool) -> some View {
        HStack(spacing: 12) {
            // Left: status indicator
            if isComplete {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.green)
            } else {
                ZStack {
                    Circle()
                        .fill(currentStageColor.opacity(0.2))
                        .frame(width: 32, height: 32)
                    ProgressView()
                        .controlSize(.small)
                        .tint(currentStageColor)
                }
            }

            // Center: narration text
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(step.rawValue)
                        .font(.callout.bold())
                        .foregroundStyle(.primary)

                    if isDockingPhase {
                        // Live stats
                        dockingLiveStats
                    }
                }

                if !viewModel.demoNarration.isEmpty {
                    Text(viewModel.demoNarration)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Right: cancel/dismiss button
            if isComplete {
                Button("Explore") {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        viewModel.demoStep = .idle
                        viewModel.demoNarration = ""
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.accentColor)
                .controlSize(.small)
            } else {
                Button("Stop") {
                    viewModel.cancelDemo()
                }
                .controlSize(.small)
                .buttonStyle(.bordered)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(currentStageColor.opacity(0.25), lineWidth: 1)
                )
                .shadow(color: Color(nsColor: .shadowColor).opacity(0.2), radius: 8, y: 3)
        )
    }

    // MARK: - Progress Dots

    private var demoProgressDots: some View {
        HStack(spacing: 8) {
            ForEach(DemoStage.allCases, id: \.self) { stage in
                let state = stageState(stage)
                HStack(spacing: 4) {
                    Circle()
                        .fill(dotColor(state))
                        .frame(width: 8, height: 8)
                        .overlay {
                            if state == .active {
                                Circle()
                                    .stroke(dotColor(state), lineWidth: 1.5)
                                    .frame(width: 14, height: 14)
                                    .opacity(0.5)
                            }
                        }

                    Text(stage.rawValue)
                        .font(.footnote.weight(state == .active ? .bold : .medium))
                        .foregroundStyle(state == .upcoming ? .secondary : .primary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            Capsule()
                .fill(.ultraThinMaterial)
        )
    }

    // MARK: - Docking Live Stats

    @ViewBuilder
    private var dockingLiveStats: some View {
        let gen = viewModel.docking.dockingGeneration + 1
        let total = viewModel.docking.dockingTotalGenerations
        let energy = viewModel.docking.dockingBestEnergy

        if viewModel.docking.isDocking {
            HStack(spacing: 8) {
                Text("Gen \(gen)/\(total)")
                    .font(.footnote.weight(.medium).monospaced())
                    .foregroundStyle(.cyan)

                if energy < .infinity {
                    let sm = viewModel.docking.scoringMethod
                    Text(String(format: "%.1f %@", energy, sm.unitLabel))
                        .font(.footnote.bold().monospaced())
                        .foregroundStyle(energy < -6 ? .green : energy < -3 ? .yellow : .orange)
                }

                ProgressView(value: Double(gen), total: Double(max(total, 1)))
                    .progressViewStyle(.linear)
                    .frame(width: 60)
                    .tint(.cyan)
            }
        }
    }

    // MARK: - Demo Stage Mapping

    private enum DemoStage: String, CaseIterable {
        case load = "Load"
        case explore = "Explore"
        case pocket = "Pocket"
        case dock = "Dock"
        case results = "Results"
    }

    private enum StageState { case completed, active, upcoming }

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
            if [.scoring, .bestPose, .interactions, .complete].contains(step) { return .active }
            return .upcoming
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

    private var currentStageColor: Color {
        let step = viewModel.demoStep
        if [.fetching, .parsing].contains(step) { return .cyan }
        if [.overview, .ribbon].contains(step) { return .blue }
        if [.pocketScan, .pocketFound, .gridSetup].contains(step) { return .orange }
        if [.dockingStart, .dockingRun, .dockingConverge].contains(step) { return .purple }
        if [.scoring, .bestPose, .interactions].contains(step) { return .green }
        if step == .complete { return .green }
        return .secondary
    }

    private func dotColor(_ state: StageState) -> Color {
        switch state {
        case .completed: .green
        case .active: currentStageColor
        case .upcoming: .secondary.opacity(0.3)
        }
    }
}

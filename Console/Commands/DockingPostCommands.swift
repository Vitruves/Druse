// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - refine, rescore, strain (post-docking operations on the current results)

extension DruseInterpreter {

    // MARK: refine

    /// `refine`: post-docking refinement of the best poses with OpenMM or GFN2-xTB.
    static func refineCommand() -> ConsoleCommand {
        let gfn2 = DockingConfig().gfn2Refinement
        return ConsoleCommand(
            name: "refine",
            summary: "Refine the current docking poses with OpenMM or GFN2-xTB",
            category: .docking,
            forms: [
                ConsoleForm(path: ["openmm"], summary: "OpenMM pocket minimization of the top poses; stores the interaction energy (ranking unchanged)",
                            options: [ConsoleParameter(name: "top", type: .int,
                                                       summary: "Poses to refine (default \(AppViewModel.defaultOpenMMRefinedPoses))")]) { inv in
                    let top = try positiveCount(inv, "top", default: AppViewModel.defaultOpenMMRefinedPoses)
                    let count = try await runPostProcessing { try await inv.viewModel.refineDockingResultsWithOpenMM(top: top) }
                    guard count > 0 else {
                        throw ConsoleError("OpenMM minimization diverged for every pose — poses and receptor left unchanged")
                    }
                    inv.emit(.success, "OpenMM refined \(count) pose(s); the receptor shows the best pose's relaxed pocket — 'results show 1' lists the energy")
                },
                ConsoleForm(path: ["gfn2"], summary: "GFN2-xTB optimization of the top poses with D4 dispersion and implicit solvation; re-ranks by the blended score",
                            options: [
                                ConsoleParameter(name: "top", type: .int, summary: "Poses to optimize (default: dock setting gfn2-top)"),
                                ConsoleParameter(name: "solvation", type: .choice(DockingNames.names(DockingNames.gfn2Solvation)),
                                                 summary: "Implicit solvation (default: gfn2-solvation)"),
                                ConsoleParameter(name: "level", type: .choice(DockingNames.names(DockingNames.gfn2Levels)),
                                                 summary: "Convergence level (default: gfn2-level)"),
                                ConsoleParameter(name: "steps", type: .int, summary: "Maximum optimization steps, 0 = automatic (default: gfn2-steps)"),
                                ConsoleParameter(name: "blend", type: .float, summary: "Weight of the GFN2 energy in the ranking, 0–1 (default \(DockingSettings.format(gfn2.blendWeight)))"),
                                ConsoleParameter(name: "restraint", type: .float, summary: "Positional restraint in Hartree/Å², 0 = none (default: gfn2-restraint)"),
                                ConsoleParameter(name: "max-rmsd", type: .float, summary: "Keep the docked geometry when optimization moves it more, Å (default: gfn2-max-rmsd)"),
                                ConsoleParameter(name: "update", type: .bool, summary: "Use the optimized coordinates (default: gfn2-update)")
                            ]) { inv in
                    try await refineGFN2(inv)
                }
            ],
            examples: ["refine openmm", "refine openmm top=10", "refine gfn2", "refine gfn2 top=5 solvation=gbsa level=tight",
                       "refine gfn2 blend=0 update=off"]
        )
    }

    /// One-off GFN2 settings: the session's `gfn2-*` dock settings overridden by the options.
    private static func refineGFN2(_ inv: ConsoleInvocation) async throws {
        var config = inv.viewModel.docking.dockingConfig.gfn2Refinement
        config.topPosesToRefine = try positiveCount(inv, "top", default: config.topPosesToRefine)
        if let solvation = ConsoleNames.lookup(inv.string("solvation"), in: DockingNames.gfn2Solvation) { config.solvation = solvation }
        if let level = ConsoleNames.lookup(inv.string("level"), in: DockingNames.gfn2Levels) { config.optLevel = level }
        if let steps = inv.int("steps") {
            guard (0...10_000).contains(steps) else { throw ConsoleError("steps must be between 0 and 10000") }
            config.maxSteps = Int32(steps)
        }
        if let blend = inv.double("blend") {
            guard (0...1).contains(blend) else { throw ConsoleError("blend must be between 0 and 1") }
            config.blendWeight = Float(blend)
        }
        config.restraintStrength = try positiveFloat(inv, "restraint", default: config.restraintStrength)
        config.maxRMSD = try positiveFloat(inv, "max-rmsd", default: config.maxRMSD)
        if let update = inv.bool("update") { config.updateCoordinates = update }

        inv.emit(.output, "GFN2-xTB: top \(config.topPosesToRefine) poses, \(config.solvation.rawValue), \(config.optLevel.rawValue.lowercased()) convergence…")
        let (optimized, converged) = try await runPostProcessing { try await inv.viewModel.refineDockingResultsWithGFN2(config: config) }
        guard optimized > 0 else { throw ConsoleError("GFN2-xTB optimized no pose — see the activity log (is xTB available in this build?)") }
        inv.emit(.success, "GFN2-xTB optimized \(optimized) pose(s), \(converged) converged; best now \(bestScore(inv.viewModel))")
    }

    // MARK: rescore

    /// `rescore`: rescoring of the current poses with a neural model, leaving their coordinates unchanged.
    static func rescoreCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "rescore",
            summary: "Rescore the current poses with a neural model without moving them",
            category: .docking,
            forms: [
                ConsoleForm(path: [], summary: "Score every pose with DruseAF (pKd) or PIGNet2 (kcal/mol); that model becomes the active scoring",
                            arguments: [ConsoleParameter(name: "model", type: .choice(["druseaf", "pignet2"]),
                                                         summary: "druseaf = DruseAF v4 affinity, pignet2 = physics-informed GNN")]) { inv in
                    let method: ScoringMethod = inv.string("model") == "pignet2" ? .pignet2 : .druseAffinity
                    let count = try await runPostProcessing { try await inv.viewModel.rescoreDockingResults(with: method) }
                    inv.emit(.success, "Rescored \(count) poses with \(method.rawValue); best now \(bestScore(inv.viewModel))")
                }
            ],
            examples: ["rescore druseaf", "rescore pignet2"]
        )
    }

    // MARK: strain

    /// `strain`: MMFF94 ligand strain of the best poses, optionally added to their scores as a penalty.
    static func strainCommand() -> ConsoleCommand {
        let config = DockingConfig()
        return ConsoleCommand(
            name: "strain",
            summary: "MMFF94 ligand strain of the current poses, with an optional score penalty",
            category: .docking,
            forms: [
                ConsoleForm(path: [], summary: "Compute docked − relaxed MMFF94 energy; poses above the threshold get weight × excess added to their score",
                            options: [
                                ConsoleParameter(name: "top", type: .int, summary: "Poses to evaluate (default 50)"),
                                ConsoleParameter(name: "threshold", type: .float,
                                                 summary: "Tolerated strain in kcal/mol (default: dock setting strain-threshold, \(DockingSettings.format(config.strainPenaltyThreshold)))"),
                                ConsoleParameter(name: "weight", type: .float,
                                                 summary: "Penalty weight, 0 = report only (default: strain-weight, \(DockingSettings.format(config.strainPenaltyWeight)))")
                            ]) { inv in
                    let vm = inv.viewModel
                    let top = try positiveCount(inv, "top", default: 50)
                    let threshold = try positiveFloat(inv, "threshold", default: vm.docking.dockingConfig.strainPenaltyThreshold)
                    let weight = try positiveFloat(inv, "weight", default: vm.docking.dockingConfig.strainPenaltyWeight)
                    let (computed, penalized) = try await runPostProcessing {
                        try await vm.applyStrainPenaltiesToDockingResults(top: top, threshold: threshold, weight: weight)
                    }
                    guard computed > 0 else { throw ConsoleError("MMFF94 strain could not be computed — see the activity log") }
                    inv.emit(.success, "Strain for \(computed) pose(s); \(penalized) above \(DockingSettings.format(threshold)) kcal/mol"
                             + (weight > 0 ? " (penalized)" : "") + "; best now \(bestScore(vm))")
                }
            ],
            examples: ["strain", "strain top=20 threshold=4", "strain weight=0"]
        )
    }

    // MARK: Helpers

    /// Runs a view-model post-processing operation, turning its errors into console errors.
    private static func runPostProcessing<T>(_ operation: () async throws -> T) async throws -> T {
        do {
            return try await operation()
        } catch let error as AppViewModel.DockingPostProcessingError {
            throw ConsoleError(error.errorDescription ?? "post-processing failed")
        }
    }

    /// A `name=` count of at least 1, or `fallback` when absent.
    private static func positiveCount(_ inv: ConsoleInvocation, _ name: String, default fallback: Int) throws -> Int {
        guard let value = inv.int(name) else { return fallback }
        guard value >= 1 else { throw ConsoleError("\(name) must be at least 1") }
        return value
    }

    private static func bestScore(_ vm: AppViewModel) -> String {
        vm.docking.dockingResults.first.map { ConsoleFormat.score($0, vm.docking.scoringMethod) } ?? "—"
    }
}

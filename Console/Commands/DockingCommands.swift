// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - pocket, dock, results, pose, screen (+ flex, constraint, refine, rescore, strain)

extension DruseInterpreter {

    /// Docking vocabulary: site definition, docking runs and their settings, results,
    /// receptor flexibility, constraints, post-docking refinement and virtual screening.
    static func dockingCommands() -> [ConsoleCommand] {
        [pocketCommand(), dockCommand(), resultsCommand(), poseCommand(),
         flexCommand(), constraintCommand(), refineCommand(), rescoreCommand(), strainCommand(),
         screenCommand()]
    }

    // MARK: pocket

    /// `pocket`: binding-site detection, selection and the docking grid box.
    private static func pocketCommand() -> ConsoleCommand {
        let gridSpacing = DockingSettings.setting("grid-spacing").map { [$0.parameter] } ?? []
        return ConsoleCommand(
            name: "pocket",
            summary: "Detect, list and select binding pockets, or define and fit the docking box",
            category: .docking,
            forms: [
                ConsoleForm(path: ["detect"], summary: "Find pockets and select the best (auto = ML + geometric when the ML model is available)",
                            options: [ConsoleParameter(name: "method", type: .choice(["auto", "geometric", "ml", "ligand"]),
                                                       summary: "auto, geometric only, ML only, or around the current ligand")]) { inv in
                    try await detectPockets(inv)
                },
                ConsoleForm(path: ["list"], summary: "List detected pockets (* = selected)") { inv in
                    guard !inv.viewModel.docking.detectedPockets.isEmpty else {
                        inv.emit(.output, "No pockets — run 'pocket detect' or 'pocket box center=x,y,z size=x,y,z'")
                        return
                    }
                    inv.emit(.output, pocketTable(inv.viewModel))
                },
                ConsoleForm(path: ["select"], summary: "Select a detected pocket for docking",
                            arguments: [ConsoleParameter(name: "n", type: .int, summary: "1-based index from 'pocket list'")]) { inv in
                    let vm = inv.viewModel
                    let index = inv.requiredInt("n")
                    let pockets = vm.docking.detectedPockets
                    guard pockets.indices.contains(index - 1) else {
                        throw ConsoleError(pockets.isEmpty ? "no pockets detected — run 'pocket detect'"
                                           : "no pocket #\(index) — \(pockets.count) detected")
                    }
                    let pocket = pockets[index - 1]
                    selectPocket(pocket, in: vm)
                    inv.emit(.success, "Selected pocket #\(index): \(String(format: "%.0f", pocket.volume)) Å³, center \(ConsoleFormat.vector(pocket.center))")
                },
                ConsoleForm(path: ["residues"], summary: "Define the pocket around residues (box = their atoms + padding)",
                            arguments: [ConsoleParameter(name: "residues", type: .string,
                                                         summary: "Residues as [chain:][name]number, e.g. A:ASP189 TYR95 120",
                                                         isVariadic: true)],
                            options: [ConsoleParameter(name: "padding", type: .float, summary: "Margin around the residues in Å (default 4)")]) { inv in
                    try pocketFromResidues(inv)
                },
                ConsoleForm(path: ["box"], summary: "Set the docking box explicitly (size = full edge lengths); keeps the current center or size when omitted",
                            options: [ConsoleParameter(name: "center", type: .vec3, summary: "Box center in Å"),
                                      ConsoleParameter(name: "size", type: .vec3, summary: "Box edge lengths in Å (default: current box, else 20,20,20)")]
                                + gridSpacing) { inv in
                    try setBox(inv)
                },
                ConsoleForm(path: ["box", "show"], summary: "Print the docking box, grid spacing and estimated grid size") { inv in
                    try showBox(inv)
                },
                ConsoleForm(path: ["box", "fit"], summary: "Fit the docking box around the ligand or the selected pocket's residues",
                            options: [ConsoleParameter(name: "around", type: .choice(["ligand", "pocket"]),
                                                       summary: "What to enclose (default: ligand when one is loaded)"),
                                      ConsoleParameter(name: "padding", type: .float, summary: "Margin on every side in Å (default 4)")]
                                + gridSpacing) { inv in
                    try fitBox(inv)
                }
            ],
            examples: ["pocket detect", "pocket detect method=ligand", "pocket list; pocket select 2",
                       "pocket residues A:ASP189 SER195 HIS57 padding=5",
                       "pocket box center=12.5,-3,40.2 size=22,22,22", "pocket box grid-spacing=0.5",
                       "pocket box fit around=ligand padding=6", "pocket box show"]
        )
    }

    /// Runs the chosen pocket detector and selects the best pocket, as the Binding Site step's Detect button does.
    private static func detectPockets(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        switch inv.string("method") ?? "auto" {
        case "ligand":
            try requirePlacedLigand(vm, for: "method=ligand")
            guard vm.detectLigandGuidedPocket() else {
                throw ConsoleError("could not define a pocket around \(vm.molecules.ligand?.name ?? "the ligand")")
            }
        case "ml":
            if !vm.pocketDetectorML.isAvailable {
                inv.emit(.warning, "ML pocket model unavailable — using geometric detection")
            }
            await vm.detectPocketsML()?.value
        case "geometric":
            await vm.detectPockets(allowML: false)?.value
        default:
            await vm.detectPockets()?.value
        }
        guard let best = vm.docking.selectedPocket, !vm.docking.detectedPockets.isEmpty else {
            throw ConsoleError("no pockets detected")
        }
        syncGrid(to: best, in: vm)
        inv.emit(.output, pocketTable(vm))
        inv.emit(.success, "\(vm.docking.detectedPockets.count) pocket(s); selected #1: \(String(format: "%.0f", best.volume)) Å³, druggability \(String(format: "%.2f", best.druggability))")
    }

    /// Builds a pocket from explicit residues (`pocket residues`), padded by `padding=`.
    private static func pocketFromResidues(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        guard let protein = vm.molecules.protein else { return }
        let padding = try positiveFloat(inv, "padding", default: 4)
        let indices = try DockingResidueSpec.resolveAll(inv.requiredString("residues"), in: protein)
        let pocket = BindingSiteDetector.pocketFromResidues(protein: protein, residueIndices: indices, padding: padding)
        vm.docking.detectedPockets = [pocket]
        selectPocket(pocket, in: vm)
        inv.emit(.success, "Pocket from \(indices.count) residue(s): center \(ConsoleFormat.vector(pocket.center)), box \(ConsoleFormat.box(pocket.size)) Å")
    }

    /// Sets the grid box centre, size and spacing explicitly (`pocket box`).
    private static func setBox(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        let spacing = try DockingSettings.apply(inv, keys: ["grid-spacing"])
        guard inv.has("center") || inv.has("size") else {
            guard !spacing.isEmpty else {
                throw ConsoleError("pocket box needs center=x,y,z, size=x,y,z or grid-spacing=; 'pocket box show' prints the box")
            }
            inv.emit(.success, "Grid spacing \(DockingSettings.format(vm.docking.dockingConfig.gridSpacing)) Å")
            return
        }
        let current = currentBox(vm)
        guard let center = inv.vec3("center") ?? current?.center else {
            throw ConsoleError("no docking box yet — give center=x,y,z")
        }
        let size = inv.vec3("size") ?? current.map { $0.halfSize * 2 } ?? SIMD3<Float>(repeating: 20)
        guard size.min() > 0 else { throw ConsoleError("size values must be positive") }
        applyBox(center: center, halfSize: size / 2, in: vm)
        inv.emit(.success, "Docking box at \(ConsoleFormat.vector(center)), \(ConsoleFormat.box(size / 2)) Å, "
                 + "grid spacing \(DockingSettings.format(vm.docking.dockingConfig.gridSpacing)) Å")
    }

    /// Prints the effective grid box: centre, edges, volume, spacing, grid points and where it came from.
    private static func showBox(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let box = currentBox(vm) else {
            throw ConsoleError("no docking box — use 'pocket detect', 'pocket box' or 'pocket box fit'")
        }
        let spacing = vm.docking.dockingConfig.gridSpacing
        let edges = box.halfSize * 2
        let points = (0..<3).map { Int((edges[$0] / spacing).rounded(.up)) + 1 }
        let source = vm.docking.selectedPocket.map { $0.residueIndices.isEmpty ? "explicit box" : "pocket #\($0.number), \($0.residueIndices.count) residues" }
            ?? "grid sliders"
        let rows = [
            ["Center", ConsoleFormat.vector(box.center) + " Å"],
            ["Edges", ConsoleFormat.box(box.halfSize) + " Å"],
            ["Volume", String(format: "%.0f Å³", edges.x * edges.y * edges.z)],
            ["Spacing", DockingSettings.format(spacing) + " Å"],
            ["Search grid", "\(points[0])×\(points[1])×\(points[2]) points (the engine adds a ligand-sized margin)"],
            ["Source", source]
        ]
        inv.emit(.output, rows.map { $0[0].padding(toLength: 13, withPad: " ", startingAt: 0) + $0[1] }.joined(separator: "\n"))
    }

    /// Fits the grid box around the ligand, the selection or the current pocket plus padding (`pocket box fit`).
    private static func fitBox(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        try DockingSettings.apply(inv, keys: ["grid-spacing"])
        let padding = try positiveFloat(inv, "padding", default: 4)
        let around = inv.string("around") ?? (vm.molecules.ligand != nil ? "ligand" : "pocket")
        let positions: [SIMD3<Float>]
        let label: String
        if around == "ligand" {
            let ligand = try requirePlacedLigand(vm, for: "around=ligand")
            positions = ligand.atoms.filter { $0.element != .H }.map(\.position)
            label = ligand.name
        } else {
            guard let pocket = vm.docking.selectedPocket, let protein = vm.molecules.protein else {
                throw ConsoleError("around=pocket needs a selected pocket — use 'pocket detect' or 'pocket residues'")
            }
            guard !pocket.residueIndices.isEmpty else {
                throw ConsoleError("the selected pocket is an explicit box with no residues — use around=ligand or 'pocket residues'")
            }
            positions = pocket.residueIndices
                .filter { protein.residues.indices.contains($0) }
                .flatMap { protein.residues[$0].atomIndices }
                .filter { protein.atoms.indices.contains($0) && protein.atoms[$0].element != .H }
                .map { protein.atoms[$0].position }
            label = "pocket #\(pocket.number)"
        }
        guard let first = positions.first else { throw ConsoleError("\(label) has no heavy atoms to fit") }
        let (low, high) = positions.reduce((first, first)) { (simd_min($0.0, $1), simd_max($0.1, $1)) }
        let center = (low + high) / 2
        let halfSize = (high - low) / 2 + SIMD3<Float>(repeating: padding)
        applyBox(center: center, halfSize: halfSize, in: vm)
        inv.emit(.success, "Box fitted around \(label) (+\(DockingSettings.format(padding)) Å): center \(ConsoleFormat.vector(center)), "
                 + "\(ConsoleFormat.box(halfSize)) Å")
    }

    /// The active ligand, provided it sits in the protein (some heavy atom within 6 Å of a protein heavy atom).
    /// A ligand built from SMILES or a prepared conformer lives at the origin, so a box or pocket fitted around it
    /// would land outside the protein; refusing it here turns a silent mis-dock into an explanation.
    @discardableResult
    private static func requirePlacedLigand(_ vm: AppViewModel, for option: String) throws -> Molecule {
        guard let ligand = vm.molecules.ligand else { throw ConsoleError("\(option) needs a ligand — load or build one first") }
        guard let protein = vm.molecules.protein else { return ligand }
        let cutoff: Float = 6 * 6
        let proteinHeavy = protein.atoms.filter { $0.element != .H }.map(\.position)
        let placed = ligand.atoms.contains { atom in
            atom.element != .H && proteinHeavy.contains { simd_distance_squared($0, atom.position) <= cutoff }
        }
        guard placed else {
            throw ConsoleError("\(option): ligand '\(ligand.name)' is not placed in the protein (no protein atom within 6 Å; "
                               + "SMILES-built and prepared conformers sit at the origin) — load a posed ligand file, or use 'pocket detect' / around=pocket")
        }
        return ligand
    }

    /// The docking box in effect: the selected pocket, else the grid sliders once placed.
    private static func currentBox(_ vm: AppViewModel) -> (center: SIMD3<Float>, halfSize: SIMD3<Float>)? {
        if let pocket = vm.docking.selectedPocket { return (pocket.center, pocket.size) }
        return vm.docking.gridInitialized ? (vm.docking.gridCenter, vm.docking.gridHalfSize) : nil
    }

    /// Replaces the site by an explicit box (as the Binding Site step's sliders do).
    private static func applyBox(center: SIMD3<Float>, halfSize: SIMD3<Float>, in vm: AppViewModel) {
        let edges = halfSize * 2
        let pocket = BindingPocket(
            id: 0, center: center, size: halfSize,
            volume: edges.x * edges.y * edges.z,
            buriedness: 0.5, polarity: 0.5, druggability: 0.5,
            residueIndices: [], probePositions: []
        )
        vm.docking.detectedPockets = [pocket]
        selectPocket(pocket, in: vm)
    }

    private static func selectPocket(_ pocket: BindingPocket, in vm: AppViewModel) {
        vm.docking.selectedPocket = pocket
        syncGrid(to: pocket, in: vm)
        vm.showGridBoxForPocket(pocket)
    }

    /// Keeps the Docking tab's grid sliders on the pocket the console chose.
    private static func syncGrid(to pocket: BindingPocket, in vm: AppViewModel) {
        vm.docking.gridCenter = pocket.center
        vm.docking.gridHalfSize = pocket.size
        vm.docking.gridInitialized = true
    }

    /// Table of detected pockets with volume, buriedness, druggability and box; `*` marks the selected one.
    private static func pocketTable(_ vm: AppViewModel) -> String {
        let selectedID = vm.docking.selectedPocket?.id
        let rows = vm.docking.detectedPockets.enumerated().map { i, pocket in
            [
                "\(i + 1)\(pocket.id == selectedID ? "*" : "")",
                String(format: "%.0f", pocket.volume),
                String(format: "%.2f", pocket.buriedness),
                String(format: "%.2f", pocket.druggability),
                "\(pocket.residueIndices.count)",
                ConsoleFormat.vector(pocket.center),
                ConsoleFormat.box(pocket.size)
            ]
        }
        return ConsoleTable.render(["#", "Vol Å³", "Buried", "Drug.", "Res", "Center", "Box Å"], rows,
                                   rightAligned: [0, 1, 2, 3, 4])
    }

    /// A non-negative `name=` float option, or `fallback` when absent.
    static func positiveFloat(_ inv: ConsoleInvocation, _ name: String, default fallback: Float) throws -> Float {
        guard let value = inv.double(name) else { return fallback }
        guard value >= 0 else { throw ConsoleError("\(name) must not be negative") }
        return Float(value)
    }

    // MARK: dock

    /// Settings accepted inline by `dock` and `dock batch` (any setting via `dock set`).
    private static let inlineDockKeys = ["search", "scoring", "charges", "population", "generations", "runs"]

    /// `dock`: single-ligand and batch docking runs, session docking settings, run status and cancellation.
    private static func dockCommand() -> ConsoleCommand {
        let preset = ConsoleParameter(
            name: "preset", type: .choice(ConsoleNames.names(ConsoleNames.presets)),
            summary: "Budget preset: " + DockingPreset.allCases.map { "\($0.rawValue.lowercased()) = \($0.help)" }.joined(separator: "; ")
        )
        let inline = [preset] + inlineDockKeys.compactMap { DockingSettings.setting($0)?.parameter }
        let sections = DockingSettingSection.allCases.map(\.rawValue)
        return ConsoleCommand(
            name: "dock",
            summary: "Dock the active ligand or Ligand Database entries; view and change every docking setting",
            category: .docking,
            forms: [
                ConsoleForm(path: [], summary: "Dock the active ligand into the selected pocket; options persist for later runs",
                            options: inline) { inv in
                    try await dockActiveLigand(inv)
                },
                ConsoleForm(path: ["batch"], summary: "Dock Ligand Database entries one after another and rank them",
                            options: [ConsoleParameter(name: "ligands", type: .string,
                                                       summary: "all (default: entries with 3D coordinates), prepared, or comma-separated names")]
                                + inline) { inv in
                    try await dockBatch(inv)
                },
                ConsoleForm(path: ["set"], summary: "Change docking settings for the session (validated; see 'dock config')",
                            options: [preset] + DockingSettings.all.map(\.parameter)) { inv in
                    try setDockSettings(inv)
                },
                ConsoleForm(path: ["config"], summary: "Print every docking setting by section; * marks values changed from the default",
                            arguments: [ConsoleParameter(name: "section", type: .choice(sections),
                                                         summary: "Only this section", isOptional: true)],
                            options: [ConsoleParameter(name: "changed", type: .bool, summary: "Only settings that differ from the default")]) { inv in
                    let section = inv.string("section").flatMap(DockingSettingSection.init(rawValue:))
                    inv.emit(.output, DockingSettings.report(inv.viewModel, section: section, changedOnly: inv.bool("changed") ?? false))
                },
                ConsoleForm(path: ["config", "reset"], summary: "Restore docking settings (all, or one section) to their defaults",
                            arguments: [ConsoleParameter(name: "section", type: .choice(sections),
                                                         summary: "Only this section", isOptional: true)]) { inv in
                    let section = inv.string("section").flatMap(DockingSettingSection.init(rawValue:))
                    DockingSettings.reset(inv.viewModel, section: section)
                    inv.emit(.success, section.map { "\($0.title) settings reset to defaults" } ?? "All docking settings reset to defaults")
                },
                ConsoleForm(path: ["status"], summary: "Show whether a docking run (e.g. started from the Docking step) is in progress") { inv in
                    inv.emit(.output, dockStatus(inv.viewModel))
                },
                ConsoleForm(path: ["stop"], summary: "Stop a docking or batch docking run started from the Docking step") { inv in
                    let vm = inv.viewModel
                    if vm.docking.isBatchDocking {
                        vm.cancelBatchDocking()
                    } else if vm.docking.isDocking {
                        vm.stopDocking()
                    } else {
                        inv.emit(.output, "No docking run in progress")
                        return
                    }
                    inv.emit(.success, "Docking stopped")
                }
            ],
            examples: ["dock", "dock preset=thorough scoring=drusina", "dock search=fragment runs=5",
                       "dock set population=200 generations=150 runs=4 grid-spacing=0.375",
                       "dock set search=remc remc-replicas=12 remc-tmax=6",
                       "dock set torsion-preset=redocking ls-steps=40 strain=off",
                       "dock set gfn2=on gfn2-top=10 gfn2-solvation=alpb",
                       "dock config; dock config fragment; dock config changed=on", "dock config reset",
                       "dock batch ligands=prepared preset=fast", "dock batch ligands=aspirin,ibuprofen"]
        )
    }

    /// `preset=` first (it resets the budget), then the given setting options; returns what was set.
    @discardableResult
    private static func applyDockOptions(_ inv: ConsoleInvocation, keys: [String]?) throws -> [String] {
        let vm = inv.viewModel
        var applied: [String] = []
        if let preset = ConsoleNames.lookup(inv.string("preset"), in: ConsoleNames.presets) {
            preset.apply(to: &vm.docking.dockingConfig)
            applied.append("preset")
        }
        applied += try DockingSettings.apply(inv, keys: keys)
        // Diffusion is guided by DruseAF attention gradients (same rule as the Docking step).
        if vm.docking.dockingConfig.searchMethod == .diffusionGuided, vm.docking.scoringMethod != .druseAffinity {
            if inv.has("scoring") || inv.has("search") {
                inv.emit(.warning, "search=diffusion requires DruseAF scoring — using scoring=druseaf")
            }
            vm.docking.scoringMethod = .druseAffinity
        }
        return applied
    }

    /// Applies `key=value` docking settings to a draft and commits it only when every value is valid.
    private static func setDockSettings(_ inv: ConsoleInvocation) throws {
        guard !inv.values.isEmpty else {
            throw ConsoleError("dock set needs key=value settings, e.g. 'dock set population=200' — 'dock config' lists them, 'help dock' explains them")
        }
        let vm = inv.viewModel
        let applied = try applyDockOptions(inv, keys: nil)
        var lines: [String] = []
        if applied.contains("preset") {
            let config = vm.docking.dockingConfig
            lines.append("preset = \(inv.string("preset") ?? "") (population \(config.populationSize), generations \(config.generationsPerRun), runs \(config.numRuns))")
        }
        for key in applied where key != "preset" {
            guard let setting = DockingSettings.setting(key) else { continue }
            lines.append("\(key) = \(setting.read(vm.docking))")
        }
        inv.emit(.success, lines.joined(separator: "\n"))
    }

    /// Whether a docking run is in progress and what it is doing, for `dock status`.
    private static func dockStatus(_ vm: AppViewModel) -> String {
        let d = vm.docking
        if d.isBatchDocking {
            return "Batch docking: ligand \(d.batchProgress.current + 1)/\(d.batchProgress.total), generation \(d.dockingGeneration)/\(d.dockingTotalGenerations)"
        }
        if d.isDocking {
            let best = d.scoringMethod.isAffinityScore
                ? d.dockingBestPKi.map { String(format: "best pKd %.2f", $0) } ?? "no pose yet"
                : (d.dockingBestEnergy.isFinite ? String(format: "best %.2f kcal/mol", d.dockingBestEnergy) : "no pose yet")
            return "Docking \(vm.molecules.ligand?.name ?? "ligand"): generation \(d.dockingGeneration)/\(d.dockingTotalGenerations), \(best)"
        }
        if d.dockingResults.isEmpty { return "Idle — no docking results" }
        return "Idle — \(d.dockingResults.count) poses from the last run (\(ConsoleFormat.seconds(d.dockingDuration)))"
    }

    /// Fails early with an actionable message when the protein, pocket or ligand a run needs is missing.
    private static func requireDockingInputs(_ vm: AppViewModel, needsLigand: Bool) throws {
        try requireProtein(vm)
        if needsLigand, vm.molecules.ligand == nil {
            throw ConsoleError("no active ligand — use 'ligand smiles', 'ligand use' or 'load <file.sdf>'")
        }
        guard vm.docking.selectedPocket != nil else {
            throw ConsoleError("no pocket selected — use 'pocket detect', 'pocket residues' or 'pocket box center=x,y,z'")
        }
        guard !vm.docking.isDocking, !vm.docking.isBatchDocking else {
            throw ConsoleError("a docking run is already in progress ('dock stop' stops it)")
        }
    }

    /// Docks the active ligand with the session settings (plus per-run overrides) and awaits the run to report the best pose.
    private static func dockActiveLigand(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireDockingInputs(vm, needsLigand: true)
        try applyDockOptions(inv, keys: inlineDockKeys)
        let config = vm.docking.dockingConfig
        let method = vm.docking.scoringMethod
        inv.emit(.output, "Docking \(vm.molecules.ligand?.name ?? "ligand"): search=\(ConsoleNames.search(config.searchMethod)) "
                 + "scoring=\(ConsoleNames.scoring(method)) "
                 + (config.autoMode ? "budget=auto" : "\(config.populationSize)×\(config.generationsPerRun)×\(config.numRuns)"))
        guard let task = vm.runDocking() else {
            throw ConsoleError("docking needs a protein, a ligand and a pocket")
        }
        let succeeded = await inv.withProgress({ dockingProgressLine(vm) }) { await task.value }
        guard succeeded, let best = vm.docking.dockingResults.first else {
            throw ConsoleError("docking produced no valid poses — see the activity log")
        }
        let results = vm.docking.dockingResults
        let clusters = Set(results.map(\.clusterID)).count
        let name = vm.docking.originalDockingLigand?.name ?? vm.molecules.ligand?.name ?? "ligand"
        inv.emit(.success, "Docked \(name): best \(ConsoleFormat.score(best, vm.docking.scoringMethod)), \(results.count) poses, "
                 + "\(clusters) clusters in \(ConsoleFormat.seconds(vm.docking.dockingDuration))")
    }

    /// Live status of a single docking run: step, percentage and best score so far.
    private static func dockingProgressLine(_ vm: AppViewModel) -> String? {
        let docking = vm.docking
        guard docking.isDocking else { return nil }
        let progress = DockingProgress(docking)
        var line = "Docking: \(progress.summary) (\(progress.percentText))"
        if docking.dockingBestEnergy.isFinite {
            line += String(format: " · best %.2f kcal/mol", docking.dockingBestEnergy)
        }
        return line
    }

    /// Live status of `dock batch`: which ligand, plus the current run's progress.
    private static func batchProgressLine(_ vm: AppViewModel) -> String? {
        let docking = vm.docking
        guard docking.isBatchDocking, docking.batchProgress.total > 0 else { return nil }
        let ligand = min(docking.batchProgress.current + 1, docking.batchProgress.total)
        var line = "Batch docking ligand \(ligand)/\(docking.batchProgress.total)"
        if docking.isDocking {
            line += " · \(DockingProgress(docking).summary)"
        }
        return line
    }

    /// Live status of `screen`: pipeline phase and completed fraction.
    private static func screeningProgressLine(_ vm: AppViewModel) -> String? {
        let phase: String
        switch vm.docking.screeningState {
        case .preparing: phase = "preparing ligands"
        case .docking: phase = "docking"
        case .reranking: phase = "re-ranking hits"
        case .idle, .complete, .failed: return nil
        }
        let percent = Int((min(max(vm.docking.screeningProgress, 0), 1) * 100).rounded(.down))
        return "Screening: \(phase) (\(percent)%)"
    }

    /// Docks the selected Ligand Database entries one after another and prints the ranking.
    private static func dockBatch(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireDockingInputs(vm, needsLigand: false)
        let entries = try batchEntries(inv.string("ligands") ?? "all", in: vm)
        try applyDockOptions(inv, keys: inlineDockKeys)
        let start = Date()
        guard let task = vm.dockEntries(entries) else {
            throw ConsoleError("batch docking could not start — see the activity log")
        }
        await inv.withProgress({ batchProgressLine(vm) }) { await task.value }
        let method = vm.docking.scoringMethod
        let ranked = vm.docking.batchResults.filter { !$0.results.isEmpty }
        guard let best = ranked.first, let bestPose = best.results.first else {
            throw ConsoleError("batch docking produced no poses — see the activity log")
        }
        inv.emit(.output, batchTable(vm, top: 20))
        inv.emit(.success, "Batch docked \(ranked.count)/\(entries.count) ligands in \(ConsoleFormat.seconds(Date().timeIntervalSince(start))); "
                 + "best \(best.ligandName) \(ConsoleFormat.score(bestPose, method))")
    }

    /// Ligand Database entries selected by `ligands=`: all, prepared, or a comma-separated name list
    /// (a source molecule's enumerated forms stand in for the molecule itself).
    private static func batchEntries(_ selector: String, in vm: AppViewModel) throws -> [LigandEntry] {
        let dockable = vm.ligandDB.entries.filter { !$0.atoms.isEmpty }
        var entries: [LigandEntry]
        switch selector.lowercased() {
        case "all":
            entries = dockable
        case "prepared":
            entries = dockable.filter(\.isPrepared)
        default:
            let names = selector.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            var picked: [LigandEntry] = []
            for name in names {
                // A prepared molecule is a parent row without coordinates plus one dockable row per
                // chemical form (`parentName` set), so a name selects every dockable form of it.
                let named = vm.ligandDB.entries.filter {
                    $0.name.caseInsensitiveCompare(name) == .orderedSame
                        || $0.parentName?.caseInsensitiveCompare(name) == .orderedSame
                }
                guard !named.isEmpty else {
                    var message = "no Ligand Database entry named '\(name)'"
                    if let hint = ConsoleSuggest.closest(name.lowercased(), in: vm.ligandDB.entries.map { $0.name.lowercased() }) {
                        message += " — did you mean '\(hint)'?"
                    }
                    throw ConsoleError(message)
                }
                let forms = named.filter { !$0.atoms.isEmpty }
                guard !forms.isEmpty else {
                    throw ConsoleError("'\(named[0].name)' has no 3D coordinates — prepare it first ('ligand prep')")
                }
                picked.append(contentsOf: forms.filter { form in !picked.contains { $0.id == form.id } })
            }
            entries = picked
        }
        // An imported molecule keeps its file coordinates after preparation; once it has been
        // enumerated into forms, those forms are what gets docked, not the raw import as well.
        let enumerated = Set(vm.ligandDB.entries.compactMap { $0.parentName?.lowercased() })
        entries = entries.filter { $0.parentName != nil || !enumerated.contains($0.name.lowercased()) }
        guard !entries.isEmpty else {
            throw ConsoleError(selector.lowercased() == "prepared"
                               ? "no prepared Ligand Database entry — prepare entries first or use ligands=all"
                               : "no Ligand Database entry has 3D coordinates — use 'ligand db import' or 'ligand smiles'")
        }
        return entries
    }

    /// Ranking table of the last batch run, best score first.
    private static func batchTable(_ vm: AppViewModel, top: Int) -> String {
        let method = vm.docking.scoringMethod
        let rows = vm.docking.batchResults.prefix(top).enumerated().map { i, entry in
            [
                "\(i + 1)",
                entry.ligandName,
                entry.results.first.map { String(format: "%.2f", $0.displayScore(method: method)) } ?? "—",
                "\(entry.results.count)",
                "\(Set(entry.results.map(\.clusterID)).count)"
            ]
        }
        return ConsoleTable.render(["Rank", "Ligand", method.unitLabel, "Poses", "Clusters"], rows, rightAligned: [0, 2, 3, 4])
    }

    // MARK: results

    /// `results`: the ranked poses of the last run, one pose in detail, clusters and the batch ranking.
    private static func resultsCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "results",
            summary: "Ranked docking poses, per-pose score breakdown, clusters and batch ranking",
            category: .docking,
            forms: [
                ConsoleForm(path: [], summary: "Table of the best poses with score terms (ranks are those of 'results show' and 'pose show')",
                            options: [ConsoleParameter(name: "top", type: .int, summary: "Number of rows (default 10)"),
                                      ConsoleParameter(name: "sort", type: .choice(["score", "cluster", "rmsd"]),
                                                       summary: "score (default), cluster = best pose of each cluster, rmsd = closest to the input pose"),
                                      ConsoleParameter(name: "cluster", type: .int, summary: "Only the poses of this cluster")]) { inv in
                    try listResults(inv)
                },
                ConsoleForm(path: ["show"], summary: "Everything known about one pose: score terms, strain, OpenMM, GFN2-xTB, validity",
                            arguments: [ConsoleParameter(name: "n", type: .int, summary: "1-based rank from 'results'")]) { inv in
                    try showResult(inv)
                },
                ConsoleForm(path: ["clusters"], summary: "Binding-mode clusters (2 Å RMSD leader clustering) with their best and mean scores") { inv in
                    try listClusters(inv)
                },
                ConsoleForm(path: ["batch"], summary: "Ranking of the last 'dock batch' run",
                            options: [ConsoleParameter(name: "top", type: .int, summary: "Number of ligands (default 20)")]) { inv in
                    guard !inv.viewModel.docking.batchResults.isEmpty else {
                        throw ConsoleError("no batch results — run 'dock batch'")
                    }
                    inv.emit(.output, batchTable(inv.viewModel, top: max(1, inv.int("top") ?? 20)))
                }
            ],
            examples: ["results", "results top=25", "results sort=cluster", "results sort=rmsd top=5", "results cluster=0",
                       "results show 1", "results clusters", "results batch top=50"]
        )
    }

    private static func requireResults(_ vm: AppViewModel) throws -> [DockingResult] {
        let results = vm.docking.dockingResults
        guard !results.isEmpty else { throw ConsoleError("no docking results — run 'dock'") }
        return results
    }

    /// Prints the top poses, sorted and optionally restricted to cluster representatives.
    private static func listResults(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let results = try requireResults(vm)
        let top = max(1, inv.int("top") ?? 10)
        let reference = vm.inputPoseReference
        var ranked = Array(results.enumerated())
        if let cluster = inv.int("cluster") {
            ranked = ranked.filter { $0.element.clusterID == cluster }
            guard !ranked.isEmpty else {
                let ids = Set(results.map(\.clusterID)).sorted().map(String.init).joined(separator: ", ")
                throw ConsoleError("no cluster \(cluster) — clusters: \(ids)")
            }
        }
        switch inv.string("sort") ?? "score" {
        case "cluster":
            var seen = Set<Int>()
            ranked = ranked.filter { seen.insert($0.element.clusterID).inserted }
        case "rmsd":
            guard let reference else {
                throw ConsoleError("sort=rmsd needs a placed input pose of the docked ligand (none after a project load or batch, or when the ligand was built from SMILES or a prepared conformer)")
            }
            ranked.sort { (rmsd($0.element, reference) ?? .infinity) < (rmsd($1.element, reference) ?? .infinity) }
        default:
            break
        }
        let method = vm.docking.scoringMethod
        let name = vm.docking.originalDockingLigand?.name ?? vm.molecules.ligand?.name ?? "ligand"
        inv.emit(.output, "\(name) — \(results.count) poses, \(Set(results.map(\.clusterID)).count) clusters (\(ConsoleNames.scoring(method))):\n"
                 + posesTable(Array(ranked.prefix(top)), method: method, reference: reference))
    }

    /// Shared pose table (score, cluster, RMSD to the input pose, score terms, strain, validity).
    private static func posesTable(_ rows: [(offset: Int, element: DockingResult)], method: ScoringMethod,
                                   reference: [SIMD3<Float>]?) -> String {
        let f = { (value: Float) in String(format: "%.2f", value) }
        let rmsdText = { (r: DockingResult) in reference.flatMap { rmsd(r, $0) }.map { String(format: "%.2f", $0) } ?? "—" }
        let valid = { (r: DockingResult) in r.validity.map { $0.passed ? "ok" : "fail" } ?? "—" }
        if method.isAffinityScore {
            let cells = rows.map { i, r in
                ["\(i + 1)", f(r.displayScore(method: method)), String(format: "%.0f%%", r.afConfidence * 100),
                 "\(r.clusterID)", rmsdText(r), valid(r)]
            }
            return ConsoleTable.render(["Rank", "pKd", "Conf", "Cluster", "RMSD", "Valid"], cells, rightAligned: [0, 1, 2, 3, 4])
        }
        let cells = rows.map { i, r in
            ["\(i + 1)", f(r.displayScore(method: method)), "\(r.clusterID)", rmsdText(r),
             f(r.stericEnergy), f(r.hydrophobicEnergy), f(r.hbondEnergy), f(r.torsionPenalty), f(r.intraAndOtherEnergy),
             r.strainEnergy.map(f) ?? "—", valid(r)]
        }
        return ConsoleTable.render(
            ["Rank", "kcal/mol", "Cluster", "RMSD", "Steric", "Hydroph", "HBond", "Torsion", "Other", "Strain", "Valid"],
            cells, rightAligned: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
    }

    /// Everything known about one pose: every score term, ML and refinement annotations, cluster and validity.
    private static func showResult(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let results = try requireResults(vm)
        let n = inv.requiredInt("n")
        guard results.indices.contains(n - 1) else { throw ConsoleError("no pose #\(n) — \(results.count) available") }
        let r = results[n - 1]
        let method = vm.docking.scoringMethod
        let f = { (value: Float) in String(format: "%.2f", value) }
        var rows: [[String]] = []
        rows.append(["Score", "\(ConsoleFormat.score(r, method))  (rank \(n)/\(results.count), cluster \(r.clusterID), member \(r.clusterRank + 1))"])
        if method.isAffinityScore {
            rows.append(["Confidence", String(format: "%.0f%%  (ranking energy %.3f = −pKd × confidence)", r.afConfidence * 100, r.energy)])
        } else {
            rows.append(["Steric", f(r.stericEnergy) + " kcal/mol  (gauss + repulsion)"])
            rows.append(["Hydrophobic", f(r.hydrophobicEnergy) + " kcal/mol"])
            rows.append(["H-bond", f(r.hbondEnergy) + " kcal/mol"])
            rows.append(["Torsion", f(r.torsionPenalty) + " kcal/mol  (rotatable-bond entropy)"])
            rows.append(["Intra/other", f(r.intraAndOtherEnergy) + " kcal/mol  (ligand internal, box/constraint penalties, corrections)"])
            if r.drusinaCorrection != 0 { rows.append(["Drusina corr.", f(r.drusinaCorrection) + " kcal/mol  (π, halogen, metal, salt-bridge terms)"]) }
        }
        if r.constraintPenalty != 0 { rows.append(["Constraints", f(r.constraintPenalty) + " kcal/mol penalty"]) }
        rows.append(["Strain", r.strainEnergy.map { f($0) + " kcal/mol (MMFF94 docked − relaxed)" } ?? "not computed ('strain')"])
        rows.append(["OpenMM", r.refinementEnergy.map { f($0) + " kcal/mol interaction energy (annotation)" } ?? "not refined ('refine openmm')"])
        if let energy = r.gfn2Energy {
            var text = String(format: "%.1f kcal/mol total", energy)
            if let dispersion = r.gfn2DispersionEnergy { text += ", D4 " + f(dispersion) }
            if let solvation = r.gfn2SolvationEnergy { text += ", solvation " + f(solvation) }
            if let converged = r.gfn2Converged { text += converged ? ", converged" : ", not converged" }
            if let steps = r.gfn2OptSteps { text += " in \(steps) steps" }
            rows.append(["GFN2-xTB", text])
        } else {
            rows.append(["GFN2-xTB", "not refined ('refine gfn2')"])
        }
        if let reference = vm.inputPoseReference, let value = rmsd(r, reference) {
            rows.append(["RMSD", String(format: "%.2f Å to the input pose", value)])
        }
        if let form = r.formLabel {
            rows.append(["Form", form + (r.formPopulation.map { String(format: " (%.0f%% population)", $0 * 100) } ?? "")])
        }
        if let validity = r.validity {
            rows.append(["Validity", validity.passed ? "passed all PoseBusters-style checks" : "FAILED: " + validity.failures.joined(separator: ", ")])
            var geometry = String(format: "worst bond ratio %.2f", validity.worstBondRatio)
            if validity.minProteinDistance < .greatestFiniteMagnitude {
                geometry += String(format: ", closest protein atom %.2f Å", validity.minProteinDistance)
            }
            if validity.minInternalDistance < .greatestFiniteMagnitude {
                geometry += String(format: ", closest internal pair %.2f Å", validity.minInternalDistance)
            }
            rows.append(["Geometry", geometry])
        } else {
            rows.append(["Validity", "not checked"])
        }
        if !r.transformedAtomPositions.isEmpty {
            let centroid = r.transformedAtomPositions.reduce(SIMD3<Float>.zero, +) / Float(r.transformedAtomPositions.count)
            rows.append(["Centroid", ConsoleFormat.vector(centroid) + " Å"])
        }
        rows.append(["Torsions", "\(r.pose.torsions.count) rotatable bond(s)" + (r.pose.chiAngles.isEmpty ? "" : ", \(r.pose.chiAngles.count) receptor chi angle(s)")])
        rows.append(["Generation", "\(r.generation)"])
        let name = vm.docking.originalDockingLigand?.name ?? vm.molecules.ligand?.name ?? "ligand"
        inv.emit(.output, "Pose #\(n) — \(name)\n" + rows.map { "  " + $0[0].padding(toLength: 14, withPad: " ", startingAt: 0) + $0[1] }.joined(separator: "\n"))
    }

    /// One row per pose cluster: size, best rank, best and mean score, RMSD of its representative.
    private static func listClusters(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let results = try requireResults(vm)
        let method = vm.docking.scoringMethod
        let reference = vm.inputPoseReference
        var order: [Int] = []
        var members: [Int: [(rank: Int, result: DockingResult)]] = [:]
        for (i, r) in results.enumerated() {
            if members[r.clusterID] == nil { order.append(r.clusterID) }
            members[r.clusterID, default: []].append((i + 1, r))
        }
        let rows = order.map { id -> [String] in
            let group = members[id] ?? []
            let scores = group.map { $0.result.displayScore(method: method) }
            let best = group[0]
            return [
                "\(id)", "\(group.count)", "\(best.rank)",
                String(format: "%.2f", best.result.displayScore(method: method)),
                String(format: "%.2f", scores.reduce(0, +) / Float(max(scores.count, 1))),
                reference.flatMap { rmsd(best.result, $0) }.map { String(format: "%.2f", $0) } ?? "—"
            ]
        }
        inv.emit(.output, ConsoleTable.render(["Cluster", "Poses", "Best rank", "Best \(method.unitLabel)", "Mean", "RMSD"],
                                              rows, rightAligned: [0, 1, 2, 3, 4, 5]))
    }

    /// Heavy-atom positions of the docked ligand's input pose (the crystal pose when redocking).

    /// Heavy-atom RMSD of a pose to `reference` (same atom order, no symmetry correction).
    private static func rmsd(_ result: DockingResult, _ reference: [SIMD3<Float>]) -> Float? {
        let positions = result.transformedAtomPositions
        guard positions.count == reference.count, !positions.isEmpty else { return nil }
        let sum = zip(positions, reference).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
        return sqrt(sum / Float(positions.count))
    }

    // MARK: pose

    /// `pose`: shows one docked pose in the 3D view as the active ligand.
    private static func poseCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "pose",
            summary: "Show a docked pose in the 3D view",
            category: .docking,
            forms: [
                ConsoleForm(path: ["show"], summary: "Display pose n with its interactions",
                            arguments: [ConsoleParameter(name: "n", type: .int, summary: "1-based rank from 'results'")],
                            isGUIOnly: true) { inv in
                    let vm = inv.viewModel
                    let index = inv.requiredInt("n") - 1
                    guard vm.docking.dockingResults.indices.contains(index) else {
                        throw ConsoleError("no pose #\(index + 1) — \(vm.docking.dockingResults.count) available")
                    }
                    vm.showDockingPose(at: index)
                    inv.emit(.success, "Showing pose #\(index + 1): \(ConsoleFormat.score(vm.docking.dockingResults[index], vm.docking.scoringMethod)), \(vm.docking.currentInteractions.count) interactions")
                }
            ],
            examples: ["pose show 1"]
        )
    }

    // MARK: screen

    /// `screen`: virtual screening of a SMILES or SDF library, its hits, and cancellation.
    private static func screenCommand() -> ConsoleCommand {
        let defaults = VirtualScreeningPipeline.ScreeningConfig()
        let options = [
            ConsoleParameter(name: "max", type: .int, summary: "Screen at most this many molecules (default \(defaults.maxMolecules))"),
            ConsoleParameter(name: "max-rotatable", type: .int, summary: "Skip molecules with more rotatable bonds (default \(defaults.maxRotatableBonds))"),
            ConsoleParameter(name: "grid-spacing", type: .float, summary: "Grid spacing in Å (default \(DockingSettings.format(defaults.gridSpacing)), coarser than 'dock')"),
            ConsoleParameter(name: "prescreen-poses", type: .int, summary: "Rigid poses per ligand in the batched Metal prescreen (default \(defaults.batchedPosesPerLigand), min 16)"),
            ConsoleParameter(name: "shortlist", type: .float, summary: "Fraction of prescreened ligands docked with the GA (default \(DockingSettings.format(defaults.gaRefinementFraction)))"),
            ConsoleParameter(name: "shortlist-min", type: .int, summary: "Always GA-dock at least this many ligands (default \(defaults.minimumGARefinementCount))"),
            ConsoleParameter(name: "shortlist-max", type: .int, summary: "Never GA-dock more than this many ligands (default \(defaults.gaRefinementMaxLigands))"),
            ConsoleParameter(name: "population", type: .int, summary: "GA population per ligand (default \(defaults.populationSize), min 20)"),
            ConsoleParameter(name: "generations", type: .int, summary: "GA generations per ligand (default \(defaults.numGenerations), min 10)"),
            ConsoleParameter(name: "strain", type: .bool, summary: "MMFF94 strain penalty on hits (default \(defaults.applyStrainPenalty ? "on" : "off"))"),
            ConsoleParameter(name: "openmm-top", type: .int, summary: "OpenMM-refine the top N hits, 0 = off (default \(defaults.openMMRefineTopN))"),
            ConsoleParameter(name: "gfn2-top", type: .int, summary: "GFN2-xTB single-point energy of the top N hits, 0 = off (default \(defaults.gfn2RefineTopN))"),
            ConsoleParameter(name: "admet", type: .bool, summary: "Predict ADMET properties of the top hits when the models are installed (default \(defaults.admetFilter ? "on" : "off"))"),
            ConsoleParameter(name: "top", type: .int, summary: "Hits to print (default 20)")
        ]
        return ConsoleCommand(
            name: "screen",
            summary: "Virtual screening of a SMILES or SDF library against the selected pocket",
            category: .docking,
            forms: [
                ConsoleForm(path: [], summary: "Prepare (RDKit 3D), prescreen, GA-dock a shortlist, refine and rank a library",
                            arguments: [ConsoleParameter(name: "library", type: .path, summary: "Library file (.smi or .sdf)")],
                            options: options) { inv in
                    try await screenLibrary(inv)
                },
                ConsoleForm(path: ["hits"], summary: "Print the hits of the last screening run",
                            options: [ConsoleParameter(name: "top", type: .int, summary: "Hits to print (default 20)")]) { inv in
                    let hits = inv.viewModel.docking.screeningHits
                    guard !hits.isEmpty else { throw ConsoleError("no screening hits — run 'screen <library>'") }
                    inv.emit(.output, hitsTable(hits, top: max(1, inv.int("top") ?? 20)))
                },
                ConsoleForm(path: ["stop"], summary: "Stop a screening run in progress") { inv in
                    guard inv.viewModel.docking.screeningState.isActive else {
                        inv.emit(.output, "No screening run in progress")
                        return
                    }
                    inv.viewModel.cancelScreening()
                    inv.emit(.success, "Screening stopped")
                }
            ],
            examples: ["screen enamine_subset.smi top=50", "screen library.sdf max=5000 shortlist=0.2 openmm-top=50",
                       "screen hits.smi population=80 generations=60 gfn2-top=10 admet=off", "screen hits top=100"]
        )
    }

    /// Screening configuration from the `screen` options, validated.
    private static func screeningConfig(_ inv: ConsoleInvocation) throws -> VirtualScreeningPipeline.ScreeningConfig {
        var config = VirtualScreeningPipeline.ScreeningConfig()
        func count(_ name: String, min lower: Int, _ assign: (Int) -> Void) throws {
            guard let value = inv.int(name) else { return }
            guard value >= lower else { throw ConsoleError("\(name) must be at least \(lower)") }
            assign(value)
        }
        try count("max", min: 1) { config.maxMolecules = $0 }
        try count("max-rotatable", min: 0) { config.maxRotatableBonds = $0 }
        try count("prescreen-poses", min: 16) { config.batchedPosesPerLigand = $0 }
        try count("shortlist-min", min: 1) { config.minimumGARefinementCount = $0 }
        try count("shortlist-max", min: 1) { config.gaRefinementMaxLigands = $0 }
        try count("population", min: 20) { config.populationSize = $0 }
        try count("generations", min: 10) { config.numGenerations = $0 }
        try count("openmm-top", min: 0) { config.openMMRefineTopN = $0 }
        try count("gfn2-top", min: 0) { config.gfn2RefineTopN = $0 }
        if let spacing = inv.double("grid-spacing") {
            guard (0.2...1.0).contains(spacing) else { throw ConsoleError("grid-spacing must be between 0.2 and 1 Å") }
            config.gridSpacing = Float(spacing)
        }
        if let fraction = inv.double("shortlist") {
            guard (0...1).contains(fraction) else { throw ConsoleError("shortlist must be a fraction between 0 and 1") }
            config.gaRefinementFraction = Float(fraction)
        }
        if config.minimumGARefinementCount > config.gaRefinementMaxLigands {
            throw ConsoleError("shortlist-min (\(config.minimumGARefinementCount)) exceeds shortlist-max (\(config.gaRefinementMaxLigands))")
        }
        if let strain = inv.bool("strain") { config.applyStrainPenalty = strain }
        if let admet = inv.bool("admet") { config.admetFilter = admet }
        return config
    }

    /// Runs the screening pipeline on a library file and awaits it before printing the hits.
    private static func screenLibrary(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireDockingInputs(vm, needsLigand: false)
        guard !vm.docking.screeningState.isActive else {
            throw ConsoleError("a screening run is already in progress ('screen stop' stops it)")
        }
        let config = try screeningConfig(inv)
        let url = inv.requiredURL("library")
        try requireReadable(url)
        let library = try await readLibrary(url)
        guard !library.isEmpty else { throw ConsoleError("no molecules found in \(url.lastPathComponent)") }

        inv.emit(.output, "Screening \(min(library.count, config.maxMolecules)) molecules from \(url.lastPathComponent)…")
        let start = Date()
        guard let task = vm.startScreening(library: library, config: config) else {
            throw ConsoleError("screening could not start — see the activity log")
        }
        await inv.withProgress({ screeningProgressLine(vm) }) { await task.value }
        if case .failed(let reason) = vm.docking.screeningState {
            throw ConsoleError("screening failed: \(reason)")
        }
        let hits = vm.docking.screeningHits
        guard !hits.isEmpty else { throw ConsoleError("screening produced no hits") }

        inv.emit(.output, hitsTable(hits, top: max(1, inv.int("top") ?? 20)))
        inv.emit(.success, "Screened \(library.count) molecules in \(ConsoleFormat.seconds(Date().timeIntervalSince(start))): \(hits.count) hits")
    }

    /// Table of the best screening hits with score and properties.
    private static func hitsTable(_ hits: [VirtualScreeningPipeline.ScreeningHit], top: Int) -> String {
        // A diverged refinement leaves a non-finite energy, which means "no value" here.
        let f = { (value: Float?) in value.flatMap { $0.isFinite ? String(format: "%.2f", $0) : nil } ?? "—" }
        let rows = hits.prefix(top).enumerated().map { i, hit in
            [
                "\(i + 1)",
                hit.name,
                f(hit.bestEnergy),
                f(hit.prescreenEnergy),
                f(hit.strainEnergy),
                f(hit.openMMEnergy),
                hit.gfn2Energy.flatMap { $0.isFinite ? String(format: "%.1f", $0) : nil } ?? "—",
                hit.smiles.count > 40 ? String(hit.smiles.prefix(39)) + "…" : hit.smiles
            ]
        }
        return ConsoleTable.render(["Rank", "Name", "kcal/mol", "Prescreen", "Strain", "OpenMM", "GFN2", "SMILES"],
                                   rows, rightAligned: [0, 2, 3, 4, 5, 6])
    }

    /// (name, SMILES) pairs: `.smi` lines are "SMILES [name]"; SDF molecules use their SMILES
    /// property or a SMILES derived from the connection table.
    private static func readLibrary(_ url: URL) async throws -> [(name: String, smiles: String)] {
        switch url.pathExtension.lowercased() {
        case "sdf", "mol":
            let molecules = try await Task.detached { try SDFParser.parse(url: url) }.value
            return molecules.enumerated().compactMap { i, molecule in
                guard let smiles = molecule.smiles ?? RDKitBridge.atomsBondsToSMILES(atoms: molecule.atoms, bonds: molecule.bonds) else {
                    return nil
                }
                return (molecule.name.isEmpty ? "Mol_\(i + 1)" : molecule.name, smiles)
            }
        case "smi", "smiles", "txt":
            let content = try String(contentsOf: url, encoding: .utf8)
            return content.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
                .enumerated()
                .map { i, line in
                    let parts = line.split(maxSplits: 1, whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
                    return (parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : "Mol_\(i + 1)", parts[0])
                }
        default:
            throw ConsoleError("unsupported library '.\(url.pathExtension)' (expected smi or sdf)")
        }
    }
}

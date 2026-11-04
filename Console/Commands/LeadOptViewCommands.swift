// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - leadopt, view
//
// `leadopt` drives the Lead Opt step (reference → generation settings → analogs → mini-docking →
// hand-off) with every setting of that step; `view` changes the 3D view. View settings live in
// WorkspaceState, so in headless mode they are kept (and used by `view snapshot`) while camera
// commands, which need a window, are skipped with a warning.

extension DruseInterpreter {

    /// `leadopt`: reference, generation settings, analog generation, analog docking and hand-off to the Ligand Database.
    static func leadOptimizationCommands() -> [ConsoleCommand] {
        [
            ConsoleCommand(
                name: "leadopt",
                summary: "Lead optimization: pick a reference, set generation and ADMET options, generate, dock and hand off analogs",
                category: .leadOptimization,
                forms: [
                    ConsoleForm(path: ["reference"], summary: "Set the optimization reference: a docked pose (default best; the active ligand when nothing is docked) or a ligand by name",
                                options: [
                                    ConsoleParameter(name: "pose", type: .int, summary: "1-based pose rank from 'results' (default 1)"),
                                    ConsoleParameter(name: "ligand", type: .string,
                                                     summary: "Active ligand or Ligand Database entry name, used without a pose score")
                                ]) { inv in
                        try selectReference(inv)
                    },
                    ConsoleForm(path: ["set"], summary: "Change generation parameters, property directions and ADMET filters (kept for later runs)",
                                options: leadOptParameterOptions) { inv in
                        guard leadOptParameterOptions.contains(where: { inv.has($0.name) }) else {
                            throw ConsoleError("leadopt set needs at least one option: "
                                               + leadOptParameterOptions.map { $0.name + "=" }.joined(separator: " "))
                        }
                        try applyLeadOptParameters(inv)
                        inv.emit(.success, "Lead-optimization settings updated — 'leadopt config' shows them")
                    },
                    ConsoleForm(path: ["config"], summary: "Show the reference, generation parameters and ADMET filters") { inv in
                        inv.emit(.output, leadOptConfig(inv.viewModel))
                    },
                    ConsoleForm(path: ["defaults"], summary: "Restore every generation parameter and filter to its default") { inv in
                        try requireLeadOptIdle(inv.viewModel)
                        inv.viewModel.leadOpt.resetParameters()
                        inv.emit(.output, leadOptConfig(inv.viewModel))
                        inv.emit(.success, "Lead-optimization settings reset to defaults")
                    },
                    ConsoleForm(path: ["generate"], summary: "Generate property-directed analogs (options as in 'leadopt set'); passing analogs go to the Ligand Database",
                                options: leadOptParameterOptions) { inv in
                        try await generateAnalogs(inv)
                    },
                    ConsoleForm(path: ["list"], summary: "List analogs with status, score and property deltas vs the reference",
                                options: [
                                    ConsoleParameter(name: "status", type: .choice(["all"] + AnalogStatusName.all.map(\.name)),
                                                     summary: "Only analogs with this status (default all)"),
                                    ConsoleParameter(name: "top", type: .int, summary: "Show at most N rows")
                                ]) { inv in
                        try listAnalogs(inv)
                    },
                    ConsoleForm(path: ["show"], summary: "Details of one analog: SMILES, descriptors and deltas, ADMET predictions, docking score",
                                arguments: [ConsoleParameter(name: "n", type: .int, summary: "1-based row from 'leadopt list'")]) { inv in
                        try await showAnalog(inv)
                    },
                    ConsoleForm(path: ["dock"], summary: "Mini-dock the new analogs with the reference's scoring function and pocket (reduced budget)") { inv in
                        try await dockAnalogs(inv)
                    },
                    ConsoleForm(path: ["stop"], summary: "Stop a running analog generation or analog docking") { inv in
                        let vm = inv.viewModel
                        guard vm.leadOpt.isGenerating || vm.leadOpt.isDocking else {
                            inv.emit(.output, "No lead-optimization run in progress")
                            return
                        }
                        vm.cancelLeadOptimization()
                        inv.emit(.success, "Lead optimization stopped")
                    },
                    ConsoleForm(path: ["send", "database"], summary: "Add the analogs to the Ligand Database (duplicates by SMILES are skipped)",
                                options: [ConsoleParameter(name: "filtered", type: .bool,
                                                           summary: "Also add analogs rejected by the ADMET filters (default off)")]) { inv in
                        let vm = inv.viewModel
                        try requireAnalogs(vm)
                        let added = vm.addAnalogsToLigandDatabase(includeFiltered: inv.bool("filtered") ?? false)
                        inv.emit(.success, added > 0
                                 ? "Added \(added) analog(s) to the Ligand Database (\(vm.ligandDB.entries.count) entries)"
                                 : "Every analog is already in the Ligand Database")
                    },
                    ConsoleForm(path: ["send", "docking"], summary: "Add the analogs to the Ligand Database and open the Docking step for a batch run") { inv in
                        let vm = inv.viewModel
                        try requireAnalogs(vm)
                        try requireLeadOptIdle(vm)
                        let before = vm.ligandDB.entries.count
                        vm.sendAnalogsToDocking()
                        inv.emit(.success, "\(vm.ligandDB.entries.count - before) analog(s) added; dock them with 'dock batch'")
                    },
                    ConsoleForm(path: ["clear"], summary: "Remove all generated analogs (the reference and settings are kept)") { inv in
                        let vm = inv.viewModel
                        try requireLeadOptIdle(vm)
                        let count = vm.leadOpt.analogs.count
                        vm.clearLeadOptimizationAnalogs()
                        inv.emit(.success, "Cleared \(count) analog(s)")
                    }
                ],
                examples: [
                    "leadopt reference pose=1",
                    "leadopt reference ligand=imatinib",
                    "leadopt set count=25 similarity=0.6 polarity=0.5 herg=on max-logp=5",
                    "leadopt generate count=30 keep-scaffold=off",
                    "leadopt list status=docked top=10",
                    "leadopt show 1",
                    "leadopt dock",
                    "leadopt send database"
                ]
            )
        ]
    }

    // MARK: leadopt parameters

    /// Options shared by `leadopt set` and `leadopt generate`; each maps to one setting of the
    /// Lead Opt step (`LeadOptimizationState`).
    private static let leadOptParameterOptions: [ConsoleParameter] = [
        ConsoleParameter(name: "count", type: .int, summary: "Maximum number of analogs to generate (1–500, default 50)"),
        ConsoleParameter(name: "similarity", type: .float, summary: "Minimum Tanimoto similarity to the reference (0–1, default 0.7)"),
        ConsoleParameter(name: "keep-scaffold", type: .bool, summary: "Edit R-groups only, never ring atoms (default on)"),
        ConsoleParameter(name: "polarity", type: .float, summary: "Polarity direction: -1 less polar … +1 more polar (default 0)"),
        ConsoleParameter(name: "rigidity", type: .float, summary: "Rigidity direction: -1 more flexible … +1 more rigid (default 0)"),
        ConsoleParameter(name: "lipophilicity", type: .float, summary: "Lipophilicity direction: -1 hydrophilic … +1 lipophilic (default 0)"),
        ConsoleParameter(name: "size", type: .float, summary: "Size direction: -1 smaller … +1 larger (default 0)"),
        ConsoleParameter(name: "lipinski", type: .bool, summary: "Reject analogs violating Lipinski's rule of five (default off)"),
        ConsoleParameter(name: "veber", type: .bool, summary: "Reject analogs violating Veber's rules (default off)"),
        ConsoleParameter(name: "herg", type: .bool, summary: "Reject analogs with predicted hERG liability > 0.5 (default off)"),
        ConsoleParameter(name: "cyp", type: .bool, summary: "Reject analogs with predicted CYP2D6 or CYP3A4 inhibition > 0.5 (default off)"),
        ConsoleParameter(name: "max-logp", type: .float, summary: "Reject analogs with logP above this value (default 7)")
    ]

    /// Validates every given parameter option first, then writes them all, so a bad value
    /// leaves the settings untouched.
    private static func applyLeadOptParameters(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireLeadOptIdle(vm)
        if let count = inv.int("count"), !(1...500).contains(count) {
            throw ConsoleError("count must be between 1 and 500")
        }
        if let similarity = inv.double("similarity"), !(0...1).contains(similarity) {
            throw ConsoleError("similarity must be between 0 and 1")
        }
        for name in ["polarity", "rigidity", "lipophilicity", "size"] {
            if let value = inv.double(name), !(-1...1).contains(value) {
                throw ConsoleError("\(name) must be between -1 and 1")
            }
        }
        if let maxLogP = inv.double("max-logp"), !(-5...15).contains(maxLogP) {
            throw ConsoleError("max-logp must be between -5 and 15")
        }
        if let count = inv.int("count") { vm.leadOpt.analogCount = count }
        if let similarity = inv.double("similarity") { vm.leadOpt.similarityThreshold = Float(similarity) }
        if let keep = inv.bool("keep-scaffold") { vm.leadOpt.keepScaffold = keep }
        if let value = inv.double("polarity") { vm.leadOpt.polarityDirection = Float(value) }
        if let value = inv.double("rigidity") { vm.leadOpt.rigidityDirection = Float(value) }
        if let value = inv.double("lipophilicity") { vm.leadOpt.lipophilicityDirection = Float(value) }
        if let value = inv.double("size") { vm.leadOpt.sizeDirection = Float(value) }
        if let on = inv.bool("lipinski") { vm.leadOpt.filterLipinski = on }
        if let on = inv.bool("veber") { vm.leadOpt.filterVeber = on }
        if let on = inv.bool("herg") { vm.leadOpt.filterHERG = on }
        if let on = inv.bool("cyp") { vm.leadOpt.filterCYP = on }
        if let maxLogP = inv.double("max-logp") { vm.leadOpt.maxLogP = Float(maxLogP) }
        if (vm.leadOpt.filterHERG || vm.leadOpt.filterCYP), !vm.admetPredictor.isAvailable {
            inv.emit(.warning, "ADMET models are not loaded — the hERG/CYP filters cannot reject anything")
        }
    }

    /// The `leadopt config` report: reference, parameters in console option spelling, filters.
    private static func leadOptConfig(_ vm: AppViewModel) -> String {
        let lead = vm.leadOpt
        let onOff = { (on: Bool) in on ? "on" : "off" }
        let direction = { (value: Float) in String(format: "%+.1f", value) }
        var reference = "none — run 'leadopt reference'"
        if lead.hasReference {
            reference = lead.referenceName
                + (lead.referenceScore.map { String(format: " (%.2f %@)", $0, lead.scoringMethod.unitLabel) } ?? " (no pose score)")
        }
        let rows: [[String]] = [
            ["reference", reference],
            ["count", "\(lead.analogCount)"],
            ["similarity", String(format: "%.2f", lead.similarityThreshold)],
            ["keep-scaffold", onOff(lead.keepScaffold)],
            ["polarity", direction(lead.polarityDirection)],
            ["rigidity", direction(lead.rigidityDirection)],
            ["lipophilicity", direction(lead.lipophilicityDirection)],
            ["size", direction(lead.sizeDirection)],
            ["lipinski", onOff(lead.filterLipinski)],
            ["veber", onOff(lead.filterVeber)],
            ["herg", onOff(lead.filterHERG)],
            ["cyp", onOff(lead.filterCYP)],
            ["max-logp", String(format: "%.1f", lead.maxLogP)],
            ["analogs", lead.analogs.isEmpty ? "none"
                : "\(lead.analogs.count) (\(lead.passedFilterCount) passed filters, \(lead.dockedAnalogCount) docked)"]
        ]
        return ConsoleTable.render(["Setting", "Value"], rows)
    }

    // MARK: leadopt reference

    /// Sets the optimization reference from a docked pose or a Ligand Database entry.
    private static func selectReference(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireLeadOptIdle(vm)
        if inv.has("pose") && inv.has("ligand") {
            throw ConsoleError("give either pose= or ligand=, not both")
        }
        let unscoredLigand: Molecule?
        if let name = inv.string("ligand") {
            unscoredLigand = try referenceLigand(named: name, in: vm)
        } else if !inv.has("pose"), vm.docking.dockingResults.isEmpty {
            unscoredLigand = vm.molecules.ligand
        } else {
            unscoredLigand = nil
        }
        if let ligand = unscoredLigand {
            guard vm.selectReferenceForOptimization(result: nil, ligand: ligand) else {
                throw ConsoleError(AppViewModel.referenceFailureMessage(for: ligand.name))
            }
        } else {
            let index = (inv.int("pose") ?? 1) - 1
            guard vm.docking.dockingResults.indices.contains(index) else {
                throw ConsoleError(vm.docking.dockingResults.isEmpty
                                   ? "no docking results — run 'dock' first, or use 'leadopt reference ligand=<name>'"
                                   : "no pose #\(index + 1) — \(vm.docking.dockingResults.count) available")
            }
            guard let ligand = vm.docking.originalDockingLigand ?? vm.molecules.ligand else {
                throw ConsoleError("the docked ligand is no longer loaded")
            }
            guard vm.selectReferenceForOptimization(result: vm.docking.dockingResults[index], ligand: ligand) else {
                throw ConsoleError(AppViewModel.referenceFailureMessage(for: ligand.name))
            }
        }
        let lead = vm.leadOpt
        let score = lead.referenceScore.map { String(format: "%.2f %@", $0, lead.scoringMethod.unitLabel) } ?? "no pose score"
        inv.emit(.success, "Reference: \(lead.referenceName) (\(score)), \(lead.referenceSMILES)")
    }

    /// The active ligand when its name matches, else the Ligand Database entry of that name
    /// (case-insensitive), as a Molecule carrying the entry's SMILES.
    private static func referenceLigand(named name: String, in vm: AppViewModel) throws -> Molecule {
        if let active = vm.molecules.ligand, active.name.caseInsensitiveCompare(name) == .orderedSame {
            return active
        }
        guard let entry = vm.ligandDB.entries.first(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame }) else {
            var message = "no ligand named '\(name)' (neither the active ligand nor a Ligand Database entry)"
            let names = ([vm.molecules.ligand?.name].compactMap { $0 } + vm.ligandDB.entries.map(\.name))
            if let hint = ConsoleSuggest.closest(name.lowercased(), in: names.map { $0.lowercased() }),
               let original = names.first(where: { $0.lowercased() == hint }) {
                message += " — did you mean '\(original)'?"
            }
            throw ConsoleError(message)
        }
        return Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.name, smiles: entry.smiles)
    }

    // MARK: leadopt generate / dock

    /// Generates analogs of the reference and awaits the generation before listing them.
    private static func generateAnalogs(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        guard vm.leadOpt.hasReference else {
            throw ConsoleError("no reference — run 'leadopt reference' first")
        }
        try applyLeadOptParameters(inv)
        let databaseBefore = vm.ligandDB.entries.count
        guard let task = vm.generateOptimizedAnalogs() else {
            throw ConsoleError("analog generation could not start: the reference SMILES of \(vm.leadOpt.referenceName) could not be tokenized")
        }
        await inv.withProgress({
            guard vm.leadOpt.isGenerating else { return nil }
            return "Generating analogs (\(Int((min(max(vm.leadOpt.generationProgress, 0), 1) * 100).rounded(.down)))%)"
        }) {
            await task.value
        }
        let active = vm.leadOpt.analogs.filter { $0.status == .generated }.count
        guard active > 0 else {
            throw ConsoleError(vm.leadOpt.analogs.isEmpty
                ? "no analogs could be generated — try keep-scaffold=off or a lower similarity"
                : "all \(vm.leadOpt.analogs.count) analogs were filtered out — loosen the ADMET filters ('leadopt config')")
        }
        let filtered = vm.leadOpt.analogs.count - active
        inv.emit(.output, analogTable(vm, analogs: Array(vm.leadOpt.analogs.enumerated())))
        let added = vm.ligandDB.entries.count - databaseBefore
        inv.emit(.success, "Generated \(active) analogs of \(vm.leadOpt.referenceName)" + (filtered > 0 ? " (\(filtered) filtered)" : "")
                 + "; \(added) added to the Ligand Database")
    }

    /// Mini-docks the new analogs with the reference's pocket and scoring, awaiting the whole batch.
    private static func dockAnalogs(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireAnalogs(vm)
        try requireLeadOptIdle(vm)
        let pending = vm.leadOpt.analogs.filter { $0.status == .generated }.count
        guard pending > 0 else {
            throw ConsoleError("every analog is already docked or filtered — run 'leadopt generate' for new ones")
        }
        guard vm.docking.dockingEngine != nil, vm.docking.selectedPocket != nil else {
            throw ConsoleError("dock the reference ligand first — analog docking reuses its receptor and pocket")
        }
        if vm.docking.isDocking || vm.docking.isBatchDocking {
            throw ConsoleError("a docking run is in progress — wait for it or 'dock stop'")
        }
        let start = Date()
        guard let task = vm.dockAnalogs() else {
            throw ConsoleError("analog docking could not start — see the activity log")
        }
        await inv.withProgress({
            let progress = vm.leadOpt.dockingProgress
            guard vm.leadOpt.isDocking, progress.total > 0 else { return nil }
            var line = "Docking analogs \(progress.current)/\(progress.total)"
            if vm.docking.isDocking { line += " · \(DockingProgress(vm.docking).summary)" }
            return line
        }) {
            await task.value
        }
        let docked = vm.leadOpt.analogs.filter { $0.status == .docked }
        guard let best = docked.first, let bestScore = best.bestScore else {
            throw ConsoleError("no analog produced a pose — see the activity log")
        }
        let method = vm.leadOpt.scoringMethod
        var line = "Docked \(docked.count)/\(pending) analogs in \(ConsoleFormat.seconds(Date().timeIntervalSince(start))); "
            + "best \(best.name) \(String(format: "%.2f", bestScore)) \(method.unitLabel)"
        if let reference = vm.leadOpt.referenceScore {
            line += String(format: " (reference %.2f)", reference)
        }
        inv.emit(.output, analogTable(vm, analogs: Array(vm.leadOpt.analogs.enumerated())))
        inv.emit(.success, line)
    }

    // MARK: leadopt list / show

    /// Lists the analogs with status, score and property deltas, filtered by status.
    private static func listAnalogs(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard !vm.leadOpt.analogs.isEmpty else {
            inv.emit(.output, "No analogs — run 'leadopt reference' then 'leadopt generate'")
            return
        }
        if let top = inv.int("top"), top < 1 { throw ConsoleError("top must be at least 1") }
        var rows = Array(vm.leadOpt.analogs.enumerated())
        if let status = inv.string("status"), status != "all" {
            rows = rows.filter { AnalogStatusName.name(of: $0.element.status) == status }
        }
        guard !rows.isEmpty else {
            inv.emit(.output, "No analog has status '\(inv.string("status") ?? "")'")
            return
        }
        let shown = Array(rows.prefix(inv.int("top") ?? rows.count))
        inv.emit(.output, analogTable(vm, analogs: shown)
                 + (shown.count < rows.count ? "\n… \(rows.count - shown.count) more (top=\(rows.count) shows all)" : ""))
    }

    /// Details of one analog: SMILES, descriptors and deltas, ADMET predictions and docking score.
    private static func showAnalog(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireAnalogs(vm)
        let index = inv.requiredInt("n") - 1
        guard vm.leadOpt.analogs.indices.contains(index) else {
            throw ConsoleError("no analog #\(index + 1) — \(vm.leadOpt.analogs.count) available ('leadopt list')")
        }
        let id = vm.leadOpt.analogs[index].id
        let admet = await vm.analogADMET(at: index)
        guard let current = vm.leadOpt.analogs.firstIndex(where: { $0.id == id }) else {
            throw ConsoleError("the analog list changed while predicting ADMET — run 'leadopt list' again")
        }
        let analog = vm.leadOpt.analogs[current]
        let method = vm.leadOpt.scoringMethod
        let reference = vm.leadOpt.referenceDescriptors
        var lines = [
            "#\(current + 1) \(analog.name) — \(AnalogStatusName.name(of: analog.status))",
            "SMILES     \(analog.smiles)",
            "Reference  \(vm.leadOpt.referenceSMILES)"
        ]
        var score = analog.bestScore.map { String(format: "%.2f %@", $0, method.unitLabel) } ?? "not docked"
        if let s = analog.bestScore, let ref = vm.leadOpt.referenceScore {
            score += String(format: " (reference %.2f, Δ %+.2f)", ref, s - ref)
        }
        lines.append("Score      \(score)")
        if let d = analog.descriptors {
            let delta = { (value: Float, ref: Float?) in ref.map { String(format: " (%+.1f)", value - $0) } ?? "" }
            let deltaInt = { (value: Int, ref: Int?) in ref.map { String(format: " (%+d)", value - $0) } ?? "" }
            lines.append("")
            lines.append(ConsoleTable.render(["Descriptor", "Value"], [
                ["MW", String(format: "%.1f", d.molecularWeight) + delta(d.molecularWeight, reference?.molecularWeight)],
                ["logP", String(format: "%.2f", d.logP) + delta(d.logP, reference?.logP)],
                ["TPSA", String(format: "%.1f", d.tpsa) + delta(d.tpsa, reference?.tpsa)],
                ["HBD / HBA", "\(d.hbd) / \(d.hba)"],
                ["Rotatable bonds", "\(d.rotatableBonds)" + deltaInt(d.rotatableBonds, reference?.rotatableBonds)],
                ["Rings (aromatic)", "\(d.rings) (\(d.aromaticRings))"],
                ["Heavy atoms", "\(d.heavyAtomCount)"],
                ["Fraction sp3", String(format: "%.2f", d.fractionCSP3)],
                ["Lipinski / Veber", "\(d.lipinski ? "pass" : "fail") / \(d.veber ? "pass" : "fail")"]
            ]))
        }
        if let admet, vm.admetPredictor.isAvailable {
            let probability = { (value: Float?) in value.map { String(format: "%.2f", $0) } ?? "—" }
            lines.append("")
            lines.append(ConsoleTable.render(["ADMET", "Prediction"], [
                ["logD7.4", admet.logD74.map { String(format: "%.2f", $0) } ?? "—"],
                ["Solubility (log S)", admet.aqueousSolubility.map { String(format: "%.2f", $0) } ?? "—"],
                ["hERG liability", probability(admet.hergLiability)],
                ["CYP2D6 inhibition", probability(admet.cyp2d6Inhibition)],
                ["CYP3A4 inhibition", probability(admet.cyp3a4Inhibition)],
                ["BBB permeability", probability(admet.bbbPermeability)],
                ["Metabolic stability", probability(admet.metabolicStability)],
                ["Drug-likeness", String(format: "%.2f", admet.drugLikeness)]
            ]))
        } else {
            lines.append("ADMET      models not loaded")
        }
        inv.emit(.output, lines.joined(separator: "\n"))
        if !inv.isHeadless {
            vm.leadOpt.selectedAnalogIndex = current
            if analog.status == .docked { vm.applyAnalogPose(at: current) }
        }
    }

    // MARK: leadopt helpers

    /// Fails with a hint to run `leadopt generate` when there are no analogs yet.
    private static func requireAnalogs(_ vm: AppViewModel) throws {
        guard !vm.leadOpt.analogs.isEmpty else {
            throw ConsoleError(vm.leadOpt.hasReference ? "no analogs — run 'leadopt generate' first"
                               : "no analogs — run 'leadopt reference' then 'leadopt generate'")
        }
    }

    private static func requireLeadOptIdle(_ vm: AppViewModel) throws {
        guard !vm.leadOpt.isGenerating, !vm.leadOpt.isDocking else {
            throw ConsoleError("lead optimization is running ('leadopt stop' stops it)")
        }
    }

    /// Analog rows numbered by their position in `leadOpt.analogs` (the numbering `show` takes).
    private static func analogTable(_ vm: AppViewModel, analogs: [(offset: Int, element: LeadOptAnalog)]) -> String {
        let f = { (value: Float?) in value.map { String(format: "%+.1f", $0) } ?? "—" }
        let rows = analogs.map { i, analog -> [String] in
            [
                "\(i + 1)", analog.name, AnalogStatusName.name(of: analog.status),
                analog.bestScore.map { String(format: "%.2f", $0) } ?? "—",
                f(analog.deltaMW), f(analog.deltaLogP), f(analog.deltaTPSA),
                analog.deltaRotBonds.map { String(format: "%+d", $0) } ?? "—",
                analog.smiles.count > 40 ? String(analog.smiles.prefix(39)) + "…" : analog.smiles
            ]
        }
        return ConsoleTable.render(["#", "Name", "Status", vm.leadOpt.scoringMethod.unitLabel, "ΔMW", "ΔlogP", "ΔTPSA", "ΔRotB", "SMILES"],
                                   rows, rightAligned: [0, 3, 4, 5, 6, 7])
    }

    // MARK: - view

    /// `hide`, `show`, `center` and `view`: everything the 3D viewport displays, plus offscreen snapshots.
    static func viewCommands() -> [ConsoleCommand] {
        let renderModes = ConsoleNames.names(ConsoleNames.renderModes)
        let ligandModes = ConsoleNames.names(ConsoleNames.renderModes).filter { $0 != "ribbon" } + ["auto"]
        let fitTargets = ["all", "protein", "ligand", "pocket", "selection"]
        return [
            ConsoleCommand(
                name: "view",
                summary: "Change the 3D view: render modes, colours, surface, clipping, camera, snapshots",
                category: .view,
                forms: [
                    ConsoleForm(path: [], summary: "Set several display options at once",
                                options: [
                                    ConsoleParameter(name: "mode", type: .choice(renderModes), summary: "Render mode of the protein (and the ligand unless ligand-mode is set)"),
                                    ConsoleParameter(name: "ligand-mode", type: .choice(ligandModes), summary: "Render mode of the ligand; auto follows the protein (ball-and-stick under ribbon)"),
                                    ConsoleParameter(name: "hydrogens", type: .bool, summary: "Show hydrogen atoms"),
                                    ConsoleParameter(name: "lighting", type: .choice(["uniform", "directional"]), summary: "Uniform or directional lighting"),
                                    ConsoleParameter(name: "scheme", type: .choice(ViewNames.names(ViewNames.colorSchemes)), summary: "Colour scheme: element (CPK), ligand-focus (grey protein, gold ligand) or chain"),
                                    ConsoleParameter(name: "surface", type: .bool, summary: "Show the molecular surface"),
                                    ConsoleParameter(name: "side-chains", type: .choice(ViewNames.names(ViewNames.sideChains)), summary: "Side chains drawn in ribbon mode")
                                ]) { inv in
                        try setGeneralView(inv)
                    },
                    ConsoleForm(path: ["protein"], summary: "Protein display: visibility, render mode, colouring, side chains",
                                options: [
                                    ConsoleParameter(name: "visible", type: .bool, summary: "Show the protein"),
                                    ConsoleParameter(name: "mode", type: .choice(renderModes), summary: "Render mode (also resets ligand-mode to auto)"),
                                    ConsoleParameter(name: "color", type: .string, summary: "element, chain, or a colour (name, #rrggbb or r,g,b) for every chain"),
                                    ConsoleParameter(name: "scope", type: .choice(["carbons", "all"]), summary: "Colour carbons only or every atom (default carbons)"),
                                    ConsoleParameter(name: "side-chains", type: .choice(ViewNames.names(ViewNames.sideChains)), summary: "Side chains drawn in ribbon mode")
                                ]) { inv in
                        try setProteinView(inv)
                    },
                    ConsoleForm(path: ["ligand"], summary: "Ligand display: visibility, render mode, carbon colour",
                                options: [
                                    ConsoleParameter(name: "visible", type: .bool, summary: "Show the ligand"),
                                    ConsoleParameter(name: "mode", type: .choice(ligandModes), summary: "Render mode; auto follows the protein"),
                                    ConsoleParameter(name: "carbon-color", type: .string, summary: "Ligand carbon colour (name, #rrggbb or r,g,b), or default"),
                                    ConsoleParameter(name: "carbons-only", type: .bool, summary: "Apply carbon-color to carbons only (on) or to every atom (off)")
                                ]) { inv in
                        try setLigandView(inv)
                    },
                    ConsoleForm(path: ["chain"], summary: "One chain: visibility and colour",
                                arguments: [ConsoleParameter(name: "id", type: .string, summary: "Chain identifier, e.g. A")],
                                options: [
                                    ConsoleParameter(name: "visible", type: .bool, summary: "Show the chain"),
                                    ConsoleParameter(name: "color", type: .string, summary: "cpk, chain (palette colour) or a colour (name, #rrggbb or r,g,b)"),
                                    ConsoleParameter(name: "scope", type: .choice(["carbons", "all"]), summary: "Colour carbons only or every atom")
                                ]) { inv in
                        try setChainView(inv)
                    },
                    ConsoleForm(path: ["surface"], summary: "Molecular surface: show/hide, colouring, opacity, probe and grid",
                                arguments: [ConsoleParameter(name: "show", type: .bool, summary: "on or off (default: leave as is)", isOptional: true)],
                                options: [
                                    ConsoleParameter(name: "color", type: .choice(ViewNames.names(ViewNames.surfaceColors)), summary: "Surface colouring"),
                                    ConsoleParameter(name: "opacity", type: .float, summary: "Surface opacity (0–1, default 0.85)"),
                                    ConsoleParameter(name: "probe", type: .float, summary: "Probe radius in Å (0.5–5, default 1.4 = water)"),
                                    ConsoleParameter(name: "spacing", type: .float, summary: "Grid spacing in Å (0.2–2, default 0.5; finer is slower)")
                                ]) { inv in
                        try await setSurfaceView(inv)
                    },
                    ConsoleForm(path: ["background"], summary: "Background gradient opacity (0 = plain white or black, 1 = full gradient)",
                                arguments: [ConsoleParameter(name: "opacity", type: .float, summary: "0–1")]) { inv in
                        guard let opacity = inv.double("opacity"), (0...1).contains(opacity) else {
                            throw ConsoleError("background opacity must be between 0 and 1")
                        }
                        let vm = inv.viewModel
                        vm.workspace.backgroundOpacity = Float(opacity)
                        vm.renderer?.backgroundOpacity = Float(opacity)
                        reportViewChange(inv, "Background opacity \(String(format: "%.2f", opacity))")
                    },
                    ConsoleForm(path: ["clip"], summary: "Z-slab clipping around the view centre (or the focused pocket)",
                                arguments: [ConsoleParameter(name: "enabled", type: .bool, summary: "on or off (default: leave as is)", isOptional: true)],
                                options: [
                                    ConsoleParameter(name: "thickness", type: .float, summary: "Slab thickness in Å (2–40)"),
                                    ConsoleParameter(name: "offset", type: .float, summary: "Slab offset along the view axis in Å (-20–20)")
                                ]) { inv in
                        try setClipping(inv)
                    },
                    ConsoleForm(path: ["grid"], summary: "Docking grid box wireframe: visibility, line width, colour",
                                arguments: [ConsoleParameter(name: "show", type: .bool, summary: "on or off (default on)", isOptional: true)],
                                options: [
                                    ConsoleParameter(name: "width", type: .float, summary: "Line width in screen pixels (0.5–10)"),
                                    ConsoleParameter(name: "color", type: .string, summary: "Line colour (name, #rrggbb or r,g,b)")
                                ],
                                isGUIOnly: true) { inv in
                        try setGridBox(inv)
                    },
                    ConsoleForm(path: ["interactions"], summary: "Protein–ligand interaction lines of the shown pose",
                                arguments: [ConsoleParameter(name: "show", type: .bool, summary: "on or off")],
                                isGUIOnly: true) { inv in
                        let vm = inv.viewModel
                        if inv.bool("show") == true {
                            guard !vm.docking.currentInteractions.isEmpty else {
                                throw ConsoleError("no interactions to show — display a pose with 'pose show <n>'")
                            }
                            vm.renderer?.updateInteractionLines(vm.docking.currentInteractions)
                            inv.emit(.success, "Showing \(vm.docking.currentInteractions.count) interaction lines")
                        } else {
                            vm.renderer?.clearInteractionLines()
                            inv.emit(.success, "Interaction lines hidden")
                        }
                    },
                    ConsoleForm(path: ["fit"], summary: "Fit the camera to part of the scene",
                                arguments: [ConsoleParameter(name: "target", type: .choice(fitTargets), summary: "What to frame (default all)", isOptional: true)],
                                isGUIOnly: true) { inv in
                        try fitCamera(inv.string("target") ?? "all", in: inv.viewModel)
                    },
                    ConsoleForm(path: ["reset"], summary: "Reset the camera orientation and refit the scene", isGUIOnly: true) { inv in
                        inv.viewModel.resetView()
                    },
                    ConsoleForm(path: ["rotate"], summary: "Rotate the camera around the view centre by angles in degrees",
                                options: [
                                    ConsoleParameter(name: "x", type: .float, summary: "Degrees around the screen's horizontal axis (tilt)"),
                                    ConsoleParameter(name: "y", type: .float, summary: "Degrees around the vertical axis (turn)"),
                                    ConsoleParameter(name: "z", type: .float, summary: "Degrees around the viewing axis (roll)")
                                ],
                                isGUIOnly: true) { inv in
                        try rotateCamera(inv)
                    },
                    ConsoleForm(path: ["zoom"], summary: "Zoom the camera by a factor (2 = twice as close, 0.5 = twice as far)",
                                arguments: [ConsoleParameter(name: "factor", type: .float, summary: "Zoom factor (0.05–20)")],
                                isGUIOnly: true) { inv in
                        guard let factor = inv.double("factor"), (0.05...20).contains(factor) else {
                            throw ConsoleError("zoom factor must be between 0.05 and 20")
                        }
                        guard let renderer = inv.viewModel.renderer else { throw ConsoleError("the 3D view is not ready") }
                        renderer.camera.distance = max(0.5, renderer.camera.distance / Float(factor))
                        renderer.setNeedsRedraw()
                    },
                    ConsoleForm(path: ["snapshot"], summary: "Render the 3D view offscreen to an image (.png, .jpg or .tiff); also works headless",
                                arguments: [ConsoleParameter(name: "file", type: .path, summary: "Output image file")],
                                options: [
                                    ConsoleParameter(name: "width", type: .int, summary: "Width in pixels (default: the viewport, or 1600)"),
                                    ConsoleParameter(name: "height", type: .int, summary: "Height in pixels (default: the viewport, or 1200)"),
                                    ConsoleParameter(name: "fit", type: .choice(fitTargets), summary: "Frame this part of the scene first (headless default all)"),
                                    ConsoleParameter(name: "theme", type: .choice(["dark", "light"]), summary: "Background theme (default: the app's)")
                                ]) { inv in
                        try await snapshot(inv)
                    },
                    ConsoleForm(path: ["status"], summary: "Show the current display settings") { inv in
                        inv.emit(.output, viewStatus(inv.viewModel))
                    }
                ],
                examples: [
                    "view mode=ribbon ligand-mode=ballstick side-chains=interacting",
                    "view scheme=ligand-focus hydrogens=off",
                    "view ligand carbon-color=gold",
                    "view chain B visible=off",
                    "view surface on color=esp opacity=0.6",
                    "view clip on thickness=12",
                    "view fit pocket; view rotate y=90",
                    "view snapshot figures/pose1.png width=2400 height=1600"
                ]
            )
        ]
    }

    // MARK: view settings

    /// Headless runs keep view settings (WorkspaceState) for `view snapshot` but have nothing to show.
    private static func reportViewChange(_ inv: ConsoleInvocation, _ summary: String) {
        if inv.isHeadless {
            inv.emit(.warning, "\(summary) — no 3D view in headless mode; the setting is used by 'view snapshot'")
        } else {
            inv.emit(.success, summary)
        }
    }

    /// Applies the global display options of `view` (render mode, ligand mode, hydrogens, lighting, colour scheme, surface, side chains).
    private static func setGeneralView(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let names = ["mode", "ligand-mode", "hydrogens", "lighting", "scheme", "surface", "side-chains"]
        guard names.contains(where: inv.has) else {
            throw ConsoleError("view needs at least one option (\(names.map { $0 + "=" }.joined(separator: " "))) "
                               + "or a subcommand — see 'help view'")
        }
        var changes: [String] = []
        if let mode = ConsoleNames.lookup(inv.string("mode"), in: ConsoleNames.renderModes) {
            vm.setRenderMode(mode)
            changes.append("mode \(mode.rawValue)")
        }
        if let name = inv.string("ligand-mode") {
            applyLigandMode(name, in: vm)
            changes.append("ligand \(name)")
        }
        if let show = inv.bool("hydrogens") {
            if show != vm.workspace.showHydrogens { vm.toggleHydrogens() }
            changes.append("hydrogens \(show ? "on" : "off")")
        }
        if let lighting = inv.string("lighting") {
            if (lighting == "directional") != vm.workspace.useDirectionalLighting { vm.toggleLighting() }
            changes.append("\(lighting) lighting")
        }
        if let scheme = ViewNames.lookup(inv.string("scheme"), in: ViewNames.colorSchemes) {
            vm.workspace.colorScheme = scheme
            vm.pushToRenderer()
            changes.append("\(scheme.rawValue) colours")
        }
        if let mode = ViewNames.lookup(inv.string("side-chains"), in: ViewNames.sideChains) {
            vm.workspace.sideChainDisplay = mode
            vm.pushToRenderer()
            changes.append("side chains \(mode.rawValue.lowercased())")
        }
        if let show = inv.bool("surface") {
            if show != vm.workspace.showSurface { vm.toggleSurface() }
            changes.append("surface \(show ? "on" : "off")")
        }
        reportViewChange(inv, "View: " + changes.joined(separator: ", "))
    }

    /// Switches the ligand's render mode by its console name.
    private static func applyLigandMode(_ name: String, in vm: AppViewModel) {
        if let mode = ConsoleNames.lookup(name, in: ConsoleNames.renderModes) {
            vm.setLigandRenderMode(mode)
        } else {
            vm.workspace.ligandRenderMode = nil
            vm.pushToRenderer()
        }
    }

    /// Applies the protein display options of `view protein`.
    private static func setProteinView(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let protein = vm.molecules.protein else {
            throw ConsoleError("no protein loaded — use 'fetch <PDBID>' or 'load <file.pdb>'")
        }
        guard ["visible", "mode", "color", "scope", "side-chains"].contains(where: inv.has) else {
            throw ConsoleError("view protein needs visible=, mode=, color=, scope= or side-chains=")
        }
        let color = try inv.string("color").map { try ChainColor.parse($0, option: "color") }
        var changes: [String] = []
        if let visible = inv.bool("visible") {
            vm.workspace.showProtein = visible
            changes.append(visible ? "shown" : "hidden")
        }
        if let mode = ConsoleNames.lookup(inv.string("mode"), in: ConsoleNames.renderModes) {
            vm.workspace.renderMode = mode
            vm.workspace.ligandRenderMode = nil
            changes.append("mode \(mode.rawValue)")
        }
        let scope: WorkspaceState.ChainColorScope? = inv.string("scope").map { $0 == "all" ? .allAtoms : .carbonsOnly }
        for chain in protein.chains {
            if let color { color.apply(toChain: chain.id, in: &vm.workspace) }
            if let scope { vm.workspace.chainColorScopes[chain.id] = scope }
        }
        if let text = inv.string("color") { changes.append("colour \(text.lowercased())") }
        if let scope = inv.string("scope") { changes.append("colour scope \(scope)") }
        if let mode = ViewNames.lookup(inv.string("side-chains"), in: ViewNames.sideChains) {
            vm.workspace.sideChainDisplay = mode
            changes.append("side chains \(mode.rawValue.lowercased())")
        }
        vm.pushToRenderer()
        reportViewChange(inv, "Protein: " + changes.joined(separator: ", "))
    }

    /// Applies the ligand display options of `view ligand`.
    private static func setLigandView(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard ["visible", "mode", "carbon-color", "carbons-only"].contains(where: inv.has) else {
            throw ConsoleError("view ligand needs visible=, mode=, carbon-color= or carbons-only=")
        }
        var carbonColor: SIMD3<Float>??
        if let text = inv.string("carbon-color") {
            carbonColor = text.lowercased() == "default" ? .some(nil) : .some(try ConsoleColor.parse(text, option: "carbon-color"))
        }
        var changes: [String] = []
        if let visible = inv.bool("visible") {
            vm.workspace.showLigand = visible
            changes.append(visible ? "shown" : "hidden")
        }
        if let name = inv.string("mode") {
            applyLigandMode(name, in: vm)
            changes.append("mode \(name)")
        }
        if let carbonColor {
            vm.workspace.ligandCarbonColor = carbonColor
            changes.append("carbon colour \(inv.string("carbon-color")?.lowercased() ?? "")")
        }
        if let carbonsOnly = inv.bool("carbons-only") {
            vm.workspace.ligandColorCarbonsOnly = carbonsOnly
            changes.append(carbonsOnly ? "colour on carbons" : "colour on every atom")
        }
        vm.pushToRenderer()
        reportViewChange(inv, "Ligand: " + changes.joined(separator: ", "))
    }

    /// Shows, hides or recolours one chain (`view chain`).
    private static func setChainView(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let protein = vm.molecules.protein else {
            throw ConsoleError("no protein loaded — use 'fetch <PDBID>' or 'load <file.pdb>'")
        }
        let requested = inv.requiredString("id")
        guard let chain = protein.chains.first(where: { $0.id.caseInsensitiveCompare(requested) == .orderedSame }) else {
            throw ConsoleError("no chain '\(requested)' — chains: \(protein.chains.map(\.id).joined(separator: ", "))")
        }
        guard ["visible", "color", "scope"].contains(where: inv.has) else {
            throw ConsoleError("view chain \(chain.id) needs visible=, color= or scope=")
        }
        let color = try inv.string("color").map { try ChainColor.parse($0, option: "color") }
        var changes: [String] = []
        if let visible = inv.bool("visible") {
            if visible == vm.workspace.hiddenChainIDs.contains(chain.id) { vm.toggleChainVisibility(chain.id) }
            changes.append(visible ? "shown" : "hidden")
        }
        if let color {
            color.apply(toChain: chain.id, in: &vm.workspace)
            changes.append("colour \(inv.string("color")?.lowercased() ?? "")")
        }
        if let scope = inv.string("scope") {
            vm.workspace.chainColorScopes[chain.id] = scope == "all" ? .allAtoms : .carbonsOnly
            changes.append("colour scope \(scope)")
        }
        vm.pushToRenderer()
        reportViewChange(inv, "Chain \(chain.id): " + changes.joined(separator: ", "))
    }

    /// Shows or hides the molecular surface and sets its colouring, opacity, probe and spacing, awaiting the surface build.
    private static func setSurfaceView(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        guard ["show", "color", "opacity", "probe", "spacing"].contains(where: inv.has) else {
            throw ConsoleError("view surface needs on|off, color=, opacity=, probe= or spacing=")
        }
        if let opacity = inv.double("opacity"), !(0...1).contains(opacity) { throw ConsoleError("opacity must be between 0 and 1") }
        if let probe = inv.double("probe"), !(0.5...5).contains(probe) { throw ConsoleError("probe must be between 0.5 and 5 Å") }
        if let spacing = inv.double("spacing"), !(0.2...2).contains(spacing) { throw ConsoleError("spacing must be between 0.2 and 2 Å") }
        if inv.bool("show") == true || (inv.bool("show") == nil && vm.workspace.showSurface) {
            guard vm.molecules.protein != nil else { throw ConsoleError("no protein loaded — the surface is computed on the protein") }
        }
        var changes: [String] = []
        var regenerate = false
        if let opacity = inv.double("opacity") {
            vm.workspace.surfaceOpacity = Float(opacity)
            vm.renderer?.surfaceOpacity = Float(opacity)
            changes.append(String(format: "opacity %.2f", opacity))
        }
        if let probe = inv.double("probe") {
            vm.workspace.surfaceProbeRadius = Float(probe)
            regenerate = true
            changes.append(String(format: "probe %.1f Å", probe))
        }
        if let spacing = inv.double("spacing") {
            vm.workspace.surfaceGridSpacing = Float(spacing)
            regenerate = true
            changes.append(String(format: "grid %.2f Å", spacing))
        }
        if let mode = ViewNames.lookup(inv.string("color"), in: ViewNames.surfaceColors) {
            vm.workspace.surfaceColorMode = mode
            regenerate = true
            changes.append("\(mode.rawValue.lowercased()) colouring")
        }
        if let show = inv.bool("show"), show != vm.workspace.showSurface {
            vm.toggleSurface()
            regenerate = false
            changes.insert(show ? "shown" : "hidden", at: 0)
        } else if regenerate {
            vm.regenerateSurface()
        }
        if !inv.isHeadless, vm.workspace.isGeneratingSurface {
            await ConsoleObservation.wait { !vm.workspace.isGeneratingSurface }
            try Task.checkCancellation()
        }
        reportViewChange(inv, "Surface: " + changes.joined(separator: ", "))
    }

    /// Turns the Z-slab clipping on or off and sets its thickness and offset.
    private static func setClipping(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard ["enabled", "thickness", "offset"].contains(where: inv.has) else {
            throw ConsoleError("view clip needs on|off, thickness= or offset=")
        }
        if let thickness = inv.double("thickness"), !(2...40).contains(thickness) { throw ConsoleError("thickness must be between 2 and 40 Å") }
        if let offset = inv.double("offset"), !(-20...20).contains(offset) { throw ConsoleError("offset must be between -20 and 20 Å") }
        if let enabled = inv.bool("enabled") { vm.workspace.enableClipping = enabled }
        if let thickness = inv.double("thickness") { vm.workspace.slabThickness = Float(thickness) }
        if let offset = inv.double("offset") { vm.workspace.slabOffset = Float(offset) }
        // Without clipping, release any stale slab anchor so a later slab follows the camera
        // (same rule as the canvas clipping toggle).
        if !vm.workspace.enableClipping { vm.renderer?.slabCenter = nil }
        vm.pushToRenderer()
        let state = vm.workspace.enableClipping ? "on" : "off"
        reportViewChange(inv, String(format: "Clipping %@: slab %.0f Å, offset %.0f Å", state,
                                     vm.workspace.slabThickness, vm.workspace.slabOffset))
    }

    /// Shows or hides the docking grid box and sets its line width and colour.
    private static func setGridBox(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let renderer = vm.renderer else { throw ConsoleError("the 3D view is not ready") }
        if let width = inv.double("width") {
            guard (0.5...10).contains(width) else { throw ConsoleError("width must be between 0.5 and 10 pixels") }
            vm.workspace.gridLineWidth = Float(width)
            renderer.gridLineWidth = Float(width)
        }
        if let text = inv.string("color") {
            let rgb = try ConsoleColor.parse(text, option: "color")
            vm.workspace.gridColor = SIMD4<Float>(rgb.x, rgb.y, rgb.z, vm.workspace.gridColor.w)
        }
        guard inv.bool("show") ?? true else {
            renderer.clearGridBox()
            inv.emit(.success, "Grid box hidden")
            return
        }
        if let pocket = vm.docking.selectedPocket {
            vm.showGridBoxForPocket(pocket)
        } else if vm.docking.gridInitialized {
            renderer.updateGridBox(center: vm.docking.gridCenter, halfSize: vm.docking.gridHalfSize, color: vm.workspace.gridColor)
        } else {
            throw ConsoleError("no docking box — use 'pocket detect' or 'pocket box center=x,y,z'")
        }
        inv.emit(.success, "Grid box shown")
    }

    // MARK: view camera

    /// Frames the camera on the whole scene, the protein, the ligand, the pocket or the selection.
    private static func fitCamera(_ target: String, in vm: AppViewModel) throws {
        guard let renderer = vm.renderer else { throw ConsoleError("the 3D view is not ready") }
        switch target {
        case "protein":
            guard vm.molecules.protein != nil else { throw ConsoleError("no protein loaded") }
            vm.fitToProtein()
        case "ligand":
            guard vm.molecules.ligand != nil else { throw ConsoleError("no ligand loaded") }
            vm.fitToLigand()
        case "pocket":
            guard let pocket = vm.docking.selectedPocket else {
                throw ConsoleError("no pocket selected — use 'pocket detect' or 'pocket select <n>'")
            }
            vm.focusOnPocket(pocket)
        case "selection":
            guard !vm.workspace.selectedAtomIndices.isEmpty else { throw ConsoleError("nothing is selected") }
            vm.fitToView()
        default:
            renderer.fitToContent()
        }
    }

    /// Rotates the camera by the given angles about the view axes.
    private static func rotateCamera(_ inv: ConsoleInvocation) throws {
        guard ["x", "y", "z"].contains(where: inv.has) else { throw ConsoleError("view rotate needs x=, y= or z= (degrees)") }
        guard let renderer = inv.viewModel.renderer else { throw ConsoleError("the 3D view is not ready") }
        let camera = renderer.camera
        let radians = { (name: String) in Float(inv.double(name) ?? 0) * .pi / 180 }
        // Same axes as mouse orbiting: turn around global Y, tilt around the camera's right
        // axis, roll around the viewing axis.
        let turn = Quat.fromAxisAngle(SIMD3<Float>(0, 1, 0), angle: radians("y"))
        let tilt = Quat.fromAxisAngle(camera.rightVector, angle: radians("x"))
        let roll = Quat.fromAxisAngle(camera.orientation.rotate(SIMD3<Float>(0, 0, 1)), angle: radians("z"))
        camera.orientation = (roll * turn * tilt * camera.orientation).normalized
        renderer.setNeedsRedraw()
    }

    // MARK: view snapshot

    /// Renders the scene offscreen to an image file, in the GUI or headless.
    private static func snapshot(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let url = inv.requiredURL("file")
        let types: [String: UTType] = ["png": .png, "jpg": .jpeg, "jpeg": .jpeg, "tif": .tiff, "tiff": .tiff]
        guard let type = types[url.pathExtension.lowercased()] else {
            throw ConsoleError("unsupported image type '.\(url.pathExtension)' (expected png, jpg or tiff)")
        }
        for name in ["width", "height"] {
            if let value = inv.int(name), !(16...Renderer.maxSnapshotSide).contains(value) {
                throw ConsoleError("\(name) must be between 16 and \(Renderer.maxSnapshotSide) pixels")
            }
        }
        guard vm.molecules.protein != nil || vm.molecules.ligand != nil else {
            throw ConsoleError("nothing to render — load a protein or a ligand first")
        }

        let renderer: Renderer
        let windowless: Bool
        if let existing = vm.renderer {
            renderer = existing
            windowless = false
        } else {
            guard let created = Renderer.makeWindowless() else {
                throw ConsoleError("cannot create a Metal renderer without a window on this machine")
            }
            renderer = created
            windowless = true
            vm.renderer = created
        }
        defer { if windowless { vm.renderer = nil } }

        if windowless {
            // A fresh renderer only knows what pushToRenderer and the scene overlays give it.
            renderer.backgroundOpacity = vm.workspace.backgroundOpacity
            renderer.surfaceOpacity = vm.workspace.surfaceOpacity
            renderer.lightingMode = vm.workspace.useDirectionalLighting ? 1 : 0
            vm.pushToRenderer()
            if let pocket = vm.docking.selectedPocket { vm.showGridBoxForPocket(pocket) }
            if !vm.docking.currentInteractions.isEmpty { renderer.updateInteractionLines(vm.docking.currentInteractions) }
            if vm.workspace.showSurface, vm.molecules.protein != nil {
                vm.generateSurface()
                await ConsoleObservation.wait { !vm.workspace.isGeneratingSurface }
                try Task.checkCancellation()
            }
        }
        if let fit = inv.string("fit") ?? (windowless ? "all" : nil) {
            try fitCamera(fit, in: vm)
        }
        let savedTheme = renderer.themeMode
        if let theme = inv.string("theme") { renderer.themeMode = theme == "light" ? 1 : 0 }
        defer { renderer.themeMode = savedTheme }

        let drawable = renderer.mtkView?.drawableSize ?? .zero
        let width = inv.int("width") ?? (drawable.width > 0 ? Int(drawable.width) : 1600)
        let height = inv.int("height") ?? (drawable.height > 0 ? Int(drawable.height) : 1200)
        guard let image = renderer.snapshot(width: width, height: height) else {
            throw ConsoleError("the snapshot could not be rendered (\(width)×\(height))")
        }
        try writeFile(url) {
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
                throw ConsoleError("cannot create \(url.lastPathComponent)")
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw ConsoleError("cannot encode \(url.lastPathComponent)")
            }
        }
        inv.emit(.success, "Wrote \(width)×\(height) snapshot to \(url.path)")
    }

    // MARK: view status

    /// Summary of the current display settings for `view status`.
    private static func viewStatus(_ vm: AppViewModel) -> String {
        let w = vm.workspace
        let onOff = { (on: Bool) in on ? "on" : "off" }
        let chainColors: String = {
            guard let protein = vm.molecules.protein, !protein.chains.isEmpty else { return "—" }
            return protein.chains.map { chain -> String in
                let mode: String
                switch w.chainColorModes[chain.id] ?? (w.colorScheme == .chainColored ? .chainDefault : .cpk) {
                case .cpk: mode = "cpk"
                case .chainDefault: mode = "chain"
                case .custom: mode = w.chainColorOverrides[chain.id].map(ConsoleColor.describe) ?? "chain"
                }
                return "\(chain.id)=\(mode)\(w.hiddenChainIDs.contains(chain.id) ? " (hidden)" : "")"
            }.joined(separator: ", ")
        }()
        let rows: [[String]] = [
            ["mode", ConsoleNames.renderModes.first { $0.1 == w.renderMode }?.0 ?? w.renderMode.rawValue],
            ["ligand-mode", w.ligandRenderMode.flatMap { mode in ConsoleNames.renderModes.first { $0.1 == mode }?.0 } ?? "auto"],
            ["protein", onOff(w.showProtein)],
            ["ligand", onOff(w.showLigand)],
            ["hydrogens", onOff(w.showHydrogens)],
            ["lighting", w.useDirectionalLighting ? "directional" : "uniform"],
            ["scheme", ViewNames.name(of: w.colorScheme, in: ViewNames.colorSchemes)],
            ["chains", chainColors],
            ["ligand carbon-color", w.ligandCarbonColor.map(ConsoleColor.describe) ?? "default"],
            ["side-chains", ViewNames.name(of: w.sideChainDisplay, in: ViewNames.sideChains)],
            ["surface", onOff(w.showSurface) + ", " + ViewNames.name(of: w.surfaceColorMode, in: ViewNames.surfaceColors)
                + String(format: ", opacity %.2f, probe %.1f Å, grid %.2f Å", w.surfaceOpacity, w.surfaceProbeRadius, w.surfaceGridSpacing)],
            ["background", String(format: "opacity %.2f", w.backgroundOpacity)],
            ["clip", onOff(w.enableClipping) + String(format: ", slab %.0f Å, offset %.0f Å", w.slabThickness, w.slabOffset)],
            ["grid line", String(format: "%.1f px, %@", w.gridLineWidth, ConsoleColor.describe(SIMD3(w.gridColor.x, w.gridColor.y, w.gridColor.z)))]
        ]
        return ConsoleTable.render(["Setting", "Value"], rows)
    }
}

// MARK: - Console spellings of view and lead-opt enums

/// Console names of the analog statuses (the short words shown in `leadopt list`).
private enum AnalogStatusName {
    static let all: [(name: String, status: LeadOptAnalog.Status)] = [
        ("new", .generated), ("3d", .prepared), ("docked", .docked), ("filtered", .filtered), ("failed", .failed)
    ]

    static func name(of status: LeadOptAnalog.Status) -> String {
        all.first { $0.status == status }?.name ?? "new"
    }
}

/// Console option values of the view enums, as lower-case words.
private enum ViewNames {
    static let colorSchemes: [(String, WorkspaceState.MoleculeColorScheme)] = [
        ("element", .element), ("ligand-focus", .ligandFocus), ("chain", .chainColored)
    ]
    static let sideChains: [(String, SideChainDisplay)] = [
        ("none", .none), ("interacting", .interacting), ("selected", .selected), ("all", .all)
    ]
    static let surfaceColors: [(String, SurfaceColorMode)] = [
        ("uniform", .uniform), ("esp", .esp), ("hydrophobicity", .hydrophobicity), ("pharmacophore", .pharmacophore)
    ]

    static func names<T>(_ table: [(String, T)]) -> [String] { table.map(\.0) }

    static func lookup<T>(_ name: String?, in table: [(String, T)]) -> T? {
        guard let name else { return nil }
        return table.first { $0.0 == name }?.1
    }

    static func name<T: Equatable>(of value: T, in table: [(String, T)]) -> String {
        table.first { $0.1 == value }?.0 ?? "\(value)"
    }
}

/// Colours typed on the console: a name, `#rrggbb`, or `r,g,b` (0–1, or 0–255 when any
/// component is above 1).
private enum ConsoleColor {
    static let named: [(String, SIMD3<Float>)] = [
        ("white", SIMD3(1, 1, 1)), ("black", SIMD3(0, 0, 0)), ("gray", SIMD3(0.55, 0.55, 0.55)),
        ("lightgray", SIMD3(0.75, 0.75, 0.75)), ("red", SIMD3(0.90, 0.20, 0.20)), ("green", SIMD3(0.25, 0.75, 0.35)),
        ("blue", SIMD3(0.25, 0.45, 0.90)), ("cyan", SIMD3(0.20, 0.80, 0.90)), ("magenta", SIMD3(0.85, 0.30, 0.75)),
        ("yellow", SIMD3(0.95, 0.85, 0.20)), ("gold", SIMD3(0.85, 0.78, 0.20)), ("orange", SIMD3(0.95, 0.55, 0.15)),
        ("pink", SIMD3(0.95, 0.55, 0.70)), ("purple", SIMD3(0.60, 0.35, 0.80)), ("teal", SIMD3(0.20, 0.65, 0.65)),
        ("salmon", SIMD3(0.98, 0.50, 0.45)), ("lime", SIMD3(0.55, 0.85, 0.25)), ("brown", SIMD3(0.55, 0.35, 0.20))
    ]

    /// Parses a colour given by name or as `#rrggbb`; `option` names the option in error messages.
    static func parse(_ text: String, option: String) throws -> SIMD3<Float> {
        let lowered = text.lowercased()
        if let color = named.first(where: { $0.0 == lowered })?.1 { return color }
        if lowered.hasPrefix("#"), lowered.count == 7, let value = UInt32(lowered.dropFirst(), radix: 16) {
            return SIMD3(Float((value >> 16) & 0xFF), Float((value >> 8) & 0xFF), Float(value & 0xFF)) / 255
        }
        let parts = lowered.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        if parts.count == 3, parts.allSatisfy({ $0 >= 0 && $0 <= 255 }) {
            let rgb = SIMD3(parts[0], parts[1], parts[2])
            return rgb.max() > 1 ? rgb / 255 : rgb
        }
        var message = "\(option) expects a colour name, #rrggbb or r,g,b, got '\(text)'"
        if let hint = ConsoleSuggest.closest(lowered, in: named.map(\.0)) { message += " — did you mean '\(hint)'?" }
        throw ConsoleError(message)
    }

    /// The colour's name when it is one of `named`, else `#rrggbb`.
    static func describe(_ color: SIMD3<Float>) -> String {
        if let name = named.first(where: { simd_distance($0.1, color) < 0.002 })?.0 { return name }
        let byte = { (value: Float) in Int((min(max(value, 0), 1) * 255).rounded()) }
        return String(format: "#%02x%02x%02x", byte(color.x), byte(color.y), byte(color.z))
    }
}

/// A chain colouring request: element colours, the chain palette, or one custom colour.
private enum ChainColor {
    case cpk
    case palette
    case custom(SIMD3<Float>)

    /// `element`/`cpk` and `chain`/`palette` select the built-in modes; anything else is a colour.
    static func parse(_ text: String, option: String) throws -> ChainColor {
        switch text.lowercased() {
        case "element", "cpk": return .cpk
        case "chain", "palette": return .palette
        default: return .custom(try ConsoleColor.parse(text, option: option))
        }
    }

    /// Stores this colouring for one chain in the workspace the renderer reads.
    func apply(toChain id: String, in workspace: inout WorkspaceState) {
        switch self {
        case .cpk:
            workspace.chainColorModes[id] = .cpk
        case .palette:
            workspace.chainColorModes[id] = .chainDefault
        case .custom(let rgb):
            workspace.chainColorModes[id] = .custom
            workspace.chainColorOverrides[id] = rgb
        }
    }
}

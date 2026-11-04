// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - prep
//
// `prep` alone runs the Preparation step's one-click pipeline; every stage of that pipeline
// is also a subcommand of its own (clean, repair, pack, protonate, hydrogens, hbonds, charges,
// minimize) so a script can prepare a receptor exactly the way it wants. `prep missing`,
// `prep states` and `prep report` only report.

extension DruseInterpreter {

    /// Registry entry: the protein preparation commands.
    static func preparationCommands() -> [ConsoleCommand] {
        [prepCommand()]
    }

    // MARK: Options

    private static let phOption = ConsoleParameter(name: "ph", type: .float, summary: "Protonation pH, 0–14 (persists; 7.4 at start)")
    private static let chargeMethodOption = ConsoleParameter(
        name: "method", type: .choice(ConsoleNames.names(ConsoleNames.chargeMethods)),
        summary: "Partial-charge method (persists): gasteiger = RDKit on the PDB text, eem/qeq = electronegativity equalisation, xtb = GFN2-xTB")
    private static let watersOption = ConsoleParameter(
        name: "waters", type: .choice(["keep", "remove", "pocket"]),
        summary: "keep all waters, remove all, or keep only waters near the selected pocket (pocket)")
    private static let radiusOption = ConsoleParameter(
        name: "radius", type: .float, summary: "With waters=pocket: retention radius around the pocket centre in Å (persists; default 5)")
    private static let nonstandardOption = ConsoleParameter(
        name: "nonstandard", type: .choice(["remove", "keep"]),
        summary: "Remove ligands, cofactors and modified residues (structural metal ions and caps are always kept)")
    private static let capsOption = ConsoleParameter(
        name: "caps", type: .bool, summary: "Cap uncapped chain-break termini with ACE/NME")

    /// `prep`: the whole Preparation step, as one pipeline or stage by stage.
    private static func prepCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "prep",
            summary: "Prepare the protein: cleanup, repair, side-chain packing, protonation, H-bond network, charges, minimisation",
            category: .preparation,
            forms: [
                ConsoleForm(path: [], summary: "Full pipeline of the Preparation step: short loops, altlocs, non-standard residues, caps, missing atoms + FASPR repack, restrained minimisation, protonation at pH, hydrogens, H-bond network, charges",
                            options: pipelineOptions()) { inv in
                    try await runPipeline(inv)
                },
                ConsoleForm(path: ["all"], summary: "Same as 'prep'", options: pipelineOptions()) { inv in
                    try await runPipeline(inv)
                },
                ConsoleForm(path: ["clean"], summary: "Cleanup only: preferred alternate locations, non-standard residues, waters, chain-break caps",
                            options: [
                                ConsoleParameter(name: "altlocs", type: .bool, summary: "Keep only the preferred alternate location of each atom (default on)"),
                                nonstandardOption, watersOption, radiusOption,
                                ConsoleParameter(name: "caps", type: .bool, summary: "Cap uncapped chain-break termini with ACE/NME (default off; needs altlocs=on)")
                            ]) { inv in
                    try await cleanProtein(inv)
                },
                ConsoleForm(path: ["altlocs"], summary: "Keep only the preferred alternate location (highest occupancy, then lowest B)") { inv in
                    try await runPrep(inv) { $0.removeAltConfs() }
                },
                ConsoleForm(path: ["nonstandard"], summary: "Remove ligands, cofactors and modified residues (keeps waters, caps and structural metal ions)") { inv in
                    try await runPrep(inv) { $0.removeNonStandardResidues() }
                },
                ConsoleForm(path: ["waters"], summary: "Remove all waters, or keep only those near the selected pocket",
                            arguments: [ConsoleParameter(name: "mode", type: .choice(["remove", "pocket"]),
                                                         summary: "remove (default) or pocket", isOptional: true)],
                            options: [radiusOption]) { inv in
                    try await handleWaters(inv, mode: inv.string("mode") ?? "remove")
                },
                ConsoleForm(path: ["caps"], summary: "Cap uncapped chain-break termini with ACE/NME") { inv in
                    try await runPrep(inv) { $0.capChainBreaks() }
                },
                ConsoleForm(path: ["missing"], summary: "Report missing residues (gaps), chain breaks and residues with missing or extra atoms",
                            options: [ConsoleParameter(name: "all", type: .bool, summary: "List every incomplete residue instead of the first 20 (default off)")]) { inv in
                    try missingReport(inv)
                },
                ConsoleForm(path: ["repair"], summary: "Build short missing loops, rebuild missing heavy atoms, re-protonate if hydrogens were present") { inv in
                    try await runPrep(inv) { await $0.detectAndFixMissingResidues()?.value }
                },
                ConsoleForm(path: ["repair", "all"], summary: "Same as 'prep repair'") { inv in
                    try await runPrep(inv) { await $0.detectAndFixMissingResidues()?.value }
                },
                ConsoleForm(path: ["repair", "atoms"], summary: "Rebuild missing heavy atoms of existing residues from templates") { inv in
                    try await runPrep(inv) { $0.repairMissingAtoms() }
                },
                ConsoleForm(path: ["repair", "loops"], summary: "Model missing loops (SEQRES identities when available, else polyalanine) and relax them",
                            options: [
                                ConsoleParameter(name: "max-gap", type: .int, summary: "Longest gap to build, in residues (default 15)"),
                                ConsoleParameter(name: "refine", type: .choice(["gpu", "openmm", "none"]),
                                                 summary: "Loop relaxation: gpu = restrained Metal minimisation (default), openmm, none = raw builder geometry")
                            ]) { inv in
                    try await buildLoops(inv)
                },
                ConsoleForm(path: ["pack"], summary: "Repack side chains with FASPR (heavy-atom model; remove hydrogens first)",
                            options: [ConsoleParameter(name: "residues", type: .choice(["incomplete", "all", "selection"]),
                                                       summary: "incomplete = rebuild + repack residues missing side-chain atoms (default), all, or the selected residues")]) { inv in
                    try await packSidechains(inv)
                },
                ConsoleForm(path: ["protonate"], summary: "Predict pKa-shifted protonation states and add polar hydrogens at pH (with H-bond network optimisation)",
                            options: [phOption]) { inv in
                    try await runPrep(inv) { await $0.assignProtonation()?.value }
                },
                ConsoleForm(path: ["states"], summary: "Protonation states chosen by the last protonation (pKa shifts, His tautomers, charges)",
                            options: [ConsoleParameter(name: "type", type: .choice(["all", "charged", "his", "shifted"]),
                                                       summary: "all titratable residues (default), charged ones, histidines, or pKa shifted by more than 1")]) { inv in
                    try protonationStates(inv)
                },
                ConsoleForm(path: ["hydrogens"], summary: "Same as 'prep hydrogens add'", options: [phOption]) { inv in
                    try await runPrep(inv) { await $0.addHydrogens()?.value }
                },
                ConsoleForm(path: ["hydrogens", "add"], summary: "Add all hydrogens (polar + non-polar) from residue templates at pH", options: [phOption]) { inv in
                    try await runPrep(inv) { await $0.addHydrogens()?.value }
                },
                ConsoleForm(path: ["hydrogens", "remove"], summary: "Remove every hydrogen from the protein (and the ligand)") { inv in
                    try await runPrep(inv) { $0.removeHydrogens() }
                },
                ConsoleForm(path: ["hbonds"], summary: "Re-optimise the H-bond network: OH/SH/NH3 rotors, Asn/Gln flips, His tautomers and flips") { inv in
                    try await hydrogenBondNetwork(inv)
                },
                ConsoleForm(path: ["charges"], summary: "Assign partial charges", options: [chargeMethodOption]) { inv in
                    try await assignCharges(inv)
                },
                ConsoleForm(path: ["minimize"], summary: "Restrained GPU minimisation (backbone strongly restrained, side chains moderately, H free)",
                            options: [ConsoleParameter(name: "steps", type: .int, summary: "Maximum L-BFGS iterations (default 100)")]) { inv in
                    try await minimize(inv)
                },
                ConsoleForm(path: ["report"], summary: "What preparation did to the current protein, step by step, and its current state") { inv in
                    try preparationReport(inv)
                }
            ],
            examples: [
                "prep",
                "prep ph=6.5 charges=eem",
                "prep waters=pocket radius=4 hydrogens=polar",
                "prep loops=off caps=off charges=none",
                "prep clean waters=remove caps=on",
                "prep missing",
                "prep repair loops refine=openmm",
                "prep repair atoms; prep pack",
                "prep protonate ph=7.0; prep states type=his",
                "prep hydrogens remove; prep pack residues=all; prep hydrogens add",
                "prep hbonds",
                "prep charges method=qeq",
                "prep minimize steps=200",
                "prep report"
            ]
        )
    }

    /// Stage switches of `prep` / `prep all`, mapped onto `AppViewModel.ProteinPreparationOptions`.
    private static func pipelineOptions() -> [ConsoleParameter] {
        [
            phOption,
            ConsoleParameter(name: "charges", type: .choice(ConsoleNames.names(ConsoleNames.chargeMethods) + ["none"]),
                             summary: "Partial-charge method (persists), or none to skip charges"),
            watersOption, radiusOption, nonstandardOption,
            ConsoleParameter(name: "caps", type: .bool, summary: "Cap uncapped chain-break termini with ACE/NME (default on)"),
            ConsoleParameter(name: "loops", type: .bool, summary: "Model missing loops of up to 15 residues before capping (default on)"),
            ConsoleParameter(name: "hydrogens", type: .choice(["all", "polar", "none"]),
                             summary: "all (default), polar only, or none = heavy-atom receptor (no protonation)")
        ]
    }

    // MARK: Shared checks

    /// Throws the standard "no protein loaded" console error.
    static func requireProtein(_ vm: AppViewModel) throws {
        guard vm.molecules.protein != nil else {
            throw ConsoleError("no protein loaded — use 'fetch <PDBID>' or 'load <file.pdb>'")
        }
    }

    /// Validates and applies `ph=`, `method=` and `radius=`, which persist like the matching
    /// Preparation-step controls.
    private static func applyPrepOptions(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        if let ph = inv.double("ph") {
            guard (0...14).contains(ph) else { throw ConsoleError("ph must be between 0 and 14") }
            vm.molecules.protonationPH = (Float(ph) * 10).rounded() / 10
        }
        if let method = ConsoleNames.lookup(inv.string("method"), in: ConsoleNames.chargeMethods) {
            vm.docking.chargeMethod = method
        }
        if let radius = inv.double("radius") {
            guard radius > 0 else { throw ConsoleError("radius must be a positive number of Å") }
            vm.molecules.pocketWaterRadius = Float(radius)
        }
    }

    /// Water retention near the pocket needs a pocket.
    private static func requirePocketForWaters(_ vm: AppViewModel) throws {
        guard vm.docking.selectedPocket != nil else {
            throw ConsoleError("keeping pocket waters needs a docking pocket — run 'pocket detect' (or 'selection pocket') first")
        }
    }

    /// Runs one preparation step and waits for it, then prints what the step recorded (or the
    /// reason nothing changed) and the protein's new size.
    private static func runPrep(
        _ inv: ConsoleInvocation,
        step: @MainActor (AppViewModel) async throws -> Void
    ) async throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        guard !vm.molecules.isMinimizing else {
            throw ConsoleError("a preparation step is already running")
        }
        try applyPrepOptions(inv)
        let recorded = vm.preparationLog.count
        let before = vm.molecules.protein?.atomCount ?? 0
        // Preparation steps announce their running phase as a "…..." status message.
        try await inv.withProgress({ vm.workspace.statusMessage.hasSuffix("...") ? vm.workspace.statusMessage : nil }) {
            try await step(vm)
        }
        guard let protein = vm.molecules.protein else {
            throw ConsoleError("the protein was lost during preparation — see the activity log")
        }
        let entries = vm.preparationLog.dropFirst(recorded)
        if entries.isEmpty {
            inv.emit(.output, "No change: \(vm.workspace.statusMessage)")
            return
        }
        for entry in entries {
            inv.emit(.success, entry.detail)
        }
        let hydrogens = protein.atoms.filter { $0.element == .H }.count
        let delta = protein.atomCount - before
        let change = delta == 0 ? "no atom change" : (delta > 0 ? "+\(delta)" : "\(delta)") + " atoms"
        inv.emit(.output, "\(protein.name): \(protein.atomCount) atoms (\(hydrogens) H, \(change)), "
                 + "pH \(String(format: "%.1f", vm.molecules.protonationPH))")
    }

    // MARK: Pipeline and cleanup

    /// `prep`: translates the options into pipeline stage switches and runs the pipeline.
    private static func runPipeline(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        var options = AppViewModel.ProteinPreparationOptions()
        if let charges = inv.string("charges") {
            if charges == "none" {
                options.assignCharges = false
            } else if let method = ConsoleNames.lookup(charges, in: ConsoleNames.chargeMethods) {
                vm.docking.chargeMethod = method
            }
        }
        try applyPrepOptions(inv)
        switch inv.string("waters") {
        case "remove": options.waters = .remove
        case "pocket":
            try requirePocketForWaters(vm)
            options.waters = .pocket(radius: vm.molecules.pocketWaterRadius)
        default: break
        }
        if let nonstandard = inv.string("nonstandard") { options.removeNonStandard = nonstandard == "remove" }
        if let caps = inv.bool("caps") { options.capChainBreaks = caps }
        if let loops = inv.bool("loops") { options.buildLoops = loops }
        if let hydrogens = inv.string("hydrogens").flatMap(AppViewModel.HydrogenMode.init(rawValue:)) {
            options.hydrogens = hydrogens
        }
        try await runPrep(inv) { await $0.runPreparationPipeline(options)?.value }
        if !options.assignCharges {
            inv.emit(.warning, "No partial charges were assigned — run 'prep charges' before docking with electrostatic scoring")
        }
    }

    /// `prep clean`: runs the Preparation step's cleanup actions in pipeline order, each as configured.
    private static func cleanProtein(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let altlocs = inv.bool("altlocs") ?? true
        let caps = inv.bool("caps") ?? false
        if caps && !altlocs {
            throw ConsoleError("caps=on also resolves alternate locations — drop altlocs=off or run 'prep caps' later")
        }
        let waters = inv.string("waters") ?? "keep"
        if waters == "pocket" { try requirePocketForWaters(vm) }
        try await runPrep(inv) { vm in
            if altlocs { vm.removeAltConfs() }
            if (inv.string("nonstandard") ?? "remove") == "remove" { vm.removeNonStandardResidues() }
            switch waters {
            case "remove": vm.removeWaters()
            case "pocket": vm.keepPocketWaters()
            default: break
            }
            if caps { vm.capChainBreaks() }
        }
    }

    /// `prep waters`: removes all waters or keeps those near the pocket.
    private static func handleWaters(_ inv: ConsoleInvocation, mode: String) async throws {
        let vm = inv.viewModel
        if mode == "pocket" {
            try requirePocketForWaters(vm)
        } else if inv.has("radius") {
            throw ConsoleError("radius= only applies to 'prep waters pocket'")
        }
        try await runPrep(inv) { vm in
            if mode == "pocket" { vm.keepPocketWaters() } else { vm.removeWaters() }
        }
    }

    // MARK: Repair and packing

    /// `prep repair loops`: models gaps up to `max-gap=` residues and relaxes them with `refine=`.
    private static func buildLoops(_ inv: ConsoleInvocation) async throws {
        let maxGap = inv.int("max-gap") ?? AppViewModel.maxBuildableLoopLength
        guard maxGap >= 1 else { throw ConsoleError("max-gap must be at least 1") }
        let refinement = inv.string("refine").flatMap(AppViewModel.LoopRefinement.init(rawValue:)) ?? .gpu
        if refinement == .openmm, !druse_openmm_available() {
            throw ConsoleError("this build has no OpenMM — use refine=gpu")
        }
        var outcome: AppViewModel.LoopBuildOutcome?
        try await runPrep(inv) { vm in
            outcome = await vm.buildMissingLoops(maxLength: maxGap, refinement: refinement)?.value
        }
        if let outcome, outcome.gapsTooLong > 0 {
            inv.emit(.warning, "\(outcome.gapsTooLong) gap(s) longer than \(maxGap) residues were left as chain breaks — raise max-gap= or use a predicted structure")
        }
        if let outcome, outcome.gapsBuilt > 0, outcome.refinement != refinement {
            inv.emit(.warning, "\(refinement.rawValue) refinement was unavailable — the loops keep the raw builder geometry")
        }
    }

    /// `prep pack`: FASPR repacking of incomplete, all or selected side chains.
    private static func packSidechains(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        guard !protein.atoms.contains(where: { $0.element == .H }) else {
            throw ConsoleError("side-chain packing works on heavy-atom models — run 'prep hydrogens remove' first, then 'prep hydrogens add' afterwards")
        }
        let scope: AppViewModel.SidechainPackingScope
        switch inv.string("residues") ?? "incomplete" {
        case "all":
            scope = .all
        case "selection":
            let residues = vm.workspace.selectedResidueIndices.filter { $0 < protein.residues.count }
            guard !residues.isEmpty else { throw ConsoleError("no residues selected") }
            scope = .residues(Set(residues.map { index -> String in
                let residue = protein.residues[index]
                let insertion = residue.atomIndices.first.map { protein.atoms[$0].insertionCode } ?? ""
                return FASPRSidechainPacker.residueKey(chainID: residue.chainID, residueSeq: residue.sequenceNumber, insertionCode: insertion)
            }))
        default:
            scope = .incomplete
        }
        try await runPrep(inv) { vm in
            guard let task = vm.packSidechains(scope) else {
                throw ConsoleError("side-chain packing is unavailable (the Dunbrack rotamer library did not load)")
            }
            _ = await task.value
        }
    }

    // MARK: Protonation, H-bonds, charges, minimisation

    /// `prep hbonds`: re-optimises the network and lists the flipped groups.
    private static func hydrogenBondNetwork(_ inv: ConsoleInvocation) async throws {
        let protein = try loadedProtein(inv.viewModel)
        guard protein.atoms.contains(where: { $0.element == .H }) else {
            throw ConsoleError("the protein has no hydrogens — run 'prep protonate' or 'prep hydrogens' first")
        }
        var report: HBondNetworkOptimizer.NetworkReport?
        try await runPrep(inv) { vm in
            report = await vm.optimizeHydrogenBondNetwork()?.value
        }
        guard let report else { return }
        let kinds = [("OH", report.rotatableOH), ("SH", report.rotatableSH), ("NH3", report.rotatableNH3),
                     ("amide", report.flipAmide), ("His", report.flipHis)]
            .filter { $0.1 > 0 }.map { "\($0.1) \($0.0)" }.joined(separator: ", ")
        if !kinds.isEmpty { inv.emit(.output, "Movable groups: \(kinds)") }
        let flips = report.decisions.filter(\.isFlipped)
        if !flips.isEmpty {
            let rows = flips.map { decision -> [String] in
                let id = decision.residueID
                return ["\(id.chainID):\(id.residueName)\(id.residueSeq)\(id.insertionCode)", decision.kind.rawValue,
                        decision.chosenLabel, String(format: "%.2f", decision.deltaEnergy)]
            }
            inv.emit(.output, ConsoleTable.render(["Residue", "Group", "State", "ΔE"], rows, rightAligned: [3]))
        }
    }

    /// `prep charges`: assigns partial charges with the chosen method.
    private static func assignCharges(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        try applyPrepOptions(inv)
        try await runPrep(inv) { vm in
            guard let task = vm.assignCharges() else {
                throw ConsoleError("cannot compute \(ConsoleNames.charge(vm.docking.chargeMethod)) charges here"
                    + (vm.docking.chargeMethod == .gasteiger ? " — Gasteiger needs the original PDB text; try method=eem" : ""))
            }
            guard await task.value else {
                throw ConsoleError("\(ConsoleNames.charge(vm.docking.chargeMethod)) charge calculation failed — see the activity log")
            }
        }
    }

    /// `prep minimize`: restrained GPU minimisation with an iteration cap.
    private static func minimize(_ inv: ConsoleInvocation) async throws {
        let steps = inv.int("steps") ?? 100
        guard (1...10_000).contains(steps) else { throw ConsoleError("steps must be between 1 and 10000") }
        try await runPrep(inv) { vm in
            guard let task = vm.minimizeProtein(maxIterations: steps) else {
                throw ConsoleError("restrained minimisation needs a Metal GPU")
            }
            _ = await task.value
        }
    }

    // MARK: Reports

    /// `prep missing`: gaps, chain breaks and residue completeness, without changing anything.
    private static func missingReport(_ inv: ConsoleInvocation) throws {
        let protein = try loadedProtein(inv.viewModel)
        let gaps = ProteinPreparation.detectMissingResidues(in: protein.atoms)
        let breaks = ProteinPreparation.detectChainBreaks(in: protein.atoms, bonds: protein.bonds)
        let incomplete = ProteinPreparation.analyzeResidueCompleteness(atoms: protein.atoms, bonds: protein.bonds)

        if gaps.isEmpty {
            inv.emit(.output, "Missing residues: none — residue numbering is contiguous")
        } else {
            let limit = AppViewModel.maxBuildableLoopLength
            let rows = gaps.map { gap -> [String] in
                let length = gap.gapEnd - gap.gapStart + 1
                return [gap.chainID, "\(gap.gapStart)-\(gap.gapEnd)", "\(length)", length <= limit ? "yes" : "no (> \(limit))"]
            }
            inv.emit(.output, "Missing residues (\(gaps.count) gap\(gaps.count == 1 ? "" : "s")):\n"
                     + ConsoleTable.render(["Chain", "Residues", "Length", "Buildable"], rows, rightAligned: [2]))
        }
        if breaks.isEmpty {
            inv.emit(.output, "Chain breaks: none")
        } else {
            let rows = breaks.map { item -> [String] in
                [item.chainID, "\(item.previousResidueSeq) → \(item.nextResidueSeq)",
                 item.carbonNitrogenDistance.isFinite ? String(format: "%.2f", item.carbonNitrogenDistance) : "—",
                 "\(item.missingResidueCount)", item.isCapped ? "yes" : "no"]
            }
            inv.emit(.output, "Chain breaks (\(breaks.count)):\n"
                     + ConsoleTable.render(["Chain", "Between", "C–N Å", "Missing", "Capped"], rows, rightAligned: [2, 3]))
        }
        if incomplete.isEmpty {
            inv.emit(.output, "Residue completeness: every templated residue is complete")
        } else {
            let heavy = incomplete.filter { !$0.missingHeavyAtoms.isEmpty }.count
            let hydrogens = incomplete.filter { !$0.missingHydrogens.isEmpty }.count
            let extra = incomplete.filter { !$0.extraAtoms.isEmpty }.count
            let shown = inv.bool("all") == true ? incomplete : Array(incomplete.prefix(20))
            let rows = shown.map { ["\($0.chainID):\($0.residueName)\($0.residueSeq)", $0.summary] }
            var text = "Incomplete residues: \(incomplete.count) (\(heavy) missing heavy atoms, \(hydrogens) missing H, \(extra) with extra atoms)\n"
                + ConsoleTable.render(["Residue", "Issue"], rows)
            if shown.count < incomplete.count { text += "\n… \(incomplete.count - shown.count) more (prep missing all=on)" }
            inv.emit(.output, text)
        }
        if !gaps.isEmpty || heavyAtomsMissing(incomplete) {
            inv.emit(.output, "Fix with 'prep repair' (loops + atoms), 'prep repair loops' or 'prep repair atoms'")
        }
    }

    /// Whether any residue lacks heavy atoms (so 'prep repair' would change something).
    private static func heavyAtomsMissing(_ residues: [ProteinResidueCompleteness]) -> Bool {
        residues.contains { !$0.missingHeavyAtoms.isEmpty }
    }

    /// `prep states`: the titratable sites of the last protonation with their pKa contributions.
    private static func protonationStates(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        let states = vm.protonationStates
        guard !states.isEmpty else {
            inv.emit(.output, "No protonation states yet — run 'prep', 'prep protonate' or 'prep hydrogens'")
            return
        }
        let filter = inv.string("type") ?? "all"
        let shown = states.filter { state in
            let charge = state.atomFormalCharges.values.reduce(0, +)
            switch filter {
            case "charged": return charge != 0
            case "his": return state.chemicalKind == .histidine
            case "shifted": return abs(state.shiftedPKa - state.modelPKa) > 1
            default: return true
            }
        }
        guard !shown.isEmpty else {
            inv.emit(.output, "No residue matches type=\(filter)")
            return
        }
        let rows = shown.map { state -> [String] in
            let charge = state.atomFormalCharges.values.reduce(0, +)
            return ["\(state.chainID):\(state.residueName)\(state.residueSeq)\(state.insertionCode)", "\(state.chemicalKind)",
                    state.state.rawValue, String(format: "%.2f", state.modelPKa), String(format: "%.2f", state.shiftedPKa),
                    String(format: "%+.2f", state.burialContribution), String(format: "%+.2f", state.hydrogenBondContribution),
                    String(format: "%+.2f", state.chargeContribution), charge == 0 ? "0" : String(format: "%+d", charge)]
        }
        inv.emit(.output, ConsoleTable.render(["Residue", "Kind", "State", "Model pKa", "pKa", "Burial", "H-bond", "Coulomb", "Charge"],
                                              rows, rightAligned: [3, 4, 5, 6, 7, 8]))
        let net = states.reduce(0) { $0 + $1.atomFormalCharges.values.reduce(0, +) }
        inv.emit(.output, "\(shown.count) of \(states.count) titratable sites at pH \(String(format: "%.1f", vm.molecules.protonationPH)); net titratable charge \(String(format: "%+d", net))")
    }

    /// `prep report`: the recorded preparation steps and whether the receptor is ready for docking.
    private static func preparationReport(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        let log = vm.preparationLog
        if log.isEmpty {
            inv.emit(.output, "No preparation steps on \(protein.name) yet — run 'prep' or individual 'prep' stages")
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "HH:mm:ss"
            let rows = log.enumerated().map { index, entry -> [String] in
                ["\(index + 1)", formatter.string(from: entry.date), entry.step, String(format: "%.1f", entry.pH), entry.detail]
            }
            inv.emit(.output, ConsoleTable.render(["#", "Time", "Step", "pH", "Detail"], rows, rightAligned: [0]))
        }
        let atoms = protein.atoms
        let hydrogens = atoms.filter { $0.element == .H }.count
        let charged = atoms.filter { abs($0.charge) > 0.001 }.count
        let formal = atoms.reduce(0) { $0 + $1.formalCharge }
        let waters = protein.residues.filter(\.isWater).count
        let lines = [
            "Current state: \(atoms.count) atoms, \(hydrogens) H, \(waters) waters"
                + (vm.molecules.keptWaterKeys.isEmpty ? "" : " (\(vm.molecules.keptWaterKeys.count) kept as pocket waters)"),
            "Charges: \(charged) atoms charged (method \(ConsoleNames.charge(vm.docking.chargeMethod))), net formal charge \(String(format: "%+d", formal))",
            "Protonation: pH \(String(format: "%.1f", vm.molecules.protonationPH)), \(vm.protonationStates.count) titratable sites assigned",
            "Ready for docking: " + (hydrogens > 0 && charged > 0 ? "yes" : "no — " + (hydrogens == 0 ? "add hydrogens" : "assign charges"))
        ]
        inv.emit(.output, lines.joined(separator: "\n"))
    }
}

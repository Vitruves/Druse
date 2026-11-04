// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - ligand

extension DruseInterpreter {

    /// The `ligand` command: building, importing, preparing, inspecting and exporting the
    /// Ligand Database, and choosing the docking ligand.
    static func ligandCommands() -> [ConsoleCommand] {
        [LigandConsole.command()]
    }
}

/// Implementation of the `ligand` command. Preparation goes through the Ligand Database's shared
/// `LigandPreparationService`, so console runs, window runs and their session defaults are one.
@MainActor
enum LigandConsole {

    // MARK: Declarations

    /// The `ligand` command declaration: every form, option, range and example used by help and completion.
    static func command() -> ConsoleCommand {
        ConsoleCommand(
            name: "ligand",
            summary: "Build, import, prepare (protomers × tautomers × conformers), inspect and export ligands",
            category: .ligand,
            forms: buildForms() + inspectionForms() + preparationForms() + analysisForms() + databaseForms(),
            examples: [
                "ligand smiles \"CC(=O)Oc1ccccc1C(=O)O\" name=aspirin",
                "ligand smiles \"Cn1cnc2c1c(=O)n(C)c(=O)n2C\" name=caffeine prep=off",
                "ligand load actives.csv smiles-col=canonical_smiles name-col=chembl_id pki-col=pchembl",
                "ligand prep all ph=7.4 max-tautomers=4 conformers=3",
                "ligand prep unprepared mode=dominant pka=rules",
                "ligand prep set ph=6.5 energy-window=6; ligand prep config",
                "ligand list forms",
                "ligand show imatinib",
                "ligand forms imatinib",
                "ligand use imatinib form=2 conformer=3",
                "ligand conformers caffeine generate count=20",
                "ligand pka imatinib ph=5.5",
                "ligand descriptors all",
                "ligand admet all",
                "ligand affinity imatinib ki=12.5",
                "ligand db export ~/Druse/ligands.csv subset=dockable",
                "ligand db stats"
            ]
        )
    }

    /// `<ligand>`: one Ligand Database entry.
    private static let ligandArgument = ConsoleParameter(
        name: "ligand", type: .string, summary: "Entry name or 1-based # from 'ligand list'")

    /// `<ligands…>`: several entries.
    private static let targetsArgument = ConsoleParameter(
        name: "ligands", type: .string,
        summary: "Names, #numbers, ranges (2-5), or all | unprepared | prepared | failed",
        isVariadic: true)

    /// The preparation options shared by `ligand prep`, `ligand prep set`, `ligand smiles`,
    /// `ligand load` and `ligand db import`. Defaults shown are the start-up session defaults.
    private static let prepOptions: [ConsoleParameter] = [
        ConsoleParameter(name: "mode", type: .choice(LigandPrepMode.allCases.map(\.rawValue)),
                         summary: "Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH)",
                         defaultValue: "enumerate"),
        ConsoleParameter(name: "ph", type: .float,
                         summary: "Target pH for protonation states and form populations",
                         defaultValue: "7.4", range: LigandPrepSettings.pHRange),
        ConsoleParameter(name: "pka", type: .choice(PKaMethod.allCases.map(\.consoleName)),
                         summary: "pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups)",
                         defaultValue: "gnn"),
        ConsoleParameter(name: "pka-threshold", type: .float,
                         summary: "Sites with |pKa − pH| below this get both protonation states",
                         defaultValue: "2", range: LigandPrepSettings.pkaThresholdRange),
        ConsoleParameter(name: "max-tautomers", type: .int,
                         summary: "Maximum tautomers per protomer (mode=enumerate)",
                         defaultValue: "10", range: range(LigandPrepSettings.maxTautomersRange)),
        ConsoleParameter(name: "max-protomers", type: .int,
                         summary: "Maximum protonation states of the parent (mode=enumerate)",
                         defaultValue: "8", range: range(LigandPrepSettings.maxProtomersRange)),
        ConsoleParameter(name: "energy-window", type: .float,
                         summary: "Discard forms more than this many kcal/mol (MMFF94) above the best one",
                         defaultValue: "10", range: LigandPrepSettings.energyCutoffRange),
        ConsoleParameter(name: "conformers", type: .int,
                         summary: "3D conformers per chemical form (enumerate/dominant)",
                         defaultValue: "10", range: range(LigandPrepSettings.conformersRange)),
        ConsoleParameter(name: "min-population", type: .float,
                         summary: "Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site)",
                         defaultValue: "1", range: LigandPrepSettings.minPopulationRange),
        ConsoleParameter(name: "temperature", type: .float,
                         summary: "Boltzmann temperature for form populations, in K",
                         defaultValue: "298.15", range: LigandPrepSettings.temperatureRange),
        ConsoleParameter(name: "hydrogens", type: .bool,
                         summary: "Add hydrogens (mode=basic; the ensemble modes always add them)", defaultValue: "on"),
        ConsoleParameter(name: "minimize", type: .bool,
                         summary: "MMFF94-minimize the 3D structure (mode=basic; always on otherwise)", defaultValue: "on"),
        ConsoleParameter(name: "charges", type: .bool,
                         summary: "Assign Gasteiger partial charges (mode=basic; always on otherwise)", defaultValue: "on")
    ]

    /// Options of `ligand load` / `ligand db import` that describe the input file.
    private static let fileOptions: [ConsoleParameter] = [
        ConsoleParameter(name: "format", type: .choice(["auto", "smi", "csv", "sdf", "mol2", "json"]),
                         summary: "File format (auto: from the extension; json is a saved Druse ligand database)",
                         defaultValue: "auto"),
        ConsoleParameter(name: "smiles-col", type: .string, summary: "CSV: SMILES column, by header name or 1-based number"),
        ConsoleParameter(name: "name-col", type: .string, summary: "CSV: name/ID column, by header name or number"),
        ConsoleParameter(name: "ki-col", type: .string, summary: "CSV: Ki column (nM)"),
        ConsoleParameter(name: "pki-col", type: .string, summary: "CSV: pKi column (−log10 Ki in M)"),
        ConsoleParameter(name: "ic50-col", type: .string, summary: "CSV: IC50 column (nM)"),
        ConsoleParameter(name: "delimiter", type: .choice(["auto", "comma", "tab", "semicolon"]),
                         summary: "CSV: field separator (auto: tab if the first line has one, else comma)",
                         defaultValue: "auto"),
        ConsoleParameter(name: "header", type: .choice(["auto", "on", "off"]),
                         summary: "CSV: whether the first row is a header", defaultValue: "auto"),
        ConsoleParameter(name: "extra-cols", type: .bool,
                         summary: "CSV: keep unmapped columns as entry properties", defaultValue: "on")
    ]

    /// Option names that only make sense for CSV/TSV input.
    private static let csvOptionNames = ["smiles-col", "name-col", "ki-col", "pki-col", "ic50-col", "delimiter", "header", "extra-cols"]

    /// An integer option range in the `Double` form `ConsoleParameter.range` takes.
    private static func range(_ ints: ClosedRange<Int>) -> ClosedRange<Double> {
        Double(ints.lowerBound)...Double(ints.upperBound)
    }

    /// Forms that add, rename, remove and annotate entries.
    private static func buildForms() -> [ConsoleForm] {
        [
            ConsoleForm(path: ["smiles"], summary: "Add a ligand from SMILES, prepare it with the session pipeline and make it the docking ligand",
                        arguments: [ConsoleParameter(name: "smiles", type: .string, summary: "SMILES string (quote it)")],
                        options: [
                            ConsoleParameter(name: "name", type: .string, summary: "Entry name (default Ligand_<n>)"),
                            ConsoleParameter(name: "prep", type: .bool, summary: "Prepare right away (off: add as an unprepared entry)", defaultValue: "on"),
                            ConsoleParameter(name: "use", type: .bool, summary: "Make the prepared ligand (its dominant form) the docking ligand", defaultValue: "on")
                        ] + prepOptions) { inv in
                try await addSMILES(inv)
            },
            ConsoleForm(path: ["load"], summary: "Add ligands from a .smi, .csv/.tsv, .sdf/.mol, .mol2 or Druse .json file (not prepared unless prep=on)",
                        arguments: [ConsoleParameter(name: "file", type: .path, summary: "Ligand file")],
                        options: fileOptions + [
                            ConsoleParameter(name: "prep", type: .bool, summary: "Prepare the new entries after loading (sdf/mol2 with prep=off keep their file 3D coordinates)", defaultValue: "off")
                        ] + prepOptions) { inv in
                try await loadFile(inv, prepareByDefault: false)
            },
            ConsoleForm(path: ["rename"], summary: "Rename an entry (its enumerated forms follow)",
                        arguments: [ligandArgument,
                                    ConsoleParameter(name: "new-name", type: .string, summary: "New entry name")]) { inv in
                try rename(inv)
            },
            ConsoleForm(path: ["remove"], summary: "Remove entries; removing a source molecule also removes its enumerated forms",
                        arguments: [targetsArgument]) { inv in
                try remove(inv)
            },
            ConsoleForm(path: ["clear"], summary: "Empty the Ligand Database, or remove only its forms, unprepared or failed entries",
                        arguments: [ConsoleParameter(name: "what", type: .choice(["all", "forms", "unprepared", "failed"]),
                                                     summary: "What to remove", isOptional: true, defaultValue: "all")]) { inv in
                try clear(inv)
            },
            ConsoleForm(path: ["affinity"], summary: "Show or set experimental affinity (copied to the entry's enumerated forms)",
                        arguments: [ligandArgument],
                        options: [
                            ConsoleParameter(name: "ki", type: .float, summary: "Ki in nM", range: 1e-6...1e12),
                            ConsoleParameter(name: "pki", type: .float, summary: "pKi (−log10 Ki in M)", range: 0...20),
                            ConsoleParameter(name: "ic50", type: .float, summary: "IC50 in nM", range: 1e-6...1e12),
                            ConsoleParameter(name: "clear", type: .bool, summary: "Remove all affinity values first")
                        ]) { inv in
                try affinity(inv)
            }
        ]
    }

    /// Forms that list, show and activate entries.
    private static func inspectionForms() -> [ConsoleForm] {
        [
            ConsoleForm(path: ["list"], summary: "List the Ligand Database (forms indented under their source molecule; * = docking ligand)",
                        arguments: [ConsoleParameter(name: "filter", type: .choice(["all", "prepared", "unprepared", "forms", "dockable", "failed"]),
                                                     summary: "Which entries to list (forms: enumerated forms with populations)",
                                                     isOptional: true, defaultValue: "all")]) { inv in
                try list(inv)
            },
            ConsoleForm(path: ["show"], summary: "Everything about one entry: SMILES, status, 3D, descriptors, Lipinski/Veber, forms, conformers, affinity, errors",
                        arguments: [ligandArgument]) { inv in
                try show(inv)
            },
            ConsoleForm(path: ["use"], summary: "Make an entry the docking ligand (a source molecule uses its most populated form unless form= is given)",
                        arguments: [ligandArgument],
                        options: [
                            ConsoleParameter(name: "form", type: .string, summary: "Form # from 'ligand forms', its label (Taut1) or entry name"),
                            ConsoleParameter(name: "conformer", type: .int, summary: "Conformer # from 'ligand conformers' (default: lowest energy)", range: 1...10_000)
                        ]) { inv in
                try use(inv)
            },
            ConsoleForm(path: ["forms"], summary: "Protomers/tautomers of a molecule: kind, population at pH, relative energy, charge, conformers",
                        arguments: [ligandArgument]) { inv in
                try forms(inv)
            },
            ConsoleForm(path: ["conformers"], summary: "List an entry's conformers with MMFF94 energies, or generate a new set",
                        arguments: [ligandArgument,
                                    ConsoleParameter(name: "action", type: .choice(["list", "generate"]),
                                                     summary: "generate replaces the conformers (keeps the entry's protonation)",
                                                     isOptional: true, defaultValue: "list")],
                        options: [
                            ConsoleParameter(name: "count", type: .int, summary: "generate: number of conformers to embed", defaultValue: "10", range: 1...500),
                            ConsoleParameter(name: "minimize", type: .bool, summary: "generate: MMFF94-minimize each conformer", defaultValue: "on")
                        ]) { inv in
                try await conformers(inv)
            }
        ]
    }

    /// `ligand prep` and its session-default subcommands.
    private static func preparationForms() -> [ConsoleForm] {
        [
            ConsoleForm(path: ["prep"], summary: "Prepare entries with the session pipeline; options override the session defaults for this run",
                        arguments: [targetsArgument], options: prepOptions) { inv in
                try await prepare(inv)
            },
            ConsoleForm(path: ["prep", "config"], summary: "Show the session preparation defaults and any run in progress") { inv in
                showConfig(inv)
            },
            ConsoleForm(path: ["prep", "set"], summary: "Change session preparation defaults (also used by the Ligand Database window)",
                        options: prepOptions) { inv in
                try setDefaults(inv)
            },
            ConsoleForm(path: ["prep", "reset"], summary: "Restore the start-up preparation defaults") { inv in
                let service = inv.viewModel.ligandDB.preparation
                service.settings = LigandPrepSettings()
                service.defaultMode = .enumerate
                inv.emit(.success, "Ligand preparation defaults restored")
                showConfig(inv)
            },
            ConsoleForm(path: ["prep", "stop"], summary: "Cancel the preparation running in the Ligand Database (window or console)") { inv in
                let service = inv.viewModel.ligandDB.preparation
                guard service.isRunning else {
                    inv.emit(.output, "No ligand preparation is running")
                    return
                }
                service.cancel()
                inv.emit(.success, "Ligand preparation cancelled; finished ligands keep their results")
            }
        ]
    }

    /// Descriptor, ADMET and pKa analyses.
    private static func analysisForms() -> [ConsoleForm] {
        [
            ConsoleForm(path: ["descriptors"], summary: "Table of RDKit descriptors (MW, logP, TPSA, HBD/HBA, rotatable bonds, rings, Fsp3, Ro5, Veber)",
                        arguments: [targetsArgument]) { inv in
                try descriptors(inv)
            },
            ConsoleForm(path: ["admet"], summary: "ADMET predictions (logP, logD7.4, logS, hERG, CYP2D6/3A4, BBB, metabolic stability, drug-likeness)",
                        arguments: [targetsArgument]) { inv in
                try await admet(inv)
            },
            ConsoleForm(path: ["pka"], summary: "Per-site pKa predictions and ionization at pH (GNN, SMARTS rules, or both side by side)",
                        arguments: [ligandArgument],
                        options: [
                            ConsoleParameter(name: "method", type: .choice(["gnn", "rules", "both"]),
                                             summary: "Prediction source", defaultValue: "both"),
                            ConsoleParameter(name: "ph", type: .float, summary: "pH for the ionized fraction (default: session pH)",
                                             range: LigandPrepSettings.pHRange)
                        ]) { inv in
                try await pka(inv)
            }
        ]
    }

    /// Database file import/export, statistics and the default database file.
    private static func databaseForms() -> [ConsoleForm] {
        [
            ConsoleForm(path: ["db", "import"], summary: "Add and prepare ligands from a file (as 'ligand load' with prep=on)",
                        arguments: [ConsoleParameter(name: "file", type: .path, summary: "Ligand file (.smi, .csv, .tsv, .sdf, .mol, .mol2, .json)")],
                        options: fileOptions + [
                            ConsoleParameter(name: "prep", type: .bool, summary: "Prepare the new entries (off: same as 'ligand load')", defaultValue: "on")
                        ] + prepOptions) { inv in
                try await loadFile(inv, prepareByDefault: true)
            },
            ConsoleForm(path: ["db", "export"], summary: "Write entries as SDF (3D), CSV (all data), SMI or Druse JSON",
                        arguments: [ConsoleParameter(name: "file", type: .path, summary: "Output file; the format follows the extension")],
                        options: [
                            ConsoleParameter(name: "format", type: .choice(["auto", "sdf", "csv", "smi", "json"]),
                                             summary: "Output format (auto: from the extension)", defaultValue: "auto"),
                            ConsoleParameter(name: "subset", type: .choice(["all", "prepared", "dockable", "forms"]),
                                             summary: "Entries to write (dockable: what 'dock batch' docks)", defaultValue: "all"),
                            ConsoleParameter(name: "conformers", type: .choice(["best", "all"]),
                                             summary: "SDF: lowest-energy conformer only, or one record per conformer", defaultValue: "best")
                        ]) { inv in
                try export(inv)
            },
            ConsoleForm(path: ["db", "stats"], summary: "Counts, preparation state, conformers, affinity coverage and property ranges") { inv in
                stats(inv)
            },
            ConsoleForm(path: ["db", "save"], summary: "Save the database to the default file (the window's Save Database)") { inv in
                let db = inv.viewModel.ligandDB
                guard db.save() else { throw ConsoleError("could not save \(LigandDatabase.defaultURL.path) — see the activity log") }
                inv.emit(.success, "Saved \(db.count) entries to \(LigandDatabase.defaultURL.path)")
            },
            ConsoleForm(path: ["db", "load"], summary: "Replace the database with the saved default file (the window's Load Saved Database)") { inv in
                let db = inv.viewModel.ligandDB
                guard FileManager.default.fileExists(atPath: LigandDatabase.defaultURL.path) else {
                    throw ConsoleError("no saved Ligand Database at \(LigandDatabase.defaultURL.path) — use 'ligand db save' first")
                }
                guard db.load() else { throw ConsoleError("could not read \(LigandDatabase.defaultURL.path) — see the activity log") }
                inv.emit(.success, "Loaded \(db.count) entries from \(LigandDatabase.defaultURL.path)")
            }
        ]
    }

    // MARK: Resolving entries

    /// Finds one entry by 1-based number or name. A name shared by a source molecule and its
    /// parent-kind form resolves to the source molecule.
    static func resolve(_ key: String, in db: LigandDatabase) throws -> LigandEntry {
        let entries = db.entries
        guard !entries.isEmpty else {
            throw ConsoleError("the Ligand Database is empty — use 'ligand smiles', 'ligand load' or 'ligand db import'")
        }
        if let number = Int(key) {
            guard entries.indices.contains(number - 1) else {
                throw ConsoleError("no ligand #\(number) — the database has \(entries.count) entr\(entries.count == 1 ? "y" : "ies")")
            }
            return entries[number - 1]
        }
        let exact = entries.filter { $0.name == key }
        let matches = exact.isEmpty ? entries.filter { $0.name.lowercased() == key.lowercased() } : exact
        if let match = matches.first(where: { !$0.isEnumerated }) ?? matches.first {
            return match
        }
        var message = "no ligand named '\(key)'"
        if let hint = ConsoleSuggest.closest(key, in: entries.map(\.name)) {
            message += " — did you mean '\(hint)'?"
        }
        throw ConsoleError(message)
    }

    /// Resolves the words of a `<ligands…>` argument, in database order without duplicates.
    /// Accepts names, numbers, ranges `a-b`, commas, and the keywords all / unprepared / prepared / failed
    /// (keywords select source molecules and standalone entries, not enumerated forms).
    static func resolveTargets(_ text: String, in db: LigandDatabase) throws -> [LigandEntry] {
        let entries = db.entries
        guard !entries.isEmpty else {
            throw ConsoleError("the Ligand Database is empty — use 'ligand smiles', 'ligand load' or 'ligand db import'")
        }
        if entries.contains(where: { $0.name == text }) {
            return [try resolve(text, in: db)]
        }
        let parentsWithForms = Set(entries.compactMap(\.parentName))
        var selected = Set<UUID>()
        for token in text.split(whereSeparator: { $0 == " " || $0 == "," }).map(String.init) {
            switch token.lowercased() {
            case "all":
                selected.formUnion(entries.filter { !$0.isEnumerated }.map(\.id))
            case "unprepared":
                selected.formUnion(entries.filter { !$0.isEnumerated && !$0.isPrepared && !parentsWithForms.contains($0.name) }.map(\.id))
            case "prepared":
                selected.formUnion(entries.filter { !$0.isEnumerated && ($0.isPrepared || parentsWithForms.contains($0.name)) }.map(\.id))
            case "failed":
                selected.formUnion(entries.filter { !$0.isPrepared && $0.preparationError != nil }.map(\.id))
            default:
                let bounds = token.split(separator: "-", maxSplits: 1).map { Int($0) }
                if !entries.contains(where: { $0.name == token }), bounds.count == 2,
                   let low = bounds[0], let high = bounds[1] {
                    guard low >= 1, high >= low, high <= entries.count else {
                        throw ConsoleError("range \(token) is outside 1-\(entries.count)")
                    }
                    selected.formUnion(entries[(low - 1)...(high - 1)].map(\.id))
                } else {
                    selected.insert(try resolve(token, in: db).id)
                }
            }
        }
        let resolved = entries.filter { selected.contains($0.id) }
        guard !resolved.isEmpty else { throw ConsoleError("'\(text)' matches no ligands") }
        return resolved
    }

    /// The enumerated forms of a source molecule, in database (population) order.
    static func forms(of entry: LigandEntry, in db: LigandDatabase) -> [LigandEntry] {
        guard !entry.isEnumerated else { return [] }
        return db.entries.filter { $0.parentName == entry.name }
    }

    /// The source molecule an enumerated form was generated from.
    static func source(of form: LigandEntry, in db: LigandDatabase) -> LigandEntry? {
        guard let parent = form.parentName else { return nil }
        return db.entries.first { $0.name == parent && !$0.isEnumerated }
    }

    // MARK: Preparation options

    /// Settings for a run: session defaults overridden by the statement's options, after checking
    /// that every option given applies to `mode`.
    private static func runSettings(_ inv: ConsoleInvocation, mode: LigandPrepMode) throws -> LigandPrepSettings {
        let given = prepOptions.map(\.name).filter { inv.has($0) && $0 != "mode" }
        switch mode {
        case .basic:
            let pHOnly = ["ph", "pka", "pka-threshold", "max-tautomers", "max-protomers", "energy-window",
                          "conformers", "min-population", "temperature"]
            if let option = given.first(where: { pHOnly.contains($0) }) {
                throw ConsoleError("\(option)= does not apply to mode=basic (no pH handling, one conformer) — use mode=dominant or mode=enumerate")
            }
        case .dominant:
            if let option = given.first(where: { ["max-tautomers", "max-protomers", "min-population"].contains($0) }) {
                throw ConsoleError("\(option)= only applies to mode=enumerate — mode=dominant keeps the single most populated form")
            }
            fallthrough
        case .enumerate:
            if let option = ["hydrogens", "minimize", "charges"].first(where: { inv.bool($0) == false }) {
                throw ConsoleError("\(option)=off needs mode=basic — the \(mode.rawValue) pipeline always adds hydrogens, minimizes with MMFF94 and assigns Gasteiger charges")
            }
        }
        return applyOptions(inv, to: inv.viewModel.ligandDB.preparation.settings)
    }

    /// Copies every preparation option present in the statement onto `base`.
    private static func applyOptions(_ inv: ConsoleInvocation, to base: LigandPrepSettings) -> LigandPrepSettings {
        var settings = base
        if let value = inv.double("ph") { settings.pH = value }
        if let value = inv.string("pka").flatMap(PKaMethod.init(consoleName:)) { settings.pkaMethod = value }
        if let value = inv.double("pka-threshold") { settings.pkaThreshold = value }
        if let value = inv.int("max-tautomers") { settings.maxTautomers = value }
        if let value = inv.int("max-protomers") { settings.maxProtomers = value }
        if let value = inv.double("energy-window") { settings.energyCutoff = value }
        if let value = inv.int("conformers") { settings.conformersPerForm = value }
        if let value = inv.double("min-population") { settings.minPopulationPercent = value }
        if let value = inv.double("temperature") { settings.temperature = value }
        if let value = inv.bool("hydrogens") { settings.addHydrogens = value }
        if let value = inv.bool("minimize") { settings.minimize = value }
        if let value = inv.bool("charges") { settings.computeCharges = value }
        return settings
    }

    /// The statement's `mode=`, else the session default mode.
    private static func mode(_ inv: ConsoleInvocation) -> LigandPrepMode {
        inv.string("mode").flatMap(LigandPrepMode.init(rawValue:)) ?? inv.viewModel.ligandDB.preparation.defaultMode
    }

    /// "enumerate at pH 7.4 (pKa gnn, ≤10 tautomers, ≤8 protomers, 10 conformers/form, window 10 kcal/mol)"
    private static func describe(_ mode: LigandPrepMode, _ s: LigandPrepSettings) -> String {
        switch mode {
        case .basic:
            return "basic (hydrogens \(onOff(s.addHydrogens)), minimize \(onOff(s.minimize)), charges \(onOff(s.computeCharges)))"
        case .dominant:
            return "dominant at pH \(number(s.pH)) (pKa \(s.pkaMethod.consoleName), \(s.conformersPerForm) conformers, window \(number(s.energyCutoff)) kcal/mol)"
        case .enumerate:
            return "enumerate at pH \(number(s.pH)) (pKa \(s.pkaMethod.consoleName), ≤\(s.maxTautomers) tautomers, ≤\(s.maxProtomers) protomers, "
                + "\(s.conformersPerForm) conformers/form, window \(number(s.energyCutoff)) kcal/mol, min population \(number(s.minPopulationPercent))%)"
        }
    }

    // MARK: Running preparation

    /// Prepares `entries` through the shared service, printing one line per ligand and a summary.
    /// Fails when another run is active or no ligand could be prepared; cancelling the statement cancels the run.
    @discardableResult
    private static func runPreparation(
        _ inv: ConsoleInvocation, entries: [LigandEntry],
        mode: LigandPrepMode, settings: LigandPrepSettings
    ) async throws -> LigandPrepReport {
        let service = inv.viewModel.ligandDB.preparation
        // Starting a run cancels the active one, which may be the Ligand Database window's.
        guard !service.isRunning else {
            throw ConsoleError("a ligand preparation is already running (\(service.progress.current)/\(service.progress.total)); wait for it or use 'ligand prep stop'")
        }
        let total = entries.count
        inv.emit(.output, "Preparing \(total) ligand\(total == 1 ? "" : "s"): \(describe(mode, settings))")
        let perLine = total <= 100
        let run = service.prepare(entries, mode: mode, settings: settings) { item, done in
            if perLine {
                inv.emit(item.succeeded ? .output : .warning, itemLine(item, mode: mode, index: done, total: total))
            } else if !item.succeeded {
                inv.emit(.warning, itemLine(item, mode: mode, index: done, total: total))
            } else if done % max(total / 10, 1) == 0 {
                inv.emit(.output, "  \(done)/\(total) prepared…")
            }
        }
        let report = await withTaskCancellationHandler {
            await inv.withProgress({
                let progress = service.progress
                return service.isRunning && progress.total > 0 ? "Preparing ligands \(progress.current)/\(progress.total)" : nil
            }) {
                await run.value
            }
        } onCancel: {
            Task { @MainActor in service.cancel() }
        }
        if report.wasCancelled {
            if report.succeeded > 0 {
                inv.emit(.warning, "\(report.succeeded) of \(total) ligands were prepared before the run stopped")
            }
            throw CancellationError()
        }
        let what = mode == .enumerate
            ? "\(count(report.entriesWritten, "form entry", "form entries")), \(count(report.conformers, "conformer"))"
            : count(report.conformers, "conformer")
        let summary = "Prepared \(report.succeeded)/\(total) ligands in \(seconds(report.seconds)) — \(what)"
        if report.failed.isEmpty {
            inv.emit(.success, summary)
        } else if report.succeeded > 0 {
            inv.emit(.warning, summary + "; \(report.failed.count) failed: " + report.failed.map(\.name).joined(separator: ", "))
        } else {
            throw ConsoleError("no ligand could be prepared — " + report.failed.map { "\($0.name): \($0.error ?? "unknown error")" }.joined(separator: "; "))
        }
        return report
    }

    /// "[2/3] ✓ caffeine: 4 forms, 12 conformers (0.84 s)" — one progress line per prepared ligand.
    private static func itemLine(_ item: LigandPrepItemResult, mode: LigandPrepMode, index: Int, total: Int) -> String {
        let counter = "[\(index)/\(total)]"
        guard item.succeeded else {
            return "  \(counter) ✗ \(item.name): \(item.error ?? "failed") (\(seconds(item.seconds)))"
        }
        let detail = mode == .enumerate
            ? "\(count(item.formCount, "form")), \(count(item.conformerCount, "conformer"))"
            : count(item.conformerCount, "conformer")
        return "  \(counter) ✓ \(item.name): \(detail) (\(seconds(item.seconds)))"
    }

    // MARK: ligand smiles / load / prep

    /// Adds one SMILES entry (validated by RDKit), prepares it unless prep=off, and activates its dominant form.
    private static func addSMILES(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let smiles = inv.requiredString("smiles").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let descriptors = RDKitBridge.computeDescriptors(smiles: smiles) else {
            throw ConsoleError("'\(smiles)' is not a valid SMILES (RDKit could not parse it)")
        }
        let name = try newName(inv.string("name"), in: db)
        let prepare = inv.bool("prep") ?? true
        let runMode = mode(inv)
        if !prepare, let option = (prepOptions.map(\.name) + ["use"]).first(where: inv.has) {
            throw ConsoleError("\(option)= has no effect with prep=off")
        }
        let settings = try runSettings(inv, mode: runMode)
        let entry = LigandEntry(name: name, smiles: smiles, descriptors: descriptors)
        db.add(entry)
        guard prepare else {
            inv.emit(.success, "Added \(name) (not prepared — 'ligand prep \(name)' when ready)")
            return
        }
        try await runPreparation(inv, entries: [entry], mode: runMode, settings: settings)
        guard inv.bool("use") ?? true else { return }
        guard let prepared = db.entries.first(where: { $0.id == entry.id }) else { return }
        inv.emit(.success, try activate(prepared, form: nil, conformer: nil, in: vm))
    }

    /// A unique entry name: `requested` if free (error if taken), else the next free Ligand_<n>.
    private static func newName(_ requested: String?, in db: LigandDatabase) throws -> String {
        let taken = Set(db.entries.map(\.name))
        if let requested {
            let trimmed = requested.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { throw ConsoleError("name= cannot be empty") }
            guard !taken.contains(trimmed) else {
                throw ConsoleError("a ligand named '\(trimmed)' already exists — choose another name= or 'ligand remove \(trimmed)' first")
            }
            return trimmed
        }
        var number = db.count + 1
        while taken.contains("Ligand_\(number)") { number += 1 }
        return "Ligand_\(number)"
    }

    /// Reads a ligand file off the main actor, appends its entries with unique names and prepares them when asked.
    private static func loadFile(_ inv: ConsoleInvocation, prepareByDefault: Bool) async throws {
        let db = inv.viewModel.ligandDB
        let url = inv.requiredURL("file")
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw ConsoleError("no readable file at \(url.path)")
        }
        let format = try fileFormat(inv.string("format"), url: url)
        if format != "csv", let option = csvOptionNames.first(where: inv.has) {
            throw ConsoleError("\(option)= only applies to CSV/TSV tables (this file is read as \(format))")
        }
        let prepare = inv.bool("prep") ?? prepareByDefault
        if !prepare, let option = prepOptions.map(\.name).first(where: inv.has) {
            throw ConsoleError("\(option)= has no effect with prep=off")
        }
        let runMode = mode(inv)
        let settings = try runSettings(inv, mode: runMode)

        let baseIndex = db.count
        var loaded: [LigandEntry]
        do {
            switch format {
            case "smi":
                loaded = try await Task.detached(priority: .userInitiated) {
                    try LigandDatabase.readSMIFile(url, baseIndex: baseIndex)
                }.value
            case "csv":
                let options = try tableOptions(inv)
                loaded = try await Task.detached(priority: .userInitiated) {
                    try LigandDatabase.readTableFile(url, options: options, baseIndex: baseIndex)
                }.value
            case "sdf", "mol2":
                let keepCoordinates = !prepare
                loaded = try await Task.detached(priority: .userInitiated) {
                    try LigandDatabase.readStructureFile(url, coordinatesArePrepared: keepCoordinates)
                }.value
            default:
                loaded = try LigandDatabase.decodeEntries(from: Data(contentsOf: url))
            }
        } catch let error as LigandImportError {
            throw ConsoleError("\(url.lastPathComponent): \(error.message)")
        } catch {
            throw ConsoleError("could not read \(url.lastPathComponent): \(error.localizedDescription)")
        }
        guard !loaded.isEmpty else { throw ConsoleError("no ligands found in \(url.lastPathComponent)") }

        // Enumeration links forms to their source by name, so imported names must be unique.
        // Saved databases (json) already carry those links and keep their names.
        var renamed: [String] = []
        if format != "json" {
            var taken = Set(db.entries.map(\.name))
            for i in loaded.indices {
                let original = loaded[i].name
                var candidate = original
                var suffix = 2
                while taken.contains(candidate) {
                    candidate = "\(original)_\(suffix)"
                    suffix += 1
                }
                if candidate != original { renamed.append("\(original) → \(candidate)") }
                loaded[i].name = candidate
                taken.insert(candidate)
            }
        }
        db.batchMutate { $0.append(contentsOf: loaded) }

        let with3D = loaded.filter { !$0.atoms.isEmpty }.count
        let withAffinity = loaded.filter { $0.effectivePKi != nil }.count
        let invalid = loaded.filter { $0.descriptors == nil }.count
        var parts = ["\(loaded.count) entries"]
        if with3D > 0 { parts.append("\(with3D) with 3D coordinates") }
        if withAffinity > 0 { parts.append("\(withAffinity) with affinity data") }
        inv.emit(.success, "Loaded \(url.lastPathComponent): " + parts.joined(separator: ", "))
        if !renamed.isEmpty {
            inv.emit(.warning, "Renamed \(renamed.count) duplicate name\(renamed.count == 1 ? "" : "s"): "
                     + renamed.prefix(5).joined(separator: ", ") + (renamed.count > 5 ? ", …" : ""))
        }
        if invalid > 0 {
            inv.emit(.warning, "\(invalid) entr\(invalid == 1 ? "y has" : "ies have") a SMILES RDKit cannot parse — they will fail preparation")
        }

        guard prepare else {
            if format == "smi" || format == "csv" {
                inv.emit(.output, "Not prepared yet — run 'ligand prep unprepared' (or load with prep=on)")
            }
            return
        }
        let toPrepare = loaded.filter { !$0.isEnumerated && !$0.isPrepared }
        guard !toPrepare.isEmpty else { return }
        try await runPreparation(inv, entries: toPrepare, mode: runMode, settings: settings)
    }

    /// The import format: the explicit `format=` value, else one inferred from the file extension.
    private static func fileFormat(_ requested: String?, url: URL) throws -> String {
        if let requested, requested != "auto" { return requested }
        switch url.pathExtension.lowercased() {
        case "smi", "smiles", "txt": return "smi"
        case "csv", "tsv": return "csv"
        case "sdf", "sd", "mol": return "sdf"
        case "mol2": return "mol2"
        case "json": return "json"
        default:
            throw ConsoleError("cannot tell the format of '.\(url.pathExtension)' — pass format=smi|csv|sdf|mol2|json")
        }
    }

    /// CSV column mapping and parsing options from the statement.
    private static func tableOptions(_ inv: ConsoleInvocation) throws -> LigandTableImportOptions {
        var options = LigandTableImportOptions()
        options.smilesColumn = inv.string("smiles-col")
        options.nameColumn = inv.string("name-col")
        options.kiColumn = inv.string("ki-col")
        options.pKiColumn = inv.string("pki-col")
        options.ic50Column = inv.string("ic50-col")
        switch inv.string("delimiter") {
        case "comma": options.delimiter = ","
        case "tab": options.delimiter = "\t"
        case "semicolon": options.delimiter = ";"
        default: options.delimiter = inv.requiredURL("file").pathExtension.lowercased() == "tsv" ? "\t" : nil
        }
        switch inv.string("header") {
        case "on": options.hasHeader = true
        case "off": options.hasHeader = false
        default: options.hasHeader = nil
        }
        options.keepExtraColumns = inv.bool("extra-cols") ?? true
        return options
    }

    /// `ligand prep`: resolves the targets (forms map to their source molecule when enumerating) and runs the pipeline.
    private static func prepare(_ inv: ConsoleInvocation) async throws {
        let db = inv.viewModel.ligandDB
        let runMode = mode(inv)
        let settings = try runSettings(inv, mode: runMode)
        var targets: [LigandEntry] = []
        var seen = Set<UUID>()
        for entry in try resolveTargets(inv.requiredString("ligands"), in: db) {
            // Enumeration always starts from the source molecule; re-enumerating replaces its forms.
            var target = entry
            if runMode == .enumerate, entry.isEnumerated {
                guard let parent = source(of: entry, in: db) else {
                    throw ConsoleError("\(entry.name) is an enumerated form whose source molecule '\(entry.parentName ?? "")' is gone — use mode=dominant or mode=basic for it")
                }
                inv.emit(.output, "\(entry.name) is a form of \(parent.name) — enumerating \(parent.name)")
                target = parent
            }
            if seen.insert(target.id).inserted { targets.append(target) }
        }
        if runMode != .enumerate {
            let withForms = targets.filter { !$0.isEnumerated && !forms(of: $0, in: db).isEmpty }.map(\.name)
            if !withForms.isEmpty {
                inv.emit(.warning, "\(withForms.joined(separator: ", ")) also \(withForms.count == 1 ? "has" : "have") enumerated forms, which 'dock batch' docks "
                         + "instead of the molecule itself — 'ligand clear forms' or re-run with mode=enumerate to keep them consistent")
            }
        }
        try await runPreparation(inv, entries: targets, mode: runMode, settings: settings)
    }

    /// Table of the session preparation defaults, the parallelism in effect and any run in progress.
    private static func showConfig(_ inv: ConsoleInvocation) {
        let service = inv.viewModel.ligandDB.preparation
        let s = service.settings
        let rows: [[String]] = [
            ["mode", service.defaultMode.rawValue, service.defaultMode.summary],
            ["ph", number(s.pH), "target pH"],
            ["pka", s.pkaMethod.consoleName, s.pkaMethod.description],
            ["pka-threshold", number(s.pkaThreshold), "both states for sites with |pKa − pH| below this"],
            ["max-tautomers", "\(s.maxTautomers)", "per protomer (enumerate)"],
            ["max-protomers", "\(s.maxProtomers)", "protonation states (enumerate)"],
            ["energy-window", number(s.energyCutoff), "kcal/mol above the best form"],
            ["conformers", "\(s.conformersPerForm)", "per form (enumerate, dominant)"],
            ["min-population", number(s.minPopulationPercent), "% (enumerate)"],
            ["temperature", number(s.temperature), "K, for populations"],
            ["hydrogens", onOff(s.addHydrogens), "basic mode"],
            ["minimize", onOff(s.minimize), "basic mode"],
            ["charges", onOff(s.computeCharges), "basic mode"],
            ["concurrency", "\(AppSettings.shared.maxConcurrentTasks)", "ligands prepared in parallel (Settings)"]
        ]
        var text = ConsoleTable.render(["Key", "Value", "Meaning"], rows)
        if service.isRunning {
            text += "\nRunning: \(service.runningMode?.rawValue ?? "") \(service.progress.current)/\(service.progress.total)"
                + (service.message.isEmpty ? "" : " — \(service.message)")
        }
        inv.emit(.output, text)
    }

    /// `ligand prep set`: stores the given options as session defaults and echoes the new values.
    private static func setDefaults(_ inv: ConsoleInvocation) throws {
        let service = inv.viewModel.ligandDB.preparation
        let changed = prepOptions.map(\.name).filter(inv.has)
        guard !changed.isEmpty else {
            throw ConsoleError("give at least one key=value, e.g. 'ligand prep set ph=6.5' — see 'ligand prep config'")
        }
        if let value = inv.string("mode").flatMap(LigandPrepMode.init(rawValue:)) { service.defaultMode = value }
        service.settings = applyOptions(inv, to: service.settings)
        let values = changed.map { name -> String in
            switch name {
            case "mode": "mode=\(service.defaultMode.rawValue)"
            default: "\(name)=\(configValue(name, service.settings))"
            }
        }
        inv.emit(.success, "Ligand prep defaults: " + values.joined(separator: " "))
    }

    /// The session value of one preparation option, formatted as it would be typed.
    private static func configValue(_ key: String, _ s: LigandPrepSettings) -> String {
        switch key {
        case "ph": number(s.pH)
        case "pka": s.pkaMethod.consoleName
        case "pka-threshold": number(s.pkaThreshold)
        case "max-tautomers": "\(s.maxTautomers)"
        case "max-protomers": "\(s.maxProtomers)"
        case "energy-window": number(s.energyCutoff)
        case "conformers": "\(s.conformersPerForm)"
        case "min-population": number(s.minPopulationPercent)
        case "temperature": number(s.temperature)
        case "hydrogens": onOff(s.addHydrogens)
        case "minimize": onOff(s.minimize)
        case "charges": onOff(s.computeCharges)
        default: "?"
        }
    }

    // MARK: ligand list / show / forms

    /// One-word preparation state for tables: forms, form kind with population, prepared, failed, raw.
    private static func status(_ entry: LigandEntry, formCount: Int) -> String {
        if entry.isEnumerated {
            return (entry.formKind?.label ?? "Form") + (entry.populationWeight.map { String(format: " %.0f%%", $0 * 100) } ?? "")
        }
        if formCount > 0 { return "\(formCount) form\(formCount == 1 ? "" : "s")" }
        if entry.isPrepared { return "prepared" }
        if entry.preparationError != nil { return "failed" }
        return entry.atoms.isEmpty ? "raw" : "3D, unprepared"
    }

    /// `ligand list`: the database (or one subset of it) as a table, with totals.
    private static func list(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let entries = db.entries
        guard !entries.isEmpty else {
            inv.emit(.output, "The Ligand Database is empty — use 'ligand smiles', 'ligand load' or 'ligand db import'")
            return
        }
        let filter = inv.string("filter") ?? "all"
        let formCounts = Dictionary(grouping: entries.compactMap(\.parentName), by: { $0 }).mapValues(\.count)
        let dockable = Set(db.dockableEntries.map(\.id))
        let numbered = entries.enumerated().filter { _, entry in
            switch filter {
            case "prepared": entry.isPrepared
            case "unprepared": !entry.isPrepared && !entry.isEnumerated && formCounts[entry.name] == nil
            case "forms": entry.isEnumerated
            case "dockable": dockable.contains(entry.id)
            case "failed": !entry.isPrepared && entry.preparationError != nil
            default: true
            }
        }
        guard !numbered.isEmpty else {
            inv.emit(.output, "No \(filter) ligands (\(entries.count) entries in the database)")
            return
        }

        let table: String
        if filter == "forms" {
            let rows = numbered.map { i, entry -> [String] in
                [marker(i, entry, vm), entry.name, entry.parentName ?? "", entry.formKind?.label ?? "—",
                 entry.populationWeight.map { String(format: "%.1f%%", $0 * 100) } ?? "—",
                 entry.relativeEnergy.map(energy) ?? "—", "\(entry.conformerCount)",
                 charge(entry), truncated(entry.smiles, 44)]
            }
            table = ConsoleTable.render(["#", "Form", "Of", "Kind", "Pop", "ΔE", "Conf", "Charge", "SMILES"],
                                        rows, rightAligned: [0, 4, 5, 6, 7])
        } else {
            let rows = numbered.map { i, entry -> [String] in
                let heavy = entry.atoms.isEmpty ? "—" : "\(entry.atoms.filter { $0.element != .H }.count)"
                return [marker(i, entry, vm), (entry.isEnumerated ? "└ " : "") + entry.name,
                        status(entry, formCount: formCounts[entry.name, default: 0]),
                        entry.conformerCount > 0 ? "\(entry.conformerCount)" : "—", heavy,
                        entry.descriptors.map { String(format: "%.1f", $0.molecularWeight) } ?? "—",
                        entry.descriptors?.logPText ?? "—",
                        entry.descriptors.map { $0.lipinski ? "pass" : "fail" } ?? "—",
                        truncated(entry.smiles, 44)]
            }
            table = ConsoleTable.render(["#", "Name", "Status", "Conf", "Heavy", "MW", "logP", "Ro5", "SMILES"],
                                        rows, rightAligned: [0, 3, 4, 5, 6])
        }
        let prepared = entries.filter(\.isPrepared).count
        let forms = entries.filter(\.isEnumerated).count
        inv.emit(.output, table + "\n\(entries.count) entries · \(prepared) prepared · \(forms) forms · \(dockable.count) dockable")
    }

    /// Row number with `*` for the docking ligand.
    private static func marker(_ index: Int, _ entry: LigandEntry, _ vm: AppViewModel) -> String {
        "\(index + 1)" + (entry.id == vm.activeLigandEntryID ? "*" : "")
    }

    /// `ligand show`: every stored fact about one entry, plus its forms table for a source molecule.
    private static func show(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let entry = try resolve(inv.requiredString("ligand"), in: db)
        let index = (db.entries.firstIndex { $0.id == entry.id } ?? 0) + 1
        let entryForms = forms(of: entry, in: db)
        var rows: [[String]] = []

        var role = entry.isEnumerated
            ? "\(entry.formKind?.label ?? "form") of \(entry.parentName ?? "?")"
            : entryForms.isEmpty ? "molecule" : "source molecule with \(entryForms.count) enumerated forms"
        if entry.id == vm.activeLigandEntryID { role += ", docking ligand" }
        rows.append(["Entry", "#\(index) — \(role)"])
        rows.append(["SMILES", entry.smiles.isEmpty ? "—" : entry.smiles])
        if entry.isPrepared {
            rows.append(["Status", "prepared" + (entry.preparationDate.map { " (\($0.formatted(date: .abbreviated, time: .shortened)))" } ?? "")])
        } else if let error = entry.preparationError {
            rows.append(["Status", "preparation failed: \(error)"])
        } else {
            rows.append(["Status", entryForms.isEmpty ? "not prepared" : "not prepared itself — dock its forms"])
        }
        if !entry.atoms.isEmpty {
            let heavy = entry.atoms.filter { $0.element != .H }.count
            let charged = entry.atoms.contains { abs($0.charge) > 0.0001 }
            rows.append(["3D", "\(entry.atoms.count) atoms (\(heavy) heavy), \(entry.bonds.count) bonds, net charge \(charge(entry))"
                         + (charged ? ", partial charges assigned" : ", no partial charges")])
        }
        if entry.isEnumerated {
            rows.append(["Population", entry.populationWeight.map { String(format: "%.1f%% at the preparation pH", $0 * 100) } ?? "—"])
            rows.append(["Rel. energy", entry.relativeEnergy.map { energy($0) + " kcal/mol vs best form" } ?? "—"])
        }
        rows.append(["Conformers", conformerSummary(entry)])

        let descriptors = entry.descriptors ?? RDKitBridge.computeDescriptors(smiles: entry.smiles)
        if let d = descriptors {
            rows.append(["Descriptors", String(format: "MW %.2f (exact %.3f), ", d.molecularWeight, d.exactMW)
                         + "logP \(d.logPText), " + String(format: "TPSA %.1f Å², HBD %d, HBA %d", d.tpsa, d.hbd, d.hba)])
            rows.append(["", String(format: "rotatable bonds %d, rings %d (%d aromatic), heavy atoms %d, Fsp3 %.2f",
                                    d.rotatableBonds, d.rings, d.aromaticRings, d.heavyAtomCount, d.fractionCSP3)])
            rows.append(["Lipinski", ruleOfFive(d)])
            rows.append(["Veber", veber(d)])
        }
        rows.append(["Affinity", affinityText(entry)])
        if !entry.userProperties.isEmpty {
            let keys = entry.userProperties.keys.sorted()
            for (i, key) in keys.enumerated() {
                rows.append([i == 0 ? "Properties" : "", "\(key) = \(entry.userProperties[key] ?? "")"])
            }
        }
        var text = "\(entry.name)\n" + rows.map { row in
            row[0].padding(toLength: 12, withPad: " ", startingAt: 0) + row[1]
        }.joined(separator: "\n")
        if !entryForms.isEmpty {
            text += "\n\nForms:\n" + formsTable(entryForms, highlight: nil, vm: vm)
        }
        inv.emit(.output, text)
    }

    /// "3, MMFF94 -45.2…-41.0 kcal/mol" for the show page.
    private static func conformerSummary(_ entry: LigandEntry) -> String {
        guard !entry.conformers.isEmpty else {
            return entry.atoms.isEmpty ? "none" : "1 (the prepared structure, no energy stored)"
        }
        let energies = entry.conformers.map(\.energy)
        guard let low = energies.min(), let high = energies.max() else { return "none" }
        return "\(entry.conformers.count), MMFF94 \(String(format: "%.1f", low))…\(String(format: "%.1f", high)) kcal/mol"
    }

    /// Lipinski verdict with the violated criteria spelled out.
    private static func ruleOfFive(_ d: LigandDescriptors) -> String {
        var violations: [String] = []
        if d.molecularWeight > 500 { violations.append("MW > 500") }
        if d.logP > 5 { violations.append("logP > 5") }
        if d.hbd > 5 { violations.append("HBD > 5") }
        if d.hba > 10 { violations.append("HBA > 10") }
        let verdict = d.lipinski ? "pass" : "fail"
        return violations.isEmpty ? "\(verdict) (0 violations)"
            : "\(verdict) (\(violations.count) violation\(violations.count == 1 ? "" : "s"): \(violations.joined(separator: ", ")))"
    }

    /// Veber verdict with the violated criteria spelled out.
    private static func veber(_ d: LigandDescriptors) -> String {
        var violations: [String] = []
        if d.rotatableBonds > 10 { violations.append("rotatable bonds > 10") }
        if d.tpsa > 140 { violations.append("TPSA > 140") }
        return (d.veber ? "pass" : "fail") + (violations.isEmpty ? "" : " (\(violations.joined(separator: ", ")))")
    }

    /// Experimental affinity values of an entry, or "none".
    private static func affinityText(_ entry: LigandEntry) -> String {
        var parts: [String] = []
        if let ki = entry.ki { parts.append("Ki \(String(format: "%g", ki)) nM") }
        if let pKi = entry.pKi { parts.append("pKi \(String(format: "%.2f", pKi))") }
        if let ic50 = entry.ic50 { parts.append("IC50 \(String(format: "%g", ic50)) nM") }
        guard !parts.isEmpty else { return "none" }
        if let effective = entry.effectivePKi, entry.pKi == nil {
            parts.append("effective pKi \(String(format: "%.2f", effective))")
        }
        return parts.joined(separator: ", ")
    }

    /// Forms of one molecule: kind, population, relative energy, net charge, conformers (← marks `highlight`).
    private static func formsTable(_ forms: [LigandEntry], highlight: UUID?, vm: AppViewModel) -> String {
        let rows = forms.enumerated().map { i, form -> [String] in
            var marks = ""
            if form.id == highlight { marks += "←" }
            if form.id == vm.activeLigandEntryID { marks += "*" }
            return ["\(i + 1)", form.name, form.formKind?.label ?? "—",
                    form.populationWeight.map { String(format: "%.1f%%", $0 * 100) } ?? "—",
                    form.relativeEnergy.map(energy) ?? "—", charge(form), "\(form.conformerCount)",
                    truncated(form.smiles, 48), marks]
        }
        return ConsoleTable.render(["#", "Name", "Kind", "Pop", "ΔE kcal/mol", "Charge", "Conf", "SMILES", ""],
                                   rows, rightAligned: [0, 3, 4, 5, 6])
    }

    /// `ligand forms`: the enumerated forms of a molecule (or of the molecule a form belongs to).
    private static func forms(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let entry = try resolve(inv.requiredString("ligand"), in: db)
        if entry.isEnumerated {
            guard let parent = source(of: entry, in: db) else {
                inv.emit(.output, "\(entry.name) is a \(entry.formKind?.label.lowercased() ?? "form") whose source molecule is no longer in the database")
                return
            }
            inv.emit(.output, "Forms of \(parent.name) (← \(entry.name)):\n" + formsTable(forms(of: parent, in: db), highlight: entry.id, vm: vm))
            return
        }
        let entryForms = forms(of: entry, in: db)
        guard !entryForms.isEmpty else {
            let state = entry.isPrepared
                ? "was prepared as a single form (\(truncated(entry.smiles, 60)))"
                : "is not prepared"
            inv.emit(.output, "\(entry.name) \(state) — 'ligand prep \(entry.name) mode=enumerate' lists its protomers and tautomers")
            return
        }
        let total = entryForms.reduce(0) { $0 + $1.conformerCount }
        inv.emit(.output, "Forms of \(entry.name) — \(count(entryForms.count, "form")), \(count(total, "conformer")) (populations at the preparation pH):\n"
                 + formsTable(entryForms, highlight: nil, vm: vm))
    }

    // MARK: ligand use / conformers

    /// Makes `entry` (or, for a source molecule, one of its forms) the docking ligand and
    /// returns the confirmation line.
    static func activate(_ entry: LigandEntry, form: String?, conformer: Int?, in vm: AppViewModel) throws -> String {
        let db = vm.ligandDB
        var chosen = entry
        let entryForms = forms(of: entry, in: db)
        if let form {
            guard !entryForms.isEmpty else {
                throw ConsoleError("\(entry.name) has no enumerated forms — form= needs 'ligand prep \(entry.name) mode=enumerate' first")
            }
            if let number = Int(form) {
                guard entryForms.indices.contains(number - 1) else {
                    throw ConsoleError("\(entry.name) has \(entryForms.count) forms — form= must be 1-\(entryForms.count)")
                }
                chosen = entryForms[number - 1]
            } else if let match = entryForms.first(where: {
                $0.name.lowercased() == form.lowercased()
                    || $0.name.lowercased().hasSuffix("_" + form.lowercased())
                    || (form.lowercased() == "parent" && $0.formKind == .parent)
            }) {
                chosen = match
            } else {
                var message = "\(entry.name) has no form '\(form)'"
                if let hint = ConsoleSuggest.closest(form, in: entryForms.map(\.name)) { message += " — did you mean '\(hint)'?" }
                throw ConsoleError(message + "; see 'ligand forms \(entry.name)'")
            }
        } else if let dominant = entryForms.max(by: { ($0.populationWeight ?? 0) < ($1.populationWeight ?? 0) }) {
            chosen = dominant
        }
        guard !chosen.atoms.isEmpty else {
            throw ConsoleError("\(chosen.name) has no 3D coordinates — prepare it first ('ligand prep \(chosen.name)')")
        }
        var atoms = chosen.atoms
        var bonds = chosen.bonds
        if let conformer {
            let count = max(chosen.conformers.count, 1)
            guard conformer <= count else {
                throw ConsoleError("\(chosen.name) has \(count) conformer\(count == 1 ? "" : "s") — conformer= must be 1-\(count)")
            }
            if !chosen.conformers.isEmpty {
                atoms = chosen.conformers[conformer - 1].atoms
                bonds = chosen.conformers[conformer - 1].bonds
            }
        }
        let ligand = Molecule(name: chosen.name, atoms: atoms, bonds: bonds, title: chosen.smiles, smiles: chosen.smiles)
        vm.setLigandForDocking(ligand, entryID: chosen.id)
        var line = "Docking ligand: \(chosen.name) (\(ligand.heavyAtomCount) heavy atoms, \(ligand.atomCount) total"
        if let population = chosen.populationWeight { line += String(format: ", %.1f%% population", population * 100) }
        if let conformer { line += ", conformer \(conformer)" }
        return line + ")"
    }

    /// `ligand use`: makes an entry, form or conformer the docking ligand.
    private static func use(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let entry = try resolve(inv.requiredString("ligand"), in: vm.ligandDB)
        inv.emit(.success, try activate(entry, form: inv.string("form"), conformer: inv.int("conformer"), in: vm))
    }

    /// `ligand conformers`: lists the stored conformers, or re-embeds a new set with RDKit (keeping the entry's protonation).
    private static func conformers(_ inv: ConsoleInvocation) async throws {
        let db = inv.viewModel.ligandDB
        var entry = try resolve(inv.requiredString("ligand"), in: db)
        let generate = inv.string("action") == "generate"
        if !generate, let option = ["count", "minimize"].first(where: inv.has) {
            throw ConsoleError("\(option)= applies to 'ligand conformers \(entry.name) generate'")
        }
        guard generate else {
            // A source molecule without its own 3D structure: show its dominant form, which 'ligand use' would dock.
            if entry.atoms.isEmpty,
               let dominant = forms(of: entry, in: db).max(by: { ($0.populationWeight ?? 0) < ($1.populationWeight ?? 0) }) {
                inv.emit(.output, "\(entry.name) is docked through its forms — showing its most populated form")
                entry = dominant
            }
            inv.emit(.output, conformerTable(entry))
            return
        }
        if !forms(of: entry, in: db).isEmpty {
            throw ConsoleError("\(entry.name) is a source molecule with enumerated forms — generate conformers for one of its forms (see 'ligand forms \(entry.name)')")
        }
        guard !entry.smiles.isEmpty else { throw ConsoleError("\(entry.name) has no SMILES to embed") }
        let smiles = entry.smiles, name = entry.name
        let count = inv.int("count") ?? 10
        let minimize = inv.bool("minimize") ?? true
        let clock = ContinuousClock()
        let start = clock.now
        let generated = await Task.detached(priority: .userInitiated) {
            RDKitBridge.generateConformers(smiles: smiles, name: name, count: count, minimize: minimize)
        }.value.sorted { $0.energy < $1.energy }
        guard let best = generated.first else {
            throw ConsoleError("RDKit could not embed any conformer of \(entry.name)")
        }
        guard var updated = db.entries.first(where: { $0.id == entry.id }) else {
            throw ConsoleError("\(entry.name) was removed while its conformers were generated")
        }
        updated.conformers = generated.enumerated().map { i, item in
            Conformer3D(id: i, atoms: item.molecule.atoms, bonds: item.molecule.bonds, energy: item.energy)
        }
        updated.atoms = best.molecule.atoms
        updated.bonds = best.molecule.bonds
        updated.conformerCount = generated.count
        updated.isPrepared = true
        updated.preparationDate = Date()
        updated.preparationError = nil
        db.update(updated)
        let elapsed = clock.now - start
        inv.emit(.success, "\(entry.name): \(generated.count)/\(count) conformers in \(seconds(elapsed))")
        inv.emit(.output, conformerTable(updated))
    }

    /// Conformers with absolute and relative MMFF94 energies.
    private static func conformerTable(_ entry: LigandEntry) -> String {
        guard !entry.conformers.isEmpty else {
            return entry.atoms.isEmpty
                ? "\(entry.name) has no 3D structure — 'ligand prep \(entry.name)' first"
                : "\(entry.name) has 1 conformer (the prepared structure; no energies stored) — 'ligand conformers \(entry.name) generate' embeds more"
        }
        let best = entry.conformers.map(\.energy).min() ?? 0
        let rows = entry.conformers.enumerated().map { i, conformer -> [String] in
            ["\(i + 1)", String(format: "%.2f", conformer.energy), String(format: "+%.2f", conformer.energy - best),
             "\(conformer.atoms.count)"]
        }
        return "Conformers of \(entry.name) (MMFF94, kcal/mol):\n"
            + ConsoleTable.render(["#", "Energy", "ΔE", "Atoms"], rows, rightAligned: [0, 1, 2, 3])
    }

    // MARK: rename / remove / clear / affinity

    /// `ligand rename`: renames an entry; a source molecule's forms keep their link and name prefix.
    private static func rename(_ inv: ConsoleInvocation) throws {
        let db = inv.viewModel.ligandDB
        let entry = try resolve(inv.requiredString("ligand"), in: db)
        let newName = inv.requiredString("new-name").trimmingCharacters(in: .whitespaces)
        guard !newName.isEmpty else { throw ConsoleError("the new name cannot be empty") }
        guard newName != entry.name else { return }
        guard !db.entries.contains(where: { $0.name == newName }) else {
            throw ConsoleError("a ligand named '\(newName)' already exists")
        }
        let oldName = entry.name
        let isSource = !entry.isEnumerated
        db.batchMutate { entries in
            for i in entries.indices {
                if entries[i].id == entry.id {
                    entries[i].name = newName
                } else if isSource, entries[i].parentName == oldName {
                    entries[i].parentName = newName
                    if entries[i].name == oldName {
                        entries[i].name = newName
                    } else if entries[i].name.hasPrefix(oldName + "_") {
                        entries[i].name = newName + entries[i].name.dropFirst(oldName.count)
                    }
                }
            }
        }
        inv.emit(.success, "Renamed \(oldName) → \(newName)")
    }

    /// `ligand remove`: deletes entries and the enumerated forms of removed source molecules.
    private static func remove(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let targets = try resolveTargets(inv.requiredString("ligands"), in: db)
        var ids = Set(targets.map(\.id))
        let sourceNames = Set(targets.filter { !$0.isEnumerated }.map(\.name))
        ids.formUnion(db.entries.filter { $0.parentName.map(sourceNames.contains) == true }.map(\.id))
        let removedActive = vm.activeLigandEntryID.map(ids.contains) == true
        db.remove(ids: ids)
        let forms = ids.count - targets.count
        inv.emit(.success, "Removed \(targets.count) entr\(targets.count == 1 ? "y" : "ies")"
                 + (forms > 0 ? " and \(forms) enumerated form\(forms == 1 ? "" : "s")" : "")
                 + (removedActive ? " (the docking ligand stays loaded)" : ""))
    }

    /// `ligand clear`: empties the database or removes one category of entries.
    private static func clear(_ inv: ConsoleInvocation) throws {
        let db = inv.viewModel.ligandDB
        let what = inv.string("what") ?? "all"
        let before = db.count
        let parentsWithForms = Set(db.entries.compactMap(\.parentName))
        switch what {
        case "forms": db.batchMutate { $0.removeAll(where: \.isEnumerated) }
        case "unprepared":
            db.batchMutate { $0.removeAll { !$0.isPrepared && !$0.isEnumerated && !parentsWithForms.contains($0.name) } }
        case "failed": db.batchMutate { $0.removeAll { !$0.isPrepared && $0.preparationError != nil } }
        default: db.removeAll()
        }
        let removed = before - db.count
        let noun = what == "forms" ? "enumerated form" : "\(what) entr"
        let plural = what == "forms" ? (removed == 1 ? "" : "s") : (removed == 1 ? "y" : "ies")
        inv.emit(.success, what == "all"
                 ? "Cleared the Ligand Database (\(removed) entries)"
                 : "Removed \(removed) \(noun)\(plural); \(db.count) left")
    }

    /// `ligand affinity`: shows or sets Ki / pKi / IC50 on an entry and its forms.
    private static func affinity(_ inv: ConsoleInvocation) throws {
        let db = inv.viewModel.ligandDB
        let entry = try resolve(inv.requiredString("ligand"), in: db)
        let ki = inv.double("ki").map(Float.init)
        let pKi = inv.double("pki").map(Float.init)
        let ic50 = inv.double("ic50").map(Float.init)
        let clear = inv.bool("clear") ?? false
        guard clear || ki != nil || pKi != nil || ic50 != nil else {
            inv.emit(.output, "\(entry.name): \(affinityText(entry))")
            return
        }
        // Enumerated forms carry their source molecule's measurements.
        let ids = Set([entry.id] + forms(of: entry, in: db).map(\.id))
        db.batchMutate { entries in
            for i in entries.indices where ids.contains(entries[i].id) {
                if clear {
                    entries[i].ki = nil
                    entries[i].pKi = nil
                    entries[i].ic50 = nil
                    for key in [LigandDatabase.kiPropertyKey, LigandDatabase.pKiPropertyKey, LigandDatabase.ic50PropertyKey] {
                        entries[i].userProperties[key] = nil
                    }
                }
                LigandDatabase.setAffinity(&entries[i], ki: ki, pKi: pKi, ic50: ic50)
            }
        }
        let updated = db.entries.first { $0.id == entry.id } ?? entry
        inv.emit(.success, "\(entry.name): \(affinityText(updated))"
                 + (ids.count > 1 ? " (also on its \(count(ids.count - 1, "form")))" : ""))
    }

    // MARK: descriptors / admet / pka

    /// `ligand descriptors`: RDKit descriptor table, computing (and storing) missing descriptors.
    private static func descriptors(_ inv: ConsoleInvocation) throws {
        let db = inv.viewModel.ligandDB
        let targets = try analysisTargets(inv.requiredString("ligands"), in: db)
        var missing: [String] = []
        var computed: [LigandEntry] = []
        let rows = targets.compactMap { entry -> [String]? in
            var current = entry
            if current.descriptors == nil {
                current.descriptors = RDKitBridge.computeDescriptors(smiles: entry.smiles)
                if current.descriptors != nil { computed.append(current) }
            }
            guard let d = current.descriptors else {
                missing.append(entry.name)
                return nil
            }
            return [entry.name, String(format: "%.1f", d.molecularWeight), d.logPText, String(format: "%.1f", d.tpsa),
                    "\(d.hbd)", "\(d.hba)", "\(d.rotatableBonds)", "\(d.rings)", "\(d.aromaticRings)",
                    "\(d.heavyAtomCount)", String(format: "%.2f", d.fractionCSP3),
                    d.lipinski ? "pass" : "fail", d.veber ? "pass" : "fail"]
        }
        for entry in computed { db.update(entry) }
        if !rows.isEmpty {
            inv.emit(.output, ConsoleTable.render(
                ["Name", "MW", "logP", "TPSA", "HBD", "HBA", "RotB", "Rings", "Arom", "Heavy", "Fsp3", "Ro5", "Veber"],
                rows, rightAligned: Set(1...10)))
        }
        if !missing.isEmpty {
            inv.emit(.warning, "No descriptors for \(missing.joined(separator: ", ")) (SMILES RDKit cannot parse)")
        }
        if rows.isEmpty { throw ConsoleError("no descriptors could be computed") }
    }

    /// `<ligands…>` for read-only analyses: the keyword `all` includes enumerated forms.
    private static func analysisTargets(_ text: String, in db: LigandDatabase) throws -> [LigandEntry] {
        if text.lowercased() == "all", !db.entries.isEmpty { return db.entries }
        return try resolveTargets(text, in: db)
    }

    /// `ligand admet`: CoreML ADMET predictions (rule-based values when the models are not bundled).
    private static func admet(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let targets = try analysisTargets(inv.requiredString("ligands"), in: vm.ligandDB).filter { !$0.smiles.isEmpty }
        guard !targets.isEmpty else { throw ConsoleError("the selected entries have no SMILES") }
        let predictor = vm.admetPredictor
        if !predictor.isAvailable { predictor.loadModels() }
        if !predictor.isAvailable {
            inv.emit(.warning, "ADMET ML models are not in this build — showing RDKit rule-based values only (logP, Ro5, Veber, drug-likeness)")
        }
        func value(_ x: Float?, _ format: String = "%.2f") -> String { x.map { String(format: format, $0) } ?? "—" }
        var rows: [[String]] = []
        for entry in targets {
            try Task.checkCancellation()
            let r = await predictor.predict(smiles: entry.smiles)
            rows.append([entry.name, value(r.logP), value(r.logD74), value(r.aqueousSolubility),
                         value(r.hergLiability), value(r.cyp2d6Inhibition), value(r.cyp3a4Inhibition),
                         value(r.bbbPermeability), value(r.metabolicStability),
                         r.lipinski ? "pass" : "fail", r.veber ? "pass" : "fail", value(r.drugLikeness)])
        }
        inv.emit(.output, ConsoleTable.render(
            ["Name", "logP", "logD7.4", "logS", "hERG", "CYP2D6", "CYP3A4", "BBB", "MetStab", "Ro5", "Veber", "DL"],
            rows, rightAligned: Set(1...8).union([11]))
            + "\nhERG/CYP/BBB/MetStab: predicted probabilities (hERG, CYP: inhibition; MetStab: stable); DL: 0–1 rule-based drug-likeness")
    }

    /// One ionizable site as seen by the GNN, the SMARTS rules, or both.
    private struct SiteRow {
        var isAcid: Bool
        var group: String?
        var gnnPKa: Double?
        var gnnProbability: Float?
        var rulesPKa: Double?
    }

    /// `ligand pka`: per-site pKa from the GNN and/or SMARTS rules, with the Henderson–Hasselbalch ionized fraction at pH.
    private static func pka(_ inv: ConsoleInvocation) async throws {
        let db = inv.viewModel.ligandDB
        let entry = try resolve(inv.requiredString("ligand"), in: db)
        // pKa is a property of the input molecule: forms use their source's SMILES.
        let smiles = entry.isEnumerated ? (source(of: entry, in: db)?.smiles ?? entry.smiles) : entry.smiles
        guard !smiles.isEmpty else { throw ConsoleError("\(entry.name) has no SMILES") }
        let settings = db.preparation.settings
        let pH = inv.double("ph") ?? settings.pH
        var method = inv.string("method") ?? "both"
        if method != "rules", !PKaGNNPredictor.isAvailable {
            inv.emit(.warning, "The pKa GNN is not available (Metal or weights missing) — using the SMARTS rules")
            method = "rules"
        }
        let useGNN = method != "rules", useRules = method != "gnn"
        let (gnn, rules, depiction) = await Task.detached(priority: .userInitiated) {
            (useGNN ? PKaGNNPredictor.predict(smiles: smiles) : [],
             useRules ? RDKitBridge.detectIonizableSites(smiles: smiles) : [],
             RDKitBridge.compute2DCoords(smiles: smiles))
        }.value
        var sites: [Int: SiteRow] = [:]
        for site in rules {
            sites[site.atomIdx] = SiteRow(isAcid: site.isAcid, group: site.groupName, rulesPKa: site.defaultPKa)
        }
        for prediction in gnn {
            var row = sites[prediction.atomIdx] ?? SiteRow(isAcid: prediction.isAcid)
            row.isAcid = prediction.isAcid
            row.gnnPKa = prediction.pKa
            row.gnnProbability = prediction.ionizableProbability
            sites[prediction.atomIdx] = row
        }
        guard !sites.isEmpty else {
            inv.emit(.output, "\(entry.name): no ionizable sites found (\(method)) — neutral at every pH")
            return
        }
        let rows = sites.keys.sorted().compactMap { atom -> [String]? in
            guard let site = sites[atom] else { return nil }
            let symbol = depiction.flatMap { atom < $0.atomicNums.count ? Element(rawValue: $0.atomicNums[atom])?.symbol : nil } ?? "?"
            let pKa = site.gnnPKa ?? site.rulesPKa ?? 0
            // Henderson–Hasselbalch: fraction in the charged state at this pH.
            let ionized = site.isAcid ? 1 / (1 + pow(10, pKa - pH)) : 1 / (1 + pow(10, pH - pKa))
            let state = ionized >= 0.5 ? (site.isAcid ? "deprotonated (−)" : "protonated (+)") : "neutral"
            let both = abs(pKa - pH) < settings.pkaThreshold ? " ±" : ""
            var row = ["\(symbol)\(atom)", site.group ?? "—", site.isAcid ? "acid" : "base"]
            if useGNN { row += [site.gnnPKa.map { String(format: "%.2f", $0) } ?? "—", site.gnnProbability.map { String(format: "%.2f", $0) } ?? "—"] }
            if useRules { row += [site.rulesPKa.map { String(format: "%.1f", $0) } ?? "—"] }
            row += [String(format: "%.1f%%", ionized * 100), state + both]
            return row
        }
        var headers = ["Atom", "Group", "Type"]
        if useGNN { headers += ["pKa GNN", "P(ion)"] }
        if useRules { headers += ["pKa rules"] }
        headers += ["Ionized", "State at pH \(number(pH))"]
        let numeric = Set(3..<(headers.count - 1))
        inv.emit(.output, "pKa sites of \(entry.name) (\(smiles)):\n"
                 + ConsoleTable.render(headers, rows, rightAligned: numeric)
                 + "\nIonized/State use the GNN pKa when present, else the rules value; the preparation pipeline may replace GNN values "
                 + "with the rules value for well-characterised groups (see the activity log).\n± = within pka-threshold "
                 + "\(number(settings.pkaThreshold)) of the pH: preparation enumerates both states. Atom numbers are RDKit heavy-atom indices.")
    }

    // MARK: db export / stats

    /// `ligand db export`: writes a subset of the database as SDF, CSV, SMI or Druse JSON.
    private static func export(_ inv: ConsoleInvocation) throws {
        let db = inv.viewModel.ligandDB
        let url = inv.requiredURL("file")
        let format: String
        if let requested = inv.string("format"), requested != "auto" {
            format = requested
        } else {
            switch url.pathExtension.lowercased() {
            case "sdf", "sd": format = "sdf"
            case "csv": format = "csv"
            case "smi", "smiles": format = "smi"
            case "json": format = "json"
            default: throw ConsoleError("cannot tell the format of '.\(url.pathExtension)' — use .sdf, .csv, .smi or .json, or pass format=")
            }
        }
        if format != "sdf", inv.has("conformers") {
            throw ConsoleError("conformers= only applies to SDF export")
        }
        let subset: [LigandEntry]
        switch inv.string("subset") ?? "all" {
        case "prepared": subset = db.entries.filter(\.isPrepared)
        case "dockable": subset = db.dockableEntries
        case "forms": subset = db.entries.filter(\.isEnumerated)
        default: subset = db.entries
        }
        guard !subset.isEmpty else {
            throw ConsoleError(db.entries.isEmpty ? "the Ligand Database is empty" : "no entries match subset=\(inv.string("subset") ?? "all")")
        }
        var written = subset.count
        var skipped = 0
        let content: Data
        switch format {
        case "sdf":
            let with3D = subset.filter { !$0.atoms.isEmpty }
            skipped = subset.count - with3D.count
            guard !with3D.isEmpty else { throw ConsoleError("none of these entries has 3D coordinates — prepare them or export CSV/SMI") }
            let (text, records) = sdf(with3D, allConformers: inv.string("conformers") == "all")
            content = Data(text.utf8)
            written = records
        case "csv":
            content = Data(csv(subset).utf8)
        case "smi":
            let lines = subset.filter { !$0.smiles.isEmpty }.map { "\($0.smiles)\t\($0.name)" }
            content = Data((lines.joined(separator: "\n") + "\n").utf8)
        default:
            do { content = try db.encodeToData(subset) } catch {
                throw ConsoleError("could not encode the database: \(error.localizedDescription)")
            }
        }
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: url, options: .atomic)
        } catch {
            throw ConsoleError("could not write \(url.path): \(error.localizedDescription)")
        }
        inv.emit(.success, "Wrote \(written) \(format == "sdf" ? "records" : "entries") to \(url.path)")
        if skipped > 0 {
            inv.emit(.warning, "Skipped \(skipped) entr\(skipped == 1 ? "y" : "ies") without 3D coordinates (SDF needs them)")
        }
    }

    /// SD records with every entry field as a data item; returns the text and the record count.
    private static func sdf(_ entries: [LigandEntry], allConformers: Bool) -> (String, Int) {
        var blocks: [String] = []
        for entry in entries {
            var properties: [String: String] = ["SMILES": entry.smiles, "Prepared": entry.isPrepared ? "yes" : "no"]
            if let d = entry.descriptors {
                properties["MW"] = String(format: "%.2f", d.molecularWeight)
                properties["LogP"] = String(format: "%.2f", d.logP)
                properties["TPSA"] = String(format: "%.1f", d.tpsa)
                properties["HBD"] = "\(d.hbd)"
                properties["HBA"] = "\(d.hba)"
                properties["RotBonds"] = "\(d.rotatableBonds)"
                properties["Lipinski"] = d.lipinski ? "PASS" : "FAIL"
            }
            if let parent = entry.parentName { properties["Parent"] = parent }
            if let kind = entry.formKind { properties["FormKind"] = kind.label }
            if let population = entry.populationWeight { properties["Population"] = String(format: "%.4f", population) }
            if let energy = entry.relativeEnergy { properties["RelativeEnergy"] = String(format: "%.3f", energy) }
            if let ki = entry.ki { properties["Ki_nM"] = String(format: "%g", ki) }
            if let pKi = entry.pKi { properties["pKi"] = String(format: "%g", pKi) }
            if let ic50 = entry.ic50 { properties["IC50_nM"] = String(format: "%g", ic50) }
            for (key, value) in entry.userProperties where properties[key] == nil { properties[key] = value }

            if allConformers, !entry.conformers.isEmpty {
                for (i, conformer) in entry.conformers.enumerated() {
                    var conformerProperties = properties
                    conformerProperties["Conformer"] = "\(i + 1)"
                    conformerProperties["MMFF_Energy"] = String(format: "%.3f", conformer.energy)
                    blocks.append(SDFWriter.molBlock(name: entry.name, atoms: conformer.atoms, bonds: conformer.bonds,
                                                     properties: conformerProperties))
                }
            } else {
                blocks.append(SDFWriter.molBlock(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, properties: properties))
            }
        }
        return (blocks.joined(), blocks.count)
    }

    /// One row per entry with identity, preparation, form, descriptor, affinity and user-property columns.
    private static func csv(_ entries: [LigandEntry]) -> String {
        let mirrored: Set<String> = [LigandDatabase.kiPropertyKey, LigandDatabase.pKiPropertyKey, LigandDatabase.ic50PropertyKey]
        let userKeys = Array(Set(entries.flatMap(\.userProperties.keys)).subtracting(mirrored)).sorted()
        let headers = ["name", "smiles", "parent", "form_kind", "population", "relative_energy", "prepared", "conformers",
                       "atoms", "mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "rings", "aromatic_rings",
                       "heavy_atoms", "fsp3", "lipinski", "veber", "ki_nm", "pki", "ic50_nm", "error"] + userKeys
        func field(_ text: String) -> String {
            text.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" })
                ? "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : text
        }
        func optional<T>(_ value: T?, _ format: (T) -> String) -> String { value.map(format) ?? "" }
        let lines = entries.map { e -> String in
            let d = e.descriptors
            let cells: [String] = [
                e.name, e.smiles, e.parentName ?? "", e.formKind?.label ?? "",
                optional(e.populationWeight) { String(format: "%.4f", $0) },
                optional(e.relativeEnergy) { String(format: "%.3f", $0) },
                e.isPrepared ? "yes" : "no", "\(e.conformerCount)", "\(e.atoms.count)",
                optional(d) { String(format: "%.2f", $0.molecularWeight) }, optional(d) { String(format: "%.2f", $0.logP) },
                optional(d) { String(format: "%.1f", $0.tpsa) }, optional(d) { "\($0.hbd)" }, optional(d) { "\($0.hba)" },
                optional(d) { "\($0.rotatableBonds)" }, optional(d) { "\($0.rings)" }, optional(d) { "\($0.aromaticRings)" },
                optional(d) { "\($0.heavyAtomCount)" }, optional(d) { String(format: "%.3f", $0.fractionCSP3) },
                optional(d) { $0.lipinski ? "pass" : "fail" }, optional(d) { $0.veber ? "pass" : "fail" },
                optional(e.ki) { String(format: "%g", $0) }, optional(e.pKi) { String(format: "%g", $0) },
                optional(e.ic50) { String(format: "%g", $0) }, e.preparationError ?? ""
            ] + userKeys.map { e.userProperties[$0] ?? "" }
            return cells.map(field).joined(separator: ",")
        }
        return ([headers.map(field).joined(separator: ",")] + lines).joined(separator: "\n") + "\n"
    }

    /// `ligand db stats`: counts, preparation state, affinity coverage and property ranges.
    private static func stats(_ inv: ConsoleInvocation) {
        let vm = inv.viewModel
        let db = vm.ligandDB
        let entries = db.entries
        guard !entries.isEmpty else {
            inv.emit(.output, "The Ligand Database is empty")
            return
        }
        let parentsWithForms = Set(entries.compactMap(\.parentName))
        let forms = entries.filter(\.isEnumerated)
        let molecules = entries.filter { !$0.isEnumerated }
        let descriptors = entries.compactMap(\.descriptors)
        var rows: [[String]] = [
            ["Entries", "\(entries.count) (\(molecules.count) molecules, \(forms.count) enumerated forms of \(parentsWithForms.count))"],
            ["Prepared", "\(entries.filter(\.isPrepared).count)"],
            ["Unprepared", "\(molecules.filter { !$0.isPrepared && !parentsWithForms.contains($0.name) }.count)"],
            ["Failed", "\(entries.filter { !$0.isPrepared && $0.preparationError != nil }.count)"],
            ["Dockable", "\(db.dockableEntries.count) (what 'dock batch' docks)"],
            ["Conformers", "\(entries.reduce(0) { $0 + $1.conformerCount })"],
            ["Affinity data", "\(entries.filter { $0.effectivePKi != nil }.count)"]
        ]
        if !descriptors.isEmpty {
            let mw = descriptors.map(\.molecularWeight), logP = descriptors.map(\.logP)
            let mean = { (values: [Float]) in values.reduce(0, +) / Float(values.count) }
            // Round first so values like -0.04 print as 0.0, not -0.0.
            let tenth = { (value: Float) -> Float in let r = (value * 10).rounded() / 10; return r == 0 ? 0 : r }
            rows.append(["MW", String(format: "%.1f–%.1f (mean %.1f)", mw.min() ?? 0, mw.max() ?? 0, mean(mw))])
            rows.append(["logP", String(format: "%.1f–%.1f (mean %.1f)", tenth(logP.min() ?? 0), tenth(logP.max() ?? 0), tenth(mean(logP)))])
            rows.append(["Lipinski pass", "\(descriptors.filter(\.lipinski).count)/\(descriptors.count)"])
            rows.append(["Veber pass", "\(descriptors.filter(\.veber).count)/\(descriptors.count)"])
        }
        let userKeys = Set(entries.flatMap(\.userProperties.keys)).sorted()
        if !userKeys.isEmpty { rows.append(["Properties", userKeys.joined(separator: ", ")]) }
        if let active = entries.first(where: { $0.id == vm.activeLigandEntryID }) { rows.append(["Docking ligand", active.name]) }
        let service = db.preparation
        rows.append(["Preparation", service.isRunning
                     ? "running \(service.runningMode?.rawValue ?? "") \(service.progress.current)/\(service.progress.total)"
                     : "idle"])
        rows.append(["Saved file", LigandDatabase.defaultURL.path])
        inv.emit(.output, rows.map { $0[0].padding(toLength: 16, withPad: " ", startingAt: 0) + $0[1] }.joined(separator: "\n"))
    }

    // MARK: Formatting

    /// Number without a needless ".0" (7.4, 10, 298.15).
    private static func number(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e9 ? String(Int(value)) : String(format: "%g", value)
    }

    /// "1 form", "3 forms": a count with its noun in the matching number.
    private static func count(_ n: Int, _ singular: String, _ plural: String? = nil) -> String {
        "\(n) \(n == 1 ? singular : plural ?? singular + "s")"
    }

    /// on/off as typed in DruseScript.
    private static func onOff(_ value: Bool) -> String { value ? "on" : "off" }

    /// Relative energy as "+1.23" ("0.00" for the best form).
    private static func energy(_ value: Double) -> String {
        value < 0.005 ? "0.00" : String(format: "+%.2f", value)
    }

    /// Net formal charge of the entry's 3D structure ("—" without one).
    private static func charge(_ entry: LigandEntry) -> String {
        guard !entry.atoms.isEmpty else { return "—" }
        let total = entry.atoms.reduce(0) { $0 + $1.formalCharge }
        return total > 0 ? "+\(total)" : "\(total)"
    }

    /// Shortens long SMILES for table cells.
    private static func truncated(_ text: String, _ length: Int) -> String {
        text.count > length ? String(text.prefix(length - 1)) + "…" : text
    }

    /// Elapsed time as "0.42 s", "12.3 s" or "2.5 min".
    private static func seconds(_ value: Double) -> String {
        value < 10 ? String(format: "%.2f s", value) : value < 120 ? String(format: "%.1f s", value) : String(format: "%.1f min", value / 60)
    }

    /// Elapsed `Duration` in the same format.
    private static func seconds(_ duration: Duration) -> String {
        seconds(Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18)
    }
}

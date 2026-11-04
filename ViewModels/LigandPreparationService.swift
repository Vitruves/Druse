// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Observation

// MARK: - Options

/// Controls how ionizable-site pKa values are determined during ligand preparation.
enum PKaMethod: String, CaseIterable, Sendable {
    /// GNN-predicted pKa via Metal GPU (fast + accurate, ~0.1ms per molecule).
    case gnn = "GNN"
    /// Use the built-in SMARTS lookup table (fast, ~ms per molecule).
    case table = "Table"

    var description: String {
        switch self {
        case .gnn: return "GNN-predicted pKa (Metal GPU, MAE ~1.0)"
        case .table: return "SMARTS lookup table (~230 rules)"
        }
    }

    /// Lower-case name used by DruseScript (`pka=gnn|rules`).
    var consoleName: String {
        switch self {
        case .gnn: "gnn"
        case .table: "rules"
        }
    }

    init?(consoleName: String) {
        guard let match = Self.allCases.first(where: { $0.consoleName == consoleName }) else { return nil }
        self = match
    }
}

/// The three preparation pipelines the Ligand Database offers.
enum LigandPrepMode: String, CaseIterable, Sendable {
    /// Full protomer × tautomer × conformer ensemble at the target pH; every chemical form
    /// becomes its own dockable entry inserted after the (unchanged) source entry.
    case enumerate
    /// Same ensemble pipeline limited to one protomer and one tautomer: the entry is updated
    /// in place with its most populated form at the target pH and that form's conformers.
    case dominant
    /// Plain RDKit preparation without pH handling: hydrogens, MMFF94 minimisation and
    /// Gasteiger charges (each switchable), one conformer.
    case basic

    var summary: String {
        switch self {
        case .enumerate: "protomers × tautomers × conformers at pH, one entry per form"
        case .dominant: "most populated form at pH, updated in place, with its conformers"
        case .basic: "RDKit 3D + H + MMFF94 + Gasteiger charges, no pH handling, one conformer"
        }
    }
}

/// Every tunable parameter of the ligand-preparation pipelines.
///
/// One value is shared by the Ligand Database window (Enumerate sheet) and the console
/// (`ligand prep set`), so a setting changed in one place applies to the next run in the other.
struct LigandPrepSettings: Sendable, Equatable {
    /// Target pH for protonation states and Henderson–Hasselbalch populations.
    var pH = 7.4
    var pkaMethod: PKaMethod = .gnn
    /// Sites whose pKa lies within this many units of the pH get both protonation states.
    var pkaThreshold = 2.0
    var maxTautomers = 10
    var maxProtomers = 8
    /// Forms more than this many kcal/mol above the best one are discarded.
    var energyCutoff = 10.0
    /// 3D conformers per chemical form (enumerate and dominant; basic always builds one).
    var conformersPerForm = 10
    /// Enumerated forms below this population (percent) are dropped; the dominant form is always kept.
    var minPopulationPercent = 1.0
    /// Boltzmann temperature (K) for form populations.
    var temperature = 298.15
    /// Basic pipeline switches (the ensemble pipelines always add H, minimise and charge).
    var addHydrogens = true
    var minimize = true
    var computeCharges = true

    /// Documented bounds, shared by validation and help text.
    static let pHRange: ClosedRange<Double> = 0...14
    static let pkaThresholdRange: ClosedRange<Double> = 0...10
    static let maxTautomersRange: ClosedRange<Int> = 1...100
    static let maxProtomersRange: ClosedRange<Int> = 1...64
    static let energyCutoffRange: ClosedRange<Double> = 0...100
    static let conformersRange: ClosedRange<Int> = 1...500
    static let minPopulationRange: ClosedRange<Double> = 0...100
    static let temperatureRange: ClosedRange<Double> = 100...1000
}

// MARK: - Results

/// Outcome of preparing one ligand.
struct LigandPrepItemResult: Sendable {
    let name: String
    let succeeded: Bool
    /// Entries written: 1 for dominant/basic, the number of forms for enumerate.
    let formCount: Int
    let conformerCount: Int
    let error: String?
    /// Wall-clock time spent on this ligand (off the main actor).
    let seconds: Double
}

/// Summary of one preparation run, returned by the task `LigandPreparationService.prepare` starts.
struct LigandPrepReport: Sendable {
    let mode: LigandPrepMode
    let requested: Int
    let results: [LigandPrepItemResult]
    let wasCancelled: Bool
    let seconds: Double

    var succeeded: Int { results.filter(\.succeeded).count }
    var failed: [LigandPrepItemResult] { results.filter { !$0.succeeded } }
    var entriesWritten: Int { results.reduce(0) { $0 + $1.formCount } }
    var conformers: Int { results.reduce(0) { $0 + $1.conformerCount } }
}

// MARK: - Service

/// The ligand-preparation pipeline of the Ligand Database, shared by its window and the console.
///
/// Runs RDKit/C++ preparation for many entries concurrently (bounded by the app's maximum
/// concurrency setting), writes the results back into the database on the main actor, and
/// exposes observable progress. Exactly one run is active at a time: starting a run cancels
/// the previous one. Each run is a `Task` the caller can await for a `LigandPrepReport`.
@MainActor
@Observable
final class LigandPreparationService {

    /// Session defaults used when a run does not pass its own settings.
    var settings = LigandPrepSettings()
    /// Pipeline used by console commands that do not name one (`ligand prep set mode=…`);
    /// the window's buttons always pick their mode explicitly.
    var defaultMode: LigandPrepMode = .enumerate

    private(set) var isRunning = false
    /// Mode of the active run (nil when idle).
    private(set) var runningMode: LigandPrepMode?
    private(set) var progress: (current: Int, total: Int) = (0, 0)
    /// "Preparing 3/40: aspirin" — the last finished ligand of the active run.
    private(set) var message = ""

    @ObservationIgnored private unowned let database: LigandDatabase
    @ObservationIgnored private var runTask: Task<LigandPrepReport, Never>?
    /// Incremented per run so a cancelled run that is still draining cannot touch the state of a newer one.
    @ObservationIgnored private var generation = 0

    init(database: LigandDatabase) {
        self.database = database
    }

    /// Prepares `entries` with `mode`, cancelling any run in progress.
    ///
    /// Entries without SMILES and without a name are skipped. `onItem` is called on the main
    /// actor after each ligand's result has been written to the database, with the number of
    /// ligands finished so far (console progress lines).
    @discardableResult
    func prepare(
        _ entries: [LigandEntry],
        mode: LigandPrepMode,
        settings override: LigandPrepSettings? = nil,
        onItem: (@MainActor @Sendable (LigandPrepItemResult, Int) -> Void)? = nil
    ) -> Task<LigandPrepReport, Never> {
        runTask?.cancel()
        generation += 1
        let runGeneration = generation
        let toProcess = entries.filter { !$0.originalSMILES.isEmpty || !$0.name.isEmpty }
        let runSettings = override ?? settings

        isRunning = !toProcess.isEmpty
        runningMode = toProcess.isEmpty ? nil : mode
        progress = (0, toProcess.count)
        message = mode == .basic && toProcess.count == 1 ? "Preparing \(toProcess[0].name)..." : ""

        let task = Task { [weak self] () -> LigandPrepReport in
            guard let self else {
                return LigandPrepReport(mode: mode, requested: toProcess.count, results: [], wasCancelled: true, seconds: 0)
            }
            let report = await self.run(toProcess, mode: mode, settings: runSettings,
                                        generation: runGeneration, onItem: onItem)
            if self.generation == runGeneration {
                self.finish()
                self.logSummary(report)
            }
            return report
        }
        runTask = task
        return task
    }

    /// Stops the active run; ligands already prepared keep their results.
    func cancel() {
        guard isRunning else { return }
        runTask?.cancel()
        runTask = nil
        generation += 1
        finish()
        ActivityLog.shared.info("Cancelled", category: .molecule)
    }

    /// Returns the observable state to idle.
    private func finish() {
        isRunning = false
        runningMode = nil
        progress = (0, 0)
        message = ""
        runTask = nil
    }

    // MARK: Running

    /// Raw per-ligand output of the off-main-actor work, applied to the database on the main actor.
    private enum Computation: Sendable {
        case basic(MoleculeData, LigandDescriptors?, swappedNameAndSMILES: Bool)
        /// Generated forms (empty on failure, with `error` set), most stable first.
        case ensemble([ChemicalForm], error: String, pka: [PKaGNNPredictor.SitePrediction])
        case failed(String)
    }

    /// One ligand's computation with the entry it was computed for and the time it took.
    private struct Outcome: Sendable {
        let entry: LigandEntry
        let computation: Computation
        let seconds: Double
    }

    /// Prepares every entry with bounded parallelism, writing each result as it arrives.
    private func run(
        _ entries: [LigandEntry],
        mode: LigandPrepMode,
        settings: LigandPrepSettings,
        generation runGeneration: Int,
        onItem: (@MainActor @Sendable (LigandPrepItemResult, Int) -> Void)?
    ) async -> LigandPrepReport {
        let clock = ContinuousClock()
        let start = clock.now
        var results: [LigandPrepItemResult] = []
        let verb = mode == .enumerate ? "Enumerating" : "Preparing"
        let maxConcurrency = max(AppSettings.shared.maxConcurrentTasks, 1)

        func handle(_ outcome: Outcome) {
            let result = apply(outcome, mode: mode, settings: settings)
            results.append(result)
            if generation == runGeneration {
                progress = (results.count, entries.count)
                message = "\(verb) \(results.count)/\(entries.count): \(outcome.entry.name)"
            }
            onItem?(result, results.count)
        }

        await withTaskGroup(of: Outcome?.self) { group in
            var enqueued = 0
            for entry in entries {
                if enqueued >= maxConcurrency, let item = await group.next(), let outcome = item {
                    handle(outcome)
                }
                guard !Task.isCancelled else { break }
                group.addTask {
                    guard !Task.isCancelled else { return nil }
                    return await Task.detached(priority: .userInitiated) {
                        Self.compute(entry, mode: mode, settings: settings)
                    }.value
                }
                enqueued += 1
            }
            for await item in group {
                guard !Task.isCancelled else { break }
                guard let outcome = item else { continue }
                handle(outcome)
            }
        }

        let elapsed = clock.now - start
        return LigandPrepReport(
            mode: mode, requested: entries.count, results: results,
            wasCancelled: Task.isCancelled || results.count < entries.count,
            seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        )
    }

    /// The RDKit / C++ / pKa-GNN work for one ligand. Runs off the main actor.
    private nonisolated static func compute(_ entry: LigandEntry, mode: LigandPrepMode,
                                            settings: LigandPrepSettings) -> Outcome {
        let clock = ContinuousClock()
        let start = clock.now
        let computation: Computation
        switch mode {
        case .basic:
            computation = computeBasic(entry, settings: settings)
        case .dominant, .enumerate:
            let smiles = entry.originalSMILES.isEmpty ? entry.name : entry.originalSMILES
            let predictions = settings.pkaMethod == .gnn ? PKaGNNPredictor.predict(smiles: smiles) : []
            let isDominant = mode == .dominant
            let minPopulation = settings.minPopulationPercent / 100
            // With no population floor every site is enumerated regardless of its pKa.
            let pkaThreshold = !isDominant && minPopulation <= 0 ? 100.0 : settings.pkaThreshold
            let result = RDKitBridge.prepareEnsembleWithSites(
                smiles: smiles, name: entry.name,
                pH: settings.pH, pkaThreshold: pkaThreshold,
                maxTautomers: isDominant ? 1 : settings.maxTautomers,
                maxProtomers: isDominant ? 1 : settings.maxProtomers,
                energyCutoff: settings.energyCutoff, conformersPerForm: settings.conformersPerForm,
                temperature: settings.temperature,
                sites: predictions
            )
            let forms = result.success ? RDKitBridge.ensembleResultToForms(result) : []
            let template = experimentalTemplate(of: entry)
            computation = .ensemble(
                forms.map { form in
                    // Ensemble members are always minimised and Gasteiger-charged.
                    guard let template, let placed = onTemplate(template, atoms: form.atoms, bonds: form.bonds, name: entry.name,
                                                                minimize: true, computeCharges: true) else { return form }
                    var updated = form
                    updated.conformers = [Conformer3D(id: 0, atoms: placed.molecule.atoms,
                                                      bonds: placed.molecule.bonds, energy: placed.energy)]
                    return updated
                },
                error: result.errorMessage, pka: predictions
            )
        }
        let elapsed = clock.now - start
        return Outcome(entry: entry, computation: computation,
                       seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
    }

    /// Basic RDKit preparation: one embedded, minimised conformer (the result keeps a single 3D
    /// structure, so `conformersPerForm` does not apply). Imports sometimes swap the name and SMILES
    /// columns, so when the SMILES fails the name is tried as SMILES.
    private nonisolated static func computeBasic(_ entry: LigandEntry, settings: LigandPrepSettings) -> Computation {
        let candidates: [(smiles: String, name: String)] = entry.smiles.isEmpty
            ? [(entry.name, "Ligand")]
            : [(entry.smiles, entry.name), (entry.name, entry.smiles)]
        var lastError = "invalid SMILES"
        for candidate in candidates {
            let (molecule, descriptors, _, error) = RDKitBridge.prepareLigand(
                smiles: candidate.smiles, name: candidate.name, numConformers: 1,
                addHydrogens: settings.addHydrogens, minimize: settings.minimize,
                computeCharges: settings.computeCharges
            )
            if let molecule {
                let swapped = candidate.smiles == entry.name && !entry.smiles.isEmpty
                let placed = experimentalTemplate(of: entry).flatMap {
                    onTemplate($0, atoms: molecule.atoms, bonds: molecule.bonds, name: entry.name,
                               minimize: settings.minimize, computeCharges: settings.computeCharges)
                }
                return .basic(placed?.molecule ?? molecule, descriptors, swappedNameAndSMILES: swapped)
            }
            if let error, !error.isEmpty { lastError = error }
        }
        return .failed(lastError)
    }

    /// The experimental 3D structure of the entry (a co-crystallized ligand, or unprepared 3D
    /// input coordinates), if it has one. 2D input (all z equal) is not a template.
    private nonisolated static func experimentalTemplate(of entry: LigandEntry) -> (atoms: [Atom], bonds: [Bond])? {
        guard entry.sourceChainID != nil || !entry.isPrepared,
              let z = entry.atoms.first?.position.z,
              entry.atoms.contains(where: { abs($0.position.z - z) > 1e-3 }) else { return nil }
        return (entry.atoms, entry.bonds)
    }

    /// A prepared structure moved onto the experimental heavy-atom coordinates, with hydrogens
    /// re-placed and relaxed. Nil when its heavy-atom graph differs from the template (e.g. a
    /// tautomer), in which case the SMILES embedding is kept.
    private nonisolated static func onTemplate(
        _ template: (atoms: [Atom], bonds: [Bond]), atoms: [Atom], bonds: [Bond], name: String,
        minimize: Bool, computeCharges: Bool
    ) -> (molecule: MoleculeData, energy: Double)? {
        let result = RDKitBridge.prepareOnTemplate(
            template: template, atoms: atoms, bonds: bonds, name: name,
            minimize: minimize, computeCharges: computeCharges
        )
        return result.molecule.map { ($0, result.energy) }
    }

    // MARK: Writing results

    /// Writes one ligand's result into the database (failures are logged and, for dominant mode, recorded on the entry).
    private func apply(_ outcome: Outcome, mode: LigandPrepMode, settings: LigandPrepSettings) -> LigandPrepItemResult {
        // Work on the current copy: the entry may have been renamed or edited while it was prepared.
        let entry = database.entries.first { $0.id == outcome.entry.id } ?? outcome.entry
        func result(_ ok: Bool, forms: Int = 0, conformers: Int = 0, error: String? = nil) -> LigandPrepItemResult {
            LigandPrepItemResult(name: entry.name, succeeded: ok, formCount: forms,
                                 conformerCount: conformers, error: error, seconds: outcome.seconds)
        }

        switch outcome.computation {
        case .failed(let error):
            ActivityLog.shared.error("Failed to prepare \(entry.name): \(error)", category: .molecule)
            return result(false, error: error)

        case .basic(let molecule, let descriptors, let swapped):
            var updated = entry
            if swapped {
                updated.smiles = entry.name
                updated.name = entry.smiles
            }
            updated.atoms = molecule.atoms
            updated.bonds = molecule.bonds
            updated.descriptors = descriptors
            updated.isPrepared = true
            updated.conformerCount = 1
            updated.conformers = []
            database.update(updated)
            ActivityLog.shared.success("Prepared \(updated.name)", category: .molecule)
            return result(true, forms: 1, conformers: 1)

        case .ensemble(let forms, let ensembleError, let pka):
            if mode == .enumerate, !pka.isEmpty {
                let pkaLog = pka.map { p in
                    "\(p.isAcid ? "acid" : "base")[\(p.atomIdx)]: \(String(format: "%.1f", p.pKa)) (p=\(String(format: "%.2f", p.ionizableProbability)))"
                }.joined(separator: ", ")
                ActivityLog.shared.info("pKa(\(entry.name)): \(pkaLog)", category: .molecule)
            }
            guard !forms.isEmpty else {
                let error = ensembleError.isEmpty ? "no forms generated" : ensembleError
                if mode == .dominant {
                    var failed = entry
                    failed.preparationError = error
                    database.update(failed)
                    ActivityLog.shared.error("Failed to prepare \(entry.name): \(error)", category: .molecule)
                } else {
                    ActivityLog.shared.error("Failed to enumerate \(entry.name): \(error)", category: .molecule)
                }
                return result(false, error: error)
            }
            return mode == .dominant
                ? applyDominant(forms, to: entry, result: result)
                : applyEnumeration(forms, of: entry, minPopulation: settings.minPopulationPercent / 100, result: result)
        }
    }

    /// Updates the entry in place with its most populated form.
    private func applyDominant(
        _ forms: [ChemicalForm], to entry: LigandEntry,
        result: (Bool, Int, Int, String?) -> LigandPrepItemResult
    ) -> LigandPrepItemResult {
        guard let best = forms.max(by: { $0.boltzmannWeight < $1.boltzmannWeight }) else {
            return result(false, 0, 0, "no forms generated")
        }
        var updated = entry
        updated.originalSMILES = best.smiles
        updated.atoms = best.atoms
        updated.bonds = best.bonds
        updated.isPrepared = true
        updated.preparationDate = Date()
        updated.conformerCount = best.conformerCount
        updated.conformers = best.conformers
        updated.preparationError = nil
        // Drug-likeness descriptors (Crippen logP, HBD/HBA, Ro5, Veber) are defined on the
        // neutral parent. The dominant form at pH is often ionized, which would shift logP by
        // several units and inflate HBD — keep the parent's descriptors and only compute them
        // (from the input SMILES) when the entry has none.
        if updated.descriptors == nil {
            updated.descriptors = RDKitBridge.computeDescriptors(smiles: entry.originalSMILES)
        }
        database.update(updated)
        return result(true, 1, best.conformerCount, nil)
    }

    /// Inserts one entry per form after the source entry (which is kept), replacing the forms of
    /// any earlier enumeration. Forms below `minPopulation` are dropped (the dominant one is kept)
    /// and the remaining populations renormalised.
    private func applyEnumeration(
        _ allForms: [ChemicalForm], of entry: LigandEntry, minPopulation: Double,
        result: (Bool, Int, Int, String?) -> LigandPrepItemResult
    ) -> LigandPrepItemResult {
        var forms = allForms
        guard !forms.isEmpty else { return result(false, 0, 0, "no forms generated") }
        if minPopulation > 0 && forms.count > 1 {
            forms = forms.enumerated().filter { index, form in
                index == 0 || form.boltzmannWeight >= minPopulation
            }.map(\.element)
            let weightSum = forms.reduce(0.0) { $0 + $1.boltzmannWeight }
            if weightSum > 0 {
                for i in forms.indices { forms[i].boltzmannWeight /= weightSum }
            }
        }

        let newEntries = forms.map { form -> LigandEntry in
            var formEntry = LigandEntry(
                name: form.kind == .parent ? entry.name : "\(entry.name)_\(form.label)",
                smiles: form.smiles, atoms: form.atoms, bonds: form.bonds,
                isPrepared: true, conformerCount: form.conformerCount
            )
            formEntry.preparationDate = Date()
            formEntry.parentName = entry.name
            formEntry.populationWeight = form.boltzmannWeight
            formEntry.formKind = form.kind
            formEntry.relativeEnergy = form.relativeEnergy
            formEntry.conformers = form.conformers
            formEntry.descriptors = RDKitBridge.computeDescriptors(smiles: form.smiles)
            formEntry.ki = entry.ki
            formEntry.pKi = entry.pKi
            formEntry.ic50 = entry.ic50
            return formEntry
        }

        database.batchMutate { entries in
            entries.removeAll { $0.parentName == entry.name }
            if let index = entries.firstIndex(where: { $0.id == entry.id }) {
                entries.insert(contentsOf: newEntries, at: index + 1)
            } else {
                entries.append(contentsOf: newEntries)
            }
        }
        return result(true, newEntries.count, newEntries.reduce(0) { $0 + $1.conformerCount }, nil)
    }

    /// Activity-log line for a finished (not cancelled) run, as the window always logged.
    private func logSummary(_ report: LigandPrepReport) {
        guard !report.wasCancelled else { return }
        switch report.mode {
        case .basic:
            break
        case .dominant:
            ActivityLog.shared.success("Prepared \(report.succeeded)/\(report.requested) molecules", category: .molecule)
        case .enumerate:
            ActivityLog.shared.success(
                "Enumerate: \(report.requested) molecules → \(report.entriesWritten) new entries, \(report.conformers) conformers",
                category: .molecule
            )
        }
    }
}

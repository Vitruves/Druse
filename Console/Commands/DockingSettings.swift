// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - Docking settings registry
//
// Every docking parameter the console can change is declared once here: its option name,
// section, one-line summary, value type, valid range and where it lives in
// `DockingCoordinator`. `dock set`, the inline options of `dock`, `dock config` and
// `dock config reset` are all generated from this table, so a parameter added here is
// settable, validated, documented and completable everywhere at once.
//
// Fields of `DockingConfig` that no engine code reads (clash overlap/scale, ensemble
// multi-start, flex refinement steps, diffusion parallel poses, fragment anchor local-search
// steps, GFN2 freezeProtein, useAFv4Rescore, FlexibleResidueConfig.rotamerResolution) are
// deliberately absent: a setting that changes nothing would mislead.

/// Group of related docking settings, in the order `dock config` prints them.
enum DockingSettingSection: String, CaseIterable {
    case method, budget, grid, operators, local, exploration, torsions, rerank, strain
    case fragment, remc, diffusion, gfn2, receptor

    /// Heading printed by `dock config`.
    var title: String {
        switch self {
        case .method: "Method"
        case .budget: "Search budget"
        case .grid: "Grid maps"
        case .operators: "Search operators (GA / Monte Carlo)"
        case .local: "Local search"
        case .exploration: "Exploration phase"
        case .torsions: "Ligand torsions"
        case .rerank: "Explicit rerank of top clusters"
        case .strain: "MMFF94 ligand strain penalty"
        case .fragment: "Fragment-based search (search=fragment)"
        case .remc: "Replica exchange (search=remc)"
        case .diffusion: "Diffusion-guided search (search=diffusion)"
        case .gfn2: "GFN2-xTB post-docking refinement"
        case .receptor: "Receptor flexibility"
        }
    }
}

/// One console-settable docking parameter.
struct DockingSetting {
    /// Option name, e.g. `population` or `fragment-beam`.
    let key: String
    let section: DockingSettingSection
    /// One-line summary; the valid range is appended by `parameter`.
    let summary: String
    let type: ConsoleValueType
    /// Human-readable range or unit appended to the summary ("8–4096", "Å").
    let rangeText: String
    /// Current value, spelled as it would be typed after `key=`.
    let read: (DockingCoordinator) -> String
    /// Validates a bound value and stores it; may adjust dependent fields.
    let write: (inout DockingCoordinator, ConsoleValue) throws -> Void
    /// Restores this setting's stored field(s) from a default-constructed coordinator.
    let restore: (inout DockingCoordinator, DockingCoordinator) -> Void

    /// Declaration used for `key=value` options, help and Tab completion.
    var parameter: ConsoleParameter {
        ConsoleParameter(name: key, type: type, summary: rangeText.isEmpty ? summary : "\(summary) (\(rangeText))")
    }
}

/// The table of docking settings and the operations generated from it.
@MainActor
enum DockingSettings {

    /// Values of a fresh session; `dock config` marks settings that differ from these.
    static var defaults: DockingCoordinator { DockingCoordinator() }

    static func setting(_ key: String) -> DockingSetting? {
        all.first { $0.key == key }
    }

    static func settings(in section: DockingSettingSection) -> [DockingSetting] {
        all.filter { $0.section == section }
    }

    /// Applies every bound setting option of `inv` (in registry order) to a draft of the
    /// docking state; the live state changes only if all of them validate.
    /// Returns the keys that were set.
    @discardableResult
    static func apply(_ inv: ConsoleInvocation, keys: [String]? = nil) throws -> [String] {
        let vm = inv.viewModel
        var draft = vm.docking
        var applied: [String] = []
        for setting in all where keys?.contains(setting.key) ?? true {
            guard let value = inv.values[setting.key] else { continue }
            try setting.write(&draft, value)
            applied.append(setting.key)
        }
        guard !applied.isEmpty else { return [] }
        commit(draft, to: vm)
        return applied
    }

    /// Restores every setting (or one section) to its default.
    static func reset(_ vm: AppViewModel, section: DockingSettingSection?) {
        var draft = vm.docking
        let fresh = defaults
        for setting in all where section == nil || setting.section == section {
            setting.restore(&draft, fresh)
        }
        commit(draft, to: vm)
    }

    /// Copies only the setting-backed fields back, so docking/run state is never overwritten.
    private static func commit(_ draft: DockingCoordinator, to vm: AppViewModel) {
        vm.docking.dockingConfig = draft.dockingConfig
        vm.docking.scoringMethod = draft.scoringMethod
        vm.docking.chargeMethod = draft.chargeMethod
        vm.docking.flexibleResidueConfig.autoFlex = draft.flexibleResidueConfig.autoFlex
        if draft.flexibleResidueConfig.autoFlex {
            vm.docking.flexibleResidueConfig.flexibleResidueIndices = draft.flexibleResidueConfig.flexibleResidueIndices
        }
    }

    /// `dock config`: every setting of the selected sections, non-defaults starred.
    static func report(_ vm: AppViewModel, section: DockingSettingSection?, changedOnly: Bool) -> String {
        let current = vm.docking
        let fresh = defaults
        let width = all.map(\.key.count).max() ?? 0
        var lines: [String] = []
        let config = current.dockingConfig
        let preset = DockingPreset.allCases.first { $0.isActive(in: config) }?.rawValue.lowercased() ?? "custom"
        if section == nil {
            lines.append("Budget preset: \(preset)   (* = changed from default; 'dock set key=value' changes a setting)")
        }
        for group in DockingSettingSection.allCases where section == nil || group == section {
            let rows = settings(in: group).compactMap { setting -> String? in
                let value = setting.read(current)
                let standard = setting.read(fresh)
                let changed = value != standard
                if changedOnly && !changed { return nil }
                let name = setting.key.padding(toLength: width + 2, withPad: " ", startingAt: 0)
                let marker = changed ? "* " : "  "
                let suffix = changed ? "   (default \(standard))" : ""
                return "  \(marker)\(name)\(value)\(suffix)"
            }
            guard !rows.isEmpty else { continue }
            lines.append("")
            lines.append(group.title + ":")
            lines.append(contentsOf: rows)
        }
        if changedOnly && lines.filter({ $0.hasPrefix("  ") }).isEmpty {
            lines.append("All docking settings are at their defaults")
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .newlines)
    }

    // MARK: Table

    static let all: [DockingSetting] = [
        // Method
        choice("search", .method, "Search algorithm", \.dockingConfig.searchMethod, ConsoleNames.searchMethods),
        choice("scoring", .method, "Scoring function used during search and for ranking", \.scoringMethod,
               ConsoleNames.scoringMethods) { draft in
            if draft.scoringMethod == .pignet2, Bundle.main.url(forResource: "PIGNet2", withExtension: "weights") == nil {
                throw ConsoleError("PIGNet2 weights are not bundled with this build — use vina, drusina or druseaf")
            }
        },
        choice("charges", .method, "Partial-charge model for the receptor and ligand", \.chargeMethod, ConsoleNames.chargeMethods),
        bool("auto-tune", .method, "Adapt population, generations, runs, grid and local search to the system at launch",
             \.dockingConfig.autoMode),

        // Budget
        int("population", .budget, "Poses per population", \.dockingConfig.populationSize, 8...4096, then: manualBudget),
        int("generations", .budget, "Generations (Monte Carlo steps) per run", \.dockingConfig.generationsPerRun, 1...10_000,
            then: manualBudget),
        int("runs", .budget, "Independent runs (restarts)", \.dockingConfig.numRuns, 1...100, then: manualBudget),
        int("live-update", .budget, "Refresh the live pose every N generations (higher = fewer GPU syncs)",
            \.dockingConfig.liveUpdateFrequency, 1...10_000),

        // Grid
        float("grid-spacing", .grid, "Affinity-grid spacing; coarsened automatically when the grid exceeds GPU memory",
              \.dockingConfig.gridSpacing, 0.2...1.0, unit: "Å"),

        // Operators
        float("mutation-rate", .operators, "Per-gene mutation probability after the exploration phase",
              \.dockingConfig.mutationRate, 0...1),
        float("crossover-rate", .operators, "Crossover probability between parent poses", \.dockingConfig.crossoverRate, 0...1),
        float("translation-step", .operators, "Maximum translation of a mutation after exploration",
              \.dockingConfig.translationStep, 0.05...20, unit: "Å"),
        float("rotation-step", .operators, "Maximum rotation of a mutation after exploration",
              \.dockingConfig.rotationStep, 0.01...3.1416, unit: "rad"),
        float("torsion-step", .operators, "Maximum torsion change of a mutation",
              \.dockingConfig.torsionStep, 0.01...3.1416, unit: "rad"),
        float("mc-temperature", .operators, "Metropolis temperature of the Monte Carlo acceptance test",
              \.dockingConfig.mcTemperature, 0.05...10, unit: "kcal/mol"),

        // Local search
        int("ls-frequency", .local, "Run gradient local search every N generations after exploration",
            \.dockingConfig.localSearchFrequency, 1...100),
        int("ls-steps", .local, "Gradient-descent steps per local search", \.dockingConfig.localSearchSteps, 0...500),
        bool("analytical-gradients", .local, "Analytical gradients (~28× fewer evaluations) instead of finite differences",
             \.dockingConfig.useAnalyticalGradients),

        // Exploration
        float("explore-ratio", .exploration, "Fraction of generations spent in the broad exploration phase",
              \.dockingConfig.explorationPhaseRatio, 0...1),
        float("explore-translation", .exploration, "Maximum translation of a mutation during exploration",
              \.dockingConfig.explorationTranslationStep, 0.05...30, unit: "Å"),
        float("explore-rotation", .exploration, "Maximum rotation of a mutation during exploration",
              \.dockingConfig.explorationRotationStep, 0.01...3.1416, unit: "rad"),
        float("explore-mutation", .exploration, "Mutation probability during exploration",
              \.dockingConfig.explorationMutationRate, 0...1),
        int("explore-ls-frequency", .exploration, "Run local search every N generations during exploration",
            \.dockingConfig.explorationLocalSearchFrequency, 1...100),

        // Torsions
        bool("ligand-flex", .torsions, "Sample ligand rotatable bonds (off = rigid input conformer)",
             \.dockingConfig.enableFlexibility),
        DockingSetting(
            key: "torsion-preset", section: .torsions,
            summary: "Torsion-sampling policy; balanced, redocking and exploratory also set the five torsion-* values",
            type: .choice(DockingNames.names(DockingNames.torsionPresets)), rangeText: "",
            read: { DockingNames.name(of: $0.dockingConfig.torsionSearchPreset, in: DockingNames.torsionPresets) },
            write: { draft, value in
                guard case .text(let word) = value,
                      let preset = ConsoleNames.lookup(word, in: DockingNames.torsionPresets) else { return }
                draft.dockingConfig.applyTorsionSearchPreset(preset)
            },
            restore: { draft, fresh in
                draft.dockingConfig.applyTorsionSearchPreset(fresh.dockingConfig.torsionSearchPreset)
            }
        ),
        float("torsion-exact", .torsions, "Fraction of torsion genes kept at the input value",
              \.dockingConfig.torsionExactFraction, 0...1, then: manualTorsions),
        float("torsion-local", .torsions, "Fraction of torsion genes perturbed locally around the input value",
              \.dockingConfig.torsionLocalFraction, 0...1, then: manualTorsions),
        float("torsion-width", .torsions, "Amplitude of local torsion perturbations",
              \.dockingConfig.torsionLocalAmplitude, 0...3.1416, unit: "rad", then: manualTorsions),
        float("torsion-reset", .torsions, "Probability of resetting a torsion to a random angle",
              \.dockingConfig.torsionRandomResetProbability, 0...1, then: manualTorsions),
        float("torsion-scale", .torsions, "Scale applied to torsion mutation steps",
              \.dockingConfig.torsionPerturbationScale, 0...3, then: manualTorsions),

        // Rerank
        int("rerank-clusters", .rerank, "Top cluster representatives rescored against explicit receptor atoms (0 = off)",
            \.dockingConfig.explicitRerankTopClusters, 0...100),
        int("rerank-variants", .rerank, "Seeded variants refined around each reranked representative",
            \.dockingConfig.explicitRerankVariantsPerCluster, 1...32),
        int("rerank-steps", .rerank, "Local-search steps of the rerank refinement",
            \.dockingConfig.explicitRerankLocalSearchSteps, 0...200),

        // Strain
        bool("strain", .strain, "Add an MMFF94 strain penalty after docking (not applied with scoring=vina)",
             \.dockingConfig.strainPenaltyEnabled),
        float("strain-threshold", .strain, "Strain tolerated before penalizing (docked − relaxed MMFF94 energy)",
              \.dockingConfig.strainPenaltyThreshold, 0...100, unit: "kcal/mol"),
        float("strain-weight", .strain, "Penalty = weight × (strain − threshold)",
              \.dockingConfig.strainPenaltyWeight, 0...10),

        // Fragment
        int("fragment-beam", .fragment, "Partial poses kept at each fragment growth step",
            \.dockingConfig.fragment.beamWidth, 1...1024),
        int("fragment-anchors", .fragment, "Anchor placements sampled in the site",
            \.dockingConfig.fragment.anchorSamplingCount, 1...8192),
        float("fragment-prune", .fragment, "Discard partial poses this far above the best one",
              \.dockingConfig.fragment.growthPruneThreshold, 0...100, unit: "kcal/mol"),
        int("fragment-torsions", .fragment, "Torsion angles sampled per connecting bond during growth",
            \.dockingConfig.fragment.torsionSamples, 1...360),
        DockingSetting(
            key: "fragment-scaffold", section: .fragment,
            summary: "SMARTS of the scaffold fragment growth must keep as anchor; 'auto' uses the largest rigid fragment",
            type: .string, rangeText: "",
            read: { $0.dockingConfig.fragment.scaffoldSMARTS ?? "auto" },
            write: { draft, value in
                guard case .text(let text) = value else { return }
                let trimmed = text.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty || trimmed.lowercased() == "auto" {
                    draft.dockingConfig.fragment.scaffoldSMARTS = nil
                    draft.dockingConfig.fragment.scaffoldMode = .auto
                } else {
                    draft.dockingConfig.fragment.scaffoldSMARTS = trimmed
                    draft.dockingConfig.fragment.scaffoldMode = .manual
                }
            },
            restore: { draft, fresh in
                draft.dockingConfig.fragment.scaffoldSMARTS = fresh.dockingConfig.fragment.scaffoldSMARTS
                draft.dockingConfig.fragment.scaffoldMode = fresh.dockingConfig.fragment.scaffoldMode
            }
        ),

        // Replica exchange
        int("remc-replicas", .remc, "Temperature replicas", \.dockingConfig.replicaExchange.numReplicas, 2...64),
        float("remc-tmin", .remc, "Lowest replica temperature", \.dockingConfig.replicaExchange.minTemperature,
              0.05...50, unit: "kcal/mol", then: temperatureOrder),
        float("remc-tmax", .remc, "Highest replica temperature", \.dockingConfig.replicaExchange.maxTemperature,
              0.05...50, unit: "kcal/mol", then: temperatureOrder),
        int("remc-swap", .remc, "Attempt replica swaps every N steps", \.dockingConfig.replicaExchange.swapInterval, 1...1000),
        int("remc-steps", .remc, "Monte Carlo steps per replica", \.dockingConfig.replicaExchange.stepsPerReplica, 1...100_000),

        // Diffusion
        int("diffusion-steps", .diffusion, "Reverse-diffusion denoising steps", \.dockingConfig.diffusion.numDenoisingSteps, 1...1000),
        choice("diffusion-schedule", .diffusion, "Noise variance schedule", \.dockingConfig.diffusion.noiseSchedule,
               DockingNames.noiseSchedules),
        float("diffusion-guidance", .diffusion, "Scale of the DruseAF attention-gradient guidance",
              \.dockingConfig.diffusion.guidanceScale, 0...10),
        int("diffusion-refine", .diffusion, "Vina gradient local-search steps after denoising",
            \.dockingConfig.diffusion.refinementSteps, 0...1000),

        // GFN2
        bool("gfn2", .gfn2, "Refine the top poses with GFN2-xTB after every docking run", \.dockingConfig.gfn2Refinement.enabled),
        int("gfn2-top", .gfn2, "Top poses optimized", \.dockingConfig.gfn2Refinement.topPosesToRefine, 1...500),
        choice("gfn2-solvation", .gfn2, "Implicit solvation (none = gas phase)", \.dockingConfig.gfn2Refinement.solvation,
               DockingNames.gfn2Solvation),
        choice("gfn2-level", .gfn2, "Optimization convergence level", \.dockingConfig.gfn2Refinement.optLevel,
               DockingNames.gfn2Levels),
        int("gfn2-steps", .gfn2, "Maximum optimization steps (0 = automatic)", \.dockingConfig.gfn2Refinement.maxSteps, 0...10_000),
        float("gfn2-blend", .gfn2, "Weight of the relative GFN2 energy added to the ranking score",
              \.dockingConfig.gfn2Refinement.blendWeight, 0...1),
        bool("gfn2-update", .gfn2, "Replace pose coordinates by the optimized geometry (within gfn2-max-rmsd)",
             \.dockingConfig.gfn2Refinement.updateCoordinates),
        float("gfn2-restraint", .gfn2, "Harmonic restraint keeping atoms near the docked pose (0 = none)",
              \.dockingConfig.gfn2Refinement.restraintStrength, 0...1, unit: "Hartree/Å²"),
        float("gfn2-max-rmsd", .gfn2, "Reject optimized geometries that moved more than this",
              \.dockingConfig.gfn2Refinement.maxRMSD, 0...10, unit: "Å"),

        // Receptor
        DockingSetting(
            key: "flex-auto", section: .receptor,
            summary: "Soft flexibility of up to 6 pocket-lining sidechains chosen at launch (replaces manual 'flex add' residues)",
            type: .bool, rangeText: "",
            read: { $0.flexibleResidueConfig.autoFlex ? "on" : "off" },
            write: { draft, value in
                guard case .bool(let on) = value else { return }
                draft.flexibleResidueConfig.autoFlex = on
                if on { draft.flexibleResidueConfig.flexibleResidueIndices.removeAll() }
            },
            restore: { draft, fresh in draft.flexibleResidueConfig.autoFlex = fresh.flexibleResidueConfig.autoFlex }
        )
    ]

    // MARK: Dependent-field rules

    /// An explicit budget turns automatic tuning off (as the Docking step's sliders do).
    nonisolated private static func manualBudget(_ draft: inout DockingCoordinator) {
        draft.dockingConfig.autoMode = false
    }

    /// Editing a torsion value switches the torsion preset to manual (as the Docking step's sliders do).
    nonisolated private static func manualTorsions(_ draft: inout DockingCoordinator) {
        draft.dockingConfig.torsionSearchPreset = .manual
    }

    /// Rejects a replica-exchange temperature ladder whose minimum is not below its maximum.
    nonisolated private static func temperatureOrder(_ draft: inout DockingCoordinator) throws {
        let rex = draft.dockingConfig.replicaExchange
        guard rex.minTemperature < rex.maxTemperature else {
            throw ConsoleError(String(format: "remc-tmin (%.2f) must be below remc-tmax (%.2f)", rex.minTemperature, rex.maxTemperature))
        }
    }

    // MARK: Builders

    /// Declares an integer docking setting bound to a coordinator key path, with its range and an optional follow-up rule.
    private static func int<T: BinaryInteger>(
        _ key: String, _ section: DockingSettingSection, _ summary: String,
        _ path: WritableKeyPath<DockingCoordinator, T>, _ range: ClosedRange<Int>,
        then: ((inout DockingCoordinator) throws -> Void)? = nil
    ) -> DockingSetting {
        DockingSetting(
            key: key, section: section, summary: summary, type: .int,
            rangeText: "\(range.lowerBound)–\(range.upperBound)",
            read: { "\($0[keyPath: path])" },
            write: { draft, value in
                guard case .int(let number) = value else { return }
                guard range.contains(number) else {
                    throw ConsoleError("\(key) must be between \(range.lowerBound) and \(range.upperBound), got \(number)")
                }
                draft[keyPath: path] = T(number)
                try then?(&draft)
            },
            restore: { draft, fresh in draft[keyPath: path] = fresh[keyPath: path] }
        )
    }

    /// Declares a numeric docking setting bound to a coordinator key path, with its range and an optional follow-up rule.
    private static func float(
        _ key: String, _ section: DockingSettingSection, _ summary: String,
        _ path: WritableKeyPath<DockingCoordinator, Float>, _ range: ClosedRange<Float>,
        unit: String = "", then: ((inout DockingCoordinator) throws -> Void)? = nil
    ) -> DockingSetting {
        let bounds = "\(format(range.lowerBound))–\(format(range.upperBound))"
        return DockingSetting(
            key: key, section: section, summary: summary, type: .float,
            rangeText: unit.isEmpty ? bounds : "\(bounds) \(unit)",
            read: { format($0[keyPath: path]) },
            write: { draft, value in
                guard case .float(let number) = value else { return }
                let x = Float(number)
                guard range.contains(x) else {
                    throw ConsoleError("\(key) must be between \(bounds)\(unit.isEmpty ? "" : " " + unit), got \(format(x))")
                }
                draft[keyPath: path] = x
                try then?(&draft)
            },
            restore: { draft, fresh in draft[keyPath: path] = fresh[keyPath: path] }
        )
    }

    /// Declares an on/off docking setting bound to a coordinator key path.
    private static func bool(
        _ key: String, _ section: DockingSettingSection, _ summary: String,
        _ path: WritableKeyPath<DockingCoordinator, Bool>
    ) -> DockingSetting {
        DockingSetting(
            key: key, section: section, summary: summary, type: .bool, rangeText: "",
            read: { $0[keyPath: path] ? "on" : "off" },
            write: { draft, value in
                guard case .bool(let on) = value else { return }
                draft[keyPath: path] = on
            },
            restore: { draft, fresh in draft[keyPath: path] = fresh[keyPath: path] }
        )
    }

    /// Declares an enumerated docking setting whose console names map to the model's enum values.
    private static func choice<T: Equatable>(
        _ key: String, _ section: DockingSettingSection, _ summary: String,
        _ path: WritableKeyPath<DockingCoordinator, T>, _ table: [(String, T)],
        validate: ((DockingCoordinator) throws -> Void)? = nil
    ) -> DockingSetting {
        DockingSetting(
            key: key, section: section, summary: summary, type: .choice(DockingNames.names(table)), rangeText: "",
            read: { DockingNames.name(of: $0[keyPath: path], in: table) },
            write: { draft, value in
                guard case .text(let word) = value, let choice = ConsoleNames.lookup(word, in: table) else { return }
                draft[keyPath: path] = choice
                try validate?(draft)
            },
            restore: { draft, fresh in draft[keyPath: path] = fresh[keyPath: path] }
        )
    }

    /// Compact number spelling for settings: no trailing zeros ("0.375", "5", "0.005").
    static func format(_ value: Float) -> String {
        var text = String(format: "%.4f", value)
        while text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text
    }
}

// MARK: - Console spellings of docking enums

/// Lower-case option values for the docking enums `ConsoleNames` does not cover.
enum DockingNames {
    static let torsionPresets: [(String, TorsionSearchPreset)] = [
        ("balanced", .balanced), ("redocking", .redocking), ("exploratory", .exploratory), ("manual", .manual)
    ]
    static let noiseSchedules: [(String, DiffusionDockingConfig.NoiseSchedule)] = [
        ("linear", .linear), ("cosine", .cosine), ("quadratic", .quadratic)
    ]
    static let gfn2Solvation: [(String, GFN2SolvationMode)] = [("none", .none), ("alpb", .water), ("gbsa", .gbsa)]
    static let gfn2Levels: [(String, GFN2OptLevel)] = [("crude", .crude), ("normal", .normal), ("tight", .tight)]
    static let constraintTypes: [(String, ConstraintInteractionType)] = [
        ("donor", .hbondDonor), ("acceptor", .hbondAcceptor), ("salt-bridge", .saltBridge),
        ("pi-stack", .piStacking), ("halogen", .halogen), ("metal", .metalCoordination), ("hydrophobic", .hydrophobic)
    ]

    static func names<T>(_ table: [(String, T)]) -> [String] { table.map(\.0) }

    static func name<T: Equatable>(of value: T, in table: [(String, T)]) -> String {
        table.first { $0.1 == value }?.0 ?? "\(value)"
    }
}

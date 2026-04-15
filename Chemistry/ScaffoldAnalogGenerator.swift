// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - Scaffold-Based Analog Generator
//
// Generates analogs from a user-drawn scaffold. Two modes:
//
//   .rgroupDecoration — scaffold must contain attachment points (`*` / `[*:n]`
//                       dummy atoms from Ketcher's R-group / attachment-point
//                       tool). Each R position is decorated with groups from
//                       a curated allow-list.
//
//   .wholeMolecule    — no attachment points required. A fixed table of
//                       SMARTS-based bioisosteric transformations is applied
//                       anywhere in the molecule. Useful for quickly scanning
//                       analogs of an existing lead.
//
// Candidates from either mode are validated by RDKit, filtered by Tanimoto
// similarity against the reference, descriptor filters (Lipinski, max logP),
// and a user-selected deny-list of unwanted substructures.
//
// This is a pure-Swift layer on top of existing RDKit C bindings — no C++
// core changes required.

/// A single curated functional group usable at an attachment point, or
/// as a deny-list pattern.
struct FunctionalGroup: Identifiable, Hashable, Sendable {
    let id: String              // stable short key, used for allow/deny sets
    let displayName: String     // shown in UI
    /// SMILES fragment to splice in at an attachment point. Must start with
    /// the atom that bonds to the scaffold.
    let attachmentSMILES: String
    /// SMARTS used to test whether the generated molecule contains this group
    /// anywhere. Used when the group is in the *deny* set.
    let denySMARTS: String
    /// Group this belongs to in the UI.
    let category: Category

    enum Category: String, CaseIterable, Sendable {
        case substituent = "Substituents"
        case functional  = "Functional groups"
        case aromatic    = "Aromatic / heterocycle"
        case saturated   = "Saturated rings / linkers"
    }
}

/// Curated list of functional groups the generator can enumerate or exclude.
enum FunctionalGroupCatalog {
    static let all: [FunctionalGroup] = [
        // --- Small substituents ----------------------------------------------
        .init(id: "H",       displayName: "Hydrogen (H)",     attachmentSMILES: "[H]",          denySMARTS: "[H]",             category: .substituent),
        .init(id: "Me",      displayName: "Methyl",           attachmentSMILES: "C",            denySMARTS: "[CH3]",           category: .substituent),
        .init(id: "Et",      displayName: "Ethyl",            attachmentSMILES: "CC",           denySMARTS: "[CH2][CH3]",      category: .substituent),
        .init(id: "nPr",     displayName: "n-Propyl",         attachmentSMILES: "CCC",          denySMARTS: "[CH2][CH2][CH3]", category: .substituent),
        .init(id: "iPr",     displayName: "Isopropyl",        attachmentSMILES: "C(C)C",        denySMARTS: "[CH]([CH3])[CH3]",category: .substituent),
        .init(id: "nBu",     displayName: "n-Butyl",          attachmentSMILES: "CCCC",         denySMARTS: "[CH2][CH2][CH2][CH3]", category: .substituent),
        .init(id: "iBu",     displayName: "Isobutyl",         attachmentSMILES: "CC(C)C",       denySMARTS: "[CH2][CH]([CH3])[CH3]", category: .substituent),
        .init(id: "tBu",     displayName: "tert-Butyl",       attachmentSMILES: "C(C)(C)C",     denySMARTS: "[CX4]([CH3])([CH3])[CH3]", category: .substituent),
        .init(id: "cPr",     displayName: "Cyclopropyl",      attachmentSMILES: "C1CC1",        denySMARTS: "C1CC1",           category: .substituent),
        .init(id: "cBu",     displayName: "Cyclobutyl",       attachmentSMILES: "C1CCC1",       denySMARTS: "C1CCC1",          category: .substituent),
        .init(id: "CF3",     displayName: "Trifluoromethyl",  attachmentSMILES: "C(F)(F)F",     denySMARTS: "C(F)(F)F",        category: .substituent),
        .init(id: "CHF2",    displayName: "Difluoromethyl",   attachmentSMILES: "C(F)F",        denySMARTS: "[CH](F)F",        category: .substituent),
        .init(id: "OCF3",    displayName: "Trifluoromethoxy", attachmentSMILES: "OC(F)(F)F",    denySMARTS: "OC(F)(F)F",       category: .substituent),
        .init(id: "OMe",     displayName: "Methoxy",          attachmentSMILES: "OC",           denySMARTS: "[OX2][CH3]",      category: .substituent),
        .init(id: "OEt",     displayName: "Ethoxy",           attachmentSMILES: "OCC",          denySMARTS: "[OX2][CH2][CH3]", category: .substituent),
        .init(id: "OH",      displayName: "Hydroxyl",         attachmentSMILES: "O",            denySMARTS: "[OX2H]",          category: .substituent),
        .init(id: "NH2",     displayName: "Amino",            attachmentSMILES: "N",            denySMARTS: "[NX3H2]",         category: .substituent),
        .init(id: "NHMe",    displayName: "Methylamino",      attachmentSMILES: "NC",           denySMARTS: "[NX3H][CH3]",     category: .substituent),
        .init(id: "NMe2",    displayName: "Dimethylamino",    attachmentSMILES: "N(C)C",        denySMARTS: "[NX3]([CH3])[CH3]", category: .substituent),
        .init(id: "F",       displayName: "Fluoro",           attachmentSMILES: "F",            denySMARTS: "[F]",             category: .substituent),
        .init(id: "Cl",      displayName: "Chloro",           attachmentSMILES: "Cl",           denySMARTS: "[Cl]",            category: .substituent),
        .init(id: "Br",      displayName: "Bromo",            attachmentSMILES: "Br",           denySMARTS: "[Br]",            category: .substituent),
        .init(id: "I",       displayName: "Iodo",             attachmentSMILES: "I",            denySMARTS: "[I]",             category: .substituent),
        .init(id: "CN",      displayName: "Cyano (nitrile)",  attachmentSMILES: "C#N",          denySMARTS: "C#N",             category: .substituent),
        .init(id: "NO2",     displayName: "Nitro",            attachmentSMILES: "[N+](=O)[O-]", denySMARTS: "[N+](=O)[O-]",    category: .substituent),
        .init(id: "SMe",     displayName: "Methylthio",       attachmentSMILES: "SC",           denySMARTS: "[SX2][CH3]",      category: .substituent),
        .init(id: "SH",      displayName: "Thiol",            attachmentSMILES: "S",            denySMARTS: "[SX2H]",          category: .substituent),

        // --- Functional groups -----------------------------------------------
        .init(id: "COOH",    displayName: "Carboxylic acid",  attachmentSMILES: "C(=O)O",       denySMARTS: "C(=O)[OH]",       category: .functional),
        .init(id: "COOMe",   displayName: "Methyl ester",     attachmentSMILES: "C(=O)OC",      denySMARTS: "C(=O)O[CH3]",     category: .functional),
        .init(id: "COOEt",   displayName: "Ethyl ester",      attachmentSMILES: "C(=O)OCC",     denySMARTS: "C(=O)O[CH2][CH3]",category: .functional),
        .init(id: "CONH2",   displayName: "Primary amide",    attachmentSMILES: "C(=O)N",       denySMARTS: "C(=O)[NX3H2]",    category: .functional),
        .init(id: "CONHMe",  displayName: "Secondary amide",  attachmentSMILES: "C(=O)NC",      denySMARTS: "C(=O)[NX3H][CH3]", category: .functional),
        .init(id: "CONMe2",  displayName: "Tertiary amide",   attachmentSMILES: "C(=O)N(C)C",   denySMARTS: "C(=O)[NX3]([CH3])[CH3]", category: .functional),
        .init(id: "NHC=O",   displayName: "Reverse amide (NHCOMe)", attachmentSMILES: "NC(=O)C", denySMARTS: "[NX3H]C(=O)[CH3]", category: .functional),
        .init(id: "SO2NH2",  displayName: "Sulfonamide",      attachmentSMILES: "S(=O)(=O)N",   denySMARTS: "S(=O)(=O)[NX3H2]", category: .functional),
        .init(id: "SO2NHMe", displayName: "N-Methyl sulfonamide", attachmentSMILES: "S(=O)(=O)NC", denySMARTS: "S(=O)(=O)[NX3H][CH3]", category: .functional),
        .init(id: "SO2Me",   displayName: "Methylsulfone",    attachmentSMILES: "S(=O)(=O)C",   denySMARTS: "S(=O)(=O)[CH3]",  category: .functional),
        .init(id: "NHSO2Me", displayName: "Methylsulfonamide (NHSO2Me)", attachmentSMILES: "NS(=O)(=O)C", denySMARTS: "[NX3H]S(=O)(=O)[CH3]", category: .functional),
        .init(id: "COMe",    displayName: "Methyl ketone",    attachmentSMILES: "C(=O)C",       denySMARTS: "[#6]C(=O)[CH3]",  category: .functional),
        .init(id: "CHO",     displayName: "Aldehyde",         attachmentSMILES: "C=O",          denySMARTS: "[CX3H1](=O)",     category: .functional),
        .init(id: "Urea",    displayName: "Urea (NHCONH2)",   attachmentSMILES: "NC(=O)N",      denySMARTS: "[NX3H]C(=O)[NX3H2]", category: .functional),
        .init(id: "Carbamate",displayName: "Carbamate (NHCOOMe)", attachmentSMILES: "NC(=O)OC", denySMARTS: "[NX3H]C(=O)O[CH3]", category: .functional),
        .init(id: "Guan",    displayName: "Guanidino",        attachmentSMILES: "N=C(N)N",      denySMARTS: "[NX2]=C([NX3])[NX3]", category: .functional),

        // --- Aromatics / heterocycles ---------------------------------------
        .init(id: "Ph",      displayName: "Phenyl",           attachmentSMILES: "c1ccccc1",     denySMARTS: "c1ccccc1",        category: .aromatic),
        .init(id: "4Fph",    displayName: "4-Fluorophenyl",   attachmentSMILES: "c1ccc(F)cc1",  denySMARTS: "c1ccc(F)cc1",     category: .aromatic),
        .init(id: "4Clph",   displayName: "4-Chlorophenyl",   attachmentSMILES: "c1ccc(Cl)cc1", denySMARTS: "c1ccc(Cl)cc1",    category: .aromatic),
        .init(id: "4OMeph",  displayName: "4-Methoxyphenyl",  attachmentSMILES: "c1ccc(OC)cc1", denySMARTS: "c1ccc(OC)cc1",    category: .aromatic),
        .init(id: "Benzyl",  displayName: "Benzyl (CH2Ph)",   attachmentSMILES: "Cc1ccccc1",    denySMARTS: "[CH2]c1ccccc1",   category: .aromatic),
        .init(id: "2Py",     displayName: "2-Pyridyl",        attachmentSMILES: "c1ccccn1",     denySMARTS: "c1ccccn1",        category: .aromatic),
        .init(id: "3Py",     displayName: "3-Pyridyl",        attachmentSMILES: "c1cccnc1",     denySMARTS: "c1cccnc1",        category: .aromatic),
        .init(id: "4Py",     displayName: "4-Pyridyl",        attachmentSMILES: "c1ccncc1",     denySMARTS: "c1ccncc1",        category: .aromatic),
        .init(id: "Pyrm",    displayName: "2-Pyrimidinyl",    attachmentSMILES: "c1ncccn1",     denySMARTS: "c1ncccn1",        category: .aromatic),
        .init(id: "Furan",   displayName: "2-Furyl",          attachmentSMILES: "c1ccco1",      denySMARTS: "c1ccco1",         category: .aromatic),
        .init(id: "Thio",    displayName: "2-Thienyl",        attachmentSMILES: "c1cccs1",      denySMARTS: "c1cccs1",         category: .aromatic),
        .init(id: "Imid",    displayName: "Imidazolyl",       attachmentSMILES: "c1ncc[nH]1",   denySMARTS: "c1ncc[nH]1",      category: .aromatic),
        .init(id: "Pyra",    displayName: "Pyrazolyl",        attachmentSMILES: "c1cc[nH]n1",   denySMARTS: "c1cc[nH]n1",      category: .aromatic),
        .init(id: "Oxaz",    displayName: "1,3-Oxazolyl",     attachmentSMILES: "c1ocnc1",      denySMARTS: "c1ocnc1",         category: .aromatic),
        .init(id: "Thiaz",   displayName: "1,3-Thiazolyl",    attachmentSMILES: "c1scnc1",      denySMARTS: "c1scnc1",         category: .aromatic),
        .init(id: "Tria",    displayName: "1,2,3-Triazolyl",  attachmentSMILES: "c1cn[nH]n1",   denySMARTS: "c1cn[nH]n1",      category: .aromatic),
        .init(id: "Tetra",   displayName: "Tetrazolyl",       attachmentSMILES: "c1nnn[nH]1",   denySMARTS: "c1nnn[nH]1",      category: .aromatic),
        .init(id: "Indole",  displayName: "3-Indolyl",        attachmentSMILES: "c1ccc2[nH]ccc2c1", denySMARTS: "c1ccc2[nH]ccc2c1", category: .aromatic),

        // --- Saturated rings & linkers --------------------------------------
        .init(id: "Morph",   displayName: "Morpholino",       attachmentSMILES: "N1CCOCC1",     denySMARTS: "N1CCOCC1",        category: .saturated),
        .init(id: "Pip",     displayName: "Piperidinyl",      attachmentSMILES: "N1CCCCC1",     denySMARTS: "N1CCCCC1",        category: .saturated),
        .init(id: "Pipz",    displayName: "Piperazinyl",      attachmentSMILES: "N1CCNCC1",     denySMARTS: "N1CCNCC1",        category: .saturated),
        .init(id: "NMePipz", displayName: "4-Methylpiperazinyl", attachmentSMILES: "N1CCN(C)CC1", denySMARTS: "N1CCN(C)CC1",   category: .saturated),
        .init(id: "Pyrrol",  displayName: "Pyrrolidinyl",     attachmentSMILES: "N1CCCC1",      denySMARTS: "N1CCCC1",         category: .saturated),
        .init(id: "Azet",    displayName: "Azetidinyl",       attachmentSMILES: "N1CCC1",       denySMARTS: "N1CCC1",          category: .saturated),
        .init(id: "cPent",   displayName: "Cyclopentyl",      attachmentSMILES: "C1CCCC1",      denySMARTS: "C1CCCC1",         category: .saturated),
        .init(id: "cHex",    displayName: "Cyclohexyl",       attachmentSMILES: "C1CCCCC1",     denySMARTS: "C1CCCCC1",        category: .saturated),
        .init(id: "THP",     displayName: "Tetrahydropyranyl",attachmentSMILES: "C1CCOCC1",     denySMARTS: "C1CCOCC1",        category: .saturated),
        .init(id: "THF",     displayName: "Tetrahydrofuryl",  attachmentSMILES: "C1CCOC1",      denySMARTS: "C1CCOC1",         category: .saturated),
    ]

    /// Default allow-set: everything except iodo, nitro, aldehyde, guanidine
    /// (rarely-wanted first-pass substituents).
    static let defaultAllowIDs: Set<String> = Set(
        all.filter { !["I", "NO2", "CHO", "Guan"].contains($0.id) }.map { $0.id }
    )
}

// MARK: - Generation mode

enum ScaffoldGenerationMode: String, CaseIterable, Sendable {
    /// Scaffold must contain attachment points. Decorates only at those sites.
    case rgroupDecoration = "R-group decoration"
    /// Ignores attachment points. Applies fixed SMARTS transformations anywhere.
    case wholeMolecule    = "Whole-molecule transforms"
}

// MARK: - Generator config

struct ScaffoldGeneratorConfig: Sendable {
    var mode: ScaffoldGenerationMode = .rgroupDecoration
    /// Maximum analogs to emit (hard cap; enumeration stops early once reached).
    var maxAnalogs: Int = 50
    /// Minimum Tanimoto similarity to the reference. 0 disables the filter.
    var minSimilarity: Float = 0.3
    /// IDs of groups that may be attached at R positions (R-group mode).
    var allowedGroupIDs: Set<String> = FunctionalGroupCatalog.defaultAllowIDs
    /// IDs of groups whose SMARTS must NOT be present in a generated analog.
    var deniedGroupIDs: Set<String> = []
    var filterLipinski: Bool = false
    /// Maximum LogP; ignored if ≥ 10.
    var maxLogP: Float = 7.0
}

// MARK: - Whole-molecule transformation table

private struct Transformation {
    let pattern: String       // SMILES substring to match
    let replacement: String
    let label: String
}

/// Fixed SMARTS-like transformations for whole-molecule mode. These are
/// textual SMILES substitutions, not true SMARTS — they're cheap and robust
/// enough for a first-pass analog scan.
private let wholeMoleculeTransforms: [Transformation] = [
    // Halogen swaps
    .init(pattern: "F",        replacement: "Cl",        label: "F→Cl"),
    .init(pattern: "Cl",       replacement: "F",         label: "Cl→F"),
    .init(pattern: "Cl",       replacement: "Br",        label: "Cl→Br"),
    .init(pattern: "Br",       replacement: "Cl",        label: "Br→Cl"),
    .init(pattern: "F",        replacement: "CF3",       label: "F→CF3"),
    // Alkyl length
    .init(pattern: "CC",       replacement: "CCC",       label: "Et→Pr"),
    .init(pattern: "CC",       replacement: "C(C)C",     label: "Et→iPr"),
    .init(pattern: "CCC",      replacement: "CC",        label: "Pr→Et"),
    // Heteroatom swaps
    .init(pattern: "O",        replacement: "N",         label: "O→N"),
    .init(pattern: "O",        replacement: "S",         label: "O→S"),
    .init(pattern: "S",        replacement: "O",         label: "S→O"),
    // Functional group swaps
    .init(pattern: "C(=O)O",   replacement: "C(=O)N",    label: "COOH→CONH2"),
    .init(pattern: "C(=O)N",   replacement: "C(=O)O",    label: "CONH2→COOH"),
    .init(pattern: "C(=O)O",   replacement: "S(=O)(=O)N",label: "COOH→SO2NH2"),
    .init(pattern: "C(=O)",    replacement: "S(=O)(=O)", label: "C=O→SO2"),
    // Aromatic swaps
    .init(pattern: "c1ccccc1", replacement: "c1ccncc1",  label: "Ph→4Py"),
    .init(pattern: "c1ccccc1", replacement: "c1cccnc1",  label: "Ph→3Py"),
    .init(pattern: "c1ccncc1", replacement: "c1ccccc1",  label: "4Py→Ph"),
    .init(pattern: "c1ccco1",  replacement: "c1cccs1",   label: "Fur→Thio"),
    // Ethers / thioethers
    .init(pattern: "OC",       replacement: "SC",        label: "OMe→SMe"),
    .init(pattern: "OC",       replacement: "NC",        label: "OMe→NHMe"),
    // Ring expansion / contraction
    .init(pattern: "C1CCCC1",  replacement: "C1CCCCC1",  label: "cPent→cHex"),
    .init(pattern: "C1CCCCC1", replacement: "C1CCCC1",   label: "cHex→cPent"),
    .init(pattern: "N1CCCC1",  replacement: "N1CCCCC1",  label: "Pyrrol→Pip"),
    .init(pattern: "N1CCCCC1", replacement: "N1CCOCC1",  label: "Pip→Morph"),
    .init(pattern: "N1CCCCC1", replacement: "N1CCNCC1",  label: "Pip→Pipz"),
]

// MARK: - Generator

enum ScaffoldAnalogGenerator {

    struct Analog: Sendable {
        let smiles: String
        let descriptors: LigandDescriptors
        let similarity: Float
        /// Short label describing what was changed.
        let label: String
    }

    enum GenerationError: Error, LocalizedError {
        case noAttachmentPoints
        case noAllowedGroups
        case invalidScaffold
        /// Whole-molecule mode: no transformation pattern matched the SMILES
        /// text at all. Nothing was generated to filter.
        case noTransformsMatched
        /// N candidates were generated but all were rejected by similarity /
        /// Lipinski / logP / deny filters.
        case allCandidatesFiltered(generated: Int)

        var errorDescription: String? {
            switch self {
            case .noAttachmentPoints:
                return "Scaffold contains no attachment points. Draw an R-group / attachment-point atom in Ketcher to mark decoration sites, or switch to Whole-molecule transforms."
            case .noAllowedGroups:
                return "No functional groups are allowed. Tick at least one in the Allowed list."
            case .invalidScaffold:
                return "Scaffold SMILES is not a valid molecule."
            case .noTransformsMatched:
                return "Whole-molecule mode: no transformations matched this scaffold's SMILES. This mode uses textual pattern matching and can miss fused rings or unusual atom orderings. Try R-group decoration mode instead — add an R-group to a ring position and enumerate substituents."
            case .allCandidatesFiltered(let n):
                return "Generated \(n) candidate\(n == 1 ? "" : "s"), but all were rejected by the similarity / descriptor filters. Try lowering Min Tanimoto similarity, raising Max LogP, or turning off the Lipinski filter."
            }
        }
    }

    /// Resolve the working scaffold SMILES. If `molfile` is non-empty, it is
    /// converted via RDKit — this preserves R-group / attachment-point dummy
    /// atoms that Ketcher's `getSmiles()` sometimes strips on aromatic atoms.
    /// Falls back to the raw SMILES if the molfile path fails or is empty.
    static func resolveScaffoldSMILES(smiles: String, molfile: String) -> String {
        if !molfile.isEmpty, let fromMol = RDKitBridge.smilesFromMolBlock(molfile), !fromMol.isEmpty {
            return fromMol
        }
        return smiles
    }

    /// Generate analogs according to `config.mode`.
    ///
    /// Cancellation-aware: call from within a `Task` and cancel the task to abort.
    /// Progress is reported via `progress` in 0…1 (coarse, per-candidate).
    static func generate(
        scaffoldSmiles: String,
        config: ScaffoldGeneratorConfig,
        progress: @Sendable (Float) -> Void = { _ in }
    ) throws -> [Analog] {
        switch config.mode {
        case .rgroupDecoration:
            return try generateRGroupDecoration(scaffoldSmiles: scaffoldSmiles, config: config, progress: progress)
        case .wholeMolecule:
            return try generateWholeMolecule(scaffoldSmiles: scaffoldSmiles, config: config, progress: progress)
        }
    }

    // MARK: - R-group decoration mode

    private static func generateRGroupDecoration(
        scaffoldSmiles: String,
        config: ScaffoldGeneratorConfig,
        progress: @Sendable (Float) -> Void
    ) throws -> [Analog] {

        let attachmentRanges = findAttachmentPointRanges(in: scaffoldSmiles)
        guard !attachmentRanges.isEmpty else { throw GenerationError.noAttachmentPoints }

        // Sentinel scaffold: replace each attachment token (right-to-left so
        // earlier indices stay valid) with `{R0}`, `{R1}`, …
        var sentinelScaffold = scaffoldSmiles
        for (i, r) in attachmentRanges.enumerated().reversed() {
            sentinelScaffold.replaceSubrange(r, with: "{R\(i)}")
        }
        let numSites = attachmentRanges.count

        // Reference for similarity = scaffold with all R → H.
        var bareScaffold = sentinelScaffold
        for i in 0..<numSites { bareScaffold = bareScaffold.replacingOccurrences(of: "{R\(i)}", with: "[H]") }
        guard RDKitBridge.computeDescriptors(smiles: bareScaffold) != nil else {
            throw GenerationError.invalidScaffold
        }

        let allowed = FunctionalGroupCatalog.all.filter { config.allowedGroupIDs.contains($0.id) }
        guard !allowed.isEmpty else { throw GenerationError.noAllowedGroups }
        let denied = FunctionalGroupCatalog.all.filter { config.deniedGroupIDs.contains($0.id) }

        // With multiple R-groups, a sequential mixed-radix counter advances
        // only the least-significant index across the first n iterations,
        // leaving all other R positions pinned to the baseline group. When
        // `maxAnalogs` is reached before the counter wraps, the user sees
        // analogs that vary only in R1. Instead, sample random tuples from
        // the full Cartesian product so every site varies independently.
        var analogs: [Analog] = []
        var seen = Set<String>()
        var generatedCount = 0  // valid RDKit parses pre-filter
        let n = allowed.count
        let totalCombinations: Int = {
            var t = 1
            for _ in 0..<numSites {
                if t > 1_000_000 { return Int.max }
                t *= n
            }
            return t
        }()
        var counter = Array(repeating: 0, count: numSites)
        var iterations = 0
        let hardIterationCap = min(totalCombinations, max(config.maxAnalogs * 20, 200))
        var rng = SystemRandomNumberGenerator()

        while analogs.count < config.maxAnalogs && iterations < hardIterationCap {
            if Task.isCancelled { break }

            // Iteration 0 keeps the all-baseline tuple as a reference point;
            // subsequent iterations sample each site independently at random.
            if iterations > 0 {
                for i in 0..<numSites {
                    counter[i] = Int.random(in: 0..<n, using: &rng)
                }
            }

            var candidate = sentinelScaffold
            var labelParts: [String] = []
            for i in 0..<numSites {
                let grp = allowed[counter[i]]
                candidate = candidate.replacingOccurrences(of: "{R\(i)}", with: grp.attachmentSMILES)
                labelParts.append(grp.id)
            }

            if !seen.contains(candidate),
               let desc = RDKitBridge.computeDescriptors(smiles: candidate) {
                seen.insert(candidate)
                generatedCount += 1
                if let analog = filterCandidate(
                    smiles: candidate, desc: desc, reference: bareScaffold,
                    config: config, denied: denied,
                    label: labelParts.joined(separator: "/")
                ) {
                    analogs.append(analog)
                }
            }

            iterations += 1
            progress(Float(iterations) / Float(hardIterationCap))
        }

        if analogs.isEmpty && generatedCount > 0 {
            throw GenerationError.allCandidatesFiltered(generated: generatedCount)
        }
        analogs.sort { $0.similarity > $1.similarity }
        return analogs
    }

    // MARK: - Whole-molecule transform mode

    private static func generateWholeMolecule(
        scaffoldSmiles: String,
        config: ScaffoldGeneratorConfig,
        progress: @Sendable (Float) -> Void
    ) throws -> [Analog] {
        // Strip any stray attachment-point tokens (the user may have switched
        // modes after drawing) — replace each with [H].
        var base = scaffoldSmiles
        for r in findAttachmentPointRanges(in: base).reversed() {
            base.replaceSubrange(r, with: "[H]")
        }
        guard RDKitBridge.computeDescriptors(smiles: base) != nil else {
            throw GenerationError.invalidScaffold
        }

        let denied = FunctionalGroupCatalog.all.filter { config.deniedGroupIDs.contains($0.id) }

        var analogs: [Analog] = []
        var seen = Set<String>([base])
        var generatedCount = 0
        let total = Float(wholeMoleculeTransforms.count)

        for (i, tr) in wholeMoleculeTransforms.enumerated() {
            if analogs.count >= config.maxAnalogs || Task.isCancelled { break }
            progress(Float(i) / total)

            // Generate both "first occurrence" and "all occurrences" variants.
            var candidates: [String] = []
            if let r = base.range(of: tr.pattern) {
                candidates.append(base.replacingCharacters(in: r, with: tr.replacement))
            }
            let all = base.replacingOccurrences(of: tr.pattern, with: tr.replacement)
            if all != base { candidates.append(all) }

            for cand in candidates {
                if analogs.count >= config.maxAnalogs { break }
                guard !seen.contains(cand),
                      let desc = RDKitBridge.computeDescriptors(smiles: cand) else { continue }
                seen.insert(cand)
                generatedCount += 1
                if let analog = filterCandidate(
                    smiles: cand, desc: desc, reference: base,
                    config: config, denied: denied, label: tr.label
                ) {
                    analogs.append(analog)
                }
            }
        }

        if analogs.isEmpty {
            if generatedCount == 0 {
                throw GenerationError.noTransformsMatched
            }
            throw GenerationError.allCandidatesFiltered(generated: generatedCount)
        }
        analogs.sort { $0.similarity > $1.similarity }
        return analogs
    }

    // MARK: - Shared candidate filter

    private static func filterCandidate(
        smiles: String,
        desc: LigandDescriptors,
        reference: String,
        config: ScaffoldGeneratorConfig,
        denied: [FunctionalGroup],
        label: String
    ) -> Analog? {
        if config.filterLipinski && !desc.lipinski { return nil }
        if config.maxLogP < 10 && desc.logP > config.maxLogP { return nil }

        var sim: Float = 1.0
        if config.minSimilarity > 0 {
            sim = RDKitBridge.tanimotoSimilarity(reference, smiles)
            if sim < config.minSimilarity { return nil }
        }

        for d in denied {
            if RDKitBridge.containsSubstructure(smiles: smiles, smarts: d.denySMARTS) {
                return nil
            }
        }
        return Analog(smiles: smiles, descriptors: desc, similarity: sim, label: label)
    }

    // MARK: - SMILES attachment-point tokenizer
    //
    // Finds ranges in a SMILES string corresponding to attachment-point atoms.
    // Recognised tokens:
    //   `[*:n]`  — mapped dummy atom (R-group label, Ketcher default for R1..Rn)
    //   `[*]`    — bracketed dummy atom
    //   `*`      — bare dummy atom
    //
    static func findAttachmentPointRanges(in smiles: String) -> [Range<String.Index>] {
        var result: [Range<String.Index>] = []
        var i = smiles.startIndex
        while i < smiles.endIndex {
            let c = smiles[i]
            if c == "[" {
                if let close = smiles[i...].firstIndex(of: "]") {
                    let inside = smiles[smiles.index(after: i)..<close]
                    let trimmed = inside.trimmingCharacters(in: .whitespaces)
                    if trimmed.hasPrefix("*") {
                        let rest = trimmed.dropFirst()
                        if rest.isEmpty || rest.hasPrefix(":") {
                            result.append(i..<smiles.index(after: close))
                            i = smiles.index(after: close)
                            continue
                        }
                    }
                    i = smiles.index(after: close)
                    continue
                } else {
                    break
                }
            } else if c == "*" {
                result.append(i..<smiles.index(after: i))
            }
            i = smiles.index(after: i)
        }
        return result
    }
}

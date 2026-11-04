// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - interactions, contacts, rmsd, fingerprint, validity
//
// Pose analysis. Every command works on a docked pose (`pose=N`, the 1-based rank of
// 'results'; pose 1 by default) or, with `ligand=on` or before any docking, on the loaded
// ligand where it stands — e.g. the co-crystallised ligand of a fetched complex.

extension DruseInterpreter {

    /// Registry entry: pose analysis commands.
    static func analysisCommands() -> [ConsoleCommand] {
        [interactionsCommand(), contactsCommand(), rmsdCommand(), fingerprintCommand(), validityCommand()]
    }

    // MARK: Pose selection

    private static let poseOption = ConsoleParameter(name: "pose", type: .int, summary: "Docked pose to analyse, 1-based rank from 'results' (default 1)")
    private static let ligandOption = ConsoleParameter(name: "ligand", type: .bool,
                                                       summary: "Analyse the loaded ligand where it stands (e.g. the crystal ligand) instead of a docked pose")
    private static let topOption = { (fallback: Int) in
        ConsoleParameter(name: "top", type: .int, summary: "Number of best-ranked poses (default \(fallback))")
    }

    /// The placement a pose-analysis command addresses: `pose=N`, the loaded ligand with
    /// `ligand=on`, otherwise pose 1 — or the loaded ligand when nothing has been docked.
    private static func analysedPlacement(_ inv: ConsoleInvocation) throws -> AppViewModel.LigandPlacement {
        let vm = inv.viewModel
        _ = try loadedProtein(vm)
        if inv.bool("ligand") == true {
            if inv.has("pose") { throw ConsoleError("use either pose= or ligand=on, not both") }
            guard let placement = vm.ligandPlacement(pose: nil) else { throw ConsoleError("no ligand loaded") }
            return placement
        }
        let results = vm.docking.dockingResults
        guard let rank = inv.int("pose") ?? (results.isEmpty ? nil : 1) else {
            guard let placement = vm.ligandPlacement(pose: nil) else {
                throw ConsoleError("no docking results and no ligand loaded — run 'dock' or load a ligand")
            }
            return placement
        }
        guard let placement = vm.ligandPlacement(pose: rank - 1) else {
            throw ConsoleError(results.isEmpty ? "no docking results — run 'dock' first (or use ligand=on)"
                               : "no pose #\(rank) — \(results.count) available")
        }
        return placement
    }

    /// The first `top` result indices, validated.
    private static func topPoseIndices(_ inv: ConsoleInvocation, default fallback: Int) throws -> [Int] {
        let results = inv.viewModel.docking.dockingResults
        guard !results.isEmpty else { throw ConsoleError("no docking results — run 'dock' first") }
        let top = inv.int("top") ?? fallback
        guard top >= 1 else { throw ConsoleError("top must be at least 1") }
        return Array(0..<min(top, results.count))
    }

    /// PDB atom name, or element + 1-based index for unnamed ligand atoms.
    private static func atomLabel(_ atom: Atom, index: Int) -> String {
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? "\(atom.element.symbol)\(index + 1)" : name
    }

    /// "A:ASP25" for the residue a protein atom belongs to.
    private static func proteinResidueLabel(_ atom: Atom) -> String {
        "\(atom.chainID):\(atom.residueName)\(atom.residueSeq)"
    }

    // MARK: Interaction type names

    /// Console spellings of `MolecularInteraction.InteractionType`, and the short codes used
    /// in fingerprint tables.
    private static let interactionNames: [(name: String, code: String, type: MolecularInteraction.InteractionType)] = [
        ("hbond", "HB", .hbond), ("hydrophobic", "HY", .hydrophobic), ("salt-bridge", "SB", .saltBridge),
        ("pi-stack", "PS", .piStack), ("pi-cation", "PC", .piCation), ("halogen", "XB", .halogen),
        ("metal", "MC", .metalCoord), ("ch-pi", "CP", .chPi), ("amide-pi", "AP", .amideStack), ("chalcogen", "CB", .chalcogen)
    ]

    /// Console spelling of an interaction type.
    private static func interactionName(_ type: MolecularInteraction.InteractionType) -> String {
        interactionNames.first { $0.type == type }?.name ?? type.label
    }

    /// Two-letter fingerprint code of an interaction type.
    private static func interactionCode(_ type: MolecularInteraction.InteractionType) -> String {
        interactionNames.first { $0.type == type }?.code ?? "?"
    }

    private static let allTypesOption = ConsoleParameter(
        name: "all", type: .bool,
        summary: "Report every interaction type, not only those the active scoring function accounts for (default off)")

    // MARK: interactions

    /// `interactions`: the interaction list of the Results step for any pose.
    private static func interactionsCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "interactions",
            summary: "Protein–ligand interactions of a pose: type, residue, atoms and distance",
            category: .analysis,
            forms: [
                ConsoleForm(path: [], summary: "List the interactions of a pose (or the loaded ligand), grouped by type",
                            options: [
                                poseOption, ligandOption,
                                ConsoleParameter(name: "type", type: .choice(interactionNames.map(\.name)), summary: "Only this interaction type"),
                                allTypesOption
                            ]) { inv in
                    try listInteractions(inv)
                }
            ],
            examples: ["interactions", "interactions pose=3", "interactions type=hbond", "interactions ligand=on all=on"]
        )
    }

    /// Prints the interactions of the addressed placement, sorted by type then distance, with a per-type summary.
    private static func listInteractions(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let placement = try analysedPlacement(inv)
        let allTypes = inv.bool("all") ?? false
        if !allTypes, vm.docking.scoringMethod.accountedInteractionTypes.isEmpty {
            throw ConsoleError("the \(ConsoleNames.scoring(vm.docking.scoringMethod)) scoring function has no per-interaction terms — add all=on")
        }
        let wanted = inv.string("type").flatMap { name in interactionNames.first { $0.name == name }?.type }
        let found = vm.interactions(of: placement, allTypes: allTypes)
            .filter { wanted == nil || $0.interaction.type == wanted }
            .sorted { ($0.interaction.type.rawValue, $0.interaction.distance) < ($1.interaction.type.rawValue, $1.interaction.distance) }
        guard !found.isEmpty else {
            inv.emit(.output, "No \(inv.string("type") ?? "") interactions for \(placement.label)"
                     + (allTypes ? "" : " (types counted by \(ConsoleNames.scoring(vm.docking.scoringMethod)); all=on shows every type)"))
            return
        }
        let rows = found.map { item -> [String] in
            let ligandIndex = item.interaction.ligandAtomIndex
            let ligandAtom = placement.atoms.indices.contains(ligandIndex)
                ? atomLabel(placement.atoms[ligandIndex], index: ligandIndex) : "ring"
            return [interactionName(item.interaction.type), ligandAtom, proteinResidueLabel(item.proteinAtom),
                    item.proteinAtom.name.trimmingCharacters(in: .whitespaces), String(format: "%.2f", item.interaction.distance)]
        }
        inv.emit(.output, ConsoleTable.render(["Type", "Ligand", "Residue", "Atom", "Å"], rows, rightAligned: [4]))
        var counts: [MolecularInteraction.InteractionType: Int] = [:]
        for item in found { counts[item.interaction.type, default: 0] += 1 }
        let residues = Set(found.map { proteinResidueLabel($0.proteinAtom) }).count
        let summary = counts.sorted { $0.key.rawValue < $1.key.rawValue }.map { "\($0.value) \(interactionName($0.key))" }
        inv.emit(.output, "\(placement.label): \(found.count) interactions with \(residues) residues — " + summary.joined(separator: ", "))
    }

    // MARK: contacts

    /// `contacts`: distance-based residue contacts, independent of interaction typing.
    private static func contactsCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "contacts",
            summary: "Residues in contact with a pose: closest distance and number of atom pairs",
            category: .analysis,
            forms: [
                ConsoleForm(path: [], summary: "Protein residues with a heavy atom within the cutoff of the ligand's heavy atoms",
                            options: [poseOption, ligandOption,
                                      ConsoleParameter(name: "cutoff", type: .float, summary: "Contact distance in Å (default 4)")]) { inv in
                    try listContacts(inv)
                }
            ],
            examples: ["contacts", "contacts pose=2 cutoff=4.5", "contacts ligand=on cutoff=3.5"]
        )
    }

    /// Prints residues within the cutoff of the placement's heavy atoms, closest first.
    private static func listContacts(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let placement = try analysedPlacement(inv)
        let protein = try loadedProtein(vm)
        let cutoff = Float(inv.double("cutoff") ?? 4)
        guard cutoff > 0, cutoff <= 15 else { throw ConsoleError("cutoff must be between 0 and 15 Å") }
        let cutoffSq = cutoff * cutoff

        /// Closest distance and number of atom pairs between the pose and one residue.
        struct Contact {
            var closest: Float = .greatestFiniteMagnitude
            var pairs = 0
            var ligandAtom = ""
            var proteinAtom = ""
        }
        var contacts: [Int: Contact] = [:]
        for (residueIndex, residue) in protein.residues.enumerated() {
            for atomIndex in residue.atomIndices where atomIndex < protein.atoms.count {
                let atom = protein.atoms[atomIndex]
                guard atom.element != .H else { continue }
                for (ligandIndex, position) in placement.positions.enumerated() {
                    let distanceSq = simd_distance_squared(atom.position, position)
                    guard distanceSq <= cutoffSq else { continue }
                    var contact = contacts[residueIndex, default: Contact()]
                    contact.pairs += 1
                    let distance = distanceSq.squareRoot()
                    if distance < contact.closest {
                        contact.closest = distance
                        contact.ligandAtom = atomLabel(placement.atoms[ligandIndex], index: ligandIndex)
                        contact.proteinAtom = atom.name.trimmingCharacters(in: .whitespaces)
                    }
                    contacts[residueIndex] = contact
                }
            }
        }
        guard !contacts.isEmpty else {
            inv.emit(.output, "No residue within \(String(format: "%.1f", cutoff)) Å of \(placement.label)")
            return
        }
        let rows = contacts.sorted { $0.value.closest < $1.value.closest }.map { index, contact -> [String] in
            [residueLabel(protein.residues[index]), String(format: "%.2f", contact.closest),
             "\(contact.ligandAtom)–\(contact.proteinAtom)", "\(contact.pairs)"]
        }
        inv.emit(.output, ConsoleTable.render(["Residue", "Closest Å", "Atoms", "Pairs"], rows, rightAligned: [1, 3]))
        inv.emit(.output, "\(contacts.count) residues within \(String(format: "%.1f", cutoff)) Å of \(placement.label)")
    }

    // MARK: rmsd

    /// `rmsd`: pose–pose RMSD, the pairwise matrix of the multi-pose analysis, and RMSD to the input ligand.
    private static func rmsdCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "rmsd",
            summary: "Heavy-atom RMSD between poses, or of each pose against the input ligand",
            category: .analysis,
            forms: [
                ConsoleForm(path: [], summary: "RMSD between two docked poses",
                            arguments: [ConsoleParameter(name: "a", type: .int, summary: "First pose (1-based rank)"),
                                        ConsoleParameter(name: "b", type: .int, summary: "Second pose (1-based rank)")]) { inv in
                    let vm = inv.viewModel
                    let a = inv.requiredInt("a"), b = inv.requiredInt("b")
                    let count = vm.docking.dockingResults.count
                    guard count > 0 else { throw ConsoleError("no docking results — run 'dock' first") }
                    for rank in [a, b] where !(1...count).contains(rank) {
                        throw ConsoleError("no pose #\(rank) — \(count) available")
                    }
                    let matrix = vm.computePairwiseRMSD(indices: [a - 1, b - 1])
                    guard matrix.count == 2 else { throw ConsoleError("these poses have no comparable coordinates") }
                    inv.emit(.output, String(format: "RMSD pose %d – pose %d: %.2f Å", a, b, matrix[0][1]))
                },
                ConsoleForm(path: ["matrix"], summary: "Pairwise RMSD matrix of the best poses", options: [topOption(5)]) { inv in
                    let vm = inv.viewModel
                    let indices = try topPoseIndices(inv, default: 5)
                    let matrix = vm.computePairwiseRMSD(indices: indices)
                    guard matrix.count == indices.count else { throw ConsoleError("the poses have no comparable coordinates") }
                    let rows = indices.enumerated().map { i, index -> [String] in
                        ["\(index + 1)"] + matrix[i].enumerated().map { j, value in i == j ? "—" : String(format: "%.2f", value) }
                    }
                    inv.emit(.output, ConsoleTable.render(["Pose"] + indices.map { "\($0 + 1)" }, rows,
                                                          rightAligned: Set(0...indices.count)))
                },
                ConsoleForm(path: ["reference"], summary: "RMSD of each pose against the input ligand coordinates (redocking RMSD for a crystal ligand)",
                            options: [topOption(10)]) { inv in
                    let vm = inv.viewModel
                    let indices = try topPoseIndices(inv, default: 10)
                    let method = vm.docking.scoringMethod
                    let rows = indices.map { index -> [String] in
                        let rmsd = vm.rmsdToInputLigand(pose: index)
                        return ["\(index + 1)", ConsoleFormat.score(vm.docking.dockingResults[index], method),
                                rmsd.map { String(format: "%.2f", $0) } ?? "—", rmsd.map { $0 <= 2 ? "yes" : "" } ?? ""]
                    }
                    guard rows.contains(where: { $0[2] != "—" }) else {
                        throw ConsoleError("no pose is comparable with the input ligand (different atoms)")
                    }
                    inv.emit(.output, ConsoleTable.render(["Pose", "Score", "RMSD Å", "≤ 2 Å"], rows, rightAligned: [0, 1, 2]))
                }
            ],
            examples: ["rmsd 1 2", "rmsd matrix top=3", "rmsd reference top=5"]
        )
    }

    // MARK: fingerprint

    /// `fingerprint`: the multi-pose interaction fingerprint as a residue × pose table.
    private static func fingerprintCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "fingerprint",
            summary: "Residue interaction fingerprints of the best poses",
            category: .analysis,
            forms: [
                ConsoleForm(path: [], summary: "Residue × pose table of interaction types, with how often each residue is contacted",
                            options: [topOption(5)]) { inv in
                    let vm = inv.viewModel
                    _ = try loadedProtein(vm)
                    let indices = try topPoseIndices(inv, default: 5)
                    let (keys, labels, fingerprints) = vm.computeInteractionFingerprints(indices: indices)
                    guard !keys.isEmpty else {
                        throw ConsoleError("no interactions in the selected poses"
                            + (vm.docking.scoringMethod.accountedInteractionTypes.isEmpty ? " (the scoring function reports none)" : ""))
                    }
                    let rows = keys.map { key -> [String] in
                        let cells = fingerprints.map { pose in
                            (pose[key] ?? []).sorted { $0.rawValue < $1.rawValue }.map(interactionCode).joined(separator: "+")
                        }
                        let frequency = cells.filter { !$0.isEmpty }.count
                        return [labels[key] ?? key] + cells.map { $0.isEmpty ? "·" : $0 } + ["\(frequency * 100 / max(indices.count, 1))%"]
                    }
                    inv.emit(.output, ConsoleTable.render(["Residue"] + indices.map { "P\($0 + 1)" } + ["Freq"], rows,
                                                          rightAligned: [indices.count + 1]))
                    let used = Set(fingerprints.flatMap { $0.values.flatMap { $0 } })
                    let legend = interactionNames.filter { used.contains($0.type) }.map { "\($0.code) \($0.name)" }
                    inv.emit(.output, "Legend: " + legend.joined(separator: ", "))
                }
            ],
            examples: ["fingerprint", "fingerprint top=10"]
        )
    }

    // MARK: validity

    /// `validity`: the PoseBusters-style checks run after docking, for chosen poses or the loaded ligand.
    private static func validityCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "validity",
            summary: "PoseBusters-style physical validity checks of poses",
            category: .analysis,
            forms: [
                ConsoleForm(path: [], summary: "Bond lengths and angles, internal and protein clashes, connectivity and extent of one pose or the best poses",
                            options: [poseOption, ligandOption, topOption(10)]) { inv in
                    try validityReport(inv)
                }
            ],
            examples: ["validity", "validity pose=1", "validity top=20", "validity ligand=on"]
        )
    }

    /// Prints the validity checks of one pose, the best poses or the loaded ligand (recomputed when a pose has none stored).
    private static func validityReport(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        let proteinHeavy = protein.atoms.filter { $0.element != .H }
        var rows: [[String]] = []

        func row(_ label: String, _ validity: PoseValidity) -> [String] {
            [label, validity.passed ? "pass" : "FAIL",
             String(format: "%.2f", validity.worstBondRatio),
             validity.minInternalDistance < .greatestFiniteMagnitude ? String(format: "%.2f", validity.minInternalDistance) : "—",
             validity.minProteinDistance < .greatestFiniteMagnitude ? String(format: "%.2f", validity.minProteinDistance) : "—",
             validity.failures.joined(separator: ", ")]
        }

        if inv.bool("ligand") == true || (vm.docking.dockingResults.isEmpty && !inv.has("pose") && !inv.has("top")) {
            let placement = try analysedPlacement(inv)
            rows.append(row(placement.label, PoseValidator.validate(
                ligandHeavyAtoms: placement.atoms, ligandHeavyBonds: placement.bonds,
                ligandPositions: placement.positions, proteinHeavyAtoms: proteinHeavy)))
        } else {
            let indices: [Int]
            if let rank = inv.int("pose") {
                if inv.has("top") { throw ConsoleError("use either pose= or top=, not both") }
                guard vm.docking.dockingResults.indices.contains(rank - 1) else {
                    throw ConsoleError(vm.docking.dockingResults.isEmpty ? "no docking results — run 'dock' first"
                                       : "no pose #\(rank) — \(vm.docking.dockingResults.count) available")
                }
                indices = [rank - 1]
            } else {
                indices = try topPoseIndices(inv, default: 10)
            }
            for index in indices {
                let result = vm.docking.dockingResults[index]
                if let validity = result.validity {
                    rows.append(row("pose \(index + 1)", validity))
                } else if let placement = vm.ligandPlacement(pose: index) {
                    rows.append(row("pose \(index + 1)", PoseValidator.validate(
                        ligandHeavyAtoms: placement.atoms, ligandHeavyBonds: placement.bonds,
                        ligandPositions: placement.positions, proteinHeavyAtoms: proteinHeavy)))
                }
            }
        }
        guard !rows.isEmpty else { throw ConsoleError("no pose has coordinates to check") }
        inv.emit(.output, ConsoleTable.render(["Pose", "Valid", "Worst bond", "Min intra Å", "Min protein Å", "Failed checks"],
                                              rows, rightAligned: [2, 3, 4]))
        let failed = rows.filter { $0[1] == "FAIL" }.count
        inv.emit(failed == 0 ? .success : .warning,
                 failed == 0 ? "All \(rows.count) checked pose(s) pass" : "\(failed)/\(rows.count) pose(s) fail at least one check")
    }
}

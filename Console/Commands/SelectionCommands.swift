// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - select, selection, subset, hide, show, center
//
// The selection is the same residue/atom selection the 3D view shows (`WorkspaceState`), so
// it exists in headless runs too: commands that act "on the selection" (pocket definition,
// subsets, side-chain packing, deletion) work identically in both modes. Only hiding,
// showing and centring are view-only.

extension DruseInterpreter {

    /// Registry entry: selection, subset and visibility commands.
    static func selectionCommands() -> [ConsoleCommand] {
        [selectCommand(), selectionCommand(), subsetCommand(), hideCommand(), showCommand(), centerCommand()]
    }

    // MARK: Residue specifications

    /// Help text for the residue-specification argument, shared by every command that takes one.
    static let residueSpecSummary =
        "Residues as [chain:]number, [chain:]first-last, [chain:]NAME or [chain:]NAMEnumber (e.g. A:25, A:25-30, HIS57, B:HOH)"

    /// The protein, or a console error telling the user how to load one.
    static func loadedProtein(_ vm: AppViewModel) throws -> Molecule {
        guard let protein = vm.molecules.protein else {
            throw ConsoleError("no protein loaded — use 'fetch <PDBID>' or 'load <file.pdb>'")
        }
        return protein
    }

    /// Residue indices (into `protein.residues`, in structure order) matched by residue
    /// specifications; see `residueSpecSummary`. A specification that matches nothing is an
    /// error, so typos do not silently select less than intended.
    static func residueIndices(matching specs: [String], in protein: Molecule) throws -> [Int] {
        var matched = Set<Int>()
        for spec in specs {
            let parts = spec.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            let chain: String? = parts.count == 2 ? parts[0] : nil
            let body = (parts.count == 2 ? parts[1] : parts[0]).uppercased()
            let letters = String(body.prefix { $0.isLetter })
            let numbers = body.dropFirst(letters.count)
            var range: ClosedRange<Int>?
            if !numbers.isEmpty {
                let bounds = numbers.split(separator: "-", omittingEmptySubsequences: false)
                guard (1...2).contains(bounds.count), let first = Int(bounds[0]),
                      let last = bounds.count == 2 ? Int(bounds[1]) : first, first <= last else {
                    throw ConsoleError("cannot read residue '\(spec)' — \(residueSpecSummary)")
                }
                range = first...last
            }
            guard !letters.isEmpty || range != nil else {
                throw ConsoleError("cannot read residue '\(spec)' — \(residueSpecSummary)")
            }
            if let chain, !protein.chains.contains(where: { $0.id == chain }) {
                throw ConsoleError(unknownChainMessage(chain, in: protein))
            }
            let hits = protein.residues.indices.filter { index in
                let residue = protein.residues[index]
                if let chain, residue.chainID != chain { return false }
                if !letters.isEmpty, residue.name.uppercased() != letters { return false }
                if let range, !range.contains(residue.sequenceNumber) { return false }
                return true
            }
            guard !hits.isEmpty else {
                throw ConsoleError("no residue matches '\(spec)'")
            }
            matched.formUnion(hits)
        }
        return matched.sorted()
    }

    /// "chain X not found" with the available chains and a did-you-mean hint.
    static func unknownChainMessage(_ chain: String, in protein: Molecule) -> String {
        let ids = protein.chains.map(\.id)
        var message = "no chain '\(chain)' — chains: \(ids.joined(separator: ", "))"
        if let hint = ids.first(where: { $0.lowercased() == chain.lowercased() }) {
            message += " (chain IDs are case-sensitive: did you mean '\(hint)'?)"
        }
        return message
    }

    /// "A:ASP25", the label every console table uses for a residue.
    static func residueLabel(_ residue: Residue) -> String {
        "\(residue.chainID):\(residue.name)\(residue.sequenceNumber)"
    }

    /// Residues grouped by chain with consecutive numbers collapsed ("A: 25-31, 45; B: 102"),
    /// for printing selections and pockets compactly.
    static func compactResidueList(_ indices: [Int], in protein: Molecule) -> String {
        var byChain: [String: [Int]] = [:]
        var order: [String] = []
        for index in indices.sorted() where index < protein.residues.count {
            let residue = protein.residues[index]
            if byChain[residue.chainID] == nil { order.append(residue.chainID) }
            byChain[residue.chainID, default: []].append(residue.sequenceNumber)
        }
        return order.map { chain -> String in
            let numbers = Array(Set(byChain[chain] ?? [])).sorted()
            var runs: [String] = []
            var start = numbers.first ?? 0
            var previous = start
            for number in numbers.dropFirst() + [Int.max] {
                if number == previous + 1 {
                    previous = number
                    continue
                }
                runs.append(start == previous ? "\(start)" : "\(start)-\(previous)")
                start = number
                previous = number
            }
            return "\(chain): " + runs.joined(separator: ", ")
        }.joined(separator: "; ")
    }

    /// Prints the size of the current selection after a selection command.
    private static func reportSelection(_ inv: ConsoleInvocation, _ what: String) {
        let workspace = inv.viewModel.workspace
        let residues = workspace.selectedResidueIndices.count
        inv.emit(.success, "\(what): \(residues) residue\(residues == 1 ? "" : "s"), \(workspace.selectedAtomIndices.count) atoms selected")
    }

    // MARK: select

    /// `select`: builds the residue selection from specifications, chains, distances, secondary structure or the pocket.
    private static func selectCommand() -> ConsoleCommand {
        let add = ConsoleParameter(name: "add", type: .bool, summary: "Add to the current selection instead of replacing it (default off)")
        let count = ConsoleParameter(name: "count", type: .int, summary: "Number of residues to grow or shrink by at each end (default 1)")
        return ConsoleCommand(
            name: "select",
            summary: "Select protein residues by number, chain, distance, secondary structure or pocket",
            category: .selection,
            forms: [
                ConsoleForm(path: ["residue"], summary: "Select residues by specification",
                            arguments: [ConsoleParameter(name: "residues", type: .string, summary: residueSpecSummary, isVariadic: true)],
                            options: [add]) { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let specs = inv.requiredString("residues").split(separator: " ").map(String.init)
                    let indices = try residueIndices(matching: specs, in: protein)
                    vm.selectResidues(Set(indices), extend: inv.bool("add") ?? false)
                    reportSelection(inv, "Selected \(compactResidueList(indices, in: protein))")
                },
                ConsoleForm(path: ["chain"], summary: "Select every residue of a chain (and ligand atoms carrying that chain ID)",
                            arguments: [ConsoleParameter(name: "id", type: .string, summary: "Chain ID (case-sensitive)")]) { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let id = inv.requiredString("id")
                    guard protein.chains.contains(where: { $0.id == id }) else {
                        throw ConsoleError(unknownChainMessage(id, in: protein))
                    }
                    vm.selectChain(id)
                    reportSelection(inv, "Selected chain \(id)")
                },
                ConsoleForm(path: ["within"], summary: "Select residues with any atom within a distance of the ligand, the pocket or the current selection",
                            arguments: [
                                ConsoleParameter(name: "distance", type: .float, summary: "Cutoff in Å"),
                                ConsoleParameter(name: "of", type: .choice(["of"]), summary: "The word 'of'"),
                                ConsoleParameter(name: "target", type: .choice(["ligand", "pocket", "selection"]),
                                                 summary: "ligand = displayed ligand (or pose=N), pocket = docking pocket probes, selection = grow the selection")
                            ],
                            options: [
                                add,
                                ConsoleParameter(name: "pose", type: .int, summary: "With 'of ligand': measure from docked pose N instead of the displayed ligand")
                            ]) { inv in
                    try selectWithin(inv)
                },
                ConsoleForm(path: ["ss"], summary: "Select residues by secondary structure (assigned from backbone H-bonds when the file has none)",
                            arguments: [ConsoleParameter(name: "type", type: .choice(["helix", "sheet", "turn", "loop"]), summary: "Secondary-structure class; loop = coil")],
                            options: [ConsoleParameter(name: "chain", type: .string, summary: "Only this chain")]) { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let chain = inv.string("chain")
                    if let chain, !protein.chains.contains(where: { $0.id == chain }) {
                        throw ConsoleError(unknownChainMessage(chain, in: protein))
                    }
                    let type: SecondaryStructure = switch inv.requiredString("type") {
                    case "helix": .helix
                    case "sheet": .sheet
                    case "turn": .turn
                    default: .coil
                    }
                    vm.selectBySecondaryStructure(type, chainID: chain)
                    reportSelection(inv, "Selected \(inv.requiredString("type")) residues" + (chain.map { " of chain \($0)" } ?? ""))
                },
                ConsoleForm(path: ["pocket"], summary: "Select the residues lining the selected docking pocket", options: [add]) { inv in
                    let vm = inv.viewModel
                    _ = try loadedProtein(vm)
                    guard let pocket = vm.docking.selectedPocket else {
                        throw ConsoleError("no pocket selected — use 'pocket detect' or 'selection pocket'")
                    }
                    guard !pocket.residueIndices.isEmpty else {
                        throw ConsoleError("the selected pocket lists no residues (an explicit box) — use 'select within 4 of pocket'")
                    }
                    vm.selectResidues(Set(pocket.residueIndices), extend: inv.bool("add") ?? false)
                    reportSelection(inv, "Selected pocket residues")
                },
                ConsoleForm(path: ["all"], summary: "Select every protein residue") { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    vm.selectResidues(Set(protein.residues.indices), extend: false)
                    reportSelection(inv, "Selected all")
                },
                ConsoleForm(path: ["none"], summary: "Clear the selection") { inv in
                    inv.viewModel.deselectAll()
                    inv.emit(.success, "Selection cleared")
                },
                ConsoleForm(path: ["invert"], summary: "Select every protein residue that is not selected, and deselect the rest") { inv in
                    _ = try loadedProtein(inv.viewModel)
                    inv.viewModel.invertSelection()
                    reportSelection(inv, "Inverted")
                },
                ConsoleForm(path: ["extend"], summary: "Grow the selection along the chain by residues at each end", options: [count]) { inv in
                    try resizeSelection(inv, grow: true)
                },
                ConsoleForm(path: ["shrink"], summary: "Remove residues from each end of the selected segments", options: [count]) { inv in
                    try resizeSelection(inv, grow: false)
                }
            ],
            examples: [
                "select residue A:25 A:27-30 HIS57",
                "select residue B:HOH add=on",
                "select chain A",
                "select within 5 of ligand",
                "select within 4.5 of ligand pose=1",
                "select within 3 of selection",
                "select ss helix chain=A",
                "select pocket",
                "select invert",
                "select extend count=2",
                "select none"
            ]
        )
    }

    /// `select within <Å> of …`: residues near the ligand (or a pose), the pocket probes, or the current selection.
    private static func selectWithin(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        _ = try loadedProtein(vm)
        guard let distance = inv.double("distance"), distance > 0 else {
            throw ConsoleError("distance must be a positive number of Å")
        }
        let extend = inv.bool("add") ?? false
        let target = inv.requiredString("target")
        if inv.has("pose"), target != "ligand" {
            throw ConsoleError("pose= only applies to 'select within … of ligand'")
        }
        let positions: [SIMD3<Float>]
        switch target {
        case "ligand":
            let poseIndex = inv.int("pose").map { $0 - 1 }
            guard let placement = vm.ligandPlacement(pose: poseIndex) else {
                throw ConsoleError(poseIndex == nil ? "no ligand loaded — load one or use 'chain extract'"
                                   : "no pose #\(poseIndex! + 1) — \(vm.docking.dockingResults.count) available")
            }
            positions = placement.positions
        case "pocket":
            guard let pocket = vm.docking.selectedPocket else {
                throw ConsoleError("no pocket selected — use 'pocket detect' first")
            }
            positions = pocket.probePositions.isEmpty ? [pocket.center] : pocket.probePositions
        default:
            guard !vm.workspace.selectedAtomIndices.isEmpty || !vm.workspace.selectedResidueIndices.isEmpty else {
                throw ConsoleError("nothing is selected — select something first")
            }
            vm.selectResiduesWithinDistance(Float(distance))
            reportSelection(inv, "Grew the selection by \(String(format: "%.1f", distance)) Å")
            return
        }
        let added = vm.selectResidues(within: Float(distance), of: positions, extend: extend)
        reportSelection(inv, "\(extend ? "Added \(added) residues" : "Selected residues") within \(String(format: "%.1f", distance)) Å of the \(target)")
    }

    /// `select extend|shrink`: grows or trims selected chain segments `count=` residues at a time.
    private static func resizeSelection(_ inv: ConsoleInvocation, grow: Bool) throws {
        let vm = inv.viewModel
        _ = try loadedProtein(vm)
        guard !vm.workspace.selectedResidueIndices.isEmpty else {
            throw ConsoleError("no residues selected")
        }
        let steps = inv.int("count") ?? 1
        guard steps >= 1 else { throw ConsoleError("count must be at least 1") }
        for _ in 0..<steps {
            if grow { vm.extendSelectionByOneResidue() } else { vm.shrinkSelectionByOneResidue() }
        }
        reportSelection(inv, grow ? "Extended" : "Shrunk")
    }

    // MARK: selection

    /// `selection`: reports on the selection or acts on it as a whole (pocket definition, deletion).
    private static func selectionCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "selection",
            summary: "Inspect the selection, turn it into the docking pocket, or delete it",
            category: .selection,
            forms: [
                ConsoleForm(path: ["info"], summary: "Selected residues by chain, atom count, centre and extent",
                            options: [ConsoleParameter(name: "residues", type: .bool, summary: "Also list every residue with its name (default off)")]) { inv in
                    try selectionInfo(inv)
                },
                ConsoleForm(path: ["pocket"], summary: "Define the docking pocket (and grid box) from the selected residues") { inv in
                    let vm = inv.viewModel
                    _ = try loadedProtein(vm)
                    guard !vm.workspace.selectedResidueIndices.isEmpty else {
                        throw ConsoleError("no residues selected — e.g. 'select within 6 of ligand'")
                    }
                    vm.definePocketFromSelection()
                    guard let pocket = vm.docking.selectedPocket else {
                        throw ConsoleError("could not define a pocket from the selection")
                    }
                    inv.emit(.success, "Pocket from \(vm.workspace.selectedResidueIndices.count) residues: centre \(ConsoleFormat.vector(pocket.center)), "
                             + "box \(ConsoleFormat.box(pocket.size)) Å, \(String(format: "%.0f", pocket.volume)) Å³")
                },
                ConsoleForm(path: ["delete"], summary: "Delete the selected residues from the protein (cannot be undone)") { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let count = vm.workspace.selectedResidueIndices.count
                    guard count > 0 else { throw ConsoleError("no residues selected") }
                    let before = protein.atomCount
                    vm.deleteSelectedResidues()
                    inv.emit(.success, "Deleted \(count) residue\(count == 1 ? "" : "s") (\(before - (vm.molecules.protein?.atomCount ?? 0)) atoms)")
                }
            ],
            examples: ["selection info", "selection info residues=on", "select within 6 of ligand; selection pocket", "selection delete"]
        )
    }

    /// `selection info`: residue ranges per chain, atom count, centre and extent of the selection.
    private static func selectionInfo(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        let residues = vm.workspace.selectedResidueIndices.sorted()
        let atomIndices = vm.workspace.selectedAtomIndices
        guard !residues.isEmpty || !atomIndices.isEmpty else {
            inv.emit(.output, "Nothing selected")
            return
        }
        let allAtoms = protein.atoms + (vm.molecules.ligand?.atoms ?? [])
        let positions = atomIndices.filter { $0 < allAtoms.count }.map { allAtoms[$0].position }
        var lines = ["\(residues.count) residue\(residues.count == 1 ? "" : "s"), \(atomIndices.count) atoms"]
        if !residues.isEmpty {
            lines.append("Residues: " + compactResidueList(residues, in: protein))
        }
        if let first = positions.first {
            let low = positions.reduce(first) { simd_min($0, $1) }
            let high = positions.reduce(first) { simd_max($0, $1) }
            let center = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
            lines.append("Centre \(ConsoleFormat.vector(center)), extent \(ConsoleFormat.box((high - low) / 2)) Å")
        }
        inv.emit(.output, lines.joined(separator: "\n"))
        if inv.bool("residues") == true, !residues.isEmpty {
            let rows = residues.map { index -> [String] in
                let residue = protein.residues[index]
                return [residue.chainID, residue.name, "\(residue.sequenceNumber)", "\(residue.atomIndices.count)"]
            }
            inv.emit(.output, ConsoleTable.render(["Chain", "Name", "Number", "Atoms"], rows, rightAligned: [2, 3]))
        }
    }

    // MARK: subset

    /// `subset`: named residue sets, the console side of the Sequence step's subsets.
    private static func subsetCommand() -> ConsoleCommand {
        let name = ConsoleParameter(name: "name", type: .string, summary: "Subset name (quote names with spaces)")
        return ConsoleCommand(
            name: "subset",
            summary: "Named residue sets: save the selection, recall it, show or hide it",
            category: .selection,
            forms: [
                ConsoleForm(path: ["save"], summary: "Save the selected residues as a named subset", arguments: [name]) { inv in
                    let vm = inv.viewModel
                    _ = try loadedProtein(vm)
                    let subsetName = inv.requiredString("name")
                    guard !vm.workspace.selectedResidueIndices.isEmpty else {
                        throw ConsoleError("no residues selected")
                    }
                    guard !vm.workspace.residueSubsets.contains(where: { $0.name == subsetName }) else {
                        throw ConsoleError("a subset named '\(subsetName)' already exists — 'subset remove \(subsetName)' first")
                    }
                    vm.createSubsetFromSelection(name: subsetName)
                    inv.emit(.success, "Saved subset '\(subsetName)' (\(vm.workspace.selectedResidueIndices.count) residues)")
                },
                ConsoleForm(path: ["list"], summary: "List saved subsets") { inv in
                    let vm = inv.viewModel
                    let subsets = vm.workspace.residueSubsets
                    guard !subsets.isEmpty else {
                        inv.emit(.output, "No subsets — select residues and use 'subset save <name>'")
                        return
                    }
                    let rows = subsets.map { subset -> [String] in
                        let residues = vm.molecules.protein.map { compactResidueList(subset.residueIndices, in: $0) } ?? "—"
                        return [subset.name, "\(subset.residueIndices.count)", subset.isVisible ? "yes" : "no",
                                residues.count > 60 ? String(residues.prefix(59)) + "…" : residues]
                    }
                    inv.emit(.output, ConsoleTable.render(["Name", "Residues", "Shown", "Members"], rows, rightAligned: [1]))
                },
                ConsoleForm(path: ["select"], summary: "Make a subset the current selection", arguments: [name]) { inv in
                    let vm = inv.viewModel
                    _ = try loadedProtein(vm)
                    vm.selectSubset(id: try subset(named: inv.requiredString("name"), in: vm).id)
                    reportSelection(inv, "Selected subset '\(inv.requiredString("name"))'")
                },
                ConsoleForm(path: ["remove"], summary: "Delete a subset (the residues stay)", arguments: [name]) { inv in
                    let vm = inv.viewModel
                    vm.deleteSubset(id: try subset(named: inv.requiredString("name"), in: vm).id)
                    inv.emit(.success, "Removed subset '\(inv.requiredString("name"))'")
                },
                ConsoleForm(path: ["show"], summary: "Highlight a subset in the 3D view", arguments: [name], isGUIOnly: true) { inv in
                    try setSubsetVisibility(inv, visible: true)
                },
                ConsoleForm(path: ["hide"], summary: "Stop highlighting a subset in the 3D view", arguments: [name], isGUIOnly: true) { inv in
                    try setSubsetVisibility(inv, visible: false)
                }
            ],
            examples: ["select within 5 of ligand; subset save site", "subset list", "subset select site", "subset hide site", "subset remove site"]
        )
    }

    /// The subset with this name (exact, then case-insensitive), or an error with a did-you-mean hint.
    private static func subset(named name: String, in vm: AppViewModel) throws -> ResidueSubset {
        let subsets = vm.workspace.residueSubsets
        if let match = subsets.first(where: { $0.name == name }) ?? subsets.first(where: { $0.name.lowercased() == name.lowercased() }) {
            return match
        }
        var message = "no subset named '\(name)'"
        if let hint = ConsoleSuggest.closest(name, in: subsets.map(\.name)) {
            message += " — did you mean '\(hint)'?"
        } else if subsets.isEmpty {
            message += " — there are no subsets yet"
        }
        throw ConsoleError(message)
    }

    /// Shows or hides a subset's highlight; a no-op when it is already in that state.
    private static func setSubsetVisibility(_ inv: ConsoleInvocation, visible: Bool) throws {
        let vm = inv.viewModel
        let target = try subset(named: inv.requiredString("name"), in: vm)
        if target.isVisible != visible {
            vm.toggleSubsetVisibility(id: target.id)
        }
    }

    // MARK: hide, show, center (GUI only)

    /// The `<id>` chain argument shared by the hide/show chain forms.
    private static func chainArgument() -> ConsoleParameter {
        ConsoleParameter(name: "id", type: .string, summary: "Chain ID (case-sensitive)")
    }

    /// The `id` argument, checked against the protein's chains.
    private static func requireChain(_ inv: ConsoleInvocation) throws -> String {
        let protein = try loadedProtein(inv.viewModel)
        let id = inv.requiredString("id")
        guard protein.chains.contains(where: { $0.id == id }) else {
            throw ConsoleError(unknownChainMessage(id, in: protein))
        }
        return id
    }

    /// `hide`: view-only hiding of the selection or a chain.
    private static func hideCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "hide",
            summary: "Hide atoms in the 3D view (GUI only)",
            category: .view,
            forms: [
                ConsoleForm(path: ["selection"], summary: "Hide the selected atoms and clear the selection", isGUIOnly: true) { inv in
                    guard !inv.viewModel.workspace.selectedAtomIndices.isEmpty else {
                        throw ConsoleError("nothing is selected")
                    }
                    inv.viewModel.hideSelection()
                },
                ConsoleForm(path: ["chain"], summary: "Hide a whole chain", arguments: [chainArgument()], isGUIOnly: true) { inv in
                    let id = try requireChain(inv)
                    if !inv.viewModel.workspace.hiddenChainIDs.contains(id) {
                        inv.viewModel.toggleChainVisibility(id)
                    }
                }
            ],
            examples: ["select chain B; hide selection", "hide chain B"]
        )
    }

    /// `show`: view-only un-hiding of everything or one chain.
    private static func showCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "show",
            summary: "Show hidden atoms in the 3D view (GUI only)",
            category: .view,
            forms: [
                ConsoleForm(path: ["all"], summary: "Show every hidden atom and chain", isGUIOnly: true) { inv in
                    let vm = inv.viewModel
                    vm.showAllAtoms()
                    for chain in vm.workspace.hiddenChainIDs.sorted() {
                        vm.toggleChainVisibility(chain)
                    }
                },
                ConsoleForm(path: ["chain"], summary: "Show a hidden chain", arguments: [chainArgument()], isGUIOnly: true) { inv in
                    let id = try requireChain(inv)
                    if inv.viewModel.workspace.hiddenChainIDs.contains(id) {
                        inv.viewModel.toggleChainVisibility(id)
                    }
                }
            ],
            examples: ["show all", "show chain B"]
        )
    }

    /// `center`: camera moves to the selection, ligand or protein.
    private static func centerCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "center",
            summary: "Point the camera at the selection, the ligand or the protein (GUI only)",
            category: .view,
            forms: [
                ConsoleForm(path: ["selection"], summary: "Centre the view on the selected atoms", isGUIOnly: true) { inv in
                    guard !inv.viewModel.workspace.selectedAtomIndices.isEmpty else {
                        throw ConsoleError("nothing is selected")
                    }
                    inv.viewModel.centerOnSelection()
                },
                ConsoleForm(path: ["ligand"], summary: "Fit the view to the ligand", isGUIOnly: true) { inv in
                    guard inv.viewModel.molecules.ligand != nil else { throw ConsoleError("no ligand loaded") }
                    inv.viewModel.fitToLigand()
                },
                ConsoleForm(path: ["protein"], summary: "Fit the view to the protein", isGUIOnly: true) { inv in
                    _ = try loadedProtein(inv.viewModel)
                    inv.viewModel.fitToProtein()
                }
            ],
            examples: ["select within 5 of ligand; center selection", "center ligand"]
        )
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import Foundation

// MARK: - load, fetch, project, chain, residue, protein

extension DruseInterpreter {

    /// Registry entry: loading, projects, and chain/residue/protein structure commands.
    static func structureCommands() -> [ConsoleCommand] {
        [loadCommand(), fetchCommand(), projectCommand(), chainCommand(), residueCommand(), proteinCommand()]
    }

    // MARK: Load filters

    /// `chains=` and `waters=` of `load` and `fetch`: applied to the protein right after it
    /// is read, before anything else touches it.
    private static let loadFilterOptions: [ConsoleParameter] = [
        ConsoleParameter(name: "chains", type: .string, summary: "Keep only these protein chains, comma-separated (e.g. A or A,B)"),
        ConsoleParameter(name: "waters", type: .choice(["keep", "remove"]), summary: "Keep crystallographic waters (default) or delete them on load")
    ]

    /// Applies `chains=` / `waters=` to a freshly loaded protein and describes what changed.
    private static func applyLoadFilters(_ inv: ConsoleInvocation) throws -> [String] {
        let vm = inv.viewModel
        guard inv.has("chains") || inv.has("waters") else { return [] }
        guard let protein = vm.molecules.protein else {
            throw ConsoleError("chains= and waters= apply to protein structures (PDB/mmCIF) only")
        }
        var notes: [String] = []
        if let list = inv.string("chains") {
            let wanted = Set(list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })
            let missing = wanted.filter { id in !protein.chains.contains { $0.id == id } }
            guard missing.isEmpty else {
                throw ConsoleError("no chain \(missing.sorted().joined(separator: ", ")) in \(protein.name) — chains: \(protein.chains.map(\.id).joined(separator: ", "))")
            }
            let removed = vm.keepOnlyChains(wanted)
            if !removed.isEmpty { notes.append("removed chain\(removed.count == 1 ? "" : "s") \(removed.joined(separator: ", "))") }
        }
        if inv.string("waters") == "remove" {
            let before = vm.molecules.protein?.atomCount ?? 0
            vm.removeWaters()
            let removed = before - (vm.molecules.protein?.atomCount ?? 0)
            if removed > 0 { notes.append("removed waters (\(removed) atoms)") }
        }
        return notes
    }

    // MARK: load

    /// `load`: reads a structure or ligand file through the same loader as File ▸ Open.
    private static func loadCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "load",
            summary: "Load a structure or ligand file (pdb, cif, sdf, mol, mol2, smi, csv)",
            category: .structure,
            forms: [
                ConsoleForm(path: [], summary: "PDB/mmCIF replace the protein; SDF/MOL/MOL2 set the ligand; SMI/CSV import into the Ligand Database",
                            arguments: [ConsoleParameter(name: "file", type: .path, summary: "File to load")],
                            options: loadFilterOptions) { inv in
                    let vm = inv.viewModel
                    let url = inv.requiredURL("file")
                    try requireReadable(url)
                    let previousProtein = vm.molecules.protein
                    let previousLigand = vm.molecules.ligand
                    let previousEntries = vm.ligandDB.entries.count
                    guard let task = vm.loadFromFile(url: url) else {
                        throw ConsoleError("unsupported file type '.\(url.pathExtension)' (expected pdb, cif, sdf, mol, mol2, smi or csv)")
                    }
                    guard await task.value else {
                        throw ConsoleError("could not load \(url.lastPathComponent) — see the activity log")
                    }
                    // SMI/CSV imports land in the Ligand Database asynchronously.
                    await ConsoleObservation.wait { !vm.ligandDB.isProcessing }
                    try Task.checkCancellation()

                    let proteinChanged = vm.molecules.protein.map { $0 !== previousProtein } ?? false
                    if (inv.has("chains") || inv.has("waters")) && !proteinChanged {
                        throw ConsoleError("chains= and waters= apply to protein structures (PDB/mmCIF) only")
                    }
                    let notes = try applyLoadFilters(inv)

                    var loaded: [String] = []
                    if proteinChanged, let protein = vm.molecules.protein {
                        loaded.append("protein \(protein.name) (\(protein.atomCount) atoms, chains \(protein.chains.map(\.id).joined(separator: ",")))")
                    }
                    if let ligand = vm.molecules.ligand, ligand !== previousLigand {
                        loaded.append("ligand \(ligand.name) (\(ligand.heavyAtomCount) heavy atoms)")
                    }
                    let added = vm.ligandDB.entries.count - previousEntries
                    if added > 0 {
                        loaded.append("\(added) Ligand Database entr\(added == 1 ? "y" : "ies") (not prepared)")
                    }
                    guard !loaded.isEmpty else {
                        throw ConsoleError("\(url.lastPathComponent) contained nothing Druse could load")
                    }
                    inv.emit(.success, "Loaded \(url.lastPathComponent): " + (loaded + notes).joined(separator: ", "))
                }
            ],
            examples: ["load 1hsg.pdb", "load complex.cif chains=A waters=remove", "load ~/ligands/aspirin.sdf", "load library.smi"]
        )
    }

    // MARK: fetch

    /// `fetch`: downloads a PDB entry through the same path as the Search step.
    private static func fetchCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "fetch",
            summary: "Download a structure from the RCSB PDB (cached locally)",
            category: .structure,
            forms: [
                ConsoleForm(path: [], summary: "Load a PDB entry by its 4-character ID; its first HET ligand becomes the active ligand",
                            arguments: [ConsoleParameter(name: "id", type: .string, summary: "PDB ID, e.g. 1HSG")],
                            options: loadFilterOptions) { inv in
                    let vm = inv.viewModel
                    let id = inv.requiredString("id").uppercased()
                    guard id.count == 4, id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber) }) else {
                        throw ConsoleError("'\(id)' is not a PDB ID (4 letters/digits, e.g. 1HSG)")
                    }
                    guard await vm.loadFromPDB(id: id).value else {
                        throw ConsoleError("could not load \(id) — see the activity log")
                    }
                    let notes = try applyLoadFilters(inv)
                    var parts = ["\(vm.molecules.protein?.atomCount ?? 0) protein atoms"]
                    if let protein = vm.molecules.protein {
                        parts.append("chains \(protein.chains.map(\.id).joined(separator: ","))")
                    }
                    if let ligand = vm.molecules.ligand {
                        parts.append("ligand \(ligand.name)")
                    }
                    inv.emit(.success, "Loaded \(id): " + (parts + notes).joined(separator: ", "))
                }
            ],
            examples: ["fetch 1HSG", "fetch 4HHB chains=A,B waters=remove"]
        )
    }

    // MARK: project

    /// `project`: saves or opens a `.druse` project folder.
    private static func projectCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "project",
            summary: "Save or open a .druse project folder",
            category: .structure,
            forms: [
                ConsoleForm(path: ["save"], summary: "Save protein, ligands, results and settings",
                            arguments: [ConsoleParameter(name: "dir", type: .path, summary: "Project folder (…/name.druse)")]) { inv in
                    let url = inv.requiredURL("dir")
                    do {
                        try DruseProjectIO.save(to: url, viewModel: inv.viewModel)
                    } catch {
                        throw ConsoleError("could not save \(url.path): \(error.localizedDescription)")
                    }
                    inv.emit(.success, "Project saved to \(url.path)")
                },
                ConsoleForm(path: ["open"], summary: "Replace the workspace with a saved project",
                            arguments: [ConsoleParameter(name: "dir", type: .path, summary: "Project folder")]) { inv in
                    let url = inv.requiredURL("dir")
                    var isDirectory: ObjCBool = false
                    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                        throw ConsoleError("no project folder at \(url.path)")
                    }
                    do {
                        try await DruseProjectIO.load(from: url, into: inv.viewModel)
                    } catch {
                        throw ConsoleError("could not open \(url.path): \(error.localizedDescription)")
                    }
                    inv.emit(.success, "Opened project \(url.lastPathComponent)")
                }
            ],
            examples: ["project save ~/Druse/hiv_protease.druse", "project open ~/Druse/hiv_protease.druse"]
        )
    }

    // MARK: chain

    /// `chain`: the chain operations of the Sequence step and the chain context menu.
    private static func chainCommand() -> ConsoleCommand {
        let id = ConsoleParameter(name: "id", type: .string, summary: "Chain ID (case-sensitive)")
        return ConsoleCommand(
            name: "chain",
            summary: "List, remove, rename, merge or extract protein chains; print their sequence",
            category: .structure,
            forms: [
                ConsoleForm(path: ["list"], summary: "Chains with type, residue range, residue and atom counts") { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let rows = protein.chains.map { chain -> [String] in
                        let residues = chain.residueIndices.compactMap { $0 < protein.residues.count ? protein.residues[$0] : nil }
                        let numbers = residues.map(\.sequenceNumber)
                        let range = numbers.isEmpty ? "—" : "\(numbers.min() ?? 0)-\(numbers.max() ?? 0)"
                        let atoms = residues.reduce(0) { $0 + $1.atomIndices.count }
                        return [chain.id, chain.type.label, "\(residues.count)", "\(residues.filter(\.isStandard).count)",
                                range, "\(atoms)", vm.workspace.hiddenChainIDs.contains(chain.id) ? "yes" : "no"]
                    }
                    inv.emit(.output, ConsoleTable.render(["Chain", "Type", "Residues", "Amino acids", "Numbers", "Atoms", "Hidden"],
                                                          rows, rightAligned: [2, 3, 5]))
                },
                ConsoleForm(path: ["remove"], summary: "Delete a chain from the protein (and its extracted ligand, if any)", arguments: [id]) { inv in
                    let vm = inv.viewModel
                    let chain = try existingChain(inv)
                    let before = vm.molecules.protein?.atomCount ?? 0
                    vm.removeChain(chainID: chain)
                    inv.emit(.success, "Removed chain \(chain) (\(before - (vm.molecules.protein?.atomCount ?? 0)) atoms)")
                },
                ConsoleForm(path: ["rename"], summary: "Give a chain a new ID",
                            arguments: [id, ConsoleParameter(name: "new", type: .string, summary: "New chain ID (1–3 printable characters)")]) { inv in
                    let vm = inv.viewModel
                    let chain = try existingChain(inv)
                    let newID = inv.requiredString("new")
                    guard (1...3).contains(newID.count), newID.allSatisfy({ $0.isASCII && !$0.isWhitespace }) else {
                        throw ConsoleError("chain IDs are 1–3 printable characters")
                    }
                    guard !(vm.molecules.protein?.chains.contains { $0.id == newID } ?? false) else {
                        throw ConsoleError("chain \(newID) already exists — use 'chain merge \(chain) \(newID)' to combine them")
                    }
                    vm.renameChain(from: chain, to: newID)
                    inv.emit(.success, "Renamed chain \(chain) to \(newID)")
                },
                ConsoleForm(path: ["merge"], summary: "Move every residue of one chain into another",
                            arguments: [id, ConsoleParameter(name: "into", type: .string, summary: "Target chain ID")]) { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let source = try existingChain(inv)
                    let target = inv.requiredString("into")
                    guard protein.chains.contains(where: { $0.id == target }) else {
                        throw ConsoleError(unknownChainMessage(target, in: protein))
                    }
                    guard source != target else { throw ConsoleError("cannot merge a chain into itself") }
                    vm.mergeChains(from: source, into: target)
                    inv.emit(.success, "Merged chain \(source) into \(target)")
                },
                ConsoleForm(path: ["extract"], summary: "Move a chain out of the protein and make it the active ligand (added to the Ligand Database)",
                            arguments: [id]) { inv in
                    let vm = inv.viewModel
                    let chain = try existingChain(inv)
                    vm.extractChainAsLigand(chainID: chain)
                    guard let ligand = vm.molecules.ligand else {
                        throw ConsoleError("could not extract chain \(chain) — see the activity log")
                    }
                    inv.emit(.success, "Extracted chain \(chain) as ligand \(ligand.name) (\(ligand.heavyAtomCount) heavy atoms)"
                             + (ligand.smiles.map { ", \($0)" } ?? ""))
                },
                ConsoleForm(path: ["sequence"], summary: "Print the one-letter amino-acid sequence of a chain (all chains when omitted)",
                            arguments: [ConsoleParameter(name: "id", type: .string, summary: "Chain ID (default: every protein chain)", isOptional: true)],
                            options: [
                                ConsoleParameter(name: "format", type: .choice(["plain", "fasta"]), summary: "plain (default) or FASTA records with a >name_chain header"),
                                ConsoleParameter(name: "copy", type: .bool, summary: "Also copy it to the clipboard (GUI only, default off)")
                            ]) { inv in
                    try printSequence(inv)
                }
            ],
            examples: ["chain list", "chain remove B", "chain rename B C", "chain merge B A",
                       "chain extract L", "chain sequence A", "chain sequence format=fasta copy=on"]
        )
    }

    /// The `id` argument, checked against the protein's chains.
    private static func existingChain(_ inv: ConsoleInvocation) throws -> String {
        let protein = try loadedProtein(inv.viewModel)
        let id = inv.requiredString("id")
        guard protein.chains.contains(where: { $0.id == id }) else {
            throw ConsoleError(unknownChainMessage(id, in: protein))
        }
        return id
    }

    /// `chain sequence`: one-letter sequences as plain lines or FASTA records, optionally copied to the clipboard.
    private static func printSequence(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        let chains: [Chain]
        if inv.has("id") {
            let id = try existingChain(inv)
            chains = protein.chains.filter { $0.id == id }
        } else {
            chains = protein.chains.filter { $0.type == .protein }
        }
        let fasta = inv.string("format") == "fasta"
        var records: [String] = []
        for chain in chains {
            let sequence = vm.oneLetterSequence(residueIndices: chain.residueIndices)
            guard !sequence.isEmpty else { continue }
            if fasta {
                let wrapped = stride(from: 0, to: sequence.count, by: 60).map { offset -> String in
                    let start = sequence.index(sequence.startIndex, offsetBy: offset)
                    let end = sequence.index(start, offsetBy: 60, limitedBy: sequence.endIndex) ?? sequence.endIndex
                    return String(sequence[start..<end])
                }
                records.append(">\(protein.name)_\(chain.id) length=\(sequence.count)\n" + wrapped.joined(separator: "\n"))
            } else {
                records.append(chains.count > 1 ? "\(chain.id): \(sequence)" : sequence)
            }
        }
        guard !records.isEmpty else {
            throw ConsoleError("no amino-acid residues in \(inv.string("id").map { "chain \($0)" } ?? "the protein")")
        }
        let text = records.joined(separator: "\n")
        inv.emit(.output, text)
        if inv.bool("copy") == true {
            if inv.isHeadless {
                inv.emit(.warning, "copy=on only applies in the GUI — the sequence was printed instead")
            } else {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                inv.emit(.success, "Copied to the clipboard")
            }
        }
    }

    // MARK: residue

    /// `residue`: residue listing, per-residue atom detail and residue deletion.
    private static func residueCommand() -> ConsoleCommand {
        let chain = ConsoleParameter(name: "chain", type: .string, summary: "Only this chain")
        return ConsoleCommand(
            name: "residue",
            summary: "List residues, inspect one residue's atoms and protonation, or delete residues",
            category: .structure,
            forms: [
                ConsoleForm(path: ["list"], summary: "Residues of the protein, filtered by chain, kind or name",
                            options: [
                                chain,
                                ConsoleParameter(name: "type", type: .choice(["all", "amino", "het", "water"]),
                                                 summary: "all (default), amino = standard amino acids, het = ligands/ions/modified residues, water"),
                                ConsoleParameter(name: "name", type: .string, summary: "Only residues with this 3-letter name (e.g. HIS)")
                            ]) { inv in
                    try listResidues(inv, kind: inv.string("type") ?? "all")
                },
                ConsoleForm(path: ["list", "het"], summary: "Hetero groups only: ligands, ions, cofactors, modified residues (no water)",
                            options: [chain]) { inv in
                    try listResidues(inv, kind: "het")
                },
                ConsoleForm(path: ["info"], summary: "Atoms of the matched residues with charges, B-factors and the assigned protonation state",
                            arguments: [ConsoleParameter(name: "residues", type: .string, summary: residueSpecSummary, isVariadic: true)]) { inv in
                    try residueInfo(inv)
                },
                ConsoleForm(path: ["remove"], summary: "Delete residues from the protein (cannot be undone; clears the selection and subsets)",
                            arguments: [ConsoleParameter(name: "residues", type: .string, summary: residueSpecSummary, isVariadic: true)]) { inv in
                    let vm = inv.viewModel
                    let protein = try loadedProtein(vm)
                    let indices = try residueIndices(matching: inv.requiredString("residues").split(separator: " ").map(String.init), in: protein)
                    let before = protein.atomCount
                    let description = compactResidueList(indices, in: protein)
                    vm.selectResidues(Set(indices), extend: false)
                    vm.deleteSelectedResidues()
                    inv.emit(.success, "Deleted \(indices.count) residue\(indices.count == 1 ? "" : "s") (\(description)), "
                             + "\(before - (vm.molecules.protein?.atomCount ?? 0)) atoms")
                }
            ],
            examples: ["residue list chain=A", "residue list het", "residue list type=water", "residue list name=HIS",
                       "residue info A:HIS57", "residue info A:25-27", "residue remove A:SO4"]
        )
    }

    /// Prints the residues of one kind (all, amino, het, water) with heavy/H atom counts and formal charge.
    private static func listResidues(_ inv: ConsoleInvocation, kind: String) throws {
        let protein = try loadedProtein(inv.viewModel)
        let chain = inv.string("chain")
        if let chain, !protein.chains.contains(where: { $0.id == chain }) {
            throw ConsoleError(unknownChainMessage(chain, in: protein))
        }
        let name = inv.string("name")?.uppercased()
        let residues = protein.residues.filter { residue in
            if let chain, residue.chainID != chain { return false }
            if let name, residue.name.uppercased() != name { return false }
            switch kind {
            case "amino": return residue.isStandard
            case "het": return !residue.isStandard && !residue.isWater
            case "water": return residue.isWater
            default: return true
            }
        }
        guard !residues.isEmpty else {
            inv.emit(.output, "No matching residues")
            return
        }
        let rows = residues.map { residue -> [String] in
            let atoms = residue.atomIndices.compactMap { $0 < protein.atoms.count ? protein.atoms[$0] : nil }
            let heavy = atoms.filter { $0.element != .H }.count
            let charge = atoms.reduce(0) { $0 + $1.formalCharge }
            let kindLabel = residue.isWater ? "water" : residue.isStandard ? "amino" : "het"
            return [residue.chainID, residue.name, "\(residue.sequenceNumber)", kindLabel, "\(heavy)",
                    "\(atoms.count - heavy)", charge == 0 ? "" : String(format: "%+d", charge)]
        }
        inv.emit(.output, ConsoleTable.render(["Chain", "Name", "Number", "Kind", "Heavy", "H", "Charge"],
                                              rows, rightAligned: [2, 4, 5, 6]))
        inv.emit(.output, "\(residues.count) residue\(residues.count == 1 ? "" : "s")")
    }

    /// `residue info`: every atom of the matched residues plus the protonation state assigned to them.
    private static func residueInfo(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        let indices = try residueIndices(matching: inv.requiredString("residues").split(separator: " ").map(String.init), in: protein)
        guard indices.count <= 20 else {
            throw ConsoleError("\(indices.count) residues match — 'residue info' shows at most 20; narrow the specification")
        }
        for index in indices {
            let residue = protein.residues[index]
            let atoms = residue.atomIndices.compactMap { $0 < protein.atoms.count ? protein.atoms[$0] : nil }
            var header = "\(residueLabel(residue)) — \(atoms.count) atoms"
            let states = vm.protonationStates.filter {
                $0.chainID == residue.chainID && $0.residueSeq == residue.sequenceNumber
            }
            if !states.isEmpty {
                header += ", protonation " + states.map {
                    "\($0.chemicalKind) \($0.state.rawValue) (pKa \(String(format: "%.2f", $0.shiftedPKa)), model \(String(format: "%.2f", $0.modelPKa)))"
                }.joined(separator: "; ")
            }
            let partial = atoms.reduce(Float(0)) { $0 + $1.charge }
            header += String(format: ", partial charge %+.3f", partial)
            let rows = atoms.map { atom -> [String] in
                [atom.name, atom.element.symbol, String(format: "%+.3f", atom.charge),
                 atom.formalCharge == 0 ? "" : String(format: "%+d", atom.formalCharge),
                 String(format: "%.2f", atom.occupancy), String(format: "%.1f", atom.tempFactor),
                 String(format: "%.2f,%.2f,%.2f", atom.position.x, atom.position.y, atom.position.z)]
            }
            inv.emit(.output, header + "\n" + ConsoleTable.render(["Atom", "El", "q", "Formal", "Occ", "B", "Position"],
                                                                 rows, rightAligned: [2, 3, 4, 5]))
        }
    }

    // MARK: protein

    /// `protein`: a one-screen summary of the loaded protein.
    private static func proteinCommand() -> ConsoleCommand {
        ConsoleCommand(
            name: "protein",
            summary: "Summarise the loaded protein",
            category: .structure,
            forms: [
                ConsoleForm(path: ["info"], summary: "Atoms, residues, chains, waters, HET groups, mass, charge, secondary structure and preparation state") { inv in
                    try proteinInfo(inv)
                }
            ],
            examples: ["protein info"]
        )
    }

    /// `protein info`: composition, mass, charge, secondary structure, gaps and preparation state.
    private static func proteinInfo(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let protein = try loadedProtein(vm)
        vm.ensureSecondaryStructure()
        let atoms = protein.atoms
        let hydrogens = atoms.filter { $0.element == .H }.count
        let amino = protein.residues.filter(\.isStandard)
        let waters = protein.residues.filter(\.isWater)
        let het = protein.residues.filter { !$0.isStandard && !$0.isWater }
        var hetCounts: [String: Int] = [:]
        for residue in het { hetCounts[residue.name, default: 0] += 1 }
        let mass = atoms.reduce(Float(0)) { $0 + $1.element.mass }
        let formal = atoms.reduce(0) { $0 + $1.formalCharge }
        let partial = atoms.reduce(Float(0)) { $0 + $1.charge }
        let charged = atoms.filter { abs($0.charge) > 0.001 }.count

        func residuesIn(_ type: SecondaryStructure) -> Int {
            amino.filter { residue in
                protein.secondaryStructureAssignments.contains {
                    $0.type == type && $0.chain == residue.chainID && (($0.start)...($0.end)).contains(residue.sequenceNumber)
                }
            }.count
        }
        let helices = protein.secondaryStructureAssignments.filter { $0.type == .helix }.count
        let strands = protein.secondaryStructureAssignments.filter { $0.type == .sheet }.count
        let helixResidues = residuesIn(.helix)
        let strandResidues = residuesIn(.sheet)
        let gaps = ProteinPreparation.detectMissingResidues(in: atoms)

        let chainSummary: String = protein.chains
            .map { "\($0.id) (\($0.type.label.lowercased()), \($0.residueIndices.count))" }
            .joined(separator: ", ")
        let hetSummary: String = hetCounts.isEmpty ? "none" : hetCounts.sorted { $0.key < $1.key }
            .map { $0.value > 1 ? "\($0.key)×\($0.value)" : $0.key }
            .joined(separator: ", ")
        let massNote: String = hydrogens == 0 ? " (heavy atoms only — no hydrogens yet)" : ""
        let missingResidues: Int = gaps.reduce(0) { $0 + $1.gapEnd - $1.gapStart + 1 }
        let gapSummary: String = gaps.isEmpty ? "none" : "\(gaps.count) (\(missingResidues) missing residues — see 'prep missing')"
        let phText = String(format: "%.1f", vm.molecules.protonationPH)
        let prepSummary: String = vm.preparationLog.isEmpty
            ? "none (raw structure)"
            : "\(vm.preparationLog.count) step(s), pH \(phText) — see 'prep report'"
        let coil = amino.count - helixResidues - strandResidues
        let title: String = protein.title.isEmpty ? protein.name : "\(protein.name) — \(protein.title)"

        var lines: [String] = [title]
        lines.append("Atoms         \(atoms.count) (\(atoms.count - hydrogens) heavy, \(hydrogens) H)")
        lines.append("Residues      \(protein.residues.count): \(amino.count) amino acids, \(het.count) hetero, \(waters.count) waters")
        lines.append("Chains        " + chainSummary)
        lines.append("HET groups    " + hetSummary)
        lines.append("Mass          " + String(format: "%.1f kDa", mass / 1000) + massNote)
        lines.append("Charge        formal \(String(format: "%+d", formal)), partial \(String(format: "%+.2f", partial)) on \(charged) atoms")
        lines.append("Sec. struct.  \(helices) helices (\(helixResidues) residues), \(strands) strands (\(strandResidues) residues), \(coil) coil/turn")
        lines.append("Gaps          " + gapSummary)
        lines.append("Preparation   " + prepSummary)
        if let ligand = vm.molecules.ligand {
            lines.append("Ligand        \(ligand.name) (\(ligand.heavyAtomCount) heavy atoms)")
        }
        inv.emit(.output, lines.joined(separator: "\n"))
    }

    // MARK: Shared file helpers

    /// Throws a console error naming the path when the file cannot be read.
    static func requireReadable(_ url: URL) throws {
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw ConsoleError("no readable file at \(url.path)")
        }
    }

    /// Creates the parent folder and runs the write, turning failures into a console error naming the path.
    @discardableResult
    static func writeFile<T>(_ url: URL, _ write: () throws -> T) throws -> T {
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            return try write()
        } catch {
            throw ConsoleError("could not write \(url.path): \(error.localizedDescription)")
        }
    }
}

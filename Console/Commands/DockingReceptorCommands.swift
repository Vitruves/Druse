// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

// MARK: - flex, constraint

extension DruseInterpreter {

    // MARK: flex

    /// `flex`: the receptor side chains the search may rotate (induced fit).
    static func flexCommand() -> ConsoleCommand {
        let residues = ConsoleParameter(name: "residues", type: .string,
                                        summary: "Residues as [chain:][name]number, e.g. TYR95 A:ASP189 120",
                                        isVariadic: true)
        return ConsoleCommand(
            name: "flex",
            summary: "Receptor flexibility: sidechains the docking search may rotate (induced fit)",
            category: .docking,
            forms: [
                ConsoleForm(path: ["add"], summary: "Make sidechains fully flexible (max \(FlexibleResidueConfig.maxFlexibleResidues); matched by residue number)",
                            arguments: [residues]) { inv in
                    try addFlexResidues(inv)
                },
                ConsoleForm(path: ["remove"], summary: "Make sidechains rigid again", arguments: [residues]) { inv in
                    try removeFlexResidues(inv)
                },
                ConsoleForm(path: ["auto"], summary: "Soft flexibility of the pocket-lining sidechains chosen at launch (replaces manual residues)",
                            arguments: [ConsoleParameter(name: "state", type: .bool, summary: "on (default) or off", isOptional: true)]) { inv in
                    let vm = inv.viewModel
                    let on = inv.bool("state") ?? true
                    vm.docking.flexibleResidueConfig.autoFlex = on
                    if on { vm.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll() }
                    inv.emit(.success, on ? "Soft receptor flexibility on — " + autoFlexPreview(vm) : "Soft receptor flexibility off")
                },
                ConsoleForm(path: ["list"], summary: "Show the flexibility mode and the flexible residues") { inv in
                    inv.emit(.output, flexReport(inv.viewModel))
                },
                ConsoleForm(path: ["clear"], summary: "Rigid receptor: remove every flexible residue and turn soft flexibility off") { inv in
                    inv.viewModel.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll()
                    inv.viewModel.docking.flexibleResidueConfig.autoFlex = false
                    inv.emit(.success, "Receptor is rigid")
                }
            ],
            examples: ["flex add TYR95 ASP189", "flex add A:120", "flex remove TYR95", "flex auto", "flex auto off",
                       "flex list", "flex clear"]
        )
    }

    /// Marks residues fully flexible (refusing ones without rotatable side chains, up to the engine's limit) and turns soft flexibility off.
    private static func addFlexResidues(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        guard let protein = vm.molecules.protein else { return }
        var config = vm.docking.flexibleResidueConfig
        var added: [String] = []
        for index in try DockingResidueSpec.resolveAll(inv.requiredString("residues"), in: protein) {
            let residue = protein.residues[index]
            guard RotamerLibrary.rotamers(for: residue.name) != nil else {
                throw ConsoleError("\(DockingResidueSpec.label(residue)) has no rotatable sidechain (\(residue.name))")
            }
            guard !config.flexibleResidueIndices.contains(residue.sequenceNumber) else { continue }
            guard config.flexibleResidueIndices.count < FlexibleResidueConfig.maxFlexibleResidues else {
                throw ConsoleError("at most \(FlexibleResidueConfig.maxFlexibleResidues) flexible residues — 'flex remove' one first")
            }
            config.flexibleResidueIndices.append(residue.sequenceNumber)
            added.append(DockingResidueSpec.label(residue))
        }
        if config.autoFlex {
            config.autoFlex = false
            inv.emit(.warning, "soft flexibility turned off: the listed residues are fully flexible instead")
        }
        vm.docking.flexibleResidueConfig = config
        inv.emit(.success, added.isEmpty ? "Already flexible — \(config.flexibleResidueIndices.count) residue(s)"
                 : "Flexible: \(added.joined(separator: ", ")) (\(config.flexibleResidueIndices.count)/\(FlexibleResidueConfig.maxFlexibleResidues))")
    }

    /// Removes residues from the flexible set.
    private static func removeFlexResidues(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        guard let protein = vm.molecules.protein else { return }
        let numbers = Set(try DockingResidueSpec.resolveAll(inv.requiredString("residues"), in: protein)
            .map { protein.residues[$0].sequenceNumber })
        let before = vm.docking.flexibleResidueConfig.flexibleResidueIndices.count
        vm.docking.flexibleResidueConfig.flexibleResidueIndices.removeAll { numbers.contains($0) }
        let removed = before - vm.docking.flexibleResidueConfig.flexibleResidueIndices.count
        inv.emit(.success, "Removed \(removed) flexible residue(s); \(vm.docking.flexibleResidueConfig.flexibleResidueIndices.count) left")
    }

    /// Lists the flexible residues and whether flexible docking is on.
    private static func flexReport(_ vm: AppViewModel) -> String {
        let config = vm.docking.flexibleResidueConfig
        if config.autoFlex {
            return "Soft flexibility (weight \(DockingSettings.format(FlexibleResidueConfig.softFlexWeight)), "
                + "chi step \(DockingSettings.format(FlexibleResidueConfig.softChiStep)) rad): " + autoFlexPreview(vm)
        }
        guard !config.flexibleResidueIndices.isEmpty else {
            return "Rigid receptor — 'flex add <residues…>' or 'flex auto' adds flexibility"
        }
        let rows = config.flexibleResidueIndices.map { number -> [String] in
            let residue = vm.molecules.protein?.residues.first { $0.sequenceNumber == number && $0.isStandard }
            let chi = residue.flatMap { RotamerLibrary.rotamers(for: $0.name)?.chiAngles.count } ?? 0
            return [residue.map(DockingResidueSpec.label) ?? "\(number)", "\(chi)"]
        }
        return "Full flexibility (chi step \(DockingSettings.format(FlexibleResidueConfig.fullChiStep)) rad):\n"
            + ConsoleTable.render(["Residue", "Chi"], rows, rightAligned: [1])
    }

    /// The residues soft flexibility would pick for the selected pocket.
    private static func autoFlexPreview(_ vm: AppViewModel) -> String {
        guard let protein = vm.molecules.protein, let pocket = vm.docking.selectedPocket else {
            return "residues are chosen when docking starts (no pocket selected yet)"
        }
        let numbers = FlexibleResidueConfig.autoSelectResidues(
            protein: protein.atoms, pocket: (center: pocket.center, residueIndices: pocket.residueIndices)
        )
        guard !numbers.isEmpty else { return "no rotatable residue lines the selected pocket" }
        let labels = numbers.map { number in
            protein.residues.first { $0.sequenceNumber == number && $0.isStandard }.map(DockingResidueSpec.label) ?? "\(number)"
        }
        return "\(labels.count) pocket residue(s): \(labels.joined(separator: ", "))"
    }

    // MARK: constraint

    /// `constraint`: pharmacophore constraints the docked pose must (hard) or should (soft) satisfy.
    static func constraintCommand() -> ConsoleCommand {
        let index = ConsoleParameter(name: "n", type: .int, summary: "1-based index from 'constraint list'")
        return ConsoleCommand(
            name: "constraint",
            summary: "Pharmacophore constraints: interactions the docked pose must (hard) or should (soft) make",
            category: .docking,
            forms: [
                ConsoleForm(path: ["add"], summary: "Add a receptor-side constraint at a residue, an atom of it, or a point",
                            arguments: [ConsoleParameter(name: "type", type: .choice(DockingNames.names(DockingNames.constraintTypes)),
                                                         summary: "Interaction the ligand must make")],
                            options: [
                                ConsoleParameter(name: "residue", type: .string, summary: "Target residue as [chain:][name]number, e.g. A:ASP189"),
                                ConsoleParameter(name: "atom", type: .string, summary: "Atom name within the residue (e.g. OD1); implies scope=atom"),
                                ConsoleParameter(name: "position", type: .vec3, summary: "Target point in Å instead of a residue"),
                                ConsoleParameter(name: "scope", type: .choice(["atom", "residue"]),
                                                 summary: "residue = satisfied by any relevant atom of the residue (default without atom=)"),
                                ConsoleParameter(name: "radius", type: .float, summary: "Distance within which the interaction counts, 1.5–10 Å (default 3.5)"),
                                ConsoleParameter(name: "strength", type: .choice(["soft", "hard"]),
                                                 summary: "soft = penalty that biases the search (default), hard = poses violating it are rejected"),
                                ConsoleParameter(name: "weight", type: .float, summary: "Soft penalty in kcal/mol/Å², 0.1–100 (default 5)")
                            ]) { inv in
                    try addConstraint(inv)
                },
                ConsoleForm(path: ["list"], summary: "List constraints with their state, target and strength") { inv in
                    inv.emit(.output, constraintTable(inv.viewModel))
                },
                ConsoleForm(path: ["enable"], summary: "Use a constraint in the next docking run", arguments: [index]) { inv in
                    try setConstraint(inv, enabled: true)
                },
                ConsoleForm(path: ["disable"], summary: "Keep a constraint but ignore it when docking", arguments: [index]) { inv in
                    try setConstraint(inv, enabled: false)
                },
                ConsoleForm(path: ["remove"], summary: "Delete a constraint", arguments: [index]) { inv in
                    let vm = inv.viewModel
                    let i = try constraintIndex(inv)
                    let removed = vm.docking.pharmacophoreConstraints.remove(at: i)
                    vm.pushToRenderer()
                    inv.emit(.success, "Removed constraint #\(i + 1) (\(removed.interactionType.rawValue) → \(removed.targetLabel))")
                },
                ConsoleForm(path: ["clear"], summary: "Delete every constraint") { inv in
                    let count = inv.viewModel.docking.pharmacophoreConstraints.count
                    inv.viewModel.docking.pharmacophoreConstraints.removeAll()
                    inv.viewModel.pushToRenderer()
                    inv.emit(.success, "Removed \(count) constraint(s)")
                }
            ],
            examples: ["constraint add salt-bridge residue=ASP189 radius=4", "constraint add acceptor residue=A:SER195 atom=OG strength=hard",
                       "constraint add hydrophobic position=12.1,4.5,-3.2 weight=10", "constraint list",
                       "constraint disable 2", "constraint remove 1", "constraint clear"]
        )
    }

    /// Adds an interaction or distance constraint on a protein residue or atom, with its strength and radius.
    private static func addConstraint(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        guard let protein = vm.molecules.protein,
              let type = ConsoleNames.lookup(inv.string("type"), in: DockingNames.constraintTypes) else { return }
        let radius = Float(inv.double("radius") ?? 3.5)
        guard (1.5...10).contains(radius) else { throw ConsoleError("radius must be between 1.5 and 10 Å") }
        let weight = Float(inv.double("weight") ?? 5)
        guard (0.1...100).contains(weight) else { throw ConsoleError("weight must be between 0.1 and 100 kcal/mol/Å²") }
        if inv.has("weight"), inv.string("strength") == "hard" {
            inv.emit(.warning, "weight= is ignored for strength=hard")
        }
        let strength: ConstraintStrength = inv.string("strength") == "hard" ? .hard : .soft(kcalPerAngstromSq: weight)

        var constraint: PharmacophoreConstraintDef
        if let position = inv.vec3("position") {
            guard !inv.has("residue"), !inv.has("atom") else {
                throw ConsoleError("give either position= or residue= (with optional atom=), not both")
            }
            constraint = PharmacophoreConstraintDef(
                targetScope: .atom, interactionType: type, strength: strength, distanceThreshold: radius,
                sourceType: .receptor, residueName: "point \(ConsoleFormat.vector(position))"
            )
            constraint.targetPositions = [position]
        } else {
            guard let spec = inv.string("residue") else {
                throw ConsoleError("constraint add needs residue=[chain:][name]number or position=x,y,z")
            }
            let residueIndex = try DockingResidueSpec.resolveOne(spec, in: protein)
            let residue = protein.residues[residueIndex]
            let scope: TargetScope = (inv.string("scope") ?? (inv.has("atom") ? "atom" : "residue")) == "atom" ? .atom : .residue
            var atomIndex: Int?
            var atomName: String?
            if let name = inv.string("atom") {
                guard let found = residue.atomIndices.first(where: {
                    protein.atoms[$0].name.trimmingCharacters(in: .whitespaces).caseInsensitiveCompare(name) == .orderedSame
                }) else {
                    let names = residue.atomIndices.map { protein.atoms[$0].name.trimmingCharacters(in: .whitespaces) }
                    throw ConsoleError("\(DockingResidueSpec.label(residue)) has no atom '\(name)' — atoms: \(names.joined(separator: " "))")
                }
                atomIndex = found
                atomName = protein.atoms[found].name.trimmingCharacters(in: .whitespaces)
            } else if scope == .atom {
                throw ConsoleError("scope=atom needs atom=<name>")
            }
            constraint = PharmacophoreConstraintDef(
                targetScope: scope, interactionType: type, strength: strength, distanceThreshold: radius,
                sourceType: .receptor, proteinAtomIndex: atomIndex, residueIndex: residueIndex,
                chainID: residue.chainID, residueName: "\(residue.name) \(residue.sequenceNumber)", atomName: atomName
            )
        }
        vm.addPharmacophoreConstraint(constraint)
        guard let added = vm.docking.pharmacophoreConstraints.last, !added.targetPositions.isEmpty else {
            vm.docking.pharmacophoreConstraints.removeLast()
            vm.pushToRenderer()
            throw ConsoleError("no target atom found for this constraint")
        }
        inv.emit(.success, "Constraint #\(vm.docking.pharmacophoreConstraints.count): \(type.rawValue) → \(added.targetLabel), "
                 + "\(added.targetPositions.count) target point(s), within \(DockingSettings.format(radius)) Å, \(strength.label)")
    }

    /// Resolves the 1-based constraint number given to `constraint enable|disable|remove`.
    private static func constraintIndex(_ inv: ConsoleInvocation) throws -> Int {
        let constraints = inv.viewModel.docking.pharmacophoreConstraints
        let n = inv.requiredInt("n")
        guard constraints.indices.contains(n - 1) else {
            throw ConsoleError(constraints.isEmpty ? "no constraints — 'constraint add' creates one"
                               : "no constraint #\(n) — \(constraints.count) defined")
        }
        return n - 1
    }

    /// Turns one constraint on or off without removing it.
    private static func setConstraint(_ inv: ConsoleInvocation, enabled: Bool) throws {
        let vm = inv.viewModel
        let i = try constraintIndex(inv)
        vm.docking.pharmacophoreConstraints[i].isEnabled = enabled
        vm.pushToRenderer()
        inv.emit(.success, "Constraint #\(i + 1) \(enabled ? "enabled" : "disabled")")
    }

    /// Table of constraints: state, type, target, scope, radius, strength and point count.
    private static func constraintTable(_ vm: AppViewModel) -> String {
        let constraints = vm.docking.pharmacophoreConstraints
        guard !constraints.isEmpty else { return "No constraints — 'constraint add <type> residue=…' creates one" }
        let rows = constraints.enumerated().map { i, c in
            [
                "\(i + 1)", c.isEnabled ? "on" : "off",
                DockingNames.name(of: c.interactionType, in: DockingNames.constraintTypes),
                c.targetLabel + (c.chainID.map { $0.isEmpty ? "" : " (\($0))" } ?? ""),
                c.sourceType == .ligand ? "ligand" : (c.targetScope == .atom ? "atom" : "residue"),
                DockingSettings.format(c.distanceThreshold),
                c.strength.label,
                "\(c.targetPositions.count)"
            ]
        }
        return ConsoleTable.render(["#", "On", "Type", "Target", "Scope", "Radius Å", "Strength", "Points"], rows, rightAligned: [0, 5, 7])
    }
}

// MARK: - Residue references

/// Residues typed as `[chain:][name]number` — `TYR95`, `A:ASP189`, `120`, `B:42` — the
/// spelling shared by `flex`, `constraint` and `pocket residues`.
@MainActor
enum DockingResidueSpec {

    /// `A:ASP189` label of a residue (chain omitted when empty).
    static func label(_ residue: Residue) -> String {
        (residue.chainID.isEmpty ? "" : residue.chainID + ":") + residue.name + "\(residue.sequenceNumber)"
    }

    /// Indices into `protein.residues` for every space- or comma-separated reference;
    /// each reference must match at least one residue.
    static func resolveAll(_ text: String, in protein: Molecule) throws -> [Int] {
        let words = text.split(whereSeparator: { $0 == " " || $0 == "," }).map(String.init)
        guard !words.isEmpty else { throw ConsoleError("no residues given") }
        var indices: [Int] = []
        for word in words {
            for index in try matches(word, in: protein) where !indices.contains(index) {
                indices.append(index)
            }
        }
        return indices
    }

    /// The single residue a reference names; ambiguous references (same number in several
    /// chains) must add the chain.
    static func resolveOne(_ word: String, in protein: Molecule) throws -> Int {
        let found = try matches(word, in: protein)
        guard found.count == 1 else {
            let labels = found.map { label(protein.residues[$0]) }.joined(separator: ", ")
            throw ConsoleError("'\(word)' matches \(found.count) residues (\(labels)) — add the chain, e.g. \(label(protein.residues[found[0]]))")
        }
        return found[0]
    }

    /// Indices of the protein residues a `[chain:]NAMEnumber` word designates (several when the chain is omitted).
    private static func matches(_ word: String, in protein: Molecule) throws -> [Int] {
        var rest = Substring(word)
        var chain: String?
        if let colon = rest.firstIndex(of: ":") {
            chain = String(rest[..<colon])
            rest = rest[rest.index(after: colon)...]
        }
        let name = rest.prefix { $0.isLetter }
        let digits = rest.dropFirst(name.count)
        guard let number = Int(digits) else {
            throw ConsoleError("'\(word)' is not a residue reference — expected [chain:][name]number, e.g. A:ASP189 or 95")
        }
        let found = protein.residues.indices.filter { i in
            let residue = protein.residues[i]
            return !residue.isWater
                && residue.sequenceNumber == number
                && (chain.map { $0.caseInsensitiveCompare(residue.chainID) == .orderedSame } ?? true)
                && (name.isEmpty || name.caseInsensitiveCompare(residue.name) == .orderedSame)
        }
        guard !found.isEmpty else {
            let atNumber = protein.residues.filter { $0.sequenceNumber == number && !$0.isWater }.map(label)
            throw ConsoleError("no residue \(word) in \(protein.name)" + (atNumber.isEmpty ? "" : " — residue \(number) is \(atNumber.joined(separator: ", "))"))
        }
        return found
    }
}

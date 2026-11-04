// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// MARK: - export
//
// Every `export` form writes one file whose format follows the file extension. Docking poses
// and results share `DockingPoseRecord`, so the SDF properties and the CSV columns agree.
// `top=N` and `cluster-representatives=on` select which ranked poses are written; exported
// poses keep their rank from 'results'.

extension DruseInterpreter {

    /// `export`: every file the workflow produces (poses, results, structures, interactions, analogs, hits, batch ranking, session log).
    static func exportCommands() -> [ConsoleCommand] {
        let file = { (kind: String) in ConsoleParameter(name: "file", type: .path, summary: "Output file (\(kind))") }
        let poseNumber = ConsoleParameter(name: "n", type: .int, summary: "1-based pose rank from 'results'")
        // `export ligands` is `ligand db export` under the export verb: same arguments, options and code.
        let ligandExport = ligandCommands().first { $0.name == "ligand" }?.forms.first { $0.path == ["db", "export"] }
        let selection = [
            ConsoleParameter(name: "top", type: .int, summary: "Write only the N best poses (after cluster filtering)"),
            ConsoleParameter(name: "cluster-representatives", type: .bool, summary: "Write only the best pose of each cluster (default off)")
        ]
        var forms: [ConsoleForm] = [
            ConsoleForm(path: ["poses"], summary: "Ranked docked poses: SDF with every score term, or multi-model PDB (default top \(AppViewModel.exportedPoseLimit))",
                        arguments: [file(".sdf or .pdb")],
                        options: selection + [
                            ConsoleParameter(name: "protein", type: .bool, summary: "PDB only: write the protein once, poses as residues of chain X (default off)")
                        ]) { inv in
                try exportPoses(inv)
            },
            ConsoleForm(path: ["results"], summary: "Results table as CSV: score, every term incl. intra-and-other, cluster, validity, refinement",
                        arguments: [file(".csv")], options: selection) { inv in
                try exportResults(inv)
            },
            ConsoleForm(path: ["pose"], summary: "One docked ligand pose without the protein (.sdf with scores, or .pdb)",
                        arguments: [poseNumber, file(".sdf or .pdb")]) { inv in
                try exportPose(inv)
            },
            ConsoleForm(path: ["complex"], summary: "One docked pose together with the protein as a single PDB",
                        arguments: [poseNumber, file(".pdb")]) { inv in
                let vm = inv.viewModel
                let url = inv.requiredURL("file")
                try requireExtension(url, ["pdb", "ent"], what: "complex")
                try requireProtein(vm)
                let index = try poseIndex(inv)
                try writeFile(url) { try vm.writeDockingPosePDB(at: index, to: url) }
                inv.emit(.success, "Wrote pose #\(index + 1) with \(vm.molecules.protein?.name ?? "the protein") to \(url.path)")
            },
            ConsoleForm(path: ["protein"], summary: "The current (prepared) protein as PDB",
                        arguments: [file(".pdb")],
                        options: [
                            ConsoleParameter(name: "hydrogens", type: .bool, summary: "Include hydrogen atoms (default on)"),
                            ConsoleParameter(name: "ligand", type: .bool, summary: "Append the active ligand at its current coordinates as HETATM (default off)")
                        ]) { inv in
                try exportProtein(inv)
            },
            ConsoleForm(path: ["ligand"], summary: "The active ligand at its current coordinates (.sdf/.mol with all atoms, or .pdb)",
                        arguments: [file(".sdf, .mol or .pdb")]) { inv in
                try exportActiveLigand(inv)
            },
            ConsoleForm(path: ["interactions"], summary: "Protein–ligand interactions of one pose as CSV (type, distance, atoms, residue)",
                        arguments: [file(".csv")],
                        options: [
                            ConsoleParameter(name: "pose", type: .int, summary: "1-based pose rank (default 1)"),
                            ConsoleParameter(name: "types", type: .choice(["all", "scored"]),
                                             summary: "all detected contact types, or only those the scoring function accounts for (default all)")
                        ]) { inv in
                try exportInteractions(inv)
            },
            ConsoleForm(path: ["analogs"], summary: "Lead-optimization analogs: CSV (properties, deltas, ADMET, score) or SDF (3D, docked pose when available)",
                        arguments: [file(".csv or .sdf")],
                        options: [
                            ConsoleParameter(name: "status", type: .choice(["all", "passed", "docked"]),
                                             summary: "all, passed (not filtered or failed) or docked analogs only (default all)")
                        ]) { inv in
                try await exportAnalogs(inv)
            },
            ConsoleForm(path: ["hits"], summary: "Virtual-screening hits from 'screen': CSV table or SDF with best poses",
                        arguments: [file(".csv or .sdf")],
                        options: [ConsoleParameter(name: "top", type: .int, summary: "SDF only: the N best hits (default 100)")]) { inv in
                try exportScreeningHits(inv)
            },
            ConsoleForm(path: ["batch"], summary: "Batch-docking ranking from 'dock batch' as CSV (best pose terms per ligand)",
                        arguments: [file(".csv")]) { inv in
                try exportBatchRanking(inv)
            },
            ConsoleForm(path: ["session-log"], summary: "This session's command history and activity log as text",
                        arguments: [file(".txt or .log")]) { inv in
                try exportSessionLog(inv)
            }
        ]
        if let delegate = ligandExport {
            forms.insert(ConsoleForm(path: ["ligands"], summary: "The Ligand Database — same as 'ligand db export'",
                                     arguments: delegate.arguments, options: delegate.options, run: delegate.run),
                         at: forms.firstIndex { $0.path == ["interactions"] } ?? forms.endIndex)
        }
        return [
            ConsoleCommand(
                name: "export",
                summary: "Write poses, results, structures, interactions, analogs and logs to files",
                category: .export,
                forms: forms,
                examples: [
                    "export poses out/poses.sdf top=10",
                    "export poses out/representatives.pdb cluster-representatives=on",
                    "export results out/results.csv",
                    "export complex 1 out/best_complex.pdb",
                    "export pose 3 out/pose3.sdf",
                    "export protein out/prepared.pdb hydrogens=off",
                    "export ligands out/library.sdf subset=prepared",
                    "export interactions out/contacts.csv pose=1",
                    "export analogs out/analogs.csv status=docked",
                    "export session-log out/session.log"
                ]
            )
        ]
    }

    // MARK: Docking poses and results

    /// Ranked poses chosen by `top=` and `cluster-representatives=`, with their 1-based ranks.
    private static func selectedPoses(_ inv: ConsoleInvocation, defaultTop: Int?) throws -> [(rank: Int, result: DockingResult)] {
        let results = inv.viewModel.docking.dockingResults
        guard !results.isEmpty else { throw ConsoleError("no docking results — run 'dock' first") }
        if let top = inv.int("top"), top < 1 { throw ConsoleError("top must be at least 1") }
        var ranked = results.enumerated().map { (rank: $0.offset + 1, result: $0.element) }
        if inv.bool("cluster-representatives") == true {
            var seen = Set<Int>()
            ranked = ranked.filter { seen.insert($0.result.clusterID).inserted }
        }
        if let top = inv.int("top") ?? defaultTop { ranked = Array(ranked.prefix(top)) }
        return ranked
    }

    /// The ligand the poses were docked from (its heavy atoms are the pose template).
    private static func dockedLigand(_ vm: AppViewModel) throws -> Molecule {
        guard let ligand = vm.docking.originalDockingLigand ?? vm.molecules.ligand else {
            throw ConsoleError("the docked ligand is no longer loaded")
        }
        return ligand
    }

    /// Writes the ranked poses as SDF with every score term, or as a multi-model PDB.
    private static func exportPoses(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let url = inv.requiredURL("file")
        let format = try requireExtension(url, ["sdf", "pdb", "ent"], what: "poses")
        let poses = try selectedPoses(inv, defaultTop: AppViewModel.exportedPoseLimit)
        let ligand = try dockedLigand(vm)
        let method = vm.docking.scoringMethod
        if format == "sdf" {
            if inv.has("protein") { inv.emit(.warning, "protein= applies to PDB output only — ignored for SDF") }
            try writeFile(url) { try SDFWriter.save(SDFWriter.writeDockingPoses(poses, ligand: ligand, method: method), to: url) }
        } else {
            let models = poses.map { pose in
                (rank: pose.rank, positions: pose.result.transformedAtomPositions,
                 scoreLine: String(format: "Druse pose #%d  score=%.2f %@  cluster=%d", pose.rank,
                                   pose.result.displayScore(method: method), method.unitLabel, pose.result.clusterID))
            }
            let title = "Druse docking poses of \(ligand.name) (\(ConsoleNames.scoring(method)))"
            let pdb = inv.bool("protein") == true
                ? PDBWriter.combinedMultiPose(protein: vm.molecules.protein, ligandTemplate: ligand, poses: models, title: title)
                : PDBWriter.multiModelPoses(ligandTemplate: ligand, poses: models, title: title)
            try writeFile(url) { try PDBWriter.save(pdb, to: url) }
        }
        inv.emit(.success, "Wrote \(poses.count) of \(vm.docking.dockingResults.count) poses to \(url.path)")
    }

    /// Writes the results table as CSV.
    private static func exportResults(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let url = inv.requiredURL("file")
        try requireExtension(url, ["csv"], what: "results")
        let poses = try selectedPoses(inv, defaultTop: nil)
        let method = vm.docking.scoringMethod
        let ligandName = (try? dockedLigand(vm))?.name ?? "ligand"
        var rows: [[String]] = []
        for pose in poses {
            let fields = DockingPoseRecord.fields(rank: pose.rank, result: pose.result, method: method)
            if rows.isEmpty { rows.append(["Ligand"] + fields.map(\.name)) }
            rows.append([ligandName] + fields.map(\.value))
        }
        try writeFile(url) { try CSV.text(rows).write(to: url, atomically: true, encoding: .utf8) }
        inv.emit(.success, "Wrote \(poses.count) results (\(rows.first?.count ?? 0) columns) to \(url.path)")
    }

    /// Resolves a 1-based pose number against the current results.
    private static func poseIndex(_ inv: ConsoleInvocation, argument: String = "n") throws -> Int {
        let results = inv.viewModel.docking.dockingResults
        let index = (inv.int(argument) ?? 1) - 1
        guard results.indices.contains(index) else {
            throw ConsoleError(results.isEmpty ? "no docking results — run 'dock' first"
                               : "no pose #\(index + 1) — \(results.count) available")
        }
        guard !results[index].transformedAtomPositions.isEmpty else {
            throw ConsoleError("pose #\(index + 1) has no coordinates")
        }
        return index
    }

    /// Writes one pose without the protein (`export pose`).
    private static func exportPose(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let url = inv.requiredURL("file")
        let format = try requireExtension(url, ["sdf", "mol", "pdb", "ent"], what: "pose")
        let index = try poseIndex(inv)
        let ligand = try dockedLigand(vm)
        let result = vm.docking.dockingResults[index]
        let method = vm.docking.scoringMethod
        if format == "pdb" || format == "ent" {
            let title = String(format: "Druse pose %d  score=%.2f %@", index + 1, result.displayScore(method: method), method.unitLabel)
            let pdb = PDBWriter.combinedPose(protein: nil, ligandTemplate: ligand,
                                             ligandPositions: result.transformedAtomPositions, title: title)
            try writeFile(url) { try PDBWriter.save(pdb, to: url) }
        } else {
            try writeFile(url) {
                try SDFWriter.save(SDFWriter.writeDockingPoses([(rank: index + 1, result: result)], ligand: ligand, method: method), to: url)
            }
        }
        inv.emit(.success, "Wrote pose #\(index + 1) (\(ConsoleFormat.score(result, method))) to \(url.path)")
    }

    // MARK: Structures

    /// Writes the current protein as PDB, optionally without hydrogens or with the active ligand.
    private static func exportProtein(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        let url = inv.requiredURL("file")
        try requireExtension(url, ["pdb", "ent"], what: "protein")
        guard let protein = vm.molecules.protein else { return }
        let ligand = inv.bool("ligand") == true ? vm.molecules.ligand : nil
        if inv.bool("ligand") == true, ligand == nil {
            throw ConsoleError("ligand=on needs an active ligand — load or build one first")
        }
        let hydrogens = inv.bool("hydrogens") ?? true
        let pdb = PDBWriter.combinedPose(
            protein: protein, ligandTemplate: ligand,
            ligandPositions: ligand?.atoms.filter { $0.element != .H }.map(\.position) ?? [],
            title: protein.title.isEmpty ? protein.name : protein.title,
            proteinHydrogens: hydrogens
        )
        try writeFile(url) { try PDBWriter.save(pdb, to: url) }
        let atoms = hydrogens ? protein.atomCount : protein.heavyAtomCount
        inv.emit(.success, "Wrote \(atoms) protein atoms" + (ligand.map { " and \($0.name)" } ?? "") + " to \(url.path)")
    }

    /// Writes the active ligand at its current coordinates.
    private static func exportActiveLigand(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let ligand = vm.molecules.ligand else {
            throw ConsoleError("no active ligand — use 'ligand smiles', 'ligand use' or 'load <file.sdf>'")
        }
        let url = inv.requiredURL("file")
        let format = try requireExtension(url, ["sdf", "mol", "pdb", "ent"], what: "ligand")
        if format == "pdb" || format == "ent" {
            let pdb = PDBWriter.combinedPose(protein: nil, ligandTemplate: ligand,
                                             ligandPositions: ligand.atoms.filter { $0.element != .H }.map(\.position),
                                             title: ligand.name)
            try writeFile(url) { try PDBWriter.save(pdb, to: url) }
            inv.emit(.success, "Wrote \(ligand.name) (\(ligand.heavyAtomCount) heavy atoms) to \(url.path)")
        } else {
            var properties: [String: String] = [:]
            if let smiles = ligand.smiles, !smiles.isEmpty { properties["SMILES"] = smiles }
            let sdf = SDFWriter.molBlock(name: ligand.name, atoms: ligand.atoms, bonds: ligand.bonds, properties: properties)
            try writeFile(url) { try SDFWriter.save(sdf, to: url) }
            inv.emit(.success, "Wrote \(ligand.name) (\(ligand.atomCount) atoms) to \(url.path)")
        }
    }

    // MARK: Interactions

    /// Writes the protein–ligand interactions of one pose as CSV.
    private static func exportInteractions(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        try requireProtein(vm)
        let url = inv.requiredURL("file")
        try requireExtension(url, ["csv"], what: "interactions")
        let index = try poseIndex(inv, argument: "pose")
        guard let protein = vm.molecules.protein else { return }
        let ligand = try dockedLigand(vm)
        let result = vm.docking.dockingResults[index]
        let template = SDFWriter.heavyAtomTemplate(ligand)
        let proteinAtoms = protein.atoms.filter { $0.element != .H }
        // Drusina accounts for every interaction type, so it serves as "report all types".
        let method: ScoringMethod = inv.string("types") == "scored" ? vm.docking.scoringMethod : .drusina
        let interactions = InteractionDetector.detect(
            ligandAtoms: template.atoms, ligandPositions: result.transformedAtomPositions,
            proteinAtoms: proteinAtoms, ligandBonds: template.bonds, scoringMethod: method
        )
        let f = { (value: Float) in String(format: "%.3f", value) }
        var rows: [[String]] = [["Pose", "Type", "Distance", "Ligand_Atom", "Ligand_Element", "Ligand_Atom_Name",
                                 "Chain", "Residue", "Residue_Number", "Protein_Atom", "Protein_Element",
                                 "Ligand_X", "Ligand_Y", "Ligand_Z", "Protein_X", "Protein_Y", "Protein_Z"]]
        for interaction in interactions.sorted(by: { ($0.type.rawValue, $0.distance) < ($1.type.rawValue, $1.distance) }) {
            let ligandAtom = template.atoms.indices.contains(interaction.ligandAtomIndex) ? template.atoms[interaction.ligandAtomIndex] : nil
            let proteinAtom = proteinAtoms.indices.contains(interaction.proteinAtomIndex) ? proteinAtoms[interaction.proteinAtomIndex] : nil
            rows.append([
                "\(index + 1)", interaction.type.label, f(interaction.distance),
                "\(interaction.ligandAtomIndex + 1)", ligandAtom?.element.symbol ?? "", ligandAtom?.name ?? "",
                proteinAtom?.chainID ?? "", proteinAtom?.residueName ?? "", proteinAtom.map { "\($0.residueSeq)" } ?? "",
                proteinAtom?.name ?? "", proteinAtom?.element.symbol ?? "",
                f(interaction.ligandPosition.x), f(interaction.ligandPosition.y), f(interaction.ligandPosition.z),
                f(interaction.proteinPosition.x), f(interaction.proteinPosition.y), f(interaction.proteinPosition.z)
            ])
        }
        try writeFile(url) { try CSV.text(rows).write(to: url, atomically: true, encoding: .utf8) }
        let counts = Dictionary(grouping: interactions, by: \.type.label).map { "\($0.value.count) \($0.key)" }.sorted()
        inv.emit(.success, "Wrote \(interactions.count) interactions of pose #\(index + 1) to \(url.path)"
                 + (counts.isEmpty ? "" : " (" + counts.joined(separator: ", ") + ")"))
    }

    // MARK: Analogs

    /// Writes the lead-optimization analogs as CSV, or as SDF with their docked poses when available.
    private static func exportAnalogs(_ inv: ConsoleInvocation) async throws {
        let vm = inv.viewModel
        let url = inv.requiredURL("file")
        let format = try requireExtension(url, ["csv", "sdf"], what: "analogs")
        var analogs = vm.leadOpt.analogs
        switch inv.string("status") {
        case "passed": analogs = analogs.filter { $0.status != .filtered && $0.status != .failed }
        case "docked": analogs = analogs.filter { $0.status == .docked }
        default: break
        }
        guard !analogs.isEmpty else {
            throw ConsoleError(vm.leadOpt.analogs.isEmpty ? "no analogs — run 'leadopt generate' first"
                               : "no analog matches status=\(inv.string("status") ?? "all")")
        }
        let method = vm.leadOpt.scoringMethod
        if format == "csv" {
            let rows = [AnalogRecord.header] + analogs.map { AnalogRecord.row($0, method: method) }
            try writeFile(url) { try CSV.text(rows).write(to: url, atomically: true, encoding: .utf8) }
            inv.emit(.success, "Wrote \(analogs.count) analogs to \(url.path)")
            return
        }
        var blocks: [String] = []
        var skipped: [String] = []
        for analog in analogs {
            guard let structure = await AnalogRecord.structure(of: analog) else {
                skipped.append(analog.name)
                continue
            }
            let properties = Dictionary(
                zip(AnalogRecord.header, AnalogRecord.row(analog, method: method)).filter { !$0.1.isEmpty },
                uniquingKeysWith: { first, _ in first }
            )
            blocks.append(SDFWriter.molBlock(name: analog.name, atoms: structure.atoms, bonds: structure.bonds, properties: properties))
        }
        guard !blocks.isEmpty else { throw ConsoleError("no analog could be embedded in 3D") }
        try writeFile(url) { try SDFWriter.save(blocks.joined(), to: url) }
        if !skipped.isEmpty {
            inv.emit(.warning, "\(skipped.count) analog(s) could not be embedded in 3D and were skipped: \(skipped.prefix(5).joined(separator: ", "))")
        }
        inv.emit(.success, "Wrote \(blocks.count) analogs to \(url.path)")
    }

    // MARK: Screening and batch

    /// Writes the virtual-screening hits as CSV or SDF.
    private static func exportScreeningHits(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        guard let pipeline = vm.docking.screeningPipeline, !vm.docking.screeningHits.isEmpty else {
            throw ConsoleError("no screening hits — run 'screen <library>' first")
        }
        let url = inv.requiredURL("file")
        let format = try requireExtension(url, ["csv", "sdf"], what: "hits")
        if let top = inv.int("top"), top < 1 { throw ConsoleError("top must be at least 1") }
        if format == "csv" {
            if inv.has("top") { inv.emit(.warning, "top= applies to SDF output only — the CSV lists every hit") }
            try writeFile(url) { try pipeline.exportCSV().write(to: url, atomically: true, encoding: .utf8) }
            inv.emit(.success, "Wrote \(vm.docking.screeningHits.count) hits to \(url.path)")
        } else {
            let top = inv.int("top") ?? 100
            try writeFile(url) { try pipeline.exportSDF(topN: top).write(to: url, atomically: true, encoding: .utf8) }
            inv.emit(.success, "Wrote \(min(top, vm.docking.screeningHits.count)) hits to \(url.path)")
        }
    }

    /// Writes the batch-docking ranking as CSV.
    private static func exportBatchRanking(_ inv: ConsoleInvocation) throws {
        let vm = inv.viewModel
        let ranked = vm.docking.batchResults.filter { !$0.results.isEmpty }
        guard !ranked.isEmpty else { throw ConsoleError("no batch results — run 'dock batch' first") }
        let url = inv.requiredURL("file")
        try requireExtension(url, ["csv"], what: "batch")
        let method = vm.docking.scoringMethod
        var rows: [[String]] = []
        for (i, entry) in ranked.enumerated() {
            guard let best = entry.results.first else { continue }
            let fields = DockingPoseRecord.fields(rank: i + 1, result: best, method: method)
            if rows.isEmpty { rows.append(["Ligand", "Poses", "Clusters"] + fields.map(\.name)) }
            rows.append([entry.ligandName, "\(entry.results.count)", "\(Set(entry.results.map(\.clusterID)).count)"] + fields.map(\.value))
        }
        try writeFile(url) { try CSV.text(rows).write(to: url, atomically: true, encoding: .utf8) }
        inv.emit(.success, "Wrote the ranking of \(ranked.count) ligands to \(url.path)")
    }

    // MARK: Session log

    /// Commands typed in this window (when the console keeps a history) followed by the
    /// activity log, one entry per line with time, level and category.
    private static func exportSessionLog(_ inv: ConsoleInvocation) throws {
        let url = inv.requiredURL("file")
        let stamp = ISO8601DateFormatter().string(from: Date())
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        var lines = ["# Druse \(version) session log, written \(stamp)"]
        if let history = inv.interpreter.history?.entries, !history.isEmpty {
            lines += ["", "## Command history (\(history.count))"] + history
        }
        let entries = ActivityLog.shared.entries
        lines += ["", "## Activity log (\(entries.count) entries)"]
        lines += entries.map { "\($0.formattedTime)  [\($0.level.rawValue)]  [\($0.category.rawValue)]  \($0.message)" }
        try writeFile(url) { try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8) }
        inv.emit(.success, "Wrote \(entries.count) log entries to \(url.path)")
    }

    // MARK: Helpers

    /// The lower-cased extension of `url` when it is one of `allowed`; otherwise a console error
    /// listing the accepted extensions.
    @discardableResult
    private static func requireExtension(_ url: URL, _ allowed: [String], what: String) throws -> String {
        let ext = url.pathExtension.lowercased()
        guard allowed.contains(ext) else {
            var message = "cannot export \(what) as '.\(ext.isEmpty ? "(none)" : ext)' — use " + allowed.map { "." + $0 }.joined(separator: ", ")
            if ext == "mol2" { message += " (MOL2 writing is not available)" }
            if ext == "cif" || ext == "mmcif" { message += " (mmCIF writing is not available)" }
            throw ConsoleError(message)
        }
        return ext
    }
}

// MARK: - CSV and analog records

/// RFC 4180 CSV text: fields containing a comma, quote or line break are quoted.
private enum CSV {
    static func text(_ rows: [[String]]) -> String {
        rows.map { $0.map(field).joined(separator: ",") }.joined(separator: "\n") + "\n"
    }

    private static func field(_ value: String) -> String {
        guard value.contains(where: { $0 == "," || $0 == "\"" || $0.isNewline }) else { return value }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}

/// Columns of an exported lead-optimization analog, shared by the CSV rows and the SDF properties.
private enum AnalogRecord {
    /// Column names, in the order of `row`.
    static let header = [
        "Name", "SMILES", "Status", "Score", "Score_Unit", "Delta_MW", "Delta_LogP", "Delta_TPSA", "Delta_RotBonds",
        "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Lipinski", "Veber",
        "hERG", "CYP2D6", "CYP3A4", "BBB", "Metabolic_Stability", "LogS", "Drug_Likeness"
    ]

    /// One analog's values; empty strings for properties that were not computed.
    static func row(_ analog: LeadOptAnalog, method: ScoringMethod) -> [String] {
        let f2 = { (value: Float?) in value.map { String(format: "%.2f", $0) } ?? "" }
        let d = analog.descriptors
        let a = analog.admet
        let status: String
        switch analog.status {
        case .generated: status = "new"
        case .prepared: status = "3d"
        case .docked: status = "docked"
        case .filtered: status = "filtered"
        case .failed: status = "failed"
        }
        return [
            analog.name, analog.smiles, status, f2(analog.bestScore), analog.bestScore == nil ? "" : method.unitLabel,
            f2(analog.deltaMW), f2(analog.deltaLogP), f2(analog.deltaTPSA), analog.deltaRotBonds.map { "\($0)" } ?? "",
            f2(d?.molecularWeight), f2(d?.logP), f2(d?.tpsa), d.map { "\($0.hbd)" } ?? "", d.map { "\($0.hba)" } ?? "",
            d.map { "\($0.rotatableBonds)" } ?? "", d.map { $0.lipinski ? "pass" : "fail" } ?? "", d.map { $0.veber ? "pass" : "fail" } ?? "",
            f2(a?.hergLiability), f2(a?.cyp2d6Inhibition), f2(a?.cyp3a4Inhibition), f2(a?.bbbPermeability),
            f2(a?.metabolicStability), f2(a?.aqueousSolubility), a.map { String(format: "%.2f", $0.drugLikeness) } ?? ""
        ]
    }

    /// 3D structure for the SDF: the docked heavy-atom pose when the analog was docked, its
    /// prepared conformer otherwise, else a fresh RDKit conformer (off the main actor).
    static func structure(of analog: LeadOptAnalog) async -> (atoms: [Atom], bonds: [Bond])? {
        if let molecule = analog.molecule {
            guard !analog.bestPoseAtoms.isEmpty else { return (molecule.atoms, molecule.bonds) }
            var heavyIndex: [Int: Int] = [:]
            var atoms: [Atom] = []
            for (i, atom) in molecule.atoms.enumerated() where atom.element != .H {
                guard atoms.count < analog.bestPoseAtoms.count else { break }
                heavyIndex[i] = atoms.count
                var posed = atom
                posed.position = analog.bestPoseAtoms[atoms.count]
                atoms.append(posed)
            }
            let bonds = molecule.bonds.compactMap { bond -> Bond? in
                guard let a = heavyIndex[bond.atomIndex1], let b = heavyIndex[bond.atomIndex2] else { return nil }
                return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
            }
            return (atoms, bonds)
        }
        let smiles = analog.smiles
        let name = analog.name
        let embedded = await Task.detached(priority: .userInitiated) {
            RDKitBridge.smilesToMolecule(smiles: smiles, name: name, numConformers: 1, minimize: true).molecule
        }.value
        return embedded.map { ($0.atoms, $0.bonds) }
    }
}

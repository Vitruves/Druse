import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Batch Operations Extension
// Batch action panel UI and all batch processing functions (prepare, conformers, tautomers, protomers).

extension LigandDatabaseWindow {

    // MARK: - Batch Action Panel

    @ViewBuilder
    var batchActionPanel: some View {
        let selectedRows = flatRows.filter { selectedIDs.contains($0.id) }
        let nParents = selectedRows.filter { $0.kind == .parent || $0.kind == nil }.count
        let nTaut = selectedRows.filter { $0.kind == .tautomer || $0.kind == .tautomerProtomer }.count
        let nProt = selectedRows.filter { $0.kind == .protomer || $0.kind == .tautomerProtomer }.count
        let nPrepared = selectedRows.filter(\.isPrepared).count
        // Unique parent entries for Populate & Prepare
        let uniqueEntryIDs = Set(selectedRows.map(\.entryID))

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(selectedRows.count) forms selected", systemImage: "checkmark.square.fill")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Deselect All") { selectedIDs.removeAll() }
                    .font(.system(size: 9))
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                if nParents > 0 {
                    HStack(spacing: 3) {
                        Circle().fill(.green).frame(width: 6, height: 6)
                        Text("\(nParents) parents").font(.system(size: 10)).foregroundStyle(.secondary)
                    }
                }
                if nTaut > 0 {
                    HStack(spacing: 3) {
                        Circle().fill(.cyan).frame(width: 6, height: 6)
                        Text("\(nTaut) tautomers").font(.system(size: 10)).foregroundStyle(.secondary)
                    }
                }
                if nProt > 0 {
                    HStack(spacing: 3) {
                        Circle().fill(.orange).frame(width: 6, height: 6)
                        Text("\(nProt) protomers").font(.system(size: 10)).foregroundStyle(.secondary)
                    }
                }
                Text("\(nPrepared) prepared").font(.system(size: 10)).foregroundStyle(.green)
            }

            if isBatchProcessing {
                VStack(spacing: 4) {
                    ProgressView(value: Double(batchProgress.current), total: Double(max(batchProgress.total, 1)))
                        .controlSize(.small)
                    Text(processingMessage.isEmpty
                         ? "Processing \(batchProgress.current)/\(batchProgress.total)..."
                         : processingMessage)
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Unified Populate & Prepare for batch
            Label("Populate & Prepare", systemImage: "wand.and.stars")
                .font(.system(size: 11, weight: .semibold))

            Text("Full pipeline for \(uniqueEntryIDs.count) molecules: add polar H → MMFF94 minimize → Gasteiger charges → enumerate tautomers & protomers → generate conformers → filter by Boltzmann population.")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Configuration (same parameters as single entry)
            Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 6) {
                GridRow {
                    Text("Target pH").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Slider(value: $variantPH, in: 1...14, step: 0.1).controlSize(.mini)
                    Text(String(format: "%.1f", variantPH))
                        .font(.system(size: 10, design: .monospaced)).frame(width: 25)
                }
                GridRow {
                    Text("Conformers/form").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Picker("", selection: $prepNumConformers) {
                        ForEach([1, 10, 50, 100], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Max tautomers").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Picker("", selection: $variantMaxTautomers) {
                        ForEach([5, 10, 25], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Max protomers").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Picker("", selection: $variantMaxProtomers) {
                        ForEach([4, 8, 16], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                    Spacer()
                }
                GridRow {
                    Text("Energy cutoff").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Slider(value: $variantEnergyCutoff, in: 5...50, step: 1).controlSize(.mini)
                    Text(String(format: "%.0f kcal", variantEnergyCutoff))
                        .font(.system(size: 10, design: .monospaced)).frame(width: 45)
                }
                GridRow {
                    Text("Min population").font(.system(size: 10)).frame(width: 90, alignment: .leading)
                    Slider(value: $variantMinPopulation, in: 0...20, step: 0.5).controlSize(.mini)
                    Text(String(format: "%.1f%%", variantMinPopulation))
                        .font(.system(size: 10, design: .monospaced)).frame(width: 45)
                }
            }

            if isBatchProcessing {
                Button(action: { cancelPopulateAndPrepare() }) {
                    Label("Stop Processing", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.regular)
            } else {
                let parentEntries = db.entries.filter { uniqueEntryIDs.contains($0.id) && $0.parentID == nil }
                Button(action: {
                    runPopulateAndPrepare(entries: parentEntries)
                }) {
                    Label("Populate & Prepare (\(parentEntries.count) molecules)",
                          systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(parentEntries.isEmpty)
            }

            Divider()

            // Docking action — each selected form is a docking candidate
            Button(action: { useSelectedForDocking() }) {
                Label(nPrepared > 1
                      ? "Dock \(nPrepared) Forms"
                      : nPrepared == 1 ? "Use for Docking" : "No Prepared Forms",
                      systemImage: "arrow.right.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(nPrepared == 0)
            .help("Send each selected chemical form to docking")

            Button(action: { deleteSelected() }) {
                Label("Delete Selected (\(selectedRows.count))", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .foregroundStyle(.red)
        }
        .padding()
    }

    // MARK: - Batch Operation Functions

    func prepareBatchEntries(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges, nc = prepNumConformers
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let (mol, desc, _) = await Task.detached { @Sendable in
                    RDKitBridge.prepareLigand(smiles: smi, name: nm, numConformers: nc,
                                              addHydrogens: ah, minimize: mn, computeCharges: cc)
                }.value

                if let mol {
                    var updated = entry
                    updated.atoms = mol.atoms
                    updated.bonds = mol.bonds
                    updated.descriptors = desc
                    updated.isPrepared = true
                    updated.conformerCount = 1
                    db.update(updated)
                }
            }
            viewModel.log.success("Batch prepared \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    func generateConformersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let nc = prepNumConformers, mn = prepMinimize
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.generateConformers(smiles: smi, name: nm, count: nc, minimize: mn)
                }.value

                if let best = results.first {
                    var updated = entry
                    updated.atoms = best.molecule.atoms
                    updated.bonds = best.molecule.bonds
                    updated.conformerCount = results.count
                    updated.isPrepared = true
                    db.update(updated)
                }
            }
            viewModel.log.success("Batch generated conformers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    func generateTautomersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let maxT = variantMaxTautomers
            let cutoff = variantEnergyCutoff
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.enumerateTautomers(smiles: smi, name: nm,
                                                    maxTautomers: maxT, energyCutoff: cutoff)
                }.value

                // Batch remove + add to avoid save storm
                let entryID = entry.id
                db.batchMutate { entries in
                    let existingIDs = Set(entries.filter { $0.parentID == entryID && $0.variantKind == .tautomer }.map(\.id))
                    if !existingIDs.isEmpty {
                        entries.removeAll { existingIDs.contains($0.id) }
                    }
                    let parentName = entries.first(where: { $0.id == entryID })?.name ?? entry.name
                    for r in results {
                        entries.append(LigandEntry(
                            name: "\(parentName)_\(r.label)", smiles: r.smiles,
                            atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                            isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0,
                            variantLineage: "\(r.label) of \(parentName)",
                            parentID: entryID, variantKind: .tautomer,
                            relativeEnergy: r.score
                        ))
                    }
                }
                expandedEntryIDs.insert(entry.id)
            }
            viewModel.log.success("Batch generated tautomers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }

    func generateProtomersBatch(_ entries: [LigandEntry]) {
        let toProcess = entries.filter { !$0.smiles.isEmpty }
        guard !toProcess.isEmpty else { return }
        isBatchProcessing = true
        batchProgress = (0, toProcess.count)

        Task {
            let maxP = variantMaxProtomers
            let pH = variantPH
            for (i, entry) in toProcess.enumerated() {
                if Task.isCancelled { break }
                batchProgress = (i + 1, toProcess.count)

                let smi = entry.smiles, nm = entry.name
                let results = await Task.detached { @Sendable in
                    RDKitBridge.enumerateProtomers(smiles: smi, name: nm,
                                                    maxProtomers: maxP, pH: pH)
                }.value

                // Batch remove + add to avoid save storm
                let entryID = entry.id
                db.batchMutate { entries in
                    let existingIDs = Set(entries.filter { $0.parentID == entryID && $0.variantKind == .protomer }.map(\.id))
                    if !existingIDs.isEmpty {
                        entries.removeAll { existingIDs.contains($0.id) }
                    }
                    let parentName = entries.first(where: { $0.id == entryID })?.name ?? entry.name
                    for r in results {
                        entries.append(LigandEntry(
                            name: "\(parentName)_\(r.label)", smiles: r.smiles,
                            atoms: r.molecule.atoms, bonds: r.molecule.bonds,
                            isPrepared: !r.molecule.atoms.isEmpty, conformerCount: 0,
                            variantLineage: "\(r.label) of \(parentName)",
                            parentID: entryID, variantKind: .protomer,
                            relativeEnergy: r.score
                        ))
                    }
                }
                expandedEntryIDs.insert(entry.id)
            }
            viewModel.log.success("Batch generated protomers for \(toProcess.count) entries", category: .molecule)
            isBatchProcessing = false
        }
    }
}

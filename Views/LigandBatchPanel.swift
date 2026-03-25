import SwiftUI
import AppKit

// MARK: - LigandDatabaseWindow Batch Operations Extension
// Batch action panel UI and all batch processing functions (prepare, conformers, tautomers, protomers).

extension LigandDatabaseWindow {

    // MARK: - Batch Action Panel

    @ViewBuilder
    var batchActionPanel: some View {
        let selectedEntries = db.entries.filter { selectedIDs.contains($0.id) }

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(selectedEntries.count) entries selected", systemImage: "checkmark.square.fill")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Deselect All") { selectedIDs.removeAll() }
                    .font(.system(size: 9))
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            if isBatchProcessing {
                VStack(spacing: 4) {
                    ProgressView(value: Double(batchProgress.current), total: Double(max(batchProgress.total, 1)))
                        .controlSize(.small)
                    Text("Processing \(batchProgress.current)/\(batchProgress.total)...")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Batch parameters (shared with single-entry panel)
            VStack(alignment: .leading, spacing: 6) {
                Label("Batch Preparation", systemImage: "wand.and.stars")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                Toggle("Add Hydrogens", isOn: $prepAddHydrogens)
                    .toggleStyle(.switch).controlSize(.small)
                Toggle("MMFF94 Minimization", isOn: $prepMinimize)
                    .toggleStyle(.switch).controlSize(.small)

                HStack {
                    Text("Conformers:")
                        .font(.system(size: 10))
                    Picker("", selection: $prepNumConformers) {
                        ForEach([1, 10, 50, 100], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }
            }

            HStack(spacing: 8) {
                Button(action: { prepareBatchEntries(selectedEntries) }) {
                    Label("Prepare All", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.small)
                .disabled(isBatchProcessing)

                Button(action: { generateConformersBatch(selectedEntries) }) {
                    Label("Conformers", systemImage: "cube.transparent")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Label("Batch Variants", systemImage: "arrow.triangle.branch")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max tautomers:")
                        .font(.system(size: 10))
                    Picker("", selection: $variantMaxTautomers) {
                        ForEach([5, 10, 25], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }

                HStack {
                    Text("Max protomers:")
                        .font(.system(size: 10))
                    Picker("", selection: $variantMaxProtomers) {
                        ForEach([4, 8, 16], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented).controlSize(.small)
                }
            }

            HStack(spacing: 8) {
                Button(action: { generateTautomersBatch(selectedEntries) }) {
                    Label("Tautomers", systemImage: "arrow.2.squarepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)

                Button(action: { generateProtomersBatch(selectedEntries) }) {
                    Label("Protomers", systemImage: "bolt")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .disabled(isBatchProcessing)
            }
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

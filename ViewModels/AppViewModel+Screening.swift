// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Batch Docking, Virtual Screening, Analog Generation, Export

extension AppViewModel {

    // MARK: - Batch Docking

    func queueLigandsForBatchDocking(_ entries: [LigandEntry]) {
        let prepared = entries.filter { !$0.atoms.isEmpty }
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in selection — prepare them first", category: .dock)
            return
        }
        docking.batchQueue = prepared
        let first = prepared[0]
        let mol = Molecule(name: first.name, atoms: first.atoms, bonds: first.bonds,
                           title: first.smiles, smiles: first.smiles)
        setLigandForDocking(mol)
        log.success("Queued \(prepared.count) ligands for batch docking", category: .dock)
        workspace.statusMessage = "\(prepared.count) ligands queued — run from Docking tab"
    }

    func dockEntries(_ entries: [LigandEntry]) {
        let prepared = entries.filter { !$0.atoms.isEmpty }
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in selection — prepare them first", category: .dock)
            return
        }
        guard let pocket = docking.selectedPocket, let prot = molecules.protein else {
            log.error("Need protein and pocket for batch docking", category: .dock)
            return
        }

        docking.isBatchDocking = true
        docking.batchResults = []
        docking.dockingResults = []
        docking.batchProgress = (0, prepared.count)
        log.info("Starting batch docking: \(prepared.count) ligands, scoring=\(docking.scoringMethod.rawValue)", category: .dock)
        workspace.statusMessage = "Batch docking 0/\(prepared.count)..."

        docking.batchDockingTask = Task {
            if docking.dockingEngine == nil, let device = renderer?.device {
                docking.dockingEngine = DockingEngine(device: device)
            }
            guard let engine = docking.dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                docking.isBatchDocking = false
                return
            }

            // Prepare protein for scoring (uses cache — instant if protein/pH unchanged)
            let dockingPH = molecules.protonationPH
            let (scoringProtein, _) = await preparedProteinForDocking(protein: prot, pH: dockingPH)

            engine.computeGridMaps(protein: scoringProtein, pocket: pocket, spacing: docking.dockingConfig.gridSpacing)

            // Bind pharmacophore buffers (required by all scoring kernels, even if empty)
            engine.prepareConstraintBuffers(
                docking.pharmacophoreConstraints.filter(\.isEnabled),
                atoms: scoringProtein.atoms,
                residues: scoringProtein.residues
            )
            log.success("[Batch] Grid maps computed", category: .dock)

            for (i, entry) in prepared.enumerated() {
                guard !Task.isCancelled, docking.isBatchDocking else {
                    log.info("Batch docking cancelled at \(i)/\(prepared.count)", category: .dock)
                    break
                }

                docking.batchProgress = (i, prepared.count)
                workspace.statusMessage = "Docking \(i + 1)/\(prepared.count): \(entry.name)..."

                // Validate ligand data before GPU dispatch
                let heavyAtoms = entry.atoms.filter { $0.element != .H }
                let hasNaN = entry.atoms.contains { $0.position.x.isNaN || $0.position.y.isNaN || $0.position.z.isNaN }
                guard !heavyAtoms.isEmpty, !hasNaN else {
                    log.warn("[Batch] Skipping \(entry.name): invalid atom data (\(heavyAtoms.count) heavy, hasNaN=\(hasNaN))", category: .dock)
                    continue
                }
                guard heavyAtoms.count <= 128 else {
                    log.warn("[Batch] Skipping \(entry.name): \(heavyAtoms.count) heavy atoms exceeds 128-atom GPU limit", category: .dock)
                    continue
                }

                log.info("[Batch] Docking \(i+1)/\(prepared.count): \(entry.name) (\(heavyAtoms.count) heavy atoms)", category: .dock)

                let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds,
                                   title: entry.smiles, smiles: entry.smiles)
                molecules.ligand = mol
                docking.originalDockingLigand = mol
                docking.currentInteractions = []
                renderer?.updateInteractionLines([])
                pushToRenderer()

                engine.onPoseUpdate = { [weak self] result, interactions in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.docking.dockingBestEnergy = result.energy
                        self.docking.currentInteractions = interactions
                        if let (newAtoms, newBonds) = self.buildTransformedLigand(result: result, originalLigand: mol) {
                            self.renderer?.updateGhostPose(atoms: newAtoms, bonds: newBonds)
                        }
                        if !interactions.isEmpty {
                            self.renderer?.updateInteractionLines(interactions)
                        }
                    }
                }
                engine.onGenerationComplete = { [weak self] gen, energy in
                    Task { @MainActor [weak self] in
                        self?.docking.dockingGeneration = gen
                    }
                }

                // Auto-tune per ligand (protein features already computed once)
                var batchConfig = docking.dockingConfig
                if batchConfig.autoMode {
                    let heavyAtoms = entry.atoms.filter { $0.element != .H }.count
                    let torsions = RDKitBridge.buildTorsionTree(smiles: entry.smiles)?.count ?? 0
                    let tuned = DockingConfig.autoTune(
                        proteinAtomCount: scoringProtein.atoms.count,
                        pocketVolume: pocket.volume,
                        pocketBuriedness: pocket.buriedness,
                        ligandHeavyAtoms: heavyAtoms,
                        ligandRotatableBonds: torsions
                    )
                    batchConfig.populationSize = tuned.populationSize
                    batchConfig.generationsPerRun = tuned.generationsPerRun
                    batchConfig.numRuns = tuned.numRuns
                    batchConfig.gridSpacing = tuned.gridSpacing
                    batchConfig.localSearchFrequency = tuned.localSearchFrequency
                    batchConfig.localSearchSteps = tuned.localSearchSteps
                    batchConfig.explorationPhaseRatio = tuned.explorationPhaseRatio
                    batchConfig.explorationTranslationStep = tuned.explorationTranslationStep
                    batchConfig.explorationMutationRate = tuned.explorationMutationRate
                    log.info("  Auto: pop=\(tuned.populationSize) gen=\(tuned.generationsPerRun) runs=\(tuned.numRuns) (heavy=\(heavyAtoms) torsions=\(torsions))", category: .dock)
                }

                docking.isDocking = true
                docking.dockingBestEnergy = .infinity
                docking.dockingGeneration = 0
                docking.dockingTotalGenerations = batchConfig.numGenerations

                // One entry = one docking job (flat model, no ensemble)
                var entryResults: [DockingResult] = []
                do {
                    guard !Task.isCancelled, docking.isBatchDocking else { break }
                    let startResults = await engine.runDocking(
                        ligand: mol, pocket: pocket,
                        config: batchConfig, scoringMethod: docking.scoringMethod)

                    entryResults.append(contentsOf: startResults)
                }
                entryResults.sort { $0.energy < $1.energy }

                docking.isDocking = false
                log.info("[Batch] \(entry.name): \(entryResults.count) poses returned", category: .dock)

                docking.batchResults.append((ligandName: entry.name, results: entryResults))

                if let best = entryResults.first {
                    let sm = docking.scoringMethod
                    log.info(String(format: "  %@ best: %.1f %@", entry.name, best.displayScore(method: sm), sm.unitLabel), category: .dock)
                    applyDockingPose(best, originalLigand: mol)
                }
            }

            docking.batchProgress = (prepared.count, prepared.count)
            docking.isBatchDocking = false

            docking.batchResults.sort { a, b in
                let bestA = a.results.first?.energy ?? .infinity
                let bestB = b.results.first?.energy ?? .infinity
                return bestA < bestB
            }

            if let bestBatch = docking.batchResults.first, let bestResult = bestBatch.results.first {
                if let bestEntry = prepared.first(where: { $0.name == bestBatch.ligandName }) {
                    let mol = Molecule(name: bestEntry.name, atoms: bestEntry.atoms,
                                       bonds: bestEntry.bonds, title: bestEntry.smiles, smiles: bestEntry.smiles)
                    molecules.ligand = mol
                    docking.originalDockingLigand = mol
                    applyDockingPose(bestResult, originalLigand: mol)
                }
                docking.dockingResults = bestBatch.results
            }

            if let pocket = docking.selectedPocket {
                showGridBoxForPocket(pocket)
            }

            let sm = docking.scoringMethod
            let bestScore = docking.batchResults.first?.results.first.map { $0.displayScore(method: sm) } ?? 0
            log.success("Batch complete: \(docking.batchResults.count) ligands, best: \(docking.batchResults.first?.ligandName ?? "?") (\(String(format: "%.1f", bestScore)) \(sm.unitLabel))", category: .dock)
            workspace.statusMessage = "Batch complete — \(docking.batchResults.count) ligands ranked"
        }
    }

    func dockAllFromDatabase() {
        guard let pocket = docking.selectedPocket,
              let prot = molecules.protein else {
            log.error("Need protein and pocket for batch docking", category: .dock)
            return
        }

        let prepared = ligandDB.entries.filter(\.isPrepared)
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in database", category: .dock)
            return
        }

        docking.isBatchDocking = true
        docking.batchResults = []
        docking.batchProgress = (0, prepared.count)
        log.info("Starting batch docking: \(prepared.count) ligands", category: .dock)
        workspace.statusMessage = "Batch docking 0/\(prepared.count)..."

        docking.batchDockingTask = Task {
            if docking.dockingEngine == nil, let device = renderer?.device {
                docking.dockingEngine = DockingEngine(device: device)
            }
            guard let engine = docking.dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                docking.isBatchDocking = false
                return
            }

            engine.computeGridMaps(protein: prot, pocket: pocket, spacing: docking.dockingConfig.gridSpacing)

            for (i, entry) in prepared.enumerated() {
                guard !Task.isCancelled, docking.isBatchDocking else {
                    log.info("Batch docking cancelled at \(i)/\(prepared.count)", category: .dock)
                    break
                }

                docking.batchProgress = (i, prepared.count)
                workspace.statusMessage = "Batch docking \(i + 1)/\(prepared.count): \(entry.name)..."

                let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)

                engine.onPoseUpdate = nil
                engine.onGenerationComplete = nil

                let results = await engine.runDocking(ligand: mol, pocket: pocket, config: docking.dockingConfig)
                docking.batchResults.append((ligandName: entry.name, results: results))

                if let best = results.first {
                    let sm = docking.scoringMethod
                    log.info(String(format: "  %@ best: %.1f %@", entry.name, best.displayScore(method: sm), sm.unitLabel), category: .dock)
                }
            }

            docking.batchProgress = (prepared.count, prepared.count)
            docking.isBatchDocking = false

            docking.batchResults.sort { a, b in
                let bestA = a.results.first?.energy ?? .infinity
                let bestB = b.results.first?.energy ?? .infinity
                return bestA < bestB
            }

            log.success("Batch docking complete: \(docking.batchResults.count) ligands docked", category: .dock)
            workspace.statusMessage = "Batch docking complete"
        }
    }

    func cancelBatchDocking() {
        docking.batchDockingTask?.cancel()
        docking.dockingEngine?.stopDocking()
        docking.isDocking = false
        docking.isBatchDocking = false
        log.info("Batch docking cancelled", category: .dock)
        workspace.statusMessage = "Batch docking cancelled"
    }

    // MARK: - Virtual Screening

    func startScreening(library: [(name: String, smiles: String)]) {
        guard let pocket = docking.selectedPocket, let prot = molecules.protein else {
            log.error("Need protein and pocket for virtual screening", category: .dock)
            return
        }

        let pipeline = docking.screeningPipeline ?? VirtualScreeningPipeline()
        docking.screeningPipeline = pipeline
        docking.screeningState = .preparing(current: 0, total: library.count)
        docking.screeningProgress = 0
        docking.screeningHits = []

        log.info("Starting virtual screening: \(library.count) molecules", category: .dock)
        workspace.statusMessage = "Virtual screening..."

        let progressTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled else { break }
                docking.screeningState = pipeline.state
                docking.screeningProgress = pipeline.progress
            }
        }

        docking.screeningTask = Task {
            if docking.dockingEngine == nil, let device = renderer?.device {
                docking.dockingEngine = DockingEngine(device: device)
            }
            guard let engine = docking.dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                docking.screeningState = .failed("Docking engine unavailable")
                progressTask.cancel()
                return
            }

            await pipeline.screen(
                smilesLibrary: library,
                dockingEngine: engine,
                protein: prot,
                pocket: pocket,
                admetPredictor: pipeline.config.admetFilter && admetPredictor.isAvailable ? admetPredictor : nil,
                constraints: docking.pharmacophoreConstraints
            )

            progressTask.cancel()

            docking.screeningState = pipeline.state
            docking.screeningProgress = pipeline.progress
            docking.screeningHits = pipeline.hits

            if case .complete(let hits, let total) = pipeline.state {
                log.success("Screening complete: \(hits) hits from \(total) molecules", category: .dock)
                workspace.statusMessage = "Screening: \(hits) hits"
            }
        }
    }

    func cancelScreening() {
        docking.screeningPipeline?.cancel()
        docking.screeningTask?.cancel()
        docking.screeningState = .idle
        docking.screeningProgress = 0
        log.info("Virtual screening cancelled", category: .dock)
        workspace.statusMessage = "Screening cancelled"
    }

    // MARK: - Export

    func exportDockingResultsSDF() {
        guard !docking.dockingResults.isEmpty, let lig = molecules.ligand else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "sdf")].compactMap { $0 }
        panel.nameFieldStringValue = "docking_poses.sdf"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        var molBlocks: [String] = []
        molBlocks.reserveCapacity(min(docking.dockingResults.count, 50))
        for (i, result) in docking.dockingResults.prefix(50).enumerated() {
            let heavyAtoms = lig.atoms.filter { $0.element != .H }
            var poseAtoms: [Atom] = []
            var idMap: [Int: Int] = [:]
            for (j, atom) in heavyAtoms.enumerated() {
                guard j < result.transformedAtomPositions.count else { break }
                idMap[atom.id] = poseAtoms.count
                poseAtoms.append(Atom(
                    id: poseAtoms.count, element: atom.element,
                    position: result.transformedAtomPositions[j],
                    name: atom.name, residueName: atom.residueName,
                    residueSeq: atom.residueSeq, chainID: atom.chainID,
                    charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom
                ))
            }
            var poseBonds: [Bond] = []
            for bond in lig.bonds {
                if let a = idMap[bond.atomIndex1], let b = idMap[bond.atomIndex2] {
                    poseBonds.append(Bond(id: poseBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
            let props: [String: String] = [
                "Rank": "\(i + 1)",
                "Energy": String(format: "%.3f", result.energy),
                "VdW": String(format: "%.3f", result.vdwEnergy),
                "Elec": String(format: "%.3f", result.elecEnergy),
                "HBond": String(format: "%.3f", result.hbondEnergy),
                "Cluster": "\(result.clusterID)"
            ]
            molBlocks.append(SDFWriter.molBlock(name: "\(lig.name)_pose\(i + 1)", atoms: poseAtoms,
                                       bonds: poseBonds, properties: props))
        }
        let sdf = molBlocks.joined()

        do {
            try SDFWriter.save(sdf, to: url)
            log.success("Exported \(min(docking.dockingResults.count, 50)) poses to SDF", category: .dock)
        } catch {
            log.error("SDF export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    /// Export a single docked pose (protein + transformed ligand) as a combined PDB file.
    /// Works for newly-computed poses and for poses restored from a saved `.druse` project,
    /// since `transformedAtomPositions` is persisted in the project file.
    func exportDockingPosePDB(at index: Int) {
        guard docking.dockingResults.indices.contains(index) else { return }
        let result = docking.dockingResults[index]
        let ligand = docking.originalDockingLigand ?? molecules.ligand
        guard !result.transformedAtomPositions.isEmpty else {
            log.warn("Cannot export pose #\(index + 1): missing transformed coordinates", category: .dock)
            return
        }
        guard ligand != nil || molecules.protein != nil else {
            log.warn("Cannot export pose #\(index + 1): no protein or ligand loaded", category: .dock)
            return
        }

        let ligName = ligand?.name ?? "ligand"
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "pdb")].compactMap { $0 }
        panel.nameFieldStringValue = "\(ligName)_pose\(index + 1).pdb"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let title = String(
            format: "Druse pose %d  score=%.2f %@",
            index + 1,
            result.displayScore(method: docking.scoringMethod),
            docking.scoringMethod.unitLabel
        )
        let pdb = PDBWriter.combinedPose(
            protein: molecules.protein,
            ligandTemplate: ligand,
            ligandPositions: result.transformedAtomPositions,
            title: title
        )

        do {
            try PDBWriter.save(pdb, to: url)
            log.success("Exported pose #\(index + 1) to \(url.lastPathComponent)", category: .dock)
        } catch {
            log.error("PDB export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    func exportDockingResultsCSV() {
        guard !docking.dockingResults.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "csv")].compactMap { $0 }
        panel.nameFieldStringValue = "docking_results.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let csv = exportResultsCSV()
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            log.success("Exported docking results to CSV", category: .dock)
        } catch {
            log.error("CSV export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    func exportScreeningHits() {
        guard let pipeline = docking.screeningPipeline, !docking.screeningHits.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "csv")].compactMap { $0 }
        panel.nameFieldStringValue = "screening_hits.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let csv = pipeline.exportCSV()
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            log.success("Exported \(docking.screeningHits.count) screening hits to CSV", category: .dock)

            let sdfURL = url.deletingPathExtension().appendingPathExtension("sdf")
            let sdf = pipeline.exportSDF(topN: 100)
            try sdf.write(to: sdfURL, atomically: true, encoding: .utf8)
            log.success("Exported top 100 hits to SDF", category: .dock)
        } catch {
            log.error("Export failed: \(error.localizedDescription)", category: .dock)
        }
    }

}

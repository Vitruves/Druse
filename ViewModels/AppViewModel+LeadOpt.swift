// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Lead Optimization (Analog Generation, Mini-Docking, Comparison)

extension AppViewModel {

    // MARK: - Reference Selection

    /// Set a docking result as the reference for lead optimization.
    /// Called from the Results tab "Optimize" button.
    func selectReferenceForOptimization(result: DockingResult, ligand: Molecule) {
        guard let smiles = Self.referenceSMILES(for: ligand),
              let descriptors = RDKitBridge.computeDescriptors(smiles: smiles) else {
            let msg = "Cannot optimize \(ligand.name): RDKit could not derive a valid SMILES from its structure"
            log.error(msg, category: .molecule)
            workspace.statusMessage = msg
            return
        }
        leadOpt.referenceResult = result
        leadOpt.referenceMolecule = ligand
        leadOpt.referenceName = ligand.name
        leadOpt.referenceSMILES = smiles
        leadOpt.referenceDescriptors = descriptors
        leadOpt.scoringMethod = docking.scoringMethod
        workspace.requestedTab = .leadOptimization
        workspace.requestedTabToken &+= 1
        workspace.statusMessage = "Reference set: \(ligand.name) — opened Lead Opt"
        log.info("Reference for lead optimization: \(ligand.name) (\(smiles))", category: .molecule)
    }

    /// SMILES of a docked ligand. SDF / PDB ligands carry no SMILES (their title is a record
    /// name such as "6o0m_ligand"), so fall back to the heavy-atom graph, which RDKit turns
    /// into canonical SMILES with implicit hydrogens. Explicit H atoms are dropped first —
    /// left in, they would surface as "[H]" atoms that break substituent edits.
    private static func referenceSMILES(for ligand: Molecule) -> String? {
        if let recorded = ligand.smiles?.trimmingCharacters(in: .whitespacesAndNewlines),
           !recorded.isEmpty, RDKitBridge.computeDescriptors(smiles: recorded) != nil {
            return recorded
        }
        var heavyIndex: [Int: Int] = [:]
        var heavyAtoms: [Atom] = []
        for (i, atom) in ligand.atoms.enumerated() where atom.element != .H {
            heavyIndex[i] = heavyAtoms.count
            heavyAtoms.append(atom)
        }
        let heavyBonds: [Bond] = ligand.bonds.compactMap { bond in
            guard let a = heavyIndex[bond.atomIndex1], let b = heavyIndex[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }
        return RDKitBridge.atomsBondsToSMILES(atoms: heavyAtoms, bonds: heavyBonds)
    }

    /// Set a screening hit as the reference.
    func selectReferenceFromHit(_ hit: VirtualScreeningPipeline.ScreeningHit) {
        leadOpt.referenceSMILES = hit.smiles
        leadOpt.referenceName = hit.name
        leadOpt.referenceDescriptors = hit.descriptors
        leadOpt.scoringMethod = docking.scoringMethod
        leadOpt.referenceMolecule = nil
        leadOpt.referenceResult = nil
        if !hit.bestPoseAtoms.isEmpty {
            // Build a minimal DockingResult for the reference pose
            let refResult = DockingResult(
                id: 0,
                pose: DockPoseSwift(translation: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                energy: hit.bestEnergy, stericEnergy: 0, hydrophobicEnergy: 0,
                hbondEnergy: 0, torsionPenalty: 0, generation: 0,
                transformedAtomPositions: hit.bestPoseAtoms
            )
            leadOpt.referenceResult = refResult
        }
        workspace.requestedTab = .leadOptimization
        workspace.requestedTabToken &+= 1
        workspace.statusMessage = "Reference set: \(hit.name) — opened Lead Opt"
        log.info("Reference for lead optimization: \(hit.name) (from screening)", category: .molecule)
    }

    // MARK: - Analog Generation

    /// Medicinal-chemistry edits, applied to SMILES atom tokens (see `SMILESAtomGraph`), with
    /// property-direction bias annotations. `altersScaffold` edits change ring atoms and are
    /// skipped while "Keep scaffold" is on.
    private static let editTable: [(edit: AnalogEdit, label: String, altersScaffold: Bool, bias: ReplacementBias)] = [
        // Halogen swaps
        (.swapTerminalAtom(from: "F", to: "Cl"), "F→Cl", false, .init(polarity: -0.2, lipophilicity: 0.3, size: 0.1)),
        (.swapTerminalAtom(from: "Cl", to: "F"), "Cl→F", false, .init(polarity: 0.2, lipophilicity: -0.3, size: -0.1)),
        (.swapTerminalAtom(from: "F", to: "Br"), "F→Br", false, .init(polarity: -0.3, lipophilicity: 0.4, size: 0.2)),
        (.swapTerminalAtom(from: "Cl", to: "Br"), "Cl→Br", false, .init(polarity: -0.1, lipophilicity: 0.2, size: 0.1)),
        (.swapTerminalAtom(from: "Br", to: "Cl"), "Br→Cl", false, .init(polarity: 0.1, lipophilicity: -0.2, size: -0.1)),
        // Aromatic C–H scans
        (.substituteAromaticCH("F"), "H→F", false, .init(polarity: -0.1, lipophilicity: 0.1)),
        (.substituteAromaticCH("Cl"), "H→Cl", false, .init(polarity: -0.2, lipophilicity: 0.4, size: 0.2)),
        (.substituteAromaticCH("C"), "H→Me", false, .init(lipophilicity: 0.3, size: 0.2)),
        (.substituteAromaticCH("O"), "H→OH", false, .init(polarity: 0.4, lipophilicity: -0.3, size: 0.1)),
        // Methyl elaboration
        (.branchOnMethyl("(C)"), "Me→Et", false, .init(rigidity: -0.1, lipophilicity: 0.3, size: 0.3)),
        (.branchOnMethyl("(C)(C)"), "Me→iPr", false, .init(rigidity: 0.1, lipophilicity: 0.4, size: 0.4)),
        (.branchOnMethyl("(F)(F)(F)"), "Me→CF3", false, .init(polarity: -0.3, rigidity: 0.2, lipophilicity: 0.5, size: 0.2)),
        // Acyclic heteroatom swaps (amide NH→ester O, carbonyl O→S, ether O→S, …)
        (.swapAcyclicAtom(from: "O", to: "N"), "O→NH", false, .init(polarity: 0.3)),
        (.swapAcyclicAtom(from: "N", to: "O"), "NH→O", false, .init(polarity: -0.2)),
        (.swapAcyclicAtom(from: "O", to: "S"), "O→S", false, .init(polarity: -0.4, lipophilicity: 0.3, size: 0.1)),
        (.swapAcyclicAtom(from: "S", to: "O"), "S→O", false, .init(polarity: 0.4, lipophilicity: -0.3, size: -0.1)),
        // Functional-group swaps on acyclic atoms
        (.replaceGroup("C(=O)O", "C(=O)N"), "CO2→CONH", false, .init(polarity: 0.2)),
        (.replaceGroup("C(=O)N", "C(=O)O"), "CONH→CO2", false, .init(polarity: -0.1)),
        (.replaceGroup("C(=O)", "S(=O)(=O)"), "CO→SO2", false, .init(polarity: 0.3, size: 0.1)),
        // Ring-atom edits (aza-walk / de-aza, ring size)
        (.swapAromaticAtom(from: "c", to: "n"), "CH→N", true, .init(polarity: 0.5, lipophilicity: -0.3)),
        (.swapAromaticAtom(from: "n", to: "c"), "N→CH", true, .init(polarity: -0.5, lipophilicity: 0.3)),
        (.replaceGroup("C1CCCC1", "C1CCCCC1"), "cPent→cHex", true, .init(rigidity: 0.1, size: 0.2)),
        (.replaceGroup("C1CCCCC1", "C1CCCC1"), "cHex→cPent", true, .init(rigidity: -0.1, size: -0.2)),
    ]

    private struct ReplacementBias {
        var polarity: Float = 0
        var rigidity: Float = 0
        var lipophilicity: Float = 0
        var size: Float = 0

        /// Score how well this replacement aligns with the requested property directions.
        func alignment(polDir: Float, rigDir: Float, lipDir: Float, sizeDir: Float) -> Float {
            polarity * polDir + rigidity * rigDir + lipophilicity * lipDir + size * sizeDir
        }
    }

    /// Generate property-directed analogs with similarity, scaffold and ADMET filtering.
    func generateOptimizedAnalogs() {
        guard leadOpt.hasReference else { return }
        let smiles = leadOpt.referenceSMILES
        let name = leadOpt.referenceName
        guard let graph = SMILESAtomGraph(smiles) else {
            let msg = "Cannot generate analogs: the reference SMILES of \(name) could not be tokenized"
            log.error(msg, category: .molecule)
            workspace.statusMessage = msg
            return
        }

        leadOpt.isGenerating = true
        leadOpt.generationProgress = 0
        leadOpt.analogs = []
        leadOpt.selectedAnalogIndex = nil

        let count = leadOpt.analogCount
        let refDesc = leadOpt.referenceDescriptors
        let minSimilarity = leadOpt.similarityThreshold
        let keepScaffold = leadOpt.keepScaffold
        let polDir = leadOpt.polarityDirection
        let rigDir = leadOpt.rigidityDirection
        let lipDir = leadOpt.lipophilicityDirection
        let sizeDir = leadOpt.sizeDirection
        let filterConfig = (
            lipinski: leadOpt.filterLipinski, veber: leadOpt.filterVeber,
            herg: leadOpt.filterHERG, cyp: leadOpt.filterCYP,
            maxLogP: leadOpt.maxLogP
        )

        log.info("Generating up to \(count) optimized analogs of \(name)...", category: .molecule)
        workspace.statusMessage = "Generating analogs..."

        leadOpt.generationTask = Task {
            defer {
                self.leadOpt.isGenerating = false
                self.leadOpt.generationProgress = 1.0
            }

            // Score and sort edits by alignment with property directions
            let hasDirectionBias = abs(polDir) + abs(rigDir) + abs(lipDir) + abs(sizeDir) > 0.01
            var edits = Self.editTable.filter { !(keepScaffold && $0.altersScaffold) }
            if hasDirectionBias {
                edits.sort {
                    $0.bias.alignment(polDir: polDir, rigDir: rigDir, lipDir: lipDir, sizeDir: sizeDir)
                        > $1.bias.alignment(polDir: polDir, rigDir: rigDir, lipDir: lipDir, sizeDir: sizeDir)
                }
            }

            // Every site of every edit, grouped per edit so the pool can be interleaved:
            // one site per edit per round keeps the list chemically diverse instead of
            // filling it with ten positions of the first matching edit.
            let sitesPerEdit: [[(label: String, smiles: String)]] = edits.map { entry in
                let sites = graph.applying(entry.edit, keepScaffold: keepScaffold)
                return sites.map { site in
                    (label: sites.count > 1 ? "\(entry.label)@\(site.atom + 1)" : entry.label, smiles: site.smiles)
                }
            }
            var pool: [(label: String, smiles: String)] = []
            for round in 0..<(sitesPerEdit.map(\.count).max() ?? 0) {
                for sites in sitesPerEdit where round < sites.count { pool.append(sites[round]) }
            }

            var generated: [LeadOptAnalog] = []
            var seenFingerprints: [[Float]] = [RDKitBridge.morganFingerprint(smiles: smiles)]
            var invalidCount = 0
            var dissimilarCount = 0
            for (i, candidate) in pool.enumerated() {
                if generated.count >= count || Task.isCancelled { break }
                self.leadOpt.generationProgress = Float(i + 1) / Float(max(pool.count, 1)) * 0.5

                // RDKit validation rejects edits that break valence or aromaticity.
                guard let desc = RDKitBridge.computeDescriptors(smiles: candidate.smiles) else {
                    invalidCount += 1
                    continue
                }
                // Symmetry-equivalent sites give different strings for the same molecule.
                let fingerprint = RDKitBridge.morganFingerprint(smiles: candidate.smiles)
                guard !seenFingerprints.contains(fingerprint) else { continue }
                seenFingerprints.append(fingerprint)
                if RDKitBridge.tanimotoSimilarity(smiles, candidate.smiles) < minSimilarity {
                    dissimilarCount += 1
                    continue
                }
                var analog = LeadOptAnalog(id: UUID(), name: "\(name)_\(candidate.label)", smiles: candidate.smiles)
                analog.descriptors = desc
                self.computeDeltaProperties(for: &analog, reference: refDesc)
                generated.append(analog)
            }

            if invalidCount > 0 || dissimilarCount > 0 {
                self.log.info(
                    "Analog generation skipped \(invalidCount) chemically invalid and \(dissimilarCount) below-similarity (< \(Int(minSimilarity * 100))% Tanimoto) candidates",
                    category: .molecule
                )
            }

            // ADMET filtering (remaining 50% of progress)
            var passed: [LeadOptAnalog] = []
            for (i, var analog) in generated.enumerated() {
                if Task.isCancelled { break }

                // Descriptor-based pre-filter
                if let desc = analog.descriptors {
                    if filterConfig.lipinski && !desc.lipinski { analog.status = .filtered; passed.append(analog); continue }
                    if filterConfig.veber && !desc.veber { analog.status = .filtered; passed.append(analog); continue }
                    if desc.logP > filterConfig.maxLogP { analog.status = .filtered; passed.append(analog); continue }
                }

                // ML ADMET prediction
                if filterConfig.herg || filterConfig.cyp {
                    let admet = await self.admetPredictor.predict(smiles: analog.smiles)
                    analog.admet = admet
                    if filterConfig.herg, let herg = admet.hergLiability, herg > 0.5 {
                        analog.status = .filtered; passed.append(analog); continue
                    }
                    if filterConfig.cyp {
                        if let cyp2d6 = admet.cyp2d6Inhibition, cyp2d6 > 0.5 { analog.status = .filtered; passed.append(analog); continue }
                        if let cyp3a4 = admet.cyp3a4Inhibition, cyp3a4 > 0.5 { analog.status = .filtered; passed.append(analog); continue }
                    }
                }

                passed.append(analog)
                self.leadOpt.generationProgress = 0.5 + Float(i + 1) / Float(max(generated.count, 1)) * 0.5
            }

            self.leadOpt.analogs = passed

            let activeCount = passed.filter { $0.status == .generated }.count
            let filteredCount = passed.filter { $0.status == .filtered }.count
            let totalGenerated = generated.count

            if totalGenerated == 0 {
                let reason = pool.isEmpty
                    ? "no edit site matched\(keepScaffold ? " outside the scaffold — turn off Keep scaffold to allow ring edits" : "")"
                    : "every candidate was invalid or below the \(Int(minSimilarity * 100))% similarity threshold"
                self.log.warn("Could not generate any analogs from \(name): \(reason)", category: .molecule)
                self.workspace.statusMessage = "No analogs could be generated"
            } else if activeCount > 0 {
                self.log.success("Generated \(activeCount) analogs (\(filteredCount) filtered, \(totalGenerated) total candidates)", category: .molecule)
                // Generated analogs go straight into the Ligand Database (duplicates skipped).
                let added = self.addAnalogsToLigandDatabase()
                self.workspace.statusMessage = "\(activeCount) analogs ready · \(added) added to Ligand Database"
            } else {
                self.log.warn("\(totalGenerated) analogs generated but all \(filteredCount) filtered out — try loosening ADMET filters", category: .molecule)
                self.workspace.statusMessage = "All \(totalGenerated) analogs filtered — loosen filters"
            }
        }
    }

    private func computeDeltaProperties(for analog: inout LeadOptAnalog, reference: LigandDescriptors?) {
        // Descriptors may already be set during generation (with RDKit validation).
        // Only compute if missing.
        if analog.descriptors == nil {
            analog.descriptors = RDKitBridge.computeDescriptors(smiles: analog.smiles)
        }
        guard let desc = analog.descriptors, let ref = reference else { return }
        analog.deltaMW = desc.molecularWeight - ref.molecularWeight
        analog.deltaLogP = desc.logP - ref.logP
        analog.deltaTPSA = desc.tpsa - ref.tpsa
        analog.deltaRotBonds = desc.rotatableBonds - ref.rotatableBonds
    }

    // MARK: - Add to Ligand Database

    /// Copy generated/passing analogs into the main ligand database so they can
    /// be docked using the standard Docking tab pipeline (with full UI feedback,
    /// per-ligand progress, ensemble preparation, etc.). Returns the count added.
    @discardableResult
    func addAnalogsToLigandDatabase(includeFiltered: Bool = false) -> Int {
        let candidates = leadOpt.analogs.filter { analog in
            switch analog.status {
            case .generated, .prepared, .docked: return true
            case .filtered: return includeFiltered
            case .failed: return false
            }
        }
        guard !candidates.isEmpty else {
            workspace.statusMessage = "No analogs to add"
            return 0
        }

        let existingSMILES = Set(ligandDB.entries.map { $0.smiles })
        var added = 0
        for analog in candidates where !existingSMILES.contains(analog.smiles) {
            var entry = LigandEntry(name: analog.name, smiles: analog.smiles)
            entry.descriptors = analog.descriptors
            ligandDB.add(entry)
            added += 1
        }
        let skipped = candidates.count - added
        if added > 0 {
            log.success("Added \(added) analog(s) to Ligand Database\(skipped > 0 ? " (\(skipped) duplicates skipped)" : "")", category: .molecule)
            workspace.statusMessage = "Added \(added) analog(s) to Ligand Database"
        } else {
            workspace.statusMessage = "All analogs already in Ligand Database"
        }
        return added
    }

    /// Send all generated analogs to the Docking tab as queued ligand-DB entries
    /// and switch the user there so they can launch a batch run.
    func sendAnalogsToDocking() {
        let added = addAnalogsToLigandDatabase()
        guard added > 0 || !leadOpt.analogs.isEmpty else { return }
        workspace.requestedTab = .dock
        workspace.requestedTabToken &+= 1
    }

    // MARK: - Mini-Docking

    /// Lowest-MMFF-energy conformer whose heavy atoms do not interpenetrate.
    /// MMFF can "converge" charged molecules into collapsed geometries (non-bonded heavy
    /// atoms < 1 Å apart) that still rank lowest in energy. The docking engine takes its
    /// intramolecular reference from the input conformer, so docking such a conformer
    /// credits the strain it releases to binding (a −55.9 kcal/mol "Vina" score).
    nonisolated private static func dockableConformer(smiles: String, name: String) -> MoleculeData? {
        let minNonBondedHeavyDistance: Float = 2.0 // 1-3 heavy-atom distances are ≥ 2.2 Å
        return RDKitBridge.generateConformers(smiles: smiles, name: name, count: 10, minimize: true)
            .map(\.molecule)
            .first { conformer in
                let heavy = conformer.atoms.indices.filter { conformer.atoms[$0].element != .H }
                let bonded = Set(conformer.bonds.map { [min($0.atomIndex1, $0.atomIndex2), max($0.atomIndex1, $0.atomIndex2)] })
                for (k, i) in heavy.enumerated() {
                    for j in heavy[(k + 1)...] where !bonded.contains([i, j]) {
                        if simd_distance(conformer.atoms[i].position, conformer.atoms[j].position) < minNonBondedHeavyDistance {
                            return false
                        }
                    }
                }
                return true
            }
    }

    /// Dock all generated analogs into the reference pocket with the reference's scoring
    /// method and docking settings (reduced search budget), so scores compare 1:1.
    func dockAnalogs() {
        let activeAnalogs = leadOpt.analogs.enumerated().filter { $0.element.status == .generated }
        guard !activeAnalogs.isEmpty else {
            workspace.statusMessage = "No generated analogs to dock"
            return
        }
        guard let engine = docking.dockingEngine, let pocket = docking.selectedPocket else {
            let msg = "Run docking on the reference ligand first — analog mini-docking reuses its receptor and pocket"
            log.warn(msg, category: .dock)
            workspace.statusMessage = msg
            return
        }
        guard !engine.isRunning else {
            let msg = "The docking engine is busy — wait for the current run to finish before docking analogs"
            log.warn(msg, category: .dock)
            workspace.statusMessage = msg
            return
        }

        let scoringMethod = leadOpt.scoringMethod
        var analogConfig = docking.dockingConfig
        analogConfig.populationSize = min(analogConfig.populationSize, 150)
        analogConfig.numRuns = 1
        analogConfig.generationsPerRun = min(analogConfig.generationsPerRun, 150)
        analogConfig.liveUpdateFrequency = analogConfig.numGenerations + 1 // no live viewport updates for batch

        leadOpt.isDocking = true
        leadOpt.dockingProgress = (0, activeAnalogs.count)
        log.info(
            "Docking \(activeAnalogs.count) analogs (\(scoringMethod.rawValue), pop=\(analogConfig.populationSize), gen=\(analogConfig.generationsPerRun))...",
            category: .dock
        )
        workspace.statusMessage = "Docking analogs..."

        leadOpt.dockingTask = Task {
            defer {
                self.leadOpt.isDocking = false
                self.workspace.statusMessage = "Analog docking complete"
            }

            for (ordinal, (_, analog)) in activeAnalogs.enumerated() {
                if Task.isCancelled { break }
                self.leadOpt.dockingProgress = (ordinal + 1, activeAnalogs.count)

                // Prepare 3D molecule (RDKit embedding + MMFF + Gasteiger) off the main actor
                let analogSMILES = analog.smiles
                let analogName = analog.name
                let molData = await Task.detached(priority: .userInitiated) {
                    Self.dockableConformer(smiles: analogSMILES, name: analogName)
                }.value
                if Task.isCancelled { break }
                // The analog list can be regenerated while this task is suspended, so
                // re-resolve the row by ID after every await instead of trusting `idx`.
                guard let prepIdx = self.leadOpt.analogs.firstIndex(where: { $0.id == analog.id }) else { continue }
                guard let md = molData else {
                    self.log.warn("\(analogName): no clash-free 3D conformer — skipped", category: .dock)
                    self.leadOpt.analogs[prepIdx].status = .failed
                    continue
                }

                self.leadOpt.analogs[prepIdx].molecule = md
                self.leadOpt.analogs[prepIdx].status = .prepared

                let ligand = Molecule(name: analog.name, atoms: md.atoms, bonds: md.bonds, title: analog.name, smiles: analog.smiles)
                let results = await engine.runDocking(
                    ligand: ligand, pocket: pocket, config: analogConfig, scoringMethod: scoringMethod
                )
                guard let dockIdx = self.leadOpt.analogs.firstIndex(where: { $0.id == analog.id }) else { continue }

                let ranked = results.sorted { scoringMethod.ranks($0.displayScore(method: scoringMethod), before: $1.displayScore(method: scoringMethod)) }
                if let best = ranked.first {
                    self.leadOpt.analogs[dockIdx].dockingResults = Array(ranked.prefix(5))
                    self.leadOpt.analogs[dockIdx].bestScore = best.displayScore(method: scoringMethod)
                    self.leadOpt.analogs[dockIdx].bestPoseAtoms = best.transformedAtomPositions
                    self.leadOpt.analogs[dockIdx].status = .docked
                } else {
                    self.log.warn("\(analogName): docking returned no poses", category: .dock)
                    self.leadOpt.analogs[dockIdx].status = .failed
                }
            }

            // Best-scoring docked analogs first, then the rest in their original order
            let selectedID = self.leadOpt.selectedAnalogIndex.flatMap { idx in
                self.leadOpt.analogs.indices.contains(idx) ? self.leadOpt.analogs[idx].id : nil
            }
            self.leadOpt.analogs.sort { a, b in
                switch (a.bestScore, b.bestScore) {
                case let (sa?, sb?): return scoringMethod.ranks(sa, before: sb)
                case (.some, .none): return true
                default: return false
                }
            }
            self.leadOpt.selectedAnalogIndex = selectedID.flatMap { id in self.leadOpt.analogs.firstIndex { $0.id == id } }

            let dockedCount = self.leadOpt.analogs.filter { $0.status == .docked }.count
            self.log.success("Docked \(dockedCount)/\(activeAnalogs.count) analogs", category: .dock)
        }
    }

    // MARK: - Cancellation

    func cancelLeadOptimization() {
        leadOpt.generationTask?.cancel()
        leadOpt.dockingTask?.cancel()
        leadOpt.isGenerating = false
        leadOpt.isDocking = false
        workspace.statusMessage = "Lead optimization cancelled"
    }

    // MARK: - 3D Overlay

    /// Show the selected analog's best pose as a ghost overlay, with reference as main ligand.
    func applyAnalogPose(at index: Int) {
        guard index < leadOpt.analogs.count else { return }
        let analog = leadOpt.analogs[index]
        guard !analog.bestPoseAtoms.isEmpty, let md = analog.molecule else { return }

        // Build atom array from the docked positions
        let heavyAtoms = md.atoms.filter { $0.element != .H }
        var ghostAtoms: [Atom] = []
        for (i, atom) in heavyAtoms.enumerated() {
            guard i < analog.bestPoseAtoms.count else { break }
            ghostAtoms.append(Atom(
                id: i, element: atom.element, position: analog.bestPoseAtoms[i],
                name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                isHetAtom: true
            ))
        }

        // Remap bonds
        var oldToNew: [Int: Int] = [:]
        for (newIdx, atom) in heavyAtoms.enumerated() {
            oldToNew[atom.id] = newIdx
        }
        var ghostBonds: [Bond] = []
        for bond in md.bonds {
            if let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] {
                ghostBonds.append(Bond(id: ghostBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }

        renderer?.updateGhostPose(atoms: ghostAtoms, bonds: ghostBonds)
        workspace.statusMessage = "Showing \(analog.name) overlay"
        let method = leadOpt.scoringMethod
        if let score = analog.bestScore {
            log.info(String(format: "Analog overlay → %@ (best %.2f %@)", analog.name, score, method.unitLabel), category: .dock)
        } else {
            log.info("Analog overlay → \(analog.name)", category: .dock)
        }
    }
}

// MARK: - SMILES edit engine

/// One medicinal-chemistry edit, expressed on SMILES atom tokens.
private enum AnalogEdit {
    /// Replace a terminal (one heavy neighbour) organic-subset atom, e.g. a halogen.
    case swapTerminalAtom(from: String, to: String)
    /// Add a substituent on an aromatic C–H (`c` with two heavy neighbours).
    case substituteAromaticCH(String)
    /// Grow a terminal aliphatic methyl (single-bonded) by appending branches to it.
    case branchOnMethyl(String)
    /// Swap an aliphatic heteroatom that is not in a ring.
    case swapAcyclicAtom(from: String, to: String)
    /// Swap an unsubstituted aromatic ring atom (aza-walk / de-aza).
    case swapAromaticAtom(from: String, to: String)
    /// Textual group replacement anchored on atom tokens; all covered atoms must be
    /// acyclic unless the pattern itself contains a ring closure.
    case replaceGroup(String, String)
}

/// Minimal SMILES tokenizer: atom tokens with their text ranges, the bonds between them
/// and ring membership. Edits target atom tokens instead of raw substrings, so a methyl
/// homologation can never land inside a ring (raw "CC"→"CCC" turned imatinib's piperazine
/// into a homopiperazine) or inside a bracket atom such as `[NH+]`.
private struct SMILESAtomGraph {
    struct AtomToken {
        let range: Range<String.Index>
        let symbol: String
        let isBracket: Bool
    }

    let smiles: String
    private(set) var atoms: [AtomToken] = []
    /// Heavy-atom adjacency with the bond symbol ("-" for implicit single/aromatic).
    private(set) var neighbors: [[(atom: Int, bond: Character)]] = []
    private(set) var inRing: [Bool] = []

    init?(_ smiles: String) {
        self.smiles = smiles
        var branchStack: [Int] = []
        var previous: Int?
        var pendingBond: Character?
        var openRings: [Int: (atom: Int, bond: Character?)] = [:]
        var edges: [(Int, Int)] = []
        var i = smiles.startIndex

        func addAtom(_ range: Range<String.Index>, symbol: String, isBracket: Bool) {
            let index = atoms.count
            atoms.append(AtomToken(range: range, symbol: symbol, isBracket: isBracket))
            neighbors.append([])
            if let p = previous { link(p, index, pendingBond) }
            pendingBond = nil
            previous = index
        }
        func link(_ a: Int, _ b: Int, _ bond: Character?) {
            let symbol = bond.map { $0 == "/" || $0 == "\\" ? "-" : $0 } ?? "-"
            neighbors[a].append((b, symbol))
            neighbors[b].append((a, symbol))
            edges.append((a, b))
        }

        while i < smiles.endIndex {
            let ch = smiles[i]
            let next = smiles.index(after: i)
            switch ch {
            case "(":
                guard let p = previous else { return nil }
                branchStack.append(p)
                i = next
            case ")":
                guard let p = branchStack.popLast() else { return nil }
                previous = p
                i = next
            case "-", "=", "#", "$", ":", "/", "\\":
                pendingBond = ch
                i = next
            case ".":
                previous = nil
                pendingBond = nil
                i = next
            case "0"..."9", "%":
                let digitsEnd: String.Index
                if ch == "%" {
                    guard let end = smiles.index(next, offsetBy: 2, limitedBy: smiles.endIndex) else { return nil }
                    digitsEnd = end
                } else {
                    digitsEnd = next
                }
                guard let ring = Int(smiles[(ch == "%" ? next : i)..<digitsEnd]), let p = previous else { return nil }
                if let open = openRings.removeValue(forKey: ring) {
                    link(open.atom, p, pendingBond ?? open.bond)
                } else {
                    openRings[ring] = (p, pendingBond)
                }
                pendingBond = nil
                i = digitsEnd
            case "[":
                guard let close = smiles[i...].firstIndex(of: "]") else { return nil }
                let body = smiles[next..<close].drop { $0.isNumber }
                guard let first = body.first, first.isLetter || first == "*" else { return nil }
                addAtom(i..<smiles.index(after: close), symbol: String(first), isBracket: true)
                i = smiles.index(after: close)
            default:
                let twoLetter = smiles[i...].prefix(2)
                if twoLetter == "Cl" || twoLetter == "Br" {
                    let end = smiles.index(i, offsetBy: 2)
                    addAtom(i..<end, symbol: String(twoLetter), isBracket: false)
                    i = end
                } else if "BCNOPSFIbcnops*".contains(ch) {
                    addAtom(i..<next, symbol: String(ch), isBracket: false)
                    i = next
                } else {
                    return nil
                }
            }
        }
        guard branchStack.isEmpty, openRings.isEmpty else { return nil }

        // An edge is in a ring when its endpoints stay connected without it.
        inRing = Array(repeating: false, count: atoms.count)
        for (a, b) in edges where !(inRing[a] && inRing[b]) {
            var seen: Set<Int> = [a]
            var queue = [a]
            var connected = false
            while let current = queue.popLast(), !connected {
                for n in neighbors[current] where !seen.contains(n.atom) {
                    if current == a && n.atom == b { continue }
                    if n.atom == b { connected = true; break }
                    seen.insert(n.atom)
                    queue.append(n.atom)
                }
            }
            if connected {
                inRing[a] = true
                inRing[b] = true
            }
        }
    }

    /// Every site where `edit` applies, as (first edited atom, edited SMILES).
    func applying(_ edit: AnalogEdit, keepScaffold: Bool) -> [(atom: Int, smiles: String)] {
        func replaced(_ range: Range<String.Index>, with text: String) -> String {
            smiles.replacingCharacters(in: range, with: text)
        }
        func plainAtoms(_ symbol: String) -> [Int] {
            atoms.indices.filter { !atoms[$0].isBracket && atoms[$0].symbol == symbol }
        }

        switch edit {
        case .swapTerminalAtom(let from, let to):
            return plainAtoms(from)
                .filter { neighbors[$0].count == 1 }
                .map { ($0, replaced(atoms[$0].range, with: to)) }

        case .substituteAromaticCH(let substituent):
            return plainAtoms("c")
                .filter { neighbors[$0].count == 2 }
                .map { ($0, replaced(atoms[$0].range, with: "c(\(substituent))")) }

        case .branchOnMethyl(let branches):
            return plainAtoms("C")
                .filter { neighbors[$0].count == 1 && neighbors[$0][0].bond == "-" }
                .map { ($0, replaced(atoms[$0].range, with: "C\(branches)")) }

        case .swapAcyclicAtom(let from, let to):
            return plainAtoms(from)
                .filter { !inRing[$0] }
                .map { ($0, replaced(atoms[$0].range, with: to)) }

        case .swapAromaticAtom(let from, let to):
            return plainAtoms(from)
                .filter { neighbors[$0].count == 2 }
                .map { ($0, replaced(atoms[$0].range, with: to)) }

        case .replaceGroup(let pattern, let replacement):
            let patternHasRing = pattern.contains { $0.isNumber }
            var sites: [(atom: Int, smiles: String)] = []
            var searchStart = smiles.startIndex
            while let match = smiles.range(of: pattern, range: searchStart..<smiles.endIndex) {
                searchStart = smiles.index(after: match.lowerBound)
                // Anchor on whole atom tokens: the match must start at an atom token and
                // must not end inside one (e.g. "C" of "Cl", or a ring-closure digit that
                // would leave the ring half-replaced).
                guard let first = atoms.firstIndex(where: { $0.range.lowerBound == match.lowerBound }) else { continue }
                let covered = atoms.indices.filter { match.overlaps(atoms[$0].range) }
                guard covered.allSatisfy({ atoms[$0].range.upperBound <= match.upperBound }) else { continue }
                if match.upperBound < smiles.endIndex, smiles[match.upperBound].isNumber || smiles[match.upperBound] == "%" { continue }
                if !patternHasRing, covered.contains(where: { inRing[$0] }) { continue }
                if keepScaffold, patternHasRing { continue }
                sites.append((first, replaced(match, with: replacement)))
            }
            return sites
        }
    }
}

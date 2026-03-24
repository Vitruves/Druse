import SwiftUI

// MARK: - Lead Optimization (Analog Generation, Mini-Docking, Comparison)

extension AppViewModel {

    // MARK: - Reference Selection

    /// Set a docking result as the reference for lead optimization.
    /// Called from the Results tab "Optimize" button.
    func selectReferenceForOptimization(result: DockingResult, ligand: Molecule) {
        leadOpt.referenceResult = result
        leadOpt.referenceMolecule = ligand
        leadOpt.referenceName = ligand.name
        leadOpt.referenceSMILES = ligand.smiles ?? ligand.title
        leadOpt.referenceDescriptors = RDKitBridge.computeDescriptors(smiles: leadOpt.referenceSMILES)
        log.info("Reference for lead optimization: \(ligand.name)", category: .molecule)
    }

    /// Set a screening hit as the reference.
    func selectReferenceFromHit(_ hit: VirtualScreeningPipeline.ScreeningHit) {
        leadOpt.referenceSMILES = hit.smiles
        leadOpt.referenceName = hit.name
        leadOpt.referenceDescriptors = hit.descriptors
        if !hit.bestPoseAtoms.isEmpty {
            // Build a minimal DockingResult for the reference pose
            var refResult = DockingResult(
                id: 0,
                pose: DockPoseSwift(translation: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsions: []),
                energy: hit.bestEnergy, stericEnergy: 0, hydrophobicEnergy: 0,
                hbondEnergy: 0, torsionPenalty: 0, generation: 0,
                transformedAtomPositions: hit.bestPoseAtoms
            )
            refResult.mlDockingScore = hit.mlScore
            leadOpt.referenceResult = refResult
        }
        log.info("Reference for lead optimization: \(hit.name) (from screening)", category: .molecule)
    }

    // MARK: - Analog Generation

    /// SMARTS-like replacements with property-direction bias annotations.
    private static let replacementTable: [(pattern: String, replacement: String, label: String, bias: ReplacementBias)] = [
        // Halogen swaps
        ("F", "Cl", "F→Cl", .init(polarity: -0.2, lipophilicity: 0.3, size: 0.1)),
        ("Cl", "F", "Cl→F", .init(polarity: 0.2, lipophilicity: -0.3, size: -0.1)),
        ("F", "Br", "F→Br", .init(polarity: -0.3, lipophilicity: 0.4, size: 0.2)),
        ("Cl", "Br", "Cl→Br", .init(polarity: -0.1, lipophilicity: 0.2, size: 0.1)),
        ("Br", "Cl", "Br→Cl", .init(polarity: 0.1, lipophilicity: -0.2, size: -0.1)),
        // Alkyl extensions
        ("C", "CC", "Me→Et", .init(rigidity: 0, lipophilicity: 0.3, size: 0.3)),
        ("CC", "CCC", "Et→Pr", .init(rigidity: -0.2, lipophilicity: 0.4, size: 0.3)),
        ("CC", "C(C)C", "Et→iPr", .init(rigidity: 0.1, lipophilicity: 0.3, size: 0.2)),
        // Heteroatom swaps
        ("O", "N", "O→NH", .init(polarity: 0.3)),
        ("N", "O", "NH→O", .init(polarity: -0.2)),
        ("O", "S", "O→S", .init(polarity: -0.4, lipophilicity: 0.3, size: 0.1)),
        // Functional groups
        ("C", "C(F)(F)F", "Me→CF3", .init(polarity: -0.3, rigidity: 0.2, lipophilicity: 0.5, size: 0.2)),
        ("C(=O)O", "C(=O)N", "COOH→CONH2", .init(polarity: 0.2)),
        ("C(=O)N", "C(=O)O", "CONH2→COOH", .init(polarity: -0.1)),
        // Aromatic swaps
        ("c1ccccc1", "c1ccncc1", "Ph→Pyr", .init(polarity: 0.5, lipophilicity: -0.3)),
        ("c1ccncc1", "c1ccccc1", "Pyr→Ph", .init(polarity: -0.5, lipophilicity: 0.3)),
        // Sulfonyl / ether
        ("C(=O)", "S(=O)(=O)", "CO→SO2", .init(polarity: 0.3, size: 0.1)),
        ("COC", "CSC", "OMe→SMe", .init(polarity: -0.3, lipophilicity: 0.2)),
        ("CSC", "COC", "SMe→OMe", .init(polarity: 0.3, lipophilicity: -0.2)),
        // Ring size
        ("C1CCCC1", "C1CCCCC1", "cPent→cHex", .init(rigidity: 0.1, size: 0.2)),
        ("C1CCCCC1", "C1CCCC1", "cHex→cPent", .init(rigidity: -0.1, size: -0.2)),
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

    /// Generate property-directed analogs with ADMET filtering.
    func generateOptimizedAnalogs() {
        guard leadOpt.hasReference else { return }
        leadOpt.isGenerating = true
        leadOpt.generationProgress = 0
        leadOpt.analogs = []

        let smiles = leadOpt.referenceSMILES
        let name = leadOpt.referenceName
        let count = leadOpt.analogCount
        let refDesc = leadOpt.referenceDescriptors
        let polDir = leadOpt.polarityDirection
        let rigDir = leadOpt.rigidityDirection
        let lipDir = leadOpt.lipophilicityDirection
        let sizeDir = leadOpt.sizeDirection
        let filterConfig = (
            lipinski: leadOpt.filterLipinski, veber: leadOpt.filterVeber,
            herg: leadOpt.filterHERG, cyp: leadOpt.filterCYP,
            maxLogP: leadOpt.maxLogP, minSol: leadOpt.minSolubility
        )

        log.info("Generating up to \(count) optimized analogs of \(name)...", category: .molecule)
        workspace.statusMessage = "Generating analogs..."

        leadOpt.generationTask = Task {
            defer {
                self.leadOpt.isGenerating = false
                self.leadOpt.generationProgress = 1.0
            }

            // Score and sort replacements by alignment with property directions
            let hasDirectionBias = abs(polDir) + abs(rigDir) + abs(lipDir) + abs(sizeDir) > 0.01
            var scoredReplacements = Self.replacementTable.map { rep in
                (rep: rep, score: hasDirectionBias ? rep.bias.alignment(polDir: polDir, rigDir: rigDir, lipDir: lipDir, sizeDir: sizeDir) : 0.0)
            }
            if hasDirectionBias {
                scoredReplacements.sort { $0.score > $1.score }
            }

            var generated: [LeadOptAnalog] = []
            var seenSMILES = Set<String>([smiles])
            let total = Float(scoredReplacements.count)

            for (i, entry) in scoredReplacements.enumerated() {
                if generated.count >= count || Task.isCancelled { break }

                let rep = entry.rep
                guard smiles.contains(rep.pattern) else { continue }

                // Single substitution
                if let range = smiles.range(of: rep.pattern) {
                    let newSmi = smiles.replacingCharacters(in: range, with: rep.replacement)
                    if newSmi != smiles && !seenSMILES.contains(newSmi) {
                        seenSMILES.insert(newSmi)
                        var analog = LeadOptAnalog(
                            id: UUID(), name: "\(name)_\(rep.label)",
                            smiles: newSmi
                        )
                        self.computeDeltaProperties(for: &analog, reference: refDesc)
                        generated.append(analog)
                    }
                }

                // Global substitution
                if generated.count < count {
                    let allReplaced = smiles.replacingOccurrences(of: rep.pattern, with: rep.replacement)
                    if allReplaced != smiles && !seenSMILES.contains(allReplaced) {
                        seenSMILES.insert(allReplaced)
                        var analog = LeadOptAnalog(
                            id: UUID(), name: "\(name)_\(rep.label)_all",
                            smiles: allReplaced
                        )
                        self.computeDeltaProperties(for: &analog, reference: refDesc)
                        generated.append(analog)
                    }
                }

                self.leadOpt.generationProgress = Float(i + 1) / total * 0.5  // 50% for generation
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
            if activeCount > 0 {
                self.log.success("Generated \(activeCount) analogs (\(filteredCount) filtered out)", category: .molecule)
                self.workspace.statusMessage = "\(activeCount) analogs ready"
            } else {
                self.log.warn("No analogs passed filters", category: .molecule)
                self.workspace.statusMessage = "No analogs passed filters"
            }
        }
    }

    private func computeDeltaProperties(for analog: inout LeadOptAnalog, reference: LigandDescriptors?) {
        guard let desc = RDKitBridge.computeDescriptors(smiles: analog.smiles) else { return }
        analog.descriptors = desc
        if let ref = reference {
            analog.deltaMW = desc.molecularWeight - ref.molecularWeight
            analog.deltaLogP = desc.logP - ref.logP
            analog.deltaTPSA = desc.tpsa - ref.tpsa
            analog.deltaRotBonds = desc.rotatableBonds - ref.rotatableBonds
        }
    }

    // MARK: - Mini-Docking

    /// Dock all generated analogs using the existing grid maps and a light config.
    func dockAnalogs() {
        let activeAnalogs = leadOpt.analogs.enumerated().filter { $0.element.status == .generated }
        guard !activeAnalogs.isEmpty else { return }
        guard let engine = docking.dockingEngine, let pocket = docking.selectedPocket else {
            log.warn("Docking engine or pocket not available — dock a reference ligand first", category: .dock)
            return
        }

        leadOpt.isDocking = true
        leadOpt.dockingProgress = (0, activeAnalogs.count)
        log.info("Docking \(activeAnalogs.count) analogs (light config)...", category: .dock)
        workspace.statusMessage = "Docking analogs..."

        let refPositions = leadOpt.referenceResult?.transformedAtomPositions ?? []

        leadOpt.dockingTask = Task {
            defer {
                self.leadOpt.isDocking = false
                self.workspace.statusMessage = "Analog docking complete"
            }

            var lightConfig = DockingConfig()
            lightConfig.populationSize = 100
            lightConfig.numRuns = 1
            lightConfig.generationsPerRun = 80
            lightConfig.liveUpdateFrequency = 999 // no live updates for batch

            for (idx, analog) in activeAnalogs {
                if Task.isCancelled { break }
                self.leadOpt.dockingProgress = (idx + 1, activeAnalogs.count)

                // Prepare 3D molecule
                let (molData, _) = RDKitBridge.smilesToMolecule(
                    smiles: analog.smiles, name: analog.name, numConformers: 5, minimize: true
                )
                guard let md = molData else {
                    self.leadOpt.analogs[idx].status = .failed
                    continue
                }

                self.leadOpt.analogs[idx].molecule = md
                self.leadOpt.analogs[idx].status = .prepared

                let ligand = Molecule(name: analog.name, atoms: md.atoms, bonds: md.bonds, title: analog.smiles)

                // Dock (grid maps already computed for the reference)
                let results = await engine.runDocking(
                    ligand: ligand, pocket: pocket, config: lightConfig
                )

                if let best = results.first {
                    self.leadOpt.analogs[idx].dockingResults = Array(results.prefix(5))
                    self.leadOpt.analogs[idx].bestEnergy = best.energy
                    self.leadOpt.analogs[idx].bestPoseAtoms = best.transformedAtomPositions

                    // RMSD to reference pose (heavy atom overlap)
                    if !refPositions.isEmpty && !best.transformedAtomPositions.isEmpty {
                        let n = min(refPositions.count, best.transformedAtomPositions.count)
                        if n > 0 {
                            var sumSq: Float = 0
                            for i in 0..<n {
                                sumSq += simd_distance_squared(refPositions[i], best.transformedAtomPositions[i])
                            }
                            self.leadOpt.analogs[idx].rmsdToReference = sqrt(sumSq / Float(n))
                        }
                    }
                    self.leadOpt.analogs[idx].status = .docked
                } else {
                    self.leadOpt.analogs[idx].status = .failed
                }
            }

            // Sort analogs by energy (docked first, then generated, then filtered)
            self.leadOpt.analogs.sort { a, b in
                if a.status == .docked && b.status == .docked {
                    return (a.bestEnergy ?? 0) < (b.bestEnergy ?? 0)
                }
                return a.status == .docked && b.status != .docked
            }

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
    }
}

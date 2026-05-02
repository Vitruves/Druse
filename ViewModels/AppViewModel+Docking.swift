// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Pocket Detection, Docking, Pose Display, ML, Export

extension AppViewModel {

    // MARK: - Pocket Detection

    func detectPockets(excludedChainIDs: Set<String> = []) {
        guard let prot = molecules.protein else { return }
        let usesHybrid = excludedChainIDs.isEmpty && pocketDetectorML.isAvailable
        log.info(usesHybrid ? "Detecting binding pockets (hybrid)..." : "Detecting binding pockets...", category: .dock)
        workspace.statusMessage = "Detecting pockets..."

        Task {
            let geometricPockets = BindingSiteDetector.detectPockets(
                protein: prot,
                excludedChainIDs: excludedChainIDs
            )
            let mlPockets = usesHybrid ? await pocketDetectorML.detectPockets(protein: prot) : []
            let rankedCandidates = PocketSelectionHeuristics.rankedHybridCandidates(
                mlPockets: mlPockets,
                geometricPockets: geometricPockets
            )
            let pockets = rankedCandidates.map(\.pocket)

            docking.detectedPockets = pockets
            if let bestCandidate = rankedCandidates.first {
                let best = bestCandidate.pocket
                docking.selectedPocket = best
                showGridBoxForPocket(best)
                let methodLabel = bestCandidate.method == .ml ? "ML" : "geometric"
                log.success("Found \(pockets.count) pocket(s), best via \(methodLabel): \(String(format: "%.0f", best.volume)) ų, druggability: \(String(format: "%.1f", best.druggability))", category: .dock)
                log.info("  Best pocket: \(best.residueIndices.count) residues, center=(\(String(format: "%.1f, %.1f, %.1f", best.center.x, best.center.y, best.center.z)))", category: .dock)
            } else {
                renderer?.clearGridBox()
                log.warn("No pockets detected", category: .dock)
            }
            workspace.statusMessage = "\(pockets.count) pocket(s) found"
        }
    }

    func detectLigandGuidedPocket() {
        guard let prot = molecules.protein, let lig = molecules.ligand else { return }
        log.info("Defining pocket from ligand position...", category: .dock)

        if let pocket = BindingSiteDetector.ligandGuidedPocket(protein: prot, ligand: lig) {
            docking.detectedPockets = [pocket]
            docking.selectedPocket = pocket
            showGridBoxForPocket(pocket)
            log.success("Ligand-guided pocket: \(String(format: "%.0f", pocket.volume)) ų, \(pocket.residueIndices.count) residues", category: .dock)
        }
    }

    func pocketFromSelection() {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else { return }
        let pocket = BindingSiteDetector.pocketFromResidues(
            protein: prot, residueIndices: Array(workspace.selectedResidueIndices)
        )
        docking.detectedPockets = [pocket]
        docking.selectedPocket = pocket
        showGridBoxForPocket(pocket)
        log.success("Manual pocket from \(workspace.selectedResidueIndices.count) residues", category: .dock)
    }

    func showGridBoxForPocket(_ pocket: BindingPocket) {
        renderer?.gridLineWidth = workspace.gridLineWidth
        renderer?.updateGridBox(
            center: pocket.center,
            halfSize: pocket.size,
            color: workspace.gridColor
        )
    }

    /// Focus camera on pocket — stores object-space slab center but does NOT auto-enable clipping.
    func focusOnPocket(_ pocket: BindingPocket) {
        renderer?.focusOnPocket(center: pocket.center, halfExtent: pocket.size)
        // Sync slab values (but not enableClipping) so they're ready if user toggles clipping
        workspace.clipNearZ = renderer?.clipNearZ ?? 0
        workspace.clipFarZ = renderer?.clipFarZ ?? 100
        workspace.slabThickness = (renderer?.slabHalfThickness ?? 10.0) * 2.0
        workspace.slabOffset = renderer?.slabOffset ?? 0
        showGridBoxForPocket(pocket)
        log.info("Focused on pocket #\(pocket.id) (vol \(String(format: "%.0f", pocket.volume)) Å³)", category: .dock)
    }

    func updateGridBoxVisualization(center: SIMD3<Float>, halfSize: SIMD3<Float>) {
        renderer?.updateGridBox(
            center: center,
            halfSize: halfSize,
            color: SIMD4<Float>(0.2, 1.0, 0.4, 0.6)
        )
    }

    // MARK: - Prepared Protein Cache

    /// Returns a prepared protein for docking scoring, using a cache to avoid re-running
    /// the full preparation pipeline (H addition, H-bond network, Gasteiger charges)
    /// when the protein and pH haven't changed.
    func preparedProteinForDocking(
        protein: Molecule, pH: Float
    ) async -> (protein: Molecule, report: ProteinPreparation.DockingPreparationReport) {
        // Build a lightweight cache key from atom count + name + pH
        var hasher = Hasher()
        hasher.combine(protein.atoms.count)
        hasher.combine(protein.name)
        hasher.combine(pH)
        let key = hasher.finalize()

        if let cached = docking.preparedProteinCache, cached.key == key {
            log.info("[Preparation] Using cached prepared receptor (\(cached.protein.atoms.count) atoms)", category: .dock)
            return (cached.protein, cached.report)
        }

        let proteinAtoms = protein.atoms
        let proteinBonds = protein.bonds
        let pdbContent = molecules.rawPDBContent
        let hCount = proteinAtoms.filter { $0.element == .H }.count
        log.info("[Preparation] Preparing receptor for scoring (\(proteinAtoms.count) atoms, \(hCount) H, pH \(pH), rawPDB=\(pdbContent != nil ? "\(pdbContent!.count) chars" : "nil"))...", category: .dock)

        let t0 = CFAbsoluteTimeGetCurrent()
        let prepared = await Task.detached {
            ProteinPreparation.prepareForDocking(
                atoms: proteinAtoms,
                bonds: proteinBonds,
                rawPDBContent: pdbContent,
                pH: pH
            )
        }.value
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let scoringProtein = Molecule(
            name: protein.name,
            atoms: prepared.atoms,
            bonds: prepared.bonds,
            title: protein.title,
            smiles: protein.smiles
        )
        scoringProtein.secondaryStructureAssignments = protein.secondaryStructureAssignments

        docking.preparedProteinCache = .init(protein: scoringProtein, report: prepared.report, key: key)
        log.info("[Preparation] Receptor prepared in \(String(format: "%.1f", elapsed))s: \(prepared.atoms.count) atoms (+\(prepared.report.hydrogensAdded) H, \(prepared.report.nonZeroChargeAtoms) charged, \(prepared.report.rdkitChargeMatches) RDKit matches)", category: .dock)

        return (scoringProtein, prepared.report)
    }

    // MARK: - Docking

    func runDocking() {
        guard let pocket = docking.selectedPocket,
              let prot = molecules.protein,
              let lig = molecules.ligand
        else {
            log.error("Need protein, ligand, and pocket to dock", category: .dock)
            return
        }

        docking.originalDockingLigand = Molecule(name: lig.name, atoms: lig.atoms, bonds: lig.bonds, title: lig.title)

        docking.isDocking = true
        docking.dockingGeneration = 0
        docking.dockingTotalGenerations = docking.dockingConfig.numGenerations
        docking.dockingResults = []
        docking.currentInteractions = []
        docking.dockingBestEnergy = .infinity
        docking.dockingBestPKi = nil
        docking.dockingStartTime = Date()
        docking.dockingDuration = 0
        docking.selectedPoseIndices = []
        let cfg = docking.dockingConfig
        log.info("Starting docking: pop=\(cfg.populationSize), gen=\(cfg.numGenerations) (\(cfg.numRuns)×\(cfg.generationsPerRun)), grid=\(String(format: "%.3f", cfg.gridSpacing)) Å", category: .dock)
        log.info("  Ligand: \(lig.name) (\(lig.atoms.filter { $0.element != .H }.count) heavy atoms, \(lig.atoms.count) total, \(lig.bondCount) bonds)", category: .dock)
        log.info("  Pocket: center=(\(String(format: "%.1f, %.1f, %.1f", pocket.center.x, pocket.center.y, pocket.center.z))), size=(\(String(format: "%.1f, %.1f, %.1f", pocket.size.x, pocket.size.y, pocket.size.z))), volume=\(String(format: "%.0f", pocket.volume)) ų", category: .dock)
        log.info("  Scoring: \(docking.scoringMethod.rawValue), charges: \(docking.chargeMethod.rawValue)", category: .dock)
        log.info("  Config: localSearchFreq=\(cfg.localSearchFrequency), localSearchSteps=\(cfg.localSearchSteps), explorationRatio=\(String(format: "%.2f", cfg.explorationPhaseRatio))", category: .dock)
        log.info("  Config: torsionFlex=\(cfg.enableFlexibility), numRuns=\(cfg.numRuns)", category: .dock)

        Task {
            if docking.dockingEngine == nil, let device = renderer?.device {
                docking.dockingEngine = DockingEngine(device: device)
            }
            guard let engine = docking.dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                docking.isDocking = false
                docking.dockingBestEnergy = .infinity
                docking.dockingResults = []
                workspace.statusMessage = "Docking engine unavailable"
                return
            }

            let dockingPH = molecules.protonationPH
            let (scoringProtein, _) = await preparedProteinForDocking(protein: prot, pH: dockingPH)

            // Ensure grid box stays visible during docking; hide original ligand
            // (the live ghost pose renders the moving ligand in the pocket)
            showGridBoxForPocket(pocket)
            pushToRenderer()

            // Receptor flexibility: prepare flex buffers but keep ALL atoms in grid.
            // The flex scoring kernel computes a delta: pairwise(rotated) - pairwise(reference).
            // Since the grid includes flex atoms at reference positions, the delta captures
            // exactly the energy change from sidechain rotation. When chi=0, delta=0 → rigid docking.
            var flexConfig = docking.flexibleResidueConfig
            let isAutoFlex = flexConfig.autoFlex

            // Auto-flex: auto-select pocket-lining residues with rotatable sidechains
            if isAutoFlex && flexConfig.flexibleResidueIndices.isEmpty {
                let autoIndices = FlexibleResidueConfig.autoSelectResidues(
                    protein: scoringProtein.atoms,
                    pocket: (center: pocket.center, residueIndices: pocket.residueIndices)
                )
                flexConfig.flexibleResidueIndices = autoIndices
                if !autoIndices.isEmpty {
                    let names = autoIndices.compactMap { seq -> String? in
                        scoringProtein.atoms.first(where: { $0.residueSeq == seq }).map { "\($0.residueName)\(seq)" }
                    }
                    log.info("Auto-flex: selected \(autoIndices.count) pocket-lining residue(s): \(names.joined(separator: ", "))", category: .dock)
                }
            }

            let flexWeight = isAutoFlex ? FlexibleResidueConfig.softFlexWeight : Float(1.0)
            let chiStep = isAutoFlex ? FlexibleResidueConfig.softChiStep : FlexibleResidueConfig.fullChiStep

            if !flexConfig.flexibleResidueIndices.isEmpty {
                if docking.flexDockingEngine == nil, let device = renderer?.device {
                    docking.flexDockingEngine = FlexDockingEngine(device: device, commandQueue: engine.commandQueue)
                }
                if let flexEngine = docking.flexDockingEngine {
                    let vinaTypes = engine.vinaTypesForProtein(scoringProtein)
                    let exclusion = flexEngine.excludeFlexAtoms(
                        proteinAtoms: scoringProtein.atoms,
                        proteinBonds: scoringProtein.bonds,
                        flexConfig: flexConfig,
                        vinaTypes: vinaTypes
                    )
                    flexEngine.prepareFlexBuffers(exclusion: exclusion, flexWeight: flexWeight, chiStep: chiStep)
                    engine.flexEngine = flexEngine
                    let mode = isAutoFlex ? "soft" : "full"
                    log.info(
                        "Receptor flexibility (\(mode)): \(flexConfig.flexibleResidueIndices.count) residue(s), " +
                        "\(exclusion.flexAtoms.count) sidechain atoms, \(exclusion.chiSlotCount) chi angles " +
                        "(weight=\(String(format: "%.2f", flexWeight)), chiStep=\(String(format: "%.2f", chiStep)) rad)",
                        category: .dock
                    )
                }
            }

            // Auto-tune: adapt config to system complexity
            if docking.dockingConfig.autoMode {
                let heavyAtoms = lig.atoms.filter { $0.element != .H }.count
                let torsions = RDKitBridge.buildTorsionTree(smiles: lig.smiles ?? "")?.count ?? 0
                let tuned = DockingConfig.autoTune(
                    proteinAtomCount: scoringProtein.atoms.count,
                    pocketVolume: pocket.volume,
                    pocketBuriedness: pocket.buriedness,
                    ligandHeavyAtoms: heavyAtoms,
                    ligandRotatableBonds: torsions
                )
                // Preserve user's scoring/charge/flex choices, override search params
                docking.dockingConfig.populationSize = tuned.populationSize
                docking.dockingConfig.generationsPerRun = tuned.generationsPerRun
                docking.dockingConfig.numRuns = tuned.numRuns
                docking.dockingConfig.gridSpacing = tuned.gridSpacing
                docking.dockingConfig.localSearchFrequency = tuned.localSearchFrequency
                docking.dockingConfig.localSearchSteps = tuned.localSearchSteps
                docking.dockingConfig.explorationPhaseRatio = tuned.explorationPhaseRatio
                docking.dockingConfig.explorationTranslationStep = tuned.explorationTranslationStep
                docking.dockingConfig.explorationMutationRate = tuned.explorationMutationRate
                docking.dockingTotalGenerations = tuned.numGenerations
                log.info("Auto-tune: pop=\(tuned.populationSize) gen=\(tuned.generationsPerRun) runs=\(tuned.numRuns) " +
                         "(protein=\(scoringProtein.atoms.count) atoms, pocket=\(String(format: "%.0f", pocket.volume))ų " +
                         "buried=\(String(format: "%.2f", pocket.buriedness)), ligand=\(heavyAtoms) heavy/\(torsions) torsions)",
                         category: .dock)
            }

            log.info("Computing grid maps (spacing=\(String(format: "%.2f", docking.dockingConfig.gridSpacing)) Å)...", category: .dock)
            engine.computeGridMaps(protein: scoringProtein, pocket: pocket, spacing: docking.dockingConfig.gridSpacing)
            log.success("Grid maps computed — \(scoringProtein.atoms.count) receptor atoms", category: .dock)

            let origLig = self.docking.originalDockingLigand ?? lig

            log.info("  Original ligand: \(origLig.name), \(origLig.atoms.count) atoms, \(origLig.bonds.count) bonds", category: .dock)

            engine.onPoseUpdate = { [weak self] result, interactions in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.docking.dockingBestEnergy = result.energy
                    if self.docking.scoringMethod.isAffinityScore {
                        self.docking.dockingBestPKi = result.stericEnergy  // pKd stored by DruseAF shader
                    }
                    self.docking.currentInteractions = interactions
                    // Use lightweight ghost pose pipeline — avoids full pushToRenderer() rebuild
                    if let (newAtoms, newBonds) = self.buildTransformedLigand(result: result, originalLigand: origLig) {
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

            // Prepare pharmacophore constraint buffers (zero-cost if no constraints)
            let activeConstraints = docking.pharmacophoreConstraints.filter(\.isEnabled)
            engine.prepareConstraintBuffers(
                activeConstraints,
                atoms: scoringProtein.atoms,
                residues: scoringProtein.residues
            )
            if !activeConstraints.isEmpty {
                log.info("Pharmacophore constraints: \(activeConstraints.count) active", category: .dock)
            }

            log.info("Launching GA docking engine...", category: .dock)

            // One entry = one docking job (flat model)
            let allResults = await engine.runDocking(
                ligand: lig, pocket: pocket,
                config: docking.dockingConfig,
                scoringMethod: docking.scoringMethod)

            let results = allResults.sorted { $0.energy < $1.energy }

            log.info("GA complete: \(results.count) poses returned", category: .dock)
            if let best = results.first {
                let method = docking.scoringMethod
                let scoreStr = method.isAffinityScore
                    ? "pKi=\(String(format: "%.3f", best.displayScore(method: method)))"
                    : "E=\(String(format: "%.3f", best.energy)) kcal/mol"
                log.info("  Best pose: \(scoreStr), cluster=\(best.clusterID), gen=\(best.generation)",
                         category: .dock)
            }

            // Signal that GA search is done — post-processing begins
            workspace.statusMessage = "Post-processing docking results..."

            var rankedResults = results

            let openMMRefiner = OpenMMPocketRefiner.shared
            if openMMRefiner.isAvailable {
                workspace.statusMessage = "OpenMM refinement..."
                let refinement = await refineTopDockingResultsWithOpenMM(
                    results: rankedResults,
                    protein: scoringProtein,
                    originalLigand: origLig,
                    pocket: pocket
                )
                rankedResults = refinement.results
                if let refinedPositions = refinement.bestProteinHeavyPositions,
                   let updatedProteinAtoms = applyingRefinedHeavyAtomPositions(refinedPositions, to: prot.atoms) {
                    let refinedProtein = Molecule(name: prot.name, atoms: updatedProteinAtoms, bonds: prot.bonds, title: prot.title)
                    refinedProtein.secondaryStructureAssignments = prot.secondaryStructureAssignments
                    molecules.protein = refinedProtein
                    pushToRenderer()
                }
            } else {
                log.info("OpenMM refinement skipped: \(openMMRefiner.availabilitySummary)", category: .dock)
            }

            // MMFF94 ligand strain penalty (post-docking, CPU-side)
            let strainSmiles = origLig.smiles ?? (origLig.title.isEmpty ? nil : origLig.title)
            if docking.dockingConfig.strainPenaltyEnabled,
               docking.scoringMethod != .vina,
               let smiles = strainSmiles {
                rankedResults = await computeStrainPenalties(
                    results: rankedResults, smiles: smiles, config: docking.dockingConfig
                )
            }

            // GFN2-xTB post-docking refinement (geometry opt + D4 + solvation scoring)
            let gfn2Config = docking.dockingConfig.gfn2Refinement
            if gfn2Config.enabled {
                workspace.statusMessage = "GFN2-xTB refinement..."
                log.info("GFN2-xTB post-docking refinement: top \(gfn2Config.topPosesToRefine) poses, \(gfn2Config.solvation.rawValue), opt=\(gfn2Config.optLevel.rawValue)", category: .dock)
                rankedResults = await refineWithGFN2(
                    results: rankedResults,
                    originalLigand: origLig,
                    config: gfn2Config
                )
                let refined = rankedResults.prefix(gfn2Config.topPosesToRefine).filter { $0.gfn2Converged == true }.count
                log.success("GFN2-xTB refinement complete: \(refined)/\(min(gfn2Config.topPosesToRefine, rankedResults.count)) converged", category: .dock)
            }

            // GFN2 single-point energy on best pose (always-on, ~2ms, informational)
            if let bestResult = rankedResults.first {
                let spHeavyAtoms = origLig.atoms.filter { $0.element != .H }
                let spPositions = bestResult.transformedAtomPositions
                if spHeavyAtoms.count >= 2, spPositions.count == spHeavyAtoms.count {
                    var spAtoms = spHeavyAtoms
                    for j in 0..<spAtoms.count { spAtoms[j].position = spPositions[j] }
                    let formalCharge = spHeavyAtoms.reduce(0) { $0 + $1.formalCharge }
                    if let sp = try? await GFN2Refiner.computeEnergy(
                        atoms: spAtoms, totalCharge: formalCharge, solvation: .water
                    ) {
                        rankedResults[0].gfn2Energy = sp.totalEnergy_kcal
                        rankedResults[0].gfn2DispersionEnergy = sp.dispersionEnergy * 627.509
                        rankedResults[0].gfn2SolvationEnergy = sp.solvationEnergy * 627.509
                        rankedResults[0].gfn2Converged = sp.converged
                        log.info(String(format: "GFN2-xTB best pose: %.1f kcal/mol (D4:%.2f, solv:%.2f)",
                                        sp.totalEnergy_kcal, sp.dispersionEnergy * 627.509, sp.solvationEnergy * 627.509),
                                 category: .dock)
                    }
                }
            }

            docking.dockingResults = rankedResults
            docking.isDocking = false
            docking.dockingDuration = Date().timeIntervalSince(docking.dockingStartTime ?? Date())

            if let best = docking.dockingResults.first {
                applyDockingPose(best, originalLigand: origLig)

                if molecules.protein != nil {
                    let heavyAtoms = origLig.atoms.filter { $0.element != .H }
                    let heavyBonds = buildHeavyBonds(from: origLig)
                    docking.currentInteractions = InteractionDetector.detect(
                        ligandAtoms: heavyAtoms,
                        ligandPositions: best.transformedAtomPositions,
                        proteinAtoms: scoringProtein.atoms.filter { $0.element != .H },
                        ligandBonds: heavyBonds
                    )
                    renderer?.updateInteractionLines(docking.currentInteractions)
                }
            }

            if let pocket = docking.selectedPocket {
                showGridBoxForPocket(pocket)
            }

            let clusterCount = Set(results.map(\.clusterID)).count
            let elapsed = Date().timeIntervalSince(docking.dockingStartTime ?? Date())
            if let best = rankedResults.first {
                let method = docking.scoringMethod
                let bestDisplay = best.displayScore(method: method)
                let unit = method.unitLabel
                if method == .drusina {
                    let drCorr = best.drusinaCorrection
                    log.success(String(format: "Docking complete (Drusina): best %.1f %@ (correction: %.2f), %d poses, %d clusters (%.1fs)",
                                       bestDisplay, unit, drCorr, rankedResults.count, clusterCount, elapsed), category: .dock)
                } else if method.isAffinityScore {
                    log.success(String(format: "Docking complete (DruseAF): best pKi %.1f (conf: %.0f%%), %d poses, %d clusters (%.1fs)",
                                       bestDisplay, best.afConfidence * 100, rankedResults.count, clusterCount, elapsed), category: .dock)
                } else {
                    log.success(String(format: "Docking complete: best %.1f %@, %d poses, %d clusters (%.1fs)",
                                       bestDisplay, unit, rankedResults.count, clusterCount, elapsed), category: .dock)
                }
                workspace.statusMessage = String(format: "Best: %.1f %@", bestDisplay, unit)
                if docking.currentInteractions.count > 0 {
                    let hbonds = docking.currentInteractions.filter { $0.type == .hbond }.count
                    let hydro = docking.currentInteractions.filter { $0.type == .hydrophobic }.count
                    let pipi = docking.currentInteractions.filter { $0.type == .piStack }.count
                    log.info("  Best pose interactions: \(hbonds) H-bonds, \(hydro) hydrophobic, \(pipi) π-stacking", category: .dock)
                }
            }
        }
    }

    func stopDocking() {
        docking.dockingEngine?.stopDocking()
        docking.isDocking = false
        docking.liveScoringTask?.cancel()
        docking.liveScoringTask = nil
        renderer?.clearGhostPose()
        pushToRenderer()
        log.info("Docking stopped", category: .dock)
    }

    // (Ensemble machinery removed — each entry is docked independently as one job)

    /// Remove all displayed poses and interaction lines from the 3D viewport.
    /// The docking results data is preserved for re-viewing later.
    func clearPosesFromView() {
        // Preserve the ligand template so "View" on a pose card can rebuild it.
        // After a `.druse` project load, `originalDockingLigand` is nil — without
        // this snapshot, clearing the view would also strand any saved poses.
        if docking.originalDockingLigand == nil, let lig = molecules.ligand {
            docking.originalDockingLigand = Molecule(
                name: lig.name, atoms: lig.atoms, bonds: lig.bonds, title: lig.title
            )
        }
        molecules.ligand = nil
        docking.currentInteractions = []
        docking.selectedPoseIndices = []
        renderer?.updateInteractionLines([])
        renderer?.clearGhostPose()
        pushToRenderer()
        workspace.statusMessage = "Poses cleared from view"
        log.info("Cleared poses from view (\(docking.dockingResults.count) results retained)", category: .dock)
    }

    func showDockingPose(at index: Int) {
        guard index < docking.dockingResults.count else { return }
        guard let origLig = docking.originalDockingLigand ?? molecules.ligand else { return }
        let result = docking.dockingResults[index]
        applyDockingPose(result, originalLigand: origLig)

        if let prot = molecules.protein {
            let heavyAtoms = origLig.atoms.filter { $0.element != .H }
            let heavyBonds = buildHeavyBonds(from: origLig)
            docking.currentInteractions = InteractionDetector.detect(
                ligandAtoms: heavyAtoms,
                ligandPositions: result.transformedAtomPositions,
                proteinAtoms: prot.atoms.filter { $0.element != .H },
                ligandBonds: heavyBonds
            )
            renderer?.updateInteractionLines(docking.currentInteractions)

            if workspace.renderMode == .ribbon && workspace.sideChainDisplay == .interacting {
                pushToRenderer()
            }
        }

        let method = docking.scoringMethod
        workspace.statusMessage = String(format: "Pose #%d: %.1f %@", index + 1, result.displayScore(method: method), method.unitLabel)
        log.info(String(format: "Showed pose #%d (%.2f %@, %d interactions)", index + 1, result.displayScore(method: method), method.unitLabel, docking.currentInteractions.count), category: .dock)
    }

    func togglePoseSelection(at index: Int) {
        let added: Bool
        if docking.selectedPoseIndices.contains(index) {
            docking.selectedPoseIndices.remove(index)
            added = false
        } else {
            docking.selectedPoseIndices.insert(index)
            added = true
        }
        log.info("Pose #\(index + 1) \(added ? "added to" : "removed from") multi-select (\(docking.selectedPoseIndices.count) total)", category: .dock)
    }

    func showSelectedPoses() {
        guard !docking.selectedPoseIndices.isEmpty else { return }
        guard let origLig = docking.originalDockingLigand ?? molecules.ligand else { return }

        let sortedIndices = docking.selectedPoseIndices.sorted { a, b in
            guard a < docking.dockingResults.count, b < docking.dockingResults.count else { return a < b }
            return docking.dockingResults[a].energy < docking.dockingResults[b].energy
        }

        if let primaryIdx = sortedIndices.first, primaryIdx < docking.dockingResults.count {
            let result = docking.dockingResults[primaryIdx]
            applyDockingPose(result, originalLigand: origLig)

            if let prot = molecules.protein {
                let heavyAtoms = origLig.atoms.filter { $0.element != .H }
                let heavyBonds = buildHeavyBonds(from: origLig)
                docking.currentInteractions = InteractionDetector.detect(
                    ligandAtoms: heavyAtoms,
                    ligandPositions: result.transformedAtomPositions,
                    proteinAtoms: prot.atoms.filter { $0.element != .H },
                    ligandBonds: heavyBonds
                )
                renderer?.updateInteractionLines(docking.currentInteractions)
            }
        }

        var ghostAtoms: [Atom] = []
        var ghostBonds: [Bond] = []
        for idx in sortedIndices.dropFirst() {
            guard idx < docking.dockingResults.count else { continue }
            let result = docking.dockingResults[idx]
            if let (atoms, bonds) = buildTransformedLigand(result: result, originalLigand: origLig) {
                let offset = ghostAtoms.count
                for atom in atoms {
                    ghostAtoms.append(Atom(
                        id: offset + atom.id, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom
                    ))
                }
                for bond in bonds {
                    ghostBonds.append(Bond(
                        id: ghostBonds.count,
                        atomIndex1: bond.atomIndex1 + offset,
                        atomIndex2: bond.atomIndex2 + offset,
                        order: bond.order
                    ))
                }
            }
        }

        if !ghostAtoms.isEmpty {
            renderer?.updateGhostPose(atoms: ghostAtoms, bonds: ghostBonds)
        }

        workspace.statusMessage = "\(docking.selectedPoseIndices.count) poses displayed"
    }

    func refineTopDockingResultsWithOpenMM(
        results: [DockingResult],
        protein: Molecule,
        originalLigand: Molecule,
        pocket: BindingPocket
    ) async -> (results: [DockingResult], bestProteinHeavyPositions: [SIMD3<Float>]?) {
        guard !results.isEmpty else { return (results, nil) }

        var refinedResults = results
        var refinedProteinByResultID: [Int: [SIMD3<Float>]] = [:]
        let refiner = OpenMMPocketRefiner.shared
        let topN = min(3, refinedResults.count)

        log.info("Refining top \(topN) docking poses with OpenMM pocket minimization...", category: .dock)
        for index in 0..<topN {
            let result = refinedResults[index]
            let heavyAtoms = originalLigand.atoms.filter { $0.element != .H }
            var ligandAtoms = heavyAtoms
            for atomIndex in 0..<min(ligandAtoms.count, result.transformedAtomPositions.count) {
                ligandAtoms[atomIndex].position = result.transformedAtomPositions[atomIndex]
            }

            if let refined = await refiner.refine(
                proteinAtoms: protein.atoms,
                ligandAtoms: ligandAtoms,
                pocketCenter: pocket.center,
                pocketHalfExtent: pocket.size
            ) {
                refinedResults[index].refinementEnergy = refined.interactionEnergyKcal
                refinedProteinByResultID[result.id] = refined.refinedHeavyAtomPositions
            }
        }

        refinedResults.sort { lhs, rhs in
            let lhsScore = lhs.refinementEnergy ?? lhs.energy
            let rhsScore = rhs.refinementEnergy ?? rhs.energy
            return lhsScore < rhsScore
        }

        let bestProteinHeavyPositions = refinedResults.first.flatMap { refinedProteinByResultID[$0.id] }
        return (refinedResults, bestProteinHeavyPositions)
    }

    func buildTransformedLigand(result: DockingResult, originalLigand: Molecule) -> (atoms: [Atom], bonds: [Bond])? {
        guard !result.transformedAtomPositions.isEmpty else { return nil }

        // Build index mapping from original array position → new heavy-atom index.
        // Bond.atomIndex1/2 are array positions, NOT atom IDs — we must map by position.
        var originalIdxToNew: [Int: Int] = [:]
        var newAtoms: [Atom] = []
        for (origIdx, atom) in originalLigand.atoms.enumerated() {
            guard atom.element != .H else { continue }
            let newIdx = newAtoms.count
            guard newIdx < result.transformedAtomPositions.count else { break }
            originalIdxToNew[origIdx] = newIdx
            newAtoms.append(Atom(
                id: newIdx, element: atom.element, position: result.transformedAtomPositions[newIdx],
                name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                isHetAtom: atom.isHetAtom
            ))
        }

        var newBonds: [Bond] = []
        for bond in originalLigand.bonds {
            if let a = originalIdxToNew[bond.atomIndex1], let b = originalIdxToNew[bond.atomIndex2] {
                newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }
        return (newAtoms, newBonds)
    }

    func applyDockingPose(_ result: DockingResult, originalLigand: Molecule) {
        guard let (newAtoms, newBonds) = buildTransformedLigand(result: result, originalLigand: originalLigand) else { return }

        renderer?.clearGhostPose()
        molecules.ligand = Molecule(name: originalLigand.name, atoms: newAtoms, bonds: newBonds, title: originalLigand.title)
        pushToRenderer()

        if !docking.currentInteractions.isEmpty {
            renderer?.updateInteractionLines(docking.currentInteractions)
        }
    }

    func applyLiveDockingPose(_ result: DockingResult, originalLigand: Molecule) {
        guard let (newAtoms, newBonds) = buildTransformedLigand(result: result, originalLigand: originalLigand) else { return }

        molecules.ligand = Molecule(name: originalLigand.name, atoms: newAtoms, bonds: newBonds, title: originalLigand.title)
        pushToRenderer()

        if !docking.currentInteractions.isEmpty {
            renderer?.updateInteractionLines(docking.currentInteractions)
        }
    }

    // MARK: - Export Results

    func exportResultsCSV() -> String {
        var rows: [String] = ["Rank,Cluster,Energy_kcal_mol,VdW,HBond,Torsion,StrainEnergy,Generation,ML_pKd,ML_Confidence,ML_DockingScore"]
        rows.reserveCapacity(docking.dockingResults.count + 1)
        for (i, r) in docking.dockingResults.enumerated() {
            let fields: [String] = [
                "\(i+1)",
                "\(r.clusterID)",
                String(format: "%.2f", r.energy),
                String(format: "%.2f", r.vdwEnergy),
                String(format: "%.2f", r.hbondEnergy),
                String(format: "%.2f", r.torsionPenalty),
                r.strainEnergy.map { String(format: "%.2f", $0) } ?? "",
                "\(r.generation)",
                "",
                "",
                ""
            ]
            rows.append(fields.joined(separator: ","))
        }
        return rows.joined(separator: "\n") + "\n"
    }

    // MARK: - ML Model Loading

    func loadMLModels() {
        pocketDetectorML.loadModel()
        admetPredictor.loadModels()
        if pocketDetectorML.isAvailable {
            log.success("PocketDetector ML model loaded", category: .dock)
        }
        if admetPredictor.isAvailable {
            log.success("ADMET models loaded", category: .dock)
        }
    }

    // MARK: - ML Pocket Detection

    // MARK: - Strain Penalty

    /// Compute MMFF94 strain energy for top docking poses and apply penalty to scores.
    // MARK: - GFN2-xTB Post-Docking Refinement

    /// Refine top docked poses using GFN2-xTB geometry optimization.
    ///
    /// For each top pose:
    /// 1. Build a temporary Atom array with the docked ligand coordinates
    /// 2. Run GFN2-xTB optimization (with D4 dispersion + implicit solvation)
    /// 3. Store energy decomposition and optionally update coordinates
    /// 4. Blend GFN2 energy into ranking score
    func refineWithGFN2(
        results: [DockingResult],
        originalLigand: Molecule,
        config: GFN2RefinementConfig
    ) async -> [DockingResult] {
        let topN = min(results.count, config.topPosesToRefine)
        guard topN > 0 else { return results }

        let heavyAtoms = originalLigand.atoms.filter { $0.element != .H }
        guard heavyAtoms.count >= 2 else { return results }

        let formalCharge = originalLigand.atoms.reduce(0) { $0 + $1.formalCharge }

        var updated = results

        // Process top poses in parallel (each on its own detached task)
        let refinements: [(Int, GFN2RefinementResult?)] = await withTaskGroup(
            of: (Int, GFN2RefinementResult?).self
        ) { group in
            for i in 0..<topN {
                let positions = results[i].transformedAtomPositions
                guard positions.count == heavyAtoms.count else { continue }

                group.addTask {
                    // Build temporary atom array with docked coordinates
                    var dockedAtoms = heavyAtoms
                    for j in 0..<dockedAtoms.count {
                        dockedAtoms[j].position = positions[j]
                    }

                    do {
                        let result = try await GFN2Refiner.optimizeGeometry(
                            atoms: dockedAtoms,
                            totalCharge: formalCharge,
                            solvation: config.solvation,
                            optLevel: config.optLevel,
                            maxSteps: config.maxSteps,
                            referencePositions: config.restraintStrength > 0 ? positions : nil,
                            restraintStrength: config.restraintStrength
                        )
                        return (i, result)
                    } catch {
                        return (i, nil)
                    }
                }
            }

            var collected = [(Int, GFN2RefinementResult?)]()
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        // Apply results
        // Compute reference energy for blending (first successful result as baseline)
        var gfn2Energies = [Float]()
        for (i, gfn2Result) in refinements {
            guard let gfn2Result else { continue }

            updated[i].gfn2Energy = gfn2Result.totalEnergy_kcal
            updated[i].gfn2DispersionEnergy = gfn2Result.dispersionEnergy * 627.509
            updated[i].gfn2SolvationEnergy = gfn2Result.solvationEnergy * 627.509
            updated[i].gfn2Converged = gfn2Result.converged
            updated[i].gfn2OptSteps = gfn2Result.steps
            gfn2Energies.append(gfn2Result.totalEnergy_kcal)

            // Update coordinates if requested, with RMSD guard
            if config.updateCoordinates, let optPos = gfn2Result.optimizedPositions {
                // Compute heavy-atom RMSD between docked and optimized pose
                let origPos = updated[i].transformedAtomPositions
                if origPos.count == optPos.count {
                    var sumSq: Float = 0
                    for j in 0..<origPos.count {
                        sumSq += simd_distance_squared(origPos[j], optPos[j])
                    }
                    let rmsd = sqrt(sumSq / Float(origPos.count))
                    if rmsd <= config.maxRMSD {
                        updated[i].transformedAtomPositions = optPos
                    }
                    // else: RMSD too large, keep original docked coordinates
                }
            }
        }

        // Blend GFN2 energy into ranking: use relative GFN2 energy (ΔE from best)
        if let minGFN2 = gfn2Energies.min(), config.blendWeight > 0 {
            for i in 0..<updated.count {
                if let gfn2E = updated[i].gfn2Energy {
                    let relativeGFN2 = gfn2E - minGFN2  // relative to best GFN2 pose
                    updated[i].energy += config.blendWeight * relativeGFN2
                }
            }
            // Re-sort after blending
            updated.sort { $0.energy < $1.energy }
        }

        return updated
    }

    func computeStrainPenalties(
        results: [DockingResult], smiles: String, config: DockingConfig
    ) async -> [DockingResult] {
        workspace.statusMessage = "Computing ligand strain..."

        let topN = min(results.count, 50) // only compute for top 50 poses (CPU-bound)
        let threshold = config.strainPenaltyThreshold
        let weight = config.strainPenaltyWeight

        // Compute reference energy once (free ligand, relaxed)
        let refEnergy = await Task.detached(priority: .userInitiated) {
            RDKitBridge.mmffReferenceEnergy(smiles: smiles)
        }.value

        guard let refEnergy else {
            log.warn("MMFF reference energy failed for \(smiles.prefix(40)) — skipping strain", category: .dock)
            return results
        }

        // Compute strain for each top pose
        var updated = results
        let positionsSlice = results.prefix(topN).map { $0.transformedAtomPositions }

        let strainResults: [(Int, Float?)] = await Task.detached(priority: .userInitiated) {
            positionsSlice.enumerated().map { (i, positions) in
                guard !positions.isEmpty else { return (i, nil as Float?) }
                if let dockedEnergy = RDKitBridge.mmffStrainEnergy(smiles: smiles, heavyPositions: positions) {
                    let strain = Float(dockedEnergy - refEnergy)
                    return (i, strain)
                }
                return (i, nil as Float?)
            }
        }.value

        var strainedCount = 0
        for (i, strain) in strainResults {
            guard let strain else { continue }
            updated[i].strainEnergy = strain
            if strain > threshold {
                let penalty = weight * (strain - threshold)
                updated[i].energy += penalty
                strainedCount += 1
            }
        }

        if strainedCount > 0 {
            // Re-sort since energies changed
            updated.sort { $0.energy < $1.energy }
            log.info("Strain penalty applied: \(strainedCount)/\(topN) poses exceeded \(String(format: "%.0f", threshold)) kcal/mol threshold (ref: \(String(format: "%.1f", refEnergy)) kcal/mol)", category: .dock)
        } else {
            log.info("Strain check: all \(topN) poses within \(String(format: "%.0f", threshold)) kcal/mol threshold (ref: \(String(format: "%.1f", refEnergy)) kcal/mol)", category: .dock)
        }
        return updated
    }

    // MARK: - ML Pocket Detection

    func detectPocketsML() {
        guard let prot = molecules.protein else { return }
        guard pocketDetectorML.isAvailable else {
            log.warn("PocketDetector model not available, falling back to auto detection", category: .dock)
            detectPockets()
            return
        }
        log.info("Detecting pockets (ML)...", category: .dock)
        workspace.statusMessage = "ML pocket detection..."

        Task {
            let pockets = await pocketDetectorML.detectPockets(protein: prot)
            docking.detectedPockets = pockets
            if let best = pockets.first {
                docking.selectedPocket = best
                showGridBoxForPocket(best)
                log.success("ML found \(pockets.count) pocket(s), best: \(String(format: "%.0f", best.volume)) ų", category: .dock)
            } else {
                renderer?.clearGridBox()
                log.warn("ML detected no pockets, try geometric method", category: .dock)
            }
            workspace.statusMessage = "\(pockets.count) pocket(s) found (ML)"
        }
    }
}

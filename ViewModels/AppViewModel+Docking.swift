// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import Metal

// MARK: - Pocket Detection, Docking, Pose Display, ML, Export

extension AppViewModel {

    /// Metal device for compute work: the renderer's device in the GUI, the system default
    /// device when no renderer exists (headless runs).
    var computeDevice: MTLDevice? {
        renderer?.device ?? MTLCreateSystemDefaultDevice()
    }

    // MARK: - Pocket Detection

    /// Detects pockets (hybrid ML + geometric when `allowML` and no chain is excluded).
    /// Returns nil without a protein; otherwise a task that completes once pockets are set.
    @discardableResult
    func detectPockets(excludedChainIDs: Set<String> = [], allowML: Bool = true) -> Task<Void, Never>? {
        guard let prot = molecules.protein else { return nil }
        let usesHybrid = allowML && excludedChainIDs.isEmpty && pocketDetectorML.isAvailable
        log.info(usesHybrid ? "Detecting binding pockets (hybrid)..." : "Detecting binding pockets...", category: .dock)
        workspace.statusMessage = "Detecting pockets..."

        return Task {
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

    /// Returns whether a pocket could be defined around the current ligand.
    @discardableResult
    func detectLigandGuidedPocket() -> Bool {
        guard let prot = molecules.protein, let lig = molecules.ligand else { return false }
        log.info("Defining pocket from ligand position...", category: .dock)

        guard let pocket = BindingSiteDetector.ligandGuidedPocket(protein: prot, ligand: lig) else { return false }
        docking.detectedPockets = [pocket]
        docking.selectedPocket = pocket
        showGridBoxForPocket(pocket)
        log.success("Ligand-guided pocket: \(String(format: "%.0f", pocket.volume)) ų, \(pocket.residueIndices.count) residues", category: .dock)
        return true
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
        protein: Molecule, pH: Float, chargeMethod: ChargeMethod
    ) async -> (protein: Molecule, report: ProteinPreparation.DockingPreparationReport) {
        // Cache key must reflect all inputs that change the prepared receptor.
        var hasher = Hasher()
        hasher.combine(protein.name)
        hasher.combine(pH)
        hasher.combine(chargeMethod)
        hasher.combine(molecules.rawPDBContent)
        hasher.combine(protein.atoms.count)
        for atom in protein.atoms {
            hasher.combine(atom.id)
            hasher.combine(atom.element.rawValue)
            hasher.combine(atom.name)
            hasher.combine(atom.residueName)
            hasher.combine(atom.residueSeq)
            hasher.combine(atom.chainID)
            hasher.combine(atom.position.x)
            hasher.combine(atom.position.y)
            hasher.combine(atom.position.z)
            hasher.combine(atom.formalCharge)
        }
        hasher.combine(protein.bonds.count)
        for bond in protein.bonds {
            hasher.combine(bond.atomIndex1)
            hasher.combine(bond.atomIndex2)
            hasher.combine(bond.order.rawValue)
        }
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
        let metalDevice = computeDevice
        let prepared = await Task.detached {
            ProteinPreparation.prepareForDocking(
                atoms: proteinAtoms,
                bonds: proteinBonds,
                rawPDBContent: pdbContent,
                pH: pH,
                chargeMethod: chargeMethod,
                device: metalDevice
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

    private func nonEmptySMILES(_ smiles: String?) -> String? {
        guard let trimmed = smiles?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else { return nil }
        return trimmed
    }

    private func looksLikeInlineSMILES(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        return trimmed.rangeOfCharacter(from: .whitespacesAndNewlines) == nil
    }

    /// Fragment docking requires SMILES, but the active ligand can be rebuilt
    /// from pose coordinates after a previous docking run. Recover the database
    /// SMILES before passing that molecule to the engine.
    private func ensureLigandHasSMILES(_ ligand: Molecule) -> Molecule {
        if nonEmptySMILES(ligand.smiles) != nil { return ligand }

        let resolved: String?
        if let activeLigandEntryID,
           let entry = ligandDB.entries.first(where: { $0.id == activeLigandEntryID }),
           let smiles = nonEmptySMILES(entry.smiles) {
            resolved = smiles
        } else if let entry = ligandDB.entries.first(where: { $0.name == ligand.name }),
                  let smiles = nonEmptySMILES(entry.smiles) {
            resolved = smiles
        } else if looksLikeInlineSMILES(ligand.title) {
            resolved = ligand.title.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            resolved = nil
        }

        guard let resolved else { return ligand }
        ligand.smiles = resolved
        if ligand.title.isEmpty {
            ligand.title = resolved
        }
        log.info("Recovered SMILES for \(ligand.name) from ligand metadata", category: .dock)
        return ligand
    }

    /// Ring conformations docked per run.
    static let maxRingConformationVariants = 4

    /// The distinct ring conformations of `ligand` (first) and of the active ligand's prepared
    /// conformers, plus mirror-image folds of achiral molecules, as copies of `ligand` with
    /// new coordinates. Just `ligand` when no ring is non-planar.
    private func ringConformationVariants(of ligand: Molecule) -> [Molecule] {
        let elements = ligand.atoms.map(\.element)
        let ensemble = [ligand.atoms.map(\.position)] + workspace.ligandConformers
            .filter { $0.atoms.map(\.element) == elements }
            .map { $0.atoms.map(\.position) }
        let molBlock = SDFWriter.molBlock(name: ligand.name, atoms: ligand.atoms, bonds: ligand.bonds)
        guard let variants = RDKitBridge.ringConformationVariants(
            molBlock: molBlock, conformers: ensemble,
            maxVariants: Self.maxRingConformationVariants, addMirror: true
        ), variants.count > 1 else { return [ligand] }
        return variants.map { variant in
            var atoms = ligand.atoms
            for i in atoms.indices { atoms[i].position = variant.positions[i] }
            return Molecule(name: ligand.name, atoms: atoms, bonds: ligand.bonds,
                            title: ligand.title, smiles: ligand.smiles)
        }
    }

    /// Docks the active ligand into the selected pocket. Returns nil when protein, ligand or
    /// pocket is missing; otherwise a task that completes after post-processing and yields
    /// whether any pose was produced.
    @discardableResult
    func runDocking() -> Task<Bool, Never>? {
        guard let pocket = docking.selectedPocket,
              let prot = molecules.protein,
              let lig = molecules.ligand
        else {
            log.error("Need protein, ligand, and pocket to dock", category: .dock)
            return nil
        }

        let dockingLigand = ensureLigandHasSMILES(lig)
        docking.originalDockingLigand = Molecule(
            name: dockingLigand.name,
            atoms: dockingLigand.atoms,
            bonds: dockingLigand.bonds,
            title: dockingLigand.title,
            smiles: dockingLigand.smiles
        )

        docking.isDocking = true
        docking.dockingGeneration = 0
        // Each search method has a different "step" budget; the GA-derived numGenerations
        // is wrong for Diffusion (which iterates denoising steps) and Fragment (which
        // iterates over fragment growth, harder to predict — fall back to fragment count).
        switch docking.dockingConfig.searchMethod {
        case .diffusionGuided:
            docking.dockingTotalGenerations = max(docking.dockingConfig.diffusion.numDenoisingSteps, 1)
        case .fragmentBased:
            docking.dockingTotalGenerations = max(docking.dockingConfig.fragment.beamWidth, 1)
        default:
            docking.dockingTotalGenerations = docking.dockingConfig.numGenerations
        }
        docking.dockingResults = []
        docking.currentInteractions = []
        docking.dockingBestEnergy = .infinity
        docking.dockingBestPKi = nil
        docking.dockingStartTime = Date()
        docking.dockingDuration = 0
        docking.dockingPosesEvaluated = 0
        docking.selectedPoseIndices = []
        let cfg = docking.dockingConfig
        log.info("Starting docking: pop=\(cfg.populationSize), gen=\(cfg.numGenerations) (\(cfg.numRuns)×\(cfg.generationsPerRun)), grid=\(String(format: "%.3f", cfg.gridSpacing)) Å", category: .dock)
        log.info("  Ligand: \(dockingLigand.name) (\(dockingLigand.atoms.filter { $0.element != .H }.count) heavy atoms, \(dockingLigand.atoms.count) total, \(dockingLigand.bondCount) bonds)", category: .dock)
        log.info("  Pocket: center=(\(String(format: "%.1f, %.1f, %.1f", pocket.center.x, pocket.center.y, pocket.center.z))), size=(\(String(format: "%.1f, %.1f, %.1f", pocket.size.x, pocket.size.y, pocket.size.z))), volume=\(String(format: "%.0f", pocket.volume)) ų", category: .dock)
        log.info("  Scoring: \(docking.scoringMethod.rawValue), charges: \(docking.chargeMethod.rawValue)", category: .dock)
        log.info("  Config: localSearchFreq=\(cfg.localSearchFrequency), localSearchSteps=\(cfg.localSearchSteps), explorationRatio=\(String(format: "%.2f", cfg.explorationPhaseRatio))", category: .dock)
        log.info("  Config: torsionFlex=\(cfg.enableFlexibility), numRuns=\(cfg.numRuns)", category: .dock)
        log.info("  Config: torsionSearch=\(cfg.torsionSearchPreset.rawValue), exact=\(String(format: "%.0f", cfg.torsionExactFraction * 100))%, local=\(String(format: "%.0f", cfg.torsionLocalFraction * 100))%, reset=\(String(format: "%.0f", cfg.torsionRandomResetProbability * 100))%", category: .dock)

        return Task {
            if docking.dockingEngine == nil, let device = computeDevice {
                docking.dockingEngine = DockingEngine(device: device)
            }
            guard let engine = docking.dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                docking.isDocking = false
                docking.dockingBestEnergy = .infinity
                docking.dockingResults = []
                workspace.statusMessage = "Docking engine unavailable"
                return false
            }

            let dockingPH = molecules.protonationPH
            let chargeMethod = docking.chargeMethod
            let (scoringProtein, _) = await preparedProteinForDocking(protein: prot, pH: dockingPH, chargeMethod: chargeMethod)

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
                if docking.flexDockingEngine == nil, let device = computeDevice {
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
                let heavyAtoms = dockingLigand.atoms.filter { $0.element != .H }.count
                let torsions = RDKitBridge.buildTorsionTree(smiles: dockingLigand.smiles ?? "")?.count ?? 0
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

            attachReceptor(scoringProtein, to: engine)

            let origLig = self.docking.originalDockingLigand ?? dockingLigand

            log.info("  Original ligand: \(origLig.name), \(origLig.atoms.count) atoms, \(origLig.bonds.count) bonds", category: .dock)

            engine.onPoseUpdate = { [weak self] result, interactions in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    // Each run restarts its population, so keep the best across runs.
                    self.docking.dockingBestEnergy = min(self.docking.dockingBestEnergy, result.energy)
                    if self.docking.scoringMethod.isAffinityScore {
                        let pKd = result.stericEnergy  // pKd stored by DruseAF shader
                        self.docking.dockingBestPKi = max(self.docking.dockingBestPKi ?? pKd, pKd)
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

            // Rings stay rigid during the search, so each distinct ring conformation of the
            // ligand is docked as its own job (sequentially: the engine runs one job at a time).
            let variants = ringConformationVariants(of: dockingLigand)
            let generationsPerVariant = docking.dockingTotalGenerations
            docking.dockingTotalGenerations = generationsPerVariant * variants.count
            var allResults: [DockingResult] = []
            var variantOfResult: [Int: Int] = [:]
            var evaluatedPoses = 0
            for (index, variant) in variants.enumerated() {
                guard index == 0 || docking.isDocking else { break }
                let generationOffset = index * generationsPerVariant
                engine.onGenerationComplete = { [weak self] gen, _ in
                    Task { @MainActor [weak self] in
                        self?.docking.dockingGeneration = generationOffset + gen
                    }
                }
                let variantResults = await engine.runDocking(
                    ligand: variant, pocket: pocket,
                    config: docking.dockingConfig,
                    scoringMethod: docking.scoringMethod)
                evaluatedPoses += engine.lastEvaluatedPoseCount
                for var result in variantResults {
                    result.id = allResults.count
                    variantOfResult[result.id] = index
                    allResults.append(result)
                }
            }
            if variants.count > 1 {
                allResults = await engine.clusterPoses(
                    DockingEngine.withinOutputEnergyWindow(allResults, scoringMethod: docking.scoringMethod))
                log.info("\(variants.count) ring conformations docked", category: .dock)
            }

            let results = allResults.sorted { $0.energy < $1.energy }
            if let best = results.first, let index = variantOfResult[best.id] {
                docking.originalDockingLigand = variants[index]
            }

            log.info("GA complete: \(results.count) poses returned", category: .dock)
            if let best = results.first {
                let method = docking.scoringMethod
                let scoreStr = method.isAffinityScore
                    ? "pKi=\(String(format: "%.3f", best.displayScore(method: method)))"
                    : "E=\(String(format: "%.3f", best.energy)) kcal/mol"
                log.info("  Best pose: \(scoreStr), cluster=\(best.clusterID), gen=\(best.generation)",
                         category: .dock)
            }

            guard !results.isEmpty else {
                docking.dockingResults = []
                docking.isDocking = false
                docking.dockingBestEnergy = .infinity
                docking.dockingBestPKi = nil
                docking.dockingDuration = Date().timeIntervalSince(docking.dockingStartTime ?? Date())
                docking.currentInteractions = []
                docking.selectedPoseIndices = []
                renderer?.updateInteractionLines([])
                renderer?.clearGhostPose()
                if let pocket = docking.selectedPocket {
                    showGridBoxForPocket(pocket)
                }
                workspace.statusMessage = "Docking produced no valid poses"
                log.error("Docking produced no valid poses; skipping post-processing", category: .dock)
                return false
            }

            // Signal that GA search is done — post-processing begins
            workspace.statusMessage = "Post-processing docking results..."

            var rankedResults = results

            let openMMRefiner = OpenMMPocketRefiner.shared
            if openMMRefiner.isAvailable {
                workspace.statusMessage = "OpenMM refinement..."
                rankedResults = await applyOpenMMRefinement(
                    to: rankedResults, top: Self.defaultOpenMMRefinedPoses,
                    scoringProtein: scoringProtein, displayedProtein: prot,
                    originalLigand: origLig, pocket: pocket
                )
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

            // PoseBusters-style validity checks (bond geometry, clashes, connectivity).
            let validationLigandHeavy = origLig.atoms.filter { $0.element != .H }
            let validationLigandBonds = buildHeavyBonds(from: origLig)
            let validationProteinHeavy = scoringProtein.atoms.filter { $0.element != .H }
            rankedResults = PoseValidator.validateBatch(
                results: rankedResults,
                ligandHeavyAtoms: validationLigandHeavy,
                ligandHeavyBonds: validationLigandBonds,
                proteinHeavyAtoms: validationProteinHeavy
            )
            let invalidPoses = rankedResults.filter { ($0.validity?.passed == false) }.count
            if invalidPoses > 0 {
                log.warn("\(invalidPoses)/\(rankedResults.count) poses failed PoseBusters checks", category: .dock)
            }

            docking.dockingResults = rankedResults
            docking.isDocking = false
            docking.dockingDuration = Date().timeIntervalSince(docking.dockingStartTime ?? Date())
            docking.dockingPosesEvaluated = evaluatedPoses

            if let best = docking.dockingResults.first {
                applyDockingPose(best, originalLigand: origLig)

                if molecules.protein != nil {
                    let heavyAtoms = origLig.atoms.filter { $0.element != .H }
                    let heavyBonds = buildHeavyBonds(from: origLig)
                    docking.currentInteractions = InteractionDetector.detect(
                        ligandAtoms: heavyAtoms,
                        ligandPositions: best.transformedAtomPositions,
                        proteinAtoms: scoringProtein.atoms.filter { $0.element != .H },
                        ligandBonds: heavyBonds,
                        scoringMethod: docking.scoringMethod
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
                    log.success(String(format: "Docking complete (Drusina): best %.1f %@ (correction: %.2f), %d/%d poses kept, %d clusters (%.1fs)",
                                       bestDisplay, unit, drCorr, rankedResults.count, docking.dockingPosesEvaluated, clusterCount, elapsed), category: .dock)
                } else if method.isAffinityScore {
                    log.success(String(format: "Docking complete (DruseAF): best pKi %.1f (conf: %.0f%%), %d/%d poses kept, %d clusters (%.1fs)",
                                       bestDisplay, best.afConfidence * 100, rankedResults.count, docking.dockingPosesEvaluated, clusterCount, elapsed), category: .dock)
                } else {
                    log.success(String(format: "Docking complete: best %.1f %@, %d/%d poses kept, %d clusters (%.1fs)",
                                       bestDisplay, unit, rankedResults.count, docking.dockingPosesEvaluated, clusterCount, elapsed), category: .dock)
                }
                workspace.statusMessage = String(format: "Best: %.1f %@", bestDisplay, unit)
                if docking.currentInteractions.count > 0 {
                    let hbonds = docking.currentInteractions.filter { $0.type == .hbond }.count
                    let hydro = docking.currentInteractions.filter { $0.type == .hydrophobic }.count
                    let pipi = docking.currentInteractions.filter { $0.type == .piStack }.count
                    log.info("  Best pose interactions: \(hbonds) H-bonds, \(hydro) hydrophobic, \(pipi) π-stacking", category: .dock)
                }
            }
            return true
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

    // MARK: - Receptor hand-off

    /// Poses refined with OpenMM after every docking run.
    static let defaultOpenMMRefinedPoses = 3

    /// Hands the prepared receptor to the engine without building grid maps:
    /// `DockingEngine.runDocking` builds them once the ligand's Vina atom types are known, so an
    /// earlier type-less build only produced empty affinity maps (and the "No valid Vina atom
    /// types" warning) that were thrown away.
    func attachReceptor(_ protein: Molecule, to engine: DockingEngine) {
        engine.proteinStructure = protein
        engine.proteinAtoms = protein.atoms.filter { $0.element != .H }
    }

    /// OpenMM-refines the `top` best poses and shows the refined receptor of the best one.
    /// Shared by the post-docking pipeline and `refineDockingResultsWithOpenMM(top:)`.
    private func applyOpenMMRefinement(
        to results: [DockingResult], top: Int,
        scoringProtein: Molecule, displayedProtein: Molecule,
        originalLigand: Molecule, pocket: BindingPocket
    ) async -> [DockingResult] {
        let refinement = await refineTopDockingResultsWithOpenMM(
            results: results, top: top, protein: scoringProtein,
            originalLigand: originalLigand, pocket: pocket
        )
        if let refinedPositions = refinement.bestProteinHeavyPositions,
           let updatedProteinAtoms = applyingRefinedHeavyAtomPositions(refinedPositions, to: displayedProtein.atoms) {
            let refinedProtein = Molecule(name: displayedProtein.name, atoms: updatedProteinAtoms,
                                          bonds: displayedProtein.bonds, title: displayedProtein.title)
            refinedProtein.secondaryStructureAssignments = displayedProtein.secondaryStructureAssignments
            molecules.protein = refinedProtein
            pushToRenderer()
        }
        return refinement.results
    }

    // MARK: - Post-docking operations on the current results

    /// Why a post-docking operation on the current results could not run.
    enum DockingPostProcessingError: LocalizedError {
        case noResults
        case busy
        case missingInputs
        case openMMUnavailable(String)
        case noSMILES
        case alreadyRefined
        case unsupportedRescoring(ScoringMethod)
        case scorerUnavailable(ScoringMethod)
        case staleEngineState

        var errorDescription: String? {
            switch self {
            case .noResults: "No docking results"
            case .busy: "A docking run is in progress"
            case .missingInputs: "The protein, the docked ligand and the docking pocket are all needed"
            case .openMMUnavailable(let reason): "OpenMM is unavailable: \(reason)"
            case .noSMILES: "The docked ligand has no SMILES, which the MMFF94 strain model needs (dock it from the Ligand Database or 'ligand smiles')"
            case .alreadyRefined: "These poses are already GFN2-refined — dock again to refine with other settings"
            case .unsupportedRescoring(let method): "\(method.rawValue) is not a rescoring model (use DruseAF or PIGNet2)"
            case .scorerUnavailable(let method): "\(method.rawValue) scoring kernels or weights are unavailable in this build"
            case .staleEngineState: "The docking engine has since been used for another ligand — dock this ligand again before rescoring"
            }
        }
    }

    /// Inputs shared by the post-docking operations; throws when results or inputs are missing.
    private func postProcessingInputs() throws -> (ligand: Molecule, protein: Molecule, pocket: BindingPocket) {
        guard !docking.dockingResults.isEmpty else { throw DockingPostProcessingError.noResults }
        guard !docking.isDocking, !docking.isBatchDocking else { throw DockingPostProcessingError.busy }
        guard let ligand = docking.originalDockingLigand ?? molecules.ligand,
              let protein = molecules.protein,
              let pocket = docking.selectedPocket else { throw DockingPostProcessingError.missingInputs }
        return (ligand, protein, pocket)
    }

    /// Stores post-processed results and shows the new best pose with its interactions.
    private func publishPostProcessedResults(_ results: [DockingResult]) {
        docking.dockingResults = results
        docking.selectedPoseIndices = []
        showDockingPose(at: 0)
    }

    /// OpenMM pocket minimization of the `top` best current poses (the `refine openmm` command).
    /// Returns how many poses received an OpenMM interaction energy.
    @discardableResult
    func refineDockingResultsWithOpenMM(top: Int) async throws -> Int {
        let inputs = try postProcessingInputs()
        let refiner = OpenMMPocketRefiner.shared
        guard refiner.isAvailable else { throw DockingPostProcessingError.openMMUnavailable(refiner.availabilitySummary) }
        let (scoringProtein, _) = await preparedProteinForDocking(
            protein: inputs.protein, pH: molecules.protonationPH, chargeMethod: docking.chargeMethod
        )
        workspace.statusMessage = "OpenMM refinement..."
        let refined = await applyOpenMMRefinement(
            to: docking.dockingResults, top: top,
            scoringProtein: scoringProtein, displayedProtein: inputs.protein,
            originalLigand: inputs.ligand, pocket: inputs.pocket
        )
        publishPostProcessedResults(refined)
        let count = refined.prefix(top).filter { $0.refinementEnergy != nil }.count
        log.success("OpenMM refined \(count)/\(min(top, refined.count)) poses", category: .dock)
        return count
    }

    /// GFN2-xTB optimization of the current poses with `config` (the `refine gfn2` command),
    /// followed by PoseBusters re-validation of the moved geometries.
    /// Returns (optimized, converged) pose counts.
    @discardableResult
    func refineDockingResultsWithGFN2(config: GFN2RefinementConfig) async throws -> (optimized: Int, converged: Int) {
        let inputs = try postProcessingInputs()
        let top = min(config.topPosesToRefine, docking.dockingResults.count)
        guard !docking.dockingResults.prefix(top).contains(where: { $0.gfn2Energy != nil }) else {
            throw DockingPostProcessingError.alreadyRefined
        }
        workspace.statusMessage = "GFN2-xTB refinement..."
        log.info("GFN2-xTB refinement: top \(top) poses, \(config.solvation.rawValue), opt=\(config.optLevel.rawValue)", category: .dock)
        let refined = await refineWithGFN2(results: docking.dockingResults, originalLigand: inputs.ligand, config: config)
        let (scoringProtein, _) = await preparedProteinForDocking(
            protein: inputs.protein, pH: molecules.protonationPH, chargeMethod: docking.chargeMethod
        )
        let validated = PoseValidator.validateBatch(
            results: refined,
            ligandHeavyAtoms: inputs.ligand.atoms.filter { $0.element != .H },
            ligandHeavyBonds: buildHeavyBonds(from: inputs.ligand),
            proteinHeavyAtoms: scoringProtein.atoms.filter { $0.element != .H }
        )
        publishPostProcessedResults(validated)
        let optimized = validated.filter { $0.gfn2Energy != nil }.count
        let converged = validated.filter { $0.gfn2Converged == true }.count
        log.success("GFN2-xTB refinement complete: \(converged)/\(optimized) converged", category: .dock)
        return (optimized, converged)
    }

    /// MMFF94 strain of the `top` best current poses with the given penalty (the `strain` command).
    /// Returns (poses with a strain energy, poses above the threshold).
    @discardableResult
    func applyStrainPenaltiesToDockingResults(top: Int, threshold: Float, weight: Float) async throws -> (computed: Int, penalized: Int) {
        let inputs = try postProcessingInputs()
        let ligand = ensureLigandHasSMILES(inputs.ligand)
        guard let smiles = nonEmptySMILES(ligand.smiles) ?? nonEmptySMILES(ligand.title) else {
            throw DockingPostProcessingError.noSMILES
        }
        var config = docking.dockingConfig
        config.strainPenaltyThreshold = threshold
        config.strainPenaltyWeight = weight
        let updated = await computeStrainPenalties(results: docking.dockingResults, smiles: smiles, config: config, top: top)
        publishPostProcessedResults(updated)
        let strained = updated.compactMap(\.strainEnergy)
        return (strained.count, strained.filter { $0 > threshold }.count)
    }

    /// Rescores the current poses with DruseAF or PIGNet2 without moving them (the `rescore`
    /// command) and makes that model the active scoring method, so results rank and display
    /// in its units. Returns the number of rescored poses.
    @discardableResult
    func rescoreDockingResults(with method: ScoringMethod) async throws -> Int {
        let inputs = try postProcessingInputs()
        guard method == .druseAffinity || method == .pignet2 else {
            throw DockingPostProcessingError.unsupportedRescoring(method)
        }
        guard let engine = docking.dockingEngine else { throw DockingPostProcessingError.staleEngineState }
        let results = docking.dockingResults
        let rescored = try engine.rescorePoses(results, ligand: inputs.ligand, pocket: inputs.pocket, method: method)
        docking.scoringMethod = method
        let ranked = rescored.sorted { $0.energy < $1.energy }
        publishPostProcessedResults(ranked)
        log.success("Rescored \(ranked.count) poses with \(method.rawValue)", category: .dock)
        return ranked.count
    }

    // MARK: - Pharmacophore constraints

    /// Adds a docking constraint, resolving receptor targets to 3D positions now so later
    /// preparation of the receptor (added hydrogens, renumbering) cannot move them.
    /// Shared by the constraint sheet and the `constraint add` command.
    func addPharmacophoreConstraint(_ constraint: PharmacophoreConstraintDef) {
        var constraint = constraint
        if constraint.sourceType == .receptor, constraint.targetPositions.isEmpty, let protein = molecules.protein {
            constraint.targetPositions = ConstraintAtomResolver.resolveTargetPositions(
                constraint: constraint, atoms: protein.atoms, residues: protein.residues
            )
        }
        constraint.groupID = docking.pharmacophoreConstraints.count
        docking.pharmacophoreConstraints.append(constraint)
        pushToRenderer()
    }

    /// Remove all displayed poses and interaction lines from the 3D viewport.
    /// The docking results data is preserved for re-viewing later.
    func clearPosesFromView() {
        // Preserve the ligand template so "View" on a pose card can rebuild it.
        // After a `.druse` project load, `originalDockingLigand` is nil — without
        // this snapshot, clearing the view would also strand any saved poses.
        if docking.originalDockingLigand == nil, let lig = molecules.ligand {
            docking.originalDockingLigand = Molecule(
                name: lig.name,
                atoms: lig.atoms,
                bonds: lig.bonds,
                title: lig.title,
                smiles: lig.smiles
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
                ligandBonds: heavyBonds,
                scoringMethod: docking.scoringMethod
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
                    ligandBonds: heavyBonds,
                    scoringMethod: docking.scoringMethod
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

    /// OpenMM pocket minimization of the `top` best poses. The interaction energy is stored as
    /// `refinementEnergy` (an annotation: ranking is unchanged); the refined receptor heavy-atom
    /// positions of the best pose are returned so the caller can show them.
    func refineTopDockingResultsWithOpenMM(
        results: [DockingResult],
        top: Int,
        protein: Molecule,
        originalLigand: Molecule,
        pocket: BindingPocket
    ) async -> (results: [DockingResult], bestProteinHeavyPositions: [SIMD3<Float>]?) {
        guard !results.isEmpty else { return (results, nil) }

        var refinedResults = results
        var refinedProteinByResultID: [Int: [SIMD3<Float>]] = [:]
        let refiner = OpenMMPocketRefiner.shared
        let topN = min(top, refinedResults.count)

        log.info("Refining top \(topN) docking poses with OpenMM pocket minimization...", category: .dock)
        // Charges are folded while the hydrogens still sit at their input coordinates; poses carry heavy atoms only.
        let heavyAtoms = OpenMMPocketRefiner.unitedAtomHeavyAtoms(originalLigand.atoms)
        for index in 0..<topN {
            let result = refinedResults[index]
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
                // A diverged minimization returns NaN energies/coordinates; keeping them would
                // corrupt the displayed receptor and crash its next preparation.
                guard refined.interactionEnergyKcal.isFinite,
                      refined.refinedHeavyAtomPositions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
                    log.warn("OpenMM refinement of pose #\(index + 1) diverged (non-finite result) — pose and receptor left unchanged", category: .dock)
                    continue
                }
                refinedResults[index].refinementEnergy = refined.interactionEnergyKcal
                refinedProteinByResultID[result.id] = refined.refinedHeavyAtomPositions
            }
        }

        // Ranking stays on the docking score (`energy`, the active scoring method's primary
        // score). The OpenMM interaction energy is a force-field quantity on a different
        // scale than Vina-style scores, so it is kept as an annotation only.
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
        molecules.ligand = Molecule(
            name: originalLigand.name,
            atoms: newAtoms,
            bonds: newBonds,
            title: originalLigand.title,
            smiles: originalLigand.smiles
        )
        pushToRenderer()

        if !docking.currentInteractions.isEmpty {
            renderer?.updateInteractionLines(docking.currentInteractions)
        }
    }

    func applyLiveDockingPose(_ result: DockingResult, originalLigand: Molecule) {
        guard let (newAtoms, newBonds) = buildTransformedLigand(result: result, originalLigand: originalLigand) else { return }

        molecules.ligand = Molecule(
            name: originalLigand.name,
            atoms: newAtoms,
            bonds: newBonds,
            title: originalLigand.title,
            smiles: originalLigand.smiles
        )
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
                String(format: "%.2f", r.stericEnergy),
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

        // Process top poses in parallel, at most `maxConcurrentTasks` at a time (Settings / --cores).
        let maxConcurrency = max(AppSettings.shared.maxConcurrentTasks, 1)
        let refinements: [(Int, GFN2RefinementResult?)] = await withTaskGroup(
            of: (Int, GFN2RefinementResult?).self
        ) { group in
            var collected = [(Int, GFN2RefinementResult?)]()
            var running = 0
            for i in 0..<topN {
                let positions = results[i].transformedAtomPositions
                guard positions.count == heavyAtoms.count else { continue }
                if running >= maxConcurrency, let finished = await group.next() {
                    collected.append(finished)
                    running -= 1
                }
                running += 1

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

    // MARK: - Strain Penalty

    /// Computes the MMFF94 strain (docked − relaxed free ligand) of the `top` best poses and adds
    /// `weight × (strain − threshold)` to the score of poses above the threshold. Poses that
    /// already carry a strain energy keep it and are not penalized twice.
    func computeStrainPenalties(
        results: [DockingResult], smiles: String, config: DockingConfig, top: Int = 50
    ) async -> [DockingResult] {
        workspace.statusMessage = "Computing ligand strain..."

        let topN = min(results.count, top) // CPU-bound: only the best poses
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
        let positionsSlice = results.prefix(topN).map { $0.strainEnergy == nil ? $0.transformedAtomPositions : [] }

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

    /// Returns nil without a protein; otherwise a task that completes once pockets are set.
    @discardableResult
    func detectPocketsML() -> Task<Void, Never>? {
        guard let prot = molecules.protein else { return nil }
        guard pocketDetectorML.isAvailable else {
            log.warn("PocketDetector model not available, falling back to auto detection", category: .dock)
            return detectPockets()
        }
        log.info("Detecting pockets (ML)...", category: .dock)
        workspace.statusMessage = "ML pocket detection..."

        return Task {
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

// MARK: - Post-hoc neural rescoring

extension DockingEngine {

    /// Scores finished poses with DruseAF or PIGNet2 without moving them and returns them with
    /// the new score terms (same order). The engine's ligand buffers must still describe
    /// `ligand`, which holds right after that ligand's docking run; when another ligand has
    /// been docked since, `staleEngineState` is thrown instead of scoring the wrong molecule.
    func rescorePoses(
        _ results: [DockingResult], ligand: Molecule, pocket: BindingPocket, method: ScoringMethod
    ) throws -> [DockingResult] {
        typealias Failure = AppViewModel.DockingPostProcessingError
        guard !isRunning else { throw Failure.busy }
        guard let gaParamsBuffer, let ligandAtomBuffer, !results.isEmpty else { throw Failure.staleEngineState }

        let prepared = prepareLigandGeometry(ligand)
        let atomCount = prepared.gpuAtoms.count
        let currentGA = gaParamsBuffer.contents().bindMemory(to: GAParams.self, capacity: 1).pointee
        guard Int(currentGA.numLigandAtoms) == atomCount,
              ligandAtomBuffer.length >= atomCount * MemoryLayout<DockLigandAtom>.stride else {
            throw Failure.staleEngineState
        }
        let loaded = ligandAtomBuffer.contents().bindMemory(to: DockLigandAtom.self, capacity: atomCount)
        for i in 0..<atomCount where simd_distance(loaded[i].position, prepared.gpuAtoms[i].position) > 1e-3 {
            throw Failure.staleEngineState
        }

        var poses = results.map(makeDockPose)
        var rescoreGA = currentGA
        rescoreGA.populationSize = UInt32(poses.count)
        guard let poseBuffer = device.makeBuffer(bytes: &poses, length: poses.count * MemoryLayout<DockPose>.stride,
                                                 options: .storageModeShared),
              let rescoreGABuffer = device.makeBuffer(bytes: &rescoreGA, length: MemoryLayout<GAParams>.stride,
                                                      options: .storageModeShared) else {
            throw Failure.scorerUnavailable(method)
        }

        isRunning = true
        defer { isRunning = false }
        let perPose = MTLSize(width: poses.count, height: 1, depth: 1)
        let single = MTLSize(width: 1, height: 1, depth: 1)
        switch method {
        case .druseAffinity:
            guard druseAFWeights != nil, druseAFEncodePipeline != nil,
                  prepareDruseAFBuffers(ligandAtoms: prepared.heavyAtoms, ligandBonds: prepared.heavyBonds,
                                        gpuLigAtoms: prepared.gpuAtoms, pocket: pocket, popSize: poses.count) else {
                throw Failure.scorerUnavailable(method)
            }
            scoreDruseAF(buffer: poseBuffer, gaParamsBuffer: rescoreGABuffer, tg: perPose, tgs: single)
        case .pignet2:
            guard pignet2Weights != nil, pignet2EncodePipeline != nil, pignet2ScorePipeline != nil,
                  preparePIGNet2Buffers(ligandAtoms: prepared.heavyAtoms, ligandBonds: prepared.heavyBonds,
                                        gpuLigAtoms: prepared.gpuAtoms, pocket: pocket, popSize: poses.count) else {
                throw Failure.scorerUnavailable(method)
            }
            scorePIGNet2(buffer: poseBuffer, gaParamsBuffer: rescoreGABuffer, tg: perPose, tgs: single)
        case .vina, .drusina:
            throw Failure.unsupportedRescoring(method)
        }
        // A GPU command-buffer error clears isRunning.
        guard isRunning else { throw Failure.scorerUnavailable(method) }

        let scored = poseBuffer.contents().bindMemory(to: DockPose.self, capacity: poses.count)
        return results.enumerated().map { i, result in
            var updated = result
            updated.energy = scored[i].energy
            updated.stericEnergy = scored[i].stericEnergy
            updated.hydrophobicEnergy = scored[i].hydrophobicEnergy
            updated.hbondEnergy = scored[i].hbondEnergy
            updated.torsionPenalty = scored[i].torsionPenalty
            updated.drusinaCorrection = 0
            return updated
        }
    }
}

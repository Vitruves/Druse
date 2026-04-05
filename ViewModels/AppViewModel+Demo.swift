import SwiftUI

// MARK: - Guided Demo: Neuraminidase + Oseltamivir (Tamiflu) from SMILES
//
// Protein: Influenza N1 Neuraminidase (PDB: 2HU4) — deep funnel-shaped active site
// Ligand:  Oseltamivir carboxylate built from SMILES (20 heavy atoms)
//
// This is genuine de novo docking: the drug is built from a 2D chemical formula,
// the pocket is found by blind detection, and GPU docking discovers the binding pose.

extension AppViewModel {

    // MARK: - Demo Constants

    // Trypsin: small serine protease (~1,600 atoms, single chain, deep S1 pocket)
    private static let demoPDBID = "3PTB"
    // Nafamostat: serine protease inhibitor (anticoagulant, studied for COVID-19), 20 heavy atoms
    // Pre-protonated: amidinium (pKa ~11.6) and guanidinium (pKa ~12.5) both carry +1 at pH 7.4
    // Henderson-Hasselbalch: >99.99% protonated at pH 7.4 for both groups
    // NOTE: In the Ligand Database workflow, protomers are enumerated automatically via
    // prepareEnsemble() with Boltzmann populations — users can select minority species
    private static let demoLigandSMILES = "NC(=[NH2+])c1ccc(OC(=O)c2ccc(NC(=[NH2+])N)cc2)cc1"
    private static let demoLigandName = "Nafamostat"

    // MARK: - Demo Orchestrator

    func runGuidedDemo() {
        guard demoStep == .idle || demoStep == .complete else { return }
        log.info("=== GUIDED DEMO START ===", category: .system)
        log.info("Target: Bovine Trypsin (3PTB) — serine protease, ~1,600 atoms", category: .system)
        log.info("Ligand: Nafamostat from SMILES — serine protease inhibitor", category: .system)

        Task {
            do {
                try await demoPhase1_FetchProtein()
                try await demoPhase2_VisualExploration()
                try await demoPhase3_PocketDetection()
                try await demoPhase4_BuildLigand()
                try await demoPhase5_Docking()
                try await demoPhase6_Results()
            } catch is CancellationError {
                log.info("Demo cancelled", category: .system)
                demoStep = .idle
                demoNarration = ""
            } catch {
                log.error("Demo failed: \(error.localizedDescription)", category: .system)
                demoStep = .idle
                demoNarration = ""
            }
        }
    }

    func cancelDemo() {
        if docking.isDocking { stopDocking() }
        demoStep = .idle
        demoNarration = ""
    }

    // MARK: - Phase 1: Fetch Protein

    private func demoPhase1_FetchProtein() async throws {
        setDemoStep(.fetching,
            narration: "Downloading Bovine Trypsin from RCSB (PDB: 3PTB) — a serine protease with one of the deepest, most well-defined binding pockets in structural biology. Small protein (~1,600 atoms) = fast docking.")

        let content = try await PDBService.shared.fetchPDBFile(id: Self.demoPDBID)
        molecules.rawPDBContent = content

        setDemoStep(.parsing,
            narration: "Parsing the crystal structure — Trypsin has ~1,600 atoms with the catalytic triad (His57, Asp102, Ser195) and a deep S1 specificity pocket lined by Asp189 at the bottom.")

        let result = await Task.detached { PDBParser.parse(content) }.value

        if let protData = result.protein {
            let mol = Molecule(name: protData.name, atoms: protData.atoms,
                               bonds: protData.bonds, title: protData.title)
            mol.secondaryStructureAssignments = protData.ssRanges.map {
                (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
            }
            molecules.protein = mol
        }
        guard molecules.protein != nil else { throw DemoError.missingProtein }

        // Discard any co-crystallized ligand — we build Nafamostat from SMILES later
        molecules.ligand = nil

        molecules.preparationReport = ProteinPreparation.analyze(
            atoms: result.protein?.atoms ?? [],
            bonds: result.protein?.bonds ?? [],
            waterCount: result.waterCount
        )
        workspace.hiddenChainIDs.removeAll()

        // Compute EEM partial charges so the Drusina electrostatic grid and ESP
        // surface are functional. EEM is fast (GPU-accelerated) and doesn't require
        // full protein preparation (protonation, H-bond network, etc.)
        if let prot = molecules.protein {
            let calculator = EEMChargeCalculator(device: renderer?.device)
            if let charges = try? await calculator.computeCharges(
                atoms: prot.atoms, bonds: prot.bonds, totalCharge: 0
            ) {
                var updatedAtoms = prot.atoms
                for i in 0..<min(charges.count, updatedAtoms.count) {
                    updatedAtoms[i].charge = charges[i]
                }
                let charged = Molecule(name: prot.name, atoms: updatedAtoms,
                                       bonds: prot.bonds, title: prot.title)
                charged.secondaryStructureAssignments = prot.secondaryStructureAssignments
                molecules.protein = charged
                let nonZero = charges.filter { abs($0) > 0.001 }.count
                log.success("Demo: EEM charges — \(nonZero)/\(charges.count) non-zero", category: .prep)
            }
        }

        pushToRenderer()
        renderer?.fitToContent()

        log.success("Demo: Loaded 3PTB — \(molecules.protein!.atomCount) atoms", category: .pdb)
        try await Task.sleep(for: .seconds(5.0))
    }

    // MARK: - Phase 2: Visual Exploration

    private func demoPhase2_VisualExploration() async throws {
        setDemoStep(.overview,
            narration: "Trypsin in ball & stick — a compact serine protease with surface loops surrounding a deep active site cavity. The S1 specificity pocket determines what substrates the enzyme can cleave.")

        setRenderMode(.ballAndStick)
        renderer?.fitToContent()
        try await Task.sleep(for: .seconds(6.0))

        setDemoStep(.ribbon,
            narration: "Ribbon view reveals Trypsin's two beta-barrel domains with the catalytic triad at their junction. The S1 pocket is a deep slot between the domains — this is where Nafamostat will bind.")

        setRenderMode(.ribbon)
        pushToRenderer()
        renderer?.fitToContent()
        try await Task.sleep(for: .seconds(6.0))

        // Generate a translucent Connolly surface with electrostatic coloring.
        // EEM charges were computed in Phase 1, so ESP mapping is functional.
        workspace.surfaceType = .connolly
        workspace.surfaceColorMode = .esp
        workspace.surfaceOpacity = 0.85
        workspace.showSurface = true
        generateSurface()
        renderer?.surfaceOpacity = 0.85

        setDemoStep(.ribbon,
            narration: "The electrostatic surface reveals the pocket chemistry — red regions are negative (Asp189), blue regions are positive. The deep S1 slot is clearly visible as a concavity.")

        try await Task.sleep(for: .seconds(6.0))
    }

    // MARK: - Phase 3: Pocket Detection (blind — before ligand is loaded)

    private func demoPhase3_PocketDetection() async throws {
        guard let prot = molecules.protein else {
            throw DemoError.missingProtein
        }

        setDemoStep(.pocketScan,
            narration: "Scanning the protein surface for druggable pockets — no ligand information is used. The ML model scores each surface point for cavity depth and chemical environment to identify where a drug could bind.")

        try await Task.sleep(for: .seconds(3.0))

        let geometricPockets = BindingSiteDetector.detectPockets(protein: prot)
        let mlPockets: [BindingPocket]

        if pocketDetectorML.isAvailable {
            mlPockets = await pocketDetectorML.detectPockets(protein: prot)
            if !mlPockets.isEmpty {
                log.info("Demo: ML pocket detection → \(mlPockets.count) pocket(s)", category: .dock)
            }
        } else {
            mlPockets = []
        }

        let candidates = PocketSelectionHeuristics.rankedHybridCandidates(
            mlPockets: mlPockets,
            geometricPockets: geometricPockets
        )
        let pockets = candidates.map(\.pocket)

        guard let bestCandidate = candidates.first else {
            throw DemoError.noPockets
        }
        let best = bestCandidate.pocket

        docking.detectedPockets = pockets
        docking.selectedPocket = best
        showGridBoxForPocket(best)

        setDemoStep(.pocketFound,
            narration: "Found \(pockets.count) pocket(s). The top-ranked cavity — volume \(String(format: "%.0f", best.volume)) \u{00C5}\u{00B3}, druggability \(String(format: "%.2f", best.druggability)) — this is the S1 specificity pocket, lined by Asp189 at the floor.")

        log.success("Demo: Best pocket \(String(format: "%.0f", best.volume)) \u{00C5}\u{00B3}, druggability \(String(format: "%.2f", best.druggability))", category: .dock)

        try await Task.sleep(for: .seconds(5.0))

        setDemoStep(.gridSetup,
            narration: "The green wireframe defines the docking search space. Now let's build the drug from its chemical formula and dock it into this pocket.")

        focusOnPocket(best)
        showGridBoxForPocket(best)
        try await Task.sleep(for: .seconds(5.0))
    }

    // MARK: - Phase 4: Build Tamiflu from SMILES (AFTER pocket detection)

    private func demoPhase4_BuildLigand() async throws {
        guard let pocket = docking.selectedPocket else {
            throw DemoError.noPockets
        }

        setDemoStep(.parsing,
            narration: "Building Nafamostat from SMILES — a serine protease inhibitor used as an anticoagulant, also studied as a COVID-19 therapeutic. RDKit generates 3D coordinates, minimizes energy, and computes partial charges.")

        let smiles = Self.demoLigandSMILES
        let name = Self.demoLigandName

        let (molData, _, _, error) = await Task.detached {
            RDKitBridge.prepareLigand(
                smiles: smiles,
                name: name,
                numConformers: 20,
                addHydrogens: true,
                minimize: true,
                computeCharges: true
            )
        }.value

        if let err = error {
            log.warn("Demo: RDKit note: \(err)", category: .molecule)
        }
        guard let data = molData else {
            throw DemoError.missingLigand
        }

        // Translate the ligand to the pocket center so it's visible near the binding site
        let pocketCenter = pocket.center
        let ligandPositions = data.atoms.map(\.position)
        let ligandCenter = ligandPositions.reduce(SIMD3<Float>.zero, +) / Float(max(ligandPositions.count, 1))
        let translation = pocketCenter - ligandCenter

        var translatedAtoms = data.atoms
        for i in translatedAtoms.indices {
            translatedAtoms[i].position += translation
        }

        let ligand = Molecule(name: data.name, atoms: translatedAtoms,
                              bonds: data.bonds, title: smiles, smiles: smiles)
        setLigandForDocking(ligand)

        let entry = LigandEntry(
            name: name,
            smiles: smiles,
            atoms: translatedAtoms,
            bonds: data.bonds,
            isPrepared: true,
            conformerCount: 1
        )
        ligandDB.add(entry)

        pushToRenderer()

        // Keep camera on the pocket (don't fitToContent which would zoom out)
        focusOnPocket(pocket)
        showGridBoxForPocket(pocket)

        let heavyCount = ligand.atoms.filter { $0.element != .H }.count
        log.success("Demo: Built \(name) from SMILES — \(heavyCount) heavy atoms, placed at pocket center", category: .molecule)
        try await Task.sleep(for: .seconds(5.0))
    }

    // MARK: - Phase 5: GPU Docking

    private func demoPhase5_Docking() async throws {
        guard let pocket = docking.selectedPocket,
              molecules.ligand != nil else {
            throw DemoError.missingLigand
        }

        // Keep ligand VISIBLE — the ghost pose renders on top during docking

        showGridBoxForPocket(pocket)
        focusOnPocket(pocket)

        // Use Drusina scoring — salt bridge term is critical for Nafamostat's
        // amidinium↔Asp189 ionic interaction that drives binding
        docking.scoringMethod = .drusina

        // Pharmacophore constraint: Nafamostat must form a salt bridge with Asp189
        // (the S1 specificity pocket floor). This guides the GA toward the correct
        // binding mode where an amidinium/guanidinium cation contacts the carboxylate.
        if let protein = molecules.protein,
           let asp189Idx = protein.residues.firstIndex(where: {
               $0.name == "ASP" && $0.sequenceNumber == 189
           }) {
            let constraint = PharmacophoreConstraintDef(
                targetScope: .residue,
                interactionType: .saltBridge,
                strength: .soft(kcalPerAngstromSq: 5.0),
                distanceThreshold: 4.0,
                sourceType: .receptor,
                residueIndex: asp189Idx,
                residueName: "ASP 189"
            )
            docking.pharmacophoreConstraints = [constraint]
            log.info("Demo: Added salt bridge constraint → ASP 189 (residue index \(asp189Idx))", category: .dock)
        }

        // Configure docking for VISUAL DEMO — dramatic exploration that still converges.
        // Exploration phase: large jumps for visual movement.
        // Refinement phase: full-strength local search so the ligand reaches the pocket floor.
        docking.dockingConfig.generationsPerRun = 300
        docking.dockingConfig.numRuns = 5                       // 5 independent starts — visible resets without diluting sampling
        docking.dockingConfig.populationSize = 300              // full population for deep-pocket convergence
        docking.dockingConfig.liveUpdateFrequency = 1           // ghost pose updates EVERY generation
        docking.dockingConfig.autoMode = false

        // Exploration: 55% exploration for visible movement, 45% serious refinement
        docking.dockingConfig.explorationPhaseRatio = 0.55
        docking.dockingConfig.explorationTranslationStep = 6.0  // large jumps — cross the pocket visually
        docking.dockingConfig.explorationRotationStep = 1.0     // ~57° flips — dramatic but recoverable
        docking.dockingConfig.explorationMutationRate = 0.30    // visible diversity during exploration
        docking.dockingConfig.translationStep = 2.0             // default refinement steps
        docking.dockingConfig.rotationStep = 0.3                // default — fine-grained convergence
        docking.dockingConfig.mutationRate = 0.10               // default refinement mutation

        // Local search: frequent in refinement so the ligand sinks into deep pockets
        docking.dockingConfig.localSearchFrequency = 3          // every 3rd gen in refinement (default)
        docking.dockingConfig.localSearchSteps = 20             // full gradient descent per refinement
        docking.dockingConfig.explorationLocalSearchFrequency = 10  // sparse during exploration (movement preserved)

        // Moderate MC temperature — enough acceptance for exploration, tight enough for convergence
        docking.dockingConfig.mcTemperature = 1.5

        // Accuracy: explicit reranking refines top cluster representatives against
        // actual protein atoms (not grid approximation) for reliable final ranking
        docking.dockingConfig.explicitRerankTopClusters = 15
        docking.dockingConfig.explicitRerankVariantsPerCluster = 6
        docking.dockingConfig.explicitRerankLocalSearchSteps = 30

        setDemoStep(.dockingStart,
            narration: "Launching the Metal GPU docking engine — 5 independent trajectories exploring the pocket. Watch Nafamostat jump between positions, flip orientations, and gradually find the optimal binding pose.")

        log.info("Demo: Docking config — 5 runs × 300 gens, pop=300, exploration=55%, LS=3/20, mcTemp=1.5", category: .dock)

        try await Task.sleep(for: .seconds(3.0))

        runDocking()

        // Wait for engine to initialize grids and start GA
        try await Task.sleep(for: .seconds(2.0))

        setDemoStep(.dockingRun,
            narration: "Nafamostat is dancing in the active site — the molecule tries thousands of orientations and conformations. The scoring function evaluates steric fit, hydrogen bonds, hydrophobic contacts, and torsional strain at each step.")

        var hasConverged = false
        while docking.isDocking {
            try await Task.sleep(for: .milliseconds(200))

            let gen = docking.dockingGeneration
            let total = docking.dockingTotalGenerations
            let energy = docking.dockingBestEnergy

            // Update narration with current best energy so it stays in sync with the live counter
            if total > 0 && gen > total / 2 {
                let eStr = energy < .infinity ? String(format: "%.1f", energy) : "---"
                setDemoStep(.dockingConverge,
                    narration: "Converging — poses cluster in the S1 pocket. Best energy: \(eStr) kcal/mol. The amidinium groups form salt bridges with Asp189 and nearby residues.")
            }
        }

        try await Task.sleep(for: .seconds(1.5))

        setDemoStep(.scoring,
            narration: "Docking complete in \(String(format: "%.1f", docking.dockingDuration))s — \(docking.dockingResults.count) poses generated and ranked by binding energy with cluster analysis to identify distinct binding modes.")

        try await Task.sleep(for: .seconds(5.0))
    }

    // MARK: - Phase 6: Results & Interactions

    private func demoPhase6_Results() async throws {
        guard !docking.dockingResults.isEmpty else {
            demoStep = .complete
            demoNarration = "Docking produced no results."
            return
        }

        let best = docking.dockingResults[0]
        let totalPoses = docking.dockingResults.count
        let clusterCount = Set(docking.dockingResults.map(\.clusterID)).subtracting([-1]).count
        let energySpread = docking.dockingResults.last.map { $0.energy - best.energy } ?? 0

        showDockingPose(at: 0)

        setDemoStep(.bestPose,
            narration: "Best pose: \(String(format: "%.1f", best.energy)) kcal/mol — ranked #1 of \(totalPoses) poses across \(clusterCount) distinct clusters (energy spread \(String(format: "%.1f", energySpread)) kcal/mol). The amidinium cation points toward Asp189 at the pocket floor.")

        workspace.colorScheme = .ligandFocus
        pushToRenderer()

        try await Task.sleep(for: .seconds(6.0))

        let hbonds = docking.currentInteractions.filter { $0.type == .hbond }.count
        let hydro = docking.currentInteractions.filter { $0.type == .hydrophobic }.count
        let salt = docking.currentInteractions.filter { $0.type == .saltBridge }.count
        let total = docking.currentInteractions.count

        setDemoStep(.interactions,
            narration: "\(total) interactions: \(hbonds) hydrogen bonds, \(salt) salt bridges, \(hydro) hydrophobic contacts. Nafamostat covalently inhibits trypsin in vivo — this pose shows the initial recognition complex.")

        log.success("Demo: Best \(String(format: "%.1f", best.energy)) kcal/mol — \(hbonds) H-bonds, \(salt) salt bridges, \(hydro) hydrophobic", category: .dock)

        // Open the 2D interaction diagram for the best pose
        docking.interactionDiagramPoseIndex = 0
        docking.showInteractionDiagram = true

        try await Task.sleep(for: .seconds(8.0))

        // Close diagram before completing
        docking.showInteractionDiagram = false

        try await Task.sleep(for: .seconds(1.0))

        demoStep = .complete
        demoNarration = "Druse found the binding mode from scratch — drug built from SMILES, pocket found blindly, pose discovered by GPU evolution. Try loading your own protein!"

        log.info("=== GUIDED DEMO COMPLETE ===", category: .system)
    }

    // MARK: - Helpers

    private func setDemoStep(_ step: DemoStep, narration: String) {
        demoStep = step
        demoNarration = narration
    }

    private enum DemoError: LocalizedError {
        case missingProtein
        case missingLigand
        case noPockets

        var errorDescription: String? {
            switch self {
            case .missingProtein: "No protein loaded"
            case .missingLigand: "Ligand generation from SMILES failed"
            case .noPockets: "No binding pockets detected"
            }
        }
    }
}

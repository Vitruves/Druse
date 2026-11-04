// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import Metal

// MARK: - Protein Preparation, Energy Minimization, Hydrogen/Charge Management

extension AppViewModel {

    // MARK: - Options

    /// What happens to crystallographic waters during cleanup.
    enum WaterHandling: Sendable, Equatable {
        /// Leave every water in place.
        case keep
        /// Delete every water.
        case remove
        /// Delete bulk water but keep waters within `radius` Å of the selected pocket centre
        /// (they then act as rigid receptor atoms during docking).
        case pocket(radius: Float)
    }

    /// Which hydrogens the preparation pipeline places.
    enum HydrogenMode: String, Sendable {
        /// Polar and non-polar hydrogens from residue templates (the GUI default).
        case all
        /// Polar hydrogens only (united-atom style receptor).
        case polar
        /// No hydrogens: heavy-atom model with rebuilt and repacked side chains.
        case none
    }

    /// Stage switches of the one-click "Prepare for Docking" pipeline. The defaults are the
    /// GUI's pipeline; the console exposes each switch as a `prep` option.
    struct ProteinPreparationOptions: Sendable {
        /// Model short missing loops (≤ 15 residues) before chain breaks are capped.
        var buildLoops = true
        /// Delete ligands, ions other than structural metals, and other non-standard residues.
        var removeNonStandard = true
        var waters: WaterHandling = .keep
        /// Cap the termini of remaining chain breaks with ACE/NME.
        var capChainBreaks = true
        var hydrogens: HydrogenMode = .all
        /// Assign partial charges with `docking.chargeMethod` (plus the electrostatic fallback).
        var assignCharges = true
    }

    /// How freshly built loop atoms are relaxed after the loop builder places them.
    enum LoopRefinement: String, Sendable {
        /// Restrained Metal GPU minimisation (the GUI default).
        case gpu
        /// OpenMM minimisation with backbone angle and trans-peptide restraints.
        case openmm
        /// Keep the raw loop-builder geometry.
        case none
    }

    /// Result of a loop-building pass.
    struct LoopBuildOutcome: Sendable {
        var atoms: [Atom]
        var bonds: [Bond]
        var gapsBuilt = 0
        var residuesAdded = 0
        /// Gaps longer than the size limit, left as chain breaks.
        var gapsTooLong = 0
        /// Whether the residue identities came from SEQRES (otherwise polyalanine).
        var usedSEQRES = false
        /// Refinement that actually ran (`.none` when the requested one was unavailable).
        var refinement: LoopRefinement = .none
    }

    /// Residues whose side chains `packSidechains` repacks.
    enum SidechainPackingScope: Sendable {
        /// Rebuild missing side-chain heavy atoms from templates, then repack only those residues.
        case incomplete
        /// Repack every rotameric residue.
        case all
        /// Repack the residues with these `FASPRSidechainPacker.residueKey` keys.
        case residues(Set<String>)
    }

    /// Largest gap (in residues) the loop builder models.
    static let maxBuildableLoopLength = 15

    // MARK: - Preparation history

    /// Logs a completed preparation step and keeps it for `prep report`.
    func recordPreparation(_ step: String, _ detail: String) {
        preparationLog.append(PreparationLogEntry(step: step, detail: detail, pH: molecules.protonationPH))
        log.success(detail, category: .prep)
    }

    /// Water residue keys ("chain_seq") present in `atoms`, the set that pocket-water
    /// retention marks as rigid receptor waters.
    private func waterResidueKeys(in atoms: [Atom]) -> Set<String> {
        let waterNames: Set<String> = ["HOH", "WAT", "DOD", "H2O"]
        return Set(atoms.filter { waterNames.contains($0.residueName) }.map { "\($0.chainID)_\($0.residueSeq)" })
    }

    // MARK: - Protein Preparation

    func removeWaters() {
        guard let prot = molecules.protein else {
            log.info("[Prep] removeWaters skipped: no protein loaded", category: .prep)
            return
        }
        let result = ProteinPreparation.removeWaters(atoms: prot.atoms, bonds: prot.bonds)
        if result.removedCount == 0 {
            log.info("No waters found to remove", category: .prep)
            workspace.statusMessage = "No waters to remove"
            return
        }
        let updated = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        updated.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = updated
        molecules.keptWaterKeys.removeAll()
        pushToRenderer()
        renderer?.fitToContent()
        molecules.proteinPrepared = true
        recordPreparation("Remove waters", "Removed \(result.removedCount) water molecules")
        workspace.statusMessage = "\(result.removedCount) waters removed"

        molecules.preparationReport = ProteinPreparation.analyze(
            atoms: result.atoms, bonds: result.bonds, waterCount: 0
        )
    }

    /// Remove waters but keep those within a radius of the binding pocket.
    /// Kept waters are treated as rigid receptor atoms during grid generation.
    func keepPocketWaters() {
        guard let prot = molecules.protein else {
            log.info("[Prep] keepPocketWaters skipped: no protein loaded", category: .prep)
            return
        }
        guard let pocket = docking.selectedPocket else {
            log.warn("Select a pocket first to define which waters to retain", category: .prep)
            return
        }

        let radius = molecules.pocketWaterRadius
        let result = ProteinPreparation.removeWaters(
            atoms: prot.atoms, bonds: prot.bonds,
            keepingNearby: [pocket.center], within: radius
        )

        let keptWaterResidues = waterResidueKeys(in: result.atoms)
        molecules.keptWaterKeys = keptWaterResidues

        if result.removedCount == 0 && keptWaterResidues.isEmpty {
            log.info("No waters found", category: .prep)
            workspace.statusMessage = "No waters to manage"
            return
        }

        let updated = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        updated.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = updated
        pushToRenderer()
        molecules.proteinPrepared = true

        recordPreparation(
            "Keep pocket waters",
            "Kept \(keptWaterResidues.count) bridging water(s) within \(String(format: "%.1f", radius)) Å of pocket, " +
            "removed \(result.removedCount) bulk water(s)"
        )
        workspace.statusMessage = "\(keptWaterResidues.count) waters kept, \(result.removedCount) removed"

        molecules.preparationReport = ProteinPreparation.analyze(
            atoms: result.atoms, bonds: result.bonds,
            waterCount: keptWaterResidues.count
        )
    }

    func removeNonStandardResidues() {
        guard let prot = molecules.protein else {
            log.info("[Prep] removeNonStandardResidues skipped: no protein loaded", category: .prep)
            return
        }
        let result = ProteinPreparation.removeNonStandardResidues(
            atoms: prot.atoms,
            bonds: prot.bonds,
            keepingWaters: true,
            keepingExistingCaps: true
        )

        if result.removedResidueCount == 0 {
            log.info("No non-standard residues found to remove", category: .prep)
            workspace.statusMessage = "No non-standard residues"
            return
        }

        let updated = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        updated.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = updated
        molecules.proteinPrepared = true
        pushToRenderer()
        renderer?.fitToContent()
        recordPreparation(
            "Remove non-standard residues",
            "Removed \(result.removedResidueCount) non-standard residue(s) (\(result.removedAtomCount) atoms)"
        )
        workspace.statusMessage = "\(result.removedResidueCount) non-standard residue(s) removed"
        molecules.preparationReport = ProteinPreparation.analyze(atoms: result.atoms, bonds: result.bonds)
    }

    func removeAltConfs() {
        guard let prot = molecules.protein else {
            log.info("[Prep] removeAltConfs skipped: no protein loaded", category: .prep)
            return
        }
        let resolved = ProteinPreparation.selectPreferredAltConformers(atoms: prot.atoms, bonds: prot.bonds)
        if resolved.removedAtomCount == 0 {
            log.info("No alternate conformations found", category: .prep)
            return
        }

        let updated = Molecule(name: prot.name, atoms: resolved.atoms, bonds: resolved.bonds, title: prot.title)
        updated.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = updated
        molecules.proteinPrepared = true
        pushToRenderer()
        recordPreparation("Resolve alternate locations", "Removed \(resolved.removedAtomCount) alternate conformation atoms")
        workspace.statusMessage = "\(resolved.removedAtomCount) alt confs removed"

        molecules.preparationReport = ProteinPreparation.analyze(atoms: resolved.atoms, bonds: resolved.bonds)
    }

    /// Caps the termini of uncapped chain breaks with ACE/NME so a missing segment does not
    /// leave charged artificial termini in the receptor. Like every cleanup pass it also keeps
    /// only the preferred alternate conformation. Returns the number of cap residues added.
    @discardableResult
    func capChainBreaks() -> Int {
        guard let prot = molecules.protein else {
            log.info("[Prep] capChainBreaks skipped: no protein loaded", category: .prep)
            return 0
        }
        let options = ProteinPreparation.CleanupOptions(
            removeNonStandardResidues: false, waterPolicy: .keepAll,
            capChainBreaks: true, keepExistingCaps: true
        )
        let cleaned = ProteinPreparation.cleanupStructure(atoms: prot.atoms, bonds: prot.bonds, options: options)
        let added = cleaned.report.addedCappingResidues
        guard added > 0 || cleaned.report.removedAltConformerAtoms > 0 else {
            log.info("No uncapped chain breaks found", category: .prep)
            workspace.statusMessage = "No chain breaks to cap"
            return 0
        }
        let updated = Molecule(name: prot.name, atoms: cleaned.atoms, bonds: cleaned.bonds, title: prot.title)
        updated.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = updated
        molecules.proteinPrepared = true
        pushToRenderer()
        let remaining = cleaned.report.chainBreaks.filter { !$0.isCapped }.count
        recordPreparation(
            "Cap chain breaks",
            "Added \(added) ACE/NME cap residue(s); \(cleaned.report.chainBreaks.count) chain break(s), \(remaining) uncapped"
        )
        workspace.statusMessage = "\(added) cap residue(s) added"
        molecules.preparationReport = ProteinPreparation.analyze(atoms: cleaned.atoms, bonds: cleaned.bonds)
        return added
    }

    /// Returns nil without a protein; otherwise a task that completes once hydrogens are placed.
    @discardableResult
    func assignProtonation() -> Task<Void, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] assignProtonation skipped: no protein loaded", category: .prep)
            return nil
        }
        let pH = molecules.protonationPH
        molecules.isMinimizing = true
        workspace.statusMessage = "Adding polar hydrogens..."
        log.info("Adding polar hydrogens at pH \(String(format: "%.1f", pH))...", category: .prep)

        let atoms = prot.atoms
        let bonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments

        return Task.detached(priority: .userInitiated) {
            let completed = ProteinPreparation.completePhase23(
                atoms: atoms, bonds: bonds, pH: pH, polarOnly: true
            )
            let protonatedResidues = completed.protonation.filter {
                !$0.atomFormalCharges.isEmpty || !$0.protonatedAtoms.isEmpty
            }.count

            await MainActor.run {
                let mol = Molecule(name: name, atoms: completed.atoms, bonds: completed.bonds, title: title)
                mol.secondaryStructureAssignments = secondaryStructure
                self.molecules.protein = mol
                self.protonationStates = completed.protonation
                self.molecules.preparationReport = ProteinPreparation.analyze(atoms: completed.atoms, bonds: completed.bonds)
                self.pushToRenderer()

                let heavyRebuilt = completed.report.heavyAtomsAdded
                var parts: [String] = []
                if heavyRebuilt > 0 { parts.append("rebuilt \(heavyRebuilt) missing heavy atoms") }
                parts.append("\(protonatedResidues) titratable residues")
                parts.append("\(completed.report.hydrogensAdded) polar H added")
                if let nr = completed.report.networkReport, nr.moveableGroups > 0 {
                    parts.append("\(nr.flipsAccepted) flips, \(nr.rotationsOptimized) rotations optimized")
                }
                self.recordPreparation(
                    "Protonate",
                    "Polar hydrogens at pH \(String(format: "%.1f", pH)): \(parts.joined(separator: ", "))"
                )
                self.workspace.statusMessage = "Polar H at pH \(String(format: "%.1f", pH))"
                self.molecules.isMinimizing = false
            }
        }
    }

    // MARK: - Preparation Pipeline

    /// The one-click "Prepare for Docking" pipeline with the GUI's stage settings.
    /// Returns nil without a protein; otherwise a task that completes once the prepared
    /// protein is in place.
    @discardableResult
    func runEnergyMinimization() -> Task<Void, Never>? {
        runPreparationPipeline(ProteinPreparationOptions())
    }

    /// The full preparation pipeline: short-loop modelling, cleanup (alternate locations,
    /// non-standard residues, waters, chain-break caps), heavy-atom completion with FASPR
    /// repacking of rebuilt side chains, restrained minimisation, protonation at
    /// `molecules.protonationPH`, template hydrogens, H-bond network optimisation and partial
    /// charges — each stage switchable through `options`. Returns nil without a protein, or
    /// when pocket-water retention is requested without a selected pocket.
    @discardableResult
    func runPreparationPipeline(_ options: ProteinPreparationOptions) -> Task<Void, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] runPreparationPipeline skipped: no protein loaded", category: .prep)
            return nil
        }

        let waterPolicy: ProteinPreparation.WaterPolicy
        switch options.waters {
        case .keep:
            waterPolicy = .keepAll
        case .remove:
            waterPolicy = .removeAll
        case .pocket(let radius):
            guard let pocket = docking.selectedPocket else {
                log.warn("Select a pocket first to define which waters to retain", category: .prep)
                return nil
            }
            waterPolicy = .keepNearby(centers: [pocket.center], radius: radius)
        }
        let cleanupOptions = ProteinPreparation.CleanupOptions(
            removeNonStandardResidues: options.removeNonStandard,
            waterPolicy: waterPolicy,
            capChainBreaks: options.capChainBreaks,
            keepExistingCaps: true
        )

        molecules.isMinimizing = true
        log.info("Running protein structure cleanup (loops + protonation + charges)...", category: .prep)
        workspace.statusMessage = "Protein cleanup..."

        let initialAtoms = prot.atoms
        let initialBonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments
        let rawPDBContent = molecules.rawPDBContent
        let pH = molecules.protonationPH
        let chargeMethod = docking.chargeMethod
        let metalDevice = computeDevice

        return Task {
            // Short loops are built before capping — otherwise the gap gets capped off and
            // the loop builder can no longer bridge the anchors.
            let loopOutcome = options.buildLoops
                ? await Self.buildShortMissingLoops(
                    atoms: initialAtoms, bonds: initialBonds,
                    rawPDBContent: rawPDBContent, metalDevice: metalDevice)
                : LoopBuildOutcome(atoms: initialAtoms, bonds: initialBonds)

            let staged = await Task.detached(priority: .userInitiated) {
                let cleanup = ProteinPreparation.cleanupStructure(
                    atoms: loopOutcome.atoms, bonds: loopOutcome.bonds, options: cleanupOptions)
                switch options.hydrogens {
                case .all, .polar:
                    let completed = ProteinPreparation.completePhase23(
                        atoms: cleanup.atoms, bonds: cleanup.bonds, pH: pH,
                        polarOnly: options.hydrogens == .polar,
                        device: MTLCreateSystemDefaultDevice()
                    )
                    return (cleanup: cleanup.report, atoms: completed.atoms, bonds: completed.bonds,
                            heavyAtomsAdded: completed.report.heavyAtomsAdded,
                            hydrogensAdded: completed.report.hydrogensAdded,
                            protonation: completed.protonation)
                case .none:
                    let rebuilt = Self.reconstructAndRepack(
                        atoms: cleanup.atoms, bonds: cleanup.bonds, device: MTLCreateSystemDefaultDevice())
                    return (cleanup: cleanup.report, atoms: rebuilt.atoms, bonds: rebuilt.bonds,
                            heavyAtomsAdded: rebuilt.heavyAtomsAdded, hydrogensAdded: 0,
                            protonation: [Protonation.ResiduePrediction]())
                }
            }.value

            var finalAtoms = staged.atoms
            if options.assignCharges {
                let charged = await Self.computeProteinCharges(
                    atoms: staged.atoms, bonds: staged.bonds,
                    method: chargeMethod, rawPDBContent: rawPDBContent)
                finalAtoms = ProteinPreparation.applyElectrostaticFallback(to: charged.atoms)
            }

            let mol = Molecule(name: name, atoms: finalAtoms, bonds: staged.bonds, title: title)
            mol.secondaryStructureAssignments = secondaryStructure
            molecules.protein = mol
            protonationStates = staged.protonation
            switch options.waters {
            case .keep: break
            case .remove: molecules.keptWaterKeys.removeAll()
            case .pocket: molecules.keptWaterKeys = waterResidueKeys(in: finalAtoms)
            }
            pushToRenderer()
            renderer?.fitToContent()
            molecules.preparationReport = ProteinPreparation.analyze(atoms: finalAtoms, bonds: staged.bonds)

            let cleanup = staged.cleanup
            let protonationUpdates = staged.protonation.reduce(0) { $0 + $1.atomFormalCharges.count }
            let chargedAtoms = finalAtoms.filter { abs($0.charge) > 0.001 }.count
            let loopSummary: String? = loopOutcome.gapsBuilt > 0
                ? "\(loopOutcome.gapsBuilt) loop\(loopOutcome.gapsBuilt == 1 ? "" : "s") built (\(loopOutcome.residuesAdded) residues)"
                : nil
            let summary = [
                loopSummary,
                cleanup.removedAltConformerAtoms > 0 ? "\(cleanup.removedAltConformerAtoms) altloc atoms removed" : nil,
                cleanup.removedHeterogenResidues > 0 ? "\(cleanup.removedHeterogenResidues) non-standard residues removed" : nil,
                cleanup.removedWaterResidues > 0 ? "\(cleanup.removedWaterResidues) waters removed" : nil,
                cleanup.addedCappingResidues > 0 ? "\(cleanup.addedCappingResidues) cap residues added" : nil,
                staged.heavyAtomsAdded > 0 ? "\(staged.heavyAtomsAdded) heavy atoms rebuilt" : nil,
                staged.hydrogensAdded > 0 ? "\(staged.hydrogensAdded) hydrogens added" : nil,
                protonationUpdates > 0 ? "\(protonationUpdates) ionization sites updated" : nil
            ].compactMap { $0 }.joined(separator: ", ")
            let chargeSummary = options.assignCharges ? "charges on \(chargedAtoms) atoms" : "no charges assigned"

            recordPreparation(
                "Prepare for docking",
                "Structure cleanup complete (\(summary.isEmpty ? "no structural edits" : summary), " +
                "pH \(String(format: "%.1f", molecules.protonationPH)), \(chargeSummary))"
            )
            if loopOutcome.gapsTooLong > 0 {
                log.warn(
                    "\(loopOutcome.gapsTooLong) gap\(loopOutcome.gapsTooLong == 1 ? "" : "s") longer than \(Self.maxBuildableLoopLength) residues left as chain breaks (capped) — use ML structure prediction (ESM-Fold/AlphaFold) for missing domains",
                    category: .prep
                )
            }
            workspace.statusMessage = "Cleanup complete"
            molecules.proteinPrepared = true
            molecules.isMinimizing = false
        }
    }

    /// Rebuilds missing heavy atoms from residue templates and repacks the rebuilt side chains
    /// with FASPR (when the rotamer library is available): the heavy-atom part of preparation,
    /// used where hydrogens are not wanted.
    nonisolated static func reconstructAndRepack(
        atoms: [Atom], bonds: [Bond], device: MTLDevice?
    ) -> (atoms: [Atom], bonds: [Bond], heavyAtomsAdded: Int, residuesPacked: Int) {
        let rebuilt = ProteinPreparation.reconstructMissingHeavyAtoms(atoms: atoms, bonds: bonds)
        guard DunbrackRotamerLibrary.shared.isLoaded, !rebuilt.sidechainRebuiltResidueKeys.isEmpty else {
            return (rebuilt.atoms, rebuilt.bonds, rebuilt.addedAtomCount, 0)
        }
        let packed = FASPRSidechainPacker.packSidechains(
            atoms: rebuilt.atoms, bonds: rebuilt.bonds, device: device,
            onlyIncomplete: true, reconstructedResidueKeys: rebuilt.sidechainRebuiltResidueKeys
        )
        return (packed.atoms, packed.bonds, rebuilt.addedAtomCount, packed.report.residuesPacked)
    }

    /// Partial charges for a protein with the rules of the docking preparation: Gasteiger
    /// charges come from RDKit on the original PDB text (≤ 6000 atoms) merged by atom
    /// identity; EEM, QEq and xTB run on the atoms directly. `assigned` is false when the
    /// method could not run; the atoms are then returned unchanged. The electrostatic
    /// fallback is left to the caller.
    nonisolated static func computeProteinCharges(
        atoms: [Atom], bonds: [Bond], method: ChargeMethod, rawPDBContent: String?
    ) async -> (atoms: [Atom], assigned: Bool) {
        switch method {
        case .gasteiger:
            let maxAtomsForRDKitCharges = 6000
            guard let pdbContent = rawPDBContent, atoms.count <= maxAtomsForRDKitCharges else {
                return (atoms, false)
            }
            let chargeData = await Task.detached { RDKitBridge.computeChargesPDB(pdbContent: pdbContent) }.value
            guard let chargeData else { return (atoms, false) }
            let merged = ProteinPreparation.mergeProteinAtoms(
                currentAtoms: atoms, sourceAtoms: chargeData.atoms
            ) { current, source in current.charge = source.charge }
            return (merged.atoms, true)
        case .eem, .qeq, .xtb:
            let calculator: ChargeCalculator = switch method {
            case .eem: EEMChargeCalculator(device: nil)
            case .qeq: QEqChargeCalculator(device: nil)
            default: XTBChargeCalculator()
            }
            guard let charges = try? await calculator.computeCharges(atoms: atoms, bonds: bonds, totalCharge: 0) else {
                return (atoms, false)
            }
            var charged = atoms
            for i in 0..<min(charges.count, charged.count) {
                charged[i].charge = charges[i]
            }
            return (charged, true)
        }
    }

    // MARK: - Loop Building

    /// Detects residue-numbering gaps of at most `maxBuildableLoopLength` residues and models
    /// them with GPU refinement, so the chain is bridged before downstream chain-break
    /// capping. Longer gaps are only counted — they need homology or ML structure prediction.
    fileprivate static func buildShortMissingLoops(
        atoms: [Atom],
        bonds: [Bond],
        rawPDBContent: String?,
        metalDevice: MTLDevice?
    ) async -> LoopBuildOutcome {
        let gaps = ProteinPreparation.detectMissingResidues(in: atoms)
        return await buildLoops(atoms: atoms, bonds: bonds, gaps: gaps, maxLength: maxBuildableLoopLength,
                                rawPDBContent: rawPDBContent, refinement: .gpu, metalDevice: metalDevice)
    }

    /// Builds the gaps no longer than `maxLength` with the loop builder (residue identities
    /// from SEQRES when the original file is available, else polyalanine) and relaxes only the
    /// new loop atoms with the requested refinement.
    nonisolated static func buildLoops(
        atoms: [Atom],
        bonds: [Bond],
        gaps: [(chainID: String, gapStart: Int, gapEnd: Int)],
        maxLength: Int,
        rawPDBContent: String?,
        refinement: LoopRefinement,
        metalDevice: MTLDevice?
    ) async -> LoopBuildOutcome {
        let buildable = gaps.filter { ($0.gapEnd - $0.gapStart + 1) <= maxLength }
        let unbuilt = LoopBuildOutcome(atoms: atoms, bonds: bonds, gapsTooLong: gaps.count - buildable.count)
        guard !buildable.isEmpty else { return unbuilt }

        return await Task.detached(priority: .userInitiated) {
            var outcome = unbuilt
            var chainSequences: [GemmiBridge.ChainSequence] = []
            if let content = rawPDBContent {
                chainSequences = (try? GemmiBridge.entitySequences(content: content)) ?? []
            }
            let result = LoopBuilder.buildMissingLoops(
                atoms: atoms, bonds: bonds, gaps: buildable, chainSequences: chainSequences
            )
            guard result.gapsBuilt > 0 else { return outcome }

            outcome.atoms = result.atoms
            outcome.bonds = result.bonds
            outcome.gapsBuilt = result.gapsBuilt
            outcome.residuesAdded = result.residuesAdded
            outcome.usedSEQRES = !chainSequences.isEmpty

            switch refinement {
            case .none:
                break
            case .openmm:
                guard druse_openmm_available() else { break }
                outcome.atoms = refineLoopsWithOpenMM(
                    atoms: result.atoms, bonds: result.bonds, originalAtomCount: atoms.count)
                outcome.refinement = .openmm
            case .gpu:
                guard let dev = metalDevice, let accelerator = LoopMetalAccelerator(device: dev) else { break }
                var isLoopAtom = [Bool](repeating: false, count: result.atoms.count)
                for i in atoms.count..<result.atoms.count { isLoopAtom[i] = true }
                let loopAngles = extractLoopBackboneAngles(
                    atoms: result.atoms, bonds: result.bonds, isLoopAtom: isLoopAtom)
                let loopTorsions = extractLoopOmegaTorsions(
                    atoms: result.atoms, bonds: result.bonds, isLoopAtom: isLoopAtom)
                let refineInput = LoopMetalAccelerator.LoopRefineInput(
                    atoms: result.atoms, bonds: result.bonds,
                    isLoopAtom: isLoopAtom, angles: loopAngles, torsions: loopTorsions)
                let output = accelerator.refine(input: refineInput, maxIterations: 500, tolerance: 0.05)
                for i in 0..<result.atoms.count where isLoopAtom[i] {
                    outcome.atoms[i].position = output.positions[i]
                }
                outcome.refinement = .gpu
            }
            return outcome
        }.value
    }

    // MARK: - Fix Missing Residues (Detection + Loop Building + Atom Reconstruction)

    /// Combined action: detect gaps, build missing loops, and reconstruct missing atoms.
    /// Returns nil without a protein; otherwise a task that completes once the repair is applied.
    @discardableResult
    func detectAndFixMissingResidues() -> Task<Void, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] fixMissingResidues skipped: no protein loaded", category: .prep)
            return nil
        }

        let gaps = ProteinPreparation.detectMissingResidues(in: prot.atoms)
        if gaps.isEmpty {
            log.success("No missing residues detected -- sequence is contiguous", category: .prep)
        } else {
            log.warn("Detected \(gaps.count) gap(s) in residue numbering:", category: .prep)
            for gap in gaps {
                let count = gap.gapEnd - gap.gapStart + 1
                log.warn("  Chain \(gap.chainID): residues \(gap.gapStart)-\(gap.gapEnd) missing (\(count) residue\(count == 1 ? "" : "s"))", category: .prep)
            }
        }

        if molecules.preparationReport != nil {
            molecules.preparationReport?.missingResidues = gaps
        }

        molecules.isMinimizing = true
        workspace.statusMessage = "Fixing missing residues..."

        let atoms = prot.atoms
        let bonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments
        let rawContent = molecules.rawPDBContent
        let hadHydrogens = atoms.contains { $0.element == .H }
        let capturedPH = molecules.protonationPH
        let capturedChargeMethod = docking.chargeMethod
        let metalDevice = computeDevice

        return Task {
            // Phase 1: Build missing loop residues (if gaps exist)
            let loops = await Self.buildLoops(
                atoms: atoms, bonds: bonds, gaps: gaps, maxLength: Self.maxBuildableLoopLength,
                rawPDBContent: rawContent, refinement: .gpu, metalDevice: metalDevice)

            let repaired = await Task.detached(priority: .userInitiated) {
                // Phase 2: Reconstruct missing heavy atoms in existing residues (no hydrogen addition)
                let reconstructed = ProteinPreparation.reconstructMissingHeavyAtoms(
                    atoms: loops.atoms, bonds: loops.bonds
                )
                guard hadHydrogens else {
                    return (atoms: reconstructed.atoms, bonds: reconstructed.bonds,
                            atomsRebuilt: reconstructed.addedAtomCount, hydrogensAdded: 0,
                            protonation: [Protonation.ResiduePrediction]())
                }
                // Phase 3: the protein was already prepared (had hydrogens) — strip them and
                // re-run protonation + hydrogen addition so newly built residues are fully
                // integrated with the prior preparation state.
                let heavyIndices = reconstructed.atoms.indices.filter { reconstructed.atoms[$0].element != .H }
                let heavy = ProteinPreparation.remapSubstructure(
                    atoms: reconstructed.atoms, bonds: reconstructed.bonds, selectedIndices: heavyIndices)
                let phase23 = ProteinPreparation.completePhase23(
                    atoms: heavy.atoms, bonds: heavy.bonds, pH: capturedPH, device: metalDevice
                )
                return (atoms: phase23.atoms, bonds: phase23.bonds,
                        atomsRebuilt: reconstructed.addedAtomCount,
                        hydrogensAdded: phase23.report.hydrogensAdded,
                        protonation: phase23.protonation)
            }.value

            var finalAtoms = repaired.atoms
            if hadHydrogens {
                let charged = await Self.computeProteinCharges(
                    atoms: repaired.atoms, bonds: repaired.bonds,
                    method: capturedChargeMethod, rawPDBContent: rawContent)
                finalAtoms = ProteinPreparation.applyElectrostaticFallback(to: charged.atoms)
                protonationStates = repaired.protonation
            }

            let mol = Molecule(name: name, atoms: finalAtoms, bonds: repaired.bonds, title: title)
            mol.secondaryStructureAssignments = secondaryStructure
            molecules.protein = mol
            molecules.preparationReport = ProteinPreparation.analyze(atoms: finalAtoms, bonds: repaired.bonds)
            pushToRenderer()
            renderer?.fitToContent()

            var messages: [String] = []
            if loops.gapsBuilt > 0 { messages.append("\(loops.gapsBuilt) loop(s) built (\(loops.residuesAdded) residues)") }
            if repaired.atomsRebuilt > 0 { messages.append("\(repaired.atomsRebuilt) heavy atoms rebuilt") }
            if repaired.hydrogensAdded > 0 { messages.append("re-protonated (\(repaired.hydrogensAdded) H)") }
            if messages.isEmpty {
                log.success("Protein structure is complete — no missing residues or atoms", category: .prep)
                workspace.statusMessage = "Structure complete"
            } else {
                recordPreparation("Repair structure", "Fixed: \(messages.joined(separator: ", "))")
                workspace.statusMessage = messages.joined(separator: ", ")
            }
            molecules.isMinimizing = false
            molecules.proteinPrepared = true
        }
    }

    // MARK: - Build Missing Loops

    /// Models missing loops of at most `maxLength` residues and relaxes them with
    /// `refinement`, without touching the rest of the structure. Returns nil without a
    /// protein; otherwise a task yielding the outcome (nil when there was no gap to build).
    @discardableResult
    func buildMissingLoops(maxLength: Int = maxBuildableLoopLength, refinement: LoopRefinement = .gpu) -> Task<LoopBuildOutcome?, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] buildMissingLoops skipped: no protein loaded", category: .prep)
            return nil
        }

        let gaps = ProteinPreparation.detectMissingResidues(in: prot.atoms)
        guard !gaps.isEmpty else {
            log.success("No missing residues detected -- nothing to build", category: .prep)
            workspace.statusMessage = "No gaps to build"
            return Task { nil }
        }

        molecules.isMinimizing = true
        log.info("Building missing residue loops...", category: .prep)
        workspace.statusMessage = "Building loops..."

        let rawContent = molecules.rawPDBContent
        let metalDevice = computeDevice

        return Task {
            let outcome = await Self.buildLoops(
                atoms: prot.atoms, bonds: prot.bonds, gaps: gaps, maxLength: maxLength,
                rawPDBContent: rawContent, refinement: refinement, metalDevice: metalDevice)
            defer { molecules.isMinimizing = false }

            guard outcome.gapsBuilt > 0 else {
                log.warn("No gaps could be built (anchors missing or gaps longer than \(maxLength) residues)", category: .prep)
                workspace.statusMessage = "No loops built"
                return outcome
            }

            let mol = Molecule(name: prot.name, atoms: outcome.atoms, bonds: outcome.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
            molecules.preparationReport = ProteinPreparation.analyze(atoms: outcome.atoms, bonds: outcome.bonds)
            pushToRenderer()
            renderer?.fitToContent()

            let refined = switch outcome.refinement {
            case .gpu: ", Metal GPU-refined"
            case .openmm: ", OpenMM-refined"
            case .none: ""
            }
            recordPreparation(
                "Build loops",
                "Built \(outcome.gapsBuilt) loop(s) with \(outcome.residuesAdded) residues " +
                "(from \(outcome.usedSEQRES ? "SEQRES" : "polyalanine")\(refined))"
            )
            if outcome.gapsTooLong > 0 {
                log.warn("\(outcome.gapsTooLong) gap(s) longer than \(maxLength) residues left unbuilt", category: .prep)
            }
            workspace.statusMessage = "\(outcome.residuesAdded) loop residues built"
            molecules.proteinPrepared = true
            return outcome
        }
    }

    // MARK: - OpenMM Loop Refinement

    /// Relaxes the loop atoms appended after `originalAtomCount` with OpenMM while the rest of
    /// the structure stays restrained; returns the input unchanged when OpenMM fails.
    nonisolated static func refineLoopsWithOpenMM(
        atoms: [Atom],
        bonds: [Bond],
        originalAtomCount: Int
    ) -> [Atom] {
        // Identify loop atoms (those added by LoopBuilder)
        var isLoopAtom = [Bool](repeating: false, count: atoms.count)
        for i in originalAtomCount..<atoms.count {
            isLoopAtom[i] = true
        }

        // Build OpenMM atom array (heavy atoms only for loop refinement)
        var ommAtoms: [DruseOpenMMAtom] = []
        ommAtoms.reserveCapacity(atoms.count)
        for atom in atoms {
            var ommAtom = DruseOpenMMAtom()
            ommAtom.x = atom.position.x
            ommAtom.y = atom.position.y
            ommAtom.z = atom.position.z
            ommAtom.charge = atom.charge
            ommAtom.sigmaNm = Float(atom.element.vdwRadius) * 2.0 / 10.0 * 0.8909  // approx sigma
            ommAtom.epsilonKJ = 0.36  // generic LJ epsilon
            ommAtom.mass = atom.element.mass
            ommAtom.atomicNum = Int32(atom.element.rawValue)
            ommAtom.isPocket = false
            ommAtoms.append(ommAtom)
        }

        // Build OpenMM bond array. Backbone bonds of a freshly built loop start far
        // from ideal, so their equilibrium length is the ideal (Engh & Huber) value;
        // keeping the starting distance would freeze the bad geometry. Other bonds
        // come from ideal templates and keep their current length.
        var ommBonds: [DruseOpenMMBond] = []
        for bond in bonds {
            let a1 = bond.atomIndex1
            let a2 = bond.atomIndex2
            guard a1 >= 0, a1 < atoms.count, a2 >= 0, a2 < atoms.count else { continue }
            let lengthAngstrom = LoopMetalAccelerator.idealBackboneBondLength(atoms[a1], atoms[a2])
                ?? simd_distance(atoms[a1].position, atoms[a2].position)
            let dist = lengthAngstrom / 10.0  // Å → nm
            var ommBond = DruseOpenMMBond()
            ommBond.atom1 = Int32(a1)
            ommBond.atom2 = Int32(a2)
            ommBond.lengthNm = dist
            ommBonds.append(ommBond)
        }

        // Build backbone angle restraints for loop residues
        var ommAngles: [DruseOpenMMAngle] = []
        // Find triplets N-CA-C, CA-C-N(next), C-N(next)-CA(next) in loop region
        let loopBackboneAngles = extractLoopBackboneAngles(atoms: atoms, bonds: bonds, isLoopAtom: isLoopAtom)
        for (a1, a2, a3, angleDeg) in loopBackboneAngles {
            var angle = DruseOpenMMAngle()
            angle.atom1 = Int32(a1)
            angle.atom2 = Int32(a2)
            angle.atom3 = Int32(a3)
            angle.angleDegrees = angleDeg
            angle.forceConstant = 460.0  // kJ/mol/rad^2
            ommAngles.append(angle)
        }

        // Build omega torsion restraints (trans peptide bond: 180°)
        var ommTorsions: [DruseOpenMMTorsion] = []
        let loopOmegaTorsions = extractLoopOmegaTorsions(atoms: atoms, bonds: bonds, isLoopAtom: isLoopAtom)
        for (a1, a2, a3, a4) in loopOmegaTorsions {
            var torsion = DruseOpenMMTorsion()
            torsion.atom1 = Int32(a1)
            torsion.atom2 = Int32(a2)
            torsion.atom3 = Int32(a3)
            torsion.atom4 = Int32(a4)
            torsion.periodicity = 2
            torsion.phaseDegrees = 180.0
            torsion.forceConstant = 40.0  // kJ/mol
            ommTorsions.append(torsion)
        }

        // Call OpenMM
        let resultPtr = ommAtoms.withUnsafeBufferPointer { atomsBuf in
            ommBonds.withUnsafeBufferPointer { bondsBuf in
                ommAngles.withUnsafeBufferPointer { anglesBuf in
                    ommTorsions.withUnsafeBufferPointer { torsionsBuf in
                        isLoopAtom.withUnsafeBufferPointer { loopBuf in
                            druse_openmm_refine_loop(
                                atomsBuf.baseAddress,
                                Int32(ommAtoms.count),
                                bondsBuf.baseAddress,
                                Int32(ommBonds.count),
                                anglesBuf.baseAddress,
                                Int32(ommAngles.count),
                                torsionsBuf.baseAddress,
                                Int32(ommTorsions.count),
                                loopBuf.baseAddress,
                                1000
                            )
                        }
                    }
                }
            }
        }

        guard let ptr = resultPtr else { return atoms }
        defer { druse_free_openmm_loop_result(ptr) }

        guard ptr.pointee.success, ptr.pointee.atomCount == Int32(atoms.count) else {
            return atoms
        }

        // Apply refined positions
        var refinedAtoms = atoms
        for i in 0..<atoms.count {
            refinedAtoms[i].position = SIMD3<Float>(
                ptr.pointee.refinedPositionsX[i],
                ptr.pointee.refinedPositionsY[i],
                ptr.pointee.refinedPositionsZ[i]
            )
        }

        return refinedAtoms
    }

    nonisolated private static func extractLoopBackboneAngles(
        atoms: [Atom],
        bonds: [Bond],
        isLoopAtom: [Bool]
    ) -> [(Int, Int, Int, Float)] {
        // Find backbone triplets in loop residues
        var angles: [(Int, Int, Int, Float)] = []

        // Group loop atoms by residue
        var residueAtoms: [String: [Int]] = [:]  // "chainID_seq" → [atomIndex]
        for i in 0..<atoms.count where isLoopAtom[i] {
            let key = "\(atoms[i].chainID)_\(atoms[i].residueSeq)"
            residueAtoms[key, default: []].append(i)
        }

        for (_, indices) in residueAtoms {
            var nIdx: Int?, caIdx: Int?, cIdx: Int?
            for idx in indices {
                let name = atoms[idx].name.trimmingCharacters(in: .whitespaces)
                switch name {
                case "N": nIdx = idx
                case "CA": caIdx = idx
                case "C": cIdx = idx
                default: break
                }
            }
            // N-CA-C angle
            if let n = nIdx, let ca = caIdx, let c = cIdx {
                angles.append((n, ca, c, 111.2))
            }
        }

        return angles
    }

    nonisolated static func extractLoopOmegaTorsions(
        atoms: [Atom],
        bonds: [Bond],
        isLoopAtom: [Bool]
    ) -> [(Int, Int, Int, Int)] {
        // Find CA-C-N-CA torsions across peptide bonds in loop
        var torsions: [(Int, Int, Int, Int)] = []

        // Build adjacency for bond lookup
        var adj: [Int: Set<Int>] = [:]
        for bond in bonds {
            adj[bond.atomIndex1, default: []].insert(bond.atomIndex2)
            adj[bond.atomIndex2, default: []].insert(bond.atomIndex1)
        }

        // For each C atom in loop region, find bonded N in next residue
        for i in 0..<atoms.count where isLoopAtom[i] {
            let name = atoms[i].name.trimmingCharacters(in: .whitespaces)
            guard name == "C" else { continue }

            let cIdx = i
            let cResSeq = atoms[i].residueSeq
            let cChain = atoms[i].chainID

            // Find CA in same residue
            guard let caIdx = (0..<atoms.count).first(where: {
                atoms[$0].name.trimmingCharacters(in: .whitespaces) == "CA" &&
                atoms[$0].residueSeq == cResSeq &&
                atoms[$0].chainID == cChain
            }) else { continue }

            // Find bonded N (next residue)
            guard let nIdx = adj[cIdx]?.first(where: {
                atoms[$0].name.trimmingCharacters(in: .whitespaces) == "N" &&
                atoms[$0].residueSeq == cResSeq + 1 &&
                atoms[$0].chainID == cChain
            }) else { continue }

            // Find CA in next residue
            guard let nextCAIdx = (0..<atoms.count).first(where: {
                atoms[$0].name.trimmingCharacters(in: .whitespaces) == "CA" &&
                atoms[$0].residueSeq == cResSeq + 1 &&
                atoms[$0].chainID == cChain
            }) else { continue }

            torsions.append((caIdx, cIdx, nIdx, nextCAIdx))
        }

        return torsions
    }

    func analyzeMissingAtoms() {
        guard let prot = molecules.protein else {
            log.info("[Prep] analyzeMissingAtoms skipped: no protein loaded", category: .prep)
            return
        }

        let completeness = ProteinPreparation.analyzeResidueCompleteness(atoms: prot.atoms, bonds: prot.bonds)
        if molecules.preparationReport != nil {
            molecules.preparationReport?.residueCompleteness = completeness
        } else {
            molecules.preparationReport = ProteinPreparation.analyze(atoms: prot.atoms, bonds: prot.bonds)
        }

        if completeness.isEmpty {
            log.success("No missing or extra atoms detected in templated protein residues", category: .prep)
            workspace.statusMessage = "No missing atoms"
            return
        }

        let heavyIssues = completeness.filter { !$0.missingHeavyAtoms.isEmpty }.count
        let hydrogenIssues = completeness.filter { !$0.missingHydrogens.isEmpty }.count
        let extraIssues = completeness.filter { !$0.extraAtoms.isEmpty }.count

        log.warn(
            "Residue completeness issues: \(completeness.count) residue(s), " +
            "\(heavyIssues) with missing heavy atoms, \(hydrogenIssues) with missing hydrogens, " +
            "\(extraIssues) with extra atoms",
            category: .prep
        )
        for residue in completeness.prefix(8) {
            log.warn(
                "  Chain \(residue.chainID) \(residue.residueName) \(residue.residueSeq): \(residue.summary)",
                category: .prep
            )
        }
        if completeness.count > 8 {
            log.info("... plus \(completeness.count - 8) more incomplete residue(s)", category: .prep)
        }
        workspace.statusMessage = "\(completeness.count) residue issue(s)"
    }

    func repairMissingAtoms() {
        guard let prot = molecules.protein else {
            log.info("[Prep] repairMissingAtoms skipped: no protein loaded", category: .prep)
            return
        }

        let rebuilt = ProteinPreparation.reconstructMissingHeavyAtoms(atoms: prot.atoms, bonds: prot.bonds)
        guard rebuilt.addedAtomCount > 0 else {
            log.success("No missing heavy atoms required reconstruction", category: .prep)
            workspace.statusMessage = "No heavy atoms rebuilt"
            return
        }

        let mol = Molecule(name: prot.name, atoms: rebuilt.atoms, bonds: rebuilt.bonds, title: prot.title)
        mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = mol
        molecules.preparationReport = ProteinPreparation.analyze(atoms: rebuilt.atoms, bonds: rebuilt.bonds)
        pushToRenderer()
        renderer?.fitToContent()

        recordPreparation("Rebuild missing atoms", "Rebuilt \(rebuilt.addedAtomCount) missing heavy atom(s) from residue templates")
        workspace.statusMessage = "\(rebuilt.addedAtomCount) heavy atom(s) rebuilt"
    }

    func removeHydrogens() {
        guard let prot = molecules.protein else {
            log.info("[Prep] removeHydrogens skipped: no protein loaded", category: .prep)
            return
        }
        let kept = prot.atoms.indices.filter { prot.atoms[$0].element != .H }
        let result = ProteinPreparation.remapSubstructure(
            atoms: prot.atoms, bonds: prot.bonds, selectedIndices: kept
        )
        let mol = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = mol
        protonationStates = []

        if let lig = molecules.ligand, lig.atoms.contains(where: { $0.element == .H }) {
            let ligKept = lig.atoms.indices.filter { lig.atoms[$0].element != .H }
            let ligResult = ProteinPreparation.remapSubstructure(
                atoms: lig.atoms, bonds: lig.bonds, selectedIndices: ligKept
            )
            molecules.ligand = Molecule(name: lig.name, atoms: ligResult.atoms,
                              bonds: ligResult.bonds, title: lig.title, smiles: lig.smiles)
        }

        pushToRenderer()
        renderer?.fitToContent()
        molecules.preparationReport = ProteinPreparation.analyze(atoms: result.atoms, bonds: result.bonds)
        let removed = prot.atoms.count - result.atoms.count
        recordPreparation("Remove hydrogens", "Removed \(removed) hydrogen atoms")
        workspace.statusMessage = "\(removed) H removed"
    }

    // MARK: - C++ Protein Preparation

    /// Returns nil without a protein; otherwise a task that completes once hydrogens are added.
    @discardableResult
    func addHydrogens() -> Task<Void, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] addHydrogens skipped: no protein loaded", category: .prep)
            return nil
        }

        log.info("Adding all hydrogens (polar + nonpolar) from residue templates...", category: .prep)
        workspace.statusMessage = "Adding hydrogens..."

        return Task {
            let completed = ProteinPreparation.completePhase23(
                atoms: prot.atoms,
                bonds: prot.bonds,
                pH: molecules.protonationPH
            )

            let mol = Molecule(name: prot.name, atoms: completed.atoms, bonds: completed.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
            protonationStates = completed.protonation
            pushToRenderer()
            renderer?.fitToContent()

            let added = completed.report.hydrogensAdded
            let heavyRebuilt = completed.report.heavyAtomsAdded
            var parts: [String] = []
            if heavyRebuilt > 0 { parts.append("rebuilt \(heavyRebuilt) missing heavy atoms") }
            if added > 0 { parts.append("added \(added) hydrogens") }
            if let nr = completed.report.networkReport, nr.moveableGroups > 0 {
                parts.append("H-bond network: \(nr.flipsAccepted) flips, \(nr.rotationsOptimized) rotations")
            }

            if parts.isEmpty {
                log.success("Structure already complete — no missing atoms or hydrogens", category: .prep)
                workspace.statusMessage = "No hydrogens added"
            } else {
                recordPreparation("Add hydrogens", parts.joined(separator: ", "))
                workspace.statusMessage = added > 0 ? "\(added) H added" : "No hydrogens added"
            }
            molecules.preparationReport = ProteinPreparation.analyze(atoms: completed.atoms, bonds: completed.bonds)
        }
    }

    // MARK: - H-bond Network, Side-chain Packing, Minimization

    /// Re-optimises the hydrogen-bond network of an already protonated protein: OH/SH/NH3⁺
    /// rotors, Asn/Gln amide flips and His tautomer/flip states, scored as one network.
    /// Uses the protonation states of the last protonation (or predicts them at the current
    /// pH). Returns nil without a protonated protein; otherwise a task yielding the report.
    @discardableResult
    func optimizeHydrogenBondNetwork() -> Task<HBondNetworkOptimizer.NetworkReport, Never>? {
        guard let prot = molecules.protein, prot.atoms.contains(where: { $0.element == .H }) else {
            log.warn("[Prep] H-bond optimization needs a protonated protein — add hydrogens first", category: .prep)
            return nil
        }
        molecules.isMinimizing = true
        workspace.statusMessage = "Optimizing H-bond network..."
        let atoms = prot.atoms
        let bonds = prot.bonds
        let pH = molecules.protonationPH
        let known = protonationStates
        let device = computeDevice

        return Task {
            let network = await Task.detached(priority: .userInitiated) {
                let predictions = known.isEmpty
                    ? Protonation.predictResidueStates(atoms: atoms, bonds: bonds, pH: pH)
                    : known
                return HBondNetworkOptimizer.optimizeNetwork(
                    atoms: atoms, bonds: bonds, predictions: predictions, device: device)
            }.value

            let mol = Molecule(name: prot.name, atoms: network.atoms, bonds: network.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
            protonationStates = network.predictions
            pushToRenderer()
            let report = network.report
            recordPreparation(
                "Optimize H-bond network",
                "H-bond network: \(report.moveableGroups) movable groups in \(report.cliques) cliques, " +
                "\(report.flipsAccepted) flips, \(report.rotationsOptimized) rotations, " +
                String(format: "%.1f", report.totalEnergyImprovement) + " improvement"
            )
            workspace.statusMessage = "H-bond network optimized"
            molecules.isMinimizing = false
            return report
        }
    }

    /// Repacks side chains with FASPR (Dunbrack rotamers, DEE + tree decomposition).
    /// Works on heavy-atom models only — hydrogens would no longer match repacked side chains.
    /// Returns nil without a protein, with hydrogens present, or without the rotamer library;
    /// otherwise a task yielding the packing report.
    @discardableResult
    func packSidechains(_ scope: SidechainPackingScope) -> Task<FASPRSidechainPacker.PackingReport, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] packSidechains skipped: no protein loaded", category: .prep)
            return nil
        }
        guard !prot.atoms.contains(where: { $0.element == .H }) else {
            log.warn("[Prep] Side-chain packing needs a heavy-atom model — remove hydrogens first", category: .prep)
            return nil
        }
        guard DunbrackRotamerLibrary.shared.isLoaded else {
            log.error("[Prep] Dunbrack rotamer library is not available", category: .prep)
            return nil
        }
        molecules.isMinimizing = true
        workspace.statusMessage = "Packing side chains..."
        let atoms = prot.atoms
        let bonds = prot.bonds
        let device = computeDevice

        return Task {
            let packed = await Task.detached(priority: .userInitiated) {
                switch scope {
                case .incomplete:
                    let rebuilt = ProteinPreparation.reconstructMissingHeavyAtoms(atoms: atoms, bonds: bonds)
                    guard !rebuilt.sidechainRebuiltResidueKeys.isEmpty else {
                        return (atoms: rebuilt.atoms, bonds: rebuilt.bonds, report: FASPRSidechainPacker.PackingReport())
                    }
                    return FASPRSidechainPacker.packSidechains(
                        atoms: rebuilt.atoms, bonds: rebuilt.bonds, device: device,
                        onlyIncomplete: true, reconstructedResidueKeys: rebuilt.sidechainRebuiltResidueKeys)
                case .all:
                    return FASPRSidechainPacker.packSidechains(
                        atoms: atoms, bonds: bonds, device: device, onlyIncomplete: false)
                case .residues(let keys):
                    return FASPRSidechainPacker.packSidechains(
                        atoms: atoms, bonds: bonds, device: device,
                        onlyIncomplete: true, reconstructedResidueKeys: keys)
                }
            }.value

            let mol = Molecule(name: prot.name, atoms: packed.atoms, bonds: packed.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
            molecules.preparationReport = ProteinPreparation.analyze(atoms: packed.atoms, bonds: packed.bonds)
            pushToRenderer()
            if packed.report.residuesPacked > 0 {
                recordPreparation(
                    "Pack side chains",
                    "FASPR repacked \(packed.report.residuesPacked) side chain(s) " +
                    "(\(packed.report.totalRotamersEvaluated) rotamers, \(String(format: "%.0f", packed.report.elapsedMs)) ms)"
                )
                workspace.statusMessage = "\(packed.report.residuesPacked) side chains packed"
            } else {
                log.info("No side chains needed packing", category: .prep)
                workspace.statusMessage = "No side chains packed"
            }
            molecules.isMinimizing = false
            return packed.report
        }
    }

    /// Restrained energy minimisation of the whole protein on the GPU (L-BFGS on VDW + bond
    /// terms): backbone atoms strongly restrained, side chains moderately, hydrogens free —
    /// the relaxation the preparation pipeline runs after repacking. Returns nil without a
    /// protein or Metal device; otherwise a task yielding the minimiser output.
    @discardableResult
    func minimizeProtein(maxIterations: Int) -> Task<PreparationMinimizer.MinimizationOutput, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] minimizeProtein skipped: no protein loaded", category: .prep)
            return nil
        }
        guard let device = computeDevice, let minimizer = PreparationMinimizer(device: device) else {
            log.error("[Prep] Restrained minimization needs a Metal device", category: .prep)
            return nil
        }
        molecules.isMinimizing = true
        workspace.statusMessage = "Minimizing protein..."
        let atoms = prot.atoms
        let bonds = prot.bonds

        return Task {
            let result = await Task.detached(priority: .userInitiated) {
                let regions: [PreparationMinimizer.AtomRegion] = atoms.map { atom in
                    if atom.element == .H { return .hydrogen }
                    let name = ProteinResidueTemplateStore.normalizeAtomName(atom.name)
                    return ["N", "CA", "C", "O"].contains(name) ? .backbone : .existingSidechain
                }
                return minimizer.minimize(
                    input: PreparationMinimizer.MinimizationInput(atoms: atoms, bonds: bonds, atomRegions: regions),
                    maxIterations: maxIterations
                )
            }.value

            var minimized = atoms
            for i in 0..<min(result.positions.count, minimized.count) {
                minimized[i].position = result.positions[i]
            }
            let mol = Molecule(name: prot.name, atoms: minimized, bonds: bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
            pushToRenderer()
            recordPreparation(
                "Minimize",
                "Restrained minimization: E = \(String(format: "%.1f", result.finalEnergy)) kcal/mol after " +
                "\(result.iterations) iteration(s)\(result.converged ? ", converged" : ", not converged")"
            )
            workspace.statusMessage = "Protein minimized"
            molecules.isMinimizing = false
            return result
        }
    }

    // MARK: - Charges

    /// Assigns partial charges with `docking.chargeMethod`. Returns nil when charges cannot be
    /// computed at all; otherwise a task yielding whether the charges were assigned.
    @discardableResult
    func assignCharges() -> Task<Bool, Never>? {
        let method = docking.chargeMethod
        if method == .gasteiger {
            return assignGasteigerCharges()
        }

        guard let prot = molecules.protein else {
            log.info("[Prep] assignCharges skipped: no protein loaded", category: .prep)
            return nil
        }
        log.info("Computing \(method.rawValue) charges...", category: .prep)
        workspace.statusMessage = "Computing \(method.rawValue) charges..."

        return Task {
            do {
                let calculator = ChargeCalculatorFactory.calculator(for: method, device: computeDevice)
                let charges = try await calculator.computeCharges(
                    atoms: prot.atoms, bonds: prot.bonds, totalCharge: 0
                )

                var updatedAtoms = prot.atoms
                for i in 0..<min(charges.count, updatedAtoms.count) {
                    updatedAtoms[i].charge = charges[i]
                }

                let mol = Molecule(name: prot.name, atoms: updatedAtoms, bonds: prot.bonds, title: prot.title)
                mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
                molecules.protein = mol
                pushToRenderer()

                let nonZero = charges.filter { abs($0) > 0.001 }.count
                recordPreparation("Assign charges", "\(method.rawValue) charges assigned: \(nonZero)/\(charges.count) non-zero")
                workspace.statusMessage = "\(method.rawValue) charges assigned"

                molecules.preparationReport = ProteinPreparation.analyze(
                    atoms: updatedAtoms, bonds: prot.bonds
                )
                return true
            } catch {
                log.warn("Charge calculation failed: \(error.localizedDescription)", category: .prep)
                workspace.statusMessage = "Charge calculation failed"
                return false
            }
        }
    }

    @discardableResult
    func assignGasteigerCharges() -> Task<Bool, Never>? {
        guard let prot = molecules.protein else {
            log.info("[Prep] assignGasteigerCharges skipped: no protein loaded", category: .prep)
            return nil
        }
        log.info("Computing Gasteiger charges via RDKit...", category: .prep)
        workspace.statusMessage = "Computing charges..."

        guard let pdbContent = molecules.rawPDBContent else {
            log.warn("[Prep] No raw PDB content available for Gasteiger charges", category: .prep)
            return nil
        }

        return Task {
            let result = await Task.detached {
                RDKitBridge.computeChargesPDB(pdbContent: pdbContent)
            }.value

            if let data = result {
                let merged = mergeProteinAtoms(currentAtoms: prot.atoms, sourceAtoms: data.atoms) { current, source in
                    current.charge = source.charge
                }
                let updated = applyElectrostaticFallback(to: merged.atoms)
                molecules.protein = Molecule(name: prot.name, atoms: updated, bonds: prot.bonds, title: prot.title)
                molecules.protein?.secondaryStructureAssignments = prot.secondaryStructureAssignments
                pushToRenderer()
                let charged = updated.filter { abs($0.charge) > 0.001 }.count
                recordPreparation("Assign charges", "Gasteiger charges assigned: \(charged)/\(updated.count) atoms with non-zero charge (\(merged.matchedCount) matched)")
                workspace.statusMessage = "\(charged) atoms charged"
            } else {
                log.error("RDKit charge computation returned nil", category: .prep)
                workspace.statusMessage = "Charge computation failed"
            }

            molecules.preparationReport = ProteinPreparation.analyze(
                atoms: molecules.protein?.atoms ?? prot.atoms,
                bonds: molecules.protein?.bonds ?? prot.bonds
            )
            return result != nil
        }
    }
}

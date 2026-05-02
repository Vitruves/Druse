// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import Metal

// MARK: - Protein Preparation, Energy Minimization, Hydrogen/Charge Management

extension AppViewModel {

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
        molecules.protein = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        molecules.keptWaterKeys.removeAll()
        pushToRenderer()
        renderer?.fitToContent()
        molecules.proteinPrepared = true
        log.success("Removed \(result.removedCount) water molecules", category: .prep)
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

        // Count how many waters we kept
        let waterNames: Set<String> = ["HOH", "WAT", "DOD", "H2O"]
        let keptWaterAtoms = result.atoms.filter { waterNames.contains($0.residueName) }
        let keptWaterResidues = Set(keptWaterAtoms.map { "\($0.chainID)_\($0.residueSeq)" })
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

        log.success(
            "Kept \(keptWaterResidues.count) bridging water(s) within \(String(format: "%.1f", radius)) Å of pocket, " +
            "removed \(result.removedCount) bulk water(s)",
            category: .prep
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
        log.success(
            "Removed \(result.removedResidueCount) non-standard residue(s) (\(result.removedAtomCount) atoms)",
            category: .prep
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
        log.success("Removed \(resolved.removedAtomCount) alternate conformation atoms", category: .prep)
        workspace.statusMessage = "\(resolved.removedAtomCount) alt confs removed"

        molecules.preparationReport = ProteinPreparation.analyze(atoms: resolved.atoms, bonds: resolved.bonds)
    }

    func assignProtonation() {
        guard let prot = molecules.protein else {
            log.info("[Prep] assignProtonation skipped: no protein loaded", category: .prep)
            return
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

        Task.detached(priority: .userInitiated) {
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
                self.log.success(
                    "Polar hydrogens at pH \(String(format: "%.1f", pH)): \(parts.joined(separator: ", "))",
                    category: .prep
                )
                self.workspace.statusMessage = "Polar H at pH \(String(format: "%.1f", pH))"
                self.molecules.isMinimizing = false
            }
        }
    }

    // MARK: - Energy Minimization

    func runEnergyMinimization() {
        guard let prot = molecules.protein else {
            log.info("[Prep] runEnergyMinimization skipped: no protein loaded", category: .prep)
            return
        }

        molecules.isMinimizing = true
        log.info("Running protein structure cleanup (protonation + charges)...", category: .prep)
        workspace.statusMessage = "Protein cleanup..."

        let atoms = prot.atoms
        let bonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments
        let rawPDBContent = molecules.rawPDBContent
        let pH = molecules.protonationPH
        let chargeMethod = docking.chargeMethod

        Task {
            let prepared = await Task.detached(priority: .userInitiated) {
                ProteinPreparation.prepareForDocking(
                    atoms: atoms,
                    bonds: bonds,
                    rawPDBContent: rawPDBContent,
                    pH: pH,
                    chargeMethod: chargeMethod,
                    device: MTLCreateSystemDefaultDevice()
                )
            }.value

            let mol = Molecule(name: name, atoms: prepared.atoms, bonds: prepared.bonds, title: title)
            mol.secondaryStructureAssignments = secondaryStructure
            molecules.protein = mol
            pushToRenderer()
            renderer?.fitToContent()
            molecules.preparationReport = ProteinPreparation.analyze(atoms: prepared.atoms, bonds: prepared.bonds)

            let summary = [
                prepared.report.altConformerAtomsRemoved > 0 ? "\(prepared.report.altConformerAtomsRemoved) altloc atoms removed" : nil,
                prepared.report.heterogenResiduesRemoved > 0 ? "\(prepared.report.heterogenResiduesRemoved) non-standard residues removed" : nil,
                prepared.report.cappingResiduesAdded > 0 ? "\(prepared.report.cappingResiduesAdded) cap residues added" : nil,
                prepared.report.missingHeavyAtomsAdded > 0 ? "\(prepared.report.missingHeavyAtomsAdded) heavy atoms rebuilt" : nil,
                prepared.report.hydrogensAdded > 0 ? "\(prepared.report.hydrogensAdded) hydrogens added" : nil,
                prepared.report.protonationUpdates > 0 ? "\(prepared.report.protonationUpdates) ionization sites updated" : nil
            ].compactMap { $0 }.joined(separator: ", ")

            log.success(
                "Structure cleanup complete (\(summary.isEmpty ? "no structural edits" : summary), " +
                "pH \(String(format: "%.1f", molecules.protonationPH)), charges on \(prepared.report.nonZeroChargeAtoms) atoms)",
                category: .prep
            )
            workspace.statusMessage = "Cleanup complete"
            molecules.proteinPrepared = true
            molecules.isMinimizing = false
        }
    }

    // MARK: - Fix Missing Residues (Detection + Loop Building + Atom Reconstruction)

    /// Combined action: detect gaps, build missing loops, and reconstruct missing atoms.
    func detectAndFixMissingResidues() {
        guard let prot = molecules.protein else {
            log.info("[Prep] fixMissingResidues skipped: no protein loaded", category: .prep)
            return
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
        let buildableGaps = gaps.filter { ($0.gapEnd - $0.gapStart + 1) <= 15 }
        let hadHydrogens = atoms.contains { $0.element == .H }
        let capturedPH = molecules.protonationPH
        let capturedChargeMethod = docking.chargeMethod

        Task.detached(priority: .userInitiated) {
            var workAtoms = atoms
            var workBonds = bonds
            var loopsBuilt = 0
            var residuesAdded = 0

            // Phase 1: Build missing loop residues (if gaps exist)
            if !buildableGaps.isEmpty {
                var chainSequences: [GemmiBridge.ChainSequence] = []
                if let content = rawContent {
                    do {
                        chainSequences = try GemmiBridge.entitySequences(content: content)
                    } catch { /* fall back to polyalanine */ }
                }

                let loopResult = LoopBuilder.buildMissingLoops(
                    atoms: workAtoms, bonds: workBonds,
                    gaps: buildableGaps, chainSequences: chainSequences
                )
                if loopResult.gapsBuilt > 0 {
                    // GPU refine loop regions
                    let metalDevice = await MainActor.run { self.renderer?.device }
                    var finalAtoms = loopResult.atoms
                    if let dev = metalDevice, let accelerator = LoopMetalAccelerator(device: dev) {
                        var isLoopAtom = [Bool](repeating: false, count: loopResult.atoms.count)
                        for i in workAtoms.count..<loopResult.atoms.count { isLoopAtom[i] = true }
                        let loopAngles = AppViewModel.extractLoopBackboneAngles(
                            atoms: loopResult.atoms, bonds: loopResult.bonds, isLoopAtom: isLoopAtom)
                        let loopTorsions = AppViewModel.extractLoopOmegaTorsions(
                            atoms: loopResult.atoms, bonds: loopResult.bonds, isLoopAtom: isLoopAtom)
                        let refineInput = LoopMetalAccelerator.LoopRefineInput(
                            atoms: loopResult.atoms, bonds: loopResult.bonds,
                            isLoopAtom: isLoopAtom, angles: loopAngles, torsions: loopTorsions)
                        let output = accelerator.refine(input: refineInput, maxIterations: 500, tolerance: 0.05)
                        for i in 0..<loopResult.atoms.count where isLoopAtom[i] {
                            finalAtoms[i] = Atom(
                                id: finalAtoms[i].id, element: finalAtoms[i].element,
                                position: output.positions[i], name: finalAtoms[i].name,
                                residueName: finalAtoms[i].residueName, residueSeq: finalAtoms[i].residueSeq,
                                chainID: finalAtoms[i].chainID, charge: finalAtoms[i].charge,
                                formalCharge: finalAtoms[i].formalCharge, isHetAtom: finalAtoms[i].isHetAtom,
                                occupancy: finalAtoms[i].occupancy, tempFactor: finalAtoms[i].tempFactor)
                        }
                    }
                    workAtoms = finalAtoms
                    workBonds = loopResult.bonds
                    loopsBuilt = loopResult.gapsBuilt
                    residuesAdded = loopResult.residuesAdded
                }
            }

            // Phase 2: Reconstruct missing heavy atoms in existing residues (no hydrogen addition)
            let reconstructed = ProteinPreparation.reconstructMissingHeavyAtoms(
                atoms: workAtoms, bonds: workBonds
            )
            let atomsRebuilt = reconstructed.addedAtomCount
            let capturedLoopsBuilt = loopsBuilt
            let capturedResiduesAdded = residuesAdded

            var finalAtoms = reconstructed.atoms
            var finalBonds = reconstructed.bonds

            // Phase 3: If the protein was already prepared (had hydrogens), re-run
            // protonation + hydrogen addition + charge assignment so that newly built
            // residues are fully integrated with the prior preparation state.
            var hAdded = 0
            if hadHydrogens {
                // Strip existing hydrogens — completePhase23 will re-add them consistently
                let heavyAtoms = finalAtoms.filter { $0.element != .H }
                let heavyIndices = finalAtoms.indices.filter { finalAtoms[$0].element != .H }
                let indexMap = Dictionary(uniqueKeysWithValues: heavyIndices.enumerated().map { ($0.element, $0.offset) })
                let heavyBonds = finalBonds.compactMap { bond -> Bond? in
                    guard let i1 = indexMap[bond.atomIndex1], let i2 = indexMap[bond.atomIndex2] else { return nil }
                    return Bond(id: i1 * 100000 + i2, atomIndex1: i1, atomIndex2: i2, order: bond.order)
                }

                let metalDevice = await MainActor.run { self.renderer?.device }
                let phase23 = ProteinPreparation.completePhase23(
                    atoms: heavyAtoms, bonds: heavyBonds, pH: capturedPH, device: metalDevice
                )
                finalAtoms = phase23.atoms
                finalBonds = phase23.bonds
                hAdded = phase23.report.hydrogensAdded

                // Re-apply charges
                switch capturedChargeMethod {
                case .gasteiger:
                    if let pdbContent = rawContent, finalAtoms.count <= 6000,
                       let chargeData = RDKitBridge.computeChargesPDB(pdbContent: pdbContent) {
                        let merged = ProteinPreparation.mergeProteinAtoms(
                            currentAtoms: finalAtoms, sourceAtoms: chargeData.atoms
                        ) { current, source in current.charge = source.charge }
                        finalAtoms = merged.atoms
                    }
                case .eem, .qeq, .xtb:
                    let calculator: ChargeCalculator = {
                        switch capturedChargeMethod {
                        case .eem: return EEMChargeCalculator(device: nil)
                        case .qeq: return QEqChargeCalculator(device: nil)
                        case .xtb: return XTBChargeCalculator()
                        default:   return GasteigerChargeCalculator()
                        }
                    }()
                    if let charges = try? await calculator.computeCharges(
                        atoms: finalAtoms, bonds: finalBonds, totalCharge: 0
                    ) {
                        for i in 0..<min(charges.count, finalAtoms.count) {
                            finalAtoms[i].charge = charges[i]
                        }
                    }
                }
                finalAtoms = ProteinPreparation.applyElectrostaticFallback(to: finalAtoms)
            }

            let capturedFinalAtoms = finalAtoms
            let capturedFinalBonds = finalBonds
            let capturedHAdded = hAdded

            await MainActor.run {
                let mol = Molecule(name: name, atoms: capturedFinalAtoms, bonds: capturedFinalBonds, title: title)
                mol.secondaryStructureAssignments = secondaryStructure
                self.molecules.protein = mol
                self.molecules.preparationReport = ProteinPreparation.analyze(atoms: capturedFinalAtoms, bonds: capturedFinalBonds)
                self.pushToRenderer()
                self.renderer?.fitToContent()

                var messages: [String] = []
                if capturedLoopsBuilt > 0 { messages.append("\(capturedLoopsBuilt) loop(s) built (\(capturedResiduesAdded) residues)") }
                if atomsRebuilt > 0 { messages.append("\(atomsRebuilt) heavy atoms rebuilt") }
                if capturedHAdded > 0 { messages.append("re-protonated (\(capturedHAdded) H)") }
                if messages.isEmpty {
                    self.log.success("Protein structure is complete — no missing residues or atoms", category: .prep)
                    self.workspace.statusMessage = "Structure complete"
                } else {
                    self.log.success("Fixed: \(messages.joined(separator: ", "))", category: .prep)
                    self.workspace.statusMessage = messages.joined(separator: ", ")
                }
                self.molecules.isMinimizing = false
                self.molecules.proteinPrepared = true
            }
        }
    }

    func detectAndReportMissingResidues() {
        guard let prot = molecules.protein else {
            log.info("[Prep] detectMissingResidues skipped: no protein loaded", category: .prep)
            return
        }

        let gaps = ProteinPreparation.detectMissingResidues(in: prot.atoms)
        if gaps.isEmpty {
            log.success("No missing residues detected -- sequence is contiguous", category: .prep)
            workspace.statusMessage = "No gaps found"
        } else {
            log.warn("Detected \(gaps.count) gap(s) in residue numbering:", category: .prep)
            for gap in gaps {
                let count = gap.gapEnd - gap.gapStart + 1
                log.warn("  Chain \(gap.chainID): residues \(gap.gapStart)-\(gap.gapEnd) missing (\(count) residue\(count == 1 ? "" : "s"))", category: .prep)
            }
            let buildableGaps = gaps.filter { ($0.gapEnd - $0.gapStart + 1) <= 15 }
            if !buildableGaps.isEmpty {
                let totalResidues = buildableGaps.reduce(0) { $0 + ($1.gapEnd - $1.gapStart + 1) }
                log.info("Use 'Build Missing Loops' to model \(buildableGaps.count) gap(s) (\(totalResidues) residues, max 15 per loop)", category: .prep)
            }
            let longGaps = gaps.filter { ($0.gapEnd - $0.gapStart + 1) > 15 }
            for gap in longGaps {
                let count = gap.gapEnd - gap.gapStart + 1
                log.warn("  Chain \(gap.chainID): gap of \(count) residues too long for ab initio loop building (max 15)", category: .prep)
            }
            workspace.statusMessage = "\(gaps.count) gap(s) detected"
        }

        if molecules.preparationReport != nil {
            molecules.preparationReport?.missingResidues = gaps
        }

        // Reconstruct missing heavy atoms from residue templates
        molecules.isMinimizing = true
        log.info("Reconstructing missing heavy atoms from templates...", category: .prep)

        let atoms = prot.atoms
        let bonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments

        Task.detached(priority: .userInitiated) {
            let reconstructed = ProteinPreparation.reconstructMissingHeavyAtoms(
                atoms: atoms, bonds: bonds
            )

            await MainActor.run {
                let mol = Molecule(name: name, atoms: reconstructed.atoms, bonds: reconstructed.bonds, title: title)
                mol.secondaryStructureAssignments = secondaryStructure
                self.molecules.protein = mol
                self.molecules.preparationReport = ProteinPreparation.analyze(atoms: reconstructed.atoms, bonds: reconstructed.bonds)
                self.pushToRenderer()
                if reconstructed.addedAtomCount > 0 {
                    self.log.success("Rebuilt \(reconstructed.addedAtomCount) missing heavy atoms from residue templates", category: .prep)
                    self.workspace.statusMessage = "\(reconstructed.addedAtomCount) atoms rebuilt"
                } else {
                    self.log.info("No missing heavy atoms found to rebuild", category: .prep)
                }
                self.molecules.isMinimizing = false
            }
        }
    }

    // MARK: - Build Missing Loops

    func buildMissingLoops() {
        guard let prot = molecules.protein else {
            log.info("[Prep] buildMissingLoops skipped: no protein loaded", category: .prep)
            return
        }

        let gaps = ProteinPreparation.detectMissingResidues(in: prot.atoms)
        guard !gaps.isEmpty else {
            log.success("No missing residues detected -- nothing to build", category: .prep)
            workspace.statusMessage = "No gaps to build"
            return
        }

        molecules.isMinimizing = true
        log.info("Building missing residue loops...", category: .prep)
        workspace.statusMessage = "Building loops..."

        let atoms = prot.atoms
        let bonds = prot.bonds
        let name = prot.name
        let title = prot.title
        let secondaryStructure = prot.secondaryStructureAssignments
        let rawContent = molecules.rawPDBContent

        Task.detached(priority: .userInitiated) {
            // 1. Extract SEQRES sequences
            var chainSequences: [GemmiBridge.ChainSequence] = []
            if let content = rawContent {
                do {
                    chainSequences = try GemmiBridge.entitySequences(content: content)
                } catch {
                    // Fall back to polyalanine
                }
            }

            // 2. Build loops (backbone + sidechains)
            let result = LoopBuilder.buildMissingLoops(
                atoms: atoms,
                bonds: bonds,
                gaps: gaps,
                chainSequences: chainSequences
            )

            guard result.gapsBuilt > 0 else {
                await MainActor.run {
                    self.log.warn("No gaps could be built (anchors missing or gaps too long)", category: .prep)
                    self.workspace.statusMessage = "No loops built"
                    self.molecules.isMinimizing = false
                }
                return
            }

            // 3. Metal GPU refinement of loop regions
            let metalDevice = await MainActor.run { self.renderer?.device }
            var finalAtoms = result.atoms
            var wasRefined = false
            if let dev = metalDevice, let accelerator = LoopMetalAccelerator(device: dev) {
                var isLoopAtom = [Bool](repeating: false, count: result.atoms.count)
                for i in atoms.count..<result.atoms.count { isLoopAtom[i] = true }

                let loopAngles = AppViewModel.extractLoopBackboneAngles(
                    atoms: result.atoms, bonds: result.bonds, isLoopAtom: isLoopAtom
                )
                let loopTorsions = AppViewModel.extractLoopOmegaTorsions(
                    atoms: result.atoms, bonds: result.bonds, isLoopAtom: isLoopAtom
                )

                let refineInput = LoopMetalAccelerator.LoopRefineInput(
                    atoms: result.atoms,
                    bonds: result.bonds,
                    isLoopAtom: isLoopAtom,
                    angles: loopAngles,
                    torsions: loopTorsions
                )
                let output = accelerator.refine(input: refineInput, maxIterations: 500, tolerance: 0.05)
                for i in 0..<result.atoms.count where isLoopAtom[i] {
                    finalAtoms[i] = Atom(
                        id: finalAtoms[i].id,
                        element: finalAtoms[i].element,
                        position: output.positions[i],
                        name: finalAtoms[i].name,
                        residueName: finalAtoms[i].residueName,
                        residueSeq: finalAtoms[i].residueSeq,
                        chainID: finalAtoms[i].chainID,
                        charge: finalAtoms[i].charge,
                        formalCharge: finalAtoms[i].formalCharge,
                        isHetAtom: finalAtoms[i].isHetAtom,
                        occupancy: finalAtoms[i].occupancy,
                        tempFactor: finalAtoms[i].tempFactor
                    )
                }
                wasRefined = true
            }

            let capturedAtoms = finalAtoms
            let capturedChainSequences = chainSequences
            let capturedWasRefined = wasRefined
            await MainActor.run {
                let mol = Molecule(name: name, atoms: capturedAtoms, bonds: result.bonds, title: title)
                mol.secondaryStructureAssignments = secondaryStructure
                self.molecules.protein = mol
                self.molecules.preparationReport = ProteinPreparation.analyze(atoms: capturedAtoms, bonds: result.bonds)
                self.pushToRenderer()
                self.renderer?.fitToContent()

                let seqSource = capturedChainSequences.isEmpty ? "polyalanine" : "SEQRES"
                let refinedStr = capturedWasRefined ? ", Metal GPU-refined" : ""
                self.log.success(
                    "Built \(result.gapsBuilt) loop(s) with \(result.residuesAdded) residues " +
                    "(from \(seqSource)\(refinedStr))",
                    category: .prep
                )
                self.workspace.statusMessage = "\(result.residuesAdded) loop residues built"
                self.molecules.isMinimizing = false
                self.molecules.proteinPrepared = true
            }
        }
    }

    // MARK: - OpenMM Loop Refinement

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

        // Build OpenMM bond array
        var ommBonds: [DruseOpenMMBond] = []
        for bond in bonds {
            let a1 = bond.atomIndex1
            let a2 = bond.atomIndex2
            guard a1 >= 0, a1 < atoms.count, a2 >= 0, a2 < atoms.count else { continue }
            let dist = simd_distance(atoms[a1].position, atoms[a2].position) / 10.0  // Å → nm
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

        log.success("Rebuilt \(rebuilt.addedAtomCount) missing heavy atom(s) from residue templates", category: .prep)
        workspace.statusMessage = "\(rebuilt.addedAtomCount) heavy atom(s) rebuilt"
    }

    // MARK: - Solvation Shell

    func addSolvationShell() {
        // Redirect to pocket waters — the solvation shell button now calls this
        keepPocketWaters()
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
        log.success("Removed \(removed) hydrogen atoms", category: .prep)
        workspace.statusMessage = "\(removed) H removed"
    }

    // MARK: - C++ Protein Preparation

    func addHydrogens() {
        guard let prot = molecules.protein else {
            log.info("[Prep] addHydrogens skipped: no protein loaded", category: .prep)
            return
        }

        log.info("Adding all hydrogens (polar + nonpolar) from residue templates...", category: .prep)
        workspace.statusMessage = "Adding hydrogens..."

        Task {
            let completed = ProteinPreparation.completePhase23(
                atoms: prot.atoms,
                bonds: prot.bonds,
                pH: molecules.protonationPH
            )

            let mol = Molecule(name: prot.name, atoms: completed.atoms, bonds: completed.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            molecules.protein = mol
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
                log.success(parts.joined(separator: ", "), category: .prep)
                workspace.statusMessage = added > 0 ? "\(added) H added" : "No hydrogens added"
            }
            molecules.preparationReport = ProteinPreparation.analyze(atoms: completed.atoms, bonds: completed.bonds)
        }
    }

    func assignCharges() {
        let method = docking.chargeMethod
        if method == .gasteiger {
            assignGasteigerCharges()
            return
        }

        guard let prot = molecules.protein else {
            log.info("[Prep] assignCharges skipped: no protein loaded", category: .prep)
            return
        }
        log.info("Computing \(method.rawValue) charges...", category: .prep)
        workspace.statusMessage = "Computing \(method.rawValue) charges..."

        Task {
            do {
                let calculator = ChargeCalculatorFactory.calculator(for: method, device: renderer?.device)
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
                log.success("\(method.rawValue) charges assigned: \(nonZero)/\(charges.count) non-zero", category: .prep)
                workspace.statusMessage = "\(method.rawValue) charges assigned"

                molecules.preparationReport = ProteinPreparation.analyze(
                    atoms: updatedAtoms, bonds: prot.bonds
                )
            } catch {
                log.warn("Charge calculation failed: \(error.localizedDescription)", category: .prep)
                workspace.statusMessage = "Charge calculation failed"
            }
        }
    }

    func assignGasteigerCharges() {
        guard let prot = molecules.protein else {
            log.info("[Prep] assignGasteigerCharges skipped: no protein loaded", category: .prep)
            return
        }
        log.info("Computing Gasteiger charges via RDKit...", category: .prep)
        workspace.statusMessage = "Computing charges..."

        guard let pdbContent = molecules.rawPDBContent else {
            log.warn("[Prep] No raw PDB content available for Gasteiger charges", category: .prep)
            return
        }

        Task {
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
                log.success("Gasteiger charges assigned: \(charged)/\(updated.count) atoms with non-zero charge (\(merged.matchedCount) matched)", category: .prep)
                workspace.statusMessage = "\(charged) atoms charged"
            } else {
                log.error("RDKit charge computation returned nil", category: .prep)
                workspace.statusMessage = "Charge computation failed"
            }

            molecules.preparationReport = ProteinPreparation.analyze(
                atoms: molecules.protein?.atoms ?? prot.atoms,
                bonds: molecules.protein?.bonds ?? prot.bonds
            )
        }
    }
}

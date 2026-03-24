import SwiftUI

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
                let netSummary: String
                if let nr = completed.report.networkReport, nr.moveableGroups > 0 {
                    netSummary = ", \(nr.flipsAccepted) flips, \(nr.rotationsOptimized) rotations optimized"
                } else {
                    netSummary = ""
                }
                self.log.success(
                    "Added polar hydrogens at pH \(String(format: "%.1f", pH)) " +
                    "(\(protonatedResidues) titratable, \(completed.report.hydrogensAdded) polar H\(netSummary))",
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

        Task {
            let prepared = ProteinPreparation.prepareForDocking(
                atoms: prot.atoms,
                bonds: prot.bonds,
                rawPDBContent: molecules.rawPDBContent,
                pH: molecules.protonationPH,
                chargeMethod: docking.chargeMethod
            )

            let mol = Molecule(name: prot.name, atoms: prepared.atoms, bonds: prepared.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
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

    // MARK: - Fix Missing Residues (Detection)

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
            log.info("Actual modeling of missing residues requires homology tools (not yet available)", category: .prep)
            workspace.statusMessage = "\(gaps.count) gap(s) detected"
        }

        if molecules.preparationReport != nil {
            molecules.preparationReport?.missingResidues = gaps
        }
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

        log.info("Adding template-driven hydrogens...", category: .prep)
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
            let netSummary: String
            if let nr = completed.report.networkReport, nr.moveableGroups > 0 {
                netSummary = ", H-bond network: \(nr.flipsAccepted) flips, \(nr.rotationsOptimized) rotations"
            } else {
                netSummary = ""
            }
            if added == 0 {
                log.success("No additional hydrogens were required after template completion", category: .prep)
                workspace.statusMessage = "No hydrogens added"
            } else {
                log.success(
                    "Added \(added) hydrogens" +
                    (completed.report.heavyAtomsAdded > 0 ? " (rebuilt \(completed.report.heavyAtomsAdded) heavy atoms)" : "") +
                    netSummary,
                    category: .prep
                )
                workspace.statusMessage = "\(added) H added"
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

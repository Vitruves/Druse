import SwiftUI
import MetalKit

@Observable
@MainActor
final class AppViewModel {
    // MARK: - Sub-objects (grouped state)

    var workspace = WorkspaceState()
    var molecules = MoleculeManager()
    var docking = DockingCoordinator()
    var leadOpt = LeadOptimizationState()

    // MARK: - Shared services

    /// Renderer (set after MetalView is created)
    var renderer: Renderer?

    /// Ligand database
    let ligandDB = LigandDatabase()

    /// Activity log
    let log = ActivityLog.shared

    // ML models (scoring/detection)
    let druseMLScoring = DruseMLScoringInference()
    let druseRescoring = DruseRescoringInference()
    let pocketDetectorML = PocketDetectorInference()
    let admetPredictor = ADMETPredictor()

    // MARK: - Context menu target

    /// Persistent target object for context menu actions (required for NSMenu selectors).
    private var _contextMenuTarget: ContextMenuTarget?
    var contextMenuTarget: ContextMenuTarget {
        if let t = _contextMenuTarget { return t }
        let t = ContextMenuTarget(viewModel: self)
        _contextMenuTarget = t
        return t
    }

    init() {}

    // MARK: - Ligand Management (Single Entry Point)

    /// Set any molecule as the active ligand for docking.
    /// ALL import paths (LigandDatabaseView, LigandDatabaseWindow,
    /// DockingTabView picker, PDB co-crystallized) MUST call this single method.
    func setLigandForDocking(_ molecule: Molecule) {
        let normalizedAtoms = molecule.atoms.enumerated().map { i, atom in
            Atom(id: i, element: atom.element, position: atom.position,
                 name: atom.name,
                 residueName: "LIG",
                 residueSeq: 1,
                 chainID: "L",
                 charge: atom.charge,
                 formalCharge: atom.formalCharge,
                 isHetAtom: true,
                 occupancy: atom.occupancy,
                 tempFactor: atom.tempFactor,
                 altLoc: atom.altLoc)
        }
        let normalizedLigand = Molecule(name: molecule.name, atoms: normalizedAtoms,
                                         bonds: molecule.bonds, title: molecule.title)

        molecules.ligand = normalizedLigand
        docking.originalDockingLigand = nil
        docking.dockingResults = []
        docking.currentInteractions = []
        renderer?.updateInteractionLines([])
        pushToRenderer()
        // Only fit camera when no protein is loaded (i.e., ligand-only view).
        // When protein is present, preserve the user's current camera orientation
        // and position — the ligand will appear at its coordinates in the existing scene.
        if molecules.protein == nil {
            renderer?.fitToContent()
        }
        let heavyCount = molecule.atoms.filter { $0.element != .H }.count
        log.success("Set \(molecule.name) as active ligand (\(molecule.atomCount) atoms, \(heavyCount) heavy, \(molecule.bondCount) bonds)", category: .molecule)
        if let smiles = molecule.smiles, !smiles.isEmpty {
            log.info("  SMILES: \(smiles.prefix(80))\(smiles.count > 80 ? "..." : "")", category: .molecule)
        }
        workspace.statusMessage = "\(molecule.name) ready for docking"
        // Clear conformers when switching ligand (will be set separately if available)
        workspace.ligandConformers = []
        workspace.activeConformerIndex = 0
    }

    func setLigandConformers(_ conformers: [(atoms: [Atom], bonds: [Bond], energy: Double)]) {
        workspace.ligandConformers = conformers.map {
            WorkspaceState.Conformer(atoms: $0.atoms, bonds: $0.bonds, energy: $0.energy)
        }
        workspace.activeConformerIndex = 0
    }

    func switchConformer(to index: Int) {
        guard index >= 0, index < workspace.ligandConformers.count else { return }
        let conf = workspace.ligandConformers[index]
        workspace.activeConformerIndex = index
        let name = molecules.ligand?.name ?? "Ligand"
        let smiles = molecules.ligand?.smiles
        molecules.ligand = Molecule(name: name, atoms: conf.atoms, bonds: conf.bonds, title: molecules.ligand?.title ?? "", smiles: smiles)
        pushToRenderer()
    }

    /// Extract a chain from the protein, set it as the active ligand, and add
    /// it to the ligand database. The chain is removed from the protein molecule.
    func extractChainAsLigand(chainID: String) {
        guard let prot = molecules.protein else { return }
        guard let chain = prot.chains.first(where: { $0.id == chainID }) else {
            log.warn("Chain \(chainID) not found", category: .molecule)
            return
        }

        let chainAtomIndices = Set(chain.residueIndices.flatMap { resIdx -> [Int] in
            guard resIdx < prot.residues.count else { return [] }
            return prot.residues[resIdx].atomIndices
        })
        guard !chainAtomIndices.isEmpty else {
            log.warn("Chain \(chainID) has no atoms", category: .molecule)
            return
        }

        let firstRes = chain.residueIndices.first.flatMap { idx in
            idx < prot.residues.count ? prot.residues[idx] : nil
        }
        let ligandName = firstRes?.name ?? "Chain_\(chainID)"

        let (ligAtoms, ligBonds) = ProteinPreparation.remapSubstructure(
            atoms: prot.atoms,
            bonds: prot.bonds,
            selectedIndices: Array(chainAtomIndices).sorted()
        )

        let ligMol = Molecule(name: ligandName, atoms: ligAtoms, bonds: ligBonds)
        setLigandForDocking(ligMol)

        var smiles = ""
        if !ligBonds.isEmpty {
            smiles = RDKitBridge.atomsBondsToSMILES(atoms: ligAtoms, bonds: ligBonds) ?? ""
        }
        if !smiles.isEmpty {
            ligMol.smiles = smiles
            log.info("Generated SMILES for \(ligandName): \(smiles.prefix(60))...", category: .molecule)
        }

        let dbEntry = LigandEntry(
            name: ligandName,
            smiles: smiles,
            atoms: ligAtoms,
            bonds: ligBonds,
            isPrepared: true,
            conformerCount: 1
        )
        ligandDB.add(dbEntry)

        let keepIndices = prot.atoms.indices.filter { !chainAtomIndices.contains($0) }
        let (newProtAtoms, newProtBonds) = ProteinPreparation.remapSubstructure(
            atoms: prot.atoms,
            bonds: prot.bonds,
            selectedIndices: keepIndices
        )
        let newProt = Molecule(name: prot.name, atoms: newProtAtoms,
                               bonds: newProtBonds, title: prot.title)
        newProt.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = newProt

        pushToRenderer()
        renderer?.fitToContent()
        log.success("Extracted \(ligandName) (\(ligAtoms.count) atoms) from chain \(chainID) as ligand", category: .molecule)
        workspace.statusMessage = "\(ligandName) extracted as ligand"
    }

    /// Define a docking pocket from the current active ligand's position.
    func definePocketFromLigand() {
        guard let prot = molecules.protein, let lig = molecules.ligand else { return }
        if let pocket = BindingSiteDetector.ligandGuidedPocket(
            protein: prot, ligand: lig, excludedChainIDs: workspace.hiddenChainIDs
        ) {
            docking.detectedPockets = [pocket]
            docking.selectedPocket = pocket
            log.success("Pocket from ligand: \(pocket.residueIndices.count) residues, \(Int(pocket.volume)) A\u{00B3}", category: .dock)
        }
    }

    /// Remove a specific chain from the protein molecule.
    func removeChain(chainID: String) {
        guard let prot = molecules.protein else { return }
        guard let chain = prot.chains.first(where: { $0.id == chainID }) else { return }

        let chainAtomIndices = Set(chain.residueIndices.flatMap { resIdx -> [Int] in
            guard resIdx < prot.residues.count else { return [] }
            return prot.residues[resIdx].atomIndices
        })
        guard !chainAtomIndices.isEmpty else { return }

        let keepIndices = prot.atoms.indices.filter { !chainAtomIndices.contains($0) }
        if keepIndices.isEmpty {
            clearProtein()
            return
        }

        let (newAtoms, newBonds) = ProteinPreparation.remapSubstructure(
            atoms: prot.atoms, bonds: prot.bonds, selectedIndices: keepIndices
        )
        let newProt = Molecule(name: prot.name, atoms: newAtoms,
                               bonds: newBonds, title: prot.title)
        newProt.secondaryStructureAssignments = prot.secondaryStructureAssignments
        molecules.protein = newProt
        workspace.hiddenChainIDs.remove(chainID)
        pushToRenderer()
        log.info("Removed chain \(chainID) (\(chainAtomIndices.count) atoms)", category: .molecule)
        workspace.statusMessage = "Chain \(chainID) removed"
    }

    /// Clear the entire protein.
    func clearProtein() {
        let name = molecules.protein?.name ?? "protein"
        molecules.protein = nil
        molecules.rawPDBContent = nil
        molecules.preparationReport = nil
        docking.detectedPockets = []
        docking.selectedPocket = nil
        workspace.hiddenChainIDs.removeAll()
        pushToRenderer()
        log.info("Cleared \(name)", category: .molecule)
        workspace.statusMessage = "No protein loaded"
    }

    /// Clear the active ligand.
    func clearLigand() {
        let name = molecules.ligand?.name ?? "ligand"
        molecules.ligand = nil
        docking.originalDockingLigand = nil
        docking.dockingResults = []
        docking.currentInteractions = []
        renderer?.updateInteractionLines([])
        pushToRenderer()
        log.info("Cleared \(name)", category: .molecule)
        workspace.statusMessage = "No ligand loaded"
    }

    // MARK: - Side Chain Helpers

    func sideChainResidueSet(protein prot: Molecule) -> Set<Int> {
        switch workspace.sideChainDisplay {
        case .none:
            return []
        case .selected:
            return workspace.selectedResidueIndices
        case .interacting:
            guard let lig = molecules.ligand else { return [] }
            var residues = Set<Int>()
            let distSq: Float = 25  // 5 Å cutoff
            for ligAtom in lig.atoms where ligAtom.element != .H {
                for (resIdx, residue) in prot.residues.enumerated() {
                    if residues.contains(resIdx) { continue }
                    for atomIdx in residue.atomIndices {
                        guard atomIdx < prot.atoms.count else { continue }
                        if simd_distance_squared(ligAtom.position, prot.atoms[atomIdx].position) <= distSq {
                            residues.insert(resIdx)
                            break
                        }
                    }
                }
            }
            return residues
        case .all:
            return Set(0..<prot.residues.count)
        }
    }

    // MARK: - Push Data to Renderer

    func pushToRenderer() {
        guard let renderer else { return }

        var allPositions: [SIMD3<Float>] = []
        if workspace.showProtein, let prot = molecules.protein {
            if workspace.hiddenChainIDs.isEmpty {
                allPositions.append(contentsOf: prot.atoms.map(\.position))
            } else {
                allPositions.append(contentsOf: prot.atoms.filter { !workspace.hiddenChainIDs.contains($0.chainID) }.map(\.position))
            }
        }
        if workspace.showLigand, let lig = molecules.ligand {
            allPositions.append(contentsOf: lig.atoms.map(\.position))
        }
        renderer.updateAllMoleculePositions(allPositions)

        var allAtoms: [Atom] = []
        var allBonds: [Bond] = []

        // Ribbon mode
        if workspace.renderMode == .ribbon, let prot = molecules.protein, workspace.showProtein {
            let visibleProt: Molecule
            if workspace.hiddenChainIDs.isEmpty {
                visibleProt = prot
            } else {
                let visAtoms = prot.atoms.filter { !workspace.hiddenChainIDs.contains($0.chainID) }
                var idMap: [Int: Int] = [:]
                var remapped: [Atom] = []
                for atom in visAtoms {
                    idMap[atom.id] = remapped.count
                    remapped.append(Atom(
                        id: remapped.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                var remappedBonds: [Bond] = []
                for bond in prot.bonds {
                    if let a = idMap[bond.atomIndex1], let b = idMap[bond.atomIndex2] {
                        remappedBonds.append(Bond(id: remappedBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
                visibleProt = Molecule(name: prot.name, atoms: remapped, bonds: remappedBonds, title: prot.title)
                visibleProt.secondaryStructureAssignments = prot.secondaryStructureAssignments
            }

            let (ribbonVerts, ribbonIdxs) = RibbonMeshGenerator.generateForMolecule(visibleProt)
            renderer.updateRibbonMesh(vertices: ribbonVerts, indices: ribbonIdxs)

            var caControlPoints: [(position: SIMD3<Float>, atomID: Int)] = []
            for atom in prot.atoms {
                if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                if atom.name.trimmingCharacters(in: .whitespaces) == "CA" {
                    caControlPoints.append((position: atom.position, atomID: atom.id))
                }
            }
            renderer.updateRibbonCAControlPoints(caControlPoints)

            if workspace.showLigand, !docking.isDocking, let lig = molecules.ligand {
                let filteredAtoms = workspace.showHydrogens ? lig.atoms : lig.atoms.filter { $0.element != .H }
                var ligIdMap: [Int: Int] = [:]
                for atom in filteredAtoms {
                    ligIdMap[atom.id] = allAtoms.count
                    allAtoms.append(Atom(
                        id: allAtoms.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                for bond in lig.bonds {
                    if let a = ligIdMap[bond.atomIndex1], let b = ligIdMap[bond.atomIndex2] {
                        allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
            }

            // Side chain display
            if workspace.sideChainDisplay != .none, workspace.showProtein {
                let sideChainResidueIndices = sideChainResidueSet(protein: prot)
                if !sideChainResidueIndices.isEmpty {
                    var scIdMap: [Int: Int] = [:]
                    for residueIdx in sideChainResidueIndices {
                        guard residueIdx < prot.residues.count else { continue }
                        let residue = prot.residues[residueIdx]
                        for atomIdx in residue.atomIndices {
                            guard atomIdx < prot.atoms.count else { continue }
                            let atom = prot.atoms[atomIdx]
                            if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                            if !workspace.showHydrogens && atom.element == .H { continue }
                            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)
                            if SideChainDisplay.backboneAtomNames.contains(trimmedName) { continue }
                            scIdMap[atom.id] = allAtoms.count
                            allAtoms.append(Atom(
                                id: allAtoms.count, element: atom.element, position: atom.position,
                                name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                                chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                                isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                            ))
                        }
                    }
                    for bond in prot.bonds {
                        if let a = scIdMap[bond.atomIndex1], let b = scIdMap[bond.atomIndex2] {
                            allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                        }
                    }
                }
            }

            renderer.selectedAtomIndex = workspace.selectedAtomIndex ?? -1
            renderer.selectedResidueAtomIndices = []
            renderer.renderMode = .ballAndStick
            renderer.updateMoleculeData(atoms: allAtoms, bonds: allBonds)
            return
        }

        // Non-ribbon mode
        renderer.clearRibbonMesh()

        if workspace.showProtein, let prot = molecules.protein {
            var protIdMap: [Int: Int] = [:]
            for atom in prot.atoms {
                if !workspace.showHydrogens && atom.element == .H { continue }
                if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                if workspace.hiddenAtomIndices.contains(atom.id) { continue }
                protIdMap[atom.id] = allAtoms.count
                allAtoms.append(Atom(
                    id: allAtoms.count, element: atom.element, position: atom.position,
                    name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                    chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                ))
            }
            for bond in prot.bonds {
                if let a = protIdMap[bond.atomIndex1], let b = protIdMap[bond.atomIndex2] {
                    allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
        }

        if workspace.showLigand, !docking.isDocking, let lig = molecules.ligand {
            let ligOffset = molecules.protein?.atoms.count ?? 0
            var ligIdMap: [Int: Int] = [:]
            for atom in lig.atoms {
                if !workspace.showHydrogens && atom.element == .H { continue }
                if workspace.hiddenAtomIndices.contains(atom.id + ligOffset) { continue }
                ligIdMap[atom.id] = allAtoms.count
                allAtoms.append(Atom(
                    id: allAtoms.count, element: atom.element, position: atom.position,
                    name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                    chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                ))
            }
            for bond in lig.bonds {
                if let a = ligIdMap[bond.atomIndex1], let b = ligIdMap[bond.atomIndex2] {
                    allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
        }

        renderer.selectedAtomIndex = workspace.selectedAtomIndex ?? -1
        renderer.selectedResidueAtomIndices = Set(
            workspace.selectedResidueIndices.flatMap { resIdx -> [Int] in
                if let prot = molecules.protein, resIdx < prot.residues.count {
                    return prot.residues[resIdx].atomIndices
                }
                return []
            }
        )
        renderer.renderMode = workspace.renderMode

        renderer.enableClipping = workspace.enableClipping
        renderer.clipNearZ = workspace.clipNearZ
        renderer.clipFarZ = workspace.clipFarZ

        renderer.updateMoleculeData(atoms: allAtoms, bonds: allBonds)

        // Update constraint indicators
        if !docking.pharmacophoreConstraints.isEmpty {
            renderer.updateConstraintIndicators(docking.pharmacophoreConstraints, atoms: allAtoms)
        } else {
            renderer.clearConstraintIndicators()
        }
    }

    // MARK: - Computed Properties

    var selectedAtom: Atom? {
        guard let idx = workspace.selectedAtomIndex else { return nil }
        let allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])
        guard idx < allAtoms.count else { return nil }
        return allAtoms[idx]
    }

    var proteinHasHydrogens: Bool {
        molecules.protein?.atoms.contains { $0.element == .H } ?? false
    }

    var allChains: [Chain] {
        var chainsByID: [String: Chain] = [:]
        var order: [String] = []

        if let prot = molecules.protein {
            for chain in prot.chains {
                chainsByID[chain.id] = chain
                order.append(chain.id)
            }
        }
        if let lig = molecules.ligand {
            for chain in lig.chains {
                if chainsByID[chain.id] == nil {
                    chainsByID[chain.id] = chain
                    order.append(chain.id)
                }
            }
        }

        return order.compactMap { chainsByID[$0] }
    }

    // MARK: - Heavy Atom Bond Remapping

    /// Build bonds for heavy atoms only (excludes hydrogens), with remapped indices.
    func buildHeavyBonds(from molecule: Molecule) -> [Bond] {
        var oldToNew: [Int: Int] = [:]
        var newIdx = 0
        for atom in molecule.atoms where atom.element != .H {
            oldToNew[atom.id] = newIdx
            newIdx += 1
        }
        return molecule.bonds.compactMap { bond in
            guard let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }
    }

    // MARK: - Protein Atom Merging Utilities

    struct ProteinAtomIdentity: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let atomName: String
        let atomicNumber: Int
        let isHetAtom: Bool
    }

    func proteinAtomIdentity(for atom: Atom) -> ProteinAtomIdentity {
        ProteinAtomIdentity(
            chainID: atom.chainID,
            residueSeq: atom.residueSeq,
            residueName: atom.residueName,
            atomName: atom.name.trimmingCharacters(in: .whitespaces),
            atomicNumber: atom.element.rawValue,
            isHetAtom: atom.isHetAtom
        )
    }

    func mergeProteinAtoms(
        currentAtoms: [Atom],
        sourceAtoms: [Atom],
        update: (inout Atom, Atom) -> Void
    ) -> (atoms: [Atom], matchedCount: Int) {
        var buckets: [ProteinAtomIdentity: [Atom]] = [:]
        for atom in sourceAtoms {
            buckets[proteinAtomIdentity(for: atom), default: []].append(atom)
        }

        var offsets: [ProteinAtomIdentity: Int] = [:]
        var updated = currentAtoms
        var matchedCount = 0

        for index in updated.indices {
            let identity = proteinAtomIdentity(for: updated[index])
            let offset = offsets[identity, default: 0]
            guard let bucket = buckets[identity], offset < bucket.count else { continue }
            update(&updated[index], bucket[offset])
            offsets[identity] = offset + 1
            matchedCount += 1
        }

        return (updated, matchedCount)
    }

    func applyElectrostaticFallback(to atoms: [Atom]) -> [Atom] {
        atoms.map { atom in
            var atom = atom
            if abs(atom.charge) <= 0.0001 {
                atom.charge = Float(atom.formalCharge)
            }
            return atom
        }
    }

    func canAdoptRDKitProteinGeometry(currentAtoms: [Atom], sourceAtoms: [Atom]) -> Bool {
        guard currentAtoms.count == sourceAtoms.count, !currentAtoms.isEmpty else { return false }

        var buckets: [ProteinAtomIdentity: [Atom]] = [:]
        for atom in sourceAtoms {
            buckets[proteinAtomIdentity(for: atom), default: []].append(atom)
        }

        var offsets: [ProteinAtomIdentity: Int] = [:]
        var matchedCount = 0
        var totalDrift: Float = 0
        var maxDrift: Float = 0

        for atom in currentAtoms {
            let identity = proteinAtomIdentity(for: atom)
            let offset = offsets[identity, default: 0]
            guard let bucket = buckets[identity], offset < bucket.count else { return false }

            let source = bucket[offset]
            offsets[identity] = offset + 1
            matchedCount += 1

            let drift = simd_distance(atom.position, source.position)
            totalDrift += drift
            maxDrift = max(maxDrift, drift)
        }

        let averageDrift = totalDrift / Float(max(matchedCount, 1))
        return matchedCount == currentAtoms.count && averageDrift <= 0.15 && maxDrift <= 0.5
    }
}

// MARK: - Context Menu Target (NSObject for @objc selectors)

/// A helper NSObject that bridges NSMenu target-action to the @MainActor AppViewModel.
@MainActor
final class ContextMenuTarget: NSObject {
    private weak var viewModel: AppViewModel?

    init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    @objc func selectChainAction(_ sender: NSMenuItem) {
        guard let chainID = sender.representedObject as? String else { return }
        viewModel?.selectChain(chainID)
    }

    @objc func selectNearbyResiduesAction() {
        viewModel?.selectResiduesWithinDistance(5.0)
    }

    @objc func definePocketAction() {
        viewModel?.definePocketFromSelection()
    }

    @objc func hideSelectionAction() {
        viewModel?.hideSelection()
    }

    @objc func showAllAction() {
        viewModel?.showAllAtoms()
    }

    @objc func removeSelectionAction() {
        viewModel?.removeSelection()
    }

    @objc func centerOnSelectionAction() {
        viewModel?.centerOnSelection()
    }

    @objc func resetViewAction() {
        viewModel?.resetView()
    }

    @objc func addDockingConstraintAction() {
        viewModel?.showConstraintSheetFromSelection()
    }
}

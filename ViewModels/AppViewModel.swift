// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

    /// Tracks which LigandEntry is currently active in the 3D view / docking.
    var activeLigandEntryID: UUID?

    /// Activity log
    let log = ActivityLog.shared

    // ML models (detection/ADMET)
    let pocketDetectorML = PocketDetectorInference()
    let admetPredictor = ADMETPredictor()

    // MARK: - Guided Demo

    enum DemoStep: String, CaseIterable {
        case idle
        case fetching        = "Fetching Trypsin from RCSB..."
        case parsing         = "Building drug & parsing structure..."
        case overview        = "Exploring the protein"
        case ribbon          = "Ribbon view — secondary structure"
        case pocketScan      = "Scanning surface for binding pockets..."
        case pocketFound     = "S1 pocket found"
        case gridSetup       = "Setting up docking search grid"
        case dockingStart    = "Launching GPU genetic algorithm..."
        case dockingRun      = "Nafamostat searching the pocket..."
        case dockingConverge = "Converging on optimal binding pose..."
        case scoring         = "Scoring & ranking poses"
        case bestPose        = "Best pose — analyzing interactions"
        case interactions    = "Molecular interactions mapped"
        case complete        = "Demo complete — explore the results!"
    }

    var demoStep: DemoStep = .idle
    var demoNarration: String = ""
    var isDemoRunning: Bool {
        demoStep != .idle && demoStep != .complete
    }

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
    /// - Parameters:
    ///   - molecule: The molecule to set as active ligand.
    ///   - entryID: Optional UUID of the `LigandEntry` this molecule came from.
    func setLigandForDocking(_ molecule: Molecule, entryID: UUID? = nil) {
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
                                         bonds: molecule.bonds, title: molecule.title,
                                         smiles: molecule.smiles)

        molecules.ligand = normalizedLigand
        activeLigandEntryID = entryID
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

        var smiles = ""
        if !ligBonds.isEmpty {
            smiles = RDKitBridge.atomsBondsToSMILES(atoms: ligAtoms, bonds: ligBonds) ?? ""
        }

        let ligMol = Molecule(
            name: ligandName,
            atoms: ligAtoms,
            bonds: ligBonds,
            title: smiles,
            smiles: smiles.isEmpty ? nil : smiles
        )
        setLigandForDocking(ligMol)

        if !smiles.isEmpty {
            log.info("Generated SMILES for \(ligandName): \(smiles.prefix(60))...", category: .molecule)
        }

        // Check if an entry with the same sourceChainID, name, or SMILES already exists
        // to avoid duplicates (e.g., when user re-imports the same PDB file)
        if let existingIdx = ligandDB.entries.firstIndex(where: {
            $0.sourceChainID == chainID ||
            $0.name == ligandName ||
            (!smiles.isEmpty && $0.smiles == smiles)
        }) {
            // Update existing entry instead of adding a duplicate
            ligandDB.entries[existingIdx].atoms = ligAtoms
            ligandDB.entries[existingIdx].bonds = ligBonds
            ligandDB.entries[existingIdx].isPrepared = true
            ligandDB.entries[existingIdx].sourceChainID = chainID
            if !smiles.isEmpty { ligandDB.entries[existingIdx].smiles = smiles }
            activeLigandEntryID = ligandDB.entries[existingIdx].id
            log.info("Updated existing \(ligandName) in ligand database", category: .molecule)
        } else {
            let dbEntry = LigandEntry(
                name: ligandName,
                smiles: smiles,
                atoms: ligAtoms,
                bonds: ligBonds,
                isPrepared: true,
                conformerCount: 1,
                sourceChainID: chainID
            )
            ligandDB.add(dbEntry)
            activeLigandEntryID = dbEntry.id
        }

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

    /// Remove a specific chain from the protein molecule.
    func removeChain(chainID: String) {
        guard let prot = molecules.protein else { return }
        guard let chain = prot.chains.first(where: { $0.id == chainID }) else { return }

        let chainAtomIndices = Set(chain.residueIndices.flatMap { resIdx -> [Int] in
            guard resIdx < prot.residues.count else { return [] }
            return prot.residues[resIdx].atomIndices
        })
        guard !chainAtomIndices.isEmpty else { return }

        // If the active ligand was extracted from this chain, clear it
        if let lig = molecules.ligand {
            let firstRes = chain.residueIndices.first.flatMap { idx in
                idx < prot.residues.count ? prot.residues[idx] : nil
            }
            let chainLigandName = firstRes?.name ?? "Chain_\(chainID)"
            if lig.name == chainLigandName {
                molecules.ligand = nil
                docking.originalDockingLigand = nil
                docking.currentInteractions = []
                renderer?.updateInteractionLines([])
                renderer?.clearGhostPose()
            }
        }

        // Also remove from ligand database if it was added there
        let firstRes = chain.residueIndices.first.flatMap { idx in
            idx < prot.residues.count ? prot.residues[idx] : nil
        }
        let chainLigandName = firstRes?.name ?? "Chain_\(chainID)"
        if let dbIdx = ligandDB.entries.firstIndex(where: {
            $0.sourceChainID == chainID || $0.name == chainLigandName
        }) {
            ligandDB.entries.remove(at: dbIdx)
            log.info("Also removed \(chainLigandName) from ligand database", category: .molecule)
        }

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
        activeLigandEntryID = nil
        docking.originalDockingLigand = nil
        docking.dockingResults = []
        docking.currentInteractions = []
        docking.selectedPoseIndices = []
        renderer?.updateInteractionLines([])
        renderer?.clearGhostPose()
        pushToRenderer()
        log.info("Cleared \(name)", category: .molecule)
        workspace.statusMessage = "No ligand loaded"
    }

    /// Remove ligand from the 3D view and from the ligand database.
    /// Also clears any ghost poses and interaction lines.
    func removeLigandFromView() {
        let name = molecules.ligand?.name ?? "ligand"

        // Remove from database using the tracked entry ID (robust),
        // falling back to name match for legacy callers.
        if let entryID = activeLigandEntryID {
            ligandDB.removeWithChildren(id: entryID)
        } else if let entry = ligandDB.entries.first(where: { $0.name == name }) {
            ligandDB.removeWithChildren(id: entry.id)
        }

        molecules.ligand = nil
        activeLigandEntryID = nil
        docking.currentInteractions = []
        docking.selectedPoseIndices = []
        renderer?.updateInteractionLines([])
        renderer?.clearGhostPose()
        pushToRenderer()
        log.info("Removed \(name) from view and database", category: .molecule)
        workspace.statusMessage = "Ligand removed"
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
        // During docking, exclude ligand from camera fitting positions
        // (the ghost pose handles visualization; original ligand coords are stale)
        if workspace.showLigand, !docking.isDocking, let lig = molecules.ligand {
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
                // Map by array position (bond.atomIndex is an array position)
                var posToNew: [Int: Int] = [:]
                var remapped: [Atom] = []
                for (origIdx, atom) in prot.atoms.enumerated() {
                    if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                    posToNew[origIdx] = remapped.count
                    remapped.append(Atom(
                        id: remapped.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                var remappedBonds: [Bond] = []
                for bond in prot.bonds {
                    if let a = posToNew[bond.atomIndex1], let b = posToNew[bond.atomIndex2] {
                        remappedBonds.append(Bond(id: remappedBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
                visibleProt = Molecule(name: prot.name, atoms: remapped, bonds: remappedBonds, title: prot.title)
                visibleProt.secondaryStructureAssignments = prot.secondaryStructureAssignments
            }

            // Build chain color map early so ribbon generator can use it
            var ribbonChainColors: [String: SIMD3<Float>] = [:]
            if workspace.colorScheme == .chainColored {
                let palette = WorkspaceState.MoleculeColorScheme.chainPalette
                for (i, chain) in prot.chains.enumerated() {
                    ribbonChainColors[chain.id] = palette[i % palette.count]
                }
            }
            let selectedResidueKeys = Set(
                workspace.selectedResidueIndices.compactMap { resIdx -> String? in
                    guard resIdx < prot.residues.count else { return nil }
                    let residue = prot.residues[resIdx]
                    return "\(residue.chainID)|\(residue.sequenceNumber)"
                }
            )
            let (ribbonVerts, ribbonIdxs) = RibbonMeshGenerator.generateForMolecule(
                visibleProt,
                chainColorMap: ribbonChainColors,
                selectedResidueKeys: selectedResidueKeys
            )
            renderer.updateRibbonMesh(vertices: ribbonVerts, indices: ribbonIdxs)

            var caControlPoints: [(position: SIMD3<Float>, atomID: Int)] = []
            for atom in prot.atoms {
                if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                if atom.name.trimmingCharacters(in: .whitespaces) == "CA" {
                    caControlPoints.append((position: atom.position, atomID: atom.id))
                }
            }
            renderer.updateRibbonCAControlPoints(caControlPoints)

            // Non-protein chains (nucleic acid, ions, etc.) as ball-and-stick overlay
            // so they remain visible in ribbon mode
            if workspace.showProtein {
                let proteinChainIDs = Set(prot.chains.filter { $0.type == .protein }.map(\.id))
                var npPosToNew: [Int: Int] = [:]
                for (origIdx, atom) in prot.atoms.enumerated() {
                    if proteinChainIDs.contains(atom.chainID) { continue } // skip protein chains (have ribbon)
                    if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                    if !workspace.showHydrogens && atom.element == .H { continue }
                    npPosToNew[origIdx] = allAtoms.count
                    allAtoms.append(Atom(
                        id: allAtoms.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                for bond in prot.bonds {
                    if let a = npPosToNew[bond.atomIndex1], let b = npPosToNew[bond.atomIndex2] {
                        allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
            }

            // Side chain display — include CA atom to visually connect to ribbon backbone
            var selectedRibbonResidueAtomIDs: Set<Int> = []
            if workspace.sideChainDisplay != .none, workspace.showProtein {
                let sideChainResidueIndices = sideChainResidueSet(protein: prot)
                if !sideChainResidueIndices.isEmpty {
                    // Backbone atoms to exclude (but keep CA as the anchor connecting
                    // side chains to the ribbon backbone)
                    let backboneExcludeNames: Set<String> = ["N", "C", "O", "OXT", "H", "HA"]

                    // Collect positions of protein atoms involved in interactions so
                    // backbone atoms at interaction endpoints are rendered as spheres
                    // (otherwise H-bond lines to backbone N/O point to empty space)
                    struct PosKey: Hashable {
                        let x, y, z: Int32
                        init(_ p: SIMD3<Float>) {
                            x = Int32((p.x * 100).rounded())
                            y = Int32((p.y * 100).rounded())
                            z = Int32((p.z * 100).rounded())
                        }
                    }
                    var interactingPositions = Set<PosKey>()
                    // For atom-based interactions, the proteinPosition IS an atom position.
                    // For pi-type interactions, proteinPosition is a ring centroid (virtual
                    // point) that won't match any real atom. For those, resolve the actual
                    // protein residue atoms so they get included in the sidechain display.
                    let piInteractionTypes: Set<MolecularInteraction.InteractionType> = [
                        .piStack, .piCation, .chPi, .amideStack
                    ]
                    // Build heavy-atom index → original atom index mapping once
                    var heavyToOriginal: [Int: Int] = [:]
                    var heavyIdx = 0
                    for (origIdx, atom) in prot.atoms.enumerated() {
                        guard atom.element != .H else { continue }
                        heavyToOriginal[heavyIdx] = origIdx
                        heavyIdx += 1
                    }
                    for ixn in docking.currentInteractions {
                        if piInteractionTypes.contains(ixn.type) {
                            // Add all ring atom positions from the residue so the
                            // sidechain around the centroid is fully rendered
                            if let origIdx = heavyToOriginal[ixn.proteinAtomIndex],
                               origIdx < prot.atoms.count {
                                let resSeq = prot.atoms[origIdx].residueSeq
                                let chain = prot.atoms[origIdx].chainID
                                for res in prot.residues where res.sequenceNumber == resSeq && res.chainID == chain {
                                    for ai in res.atomIndices where ai < prot.atoms.count {
                                        interactingPositions.insert(PosKey(prot.atoms[ai].position))
                                    }
                                }
                            }
                        } else {
                            interactingPositions.insert(PosKey(ixn.proteinPosition))
                        }
                    }

                    // First pass: collect all side chain atom indices including CA
                    var scPosToNew: [Int: Int] = [:]
                    for residueIdx in sideChainResidueIndices {
                        guard residueIdx < prot.residues.count else { continue }
                        let residue = prot.residues[residueIdx]
                        for atomIdx in residue.atomIndices {
                            guard atomIdx < prot.atoms.count else { continue }
                            let atom = prot.atoms[atomIdx]
                            if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                            if !workspace.showHydrogens && atom.element == .H { continue }
                            let trimmedName = atom.name.trimmingCharacters(in: .whitespaces)
                            // Keep CA (alpha carbon) — it's the attachment point to the ribbon
                            // Also keep backbone atoms involved in interactions (e.g. H-bonds to N/O)
                            if backboneExcludeNames.contains(trimmedName)
                                && !interactingPositions.contains(PosKey(atom.position)) { continue }
                            scPosToNew[atomIdx] = allAtoms.count
                            if workspace.selectedResidueIndices.contains(residueIdx) {
                                selectedRibbonResidueAtomIDs.insert(allAtoms.count)
                            }
                            allAtoms.append(Atom(
                                id: allAtoms.count, element: atom.element, position: atom.position,
                                name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                                chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                                isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                            ))
                        }
                    }
                    // Also add protein atoms from interactions whose residues aren't in
                    // the sidechain set (e.g. contacts just outside the 5 Å cutoff)
                    if !interactingPositions.isEmpty {
                        for (atomIdx, atom) in prot.atoms.enumerated() {
                            guard scPosToNew[atomIdx] == nil else { continue }
                            if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                            if atom.element == .H { continue }
                            if interactingPositions.contains(PosKey(atom.position)) {
                                scPosToNew[atomIdx] = allAtoms.count
                                allAtoms.append(Atom(
                                    id: allAtoms.count, element: atom.element, position: atom.position,
                                    name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                                    chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                                    isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                                ))
                            }
                        }
                    }

                    // Second pass: remap bonds where at least both endpoints are in the side chain set
                    for bond in prot.bonds {
                        if let a = scPosToNew[bond.atomIndex1], let b = scPosToNew[bond.atomIndex2] {
                            allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                        }
                    }
                }
            }

            let ribbonProteinAtomCount = allAtoms.count

            if workspace.showLigand, !docking.isDocking, let lig = molecules.ligand {
                // Map by array position (bond.atomIndex is an array position, not atom.id)
                var ligPosToNew: [Int: Int] = [:]
                for (origIdx, atom) in lig.atoms.enumerated() {
                    if !workspace.showHydrogens && atom.element == .H { continue }
                    ligPosToNew[origIdx] = allAtoms.count
                    allAtoms.append(Atom(
                        id: allAtoms.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                for bond in lig.bonds {
                    if let a = ligPosToNew[bond.atomIndex1], let b = ligPosToNew[bond.atomIndex2] {
                        allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
            }

            renderer.selectedAtomIndex = workspace.selectedAtomIndex ?? -1
            renderer.selectedResidueAtomIndices = selectedRibbonResidueAtomIDs
            renderer.renderMode = .ballAndStick
            renderer.ligandRenderMode = workspace.effectiveLigandRenderMode
            renderer.proteinAtomCount = ribbonProteinAtomCount
            renderer.ligandVisible = workspace.showLigand && !docking.isDocking && molecules.ligand != nil
            renderer.updateMoleculeData(atoms: allAtoms, bonds: allBonds)
            return
        }

        // Non-ribbon mode
        renderer.clearRibbonMesh()

        if workspace.showProtein, let prot = molecules.protein {
            // Map by array position (bond.atomIndex is an array position, not atom.id)
            var protPosToNew: [Int: Int] = [:]
            for (origIdx, atom) in prot.atoms.enumerated() {
                if !workspace.showHydrogens && atom.element == .H { continue }
                if workspace.hiddenChainIDs.contains(atom.chainID) { continue }
                if workspace.hiddenAtomIndices.contains(origIdx) { continue }
                protPosToNew[origIdx] = allAtoms.count
                allAtoms.append(Atom(
                    id: allAtoms.count, element: atom.element, position: atom.position,
                    name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                    chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                ))
            }
            for bond in prot.bonds {
                if let a = protPosToNew[bond.atomIndex1], let b = protPosToNew[bond.atomIndex2] {
                    allBonds.append(Bond(id: allBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
        }

        if workspace.showLigand, !docking.isDocking, let lig = molecules.ligand {
            let ligOffset = molecules.protein?.atoms.count ?? 0
            // Map by array position (bond.atomIndex is an array position, not atom.id)
            var ligPosToNew: [Int: Int] = [:]
            for (origIdx, atom) in lig.atoms.enumerated() {
                if !workspace.showHydrogens && atom.element == .H { continue }
                if workspace.hiddenAtomIndices.contains(origIdx + ligOffset) { continue }
                ligPosToNew[origIdx] = allAtoms.count
                allAtoms.append(Atom(
                    id: allAtoms.count, element: atom.element, position: atom.position,
                    name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                    chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                ))
            }
            for bond in lig.bonds {
                if let a = ligPosToNew[bond.atomIndex1], let b = ligPosToNew[bond.atomIndex2] {
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
        renderer.ligandRenderMode = workspace.effectiveLigandRenderMode

        renderer.enableClipping = workspace.enableClipping
        renderer.clipNearZ = workspace.clipNearZ
        renderer.clipFarZ = workspace.clipFarZ
        renderer.slabHalfThickness = workspace.slabThickness / 2.0
        renderer.slabOffset = workspace.slabOffset

        // Protein uniform color override (e.g., ligand focus mode)
        renderer.uniformProteinColor = workspace.colorScheme.proteinColor

        // Build chain color map respecting per-chain color modes (CPK / Chain / Custom)
        if let prot = molecules.protein {
            var colorMap: [String: SIMD3<Float>] = [:]
            var carbonsOnly: Set<String> = []
            let palette = WorkspaceState.MoleculeColorScheme.chainPalette
            // Default per-chain mode depends on the global color scheme:
            // - .chainColored → chains get palette colors by default
            // - .element / .ligandFocus → chains get element (CPK) colors by default
            let defaultMode: WorkspaceState.ChainColorMode = workspace.colorScheme == .chainColored ? .chainDefault : .cpk
            for (i, chain) in prot.chains.enumerated() {
                let mode = workspace.chainColorModes[chain.id] ?? defaultMode
                switch mode {
                case .cpk:
                    break // omit from map → renderer falls through to element colors
                case .chainDefault:
                    colorMap[chain.id] = palette[i % palette.count]
                case .custom:
                    if let custom = workspace.chainColorOverrides[chain.id] {
                        colorMap[chain.id] = custom
                    } else {
                        colorMap[chain.id] = palette[i % palette.count]
                    }
                }
                if colorMap[chain.id] != nil {
                    let scope = workspace.chainColorScopes[chain.id] ?? .carbonsOnly
                    if scope == .carbonsOnly { carbonsOnly.insert(chain.id) }
                }
            }
            renderer.chainColorMap = colorMap
            renderer.chainColorCarbonsOnly = carbonsOnly
        } else {
            renderer.chainColorMap = [:]
            renderer.chainColorCarbonsOnly = []
        }

        // Ligand carbon color: user override from inspector, otherwise from color scheme.
        // Scope (carbons-only vs all atoms) is user-controlled via the inspector.
        if let userLigColor = workspace.ligandCarbonColor {
            renderer.ligandCarbonColor = userLigColor
        } else {
            renderer.ligandCarbonColor = workspace.colorScheme.ligandCarbon
        }
        renderer.ligandColorCarbonsOnly = workspace.ligandColorCarbonsOnly

        // Track how many protein atoms are in the buffer so Renderer can distinguish them
        let protCount = workspace.showProtein ? (molecules.protein?.atoms.filter { atom in
            if !workspace.showHydrogens && atom.element == .H { return false }
            if workspace.hiddenChainIDs.contains(atom.chainID) { return false }
            return true
        }.count ?? 0) : 0
        renderer.proteinAtomCount = protCount

        // Track ligand visibility so renderer can hide interaction lines/grid when ligand is hidden
        renderer.ligandVisible = workspace.showLigand && !docking.isDocking && molecules.ligand != nil

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
        // Map by array position (bond.atomIndex is an array position, not atom.id)
        var posToNew: [Int: Int] = [:]
        var newIdx = 0
        for (origIdx, atom) in molecule.atoms.enumerated() where atom.element != .H {
            posToNew[origIdx] = newIdx
            newIdx += 1
        }
        return molecule.bonds.compactMap { bond in
            guard let a = posToNew[bond.atomIndex1], let b = posToNew[bond.atomIndex2] else { return nil }
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

    @objc func selectNearbyAction(_ sender: NSMenuItem) {
        guard let distance = sender.representedObject as? Float else { return }
        viewModel?.selectResiduesWithinDistance(distance)
    }

    @objc func extendByOneResidueAction() {
        viewModel?.extendSelectionByOneResidue()
    }

    @objc func shrinkByOneResidueAction() {
        viewModel?.shrinkSelectionByOneResidue()
    }

    @objc func invertSelectionAction() {
        viewModel?.invertSelection()
    }

    @objc func createSubsetAction() {
        viewModel?.createSubsetFromSelection()
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

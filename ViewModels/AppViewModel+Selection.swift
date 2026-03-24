import SwiftUI

// MARK: - Selection, Visibility, Context Menu, Subsets, Sequence Editing, Surface

extension AppViewModel {

    // MARK: - Selection

    func selectAtom(_ atomIndex: Int?, toggle: Bool = false) {
        if !toggle {
            workspace.selectedResidueIndices.removeAll()
        }

        workspace.selectedAtomIndex = atomIndex

        if let idx = atomIndex {
            if let prot = molecules.protein, let resIdx = prot.residueIndex(forAtom: idx) {
                if toggle && workspace.selectedResidueIndices.contains(resIdx) {
                    workspace.selectedResidueIndices.remove(resIdx)
                } else {
                    workspace.selectedResidueIndices.insert(resIdx)
                }
            }
            if let lig = molecules.ligand {
                let adjustedIdx = idx - (molecules.protein?.atoms.count ?? 0)
                if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                    if toggle && workspace.selectedResidueIndices.contains(resIdx) {
                        workspace.selectedResidueIndices.remove(resIdx)
                    } else {
                        workspace.selectedResidueIndices.insert(resIdx)
                    }
                }
            }
            log.debug("[Selection] Atom \(idx) selected (toggle=\(toggle), \(workspace.selectedResidueIndices.count) residues)", category: .selection)
        } else {
            log.debug("[Selection] Selection cleared via nil atom", category: .selection)
        }

        pushToRenderer()
    }

    func deselectAll() {
        let atomCount = workspace.selectedAtomIndices.count
        let resCount = workspace.selectedResidueIndices.count
        workspace.selectedAtomIndex = nil
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedResidueIndices.removeAll()
        pushToRenderer()
        if atomCount > 0 || resCount > 0 {
            log.debug("[Selection] Deselected all (\(atomCount) atoms, \(resCount) residues cleared)", category: .selection)
        }
    }

    func focusOnAtom(_ atomIndex: Int) {
        let allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])
        guard atomIndex < allAtoms.count else {
            log.debug("[Selection] focusOnAtom skipped: index \(atomIndex) out of range (\(allAtoms.count) atoms)", category: .selection)
            return
        }
        renderer?.camera.focusOnPoint(allAtoms[atomIndex].position)
    }

    // MARK: - Visibility

    func toggleChainVisibility(_ chainID: String) {
        if workspace.hiddenChainIDs.contains(chainID) {
            workspace.hiddenChainIDs.remove(chainID)
        } else {
            workspace.hiddenChainIDs.insert(chainID)
        }
        pushToRenderer()
    }

    func setRenderMode(_ mode: RenderMode) {
        workspace.renderMode = mode
        pushToRenderer()
    }

    func toggleHydrogens() {
        workspace.showHydrogens.toggle()
        pushToRenderer()
    }

    func toggleLighting() {
        workspace.useDirectionalLighting.toggle()
        renderer?.lightingMode = workspace.useDirectionalLighting ? 1 : 0
    }

    // MARK: - Box Selection

    func handleBoxSelection(atomIndices: [Int], addToExisting: Bool) {
        if !addToExisting {
            workspace.selectedAtomIndices.removeAll()
            workspace.selectedResidueIndices.removeAll()
        }
        workspace.selectedAtomIndices.formUnion(atomIndices)

        for idx in atomIndices {
            if let prot = molecules.protein, let resIdx = prot.residueIndex(forAtom: idx) {
                workspace.selectedResidueIndices.insert(resIdx)
            }
            if let lig = molecules.ligand {
                let adjustedIdx = idx - (molecules.protein?.atoms.count ?? 0)
                if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                    workspace.selectedResidueIndices.insert(resIdx)
                }
            }
        }

        workspace.selectedAtomIndex = atomIndices.first
        pushToRenderer()

        let count = workspace.selectedAtomIndices.count
        log.info("Selected \(count) atom\(count == 1 ? "" : "s") by box selection", category: .molecule)
        workspace.statusMessage = "\(count) atom\(count == 1 ? "" : "s") selected"
    }

    // MARK: - Molecular Surface

    func toggleSurface() {
        workspace.showSurface.toggle()
        if workspace.showSurface {
            generateSurface()
        } else {
            renderer?.clearSurfaceMesh()
        }
    }

    func setSurfaceColorMode(_ mode: SurfaceColorMode) {
        workspace.surfaceColorMode = mode
        if workspace.showSurface {
            generateSurface()
        }
    }

    func generateSurface() {
        guard let prot = molecules.protein, let renderer else {
            log.debug("[Surface] generateSurface skipped: no protein or renderer", category: .selection)
            return
        }
        workspace.isGeneratingSurface = true
        workspace.statusMessage = "Generating surface..."

        Task {
            if workspace.surfaceGenerator == nil {
                workspace.surfaceGenerator = try? SurfaceGenerator(
                    device: renderer.device,
                    commandQueue: renderer.commandQueue
                )
            }
            guard let gen = workspace.surfaceGenerator else {
                workspace.isGeneratingSurface = false
                log.error("Failed to create surface generator", category: .molecule)
                return
            }

            gen.fieldType = workspace.surfaceType
            gen.colorMode = workspace.surfaceColorMode

            let atoms = prot.atoms.filter { $0.element != .H }
            if let result = gen.generateSurface(atoms: atoms) {
                renderer.updateSurfaceMesh(result)
                log.success("Surface: \(result.vertexCount) vertices, \(result.indexCount / 3) triangles", category: .molecule)
                workspace.statusMessage = "Surface generated"
            } else {
                log.error("Surface generation failed", category: .molecule)
                workspace.statusMessage = "Surface failed"
            }
            workspace.isGeneratingSurface = false
        }
    }

    // MARK: - Context Menu Actions

    func selectChain(_ chainID: String) {
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedResidueIndices.removeAll()

        if let prot = molecules.protein {
            let indices = prot.atomIndices(forChainID: chainID)
            workspace.selectedAtomIndices.formUnion(indices)
            if let chain = prot.chains.first(where: { $0.id == chainID }) {
                workspace.selectedResidueIndices.formUnion(chain.residueIndices)
            }
        }
        if let lig = molecules.ligand {
            let offset = molecules.protein?.atoms.count ?? 0
            let indices = lig.atomIndices(forChainID: chainID).map { $0 + offset }
            workspace.selectedAtomIndices.formUnion(indices)
        }

        workspace.selectedAtomIndex = workspace.selectedAtomIndices.first
        pushToRenderer()
        log.info("Selected chain \(chainID) (\(workspace.selectedAtomIndices.count) atoms)", category: .molecule)
        workspace.statusMessage = "Chain \(chainID) selected"
    }

    func selectResiduesWithinDistance(_ distance: Float) {
        guard let prot = molecules.protein else {
            log.debug("[Selection] selectResiduesWithinDistance skipped: no protein loaded", category: .selection)
            return
        }
        let protAtoms = prot.atoms
        let combinedAtoms: [Atom] = protAtoms + (molecules.ligand?.atoms ?? [])

        var selectedPositions: [SIMD3<Float>] = []
        for idx in workspace.selectedAtomIndices {
            if idx < combinedAtoms.count {
                selectedPositions.append(combinedAtoms[idx].position)
            }
        }
        for resIdx in workspace.selectedResidueIndices {
            guard resIdx < prot.residues.count else { continue }
            for atomIdx in prot.residues[resIdx].atomIndices {
                if atomIdx < protAtoms.count {
                    selectedPositions.append(protAtoms[atomIdx].position)
                }
            }
        }
        guard !selectedPositions.isEmpty else {
            log.debug("[Selection] selectResiduesWithinDistance skipped: no selected positions", category: .selection)
            return
        }

        let distSq = distance * distance
        var newResidues: Set<Int> = workspace.selectedResidueIndices

        for (resIdx, residue) in prot.residues.enumerated() {
            if newResidues.contains(resIdx) { continue }
            outer: for atomIdx in residue.atomIndices {
                guard atomIdx < protAtoms.count else { continue }
                let pos = protAtoms[atomIdx].position
                for sp in selectedPositions {
                    if simd_distance_squared(pos, sp) <= distSq {
                        newResidues.insert(resIdx)
                        break outer
                    }
                }
            }
        }

        let addedCount = newResidues.count - workspace.selectedResidueIndices.count
        workspace.selectedResidueIndices = newResidues

        for resIdx in newResidues {
            guard resIdx < prot.residues.count else { continue }
            workspace.selectedAtomIndices.formUnion(prot.residues[resIdx].atomIndices)
        }

        pushToRenderer()
        log.info("Extended selection by \(addedCount) residues within \(String(format: "%.1f", distance)) \u{00C5}", category: .molecule)
        workspace.statusMessage = "\(workspace.selectedResidueIndices.count) residues selected"
    }

    func definePocketFromSelection() {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else {
            log.warn("Select residues first to define a pocket", category: .dock)
            return
        }
        let pocket = BindingSiteDetector.pocketFromResidues(
            protein: prot, residueIndices: Array(workspace.selectedResidueIndices)
        )
        docking.detectedPockets = [pocket]
        docking.selectedPocket = pocket
        showGridBoxForPocket(pocket)
        log.success("Pocket defined from \(workspace.selectedResidueIndices.count) selected residues", category: .dock)
        workspace.statusMessage = "Pocket defined"
    }

    func hideSelection() {
        guard !workspace.selectedAtomIndices.isEmpty else { return }
        workspace.hiddenAtomIndices.formUnion(workspace.selectedAtomIndices)
        let count = workspace.selectedAtomIndices.count
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndex = nil
        pushToRenderer()
        log.info("Hidden \(count) atoms", category: .molecule)
        workspace.statusMessage = "\(count) atoms hidden"
    }

    func showAllAtoms() {
        let count = workspace.hiddenAtomIndices.count
        workspace.hiddenAtomIndices.removeAll()
        pushToRenderer()
        if count > 0 {
            log.info("Showing all atoms (\(count) were hidden)", category: .molecule)
        }
        workspace.statusMessage = "All atoms visible"
    }

    func removeSelection() {
        guard !workspace.selectedAtomIndices.isEmpty else { return }
        let removedCount = workspace.selectedAtomIndices.count

        if let prot = molecules.protein {
            let keepAtoms = prot.atoms.filter { !workspace.selectedAtomIndices.contains($0.id) }
            if keepAtoms.count < prot.atoms.count {
                var newAtoms: [Atom] = []
                var indexMap: [Int: Int] = [:]
                for atom in keepAtoms {
                    indexMap[atom.id] = newAtoms.count
                    newAtoms.append(Atom(
                        id: newAtoms.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom, occupancy: atom.occupancy, tempFactor: atom.tempFactor
                    ))
                }
                var newBonds: [Bond] = []
                for bond in prot.bonds {
                    if let a = indexMap[bond.atomIndex1], let b = indexMap[bond.atomIndex2] {
                        newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
                molecules.protein = Molecule(name: prot.name, atoms: newAtoms, bonds: newBonds, title: prot.title)
                molecules.protein?.secondaryStructureAssignments = prot.secondaryStructureAssignments
            }
        }

        if let lig = molecules.ligand {
            let offset = molecules.protein?.atoms.count ?? 0
            let ligRemoveIDs = Set(workspace.selectedAtomIndices.compactMap { idx -> Int? in
                let adj = idx - offset
                return adj >= 0 && adj < lig.atoms.count ? lig.atoms[adj].id : nil
            })
            if !ligRemoveIDs.isEmpty {
                let keepAtoms = lig.atoms.filter { !ligRemoveIDs.contains($0.id) }
                var newAtoms: [Atom] = []
                var indexMap: [Int: Int] = [:]
                for atom in keepAtoms {
                    indexMap[atom.id] = newAtoms.count
                    newAtoms.append(Atom(
                        id: newAtoms.count, element: atom.element, position: atom.position,
                        name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                        chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                        isHetAtom: atom.isHetAtom
                    ))
                }
                var newBonds: [Bond] = []
                for bond in lig.bonds {
                    if let a = indexMap[bond.atomIndex1], let b = indexMap[bond.atomIndex2] {
                        newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                    }
                }
                molecules.ligand = Molecule(name: lig.name, atoms: newAtoms, bonds: newBonds, title: lig.title)
            }
        }

        workspace.selectedAtomIndices.removeAll()
        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndex = nil
        pushToRenderer()
        log.success("Removed \(removedCount) atoms", category: .molecule)
        workspace.statusMessage = "\(removedCount) atoms removed"
    }

    func centerOnSelection() {
        let allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])
        let selectedPositions = workspace.selectedAtomIndices.compactMap { idx -> SIMD3<Float>? in
            idx < allAtoms.count ? allAtoms[idx].position : nil
        }
        guard !selectedPositions.isEmpty else {
            log.debug("[Selection] centerOnSelection skipped: no selected positions", category: .selection)
            return
        }
        let c = centroid(selectedPositions)
        renderer?.camera.focusOnPoint(c)
        log.info("Centered on selection", category: .molecule)
    }

    // MARK: - Residue Subsets (MOE-style)

    func createSubsetFromSelection(name: String? = nil) {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else {
            log.error("Select residues first to create a subset", category: .molecule)
            return
        }
        let indices = Array(workspace.selectedResidueIndices).sorted()
        let subsetName = name ?? "Subset \(workspace.residueSubsets.count + 1)"
        let colors: [SIMD4<Float>] = [
            SIMD4(0.2, 0.8, 0.9, 1.0),
            SIMD4(0.9, 0.5, 0.2, 1.0),
            SIMD4(0.5, 0.9, 0.3, 1.0),
            SIMD4(0.9, 0.3, 0.7, 1.0),
            SIMD4(0.6, 0.4, 0.9, 1.0),
        ]
        let color = colors[workspace.residueSubsets.count % colors.count]
        let subset = ResidueSubset(name: subsetName, residueIndices: indices, color: color)
        workspace.residueSubsets.append(subset)

        let residueNames = indices.prefix(5).map { idx -> String in
            guard idx < prot.residues.count else { return "?" }
            return "\(prot.residues[idx].name)\(prot.residues[idx].sequenceNumber)"
        }.joined(separator: ", ")
        let suffix = indices.count > 5 ? "... (\(indices.count) residues)" : ""
        log.success("Created subset '\(subsetName)': \(residueNames)\(suffix)", category: .molecule)
    }

    func deleteSubset(id: UUID) {
        workspace.residueSubsets.removeAll { $0.id == id }
    }

    func toggleSubsetVisibility(id: UUID) {
        if let idx = workspace.residueSubsets.firstIndex(where: { $0.id == id }) {
            workspace.residueSubsets[idx].isVisible.toggle()
            pushToRenderer()
        }
    }

    func selectSubset(id: UUID) {
        guard let prot = molecules.protein,
              let subset = workspace.residueSubsets.first(where: { $0.id == id })
        else {
            log.debug("[Selection] selectSubset skipped: no protein or subset not found", category: .selection)
            return
        }
        workspace.selectedResidueIndices = Set(subset.residueIndices)
        workspace.selectedAtomIndices = Set(subset.atomIndices(in: prot))
        log.debug("[Selection] Selected subset '\(subset.name)': \(workspace.selectedResidueIndices.count) residues, \(workspace.selectedAtomIndices.count) atoms", category: .selection)
        pushToRenderer()
    }

    func definePocketFromSubset(id: UUID) {
        guard let prot = molecules.protein,
              let subset = workspace.residueSubsets.first(where: { $0.id == id })
        else {
            log.debug("[Selection] definePocketFromSubset skipped: no protein or subset not found", category: .selection)
            return
        }
        Task {
            let pocket = BindingSiteDetector.pocketFromResidues(
                protein: prot, residueIndices: subset.residueIndices
            )
            docking.selectedPocket = pocket
            docking.detectedPockets = [pocket]
            log.success("Defined pocket from subset '\(subset.name)': \(String(format: "%.0f", pocket.volume)) Å³", category: .dock)
        }
    }

    // MARK: - Sequence Editing

    func deleteSelectedResidues() {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else {
            log.warn("Select residues first", category: .molecule)
            return
        }
        let count = workspace.selectedResidueIndices.count
        prot.removeResidues(at: workspace.selectedResidueIndices)

        workspace.residueSubsets.removeAll()
        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedAtomIndex = nil
        workspace.hiddenAtomIndices.removeAll()
        pushToRenderer()
        log.success("Deleted \(count) residues (\(prot.atoms.count) atoms remaining)", category: .molecule)
        workspace.statusMessage = "\(count) residues deleted"
    }

    func renameChain(from oldID: String, to newID: String) {
        guard let prot = molecules.protein else { return }
        prot.renameChain(from: oldID, to: newID)

        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedAtomIndex = nil
        workspace.residueSubsets.removeAll()
        pushToRenderer()
        log.success("Renamed chain \(oldID) → \(newID)", category: .molecule)
        workspace.statusMessage = "Chain \(oldID) renamed to \(newID)"
    }

    func mergeChains(from sourceID: String, into targetID: String) {
        guard let prot = molecules.protein else { return }
        prot.mergeChains(from: sourceID, into: targetID)

        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndices.removeAll()
        workspace.selectedAtomIndex = nil
        workspace.residueSubsets.removeAll()
        pushToRenderer()
        log.success("Merged chain \(sourceID) into \(targetID)", category: .molecule)
        workspace.statusMessage = "Chain \(sourceID) merged into \(targetID)"
    }

    func copySequenceToClipboard(chainID: String? = nil) {
        guard let prot = molecules.protein else { return }
        let threeToOne: [String: String] = [
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        ]

        let indices: [Int]
        if !workspace.selectedResidueIndices.isEmpty {
            indices = workspace.selectedResidueIndices.sorted()
        } else if let cid = chainID, let chain = prot.chains.first(where: { $0.id == cid }) {
            indices = chain.residueIndices
        } else {
            indices = Array(0..<prot.residues.count)
        }

        let seq = indices.compactMap { idx -> String? in
            guard idx < prot.residues.count, prot.residues[idx].isStandard else { return nil }
            return threeToOne[prot.residues[idx].name] ?? "X"
        }.joined()

        guard !seq.isEmpty else { return }

        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(seq, forType: .string)
        log.info("Copied \(seq.count) residue sequence to clipboard", category: .molecule)
        workspace.statusMessage = "\(seq.count)-residue sequence copied"
    }

    func selectBySecondaryStructure(_ ssType: SecondaryStructure, chainID: String? = nil) {
        guard let prot = molecules.protein else { return }
        workspace.selectedResidueIndices.removeAll()
        workspace.selectedAtomIndices.removeAll()

        for (idx, res) in prot.residues.enumerated() {
            guard res.isStandard else { continue }
            if let cid = chainID, res.chainID != cid { continue }

            var ss: SecondaryStructure = .coil
            for ssa in prot.secondaryStructureAssignments {
                if ssa.chain == res.chainID && res.sequenceNumber >= ssa.start && res.sequenceNumber <= ssa.end {
                    ss = ssa.type
                    break
                }
            }
            if ss == ssType {
                workspace.selectedResidueIndices.insert(idx)
                workspace.selectedAtomIndices.formUnion(res.atomIndices)
            }
        }
        workspace.selectedAtomIndex = workspace.selectedAtomIndices.first
        pushToRenderer()
        let ssName = switch ssType { case .helix: "helices"; case .sheet: "sheets"; case .coil: "coils"; case .turn: "turns" }
        workspace.statusMessage = "\(workspace.selectedResidueIndices.count) \(ssName) selected"
    }

    func resetView() {
        renderer?.camera.reset()
        pushToRenderer()
        renderer?.fitToContent()
        log.info("View reset", category: .molecule)
        workspace.statusMessage = "View reset"
    }

    func fitToView() {
        guard let renderer else { return }
        let allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])

        if !workspace.selectedAtomIndices.isEmpty {
            let selectedPositions = workspace.selectedAtomIndices.compactMap { idx -> SIMD3<Float>? in
                idx < allAtoms.count ? allAtoms[idx].position : nil
            }
            if !selectedPositions.isEmpty {
                renderer.fitToPositions(selectedPositions)
                log.info("Fit to selection (\(selectedPositions.count) atoms)", category: .molecule)
                return
            }
        }

        renderer.fitToContent()
    }

    func fitToLigand() {
        guard let renderer, let lig = molecules.ligand, !lig.atoms.isEmpty else { return }
        renderer.fitToPositions(lig.atoms.map(\.position))
    }

    // MARK: - Context Menu Builder

    func showContextMenu(event: NSEvent, view: NSView) {
        let menu = NSMenu(title: "Context")

        let chainMenu = NSMenu(title: "Chains")
        let chainItem = NSMenuItem(title: "Select Chain", action: nil, keyEquivalent: "")
        chainItem.submenu = chainMenu
        for chain in allChains {
            let item = NSMenuItem(title: "Chain \(chain.id)", action: #selector(ContextMenuTarget.selectChainAction(_:)), keyEquivalent: "")
            item.representedObject = chain.id
            item.target = contextMenuTarget
            chainMenu.addItem(item)
        }
        if !allChains.isEmpty {
            menu.addItem(chainItem)
        }

        let extendItem = NSMenuItem(title: "Select Residues Within 5 \u{00C5}", action: #selector(ContextMenuTarget.selectNearbyResiduesAction), keyEquivalent: "")
        extendItem.target = contextMenuTarget
        extendItem.isEnabled = !workspace.selectedAtomIndices.isEmpty
        menu.addItem(extendItem)

        let pocketItem = NSMenuItem(title: "Define Pocket from Selection", action: #selector(ContextMenuTarget.definePocketAction), keyEquivalent: "")
        pocketItem.target = contextMenuTarget
        pocketItem.isEnabled = !workspace.selectedResidueIndices.isEmpty
        menu.addItem(pocketItem)

        let constraintItem = NSMenuItem(title: "Add Docking Constraint\u{2026}", action: #selector(ContextMenuTarget.addDockingConstraintAction), keyEquivalent: "")
        constraintItem.target = contextMenuTarget
        constraintItem.isEnabled = workspace.selectedAtomIndex != nil || !workspace.selectedResidueIndices.isEmpty
        menu.addItem(constraintItem)

        menu.addItem(NSMenuItem.separator())

        let hideItem = NSMenuItem(title: "Hide Selection", action: #selector(ContextMenuTarget.hideSelectionAction), keyEquivalent: "")
        hideItem.target = contextMenuTarget
        hideItem.isEnabled = !workspace.selectedAtomIndices.isEmpty
        menu.addItem(hideItem)

        let showAllItem = NSMenuItem(title: "Show All", action: #selector(ContextMenuTarget.showAllAction), keyEquivalent: "")
        showAllItem.target = contextMenuTarget
        showAllItem.isEnabled = !workspace.hiddenAtomIndices.isEmpty
        menu.addItem(showAllItem)

        let removeItem = NSMenuItem(title: "Remove Selection", action: #selector(ContextMenuTarget.removeSelectionAction), keyEquivalent: "")
        removeItem.target = contextMenuTarget
        removeItem.isEnabled = !workspace.selectedAtomIndices.isEmpty
        menu.addItem(removeItem)

        menu.addItem(NSMenuItem.separator())

        let centerItem = NSMenuItem(title: "Center on Selection", action: #selector(ContextMenuTarget.centerOnSelectionAction), keyEquivalent: "")
        centerItem.target = contextMenuTarget
        centerItem.isEnabled = !workspace.selectedAtomIndices.isEmpty
        menu.addItem(centerItem)

        let resetItem = NSMenuItem(title: "Reset View", action: #selector(ContextMenuTarget.resetViewAction), keyEquivalent: "")
        resetItem.target = contextMenuTarget
        menu.addItem(resetItem)

        NSMenu.popUpContextMenu(menu, with: event, for: view)
    }

    // MARK: - Constraint Sheet

    /// Opens the constraint configuration sheet from the current 3D viewport selection.
    func showConstraintSheetFromSelection() {
        guard let protein = molecules.protein else {
            log.debug("[Selection] showConstraintSheet skipped: no protein loaded", category: .selection)
            return
        }

        var context = ConstraintSheetContext(sourceType: .receptor)

        if let atomIdx = workspace.selectedAtomIndex, atomIdx < protein.atoms.count {
            let atom = protein.atoms[atomIdx]
            context.atomIndex = atomIdx
            context.atomName = atom.name
            context.residueName = "\(atom.residueName) \(atom.residueSeq)"
            context.chainID = atom.chainID
            context.element = atom.element

            // Find residue index
            if let resIdx = protein.residues.firstIndex(where: {
                $0.sequenceNumber == atom.residueSeq && $0.chainID == atom.chainID
            }) {
                context.residueIndex = resIdx
            }
        } else if let resIdx = workspace.selectedResidueIndices.first, resIdx < protein.residues.count {
            let residue = protein.residues[resIdx]
            context.residueIndex = resIdx
            context.residueName = "\(residue.name) \(residue.sequenceNumber)"
            context.chainID = residue.chainID
            // Pick first heavy atom for element hint
            if let firstAtomIdx = residue.atomIndices.first, firstAtomIdx < protein.atoms.count {
                context.element = protein.atoms[firstAtomIdx].element
            }
        }

        // Check if selection is on the ligand
        if let ligand = molecules.ligand, let atomIdx = workspace.selectedAtomIndex {
            let protCount = protein.atoms.count
            if atomIdx >= protCount && atomIdx < protCount + ligand.atoms.count {
                let ligAtomIdx = atomIdx - protCount
                let atom = ligand.atoms[ligAtomIdx]
                context.sourceType = .ligand
                context.atomIndex = ligAtomIdx
                context.atomName = atom.name
                context.residueName = atom.residueName
                context.element = atom.element
                context.residueIndex = nil
            }
        }

        workspace.constraintSheetContext = context
        workspace.showingConstraintSheet = true
    }

    /// Opens the constraint sheet for a specific residue (from SequenceView).
    func showConstraintSheetForResidue(_ residueIndex: Int) {
        guard let protein = molecules.protein,
              residueIndex < protein.residues.count else {
            log.debug("[Selection] showConstraintSheetForResidue skipped: no protein or index out of range", category: .selection)
            return
        }

        let residue = protein.residues[residueIndex]
        var context = ConstraintSheetContext(sourceType: .receptor)
        context.residueIndex = residueIndex
        context.residueName = "\(residue.name) \(residue.sequenceNumber)"
        context.chainID = residue.chainID
        if let firstAtomIdx = residue.atomIndices.first, firstAtomIdx < protein.atoms.count {
            context.element = protein.atoms[firstAtomIdx].element
            context.atomName = protein.atoms[firstAtomIdx].name
        }

        workspace.constraintSheetContext = context
        workspace.showingConstraintSheet = true
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Selection, Visibility, Context Menu, Subsets, Sequence Editing, Surface

extension AppViewModel {

    // MARK: - Selection

    func selectAtom(_ atomIndex: Int?, toggle: Bool = false) {
        if !toggle {
            workspace.selectedAtomIndices.removeAll()
            workspace.selectedResidueIndices.removeAll()
        }

        workspace.selectedAtomIndex = atomIndex

        if let idx = atomIndex {
            if workspace.selectionMode == .atom {
                // Atom mode: track individual atoms, don't auto-select residues
                if toggle && workspace.selectedAtomIndices.contains(idx) {
                    workspace.selectedAtomIndices.remove(idx)
                } else {
                    workspace.selectedAtomIndices.insert(idx)
                }
            } else {
                // Residue mode: select the full parent residue
                if let prot = molecules.protein, let resIdx = prot.residueIndex(forAtom: idx) {
                    if toggle && workspace.selectedResidueIndices.contains(resIdx) {
                        workspace.selectedResidueIndices.remove(resIdx)
                        let resAtoms = Set(prot.residues[resIdx].atomIndices)
                        workspace.selectedAtomIndices.subtract(resAtoms)
                    } else {
                        workspace.selectedResidueIndices.insert(resIdx)
                        workspace.selectedAtomIndices.formUnion(prot.residues[resIdx].atomIndices)
                    }
                }
                if let lig = molecules.ligand {
                    let adjustedIdx = idx - (molecules.protein?.atoms.count ?? 0)
                    if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                        let offset = molecules.protein?.atoms.count ?? 0
                        if toggle && workspace.selectedResidueIndices.contains(resIdx) {
                            workspace.selectedResidueIndices.remove(resIdx)
                            let resAtoms = Set(lig.residues[resIdx].atomIndices.map { $0 + offset })
                            workspace.selectedAtomIndices.subtract(resAtoms)
                        } else {
                            workspace.selectedResidueIndices.insert(resIdx)
                            workspace.selectedAtomIndices.formUnion(lig.residues[resIdx].atomIndices.map { $0 + offset })
                        }
                    }
                }
            }
            log.debug("[Selection] Atom \(idx) selected (mode=\(workspace.selectionMode.rawValue), toggle=\(toggle))", category: .selection)
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
        renderer?.setNeedsRedraw()
    }

    // MARK: - Extended Selection

    /// Double-click: select the entire chain that the atom belongs to.
    func selectChainOfAtom(_ atomIndex: Int) {
        let allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])
        guard atomIndex < allAtoms.count else { return }
        let chainID = allAtoms[atomIndex].chainID
        selectChain(chainID)
    }

    /// Invert the current selection: deselect what's selected, select what wasn't.
    func invertSelection() {
        guard let prot = molecules.protein else { return }
        let allResidues = Set(0..<prot.residues.count)
        let newResidues = allResidues.subtracting(workspace.selectedResidueIndices)
        workspace.selectedResidueIndices = newResidues

        var newAtoms = Set<Int>()
        for resIdx in newResidues {
            guard resIdx < prot.residues.count else { continue }
            newAtoms.formUnion(prot.residues[resIdx].atomIndices)
        }
        workspace.selectedAtomIndices = newAtoms
        workspace.selectedAtomIndex = newAtoms.first
        pushToRenderer()
        log.info("Inverted selection: \(newResidues.count) residues", category: .molecule)
        workspace.statusMessage = "\(newResidues.count) residues selected"
    }

    /// Extend selection by one residue in each direction along the chain sequence.
    func extendSelectionByOneResidue() {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else { return }

        var newResidues = workspace.selectedResidueIndices
        for chain in prot.chains {
            let chainResidues = chain.residueIndices.sorted()
            for resIdx in workspace.selectedResidueIndices {
                guard let pos = chainResidues.firstIndex(of: resIdx) else { continue }
                if pos > 0 { newResidues.insert(chainResidues[pos - 1]) }
                if pos < chainResidues.count - 1 { newResidues.insert(chainResidues[pos + 1]) }
            }
        }

        let added = newResidues.count - workspace.selectedResidueIndices.count
        workspace.selectedResidueIndices = newResidues
        for resIdx in newResidues {
            guard resIdx < prot.residues.count else { continue }
            workspace.selectedAtomIndices.formUnion(prot.residues[resIdx].atomIndices)
        }
        pushToRenderer()
        log.info("Extended selection by \(added) adjacent residues", category: .molecule)
        workspace.statusMessage = "\(workspace.selectedResidueIndices.count) residues selected"
    }

    /// Shrink selection by one residue from each end along the chain sequence.
    func shrinkSelectionByOneResidue() {
        guard let prot = molecules.protein, !workspace.selectedResidueIndices.isEmpty else { return }

        // Find boundary residues (those with an unselected neighbor) and remove them
        var boundary = Set<Int>()
        for chain in prot.chains {
            let chainResidues = chain.residueIndices.sorted()
            for resIdx in workspace.selectedResidueIndices {
                guard let pos = chainResidues.firstIndex(of: resIdx) else { continue }
                let prevSelected = pos > 0 && workspace.selectedResidueIndices.contains(chainResidues[pos - 1])
                let nextSelected = pos < chainResidues.count - 1 && workspace.selectedResidueIndices.contains(chainResidues[pos + 1])
                if !prevSelected || !nextSelected {
                    boundary.insert(resIdx)
                }
            }
        }

        let newResidues = workspace.selectedResidueIndices.subtracting(boundary)
        workspace.selectedResidueIndices = newResidues

        // Rebuild atom selection
        var newAtoms = Set<Int>()
        for resIdx in newResidues {
            guard resIdx < prot.residues.count else { continue }
            newAtoms.formUnion(prot.residues[resIdx].atomIndices)
        }
        workspace.selectedAtomIndices = newAtoms
        workspace.selectedAtomIndex = newAtoms.first
        pushToRenderer()
        workspace.statusMessage = "\(newResidues.count) residues selected"
    }

    // MARK: - Visibility

    func toggleChainVisibility(_ chainID: String) {
        let willHide = !workspace.hiddenChainIDs.contains(chainID)
        if willHide {
            workspace.hiddenChainIDs.insert(chainID)
        } else {
            workspace.hiddenChainIDs.remove(chainID)
        }
        pushToRenderer()
        log.info("Chain \(chainID) \(willHide ? "hidden" : "shown")", category: .selection)
    }

    func setRenderMode(_ mode: RenderMode) {
        workspace.renderMode = mode
        workspace.ligandRenderMode = nil
        pushToRenderer()
        log.info("Render mode → \(mode.rawValue)", category: .render)
    }

    func setLigandRenderMode(_ mode: RenderMode) {
        workspace.ligandRenderMode = mode
        pushToRenderer()
        log.info("Ligand render mode → \(mode.rawValue)", category: .render)
    }

    func toggleHydrogens() {
        workspace.showHydrogens.toggle()
        pushToRenderer()
        log.info("Hydrogens \(workspace.showHydrogens ? "shown" : "hidden")", category: .render)
    }

    func toggleLighting() {
        workspace.useDirectionalLighting.toggle()
        renderer?.lightingMode = workspace.useDirectionalLighting ? 1 : 0
        log.info("Directional lighting \(workspace.useDirectionalLighting ? "on" : "off")", category: .render)
    }

    // MARK: - Box Selection

    func handleBoxSelection(atomIndices: [Int], addToExisting: Bool) {
        if !addToExisting {
            workspace.selectedAtomIndices.removeAll()
            workspace.selectedResidueIndices.removeAll()
        }

        if workspace.selectionMode == .atom {
            workspace.selectedAtomIndices.formUnion(atomIndices)
        } else {
            let proteinOffset = molecules.protein?.atoms.count ?? 0

            for idx in atomIndices {
                if let prot = molecules.protein, let resIdx = prot.residueIndex(forAtom: idx) {
                    workspace.selectedResidueIndices.insert(resIdx)
                    workspace.selectedAtomIndices.formUnion(prot.residues[resIdx].atomIndices)
                }
                if let lig = molecules.ligand {
                    let adjustedIdx = idx - proteinOffset
                    if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                        workspace.selectedResidueIndices.insert(resIdx)
                        workspace.selectedAtomIndices.formUnion(
                            lig.residues[resIdx].atomIndices.map { $0 + proteinOffset }
                        )
                    }
                }
            }
        }

        workspace.selectedAtomIndex = atomIndices.first
        pushToRenderer()

        if workspace.selectionMode == .atom {
            let count = workspace.selectedAtomIndices.count
            log.info("Selected \(count) atom\(count == 1 ? "" : "s") by box selection", category: .molecule)
            workspace.statusMessage = "\(count) atom\(count == 1 ? "" : "s") selected"
        } else {
            let count = workspace.selectedResidueIndices.count
            log.info("Selected \(count) residue\(count == 1 ? "" : "s") by box selection", category: .molecule)
            workspace.statusMessage = "\(count) residue\(count == 1 ? "" : "s") selected"
        }
    }

    // MARK: - Molecular Surface

    func toggleSurface() {
        workspace.showSurface.toggle()
        log.info("Surface \(workspace.showSurface ? "enabled" : "disabled")", category: .render)
        if workspace.showSurface {
            generateSurface()
        } else {
            workspace.surfaceLegend = nil
            renderer?.clearSurfaceMesh()
        }
    }

    func setSurfaceColorMode(_ mode: SurfaceColorMode) {
        workspace.surfaceColorMode = mode
        log.info("Surface color mode → \(mode.rawValue)", category: .render)
        if workspace.showSurface {
            generateSurface()
        }
    }

    /// Regenerate the surface with current probe radius / grid spacing settings.
    func regenerateSurface() {
        if workspace.showSurface {
            log.info("Regenerating surface", category: .render)
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
            // Auto-compute charges for electrostatic surface if not yet available
            if workspace.surfaceColorMode == .esp {
                let hasCharges = prot.atoms.contains { abs($0.charge) > 0.0001 }
                if !hasCharges {
                    workspace.statusMessage = "Computing charges for electrostatic surface..."
                    let calculator = EEMChargeCalculator(device: renderer.device)
                    if let charges = try? await calculator.computeCharges(
                        atoms: prot.atoms, bonds: prot.bonds, totalCharge: 0
                    ) {
                        for i in 0..<min(charges.count, prot.atoms.count) {
                            molecules.protein?.atoms[i].charge = charges[i]
                        }
                        // Apply formal charge fallback for any atoms still uncharged
                        if var atoms = molecules.protein?.atoms {
                            atoms = applyElectrostaticFallback(to: atoms)
                            molecules.protein?.atoms = atoms
                        }
                        log.info("Auto-computed EEM charges for electrostatic surface", category: .molecule)
                    }
                }
            }

            guard let prot = molecules.protein else {
                workspace.isGeneratingSurface = false
                return
            }

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

            let settings = currentSurfaceSettings()
            let atoms = prot.atoms.filter { $0.element != .H }
            // Runs off the main actor (grid + marching cubes + prefix sum on large proteins).
            let result = await gen.generateSurface(atoms: atoms, settings: settings)

            // The surface was hidden meanwhile, or a newer request with different settings
            // is in flight: drop this result instead of overwriting the newer state.
            guard workspace.showSurface else {
                workspace.isGeneratingSurface = false
                return
            }
            guard currentSurfaceSettings() == settings else { return }

            if let result {
                renderer.updateSurfaceMesh(result)
                workspace.surfaceLegend = result.legend
                log.success("Surface: \(result.vertexCount) vertices, \(result.indexCount / 3) triangles", category: .molecule)
                workspace.statusMessage = "Surface generated"
            } else {
                workspace.surfaceLegend = nil
                log.error("Surface generation failed", category: .molecule)
                workspace.statusMessage = "Surface failed"
            }
            workspace.isGeneratingSurface = false
        }
    }

    private func currentSurfaceSettings() -> SurfaceGenerator.Settings {
        SurfaceGenerator.Settings(
            fieldType: workspace.surfaceType,
            colorMode: workspace.surfaceColorMode,
            gridSpacing: workspace.surfaceGridSpacing,
            probeRadius: workspace.surfaceProbeRadius
        )
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
                molecules.ligand = Molecule(
                    name: lig.name,
                    atoms: newAtoms,
                    bonds: newBonds,
                    title: lig.title,
                    smiles: lig.smiles
                )
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
        renderer?.setNeedsRedraw()
        log.info("Centered on selection", category: .molecule)
    }

    // MARK: - Residue Subsets

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

    /// Fills `secondaryStructureAssignments` from backbone geometry when the source file
    /// carried no HELIX/SHEET records, so the ribbon, the Sequence step and structure-based
    /// selection all read the same per-residue assignment.
    func ensureSecondaryStructure() {
        guard let prot = molecules.protein, prot.secondaryStructureAssignments.isEmpty else { return }
        let backbone: [BackboneResidue] = prot.chains.filter { $0.type == .protein }.flatMap { chain in
            chain.residueIndices.compactMap { idx -> BackboneResidue? in
                guard idx < prot.residues.count, prot.residues[idx].isStandard else { return nil }
                let residue = prot.residues[idx]
                var named: [String: SIMD3<Float>] = [:]
                for atomIdx in residue.atomIndices where atomIdx < prot.atoms.count {
                    let atom = prot.atoms[atomIdx]
                    let name = atom.name.trimmingCharacters(in: .whitespaces)
                    if named[name] == nil { named[name] = atom.position }
                }
                guard let n = named["N"], let ca = named["CA"], let c = named["C"], let o = named["O"] else { return nil }
                return BackboneResidue(
                    chainID: residue.chainID, sequenceNumber: residue.sequenceNumber,
                    n: n, ca: ca, c: c, o: o, h: named["H"] ?? named["HN"],
                    isProline: residue.name == "PRO"
                )
            }
        }
        let segments = BackboneSecondaryStructure.segments(backbone, BackboneSecondaryStructure.assign(backbone))
        guard !segments.isEmpty else { return }
        prot.secondaryStructureAssignments = segments
        pushToRenderer()
        let helices = segments.filter { $0.type == .helix }.count
        let strands = segments.filter { $0.type == .sheet }.count
        log.info("Assigned secondary structure from backbone H-bonds: \(helices) helices, \(strands) strands", category: .molecule)
    }

    func selectBySecondaryStructure(_ ssType: SecondaryStructure, chainID: String? = nil) {
        ensureSecondaryStructure()
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
        let residueCount = workspace.selectedResidueIndices.count
        let residueText = "\(residueCount) residue\(residueCount == 1 ? "" : "s")"
        if ssType == .coil {
            workspace.statusMessage = "\(residueText) in coil selected"
        } else {
            let segmentCount = prot.secondaryStructureAssignments.filter {
                $0.type == ssType && (chainID == nil || $0.chain == chainID)
            }.count
            let segmentName = switch ssType {
            case .helix: segmentCount == 1 ? "helix" : "helices"
            case .sheet: segmentCount == 1 ? "strand" : "strands"
            case .turn: segmentCount == 1 ? "turn" : "turns"
            case .coil: ""
            }
            workspace.statusMessage = "\(segmentCount) \(segmentName) selected (\(residueText))"
        }
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

        // During docking, only use protein positions (ligand is replaced by ghost pose)
        let allAtoms: [Atom]
        if docking.isDocking {
            allAtoms = molecules.protein?.atoms ?? []
        } else {
            allAtoms = (molecules.protein?.atoms ?? []) + (molecules.ligand?.atoms ?? [])
        }

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
        guard let renderer else { return }
        // During docking, focus on the pocket center instead of the moving ligand
        if docking.isDocking, let pocket = docking.selectedPocket {
            renderer.fitToPositions([pocket.center])
            log.info("Camera fit to pocket center (live docking)", category: .render)
            return
        }
        guard let lig = molecules.ligand, !lig.atoms.isEmpty else { return }
        renderer.fitToPositions(lig.atoms.map(\.position))
        log.info("Camera fit to ligand (\(lig.name))", category: .render)
    }

    func fitToProtein() {
        guard let renderer else { return }
        guard let prot = molecules.protein, !prot.atoms.isEmpty else { return }
        renderer.fitToPositions(prot.atoms.map(\.position))
        log.info("Camera fit to protein (\(prot.name))", category: .render)
    }

    // MARK: - Context Menu Builder

    func showContextMenu(event: NSEvent, view: NSView) {
        let menu = NSMenu(title: "Context")
        // Item state below follows the current selection; AppKit's auto-enabling would
        // ignore it and enable every item whose target answers its action.
        menu.autoenablesItems = false

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

        // Extend selection submenu
        let hasSelection = !workspace.selectedAtomIndices.isEmpty
        let extendMenu = NSMenu(title: "Extend")
        let extendItem = NSMenuItem(title: "Extend Selection", action: nil, keyEquivalent: "")
        extendItem.submenu = extendMenu
        extendItem.isEnabled = hasSelection

        for distance: Float in [5, 6, 8, 10] {
            let label = String(format: "Nearby Residues Within %.0f \u{00C5}", distance)
            let item = NSMenuItem(title: label, action: #selector(ContextMenuTarget.selectNearbyAction(_:)), keyEquivalent: "")
            item.representedObject = distance
            item.target = contextMenuTarget
            extendMenu.addItem(item)
        }
        extendMenu.addItem(NSMenuItem.separator())

        let extendOneItem = NSMenuItem(title: "Extend by \u{00B1}1 Residue", action: #selector(ContextMenuTarget.extendByOneResidueAction), keyEquivalent: "")
        extendOneItem.target = contextMenuTarget
        extendMenu.addItem(extendOneItem)

        let shrinkOneItem = NSMenuItem(title: "Shrink by 1 Residue", action: #selector(ContextMenuTarget.shrinkByOneResidueAction), keyEquivalent: "")
        shrinkOneItem.target = contextMenuTarget
        extendMenu.addItem(shrinkOneItem)

        menu.addItem(extendItem)

        let invertItem = NSMenuItem(title: "Invert Selection", action: #selector(ContextMenuTarget.invertSelectionAction), keyEquivalent: "")
        invertItem.target = contextMenuTarget
        invertItem.isEnabled = hasSelection
        menu.addItem(invertItem)

        let subsetItem = NSMenuItem(title: "Create Subset from Selection", action: #selector(ContextMenuTarget.createSubsetAction), keyEquivalent: "")
        subsetItem.target = contextMenuTarget
        subsetItem.isEnabled = !workspace.selectedResidueIndices.isEmpty
        menu.addItem(subsetItem)

        menu.addItem(NSMenuItem.separator())

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
        hideItem.isEnabled = hasSelection
        menu.addItem(hideItem)

        let showAllItem = NSMenuItem(title: "Show All", action: #selector(ContextMenuTarget.showAllAction), keyEquivalent: "")
        showAllItem.target = contextMenuTarget
        showAllItem.isEnabled = !workspace.hiddenAtomIndices.isEmpty
        menu.addItem(showAllItem)

        let removeItem = NSMenuItem(title: "Remove Selection", action: #selector(ContextMenuTarget.removeSelectionAction), keyEquivalent: "")
        removeItem.target = contextMenuTarget
        removeItem.isEnabled = hasSelection
        menu.addItem(removeItem)

        menu.addItem(NSMenuItem.separator())

        let centerItem = NSMenuItem(title: "Center on Selection", action: #selector(ContextMenuTarget.centerOnSelectionAction), keyEquivalent: "")
        centerItem.target = contextMenuTarget
        centerItem.isEnabled = hasSelection
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

// MARK: - Secondary structure from backbone geometry

/// Backbone atoms of one amino-acid residue, listed in chain order.
struct BackboneResidue {
    let chainID: String
    let sequenceNumber: Int
    let n: SIMD3<Float>
    let ca: SIMD3<Float>
    let c: SIMD3<Float>
    let o: SIMD3<Float>
    /// Amide hydrogen when the model carries one; otherwise it is placed geometrically.
    let h: SIMD3<Float>?
    let isProline: Bool
}

/// Kabsch–Sander (DSSP) secondary-structure assignment from backbone H-bonds, used when a
/// structure ships without HELIX/SHEET records (CASF/PDBbind exports, predicted models,
/// prepared or re-written PDBs). α (H), 3₁₀ (G) and π (I) helices map to `.helix`, ladder
/// strands (E) to `.sheet`, H-bonded turns (T) to `.turn`; isolated bridges and bends stay coil.
enum BackboneSecondaryStructure {
    private enum Label { case none, alpha, strand, threeTen, pi, turn }

    /// Electrostatic H-bond energy (kcal/mol) between the C=O of `acceptor` and the
    /// N-H of the donor: E = 0.084·332·(1/rON + 1/rCH − 1/rOH − 1/rCN).
    private static func hbondEnergy(acceptor a: BackboneResidue, donorN n: SIMD3<Float>, donorH h: SIMD3<Float>) -> Float {
        let rON = simd_distance(a.o, n), rCH = simd_distance(a.c, h)
        let rOH = simd_distance(a.o, h), rCN = simd_distance(a.c, n)
        guard min(rON, rCH, rOH, rCN) > 0.5 else { return -9.9 }
        return max(27.888 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN), -9.9)
    }

    static func assign(_ res: [BackboneResidue]) -> [SecondaryStructure] {
        let count = res.count
        guard count >= 3 else { return Array(repeating: .coil, count: count) }

        // linked[i]: residue i is peptide-bonded to residue i + 1.
        let linked: [Bool] = (0..<count).map { i in
            i + 1 < count && res[i].chainID == res[i + 1].chainID
                && simd_distance(res[i].c, res[i + 1].n) < 2.5
        }
        func continuous(_ a: Int, _ b: Int) -> Bool {
            guard a >= 0, b < count, a <= b else { return false }
            return (a..<b).allSatisfy { linked[$0] }
        }

        // Amide H: model coordinate, else 1 Å from N opposite the preceding C=O (DSSP convention).
        let amideH: [SIMD3<Float>?] = (0..<count).map { i in
            if res[i].isProline { return nil }
            if let h = res[i].h { return h }
            guard i > 0, linked[i - 1] else { return nil }
            return res[i].n + simd_normalize(res[i - 1].c - res[i - 1].o)
        }

        // accepts[i] = donors j whose N-H bonds to the C=O of i.
        var accepts = [Set<Int>](repeating: [], count: count)
        for i in 0..<count {
            for j in 0..<count where abs(i - j) >= 2 {
                guard let h = amideH[j], simd_distance(res[i].ca, res[j].ca) < 9 else { continue }
                if hbondEnergy(acceptor: res[i], donorN: res[j].n, donorH: h) < -0.5 {
                    accepts[i].insert(j)
                }
            }
        }
        func hb(_ i: Int, _ j: Int) -> Bool {
            i >= 0 && i < count && j >= 0 && j < count && accepts[i].contains(j)
        }
        func turn(_ n: Int, at i: Int) -> Bool { hb(i, i + n) && continuous(i, i + n) }

        var labels = [Label](repeating: .none, count: count)

        // α-helix: two consecutive 4-turns at i-1 and i → residues i...i+3.
        for i in 1..<count where turn(4, at: i - 1) && turn(4, at: i) {
            for k in i...min(i + 3, count - 1) { labels[k] = .alpha }
        }

        // β-bridges, then ladders (consecutive bridges) become strands.
        struct Bridge: Hashable { let i: Int; let j: Int; let parallel: Bool }
        var bridges = Set<Bridge>()
        for i in 1..<(count - 1) where continuous(i - 1, i + 1) {
            for j in (i + 3)..<max(i + 3, count - 1) where continuous(j - 1, j + 1) {
                if (hb(i - 1, j) && hb(j, i + 1)) || (hb(j - 1, i) && hb(i, j + 1)) {
                    bridges.insert(Bridge(i: i, j: j, parallel: true))
                }
                if (hb(i, j) && hb(j, i)) || (hb(i - 1, j + 1) && hb(j - 1, i + 1)) {
                    bridges.insert(Bridge(i: i, j: j, parallel: false))
                }
            }
        }
        for b in bridges {
            let step = b.parallel ? 1 : -1
            let inLadder = bridges.contains(Bridge(i: b.i + 1, j: b.j + step, parallel: b.parallel))
                || bridges.contains(Bridge(i: b.i - 1, j: b.j - step, parallel: b.parallel))
            guard inLadder else { continue }
            for k in [b.i, b.j] where labels[k] == .none { labels[k] = .strand }
        }
        // Single-residue β-bulges inside a strand.
        for i in 1..<(count - 1)
        where labels[i] == .none && labels[i - 1] == .strand && labels[i + 1] == .strand && continuous(i - 1, i + 1) {
            labels[i] = .strand
        }

        // 3₁₀ and π helices only where no α/strand residue is overwritten.
        for (n, label) in [(3, Label.threeTen), (5, Label.pi)] {
            for i in 1..<count where turn(n, at: i - 1) && turn(n, at: i) {
                let span = i...min(i + n - 1, count - 1)
                if span.allSatisfy({ labels[$0] == .none || labels[$0] == label }) {
                    for k in span { labels[k] = label }
                }
            }
        }

        // H-bonded turns.
        for n in 3...5 {
            for i in 0..<count where turn(n, at: i) {
                for k in (i + 1)..<min(i + n, count) where labels[k] == .none { labels[k] = .turn }
            }
        }

        return labels.map { label in
            switch label {
            case .alpha, .threeTen, .pi: .helix
            case .strand: .sheet
            case .turn: .turn
            case .none: .coil
            }
        }
    }

    /// Collapses per-residue assignments into contiguous non-coil segments.
    static func segments(
        _ res: [BackboneResidue], _ ss: [SecondaryStructure]
    ) -> [(start: Int, end: Int, type: SecondaryStructure, chain: String)] {
        var out: [(start: Int, end: Int, type: SecondaryStructure, chain: String)] = []
        var runStart: Int?
        for i in 0...res.count {
            let continues = i < res.count && i > 0 && runStart != nil
                && ss[i] == ss[i - 1] && res[i].chainID == res[i - 1].chainID
                && res[i].sequenceNumber == res[i - 1].sequenceNumber + 1
            if continues { continue }
            if let s = runStart, ss[s] != .coil {
                out.append((start: res[s].sequenceNumber, end: res[i - 1].sequenceNumber,
                            type: ss[s], chain: res[s].chainID))
            }
            runStart = i < res.count ? i : nil
        }
        return out
    }
}

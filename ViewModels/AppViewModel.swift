import SwiftUI
import MetalKit

@Observable
@MainActor
final class AppViewModel {
    // Molecules
    var protein: Molecule?
    var ligand: Molecule?

    // Render state
    var renderMode: RenderMode = .ballAndStick
    var showHydrogens: Bool = true
    var showProtein: Bool = true
    var showLigand: Bool = true
    var hiddenChainIDs: Set<String> = []
    var useDirectionalLighting: Bool = false

    // Z-slab clipping
    var enableClipping: Bool = false
    var clipNearZ: Float = 0
    var clipFarZ: Float = 100

    // Molecular surface
    var showSurface: Bool = false
    var surfaceType: SurfaceFieldType = .connolly
    var surfaceColorByESP: Bool = false
    var isGeneratingSurface: Bool = false
    private var surfaceGenerator: SurfaceGenerator?

    // Selection
    var selectedAtomIndex: Int? = nil
    var selectedAtomIndices: Set<Int> = []
    var selectedResidueIndices: Set<Int> = []
    var hiddenAtomIndices: Set<Int> = []

    // Residue subsets (MOE-style user-defined groups)
    var residueSubsets: [ResidueSubset] = []

    // Status
    var statusMessage: String = "Ready"

    // Loading state
    var isLoading: Bool = false
    var loadingMessage: String = ""

    // Search state
    var searchResults: [PDBSearchResult] = []
    var isSearching: Bool = false

    // Preparation state
    var preparationReport: ProteinPreparation.PreparationReport?
    var protonationPH: Float = 7.4
    var isMinimizing: Bool = false

    // Docking state
    var detectedPockets: [BindingPocket] = []
    var selectedPocket: BindingPocket?
    var isDocking: Bool = false
    var dockingGeneration: Int = 0
    var dockingTotalGenerations: Int = 100
    var dockingBestEnergy: Float = .infinity
    var dockingResults: [DockingResult] = []
    var dockingConfig = DockingConfig()
    var dockingEngine: DockingEngine?
    var currentInteractions: [MolecularInteraction] = []
    var showInteractionDiagram: Bool = false
    var interactionDiagramPoseIndex: Int = 0

    /// Original ligand preserved before docking mutates self.ligand with pose transforms.
    /// Used by showDockingPose to always apply transforms from the original coordinates.
    private var originalDockingLigand: Molecule?

    // Batch docking state
    var batchResults: [(ligandName: String, results: [DockingResult])] = []
    var isBatchDocking: Bool = false
    var batchProgress: (current: Int, total: Int) = (0, 0)
    var batchQueue: [LigandEntry] = []  // ligands queued for batch docking
    private var batchDockingTask: Task<Void, Never>?

    // Virtual screening state
    var screeningPipeline: VirtualScreeningPipeline?
    var screeningState: VirtualScreeningPipeline.ScreeningState = .idle
    var screeningProgress: Float = 0
    var screeningHits: [VirtualScreeningPipeline.ScreeningHit] = []
    private var screeningTask: Task<Void, Never>?

    // Analog generation state
    var isGeneratingAnalogs: Bool = false
    var analogGenerationProgress: Float = 0

    // Results filter state (persists across tab switches)
    var resultsLipinskiFilter: Bool = false
    var resultsEnergyCutoff: Float = 0
    var resultsMLScoreCutoff: Float = 0
    var resultsHasInitializedCutoffs: Bool = false

    // Raw PDB content (cached for C++ protein prep)
    var rawPDBContent: String?

    // Renderer (set after MetalView is created)
    var renderer: Renderer?

    // ML models (alternative scoring/detection)
    let druseScore = DruseScoreInference()
    let pocketDetectorML = PocketDetectorInference()
    let admetPredictor = ADMETPredictor()
    var useDruseScoreReranking: Bool = false

    // Ligand database
    let ligandDB = LigandDatabase()

    // Activity log
    let log = ActivityLog.shared

    init() {}

    // MARK: - Ligand Management (Single Entry Point)

    /// Set any molecule as the active ligand for docking.
    /// ALL import paths (LigandDatabaseView, LigandDatabaseWindow,
    /// DockingTabView picker, PDB co-crystallized) MUST call this single method.
    func setLigandForDocking(_ molecule: Molecule) {
        // Normalize ligand chain IDs to "L" and residue to "LIG" so the inspector
        // treats the entire ligand as a single chain, not N bugged chains from RDKit.
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

        ligand = normalizedLigand
        originalDockingLigand = nil  // clear stale docking reference
        dockingResults = []
        currentInteractions = []
        renderer?.updateInteractionLines([])
        pushToRenderer()
        renderer?.fitToContent()
        log.success("Set \(molecule.name) as active ligand (\(molecule.atomCount) atoms)", category: .molecule)
        statusMessage = "\(molecule.name) ready for docking"
    }

    /// Clear the active ligand (removes co-crystallized or any loaded ligand).
    func clearLigand() {
        let name = ligand?.name ?? "ligand"
        ligand = nil
        originalDockingLigand = nil
        dockingResults = []
        currentInteractions = []
        renderer?.updateInteractionLines([])
        pushToRenderer()
        log.info("Cleared \(name)", category: .molecule)
        statusMessage = "No ligand loaded"
    }

    // MARK: - Load Test Data

    func loadCaffeine() {
        let mol = TestMolecules.caffeine()
        ligand = mol
        protein = nil
        pushToRenderer()
        renderer?.fitToContent()
        log.success("Loaded caffeine (\(mol.atomCount) atoms, \(mol.bondCount) bonds)", category: .molecule)
        statusMessage = "Caffeine loaded"
    }

    func loadAlanineDipeptide() {
        let mol = TestMolecules.alanineDipeptide()
        protein = mol
        ligand = nil
        pushToRenderer()
        renderer?.fitToContent()
        log.success("Loaded alanine dipeptide (\(mol.atomCount) atoms, \(mol.bondCount) bonds)", category: .molecule)
        statusMessage = "Ala dipeptide loaded"
    }

    func loadBoth() {
        protein = TestMolecules.alanineDipeptide()
        ligand = TestMolecules.caffeine()
        // Offset caffeine so it doesn't overlap
        if let lig = ligand {
            for i in 0..<lig.atoms.count {
                lig.atoms[i].position += SIMD3<Float>(5, 0, 0)
            }
            lig.rebuildDerivedData()
        }
        pushToRenderer()
        renderer?.fitToContent()
        log.success("Loaded protein + ligand", category: .molecule)
        statusMessage = "Both molecules loaded"
    }

    // MARK: - Push Data to Renderer

    func pushToRenderer() {
        guard let renderer else { return }

        // Always provide full molecule positions for camera fitting (independent of render mode)
        var allPositions: [SIMD3<Float>] = []
        if showProtein, let prot = protein {
            allPositions.append(contentsOf: prot.atoms.map(\.position))
        }
        if showLigand, let lig = ligand {
            allPositions.append(contentsOf: lig.atoms.map(\.position))
        }
        renderer.updateAllMoleculePositions(allPositions)

        var allAtoms: [Atom] = []
        var allBonds: [Bond] = []

        // Ribbon mode: generate proper triangle mesh for protein backbone
        if renderMode == .ribbon, let prot = protein, showProtein {
            // Generate ribbon mesh only for visible chains
            let visibleProt: Molecule
            if hiddenChainIDs.isEmpty {
                visibleProt = prot
            } else {
                // Create a temporary molecule with only visible chains for ribbon generation
                let visAtoms = prot.atoms.filter { !hiddenChainIDs.contains($0.chainID) }
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

            // Build CA control points for ribbon hit-testing (using original protein atom IDs)
            var caControlPoints: [(position: SIMD3<Float>, atomID: Int)] = []
            for atom in prot.atoms {
                if hiddenChainIDs.contains(atom.chainID) { continue }
                if atom.name.trimmingCharacters(in: .whitespaces) == "CA" {
                    caControlPoints.append((position: atom.position, atomID: atom.id))
                }
            }
            renderer.updateRibbonCAControlPoints(caControlPoints)

            // In ribbon mode, ALWAYS render ligand as ball-and-stick overlay.
            // Also render any co-crystallized HETATM from protein (non-water, non-standard residues).
            if showLigand, let lig = ligand {
                let filteredAtoms = showHydrogens ? lig.atoms : lig.atoms.filter { $0.element != .H }
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

            renderer.selectedAtomIndex = selectedAtomIndex ?? -1
            renderer.selectedResidueAtomIndices = []
            // Force ball-and-stick for the ligand atoms overlaid on ribbon
            renderer.renderMode = .ballAndStick
            renderer.updateMoleculeData(atoms: allAtoms, bonds: allBonds)
            return
        }

        // Non-ribbon mode: clear ribbon mesh
        renderer.clearRibbonMesh()

        // Protein — build index remap for visible atoms
        if showProtein, let prot = protein {
            var protIdMap: [Int: Int] = [:]  // original atom.id → allAtoms index
            for atom in prot.atoms {
                if !showHydrogens && atom.element == .H { continue }
                if hiddenChainIDs.contains(atom.chainID) { continue }
                if hiddenAtomIndices.contains(atom.id) { continue }
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

        // Ligand — same remap approach
        if showLigand, let lig = ligand {
            let ligOffset = protein?.atoms.count ?? 0
            var ligIdMap: [Int: Int] = [:]
            for atom in lig.atoms {
                if !showHydrogens && atom.element == .H { continue }
                if hiddenAtomIndices.contains(atom.id + ligOffset) { continue }
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

        // Update selection state
        renderer.selectedAtomIndex = selectedAtomIndex ?? -1
        renderer.selectedResidueAtomIndices = Set(
            selectedResidueIndices.flatMap { resIdx -> [Int] in
                // Resolve from whichever molecule contains this residue
                if let prot = protein, resIdx < prot.residues.count {
                    return prot.residues[resIdx].atomIndices
                }
                return []
            }
        )
        renderer.renderMode = renderMode

        // Sync clipping state
        renderer.enableClipping = enableClipping
        renderer.clipNearZ = clipNearZ
        renderer.clipFarZ = clipFarZ

        renderer.updateMoleculeData(atoms: allAtoms, bonds: allBonds)
    }

    // MARK: - Selection

    func selectAtom(_ atomIndex: Int?, toggle: Bool = false) {
        if !toggle {
            // Normal click: clear previous selection
            selectedResidueIndices.removeAll()
        }

        selectedAtomIndex = atomIndex

        if let idx = atomIndex {
            // Find residue for this atom
            if let prot = protein, let resIdx = prot.residueIndex(forAtom: idx) {
                if toggle && selectedResidueIndices.contains(resIdx) {
                    selectedResidueIndices.remove(resIdx)
                } else {
                    selectedResidueIndices.insert(resIdx)
                }
            }
            if let lig = ligand {
                let adjustedIdx = idx - (protein?.atoms.count ?? 0)
                if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                    if toggle && selectedResidueIndices.contains(resIdx) {
                        selectedResidueIndices.remove(resIdx)
                    } else {
                        selectedResidueIndices.insert(resIdx)
                    }
                }
            }
        }

        pushToRenderer()
    }

    func deselectAll() {
        selectedAtomIndex = nil
        selectedAtomIndices.removeAll()
        selectedResidueIndices.removeAll()
        pushToRenderer()
    }

    func focusOnAtom(_ atomIndex: Int) {
        // Find position from active molecules
        let allAtoms = (protein?.atoms ?? []) + (ligand?.atoms ?? [])
        guard atomIndex < allAtoms.count else { return }
        renderer?.camera.focusOnPoint(allAtoms[atomIndex].position)
    }

    // MARK: - Selected Atom Info

    var selectedAtom: Atom? {
        guard let idx = selectedAtomIndex else { return nil }
        let allAtoms = (protein?.atoms ?? []) + (ligand?.atoms ?? [])
        guard idx < allAtoms.count else { return nil }
        return allAtoms[idx]
    }

    // MARK: - Visibility

    func toggleChainVisibility(_ chainID: String) {
        if hiddenChainIDs.contains(chainID) {
            hiddenChainIDs.remove(chainID)
        } else {
            hiddenChainIDs.insert(chainID)
        }
        pushToRenderer()
    }

    func setRenderMode(_ mode: RenderMode) {
        renderMode = mode
        pushToRenderer()
    }

    func toggleHydrogens() {
        showHydrogens.toggle()
        pushToRenderer()
    }

    func toggleLighting() {
        useDirectionalLighting.toggle()
        renderer?.lightingMode = useDirectionalLighting ? 1 : 0
    }

    // MARK: - All Chains

    var allChains: [Chain] {
        var chainsByID: [String: Chain] = [:]
        var order: [String] = []

        if let prot = protein {
            for chain in prot.chains {
                chainsByID[chain.id] = chain
                order.append(chain.id)
            }
        }
        if let lig = ligand {
            for chain in lig.chains {
                if chainsByID[chain.id] == nil {
                    // New chain ID not seen in protein
                    chainsByID[chain.id] = chain
                    order.append(chain.id)
                }
                // If already present from protein, skip the duplicate
            }
        }

        return order.compactMap { chainsByID[$0] }
    }

    // MARK: - PDB Loading

    func loadFromPDB(id: String) {
        isLoading = true
        loadingMessage = "Fetching \(id.uppercased())..."
        log.info("Fetching PDB \(id.uppercased()) from RCSB...", category: .network)

        Task {
            do {
                let content = try await PDBService.shared.fetchPDBFile(id: id)
                rawPDBContent = content  // Cache for C++ protein prep
                loadingMessage = "Parsing..."

                let result = await Task.detached { PDBParser.parse(content) }.value

                // Construct Molecules on MainActor
                if let protData = result.protein {
                    let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                       bonds: protData.bonds, title: protData.title)
                    mol.secondaryStructureAssignments = protData.ssRanges.map {
                        (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                    }
                    protein = mol
                } else {
                    protein = nil
                }

                ligand = result.ligands.first.map {
                    Molecule(name: $0.name, atoms: $0.atoms, bonds: $0.bonds, title: $0.title)
                }

                preparationReport = ProteinPreparation.analyze(
                    atoms: result.protein?.atoms ?? [],
                    bonds: result.protein?.bonds ?? [],
                    waterCount: result.waterCount
                )

                hiddenChainIDs.removeAll()
                pushToRenderer()
                renderer?.fitToContent()

                let atomCount = protein?.atomCount ?? 0
                let ligCount = result.ligands.count
                log.success("Loaded \(id.uppercased()): \(atomCount) protein atoms, \(ligCount) ligand(s), \(result.waterCount) waters removed", category: .pdb)

                for w in result.warnings.prefix(5) {
                    log.warn(w, category: .pdb)
                }

                statusMessage = "\(id.uppercased()) loaded"
            } catch {
                log.error("Failed to load \(id): \(error.localizedDescription)", category: .pdb)
                statusMessage = "Failed to load \(id)"
            }

            isLoading = false
            loadingMessage = ""
        }
    }

    // MARK: - File Loading

    func loadFromFile(url: URL) {
        guard let format = FileImportHandler.detectFormat(url: url) else {
            log.error("Unsupported file format: \(url.pathExtension)", category: .system)
            return
        }

        if format != .pdb {
            rawPDBContent = nil
        }

        isLoading = true
        loadingMessage = "Loading \(url.lastPathComponent)..."
        log.info("Loading \(url.lastPathComponent)...", category: .molecule)

        Task {
            do {
                switch format {
                case .pdb:
                    let content = try String(contentsOf: url, encoding: .utf8)
                    rawPDBContent = content
                    let result = await Task.detached { PDBParser.parse(content) }.value

                    if let protData = result.protein {
                        let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                           bonds: protData.bonds, title: protData.title)
                        mol.secondaryStructureAssignments = protData.ssRanges.map {
                            (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                        }
                        protein = mol
                    }
                    ligand = result.ligands.first.map {
                        Molecule(name: $0.name, atoms: $0.atoms, bonds: $0.bonds, title: $0.title)
                    }

                    preparationReport = ProteinPreparation.analyze(
                        atoms: result.protein?.atoms ?? [],
                        bonds: result.protein?.bonds ?? [],
                        waterCount: result.waterCount
                    )

                    log.success("Loaded PDB: \(protein?.atomCount ?? 0) atoms", category: .pdb)

                case .sdf, .mol:
                    let molecules = try await Task.detached { try SDFParser.parse(url: url) }.value

                    if let first = molecules.first {
                        ligand = Molecule(name: first.name, atoms: first.atoms,
                                          bonds: first.bonds, title: first.title)
                        log.success("Loaded \(molecules.count) molecule(s) from SDF", category: .molecule)
                    }

                case .smi:
                    try ligandDB.importSMIFile(url: url, prepare: false)
                    log.success("Imported SMILES file into ligand database", category: .molecule)

                case .csv:
                    try ligandDB.importCSV(url: url, prepare: false)
                    log.success("Imported CSV file into ligand database", category: .molecule)

                case .mol2:
                    let mols = try await Task.detached { try MOL2Parser.parse(url: url) }.value
                    if let first = mols.first {
                        ligand = Molecule(name: first.name, atoms: first.atoms,
                                          bonds: first.bonds, title: first.title)
                        log.success("Loaded \(mols.count) molecule(s) from MOL2", category: .molecule)
                    }

                case .mmcif:
                    let content = try String(contentsOf: url, encoding: .utf8)
                    // Try dedicated mmCIF parser first (C++ bridge), fall back to PDB parser
                    do {
                        let mmcifResult = try await Task.detached { try MMCIFParser.parse(content: content) }.value
                        let mol = Molecule(name: mmcifResult.name, atoms: mmcifResult.atoms,
                                           bonds: mmcifResult.bonds, title: mmcifResult.title)
                        protein = mol
                    } catch {
                        // Fallback to PDB parser for compatibility
                        let result = await Task.detached { PDBParser.parse(content) }.value
                        if let protData = result.protein {
                            let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                               bonds: protData.bonds, title: protData.title)
                            mol.secondaryStructureAssignments = protData.ssRanges.map {
                                (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                            }
                            protein = mol
                        }
                    }
                    log.success("Loaded mmCIF: \(protein?.atomCount ?? 0) atoms", category: .pdb)
                }

                hiddenChainIDs.removeAll()
                pushToRenderer()
                renderer?.fitToContent()
                statusMessage = "\(url.lastPathComponent) loaded"
            } catch {
                log.error("Failed to load file: \(error.localizedDescription)", category: .system)
                statusMessage = "Failed to load file"
            }

            isLoading = false
            loadingMessage = ""
        }
    }

    func importFile() {
        guard let url = FileImportHandler.showOpenPanel() else { return }
        loadFromFile(url: url)
    }

    // MARK: - PDB Search

    func searchPDB(query: String) {
        guard !query.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        isSearching = true
        searchResults = []
        log.info("Searching RCSB for '\(query)'...", category: .network)

        Task {
            do {
                let results = try await PDBService.shared.search(query: query)
                searchResults = results
                log.success("Found \(results.count) results for '\(query)'", category: .network)
            } catch {
                log.error("Search failed: \(error.localizedDescription)", category: .network)
            }
            isSearching = false
        }
    }

    // MARK: - Protein Preparation

    func removeWaters() {
        guard let prot = protein else { return }
        let result = ProteinPreparation.removeWaters(atoms: prot.atoms, bonds: prot.bonds)
        if result.removedCount == 0 {
            log.info("No waters found to remove", category: .prep)
            statusMessage = "No waters to remove"
            return
        }
        protein = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
        pushToRenderer()
        renderer?.fitToContent()
        log.success("Removed \(result.removedCount) water molecules", category: .prep)
        statusMessage = "\(result.removedCount) waters removed"

        preparationReport = ProteinPreparation.analyze(
            atoms: result.atoms, bonds: result.bonds, waterCount: 0
        )
    }

    func removeAltConfs() {
        guard let prot = protein else { return }
        let filtered = prot.atoms.filter { $0.altLoc.isEmpty || $0.altLoc == "A" }
        let removed = prot.atoms.count - filtered.count
        if removed == 0 {
            log.info("No alternate conformations found", category: .prep)
            return
        }

        // Reindex
        var newAtoms: [Atom] = []
        var indexMap: [Int: Int] = [:]
        for (_, atom) in filtered.enumerated() {
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

        protein = Molecule(name: prot.name, atoms: newAtoms, bonds: newBonds, title: prot.title)
        pushToRenderer()
        log.success("Removed \(removed) alternate conformation atoms", category: .prep)
        statusMessage = "\(removed) alt confs removed"

        preparationReport = ProteinPreparation.analyze(atoms: newAtoms, bonds: newBonds)
    }

    func assignProtonation() {
        guard let prot = protein else { return }
        let pH = protonationPH
        let protonated = Protonation.applyProtonation(atoms: prot.atoms, pH: pH)
        let chargedCount = zip(prot.atoms, protonated).filter { $0.formalCharge != $1.formalCharge }.count
        protein = Molecule(name: prot.name, atoms: protonated, bonds: prot.bonds, title: prot.title)
        protein?.secondaryStructureAssignments = prot.secondaryStructureAssignments
        pushToRenderer()
        log.success("Assigned protonation states at pH \(String(format: "%.1f", pH)) (\(chargedCount) residues charged)", category: .prep)
        statusMessage = "Protonation at pH \(String(format: "%.1f", pH))"
    }

    // MARK: - Energy Minimization

    func runEnergyMinimization() {
        guard let prot = protein else { return }

        // NOTE: True protein energy minimization requires AMBER/CHARMM force fields
        // (e.g., via OpenMM). MMFF94 is designed for small molecules and is NOT
        // appropriate for proteins. Instead, we apply protonation + charge assignment
        // as the "structure cleanup" step. Full MM/MD support is planned.
        isMinimizing = true
        log.info("Running protein structure cleanup (protonation + charges)...", category: .prep)
        statusMessage = "Protein cleanup..."

        Task {
            let protonated = Protonation.applyProtonation(atoms: prot.atoms, pH: protonationPH)
            var charged = protonated
            var rdkitMatchedAtoms = 0

            if let pdbContent = rawPDBContent,
               let chargeData = await Task.detached(operation: {
                   RDKitBridge.computeChargesPDB(pdbContent: pdbContent)
               }).value {
                let merged = mergeProteinAtoms(currentAtoms: protonated, sourceAtoms: chargeData.atoms) { current, source in
                    current.charge = source.charge
                }
                charged = merged.atoms
                rdkitMatchedAtoms = merged.matchedCount
            }

            charged = applyElectrostaticFallback(to: charged)

            let mol = Molecule(name: prot.name, atoms: charged, bonds: prot.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            protein = mol
            pushToRenderer()
            renderer?.fitToContent()
            preparationReport = ProteinPreparation.analyze(atoms: charged, bonds: prot.bonds)

            let chargedCount = zip(prot.atoms, charged).filter { $0.formalCharge != $1.formalCharge }.count
            if rdkitMatchedAtoms > 0 {
                log.success("Structure cleanup complete (pH \(String(format: "%.1f", protonationPH)), \(chargedCount) ionization updates, RDKit charges merged onto \(rdkitMatchedAtoms) atoms)", category: .prep)
            } else {
                log.success("Structure cleanup complete (pH \(String(format: "%.1f", protonationPH)), \(chargedCount) ionization updates, formal-charge fallback applied)", category: .prep)
            }
            statusMessage = "Cleanup complete"
            isMinimizing = false
        }
    }

    // MARK: - Fix Missing Residues (Detection)

    func detectAndReportMissingResidues() {
        guard let prot = protein else { return }

        let gaps = ProteinPreparation.detectMissingResidues(in: prot.atoms)
        if gaps.isEmpty {
            log.success("No missing residues detected -- sequence is contiguous", category: .prep)
            statusMessage = "No gaps found"
        } else {
            log.warn("Detected \(gaps.count) gap(s) in residue numbering:", category: .prep)
            for gap in gaps {
                let count = gap.gapEnd - gap.gapStart + 1
                log.warn("  Chain \(gap.chainID): residues \(gap.gapStart)-\(gap.gapEnd) missing (\(count) residue\(count == 1 ? "" : "s"))", category: .prep)
            }
            log.info("Actual modeling of missing residues requires homology tools (not yet available)", category: .prep)
            statusMessage = "\(gaps.count) gap(s) detected"
        }

        // Update the report if we have one
        if preparationReport != nil {
            preparationReport?.missingResidues = gaps
        }
    }

    // MARK: - Solvation Shell

    func addSolvationShell() {
        log.info("Solvation shell not yet implemented -- use explicit water placement in a future release", category: .prep)
        log.info("For binding site analysis, consider using the pocket detection tools to identify solvent-exposed regions", category: .prep)
        statusMessage = "Solvation: planned for future release"
    }

    // MARK: - C++ Protein Preparation

    func addHydrogens() {
        guard let prot = protein else { return }

        log.info("Adding polar hydrogens...", category: .prep)
        statusMessage = "Adding hydrogens..."

        let existingHCount = prot.atoms.filter { $0.element == .H }.count
        if existingHCount > 0 {
            log.warn("Protein already has \(existingHCount) hydrogens — skipping", category: .prep)
            statusMessage = "Hydrogens already present"
            return
        }

        Task {
            let rdkitHydrogenated: MoleculeData? = if let pdbContent = rawPDBContent {
                await Task.detached {
                    RDKitBridge.addHydrogensToPDB(pdbContent: pdbContent)
                }.value
            } else {
                nil
            }

            let result: (atoms: [Atom], bonds: [Bond], method: String)
            if let hydrogenated = rdkitHydrogenated,
               canAdoptRDKitProteinGeometry(currentAtoms: prot.atoms, sourceAtoms: hydrogenated.atoms) {
                result = (hydrogenated.atoms, hydrogenated.bonds, "RDKit")
            } else {
                let atoms = prot.atoms
                let bonds = prot.bonds
                let fallback = await Task.detached {
                    ProteinPreparation.addPolarHydrogens(atoms: atoms, bonds: bonds)
                }.value
                result = (fallback.atoms, fallback.bonds, "geometry fallback")
            }

            let mol = Molecule(name: prot.name, atoms: result.atoms, bonds: result.bonds, title: prot.title)
            mol.secondaryStructureAssignments = prot.secondaryStructureAssignments
            protein = mol
            pushToRenderer()
            renderer?.fitToContent()

            let added = result.atoms.count - prot.atoms.count
            log.success("Added \(added) hydrogens via \(result.method)", category: .prep)
            statusMessage = "\(added) hydrogens added"
            preparationReport = ProteinPreparation.analyze(atoms: result.atoms, bonds: result.bonds)
        }
    }

    func assignGasteigerCharges() {
        guard let prot = protein else { return }
        log.info("Computing Gasteiger charges via RDKit...", category: .prep)
        statusMessage = "Computing charges..."

        guard let pdbContent = rawPDBContent else {
            log.warn("No raw PDB content available", category: .prep)
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
                protein = Molecule(name: prot.name, atoms: updated, bonds: prot.bonds, title: prot.title)
                protein?.secondaryStructureAssignments = prot.secondaryStructureAssignments
                pushToRenderer()
                let charged = updated.filter { abs($0.charge) > 0.001 }.count
                log.success("Assigned Gasteiger charges to \(charged) atoms (\(merged.matchedCount) matched by residue identity)", category: .prep)
                statusMessage = "Charges assigned"
            } else {
                log.error("Failed to compute charges", category: .prep)
            }
        }
    }

    // MARK: - Pocket Detection

    func detectPockets() {
        guard let prot = protein else { return }
        log.info("Detecting binding pockets...", category: .dock)
        statusMessage = "Detecting pockets..."

        Task {
            let pockets = BindingSiteDetector.detectPockets(protein: prot)
            detectedPockets = pockets
            if let best = pockets.first {
                selectedPocket = best
                showGridBoxForPocket(best)
                log.success("Found \(pockets.count) pocket(s), best: \(String(format: "%.0f", best.volume)) ų, druggability: \(String(format: "%.1f", best.druggability))", category: .dock)
            } else {
                renderer?.clearGridBox()
                log.warn("No pockets detected", category: .dock)
            }
            statusMessage = "\(pockets.count) pocket(s) found"
        }
    }

    func detectLigandGuidedPocket() {
        guard let prot = protein, let lig = ligand else { return }
        log.info("Defining pocket from ligand position...", category: .dock)

        if let pocket = BindingSiteDetector.ligandGuidedPocket(protein: prot, ligand: lig) {
            detectedPockets = [pocket]
            selectedPocket = pocket
            showGridBoxForPocket(pocket)
            log.success("Ligand-guided pocket: \(String(format: "%.0f", pocket.volume)) ų, \(pocket.residueIndices.count) residues", category: .dock)
        }
    }

    func pocketFromSelection() {
        guard let prot = protein, !selectedResidueIndices.isEmpty else { return }
        let pocket = BindingSiteDetector.pocketFromResidues(
            protein: prot, residueIndices: Array(selectedResidueIndices)
        )
        detectedPockets = [pocket]
        selectedPocket = pocket
        showGridBoxForPocket(pocket)
        log.success("Manual pocket from \(selectedResidueIndices.count) residues", category: .dock)
    }

    /// Show the grid box wireframe for a pocket in the viewport
    func showGridBoxForPocket(_ pocket: BindingPocket) {
        renderer?.updateGridBox(
            center: pocket.center,
            halfSize: pocket.size,
            color: SIMD4<Float>(0.2, 1.0, 0.4, 0.6)
        )
    }

    /// Update the grid box visualization with custom parameters
    func updateGridBoxVisualization(center: SIMD3<Float>, halfSize: SIMD3<Float>) {
        renderer?.updateGridBox(
            center: center,
            halfSize: halfSize,
            color: SIMD4<Float>(0.2, 1.0, 0.4, 0.6)
        )
    }

    // MARK: - Docking

    func runDocking() {
        guard let pocket = selectedPocket,
              let prot = protein,
              let lig = ligand
        else {
            log.error("Need protein, ligand, and pocket to dock", category: .dock)
            return
        }

        // Save original ligand BEFORE any pose transforms mutate self.ligand.
        // This is critical: live updates and showDockingPose must always transform
        // from the original coordinates, not from an already-transformed version.
        originalDockingLigand = Molecule(name: lig.name, atoms: lig.atoms, bonds: lig.bonds, title: lig.title)

        isDocking = true
        dockingGeneration = 0
        dockingTotalGenerations = dockingConfig.numGenerations
        dockingResults = []
        currentInteractions = []
        dockingBestEnergy = .infinity
        log.info("Starting docking: \(dockingConfig.populationSize) pop, \(dockingConfig.numGenerations) gen", category: .dock)

        Task {
            if dockingEngine == nil, let device = renderer?.device {
                dockingEngine = DockingEngine(device: device)
            }
            guard let engine = dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                isDocking = false
                dockingBestEnergy = .infinity
                dockingResults = []
                statusMessage = "Docking engine unavailable"
                return
            }

            let proteinAtoms = prot.atoms
            let proteinBonds = prot.bonds
            let dockingPH = protonationPH
            let pdbContent = rawPDBContent
            let preparedProtein = await Task.detached {
                ProteinPreparation.prepareForDocking(
                    atoms: proteinAtoms,
                    bonds: proteinBonds,
                    rawPDBContent: pdbContent,
                    pH: dockingPH
                )
            }.value

            let scoringProtein = Molecule(
                name: prot.name,
                atoms: preparedProtein.atoms,
                bonds: preparedProtein.bonds,
                title: prot.title,
                smiles: prot.smiles
            )
            scoringProtein.secondaryStructureAssignments = prot.secondaryStructureAssignments
            log.info(
                "Using prepared receptor for scoring: +\(preparedProtein.report.hydrogensAdded) H via \(preparedProtein.report.hydrogenMethod), " +
                "\(preparedProtein.report.nonZeroChargeAtoms) charged atoms, \(preparedProtein.report.rdkitChargeMatches) RDKit charge matches",
                category: .dock
            )

            // Compute grid maps
            log.info("Computing grid maps...", category: .dock)
            engine.computeGridMaps(protein: scoringProtein, pocket: pocket, spacing: dockingConfig.gridSpacing)
            log.success("Grid maps computed", category: .dock)

            // Capture the original ligand for all callbacks (not self.ligand which gets mutated)
            let origLig = self.originalDockingLigand ?? lig

            // Live visualization callbacks
            engine.onPoseUpdate = { [weak self] result, interactions in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.dockingBestEnergy = result.energy
                    self.currentInteractions = interactions
                    self.applyDockingPose(result, originalLigand: origLig)
                }
            }

            engine.onGenerationComplete = { [weak self] gen, energy in
                Task { @MainActor [weak self] in
                    self?.dockingGeneration = gen
                }
            }

            // Run GA
            let results = await engine.runDocking(ligand: lig, pocket: pocket, config: dockingConfig)
            // Optional ML re-ranking with DruseScore
            var rankedResults: [DockingResult]
            if useDruseScoreReranking && druseScore.isAvailable, let pocket = selectedPocket {
                log.info("Re-ranking with DruseScore ML...", category: .dock)
                let reranked = await druseScore.rerankPoses(
                    results: results,
                    proteinAtoms: scoringProtein.atoms,
                    ligandAtoms: origLig.atoms,
                    pocketCenter: pocket.center
                )
                rankedResults = reranked
                log.success("ML re-ranking complete", category: .dock)
            } else {
                rankedResults = results
            }

            let openMMRefiner = OpenMMPocketRefiner.shared
            if openMMRefiner.isAvailable {
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
                    protein = refinedProtein
                    pushToRenderer()
                }
            } else {
                log.info("OpenMM refinement skipped: \(openMMRefiner.availabilitySummary)", category: .dock)
            }

            dockingResults = rankedResults
            isDocking = false

            // Apply best pose from final results
            if let best = dockingResults.first {
                applyDockingPose(best, originalLigand: origLig)

                // Detect interactions for best pose
                if protein != nil {
                    let heavyAtoms = origLig.atoms.filter { $0.element != .H }
                    let heavyBonds = buildHeavyBonds(from: origLig)
                    currentInteractions = InteractionDetector.detect(
                        ligandAtoms: heavyAtoms,
                        ligandPositions: best.transformedAtomPositions,
                        proteinAtoms: scoringProtein.atoms.filter { $0.element != .H },
                        ligandBonds: heavyBonds
                    )
                    renderer?.updateInteractionLines(currentInteractions)
                }
            }

            let clusterCount = Set(results.map(\.clusterID)).count
            if let best = results.first {
                log.success(String(format: "Docking complete: best %.1f kcal/mol, %d clusters",
                                   best.energy, clusterCount), category: .dock)
                statusMessage = String(format: "Best: %.1f kcal/mol", best.energy)
            }
        }
    }

    func stopDocking() {
        dockingEngine?.stopDocking()
        isDocking = false
        renderer?.clearGhostPose()
        log.info("Docking stopped", category: .dock)
    }

    func showDockingPose(at index: Int) {
        guard index < dockingResults.count else { return }
        // Always use the original (pre-docking) ligand for transforms, not the
        // mutated self.ligand which has been replaced by a previous pose application.
        guard let origLig = originalDockingLigand ?? ligand else { return }
        let result = dockingResults[index]
        applyDockingPose(result, originalLigand: origLig)

        // Detect interactions for this pose using original heavy atoms + transformed positions
        if let prot = protein {
            let heavyAtoms = origLig.atoms.filter { $0.element != .H }
            let heavyBonds = buildHeavyBonds(from: origLig)
            currentInteractions = InteractionDetector.detect(
                ligandAtoms: heavyAtoms,
                ligandPositions: result.transformedAtomPositions,
                proteinAtoms: prot.atoms.filter { $0.element != .H },
                ligandBonds: heavyBonds
            )
            renderer?.updateInteractionLines(currentInteractions)
        }

        statusMessage = String(format: "Pose #%d: %.1f kcal/mol", index + 1, result.energy)
    }

    private func refineTopDockingResultsWithOpenMM(
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

    /// Apply a docking pose for display.
    /// During live docking (isDocking == true), renders as a translucent ghost overlay
    /// without mutating the active ligand. On final pose display, snaps the ligand.
    private func applyDockingPose(_ result: DockingResult, originalLigand: Molecule) {
        guard !result.transformedAtomPositions.isEmpty else { return }
        let heavyAtoms = originalLigand.atoms.filter { $0.element != .H }

        var newAtoms: [Atom] = []
        for (i, atom) in heavyAtoms.enumerated() {
            guard i < result.transformedAtomPositions.count else { break }
            newAtoms.append(Atom(
                id: i, element: atom.element, position: result.transformedAtomPositions[i],
                name: atom.name, residueName: atom.residueName, residueSeq: atom.residueSeq,
                chainID: atom.chainID, charge: atom.charge, formalCharge: atom.formalCharge,
                isHetAtom: atom.isHetAtom
            ))
        }

        // Rebuild bonds for heavy atoms only
        var oldToNew: [Int: Int] = [:]
        for (newIdx, atom) in heavyAtoms.enumerated() {
            oldToNew[atom.id] = newIdx
        }
        var newBonds: [Bond] = []
        for bond in originalLigand.bonds {
            if let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] {
                newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
            }
        }

        if isDocking {
            // During live docking: render as translucent ghost, don't mutate self.ligand
            renderer?.updateGhostPose(atoms: newAtoms, bonds: newBonds)
        } else {
            // Final pose display: snap the ligand and clear ghost
            renderer?.clearGhostPose()
            ligand = Molecule(name: originalLigand.name, atoms: newAtoms, bonds: newBonds, title: originalLigand.title)
            pushToRenderer()
        }

        // Update interaction lines
        if !currentInteractions.isEmpty {
            renderer?.updateInteractionLines(currentInteractions)
        }
    }

    // MARK: - Export Results

    func exportResultsCSV() -> String {
        var csv = "Rank,Cluster,Energy,VdW,Elec,HBond,Desolv,Generation\n"
        for (i, r) in dockingResults.enumerated() {
            csv += "\(i+1),\(r.clusterID),\(String(format: "%.2f", r.energy)),"
            csv += "\(String(format: "%.2f", r.vdwEnergy)),\(String(format: "%.2f", r.elecEnergy)),"
            csv += "\(String(format: "%.2f", r.hbondEnergy)),\(String(format: "%.2f", r.desolvEnergy)),"
            csv += "\(r.generation)\n"
        }
        return csv
    }

    // MARK: - Box Selection

    /// Handle box (lasso) selection from shift+drag in the viewport.
    /// `atomIndices` are renderer-space IDs. `addToExisting` unions with current selection.
    func handleBoxSelection(atomIndices: [Int], addToExisting: Bool) {
        if !addToExisting {
            selectedAtomIndices.removeAll()
            selectedResidueIndices.removeAll()
        }
        selectedAtomIndices.formUnion(atomIndices)

        // Also select the residues these atoms belong to
        for idx in atomIndices {
            if let prot = protein, let resIdx = prot.residueIndex(forAtom: idx) {
                selectedResidueIndices.insert(resIdx)
            }
            if let lig = ligand {
                let adjustedIdx = idx - (protein?.atoms.count ?? 0)
                if adjustedIdx >= 0, let resIdx = lig.residueIndex(forAtom: adjustedIdx) {
                    selectedResidueIndices.insert(resIdx)
                }
            }
        }

        selectedAtomIndex = atomIndices.first
        pushToRenderer()

        let count = selectedAtomIndices.count
        log.info("Selected \(count) atom\(count == 1 ? "" : "s") by box selection", category: .molecule)
        statusMessage = "\(count) atom\(count == 1 ? "" : "s") selected"
    }

    // MARK: - Heavy Atom Bond Remapping

    /// Build bonds for heavy atoms only (excludes hydrogens), with remapped indices.
    private func buildHeavyBonds(from molecule: Molecule) -> [Bond] {
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

    // MARK: - Molecular Surface

    func toggleSurface() {
        showSurface.toggle()
        if showSurface {
            generateSurface()
        } else {
            renderer?.clearSurfaceMesh()
        }
    }

    func toggleESPColoring() {
        surfaceColorByESP.toggle()
        if showSurface {
            generateSurface()
        }
    }

    func generateSurface() {
        guard let prot = protein, let renderer else { return }
        isGeneratingSurface = true
        statusMessage = "Generating surface..."

        Task {
            if surfaceGenerator == nil {
                surfaceGenerator = try? SurfaceGenerator(
                    device: renderer.device,
                    commandQueue: renderer.commandQueue
                )
            }
            guard let gen = surfaceGenerator else {
                isGeneratingSurface = false
                log.error("Failed to create surface generator", category: .molecule)
                return
            }

            gen.fieldType = surfaceType
            gen.colorByESP = surfaceColorByESP

            let atoms = prot.atoms.filter { $0.element != .H }
            if let result = gen.generateSurface(atoms: atoms) {
                renderer.updateSurfaceMesh(result)
                log.success("Surface: \(result.vertexCount) vertices, \(result.indexCount / 3) triangles", category: .molecule)
                statusMessage = "Surface generated"
            } else {
                log.error("Surface generation failed", category: .molecule)
                statusMessage = "Surface failed"
            }
            isGeneratingSurface = false
        }
    }

    // MARK: - ML Model Loading

    func loadMLModels() {
        druseScore.loadModel()
        pocketDetectorML.loadModel()
        admetPredictor.loadModels()
        if druseScore.isAvailable {
            log.success("DruseScore ML model loaded", category: .dock)
        }
        if pocketDetectorML.isAvailable {
            log.success("PocketDetector ML model loaded", category: .dock)
        }
        if admetPredictor.isAvailable {
            log.success("ADMET models loaded", category: .dock)
        }
    }

    // MARK: - ML Pocket Detection (alternative to alpha-sphere)

    func detectPocketsML() {
        guard let prot = protein else { return }
        guard pocketDetectorML.isAvailable else {
            log.warn("PocketDetector model not available, falling back to geometric detection", category: .dock)
            detectPockets()
            return
        }
        log.info("Detecting pockets (ML)...", category: .dock)
        statusMessage = "ML pocket detection..."

        Task {
            let pockets = await pocketDetectorML.detectPockets(protein: prot)
            detectedPockets = pockets
            if let best = pockets.first {
                selectedPocket = best
                showGridBoxForPocket(best)
                log.success("ML found \(pockets.count) pocket(s), best: \(String(format: "%.0f", best.volume)) ų", category: .dock)
            } else {
                renderer?.clearGridBox()
                log.warn("ML detected no pockets, try geometric method", category: .dock)
            }
            statusMessage = "\(pockets.count) pocket(s) found (ML)"
        }
    }

    // MARK: - Context Menu Actions

    /// Select all atoms belonging to a given chain ID.
    func selectChain(_ chainID: String) {
        selectedAtomIndices.removeAll()
        selectedResidueIndices.removeAll()

        if let prot = protein {
            let indices = prot.atomIndices(forChainID: chainID)
            selectedAtomIndices.formUnion(indices)
            if let chain = prot.chains.first(where: { $0.id == chainID }) {
                selectedResidueIndices.formUnion(chain.residueIndices)
            }
        }
        if let lig = ligand {
            let offset = protein?.atoms.count ?? 0
            let indices = lig.atomIndices(forChainID: chainID).map { $0 + offset }
            selectedAtomIndices.formUnion(indices)
        }

        selectedAtomIndex = selectedAtomIndices.first
        pushToRenderer()
        log.info("Selected chain \(chainID) (\(selectedAtomIndices.count) atoms)", category: .molecule)
        statusMessage = "Chain \(chainID) selected"
    }

    /// Extend current selection to include all residues within a given distance (Angstroms)
    /// of any currently selected atom. Works with both protein and ligand atom indices.
    func selectResiduesWithinDistance(_ distance: Float) {
        guard let prot = protein else { return }
        let protAtoms = prot.atoms
        let combinedAtoms: [Atom] = protAtoms + (ligand?.atoms ?? [])

        // Gather positions of currently selected atoms (may include both protein and ligand)
        var selectedPositions: [SIMD3<Float>] = []
        for idx in selectedAtomIndices {
            if idx < combinedAtoms.count {
                selectedPositions.append(combinedAtoms[idx].position)
            }
        }
        // Also include positions from selected residues' atoms (handles residue-only selection)
        for resIdx in selectedResidueIndices {
            guard resIdx < prot.residues.count else { continue }
            for atomIdx in prot.residues[resIdx].atomIndices {
                if atomIdx < protAtoms.count {
                    selectedPositions.append(protAtoms[atomIdx].position)
                }
            }
        }
        guard !selectedPositions.isEmpty else { return }

        let distSq = distance * distance
        var newResidues: Set<Int> = selectedResidueIndices

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

        let addedCount = newResidues.count - selectedResidueIndices.count
        selectedResidueIndices = newResidues

        // Add all atom indices from newly selected residues
        for resIdx in newResidues {
            guard resIdx < prot.residues.count else { continue }
            selectedAtomIndices.formUnion(prot.residues[resIdx].atomIndices)
        }

        pushToRenderer()
        log.info("Extended selection by \(addedCount) residues within \(String(format: "%.1f", distance)) \u{00C5}", category: .molecule)
        statusMessage = "\(selectedResidueIndices.count) residues selected"
    }

    /// Create a binding pocket from the currently selected residues.
    func definePocketFromSelection() {
        guard let prot = protein, !selectedResidueIndices.isEmpty else {
            log.warn("Select residues first to define a pocket", category: .dock)
            return
        }
        let pocket = BindingSiteDetector.pocketFromResidues(
            protein: prot, residueIndices: Array(selectedResidueIndices)
        )
        detectedPockets = [pocket]
        selectedPocket = pocket
        showGridBoxForPocket(pocket)
        log.success("Pocket defined from \(selectedResidueIndices.count) selected residues", category: .dock)
        statusMessage = "Pocket defined"
    }

    /// Hide the currently selected atoms from the viewport.
    func hideSelection() {
        guard !selectedAtomIndices.isEmpty else { return }
        hiddenAtomIndices.formUnion(selectedAtomIndices)
        let count = selectedAtomIndices.count
        selectedAtomIndices.removeAll()
        selectedResidueIndices.removeAll()
        selectedAtomIndex = nil
        pushToRenderer()
        log.info("Hidden \(count) atoms", category: .molecule)
        statusMessage = "\(count) atoms hidden"
    }

    /// Show all hidden atoms again.
    func showAllAtoms() {
        let count = hiddenAtomIndices.count
        hiddenAtomIndices.removeAll()
        pushToRenderer()
        if count > 0 {
            log.info("Showing all atoms (\(count) were hidden)", category: .molecule)
        }
        statusMessage = "All atoms visible"
    }

    /// Remove the currently selected atoms/residues from the molecule.
    func removeSelection() {
        guard !selectedAtomIndices.isEmpty else { return }
        let removedCount = selectedAtomIndices.count

        // Remove from protein
        if let prot = protein {
            let keepAtoms = prot.atoms.filter { !selectedAtomIndices.contains($0.id) }
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
                protein = Molecule(name: prot.name, atoms: newAtoms, bonds: newBonds, title: prot.title)
                protein?.secondaryStructureAssignments = prot.secondaryStructureAssignments
            }
        }

        // Remove from ligand (adjust indices by protein offset)
        if let lig = ligand {
            let offset = protein?.atoms.count ?? 0
            let ligRemoveIDs = Set(selectedAtomIndices.compactMap { idx -> Int? in
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
                ligand = Molecule(name: lig.name, atoms: newAtoms, bonds: newBonds, title: lig.title)
            }
        }

        selectedAtomIndices.removeAll()
        selectedResidueIndices.removeAll()
        selectedAtomIndex = nil
        pushToRenderer()
        log.success("Removed \(removedCount) atoms", category: .molecule)
        statusMessage = "\(removedCount) atoms removed"
    }

    /// Center the camera on the centroid of selected atoms.
    func centerOnSelection() {
        let allAtoms = (protein?.atoms ?? []) + (ligand?.atoms ?? [])
        let selectedPositions = selectedAtomIndices.compactMap { idx -> SIMD3<Float>? in
            idx < allAtoms.count ? allAtoms[idx].position : nil
        }
        guard !selectedPositions.isEmpty else { return }
        let c = centroid(selectedPositions)
        renderer?.camera.focusOnPoint(c)
        log.info("Centered on selection", category: .molecule)
    }

    // MARK: - Residue Subsets (MOE-style)

    /// Create a new residue subset from the current selection.
    func createSubsetFromSelection(name: String? = nil) {
        guard let prot = protein, !selectedResidueIndices.isEmpty else {
            log.error("Select residues first to create a subset", category: .molecule)
            return
        }
        let indices = Array(selectedResidueIndices).sorted()
        let subsetName = name ?? "Subset \(residueSubsets.count + 1)"
        let colors: [SIMD4<Float>] = [
            SIMD4(0.2, 0.8, 0.9, 1.0), // cyan
            SIMD4(0.9, 0.5, 0.2, 1.0), // orange
            SIMD4(0.5, 0.9, 0.3, 1.0), // green
            SIMD4(0.9, 0.3, 0.7, 1.0), // pink
            SIMD4(0.6, 0.4, 0.9, 1.0), // purple
        ]
        let color = colors[residueSubsets.count % colors.count]
        let subset = ResidueSubset(name: subsetName, residueIndices: indices, color: color)
        residueSubsets.append(subset)

        let residueNames = indices.prefix(5).map { idx -> String in
            guard idx < prot.residues.count else { return "?" }
            return "\(prot.residues[idx].name)\(prot.residues[idx].sequenceNumber)"
        }.joined(separator: ", ")
        let suffix = indices.count > 5 ? "... (\(indices.count) residues)" : ""
        log.success("Created subset '\(subsetName)': \(residueNames)\(suffix)", category: .molecule)
    }

    /// Delete a residue subset.
    func deleteSubset(id: UUID) {
        residueSubsets.removeAll { $0.id == id }
    }

    /// Toggle visibility of a residue subset.
    func toggleSubsetVisibility(id: UUID) {
        if let idx = residueSubsets.firstIndex(where: { $0.id == id }) {
            residueSubsets[idx].isVisible.toggle()
            pushToRenderer()
        }
    }

    /// Select all atoms in a residue subset (for further operations).
    func selectSubset(id: UUID) {
        guard let prot = protein,
              let subset = residueSubsets.first(where: { $0.id == id })
        else { return }
        selectedResidueIndices = Set(subset.residueIndices)
        selectedAtomIndices = Set(subset.atomIndices(in: prot))
        pushToRenderer()
    }

    /// Define a binding pocket from a residue subset.
    func definePocketFromSubset(id: UUID) {
        guard let prot = protein,
              let subset = residueSubsets.first(where: { $0.id == id })
        else { return }
        Task {
            let pocket = BindingSiteDetector.pocketFromResidues(
                protein: prot, residueIndices: subset.residueIndices
            )
            selectedPocket = pocket
            detectedPockets = [pocket]
            log.success("Defined pocket from subset '\(subset.name)': \(String(format: "%.0f", pocket.volume)) Å³", category: .dock)
        }
    }

    // MARK: - Sequence Editing

    /// Delete the currently selected residues from the protein.
    func deleteSelectedResidues() {
        guard let prot = protein, !selectedResidueIndices.isEmpty else {
            log.warn("Select residues first", category: .molecule)
            return
        }
        let count = selectedResidueIndices.count
        prot.removeResidues(at: selectedResidueIndices)

        // Invalidate residue subsets (indices are now stale)
        residueSubsets.removeAll()

        selectedResidueIndices.removeAll()
        selectedAtomIndices.removeAll()
        selectedAtomIndex = nil
        hiddenAtomIndices.removeAll()
        pushToRenderer()
        log.success("Deleted \(count) residues (\(prot.atoms.count) atoms remaining)", category: .molecule)
        statusMessage = "\(count) residues deleted"
    }

    /// Rename a protein chain.
    func renameChain(from oldID: String, to newID: String) {
        guard let prot = protein else { return }
        prot.renameChain(from: oldID, to: newID)

        // Clear stale selections
        selectedResidueIndices.removeAll()
        selectedAtomIndices.removeAll()
        selectedAtomIndex = nil
        residueSubsets.removeAll()
        pushToRenderer()
        log.success("Renamed chain \(oldID) → \(newID)", category: .molecule)
        statusMessage = "Chain \(oldID) renamed to \(newID)"
    }

    /// Merge source chain into target chain.
    func mergeChains(from sourceID: String, into targetID: String) {
        guard let prot = protein else { return }
        prot.mergeChains(from: sourceID, into: targetID)

        selectedResidueIndices.removeAll()
        selectedAtomIndices.removeAll()
        selectedAtomIndex = nil
        residueSubsets.removeAll()
        pushToRenderer()
        log.success("Merged chain \(sourceID) into \(targetID)", category: .molecule)
        statusMessage = "Chain \(sourceID) merged into \(targetID)"
    }

    /// Copy the one-letter sequence of selected residues (or all if none selected) to clipboard.
    func copySequenceToClipboard(chainID: String? = nil) {
        guard let prot = protein else { return }
        let threeToOne: [String: String] = [
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        ]

        let indices: [Int]
        if !selectedResidueIndices.isEmpty {
            indices = selectedResidueIndices.sorted()
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
        statusMessage = "\(seq.count)-residue sequence copied"
    }

    /// Select all residues with a given secondary structure type.
    func selectBySecondaryStructure(_ ssType: SecondaryStructure, chainID: String? = nil) {
        guard let prot = protein else { return }
        selectedResidueIndices.removeAll()
        selectedAtomIndices.removeAll()

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
                selectedResidueIndices.insert(idx)
                selectedAtomIndices.formUnion(res.atomIndices)
            }
        }
        selectedAtomIndex = selectedAtomIndices.first
        pushToRenderer()
        let ssName = switch ssType { case .helix: "helices"; case .sheet: "sheets"; case .coil: "coils"; case .turn: "turns" }
        statusMessage = "\(selectedResidueIndices.count) \(ssName) selected"
    }

    /// Reset the camera to the default view.
    func resetView() {
        renderer?.camera.reset()
        pushToRenderer()
        renderer?.fitToContent()
        log.info("View reset", category: .molecule)
        statusMessage = "View reset"
    }

    /// Context-aware fit-to-view:
    /// 1. If atoms/residues are selected, center and fit on the selection.
    /// 2. Otherwise, fit to the entire loaded complex (protein + ligand).
    func fitToView() {
        guard let renderer else { return }
        let allAtoms = (protein?.atoms ?? []) + (ligand?.atoms ?? [])

        // If there is a selection, fit to selected atoms
        if !selectedAtomIndices.isEmpty {
            let selectedPositions = selectedAtomIndices.compactMap { idx -> SIMD3<Float>? in
                idx < allAtoms.count ? allAtoms[idx].position : nil
            }
            if !selectedPositions.isEmpty {
                renderer.fitToPositions(selectedPositions)
                log.info("Fit to selection (\(selectedPositions.count) atoms)", category: .molecule)
                return
            }
        }

        // No selection: fit to entire complex
        renderer.fitToContent()
    }

    // MARK: - Context Menu Builder

    /// Build and show the right-click context menu at the given event location.
    func showContextMenu(event: NSEvent, view: NSView) {
        let menu = NSMenu(title: "Context")

        // "Select Chain" submenu
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

        // "Select Residues Within 5 A"
        let extendItem = NSMenuItem(title: "Select Residues Within 5 \u{00C5}", action: #selector(ContextMenuTarget.selectNearbyResiduesAction), keyEquivalent: "")
        extendItem.target = contextMenuTarget
        extendItem.isEnabled = !selectedAtomIndices.isEmpty
        menu.addItem(extendItem)

        // "Define Pocket from Selection"
        let pocketItem = NSMenuItem(title: "Define Pocket from Selection", action: #selector(ContextMenuTarget.definePocketAction), keyEquivalent: "")
        pocketItem.target = contextMenuTarget
        pocketItem.isEnabled = !selectedResidueIndices.isEmpty
        menu.addItem(pocketItem)

        menu.addItem(NSMenuItem.separator())

        // "Hide Selection"
        let hideItem = NSMenuItem(title: "Hide Selection", action: #selector(ContextMenuTarget.hideSelectionAction), keyEquivalent: "")
        hideItem.target = contextMenuTarget
        hideItem.isEnabled = !selectedAtomIndices.isEmpty
        menu.addItem(hideItem)

        // "Show All"
        let showAllItem = NSMenuItem(title: "Show All", action: #selector(ContextMenuTarget.showAllAction), keyEquivalent: "")
        showAllItem.target = contextMenuTarget
        showAllItem.isEnabled = !hiddenAtomIndices.isEmpty
        menu.addItem(showAllItem)

        // "Remove Selection"
        let removeItem = NSMenuItem(title: "Remove Selection", action: #selector(ContextMenuTarget.removeSelectionAction), keyEquivalent: "")
        removeItem.target = contextMenuTarget
        removeItem.isEnabled = !selectedAtomIndices.isEmpty
        menu.addItem(removeItem)

        menu.addItem(NSMenuItem.separator())

        // "Center on Selection"
        let centerItem = NSMenuItem(title: "Center on Selection", action: #selector(ContextMenuTarget.centerOnSelectionAction), keyEquivalent: "")
        centerItem.target = contextMenuTarget
        centerItem.isEnabled = !selectedAtomIndices.isEmpty
        menu.addItem(centerItem)

        // "Reset View"
        let resetItem = NSMenuItem(title: "Reset View", action: #selector(ContextMenuTarget.resetViewAction), keyEquivalent: "")
        resetItem.target = contextMenuTarget
        menu.addItem(resetItem)

        NSMenu.popUpContextMenu(menu, with: event, for: view)
    }

    /// Persistent target object for context menu actions (required for NSMenu selectors).
    private var _contextMenuTarget: ContextMenuTarget?
    private var contextMenuTarget: ContextMenuTarget {
        if let t = _contextMenuTarget { return t }
        let t = ContextMenuTarget(viewModel: self)
        _contextMenuTarget = t
        return t
    }

    // MARK: - Batch Docking

    /// Dock a specific set of ligand entries sequentially.
    /// Queue ligands for batch docking (from Ligand Database multi-select).
    /// Sets the first as active ligand; batch is launched from Docking tab.
    func queueLigandsForBatchDocking(_ entries: [LigandEntry]) {
        let prepared = entries.filter { !$0.atoms.isEmpty }
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in selection — prepare them first", category: .dock)
            return
        }
        batchQueue = prepared
        // Set first ligand as active for preview
        let first = prepared[0]
        let mol = Molecule(name: first.name, atoms: first.atoms, bonds: first.bonds,
                           title: first.smiles, smiles: first.smiles)
        setLigandForDocking(mol)
        log.success("Queued \(prepared.count) ligands for batch docking", category: .dock)
        statusMessage = "\(prepared.count) ligands queued — run from Docking tab"
    }

    /// Dock a specific set of ligand entries sequentially.
    /// Results are combined, sorted by energy, and stored in `batchResults`.
    func dockEntries(_ entries: [LigandEntry]) {
        let prepared = entries.filter { !$0.atoms.isEmpty }
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in selection — prepare them first", category: .dock)
            return
        }
        guard let pocket = selectedPocket, let prot = protein else {
            log.error("Need protein and pocket for batch docking", category: .dock)
            return
        }

        isBatchDocking = true
        batchResults = []
        dockingResults = []
        batchProgress = (0, prepared.count)
        log.info("Starting batch docking: \(prepared.count) ligands", category: .dock)
        statusMessage = "Batch docking 0/\(prepared.count)..."

        batchDockingTask = Task {
            if dockingEngine == nil, let device = renderer?.device {
                dockingEngine = DockingEngine(device: device)
            }
            guard let engine = dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                isBatchDocking = false
                return
            }

            engine.computeGridMaps(protein: prot, pocket: pocket, spacing: dockingConfig.gridSpacing)

            for (i, entry) in prepared.enumerated() {
                guard !Task.isCancelled, isBatchDocking else {
                    log.info("Batch docking cancelled at \(i)/\(prepared.count)", category: .dock)
                    break
                }

                batchProgress = (i, prepared.count)
                statusMessage = "Docking \(i + 1)/\(prepared.count): \(entry.name)..."

                // Swap ligand in viewport WITHOUT resetting protein or camera.
                // This is a lightweight version of setLigandForDocking for batch mode.
                let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds,
                                   title: entry.smiles, smiles: entry.smiles)
                ligand = mol
                originalDockingLigand = mol
                currentInteractions = []
                renderer?.updateInteractionLines([])
                pushToRenderer()  // protein stays, ligand swapped

                // Wire live callbacks so user sees each ligand dock in real-time
                engine.onPoseUpdate = { [weak self] result, interactions in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.dockingBestEnergy = result.energy
                        self.currentInteractions = interactions
                        self.applyDockingPose(result, originalLigand: mol)
                    }
                }
                engine.onGenerationComplete = { [weak self] gen, energy in
                    Task { @MainActor [weak self] in
                        self?.dockingGeneration = gen
                    }
                }

                isDocking = true
                dockingBestEnergy = .infinity
                dockingGeneration = 0
                let results = await engine.runDocking(ligand: mol, pocket: pocket, config: dockingConfig)
                isDocking = false

                batchResults.append((ligandName: entry.name, results: results))

                if let best = results.first {
                    log.info(String(format: "  %@ best: %.1f kcal/mol", entry.name, best.energy), category: .dock)
                    applyDockingPose(best, originalLigand: mol)
                }
            }

            batchProgress = (prepared.count, prepared.count)
            isBatchDocking = false

            batchResults.sort { a, b in
                let bestA = a.results.first?.energy ?? .infinity
                let bestB = b.results.first?.energy ?? .infinity
                return bestA < bestB
            }

            // Show the winning ligand in the viewport
            if let bestBatch = batchResults.first, let bestResult = bestBatch.results.first {
                if let bestEntry = prepared.first(where: { $0.name == bestBatch.ligandName }) {
                    let mol = Molecule(name: bestEntry.name, atoms: bestEntry.atoms,
                                       bonds: bestEntry.bonds, title: bestEntry.smiles, smiles: bestEntry.smiles)
                    ligand = mol
                    originalDockingLigand = mol
                    applyDockingPose(bestResult, originalLigand: mol)
                }
                dockingResults = bestBatch.results
            }

            log.success("Batch complete: \(batchResults.count) ligands, best: \(batchResults.first?.ligandName ?? "?") (\(String(format: "%.1f", batchResults.first?.results.first?.energy ?? 0)) kcal/mol)", category: .dock)
            statusMessage = "Batch complete — \(batchResults.count) ligands ranked"
        }
    }

    /// Dock every prepared ligand in the ligand database against the current protein/pocket.
    func dockAllFromDatabase() {
        guard let pocket = selectedPocket,
              let prot = protein else {
            log.error("Need protein and pocket for batch docking", category: .dock)
            return
        }

        let prepared = ligandDB.entries.filter(\.isPrepared)
        guard !prepared.isEmpty else {
            log.warn("No prepared ligands in database", category: .dock)
            return
        }

        isBatchDocking = true
        batchResults = []
        batchProgress = (0, prepared.count)
        log.info("Starting batch docking: \(prepared.count) ligands", category: .dock)
        statusMessage = "Batch docking 0/\(prepared.count)..."

        batchDockingTask = Task {
            if dockingEngine == nil, let device = renderer?.device {
                dockingEngine = DockingEngine(device: device)
            }
            guard let engine = dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                isBatchDocking = false
                return
            }

            // Compute grid maps once for the protein/pocket
            engine.computeGridMaps(protein: prot, pocket: pocket, spacing: dockingConfig.gridSpacing)

            for (i, entry) in prepared.enumerated() {
                // Check for cancellation
                guard !Task.isCancelled, isBatchDocking else {
                    log.info("Batch docking cancelled at \(i)/\(prepared.count)", category: .dock)
                    break
                }

                batchProgress = (i, prepared.count)
                statusMessage = "Batch docking \(i + 1)/\(prepared.count): \(entry.name)..."

                // Build a Molecule from the entry
                let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)

                // Suppress live visualization during batch
                engine.onPoseUpdate = nil
                engine.onGenerationComplete = nil

                let results = await engine.runDocking(ligand: mol, pocket: pocket, config: dockingConfig)
                batchResults.append((ligandName: entry.name, results: results))

                if let best = results.first {
                    log.info(String(format: "  %@ best: %.1f kcal/mol", entry.name, best.energy), category: .dock)
                }
            }

            batchProgress = (prepared.count, prepared.count)
            isBatchDocking = false

            // Sort batch results by best energy
            batchResults.sort { a, b in
                let bestA = a.results.first?.energy ?? .infinity
                let bestB = b.results.first?.energy ?? .infinity
                return bestA < bestB
            }

            log.success("Batch docking complete: \(batchResults.count) ligands docked", category: .dock)
            statusMessage = "Batch docking complete"
        }
    }

    /// Cancel an ongoing batch docking run.
    func cancelBatchDocking() {
        batchDockingTask?.cancel()
        dockingEngine?.stopDocking()
        isBatchDocking = false
        log.info("Batch docking cancelled", category: .dock)
        statusMessage = "Batch docking cancelled"
    }

    // MARK: - Virtual Screening

    /// Start virtual screening on the given library (typically from ligandDB entries).
    /// If `screeningPipeline` is already set (e.g. configured by the UI), it is reused.
    func startScreening(library: [(name: String, smiles: String)]) {
        guard let pocket = selectedPocket, let prot = protein else {
            log.error("Need protein and pocket for virtual screening", category: .dock)
            return
        }

        let pipeline = screeningPipeline ?? VirtualScreeningPipeline()
        screeningPipeline = pipeline
        screeningState = .preparing(current: 0, total: library.count)
        screeningProgress = 0
        screeningHits = []

        log.info("Starting virtual screening: \(library.count) molecules", category: .dock)
        statusMessage = "Virtual screening..."

        // Progress polling task: sync pipeline state to viewmodel every 0.5s
        let progressTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled else { break }
                screeningState = pipeline.state
                screeningProgress = pipeline.progress
            }
        }

        screeningTask = Task {
            if dockingEngine == nil, let device = renderer?.device {
                dockingEngine = DockingEngine(device: device)
            }
            guard let engine = dockingEngine else {
                log.error("Failed to initialize docking engine", category: .dock)
                screeningState = .failed("Docking engine unavailable")
                progressTask.cancel()
                return
            }

            await pipeline.screen(
                smilesLibrary: library,
                dockingEngine: engine,
                protein: prot,
                pocket: pocket,
                mlScorer: pipeline.config.useMLReranking && druseScore.isAvailable ? druseScore : nil,
                admetPredictor: pipeline.config.admetFilter && admetPredictor.isAvailable ? admetPredictor : nil
            )

            progressTask.cancel()

            screeningState = pipeline.state
            screeningProgress = pipeline.progress
            screeningHits = pipeline.hits

            if case .complete(let hits, let total) = pipeline.state {
                log.success("Screening complete: \(hits) hits from \(total) molecules", category: .dock)
                statusMessage = "Screening: \(hits) hits"
            }
        }
    }

    /// Cancel an ongoing virtual screening run.
    func cancelScreening() {
        screeningPipeline?.cancel()
        screeningTask?.cancel()
        screeningState = .idle
        screeningProgress = 0
        log.info("Virtual screening cancelled", category: .dock)
        statusMessage = "Screening cancelled"
    }

    private struct ProteinAtomIdentity: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let atomName: String
        let atomicNumber: Int
        let isHetAtom: Bool
    }

    private func proteinAtomIdentity(for atom: Atom) -> ProteinAtomIdentity {
        ProteinAtomIdentity(
            chainID: atom.chainID,
            residueSeq: atom.residueSeq,
            residueName: atom.residueName,
            atomName: atom.name.trimmingCharacters(in: .whitespaces),
            atomicNumber: atom.element.rawValue,
            isHetAtom: atom.isHetAtom
        )
    }

    private func mergeProteinAtoms(
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

    private func applyElectrostaticFallback(to atoms: [Atom]) -> [Atom] {
        atoms.map { atom in
            var atom = atom
            if abs(atom.charge) <= 0.0001 {
                atom.charge = Float(atom.formalCharge)
            }
            return atom
        }
    }

    private func canAdoptRDKitProteinGeometry(currentAtoms: [Atom], sourceAtoms: [Atom]) -> Bool {
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

    /// Export docking results as multi-mol SDF via a save panel.
    func exportDockingResultsSDF() {
        guard !dockingResults.isEmpty, let lig = ligand else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "sdf")].compactMap { $0 }
        panel.nameFieldStringValue = "docking_poses.sdf"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        var sdf = ""
        for (i, result) in dockingResults.prefix(50).enumerated() {
            let heavyAtoms = lig.atoms.filter { $0.element != .H }
            var poseAtoms: [Atom] = []
            var idMap: [Int: Int] = [:]
            for (j, atom) in heavyAtoms.enumerated() {
                guard j < result.transformedAtomPositions.count else { break }
                idMap[atom.id] = poseAtoms.count
                poseAtoms.append(Atom(
                    id: poseAtoms.count, element: atom.element,
                    position: result.transformedAtomPositions[j],
                    name: atom.name, residueName: atom.residueName,
                    residueSeq: atom.residueSeq, chainID: atom.chainID,
                    charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom
                ))
            }
            var poseBonds: [Bond] = []
            for bond in lig.bonds {
                if let a = idMap[bond.atomIndex1], let b = idMap[bond.atomIndex2] {
                    poseBonds.append(Bond(id: poseBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
            let props: [String: String] = [
                "Rank": "\(i + 1)",
                "Energy": String(format: "%.3f", result.energy),
                "VdW": String(format: "%.3f", result.vdwEnergy),
                "Elec": String(format: "%.3f", result.elecEnergy),
                "HBond": String(format: "%.3f", result.hbondEnergy),
                "Cluster": "\(result.clusterID)"
            ]
            sdf += SDFWriter.molBlock(name: "\(lig.name)_pose\(i + 1)", atoms: poseAtoms,
                                       bonds: poseBonds, properties: props)
        }

        do {
            try SDFWriter.save(sdf, to: url)
            log.success("Exported \(min(dockingResults.count, 50)) poses to SDF", category: .dock)
        } catch {
            log.error("SDF export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    /// Export docking results as CSV via a save panel.
    func exportDockingResultsCSV() {
        guard !dockingResults.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "csv")].compactMap { $0 }
        panel.nameFieldStringValue = "docking_results.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let csv = exportResultsCSV()
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            log.success("Exported docking results to CSV", category: .dock)
        } catch {
            log.error("CSV export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    /// Export screening hits via a save panel (CSV + SDF).
    func exportScreeningHits() {
        guard let pipeline = screeningPipeline, !screeningHits.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "csv")].compactMap { $0 }
        panel.nameFieldStringValue = "screening_hits.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let csv = pipeline.exportCSV()
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            log.success("Exported \(screeningHits.count) screening hits to CSV", category: .dock)

            let sdfURL = url.deletingPathExtension().appendingPathExtension("sdf")
            let sdf = pipeline.exportSDF(topN: 100)
            try sdf.write(to: sdfURL, atomically: true, encoding: .utf8)
            log.success("Exported top 100 hits to SDF", category: .dock)
        } catch {
            log.error("Export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    // MARK: - Analog Generation

    /// Generate simple structural analogs of a reference molecule and add to the ligand database.
    func generateAnalogs(
        referenceSmiles: String,
        referenceName: String,
        count: Int,
        similarityThreshold: Float,
        keepScaffold: Bool
    ) {
        isGeneratingAnalogs = true
        analogGenerationProgress = 0

        log.info("Generating up to \(count) analogs of \(referenceName)...", category: .molecule)
        statusMessage = "Generating analogs..."

        Task {
            defer { self.isGeneratingAnalogs = false }
            let replacements: [(pattern: String, replacement: String, label: String)] = [
                ("F", "Cl", "F_Cl"),
                ("F", "Br", "F_Br"),
                ("Cl", "F", "Cl_F"),
                ("Cl", "Br", "Cl_Br"),
                ("Br", "Cl", "Br_Cl"),
                ("C", "CC", "Me_Et"),
                ("CC", "CCC", "Et_Pr"),
                ("CC", "C(C)C", "Et_iPr"),
                ("O", "N", "OH_NH2"),
                ("N", "O", "NH2_OH"),
                ("O", "S", "OH_SH"),
                ("C", "C(F)(F)F", "Me_CF3"),
                ("C(=O)O", "C(=O)N", "COOH_CONH2"),
                ("C(=O)N", "C(=O)O", "CONH2_COOH"),
                ("c1ccccc1", "c1ccncc1", "Ph_Pyr"),
                ("c1ccncc1", "c1ccccc1", "Pyr_Ph"),
                ("C(=O)", "S(=O)(=O)", "CO_SO2"),
                ("COC", "CSC", "OMe_SMe"),
                ("CSC", "COC", "SMe_OMe"),
                ("C1CCCC1", "C1CCCCC1", "cPent_cHex"),
                ("C1CCCCC1", "C1CCCC1", "cHex_cPent"),
            ]

            var generated: [(smiles: String, name: String)] = []
            let total = Float(replacements.count)

            for (i, rep) in replacements.enumerated() {
                if generated.count >= count { break }
                if Task.isCancelled { break }

                let smi = referenceSmiles
                if smi.contains(rep.pattern) {
                    if let range = smi.range(of: rep.pattern) {
                        let newSmi = smi.replacingCharacters(in: range, with: rep.replacement)
                        if newSmi != smi && !generated.contains(where: { $0.smiles == newSmi }) {
                            generated.append((newSmi, "\(referenceName)_\(rep.label)_\(generated.count + 1)"))
                        }
                    }
                    let allReplaced = smi.replacingOccurrences(of: rep.pattern, with: rep.replacement)
                    if allReplaced != smi && allReplaced != generated.last?.smiles
                       && !generated.contains(where: { $0.smiles == allReplaced }) {
                        generated.append((allReplaced, "\(referenceName)_\(rep.label)_all_\(generated.count + 1)"))
                    }
                }

                analogGenerationProgress = min(1.0, Float(i + 1) / total)
            }

            let addCount = min(generated.count, count)
            for analog in generated.prefix(addCount) {
                ligandDB.addFromSMILES(analog.smiles, name: analog.name)
            }

            analogGenerationProgress = 1.0

            if addCount > 0 {
                log.success("Generated \(addCount) analogs of \(referenceName)", category: .molecule)
                statusMessage = "\(addCount) analogs generated"
            } else {
                log.warn("No valid analogs could be generated", category: .molecule)
                statusMessage = "No analogs generated"
            }
        }
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
}

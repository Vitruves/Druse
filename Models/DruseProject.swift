// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

/// A `.druse` project is a directory bundle containing:
///   protein.pdb        — raw input PDB text (as downloaded/imported)
///   ligands.json       — ligand database entries
///   results.json       — docking results, pockets, config, prepared protein + active/docking ligand
///   settings.json      — workspace/rendering preferences
///
/// This preserves all user edits as real files that can also be inspected outside Druse.
enum DruseProjectIO {

    static let proteinFile = "protein.pdb"
    static let ligandsFile = "ligands.json"
    static let resultsFile = "results.json"
    static let settingsFile = "settings.json"

    // MARK: - Save

    @MainActor
    static func save(to url: URL, viewModel: AppViewModel) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: url, withIntermediateDirectories: true)

        let proteinName = viewModel.molecules.protein?.name ?? "none"
        let ligandCount = viewModel.ligandDB.entries.count
        let resultCount = viewModel.docking.dockingResults.count
        ActivityLog.shared.info("[Project] Saving: protein=\(proteinName), \(ligandCount) ligands, \(resultCount) results", category: .system)

        // 1. Protein PDB (raw input text, needed by the C++ preparation path and
        //    inspectable outside Druse; the prepared structure itself is stored in results.json).
        let proteinURL = url.appendingPathComponent(proteinFile)
        if let pdbContent = viewModel.molecules.rawPDBContent {
            try pdbContent.write(to: proteinURL, atomically: true, encoding: .utf8)
        } else {
            // Re-saving over an existing project must not leave a previous protein behind,
            // otherwise it would be loaded back as this project's receptor.
            if fm.fileExists(atPath: proteinURL.path) {
                try fm.removeItem(at: proteinURL)
            }
            ActivityLog.shared.debug("[Project] No raw PDB content to save", category: .system)
        }

        // 2. Ligand database
        let ligandsURL = url.appendingPathComponent(ligandsFile)
        let ligandsData = try viewModel.ligandDB.encodeToData()
        try ligandsData.write(to: ligandsURL, options: .atomic)

        // 3. Docking results + pockets + config + current molecules
        let results = ProjectResults(
            pockets: viewModel.docking.detectedPockets.map(SerializedPocket.init),
            selectedPocketID: viewModel.docking.selectedPocket?.id,
            dockingConfig: SerializedDockingConfig(from: viewModel.docking.dockingConfig),
            dockingResults: viewModel.docking.dockingResults.map(SerializedDockingResult.init),
            scoringMethod: viewModel.docking.scoringMethod.rawValue,
            ligandName: viewModel.molecules.ligand?.name,
            ligandSMILES: viewModel.molecules.ligand?.smiles,
            proteinPrepared: viewModel.molecules.proteinPrepared,
            protonationPH: viewModel.molecules.protonationPH,
            protein: viewModel.molecules.protein.map(SerializedMolecule.init),
            ligand: viewModel.molecules.ligand.map(SerializedMolecule.init),
            dockingLigand: viewModel.docking.originalDockingLigand.map(SerializedMolecule.init)
        )
        let resultsURL = url.appendingPathComponent(resultsFile)
        try JSONEncoder().encode(results).write(to: resultsURL, options: .atomic)

        // 4. Workspace settings
        let ws = viewModel.workspace
        let settings = ProjectSettings(
            renderMode: ws.renderMode.rawValue,
            showHydrogens: ws.showHydrogens,
            showProtein: ws.showProtein,
            showLigand: ws.showLigand,
            surfaceOpacity: ws.surfaceOpacity,
            backgroundOpacity: ws.backgroundOpacity,
            enableClipping: ws.enableClipping,
            clipNearZ: ws.clipNearZ,
            clipFarZ: ws.clipFarZ,
            slabThickness: ws.slabThickness,
            slabOffset: ws.slabOffset,
            showSurface: ws.showSurface,
            surfaceColorMode: ws.surfaceColorMode.rawValue
        )
        let settingsURL = url.appendingPathComponent(settingsFile)
        try JSONEncoder().encode(settings).write(to: settingsURL, options: .atomic)

        ActivityLog.shared.success("Project saved to \(url.lastPathComponent)", category: .system)
    }

    // MARK: - Load

    @MainActor
    static func load(from url: URL, into viewModel: AppViewModel) async throws {
        let fm = FileManager.default

        // Read and decode every file before touching the view model, so a corrupt
        // file aborts the load instead of leaving a half-replaced workspace.
        let proteinURL = url.appendingPathComponent(proteinFile)
        let pdbContent: String? = try fm.fileExists(atPath: proteinURL.path)
            ? String(contentsOf: proteinURL, encoding: .utf8)
            : nil

        let ligandsURL = url.appendingPathComponent(ligandsFile)
        let ligandEntries: [LigandEntry]? = try fm.fileExists(atPath: ligandsURL.path)
            ? LigandDatabase.decodeEntries(from: Data(contentsOf: ligandsURL))
            : nil

        let resultsURL = url.appendingPathComponent(resultsFile)
        let results: ProjectResults? = try fm.fileExists(atPath: resultsURL.path)
            ? JSONDecoder().decode(ProjectResults.self, from: Data(contentsOf: resultsURL))
            : nil

        let settingsURL = url.appendingPathComponent(settingsFile)
        let settings: ProjectSettings? = try fm.fileExists(atPath: settingsURL.path)
            ? JSONDecoder().decode(ProjectSettings.self, from: Data(contentsOf: settingsURL))
            : nil

        let restoredProtein = try results?.protein?.toMolecule()
        let restoredLigand = try results?.ligand?.toMolecule()
        let restoredDockingLigand = try results?.dockingLigand?.toMolecule()

        // Projects written before the prepared structure was serialized only have protein.pdb.
        var parsed: PDBParseResult?
        if let pdbContent, restoredProtein == nil || restoredLigand == nil {
            parsed = await Task.detached { PDBParser.parse(pdbContent) }.value
        }

        // 1. Protein — prefer the saved (prepared) structure; fall back to parsing protein.pdb.
        if pdbContent != nil || restoredProtein != nil {
            viewModel.molecules.rawPDBContent = pdbContent
            if let restoredProtein {
                viewModel.molecules.protein = restoredProtein
                ActivityLog.shared.debug("[Project] Loaded protein: \(restoredProtein.name) (\(restoredProtein.atomCount) atoms)", category: .system)
            } else if let protData = parsed?.protein {
                let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                   bonds: protData.bonds, title: protData.title)
                mol.secondaryStructureAssignments = protData.ssRanges.map {
                    (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                }
                viewModel.molecules.protein = mol
                ActivityLog.shared.debug("[Project] Loaded protein: \(protData.name) (\(protData.atoms.count) atoms)", category: .system)
            } else {
                ActivityLog.shared.warn("[Project] protein.pdb present but no protein data parsed", category: .system)
            }
        } else {
            ActivityLog.shared.debug("[Project] No protein found in project", category: .system)
        }

        // Ligand — prefer the saved active ligand; fall back to the first PDB HETATM ligand.
        if let restoredLigand {
            viewModel.setLigandForDocking(restoredLigand)
            ActivityLog.shared.debug("[Project] Restored ligand: \(restoredLigand.name)", category: .system)
        } else if let firstLig = parsed?.ligands.first, viewModel.molecules.ligand == nil {
            let ligMol = Molecule(name: firstLig.name, atoms: firstLig.atoms,
                                  bonds: firstLig.bonds, title: firstLig.title)
            viewModel.setLigandForDocking(ligMol)
            ActivityLog.shared.debug("[Project] Restored ligand from PDB: \(firstLig.name)", category: .system)
        }

        // 2. Ligand database (replaces current entries)
        if let ligandEntries {
            viewModel.ligandDB.replaceWithProjectEntries(ligandEntries)
            ActivityLog.shared.debug("[Project] Loaded ligand database: \(ligandEntries.count) entries", category: .system)
        } else {
            ActivityLog.shared.debug("[Project] No ligands.json found, keeping current database", category: .system)
        }

        // 3. Results + pockets + config
        if let results {
            viewModel.docking.detectedPockets = results.pockets.map { $0.toPocket() }
            viewModel.docking.selectedPocket = results.selectedPocketID.flatMap { selID in
                viewModel.docking.detectedPockets.first { $0.id == selID }
            }
            if let cfg = results.dockingConfig {
                viewModel.docking.dockingConfig = cfg.toConfig()
            }
            // Scores are interpreted (units, sign, display) according to the scoring method
            // they were produced with.
            if let method = ScoringMethod(rawValue: results.scoringMethod) {
                viewModel.docking.scoringMethod = method
            }
            // Poses are stored as transformed coordinates of the docking ligand's heavy atoms.
            if let restoredDockingLigand {
                viewModel.docking.originalDockingLigand = restoredDockingLigand
            }
            viewModel.docking.dockingResults = results.dockingResults.map { $0.toResult() }
            viewModel.docking.dockingPosesEvaluated = 0
            viewModel.molecules.proteinPrepared = results.proteinPrepared
            viewModel.molecules.protonationPH = results.protonationPH
            ActivityLog.shared.debug("[Project] Loaded results: \(results.pockets.count) pockets, \(results.dockingResults.count) docking results", category: .system)
        } else {
            ActivityLog.shared.debug("[Project] No results.json found in project", category: .system)
        }

        // 4. Workspace settings
        if let settings {
            settings.apply(to: &viewModel.workspace)
            ActivityLog.shared.debug("[Project] Loaded workspace settings", category: .system)
        } else {
            ActivityLog.shared.debug("[Project] No settings.json found, using defaults", category: .system)
        }

        viewModel.pushToRenderer()

        // Show pocket + best pose if results exist
        if let pocket = viewModel.docking.selectedPocket {
            viewModel.showGridBoxForPocket(pocket)
        }
        if !viewModel.docking.dockingResults.isEmpty {
            viewModel.showDockingPose(at: 0)
        }

        ActivityLog.shared.success("Project loaded from \(url.lastPathComponent)", category: .system)
    }
}

// MARK: - Serializable Types

struct ProjectResults: Codable {
    var pockets: [SerializedPocket]
    var selectedPocketID: Int?
    var dockingConfig: SerializedDockingConfig?
    var dockingResults: [SerializedDockingResult]
    var scoringMethod: String
    var ligandName: String?
    var ligandSMILES: String?
    var proteinPrepared: Bool
    var protonationPH: Float
    // Optional for backward compatibility with projects saved before molecules were serialized.
    var protein: SerializedMolecule?
    var ligand: SerializedMolecule?
    var dockingLigand: SerializedMolecule?
}

enum ProjectDecodingError: LocalizedError {
    case inconsistentMolecule(String)

    var errorDescription: String? {
        switch self {
        case .inconsistentMolecule(let name): "Corrupt molecule data for \(name) in project"
        }
    }
}

/// Full-fidelity molecule snapshot (prepared hydrogens, partial/formal charges, bonds,
/// secondary structure) stored column-wise to keep project files compact.
struct SerializedMolecule: Codable {
    let name: String
    let title: String
    let smiles: String?
    let elements: [Int]
    let positions: [Float]          // flat [x,y,z,...]
    let atomNames: [String]
    let residueNames: [String]
    let residueSeqs: [Int]
    let chainIDs: [String]
    let charges: [Float]
    let formalCharges: [Int]
    let hetFlags: [Bool]
    let occupancies: [Float]
    let tempFactors: [Float]
    let altLocs: [String]
    let insertionCodes: [String]
    let bonds: [Int]                // flat [atom1, atom2, order, rotatable(0/1), ...]
    let secondaryStructure: [Int]   // flat [start, end, type, ...] (type: 0 coil, 1 helix, 2 sheet, 3 turn)
    let secondaryStructureChains: [String]

    @MainActor
    init(from mol: Molecule) {
        name = mol.name
        title = mol.title
        smiles = mol.smiles
        let atoms = mol.atoms
        elements = atoms.map { $0.element.rawValue }
        positions = atoms.flatMap { [$0.position.x, $0.position.y, $0.position.z] }
        atomNames = atoms.map(\.name)
        residueNames = atoms.map(\.residueName)
        residueSeqs = atoms.map(\.residueSeq)
        chainIDs = atoms.map(\.chainID)
        charges = atoms.map(\.charge)
        formalCharges = atoms.map(\.formalCharge)
        hetFlags = atoms.map(\.isHetAtom)
        occupancies = atoms.map(\.occupancy)
        tempFactors = atoms.map(\.tempFactor)
        altLocs = atoms.map(\.altLoc)
        insertionCodes = atoms.map(\.insertionCode)
        bonds = mol.bonds.flatMap { [$0.atomIndex1, $0.atomIndex2, $0.order.rawValue, $0.isRotatable ? 1 : 0] }
        secondaryStructure = mol.secondaryStructureAssignments.flatMap { ss -> [Int] in
            let type: Int = switch ss.type {
            case .coil: 0
            case .helix: 1
            case .sheet: 2
            case .turn: 3
            }
            return [ss.start, ss.end, type]
        }
        secondaryStructureChains = mol.secondaryStructureAssignments.map(\.chain)
    }

    @MainActor
    func toMolecule() throws -> Molecule {
        let n = elements.count
        let columnCounts = [atomNames.count, residueNames.count, residueSeqs.count, chainIDs.count,
                            charges.count, formalCharges.count, hetFlags.count, occupancies.count,
                            tempFactors.count, altLocs.count, insertionCodes.count]
        guard positions.count == n * 3, columnCounts.allSatisfy({ $0 == n }),
              bonds.count % 4 == 0,
              secondaryStructure.count == secondaryStructureChains.count * 3 else {
            throw ProjectDecodingError.inconsistentMolecule(name)
        }

        var atoms: [Atom] = []
        atoms.reserveCapacity(n)
        for i in 0..<n {
            atoms.append(Atom(
                id: i,
                element: Element(rawValue: elements[i]) ?? .C,
                position: SIMD3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]),
                name: atomNames[i],
                residueName: residueNames[i],
                residueSeq: residueSeqs[i],
                chainID: chainIDs[i],
                charge: charges[i],
                formalCharge: formalCharges[i],
                isHetAtom: hetFlags[i],
                occupancy: occupancies[i],
                tempFactor: tempFactors[i],
                altLoc: altLocs[i],
                insertionCode: insertionCodes[i]
            ))
        }

        var molBonds: [Bond] = []
        molBonds.reserveCapacity(bonds.count / 4)
        for b in stride(from: 0, to: bonds.count, by: 4) {
            let a1 = bonds[b], a2 = bonds[b + 1]
            guard a1 >= 0, a1 < n, a2 >= 0, a2 < n else {
                throw ProjectDecodingError.inconsistentMolecule(name)
            }
            molBonds.append(Bond(id: molBonds.count, atomIndex1: a1, atomIndex2: a2,
                                 order: BondOrder(rawValue: bonds[b + 2]) ?? .single,
                                 isRotatable: bonds[b + 3] != 0))
        }

        let mol = Molecule(name: name, atoms: atoms, bonds: molBonds, title: title, smiles: smiles)
        mol.secondaryStructureAssignments = secondaryStructureChains.indices.map { k in
            let type: SecondaryStructure = switch secondaryStructure[3 * k + 2] {
            case 1: .helix
            case 2: .sheet
            case 3: .turn
            default: .coil
            }
            return (start: secondaryStructure[3 * k], end: secondaryStructure[3 * k + 1],
                    type: type, chain: secondaryStructureChains[k])
        }
        return mol
    }
}

struct ProjectSettings: Codable {
    var renderMode: String
    var showHydrogens: Bool
    var showProtein: Bool
    var showLigand: Bool
    var surfaceOpacity: Float
    var backgroundOpacity: Float
    var enableClipping: Bool
    var clipNearZ: Float
    var clipFarZ: Float
    var slabThickness: Float?
    var slabOffset: Float?
    var showSurface: Bool
    var surfaceColorMode: String

    func apply(to ws: inout WorkspaceState) {
        ws.renderMode = RenderMode(rawValue: renderMode) ?? ws.renderMode
        ws.showHydrogens = showHydrogens
        ws.showProtein = showProtein
        ws.showLigand = showLigand
        ws.surfaceOpacity = surfaceOpacity
        ws.backgroundOpacity = backgroundOpacity
        ws.enableClipping = enableClipping
        ws.clipNearZ = clipNearZ
        ws.clipFarZ = clipFarZ
        if let t = slabThickness { ws.slabThickness = t }
        if let o = slabOffset { ws.slabOffset = o }
        ws.showSurface = showSurface
        ws.surfaceColorMode = SurfaceColorMode(rawValue: surfaceColorMode) ?? ws.surfaceColorMode
    }
}

struct SerializedPocket: Codable {
    let id: Int
    let cx, cy, cz: Float
    let sx, sy, sz: Float
    let volume, buriedness, polarity, druggability: Float
    let residueIndices: [Int]

    init(from p: BindingPocket) {
        id = p.id; cx = p.center.x; cy = p.center.y; cz = p.center.z
        sx = p.size.x; sy = p.size.y; sz = p.size.z
        volume = p.volume; buriedness = p.buriedness
        polarity = p.polarity; druggability = p.druggability
        residueIndices = p.residueIndices
    }

    func toPocket() -> BindingPocket {
        BindingPocket(id: id, center: .init(cx, cy, cz), size: .init(sx, sy, sz),
                      volume: volume, buriedness: buriedness, polarity: polarity,
                      druggability: druggability, residueIndices: residueIndices, probePositions: [])
    }
}

struct SerializedDockingConfig: Codable {
    let populationSize, numRuns, generationsPerRun: Int
    let gridSpacing: Float
    let enableFlexibility, useAnalyticalGradients: Bool
    let torsionSearchPreset: TorsionSearchPreset?
    let torsionExactFraction: Float?
    let torsionLocalFraction: Float?
    let torsionLocalAmplitude: Float?
    let torsionRandomResetProbability: Float?
    let torsionPerturbationScale: Float?

    init(from c: DockingConfig) {
        populationSize = c.populationSize; numRuns = c.numRuns
        generationsPerRun = c.generationsPerRun; gridSpacing = c.gridSpacing
        enableFlexibility = c.enableFlexibility; useAnalyticalGradients = c.useAnalyticalGradients
        torsionSearchPreset = c.torsionSearchPreset
        torsionExactFraction = c.torsionExactFraction
        torsionLocalFraction = c.torsionLocalFraction
        torsionLocalAmplitude = c.torsionLocalAmplitude
        torsionRandomResetProbability = c.torsionRandomResetProbability
        torsionPerturbationScale = c.torsionPerturbationScale
    }

    func toConfig() -> DockingConfig {
        var c = DockingConfig()
        c.populationSize = populationSize; c.numRuns = numRuns
        c.generationsPerRun = generationsPerRun; c.gridSpacing = gridSpacing
        c.enableFlexibility = enableFlexibility; c.useAnalyticalGradients = useAnalyticalGradients
        if let preset = torsionSearchPreset {
            c.applyTorsionSearchPreset(preset)
        }
        if let torsionExactFraction {
            c.torsionSearchPreset = .manual
            c.torsionExactFraction = torsionExactFraction
        }
        if let torsionLocalFraction {
            c.torsionSearchPreset = .manual
            c.torsionLocalFraction = torsionLocalFraction
        }
        if let torsionLocalAmplitude {
            c.torsionSearchPreset = .manual
            c.torsionLocalAmplitude = torsionLocalAmplitude
        }
        if let torsionRandomResetProbability {
            c.torsionSearchPreset = .manual
            c.torsionRandomResetProbability = torsionRandomResetProbability
        }
        if let torsionPerturbationScale {
            c.torsionSearchPreset = .manual
            c.torsionPerturbationScale = torsionPerturbationScale
        }
        return c
    }
}

struct SerializedDockingResult: Codable {
    let id: Int
    let energy, stericEnergy, hydrophobicEnergy, hbondEnergy, torsionPenalty: Float
    let generation, clusterID, clusterRank: Int
    let tx, ty, tz, qx, qy, qz, qw: Float
    let torsions: [Float]
    let positions: [Float]
    let refinementEnergy: Float?

    init(from r: DockingResult) {
        id = r.id; energy = r.energy; stericEnergy = r.stericEnergy
        hydrophobicEnergy = r.hydrophobicEnergy; hbondEnergy = r.hbondEnergy
        torsionPenalty = r.torsionPenalty; generation = r.generation
        clusterID = r.clusterID; clusterRank = r.clusterRank
        tx = r.pose.translation.x; ty = r.pose.translation.y; tz = r.pose.translation.z
        qx = r.pose.rotation.imag.x; qy = r.pose.rotation.imag.y
        qz = r.pose.rotation.imag.z; qw = r.pose.rotation.real
        torsions = r.pose.torsions
        positions = r.transformedAtomPositions.flatMap { [$0.x, $0.y, $0.z] }
        refinementEnergy = r.refinementEnergy
    }

    func toResult() -> DockingResult {
        var r = DockingResult(
            id: id,
            pose: DockPoseSwift(
                translation: .init(tx, ty, tz),
                rotation: simd_quatf(ix: qx, iy: qy, iz: qz, r: qw),
                torsions: torsions
            ),
            energy: energy, stericEnergy: stericEnergy,
            hydrophobicEnergy: hydrophobicEnergy, hbondEnergy: hbondEnergy,
            torsionPenalty: torsionPenalty, generation: generation
        )
        r.clusterID = clusterID; r.clusterRank = clusterRank
        r.refinementEnergy = refinementEnergy
        var pos: [SIMD3<Float>] = []
        for i in stride(from: 0, to: positions.count - 2, by: 3) {
            pos.append(.init(positions[i], positions[i+1], positions[i+2]))
        }
        r.transformedAtomPositions = pos
        return r
    }
}

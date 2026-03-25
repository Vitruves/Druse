import Foundation
import simd

/// A `.druse` project is a directory bundle containing:
///   protein.pdb        — raw PDB text (with user's preparation changes)
///   ligands.json       — ligand database entries
///   results.json       — docking results, pockets, config
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

        // 1. Protein PDB (raw content preserves user's edits/preparation)
        if let pdbContent = viewModel.molecules.rawPDBContent {
            let proteinURL = url.appendingPathComponent(proteinFile)
            try pdbContent.write(to: proteinURL, atomically: true, encoding: .utf8)
        } else {
            ActivityLog.shared.debug("[Project] No raw PDB content to save", category: .system)
        }

        // 2. Ligand database
        let ligandsURL = url.appendingPathComponent(ligandsFile)
        let ligandsData = viewModel.ligandDB.encodeToData()
        try ligandsData.write(to: ligandsURL, options: .atomic)

        // 3. Docking results + pockets + config
        let results = ProjectResults(
            pockets: viewModel.docking.detectedPockets.map(SerializedPocket.init),
            selectedPocketID: viewModel.docking.selectedPocket?.id,
            dockingConfig: SerializedDockingConfig(from: viewModel.docking.dockingConfig),
            dockingResults: viewModel.docking.dockingResults.map(SerializedDockingResult.init),
            scoringMethod: viewModel.docking.scoringMethod.rawValue,
            ligandName: viewModel.molecules.ligand?.name,
            ligandSMILES: viewModel.molecules.ligand?.smiles,
            proteinPrepared: viewModel.molecules.proteinPrepared,
            protonationPH: viewModel.molecules.protonationPH
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

        // 1. Protein — parse from saved PDB text (preserves user's preparation)
        let proteinURL = url.appendingPathComponent(proteinFile)
        if fm.fileExists(atPath: proteinURL.path) {
            let pdbContent = try String(contentsOf: proteinURL, encoding: .utf8)
            viewModel.molecules.rawPDBContent = pdbContent

            let parsed = await Task.detached { PDBParser.parse(pdbContent) }.value
            if let protData = parsed.protein {
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

            // Restore ligands from PDB HETATM records
            if let firstLig = parsed.ligands.first, viewModel.molecules.ligand == nil {
                let ligMol = Molecule(name: firstLig.name, atoms: firstLig.atoms,
                                      bonds: firstLig.bonds, title: firstLig.title)
                viewModel.setLigandForDocking(ligMol)
                ActivityLog.shared.debug("[Project] Restored ligand from PDB: \(firstLig.name)", category: .system)
            }
        } else {
            ActivityLog.shared.debug("[Project] No protein.pdb file found in project", category: .system)
        }

        // 2. Ligand database (replaces current entries)
        let ligandsURL = url.appendingPathComponent(ligandsFile)
        if fm.fileExists(atPath: ligandsURL.path) {
            let data = try Data(contentsOf: ligandsURL)
            viewModel.ligandDB.decodeFromData(data)
            ActivityLog.shared.debug("[Project] Loaded ligand database: \(viewModel.ligandDB.entries.count) entries", category: .system)
        } else {
            ActivityLog.shared.debug("[Project] No ligands.json found, keeping current database", category: .system)
        }

        // 3. Results + pockets + config
        let resultsURL = url.appendingPathComponent(resultsFile)
        if fm.fileExists(atPath: resultsURL.path) {
            let data = try Data(contentsOf: resultsURL)
            let results = try JSONDecoder().decode(ProjectResults.self, from: data)

            viewModel.docking.detectedPockets = results.pockets.map { $0.toPocket() }
            if let selID = results.selectedPocketID {
                viewModel.docking.selectedPocket = viewModel.docking.detectedPockets.first { $0.id == selID }
            }
            if let cfg = results.dockingConfig {
                viewModel.docking.dockingConfig = cfg.toConfig()
            }
            viewModel.docking.dockingResults = results.dockingResults.map { $0.toResult() }
            viewModel.molecules.proteinPrepared = results.proteinPrepared
            viewModel.molecules.protonationPH = results.protonationPH
            ActivityLog.shared.debug("[Project] Loaded results: \(results.pockets.count) pockets, \(results.dockingResults.count) docking results", category: .system)
        } else {
            ActivityLog.shared.debug("[Project] No results.json found in project", category: .system)
        }

        // 4. Workspace settings
        let settingsURL = url.appendingPathComponent(settingsFile)
        if fm.fileExists(atPath: settingsURL.path) {
            let data = try Data(contentsOf: settingsURL)
            let settings = try JSONDecoder().decode(ProjectSettings.self, from: data)
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

    init(from c: DockingConfig) {
        populationSize = c.populationSize; numRuns = c.numRuns
        generationsPerRun = c.generationsPerRun; gridSpacing = c.gridSpacing
        enableFlexibility = c.enableFlexibility; useAnalyticalGradients = c.useAnalyticalGradients
    }

    func toConfig() -> DockingConfig {
        var c = DockingConfig()
        c.populationSize = populationSize; c.numRuns = numRuns
        c.generationsPerRun = generationsPerRun; c.gridSpacing = gridSpacing
        c.enableFlexibility = enableFlexibility; c.useAnalyticalGradients = useAnalyticalGradients
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
    let mlDockingScore, mlPKd, mlPoseConfidence, refinementEnergy: Float?

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
        mlDockingScore = r.mlDockingScore; mlPKd = r.mlPKd
        mlPoseConfidence = r.mlPoseConfidence; refinementEnergy = r.refinementEnergy
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
        r.mlDockingScore = mlDockingScore; r.mlPKd = mlPKd
        r.mlPoseConfidence = mlPoseConfidence; r.refinementEnergy = refinementEnergy
        var pos: [SIMD3<Float>] = []
        for i in stride(from: 0, to: positions.count - 2, by: 3) {
            pos.append(.init(positions[i], positions[i+1], positions[i+2]))
        }
        r.transformedAtomPositions = pos
        return r
    }
}

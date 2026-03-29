import Foundation
import simd

// MARK: - CASF-2016 Manifest (read from Python-generated JSON)

struct CASFManifest: Codable {
    let benchmark: String
    let description: String
    let complexes: [CASFComplex]
}

struct CASFComplex: Codable {
    let pdbId: String
    let proteinPdb: String
    let pocketPdb: String?
    let ligandSdf: String
    let smiles: String
    let crystalPositions: [[Float]]             // Heavy atom positions in SDF file order
    let crystalPositionsSmiles: [[Float]]?       // Heavy atom positions in SMILES canonical order (pre-computed)
    let heavyAtomCount: Int
    let pKd: Float
    let smilesAtomMap: [Int]?                    // smilesAtomMap[smiles_idx] = sdf_idx (for diagnostics)

    enum CodingKeys: String, CodingKey {
        case pdbId = "pdb_id"
        case proteinPdb = "protein_pdb"
        case pocketPdb = "pocket_pdb"
        case ligandSdf = "ligand_sdf"
        case smiles
        case crystalPositions = "crystal_positions"
        case crystalPositionsSmiles = "crystal_positions_smiles"
        case heavyAtomCount = "heavy_atom_count"
        case pKd
        case smilesAtomMap = "smiles_atom_map"
    }

    /// Crystal positions in SDF order (for redock mode).
    var crystalPositionsSIMD: [SIMD3<Float>] {
        crystalPositions.compactMap { arr in
            guard arr.count >= 3 else { return nil }
            return SIMD3<Float>(arr[0], arr[1], arr[2])
        }
    }

    /// Crystal positions in SMILES canonical order (for full pipeline mode).
    var crystalPositionsSMILESOrder: [SIMD3<Float>]? {
        crystalPositionsSmiles?.compactMap { arr in
            guard arr.count >= 3 else { return nil }
            return SIMD3<Float>(arr[0], arr[1], arr[2])
        }
    }
}

// MARK: - Benchmark Results (written by Swift, read by Python)

struct BenchmarkResults: Codable {
    let benchmark: String
    let version: String
    let scoringMethod: String     // "vina", "drusina", or "druseAF"
    let timestamp: String
    let config: BenchmarkConfigRecord
    var entries: [BenchmarkResultEntry]
}

struct BenchmarkConfigRecord: Codable {
    let populationSize: Int
    let generations: Int
    let gridSpacing: Float
    let numRuns: Int
}

struct BenchmarkResultEntry: Codable {
    let pdbId: String
    var bestEnergy: Float             // internal energy (kcal/mol for Vina/Drusina/PIGNet2; -pKd*conf for DruseAF)
    var bestDisplayScore: Float?      // display score: same as bestEnergy for energy methods, pKi for DruseAF
    var bestRmsd: Float?              // vs crystal pose
    var experimentalPKd: Float?
    var numPoses: Int
    var dockingTimeMs: Double
    var success: Bool
    var error: String?
    var gfn2Energy: Float?            // GFN2-xTB total energy (kcal/mol)
    var gfn2DispersionEnergy: Float?  // D4 component (kcal/mol)
    var gfn2SolvationEnergy: Float?   // GBSA/ALPB component (kcal/mol)

    // Debug diagnostics (populated when config enables them)
    var pocketCenter: [Float]?        // detected pocket center [x, y, z]
    var crystalCenter: [Float]?       // crystal ligand centroid [x, y, z]
    var pocketDistance: Float?         // distance between pocket center and crystal centroid
    var pocketVolume: Float?           // detected pocket volume (A^3)
    var pocketBuriedness: Float?       // detected pocket buriedness (0-1)
    var pocketMethod: String?          // "ml", "geometric", or "ligand-guided"
    var searchBoxSize: [Float]?        // pocket half-extents [x, y, z]
    var ligandHeavyCount: Int?         // heavy atoms in prepared ligand
    var crystalHeavyCount: Int?        // heavy atoms in crystal ligand
    var conformerRmsd: Float?           // RMSD of SMILES-derived conformer vs crystal (measures 3D embedding quality, not docking)
    var strainEnergy: Float?           // MMFF strain of best pose (kcal/mol)
    var allPoseRmsds: [Float]?         // RMSDs of top-N poses (for convergence analysis)
    var drusinaDecomposition: [String: Float]?  // per-term Drusina scores on crystal pose

    enum CodingKeys: String, CodingKey {
        case pdbId = "pdb_id"
        case bestEnergy = "best_energy"
        case bestDisplayScore = "best_display_score"
        case bestRmsd = "best_rmsd"
        case experimentalPKd = "experimental_pKd"
        case numPoses = "num_poses"
        case dockingTimeMs = "docking_time_ms"
        case success, error
        case gfn2Energy = "gfn2_energy"
        case gfn2DispersionEnergy = "gfn2_dispersion"
        case gfn2SolvationEnergy = "gfn2_solvation"
        case pocketCenter = "pocket_center"
        case crystalCenter = "crystal_center"
        case pocketDistance = "pocket_distance"
        case pocketVolume = "pocket_volume"
        case pocketBuriedness = "pocket_buriedness"
        case pocketMethod = "pocket_method"
        case searchBoxSize = "search_box_size"
        case ligandHeavyCount = "ligand_heavy_count"
        case crystalHeavyCount = "crystal_heavy_count"
        case conformerRmsd = "conformer_rmsd"
        case strainEnergy = "strain_energy"
        case allPoseRmsds = "all_pose_rmsds"
        case drusinaDecomposition = "drusina_decomposition"
    }
}

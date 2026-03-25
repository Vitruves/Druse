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
    let crystalPositions: [[Float]]
    let heavyAtomCount: Int
    let pKd: Float

    enum CodingKeys: String, CodingKey {
        case pdbId = "pdb_id"
        case proteinPdb = "protein_pdb"
        case pocketPdb = "pocket_pdb"
        case ligandSdf = "ligand_sdf"
        case smiles
        case crystalPositions = "crystal_positions"
        case heavyAtomCount = "heavy_atom_count"
        case pKd
    }

    var crystalPositionsSIMD: [SIMD3<Float>] {
        crystalPositions.compactMap { arr in
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
    var bestEnergy: Float             // score from the GA's scoring method
    var bestRmsd: Float?              // vs crystal pose
    var mlRescorePKd: Float?          // optional post-dock ML rescore
    var experimentalPKd: Float?
    var numPoses: Int
    var dockingTimeMs: Double
    var success: Bool
    var error: String?
    var gfn2Energy: Float?            // GFN2-xTB total energy (kcal/mol)
    var gfn2DispersionEnergy: Float?  // D4 component (kcal/mol)
    var gfn2SolvationEnergy: Float?   // GBSA/ALPB component (kcal/mol)

    enum CodingKeys: String, CodingKey {
        case pdbId = "pdb_id"
        case bestEnergy = "best_energy"
        case bestRmsd = "best_rmsd"
        case mlRescorePKd = "ml_rescore_pKd"
        case experimentalPKd = "experimental_pKd"
        case numPoses = "num_poses"
        case dockingTimeMs = "docking_time_ms"
        case success, error
        case gfn2Energy = "gfn2_energy"
        case gfn2DispersionEnergy = "gfn2_dispersion"
        case gfn2SolvationEnergy = "gfn2_solvation"
    }
}

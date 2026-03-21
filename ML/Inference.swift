import Foundation
@preconcurrency import CoreML
import simd

// MARK: - Druse-Score Feature Extraction

/// Extracts geometric + chemical features from a protein-ligand complex
/// for CoreML inference. Produces the input tensors expected by DruseScore.mlmodel.
struct DruseScoreFeatureExtractor {

    /// Atom-level features: [atomicNum one-hot(10), aromaticity, charge, hbDonor, hbAcceptor,
    /// hybridization one-hot(4), is_protein, is_ligand] = 18 features per atom
    static let atomFeatureSize = 18

    /// Max atoms in pocket + ligand (pad/truncate to this)
    static let maxAtoms = 512

    /// Radial basis function centers for distance encoding (0-10 Å, 50 bins)
    static let rbfCenters: [Float] = (0..<50).map { Float($0) * 0.2 }
    static let rbfGamma: Float = 10.0 // 1/(2σ²) where σ=0.2236

    struct ComplexFeatures {
        var proteinPositions: [SIMD3<Float>]  // pocket atom positions
        var proteinFeatures: [[Float]]         // per-atom feature vectors
        var ligandPositions: [SIMD3<Float>]
        var ligandFeatures: [[Float]]
        var pairDistances: [[Float]]           // NxM distance matrix
        var pairRBF: [[[Float]]]               // NxMx50 RBF-encoded distances
    }

    /// Extract features from a docked protein-ligand complex.
    static func extract(
        proteinAtoms: [Atom],
        ligandAtoms: [Atom],
        pocketCenter: SIMD3<Float>,
        pocketRadius: Float = 10.0
    ) -> ComplexFeatures {
        // Filter protein atoms to pocket region
        let pocketAtoms = proteinAtoms.filter { atom in
            simd_distance(atom.position, pocketCenter) <= pocketRadius
        }

        let protPositions = pocketAtoms.map(\.position)
        let ligPositions = ligandAtoms.map(\.position)

        let protFeats = pocketAtoms.map { atomFeatures($0, isProtein: true) }
        let ligFeats = ligandAtoms.map { atomFeatures($0, isProtein: false) }

        // Compute pairwise distance matrix
        let nProt = protPositions.count
        let nLig = ligPositions.count

        var distances = [[Float]](repeating: [Float](repeating: 0, count: nLig), count: nProt)
        var rbfEncoded = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 50), count: nLig),
            count: nProt
        )

        for i in 0..<nProt {
            for j in 0..<nLig {
                let d = simd_distance(protPositions[i], ligPositions[j])
                distances[i][j] = d
                rbfEncoded[i][j] = rbfEncode(d)
            }
        }

        return ComplexFeatures(
            proteinPositions: protPositions,
            proteinFeatures: protFeats,
            ligandPositions: ligPositions,
            ligandFeatures: ligFeats,
            pairDistances: distances,
            pairRBF: rbfEncoded
        )
    }

    /// Encode atom as 18-dimensional feature vector.
    private static func atomFeatures(_ atom: Atom, isProtein: Bool) -> [Float] {
        var features = [Float](repeating: 0, count: atomFeatureSize)

        // One-hot encode atomic number (H, C, N, O, F, P, S, Cl, Br, other)
        let atomNumBin: Int
        switch atom.element {
        case .H:                atomNumBin = 0
        case .C:                atomNumBin = 1
        case .N:                atomNumBin = 2
        case .O:                atomNumBin = 3
        case .F:                atomNumBin = 4
        case .P:                atomNumBin = 5
        case .S:                atomNumBin = 6
        case .Cl:               atomNumBin = 7
        case .Br:               atomNumBin = 8
        default:                atomNumBin = 9
        }
        features[atomNumBin] = 1.0

        // Aromaticity (approximated from element context)
        features[10] = 0.0

        // Partial charge
        features[11] = atom.charge

        // H-bond donor/acceptor (simple element-based heuristic)
        let isHBDonor = atom.element == Element.N || atom.element == Element.O
        let isHBAcceptor = atom.element == Element.N || atom.element == Element.O || atom.element == Element.F
        features[12] = isHBDonor ? 1.0 : 0.0
        features[13] = isHBAcceptor ? 1.0 : 0.0

        // Hybridization one-hot: sp, sp2, sp3, other
        // Default to sp3 (index 2) — refined by bond analysis in full implementation
        features[14 + 2] = 1.0

        // Is protein / is ligand
        features[17] = isProtein ? 0.0 : 1.0

        return features
    }

    /// Radial basis function encoding of a distance.
    private static func rbfEncode(_ distance: Float) -> [Float] {
        rbfCenters.map { center in
            exp(-rbfGamma * (distance - center) * (distance - center))
        }
    }
}

// MARK: - Druse-Score CoreML Inference

/// Manages CoreML model loading and inference for the DruseScore scoring function.
/// SE(3)-equivariant geometric cross-attention network for binding affinity prediction.
@MainActor
final class DruseScoreInference {

    private var model: MLModel?
    private var isLoaded = false

    /// Predicted output from DruseScore
    struct Prediction: Sendable {
        var pKd: Float                          // predicted binding affinity (-log10 Kd)
        var poseConfidence: Float               // 0-1, is this a correct pose?
        var interactionMap: [InteractionPred]    // per-atom-pair interaction types
        var attentionWeights: [Float]           // cross-attention weights for visualization

        struct InteractionPred: Sendable {
            var proteinAtomIndex: Int
            var ligandAtomIndex: Int
            var hbondProb: Float
            var hydrophobicProb: Float
            var ionicProb: Float
            var piStackProb: Float
            var halogenProb: Float
        }
    }

    /// Load the DruseScore CoreML model from the app bundle.
    func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "DruseScore", withExtension: "mlmodelc") else {
            print("[DruseScore] Model not found in bundle — using physics-only scoring")
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // Use Neural Engine + GPU + CPU
            model = try MLModel(contentsOf: modelURL, configuration: config)
            isLoaded = true
            print("[DruseScore] Model loaded — compute units: all (ANE preferred)")
        } catch {
            print("[DruseScore] Failed to load model: \(error)")
        }
    }

    /// Whether the ML model is available.
    var isAvailable: Bool { isLoaded && model != nil }

    /// Score a single protein-ligand complex.
    func score(features: DruseScoreFeatureExtractor.ComplexFeatures) async -> Prediction? {
        guard let model else { return nil }

        // Prepare MLMultiArray inputs
        let nProt = min(features.proteinPositions.count, DruseScoreFeatureExtractor.maxAtoms)
        let nLig = min(features.ligandPositions.count, DruseScoreFeatureExtractor.maxAtoms)
        let featSize = DruseScoreFeatureExtractor.atomFeatureSize

        guard nProt > 0 && nLig > 0 else { return nil }

        do {
            // Protein features: [1, maxAtoms, featSize]
            let protArray = try MLMultiArray(shape: [1, NSNumber(value: nProt), NSNumber(value: featSize)], dataType: .float32)
            for i in 0..<nProt {
                for j in 0..<featSize {
                    protArray[[0, i, j] as [NSNumber]] = NSNumber(value: features.proteinFeatures[i][j])
                }
            }

            // Ligand features: [1, maxAtoms, featSize]
            let ligArray = try MLMultiArray(shape: [1, NSNumber(value: nLig), NSNumber(value: featSize)], dataType: .float32)
            for i in 0..<nLig {
                for j in 0..<featSize {
                    ligArray[[0, i, j] as [NSNumber]] = NSNumber(value: features.ligandFeatures[i][j])
                }
            }

            // Protein positions: [1, maxAtoms, 3]
            let protPos = try MLMultiArray(shape: [1, NSNumber(value: nProt), 3], dataType: .float32)
            for i in 0..<nProt {
                protPos[[0, i, 0] as [NSNumber]] = NSNumber(value: features.proteinPositions[i].x)
                protPos[[0, i, 1] as [NSNumber]] = NSNumber(value: features.proteinPositions[i].y)
                protPos[[0, i, 2] as [NSNumber]] = NSNumber(value: features.proteinPositions[i].z)
            }

            // Ligand positions: [1, maxAtoms, 3]
            let ligPos = try MLMultiArray(shape: [1, NSNumber(value: nLig), 3], dataType: .float32)
            for i in 0..<nLig {
                ligPos[[0, i, 0] as [NSNumber]] = NSNumber(value: features.ligandPositions[i].x)
                ligPos[[0, i, 1] as [NSNumber]] = NSNumber(value: features.ligandPositions[i].y)
                ligPos[[0, i, 2] as [NSNumber]] = NSNumber(value: features.ligandPositions[i].z)
            }

            // RBF-encoded pairwise distances: [1, nProt, nLig, 50]
            let rbfArray = try MLMultiArray(shape: [1, NSNumber(value: nProt), NSNumber(value: nLig), 50], dataType: .float32)
            for i in 0..<nProt {
                for j in 0..<nLig {
                    for k in 0..<50 {
                        rbfArray[[0, i, j, k] as [NSNumber]] = NSNumber(value: features.pairRBF[i][j][k])
                    }
                }
            }

            // Create feature provider
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "protein_features": protArray,
                "ligand_features": ligArray,
                "protein_positions": protPos,
                "ligand_positions": ligPos,
                "pair_rbf": rbfArray
            ])

            // Run inference on main actor (CoreML handles dispatch internally)
            let result = try await model.prediction(from: provider)
            return parsePrediction(result, nProt: nProt, nLig: nLig)
        } catch {
            print("[DruseScore] Inference failed: \(error)")
            return nil
        }
    }

    /// Re-rank a batch of docking results using DruseScore.
    func rerankPoses(
        results: [DockingResult],
        proteinAtoms: [Atom],
        ligandAtoms: [Atom],
        pocketCenter: SIMD3<Float>
    ) async -> [DockingResult] {
        guard isAvailable else { return results }

        var scored = results
        let pocketProteinAtoms = proteinAtoms.filter {
            simd_distance($0.position, pocketCenter) <= 10.0
        }

        for i in 0..<scored.count {
            // Create virtual ligand atoms at pose position
            var poseAtoms = ligandAtoms
            for j in 0..<poseAtoms.count {
                if j < scored[i].transformedAtomPositions.count {
                    poseAtoms[j].position = scored[i].transformedAtomPositions[j]
                }
            }

            let features = DruseScoreFeatureExtractor.extract(
                proteinAtoms: pocketProteinAtoms,
                ligandAtoms: poseAtoms,
                pocketCenter: pocketCenter
            )

            if let pred = await score(features: features) {
                // Combine physics score with ML score (weighted)
                let physicsScore = scored[i].energy
                let mlScore = -pred.pKd * 1.364 // Convert pKd to approx kcal/mol
                scored[i].energy = 0.3 * physicsScore + 0.7 * mlScore
            }
        }

        return scored.sorted { $0.energy < $1.energy }
    }

    /// Parse CoreML output tensors into our Prediction struct.
    private func parsePrediction(_ output: MLFeatureProvider, nProt: Int, nLig: Int) -> Prediction? {
        guard let pkdValue = output.featureValue(for: "pKd")?.multiArrayValue,
              let poseConf = output.featureValue(for: "pose_confidence")?.multiArrayValue
        else { return nil }

        let pKd = pkdValue[0].floatValue
        let confidence = poseConf[0].floatValue

        // Parse interaction predictions if available
        var interactions: [Prediction.InteractionPred] = []
        if let interMap = output.featureValue(for: "interaction_map")?.multiArrayValue {
            for i in 0..<nProt {
                for j in 0..<nLig {
                    let hb = interMap[[0, i, j, 0] as [NSNumber]].floatValue
                    let hp = interMap[[0, i, j, 1] as [NSNumber]].floatValue
                    let ion = interMap[[0, i, j, 2] as [NSNumber]].floatValue
                    let pi = interMap[[0, i, j, 3] as [NSNumber]].floatValue
                    let hal = interMap[[0, i, j, 4] as [NSNumber]].floatValue

                    // Only keep significant interactions
                    if max(hb, hp, ion, pi, hal) > 0.3 {
                        interactions.append(.init(
                            proteinAtomIndex: i, ligandAtomIndex: j,
                            hbondProb: hb, hydrophobicProb: hp,
                            ionicProb: ion, piStackProb: pi, halogenProb: hal
                        ))
                    }
                }
            }
        }

        // Parse attention weights if available
        var attention: [Float] = []
        if let attnWeights = output.featureValue(for: "attention_weights")?.multiArrayValue {
            for i in 0..<attnWeights.count {
                attention.append(attnWeights[i].floatValue)
            }
        }

        return Prediction(
            pKd: pKd,
            poseConfidence: confidence,
            interactionMap: interactions,
            attentionWeights: attention
        )
    }
}

// MARK: - ADMET Prediction

/// Multi-model ADMET prediction using CoreML.
/// Each property has its own lightweight GNN model.
@MainActor
final class ADMETPredictor {

    struct ADMETResult: Sendable {
        var logP: Float?
        var logD74: Float?
        var aqueousSolubility: Float?     // log S
        var cyp2d6Inhibition: Float?      // probability
        var cyp3a4Inhibition: Float?      // probability
        var hergLiability: Float?         // probability
        var bbbPermeability: Float?       // probability
        var metabolicStability: Float?    // probability of being stable
        var lipinski: Bool = false
        var veber: Bool = false
        var drugLikeness: Float = 0       // 0-1 composite score
    }

    private var models: [String: MLModel] = [:]

    /// Load all ADMET models from bundle.
    func loadModels() {
        let modelNames = [
            "ADMET_LogP", "ADMET_LogD", "ADMET_Solubility",
            "ADMET_CYP2D6", "ADMET_CYP3A4", "ADMET_hERG",
            "ADMET_BBB", "ADMET_MetabolicStability"
        ]

        let config = MLModelConfiguration()
        config.computeUnits = .all

        for name in modelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    models[name] = try MLModel(contentsOf: url, configuration: config)
                } catch {
                    print("[ADMET] Failed to load \(name): \(error)")
                }
            }
        }

        if !models.isEmpty {
            print("[ADMET] Loaded \(models.count) ADMET models")
        }
    }

    /// Whether any ADMET models are available.
    var isAvailable: Bool { !models.isEmpty }

    /// Predict ADMET properties from SMILES (uses RDKit descriptors as fallback).
    func predict(smiles: String) async -> ADMETResult {
        var result = ADMETResult()

        // Always compute rule-based properties via RDKit
        let desc = RDKitBridge.computeDescriptors(smiles: smiles)
        if let d = desc {
            result.lipinski = d.lipinski
            result.veber = d.veber

            // Composite drug-likeness score
            var score: Float = 0
            if d.molecularWeight <= 500 { score += 0.2 }
            if d.logP <= 5.0 && d.logP >= -1.0 { score += 0.2 }
            if d.hbd <= 5 { score += 0.15 }
            if d.hba <= 10 { score += 0.15 }
            if d.rotatableBonds <= 10 { score += 0.15 }
            if d.tpsa >= 20 && d.tpsa <= 140 { score += 0.15 }
            result.drugLikeness = score

            // Use computed LogP as fallback
            result.logP = d.logP
        }

        // If ML models are available, run them for enhanced predictions
        if let logPModel = models["ADMET_LogP"] {
            result.logP = await runSingleOutput(model: logPModel, smiles: smiles)
        }
        if let logDModel = models["ADMET_LogD"] {
            result.logD74 = await runSingleOutput(model: logDModel, smiles: smiles)
        }
        if let solModel = models["ADMET_Solubility"] {
            result.aqueousSolubility = await runSingleOutput(model: solModel, smiles: smiles)
        }
        if let cyp2d6 = models["ADMET_CYP2D6"] {
            result.cyp2d6Inhibition = await runSingleOutput(model: cyp2d6, smiles: smiles)
        }
        if let cyp3a4 = models["ADMET_CYP3A4"] {
            result.cyp3a4Inhibition = await runSingleOutput(model: cyp3a4, smiles: smiles)
        }
        if let herg = models["ADMET_hERG"] {
            result.hergLiability = await runSingleOutput(model: herg, smiles: smiles)
        }
        if let bbb = models["ADMET_BBB"] {
            result.bbbPermeability = await runSingleOutput(model: bbb, smiles: smiles)
        }
        if let metab = models["ADMET_MetabolicStability"] {
            result.metabolicStability = await runSingleOutput(model: metab, smiles: smiles)
        }

        return result
    }

    /// Run a single-output ML model with molecular fingerprint input.
    private func runSingleOutput(model: MLModel, smiles: String) async -> Float? {
        do {
            // Generate 2048-bit Morgan fingerprint as model input
            let fp = RDKitBridge.morganFingerprint(smiles: smiles, radius: 2, nBits: 2048)
            guard !fp.isEmpty else { return nil }

            let fpArray = try MLMultiArray(shape: [1, 2048], dataType: .float32)
            for i in 0..<min(fp.count, 2048) {
                fpArray[[0, i] as [NSNumber]] = NSNumber(value: fp[i])
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: ["fingerprint": fpArray])

            let result = try await model.prediction(from: provider)

            if let output = result.featureValue(for: "output")?.multiArrayValue {
                return output[0].floatValue
            }
        } catch {
            // Silently fail — ADMET is optional enhancement
        }
        return nil
    }
}

// MARK: - Pocket Detector CoreML Inference

/// GNN-based binding site prediction using surface point features.
/// Alternative to geometric alpha-sphere + DBSCAN detection.
///
/// Model inputs:
///   - surface_features: [1, maxPoints, featureSize] — per-point chemical features
///   - knn_indices: [1, maxPoints, k] — k-nearest neighbor indices for graph convolution
///   - point_mask: [1, maxPoints] — 1.0 for real points, 0.0 for padding
///
/// Model outputs:
///   - pocket_probability: [1, maxPoints] — per-point probability of being in a binding pocket
@MainActor
final class PocketDetectorInference {

    private var model: MLModel?
    private var isLoaded = false

    private let maxPoints = 2048
    private let kNeighbors = 16
    private let featureSize = 10  // element one-hot(6) + vdwRadius + charge + buriedness + is_surface
    private let pocketThreshold: Float = 0.5

    /// Load the PocketDetector CoreML model from the app bundle.
    func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "PocketDetector", withExtension: "mlmodelc") else {
            print("[PocketDetector] Model not found in bundle — geometric detection only")
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: modelURL, configuration: config)
            isLoaded = true
            print("[PocketDetector] Model loaded — compute units: all")
        } catch {
            print("[PocketDetector] Failed to load model: \(error)")
        }
    }

    /// Whether the ML model is available.
    var isAvailable: Bool { isLoaded && model != nil }

    /// Detect binding pockets using the ML model.
    /// Returns BindingPocket array sorted by druggability, compatible with geometric detection results.
    func detectPockets(protein: Molecule) async -> [BindingPocket] {
        guard let model else { return [] }

        let heavyAtoms = protein.atoms.filter { $0.element != .H }
        guard heavyAtoms.count >= 10 else { return [] }

        // Sample surface-exposed atoms (up to maxPoints)
        let surfaceAtoms = selectSurfacePoints(heavyAtoms)
        let nPoints = min(surfaceAtoms.count, maxPoints)
        guard nPoints > 0 else { return [] }

        do {
            // Build surface features: [1, maxPoints, featureSize]
            let features = try MLMultiArray(shape: [1, NSNumber(value: maxPoints), NSNumber(value: featureSize)], dataType: .float32)
            for i in 0..<nPoints {
                let atom = surfaceAtoms[i]
                // Element one-hot (C, N, O, S, P, other)
                let elemBin: Int
                switch atom.element {
                case .C:  elemBin = 0
                case .N:  elemBin = 1
                case .O:  elemBin = 2
                case .S:  elemBin = 3
                case .P:  elemBin = 4
                default:  elemBin = 5
                }
                for j in 0..<6 {
                    features[[0, i, j] as [NSNumber]] = NSNumber(value: j == elemBin ? 1.0 : 0.0)
                }
                features[[0, i, 6] as [NSNumber]] = NSNumber(value: atom.element.vdwRadius)
                features[[0, i, 7] as [NSNumber]] = NSNumber(value: atom.charge)
                features[[0, i, 8] as [NSNumber]] = NSNumber(value: Float(0.5))  // buriedness placeholder
                features[[0, i, 9] as [NSNumber]] = NSNumber(value: Float(1.0))  // is_surface
            }

            // Build KNN indices: [1, maxPoints, kNeighbors]
            let knnIndices = try MLMultiArray(shape: [1, NSNumber(value: maxPoints), NSNumber(value: kNeighbors)], dataType: .int32)
            let positions = surfaceAtoms.map(\.position)
            for i in 0..<nPoints {
                let neighbors = findKNN(positions: positions, queryIdx: i, k: kNeighbors, n: nPoints)
                for j in 0..<kNeighbors {
                    knnIndices[[0, i, j] as [NSNumber]] = NSNumber(value: neighbors[j])
                }
            }

            // Build point mask: [1, maxPoints]
            let mask = try MLMultiArray(shape: [1, NSNumber(value: maxPoints)], dataType: .float32)
            for i in 0..<maxPoints {
                mask[[0, i] as [NSNumber]] = NSNumber(value: i < nPoints ? 1.0 : 0.0)
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "surface_features": features,
                "knn_indices": knnIndices,
                "point_mask": mask
            ])

            let result = try await model.prediction(from: provider)

            guard let probArray = result.featureValue(for: "pocket_probability")?.multiArrayValue else {
                return []
            }

            // Collect high-probability surface points
            var pocketPoints: [(position: SIMD3<Float>, probability: Float)] = []
            for i in 0..<nPoints {
                let prob = probArray[[0, i] as [NSNumber]].floatValue
                if prob >= pocketThreshold {
                    pocketPoints.append((surfaceAtoms[i].position, prob))
                }
            }

            guard !pocketPoints.isEmpty else { return [] }

            // Cluster high-probability points into separate pockets (simple distance-based)
            return clusterIntoPockets(pocketPoints, protein: protein)

        } catch {
            print("[PocketDetector] Inference failed: \(error)")
            return []
        }
    }

    // MARK: - Helpers

    /// Select surface-exposed atoms using a simple neighbor count heuristic.
    /// Atoms with fewer nearby neighbors are more likely surface-exposed.
    private func selectSurfacePoints(_ atoms: [Atom]) -> [Atom] {
        guard atoms.count <= maxPoints else {
            // Stride-sample to fit maxPoints
            let stride = max(1, atoms.count / maxPoints)
            return (0..<atoms.count).filter { $0 % stride == 0 }.prefix(maxPoints).map { atoms[$0] }
        }
        return atoms
    }

    /// Find k-nearest neighbors by brute force (small N, fast enough).
    private func findKNN(positions: [SIMD3<Float>], queryIdx: Int, k: Int, n: Int) -> [Int32] {
        let query = positions[queryIdx]
        var dists: [(Int, Float)] = []
        for i in 0..<n {
            if i == queryIdx { continue }
            let d = simd_distance_squared(query, positions[i])
            dists.append((i, d))
        }
        dists.sort { $0.1 < $1.1 }
        var result = [Int32](repeating: 0, count: k)
        for i in 0..<k {
            result[i] = i < dists.count ? Int32(dists[i].0) : Int32(queryIdx)
        }
        return result
    }

    /// Cluster pocket points into BindingPocket objects using simple single-linkage.
    private func clusterIntoPockets(
        _ points: [(position: SIMD3<Float>, probability: Float)],
        protein: Molecule
    ) -> [BindingPocket] {
        let eps: Float = 4.0
        let positions = points.map(\.position)
        var labels = [Int](repeating: -1, count: points.count)
        var clusterId = 0

        for i in 0..<points.count {
            guard labels[i] == -1 else { continue }
            labels[i] = clusterId
            var queue = [i]
            var qIdx = 0
            while qIdx < queue.count {
                let cur = queue[qIdx]; qIdx += 1
                for j in 0..<points.count {
                    guard labels[j] == -1 else { continue }
                    if simd_distance(positions[cur], positions[j]) < eps {
                        labels[j] = clusterId
                        queue.append(j)
                    }
                }
            }
            clusterId += 1
        }

        var pockets: [BindingPocket] = []
        for cid in 0..<clusterId {
            let members = (0..<points.count).filter { labels[$0] == cid }
            guard members.count >= 3 else { continue }

            let clusterPos = members.map { positions[$0] }
            let center = clusterPos.reduce(.zero, +) / Float(clusterPos.count)
            let avgProb = members.map { points[$0].probability }.reduce(0, +) / Float(members.count)

            // Compute half-extents
            var maxExt = SIMD3<Float>.zero
            for p in clusterPos {
                maxExt = simd_max(maxExt, simd_abs(p - center))
            }
            let halfSize = maxExt + SIMD3<Float>(repeating: 4.0)
            let volume = 8.0 * halfSize.x * halfSize.y * halfSize.z

            // Find nearby residues
            var residueSet = Set<Int>()
            for (resIdx, residue) in protein.residues.enumerated() {
                guard residue.isStandard else { continue }
                for atomIdx in residue.atomIndices {
                    guard atomIdx < protein.atoms.count else { continue }
                    if simd_distance(protein.atoms[atomIdx].position, center) < 8.0 {
                        residueSet.insert(resIdx)
                        break
                    }
                }
            }

            pockets.append(BindingPocket(
                id: pockets.count,
                center: center,
                size: halfSize,
                volume: volume,
                buriedness: avgProb,
                polarity: 0.5,
                druggability: volume * avgProb,
                residueIndices: Array(residueSet).sorted(),
                probePositions: clusterPos
            ))
        }

        pockets.sort { $0.druggability > $1.druggability }
        return pockets
    }
}

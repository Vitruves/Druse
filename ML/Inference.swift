import Foundation
@preconcurrency import CoreML
import Metal
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
        var pairDistances: [Float]             // flattened NxM distance matrix
        var pairRBF: [Float]                   // flattened NxMx50 RBF-encoded distances
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

        // Compute pairwise distance matrix + RBF encoding
        let nProt = protPositions.count
        let nLig = ligPositions.count
        let nPairs = nProt * nLig

        // Try GPU-accelerated RBF computation, fall back to CPU
        let distances: [Float]
        let rbfEncoded: [Float]

        if let gpu = FeatureComputeAccelerator.shared,
           let result = gpu.computeRBF(protPositions: protPositions, ligPositions: ligPositions) {
            distances = result.distances
            rbfEncoded = result.rbf
        } else {
            // CPU fallback with flat arrays
            var cpuDist = [Float](repeating: 0, count: nPairs)
            var cpuRBF = [Float](repeating: 0, count: nPairs * 50)
            for i in 0..<nProt {
                for j in 0..<nLig {
                    let idx = i * nLig + j
                    let d = simd_distance(protPositions[i], ligPositions[j])
                    cpuDist[idx] = d
                    let rbfBase = idx * 50
                    for k in 0..<50 {
                        let center = Float(k) * 0.2
                        let diff = d - center
                        cpuRBF[rbfBase + k] = exp(-rbfGamma * diff * diff)
                    }
                }
            }
            distances = cpuDist
            rbfEncoded = cpuRBF
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
            // Zero-copy MLMultiArray: allocate contiguous Float buffer and wrap directly.
            // Avoids per-element NSNumber boxing which dominates inference prep time.
            func makeMLArray(shape: [Int], fill: (UnsafeMutablePointer<Float>) -> Void) throws -> MLMultiArray {
                let totalCount = shape.reduce(1, *)
                let byteCount = totalCount * MemoryLayout<Float>.stride
                let raw = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<Float>.alignment)
                let floatPtr = raw.bindMemory(to: Float.self, capacity: totalCount)
                fill(floatPtr)
                let strides = shape.indices.map { i in
                    NSNumber(value: shape[(i+1)...].reduce(1, *))
                }
                return try MLMultiArray(
                    dataPointer: raw,
                    shape: shape.map { NSNumber(value: $0) },
                    dataType: .float32,
                    strides: strides,
                    deallocator: { ptr in ptr.deallocate() }
                )
            }

            // Protein features: [1, nProt, featSize]
            let protArray = try makeMLArray(shape: [1, nProt, featSize]) { ptr in
                for i in 0..<nProt {
                    let base = i * featSize
                    for j in 0..<featSize {
                        ptr[base + j] = features.proteinFeatures[i][j]
                    }
                }
            }

            // Ligand features: [1, nLig, featSize]
            let ligArray = try makeMLArray(shape: [1, nLig, featSize]) { ptr in
                for i in 0..<nLig {
                    let base = i * featSize
                    for j in 0..<featSize {
                        ptr[base + j] = features.ligandFeatures[i][j]
                    }
                }
            }

            // Protein positions: [1, nProt, 3]
            let protPos = try makeMLArray(shape: [1, nProt, 3]) { ptr in
                for i in 0..<nProt {
                    let base = i * 3
                    ptr[base] = features.proteinPositions[i].x
                    ptr[base + 1] = features.proteinPositions[i].y
                    ptr[base + 2] = features.proteinPositions[i].z
                }
            }

            // Ligand positions: [1, nLig, 3]
            let ligPos = try makeMLArray(shape: [1, nLig, 3]) { ptr in
                for i in 0..<nLig {
                    let base = i * 3
                    ptr[base] = features.ligandPositions[i].x
                    ptr[base + 1] = features.ligandPositions[i].y
                    ptr[base + 2] = features.ligandPositions[i].z
                }
            }

            // RBF-encoded pairwise distances: [1, nProt, nLig, 50] — direct memcpy from flat buffer
            let rbfArray = try makeMLArray(shape: [1, nProt, nLig, 50]) { ptr in
                let count = min(features.pairRBF.count, nProt * nLig * 50)
                features.pairRBF.withUnsafeBufferPointer { src in
                    ptr.update(from: src.baseAddress!, count: count)
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
/// Model inputs (matching train_pocket_detector.py features):
///   - surface_features: [1, N, 11] — per-point: normal(3) + dist + hydrophobicity + charge +
///                                     aromatic + donor + acceptor + buriedness + curvature
///   - neighbor_features: [1, N, 11] — mean of k-nearest neighbor features (pre-aggregated GCN)
///
/// Model outputs:
///   - pocket_probability: [1, N, 1] — per-point probability of being in a binding pocket
@MainActor
final class PocketDetectorInference {

    private var model: MLModel?
    private var isLoaded = false

    private nonisolated let maxPoints = 4096
    private nonisolated let kNeighbors = 16
    private nonisolated let featureSize = 11  // matches training SURFACE_FEAT_DIM
    private nonisolated let pocketThreshold: Float = 0.49  // optimal threshold from training

    // Kyte-Doolittle hydrophobicity scale (normalized by /4.5, matching training)
    private nonisolated static let hydrophobicity: [String: Float] = [
        "ALA":  1.8 / 4.5, "ARG": -4.5 / 4.5, "ASN": -3.5 / 4.5, "ASP": -3.5 / 4.5,
        "CYS":  2.5 / 4.5, "GLU": -3.5 / 4.5, "GLN": -3.5 / 4.5, "GLY": -0.4 / 4.5,
        "HIS": -3.2 / 4.5, "ILE":  4.5 / 4.5, "LEU":  3.8 / 4.5, "LYS": -3.9 / 4.5,
        "MET":  1.9 / 4.5, "PHE":  2.8 / 4.5, "PRO": -1.6 / 4.5, "SER": -0.8 / 4.5,
        "THR": -0.7 / 4.5, "TRP": -0.9 / 4.5, "TYR": -1.3 / 4.5, "VAL":  4.2 / 4.5,
    ]

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

        // Snapshot all data we need on MainActor before moving off-thread
        let heavyAtoms = protein.atoms.filter { $0.element != .H && !$0.isHetAtom }
        guard heavyAtoms.count >= 10 else { return [] }
        let residueSnapshot = protein.residues
        let allAtoms = protein.atoms

        // Run heavy computation off the main actor
        let (surfaceData, nPoints): ([SurfacePoint], Int) = await Task.detached(priority: .userInitiated) { [maxPoints, featureSize] in
            let sd = Self.computeSurfaceFeatures(atoms: heavyAtoms, maxPoints: maxPoints, featureSize: featureSize)
            return (sd, sd.count)
        }.value

        guard nPoints >= 10 else { return [] }

        do {
            // Build MLMultiArrays off the main actor
            let (features, neighborFeatures) = try await Task.detached(priority: .userInitiated) { [featureSize, kNeighbors] in
                let n = NSNumber(value: nPoints)

                let features = try MLMultiArray(
                    shape: [1, n, NSNumber(value: featureSize)],
                    dataType: .float32
                )
                for i in 0..<nPoints {
                    let feat = surfaceData[i].features
                    for j in 0..<featureSize {
                        features[[0, i, j] as [NSNumber]] = NSNumber(value: feat[j])
                    }
                }

                let neighborFeatures = try MLMultiArray(
                    shape: [1, n, NSNumber(value: featureSize)],
                    dataType: .float32
                )
                let positions = surfaceData.map(\.position)
                for i in 0..<nPoints {
                    let neighbors = Self.findKNN(positions: positions, queryIdx: i, k: kNeighbors, n: nPoints)
                    var meanFeat = [Float](repeating: 0, count: featureSize)
                    for ni in neighbors {
                        let nFeat = surfaceData[Int(ni)].features
                        for j in 0..<featureSize { meanFeat[j] += nFeat[j] }
                    }
                    let scale = 1.0 / Float(neighbors.count)
                    for j in 0..<featureSize {
                        neighborFeatures[[0, i, j] as [NSNumber]] = NSNumber(value: meanFeat[j] * scale)
                    }
                }

                return (features, neighborFeatures)
            }.value

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "surface_features": features,
                "neighbor_features": neighborFeatures,
            ])

            let result = try await model.prediction(from: provider)

            guard let probArray = result.featureValue(for: "pocket_probability")?.multiArrayValue else {
                print("[PocketDetector] Missing 'pocket_probability' in model output")
                return []
            }

            // Collect high-probability surface points and cluster off main actor
            let threshold = pocketThreshold
            return await Task.detached(priority: .userInitiated) {
                var pocketPoints: [(position: SIMD3<Float>, probability: Float)] = []
                for i in 0..<nPoints {
                    let prob = probArray[[0, i, 0] as [NSNumber]].floatValue
                    if prob >= threshold {
                        pocketPoints.append((surfaceData[i].position, prob))
                    }
                }

                guard !pocketPoints.isEmpty else { return [] }

                return Self.clusterIntoPockets(pocketPoints, residues: residueSnapshot, atoms: allAtoms)
            }.value

        } catch {
            print("[PocketDetector] Inference failed: \(error)")
            return []
        }
    }

    // MARK: - Surface Feature Computation (matches training pipeline)

    private struct SurfacePoint: Sendable {
        let position: SIMD3<Float>
        let features: [Float]  // 11-dim matching SURFACE_FEAT_DIM
    }

    /// Compute surface points and 11-dim features matching train_pocket_detector.py.
    ///
    /// Features: normal(3) + dist_to_atom + hydrophobicity + charge +
    ///           aromatic_nearby + donor_nearby + acceptor_nearby + buriedness + curvature
    private nonisolated static func computeSurfaceFeatures(atoms: [Atom], maxPoints: Int, featureSize: Int) -> [SurfacePoint] {
        let atomPositions = atoms.map(\.position)

        // Compute bounding box for grid generation
        var bboxMin = SIMD3<Float>(repeating: .infinity)
        var bboxMax = SIMD3<Float>(repeating: -.infinity)
        for pos in atomPositions {
            bboxMin = simd_min(bboxMin, pos)
            bboxMax = simd_max(bboxMax, pos)
        }
        bboxMin -= SIMD3<Float>(repeating: 6.0)
        bboxMax += SIMD3<Float>(repeating: 6.0)

        // Generate grid points (spacing = 1.0 A, matching training)
        let spacing: Float = 1.0
        var gridPoints: [SIMD3<Float>] = []
        var x = bboxMin.x
        while x < bboxMax.x {
            var y = bboxMin.y
            while y < bboxMax.y {
                var z = bboxMin.z
                while z < bboxMax.z {
                    gridPoints.append(SIMD3<Float>(x, y, z))
                    z += spacing
                }
                y += spacing
            }
            x += spacing
        }

        // VdW radii matching training
        func vdwRadius(for element: Element) -> Float {
            switch element {
            case .C: return 1.7
            case .N: return 1.55
            case .O: return 1.52
            case .S: return 1.8
            case .P: return 1.8
            default: return 1.7
            }
        }

        // Filter to surface region: distance to nearest atom surface ~ probe_radius
        let probeRadius: Float = 1.4
        var surfacePoints: [SurfacePoint] = []

        for gridPt in gridPoints {
            // Find nearest atom
            var minDist: Float = .infinity
            var nearestIdx = 0
            for (ai, aPos) in atomPositions.enumerated() {
                let d = simd_distance(gridPt, aPos)
                if d < minDist {
                    minDist = d
                    nearestIdx = ai
                }
            }

            let surfaceDist = minDist - vdwRadius(for: atoms[nearestIdx].element)
            guard surfaceDist >= -0.5 && surfaceDist <= probeRadius + 0.5 else { continue }

            let atom = atoms[nearestIdx]
            let atomPos = atomPositions[nearestIdx]

            // Feature [0-2]: surface normal (point - nearest atom, normalized)
            var normal = gridPt - atomPos
            let normLen = simd_length(normal)
            if normLen > 0 { normal /= normLen }

            // Feature [3]: distance to nearest atom
            let distFeat = normLen

            // Feature [4]: hydrophobicity of nearest residue (normalized)
            let resName = atom.residueName
            let hydro = Self.hydrophobicity[resName] ?? 0.0

            // Feature [5]: charge
            let charge = atom.charge

            // Feature [6]: aromatic proxy (is carbon)
            let aromatic: Float = atom.element == .C ? 1.0 : 0.0

            // Feature [7]: H-bond donor nearby
            let donor: Float = (atom.element == .N || atom.element == .O) ? 1.0 : 0.0

            // Feature [8]: H-bond acceptor nearby
            let acceptor: Float = (atom.element == .N || atom.element == .O || atom.element == .F) ? 1.0 : 0.0

            // Feature [9]: buriedness (count atoms within 6A, normalized)
            var nearbyCount = 0
            for aPos in atomPositions {
                if simd_distance(gridPt, aPos) <= 6.0 { nearbyCount += 1 }
            }
            let buriedness = min(Float(nearbyCount) / 20.0, 1.0)

            // Feature [10]: curvature placeholder (matching training)
            let curvature: Float = 0.5

            surfacePoints.append(SurfacePoint(
                position: gridPt,
                features: [normal.x, normal.y, normal.z, distFeat, hydro, charge,
                           aromatic, donor, acceptor, buriedness, curvature]
            ))
        }

        // Subsample to maxPoints (matching training's 5000 cap, but we use maxPoints)
        if surfacePoints.count > maxPoints {
            // Stride-sample to fit
            let stride = max(1, surfacePoints.count / maxPoints)
            surfacePoints = (0..<surfacePoints.count)
                .filter { $0 % stride == 0 }
                .prefix(maxPoints)
                .map { surfacePoints[$0] }
        }

        return surfacePoints
    }

    /// Find k-nearest neighbors by brute force.
    private nonisolated static func findKNN(positions: [SIMD3<Float>], queryIdx: Int, k: Int, n: Int) -> [Int32] {
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
    private nonisolated static func clusterIntoPockets(
        _ points: [(position: SIMD3<Float>, probability: Float)],
        residues: [Residue],
        atoms: [Atom]
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
            for (resIdx, residue) in residues.enumerated() {
                guard residue.isStandard else { continue }
                for atomIdx in residue.atomIndices {
                    guard atomIdx < atoms.count else { continue }
                    if simd_distance(atoms[atomIdx].position, center) < 8.0 {
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

// MARK: - Metal-Accelerated Feature Compute

/// Singleton Metal accelerator for RBF distance encoding.
/// Falls back gracefully to CPU if Metal is unavailable.
private final class FeatureComputeAccelerator {
    nonisolated(unsafe) static let shared: FeatureComputeAccelerator? = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let rbfFunction = library.makeFunction(name: "computeRBFDistances")
        else { return nil }

        do {
            let pipeline = try device.makeComputePipelineState(function: rbfFunction)
            return FeatureComputeAccelerator(device: device, commandQueue: commandQueue, rbfPipeline: pipeline)
        } catch {
            return nil
        }
    }()

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rbfPipeline: MTLComputePipelineState

    private init(device: MTLDevice, commandQueue: MTLCommandQueue, rbfPipeline: MTLComputePipelineState) {
        self.device = device
        self.commandQueue = commandQueue
        self.rbfPipeline = rbfPipeline
    }

    /// Compute pairwise distances and RBF-encoded features on GPU.
    /// Returns flat arrays: distances[nProt * nLig], rbf[nProt * nLig * 50].
    func computeRBF(protPositions: [SIMD3<Float>], ligPositions: [SIMD3<Float>]) -> (distances: [Float], rbf: [Float])? {
        let nProt = protPositions.count
        let nLig = ligPositions.count
        guard nProt > 0, nLig > 0 else { return nil }

        let nPairs = nProt * nLig
        var protPos = protPositions
        var ligPos = ligPositions
        var params = RBFParams(nProt: UInt32(nProt), nLig: UInt32(nLig), numBins: 50, gamma: 10.0, binSpacing: 0.2, _pad0: 0)

        guard let protBuffer = device.makeBuffer(bytes: &protPos, length: nProt * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let ligBuffer = device.makeBuffer(bytes: &ligPos, length: nLig * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let rbfBuffer = device.makeBuffer(length: nPairs * 50 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let distBuffer = device.makeBuffer(length: nPairs * MemoryLayout<Float>.stride, options: .storageModeShared),
              let paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<RBFParams>.stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else { return nil }

        enc.setComputePipelineState(rbfPipeline)
        enc.setBuffer(protBuffer, offset: 0, index: 0)
        enc.setBuffer(ligBuffer, offset: 0, index: 1)
        enc.setBuffer(rbfBuffer, offset: 0, index: 2)
        enc.setBuffer(distBuffer, offset: 0, index: 3)
        enc.setBuffer(paramsBuffer, offset: 0, index: 4)

        let tgSize = MTLSize(width: min(nPairs, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (nPairs + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let distPtr = distBuffer.contents().bindMemory(to: Float.self, capacity: nPairs)
        let rbfPtr = rbfBuffer.contents().bindMemory(to: Float.self, capacity: nPairs * 50)

        return (
            distances: Array(UnsafeBufferPointer(start: distPtr, count: nPairs)),
            rbf: Array(UnsafeBufferPointer(start: rbfPtr, count: nPairs * 50))
        )
    }
}

import Foundation
@preconcurrency import CoreML
import Metal
import simd

// MARK: - Druse-Score Feature Extraction

/// Extracts geometric + chemical features from a protein-ligand complex
/// for CoreML inference. Produces the input tensors expected by DruseScore.mlmodel.
struct DruseScoreFeatureExtractor {

    /// Atom-level features (v2): [atomicNum one-hot(10), aromaticity, charge, hbDonor, hbAcceptor,
    /// hybridization one-hot(3), is_ligand, formal_charge, in_ring] = 20 features per atom
    static let atomFeatureSize = 20

    /// Known ring atoms in standard amino acid residues (for protein ring membership feature)
    private static let ringResidueAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "PRO": ["N", "CA", "CB", "CG", "CD"],
    ]

    /// Max atoms in pocket + ligand (pad/truncate to this)
    static let maxAtoms = 512

    /// Radial basis function centers for distance encoding (0-10 Å, 50 bins)
    /// Must match torch.linspace(0, 10, 50) — spacing is 10/49 ≈ 0.20408, NOT 10/50
    static let rbfCenters: [Float] = (0..<50).map { Float($0) * (10.0 / 49.0) }
    static let rbfGamma: Float = 10.0 // 1/(2σ²) where σ=0.2236

    struct ComplexFeatures {
        var proteinPositions: [SIMD3<Float>]  // pocket atom positions
        var proteinFeatures: [[Float]]         // per-atom feature vectors
        var ligandPositions: [SIMD3<Float>]
        var ligandFeatures: [[Float]]
        var pairDistances: [Float]             // flattened NxM distance matrix
        var pairRBF: [Float]                   // flattened NxMx50 RBF-encoded distances
    }

    /// Determine hybridization for each atom based on bond orders.
    /// Returns a dictionary mapping atom ID → Hybridization.
    enum Hybridization { case sp, sp2, sp3 }

    struct AtomChemInfo {
        var hybridization: Hybridization
        var isAromatic: Bool
    }

    static func buildHybridizationMap(atoms: [Atom], bonds: [Bond]) -> [Int: AtomChemInfo] {
        // Build adjacency: atom index → list of bond orders
        var atomBonds: [Int: [BondOrder]] = [:]
        for bond in bonds {
            atomBonds[bond.atomIndex1, default: []].append(bond.order)
            atomBonds[bond.atomIndex2, default: []].append(bond.order)
        }

        var result: [Int: AtomChemInfo] = [:]
        for atom in atoms {
            let orders = atomBonds[atom.id] ?? []
            let hasAromatic = orders.contains(.aromatic)
            let hasDouble = orders.contains(.double)
            let hasTriple = orders.contains(.triple)

            let hyb: Hybridization
            if hasTriple {
                hyb = .sp
            } else if hasAromatic || hasDouble {
                hyb = .sp2
            } else {
                hyb = .sp3
            }
            result[atom.id] = AtomChemInfo(hybridization: hyb, isAromatic: hasAromatic)
        }
        return result
    }

    /// Extract features from a docked protein-ligand complex.
    static func extract(
        proteinAtoms: [Atom],
        ligandAtoms: [Atom],
        pocketCenter: SIMD3<Float>,
        pocketRadius: Float = 10.0,
        proteinBonds: [Bond] = [],
        ligandBonds: [Bond] = []
    ) -> ComplexFeatures {
        // Filter protein atoms to pocket region
        let pocketAtoms = proteinAtoms.filter { atom in
            simd_distance(atom.position, pocketCenter) <= pocketRadius
        }

        let protHybrid = buildHybridizationMap(atoms: pocketAtoms, bonds: proteinBonds)
        let ligHybrid = buildHybridizationMap(atoms: ligandAtoms, bonds: ligandBonds)

        let protPositions = pocketAtoms.map(\.position)
        let ligPositions = ligandAtoms.map(\.position)

        let protFeats = pocketAtoms.map { atomFeatures($0, isProtein: true, chemInfo: protHybrid[$0.id]) }
        let ligFeats = ligandAtoms.map { atomFeatures($0, isProtein: false, chemInfo: ligHybrid[$0.id]) }

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
                        let center = Float(k) * (10.0 / 49.0)
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

    // ── Residue-level lookup tables (must match train_druse_pKi_v2.py exactly) ──

    /// Aromatic atoms per residue (training: AROMATIC_RESIDUE_ATOMS)
    private static let aromaticResidueAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    ]

    /// Partial charges per (residue, atom) (training: RESIDUE_ATOM_CHARGES)
    private static let residueAtomCharges: [String: Float] = [
        "ASP:OD1": -0.5, "ASP:OD2": -0.5,
        "GLU:OE1": -0.5, "GLU:OE2": -0.5,
        "LYS:NZ":   1.0,
        "ARG:NH1":  0.33, "ARG:NH2": 0.33, "ARG:NE": 0.33,
        "HIS:ND1":  0.25, "HIS:NE2": 0.25,
    ]

    /// H-bond donors per residue (training: HBD_RESIDUE_ATOMS)
    private static let hbdResidueAtoms: [String: Set<String>] = [
        "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
        "ASN": ["ND2"], "GLN": ["NE2"],
        "LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"],
        "HIS": ["ND1", "NE2"], "TRP": ["NE1"],
        "CYS": ["SG"],
    ]

    /// H-bond acceptors per residue (training: HBA_RESIDUE_ATOMS)
    private static let hbaResidueAtoms: [String: Set<String>] = [
        "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"],
        "ASN": ["OD1"], "GLN": ["OE1"],
        "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
        "HIS": ["ND1", "NE2"],
        "MET": ["SD"], "CYS": ["SG"],
    ]

    /// SP2 atoms per residue (training: SP2_RESIDUE_ATOMS)
    private static let sp2ResidueAtoms: [String: Set<String>] = [
        "ASP": ["CG", "OD1", "OD2"], "GLU": ["CD", "OE1", "OE2"],
        "ASN": ["CG", "OD1", "ND2"], "GLN": ["CD", "OE1", "NE2"],
        "ARG": ["CZ", "NH1", "NH2"],
    ]

    /// Formal charges per (residue, atom) (training: RESIDUE_FORMAL_CHARGES)
    private static let residueFormalCharges: [String: Float] = [
        "ASP:OD1": -1, "ASP:OD2": -1,
        "GLU:OE1": -1, "GLU:OE2": -1,
        "LYS:NZ":   1,
        "ARG:NH1":  1, "ARG:NH2": 1, "ARG:NE": 1, "ARG:CZ": 1,
        "HIS:ND1":  0, "HIS:NE2": 0,
    ]

    /// Encode atom as 20-dimensional feature vector (v2).
    /// Must exactly match pdb_atom_features() / mol2_atom_features() in train_druse_pKi_v2.py.
    static func atomFeatures(_ atom: Atom, isProtein: Bool, chemInfo: AtomChemInfo? = nil) -> [Float] {
        var features = [Float](repeating: 0, count: atomFeatureSize)

        // [0-9] One-hot encode element (H, C, N, O, F, P, S, Cl, Br, other)
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

        if isProtein {
            // ── Protein features: residue-level lookups (match pdb_atom_features) ──
            let res = atom.residueName
            let name = atom.name

            // [10] Aromatic — residue-specific atom set
            let aroSet = aromaticResidueAtoms[res] ?? []
            let isAromatic = aroSet.contains(name)
            features[10] = isAromatic ? 1.0 : 0.0

            // [11] Partial charge — residue-level fixed charges (NOT Gasteiger)
            let key = "\(res):\(name)"
            features[11] = residueAtomCharges[key] ?? 0.0

            // [12] H-bond donor — backbone N (not PRO) + sidechain donors
            let isBackboneHBD = (name == "N" && res != "PRO")
            let isSidechainHBD = (hbdResidueAtoms[res] ?? []).contains(name)
            features[12] = (isBackboneHBD || isSidechainHBD) ? 1.0 : 0.0

            // [13] H-bond acceptor — backbone O + sidechain acceptors
            let isBackboneHBA = (name == "O")
            let isSidechainHBA = (hbaResidueAtoms[res] ?? []).contains(name)
            features[13] = (isBackboneHBA || isSidechainHBA) ? 1.0 : 0.0

            // [14-16] Hybridization
            let isSP2 = isAromatic
                || (sp2ResidueAtoms[res] ?? []).contains(name)
                || (name == "C" && atom.element == .C)
                || (name == "O" && atom.element == .O)
            features[14] = 0.0         // sp (not assigned for protein in training)
            features[15] = isSP2 ? 1.0 : 0.0
            features[16] = isSP2 ? 0.0 : 1.0

            // [17] Is ligand
            features[17] = 0.0

            // [18] Formal charge — residue-level (NOT atom.formalCharge)
            features[18] = residueFormalCharges[key] ?? 0.0

            // [19] Ring membership
            let ringSet = ringResidueAtoms[res] ?? []
            features[19] = ringSet.contains(name) ? 1.0 : 0.0

        } else {
            // ── Ligand features: element/hybridization based (match mol2_atom_features) ──
            let info = chemInfo ?? AtomChemInfo(hybridization: .sp3, isAromatic: false)
            let hyb = info.hybridization
            // Aromatic only when atom has aromatic bonds (matches SYBYL "ar" subtype)
            let isAromatic = info.isAromatic

            // [10] Aromatic
            features[10] = isAromatic ? 1.0 : 0.0

            // [11] Partial charge (Gasteiger — same as MOL2 training data)
            features[11] = atom.charge

            // [12] H-bond donor — matches SYBYL subtype logic:
            //   N: sp3 ("3"), sp3-positive ("4"), amide ("am"), planar ("pl3")
            //   O: sp3 only; S: sp3 only
            var isHBD = false
            if atom.element == .N && (hyb == .sp3 || (hyb == .sp2 && !isAromatic)) { isHBD = true }
            if atom.element == .O && hyb == .sp3 { isHBD = true }
            if atom.element == .S && hyb == .sp3 { isHBD = true }
            features[12] = isHBD ? 1.0 : 0.0

            // [13] H-bond acceptor — N, O, S, F (training includes S!)
            let isHBA = atom.element == .N || atom.element == .O
                || atom.element == .S || atom.element == .F
            features[13] = isHBA ? 1.0 : 0.0

            // [14-16] Hybridization
            switch hyb {
            case .sp:  features[14] = 1.0
            case .sp2: features[15] = 1.0
            case .sp3: features[16] = 1.0
            }

            // [17] Is ligand
            features[17] = 1.0

            // [18] Formal charge
            var formal: Float = 0
            if abs(atom.charge) > 0.5 { formal = Float(Int(atom.charge.rounded())) }
            // Quaternary nitrogen → +1
            if atom.element == .N && hyb == .sp3 && atom.charge > 0.3 { formal = 1 }
            features[18] = formal

            // [19] Ring membership (aromatic proxy — matches training)
            features[19] = isAromatic ? 1.0 : 0.0
        }

        return features
    }

    /// Radial basis function encoding of a distance.
    private static func rbfEncode(_ distance: Float) -> [Float] {
        rbfCenters.map { center in
            exp(-rbfGamma * (distance - center) * (distance - center))
        }
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
                    ActivityLog.shared.warn("[ADMET] Failed to load \(name): \(error)", category: .system)
                }
            }
        }

        if !models.isEmpty {
            ActivityLog.shared.info("[ADMET] Loaded \(models.count) ADMET models", category: .system)
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
            guard !fp.isEmpty else {
                ActivityLog.shared.debug("[ADMET] Empty fingerprint for SMILES: \(smiles.prefix(40))", category: .dock)
                return nil
            }

            let fpArray = try MLMultiArray(shape: [1, 2048], dataType: .float32)
            for i in 0..<min(fp.count, 2048) {
                fpArray[[0, i] as [NSNumber]] = NSNumber(value: fp[i])
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: ["fingerprint": fpArray])

            let result = try await model.prediction(from: provider)

            if let output = result.featureValue(for: "output")?.multiArrayValue {
                return output[0].floatValue
            }
            ActivityLog.shared.debug("[ADMET] Missing 'output' feature in model result", category: .dock)
        } catch {
            ActivityLog.shared.debug("[ADMET] Inference failed for SMILES \(smiles.prefix(40)): \(error)", category: .dock)
        }
        return nil
    }
}

// MARK: - Pocket Detector CoreML Inference

/// GNN-based binding site prediction using surface point features.
/// 3D U-Net pocket detector: voxelizes protein into a 64³ grid with 10 feature channels,
/// runs CoreML inference, thresholds per-voxel probabilities, and clusters into binding sites.
@MainActor
final class PocketDetectorInference {

    private var model: MLModel?
    private var isLoaded = false
    private var pocketThreshold: Float = 0.01  // overridden from model metadata

    private nonisolated static let gridSize = 64
    private nonisolated static let nChannels = 10
    private nonisolated static let resolution: Float = 1.0  // Angstroms per voxel

    // Kyte-Doolittle hydrophobicity scale (normalized by /4.5, matching training)
    private nonisolated static let hydrophobicity: [String: Float] = [
        "ALA":  1.8 / 4.5, "ARG": -4.5 / 4.5, "ASN": -3.5 / 4.5, "ASP": -3.5 / 4.5,
        "CYS":  2.5 / 4.5, "GLU": -3.5 / 4.5, "GLN": -3.5 / 4.5, "GLY": -0.4 / 4.5,
        "HIS": -3.2 / 4.5, "ILE":  4.5 / 4.5, "LEU":  3.8 / 4.5, "LYS": -3.9 / 4.5,
        "MET":  1.9 / 4.5, "PHE":  2.8 / 4.5, "PRO": -1.6 / 4.5, "SER": -0.8 / 4.5,
        "THR": -0.7 / 4.5, "TRP": -0.9 / 4.5, "TYR": -1.3 / 4.5, "VAL":  4.2 / 4.5,
    ]

    // Partial charges at pH 7.4 (matching training)
    private nonisolated static let partialCharges: [String: [String: Float]] = [
        "ASP": ["OD1": -0.5, "OD2": -0.5],
        "GLU": ["OE1": -0.5, "OE2": -0.5],
        "LYS": ["NZ": 1.0],
        "ARG": ["NH1": 0.33, "NH2": 0.33, "NE": 0.33],
        "HIS": ["ND1": 0.25, "NE2": 0.25],
    ]

    // Aromatic ring atoms in standard residues
    private nonisolated static let aromaticResidueAtoms: Set<String> = ["PHE", "TYR", "TRP", "HIS"]

    // H-bond donor atoms (backbone N except PRO, plus side-chain donors)
    private nonisolated static let hbondDonorAtoms: [String: Set<String>] = [
        "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "CYS": ["SG"],
        "ASN": ["ND2"], "GLN": ["NE2"], "TRP": ["NE1"],
        "LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"], "HIS": ["ND1", "NE2"],
    ]

    // H-bond acceptor atoms (backbone O, plus side-chain acceptors)
    private nonisolated static let hbondAcceptorAtoms: [String: Set<String>] = [
        "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
        "ASN": ["OD1"], "ASP": ["OD1", "OD2"], "GLN": ["OE1"], "GLU": ["OE1", "OE2"],
        "HIS": ["ND1", "NE2"], "MET": ["SD"],
    ]

    /// Load the PocketDetector CoreML model from the app bundle.
    func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "PocketDetector", withExtension: "mlmodelc") else {
            ActivityLog.shared.warn("[PocketDetector] Model not found in bundle — geometric detection only", category: .system)
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: modelURL, configuration: config)
            isLoaded = true

            // Read threshold from model metadata
            if let threshStr = model?.modelDescription.metadata[MLModelMetadataKey(rawValue: "pocket_threshold")] as? String,
               let thresh = Float(threshStr) {
                pocketThreshold = thresh
            }

            ActivityLog.shared.info("[PocketDetector] v3 loaded (3D U-Net, threshold=\(String(format: "%.6f", pocketThreshold)))", category: .system)
        } catch {
            ActivityLog.shared.error("[PocketDetector] Failed to load model: \(error)", category: .system)
        }
    }

    /// Whether the ML model is available.
    var isAvailable: Bool { isLoaded && model != nil }

    /// Detect binding pockets using the 3D U-Net model.
    /// Returns BindingPocket array sorted by druggability, compatible with geometric detection results.
    func detectPockets(protein: Molecule) async -> [BindingPocket] {
        guard let model else {
            ActivityLog.shared.warn("[PocketDetector] detectPockets: model not loaded", category: .dock)
            return []
        }

        let heavyAtoms = protein.atoms.filter { $0.element != .H && !$0.isHetAtom }
        guard heavyAtoms.count >= 10 else {
            ActivityLog.shared.warn("[PocketDetector] detectPockets: too few heavy atoms (\(heavyAtoms.count))", category: .dock)
            return []
        }
        let residueSnapshot = protein.residues
        let allAtoms = protein.atoms
        let threshold = pocketThreshold

        // Voxelize protein on background thread
        let (voxelGrid, center) = await Task.detached(priority: .userInitiated) {
            Self.voxelizeProtein(heavyAtoms: heavyAtoms, residues: residueSnapshot)
        }.value

        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "voxel_grid": voxelGrid,
            ])
            let result = try await model.prediction(from: provider)

            guard let probGrid = result.featureValue(for: "pocket_probability")?.multiArrayValue else {
                ActivityLog.shared.error("[PocketDetector] Missing 'pocket_probability' in model output", category: .dock)
                return []
            }

            // Extract pocket voxels and cluster on background thread
            return await Task.detached(priority: .userInitiated) {
                let pocketPoints = Self.extractPocketPoints(probGrid: probGrid, center: center, threshold: threshold)
                guard !pocketPoints.isEmpty else { return [] }
                return Self.clusterIntoPockets(pocketPoints, residues: residueSnapshot, atoms: allAtoms)
            }.value
        } catch {
            ActivityLog.shared.warn("[PocketDetector] Inference failed: \(error)", category: .dock)
            return []
        }
    }

    // MARK: - Voxelization

    /// Voxelize protein atoms into a 10-channel 64³ grid for the 3D U-Net.
    /// Returns the MLMultiArray grid and the centroid used for coordinate conversion.
    private nonisolated static func voxelizeProtein(
        heavyAtoms: [Atom], residues: [Residue]
    ) -> (MLMultiArray, SIMD3<Float>) {
        let gs = gridSize
        let res = resolution
        let half = Float(gs) * res / 2.0

        // Compute protein centroid (center of the grid)
        var sum = SIMD3<Float>.zero
        for a in heavyAtoms { sum += a.position }
        let center = sum / Float(heavyAtoms.count)

        // Build residue lookup for atoms
        var atomResidueMap: [Int: Int] = [:]
        for (ri, residue) in residues.enumerated() {
            for ai in residue.atomIndices {
                atomResidueMap[ai] = ri
            }
        }

        // Create grid [1, 10, 64, 64, 64]
        let grid: MLMultiArray
        do {
            grid = try MLMultiArray(
                shape: [1, NSNumber(value: nChannels), NSNumber(value: gs), NSNumber(value: gs), NSNumber(value: gs)],
                dataType: .float16
            )
        } catch {
            // Fallback — should never fail
            return (try! MLMultiArray(shape: [1, 10, 64, 64, 64], dataType: .float16), center)
        }

        // Zero-fill (MLMultiArray is not guaranteed to be zeroed)
        let totalElements = 1 * nChannels * gs * gs * gs
        let ptr = grid.dataPointer.bindMemory(to: Float16.self, capacity: totalElements)
        for i in 0..<totalElements { ptr[i] = 0 }

        // Strides for [1, C, X, Y, Z] layout
        let strideC = gs * gs * gs
        let strideX = gs * gs
        let strideY = gs

        for atom in heavyAtoms {
            let vxf = (atom.position.x - center.x + half) / res
            let vyf = (atom.position.y - center.y + half) / res
            let vzf = (atom.position.z - center.z + half) / res

            let vx = Int(vxf)
            let vy = Int(vyf)
            let vz = Int(vzf)

            guard vx >= 0, vx < gs, vy >= 0, vy < gs, vz >= 0, vz < gs else { continue }

            let spatialIdx = vx * strideX + vy * strideY + vz

            // Channel 0: all-atom density
            ptr[0 * strideC + spatialIdx] += 1.0

            // Channels 1-4: element type
            switch atom.element {
            case .C: ptr[1 * strideC + spatialIdx] += 1.0
            case .N: ptr[2 * strideC + spatialIdx] += 1.0
            case .O: ptr[3 * strideC + spatialIdx] += 1.0
            default: ptr[4 * strideC + spatialIdx] += 1.0  // S, P, etc.
            }

            // Channel 5: hydrophobicity (from residue)
            let hydro = hydrophobicity[atom.residueName] ?? 0.0
            ptr[5 * strideC + spatialIdx] += Float16(hydro)

            // Channel 6: partial charge
            let charge = partialCharges[atom.residueName]?[atom.name] ?? 0.0
            ptr[6 * strideC + spatialIdx] += Float16(charge)

            // Channel 7: aromatic
            if aromaticResidueAtoms.contains(atom.residueName) {
                let aroNames: Set<String> = [
                    "CG", "CD1", "CD2", "CE1", "CE2", "CZ",  // PHE/TYR
                    "NE1", "CE3", "CZ2", "CZ3", "CH2",       // TRP
                    "ND1", "NE2",                              // HIS
                ]
                if aroNames.contains(atom.name) {
                    ptr[7 * strideC + spatialIdx] += 1.0
                }
            }

            // Channel 8: H-bond donor
            let isDonor: Bool = {
                // Backbone N (except PRO)
                if atom.name == "N" && atom.residueName != "PRO" { return true }
                // Side-chain donors
                if let donors = hbondDonorAtoms[atom.residueName], donors.contains(atom.name) { return true }
                return false
            }()
            if isDonor { ptr[8 * strideC + spatialIdx] += 1.0 }

            // Channel 9: H-bond acceptor
            let isAcceptor: Bool = {
                // Backbone O
                if atom.name == "O" { return true }
                // Side-chain acceptors
                if let acceptors = hbondAcceptorAtoms[atom.residueName], acceptors.contains(atom.name) { return true }
                return false
            }()
            if isAcceptor { ptr[9 * strideC + spatialIdx] += 1.0 }
        }

        return (grid, center)
    }

    // MARK: - Pocket Extraction

    /// Extract above-threshold voxels from the probability grid and convert to world coordinates.
    private nonisolated static func extractPocketPoints(
        probGrid: MLMultiArray, center: SIMD3<Float>, threshold: Float
    ) -> [(position: SIMD3<Float>, probability: Float)] {
        let gs = gridSize
        let res = resolution
        let half = Float(gs) * res / 2.0

        let ptr = probGrid.dataPointer.bindMemory(to: Float16.self, capacity: gs * gs * gs)
        var points: [(position: SIMD3<Float>, probability: Float)] = []

        let strideX = gs * gs
        let strideY = gs

        for x in 0..<gs {
            for y in 0..<gs {
                for z in 0..<gs {
                    let prob = Float(ptr[x * strideX + y * strideY + z])
                    if prob > threshold {
                        let worldPos = SIMD3<Float>(
                            Float(x) * res - half + center.x,
                            Float(y) * res - half + center.y,
                            Float(z) * res - half + center.z
                        )
                        points.append((worldPos, prob))
                    }
                }
            }
        }

        return points
    }

    // MARK: - Clustering

    /// Cluster pocket points into BindingPocket objects using connected-component flood fill.
    private nonisolated static func clusterIntoPockets(
        _ points: [(position: SIMD3<Float>, probability: Float)],
        residues: [Residue],
        atoms: [Atom]
    ) -> [BindingPocket] {
        let eps: Float = 4.0
        let contactDistance: Float = 4.5
        let positions = points.map(\.position)
        var labels = [Int](repeating: -1, count: points.count)
        var clusterId = 0
        var atomResidueMap: [Int: Int] = [:]

        for (resIdx, residue) in residues.enumerated() {
            for atomIdx in residue.atomIndices {
                atomResidueMap[atomIdx] = resIdx
            }
        }

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

            let searchRadius = max(maxExt.x, max(maxExt.y, maxExt.z)) + contactDistance
            let contactDistSq = contactDistance * contactDistance
            let searchRadiusSq = searchRadius * searchRadius
            var pocketAtomIndices: [Int] = []
            pocketAtomIndices.reserveCapacity(64)

            for atomIndex in atoms.indices {
                let atom = atoms[atomIndex]
                if atom.element == .H || atom.isHetAtom { continue }
                if simd_distance_squared(atom.position, center) > searchRadiusSq { continue }

                for clusterPoint in clusterPos {
                    if simd_distance_squared(atom.position, clusterPoint) <= contactDistSq {
                        pocketAtomIndices.append(atomIndex)
                        break
                    }
                }
            }

            var residueSet = Set<Int>()
            var polarCount = 0
            var hydrophobicCount = 0
            for atomIndex in pocketAtomIndices {
                let atom = atoms[atomIndex]
                if let resIdx = atomResidueMap[atomIndex],
                   resIdx >= 0,
                   resIdx < residues.count,
                   residues[resIdx].isStandard {
                    residueSet.insert(resIdx)
                }

                if atom.element == .N || atom.element == .O {
                    polarCount += 1
                }
                if atom.element == .C || atom.element == .S {
                    hydrophobicCount += 1
                }
            }

            let totalPocketAtoms = max(Float(pocketAtomIndices.count), 1)
            let polarity = Float(polarCount) / totalPocketAtoms
            let hydrophobicity = Float(hydrophobicCount) / totalPocketAtoms
            let localPolarity: Float
            if pocketAtomIndices.isEmpty {
                localPolarity = 0.5
            } else {
                localPolarity = min(max((polarity + (1.0 - hydrophobicity)) * 0.5, 0.0), 1.0)
            }

            let pocket = BindingPocket(
                id: pockets.count,
                center: center,
                size: halfSize,
                volume: volume,
                buriedness: avgProb,
                polarity: localPolarity,
                druggability: 0,
                residueIndices: Array(residueSet).sorted(),
                probePositions: clusterPos
            )

            pockets.append(BindingPocket(
                id: pocket.id,
                center: pocket.center,
                size: pocket.size,
                volume: pocket.volume,
                buriedness: pocket.buriedness,
                polarity: pocket.polarity,
                druggability: PocketSelectionHeuristics.score(pocket, method: .ml),
                residueIndices: pocket.residueIndices,
                probePositions: pocket.probePositions
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
        guard nProt > 0, nLig > 0 else {
            Task { @MainActor in ActivityLog.shared.warn("[FeatureCompute] computeRBF: empty positions (prot=\(nProt), lig=\(nLig))", category: .system) }
            return nil
        }

        let nPairs = nProt * nLig
        var protPos = protPositions
        var ligPos = ligPositions
        var params = RBFParams(nProt: UInt32(nProt), nLig: UInt32(nLig), numBins: 50, gamma: 10.0, binSpacing: 10.0 / 49.0, _pad0: 0)

        guard let protBuffer = device.makeBuffer(bytes: &protPos, length: nProt * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let ligBuffer = device.makeBuffer(bytes: &ligPos, length: nLig * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared),
              let rbfBuffer = device.makeBuffer(length: nPairs * 50 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let distBuffer = device.makeBuffer(length: nPairs * MemoryLayout<Float>.stride, options: .storageModeShared),
              let paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<RBFParams>.stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            Task { @MainActor in ActivityLog.shared.warn("[FeatureCompute] computeRBF: Metal buffer allocation failed", category: .system) }
            return nil
        }

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

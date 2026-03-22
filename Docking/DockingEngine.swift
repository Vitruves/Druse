import Foundation
import MetalKit
import simd

// MARK: - Docking Configuration

struct DockingConfig: Sendable {
    // Population and search
    var populationSize: Int = 300
    var numRuns: Int = 1             // independent Monte Carlo trajectory batches
    var generationsPerRun: Int = 300 // Monte Carlo steps per run
    var gridSpacing: Float = 0.375

    // Search operators
    var mutationRate: Float = 0.08
    var crossoverRate: Float = 0.75
    var translationStep: Float = 2.0 // Angstroms, aligned with Vina mutation amplitude
    var rotationStep: Float = 0.3    // radians (~17°)
    var torsionStep: Float = 0.8     // radians (~46°) — large enough to escape tangled conformers
    var mcTemperature: Float = 1.2   // kcal/mol, matches Vina's default Metropolis temperature
    var explicitRerankTopClusters: Int = 12 // top basin representatives rescored against explicit receptor atoms
    var explicitRerankVariantsPerCluster: Int = 4 // seeded local refinement around each top basin representative
    var explicitRerankLocalSearchSteps: Int = 20 // short second-pass refinement on rerank seeds

    // Local search (Vina-like basin hopping: refine every MC step by default)
    var localSearchFrequency: Int = 1   // every N generations
    var localSearchSteps: Int = 30      // gradient descent steps per refinement
    var liveUpdateFrequency: Int = 3    // visual update every N generations (lower = smoother animation)

    // Flexibility
    var enableFlexibility: Bool = true  // torsion flexibility during docking
    var flexRefinementSteps: Int = 50   // extra torsion refinement steps after GA

    // Clash handling
    var maxClashOverlap: Float = 0.4    // Angstroms of VdW overlap allowed
    var clashPenaltyScale: Float = 5.0  // kcal/mol per Angstrom of excess overlap

    // Exploration: broader initial search with higher translation/rotation steps
    // before switching to fine-grained local refinement
    var explorationPhaseRatio: Float = 0.4  // first 40% of generations use broader search
    var explorationTranslationStep: Float = 4.0  // wider initial translation (vs 2.0 during refinement)
    var explorationRotationStep: Float = 0.6     // wider initial rotation (vs 0.3)
    var explorationMutationRate: Float = 0.15    // higher mutation during exploration

    // Legacy flat-generation count (for backward compatibility)
    var numGenerations: Int { numRuns * generationsPerRun }
}

// MARK: - Docking Result

struct DockingResult: Identifiable, Sendable {
    let id: Int
    var pose: DockPoseSwift
    var energy: Float           // total Vina score (kcal/mol)
    var stericEnergy: Float     // gauss1 + gauss2 + repulsion
    var hydrophobicEnergy: Float
    var hbondEnergy: Float
    var torsionPenalty: Float   // rotational entropy
    var generation: Int
    var clusterID: Int = -1
    var clusterRank: Int = 0
    var transformedAtomPositions: [SIMD3<Float>] = []
    var refinementEnergy: Float? = nil

    // Backward-compatible aliases
    var vdwEnergy: Float { stericEnergy }
    var elecEnergy: Float { hydrophobicEnergy }
    var desolvEnergy: Float { torsionPenalty }
}

struct DockPoseSwift: Sendable {
    var translation: SIMD3<Float>
    var rotation: simd_quatf
    var torsions: [Float]
}

struct PreparedDockingLigand {
    var heavyAtoms: [Atom]
    var heavyBonds: [Bond]
    var centroid: SIMD3<Float>
    var gpuAtoms: [DockLigandAtom]
}

struct DockingGridSnapshot {
    let stericGridBuffer: MTLBuffer
    let hydrophobicGridBuffer: MTLBuffer
    let hbondGridBuffer: MTLBuffer
    let gridParamsBuffer: MTLBuffer
    let gridParams: GridParams
}

// MARK: - Interaction Detection

struct MolecularInteraction: Identifiable, Sendable {
    let id: Int
    var ligandAtomIndex: Int
    var proteinAtomIndex: Int
    var type: InteractionType
    var distance: Float
    var ligandPosition: SIMD3<Float>
    var proteinPosition: SIMD3<Float>

    enum InteractionType: Int, CaseIterable, Sendable {
        case hbond = 0        // H-bond: N/O donor ↔ acceptor, 2.2-3.5 Å
        case hydrophobic = 1  // Hydrophobic contact: C/S ↔ C/S, 3.3-4.5 Å (filtered)
        case saltBridge = 2   // Salt bridge: charged group pairs, < 4.0 Å
        case piStack = 3      // π-π stacking: aromatic ring centroids, 3.3-5.5 Å
        case piCation = 4     // π-cation: aromatic ring ↔ cation, < 6.0 Å
        case halogen = 5      // Halogen bond: F/Cl/Br ↔ N/O, 2.5-3.5 Å
        case metalCoord = 6   // Metal coordination: Zn/Fe/Mg ↔ N/O/S, < 2.8 Å
        case chPi = 7         // CH-π: C-H ↔ aromatic ring, 3.5-4.5 Å

        var color: SIMD4<Float> {
            switch self {
            case .hbond:       SIMD4(0.2, 0.8, 1.0, 1.0)    // cyan
            case .hydrophobic: SIMD4(0.5, 0.5, 0.5, 0.6)    // gray, subtle
            case .saltBridge:  SIMD4(1.0, 0.5, 0.1, 1.0)    // orange
            case .piStack:     SIMD4(0.7, 0.3, 1.0, 1.0)    // purple
            case .piCation:    SIMD4(1.0, 0.3, 0.7, 1.0)    // magenta
            case .halogen:     SIMD4(0.2, 1.0, 0.5, 1.0)    // green
            case .metalCoord:  SIMD4(1.0, 0.85, 0.0, 1.0)   // gold
            case .chPi:        SIMD4(0.6, 0.5, 0.8, 0.7)    // light purple
            }
        }

        var label: String {
            switch self {
            case .hbond:       "H-bond"
            case .hydrophobic: "Hydrophobic"
            case .saltBridge:  "Salt bridge"
            case .piStack:     "π-π stack"
            case .piCation:    "π-cation"
            case .halogen:     "Halogen bond"
            case .metalCoord:  "Metal coord."
            case .chPi:        "CH-π"
            }
        }
    }
}

// MARK: - Docking Engine

@MainActor
final class DockingEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private var stericGridPipeline: MTLComputePipelineState!
    private var hydrophobicGridPipeline: MTLComputePipelineState!
    private var hbondGridPipeline: MTLComputePipelineState!
    private var vinaAffinityGridPipeline: MTLComputePipelineState!
    private var scorePipeline: MTLComputePipelineState!
    private var initPopPipeline: MTLComputePipelineState!
    private var evolvePipeline: MTLComputePipelineState!
    private var localSearchPipeline: MTLComputePipelineState!
    private var mcPerturbPipeline: MTLComputePipelineState!
    private var metropolisAcceptPipeline: MTLComputePipelineState!
    private var explicitScorePipeline: MTLComputePipelineState!

    private var stericGridBuffer: MTLBuffer?
    private var hydrophobicGridBuffer: MTLBuffer?
    private var hbondGridBuffer: MTLBuffer?
    private var vinaAffinityGridBuffer: MTLBuffer?
    private var vinaTypeIndexBuffer: MTLBuffer?
    private var vinaAffinityTypeBuffer: MTLBuffer?
    private var proteinAtomBuffer: MTLBuffer?
    private var gridParamsBuffer: MTLBuffer?
    private var populationBuffer: MTLBuffer?
    private var offspringBuffer: MTLBuffer?
    private var bestPopulationBuffer: MTLBuffer?
    private var ligandAtomBuffer: MTLBuffer?
    private var gaParamsBuffer: MTLBuffer?
    private var torsionEdgeBuffer: MTLBuffer?
    private var movingIndicesBuffer: MTLBuffer?
    private var exclusionMaskBuffer: MTLBuffer?

    private(set) var isRunning = false
    private(set) var currentGeneration = 0
    private(set) var bestEnergy: Float = .infinity
    private var gridParams = GridParams()
    private var config = DockingConfig()
    /// Tracks the last allocated population buffer capacity to avoid redundant reallocation.
    private var lastPopulationBufferCapacity: Int = 0

    /// Diagnostics from the last completed docking run.
    private(set) var lastDiagnostics: DockingDiagnostics?

    var onPoseUpdate: ((DockingResult, [MolecularInteraction]) -> Void)?
    var onGenerationComplete: ((Int, Float) -> Void)?
    var onDockingComplete: (([DockingResult]) -> Void)?

    // Reference to protein atoms for interaction detection
    var proteinAtoms: [Atom] = []
    private var proteinStructure: Molecule?

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else { return nil }
        self.commandQueue = queue

        do {
            stericGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeStericGrid")!)
            hydrophobicGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHydrophobicGrid")!)
            hbondGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeHBondGrid")!)
            vinaAffinityGridPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "computeVinaAffinityMaps")!)
            scorePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "scorePoses")!)
            initPopPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "initializePopulation")!)
            evolvePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "gaEvolve")!)
            localSearchPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "localSearch")!)
            mcPerturbPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "mcPerturb")!)
            metropolisAcceptPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "metropolisAccept")!)
            explicitScorePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "scorePosesExplicit")!)
        } catch {
            print("Failed to create docking pipelines: \(error)")
            return nil
        }
    }

    // MARK: - Grid Map Computation

    // MARK: - Vina Atom Typing

    private func hasAttachedHydrogen(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains { molecule.atoms[$0].element == .H }
    }

    private func isBondedToHeteroatom(atomIndex: Int, in molecule: Molecule) -> Bool {
        molecule.neighbors(of: atomIndex).contains {
            let element = molecule.atoms[$0].element
            return element != .H && element != .C
        }
    }

    private func vinaTypeID(_ type: VinaAtomType) -> Int32 {
        Int32(type.rawValue)
    }

    private var maxSupportedVinaType: Int32 {
        vinaTypeID(VINA_MET_D)
    }

    /// Approximate protein XS typing in the same space Vina uses upstream.
    /// Donor/acceptor assignment leans on residue chemistry while carbon polarity
    /// uses the actual bond graph when available.
    private func vinaProteinAtomType(for atomIndex: Int, in molecule: Molecule) -> Int32 {
        let atom = molecule.atoms[atomIndex]
        let name = atom.name.trimmingCharacters(in: .whitespaces)
        let res = atom.residueName
        let donor = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
            name == "N" || name == "NZ" || name == "NE" || name == "NH1" || name == "NH2" ||
            name == "ND2" || name == "NE2"

        switch atom.element {
        case .C:
            return isBondedToHeteroatom(atomIndex: atomIndex, in: molecule) ? vinaTypeID(VINA_C_P) : vinaTypeID(VINA_C_H)

        case .N:
            let acceptor = res == "HIS" && (name == "ND1" || name == "NE2")
            if donor && acceptor { return vinaTypeID(VINA_N_DA) }
            if acceptor { return vinaTypeID(VINA_N_A) }
            if donor { return vinaTypeID(VINA_N_D) }
            return vinaTypeID(VINA_N_P)

        case .O:
            let donorO = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
                name == "OG" || name == "OG1" || name == "OH"
            let acceptor = atom.formalCharge <= 0 || name == "O" || name.hasPrefix("OD") || name.hasPrefix("OE")
            if donorO && acceptor { return vinaTypeID(VINA_O_DA) }
            if acceptor { return vinaTypeID(VINA_O_A) }
            if donorO { return vinaTypeID(VINA_O_D) }
            return vinaTypeID(VINA_O_P)

        case .S:  return vinaTypeID(VINA_S_P)
        case .P:  return vinaTypeID(VINA_P_P)
        case .F:  return vinaTypeID(VINA_F_H)
        case .Cl: return vinaTypeID(VINA_Cl_H)
        case .Br: return vinaTypeID(VINA_Br_H)
        case .Na, .Mg, .Ca, .Sc, .Ti, .V, .Cr, .Mn, .Fe, .Co, .Ni, .Cu, .Zn:
            return vinaTypeID(VINA_MET_D)
        default:  return vinaTypeID(VINA_OTHER)
        }
    }

    private func fallbackLigandVinaAtomType(for atomIndex: Int, in molecule: Molecule) -> Int32 {
        let atom = molecule.atoms[atomIndex]
        let donor = hasAttachedHydrogen(atomIndex: atomIndex, in: molecule) ||
            (atom.element == .N && atom.formalCharge > 0)
        let acceptor: Bool

        switch atom.element {
        case .N:
            let heavyNeighborCount = molecule.neighbors(of: atomIndex).filter { molecule.atoms[$0].element != .H }.count
            acceptor = atom.formalCharge <= 0 && !donor && heavyNeighborCount < 4
        case .O:
            acceptor = atom.formalCharge <= 0
        default:
            acceptor = false
        }

        switch atom.element {
        case .C:
            return isBondedToHeteroatom(atomIndex: atomIndex, in: molecule) ? vinaTypeID(VINA_C_P) : vinaTypeID(VINA_C_H)
        case .N:
            if donor && acceptor { return vinaTypeID(VINA_N_DA) }
            if acceptor { return vinaTypeID(VINA_N_A) }
            if donor { return vinaTypeID(VINA_N_D) }
            return vinaTypeID(VINA_N_P)
        case .O:
            if donor && acceptor { return vinaTypeID(VINA_O_DA) }
            if acceptor { return vinaTypeID(VINA_O_A) }
            if donor { return vinaTypeID(VINA_O_D) }
            return vinaTypeID(VINA_O_P)
        case .S:  return vinaTypeID(VINA_S_P)
        case .P:  return vinaTypeID(VINA_P_P)
        case .F:  return vinaTypeID(VINA_F_H)
        case .Cl: return vinaTypeID(VINA_Cl_H)
        case .Br: return vinaTypeID(VINA_Br_H)
        case .Na, .Mg, .Ca, .Sc, .Ti, .V, .Cr, .Mn, .Fe, .Co, .Ni, .Cu, .Zn:
            return vinaTypeID(VINA_MET_D)
        default:  return vinaTypeID(VINA_OTHER)
        }
    }

    private func ligandVinaTypes(_ ligand: Molecule) -> [Int32] {
        guard !ligand.atoms.isEmpty else { return [] }

        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        if let rdkitTypes = RDKitBridge.computeVinaTypesMolBlock(molBlock, atomCount: ligand.atoms.count),
           rdkitTypes.count == ligand.atoms.count {
            return rdkitTypes
        }

        return ligand.atoms.indices.map { fallbackLigandVinaAtomType(for: $0, in: ligand) }
    }

    private func swiftXSIsHydrophobic(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_C_H) || xsType == vinaTypeID(VINA_F_H) ||
        xsType == vinaTypeID(VINA_Cl_H) || xsType == vinaTypeID(VINA_Br_H) ||
        xsType == vinaTypeID(VINA_I_H)
    }

    private func swiftXSIsAcceptor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_A) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_A) || xsType == vinaTypeID(VINA_O_DA)
    }

    private func swiftXSIsDonor(_ xsType: Int32) -> Bool {
        xsType == vinaTypeID(VINA_N_D) || xsType == vinaTypeID(VINA_N_DA) ||
        xsType == vinaTypeID(VINA_O_D) || xsType == vinaTypeID(VINA_O_DA) ||
        xsType == vinaTypeID(VINA_MET_D)
    }

    private func swiftXSRadius(_ xsType: Int32) -> Float {
        let radii: [Float] = [
            1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7,
            2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2
        ]
        let index = Int(xsType)
        guard index >= 0, index < radii.count else { return 0 }
        return radii[index]
    }

    private func swiftSlopeStep(xBad: Float, xGood: Float, x: Float) -> Float {
        if xBad < xGood {
            if x <= xBad { return 0 }
            if x >= xGood { return 1 }
        } else {
            if x >= xBad { return 0 }
            if x <= xGood { return 1 }
        }
        return (x - xBad) / (xGood - xBad)
    }

    private func swiftVinaPairEnergy(_ type1: Int32, _ type2: Int32, distance r: Float) -> Float {
        guard r < 8.0,
              type1 >= 0, type1 <= maxSupportedVinaType,
              type2 >= 0, type2 <= maxSupportedVinaType else {
            return 0
        }

        let d = r - (swiftXSRadius(type1) + swiftXSRadius(type2))
        let gauss1 = -0.035579 * exp(-pow(d * 2.0, 2.0))
        let gauss2 = -0.005156 * exp(-pow((d - 3.0) * 0.5, 2.0))
        let repulsion = d < 0 ? 0.840245 * d * d : 0
        let hydrophobic = (swiftXSIsHydrophobic(type1) && swiftXSIsHydrophobic(type2))
            ? -0.035069 * swiftSlopeStep(xBad: 1.5, xGood: 0.5, x: d)
            : 0
        let hbond = ((swiftXSIsDonor(type1) && swiftXSIsAcceptor(type2)) ||
                     (swiftXSIsDonor(type2) && swiftXSIsAcceptor(type1)))
            ? -0.587439 * swiftSlopeStep(xBad: 0.0, xGood: -0.7, x: d)
            : 0
        return gauss1 + gauss2 + repulsion + hydrophobic + hbond
    }

    private func intramolecularReferenceEnergy(
        ligandAtoms: [DockLigandAtom],
        exclusionMask: [UInt32],
        maxAtoms: Int
    ) -> Float {
        guard ligandAtoms.count > 1 else { return 0 }

        var total: Float = 0
        for i in 0..<ligandAtoms.count {
            for j in (i + 1)..<ligandAtoms.count {
                let pairIndex = i * maxAtoms + j
                let word = pairIndex / 32
                let bit = pairIndex % 32
                if exclusionMask[word] & (1 << bit) != 0 { continue }

                let r = simd_distance(ligandAtoms[i].position, ligandAtoms[j].position)
                total += swiftVinaPairEnergy(ligandAtoms[i].vinaType, ligandAtoms[j].vinaType, distance: r)
            }
        }
        return total
    }

    private struct RerankRNG {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed &+ 0x9E3779B97F4A7C15
        }

        mutating func nextUInt32() -> UInt32 {
            state = state &* 2862933555777941757 &+ 3037000493
            return UInt32(truncatingIfNeeded: state >> 16)
        }

        mutating func nextFloat() -> Float {
            Float(nextUInt32()) / Float(UInt32.max)
        }

        mutating func signed(amplitude: Float) -> Float {
            (nextFloat() * 2 - 1) * amplitude
        }

        mutating func vectorInUnitSphere(scale: Float) -> SIMD3<Float> {
            for _ in 0..<16 {
                let v = SIMD3<Float>(
                    signed(amplitude: 1),
                    signed(amplitude: 1),
                    signed(amplitude: 1)
                )
                let len2 = simd_length_squared(v)
                if len2 > 1e-4, len2 <= 1 {
                    return v * scale
                }
            }
            return SIMD3<Float>(scale, 0, 0)
        }
    }

    private func wrappedAngle(_ angle: Float) -> Float {
        var wrapped = angle
        while wrapped > .pi { wrapped -= 2 * .pi }
        while wrapped < -.pi { wrapped += 2 * .pi }
        return wrapped
    }

    private func torsions(from pose: DockPose) -> [Float] {
        let count = max(0, min(Int(pose.numTorsions), 32))
        guard count > 0 else { return [] }
        return withUnsafePointer(to: pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                Array(UnsafeBufferPointer(start: buffer, count: count))
            }
        }
    }

    private func makeDockPose(from result: DockingResult) -> DockPose {
        var pose = DockPose()
        pose.translation = result.pose.translation
        pose.energy = result.energy
        pose.rotation = SIMD4<Float>(
            result.pose.rotation.imag.x,
            result.pose.rotation.imag.y,
            result.pose.rotation.imag.z,
            result.pose.rotation.real
        )
        let torsions = result.pose.torsions
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(min(torsions.count, 32))
        pose.generation = Int32(result.generation)
        pose.stericEnergy = result.stericEnergy
        pose.hydrophobicEnergy = result.hydrophobicEnergy
        pose.hbondEnergy = result.hbondEnergy
        pose.torsionPenalty = result.torsionPenalty
        pose.clashPenalty = 0
        pose._pad0 = 0
        return pose
    }

    private func makeRerankSeedPose(from result: DockingResult, variantIndex: Int) -> DockPose {
        var pose = makeDockPose(from: result)
        guard variantIndex > 0 else { return pose }

        let seed = UInt64(bitPattern: Int64(result.id &* 1_315_423_911
            ^ result.clusterID &* 374_761_393
            ^ variantIndex &* 668_265_263))
        var rng = RerankRNG(seed: seed)

        pose.translation += rng.vectorInUnitSphere(scale: 0.75)

        let axis = simd_normalize(rng.vectorInUnitSphere(scale: 1))
        let deltaRotation = simd_quatf(angle: rng.signed(amplitude: 0.18), axis: axis)
        let currentRotation = simd_quatf(ix: pose.rotation.x, iy: pose.rotation.y, iz: pose.rotation.z, r: pose.rotation.w)
        let updatedRotation = deltaRotation * currentRotation
        pose.rotation = SIMD4<Float>(
            updatedRotation.imag.x,
            updatedRotation.imag.y,
            updatedRotation.imag.z,
            updatedRotation.real
        )

        let torsionCount = max(0, min(Int(pose.numTorsions), 32))
        guard torsionCount > 0 else { return pose }
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for index in 0..<torsionCount {
                    buffer[index] = wrappedAngle(buffer[index] + rng.signed(amplitude: 0.30))
                }
            }
        }
        return pose
    }

    private func scorePopulationExplicit(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        let tgSize = MTLSize(width: min(max(populationSize, 1), 64), height: 1, depth: 1)
        let tgCount = MTLSize(width: (max(populationSize, 1) + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        dispatchCompute(pipeline: explicitScorePipeline, buffers: [
            (buffer, 0), (ligandAtomBuffer!, 1), (proteinAtomBuffer!, 2),
            (gridParamsBuffer, 3), (gaParamsBuffer, 4),
            (torsionEdgeBuffer!, 5), (movingIndicesBuffer!, 6),
            (exclusionMaskBuffer!, 7)
        ], threadGroups: tgCount, threadGroupSize: tgSize)
    }

    private func localOptimizeGrid(
        buffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        gaParamsBuffer: MTLBuffer,
        populationSize: Int
    ) {
        let tgSize = MTLSize(width: min(max(populationSize, 1), 64), height: 1, depth: 1)
        let tgCount = MTLSize(width: (max(populationSize, 1) + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        dispatchCompute(pipeline: localSearchPipeline, buffers: [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer, 4), (gaParamsBuffer, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ], threadGroups: tgCount, threadGroupSize: tgSize)
    }

    private func rerankClusterRepresentativesExplicit(
        _ results: [DockingResult],
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>
    ) -> [DockingResult] {
        guard config.explicitRerankTopClusters > 0,
              proteinAtomBuffer != nil,
              let gridParamsBuffer,
              let gaParamsBuffer,
              !results.isEmpty else {
            return results
        }

        let grouped = Dictionary(grouping: results, by: \.clusterID)
        let leaders = results
            .filter { $0.clusterRank == 0 }
            .sorted { $0.energy < $1.energy }
        guard !leaders.isEmpty else { return results }

        let rerankCount = min(config.explicitRerankTopClusters, leaders.count)
        let variantsPerCluster = max(config.explicitRerankVariantsPerCluster, 1)
        let rerankLeaders = Array(leaders.prefix(rerankCount))
        var representativePoses: [DockPose] = []
        var variantClusterIDs: [Int] = []
        representativePoses.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        variantClusterIDs.reserveCapacity(rerankLeaders.count * variantsPerCluster)
        for leader in rerankLeaders {
            for variantIndex in 0..<variantsPerCluster {
                representativePoses.append(makeRerankSeedPose(from: leader, variantIndex: variantIndex))
                variantClusterIDs.append(leader.clusterID)
            }
        }

        let repBuffer = device.makeBuffer(
            bytes: &representativePoses,
            length: representativePoses.count * MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )
        guard let repBuffer else { return results }

        let currentGA = gaParamsBuffer.contents().bindMemory(to: GAParams.self, capacity: 1).pointee
        var rerankGA = currentGA
        rerankGA.populationSize = UInt32(representativePoses.count)
        rerankGA.localSearchSteps = UInt32(max(config.explicitRerankLocalSearchSteps, 1))
        let rerankGABuffer = device.makeBuffer(
            bytes: &rerankGA,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )
        guard let rerankGABuffer else { return results }

        localOptimizeGrid(
            buffer: repBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gaParamsBuffer: rerankGABuffer,
            populationSize: representativePoses.count
        )
        scorePopulationExplicit(
            buffer: repBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gaParamsBuffer: rerankGABuffer,
            populationSize: representativePoses.count
        )

        let rescoredLeaders = extractAllResults(
            from: repBuffer,
            ligandAtoms: ligandAtoms,
            centroid: centroid,
            idOffset: 0,
            sortByEnergy: false
        )
        guard rescoredLeaders.count == representativePoses.count else { return results }

        var representativeByCluster: [Int: DockingResult] = [:]
        for (index, rescored) in rescoredLeaders.enumerated() {
            let sourceClusterID = variantClusterIDs[index]
            var updated = rescored
            updated.clusterID = sourceClusterID
            updated.clusterRank = 0
            if updated.energy < (representativeByCluster[sourceClusterID]?.energy ?? .infinity) {
                representativeByCluster[sourceClusterID] = updated
            }
        }

        let sortedClusterIDs = leaders
            .map(\.clusterID)
            .sorted {
                let lhs = representativeByCluster[$0]?.energy ?? grouped[$0]?.first?.energy ?? .infinity
                let rhs = representativeByCluster[$1]?.energy ?? grouped[$1]?.first?.energy ?? .infinity
                return lhs < rhs
            }

        var reranked: [DockingResult] = []
        reranked.reserveCapacity(results.count)

        for (newClusterID, oldClusterID) in sortedClusterIDs.enumerated() {
            guard let members = grouped[oldClusterID] else { continue }
            let originalLeader = members.first { $0.clusterRank == 0 }
            var leader = representativeByCluster[oldClusterID] ?? originalLeader ?? members[0]
            leader.clusterID = newClusterID
            leader.clusterRank = 0
            reranked.append(leader)

            var rank = 1
            for member in members.sorted(by: { $0.energy < $1.energy }) where member.clusterRank != 0 {
                var updated = member
                updated.clusterID = newClusterID
                updated.clusterRank = rank
                rank += 1
                reranked.append(updated)
            }
        }

        return reranked.sorted {
            if $0.energy != $1.energy { return $0.energy < $1.energy }
            if $0.clusterID != $1.clusterID { return $0.clusterID < $1.clusterID }
            if $0.clusterRank != $1.clusterRank { return $0.clusterRank < $1.clusterRank }
            return $0.id < $1.id
        }
    }

    func computeGridMaps(protein: Molecule, pocket: BindingPocket, spacing: Float = 0.375,
                          ligandExtent: SIMD3<Float>? = nil,
                          requiredVinaTypes: [Int32] = []) {
        proteinStructure = protein
        let heavyAtoms = protein.atoms.filter { $0.element != .H }
        self.proteinAtoms = heavyAtoms

        let gpuAtoms = protein.atoms.enumerated().compactMap { atomIndex, atom -> GridProteinAtom? in
            guard atom.element != .H else { return nil }
            return GridProteinAtom(
                position: atom.position,
                vdwRadius: atom.element.vdwRadius,
                charge: electrostaticCharge(for: atom),
                vinaType: vinaProteinAtomType(for: atomIndex, in: protein),
                _pad0: 0, _pad1: 0
            )
        }

        let activeVinaTypes = Array(Set(requiredVinaTypes.filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()

        let gridMapCount: UInt64 = activeVinaTypes.isEmpty ? 3 : UInt64(3 + activeVinaTypes.count)

        // Keep the search box centered on the selected pocket. Covering the whole
        // protein here turns pocket docking into accidental global surface docking,
        // which allows the optimizer to converge on unrelated basins.
        // Pocket detectors already return padded half-extents, so avoid inflating
        // the translation domain a second time here.
        let searchPadding: Float = 0.0
        let gridPadding: Float = 3.0
        let ligandMargin = ligandExtent ?? SIMD3<Float>(repeating: 4.0)
        let searchCenter = pocket.center
        let searchHalfExtent = pocket.size + SIMD3<Float>(repeating: searchPadding)
        let gridHalfExtent = searchHalfExtent + ligandMargin + SIMD3<Float>(repeating: gridPadding)
        let boxMin = searchCenter - gridHalfExtent
        let boxMax = searchCenter + gridHalfExtent
        let boxSize = boxMax - boxMin

        // Memory guard: if the typed maps would exceed the GPU budget, coarsen spacing.
        var effectiveSpacing = spacing
        let maxGridFloatValues: UInt64 = 24_000_000

        // Safely compute grid dimensions — clamp to avoid overflow from inf/NaN
        func gridDim(_ length: Float, _ sp: Float) -> UInt64 {
            let raw = ceil(length / sp)
            guard raw.isFinite && raw > 0 else { return 1 }
            return UInt64(min(raw, 10000))
        }
        let ex = gridDim(boxSize.x, effectiveSpacing)
        let ey = gridDim(boxSize.y, effectiveSpacing)
        let ez = gridDim(boxSize.z, effectiveSpacing)
        let estimatedPoints = ex * ey * ez
        if estimatedPoints * gridMapCount > maxGridFloatValues {
            let scaleFactor = pow(Float(estimatedPoints * gridMapCount) / Float(maxGridFloatValues), 1.0 / 3.0)
            effectiveSpacing = spacing * max(scaleFactor, 1.001)
        }
        let finalBoxSize = boxMax - boxMin
        var nx = UInt32(gridDim(finalBoxSize.x, effectiveSpacing)) + 1
        var ny = UInt32(gridDim(finalBoxSize.y, effectiveSpacing)) + 1
        var nz = UInt32(gridDim(finalBoxSize.z, effectiveSpacing)) + 1
        while UInt64(nx) * UInt64(ny) * UInt64(nz) * gridMapCount > maxGridFloatValues {
            effectiveSpacing *= 1.2
            nx = UInt32(gridDim(finalBoxSize.x, effectiveSpacing)) + 1
            ny = UInt32(gridDim(finalBoxSize.y, effectiveSpacing)) + 1
            nz = UInt32(gridDim(finalBoxSize.z, effectiveSpacing)) + 1
        }
        let totalPoints = UInt32(UInt64(nx) * UInt64(ny) * UInt64(nz))

        gridParams = GridParams(
            origin: boxMin, spacing: effectiveSpacing,
            dims: SIMD3(nx, ny, nz), _pad0: 0,
            totalPoints: totalPoints,
            numProteinAtoms: UInt32(gpuAtoms.count),
            numAffinityTypes: UInt32(activeVinaTypes.count), _pad2: 0,
            searchCenter: searchCenter, _pad3: 0,
            searchHalfExtent: searchHalfExtent, _pad4: 0
        )

        var proteinGPUAtoms = gpuAtoms
        proteinAtomBuffer = device.makeBuffer(bytes: &proteinGPUAtoms, length: proteinGPUAtoms.count * MemoryLayout<GridProteinAtom>.stride, options: .storageModeShared)
        gridParamsBuffer = device.makeBuffer(bytes: &gridParams, length: MemoryLayout<GridParams>.stride, options: .storageModeShared)

        let gridByteSize = Int(totalPoints) * MemoryLayout<Float>.stride
        stericGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hydrophobicGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)
        hbondGridBuffer = device.makeBuffer(length: gridByteSize, options: .storageModeShared)

        if activeVinaTypes.isEmpty {
            vinaAffinityGridBuffer = nil
            vinaTypeIndexBuffer = nil
            vinaAffinityTypeBuffer = nil
        } else {
            vinaAffinityGridBuffer = device.makeBuffer(
                length: gridByteSize * activeVinaTypes.count,
                options: .storageModeShared
            )

            var typeLookup = [Int32](repeating: -1, count: 32)
            for (slot, type) in activeVinaTypes.enumerated() where Int(type) < typeLookup.count {
                typeLookup[Int(type)] = Int32(slot)
            }
            vinaTypeIndexBuffer = device.makeBuffer(
                bytes: &typeLookup,
                length: typeLookup.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
            var affinityTypes = activeVinaTypes
            vinaAffinityTypeBuffer = device.makeBuffer(
                bytes: &affinityTypes,
                length: affinityTypes.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        }

        let gridThreads = 128
        let tgSize = MTLSize(width: gridThreads, height: 1, depth: 1)
        let tgCount = MTLSize(width: (Int(totalPoints) + gridThreads - 1) / gridThreads, height: 1, depth: 1)

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        guard let stericBuf = stericGridBuffer,
              let hydroBuf = hydrophobicGridBuffer,
              let hbondBuf = hbondGridBuffer,
              let stericPipe = stericGridPipeline,
              let hydroPipe = hydrophobicGridPipeline,
              let hbondPipe = hbondGridPipeline else { return }

        for (pipeline, gridBuf) in [(stericPipe, stericBuf),
                                     (hydroPipe, hydroBuf),
                                     (hbondPipe, hbondBuf)] {
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gridBuf, offset: 0, index: 0)
            enc.setBuffer(proteinAtomBuffer, offset: 0, index: 1)
            enc.setBuffer(gridParamsBuffer, offset: 0, index: 2)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        }

        if let affinityBuf = vinaAffinityGridBuffer,
           let affinityTypes = vinaAffinityTypeBuffer {
            let affinityEntryCount = Int(totalPoints) * activeVinaTypes.count
            let affinityTGCount = MTLSize(
                width: (affinityEntryCount + gridThreads - 1) / gridThreads,
                height: 1,
                depth: 1
            )
            enc.setComputePipelineState(vinaAffinityGridPipeline)
            enc.setBuffer(affinityBuf, offset: 0, index: 0)
            enc.setBuffer(proteinAtomBuffer, offset: 0, index: 1)
            enc.setBuffer(gridParamsBuffer, offset: 0, index: 2)
            enc.setBuffer(affinityTypes, offset: 0, index: 3)
            enc.dispatchThreadgroups(affinityTGCount, threadsPerThreadgroup: tgSize)
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    func gridSnapshot() -> DockingGridSnapshot? {
        guard let stericGridBuffer,
              let hydrophobicGridBuffer,
              let hbondGridBuffer,
              let gridParamsBuffer else {
            return nil
        }
        return DockingGridSnapshot(
            stericGridBuffer: stericGridBuffer,
            hydrophobicGridBuffer: hydrophobicGridBuffer,
            hbondGridBuffer: hbondGridBuffer,
            gridParamsBuffer: gridParamsBuffer,
            gridParams: gridParams
        )
    }

    func prepareLigandGeometry(_ ligand: Molecule) -> PreparedDockingLigand {
        let chargedLigand = ligandWithDockingCharges(ligand)
        let allVinaTypes = ligandVinaTypes(chargedLigand)
        let heavyEntries = chargedLigand.atoms.enumerated().filter { $0.element.element != .H }
        let heavyAtoms = heavyEntries.map(\.element)
        let centroid = heavyAtoms.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(max(heavyAtoms.count, 1))

        var oldToNew: [Int: Int] = [:]
        for (newIdx, entry) in heavyEntries.enumerated() {
            oldToNew[entry.offset] = newIdx
        }

        let heavyBonds: [Bond] = chargedLigand.bonds.compactMap { bond in
            guard let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] else { return nil }
            return Bond(id: bond.id, atomIndex1: a, atomIndex2: b, order: bond.order)
        }

        let gpuAtoms: [DockLigandAtom] = heavyEntries.map { entry in
            let atom = entry.element
            return DockLigandAtom(
                position: atom.position - centroid,
                vdwRadius: atom.element.vdwRadius,
                charge: electrostaticCharge(for: atom),
                vinaType: allVinaTypes.indices.contains(entry.offset)
                    ? allVinaTypes[entry.offset]
                    : fallbackLigandVinaAtomType(for: entry.offset, in: chargedLigand),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        return PreparedDockingLigand(
            heavyAtoms: heavyAtoms,
            heavyBonds: heavyBonds,
            centroid: centroid,
            gpuAtoms: gpuAtoms
        )
    }

    /// Debug: read back grid map statistics
    func gridDiagnostics() -> String {
        var lines: [String] = []
        for (name, buf) in [("Steric", stericGridBuffer), ("Hydrophobic", hydrophobicGridBuffer),
                             ("HBond", hbondGridBuffer)] {
            guard let buf else { lines.append("  \(name): nil"); continue }
            let count = buf.length / MemoryLayout<Float>.stride
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            var minV: Float = .infinity, maxV: Float = -.infinity, nonZero = 0
            var sum: Float = 0
            for i in 0..<count {
                let v = ptr[i]
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                if abs(v) > 1e-6 { nonZero += 1 }
                sum += v
            }
            lines.append("  \(name): \(count) pts, min=\(String(format: "%.3f", minV)) max=\(String(format: "%.3f", maxV)) nonzero=\(nonZero) mean=\(String(format: "%.4f", sum/Float(count)))")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Run Docking

    func runDocking(
        ligand: Molecule, pocket: BindingPocket, config: DockingConfig = DockingConfig()
    ) async -> [DockingResult] {
        self.config = config
        isRunning = true
        currentGeneration = 0
        bestEnergy = .infinity

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        // Compute ligand bounding half-extent (centroid-subtracted coordinates)
        var ligMin = SIMD3<Float>(repeating: .infinity)
        var ligMax = SIMD3<Float>(repeating: -.infinity)
        var ligandRadiusSquared: Float = 0
        for a in gpuLigAtoms {
            ligMin = simd_min(ligMin, a.position)
            ligMax = simd_max(ligMax, a.position)
            ligandRadiusSquared = max(ligandRadiusSquared, simd_length_squared(a.position))
        }
        let ligandHalfExtent = (ligMax - ligMin) * 0.5
        let ligandRadius = max(sqrt(ligandRadiusSquared / Float(max(gpuLigAtoms.count, 1))), 1.0)

        let requiredVinaTypes = Array(Set(gpuLigAtoms.map(\.vinaType).filter { $0 >= 0 && $0 <= maxSupportedVinaType })).sorted()
        if let protein = proteinStructure ?? (!proteinAtoms.isEmpty ? Molecule(name: "cached", atoms: proteinAtoms, bonds: [], title: "") : nil) {
            computeGridMaps(
                protein: protein,
                pocket: pocket,
                spacing: config.gridSpacing,
                ligandExtent: ligandHalfExtent,
                requiredVinaTypes: requiredVinaTypes
            )
        }

        ligandAtomBuffer = device.makeBuffer(bytes: &gpuLigAtoms, length: gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride, options: .storageModeShared)

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map { Int32($0) })
        }
        let numTorsions = min(torsionEdges.count, 32)

        // Create torsion Metal buffers (even if empty, need valid pointers)
        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }
        torsionEdgeBuffer = device.makeBuffer(bytes: &torsionEdges, length: torsionEdges.count * MemoryLayout<TorsionEdge>.stride, options: .storageModeShared)
        movingIndicesBuffer = device.makeBuffer(bytes: &movingIndices, length: movingIndices.count * MemoryLayout<Int32>.stride, options: .storageModeShared)

        // Build exclusion mask for intramolecular clash detection.
        // Marks 1-2 (bonded) and 1-3 (angle) pairs to skip during clash evaluation.
        let maxAtoms = 128
        let maskWords = (maxAtoms * maxAtoms + 31) / 32
        var mask = [UInt32](repeating: 0, count: maskWords)

        // Build adjacency from heavy bonds
        var adj: [[Int]] = Array(repeating: [], count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        // Mark 1-2 and 1-3 pairs in the bitmask
        for i in 0..<gpuLigAtoms.count {
            for j in adj[i] where j > i {
                // 1-2 pair (bonded)
                let idx12 = i * maxAtoms + j
                mask[idx12 / 32] |= 1 << (idx12 % 32)
                // 1-3 pairs (share a bond with i)
                for k in adj[j] where k > i && k != i {
                    let idx13 = min(i, k) * maxAtoms + max(i, k)
                    mask[idx13 / 32] |= 1 << (idx13 % 32)
                }
            }
        }

        exclusionMaskBuffer = device.makeBuffer(
            bytes: &mask,
            length: mask.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
        let referenceIntraEnergy = intramolecularReferenceEnergy(
            ligandAtoms: gpuLigAtoms,
            exclusionMask: mask,
            maxAtoms: maxAtoms
        )

        let popSize = config.populationSize
        let poseSize = popSize * MemoryLayout<DockPose>.stride
        // Reuse population buffers if they're already large enough
        if poseSize > lastPopulationBufferCapacity {
            populationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            offspringBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            bestPopulationBuffer = device.makeBuffer(length: poseSize, options: .storageModeShared)
            lastPopulationBufferCapacity = poseSize
        }

        var gaParams = GAParams(
            populationSize: UInt32(popSize),
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: UInt32(max(config.localSearchSteps, 1)),
            mutationRate: config.mutationRate,
            crossoverRate: config.crossoverRate,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: config.gridSpacing,
            ligandRadius: ligandRadius,
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: referenceIntraEnergy
        )
        gaParamsBuffer = device.makeBuffer(bytes: &gaParams, length: MemoryLayout<GAParams>.stride, options: .storageModeShared)

        let tgSize = MTLSize(width: min(popSize, 256), height: 1, depth: 1)
        let tgCount = MTLSize(width: (popSize + 255) / 256, height: 1, depth: 1)

        if !config.enableFlexibility {
            gaParams.numTorsions = 0
            gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)
        }

        let totalRuns = max(config.numRuns, 1)
        var aggregatedResults: [DockingResult] = []
        let localSearchFrequency = max(config.localSearchFrequency, 1)
        let liveUpdateFrequency = max(config.liveUpdateFrequency, 1)

        func emitLiveUpdate(generation: Int) {
            if let best = extractBestPose(from: bestPopulationBuffer, ligandAtoms: heavyAtoms, centroid: centroid) {
                bestEnergy = min(bestEnergy, best.energy)
                let interactions = InteractionDetector.detect(
                    ligandAtoms: heavyAtoms,
                    ligandPositions: best.transformedAtomPositions,
                    proteinAtoms: proteinAtoms,
                    ligandBonds: heavyBonds
                )
                onPoseUpdate?(best, interactions)
            }
            onGenerationComplete?(generation, bestEnergy)
        }

        runLoop: for runIndex in 0..<totalRuns {
            guard isRunning else { break }

            dispatchCompute(pipeline: initPopPipeline, buffers: [
                (populationBuffer!, 0), (gridParamsBuffer!, 1), (gaParamsBuffer!, 2)
            ], threadGroups: tgCount, threadGroupSize: tgSize)
            localOptimize(buffer: populationBuffer!, tg: tgCount, tgs: tgSize)
            scorePopulation(buffer: populationBuffer!, tg: tgCount, tgs: tgSize)
            copyPoseBuffer(from: populationBuffer!, to: bestPopulationBuffer!, poseCount: popSize)

            let generationBase = runIndex * config.generationsPerRun

            let explorationCutoff = Int(Float(config.generationsPerRun) * config.explorationPhaseRatio)

            for step in 0..<config.generationsPerRun {
                guard isRunning else {
                    aggregatedResults.append(contentsOf: extractAllResults(
                        from: bestPopulationBuffer,
                        ligandAtoms: heavyAtoms,
                        centroid: centroid,
                        idOffset: aggregatedResults.count
                    ))
                    break runLoop
                }
                let globalGeneration = generationBase + step
                currentGeneration = globalGeneration
                gaParams.generation = UInt32(globalGeneration)

                // Two-phase search: exploration phase uses wider steps for broader grid coverage,
                // refinement phase uses tighter steps for precise local optimization
                if step < explorationCutoff {
                    // Exploration phase: broader search
                    gaParams.translationStep = config.explorationTranslationStep
                    gaParams.rotationStep = config.explorationRotationStep
                    gaParams.mutationRate = config.explorationMutationRate
                } else if step == explorationCutoff {
                    // Switch to refinement phase
                    gaParams.translationStep = config.translationStep
                    gaParams.rotationStep = config.rotationStep
                    gaParams.mutationRate = config.mutationRate
                }

                gaParamsBuffer?.contents().copyMemory(from: &gaParams, byteCount: MemoryLayout<GAParams>.stride)

                let perturbBuffers: [(MTLBuffer, Int)] = [
                    (offspringBuffer!, 0), (populationBuffer!, 1),
                    (gaParamsBuffer!, 2), (gridParamsBuffer!, 3)
                ]
                let scoreBuffers: [(MTLBuffer, Int)] = [
                    (offspringBuffer!, 0), (ligandAtomBuffer!, 1),
                    (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
                    (gridParamsBuffer!, 4), (gaParamsBuffer!, 5),
                    (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
                    (exclusionMaskBuffer!, 8)
                ]
                let acceptBuffers: [(MTLBuffer, Int)] = [
                    (populationBuffer!, 0), (offspringBuffer!, 1),
                    (bestPopulationBuffer!, 2), (gaParamsBuffer!, 3)
                ]
                var dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])] = [
                    (pipeline: mcPerturbPipeline, buffers: perturbBuffers)
                ]
                if step % localSearchFrequency == 0 {
                    dispatches.append((pipeline: localSearchPipeline, buffers: scoreBuffers))
                }
                dispatches.append((pipeline: scorePipeline, buffers: scoreBuffers))
                dispatches.append((pipeline: metropolisAcceptPipeline, buffers: acceptBuffers))
                dispatchBatch(dispatches, threadGroups: tgCount, threadGroupSize: tgSize)

                // Update more frequently during exploration phase for smoother visualization
                // of ligand moving through the binding site, less during refinement
                let updateFreq = step < explorationCutoff
                    ? max(liveUpdateFrequency / 2, 1)  // 2x more frequent during exploration
                    : liveUpdateFrequency
                if step % updateFreq == 0 || step == config.generationsPerRun - 1 {
                    emitLiveUpdate(generation: globalGeneration)
                }

                await Task.yield()
            }

            aggregatedResults.append(contentsOf: extractAllResults(
                from: bestPopulationBuffer,
                ligandAtoms: heavyAtoms,
                centroid: centroid,
                idOffset: aggregatedResults.count
            ))
        }

        let clustered = clusterPoses(aggregatedResults)
        let reranked = rerankClusterRepresentativesExplicit(
            clustered,
            ligandAtoms: heavyAtoms,
            centroid: centroid
        )

        // Compute and store quality diagnostics
        lastDiagnostics = computeDiagnostics(
            results: reranked,
            ligandAtoms: heavyAtoms,
            heavyBonds: heavyBonds
        )

        isRunning = false
        onDockingComplete?(reranked)
        return reranked
    }

    func stopDocking() { isRunning = false }

    // MARK: - Debug Scoring

    /// Score a user-specified pose against the currently loaded typed Vina maps.
    /// Useful for checking whether the crystal/native pose is actually favorable
    /// under the same Metal kernels used by the GA/ILS search.
    func debugScorePose(
        ligand: Molecule,
        translation: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        torsions: [Float] = []
    ) -> DockingResult? {
        guard vinaAffinityGridBuffer != nil,
              vinaTypeIndexBuffer != nil,
              gridParamsBuffer != nil else {
            return nil
        }

        let preparedLigand = prepareLigandGeometry(ligand)
        let heavyAtoms = preparedLigand.heavyAtoms
        let heavyBonds = preparedLigand.heavyBonds
        let centroid = preparedLigand.centroid
        var gpuLigAtoms = preparedLigand.gpuAtoms

        ligandAtomBuffer = device.makeBuffer(
            bytes: &gpuLigAtoms,
            length: gpuLigAtoms.count * MemoryLayout<DockLigandAtom>.stride,
            options: .storageModeShared
        )

        var torsionEdges: [TorsionEdge] = []
        var movingIndices: [Int32] = []
        for edge in buildTorsionTree(for: ligand, heavyBonds: heavyBonds) {
            torsionEdges.append(TorsionEdge(
                atom1: Int32(edge.atom1),
                atom2: Int32(edge.atom2),
                movingStart: Int32(movingIndices.count),
                movingCount: Int32(edge.movingAtoms.count)
            ))
            movingIndices.append(contentsOf: edge.movingAtoms.map(Int32.init))
        }
        let numTorsions = min(torsionEdges.count, 32)

        if torsionEdges.isEmpty {
            torsionEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0))
        }
        if movingIndices.isEmpty {
            movingIndices.append(0)
        }

        torsionEdgeBuffer = device.makeBuffer(
            bytes: &torsionEdges,
            length: torsionEdges.count * MemoryLayout<TorsionEdge>.stride,
            options: .storageModeShared
        )
        movingIndicesBuffer = device.makeBuffer(
            bytes: &movingIndices,
            length: movingIndices.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        )

        let maxAtoms = 128
        let maskWords = (maxAtoms * maxAtoms + 31) / 32
        var mask = [UInt32](repeating: 0, count: maskWords)
        var adjacency = Array(repeating: [Int](), count: gpuLigAtoms.count)
        for bond in heavyBonds {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard a < gpuLigAtoms.count, b < gpuLigAtoms.count else { continue }
            adjacency[a].append(b)
            adjacency[b].append(a)
        }
        for i in 0..<gpuLigAtoms.count {
            for j in adjacency[i] where j > i {
                let idx12 = i * maxAtoms + j
                mask[idx12 / 32] |= 1 << (idx12 % 32)
                for k in adjacency[j] where k > i && k != i {
                    let idx13 = min(i, k) * maxAtoms + max(i, k)
                    mask[idx13 / 32] |= 1 << (idx13 % 32)
                }
            }
        }
        exclusionMaskBuffer = device.makeBuffer(
            bytes: &mask,
            length: mask.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        var pose = DockPose()
        pose.translation = translation
        pose.energy = 1e10
        pose.rotation = SIMD4<Float>(rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafeMutablePointer(to: &pose.torsions) {
            $0.withMemoryRebound(to: Float.self, capacity: 32) { buffer in
                for i in 0..<min(torsions.count, 32) {
                    buffer[i] = torsions[i]
                }
            }
        }
        pose.numTorsions = Int32(numTorsions)
        pose.generation = 0
        pose.stericEnergy = 0
        pose.hydrophobicEnergy = 0
        pose.hbondEnergy = 0
        pose.torsionPenalty = 0
        pose.clashPenalty = 0
        pose._pad0 = 0

        populationBuffer = device.makeBuffer(
            bytes: &pose,
            length: MemoryLayout<DockPose>.stride,
            options: .storageModeShared
        )

        var gaParams = GAParams(
            populationSize: 1,
            numLigandAtoms: UInt32(gpuLigAtoms.count),
            numTorsions: UInt32(numTorsions),
            generation: 0,
            localSearchSteps: 1,
            mutationRate: 0,
            crossoverRate: 0,
            translationStep: config.translationStep,
            rotationStep: config.rotationStep,
            torsionStep: config.torsionStep,
            gridSpacing: gridParams.spacing,
            ligandRadius: max(sqrt(gpuLigAtoms.map { simd_length_squared($0.position) }.reduce(0, +) / Float(max(gpuLigAtoms.count, 1))), 1.0),
            mcTemperature: config.mcTemperature,
            referenceIntraEnergy: intramolecularReferenceEnergy(
                ligandAtoms: gpuLigAtoms,
                exclusionMask: mask,
                maxAtoms: maxAtoms
            )
        )
        gaParamsBuffer = device.makeBuffer(
            bytes: &gaParams,
            length: MemoryLayout<GAParams>.stride,
            options: .storageModeShared
        )

        let tgSize = MTLSize(width: 1, height: 1, depth: 1)
        let tgCount = MTLSize(width: 1, height: 1, depth: 1)
        let wasRunning = isRunning
        isRunning = true
        scorePopulation(buffer: populationBuffer!, tg: tgCount, tgs: tgSize)
        isRunning = wasRunning

        return extractBestPose(ligandAtoms: heavyAtoms, centroid: centroid)
    }

    // MARK: - GPU Helpers

    private func localOptimize(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        dispatchCompute(pipeline: localSearchPipeline, buffers: [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer!, 4), (gaParamsBuffer!, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ], threadGroups: tg, threadGroupSize: tgs)
    }

    private func scorePopulation(buffer: MTLBuffer, tg: MTLSize, tgs: MTLSize) {
        dispatchCompute(pipeline: scorePipeline, buffers: [
            (buffer, 0), (ligandAtomBuffer!, 1),
            (vinaAffinityGridBuffer!, 2), (vinaTypeIndexBuffer!, 3),
            (gridParamsBuffer!, 4), (gaParamsBuffer!, 5),
            (torsionEdgeBuffer!, 6), (movingIndicesBuffer!, 7),
            (exclusionMaskBuffer!, 8)
        ], threadGroups: tg, threadGroupSize: tgs)
    }

    private func dispatchCompute(
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        threadGroups: MTLSize, threadGroupSize: MTLSize
    ) {
        guard isRunning else { return }  // Don't dispatch if stopped
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    /// Batch multiple GPU dispatches into a single command buffer (reduces CPU/GPU sync barriers).
    private func dispatchBatch(_ dispatches: [(pipeline: MTLComputePipelineState, buffers: [(MTLBuffer, Int)])],
                                threadGroups: MTLSize, threadGroupSize: MTLSize) {
        guard isRunning else { return }
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        for d in dispatches {
            enc.setComputePipelineState(d.pipeline)
            for (buf, idx) in d.buffers { enc.setBuffer(buf, offset: 0, index: idx) }
            enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func copyPoseBuffer(from source: MTLBuffer, to destination: MTLBuffer, poseCount: Int) {
        let byteCount = poseCount * MemoryLayout<DockPose>.stride
        destination.contents().copyMemory(from: source.contents(), byteCount: byteCount)
    }

    // MARK: - Result Extraction

    // Vina scores are already in kcal/mol — no scaling needed.

    /// Apply rigid-body + torsion transform to recover docked atom positions.
    /// Uses the exact same formulas as the GPU kernels for bit-exact consistency:
    ///   - Rigid body: quatRotate() formula from DockingCompute.metal line 75-79
    ///   - Torsions: Rodrigues rotation from DockingCompute.metal line 114-124
    private func applyPoseTransform(_ pose: DockPose, ligandAtoms: [Atom], centroid: SIMD3<Float>) -> [SIMD3<Float>] {
        let q = SIMD4<Float>(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)
        let u = SIMD3<Float>(q.x, q.y, q.z)
        let s = q.w
        let trans = SIMD3<Float>(pose.translation.x, pose.translation.y, pose.translation.z)

        // Step 1: rigid-body transform — matches GPU quatRotate exactly:
        //   result = 2*dot(u,v)*u + (s*s - dot(u,u))*v + 2*s*cross(u,v)
        var positions = ligandAtoms.map { atom -> SIMD3<Float> in
            let v = atom.position - centroid
            let rotated = 2.0 * simd_dot(u, v) * u + (s * s - simd_dot(u, u)) * v + 2.0 * s * simd_cross(u, v)
            return rotated + trans
        }

        // Step 2: apply torsion rotations — matches GPU Rodrigues exactly:
        //   rotated = v*cos(a) + cross(axis,v)*sin(a) + axis*dot(axis,v)*(1-cos(a))
        if let edgeBuf = torsionEdgeBuffer, let idxBuf = movingIndicesBuffer {
            let edges = edgeBuf.contents().bindMemory(to: TorsionEdge.self, capacity: Int(pose.numTorsions))
            let moving = idxBuf.contents().bindMemory(to: Int32.self, capacity: idxBuf.length / MemoryLayout<Int32>.stride)

            for t in 0..<Int(pose.numTorsions) {
                let angle = withUnsafePointer(to: pose.torsions) {
                    $0.withMemoryRebound(to: Float.self, capacity: 32) { $0[t] }
                }
                if abs(angle) < 1e-6 { continue }

                let edge = edges[t]
                let pivotIdx = Int(edge.atom1)
                let axisIdx = Int(edge.atom2)
                guard pivotIdx < positions.count, axisIdx < positions.count else { continue }

                let pivot = positions[pivotIdx]
                let axis = simd_normalize(positions[axisIdx] - pivot)
                let cosA = cos(angle)
                let sinA = sin(angle)

                for i in 0..<Int(edge.movingCount) {
                    let atomIdx = Int(moving[Int(edge.movingStart) + i])
                    guard atomIdx >= 0, atomIdx < positions.count else { continue }
                    let v = positions[atomIdx] - pivot
                    // Rodrigues rotation (exact GPU formula)
                    let rotated = v * cosA + simd_cross(axis, v) * sinA + axis * simd_dot(axis, v) * (1.0 - cosA)
                    positions[atomIdx] = pivot + rotated
                }
            }
        }

        return positions
    }

    private func extractBestPose(from buffer: MTLBuffer? = nil, ligandAtoms: [Atom], centroid: SIMD3<Float>) -> DockingResult? {
        guard let buffer = buffer ?? populationBuffer else { return nil }
        let poseCount = buffer.length / MemoryLayout<DockPose>.stride
        guard poseCount > 0 else { return nil }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        // Find best VALID pose (finite energy, below sentinel threshold)
        var bestIdx = -1
        var bestE: Float = .infinity
        for i in 0..<poseCount {
            let e = poses[i].energy
            guard e.isFinite, e < 1e9 else { continue }
            if e < bestE { bestE = e; bestIdx = i }
        }
        guard bestIdx >= 0 else { return nil }

        let p = poses[bestIdx]
        let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
        let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)

        let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

        // Vina scores are already in kcal/mol — no kCalScale division needed
        return DockingResult(
            id: bestIdx,
            pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p)),
            energy: p.energy,
            stericEnergy: p.stericEnergy,
            hydrophobicEnergy: p.hydrophobicEnergy,
            hbondEnergy: p.hbondEnergy,
            torsionPenalty: p.torsionPenalty,
            generation: Int(p.generation),
            transformedAtomPositions: transformed
        )
    }

    private func extractAllResults(
        from buffer: MTLBuffer? = nil,
        ligandAtoms: [Atom],
        centroid: SIMD3<Float>,
        idOffset: Int = 0,
        sortByEnergy: Bool = true
    ) -> [DockingResult] {
        guard let buffer = buffer ?? populationBuffer else { return [] }
        let poseCount = buffer.length / MemoryLayout<DockPose>.stride
        guard poseCount > 0 else { return [] }
        let poses = buffer.contents().bindMemory(to: DockPose.self, capacity: poseCount)

        var results: [DockingResult] = []
        results.reserveCapacity(poseCount)

        for i in 0..<poseCount {
            let p = poses[i]
            // Skip invalid poses: sentinel energy (1e10 from initialization), NaN, or inf
            guard p.energy.isFinite, p.energy < 1e9 else { continue }

            let quat = simd_quatf(ix: p.rotation.x, iy: p.rotation.y, iz: p.rotation.z, r: p.rotation.w)
            let trans = SIMD3<Float>(p.translation.x, p.translation.y, p.translation.z)
            let transformed = applyPoseTransform(p, ligandAtoms: ligandAtoms, centroid: centroid)

            // Skip poses with NaN positions (degenerate quaternion/torsion)
            guard transformed.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { continue }

            results.append(DockingResult(
                id: results.count + idOffset,
                pose: DockPoseSwift(translation: trans, rotation: quat, torsions: torsions(from: p)),
                energy: p.energy,
                stericEnergy: p.stericEnergy,
                hydrophobicEnergy: p.hydrophobicEnergy,
                hbondEnergy: p.hbondEnergy,
                torsionPenalty: p.torsionPenalty,
                generation: Int(p.generation),
                transformedAtomPositions: transformed
            ))
        }
        return sortByEnergy ? results.sorted { $0.energy < $1.energy } : results
    }

    // MARK: - RMSD Clustering

    private func clusterPoses(_ results: [DockingResult]) -> [DockingResult] {
        guard !results.isEmpty else { return [] }
        let threshold: Float = 2.0
        var out = results

        var clusterID = 0
        for i in 0..<out.count {
            guard out[i].clusterID == -1 else { continue }
            out[i].clusterID = clusterID
            out[i].clusterRank = 0
            var rank = 1
            for j in (i+1)..<out.count {
                guard out[j].clusterID == -1 else { continue }
                if rmsd(out[i].transformedAtomPositions, out[j].transformedAtomPositions) < threshold {
                    out[j].clusterID = clusterID
                    out[j].clusterRank = rank
                    rank += 1
                }
            }
            clusterID += 1
        }
        return out
    }

    private func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        let s = zip(a, b).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
        return sqrt(s / Float(a.count))
    }

    private func ligandWithDockingCharges(_ ligand: Molecule) -> Molecule {
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        guard !heavyAtoms.isEmpty else { return ligand }

        let hasPartialCharges = heavyAtoms.contains { abs($0.charge) > 1e-4 }
        guard !hasPartialCharges else { return ligand }

        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        guard let charged = RDKitBridge.computeChargesMolBlock(molBlock),
              charged.atoms.count == ligand.atoms.count else {
            return ligand
        }

        var mergedAtoms = ligand.atoms
        for i in mergedAtoms.indices {
            mergedAtoms[i].charge = charged.atoms[i].charge
            if mergedAtoms[i].formalCharge == 0 {
                mergedAtoms[i].formalCharge = charged.atoms[i].formalCharge
            }
        }

        return Molecule(
            name: ligand.name,
            atoms: mergedAtoms,
            bonds: ligand.bonds,
            title: ligand.title,
            smiles: ligand.smiles
        )
    }

    private func electrostaticCharge(for atom: Atom) -> Float {
        abs(atom.charge) > 1e-4 ? atom.charge : Float(atom.formalCharge)
    }

    private func buildTorsionTree(for ligand: Molecule, heavyBonds: [Bond]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        let molBlock = SDFWriter.molBlock(
            name: ligand.name,
            atoms: ligand.atoms,
            bonds: ligand.bonds,
            includeTerminator: false
        )
        if let tree = RDKitBridge.buildTorsionTreeMolBlock(molBlock), !tree.isEmpty {
            return tree
        }

        // Prefer the explicit SMILES property (set when molecule comes from SMILES input).
        // Falls back to title only if it looks like a valid SMILES string (contains ring digits
        // or parentheses, and no spaces — distinguishes SMILES from PDB titles).
        let smilesSource = ligand.smiles
            ?? (ligand.title.contains(where: { $0 == "(" || $0 == ")" || $0.isNumber })
                && !ligand.title.contains(" ")
                ? ligand.title : nil)

        if let smi = smilesSource, !smi.isEmpty,
           let tree = RDKitBridge.buildTorsionTree(smiles: smi), !tree.isEmpty {
            return tree
        }
        return buildGraphTorsionTree(atomCount: ligand.heavyAtomCount, bonds: heavyBonds)
    }

    private func buildGraphTorsionTree(atomCount: Int, bonds: [Bond]) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        guard atomCount > 1 else { return [] }

        var adjacency = Array(repeating: [Int](), count: atomCount)
        for bond in bonds {
            guard bond.atomIndex1 < atomCount, bond.atomIndex2 < atomCount else { continue }
            adjacency[bond.atomIndex1].append(bond.atomIndex2)
            adjacency[bond.atomIndex2].append(bond.atomIndex1)
        }

        func hasAlternatePath(from start: Int, to target: Int, excluding edge: (Int, Int)) -> Bool {
            var visited: Set<Int> = [start]
            var queue = [start]
            while !queue.isEmpty {
                let current = queue.removeFirst()
                for next in adjacency[current] {
                    if (current == edge.0 && next == edge.1) || (current == edge.1 && next == edge.0) {
                        continue
                    }
                    if next == target { return true }
                    if visited.insert(next).inserted {
                        queue.append(next)
                    }
                }
            }
            return false
        }

        func bfsSide(start: Int, excluding: Int) -> Set<Int> {
            var visited: Set<Int> = [start]
            var queue = [start]
            while !queue.isEmpty {
                let current = queue.removeFirst()
                for next in adjacency[current] where next != excluding {
                    if visited.insert(next).inserted {
                        queue.append(next)
                    }
                }
            }
            return visited
        }

        var bfsOrder = Array(repeating: Int.max, count: atomCount)
        var orderCounter = 0
        for root in 0..<atomCount where bfsOrder[root] == Int.max {
            var queue = [root]
            bfsOrder[root] = orderCounter
            orderCounter += 1
            while !queue.isEmpty {
                let current = queue.removeFirst()
                for next in adjacency[current] where bfsOrder[next] == Int.max {
                    bfsOrder[next] = orderCounter
                    orderCounter += 1
                    queue.append(next)
                }
            }
        }

        var torsions: [(atom1: Int, atom2: Int, movingAtoms: [Int])] = []
        for bond in bonds where bond.order == .single {
            let a = bond.atomIndex1
            let b = bond.atomIndex2
            guard adjacency[a].count > 1, adjacency[b].count > 1 else { continue }
            guard !hasAlternatePath(from: a, to: b, excluding: (a, b)) else { continue }

            let forward = bfsSide(start: b, excluding: a)
            let backward = bfsSide(start: a, excluding: b)

            let edge: (atom1: Int, atom2: Int, movingAtoms: [Int])
            if forward.count <= backward.count {
                edge = (a, b, forward.sorted())
            } else {
                edge = (b, a, backward.sorted())
            }

            guard !edge.movingAtoms.isEmpty else { continue }
            torsions.append(edge)
        }

        torsions.sort {
            let lhsKey = (bfsOrder[$0.atom1], bfsOrder[$0.atom2], $0.movingAtoms.count)
            let rhsKey = (bfsOrder[$1.atom1], bfsOrder[$1.atom2], $1.movingAtoms.count)
            return lhsKey < rhsKey
        }
        return torsions
    }

    // MARK: - Force Field Parameters

    // MARK: - Docking Diagnostics

    /// Compute quality diagnostics for a set of docking results.
    func computeDiagnostics(
        results: [DockingResult],
        ligandAtoms: [Atom],
        heavyBonds: [Bond]
    ) -> DockingDiagnostics {
        let gridOrigin = SIMD3<Float>(gridParams.origin.x, gridParams.origin.y, gridParams.origin.z)
        let gridEnd = gridOrigin + SIMD3<Float>(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)) * gridParams.spacing
        let gridCenter = (gridOrigin + gridEnd) * 0.5

        let validResults = results.filter { $0.energy.isFinite && $0.energy < 1e9 }
        let energies = validResults.map(\.energy)

        // Energy statistics
        let minE = energies.min() ?? .infinity
        let maxE = energies.max() ?? -.infinity
        let meanE = energies.isEmpty ? 0 : energies.reduce(0, +) / Float(energies.count)
        let variance = energies.isEmpty ? 0 : energies.map { ($0 - meanE) * ($0 - meanE) }.reduce(0, +) / Float(energies.count)
        let stddevE = sqrt(variance)

        // Pose location analysis
        var insideGrid = 0
        var outsideGrid = 0
        var centroidDistances: [Float] = []
        var minProteinDistances: [Float] = []  // closest protein atom per pose

        for r in validResults {
            let positions = r.transformedAtomPositions
            let allInside = positions.allSatisfy { p in
                p.x >= gridOrigin.x && p.x <= gridEnd.x &&
                p.y >= gridOrigin.y && p.y <= gridEnd.y &&
                p.z >= gridOrigin.z && p.z <= gridEnd.z
            }
            if allInside { insideGrid += 1 } else { outsideGrid += 1 }

            // Centroid distance to grid center
            if !positions.isEmpty {
                let centroid = positions.reduce(.zero, +) / Float(positions.count)
                centroidDistances.append(simd_distance(centroid, gridCenter))

                // Minimum distance from any ligand atom to any protein atom
                if !proteinAtoms.isEmpty {
                    var minDist: Float = .infinity
                    for lp in positions {
                        for pa in proteinAtoms {
                            let d = simd_distance(lp, pa.position)
                            if d < minDist { minDist = d }
                        }
                    }
                    minProteinDistances.append(minDist)
                }
            }
        }

        let meanCentroidDist = centroidDistances.isEmpty ? 0 : centroidDistances.reduce(0, +) / Float(centroidDistances.count)
        let meanProteinDist = minProteinDistances.isEmpty ? 0 : minProteinDistances.reduce(0, +) / Float(minProteinDistances.count)

        // Poses making protein contact (min distance < 4 Å)
        let contactPoses = minProteinDistances.filter { $0 < 4.0 }.count

        // Bond length analysis (top 10 poses)
        var bondDeviations: [Float] = []
        for r in validResults.prefix(10) {
            let positions = r.transformedAtomPositions
            for b in heavyBonds {
                guard b.atomIndex1 < ligandAtoms.count, b.atomIndex2 < ligandAtoms.count,
                      b.atomIndex1 < positions.count, b.atomIndex2 < positions.count else { continue }
                let orig = simd_distance(ligandAtoms[b.atomIndex1].position, ligandAtoms[b.atomIndex2].position)
                guard orig > 0.01 else { continue }
                let docked = simd_distance(positions[b.atomIndex1], positions[b.atomIndex2])
                guard docked.isFinite else { continue }
                bondDeviations.append(abs(docked - orig))
            }
        }
        let meanBondDev = bondDeviations.isEmpty ? 0 : bondDeviations.reduce(0, +) / Float(bondDeviations.count)
        let maxBondDev = bondDeviations.max() ?? 0

        // Cluster count
        let clusterCount = Set(validResults.map(\.clusterID)).count

        // Spatial exploration: how spread out are the pose centroids?
        let centroidSpread: Float
        if centroidDistances.count >= 2 {
            let meanCD = centroidDistances.reduce(0, +) / Float(centroidDistances.count)
            let cdVar = centroidDistances.map { ($0 - meanCD) * ($0 - meanCD) }.reduce(0, +) / Float(centroidDistances.count)
            centroidSpread = sqrt(cdVar)
        } else {
            centroidSpread = 0
        }

        return DockingDiagnostics(
            totalPopulation: results.count,
            validPoses: validResults.count,
            invalidPoses: results.count - validResults.count,
            posesInsideGrid: insideGrid,
            posesOutsideGrid: outsideGrid,
            posesWithProteinContact: contactPoses,
            meanLigandProteinDistance: meanProteinDist,
            meanCentroidToGridCenter: meanCentroidDist,
            centroidSpread: centroidSpread,
            minEnergy: minE,
            maxEnergy: maxE,
            meanEnergy: meanE,
            energyStdDev: stddevE,
            clusterCount: clusterCount,
            meanBondLengthDeviation: meanBondDev,
            maxBondLengthDeviation: maxBondDev,
            gridDimensions: SIMD3(Float(gridParams.dims.x), Float(gridParams.dims.y), Float(gridParams.dims.z)),
            gridSpacing: gridParams.spacing,
            gridBoxSize: gridEnd - gridOrigin
        )
    }
}

/// Docking quality metrics computed after a docking run.
struct DockingDiagnostics: Sendable {
    // Population
    var totalPopulation: Int
    var validPoses: Int         // energy finite and < 1e9
    var invalidPoses: Int       // sentinel, NaN, or inf

    // Spatial quality
    var posesInsideGrid: Int    // all atoms within grid bounds
    var posesOutsideGrid: Int   // any atom outside grid
    var posesWithProteinContact: Int // min ligand-protein distance < 4 Å
    var meanLigandProteinDistance: Float // avg min distance to protein per pose
    var meanCentroidToGridCenter: Float
    var centroidSpread: Float   // stddev of centroid distances (exploration metric)

    // Energy
    var minEnergy: Float
    var maxEnergy: Float
    var meanEnergy: Float
    var energyStdDev: Float

    // Clustering
    var clusterCount: Int

    // Geometry preservation
    var meanBondLengthDeviation: Float  // avg Å deviation from input
    var maxBondLengthDeviation: Float

    // Grid info
    var gridDimensions: SIMD3<Float>    // grid points per axis
    var gridSpacing: Float
    var gridBoxSize: SIMD3<Float>       // Å per axis

    /// Human-readable summary
    var summary: String {
        """
        Docking Diagnostics:
          Population: \(totalPopulation) total, \(validPoses) valid, \(invalidPoses) invalid
          Grid: \(String(format: "%.0f×%.0f×%.0f", gridDimensions.x, gridDimensions.y, gridDimensions.z)) pts (\(String(format: "%.0f×%.0f×%.0f", gridBoxSize.x, gridBoxSize.y, gridBoxSize.z)) Å at \(gridSpacing) spacing)
          Poses inside grid: \(posesInsideGrid)/\(validPoses), outside: \(posesOutsideGrid)
          Protein contacts: \(posesWithProteinContact)/\(validPoses) (mean dist \(String(format: "%.1f", meanLigandProteinDistance)) Å)
          Centroid spread: \(String(format: "%.1f", centroidSpread)) Å (exploration diversity)
          Energy: min=\(String(format: "%.1f", minEnergy)) max=\(String(format: "%.1f", maxEnergy)) mean=\(String(format: "%.1f", meanEnergy)) σ=\(String(format: "%.1f", energyStdDev))
          Clusters: \(clusterCount)
          Bond preservation: mean=\(String(format: "%.3f", meanBondLengthDeviation))Å max=\(String(format: "%.3f", maxBondLengthDeviation))Å
        """
    }
}

// MARK: - Interaction Detection

enum InteractionDetector {

    // MARK: - Aromatic Ring Detection

    /// Detect aromatic rings from protein residues (known aromatic sidechains).
    /// Returns ring centroids and approximate normal vectors.
    struct AromaticRing: Sendable {
        let centroid: SIMD3<Float>
        let normal: SIMD3<Float>
        let atomIndices: [Int]  // indices into the atoms array
    }

    /// Known aromatic sidechain atoms for standard amino acids
    private static let aromaticResidueAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  // both rings
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    ]

    static func detectAromaticRings(
        atoms: [Atom],
        positions: [SIMD3<Float>]? = nil,
        bonds: [Bond]? = nil
    ) -> [AromaticRing] {
        var rings: [AromaticRing] = []

        // 1. Standard protein aromatic residues (PHE, TYR, TRP, HIS)
        var residueAtoms: [String: [(Int, Atom)]] = [:]
        for (i, atom) in atoms.enumerated() {
            let key = "\(atom.chainID)_\(atom.residueSeq)_\(atom.residueName)"
            residueAtoms[key, default: []].append((i, atom))
        }

        for (_, atomList) in residueAtoms {
            guard let resName = atomList.first?.1.residueName,
                  let targetNames = aromaticResidueAtoms[resName] else { continue }

            let ringAtoms = atomList.filter { targetNames.contains($0.1.name.trimmingCharacters(in: .whitespaces)) }
            guard ringAtoms.count >= 5 else { continue }

            if let ring = buildRing(indices: ringAtoms.map(\.0), atoms: atoms, positions: positions) {
                rings.append(ring)
            }
        }

        // 2. Graph-based ring detection for non-standard residues (ligands)
        //    Find 5- and 6-membered rings from bond connectivity, then check planarity.
        if let bonds = bonds {
            let graphRings = detectRingsFromBonds(atoms: atoms, bonds: bonds, positions: positions)
            rings.append(contentsOf: graphRings)
        }

        return rings
    }

    /// Build an AromaticRing from known atom indices.
    private static func buildRing(
        indices: [Int], atoms: [Atom], positions: [SIMD3<Float>]?
    ) -> AromaticRing? {
        let ringPositions = indices.map { positions?[$0] ?? atoms[$0].position }
        let centroid = ringPositions.reduce(.zero, +) / Float(ringPositions.count)

        var normal = SIMD3<Float>(0, 1, 0)
        if ringPositions.count >= 3 {
            let v1 = ringPositions[1] - ringPositions[0]
            let v2 = ringPositions[2] - ringPositions[0]
            let n = simd_cross(v1, v2)
            let len = simd_length(n)
            if len > 1e-6 { normal = n / len }
        }
        return AromaticRing(centroid: centroid, normal: normal, atomIndices: indices)
    }

    /// Detect aromatic rings from bond connectivity (for ligands/non-standard residues).
    /// Finds all 5- and 6-membered rings, then filters by planarity and element composition.
    private static func detectRingsFromBonds(
        atoms: [Atom], bonds: [Bond], positions: [SIMD3<Float>]?
    ) -> [AromaticRing] {
        let n = atoms.count
        guard n > 4 else { return [] }

        // Build adjacency list (only C/N/O/S — skip H and metals)
        let aromaticElements: Set<Element> = [.C, .N, .O, .S]
        var adj: [[Int]] = Array(repeating: [], count: n)
        for bond in bonds {
            let a = bond.atomIndex1, b = bond.atomIndex2
            guard a < n, b < n else { continue }
            guard aromaticElements.contains(atoms[a].element),
                  aromaticElements.contains(atoms[b].element) else { continue }
            adj[a].append(b)
            adj[b].append(a)
        }

        // Find all simple cycles of length 5 or 6 using DFS from each atom.
        // To avoid duplicates, only start from the smallest-index atom in the cycle.
        var foundRings: Set<[Int]> = []

        for start in 0..<n {
            guard aromaticElements.contains(atoms[start].element) else { continue }
            // BFS/DFS limited depth search for cycles back to start
            findCycles(start: start, adj: adj, maxLen: 6, found: &foundRings)
        }

        // Filter by planarity: all atoms within 0.5 Å of the best-fit plane
        var result: [AromaticRing] = []
        for ringIndices in foundRings {
            let ringPos = ringIndices.map { positions?[$0] ?? atoms[$0].position }

            // Check planarity: compute normal from first 3 atoms, then check all are near the plane
            guard ringPos.count >= 5 else { continue }
            let v1 = ringPos[1] - ringPos[0]
            let v2 = ringPos[2] - ringPos[0]
            let normal = simd_cross(v1, v2)
            let normalLen = simd_length(normal)
            guard normalLen > 1e-6 else { continue }
            let n = normal / normalLen
            let centroid = ringPos.reduce(.zero, +) / Float(ringPos.count)

            // All atoms should be within 0.5 Å of the plane through centroid
            var planar = true
            for p in ringPos {
                let dist = abs(simd_dot(p - centroid, n))
                if dist > 0.5 { planar = false; break }
            }
            guard planar else { continue }

            // Must contain at least 4 carbon/nitrogen atoms (skip saturated rings)
            let cnCount = ringIndices.filter { atoms[$0].element == .C || atoms[$0].element == .N }.count
            guard cnCount >= ringIndices.count - 1 else { continue }

            result.append(AromaticRing(centroid: centroid, normal: n, atomIndices: ringIndices))
        }
        return result
    }

    /// Find simple cycles of length 5-6 that include `start` as the smallest index.
    private static func findCycles(
        start: Int, adj: [[Int]], maxLen: Int, found: inout Set<[Int]>
    ) {
        // DFS with path tracking, looking for paths that return to start
        var stack: [(node: Int, path: [Int])] = [(start, [start])]

        while !stack.isEmpty {
            let (node, path) = stack.removeLast()
            guard path.count <= maxLen else { continue }

            for neighbor in adj[node] {
                if neighbor == start && (path.count == 5 || path.count == 6) {
                    // Found a cycle — normalize so smallest index is first
                    let ring = path.sorted()
                    if ring[0] == start {  // only record if start is the smallest
                        found.insert(ring)
                    }
                } else if neighbor > start && !path.contains(neighbor) && path.count < maxLen {
                    stack.append((neighbor, path + [neighbor]))
                }
            }
        }
    }

    // MARK: - Full Interaction Detection

    static func detect(
        ligandAtoms: [Atom],
        ligandPositions: [SIMD3<Float>],
        proteinAtoms: [Atom],
        ligandBonds: [Bond] = []
    ) -> [MolecularInteraction] {
        var result: [MolecularInteraction] = []
        var idCounter = 0

        // Pre-detect aromatic rings
        let proteinRings = detectAromaticRings(atoms: proteinAtoms)
        let ligandRings = detectAromaticRings(
            atoms: ligandAtoms,
            positions: ligandPositions,
            bonds: ligandBonds.isEmpty ? nil : ligandBonds
        )

        // Track which ligand atoms already have a strong interaction (to limit hydrophobic clutter)
        var ligandHasStrongInteraction: Set<Int> = []

        // Charged residue atoms for salt bridges
        let positiveResAtoms: Set<String> = ["NZ", "NH1", "NH2", "NE"]  // Lys, Arg
        let negativeResAtoms: Set<String> = ["OD1", "OD2", "OE1", "OE2"]  // Asp, Glu

        // Metal atoms
        let metals: Set<Element> = [.Fe, .Zn, .Ca, .Mg, .Mn, .Cu]

        for (li, ligAtom) in ligandAtoms.enumerated() {
            guard li < ligandPositions.count else { continue }
            let lp = ligandPositions[li]

            for (pi, protAtom) in proteinAtoms.enumerated() {
                let d = simd_distance(lp, protAtom.position)
                guard d < 6.0 else { continue }  // coarse distance cutoff

                let protName = protAtom.name.trimmingCharacters(in: .whitespaces)

                // ---- Metal coordination: < 2.8 Å, metal ↔ N/O/S ----
                if d < 2.8 {
                    let ligCoord = ligAtom.element == .N || ligAtom.element == .O || ligAtom.element == .S
                    let protMetal = metals.contains(protAtom.element)
                    let ligMetal = metals.contains(ligAtom.element)
                    let protCoord = protAtom.element == .N || protAtom.element == .O || protAtom.element == .S

                    if (protMetal && ligCoord) || (ligMetal && protCoord) {
                        result.append(MolecularInteraction(
                            id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                            type: .metalCoord, distance: d,
                            ligandPosition: lp, proteinPosition: protAtom.position))
                        idCounter += 1
                        ligandHasStrongInteraction.insert(li)
                        continue
                    }
                }

                // ---- Salt bridge: < 4.0 Å, charged group ↔ charged group ----
                if d < 4.0 {
                    let protPositive = positiveResAtoms.contains(protName)
                    let protNegative = negativeResAtoms.contains(protName)
                    let ligPositive = ligAtom.formalCharge > 0
                    let ligNegative = ligAtom.formalCharge < 0

                    if (protPositive && ligNegative) || (protNegative && ligPositive) {
                        result.append(MolecularInteraction(
                            id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                            type: .saltBridge, distance: d,
                            ligandPosition: lp, proteinPosition: protAtom.position))
                        idCounter += 1
                        ligandHasStrongInteraction.insert(li)
                        continue
                    }
                }

                // ---- H-Bond: 2.2-3.5 Å between donor/acceptor (N/O) ----
                if d >= 2.2 && d <= 3.5 {
                    let ligDA = ligAtom.element == .N || ligAtom.element == .O
                    let proDA = protAtom.element == .N || protAtom.element == .O
                    if ligDA && proDA {
                        result.append(MolecularInteraction(
                            id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                            type: .hbond, distance: d,
                            ligandPosition: lp, proteinPosition: protAtom.position))
                        idCounter += 1
                        ligandHasStrongInteraction.insert(li)
                        continue
                    }
                }

                // ---- Halogen bond: 2.5-3.5 Å, halogen ↔ N/O ----
                if d >= 2.5 && d <= 3.5 {
                    let halogen = ligAtom.element == .F || ligAtom.element == .Cl || ligAtom.element == .Br
                    let acceptor = protAtom.element == .N || protAtom.element == .O
                    if halogen && acceptor {
                        result.append(MolecularInteraction(
                            id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                            type: .halogen, distance: d,
                            ligandPosition: lp, proteinPosition: protAtom.position))
                        idCounter += 1
                        ligandHasStrongInteraction.insert(li)
                    }
                }
            }
        }

        // ---- π-π stacking: ring centroid distance 3.3-5.5 Å ----
        for ligRing in ligandRings {
            for protRing in proteinRings {
                let d = simd_distance(ligRing.centroid, protRing.centroid)
                guard d >= 3.3 && d <= 5.5 else { continue }

                // Check angle between ring normals
                let dotN = abs(simd_dot(ligRing.normal, protRing.normal))
                // Face-to-face: angle < 30° (dot > 0.87) or edge-to-face: angle > 60° (dot < 0.5)
                let isFaceToFace = dotN > 0.85 && d < 4.2
                let isEdgeToFace = dotN < 0.5 && d >= 4.0 && d <= 5.5

                if isFaceToFace || isEdgeToFace {
                    let ligIdx = ligRing.atomIndices.first ?? 0
                    let protIdx = protRing.atomIndices.first ?? 0
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligIdx, proteinAtomIndex: protIdx,
                        type: .piStack, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        // ---- π-cation: ring centroid ↔ cation, < 6.0 Å ----
        // Protein cations near ligand rings
        for ligRing in ligandRings {
            for (pi, protAtom) in proteinAtoms.enumerated() {
                let protName = protAtom.name.trimmingCharacters(in: .whitespaces)
                guard positiveResAtoms.contains(protName) || protAtom.formalCharge > 0 else { continue }
                let d = simd_distance(ligRing.centroid, protAtom.position)
                guard d < 6.0 else { continue }

                // Check angle: cation should be roughly above/below the ring
                let toAtom = simd_normalize(protAtom.position - ligRing.centroid)
                let dotN = abs(simd_dot(toAtom, ligRing.normal))
                if dotN > 0.5 {  // within ~60° of ring normal
                    let ligIdx = ligRing.atomIndices.first ?? 0
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: ligIdx, proteinAtomIndex: pi,
                        type: .piCation, distance: d,
                        ligandPosition: ligRing.centroid, proteinPosition: protAtom.position))
                    idCounter += 1
                }
            }
        }
        // Ligand cations near protein rings
        for protRing in proteinRings {
            for (li, ligAtom) in ligandAtoms.enumerated() {
                guard li < ligandPositions.count else { continue }
                guard ligAtom.formalCharge > 0 || ligAtom.element == .N else { continue }
                let lp = ligandPositions[li]
                let d = simd_distance(protRing.centroid, lp)
                guard d < 6.0 else { continue }

                let toAtom = simd_normalize(lp - protRing.centroid)
                let dotN = abs(simd_dot(toAtom, protRing.normal))
                if dotN > 0.5 {
                    let protIdx = protRing.atomIndices.first ?? 0
                    result.append(MolecularInteraction(
                        id: idCounter, ligandAtomIndex: li, proteinAtomIndex: protIdx,
                        type: .piCation, distance: d,
                        ligandPosition: lp, proteinPosition: protRing.centroid))
                    idCounter += 1
                }
            }
        }

        // ---- Hydrophobic contacts: C/S ↔ C/S, 3.3-4.5 Å ----
        // Only for ligand atoms that don't already have a stronger interaction.
        // Limit to max 3 per ligand atom to avoid visual clutter.
        var hydroCountPerLigAtom: [Int: Int] = [:]
        let maxHydroPerAtom = 3

        for (li, ligAtom) in ligandAtoms.enumerated() {
            guard li < ligandPositions.count else { continue }
            guard !ligandHasStrongInteraction.contains(li) else { continue }
            guard ligAtom.element == .C || ligAtom.element == .S else { continue }
            let lp = ligandPositions[li]

            for (pi, protAtom) in proteinAtoms.enumerated() {
                guard protAtom.element == .C || protAtom.element == .S else { continue }
                let d = simd_distance(lp, protAtom.position)
                guard d >= 3.3 && d <= 4.5 else { continue }

                let count = hydroCountPerLigAtom[li, default: 0]
                guard count < maxHydroPerAtom else { break }

                result.append(MolecularInteraction(
                    id: idCounter, ligandAtomIndex: li, proteinAtomIndex: pi,
                    type: .hydrophobic, distance: d,
                    ligandPosition: lp, proteinPosition: protAtom.position))
                idCounter += 1
                hydroCountPerLigAtom[li] = count + 1
            }
        }

        return result
    }
}

import Foundation
import Metal
import simd

// MARK: - Virtual Screening Pipeline

/// High-throughput virtual screening: batch dock a library of ligands against a protein pocket.
/// Pipeline: RDKit 3D generation → Metal grid scoring → ML re-ranking → export hits.
@MainActor
final class VirtualScreeningPipeline: @unchecked Sendable {

    struct ScreeningConfig: Sendable {
        var maxMolecules: Int = 100_000
        var conformersPerMolecule: Int = 1
        var gridSpacing: Float = 0.5  // coarser for speed
        var populationSize: Int = 50
        var numGenerations: Int = 30
        var batchedPosesPerLigand: Int = 72
        var gaRefinementFraction: Float = 0.35
        var minimumGARefinementCount: Int = 128
        var gaRefinementMaxLigands: Int = 4_096
        var useMLReranking: Bool = true
        var mlRerankTopN: Int = 10_000
        var admetFilter: Bool = true
        var openMMRefineTopN: Int = 25
        var maxRotatableBonds: Int = 15 // skip very flexible molecules
    }

    struct ScreeningHit: Identifiable, Sendable {
        let id: Int
        var name: String
        var smiles: String
        var bestEnergy: Float
        var mlScore: Float?          // pKd from DruseScore
        var compositeScore: Float    // combined ranking score
        var bestPoseAtoms: [SIMD3<Float>]
        var ligandAtoms: [Atom]
        var ligandBonds: [Bond]
        var descriptors: LigandDescriptors?
        var prescreenEnergy: Float?
        var openMMEnergy: Float?
        var admet: ADMETPredictor.ADMETResult?
    }

    enum ScreeningState: Sendable {
        case idle
        case preparing(current: Int, total: Int)
        case docking(current: Int, total: Int)
        case reranking(current: Int, total: Int)
        case complete(hits: Int, total: Int)
        case failed(String)
    }

    private(set) var state: ScreeningState = .idle
    private(set) var hits: [ScreeningHit] = []
    private(set) var progress: Float = 0

    private var isCancelled = false

    var config = ScreeningConfig()

    private struct PrescoredEntry: Sendable {
        var index: Int
        var name: String
        var smiles: String
        var molecule: MoleculeData
        var descriptors: LigandDescriptors?
        var prescreen: BatchedDockingMetalAccelerator.PrescoreResult?
    }

    /// Run virtual screening on a SMILES library.
    func screen(
        smilesLibrary: [(name: String, smiles: String)],
        dockingEngine: DockingEngine,
        protein: Molecule,
        pocket: BindingPocket,
        mlScorer: DruseScoreInference?,
        admetPredictor: ADMETPredictor?
    ) async {
        isCancelled = false
        hits = []

        let total = min(smilesLibrary.count, config.maxMolecules)
        let library = Array(smilesLibrary.prefix(total))
        guard total > 0 else {
            state = .complete(hits: 0, total: 0)
            progress = 1.0
            return
        }

        // Phase 1: Prepare ligands (batch, TBB-parallel in C++)
        state = .preparing(current: 0, total: total)

        // Pre-filter by rotatable bonds
        var prepared: [(index: Int, name: String, smiles: String, molecule: MoleculeData, descriptors: LigandDescriptors?)] = []

        let batchSize = 256
        for batchStart in stride(from: 0, to: total, by: batchSize) {
            if isCancelled { state = .idle; return }

            let batchEnd = min(batchStart + batchSize, total)
            let batch = library[batchStart..<batchEnd]

            // Use TBB-parallel batch processing via C++
            let smilesArray = batch.map(\.smiles)
            let nameArray = batch.map(\.name)

            let results = await Task.detached {
                RDKitBridge.batchProcess(
                    entries: zip(smilesArray, nameArray).map { ($0, $1) },
                    addHydrogens: true, minimize: true, computeCharges: true
                )
            }.value

            for (i, result) in results.enumerated() {
                if let mol = result.molecule {
                    let idx = batchStart + i
                    let desc = RDKitBridge.computeDescriptors(smiles: library[idx].smiles)
                    if let d = desc, d.rotatableBonds <= config.maxRotatableBonds {
                        prepared.append((idx, library[idx].name, library[idx].smiles, mol, desc))
                    }
                }
            }

            state = .preparing(current: batchEnd, total: total)
            progress = Float(batchEnd) / Float(total) * 0.3
        }

        ActivityLog.shared.info("Prepared \(prepared.count)/\(total) molecules for screening", category: .dock)
        guard !prepared.isEmpty else {
            hits = []
            progress = 1.0
            state = .complete(hits: 0, total: total)
            ActivityLog.shared.warn("Screening library preparation produced no dockable molecules", category: .dock)
            return
        }

        // Phase 2: Dock each prepared molecule
        state = .docking(current: 0, total: prepared.count)

        dockingEngine.computeGridMaps(protein: protein, pocket: pocket, spacing: config.gridSpacing)

        let savedPoseUpdate = dockingEngine.onPoseUpdate
        let savedGenerationComplete = dockingEngine.onGenerationComplete
        let savedDockingComplete = dockingEngine.onDockingComplete
        dockingEngine.onPoseUpdate = nil
        dockingEngine.onGenerationComplete = nil
        dockingEngine.onDockingComplete = nil
        defer {
            dockingEngine.onPoseUpdate = savedPoseUpdate
            dockingEngine.onGenerationComplete = savedGenerationComplete
            dockingEngine.onDockingComplete = savedDockingComplete
        }

        let gridSnapshot = dockingEngine.gridSnapshot()
        let prescreenAccelerator = gridSnapshot.flatMap { BatchedDockingMetalAccelerator(device: $0.stericGridBuffer.device) }

        var prescoredEntries = prepared.map {
            PrescoredEntry(
                index: $0.index,
                name: $0.name,
                smiles: $0.smiles,
                molecule: $0.molecule,
                descriptors: $0.descriptors,
                prescreen: nil
            )
        }

        if let gridSnapshot, let prescreenAccelerator {
            prescoredEntries = await prescreenPreparedLigands(
                prescoredEntries,
                engine: dockingEngine,
                accelerator: prescreenAccelerator,
                gridSnapshot: gridSnapshot
            )
        }

        let shortlisted = shortlistForFlexibleDocking(from: prescoredEntries)
        ActivityLog.shared.info(
            "Batched Metal prescreen shortlisted \(shortlisted.count)/\(prescoredEntries.count) ligands for flexible docking",
            category: .dock
        )

        var allHits: [ScreeningHit] = []

        for (i, entry) in shortlisted.enumerated() {
            if isCancelled { state = .idle; return }

            let mol = Molecule(
                name: entry.name,
                atoms: entry.molecule.atoms,
                bonds: entry.molecule.bonds,
                title: entry.smiles
            )

            let result = await dockSingleMolecule(mol, engine: dockingEngine, protein: protein, pocket: pocket)
            let rigidFallback: ScreeningHit? = entry.prescreen.map { result in
                let preparedLigand = dockingEngine.prepareLigandGeometry(mol)
                return ScreeningHit(
                    id: allHits.count,
                    name: entry.name,
                    smiles: entry.smiles,
                    bestEnergy: result.energy,
                    mlScore: nil,
                    compositeScore: result.energy,
                    bestPoseAtoms: transformedPositions(
                        for: preparedLigand.heavyAtoms,
                        centroid: preparedLigand.centroid,
                        prescore: result
                    ),
                    ligandAtoms: preparedLigand.heavyAtoms,
                    ligandBonds: entry.molecule.bonds,
                    descriptors: entry.descriptors,
                    prescreenEnergy: result.energy,
                    openMMEnergy: nil,
                    admet: nil
                )
            }

            if let best = result {
                allHits.append(ScreeningHit(
                    id: allHits.count,
                    name: entry.name,
                    smiles: entry.smiles,
                    bestEnergy: best.energy,
                    mlScore: nil,
                    compositeScore: best.energy,
                    bestPoseAtoms: best.transformedAtomPositions,
                    ligandAtoms: entry.molecule.atoms.filter { $0.element != .H },
                    ligandBonds: entry.molecule.bonds,
                    descriptors: entry.descriptors,
                    prescreenEnergy: entry.prescreen?.energy,
                    openMMEnergy: nil,
                    admet: nil
                ))
            } else if let rigidFallback {
                allHits.append(rigidFallback)
            }

            if i % 10 == 0 {
                state = .docking(current: i, total: shortlisted.count)
                progress = 0.3 + Float(i) / Float(max(shortlisted.count, 1)) * 0.5
            }
        }

        allHits.sort { $0.bestEnergy < $1.bestEnergy }

        // Phase 3: ML re-ranking of top hits
        if config.useMLReranking, let scorer = mlScorer, scorer.isAvailable {
            let topN = min(config.mlRerankTopN, allHits.count)
            state = .reranking(current: 0, total: topN)

            let proteinAtoms = protein.atoms.filter { $0.element != .H }

            for i in 0..<topN {
                if isCancelled { state = .idle; return }

                var atoms = allHits[i].ligandAtoms
                for j in 0..<atoms.count {
                    if j < allHits[i].bestPoseAtoms.count {
                        atoms[j].position = allHits[i].bestPoseAtoms[j]
                    }
                }

                let features = DruseScoreFeatureExtractor.extract(
                    proteinAtoms: proteinAtoms,
                    ligandAtoms: atoms,
                    pocketCenter: pocket.center
                )

                if let pred = await scorer.score(features: features) {
                    allHits[i].mlScore = pred.pKd
                    allHits[i].compositeScore = 0.3 * allHits[i].bestEnergy + 0.7 * (-pred.pKd * 1.364)
                }

                if i % 100 == 0 {
                    state = .reranking(current: i, total: topN)
                    progress = 0.8 + Float(i) / Float(topN) * 0.15
                }
            }

            allHits.sort { $0.compositeScore < $1.compositeScore }
        }

        if config.openMMRefineTopN > 0 {
            allHits = await refineTopHitsWithOpenMM(
                hits: allHits,
                protein: protein,
                pocket: pocket
            )
        }

        // Phase 4: ADMET filtering
        if config.admetFilter, let admet = admetPredictor {
            for i in 0..<min(1000, allHits.count) {
                allHits[i].admet = await admet.predict(smiles: allHits[i].smiles)
            }
        }

        hits = allHits
        progress = 1.0
        state = .complete(hits: allHits.count, total: total)
        ActivityLog.shared.success("Screening complete: \(allHits.count) hits from \(total) molecules", category: .dock)
    }

    func cancel() {
        isCancelled = true
    }

    private func prescreenPreparedLigands(
        _ entries: [PrescoredEntry],
        engine: DockingEngine,
        accelerator: BatchedDockingMetalAccelerator,
        gridSnapshot: DockingGridSnapshot
    ) async -> [PrescoredEntry] {
        guard !entries.isEmpty else { return entries }

        var output = entries
        let batchSize = 128

        for batchStart in stride(from: 0, to: entries.count, by: batchSize) {
            if isCancelled { break }

            let batchEnd = min(batchStart + batchSize, entries.count)
            let preparedLigands = entries[batchStart..<batchEnd].map {
                engine.prepareLigandGeometry(
                    Molecule(name: $0.name, atoms: $0.molecule.atoms, bonds: $0.molecule.bonds, title: $0.smiles, smiles: $0.smiles)
                )
            }

            let results = accelerator.prescore(
                ligands: preparedLigands,
                gridSnapshot: gridSnapshot,
                posesPerLigand: max(config.batchedPosesPerLigand, 16)
            )

            for (offset, result) in results.enumerated() {
                output[batchStart + offset].prescreen = result
            }

            let completed = Float(batchEnd) / Float(entries.count)
            state = .docking(current: 0, total: entries.count)
            progress = 0.3 + completed * 0.1
            await Task.yield()
        }

        return output
    }

    private func shortlistForFlexibleDocking(from entries: [PrescoredEntry]) -> [PrescoredEntry] {
        guard !entries.isEmpty else { return [] }

        let sorted = entries.sorted {
            ($0.prescreen?.energy ?? .infinity) < ($1.prescreen?.energy ?? .infinity)
        }

        guard sorted.count > config.minimumGARefinementCount else { return sorted }

        let target = max(
            config.minimumGARefinementCount,
            Int(ceil(Float(sorted.count) * config.gaRefinementFraction))
        )
        let capped = min(sorted.count, max(1, min(target, config.gaRefinementMaxLigands)))
        return Array(sorted.prefix(capped))
    }

    private func refineTopHitsWithOpenMM(
        hits: [ScreeningHit],
        protein: Molecule,
        pocket: BindingPocket
    ) async -> [ScreeningHit] {
        guard !hits.isEmpty else { return hits }

        let refiner = OpenMMPocketRefiner.shared
        guard refiner.isAvailable else {
            ActivityLog.shared.info("OpenMM refinement skipped: \(refiner.availabilitySummary)", category: .dock)
            return hits
        }

        var refinedHits = hits
        let topN = min(config.openMMRefineTopN, refinedHits.count)
        ActivityLog.shared.info("Refining top \(topN) screening hits with OpenMM pocket minimization", category: .dock)

        for i in 0..<topN {
            if isCancelled { break }

            var ligandAtoms = refinedHits[i].ligandAtoms
            for j in 0..<min(ligandAtoms.count, refinedHits[i].bestPoseAtoms.count) {
                ligandAtoms[j].position = refinedHits[i].bestPoseAtoms[j]
            }

            if let result = await refiner.refine(
                proteinAtoms: protein.atoms,
                ligandAtoms: ligandAtoms,
                pocketCenter: pocket.center,
                pocketHalfExtent: pocket.size
            ) {
                refinedHits[i].openMMEnergy = result.interactionEnergyKcal
                refinedHits[i].compositeScore = 0.5 * refinedHits[i].compositeScore + 0.5 * result.interactionEnergyKcal
            }
        }

        refinedHits.sort { $0.compositeScore < $1.compositeScore }
        return refinedHits
    }

    /// Dock a single molecule with reduced parameters (for speed).
    private func dockSingleMolecule(
        _ ligand: Molecule,
        engine: DockingEngine,
        protein: Molecule,
        pocket: BindingPocket
    ) async -> DockingResult? {
        var dockConfig = DockingConfig()
        dockConfig.populationSize = max(20, config.populationSize)
        dockConfig.numRuns = 1
        dockConfig.generationsPerRun = max(10, config.numGenerations)
        dockConfig.gridSpacing = config.gridSpacing
        dockConfig.enableFlexibility = true
        dockConfig.liveUpdateFrequency = max(config.numGenerations + 1, 999)
        dockConfig.localSearchFrequency = max(2, min(5, dockConfig.generationsPerRun / 4))

        let results = await engine.runDocking(ligand: ligand, pocket: pocket, config: dockConfig)
        return results.first
    }

    /// Export screening results as CSV.
    func exportCSV() -> String {
        var csv = "Rank,Name,SMILES,Energy,PrescreenEnergy,OpenMMEnergy,ML_Score,Composite,MW,LogP,HBD,HBA,TPSA,RotBonds,Lipinski,DrugLikeness\n"
        for (i, hit) in hits.enumerated() {
            let desc = hit.descriptors
            csv += "\(i+1),\"\(hit.name)\",\"\(hit.smiles)\",\(String(format: "%.2f", hit.bestEnergy)),"
            csv += "\(hit.prescreenEnergy.map { String(format: "%.2f", $0) } ?? "N/A"),"
            csv += "\(hit.openMMEnergy.map { String(format: "%.2f", $0) } ?? "N/A"),"
            csv += "\(hit.mlScore.map { String(format: "%.2f", $0) } ?? "N/A"),"
            csv += "\(String(format: "%.2f", hit.compositeScore)),"
            csv += "\(desc.map { String(format: "%.1f", $0.molecularWeight) } ?? "N/A"),"
            csv += "\(desc.map { String(format: "%.2f", $0.logP) } ?? "N/A"),"
            csv += "\(desc?.hbd ?? 0),\(desc?.hba ?? 0),"
            csv += "\(desc.map { String(format: "%.1f", $0.tpsa) } ?? "N/A"),"
            csv += "\(desc?.rotatableBonds ?? 0),"
            csv += "\(desc?.lipinski == true ? "Yes" : "No"),"
            csv += "\(hit.admet.map { String(format: "%.2f", $0.drugLikeness) } ?? "N/A")\n"
        }
        return csv
    }

    /// Export top hits as multi-molecule SDF with scores.
    func exportSDF(topN: Int = 100) -> String {
        let molecules = hits.prefix(topN).map { hit -> (name: String, atoms: [Atom], bonds: [Bond], properties: [String: String]) in
            var dockedAtoms = hit.ligandAtoms
            for index in 0..<min(dockedAtoms.count, hit.bestPoseAtoms.count) {
                dockedAtoms[index].position = hit.bestPoseAtoms[index]
            }

            var properties: [String: String] = [
                "Name": hit.name,
                "SMILES": hit.smiles,
                "Energy": String(format: "%.3f", hit.bestEnergy),
                "CompositeScore": String(format: "%.3f", hit.compositeScore)
            ]
            if let prescreenEnergy = hit.prescreenEnergy {
                properties["PrescreenEnergy"] = String(format: "%.3f", prescreenEnergy)
            }
            if let openMMEnergy = hit.openMMEnergy {
                properties["OpenMMEnergy"] = String(format: "%.3f", openMMEnergy)
            }
            if let ml = hit.mlScore {
                properties["ML_Score"] = String(format: "%.3f", ml)
            }
            return (hit.name, dockedAtoms, hit.ligandBonds, properties)
        }
        return SDFWriter.write(molecules: molecules)
    }
}

private func transformedPositions(
    for heavyAtoms: [Atom],
    centroid: SIMD3<Float>,
    prescore: BatchedDockingMetalAccelerator.PrescoreResult
) -> [SIMD3<Float>] {
    heavyAtoms.map { atom in
        prescore.rotation.act(atom.position - centroid) + prescore.translation
    }
}

func applyingRefinedHeavyAtomPositions(
    _ refinedPositions: [SIMD3<Float>],
    to proteinAtoms: [Atom]
) -> [Atom]? {
    var updated = proteinAtoms
    var heavyIndex = 0
    for index in updated.indices where updated[index].element != .H {
        guard heavyIndex < refinedPositions.count else { break }
        updated[index].position = refinedPositions[heavyIndex]
        heavyIndex += 1
    }
    return heavyIndex == refinedPositions.count ? updated : nil
}

final class BatchedDockingMetalAccelerator {
    struct PrescoreResult: Sendable {
        var energy: Float
        var translation: SIMD3<Float>
        var rotation: simd_quatf
        var stericEnergy: Float
        var hydrophobicEnergy: Float
        var hbondEnergy: Float
        var clashPenalty: Float
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let initializePipeline: MTLComputePipelineState
    private let scorePipeline: MTLComputePipelineState

    init?(device: MTLDevice) {
        self.device = device
        guard let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let initializeFunction = library.makeFunction(name: "initializeBatchScreenPoses"),
              let scoreFunction = library.makeFunction(name: "scoreBatchRigidPoses")
        else {
            return nil
        }

        self.commandQueue = commandQueue
        do {
            initializePipeline = try device.makeComputePipelineState(function: initializeFunction)
            scorePipeline = try device.makeComputePipelineState(function: scoreFunction)
        } catch {
            return nil
        }
    }

    func prescore(
        ligands: [PreparedDockingLigand],
        gridSnapshot: DockingGridSnapshot,
        posesPerLigand: Int
    ) -> [PrescoreResult?] {
        guard !ligands.isEmpty else { return [] }

        let clampedPosesPerLigand = max(posesPerLigand, 8)
        var ligandInfos: [BatchLigandInfo] = []
        var atomStorage: [DockLigandAtom] = []
        var sourceIndices: [Int] = []
        var results = [PrescoreResult?](repeating: nil, count: ligands.count)

        var poseStart: UInt32 = 0
        for (index, ligand) in ligands.enumerated() {
            let atomCount = ligand.gpuAtoms.count
            guard atomCount > 0, atomCount <= 128 else { continue }
            ligandInfos.append(BatchLigandInfo(
                atomStart: UInt32(atomStorage.count),
                atomCount: UInt32(atomCount),
                poseStart: poseStart,
                poseCount: UInt32(clampedPosesPerLigand)
            ))
            atomStorage.append(contentsOf: ligand.gpuAtoms)
            sourceIndices.append(index)
            poseStart += UInt32(clampedPosesPerLigand)
        }

        guard !ligandInfos.isEmpty,
              let ligandInfoBuffer = device.makeBuffer(
                bytes: ligandInfos,
                length: ligandInfos.count * MemoryLayout<BatchLigandInfo>.stride,
                options: .storageModeShared
              ),
              let ligandAtomBuffer = device.makeBuffer(
                bytes: atomStorage,
                length: atomStorage.count * MemoryLayout<DockLigandAtom>.stride,
                options: .storageModeShared
              )
        else {
            return results
        }

        let totalPoses = Int(poseStart)
        guard totalPoses > 0 else { return results }

        var params = BatchScreenParams(
            totalPoses: UInt32(totalPoses),
            posesPerLigand: UInt32(clampedPosesPerLigand),
            numLigands: UInt32(ligandInfos.count),
            seed: UInt32(Date().timeIntervalSinceReferenceDate.bitPattern & 0xffff_ffff)
        )

        guard let paramsBuffer = device.makeBuffer(
                bytes: &params,
                length: MemoryLayout<BatchScreenParams>.stride,
                options: .storageModeShared
              ),
              let poseBuffer = device.makeBuffer(
                length: totalPoses * MemoryLayout<BatchScreenPose>.stride,
                options: .storageModeShared
              ),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else {
            return results
        }

        let threadWidth = 256
        let threadgroups = MTLSize(width: (totalPoses + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: threadWidth, height: 1, depth: 1)

        guard let initializeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return results
        }
        initializeEncoder.setComputePipelineState(initializePipeline)
        initializeEncoder.setBuffer(poseBuffer, offset: 0, index: 0)
        initializeEncoder.setBuffer(ligandInfoBuffer, offset: 0, index: 1)
        initializeEncoder.setBuffer(gridSnapshot.gridParamsBuffer, offset: 0, index: 2)
        initializeEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)
        initializeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        initializeEncoder.endEncoding()

        guard let scoreEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return results
        }
        scoreEncoder.setComputePipelineState(scorePipeline)
        scoreEncoder.setBuffer(poseBuffer, offset: 0, index: 0)
        scoreEncoder.setBuffer(ligandInfoBuffer, offset: 0, index: 1)
        scoreEncoder.setBuffer(ligandAtomBuffer, offset: 0, index: 2)
        scoreEncoder.setBuffer(gridSnapshot.stericGridBuffer, offset: 0, index: 3)
        scoreEncoder.setBuffer(gridSnapshot.hydrophobicGridBuffer, offset: 0, index: 4)
        scoreEncoder.setBuffer(gridSnapshot.hbondGridBuffer, offset: 0, index: 5)
        scoreEncoder.setBuffer(gridSnapshot.gridParamsBuffer, offset: 0, index: 6)
        scoreEncoder.setBuffer(paramsBuffer, offset: 0, index: 7)
        scoreEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        scoreEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let poses = poseBuffer.contents().bindMemory(to: BatchScreenPose.self, capacity: totalPoses)
        var bestByLigand = [BatchScreenPose?](repeating: nil, count: ligandInfos.count)
        for poseIndex in 0..<totalPoses {
            let pose = poses[poseIndex]
            let ligandIndex = Int(pose.ligandIndex)
            guard ligandIndex < bestByLigand.count else { continue }
            if let best = bestByLigand[ligandIndex], best.energy <= pose.energy {
                continue
            }
            bestByLigand[ligandIndex] = pose
        }

        let gridSize = SIMD3<Float>(
            Float(gridSnapshot.gridParams.dims.x),
            Float(gridSnapshot.gridParams.dims.y),
            Float(gridSnapshot.gridParams.dims.z)
        ) * gridSnapshot.gridParams.spacing
        let fallbackTranslation = gridSnapshot.gridParams.origin + gridSize * 0.5

        for (compressedIndex, sourceIndex) in sourceIndices.enumerated() {
            if let best = bestByLigand[compressedIndex] {
                results[sourceIndex] = PrescoreResult(
                    energy: best.energy,
                    translation: best.translation,
                    rotation: simd_quatf(ix: best.rotation.x, iy: best.rotation.y, iz: best.rotation.z, r: best.rotation.w),
                    stericEnergy: best.stericEnergy,
                    hydrophobicEnergy: best.hydrophobicEnergy,
                    hbondEnergy: best.hbondEnergy,
                    clashPenalty: best.clashPenalty
                )
            } else {
                results[sourceIndex] = PrescoreResult(
                    energy: 1_000_000,
                    translation: fallbackTranslation,
                    rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)),
                    stericEnergy: 0,
                    hydrophobicEnergy: 0,
                    hbondEnergy: 0,
                    clashPenalty: 1_000_000
                )
            }
        }

        return results
    }
}

final class OpenMMPocketRefiner {
    struct RefinementResult: Sendable {
        var interactionEnergyKcal: Float
        var refinedHeavyAtomPositions: [SIMD3<Float>]
    }

    struct Availability: Sendable {
        var pythonAvailable: Bool
        var openMMAvailable: Bool
        var message: String
    }

    private struct PythonCommand: Sendable {
        var executable: String
        var prefixArguments: [String]
    }

    private struct ProteinHeavyKey: Codable, Sendable {
        var name: String
        var residueName: String
        var residueSeq: Int
        var chainID: String
    }

    private struct LigandSite: Codable, Sendable {
        var position: [Float]
        var charge: Float
        var sigmaNm: Float
        var epsilonKJ: Float
    }

    private struct Payload: Codable, Sendable {
        var proteinPDB: String
        var proteinHeavyKeys: [ProteinHeavyKey]
        var ligandSites: [LigandSite]
        var pocketCenter: [Float]
        var pocketHalfExtent: [Float]
        var pocketRadiusAngstrom: Float
        var maxIterations: Int
    }

    private struct Response: Decodable {
        var interactionEnergyKcal: Float
        var refinedHeavyPositions: [[Float]]
    }

    nonisolated(unsafe) static let shared = OpenMMPocketRefiner()

    let availability: Availability
    var isAvailable: Bool { availability.openMMAvailable }
    var availabilitySummary: String { availability.message }

    private let command: PythonCommand?

    private init() {
        let detected = Self.detectAvailability()
        self.command = detected.command
        self.availability = detected.availability
    }

    func refine(
        proteinAtoms: [Atom],
        ligandAtoms: [Atom],
        pocketCenter: SIMD3<Float>,
        pocketHalfExtent: SIMD3<Float>
    ) async -> RefinementResult? {
        guard let command else { return nil }

        let heavyProteinAtoms = proteinAtoms.filter { $0.element != .H }
        let heavyLigandAtoms = ligandAtoms.filter { $0.element != .H }
        guard !heavyProteinAtoms.isEmpty, !heavyLigandAtoms.isEmpty else { return nil }

        let payload = Payload(
            proteinPDB: Self.pdbString(from: proteinAtoms),
            proteinHeavyKeys: heavyProteinAtoms.map {
                ProteinHeavyKey(
                    name: $0.name,
                    residueName: $0.residueName,
                    residueSeq: $0.residueSeq,
                    chainID: $0.chainID
                )
            },
            ligandSites: heavyLigandAtoms.map {
                LigandSite(
                    position: [$0.position.x / 10.0, $0.position.y / 10.0, $0.position.z / 10.0],
                    charge: abs($0.charge) > 0.0001 ? $0.charge : Float($0.formalCharge),
                    sigmaNm: Self.ljSigmaNm(for: $0),
                    epsilonKJ: Self.ljEpsilonKJ(for: $0)
                )
            },
            pocketCenter: [pocketCenter.x / 10.0, pocketCenter.y / 10.0, pocketCenter.z / 10.0],
            pocketHalfExtent: [pocketHalfExtent.x / 10.0, pocketHalfExtent.y / 10.0, pocketHalfExtent.z / 10.0],
            pocketRadiusAngstrom: max(pocketHalfExtent.x, max(pocketHalfExtent.y, pocketHalfExtent.z)) + 4.0,
            maxIterations: 250
        )

        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: Self.runRefinement(command: command, payload: payload))
            }
        }
    }

    private static func detectAvailability() -> (command: PythonCommand?, availability: Availability) {
        let command = pythonCommand()
        guard let command else {
            return (nil, Availability(pythonAvailable: false, openMMAvailable: false, message: "python3 not found"))
        }

        let pythonProbe = run(command: command, arguments: ["-c", "import sys; print(sys.executable)"])
        guard pythonProbe.status == 0 else {
            return (nil, Availability(pythonAvailable: false, openMMAvailable: false, message: "python3 could not be launched"))
        }

        let openMMProbe = run(
            command: command,
            arguments: ["-c", "import openmm, openmm.app, openmm.unit; print('openmm-ok')"]
        )
        if openMMProbe.status == 0, openMMProbe.stdout.contains("openmm-ok") {
            return (command, Availability(pythonAvailable: true, openMMAvailable: true, message: "OpenMM available"))
        }

        let stderr = openMMProbe.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = stderr.isEmpty ? "python3 is available, but OpenMM is not installed" : stderr
        return (command, Availability(pythonAvailable: true, openMMAvailable: false, message: detail))
    }

    private static func pythonCommand() -> PythonCommand? {
        if let explicit = ProcessInfo.processInfo.environment["DRUSE_OPENMM_PYTHON"], !explicit.isEmpty {
            return PythonCommand(executable: explicit, prefixArguments: [])
        }
        return PythonCommand(executable: "/usr/bin/env", prefixArguments: ["python3"])
    }

    private static func runRefinement(command: PythonCommand, payload: Payload) -> RefinementResult? {
        guard let scriptPath = try? ensureScriptPath(),
              let input = try? JSONEncoder().encode(payload)
        else {
            return nil
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: command.executable)
        process.arguments = command.prefixArguments + [scriptPath]

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            return nil
        }

        stdinPipe.fileHandleForWriting.write(input)
        try? stdinPipe.fileHandleForWriting.close()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            return nil
        }

        guard let response = try? JSONDecoder().decode(Response.self, from: stdoutPipe.fileHandleForReading.readDataToEndOfFile()) else {
            return nil
        }

        return RefinementResult(
            interactionEnergyKcal: response.interactionEnergyKcal,
            refinedHeavyAtomPositions: response.refinedHeavyPositions.compactMap {
                guard $0.count == 3 else { return nil }
                return SIMD3<Float>($0[0], $0[1], $0[2])
            }
        )
    }

    private static func run(command: PythonCommand, arguments: [String]) -> (status: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command.executable)
        process.arguments = command.prefixArguments + arguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return (1, "", error.localizedDescription)
        }

        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return (process.terminationStatus, stdout, stderr)
    }

    private static func ensureScriptPath() throws -> String {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("druse_openmm_refine.py")
        if !FileManager.default.fileExists(atPath: url.path) {
            try script.write(to: url, atomically: true, encoding: .utf8)
        }
        return url.path
    }

    private static func pdbString(from atoms: [Atom]) -> String {
        var lines: [String] = []
        lines.reserveCapacity(atoms.count + 2)

        for (serial, atom) in atoms.enumerated() {
            let record = atom.isHetAtom ? "HETATM" : "ATOM"
            let atomName = pdbAtomName(atom.name, element: atom.element.symbol)
            let residueName = padded(atom.residueName.isEmpty ? "UNK" : String(atom.residueName.prefix(3)), width: 3, left: true)
            let chainID = String((atom.chainID.isEmpty ? "A" : atom.chainID).prefix(1))
            let resSeq = padded(String(atom.residueSeq), width: 4, left: false)
            let x = String(format: "%8.3f", atom.position.x)
            let y = String(format: "%8.3f", atom.position.y)
            let z = String(format: "%8.3f", atom.position.z)
            let occ = String(format: "%6.2f", atom.occupancy > 0 ? atom.occupancy : 1.0)
            let temp = String(format: "%6.2f", atom.tempFactor)
            let element = padded(atom.element.symbol, width: 2, left: false)

            lines.append(
                "\(padded(record, width: 6, left: true))\(padded(String(serial + 1), width: 5, left: false)) " +
                "\(atomName) \(residueName) \(chainID)\(resSeq)    \(x)\(y)\(z)\(occ)\(temp)          \(element)"
            )
        }

        lines.append("END")
        return lines.joined(separator: "\n")
    }

    private static func pdbAtomName(_ name: String, element: String) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = trimmed.isEmpty ? element : trimmed
        if base.count >= 4 {
            return String(base.prefix(4))
        }
        if element.count == 1 {
            return padded(base, width: 4, left: false)
        }
        return padded(base, width: 4, left: true)
    }

    private static func padded(_ string: String, width: Int, left: Bool) -> String {
        if string.count >= width {
            return String(string.prefix(width))
        }
        let padding = String(repeating: " ", count: width - string.count)
        return left ? string + padding : padding + string
    }

    private static func ljSigmaNm(for atom: Atom) -> Float {
        let sigmaAngstrom = max(1.2, (atom.element.vdwRadius * 2.0) / 1.122462)
        return sigmaAngstrom / 10.0
    }

    private static func ljEpsilonKJ(for atom: Atom) -> Float {
        switch atom.element {
        case .C:  return 0.45
        case .N:  return 0.30
        case .O:  return 0.35
        case .S, .P: return 0.75
        case .F, .Cl, .Br: return 0.65
        case .Mg, .Ca, .Mn, .Fe, .Zn, .Cu, .Ni, .Co: return 0.15
        default:  return 0.25
        }
    }

    private static let script = #"""
import io
import json
import sys

try:
    import openmm as mm
    from openmm import app, unit
except Exception as exc:
    sys.stderr.write(f"OpenMM import failed: {exc}\n")
    sys.exit(2)


def residue_key(residue):
    return (getattr(residue.chain, "id", ""), str(getattr(residue, "id", "")), getattr(residue, "name", ""))


def vec3_nm(position):
    value = position.value_in_unit(unit.nanometer)
    return (float(value[0]), float(value[1]), float(value[2]))


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def make_forcefield():
    candidates = [
        ("amber14-all.xml", "amber14/tip3pfb.xml"),
        ("amber14-all.xml",),
        ("amber99sb.xml",),
    ]
    last_error = None
    for spec in candidates:
        try:
            return app.ForceField(*spec)
        except Exception as exc:
            last_error = exc
    raise last_error


def match_heavy_atom_indices(topology, keys):
    heavy_atoms = []
    for index, atom in enumerate(topology.atoms()):
        element = getattr(atom, "element", None)
        if element is None or getattr(element, "symbol", "") == "H":
            continue
        heavy_atoms.append((index, atom))

    matched = []
    cursor = 0
    for key in keys:
        found = None
        for probe in range(cursor, len(heavy_atoms)):
            index, atom = heavy_atoms[probe]
            if (
                atom.name == key["name"]
                and atom.residue.name == key["residueName"]
                and getattr(atom.residue.chain, "id", "") == key["chainID"]
                and str(getattr(atom.residue, "id", "")) == str(key["residueSeq"])
            ):
                found = index
                cursor = probe + 1
                break
        if found is None:
            raise RuntimeError(f"Failed to map heavy atom {key}")
        matched.append(found)
    return matched


def collect_pocket_residues(topology, positions_nm, ligand_sites_nm, cutoff_nm):
    cutoff2 = cutoff_nm * cutoff_nm
    pocket_residues = set()
    for atom_index, atom in enumerate(topology.atoms()):
        atom_xyz = positions_nm[atom_index]
        for ligand_xyz in ligand_sites_nm:
            if dist2(atom_xyz, ligand_xyz) <= cutoff2:
                pocket_residues.add(residue_key(atom.residue))
                break
    return pocket_residues


def platform():
    for name in ("CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


payload = json.load(sys.stdin)
pdb = app.PDBFile(io.StringIO(payload["proteinPDB"]))
forcefield = make_forcefield()
modeller = app.Modeller(pdb.topology, pdb.positions)

try:
    modeller.addHydrogens(forcefield)
except Exception:
    pass

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds
)

protein_particle_count = system.getNumParticles()
all_positions = list(modeller.positions)
positions_nm = [vec3_nm(pos) for pos in all_positions]
ligand_sites_nm = [tuple(site["position"]) for site in payload["ligandSites"]]
heavy_atom_indices = match_heavy_atom_indices(modeller.topology, payload["proteinHeavyKeys"])

nonbonded = None
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        nonbonded = force
        break
if nonbonded is None:
    raise RuntimeError("Protein system missing NonbondedForce")

pocket_cutoff_nm = payload["pocketRadiusAngstrom"] / 10.0
pocket_residues = collect_pocket_residues(modeller.topology, positions_nm, ligand_sites_nm, pocket_cutoff_nm)

restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
restraint.addPerParticleParameter("k")
restraint.addPerParticleParameter("x0")
restraint.addPerParticleParameter("y0")
restraint.addPerParticleParameter("z0")
restraint.setForceGroup(30)

topology_atoms = list(modeller.topology.atoms())
for index, atom in enumerate(topology_atoms):
    xyz = positions_nm[index]
    is_pocket = residue_key(atom.residue) in pocket_residues
    k_value = 600.0 if is_pocket else 8000.0
    restraint.addParticle(index, [k_value, xyz[0], xyz[1], xyz[2]])
system.addForce(restraint)

interaction = mm.CustomNonbondedForce(
    "138.935456*charge1*charge2/r + "
    "4*sqrt(epsilon1*epsilon2)*(pow((0.5*(sigma1+sigma2))/r, 12) - pow((0.5*(sigma1+sigma2))/r, 6))"
)
interaction.addPerParticleParameter("charge")
interaction.addPerParticleParameter("sigma")
interaction.addPerParticleParameter("epsilon")
interaction.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
interaction.setForceGroup(31)

for index in range(protein_particle_count):
    charge, sigma, epsilon = nonbonded.getParticleParameters(index)
    interaction.addParticle([
        float(charge.value_in_unit(unit.elementary_charge)),
        float(sigma.value_in_unit(unit.nanometer)),
        float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
    ])

ligand_indices = []
for site in payload["ligandSites"]:
    ligand_index = system.addParticle(0.0)
    ligand_indices.append(ligand_index)
    interaction.addParticle([
        float(site["charge"]),
        float(site["sigmaNm"]),
        float(site["epsilonKJ"]),
    ])
    all_positions.append(mm.Vec3(*site["position"]) * unit.nanometer)

interaction.addInteractionGroup(set(range(protein_particle_count)), set(ligand_indices))
system.addForce(interaction)

integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
selected_platform = platform()
if selected_platform is None:
    simulation = app.Simulation(modeller.topology, system, integrator)
else:
    simulation = app.Simulation(modeller.topology, system, integrator, selected_platform)

simulation.context.setPositions(all_positions)
simulation.minimizeEnergy(maxIterations=int(payload["maxIterations"]))

interaction_state = simulation.context.getState(getEnergy=True, groups=1 << 31)
position_state = simulation.context.getState(getPositions=True)

interaction_kj = interaction_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
final_positions = position_state.getPositions()
heavy_positions = []
for index in heavy_atom_indices:
    xyz = vec3_nm(final_positions[index])
    heavy_positions.append([xyz[0] * 10.0, xyz[1] * 10.0, xyz[2] * 10.0])

json.dump(
    {
        "interactionEnergyKcal": float(interaction_kj) * 0.239005736,
        "refinedHeavyPositions": heavy_positions,
    },
    sys.stdout,
)
"""#
}

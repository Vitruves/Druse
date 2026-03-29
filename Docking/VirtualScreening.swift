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
        var admetFilter: Bool = true
        var openMMRefineTopN: Int = 25
        var gfn2RefineTopN: Int = 0  // 0 = disabled; >0 = refine top N hits with GFN2-xTB (D4 + solvation)
        var applyStrainPenalty: Bool = false
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
        var strainEnergy: Float?
        var gfn2Energy: Float?            // GFN2-xTB total energy (kcal/mol)
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

    /// Cancellation token backed by heap-allocated Int32 so it can be shared
    /// across Swift async tasks AND passed to C++ TBB parallel_for.
    /// Uses atomic load/store via C helpers to avoid data races.
    private let cancelToken = CancellationToken()

    private var isCancelled: Bool { cancelToken.value }

    final class CancellationToken: @unchecked Sendable {
        private let ptr: UnsafeMutablePointer<Int32>

        init() {
            ptr = .allocate(capacity: 1)
            ptr.initialize(to: 0)
        }

        deinit { ptr.deallocate() }

        var value: Bool { druse_atomic_cancel_load(ptr) != 0 }

        func cancel() { druse_atomic_cancel_store(ptr, 1) }

        func reset() { druse_atomic_cancel_store(ptr, 0) }

        /// Raw pointer for passing to C functions (atomic cancel flag).
        var unsafePointer: UnsafePointer<Int32> { UnsafePointer(ptr) }
    }

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
        admetPredictor: ADMETPredictor?,
        constraints: [PharmacophoreConstraintDef] = []
    ) async {
        cancelToken.reset()
        hits = []
        let screenStartTime = CFAbsoluteTimeGetCurrent()
        ActivityLog.shared.info("Virtual screening started: \(smilesLibrary.count) input molecules", category: .dock)

        // Prepare pharmacophore constraint buffers once (shared across all ligands)
        let activeConstraints = constraints.filter(\.isEnabled)
        dockingEngine.prepareConstraintBuffers(activeConstraints, atoms: protein.atoms, residues: protein.residues)

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

            let cancelPtr = cancelToken.unsafePointer
            let results = await Task.detached {
                RDKitBridge.batchProcess(
                    entries: zip(smilesArray, nameArray).map { ($0, $1) },
                    addHydrogens: true, minimize: true, computeCharges: true,
                    cancelFlag: cancelPtr
                )
            }.value

            var batchRejectedParse = 0
            var batchRejectedRotBonds = 0
            for (i, result) in results.enumerated() {
                if let mol = result.molecule {
                    let idx = batchStart + i
                    let desc = RDKitBridge.computeDescriptors(smiles: library[idx].smiles)
                    if let d = desc, d.rotatableBonds <= config.maxRotatableBonds {
                        prepared.append((idx, library[idx].name, library[idx].smiles, mol, desc))
                    } else {
                        batchRejectedRotBonds += 1
                    }
                } else {
                    batchRejectedParse += 1
                }
            }
            if batchRejectedParse > 0 || batchRejectedRotBonds > 0 {
                ActivityLog.shared.debug("Batch \(batchStart/batchSize): rejected \(batchRejectedParse) parse failures, \(batchRejectedRotBonds) exceeded \(config.maxRotatableBonds) rotatable bonds", category: .dock)
            }

            state = .preparing(current: batchEnd, total: total)
            progress = Float(batchEnd) / Float(total) * 0.3
        }

        let rejected = total - prepared.count
        ActivityLog.shared.info("Prepared \(prepared.count)/\(total) molecules for screening (\(rejected) rejected)", category: .dock)
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

        // Phase 2: Batched GA docking — process multiple ligands simultaneously on GPU
        let batchedGA = gridSnapshot.flatMap { snap in
            snap.vinaAffinityGridBuffer != nil ? BatchedGAAccelerator(device: snap.stericGridBuffer.device) : nil
        }

        if let batchedGA, let snap = gridSnapshot {
            // Prepare all ligands for batched GA
            let batchSize = 32  // ligands per GPU batch
            for batchStart in stride(from: 0, to: shortlisted.count, by: batchSize) {
                if isCancelled { state = .idle; return }
                let batchEnd = min(batchStart + batchSize, shortlisted.count)
                let batch = Array(shortlisted[batchStart..<batchEnd])

                var gaData = [LigandGAData?]()
                var molecules = [Molecule]()
                for entry in batch {
                    let mol = Molecule(name: entry.name, atoms: entry.molecule.atoms, bonds: entry.molecule.bonds, title: entry.smiles)
                    molecules.append(mol)
                    gaData.append(dockingEngine.prepareLigandGA(mol))
                }

                let validData = gaData.compactMap { $0 }
                let validIndices = gaData.enumerated().compactMap { $0.element != nil ? $0.offset : nil }

                let gaResults = batchedGA.runBatchedGA(
                    ligands: validData, gridSnapshot: snap,
                    populationSize: max(20, config.populationSize),
                    generations: max(10, config.numGenerations),
                    localSearchSteps: 8
                )

                for (resultIdx, localIdx) in validIndices.enumerated() {
                    let entry = batch[localIdx]
                    if let gaResult = gaResults[resultIdx] {
                        // Transform atoms using the GA result pose
                        let data = validData[resultIdx]
                        let positions = data.gpuAtoms.map { atom in
                            gaResult.rotation.act(atom.position) + gaResult.translation
                        }
                        allHits.append(ScreeningHit(
                            id: allHits.count, name: entry.name, smiles: entry.smiles,
                            bestEnergy: gaResult.energy, mlScore: nil, compositeScore: gaResult.energy,
                            bestPoseAtoms: positions,
                            ligandAtoms: data.heavyAtoms, ligandBonds: entry.molecule.bonds,
                            descriptors: entry.descriptors, prescreenEnergy: entry.prescreen?.energy,
                            openMMEnergy: nil, admet: nil
                        ))
                    } else if let prescore = entry.prescreen {
                        let prepLig = dockingEngine.prepareLigandGeometry(molecules[localIdx])
                        allHits.append(ScreeningHit(
                            id: allHits.count, name: entry.name, smiles: entry.smiles,
                            bestEnergy: prescore.energy, mlScore: nil, compositeScore: prescore.energy,
                            bestPoseAtoms: transformedPositions(for: prepLig.heavyAtoms, centroid: prepLig.centroid, prescore: prescore),
                            ligandAtoms: prepLig.heavyAtoms, ligandBonds: entry.molecule.bonds,
                            descriptors: entry.descriptors, prescreenEnergy: prescore.energy,
                            openMMEnergy: nil, admet: nil
                        ))
                    }
                }

                state = .docking(current: min(batchEnd, shortlisted.count), total: shortlisted.count)
                progress = 0.3 + Float(batchEnd) / Float(max(shortlisted.count, 1)) * 0.5
            }
        } else {
            // Fallback: sequential GA docking per ligand
            for (i, entry) in shortlisted.enumerated() {
                if isCancelled { state = .idle; return }
                let mol = Molecule(name: entry.name, atoms: entry.molecule.atoms, bonds: entry.molecule.bonds, title: entry.smiles)
                let result = await dockSingleMolecule(mol, engine: dockingEngine, protein: protein, pocket: pocket)

                if let best = result {
                    allHits.append(ScreeningHit(
                        id: allHits.count, name: entry.name, smiles: entry.smiles,
                        bestEnergy: best.energy, mlScore: nil, compositeScore: best.energy,
                        bestPoseAtoms: best.transformedAtomPositions,
                        ligandAtoms: entry.molecule.atoms.filter { $0.element != .H },
                        ligandBonds: entry.molecule.bonds, descriptors: entry.descriptors,
                        prescreenEnergy: entry.prescreen?.energy, openMMEnergy: nil, admet: nil
                    ))
                } else if let prescore = entry.prescreen {
                    let prepLig = dockingEngine.prepareLigandGeometry(mol)
                    allHits.append(ScreeningHit(
                        id: allHits.count, name: entry.name, smiles: entry.smiles,
                        bestEnergy: prescore.energy, mlScore: nil, compositeScore: prescore.energy,
                        bestPoseAtoms: transformedPositions(for: prepLig.heavyAtoms, centroid: prepLig.centroid, prescore: prescore),
                        ligandAtoms: prepLig.heavyAtoms, ligandBonds: entry.molecule.bonds,
                        descriptors: entry.descriptors, prescreenEnergy: prescore.energy,
                        openMMEnergy: nil, admet: nil
                    ))
                }
                if i % 10 == 0 {
                    state = .docking(current: i, total: shortlisted.count)
                    progress = 0.3 + Float(i) / Float(max(shortlisted.count, 1)) * 0.5
                }
            }
        }

        // Keep Vina screening scorer-pure by default; strain is optional product-side post-processing.
        if config.applyStrainPenalty {
            for i in 0..<allHits.count {
                if isCancelled { state = .idle; return }
                let hit = allHits[i]
                guard !hit.bestPoseAtoms.isEmpty else { continue }
                if let refE = RDKitBridge.mmffReferenceEnergy(smiles: hit.smiles),
                   let dockedE = RDKitBridge.mmffStrainEnergy(smiles: hit.smiles, heavyPositions: hit.bestPoseAtoms) {
                    let strain = Float(dockedE - refE)
                    allHits[i].strainEnergy = strain
                    if strain > 6.0 {
                        allHits[i].bestEnergy += 0.5 * (strain - 6.0)
                        allHits[i].compositeScore = allHits[i].bestEnergy
                    }
                }
            }
        }

        allHits.sort { $0.bestEnergy < $1.bestEnergy }

        if config.openMMRefineTopN > 0 {
            allHits = await refineTopHitsWithOpenMM(
                hits: allHits,
                protein: protein,
                pocket: pocket
            )
        }

        // GFN2-xTB refinement of top screening hits (optional, ~15ms/hit)
        if config.gfn2RefineTopN > 0 {
            let topN = min(config.gfn2RefineTopN, allHits.count)
            ActivityLog.shared.info("GFN2-xTB refining top \(topN) screening hits...", category: .dock)
            for i in 0..<topN {
                let atoms = allHits[i].ligandAtoms
                let positions = allHits[i].bestPoseAtoms
                guard atoms.count >= 2, positions.count == atoms.count else { continue }
                var dockedAtoms = atoms
                for j in 0..<dockedAtoms.count { dockedAtoms[j].position = positions[j] }
                let charge = atoms.reduce(0) { $0 + $1.formalCharge }
                if let result = try? await GFN2Refiner.computeEnergy(
                    atoms: dockedAtoms, totalCharge: charge, solvation: .water
                ) {
                    allHits[i].gfn2Energy = result.totalEnergy_kcal
                }
            }
            let refined = allHits.prefix(topN).filter { $0.gfn2Energy != nil }.count
            ActivityLog.shared.info("GFN2-xTB: \(refined)/\(topN) hits scored", category: .dock)
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
        let screenElapsed = CFAbsoluteTimeGetCurrent() - screenStartTime
        let bestScore = allHits.first?.compositeScore
        let bestScoreStr = bestScore.map { String(format: "%.2f", $0) } ?? "N/A"
        ActivityLog.shared.success(
            "Screening complete: \(allHits.count) hits from \(total) molecules in \(String(format: "%.1f", screenElapsed))s — best composite score: \(bestScoreStr)",
            category: .dock
        )
    }

    func cancel() {
        cancelToken.cancel()
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
        var rows: [String] = ["Rank,Name,SMILES,Energy,StrainEnergy,PrescreenEnergy,OpenMMEnergy,ML_Score,Composite,MW,LogP,HBD,HBA,TPSA,RotBonds,Lipinski,DrugLikeness"]
        rows.reserveCapacity(hits.count + 1)
        for (i, hit) in hits.enumerated() {
            let desc = hit.descriptors
            let fields: [String] = [
                "\(i+1)",
                "\"\(hit.name)\"",
                "\"\(hit.smiles)\"",
                String(format: "%.2f", hit.bestEnergy),
                hit.strainEnergy.map { String(format: "%.2f", $0) } ?? "N/A",
                hit.prescreenEnergy.map { String(format: "%.2f", $0) } ?? "N/A",
                hit.openMMEnergy.map { String(format: "%.2f", $0) } ?? "N/A",
                hit.mlScore.map { String(format: "%.2f", $0) } ?? "N/A",
                String(format: "%.2f", hit.compositeScore),
                desc.map { String(format: "%.1f", $0.molecularWeight) } ?? "N/A",
                desc.map { String(format: "%.2f", $0.logP) } ?? "N/A",
                "\(desc?.hbd ?? 0)",
                "\(desc?.hba ?? 0)",
                desc.map { String(format: "%.1f", $0.tpsa) } ?? "N/A",
                "\(desc?.rotatableBonds ?? 0)",
                desc?.lipinski == true ? "Yes" : "No",
                hit.admet.map { String(format: "%.2f", $0.drugLikeness) } ?? "N/A"
            ]
            rows.append(fields.joined(separator: ","))
        }
        return rows.joined(separator: "\n") + "\n"
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

/// GPU-batched GA docking: runs the full genetic algorithm for multiple ligands simultaneously.
/// Uses batched Metal kernels (mcPerturbBatched, scorePosesBatched, metropolisAcceptBatched)
/// to process K ligands × populationSize poses in a single GPU dispatch.
final class BatchedGAAccelerator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let scorePipeline: MTLComputePipelineState
    private let perturbPipeline: MTLComputePipelineState
    private let acceptPipeline: MTLComputePipelineState
    private let initPopPipeline: MTLComputePipelineState

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let lib = device.makeDefaultLibrary(),
              let scoreF = lib.makeFunction(name: "scorePosesBatched"),
              let perturbF = lib.makeFunction(name: "mcPerturbBatched"),
              let acceptF = lib.makeFunction(name: "metropolisAcceptBatched"),
              let initF = lib.makeFunction(name: "initializePopulation")
        else { return nil }
        self.commandQueue = queue
        do {
            scorePipeline = try device.makeComputePipelineState(function: scoreF)
            perturbPipeline = try device.makeComputePipelineState(function: perturbF)
            acceptPipeline = try device.makeComputePipelineState(function: acceptF)
            initPopPipeline = try device.makeComputePipelineState(function: initF)
        } catch { return nil }
    }

    struct BatchGAResult {
        var energy: Float
        var translation: SIMD3<Float>
        var rotation: simd_quatf
        var torsions: [Float]
    }

    /// Run batched GA for multiple ligands simultaneously.
    /// Returns per-ligand best pose or nil on failure.
    func runBatchedGA(
        ligands: [LigandGAData],
        gridSnapshot: DockingGridSnapshot,
        populationSize: Int = 32,
        generations: Int = 40,
        localSearchSteps: Int = 8,
        temperature: Float = 1.2
    ) -> [BatchGAResult?] {
        guard !ligands.isEmpty,
              let affinityBuf = gridSnapshot.vinaAffinityGridBuffer,
              let typeIdxBuf = gridSnapshot.vinaTypeIndexBuffer
        else { return Array(repeating: nil, count: ligands.count) }

        let popSize = max(populationSize, 8)
        let totalPoses = ligands.count * popSize
        var results = [BatchGAResult?](repeating: nil, count: ligands.count)

        // Flatten per-ligand data into contiguous GPU buffers
        var allAtoms = [DockLigandAtom]()
        var allEdges = [TorsionEdge]()
        var allMoving = [Int32]()
        var allPairs = [UInt32]()
        var infos = [BatchedGALigandInfo]()

        for lig in ligands {
            // Remap torsion edge movingStart to global offset
            var edges = lig.torsionEdges
            let movingBase = Int32(allMoving.count)
            for i in 0..<edges.count {
                edges[i].movingStart += movingBase
            }

            infos.append(BatchedGALigandInfo(
                atomStart: UInt32(allAtoms.count),
                atomCount: UInt32(lig.gpuAtoms.count),
                torsionEdgeStart: UInt32(allEdges.count),
                torsionEdgeCount: UInt32(min(lig.torsionEdges.count, 32)),
                movingIndicesStart: UInt32(allMoving.count),
                pairListStart: UInt32(allPairs.count),
                numPairs: UInt32(lig.pairList.count),
                referenceIntraEnergy: lig.referenceIntraEnergy,
                ligandRadius: lig.ligandRadius,
                movingIndicesCount: UInt32(lig.movingIndices.count),
                _pad0: 0, _pad1: 0
            ))
            allAtoms.append(contentsOf: lig.gpuAtoms)
            allEdges.append(contentsOf: edges)
            allMoving.append(contentsOf: lig.movingIndices)
            allPairs.append(contentsOf: lig.pairList)
        }

        // Ensure non-empty arrays for buffer creation
        if allEdges.isEmpty { allEdges.append(TorsionEdge(atom1: 0, atom2: 0, movingStart: 0, movingCount: 0)) }
        if allMoving.isEmpty { allMoving.append(0) }
        if allPairs.isEmpty { allPairs.append(0) }

        guard let atomBuf = device.makeBuffer(bytes: allAtoms, length: allAtoms.count * MemoryLayout<DockLigandAtom>.stride, options: .storageModeShared),
              let edgeBuf = device.makeBuffer(bytes: allEdges, length: allEdges.count * MemoryLayout<TorsionEdge>.stride, options: .storageModeShared),
              let movBuf = device.makeBuffer(bytes: allMoving, length: allMoving.count * MemoryLayout<Int32>.stride, options: .storageModeShared),
              let pairBuf = device.makeBuffer(bytes: allPairs, length: allPairs.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let infoBuf = device.makeBuffer(bytes: infos, length: infos.count * MemoryLayout<BatchedGALigandInfo>.stride, options: .storageModeShared),
              let popBuf = device.makeBuffer(length: totalPoses * MemoryLayout<DockPose>.stride, options: .storageModeShared),
              let offBuf = device.makeBuffer(length: totalPoses * MemoryLayout<DockPose>.stride, options: .storageModeShared),
              let bestBuf = device.makeBuffer(length: totalPoses * MemoryLayout<DockPose>.stride, options: .storageModeShared)
        else { return results }

        // Initialize best buffer to infinity
        memset(bestBuf.contents(), 0, totalPoses * MemoryLayout<DockPose>.stride)
        let bestPoses = bestBuf.contents().bindMemory(to: DockPose.self, capacity: totalPoses)
        for i in 0..<totalPoses { bestPoses[i].energy = .infinity }

        var batchParams = BatchedGAParams(
            numLigands: UInt32(ligands.count),
            populationSizePerLigand: UInt32(popSize),
            totalPoses: UInt32(totalPoses),
            generation: 0,
            localSearchSteps: UInt32(localSearchSteps),
            mutationRate: 0.3,
            crossoverRate: 0.6,
            translationStep: 0.8,
            rotationStep: 0.3,
            torsionStep: 0.5,
            mcTemperature: temperature,
            _pad0: 0
        )
        guard let batchParamsBuf = device.makeBuffer(bytes: &batchParams, length: MemoryLayout<BatchedGAParams>.stride, options: .storageModeShared)
        else { return results }

        let tgWidth = 256
        let tgSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        let tgCount = MTLSize(width: (totalPoses + tgWidth - 1) / tgWidth, height: 1, depth: 1)

        // Initialize population using per-ligand initializePopulation (reuse existing kernel)
        // We need per-ligand GAParams for init. Instead, manually init via batched perturb from zero.
        // Set all poses to random within search box by running perturbBatched from zeroed state.
        memset(popBuf.contents(), 0, totalPoses * MemoryLayout<DockPose>.stride)
        let poses = popBuf.contents().bindMemory(to: DockPose.self, capacity: totalPoses)
        for i in 0..<totalPoses {
            poses[i].energy = .infinity
            poses[i].rotation = SIMD4<Float>(0, 0, 0, 1)
            poses[i].translation = gridSnapshot.gridParams.searchCenter
        }

        // Run batched perturbation to randomize initial population
        batchParams.generation = 0
        batchParams.mutationRate = 1.0  // force full mutation for init
        batchParamsBuf.contents().copyMemory(from: &batchParams, byteCount: MemoryLayout<BatchedGAParams>.stride)

        if let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(perturbPipeline)
            enc.setBuffer(offBuf, offset: 0, index: 0)
            enc.setBuffer(popBuf, offset: 0, index: 1)
            enc.setBuffer(batchParamsBuf, offset: 0, index: 2)
            enc.setBuffer(gridSnapshot.gridParamsBuffer, offset: 0, index: 3)
            enc.setBuffer(infoBuf, offset: 0, index: 4)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }
        // Copy randomized offspring → population
        memcpy(popBuf.contents(), offBuf.contents(), totalPoses * MemoryLayout<DockPose>.stride)

        // Restore mutation rate
        batchParams.mutationRate = 0.3

        // === GA loop ===
        for gen in 0..<generations {
            batchParams.generation = UInt32(gen)
            batchParamsBuf.contents().copyMemory(from: &batchParams, byteCount: MemoryLayout<BatchedGAParams>.stride)

            guard let cb = commandQueue.makeCommandBuffer() else { continue }

            // 1. Perturb
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(perturbPipeline)
                enc.setBuffer(offBuf, offset: 0, index: 0)
                enc.setBuffer(popBuf, offset: 0, index: 1)
                enc.setBuffer(batchParamsBuf, offset: 0, index: 2)
                enc.setBuffer(gridSnapshot.gridParamsBuffer, offset: 0, index: 3)
                enc.setBuffer(infoBuf, offset: 0, index: 4)
                enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }

            // 2. Score
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(scorePipeline)
                enc.setBuffer(offBuf, offset: 0, index: 0)
                enc.setBuffer(atomBuf, offset: 0, index: 1)
                enc.setBuffer(affinityBuf, offset: 0, index: 2)
                enc.setBuffer(typeIdxBuf, offset: 0, index: 3)
                enc.setBuffer(gridSnapshot.gridParamsBuffer, offset: 0, index: 4)
                enc.setBuffer(batchParamsBuf, offset: 0, index: 5)
                enc.setBuffer(edgeBuf, offset: 0, index: 6)
                enc.setBuffer(movBuf, offset: 0, index: 7)
                enc.setBuffer(pairBuf, offset: 0, index: 8)
                enc.setBuffer(infoBuf, offset: 0, index: 9)
                enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }

            // 3. Accept
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(acceptPipeline)
                enc.setBuffer(popBuf, offset: 0, index: 0)
                enc.setBuffer(offBuf, offset: 0, index: 1)
                enc.setBuffer(bestBuf, offset: 0, index: 2)
                enc.setBuffer(batchParamsBuf, offset: 0, index: 3)
                enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }

            cb.commit()
            // Don't wait every generation — batch submit and wait periodically
            if gen == generations - 1 || gen % 10 == 9 {
                cb.waitUntilCompleted()
            }
        }

        // Extract best poses per ligand
        let bestResults = bestBuf.contents().bindMemory(to: DockPose.self, capacity: totalPoses)
        for ligIdx in 0..<ligands.count {
            var bestE: Float = .infinity
            var bestPoseIdx = -1
            let start = ligIdx * popSize
            for p in 0..<popSize {
                let e = bestResults[start + p].energy
                if e < bestE { bestE = e; bestPoseIdx = start + p }
            }
            guard bestPoseIdx >= 0, bestE < 1e9 else { continue }
            let bp = bestResults[bestPoseIdx]
            var torsions = [Float]()
            withUnsafePointer(to: bp.torsions) { ptr in
                ptr.withMemoryRebound(to: Float.self, capacity: 32) { buf in
                    for t in 0..<min(Int(bp.numTorsions), 32) { torsions.append(buf[t]) }
                }
            }
            results[ligIdx] = BatchGAResult(
                energy: bp.energy,
                translation: bp.translation,
                rotation: simd_quatf(ix: bp.rotation.x, iy: bp.rotation.y, iz: bp.rotation.z, r: bp.rotation.w),
                torsions: torsions
            )
        }

        return results
    }
}

final class OpenMMPocketRefiner {
    struct RefinementResult: Sendable {
        var interactionEnergyKcal: Float
        var refinedHeavyAtomPositions: [SIMD3<Float>]
    }

    nonisolated(unsafe) static let shared = OpenMMPocketRefiner()

    var isAvailable: Bool { druse_openmm_available() }
    var availabilitySummary: String {
        isAvailable ? "Native C++ OpenMM available" : "OpenMM not compiled (DRUSE_HAS_OPENMM not set)"
    }

    private init() {}

    func refine(
        proteinAtoms: [Atom],
        ligandAtoms: [Atom],
        pocketCenter: SIMD3<Float>,
        pocketHalfExtent: SIMD3<Float>
    ) async -> RefinementResult? {
        let heavyProteinAtoms = proteinAtoms.filter { $0.element != .H }
        let heavyLigandAtoms = ligandAtoms.filter { $0.element != .H }
        guard !heavyProteinAtoms.isEmpty, !heavyLigandAtoms.isEmpty else { return nil }

        return await refineNative(
            heavyProteinAtoms: heavyProteinAtoms,
            heavyLigandAtoms: heavyLigandAtoms,
            proteinAtoms: proteinAtoms,
            pocketCenter: pocketCenter,
            pocketHalfExtent: pocketHalfExtent
        )
    }

    // MARK: - Native C++ OpenMM Refinement

    private func refineNative(
        heavyProteinAtoms: [Atom],
        heavyLigandAtoms: [Atom],
        proteinAtoms: [Atom],
        pocketCenter: SIMD3<Float>,
        pocketHalfExtent: SIMD3<Float>
    ) async -> RefinementResult? {
        let pocketRadius = max(pocketHalfExtent.x, max(pocketHalfExtent.y, pocketHalfExtent.z)) + 4.0
        let pocketRadiusSq = pocketRadius * pocketRadius

        // Build protein atom array
        var cAtoms: [DruseOpenMMAtom] = []
        for atom in heavyProteinAtoms {
            // Determine if atom is in pocket (within pocketRadius of pocketCenter)
            let dx = atom.position.x - pocketCenter.x
            let dy = atom.position.y - pocketCenter.y
            let dz = atom.position.z - pocketCenter.z
            let isPocket = (dx * dx + dy * dy + dz * dz) <= pocketRadiusSq

            cAtoms.append(DruseOpenMMAtom(
                x: atom.position.x,
                y: atom.position.y,
                z: atom.position.z,
                charge: abs(atom.charge) > 0.0001 ? atom.charge : Float(atom.formalCharge),
                sigmaNm: Self.ljSigmaNm(for: atom),
                epsilonKJ: Self.ljEpsilonKJ(for: atom),
                mass: atom.element.mass,
                atomicNum: Int32(atom.element.rawValue),
                isPocket: isPocket
            ))
        }

        // Build bond array from all protein atoms (heavy only)
        // We need to find bonds between heavy atoms from the full atom list
        var heavyIdMap: [Int: Int] = [:]  // original atom id → heavy-only index
        for (newIdx, atom) in heavyProteinAtoms.enumerated() {
            heavyIdMap[atom.id] = newIdx
        }

        var cBonds: [DruseOpenMMBond] = []
        // Estimate bonds from distance (< 1.9 Å between bonded heavy atoms)
        let bondCutoffSq: Float = 1.9 * 1.9
        for i in 0..<heavyProteinAtoms.count {
            for j in (i+1)..<heavyProteinAtoms.count {
                let ai = heavyProteinAtoms[i]
                let aj = heavyProteinAtoms[j]
                // Only bond atoms in the same residue or adjacent backbone
                let sameResidue = ai.residueSeq == aj.residueSeq && ai.chainID == aj.chainID
                let adjacent = abs(ai.residueSeq - aj.residueSeq) <= 1 && ai.chainID == aj.chainID
                guard sameResidue || adjacent else { continue }

                let dSq = simd_distance_squared(ai.position, aj.position)
                if dSq < bondCutoffSq && dSq > 0.5 * 0.5 {
                    let lengthNm = sqrtf(dSq) / 10.0  // Å → nm
                    cBonds.append(DruseOpenMMBond(
                        atom1: Int32(i),
                        atom2: Int32(j),
                        lengthNm: lengthNm
                    ))
                }
            }
        }

        // Build ligand site array (positions already in nm for OpenMM)
        var cLigands: [DruseOpenMMLigandSite] = []
        for atom in heavyLigandAtoms {
            cLigands.append(DruseOpenMMLigandSite(
                x: atom.position.x / 10.0,
                y: atom.position.y / 10.0,
                z: atom.position.z / 10.0,
                charge: abs(atom.charge) > 0.0001 ? atom.charge : Float(atom.formalCharge),
                sigmaNm: Self.ljSigmaNm(for: atom),
                epsilonKJ: Self.ljEpsilonKJ(for: atom)
            ))
        }

        // Call C++ OpenMM on background thread
        let atomArray = cAtoms
        let bondArray = cBonds
        let ligandArray = cLigands
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let result = atomArray.withUnsafeBufferPointer { atomsBuf in
                    bondArray.withUnsafeBufferPointer { bondsBuf in
                        ligandArray.withUnsafeBufferPointer { ligandsBuf in
                            druse_openmm_refine(
                                atomsBuf.baseAddress,
                                Int32(atomArray.count),
                                bondsBuf.baseAddress,
                                Int32(bondArray.count),
                                ligandsBuf.baseAddress,
                                Int32(ligandArray.count),
                                600.0,   // pocketK
                                8000.0,  // backboneK
                                250      // maxIterations
                            )
                        }
                    }
                }

                defer { druse_free_openmm_result(result) }

                guard let result, result.pointee.success else {
                    if let result {
                        let msg = withUnsafeBytes(of: result.pointee.errorMessage) { buf in
                            String(cString: buf.baseAddress!.assumingMemoryBound(to: CChar.self))
                        }
                        Task { @MainActor in ActivityLog.shared.warn("OpenMM refinement failed: \(msg)", category: .dock) }
                    }
                    continuation.resume(returning: nil)
                    return
                }

                let count = Int(result.pointee.atomCount)
                var positions: [SIMD3<Float>] = []
                positions.reserveCapacity(count)
                for i in 0..<count {
                    positions.append(SIMD3<Float>(
                        result.pointee.refinedPositionsX[i],
                        result.pointee.refinedPositionsY[i],
                        result.pointee.refinedPositionsZ[i]
                    ))
                }

                continuation.resume(returning: RefinementResult(
                    interactionEnergyKcal: result.pointee.interactionEnergyKcal,
                    refinedHeavyAtomPositions: positions
                ))
            }
        }
    }

    // MARK: - LJ Parameters

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

}

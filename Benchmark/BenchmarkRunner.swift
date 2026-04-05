import XCTest
import simd
import MetalKit
@testable import Druse

// =============================================================================
// MARK: - CASF-2016 Benchmark Runner
// =============================================================================
//
// Three independent benchmarks — one per scoring method:
//
//   xcodebuild test ... -only-testing:DruseTests/BenchmarkRunner/testCASF_Vina
//   xcodebuild test ... -only-testing:DruseTests/BenchmarkRunner/testCASF_Drusina
//   xcodebuild test ... -only-testing:DruseTests/BenchmarkRunner/testCASF_DruseAF
//   xcodebuild test ... -only-testing:DruseTests/BenchmarkRunner/testCASF_PIGNet2
//
// Each writes its own results JSON. The Python analyzer compares all three.
//
// Prerequisites:
//   python Benchmark/download.py --setup
//   python Benchmark/prepare.py

final class BenchmarkRunner: XCTestCase {

    // MARK: - Shared Engine & DruseAF Scorer

    @MainActor private static var _engine: DockingEngine?
    @MainActor private static var _pocketDetector: PocketDetectorInference?

    @MainActor
    private func engine() throws -> DockingEngine {
        if let e = Self._engine { return e }
        guard let device = MTLCreateSystemDefaultDevice(),
              let e = DockingEngine(device: device) else {
            throw XCTSkip("No Metal GPU / DockingEngine")
        }
        Self._engine = e
        print("[Benchmark] DockingEngine on \(device.name)")
        return e
    }

    private var projectRoot: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func resolve(_ rel: String) -> String {
        projectRoot.appendingPathComponent(rel).path
    }

    private func resolveConfiguredPath(_ path: String) -> String {
        if path.hasPrefix("/") {
            return path
        }
        return projectRoot.appendingPathComponent(path).path
    }

    private func resolveOutputPath(_ outputFile: String) -> String {
        let configured = Self.cfgOutputDirectory
        let baseURL: URL
        if let configured, !configured.isEmpty {
            if configured.hasPrefix("/") {
                baseURL = URL(fileURLWithPath: configured, isDirectory: true)
            } else {
                baseURL = projectRoot.appendingPathComponent(configured, isDirectory: true)
            }
        } else {
            baseURL = projectRoot.appendingPathComponent("Benchmark/results", isDirectory: true)
        }
        return baseURL.appendingPathComponent(outputFile).path
    }

    // MARK: - Helpers

    /// Write directly to stdout with immediate flush (bypasses xcodebuild buffering).
    private func emit(_ msg: String) {
        let line = msg + "\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardOutput.write(data)
        }
    }

    private func computeRMSD(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return .infinity }
        var sum: Float = 0
        for i in 0..<n { sum += simd_distance_squared(a[i], b[i]) }
        return sqrt(sum / Float(n))
    }

    /// Symmetry-corrected RMSD: matches atoms by element type then finds the
    /// minimum-RMSD assignment within each element group.
    /// `refPositions`/`refElements` = crystal; `probePositions`/`probeElements` = docked.
    private func computeSymmetryRMSD(
        refPositions: [SIMD3<Float>], refElements: [Element],
        probePositions: [SIMD3<Float>], probeElements: [Element]
    ) -> Float {
        guard refPositions.count == probePositions.count,
              refPositions.count == refElements.count,
              probePositions.count == probeElements.count,
              !refPositions.isEmpty else { return .infinity }

        let n = refPositions.count
        // Group indices by element
        var refGroups: [Element: [Int]] = [:]
        var probeGroups: [Element: [Int]] = [:]
        for i in 0..<n {
            refGroups[refElements[i], default: []].append(i)
            probeGroups[probeElements[i], default: []].append(i)
        }

        // Verify element counts match
        for (elem, refIdxs) in refGroups {
            guard let probeIdxs = probeGroups[elem], probeIdxs.count == refIdxs.count else {
                return .infinity
            }
        }

        // For each element group, find optimal assignment (minimize sum of squared distances).
        // Use Hungarian for groups > 6, brute force for small groups.
        var mapping = [Int](repeating: -1, count: n)  // mapping[probe_i] = ref_i

        for (elem, refIdxs) in refGroups {
            let probeIdxs = probeGroups[elem]!
            let k = refIdxs.count

            if k == 1 {
                mapping[probeIdxs[0]] = refIdxs[0]
            } else if k <= 8 {
                // Brute-force: try all permutations of refIdxs, pick min cost
                var bestCost: Float = .infinity
                var bestPerm = refIdxs
                permute(refIdxs) { perm in
                    var cost: Float = 0
                    for j in 0..<k {
                        cost += simd_distance_squared(probePositions[probeIdxs[j]], refPositions[perm[j]])
                    }
                    if cost < bestCost {
                        bestCost = cost
                        bestPerm = perm
                    }
                }
                for j in 0..<k { mapping[probeIdxs[j]] = bestPerm[j] }
            } else {
                // Greedy nearest-neighbor for large groups
                var availableRef = Set(refIdxs)
                for pi in probeIdxs {
                    var bestRef = -1
                    var bestDist: Float = .infinity
                    for ri in availableRef {
                        let d = simd_distance_squared(probePositions[pi], refPositions[ri])
                        if d < bestDist { bestDist = d; bestRef = ri }
                    }
                    mapping[pi] = bestRef
                    availableRef.remove(bestRef)
                }
            }
        }

        var sum: Float = 0
        for i in 0..<n {
            sum += simd_distance_squared(probePositions[i], refPositions[mapping[i]])
        }
        return sqrt(sum / Float(n))
    }

    /// Generate all permutations and call closure for each.
    private func permute(_ arr: [Int], _ body: ([Int]) -> Void) {
        var a = arr
        func helper(_ n: Int) {
            if n == 1 { body(a); return }
            for i in 0..<n {
                helper(n - 1)
                a.swapAt(n % 2 == 0 ? i : 0, n - 1)
            }
        }
        helper(a.count)
    }

    private func pearsonR(_ x: [Float], _ y: [Float]) -> Float {
        let n = Float(x.count)
        guard n > 2 else { return 0 }
        let sx = x.reduce(0, +), sy = y.reduce(0, +)
        let sxy = zip(x, y).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let sx2 = x.reduce(Float(0)) { $0 + $1 * $1 }
        let sy2 = y.reduce(Float(0)) { $0 + $1 * $1 }
        let den = sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy))
        return den > 0 ? (n * sxy - sx * sy) / den : 0
    }

    @MainActor
    private func loadProtein(path: String) -> Molecule? {
        guard let content = try? String(contentsOfFile: path, encoding: .utf8),
              let data = PDBParser.parse(content).protein else { return nil }
        return Molecule(name: data.name, atoms: data.atoms, bonds: data.bonds, title: data.title)
    }

    @MainActor
    private func loadCrystalLigand(sdfPath: String) -> Molecule? {
        guard let content = try? String(contentsOfFile: sdfPath, encoding: .utf8),
              let data = SDFParser.parse(content).first else { return nil }
        return Molecule(name: data.name, atoms: data.atoms, bonds: data.bonds, title: data.title)
    }

    private func writeResults(_ results: BenchmarkResults, to path: String) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? enc.encode(results) else { return }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                  withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    private func loadManifest() throws -> CASFManifest {
        let path = Self.cfgManifestPath.map(resolveConfiguredPath(_:))
            ?? resolve("Benchmark/manifests/casf_manifest.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw XCTSkip("Manifest not found at \(path). Run:\n  python Benchmark/prepare.py")
        }
        return try JSONDecoder().decode(CASFManifest.self, from: data)
    }

    @MainActor
    private func pocketDetector() -> PocketDetectorInference {
        if let detector = Self._pocketDetector { return detector }
        let detector = PocketDetectorInference()
        detector.loadModel()
        Self._pocketDetector = detector
        return detector
    }

    @MainActor
    private func prepareBenchmarkLigand(from complex: CASFComplex) throws -> Molecule {
        let (prepared, _, _, error) = RDKitBridge.prepareLigand(
            smiles: complex.smiles,
            name: complex.pdbId,
            numConformers: 1,
            addHydrogens: false,
            minimize: true,
            computeCharges: true
        )
        if let error {
            throw BenchmarkError.step("ligand prep: \(error)")
        }
        guard let prepared else {
            throw BenchmarkError.step("ligand prep returned nil")
        }
        return Molecule(
            name: prepared.name,
            atoms: prepared.atoms,
            bonds: prepared.bonds,
            title: complex.smiles,
            smiles: complex.smiles
        )
    }

    @MainActor
    private func detectBenchmarkPocket(
        protein: Molecule,
        crystalLigand: Molecule?
    ) async throws -> (pocket: BindingPocket, method: String) {
        switch Self.cfgPocketMode {
        case "ligand-guided":
            guard let crystalLigand,
                  let pocket = BindingSiteDetector.ligandGuidedPocket(
                    protein: protein,
                    ligand: crystalLigand,
                    distance: 6.0
                  ) else {
                throw BenchmarkError.step("ligand-guided pocket detection")
            }
            return (pocket, "ligand-guided")

        case "ml":
            let detector = pocketDetector()
            let pockets = await detector.detectPockets(protein: protein)
            guard let pocket = pockets.first else {
                throw BenchmarkError.step("ml pocket detection")
            }
            return (pocket, "ml")

        case "hybrid":
            let detector = pocketDetector()
            let mlPockets = await detector.detectPockets(protein: protein)
            let geometricPockets = BindingSiteDetector.detectPockets(protein: protein)
            if let selected = PocketSelectionHeuristics.bestHybridCandidate(
                mlPockets: mlPockets,
                geometricPockets: geometricPockets
            ) {
                return (selected.pocket, selected.method.rawValue)
            }
            fallthrough

        default:
            guard let pocket = BindingSiteDetector.detectPockets(protein: protein).first else {
                throw BenchmarkError.step("geometric pocket detection")
            }
            return (pocket, "geometric")
        }
    }

    @MainActor
    private func configureFlexibleResiduesIfNeeded(
        engine: DockingEngine,
        protein: Molecule,
        pocket: BindingPocket
    ) {
        guard Self.cfgFlexResidues else {
            engine.flexEngine = nil
            return
        }

        let selectedResidues = FlexibleResidueConfig.autoSelectResidues(
            protein: protein.atoms,
            pocket: (center: pocket.center, residueIndices: pocket.residueIndices)
        )
        guard !selectedResidues.isEmpty else {
            engine.flexEngine = nil
            return
        }

        let flexConfig = FlexibleResidueConfig(
            flexibleResidueIndices: selectedResidues,
            rotamerResolution: 10.0,
            autoFlex: true
        )
        let flexEngine = FlexDockingEngine(device: engine.device, commandQueue: engine.commandQueue)
        let vinaTypes = engine.vinaTypesForProtein(protein)
        let exclusion = flexEngine.excludeFlexAtoms(
            proteinAtoms: protein.atoms,
            proteinBonds: protein.bonds,
            flexConfig: flexConfig,
            vinaTypes: vinaTypes
        )
        let flexWeight = FlexibleResidueConfig.softFlexWeight
        let chiStep = FlexibleResidueConfig.softChiStep
        flexEngine.prepareFlexBuffers(exclusion: exclusion, flexWeight: flexWeight, chiStep: chiStep)
        engine.flexEngine = flexEngine
    }

    private func applyStrainPenalties(
        results: [DockingResult],
        smiles: String,
        config: DockingConfig
    ) async -> [DockingResult] {
        let topN = min(results.count, 50)
        guard topN > 0 else { return results }

        let threshold = config.strainPenaltyThreshold
        let weight = config.strainPenaltyWeight
        let refEnergy = await Task.detached(priority: .userInitiated) {
            RDKitBridge.mmffReferenceEnergy(smiles: smiles)
        }.value
        guard let refEnergy else { return results }

        var updated = results
        let positionsSlice = results.prefix(topN).map(\.transformedAtomPositions)
        let strainResults: [(Int, Float?)] = await Task.detached(priority: .userInitiated) {
            positionsSlice.enumerated().map { (i, positions) in
                guard !positions.isEmpty else { return (i, nil as Float?) }
                if let dockedEnergy = RDKitBridge.mmffStrainEnergy(smiles: smiles, heavyPositions: positions) {
                    return (i, Float(dockedEnergy - refEnergy))
                }
                return (i, nil)
            }
        }.value

        var changed = false
        for (i, strain) in strainResults {
            guard let strain else { continue }
            let clamped = min(max(strain, 0.0), 100.0)
            updated[i].strainEnergy = clamped
            if clamped > threshold {
                updated[i].energy += weight * (clamped - threshold)
                changed = true
            }
        }
        if changed {
            updated.sort { $0.energy < $1.energy }
        }
        return updated
    }

    @MainActor
    private func refineWithGFN2(
        results: [DockingResult],
        originalLigand: Molecule,
        config: GFN2RefinementConfig
    ) async -> [DockingResult] {
        let topN = min(results.count, config.topPosesToRefine)
        guard topN > 0 else { return results }

        let heavyAtoms = originalLigand.atoms.filter { $0.element != .H }
        guard heavyAtoms.count >= 2 else { return results }

        let formalCharge = originalLigand.atoms.reduce(0) { $0 + $1.formalCharge }
        var updated = results

        let refinements: [(Int, GFN2RefinementResult?)] = await withTaskGroup(
            of: (Int, GFN2RefinementResult?).self
        ) { group in
            for i in 0..<topN {
                let positions = results[i].transformedAtomPositions
                guard positions.count == heavyAtoms.count else { continue }
                group.addTask {
                    var dockedAtoms = heavyAtoms
                    for j in 0..<dockedAtoms.count {
                        dockedAtoms[j].position = positions[j]
                    }
                    do {
                        let result = try await GFN2Refiner.optimizeGeometry(
                            atoms: dockedAtoms,
                            totalCharge: formalCharge,
                            solvation: config.solvation,
                            optLevel: config.optLevel,
                            maxSteps: config.maxSteps,
                            referencePositions: config.restraintStrength > 0 ? positions : nil,
                            restraintStrength: config.restraintStrength
                        )
                        return (i, result)
                    } catch {
                        return (i, nil)
                    }
                }
            }

            var collected = [(Int, GFN2RefinementResult?)]()
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        var gfn2Energies = [Float]()
        for (i, gfn2Result) in refinements {
            guard let gfn2Result else { continue }
            updated[i].gfn2Energy = gfn2Result.totalEnergy_kcal
            updated[i].gfn2DispersionEnergy = gfn2Result.dispersionEnergy * 627.509
            updated[i].gfn2SolvationEnergy = gfn2Result.solvationEnergy * 627.509
            updated[i].gfn2Converged = gfn2Result.converged
            updated[i].gfn2OptSteps = gfn2Result.steps
            gfn2Energies.append(gfn2Result.totalEnergy_kcal)

            if config.updateCoordinates, let optPos = gfn2Result.optimizedPositions {
                let origPos = updated[i].transformedAtomPositions
                if origPos.count == optPos.count {
                    var sumSq: Float = 0
                    for j in 0..<origPos.count {
                        sumSq += simd_distance_squared(origPos[j], optPos[j])
                    }
                    let rmsd = sqrt(sumSq / Float(origPos.count))
                    if rmsd <= config.maxRMSD {
                        updated[i].transformedAtomPositions = optPos
                    }
                }
            }
        }

        if let minGFN2 = gfn2Energies.min(), config.blendWeight > 0 {
            for i in 0..<updated.count {
                if let gfn2E = updated[i].gfn2Energy {
                    updated[i].energy += config.blendWeight * (gfn2E - minGFN2)
                }
            }
            updated.sort { $0.energy < $1.energy }
        }

        return updated
    }

    // ==========================================================================
    // MARK: - Core Docking Loop
    // ==========================================================================

    /// Dock all complexes with a given scoring method.
    /// - Parameter maxComplexes: If > 0, limit to first N complexes for quick validation.
    @MainActor
    private func runBenchmark(
        scoringMethod: ScoringMethod,
        label: String,
        outputFile: String,
        maxComplexes: Int = 0,
        enableGFN2Scoring: Bool = false
    ) async throws {
        let manifest = try loadManifest()
        let complexes = maxComplexes > 0
            ? Array(manifest.complexes.prefix(maxComplexes))
            : manifest.complexes
        let suffix = maxComplexes > 0 ? " (first \(complexes.count))" : ""
        print("[\(label)] \(complexes.count) complexes\(suffix)")

        let eng = try engine()

        var config = DockingConfig()
        config = Self.configFromFile(base: config)
        // Override GFN2 scoring flag from function parameter
        if enableGFN2Scoring {
            config.gfn2Refinement.enabled = config.gfn2Refinement.enabled // preserve opt setting
        }

        let resultsPath = resolveOutputPath(outputFile)
        let version = (try? String(contentsOfFile: resolve("VERSION"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
        var bench = BenchmarkResults(
            benchmark: Self.cfgBenchmarkName ?? manifest.benchmark,
            version: version,
            scoringMethod: label,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            config: BenchmarkConfigRecord(
                populationSize: config.populationSize,
                generations: config.generationsPerRun,
                gridSpacing: config.gridSpacing,
                numRuns: config.numRuns
            ),
            entries: []
        )

        let total = complexes.count
        var succeeded = 0, failed = 0

        for (idx, complex) in complexes.enumerated() {
            let t0 = CFAbsoluteTimeGetCurrent()
            var entry = BenchmarkResultEntry(
                pdbId: complex.pdbId,
                bestEnergy: 1e9,
                bestDisplayScore: nil,
                bestRmsd: nil,
                experimentalPKd: complex.pKd,
                numPoses: 0,
                dockingTimeMs: 0,
                success: false,
                error: nil
            )

            do {
                // 1. Load + prepare protein (full pipeline: protonation, H, charges)
                guard let rawProtein = loadProtein(path: resolve(complex.proteinPdb)) else {
                    throw BenchmarkError.step("parse protein")
                }
                let pdbContent = try? String(contentsOfFile: resolve(complex.proteinPdb), encoding: .utf8)
                let rawAtoms = rawProtein.atoms
                let rawBonds = rawProtein.bonds
                let prepared = await Task.detached { @Sendable in
                    ProteinPreparation.prepareForDocking(
                        atoms: rawAtoms, bonds: rawBonds,
                        rawPDBContent: pdbContent,
                        pH: 7.4,
                        chargeMethod: Self.cfgChargeMethod
                    )
                }.value
                let protein = Molecule(name: rawProtein.name,
                                       atoms: prepared.atoms, bonds: prepared.bonds,
                                       title: rawProtein.title)

                // 2. Load crystal ligand from SDF (reference for RMSD and optional redocking)
                guard let crystalLig = loadCrystalLigand(sdfPath: resolve(complex.ligandSdf)) else {
                    throw BenchmarkError.step("parse crystal ligand SDF")
                }

                // 3. Ligand / pocket selection depends on benchmark mode.
                let ligand: Molecule
                let pocket: BindingPocket
                let pocketMethod: String
                if Self.cfgPipelineMode == "full" {
                    ligand = try prepareBenchmarkLigand(from: complex)
                    let detected = try await detectBenchmarkPocket(protein: protein, crystalLigand: nil)
                    pocket = detected.pocket
                    pocketMethod = detected.method
                } else {
                    ligand = crystalLig
                    let detected = try await detectBenchmarkPocket(protein: protein, crystalLigand: crystalLig)
                    pocket = detected.pocket
                    pocketMethod = detected.method
                }

                // -- Debug diagnostics --
                let smilesOrder = complex.crystalPositionsSMILESOrder
                let crystalPositions: [SIMD3<Float>]
                if Self.cfgPipelineMode == "full", let order = smilesOrder {
                    crystalPositions = order
                } else {
                    crystalPositions = complex.crystalPositionsSIMD
                }
                let crystalCentroid: SIMD3<Float> = {
                    guard !crystalPositions.isEmpty else { return .zero }
                    let sum = crystalPositions.reduce(SIMD3<Float>.zero, +)
                    return sum / Float(crystalPositions.count)
                }()
                let pocketDist = simd_distance(pocket.center, crystalCentroid)

                entry.pocketCenter = [pocket.center.x, pocket.center.y, pocket.center.z]
                entry.crystalCenter = [crystalCentroid.x, crystalCentroid.y, crystalCentroid.z]
                entry.pocketDistance = pocketDist
                entry.pocketVolume = pocket.volume
                entry.pocketBuriedness = pocket.buriedness
                entry.pocketMethod = pocketMethod
                entry.searchBoxSize = [pocket.size.x, pocket.size.y, pocket.size.z]
                entry.crystalHeavyCount = crystalPositions.count
                entry.ligandHeavyCount = ligand.atoms.filter { $0.element != .H }.count

                // Conformer RMSD: SMILES-derived 3D embedding vs crystal (large values expected — measures embedding quality, not docking)
                if Self.cfgPipelineMode == "full" {
                    let preparedHeavy = ligand.atoms.filter { $0.element != .H }.map(\.position)
                    if preparedHeavy.count == crystalPositions.count {
                        entry.conformerRmsd = computeRMSD(preparedHeavy, crystalPositions)
                    }
                }

                // 4. Set protein on engine and optionally configure flexible residues.
                eng.setProtein(protein.atoms, protein.bonds)
                configureFlexibleResiduesIfNeeded(engine: eng, protein: protein, pocket: pocket)

                // 5. Dock (DruseAF runs its own ML scoring on Metal during the GA)
                let gaResults = await eng.runDocking(
                    ligand: ligand, pocket: pocket,
                    config: config, scoringMethod: scoringMethod
                )

                let results: [DockingResult]
                if Self.cfgPipelineMode == "full" {
                    var rankedResults = gaResults.sorted { $0.energy < $1.energy }
                    // Keep the benchmark Vina path scorer-pure; strain reranking is a Druse-side heuristic.
                    if config.strainPenaltyEnabled && scoringMethod != .vina {
                        rankedResults = await applyStrainPenalties(
                            results: rankedResults,
                            smiles: complex.smiles,
                            config: config
                        )
                    }
                    if config.gfn2Refinement.enabled {
                        rankedResults = await refineWithGFN2(
                            results: rankedResults,
                            originalLigand: ligand,
                            config: config.gfn2Refinement
                        )
                    }
                    results = rankedResults
                } else {
                    results = gaResults
                }

                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                entry.numPoses = results.count
                entry.dockingTimeMs = elapsed

                if let best = results.first {
                    entry.bestEnergy = best.energy
                    entry.bestDisplayScore = best.displayScore(method: scoringMethod)
                    entry.success = true

                    // RMSD
                    let docked = best.transformedAtomPositions
                    if Self.cfgPipelineMode == "full" {
                        // Full mode: atom orderings differ (SDF vs SMILES).
                        // Use symmetry-corrected RMSD with element-based matching.
                        let crystalHeavy = crystalLig.atoms.filter { $0.element != .H }
                        let dockedHeavy = ligand.atoms.filter { $0.element != .H }
                        if crystalHeavy.count == docked.count && dockedHeavy.count == docked.count {
                            let refPositions = crystalHeavy.map(\.position)
                            let refElements = crystalHeavy.map(\.element)
                            let probeElements = dockedHeavy.map(\.element)
                            entry.bestRmsd = computeSymmetryRMSD(
                                refPositions: refPositions, refElements: refElements,
                                probePositions: docked, probeElements: probeElements)
                        }
                    } else {
                        // Redock mode: same atom ordering (both from SDF).
                        let crystal = complex.crystalPositionsSIMD
                        if crystal.count == docked.count {
                            entry.bestRmsd = computeRMSD(crystal, docked)
                        }
                    }

                    // Top-N pose RMSDs for convergence analysis
                    let topN = min(results.count, 10)
                    var poseRmsds = [Float]()
                    if Self.cfgPipelineMode == "full" {
                        let crystalHeavy = crystalLig.atoms.filter { $0.element != .H }
                        let dockedHeavy = ligand.atoms.filter { $0.element != .H }
                        let refPositions = crystalHeavy.map(\.position)
                        let refElements = crystalHeavy.map(\.element)
                        let probeElements = dockedHeavy.map(\.element)
                        for i in 0..<topN {
                            let pos = results[i].transformedAtomPositions
                            if refPositions.count == pos.count {
                                poseRmsds.append(computeSymmetryRMSD(
                                    refPositions: refPositions, refElements: refElements,
                                    probePositions: pos, probeElements: probeElements))
                            }
                        }
                    } else {
                        let crystal = complex.crystalPositionsSIMD
                        for i in 0..<topN {
                            let pos = results[i].transformedAtomPositions
                            if crystal.count == pos.count {
                                poseRmsds.append(computeRMSD(crystal, pos))
                            }
                        }
                    }
                    if !poseRmsds.isEmpty {
                        entry.allPoseRmsds = poseRmsds
                    }

                    // Strain energy from best pose
                    if let strain = best.strainEnergy {
                        entry.strainEnergy = strain
                    }

                    // Drusina per-term decomposition (works with any scoring method)
                    if let popBuf = eng.populationBuffer,
                       let gaBuf = eng.gaParamsBuffer {
                        if let decomps = eng.computeDrusinaDecomposition(
                            poseBuffer: popBuf, gaParamsBuffer: gaBuf, poseCount: 1
                        ), let d = decomps.first {
                            entry.drusinaDecomposition = [
                                "pi_pi": d.piPi,
                                "pi_cation": d.piCation,
                                "salt_bridge": d.saltBridge,
                                "amide_pi": d.amidePi,
                                "halogen_bond": d.halogenBond,
                                "chalcogen_bond": d.chalcogenBond,
                                "metal_coord": d.metalCoord,
                                "coulomb": d.coulomb,
                                "ch_pi": d.chPi,
                                "torsion_strain": d.torsionStrain,
                                "cooperativity": d.cooperativity,
                                "hbond_dir": d.hbondDir,
                                "desolv_polar": d.desolvPolar,
                                "desolv_hydrophobic": d.desolvHydrophobic,
                                "total": d.total,
                            ]
                        }
                    }

                    // GFN2-xTB single-point scoring (optional, ~2ms per complex)
                    if enableGFN2Scoring {
                        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
                        if heavyAtoms.count >= 2, docked.count == heavyAtoms.count {
                            var spAtoms = heavyAtoms
                            for j in 0..<spAtoms.count { spAtoms[j].position = docked[j] }
                            let charge = heavyAtoms.reduce(0) { $0 + $1.formalCharge }
                            if let sp = try? await GFN2Refiner.computeEnergy(
                                atoms: spAtoms, totalCharge: charge, solvation: .water
                            ) {
                                entry.gfn2Energy = sp.totalEnergy_kcal
                                entry.gfn2DispersionEnergy = sp.dispersionEnergy * 627.509
                                entry.gfn2SolvationEnergy = sp.solvationEnergy * 627.509
                            }
                        }
                    }

                    let rmsd = entry.bestRmsd.map { String(format: "%.2f", $0) } ?? "N/A"
                    let gfn2Str = entry.gfn2Energy.map { String(format: "  GFN2=%.0f", $0) } ?? ""
                    let debugStr = Self.cfgDebug
                        ? "  pocket=\(pocketMethod) dist=\(String(format: "%.1f", pocketDist))A vol=\(String(format: "%.0f", pocket.volume))A3"
                        : ""
                    let scoreStr: String
                    if scoringMethod.isAffinityScore {
                        let pKd = best.displayScore(method: scoringMethod)
                        let conf = Int(best.afConfidence * 100)
                        scoreStr = "pKi=\(String(format: "%.2f", pKd))  conf=\(conf)%"
                    } else {
                        scoreStr = "E=\(String(format: "%.2f", best.energy))"
                    }
                    emit("  [\(idx+1)/\(total)] \(complex.pdbId)  \(scoreStr)  RMSD=\(rmsd)A\(gfn2Str)\(debugStr)  \(String(format: "%.0f", elapsed))ms")
                    succeeded += 1
                } else {
                    entry.error = "0 poses"
                    entry.dockingTimeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    failed += 1
                }

            } catch let e as BenchmarkError {
                entry.error = e.message
                entry.dockingTimeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                failed += 1
                emit("  [\(idx+1)/\(total)] \(complex.pdbId)  FAILED: \(e.message)")
            } catch {
                entry.error = "\(error)"
                entry.dockingTimeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                failed += 1
                emit("  [\(idx+1)/\(total)] \(complex.pdbId)  FAILED: \(error)")
            }

            bench.entries.append(entry)
            if (idx + 1) % 10 == 0 || idx == total - 1 {
                writeResults(bench, to: resultsPath)
            }
        }

        // --- Summary ---
        printSummary(label: label, bench: bench, total: total,
                     succeeded: succeeded, failed: failed, resultsPath: resultsPath)
        writeResults(bench, to: resultsPath)
    }

    private func printSummary(label: String, bench: BenchmarkResults,
                              total: Int, succeeded: Int, failed: Int, resultsPath: String) {
        let ok = bench.entries.filter(\.success)
        let rmsds = ok.compactMap(\.bestRmsd)
        let below2 = rmsds.filter { $0 < 2.0 }.count
        let below3 = rmsds.filter { $0 < 3.0 }.count
        let below5 = rmsds.filter { $0 < 5.0 }.count

        print("\n=== \(label) Summary ===")
        print("  \(succeeded)/\(total) succeeded, \(failed) failed")

        if !rmsds.isEmpty {
            let mean = rmsds.reduce(0, +) / Float(rmsds.count)
            let sorted = rmsds.sorted()
            let median = sorted[sorted.count / 2]
            print("\n  Docking Power:")
            print("    RMSD < 2.0A: \(below2)/\(rmsds.count) (\(String(format: "%.1f", Float(below2)/Float(rmsds.count)*100))%)")
            print("    RMSD < 3.0A: \(below3)/\(rmsds.count) (\(String(format: "%.1f", Float(below3)/Float(rmsds.count)*100))%)")
            print("    RMSD < 5.0A: \(below5)/\(rmsds.count) (\(String(format: "%.1f", Float(below5)/Float(rmsds.count)*100))%)")
            print("    Mean: \(String(format: "%.2f", mean))A  Median: \(String(format: "%.2f", median))A")
        }

        let scored = ok.filter { $0.experimentalPKd != nil }
        if scored.count >= 3 {
            print("\n  Scoring Power:")

            // Vina/Drusina energy correlation
            let r = pearsonR(scored.map { $0.experimentalPKd! }, scored.map(\.bestEnergy))
            print("    Pearson r (score vs pKd): \(String(format: "%.4f", r))  (n=\(scored.count))")

            // GFN2 correlation if available
            let gfn2Scored = scored.filter { $0.gfn2Energy != nil }
            if gfn2Scored.count > 10 {
                let rGFN2 = pearsonR(gfn2Scored.map { $0.experimentalPKd! },
                                      gfn2Scored.map { $0.gfn2Energy! })
                print("    Pearson r (GFN2 vs pKd): \(String(format: "%.4f", rGFN2))  (n=\(gfn2Scored.count))")
            }
        }

        let totalMs = bench.entries.reduce(0.0) { $0 + $1.dockingTimeMs }
        print("\n  Performance: \(String(format: "%.0f", totalMs/Double(total)))ms/complex, \(String(format: "%.1f", totalMs/1000))s total")
        print("  Results: \(resultsPath)")
    }

    // ==========================================================================
    // MARK: - File-Driven Configuration
    // ==========================================================================

    /// Benchmark config written by run_benchmark.py as Benchmark/.bench_config.json.
    /// Using a file avoids the xcodebuild test-host env var propagation issue.
    private static var _cachedFileConfig: [String: Any]?

    private static func loadFileConfig() -> [String: Any] {
        if let cached = _cachedFileConfig { return cached }

        // Locate config relative to this source file (works from both xcodebuild and Xcode)
        let configPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // Benchmark/
            .appendingPathComponent(".bench_config.json")

        if let data = try? Data(contentsOf: configPath),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            _cachedFileConfig = dict
            print("[Benchmark] Config loaded from \(configPath.lastPathComponent)")

            // Enable stdout log mirroring if configured
            if dict["stdoutLogs"] as? Bool == true {
                Task { @MainActor in
                    ActivityLog.shared.enableStdoutMirroring()
                }
            }

            return dict
        }

        print("[Benchmark] No .bench_config.json found — using defaults")
        _cachedFileConfig = [:]
        return [:]
    }

    /// Read DockingConfig from Benchmark/.bench_config.json.
    /// Falls back to sensible benchmark defaults when config file is absent.
    private static func configFromFile(base: DockingConfig) -> DockingConfig {
        var c = base
        let cfg = loadFileConfig()

        func cfgInt(_ key: String) -> Int? { cfg[key] as? Int }
        func cfgFloat(_ key: String) -> Float? { (cfg[key] as? Double).map(Float.init) }
        func cfgBool(_ key: String) -> Bool? { cfg[key] as? Bool }

        // GA parameters
        c.populationSize = cfgInt("population") ?? 200
        c.generationsPerRun = cfgInt("generations") ?? 200
        c.numRuns = cfgInt("runs") ?? 1
        c.gridSpacing = cfgFloat("gridSpacing") ?? 0.375
        c.mutationRate = cfgFloat("mutationRate") ?? c.mutationRate
        c.mcTemperature = cfgFloat("mcTemperature") ?? c.mcTemperature
        c.autoMode = cfgBool("autoMode") ?? false

        // Local search
        c.localSearchFrequency = cfgInt("localSearchFreq") ?? 3
        c.localSearchSteps = cfgInt("localSearchSteps") ?? 30
        c.useAnalyticalGradients = cfgBool("analyticalGradients") ?? true
        c.liveUpdateFrequency = 100  // always high for benchmarks (fewer GPU syncs)

        // Flexibility
        c.enableFlexibility = cfgBool("ligandFlex") ?? true
        c.flexRefinementSteps = cfgInt("flexRefineSteps") ?? 50

        // Strain penalty
        c.strainPenaltyEnabled = cfgBool("strainPenalty") ?? true
        c.strainPenaltyThreshold = cfgFloat("strainThreshold") ?? 6.0
        c.strainPenaltyWeight = cfgFloat("strainWeight") ?? 0.5

        // Exploration
        c.explorationPhaseRatio = cfgFloat("explorationRatio") ?? c.explorationPhaseRatio
        c.explorationTranslationStep = cfgFloat("explorationTranslation") ?? c.explorationTranslationStep
        c.explorationRotationStep = cfgFloat("explorationRotation") ?? c.explorationRotationStep

        // Reranking
        c.explicitRerankTopClusters = cfgInt("rerankTop") ?? 12
        c.explicitRerankVariantsPerCluster = cfgInt("rerankVariants") ?? 4

        // GFN2-xTB refinement
        c.gfn2Refinement.enabled = cfgBool("gfn2Opt") ?? false
        c.gfn2Refinement.topPosesToRefine = cfgInt("gfn2TopPoses") ?? 20
        c.gfn2Refinement.blendWeight = cfgFloat("gfn2BlendWeight") ?? 0.3
        c.gfn2Refinement.maxSteps = Int32(cfgInt("gfn2MaxSteps") ?? 0)
        if let solvStr = cfg["gfn2Solvation"] as? String {
            switch solvStr {
            case "none":  c.gfn2Refinement.solvation = .none
            case "gbsa":  c.gfn2Refinement.solvation = .gbsa
            default:      c.gfn2Refinement.solvation = .water
            }
        }
        if let optStr = cfg["gfn2OptLevel"] as? String {
            switch optStr {
            case "crude": c.gfn2Refinement.optLevel = .crude
            case "tight": c.gfn2Refinement.optLevel = .tight
            default:      c.gfn2Refinement.optLevel = .normal
            }
        }

        return c
    }

    /// Read max complexes from config file (0 = all).
    private static var cfgMaxComplexes: Int {
        (loadFileConfig()["maxComplexes"] as? Int) ?? 0
    }

    /// Read output directory from config file.
    private static var cfgOutputDirectory: String? {
        loadFileConfig()["outputDir"] as? String
    }

    /// Read manifest path from config file.
    private static var cfgManifestPath: String? {
        guard let path = loadFileConfig()["manifestPath"] as? String,
              !path.isEmpty else { return nil }
        return path
    }

    /// Read benchmark label override from config file.
    private static var cfgBenchmarkName: String? {
        guard let name = loadFileConfig()["benchmarkName"] as? String,
              !name.isEmpty else { return nil }
        return name
    }

    /// Read pipeline mode from config file.
    private static var cfgPipelineMode: String {
        (loadFileConfig()["pipelineMode"] as? String) ?? "redock"
    }

    /// Read pocket mode from config file.
    private static var cfgPocketMode: String {
        (loadFileConfig()["pocketMode"] as? String) ?? "hybrid"
    }

    /// Read receptor flexibility flag from config file.
    private static var cfgFlexResidues: Bool {
        (loadFileConfig()["flexResidues"] as? Bool) ?? false
    }

    /// Read receptor charge method from config file.
    private static var cfgChargeMethod: ChargeMethod {
        guard let raw = (loadFileConfig()["chargeMethod"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() else {
            return .gasteiger
        }

        switch raw {
        case "eem":
            return .eem
        case "qeq":
            return .qeq
        case "xtb":
            return .xtb
        default:
            return .gasteiger
        }
    }

    /// Read GFN2 scoring flag from config file.
    private static var cfgGFN2Scoring: Bool {
        (loadFileConfig()["gfn2Scoring"] as? Bool) ?? false
    }

    /// Read debug diagnostics flag from config file.
    private static var cfgDebug: Bool {
        (loadFileConfig()["debug"] as? Bool) ?? false
    }

    /// Read outputFile from .bench_config.json (set by run_benchmark.py per scoring method).
    /// Falls back to a versioned+timestamped name if not set.
    private static func cfgOutputFile(fallbackMethod: String) -> String {
        if let f = loadFileConfig()["outputFile"] as? String, !f.isEmpty {
            return f
        }
        // Fallback: version_timestamp format
        let version = (try? String(contentsOfFile: URL(fileURLWithPath: #file)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("VERSION").path, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
        let ts = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: "-", with: "")
        return "casf_\(fallbackMethod)_\(version)_\(ts).json"
    }

    // ==========================================================================
    // MARK: - Test Methods (one per scoring method)
    // ==========================================================================

    /// Dock with Vina scoring (baseline grid-based Vina).
    /// Reads config from .bench_config.json (written by run_benchmark.py).
    @MainActor
    func testCASF_Vina() async throws {
        try await runBenchmark(
            scoringMethod: .vina,
            label: "Vina",
            outputFile: Self.cfgOutputFile(fallbackMethod: "vina"),
            maxComplexes: Self.cfgMaxComplexes,
            enableGFN2Scoring: Self.cfgGFN2Scoring
        )
    }

    /// Dock with Drusina scoring (Vina + pi-pi, pi-cation, halogen, metal).
    @MainActor
    func testCASF_Drusina() async throws {
        try await runBenchmark(
            scoringMethod: .drusina,
            label: "Drusina",
            outputFile: Self.cfgOutputFile(fallbackMethod: "drusina"),
            maxComplexes: Self.cfgMaxComplexes,
            enableGFN2Scoring: Self.cfgGFN2Scoring
        )
    }

    /// Dock with Druse Affinity scoring (ML-driven GA).
    @MainActor
    func testCASF_DruseAF() async throws {
        try await runBenchmark(
            scoringMethod: .druseAffinity,
            label: "DruseAF",
            outputFile: Self.cfgOutputFile(fallbackMethod: "druseaf"),
            maxComplexes: Self.cfgMaxComplexes,
            enableGFN2Scoring: Self.cfgGFN2Scoring
        )
    }

    /// Dock with PIGNet2 physics-informed GNN scoring (native Metal).
    @MainActor
    func testCASF_PIGNet2() async throws {
        try await runBenchmark(
            scoringMethod: .pignet2,
            label: "PIGNet2",
            outputFile: Self.cfgOutputFile(fallbackMethod: "pignet2"),
            maxComplexes: Self.cfgMaxComplexes,
            enableGFN2Scoring: Self.cfgGFN2Scoring
        )
    }

    /// Dock with Drusina + GFN2-xTB single-point scoring (D4 dispersion + ALPB solvation).
    @MainActor
    func testCASF_DrusinaGFN2() async throws {
        try await runBenchmark(
            scoringMethod: .drusina,
            label: "Drusina+GFN2",
            outputFile: Self.cfgOutputFile(fallbackMethod: "drusina_gfn2"),
            maxComplexes: Self.cfgMaxComplexes,
            enableGFN2Scoring: true
        )
    }

    /// Run ALL scoring methods sequentially (full overnight benchmark).
    @MainActor
    func testCASF_All() async throws {
        let n = Self.cfgMaxComplexes
        let suffix = n > 0 ? " (first \(n))" : ""
        print("=== Full CASF-2016 Benchmark (all scoring methods)\(suffix) ===\n")
        let t0 = CFAbsoluteTimeGetCurrent()

        try await runBenchmark(scoringMethod: .vina, label: "Vina", outputFile: Self.cfgOutputFile(fallbackMethod: "vina"), maxComplexes: n)
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .drusina, label: "Drusina", outputFile: Self.cfgOutputFile(fallbackMethod: "drusina"), maxComplexes: n)
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .druseAffinity, label: "DruseAF", outputFile: Self.cfgOutputFile(fallbackMethod: "druseaf"), maxComplexes: n)
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .pignet2, label: "PIGNet2", outputFile: Self.cfgOutputFile(fallbackMethod: "pignet2"), maxComplexes: n)
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .drusina, label: "Drusina+GFN2", outputFile: Self.cfgOutputFile(fallbackMethod: "drusina_gfn2"), maxComplexes: n, enableGFN2Scoring: true)

        let total = CFAbsoluteTimeGetCurrent() - t0
        print("\n=== Full benchmark completed in \(String(format: "%.1f", total / 60)) minutes ===")
    }

    // ==========================================================================
    // MARK: - Quick Validation (first N complexes)
    // ==========================================================================

    /// Quick Vina vs Drusina comparison on first N complexes.
    @MainActor
    func testCASF_Quick(n: Int = 10) async throws {
        print("=== Quick Validation: Vina vs Drusina (first \(n)) ===\n")
        let t0 = CFAbsoluteTimeGetCurrent()

        try await runBenchmark(scoringMethod: .vina, label: "Vina", outputFile: Self.cfgOutputFile(fallbackMethod: "vina_quick"), maxComplexes: n)
        print("\n" + String(repeating: "-", count: 50) + "\n")
        try await runBenchmark(scoringMethod: .drusina, label: "Drusina", outputFile: Self.cfgOutputFile(fallbackMethod: "drusina_quick"), maxComplexes: n)

        let total = CFAbsoluteTimeGetCurrent() - t0
        print("\n=== Quick validation completed in \(String(format: "%.1f", total / 60)) minutes ===")
    }
}

// MARK: - Error

private enum BenchmarkError: Error {
    case step(String)
    var message: String {
        switch self { case .step(let s): return s }
    }
}

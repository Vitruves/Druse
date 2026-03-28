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
        let path = resolve("Benchmark/manifests/casf_manifest.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw XCTSkip("Manifest not found. Run:\n  python Benchmark/download.py --setup\n  python Benchmark/prepare.py")
        }
        return try JSONDecoder().decode(CASFManifest.self, from: data)
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

        let resultsPath = resolve("Benchmark/results/\(outputFile)")
        let version = (try? String(contentsOfFile: resolve("VERSION"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
        var bench = BenchmarkResults(
            benchmark: "casf-2016",
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
                bestEnergy: .infinity,
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
                        rawPDBContent: pdbContent, pH: 7.4
                    )
                }.value
                let protein = Molecule(name: rawProtein.name,
                                       atoms: prepared.atoms, bonds: prepared.bonds,
                                       title: rawProtein.title)

                // 2. Load crystal ligand from SDF (preserves atom order for RMSD)
                guard let crystalLig = loadCrystalLigand(sdfPath: resolve(complex.ligandSdf)) else {
                    throw BenchmarkError.step("parse crystal ligand SDF")
                }

                // 3. Pocket from crystal ligand position
                guard let pocket = BindingSiteDetector.ligandGuidedPocket(
                    protein: protein, ligand: crystalLig, distance: 6.0
                ) else {
                    throw BenchmarkError.step("pocket detection")
                }

                // 4. Use crystal ligand directly for docking (preserves atom order for RMSD).
                //    Charges are computed by the docking engine's Vina atom typing.
                //    This matches standard redocking benchmarks (Vina, Glide, GOLD).
                let ligand = crystalLig

                // 5. Set protein on engine (required for grid map computation)
                eng.setProtein(protein.atoms, protein.bonds)

                // 6. Dock (DruseAF runs its own ML scoring on Metal during the GA)
                let gaResults = await eng.runDocking(
                    ligand: ligand, pocket: pocket,
                    config: config, scoringMethod: scoringMethod
                )

                let results = gaResults

                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                entry.numPoses = results.count
                entry.dockingTimeMs = elapsed

                if let best = results.first {
                    entry.bestEnergy = best.energy
                    entry.success = true

                    // RMSD
                    let crystal = complex.crystalPositionsSIMD
                    let docked = best.transformedAtomPositions
                    if crystal.count == docked.count {
                        entry.bestRmsd = computeRMSD(crystal, docked)
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
                    emit("  [\(idx+1)/\(total)] \(complex.pdbId)  E=\(String(format: "%.2f", best.energy))  RMSD=\(rmsd)A\(gfn2Str)  \(String(format: "%.0f", elapsed))ms")
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

    /// Read GFN2 scoring flag from config file.
    private static var cfgGFN2Scoring: Bool {
        (loadFileConfig()["gfn2Scoring"] as? Bool) ?? false
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

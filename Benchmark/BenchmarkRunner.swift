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
//
// Each writes its own results JSON. The Python analyzer compares all three.
//
// Prerequisites:
//   python Benchmark/download.py --setup
//   python Benchmark/prepare.py

final class BenchmarkRunner: XCTestCase {

    // MARK: - Shared Engine

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
        maxComplexes: Int = 0
    ) async throws {
        let manifest = try loadManifest()
        let complexes = maxComplexes > 0
            ? Array(manifest.complexes.prefix(maxComplexes))
            : manifest.complexes
        let suffix = maxComplexes > 0 ? " (first \(complexes.count))" : ""
        print("[\(label)] \(complexes.count) complexes\(suffix)")

        let eng = try engine()

        var config = DockingConfig()
        config.populationSize = 200
        config.generationsPerRun = 200
        config.numRuns = 1
        config.gridSpacing = 0.375
        config.localSearchFrequency = 3
        config.liveUpdateFrequency = 100

        let resultsPath = resolve("Benchmark/results/\(outputFile)")
        var bench = BenchmarkResults(
            benchmark: "casf-2016",
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
                mlRescorePKd: nil,
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
                let rawAtomCount = rawProtein.atoms.count
                print("    protein: \(rawAtomCount) raw atoms")
                fflush(stdout)

                let pdbContent = try? String(contentsOfFile: resolve(complex.proteinPdb), encoding: .utf8)
                let rawAtoms = rawProtein.atoms
                let rawBonds = rawProtein.bonds
                let prepared = await Task.detached { @Sendable in
                    ProteinPreparation.prepareForDocking(
                        atoms: rawAtoms, bonds: rawBonds,
                        rawPDBContent: pdbContent, pH: 7.4
                    )
                }.value
                print("    prepared: \(prepared.atoms.count) atoms (+\(prepared.report.hydrogensAdded) H)")
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

                // 5. Grid maps + dock with the specified scoring method
                print("    grid maps...")
                eng.computeGridMaps(protein: protein, pocket: pocket, spacing: config.gridSpacing)
                print("    docking (pop=\(config.populationSize) gen=\(config.generationsPerRun))...")
                let results = await eng.runDocking(
                    ligand: ligand, pocket: pocket,
                    config: config, scoringMethod: scoringMethod
                )
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

                    let rmsd = entry.bestRmsd.map { String(format: "%.2f", $0) } ?? "N/A"
                    print("  [\(idx+1)/\(total)] \(complex.pdbId)  E=\(String(format: "%.2f", best.energy))  RMSD=\(rmsd)A  \(String(format: "%.0f", elapsed))ms")
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
                print("  [\(idx+1)/\(total)] \(complex.pdbId)  FAILED: \(e.message)")
            } catch {
                entry.error = "\(error)"
                entry.dockingTimeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                failed += 1
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
        if scored.count > 10 {
            let r = pearsonR(scored.map { $0.experimentalPKd! }, scored.map(\.bestEnergy))
            print("\n  Scoring Power:")
            print("    Pearson r (score vs pKd): \(String(format: "%.4f", r))  (n=\(scored.count))")
        }

        let totalMs = bench.entries.reduce(0.0) { $0 + $1.dockingTimeMs }
        print("\n  Performance: \(String(format: "%.0f", totalMs/Double(total)))ms/complex, \(String(format: "%.1f", totalMs/1000))s total")
        print("  Results: \(resultsPath)")
    }

    // ==========================================================================
    // MARK: - Test Methods (one per scoring method)
    // ==========================================================================

    /// Dock with Vina scoring (baseline grid-based Vina).
    @MainActor
    func testCASF_Vina() async throws {
        try await runBenchmark(
            scoringMethod: .vina,
            label: "Vina",
            outputFile: "casf_vina.json"
        )
    }

    /// Dock with Drusina scoring (Vina + pi-pi, pi-cation, halogen, metal).
    @MainActor
    func testCASF_Drusina() async throws {
        try await runBenchmark(
            scoringMethod: .drusina,
            label: "Drusina",
            outputFile: "casf_drusina.json"
        )
    }

    /// Dock with Druse Affinity scoring (ML-driven GA).
    @MainActor
    func testCASF_DruseAF() async throws {
        try await runBenchmark(
            scoringMethod: .druseAffinity,
            label: "DruseAF",
            outputFile: "casf_druseaf.json"
        )
    }

    /// Run ALL three scoring methods sequentially (full overnight benchmark).
    @MainActor
    func testCASF_All() async throws {
        print("=== Full CASF-2016 Benchmark (all 3 scoring methods) ===\n")
        let t0 = CFAbsoluteTimeGetCurrent()

        try await runBenchmark(scoringMethod: .vina, label: "Vina", outputFile: "casf_vina.json")
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .drusina, label: "Drusina", outputFile: "casf_drusina.json")
        print("\n" + String(repeating: "=", count: 60) + "\n")
        try await runBenchmark(scoringMethod: .druseAffinity, label: "DruseAF", outputFile: "casf_druseaf.json")

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

        try await runBenchmark(scoringMethod: .vina, label: "Vina", outputFile: "casf_vina_quick.json", maxComplexes: n)
        print("\n" + String(repeating: "-", count: 50) + "\n")
        try await runBenchmark(scoringMethod: .drusina, label: "Drusina", outputFile: "casf_drusina_quick.json", maxComplexes: n)

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

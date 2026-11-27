// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import XCTest
import simd
import MetalKit
@testable import Druse

/// GPU pipeline tests — requires Metal, no network. ~25s total.
/// Tests the full docking pipeline using inline molecular data (alanine dipeptide + SMILES ligands)
/// with tuned-down GA parameters for fast execution while still validating correctness.
final class PipelineTests: DockingTestCase {

    func testPopulationStartsInsideTranslatedSearchBox() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let library = try device.makeDefaultLibrary(bundle: Bundle(for: DockingEngine.self))
        let kernel = try XCTUnwrap(library.makeFunction(name: "initializePopulation"))
        let pipeline = try device.makeComputePipelineState(function: kernel)
        var grid = GridParams()
        grid.searchCenter = SIMD3(31, -23, 11)
        grid.searchHalfExtent = SIMD3(5, 7, 11)
        var params = GAParams()
        params.populationSize = 256
        params.runSeed = 7
        let count = Int(params.populationSize)
        let buffer = try XCTUnwrap(device.makeBuffer(length: count * MemoryLayout<DockPose>.stride, options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&grid, length: MemoryLayout<GridParams>.stride, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<GAParams>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let poses = buffer.contents().assumingMemoryBound(to: DockPose.self)
        for index in 0..<count {
            let delta = simd_abs(poses[index].translation - grid.searchCenter)
            XCTAssertLessThanOrEqual(simd_reduce_max(delta - grid.searchHalfExtent), 0)
        }
    }

    func testReceptorGridsRespectAtomTypesAndDistanceCutoffs() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let library = try device.makeDefaultLibrary(bundle: Bundle(for: DockingEngine.self))
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "computeReceptorGrids")))
        var carbon = GridProteinAtom()
        carbon.vdwRadius = 1.5
        carbon.charge = 0.1
        carbon.vinaType = Int32(VINA_C_H.rawValue)
        var oxygen = carbon
        oxygen.vinaType = Int32(VINA_O_A.rawValue)
        var distant = carbon
        distant.position = SIMD3(repeating: 200)
        let atoms = [carbon] + Array(repeating: distant, count: 127) + [oxygen]
        var params = GridParams()
        params.dims = SIMD3(14, 1, 1)
        params.spacing = 1
        params.totalPoints = params.dims.x
        params.numProteinAtoms = UInt32(atoms.count)
        let outputs = try (0..<4).map { _ in try XCTUnwrap(device.makeBuffer(length: Int(params.totalPoints) * 2, options: .storageModeShared)) }
        let receptor = try XCTUnwrap(device.makeBuffer(bytes: atoms, length: atoms.count * MemoryLayout<GridProteinAtom>.stride))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(receptor, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<GridParams>.stride, index: 2)
        for (output, slot) in zip(outputs, [0, 3, 4, 5]) { encoder.setBuffer(output, offset: 0, index: slot) }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        for index in 0..<Int(params.totalPoints) {
            let radius = Float(index)
            let gap = max(radius, 0.1) - 3
            let near = 2 * gap, far = (gap - 3) / 2, overlap = min(gap, 0)
            let steric: Float = -0.035579 * exp(-near * near) - 0.005156 * exp(-far * far) + 0.840245 * overlap * overlap
            let expected: [Float] = [
                radius <= 8 ? 2 * steric : 0,
                radius <= 8 ? -0.035069 * min(1, max(0, 1.5 - gap)) : 0,
                radius <= 8 ? -0.587439 * min(1, max(0, -gap / 0.7)) : 0,
                radius <= 12 ? 16.6 / pow(max(radius, 0.8), 2) : 0,
            ]
            for (output, value) in zip(outputs, expected) {
                let actual = output.contents().assumingMemoryBound(to: Float16.self)[index]
                XCTAssertEqual(Float(actual), value, accuracy: max(1e-6, abs(value) * 0.001))
            }
        }
    }

    func testReplicaExchangeUsesEachWalkersEnergyAndAlternatesRounds() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let library = try device.makeDefaultLibrary(bundle: Bundle(for: DockingEngine.self))
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "replicaSwap")))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let energies: [Float] = [10000, 0, 0, 10000, 0, 0]
        let poses = energies.enumerated().map { index, energy in
            var pose = DockPose()
            pose.generation = Int32(index)
            pose.energy = energy
            return pose
        }
        let bytes = poses.count * MemoryLayout<DockPose>.stride
        let current = try XCTUnwrap(device.makeBuffer(bytes: poses, length: bytes, options: .storageModeShared))
        let best = try XCTUnwrap(device.makeBuffer(bytes: poses, length: bytes, options: .storageModeShared))
        var replicas = ReplicaParams()
        replicas.numReplicas = 3
        replicas.populationPerReplica = 2
        replicas.totalPoses = 6
        replicas.temperatures.0 = 1
        replicas.temperatures.1 = 2
        replicas.temperatures.2 = 3
        replicas.swapInterval = 4
        func exchange(generation: UInt32, expected: [Int32]) throws {
            var params = GAParams()
            params.generation = generation
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(current, offset: 0, index: 0)
            encoder.setBuffer(best, offset: 0, index: 1)
            encoder.setBytes(&replicas, length: MemoryLayout<ReplicaParams>.stride, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<GAParams>.stride, index: 3)
            let threads = MTLSize(width: 2, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            XCTAssertNil(command.error)
            for buffer in [current, best] {
                let output = buffer.contents().assumingMemoryBound(to: DockPose.self)
                XCTAssertEqual(poses.indices.map { output[$0].generation }, expected)
            }
        }
        try exchange(generation: 3, expected: [2, 1, 0, 3, 4, 5])
        try exchange(generation: 7, expected: [2, 1, 4, 5, 0, 3])
    }

    // MARK: - GA Docking: Core Validation

    /// Full GA pipeline with aspirin — validates results are non-empty, finite, sorted,
    /// and have valid atom positions.
    @MainActor
    func testGADockingProducesValidResults() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, flexibility: false)

        XCTAssertFalse(results.isEmpty, "GA docking produced no results")

        for (i, r) in results.enumerated() {
            XCTAssertTrue(r.energy.isFinite, "Pose \(i) energy not finite: \(r.energy)")
            XCTAssertTrue(r.stericEnergy.isFinite, "Pose \(i) steric not finite")
            XCTAssertTrue(r.hydrophobicEnergy.isFinite, "Pose \(i) hydro not finite")
            XCTAssertTrue(r.hbondEnergy.isFinite, "Pose \(i) hbond not finite")
        }
        // Sorted by energy
        for i in 1..<min(results.count, 20) {
            XCTAssertLessThanOrEqual(results[i-1].energy, results[i].energy,
                "Results not sorted at \(i)")
        }
        // Best pose sanity
        let best = results[0]
        XCTAssertFalse(best.transformedAtomPositions.isEmpty, "Best pose has no positions")
        XCTAssertLessThan(best.energy, 5.0, "Best energy should be reasonable")
        for pos in best.transformedAtomPositions {
            XCTAssertFalse(pos.x.isNaN || pos.y.isNaN || pos.z.isNaN, "Position NaN: \(pos)")
        }
    }

    /// Caffeine: 14 heavy atoms, 0 rotatable bonds — tests rigid-body docking path.
    @MainActor
    func testRigidDocking() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", ligandName: "Caffeine")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 48, generations: 40, flexibility: false)

        XCTAssertFalse(results.isEmpty, "Rigid docking produced no results")
        XCTAssertTrue(results[0].energy.isFinite)
    }

    /// Ibuprofen: 13 heavy atoms, 4+ rotatable bonds — tests torsion flexibility.
    @MainActor
    func testFlexibleDocking() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", ligandName: "Ibuprofen")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, flexibility: true)

        XCTAssertFalse(results.isEmpty, "Flexible docking produced no results")
        XCTAssertTrue(results[0].energy.isFinite)
        XCTAssertLessThan(results[0].energy, 5.0)
    }

    // MARK: - REMC (Parallel Tempering)

    @MainActor
    func testREMCDocking() async throws {
        let (engine, _, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "c1ccccc1", ligandName: "Benzene")

        var config = DockingConfig()
        config.explicitRerankTopClusters = 1
        config.randomSeed = 19
        config.searchMethod = .parallelTempering
        config.populationSize = 32
        config.generationsPerRun = 30
        config.numRuns = 1
        config.localSearchSteps = 5
        config.liveUpdateFrequency = 999
        config.replicaExchange.numReplicas = 4
        config.replicaExchange.stepsPerReplica = 30

        let results = await engine.runDocking(
            ligand: ligand, pocket: pocket, config: config, scoringMethod: .vina)

        let best = try XCTUnwrap(results.first)
        XCTAssertGreaterThan(Set(results.map(\.clusterID)).count, 1)
        XCTAssertEqual(results.map(\.energy), results.map(\.energy).sorted())
        for r in results {
            XCTAssertLessThanOrEqual(r.energy - best.energy, DockingEngine.outputEnergyRangeKcal + 1e-5)
            // Rigid benzene has no rotatable bonds. Every mode, including those outside
            // the refinement budget, must carry the explicit intermolecular score.
            XCTAssertEqual(r.energy, r.stericEnergy + r.hydrophobicEnergy + r.hbondEnergy, accuracy: 1e-4)
        }
    }

    // MARK: - Drusina vs Vina Scoring

    /// Verify both scoring methods produce valid, favorable energies on the same system.
    @MainActor
    func testDrusinaScoringVsVina() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let vinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, scoringMethod: .vina)
        let drusinaResults = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, scoringMethod: .drusina)

        XCTAssertFalse(vinaResults.isEmpty, "Vina should produce results")
        XCTAssertFalse(drusinaResults.isEmpty, "Drusina should produce results")
        XCTAssertLessThan(vinaResults[0].energy, 5.0, "Vina best energy should be reasonable")
        XCTAssertLessThan(drusinaResults[0].energy, 5.0, "Drusina best energy should be reasonable")

        for (i, r) in drusinaResults.prefix(10).enumerated() {
            XCTAssertTrue(r.energy.isFinite, "Drusina pose \(i) energy not finite: \(r.energy)")
        }

        print("  [Scoring] Vina: \(String(format: "%.2f", vinaResults[0].energy)), " +
              "Drusina: \(String(format: "%.2f", drusinaResults[0].energy))")
    }

    // MARK: - Grid & Pose Validation

    /// Verify all poses stay inside the grid box and centroids are near the pocket center.
    @MainActor
    func testGridBoundsAndPoseCentering() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 48, generations: 40)

        // Grid bounds check: ≤5% atoms outside padded box
        let padding: Float = 8.0
        let boxMin = pocket.center - pocket.size - SIMD3(repeating: padding)
        let boxMax = pocket.center + pocket.size + SIMD3(repeating: padding)
        var outsideCount = 0
        var totalAtoms = 0
        for r in results {
            for pos in r.transformedAtomPositions {
                totalAtoms += 1
                if pos.x < boxMin.x || pos.x > boxMax.x ||
                   pos.y < boxMin.y || pos.y > boxMax.y ||
                   pos.z < boxMin.z || pos.z > boxMax.z {
                    outsideCount += 1
                }
            }
        }
        let frac = Float(outsideCount) / Float(max(totalAtoms, 1))
        XCTAssertLessThan(frac, 0.05,
            "\(outsideCount)/\(totalAtoms) atoms outside grid (\(String(format: "%.1f%%", frac * 100)))")

        // Pose centering check
        let maxR = simd_length(pocket.size + SIMD3<Float>(repeating: 2.0)) + 10.0
        for (i, r) in results.prefix(5).enumerated() {
            guard !r.transformedAtomPositions.isEmpty else { continue }
            let centroid = r.transformedAtomPositions.reduce(.zero, +) /
                Float(r.transformedAtomPositions.count)
            let dist = simd_distance(centroid, pocket.center)
            XCTAssertLessThan(dist, maxR,
                "Pose \(i) centroid \(String(format: "%.1f", dist))Å from pocket")
        }
    }

    // MARK: - Chemical Validity

    /// Verify rigid bond lengths are preserved after docking (hexane: only single bonds).
    @MainActor
    func testBondLengthPreservation() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CCCCCC", ligandName: "Hexane")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, flexibility: true)

        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        let hBonds = heavyBonds(from: ligand.atoms, bonds: ligand.bonds)

        for (pi, r) in results.prefix(3).enumerated() {
            let pos = r.transformedAtomPositions
            guard pos.count == heavyAtoms.count else { continue }
            for b in hBonds {
                guard b.atomIndex1 < pos.count, b.atomIndex2 < pos.count,
                      b.atomIndex1 < heavyAtoms.count, b.atomIndex2 < heavyAtoms.count else { continue }
                let orig = simd_distance(
                    heavyAtoms[b.atomIndex1].position, heavyAtoms[b.atomIndex2].position)
                guard orig > 0.01 else { continue }
                let docked = simd_distance(pos[b.atomIndex1], pos[b.atomIndex2])
                guard docked.isFinite else { continue }
                XCTAssertEqual(Double(docked), Double(orig), accuracy: 0.15,
                    "Pose \(pi) bond \(b.atomIndex1)-\(b.atomIndex2): " +
                    "\(String(format: "%.3f", orig))→\(String(format: "%.3f", docked))Å")
            }
        }
    }

    /// Verify no severe intra-molecular clashes (<0.5 Å) in docked poses.
    @MainActor
    func testNoIntraMolecularClashes() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, flexibility: true)

        // Build topology-close pairs (already close in input structure)
        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        var topoPairs: Set<String> = []
        for i in 0..<heavyAtoms.count {
            for j in (i+1)..<heavyAtoms.count {
                if simd_distance(heavyAtoms[i].position, heavyAtoms[j].position) < 3.0 {
                    topoPairs.insert("\(i)-\(j)")
                }
            }
        }

        for (pi, r) in results.prefix(3).enumerated() {
            let pos = r.transformedAtomPositions
            var clashes = 0
            for i in 0..<pos.count {
                for j in (i+1)..<pos.count {
                    guard !topoPairs.contains("\(i)-\(j)") else { continue }
                    if simd_distance(pos[i], pos[j]) < 0.5 { clashes += 1 }
                }
            }
            XCTAssertEqual(clashes, 0, "Pose \(pi) has \(clashes) severe clashes (<0.5Å)")
        }
    }

    // MARK: - Interaction Detection

    @MainActor
    func testInteractionDetection() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50)
        guard let best = results.first else { throw XCTSkip("No docking results") }

        let heavyAtoms = ligand.atoms.filter { $0.element != .H }
        let hBonds = heavyBonds(from: ligand.atoms, bonds: ligand.bonds)
        let proteinHeavy = protein.atoms.filter { $0.element != .H }

        let interactions = InteractionDetector.detect(
            ligandAtoms: heavyAtoms, ligandPositions: best.transformedAtomPositions,
            proteinAtoms: proteinHeavy, ligandBonds: hBonds,
            scoringMethod: .drusina)

        XCTAssertGreaterThanOrEqual(interactions.count, 2, "Should detect ≥2 interactions")
        for inter in interactions {
            XCTAssertTrue(inter.distance > 0 && inter.distance < 10.0,
                "Interaction distance \(inter.distance) out of range")
            XCTAssertLessThan(inter.ligandAtomIndex, best.transformedAtomPositions.count)
            XCTAssertLessThan(inter.proteinAtomIndex, proteinHeavy.count)
        }
    }

    // MARK: - SDF Export Round-Trip

    @MainActor
    func testSDFExportRoundTrip() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 32, generations: 30, flexibility: false)
        XCTAssertFalse(results.isEmpty)

        let sdf = SDFWriter.writeDockingResults(results, ligand: ligand)
        XCTAssertTrue(sdf.contains("$$$$"), "Missing SDF delimiter")
        XCTAssertTrue(sdf.contains("V2000"), "Missing V2000 marker")
        XCTAssertTrue(sdf.contains("> <Energy>"), "Missing energy field")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).sdf")
        try SDFWriter.save(sdf, to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let parsed = try SDFParser.parse(url: url)
        XCTAssertEqual(parsed.count, results.count, "Re-parsed count should match")
    }

    // MARK: - Graceful Stop

    @MainActor
    func testStopDocking() async throws {
        let (engine, _, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "c1ccccc1", ligandName: "Benzene")

        var config = DockingConfig()
        config.populationSize = 32
        config.generationsPerRun = 1000  // intentionally long
        config.numRuns = 1
        config.liveUpdateFrequency = 999

        let task = Task { @MainActor in
            await engine.runDocking(
                ligand: ligand, pocket: pocket, config: config, scoringMethod: .vina)
        }
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5s
        engine.stopDocking()

        let results = await task.value
        // Key: doesn't crash. May or may not have partial results.
        XCTAssertFalse(engine.isRunning, "Engine should not be running after stop")
        print("  [Stop] \(results.count) results after early stop")
    }

    // MARK: - Pharmacophore Constrained Docking

    @MainActor
    func testConstrainedDocking() async throws {
        let (engine, protein, ligand, pocket) = try inlineTestSystem(
            ligandSmiles: "CC(=O)Oc1ccccc1C(=O)O", ligandName: "Aspirin")

        // Soft hydrophobic constraint at pocket center — easily satisfiable
        var constraint = PharmacophoreConstraintDef(
            targetScope: .atom,
            interactionType: .hydrophobic,
            strength: .soft(kcalPerAngstromSq: 5.0),
            distanceThreshold: 5.0,
            sourceType: .receptor,
            proteinAtomIndex: 0
        )
        constraint.targetPositions = [pocket.center]

        engine.prepareConstraintBuffers(
            [constraint], atoms: protein.atoms, residues: protein.residues)
        defer { engine.prepareConstraintBuffers([], atoms: [], residues: []) }

        let results = await runTestDocking(
            engine: engine, protein: protein, ligand: ligand, pocket: pocket,
            populationSize: 64, generations: 50, flexibility: false)

        XCTAssertFalse(results.isEmpty, "Constrained docking must produce results")
        for (i, r) in results.prefix(5).enumerated() {
            XCTAssertTrue(r.energy.isFinite, "Pose \(i) energy not finite")
            XCTAssertTrue(r.constraintPenalty.isFinite, "Pose \(i) penalty not finite")
            XCTAssertGreaterThanOrEqual(r.constraintPenalty, 0,
                "Constraint penalty should be ≥0, got \(r.constraintPenalty)")
        }
    }
}

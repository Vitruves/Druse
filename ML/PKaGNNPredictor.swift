// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal

/// GPU-accelerated per-atom pKa prediction using a trained GNN.
///
/// Replaces the SMARTS lookup table + GFN2-xTB pipeline with a single
/// Metal shader dispatch. Takes SMILES → returns ionizable sites with
/// predicted pKa and acid/base classification.
enum PKaGNNPredictor {

    /// Per-atom prediction result.
    struct SitePrediction: Sendable {
        let atomIdx: Int
        let pKa: Double
        let isAcid: Bool
        let ionizableProbability: Float
        let acidProbability: Float
    }

    /// Metal resources (lazy-initialized on first use).
    private nonisolated(unsafe) static var pipeline: MTLComputePipelineState?
    private nonisolated(unsafe) static var weights: PKaGNNWeightLoader.LoadedWeights?
    private nonisolated(unsafe) static var device: MTLDevice?
    private nonisolated(unsafe) static var queue: MTLCommandQueue?
    private nonisolated(unsafe) static var initState: InitState = .pending
    private static let initLock = NSLock()

    private enum InitState { case pending, initialized, failed }

    /// Lazy initialization — called automatically on first predict().
    @discardableResult
    private static func ensureInitialized() -> Bool {
        initLock.lock()
        defer { initLock.unlock() }

        guard initState == .pending else { return initState == .initialized }

        guard let dev = MTLCreateSystemDefaultDevice() else {
            print("[PKaGNN] No Metal device")
            initState = .failed
            return false
        }

        guard let lib = dev.makeDefaultLibrary() else {
            print("[PKaGNN] No Metal library")
            initState = .failed
            return false
        }

        guard let w = PKaGNNWeightLoader.loadFromBundle(device: dev) else {
            print("[PKaGNN] No weights found — GNN pKa prediction unavailable")
            initState = .failed
            return false
        }

        guard let fn = lib.makeFunction(name: "pkaGNNFull"),
              let pso = try? dev.makeComputePipelineState(function: fn) else {
            print("[PKaGNN] Failed to create compute pipeline")
            initState = .failed
            return false
        }

        guard let q = dev.makeCommandQueue() else {
            print("[PKaGNN] Failed to create command queue")
            initState = .failed
            return false
        }

        self.weights = w
        self.pipeline = pso
        self.device = dev
        self.queue = q
        initState = .initialized
        print("[PKaGNN] Initialized — ready for inference")
        return true
    }

    /// Check if the GNN predictor is available.
    static var isAvailable: Bool { ensureInitialized() }

    /// Predict pKa for all atoms in a molecule.
    ///
    /// Returns only atoms with ionizable probability > threshold.
    /// Typical threshold: 0.5
    static func predict(
        smiles: String,
        ionThreshold: Float = 0.5
    ) -> [SitePrediction] {
        guard ensureInitialized(),
              let device = self.device,
              let pipeline = self.pipeline,
              let weights = self.weights,
              let queue = self.queue else {
            return []
        }

        // 1. Featurize via C++ (RDKit)
        guard let graphPtr = druse_featurize_pka_graph(smiles) else { return [] }
        defer { druse_free_pka_graph(graphPtr) }

        let graph = graphPtr.pointee
        guard graph.success, graph.numAtoms > 0 else { return [] }

        let N = Int(graph.numAtoms)
        let E = Int(graph.numEdges)
        let atomFeatDim = 25
        let bondFeatDim = 10

        // 2. Create Metal buffers from heap-allocated C arrays
        guard let atomBuf = device.makeBuffer(
            bytes: graph.atomFeatures, length: N * atomFeatDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return [] }

        guard let bondBuf = device.makeBuffer(
            bytes: graph.bondFeatures, length: E * bondFeatDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return [] }

        guard let srcBuf = device.makeBuffer(
            bytes: graph.edgeSrc, length: E * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else { return [] }

        guard let dstBuf = device.makeBuffer(
            bytes: graph.edgeDst, length: E * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else { return [] }

        // Params buffer
        var params = (UInt32(N), UInt32(E), UInt32(0), UInt32(0))
        guard let paramBuf = device.makeBuffer(
            bytes: &params, length: 16, options: .storageModeShared
        ) else { return [] }

        // Output buffer
        let outputStride = MemoryLayout<Float>.stride * 4  // ionProb, pka, acidProb, pad
        guard let outputBuf = device.makeBuffer(
            length: N * outputStride, options: .storageModeShared
        ) else { return [] }

        // 3. Encode and dispatch
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return [] }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(atomBuf, offset: 0, index: 0)
        encoder.setBuffer(bondBuf, offset: 0, index: 1)
        encoder.setBuffer(srcBuf, offset: 0, index: 2)
        encoder.setBuffer(dstBuf, offset: 0, index: 3)
        encoder.setBuffer(paramBuf, offset: 0, index: 4)
        encoder.setBuffer(weights.weightBuffer, offset: 0, index: 5)
        encoder.setBuffer(weights.entryBuffer, offset: 0, index: 6)
        encoder.setBuffer(outputBuf, offset: 0, index: 7)

        // Single thread dispatch (pkaGNNFull processes all atoms on thread 0)
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // 4. Read results
        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: N * 4)
        var results: [SitePrediction] = []

        for i in 0..<N {
            let ionProb = outPtr[i * 4 + 0]
            let pka = outPtr[i * 4 + 1]
            let acidProb = outPtr[i * 4 + 2]

            if ionProb >= ionThreshold {
                results.append(SitePrediction(
                    atomIdx: i,
                    pKa: Double(pka),
                    isAcid: acidProb > 0.5,
                    ionizableProbability: ionProb,
                    acidProbability: acidProb
                ))
            }
        }

        return results
    }

    /// Convenience: return sites as (atomIdx, pKa, isAcid) tuples compatible
    /// with the ensemble generator's expected input format.
    static func predictForEnsemble(smiles: String) -> (sites: [(atomIdx: Int, pKa: Double, isAcid: Bool)], success: Bool) {
        let predictions = predict(smiles: smiles)
        if predictions.isEmpty {
            return ([], false)
        }
        let sites = predictions.map { ($0.atomIdx, $0.pKa, $0.isAcid) }
        return (sites, true)
    }
}

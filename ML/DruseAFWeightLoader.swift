// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal

/// Loads DruseAF neural network weights from a binary file into Metal buffers.
///
/// Binary format (from export_druseaf_weights.py):
///   Header: "DRAF" magic, version, numTensors, totalFloats, offset table
///   Data:   concatenated float32 weights
///
/// Produces a single weight `MTLBuffer` + a `DruseAFWeightEntry` offset buffer
/// for the Metal shader to index individual tensors.
struct DruseAFWeightLoader {

    struct LoadedWeights {
        let weightBuffer: MTLBuffer       // packed float32 weights
        let entryBuffer: MTLBuffer        // DruseAFWeightEntry array
        let numTensors: Int
        let totalFloats: Int
        let version: Int                  // 1 = v3 cross-attention, 2 = v4 PGN
    }

    /// Load DruseAF weights from the app bundle (prefers v4, falls back to v3).
    static func loadFromBundle(device: MTLDevice) -> LoadedWeights? {
        // Prefer v4 weights, fall back to v3
        for filename in ["DruseAFv4", "DruseAF"] {
            if let url = Bundle.main.url(forResource: filename, withExtension: "weights"),
               let weights = load(from: url, device: device) {
                return weights
            }
        }
        print("[DruseAF] No usable weight file found in bundle (tried DruseAFv4.weights, DruseAF.weights)")
        return nil
    }

    /// Per-tensor float counts read by DruseAFCompute.metal (DRAF v1, 56 tensors).
    static let v3ExpectedCounts: [Int] = {
        let encoder: [Int] = [256 * 20, 256, 256 * 256, 256]
        let crossAttnLayer: [Int] = [256 * 256, 256, 256 * 256, 256, 256 * 256, 256,  // Q, K, V
                                     4 * 50, 4,                                        // RBF bias
                                     256 * 256, 256, 256, 256]                         // out proj, norm
        var counts: [Int] = []
        counts += encoder                                // protein encoder
        counts += encoder                                // ligand encoder
        for _ in 0..<3 { counts += crossAttnLayer }
        counts += [128 * 256, 128, 128, 1]               // gate
        counts += [256 * 256, 256, 256, 1]               // affinity head
        counts += [128 * 256, 128, 128, 1]               // confidence head
        return counts
    }()

    /// Per-tensor float counts read by DruseAFv4Compute.metal (DRAF v2, 56 tensors;
    /// order = EXPORT_WEIGHT_ORDER in trainDruseAFv4.py).
    static let v4ExpectedCounts: [Int] = {
        let encoder: [Int] = [128 * 20, 128, 128 * 128, 128]
        let msgLayer: [Int] = [128 * 128, 128, 32 * 16, 32, 32, 1, 128, 128]
        let head: [Int] = [64 * 128, 64, 64, 1]
        var counts: [Int] = []
        counts += encoder                                // protein encoder
        for _ in 0..<3 { counts += msgLayer }
        counts += encoder                                // ligand encoder
        counts += [64 * 128, 64, 64 * 128, 64]           // pair projections
        counts += [64 * 24, 64]                          // RBF projection
        counts += [64, 1]                                // pair energy
        counts += [64, 1]                                // context gate
        counts += [128 * 64, 128]                        // context projection
        counts += [128, 128]                             // context LayerNorm
        counts += head                                   // affinity head
        counts += head                                   // confidence head
        counts += [1, 1]                                 // pair_scale, pair_bias
        return counts
    }()

    /// Validate an offset table against the data section and the tensor sizes a
    /// Metal kernel reads. Kernels index tensors by fixed position with no bounds
    /// checks, so a short table, a tensor overrunning the data, or an
    /// architecture mismatch must be rejected here rather than read out of
    /// bounds (or silently mis-scored) on the GPU.
    static func validateEntries(
        _ entries: [DruseAFWeightEntry],
        totalFloats: Int,
        expectedCounts: [Int],
        tag: String
    ) -> Bool {
        guard entries.count >= expectedCounts.count else {
            print("[\(tag)] Weight file has \(entries.count) tensors, expected at least \(expectedCounts.count)")
            return false
        }
        for (i, entry) in entries.enumerated() {
            if Int(entry.offset) + Int(entry.count) > totalFloats {
                print("[\(tag)] Tensor \(i) (offset \(entry.offset), count \(entry.count)) overruns \(totalFloats) floats")
                return false
            }
            if i < expectedCounts.count && Int(entry.count) != expectedCounts[i] {
                print("[\(tag)] Tensor \(i) has \(entry.count) floats, expected \(expectedCounts[i]) — architecture mismatch")
                return false
            }
        }
        return true
    }

    /// Load DruseAF weights from a file URL.
    static func load(from url: URL, device: MTLDevice) -> LoadedWeights? {
        guard let data = try? Data(contentsOf: url) else {
            print("[DruseAF] Failed to read weight file: \(url.path)")
            return nil
        }

        // Parse header
        guard data.count >= 16 else {
            print("[DruseAF] Weight file too small: \(data.count) bytes")
            return nil
        }

        let magic = String(data: data[0..<4], encoding: .ascii) ?? ""
        guard magic == "DRAF" else {
            print("[DruseAF] Invalid magic: '\(magic)' (expected 'DRAF')")
            return nil
        }

        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        let numTensors = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) })
        let totalFloats = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt32.self) })

        guard version == 1 || version == 2 else {
            print("[DruseAF] Unsupported weight file version: \(version)")
            return nil
        }
        let isV4 = (version == 2)

        // Parse offset table: numTensors × (offset: UInt32, count: UInt32)
        let headerSize = 16
        let tableSize = numTensors * 8 // 2 × UInt32 per tensor
        let dataStart = headerSize + tableSize

        guard data.count >= dataStart + totalFloats * 4 else {
            print("[DruseAF] Weight file truncated: \(data.count) bytes, expected \(dataStart + totalFloats * 4)")
            return nil
        }

        // Build DruseAFWeightEntry array for Metal
        var entries = [DruseAFWeightEntry]()
        entries.reserveCapacity(numTensors)
        for i in 0..<numTensors {
            let entryOffset = headerSize + i * 8
            let offset = data.withUnsafeBytes { $0.load(fromByteOffset: entryOffset, as: UInt32.self) }
            let count = data.withUnsafeBytes { $0.load(fromByteOffset: entryOffset + 4, as: UInt32.self) }
            entries.append(DruseAFWeightEntry(offset: offset, count: count))
        }

        guard validateEntries(
            entries, totalFloats: totalFloats,
            expectedCounts: isV4 ? v4ExpectedCounts : v3ExpectedCounts,
            tag: "DruseAF"
        ) else { return nil }

        // Create Metal buffers
        let weightBytes = totalFloats * MemoryLayout<Float>.stride
        guard let weightBuffer = data.withUnsafeBytes({ rawBuf -> MTLBuffer? in
            let floatPtr = rawBuf.baseAddress!.advanced(by: dataStart)
            return device.makeBuffer(bytes: floatPtr, length: weightBytes, options: .storageModeShared)
        }) else {
            print("[DruseAF] Failed to create weight MTLBuffer (\(weightBytes) bytes)")
            return nil
        }

        let entryBytes = entries.count * MemoryLayout<DruseAFWeightEntry>.stride
        guard let entryBuffer = device.makeBuffer(bytes: &entries, length: entryBytes, options: .storageModeShared) else {
            print("[DruseAF] Failed to create entry MTLBuffer")
            return nil
        }

        let versionLabel = isV4 ? "v4 PGN" : "v3 cross-attention"
        print("[DruseAF] Weights loaded (\(versionLabel)): \(numTensors) tensors, \(totalFloats) params (\(weightBytes / 1024) KB)")

        return LoadedWeights(
            weightBuffer: weightBuffer,
            entryBuffer: entryBuffer,
            numTensors: numTensors,
            totalFloats: totalFloats,
            version: Int(version)
        )
    }
}

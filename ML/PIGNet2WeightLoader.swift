import Foundation
@preconcurrency import Metal

/// Loads PIGNet2 physics-informed GNN weights from a binary file into Metal buffers.
///
/// Binary format (from export_pignet2_weights.py):
///   Header: "PIG2" magic, version, numTensors, totalFloats, offset table
///   Data:   concatenated float32 weights
///
/// Produces a single weight `MTLBuffer` + a `DruseAFWeightEntry` offset buffer
/// (reuses the same entry struct as DruseAF).
struct PIGNet2WeightLoader {

    struct LoadedWeights: Sendable {
        let weightBuffer: MTLBuffer
        let entryBuffer: MTLBuffer
        let numTensors: Int
        let totalFloats: Int
    }

    /// Load PIGNet2 weights from the app bundle.
    static func loadFromBundle(device: MTLDevice, filename: String = "PIGNet2") -> LoadedWeights? {
        guard let url = Bundle.main.url(forResource: filename, withExtension: "weights") else {
            print("[PIGNet2] Weight file '\(filename).weights' not found in bundle")
            return nil
        }
        return load(from: url, device: device)
    }

    /// Load PIGNet2 weights from a file URL.
    static func load(from url: URL, device: MTLDevice) -> LoadedWeights? {
        guard let data = try? Data(contentsOf: url) else {
            print("[PIGNet2] Failed to read weight file: \(url.path)")
            return nil
        }

        guard data.count >= 16 else {
            print("[PIGNet2] Weight file too small: \(data.count) bytes")
            return nil
        }

        let magic = String(data: data[0..<4], encoding: .ascii) ?? ""
        guard magic == "PIG2" else {
            print("[PIGNet2] Invalid magic: '\(magic)' (expected 'PIG2')")
            return nil
        }

        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        let numTensors = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) })
        let totalFloats = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt32.self) })

        guard version == 1 else {
            print("[PIGNet2] Unsupported weight file version: \(version)")
            return nil
        }

        let headerSize = 16
        let tableSize = numTensors * 8
        let dataStart = headerSize + tableSize

        guard data.count >= dataStart + totalFloats * 4 else {
            print("[PIGNet2] Weight file truncated: \(data.count) bytes, expected \(dataStart + totalFloats * 4)")
            return nil
        }

        // Build weight entry array (reuses DruseAFWeightEntry struct)
        var entries = [DruseAFWeightEntry]()
        entries.reserveCapacity(numTensors)
        for i in 0..<numTensors {
            let entryOffset = headerSize + i * 8
            let offset = data.withUnsafeBytes { $0.load(fromByteOffset: entryOffset, as: UInt32.self) }
            let count = data.withUnsafeBytes { $0.load(fromByteOffset: entryOffset + 4, as: UInt32.self) }
            entries.append(DruseAFWeightEntry(offset: offset, count: count))
        }

        let weightBytes = totalFloats * MemoryLayout<Float>.stride
        guard let weightBuffer = data.withUnsafeBytes({ rawBuf -> MTLBuffer? in
            let floatPtr = rawBuf.baseAddress!.advanced(by: dataStart)
            return device.makeBuffer(bytes: floatPtr, length: weightBytes, options: .storageModeShared)
        }) else {
            print("[PIGNet2] Failed to create weight MTLBuffer (\(weightBytes) bytes)")
            return nil
        }

        let entryBytes = entries.count * MemoryLayout<DruseAFWeightEntry>.stride
        guard let entryBuffer = device.makeBuffer(bytes: &entries, length: entryBytes, options: .storageModeShared) else {
            print("[PIGNet2] Failed to create entry MTLBuffer")
            return nil
        }

        print("[PIGNet2] Weights loaded: \(numTensors) tensors, \(totalFloats) params (\(weightBytes / 1024) KB)")

        return LoadedWeights(
            weightBuffer: weightBuffer,
            entryBuffer: entryBuffer,
            numTensors: numTensors,
            totalFloats: totalFloats
        )
    }
}

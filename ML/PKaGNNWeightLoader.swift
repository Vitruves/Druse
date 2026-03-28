import Foundation
import Metal

/// Loads pKa GNN weights from a binary file into Metal buffers.
///
/// Binary format (from train_pka.py export):
///   Header: "PKA1" magic, version, numTensors, totalFloats, offset table
///   Data:   concatenated float32 weights
///
/// Reuses `DruseAFWeightEntry` for the offset table (same layout).
struct PKaGNNWeightLoader {

    struct LoadedWeights {
        let weightBuffer: MTLBuffer
        let entryBuffer: MTLBuffer
        let numTensors: Int
        let totalFloats: Int
    }

    static func loadFromBundle(device: MTLDevice) -> LoadedWeights? {
        guard let url = Bundle.main.url(forResource: "PkaGNN", withExtension: "weights") else {
            print("[PKaGNN] PkaGNN.weights not found in bundle")
            return nil
        }
        return load(from: url, device: device)
    }

    static func load(from url: URL, device: MTLDevice) -> LoadedWeights? {
        guard let data = try? Data(contentsOf: url) else {
            print("[PKaGNN] Failed to read: \(url.path)")
            return nil
        }

        guard data.count >= 16 else {
            print("[PKaGNN] File too small: \(data.count) bytes")
            return nil
        }

        let magic = String(data: data[0..<4], encoding: .ascii) ?? ""
        guard magic == "PKA1" else {
            print("[PKaGNN] Invalid magic: '\(magic)' (expected 'PKA1')")
            return nil
        }

        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        guard version == 1 else {
            print("[PKaGNN] Unsupported version: \(version)")
            return nil
        }

        let numTensors = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) })
        let totalFloats = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt32.self) })

        let headerSize = 16
        let tableSize = numTensors * 8
        let dataStart = headerSize + tableSize

        guard data.count >= dataStart + totalFloats * 4 else {
            print("[PKaGNN] File truncated: \(data.count) bytes, need \(dataStart + totalFloats * 4)")
            return nil
        }

        var entries = [DruseAFWeightEntry]()
        entries.reserveCapacity(numTensors)
        for i in 0..<numTensors {
            let off = headerSize + i * 8
            let offset = data.withUnsafeBytes { $0.load(fromByteOffset: off, as: UInt32.self) }
            let count = data.withUnsafeBytes { $0.load(fromByteOffset: off + 4, as: UInt32.self) }
            entries.append(DruseAFWeightEntry(offset: offset, count: count))
        }

        let weightBytes = totalFloats * MemoryLayout<Float>.stride
        guard let weightBuffer = data.withUnsafeBytes({ rawBuf -> MTLBuffer? in
            let ptr = rawBuf.baseAddress!.advanced(by: dataStart)
            return device.makeBuffer(bytes: ptr, length: weightBytes, options: .storageModeShared)
        }) else {
            print("[PKaGNN] Failed to create weight buffer (\(weightBytes) bytes)")
            return nil
        }

        let entryBytes = entries.count * MemoryLayout<DruseAFWeightEntry>.stride
        guard let entryBuffer = device.makeBuffer(bytes: &entries, length: entryBytes, options: .storageModeShared) else {
            print("[PKaGNN] Failed to create entry buffer")
            return nil
        }

        print("[PKaGNN] Weights loaded: \(numTensors) tensors, \(totalFloats) params (\(weightBytes / 1024) KB)")
        return LoadedWeights(
            weightBuffer: weightBuffer,
            entryBuffer: entryBuffer,
            numTensors: numTensors,
            totalFloats: totalFloats
        )
    }
}

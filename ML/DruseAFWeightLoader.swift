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
    }

    /// Load DruseAF weights from the app bundle.
    static func loadFromBundle(device: MTLDevice, filename: String = "DruseAF") -> LoadedWeights? {
        guard let url = Bundle.main.url(forResource: filename, withExtension: "weights") else {
            print("[DruseAF] Weight file '\(filename).weights' not found in bundle")
            return nil
        }
        return load(from: url, device: device)
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

        guard version == 1 else {
            print("[DruseAF] Unsupported weight file version: \(version)")
            return nil
        }

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

        print("[DruseAF] Weights loaded: \(numTensors) tensors, \(totalFloats) params (\(weightBytes / 1024) KB)")

        return LoadedWeights(
            weightBuffer: weightBuffer,
            entryBuffer: entryBuffer,
            numTensors: numTensors,
            totalFloats: totalFloats
        )
    }
}

import MetalKit
import simd

// MARK: - GPU Surface Atom (must match SurfaceAtom in SurfaceCompute.metal)

private struct GPUSurfaceAtom {
    var position: (Float, Float, Float) = (0, 0, 0) // packed_float3
    var vdwRadius: Float = 0
    var charge: Float = 0
    var atomicNum: UInt32 = 0
    var isAromatic: UInt32 = 0
    var _pad0: Float = 0
}

// MARK: - GPU Surface Grid Params (must match SurfaceGridParams in SurfaceCompute.metal)

private struct GPUSurfaceGridParams {
    var origin: (Float, Float, Float) = (0, 0, 0) // packed_float3
    var spacing: Float = 0
    var dims: SIMD3<UInt32> = .zero
    var totalPoints: UInt32 = 0
    var isovalue: Float = 0
    var numAtoms: UInt32 = 0
    var maxVertices: UInt32 = 0
    var maxIndices: UInt32 = 0
    var probeRadius: Float = 1.4
    var _pad0: Float = 0
    var _pad1: Float = 0
    var _pad2: Float = 0
}

// MARK: - GPU Surface Vertex (must match SurfaceVertex in SurfaceCompute.metal)

private struct GPUSurfaceVertex {
    var position: (Float, Float, Float) = (0, 0, 0) // packed_float3
    var normal: (Float, Float, Float) = (0, 0, 0)   // packed_float3
    var color: SIMD4<Float> = .zero
}

// MARK: - Surface Type

enum SurfaceFieldType {
    case connolly   // Signed distance field (SES/Connolly)
    case gaussian   // Gaussian blob surface
}

enum SurfaceColorMode: String, CaseIterable {
    case uniform       = "Uniform"
    case esp           = "Electrostatic"
    case hydrophobicity = "Hydrophobicity"
    case pharmacophore = "Pharmacophore"
}

// MARK: - Surface Generation Result

struct SurfaceResult {
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let vertexCount: Int
    let indexCount: Int
}

// MARK: - Surface Generator

/// GPU-accelerated molecular surface generator using marching cubes.
/// Computes Connolly (SES) or Gaussian surfaces on the GPU, optionally
/// colored by electrostatic potential.
@MainActor
final class SurfaceGenerator {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // Compute pipeline states
    private let distanceFieldPipeline: MTLComputePipelineState
    private let gaussianFieldPipeline: MTLComputePipelineState
    private let marchingCubesSinglePassPipeline: MTLComputePipelineState
    private let marchingCubesCountPipeline: MTLComputePipelineState
    private let marchingCubesEmitPipeline: MTLComputePipelineState
    private let espColoringPipeline: MTLComputePipelineState
    private let hydrophobicityPipeline: MTLComputePipelineState
    private let pharmacophorePipeline: MTLComputePipelineState

    // Configuration
    var gridSpacing: Float = 0.5       // Angstroms per grid cell
    var padding: Float = 6.0           // Angstroms of padding around bounding box
    var probeRadius: Float = 1.4       // Water probe radius in Angstroms
    var isovalue: Float = 0.5          // Gaussian field isovalue (only for gaussian mode)
    var fieldType: SurfaceFieldType = .connolly
    var colorMode: SurfaceColorMode = .uniform

    // Buffer limits
    private let maxVertices: UInt32 = 4_000_000
    private let maxIndices: UInt32 = 12_000_000

    init(device: MTLDevice, commandQueue: MTLCommandQueue) throws {
        self.device = device
        self.commandQueue = commandQueue

        guard let lib = device.makeDefaultLibrary() else {
            throw SurfaceGeneratorError.libraryCreationFailed
        }
        self.library = lib

        // Build compute pipelines
        guard let distFieldFn = lib.makeFunction(name: "computeDistanceField") else {
            throw SurfaceGeneratorError.kernelNotFound("computeDistanceField")
        }
        self.distanceFieldPipeline = try device.makeComputePipelineState(function: distFieldFn)

        guard let gaussFn = lib.makeFunction(name: "computeGaussianField") else {
            throw SurfaceGeneratorError.kernelNotFound("computeGaussianField")
        }
        self.gaussianFieldPipeline = try device.makeComputePipelineState(function: gaussFn)

        guard let mcSinglePassFn = lib.makeFunction(name: "marchingCubesSinglePass") else {
            throw SurfaceGeneratorError.kernelNotFound("marchingCubesSinglePass")
        }
        self.marchingCubesSinglePassPipeline = try device.makeComputePipelineState(function: mcSinglePassFn)

        guard let mcCountFn = lib.makeFunction(name: "marchingCubesCount") else {
            throw SurfaceGeneratorError.kernelNotFound("marchingCubesCount")
        }
        self.marchingCubesCountPipeline = try device.makeComputePipelineState(function: mcCountFn)

        guard let mcEmitFn = lib.makeFunction(name: "marchingCubesEmit") else {
            throw SurfaceGeneratorError.kernelNotFound("marchingCubesEmit")
        }
        self.marchingCubesEmitPipeline = try device.makeComputePipelineState(function: mcEmitFn)

        guard let espFn = lib.makeFunction(name: "computeSurfaceESP") else {
            throw SurfaceGeneratorError.kernelNotFound("computeSurfaceESP")
        }
        self.espColoringPipeline = try device.makeComputePipelineState(function: espFn)

        guard let hydroFn = lib.makeFunction(name: "computeSurfaceHydrophobicity") else {
            throw SurfaceGeneratorError.kernelNotFound("computeSurfaceHydrophobicity")
        }
        self.hydrophobicityPipeline = try device.makeComputePipelineState(function: hydroFn)

        guard let pharmacoFn = lib.makeFunction(name: "computeSurfacePharmacophore") else {
            throw SurfaceGeneratorError.kernelNotFound("computeSurfacePharmacophore")
        }
        self.pharmacophorePipeline = try device.makeComputePipelineState(function: pharmacoFn)
    }

    // MARK: - Public API

    /// Generate a molecular surface from atom positions, VdW radii, and partial charges.
    /// - Parameters:
    ///   - atoms: Array of atoms with positions and element types
    ///   - Returns: SurfaceResult with vertex and index buffers ready for rendering
    func generateSurface(atoms: [Atom]) -> SurfaceResult? {
        guard !atoms.isEmpty else {
            ActivityLog.shared.debug("[Surface] generateSurface skipped: empty atom array", category: .render)
            return nil
        }
        ActivityLog.shared.info("[Surface] Generating \(fieldType == .connolly ? "Connolly" : "Gaussian") surface: \(atoms.count) atoms, spacing=\(gridSpacing), probe=\(probeRadius), color=\(colorMode.rawValue)", category: .render)

        // Prepare GPU atom data
        let gpuAtoms = atoms.map { atom -> GPUSurfaceAtom in
            GPUSurfaceAtom(
                position: (atom.position.x, atom.position.y, atom.position.z),
                vdwRadius: atom.element.vdwRadius,
                charge: abs(atom.charge) > 0.0001 ? atom.charge : Float(atom.formalCharge),
                atomicNum: UInt32(atom.element.rawValue),
                isAromatic: 0,
                _pad0: 0
            )
        }

        return generateSurface(
            gpuAtoms: gpuAtoms,
            positions: atoms.map(\.position)
        )
    }

    /// Generate a molecular surface from raw arrays.
    /// - Parameters:
    ///   - positions: Atom center positions
    ///   - radii: VdW radii per atom
    ///   - charges: Partial charges per atom
    /// - Returns: SurfaceResult with vertex and index buffers
    func generateSurface(
        positions: [SIMD3<Float>],
        radii: [Float],
        charges: [Float]
    ) -> SurfaceResult? {
        guard positions.count == radii.count,
              positions.count == charges.count,
              !positions.isEmpty else {
            ActivityLog.shared.debug("[Surface] generateSurface(raw) skipped: mismatched or empty arrays (pos=\(positions.count), radii=\(radii.count), charges=\(charges.count))", category: .render)
            return nil
        }

        let gpuAtoms = (0..<positions.count).map { i -> GPUSurfaceAtom in
            GPUSurfaceAtom(
                position: (positions[i].x, positions[i].y, positions[i].z),
                vdwRadius: radii[i],
                charge: charges[i],
                atomicNum: 6, // default to carbon for raw arrays
                isAromatic: 0,
                _pad0: 0
            )
        }

        return generateSurface(gpuAtoms: gpuAtoms, positions: positions)
    }

    // MARK: - Private Implementation

    private func generateSurface(
        gpuAtoms: [GPUSurfaceAtom],
        positions: [SIMD3<Float>]
    ) -> SurfaceResult? {
        // Compute bounding box with padding
        let (bbMin, bbMax) = boundingBox(positions)
        let paddedMin = bbMin - SIMD3<Float>(repeating: padding)
        let paddedMax = bbMax + SIMD3<Float>(repeating: padding)

        let gridSize = paddedMax - paddedMin
        let nx = UInt32(ceil(gridSize.x / gridSpacing)) + 1
        let ny = UInt32(ceil(gridSize.y / gridSpacing)) + 1
        let nz = UInt32(ceil(gridSize.z / gridSpacing)) + 1
        let totalPoints = nx * ny * nz

        // Safety check: grid must be reasonable
        guard totalPoints > 0, totalPoints < 256_000_000 else {
            ActivityLog.shared.warn("[Surface] Grid too large: \(nx)x\(ny)x\(nz) = \(totalPoints) points (max 256M)", category: .render)
            return nil
        }

        let totalVoxels = Int((nx - 1) * (ny - 1) * (nz - 1))
        guard totalVoxels > 0 else {
            ActivityLog.shared.warn("[Surface] Zero voxels after grid setup", category: .render)
            return nil
        }

        // Build grid params
        let effectiveIso: Float = (fieldType == .connolly) ? 0.0 : isovalue
        var gridParams = GPUSurfaceGridParams(
            origin: (paddedMin.x, paddedMin.y, paddedMin.z),
            spacing: gridSpacing,
            dims: SIMD3<UInt32>(nx, ny, nz),
            totalPoints: totalPoints,
            isovalue: effectiveIso,
            numAtoms: UInt32(gpuAtoms.count),
            maxVertices: maxVertices,
            maxIndices: maxIndices,
            probeRadius: probeRadius,
            _pad0: 0, _pad1: 0, _pad2: 0
        )

        // Create GPU buffers
        guard let atomBuffer = device.makeBuffer(
            bytes: gpuAtoms,
            length: gpuAtoms.count * MemoryLayout<GPUSurfaceAtom>.stride,
            options: .storageModeShared
        ) else {
            ActivityLog.shared.error("[Surface] Failed to create atom buffer (\(gpuAtoms.count) atoms)", category: .render)
            return nil
        }
        atomBuffer.label = "SurfaceAtoms"

        guard let scalarFieldBuffer = device.makeBuffer(
            length: Int(totalPoints) * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return nil }
        scalarFieldBuffer.label = "ScalarField"

        guard let paramsBuffer = device.makeBuffer(
            bytes: &gridParams,
            length: MemoryLayout<GPUSurfaceGridParams>.stride,
            options: .storageModeShared
        ) else { return nil }
        paramsBuffer.label = "SurfaceGridParams"

        // Per-voxel triangle count buffer (Pass 1 output)
        guard let triCountBuffer = device.makeBuffer(
            length: totalVoxels * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else { return nil }
        triCountBuffer.label = "TriCountPerVoxel"

        // Global triangle count (single atomic uint, Pass 1 output)
        guard let globalTriCountBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else { return nil }
        globalTriCountBuffer.label = "GlobalTriCount"
        memset(globalTriCountBuffer.contents(), 0, MemoryLayout<UInt32>.stride)

        // ====================================================================
        // Command Buffer 1: Scalar field + Marching cubes count pass
        // ====================================================================
        guard let cmdBuf1 = commandQueue.makeCommandBuffer() else { return nil }
        cmdBuf1.label = "SurfaceGeneration_ScalarField+Count"

        // Compute scalar field (distance field or Gaussian)
        if let encoder = cmdBuf1.makeComputeCommandEncoder() {
            encoder.label = "ScalarField"

            let pipeline = (fieldType == .connolly)
                ? distanceFieldPipeline
                : gaussianFieldPipeline

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scalarFieldBuffer, offset: 0, index: 0)
            encoder.setBuffer(atomBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

            let threadGroupSize = min(
                pipeline.maxTotalThreadsPerThreadgroup,
                Int(totalPoints)
            )
            let threadGroups = MTLSize(
                width: (Int(totalPoints) + threadGroupSize - 1) / threadGroupSize,
                height: 1,
                depth: 1
            )
            let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        // Marching cubes Pass 1: count triangles per voxel
        if let encoder = cmdBuf1.makeComputeCommandEncoder() {
            encoder.label = "MarchingCubesCount"
            encoder.setComputePipelineState(marchingCubesCountPipeline)
            encoder.setBuffer(scalarFieldBuffer, offset: 0, index: 0)
            encoder.setBuffer(triCountBuffer, offset: 0, index: 1)
            encoder.setBuffer(globalTriCountBuffer, offset: 0, index: 2)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 3)

            let threadGroupSize = min(
                marchingCubesCountPipeline.maxTotalThreadsPerThreadgroup,
                totalVoxels
            )
            let threadGroups = MTLSize(
                width: (totalVoxels + threadGroupSize - 1) / threadGroupSize,
                height: 1,
                depth: 1
            )
            let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        // ====================================================================
        // CPU: Read back total triangle count, compute exclusive prefix sum
        // ====================================================================
        let totalTriangles = Int(globalTriCountBuffer.contents()
            .bindMemory(to: UInt32.self, capacity: 1).pointee)

        guard totalTriangles > 0 else {
            ActivityLog.shared.info("[Surface] Marching cubes produced 0 triangles", category: .render)
            return nil
        }

        let totalVertices = totalTriangles * 3
        let totalIndices = totalTriangles * 3

        // Clamp to buffer limits
        guard totalVertices <= Int(maxVertices), totalIndices <= Int(maxIndices) else {
            // Surface too large for allocated buffers
            ActivityLog.shared.warn("[Surface] Surface too large: \(totalVertices) vertices (max \(maxVertices)), \(totalIndices) indices (max \(maxIndices))", category: .render)
            return nil
        }

        // Compute exclusive prefix sum of per-voxel triangle counts on CPU
        let triCountPtr = triCountBuffer.contents().bindMemory(to: UInt32.self, capacity: totalVoxels)

        guard let offsetBuffer = device.makeBuffer(
            length: totalVoxels * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else { return nil }
        offsetBuffer.label = "TriOffsets"
        let offsetPtr = offsetBuffer.contents().bindMemory(to: UInt32.self, capacity: totalVoxels)

        var runningSum: UInt32 = 0
        for i in 0..<totalVoxels {
            offsetPtr[i] = runningSum
            runningSum += triCountPtr[i]
        }

        // ====================================================================
        // Command Buffer 2: Marching cubes emit pass + surface coloring
        // ====================================================================

        // Allocate vertex/index buffers to exact size
        let vertexBufferSize = totalVertices * MemoryLayout<GPUSurfaceVertex>.stride
        guard let vertexBuffer = device.makeBuffer(
            length: vertexBufferSize,
            options: .storageModeShared
        ) else {
            ActivityLog.shared.error("[Surface] Failed to create vertex buffer (\(vertexBufferSize / 1024) KB)", category: .render)
            return nil
        }
        vertexBuffer.label = "SurfaceVertices"

        let indexBufferSize = totalIndices * MemoryLayout<UInt32>.stride
        guard let indexBuffer = device.makeBuffer(
            length: indexBufferSize,
            options: .storageModeShared
        ) else {
            ActivityLog.shared.error("[Surface] Failed to create index buffer (\(indexBufferSize / 1024) KB)", category: .render)
            return nil
        }
        indexBuffer.label = "SurfaceIndices"

        guard let cmdBuf2 = commandQueue.makeCommandBuffer() else { return nil }
        cmdBuf2.label = "SurfaceGeneration_Emit+Color"

        // Marching cubes Pass 2: emit triangles at prefix-sum offsets (no atomics)
        if let encoder = cmdBuf2.makeComputeCommandEncoder() {
            encoder.label = "MarchingCubesEmit"
            encoder.setComputePipelineState(marchingCubesEmitPipeline)
            encoder.setBuffer(scalarFieldBuffer, offset: 0, index: 0)
            encoder.setBuffer(vertexBuffer, offset: 0, index: 1)
            encoder.setBuffer(indexBuffer, offset: 0, index: 2)
            encoder.setBuffer(offsetBuffer, offset: 0, index: 3)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 4)

            let threadGroupSize = min(
                marchingCubesEmitPipeline.maxTotalThreadsPerThreadgroup,
                totalVoxels
            )
            let threadGroups = MTLSize(
                width: (totalVoxels + threadGroupSize - 1) / threadGroupSize,
                height: 1,
                depth: 1
            )
            let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        // Surface coloring (optional, based on colorMode)
        // For coloring kernels, we store the vertex count in a small buffer
        if colorMode != .uniform {
            let colorPipeline: MTLComputePipelineState
            let label: String
            switch colorMode {
            case .esp:
                colorPipeline = espColoringPipeline
                label = "ESPColoring"
            case .hydrophobicity:
                colorPipeline = hydrophobicityPipeline
                label = "HydrophobicityColoring"
            case .pharmacophore:
                colorPipeline = pharmacophorePipeline
                label = "PharmacophoreColoring"
            case .uniform:
                fatalError("unreachable")
            }

            // The coloring kernels read vertexCount[0] to know how many vertices to process
            guard let vertexCounterBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ) else { return nil }
            vertexCounterBuffer.label = "VertexCounter"
            vertexCounterBuffer.contents()
                .bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(totalVertices)

            if let encoder = cmdBuf2.makeComputeCommandEncoder() {
                encoder.label = label
                encoder.setComputePipelineState(colorPipeline)
                encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setBuffer(atomBuffer, offset: 0, index: 1)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
                encoder.setBuffer(vertexCounterBuffer, offset: 0, index: 3)

                let dispatchCount = totalVertices
                let threadGroupSize = min(
                    colorPipeline.maxTotalThreadsPerThreadgroup,
                    dispatchCount
                )
                let threadGroups = MTLSize(
                    width: (dispatchCount + threadGroupSize - 1) / threadGroupSize,
                    height: 1,
                    depth: 1
                )
                let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
                encoder.endEncoding()
            }
        }

        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        ActivityLog.shared.debug("[Surface] Complete: \(totalVertices) vertices, \(totalTriangles) triangles, grid \(nx)x\(ny)x\(nz)", category: .render)
        return SurfaceResult(
            vertexBuffer: vertexBuffer,
            indexBuffer: indexBuffer,
            vertexCount: totalVertices,
            indexCount: totalIndices
        )
    }
}

// MARK: - Errors

enum SurfaceGeneratorError: Error, CustomStringConvertible {
    case libraryCreationFailed
    case kernelNotFound(String)
    case bufferCreationFailed
    case computeFailed

    var description: String {
        switch self {
        case .libraryCreationFailed:
            return "Failed to create Metal library for surface compute"
        case .kernelNotFound(let name):
            return "Metal kernel function '\(name)' not found"
        case .bufferCreationFailed:
            return "Failed to create GPU buffer for surface generation"
        case .computeFailed:
            return "Surface compute pass failed"
        }
    }
}

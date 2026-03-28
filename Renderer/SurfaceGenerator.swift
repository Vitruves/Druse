import MetalKit
import simd

// MARK: - GPU Surface Atom (must match SurfaceAtom in SurfaceCompute.metal)

private struct GPUSurfaceAtom {
    var position: (Float, Float, Float) = (0, 0, 0) // packed_float3
    var vdwRadius: Float = 0
    var charge: Float = 0
    var atomicNum: UInt32 = 0
    var flags: UInt32 = 0
    var hydrophobicity: Float = 0
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

struct SurfaceLegend: Sendable {
    struct Entry: Sendable {
        let label: String
        let color: SIMD4<Float>
    }

    enum Kind: Sendable {
        case gradient(minLabel: String, midLabel: String?, maxLabel: String, colors: [SIMD4<Float>])
        case categorical([Entry])
    }

    let title: String
    let kind: Kind
}

// MARK: - Surface Generation Result

struct SurfaceResult {
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let vertexCount: Int
    let indexCount: Int
    let legend: SurfaceLegend?
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
    private let espValuePipeline: MTLComputePipelineState
    private let hydrophobicityValuePipeline: MTLComputePipelineState
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

        guard let espFn = lib.makeFunction(name: "computeSurfaceESPValues") else {
            throw SurfaceGeneratorError.kernelNotFound("computeSurfaceESPValues")
        }
        self.espValuePipeline = try device.makeComputePipelineState(function: espFn)

        guard let hydroFn = lib.makeFunction(name: "computeSurfaceHydrophobicityValues") else {
            throw SurfaceGeneratorError.kernelNotFound("computeSurfaceHydrophobicityValues")
        }
        self.hydrophobicityValuePipeline = try device.makeComputePipelineState(function: hydroFn)

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
            makeSurfaceAtom(from: atom)
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
                flags: SurfaceAtomFlags.hydrophobic,
                hydrophobicity: 0.35
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
        let legend: SurfaceLegend?
        if colorMode != .uniform {
            guard let vertexCounterBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ) else { return nil }
            vertexCounterBuffer.label = "VertexCounter"
            vertexCounterBuffer.contents()
                .bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(totalVertices)

            switch colorMode {
            case .esp, .hydrophobicity:
                guard let metricBuffer = device.makeBuffer(
                    length: totalVertices * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                ) else { return nil }
                metricBuffer.label = colorMode == .esp ? "SurfaceESPValues" : "SurfaceHydrophobicityValues"

                if let encoder = cmdBuf2.makeComputeCommandEncoder() {
                    encoder.label = colorMode == .esp ? "ESPValues" : "HydrophobicityValues"
                    encoder.setComputePipelineState(colorMode == .esp ? espValuePipeline : hydrophobicityValuePipeline)
                    encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
                    encoder.setBuffer(atomBuffer, offset: 0, index: 1)
                    encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
                    encoder.setBuffer(vertexCounterBuffer, offset: 0, index: 3)
                    encoder.setBuffer(metricBuffer, offset: 0, index: 4)

                    let dispatchCount = totalVertices
                    let pipeline = colorMode == .esp ? espValuePipeline : hydrophobicityValuePipeline
                    let threadGroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, dispatchCount)
                    let threadGroups = MTLSize(
                        width: (dispatchCount + threadGroupSize - 1) / threadGroupSize,
                        height: 1,
                        depth: 1
                    )
                    let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
                    encoder.endEncoding()
                }
                legend = nil
                cmdBuf2.commit()
                cmdBuf2.waitUntilCompleted()
                let computedLegend = applyMetricColors(
                    to: vertexBuffer,
                    vertexCount: totalVertices,
                    values: metricBuffer,
                    mode: colorMode
                )
                ActivityLog.shared.debug("[Surface] Complete: \(totalVertices) vertices, \(totalTriangles) triangles, grid \(nx)x\(ny)x\(nz)", category: .render)
                return SurfaceResult(
                    vertexBuffer: vertexBuffer,
                    indexBuffer: indexBuffer,
                    vertexCount: totalVertices,
                    indexCount: totalIndices,
                    legend: computedLegend
                )
            case .pharmacophore:
                if let encoder = cmdBuf2.makeComputeCommandEncoder() {
                    encoder.label = "PharmacophoreColoring"
                    encoder.setComputePipelineState(pharmacophorePipeline)
                    encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
                    encoder.setBuffer(atomBuffer, offset: 0, index: 1)
                    encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
                    encoder.setBuffer(vertexCounterBuffer, offset: 0, index: 3)

                    let dispatchCount = totalVertices
                    let threadGroupSize = min(pharmacophorePipeline.maxTotalThreadsPerThreadgroup, dispatchCount)
                    let threadGroups = MTLSize(
                        width: (dispatchCount + threadGroupSize - 1) / threadGroupSize,
                        height: 1,
                        depth: 1
                    )
                    let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
                    encoder.endEncoding()
                }
                legend = SurfaceLegend(
                    title: "Pharmacophore",
                    kind: .categorical([
                        .init(label: "Donor", color: SIMD4(0.30, 0.60, 0.95, 1.0)),
                        .init(label: "Acceptor", color: SIMD4(0.92, 0.35, 0.35, 1.0)),
                        .init(label: "Positive", color: SIMD4(0.10, 0.25, 0.90, 1.0)),
                        .init(label: "Negative", color: SIMD4(0.90, 0.10, 0.10, 1.0)),
                        .init(label: "Aromatic", color: SIMD4(0.62, 0.28, 0.82, 1.0)),
                        .init(label: "Hydrophobe", color: SIMD4(0.88, 0.72, 0.18, 1.0))
                    ])
                )
            case .uniform:
                fatalError("unreachable")
            }
        } else {
            legend = nil
        }

        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        ActivityLog.shared.debug("[Surface] Complete: \(totalVertices) vertices, \(totalTriangles) triangles, grid \(nx)x\(ny)x\(nz)", category: .render)
        return SurfaceResult(
            vertexBuffer: vertexBuffer,
            indexBuffer: indexBuffer,
            vertexCount: totalVertices,
            indexCount: totalIndices,
            legend: legend
        )
    }

    // MARK: - Surface Chemistry

    private func makeSurfaceAtom(from atom: Atom) -> GPUSurfaceAtom {
        let atomName = atom.name.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let residue = atom.residueName.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let charge = abs(atom.charge) > 0.0001 ? atom.charge : Float(atom.formalCharge)
        var flags: UInt32 = 0

        if Self.isAromaticAtom(atomName, residue: residue) {
            flags |= SurfaceAtomFlags.aromatic
        }
        if Self.isDonor(atom: atom, atomName: atomName, residue: residue) {
            flags |= SurfaceAtomFlags.donor
        }
        if Self.isAcceptor(atom: atom, atomName: atomName, residue: residue) {
            flags |= SurfaceAtomFlags.acceptor
        }
        if Self.isPositivelyCharged(atom: atom, atomName: atomName, residue: residue, charge: charge) {
            flags |= SurfaceAtomFlags.positive
        }
        if Self.isNegativelyCharged(atom: atom, atomName: atomName, residue: residue, charge: charge) {
            flags |= SurfaceAtomFlags.negative
        }
        if Self.isHydrophobic(atom: atom, atomName: atomName, residue: residue) {
            flags |= SurfaceAtomFlags.hydrophobic
        }

        return GPUSurfaceAtom(
            position: (atom.position.x, atom.position.y, atom.position.z),
            vdwRadius: atom.element.vdwRadius,
            charge: charge,
            atomicNum: UInt32(atom.element.rawValue),
            flags: flags,
            hydrophobicity: Self.normalizedHydrophobicity(for: residue)
        )
    }

    private func applyMetricColors(
        to vertexBuffer: MTLBuffer,
        vertexCount: Int,
        values metricBuffer: MTLBuffer,
        mode: SurfaceColorMode
    ) -> SurfaceLegend {
        let vertices = vertexBuffer.contents().bindMemory(to: GPUSurfaceVertex.self, capacity: vertexCount)
        let values = metricBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount)

        switch mode {
        case .esp:
            let maxAbs = robustSymmetricScale(values: values, count: vertexCount, fallback: 5.0)
            for i in 0..<vertexCount {
                let value = values[i]
                let normalized = simd_clamp(value / maxAbs, -1.0, 1.0)
                let color: SIMD4<Float>
                if normalized < 0 {
                    let t = -normalized
                    color = SIMD4(Self.mix(SIMD3(0.98, 0.98, 0.98), SIMD3(0.90, 0.16, 0.16), t), 0.85)
                } else {
                    let t = normalized
                    color = SIMD4(Self.mix(SIMD3(0.98, 0.98, 0.98), SIMD3(0.14, 0.26, 0.92), t), 0.85)
                }
                vertices[i].color = color
            }
            return SurfaceLegend(
                title: "Electrostatic",
                kind: .gradient(
                    minLabel: String(format: "-%.1f", maxAbs),
                    midLabel: "0",
                    maxLabel: String(format: "+%.1f", maxAbs),
                    colors: [
                        SIMD4(0.90, 0.16, 0.16, 1.0),
                        SIMD4(0.98, 0.98, 0.98, 1.0),
                        SIMD4(0.14, 0.26, 0.92, 1.0)
                    ]
                )
            )
        case .hydrophobicity:
            let range = robustRange(values: values, count: vertexCount, fallbackMin: -1.0, fallbackMax: 1.0)
            let negativeScale = max(abs(min(range.min, 0)), 0.05)
            let positiveScale = max(range.max, 0.05)
            for i in 0..<vertexCount {
                let value = values[i]
                let color: SIMD4<Float>
                if value < 0 {
                    let t = simd_clamp(-value / negativeScale, 0.0, 1.0)
                    color = SIMD4(Self.mix(SIMD3(0.96, 0.97, 0.98), SIMD3(0.16, 0.45, 0.84), t), 0.85)
                } else {
                    let t = simd_clamp(value / positiveScale, 0.0, 1.0)
                    color = SIMD4(Self.mix(SIMD3(0.96, 0.97, 0.98), SIMD3(0.86, 0.64, 0.18), t), 0.85)
                }
                vertices[i].color = color
            }
            return SurfaceLegend(
                title: "Hydrophobicity",
                kind: .gradient(
                    minLabel: String(format: "%.2f", range.min),
                    midLabel: "0",
                    maxLabel: String(format: "%.2f", range.max),
                    colors: [
                        SIMD4(0.16, 0.45, 0.84, 1.0),
                        SIMD4(0.96, 0.97, 0.98, 1.0),
                        SIMD4(0.86, 0.64, 0.18, 1.0)
                    ]
                )
            )
        case .uniform, .pharmacophore:
            return SurfaceLegend(
                title: "",
                kind: .categorical([])
            )
        }
    }

    private func robustSymmetricScale(values: UnsafePointer<Float>, count: Int, fallback: Float) -> Float {
        let step = max(1, count / 4096)
        var samples: [Float] = []
        samples.reserveCapacity((count + step - 1) / step)
        for i in stride(from: 0, to: count, by: step) {
            samples.append(abs(values[i]))
        }
        let percentile = Self.percentile(samples, q: 0.95)
        return max(percentile, fallback)
    }

    private func robustRange(values: UnsafePointer<Float>, count: Int, fallbackMin: Float, fallbackMax: Float) -> (min: Float, max: Float) {
        let step = max(1, count / 4096)
        var samples: [Float] = []
        samples.reserveCapacity((count + step - 1) / step)
        for i in stride(from: 0, to: count, by: step) {
            samples.append(values[i])
        }
        let lo = Self.percentile(samples, q: 0.05)
        let hi = Self.percentile(samples, q: 0.95)
        if hi - lo < 0.1 {
            return (fallbackMin, fallbackMax)
        }
        return (lo, hi)
    }

    private static func percentile(_ values: [Float], q: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let pos = Int(Float(sorted.count - 1) * simd_clamp(q, 0, 1))
        return sorted[pos]
    }

    private static func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
        a + (b - a) * t
    }

    private enum SurfaceAtomFlags {
        static let donor: UInt32 = 1 << 0
        static let acceptor: UInt32 = 1 << 1
        static let aromatic: UInt32 = 1 << 2
        static let positive: UInt32 = 1 << 3
        static let negative: UInt32 = 1 << 4
        static let hydrophobic: UInt32 = 1 << 5
    }

    private static let residueHydrophobicity: [String: Float] = [
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
        "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
        "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
        "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
    ]

    private static let aromaticAtoms: [String: Set<String>] = [
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]
    ]

    private static let donorAtoms: [String: Set<String>] = [
        "ARG": ["NE", "NH1", "NH2"],
        "ASN": ["ND2"],
        "GLN": ["NE2"],
        "HIS": ["ND1", "NE2"],
        "LYS": ["NZ"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TRP": ["NE1"],
        "TYR": ["OH"],
        "CYS": ["SG"]
    ]

    private static let acceptorAtoms: [String: Set<String>] = [
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
        "ASN": ["OD1"],
        "GLN": ["OE1"],
        "HIS": ["ND1", "NE2"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "CYS": ["SG"],
        "MET": ["SD"]
    ]

    private static let hydrophobicResidues: Set<String> = [
        "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS"
    ]

    private static let positiveAtoms: [String: Set<String>] = [
        "LYS": ["NZ"],
        "ARG": ["NE", "NH1", "NH2"],
        "HIS": ["ND1", "NE2"]
    ]

    private static let negativeAtoms: [String: Set<String>] = [
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"]
    ]

    private static func normalizedHydrophobicity(for residue: String) -> Float {
        let raw = residueHydrophobicity[residue, default: 0]
        return simd_clamp(raw / 4.5, -1.0, 1.0)
    }

    private static func isAromaticAtom(_ atomName: String, residue: String) -> Bool {
        aromaticAtoms[residue]?.contains(atomName) == true
    }

    private static func isDonor(atom: Atom, atomName: String, residue: String) -> Bool {
        if atom.element == .N && atomName == "N" && residue != "PRO" {
            return true
        }
        return donorAtoms[residue]?.contains(atomName) == true
    }

    private static func isAcceptor(atom: Atom, atomName: String, residue: String) -> Bool {
        if atom.element == .O && (atomName == "O" || atomName == "OXT") {
            return true
        }
        return acceptorAtoms[residue]?.contains(atomName) == true
    }

    private static func isPositivelyCharged(atom: Atom, atomName: String, residue: String, charge: Float) -> Bool {
        if charge > 0.25 {
            return true
        }
        return positiveAtoms[residue]?.contains(atomName) == true && atom.formalCharge >= 0
    }

    private static func isNegativelyCharged(atom: Atom, atomName: String, residue: String, charge: Float) -> Bool {
        if charge < -0.25 {
            return true
        }
        return negativeAtoms[residue]?.contains(atomName) == true
    }

    private static func isHydrophobic(atom: Atom, atomName: String, residue: String) -> Bool {
        let backbone = ["N", "CA", "C", "O", "OXT"]
        if backbone.contains(atomName) {
            return false
        }
        if atom.element == .F || atom.element == .Cl || atom.element == .Br {
            return true
        }
        return hydrophobicResidues.contains(residue) && (atom.element == .C || atom.element == .S)
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

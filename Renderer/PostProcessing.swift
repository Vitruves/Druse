// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import MetalKit
import simd

// MARK: - Post-Process Params (must match PostProcessParams in PostProcessing.metal)

private struct GPUPostProcessParams {
    var ssaoRadius: Float = 0.05
    var ssaoBias: Float = 0.025
    var ssaoNumSamples: Int32 = 16
    var ssaoStrength: Float = 0.8
    var edgeStrength: Float = 0.6
    var nearPlane: Float = 0.1
    var farPlane: Float = 500.0
    var width: UInt32 = 0
    var height: UInt32 = 0
    var dofFocalDistance: Float = 15.0
    var dofFocalRange: Float = 10.0
    var dofMaxBlur: Float = 6.0
}

// MARK: - Post-Processing Pipeline

/// GPU post-processing pipeline for molecular visualization.
/// Manages SSAO, edge detection, depth of field, and final compositing.
@MainActor
final class PostProcessingPipeline {

    private let device: MTLDevice
    private let library: MTLLibrary

    // Compute pipeline states
    private let ssaoPipeline: MTLComputePipelineState
    private let ssaoBlurPipeline: MTLComputePipelineState
    private let edgeDetectPipeline: MTLComputePipelineState
    private let dofPipeline: MTLComputePipelineState
    private let compositePipeline: MTLComputePipelineState
    private let fillPipeline: MTLComputePipelineState

    // Intermediate textures (recreated on resize)
    private var ssaoTexture: MTLTexture?
    private var ssaoBlurTexture: MTLTexture?
    private var edgeTexture: MTLTexture?
    private var dofTexture: MTLTexture?

    // Params buffer
    private var paramsBuffer: MTLBuffer?

    // Tracked dimensions for texture recreation
    private var currentWidth: Int = 0
    private var currentHeight: Int = 0

    // MARK: - Configurable Properties

    /// Enable or disable screen-space ambient occlusion.
    var ssaoEnabled: Bool = true

    /// Radius of the SSAO sampling hemisphere (in clip space). Smaller = finer detail.
    var ssaoRadius: Float = 0.05

    /// Intensity of the ambient occlusion darkening (0 = no effect, 1 = full).
    var ssaoIntensity: Float = 0.8

    /// Enable or disable silhouette edge detection.
    var edgeEnabled: Bool = true

    /// Strength of edge darkening (0 = no edges, 1 = strong black outlines).
    var edgeStrength: Float = 0.6

    /// Enable or disable depth of field blur.
    var dofEnabled: Bool = false

    /// The distance from the camera that is in perfect focus (in world units).
    var dofFocalDistance: Float = 15.0

    /// Range around the focal distance that remains sharp (in world units).
    var dofFocalRange: Float = 10.0

    /// Maximum blur radius in pixels at maximum defocus.
    var dofMaxBlur: Float = 6.0

    /// Camera near plane (must match renderer).
    var nearPlane: Float = 0.1

    /// Camera far plane (must match renderer).
    var farPlane: Float = 500.0

    // MARK: - Init

    init(device: MTLDevice) throws {
        self.device = device

        guard let lib = device.makeDefaultLibrary() else {
            throw PostProcessError.libraryCreationFailed
        }
        self.library = lib

        // Build pipeline states from shader functions
        self.ssaoPipeline = try Self.makePipeline(device: device, library: lib, name: "ssaoCompute")
        self.ssaoBlurPipeline = try Self.makePipeline(device: device, library: lib, name: "ssaoBlur")
        self.edgeDetectPipeline = try Self.makePipeline(device: device, library: lib, name: "edgeDetect")
        self.dofPipeline = try Self.makePipeline(device: device, library: lib, name: "depthOfField")
        self.compositePipeline = try Self.makePipeline(device: device, library: lib, name: "composite")
        self.fillPipeline = try Self.makePipeline(device: device, library: lib, name: "fillTexture")

        self.paramsBuffer = device.makeBuffer(
            length: MemoryLayout<GPUPostProcessParams>.stride,
            options: .storageModeShared
        )
        paramsBuffer?.label = "PostProcessParams"
    }

    private static func makePipeline(
        device: MTLDevice,
        library: MTLLibrary,
        name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw PostProcessError.kernelNotFound(name)
        }
        return try device.makeComputePipelineState(function: function)
    }

    // MARK: - Apply Post-Processing

    /// Apply the full post-processing chain to the rendered scene.
    /// - Parameters:
    ///   - colorTexture: The rendered color buffer (read-only)
    ///   - depthTexture: The depth buffer (read-only)
    ///   - normalTexture: Optional view-space normal buffer (unused; normals reconstructed from depth)
    ///   - commandBuffer: The command buffer to encode into
    ///   - outputTexture: The final output texture (write-only)
    func apply(
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        normalTexture: MTLTexture? = nil,
        commandBuffer: MTLCommandBuffer,
        outputTexture: MTLTexture
    ) {
        let width = colorTexture.width
        let height = colorTexture.height

        // Recreate intermediate textures if dimensions changed
        if width != currentWidth || height != currentHeight {
            rebuildIntermediateTextures(width: width, height: height)
            currentWidth = width
            currentHeight = height
        }

        // Update params
        updateParams(width: UInt32(width), height: UInt32(height))

        guard let params = paramsBuffer,
              let ssaoTex = ssaoTexture,
              let ssaoBlurTex = ssaoBlurTexture,
              let edgeTex = edgeTexture,
              let dofTex = dofTexture
        else { return }

        let threadGroupWidth = 16
        let threadGroupHeight = 16
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: threadGroupHeight, depth: 1)
        let gridSize = MTLSize(
            width: (width + threadGroupWidth - 1) / threadGroupWidth,
            height: (height + threadGroupHeight - 1) / threadGroupHeight,
            depth: 1
        )

        // Pass 1: SSAO
        if ssaoEnabled {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SSAO"
                encoder.setComputePipelineState(ssaoPipeline)
                encoder.setTexture(depthTexture, index: 0)
                encoder.setTexture(ssaoTex, index: 1)
                encoder.setBuffer(params, offset: 0, index: 0)
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
                encoder.endEncoding()
            }

            // Pass 1b: SSAO Blur
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SSAO Blur"
                encoder.setComputePipelineState(ssaoBlurPipeline)
                encoder.setTexture(ssaoTex, index: 0)
                encoder.setTexture(depthTexture, index: 1)
                encoder.setTexture(ssaoBlurTex, index: 2)
                encoder.setBuffer(params, offset: 0, index: 0)
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
                encoder.endEncoding()
            }
        } else {
            // Fill SSAO blur texture with white (no occlusion)
            clearTexture(ssaoBlurTex, value: 1.0, commandBuffer: commandBuffer)
        }

        // Pass 2: Edge Detection
        if edgeEnabled {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Edge Detection"
                encoder.setComputePipelineState(edgeDetectPipeline)
                encoder.setTexture(depthTexture, index: 0)
                encoder.setTexture(edgeTex, index: 1)
                encoder.setBuffer(params, offset: 0, index: 0)
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
                encoder.endEncoding()
            }
        } else {
            clearTexture(edgeTex, value: 0.0, commandBuffer: commandBuffer)
        }

        // Pass 3: Depth of Field
        if dofEnabled {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Depth of Field"
                encoder.setComputePipelineState(dofPipeline)
                encoder.setTexture(colorTexture, index: 0)
                encoder.setTexture(depthTexture, index: 1)
                encoder.setTexture(dofTex, index: 2)
                encoder.setBuffer(params, offset: 0, index: 0)
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
                encoder.endEncoding()
            }
        } else {
            // When DOF is disabled, copy the color texture into DOF texture
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.label = "Copy Color to DOF"
                let sourceSize = MTLSize(width: width, height: height, depth: 1)
                blitEncoder.copy(
                    from: colorTexture,
                    sourceSlice: 0,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: sourceSize,
                    to: dofTex,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )
                blitEncoder.endEncoding()
            }
        }

        // Pass 4: Composite
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Composite"
            encoder.setComputePipelineState(compositePipeline)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(ssaoBlurTex, index: 1)
            encoder.setTexture(edgeTex, index: 2)
            encoder.setTexture(dofTex, index: 3)
            encoder.setTexture(outputTexture, index: 4)
            encoder.setBuffer(params, offset: 0, index: 0)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
        }
    }

    // MARK: - Internal Helpers

    private func updateParams(width: UInt32, height: UInt32) {
        guard let buffer = paramsBuffer else { return }

        var params = GPUPostProcessParams(
            ssaoRadius: ssaoRadius,
            ssaoBias: 0.025,
            ssaoNumSamples: 16,
            ssaoStrength: ssaoEnabled ? ssaoIntensity : 0.0,
            edgeStrength: edgeEnabled ? edgeStrength : 0.0,
            nearPlane: nearPlane,
            farPlane: farPlane,
            width: width,
            height: height,
            dofFocalDistance: dofFocalDistance,
            dofFocalRange: dofFocalRange,
            dofMaxBlur: dofMaxBlur
        )

        buffer.contents().copyMemory(from: &params, byteCount: MemoryLayout<GPUPostProcessParams>.stride)
    }

    private func rebuildIntermediateTextures(width: Int, height: Int) {
        // All intermediate textures use R32Float for single-channel data,
        // except DOF which uses the same format as the color buffer.
        ssaoTexture = makeTexture(
            width: width, height: height,
            format: .r32Float,
            label: "SSAO"
        )
        ssaoBlurTexture = makeTexture(
            width: width, height: height,
            format: .r32Float,
            label: "SSAO Blurred"
        )
        edgeTexture = makeTexture(
            width: width, height: height,
            format: .r32Float,
            label: "Edge"
        )
        dofTexture = makeTexture(
            width: width, height: height,
            format: .bgra8Unorm,
            label: "DOF"
        )
    }

    private func makeTexture(width: Int, height: Int, format: MTLPixelFormat, label: String) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        let tex = device.makeTexture(descriptor: desc)
        tex?.label = label
        return tex
    }

    /// Fill a single-channel (R32Float) texture with a constant value using the GPU fill kernel.
    private func clearTexture(_ texture: MTLTexture, value: Float, commandBuffer: MTLCommandBuffer) {
        guard let params = paramsBuffer,
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        encoder.label = "Fill \(texture.label ?? "texture") = \(value)"
        encoder.setComputePipelineState(fillPipeline)
        encoder.setTexture(texture, index: 0)

        var fillValue = value
        encoder.setBytes(&fillValue, length: MemoryLayout<Float>.stride, index: 0)
        encoder.setBuffer(params, offset: 0, index: 1)

        let threadGroupWidth = 16
        let threadGroupHeight = 16
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: threadGroupHeight, depth: 1)
        let gridSize = MTLSize(
            width: (texture.width + threadGroupWidth - 1) / threadGroupWidth,
            height: (texture.height + threadGroupHeight - 1) / threadGroupHeight,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}

// MARK: - Errors

enum PostProcessError: Error, CustomStringConvertible {
    case libraryCreationFailed
    case kernelNotFound(String)
    case textureCreationFailed

    var description: String {
        switch self {
        case .libraryCreationFailed:
            return "Failed to create Metal library for post-processing"
        case .kernelNotFound(let name):
            return "Metal kernel function '\(name)' not found"
        case .textureCreationFailed:
            return "Failed to create intermediate texture for post-processing"
        }
    }
}

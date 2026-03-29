import SwiftUI
import MetalKit
import simd

// MARK: - Rotating Protein Ribbon Background
//
// Uses the actual Druse Metal ribbon renderer (same shaders, same mesh generator)
// to render a slowly rotating protein ribbon as a decorative background.
// Self-contained: creates its own Metal pipeline, no dependency on AppViewModel.
// Renders PDB 1A30 (HIV-1 protease dimer, 198 Cα atoms) with proper secondary structure.

struct WelcomeRibbonBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> MTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return MTKView()
        }
        let view = MTKView(frame: .zero, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1 // No MSAA — simpler, avoids texture mismatch asserts
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.layer?.isOpaque = false
        view.preferredFramesPerSecond = 30
        view.isPaused = false

        let renderer = WelcomeRibbonRenderer(device: device, view: view)
        context.coordinator.renderer = renderer
        view.delegate = context.coordinator

        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        var renderer: WelcomeRibbonRenderer?

        nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            let w = Float(size.width), h = Float(size.height)
            MainActor.assumeIsolated {
                renderer?.viewportSize = SIMD2<Float>(w, h)
            }
        }

        nonisolated func draw(in view: MTKView) {
            MainActor.assumeIsolated {
                renderer?.draw(in: view)
            }
        }
    }
}

// MARK: - Minimal Metal Ribbon Renderer

@MainActor
final class WelcomeRibbonRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ribbonPipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let uniformBuffer: MTLBuffer
    private let ribbonVertexBuffer: MTLBuffer
    private let ribbonIndexBuffer: MTLBuffer
    private let ribbonIndexCount: Int

    var viewportSize: SIMD2<Float> = SIMD2<Float>(800, 600)
    private let startTime = CFAbsoluteTimeGetCurrent()

    init?(device: MTLDevice, view: MTKView) {
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary() else { return nil }
        self.commandQueue = queue

        // Build ribbon pipeline (same shaders as main renderer)
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "WelcomeRibbon"
        desc.vertexFunction = library.makeFunction(name: "ribbonVertex")
        desc.fragmentFunction = library.makeFunction(name: "ribbonFragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        // Enable alpha blending so the background shows through
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float
        desc.rasterSampleCount = 1
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }
        self.ribbonPipeline = pipeline

        // Depth stencil
        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .less
        dsDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: dsDesc) else { return nil }
        self.depthState = ds

        // Uniform buffer
        guard let ub = device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared) else { return nil }
        self.uniformBuffer = ub

        // Generate ribbon mesh from hardcoded protein data
        let (vertices, indices) = Self.generateRibbonMesh()
        guard !vertices.isEmpty else { return nil }

        guard let vb = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<RibbonVertex>.stride, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else { return nil }
        self.ribbonVertexBuffer = vb
        self.ribbonIndexBuffer = ib
        self.ribbonIndexCount = indices.count
    }

    func draw(in view: MTKView) {
        // Skip if view has zero size (not yet laid out)
        let size = view.drawableSize
        guard size.width > 0 && size.height > 0 else { return }

        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        updateUniforms()

        encoder.setRenderPipelineState(ribbonPipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none) // Two-sided ribbon
        encoder.setVertexBuffer(ribbonVertexBuffer, offset: 0, index: Int(BufferIndexVertices.rawValue))
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: ribbonIndexCount,
            indexType: .uint32,
            indexBuffer: ribbonIndexBuffer,
            indexBufferOffset: 0
        )
        encoder.endEncoding()

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    private func updateUniforms() {
        let elapsed = Float(CFAbsoluteTimeGetCurrent() - startTime)
        let angle = elapsed * (2.0 * .pi / 50.0) // Full rotation every 50s

        // Camera orbits around the protein
        let distance: Float = 55.0
        let eyeX = sinf(angle) * distance
        let eyeZ = cosf(angle) * distance
        let eyeY: Float = 15.0 // Slight elevated view

        let eye = SIMD3<Float>(eyeX, eyeY, eyeZ)
        let center = SIMD3<Float>(0, 0, 0)
        let up = SIMD3<Float>(0, 1, 0)

        let aspect = viewportSize.x / max(viewportSize.y, 1)
        let viewMatrix = Mat4.lookAt(eye: eye, center: center, up: up)
        let projMatrix = Mat4.perspective(fovY: .pi / 4.0, aspect: aspect, near: 0.1, far: 500)

        var uniforms = Uniforms()
        uniforms.modelMatrix = matrix_identity_float4x4
        uniforms.viewMatrix = viewMatrix
        uniforms.projectionMatrix = projMatrix
        uniforms.normalMatrix = Mat4.normalMatrix(viewMatrix)
        uniforms.cameraPosition = eye
        uniforms.lightDirection = simd_normalize(SIMD3<Float>(0.5, 1.0, 0.8))
        uniforms.lightColor = SIMD3<Float>(1.0, 0.98, 0.95)
        uniforms.ambientIntensity = 0.25
        uniforms.time = elapsed
        uniforms.selectedAtomIndex = -1
        uniforms.atomRadiusScale = 1.0
        uniforms.bondRadiusScale = 1.0
        uniforms.lightingMode = 1 // Directional lighting for nice shading
        uniforms.enableClipping = 0
        uniforms.themeMode = 0
        uniforms.backgroundOpacity = 0.0 // Transparent background
        uniforms.surfaceOpacity = 1.0
        uniforms.gridLineWidth = 0

        uniformBuffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<Uniforms>.stride)
    }

    // MARK: - Ribbon Mesh from Hardcoded PDB 1A30

    private static func generateRibbonMesh() -> (vertices: [RibbonVertex], indices: [UInt32]) {
        // Cα positions from PDB 1A30 (HIV-1 Protease, centered at origin)
        let caPositions: [SIMD3<Float>] = [
            SIMD3<Float>(14.1, 12.9, 1.0), SIMD3<Float>(13.9, 10.3, -1.8),
            SIMD3<Float>(13.0, 6.8, -0.7), SIMD3<Float>(13.8, 3.9, -3.0),
            SIMD3<Float>(11.5, 0.8, -2.9), SIMD3<Float>(14.2, -1.9, -2.8),
            SIMD3<Float>(12.8, -2.7, 0.7), SIMD3<Float>(9.9, -1.7, 2.8),
            SIMD3<Float>(9.5, 2.1, 3.1), SIMD3<Float>(9.5, 2.1, 7.0),
            SIMD3<Float>(9.9, 5.4, 8.9), SIMD3<Float>(9.8, 6.5, 12.5),
            SIMD3<Float>(6.7, 8.3, 13.6), SIMD3<Float>(5.4, 10.0, 16.7),
            SIMD3<Float>(1.8, 9.6, 17.8), SIMD3<Float>(0.3, 10.1, 21.2),
            SIMD3<Float>(3.7, 10.8, 22.6), SIMD3<Float>(5.0, 7.4, 21.5),
            SIMD3<Float>(7.7, 6.7, 18.9), SIMD3<Float>(6.8, 3.9, 16.5),
            SIMD3<Float>(7.7, 2.4, 13.1), SIMD3<Float>(5.3, 2.6, 10.2),
            SIMD3<Float>(5.1, 1.8, 6.5), SIMD3<Float>(4.6, 4.5, 3.9),
            SIMD3<Float>(1.6, 3.0, 2.1), SIMD3<Float>(0.3, 4.6, -1.1),
            SIMD3<Float>(-2.1, 1.7, -1.4), SIMD3<Float>(-3.9, 2.8, 1.8),
            SIMD3<Float>(-6.8, 5.3, 1.9), SIMD3<Float>(-6.4, 5.6, 5.7),
            SIMD3<Float>(-3.7, 5.5, 8.4), SIMD3<Float>(-4.0, 2.5, 10.6),
            SIMD3<Float>(-2.0, 2.1, 13.9), SIMD3<Float>(-1.7, -1.1, 16.0),
            SIMD3<Float>(-3.8, -1.2, 19.1), SIMD3<Float>(-3.0, 1.7, 21.4),
            SIMD3<Float>(-4.8, 4.5, 23.3), SIMD3<Float>(-5.6, 7.9, 21.9),
            SIMD3<Float>(-7.6, 10.7, 23.6), SIMD3<Float>(-10.9, 12.0, 22.3),
            SIMD3<Float>(-14.1, 10.5, 21.0), SIMD3<Float>(-14.0, 7.6, 18.5),
            SIMD3<Float>(-16.5, 5.7, 16.3), SIMD3<Float>(-16.4, 2.2, 15.0),
            SIMD3<Float>(-15.4, 1.6, 11.4), SIMD3<Float>(-14.7, -1.4, 9.2),
            SIMD3<Float>(-11.8, -1.3, 6.8), SIMD3<Float>(-10.8, -3.9, 4.2),
            SIMD3<Float>(-8.0, -5.1, 2.0), SIMD3<Float>(-6.2, -8.3, 1.3),
            SIMD3<Float>(-7.7, -11.0, 3.4), SIMD3<Float>(-10.9, -9.2, 4.3),
            SIMD3<Float>(-12.0, -6.6, 6.8), SIMD3<Float>(-11.1, -5.6, 10.4),
            SIMD3<Float>(-13.0, -3.4, 12.9), SIMD3<Float>(-11.2, -0.3, 14.0),
            SIMD3<Float>(-11.7, 2.8, 16.2), SIMD3<Float>(-11.7, 6.0, 14.2),
            SIMD3<Float>(-10.1, 9.0, 15.9), SIMD3<Float>(-10.2, 12.4, 14.1),
            SIMD3<Float>(-7.8, 15.3, 14.2), SIMD3<Float>(-4.9, 13.4, 15.6),
            SIMD3<Float>(-1.4, 14.9, 15.2), SIMD3<Float>(1.4, 12.6, 14.0),
            SIMD3<Float>(5.0, 13.4, 13.2), SIMD3<Float>(6.4, 11.5, 10.1),
            SIMD3<Float>(10.1, 12.1, 9.4), SIMD3<Float>(10.2, 15.6, 11.1),
            SIMD3<Float>(7.0, 16.6, 9.2), SIMD3<Float>(4.1, 17.3, 11.4),
            SIMD3<Float>(0.6, 16.5, 10.1), SIMD3<Float>(-3.0, 16.1, 11.4),
            SIMD3<Float>(-5.8, 13.8, 10.3), SIMD3<Float>(-8.0, 10.8, 11.1),
            SIMD3<Float>(-6.1, 7.9, 12.6), SIMD3<Float>(-7.7, 4.4, 12.8),
            SIMD3<Float>(-6.6, 2.1, 15.7), SIMD3<Float>(-7.1, -1.6, 15.5),
            SIMD3<Float>(-5.7, -5.2, 14.8), SIMD3<Float>(-3.5, -4.2, 11.9),
            SIMD3<Float>(-0.4, -6.5, 11.3), SIMD3<Float>(1.8, -3.4, 10.7),
            SIMD3<Float>(1.5, 0.3, 11.3), SIMD3<Float>(0.6, 2.0, 8.0),
            SIMD3<Float>(0.6, 5.7, 7.0), SIMD3<Float>(-2.0, 6.1, 4.3),
            SIMD3<Float>(-2.8, 8.8, 1.7), SIMD3<Float>(-4.7, 11.0, 4.2),
            SIMD3<Float>(-1.4, 11.8, 5.9), SIMD3<Float>(1.0, 11.1, 3.1),
            SIMD3<Float>(-0.4, 14.1, 1.2), SIMD3<Float>(0.0, 16.4, 4.2),
            SIMD3<Float>(3.8, 15.8, 4.4), SIMD3<Float>(4.0, 16.4, 0.6),
            SIMD3<Float>(4.8, 12.8, -0.4), SIMD3<Float>(4.4, 11.8, -4.1),
            SIMD3<Float>(5.3, 8.9, -6.3), SIMD3<Float>(8.0, 9.6, -8.9),
            SIMD3<Float>(9.3, 7.6, -11.9), SIMD3<Float>(5.8, 11.4, -14.1),
            SIMD3<Float>(4.8, 12.7, -10.6), SIMD3<Float>(1.7, 11.3, -8.9),
            SIMD3<Float>(0.2, 13.1, -5.9), SIMD3<Float>(-1.8, 11.1, -3.4),
            SIMD3<Float>(-5.0, 13.2, -3.0), SIMD3<Float>(-6.8, 10.4, -4.8),
            SIMD3<Float>(-5.9, 6.7, -5.3), SIMD3<Float>(-2.8, 6.2, -7.4),
            SIMD3<Float>(-4.5, 4.0, -10.1), SIMD3<Float>(-2.8, 3.6, -13.4),
            SIMD3<Float>(-3.5, 1.6, -16.6), SIMD3<Float>(-1.5, -1.6, -16.9),
            SIMD3<Float>(-1.3, -4.0, -19.8), SIMD3<Float>(-1.0, -7.7, -19.0),
            SIMD3<Float>(-1.5, -10.2, -21.8), SIMD3<Float>(-2.8, -7.6, -24.2),
            SIMD3<Float>(-5.7, -6.7, -22.0), SIMD3<Float>(-5.8, -3.3, -20.2),
            SIMD3<Float>(-6.6, -3.2, -16.5), SIMD3<Float>(-6.6, -0.7, -13.7),
            SIMD3<Float>(-4.0, -1.2, -10.9), SIMD3<Float>(-3.0, 0.7, -7.7),
            SIMD3<Float>(0.6, 1.7, -7.3), SIMD3<Float>(1.1, 0.3, -3.8),
            SIMD3<Float>(4.3, 1.0, -1.9), SIMD3<Float>(2.8, -0.9, 1.0),
            SIMD3<Float>(3.0, -4.1, -1.0), SIMD3<Float>(6.0, -6.4, -1.3),
            SIMD3<Float>(4.4, -8.1, -4.3), SIMD3<Float>(2.2, -7.5, -7.3),
            SIMD3<Float>(-1.2, -9.2, -7.0), SIMD3<Float>(-3.7, -9.3, -9.8),
            SIMD3<Float>(-7.4, -10.4, -9.8), SIMD3<Float>(-8.3, -13.8, -11.2),
            SIMD3<Float>(-7.3, -14.1, -14.8), SIMD3<Float>(-5.4, -16.6, -17.2),
            SIMD3<Float>(-1.6, -16.4, -17.3), SIMD3<Float>(0.7, -18.8, -19.3),
            SIMD3<Float>(3.3, -21.1, -17.7), SIMD3<Float>(3.4, -23.1, -14.5),
            SIMD3<Float>(2.7, -21.6, -11.1), SIMD3<Float>(3.8, -22.2, -7.6),
            SIMD3<Float>(1.8, -21.8, -4.3), SIMD3<Float>(2.2, -18.6, -2.2),
            SIMD3<Float>(0.5, -17.2, 0.9), SIMD3<Float>(0.3, -13.4, 1.3),
            SIMD3<Float>(-0.8, -11.4, 4.3), SIMD3<Float>(-2.8, -8.2, 4.2),
            SIMD3<Float>(-4.9, -6.0, 6.5), SIMD3<Float>(-7.7, -8.6, 6.9),
            SIMD3<Float>(-5.6, -11.8, 7.1), SIMD3<Float>(-4.0, -14.2, 4.7),
            SIMD3<Float>(-5.0, -15.4, 1.2), SIMD3<Float>(-3.6, -18.1, -1.1),
            SIMD3<Float>(-2.4, -16.9, -4.5), SIMD3<Float>(-0.8, -18.5, -7.6),
            SIMD3<Float>(2.7, -17.3, -8.4), SIMD3<Float>(3.7, -16.7, -12.0),
            SIMD3<Float>(7.4, -15.8, -12.7), SIMD3<Float>(8.7, -13.7, -15.6),
            SIMD3<Float>(5.4, -12.2, -16.6), SIMD3<Float>(5.5, -9.3, -19.1),
            SIMD3<Float>(3.4, -6.4, -17.8), SIMD3<Float>(3.2, -2.8, -19.0),
            SIMD3<Float>(2.7, -0.3, -16.2), SIMD3<Float>(2.0, 3.3, -17.2),
            SIMD3<Float>(4.0, 3.0, -20.5), SIMD3<Float>(6.9, 1.3, -18.8),
            SIMD3<Float>(7.6, -2.4, -19.5), SIMD3<Float>(8.3, -4.6, -16.5),
            SIMD3<Float>(8.9, -8.4, -16.5), SIMD3<Float>(8.7, -10.1, -13.2),
            SIMD3<Float>(6.7, -12.1, -10.6), SIMD3<Float>(2.9, -11.7, -10.7),
            SIMD3<Float>(0.6, -13.2, -8.1), SIMD3<Float>(-3.0, -14.1, -9.1),
            SIMD3<Float>(-5.7, -14.5, -6.4), SIMD3<Float>(-8.7, -13.0, -4.5),
            SIMD3<Float>(-7.5, -9.4, -4.1), SIMD3<Float>(-10.0, -6.5, -3.7),
            SIMD3<Float>(-8.1, -4.5, -6.4), SIMD3<Float>(-5.1, -5.1, -8.7),
            SIMD3<Float>(-2.0, -3.9, -7.0), SIMD3<Float>(1.5, -3.1, -8.3),
            SIMD3<Float>(3.9, -4.0, -5.5), SIMD3<Float>(7.5, -2.9, -4.9),
            SIMD3<Float>(8.9, -5.8, -7.0), SIMD3<Float>(7.7, -4.1, -10.1),
            SIMD3<Float>(7.7, -0.5, -8.9), SIMD3<Float>(11.5, -0.8, -8.6),
            SIMD3<Float>(11.6, -2.0, -12.3), SIMD3<Float>(9.9, 1.1, -13.5),
            SIMD3<Float>(12.2, 3.2, -11.2), SIMD3<Float>(9.5, 4.5, -8.9),
            SIMD3<Float>(10.4, 6.2, -5.6), SIMD3<Float>(8.6, 8.1, -2.8),
            SIMD3<Float>(9.6, 11.8, -2.4), SIMD3<Float>(9.0, 14.5, 0.1),
        ]

        // Approximate secondary structure assignment for HIV-1 protease
        // Chain A: residues 1-99, Chain B: residues 100-198
        // Known SS: beta sheets at key positions, with coil/turn between
        let n = caPositions.count
        var ss = [SecondaryStructure](repeating: .coil, count: n)

        // HIV-1 protease has predominantly beta-sheet structure
        // Sheet regions (approximate, per chain of 99 residues):
        let sheetRanges = [
            1...4, 9...15, 18...24, 30...35, 43...49, 52...57, 63...66,
            69...72, 74...78, 83...85, 87...90, 95...98
        ]
        for range in sheetRanges {
            for i in range where i < 99 { ss[i] = .sheet }
            for i in range where i < 99 { ss[i + 99] = .sheet } // Mirror for chain B
        }
        // Short helix (flap region)
        for i in 46...52 where i < 99 { ss[i] = .helix }
        for i in 46...52 where i < 99 { ss[i + 99] = .helix }

        return RibbonMeshGenerator.generate(
            caPositions: caPositions,
            ssAssignments: ss
        )
    }
}

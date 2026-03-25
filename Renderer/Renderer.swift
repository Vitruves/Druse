import MetalKit
import simd

@MainActor
final class Renderer: NSObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let camera = Camera()

    // Pipeline states
    private var backgroundPipeline: MTLRenderPipelineState!
    private var atomPipeline: MTLRenderPipelineState!
    private var bondPipeline: MTLRenderPipelineState!
    private var ribbonPipeline: MTLRenderPipelineState!
    private var interactionLinePipeline: MTLRenderPipelineState!

    // Depth stencil states
    private var depthStateDisabled: MTLDepthStencilState!
    private var depthStateReadWrite: MTLDepthStencilState!

    // Triple-buffered uniforms to avoid CPU-GPU race on shared memory
    private static let maxFramesInFlight = 3
    private var uniformBuffers: [MTLBuffer] = []
    private var currentUniformBufferIndex = 0
    private let frameSemaphore = DispatchSemaphore(value: maxFramesInFlight)
    private var atomInstanceBuffer: MTLBuffer?
    private var bondInstanceBuffer: MTLBuffer?
    private var atomInstanceCount: Int = 0
    private var bondInstanceCount: Int = 0
    private var atomInstanceCapacity: Int = 0   // pooled capacity (in instance count)
    private var bondInstanceCapacity: Int = 0

    // Ribbon buffers
    private var ribbonVertexBuffer: MTLBuffer?
    private var ribbonIndexBuffer: MTLBuffer?
    private var ribbonIndexCount: Int = 0

    // Ribbon CA control points for hit-testing in ribbon mode
    // Each entry: (position, originalProteinAtomID)
    private var ribbonCAControlPoints: [(position: SIMD3<Float>, atomID: Int)] = []

    // Interaction line buffers
    private var interactionLineBuffer: MTLBuffer?
    private var interactionLineCount: Int = 0
    private var constraintIndicatorBuffer: MTLBuffer?
    private var constraintIndicatorCount: Int = 0

    // Grid box wireframe
    private var gridBoxPipeline: MTLRenderPipelineState!
    private var gridBoxVertexBuffer: MTLBuffer?
    private var gridBoxVertexCount: Int = 0

    // Molecular surface
    private var surfacePipeline: MTLRenderPipelineState!
    private var surfaceVertexBuffer: MTLBuffer?
    private var surfaceIndexBuffer: MTLBuffer?
    private var surfaceIndexCount: Int = 0

    // Ghost ligand (translucent docking preview — does not mutate active ligand)
    private var ghostAtomPipeline: MTLRenderPipelineState!
    private var ghostBondPipeline: MTLRenderPipelineState!
    private var ghostAtomBuffer: MTLBuffer?
    private var ghostBondBuffer: MTLBuffer?
    private var ghostAtomCount: Int = 0
    private var ghostBondCount: Int = 0
    private var depthStateReadOnly: MTLDepthStencilState!

    // GPU object-ID picking (renders atom IDs to an off-screen R32Uint texture)
    private var pickPipeline: MTLRenderPipelineState!
    private var pickTexture: MTLTexture?
    private var pickDepthTexture: MTLTexture?
    private var pickTextureWidth: Int = 0
    private var pickTextureHeight: Int = 0

    // State
    private var startTime = CFAbsoluteTimeGetCurrent()
    var selectedAtomIndex: Int = -1
    var selectedResidueAtomIndices: Set<Int> = []
    var renderMode: RenderMode = .ballAndStick
    var lightingMode: Int32 = 0 // 0 = uniform, 1 = directional
    var themeMode: Int32 = 0    // 0 = dark, 1 = light
    var backgroundOpacity: Float = 1.0
    var surfaceOpacity: Float = 0.85
    var gridLineWidth: Float = 2.5

    // Z-slab clipping
    var enableClipping: Bool = false
    var clipNearZ: Float = 0    // view-space distance from camera to near clip (legacy/fallback)
    var clipFarZ: Float = 100   // view-space distance from camera to far clip (legacy/fallback)

    // Object-space slab clipping (zoom-invariant)
    var slabCenter: SIMD3<Float>? = nil
    var slabHalfThickness: Float = 10.0
    var slabOffset: Float = 0.0

    // Ligand Focus color overrides (set from AppViewModel based on WorkspaceState.colorScheme)
    /// When set, protein atoms use this uniform color instead of element colors
    var uniformProteinColor: SIMD3<Float>? = nil
    /// When set, ligand carbon atoms use this color (e.g., gold/yellow for Ligand Focus)
    var ligandCarbonColor: SIMD3<Float>? = nil
    /// Number of protein atoms in the current data (used to distinguish protein vs ligand)
    var proteinAtomCount: Int = 0
    /// Per-chain color map for chain-colored mode (protein atoms only; ligand keeps CPK)
    var chainColorMap: [String: SIMD3<Float>] = [:]

    /// Whether a ligand is currently visible (controls interaction lines and grid visibility)
    var ligandVisible: Bool = true

    // Molecule data (set from AppViewModel)
    private var currentAtoms: [Atom] = []
    private var currentBonds: [Bond] = []

    // All molecule atom positions for camera fitting (independent of render mode filtering)
    private var allMoleculePositions: [SIMD3<Float>] = []

    init?(mtkView: MTKView) {
        guard let device = mtkView.device ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let lib = device.makeDefaultLibrary()
        else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
        self.library = lib

        super.init()

        mtkView.device = device
        buildPipelines(mtkView: mtkView)
        buildDepthStencilStates()
        buildBuffers()
    }

    // MARK: - Pipeline Construction

    private func buildPipelines(mtkView: MTKView) {
        let colorFormat = mtkView.colorPixelFormat
        let depthFormat = mtkView.depthStencilPixelFormat
        let sampleCount = mtkView.sampleCount

        // Background pipeline (no depth write)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "Background"
            desc.vertexFunction = library.makeFunction(name: "backgroundVertex")
            desc.fragmentFunction = library.makeFunction(name: "backgroundFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            backgroundPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create background pipeline: \(error)")
        }

        // Atom pipeline (impostor spheres, depth write from fragment)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "Atoms"
            desc.vertexFunction = library.makeFunction(name: "atomVertex")
            desc.fragmentFunction = library.makeFunction(name: "atomFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            atomPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create atom pipeline: \(error)")
        }

        // Bond pipeline (impostor cylinders, depth write from fragment)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "Bonds"
            desc.vertexFunction = library.makeFunction(name: "bondVertex")
            desc.fragmentFunction = library.makeFunction(name: "bondFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            bondPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create bond pipeline: \(error)")
        }

        // Ribbon pipeline (standard triangle mesh)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "Ribbon"
            desc.vertexFunction = library.makeFunction(name: "ribbonVertex")
            desc.fragmentFunction = library.makeFunction(name: "ribbonFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            ribbonPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create ribbon pipeline: \(error)")
        }

        // Interaction line pipeline (dashed lines between ligand and protein)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "InteractionLines"
            desc.vertexFunction = library.makeFunction(name: "interactionLineVertex")
            desc.fragmentFunction = library.makeFunction(name: "interactionLineFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            // Enable alpha blending for translucent lines
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            interactionLinePipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create interaction line pipeline: \(error)")
        }

        // Grid box wireframe pipeline (line primitives for docking grid visualization)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "GridBox"
            desc.vertexFunction = library.makeFunction(name: "gridBoxVertex")
            desc.fragmentFunction = library.makeFunction(name: "gridBoxFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            gridBoxPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create grid box pipeline: \(error)")
        }

        // Molecular surface pipeline (triangle mesh from marching cubes)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "MolecularSurface"
            desc.vertexFunction = library.makeFunction(name: "surfaceVertex")
            desc.fragmentFunction = library.makeFunction(name: "surfaceFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            surfacePipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create molecular surface pipeline: \(error)")
        }

        // Ghost atom pipeline (translucent impostor spheres for live docking preview)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "GhostAtoms"
            desc.vertexFunction = library.makeFunction(name: "atomVertex")
            desc.fragmentFunction = library.makeFunction(name: "atomFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[0].sourceAlphaBlendFactor = .one
            desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            ghostAtomPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create ghost atom pipeline: \(error)")
        }

        // Ghost bond pipeline (translucent impostor cylinders)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "GhostBonds"
            desc.vertexFunction = library.makeFunction(name: "bondVertex")
            desc.fragmentFunction = library.makeFunction(name: "bondFragment")
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[0].sourceAlphaBlendFactor = .one
            desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = sampleCount
            ghostBondPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create ghost bond pipeline: \(error)")
        }

        // GPU pick pipeline (non-MSAA, writes atom ID to R32Uint texture)
        do {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "AtomPick"
            desc.vertexFunction = library.makeFunction(name: "atomVertex")
            desc.fragmentFunction = library.makeFunction(name: "atomPickFragment")
            desc.colorAttachments[0].pixelFormat = .r32Uint
            desc.depthAttachmentPixelFormat = .depth32Float
            desc.rasterSampleCount = 1  // no MSAA for pick buffer
            pickPipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            fatalError("Failed to create pick pipeline: \(error)")
        }
    }

    private func buildDepthStencilStates() {
        // No depth test/write (for background)
        let disabledDesc = MTLDepthStencilDescriptor()
        disabledDesc.depthCompareFunction = .always
        disabledDesc.isDepthWriteEnabled = false
        depthStateDisabled = device.makeDepthStencilState(descriptor: disabledDesc)

        // Standard depth test + write (for atoms, bonds)
        let rwDesc = MTLDepthStencilDescriptor()
        rwDesc.depthCompareFunction = .less
        rwDesc.isDepthWriteEnabled = true
        depthStateReadWrite = device.makeDepthStencilState(descriptor: rwDesc)

        // Read-only depth test (for ghost/translucent passes — test but don't write)
        let roDesc = MTLDepthStencilDescriptor()
        roDesc.depthCompareFunction = .less
        roDesc.isDepthWriteEnabled = false
        depthStateReadOnly = device.makeDepthStencilState(descriptor: roDesc)
    }

    private func buildBuffers() {
        uniformBuffers = (0..<Self.maxFramesInFlight).map { i in
            let buf = device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared)!
            buf.label = "Uniforms[\(i)]"
            return buf
        }
    }

    // MARK: - Update Molecule Data

    func updateMoleculeData(atoms: [Atom], bonds: [Bond]) {
        currentAtoms = atoms
        currentBonds = bonds
        rebuildInstanceBuffers()
    }

    /// Store all molecule atom positions for camera fitting, independent of render mode.
    /// Call this from AppViewModel before fitToContent() so ribbon mode still centers correctly.
    func updateAllMoleculePositions(_ positions: [SIMD3<Float>]) {
        allMoleculePositions = positions
    }

    private func rebuildInstanceBuffers() {
        // Atom instances
        let atomScale = renderMode.atomRadiusScale
        var atomInstances: [AtomInstance] = []
        atomInstances.reserveCapacity(currentAtoms.count)

        for atom in currentAtoms {
            var flags: Int32 = 0
            if atom.id == selectedAtomIndex { flags |= 1 }
            if selectedResidueAtomIndices.contains(atom.id) { flags |= 2 }

            // Apply color overrides
            let isProteinAtom = atom.id < proteinAtomCount
            var color = atom.element.color
            if !chainColorMap.isEmpty, isProteinAtom {
                // Per-chain coloring mode
                if let chainColor = chainColorMap[atom.chainID] {
                    color = SIMD4<Float>(chainColor.x, chainColor.y, chainColor.z, 1.0)
                }
            } else if isProteinAtom, let protColor = uniformProteinColor {
                color = SIMD4<Float>(protColor.x, protColor.y, protColor.z, 1.0)
            } else if !isProteinAtom, atom.element == .C, let ligCarbon = ligandCarbonColor {
                color = SIMD4<Float>(ligCarbon.x, ligCarbon.y, ligCarbon.z, 1.0)
            }

            atomInstances.append(AtomInstance(
                position: atom.position,
                radius: atom.element.vdwRadius * atomScale,
                color: color,
                atomIndex: Int32(atom.id),
                flags: flags,
                _pad0: 0,
                _pad1: 0
            ))
        }

        atomInstanceCount = atomInstances.count
        if atomInstanceCount > 0 {
            let size = atomInstanceCount * MemoryLayout<AtomInstance>.stride
            if atomInstanceCount > atomInstanceCapacity {
                // Grow with 50% headroom to reduce future reallocations
                let allocCount = max(atomInstanceCount, atomInstanceCapacity * 3 / 2)
                let allocSize = allocCount * MemoryLayout<AtomInstance>.stride
                atomInstanceBuffer = device.makeBuffer(length: allocSize, options: .storageModeShared)
                atomInstanceBuffer?.label = "AtomInstances"
                atomInstanceCapacity = allocCount
            }
            atomInstanceBuffer?.contents().copyMemory(from: atomInstances, byteCount: size)
        } else {
            atomInstanceBuffer = nil
            atomInstanceCapacity = 0
        }

        // Bond instances
        let bondScale = renderMode.bondRadiusScale
        guard bondScale > 0 else {
            bondInstanceCount = 0
            // Keep bondInstanceBuffer and bondInstanceCapacity intact so bonds
            // reappear immediately when switching back to a bond-showing mode.
            return
        }

        var bondInstances: [BondInstance] = []
        bondInstances.reserveCapacity(currentBonds.count * 2) // headroom for double/triple bonds

        for bond in currentBonds {
            guard bond.atomIndex1 < currentAtoms.count,
                  bond.atomIndex2 < currentAtoms.count else { continue }

            let a1 = currentAtoms[bond.atomIndex1]
            let a2 = currentAtoms[bond.atomIndex2]

            // Apply color overrides to bond half-colors
            func atomColor(_ atom: Atom) -> SIMD4<Float> {
                let isProt = atom.id < proteinAtomCount
                if !chainColorMap.isEmpty, isProt {
                    if let chainColor = chainColorMap[atom.chainID] {
                        return SIMD4<Float>(chainColor.x, chainColor.y, chainColor.z, 1.0)
                    }
                } else if isProt, let protColor = uniformProteinColor {
                    return SIMD4<Float>(protColor.x, protColor.y, protColor.z, 1.0)
                } else if !isProt, atom.element == .C, let ligCarbon = ligandCarbonColor {
                    return SIMD4<Float>(ligCarbon.x, ligCarbon.y, ligCarbon.z, 1.0)
                }
                return atom.element.color
            }

            let cA = atomColor(a1)
            let cB = atomColor(a2)

            switch bond.order {
            case .single:
                let r = bond.order.displayRadius * bondScale
                bondInstances.append(BondInstance(
                    positionA: a1.position, radiusA: r,
                    positionB: a2.position, radiusB: r,
                    colorA: cA, colorB: cB
                ))

            case .aromatic:
                // Solid cylinder + thinner parallel cylinder ("1.5 bond")
                let bondVec = a2.position - a1.position
                let bondLen = simd_length(bondVec)
                let r = BondOrder.single.displayRadius * bondScale
                let rThin = r * 0.55  // thinner secondary cylinder
                guard bondLen > 0.01 else {
                    bondInstances.append(BondInstance(
                        positionA: a1.position, radiusA: r,
                        positionB: a2.position, radiusB: r,
                        colorA: cA, colorB: cB
                    ))
                    continue
                }
                let bondDir = bondVec / bondLen
                let ref: SIMD3<Float> = abs(bondDir.y) < 0.99
                    ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
                let perp = simd_normalize(simd_cross(bondDir, ref))
                let separation: Float = r * 1.3

                // Primary solid cylinder (offset slightly)
                let offA = perp * (-separation * 0.5)
                bondInstances.append(BondInstance(
                    positionA: a1.position + offA, radiusA: r,
                    positionB: a2.position + offA, radiusB: r,
                    colorA: cA, colorB: cB
                ))
                // Secondary thinner cylinder (aromatic partial bond)
                let offB = perp * (separation * 0.5)
                bondInstances.append(BondInstance(
                    positionA: a1.position + offB, radiusA: rThin,
                    positionB: a2.position + offB, radiusB: rThin,
                    colorA: cA, colorB: cB
                ))

            case .double:
                // Two parallel cylinders offset perpendicular to the bond axis
                let bondVec = a2.position - a1.position
                let bondLen = simd_length(bondVec)
                let r = bond.order.displayRadius * bondScale
                guard bondLen > 0.01 else {
                    bondInstances.append(BondInstance(
                        positionA: a1.position, radiusA: r,
                        positionB: a2.position, radiusB: r,
                        colorA: cA, colorB: cB
                    ))
                    continue
                }
                let bondDir = bondVec / bondLen
                let ref: SIMD3<Float> = abs(bondDir.y) < 0.99
                    ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
                let perp = simd_normalize(simd_cross(bondDir, ref))
                let separation: Float = r * 1.4  // gap between cylinder centers

                for sign: Float in [-1.0, 1.0] {
                    let off = perp * (sign * separation * 0.5)
                    bondInstances.append(BondInstance(
                        positionA: a1.position + off, radiusA: r,
                        positionB: a2.position + off, radiusB: r,
                        colorA: cA, colorB: cB
                    ))
                }

            case .triple:
                // Three parallel cylinders: one centered, two offset
                let bondVec = a2.position - a1.position
                let bondLen = simd_length(bondVec)
                let r = bond.order.displayRadius * bondScale
                guard bondLen > 0.01 else {
                    bondInstances.append(BondInstance(
                        positionA: a1.position, radiusA: r,
                        positionB: a2.position, radiusB: r,
                        colorA: cA, colorB: cB
                    ))
                    continue
                }
                let bondDir = bondVec / bondLen
                let ref: SIMD3<Float> = abs(bondDir.y) < 0.99
                    ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
                let perp = simd_normalize(simd_cross(bondDir, ref))
                let separation: Float = r * 1.5  // gap from center to outer cylinder

                // Center cylinder
                bondInstances.append(BondInstance(
                    positionA: a1.position, radiusA: r,
                    positionB: a2.position, radiusB: r,
                    colorA: cA, colorB: cB
                ))
                // Two outer cylinders
                for sign: Float in [-1.0, 1.0] {
                    let off = perp * (sign * separation)
                    bondInstances.append(BondInstance(
                        positionA: a1.position + off, radiusA: r,
                        positionB: a2.position + off, radiusB: r,
                        colorA: cA, colorB: cB
                    ))
                }
            }
        }

        bondInstanceCount = bondInstances.count
        if bondInstanceCount > 0 {
            let size = bondInstanceCount * MemoryLayout<BondInstance>.stride
            if bondInstanceCount > bondInstanceCapacity {
                let allocCount = max(bondInstanceCount, bondInstanceCapacity * 3 / 2)
                let allocSize = allocCount * MemoryLayout<BondInstance>.stride
                bondInstanceBuffer = device.makeBuffer(length: allocSize, options: .storageModeShared)
                bondInstanceBuffer?.label = "BondInstances"
                bondInstanceCapacity = allocCount
            }
            bondInstanceBuffer?.contents().copyMemory(from: bondInstances, byteCount: size)
        } else {
            bondInstanceBuffer = nil
            bondInstanceCapacity = 0
        }
    }

    // MARK: - Ribbon Mesh

    func updateRibbonMesh(vertices: [RibbonVertex], indices: [UInt32]) {
        guard !vertices.isEmpty, !indices.isEmpty else {
            ribbonVertexBuffer = nil
            ribbonIndexBuffer = nil
            ribbonIndexCount = 0
            return
        }

        let vertSize = vertices.count * MemoryLayout<RibbonVertex>.stride
        ribbonVertexBuffer = device.makeBuffer(bytes: vertices, length: vertSize, options: .storageModeShared)
        ribbonVertexBuffer?.label = "RibbonVertices"

        let idxSize = indices.count * MemoryLayout<UInt32>.stride
        ribbonIndexBuffer = device.makeBuffer(bytes: indices, length: idxSize, options: .storageModeShared)
        ribbonIndexBuffer?.label = "RibbonIndices"

        ribbonIndexCount = indices.count
    }

    func updateRibbonCAControlPoints(_ points: [(position: SIMD3<Float>, atomID: Int)]) {
        ribbonCAControlPoints = points
    }

    func clearRibbonMesh() {
        ribbonVertexBuffer = nil
        ribbonIndexBuffer = nil
        ribbonIndexCount = 0
        ribbonCAControlPoints = []
    }

    // MARK: - Interaction Lines

    func updateInteractionLines(_ interactions: [MolecularInteraction]) {
        guard !interactions.isEmpty else {
            interactionLineBuffer = nil
            interactionLineCount = 0
            return
        }

        var lines: [InteractionLineVertex] = interactions.map { inter in
            InteractionLineVertex(
                positionA: inter.ligandPosition,
                positionB: inter.proteinPosition,
                color: inter.type.color,
                dashLength: 0.15,
                interactionType: Int32(inter.type.rawValue),
                _pad0: 0, _pad1: 0
            )
        }

        interactionLineCount = lines.count
        interactionLineBuffer = device.makeBuffer(
            bytes: &lines, length: lines.count * MemoryLayout<InteractionLineVertex>.stride,
            options: .storageModeShared
        )
        interactionLineBuffer?.label = "InteractionLines"
    }

    func clearInteractionLines() {
        interactionLineBuffer = nil
        interactionLineCount = 0
    }

    // MARK: - Constraint Indicators

    /// Render pharmacophore constraints as colored lines from source atom to target position.
    /// Uses the same InteractionLineVertex pipeline but with solid lines (dashLength = 0).
    func updateConstraintIndicators(_ constraints: [PharmacophoreConstraintDef], atoms: [Atom]) {
        guard !constraints.isEmpty else {
            constraintIndicatorBuffer = nil
            constraintIndicatorCount = 0
            return
        }

        var lines: [InteractionLineVertex] = []
        for constraint in constraints where constraint.isEnabled {
            let color = constraint.interactionType.color
            // Draw a line from the source atom to each target position
            let sourcePos: SIMD3<Float>
            if let atomIdx = constraint.proteinAtomIndex ?? constraint.ligandAtomIndex,
               atomIdx >= 0, atomIdx < atoms.count {
                sourcePos = atoms[atomIdx].position
            } else if constraint.residueIndex != nil,
                      let positions = constraint.targetPositions.first {
                // Use first target position as source for residue-level constraints
                sourcePos = positions
            } else {
                continue
            }

            for targetPos in constraint.targetPositions {
                lines.append(InteractionLineVertex(
                    positionA: sourcePos,
                    positionB: targetPos,
                    color: color,
                    dashLength: constraint.strength.isHard ? 0.0 : 0.1,  // solid for hard, dashed for soft
                    interactionType: Int32(constraint.interactionType.gpuType),
                    _pad0: 0, _pad1: 0
                ))
            }
        }

        constraintIndicatorCount = lines.count
        if lines.isEmpty {
            constraintIndicatorBuffer = nil
        } else {
            constraintIndicatorBuffer = device.makeBuffer(
                bytes: &lines, length: lines.count * MemoryLayout<InteractionLineVertex>.stride,
                options: .storageModeShared
            )
            constraintIndicatorBuffer?.label = "ConstraintIndicators"
        }
    }

    func clearConstraintIndicators() {
        constraintIndicatorBuffer = nil
        constraintIndicatorCount = 0
    }

    // MARK: - Ghost Ligand Pose (translucent docking preview)

    /// Update the ghost ligand overlay with a docked pose.
    /// Rendered as translucent spheres/bonds so the original ligand position is preserved.
    func updateGhostPose(atoms: [Atom], bonds: [Bond]) {
        guard !atoms.isEmpty else { clearGhostPose(); return }

        let ghostAlpha: Float = 1.0
        let atomScale: Float = RenderMode.ballAndStick.atomRadiusScale

        var atomInstances: [AtomInstance] = atoms.map { atom in
            var color = atom.element.color
            color.w = ghostAlpha
            return AtomInstance(
                position: atom.position,
                radius: atom.element.vdwRadius * atomScale,
                color: color,
                atomIndex: Int32(atom.id),
                flags: 0,
                _pad0: 0, _pad1: 0
            )
        }

        ghostAtomCount = atomInstances.count
        let atomSize = ghostAtomCount * MemoryLayout<AtomInstance>.stride
        ghostAtomBuffer = device.makeBuffer(bytes: &atomInstances, length: atomSize, options: .storageModeShared)
        ghostAtomBuffer?.label = "GhostAtomInstances"

        let bondScale = RenderMode.ballAndStick.bondRadiusScale
        var bondInstances: [BondInstance] = []
        bondInstances.reserveCapacity(bonds.count * 2)

        for bond in bonds {
            guard bond.atomIndex1 < atoms.count, bond.atomIndex2 < atoms.count else { continue }
            let a1 = atoms[bond.atomIndex1]
            let a2 = atoms[bond.atomIndex2]
            var c1 = a1.element.color; c1.w = ghostAlpha
            var c2 = a2.element.color; c2.w = ghostAlpha
            let r = bond.order.displayRadius * bondScale

            switch bond.order {
            case .single, .aromatic:
                bondInstances.append(BondInstance(
                    positionA: a1.position, radiusA: r,
                    positionB: a2.position, radiusB: r,
                    colorA: c1, colorB: c2
                ))
            case .double:
                let bondVec = a2.position - a1.position
                let bondLen = simd_length(bondVec)
                guard bondLen > 0.01 else {
                    bondInstances.append(BondInstance(
                        positionA: a1.position, radiusA: r,
                        positionB: a2.position, radiusB: r,
                        colorA: c1, colorB: c2
                    ))
                    continue
                }
                let bondDir = bondVec / bondLen
                let ref: SIMD3<Float> = abs(bondDir.y) < 0.99
                    ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
                let perp = simd_normalize(simd_cross(bondDir, ref))
                let sep: Float = r * 1.4
                for sign: Float in [-1.0, 1.0] {
                    let off = perp * (sign * sep * 0.5)
                    bondInstances.append(BondInstance(
                        positionA: a1.position + off, radiusA: r,
                        positionB: a2.position + off, radiusB: r,
                        colorA: c1, colorB: c2
                    ))
                }
            case .triple:
                let bondVec = a2.position - a1.position
                let bondLen = simd_length(bondVec)
                guard bondLen > 0.01 else {
                    bondInstances.append(BondInstance(
                        positionA: a1.position, radiusA: r,
                        positionB: a2.position, radiusB: r,
                        colorA: c1, colorB: c2
                    ))
                    continue
                }
                let bondDir = bondVec / bondLen
                let ref: SIMD3<Float> = abs(bondDir.y) < 0.99
                    ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
                let perp = simd_normalize(simd_cross(bondDir, ref))
                let sep: Float = r * 1.5
                bondInstances.append(BondInstance(
                    positionA: a1.position, radiusA: r,
                    positionB: a2.position, radiusB: r,
                    colorA: c1, colorB: c2
                ))
                for sign: Float in [-1.0, 1.0] {
                    let off = perp * (sign * sep)
                    bondInstances.append(BondInstance(
                        positionA: a1.position + off, radiusA: r,
                        positionB: a2.position + off, radiusB: r,
                        colorA: c1, colorB: c2
                    ))
                }
            }
        }

        ghostBondCount = bondInstances.count
        if ghostBondCount > 0 {
            let bondSize = ghostBondCount * MemoryLayout<BondInstance>.stride
            ghostBondBuffer = device.makeBuffer(bytes: &bondInstances, length: bondSize, options: .storageModeShared)
            ghostBondBuffer?.label = "GhostBondInstances"
        } else {
            ghostBondBuffer = nil
        }
    }

    func clearGhostPose() {
        ghostAtomBuffer = nil
        ghostBondBuffer = nil
        ghostAtomCount = 0
        ghostBondCount = 0
    }

    // MARK: - Grid Box Wireframe

    func updateGridBox(center: SIMD3<Float>, halfSize: SIMD3<Float>, color: SIMD4<Float>) {
        let c = center
        let h = halfSize
        let corners = [
            c + SIMD3<Float>(-h.x, -h.y, -h.z),
            c + SIMD3<Float>( h.x, -h.y, -h.z),
            c + SIMD3<Float>( h.x,  h.y, -h.z),
            c + SIMD3<Float>(-h.x,  h.y, -h.z),
            c + SIMD3<Float>(-h.x, -h.y,  h.z),
            c + SIMD3<Float>( h.x, -h.y,  h.z),
            c + SIMD3<Float>( h.x,  h.y,  h.z),
            c + SIMD3<Float>(-h.x,  h.y,  h.z)
        ]
        let edges: [(Int, Int)] = [
            (0,1),(1,2),(2,3),(3,0),  // bottom face
            (4,5),(5,6),(6,7),(7,4),  // top face
            (0,4),(1,5),(2,6),(3,7)   // vertical edges
        ]

        var vertices: [GridBoxVertex] = []
        vertices.reserveCapacity(24)
        for (a, b) in edges {
            vertices.append(GridBoxVertex(position: corners[a], color: color))
            vertices.append(GridBoxVertex(position: corners[b], color: color))
        }

        gridBoxVertexCount = vertices.count
        let size = gridBoxVertexCount * MemoryLayout<GridBoxVertex>.stride
        gridBoxVertexBuffer = device.makeBuffer(bytes: vertices, length: size, options: .storageModeShared)
        gridBoxVertexBuffer?.label = "GridBoxVertices"
    }

    func clearGridBox() {
        gridBoxVertexBuffer = nil
        gridBoxVertexCount = 0
    }

    // MARK: - Molecular Surface

    func updateSurfaceMesh(_ result: SurfaceResult) {
        surfaceVertexBuffer = result.vertexBuffer
        surfaceIndexBuffer = result.indexBuffer
        surfaceIndexCount = result.indexCount
    }

    func clearSurfaceMesh() {
        surfaceVertexBuffer = nil
        surfaceIndexBuffer = nil
        surfaceIndexCount = 0
    }

    // MARK: - Draw

    func draw(in view: MTKView) {
        // Wait for a frame slot (triple-buffered)
        frameSemaphore.wait()

        camera.update()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let descriptor = view.currentRenderPassDescriptor
        else {
            frameSemaphore.signal()
            return
        }

        // Signal semaphore when GPU finishes this frame
        commandBuffer.addCompletedHandler { [semaphore = frameSemaphore] _ in
            semaphore.signal()
        }

        // Rotate uniform buffer index
        currentUniformBufferIndex = (currentUniformBufferIndex + 1) % Self.maxFramesInFlight
        let uniformBuffer = uniformBuffers[currentUniformBufferIndex]

        // MSAA: storeAndMultisampleResolve
        descriptor.colorAttachments[0].storeAction = view.sampleCount > 1 ? .storeAndMultisampleResolve : .store
        descriptor.colorAttachments[0].loadAction = .clear
        let clearColor: MTLClearColor
        if backgroundOpacity <= 0.01 {
            clearColor = themeMode == 1
                ? MTLClearColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
                : MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        } else {
            clearColor = themeMode == 1
                ? MTLClearColor(red: 0.96, green: 0.96, blue: 0.98, alpha: 1.0)
                : MTLClearColor(red: 0.08, green: 0.09, blue: 0.12, alpha: 1.0)
        }
        descriptor.colorAttachments[0].clearColor = clearColor

        // Update uniforms into the current ring buffer slot
        updateUniforms(into: uniformBuffer, viewportSize: SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height)))

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        // 1. Background gradient
        encoder.setRenderPipelineState(backgroundPipeline)
        encoder.setDepthStencilState(depthStateDisabled)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // 2. Atoms (impostor spheres)
        if atomInstanceCount > 0, let atomBuf = atomInstanceBuffer {
            encoder.setRenderPipelineState(atomPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(atomBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: atomInstanceCount)
        }

        // 3. Bonds (impostor cylinders)
        if bondInstanceCount > 0, let bondBuf = bondInstanceBuffer {
            encoder.setRenderPipelineState(bondPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(bondBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 8, instanceCount: bondInstanceCount)
        }

        // 3b. Live docking ligand — rendered as standard opaque ball-and-stick
        //     BEFORE ribbon/surface so it participates in depth buffer normally
        //     and translucent surfaces blend on top of it correctly.
        if ghostAtomCount > 0, let ghostAtomBuf = ghostAtomBuffer {
            encoder.setRenderPipelineState(atomPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(ghostAtomBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: ghostAtomCount)
        }
        if ghostBondCount > 0, let ghostBondBuf = ghostBondBuffer {
            encoder.setRenderPipelineState(bondPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(ghostBondBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 8, instanceCount: ghostBondCount)
        }

        // 4. Ribbon (triangle mesh)
        if ribbonIndexCount > 0, let vertBuf = ribbonVertexBuffer, let idxBuf = ribbonIndexBuffer {
            encoder.setRenderPipelineState(ribbonPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setCullMode(.none) // Two-sided rendering
            encoder.setVertexBuffer(vertBuf, offset: 0, index: Int(BufferIndexVertices.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: ribbonIndexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
            encoder.setCullMode(.back) // Restore
        }

        // 5. Interaction lines (dashed billboard quads, between ligand and protein atoms)
        // Hide when ligand is hidden or ghost pose is active (during docking, ghost pose has its own update path)
        if interactionLineCount > 0, let lineBuf = interactionLineBuffer, ligandVisible {
            encoder.setRenderPipelineState(interactionLinePipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(lineBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: interactionLineCount)
        }

        // 5b. Constraint indicators (same pipeline as interaction lines)
        if constraintIndicatorCount > 0, let cstBuf = constraintIndicatorBuffer {
            encoder.setRenderPipelineState(interactionLinePipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(cstBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: constraintIndicatorCount)
        }

        // 6. Grid box wireframe (docking pocket visualization — thick quads)
        if gridBoxVertexCount > 0, let boxBuf = gridBoxVertexBuffer {
            encoder.setRenderPipelineState(gridBoxPipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setVertexBuffer(boxBuf, offset: 0, index: Int(BufferIndexVertices.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            // 6 vertices per edge (2 triangles), gridBoxVertexCount/2 edges
            let edgeCount = gridBoxVertexCount / 2
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: edgeCount * 6)
        }

        // 7. Molecular surface (triangle mesh from marching cubes)
        if surfaceIndexCount > 0, let vertBuf = surfaceVertexBuffer, let idxBuf = surfaceIndexBuffer {
            encoder.setRenderPipelineState(surfacePipeline)
            encoder.setDepthStencilState(depthStateReadWrite)
            encoder.setCullMode(.none) // Two-sided surface
            encoder.setVertexBuffer(vertBuf, offset: 0, index: Int(BufferIndexVertices.rawValue))
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: surfaceIndexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
            encoder.setCullMode(.back) // Restore
        }

        encoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }

    private func updateUniforms(into buffer: MTLBuffer, viewportSize: SIMD2<Float>) {
        camera.aspectRatio = viewportSize.x / viewportSize.y

        var uniforms = Uniforms()
        uniforms.modelMatrix = Mat4.identity
        uniforms.viewMatrix = camera.viewMatrix
        uniforms.projectionMatrix = camera.projectionMatrix
        uniforms.normalMatrix = Mat4.normalMatrix(camera.viewMatrix)
        uniforms.cameraPosition = camera.eyePosition
        uniforms.lightDirection = simd_normalize(SIMD3<Float>(0.5, 1.0, 0.8))
        uniforms.lightColor = SIMD3<Float>(1.0, 0.98, 0.95)
        uniforms.ambientIntensity = 0.18
        uniforms.time = Float(CFAbsoluteTimeGetCurrent() - startTime)
        uniforms.selectedAtomIndex = Int32(selectedAtomIndex)
        uniforms.atomRadiusScale = renderMode.atomRadiusScale
        uniforms.bondRadiusScale = renderMode.bondRadiusScale
        uniforms.lightingMode = lightingMode
        if enableClipping {
            if let center = slabCenter {
                // Project pocket center into view space to get its Z depth
                let viewPos = camera.viewMatrix * SIMD4<Float>(center.x, center.y, center.z, 1.0)
                let centerZ = -viewPos.z  // camera looks down -Z, so negate
                let offsetCenterZ = centerZ + slabOffset
                uniforms.clipNearZ = max(offsetCenterZ - slabHalfThickness, 0.1)
                uniforms.clipFarZ = offsetCenterZ + slabHalfThickness
            } else {
                uniforms.clipNearZ = clipNearZ
                uniforms.clipFarZ = clipFarZ
            }
            uniforms.enableClipping = 1
        } else {
            uniforms.enableClipping = 0
        }
        uniforms.themeMode = themeMode
        uniforms.backgroundOpacity = backgroundOpacity
        uniforms.surfaceOpacity = surfaceOpacity
        uniforms.gridLineWidth = gridLineWidth

        buffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<Uniforms>.stride)
    }

    // MARK: - Atom Picking (GPU object-ID buffer)

    /// Pick an atom by rendering atom IDs to an off-screen R32Uint texture and
    /// reading back the pixel at the click location. O(1) on the CPU, pixel-perfect.
    func pickAtom(at screenPoint: SIMD2<Float>, viewportSize: SIMD2<Float>) -> Int? {
        guard atomInstanceCount > 0, let atomBuf = atomInstanceBuffer else { return nil }

        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)
        guard width > 0, height > 0 else { return nil }

        // Recreate pick textures if viewport size changed
        if width != pickTextureWidth || height != pickTextureHeight {
            rebuildPickTextures(width: width, height: height)
        }
        guard let pickTex = pickTexture, let pickDepthTex = pickDepthTexture else { return nil }

        // Render atom IDs into the pick texture (non-MSAA, single pass, on-demand)
        let passDesc = MTLRenderPassDescriptor()
        passDesc.colorAttachments[0].texture = pickTex
        passDesc.colorAttachments[0].loadAction = .clear
        passDesc.colorAttachments[0].storeAction = .store
        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: Double(0xFFFFFFFF), green: 0, blue: 0, alpha: 0)
        passDesc.depthAttachment.texture = pickDepthTex
        passDesc.depthAttachment.loadAction = .clear
        passDesc.depthAttachment.storeAction = .dontCare
        passDesc.depthAttachment.clearDepth = 1.0

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeRenderCommandEncoder(descriptor: passDesc)
        else { return nil }

        encoder.setRenderPipelineState(pickPipeline)
        encoder.setDepthStencilState(depthStateReadWrite)
        let pickUniformBuffer = uniformBuffers[currentUniformBufferIndex]
        encoder.setVertexBuffer(atomBuf, offset: 0, index: Int(BufferIndexInstances.rawValue))
        encoder.setVertexBuffer(pickUniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
        encoder.setFragmentBuffer(pickUniformBuffer, offset: 0, index: Int(BufferIndexUniforms.rawValue))
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: atomInstanceCount)
        encoder.endEncoding()

        // Synchronize for CPU readback (macOS managed storage)
        if let blit = cmdBuf.makeBlitCommandEncoder() {
            blit.synchronize(resource: pickTex)
            blit.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back the single pixel at the click location
        let px = Int(screenPoint.x)
        let py = Int(screenPoint.y)
        guard px >= 0, px < width, py >= 0, py < height else { return nil }

        var pixelValue: UInt32 = 0xFFFFFFFF
        pickTex.getBytes(&pixelValue,
                         bytesPerRow: width * MemoryLayout<UInt32>.stride,
                         from: MTLRegion(origin: MTLOrigin(x: px, y: py, z: 0),
                                         size: MTLSize(width: 1, height: 1, depth: 1)),
                         mipmapLevel: 0)

        // 0xFFFFFFFF = background (no hit)
        guard pixelValue != 0xFFFFFFFF else { return nil }
        return Int(pixelValue)
    }

    /// Create or resize the off-screen pick textures.
    private func rebuildPickTextures(width: Int, height: Int) {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Uint, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .managed
        pickTexture = device.makeTexture(descriptor: colorDesc)
        pickTexture?.label = "PickBuffer"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDesc.usage = .renderTarget
        depthDesc.storageMode = .private
        pickDepthTexture = device.makeTexture(descriptor: depthDesc)
        pickDepthTexture?.label = "PickDepth"

        pickTextureWidth = width
        pickTextureHeight = height
    }

    // MARK: - Ribbon Residue Picking (via CA control points)

    /// Pick a residue in ribbon mode by testing against CA atom positions.
    /// Returns the original protein atom ID of the nearest CA atom, or nil.
    func pickRibbonResidue(at screenPoint: SIMD2<Float>, viewportSize: SIMD2<Float>) -> Int? {
        guard !ribbonCAControlPoints.isEmpty else { return nil }
        let (rayOrigin, rayDir) = camera.screenToWorldRay(screenPoint: screenPoint, viewportSize: viewportSize)

        var closestT: Float = .infinity
        var closestAtomID: Int? = nil
        // Use a generous hit radius for CA picking (roughly ribbon width)
        let hitRadius: Float = 2.0

        for cp in ribbonCAControlPoints {
            if let t = raySphereIntersect(rayOrigin: rayOrigin, rayDir: rayDir,
                                           sphereCenter: cp.position, sphereRadius: hitRadius) {
                if t < closestT {
                    closestT = t
                    closestAtomID = cp.atomID
                }
            }
        }
        return closestAtomID
    }

    // MARK: - World to Screen Projection

    /// Project a world-space position to screen-space pixel coordinates.
    /// Returns nil if the point is behind the camera.
    func worldToScreen(_ worldPos: SIMD3<Float>, viewportSize: SIMD2<Float>) -> SIMD2<Float>? {
        let viewProj = camera.projectionMatrix * camera.viewMatrix
        let clip = viewProj * SIMD4<Float>(worldPos.x, worldPos.y, worldPos.z, 1.0)
        guard clip.w > 0 else { return nil }  // behind camera
        let ndc = SIMD3<Float>(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
        // NDC [-1,1] to screen pixels
        let sx = (ndc.x * 0.5 + 0.5) * viewportSize.x
        let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * viewportSize.y  // flip Y
        return SIMD2<Float>(sx, sy)
    }

    /// Find all atom IDs whose screen projection falls within a given rectangle.
    /// `rectMin` and `rectMax` are in screen-space pixel coordinates.
    func atomsInRect(rectMin: SIMD2<Float>, rectMax: SIMD2<Float>, viewportSize: SIMD2<Float>) -> [Int] {
        var result: [Int] = []
        for atom in currentAtoms {
            guard let sp = worldToScreen(atom.position, viewportSize: viewportSize) else { continue }
            if sp.x >= rectMin.x && sp.x <= rectMax.x &&
               sp.y >= rectMin.y && sp.y <= rectMax.y {
                result.append(atom.id)
            }
        }
        return result
    }

    // MARK: - Fit Camera

    func fitToContent() {
        // Prefer allMoleculePositions (includes protein atoms even in ribbon mode)
        let positions = !allMoleculePositions.isEmpty
            ? allMoleculePositions
            : currentAtoms.map(\.position)
        guard !positions.isEmpty else { return }
        let c = centroid(positions)
        let r = boundingRadius(positions: positions, center: c)
        camera.fitToSphere(center: c, radius: max(r, 2.0))
    }

    /// Fit camera to a specific set of world-space positions (e.g., a selection).
    func fitToPositions(_ positions: [SIMD3<Float>]) {
        guard !positions.isEmpty else { return }
        let c = centroid(positions)
        let r = boundingRadius(positions: positions, center: c)
        camera.fitToSphere(center: c, radius: max(r, 2.0))
    }

    /// Focus camera on a pocket and set Z-clipping slab to frame it.
    /// Stores pocket center in object space so slab stays correct when zooming.
    func focusOnPocket(center: SIMD3<Float>, halfExtent: SIMD3<Float>) {
        let pocketRadius = max(halfExtent.x, max(halfExtent.y, halfExtent.z))
        camera.fitToSphere(center: center, radius: max(pocketRadius + 4.0, 6.0))

        // Store object-space pocket center for zoom-invariant slab clipping
        slabCenter = center
        slabHalfThickness = pocketRadius + 6.0
        slabOffset = 0.0

        // Also update legacy clip values for backward compat / fallback
        let camDist = camera.distance
        clipNearZ = max(camDist - slabHalfThickness, 0.5)
        clipFarZ = camDist + slabHalfThickness
    }
}

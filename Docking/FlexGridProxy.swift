import Foundation
@preconcurrency import MetalKit

/// Injects repulsive grid proxies at the reference positions of excluded flexible
/// sidechain atoms. This prevents the ligand from being attracted to the "holes"
/// left in the affinity maps when sidechain atoms are removed for flexible docking.
///
/// Call `injectRepulsion()` once after grid map computation and before docking starts.
final class FlexGridProxy {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var repulsionPipeline: MTLComputePipelineState?
    private var repulsionLegacyPipeline: MTLComputePipelineState?

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue

        guard let library = device.makeDefaultLibrary() else { return }
        do {
            if let f = library.makeFunction(name: "injectFlexRepulsion") {
                repulsionPipeline = try device.makeComputePipelineState(function: f)
            }
            if let f = library.makeFunction(name: "injectFlexRepulsionLegacy") {
                repulsionLegacyPipeline = try device.makeComputePipelineState(function: f)
            }
        } catch {
            print("FlexGridProxy: failed to create pipeline: \(error)")
        }
    }

    /// Inject repulsive proxies into the typed affinity maps and the legacy steric grid.
    func injectRepulsion(
        affinityGridBuffer: MTLBuffer,
        stericGridBuffer: MTLBuffer,
        flexAtomBuffer: MTLBuffer,
        gridParamsBuffer: MTLBuffer,
        flexParamsBuffer: MTLBuffer,
        totalPoints: Int,
        numAffinityTypes: Int
    ) {
        // Typed affinity maps: one repulsion pass across all (type, point) pairs
        if let pipeline = repulsionPipeline, numAffinityTypes > 0 {
            let totalEntries = totalPoints * numAffinityTypes
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(affinityGridBuffer, offset: 0, index: 0)
            enc.setBuffer(flexAtomBuffer, offset: 0, index: 1)
            enc.setBuffer(gridParamsBuffer, offset: 0, index: 2)
            enc.setBuffer(flexParamsBuffer, offset: 0, index: 3)

            let tgs = MTLSize(width: min(128, totalEntries), height: 1, depth: 1)
            let tgc = MTLSize(width: (totalEntries + tgs.width - 1) / tgs.width, height: 1, depth: 1)
            enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Legacy steric grid: single-map pass
        if let pipeline = repulsionLegacyPipeline {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(stericGridBuffer, offset: 0, index: 0)
            enc.setBuffer(flexAtomBuffer, offset: 0, index: 1)
            enc.setBuffer(gridParamsBuffer, offset: 0, index: 2)
            enc.setBuffer(flexParamsBuffer, offset: 0, index: 3)

            let tgs = MTLSize(width: min(128, totalPoints), height: 1, depth: 1)
            let tgc = MTLSize(width: (totalPoints + tgs.width - 1) / tgs.width, height: 1, depth: 1)
            enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// FASPREnergyCalculator.swift — FASPR energy computation orchestrator
//
// Computes self-energy (rotamer vs backbone + fixed sidechains) and pair-energy
// (rotamer vs rotamer) for sidechain packing, following FASPR's energy model:
//   - CHARMM19 VDW with the H-bond / S-S equilibrium-distance override
//   - sidechain–backbone and sidechain–sidechain H-bond and S-S bond terms
//   - rotamer preference from the Dunbrack probability
//   - self-energy pruning (SEC_CUT) before preference/fixed-sidechain terms
//
// VDW pairs whose equilibrium distance can never be overridden (at least one
// atom is non-polar) are evaluated on the GPU via FASPRMetalAccelerator; all
// pairs involving two override-capable atoms, and all polar terms, run on CPU.
//
// Translated from FASPR (Huang 2020, MIT License):
//   temp/FASPR/src/SelfEnergy.cpp
//   temp/FASPR/src/PairEnergy.cpp
// ============================================================================

import Foundation
import Metal
import simd

// MARK: - Site Representation

/// A packing site: one residue that needs sidechain optimization.
struct FASPRSite {
    let siteIndex: Int              // index in the global site array
    let residueType: Character      // one-letter code
    let backbone: [SIMD3<Float>]    // [N, CA, C, O, CB] positions (CB absent for Gly / missing)
    let atomParams: [(radius: Float, depth: Float)]  // VDW params per atom
    let atomTypeIndices: [Int]      // 1-based FASPR atom type per atom
    let isVariable: Bool            // rotamers come from the library (site is being repacked)
    let rotamerProbs: [Float]       // probability per rotamer
    let maxProb: Float              // max probability over the library rotamers
    let rotamerCoords: [[SIMD3<Float>]]  // [rot][atomIdx] sidechain atoms beyond CB, topology order

    var nrots: Int { rotamerCoords.count }

    /// Copy of this site restricted to the given rotamer indices.
    func keepingRotamers(_ keep: [Int]) -> FASPRSite {
        FASPRSite(
            siteIndex: siteIndex, residueType: residueType, backbone: backbone,
            atomParams: atomParams, atomTypeIndices: atomTypeIndices, isVariable: isVariable,
            rotamerProbs: keep.map { rotamerProbs[$0] }, maxProb: maxProb,
            rotamerCoords: keep.map { rotamerCoords[$0] })
    }

    /// Residue types with sidechain H-bond or S-S terms in FASPR.
    private static let polarTypes: Set<Character> = ["C", "D", "E", "H", "K", "N", "Q", "R", "S", "T", "W", "Y"]

    /// Whether rotamer `r` takes part in polar terms: polar residue type with a
    /// backbone including CB and a complete sidechain (the terms index both).
    func polarReady(rotamer r: Int) -> Bool {
        Self.polarTypes.contains(residueType)
            && backbone.count >= 5
            && rotamerCoords[r].count >= (FASPRVDWParameters.sidechainAtomCount[residueType] ?? Int.max)
    }
}

// MARK: - Energy Calculator

final class FASPREnergyCalculator {

    /// Self-energy window above the per-site minimum kept before search (FASPR SEC_CUT).
    private static let selfEnergyCut: Float = 15.0

    let accelerator: FASPRMetalAccelerator?

    init(device: MTLDevice?) {
        if let device = device {
            accelerator = FASPRMetalAccelerator(device: device)
        } else {
            accelerator = nil
        }
    }

    // MARK: - Per-site atom cache

    /// Flattened sidechain atoms of all rotamers of a site, split by whether the
    /// atom type can take part in a VDW equilibrium-distance override.
    private struct SiteAtoms {
        /// Non-polar sidechain atoms of every rotamer (GPU input) and their offsets.
        let nonpolarGPU: [FASPRGPUAtom]
        let nonpolarOffsets: [UInt32]
        /// All sidechain atoms of every rotamer (GPU input) and their offsets.
        let allGPU: [FASPRGPUAtom]
        let allOffsets: [UInt32]
        /// Per-rotamer `FASPRSite.polarReady`.
        let polarReady: [Bool]

        init(site: FASPRSite) {
            polarReady = (0..<site.nrots).map { site.polarReady(rotamer: $0) }
            var nonpolar: [FASPRGPUAtom] = []
            var nonpolarOff: [UInt32] = [0]
            var all: [FASPRGPUAtom] = []
            var allOff: [UInt32] = [0]
            for coords in site.rotamerCoords {
                for k in 0..<coords.count where k + 5 < site.atomParams.count {
                    let atom = FASPRGPUAtom(
                        position: coords[k],
                        radius: site.atomParams[k + 5].radius,
                        depth: site.atomParams[k + 5].depth,
                        atomTypeIdx: UInt32(site.atomTypeIndices[k + 5]),
                        _pad0: 0)
                    all.append(atom)
                    if !FASPRVDWParameters.hasDistanceOverride(site.atomTypeIndices[k + 5]) {
                        nonpolar.append(atom)
                    }
                }
                allOff.append(UInt32(all.count))
                nonpolarOff.append(UInt32(nonpolar.count))
            }
            nonpolarGPU = nonpolar
            nonpolarOffsets = nonpolarOff
            allGPU = all
            allOffsets = allOff
        }
    }

    // MARK: - Self-Energy (backbone part)

    /// Sidechain vs. own backbone, neighbour backbones, and sidechain–backbone H-bonds.
    /// Translated from SelfEnergy::EnergySidechainAndBackbone().
    private func sidechainBackboneEnergy(
        site: FASPRSite,
        cache: SiteAtoms,
        allSites: [FASPRSite],
        neighbors: [Int]
    ) -> [Float] {
        var energies = [Float](repeating: 0, count: site.nrots)
        let types = site.atomTypeIndices
        let params = site.atomParams

        // Residue-internal energy: own N, CA, C, O vs. sidechain atoms beyond CG
        // (CG is 1-3 to CA and is excluded, as in FASPR: j < 4, k >= 6).
        let ownBB = min(4, site.backbone.count)
        for r in 0..<site.nrots {
            let sc = site.rotamerCoords[r]
            var e: Float = 0
            for k in 1..<max(1, sc.count) where k + 5 < params.count {
                for j in 0..<ownBB {
                    e += FASPRVDWParameters.pairVDW(
                        sc[k], types[k + 5], params[k + 5].radius, params[k + 5].depth,
                        site.backbone[j], types[j], params[j].radius, params[j].depth)
                }
            }
            energies[r] = e
        }

        // Neighbour backbone atoms (N, CA, C, O and CB for non-Gly).
        var env: [FASPRGPUAtom] = []
        for nb in neighbors {
            let neighbor = allSites[nb]
            let n = min(neighbor.residueType == "G" ? 4 : 5, neighbor.backbone.count, neighbor.atomParams.count)
            for j in 0..<n {
                env.append(FASPRGPUAtom(
                    position: neighbor.backbone[j],
                    radius: neighbor.atomParams[j].radius,
                    depth: neighbor.atomParams[j].depth,
                    atomTypeIdx: UInt32(neighbor.atomTypeIndices[j]),
                    _pad0: 0))
            }
        }

        if !env.isEmpty {
            // Non-polar sidechain atoms never trigger the distance override: GPU-eligible.
            var gpuDone = false
            if let accel = accelerator, site.nrots >= 4, !cache.nonpolarGPU.isEmpty,
               let vdw = accel.computeSelfVDW(
                   rotamerAtoms: cache.nonpolarGPU, backboneAtoms: env,
                   rotamerOffsets: cache.nonpolarOffsets, rotamerCount: site.nrots) {
                for r in 0..<site.nrots { energies[r] += vdw[r] }
                gpuDone = true
            }

            for r in 0..<site.nrots {
                let start = Int(cache.allOffsets[r])
                let end = Int(cache.allOffsets[r + 1])
                var e: Float = 0
                for s in start..<end {
                    let a = cache.allGPU[s]
                    let typeA = Int(a.atomTypeIdx)
                    if gpuDone && !FASPRVDWParameters.hasDistanceOverride(typeA) { continue }
                    for b in env {
                        e += FASPRVDWParameters.pairVDW(
                            a.position, typeA, a.radius, a.depth,
                            b.position, Int(b.atomTypeIdx), b.radius, b.depth)
                    }
                }
                energies[r] += e
            }
        }

        // Sidechain–backbone H-bonds (skipped for sequence neighbours).
        for nb in neighbors {
            let neighbor = allSites[nb]
            guard abs(site.siteIndex - neighbor.siteIndex) > 1, neighbor.backbone.count >= 4 else { continue }
            for r in 0..<site.nrots where cache.polarReady[r] {
                energies[r] += FASPRVDWParameters.polarSidechainBackbone(
                    t1: site.residueType, b1: site.backbone,
                    s1: site.rotamerCoords[r], b2: neighbor.backbone)
            }
        }

        return energies
    }

    // MARK: - Fixed-sidechain contribution

    /// Rotamer sidechain vs. sidechains of fixed (single-rotamer) neighbours.
    /// Translated from SelfEnergy::EnergyRotamerSidechainAndFixedSidechain().
    private func fixedSidechainEnergy(
        site: FASPRSite,
        fixed: FASPRSite,
        energies: inout [Float]
    ) {
        let fixedSC = fixed.rotamerCoords[0]
        let fixedPolar = fixed.polarReady(rotamer: 0)
        for r in 0..<site.nrots {
            let sc = site.rotamerCoords[r]
            var e: Float = 0
            for k in 0..<sc.count where k + 5 < site.atomParams.count {
                for l in 0..<fixedSC.count where l + 5 < fixed.atomParams.count {
                    e += FASPRVDWParameters.pairVDW(
                        sc[k], site.atomTypeIndices[k + 5],
                        site.atomParams[k + 5].radius, site.atomParams[k + 5].depth,
                        fixedSC[l], fixed.atomTypeIndices[l + 5],
                        fixed.atomParams[l + 5].radius, fixed.atomParams[l + 5].depth)
                }
            }
            if fixedPolar && site.polarReady(rotamer: r) {
                e += FASPRVDWParameters.polarSidechainSidechain(
                    t1: site.residueType, b1: site.backbone, s1: sc,
                    t2: fixed.residueType, b2: fixed.backbone, s2: fixedSC)
            }
            energies[r] += e
        }
    }

    // MARK: - Pair-Energy

    /// Compute pair-energy between rotamers of two sites (VDW + polar terms).
    /// Returns nil when every entry is zero (no interaction), as FASPR does.
    /// Translated from PairEnergy::EnergyRotamerSidechainAndRotamerSidechain().
    private func computePairEnergy(
        site1: FASPRSite, cache1: SiteAtoms,
        site2: FASPRSite, cache2: SiteAtoms
    ) -> [[Float]]? {
        let n1 = site1.nrots
        let n2 = site2.nrots
        var result = [[Float]](repeating: [Float](repeating: 0, count: n2), count: n1)

        // Pairs with a non-polar atom on site 1 are override-free: GPU-eligible.
        var gpuDone = false
        if let accel = accelerator, n1 * n2 >= 8, !cache1.nonpolarGPU.isEmpty, !cache2.allGPU.isEmpty,
           let flat = accel.computePairVDW(
               site1Atoms: cache1.nonpolarGPU, site2Atoms: cache2.allGPU,
               offsets1: cache1.nonpolarOffsets, offsets2: cache2.allOffsets,
               rot1Count: n1, rot2Count: n2) {
            for r1 in 0..<n1 {
                for r2 in 0..<n2 { result[r1][r2] = flat[r1 * n2 + r2] }
            }
            gpuDone = true
        }

        for r1 in 0..<n1 {
            let s1Start = Int(cache1.allOffsets[r1])
            let s1End = Int(cache1.allOffsets[r1 + 1])
            for r2 in 0..<n2 {
                let s2Start = Int(cache2.allOffsets[r2])
                let s2End = Int(cache2.allOffsets[r2 + 1])
                var e: Float = 0
                for i in s1Start..<s1End {
                    let a = cache1.allGPU[i]
                    let typeA = Int(a.atomTypeIdx)
                    if gpuDone && !FASPRVDWParameters.hasDistanceOverride(typeA) { continue }
                    for j in s2Start..<s2End {
                        let b = cache2.allGPU[j]
                        e += FASPRVDWParameters.pairVDW(
                            a.position, typeA, a.radius, a.depth,
                            b.position, Int(b.atomTypeIdx), b.radius, b.depth)
                    }
                }
                if cache1.polarReady[r1] && cache2.polarReady[r2] {
                    e += FASPRVDWParameters.polarSidechainSidechain(
                        t1: site1.residueType, b1: site1.backbone, s1: site1.rotamerCoords[r1],
                        t2: site2.residueType, b2: site2.backbone, s2: site2.rotamerCoords[r2])
                }
                result[r1][r2] += e
            }
        }

        return result.contains(where: { row in row.contains { $0 != 0 } }) ? result : nil
    }

    // MARK: - Full Pipeline

    /// Compute all self-energies and pair-energies for the packing problem.
    ///
    /// Follows SelfEnergy::CalcSelfEnergy(): rotamers whose sidechain–backbone
    /// energy exceeds the site minimum by more than SEC_CUT are removed from
    /// `sites` (a site left with one rotamer becomes fixed), then rotamer
    /// preference and fixed-sidechain interactions are added.
    func computeAllEnergies(
        sites: inout [FASPRSite],
        conMap: [[Int]]
    ) -> (selfEnergies: [[Float]], pairEnergies: [Int: [Int: [[Float]]]]) {
        let nSites = sites.count
        var selfEnergies = [[Float]](repeating: [], count: nSites)

        // 1. Sidechain vs. backbone energy, then SEC_CUT pruning.
        for i in 0..<nSites where sites[i].nrots >= 2 {
            let e = sidechainBackboneEnergy(
                site: sites[i], cache: SiteAtoms(site: sites[i]),
                allSites: sites, neighbors: conMap[i])
            let emin = e.min() ?? 0
            let keep = e.indices.filter { e[$0] <= emin + Self.selfEnergyCut }
            if keep.count < e.count {
                sites[i] = sites[i].keepingRotamers(keep)
                selfEnergies[i] = keep.map { e[$0] }
            } else {
                selfEnergies[i] = e
            }
        }

        // 2. Rotamer preference + interactions with fixed sidechains.
        for i in 0..<nSites where sites[i].nrots >= 2 {
            let site = sites[i]
            for r in 0..<site.nrots {
                selfEnergies[i][r] += FASPRVDWParameters.rotamerPreferenceEnergy(
                    probability: site.rotamerProbs[r],
                    maxProbability: site.maxProb,
                    residueType: site.residueType)
            }
            for j in conMap[i] where sites[j].nrots == 1 {
                fixedSidechainEnergy(site: site, fixed: sites[j], energies: &selfEnergies[i])
            }
        }

        // 3. Pair energies between variable neighbours.
        let caches: [SiteAtoms?] = sites.map { $0.nrots >= 2 ? SiteAtoms(site: $0) : nil }
        var pairEnergies: [Int: [Int: [[Float]]]] = [:]
        for i in 0..<nSites {
            guard let cacheI = caches[i] else { continue }
            for j in conMap[i] {
                guard j > i, let cacheJ = caches[j],
                      let pairE = computePairEnergy(site1: sites[i], cache1: cacheI,
                                                    site2: sites[j], cache2: cacheJ)
                else { continue }
                pairEnergies[i, default: [:]][j] = pairE
                var transposed = [[Float]](repeating: [Float](repeating: 0, count: sites[i].nrots),
                                           count: sites[j].nrots)
                for r1 in 0..<sites[i].nrots {
                    for r2 in 0..<sites[j].nrots {
                        transposed[r2][r1] = pairE[r1][r2]
                    }
                }
                pairEnergies[j, default: [:]][i] = transposed
            }
        }

        return (selfEnergies, pairEnergies)
    }
}

// MARK: - GPU Atom Type (bridge to Metal)

/// FASPRGPUAtom layout matching ShaderTypes.h.
/// This struct must be exactly 32 bytes to match the Metal shader layout.
struct FASPRGPUAtom {
    var position: SIMD3<Float>
    var radius: Float
    var depth: Float
    var atomTypeIdx: UInt32
    var _pad0: UInt32
}

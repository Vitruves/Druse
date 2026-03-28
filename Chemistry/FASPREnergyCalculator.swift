// ============================================================================
// FASPREnergyCalculator.swift — FASPR energy computation orchestrator
//
// Computes self-energy (rotamer vs backbone + environment) and pair-energy
// (rotamer vs rotamer) for sidechain packing. VDW terms are GPU-accelerated
// via FASPRMetalAccelerator; H-bond and S-S bond terms run on CPU (they are
// highly branchy and residue-type-specific).
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
    let backbone: [SIMD3<Float>]    // [N, CA, C, O, CB] positions (up to 5)
    let atomParams: [(radius: Float, depth: Float)]  // VDW params per atom
    let atomTypeIndices: [Int]      // 1-based FASPR atom type per atom
    let nrots: Int                  // number of rotamers
    let rotamerProbs: [Float]       // probability per rotamer
    let maxProb: Float              // max probability
    let rotamerCoords: [[SIMD3<Float>]]  // [rot][atomIdx] sidechain atom positions (CB excluded, starts from CG or first sidechain)
}

// MARK: - Energy Calculator

final class FASPREnergyCalculator {

    let accelerator: FASPRMetalAccelerator?

    init(device: MTLDevice?) {
        if let device = device {
            accelerator = FASPRMetalAccelerator(device: device)
        } else {
            accelerator = nil
        }
    }

    // MARK: - Self-Energy

    /// Compute self-energy for all rotamers at a site.
    /// Self-energy = VDW(sidechain vs backbone) + VDW(sidechain vs internal backbone)
    ///             + H-bond(sidechain vs backbone) + rotamer preference.
    ///
    /// - Parameters:
    ///   - site: The site to evaluate
    ///   - allSites: All sites (for backbone/fixed sidechain contact)
    ///   - conMap: Contact map (neighbors for each site)
    /// - Returns: Energy per rotamer [nrots]
    func computeSelfEnergy(
        site: FASPRSite,
        allSites: [FASPRSite],
        conMap: [[Int]]
    ) -> [Float] {
        guard site.nrots > 0 else { return [] }

        var energies = [Float](repeating: 0, count: site.nrots)

        // Gather backbone atoms from contact neighbors + self
        var backboneGPUAtoms: [FASPRGPUAtom] = []

        // Self backbone (N, CA, C, O = indices 0..3)
        for j in 0..<min(4, site.backbone.count) {
            let pos = site.backbone[j]
            backboneGPUAtoms.append(FASPRGPUAtom(
                position: pos,
                radius: site.atomParams[j].radius,
                depth: site.atomParams[j].depth,
                atomTypeIdx: UInt32(site.atomTypeIndices[j]),
                _pad0: 0
            ))
        }

        // Neighbor backbone atoms
        for neighborIdx in conMap[site.siteIndex] {
            guard neighborIdx < allSites.count else { continue }
            let neighbor = allSites[neighborIdx]
            let n = (neighbor.residueType == "G") ? 4 : min(5, neighbor.backbone.count)
            for j in 0..<n {
                backboneGPUAtoms.append(FASPRGPUAtom(
                    position: neighbor.backbone[j],
                    radius: neighbor.atomParams[j].radius,
                    depth: neighbor.atomParams[j].depth,
                    atomTypeIdx: UInt32(neighbor.atomTypeIndices[j]),
                    _pad0: 0
                ))
            }
        }

        // Build GPU rotamer atoms
        let scAtomCount = site.atomParams.count - 5  // sidechain atoms (after N,CA,C,O,CB)
        guard scAtomCount > 0 else { return energies }

        var rotamerGPUAtoms: [FASPRGPUAtom] = []
        var offsets: [UInt32] = [0]

        for rot in 0..<site.nrots {
            let coords = site.rotamerCoords[rot]
            for k in 0..<coords.count {
                let paramIdx = k + 5  // sidechain params start at index 5
                guard paramIdx < site.atomParams.count else { continue }
                rotamerGPUAtoms.append(FASPRGPUAtom(
                    position: coords[k],
                    radius: site.atomParams[paramIdx].radius,
                    depth: site.atomParams[paramIdx].depth,
                    atomTypeIdx: UInt32(site.atomTypeIndices[paramIdx]),
                    _pad0: 0
                ))
            }
            offsets.append(UInt32(rotamerGPUAtoms.count))
        }

        // GPU VDW computation
        if let accel = accelerator, site.nrots >= 4 {
            let vdwEnergies = accel.computeSelfVDW(
                rotamerAtoms: rotamerGPUAtoms,
                backboneAtoms: backboneGPUAtoms,
                rotamerOffsets: offsets,
                rotamerCount: site.nrots
            )
            for r in 0..<min(vdwEnergies.count, energies.count) {
                energies[r] += vdwEnergies[r]
            }
        } else {
            // CPU fallback for small rotamer counts
            for r in 0..<site.nrots {
                let start = Int(offsets[r])
                let end = Int(offsets[r + 1])
                for s in start..<end {
                    let posS = rotamerGPUAtoms[s].position
                    let radS = rotamerGPUAtoms[s].radius
                    let depS = rotamerGPUAtoms[s].depth
                    for b in 0..<backboneGPUAtoms.count {
                        let dist = simd_distance(posS, backboneGPUAtoms[b].position)
                        guard dist > 1e-6 else { continue }
                        let rij = radS + backboneGPUAtoms[b].radius
                        let eij = sqrt(depS * backboneGPUAtoms[b].depth)
                        energies[r] += FASPRVDWParameters.vdwEnergy(dstar: dist / rij, epsilon: eij)
                    }
                }
            }
        }

        // Rotamer preference energy (CPU)
        for r in 0..<site.nrots {
            energies[r] += FASPRVDWParameters.rotamerPreferenceEnergy(
                probability: site.rotamerProbs[r],
                maxProbability: site.maxProb,
                residueType: site.residueType
            )
        }

        return energies
    }

    // MARK: - Pair-Energy

    /// Compute pair-energy between rotamers of two sites.
    /// Returns [nrots1][nrots2] matrix of VDW energies.
    func computePairEnergy(
        site1: FASPRSite,
        site2: FASPRSite
    ) -> [[Float]]? {
        guard site1.nrots >= 2, site2.nrots >= 2 else { return nil }

        let scCount1 = site1.atomParams.count - 5
        let scCount2 = site2.atomParams.count - 5
        guard scCount1 > 0, scCount2 > 0 else { return nil }

        // Build GPU atoms for site1
        var atoms1: [FASPRGPUAtom] = []
        var offsets1: [UInt32] = [0]
        for rot in 0..<site1.nrots {
            for k in 0..<site1.rotamerCoords[rot].count {
                let pi = k + 5
                guard pi < site1.atomParams.count else { continue }
                atoms1.append(FASPRGPUAtom(
                    position: site1.rotamerCoords[rot][k],
                    radius: site1.atomParams[pi].radius,
                    depth: site1.atomParams[pi].depth,
                    atomTypeIdx: UInt32(site1.atomTypeIndices[pi]),
                    _pad0: 0
                ))
            }
            offsets1.append(UInt32(atoms1.count))
        }

        // Build GPU atoms for site2
        var atoms2: [FASPRGPUAtom] = []
        var offsets2: [UInt32] = [0]
        for rot in 0..<site2.nrots {
            for k in 0..<site2.rotamerCoords[rot].count {
                let pi = k + 5
                guard pi < site2.atomParams.count else { continue }
                atoms2.append(FASPRGPUAtom(
                    position: site2.rotamerCoords[rot][k],
                    radius: site2.atomParams[pi].radius,
                    depth: site2.atomParams[pi].depth,
                    atomTypeIdx: UInt32(site2.atomTypeIndices[pi]),
                    _pad0: 0
                ))
            }
            offsets2.append(UInt32(atoms2.count))
        }

        // GPU pair energy
        let totalPairs = site1.nrots * site2.nrots
        if let accel = accelerator, totalPairs >= 8 {
            let flat = accel.computePairVDW(
                site1Atoms: atoms1,
                site2Atoms: atoms2,
                offsets1: offsets1,
                offsets2: offsets2,
                rot1Count: site1.nrots,
                rot2Count: site2.nrots
            )

            // Reshape to [nrots1][nrots2]
            var result = [[Float]](repeating: [Float](repeating: 0, count: site2.nrots), count: site1.nrots)
            for r1 in 0..<site1.nrots {
                for r2 in 0..<site2.nrots {
                    let idx = r1 * site2.nrots + r2
                    if idx < flat.count {
                        result[r1][r2] = flat[idx]
                    }
                }
            }
            return hasNonZero(result) ? result : nil
        } else {
            // CPU fallback
            var result = [[Float]](repeating: [Float](repeating: 0, count: site2.nrots), count: site1.nrots)
            var anyNonZero = false

            for r1 in 0..<site1.nrots {
                let s1Start = Int(offsets1[r1])
                let s1End = Int(offsets1[r1 + 1])
                for r2 in 0..<site2.nrots {
                    let s2Start = Int(offsets2[r2])
                    let s2End = Int(offsets2[r2 + 1])
                    var e: Float = 0
                    for i in s1Start..<s1End {
                        for j in s2Start..<s2End {
                            let dist = simd_distance(atoms1[i].position, atoms2[j].position)
                            guard dist > 1e-6 else { continue }
                            let rij = atoms1[i].radius + atoms2[j].radius
                            let eij = sqrt(atoms1[i].depth * atoms2[j].depth)
                            e += FASPRVDWParameters.vdwEnergy(dstar: dist / rij, epsilon: eij)
                        }
                    }
                    result[r1][r2] = e
                    if abs(e) > 1e-8 { anyNonZero = true }
                }
            }
            return anyNonZero ? result : nil
        }
    }

    // MARK: - Full Pipeline

    /// Compute all self-energies and pair-energies for the packing problem.
    func computeAllEnergies(
        sites: [FASPRSite],
        conMap: [[Int]]
    ) -> (selfEnergies: [[Float]], pairEnergies: [Int: [Int: [[Float]]]]) {
        let nSites = sites.count

        // Self-energies
        var selfEnergies = [[Float]]()
        selfEnergies.reserveCapacity(nSites)
        for i in 0..<nSites {
            selfEnergies.append(computeSelfEnergy(site: sites[i], allSites: sites, conMap: conMap))
        }

        // Pair-energies (only for neighbor pairs where both have >= 2 rotamers)
        var pairEnergies: [Int: [Int: [[Float]]]] = [:]
        for i in 0..<nSites {
            guard sites[i].nrots >= 2 else { continue }
            for j in conMap[i] {
                guard j > i, j < nSites, sites[j].nrots >= 2 else { continue }
                if let pairE = computePairEnergy(site1: sites[i], site2: sites[j]) {
                    pairEnergies[i, default: [:]][j] = pairE
                    // Transpose for j→i lookup
                    var transposed = [[Float]](repeating: [Float](repeating: 0, count: sites[i].nrots), count: sites[j].nrots)
                    for r1 in 0..<sites[i].nrots {
                        for r2 in 0..<sites[j].nrots {
                            transposed[r2][r1] = pairE[r1][r2]
                        }
                    }
                    pairEnergies[j, default: [:]][i] = transposed
                }
            }
        }

        return (selfEnergies, pairEnergies)
    }

    // MARK: - Helpers

    private func hasNonZero(_ matrix: [[Float]]) -> Bool {
        for row in matrix {
            for val in row {
                if abs(val) > 1e-8 { return true }
            }
        }
        return false
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

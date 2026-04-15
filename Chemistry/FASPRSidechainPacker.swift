// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// FASPRSidechainPacker.swift — FASPR sidechain packing orchestrator
//
// Main entry point for sidechain packing. Orchestrates:
//   1. Backbone extraction and phi/psi computation
//   2. Rotamer loading from Dunbrack library
//   3. Sidechain coordinate construction
//   4. Self-energy and pair-energy computation (GPU-accelerated)
//   5. DEE search + tree decomposition (CPU)
//   6. Atom replacement with optimal rotamers
//
// Translated from FASPR (Huang 2020, MIT License):
//   temp/FASPR/src/FASPR.cpp (main pipeline)
//   temp/FASPR/src/RotamerBuilder.cpp (BuildSidechain, PhiPsi)
// ============================================================================

import Foundation
import Metal
import simd

enum FASPRSidechainPacker {

    // MARK: - Report

    struct PackingReport: Sendable {
        var residuesPacked: Int = 0
        var totalRotamersEvaluated: Int = 0
        var deeRotamersEliminated: Int = 0
        var finalEnergy: Float = 0
        var elapsedMs: Double = 0
    }

    // MARK: - Main Entry Point

    /// Pack sidechains using FASPR algorithm.
    ///
    /// - Parameters:
    ///   - atoms: Input atoms (with backbone complete)
    ///   - bonds: Input bonds
    ///   - device: Metal device for GPU acceleration (nil = CPU-only)
    ///   - onlyIncomplete: If true, only repack residues with missing/incomplete sidechains
    ///   - reconstructedResidueKeys: Set of (chainID, residueSeq) for residues that were reconstructed
    /// - Returns: Updated atoms/bonds and report
    static func packSidechains(
        atoms: [Atom],
        bonds: [Bond],
        device: MTLDevice? = nil,
        onlyIncomplete: Bool = true,
        reconstructedResidueKeys: Set<String>? = nil
    ) -> (atoms: [Atom], bonds: [Bond], report: PackingReport) {
        let start = CFAbsoluteTimeGetCurrent()
        var report = PackingReport()

        let logSync = { (msg: String) in
            _ = Task { @MainActor in ActivityLog.shared.debug(msg, category: .prep) }
        }

        // Step 1: Group atoms by residue and extract backbone
        let residueGroups = groupAtomsByResidue(atoms: atoms)
        logSync("[FASPR] \(residueGroups.count) residues found")

        // Step 2: Build site list
        let (sites, residueToSiteMap) = buildSites(
            residueGroups: residueGroups,
            atoms: atoms,
            onlyIncomplete: onlyIncomplete,
            reconstructedKeys: reconstructedResidueKeys
        )

        let variableSites = sites.filter { $0.nrots >= 2 }
        if variableSites.isEmpty {
            logSync("[FASPR] No variable sites to pack")
            report.elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            return (atoms, bonds, report)
        }

        logSync("[FASPR] \(variableSites.count) variable sites, packing...")
        report.residuesPacked = variableSites.count
        report.totalRotamersEvaluated = sites.reduce(0) { $0 + $1.nrots }

        // Step 3: Build contact map
        let conMap = buildContactMap(sites: sites)

        // Step 4: Compute energies
        let calculator = FASPREnergyCalculator(device: device)
        let (selfEnergies, pairEnergies) = calculator.computeAllEnergies(sites: sites, conMap: conMap)

        // Step 5: Search for optimal rotamers
        let nrots = sites.map { $0.nrots }
        let bestrot = FASPRSearch.search(
            nrots: nrots,
            eTableSelf: selfEnergies,
            eTablePair: pairEnergies,
            conMap: conMap
        )

        // Step 6: Apply optimal rotamers to atoms
        var newAtoms = atoms
        var newBonds = bonds
        applyBestRotamers(
            bestrot: bestrot,
            sites: sites,
            residueGroups: residueGroups,
            residueToSiteMap: residueToSiteMap,
            atoms: &newAtoms,
            bonds: &newBonds
        )

        report.elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        logSync("[FASPR] Packing done in \(String(format: "%.0f", report.elapsedMs))ms")

        return (newAtoms, newBonds, report)
    }

    // MARK: - Step 1: Group Atoms by Residue

    struct ResidueGroup {
        let key: String           // "chainID_residueSeq"
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let atomIndices: [Int]    // indices into the atom array
    }

    private static func groupAtomsByResidue(atoms: [Atom]) -> [ResidueGroup] {
        var groups: [String: (chainID: String, seq: Int, name: String, indices: [Int])] = [:]
        var order: [String] = []

        for (i, atom) in atoms.enumerated() {
            guard !atom.isHetAtom, atom.element != .H else { continue }
            let key = "\(atom.chainID)_\(atom.residueSeq)"
            if groups[key] == nil {
                groups[key] = (atom.chainID, atom.residueSeq, atom.residueName, [i])
                order.append(key)
            } else {
                groups[key]!.indices.append(i)
            }
        }

        return order.compactMap { key in
            guard let g = groups[key] else { return nil }
            return ResidueGroup(key: key, chainID: g.chainID, residueSeq: g.seq,
                               residueName: g.name, atomIndices: g.indices)
        }
    }

    // MARK: - Step 2: Build Sites

    private static func buildSites(
        residueGroups: [ResidueGroup],
        atoms: [Atom],
        onlyIncomplete: Bool,
        reconstructedKeys: Set<String>?
    ) -> ([FASPRSite], [String: Int]) {
        let library = DunbrackRotamerLibrary.shared
        var sites: [FASPRSite] = []
        var residueToSiteMap: [String: Int] = [:]

        for (groupIdx, group) in residueGroups.enumerated() {
            guard let oneLetterCode = SidechainTopologyStore.three2one[group.residueName] else {
                // Non-standard residue, skip
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: "?",
                    backbone: [], atomParams: [], atomTypeIndices: [],
                    nrots: 0, rotamerProbs: [], maxProb: 0, rotamerCoords: []))
                residueToSiteMap[group.key] = groupIdx
                continue
            }

            residueToSiteMap[group.key] = groupIdx

            // Find backbone atoms
            let backboneNames = [" N  ", " CA ", " C  ", " O  ", " CB "]
            var backbonePos = [SIMD3<Float>?](repeating: nil, count: 5)
            for atomIdx in group.atomIndices {
                let name = atoms[atomIdx].name.padding(toLength: 4, withPad: " ", startingAt: 0)
                for (bi, bname) in backboneNames.enumerated() {
                    if name == bname { backbonePos[bi] = atoms[atomIdx].position }
                }
            }

            guard let n = backbonePos[0], let ca = backbonePos[1], let c = backbonePos[2] else {
                // Incomplete backbone
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: oneLetterCode,
                    backbone: [], atomParams: [], atomTypeIndices: [],
                    nrots: 0, rotamerProbs: [], maxProb: 0, rotamerCoords: []))
                continue
            }

            // Determine if this site needs packing
            let needsPacking: Bool
            if !onlyIncomplete {
                needsPacking = SidechainTopologyStore.rotamericResidues.contains(oneLetterCode)
            } else if let keys = reconstructedKeys {
                needsPacking = keys.contains(group.key) && SidechainTopologyStore.rotamericResidues.contains(oneLetterCode)
            } else {
                needsPacking = SidechainTopologyStore.rotamericResidues.contains(oneLetterCode)
            }

            // Compute phi/psi from neighbors
            var phi: Float = -60.0  // default
            var psi: Float = 60.0   // default

            if groupIdx > 0 {
                let prev = residueGroups[groupIdx - 1]
                if prev.chainID == group.chainID {
                    if let prevC = findBackboneAtom(" C  ", in: prev, atoms: atoms) {
                        let dist = simd_distance(prevC, n)
                        if dist < 2.1 {
                            phi = SidechainTopologyStore.phi(prevC: prevC, n: n, ca: ca, c: c)
                        }
                    }
                }
            }
            if groupIdx < residueGroups.count - 1 {
                let next = residueGroups[groupIdx + 1]
                if next.chainID == group.chainID {
                    if let nextN = findBackboneAtom(" N  ", in: next, atoms: atoms) {
                        let dist = simd_distance(c, nextN)
                        if dist < 2.1 {
                            psi = SidechainTopologyStore.psi(n: n, ca: ca, c: c, nextN: nextN)
                        }
                    }
                }
            }

            // Get atom VDW parameters
            let atomParams = FASPRVDWParameters.resolveAtomParams(residueType: oneLetterCode)
            let atomTypeIndices = FASPRVDWParameters.atomTypeIndices[oneLetterCode] ?? []

            // Build backbone positions array
            var bbPositions: [SIMD3<Float>] = [n, ca, c]
            if let o = backbonePos[3] { bbPositions.append(o) } else { bbPositions.append(.zero) }
            if let cb = backbonePos[4] { bbPositions.append(cb) }

            if !needsPacking || oneLetterCode == "A" || oneLetterCode == "G" {
                // Fixed site (0 or 1 rotamer)
                let nrot = (oneLetterCode == "A" || oneLetterCode == "G") ? 0 : 1
                var rotCoords: [[SIMD3<Float>]] = []
                if nrot == 1 {
                    // Use existing sidechain coordinates
                    let scCoords = extractExistingSidechain(group: group, atoms: atoms, backboneNames: backboneNames)
                    rotCoords = [scCoords]
                }
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: oneLetterCode,
                    backbone: bbPositions, atomParams: atomParams, atomTypeIndices: atomTypeIndices,
                    nrots: nrot, rotamerProbs: nrot == 1 ? [1.0] : [], maxProb: 1.0,
                    rotamerCoords: rotCoords))
                continue
            }

            // Load rotamers
            let rotamers = library.rotamers(for: oneLetterCode, phi: phi, psi: psi)
            guard !rotamers.isEmpty else {
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: oneLetterCode,
                    backbone: bbPositions, atomParams: atomParams, atomTypeIndices: atomTypeIndices,
                    nrots: 1, rotamerProbs: [1.0], maxProb: 1.0,
                    rotamerCoords: [extractExistingSidechain(group: group, atoms: atoms, backboneNames: backboneNames)]))
                continue
            }

            // Build sidechain coordinates for each rotamer
            let maxP = rotamers.first?.probability ?? 1.0
            var rotamerCoords: [[SIMD3<Float>]] = []
            var probs: [Float] = []

            for rot in rotamers {
                let scAtoms = SidechainTopologyStore.buildSidechain(
                    n: n, ca: ca, c: c,
                    residueType: oneLetterCode,
                    chiAngles: rot.chiAngles
                )
                // CB is atom[0] in sidechain build, remaining are the actual sidechain
                // Skip CB (index 0), keep the rest as "sidechain atoms beyond CB"
                let coords = scAtoms.dropFirst().map(\.position)
                rotamerCoords.append(Array(coords))
                probs.append(rot.probability)
            }

            sites.append(FASPRSite(
                siteIndex: groupIdx, residueType: oneLetterCode,
                backbone: bbPositions, atomParams: atomParams, atomTypeIndices: atomTypeIndices,
                nrots: rotamers.count, rotamerProbs: probs, maxProb: maxP,
                rotamerCoords: rotamerCoords))
        }

        return (sites, residueToSiteMap)
    }

    // MARK: - Step 3: Contact Map

    private static func buildContactMap(sites: [FASPRSite]) -> [[Int]] {
        let n = sites.count
        var conMap = [[Int]](repeating: [], count: n)

        for i in 0..<n {
            guard sites[i].nrots >= 2 else { continue }
            let cbI = sites[i].backbone.count >= 5 ? sites[i].backbone[4] : sites[i].backbone[1]  // CB or CA
            let radI = FASPRVDWParameters.residueRadius[sites[i].residueType] ?? 3.2

            for j in 0..<n where j != i {
                let cbJ: SIMD3<Float>
                if sites[j].residueType == "G" || sites[j].residueType == "A" {
                    cbJ = sites[j].backbone.count >= 2 ? sites[j].backbone[1] : .zero
                } else {
                    cbJ = sites[j].backbone.count >= 5 ? sites[j].backbone[4] : sites[j].backbone[1]
                }
                let radJ = FASPRVDWParameters.residueRadius[sites[j].residueType] ?? 3.2

                let cbDist = simd_distance(cbI, cbJ)
                let caDist = simd_distance(sites[i].backbone[1], sites[j].backbone.count >= 2 ? sites[j].backbone[1] : .zero)

                if cbDist < radI + radJ + FASPRVDWParameters.residueDistCut
                    && cbDist < caDist + FASPRVDWParameters.cacbDistCut {
                    conMap[i].append(j)
                }
            }
        }

        return conMap
    }

    // MARK: - Step 6: Apply Results

    private static func applyBestRotamers(
        bestrot: [Int],
        sites: [FASPRSite],
        residueGroups: [ResidueGroup],
        residueToSiteMap: [String: Int],
        atoms: inout [Atom],
        bonds: inout [Bond]
    ) {
        for (siteIdx, rot) in bestrot.enumerated() {
            guard rot >= 0, siteIdx < sites.count else { continue }
            let site = sites[siteIdx]
            guard site.nrots >= 2, rot < site.rotamerCoords.count else { continue }

            let group = residueGroups[siteIdx]
            let rotCoords = site.rotamerCoords[rot]

            // Get sidechain atom names from topology
            guard let topo = SidechainTopologyStore.topologies[site.residueType] else { continue }
            let scAtomNames = topo.atoms.dropFirst().map(\.name)  // skip CB

            // Update or add sidechain atoms
            let backboneNames: Set<String> = [" N  ", " CA ", " C  ", " O  ", " CB "]

            // Find existing sidechain atoms to update
            var existingSCAtoms: [String: Int] = [:]  // atomName -> atomIndex
            for atomIdx in group.atomIndices {
                let name = atoms[atomIdx].name.padding(toLength: 4, withPad: " ", startingAt: 0)
                if !backboneNames.contains(name) {
                    existingSCAtoms[name] = atomIdx
                }
            }

            // Update positions
            for (k, atomName) in scAtomNames.enumerated() {
                guard k < rotCoords.count else { break }
                if let existingIdx = existingSCAtoms[atomName] {
                    atoms[existingIdx].position = rotCoords[k]
                }
                // If atom doesn't exist, we'd need to add it - but reconstruction should
                // have already added all heavy atoms. Just update positions.
            }
        }
    }

    // MARK: - Helpers

    private static func findBackboneAtom(_ name: String, in group: ResidueGroup, atoms: [Atom]) -> SIMD3<Float>? {
        for idx in group.atomIndices {
            let atomName = atoms[idx].name.padding(toLength: 4, withPad: " ", startingAt: 0)
            if atomName == name { return atoms[idx].position }
        }
        return nil
    }

    private static func extractExistingSidechain(
        group: ResidueGroup,
        atoms: [Atom],
        backboneNames: [String]
    ) -> [SIMD3<Float>] {
        let bbSet = Set(backboneNames)
        var coords: [SIMD3<Float>] = []
        for idx in group.atomIndices {
            let name = atoms[idx].name.padding(toLength: 4, withPad: " ", startingAt: 0)
            if !bbSet.contains(name) && atoms[idx].element != .H {
                coords.append(atoms[idx].position)
            }
        }
        return coords
    }
}

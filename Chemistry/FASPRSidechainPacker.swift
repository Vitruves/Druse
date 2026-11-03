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
    ///   - reconstructedResidueKeys: Keys (see `residueKey`) of residues whose sidechain was
    ///     reconstructed. With `onlyIncomplete`, only these are repacked; nil repacks every
    ///     rotameric residue.
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
        let built = buildSites(
            residueGroups: residueGroups,
            atoms: atoms,
            onlyIncomplete: onlyIncomplete,
            reconstructedKeys: reconstructedResidueKeys
        )
        var sites = built.sites
        let residueToSiteMap = built.residueToSiteMap

        let variableSites = sites.filter(\.isVariable)
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

        // Step 4: Compute energies (may prune high-energy rotamers from `sites`)
        let calculator = FASPREnergyCalculator(device: device)
        let (selfEnergies, pairEnergies) = calculator.computeAllEnergies(sites: &sites, conMap: conMap)

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

    /// Residue identity used for grouping and for `reconstructedResidueKeys`.
    /// Includes the insertion code so residues like 52 and 52A stay distinct.
    static func residueKey(chainID: String, residueSeq: Int, insertionCode: String) -> String {
        "\(chainID)_\(residueSeq)_\(insertionCode.trimmingCharacters(in: .whitespaces))"
    }

    /// Atom names are stored trimmed ("CA") by the parsers and reconstruction, while
    /// FASPR topology names are PDB-padded (" CA "); compare both in trimmed form.
    private static func normalizedName(_ name: String) -> String {
        ProteinResidueTemplateStore.normalizeAtomName(name)
    }

    private static let backboneAtomNames = ["N", "CA", "C", "O", "CB"]

    struct ResidueGroup {
        let key: String           // see residueKey(chainID:residueSeq:insertionCode:)
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
            let key = residueKey(chainID: atom.chainID, residueSeq: atom.residueSeq,
                                 insertionCode: atom.insertionCode)
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
    ) -> (sites: [FASPRSite], residueToSiteMap: [String: Int]) {
        let library = DunbrackRotamerLibrary.shared
        var sites: [FASPRSite] = []
        var residueToSiteMap: [String: Int] = [:]

        for (groupIdx, group) in residueGroups.enumerated() {
            guard let oneLetterCode = SidechainTopologyStore.three2one[group.residueName] else {
                // Non-standard residue, skip
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: "?",
                    backbone: [], atomParams: [], atomTypeIndices: [], isVariable: false,
                    rotamerProbs: [], maxProb: 0, rotamerCoords: []))
                residueToSiteMap[group.key] = groupIdx
                continue
            }

            residueToSiteMap[group.key] = groupIdx

            // Find backbone atoms
            var backbonePos = [SIMD3<Float>?](repeating: nil, count: 5)
            for atomIdx in group.atomIndices {
                let name = normalizedName(atoms[atomIdx].name)
                for (bi, bname) in backboneAtomNames.enumerated() {
                    if name == bname { backbonePos[bi] = atoms[atomIdx].position }
                }
            }

            guard let n = backbonePos[0], let ca = backbonePos[1], let c = backbonePos[2] else {
                // Incomplete backbone
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: oneLetterCode,
                    backbone: [], atomParams: [], atomTypeIndices: [], isVariable: false,
                    rotamerProbs: [], maxProb: 0, rotamerCoords: []))
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
                    if let prevC = findBackboneAtom("C", in: prev, atoms: atoms) {
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
                    if let nextN = findBackboneAtom("N", in: next, atoms: atoms) {
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

            let rotamers = (needsPacking && oneLetterCode != "A" && oneLetterCode != "G")
                ? library.rotamers(for: oneLetterCode, phi: phi, psi: psi)
                : []
            if rotamers.isEmpty {
                // Fixed site: Ala/Gly (no rotamer) or existing sidechain kept as the single rotamer.
                let rotCoords: [[SIMD3<Float>]] = (oneLetterCode == "A" || oneLetterCode == "G")
                    ? []
                    : [extractExistingSidechain(group: group, atoms: atoms, residueType: oneLetterCode)]
                sites.append(FASPRSite(
                    siteIndex: groupIdx, residueType: oneLetterCode,
                    backbone: bbPositions, atomParams: atomParams, atomTypeIndices: atomTypeIndices,
                    isVariable: false, rotamerProbs: rotCoords.isEmpty ? [] : [1.0], maxProb: 1.0,
                    rotamerCoords: rotCoords))
                continue
            }

            // Build sidechain coordinates for each rotamer
            let maxP = rotamers.first?.probability ?? 1.0
            var rotamerCoords: [[SIMD3<Float>]] = []
            var probs: [Float] = []
            var builtCB: SIMD3<Float>?

            for rot in rotamers {
                let scAtoms = SidechainTopologyStore.buildSidechain(
                    n: n, ca: ca, c: c,
                    residueType: oneLetterCode,
                    chiAngles: rot.chiAngles
                )
                // CB is atom[0] in sidechain build (chi-independent); the rest are
                // the sidechain atoms beyond CB in topology order.
                builtCB = builtCB ?? scAtoms.first?.position
                rotamerCoords.append(scAtoms.dropFirst().map(\.position))
                probs.append(rot.probability)
            }

            // Energy terms index CB (backbone[4]); use the ideal CB when the input lacks one.
            if bbPositions.count == 4, let cb = builtCB { bbPositions.append(cb) }

            sites.append(FASPRSite(
                siteIndex: groupIdx, residueType: oneLetterCode,
                backbone: bbPositions, atomParams: atomParams, atomTypeIndices: atomTypeIndices,
                isVariable: true, rotamerProbs: probs, maxProb: maxP,
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

            // Sites without a usable backbone (non-standard / incomplete residues)
            // carry no atom parameters and cannot take part in energy terms.
            for j in 0..<n where j != i && sites[j].backbone.count >= 4 {
                let cbJ: SIMD3<Float>
                if sites[j].residueType == "G" || sites[j].residueType == "A" {
                    cbJ = sites[j].backbone[1]
                } else {
                    cbJ = sites[j].backbone.count >= 5 ? sites[j].backbone[4] : sites[j].backbone[1]
                }
                let radJ = FASPRVDWParameters.residueRadius[sites[j].residueType] ?? 3.2

                let cbDist = simd_distance(cbI, cbJ)
                let caDist = simd_distance(sites[i].backbone[1], sites[j].backbone[1])

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
        for (siteIdx, searchRot) in bestrot.enumerated() {
            guard siteIdx < sites.count else { continue }
            let site = sites[siteIdx]
            // Sites left with a single rotamer (after self-energy pruning) are not
            // assigned by the search; their only rotamer is the optimum.
            let rot = site.nrots == 1 ? 0 : searchRot
            guard site.isVariable, rot >= 0, rot < site.rotamerCoords.count else { continue }

            let group = residueGroups[siteIdx]
            let rotCoords = site.rotamerCoords[rot]

            // Get sidechain atom names from topology
            guard let topo = SidechainTopologyStore.topologies[site.residueType] else { continue }
            let scAtomNames = topo.atoms.dropFirst().map { normalizedName($0.name) }  // skip CB

            // Find existing sidechain atoms to update
            var existingSCAtoms: [String: Int] = [:]  // atomName -> atomIndex
            for atomIdx in group.atomIndices {
                let name = normalizedName(atoms[atomIdx].name)
                if !backboneAtomNames.contains(name) {
                    existingSCAtoms[name] = atomIdx
                }
            }

            // Update positions (reconstruction has already added all heavy atoms)
            for (k, atomName) in scAtomNames.enumerated() {
                guard k < rotCoords.count else { break }
                if let existingIdx = existingSCAtoms[atomName] {
                    atoms[existingIdx].position = rotCoords[k]
                }
            }
        }
    }

    // MARK: - Helpers

    private static func findBackboneAtom(_ name: String, in group: ResidueGroup, atoms: [Atom]) -> SIMD3<Float>? {
        for idx in group.atomIndices {
            if normalizedName(atoms[idx].name) == name { return atoms[idx].position }
        }
        return nil
    }

    /// Existing sidechain atoms beyond CB in FASPR topology order, so that index k
    /// maps to atom parameter k + 5. Stops at the first missing atom to keep that
    /// mapping valid; atoms outside the topology (e.g. OXT, altlocs) are ignored.
    private static func extractExistingSidechain(
        group: ResidueGroup,
        atoms: [Atom],
        residueType: Character
    ) -> [SIMD3<Float>] {
        guard let topo = SidechainTopologyStore.topologies[residueType] else { return [] }
        var byName: [String: SIMD3<Float>] = [:]
        for idx in group.atomIndices {
            let name = normalizedName(atoms[idx].name)
            if byName[name] == nil { byName[name] = atoms[idx].position }
        }
        var coords: [SIMD3<Float>] = []
        for def in topo.atoms.dropFirst() {
            guard let pos = byName[normalizedName(def.name)] else { break }
            coords.append(pos)
        }
        return coords
    }
}

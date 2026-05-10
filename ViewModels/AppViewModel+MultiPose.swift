// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import Foundation
import simd

/// Multi-pose analysis utilities: pairwise heavy-atom RMSD between selected
/// docking poses, residue-level interaction fingerprints (presence + type),
/// and SDF/PDB/CSV export of the selection.
@MainActor
extension AppViewModel {

    /// Sorted (low → high energy) array of indices currently in the
    /// `selectedPoseIndices` set, clamped to valid result indices.
    var sortedSelectedPoseIndices: [Int] {
        docking.selectedPoseIndices
            .filter { $0 >= 0 && $0 < docking.dockingResults.count }
            .sorted { docking.dockingResults[$0].energy < docking.dockingResults[$1].energy }
    }

    // MARK: - Pairwise heavy-atom RMSD

    /// Pose-by-pose heavy-atom RMSD matrix in the order of `indices`.
    /// `matrix[i][j]` is the RMSD between `indices[i]` and `indices[j]`.
    /// Returns an empty matrix if any pose lacks transformed coordinates or
    /// the selected poses have inconsistent atom counts.
    func computePairwiseRMSD(indices: [Int]) -> [[Float]] {
        let results = docking.dockingResults
        guard !indices.isEmpty else { return [] }
        let coords: [[SIMD3<Float>]] = indices.compactMap { idx in
            guard idx >= 0, idx < results.count else { return nil }
            let p = results[idx].transformedAtomPositions
            return p.isEmpty ? nil : p
        }
        guard coords.count == indices.count, let first = coords.first else { return [] }
        let atomCount = first.count
        // Bail if pose ensembles differ in atom count (different ligands?)
        guard coords.allSatisfy({ $0.count == atomCount }) else { return [] }

        let n = indices.count
        var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                var sumSq: Float = 0
                for k in 0..<atomCount {
                    let d = coords[i][k] - coords[j][k]
                    sumSq += dot(d, d)
                }
                let rmsd = sqrt(sumSq / Float(max(atomCount, 1)))
                matrix[i][j] = rmsd
                matrix[j][i] = rmsd
            }
        }
        return matrix
    }

    // MARK: - Interaction fingerprints

    /// Per-residue interaction fingerprint for each pose in `indices`.
    /// Returns the union of contacted residue keys (ordered by chain+seq) and
    /// a parallel array of dictionaries mapping residue key → set of types.
    func computeInteractionFingerprints(
        indices: [Int]
    ) -> (residueKeys: [String], residueLabels: [String: String],
          fingerprints: [[String: Set<MolecularInteraction.InteractionType>]]) {
        guard let origLig = docking.originalDockingLigand ?? molecules.ligand,
              let prot = molecules.protein else {
            return ([], [:], [])
        }

        let heavyLigAtoms = origLig.atoms.filter { $0.element != .H }
        let heavyBonds = buildHeavyBonds(from: origLig)
        let heavyProtAtoms = prot.atoms.filter { $0.element != .H }

        var allKeys = Set<String>()
        var labelByKey: [String: String] = [:]
        var perPose: [[String: Set<MolecularInteraction.InteractionType>]] = []

        for idx in indices {
            guard idx >= 0, idx < docking.dockingResults.count else {
                perPose.append([:])
                continue
            }
            let pose = docking.dockingResults[idx]
            guard !pose.transformedAtomPositions.isEmpty else {
                perPose.append([:])
                continue
            }
            let interactions = InteractionDetector.detect(
                ligandAtoms: heavyLigAtoms,
                ligandPositions: pose.transformedAtomPositions,
                proteinAtoms: heavyProtAtoms,
                ligandBonds: heavyBonds,
                scoringMethod: docking.scoringMethod
            )
            var poseMap: [String: Set<MolecularInteraction.InteractionType>] = [:]
            for inter in interactions {
                guard inter.proteinAtomIndex >= 0, inter.proteinAtomIndex < heavyProtAtoms.count else { continue }
                let a = heavyProtAtoms[inter.proteinAtomIndex]
                let key = "\(a.chainID)_\(a.residueSeq)_\(a.residueName)"
                allKeys.insert(key)
                labelByKey[key] = "\(a.residueName)\(a.residueSeq)\(a.chainID.isEmpty ? "" : "/\(a.chainID)")"
                poseMap[key, default: []].insert(inter.type)
            }
            perPose.append(poseMap)
        }

        let ordered = allKeys.sorted { lhs, rhs in
            let lp = lhs.split(separator: "_")
            let rp = rhs.split(separator: "_")
            let lc = lp.count > 0 ? String(lp[0]) : ""
            let rc = rp.count > 0 ? String(rp[0]) : ""
            if lc != rc { return lc < rc }
            let ls = lp.count > 1 ? Int(lp[1]) ?? 0 : 0
            let rs = rp.count > 1 ? Int(rp[1]) ?? 0 : 0
            return ls < rs
        }
        return (ordered, labelByKey, perPose)
    }

    // MARK: - Selected-subset exports

    /// Export only the currently-selected poses to a multi-mol SDF.
    func exportSelectedDockingPosesSDF() {
        let indices = sortedSelectedPoseIndices
        guard !indices.isEmpty,
              let lig = docking.originalDockingLigand ?? molecules.ligand else {
            log.warn("No selected poses to export", category: .dock)
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "sdf")].compactMap { $0 }
        panel.nameFieldStringValue = "selected_poses.sdf"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let heavyAtoms = lig.atoms.filter { $0.element != .H }
        var molBlocks: [String] = []
        molBlocks.reserveCapacity(indices.count)
        let method = docking.scoringMethod

        for idx in indices {
            let result = docking.dockingResults[idx]
            var poseAtoms: [Atom] = []
            var idMap: [Int: Int] = [:]
            for (j, atom) in heavyAtoms.enumerated() {
                guard j < result.transformedAtomPositions.count else { break }
                idMap[atom.id] = poseAtoms.count
                poseAtoms.append(Atom(
                    id: poseAtoms.count, element: atom.element,
                    position: result.transformedAtomPositions[j],
                    name: atom.name, residueName: atom.residueName,
                    residueSeq: atom.residueSeq, chainID: atom.chainID,
                    charge: atom.charge, formalCharge: atom.formalCharge,
                    isHetAtom: atom.isHetAtom
                ))
            }
            var poseBonds: [Bond] = []
            for bond in lig.bonds {
                if let a = idMap[bond.atomIndex1], let b = idMap[bond.atomIndex2] {
                    poseBonds.append(Bond(id: poseBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }
            let props: [String: String] = [
                "Rank": "\(idx + 1)",
                "Score": String(format: "%.3f", result.displayScore(method: method)),
                "ScoreUnit": method.unitLabel,
                "Energy": String(format: "%.3f", result.energy),
                "VdW": String(format: "%.3f", result.vdwEnergy),
                "Elec": String(format: "%.3f", result.elecEnergy),
                "HBond": String(format: "%.3f", result.hbondEnergy),
                "Cluster": "\(result.clusterID)"
            ]
            molBlocks.append(SDFWriter.molBlock(
                name: "\(lig.name)_pose\(idx + 1)",
                atoms: poseAtoms, bonds: poseBonds, properties: props
            ))
        }

        do {
            try SDFWriter.save(molBlocks.joined(), to: url)
            log.success("Exported \(indices.count) selected pose(s) to SDF", category: .dock)
        } catch {
            log.error("SDF export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    /// Export the protein and the currently-selected poses as a single PDB
    /// where each pose appears as a residue under chain X with `residueSeq`
    /// equal to its rank.
    func exportSelectedDockingPosesPDB() {
        let indices = sortedSelectedPoseIndices
        guard !indices.isEmpty,
              let lig = docking.originalDockingLigand ?? molecules.ligand else {
            log.warn("No selected poses to export", category: .dock)
            return
        }
        guard indices.allSatisfy({ idx in
            !docking.dockingResults[idx].transformedAtomPositions.isEmpty
        }) else {
            log.warn("Selected poses are missing transformed coordinates", category: .dock)
            return
        }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "pdb")].compactMap { $0 }
        panel.nameFieldStringValue = "\(lig.name)_selected.pdb"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let method = docking.scoringMethod
        let poses = indices.map { idx -> (rank: Int, positions: [SIMD3<Float>], scoreLine: String) in
            let r = docking.dockingResults[idx]
            let line = String(format: "Druse pose #%d  score=%.2f %@",
                              idx + 1, r.displayScore(method: method), method.unitLabel)
            return (rank: idx + 1, positions: r.transformedAtomPositions, scoreLine: line)
        }
        let title = "Druse selected poses (n=\(indices.count))"
        let pdb = PDBWriter.combinedMultiPose(
            protein: molecules.protein,
            ligandTemplate: lig,
            poses: poses,
            title: title
        )
        do {
            try PDBWriter.save(pdb, to: url)
            log.success("Exported \(indices.count) selected pose(s) to \(url.lastPathComponent)", category: .dock)
        } catch {
            log.error("PDB export failed: \(error.localizedDescription)", category: .dock)
        }
    }

    /// Export only the currently-selected pose rows to CSV.
    func exportSelectedDockingPosesCSV() {
        let indices = sortedSelectedPoseIndices
        guard !indices.isEmpty else {
            log.warn("No selected poses to export", category: .dock)
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "csv")].compactMap { $0 }
        panel.nameFieldStringValue = "selected_poses.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        var rows: [String] = ["Rank,Cluster,Energy_kcal_mol,VdW,HBond,Torsion,StrainEnergy,Generation"]
        rows.reserveCapacity(indices.count + 1)
        for idx in indices {
            let r = docking.dockingResults[idx]
            let fields: [String] = [
                "\(idx + 1)",
                "\(r.clusterID)",
                String(format: "%.2f", r.energy),
                String(format: "%.2f", r.vdwEnergy),
                String(format: "%.2f", r.hbondEnergy),
                String(format: "%.2f", r.torsionPenalty),
                r.strainEnergy.map { String(format: "%.2f", $0) } ?? "",
                "\(r.generation)"
            ]
            rows.append(fields.joined(separator: ","))
        }
        do {
            try (rows.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
            log.success("Exported \(indices.count) selected pose(s) to CSV", category: .dock)
        } catch {
            log.error("CSV export failed: \(error.localizedDescription)", category: .dock)
        }
    }
}

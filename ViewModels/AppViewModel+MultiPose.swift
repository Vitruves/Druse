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

    // MARK: - Ligand placements and interactions

    /// Heavy-atom geometry of a ligand placed in the receptor: a docked pose or the loaded
    /// ligand as it stands (e.g. the co-crystallised ligand). Input of interaction, contact
    /// and validity analysis.
    struct LigandPlacement {
        /// "pose 3" or the ligand's name.
        let label: String
        let atoms: [Atom]
        /// Bonds re-indexed onto `atoms`.
        let bonds: [Bond]
        let positions: [SIMD3<Float>]
    }

    /// Placement of docking pose `index` (0-based into `docking.dockingResults`), or of the
    /// loaded ligand when `index` is nil. Nil when the pose or ligand does not exist or the
    /// pose carries no coordinates.
    func ligandPlacement(pose index: Int?) -> LigandPlacement? {
        if let index {
            guard docking.dockingResults.indices.contains(index),
                  let ligand = docking.originalDockingLigand ?? molecules.ligand else { return nil }
            let positions = docking.dockingResults[index].transformedAtomPositions
            let heavy = ligand.atoms.filter { $0.element != .H }
            guard !positions.isEmpty, positions.count == heavy.count else { return nil }
            return LigandPlacement(label: "pose \(index + 1)", atoms: heavy,
                                   bonds: buildHeavyBonds(from: ligand), positions: positions)
        }
        guard let ligand = molecules.ligand else { return nil }
        let heavy = ligand.atoms.filter { $0.element != .H }
        return LigandPlacement(label: ligand.name, atoms: heavy,
                               bonds: buildHeavyBonds(from: ligand), positions: heavy.map(\.position))
    }

    /// Protein–ligand interactions of a placement, each with the protein heavy atom it
    /// involves. By default only the interaction types the active scoring function accounts
    /// for are reported (as in the 3D view); `allTypes` reports every type the detector knows.
    func interactions(of placement: LigandPlacement, allTypes: Bool = false)
        -> [(interaction: MolecularInteraction, proteinAtom: Atom)] {
        guard let prot = molecules.protein else { return [] }
        let heavyProtAtoms = prot.atoms.filter { $0.element != .H }
        // Drusina accounts for every interaction type, so it serves as the "all types" filter.
        let found = InteractionDetector.detect(
            ligandAtoms: placement.atoms,
            ligandPositions: placement.positions,
            proteinAtoms: heavyProtAtoms,
            ligandBonds: placement.bonds,
            scoringMethod: allTypes ? .drusina : docking.scoringMethod
        )
        return found.compactMap { inter in
            guard heavyProtAtoms.indices.contains(inter.proteinAtomIndex) else { return nil }
            return (inter, heavyProtAtoms[inter.proteinAtomIndex])
        }
    }

    /// Heavy-atom positions of the docked ligand's input coordinates when they are a placement in
    /// the docking box (a co-crystallised or previously posed ligand). Nil when there is no input
    /// ligand or its centroid lies outside the box (2 Å margin): SMILES-built and prepared conformers
    /// sit at the origin, and an "RMSD to the input pose" of 40 Å would only mislead.
    var inputPoseReference: [SIMD3<Float>]? {
        guard let ligand = docking.originalDockingLigand else { return nil }
        let reference = ligand.atoms.filter { $0.element != .H }.map(\.position)
        guard !reference.isEmpty else { return nil }
        if let pocket = docking.selectedPocket {
            let centroid = reference.reduce(SIMD3<Float>.zero, +) / Float(reference.count)
            let offset = abs(centroid - pocket.center) - pocket.size
            guard offset.max() <= 2 else { return nil }
        }
        return reference
    }

    /// Heavy-atom RMSD (Å) of docking pose `index` against the input coordinates of the docked
    /// ligand — the redocking RMSD when the input was a co-crystallised ligand. Nil when the
    /// pose has no coordinates, the atom counts differ or the input is not placed in the box.
    func rmsdToInputLigand(pose index: Int) -> Float? {
        guard docking.dockingResults.indices.contains(index),
              let reference = inputPoseReference else { return nil }
        let pose = docking.dockingResults[index].transformedAtomPositions
        guard !reference.isEmpty, pose.count == reference.count else { return nil }
        let sumSq = zip(reference, pose).reduce(Float(0)) { $0 + simd_distance_squared($1.0, $1.1) }
        return sqrt(sumSq / Float(reference.count))
    }

    // MARK: - Interaction fingerprints

    /// Per-residue interaction fingerprint for each pose in `indices`.
    /// Returns the union of contacted residue keys (ordered by chain+seq) and
    /// a parallel array of dictionaries mapping residue key → set of types.
    func computeInteractionFingerprints(
        indices: [Int]
    ) -> (residueKeys: [String], residueLabels: [String: String],
          fingerprints: [[String: Set<MolecularInteraction.InteractionType>]]) {
        guard docking.originalDockingLigand ?? molecules.ligand != nil, molecules.protein != nil else {
            return ([], [:], [])
        }

        var allKeys = Set<String>()
        var labelByKey: [String: String] = [:]
        var perPose: [[String: Set<MolecularInteraction.InteractionType>]] = []

        for idx in indices {
            guard let placement = ligandPlacement(pose: idx) else {
                perPose.append([:])
                continue
            }
            var poseMap: [String: Set<MolecularInteraction.InteractionType>] = [:]
            for (inter, a) in interactions(of: placement) {
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
                "VdW": String(format: "%.3f", result.stericEnergy),
                "Hydrophobic": String(format: "%.3f", result.hydrophobicEnergy),
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
                String(format: "%.2f", r.stericEnergy),
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

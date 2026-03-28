import Foundation
import Metal
import simd

extension ProteinPreparation {

    struct ReconstructionReport: Sendable {
        var heavyAtomsAdded: Int = 0
        var hydrogensAdded: Int = 0
        var networkReport: HBondNetworkOptimizer.NetworkReport?
    }

    private struct ReconstructionResidueKey: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let isHetAtom: Bool
    }

    private struct OverlayTransform {
        let rotation: simd_float3x3
        let movingCenter: SIMD3<Float>
        let referenceCenter: SIMD3<Float>

        func apply(_ point: SIMD3<Float>) -> SIMD3<Float> {
            (rotation * (point - movingCenter)) + referenceCenter
        }
    }

    static func completePhase23(
        atoms: [Atom],
        bonds: [Bond],
        pH: Float = 7.4,
        polarOnly: Bool = false,
        device: MTLDevice? = nil
    ) -> (atoms: [Atom], bonds: [Bond], report: ReconstructionReport, protonation: [Protonation.ResiduePrediction]) {
        let logSync = { (msg: String) in
            _ = Task { @MainActor in ActivityLog.shared.debug(msg, category: .prep) }
        }

        logSync("[Phase2] Reconstructing heavy atoms (\(atoms.count) input)...")
        let reconstructed = reconstructMissingHeavyAtoms(atoms: atoms, bonds: bonds)
        logSync("[Phase2] Reconstruction done: \(reconstructed.atoms.count) atoms (+\(reconstructed.addedAtomCount) added)")

        // Phase 2b: FASPR sidechain packing
        var workAtoms = reconstructed.atoms
        var workBonds = reconstructed.bonds
        if DunbrackRotamerLibrary.shared.isLoaded {
            logSync("[Phase2b] FASPR sidechain packing (\(workAtoms.count) atoms)...")
            let packed = FASPRSidechainPacker.packSidechains(
                atoms: workAtoms,
                bonds: workBonds,
                device: device,
                onlyIncomplete: true
            )
            workAtoms = packed.atoms
            workBonds = packed.bonds
            logSync("[Phase2b] Packing done: \(packed.report.residuesPacked) residues in \(String(format: "%.0f", packed.report.elapsedMs))ms")
        }

        // Phase 2c: Preparation minimization
        if let dev = device, let minimizer = PreparationMinimizer(device: dev) {
            logSync("[Phase2c] Preparation minimization (\(workAtoms.count) atoms)...")
            let regions: [PreparationMinimizer.AtomRegion] = workAtoms.map { atom in
                if atom.element == .H { return .hydrogen }
                let name = atom.name.padding(toLength: 4, withPad: " ", startingAt: 0)
                if name == " N  " || name == " CA " || name == " C  " || name == " O  " {
                    return .backbone
                }
                return .existingSidechain
            }

            let result = minimizer.minimize(
                input: PreparationMinimizer.MinimizationInput(
                    atoms: workAtoms,
                    bonds: workBonds,
                    atomRegions: regions,
                    angles: [],
                    torsions: []
                ),
                maxIterations: 100
            )

            for i in 0..<min(result.positions.count, workAtoms.count) {
                workAtoms[i].position = result.positions[i]
            }
            logSync("[Phase2c] Minimization done: E=\(String(format: "%.1f", result.finalEnergy)) kcal/mol, \(result.iterations) iterations, converged=\(result.converged)")
        }

        logSync("[Phase2] Predicting protonation states at pH \(pH)...")
        let protonation = Protonation.predictResidueStates(
            atoms: workAtoms,
            bonds: workBonds,
            pH: pH
        )
        logSync("[Phase2] Protonation: \(protonation.count) predictions")

        let chargedAtoms = Protonation.applyProtonation(atoms: workAtoms, predictions: protonation)

        logSync("[Phase3] Adding template hydrogens (\(chargedAtoms.count) atoms, polarOnly=\(polarOnly))...")
        let hydrogenated = addTemplateHydrogens(
            atoms: chargedAtoms,
            bonds: workBonds,
            pH: pH,
            predictions: protonation,
            polarOnly: polarOnly
        )
        logSync("[Phase3] Hydrogens added: \(hydrogenated.atoms.count) atoms (+\(hydrogenated.addedAtomCount) H)")

        // Phase 4: H-bond network optimization (rotamer search, flips, scoring)
        logSync("[Phase4] H-bond network optimization (\(hydrogenated.atoms.count) atoms)...")
        let network = HBondNetworkOptimizer.optimizeNetwork(
            atoms: hydrogenated.atoms,
            bonds: hydrogenated.bonds,
            predictions: protonation
        )
        logSync("[Phase4] Network done: \(network.report.moveableGroups) groups, \(network.report.cliques) cliques")

        var report = ReconstructionReport(
            heavyAtomsAdded: reconstructed.addedAtomCount,
            hydrogensAdded: hydrogenated.addedAtomCount
        )
        report.networkReport = network.report

        return (
            atoms: network.atoms,
            bonds: network.bonds,
            report: report,
            protonation: protonation
        )
    }

    /// Canonical bond key for O(1) lookup. Encodes two atom indices as a single UInt64.
    private static func bondKey(_ a: Int, _ b: Int) -> UInt64 {
        let lo = UInt64(min(a, b))
        let hi = UInt64(max(a, b))
        return (hi << 32) | lo
    }

    static func reconstructMissingHeavyAtoms(
        atoms: [Atom],
        bonds: [Bond]
    ) -> (atoms: [Atom], bonds: [Bond], addedAtomCount: Int) {
        var workingAtoms = atoms
        var workingBonds = bonds
        var addedAtomCount = 0

        // Tier 3: Build Set<UInt64> for O(1) bond existence checks
        var bondSet = Set<UInt64>(minimumCapacity: bonds.count)
        for bond in bonds {
            bondSet.insert(bondKey(bond.atomIndex1, bond.atomIndex2))
        }

        let residueOrder = residueKeys(in: workingAtoms)
        let externalBondedAtoms = externalBondedAtomNamesByResidue(atoms: workingAtoms, bonds: workingBonds)

        for residueKey in residueOrder {
            guard let template = ProteinResidueReferenceTemplateStore.template(for: residueKey.residueName) else {
                continue
            }

            let atomIndices = atomIndexMap(for: residueKey, atoms: workingAtoms, template: template)
            let filteredAtomNames = template.filteredAtomNames(
                externalBondedAtoms: externalBondedAtoms[residueKey, default: []],
                includeHydrogens: false
            )

            let anchorNames = filteredAtomNames.filter { atomIndices[$0] != nil }
            guard let transform = overlayTransform(
                template: template,
                anchorNames: anchorNames,
                atomIndices: atomIndices,
                atoms: workingAtoms
            ) else {
                continue
            }

            for atomName in filteredAtomNames {
                guard atomIndices[atomName] == nil,
                      let atomTemplate = template.atom(named: atomName),
                      !atomTemplate.isHydrogen else {
                    continue
                }

                workingAtoms.append(Atom(
                    id: workingAtoms.count,
                    element: atomTemplate.element,
                    position: transform.apply(atomTemplate.idealPosition),
                    name: atomTemplate.atomName,
                    residueName: residueKey.residueName,
                    residueSeq: residueKey.residueSeq,
                    chainID: residueKey.chainID,
                    formalCharge: atomTemplate.formalCharge,
                    isHetAtom: residueKey.isHetAtom
                ))
                addedAtomCount += 1
            }

            let refreshedAtomIndices = atomIndexMap(for: residueKey, atoms: workingAtoms, template: template)
            for bondTemplate in template.bonds {
                guard let atomIndex1 = refreshedAtomIndices[bondTemplate.atomName1],
                      let atomIndex2 = refreshedAtomIndices[bondTemplate.atomName2],
                      workingAtoms[atomIndex1].element != .H,
                      workingAtoms[atomIndex2].element != .H,
                      !bondSet.contains(bondKey(atomIndex1, atomIndex2)) else {
                    continue
                }

                workingBonds.append(Bond(
                    id: workingBonds.count,
                    atomIndex1: atomIndex1,
                    atomIndex2: atomIndex2,
                    order: bondTemplate.order
                ))
                bondSet.insert(bondKey(atomIndex1, atomIndex2))
            }
        }

        return (workingAtoms, workingBonds, addedAtomCount)
    }

    /// Builds a pre-computed adjacency list from bonds, for O(1) neighbor lookup per atom.
    private static func buildAdjacencyList(atomCount: Int, bonds: [Bond]) -> [[Int]] {
        var adj = [[Int]](repeating: [], count: atomCount)
        for bond in bonds {
            adj[bond.atomIndex1].append(bond.atomIndex2)
            adj[bond.atomIndex2].append(bond.atomIndex1)
        }
        return adj
    }

    static func addTemplateHydrogens(
        atoms: [Atom],
        bonds: [Bond],
        pH: Float = 7.4,
        predictions: [Protonation.ResiduePrediction]? = nil,
        polarOnly: Bool = false
    ) -> (atoms: [Atom], bonds: [Bond], addedAtomCount: Int) {
        let protonation = predictions ?? Protonation.predictResidueStates(
            atoms: atoms,
            bonds: bonds,
            pH: pH
        )
        let stripped = removeProteinAndCapHydrogens(atoms: atoms, bonds: bonds)

        var workingAtoms = stripped.atoms
        var workingBonds = stripped.bonds
        var addedAtomCount = 0

        // Tier 3: Pre-build adjacency list for O(degree) neighbor lookups instead of O(B) scans
        var adjacency = buildAdjacencyList(atomCount: workingAtoms.count, bonds: workingBonds)

        let residueOrder = residueKeys(in: workingAtoms)
        let externalBondedAtoms = externalBondedAtomNamesByResidue(atoms: workingAtoms, bonds: workingBonds)
        let predictionMap = Dictionary(uniqueKeysWithValues: protonation.map {
            ("\($0.chainID):\($0.residueSeq):\($0.residueName)", $0)
        })

        for residueKey in residueOrder {
            guard let template = ProteinResidueReferenceTemplateStore.template(for: residueKey.residueName) else {
                continue
            }

            let residueExternalBondedAtoms = externalBondedAtoms[residueKey, default: []]
            let atomIndices = atomIndexMap(for: residueKey, atoms: workingAtoms, template: template)
            let anchorNames = template.filteredAtomNames(
                externalBondedAtoms: residueExternalBondedAtoms,
                includeHydrogens: false
            ).filter { atomIndices[$0] != nil }

            guard let transform = overlayTransform(
                template: template,
                anchorNames: anchorNames,
                atomIndices: atomIndices,
                atoms: workingAtoms
            ) else {
                continue
            }

            let directPrediction = predictionMap["\(residueKey.chainID):\(residueKey.residueSeq):\(residueKey.residueName)"]
            let nTermPrediction = predictionMap["\(residueKey.chainID):\(residueKey.residueSeq):*N-term*"]
            let cTermPrediction = predictionMap["\(residueKey.chainID):\(residueKey.residueSeq):*C-term*"]
            let freeNTerminus = !residueExternalBondedAtoms.contains("N")
            let freeCTerminus = !residueExternalBondedAtoms.contains("C")
            let directTitratableAtoms = titratableParentAtoms(for: residueKey.residueName)
            let protonatedDirectAtoms = directPrediction?.protonatedAtoms ?? []
            let protonatedCTermAtoms = cTermPrediction?.protonatedAtoms ?? []

            let filteredTemplateAtomNames = template.filteredAtomNames(
                externalBondedAtoms: residueExternalBondedAtoms,
                includeHydrogens: true
            )

            for atomName in filteredTemplateAtomNames {
                guard let atomTemplate = template.atom(named: atomName),
                      atomTemplate.isHydrogen else {
                    continue
                }

                let parentAtomName = template.bondedAtomNames(to: atomTemplate.atomName).first(where: { candidate in
                    template.atom(named: candidate)?.isHydrogen == false
                })
                guard let parentAtomName,
                      let parentAtomIndex = atomIndices[parentAtomName] else {
                    continue
                }

                // In polar-only mode, skip hydrogens bonded to carbon
                if polarOnly && workingAtoms[parentAtomIndex].element == .C {
                    continue
                }

                if freeCTerminus && (parentAtomName == "OXT" || parentAtomName == "O") {
                    guard protonatedCTermAtoms.contains(parentAtomName) else {
                        continue
                    }
                } else if directTitratableAtoms.contains(parentAtomName) {
                    guard protonatedDirectAtoms.contains(parentAtomName) else {
                        continue
                    }
                }

                let hydrogenIndex = workingAtoms.count
                var hPosition = localHydrogenPosition(
                    hydrogenTemplate: atomTemplate,
                    parentAtomName: parentAtomName,
                    template: template,
                    atomIndices: atomIndices,
                    atoms: workingAtoms,
                    fallbackTransform: transform
                )

                // Guard against non-finite positions from degenerate geometry
                if !hPosition.x.isFinite || !hPosition.y.isFinite || !hPosition.z.isFinite {
                    let parentPos = workingAtoms[parentAtomIndex].position
                    hPosition = parentPos + SIMD3<Float>(0, 1.0, 0)
                }

                // Guard against unreasonable bond lengths (> 5 Å from parent)
                let bondLen = simd_distance(hPosition, workingAtoms[parentAtomIndex].position)
                if bondLen > 5.0 {
                    let parentPos = workingAtoms[parentAtomIndex].position
                    // Tier 3: Use adjacency list for O(degree) neighbor lookup
                    let neighborPositions = parentAtomIndex < adjacency.count
                        ? adjacency[parentAtomIndex].map { workingAtoms[$0].position }
                        : bondedNeighborPositions(of: parentAtomIndex, atoms: workingAtoms, bonds: workingBonds)
                    hPosition = placeHydrogenLike(on: parentPos, awayFrom: neighborPositions, distance: 1.0)
                }

                workingAtoms.append(Atom(
                    id: hydrogenIndex,
                    element: .H,
                    position: hPosition,
                    name: atomTemplate.atomName,
                    residueName: residueKey.residueName,
                    residueSeq: residueKey.residueSeq,
                    chainID: residueKey.chainID,
                    isHetAtom: residueKey.isHetAtom
                ))
                workingBonds.append(Bond(
                    id: workingBonds.count,
                    atomIndex1: parentAtomIndex,
                    atomIndex2: hydrogenIndex,
                    order: .single
                ))
                // Maintain adjacency list for newly added atoms
                adjacency.append([parentAtomIndex])  // hydrogenIndex's neighbors
                if parentAtomIndex < adjacency.count - 1 {
                    adjacency[parentAtomIndex].append(hydrogenIndex)
                }
                addedAtomCount += 1
            }

            if freeNTerminus, nTermPrediction?.state == .protonated, let nitrogenIndex = atomIndices["N"] {
                let existingHydrogenNames = Set(residueIndices(for: residueKey, atoms: workingAtoms).map { index in
                    ProteinResidueTemplateStore.normalizeAtomName(workingAtoms[index].name)
                })
                if let extraHydrogenName = nextTerminalHydrogenName(existingHydrogenNames: existingHydrogenNames) {
                    let extraHydrogenIndex = workingAtoms.count
                    // Tier 3: Use adjacency list for O(degree) neighbor lookup
                    let nitrogenNeighborPositions = nitrogenIndex < adjacency.count
                        ? adjacency[nitrogenIndex].map { workingAtoms[$0].position }
                        : bondedNeighborPositions(of: nitrogenIndex, atoms: workingAtoms, bonds: workingBonds)
                    let position = placeHydrogenLike(
                        on: workingAtoms[nitrogenIndex].position,
                        awayFrom: nitrogenNeighborPositions,
                        distance: 1.01
                    )
                    workingAtoms.append(Atom(
                        id: extraHydrogenIndex,
                        element: .H,
                        position: position,
                        name: extraHydrogenName,
                        residueName: residueKey.residueName,
                        residueSeq: residueKey.residueSeq,
                        chainID: residueKey.chainID,
                        isHetAtom: residueKey.isHetAtom
                    ))
                    workingBonds.append(Bond(
                        id: workingBonds.count,
                        atomIndex1: nitrogenIndex,
                        atomIndex2: extraHydrogenIndex,
                        order: .single
                    ))
                    // Maintain adjacency list
                    adjacency.append([nitrogenIndex])
                    if nitrogenIndex < adjacency.count - 1 {
                        adjacency[nitrogenIndex].append(extraHydrogenIndex)
                    }
                    addedAtomCount += 1
                }
            }
        }

        return (workingAtoms, workingBonds, addedAtomCount)
    }

    private static func removeProteinAndCapHydrogens(
        atoms: [Atom],
        bonds: [Bond]
    ) -> (atoms: [Atom], bonds: [Bond]) {
        let retainedIndices = atoms.indices.filter { index in
            let atom = atoms[index]
            guard atom.element == .H || ProteinResidueTemplateStore.normalizeAtomName(atom.name).hasPrefix("H") else {
                return true
            }
            let residueName = ProteinResidueTemplateStore.normalizeAtomName(atom.residueName)
            return ProteinResidueReferenceTemplateStore.template(for: residueName) == nil
        }
        return remapSubstructure(atoms: atoms, bonds: bonds, selectedIndices: retainedIndices)
    }

    private static func residueKeys(in atoms: [Atom]) -> [ReconstructionResidueKey] {
        var orderedKeys: [ReconstructionResidueKey] = []
        var seen: Set<ReconstructionResidueKey> = []

        for atom in atoms {
            let residueName = ProteinResidueTemplateStore.normalizeAtomName(atom.residueName)
            guard ProteinResidueReferenceTemplateStore.template(for: residueName) != nil else {
                continue
            }

            let key = ReconstructionResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: residueName,
                isHetAtom: atom.isHetAtom
            )
            if seen.insert(key).inserted {
                orderedKeys.append(key)
            }
        }

        return orderedKeys.sorted { lhs, rhs in
            if lhs.chainID != rhs.chainID {
                return lhs.chainID < rhs.chainID
            }
            if lhs.residueSeq != rhs.residueSeq {
                return lhs.residueSeq < rhs.residueSeq
            }
            return lhs.residueName < rhs.residueName
        }
    }

    private static func residueIndices(
        for residueKey: ReconstructionResidueKey,
        atoms: [Atom]
    ) -> [Int] {
        atoms.indices.filter { index in
            let atom = atoms[index]
            return atom.chainID == residueKey.chainID &&
                atom.residueSeq == residueKey.residueSeq &&
                ProteinResidueTemplateStore.normalizeAtomName(atom.residueName) == residueKey.residueName
        }
    }

    private static func atomIndexMap(
        for residueKey: ReconstructionResidueKey,
        atoms: [Atom],
        template: ProteinResidueReferenceTemplateStore.Template? = nil
    ) -> [String: Int] {
        var map: [String: Int] = [:]
        for index in residueIndices(for: residueKey, atoms: atoms) {
            let normalizedName = ProteinResidueTemplateStore.normalizeAtomName(atoms[index].name)
            let canonicalName = template?.canonicalAtomName(normalizedName) ?? normalizedName
            map[canonicalName] = index
        }
        return map
    }

    private static func externalBondedAtomNamesByResidue(
        atoms: [Atom],
        bonds: [Bond]
    ) -> [ReconstructionResidueKey: Set<String>] {
        var result: [ReconstructionResidueKey: Set<String>] = [:]

        for bond in bonds {
            let atom1 = atoms[bond.atomIndex1]
            let atom2 = atoms[bond.atomIndex2]

            let residueKey1 = ReconstructionResidueKey(
                chainID: atom1.chainID,
                residueSeq: atom1.residueSeq,
                residueName: ProteinResidueTemplateStore.normalizeAtomName(atom1.residueName),
                isHetAtom: atom1.isHetAtom
            )
            let residueKey2 = ReconstructionResidueKey(
                chainID: atom2.chainID,
                residueSeq: atom2.residueSeq,
                residueName: ProteinResidueTemplateStore.normalizeAtomName(atom2.residueName),
                isHetAtom: atom2.isHetAtom
            )

            guard residueKey1 != residueKey2 else { continue }
            result[residueKey1, default: []].insert(ProteinResidueTemplateStore.normalizeAtomName(atom1.name))
            result[residueKey2, default: []].insert(ProteinResidueTemplateStore.normalizeAtomName(atom2.name))
        }

        return result
    }

    private static func overlayTransform(
        template: ProteinResidueReferenceTemplateStore.Template,
        anchorNames: [String],
        atomIndices: [String: Int],
        atoms: [Atom]
    ) -> OverlayTransform? {
        guard !anchorNames.isEmpty else { return nil }

        var referencePoints: [SIMD3<Float>] = []
        var movingPoints: [SIMD3<Float>] = []
        referencePoints.reserveCapacity(anchorNames.count)
        movingPoints.reserveCapacity(anchorNames.count)

        for atomName in anchorNames {
            guard let atomIndex = atomIndices[atomName],
                  let atomTemplate = template.atom(named: atomName) else {
                continue
            }
            referencePoints.append(atoms[atomIndex].position)
            movingPoints.append(atomTemplate.idealPosition)
        }

        guard !referencePoints.isEmpty, referencePoints.count == movingPoints.count else {
            return nil
        }

        let referenceCenter = centroid(of: referencePoints)
        let movingCenter = centroid(of: movingPoints)
        let rotation = bestFitRotation(referencePoints: referencePoints, movingPoints: movingPoints)
        return OverlayTransform(rotation: rotation, movingCenter: movingCenter, referenceCenter: referenceCenter)
    }

    private static func centroid(of points: [SIMD3<Float>]) -> SIMD3<Float> {
        points.reduce(SIMD3<Float>.zero, +) / Float(points.count)
    }

    private static func bestFitRotation(
        referencePoints: [SIMD3<Float>],
        movingPoints: [SIMD3<Float>]
    ) -> simd_float3x3 {
        if referencePoints.count <= 1 {
            return matrix_identity_float3x3
        }

        let referenceCenter = centroid(of: referencePoints)
        let movingCenter = centroid(of: movingPoints)
        var covariance = simd_float3x3()

        for (referencePoint, movingPoint) in zip(referencePoints, movingPoints) {
            let x = referencePoint - referenceCenter
            let y = movingPoint - movingCenter
            covariance[0, 0] += y.x * x.x
            covariance[0, 1] += y.x * x.y
            covariance[0, 2] += y.x * x.z
            covariance[1, 0] += y.y * x.x
            covariance[1, 1] += y.y * x.y
            covariance[1, 2] += y.y * x.z
            covariance[2, 0] += y.z * x.x
            covariance[2, 1] += y.z * x.y
            covariance[2, 2] += y.z * x.z
        }

        let mxx = covariance[0, 0]
        let mxy = covariance[0, 1]
        let mxz = covariance[0, 2]
        let myx = covariance[1, 0]
        let myy = covariance[1, 1]
        let myz = covariance[1, 2]
        let mzx = covariance[2, 0]
        let mzy = covariance[2, 1]
        let mzz = covariance[2, 2]
        let trace = mxx + myy + mzz

        let rows: [SIMD4<Float>] = [
            SIMD4(trace, myz - mzy, mzx - mxz, mxy - myx),
            SIMD4(myz - mzy, mxx - myy - mzz, mxy + myx, mzx + mxz),
            SIMD4(mzx - mxz, mxy + myx, -mxx + myy - mzz, myz + mzy),
            SIMD4(mxy - myx, mzx + mxz, myz + mzy, -mxx - myy + mzz)
        ]

        var quaternion = SIMD4<Float>(1, 0, 0, 0)
        for _ in 0..<32 {
            let next = SIMD4<Float>(
                simd_dot(rows[0], quaternion),
                simd_dot(rows[1], quaternion),
                simd_dot(rows[2], quaternion),
                simd_dot(rows[3], quaternion)
            )
            let length = simd_length(next)
            guard length > 1e-8 else {
                return matrix_identity_float3x3
            }
            quaternion = next / length
        }

        return rotationMatrix(from: quaternion)
    }

    private static func rotationMatrix(from quaternion: SIMD4<Float>) -> simd_float3x3 {
        let normalized = quaternion / max(simd_length(quaternion), 1e-8)
        let w = normalized.x
        let x = normalized.y
        let y = normalized.z
        let z = normalized.w

        return simd_float3x3(rows: [
            SIMD3<Float>(
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w)
            ),
            SIMD3<Float>(
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w)
            ),
            SIMD3<Float>(
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y)
            )
        ])
    }

    private static func titratableParentAtoms(for residueName: String) -> Set<String> {
        switch residueName {
        case "ASP":
            return ["OD1", "OD2"]
        case "GLU":
            return ["OE1", "OE2"]
        case "CYS":
            return ["SG"]
        case "TYR":
            return ["OH"]
        case "HIS":
            return ["ND1", "NE2"]
        case "LYS":
            return ["NZ"]
        case "ARG":
            return ["NE", "NH1", "NH2"]
        default:
            return []
        }
    }

    private static func nextTerminalHydrogenName(existingHydrogenNames: Set<String>) -> String? {
        for candidate in ["H2", "H3"] where !existingHydrogenNames.contains(candidate) {
            return candidate
        }
        return nil
    }

    private static func containsBond(
        _ atomIndex1: Int,
        _ atomIndex2: Int,
        bonds: [Bond]
    ) -> Bool {
        bonds.contains {
            ($0.atomIndex1 == atomIndex1 && $0.atomIndex2 == atomIndex2) ||
            ($0.atomIndex1 == atomIndex2 && $0.atomIndex2 == atomIndex1)
        }
    }

    private static func bondedNeighborPositions(
        of atomIndex: Int,
        atoms: [Atom],
        bonds: [Bond]
    ) -> [SIMD3<Float>] {
        bonds.compactMap { bond in
            if bond.atomIndex1 == atomIndex {
                return atoms[bond.atomIndex2].position
            }
            if bond.atomIndex2 == atomIndex {
                return atoms[bond.atomIndex1].position
            }
            return nil
        }
    }

    /// Place a hydrogen using internal coordinate reconstruction (NERF algorithm,
    /// as used by Reduce). Computes bond length, bond angle, and dihedral from
    /// the template ideal geometry, then reconstructs the position using the
    /// actual parent/reference atom positions. This correctly handles arbitrary
    /// sidechain conformations (chi angles) because each H's position is computed
    /// from the actual local geometry rather than a rigid overlay.
    private static func localHydrogenPosition(
        hydrogenTemplate: ProteinResidueReferenceTemplateStore.AtomTemplate,
        parentAtomName: String,
        template: ProteinResidueReferenceTemplateStore.Template,
        atomIndices: [String: Int],
        atoms: [Atom],
        fallbackTransform: OverlayTransform
    ) -> SIMD3<Float> {
        guard let parentIndex = atomIndices[parentAtomName],
              let parentTemplate = template.atom(named: parentAtomName) else {
            let pos = fallbackTransform.apply(hydrogenTemplate.idealPosition)
            guard pos.x.isFinite && pos.y.isFinite && pos.z.isFinite else {
                return SIMD3<Float>(0, 1, 0)  // degenerate transform
            }
            return pos
        }

        let parentPos = atoms[parentIndex].position
        let parentIdeal = parentTemplate.idealPosition
        let hIdeal = hydrogenTemplate.idealPosition
        let bondLength = simd_distance(hIdeal, parentIdeal)

        // Collect all heavy neighbors of parent that exist in the actual structure
        let parentHeavyNeighbors = template.bondedAtomNames(to: parentAtomName)
            .filter { template.atom(named: $0)?.isHydrogen == false }
        let availableNeighbors = parentHeavyNeighbors.filter { atomIndices[$0] != nil }

        guard let angleRefName = availableNeighbors.first,
              let angleRefIndex = atomIndices[angleRefName],
              let angleRefTemplate = template.atom(named: angleRefName) else {
            // No reference — place opposite to whatever heavy neighbors exist
            let neighborPositions = parentHeavyNeighbors.compactMap { name -> SIMD3<Float>? in
                guard let idx = atomIndices[name] else { return nil }
                return atoms[idx].position
            }
            return placeHydrogenLike(on: parentPos, awayFrom: neighborPositions, distance: bondLength)
        }

        let angleRefPos = atoms[angleRefIndex].position
        let angleRefIdeal = angleRefTemplate.idealPosition
        let bondAngle = computeAngle(hIdeal, parentIdeal, angleRefIdeal)

        // Strategy 1: use another parent neighbor as dihedral reference (most reliable —
        // all atoms are in the same residue so they're always available)
        for siblingName in availableNeighbors where siblingName != angleRefName {
            guard let siblingIndex = atomIndices[siblingName],
                  let siblingTemplate = template.atom(named: siblingName) else { continue }
            let torsion = computeDihedral(siblingTemplate.idealPosition, angleRefIdeal, parentIdeal, hIdeal)
            return nerfPlace(
                dihedralRef: atoms[siblingIndex].position,
                angleRef: angleRefPos,
                parent: parentPos,
                bondLength: bondLength,
                bondAngle: bondAngle,
                torsion: torsion
            )
        }

        // Strategy 2: use a neighbor of angleRef (grandparent) as dihedral reference
        let angleRefNeighbors = template.bondedAtomNames(to: angleRefName)
            .filter { $0 != parentAtomName && template.atom(named: $0)?.isHydrogen == false }
        for grandparentName in angleRefNeighbors {
            guard let grandparentIndex = atomIndices[grandparentName],
                  let grandparentTemplate = template.atom(named: grandparentName) else { continue }
            let torsion = computeDihedral(grandparentTemplate.idealPosition, angleRefIdeal, parentIdeal, hIdeal)
            return nerfPlace(
                dihedralRef: atoms[grandparentIndex].position,
                angleRef: angleRefPos,
                parent: parentPos,
                bondLength: bondLength,
                bondAngle: bondAngle,
                torsion: torsion
            )
        }

        // Last resort: correct bond length and angle but arbitrary dihedral
        let prVec = angleRefPos - parentPos
        let prLen = simd_length(prVec)
        guard prLen > 1e-6 else {
            return placeHydrogenLike(on: parentPos, awayFrom: [], distance: bondLength)
        }
        let pr = prVec / prLen
        let perp = arbitraryPerpendicular(to: pr)
        return parentPos + bondLength * (-pr * cosf(bondAngle) + perp * sinf(bondAngle))
    }

    // MARK: - NERF Internal Coordinate Placement

    /// Compute angle at vertex b (in radians).
    private static func computeAngle(
        _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>
    ) -> Float {
        let ba = simd_normalize(a - b)
        let bc = simd_normalize(c - b)
        return acosf(max(-1.0, min(1.0, simd_dot(ba, bc))))
    }

    /// Compute dihedral angle A-B-C-D (in radians).
    private static func computeDihedral(
        _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d: SIMD3<Float>
    ) -> Float {
        let b1 = b - a
        let b2 = c - b
        let b3 = d - c
        let n1 = simd_cross(b1, b2)
        let n2 = simd_cross(b2, b3)
        let b2n = simd_normalize(b2)
        return atan2f(simd_dot(simd_cross(n1, n2), b2n), simd_dot(n1, n2))
    }

    /// NERF algorithm: place atom D given atoms A (dihedral ref), B (angle ref),
    /// C (parent). bondAngle = angle(B, C, D) at vertex C.
    /// torsion = dihedral(A, B, C, D).
    private static func nerfPlace(
        dihedralRef a: SIMD3<Float>,
        angleRef b: SIMD3<Float>,
        parent c: SIMD3<Float>,
        bondLength: Float,
        bondAngle: Float,
        torsion: Float
    ) -> SIMD3<Float> {
        let cbVec = c - b
        let cbLen = simd_length(cbVec)
        guard cbLen > 1e-6 else {
            // Parent and angle reference are coincident — fall back to arbitrary placement
            return c + SIMD3<Float>(0, bondLength, 0)
        }
        let bc = cbVec / cbLen
        let ba = b - a
        let crossBaBc = simd_cross(ba, bc)
        let crossLen = simd_length(crossBaBc)

        // If A, B, C are collinear, pick an arbitrary perpendicular
        let n: SIMD3<Float>
        if crossLen < 1e-6 {
            n = arbitraryPerpendicular(to: bc)
        } else {
            n = crossBaBc / crossLen
        }
        let m = simd_cross(n, bc)

        let dx = -bondLength * cosf(bondAngle)
        let dy = bondLength * sinf(bondAngle) * cosf(torsion)
        let dz = bondLength * sinf(bondAngle) * sinf(torsion)

        let result = c + bc * dx + m * dy + n * dz

        // Guard against NaN/Inf from degenerate geometry
        guard result.x.isFinite && result.y.isFinite && result.z.isFinite else {
            return c + SIMD3<Float>(0, bondLength, 0)
        }
        return result
    }

    /// Return an arbitrary unit vector perpendicular to the given vector.
    private static func arbitraryPerpendicular(to v: SIMD3<Float>) -> SIMD3<Float> {
        let candidate: SIMD3<Float> = abs(v.x) < 0.9
            ? SIMD3<Float>(1, 0, 0)
            : SIMD3<Float>(0, 1, 0)
        return simd_normalize(simd_cross(v, candidate))
    }

    private static func placeHydrogenLike(
        on center: SIMD3<Float>,
        awayFrom neighbors: [SIMD3<Float>],
        distance: Float
    ) -> SIMD3<Float> {
        if neighbors.isEmpty {
            return center + SIMD3<Float>(0, distance, 0)
        }

        let average = neighbors.reduce(SIMD3<Float>.zero) { partial, neighbor in
            partial + (neighbor - center)
        } / Float(neighbors.count)

        let length = simd_length(average)
        if length < 1e-6 {
            return center + SIMD3<Float>(0, distance, 0)
        }

        return center + (-average / length) * distance
    }
}

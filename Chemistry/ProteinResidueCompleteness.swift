import Foundation

struct ProteinResidueCompleteness: Sendable, Hashable {
    let chainID: String
    let residueSeq: Int
    let residueName: String
    let missingHeavyAtoms: [String]
    let missingHydrogens: [String]
    let extraAtoms: [String]

    var isComplete: Bool {
        missingHeavyAtoms.isEmpty && missingHydrogens.isEmpty && extraAtoms.isEmpty
    }

    var summary: String {
        var parts: [String] = []
        if !missingHeavyAtoms.isEmpty {
            parts.append("missing heavy: \(missingHeavyAtoms.joined(separator: ", "))")
        }
        if !missingHydrogens.isEmpty {
            parts.append("missing H: \(missingHydrogens.joined(separator: ", "))")
        }
        if !extraAtoms.isEmpty {
            parts.append("extra: \(extraAtoms.joined(separator: ", "))")
        }
        return parts.joined(separator: "; ")
    }
}

extension ProteinPreparation {
    private struct CompletenessResidueKey: Hashable {
        let chainID: String
        let residueSeq: Int
        let residueName: String
        let isHetAtom: Bool
    }

    static func analyzeResidueCompleteness(atoms: [Atom]) -> [ProteinResidueCompleteness] {
        guard !atoms.isEmpty else { return [] }

        var groupedIndices: [CompletenessResidueKey: [Int]] = [:]
        for index in atoms.indices {
            let atom = atoms[index]
            let key = CompletenessResidueKey(
                chainID: atom.chainID,
                residueSeq: atom.residueSeq,
                residueName: ProteinResidueTemplateStore.normalizeAtomName(atom.residueName),
                isHetAtom: atom.isHetAtom
            )
            groupedIndices[key, default: []].append(index)
        }

        let hasHydrogens = atoms.contains { $0.element == .H }
        let sortedKeys = groupedIndices.keys.sorted { lhs, rhs in
            if lhs.chainID != rhs.chainID {
                return lhs.chainID < rhs.chainID
            }
            if lhs.residueSeq != rhs.residueSeq {
                return lhs.residueSeq < rhs.residueSeq
            }
            if lhs.residueName != rhs.residueName {
                return lhs.residueName < rhs.residueName
            }
            return lhs.isHetAtom && !rhs.isHetAtom
        }

        var completeness: [ProteinResidueCompleteness] = []

        for key in sortedKeys {
            guard let template = ProteinResidueTemplateStore.template(for: key.residueName),
                  let atomIndices = groupedIndices[key] else {
                continue
            }

            var presentHeavyAtoms: Set<String> = []
            var presentHydrogenGroups: [String: Int] = [:]
            var extraHeavyAtoms: Set<String> = []

            for atomIndex in atomIndices {
                let atom = atoms[atomIndex]
                let atomName = ProteinResidueTemplateStore.normalizeAtomName(atom.name)

                if atom.element == .H || atomName.hasPrefix("H") {
                    if let groupName = ProteinResidueTemplateStore.hydrogenGroupName(for: atom.name) {
                        presentHydrogenGroups[groupName, default: 0] += 1
                    }
                    continue
                }

                if template.heavyAtomSet.contains(atomName) || template.optionalAtoms.contains(atomName) {
                    presentHeavyAtoms.insert(atomName)
                } else {
                    extraHeavyAtoms.insert(atomName)
                }
            }

            let missingHeavyAtoms = template.heavyAtoms.filter { !presentHeavyAtoms.contains($0) }
            let missingHydrogens: [String] = if hasHydrogens {
                template.requiredHydrogenGroups.compactMap { group in
                    let presentCount = presentHydrogenGroups[group.baseName, default: 0]
                    guard presentCount < group.count else { return nil }
                    let missingCount = group.count - presentCount
                    return missingCount == 1 ? group.baseName : "\(group.baseName) x\(missingCount)"
                }
            } else {
                []
            }

            let residueCompleteness = ProteinResidueCompleteness(
                chainID: key.chainID,
                residueSeq: key.residueSeq,
                residueName: key.residueName,
                missingHeavyAtoms: missingHeavyAtoms,
                missingHydrogens: missingHydrogens,
                extraAtoms: extraHeavyAtoms.sorted()
            )

            if !residueCompleteness.isComplete {
                completeness.append(residueCompleteness)
            }
        }

        return completeness
    }
}

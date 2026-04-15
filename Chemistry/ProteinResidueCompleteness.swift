// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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

    static func analyzeResidueCompleteness(
        atoms: [Atom],
        bonds: [Bond]? = nil
    ) -> [ProteinResidueCompleteness] {
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
        let externalBondedAtoms = externalBondedAtomNamesByResidue(atoms: atoms, bonds: bonds)
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
            guard let template = ProteinResidueReferenceTemplateStore.template(for: key.residueName),
                  let atomIndices = groupedIndices[key] else {
                continue
            }

            let residueExternalBondedAtoms = externalBondedAtoms[key, default: []]
            let filteredAtomNames = template.filteredAtomNames(
                externalBondedAtoms: residueExternalBondedAtoms,
                includeHydrogens: true
            )
            let expectedHeavyAtoms = filteredAtomNames.filter {
                template.atom(named: $0)?.isHydrogen == false
            }
            let expectedHeavyAtomSet = Set(expectedHeavyAtoms)
            var presentHeavyAtoms: Set<String> = []
            var presentHydrogenGroups: [String: Int] = [:]
            var extraHeavyAtoms: Set<String> = []

            for atomIndex in atomIndices {
                let atom = atoms[atomIndex]
                let atomName = ProteinResidueTemplateStore.normalizeAtomName(atom.name)

                if atom.element == .H || atomName.hasPrefix("H") {
                    let canonicalHydrogenName = template.atom(named: atomName)?.atomName ?? atom.name
                    if let groupName = ProteinResidueTemplateStore.hydrogenGroupName(for: canonicalHydrogenName) {
                        presentHydrogenGroups[groupName, default: 0] += 1
                    }
                    continue
                }

                if let templateAtom = template.atom(named: atomName),
                   !templateAtom.isHydrogen,
                   expectedHeavyAtomSet.contains(templateAtom.atomName) {
                    presentHeavyAtoms.insert(templateAtom.atomName)
                } else {
                    extraHeavyAtoms.insert(atomName)
                }
            }

            let missingHeavyAtoms = expectedHeavyAtoms.filter { !presentHeavyAtoms.contains($0) }
            let missingHydrogens: [String] = if hasHydrogens {
                requiredHydrogenGroupCounts(
                    template: template,
                    residueName: key.residueName,
                    externalBondedAtoms: residueExternalBondedAtoms
                ).compactMap { groupName, requiredCount in
                    let presentCount = presentHydrogenGroups[groupName, default: 0]
                    guard presentCount < requiredCount else { return nil }
                    let missingCount = requiredCount - presentCount
                    return missingCount == 1 ? groupName : "\(groupName) x\(missingCount)"
                }.sorted()
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

    private static func externalBondedAtomNamesByResidue(
        atoms: [Atom],
        bonds: [Bond]? = nil
    ) -> [CompletenessResidueKey: Set<String>] {
        if let bonds, !bonds.isEmpty {
            return externalBondedAtomNamesByResidueFromBonds(atoms: atoms, bonds: bonds)
        }
        return externalBondedAtomNamesByResidueFromSequence(atoms: atoms)
    }

    private static func externalBondedAtomNamesByResidueFromBonds(
        atoms: [Atom],
        bonds: [Bond]
    ) -> [CompletenessResidueKey: Set<String>] {
        var result: [CompletenessResidueKey: Set<String>] = [:]

        for bond in bonds {
            let atom1 = atoms[bond.atomIndex1]
            let atom2 = atoms[bond.atomIndex2]
            let residue1 = completenessResidueKey(for: atom1)
            let residue2 = completenessResidueKey(for: atom2)
            guard residue1 != residue2 else { continue }

            result[residue1, default: []].insert(ProteinResidueTemplateStore.normalizeAtomName(atom1.name))
            result[residue2, default: []].insert(ProteinResidueTemplateStore.normalizeAtomName(atom2.name))
        }

        return result
    }

    private static func externalBondedAtomNamesByResidueFromSequence(
        atoms: [Atom]
    ) -> [CompletenessResidueKey: Set<String>] {
        let residues = Array(Set(atoms.map(completenessResidueKey(for:))))
        let sortedResidues = residues.sorted { lhs, rhs in
            if lhs.chainID != rhs.chainID {
                return lhs.chainID < rhs.chainID
            }
            if lhs.residueSeq != rhs.residueSeq {
                return lhs.residueSeq < rhs.residueSeq
            }
            return lhs.residueName < rhs.residueName
        }

        var result: [CompletenessResidueKey: Set<String>] = [:]

        for pairIndex in 0..<max(0, sortedResidues.count - 1) {
            let lhs = sortedResidues[pairIndex]
            let rhs = sortedResidues[pairIndex + 1]
            guard lhs.chainID == rhs.chainID,
                  lhs.residueSeq + 1 == rhs.residueSeq,
                  canFormPeptideLink(from: lhs, to: rhs) else {
                continue
            }

            result[lhs, default: []].insert("C")
            result[rhs, default: []].insert("N")
        }

        return result
    }

    private static func completenessResidueKey(for atom: Atom) -> CompletenessResidueKey {
        CompletenessResidueKey(
            chainID: atom.chainID,
            residueSeq: atom.residueSeq,
            residueName: ProteinResidueTemplateStore.normalizeAtomName(atom.residueName),
            isHetAtom: atom.isHetAtom
        )
    }

    private static func canFormPeptideLink(
        from lhs: CompletenessResidueKey,
        to rhs: CompletenessResidueKey
    ) -> Bool {
        let lhsName = lhs.residueName
        let rhsName = rhs.residueName
        let lhsIsProtein = isStandardProteinResidue(lhsName)
        let rhsIsProtein = isStandardProteinResidue(rhsName)
        let lhsIsCap = lhsName == "ACE"
        let rhsIsCap = rhsName == "NME"

        return (lhsIsProtein || lhsIsCap) && (rhsIsProtein || rhsIsCap)
    }

    private static func isStandardProteinResidue(_ residueName: String) -> Bool {
        guard ProteinResidueReferenceTemplateStore.template(for: residueName) != nil else {
            return false
        }
        return residueName != "ACE" && residueName != "NME"
    }

    private static func requiredHydrogenGroupCounts(
        template: ProteinResidueReferenceTemplateStore.Template,
        residueName: String,
        externalBondedAtoms: Set<String>
    ) -> [(String, Int)] {
        let freeNTerminus = !externalBondedAtoms.contains("N")
        let freeCTerminus = !externalBondedAtoms.contains("C")
        let titratableParents = directTitratableParentAtoms(for: residueName)

        var groupCounts: [String: Int] = [:]
        for atomName in template.filteredAtomNames(externalBondedAtoms: externalBondedAtoms, includeHydrogens: true) {
            guard let atomTemplate = template.atom(named: atomName),
                  atomTemplate.isHydrogen,
                  let groupName = ProteinResidueTemplateStore.hydrogenGroupName(for: atomName) else {
                continue
            }

            let parentAtomName = template.bondedAtomNames(to: atomName).first(where: { candidate in
                template.atom(named: candidate)?.isHydrogen == false
            })
            guard let parentAtomName else { continue }

            if titratableParents.contains(parentAtomName) {
                continue
            }
            if freeNTerminus && parentAtomName == "N" {
                continue
            }
            if freeCTerminus && (parentAtomName == "O" || parentAtomName == "OXT") {
                continue
            }

            groupCounts[groupName, default: 0] += 1
        }

        return groupCounts.keys.sorted().map { ($0, groupCounts[$0] ?? 0) }
    }

    private static func directTitratableParentAtoms(for residueName: String) -> Set<String> {
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
}

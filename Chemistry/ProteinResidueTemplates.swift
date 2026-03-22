import Foundation

enum ProteinResidueTemplateStore {

    struct HydrogenGroup: Sendable {
        let baseName: String
        let count: Int
    }

    struct Template: Sendable {
        let residueName: String
        let heavyAtoms: [String]
        let requiredHydrogenGroups: [HydrogenGroup]
        let optionalHydrogenGroups: [HydrogenGroup]
        let optionalAtoms: Set<String>

        let heavyAtomSet: Set<String>
    }

    static func template(for residueName: String) -> Template? {
        templates[normalizeAtomName(residueName)]
    }

    static func normalizeAtomName(_ atomName: String) -> String {
        atomName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .uppercased()
    }

    static func hydrogenGroupName(for atomName: String) -> String? {
        let normalized = normalizeAtomName(atomName)
        guard !normalized.isEmpty else { return nil }

        if normalized.first?.isNumber == true {
            let ordinalPrefix = normalized.prefix { $0.isNumber }
            let remainder = normalized.dropFirst(ordinalPrefix.count)
            guard remainder.hasPrefix("H") else { return nil }
            return String(remainder)
        }

        guard normalized.hasPrefix("H") else { return nil }
        if normalized.count > 1, normalized.last?.isNumber == true {
            return String(normalized.dropLast())
        }
        return normalized
    }

    private static func peptideTemplate(
        _ residueName: String,
        heavyAtoms: [String],
        requiredHydrogens: [(String, Int)],
        optionalHydrogens: [(String, Int)] = []
    ) -> Template {
        template(
            residueName,
            heavyAtoms: heavyAtoms,
            requiredHydrogens: requiredHydrogens,
            optionalHydrogens: optionalHydrogens,
            optionalAtoms: ["OXT", "OT1", "OT2"]
        )
    }

    private static func template(
        _ residueName: String,
        heavyAtoms: [String],
        requiredHydrogens: [(String, Int)],
        optionalHydrogens: [(String, Int)] = [],
        optionalAtoms: [String] = []
    ) -> Template {
        let normalizedHeavyAtoms = heavyAtoms.map(normalizeAtomName)
        return Template(
            residueName: normalizeAtomName(residueName),
            heavyAtoms: normalizedHeavyAtoms,
            requiredHydrogenGroups: requiredHydrogens.map {
                HydrogenGroup(baseName: normalizeAtomName($0.0), count: $0.1)
            },
            optionalHydrogenGroups: optionalHydrogens.map {
                HydrogenGroup(baseName: normalizeAtomName($0.0), count: $0.1)
            },
            optionalAtoms: Set(optionalAtoms.map(normalizeAtomName)),
            heavyAtomSet: Set(normalizedHeavyAtoms)
        )
    }

    private static let templates: [String: Template] = {
        let allTemplates: [Template] = [
            template("ACE", heavyAtoms: ["CH3", "C", "O"], requiredHydrogens: [("H", 3)]),
            peptideTemplate("ALA", heavyAtoms: ["N", "CA", "C", "O", "CB"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 3)]),
            peptideTemplate("ARG", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 2), ("HD", 2)], optionalHydrogens: [("HE", 1), ("HH1", 2), ("HH2", 2)]),
            peptideTemplate("ASN", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2)], optionalHydrogens: [("HD2", 2)]),
            peptideTemplate("ASP", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2)], optionalHydrogens: [("HD1", 1), ("HD2", 1)]),
            peptideTemplate("CYS", heavyAtoms: ["N", "CA", "C", "O", "CB", "SG"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2)], optionalHydrogens: [("HG", 1)]),
            peptideTemplate("GLN", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 2)], optionalHydrogens: [("HE2", 2)]),
            peptideTemplate("GLU", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 2)], optionalHydrogens: [("HE1", 1), ("HE2", 1)]),
            peptideTemplate("GLY", heavyAtoms: ["N", "CA", "C", "O"], requiredHydrogens: [("H", 1), ("HA", 2)]),
            peptideTemplate("HIS", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HD2", 1), ("HE1", 1)], optionalHydrogens: [("HD1", 1), ("HE2", 1)]),
            peptideTemplate("ILE", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 1), ("HG1", 2), ("HG2", 3), ("HD1", 3)]),
            peptideTemplate("LEU", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 1), ("HD1", 3), ("HD2", 3)]),
            peptideTemplate("LYS", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 2), ("HD", 2), ("HE", 2)], optionalHydrogens: [("HZ", 3)]),
            peptideTemplate("MET", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HG", 2), ("HE", 3)]),
            template("NME", heavyAtoms: ["N", "CH3"], requiredHydrogens: [("H", 4)]),
            peptideTemplate("PHE", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HD1", 1), ("HD2", 1), ("HE1", 1), ("HE2", 1), ("HZ", 1)]),
            peptideTemplate("PRO", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD"], requiredHydrogens: [("HA", 1), ("HB", 2), ("HG", 2), ("HD", 2)]),
            peptideTemplate("SER", heavyAtoms: ["N", "CA", "C", "O", "CB", "OG"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2)], optionalHydrogens: [("HG", 1)]),
            peptideTemplate("THR", heavyAtoms: ["N", "CA", "C", "O", "CB", "OG1", "CG2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 1), ("HG2", 3)], optionalHydrogens: [("HG1", 1)]),
            peptideTemplate("TRP", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HD1", 1), ("HE3", 1), ("HZ2", 1), ("HZ3", 1), ("HH2", 1)], optionalHydrogens: [("HE1", 1)]),
            peptideTemplate("TYR", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 2), ("HD1", 1), ("HD2", 1), ("HE1", 1), ("HE2", 1)], optionalHydrogens: [("HH", 1)]),
            peptideTemplate("VAL", heavyAtoms: ["N", "CA", "C", "O", "CB", "CG1", "CG2"], requiredHydrogens: [("H", 1), ("HA", 1), ("HB", 1), ("HG1", 3), ("HG2", 3)])
        ]

        return Dictionary(uniqueKeysWithValues: allTemplates.map { ($0.residueName, $0) })
    }()
}

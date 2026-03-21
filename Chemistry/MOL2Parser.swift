import Foundation
import simd

/// Parser for Tripos MOL2 file format (.mol2 files).
/// Supports single and multi-molecule files with @<TRIPOS>MOLECULE, @<TRIPOS>ATOM,
/// @<TRIPOS>BOND, and @<TRIPOS>SUBSTRUCTURE sections.
enum MOL2Parser {

    // MARK: - Public API

    /// Parse a MOL2 file from URL. Returns an array of tuples matching the SDFParser pattern.
    static func parse(url: URL) throws -> [(name: String, atoms: [Atom], bonds: [Bond], title: String)] {
        let data = try Data(contentsOf: url)
        let content = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .ascii)
            ?? ""
        guard !content.isEmpty else {
            throw MOL2Error.emptyFile
        }
        return parse(content)
    }

    /// Parse MOL2 content string. Handles multi-molecule files separated by @<TRIPOS>MOLECULE.
    static func parse(_ content: String) -> [(name: String, atoms: [Atom], bonds: [Bond], title: String)] {
        // Split on @<TRIPOS>MOLECULE to get individual molecule blocks.
        // The first element before the first @<TRIPOS>MOLECULE is header/comments, skip it.
        let blocks = content.components(separatedBy: "@<TRIPOS>MOLECULE")
        var results: [(name: String, atoms: [Atom], bonds: [Bond], title: String)] = []

        for (idx, block) in blocks.enumerated() {
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            // Reconstruct the full block with the section header
            let fullBlock = "@<TRIPOS>MOLECULE" + block

            if let mol = parseMoleculeBlock(fullBlock, index: idx) {
                results.append(mol)
            }
        }

        return results
    }

    // MARK: - Error

    enum MOL2Error: Error, LocalizedError {
        case emptyFile
        case noAtoms
        case malformedSection(String)

        var errorDescription: String? {
            switch self {
            case .emptyFile:
                return "MOL2 file is empty or could not be decoded."
            case .noAtoms:
                return "No atoms found in MOL2 file."
            case .malformedSection(let section):
                return "Malformed @<TRIPOS>\(section) section."
            }
        }
    }

    // MARK: - Single Molecule Block

    private static func parseMoleculeBlock(
        _ block: String,
        index: Int = 0
    ) -> (name: String, atoms: [Atom], bonds: [Bond], title: String)? {
        let sections = splitSections(block)

        // MOLECULE section is required for the name and counts
        guard let moleculeSection = sections["MOLECULE"] else { return nil }

        // ATOM section is required
        guard let atomSection = sections["ATOM"] else { return nil }

        // Parse MOLECULE header lines
        // Line 0: molecule name
        // Line 1: num_atoms [num_bonds [num_subst [num_feat [num_sets]]]]
        // Line 2: molecule type (SMALL, BIOPOLYMER, PROTEIN, etc.)
        // Line 3: charge type (GASTEIGER, NO_CHARGES, etc.)
        // Line 4+: optional comments (used as title)
        let molLines = moleculeSection
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        let moleculeName = molLines.first ?? "Molecule \(index + 1)"

        var moleculeType = ""
        if molLines.count > 2 {
            moleculeType = molLines[2]
        }

        // Comments after the charge type become the title
        var title = ""
        if molLines.count > 4 {
            title = molLines[4...].joined(separator: " ").trimmingCharacters(in: .whitespaces)
        }
        if title.isEmpty {
            title = moleculeType.isEmpty ? "MOL2" : moleculeType
        }

        // Parse ATOM section
        let atomLines = atomSection
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        var atoms: [Atom] = []
        // Map from MOL2 atom_id (1-based) to our 0-based index
        var mol2IdToIndex: [Int: Int] = [:]

        for line in atomLines {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            // Minimum fields: atom_id atom_name x y z atom_type
            guard parts.count >= 6 else { continue }

            guard let mol2AtomId = Int(parts[0]),
                  let x = Float(parts[2]),
                  let y = Float(parts[3]),
                  let z = Float(parts[4])
            else { continue }

            let atomName = parts[1]
            let sybylType = parts[5]
            let element = elementFromSybylType(sybylType)

            // Substructure/residue info (optional columns 6 and 7)
            let residueSeq = parts.count > 6 ? (Int(parts[6]) ?? 1) : 1
            let residueName: String
            if parts.count > 7 {
                // The substructure name can contain residue+seq like "ALA1" or just "LIG"
                residueName = cleanResidueName(parts[7])
            } else {
                residueName = "LIG"
            }

            // Partial charge (column 8)
            let charge: Float = parts.count > 8 ? (Float(parts[8]) ?? 0.0) : 0.0

            // Status bits (column 9) - ignored for now

            let currentIndex = atoms.count
            mol2IdToIndex[mol2AtomId] = currentIndex

            atoms.append(Atom(
                id: currentIndex,
                element: element,
                position: SIMD3<Float>(x, y, z),
                name: atomName,
                residueName: residueName,
                residueSeq: residueSeq,
                chainID: "A",
                charge: charge,
                formalCharge: formalChargeFromSybylType(sybylType),
                isHetAtom: true
            ))
        }

        guard !atoms.isEmpty else { return nil }

        // Parse BOND section (optional)
        var bonds: [Bond] = []
        if let bondSection = sections["BOND"] {
            let bondLines = bondSection
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }

            for line in bondLines {
                let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
                // Minimum: bond_id origin_atom_id target_atom_id bond_type
                guard parts.count >= 4 else { continue }

                guard let originId = Int(parts[1]),
                      let targetId = Int(parts[2])
                else { continue }

                let bondTypeStr = parts[3].lowercased()
                let order = bondOrderFromMOL2Type(bondTypeStr)

                // Resolve 1-based MOL2 atom IDs to 0-based indices via our map.
                // This handles non-contiguous atom IDs correctly.
                guard let idx1 = mol2IdToIndex[originId],
                      let idx2 = mol2IdToIndex[targetId]
                else { continue }

                guard idx1 >= 0, idx1 < atoms.count,
                      idx2 >= 0, idx2 < atoms.count
                else { continue }

                bonds.append(Bond(
                    id: bonds.count,
                    atomIndex1: idx1,
                    atomIndex2: idx2,
                    order: order
                ))
            }
        }

        let finalName = moleculeName.isEmpty ? "Molecule \(index + 1)" : moleculeName

        return (name: finalName, atoms: atoms, bonds: bonds, title: title)
    }

    // MARK: - Section Splitting

    /// Split a MOL2 block into sections keyed by section name (without the @<TRIPOS> prefix).
    private static func splitSections(_ block: String) -> [String: String] {
        var sections: [String: String] = [:]
        var currentKey: String?
        var currentContent: [String] = []

        let prefix = "@<TRIPOS>"

        for line in block.components(separatedBy: .newlines) {
            let trimmedLine = line.trimmingCharacters(in: .init(charactersIn: "\r"))
            if trimmedLine.hasPrefix(prefix) {
                // Flush previous section
                if let key = currentKey {
                    sections[key] = currentContent.joined(separator: "\n")
                }
                let sectionName = String(trimmedLine.dropFirst(prefix.count))
                    .trimmingCharacters(in: .whitespaces)
                currentKey = sectionName
                currentContent = []
            } else {
                currentContent.append(trimmedLine)
            }
        }

        // Flush final section
        if let key = currentKey {
            sections[key] = currentContent.joined(separator: "\n")
        }

        return sections
    }

    // MARK: - Bond Order Mapping

    /// Map MOL2/Tripos bond type strings to BondOrder.
    ///  1  = single
    ///  2  = double
    ///  3  = triple
    ///  ar = aromatic
    ///  am = amide (treated as single with partial double character)
    ///  du = dummy
    ///  un = unknown
    ///  nc = not connected
    private static func bondOrderFromMOL2Type(_ typeStr: String) -> BondOrder {
        switch typeStr {
        case "1", "single":   return .single
        case "2", "double":   return .double
        case "3", "triple":   return .triple
        case "ar", "aromatic": return .aromatic
        case "am", "amide":   return .single   // amide has partial double character, map to single
        case "du", "dummy":   return .single
        case "un", "unknown": return .single
        case "nc":            return .single    // not connected, but keep as single if present
        default:              return .single
        }
    }

    // MARK: - Sybyl Atom Type -> Element

    /// Map Tripos/Sybyl atom types to Element (atomic number).
    ///
    /// Sybyl types use the format "ELEMENT.hybridization", e.g.:
    ///   C.3  = sp3 carbon (atomic number 6)
    ///   C.2  = sp2 carbon
    ///   C.1  = sp carbon
    ///   C.ar = aromatic carbon
    ///   C.cat = carbocation
    ///   N.3  = sp3 nitrogen (atomic number 7)
    ///   N.2  = sp2 nitrogen
    ///   N.1  = sp nitrogen
    ///   N.ar = aromatic nitrogen
    ///   N.am = amide nitrogen
    ///   N.pl3 = trigonal planar nitrogen
    ///   N.4  = sp3 positively charged nitrogen
    ///   O.3  = sp3 oxygen (atomic number 8)
    ///   O.2  = sp2 oxygen
    ///   O.co2 = carboxylate oxygen
    ///   O.spc = SPC water oxygen
    ///   O.t3p = TIP3P water oxygen
    ///   S.3  = sp3 sulfur (atomic number 16)
    ///   S.2  = sp2 sulfur
    ///   S.O  = sulfoxide sulfur
    ///   S.O2 = sulfone sulfur
    ///   P.3  = sp3 phosphorus (atomic number 15)
    ///   H, H.spc, H.t3p = hydrogen
    ///   LP = lone pair (ignored, mapped to H as placeholder)
    ///   Du = dummy atom (mapped to C)
    private static func elementFromSybylType(_ sybylType: String) -> Element {
        // Extract base element symbol before the dot
        let base: String
        if let dotIndex = sybylType.firstIndex(of: ".") {
            base = String(sybylType[sybylType.startIndex..<dotIndex])
        } else {
            base = sybylType
        }

        let upper = base.uppercased()

        switch upper {
        case "H":   return .H
        case "HE":  return .He
        case "LI":  return .Li
        case "BE":  return .Be
        case "B":   return .B
        case "C":   return .C
        case "N":   return .N
        case "O":   return .O
        case "F":   return .F
        case "NE":  return .Ne
        case "NA":  return .Na
        case "MG":  return .Mg
        case "AL":  return .Al
        case "SI":  return .Si
        case "P":   return .P
        case "S":   return .S
        case "CL":  return .Cl
        case "AR":  return .Ar
        case "K":   return .K
        case "CA":  return .Ca
        case "SC":  return .Sc
        case "TI":  return .Ti
        case "V":   return .V
        case "CR":  return .Cr
        case "MN":  return .Mn
        case "FE":  return .Fe
        case "CO":  return .Co
        case "NI":  return .Ni
        case "CU":  return .Cu
        case "ZN":  return .Zn
        case "GA":  return .Ga
        case "GE":  return .Ge
        case "AS":  return .As
        case "SE":  return .Se
        case "BR":  return .Br
        case "KR":  return .Kr
        case "LP":  return .H  // lone pair placeholder
        case "DU":  return .C  // dummy atom
        default:    return .C  // fallback
        }
    }

    // MARK: - Formal Charge from Sybyl Type

    /// Infer formal charge from special Sybyl atom types.
    ///   N.4   → +1 (quaternary/protonated nitrogen)
    ///   C.cat → +1 (carbocation)
    ///   O.co2 → -1 (carboxylate, approximation for one oxygen)
    private static func formalChargeFromSybylType(_ sybylType: String) -> Int {
        let lower = sybylType.lowercased()
        switch lower {
        case "n.4":   return 1   // positively charged sp3 nitrogen
        case "c.cat": return 1   // carbocation
        case "o.co2": return -1  // carboxylate oxygen (approximate: formal -1 per O)
        default:      return 0
        }
    }

    // MARK: - Residue Name Cleanup

    /// Clean residue/substructure name from MOL2.
    /// MOL2 substructure names can look like "ALA1", "LIG1", "****", etc.
    /// Strip trailing digits to get the 3-letter residue code, or return as-is if short.
    private static func cleanResidueName(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return "LIG" }

        // If it looks like a standard residue name (3 letters), return it
        if trimmed.count <= 4 && trimmed.allSatisfy({ $0.isLetter }) {
            return trimmed.uppercased()
        }

        // Strip trailing digits: "ALA1" -> "ALA", "HOH123" -> "HOH"
        var name = trimmed
        while let last = name.last, last.isNumber {
            name.removeLast()
        }
        if name.isEmpty { return trimmed }

        return name.uppercased()
    }
}

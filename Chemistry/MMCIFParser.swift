import Foundation
import simd

/// Parser for mmCIF (PDBx/mmCIF) format files.
/// Wraps the C++ `druse_parse_mmcif` function from druse_core to parse
/// macromolecular structure data into native Swift Atom/Bond arrays.
enum MMCIFParser {

    private struct AtomSiteMetadata {
        var atomName: String?
        var residueName: String?
        var chainID: String?
        var residueSeq: Int?
        var isHetAtom: Bool?
        var occupancy: Float?
        var tempFactor: Float?
        var altLoc: String?
    }

    // MARK: - Error

    enum MMCIFError: Error, LocalizedError {
        case emptyFile
        case parseFailed(String)
        case noAtoms

        var errorDescription: String? {
            switch self {
            case .emptyFile:
                return "mmCIF file is empty or could not be decoded."
            case .parseFailed(let reason):
                return "mmCIF parsing failed: \(reason)"
            case .noAtoms:
                return "No atoms found in mmCIF file."
            }
        }
    }

    // MARK: - Public API

    /// Parse an mmCIF file from a URL.
    /// Returns a single molecule structure with name, atoms, bonds, and title.
    static func parse(url: URL) throws -> (name: String, atoms: [Atom], bonds: [Bond], title: String) {
        let data = try Data(contentsOf: url)
        let content = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .ascii)
            ?? ""
        guard !content.isEmpty else {
            throw MMCIFError.emptyFile
        }
        return try parse(content: content, fileName: url.deletingPathExtension().lastPathComponent)
    }

    /// Parse mmCIF content string via the C++ druse_core bridge.
    static func parse(
        content: String,
        fileName: String = "mmCIF"
    ) throws -> (name: String, atoms: [Atom], bonds: [Bond], title: String) {
        // Call into the C++ druse_core parser
        guard let result = content.withCString({ cStr in
            druse_parse_mmcif(cStr)
        }) else {
            throw MMCIFError.parseFailed("druse_parse_mmcif returned nil")
        }
        defer { druse_free_molecule_result(result) }

        let r = result.pointee

        // Check success flag
        guard r.success else {
            let errorMsg = withUnsafePointer(to: r.errorMessage) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: 512) { cStr in
                    String(cString: cStr)
                }
            }
            throw MMCIFError.parseFailed(errorMsg)
        }

        guard r.atomCount > 0 else {
            throw MMCIFError.noAtoms
        }

        // Convert DruseAtom array to [Atom]
        var atoms: [Atom] = []
        atoms.reserveCapacity(Int(r.atomCount))

        func cString<T>(_ value: T) -> String {
            withUnsafePointer(to: value) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: MemoryLayout<T>.size) {
                    String(cString: $0)
                }
            }
        }

        for i in 0..<Int(r.atomCount) {
            let da = r.atoms[i]
            let element = Element(rawValue: Int(da.atomicNum)) ?? .C
            let atomName = cString(da.name)
            let symbolStr = cString(da.symbol)
            let residueName = cString(da.residueName)
            let chainID = cString(da.chainID)
            let altLoc = cString(da.altLoc)

            atoms.append(Atom(
                id: i,
                element: element,
                position: SIMD3<Float>(da.x, da.y, da.z),
                name: atomName.isEmpty ? symbolStr : atomName,
                residueName: residueName.isEmpty ? "UNK" : residueName,
                residueSeq: da.residueSeq == 0 ? 1 : Int(da.residueSeq),
                chainID: chainID.isEmpty ? "A" : chainID,
                charge: da.charge,
                formalCharge: Int(da.formalCharge),
                isHetAtom: da.isHetAtom,
                occupancy: da.occupancy > 0 ? da.occupancy : 1.0,
                tempFactor: da.tempFactor,
                altLoc: altLoc
            ))
        }

        let metadata = parseAtomSiteMetadata(from: content)
        if !metadata.isEmpty {
            if metadata.count != atoms.count {
                print("[mmCIF] Warning: metadata count (\(metadata.count)) differs from atom count (\(atoms.count)) — partial metadata applied")
            }
            let count = min(metadata.count, atoms.count)
            for i in 0..<count {
                let meta = metadata[i]
                if let atomName = meta.atomName, !atomName.isEmpty {
                    atoms[i].name = atomName
                }
                if let residueName = meta.residueName, !residueName.isEmpty {
                    atoms[i].residueName = residueName
                }
                if let chainID = meta.chainID, !chainID.isEmpty {
                    atoms[i].chainID = chainID
                }
                if let residueSeq = meta.residueSeq {
                    atoms[i].residueSeq = residueSeq
                }
                if let isHetAtom = meta.isHetAtom {
                    atoms[i].isHetAtom = isHetAtom
                }
                if let occupancy = meta.occupancy {
                    atoms[i].occupancy = occupancy
                }
                if let tempFactor = meta.tempFactor {
                    atoms[i].tempFactor = tempFactor
                }
                if let altLoc = meta.altLoc, !altLoc.isEmpty, altLoc != "." && altLoc != "?" {
                    atoms[i].altLoc = altLoc
                }
            }
        }

        // Convert DruseBond array to [Bond]
        var bonds: [Bond] = []
        bonds.reserveCapacity(Int(r.bondCount))

        for i in 0..<Int(r.bondCount) {
            let db = r.bonds[i]
            let order: BondOrder = switch Int(db.order) {
            case 2: .double
            case 3: .triple
            case 4: .aromatic
            default: .single
            }
            bonds.append(Bond(
                id: i,
                atomIndex1: Int(db.atom1),
                atomIndex2: Int(db.atom2),
                order: order
            ))
        }

        // Extract molecule name from the DruseMoleculeResult
        let name = cString(r.name)
        let smiles = cString(r.smiles)

        let moleculeName = (name.isEmpty || name == "mmcif_structure") ? fileName : name
        let title = smiles.isEmpty ? fileName : smiles

        return (name: moleculeName, atoms: atoms, bonds: bonds, title: title)
    }

    private static func parseAtomSiteMetadata(from content: String) -> [AtomSiteMetadata] {
        let lines = content.split(whereSeparator: \.isNewline).map(String.init)
        var columnNames: [String] = []
        var rows: [AtomSiteMetadata] = []
        var inAtomSiteLoop = false
        var readingColumns = false

        var colGroup = -1
        var colAtomID = -1
        var colCompID = -1
        var colAsymID = -1
        var colSeqID = -1
        var colOccupancy = -1
        var colBFactor = -1
        var colAltID = -1

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }

            if line == "loop_" {
                if inAtomSiteLoop && !columnNames.isEmpty && !rows.isEmpty { break }
                inAtomSiteLoop = false
                readingColumns = false
                columnNames.removeAll(keepingCapacity: true)
                continue
            }

            if line.hasPrefix("_atom_site.") {
                if !inAtomSiteLoop {
                    inAtomSiteLoop = true
                    readingColumns = true
                    columnNames.removeAll(keepingCapacity: true)
                    colGroup = -1
                    colAtomID = -1
                    colCompID = -1
                    colAsymID = -1
                    colSeqID = -1
                    colOccupancy = -1
                    colBFactor = -1
                    colAltID = -1
                }

                let columnName = String(line.dropFirst("_atom_site.".count))
                let index = columnNames.count
                columnNames.append(columnName)

                if columnName == "group_PDB" { colGroup = index }
                else if columnName == "label_atom_id" || columnName == "auth_atom_id" {
                    if colAtomID < 0 { colAtomID = index }
                } else if columnName == "label_comp_id" || columnName == "auth_comp_id" {
                    if colCompID < 0 { colCompID = index }
                } else if columnName == "label_asym_id" || columnName == "auth_asym_id" {
                    if colAsymID < 0 { colAsymID = index }
                } else if columnName == "label_seq_id" || columnName == "auth_seq_id" {
                    if colSeqID < 0 { colSeqID = index }
                } else if columnName == "occupancy" {
                    colOccupancy = index
                } else if columnName == "B_iso_or_equiv" {
                    colBFactor = index
                } else if columnName == "label_alt_id" || columnName == "auth_alt_id" {
                    if colAltID < 0 { colAltID = index }
                }
                continue
            }

            if !inAtomSiteLoop { continue }

            if line == "#" {
                if !rows.isEmpty { break }
                continue
            }

            if readingColumns {
                readingColumns = false
            }

            let tokens = tokenizeMMCIFRow(line)
            guard tokens.count >= columnNames.count else { continue }

            var metadata = AtomSiteMetadata()
            if colAtomID >= 0 { metadata.atomName = tokens[colAtomID] }
            if colCompID >= 0 { metadata.residueName = tokens[colCompID] }
            if colAsymID >= 0 { metadata.chainID = tokens[colAsymID] }
            if colSeqID >= 0 {
                let seq = tokens[colSeqID]
                if seq != "." && seq != "?" {
                    metadata.residueSeq = Int(seq)
                }
            }
            if colGroup >= 0 {
                metadata.isHetAtom = tokens[colGroup] == "HETATM"
            }
            if colOccupancy >= 0 {
                metadata.occupancy = Float(tokens[colOccupancy])
            }
            if colBFactor >= 0 {
                metadata.tempFactor = Float(tokens[colBFactor])
            }
            if colAltID >= 0 {
                metadata.altLoc = tokens[colAltID]
            }
            rows.append(metadata)
        }

        return rows
    }

    private static func tokenizeMMCIFRow(_ line: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        var quote: Character?

        for character in line {
            if let activeQuote = quote {
                if character == activeQuote {
                    tokens.append(current)
                    current = ""
                    quote = nil
                } else {
                    current.append(character)
                }
                continue
            }

            if character == "'" || character == "\"" {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                quote = character
            } else if character.isWhitespace {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
            } else {
                current.append(character)
            }
        }

        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }
}

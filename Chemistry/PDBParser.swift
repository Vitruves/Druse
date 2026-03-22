import Foundation
import simd

// MARK: - Parse Result

struct PDBHeader: Sendable {
    var classification: String = ""
    var date: String = ""
    var pdbID: String = ""
    var title: String = ""
    var resolution: Float? = nil
    var experimentMethod: String? = nil
}

struct PDBParseResult: Sendable {
    let protein: MoleculeData?
    let ligands: [MoleculeData]
    let waterCount: Int
    let header: PDBHeader
    let warnings: [String]
}

struct SecondaryStructureRange: Sendable {
    let start: Int
    let end: Int
    let type: SecondaryStructure
    let chain: String
}

/// Sendable molecule data that can cross actor boundaries
struct MoleculeData: Sendable {
    let name: String
    let title: String
    let atoms: [Atom]
    let bonds: [Bond]
    var ssRanges: [SecondaryStructureRange] = []
}

// MARK: - PDB Parser

enum PDBParser {

    static func parse(_ content: String) -> PDBParseResult {
        let metadata = parseHeaderMetadata(from: content)
        do {
            return try parseWithGemmi(content, header: metadata.header, ssRanges: metadata.ssRanges)
        } catch {
            return parseLegacy(content)
        }
    }

    private static func parseLegacy(_ content: String) -> PDBParseResult {
        var header = PDBHeader()
        var proteinAtoms: [Atom] = []
        var hetAtoms: [Atom] = [] // non-water HETATM
        var waterCount = 0
        var warnings: [String] = []

        // CONECT records: serial → [serial]
        var conectMap: [Int: [Int]] = [:]
        // Serial number → atom index mapping (for CONECT)
        var serialToProteinIndex: [Int: Int] = [:]
        var serialToHetIndex: [Int: Int] = [:]

        var allProteinAtoms: [Atom] = []
        var allHetAtoms: [Atom] = []
        var allProteinSerials: [Int] = []
        var allHetSerials: [Int] = []
        var ssRanges: [SecondaryStructureRange] = []

        let lines = content.split(separator: "\n", omittingEmptySubsequences: false)

        for line in lines {
            let s = String(line)
            guard s.count >= 6 else { continue }
            let record = substr(s, 0, 6).trimmingCharacters(in: .whitespaces)

            switch record {
            case "HEADER":
                header.classification = substr(s, 10, 40).trimmingCharacters(in: .whitespaces)
                header.date = substr(s, 50, 9).trimmingCharacters(in: .whitespaces)
                header.pdbID = substr(s, 62, 4).trimmingCharacters(in: .whitespaces)

            case "TITLE":
                let text = substr(s, 10, 70).trimmingCharacters(in: .whitespaces)
                header.title = header.title.isEmpty ? text : header.title + " " + text

            case "REMARK":
                parseRemark(s, header: &header)

            case "ATOM", "HETATM":
                guard let parsed = parseAtomLine(s, warnings: &warnings) else { continue }
                let (atom, serial) = parsed

                if record == "HETATM" {
                    let isWater = atom.residueName == "HOH" || atom.residueName == "WAT" || atom.residueName == "DOD"
                    if isWater {
                        waterCount += 1
                        // Keep waters in protein atoms for user-controlled removal
                        allProteinAtoms.append(atom)
                        allProteinSerials.append(serial)
                    } else {
                        allHetAtoms.append(atom)
                        allHetSerials.append(serial)
                    }
                } else {
                    allProteinAtoms.append(atom)
                    allProteinSerials.append(serial)
                }

            case "HELIX":
                // HELIX record: chain at col 19, start resSeq 21-25, end resSeq 33-37
                let chainID = substr(s, 19, 1).trimmingCharacters(in: .whitespaces)
                if let startSeq = Int(substr(s, 21, 4).trimmingCharacters(in: .whitespaces)),
                   let endSeq = Int(substr(s, 33, 4).trimmingCharacters(in: .whitespaces)) {
                    ssRanges.append(SecondaryStructureRange(start: startSeq, end: endSeq, type: .helix, chain: chainID))
                }

            case "SHEET":
                // SHEET record: chain at col 21, start resSeq 22-25, end resSeq 33-36
                let chainID = substr(s, 21, 1).trimmingCharacters(in: .whitespaces)
                if let startSeq = Int(substr(s, 22, 4).trimmingCharacters(in: .whitespaces)),
                   let endSeq = Int(substr(s, 33, 4).trimmingCharacters(in: .whitespaces)) {
                    ssRanges.append(SecondaryStructureRange(start: startSeq, end: endSeq, type: .sheet, chain: chainID))
                }

            case "CONECT":
                parseConect(s, conectMap: &conectMap)

            default:
                break
            }
        }

        // Resolve alt-confs for protein atoms
        proteinAtoms = resolveAltConfs(allProteinAtoms, serials: allProteinSerials,
                                        serialMap: &serialToProteinIndex)

        // Resolve alt-confs for het atoms
        hetAtoms = resolveAltConfs(allHetAtoms, serials: allHetSerials,
                                   serialMap: &serialToHetIndex)

        // Reassign sequential IDs
        for i in 0..<proteinAtoms.count {
            proteinAtoms[i] = withID(proteinAtoms[i], id: i)
        }
        for i in 0..<hetAtoms.count {
            hetAtoms[i] = withID(hetAtoms[i], id: i)
        }

        // Perceive protein bonds from distances
        let proteinBonds = BondPerception.perceiveBonds(in: proteinAtoms)

        // Build ligand bonds from CONECT records
        let hetBonds = buildHetBonds(hetAtoms: hetAtoms, conectMap: conectMap,
                                      serialMap: serialToHetIndex)

        // Group het atoms into separate ligands by (chainID, residueName, residueSeq)
        let ligandGroups = groupLigands(hetAtoms: hetAtoms, hetBonds: hetBonds)

        // Build results
        let proteinData: MoleculeData? = proteinAtoms.isEmpty ? nil : MoleculeData(
            name: header.pdbID.isEmpty ? "Protein" : header.pdbID,
            title: header.title,
            atoms: proteinAtoms,
            bonds: proteinBonds,
            ssRanges: ssRanges
        )

        return PDBParseResult(
            protein: proteinData,
            ligands: ligandGroups,
            waterCount: waterCount,
            header: header,
            warnings: warnings
        )
    }

    static func parse(url: URL) throws -> PDBParseResult {
        let content = try String(contentsOf: url, encoding: .utf8)
        return parse(content)
    }

    private static func parseWithGemmi(
        _ content: String,
        header: PDBHeader,
        ssRanges: [SecondaryStructureRange]
    ) throws -> PDBParseResult {
        let structure = try GemmiBridge.parseStructure(
            content: content,
            fileName: header.pdbID.isEmpty ? "Protein" : header.pdbID
        )
        let resolved = ProteinPreparation.selectPreferredAltConformers(
            atoms: structure.atoms,
            bonds: structure.bonds
        )

        let proteinSubstructure = buildSubstructure(atoms: resolved.atoms, bonds: resolved.bonds) { atom in
            !atom.isHetAtom || isWaterResidueName(atom.residueName)
        }
        let proteinBonds = proteinSubstructure.bonds.isEmpty
            ? BondPerception.perceiveBonds(in: proteinSubstructure.atoms)
            : proteinSubstructure.bonds

        let heterogenSubstructure = buildSubstructure(atoms: resolved.atoms, bonds: resolved.bonds) { atom in
            atom.isHetAtom && !isWaterResidueName(atom.residueName)
        }
        let heterogenBonds = heterogenSubstructure.bonds.isEmpty
            ? BondPerception.perceiveBonds(in: heterogenSubstructure.atoms)
            : heterogenSubstructure.bonds

        let ligands = groupLigands(hetAtoms: heterogenSubstructure.atoms, hetBonds: heterogenBonds)
        let proteinName = header.pdbID.isEmpty ? structure.name : header.pdbID
        let proteinData = proteinSubstructure.atoms.isEmpty ? nil : MoleculeData(
            name: proteinName.isEmpty ? "Protein" : proteinName,
            title: header.title,
            atoms: proteinSubstructure.atoms,
            bonds: proteinBonds,
            ssRanges: ssRanges
        )

        var warnings: [String] = []
        if resolved.removedAtomCount > 0 {
            warnings.append("Selected preferred alternate conformations (\(resolved.removedAtomCount) atoms removed)")
        }

        return PDBParseResult(
            protein: proteinData,
            ligands: ligands,
            waterCount: countWaterResidues(in: resolved.atoms),
            header: header,
            warnings: warnings
        )
    }

    private static func parseHeaderMetadata(from content: String) -> (header: PDBHeader, ssRanges: [SecondaryStructureRange]) {
        var header = PDBHeader()
        var ssRanges: [SecondaryStructureRange] = []

        for rawLine in content.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            guard line.count >= 6 else { continue }
            let record = substr(line, 0, 6).trimmingCharacters(in: .whitespaces)

            switch record {
            case "HEADER":
                header.classification = substr(line, 10, 40).trimmingCharacters(in: .whitespaces)
                header.date = substr(line, 50, 9).trimmingCharacters(in: .whitespaces)
                header.pdbID = substr(line, 62, 4).trimmingCharacters(in: .whitespaces)

            case "TITLE":
                let text = substr(line, 10, 70).trimmingCharacters(in: .whitespaces)
                header.title = header.title.isEmpty ? text : header.title + " " + text

            case "REMARK":
                parseRemark(line, header: &header)

            case "HELIX":
                let chainID = substr(line, 19, 1).trimmingCharacters(in: .whitespaces)
                if let startSeq = Int(substr(line, 21, 4).trimmingCharacters(in: .whitespaces)),
                   let endSeq = Int(substr(line, 33, 4).trimmingCharacters(in: .whitespaces)) {
                    ssRanges.append(.init(start: startSeq, end: endSeq, type: .helix, chain: chainID))
                }

            case "SHEET":
                let chainID = substr(line, 21, 1).trimmingCharacters(in: .whitespaces)
                if let startSeq = Int(substr(line, 22, 4).trimmingCharacters(in: .whitespaces)),
                   let endSeq = Int(substr(line, 33, 4).trimmingCharacters(in: .whitespaces)) {
                    ssRanges.append(.init(start: startSeq, end: endSeq, type: .sheet, chain: chainID))
                }

            default:
                break
            }
        }

        return (header, ssRanges)
    }

    private static func buildSubstructure(
        atoms: [Atom],
        bonds: [Bond],
        include: (Atom) -> Bool
    ) -> (atoms: [Atom], bonds: [Bond]) {
        let selectedIndices = atoms.indices.filter { include(atoms[$0]) }
        return ProteinPreparation.remapSubstructure(
            atoms: atoms,
            bonds: bonds,
            selectedIndices: selectedIndices
        )
    }

    private static func isWaterResidueName(_ residueName: String) -> Bool {
        ["HOH", "WAT", "DOD", "H2O"].contains(residueName.uppercased())
    }

    private static func countWaterResidues(in atoms: [Atom]) -> Int {
        struct WaterKey: Hashable {
            let chainID: String
            let residueSeq: Int
            let residueName: String
        }

        return Set(atoms.compactMap { atom -> WaterKey? in
            guard isWaterResidueName(atom.residueName) else { return nil }
            return WaterKey(chainID: atom.chainID, residueSeq: atom.residueSeq, residueName: atom.residueName)
        }).count
    }

    // MARK: - ATOM/HETATM Line Parsing

    private static func parseAtomLine(_ line: String, warnings: inout [String]) -> (Atom, Int)? {
        let record = substr(line, 0, 6).trimmingCharacters(in: .whitespaces)
        let isHet = record == "HETATM"

        guard let serial = Int(substr(line, 6, 5).trimmingCharacters(in: .whitespaces)) else { return nil }

        let atomName = substr(line, 12, 4).trimmingCharacters(in: .whitespaces)
        let altLoc = substr(line, 16, 1).trimmingCharacters(in: .whitespaces)
        let resName = substr(line, 17, 3).trimmingCharacters(in: .whitespaces)
        let chainID = substr(line, 21, 1)
        let resSeqStr = substr(line, 22, 4).trimmingCharacters(in: .whitespaces)

        guard let x = Float(substr(line, 30, 8).trimmingCharacters(in: .whitespaces)),
              let y = Float(substr(line, 38, 8).trimmingCharacters(in: .whitespaces)),
              let z = Float(substr(line, 46, 8).trimmingCharacters(in: .whitespaces))
        else {
            warnings.append("Bad coordinates at serial \(serial)")
            return nil
        }

        let occupancy = Float(substr(line, 54, 6).trimmingCharacters(in: .whitespaces)) ?? 1.0
        let tempFactor = Float(substr(line, 60, 6).trimmingCharacters(in: .whitespaces)) ?? 0.0

        // Element detection: prefer columns 77-78, fallback to atom name
        let element = detectElement(line: line, atomName: atomName, resName: resName, isHet: isHet)

        let resSeq = Int(resSeqStr) ?? 0
        let charge = parseCharge(substr(line, 78, 2))

        let atom = Atom(
            id: 0, // will be reassigned later
            element: element,
            position: SIMD3<Float>(x, y, z),
            name: atomName,
            residueName: resName,
            residueSeq: resSeq,
            chainID: chainID.isEmpty ? "A" : chainID,
            charge: 0,
            formalCharge: charge,
            isHetAtom: isHet,
            occupancy: occupancy,
            tempFactor: tempFactor,
            altLoc: altLoc
        )

        return (atom, serial)
    }

    // MARK: - Element Detection

    private static func detectElement(line: String, atomName: String, resName: String, isHet: Bool) -> Element {
        // Try columns 77-78 first
        let elemCol = substr(line, 76, 2).trimmingCharacters(in: .whitespaces)
        if !elemCol.isEmpty, let elem = Element.from(symbol: elemCol) {
            return elem
        }

        // Fallback: infer from atom name
        return elementFromAtomName(atomName, resName: resName, isHet: isHet)
    }

    private static func elementFromAtomName(_ name: String, resName: String, isHet: Bool) -> Element {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return .C }

        // Known metal ions
        let metalIons: Set<String> = ["CA", "ZN", "FE", "MG", "MN", "CU", "CO", "NI", "NA", "K"]
        if isHet && metalIons.contains(trimmed.uppercased()) {
            // Check if it's a single-residue HETATM (ion)
            let ionResidues: Set<String> = ["CA", "ZN", "FE", "MG", "MN", "CU", "CO", "NI", "NA", "K"]
            if ionResidues.contains(resName.uppercased()) {
                if let elem = Element.from(symbol: trimmed) { return elem }
            }
        }

        // Strip leading digits and spaces
        var stripped = ""
        for ch in trimmed {
            if ch.isLetter { stripped.append(ch) }
        }
        guard !stripped.isEmpty else { return .C }

        // Try two-character symbol first, then one-character
        if stripped.count >= 2 {
            let twoChar = String(stripped.prefix(2))
            if let elem = Element.from(symbol: twoChar) { return elem }
        }
        if let elem = Element.from(symbol: String(stripped.prefix(1))) { return elem }

        return .C // fallback
    }

    private static func parseCharge(_ s: String) -> Int {
        let trimmed = s.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return 0 }
        // PDB charge format: "2+" or "1-"
        if trimmed.hasSuffix("+") {
            return Int(String(trimmed.dropLast())) ?? 0
        } else if trimmed.hasSuffix("-") {
            return -(Int(String(trimmed.dropLast())) ?? 0)
        }
        return 0
    }

    // MARK: - REMARK Parsing

    private static func parseRemark(_ line: String, header: inout PDBHeader) {
        let remarkNum = substr(line, 7, 3).trimmingCharacters(in: .whitespaces)
        if remarkNum == "2" {
            // RESOLUTION
            let text = substr(line, 10, 70)
            if text.contains("RESOLUTION") {
                let parts = text.split(separator: " ")
                for (i, part) in parts.enumerated() {
                    if part == "RESOLUTION" || part == "RESOLUTION.", i + 1 < parts.count {
                        if let res = Float(String(parts[i + 1])) {
                            header.resolution = res
                        }
                    }
                }
            }
        } else if remarkNum == "200" {
            let text = substr(line, 10, 70)
            if text.contains("EXPERIMENT TYPE") {
                let parts = text.split(separator: ":")
                if parts.count > 1 {
                    header.experimentMethod = parts.last?.trimmingCharacters(in: .whitespaces)
                }
            }
        }
    }

    // MARK: - CONECT Records

    private static func parseConect(_ line: String, conectMap: inout [Int: [Int]]) {
        guard let source = Int(substr(line, 6, 5).trimmingCharacters(in: .whitespaces)) else { return }
        let offsets = [11, 16, 21, 26]
        for offset in offsets {
            let s = substr(line, offset, 5).trimmingCharacters(in: .whitespaces)
            if let target = Int(s), target > 0 {
                conectMap[source, default: []].append(target)
            }
        }
    }

    // MARK: - Alt-Conf Resolution

    private static func resolveAltConfs(
        _ atoms: [Atom],
        serials: [Int],
        serialMap: inout [Int: Int]
    ) -> [Atom] {
        // Key: (chainID, resSeq, atomName) → best (occupancy, source index)
        var bestMap: [String: (occ: Float, srcIdx: Int)] = [:]
        var keepIndices: [Int] = []

        for (i, atom) in atoms.enumerated() {
            if atom.altLoc.isEmpty {
                keepIndices.append(i)
                continue
            }

            let key = "\(atom.chainID):\(atom.residueSeq):\(atom.name)"
            if let existing = bestMap[key] {
                if atom.occupancy > existing.occ {
                    bestMap[key] = (atom.occupancy, i)
                }
            } else {
                bestMap[key] = (atom.occupancy, i)
            }
        }

        // Collect best alt-conf indices
        let altConfIndices = Set(bestMap.values.map(\.srcIdx))
        for (i, atom) in atoms.enumerated() {
            if !atom.altLoc.isEmpty && altConfIndices.contains(i) {
                keepIndices.append(i)
            }
        }

        keepIndices.sort()

        // Build result with serial mapping
        var result: [Atom] = []
        for srcIdx in keepIndices {
            let newIdx = result.count
            serialMap[serials[srcIdx]] = newIdx
            result.append(atoms[srcIdx])
        }

        return result
    }

    // MARK: - Het Bond Construction

    private static func buildHetBonds(
        hetAtoms: [Atom],
        conectMap: [Int: [Int]],
        serialMap: [Int: Int]
    ) -> [Bond] {
        guard !conectMap.isEmpty else {
            // No CONECT records: fall back to distance-based
            return BondPerception.perceiveBonds(in: hetAtoms)
        }

        var bonds: [Bond] = []
        var seen: Set<String> = []
        var bondID = 0

        for (srcSerial, targets) in conectMap {
            guard let srcIdx = serialMap[srcSerial] else { continue }
            for tgtSerial in targets {
                guard let tgtIdx = serialMap[tgtSerial] else { continue }
                let key = srcIdx < tgtIdx ? "\(srcIdx)-\(tgtIdx)" : "\(tgtIdx)-\(srcIdx)"
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                bonds.append(Bond(id: bondID, atomIndex1: srcIdx, atomIndex2: tgtIdx))
                bondID += 1
            }
        }

        // If CONECT produced nothing useful, fallback
        if bonds.isEmpty {
            return BondPerception.perceiveBonds(in: hetAtoms)
        }

        return bonds
    }

    // MARK: - Ligand Grouping

    private static func groupLigands(hetAtoms: [Atom], hetBonds: [Bond]) -> [MoleculeData] {
        // Group by (chainID, residueName, residueSeq)
        struct LigandKey: Hashable {
            let chainID: String
            let resName: String
            let resSeq: Int
        }

        var groups: [LigandKey: [Int]] = [:]
        var groupOrder: [LigandKey] = []

        for (i, atom) in hetAtoms.enumerated() {
            let key = LigandKey(chainID: atom.chainID, resName: atom.residueName, resSeq: atom.residueSeq)
            if groups[key] == nil { groupOrder.append(key) }
            groups[key, default: []].append(i)
        }

        var results: [MoleculeData] = []

        for key in groupOrder {
            guard let indices = groups[key] else { continue }
            let indexSet = Set(indices)

            // Remap atom indices
            var oldToNew: [Int: Int] = [:]
            var newAtoms: [Atom] = []
            for (newIdx, oldIdx) in indices.enumerated() {
                oldToNew[oldIdx] = newIdx
                var atom = hetAtoms[oldIdx]
                atom = withID(atom, id: newIdx)
                newAtoms.append(atom)
            }

            // Remap bonds
            var newBonds: [Bond] = []
            for bond in hetBonds {
                if indexSet.contains(bond.atomIndex1) && indexSet.contains(bond.atomIndex2),
                   let a = oldToNew[bond.atomIndex1], let b = oldToNew[bond.atomIndex2] {
                    newBonds.append(Bond(id: newBonds.count, atomIndex1: a, atomIndex2: b, order: bond.order))
                }
            }

            let template = ChemicalComponentStore.template(for: key.resName)
            let repaired = repairLigandChemistry(
                template: template,
                atoms: newAtoms,
                fallbackBonds: newBonds
            )

            results.append(MoleculeData(
                name: key.resName,
                title: template?.canonicalSmiles ?? "\(key.resName) chain \(key.chainID)",
                atoms: repaired.atoms,
                bonds: repaired.bonds
            ))
        }

        return results
    }

    // MARK: - Helpers

    private static func withID(_ atom: Atom, id: Int) -> Atom {
        Atom(
            id: id,
            element: atom.element,
            position: atom.position,
            name: atom.name,
            residueName: atom.residueName,
            residueSeq: atom.residueSeq,
            chainID: atom.chainID,
            charge: atom.charge,
            formalCharge: atom.formalCharge,
            isHetAtom: atom.isHetAtom,
            occupancy: atom.occupancy,
            tempFactor: atom.tempFactor,
            altLoc: atom.altLoc
        )
    }

    private static func repairLigandChemistry(
        template: ChemicalComponentTemplate?,
        atoms: [Atom],
        fallbackBonds: [Bond]
    ) -> (atoms: [Atom], bonds: [Bond]) {
        guard let template else {
            return (atoms, fallbackBonds)
        }

        var repairedAtoms = atoms
        var atomNameToIndex: [String: Int] = [:]
        var matchedAtoms = 0

        for (index, atom) in repairedAtoms.enumerated() {
            let atomName = atom.name.trimmingCharacters(in: .whitespaces)
            atomNameToIndex[atomName] = index
            guard let templateAtom = template.atomsByName[atomName] else { continue }
            repairedAtoms[index].element = templateAtom.element
            repairedAtoms[index].formalCharge = templateAtom.formalCharge
            matchedAtoms += 1
        }

        var repairedBonds: [Bond] = []
        repairedBonds.reserveCapacity(template.bonds.count)
        for bond in template.bonds {
            guard let atomIndex1 = atomNameToIndex[bond.atomID1],
                  let atomIndex2 = atomNameToIndex[bond.atomID2] else {
                continue
            }
            repairedBonds.append(Bond(
                id: repairedBonds.count,
                atomIndex1: atomIndex1,
                atomIndex2: atomIndex2,
                order: bond.order
            ))
        }

        let atomCoverageOK = matchedAtoms >= max(repairedAtoms.count * 2 / 3, 4)
        let bondCoverageOK = !repairedBonds.isEmpty && (
            fallbackBonds.isEmpty || repairedBonds.count >= max(fallbackBonds.count / 2, 1)
        )

        guard atomCoverageOK, bondCoverageOK else {
            return (atoms, fallbackBonds)
        }

        return (repairedAtoms, repairedBonds)
    }

    /// Safe substring extraction by character offset and length.
    /// Pads with spaces if the line is too short.
    private static func substr(_ s: String, _ start: Int, _ length: Int) -> String {
        let chars = Array(s)
        let end = min(start + length, chars.count)
        guard start < chars.count else { return String(repeating: " ", count: length) }
        return String(chars[start..<end])
    }
}

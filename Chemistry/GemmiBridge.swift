import Foundation
import simd

enum GemmiBridge {

    struct StructureData: Sendable {
        let name: String
        let atoms: [Atom]
        let bonds: [Bond]
    }

    struct ResidueTopology: Sendable {
        struct AtomTemplate: Sendable {
            let atomName: String
            let element: Element
            let formalCharge: Int
            let isHydrogen: Bool
        }

        struct BondTemplate: Sendable {
            let atomName1: String
            let atomName2: String
            let order: BondOrder
            let idealLength: Float
        }

        struct AngleTemplate: Sendable {
            let atomName1: String
            let atomName2: String
            let atomName3: String
            let idealAngleDegrees: Float
        }

        let residueName: String
        let atoms: [AtomTemplate]
        let bonds: [BondTemplate]
        let angles: [AngleTemplate]
    }

    enum GemmiError: Error, LocalizedError {
        case parseFailed(String)
        case topologyFailed(String)

        var errorDescription: String? {
            switch self {
            case .parseFailed(let reason):
                return "gemmi structure parsing failed: \(reason)"
            case .topologyFailed(let reason):
                return "gemmi topology parsing failed: \(reason)"
            }
        }
    }

    static func parseStructure(content: String, fileName: String = "structure") throws -> StructureData {
        guard let result = content.withCString({ druse_parse_structure($0) }) else {
            throw GemmiError.parseFailed("druse_parse_structure returned nil")
        }
        defer { druse_free_molecule_result(result) }

        let parsed = result.pointee
        guard parsed.success else {
            throw GemmiError.parseFailed(fixedCString(parsed.errorMessage))
        }

        var atoms: [Atom] = []
        atoms.reserveCapacity(Int(parsed.atomCount))
        for index in 0..<Int(parsed.atomCount) {
            let atom = parsed.atoms[index]
            let element = Element(rawValue: Int(atom.atomicNum)) ?? .C
            let symbol = fixedCString(atom.symbol)
            let atomName = fixedCString(atom.name)
            let residueName = fixedCString(atom.residueName)
            let chainID = fixedCString(atom.chainID)
            let altLoc = fixedCString(atom.altLoc)
            atoms.append(Atom(
                id: index,
                element: element,
                position: SIMD3<Float>(atom.x, atom.y, atom.z),
                name: atomName.isEmpty ? symbol : atomName,
                residueName: residueName.isEmpty ? "UNK" : residueName,
                residueSeq: atom.residueSeq == 0 ? 1 : Int(atom.residueSeq),
                chainID: chainID.isEmpty ? "A" : chainID,
                charge: atom.charge,
                formalCharge: Int(atom.formalCharge),
                isHetAtom: atom.isHetAtom,
                occupancy: atom.occupancy > 0 ? atom.occupancy : 1.0,
                tempFactor: atom.tempFactor,
                altLoc: altLoc
            ))
        }

        var bonds: [Bond] = []
        bonds.reserveCapacity(Int(parsed.bondCount))
        for index in 0..<Int(parsed.bondCount) {
            let bond = parsed.bonds[index]
            let order: BondOrder = switch Int(bond.order) {
            case 2: .double
            case 3: .triple
            case 4: .aromatic
            default: .single
            }
            bonds.append(Bond(
                id: index,
                atomIndex1: Int(bond.atom1),
                atomIndex2: Int(bond.atom2),
                order: order
            ))
        }

        let parsedName = fixedCString(parsed.name)
        let name = parsedName.isEmpty ? fileName : parsedName
        return StructureData(name: name, atoms: atoms, bonds: bonds)
    }

    static func parseChemCompCIF(_ cifContent: String) throws -> ResidueTopology {
        guard let result = cifContent.withCString({ druse_parse_chemcomp_cif($0) }) else {
            throw GemmiError.topologyFailed("druse_parse_chemcomp_cif returned nil")
        }
        defer { druse_free_residue_topology_result(result) }

        let parsed = result.pointee
        guard parsed.success else {
            throw GemmiError.topologyFailed(fixedCString(parsed.errorMessage))
        }

        var atoms: [ResidueTopology.AtomTemplate] = []
        atoms.reserveCapacity(Int(parsed.atomCount))
        for index in 0..<Int(parsed.atomCount) {
            let atom = parsed.atoms[index]
            atoms.append(.init(
                atomName: fixedCString(atom.atomName),
                element: Element(rawValue: Int(atom.atomicNum)) ?? .C,
                formalCharge: Int(atom.formalCharge),
                isHydrogen: atom.isHydrogen
            ))
        }

        var bonds: [ResidueTopology.BondTemplate] = []
        bonds.reserveCapacity(Int(parsed.bondCount))
        for index in 0..<Int(parsed.bondCount) {
            let bond = parsed.bonds[index]
            let order: BondOrder = switch Int(bond.order) {
            case 2: .double
            case 3: .triple
            case 4: .aromatic
            default: .single
            }
            bonds.append(.init(
                atomName1: fixedCString(bond.atom1),
                atomName2: fixedCString(bond.atom2),
                order: order,
                idealLength: bond.idealLength
            ))
        }

        var angles: [ResidueTopology.AngleTemplate] = []
        angles.reserveCapacity(Int(parsed.angleCount))
        for index in 0..<Int(parsed.angleCount) {
            let angle = parsed.angles[index]
            angles.append(.init(
                atomName1: fixedCString(angle.atom1),
                atomName2: fixedCString(angle.atom2),
                atomName3: fixedCString(angle.atom3),
                idealAngleDegrees: angle.idealAngleDegrees
            ))
        }

        return ResidueTopology(
            residueName: fixedCString(parsed.residueName),
            atoms: atoms,
            bonds: bonds,
            angles: angles
        )
    }

    static func neighborIndices(
        content: String,
        queryPoint: SIMD3<Float>,
        radius: Float,
        includeHydrogens: Bool = false,
        maxResults: Int = 256
    ) -> [Int] {
        guard radius > 0, maxResults > 0 else { return [] }

        var query = [queryPoint.x, queryPoint.y, queryPoint.z]
        var indices = [Int32](repeating: -1, count: maxResults)
        let count: Int32 = content.withCString { cString -> Int32 in
            query.withUnsafeMutableBufferPointer { queryBuffer -> Int32 in
                indices.withUnsafeMutableBufferPointer { indexBuffer -> Int32 in
                    guard let queryPointer = queryBuffer.baseAddress,
                          let indexPointer = indexBuffer.baseAddress else {
                        return -1
                    }
                    return druse_find_structure_neighbors(
                        cString,
                        queryPointer,
                        radius,
                        includeHydrogens,
                        indexPointer,
                        Int32(maxResults)
                    )
                }
            }
        }

        guard count > 0 else { return [] }
        return indices.prefix(Int(count)).map(Int.init)
    }

    private static func fixedCString<T>(_ value: T) -> String {
        withUnsafePointer(to: value) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: MemoryLayout<T>.size) { cString in
                String(cString: cString)
            }
        }
    }
}

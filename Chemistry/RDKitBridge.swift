import Foundation
import simd

/// Swift wrapper for the C druse_core RDKit bridge.
enum RDKitBridge {

    /// Convert SMILES to a 3D Molecule with MMFF94 minimization.
    static func smilesToMolecule(smiles: String, name: String = "", numConformers: Int = 50, minimize: Bool = true) -> (molecule: MoleculeData?, error: String?) {
        guard let result = druse_smiles_to_3d_conformers(smiles, name, Int32(numConformers), minimize) else {
            return (nil, "RDKit call returned nil")
        }
        defer { druse_free_molecule_result(result) }

        guard result.pointee.success else {
            let errMsg = withUnsafePointer(to: result.pointee.errorMessage) {
                $0.withMemoryRebound(to: CChar.self, capacity: 512) { String(cString: $0) }
            }
            return (nil, errMsg)
        }

        return (convertResult(result.pointee), nil)
    }

    /// Full ligand preparation: sanitize → addHs → 3D → MMFF → Gasteiger
    static func prepareLigand(
        smiles: String,
        name: String = "",
        numConformers: Int = 50,
        addHydrogens: Bool = true,
        minimize: Bool = true,
        computeCharges: Bool = true
    ) -> (molecule: MoleculeData?, descriptors: LigandDescriptors?, error: String?) {
        guard let result = druse_prepare_ligand(smiles, name, Int32(numConformers), addHydrogens, minimize, computeCharges) else {
            return (nil, nil, "RDKit call returned nil")
        }
        defer { druse_free_molecule_result(result) }

        guard result.pointee.success else {
            let errBuf = withUnsafePointer(to: result.pointee.errorMessage) {
                $0.withMemoryRebound(to: CChar.self, capacity: 512) { String(cString: $0) }
            }
            return (nil, nil, errBuf)
        }

        let mol = convertResult(result.pointee)

        // Compute descriptors separately for full property set
        let desc = computeDescriptors(smiles: smiles)

        return (mol, desc, nil)
    }

    /// Compute molecular descriptors from SMILES.
    static func computeDescriptors(smiles: String) -> LigandDescriptors? {
        let desc = druse_compute_descriptors(smiles)
        return LigandDescriptors(
            molecularWeight: desc.molecularWeight,
            exactMW: desc.exactMW,
            logP: desc.logP,
            tpsa: desc.tpsa,
            hbd: Int(desc.hbd),
            hba: Int(desc.hba),
            rotatableBonds: Int(desc.rotatableBonds),
            rings: Int(desc.rings),
            aromaticRings: Int(desc.aromaticRings),
            heavyAtomCount: Int(desc.heavyAtomCount),
            fractionCSP3: desc.fractionCSP3,
            lipinski: desc.lipinski,
            veber: desc.veber
        )
    }

    /// Batch process SMILES strings. Uses the TBB-backed C++ path when possible.
    /// Pass a `cancelFlag` pointer to allow early termination of the TBB parallel loop.
    static func batchProcess(
        entries: [(smiles: String, name: String)],
        addHydrogens: Bool = true,
        minimize: Bool = true,
        computeCharges: Bool = true,
        parallel: Bool = true,
        cancelFlag: UnsafePointer<Int32>? = nil
    ) -> [(molecule: MoleculeData?, error: String?)] {
        guard !entries.isEmpty else { return [] }

        if parallel, entries.count > 1, let results = batchProcessParallel(
            entries: entries,
            addHydrogens: addHydrogens,
            minimize: minimize,
            computeCharges: computeCharges,
            cancelFlag: cancelFlag
        ) {
            return results
        }

        return entries.map { entry in
            let (mol, _, err) = prepareLigand(smiles: entry.smiles, name: entry.name,
                                               addHydrogens: addHydrogens, minimize: minimize,
                                               computeCharges: computeCharges)
            return (mol, err)
        }
    }

    /// Fire-and-forget log from nonisolated context.
    private static func logAsync(_ message: String, level: LogLevel = .info, category: LogCategory = .system) {
        Task { @MainActor in ActivityLog.shared.log(message, level: level, category: category) }
    }

    /// Add hydrogens to a protein using RDKit's PDB parser (with 3D placement).
    static func addHydrogensToPDB(pdbContent: String) -> MoleculeData? {
        guard let result = druse_add_hydrogens_pdb(pdbContent) else {
            logAsync("[RDKit] addHydrogensToPDB: C++ call returned nil", level: .warning, category: .prep)
            return nil
        }
        defer { druse_free_molecule_result(result) }
        guard result.pointee.success, result.pointee.atomCount > 0 else {
            logAsync("[RDKit] addHydrogensToPDB: failed or empty result", level: .warning, category: .prep)
            return nil
        }
        return convertResult(result.pointee)
    }

    /// Compute Gasteiger charges on protein atoms via RDKit's PDB parser.
    static func computeChargesPDB(pdbContent: String) -> MoleculeData? {
        guard let result = druse_compute_charges_pdb(pdbContent) else {
            logAsync("[RDKit] computeChargesPDB: C++ call returned nil", level: .warning, category: .prep)
            return nil
        }
        defer { druse_free_molecule_result(result) }
        guard result.pointee.success, result.pointee.atomCount > 0 else {
            logAsync("[RDKit] computeChargesPDB: failed or empty result", level: .warning, category: .prep)
            return nil
        }
        return convertResult(result.pointee)
    }

    /// Compute Gasteiger charges on a ligand represented as an MDL mol block.
    static func computeChargesMolBlock(_ molBlock: String) -> MoleculeData? {
        guard let result = druse_compute_charges_molblock(molBlock) else {
            logAsync("[RDKit] computeChargesMolBlock: C++ call returned nil", level: .debug, category: .smiles)
            return nil
        }
        defer { druse_free_molecule_result(result) }
        guard result.pointee.success, result.pointee.atomCount > 0 else {
            logAsync("[RDKit] computeChargesMolBlock: failed or empty result", level: .debug, category: .smiles)
            return nil
        }
        return convertResult(result.pointee)
    }

    /// Convert an MDL mol block to canonical SMILES using RDKit's native parser.
    /// Handles both 2D and 3D mol blocks correctly.
    static func smilesFromMolBlock(_ molBlock: String) -> String? {
        guard let cStr = druse_molblock_to_smiles(molBlock) else {
            logAsync("[RDKit] smilesFromMolBlock: conversion failed", level: .debug, category: .smiles)
            return nil
        }
        let smiles = String(cString: cStr)
        druse_free_string(cStr)
        return smiles.isEmpty ? nil : smiles
    }

    /// Convert atoms+bonds (with 3D coordinates) to canonical SMILES.
    /// Used for co-crystallized ligands extracted from PDB files.
    static func atomsBondsToSMILES(atoms: [Atom], bonds: [Bond]) -> String? {
        guard atoms.count >= 2 else { return nil }

        // Validate bond indices before passing to C++ (prevents segfault on malformed data)
        let validBonds = bonds.filter {
            $0.atomIndex1 >= 0 && $0.atomIndex1 < atoms.count &&
            $0.atomIndex2 >= 0 && $0.atomIndex2 < atoms.count &&
            $0.atomIndex1 != $0.atomIndex2
        }

        var druseAtoms = [DruseAtom](repeating: DruseAtom(), count: atoms.count)
        for i in 0..<atoms.count {
            druseAtoms[i].x = atoms[i].position.x
            druseAtoms[i].y = atoms[i].position.y
            druseAtoms[i].z = atoms[i].position.z
            druseAtoms[i].atomicNum = Int32(atoms[i].element.rawValue)
            druseAtoms[i].formalCharge = Int32(atoms[i].formalCharge)
            druseAtoms[i].charge = atoms[i].charge
            let sym = atoms[i].element.symbol
            for (j, c) in sym.utf8.prefix(3).enumerated() {
                withUnsafeMutableBytes(of: &druseAtoms[i].symbol) { buf in
                    buf[j] = c
                }
            }
        }

        var druseBonds = validBonds.map { bond in
            var db = DruseBond()
            db.atom1 = Int32(bond.atomIndex1)
            db.atom2 = Int32(bond.atomIndex2)
            db.order = Int32(bond.order.rawValue)
            return db
        }
        let bondCount = Int32(druseBonds.count)

        let result: UnsafeMutablePointer<DruseMoleculeResult>?
        if druseBonds.isEmpty {
            result = druseAtoms.withUnsafeMutableBufferPointer { atomsBuf in
                druse_atoms_bonds_to_smiles(
                    atomsBuf.baseAddress,
                    Int32(atoms.count),
                    nil,
                    0,
                    "ligand"
                )
            }
        } else {
            result = druseBonds.withUnsafeMutableBufferPointer { bondsBuf in
                druseAtoms.withUnsafeMutableBufferPointer { atomsBuf in
                    druse_atoms_bonds_to_smiles(
                        atomsBuf.baseAddress,
                        Int32(atoms.count),
                        bondsBuf.baseAddress,
                        bondCount,
                        "ligand"
                    )
                }
            }
        }

        guard let result else { return nil }
        defer { druse_free_molecule_result(result) }
        guard result.pointee.success else { return nil }

        let smiles = fixedCString(result.pointee.smiles)
        return smiles.isEmpty ? nil : smiles
    }

    /// Compute upstream Vina XS atom types for a ligand represented as an MDL mol block.
    static func computeVinaTypesMolBlock(_ molBlock: String, atomCount: Int) -> [Int32]? {
        guard atomCount > 0 else { return [] }
        var types = [Int32](repeating: -1, count: atomCount)
        let assignedCount = types.withUnsafeMutableBufferPointer { buffer -> Int32 in
            guard let baseAddress = buffer.baseAddress else { return -1 }
            return druse_compute_vina_types_molblock(molBlock, baseAddress, Int32(atomCount))
        }
        guard assignedCount == atomCount else { return nil }
        return types
    }

    /// Get RDKit version string.
    static var rdkitVersion: String {
        guard let v = druse_rdkit_version() else { return "unknown" }
        return String(cString: v)
    }

    /// Compute Morgan fingerprint from SMILES.
    /// Returns array of 0.0/1.0 floats, or empty array on error.
    static func morganFingerprint(smiles: String, radius: Int = 2, nBits: Int = 2048) -> [Float] {
        guard let fp = druse_morgan_fingerprint(smiles, Int32(radius), Int32(nBits)) else { return [] }
        defer { druse_free_fingerprint(fp) }

        let count = Int(fp.pointee.numBits)
        guard let bits = fp.pointee.bits else { return [] }
        return Array(UnsafeBufferPointer(start: bits, count: count))
    }

    /// Generate all conformers sorted by energy. Returns array of (molecule, energy).
    static func generateConformers(smiles: String, name: String = "", count: Int = 50, minimize: Bool = true) -> [(molecule: MoleculeData, energy: Double)] {
        guard let set = druse_generate_conformers(smiles, name, Int32(count), minimize) else { return [] }
        defer { druse_free_conformer_set(set) }

        var results: [(MoleculeData, Double)] = []
        let n = Int(set.pointee.count)
        for i in 0..<n {
            guard let conf = set.pointee.conformers?[i], conf.pointee.success else { continue }
            let mol = convertResult(conf.pointee)
            let energy = set.pointee.energies?[i] ?? 0
            results.append((mol, energy))
        }
        return results
    }

    // MARK: - Tautomer & Protomer Enumeration

    /// Result of tautomer/protomer enumeration from C++ core.
    struct VariantResult: Sendable {
        let molecule: MoleculeData
        let smiles: String
        let score: Double
        let kind: VariantKind
        let label: String
    }

    /// Enumerate tautomers for a SMILES string.
    /// Returns array sorted by MMFF energy (lowest first).
    static func enumerateTautomers(
        smiles: String,
        name: String = "",
        maxTautomers: Int = 25,
        energyCutoff: Double = 10.0
    ) -> [VariantResult] {
        guard let set = druse_enumerate_tautomers(smiles, name, Int32(maxTautomers), energyCutoff) else { return [] }
        defer { druse_free_variant_set(set) }
        return convertVariantSet(set)
    }

    /// Enumerate protomers (protonation states) at a target pH.
    /// Returns array sorted by MMFF energy (lowest first).
    static func enumerateProtomers(
        smiles: String,
        name: String = "",
        maxProtomers: Int = 16,
        pH: Double = 7.4,
        pkaThreshold: Double = 2.0
    ) -> [VariantResult] {
        guard let set = druse_enumerate_protomers(smiles, name, Int32(maxProtomers), pH, pkaThreshold) else { return [] }
        defer { druse_free_variant_set(set) }
        return convertVariantSet(set)
    }

    /// Convert DruseVariantSet to Swift array.
    private static func convertVariantSet(_ set: UnsafeMutablePointer<DruseVariantSet>) -> [VariantResult] {
        var results: [VariantResult] = []
        let n = Int(set.pointee.count)
        for i in 0..<n {
            guard let variant = set.pointee.variants?[i], variant.pointee.success else { continue }
            let mol = convertResult(variant.pointee)
            let smiles = fixedCString(variant.pointee.smiles)
            let score = set.pointee.scores?[i] ?? 0
            let info = set.pointee.infos?[i]
            let kind = VariantKind(rawValue: Int(info?.kind ?? 0)) ?? .tautomer
            let label = info.map { fixedCString($0.label) } ?? ""
            results.append(VariantResult(molecule: mol, smiles: smiles, score: score, kind: kind, label: label))
        }
        return results
    }

    /// Build torsion tree from SMILES. Returns rotatable bond definitions.
    static func buildTorsionTree(smiles: String) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])]? {
        guard let tree = druse_build_torsion_tree(smiles) else { return nil }
        return convertTorsionTree(tree)
    }

    /// Build torsion tree from a mol block while preserving that atom order.
    static func buildTorsionTreeMolBlock(_ molBlock: String) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])]? {
        guard let tree = druse_build_torsion_tree_molblock(molBlock) else { return nil }
        return convertTorsionTree(tree)
    }

    private static func convertTorsionTree(_ tree: UnsafeMutablePointer<DruseTorsionTree>) -> [(atom1: Int, atom2: Int, movingAtoms: [Int])] {
        defer { druse_free_torsion_tree(tree) }

        let edgeCount = Int(tree.pointee.edgeCount)
        guard edgeCount > 0, let edges = tree.pointee.edges else { return [] }

        var result: [(Int, Int, [Int])] = []
        for i in 0..<edgeCount {
            let edge = edges[i]
            let start = Int(edge.movingStart)
            let count = Int(edge.movingCount)
            var moving: [Int] = []
            if let indices = tree.pointee.movingAtomIndices {
                for j in start..<(start + count) {
                    moving.append(Int(indices[j]))
                }
            }
            result.append((Int(edge.atom1), Int(edge.atom2), moving))
        }
        return result
    }

    // MARK: - MMFF94 Strain Energy

    /// Compute MMFF94 reference energy for a free (relaxed) ligand conformation.
    /// Returns energy in kcal/mol, or nil on failure.
    static func mmffReferenceEnergy(smiles: String) -> Double? {
        let e = druse_mmff_reference_energy(smiles)
        return e.isNaN ? nil : e
    }

    /// Compute MMFF94 energy of a docked ligand pose with given heavy atom positions.
    /// Heavy atoms are placed at the given coordinates; only hydrogens are relaxed.
    /// Returns energy in kcal/mol, or nil on failure.
    static func mmffStrainEnergy(smiles: String, heavyPositions: [SIMD3<Float>]) -> Double? {
        var interleaved = [Float](repeating: 0, count: heavyPositions.count * 3)
        for (i, pos) in heavyPositions.enumerated() {
            interleaved[i * 3]     = pos.x
            interleaved[i * 3 + 1] = pos.y
            interleaved[i * 3 + 2] = pos.z
        }
        let e = druse_mmff_strain_energy(smiles, interleaved, Int32(heavyPositions.count))
        return e.isNaN ? nil : e
    }

    // MARK: - 2D Coordinates

    /// Result of 2D depiction coordinate generation for a ligand.
    struct Coords2D: Sendable {
        let positions: [CGPoint]     // one per heavy atom
        let atomicNums: [Int]        // atomic number per heavy atom
        let bonds: [(Int, Int, Int)] // (atom1, atom2, order)
    }

    /// Compute 2D depiction coordinates from SMILES (heavy atoms only).
    static func compute2DCoords(smiles: String) -> Coords2D? {
        guard let result = druse_compute_2d_coords(smiles) else { return nil }
        defer { druse_free_2d_result(result) }

        let n = Int(result.pointee.atomCount)
        guard n > 0, let coords = result.pointee.coords, let nums = result.pointee.atomicNums else { return nil }

        var positions: [CGPoint] = []
        positions.reserveCapacity(n)
        var atomicNums: [Int] = []
        atomicNums.reserveCapacity(n)

        for i in 0..<n {
            positions.append(CGPoint(x: CGFloat(coords[i * 2]), y: CGFloat(coords[i * 2 + 1])))
            atomicNums.append(Int(nums[i]))
        }

        var bonds: [(Int, Int, Int)] = []
        let nb = Int(result.pointee.bondCount)
        if let bondPtr = result.pointee.bonds {
            bonds.reserveCapacity(nb)
            for i in 0..<nb {
                let b = bondPtr[i]
                bonds.append((Int(b.atom1), Int(b.atom2), Int(b.order)))
            }
        }

        return Coords2D(positions: positions, atomicNums: atomicNums, bonds: bonds)
    }

    // MARK: - Internal

    private static func convertResult(_ r: DruseMoleculeResult) -> MoleculeData {
        var atoms: [Atom] = []
        var bonds: [Bond] = []

        for i in 0..<Int(r.atomCount) {
            let da = r.atoms[i]
            let elem = Element(rawValue: Int(da.atomicNum)) ?? .C
            let symbol = fixedCString(da.symbol)
            let atomName = fixedCString(da.name)
            let residueNameRaw = fixedCString(da.residueName)
            let chainIDRaw = fixedCString(da.chainID)
            let altLoc = fixedCString(da.altLoc)
            let hasResidueMetadata = !residueNameRaw.isEmpty || !chainIDRaw.isEmpty || da.residueSeq != 0
            atoms.append(Atom(
                id: i,
                element: elem,
                position: SIMD3<Float>(da.x, da.y, da.z),
                name: atomName.isEmpty ? symbol : atomName,
                residueName: residueNameRaw.isEmpty ? "LIG" : residueNameRaw,
                residueSeq: da.residueSeq == 0 ? 1 : Int(da.residueSeq),
                chainID: chainIDRaw.isEmpty ? (hasResidueMetadata ? "A" : "L") : chainIDRaw,
                charge: da.charge,
                formalCharge: Int(da.formalCharge),
                isHetAtom: hasResidueMetadata ? da.isHetAtom : true,
                occupancy: da.occupancy > 0 ? da.occupancy : 1.0,
                tempFactor: da.tempFactor,
                altLoc: altLoc
            ))
        }

        for i in 0..<Int(r.bondCount) {
            let db = r.bonds[i]
            let a1 = Int(db.atom1)
            let a2 = Int(db.atom2)
            guard a1 >= 0, a1 < atoms.count, a2 >= 0, a2 < atoms.count else { continue }
            let order: BondOrder = switch Int(db.order) {
            case 2: .double
            case 3: .triple
            case 4: .aromatic
            default: .single
            }
            bonds.append(Bond(id: bonds.count, atomIndex1: a1, atomIndex2: a2, order: order))
        }

        let name = fixedCString(r.name)
        let smiles = fixedCString(r.smiles)

        return MoleculeData(name: name.isEmpty ? smiles.prefix(30).description : name,
                            title: smiles, atoms: atoms, bonds: bonds)
    }

    private static func batchProcessParallel(
        entries: [(smiles: String, name: String)],
        addHydrogens: Bool,
        minimize: Bool,
        computeCharges: Bool,
        cancelFlag: UnsafePointer<Int32>? = nil
    ) -> [(molecule: MoleculeData?, error: String?)]? {
        let smilesStorage = entries.map { strdup($0.smiles) }
        let nameStorage = entries.map { strdup($0.name) }
        var smilesPointers = smilesStorage.map { $0.map { UnsafePointer<CChar>($0) } }
        var namePointers = nameStorage.map { $0.map { UnsafePointer<CChar>($0) } }
        defer {
            smilesStorage.forEach { free($0) }
            nameStorage.forEach { free($0) }
        }

        return smilesPointers.withUnsafeMutableBufferPointer { smilesBuf in
            namePointers.withUnsafeMutableBufferPointer { namesBuf in
                guard let smilesPtr = smilesBuf.baseAddress,
                      let namesPtr = namesBuf.baseAddress,
                      let results = druse_batch_process_parallel(
                        smilesPtr,
                        namesPtr,
                        Int32(entries.count),
                        addHydrogens,
                        minimize,
                        computeCharges,
                        cancelFlag
                      )
                else {
                    return nil
                }
                defer { druse_free_batch_results(results, Int32(entries.count)) }

                var converted: [(molecule: MoleculeData?, error: String?)] = []
                converted.reserveCapacity(entries.count)

                for i in 0..<entries.count {
                    guard let result = results[i] else {
                        converted.append((nil, "RDKit batch result was nil"))
                        continue
                    }
                    if result.pointee.success {
                        converted.append((convertResult(result.pointee), nil))
                    } else {
                        converted.append((nil, errorMessage(from: result.pointee)))
                    }
                }

                return converted
            }
        }
    }

    private static func fixedCString<T>(_ value: T) -> String {
        withUnsafePointer(to: value) { ptr in
            ptr.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout<T>.size) { bytes in
                let maxLen = MemoryLayout<T>.size
                var length = 0
                while length < maxLen && bytes[length] != 0 {
                    length += 1
                }
                guard length > 0 else { return "" }
                return String(bytes: UnsafeBufferPointer(start: bytes, count: length), encoding: .utf8)
                    ?? String(bytes: UnsafeBufferPointer(start: bytes, count: length), encoding: .ascii)
                    ?? ""
            }
        }
    }

    private static func errorMessage(from result: DruseMoleculeResult) -> String {
        fixedCString(result.errorMessage)
    }
}

// MARK: - Ligand Descriptors

struct LigandDescriptors: Sendable {
    let molecularWeight: Float
    let exactMW: Float
    let logP: Float
    let tpsa: Float
    let hbd: Int
    let hba: Int
    let rotatableBonds: Int
    let rings: Int
    let aromaticRings: Int
    let heavyAtomCount: Int
    let fractionCSP3: Float
    let lipinski: Bool
    let veber: Bool
}

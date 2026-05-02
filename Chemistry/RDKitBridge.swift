// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import simd

/// Swift wrapper for the C druse_core RDKit bridge.
enum RDKitBridge {

    /// RDKit isn't fully thread-safe (shared aromaticity/depiction state, palette
    /// initialization). Calls that may run concurrently from detached tasks (e.g.
    /// the ligand-detail SVG preview firing back-to-back) must funnel through this
    /// serial queue to avoid heap corruption / EXC_BAD_ACCESS.
    static let serialQueue = DispatchQueue(label: "com.druse.rdkit.serial", qos: .userInitiated)

    /// A dedicated background thread for RDKit drawing / depiction work.
    /// We run depiction on a real `Thread` (8 MB stack) instead of the Swift
    /// concurrency cooperative pool because RDKit's coord generation can recurse
    /// deeply on complex inputs and overflow the small Swift-task stack — that's
    /// the deterministic EXC_BAD_ACCESS we hit when the ligand database opens
    /// alongside the detail panel's SVG preview.
    private static let depictionWorker = RDKitDepictionWorker()

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
    ) -> (molecule: MoleculeData?, descriptors: LigandDescriptors?, canonicalSMILES: String?, error: String?) {
        guard let result = druse_prepare_ligand(smiles, name, Int32(numConformers), addHydrogens, minimize, computeCharges) else {
            return (nil, nil, nil, "RDKit call returned nil")
        }
        defer { druse_free_molecule_result(result) }

        guard result.pointee.success else {
            let errBuf = withUnsafePointer(to: result.pointee.errorMessage) {
                $0.withMemoryRebound(to: CChar.self, capacity: 512) { String(cString: $0) }
            }
            return (nil, nil, nil, errBuf)
        }

        let mol = convertResult(result.pointee)
        let canonSMILES = fixedCString(result.pointee.smiles)

        // Compute descriptors separately for full property set
        let desc = computeDescriptors(smiles: canonSMILES.isEmpty ? smiles : canonSMILES)

        return (mol, desc, canonSMILES.isEmpty ? nil : canonSMILES, nil)
    }

    /// Compute molecular descriptors from SMILES.
    /// Returns `nil` when the SMILES is unparseable or sanitizes to an empty/zero-mass
    /// molecule — RDKit's C bridge fills the struct with zeros in that case rather than
    /// reporting failure, so callers (e.g. analog generation) need this to filter out
    /// chemically-broken candidates that string substitution can produce.
    static func computeDescriptors(smiles: String) -> LigandDescriptors? {
        let trimmed = smiles.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let desc = druse_compute_descriptors(trimmed)
        guard desc.heavyAtomCount > 0, desc.molecularWeight > 0 else { return nil }
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
            let (mol, _, _, err) = prepareLigand(smiles: entry.smiles, name: entry.name,
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

    /// Tanimoto similarity between two SMILES (Morgan radius=2, 2048 bits).
    /// Returns 0 if either molecule is invalid.
    static func tanimotoSimilarity(_ smilesA: String, _ smilesB: String) -> Float {
        return druse_tanimoto_similarity(smilesA, smilesB)
    }

    /// Test whether `smiles` contains the SMARTS `pattern` as a substructure.
    static func containsSubstructure(smiles: String, smarts: String) -> Bool {
        guard let res = druse_match_scaffold(smiles, smarts) else { return false }
        defer { druse_free_scaffold_match(res) }
        return res.pointee.hasMatch
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

    // MARK: - Unified Ensemble Preparation

    /// One member of a prepared ligand ensemble (protomer × tautomer × conformer).
    struct EnsembleMember: Sendable, Identifiable {
        let id: Int                    // index in the ensemble
        let molecule: MoleculeData     // fully prepared 3D structure (H, charges, minimized)
        let smiles: String             // canonical SMILES for this chemical form
        let mmffEnergy: Double         // MMFF94 energy (kcal/mol)
        let boltzmannWeight: Double    // population fraction (0-1)
        let kind: Int                  // 0=parent, 1=tautomer, 2=protomer, 3=taut+prot
        let label: String              // e.g. "Taut2_ProtAmineH_Conf3"
        let formIndex: Int             // which chemical form this conformer belongs to
        let conformerIndex: Int        // conformer rank within its form

        var kindLabel: String {
            switch kind {
            case 0: return "Parent"
            case 1: return "Tautomer"
            case 2: return "Protomer"
            case 3: return "Taut+Prot"
            default: return "Unknown"
            }
        }

        var populationPercent: Double { boltzmannWeight * 100.0 }
    }

    /// Result of unified ensemble preparation.
    struct EnsembleResult: Sendable {
        let members: [EnsembleMember]
        let numForms: Int              // distinct chemical forms (before conformer expansion)
        let conformersPerForm: Int
        let success: Bool
        let errorMessage: String
    }

    /// Unified ligand preparation pipeline: protomers × tautomers × conformers.
    ///
    /// Produces a chemically realistic batch of molecular forms at the target pH.
    /// Each member is fully prepared: polar H, MMFF94 minimized, Gasteiger charges.
    /// Boltzmann weights reflect population fractions based on MMFF94 energetics.
    ///
    /// - Parameters:
    ///   - smiles: Input SMILES
    ///   - name: Molecule name
    ///   - pH: Target pH (default 7.4)
    ///   - pkaThreshold: Ambiguity window for Henderson-Hasselbalch (default 2.0)
    ///   - maxTautomers: Max tautomers per protomer (default 10)
    ///   - maxProtomers: Max protomers from parent (default 8)
    ///   - energyCutoff: Discard forms with E > E_best + cutoff kcal/mol (default 15)
    ///   - conformersPerForm: 3D conformers per chemical form (default 5)
    ///   - temperature: Boltzmann temperature in K (default 298.15)
    static func prepareEnsemble(
        smiles: String,
        name: String = "",
        pH: Double = 7.4,
        pkaThreshold: Double = 2.0,
        maxTautomers: Int = 10,
        maxProtomers: Int = 8,
        energyCutoff: Double = 15.0,
        conformersPerForm: Int = 5,
        temperature: Double = 298.15
    ) -> EnsembleResult {
        guard let cResult = druse_prepare_ligand_ensemble(
            smiles, name, pH, pkaThreshold,
            Int32(maxTautomers), Int32(maxProtomers),
            energyCutoff, Int32(conformersPerForm), temperature
        ) else {
            return EnsembleResult(members: [], numForms: 0, conformersPerForm: conformersPerForm,
                                  success: false, errorMessage: "Ensemble preparation returned nil")
        }
        defer { druse_free_ensemble_result(cResult) }

        guard cResult.pointee.success else {
            let msg = fixedCString(cResult.pointee.errorMessage)
            return EnsembleResult(members: [], numForms: 0, conformersPerForm: conformersPerForm,
                                  success: false, errorMessage: msg)
        }

        var members: [EnsembleMember] = []
        let count = Int(cResult.pointee.count)
        for i in 0..<count {
            let m = cResult.pointee.members[i]
            guard let molPtr = m.molecule, molPtr.pointee.success else { continue }
            let mol = convertResult(molPtr.pointee)
            members.append(EnsembleMember(
                id: i,
                molecule: mol,
                smiles: fixedCString(m.smiles),
                mmffEnergy: m.mmffEnergy,
                boltzmannWeight: m.boltzmannWeight,
                kind: Int(m.kind),
                label: fixedCString(m.label),
                formIndex: Int(m.formIndex),
                conformerIndex: Int(m.conformerIndex)
            ))
        }

        return EnsembleResult(
            members: members,
            numForms: Int(cResult.pointee.numForms),
            conformersPerForm: Int(cResult.pointee.numConformersPerForm),
            success: true,
            errorMessage: ""
        )
    }

    /// Convert an EnsembleResult into ChemicalForm array (the new hierarchical model).
    ///
    /// Groups EnsembleMember items by formIndex, creates one ChemicalForm per distinct form,
    /// and populates conformer arrays sorted by energy.
    static func ensembleResultToForms(_ result: EnsembleResult) -> [ChemicalForm] {
        guard !result.members.isEmpty else { return [] }

        // Group members by formIndex
        let grouped = Dictionary(grouping: result.members, by: { $0.formIndex })

        // Build one ChemicalForm per distinct form, sorted by best energy
        var forms: [ChemicalForm] = []
        for (_, members) in grouped.sorted(by: { a, b in
            let aMin = a.value.map(\.mmffEnergy).filter { !$0.isNaN }.min() ?? .infinity
            let bMin = b.value.map(\.mmffEnergy).filter { !$0.isNaN }.min() ?? .infinity
            return aMin < bMin
        }) {
            guard let rep = members.first else { continue }

            // Sort conformers by energy within this form
            let sortedMembers = members.sorted(by: { $0.mmffEnergy < $1.mmffEnergy })

            let conformers = sortedMembers.enumerated().map { idx, member in
                Conformer3D(
                    id: idx,
                    atoms: member.molecule.atoms,
                    bonds: member.molecule.bonds,
                    energy: member.mmffEnergy
                )
            }

            let kind = ChemicalFormKind(rawValue: rep.kind) ?? .parent

            // Sum Boltzmann weights across conformers of this form
            let totalWeight = members.reduce(0.0) { $0 + $1.boltzmannWeight }

            forms.append(ChemicalForm(
                smiles: rep.smiles,
                kind: kind,
                label: rep.label.replacingOccurrences(of: "_Conf\\d+$", with: "",
                    options: .regularExpression), // strip conformer suffix from label
                boltzmannWeight: totalWeight,
                relativeEnergy: 0, // will be set below
                conformers: conformers
            ))
        }

        // Compute relative energies based on best conformer of each form
        if let bestE = forms.first?.conformers.first?.energy {
            for i in forms.indices {
                let formBestE = forms[i].conformers.first?.energy ?? bestE
                forms[i].relativeEnergy = formBestE - bestE
            }
        }

        // Normalize Boltzmann weights to sum to 1.0
        let weightSum = forms.reduce(0.0) { $0 + $1.boltzmannWeight }
        if weightSum > 0 {
            for i in forms.indices {
                forms[i].boltzmannWeight /= weightSum
            }
        }

        return forms
    }

    // MARK: - Ionizable Site Detection (for GFN2-xTB pKa Prediction)

    /// A detected ionizable site in a molecule.
    struct IonizableSite: Sendable {
        let atomIdx: Int
        let isAcid: Bool          // true = loses H (deprotonates), false = gains H (protonates)
        let defaultPKa: Double    // from the built-in lookup table
        let groupName: String     // e.g. "Piperazine N"
    }

    /// Detect all ionizable sites in a molecule (regardless of pH).
    static func detectIonizableSites(smiles: String) -> [IonizableSite] {
        guard let result = druse_detect_ionizable_sites(smiles) else { return [] }
        defer { druse_free_ion_sites(result) }

        let count = Int(result.pointee.count)
        guard count > 0, let sites = result.pointee.sites else { return [] }

        return (0..<count).map { i in
            let s = sites[i]
            let name = withUnsafePointer(to: s.groupName) {
                $0.withMemoryRebound(to: CChar.self, capacity: 64) { String(cString: $0) }
            }
            return IonizableSite(
                atomIdx: Int(s.atomIdx),
                isAcid: s.isAcid,
                defaultPKa: s.defaultPKa,
                groupName: name
            )
        }
    }

    /// A pair of protonated + deprotonated 3D structures for a single ionizable site.
    struct SiteProtomerPair: Sendable {
        let protonatedAtoms: [Atom]
        let protonatedBonds: [Bond]
        let protonatedCharge: Int
        let deprotonatedAtoms: [Atom]
        let deprotonatedBonds: [Bond]
        let deprotonatedCharge: Int
    }

    /// Generate protonated and deprotonated 3D structures for a specific ionizable site.
    /// Both forms are MMFF94-minimized with Gasteiger charges.
    static func generateSiteProtomers(smiles: String, atomIdx: Int, isAcid: Bool) -> SiteProtomerPair? {
        guard let result = druse_generate_site_protomers(smiles, Int32(atomIdx), isAcid) else { return nil }
        defer { druse_free_site_protomer_pair(result) }

        guard result.pointee.success,
              let prot = result.pointee.protonated, prot.pointee.success,
              let deprot = result.pointee.deprotonated, deprot.pointee.success else {
            return nil
        }

        let protMol = convertResult(prot.pointee)
        let deprotMol = convertResult(deprot.pointee)

        return SiteProtomerPair(
            protonatedAtoms: protMol.atoms,
            protonatedBonds: protMol.bonds,
            protonatedCharge: Int(result.pointee.protonatedCharge),
            deprotonatedAtoms: deprotMol.atoms,
            deprotonatedBonds: deprotMol.bonds,
            deprotonatedCharge: Int(result.pointee.deprotonatedCharge)
        )
    }

    /// Prepare ensemble with GNN-predicted ionizable sites.
    /// If sites is empty, falls back to SMARTS detection + lookup table.
    static func prepareEnsembleWithSites(
        smiles: String, name: String = "",
        pH: Double = 7.4, pkaThreshold: Double = 2.0,
        maxTautomers: Int = 10, maxProtomers: Int = 8,
        energyCutoff: Double = 15.0, conformersPerForm: Int = 5,
        temperature: Double = 298.15,
        sites: [PKaGNNPredictor.SitePrediction] = []
    ) -> EnsembleResult {
        let cResult: UnsafeMutablePointer<DruseEnsembleResult>?
        if sites.isEmpty {
            cResult = druse_prepare_ligand_ensemble(
                smiles, name, pH, pkaThreshold,
                Int32(maxTautomers), Int32(maxProtomers),
                energyCutoff, Int32(conformersPerForm), temperature
            )
        } else {
            var cSites = sites.map { s in
                DruseIonSiteDef(atomIdx: Int32(s.atomIdx), pKa: s.pKa, isAcid: s.isAcid)
            }
            cResult = cSites.withUnsafeMutableBufferPointer { buf in
                druse_prepare_ligand_ensemble_ex(
                    smiles, name, pH, pkaThreshold,
                    Int32(maxTautomers), Int32(maxProtomers),
                    energyCutoff, Int32(conformersPerForm), temperature,
                    buf.baseAddress, Int32(sites.count)
                )
            }
        }
        guard let cResult else {
            return EnsembleResult(members: [], numForms: 0, conformersPerForm: conformersPerForm,
                                  success: false, errorMessage: "Ensemble preparation returned nil")
        }
        defer { druse_free_ensemble_result(cResult) }

        guard cResult.pointee.success else {
            let msg = fixedCString(cResult.pointee.errorMessage)
            return EnsembleResult(members: [], numForms: 0, conformersPerForm: conformersPerForm,
                                  success: false, errorMessage: msg)
        }

        var members: [EnsembleMember] = []
        let count = Int(cResult.pointee.count)
        for i in 0..<count {
            let m = cResult.pointee.members[i]
            guard let molPtr = m.molecule, molPtr.pointee.success else { continue }
            let mol = convertResult(molPtr.pointee)
            members.append(EnsembleMember(
                id: i, molecule: mol,
                smiles: fixedCString(m.smiles),
                mmffEnergy: m.mmffEnergy,
                boltzmannWeight: m.boltzmannWeight,
                kind: Int(m.kind),
                label: fixedCString(m.label),
                formIndex: Int(m.formIndex),
                conformerIndex: Int(m.conformerIndex)
            ))
        }

        return EnsembleResult(
            members: members,
            numForms: Int(cResult.pointee.numForms),
            conformersPerForm: Int(cResult.pointee.numConformersPerForm),
            success: true, errorMessage: ""
        )
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

    // MARK: - SVG 2D Depiction

    /// Generate a publication-quality SVG depiction of a molecule from SMILES.
    /// Uses RDKit MolDraw2DSVG with proper wedge/dash stereo bonds, aromatic notation,
    /// and element coloring. Returns nil on failure.
    static func moleculeToSVG(smiles: String, width: Int = 400, height: Int = 300) async -> String? {
        let trimmed = smiles.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return await depictionWorker.run {
            guard let cStr = druse_mol_to_svg(trimmed, Int32(width), Int32(height)) else { return nil }
            defer { druse_free_string(cStr) }
            return String(cString: cStr)
        }
    }

    // MARK: - 2D Coordinates (legacy, used by interaction diagram)

    /// Result of 2D depiction coordinate generation for a ligand.
    struct Coords2D: Sendable {
        let positions: [CGPoint]     // one per heavy atom
        let atomicNums: [Int]        // atomic number per heavy atom
        let bonds: [(Int, Int, Int)] // (atom1, atom2, order) — Kekulized: no order 4
        let isAromatic: [Bool]       // per-atom aromaticity (from before Kekulization)
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
        var isAromatic: [Bool] = []
        isAromatic.reserveCapacity(n)

        let aroPtr = result.pointee.isAromatic
        for i in 0..<n {
            positions.append(CGPoint(x: CGFloat(coords[i * 2]), y: CGFloat(coords[i * 2 + 1])))
            atomicNums.append(Int(nums[i]))
            isAromatic.append(aroPtr?[i] ?? false)
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

        return Coords2D(positions: positions, atomicNums: atomicNums, bonds: bonds, isAromatic: isAromatic)
    }

    // MARK: - Pharmacophore Feature Detection

    /// Detected pharmacophore feature from RDKit BaseFeatures.fdef
    struct DetectedFeature: Sendable {
        let type: PharmacophoreFeatureType
        let position: SIMD3<Float>
        let atomIndices: [Int32]
        let familyName: String
    }

    /// Detect pharmacophore features from a SMILES string.
    /// Generates a 3D conformer internally, then uses RDKit's feature factory.
    static func detectPharmacophoreFeatures(smiles: String) -> [DetectedFeature]? {
        guard let result = druse_detect_pharmacophore_features(smiles) else { return nil }
        defer { druse_free_pharmacophore_features(result) }
        guard result.pointee.success else { return nil }

        return convertFeatureResult(result)
    }

    /// Detect pharmacophore features using pre-existing 3D coordinates.
    /// heavyPositions: array of heavy atom positions matching the SMILES heavy atom order.
    static func detectPharmacophoreFeatures(smiles: String, heavyPositions: [SIMD3<Float>]) -> [DetectedFeature]? {
        var interleaved = [Float](repeating: 0, count: heavyPositions.count * 3)
        for (i, pos) in heavyPositions.enumerated() {
            interleaved[i * 3] = pos.x
            interleaved[i * 3 + 1] = pos.y
            interleaved[i * 3 + 2] = pos.z
        }
        guard let result = druse_detect_pharmacophore_features_with_coords(
            smiles, interleaved, Int32(heavyPositions.count)) else { return nil }
        defer { druse_free_pharmacophore_features(result) }
        guard result.pointee.success else { return nil }

        return convertFeatureResult(result)
    }

    private static func convertFeatureResult(
        _ result: UnsafeMutablePointer<DrusePharmacophoreFeatureResult>
    ) -> [DetectedFeature] {
        let count = Int(result.pointee.featureCount)
        var features: [DetectedFeature] = []
        features.reserveCapacity(count)

        for i in 0..<count {
            let f = result.pointee.features[i]
            guard let featureType = PharmacophoreFeatureType(cType: f.type) else { continue }

            let indices: [Int32]
            if let ptr = f.atomIndices, f.atomCount > 0 {
                indices = Array(UnsafeBufferPointer(start: ptr, count: Int(f.atomCount)))
            } else {
                indices = []
            }

            let name = withUnsafePointer(to: f.familyName) {
                $0.withMemoryRebound(to: CChar.self, capacity: 32) { String(cString: $0) }
            }

            features.append(DetectedFeature(
                type: featureType,
                position: SIMD3<Float>(f.x, f.y, f.z),
                atomIndices: indices,
                familyName: name
            ))
        }
        return features
    }

    // MARK: - Maximum Common Substructure (MCS)

    /// Result of MCS computation.
    struct MCSResult: Sendable {
        let smartsPattern: String
        let numAtoms: Int
        let numBonds: Int
        let completed: Bool
    }

    /// Find the Maximum Common Substructure among multiple SMILES.
    static func findMCS(smilesArray: [String], timeoutSeconds: Int = 30) -> MCSResult? {
        guard smilesArray.count >= 2 else { return nil }

        let cStrings = smilesArray.map { strdup($0) }
        defer { cStrings.forEach { free($0) } }

        var ptrs: [UnsafePointer<CChar>?] = cStrings.map { UnsafePointer($0) }
        guard let result = druse_find_mcs(&ptrs, Int32(smilesArray.count), Int32(timeoutSeconds)) else {
            return nil
        }
        defer { druse_free_mcs_result(result) }
        guard result.pointee.success else { return nil }

        let smarts = withUnsafePointer(to: result.pointee.smartsPattern) {
            $0.withMemoryRebound(to: CChar.self, capacity: 2048) { String(cString: $0) }
        }

        return MCSResult(
            smartsPattern: smarts,
            numAtoms: Int(result.pointee.numAtoms),
            numBonds: Int(result.pointee.numBonds),
            completed: result.pointee.completed
        )
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

// MARK: - RDKit Depiction Worker

/// A single dedicated worker `Thread` with an 8 MB stack used to host RDKit
/// drawing/depiction calls. Required because RDKit's `compute2DCoords` /
/// `MolDraw2DSVG` can recurse deeply on complex inputs and exceed the
/// ~512 KB cooperative-thread stack used by Swift concurrency (the
/// deterministic EXC_BAD_ACCESS we observed). Calls are also serialized,
/// which protects RDKit's shared depiction/palette state.
final class RDKitDepictionWorker: NSObject, @unchecked Sendable {
    private let queueLock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var pending: [() -> Void] = []

    override init() {
        super.init()
        let t = Thread(target: self, selector: #selector(runLoop), object: nil)
        t.stackSize = 8 * 1024 * 1024
        t.qualityOfService = .userInitiated
        t.name = "com.druse.rdkit.depiction"
        t.start()
    }

    @objc private func runLoop() {
        while true {
            semaphore.wait()
            queueLock.lock()
            let work = pending.isEmpty ? nil : pending.removeFirst()
            queueLock.unlock()
            work?()
        }
    }

    func run<T: Sendable>(_ block: @Sendable @escaping () -> T) async -> T {
        await withCheckedContinuation { (cont: CheckedContinuation<T, Never>) in
            queueLock.lock()
            pending.append {
                let value = block()
                cont.resume(returning: value)
            }
            queueLock.unlock()
            semaphore.signal()
        }
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

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Chemistry-aware H-bond donor/acceptor classification for ligand atoms.
///
/// Shared between the post-docking interaction analysis (DockingEngine+Analysis)
/// and the DruseAFv4 / Druscore feature extractor (ML/Inference). Extracted
/// here to avoid a cross-module dependency from ML → @MainActor DockingEngine.
enum HBondClassifier {

    /// Default neutral valence.
    static func defaultValence(_ element: Element) -> Int {
        switch element {
        case .H:                      return 1
        case .C:                      return 4
        case .N:                      return 3
        case .O:                      return 2
        case .F, .Cl, .Br:            return 1
        case .P:                      return 3
        case .S:                      return 2
        default:                      return 0
        }
    }

    /// Adjusted valence given the formal charge.
    /// For pnictogens/chalcogens (N, O, P, S) a positive charge increases the
    /// valence by 1 (e.g. NH₄⁺ has 4 bonds), a negative charge decreases it.
    /// For carbon, |charge| reduces the valence (carbocations and carbanions
    /// both have one fewer bond than neutral C).
    static func adjustedValence(_ element: Element, formalCharge: Int) -> Int {
        let v = defaultValence(element)
        switch element {
        case .N, .O, .P, .S:  return v + formalCharge
        case .C:              return v - abs(formalCharge)
        default:              return v
        }
    }

    /// Boolean per ligand atom: true iff the atom has at least one implicit
    /// hydrogen attached (i.e. it could donate an H-bond). Only N/O/S atoms
    /// can be H-bond donors so other elements always return false.
    static func computeLigandHBondDonorFlags(atoms: [Atom], bonds: [Bond]) -> [Bool] {
        let n = atoms.count
        var bondOrderSum = Array<Float>(repeating: 0, count: n)
        for bond in bonds {
            let order: Float
            switch bond.order {
            case .single:    order = 1
            case .double:    order = 2
            case .triple:    order = 3
            case .aromatic:  order = 1.5
            }
            if bond.atomIndex1 < n { bondOrderSum[bond.atomIndex1] += order }
            if bond.atomIndex2 < n { bondOrderSum[bond.atomIndex2] += order }
        }
        return atoms.enumerated().map { (i, atom) in
            switch atom.element {
            case .N, .O, .S: break
            default: return false
            }
            let valence = adjustedValence(atom.element, formalCharge: atom.formalCharge)
            let implicitH = Int((Float(valence) - bondOrderSum[i]).rounded())
            return implicitH > 0
        }
    }

    /// Boolean per ligand atom: true iff the atom can ACCEPT an H-bond
    /// (has a lone pair available). Mirrors PIGNet2's chemistry-aware SMARTS:
    /// `[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]`
    /// — i.e. N/O/S that are not aromatic O/S, not pyrrole-like aromatic N,
    /// not pentavalent N/P, not tetra/hexavalent S, and not positively charged.
    static func computeLigandHBondAcceptorFlags(atoms: [Atom], bonds: [Bond]) -> [Bool] {
        let n = atoms.count
        var bondOrderSum = Array<Float>(repeating: 0, count: n)
        var inAromaticBond = Array<Bool>(repeating: false, count: n)
        for bond in bonds {
            let order: Float
            switch bond.order {
            case .single:    order = 1
            case .double:    order = 2
            case .triple:    order = 3
            case .aromatic:  order = 1.5
            }
            if bond.atomIndex1 < n {
                bondOrderSum[bond.atomIndex1] += order
                if bond.order == .aromatic { inAromaticBond[bond.atomIndex1] = true }
            }
            if bond.atomIndex2 < n {
                bondOrderSum[bond.atomIndex2] += order
                if bond.order == .aromatic { inAromaticBond[bond.atomIndex2] = true }
            }
        }
        return atoms.enumerated().map { (i, atom) in
            switch atom.element {
            case .N, .O, .S: break
            default: return false
            }
            if atom.formalCharge > 0 { return false }
            if inAromaticBond[i] && (atom.element == .O || atom.element == .S) {
                return false
            }
            let valence = adjustedValence(atom.element, formalCharge: atom.formalCharge)
            let bondOrder = Int(bondOrderSum[i].rounded())
            if inAromaticBond[i] && atom.element == .N {
                let implicitH = max(0, valence - bondOrder)
                if implicitH > 0 { return false }
            }
            if (atom.element == .N || atom.element == .P) && bondOrder >= 5 {
                return false
            }
            if atom.element == .S && (bondOrder == 4 || bondOrder == 6) {
                return false
            }
            return true
        }
    }

    // MARK: - Protein-side lookups

    /// True if a protein atom has at least one attached H — uses a residue+atom
    /// name lookup table. Backbone N (except PRO) always has 1 H. Side-chain
    /// donors are listed explicitly; everything else is treated as no-H.
    static func proteinHBondDonorFlag(_ atom: Atom) -> Bool {
        let resName = atom.residueName
        let atomName = atom.name.trimmingCharacters(in: .whitespaces)
        if atomName == "N" && resName != "PRO" { return true }
        switch (resName, atomName) {
        case ("SER", "OG"),  ("THR", "OG1"), ("TYR", "OH"),
             ("CYS", "SG"),
             ("ASN", "ND2"), ("GLN", "NE2"),
             ("LYS", "NZ"),
             ("ARG", "NE"),  ("ARG", "NH1"), ("ARG", "NH2"),
             ("TRP", "NE1"),
             ("HIS", "ND1"), ("HIS", "NE2"):
            return true
        default:
            return false
        }
    }

    /// Protein-side H-bond acceptor lookup. Returns true for atoms whose
    /// chemistry permits accepting an H. Mirrors the standard biochem rules.
    static func proteinHBondAcceptorFlag(_ atom: Atom) -> Bool {
        let resName = atom.residueName
        let atomName = atom.name.trimmingCharacters(in: .whitespaces)
        if atomName == "O" || atomName == "OXT" { return true }
        switch (resName, atomName) {
        case ("ASP", "OD1"), ("ASP", "OD2"),
             ("GLU", "OE1"), ("GLU", "OE2"),
             ("ASN", "OD1"),
             ("GLN", "OE1"),
             ("SER", "OG"),  ("THR", "OG1"), ("TYR", "OH"),
             ("CYS", "SG"),
             ("MET", "SD"),
             ("HIS", "ND1"), ("HIS", "NE2"):
            return true
        default:
            return false
        }
    }
}

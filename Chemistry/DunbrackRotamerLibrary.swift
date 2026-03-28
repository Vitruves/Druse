// ============================================================================
// DunbrackRotamerLibrary.swift — Backbone-dependent rotamer library (2010)
//
// Loads the Dunbrack 2010 backbone-dependent rotamer library from a compact
// binary file (dun2010bbdep.bin, ~14MB).  For each of the 18 amino acid types
// with chi angles (all except Ala/Gly), the library stores rotamer chi angles
// and probabilities indexed by backbone phi/psi in 10-degree bins.
//
// Reference: Shapovalov & Dunbrack, Structure 2011, 19(6):844-858.
//
// Binary format (per rotamer record, 20 bytes):
//   float  probability          (4 bytes)
//   short  chi1 × 10            (2 bytes)
//   short  chi2 × 10            (2 bytes)
//   short  chi3 × 10            (2 bytes)
//   short  chi4 × 10            (2 bytes)
//   short  variance1-4 × 10     (8 bytes, unused)
//
// Translated from FASPR (Huang 2020, MIT License):
//   temp/FASPR/src/RotamerBuilder.cpp::LoadBBdepRotlib2010()
// ============================================================================

import Foundation

// MARK: - Data Types

/// A single rotamer: probability + chi angles (degrees).
struct RotamerEntry: Sendable {
    let probability: Float
    let chiAngles: [Float]      // 1–4 chi angles in degrees
}

// MARK: - Library

/// Backbone-dependent rotamer library (Dunbrack 2010).
///
/// Usage:
///   let rots = DunbrackRotamerLibrary.shared.rotamers(for: "L", phi: -60, psi: -40)
///   // rots is sorted descending by probability, trimmed at P < 0.01 or cumP > 0.97
final class DunbrackRotamerLibrary: @unchecked Sendable {

    static let shared = DunbrackRotamerLibrary()

    /// Number of chi angles per residue type (1-letter code).
    static let chiCount: [Character: Int] = [
        "R": 4, "N": 2, "D": 2, "C": 1, "Q": 3, "E": 3, "H": 2, "I": 2,
        "L": 2, "K": 4, "M": 3, "F": 2, "P": 2, "S": 1, "T": 1, "W": 2,
        "Y": 2, "V": 1
    ]

    /// Maximum rotamer count per phi/psi bin (determines binary record count per bin).
    static let maxRotamers: [Character: Int] = [
        "R": 75, "N": 36, "D": 18, "C": 3, "Q": 108, "E": 54, "H": 36,
        "I": 9,  "L": 9,  "K": 73, "M": 27,  "F": 18, "P": 2,  "S": 3,
        "T": 3,  "W": 36, "Y": 18, "V": 3
    ]

    /// Cumulative record offset for each residue type in the binary file.
    /// offset = Rotl[aa] × 1296 records before this aa's data begins.
    /// This is the "skip lines" value — multiply by (1296 × 20) to get byte offset.
    static let recordOffset: [Character: Int] = [
        "R": 0,   "N": 75,  "D": 111, "C": 129, "Q": 132, "E": 240, "H": 294,
        "I": 330, "L": 339, "K": 348, "M": 421, "F": 448, "P": 466, "S": 468,
        "T": 471, "W": 474, "Y": 510, "V": 528
    ]

    /// Probability cutoffs.
    private static let probCutMin: Float = 0.01
    private static let probCutAcc: Float = 0.97

    /// The raw binary data (loaded once from bundle).
    private var data: Data?

    private init() {
        loadFromBundle()
    }

    // MARK: - Loading

    private func loadFromBundle() {
        if let url = Bundle.main.url(forResource: "dun2010bbdep", withExtension: "bin") {
            data = try? Data(contentsOf: url)
        }
    }

    /// Whether the library loaded successfully.
    var isLoaded: Bool { data != nil }

    // MARK: - Lookup

    /// Retrieve rotamers for a given residue type and backbone angles.
    ///
    /// - Parameters:
    ///   - residueType: One-letter amino acid code (e.g. "L" for Leu)
    ///   - phi: Backbone phi angle in degrees (-180..+180)
    ///   - psi: Backbone psi angle in degrees (-180..+180)
    /// - Returns: Rotamers sorted descending by probability, trimmed by cutoffs.
    ///   Empty for Ala, Gly, or unknown types.
    func rotamers(for residueType: Character, phi: Float, psi: Float) -> [RotamerEntry] {
        guard let data = data,
              let nchi = Self.chiCount[residueType],
              let maxRot = Self.maxRotamers[residueType],
              let rotl = Self.recordOffset[residueType]
        else { return [] }

        // Discretize phi/psi to 10-degree bins (0..35)
        let phiBin = Self.angleToBin(phi)
        let psiBin = Self.angleToBin(psi)

        // Byte offset into binary file
        let recordsPerAA = 1296  // 36 × 36
        let bytesPerRecord = 20
        let byteOffset = (recordsPerAA * rotl + (36 * phiBin + psiBin) * maxRot) * bytesPerRecord

        guard byteOffset >= 0, byteOffset + maxRot * bytesPerRecord <= data.count else { return [] }

        var result: [RotamerEntry] = []
        var cumulativeProb: Float = 0

        for j in 0..<maxRot {
            let recordStart = byteOffset + j * bytesPerRecord

            // Read probability (4 bytes, little-endian float)
            let prob: Float = data.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: recordStart, as: Float.self)
            }

            if prob < Self.probCutMin { break }

            let safeProbability = max(prob, 1e-7)

            // Read chi angles (nchi × 2-byte shorts, scaled by ×10)
            var chiAngles = [Float]()
            chiAngles.reserveCapacity(nchi)
            for k in 0..<nchi {
                let shortOffset = recordStart + 4 + k * 2
                let shortVal: Int16 = data.withUnsafeBytes { ptr in
                    ptr.load(fromByteOffset: shortOffset, as: Int16.self)
                }
                chiAngles.append(Float(shortVal) / 10.0)
            }

            result.append(RotamerEntry(probability: safeProbability, chiAngles: chiAngles))

            cumulativeProb += prob
            if cumulativeProb > Self.probCutAcc { break }
        }

        return result
    }

    /// Maximum probability among all rotamers for a given residue/phi/psi.
    func maxProbability(for residueType: Character, phi: Float, psi: Float) -> Float {
        let rots = rotamers(for: residueType, phi: phi, psi: psi)
        return rots.first?.probability ?? 0
    }

    // MARK: - Helpers

    /// Convert angle in degrees (-180..+180) to bin index (0..35).
    /// Bin centers: -175, -165, ..., +175 (10° spacing).
    private static func angleToBin(_ angle: Float) -> Int {
        let shifted = angle + 180.0
        var bin = Int(floor((shifted + 5.0) / 10.0))
        if bin >= 36 { bin -= 36 }
        if bin < 0 { bin += 36 }
        return bin
    }
}

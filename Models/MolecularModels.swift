import simd
import SwiftUI

// MARK: - Element

enum Element: Int, CaseIterable, Sendable {
    case H = 1, He, Li, Be, B, C, N, O, F, Ne
    case Na, Mg, Al, Si, P, S, Cl, Ar
    case K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
    case Ga, Ge, As, Se, Br, Kr

    var symbol: String {
        switch self {
        case .H:  "H"  case .He: "He" case .Li: "Li" case .Be: "Be"
        case .B:  "B"  case .C:  "C"  case .N:  "N"  case .O:  "O"
        case .F:  "F"  case .Ne: "Ne" case .Na: "Na" case .Mg: "Mg"
        case .Al: "Al" case .Si: "Si" case .P:  "P"  case .S:  "S"
        case .Cl: "Cl" case .Ar: "Ar" case .K:  "K"  case .Ca: "Ca"
        case .Sc: "Sc" case .Ti: "Ti" case .V:  "V"  case .Cr: "Cr"
        case .Mn: "Mn" case .Fe: "Fe" case .Co: "Co" case .Ni: "Ni"
        case .Cu: "Cu" case .Zn: "Zn" case .Ga: "Ga" case .Ge: "Ge"
        case .As: "As" case .Se: "Se" case .Br: "Br" case .Kr: "Kr"
        }
    }

    var name: String {
        switch self {
        case .H: "Hydrogen"   case .He: "Helium"     case .Li: "Lithium"
        case .Be: "Beryllium" case .B: "Boron"       case .C: "Carbon"
        case .N: "Nitrogen"   case .O: "Oxygen"      case .F: "Fluorine"
        case .Ne: "Neon"      case .Na: "Sodium"     case .Mg: "Magnesium"
        case .Al: "Aluminum"  case .Si: "Silicon"    case .P: "Phosphorus"
        case .S: "Sulfur"     case .Cl: "Chlorine"   case .Ar: "Argon"
        case .K: "Potassium"  case .Ca: "Calcium"    case .Sc: "Scandium"
        case .Ti: "Titanium"  case .V: "Vanadium"    case .Cr: "Chromium"
        case .Mn: "Manganese" case .Fe: "Iron"       case .Co: "Cobalt"
        case .Ni: "Nickel"    case .Cu: "Copper"     case .Zn: "Zinc"
        case .Ga: "Gallium"   case .Ge: "Germanium"  case .As: "Arsenic"
        case .Se: "Selenium"  case .Br: "Bromine"    case .Kr: "Krypton"
        }
    }

    var mass: Float {
        switch self {
        case .H:  1.008   case .He: 4.003   case .Li: 6.941   case .Be: 9.012
        case .B:  10.81   case .C:  12.011  case .N:  14.007  case .O:  15.999
        case .F:  18.998  case .Ne: 20.180  case .Na: 22.990  case .Mg: 24.305
        case .Al: 26.982  case .Si: 28.086  case .P:  30.974  case .S:  32.065
        case .Cl: 35.453  case .Ar: 39.948  case .K:  39.098  case .Ca: 40.078
        case .Sc: 44.956  case .Ti: 47.867  case .V:  50.942  case .Cr: 51.996
        case .Mn: 54.938  case .Fe: 55.845  case .Co: 58.933  case .Ni: 58.693
        case .Cu: 63.546  case .Zn: 65.38   case .Ga: 69.723  case .Ge: 72.63
        case .As: 74.922  case .Se: 78.96   case .Br: 79.904  case .Kr: 83.798
        }
    }

    var vdwRadius: Float {
        switch self {
        case .H:  1.20  case .He: 1.40  case .Li: 1.82  case .Be: 1.53
        case .B:  1.92  case .C:  1.70  case .N:  1.55  case .O:  1.52
        case .F:  1.47  case .Ne: 1.54  case .Na: 2.27  case .Mg: 1.73
        case .Al: 1.84  case .Si: 2.10  case .P:  1.80  case .S:  1.80
        case .Cl: 1.75  case .Ar: 1.88  case .K:  2.75  case .Ca: 2.31
        case .Sc: 2.15  case .Ti: 2.00  case .V:  2.05  case .Cr: 2.05
        case .Mn: 2.05  case .Fe: 2.04  case .Co: 2.00  case .Ni: 1.97
        case .Cu: 1.96  case .Zn: 2.01  case .Ga: 1.87  case .Ge: 2.11
        case .As: 1.85  case .Se: 1.90  case .Br: 1.85  case .Kr: 2.02
        }
    }

    var covalentRadius: Float {
        switch self {
        case .H:  0.31  case .He: 0.28  case .Li: 1.28  case .Be: 0.96
        case .B:  0.84  case .C:  0.76  case .N:  0.71  case .O:  0.66
        case .F:  0.57  case .Ne: 0.58  case .Na: 1.66  case .Mg: 1.41
        case .Al: 1.21  case .Si: 1.11  case .P:  1.07  case .S:  1.05
        case .Cl: 1.02  case .Ar: 1.06  case .K:  2.03  case .Ca: 1.76
        case .Sc: 1.70  case .Ti: 1.60  case .V:  1.53  case .Cr: 1.39
        case .Mn: 1.39  case .Fe: 1.32  case .Co: 1.26  case .Ni: 1.24
        case .Cu: 1.32  case .Zn: 1.22  case .Ga: 1.22  case .Ge: 1.20
        case .As: 1.19  case .Se: 1.20  case .Br: 1.20  case .Kr: 1.16
        }
    }

    /// CPK color scheme (Corey-Pauling-Koltun)
    var color: SIMD4<Float> {
        switch self {
        case .H:  SIMD4(0.95, 0.95, 0.95, 1.0) // white
        case .C:  SIMD4(0.56, 0.56, 0.56, 1.0) // dark gray
        case .N:  SIMD4(0.19, 0.31, 0.97, 1.0) // blue
        case .O:  SIMD4(0.90, 0.16, 0.16, 1.0) // red
        case .F:  SIMD4(0.56, 0.88, 0.31, 1.0) // green
        case .Cl: SIMD4(0.12, 0.94, 0.12, 1.0) // bright green
        case .Br: SIMD4(0.65, 0.16, 0.16, 1.0) // dark red
        case .S:  SIMD4(0.90, 0.78, 0.20, 1.0) // yellow
        case .P:  SIMD4(1.00, 0.50, 0.00, 1.0) // orange
        case .Fe: SIMD4(0.88, 0.40, 0.20, 1.0) // dark orange
        case .Na: SIMD4(0.67, 0.36, 0.95, 1.0) // purple
        case .Mg: SIMD4(0.54, 1.00, 0.00, 1.0) // bright green
        case .Ca: SIMD4(0.24, 1.00, 0.00, 1.0) // green
        case .Zn: SIMD4(0.49, 0.50, 0.69, 1.0) // slate
        case .Cu: SIMD4(0.78, 0.50, 0.20, 1.0) // copper
        default:  SIMD4(0.78, 0.50, 0.78, 1.0) // pink (unknown)
        }
    }

    /// Display radius for ball-and-stick mode (fraction of VdW)
    var displayRadius: Float {
        vdwRadius * 0.3
    }

    static func from(symbol: String) -> Element? {
        let s = symbol.trimmingCharacters(in: .whitespaces)
        return Element.allCases.first { $0.symbol.lowercased() == s.lowercased() }
    }
}

// MARK: - Bond Order

enum BondOrder: Int, Sendable {
    case single = 1
    case double = 2
    case triple = 3
    case aromatic = 4

    var displayRadius: Float {
        switch self {
        case .single:   0.22   // thicker bonds (MOE-like)
        case .double:   0.18
        case .triple:   0.15
        case .aromatic: 0.20
        }
    }
}

// MARK: - Secondary Structure

enum SecondaryStructure: Sendable {
    case coil
    case helix
    case sheet
    case turn
}

// MARK: - Chain Type

enum ChainType: Sendable {
    case protein
    case nucleicAcid   // DNA or RNA
    case ligand
    case water
    case ion
    case unknown

    var label: String {
        switch self {
        case .protein:     "Protein"
        case .nucleicAcid: "DNA/RNA"
        case .ligand:      "Ligand"
        case .water:       "Water"
        case .ion:         "Ion"
        case .unknown:     "Other"
        }
    }

    /// Standard nucleic acid residue names (DNA + RNA, including common modified forms).
    static let nucleicAcidResidues: Set<String> = [
        // DNA
        "DA", "DT", "DG", "DC", "DU", "DI",
        // RNA
        "A", "U", "G", "C", "I",
        // PDB 3-letter (less common)
        "ADE", "THY", "GUA", "CYT", "URA",
    ]
}

// MARK: - Atom

struct Atom: Identifiable, Sendable {
    let id: Int
    var element: Element
    var position: SIMD3<Float>
    var name: String            // PDB atom name, e.g. "CA", "N", "O"
    var residueName: String     // e.g. "ALA", "GLY"
    var residueSeq: Int
    var chainID: String
    var charge: Float
    var formalCharge: Int
    var isHetAtom: Bool
    var occupancy: Float
    var tempFactor: Float
    var altLoc: String
    var insertionCode: String

    init(
        id: Int,
        element: Element,
        position: SIMD3<Float>,
        name: String = "",
        residueName: String = "",
        residueSeq: Int = 0,
        chainID: String = "A",
        charge: Float = 0,
        formalCharge: Int = 0,
        isHetAtom: Bool = false,
        occupancy: Float = 1.0,
        tempFactor: Float = 0.0,
        altLoc: String = "",
        insertionCode: String = ""
    ) {
        self.id = id
        self.element = element
        self.position = position
        self.name = name
        self.residueName = residueName
        self.residueSeq = residueSeq
        self.chainID = chainID
        self.charge = charge
        self.formalCharge = formalCharge
        self.isHetAtom = isHetAtom
        self.occupancy = occupancy
        self.tempFactor = tempFactor
        self.altLoc = altLoc
        self.insertionCode = insertionCode
    }
}

// MARK: - Bond

struct Bond: Identifiable, Sendable {
    let id: Int
    var atomIndex1: Int
    var atomIndex2: Int
    var order: BondOrder
    var isRotatable: Bool

    init(id: Int, atomIndex1: Int, atomIndex2: Int, order: BondOrder = .single, isRotatable: Bool = false) {
        self.id = id
        self.atomIndex1 = atomIndex1
        self.atomIndex2 = atomIndex2
        self.order = order
        self.isRotatable = isRotatable
    }
}

// MARK: - Residue

struct Residue: Identifiable, Sendable {
    let id: Int
    var name: String            // 3-letter code: ALA, GLY, etc.
    var sequenceNumber: Int
    var chainID: String
    var atomIndices: [Int]
    var secondaryStructure: SecondaryStructure
    var isStandard: Bool        // standard amino acid
    var isWater: Bool

    init(
        id: Int,
        name: String,
        sequenceNumber: Int,
        chainID: String = "A",
        atomIndices: [Int] = [],
        secondaryStructure: SecondaryStructure = .coil,
        isStandard: Bool = true,
        isWater: Bool = false
    ) {
        self.id = id
        self.name = name
        self.sequenceNumber = sequenceNumber
        self.chainID = chainID
        self.atomIndices = atomIndices
        self.secondaryStructure = secondaryStructure
        self.isStandard = isStandard
        self.isWater = isWater
    }
}

// MARK: - Chain

struct Chain: Identifiable, Sendable {
    let id: String
    var residueIndices: [Int]
    var type: ChainType

    var displayColor: Color {
        let colors: [Color] = [.cyan, .green, .orange, .pink, .purple, .yellow, .mint, .teal]
        let hash = abs(id.hashValue)
        return colors[hash % colors.count]
    }
}

// MARK: - Residue Subset (MOE-style grouping)

/// A user-defined group of residues for operations like pocket definition, surface display, etc.
struct ResidueSubset: Identifiable, Sendable {
    let id: UUID
    var name: String
    var residueIndices: [Int]      // indices into Molecule.residues
    var color: SIMD4<Float>        // display color
    var isVisible: Bool = true

    init(name: String, residueIndices: [Int], color: SIMD4<Float> = SIMD4(0.2, 0.8, 0.9, 1.0)) {
        self.id = UUID()
        self.name = name
        self.residueIndices = residueIndices
        self.color = color
    }

    /// All atom indices belonging to this subset's residues.
    @MainActor
    func atomIndices(in molecule: Molecule) -> [Int] {
        residueIndices.flatMap { idx -> [Int] in
            guard idx < molecule.residues.count else { return [] }
            return molecule.residues[idx].atomIndices
        }
    }
}

// MARK: - Scoring Method

/// Which scoring function to use for ranking docked poses.
enum ScoringMethod: String, CaseIterable, Sendable {
    case vina = "Vina"
    case drusina = "Drusina"
    case druseAffinity = "Druse Affinity"

    /// Short label for compact UI (picker buttons).
    var shortLabel: String {
        switch self {
        case .vina:          "Vina"
        case .drusina:       "Drusina"
        case .druseAffinity: "Druse AF"
        }
    }

    var icon: String {
        switch self {
        case .vina:          "function"
        case .drusina:       "sparkles"
        case .druseAffinity: "brain"
        }
    }

    var description: String {
        switch self {
        case .vina:          "Empirical energy scoring (kcal/mol)"
        case .drusina:       "Extended Vina + π-π, π-cation, halogen bond, metal coord (kcal/mol)"
        case .druseAffinity: "Neural network affinity prediction (pKi)"
        }
    }
}

// MARK: - Search Method

/// Which search engine to use for pose sampling during docking.
enum SearchMethod: String, CaseIterable, Sendable {
    case genetic          = "GA"
    case fragmentBased    = "Fragment"
    case diffusionGuided  = "Diffusion"
    case parallelTempering = "Replica Exchange"
    case auto             = "Auto"

    var shortLabel: String {
        switch self {
        case .genetic:          "GA"
        case .fragmentBased:    "Fragment"
        case .diffusionGuided:  "Diffusion"
        case .parallelTempering: "REMC"
        case .auto:             "Auto"
        }
    }

    var icon: String {
        switch self {
        case .genetic:          "arrow.triangle.branch"
        case .fragmentBased:    "puzzlepiece.extension"
        case .diffusionGuided:  "waveform.path"
        case .parallelTempering: "thermometer.variable"
        case .auto:             "gearshape.2"
        }
    }

    var description: String {
        switch self {
        case .genetic:          "Genetic algorithm + Metropolis ILS (default)"
        case .fragmentBased:    "Incremental fragment construction with beam search"
        case .diffusionGuided:  "DruseAF attention-guided reverse diffusion"
        case .parallelTempering: "Replica exchange Monte Carlo (multiple temperatures)"
        case .auto:             "Automatically select based on ligand flexibility"
        }
    }
}

// MARK: - Affinity Display Unit

/// How to display binding affinity from the Druse ML scorer.
enum AffinityDisplayUnit: String, CaseIterable, Sendable {
    case pKi = "pKi"
    case ki  = "Ki"

    /// Format a pKd value for display in the selected unit.
    func format(_ pKd: Float) -> String {
        switch self {
        case .pKi:
            return String(format: "%.2f", pKd)
        case .ki:
            return Self.formatKi(pKd: pKd)
        }
    }

    /// Unit label for display.
    var unitLabel: String {
        switch self {
        case .pKi: "pKi"
        case .ki:  "nM"
        }
    }

    /// Convert pKd to Ki in nanomolar and format with appropriate unit prefix.
    static func formatKi(pKd: Float) -> String {
        let ki_M = pow(10, -Double(pKd))  // Ki in molar
        let ki_nM = ki_M * 1e9
        if ki_nM >= 1_000_000 {
            return String(format: "%.1f mM", ki_nM / 1_000_000)
        } else if ki_nM >= 1_000 {
            return String(format: "%.1f \u{00B5}M", ki_nM / 1_000)
        } else if ki_nM >= 1 {
            return String(format: "%.1f nM", ki_nM)
        } else {
            return String(format: "%.2f pM", ki_nM * 1_000)
        }
    }

    /// Convert pKd to Ki in molar.
    static func pKdToKi(pKd: Float) -> Double {
        pow(10, -Double(pKd))
    }
}

// MARK: - Charge Method

enum ChargeMethod: String, CaseIterable, Sendable {
    case gasteiger = "Gasteiger"
    case eem = "EEM"
    case qeq = "QEq"
    case xtb = "GFN2-xTB"

    var icon: String {
        switch self {
        case .gasteiger: "bolt"
        case .eem:       "equal.circle"
        case .qeq:       "waveform"
        case .xtb:       "atom"
        }
    }

    var description: String {
        switch self {
        case .gasteiger: "Gasteiger-Marsili empirical (fast, lowest accuracy)"
        case .eem:       "Electronegativity Equalization Method (fast, good accuracy)"
        case .qeq:       "Charge Equilibration - Rapp\u{00E9} & Goddard (fast, good accuracy)"
        case .xtb:       "GFN2-xTB semi-empirical quantum (slow, highest accuracy)"
        }
    }
}

// MARK: - Side Chain Display (ribbon mode)

/// Controls which residue side chains are shown as ball-and-stick in ribbon mode.
enum SideChainDisplay: String, CaseIterable, Sendable {
    case none        = "None"
    case interacting = "Interacting"   // Only residues with detected interactions
    case selected    = "Selected"       // Only user-selected residues
    case all         = "All"

    var icon: String {
        switch self {
        case .none:        "eye.slash"
        case .interacting: "link"
        case .selected:    "hand.tap"
        case .all:         "eye"
        }
    }

    /// Backbone atom names to EXCLUDE when showing side chains.
    static let backboneAtomNames: Set<String> = ["N", "CA", "C", "O", "OXT", "H", "HA"]
}

// MARK: - Variant Kind (Tautomer / Protomer) — legacy, kept for serialization compat
enum VariantKind: Int, Sendable, Codable {
    case tautomer = 0
    case protomer = 1
}

// MARK: - Chemical Form (tautomer/protomer/conformer hierarchy)

/// Kind of chemical form relative to the parent molecule.
/// Maps directly to C++ `kind` field in DruseEnsembleMember.
enum ChemicalFormKind: Int, Sendable, Codable {
    case parent = 0
    case tautomer = 1
    case protomer = 2
    case tautomerProtomer = 3

    var label: String {
        switch self {
        case .parent: "Parent"
        case .tautomer: "Tautomer"
        case .protomer: "Protomer"
        case .tautomerProtomer: "Taut+Prot"
        }
    }

    var symbol: String {
        switch self {
        case .parent: "P"
        case .tautomer: "T"
        case .protomer: "H"
        case .tautomerProtomer: "TH"
        }
    }

    var color: String {
        switch self {
        case .parent: "green"
        case .tautomer: "cyan"
        case .protomer: "orange"
        case .tautomerProtomer: "purple"
        }
    }
}

/// A single 3D conformer of a chemical form.
struct Conformer3D: Identifiable, Sendable {
    let id: Int              // 0-based index within parent form
    var atoms: [Atom]
    var bonds: [Bond]
    var energy: Double       // MMFF94 kcal/mol
}

/// A distinct chemical form (protonation/tautomeric state) of a molecule.
/// Contains multiple 3D conformers sorted by energy.
struct ChemicalForm: Identifiable, Sendable {
    let id: UUID
    var smiles: String                      // canonical SMILES for this form
    var kind: ChemicalFormKind              // .parent, .tautomer, .protomer, .tautomerProtomer
    var label: String                       // e.g. "Taut2", "prot_Amine+Taut1"
    var boltzmannWeight: Double             // population fraction (sums to ~1.0 across forms)
    var relativeEnergy: Double              // kcal/mol vs best form (0.0 for best)
    var conformers: [Conformer3D]           // 3D structures, sorted by energy ascending

    /// Best (lowest energy) conformer.
    var bestConformer: Conformer3D? { conformers.first }
    /// Atoms of the best conformer.
    var atoms: [Atom] { bestConformer?.atoms ?? [] }
    /// Bonds of the best conformer.
    var bonds: [Bond] { bestConformer?.bonds ?? [] }
    /// Conformer count.
    var conformerCount: Int { conformers.count }
    /// Energy range string.
    var energyRangeString: String {
        guard let lo = conformers.first?.energy, let hi = conformers.last?.energy else { return "—" }
        if conformers.count == 1 { return String(format: "%.1f", lo) }
        return String(format: "%.1f–%.1f", lo, hi)
    }
    /// Population percentage string.
    var populationString: String { String(format: "%.1f%%", boltzmannWeight * 100) }

    init(smiles: String, kind: ChemicalFormKind, label: String,
         boltzmannWeight: Double = 0, relativeEnergy: Double = 0,
         conformers: [Conformer3D] = []) {
        self.id = UUID()
        self.smiles = smiles
        self.kind = kind
        self.label = label
        self.boltzmannWeight = boltzmannWeight
        self.relativeEnergy = relativeEnergy
        self.conformers = conformers
    }
}

// MARK: - Render Mode

enum RenderMode: String, CaseIterable, Sendable {
    case ballAndStick = "Ball & Stick"
    case spaceFilling = "Space Filling"
    case wireframe = "Wireframe"
    case ribbon = "Ribbon"

    var icon: String {
        switch self {
        case .ballAndStick: "circle.grid.3x3"
        case .spaceFilling: "circle.fill"
        case .wireframe:    "line.3.horizontal"
        case .ribbon:       "water.waves"
        }
    }

    var atomRadiusScale: Float {
        switch self {
        case .ballAndStick: 0.3
        case .spaceFilling: 1.0
        case .wireframe:    0.12  // small dots at atom centers
        case .ribbon:       0.25  // thinner atoms when ribbon visible
        }
    }

    var bondRadiusScale: Float {
        switch self {
        case .ballAndStick: 1.0
        case .spaceFilling: 0.0
        case .wireframe:    0.8   // visible bonds (0.22 × 0.8 = 0.176Å radius)
        case .ribbon:       0.0   // no bonds in ribbon mode (side chains use ball-and-stick)
        }
    }
}

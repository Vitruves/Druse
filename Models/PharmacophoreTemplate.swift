import Foundation
import simd

// MARK: - Pharmacophore Template

/// A reusable pharmacophore template defining spatial feature requirements.
/// Can be created from a reference ligand's detected features or from MCS analysis.
/// Converts to `PharmacophoreConstraintDef` array at dock time.
struct PharmacophoreTemplate: Identifiable, Sendable {
    let id: UUID
    var name: String
    var features: [PharmacophoreFeature]
    var sourceSMILES: String?         // the ligand this was derived from
    var sourceMCS: String?            // MCS SMARTS if derived from multi-ligand analysis

    init(name: String = "Untitled", features: [PharmacophoreFeature] = [], sourceSMILES: String? = nil) {
        self.id = UUID()
        self.name = name
        self.features = features
        self.sourceSMILES = sourceSMILES
    }
}

// MARK: - Pharmacophore Feature

/// A single pharmacophore feature: a spatial/chemical requirement in the binding site.
struct PharmacophoreFeature: Identifiable, Sendable {
    let id: UUID
    var type: PharmacophoreFeatureType
    var position: SIMD3<Float>          // 3D position (from reference ligand)
    var tolerance: Float                // radius sphere in Å (default 1.5)
    var isRequired: Bool                // hard constraint vs optional
    var isEnabled: Bool                 // user can toggle on/off
    var atomIndices: [Int32]            // heavy atom indices in the source molecule
    var strength: Float                 // penalty kcal/mol/Å² for soft constraints

    init(
        type: PharmacophoreFeatureType,
        position: SIMD3<Float>,
        tolerance: Float = 1.5,
        isRequired: Bool = false,
        atomIndices: [Int32] = []
    ) {
        self.id = UUID()
        self.type = type
        self.position = position
        self.tolerance = tolerance
        self.isRequired = isRequired
        self.isEnabled = true
        self.atomIndices = atomIndices
        self.strength = 10.0
    }
}

// MARK: - Feature Type

/// Types of pharmacophore features, mapping to both RDKit detection families
/// and the existing ConstraintInteractionType for GPU enforcement.
enum PharmacophoreFeatureType: String, CaseIterable, Sendable {
    case donor = "H-Bond Donor"
    case acceptor = "H-Bond Acceptor"
    case hydrophobic = "Hydrophobic"
    case aromatic = "Aromatic"
    case positiveIonizable = "Positive Ionizable"
    case negativeIonizable = "Negative Ionizable"

    /// Map from C enum DrusePharmacophoreType
    init?(cType: Int32) {
        switch cType {
        case 0: self = .donor
        case 1: self = .acceptor
        case 2: self = .hydrophobic
        case 3: self = .aromatic
        case 4: self = .positiveIonizable
        case 5: self = .negativeIonizable
        default: return nil
        }
    }

    /// Convert to ConstraintInteractionType for GPU enforcement
    var constraintInteractionType: ConstraintInteractionType {
        switch self {
        case .donor:             return .hbondDonor
        case .acceptor:          return .hbondAcceptor
        case .hydrophobic:       return .hydrophobic
        case .aromatic:          return .piStacking
        case .positiveIonizable: return .saltBridge
        case .negativeIonizable: return .saltBridge
        }
    }

    var icon: String {
        switch self {
        case .donor:             return "arrow.up.right.circle"
        case .acceptor:          return "arrow.down.left.circle"
        case .hydrophobic:       return "drop.circle"
        case .aromatic:          return "circle.hexagongrid"
        case .positiveIonizable: return "plus.circle"
        case .negativeIonizable: return "minus.circle"
        }
    }

    var color: SIMD4<Float> {
        switch self {
        case .donor:             return SIMD4<Float>(1.0, 0.6, 0.1, 1.0)   // orange
        case .acceptor:          return SIMD4<Float>(0.1, 0.8, 0.9, 1.0)   // cyan
        case .hydrophobic:       return SIMD4<Float>(0.6, 0.6, 0.2, 1.0)   // olive
        case .aromatic:          return SIMD4<Float>(0.6, 0.3, 0.9, 1.0)   // purple
        case .positiveIonizable: return SIMD4<Float>(0.2, 0.4, 1.0, 1.0)   // blue
        case .negativeIonizable: return SIMD4<Float>(0.9, 0.2, 0.2, 1.0)   // red
        }
    }
}

// MARK: - Template → Constraints Conversion

extension PharmacophoreTemplate {

    /// Convert enabled features into PharmacophoreConstraintDef array
    /// ready for the docking engine. Each feature becomes one constraint
    /// with position in pocket space and the appropriate interaction type.
    func toConstraints() -> [PharmacophoreConstraintDef] {
        var constraints: [PharmacophoreConstraintDef] = []
        var groupID = 0

        for feature in features where feature.isEnabled {
            let strength: ConstraintStrength = feature.isRequired
                ? .hard
                : .soft(kcalPerAngstromSq: feature.strength)

            var constraint = PharmacophoreConstraintDef(
                targetScope: .atom,
                interactionType: feature.type.constraintInteractionType,
                strength: strength,
                distanceThreshold: feature.tolerance,
                sourceType: .receptor  // position is in pocket space
            )
            constraint.targetPositions = [feature.position]
            constraint.groupID = groupID
            constraint.residueName = feature.type.rawValue
            constraint.atomName = feature.type.icon

            constraints.append(constraint)
            groupID += 1
        }

        return constraints
    }
}

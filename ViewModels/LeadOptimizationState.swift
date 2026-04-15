// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Lead Optimization State

/// State for the Lead Optimization pipeline tab.
/// Manages reference selection, analog generation, property-directed modifications,
/// ADMET filtering, mini-docking, and comparison.
struct LeadOptimizationState {

    // MARK: Reference

    var referenceName: String = ""
    var referenceSMILES: String = ""
    var referenceResult: DockingResult?
    var referenceMolecule: Molecule?
    var referenceDescriptors: LigandDescriptors?

    // MARK: Generation Parameters

    var analogCount: Int = 50
    var similarityThreshold: Float = 0.7
    var keepScaffold: Bool = true

    /// Property direction biases (-1..+1). 0 = no preference.
    var polarityDirection: Float = 0       // -1 = less polar, +1 = more polar
    var rigidityDirection: Float = 0       // -1 = more flexible, +1 = more rigid
    var lipophilicityDirection: Float = 0  // -1 = hydrophilic, +1 = lipophilic
    var sizeDirection: Float = 0           // -1 = smaller, +1 = larger

    // MARK: ADMET Filters

    var filterLipinski: Bool = false     // off by default — too restrictive for initial exploration
    var filterVeber: Bool = false
    var filterHERG: Bool = false        // reject if hERG risk > 0.5
    var filterCYP: Bool = false         // reject if CYP2D6 or CYP3A4 > 0.5
    var maxLogP: Float = 7.0            // generous default — user can tighten
    var minSolubility: Float = -8.0     // generous default

    // MARK: Generation State

    var isGenerating: Bool = false
    var generationProgress: Float = 0
    var generationTask: Task<Void, Never>?

    // MARK: Analogs

    var analogs: [LeadOptAnalog] = []

    // MARK: Mini-Docking

    var isDocking: Bool = false
    var dockingProgress: (current: Int, total: Int) = (0, 0)
    var dockingTask: Task<Void, Never>?

    // MARK: Comparison

    var selectedAnalogIndex: Int?
    var showComparison: Bool = false

    // MARK: Computed

    var hasReference: Bool { !referenceSMILES.isEmpty }
    var dockedAnalogCount: Int { analogs.filter { $0.status == .docked }.count }
    var passedFilterCount: Int { analogs.filter { $0.status != .filtered && $0.status != .failed }.count }
}

// MARK: - Lead Optimization Analog

struct LeadOptAnalog: Identifiable, Sendable {
    let id: UUID
    var name: String
    var smiles: String
    var molecule: MoleculeData?
    var descriptors: LigandDescriptors?
    var admet: ADMETPredictor.ADMETResult?

    // Docking results (filled after mini-docking)
    var dockingResults: [DockingResult] = []
    var bestEnergy: Float?
    var bestPoseAtoms: [SIMD3<Float>] = []
    var rmsdToReference: Float?

    // Delta properties vs reference
    var deltaMW: Float?
    var deltaLogP: Float?
    var deltaTPSA: Float?
    var deltaRotBonds: Int?

    // Status
    var status: Status = .generated

    enum Status: Sendable, Equatable {
        case generated      // SMILES created
        case prepared       // 3D coordinates generated
        case docked         // Docking complete
        case filtered       // Failed ADMET filters
        case failed         // RDKit/docking error
    }
}

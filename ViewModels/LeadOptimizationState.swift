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
    /// Scoring method of the reference pose. Analogs are mini-docked with the same method so
    /// their scores sit on the reference's scale (kcal/mol for energies, pK for affinity models).
    var scoringMethod: ScoringMethod = .vina

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

    // MARK: Parameter defaults

    /// Restores the generation parameters, property directions and ADMET filters to their
    /// defaults, keeping the reference and the generated analogs (`leadopt defaults`).
    mutating func resetParameters() {
        let defaults = LeadOptimizationState()
        analogCount = defaults.analogCount
        similarityThreshold = defaults.similarityThreshold
        keepScaffold = defaults.keepScaffold
        polarityDirection = defaults.polarityDirection
        rigidityDirection = defaults.rigidityDirection
        lipophilicityDirection = defaults.lipophilicityDirection
        sizeDirection = defaults.sizeDirection
        filterLipinski = defaults.filterLipinski
        filterVeber = defaults.filterVeber
        filterHERG = defaults.filterHERG
        filterCYP = defaults.filterCYP
        maxLogP = defaults.maxLogP
    }

    // MARK: Computed

    var hasReference: Bool { !referenceSMILES.isEmpty }
    var referenceScore: Float? { referenceResult?.displayScore(method: scoringMethod) }
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
    /// Best pose score in `LeadOptimizationState.scoringMethod` units.
    var bestScore: Float?
    var bestPoseAtoms: [SIMD3<Float>] = []

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

extension ScoringMethod {
    /// Whether score `a` ranks ahead of `b`: higher pK for affinity models, lower energy otherwise.
    func ranks(_ a: Float, before b: Float) -> Bool {
        isAffinityScore ? a > b : a < b
    }
}

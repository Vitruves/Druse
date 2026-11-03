// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Modal sheet for configuring a pharmacophore docking constraint.
/// Presented when the user right-clicks an atom/residue and selects "Add Docking Constraint..."
struct ConstraintConfigSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) private var viewModel

    let context: ConstraintSheetContext

    @State private var targetScope: TargetScope = .atom
    @State private var interactionType: ConstraintInteractionType = .hbondAcceptor
    @State private var strengthMode: StrengthMode = .soft
    @State private var softStrength: Float = 5.0
    @State private var distanceThreshold: Float = 3.5

    enum StrengthMode: String, CaseIterable {
        case soft = "Soft"
        case hard = "Hard"
    }

    var body: some View {
        DSSheetScaffold(
            title: "Add Docking Constraint",
            subtitle: "Bias docking toward a specific interaction at this position.",
            systemImage: "scope"
        ) {
            Form {
                targetSection
                interactionSection
                strengthSection
                distanceSection
            }
            .formStyle(.grouped)
        } footer: {
            Button("Cancel", role: .cancel) { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Add Constraint") {
                addConstraint()
                dismiss()
            }
            .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
            .keyboardShortcut(.defaultAction)
        }
        .frame(minWidth: 400, idealWidth: 420, minHeight: 420, idealHeight: 540)
        .onAppear { setDefaults() }
    }

    // MARK: - Target

    private var targetSection: some View {
        Section {
            LabeledContent {
                DSTag(context.sourceType == .receptor ? "Receptor" : "Ligand",
                      tone: context.sourceType == .receptor ? .protein : .ligand)
            } label: {
                HStack(spacing: DS.Space.s) {
                    Text(targetTitle)
                        .font(context.residueName == nil ? DS.Font.body.weight(.medium) : DS.Font.mono.weight(.medium))
                    if let chain = context.chainID, !chain.isEmpty {
                        DSTag("Chain \(chain)", monospaced: true)
                    }
                }
            }
            Picker("Scope", selection: $targetScope) {
                ForEach(TargetScope.allCases, id: \.self) { scope in
                    Text(scope.rawValue).tag(scope)
                }
            }
            .pickerStyle(.segmented)
        } header: {
            Text("Target")
        } footer: {
            Text(targetScope == .atom
                 ? "Constraint applies to the exact selected atom position."
                 : "Constraint is satisfied if ANY relevant atom in the residue interacts.")
        }
    }

    private var targetTitle: String {
        if let atomName = context.atomName, let resName = context.residueName {
            return "\(atomName) @ \(resName)"
        }
        return context.residueName ?? "Selection"
    }

    // MARK: - Interaction Type

    private var interactionSection: some View {
        Section {
            Picker("Interaction", selection: $interactionType) {
                ForEach(ConstraintInteractionType.allCases, id: \.self) { type in
                    Label(type.rawValue, systemImage: type.icon).tag(type)
                }
            }
            .pickerStyle(.menu)
        } header: {
            Text("Interaction Type")
        } footer: {
            Text(interactionTypeDescription)
        }
    }

    // MARK: - Strength

    private var strengthSection: some View {
        Section {
            Picker("Strength", selection: $strengthMode) {
                ForEach(StrengthMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            if strengthMode == .soft {
                LabeledContent("Penalty") {
                    HStack(spacing: DS.Space.m) {
                        Slider(value: snappedSliderBinding($softStrength, range: 1...50, step: 1), in: 1...50)
                            .controlSize(.small)
                            .frame(minWidth: 160)
                        Text(String(format: "%.0f kcal/mol/Å²", softStrength))
                            .font(DS.Font.monoSmall)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .fixedSize()
                    }
                }
            } else {
                Label("Mandatory — poses violating this constraint will be strongly penalized (effective rejection).",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(DS.Font.secondary)
                    .foregroundStyle(DS.Color.warning)
            }
        } header: {
            Text("Constraint Strength")
        }
    }

    // MARK: - Distance

    private var distanceSection: some View {
        Section {
            LabeledContent("Threshold") {
                HStack(spacing: DS.Space.m) {
                    Slider(value: snappedSliderBinding($distanceThreshold, range: 2.0...5.0, step: 0.1), in: 2.0...5.0)
                        .controlSize(.small)
                        .frame(minWidth: 160)
                    Text(String(format: "%.1f Å", distanceThreshold))
                        .font(DS.Font.monoSmall)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .fixedSize()
                }
            }
        } header: {
            Text("Distance")
        } footer: {
            Text("Maximum distance for the constraint to be considered satisfied.")
        }
    }

    // MARK: - Helpers

    private var interactionTypeDescription: String {
        switch interactionType {
        case .hbondDonor:       return "Ligand must present an H-bond donor (N-H, O-H) near this position."
        case .hbondAcceptor:    return "Ligand must present an H-bond acceptor (N, O lone pair) near this position."
        case .saltBridge:       return "Ligand must form a charged interaction (salt bridge) near this position."
        case .piStacking:       return "Ligand must position an aromatic ring for \u{03C0}-stacking near this position."
        case .halogen:          return "Ligand must present a halogen atom (F, Cl, Br) near this position."
        case .metalCoordination: return "Ligand must coordinate the metal ion with N, O, or S atoms."
        case .hydrophobic:      return "Ligand must position hydrophobic atoms (C, halogens) near this position."
        }
    }

    private func setDefaults() {
        // Auto-detect target scope from context
        if context.atomIndex != nil {
            targetScope = .atom
        } else if context.residueIndex != nil {
            targetScope = .residue
        }

        // Auto-suggest interaction type from atom context
        if let element = context.element, let atomName = context.atomName, let resName = context.residueName {
            interactionType = ConstraintInteractionType.suggestDefault(
                element: element, atomName: atomName, residueName: resName
            )
        }
    }

    private func addConstraint() {
        let strength: ConstraintStrength = strengthMode == .hard
            ? .hard
            : .soft(kcalPerAngstromSq: softStrength)

        var constraint = PharmacophoreConstraintDef(
            targetScope: targetScope,
            interactionType: interactionType,
            strength: strength,
            distanceThreshold: distanceThreshold,
            sourceType: context.sourceType,
            proteinAtomIndex: context.sourceType == .receptor ? context.atomIndex : nil,
            residueIndex: context.residueIndex,
            ligandAtomIndex: context.sourceType == .ligand ? context.atomIndex : nil,
            chainID: context.chainID,
            residueName: context.residueName,
            atomName: context.atomName
        )

        // Resolve target positions immediately
        if context.sourceType == .receptor, let protein = viewModel.molecules.protein {
            constraint.targetPositions = ConstraintAtomResolver.resolveTargetPositions(
                constraint: constraint, atoms: protein.atoms, residues: protein.residues
            )
        }

        // Assign group ID
        constraint.groupID = viewModel.docking.pharmacophoreConstraints.count

        viewModel.docking.pharmacophoreConstraints.append(constraint)
        viewModel.pushToRenderer()
    }
}

// MARK: - Context

/// Encapsulates the selection context that triggered the constraint sheet.
struct ConstraintSheetContext {
    var atomIndex: Int?
    var residueIndex: Int?
    var sourceType: SourceType
    var atomName: String?
    var residueName: String?
    var chainID: String?
    var element: Element?
}

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
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    sourceInfoSection
                    Divider()
                    scopeSection
                    Divider()
                    interactionTypeSection
                    Divider()
                    strengthSection
                    Divider()
                    distanceSection
                }
                .padding()
            }
            Divider()
            footer
        }
        .frame(width: 380, height: 440)
        .onAppear { setDefaults() }
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        HStack {
            Image(systemName: "scope")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Color.accentColor)
            Text("Add Docking Constraint")
                .font(.body.weight(.semibold))
            Spacer()
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }

    // MARK: - Source Info

    @ViewBuilder
    private var sourceInfoSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionLabel("Target", icon: context.sourceType == .receptor ? "building.columns" : "hexagon")

            HStack(spacing: 8) {
                if let atomName = context.atomName, let resName = context.residueName {
                    Text("\(atomName) @ \(resName)")
                        .font(.callout.monospaced().weight(.medium))
                } else if let resName = context.residueName {
                    Text(resName)
                        .font(.callout.monospaced().weight(.medium))
                } else {
                    Text("Selection")
                        .font(.callout.weight(.medium))
                }

                if let chain = context.chainID, !chain.isEmpty {
                    Text("Chain \(chain)")
                        .font(.footnote)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(.quaternary)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }

                Spacer()

                Text(context.sourceType == .receptor ? "Receptor" : "Ligand")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(context.sourceType == .receptor ? Color.blue.opacity(0.15) : Color.green.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
        }
    }

    // MARK: - Scope

    @ViewBuilder
    private var scopeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Target Scope", icon: "scope")
            Picker("", selection: $targetScope) {
                ForEach(TargetScope.allCases, id: \.self) { scope in
                    Text(scope.rawValue).tag(scope)
                }
            }
            .pickerStyle(.segmented)

            Text(targetScope == .atom
                 ? "Constraint applies to the exact selected atom position."
                 : "Constraint is satisfied if ANY relevant atom in the residue interacts.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Interaction Type

    @ViewBuilder
    private var interactionTypeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Interaction Type", icon: "arrow.triangle.branch")
            Picker("", selection: $interactionType) {
                ForEach(ConstraintInteractionType.allCases, id: \.self) { type in
                    Label(type.rawValue, systemImage: type.icon).tag(type)
                }
            }
            .pickerStyle(.menu)

            Text(interactionTypeDescription)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Strength

    @ViewBuilder
    private var strengthSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Constraint Strength", icon: "gauge.with.dots.needle.33percent")
            Picker("", selection: $strengthMode) {
                ForEach(StrengthMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            if strengthMode == .soft {
                HStack {
                    Text("Penalty")
                        .font(.footnote)
                    Slider(value: $softStrength, in: 1...50, step: 1)
                    Text("\(String(format: "%.0f", softStrength)) kcal/mol/\u{00C5}\u{00B2}")
                        .font(.footnote.monospaced())
                        .frame(width: 90, alignment: .trailing)
                }
            } else {
                Text("Mandatory — poses violating this constraint will be strongly penalized (effective rejection).")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: - Distance

    @ViewBuilder
    private var distanceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Distance Threshold", icon: "ruler")
            HStack {
                Slider(value: $distanceThreshold, in: 2.0...5.0, step: 0.1)
                Text("\(String(format: "%.1f", distanceThreshold)) \u{00C5}")
                    .font(.subheadline.monospaced())
                    .frame(width: 45, alignment: .trailing)
            }
            Text("Maximum distance for the constraint to be considered satisfied.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            Button("Add Constraint") {
                addConstraint()
                dismiss()
            }
            .keyboardShortcut(.defaultAction)
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    // MARK: - Helpers

    @ViewBuilder
    private func sectionLabel(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.primary)
    }

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

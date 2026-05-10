// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Sidebar Ligands tab: compact summary + quick actions + "Open Database" button.
/// The full database management is in the LigandDatabaseWindow (Cmd+L).
struct LigandDatabaseView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var smilesInput: String = ""
    @State private var showAddConfirmation = false

    private var db: LigandDatabase { viewModel.ligandDB }

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            databaseCard
            if !db.entries.isEmpty {
                quickSelectCard
            }
            Spacer(minLength: 0)
        }
        .padding(12)
    }

    // MARK: - Database card (status + add + actions)

    @ViewBuilder
    private var databaseCard: some View {
        PanelCard(
            "Ligand Database",
            icon: "tray.full",
            accessory: {
                if db.count > 0 {
                    countBadge
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 10) {
                activeLigandRow
                quickAddSection
                PanelLabeledDivider(title: "Database")
                actionButtons
            }
        }
    }

    @ViewBuilder
    private var countBadge: some View {
        let prepared = db.entries.filter(\.isPrepared).count
        let total = db.topLevelEntries.count
        HStack(spacing: 4) {
            Text("\(total)")
                .font(PanelStyle.monoSmall.weight(.bold))
                .foregroundStyle(.secondary)
            Image(systemName: "arrow.right")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Text("\(prepared)")
                .font(PanelStyle.monoSmall.weight(.bold))
                .foregroundStyle(.green)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Color.accentColor.opacity(0.1))
        .clipShape(Capsule())
        .help("\(total) molecules imported, \(prepared) ligands ready for docking")
    }

    @ViewBuilder
    private var activeLigandRow: some View {
        if let lig = viewModel.molecules.ligand {
            PanelHighlightRow(color: .green) {
                HStack(spacing: 6) {
                    Image(systemName: "hexagon.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.green)
                    Text(lig.name)
                        .font(PanelStyle.bodyFont.weight(.medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(lig.atomCount) atoms")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                    Button(action: { viewModel.removeLigandFromView() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Remove ligand from view and database")
                    .plainButtonAccessibility(AccessibilityID.ligClearLigand)
                }
            }
        } else {
            PanelHighlightRow(color: .orange) {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.orange)
                    Text("No active ligand")
                        .font(PanelStyle.bodyFont)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var quickAddSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Add SMILES")
            HStack(spacing: 6) {
                PastableTextField(
                    text: $smilesInput,
                    placeholder: "Paste SMILES…",
                    font: .monospacedSystemFont(ofSize: 11, weight: .regular),
                    onSubmit: { if !smilesInput.isEmpty { addFromSMILES() } }
                )
                .accessibilityIdentifier(AccessibilityID.ligSmilesField)

                if showAddConfirmation {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.green)
                        .frame(width: 60)
                        .transition(.scale.combined(with: .opacity))
                } else {
                    PanelSecondaryButton(
                        title: "Add",
                        icon: "plus",
                        isDisabled: smilesInput.isEmpty,
                        help: "Add SMILES to ligand database"
                    ) { addFromSMILES() }
                    .accessibilityIdentifier(AccessibilityID.ligAddSmiles)
                    .frame(width: 60)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: showAddConfirmation)
        }
    }

    @ViewBuilder
    private var actionButtons: some View {
        VStack(spacing: 6) {
            PanelSecondaryButton(
                title: "Open Database Manager",
                icon: "tablecells",
                help: "Open full database window (Cmd+L)"
            ) { openWindow(id: "ligand-database") }
            .accessibilityIdentifier(AccessibilityID.ligOpenManager)

            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "Save", icon: "square.and.arrow.down",
                    isDisabled: db.count == 0,
                    help: "Save ligand database to disk"
                ) {
                    db.save()
                    viewModel.log.success("Saved \(db.count) ligands", category: .molecule)
                }
                .accessibilityIdentifier(AccessibilityID.ligSaveDB)

                PanelSecondaryButton(
                    title: "Load", icon: "square.and.arrow.up",
                    help: "Load ligand database from disk"
                ) { db.load() }
                .accessibilityIdentifier(AccessibilityID.ligLoadDB)
            }
        }
    }

    // MARK: - Quick select card

    @ViewBuilder
    private var quickSelectCard: some View {
        let prepared = db.entries.filter { $0.isPrepared }
        let unprepared = db.entries.filter { !$0.isPrepared }

        PanelCard(
            "Quick Select",
            icon: "hand.tap",
            accessory: {
                if prepared.count > 0 {
                    Text("\(prepared.count)")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: 6) {
                if prepared.isEmpty && !unprepared.isEmpty {
                    PanelHint(
                        text: "\(unprepared.count) molecules not yet prepared — open Database Manager and run Populate & Prepare",
                        icon: "exclamationmark.triangle",
                        color: .orange
                    )
                } else {
                    ForEach(prepared.prefix(8)) { entry in
                        quickPickRow(entry)
                    }
                    if prepared.count > 8 {
                        Text("+ \(prepared.count - 8) more — open Database Manager")
                            .font(PanelStyle.smallFont)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 2)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func quickPickRow(_ entry: LigandEntry) -> some View {
        let isActive = viewModel.molecules.ligand?.name == entry.name
        HStack(spacing: 6) {
            Circle()
                .fill(isActive ? Color.green : Color.blue.opacity(0.5))
                .frame(width: 6, height: 6)

            Text(entry.name)
                .font(PanelStyle.smallFont.weight(.medium))
                .lineLimit(1)

            Spacer()

            if let d = entry.descriptors {
                Text(String(format: "%.0f Da", d.molecularWeight))
                    .font(PanelStyle.monoSmall)
                    .foregroundStyle(.secondary)
            }

            Button(action: { useAsLigand(entry) }) {
                Text(isActive ? "Active" : "Use")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(isActive ? Color.green : Color.accentColor)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill((isActive ? Color.green : Color.accentColor).opacity(0.14))
                    )
            }
            .buttonStyle(.plain)
            .disabled(isActive)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(isActive ? Color.green.opacity(0.10) : Color.clear)
        )
    }

    // MARK: - Actions

    private func addFromSMILES() {
        let name = "Ligand_\(db.count + 1)"
        db.addFromSMILES(smilesInput, name: name)
        smilesInput = ""
        showAddConfirmation = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            showAddConfirmation = false
        }
    }

    private func useAsLigand(_ entry: LigandEntry) {
        guard !entry.atoms.isEmpty else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare it first", category: .molecule)
            return
        }
        let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(mol, entryID: entry.id)
    }
}

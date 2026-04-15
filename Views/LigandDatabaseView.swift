// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Sidebar Ligands tab: compact summary + quick actions + "Open Database" button.
/// The full database management is in the LigandDatabaseWindow (Cmd+L).
struct LigandDatabaseView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var smilesInput: String = ""

    private var db: LigandDatabase { viewModel.ligandDB }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with molecule/ligand counts
            HStack {
                Label("Ligand Database", systemImage: "tray.full")
                    .font(.callout.weight(.semibold))
                Spacer()
                if db.count > 0 {
                    let prepared = db.entries.filter(\.isPrepared).count
                    let total = db.topLevelEntries.count
                    HStack(spacing: 4) {
                        Text("\(total)")
                            .font(.footnote.monospaced().weight(.bold))
                            .foregroundStyle(.secondary)
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text("\(prepared)")
                            .font(.footnote.monospaced().weight(.bold))
                            .foregroundStyle(.green)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.1))
                    .clipShape(Capsule())
                    .help("\(total) molecules imported, \(prepared) ligands ready for docking")
                }
            }

            // Active ligand badge
            if let lig = viewModel.molecules.ligand {
                HStack(spacing: 4) {
                    Image(systemName: "hexagon.fill")
                        .font(.footnote)
                        .foregroundStyle(.green)
                    Text(lig.name)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(lig.atomCount) atoms")
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                    Button(action: { viewModel.removeLigandFromView() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Remove ligand from view and database")
                    .plainButtonAccessibility(AccessibilityID.ligClearLigand)
                }
                .padding(8)
                .background(Color.green.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                    Text("No active ligand")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(Color.orange.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            // Quick SMILES add
            HStack(spacing: 4) {
                PastableTextField(
                    text: $smilesInput,
                    placeholder: "SMILES",
                    font: .monospacedSystemFont(ofSize: 11, weight: .regular),
                    onSubmit: { if !smilesInput.isEmpty { addFromSMILES() } }
                )
                .accessibilityIdentifier(AccessibilityID.ligSmilesField)

                if showAddConfirmation {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.body)
                        .foregroundStyle(.green)
                        .transition(.scale.combined(with: .opacity))
                } else {
                    Button("Add") { addFromSMILES() }
                        .controlSize(.mini)
                        .disabled(smilesInput.isEmpty)
                        .help("Add SMILES to ligand database")
                        .accessibilityIdentifier(AccessibilityID.ligAddSmiles)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: showAddConfirmation)

            // Open full database manager
            Button(action: { openWindow(id: "ligand-database") }) {
                Label("Open Database Manager", systemImage: "tablecells")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Open full database window (Cmd+L)")
            .accessibilityIdentifier(AccessibilityID.ligOpenManager)

            // Save/Load
            HStack(spacing: 4) {
                Button(action: {
                    db.save()
                    viewModel.log.success("Saved \(db.count) ligands", category: .molecule)
                }) {
                    Label("Save", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(db.count == 0)
                .help("Save ligand database to disk")
                .accessibilityIdentifier(AccessibilityID.ligSaveDB)

                Button(action: {
                    db.load()
                }) {
                    Label("Load", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .help("Load ligand database from disk")
                .accessibilityIdentifier(AccessibilityID.ligLoadDB)
            }

            Divider()

            // Quick-pick: show up to 8 prepared ligands for fast docking selection
            if db.entries.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "tray")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                    Text("Database empty")
                        .font(.footnote)
                        .foregroundStyle(.tertiary)
                    Text("Add SMILES or open Database Manager")
                        .font(.footnote)
                        .foregroundStyle(.quaternary)
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 8)
            } else {
                Text("Quick select for docking:")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)

                let prepared = db.entries.filter { $0.isPrepared }
                let unprepared = db.entries.filter { !$0.isPrepared }

                if prepared.isEmpty && !unprepared.isEmpty {
                    Text("\(unprepared.count) molecules not yet prepared — open Database Manager and run Populate & Prepare")
                        .font(.footnote)
                        .foregroundStyle(.orange.opacity(0.8))
                }

                ForEach(prepared.prefix(8)) { entry in
                    quickPickRow(entry)
                }

                if prepared.count > 8 {
                    Text("+ \(prepared.count - 8) more — open Database Manager")
                        .font(.footnote)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(12)
    }

    // MARK: - Quick Pick Row

    @ViewBuilder
    private func quickPickRow(_ entry: LigandEntry) -> some View {
        let isActive = viewModel.molecules.ligand?.name == entry.name
        HStack(spacing: 4) {
            Circle()
                .fill(isActive ? .green : .blue.opacity(0.5))
                .frame(width: 6, height: 6)

            Text(entry.name)
                .font(.footnote.weight(.medium))
                .lineLimit(1)

            Spacer()

            if let d = entry.descriptors {
                Text(String(format: "%.0f Da", d.molecularWeight))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            Button(action: { useAsLigand(entry) }) {
                Text(isActive ? "Active" : "Use")
                    .font(.footnote)
            }
            .controlSize(.mini)
            .buttonStyle(.bordered)
            .tint(isActive ? .green : .accentColor)
            .disabled(isActive)
        }
        .padding(.vertical, 2)
        .padding(.horizontal, 4)
        .background(isActive ? Color.green.opacity(0.06) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    // MARK: - Actions

    @State private var showAddConfirmation = false

    private func addFromSMILES() {
        let name = "Ligand_\(db.count + 1)"
        db.addFromSMILES(smilesInput, name: name)
        smilesInput = ""
        // Flash confirmation
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

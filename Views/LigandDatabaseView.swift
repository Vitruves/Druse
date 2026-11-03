// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Step 3 — Ligands: the active ligand, quick SMILES entry and a searchable
/// library picker. Full library management (prepare, enumerate, import,
/// analogs) lives in the Ligand Database window (⌘L).
struct LigandDatabaseView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var smilesInput: String = ""
    @State private var lastAddedName: String?
    @State private var libraryFilter: String = ""

    private var db: LigandDatabase { viewModel.ligandDB }

    /// Library rows shown inline before deferring to the Manager window.
    private let libraryRowLimit = 12

    var body: some View {
        DSStepScaffold {
            activeLigandCard
            addLigandCard
            libraryCard
        } footer: {
            footer
        }
    }

    // MARK: - Active ligand

    /// The library entry backing the active ligand (tracked ID first, then name).
    private var activeEntry: LigandEntry? {
        guard let lig = viewModel.molecules.ligand else { return nil }
        if let id = viewModel.activeLigandEntryID,
           let entry = db.entries.first(where: { $0.id == id }) {
            return entry
        }
        return db.entries.first { $0.name == lig.name }
    }

    @ViewBuilder
    private var activeLigandCard: some View {
        PanelCard("Active Ligand", icon: "hexagon") {
            if let lig = viewModel.molecules.ligand {
                let entry = activeEntry
                let smiles = lig.smiles ?? entry?.smiles ?? ""
                VStack(alignment: .leading, spacing: DS.Space.m) {
                    HStack(spacing: DS.Space.s) {
                        Circle()
                            .fill(DS.Color.ligand)
                            .frame(width: 8, height: 8)
                        Text(lig.name)
                            .font(DS.Font.sectionTitle)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: DS.Space.xs)
                        Button {
                            viewModel.removeLigandFromView()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(DS.Font.body)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                        .help("Remove ligand from view and database")
                        .accessibilityLabel("Remove active ligand")
                        .accessibilityIdentifier(AccessibilityID.ligClearLigand)
                    }

                    if !smiles.isEmpty {
                        LigandDepictionView(smiles: smiles)
                            .frame(height: 120)

                        Text(smiles)
                            .font(DS.Font.mono)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                            .help(smiles)
                    }

                    FlowLayout(spacing: DS.Space.xs, lineSpacing: DS.Space.xs) {
                        if let d = entry?.descriptors {
                            DSTag(String(format: "MW %.0f", d.molecularWeight))
                            DSTag("HBD \(d.hbd) · HBA \(d.hba)")
                            DSTag("RotB \(d.rotatableBonds)")
                            DSTag(d.lipinski ? "Ro5 pass" : "Ro5 fail",
                                  tone: d.lipinski ? .success : .warning,
                                  systemImage: d.lipinski ? "checkmark" : "exclamationmark.triangle")
                        }
                        DSTag("\(lig.heavyAtomCount) heavy atoms")
                    }
                }
            } else {
                HStack(alignment: .top, spacing: DS.Space.m) {
                    Image(systemName: "hexagon.dashed")
                        .font(.title2)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.tertiary)
                    VStack(alignment: .leading, spacing: DS.Space.xxs) {
                        Text("No active ligand")
                            .font(DS.Font.label.weight(.medium))
                        Text("Paste a SMILES below or pick a prepared compound from the library.")
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: - Add ligand

    @ViewBuilder
    private var addLigandCard: some View {
        PanelCard("Add Ligand", icon: "plus.circle") {
            VStack(alignment: .leading, spacing: DS.Space.s) {
                HStack(spacing: DS.Space.s) {
                    PastableTextField(
                        text: $smilesInput,
                        placeholder: "Paste SMILES…",
                        font: .monospacedSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular),
                        onSubmit: { if !trimmedInput.isEmpty { addFromSMILES() } }
                    )
                    .accessibilityIdentifier(AccessibilityID.ligSmilesField)

                    Button(action: addFromSMILES) {
                        Label("Add", systemImage: "plus")
                    }
                    .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                    .disabled(trimmedInput.isEmpty)
                    .help("Add SMILES to the ligand library")
                    .accessibilityIdentifier(AccessibilityID.ligAddSmiles)
                }

                // One hint slot whose content swaps in place. Two views
                // cross-fading in a VStack overlap while the removed one
                // animates out, garbling both lines of text.
                PanelHint(
                    text: lastAddedName.map { "Added \($0) to the library — prepare it in the Manager before docking." }
                        ?? "New compounds are added unprepared — run Prepare in the Manager before docking.",
                    icon: lastAddedName == nil ? "info.circle" : "checkmark.circle.fill",
                    color: lastAddedName == nil ? .secondary : DS.Color.success
                )
            }
            .task(id: lastAddedName) {
                // Restartable: each new add cancels the previous dismissal.
                guard lastAddedName != nil else { return }
                try? await Task.sleep(for: .seconds(2.5))
                guard !Task.isCancelled else { return }
                lastAddedName = nil
            }
        }
    }

    private var trimmedInput: String {
        smilesInput.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Library

    private var filteredLibrary: [LigandEntry] {
        let query = libraryFilter.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return db.entries }
        return db.entries.filter {
            $0.name.localizedCaseInsensitiveContains(query) ||
            $0.originalSMILES.localizedCaseInsensitiveContains(query)
        }
    }

    @ViewBuilder
    private var libraryCard: some View {
        let total = db.count
        let preparedCount = db.entries.filter(\.isPrepared).count

        PanelCard("Library", icon: "books.vertical", accessory: {
            if total > 0 {
                DSTag("\(preparedCount)/\(total) ready", tone: preparedCount > 0 ? .success : .neutral)
                    .help("\(total) compounds in the library, \(preparedCount) prepared for docking")
            }
        }) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                if total == 0 {
                    Text("The library is empty. Add a SMILES above, or open the Manager to draw or import compounds.")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    if total > 6 {
                        librarySearchField
                    }
                    libraryList
                    if preparedCount == 0 {
                        PanelHint(
                            text: "No compound is prepared yet — open the Manager and run Prepare.",
                            icon: "exclamationmark.triangle.fill",
                            color: DS.Color.warning
                        )
                    }
                }

                Button { openWindow(id: "ligand-database") } label: {
                    Label("Open Ligand Manager…", systemImage: "tablecells")
                }
                .buttonStyle(.dsSecondary)
                .help("Open the Ligand Database window (⌘L)")
                .accessibilityIdentifier(AccessibilityID.ligOpenManager)

                PanelChoiceGrid(columns: 2) {
                    Button {
                        db.save()
                        viewModel.log.success("Saved \(db.count) ligands", category: .molecule)
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.dsSecondary)
                    .disabled(total == 0)
                    .help("Save the ligand library to Application Support")
                    .accessibilityIdentifier(AccessibilityID.ligSaveDB)

                    Button { db.load() } label: {
                        Label("Load", systemImage: "folder")
                    }
                    .buttonStyle(.dsSecondary)
                    .help("Reload the saved ligand library")
                    .accessibilityIdentifier(AccessibilityID.ligLoadDB)
                }
            }
        }
    }

    private var librarySearchField: some View {
        HStack(spacing: DS.Space.xs) {
            Image(systemName: "magnifyingglass")
                .font(DS.Font.secondary)
                .foregroundStyle(.tertiary)
            TextField("Filter by name or SMILES", text: $libraryFilter)
                .textFieldStyle(.plain)
                .font(DS.Font.label)
            if !libraryFilter.isEmpty {
                Button { libraryFilter = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.borderless)
                .help("Clear filter")
                .accessibilityLabel("Clear filter")
            }
        }
        .padding(.horizontal, DS.Space.m)
        .frame(height: DS.Size.buttonHeight)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(DS.Color.controlFill)
        )
    }

    @ViewBuilder
    private var libraryList: some View {
        let entries = filteredLibrary
        if entries.isEmpty {
            Text("No compound matches “\(libraryFilter)”.")
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
        } else {
            VStack(spacing: DS.Space.xxs) {
                ForEach(Array(entries.prefix(libraryRowLimit).enumerated()), id: \.element.id) { index, entry in
                    libraryRow(entry, index: index)
                }
            }
            if entries.count > libraryRowLimit {
                Button { openWindow(id: "ligand-database") } label: {
                    Text("+ \(entries.count - libraryRowLimit) more in the Manager")
                        .font(DS.Font.secondary)
                }
                .buttonStyle(.borderless)
                .help("Open the Ligand Database window (⌘L)")
            }
        }
    }

    @ViewBuilder
    private func libraryRow(_ entry: LigandEntry, index: Int) -> some View {
        let isActive = viewModel.activeLigandEntryID == entry.id
            || (viewModel.activeLigandEntryID == nil && viewModel.molecules.ligand?.name == entry.name)
        let canUse = !entry.atoms.isEmpty

        HStack(spacing: DS.Space.s) {
            Circle()
                .fill(isActive ? DS.Color.ligand : Color.clear)
                .overlay(Circle().strokeBorder(DS.Color.ligand.opacity(isActive ? 1 : 0.6), lineWidth: 1))
                .frame(width: 8, height: 8)

            VStack(alignment: .leading, spacing: 0) {
                Text(entry.name)
                    .font(DS.Font.label.weight(isActive ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let d = entry.descriptors {
                    Text(String(format: "%.0f Da", d.molecularWeight))
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }

            Spacer(minLength: DS.Space.xs)

            statusTag(entry)

            Button {
                useAsLigand(entry)
            } label: {
                Text(isActive ? "Active" : "Use")
                    .font(DS.Font.secondary.weight(.medium))
                    .frame(minWidth: 36)
            }
            .buttonStyle(.borderless)
            // An explicit tint overrides the disabled dimming, so only tint
            // buttons that can actually be pressed.
            .foregroundStyle(isActive ? DS.Color.ligand : canUse ? DS.Color.accent : Color.secondary.opacity(0.5))
            .disabled(isActive || !canUse)
            .help(isActive ? "This is the active ligand"
                  : canUse ? "Use \(entry.name) as the docking ligand"
                  : "Prepare \(entry.name) in the Manager first")
            .accessibilityIdentifier(AccessibilityID.ligUseEntry(index))
        }
        .padding(.horizontal, DS.Space.s)
        .padding(.vertical, DS.Space.xs)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(isActive ? DS.Color.ligand.opacity(0.10) : Color.clear)
        )
        .contentShape(Rectangle())
    }

    @ViewBuilder
    private func statusTag(_ entry: LigandEntry) -> some View {
        if entry.preparationError != nil {
            DSTag("Failed", tone: .danger)
                .help(entry.preparationError ?? "")
        } else if entry.isPrepared {
            DSTag("Ready", tone: .success)
        } else {
            DSTag("Raw")
        }
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: DS.Space.m) {
            if let lig = viewModel.molecules.ligand {
                Label {
                    Text(lig.name).lineLimit(1).truncationMode(.middle)
                } icon: {
                    Circle().fill(DS.Color.ligand).frame(width: 8, height: 8)
                }
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                Spacer(minLength: DS.Space.s)
                DSNextStepButton(.bindingSite)
                    .fixedSize()
            } else {
                Text("Choose a ligand to continue to the binding site.")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Actions

    private func addFromSMILES() {
        let smiles = trimmedInput
        guard !smiles.isEmpty else { return }
        let name = "Ligand_\(db.count + 1)"
        db.addFromSMILES(smiles, name: name)
        smilesInput = ""
        lastAddedName = name
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

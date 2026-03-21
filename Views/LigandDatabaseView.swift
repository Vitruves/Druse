import SwiftUI

/// Sidebar Ligands tab: compact summary + quick actions + "Open Database" button.
/// The full database management is in the LigandDatabaseWindow (Cmd+L).
struct LigandDatabaseView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.openWindow) private var openWindow
    @State private var smilesInput: String = ""

    private var db: LigandDatabase { viewModel.ligandDB }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header
            HStack {
                Label("Ligands", systemImage: "tray.full")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                if db.count > 0 {
                    Text("\(db.count)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .clipShape(Capsule())
                }
            }

            // Active ligand badge
            if let lig = viewModel.ligand {
                HStack(spacing: 4) {
                    Image(systemName: "hexagon.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.green)
                    Text(lig.name)
                        .font(.system(size: 10, weight: .medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(lig.atomCount) atoms")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Button(action: { viewModel.clearLigand() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(5)
                .background(Color.green.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.orange)
                    Text("No active ligand")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .padding(5)
                .background(Color.orange.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }

            // Quick SMILES add
            HStack(spacing: 4) {
                TextField("SMILES", text: $smilesInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 10, design: .monospaced))

                Button("Add") { addFromSMILES() }
                    .controlSize(.mini)
                    .disabled(smilesInput.isEmpty)
            }

            // Open full database manager
            Button(action: { openWindow(id: "ligand-database") }) {
                Label("Open Database Manager", systemImage: "tablecells")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Open full database window (Cmd+L)")

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

                Button(action: {
                    db.load()
                }) {
                    Label("Load", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            Divider()

            // Quick-pick: show up to 8 prepared ligands for fast docking selection
            if db.entries.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "tray")
                        .font(.system(size: 20))
                        .foregroundStyle(.tertiary)
                    Text("Database empty")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                    Text("Add SMILES or open Database Manager")
                        .font(.system(size: 9))
                        .foregroundStyle(.quaternary)
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 8)
            } else {
                Text("Quick select for docking:")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)

                let prepared = db.entries.filter { $0.isPrepared }
                let unprepared = db.entries.filter { !$0.isPrepared }

                if prepared.isEmpty && !unprepared.isEmpty {
                    Text("\(unprepared.count) ligands not prepared — open Database Manager to prepare")
                        .font(.system(size: 9))
                        .foregroundStyle(.orange.opacity(0.8))
                }

                ForEach(prepared.prefix(8)) { entry in
                    quickPickRow(entry)
                }

                if prepared.count > 8 {
                    Text("+ \(prepared.count - 8) more — open Database Manager")
                        .font(.system(size: 9))
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
        let isActive = viewModel.ligand?.name == entry.name
        HStack(spacing: 4) {
            Circle()
                .fill(isActive ? .green : .blue.opacity(0.5))
                .frame(width: 6, height: 6)

            Text(entry.name)
                .font(.system(size: 10, weight: .medium))
                .lineLimit(1)

            Spacer()

            if let d = entry.descriptors {
                Text(String(format: "%.0f", d.molecularWeight))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            Button(action: { useAsLigand(entry) }) {
                Text(isActive ? "Active" : "Use")
                    .font(.system(size: 9))
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

    private func addFromSMILES() {
        let name = "Ligand_\(db.count + 1)"
        db.addFromSMILES(smilesInput, name: name)
        smilesInput = ""
    }

    private func useAsLigand(_ entry: LigandEntry) {
        guard !entry.atoms.isEmpty else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare it first", category: .molecule)
            return
        }
        let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(mol)
    }
}

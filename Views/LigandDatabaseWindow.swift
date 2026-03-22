import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Ligand Database Manager Window

struct LigandDatabaseWindow: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    // Selection state (shift+click multi-select)
    @State private var selectedIDs: Set<UUID> = []
    @State private var lastClickedID: UUID?

    // Search / filter
    @State private var searchText: String = ""
    @State private var lipinskiFilter: Bool = false

    // Inline SMILES entry
    @State private var showSMILESEntry: Bool = false
    @State private var smilesInput: String = ""
    @State private var nameInput: String = ""

    // Processing state
    @State private var isProcessing: Bool = false
    @State private var processingMessage: String = ""

    // Sorting
    enum SortField: String { case name, smiles, mw, logP, hbd, hba, tpsa, rotB, atoms }
    @State private var sortField: SortField? = nil
    @State private var sortAscending: Bool = true

    // Detail panel: selected entry for inspection
    @State private var inspectedEntry: LigandEntry?

    // Preparation options (used in detail panel)
    @State private var prepAddHydrogens = true
    @State private var prepMinimize = true
    @State private var prepComputeCharges = true
    @State private var prepNumConformers = 50

    // Conformer state
    @State private var conformers: [ConformerEntry] = []
    @State private var selectedConformerIndex: Int = 0
    @State private var isGeneratingConformers = false

    struct ConformerEntry: Identifiable {
        let id: Int
        let molecule: MoleculeData
        let energy: Double
    }

    private var db: LigandDatabase { viewModel.ligandDB }

    private var filteredEntries: [LigandEntry] {
        var entries = db.entries
        if !searchText.isEmpty {
            entries = entries.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.smiles.localizedCaseInsensitiveContains(searchText)
            }
        }
        if lipinskiFilter {
            entries = entries.filter { $0.descriptors?.lipinski == true }
        }
        if let field = sortField {
            entries.sort { a, b in
                let result: Bool
                switch field {
                case .name:  result = a.name.localizedCompare(b.name) == .orderedAscending
                case .smiles: result = a.smiles < b.smiles
                case .mw:    result = (a.descriptors?.molecularWeight ?? 0) < (b.descriptors?.molecularWeight ?? 0)
                case .logP:  result = (a.descriptors?.logP ?? 0) < (b.descriptors?.logP ?? 0)
                case .hbd:   result = (a.descriptors?.hbd ?? 0) < (b.descriptors?.hbd ?? 0)
                case .hba:   result = (a.descriptors?.hba ?? 0) < (b.descriptors?.hba ?? 0)
                case .tpsa:  result = (a.descriptors?.tpsa ?? 0) < (b.descriptors?.tpsa ?? 0)
                case .rotB:  result = (a.descriptors?.rotatableBonds ?? 0) < (b.descriptors?.rotatableBonds ?? 0)
                case .atoms: result = a.atoms.count < b.atoms.count
                }
                return sortAscending ? result : !result
            }
        }
        return entries
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            toolbar
            Divider()

            // Main content: table + detail panel
            if db.entries.isEmpty {
                emptyState
            } else {
                HSplitView {
                    tableView
                        .frame(minWidth: 600)

                    if let entry = inspectedEntry {
                        detailPanel(entry)
                            .frame(minWidth: 320, idealWidth: 400, maxWidth: 500)
                    }
                }
            }

            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onChange(of: selectedIDs) { _, newIDs in
            // Auto-inspect when a single entry is selected
            if newIDs.count == 1, let id = newIDs.first {
                if inspectedEntry?.id != id {
                    inspectedEntry = db.entries.first { $0.id == id }
                    conformers = []
                    selectedConformerIndex = 0
                }
            }
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 8) {
            // Add SMILES
            Button(action: { showSMILESEntry.toggle() }) {
                Label("Add SMILES", systemImage: "plus")
            }
            .controlSize(.small)

            // Import buttons
            Menu {
                Button("Import .smi file") { importSMIFile() }
                Button("Import .csv file") { importCSVFile() }
                Button("Import .sdf file") { importSDFFile() }
            } label: {
                Label("Import", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)

            Divider().frame(height: 16)

            // Bulk actions (enabled when selection exists)
            Button(action: { prepareSelected() }) {
                Label("Prepare", systemImage: "wand.and.stars")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty || isProcessing)
            .help("Prepare selected ligands (3D + minimize + charges)")

            Button(action: { deleteSelected() }) {
                Label("Delete", systemImage: "trash")
            }
            .controlSize(.small)
            .foregroundStyle(.red)
            .disabled(selectedIDs.isEmpty)
            .help("Delete selected ligands")

            Button(action: { useSelectedForDocking() }) {
                Label(selectedIDs.count > 1 ? "Queue \(selectedIDs.count) for Docking" : "Use for Docking",
                      systemImage: "arrow.right.circle")
            }
            .controlSize(.small)
            .disabled(selectedIDs.isEmpty)
            .help(selectedIDs.count > 1
                  ? "Queue \(selectedIDs.count) ligands for batch docking (launch from Docking tab)"
                  : "Set selected ligand as active for docking")

            Spacer()

            // Search
            TextField("Search...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 180)
                .font(.system(size: 11))

            Toggle("Lipinski", isOn: $lipinskiFilter)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .font(.system(size: 10))

            // Save/Load
            Menu {
                Button("Save Database") {
                    db.save()
                    viewModel.log.success("Saved \(db.count) ligands", category: .molecule)
                }
                Button("Load Database") {
                    db.load()
                }
                Divider()
                Button("Export All as SDF") { db.exportSDF() }
                Divider()
                Button("Clear All", role: .destructive) {
                    db.removeAll()
                    selectedIDs.removeAll()
                    inspectedEntry = nil
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .controlSize(.small)

            // Select all / none
            Button(action: { toggleSelectAll() }) {
                Image(systemName: selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty
                      ? "checkmark.square.fill" : "square")
                    .font(.system(size: 12))
            }
            .buttonStyle(.plain)
            .help("Select all / none")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    // MARK: - SMILES Entry (inline, shown below toolbar)

    @ViewBuilder
    private var smilesEntryBar: some View {
        if showSMILESEntry {
            HStack(spacing: 8) {
                PastableTextField(text: $smilesInput, placeholder: "SMILES", font: .monospacedSystemFont(ofSize: 11, weight: .regular))
                    .frame(height: 22)

                TextField("Name", text: $nameInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11))
                    .frame(width: 120)

                Button("Add") {
                    let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
                    db.addFromSMILES(smilesInput, name: name)
                    smilesInput = ""
                    nameInput = ""
                }
                .controlSize(.small)
                .disabled(smilesInput.isEmpty)

                Button("Add & Prepare") {
                    let name = nameInput.isEmpty ? "Ligand_\(db.count + 1)" : nameInput
                    addAndPrepare(smiles: smilesInput, name: name)
                    smilesInput = ""
                    nameInput = ""
                }
                .controlSize(.small)
                .disabled(smilesInput.isEmpty)

                Button(action: { showSMILESEntry = false }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
        }
    }

    // MARK: - Table

    private var tableView: some View {
        VStack(spacing: 0) {
            smilesEntryBar

            // Processing indicator
            if isProcessing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(processingMessage)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .frame(maxWidth: .infinity)
                .background(Color.accentColor.opacity(0.05))
            }

            // Table header (click to sort)
            HStack(spacing: 0) {
                headerCell("", width: 30, field: nil)
                headerCell("Name", width: 140, field: .name)
                headerCell("SMILES", width: nil, field: .smiles)
                headerCell("MW", width: 60, field: .mw)
                headerCell("LogP", width: 50, field: .logP)
                headerCell("HBD", width: 36, field: .hbd)
                headerCell("HBA", width: 36, field: .hba)
                headerCell("TPSA", width: 50, field: .tpsa)
                headerCell("RotB", width: 36, field: .rotB)
                headerCell("Lip.", width: 30, field: nil)
                headerCell("Ki", width: 55, field: nil)
                headerCell("pKi", width: 45, field: nil)
                headerCell("IC50", width: 55, field: nil)
                headerCell("Prep", width: 36, field: nil)
                headerCell("Atoms", width: 44, field: .atoms)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))

            Divider()

            // Table body
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredEntries) { entry in
                        tableRow(entry)
                        Divider().opacity(0.3)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func headerCell(_ title: String, width: CGFloat?, field: SortField?) -> some View {
        let isSorted = field != nil && sortField == field
        Group {
            if let field {
                Button(action: {
                    if sortField == field {
                        sortAscending.toggle()
                    } else {
                        sortField = field
                        sortAscending = true
                    }
                }) {
                    HStack(spacing: 2) {
                        Text(title)
                        if isSorted {
                            Image(systemName: sortAscending ? "chevron.up" : "chevron.down")
                                .font(.system(size: 7))
                        }
                    }
                }
                .buttonStyle(.plain)
            } else {
                Text(title)
            }
        }
        .font(.system(size: 9, weight: isSorted ? .bold : .semibold))
        .foregroundStyle(isSorted ? .primary : .secondary)
        .frame(width: width, alignment: .leading)
        .frame(maxWidth: width == nil ? .infinity : nil, alignment: .leading)
    }

    @ViewBuilder
    private func tableRow(_ entry: LigandEntry) -> some View {
        let isSelected = selectedIDs.contains(entry.id)
        let isActive = viewModel.ligand?.name == entry.name
        let isInspected = inspectedEntry?.id == entry.id

        HStack(spacing: 0) {
            // Checkbox
            Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                .font(.system(size: 11))
                .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                .frame(width: 30)

            // Name
            HStack(spacing: 4) {
                if isActive {
                    Circle().fill(.green).frame(width: 5, height: 5)
                }
                Text(entry.name)
                    .font(.system(size: 10, weight: isActive ? .semibold : .regular))
                    .lineLimit(1)
            }
            .frame(width: 140, alignment: .leading)

            // SMILES
            Text(entry.smiles)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(entry.smiles)

            // Descriptors
            if let d = entry.descriptors {
                Text(String(format: "%.0f", d.molecularWeight))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 60, alignment: .trailing)
                Text(String(format: "%.1f", d.logP))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.hbd)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Text("\(d.hba)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Text(String(format: "%.0f", d.tpsa))
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 50, alignment: .trailing)
                Text("\(d.rotatableBonds)")
                    .font(.system(size: 9, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
                Image(systemName: d.lipinski ? "checkmark" : "xmark")
                    .font(.system(size: 8))
                    .foregroundStyle(d.lipinski ? .green : .red.opacity(0.6))
                    .frame(width: 30)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 60, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 50, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 50, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 36, alignment: .trailing)
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 30, alignment: .center)
            }

            // Affinity data (Ki, pKi, IC50)
            if let ki = entry.ki {
                Text(String(format: "%.1f", ki))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 55, alignment: .trailing)
            }
            if let pKi = entry.pKi ?? entry.effectivePKi {
                Text(String(format: "%.2f", pKi))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple)
                    .frame(width: 45, alignment: .trailing)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 45, alignment: .trailing)
            }
            if let ic50 = entry.ic50 {
                Text(String(format: "%.1f", ic50))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple)
                    .frame(width: 55, alignment: .trailing)
            } else {
                Text("—").font(.system(size: 9)).foregroundStyle(.tertiary).frame(width: 55, alignment: .trailing)
            }

            // Prepared status
            Image(systemName: entry.isPrepared ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 9))
                .foregroundColor(entry.isPrepared ? .green : .gray)
                .frame(width: 36)

            // Atom count
            Text(entry.atoms.isEmpty ? "—" : "\(entry.atoms.count)")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(entry.atoms.isEmpty ? .tertiary : .primary)
                .frame(width: 44, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(isInspected ? Color.accentColor.opacity(0.2) :
                     isSelected ? Color.accentColor.opacity(0.1) :
                     isActive ? Color.green.opacity(0.05) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowClick(entry.id, shiftKey: NSEvent.modifierFlags.contains(.shift),
                          cmdKey: NSEvent.modifierFlags.contains(.command))
        }
        .contextMenu {
            Button("Inspect") {
                inspectedEntry = entry
                conformers = []
                selectedConformerIndex = 0
            }
            Divider()
            Button("Use for Docking") { useEntryForDocking(entry) }
                .disabled(entry.atoms.isEmpty)
            Button("Prepare") { prepareSingleEntry(entry) }
                .disabled(entry.smiles.isEmpty)
            Divider()
            Button("Delete", role: .destructive) {
                if inspectedEntry?.id == entry.id { inspectedEntry = nil }
                db.remove(id: entry.id)
                selectedIDs.remove(entry.id)
            }
        }
    }

    // MARK: - Detail Panel (structure viz, preparation, conformers)

    @ViewBuilder
    private func detailPanel(_ entry: LigandEntry) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Header
                HStack {
                    Label(entry.name, systemImage: "hexagon.fill")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Button(action: { inspectedEntry = nil; conformers = [] }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }

                // SMILES
                if !entry.smiles.isEmpty {
                    Text(entry.smiles)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }

                Divider()

                // Structure visualization
                structureView(entry)

                Divider()

                // Descriptors
                if let d = entry.descriptors {
                    descriptorCard(d)
                    Divider()
                }

                // Binding Affinity Data
                affinitySection(entry)
                Divider()

                // Preparation options & action
                if !entry.smiles.isEmpty {
                    preparationSection(entry)
                    Divider()
                }

                // Conformer browser
                if !conformers.isEmpty {
                    conformerBrowser(entry)
                    Divider()
                }

                // Actions
                actionButtons(entry)
            }
            .padding(12)
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    // MARK: - Structure Visualization

    @ViewBuilder
    private func structureView(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Structure", systemImage: "atom")
                .font(.system(size: 11, weight: .medium))

            if !entry.atoms.isEmpty {
                // 3D atom stats
                let heavyAtoms = entry.atoms.filter { $0.element != .H }
                HStack(spacing: 12) {
                    statBadge("Atoms", "\(entry.atoms.count)")
                    statBadge("Heavy", "\(heavyAtoms.count)")
                    statBadge("Bonds", "\(entry.bonds.count)")
                    if entry.conformerCount > 1 {
                        statBadge("Conf.", "\(entry.conformerCount)")
                    }
                }

                // Simple 2D projection of 3D coords
                Canvas { context, size in
                    drawMolecule2D(entry: entry, context: context, size: size)
                }
                .frame(height: 180)
                .background(Color.black.opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 9))
                        .foregroundStyle(.orange)
                    Text("No 3D coordinates — prepare this ligand first")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(Color.orange.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func drawMolecule2D(entry: LigandEntry, context: GraphicsContext, size: CGSize) {
        let atoms = entry.atoms
        let bonds = entry.bonds
        guard !atoms.isEmpty else { return }

        // Project 3D → 2D using XY plane with padding
        let xs = atoms.map { CGFloat($0.position.x) }
        let ys = atoms.map { CGFloat($0.position.y) }
        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!
        let rangeX = max(maxX - minX, 0.1)
        let rangeY = max(maxY - minY, 0.1)
        let padding: CGFloat = 20
        let drawW = size.width - padding * 2
        let drawH = size.height - padding * 2
        let scale = min(drawW / rangeX, drawH / rangeY)
        let cx = size.width / 2
        let cy = size.height / 2
        let midX = (minX + maxX) / 2
        let midY = (minY + maxY) / 2

        func project(_ atom: Atom) -> CGPoint {
            CGPoint(
                x: cx + (CGFloat(atom.position.x) - midX) * scale,
                y: cy - (CGFloat(atom.position.y) - midY) * scale  // flip Y
            )
        }

        // Draw bonds
        for bond in bonds {
            guard bond.atomIndex1 < atoms.count, bond.atomIndex2 < atoms.count else { continue }
            let p1 = project(atoms[bond.atomIndex1])
            let p2 = project(atoms[bond.atomIndex2])
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(.gray.opacity(0.6)), lineWidth: 1.5)
        }

        // Draw atoms (only heavy atoms to keep it clean)
        for atom in atoms {
            if atom.element == .H { continue }
            let p = project(atom)
            let radius: CGFloat = 4
            let color = elementColor(atom.element)
            context.fill(Circle().path(in: CGRect(x: p.x - radius, y: p.y - radius, width: radius * 2, height: radius * 2)),
                        with: .color(color))
        }
    }

    private func elementColor(_ element: Element) -> Color {
        switch element {
        case .C:  return .white.opacity(0.8)
        case .N:  return .blue
        case .O:  return .red
        case .S:  return .yellow
        case .P:  return .orange
        case .F:  return .green
        case .Cl: return .green.opacity(0.7)
        case .Br: return .brown
        default:  return .gray
        }
    }

    // MARK: - Descriptor Card

    @ViewBuilder
    private func descriptorCard(_ d: LigandDescriptors) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Properties", systemImage: "list.bullet.rectangle")
                .font(.system(size: 11, weight: .medium))

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 12) {
                    propBadge("MW", String(format: "%.1f", d.molecularWeight))
                    propBadge("LogP", String(format: "%.2f", d.logP))
                    propBadge("TPSA", String(format: "%.1f", d.tpsa))
                }
                HStack(spacing: 12) {
                    propBadge("HBD", "\(d.hbd)")
                    propBadge("HBA", "\(d.hba)")
                    propBadge("RotB", "\(d.rotatableBonds)")
                    propBadge("Rings", "\(d.rings)")
                }
                HStack(spacing: 8) {
                    ruleTag("Lipinski", d.lipinski)
                    ruleTag("Veber", d.veber)
                    Text("fCSP3: \(String(format: "%.2f", d.fractionCSP3))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    // MARK: - Affinity Section

    @ViewBuilder
    private func affinitySection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Binding Affinity", systemImage: "chart.line.uptrend.xyaxis")
                .font(.system(size: 11, weight: .medium))

            VStack(alignment: .leading, spacing: 4) {
                affinityField("Ki (nM)", value: entry.ki, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].ki = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
                affinityField("pKi", value: entry.pKi, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].pKi = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
                affinityField("IC50 (nM)", value: entry.ic50, entryId: entry.id) { newVal in
                    if let idx = db.entries.firstIndex(where: { $0.id == entry.id }) {
                        db.entries[idx].ic50 = newVal
                        inspectedEntry = db.entries[idx]
                    }
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            if let ePKi = entry.effectivePKi {
                HStack(spacing: 4) {
                    Text("Effective pKi:")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.2f", ePKi))
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(.purple)
                }
            }
        }
    }

    @ViewBuilder
    private func affinityField(_ label: String, value: Float?, entryId: UUID, onUpdate: @escaping (Float?) -> Void) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 10))
                .frame(width: 65, alignment: .leading)

            let textBinding = Binding<String>(
                get: { value.map { String(format: "%.2f", $0) } ?? "" },
                set: { newText in
                    let trimmed = newText.trimmingCharacters(in: .whitespaces)
                    onUpdate(trimmed.isEmpty ? nil : Float(trimmed))
                }
            )
            TextField("—", text: textBinding)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 10, design: .monospaced))
                .frame(width: 90)

            if value != nil {
                Button(action: { onUpdate(nil) }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Preparation Section

    @ViewBuilder
    private func preparationSection(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Preparation", systemImage: "wand.and.stars")
                .font(.system(size: 11, weight: .medium))

            Toggle("Add Hydrogens", isOn: $prepAddHydrogens)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            Toggle("MMFF94 Minimization", isOn: $prepMinimize)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            Toggle("Gasteiger Charges", isOn: $prepComputeCharges)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.system(size: 11))

            HStack {
                Text("Conformers:")
                    .font(.system(size: 11))
                Picker("", selection: $prepNumConformers) {
                    Text("1").tag(1)
                    Text("10").tag(10)
                    Text("50").tag(50)
                    Text("100").tag(100)
                    Text("200").tag(200)
                }
                .pickerStyle(.segmented)
                .frame(width: 200)
            }

            HStack(spacing: 8) {
                Button(action: { prepareInspectedEntry(entry) }) {
                    Label("Prepare", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(entry.smiles.isEmpty || isProcessing || isGeneratingConformers)

                if entry.isPrepared && prepNumConformers > 1 {
                    Button(action: { generateConformersForEntry(entry) }) {
                        Label("Generate \(prepNumConformers) Conformers", systemImage: "cube.transparent")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(isGeneratingConformers || isProcessing)
                }
            }

            if isProcessing || isGeneratingConformers {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(isGeneratingConformers ? "Generating conformers..." : "Preparing...")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Conformer Browser

    @ViewBuilder
    private func conformerBrowser(_ entry: LigandEntry) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Conformers (\(conformers.count))", systemImage: "cube.transparent")
                    .font(.system(size: 11, weight: .medium))

                Spacer()

                Button("Use Best") { applyBestConformer(entry) }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(conformers.isEmpty)

                Button("Use Selected") { applySelectedConformer(entry) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
                    .disabled(conformers.isEmpty)
            }

            // Conformer visualization: show energy-sorted bars
            ScrollView(.vertical) {
                VStack(spacing: 2) {
                    ForEach(conformers) { conf in
                        conformerRow(conf)
                    }
                }
            }
            .frame(maxHeight: 180)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    @ViewBuilder
    private func conformerRow(_ conf: ConformerEntry) -> some View {
        let isSelected = conf.id == selectedConformerIndex
        let isBest = conf.id == (conformers.first?.id ?? -1)

        HStack(spacing: 8) {
            Text("#\(conf.id + 1)")
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .frame(width: 30, alignment: .trailing)

            // Energy bar
            let minEnergy = conformers.map(\.energy).min() ?? 0
            let maxEnergy = conformers.map(\.energy).max() ?? 1
            let range = max(maxEnergy - minEnergy, 0.1)
            let normalized = (conf.energy - minEnergy) / range

            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 2)
                    .fill(energyColor(normalized))
                    .frame(width: max(4, geo.size.width * CGFloat(1.0 - normalized)))
            }
            .frame(height: 8)

            Text(String(format: "%.2f", conf.energy))
                .font(.system(size: 10, design: .monospaced))
                .frame(width: 60, alignment: .trailing)

            Text("kcal/mol")
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)

            if isBest {
                Text("Best")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.green)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .onTapGesture {
            selectedConformerIndex = conf.id
            // Update the structure preview with this conformer's coords
            if var updated = inspectedEntry {
                updated.atoms = conf.molecule.atoms
                updated.bonds = conf.molecule.bonds
                inspectedEntry = updated
            }
        }
    }

    private func energyColor(_ normalized: Double) -> Color {
        if normalized < 0.3 { return .green }
        if normalized < 0.6 { return .yellow }
        return .orange
    }

    // MARK: - Action Buttons

    @ViewBuilder
    private func actionButtons(_ entry: LigandEntry) -> some View {
        HStack(spacing: 8) {
            Button(action: { useEntryForDocking(entry) }) {
                Label("Use for Docking", systemImage: "arrow.right.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(entry.atoms.isEmpty)
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "tray")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("No ligands in database")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Add SMILES, or import .smi / .csv / .sdf files")
                .font(.system(size: 12))
                .foregroundStyle(.tertiary)

            Button(action: { showSMILESEntry = true }) {
                Label("Add SMILES", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)

            smilesEntryBar
            Spacer()
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack(spacing: 16) {
            Text("Total: \(db.count)")
                .font(.system(size: 10))
            Text("Prepared: \(db.entries.filter(\.isPrepared).count)")
                .font(.system(size: 10))
                .foregroundStyle(.green)
            Text("Selected: \(selectedIDs.count)")
                .font(.system(size: 10))
                .foregroundStyle(Color.accentColor)

            Spacer()

            if let lig = viewModel.ligand {
                HStack(spacing: 4) {
                    Circle().fill(.green).frame(width: 5, height: 5)
                    Text("Active: \(lig.name)")
                        .font(.system(size: 10, weight: .medium))
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    // MARK: - Selection

    private func handleRowClick(_ id: UUID, shiftKey: Bool, cmdKey: Bool) {
        if shiftKey, let lastID = lastClickedID {
            let allIDs = filteredEntries.map(\.id)
            if let startIdx = allIDs.firstIndex(of: lastID),
               let endIdx = allIDs.firstIndex(of: id) {
                let range = min(startIdx, endIdx)...max(startIdx, endIdx)
                for i in range {
                    selectedIDs.insert(allIDs[i])
                }
            }
        } else if cmdKey {
            if selectedIDs.contains(id) {
                selectedIDs.remove(id)
            } else {
                selectedIDs.insert(id)
            }
        } else {
            selectedIDs = [id]
        }
        lastClickedID = id
    }

    private func toggleSelectAll() {
        if selectedIDs.count == filteredEntries.count && !filteredEntries.isEmpty {
            selectedIDs.removeAll()
        } else {
            selectedIDs = Set(filteredEntries.map(\.id))
        }
    }

    // MARK: - Actions

    private func prepareSelected() {
        let selected = db.entries.filter { selectedIDs.contains($0.id) }
        guard !selected.isEmpty else { return }

        let toProcess = selected.filter { !$0.smiles.isEmpty || !$0.name.isEmpty }
        guard !toProcess.isEmpty else { return }

        isProcessing = true
        processingMessage = "Preparing \(toProcess.count) ligands..."

        Task {
            var successCount = 0
            for (i, entry) in toProcess.enumerated() {
                processingMessage = "Preparing \(i+1)/\(toProcess.count): \(entry.name)"

                let candidates: [(smiles: String, name: String)]
                if !entry.smiles.isEmpty {
                    candidates = [
                        (entry.smiles, entry.name),
                        (entry.name, entry.smiles)
                    ]
                } else {
                    candidates = [(entry.name, "Ligand_\(i+1)")]
                }

                var prepared = false
                for candidate in candidates {
                    let smi = candidate.smiles
                    let nm = candidate.name
                    let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges
                    let (mol, desc, _) = await Task.detached { @Sendable in
                        RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                                  numConformers: 1, addHydrogens: ah,
                                                  minimize: mn, computeCharges: cc)
                    }.value

                    if let mol {
                        var updated = entry
                        if smi == entry.name && !entry.smiles.isEmpty {
                            updated.smiles = entry.name
                            updated.name = entry.smiles
                        }
                        updated.atoms = mol.atoms
                        updated.bonds = mol.bonds
                        updated.descriptors = desc
                        updated.isPrepared = true
                        updated.conformerCount = 1
                        db.update(updated)
                        successCount += 1
                        prepared = true
                        break
                    }
                }

                if !prepared {
                    viewModel.log.error("Failed to prepare \(entry.name): invalid SMILES", category: .molecule)
                }
            }
            isProcessing = false
            processingMessage = ""
            // Refresh inspected entry if it was prepared
            if let inspID = inspectedEntry?.id {
                inspectedEntry = db.entries.first { $0.id == inspID }
            }
            if successCount > 0 {
                viewModel.log.success("Prepared \(successCount)/\(toProcess.count) ligands", category: .molecule)
            } else {
                viewModel.log.error("All \(toProcess.count) preparations failed — check SMILES validity", category: .molecule)
            }
        }
    }

    private func deleteSelected() {
        for id in selectedIDs {
            db.remove(id: id)
        }
        let count = selectedIDs.count
        if let inspID = inspectedEntry?.id, selectedIDs.contains(inspID) {
            inspectedEntry = nil
            conformers = []
        }
        selectedIDs.removeAll()
        viewModel.log.info("Deleted \(count) ligands", category: .molecule)
    }

    private func useSelectedForDocking() {
        let selectedEntries = db.entries.filter { selectedIDs.contains($0.id) }
        guard !selectedEntries.isEmpty else { return }

        if selectedEntries.count == 1 {
            useEntryForDocking(selectedEntries[0])
        } else {
            viewModel.queueLigandsForBatchDocking(selectedEntries)
        }
    }

    private func useEntryForDocking(_ entry: LigandEntry) {
        guard !entry.atoms.isEmpty else {
            viewModel.log.warn("\(entry.name) has no 3D coordinates — prepare first", category: .molecule)
            return
        }
        let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)
        viewModel.setLigandForDocking(mol)
    }

    private func prepareSingleEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty || !entry.name.isEmpty else { return }
        isProcessing = true
        processingMessage = "Preparing \(entry.name)..."

        Task {
            let candidates: [(smiles: String, name: String)]
            if !entry.smiles.isEmpty {
                candidates = [(entry.smiles, entry.name), (entry.name, entry.smiles)]
            } else {
                candidates = [(entry.name, "Ligand")]
            }

            let ah = prepAddHydrogens, mn = prepMinimize, cc = prepComputeCharges

            var prepared = false
            for candidate in candidates {
                let smi = candidate.smiles
                let nm = candidate.name
                let (mol, desc, _) = await Task.detached { @Sendable in
                    RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                              numConformers: 1, addHydrogens: ah,
                                              minimize: mn, computeCharges: cc)
                }.value

                if let mol {
                    var updated = entry
                    if smi == entry.name && !entry.smiles.isEmpty {
                        updated.smiles = entry.name
                        updated.name = entry.smiles
                    }
                    updated.atoms = mol.atoms
                    updated.bonds = mol.bonds
                    updated.descriptors = desc
                    updated.isPrepared = true
                    updated.conformerCount = 1
                    db.update(updated)
                    viewModel.log.success("Prepared \(updated.name)", category: .molecule)
                    prepared = true
                    // Refresh inspected entry
                    if inspectedEntry?.id == entry.id {
                        inspectedEntry = db.entries.first { $0.id == entry.id }
                    }
                    break
                }
            }

            if !prepared {
                viewModel.log.error("Failed to prepare \(entry.name): invalid SMILES", category: .molecule)
            }
            isProcessing = false
            processingMessage = ""
        }
    }

    private func prepareInspectedEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isProcessing = true
        processingMessage = "Preparing \(entry.name)..."
        conformers = []
        selectedConformerIndex = 0

        Task {
            let smi = entry.smiles, nm = entry.name
            let nc = prepNumConformers, ah = prepAddHydrogens
            let mn = prepMinimize, cc = prepComputeCharges
            let (mol, desc, err) = await Task.detached { @Sendable in
                RDKitBridge.prepareLigand(smiles: smi, name: nm,
                                          numConformers: nc, addHydrogens: ah,
                                          minimize: mn, computeCharges: cc)
            }.value

            if let mol {
                var updated = entry
                updated.atoms = mol.atoms
                updated.bonds = mol.bonds
                updated.descriptors = desc
                updated.isPrepared = true
                updated.conformerCount = 1
                db.update(updated)
                inspectedEntry = db.entries.first { $0.id == entry.id }
                viewModel.log.success("Prepared \(nm)", category: .molecule)
            } else if let err {
                viewModel.log.error("Preparation failed: \(err)", category: .molecule)
            }
            isProcessing = false
            processingMessage = ""
        }
    }

    private func generateConformersForEntry(_ entry: LigandEntry) {
        guard !entry.smiles.isEmpty else { return }
        isGeneratingConformers = true

        Task {
            let smi = entry.smiles, nm = entry.name
            let nc = prepNumConformers, mn = prepMinimize
            let results = await Task.detached { @Sendable in
                RDKitBridge.generateConformers(smiles: smi, name: nm, count: nc, minimize: mn)
            }.value

            conformers = results.enumerated().map { idx, pair in
                ConformerEntry(id: idx, molecule: pair.molecule, energy: pair.energy)
            }

            // Auto-select best conformer and update entry
            if let best = conformers.first {
                selectedConformerIndex = best.id
                var updated = entry
                updated.atoms = best.molecule.atoms
                updated.bonds = best.molecule.bonds
                updated.conformerCount = conformers.count
                db.update(updated)
                inspectedEntry = db.entries.first { $0.id == entry.id }
            }

            isGeneratingConformers = false
            viewModel.log.success("Generated \(conformers.count) conformers for \(nm)", category: .molecule)
        }
    }

    private func applyBestConformer(_ entry: LigandEntry) {
        guard let best = conformers.first else { return }
        selectedConformerIndex = best.id
        applyConformer(best, to: entry)
    }

    private func applySelectedConformer(_ entry: LigandEntry) {
        guard selectedConformerIndex < conformers.count else { return }
        let conf = conformers[selectedConformerIndex]
        applyConformer(conf, to: entry)
    }

    private func applyConformer(_ conf: ConformerEntry, to entry: LigandEntry) {
        var updated = entry
        updated.atoms = conf.molecule.atoms
        updated.bonds = conf.molecule.bonds
        db.update(updated)
        inspectedEntry = db.entries.first { $0.id == entry.id }
        viewModel.log.info("Applied conformer #\(conf.id + 1) to \(entry.name)", category: .molecule)
    }

    private func addAndPrepare(smiles: String, name: String) {
        db.addFromSMILES(smiles, name: name)
        if let entry = db.entries.last, entry.smiles == smiles {
            prepareSingleEntry(entry)
        }
    }

    // MARK: - Import

    private func importSMIFile() {
        guard let url = FileImportHandler.showBatchOpenPanel() else { return }
        do {
            try db.importSMIFile(url: url, prepare: false)
            viewModel.log.info("Importing from \(url.lastPathComponent)...", category: .molecule)
        } catch {
            viewModel.log.error("Failed: \(error.localizedDescription)", category: .molecule)
        }
    }

    private func importCSVFile() {
        guard let url = FileImportHandler.showBatchOpenPanel() else { return }
        do {
            try db.importCSV(url: url, prepare: false)
            viewModel.log.info("Importing from \(url.lastPathComponent)...", category: .molecule)
        } catch {
            viewModel.log.error("Failed: \(error.localizedDescription)", category: .molecule)
        }
    }

    private func importSDFFile() {
        guard let url = FileImportHandler.showBatchOpenPanel() else { return }
        do {
            let mols = try SDFParser.parse(url: url)
            for mol in mols {
                let entry = LigandEntry(name: mol.name, smiles: "", atoms: mol.atoms, bonds: mol.bonds, isPrepared: true)
                db.add(entry)
            }
            viewModel.log.success("Imported \(mols.count) ligands from SDF", category: .molecule)
        } catch {
            viewModel.log.error("Failed: \(error.localizedDescription)", category: .molecule)
        }
    }

    // MARK: - Helpers

    private func statBadge(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
    }

    private func propBadge(_ label: String, _ value: String) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
            Text(label)
                .font(.system(size: 7))
                .foregroundStyle(.quaternary)
        }
    }

    private func ruleTag(_ name: String, _ passes: Bool) -> some View {
        HStack(spacing: 2) {
            Image(systemName: passes ? "checkmark.circle.fill" : "xmark.circle")
                .font(.system(size: 9))
                .foregroundStyle(passes ? .green : .red)
            Text(name)
                .font(.system(size: 10))
        }
    }
}

// MARK: - PastableTextField (NSViewRepresentable for reliable paste)

/// An `NSTextField` wrapper that reliably updates the binding on paste, drag-drop,
/// and programmatic text changes -- unlike SwiftUI's `TextField` which can miss
/// paste events until the next interaction.
struct PastableTextField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String = ""
    var font: NSFont = .systemFont(ofSize: 12)

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.placeholderString = placeholder
        field.font = font
        field.isBordered = true
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.lineBreakMode = .byTruncatingTail
        field.delegate = context.coordinator
        field.stringValue = text
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var text: Binding<String>

        init(text: Binding<String>) {
            self.text = text
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text.wrappedValue = field.stringValue
        }
    }
}

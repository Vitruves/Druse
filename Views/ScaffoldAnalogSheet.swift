// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Scaffold Analog Generator Sheet
//
// Modal sheet for generating analogs from a user-drawn scaffold. Two modes:
//
//   • R-group decoration — mark attachment points in Ketcher; enumerate
//     allowed substituents at those positions only.
//   • Whole-molecule transforms — apply a fixed SMARTS transformation table
//     anywhere in the molecule (no R-markers required).
//
// On confirm, generated analogs are pushed into the current LigandDatabase.

struct ScaffoldAnalogSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) private var viewModel

    // Drawer state — we track both SMILES and the raw molfile. The molfile
    // path preserves R-group / attachment-point atoms that Ketcher's SMILES
    // export sometimes strips on aromatic ring atoms.
    @State private var scaffoldSMILES: String = ""
    @State private var scaffoldMolfile: String = ""

    // Config
    @State private var config = ScaffoldGeneratorConfig()
    @State private var scaffoldNamePrefix: String = "scaffold"

    // Generation state
    @State private var analogs: [ScaffoldAnalogGenerator.Analog] = []
    @State private var isGenerating = false
    @State private var generationProgress: Float = 0
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?

    /// Scaffold SMILES actually handed to the generator (prefers molfile→RDKit).
    private var resolvedScaffoldSMILES: String {
        ScaffoldAnalogGenerator.resolveScaffoldSMILES(smiles: scaffoldSMILES, molfile: scaffoldMolfile)
    }

    /// Number of attachment points in the resolved scaffold.
    private var attachmentPointCount: Int {
        ScaffoldAnalogGenerator.findAttachmentPointRanges(in: resolvedScaffoldSMILES).count
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "atom")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.blue)
                Text("Generate Analogs from Scaffold")
                    .font(.headline)
                Spacer()
                Text("Draw a scaffold, mark attachment points with Ketcher's R-group tool, then generate.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            HSplitView {
                // LEFT: Ketcher + live scaffold info
                VStack(spacing: 0) {
                    // Ketcher accepts SMILES paste natively (Cmd+V with a
                    // SMILES on the clipboard), so we don't add our own load
                    // field — just surface the hint.
                    HStack(spacing: 6) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.secondary)
                        Text("Tip: paste a SMILES directly into the editor (⌘V) to load an existing structure instead of drawing from scratch.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    Divider()

                    ChemDrawerView(initialSmiles: nil, mode: .molecule)
                        .onScaffoldChanged { smi, _, mol in
                            scaffoldSMILES = smi
                            scaffoldMolfile = mol
                        }
                    Divider()
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text("Scaffold SMILES:")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(resolvedScaffoldSMILES.isEmpty
                                 ? "— draw a structure —"
                                 : resolvedScaffoldSMILES)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(resolvedScaffoldSMILES.isEmpty ? .secondary : .primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                        }
                        if !resolvedScaffoldSMILES.isEmpty {
                            HStack(spacing: 6) {
                                Image(systemName: attachmentPointCount > 0
                                      ? "checkmark.circle.fill"
                                      : "info.circle")
                                    .foregroundStyle(attachmentPointCount > 0 ? .green : .orange)
                                Text(attachmentPointCount > 0
                                     ? "\(attachmentPointCount) attachment point\(attachmentPointCount == 1 ? "" : "s") detected"
                                     : "No attachment points — use R-group tool for decoration mode, or switch to whole-molecule transforms.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                }
                .frame(minWidth: 520)

                // RIGHT: config + results
                configAndResults
                    .frame(minWidth: 380)
            }

            Divider()

            // Footer
            HStack(spacing: 12) {
                if isGenerating {
                    ProgressView(value: generationProgress)
                        .progressViewStyle(.linear)
                        .frame(width: 180)
                    Button("Stop") { generationTask?.cancel() }
                        .controlSize(.small)
                }

                if let err = errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .lineLimit(2)
                }

                Spacer()

                Button("Cancel") {
                    generationTask?.cancel()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Button {
                    runGeneration()
                } label: {
                    Label("Generate", systemImage: "sparkles")
                }
                .disabled(isGenerating || resolvedScaffoldSMILES.isEmpty)

                Button {
                    addToDatabase()
                } label: {
                    Label("Add \(analogs.count) to Database", systemImage: "plus.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(analogs.isEmpty)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
        .frame(minWidth: 1150, minHeight: 760)
    }

    // MARK: - Config panel + results list

    @ViewBuilder
    private var configAndResults: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    // --- Mode picker ---
                    groupBox("Mode") {
                        Picker("Mode", selection: $config.mode) {
                            ForEach(ScaffoldGenerationMode.allCases, id: \.self) { m in
                                Text(m.rawValue).tag(m)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .controlSize(.small)

                        Text(modeHint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    // --- Generation params ---
                    groupBox("Generation") {
                        HStack(spacing: 6) {
                            Text("Max analogs")
                            TextField("", value: $config.maxAnalogs, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .controlSize(.small)
                                .frame(width: 60)
                                .multilineTextAlignment(.trailing)
                                .onChange(of: config.maxAnalogs) { _, v in
                                    if v < 1 { config.maxAnalogs = 1 }
                                    if v > 10_000 { config.maxAnalogs = 10_000 }
                                }
                            Stepper("", value: $config.maxAnalogs, in: 1...10_000, step: 5)
                                .labelsHidden()
                                .controlSize(.small)
                            Spacer()
                            Text(candidatePoolHint)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Min Tanimoto similarity")
                                Spacer()
                                Text(String(format: "%.2f", config.minSimilarity))
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            Slider(value: $config.minSimilarity, in: 0...0.95, step: 0.05)
                        }

                        Toggle("Lipinski filter", isOn: $config.filterLipinski)
                            .controlSize(.small)

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Max LogP")
                                Spacer()
                                Text(String(format: "%.1f", config.maxLogP))
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            Slider(value: $config.maxLogP, in: 1...10, step: 0.5)
                        }

                        HStack {
                            Text("Name prefix")
                            TextField("prefix", text: $scaffoldNamePrefix)
                                .textFieldStyle(.roundedBorder)
                                .controlSize(.small)
                        }
                    }

                    // --- Allowed groups (per category, only meaningful in R-group mode) ---
                    if config.mode == .rgroupDecoration {
                        ForEach(FunctionalGroup.Category.allCases, id: \.self) { category in
                            groupBox("Allowed — \(category.rawValue)") {
                                let groups = FunctionalGroupCatalog.all.filter { $0.category == category }
                                HStack {
                                    Button("All") { toggleCategoryAllow(category, enabled: true) }
                                        .controlSize(.mini)
                                    Button("None") { toggleCategoryAllow(category, enabled: false) }
                                        .controlSize(.mini)
                                    Spacer()
                                }
                                groupCheckGrid(groups: groups, selection: allowedBinding)
                            }
                        }
                    }

                    // --- Denied substructures ---
                    groupBox("Deny — exclude any analog containing…") {
                        groupCheckGrid(groups: FunctionalGroupCatalog.all, selection: deniedBinding)
                    }
                }
                .padding(12)
            }

            Divider()

            // Results preview
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Results")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Text("\(analogs.count) analogs")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)

                if analogs.isEmpty {
                    Text(isGenerating ? "Generating..." : "No analogs yet. Draw a scaffold and click Generate.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 24)
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(analogs.enumerated()), id: \.offset) { idx, a in
                                HStack {
                                    Text("\(idx + 1).")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                        .frame(width: 30, alignment: .trailing)
                                    Text(a.smiles)
                                        .font(.system(size: 10, design: .monospaced))
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Spacer()
                                    Text(a.label)
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundStyle(.blue)
                                        .frame(width: 90, alignment: .trailing)
                                        .lineLimit(1)
                                    Text(String(format: "T=%.2f", a.similarity))
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                    Text(String(format: "MW=%.0f", a.descriptors.molecularWeight))
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                    Text(String(format: "logP=%.1f", a.descriptors.logP))
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 2)
                                .background(idx.isMultiple(of: 2) ? Color.gray.opacity(0.05) : Color.clear)
                            }
                        }
                    }
                    .frame(maxHeight: 220)
                }
            }
        }
    }

    // MARK: - Small helpers

    /// Live upper-bound on the number of candidates the generator could possibly
    /// enumerate (before similarity / descriptor / deny filters are applied).
    /// For R-group decoration: `allowed^sites`. For whole-molecule mode: fixed
    /// at the size of the transformation table (~50 candidate variants).
    private var candidatePoolHint: String {
        switch config.mode {
        case .rgroupDecoration:
            let sites = attachmentPointCount
            let n = config.allowedGroupIDs.count
            guard sites > 0, n > 0 else { return "up to — candidates" }
            // Safe exponentiation with a cap so huge scaffolds don't overflow.
            var total = 1
            for _ in 0..<sites {
                total &*= n
                if total > 1_000_000 || total < 0 { return "up to >1M candidates" }
            }
            return "up to \(total.formatted()) candidate\(total == 1 ? "" : "s")"
        case .wholeMolecule:
            return "up to ~50 candidates"
        }
    }

    private var modeHint: String {
        switch config.mode {
        case .rgroupDecoration:
            return "Requires attachment points on the scaffold (R-group or `>R1` tool in Ketcher). Each R position is decorated with every allowed group; combinations are enumerated."
        case .wholeMolecule:
            return "Ignores R-markers. Applies a fixed set of bioisosteric transformations (halogen swaps, heterocycle swaps, ring expansions, FG exchanges) anywhere in the molecule."
        }
    }

    private var allowedBinding: (FunctionalGroup) -> Binding<Bool> {
        { g in
            Binding(
                get: { config.allowedGroupIDs.contains(g.id) },
                set: { on in
                    if on { config.allowedGroupIDs.insert(g.id) }
                    else { config.allowedGroupIDs.remove(g.id) }
                }
            )
        }
    }

    private var deniedBinding: (FunctionalGroup) -> Binding<Bool> {
        { g in
            Binding(
                get: { config.deniedGroupIDs.contains(g.id) },
                set: { on in
                    if on { config.deniedGroupIDs.insert(g.id) }
                    else { config.deniedGroupIDs.remove(g.id) }
                }
            )
        }
    }

    /// Two-column checkbox grid with left-aligned labels. Uses fixed GridItem
    /// widths so toggles in the same row start at the same x-position.
    @ViewBuilder
    private func groupCheckGrid(groups: [FunctionalGroup],
                                 selection: @escaping (FunctionalGroup) -> Binding<Bool>) -> some View {
        let columns = [
            GridItem(.flexible(minimum: 120), alignment: .leading),
            GridItem(.flexible(minimum: 120), alignment: .leading),
        ]
        LazyVGrid(columns: columns, alignment: .leading, spacing: 3) {
            ForEach(groups) { g in
                Toggle(isOn: selection(g)) {
                    Text(g.displayName)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .toggleStyle(.checkbox)
                .controlSize(.mini)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    private func groupBox<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func toggleCategoryAllow(_ category: FunctionalGroup.Category, enabled: Bool) {
        let ids = FunctionalGroupCatalog.all.filter { $0.category == category }.map { $0.id }
        if enabled { config.allowedGroupIDs.formUnion(ids) }
        else { config.allowedGroupIDs.subtract(ids) }
    }

    // MARK: - Actions

    private func runGeneration() {
        errorMessage = nil
        analogs = []
        isGenerating = true
        generationProgress = 0

        let resolved = resolvedScaffoldSMILES
        let cfg = config

        // Log the actual SMILES handed to the generator — makes it easy to
        // diagnose cases where Ketcher's export dropped an R-group label.
        viewModel.log.info("Scaffold analog generation: \(resolved) (mode: \(cfg.mode.rawValue))",
                           category: .molecule)

        generationTask = Task {
            defer {
                Task { @MainActor in
                    self.isGenerating = false
                    self.generationProgress = 1.0
                }
            }
            do {
                let result = try ScaffoldAnalogGenerator.generate(
                    scaffoldSmiles: resolved,
                    config: cfg,
                    progress: { p in
                        Task { @MainActor in self.generationProgress = p }
                    }
                )
                await MainActor.run {
                    self.analogs = result
                    // Empty result with no thrown error shouldn't happen anymore
                    // (the generator throws specific errors on 0 results), but
                    // keep a fallback message just in case.
                    if result.isEmpty {
                        self.errorMessage = "No analogs produced."
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    private func addToDatabase() {
        guard !analogs.isEmpty else { return }
        let db = viewModel.ligandDB
        let prefix = scaffoldNamePrefix.isEmpty ? "scaffold" : scaffoldNamePrefix
        let startIdx = db.count
        db.batchMutate { entries in
            for (i, a) in analogs.enumerated() {
                var entry = LigandEntry(
                    name: "\(prefix)_\(startIdx + i + 1)",
                    smiles: a.smiles,
                    descriptors: a.descriptors
                )
                entry.parentName = prefix
                entries.append(entry)
            }
        }
        viewModel.log.success("Added \(analogs.count) scaffold analogs to the ligand database", category: .molecule)
        dismiss()
    }
}

// MARK: - ChemDrawerView scaffold callback modifier

private extension ChemDrawerView {
    func onScaffoldChanged(_ handler: @escaping (String, String, String) -> Void) -> ChemDrawerView {
        var copy = self
        copy.onScaffoldChanged = handler
        return copy
    }
}

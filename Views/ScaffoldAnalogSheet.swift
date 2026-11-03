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
        DSSheetScaffold(
            title: "Generate Analogs from Scaffold",
            subtitle: "Draw a scaffold, mark attachment points with Ketcher's R-group tool, then generate.",
            systemImage: "atom"
        ) {
            VStack(spacing: 0) {
                Divider()
                HSplitView {
                    editorPane
                        .frame(minWidth: 520, idealWidth: 640)
                    configAndResults
                        .frame(minWidth: 380, idealWidth: 440)
                }
            }
        } footer: {
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
                    .monospacedDigit()
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
            .disabled(analogs.isEmpty)
        }
        .frame(minWidth: 980, idealWidth: 1100, minHeight: 640, idealHeight: 720)
    }

    // MARK: - Editor pane (Ketcher + live scaffold info)

    @ViewBuilder
    private var editorPane: some View {
        VStack(spacing: 0) {
            // Ketcher accepts SMILES paste natively (⌘V with a SMILES on the
            // clipboard), so we surface the hint instead of a separate load field.
            PanelHint(
                text: "Tip: paste a SMILES directly into the editor (⌘V) to load an existing structure instead of drawing from scratch.",
                icon: "info.circle"
            )
            .padding(.horizontal, DS.Space.l)
            .padding(.vertical, DS.Space.s)
            Divider()

            ChemDrawerView(initialSmiles: nil, mode: .molecule)
                .onScaffoldChanged { smi, _, mol in
                    scaffoldSMILES = smi
                    scaffoldMolfile = mol
                }

            Divider()
            scaffoldStatus
                .padding(.horizontal, DS.Space.l)
                .padding(.vertical, DS.Space.m)
                .background(.bar)
        }
    }

    @ViewBuilder
    private var scaffoldStatus: some View {
        let resolved = resolvedScaffoldSMILES
        VStack(alignment: .leading, spacing: DS.Space.xs) {
            HStack(spacing: DS.Space.s) {
                Text("Scaffold")
                    .font(DS.Font.eyebrow)
                    .tracking(0.6)
                    .textCase(.uppercase)
                    .foregroundStyle(.tertiary)
                Text(resolved.isEmpty ? "Draw a structure" : resolved)
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(resolved.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
            }
            if !resolved.isEmpty {
                let count = attachmentPointCount
                HStack(spacing: DS.Space.s) {
                    Image(systemName: count > 0 ? "checkmark.circle.fill" : "info.circle")
                        .foregroundStyle(count > 0 ? DS.Color.success : DS.Color.warning)
                    Text(count > 0
                         ? "\(count) attachment point\(count == 1 ? "" : "s") detected"
                         : "No attachment points — use the R-group tool for decoration mode, or switch to whole-molecule transforms.")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
                .font(DS.Font.secondary)
            }
        }
    }

    // MARK: - Config form + results list

    @ViewBuilder
    private var configAndResults: some View {
        VStack(spacing: 0) {
            Form {
                Section("Mode") {
                    Picker("Mode", selection: $config.mode) {
                        ForEach(ScaffoldGenerationMode.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Text(modeHint)
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Section {
                    LabeledContent("Max analogs") {
                        HStack(spacing: DS.Space.xs) {
                            TextField("Max analogs", value: $config.maxAnalogs, format: .number)
                                .labelsHidden()
                                .textFieldStyle(.roundedBorder)
                                .monospacedDigit()
                                .multilineTextAlignment(.trailing)
                                .frame(width: DS.Size.numberField)
                                .onChange(of: config.maxAnalogs) { _, v in
                                    if v < 1 { config.maxAnalogs = 1 }
                                    if v > 10_000 { config.maxAnalogs = 10_000 }
                                }
                            Stepper("Max analogs", value: $config.maxAnalogs, in: 1...10_000, step: 5)
                                .labelsHidden()
                        }
                    }
                    sliderRow("Min Tanimoto similarity", value: $config.minSimilarity, in: 0...0.95, step: 0.05, format: "%.2f")
                    Toggle("Lipinski filter", isOn: $config.filterLipinski)
                    sliderRow("Max LogP", value: $config.maxLogP, in: 1...10, step: 0.5, format: "%.1f")
                    TextField("Name prefix", text: $scaffoldNamePrefix, prompt: Text("scaffold"))
                } header: {
                    Text("Generation")
                } footer: {
                    Text(candidatePoolHint)
                        .font(DS.Font.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }

                // Allowed groups are only meaningful in R-group mode.
                if config.mode == .rgroupDecoration {
                    ForEach(FunctionalGroup.Category.allCases, id: \.self) { category in
                        Section {
                            groupCheckGrid(
                                groups: FunctionalGroupCatalog.all.filter { $0.category == category },
                                selection: allowedBinding
                            )
                        } header: {
                            HStack {
                                Text("Allowed — \(category.rawValue)")
                                Spacer()
                                Button("All") { toggleCategoryAllow(category, enabled: true) }
                                Button("None") { toggleCategoryAllow(category, enabled: false) }
                            }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                        }
                    }
                }

                Section("Deny — exclude any analog containing…") {
                    groupCheckGrid(groups: FunctionalGroupCatalog.all, selection: deniedBinding)
                }
            }
            .formStyle(.grouped)

            Divider()
            resultsSection
        }
    }

    @ViewBuilder
    private func sliderRow(_ title: String, value: Binding<Float>, in range: ClosedRange<Float>,
                           step: Float, format: String) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
                .controlSize(.small)
                .labelsHidden()
                .accessibilityLabel(title)
        }
    }

    @ViewBuilder
    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Results", count: analogs.count) {
                if isGenerating {
                    ProgressView(value: generationProgress)
                        .progressViewStyle(.linear)
                        .controlSize(.small)
                        .frame(width: 120)
                    Button("Stop") { generationTask?.cancel() }
                        .controlSize(.small)
                }
            }
            .padding(.horizontal, DS.Space.l)
            .padding(.top, DS.Space.m)

            if let err = errorMessage {
                PanelHint(text: err, icon: "exclamationmark.triangle.fill", color: DS.Color.warning)
                    .padding(.horizontal, DS.Space.l)
            }

            if analogs.isEmpty {
                Text(isGenerating ? "Generating…" : "No analogs yet. Draw a scaffold and click Generate.")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, DS.Space.xxl)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(analogs.enumerated()), id: \.offset) { idx, a in
                            resultRow(index: idx, analog: a)
                        }
                    }
                }
                .frame(maxHeight: 220)
            }
        }
        .padding(.bottom, DS.Space.s)
    }

    @ViewBuilder
    private func resultRow(index: Int, analog a: ScaffoldAnalogGenerator.Analog) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            HStack(spacing: DS.Space.s) {
                Text("\(index + 1)")
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(.tertiary)
                    .frame(width: 28, alignment: .trailing)
                Text(a.smiles)
                    .font(DS.Font.monoSmall)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .help(a.smiles)
            }
            HStack(spacing: DS.Space.xs) {
                DSTag(a.label, tone: .info, monospaced: true)
                Spacer(minLength: DS.Space.xs)
                Text(String(format: "T %.2f · MW %.0f · logP %.1f",
                            a.similarity, a.descriptors.molecularWeight, a.descriptors.logP))
                    .font(DS.Font.monoSmall)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.leading, 28 + DS.Space.s)
        }
        .monospacedDigit()
        .padding(.horizontal, DS.Space.l)
        .padding(.vertical, DS.Space.xs)
        .background(index.isMultiple(of: 2) ? DS.Color.cardFill : Color.clear)
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
            guard sites > 0, n > 0 else { return "Up to — candidates" }
            // Safe exponentiation with a cap so huge scaffolds don't overflow.
            var total = 1
            for _ in 0..<sites {
                total &*= n
                if total > 1_000_000 || total < 0 { return "Up to >1M candidates" }
            }
            return "Up to \(total.formatted()) candidate\(total == 1 ? "" : "s")"
        case .wholeMolecule:
            return "Up to ~50 candidates"
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

    /// Two-column checkbox grid with left-aligned labels. Uses flexible GridItem
    /// minimums so toggles in the same row start at the same x-position.
    @ViewBuilder
    private func groupCheckGrid(groups: [FunctionalGroup],
                                selection: @escaping (FunctionalGroup) -> Binding<Bool>) -> some View {
        let columns = [
            GridItem(.flexible(minimum: 120), alignment: .leading),
            GridItem(.flexible(minimum: 120), alignment: .leading),
        ]
        LazyVGrid(columns: columns, alignment: .leading, spacing: DS.Space.xs) {
            ForEach(groups) { g in
                Toggle(isOn: selection(g)) {
                    Text(g.displayName)
                        .font(DS.Font.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .toggleStyle(.checkbox)
                .controlSize(.small)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
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

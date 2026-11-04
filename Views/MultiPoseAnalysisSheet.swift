// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Sheet presenting two analyses for a multi-selection of docking poses:
///   • Pairwise heavy-atom RMSD matrix (binding-mode convergence check)
///   • Per-residue interaction fingerprint heatmap (contact comparison)
struct MultiPoseAnalysisSheet: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    enum Mode: String, CaseIterable, Identifiable {
        case rmsd = "RMSD"
        case interactions = "Interactions"
        var id: String { rawValue }
    }

    @State private var mode: Mode

    init(initialMode: Mode = .rmsd) {
        self._mode = State(initialValue: initialMode)
    }

    private var indices: [Int] { viewModel.sortedSelectedPoseIndices }

    var body: some View {
        DSSheetScaffold(
            title: "Multi-Pose Analysis",
            subtitle: subtitle,
            systemImage: "square.stack.3d.up"
        ) {
            VStack(spacing: 0) {
                Picker("Analysis", selection: $mode) {
                    ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, DS.Space.xl)
                .padding(.bottom, DS.Space.l)
                Divider()
                Group {
                    switch mode {
                    case .rmsd:         RMSDMatrixView(indices: indices)
                    case .interactions: InteractionFingerprintView(indices: indices)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        } footer: {
            Button("Done") { dismiss() }
                .buttonStyle(DSPrimaryButtonStyle(fullWidth: false))
                .keyboardShortcut(.defaultAction)
        }
        .frame(minWidth: 560, idealWidth: 680, minHeight: 480, idealHeight: 580)
    }

    private var subtitle: String {
        let poses = indices.map { "#\($0 + 1)" }.joined(separator: ", ")
        return "\(indices.count) poses selected · \(poses)"
    }
}

// MARK: - Shared chrome

private struct AnalysisWarning: View {
    let text: String
    var body: some View {
        PanelHighlightRow(color: DS.Color.warning) {
            PanelHint(text: text, icon: "exclamationmark.triangle.fill", color: DS.Color.warning)
        }
    }
}

private struct HeatmapSwatch: View {
    let text: String
    let color: Color
    var body: some View {
        HStack(spacing: DS.Space.xs) {
            RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                .fill(color)
                .frame(width: 12, height: 12)
            Text(text)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - RMSD Matrix

private struct RMSDMatrixView: View {
    @Environment(AppViewModel.self) private var viewModel
    let indices: [Int]

    @State private var matrix: [[Float]] = []
    @State private var atomCountConsistent: Bool = true

    private var poseLabels: [String] {
        indices.map { "#\($0 + 1)" }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                summary
                if matrix.isEmpty {
                    AnalysisWarning(text: atomCountConsistent
                         ? "Need at least 2 poses with transformed coordinates."
                         : "Selected poses have mismatching atom counts (different ligands?). Cannot compute RMSD.")
                } else {
                    matrixGrid
                    legend
                    bindingModes
                }
            }
            .padding(DS.Space.xl)
        }
        .onAppear { recompute() }
        .onChange(of: indices) { _, _ in recompute() }
    }

    private var summary: some View {
        DSMetricGrid(columns: 4) {
            DSMetric("Min RMSD", value: minRMSD.map { String(format: "%.2f", $0) } ?? "—", unit: "Å",
                     tone: minRMSD.map { RMSDBand($0).tone } ?? .neutral)
            DSMetric("Max RMSD", value: maxRMSD.map { String(format: "%.2f", $0) } ?? "—", unit: "Å",
                     tone: maxRMSD.map { RMSDBand($0).tone } ?? .neutral)
            DSMetric("Mean RMSD", value: meanRMSD.map { String(format: "%.2f", $0) } ?? "—", unit: "Å")
            DSMetric("Convergent", value: "\(convergentPairs)/\(totalPairs)",
                     tone: totalPairs > 0 && convergentPairs == totalPairs ? .success : .neutral)
        }
    }

    private var matrixGrid: some View {
        let n = indices.count
        // Grow cells for small selections so the matrix fills the card instead of a corner of it.
        let cellSize: CGFloat = min(64, max(38, 480 / CGFloat(max(n, 1))))
        let labelWidth: CGFloat = 38
        return ScrollView([.horizontal, .vertical]) {
            VStack(alignment: .leading, spacing: 0) {
                // column headers
                HStack(spacing: 0) {
                    Color.clear.frame(width: labelWidth, height: cellSize)
                    ForEach(0..<n, id: \.self) { j in
                        Text(poseLabels[j])
                            .font(DS.Font.monoSmall.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: cellSize, height: cellSize)
                    }
                }
                ForEach(0..<n, id: \.self) { i in
                    HStack(spacing: 0) {
                        Text(poseLabels[i])
                            .font(DS.Font.monoSmall.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: labelWidth, height: cellSize)
                        ForEach(0..<n, id: \.self) { j in
                            cell(i: i, j: j)
                                .frame(width: cellSize, height: cellSize)
                        }
                    }
                }
            }
            .padding(2)
        }
        .frame(maxHeight: 360)
        .dsCard(padding: DS.Space.s)
    }

    @ViewBuilder
    private func cell(i: Int, j: Int) -> some View {
        if i == j {
            Rectangle()
                .fill(DS.Color.controlFill)
                .overlay(Text("—").font(DS.Font.monoSmall).foregroundStyle(.tertiary))
        } else {
            let value = matrix[i][j]
            Rectangle()
                .fill(RMSDBand(value).fill(value))
                .overlay(
                    Text(String(format: "%.2f", value))
                        .font(DS.Font.monoSmall.weight(.medium))
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                )
                .help(String(format: "Pose #%d ↔ #%d : %.2f Å", indices[i] + 1, indices[j] + 1, value))
        }
    }

    private var legend: some View {
        HStack(spacing: DS.Space.l) {
            ForEach(RMSDBand.allCases, id: \.self) { band in
                HeatmapSwatch(text: band.legend, color: band.fill(band.legendSample))
            }
        }
    }

    /// Binding modes = groups of poses linked by ≤ 2 Å RMSD (single linkage), the
    /// question a multi-pose RMSD check is really asking.
    @ViewBuilder
    private var bindingModes: some View {
        let modes = bindingModeGroups
        VStack(alignment: .leading, spacing: DS.Space.s) {
            DSSectionHeader("Binding modes") {
                DSTag("\(modes.count)", monospaced: true)
            }
            Text(modes.count == 1
                 ? "All selected poses share one binding mode (linked by ≤ 2 Å RMSD)."
                 : "\(modes.count) distinct binding modes — poses within a mode are linked by ≤ 2 Å RMSD.")
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            FlowLayout(spacing: DS.Space.s) {
                ForEach(Array(modes.enumerated()), id: \.offset) { number, members in
                    DSTag(
                        "Mode \(number + 1): " + members.map { "#\(indices[$0] + 1)" }.joined(separator: ", "),
                        tone: members.count > 1 ? .success : .neutral,
                        monospaced: true
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    /// Connected components of the ≤ 2 Å graph, in pose-rank order.
    private var bindingModeGroups: [[Int]] {
        let n = matrix.count
        var mode = Array(repeating: -1, count: n)
        var groups: [[Int]] = []
        for start in 0..<n where mode[start] < 0 {
            var members: [Int] = []
            var stack = [start]
            mode[start] = groups.count
            while let i = stack.popLast() {
                members.append(i)
                for j in 0..<n where mode[j] < 0 && matrix[i][j] <= RMSDBand.sameModeCutoff {
                    mode[j] = groups.count
                    stack.append(j)
                }
            }
            groups.append(members.sorted())
        }
        return groups
    }

    private func recompute() {
        matrix = viewModel.computePairwiseRMSD(indices: indices)
        // Distinguish "no poses" from "atom-count mismatch": both produce empty,
        // but the warning message differs. Detect mismatch by checking if all
        // selected poses have transformed coordinates and identical heavy-atom counts.
        let counts = indices.compactMap { idx -> Int? in
            guard idx >= 0, idx < viewModel.docking.dockingResults.count else { return nil }
            let p = viewModel.docking.dockingResults[idx].transformedAtomPositions
            return p.isEmpty ? nil : p.count
        }
        atomCountConsistent = (Set(counts).count <= 1)
    }

    private var pairs: [Float] {
        guard !matrix.isEmpty else { return [] }
        var values: [Float] = []
        for i in 0..<matrix.count {
            for j in (i + 1)..<matrix.count {
                values.append(matrix[i][j])
            }
        }
        return values
    }

    private var minRMSD: Float? { pairs.min() }
    private var maxRMSD: Float? { pairs.max() }
    private var meanRMSD: Float? {
        guard !pairs.isEmpty else { return nil }
        return pairs.reduce(0, +) / Float(pairs.count)
    }
    private var convergentPairs: Int { pairs.filter { $0 <= RMSDBand.sameModeCutoff }.count }
    private var totalPairs: Int { pairs.count }
}

/// The three RMSD bands of the legend; cells, legend swatches and summary tints all derive
/// from this one mapping so a 5 Å cell can never read as "similar".
private enum RMSDBand: CaseIterable {
    case sameMode, similar, distinct

    static let sameModeCutoff: Float = 2.0
    static let similarCutoff: Float = 4.0

    init(_ rmsd: Float) {
        self = rmsd <= Self.sameModeCutoff ? .sameMode : rmsd <= Self.similarCutoff ? .similar : .distinct
    }

    var tone: DSTone {
        switch self {
        case .sameMode: .success
        case .similar:  .warning
        case .distinct: .danger
        }
    }

    var legend: String {
        switch self {
        case .sameMode: "≤ 2 Å (same mode)"
        case .similar:  "2–4 Å (similar)"
        case .distinct: "> 4 Å (distinct)"
        }
    }

    /// A representative value inside the band, for the legend swatch.
    var legendSample: Float {
        switch self {
        case .sameMode: 1.0
        case .similar:  3.0
        case .distinct: 8.0
        }
    }

    /// Translucent band color, a little stronger toward the band's far end; text on top
    /// stays `.primary`, readable in both appearances.
    func fill(_ rmsd: Float) -> Color {
        switch self {
        case .sameMode: tone.color.opacity(0.20 + 0.15 * Double(rmsd / Self.sameModeCutoff))
        case .similar:  tone.color.opacity(0.25 + 0.15 * Double((rmsd - Self.sameModeCutoff) / (Self.similarCutoff - Self.sameModeCutoff)))
        case .distinct: tone.color.opacity(0.30 + 0.20 * Double(min((rmsd - Self.similarCutoff) / 8, 1)))
        }
    }
}

// MARK: - Interaction fingerprint

private struct InteractionFingerprintView: View {
    @Environment(AppViewModel.self) private var viewModel
    let indices: [Int]

    @State private var residueKeys: [String] = []
    @State private var residueLabels: [String: String] = [:]
    @State private var fingerprints: [[String: Set<MolecularInteraction.InteractionType>]] = []
    /// Measured width of the heatmap card's content, which the columns share.
    @State private var heatmapWidth: CGFloat = 520

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                if residueKeys.isEmpty {
                    AnalysisWarning(text: noContactsMessage)
                } else {
                    consensusSummary
                    heatmap
                    legend
                }
            }
            .padding(DS.Space.xl)
        }
        .onAppear { recompute() }
        .onChange(of: indices) { _, _ in recompute() }
    }

    private var noContactsMessage: String {
        if viewModel.docking.scoringMethod.accountedInteractionTypes.isEmpty {
            return "Current scoring method has no per-interaction terms — interaction fingerprints are unavailable. Switch to Vina or Drusina to inspect contacts."
        }
        return "No protein–ligand contacts detected for the selected poses."
    }

    private var consensusSummary: some View {
        let total = indices.count
        let consensus = residueKeys.filter { key in
            fingerprints.allSatisfy { !( $0[key]?.isEmpty ?? true) }
        }
        let majority = residueKeys.filter { key in
            let count = fingerprints.reduce(0) { $0 + (($1[key]?.isEmpty == false) ? 1 : 0) }
            return count > total / 2 && count < total
        }
        return DSMetricGrid(columns: 4) {
            DSMetric("Consensus", value: "\(consensus.count)", tone: .success)
            DSMetric("Majority", value: "\(majority.count)", tone: .warning)
            DSMetric("Residues", value: "\(residueKeys.count)")
            DSMetric("Poses", value: "\(total)")
        }
    }

    private var heatmap: some View {
        let cellHeight: CGFloat = 32
        let headerHeight: CGFloat = 84
        let leftLabelWidth: CGFloat = 64
        // Labels rotated -55° from the bottom-left of the last column reach this far past it.
        let headerOverhang = headerHeight / tan(55 * .pi / 180)
        // Columns share the card's measured width (few residues → wide cells); below 20 pt per
        // column the heatmap keeps 20 pt cells and scrolls sideways.
        let columnsWidth = heatmapWidth - leftLabelWidth - headerOverhang - 4
        let cellWidth: CGFloat = min(44, max(20, (columnsWidth / CGFloat(max(residueKeys.count, 1))).rounded(.down)))
        return ScrollView([.horizontal, .vertical]) {
            VStack(alignment: .leading, spacing: 0) {
                // Column header: residues, rotated -55° around the bottom of
                // each cell so labels read upward from left to right and can
                // exceed the cell width without truncation.
                HStack(spacing: 0) {
                    Color.clear.frame(width: leftLabelWidth, height: headerHeight)
                    ForEach(residueKeys, id: \.self) { key in
                        ZStack(alignment: .bottomLeading) {
                            Color.clear
                            Text(residueLabels[key] ?? key)
                                .font(DS.Font.monoSmall.weight(.medium))
                                .foregroundStyle(.secondary)
                                .fixedSize()
                                .rotationEffect(.degrees(-55), anchor: .bottomLeading)
                                .offset(x: cellWidth / 2, y: -2)
                        }
                        .frame(width: cellWidth, height: headerHeight)
                    }
                }

                // Rows: one per pose
                ForEach(Array(indices.enumerated()), id: \.offset) { row, idx in
                    HStack(spacing: 0) {
                        Text(verbatim: "Pose #\(idx + 1)")
                            .font(DS.Font.monoSmall.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.leading, DS.Space.xs)
                            .frame(width: leftLabelWidth, height: cellHeight, alignment: .leading)
                        ForEach(residueKeys, id: \.self) { key in
                            cell(types: fingerprints[row][key] ?? [])
                                .frame(width: cellWidth, height: cellHeight)
                                .overlay(Rectangle().strokeBorder(DS.Color.hairline, lineWidth: 0.5))
                        }
                    }
                }

                // Footer: how many of the selected poses contact each residue.
                HStack(spacing: 0) {
                    Text("Poses")
                        .font(DS.Font.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, DS.Space.xs)
                        .frame(width: leftLabelWidth, height: cellHeight - 8, alignment: .leading)
                    ForEach(residueKeys, id: \.self) { key in
                        let count = fingerprints.reduce(0) { $0 + (($1[key]?.isEmpty == false) ? 1 : 0) }
                        Text("\(count)")
                            .font(DS.Font.monoSmall.weight(count == indices.count ? .semibold : .regular))
                            .monospacedDigit()
                            .foregroundStyle(count == indices.count ? AnyShapeStyle(DS.Color.success) : AnyShapeStyle(.secondary))
                            .frame(width: cellWidth, height: cellHeight - 8)
                            .help("\(residueLabels[key] ?? key): contacted in \(count) of \(indices.count) poses")
                    }
                }
            }
            .padding(2)
            .padding(.trailing, headerOverhang)
        }
        .frame(maxHeight: 360)
        .onGeometryChange(for: CGFloat.self, of: \.size.width) { _, width in heatmapWidth = width }
        .dsCard(padding: DS.Space.s)
    }

    @ViewBuilder
    private func cell(types: Set<MolecularInteraction.InteractionType>) -> some View {
        if types.isEmpty {
            Rectangle().fill(DS.Color.controlFill)
        } else if types.count == 1, let t = types.first {
            Rectangle()
                .fill(t.themeColor.opacity(0.7))
                .help(t.label)
        } else {
            // Multiple interaction types: split cell vertically with up to 3 stripes
            let sorted = Array(types).sorted { $0.rawValue < $1.rawValue }
            HStack(spacing: 0) {
                ForEach(sorted.prefix(3), id: \.self) { t in
                    Rectangle().fill(t.themeColor.opacity(0.7))
                }
            }
            .help(sorted.map(\.label).joined(separator: ", "))
        }
    }

    private var legend: some View {
        let presentTypes = Set(fingerprints.flatMap { $0.values.flatMap { $0 } })
        let sorted = Array(presentTypes).sorted { $0.rawValue < $1.rawValue }
        return FlowLayout(spacing: DS.Space.l) {
            ForEach(sorted, id: \.self) { t in
                HeatmapSwatch(text: t.label, color: t.themeColor.opacity(0.7))
            }
        }
    }

    private func recompute() {
        let result = viewModel.computeInteractionFingerprints(indices: indices)
        residueKeys = result.residueKeys
        residueLabels = result.residueLabels
        fingerprints = result.fingerprints
    }
}

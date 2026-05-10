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
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            Picker("", selection: $mode) {
                ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            Divider()
            Group {
                switch mode {
                case .rmsd:         RMSDMatrixView(indices: indices)
                case .interactions: InteractionFingerprintView(indices: indices)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 540, minHeight: 460)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Multi-Pose Analysis")
                .font(.system(size: 14, weight: .semibold))
            Text("\(indices.count) poses")
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.primary.opacity(0.08)))
            Spacer()
            Button(action: { dismiss() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 22, height: 22)
                    .background(Color.primary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 5))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
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
            VStack(alignment: .leading, spacing: 14) {
                summary
                if matrix.isEmpty {
                    Text(atomCountConsistent
                         ? "Need at least 2 poses with transformed coordinates."
                         : "Selected poses have mismatching atom counts (different ligands?). Cannot compute RMSD.")
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.orange)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color.orange.opacity(0.10)))
                } else {
                    matrixGrid
                    legend
                }
            }
            .padding(16)
        }
        .onAppear { recompute() }
        .onChange(of: indices) { _, _ in recompute() }
    }

    private var summary: some View {
        HStack(spacing: 8) {
            PanelStat(label: "Min RMSD",
                      value: minRMSD.map { String(format: "%.2f", $0) } ?? "—",
                      unit: "Å",
                      color: .green)
            PanelStat(label: "Max RMSD",
                      value: maxRMSD.map { String(format: "%.2f", $0) } ?? "—",
                      unit: "Å",
                      color: .red)
            PanelStat(label: "Mean",
                      value: meanRMSD.map { String(format: "%.2f", $0) } ?? "—",
                      unit: "Å")
            PanelStat(label: "Convergent",
                      value: "\(convergentPairs)/\(totalPairs)",
                      color: convergentPairs == totalPairs ? .green : .secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.04)))
    }

    private var matrixGrid: some View {
        let n = indices.count
        let cellSize: CGFloat = 38
        let labelWidth: CGFloat = 38
        return ScrollView([.horizontal, .vertical]) {
            VStack(alignment: .leading, spacing: 0) {
                // column headers
                HStack(spacing: 0) {
                    Color.clear.frame(width: labelWidth, height: cellSize)
                    ForEach(0..<n, id: \.self) { j in
                        Text(poseLabels[j])
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: cellSize, height: cellSize)
                    }
                }
                ForEach(0..<n, id: \.self) { i in
                    HStack(spacing: 0) {
                        Text(poseLabels[i])
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
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
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.03)))
    }

    @ViewBuilder
    private func cell(i: Int, j: Int) -> some View {
        if i == j {
            Rectangle()
                .fill(Color.primary.opacity(0.05))
                .overlay(Text("—").font(.system(size: 10)).foregroundStyle(.tertiary))
        } else {
            let value = matrix[i][j]
            Rectangle()
                .fill(rmsdColor(value))
                .overlay(
                    Text(String(format: "%.2f", value))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(value > 4.0 ? .white : .primary)
                )
                .help(String(format: "Pose #%d ↔ #%d : %.2f Å", indices[i] + 1, indices[j] + 1, value))
        }
    }

    private var legend: some View {
        HStack(spacing: 12) {
            legendChip(text: "≤ 2.0 Å (same mode)", color: rmsdColor(1.5))
            legendChip(text: "2–4 Å (similar)", color: rmsdColor(3.0))
            legendChip(text: "> 4 Å (distinct)", color: rmsdColor(5.0))
        }
    }

    private func legendChip(text: String, color: Color) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 3).fill(color).frame(width: 14, height: 14)
            Text(text).font(PanelStyle.smallFont).foregroundStyle(.secondary)
        }
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
    private var convergentPairs: Int { pairs.filter { $0 <= 2.0 }.count }
    private var totalPairs: Int { pairs.count }

    private func rmsdColor(_ v: Float) -> Color {
        if v <= 2.0  { return Color.green.opacity(0.20 + 0.30 * Double(v / 2.0)) }
        if v <= 4.0  { return Color.yellow.opacity(0.30 + 0.30 * Double((v - 2.0) / 2.0)) }
        if v <= 6.0  { return Color.orange.opacity(0.45 + 0.25 * Double((v - 4.0) / 2.0)) }
        return Color.red.opacity(min(1.0, 0.55 + 0.20 * Double((v - 6.0) / 4.0)))
    }
}

// MARK: - Interaction fingerprint

private struct InteractionFingerprintView: View {
    @Environment(AppViewModel.self) private var viewModel
    let indices: [Int]

    @State private var residueKeys: [String] = []
    @State private var residueLabels: [String: String] = [:]
    @State private var fingerprints: [[String: Set<MolecularInteraction.InteractionType>]] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if residueKeys.isEmpty {
                    Text(noContactsMessage)
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.orange)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color.orange.opacity(0.10)))
                } else {
                    consensusSummary
                    heatmap
                    legend
                }
            }
            .padding(16)
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
        return HStack(spacing: 8) {
            PanelStat(label: "Consensus", value: "\(consensus.count)", color: .green)
            PanelStat(label: "Majority", value: "\(majority.count)", color: .yellow)
            PanelStat(label: "Residues", value: "\(residueKeys.count)")
            PanelStat(label: "Poses", value: "\(total)")
        }
        .padding(10)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.04)))
    }

    private var heatmap: some View {
        let cellWidth: CGFloat = 28
        let cellHeight: CGFloat = 28
        let headerHeight: CGFloat = 96
        let leftLabelWidth: CGFloat = 64
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
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
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
                        Text("Pose #\(idx + 1)")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: leftLabelWidth, height: cellHeight, alignment: .leading)
                            .padding(.leading, 4)
                        ForEach(residueKeys, id: \.self) { key in
                            cell(types: fingerprints[row][key] ?? [])
                                .frame(width: cellWidth, height: cellHeight)
                        }
                    }
                }
            }
            .padding(2)
        }
        .frame(maxHeight: 360)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.03)))
    }

    @ViewBuilder
    private func cell(types: Set<MolecularInteraction.InteractionType>) -> some View {
        if types.isEmpty {
            Rectangle().fill(Color.primary.opacity(0.04))
        } else if types.count == 1, let t = types.first {
            Rectangle()
                .fill(typeColor(t).opacity(0.7))
                .help(t.label)
        } else {
            // Multiple interaction types: split cell vertically with up to 3 stripes
            let sorted = Array(types).sorted { $0.rawValue < $1.rawValue }
            HStack(spacing: 0) {
                ForEach(sorted.prefix(3), id: \.self) { t in
                    Rectangle().fill(typeColor(t).opacity(0.7))
                }
            }
            .help(sorted.map(\.label).joined(separator: ", "))
        }
    }

    private var legend: some View {
        let presentTypes = Set(fingerprints.flatMap { $0.values.flatMap { $0 } })
        let sorted = Array(presentTypes).sorted { $0.rawValue < $1.rawValue }
        return FlowLayout(spacing: 6) {
            ForEach(sorted, id: \.self) { t in
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 3).fill(typeColor(t).opacity(0.7))
                        .frame(width: 12, height: 12)
                    Text(t.label).font(PanelStyle.smallFont).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.primary.opacity(0.06)))
            }
        }
    }

    private func typeColor(_ t: MolecularInteraction.InteractionType) -> Color {
        Color(red: Double(t.color.x), green: Double(t.color.y), blue: Double(t.color.z))
    }

    private func recompute() {
        let result = viewModel.computeInteractionFingerprints(indices: indices)
        residueKeys = result.residueKeys
        residueLabels = result.residueLabels
        fingerprints = result.fingerprints
    }
}

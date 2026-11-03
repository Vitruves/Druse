// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Full side-by-side comparison of reference ligand vs selected analog.
/// Shown as a sheet from LeadOptimizationTabView.
struct LeadOptComparisonView: View {
    @Environment(AppViewModel.self) private var viewModel
    let reference: LeadOptimizationState
    let analog: LeadOptAnalog

    private let referenceColor = Color.secondary
    private let analogColor = DS.Color.accent

    var body: some View {
        DSSheetScaffold(
            title: "Lead Optimization Comparison",
            subtitle: "\(reference.referenceName) vs \(analog.name)",
            systemImage: "arrow.left.arrow.right"
        ) {
            ScrollView {
                VStack(alignment: .leading, spacing: DS.Space.l) {
                    energyComparison
                    propertyComparison
                    admetComparison
                }
                .padding(.horizontal, DS.Space.xl)
                .padding(.bottom, DS.Space.xl)
            }
        } footer: {
            Button("Done") { viewModel.leadOpt.showComparison = false }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
    }

    // MARK: - Energy

    @ViewBuilder
    private var energyComparison: some View {
        let refScore = reference.referenceScore
        let anaScore = analog.bestScore
        VStack(alignment: .leading, spacing: DS.Space.m) {
            DSSectionHeader(method.isAffinityScore ? "Docking Affinity" : "Docking Energy") {
                if let refScore, let anaScore {
                    deltaTag(anaScore, reference: refScore)
                }
            }
            HStack(spacing: DS.Space.m) {
                compoundColumn(role: "Reference", name: reference.referenceName, score: refScore, tone: .neutral, dot: referenceColor)
                Image(systemName: "arrow.right")
                    .font(DS.Font.sectionTitle)
                    .foregroundStyle(.tertiary)
                compoundColumn(role: "Analog", name: analog.name, score: anaScore, tone: analogTone(anaScore, ref: refScore), dot: analogColor)
            }
            if anaScore == nil {
                PanelHint(text: "This analog has not been docked yet. Use Dock All Analogs to score it.", icon: "info.circle")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    /// Scoring method shared by the reference pose and the analog mini-docking.
    private var method: ScoringMethod { reference.scoringMethod }

    @ViewBuilder
    private func deltaTag(_ score: Float, reference ref: Float) -> some View {
        let delta = score - ref
        let tone: DSTone = abs(delta) <= 0.05 ? .neutral : method.ranks(score, before: ref) ? .success : .warning
        DSTag("Δ \(String(format: "%+.2f", delta)) \(method.unitLabel)", tone: tone, monospaced: true)
    }

    @ViewBuilder
    private func compoundColumn(role: String, name: String, score: Float?, tone: DSTone, dot: Color) -> some View {
        VStack(spacing: DS.Space.xs) {
            HStack(spacing: DS.Space.xs) {
                Circle().fill(dot).frame(width: 8, height: 8)
                Text(role)
                    .font(DS.Font.eyebrow)
                    .tracking(0.6)
                    .textCase(.uppercase)
                    .foregroundStyle(.secondary)
            }
            Text(name)
                .font(DS.Font.label.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.xxs) {
                Text(score.map { String(format: "%.2f", $0) } ?? "—")
                    .font(DS.Font.metric)
                    .foregroundStyle(tone == .neutral ? Color.primary : tone.color)
                if score != nil {
                    Text(method.unitLabel)
                        .font(DS.Font.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(DS.Space.m)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).fill(DS.Color.controlFill))
        .accessibilityElement(children: .combine)
    }

    private func analogTone(_ value: Float?, ref: Float?) -> DSTone {
        guard let v = value, let r = ref else { return .neutral }
        return method.ranks(v, before: r) ? .success : .warning
    }

    // MARK: - Properties

    @ViewBuilder
    private var propertyComparison: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            DSSectionHeader("Molecular Properties") {
                HStack(spacing: DS.Space.m) {
                    legendItem("Reference", color: referenceColor)
                    legendItem("Analog", color: analogColor)
                }
            }
            if let ref = reference.referenceDescriptors, let ana = analog.descriptors {
                HStack(alignment: .center, spacing: DS.Space.xl) {
                    radarChart(ref: ref, ana: ana)
                        .frame(width: 220, height: 200)
                    propertyDeltaTable(ref: ref, ana: ana)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                PanelHint(text: "Property data not available.", icon: "info.circle")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    @ViewBuilder
    private func legendItem(_ title: String, color: Color) -> some View {
        HStack(spacing: DS.Space.xs) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(title)
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func radarChart(ref: LigandDescriptors, ana: LigandDescriptors) -> some View {
        let axes: [(label: String, ref: Float, ana: Float)] = [
            radarAxis("MW", ref.molecularWeight, ana.molecularWeight, max: 600),
            radarAxis("LogP", ref.logP, ana.logP, max: 6),
            radarAxis("TPSA", ref.tpsa, ana.tpsa, max: 200),
            radarAxis("HBD", Float(ref.hbd), Float(ana.hbd), max: 6),
            radarAxis("HBA", Float(ref.hba), Float(ana.hba), max: 12),
            radarAxis("RotB", Float(ref.rotatableBonds), Float(ana.rotatableBonds), max: 15),
        ]
        let refColor = referenceColor
        let anaColor = analogColor

        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - 22
            let n = axes.count
            let guide = Color.secondary.opacity(0.25)

            func point(_ i: Int, _ r: CGFloat) -> CGPoint {
                let angle = CGFloat(i) / CGFloat(n) * 2 * .pi - .pi / 2
                return CGPoint(x: center.x + cos(angle) * r, y: center.y + sin(angle) * r)
            }

            // Concentric guide rings
            for ring in 1...4 {
                let r = radius * CGFloat(ring) / 4
                var path = Path()
                for i in 0..<n {
                    if i == 0 { path.move(to: point(i, r)) } else { path.addLine(to: point(i, r)) }
                }
                path.closeSubpath()
                context.stroke(path, with: .color(guide), lineWidth: 0.5)
            }

            // Spokes + labels
            for i in 0..<n {
                var spoke = Path()
                spoke.move(to: center)
                spoke.addLine(to: point(i, radius))
                context.stroke(spoke, with: .color(guide), lineWidth: 0.5)
                context.draw(
                    Text(axes[i].label).font(DS.Font.caption).foregroundStyle(.secondary),
                    at: point(i, radius + 12)
                )
            }

            func polygon(_ values: [Float], color: Color) {
                var path = Path()
                for (i, val) in values.enumerated() {
                    let p = point(i, radius * CGFloat(min(max(val, 0), 1)))
                    if i == 0 { path.move(to: p) } else { path.addLine(to: p) }
                }
                path.closeSubpath()
                context.fill(path, with: .color(color.opacity(0.15)))
                context.stroke(path, with: .color(color), lineWidth: 1.5)
            }

            polygon(axes.map(\.ref), color: refColor)
            polygon(axes.map(\.ana), color: anaColor)
        }
        .accessibilityLabel("Property radar chart comparing reference and analog")
    }

    private func radarAxis(_ label: String, _ ref: Float, _ ana: Float, max: Float) -> (label: String, ref: Float, ana: Float) {
        (label, min(ref / max, 1), min(ana / max, 1))
    }

    @ViewBuilder
    private func propertyDeltaTable(ref: LigandDescriptors, ana: LigandDescriptors) -> some View {
        let rows: [(name: String, ref: String, ana: String, delta: Float, deltaText: String)] = [
            ("MW", String(format: "%.0f", ref.molecularWeight), String(format: "%.0f", ana.molecularWeight),
             ana.molecularWeight - ref.molecularWeight, String(format: "%+.0f", ana.molecularWeight - ref.molecularWeight)),
            ("LogP", String(format: "%.2f", ref.logP), String(format: "%.2f", ana.logP),
             ana.logP - ref.logP, String(format: "%+.2f", ana.logP - ref.logP)),
            ("TPSA", String(format: "%.0f", ref.tpsa), String(format: "%.0f", ana.tpsa),
             ana.tpsa - ref.tpsa, String(format: "%+.0f", ana.tpsa - ref.tpsa)),
            ("HBD", "\(ref.hbd)", "\(ana.hbd)", Float(ana.hbd - ref.hbd), String(format: "%+d", ana.hbd - ref.hbd)),
            ("HBA", "\(ref.hba)", "\(ana.hba)", Float(ana.hba - ref.hba), String(format: "%+d", ana.hba - ref.hba)),
            ("RotBonds", "\(ref.rotatableBonds)", "\(ana.rotatableBonds)",
             Float(ana.rotatableBonds - ref.rotatableBonds),
             String(format: "%+d", ana.rotatableBonds - ref.rotatableBonds)),
        ]

        Grid(alignment: .trailing, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.s) {
            GridRow {
                columnHeader("Property").gridColumnAlignment(.leading)
                columnHeader("Ref")
                columnHeader("Analog")
                columnHeader("Δ")
            }

            Divider().gridCellUnsizedAxes(.horizontal)

            ForEach(rows, id: \.name) { row in
                GridRow {
                    Text(row.name)
                        .font(DS.Font.label)
                        .foregroundStyle(.secondary)
                    Text(row.ref).font(DS.Font.mono).monospacedDigit()
                    Text(row.ana).font(DS.Font.mono).monospacedDigit()
                    Text(row.deltaText)
                        .font(DS.Font.mono.weight(.medium))
                        .monospacedDigit()
                        .foregroundStyle(propertyDeltaColor(row.delta))
                }
            }
        }
    }

    private func columnHeader(_ title: String) -> some View {
        Text(title)
            .font(DS.Font.eyebrow)
            .tracking(0.6)
            .textCase(.uppercase)
            .foregroundStyle(.tertiary)
    }

    /// Property deltas carry no good/bad meaning on their own; tint only the direction.
    private func propertyDeltaColor(_ delta: Float) -> Color {
        abs(delta) < 0.005 ? .secondary : delta > 0 ? DS.Color.ligand : DS.Color.info
    }

    // MARK: - ADMET

    @ViewBuilder
    private var admetComparison: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            DSSectionHeader("ADMET Profile")
            if let admet = analog.admet {
                Grid(alignment: .leading, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.s) {
                    admetRow("Lipinski", admet.lipinski ? "Pass" : "Fail", admet.lipinski)
                    admetRow("Veber", admet.veber ? "Pass" : "Fail", admet.veber)
                    if let herg = admet.hergLiability {
                        admetRow("hERG risk", String(format: "%.0f%%", herg * 100), herg < 0.5)
                    }
                    if let cyp = admet.cyp2d6Inhibition {
                        admetRow("CYP2D6 inhibition", String(format: "%.0f%%", cyp * 100), cyp < 0.5)
                    }
                    if let cyp3a4 = admet.cyp3a4Inhibition {
                        admetRow("CYP3A4 inhibition", String(format: "%.0f%%", cyp3a4 * 100), cyp3a4 < 0.5)
                    }
                    if let sol = admet.aqueousSolubility {
                        admetRow("Solubility (logS)", String(format: "%.1f", sol), sol > -4)
                    }
                    if let bbb = admet.bbbPermeability {
                        admetRow("BBB permeability", String(format: "%.0f%%", bbb * 100), bbb > 0.5)
                    }
                    admetRow("Drug-likeness", String(format: "%.0f%%", admet.drugLikeness * 100), admet.drugLikeness > 0.5)
                }
            } else {
                PanelHint(
                    text: "ADMET data not available. Enable the hERG or CYP filters during generation, or dock first.",
                    icon: "info.circle"
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard()
    }

    @ViewBuilder
    private func admetRow(_ label: String, _ value: String, _ good: Bool) -> some View {
        GridRow {
            Text(label)
                .font(DS.Font.label)
                .foregroundStyle(.secondary)
            Text(value)
                .font(DS.Font.mono.weight(.semibold))
                .monospacedDigit()
                .gridColumnAlignment(.trailing)
            DSTag(good ? "Pass" : "Flag",
                  tone: good ? .success : .danger,
                  systemImage: good ? "checkmark" : "xmark")
        }
    }
}

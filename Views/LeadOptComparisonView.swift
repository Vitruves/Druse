// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

/// Full side-by-side comparison of reference ligand vs selected analog.
/// Shown as a sheet from LeadOptimizationTabView.
struct LeadOptComparisonView: View {
    @Environment(AppViewModel.self) private var viewModel
    let reference: LeadOptimizationState
    let analog: LeadOptAnalog

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Lead Optimization Comparison")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                Button("Done") { viewModel.leadOpt.showComparison = false }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
            .padding()

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    energyComparison
                    Divider()
                    propertyRadarChart
                    Divider()
                    admetComparison
                }
                .padding()
            }
        }
    }

    // MARK: - Energy Comparison

    @ViewBuilder
    private var energyComparison: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Docking Energy")
                .font(.system(size: 12, weight: .semibold))

            HStack(spacing: 0) {
                compoundColumn(
                    name: reference.referenceName,
                    energy: reference.referenceResult?.energy,
                    isReference: true
                )
                Divider().frame(height: 60)
                compoundColumn(
                    name: analog.name,
                    energy: analog.bestEnergy,
                    isReference: false
                )
            }

            if let refE = reference.referenceResult?.energy, let analogE = analog.bestEnergy {
                let delta = analogE - refE
                let sm = viewModel.docking.scoringMethod
                let label = sm.isAffinityScore ? "ΔpKi" : "ΔE"
                let betterColor: Color = sm.isAffinityScore ? (delta > 0 ? .green : .red) : (delta < 0 ? .green : delta > 1 ? .red : .secondary)
                HStack {
                    Text("\(label) = \(String(format: "%+.2f", delta)) \(sm.unitLabel)")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(betterColor)
                    if let rmsd = analog.rmsdToReference {
                        Spacer()
                        Text("RMSD = \(String(format: "%.2f", rmsd)) Å")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func compoundColumn(name: String, energy: Float?, isReference: Bool) -> some View {
        VStack(spacing: 4) {
            Text(isReference ? "Reference" : "Analog")
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.secondary)
            Text(name)
                .font(.system(size: 10, weight: .semibold))
                .lineLimit(1)
            if let e = energy {
                let sm = viewModel.docking.scoringMethod
                let isBetter = sm.isAffinityScore
                    ? e > (reference.referenceResult?.energy ?? 0)
                    : e < (reference.referenceResult?.energy ?? 0)
                Text(String(format: "%.2f", e))
                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                    .foregroundColor(isReference ? .primary : (isBetter ? .green : .orange))
                Text(sm.unitLabel)
                    .font(.system(size: 8))
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Property Radar Chart

    @ViewBuilder
    private var propertyRadarChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Molecular Properties")
                .font(.system(size: 12, weight: .semibold))

            let refDesc = reference.referenceDescriptors
            let anaDesc = analog.descriptors

            if let ref = refDesc, let ana = anaDesc {
                // Normalized property values for radar chart
                let axes = [
                    ("MW", normalize(ref.molecularWeight, ana.molecularWeight, max: 600)),
                    ("LogP", normalize(ref.logP, ana.logP, max: 6)),
                    ("TPSA", normalize(ref.tpsa, ana.tpsa, max: 200)),
                    ("HBD", normalize(Float(ref.hbd), Float(ana.hbd), max: 6)),
                    ("HBA", normalize(Float(ref.hba), Float(ana.hba), max: 12)),
                    ("RotB", normalize(Float(ref.rotatableBonds), Float(ana.rotatableBonds), max: 15)),
                ]

                Canvas { context, size in
                    let center = CGPoint(x: size.width / 2, y: size.height / 2)
                    let radius = min(size.width, size.height) / 2 - 30
                    let n = axes.count

                    // Draw axes
                    for i in 0..<n {
                        let angle = CGFloat(i) / CGFloat(n) * 2 * .pi - .pi / 2
                        let end = CGPoint(
                            x: center.x + cos(angle) * radius,
                            y: center.y + sin(angle) * radius
                        )
                        var path = Path()
                        path.move(to: center)
                        path.addLine(to: end)
                        context.stroke(path, with: .color(.gray.opacity(0.3)), lineWidth: 0.5)

                        // Label
                        let labelOffset = CGPoint(
                            x: center.x + cos(angle) * (radius + 15),
                            y: center.y + sin(angle) * (radius + 15)
                        )
                        context.draw(
                            Text(axes[i].0).font(.system(size: 8)).foregroundStyle(.secondary),
                            at: labelOffset
                        )
                    }

                    // Draw polygons
                    func polygon(values: [Float], color: Color, fill: Bool) {
                        var path = Path()
                        for (i, val) in values.enumerated() {
                            let angle = CGFloat(i) / CGFloat(n) * 2 * .pi - .pi / 2
                            let r = radius * CGFloat(min(max(val, 0), 1))
                            let pt = CGPoint(x: center.x + cos(angle) * r, y: center.y + sin(angle) * r)
                            if i == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
                        }
                        path.closeSubpath()
                        if fill {
                            context.fill(path, with: .color(color.opacity(0.15)))
                        }
                        context.stroke(path, with: .color(color), lineWidth: fill ? 1.5 : 1.5)
                    }

                    polygon(values: axes.map(\.1.ref), color: .blue, fill: true)
                    polygon(values: axes.map(\.1.ana), color: .purple, fill: true)
                }
                .frame(height: 200)

                // Legend
                HStack(spacing: 16) {
                    HStack(spacing: 4) {
                        Circle().fill(.blue).frame(width: 8, height: 8)
                        Text("Reference").font(.system(size: 9))
                    }
                    HStack(spacing: 4) {
                        Circle().fill(.purple).frame(width: 8, height: 8)
                        Text("Analog").font(.system(size: 9))
                    }
                }

                // Delta table
                propertyDeltaTable(ref: ref, ana: ana)
            } else {
                Text("Property data not available")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func normalize(_ ref: Float, _ ana: Float, max: Float) -> (ref: Float, ana: Float) {
        (ref: min(ref / max, 1.0), ana: min(ana / max, 1.0))
    }

    @ViewBuilder
    private func propertyDeltaTable(ref: LigandDescriptors, ana: LigandDescriptors) -> some View {
        let rows: [(String, String, String, String)] = [
            ("MW", String(format: "%.0f", ref.molecularWeight), String(format: "%.0f", ana.molecularWeight),
             String(format: "%+.0f", ana.molecularWeight - ref.molecularWeight)),
            ("LogP", String(format: "%.2f", ref.logP), String(format: "%.2f", ana.logP),
             String(format: "%+.2f", ana.logP - ref.logP)),
            ("TPSA", String(format: "%.0f", ref.tpsa), String(format: "%.0f", ana.tpsa),
             String(format: "%+.0f", ana.tpsa - ref.tpsa)),
            ("HBD", "\(ref.hbd)", "\(ana.hbd)", String(format: "%+d", ana.hbd - ref.hbd)),
            ("HBA", "\(ref.hba)", "\(ana.hba)", String(format: "%+d", ana.hba - ref.hba)),
            ("RotBonds", "\(ref.rotatableBonds)", "\(ana.rotatableBonds)",
             String(format: "%+d", ana.rotatableBonds - ref.rotatableBonds)),
        ]

        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
            GridRow {
                Text("").frame(width: 55)
                Text("Ref").font(.system(size: 8, weight: .semibold)).foregroundStyle(.blue)
                Text("Analog").font(.system(size: 8, weight: .semibold)).foregroundStyle(.purple)
                Text("Δ").font(.system(size: 8, weight: .semibold))
            }
            ForEach(rows, id: \.0) { row in
                GridRow {
                    Text(row.0).font(.system(size: 9, weight: .medium)).foregroundStyle(.secondary)
                    Text(row.1).font(.system(size: 9, design: .monospaced))
                    Text(row.2).font(.system(size: 9, design: .monospaced))
                    Text(row.3).font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(row.3.hasPrefix("+") ? .orange : row.3.hasPrefix("-") ? .blue : .secondary)
                }
            }
        }
    }

    // MARK: - ADMET Comparison

    @ViewBuilder
    private var admetComparison: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("ADMET Profile")
                .font(.system(size: 12, weight: .semibold))

            if let anaADMET = analog.admet {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    admetRow("Lipinski", anaADMET.lipinski ? "Pass" : "Fail", anaADMET.lipinski)
                    admetRow("Veber", anaADMET.veber ? "Pass" : "Fail", anaADMET.veber)
                    if let herg = anaADMET.hergLiability {
                        admetRow("hERG Risk", String(format: "%.0f%%", herg * 100), herg < 0.5)
                    }
                    if let cyp = anaADMET.cyp2d6Inhibition {
                        admetRow("CYP2D6 Inh.", String(format: "%.0f%%", cyp * 100), cyp < 0.5)
                    }
                    if let cyp3a4 = anaADMET.cyp3a4Inhibition {
                        admetRow("CYP3A4 Inh.", String(format: "%.0f%%", cyp3a4 * 100), cyp3a4 < 0.5)
                    }
                    if let sol = anaADMET.aqueousSolubility {
                        admetRow("Solubility", String(format: "%.1f", sol), sol > -4)
                    }
                    if let bbb = anaADMET.bbbPermeability {
                        admetRow("BBB Perm.", String(format: "%.0f%%", bbb * 100), bbb > 0.5)
                    }
                    admetRow("Drug-likeness", String(format: "%.0f%%", anaADMET.drugLikeness * 100), anaADMET.drugLikeness > 0.5)
                }
            } else {
                Text("ADMET data not available — enable hERG/CYP filters during generation, or dock first")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private func admetRow(_ label: String, _ value: String, _ good: Bool) -> some View {
        GridRow {
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(good ? .green : .red)
            Image(systemName: good ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.system(size: 9))
                .foregroundStyle(good ? .green : .red)
        }
    }
}

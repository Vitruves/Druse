// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

struct PreparationTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    @State private var manualStepsExpanded = false
    @State private var pkaReferenceExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: PanelStyle.cardSpacing) {
            if let prot = viewModel.molecules.protein {
                proteinCard(prot)
                if let report = viewModel.molecules.preparationReport {
                    if hasIssues(report) {
                        issuesCard(report)
                    }
                }
                quickPrepareCard
                manualStepsCard
                statisticsCard(prot)
            } else {
                ContentUnavailableView {
                    Label("No Protein", systemImage: "cube.transparent")
                } description: {
                    Text("Load a protein from PDB or file first")
                }
                .padding(.top, 24)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
    }

    // MARK: - Protein card

    @ViewBuilder
    private func proteinCard(_ prot: Molecule) -> some View {
        PanelCard("Protein", icon: "cube") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    Text(prot.name)
                        .font(PanelStyle.bodyFont.weight(.medium))
                        .lineLimit(1)
                    Spacer()
                    Text("\(prot.chains.count) chain(s)")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
                if let report = viewModel.molecules.preparationReport {
                    structureSummary(report)
                }
            }
        }
    }

    @ViewBuilder
    private func structureSummary(_ report: ProteinPreparation.PreparationReport) -> some View {
        VStack(spacing: 4) {
            ForEach(report.chainSummary, id: \.chainID) { chain in
                HStack {
                    Text("Chain \(chain.chainID)")
                        .font(PanelStyle.monoSmall.weight(.medium))
                    Spacer()
                    Text("\(chain.residueCount) res")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                    Text(chain.type)
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                }
            }
            if report.waterCount > 0 {
                HStack {
                    Text("Waters")
                        .font(PanelStyle.smallFont)
                    Spacer()
                    Text("\(report.waterCount)")
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                }
            }
            if !report.hetGroups.isEmpty {
                HStack {
                    Text("HETATM")
                        .font(PanelStyle.smallFont)
                    Spacer()
                    Text(report.hetGroups.map(\.name).joined(separator: ", "))
                        .font(PanelStyle.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
    }

    // MARK: - Issues card

    private func hasIssues(_ report: ProteinPreparation.PreparationReport) -> Bool {
        !report.missingResidues.isEmpty
            || !report.chainBreaks.isEmpty
            || !report.residueCompleteness.isEmpty
    }

    @ViewBuilder
    private func issuesCard(_ report: ProteinPreparation.PreparationReport) -> some View {
        PanelCard("Issues", icon: "exclamationmark.triangle") {
            VStack(alignment: .leading, spacing: 10) {
                if !report.missingResidues.isEmpty {
                    issuesGroup(title: "Missing Residues", icon: "questionmark.diamond", color: .orange) {
                        ForEach(report.missingResidues, id: \.gapStart) { gap in
                            issueLine("Chain \(gap.chainID): \(gap.gapStart)–\(gap.gapEnd)")
                        }
                    }
                }
                if !report.chainBreaks.isEmpty {
                    issuesGroup(title: "Chain Breaks", icon: "scissors", color: .orange) {
                        ForEach(Array(report.chainBreaks.enumerated()), id: \.offset) { _, brk in
                            issueLine(
                                "Chain \(brk.chainID): \(brk.previousResidueSeq)→\(brk.nextResidueSeq) "
                                + (brk.isCapped ? "(capped)" : "(uncapped)")
                            )
                        }
                    }
                }
                if !report.residueCompleteness.isEmpty {
                    issuesGroup(title: "Incomplete Residues", icon: "checklist", color: .orange) {
                        ForEach(Array(report.residueCompleteness.prefix(6)), id: \.self) { residue in
                            issueLine("Chain \(residue.chainID) \(residue.residueName) \(residue.residueSeq): \(residue.summary)")
                        }
                        if report.residueCompleteness.count > 6 {
                            issueLine("+\(report.residueCompleteness.count - 6) more incomplete residue(s)")
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func issuesGroup<Content: View>(
        title: String,
        icon: String,
        color: Color,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.caption2)
                Text(title)
                    .font(.caption.weight(.semibold))
                    .textCase(.uppercase)
            }
            .foregroundStyle(color)
            content()
        }
    }

    @ViewBuilder
    private func issueLine(_ text: String) -> some View {
        Text(text)
            .font(PanelStyle.monoSmall)
            .foregroundStyle(.secondary)
            .lineLimit(2)
    }

    // MARK: - Quick prepare card

    @ViewBuilder
    private var quickPrepareCard: some View {
        PanelCard("Prepare for Docking", icon: "wand.and.stars") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Removes alternative conformations, non-standard residues and waters; rebuilds missing atoms; adds hydrogens at pH \(String(format: "%.1f", viewModel.molecules.protonationPH)); assigns partial charges.")
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    PanelRunButton(
                        title: "Run Full Pipeline",
                        icon: "wand.and.stars",
                        color: .accentColor,
                        isDisabled: viewModel.molecules.protein == nil || viewModel.molecules.isMinimizing
                    ) {
                        viewModel.runEnergyMinimization()
                    }
                    .accessibilityIdentifier(AccessibilityID.prepStructureCleanup)
                    if viewModel.molecules.isMinimizing {
                        ProgressView().controlSize(.small)
                    }
                }
            }
        }
    }

    // MARK: - Manual steps card

    @ViewBuilder
    private var manualStepsCard: some View {
        PanelCard("Manual Steps", icon: "slider.horizontal.below.rectangle") {
            PanelDisclosure("Show all steps", isExpanded: $manualStepsExpanded) {
                VStack(alignment: .leading, spacing: 12) {
                    cleanupSection
                    Divider().opacity(0.4)
                    structureRepairSection
                    Divider().opacity(0.4)
                    hydrogensSection
                    Divider().opacity(0.4)
                    chargesSection
                }
            }
        }
    }

    @ViewBuilder
    private var cleanupSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Cleanup", icon: "paintbrush")
            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "Remove Waters", icon: "drop.triangle"
                ) { viewModel.removeWaters() }
                .accessibilityIdentifier(AccessibilityID.prepRemoveWaters)
                PanelSecondaryButton(
                    title: "Keep Pocket Waters", icon: "drop.circle",
                    isDisabled: viewModel.docking.selectedPocket == nil,
                    help: "Remove bulk waters but keep those within \(String(format: "%.0f", viewModel.molecules.pocketWaterRadius)) Å of the pocket center"
                ) { viewModel.keepPocketWaters() }
                .accessibilityIdentifier(AccessibilityID.prepKeepPocketWaters)
            }
            if !viewModel.molecules.keptWaterKeys.isEmpty {
                bridgingWaterInfo
            }
            PanelSecondaryButton(
                title: "Remove Non-standard Residues", icon: "trash.slash"
            ) { viewModel.removeNonStandardResidues() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveNonStandard)

            PanelSecondaryButton(
                title: "Remove Alt. Conformations", icon: "square.on.square.dashed"
            ) { viewModel.removeAltConfs() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveAltConfs)
        }
    }

    @ViewBuilder
    private var structureRepairSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Structure Repair", icon: "bandage")
            PanelSecondaryButton(
                title: "Fix Missing Residues", icon: "bandage",
                help: "Detect gaps in residue numbering, build missing loops (≤15 residues), and rebuild missing heavy atoms."
            ) { viewModel.detectAndFixMissingResidues() }
            .accessibilityIdentifier(AccessibilityID.prepFixMissing)

            PanelSecondaryButton(
                title: "Analyze Missing Atoms", icon: "checklist",
                help: "Compare residues against templates and report missing heavy atoms, missing hydrogens, and extra atoms."
            ) { viewModel.analyzeMissingAtoms() }
            .accessibilityIdentifier(AccessibilityID.prepAnalyzeMissing)

            PanelSecondaryButton(
                title: "Repair Missing Atoms", icon: "wrench.and.screwdriver",
                help: "Rebuild missing heavy atoms from bundled residue geometry templates."
            ) { viewModel.repairMissingAtoms() }
            .accessibilityIdentifier(AccessibilityID.prepRepairMissing)
        }
    }

    @ViewBuilder
    private var hydrogensSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Hydrogens", icon: "atom")
            PanelSecondaryButton(
                title: "Add All Hydrogens", icon: "plus.circle",
                isDisabled: viewModel.molecules.protein == nil,
                help: "Add all hydrogens (polar + nonpolar) from residue templates."
            ) { viewModel.addHydrogens() }
            .accessibilityIdentifier(AccessibilityID.prepAddHydrogens)

            protonationSection

            PanelSecondaryButton(
                title: "Remove All Hydrogens", icon: "minus.circle",
                tint: .red,
                isDisabled: !viewModel.proteinHasHydrogens,
                help: "Strip all hydrogen atoms from the protein."
            ) { viewModel.removeHydrogens() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveHydrogens)
        }
    }

    @ViewBuilder
    private var chargesSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 6) {
            PanelSubheader(title: "Charges", icon: "bolt")
            PanelLabeledRow("Method") {
                Picker("", selection: $vm.docking.chargeMethod) {
                    ForEach(ChargeMethod.allCases, id: \.self) { method in
                        Text(method.rawValue).tag(method)
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 130)
                .accessibilityIdentifier(AccessibilityID.prepChargePicker)
            }
            PanelSecondaryButton(
                title: "Assign \(viewModel.docking.chargeMethod.rawValue) Charges",
                icon: viewModel.docking.chargeMethod.icon,
                isDisabled: chargesDisabled
            ) { viewModel.assignCharges() }
            .accessibilityIdentifier(AccessibilityID.prepAssignCharges)
        }
    }

    private var chargesDisabled: Bool {
        if viewModel.docking.chargeMethod == .gasteiger {
            return viewModel.molecules.rawPDBContent == nil
        }
        return viewModel.molecules.protein == nil
    }

    // MARK: - Bridging water info

    @ViewBuilder
    private var bridgingWaterInfo: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: "drop.fill")
                    .foregroundStyle(.blue)
                    .font(.system(size: 11))
                Text("\(viewModel.molecules.keptWaterKeys.count) bridging water(s) retained")
                    .font(PanelStyle.smallFont)
                    .foregroundStyle(.secondary)
            }
            PanelSliderRow(
                label: "Radius",
                value: $vm.molecules.pocketWaterRadius,
                range: 2.0...10.0, step: 0.5,
                format: { String(format: "%.1f Å", $0) }
            )
        }
        .padding(.leading, 4)
    }

    // MARK: - Protonation section (pH slider + pKa reference)

    @ViewBuilder
    private var protonationSection: some View {
        @Bindable var vm = viewModel
        VStack(alignment: .leading, spacing: 6) {
            PanelSecondaryButton(
                title: "Add Polar Hydrogens (pH \(String(format: "%.1f", viewModel.molecules.protonationPH)))",
                icon: "flask",
                isDisabled: viewModel.molecules.protein == nil,
                help: "Add polar hydrogens (N-H, O-H, S-H) with pH-dependent protonation states."
            ) { viewModel.assignProtonation() }
            .accessibilityIdentifier(AccessibilityID.prepAddPolarH)

            PanelSliderRow(
                label: "pH",
                value: $vm.molecules.protonationPH,
                range: 1.0...14.0, step: 0.1,
                format: { String(format: "%.1f", $0) },
                labelWidth: 30
            )

            PanelDisclosure("pKa reference", icon: "books.vertical", isExpanded: $pkaReferenceExpanded) {
                VStack(alignment: .leading, spacing: 1) {
                    pkaRow("ASP", 3.65)
                    pkaRow("GLU", 4.25)
                    pkaRow("HIS", 6.00)
                    pkaRow("CYS", 8.18)
                    pkaRow("TYR", 10.07)
                    pkaRow("LYS", 10.53)
                    pkaRow("ARG", 12.48)
                }
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .fill(Color.primary.opacity(0.04))
                )
            }
        }
    }

    private func pkaRow(_ residue: String, _ pKa: Float) -> some View {
        let pH = viewModel.molecules.protonationPH
        let isAcidic = ["ASP", "GLU", "CYS", "TYR"].contains(residue)
        let protonated = pH < pKa
        let chargeState: String
        if isAcidic {
            chargeState = protonated ? "neutral" : "(-)"
        } else {
            chargeState = protonated ? "(+)" : "neutral"
        }
        let color: Color = chargeState.contains("+") ? .blue
                          : chargeState.contains("-") ? .red
                          : .secondary
        return HStack(spacing: 6) {
            Text(residue)
                .font(PanelStyle.monoSmall.weight(.medium))
                .frame(width: 32, alignment: .leading)
            Text(String(format: "%.2f", pKa))
                .font(PanelStyle.monoSmall)
                .foregroundStyle(.secondary)
                .frame(width: 38, alignment: .trailing)
            Text(chargeState)
                .font(PanelStyle.smallFont.weight(.medium))
                .foregroundStyle(color)
        }
    }

    // MARK: - Statistics card

    @ViewBuilder
    private func statisticsCard(_ prot: Molecule) -> some View {
        PanelCard("Statistics", icon: "chart.bar") {
            VStack(spacing: 4) {
                statRow("Atoms", "\(prot.atomCount)")
                statRow("Heavy atoms", "\(prot.heavyAtomCount)")
                statRow("Bonds", "\(prot.bondCount)")
                statRow("Residues", "\(prot.residues.count)")
                statRow("Chains", "\(prot.chains.count)")
                statRow("MW", String(format: "%.0f Da", prot.molecularWeight))
            }
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(PanelStyle.monoSmall)
        }
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

struct PreparationTabView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var pkaReferenceExpanded = false
    /// Remembered across launches; expanded by default so every manual tool stays one click away.
    @AppStorage("prep.manualStepsExpanded") private var manualStepsExpanded = true

    var body: some View {
        if let prot = viewModel.molecules.protein {
            DSStepScaffold {
                proteinCard(prot)
                if viewModel.molecules.proteinPrepared {
                    preparedCard(prot)
                }
                if let report = viewModel.molecules.preparationReport, hasIssues(report) {
                    issuesCard(report)
                }
                manualStepsCard
            } footer: {
                footer
            }
        } else {
            DSStepScaffold {
                DSEmptyState(
                    "No Protein Loaded",
                    systemImage: "wrench.and.screwdriver",
                    description: "Load or fetch a protein in step 1, then come back to clean, repair and protonate it."
                ) {
                    Button("Go to Search") { SidebarTab.search.request(in: viewModel) }
                        .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
                }
                .padding(.top, DS.Space.xxl)
            }
        }
    }

    // MARK: - Footer (one-click pipeline)

    private var phText: String { String(format: "%.1f", viewModel.molecules.protonationPH) }

    private var prepareTitle: String {
        viewModel.molecules.isMinimizing ? "Preparing…"
            : viewModel.molecules.proteinPrepared ? "Prepare Again" : "Prepare for Docking"
    }

    private var prepareHelp: String {
        "Keeps the preferred alternate conformation, removes non-standard residues, builds short missing loops, caps chain breaks, rebuilds missing atoms, adds hydrogens at pH \(phText) and assigns partial charges. Waters are kept — use Cleanup below to remove them."
    }

    /// Before preparation the one-click pipeline is the step's primary action. Once the
    /// structure is prepared, moving on is what users want: re-running drops to a
    /// secondary button beside the next-step link.
    @ViewBuilder
    private var footer: some View {
        if viewModel.molecules.proteinPrepared && !viewModel.molecules.isMinimizing {
            let rerun = Button { viewModel.runEnergyMinimization() } label: {
                Label(prepareTitle, systemImage: "arrow.clockwise")
            }
            .buttonStyle(DSSecondaryButtonStyle(fullWidth: false))
            .fixedSize()
            .help(prepareHelp)
            .accessibilityIdentifier(AccessibilityID.prepStructureCleanup)
            let next = DSNextStepButton(SidebarTab.preparation.suggestedNextStep(in: viewModel) ?? .ligands)
                .fixedSize()
            ViewThatFits(in: .horizontal) {
                HStack(spacing: DS.Space.m) {
                    rerun
                    Spacer(minLength: DS.Space.s)
                    next
                }
                VStack(alignment: .leading, spacing: DS.Space.s) {
                    rerun
                    HStack {
                        Spacer(minLength: 0)
                        next
                    }
                }
            }
        } else {
            Text("Cleanup · repair · hydrogens at pH \(phText) · charges")
                .font(DS.Font.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            PanelRunButton(
                title: prepareTitle,
                icon: "wand.and.stars",
                isDisabled: viewModel.molecules.isMinimizing
            ) {
                viewModel.runEnergyMinimization()
            }
            .help(prepareHelp)
            .accessibilityIdentifier(AccessibilityID.prepStructureCleanup)

            if viewModel.molecules.isMinimizing {
                ProgressView()
                    .progressViewStyle(.linear)
                    .controlSize(.small)
                    .accessibilityLabel("Preparing the protein")
            }
        }
    }

    // MARK: - Protein summary card

    private func proteinCard(_ prot: Molecule) -> some View {
        PanelCard(
            "Protein",
            icon: "cube",
            accessory: {
                if viewModel.molecules.proteinPrepared {
                    DSTag("Prepared", tone: .success)
                } else {
                    DSTag("Raw", tone: .warning)
                }
            }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.m) {
                HStack(spacing: DS.Space.m) {
                    Circle()
                        .fill(DS.Color.protein)
                        .frame(width: 8, height: 8)
                    Text(prot.name)
                        .font(DS.Font.label.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: DS.Space.xs)
                    Text("\(prot.chains.count) chain\(prot.chains.count == 1 ? "" : "s")")
                        .font(DS.Font.secondary)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                if let report = viewModel.molecules.preparationReport {
                    structureSummary(report)
                }
            }
        }
    }

    private func structureSummary(_ report: ProteinPreparation.PreparationReport) -> some View {
        Grid(alignment: .leading, horizontalSpacing: DS.Space.m, verticalSpacing: DS.Space.xs) {
            ForEach(report.chainSummary, id: \.chainID) { chain in
                GridRow {
                    Text("Chain \(chain.chainID)")
                        .font(DS.Font.monoSmall.weight(.medium))
                    Text(chain.type)
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Text("\(chain.residueCount) res")
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .gridColumnAlignment(.trailing)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            if report.waterCount > 0 {
                GridRow {
                    Text("Waters")
                        .font(DS.Font.secondary)
                    Text("\(report.waterCount)")
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(DS.Color.water)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                        .gridCellColumns(2)
                }
            }
            if !report.hetGroups.isEmpty {
                GridRow(alignment: .top) {
                    Text("HETATM")
                        .font(DS.Font.secondary)
                    FlowLayout(spacing: DS.Space.xs) {
                        ForEach(Array(report.hetGroups.enumerated()), id: \.offset) { _, het in
                            DSTag(het.name, tint: DS.Color.metal, monospaced: true)
                        }
                    }
                    .gridCellColumns(2)
                }
            }
        }
    }

    // MARK: - Prepared card

    /// What the structure looks like after "Prepare for Docking", read from the prepared
    /// molecule itself so it stays true after later manual edits: protons placed, charge model
    /// outcome, how chain breaks were handled and which waters/HET groups remain.
    private func preparedCard(_ prot: Molecule) -> some View {
        let atoms = prot.atoms
        let hydrogens = atoms.reduce(into: 0) { n, atom in if atom.element == .H { n += 1 } }
        let netCharge = atoms.reduce(Float(0)) { $0 + $1.charge }
        let chargedAtoms = atoms.reduce(into: 0) { n, atom in if atom.charge != 0 { n += 1 } }
        let waters = Set(atoms.lazy
            .filter { ["HOH", "WAT", "DOD", "H2O"].contains($0.residueName) }
            .map { "\($0.chainID):\($0.residueSeq)" }).count
        let breaks = viewModel.molecules.preparationReport?.chainBreaks ?? []
        let capped = breaks.filter(\.isCapped).count

        return PanelCard("Prepared Structure", icon: "checkmark.seal") {
            Grid(alignment: .leading, horizontalSpacing: DS.Space.m, verticalSpacing: DS.Space.xs) {
                preparedRow("Hydrogens", "\(hydrogens.formatted()) at pH \(phText)")
                preparedRow("Charges", chargedAtoms > 0
                    ? "\(chargedAtoms.formatted()) atoms · net \(String(format: "%+.1f", netCharge)) e"
                    : "none assigned")
                if !breaks.isEmpty {
                    preparedRow("Chain breaks", capped == breaks.count
                        ? "\(capped) capped (ACE/NME)"
                        : "\(capped) of \(breaks.count) capped")
                }
                preparedRow("Waters", waters > 0 ? "\(waters) kept" : "none")
            }
        }
    }

    private func preparedRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            Text(value)
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }

    // MARK: - Issues card

    private func hasIssues(_ report: ProteinPreparation.PreparationReport) -> Bool {
        !report.missingResidues.isEmpty
            || !report.chainBreaks.isEmpty
            || !report.residueCompleteness.isEmpty
    }

    private func issuesCard(_ report: ProteinPreparation.PreparationReport) -> some View {
        let total = report.missingResidues.count + report.chainBreaks.count + report.residueCompleteness.count
        return PanelCard(
            "Issues",
            icon: "exclamationmark.triangle",
            accessory: { DSTag("\(total)", tone: .warning, monospaced: true) }
        ) {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                if !report.missingResidues.isEmpty {
                    issuesGroup(title: "Missing Residues", icon: "questionmark.diamond", count: report.missingResidues.count) {
                        ForEach(report.missingResidues, id: \.gapStart) { gap in
                            issueLine("Chain \(gap.chainID): \(gap.gapStart)–\(gap.gapEnd)")
                        }
                    }
                }
                if !report.chainBreaks.isEmpty {
                    issuesGroup(title: "Chain Breaks", icon: "scissors", count: report.chainBreaks.count) {
                        ForEach(Array(report.chainBreaks.enumerated()), id: \.offset) { _, brk in
                            issueLine(
                                "Chain \(brk.chainID): \(brk.previousResidueSeq)→\(brk.nextResidueSeq) "
                                + (brk.isCapped ? "(capped)" : "(uncapped)")
                            )
                        }
                    }
                }
                if !report.residueCompleteness.isEmpty {
                    issuesGroup(title: "Incomplete Residues", icon: "checklist", count: report.residueCompleteness.count) {
                        ForEach(Array(report.residueCompleteness.prefix(6)), id: \.self) { residue in
                            residueIssueLine(
                                residue: "\(residue.chainID) \(residue.residueName) \(residue.residueSeq)",
                                summary: residue.summary
                            )
                        }
                        if report.residueCompleteness.count > 6 {
                            issueLine("+\(report.residueCompleteness.count - 6) more")
                        }
                    }
                }
                PanelHint(
                    text: "“Prepare for Docking” repairs these; Manual Steps ▸ Structure Repair fixes them one at a time.",
                    icon: "lightbulb"
                )
            }
        }
    }

    private func issuesGroup<Content: View>(
        title: String,
        icon: String,
        count: Int,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: DS.Space.xs) {
            HStack(spacing: DS.Space.xs) {
                Image(systemName: icon)
                    .font(DS.Font.caption)
                    .foregroundStyle(DS.Color.warning)
                Text(title)
                    .font(DS.Font.label.weight(.medium))
                Spacer(minLength: DS.Space.xs)
                DSTag("\(count)", tone: .warning, monospaced: true)
            }
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                content()
            }
            .padding(.leading, DS.Space.l)
        }
    }

    private func issueLine(_ text: String) -> some View {
        Text(text)
            .font(DS.Font.monoSmall)
            .foregroundStyle(.secondary)
            .lineLimit(2)
            .textSelection(.enabled)
    }

    /// One incomplete residue per line: fixed-width residue ID, summary truncated
    /// (full text on hover) so long atom lists never wrap into ragged blocks.
    private func residueIssueLine(residue: String, summary: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.s) {
            Text(residue)
                .font(DS.Font.monoSmall.weight(.medium))
                .foregroundStyle(.primary)
                .fixedSize()
            Text(summary)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .help("Chain \(residue): \(summary)")
        .textSelection(.enabled)
    }

    // MARK: - Manual steps card

    private var manualStepsCard: some View {
        PanelCard(
            "Manual Steps",
            icon: "slider.horizontal.3",
            accessory: {
                Button {
                    withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) { manualStepsExpanded.toggle() }
                } label: {
                    Image(systemName: "chevron.right")
                        .font(DS.Font.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(manualStepsExpanded ? 90 : 0))
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .help(manualStepsExpanded ? "Hide the manual preparation tools" : "Show the manual preparation tools")
                .accessibilityLabel("Manual steps")
                .accessibilityValue(manualStepsExpanded ? "Expanded" : "Collapsed")
            }
        ) {
            if manualStepsExpanded {
                VStack(alignment: .leading, spacing: DS.Space.m) {
                    PanelLabeledDivider(title: "Cleanup", icon: "paintbrush")
                    cleanupSection
                    PanelLabeledDivider(title: "Structure Repair", icon: "bandage")
                    structureRepairSection
                    PanelLabeledDivider(title: "Hydrogens & Protonation", icon: "atom")
                    hydrogensSection
                    PanelLabeledDivider(title: "Charges", icon: "bolt")
                    chargesSection
                }
            } else {
                Text("Cleanup, structure repair, hydrogens and charges, one tool at a time.")
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var cleanupSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelSecondaryButton(
                title: "Remove All Waters", icon: "drop.triangle",
                help: "Delete every water molecule"
            ) { viewModel.removeWaters() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveWaters)

            PanelSecondaryButton(
                title: "Keep Pocket Waters Only", icon: "drop.circle",
                isDisabled: viewModel.docking.selectedPocket == nil,
                help: "Remove bulk waters but keep those within \(String(format: "%.0f", viewModel.molecules.pocketWaterRadius)) Å of the pocket center (needs a pocket from Binding Site)"
            ) { viewModel.keepPocketWaters() }
            .accessibilityIdentifier(AccessibilityID.prepKeepPocketWaters)

            if !viewModel.molecules.keptWaterKeys.isEmpty {
                bridgingWaterInfo
            }
            PanelSecondaryButton(
                title: "Remove Non-standard Residues", icon: "trash.slash",
                help: "Delete ligands, ions and modified residues that are not part of the protein"
            ) { viewModel.removeNonStandardResidues() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveNonStandard)

            PanelSecondaryButton(
                title: "Remove Alt. Conformations", icon: "square.on.square.dashed",
                help: "Keep only the highest-occupancy alternate location of each atom"
            ) { viewModel.removeAltConfs() }
            .accessibilityIdentifier(AccessibilityID.prepRemoveAltConfs)
        }
    }

    private var structureRepairSection: some View {
        VStack(alignment: .leading, spacing: DS.Space.s) {
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

    private var hydrogensSection: some View {
        @Bindable var vm = viewModel
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            HStack(spacing: DS.Space.m) {
                Text("pH")
                    .font(DS.Font.label)
                    .foregroundStyle(.secondary)
                // No `step:` — on macOS it draws one tick mark per step (130 here),
                // which renders as a dark bar under the track. Round in the binding.
                Slider(
                    value: Binding(
                        get: { vm.molecules.protonationPH },
                        set: { vm.molecules.protonationPH = ($0 * 10).rounded() / 10 }
                    ),
                    in: 1.0...14.0
                )
                    .controlSize(.small)
                    .accessibilityLabel("Protonation pH")
                    .accessibilityValue(phText)
                    .accessibilityIdentifier(AccessibilityID.prepPHSlider)
                Text(phText)
                    .font(DS.Font.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(width: 32, alignment: .trailing)
            }

            PanelSecondaryButton(
                title: "Add Polar Hydrogens (pH \(phText))",
                icon: "flask",
                help: "Add polar hydrogens (N-H, O-H, S-H) with pH-dependent protonation states."
            ) { viewModel.assignProtonation() }
            .accessibilityIdentifier(AccessibilityID.prepAddPolarH)

            PanelChoiceGrid(columns: 2) {
                PanelSecondaryButton(
                    title: "Add All H", icon: "plus.circle",
                    help: "Add All Hydrogens: polar + nonpolar, from residue templates."
                ) { viewModel.addHydrogens() }
                .accessibilityIdentifier(AccessibilityID.prepAddHydrogens)

                PanelSecondaryButton(
                    title: "Remove All H", icon: "minus.circle",
                    tint: DS.Color.danger,
                    isDisabled: !viewModel.proteinHasHydrogens,
                    help: "Remove All Hydrogens: strip every hydrogen atom from the protein."
                ) { viewModel.removeHydrogens() }
                .accessibilityIdentifier(AccessibilityID.prepRemoveHydrogens)
            }

            PanelDisclosure("pKa reference", icon: "books.vertical", isExpanded: $pkaReferenceExpanded) {
                pkaReference
            }
        }
    }

    private var chargesSection: some View {
        @Bindable var vm = viewModel
        return VStack(alignment: .leading, spacing: DS.Space.s) {
            PanelLabeledRow("Method") {
                Picker("Charge method", selection: $vm.docking.chargeMethod) {
                    ForEach(ChargeMethod.allCases, id: \.self) { method in
                        Text(method.rawValue).tag(method)
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .fixedSize()
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

    private var bridgingWaterInfo: some View {
        @Bindable var vm = viewModel
        return VStack(alignment: .leading, spacing: DS.Space.xs) {
            HStack(spacing: DS.Space.xs) {
                Image(systemName: "drop.fill")
                    .font(DS.Font.caption)
                    .foregroundStyle(DS.Color.water)
                Text("\(viewModel.molecules.keptWaterKeys.count) bridging water(s) retained")
                    .font(DS.Font.secondary)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            PanelSliderRow(
                label: "Radius",
                value: $vm.molecules.pocketWaterRadius,
                range: 2.0...10.0, step: 0.5,
                format: { String(format: "%.1f Å", $0) },
                labelWidth: 44
            )
        }
        .padding(DS.Space.m)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(DS.Color.water.opacity(0.08))
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(AccessibilityID.prepRetainBridgingWaters)
    }

    // MARK: - pKa reference

    private var pkaReference: some View {
        Grid(alignment: .leading, horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.xxs) {
            GridRow {
                pkaHeader("Residue")
                pkaHeader("pKa").gridColumnAlignment(.trailing)
                pkaHeader("At pH \(phText)")
            }
            pkaRow("ASP", 3.65)
            pkaRow("GLU", 4.25)
            pkaRow("HIS", 6.00)
            pkaRow("CYS", 8.18)
            pkaRow("TYR", 10.07)
            pkaRow("LYS", 10.53)
            pkaRow("ARG", 12.48)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(DS.Space.m)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                .fill(DS.Color.controlFill)
        )
    }

    private func pkaHeader(_ text: String) -> some View {
        Text(text)
            .font(DS.Font.caption)
            .monospacedDigit()
            .foregroundStyle(.tertiary)
    }

    private func pkaRow(_ residue: String, _ pKa: Float) -> some View {
        let isAcidic = ["ASP", "GLU", "CYS", "TYR"].contains(residue)
        let protonated = viewModel.molecules.protonationPH < pKa
        let chargeState: String
        let color: Color
        if isAcidic {
            chargeState = protonated ? "neutral" : "anion (−)"
            color = protonated ? .secondary : DS.Color.danger
        } else {
            chargeState = protonated ? "cation (+)" : "neutral"
            color = protonated ? DS.Color.info : .secondary
        }
        return GridRow {
            Text(residue)
                .font(DS.Font.monoSmall.weight(.medium))
            Text(String(format: "%.2f", pKa))
                .font(DS.Font.monoSmall)
                .foregroundStyle(.secondary)
            Text(chargeState)
                .font(DS.Font.secondary.weight(.medium))
                .foregroundStyle(color)
        }
    }
}

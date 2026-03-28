import SwiftUI

struct PreparationTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let prot = viewModel.molecules.protein {
                proteinInfo(prot)
            } else {
                ContentUnavailableView {
                    Label("No Protein", systemImage: "cube.transparent")
                } description: {
                    Text("Load a protein from PDB or file first")
                }
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func proteinInfo(_ prot: Molecule) -> some View {
        // Header
        Label("Protein: \(prot.name)", systemImage: "cube")
            .font(.callout.weight(.semibold))

        // Chain summary
        if let report = viewModel.molecules.preparationReport {
            VStack(alignment: .leading, spacing: 4) {
                Label("Structure", systemImage: "link")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)

                ForEach(report.chainSummary, id: \.chainID) { chain in
                    HStack {
                        Text("Chain \(chain.chainID)")
                            .font(.subheadline.monospaced())
                        Spacer()
                        Text("\(chain.residueCount) res")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                        Text(chain.type)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                if report.waterCount > 0 {
                    HStack {
                        Text("Waters")
                            .font(.subheadline)
                        Spacer()
                        Text("\(report.waterCount)")
                            .font(.subheadline.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }

                if !report.hetGroups.isEmpty {
                    HStack {
                        Text("HETATM")
                            .font(.subheadline)
                        Spacer()
                        Text(report.hetGroups.map(\.name).joined(separator: ", "))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Missing residues
            if !report.missingResidues.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Missing Residues", systemImage: "exclamationmark.triangle")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.orange)

                    ForEach(report.missingResidues, id: \.gapStart) { gap in
                        Text("Chain \(gap.chainID): \(gap.gapStart)\u{2013}\(gap.gapEnd)")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !report.chainBreaks.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Chain Breaks", systemImage: "scissors")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.orange)

                    ForEach(Array(report.chainBreaks.enumerated()), id: \.offset) { _, chainBreak in
                        Text(
                            "Chain \(chainBreak.chainID): \(chainBreak.previousResidueSeq)->\(chainBreak.nextResidueSeq)" +
                            (chainBreak.isCapped ? " (capped)" : " (uncapped)")
                        )
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                    }
                }
            }

            if !report.residueCompleteness.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Residue Completeness", systemImage: "checklist")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.orange)

                    ForEach(Array(report.residueCompleteness.prefix(6)), id: \.self) { residue in
                        Text("Chain \(residue.chainID) \(residue.residueName) \(residue.residueSeq): \(residue.summary)")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    if report.residueCompleteness.count > 6 {
                        Text("+\(report.residueCompleteness.count - 6) more incomplete residue(s)")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }

        Divider()

        // Actions
        VStack(alignment: .leading, spacing: 8) {
            Label("Preparation", systemImage: "wand.and.stars")
                .font(.callout.weight(.semibold))

            // One-click full preparation — the primary action
            Button(action: { viewModel.runEnergyMinimization() }) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Label("Prepare for Docking", systemImage: "wand.and.stars")
                            .font(.subheadline.weight(.semibold))
                        Text("Remove alt. conformations, non-standard residues & waters, rebuild missing atoms, add hydrogens at pH \(String(format: "%.1f", viewModel.molecules.protonationPH)), assign charges")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                    Spacer()
                    if viewModel.molecules.isMinimizing {
                        ProgressView()
                            .controlSize(.mini)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(viewModel.molecules.protein == nil || viewModel.molecules.isMinimizing)
            .help("Full preparation pipeline: cleanup structure, rebuild missing atoms, add hydrogens with pH-dependent protonation, optimize H-bond network, and assign partial charges.")
            .accessibilityIdentifier(AccessibilityID.prepStructureCleanup)

            Divider()

            // Granular controls
            DisclosureGroup("Manual Steps") {
                VStack(alignment: .leading, spacing: 8) {
                    // --- Cleanup ---
                    Label("Cleanup", systemImage: "paintbrush")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    HStack(spacing: 4) {
                        Button(action: { viewModel.removeWaters() }) {
                            Label("Remove All Waters", systemImage: "drop.triangle")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityIdentifier(AccessibilityID.prepRemoveWaters)

                        Button(action: { viewModel.keepPocketWaters() }) {
                            Label("Keep Pocket Waters", systemImage: "drop.circle")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(viewModel.docking.selectedPocket == nil)
                        .help("Remove bulk waters but keep those within \(String(format: "%.0f", viewModel.molecules.pocketWaterRadius)) Å of the pocket center as bridging waters for docking.")
                        .accessibilityIdentifier(AccessibilityID.prepKeepPocketWaters)
                    }

                    if !viewModel.molecules.keptWaterKeys.isEmpty {
                        bridgingWaterInfo
                    }

                    Button(action: { viewModel.removeNonStandardResidues() }) {
                        Label("Remove Non-standard Residues", systemImage: "trash.slash")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityIdentifier(AccessibilityID.prepRemoveNonStandard)

                    Button(action: { viewModel.removeAltConfs() }) {
                        Label("Remove Alt. Conformations", systemImage: "square.on.square.dashed")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityIdentifier(AccessibilityID.prepRemoveAltConfs)

                    Divider()

                    // --- Structure Repair ---
                    Label("Structure Repair", systemImage: "bandage")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    Button(action: { viewModel.detectAndFixMissingResidues() }) {
                        Label("Fix Missing Residues", systemImage: "bandage")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Detect gaps in residue numbering, build missing loops (≤15 residues), and rebuild missing heavy atoms.")
                    .accessibilityIdentifier(AccessibilityID.prepFixMissing)

                    Button(action: { viewModel.analyzeMissingAtoms() }) {
                        Label("Analyze Missing Atoms", systemImage: "checklist")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Compare residues against templates and report missing heavy atoms, missing hydrogens, and extra atoms.")
                    .accessibilityIdentifier(AccessibilityID.prepAnalyzeMissing)

                    Button(action: { viewModel.repairMissingAtoms() }) {
                        Label("Repair Missing Atoms", systemImage: "wrench.and.screwdriver")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Rebuild missing heavy atoms from bundled residue geometry templates.")
                    .accessibilityIdentifier(AccessibilityID.prepRepairMissing)

                    Divider()

                    // --- Hydrogens ---
                    Label("Hydrogens", systemImage: "atom")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    Button(action: { viewModel.addHydrogens() }) {
                        Label("Add All Hydrogens", systemImage: "plus.circle")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.molecules.protein == nil)
                    .help("Add all hydrogens (polar + nonpolar) from residue templates. Also rebuilds any missing heavy atoms needed for placement.")
                    .accessibilityIdentifier(AccessibilityID.prepAddHydrogens)

                    protonationSection

                    Button(action: { viewModel.removeHydrogens() }) {
                        Label("Remove All Hydrogens", systemImage: "minus.circle")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(!viewModel.proteinHasHydrogens)
                    .help("Strip all hydrogen atoms from the protein.")
                    .accessibilityIdentifier(AccessibilityID.prepRemoveHydrogens)

                    Divider()

                    // --- Charges ---
                    Label("Charges", systemImage: "bolt")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    chargeMethodSection
                }
                .padding(.top, 4)
            }
            .font(.subheadline.weight(.medium))
            .foregroundStyle(.secondary)
        }

        Divider()

        // Stats
        VStack(alignment: .leading, spacing: 4) {
            Label("Statistics", systemImage: "chart.bar")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)

            statRow("Atoms", "\(prot.atomCount)")
            statRow("Heavy atoms", "\(prot.heavyAtomCount)")
            statRow("Bonds", "\(prot.bondCount)")
            statRow("Residues", "\(prot.residues.count)")
            statRow("Chains", "\(prot.chains.count)")
            statRow("MW", String(format: "%.0f Da", prot.molecularWeight))
        }
    }

    // MARK: - Bridging Water Info

    @ViewBuilder
    private var bridgingWaterInfo: some View {
        @Bindable var vm = viewModel

        HStack(spacing: 4) {
            Image(systemName: "drop.fill")
                .foregroundStyle(.blue)
                .font(.footnote)
            Text("\(viewModel.molecules.keptWaterKeys.count) bridging water(s) retained")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(.leading, 4)

        HStack {
            Text("Radius")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Slider(value: $vm.molecules.pocketWaterRadius, in: 2.0...10.0, step: 0.5)
                .controlSize(.mini)
            Text("\(String(format: "%.1f", viewModel.molecules.pocketWaterRadius)) Å")
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.leading, 4)
    }

    // MARK: - Protonation Section with pH Slider

    @ViewBuilder
    private var protonationSection: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 4) {
            Button(action: { viewModel.assignProtonation() }) {
                HStack {
                    Label("Add Polar Hydrogens", systemImage: "flask")
                    Spacer()
                    Text("pH \(String(format: "%.1f", viewModel.molecules.protonationPH))")
                        .font(.footnote.monospaced().weight(.medium))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.molecules.protein == nil)
            .help("Add polar hydrogens (N-H, O-H, S-H) with pH-dependent protonation states.")
            .accessibilityIdentifier(AccessibilityID.prepAddPolarH)

            HStack(spacing: 6) {
                Text("pH")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 18, alignment: .trailing)

                Slider(value: $vm.molecules.protonationPH, in: 1.0...14.0, step: 0.1)
                    .controlSize(.mini)

                Text(String(format: "%.1f", viewModel.molecules.protonationPH))
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(width: 28, alignment: .trailing)
            }

            // pKa reference
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
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }

    // MARK: - Charge Method Section

    @ViewBuilder
    private var chargeMethodSection: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Picker("Charges", selection: $vm.docking.chargeMethod) {
                    ForEach(ChargeMethod.allCases, id: \.self) { method in
                        Text(method.rawValue).tag(method)
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .frame(maxWidth: .infinity)
            }

            Button(action: { viewModel.assignCharges() }) {
                Label("Assign \(viewModel.docking.chargeMethod.rawValue) Charges",
                      systemImage: viewModel.docking.chargeMethod.icon)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled({
                if viewModel.docking.chargeMethod == .gasteiger {
                    return viewModel.molecules.rawPDBContent == nil
                }
                return viewModel.molecules.protein == nil
            }())
            .accessibilityIdentifier(AccessibilityID.prepAssignCharges)
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
        let color: Color = chargeState.contains("+") ? .blue : chargeState.contains("-") ? .red : .secondary

        return HStack(spacing: 6) {
            Text(residue)
                .font(.footnote.monospaced().weight(.medium))
                .frame(width: 32, alignment: .leading)
            Text(String(format: "%.2f", pKa))
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 38, alignment: .trailing)
            Text(chargeState)
                .font(.footnote.weight(.medium))
                .foregroundStyle(color)
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.footnote.monospaced())
        }
    }
}

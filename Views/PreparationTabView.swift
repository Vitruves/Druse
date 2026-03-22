import SwiftUI

struct PreparationTabView: View {
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let prot = viewModel.protein {
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
            .font(.system(size: 12, weight: .semibold))

        // Chain summary
        if let report = viewModel.preparationReport {
            VStack(alignment: .leading, spacing: 4) {
                Label("Structure", systemImage: "link")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                ForEach(report.chainSummary, id: \.chainID) { chain in
                    HStack {
                        Text("Chain \(chain.chainID)")
                            .font(.system(size: 11, design: .monospaced))
                        Spacer()
                        Text("\(chain.residueCount) res")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Text(chain.type)
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                    }
                }

                if report.waterCount > 0 {
                    HStack {
                        Text("Waters")
                            .font(.system(size: 11))
                        Spacer()
                        Text("\(report.waterCount)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }

                if !report.hetGroups.isEmpty {
                    HStack {
                        Text("HETATM")
                            .font(.system(size: 11))
                        Spacer()
                        Text(report.hetGroups.map(\.name).joined(separator: ", "))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Missing residues
            if !report.missingResidues.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Missing Residues", systemImage: "exclamationmark.triangle")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.orange)

                    ForEach(report.missingResidues, id: \.gapStart) { gap in
                        Text("Chain \(gap.chainID): \(gap.gapStart)\u{2013}\(gap.gapEnd)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !report.chainBreaks.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Chain Breaks", systemImage: "scissors")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.orange)

                    ForEach(Array(report.chainBreaks.enumerated()), id: \.offset) { _, chainBreak in
                        Text(
                            "Chain \(chainBreak.chainID): \(chainBreak.previousResidueSeq)->\(chainBreak.nextResidueSeq)" +
                            (chainBreak.isCapped ? " (capped)" : " (uncapped)")
                        )
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                    }
                }
            }

            if !report.residueCompleteness.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Label("Residue Completeness", systemImage: "checklist")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.orange)

                    ForEach(Array(report.residueCompleteness.prefix(6)), id: \.self) { residue in
                        Text("Chain \(residue.chainID) \(residue.residueName) \(residue.residueSeq): \(residue.summary)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    if report.residueCompleteness.count > 6 {
                        Text("+\(report.residueCompleteness.count - 6) more incomplete residue(s)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }

        Divider()

        // Actions
        VStack(alignment: .leading, spacing: 8) {
            Label("Preparation", systemImage: "wand.and.stars")
                .font(.system(size: 12, weight: .semibold))

            // Remove waters
            Button(action: { viewModel.removeWaters() }) {
                Label("Remove Waters", systemImage: "drop.triangle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button(action: { viewModel.removeNonStandardResidues() }) {
                Label("Remove Non-standard Residues", systemImage: "trash.slash")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            // Remove alt conformations
            Button(action: { viewModel.removeAltConfs() }) {
                Label("Remove Alt. Conformations", systemImage: "square.on.square.dashed")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Divider()

            // Hydrogen addition
            Label("Hydrogens", systemImage: "atom")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            Button(action: { viewModel.addHydrogens() }) {
                Label("Add All Hydrogens", systemImage: "plus.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.protein == nil)
            .help("Add all hydrogens (polar + nonpolar) from residue templates.")

            // Polar hydrogens at pH with slider
            protonationSection

            Button(action: { viewModel.removeHydrogens() }) {
                Label("Remove All Hydrogens", systemImage: "minus.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(!viewModel.proteinHasHydrogens)
            .help("Strip all hydrogen atoms from the protein.")

            Divider()

            // Native / RDKit-backed preparation
            Label("Preparation Pipeline", systemImage: "cpu")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            Button(action: { viewModel.assignGasteigerCharges() }) {
                Label("Assign Gasteiger Charges", systemImage: "bolt.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.rawPDBContent == nil)

            // Structure Cleanup (protonation + charge assignment)
            Button(action: { viewModel.runEnergyMinimization() }) {
                HStack {
                    Label("Structure Cleanup", systemImage: "waveform.path.ecg")
                    Spacer()
                    if viewModel.isMinimizing {
                        ProgressView()
                            .controlSize(.mini)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.protein == nil || viewModel.isMinimizing)
            .help("Apply protonation at current pH and assign charges. Protein structure is preserved.")

            // Fix Missing Residues (detection)
            Button(action: { viewModel.detectAndReportMissingResidues() }) {
                Label("Fix Missing Residues", systemImage: "bandage")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Detect gaps in residue numbering and report them in the console. Actual loop modeling requires external tools.")

            Button(action: { viewModel.analyzeMissingAtoms() }) {
                Label("Analyze Missing Atoms", systemImage: "checklist")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Compare standard residues against bundled templates and report missing heavy atoms, missing hydrogens, and extra atoms.")

            Button(action: { viewModel.repairMissingAtoms() }) {
                Label("Repair Missing Atoms", systemImage: "wrench.and.screwdriver")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Rebuild missing standard-residue heavy atoms from bundled geometry templates.")

            // Solvation Shell
            Button(action: { viewModel.addSolvationShell() }) {
                HStack {
                    Label("Solvation Shell", systemImage: "drop.circle")
                    Spacer()
                    Text("Beta")
                        .font(.system(size: 9, weight: .medium))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.2))
                        .foregroundStyle(.blue)
                        .clipShape(Capsule())
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(true)
            .help("Solvation shell generation — coming in a future release.")
        }

        Divider()

        // Stats
        VStack(alignment: .leading, spacing: 4) {
            Label("Statistics", systemImage: "chart.bar")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            statRow("Atoms", "\(prot.atomCount)")
            statRow("Heavy atoms", "\(prot.heavyAtomCount)")
            statRow("Bonds", "\(prot.bondCount)")
            statRow("Residues", "\(prot.residues.count)")
            statRow("Chains", "\(prot.chains.count)")
            statRow("MW", String(format: "%.0f Da", prot.molecularWeight))
        }
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
                    Text("pH \(String(format: "%.1f", viewModel.protonationPH))")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.protein == nil)
            .help("Add polar hydrogens (N-H, O-H, S-H) with pH-dependent protonation states.")

            HStack(spacing: 6) {
                Text("pH")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 18, alignment: .trailing)

                Slider(value: $vm.protonationPH, in: 1.0...14.0, step: 0.1)
                    .controlSize(.mini)

                Text(String(format: "%.1f", viewModel.protonationPH))
                    .font(.system(size: 10, design: .monospaced))
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
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }

    private func pkaRow(_ residue: String, _ pKa: Float) -> some View {
        let pH = viewModel.protonationPH
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
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .frame(width: 32, alignment: .leading)
            Text(String(format: "%.2f", pKa))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 38, alignment: .trailing)
            Text(chargeState)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(color)
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 10, design: .monospaced))
        }
    }
}

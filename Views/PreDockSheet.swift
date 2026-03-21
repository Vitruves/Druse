import SwiftUI

/// Confirmation sheet displayed before launching a docking run.
/// Shows protein, ligand, pocket, and configuration summaries
/// with Cancel / Start Docking actions.
struct PreDockSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) private var viewModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    proteinInfoSection
                    Divider()
                    ligandInfoSection
                    Divider()
                    pocketInfoSection
                    Divider()
                    configSummarySection
                    Divider()
                    gpuInfoSection
                }
                .padding()
            }
            Divider()
            footer
        }
        .frame(width: 440, height: 560)
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        HStack {
            Image(systemName: "arrow.triangle.merge")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Color.accentColor)
            Text("Run Docking")
                .font(.system(size: 14, weight: .semibold))
            Spacer()
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }

    // MARK: - Protein Info

    @ViewBuilder
    private var proteinInfoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionLabel("Protein", icon: "building.columns")

            if let prot = viewModel.protein {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(prot.name)
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                    }

                    if !prot.title.isEmpty {
                        Text(prot.title)
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    HStack(spacing: 16) {
                        infoChip("Atoms", "\(prot.atomCount)")
                        infoChip("Heavy", "\(prot.heavyAtomCount)")
                        infoChip("Chains", "\(prot.chains.count)")
                        infoChip("Residues", "\(prot.residues.count)")
                    }

                    if !prot.chains.isEmpty {
                        HStack(spacing: 4) {
                            Text("Chains:")
                                .font(.system(size: 9))
                                .foregroundStyle(.tertiary)
                            Text(prot.chainIDs.joined(separator: ", "))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(10)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No protein loaded")
            }
        }
    }

    // MARK: - Ligand Info

    @ViewBuilder
    private var ligandInfoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionLabel("Ligand", icon: "hexagon")

            if let lig = viewModel.ligand {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Circle().fill(.green).frame(width: 6, height: 6)
                        Text(lig.name)
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                    }

                    HStack(spacing: 16) {
                        infoChip("Atoms", "\(lig.atomCount)")
                        infoChip("Heavy", "\(lig.heavyAtomCount)")
                        infoChip("Bonds", "\(lig.bondCount)")
                        infoChip("MW", String(format: "%.1f", lig.molecularWeight))
                    }

                    let rotatableBonds = lig.bonds.filter(\.isRotatable).count
                    if rotatableBonds > 0 {
                        HStack(spacing: 4) {
                            Text("Rotatable bonds:")
                                .font(.system(size: 9))
                                .foregroundStyle(.tertiary)
                            Text("\(rotatableBonds)")
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(10)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No ligand loaded")
            }
        }
    }

    // MARK: - Pocket Info

    @ViewBuilder
    private var pocketInfoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionLabel("Binding Pocket", icon: "scope")

            if let pocket = viewModel.selectedPocket {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 16) {
                        infoChip("Volume", String(format: "%.0f A\u{00B3}", pocket.volume))
                        infoChip("Buriedness", String(format: "%.0f%%", pocket.buriedness * 100))
                        infoChip("Residues", "\(pocket.residueIndices.count)")
                    }

                    HStack(spacing: 16) {
                        infoChip("Polarity", String(format: "%.0f%%", pocket.polarity * 100))
                        infoChip("Druggability", String(format: "%.0f", pocket.druggability))
                    }

                    HStack(spacing: 4) {
                        Text("Center:")
                            .font(.system(size: 9))
                            .foregroundStyle(.tertiary)
                        Text(String(format: "(%.1f, %.1f, %.1f)",
                                    pocket.center.x, pocket.center.y, pocket.center.z))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 4) {
                        Text("Box size:")
                            .font(.system(size: 9))
                            .foregroundStyle(.tertiary)
                        Text(String(format: "%.1f x %.1f x %.1f A",
                                    pocket.size.x * 2, pocket.size.y * 2, pocket.size.z * 2))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No pocket selected. Detect a pocket in the Docking tab first.")
            }
        }
    }

    // MARK: - Configuration Summary

    @ViewBuilder
    private var configSummarySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionLabel("GA Configuration", icon: "gearshape")

            let config = viewModel.dockingConfig
            VStack(spacing: 4) {
                configLine("Population size", "\(config.populationSize)")
                configLine("Generations", "\(config.numGenerations)")
                configLine("Grid spacing", String(format: "%.3f A", config.gridSpacing))
                configLine("Mutation rate", String(format: "%.3f", config.mutationRate))
                configLine("Crossover rate", String(format: "%.1f", config.crossoverRate))
                configLine("Translation step", String(format: "%.1f A", config.translationStep))
                configLine("Rotation step", String(format: "%.2f rad", config.rotationStep))

                let estimatedEvals = config.populationSize * config.numGenerations
                Divider()
                configLine("Total evaluations", "~\(estimatedEvals)")
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            // ML re-ranking option
            if viewModel.druseScore.isAvailable {
                @Bindable var vm = viewModel
                Toggle(isOn: $vm.useDruseScoreReranking) {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                            .font(.system(size: 10))
                        Text("Re-rank with DruseScore ML")
                            .font(.system(size: 11))
                    }
                }
                .toggleStyle(.checkbox)
                .help("After docking, re-rank poses using the DruseScore neural network (70% ML + 30% physics)")
            }
        }
    }

    // MARK: - GPU Info

    @ViewBuilder
    private var gpuInfoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionLabel("Compute Device", icon: "cpu")

            HStack(spacing: 8) {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(.yellow)
                Text(viewModel.renderer?.device.name ?? "Metal GPU")
                    .font(.system(size: 11, design: .monospaced))
                Spacer()
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(.green)
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        HStack {
            Button("Cancel") {
                dismiss()
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)

            Spacer()

            let canStart = viewModel.selectedPocket != nil
                && viewModel.ligand != nil
                && viewModel.protein != nil

            Button(action: { launchDocking() }) {
                Label("Start Docking", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(!canStart)
        }
        .padding()
    }

    // MARK: - Actions

    private func launchDocking() {
        viewModel.runDocking()
        dismiss()
    }

    // MARK: - Reusable Components

    @ViewBuilder
    private func sectionLabel(_ text: String, icon: String) -> some View {
        Label(text, systemImage: icon)
            .font(.system(size: 12, weight: .medium))
    }

    @ViewBuilder
    private func warningLabel(_ text: String) -> some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.system(size: 11))
            .foregroundStyle(.orange)
    }

    @ViewBuilder
    private func infoChip(_ label: String, _ value: String) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func configLine(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
        }
    }
}

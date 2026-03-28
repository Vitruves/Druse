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
                    if !viewModel.docking.pharmacophoreConstraints.isEmpty {
                        Divider()
                        constraintInfoSection
                    }
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
                .font(.title3.weight(.semibold))
                .foregroundStyle(Color.accentColor)
            Text("Run Docking")
                .font(.body.weight(.semibold))
            Spacer()
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Close")
            .accessibilityIdentifier(AccessibilityID.preDockDismiss)
        }
        .padding()
    }

    // MARK: - Protein Info

    @ViewBuilder
    private var proteinInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Protein", icon: "building.columns")

            if let prot = viewModel.molecules.protein {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(prot.name)
                            .font(.callout.weight(.medium))
                        Spacer()
                    }

                    if !prot.title.isEmpty {
                        Text(prot.title)
                            .font(.footnote)
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
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Text(prot.chainIDs.joined(separator: ", "))
                                .font(.footnote.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No protein loaded")
            }
        }
    }

    // MARK: - Ligand Info

    @ViewBuilder
    private var ligandInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Ligand", icon: "hexagon")

            if let lig = viewModel.molecules.ligand {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Circle().fill(.green).frame(width: 6, height: 6)
                        Text(lig.name)
                            .font(.callout.weight(.medium))
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
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Text("\(rotatableBonds)")
                                .font(.footnote.monospaced().weight(.medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No ligand loaded")
            }
        }
    }

    // MARK: - Pocket Info

    @ViewBuilder
    private var pocketInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Binding Pocket", icon: "scope")

            if let pocket = viewModel.docking.selectedPocket {
                VStack(alignment: .leading, spacing: 8) {
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
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Text(String(format: "(%.1f, %.1f, %.1f)",
                                    pocket.center.x, pocket.center.y, pocket.center.z))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 4) {
                        Text("Box size:")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Text(String(format: "%.1f x %.1f x %.1f A",
                                    pocket.size.x * 2, pocket.size.y * 2, pocket.size.z * 2))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(12)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                warningLabel("No pocket selected. Detect a pocket in the Docking tab first.")
            }
        }
    }

    // MARK: - Configuration Summary

    @ViewBuilder
    private var configSummarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("GA Configuration", icon: "gearshape")

            let config = viewModel.docking.dockingConfig
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
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            // Scoring method picker
            @Bindable var vm = viewModel
            HStack(spacing: 8) {
                Image(systemName: "function")
                    .font(.footnote)
                Text("Scoring")
                    .font(.subheadline)
                Spacer()
                Picker("", selection: $vm.docking.scoringMethod) {
                    ForEach(ScoringMethod.allCases, id: \.self) { method in
                        Label(method.rawValue, systemImage: method.icon).tag(method)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 200)
            }
            if viewModel.docking.scoringMethod == .druseAffinity {
                HStack(spacing: 8) {
                    Image(systemName: "textformat.123")
                        .font(.footnote)
                    Text("Display")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.docking.affinityDisplayUnit) {
                        ForEach(AffinityDisplayUnit.allCases, id: \.self) { unit in
                            Text(unit.rawValue).tag(unit)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 100)
                }
                .help("pKi: -log10(Ki). Ki: dissociation constant in nM/\u{00B5}M")
            }
        }
    }

    // MARK: - GPU Info

    @ViewBuilder
    private var constraintInfoSection: some View {
        let constraints = viewModel.docking.pharmacophoreConstraints.filter(\.isEnabled)
        let hardCount = constraints.filter { $0.strength.isHard }.count
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Pharmacophore Constraints", icon: "scope")
            HStack(spacing: 16) {
                infoChip("Active", "\(constraints.count)")
                if hardCount > 0 {
                    infoChip("Hard", "\(hardCount)")
                }
            }
            ForEach(constraints) { c in
                HStack(spacing: 4) {
                    Image(systemName: c.interactionType.icon)
                        .font(.footnote)
                    Text("\(c.interactionType.rawValue) \u{2192} \(c.targetLabel)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(c.strength.label)
                        .font(.footnote)
                        .foregroundStyle(c.strength.isHard ? .orange : .secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var gpuInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Compute Device", icon: "cpu")

            HStack(spacing: 8) {
                Image(systemName: "bolt.fill")
                    .font(.footnote)
                    .foregroundStyle(.yellow)
                Text(viewModel.renderer?.device.name ?? "Metal GPU")
                    .font(.subheadline.monospaced())
                Spacer()
                Image(systemName: "checkmark.circle.fill")
                    .font(.footnote)
                    .foregroundStyle(.green)
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))
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
            .accessibilityIdentifier(AccessibilityID.preDockCancel)

            Spacer()

            let canStart = viewModel.docking.selectedPocket != nil
                && viewModel.molecules.ligand != nil
                && viewModel.molecules.protein != nil

            Button(action: { launchDocking() }) {
                Label("Start Docking", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(!canStart)
            .accessibilityIdentifier(AccessibilityID.preDockStart)
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
            .font(.callout.weight(.medium))
    }

    @ViewBuilder
    private func warningLabel(_ text: String) -> some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.subheadline)
            .foregroundStyle(.orange)
    }

    @ViewBuilder
    private func infoChip(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.subheadline.monospaced().weight(.semibold))
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func configLine(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.footnote.monospaced().weight(.medium))
        }
    }
}

// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Welcome Screen
//
// Centered overlay shown when no molecule is loaded.
// Four action cards: Start from Protein, Start with Ligand, Open Project, Show Me.

struct WelcomeScreen: View {
    @Environment(AppViewModel.self) private var viewModel
    @Binding var pipelineTab: SidebarTab
    @Binding var pipelinePanelOpen: Bool

    @State private var appeared = false

    var body: some View {
        let updateChecker = UpdateChecker.shared
        VStack(spacing: 0) {
            // Update banner
            if updateChecker.updateAvailable, let version = updateChecker.latestVersion {
                UpdateBanner(version: version, url: updateChecker.downloadURL)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            Spacer()

            // App identity
            VStack(spacing: 12) {
                ZStack {
                    // Glow ring
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [.cyan.opacity(0.15), .clear],
                                center: .center,
                                startRadius: 20,
                                endRadius: 60
                            )
                        )
                        .frame(width: 120, height: 120)

                    Image(systemName: "atom")
                        .font(.largeTitle)
                        .foregroundStyle(
                            .linearGradient(
                                colors: [.cyan, .blue.opacity(0.8)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .symbolEffect(.pulse, options: .repeating.speed(0.3))
                }

                Text("Druse")
                    .font(.largeTitle.bold())
                    .foregroundStyle(.primary)

                Text("GPU-Accelerated Molecular Docking for macOS")
                    .font(.body.weight(.medium))
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 40)
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : -10)

            // 2x2 action cards
            VStack(spacing: 14) {
                HStack(spacing: 14) {
                    WelcomeCard(
                        icon: "cube.fill",
                        title: "Start from Protein",
                        subtitle: "Search RCSB, fetch by PDB ID,\nor load a local .pdb file",
                        gradient: [Color.cyan, Color.blue],
                        delay: 0.1,
                        accessibilityID: AccessibilityID.welcomeStartProtein
                    ) {
                        withAnimation(.easeInOut(duration: 0.25)) {
                            pipelineTab = .search
                            pipelinePanelOpen = true
                        }
                    }

                    WelcomeCard(
                        icon: "hexagon.fill",
                        title: "Start with Ligand",
                        subtitle: "Import .sdf/.smi compounds\nor draw a molecule from SMILES",
                        gradient: [Color.green, Color.teal],
                        delay: 0.15,
                        accessibilityID: AccessibilityID.welcomeStartLigand
                    ) {
                        withAnimation(.easeInOut(duration: 0.25)) {
                            pipelineTab = .ligands
                            pipelinePanelOpen = true
                        }
                    }
                }

                HStack(spacing: 14) {
                    WelcomeCard(
                        icon: "folder.fill",
                        title: "Open Project",
                        subtitle: "Resume a saved .druse workspace\nwith all results preserved",
                        gradient: [Color.orange, Color.yellow],
                        delay: 0.2,
                        accessibilityID: AccessibilityID.welcomeOpenProject
                    ) {
                        viewModel.openProject()
                    }

                    WelcomeCard(
                        icon: "play.circle.fill",
                        title: "Show Me",
                        subtitle: "Dock Nafamostat into Trypsin\nfrom SMILES — full pipeline",
                        gradient: [Color.purple, Color.pink],
                        delay: 0.25,
                        accessibilityID: AccessibilityID.welcomeShowMe
                    ) {
                        viewModel.runGuidedDemo()
                    }
                }
            }
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 15)

            Spacer()

            // Hint
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.doc")
                    .font(.footnote)
                Text("or drag a .pdb / .sdf file anywhere")
                    .font(.subheadline)
            }
            .foregroundStyle(.secondary)
            .padding(.bottom, 24)
            .opacity(appeared ? 1 : 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            ZStack {
                Color(nsColor: .windowBackgroundColor)
                Image("WelcomeBackground")
                    .resizable()
                    .scaledToFill()
                    .opacity(0.4)
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.6).delay(0.1)) {
                appeared = true
            }
            updateChecker.checkIfNeeded()
        }
    }
}

// MARK: - Update Banner

private struct UpdateBanner: View {
    let version: String
    let url: URL?

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(.cyan)

            Text("Druse \(version) is available")
                .font(.subheadline.weight(.medium))

            if let url {
                Link(destination: url) {
                    Text("Download")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.cyan)
                }
                .onHover { inside in
                    if inside { NSCursor.pointingHand.push() }
                    else { NSCursor.pop() }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.cyan.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.cyan.opacity(0.2), lineWidth: 0.5)
                )
        )
        .padding(.top, 16)
    }
}

// MARK: - Welcome Card

private struct WelcomeCard: View {
    let icon: String
    let title: String
    let subtitle: String
    let gradient: [Color]
    let delay: Double
    var accessibilityID: String = ""
    let action: () -> Void

    @State private var isHovered = false
    @State private var appeared = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                // Icon with gradient background circle
                ZStack {
                    Circle()
                        .fill(
                            .linearGradient(
                                colors: gradient.map { $0.opacity(isHovered ? 0.25 : 0.12) },
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 48, height: 48)

                    Image(systemName: icon)
                        .font(.title2.weight(.medium))
                        .foregroundStyle(
                            .linearGradient(
                                colors: gradient,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }

                VStack(spacing: 4) {
                    Text(title)
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.primary)

                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(width: 210, height: 150)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
                    )
                    .shadow(
                        color: Color(nsColor: .shadowColor).opacity(isHovered ? 0.12 : 0.06),
                        radius: isHovered ? 10 : 4,
                        y: isHovered ? 4 : 2
                    )
            )
            .scaleEffect(isHovered ? 1.03 : 1.0)
            .contentShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier(accessibilityID)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .opacity(appeared ? 1 : 0)
        .offset(y: appeared ? 0 : 20)
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.8).delay(delay)) {
                appeared = true
            }
        }
    }
}

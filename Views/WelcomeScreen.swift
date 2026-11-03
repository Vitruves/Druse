// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Welcome Screen
//
// Fills the canvas while nothing is loaded: brand hero, four start actions,
// and a "Recent" column of cached PDB downloads (hidden when empty).

struct WelcomeScreen: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var appeared = false
    @State private var recentEntries: [CachedPDBEntry] = []

    var body: some View {
        let updateChecker = UpdateChecker.shared
        VStack(spacing: 0) {
            if updateChecker.updateAvailable, let version = updateChecker.latestVersion {
                UpdateBanner(version: version, url: updateChecker.downloadURL)
                    .padding(.top, DS.Space.xl)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            Spacer(minLength: DS.Space.xxl)

            // Recent sits beside the start actions when the canvas is wide enough,
            // below them when it is narrow, and is dropped only when neither fits.
            ViewThatFits {
                HStack(alignment: .top, spacing: DS.Space.xxxl + DS.Space.xl) {
                    startColumn
                    // Top-aligned with the action cards, not with the hero.
                    recentColumn
                        .padding(.top, Self.heroMarkSize + DS.Space.xxxl)
                }
                VStack(alignment: .leading, spacing: DS.Space.xxl) {
                    startColumn
                    recentColumn
                }
                startColumn
            }
            .padding(.horizontal, DS.Space.xxxl)
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared || reduceMotion ? 0 : 12)

            Spacer(minLength: DS.Space.xxl)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            ZStack {
                DS.Color.canvasBackground
                RadialGradient(
                    colors: [DS.Color.accent.opacity(0.06), .clear],
                    center: .center,
                    startRadius: 0,
                    endRadius: 600
                )
            }
            .ignoresSafeArea()
        }
        .onAppear {
            withAnimation(reduceMotion ? nil : .easeOut(duration: 0.5).delay(0.05)) {
                appeared = true
            }
            updateChecker.checkIfNeeded()
        }
        .task {
            let entries = await PDBService.shared.cachedEntries()
            withAnimation(DS.Motion.respecting(reduceMotion)) {
                recentEntries = Array(entries.prefix(8))
            }
        }
    }

    private static let heroMarkSize: CGFloat = 96

    private var startColumn: some View {
        VStack(alignment: .leading, spacing: DS.Space.xxxl) {
            hero
            actionGrid
            Label("Or drop a .pdb, .cif, .sdf, .mol2 or .smi file anywhere in the window", systemImage: "arrow.down.doc")
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
        }
        .fixedSize()
    }

    @ViewBuilder
    private var recentColumn: some View {
        if !recentEntries.isEmpty {
            RecentColumn(entries: recentEntries)
                .transition(.opacity)
        }
    }

    private var hero: some View {
        HStack(spacing: DS.Space.xl) {
            DruseMark(size: Self.heroMarkSize)
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                Text("Druse")
                    .font(DS.Font.display)
                    .tracking(-0.5)
                Text("GPU-accelerated molecular docking for Apple silicon")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var actionGrid: some View {
        Grid(horizontalSpacing: DS.Space.l, verticalSpacing: DS.Space.l) {
            GridRow {
                WelcomeCard(
                    icon: "square.and.arrow.down.on.square",
                    title: "Open Protein",
                    subtitle: "Fetch by PDB ID, search RCSB or open a local structure",
                    accessibilityID: AccessibilityID.welcomeStartProtein
                ) {
                    SidebarTab.search.request(in: viewModel)
                }
                WelcomeCard(
                    icon: "hexagon",
                    title: "Start with a Ligand",
                    subtitle: "Import .sdf / .smi compounds or add one from SMILES",
                    accessibilityID: AccessibilityID.welcomeStartLigand
                ) {
                    SidebarTab.ligands.request(in: viewModel)
                }
            }
            GridRow {
                WelcomeCard(
                    icon: "folder",
                    title: "Open Project",
                    subtitle: "Resume a saved .druse workspace with all results",
                    accessibilityID: AccessibilityID.welcomeOpenProject
                ) {
                    viewModel.openProject()
                }
                WelcomeCard(
                    icon: "play.circle",
                    title: "Show Me Around",
                    subtitle: "Dock nafamostat into trypsin — the full pipeline, narrated",
                    accessibilityID: AccessibilityID.welcomeShowMe
                ) {
                    viewModel.runGuidedDemo()
                }
            }
        }
    }
}

// MARK: - Recent column

private struct RecentColumn: View {
    @Environment(AppViewModel.self) private var viewModel
    let entries: [CachedPDBEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            Text("RECENT")
                .font(DS.Font.eyebrow)
                .tracking(0.6)
                .foregroundStyle(.tertiary)
                .padding(.horizontal, DS.Space.m)

            VStack(spacing: DS.Space.xxs) {
                ForEach(entries) { entry in
                    RecentRow(entry: entry) {
                        SidebarTab.search.request(in: viewModel)
                        viewModel.loadFromPDB(id: entry.id)
                    }
                    .disabled(viewModel.workspace.isLoading)
                }
            }
        }
        .frame(width: 280, alignment: .leading)
    }
}

private struct RecentRow: View {
    let entry: CachedPDBEntry
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        Button(action: action) {
            HStack(spacing: DS.Space.m) {
                Image(systemName: "cube")
                    .font(DS.Font.label)
                    .foregroundStyle(DS.Color.protein)
                    .frame(width: 18)
                Text(entry.id)
                    .font(DS.Font.mono.weight(.semibold))
                Text(entry.fetchedAt.formatted(.relative(presentation: .named)))
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, DS.Space.m)
            .frame(height: DS.Size.rowHeight + DS.Space.s)
            .background(shape.fill(isHovering ? DS.Color.controlFill : .clear))
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help("Open \(entry.id) (cached)")
    }
}

// MARK: - Update Banner

private struct UpdateBanner: View {
    let version: String
    let url: URL?

    var body: some View {
        HStack(spacing: DS.Space.m) {
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(DS.Color.info)
            Text("Druse \(version) is available")
                .font(DS.Font.secondary.weight(.medium))
            if let url {
                Link("Download", destination: url)
                    .font(DS.Font.secondary.weight(.semibold))
            }
        }
        .padding(.horizontal, DS.Space.l)
        .padding(.vertical, DS.Space.s)
        .dsFloatingCapsule()
    }
}

// MARK: - Welcome Card

private struct WelcomeCard: View {
    let icon: String
    let title: String
    let subtitle: String
    let accessibilityID: String
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isHovered = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.l, style: .continuous)
        Button(action: action) {
            VStack(alignment: .leading, spacing: DS.Space.l) {
                Image(systemName: icon)
                    .font(.title2.weight(.medium))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(DS.Color.accent)
                    .frame(width: 40, height: 40)
                    .background(
                        RoundedRectangle(cornerRadius: DS.Radius.m, style: .continuous)
                            .fill(DS.Color.accentSoft)
                    )
                VStack(alignment: .leading, spacing: DS.Space.xs) {
                    Text(title)
                        .font(DS.Font.sectionTitle)
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(DS.Space.xl)
            .frame(width: 240, height: 140, alignment: .topLeading)
            .background(shape.fill(.regularMaterial))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
            .shadow(color: Color(nsColor: .shadowColor).opacity(isHovered ? 0.08 : 0), radius: 8, y: 2)
            .scaleEffect(isHovered && !reduceMotion ? 1.01 : 1)
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier(accessibilityID)
        .onHover { hovering in
            withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) {
                isHovered = hovering
            }
        }
    }
}

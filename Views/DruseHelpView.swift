import SwiftUI

// MARK: - Help Data Model

private struct HelpSection: Identifiable {
    let id = UUID()
    let icon: String
    let title: String
    let color: Color
    let items: [HelpItem]
}

private struct HelpItem: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let shortcut: String?
    var isSubtitle: Bool = false

    init(_ title: String, _ description: String, shortcut: String? = nil) {
        self.title = title
        self.description = description
        self.shortcut = shortcut
    }

    static func subtitle(_ title: String) -> HelpItem {
        var item = HelpItem(title, "")
        item.isSubtitle = true
        return item
    }
}

private let helpSections: [HelpSection] = [
    HelpSection(
        icon: "doc.text.fill",
        title: "Loading Structures",
        color: .blue,
        items: [
            HelpItem("Open File", "Load a protein or ligand from PDB, mmCIF, MOL2, SDF, or SMILES files.", shortcut: "⌘O"),
            HelpItem("Drag & Drop", "Drag any supported file directly onto the 3D viewport."),
            HelpItem("PDB Search", "Search and fetch structures from RCSB PDB using the Search tab."),
            HelpItem("Save / Open Project", "Projects (.druse folder) save the full session: protein, ligands, docking results, and settings.", shortcut: "⌘S / ⌘⇧O"),
        ]
    ),
    HelpSection(
        icon: "wrench.and.screwdriver.fill",
        title: "Protein Preparation",
        color: .orange,
        items: [
            HelpItem("Add Hydrogens", "Add polar or all hydrogens from residue templates with pH-dependent protonation."),
            HelpItem("Strip Hydrogens", "Remove all hydrogen atoms from the protein."),
            HelpItem("Assign Charges", "Apply protonation states at the specified pH and assign partial charges."),
            HelpItem("Repair Gaps", "Detect residue numbering gaps, rebuild missing heavy atoms, and add short missing loops (≤15 residues)."),
            HelpItem("Validate", "Check residues against templates and report missing atoms or extra atoms."),
            HelpItem("Bridging Waters", "Keep selected water molecules near the pocket as part of the rigid receptor."),
        ]
    ),
    HelpSection(
        icon: "target",
        title: "Pocket Detection",
        color: .green,
        items: [
            HelpItem("Alpha-Sphere / DBSCAN", "Geometry-based pocket detection using alpha-sphere tessellation and density clustering."),
            HelpItem("ML Detection", "Machine-learning pocket detector (GNN) for more reliable binding site prediction."),
            HelpItem("From Ligand", "Define a pocket around the current ligand — useful when a co-crystallised ligand is present."),
            HelpItem("From Selection", "Define a pocket from residues selected in the sequence view or inspector."),
            HelpItem("Grid Box", "Manually adjust the docking grid in the Docking tab. Use ⌘-drag to move, ⌘⇧-drag to resize."),
        ]
    ),
    HelpSection(
        icon: "atom",
        title: "Ligand Preparation",
        color: .purple,
        items: [
            HelpItem("SMILES Input", "Enter a SMILES string in the Ligand panel to generate a 3D conformer via RDKit/MMFF94."),
            HelpItem("Import File", "Import ligands from SDF, MOL2, or SMILES files into the Ligand Database."),
            HelpItem("Populate & Prepare", "Full pipeline: add polar H → MMFF94 minimize → Gasteiger charges → enumerate tautomers/protomers → generate conformers. Choose pKa method: Table (fast lookup) or GFN2-xTB (quantum chemistry, slower but more accurate).", shortcut: "⌘L"),
            HelpItem("Tautomers / Protomers", "Enumerate chemical forms at the target pH to improve docking coverage."),
            HelpItem("Conformers", "Pre-generate 3D conformers used as GA starting geometries during docking."),
        ]
    ),
    HelpSection(
        icon: "bolt.fill",
        title: "Docking",
        color: .yellow,
        items: [
            HelpItem.subtitle("Scoring Functions"),
            HelpItem("Vina", "Classical empirical scoring: Gauss + repulsion + hydrophobic + H-bond + torsion penalty."),
            HelpItem("Drusina", "Extended Vina with salt bridge, amide-π, and chalcogen bond terms."),
            HelpItem("DruseAF", "Neural network scorer: cross-attention over protein/ligand atom pairs. Outputs pKd × confidence."),
            HelpItem.subtitle("Options"),
            HelpItem("Flexible Docking", "Allow rotatable bonds to flex during the genetic algorithm search."),
            HelpItem("Flexible Receptor", "Sample sidechain chi angles for specified residues during docking."),
            HelpItem("Pharmacophore Constraints", "Add H-bond donor/acceptor, hydrophobic, or charge constraints to bias the search."),
            HelpItem("GFN2-xTB Refinement", "Post-docking geometry optimisation with semi-empirical QM. Improves pose quality with D4 dispersion and implicit solvation."),
            HelpItem("Run Docking", "Launch GPU-accelerated docking. Results appear in the Results tab.", shortcut: "⌘⏎"),
        ]
    ),
    HelpSection(
        icon: "chart.bar.fill",
        title: "Results & Analysis",
        color: .cyan,
        items: [
            HelpItem("Results Tab", "Browse all docking poses ranked by score. Click a pose to view it in 3D."),
            HelpItem("Results Database", "Full pose analysis: interaction diagrams, correlation plots, CSV/SDF export.", shortcut: "⌘⇧R"),
            HelpItem("2D Interaction Diagram", "Schematic view of protein-ligand contacts (H-bonds, salt bridges, hydrophobic, π-π stacking)."),
            HelpItem("Export Poses", "Export top poses as SDF or results table as CSV from the Results tab or Database window."),
            HelpItem("Virtual Screening", "Batch-dock a full ligand database. Hits are ranked by score with ADMET filtering in the Screening tab."),
        ]
    ),
    HelpSection(
        icon: "move.3d",
        title: "3D Viewport",
        color: .teal,
        items: [
            HelpItem("Orbit / Pan / Zoom", "Left-drag to orbit · Right-drag or two-finger scroll to zoom · Middle-drag or Option+drag to pan."),
            HelpItem("Recenter", "Fit all visible atoms to the viewport.", shortcut: "Space"),
            HelpItem("Center on Ligand", "Focus the camera on the active ligand."),
            HelpItem("Render Modes", "Ball & Stick (⌘1), Space Filling (⌘2), Wireframe (⌘3)."),
            HelpItem("Toggle Hydrogens", "Show or hide hydrogen atoms.", shortcut: "⌘H"),
            HelpItem("Molecular Surface", "Compute and display the Connolly surface with colour by electrostatics or hydrophobicity."),
            HelpItem("Z-Slab Clipping", "Clip the view to a depth slab around the pocket for unobstructed ligand inspection."),
        ]
    ),
    HelpSection(
        icon: "keyboard.fill",
        title: "Keyboard Shortcuts",
        color: .gray,
        items: [
            HelpItem("⌘O", "Open file"),
            HelpItem("⌘S", "Save project"),
            HelpItem("⌘⇧O", "Open project"),
            HelpItem("⌘L", "Ligand Database"),
            HelpItem("⌘⇧R", "Results Database"),
            HelpItem("⌘1 / 2 / 3", "Ball & Stick / Space Filling / Wireframe"),
            HelpItem("⌘H", "Toggle hydrogens"),
            HelpItem("Space", "Recenter view"),
            HelpItem("⌘?", "This help window"),
        ]
    ),
]

// MARK: - Main View

struct DruseHelpView: View {
    @State private var selectedSection: HelpSection.ID? = helpSections.first?.id
    @State private var searchText = ""

    private var filteredSections: [HelpSection] {
        guard !searchText.isEmpty else { return helpSections }
        let q = searchText.lowercased()
        return helpSections.compactMap { section in
            let items = section.items.filter {
                $0.title.lowercased().contains(q) || $0.description.lowercased().contains(q)
            }
            if items.isEmpty && !section.title.lowercased().contains(q) { return nil }
            return HelpSection(
                icon: section.icon, title: section.title, color: section.color,
                items: items.isEmpty ? section.items : items
            )
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .navigationTitle("Druse Help")
        .searchable(text: $searchText, prompt: "Search help…")
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(filteredSections, selection: $selectedSection) { section in
            Label {
                Text(section.title)
                    .font(.system(size: 13))
            } icon: {
                Image(systemName: section.icon)
                    .foregroundStyle(section.color)
                    .frame(width: 20)
            }
            .tag(section.id)
            .padding(.vertical, 2)
        }
        .listStyle(.sidebar)
        .frame(minWidth: 200)
        .onChange(of: filteredSections.map(\.id)) { _, ids in
            if let current = selectedSection, !ids.contains(current) {
                selectedSection = ids.first
            }
        }
    }

    // MARK: - Detail

    private var detail: some View {
        Group {
            if let id = selectedSection,
               let section = filteredSections.first(where: { $0.id == id }) {
                SectionDetailView(section: section)
            } else if let first = filteredSections.first {
                SectionDetailView(section: first)
            } else {
                ContentUnavailableView("No results", systemImage: "magnifyingglass")
            }
        }
    }
}

// MARK: - Section Detail

private struct SectionDetailView: View {
    let section: HelpSection

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Header
                HStack(spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(section.color.opacity(0.15))
                            .frame(width: 52, height: 52)
                        Image(systemName: section.icon)
                            .font(.system(size: 24, weight: .medium))
                            .foregroundStyle(section.color)
                    }
                    Text(section.title)
                        .font(.system(size: 22, weight: .semibold))
                }
                .padding(.bottom, 24)

                // Items
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(section.items) { item in
                        if item.isSubtitle {
                            Text(item.title)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.secondary)
                                .textCase(.uppercase)
                                .tracking(0.5)
                                .padding(.top, 16)
                                .padding(.bottom, 4)
                        } else {
                            HelpItemRow(item: item)
                        }
                    }
                }
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(.background)
    }
}

// MARK: - Help Item Row

private struct HelpItemRow: View {
    let item: HelpItem
    @State private var isHovered = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(item.title)
                    .font(.system(size: 13, weight: .medium))
                if !item.description.isEmpty {
                    Text(item.description)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let shortcut = item.shortcut {
                Text(shortcut)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 5))
                    .padding(.top, 1)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isHovered ? Color.primary.opacity(0.05) : .clear)
        )
        .onHover { isHovered = $0 }
        .animation(.easeInOut(duration: 0.1), value: isHovered)
    }
}

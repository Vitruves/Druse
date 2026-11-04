// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Help Data Model

/// Sidebar grouping of the help topics.
private enum HelpGroup: String, CaseIterable {
    case basics = "Basics"
    case steps = "Workflow Steps"
    case tools = "Tools & Viewport"
    case reference = "Reference"
}

private struct HelpSection: Identifiable {
    enum Layout { case article, shortcuts }

    let icon: String
    let title: String
    let summary: String
    let tint: Color
    let group: HelpGroup
    /// The workflow step (or tool) this topic documents, if any. Drives the header eyebrow.
    var step: SidebarTab? = nil
    var layout: Layout = .article
    let items: [HelpItem]

    /// Titles are unique, so they make a stable identity that survives search filtering.
    var id: String { title }

    func with(items: [HelpItem]) -> HelpSection {
        HelpSection(icon: icon, title: title, summary: summary, tint: tint, group: group,
                    step: step, layout: layout, items: items)
    }
}

private struct HelpItem: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let shortcut: String?
    let icon: String?
    var isSubtitle: Bool = false

    init(_ title: String, _ description: String, shortcut: String? = nil, icon: String? = nil) {
        self.title = title
        self.description = description
        self.shortcut = shortcut
        self.icon = icon
    }

    static func subtitle(_ title: String) -> HelpItem {
        var item = HelpItem(title, "")
        item.isSubtitle = true
        return item
    }

    func matches(_ query: String) -> Bool {
        title.lowercased().contains(query) || description.lowercased().contains(query)
    }
}

// MARK: - Workflow Article

private func workflowDetail(_ tab: SidebarTab) -> String {
    switch tab {
    case .search:
        "Open a PDB, mmCIF, MOL2, SDF or SMILES file, fetch a structure by its 4-character PDB ID, or search RCSB by keyword. The Loaded card shows what is currently in the workspace."
    case .preparation:
        "Run the one-click Prepare for Docking pipeline, or use the Manual Steps (cleanup, structure repair, hydrogens & protonation, charges) individually. Issues found in the structure are listed as warnings."
    case .ligands:
        "Add ligands from SMILES, files or the chemical sketcher, then prepare them. The Library card lists the compound set; the Ligand Database window (⌘L) manages it in full."
    case .bindingSite:
        "Detect pockets (Auto, ML, From Ligand, From Selection), focus a pocket with the Pocket View slab presets, and place the docking grid box."
    case .dock:
        "Review the inputs, choose a preset, search method and scoring function, add constraints or flexible residues, then press Run. The footer shows a run summary before you start."
    case .results:
        "Inspect ranked poses and clusters, open the 2D interaction diagram, compare several poses, and export SDF or CSV. Batch and screening hits appear here too."
    case .leadOptimization:
        "Start from a docked pose, generate analogues with property biases and ADMET filters, dock them, and compare each analogue with the reference."
    case .sequence:
        "Not a numbered step: a tool for browsing chains residue by residue, selecting by secondary structure, and renaming or merging chains."
    }
}

private let workflowItems: [HelpItem] = {
    var items: [HelpItem] = [.subtitle("Pipeline")]
    for tab in SidebarTab.allCases where tab.isPipelineStep {
        items.append(HelpItem("\(tab.shortcutNumber). \(tab.rawValue) — \(tab.subtitle)", workflowDetail(tab),
                              shortcut: "⌘\(tab.shortcutNumber)", icon: tab.icon))
    }
    items.append(.subtitle("Tool"))
    items.append(HelpItem("\(SidebarTab.sequence.rawValue) — \(SidebarTab.sequence.subtitle)",
                          workflowDetail(.sequence), shortcut: "⌘\(SidebarTab.sequence.shortcutNumber)", icon: SidebarTab.sequence.icon))
    items.append(.subtitle("Moving Between Steps"))
    items.append(HelpItem("Step Buttons",
                          "The centre of the window toolbar shows one button per step, with the Sequence tool last. Hover a button to see its number, name and shortcut; click it to switch steps.",
                          shortcut: "⌘1 – ⌘8", icon: "rectangle.split.3x1"))
    items.append(HelpItem("Next / Previous Step",
                          "Workflow ▸ Next Step and Previous Step walk through the pipeline in order. Each step's footer also ends with a “Continue to …” link once the step is done.",
                          shortcut: "⌘] / ⌘[", icon: "arrow.right"))
    items.append(HelpItem("Status Badges",
                          "Each step button's number shows its status: green when the step is done, orange when it needs attention, a pulsing accent ring while it runs, neutral when it is ready, and dimmed while it waits on an earlier step.",
                          icon: "checkmark.circle"))
    return items
}()

// MARK: - Help Content

private let helpSections: [HelpSection] = [

    // ── Basics ──────────────────────────────────────────────────────────
    HelpSection(
        icon: "play.circle",
        title: "Getting Started",
        summary: "A first tour of the window and two quick-start recipes.",
        tint: DS.Color.accent,
        group: .basics,
        items: [
            HelpItem("The Main Window",
                     "The step buttons sit in the centre of the toolbar; click one (or press ⌘1–⌘8) to switch steps. The navigator on the left (⌃⌘S) shows the current step with its header, cards and a pinned footer for the main action. The 3D viewport fills the centre, the Inspector sits on the right (⌥⌘0), and the bar under the viewport opens the DruseScript terminal (⌃`) or the activity log (⇧⌘Y). Drag the inner edge of the navigator or the Inspector to resize it. On the right of the toolbar, just before the Terminal and Inspector buttons, the activity indicator shows the live status or the running operation's progress, with a stop button; hover it for the full message and the loaded molecules."),
            HelpItem("Quick Start: Redocking",
                     "1) Open a PDB file containing a co-crystallised ligand (⌘O). 2) In Preparation, click Prepare for Docking. 3) In Ligands, the co-crystallised ligand is already in the library — prepare it. 4) In Binding Site, choose From Ligand to place the pocket and grid box. 5) In Docking, review the run summary and click Run. 6) Poses appear in Results."),
            HelpItem("Quick Start: Virtual Screening",
                     "1) Load and prepare a protein. 2) Open the Ligand Database (⌘L), import an SDF library and batch-prepare all entries. 3) Detect a pocket in Binding Site. 4) Run Docking — every prepared ligand is docked and ranked. 5) Review the hits in Results."),
            HelpItem("Guided Demo",
                     "Choose Show Me Around on the welcome screen to watch a narrated, fully automatic run: Druse fetches trypsin, finds the S1 pocket, docks nafamostat and maps the interactions. Press Stop at any time, or Explore at the end to take over."),
            HelpItem("Apple Silicon GPU",
                     "Druse runs all compute-intensive tasks — grid map generation, genetic algorithm, neural network scoring, Connolly surfaces — on the Apple Silicon GPU via Metal compute shaders. No external GPU or CUDA installation is required."),
        ]
    ),

    HelpSection(
        icon: "list.number",
        title: "Workflow",
        summary: "Seven steps from structure to optimised lead, plus the Sequence tool.",
        tint: DS.Color.accent,
        group: .basics,
        items: workflowItems
    ),

    // ── Workflow steps ──────────────────────────────────────────────────
    HelpSection(
        icon: "square.and.arrow.down.on.square",
        title: "Loading Structures",
        summary: "Bring proteins and ligands into the workspace and save your session.",
        tint: DS.Color.protein,
        group: .steps,
        step: .search,
        items: [
            HelpItem("Open File",
                     "Load a protein or ligand from PDB, mmCIF, MOL2, SDF, or SMILES files with File ▸ Open, the Open button in the toolbar, or Open File… in the Search step. Loading a new protein replaces the previous one, while ligands accumulate in the database.", shortcut: "⌘O"),
            HelpItem("Drag & Drop",
                     "Drag any supported file directly onto the 3D viewport. Druse detects the file type automatically."),
            HelpItem("Fetch by PDB ID",
                     "In the Search step, type a 4-character PDB accession code (e.g. 1HSG) and press Fetch or Enter. Druse downloads the structure from RCSB PDB and loads it. Recent downloads are listed in the Search step and on the welcome screen."),
            HelpItem("Search RCSB",
                     "Enter a keyword (e.g. \"HIV protease\") in the Search RCSB card. Each result shows its PDB ID, title and resolution; click its load button to fetch the structure."),
            HelpItem("Loaded Molecules",
                     "The Loaded card in the Search step, and the chips at the top-left of the viewport, show the current protein and ligand. Use their ✕ buttons to clear either one."),
            HelpItem.subtitle("Projects"),
            HelpItem("Save / Open Project",
                     "Projects (.druse folder) save the full session: protein, ligands, docking results, grid box, workspace settings, and preparation state. Use the toolbar's Save and Open buttons or the File menu, and re-open a project to resume exactly where you left off.", shortcut: "⌘S / ⇧⌘O"),
            HelpItem("Ligand Database Files",
                     "The ligand database can be saved and loaded independently of projects with File ▸ Export / Import Ligand Database, or the Save… and Load… buttons in the Ligands step, so a compound library can be reused across sessions."),
        ]
    ),

    HelpSection(
        icon: "wrench.and.screwdriver",
        title: "Protein Preparation",
        summary: "Clean, repair and protonate the receptor before docking.",
        tint: DS.Color.protein,
        group: .steps,
        step: .preparation,
        items: [
            HelpItem("Why Prepare?",
                     "Crystal structures from the PDB often lack hydrogens, contain alternate conformations, have missing loops or atoms, and do not include protonation states. Proper preparation is critical for accurate docking — incorrect protonation alone can shift binding energies by several kcal/mol."),
            HelpItem.subtitle("Automatic Pipeline"),
            HelpItem("Prepare for Docking",
                     "The button in the step footer runs the full pipeline: 1) selects the best alternate conformer per residue (by occupancy, then B-factor), 2) removes non-standard residues and waters, 3) rebuilds missing heavy atoms from residue templates, 4) caps chain breaks with ACE/NME, 5) adds hydrogens at the specified pH, 6) assigns partial charges. When it finishes, the footer offers “Continue to Ligands”."),
            HelpItem("Issues",
                     "Problems detected in the structure — missing atoms, chain gaps, clashes — are listed in the Issues card as warnings. While issues remain, the Preparation step button in the toolbar shows an attention badge."),
            HelpItem.subtitle("Manual Steps"),
            HelpItem("Cleanup",
                     "Remove waters, non-standard residues, alternate conformations or hydrogens individually, without running the whole pipeline."),
            HelpItem("Structure Repair",
                     "Analyse missing residues and atoms, then repair them. Chain breaks (Cα–Cα distance > 4.0 Å) are reported as missing residue ranges, heavy atoms are rebuilt from standard templates, and ACE/NME caps are added at unresolved termini. Short missing loops (≤15 residues) can be rebuilt."),
            HelpItem("Hydrogens & Protonation",
                     "Adds polar or all hydrogens based on residue templates. Protonation states follow the target pH using a PROPKA-style pKa predictor that accounts for burial, hydrogen bonding, Coulomb interactions, and desolvation effects. Removing hydrogens is useful when re-preparing at a different pH."),
            HelpItem("Charges",
                     "Assigns partial charges to all atoms. Ionisable residues (Asp, Glu, His, Lys, Arg, Cys, Tyr, N/C-termini) are titrated based on their predicted pKa at the target pH."),
            HelpItem.subtitle("Options"),
            HelpItem("pH",
                     "Sets the target pH for protonation (range 2.0–12.0, default 7.4). This affects the charge states of ionisable residues and directly influences docking energetics — for example, at pH 4.0 Asp/Glu remain protonated (neutral), while at pH 7.4 they are deprotonated (negative). The pKa reference table is available in a disclosure below the slider."),
            HelpItem("H-Bond Network Optimisation",
                     "After adding hydrogens, Druse optimises the orientation of rotatable polar groups: hydroxyl rotations (Ser, Thr, Tyr — 3 orientations each), thiol rotations (Cys), amino rotations (Lys), amide flips (Asn OD1↔ND2, Gln OE1↔NE2), and histidine tautomers (6 states: 3 tautomers × 2 flip orientations). Conflicting groups are identified via a spatial hash and resolved as independent cliques using greedy energy minimisation."),
            HelpItem("Bridging Waters",
                     "Some water molecules mediate key protein–ligand hydrogen bonds. Enable this option to retain crystallographic waters near the binding pocket as part of the rigid receptor during docking."),
            HelpItem("Statistics",
                     "Atom, residue and chain counts for the prepared protein live in the Statistics section of the Inspector (⌥⌘0)."),
        ]
    ),

    HelpSection(
        icon: "hexagon",
        title: "Ligand Preparation",
        summary: "Build, prepare and enumerate the compound set to dock.",
        tint: DS.Color.ligand,
        group: .steps,
        step: .ligands,
        items: [
            HelpItem("Ligands Step",
                     "The Active Ligand card shows the current ligand with its SMILES and key properties. The Library card lists every compound with its preparation status; use a row to make it active. Manager… opens the full Ligand Database window."),
            HelpItem("Ligand Database",
                     "The Ligand Database window stores molecular properties (MW, LogP, TPSA, HBD, HBA, RotB), preparation status, and enumerated forms. Columns are sortable and filterable; you can filter by Lipinski compliance or search by name/SMILES. Select an entry to see its structure, properties and conformers in the inspector; select several to prepare, enumerate, dock or delete them together. Each compound is Prepared, Not prepared or Failed (hover the warning icon to see why).", shortcut: "⌘L"),
            HelpItem.subtitle("Adding Ligands"),
            HelpItem("SMILES Input",
                     "Type or paste a SMILES string in the Add Ligand card, or choose Add ▸ SMILES… in the Ligand Database window. Druse converts it to a 3D conformer via RDKit with MMFF94 force-field optimisation."),
            HelpItem("Draw a Structure",
                     "Choose Add ▸ Draw Structure… in the Ligand Database window to sketch a molecule in the built-in chemical editor."),
            HelpItem("Import File",
                     "In the Ligand Database window, Add ▸ Import File with Column Mapping reads SMILES lists (.smi), CSV tables (.csv) and SDF files (.sdf, multiple molecules, 3D coordinates kept). The mapping sheet lets you choose which columns hold names, SMILES and affinity values; other columns become custom table columns."),
            HelpItem("Co-crystallised Ligand",
                     "When loading a PDB with a bound ligand, it is automatically extracted and added to the database. Use From Ligand in the Binding Site step to define the site around it."),
            HelpItem.subtitle("Preparation Pipeline"),
            HelpItem("Prepare",
                     "Prepares the dominant form: add hydrogens → protonate at the target pH → MMFF94 energy minimisation → Gasteiger charges → 3D conformers. A ligand must be prepared before it can be docked; use Enumerate… to dock several tautomers and protomers instead."),
            HelpItem.subtitle("Preparation Options"),
            HelpItem("pKa Method",
                     "Choose between GNN (Metal GPU-accelerated neural network, more accurate for drug-like molecules) or Table (fast lookup from reference pKa values, faster but less accurate for unusual functional groups)."),
            HelpItem("pH",
                     "Target pH for protomer enumeration (default 7.4). Ionisable groups with pKa within the threshold of the target pH generate multiple protomers."),
            HelpItem("Max Tautomers",
                     "Maximum number of tautomeric forms to enumerate (default 10). Tautomers are ranked by stability; only the most stable forms are kept."),
            HelpItem("Max Protomers",
                     "Maximum number of protonation states to enumerate (default 8). Each protomer represents a different charge state of the molecule."),
            HelpItem("Energy Cutoff",
                     "Tautomers/protomers with relative energy above this threshold (default 10.0 kcal/mol) are discarded. Lower values are more selective."),
            HelpItem("pKa Threshold",
                     "Groups with pKa within this distance of the target pH generate protomers (default 2.0 pH units). A threshold of 2.0 at pH 7.4 means groups with pKa 5.4–9.4 are enumerated."),
            HelpItem("Min Population",
                     "Protomers below this Boltzmann population fraction (default 1.0%) are discarded. Ensures only chemically relevant forms are docked."),
            HelpItem("Number of Conformers",
                     "3D conformers generated per tautomer/protomer (default 10). These serve as starting geometries for the genetic algorithm. More conformers improve sampling but increase preparation time."),
            HelpItem.subtitle("Enumeration"),
            HelpItem("Enumerate…",
                     "Opens the enumeration sheet from the Ligand Database window to generate tautomers and protomers for the selected compounds."),
            HelpItem("Form Types",
                     "Each enumerated form is labelled Parent (original input), Tautomer (bond rearrangement), Protomer (charge state change) or Tautomer-Protomer (both). In ensemble docking, all forms are docked and the best-scoring form is reported."),
        ]
    ),

    HelpSection(
        icon: "scope",
        title: "Binding Site",
        summary: "Find the pocket and place the docking grid box.",
        tint: DS.Color.pocket,
        group: .steps,
        step: .bindingSite,
        items: [
            HelpItem("Overview",
                     "Before docking, define where on the protein the ligand should bind. The Binding Site step gathers all of it: four detection methods, the list of detected pockets, the Pocket View slab, and the grid box. Detected pockets are ranked by druggability and show their volume (Å³), buriedness and residue count."),
            HelpItem.subtitle("Detect"),
            HelpItem("Auto (Hybrid)",
                     "Combines ML and geometric detection. First attempts GNN-based prediction; falls back to alpha-sphere/DBSCAN if the ML model is unavailable or finds no pockets. Recommended for most use cases."),
            HelpItem("ML Detection (GNN)",
                     "A graph neural network trained on known binding sites. More reliable for cryptic or shallow pockets that geometric methods may miss. Requires the pocket detector CoreML model (loaded at startup)."),
            HelpItem("From Ligand",
                     "Defines a pocket centred on the current ligand position. Ideal when a co-crystallised ligand is present — the grid box tightly wraps the known binding site."),
            HelpItem("From Selection",
                     "Builds a pocket from residues you select in the Sequence tool or the 3D viewport. Useful for targeting allosteric sites or known functional residues."),
            HelpItem("Alpha-Sphere / DBSCAN",
                     "The geometric algorithm behind Auto: 1) Places probe points on a 1.0 Å grid around the protein. 2) Retains probes 2.0–5.5 Å from the nearest protein atom. 3) Scores buriedness by casting 26 rays and counting how many are blocked by protein (≥30% required). 4) Clusters surviving probes with DBSCAN (ε = 3.5 Å, minPts = 8). 5) Ranks pockets by druggability = volume × buriedness × (residue count / 100)."),
            HelpItem.subtitle("Pockets"),
            HelpItem("Focus a Pocket",
                     "Click a pocket in the Pockets card to select it and frame it in the viewport. The selected pocket drives the grid box and the Pocket View slab."),
            HelpItem.subtitle("Pocket View"),
            HelpItem("Z-Slab Clipping",
                     "Clips the 3D view to a depth slab around the pocket for unobstructed inspection. Turn Pocket View on and choose a preset (Tight 8 Å / Medium 16 Å / Wide 30 Å); enabling a preset also shows the molecular surface at 55% opacity for cavity visualisation. Fine-tune thickness (2–40 Å) and offset (−20 to +20 Å) from the clipping control's popover in the viewport."),
            HelpItem.subtitle("Grid Box"),
            HelpItem("Grid Box Controls",
                     "The grid box defines the docking search volume. Snap it to the protein, the ligand, the selected residues or the selected pocket, adjust its centre and half-size with the axis sliders, then click Apply. Once a grid is defined, the footer offers “Continue to Dock”."),
            HelpItem("Visualisation",
                     "The grid box appears as a wireframe cube in the 3D viewport. Use ⌘-drag to move the box and ⇧⌘-drag to resize it interactively.", shortcut: "⌘ Drag"),
        ]
    ),

    HelpSection(
        icon: "arrow.down.right.and.arrow.up.left",
        title: "Running Docking",
        summary: "Configure the search, review the run summary, and dock.",
        tint: DS.Color.accent,
        group: .steps,
        step: .dock,
        items: [
            HelpItem("Inputs",
                     "The Inputs card checks that a protein, at least one ligand and a binding site are ready. A missing input shows a red mark with a button that takes you to the step that provides it. Manage… opens the Ligand Database."),
            HelpItem("Search",
                     "Choose a preset, the search method, the scoring function and the search budget. Behaviour, torsion, exploration and method-specific options are grouped under Advanced. The Search card is the one place where the scoring function is chosen."),
            HelpItem("Constraints, Flexibility, Refinement",
                     "Add pharmacophore constraints, enable flexible receptor sidechains, and configure post-docking refinement in their own cards below Search."),
            HelpItem.subtitle("Run"),
            HelpItem("Run Summary",
                     "The footer lists the protein, ligands, site, method, scoring and budget that will be used, directly above the Run button."),
            HelpItem("Review & Run",
                     "With “Confirm before docking” on (Settings ▸ Docking, on by default), Run opens a review sheet before starting. Tick “Don't ask again” in the sheet to start directly next time."),
            HelpItem("Progress",
                     "While docking runs, progress, the current generation and the best score appear in the toolbar's activity indicator and in a floating panel above the viewport's display controls. Stop the run from either place, or with Cancel in the step footer. A translucent ghost ligand shows the current best pose in real time."),
            HelpItem("When Done",
                     "The Docking step shows a one-line summary of the run (pose count and best score) and a link to continue to Results."),
        ]
    ),

    HelpSection(
        icon: "function",
        title: "Scoring Functions",
        summary: "Vina, Drusina, DruseAF and Vina + PIGNet2 explained.",
        tint: DS.Color.success,
        group: .steps,
        step: .dock,
        items: [
            HelpItem("Overview",
                     "Druse provides four scoring modes, chosen in the Search card of the Docking step. Vina, Drusina, and DruseAF can guide pose ranking during search. Vina + PIGNet2 uses Vina for pose generation and local optimisation, then applies PIGNet2 to rescore and rerank the generated poses."),
            HelpItem.subtitle("Vina (Classical Empirical)"),
            HelpItem("Vina Energy Terms",
                     "AutoDock Vina scoring (Trott & Olson, 2010) decomposes binding energy into five terms:\n• Gauss₁: short-range van der Waals attraction (w = −0.0356)\n• Gauss₂: long-range van der Waals attraction (w = −0.0052)\n• Repulsion: steric clash penalty for atom overlap (w = +0.840)\n• Hydrophobic: desolvation reward for lipophilic contacts (w = −0.0351)\n• H-bond: directional hydrogen bond reward (w = −0.587)\nPlus a rotatable-bond entropy penalty of +0.058 kcal/mol per torsion."),
            HelpItem("Vina Atom Types",
                     "19 X-Score atom types encoding element + polar/donor/acceptor properties. Carbon (hydrophobic C_H or polar C_P), nitrogen (N_P, N_D, N_A, N_DA), oxygen (O_P, O_D, O_A, O_DA), sulfur, phosphorus, halogens (F, Cl, Br, I), and metals. Donor/acceptor status is assigned from residue templates and bonding environment."),
            HelpItem("Vina Distance Function",
                     "Interactions use a smooth slope step between d_bad (no interaction) and d_good (full interaction). Distances are measured surface-to-surface (inter-atomic distance minus sum of VdW radii). A sigmoid variant (k = 10) is used when analytical gradients are enabled."),
            HelpItem.subtitle("Drusina (Extended Vina)"),
            HelpItem("Drusina Overview",
                     "Drusina adds 10 interaction terms on top of Vina to capture binding features that classical Vina misses. The correction is gated by a Vina quality sigmoid — it only applies when the Vina energy is favourable (< +2 kcal/mol), preventing false attractors in empty space."),
            HelpItem("π–π Stacking (w = −0.40)",
                     "Aromatic ring–ring interactions. Parallel: ring centroid distance 3.3–5.5 Å, normal angle < 30°. T-shaped: same distance, normal angle 60–90°. Score decays as a Gaussian of distance from 3.5 Å optimal."),
            HelpItem("π–Cation (w = −0.30)",
                     "Aromatic ring to positive charge (Lys NZ, Arg guanidinium, ligand aminium). Distance < 6.0 Å, cation approach perpendicular to ring normal (angle < 30°)."),
            HelpItem("Salt Bridge (w = −0.20)",
                     "Coulombic attraction between oppositely charged groups (e.g. Asp COO⁻ ↔ ligand NH₃⁺). Distance < 4.0 Å. Uses group centroids rather than single atoms. Scaled by a burial factor based on sidechain solvent exposure."),
            HelpItem("Halogen Bond (w = −0.15)",
                     "σ-hole interaction: C–X···A where X = F/Cl/Br/I and A = N/O acceptor. Requires near-linear C–X···A angle (> 140°). Distance cutoffs depend on halogen size: 3.5 Å (F/Cl), 3.8 Å (Br), 4.0 Å (I)."),
            HelpItem("Metal Coordination (w = −0.25)",
                     "Coordination bonds to Zn, Fe, Mg, Mn, Ca, Ni, Cu, Co. Ligand N/O/S donors within 2.8 Å of the metal centre. Includes a coordination number bonus."),
            HelpItem("Amide–π Stacking (w = −0.15)",
                     "Backbone amide plane (N–C=O) interacting with ligand aromatic ring. Distance 3.0–5.0 Å, angle between amide normal and ring normal 20–70°."),
            HelpItem("Chalcogen Bond (w = −0.10)",
                     "C–S···N/O σ-hole interaction (Met SD, Cys SG as donors). Distance 2.8–3.8 Å, angle > 140°. Bidirectional — either protein or ligand sulfur can serve as donor."),
            HelpItem("CH–π (w = −0.08)",
                     "Weak interaction between aliphatic C–H groups and aromatic rings. Distance 3.5–4.5 Å. Contributes modestly but consistently in hydrophobic pockets."),
            HelpItem("Coulomb (w = −0.05)",
                     "Screened Coulomb interaction using distance-dependent dielectric (ε = 4r). A soft long-range electrostatic correction complementing the H-bond and salt bridge terms."),
            HelpItem("Desolvation Penalty",
                     "Polar desolvation (+0.15) and hydrophobic desolvation (+0.10) penalties. Account for the energetic cost of burying polar/lipophilic groups without forming compensating interactions."),
            HelpItem.subtitle("DruseAF (Neural Network)"),
            HelpItem("DruseAF Architecture",
                     "A learned scoring function predicting binding affinity (pKd). Version 3 uses cross-attention layers (ligand queries × protein keys, 256-dim, 4 heads, 3 layers). Version 4 (PGN) uses a pairwise geometric network with message-passing on atom-pair features. Output is predicted −log₁₀(Kd), higher = tighter binding."),
            HelpItem("DruseAF Usage",
                     "Can be used as the primary scoring function during docking or as a post-docking rescorer (AF Rescore in the Refinement card). Neural scoring generalises well across chemotypes but may be less reliable for very unusual scaffolds."),
            HelpItem.subtitle("Vina + PIGNet2 (Physics-Informed GNN Rerank)"),
            HelpItem("PIGNet2 Architecture",
                     "A graph neural network (128-dim, 3 GatedGAT layers, 5 Å intermolecular cutoff) that decomposes its prediction into physics-interpretable components: van der Waals (Morse potential), hydrogen bonds, metal coordination, hydrophobic contacts, and rotor entropy. Outputs kcal/mol. Max 256 protein atoms, 64 ligand atoms per evaluation."),
            HelpItem("PIGNet2 Usage",
                     "PIGNet2 is used as a pose scorer/reranker, not as the primary docking search potential. Selecting Vina + PIGNet2 keeps the search, box handling, and local optimisation Vina-guided so poses remain geometrically stable, then rescoring is applied to the final best population and explicit cluster representatives. This can change the final ordering and selected top poses without letting the GNN drive poses into clashes during sampling."),
        ]
    ),

    HelpSection(
        icon: "bolt",
        title: "Docking Algorithms",
        summary: "How the GPU search explores the pose space.",
        tint: DS.Color.accent,
        group: .steps,
        step: .dock,
        items: [
            HelpItem("Overview",
                     "Druse uses a GPU-parallel search to explore the binding pose space. The search method is auto-selected based on ligand flexibility, or you can choose it in the Search card. All methods run entirely on the Metal GPU."),
            HelpItem.subtitle("Genetic Algorithm (Default)"),
            HelpItem("Lamarckian GA",
                     "The default search method, inspired by AutoDock Vina. Each \"individual\" encodes a ligand pose as (translation, rotation quaternion, torsion angles). The GA evolves a population through selection, crossover, mutation, and Lamarckian local search (basin-hopping). Metropolis acceptance (temperature = 1.5 kcal/mol) allows uphill moves to escape local minima."),
            HelpItem("Two-Phase Search",
                     "The GA splits each run into an exploration phase and a refinement phase (controlled by Exploration Phase Ratio, default 55%). During exploration: larger translation steps (5.0 vs 2.0 Å), wider rotation (0.8 vs 0.3 rad ≈ 46° vs 17°), higher mutation rate (0.25 vs 0.10). This broad-then-narrow strategy improves coverage of the binding site."),
            HelpItem("Local Search (Basin Hopping)",
                     "Every N generations (Local Search Frequency), each individual undergoes gradient-based refinement. Two modes: Analytical gradients (default, ~28× fewer energy evaluations, uses trilinear grid interpolation with exact Vina/Drusina derivatives) or Numerical gradients (finite differences, ε = 0.01 Å, slower but always available)."),
            HelpItem("SIMD-Cooperative Local Search",
                     "A warp-level parallel variant where 32 GPU threads collaborate on a single pose's gradient computation. Significantly faster for poses with many degrees of freedom."),
            HelpItem.subtitle("Parallel Tempering (REMC)"),
            HelpItem("Replica Exchange Monte Carlo",
                     "Used for flexible ligands (7–10 rotatable bonds). Maintains 8 replicas at temperatures from 0.6 to 4.0 kcal/mol. Hot replicas explore broadly; cold replicas refine. Replicas exchange configurations every 5 generations using the Metropolis criterion. 200 MC steps per replica before extraction."),
            HelpItem.subtitle("Fragment-Based Docking"),
            HelpItem("Beam Search with Growth",
                     "For highly flexible ligands (>10 rotatable bonds), the ligand is decomposed into rigid fragments. The anchor fragment is placed first (256 initial samples), then remaining fragments are grown one at a time. At each growth step, torsion angles are sampled at 10° resolution (36 samples per bond), and a beam of width 32 (configurable 4–256) partial poses survives to the next step. Beam width controls diversity vs speed."),
            HelpItem("Scaffold Enforcement",
                     "Optional SMARTS pattern to lock the core scaffold orientation. Only R-groups are grown — useful when the binding mode of the scaffold is known from crystallography or prior docking."),
            HelpItem.subtitle("Auto-Tuning"),
            HelpItem("Automatic Parameter Selection",
                     "When the preset is \"Auto\", Druse adapts all parameters to the system:\n• Population ∝ √(rotatable bonds × heavy atoms), clamped 150–600\n• Generations ∝ pocket buriedness × volume × √(ligand size), clamped 120–500\n• Runs ∝ overall difficulty (flexibility × buriedness × protein size), clamped 2–10\n• Search method: GA for ≤6 rotatable bonds, Parallel Tempering for 7–10, Fragment-Based for >10\n• Grid spacing: 0.375 Å for proteins ≤10k atoms, 0.5 Å for larger."),
        ]
    ),

    HelpSection(
        icon: "slider.horizontal.3",
        title: "Docking Parameters",
        summary: "Presets, budget, exploration tuning, flexibility and constraints.",
        tint: DS.Color.accent,
        group: .steps,
        step: .dock,
        items: [
            HelpItem("Presets",
                     "Four presets configure all parameters at once: Auto (adapts to complexity), Fast (pop 50, gen 30, 3 runs), Standard (pop 150, gen 80, 5 runs), Thorough (pop 300, gen 200, 10 runs). You can also edit parameters individually after selecting a preset."),
            HelpItem.subtitle("Core Parameters"),
            HelpItem("Population",
                     "Number of candidate poses per generation. Larger populations explore more of the conformational space but require more GPU memory and time. Typical range: 150–600."),
            HelpItem("Generations",
                     "Number of GA evolution cycles per run. More generations allow deeper convergence. Typical range: 80–500."),
            HelpItem("Runs",
                     "Independent restarts with different random seeds. Each run produces its own best pose; results from all runs are merged and clustered. More runs reduce the chance of missing the global minimum. Typical range: 2–10."),
            HelpItem("Grid Spacing",
                     "Resolution of the pre-computed affinity grid maps in Å. Options: 0.375 (high resolution, best accuracy), 0.500 (balanced), 0.750 (fast, lower accuracy). Finer grids require more VRAM."),
            HelpItem("Mutation Rate",
                     "Probability that each degree of freedom is randomly perturbed during GA mutation. Range 0.01–0.25 (default 0.10). Higher values increase exploration but slow convergence."),
            HelpItem.subtitle("Advanced: Exploration Tuning"),
            HelpItem("Exploration Phase Ratio",
                     "Fraction of generations spent in the broad exploration phase (default 0.55 = 55%). During exploration, translation/rotation steps and mutation rate are amplified. Range 0.2–0.8."),
            HelpItem("Local Search Frequency",
                     "Run gradient-based local search every N generations (default 3). Setting to 1 refines every generation (thorough but slow). Range 1–10."),
            HelpItem("MC Temperature",
                     "Metropolis acceptance temperature in kcal/mol (default 1.5). Higher temperatures accept more uphill moves, aiding exploration. Lower temperatures converge faster but may trap in local minima. Range 0.5–4.0."),
            HelpItem("Exploration Mutation Rate",
                     "Mutation rate during the exploration phase only (default 0.25). Decays to the base mutation rate in the refinement phase. Range 0.10–0.50."),
            HelpItem.subtitle("Flexibility"),
            HelpItem("Ligand Flexibility",
                     "Toggle to allow rotatable bonds to flex during docking. When disabled, the ligand is treated as a rigid body (translation + rotation only). Enable for any ligand with rotatable bonds — rigid docking of a flexible molecule will miss the correct binding mode."),
            HelpItem("Flexible Receptor (Induced Fit)",
                     "Sample sidechain χ angles for selected pocket-lining residues during docking. Two modes: Soft (automatic selection of sidechains near the pocket, with the selected residues and χ-angle count listed) or Manual (hand-pick up to 8 residues). Increases computational cost but captures induced-fit effects."),
            HelpItem.subtitle("Constraints"),
            HelpItem("Pharmacophore Constraints",
                     "Force specific interactions during docking. Add constraints by clicking atoms/residues in the 3D viewport or in the pharmacophore editor: H-bond acceptor/donor, aromatic ring, lipophilic contact, positive/negative charge. Each constraint has a strength (Hard = strict requirement; Soft = 5.0 kcal/mol penalty) and distance threshold (2.5–4.0 Å). Poses violating hard constraints are eliminated; soft constraint violations add an energy penalty."),
            HelpItem.subtitle("Refinement"),
            HelpItem("GFN2-xTB Refinement",
                     "Semi-empirical quantum-mechanical geometry optimisation of top poses (default: top 20). Uses GFN2-xTB with D4 dispersion and implicit water solvation (ALPB or GBSA). Options: optimisation level (crude/normal/tight), harmonic position restraint (0.005 Ha/Å² ≈ 0.3 Å allowed motion), max RMSD filter (2.0 Å), blend weight (0.3 = 30% GFN2 + 70% Vina). Improves pose quality for tight binders."),
            HelpItem("AF Rescore",
                     "After docking, rescore top cluster representatives with the DruseAF v4 neural network. Particularly useful when Vina/Drusina ranking disagrees with expected SAR."),
            HelpItem.subtitle("Clash & Strain"),
            HelpItem("Clash Handling",
                     "Atom overlaps exceeding 0.4 Å trigger a penalty of 5.0 kcal/mol per Å excess. Prevents physically impossible poses from dominating the population."),
            HelpItem("Strain Penalty",
                     "Post-docking MMFF94 internal energy check. Poses with intramolecular strain > 6.0 kcal/mol above the relaxed conformer receive a weighted penalty (default weight 0.5). High strain values (> 4.0) are flagged in the Results step."),
        ]
    ),

    HelpSection(
        icon: "chart.bar.doc.horizontal",
        title: "Results & Analysis",
        summary: "Rank, inspect, compare and export docked poses.",
        tint: DS.Color.success,
        group: .steps,
        step: .results,
        items: [
            HelpItem("Summary",
                     "After docking, poses are ranked by score and grouped into clusters (RMSD threshold = 2.0 Å). The Summary card shows the best and mean score, pose and cluster counts, the method and scoring function used, and the wall time. Scores are coloured green (good), amber (fair) or orange (weak)."),
            HelpItem("Poses",
                     "Each row shows the rank, score, cluster and RMSD. Use the row buttons to view the pose in the viewport, open its interaction diagram, or send it to Lead Opt. Tick several poses to view or clear them together."),
            HelpItem("Score Landscape",
                     "A bar chart in the Summary card shows the scores of the top 50 ranked poses, coloured by score tone, with the displayed pose highlighted. Outliers are clamped so one poor pose does not flatten the chart. Click a bar to view that pose."),
            HelpItem.subtitle("Pose Details"),
            HelpItem("Energy Decomposition",
                     "Each pose shows tagged energy components: vdW (van der Waals), elec (electrostatic), hb (hydrogen bond), str (strain — flagged if > 4.0), xtb (GFN2-xTB energy if refined), cst (constraint penalty, if any)."),
            HelpItem("Cluster Tags",
                     "Each pose shows its cluster ID (C0, C1, etc.). Poses in the same cluster have similar binding modes (RMSD < 2.0 Å). The cluster representative (rank 0) is typically the best-scoring member."),
            HelpItem("Multi-Pose Analysis",
                     "Opens from the step footer to compare several poses side by side — scores, interactions and pose-to-pose RMSD."),
            HelpItem.subtitle("Interaction Diagram"),
            HelpItem("2D Interaction Diagram",
                     "A schematic window of protein–ligand contacts with the ligand centred and interacting residues arranged radially. Pick the pose to show from its toolbar. Interactions are colour-coded: H-bonds (blue, dashed), salt bridges (red), hydrophobic contacts (olive), π-stacking (green), π-cation (pink), halogen bonds (teal) and metal coordination (bronze); distances label the connecting lines. Export the diagram as PNG, SVG or PDF."),
            HelpItem("Viewport Legends",
                     "When interactions or a coloured surface are shown, floating legends appear at the top-right of the viewport. Click a legend to collapse it into a small pill."),
            HelpItem.subtitle("Results Database"),
            HelpItem("Full Pose Analysis",
                     "The Results Database window lists batches in its sidebar and their poses in a sortable table, with the selected pose's 2D diagram, pocket metrics and per-atom energy contributions in the side panel. Its toolbar holds Correlation, Interaction Diagram, Export and a search field. Open it from the Results step or View ▸ Results Database.", shortcut: "⇧⌘R"),
            HelpItem("Correlation Analysis",
                     "In batch mode, the correlation view shows a scatter plot of docking scores vs experimental affinity (if Ki/Kd values are provided). Includes a regression line with R² value and outlier identification. Useful for validating the scoring function against known actives/inactives."),
            HelpItem.subtitle("Export"),
            HelpItem("Export Poses",
                     "Export poses as SDF (with 3D coordinates and energy properties) or the results table as CSV from the Results step footer or the Results Database toolbar."),
        ]
    ),

    HelpSection(
        icon: "rectangle.stack",
        title: "Virtual Screening",
        summary: "Dock a whole library and rank the hits.",
        tint: DS.Color.info,
        group: .steps,
        step: .results,
        items: [
            HelpItem("Overview",
                     "Virtual screening docks an entire ligand database against the target pocket. All prepared ligands in the database are queued automatically when you click Run in the Docking step. Results are ranked by best score per compound."),
            HelpItem("Batch Progress",
                     "Overall progress shows in the toolbar's activity indicator; the Batch card in the Results step lists per-ligand status and has a Cancel button. Each compound is docked independently using the same grid maps and search parameters."),
            HelpItem("Hit Ranking",
                     "Compounds are ranked by their best pose energy. The Screening Hits card shows rank, ligand name, best energy, pose count, and estimated Ki (if available), and can export the hit list."),
            HelpItem("Filtering",
                     "Filter hits by affinity threshold (discard compounds weaker than a cutoff), RMSD cutoff (only keep well-converged poses), and ADMET properties (Lipinski, Veber, hERG safety)."),
            HelpItem("Ensemble Docking",
                     "When ligands have multiple enumerated forms (tautomers/protomers), all forms are docked and the best-scoring form determines the compound's rank. Population cutoff (default 5%) filters rare forms. Population weighting adjusts scores by Boltzmann probability."),
        ]
    ),

    HelpSection(
        icon: "arrow.triangle.branch",
        title: "Lead Optimisation",
        summary: "Generate, dock and compare analogues of a docked pose.",
        tint: DS.Color.ligand,
        group: .steps,
        step: .leadOptimization,
        items: [
            HelpItem("Overview",
                     "Starting from a docked pose, the Lead Opt step generates and evaluates structural analogues. Workflow: 1) Dock a compound, 2) In Results, click Optimize on your best pose, 3) Configure analogue generation, 4) Generate and dock analogues, 5) Compare with the reference."),
            HelpItem.subtitle("Generation Controls"),
            HelpItem("Analogue Count",
                     "Number of analogues to generate: 10, 25, 50, or 100. More analogues explore more chemical space but take longer to dock."),
            HelpItem("Similarity Threshold",
                     "Tanimoto similarity to the reference compound (0.3–0.95). Higher values generate closer analogues; lower values explore more diverse scaffolds."),
            HelpItem("Keep Scaffold",
                     "When enabled, only R-groups are modified — the core scaffold is preserved. Useful for SAR exploration around a validated binding mode."),
            HelpItem("Property Direction",
                     "Four sliders bias analogue generation: Polarity (less ↔ more), Rigidity (flexible ↔ rigid), Lipophilicity (hydrophilic ↔ lipophilic), Size (smaller ↔ larger). Centre position = no bias."),
            HelpItem("Scaffold Analogs",
                     "Analogs… in the Ligand Database window enumerates substitutions on a chosen scaffold and adds the results to the library."),
            HelpItem.subtitle("ADMET Filters"),
            HelpItem("Drug-Likeness Filters",
                     "Pre-filter analogues before docking: Lipinski Rule of 5 (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10), Veber (TPSA ≤ 140, RotB ≤ 10), hERG safety (cardiac toxicity risk), CYP safety (metabolic liability), Max LogP (1–8 lipophilicity cap). Filtered and failed compounds are marked in the Analogs list."),
            HelpItem.subtitle("Analysis"),
            HelpItem("Dock & Deliver",
                     "Dock all generated analogues against the same site; Stop halts the run. Each analogue row then shows its score difference against the reference."),
            HelpItem("Comparison",
                     "Select an analogue to compare it with the reference: property differences (ΔMW, ΔLogP, ΔTPSA, ΔRotB), ADMET status, docking energy difference, and Tanimoto similarity. The comparison view shows overlay structures and a detailed ADMET profile heatmap."),
            HelpItem("3D Alignment",
                     "The analogue's best pose is aligned to the reference pose in the 3D viewport, allowing direct visual comparison of binding modes."),
        ]
    ),

    // ── Tools & viewport ────────────────────────────────────────────────
    HelpSection(
        icon: "list.bullet.indent",
        title: "Sequence Tool",
        summary: "Browse chains residue by residue and edit chain topology.",
        tint: DS.Color.protein,
        group: .tools,
        step: .sequence,
        items: [
            HelpItem("Overview",
                     "The Sequence tool (the last of the toolbar's step buttons) displays a scrollable canvas of one-letter codes for each chain. It uses a custom canvas renderer for smooth scrolling even with very large proteins.", shortcut: "⌘8"),
            HelpItem.subtitle("Visual Encoding"),
            HelpItem("Secondary Structure Colouring",
                     "Each residue cell is colour-coded by secondary structure assignment (helix, sheet, coil, turn). A legend at the top of the view shows the colours."),
            HelpItem("Gap Markers",
                     "Missing residues appear as gap markers between resolved segments, showing the number of missing residues (e.g. \"gap 5\"). These correspond to unresolved density in the crystal structure."),
            HelpItem("Nucleic Acid Chains",
                     "DNA and RNA chains are displayed in a separate section below protein chains, using standard single-letter nucleotide codes."),
            HelpItem.subtitle("Selection"),
            HelpItem("Click to Select",
                     "Click a residue cell to select it. The corresponding residue highlights in the 3D viewport, and the panel below shows its sequence number, residue name, atom count, and secondary structure."),
            HelpItem("Select by Secondary Structure",
                     "Use the selection buttons to select all helices, sheets, coils, or turns across the entire protein. Clear removes the selection. Useful for visual analysis or for building a binding site from a structural motif."),
            HelpItem.subtitle("Chain Operations"),
            HelpItem("Chain Context Menu",
                     "Right-click a chain header to access: Select Chain (select all residues), Copy Chain Sequence (clipboard), Rename Chain (change chain ID), Merge Chains (combine two chains), Delete Residues (remove selected, with confirmation)."),
            HelpItem("Copy Sequence",
                     "The Copy button exports the full FASTA sequence of the selected chain to the clipboard."),
            HelpItem("Collapse / Expand",
                     "Click the chain header disclosure triangle to collapse or expand individual chains, keeping the view focused on the chain of interest."),
        ]
    ),

    HelpSection(
        icon: "move.3d",
        title: "3D Viewport",
        summary: "Navigate, select, and control how molecules are drawn.",
        tint: DS.Color.accent,
        group: .tools,
        items: [
            HelpItem.subtitle("Navigation"),
            HelpItem("Orbit",
                     "Left-click + drag to orbit (rotate) the view around the molecule centre.", shortcut: "Drag"),
            HelpItem("Pan",
                     "Shift + drag, middle-click + drag, or Option + drag to pan the view horizontally and vertically.", shortcut: "⇧ Drag"),
            HelpItem("Zoom",
                     "Scroll wheel or pinch gesture to zoom in and out. Scroll up = zoom in.", shortcut: "Scroll"),
            HelpItem("Z-Roll",
                     "Control + drag to spin the view around the viewing axis (roll).", shortcut: "⌃ Drag"),
            HelpItem("Camera Controls",
                     "The vertical cluster at the top-right of the viewport fits the view to everything, to the ligand or to the protein, and resets the camera. Fit to View is also in the View menu, or press Space with the viewport focused.", shortcut: "⌘0"),
            HelpItem.subtitle("Selection"),
            HelpItem("Click to Select",
                     "Single-click an atom (or residue in ribbon mode) to select it. Selection highlights in the viewport, the Sequence tool, and the Inspector."),
            HelpItem("Multi-Select",
                     "Option + click to toggle atoms in and out of the selection without clearing previous selections.", shortcut: "⌥ Click"),
            HelpItem("Box Select",
                     "Option + drag to draw a selection rectangle. All atoms within the box are selected. Shift + box select adds to the existing selection.", shortcut: "⌥ Drag"),
            HelpItem("Deselect All",
                     "Press Escape or click on empty space to clear the selection.", shortcut: "Esc"),
            HelpItem.subtitle("Display Controls"),
            HelpItem("Display Cluster",
                     "The floating bar at the bottom of the viewport groups the display controls: style (the four render modes), show (hydrogens, side chains, surface, surface colour, colour scheme), and look (lighting, clipping, probe radius). The active control is tinted with the accent colour. On narrow windows the last groups collapse into a “…” menu."),
            HelpItem("Molecule Chips",
                     "Chips at the top-left of the viewport name the loaded protein and ligand; their ✕ clears the molecule."),
            HelpItem.subtitle("Render Modes"),
            HelpItem("Ball & Stick",
                     "Atoms as impostor spheres, bonds as impostor cylinders. Default mode for detailed atom inspection. GPU ray-sphere/ray-cylinder intersection gives pixel-perfect smooth rendering.", shortcut: "⌃⌘1"),
            HelpItem("Space Filling (CPK)",
                     "Van der Waals spheres at full radius. Shows the molecular surface and steric interactions. Useful for visualising clashes and surface complementarity.", shortcut: "⌃⌘2"),
            HelpItem("Wireframe",
                     "Bonds only, no atom spheres. Minimal visual clutter — good for large proteins or dense binding sites.", shortcut: "⌃⌘3"),
            HelpItem("Ribbon (Cartoon)",
                     "Secondary-structure cartoon: α-helices as ribbons, β-sheets as arrows, coils as tubes. Click a ribbon element to select the corresponding residue. Best for protein fold overview.", shortcut: "⌃⌘4"),
            HelpItem.subtitle("Display Options"),
            HelpItem("Toggle Hydrogens",
                     "Show or hide hydrogen atoms globally with the “H” button in the display cluster. Hiding hydrogens reduces clutter in crowded binding sites.", shortcut: "⌃⌘H"),
            HelpItem("Molecular Surface",
                     "Compute and display the Connolly solvent-excluded surface via a Metal marching-cubes GPU kernel. Colour modes: Uniform (solid), Electrostatic (red/blue ESP gradient), Hydrophobicity (polar ↔ lipophilic), Pharmacophore (interaction-type colours). The probe radius (default 1.4 Å) is adjustable from its popover in the display cluster."),
            HelpItem("Clipping",
                     "Turn clipping on from the display cluster; its popover holds the slab Thickness and Offset sliders. The Pocket View presets in the Binding Site step set the same slab."),
            HelpItem("Lighting Mode",
                     "Uniform (flat, no depth cues) or Directional (shaded with fixed light direction for 3D perception). Directional is better for presentations and screenshots."),
            HelpItem("Atom Labels",
                     "Overlay labels on atoms: none, element symbol, atom index, or PDB serial number."),
            HelpItem.subtitle("Colouring"),
            HelpItem("CPK (Element) Colours",
                     "Default atom colouring by element: C grey, N blue, O red, S yellow, H light grey, halogens green/orange. Standard across chemistry."),
            HelpItem("Chain Colours",
                     "Each chain receives an automatic palette colour. Override it in the Inspector by clicking the chain colour swatch and choosing from 16 presets or a custom colour picker."),
            HelpItem("Ligand Carbon Colour",
                     "Set a custom colour for ligand carbon atoms to distinguish the ligand from protein atoms. Choose from 16 presets in the Inspector."),
            HelpItem.subtitle("Ghost Ligand"),
            HelpItem("Docking Preview",
                     "During docking, a translucent \"ghost\" ligand shows the current best pose updating in real time. Ghost atoms/bonds are drawn with read-only depth and 50–70% opacity so they overlay the protein without obscuring it."),
        ]
    ),

    HelpSection(
        icon: "sidebar.trailing",
        title: "Inspector & Workspace",
        summary: "The Inspector, toolbar activity, console and Settings.",
        tint: DS.Color.info,
        group: .tools,
        items: [
            HelpItem.subtitle("Inspector"),
            HelpItem("Show / Hide",
                     "The Inspector is the right-hand column. Toggle it with the toolbar button, View ▸ Show Inspector, or its close button. It opens automatically on wide windows when a protein is loaded.", shortcut: "⌥⌘0"),
            HelpItem("Selection",
                     "For a selected atom: index, name, element, coordinates (X, Y, Z), van der Waals radius, partial charge, parent residue, and chain ID. For a residue: sequence number, residue name, atom count, secondary structure, and completeness."),
            HelpItem("Chains & Display",
                     "Click the colour swatch next to a chain to choose CPK (element colouring), Chain default (automatic palette), or one of 16 custom colours. Subsets let you show or hide parts of the structure."),
            HelpItem("Statistics",
                     "Atom, residue and chain counts for the loaded protein. This is the one place protein statistics are shown."),
            HelpItem.subtitle("Toolbar"),
            HelpItem("Activity Indicator",
                     "The capsule on the right of the toolbar, next to the Terminal and Inspector buttons, names the loaded protein and ligand and shows the current operation with its progress. Its stop button cancels the running operation; click the capsule to see the most recent log lines."),
            HelpItem.subtitle("Terminal and Logs"),
            HelpItem("Terminal",
                     "The bar at the bottom of the window shows a hint for the next step, counts warnings and errors, and has two tabs. Terminal opens the DruseScript terminal: type a command after the prompt and press Return. Tab completes, → accepts the grey suggestion, ↑ and ↓ walk the history, ⌃R searches it, ⇧Return continues on a new line (pasted lines also run together as one script), ⌃C cancels the running command, ⌘K or ⌃L clears the screen. Long commands show a live progress line. The terminal shows only commands and their output. Drag the bar's top edge to resize the panel.", shortcut: "⌃`"),
            HelpItem("Logs",
                     "The Logs tab shows the activity log: search it, filter by level or category, copy selected or all entries, clear it, or reveal the log file in Finder. It opens by itself on the first error.", shortcut: "⇧⌘Y"),
            HelpItem.subtitle("Settings"),
            HelpItem("General",
                     "Appearance: Automatic (follows the system), Light or Dark."),
            HelpItem("Viewport",
                     "Viewport background and surface display options."),
            HelpItem("Docking",
                     "“Confirm before docking” shows the Review & Run sheet before every run (on by default)."),
            HelpItem("Logging",
                     "Enable file logging, choose the log directory, set the retention period (default 7 days), and manually archive or clear logs."),
            HelpItem("Performance",
                     "Maximum CPU cores Druse uses at once (C++ core, ligand preparation, OpenMM); applied immediately. The headless --cores flag overrides it for one run."),
            HelpItem("Console, Command Line, Shell Completions",
                     "Console: history length and autosuggestions. Command Line: install the 'druse' terminal command. Shell Completions: install Tab completion for fish, zsh or bash. See Console & DruseScript."),
            HelpItem.subtitle("Feedback"),
            HelpItem("Report a Bug",
                     "Help ▸ Report Bug… sends a description, optional reply-to address, system details and, if you allow it, the recent activity log directly to the developer."),
        ]
    ),

    HelpSection(
        icon: "terminal",
        title: "Console & DruseScript",
        summary: "Drive every step from the terminal panel, a script or the macOS command line.",
        tint: DS.Color.accent,
        group: .tools,
        items: [
            HelpItem.subtitle("The Language"),
            HelpItem("One Command per Line",
                     "DruseScript is a console language, not a programming language: each line is verb [subcommand] [arguments…] [key=value…], for example 'prep ph=7.4 waters=remove' or 'dock preset=thorough scoring=drusina'. ';' separates several commands on one line, a trailing \\ continues a line, # starts a comment and \"double quotes\" keep spaces (SMILES, paths)."),
            HelpItem("Variables and Session Settings",
                     "'set name = value' defines $name (or ${name}); 'vars' lists them. 'set onerror=continue' keeps a script running after a failed command (default stop). 'log level debug|info|warn|error|none' filters log output."),
            HelpItem("Batch Work",
                     "There are no loops: batch commands do the work, such as 'ligand prep all', 'dock batch ligands=prepared', 'screen library.smi' or 'leadopt dock'. 'run file.druse' runs another script; relative paths resolve against the script's folder."),
            HelpItem("Help in the Console",
                     "'help' lists every command by category; 'help dock' shows each form with its arguments, types, ranges, defaults and examples; 'apropos tautomer' searches all descriptions. A misspelt command or option gets a did-you-mean suggestion."),
            HelpItem.subtitle("Terminal Keys"),
            HelpItem("Tab", "Completes the command, subcommand, option, value, ligand or residue name, or file path under the caret. With several candidates a list opens: ↑ ↓ or ⇧Tab move, Tab or Return accept, Esc closes."),
            HelpItem("→ / ⌥→", "Accepts the grey autosuggestion (taken from history and the command registry), all of it or one word. Settings ▸ Console turns suggestions off."),
            HelpItem("↑ / ↓", "Walk the command history (kept between sessions; length in Settings ▸ Console)."),
            HelpItem("⌃R", "Reverse history search: type to filter, ⌃R again for the next older match, Return runs it, ⌃G cancels."),
            HelpItem("⇧Return", "Continues on a new line; the whole input then runs as one script. Pasted multi-line text runs the same way."),
            HelpItem("⌃C", "Cancels the running command (docking, preparation, screening…) and prints ^C; with nothing running it abandons the line being typed."),
            HelpItem("⌘K / ⌃L", "Clears the screen; history and the line being edited are kept."),
            HelpItem.subtitle("Command Line"),
            HelpItem("Install 'druse'",
                     "Settings ▸ Command Line (or 'cli install') writes a 'druse' command into ~/.local/bin, or /usr/local/bin with an administrator prompt. It runs this Druse.app without a window."),
            HelpItem("Run Scripts Headless",
                     "'druse script.druse', 'druse --eval \"load 1abc.pdb; prep\"', 'druse dock preset=fast' (one command as shell words) or 'druse' alone for an interactive prompt. Flags: --var name=value, --cores N, --log-level level, --echo. Exit codes: 0 success, 1 a command failed, 2 usage or syntax error, 130 interrupted."),
            HelpItem("Shell Completions",
                     "Settings ▸ Shell Completions (or 'cli completions install shell=all') installs Tab completion for fish, zsh and bash, generated from the same command registry as the in-app terminal."),
            HelpItem("Full Reference",
                     "Docs/DruseScript.md in the Druse repository documents every command, option and default; Examples/ holds runnable scripts for protein preparation, ligand libraries, docking, batch docking and lead optimization."),
        ]
    ),

    // ── Reference ───────────────────────────────────────────────────────
    HelpSection(
        icon: "keyboard",
        title: "Keyboard Shortcuts",
        summary: "Every shortcut in one place.",
        tint: DS.Color.accent,
        group: .reference,
        layout: .shortcuts,
        items: [
            HelpItem.subtitle("File"),
            HelpItem("⌘O", "Open file (PDB, mmCIF, SDF, MOL2, SMILES)"),
            HelpItem("⇧⌘O", "Open project"),
            HelpItem("⌘S", "Save project (.druse)"),
            HelpItem.subtitle("Workflow"),
            HelpItem("⌘1 – ⌘7", "Go to step 1–7: Search, Preparation, Ligands, Binding Site, Docking, Results, Lead Opt"),
            HelpItem("⌘8", "Open the Sequence tool"),
            HelpItem("⌘]", "Next step"),
            HelpItem("⌘[", "Previous step"),
            HelpItem.subtitle("View"),
            HelpItem("⌃⌘1", "Ball & Stick render mode"),
            HelpItem("⌃⌘2", "Space Filling (CPK) render mode"),
            HelpItem("⌃⌘3", "Wireframe render mode"),
            HelpItem("⌃⌘4", "Ribbon (cartoon) render mode"),
            HelpItem("⌃⌘H", "Toggle hydrogen atoms"),
            HelpItem("⌘0", "Fit to view"),
            HelpItem("⌃⌘S", "Show / hide the navigator"),
            HelpItem("⌥⌘0", "Show / hide the Inspector"),
            HelpItem("⌃`", "Show / hide the terminal"),
            HelpItem("⇧⌘Y", "Show / hide the logs"),
            HelpItem.subtitle("Windows"),
            HelpItem("⌘L", "Open the Ligand Database window"),
            HelpItem("⇧⌘R", "Open the Results Database window"),
            HelpItem("⌘?", "Open this help window"),
            HelpItem.subtitle("3D Viewport (when focused)"),
            HelpItem("1 – 4", "Ball & Stick, Space Filling, Wireframe, Ribbon"),
            HelpItem("H", "Toggle hydrogen atoms"),
            HelpItem("L", "Toggle lighting mode"),
            HelpItem("Space", "Fit to view"),
            HelpItem("Esc", "Deselect all"),
            HelpItem.subtitle("3D Viewport Mouse"),
            HelpItem("Drag", "Orbit (rotate view)"),
            HelpItem("⇧ Drag", "Pan (translate view)"),
            HelpItem("Scroll", "Zoom in / out"),
            HelpItem("⌃ Drag", "Z-roll (spin around view axis)"),
            HelpItem("⌥ Click", "Toggle multi-select"),
            HelpItem("⌥ Drag", "Box selection"),
            HelpItem("⌘ Drag", "Move grid box"),
            HelpItem("⇧⌘ Drag", "Resize grid box"),
        ]
    ),

    HelpSection(
        icon: "book.closed",
        title: "Algorithms Reference",
        summary: "What runs under the hood, from grid maps to atom picking.",
        tint: DS.Color.metal,
        group: .reference,
        items: [
            HelpItem.subtitle("Grid Map Pre-computation"),
            HelpItem("Affinity Grids",
                     "Before docking, Druse pre-computes 3D grid maps for each Vina atom type. Each grid point stores the interaction energy contribution of a probe atom at that position. Stored as half-precision (float16) for VRAM efficiency. During scoring, the ligand's energy is evaluated via trilinear interpolation — O(N_atoms) per pose instead of O(N_atoms × N_protein_atoms)."),
            HelpItem.subtitle("RMSD Clustering"),
            HelpItem("Pose Clustering",
                     "After all runs complete, poses are clustered by pairwise RMSD (heavy atoms only, ignoring hydrogens). Threshold = 2.0 Å. Greedy leader algorithm: the best-scoring unassigned pose becomes a cluster representative; all poses within 2.0 Å join its cluster. GPU-accelerated pairwise RMSD computation when available, CPU parallel fallback otherwise."),
            HelpItem.subtitle("Explicit Reranking"),
            HelpItem("Top-Cluster Reranking",
                     "The top 12 cluster representatives are reranked using explicit atom-pair scoring (not grid interpolation). 4 variants per cluster are generated via short local search (20 steps), and the best variant replaces the grid-scored representative. For Vina + PIGNet2, this rerank score comes from PIGNet2 after Vina local optimisation."),
            HelpItem.subtitle("pKa Prediction"),
            HelpItem("PROPKA-Style Algorithm",
                     "Ionisable residues (Asp, Glu, His, Lys, Arg, Cys, Tyr, termini) receive a pKa shift: pKa_shifted = pKa_model + Δ_burial + Δ_hbond + Δ_coulomb + Δ_desolvation. The shift accounts for the local electrostatic environment: buried charges are stabilised/destabilised, nearby H-bond partners shift the equilibrium, and desolvation penalises charge burial. A spatial grid (4.0 Å cells) enables O(1) neighbour lookups."),
            HelpItem.subtitle("H-Bond Network"),
            HelpItem("Clique Optimisation",
                     "The H-bond network optimiser identifies all movable groups (rotatable OH/SH/NH₃, flippable amides, His tautomers), builds a conflict graph connecting groups that share interaction partners, partitions into independent cliques, and greedily selects the lowest-energy state for each clique. This ensures physically consistent hydrogen placement throughout the protein."),
            HelpItem.subtitle("Surface Generation"),
            HelpItem("Connolly Surface (GPU)",
                     "Marching cubes on a 3D grid: each voxel stores the distance to the nearest atom surface (atom radius − probe radius). The zero-isosurface is extracted as a triangle mesh via the marching cubes lookup table, entirely on the GPU. Colour is mapped per-vertex using precomputed properties (ESP, hydrophobicity, pharmacophore type)."),
            HelpItem.subtitle("Atom Picking"),
            HelpItem("GPU Object-ID Picking",
                     "Atoms are rendered to an off-screen R32Uint texture with each pixel storing the atom ID. On click, the pixel under the cursor is read back — O(1) on CPU with pixel-perfect accuracy. Ribbon picking hit-tests against stored Cα control points."),
        ]
    ),
]

// MARK: - Main View

struct DruseHelpView: View {
    @State private var selectedSection: HelpSection.ID? = helpSections.first?.id
    @State private var searchText = ""

    private var query: String {
        searchText.trimmingCharacters(in: .whitespaces).lowercased()
    }

    private var filteredSections: [HelpSection] {
        let q = query
        guard !q.isEmpty else { return helpSections }
        return helpSections.compactMap { section in
            let titleMatches = section.title.lowercased().contains(q) || section.summary.lowercased().contains(q)
            let items = section.items.filter { !$0.isSubtitle && $0.matches(q) }
            if items.isEmpty && !titleMatches { return nil }
            return section.with(items: items.isEmpty ? section.items : items)
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .navigationTitle("Druse Help")
        .searchable(text: $searchText, placement: .toolbar, prompt: "Search help")
    }

    // MARK: Sidebar

    private var sidebar: some View {
        let sections = filteredSections
        return List(selection: $selectedSection) {
            ForEach(HelpGroup.allCases, id: \.self) { group in
                let members = sections.filter { $0.group == group }
                if !members.isEmpty {
                    Section(group.rawValue) {
                        ForEach(members) { section in
                            Label {
                                Text(section.title)
                            } icon: {
                                Image(systemName: section.icon)
                            }
                            .tag(section.id)
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top, spacing: 0) {
            HStack {
                DruseWordmark(markSize: 20)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, DS.Space.l)
            .padding(.top, DS.Space.xs)
            .padding(.bottom, DS.Space.m)
        }
        .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 320)
        .onChange(of: sections.map(\.id)) { _, ids in
            if let current = selectedSection, !ids.contains(current) {
                selectedSection = ids.first
            }
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        let sections = filteredSections
        if let section = sections.first(where: { $0.id == selectedSection }) ?? sections.first {
            SectionDetailView(section: section, isFiltered: !query.isEmpty)
                .id(section.id)
        } else {
            DSEmptyState("No Results", systemImage: "magnifyingglass",
                         description: "No help topics match “\(searchText)”.")
        }
    }
}

// MARK: - Section Detail

private struct SectionDetailView: View {
    let section: HelpSection
    let isFiltered: Bool

    /// Items split into runs that start at each subtitle, rendered as one card per run.
    private var groups: [(title: String?, items: [HelpItem])] {
        var result: [(title: String?, items: [HelpItem])] = []
        for item in section.items {
            if item.isSubtitle {
                result.append((item.title, []))
            } else if result.isEmpty {
                result.append((nil, [item]))
            } else {
                result[result.count - 1].items.append(item)
            }
        }
        return result.filter { !$0.items.isEmpty }
    }

    private var eyebrow: String {
        guard let step = section.step else { return section.group.rawValue.uppercased() }
        if let n = step.stepNumber { return "STEP \(n) · \(step.rawValue.uppercased())" }
        return "TOOL · \(step.rawValue.uppercased())"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.xxl) {
                header
                ForEach(Array(groups.enumerated()), id: \.offset) { _, group in
                    VStack(alignment: .leading, spacing: DS.Space.s) {
                        if let title = group.title {
                            Text(title.uppercased())
                                .font(DS.Font.eyebrow)
                                .tracking(0.6)
                                .foregroundStyle(.secondary)
                                .padding(.leading, DS.Space.xs)
                        }
                        switch section.layout {
                        case .article:
                            // One card per topic: long paragraphs read better apart than ruled together.
                            VStack(alignment: .leading, spacing: DS.Space.m) {
                                ForEach(group.items) { item in
                                    HelpArticleRow(item: item)
                                        .dsCard(padding: 0)
                                }
                            }
                        case .shortcuts:
                            VStack(alignment: .leading, spacing: 0) {
                                ForEach(Array(group.items.enumerated()), id: \.element.id) { index, item in
                                    if index > 0 { Divider().padding(.leading, DS.Space.l) }
                                    HelpShortcutRow(item: item)
                                }
                            }
                            .dsCard(padding: 0)
                        }
                    }
                }
            }
            .padding(.horizontal, DS.Space.xxxl)
            .padding(.vertical, DS.Space.xxl)
            .frame(maxWidth: 780, alignment: .topLeading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(.background)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: DS.Space.l) {
            Image(systemName: section.icon)
                .font(.title)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(section.tint)
                .frame(width: 48, height: 48)
                .background(
                    RoundedRectangle(cornerRadius: DS.Radius.m, style: .continuous)
                        .fill(section.tint.opacity(0.14))
                )
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: DS.Space.xxs) {
                Text(eyebrow)
                    .font(DS.Font.eyebrow)
                    .tracking(0.6)
                    .foregroundStyle(.tertiary)
                Text(section.title)
                    .font(DS.Font.display)
                Text(section.summary)
                    .font(DS.Font.body)
                    .foregroundStyle(.secondary)
                if isFiltered {
                    Text("Showing topics that match your search")
                        .font(DS.Font.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer(minLength: 0)
            if let step = section.step {
                KeyCap("⌘\(step.shortcutNumber)")
                    .help("Shortcut for \(step.rawValue) in the main window")
            }
        }
    }
}

// MARK: - Rows

private struct HelpArticleRow: View {
    let item: HelpItem

    var body: some View {
        HStack(alignment: .top, spacing: DS.Space.l) {
            if let icon = item.icon {
                Image(systemName: icon)
                    .font(DS.Font.body.weight(.medium))
                    .foregroundStyle(DS.Color.accent)
                    .frame(width: DS.Size.toolButton, height: DS.Size.toolButton)
                    .background(
                        RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                            .fill(DS.Color.accentSoft)
                    )
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: DS.Space.xs) {
                Text(item.title)
                    .font(DS.Font.sectionTitle)
                if !item.description.isEmpty {
                    Text(Self.keycapStyled(item.description))
                        .font(DS.Font.label)
                        .lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)

            if let shortcut = item.shortcut {
                KeyCap(shortcut)
            }
        }
        .padding(.horizontal, DS.Space.l)
        .padding(.vertical, DS.Space.m + DS.Space.xxs)
    }

    /// Modifier-key shortcuts inside running text (⌘O, ⇧⌘Y, ⌃`, ⌃⌘S…).
    private static let shortcutPattern = #/[⌃⌥⇧⌘]+(?:Return|Tab|Space|Delete|[A-Z0-9`\[\]?,])/#

    /// Body text with every inline shortcut set as a small keycap (tinted run padded by thin spaces).
    static func keycapStyled(_ text: String) -> AttributedString {
        var result = AttributedString()
        var cursor = text.startIndex
        for match in text.matches(of: shortcutPattern) {
            result += AttributedString(text[cursor..<match.range.lowerBound])
            var cap = AttributedString("\u{2009}\(text[match.range])\u{2009}")
            cap.font = DS.Font.secondary.weight(.semibold)
            cap.foregroundColor = .primary
            cap.backgroundColor = DS.Color.controlFill
            result += cap
            cursor = match.range.upperBound
        }
        result += AttributedString(text[cursor...])
        return result
    }
}

private struct HelpShortcutRow: View {
    let item: HelpItem

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.l) {
            KeyCap(item.title)
                .frame(minWidth: DS.Size.labelColumn, alignment: .leading)
            Text(item.description)
                .font(DS.Font.label)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, DS.Space.l)
        .padding(.vertical, DS.Space.m)
        .accessibilityElement(children: .combine)
    }
}

/// A keyboard-key style chip for shortcuts and mouse gestures.
private struct KeyCap: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
        Text(text)
            .font(DS.Font.secondary.weight(.medium))
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, DS.Space.s)
            .padding(.vertical, DS.Space.xxs)
            .background(shape.fill(DS.Color.controlFill))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
    }
}

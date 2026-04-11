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

// MARK: - Help Content

private let helpSections: [HelpSection] = [

    // ── 1. Getting Started ──────────────────────────────────────────────
    HelpSection(
        icon: "play.circle.fill",
        title: "Getting Started",
        color: .blue,
        items: [
            HelpItem("Pipeline Overview",
                     "Druse guides you through a 7-step pipeline shown in the toolbar: Search → Preparation → Sequence → Ligands → Docking → Results → Lead Opt. Each step unlocks as prerequisites are met. A green checkmark means the step is complete; an orange dot means it is ready to use."),
            HelpItem("Quick Start: Redocking",
                     "1) Open a PDB file containing a co-crystallised ligand (File → Open or ⌘O). 2) Click Preparation → Prepare for Docking. 3) Switch to Ligands, the co-crystallised ligand appears automatically — click Populate & Prepare. 4) Go to Docking, a pocket is auto-detected around the ligand. 5) Click Run Docking. Results appear in the Results tab."),
            HelpItem("Quick Start: Virtual Screening",
                     "1) Load a protein and prepare it. 2) Open the Ligand Database (⌘L), import an SDF library, and batch-prepare all entries. 3) In the Docking tab, detect or define a pocket, then click Run Docking. All prepared ligands are docked and ranked."),
            HelpItem("Apple Silicon GPU",
                     "Druse runs all compute-intensive tasks — grid map generation, genetic algorithm, neural network scoring, Connolly surfaces — on the Apple Silicon GPU via Metal compute shaders. No external GPU or CUDA installation is required."),
        ]
    ),

    // ── 2. Loading Structures ───────────────────────────────────────────
    HelpSection(
        icon: "doc.text.fill",
        title: "Loading Structures",
        color: .blue,
        items: [
            HelpItem("Open File",
                     "Load a protein or ligand from PDB, mmCIF, MOL2, SDF, or SMILES files. Multiple files can be loaded sequentially — the most recent protein replaces the previous one, while ligands accumulate in the database.", shortcut: "⌘O"),
            HelpItem("Drag & Drop",
                     "Drag any supported file directly onto the 3D viewport. Druse detects the file type automatically."),
            HelpItem("Fetch by PDB ID",
                     "In the Search tab, type a 4-character PDB accession code (e.g. 1HSG) and press Enter. Druse downloads the structure from RCSB PDB and loads it."),
            HelpItem("Search RCSB",
                     "Enter a keyword (e.g. \"HIV protease\") in the Search tab. Results are displayed as cards showing resolution, method, and release date. Click a card to fetch and load the structure."),
            HelpItem("Save / Open Project",
                     "Projects (.druse folder) save the full session: protein, ligands, docking results, grid box, workspace settings, and preparation state. Re-open a project to resume exactly where you left off.", shortcut: "⌘S / ⌘⇧O"),
            HelpItem("Ligand Database Persistence",
                     "The ligand database can be saved and loaded independently of projects. Use File → Save/Load Ligand Database to keep a reusable compound library across sessions.", shortcut: "⌘⇧S"),
        ]
    ),

    // ── 3. Protein Preparation ──────────────────────────────────────────
    HelpSection(
        icon: "wrench.and.screwdriver.fill",
        title: "Protein Preparation",
        color: .orange,
        items: [
            HelpItem("Why Prepare?",
                     "Crystal structures from the PDB often lack hydrogens, contain alternate conformations, have missing loops or atoms, and do not include protonation states. Proper preparation is critical for accurate docking — incorrect protonation alone can shift binding energies by several kcal/mol."),
            HelpItem.subtitle("Automatic Pipeline"),
            HelpItem("Prepare for Docking",
                     "One-click pipeline that sequentially: 1) selects the best alternate conformer per residue (by occupancy, then B-factor), 2) removes non-standard residues and waters, 3) rebuilds missing heavy atoms from residue templates, 4) caps chain breaks with ACE/NME, 5) adds hydrogens at the specified pH, 6) assigns partial charges. The pH slider (default 7.4) controls protonation states."),
            HelpItem.subtitle("Individual Steps"),
            HelpItem("Add Hydrogens",
                     "Adds polar or all hydrogens based on residue templates. Protonation states are assigned according to the target pH using a PROPKA-style pKa predictor that accounts for burial, hydrogen bonding, Coulomb interactions, and desolvation effects."),
            HelpItem("Strip Hydrogens",
                     "Removes all hydrogen atoms. Useful when re-preparing at a different pH or when importing a structure with incorrect protonation."),
            HelpItem("Assign Charges",
                     "Applies protonation states and assigns Gasteiger partial charges to all atoms. Ionisable residues (Asp, Glu, His, Lys, Arg, Cys, Tyr, N/C-termini) are titrated based on their predicted pKa at the target pH."),
            HelpItem("Repair Gaps",
                     "Detects chain breaks (Cα–Cα distance > 4.0 Å), reports missing residue ranges, rebuilds missing heavy atoms from standard templates, and adds ACE/NME capping groups at unresolved termini. Short missing loops (≤15 residues) can be rebuilt."),
            HelpItem("Validate",
                     "Checks each residue against its template and reports: missing atoms (with completeness %), extra atoms, unusual bond lengths (deviation from ideal), and steric clashes (contacts < 2.0 Å). Flagged issues appear as colour-coded badges in the Preparation tab."),
            HelpItem.subtitle("Advanced Options"),
            HelpItem("H-Bond Network Optimisation",
                     "After adding hydrogens, Druse optimises the orientation of rotatable polar groups. It considers: hydroxyl rotations (Ser, Thr, Tyr — 3 orientations each), thiol rotations (Cys), amino rotations (Lys), amide flips (Asn OD1↔ND2, Gln OE1↔NE2), and histidine tautomers (6 states: 3 tautomers × 2 flip orientations). Conflicting groups are identified via a spatial hash and resolved as independent cliques using greedy energy minimisation."),
            HelpItem("Bridging Waters",
                     "Some water molecules mediate key protein–ligand hydrogen bonds. Enable this option to retain crystallographic waters near the binding pocket as part of the rigid receptor during docking."),
            HelpItem("pH Slider",
                     "Sets the target pH for protonation (range 2.0–12.0, default 7.4). This affects the charge states of ionisable residues and directly influences docking energetics — for example, at pH 4.0 Asp/Glu remain protonated (neutral), while at pH 7.4 they are deprotonated (negative)."),
        ]
    ),

    // ── 4. Sequence View ────────────────────────────────────────────────
    HelpSection(
        icon: "text.alignleft",
        title: "Sequence View",
        color: .indigo,
        items: [
            HelpItem("Overview",
                     "The Sequence tab displays a high-performance, scrollable canvas of one-letter amino-acid codes for each chain. It is not a list of SwiftUI views — it uses a custom canvas renderer for smooth scrolling even with very large proteins."),
            HelpItem.subtitle("Visual Encoding"),
            HelpItem("Secondary Structure Colouring",
                     "Each residue cell is colour-coded by secondary structure assignment: red (α-helix), yellow (β-sheet), grey (coil), cyan (turn). A legend is shown at the top of the view."),
            HelpItem("Gap Markers",
                     "Missing residues appear as orange gap markers between resolved segments. The marker shows the number of missing residues (e.g. \"gap 5\"). These correspond to unresolved density in the crystal structure."),
            HelpItem("Nucleic Acid Chains",
                     "DNA and RNA chains are displayed in a separate section below protein chains, using standard single-letter nucleotide codes."),
            HelpItem.subtitle("Selection"),
            HelpItem("Click to Select",
                     "Click a residue cell to select it. The corresponding residue highlights in the 3D viewport. The bottom panel shows details: sequence number, residue name, atom count, and secondary structure."),
            HelpItem("Select by Secondary Structure",
                     "Use the toolbar menu to select all helices, sheets, coils, or turns across the entire protein. Useful for visual analysis or defining docking constraints on structural motifs."),
            HelpItem.subtitle("Chain Operations"),
            HelpItem("Chain Context Menu",
                     "Right-click a chain header to access: Select Chain (select all residues), Copy Chain Sequence (clipboard), Rename Chain (change chain ID), Merge Chains (combine two chains), Delete Residues (remove selected, with confirmation)."),
            HelpItem("Copy Sequence",
                     "The toolbar copy button exports the full FASTA sequence of the selected chain to the clipboard."),
            HelpItem("Collapse / Expand",
                     "Click the chain header disclosure triangle to collapse or expand individual chains, keeping the view focused on the chain of interest."),
        ]
    ),

    // ── 5. Ligand Preparation ───────────────────────────────────────────
    HelpSection(
        icon: "atom",
        title: "Ligand Preparation",
        color: .purple,
        items: [
            HelpItem("Ligand Database",
                     "All ligands are managed through the Ligand Database (⌘L). The database stores molecular properties (MW, LogP, TPSA, HBD, HBA, RotB), preparation status, and enumerated forms. Columns are sortable and filterable; you can filter by Lipinski compliance or search by name/SMILES.", shortcut: "⌘L"),
            HelpItem.subtitle("Adding Ligands"),
            HelpItem("SMILES Input",
                     "Type or paste a SMILES string in the quick-entry field (Ligands tab or database window). Druse converts it to a 3D conformer via RDKit with MMFF94 force-field optimisation."),
            HelpItem("Import File",
                     "Import ligands from SDF (multiple molecules), MOL2, or SMILES files. SDF files preserve 3D coordinates if present. All imported molecules are added to the database."),
            HelpItem("Co-crystallised Ligand",
                     "When loading a PDB with a bound ligand, it is automatically extracted and added to the database. Use \"From Ligand\" pocket detection to define the binding site around it."),
            HelpItem.subtitle("Preparation Pipeline"),
            HelpItem("Populate & Prepare",
                     "Full preparation pipeline: add polar hydrogens → MMFF94 energy minimisation → Gasteiger charge assignment → enumerate tautomers/protomers at target pH → generate 3D conformers. A ligand must be prepared (green badge) before it can be docked."),
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
                     "3D conformers generated per tautomer/protomer (default 50). These serve as starting geometries for the genetic algorithm. More conformers improve sampling but increase preparation time."),
            HelpItem.subtitle("Enumeration Badges"),
            HelpItem("Form Types",
                     "Each enumerated form is labelled: Parent (green, original input), Tautomer (cyan, bond rearrangement), Protomer (orange, charge state change), Tautomer-Protomer (purple, both). In ensemble docking, all forms are docked and the best-scoring form is reported."),
        ]
    ),

    // ── 6. Pocket Detection ─────────────────────────────────────────────
    HelpSection(
        icon: "target",
        title: "Pocket Detection",
        color: .green,
        items: [
            HelpItem("Overview",
                     "Before docking, you must define where on the protein the ligand should bind. Druse offers four detection methods plus manual grid box placement. Detected pockets are ranked by druggability score and display their volume (ų), buriedness (%), and residue count."),
            HelpItem.subtitle("Detection Methods"),
            HelpItem("Auto (Hybrid)",
                     "Combines ML and geometric detection. First attempts GNN-based prediction; falls back to alpha-sphere/DBSCAN if the ML model is unavailable or finds no pockets. Recommended for most use cases."),
            HelpItem("ML Detection (GNN)",
                     "A graph neural network trained on known binding sites. More reliable for cryptic or shallow pockets that geometric methods may miss. Requires the pocket detector CoreML model (loaded at startup)."),
            HelpItem("Alpha-Sphere / DBSCAN",
                     "Geometry-based algorithm: 1) Places probe points on a 1.0 Å grid around the protein. 2) Retains probes 2.0–5.5 Å from the nearest protein atom. 3) Scores buriedness by casting 26 rays and counting how many are blocked by protein (≥30% required). 4) Clusters surviving probes with DBSCAN (ε = 3.5 Å, minPts = 8). 5) Ranks pockets by druggability = volume × buriedness × (residue count / 100)."),
            HelpItem("From Ligand",
                     "Defines a pocket centred on the current ligand position. Ideal when a co-crystallised ligand is present — the grid box tightly wraps the known binding site."),
            HelpItem("From Selection",
                     "Builds a pocket from residues you select in the Sequence tab or 3D viewport. Useful for targeting allosteric sites or known functional residues."),
            HelpItem.subtitle("Grid Box"),
            HelpItem("Grid Box Controls",
                     "The docking grid box defines the search volume. Adjust center (X, Y, Z) and half-size (X, Y, Z) with sliders. Quick-placement buttons snap the box to the protein centroid, ligand position, selected residues, or detected pocket."),
            HelpItem("Visualisation",
                     "The grid box appears as a wireframe cube in the 3D viewport. Line thickness and colour are adjustable in Settings. Use ⌘-drag to move the box and ⌘⇧-drag to resize interactively."),
            HelpItem.subtitle("Pocket View"),
            HelpItem("Z-Slab Clipping",
                     "Clips the 3D view to a depth slab around the pocket for unobstructed inspection. Controls: Thickness (2–40 Å, orange warning below 10 Å), Offset (−20 to +20 Å), Presets (Tight 8 Å / Medium 16 Å / Wide 30 Å). Enabling a slab preset automatically turns on the molecular surface at 55% opacity for cavity visualisation."),
        ]
    ),

    // ── 7. Docking – Scoring Functions ──────────────────────────────────
    HelpSection(
        icon: "function",
        title: "Scoring Functions",
        color: .yellow,
        items: [
            HelpItem("Overview",
                     "Druse provides four scoring functions. The choice affects both how poses are ranked during the genetic algorithm search and the units of the final score. You can switch scoring functions in the Results tab to rescore existing poses without re-docking."),
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
                     "Can be used as the primary scoring function during docking or as a post-docking rescorer (enable \"AF Rescore\" in advanced settings). Neural scoring generalises well across chemotypes but may be less reliable for very unusual scaffolds."),
            HelpItem.subtitle("PIGNet2 (Physics-Informed GNN)"),
            HelpItem("PIGNet2 Architecture",
                     "A graph neural network (128-dim, 3 GatedGAT layers, 5 Å intermolecular cutoff) that decomposes its prediction into physics-interpretable components: van der Waals (Morse potential), hydrogen bonds, metal coordination, hydrophobic contacts, and rotor entropy. Outputs kcal/mol. Max 256 protein atoms, 64 ligand atoms per evaluation."),
        ]
    ),

    // ── 8. Docking – Search Algorithms ──────────────────────────────────
    HelpSection(
        icon: "bolt.fill",
        title: "Docking Algorithms",
        color: .orange,
        items: [
            HelpItem("Overview",
                     "Druse uses a GPU-parallel search to explore the binding pose space. The search method is auto-selected based on ligand flexibility, or you can choose manually. All methods run entirely on the Metal GPU."),
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

    // ── 9. Docking – Parameters ─────────────────────────────────────────
    HelpSection(
        icon: "slider.horizontal.3",
        title: "Docking Parameters",
        color: .red,
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
            HelpItem.subtitle("Exploration Tuning"),
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
                     "Sample sidechain χ angles for selected pocket-lining residues during docking. Two modes: Soft (automatic selection of sidechains near the pocket, purple badges show selected residues and χ-angle count) or Manual (hand-pick up to 8 residues). Increases computational cost but captures induced-fit effects."),
            HelpItem.subtitle("Constraints"),
            HelpItem("Pharmacophore Constraints",
                     "Force specific interactions during docking. Add constraints by clicking atoms/residues in the 3D viewport: H-bond acceptor/donor, aromatic ring, lipophilic contact, positive/negative charge. Each constraint has a strength (Hard = strict requirement; Soft = 5.0 kcal/mol penalty) and distance threshold (2.5–4.0 Å). Poses violating hard constraints are eliminated; soft constraint violations add an energy penalty."),
            HelpItem.subtitle("Post-Docking"),
            HelpItem("GFN2-xTB Refinement",
                     "Semi-empirical quantum-mechanical geometry optimisation of top poses (default: top 20). Uses GFN2-xTB with D4 dispersion and implicit water solvation (ALPB or GBSA). Options: optimisation level (crude/normal/tight), harmonic position restraint (0.005 Ha/Å² ≈ 0.3 Å allowed motion), max RMSD filter (2.0 Å), blend weight (0.3 = 30% GFN2 + 70% Vina). Improves pose quality for tight binders."),
            HelpItem("AF Rescore",
                     "After docking, rescore top cluster representatives with DruseAF v4 neural network. Particularly useful when Vina/Drusina ranking disagrees with expected SAR."),
            HelpItem.subtitle("Clash & Strain"),
            HelpItem("Clash Handling",
                     "Atom overlaps exceeding 0.4 Å trigger a penalty of 5.0 kcal/mol per Å excess. Prevents physically impossible poses from dominating the population."),
            HelpItem("Strain Penalty",
                     "Post-docking MMFF94 internal energy check. Poses with intramolecular strain > 6.0 kcal/mol above the relaxed conformer receive a weighted penalty (default weight 0.5). High strain values (> 4.0) are flagged in red in the Results tab."),
        ]
    ),

    // ── 10. Results & Analysis ──────────────────────────────────────────
    HelpSection(
        icon: "chart.bar.fill",
        title: "Results & Analysis",
        color: .cyan,
        items: [
            HelpItem("Results Tab",
                     "After docking, poses are ranked by score and grouped into clusters (RMSD threshold = 2.0 Å). Each cluster represents a distinct binding mode. The summary card shows: best energy, pose count, cluster count. Click any pose to display it in the 3D viewport."),
            HelpItem.subtitle("Energy Landscape"),
            HelpItem("Histogram",
                     "An interactive bar chart showing the energy distribution of the top 50 poses. Bars are colour-coded: green (< −8 kcal/mol, strong binding) through yellow to red (poor). Click a bar to view the corresponding pose. The spread of the histogram indicates convergence — a narrow peak suggests the search converged well."),
            HelpItem.subtitle("Pose Details"),
            HelpItem("Energy Decomposition",
                     "Each pose shows tagged energy components: vdW (van der Waals), elec (electrostatic), hb (hydrogen bond), str (strain — red warning if > 4.0), xtb (GFN2-xTB energy if refined, purple), cst (constraint penalty if any, orange)."),
            HelpItem("Cluster Badges",
                     "Each pose shows its cluster ID (C0, C1, etc.). Poses in the same cluster have similar binding modes (RMSD < 2.0 Å). The cluster representative (rank 0) is typically the best-scoring member."),
            HelpItem("Scoring Method Switching",
                     "Use the dropdown at the top to rescore poses with a different function (Vina, Drusina, Druse AF, PIGNet2) without re-running the search. This lets you compare rankings across methods."),
            HelpItem.subtitle("Interaction Diagram"),
            HelpItem("2D Interaction Diagram",
                     "A schematic view of protein–ligand contacts with the ligand centred and interacting residues arranged radially. Interactions are colour-coded: H-bonds (cyan), salt bridges (orange), π-stacking (purple), hydrophobic (grey), halogen bonds (green), metal coordination (gold). Residues are coloured by type: acidic (red), basic (blue), polar (green), hydrophobic (orange). Distances are labelled on connecting lines."),
            HelpItem.subtitle("Results Database"),
            HelpItem("Full Pose Analysis",
                     "The Results Database window (⌘⇧R) provides a comprehensive table of all poses with sortable columns, multi-column selection, energy bars, and a detail panel showing 2D interaction diagrams, pocket metrics, and per-atom energy contributions.", shortcut: "⌘⇧R"),
            HelpItem("Correlation Analysis",
                     "In batch mode, the correlation view shows a scatter plot of docking scores vs experimental affinity (if Ki/Kd values are provided). Includes a regression line with R² value and outlier identification. Useful for validating the scoring function against known actives/inactives."),
            HelpItem.subtitle("Export"),
            HelpItem("Export Poses",
                     "Export selected poses as SDF (with 3D coordinates and energy properties) or the results table as CSV. Available from both the Results tab and the Database window."),
        ]
    ),

    // ── 11. Virtual Screening ───────────────────────────────────────────
    HelpSection(
        icon: "rectangle.stack.fill",
        title: "Virtual Screening",
        color: .mint,
        items: [
            HelpItem("Overview",
                     "Virtual screening docks an entire ligand database against the target pocket. All prepared ligands in the database are queued automatically when you click Run Docking. Results are ranked by best score per compound."),
            HelpItem("Batch Progress",
                     "The Results tab shows batch docking progress with per-ligand status. Each compound is docked independently using the same grid maps and search parameters."),
            HelpItem("Hit Ranking",
                     "Compounds are ranked by their best pose energy. The batch results list shows: rank, ligand name, best energy, pose count, and estimated Ki (if available)."),
            HelpItem("Filtering",
                     "Filter hits by affinity threshold (discard compounds weaker than a cutoff), RMSD cutoff (only keep well-converged poses), and ADMET properties (Lipinski, Veber, hERG safety)."),
            HelpItem("Ensemble Docking",
                     "When ligands have multiple enumerated forms (tautomers/protomers), all forms are docked and the best-scoring form determines the compound's rank. Population cutoff (default 5%) filters rare forms. Population weighting adjusts scores by Boltzmann probability."),
        ]
    ),

    // ── 12. Lead Optimisation ───────────────────────────────────────────
    HelpSection(
        icon: "wand.and.stars",
        title: "Lead Optimisation",
        color: .pink,
        items: [
            HelpItem("Overview",
                     "Starting from a docked pose, the Lead Opt tab generates and evaluates structural analogues. Workflow: 1) Dock a compound, 2) In Results, click \"Optimize\" on your best pose, 3) Configure analogue generation, 4) Generate and dock analogues, 5) Compare with the reference."),
            HelpItem.subtitle("Generation Controls"),
            HelpItem("Analogue Count",
                     "Number of analogues to generate: 10, 25, 50, or 100. More analogues explore more chemical space but take longer to dock."),
            HelpItem("Similarity Threshold",
                     "Tanimoto similarity to the reference compound (0.3–0.95). Higher values generate closer analogues; lower values explore more diverse scaffolds."),
            HelpItem("Keep Scaffold",
                     "When enabled, only R-groups are modified — the core scaffold is preserved. Useful for SAR exploration around a validated binding mode."),
            HelpItem("Property Direction",
                     "Four sliders bias analogue generation: Polarity (less ↔ more), Rigidity (flexible ↔ rigid), Lipophilicity (hydrophilic ↔ lipophilic), Size (smaller ↔ larger). Centre position = no bias."),
            HelpItem.subtitle("ADMET Filters"),
            HelpItem("Drug-Likeness Filters",
                     "Pre-filter analogues before docking: Lipinski Rule of 5 (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10), Veber (TPSA ≤ 140, RotB ≤ 10), hERG safety (cardiac toxicity risk), CYP safety (metabolic liability), Max LogP (1–8 lipophilicity cap). Filtered compounds are marked orange; failed compounds are marked red."),
            HelpItem.subtitle("Analysis"),
            HelpItem("Comparison Card",
                     "Select an analogue to see a side-by-side comparison with the reference: property differences (ΔMW, ΔLogP, ΔTPSA, ΔRotB), ADMET status icons, docking energy difference, and Tanimoto similarity. A comparison sheet shows overlay structures and a detailed ADMET profile heatmap."),
            HelpItem("3D Alignment",
                     "The analogue's best pose is aligned to the reference pose in the 3D viewport, allowing direct visual comparison of binding modes."),
        ]
    ),

    // ── 13. 3D Viewport ─────────────────────────────────────────────────
    HelpSection(
        icon: "move.3d",
        title: "3D Viewport",
        color: .teal,
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
            HelpItem("Recenter",
                     "Fit all visible atoms to the viewport with smooth animation.", shortcut: "Space"),
            HelpItem.subtitle("Selection"),
            HelpItem("Click to Select",
                     "Single-click an atom (or residue in ribbon mode) to select it. Selection highlights in the viewport, Sequence tab, and inspector."),
            HelpItem("Multi-Select",
                     "Option + click to toggle atoms in and out of the selection without clearing previous selections.", shortcut: "⌥ Click"),
            HelpItem("Box Select",
                     "Option + drag to draw a selection rectangle. All atoms within the box are selected. Shift + box select adds to the existing selection.", shortcut: "⌥ Drag"),
            HelpItem("Deselect All",
                     "Press Escape or click on empty space to clear the selection.", shortcut: "Esc"),
            HelpItem.subtitle("Render Modes"),
            HelpItem("Ball & Stick",
                     "Atoms as impostor spheres, bonds as impostor cylinders. Default mode for detailed atom inspection. GPU ray-sphere/ray-cylinder intersection gives pixel-perfect smooth rendering.", shortcut: "⌘1"),
            HelpItem("Space Filling (CPK)",
                     "Van der Waals spheres at full radius. Shows the molecular surface and steric interactions. Useful for visualising clashes and surface complementarity.", shortcut: "⌘2"),
            HelpItem("Wireframe",
                     "Bonds only, no atom spheres. Minimal visual clutter — good for large proteins or dense binding sites.", shortcut: "⌘3"),
            HelpItem("Ribbon (Cartoon)",
                     "Secondary-structure cartoon: α-helices as ribbons, β-sheets as arrows, coils as tubes. Click a ribbon element to select the corresponding residue. Best for protein fold overview.", shortcut: "⌘4"),
            HelpItem.subtitle("Display Options"),
            HelpItem("Toggle Hydrogens",
                     "Show or hide hydrogen atoms globally. Hiding hydrogens reduces clutter in crowded binding sites.", shortcut: "⌘H"),
            HelpItem("Molecular Surface",
                     "Compute and display the Connolly solvent-excluded surface (probe radius 1.4 Å) via Metal marching-cubes GPU kernel. Colour modes: Uniform (solid), Electrostatic (red/blue ESP gradient), Hydrophobicity (polar blue ↔ lipophilic orange), Pharmacophore (interaction-type colours). Opacity is adjustable (default 85%)."),
            HelpItem("Atom Labels",
                     "Overlay labels on atoms: none, element symbol, atom index, or PDB serial number. Accessible from the toolbar."),
            HelpItem("Lighting Mode",
                     "Uniform (flat, no depth cues) or Directional (shaded with fixed light direction for 3D perception). Directional is better for presentations and screenshots."),
            HelpItem.subtitle("Colouring"),
            HelpItem("CPK (Element) Colours",
                     "Default atom colouring by element: C grey, N blue, O red, S yellow, H light grey, halogens green/orange. Standard across chemistry."),
            HelpItem("Chain Colours",
                     "Each chain receives an automatic palette colour. Override in the Inspector by clicking the chain colour swatch and choosing from 16 presets or a custom colour picker."),
            HelpItem("Ligand Carbon Colour",
                     "Set a custom colour for ligand carbon atoms to distinguish the ligand from protein atoms. Choose from 16 presets in the Inspector."),
            HelpItem.subtitle("Ghost Ligand"),
            HelpItem("Docking Preview",
                     "During docking, a translucent \"ghost\" ligand shows the current best pose updating in real time. Ghost atoms/bonds are drawn with read-only depth and 50–70% opacity so they overlay the protein without obscuring it."),
        ]
    ),

    // ── 14. Inspector & Display ─────────────────────────────────────────
    HelpSection(
        icon: "info.circle.fill",
        title: "Inspector & Status",
        color: .gray,
        items: [
            HelpItem.subtitle("Inspector Panel"),
            HelpItem("Atom Details",
                     "When an atom is selected, the inspector shows: atom index, name, element, coordinates (X, Y, Z), van der Waals radius, partial charge, parent residue, and chain ID."),
            HelpItem("Residue Details",
                     "For a selected residue: sequence number, residue name (3-letter code), atom count, secondary structure assignment, and completeness percentage."),
            HelpItem("Chain Colour Picker",
                     "Click the colour swatch next to a chain to choose from: CPK mode (element colouring), Chain default (automatic palette), or Custom (16 preset colours: blue, cyan, teal, green, lime, yellow, orange, red, pink, purple, grey, brown, slate, etc.)."),
            HelpItem.subtitle("Status Strip"),
            HelpItem("Bottom Status Bar",
                     "Shows real-time information: current operation (loading, docking, preparing), progress bar for batch operations, molecule counts (atoms, residues), selection info, grid dimensions, and memory usage (VRAM/RAM)."),
            HelpItem("Log Console",
                     "Expandable console at the bottom of the window. Entries are colour-coded: green (success), blue (info), orange (warning), red (error). Filter by level, search text, and auto-expand on errors."),
            HelpItem.subtitle("Settings"),
            HelpItem("Appearance",
                     "Theme (Auto/Light/Dark), viewport background gradient intensity (0 = solid for screenshots, 1 = full gradient), grid box line thickness (1–8 px) and colour."),
            HelpItem("Logging",
                     "Enable file logging, choose log directory, set retention period (default 7 days), manually archive or clear logs."),
        ]
    ),

    // ── 15. Keyboard Shortcuts ──────────────────────────────────────────
    HelpSection(
        icon: "keyboard.fill",
        title: "Keyboard Shortcuts",
        color: .gray,
        items: [
            HelpItem.subtitle("File"),
            HelpItem("⌘O", "Open file (PDB, mmCIF, SDF, MOL2, SMILES)"),
            HelpItem("⌘S", "Save project (.druse)"),
            HelpItem("⌘⇧O", "Open project"),
            HelpItem("⌘⇧S", "Save ligand database"),
            HelpItem.subtitle("View"),
            HelpItem("⌘1", "Ball & Stick render mode"),
            HelpItem("⌘2", "Space Filling (CPK) render mode"),
            HelpItem("⌘3", "Wireframe render mode"),
            HelpItem("⌘4", "Ribbon (cartoon) render mode"),
            HelpItem("⌘H", "Toggle hydrogen atoms"),
            HelpItem("Space", "Recenter / fit to view"),
            HelpItem("Esc", "Deselect all"),
            HelpItem.subtitle("Windows"),
            HelpItem("⌘L", "Open Ligand Database window"),
            HelpItem("⌘⇧R", "Open Results Database window"),
            HelpItem("⌘?", "Open this help window"),
            HelpItem.subtitle("Docking"),
            HelpItem("⌘⏎", "Run docking"),
            HelpItem.subtitle("3D Viewport"),
            HelpItem("Drag", "Orbit (rotate view)"),
            HelpItem("⇧ Drag", "Pan (translate view)"),
            HelpItem("Scroll", "Zoom in / out"),
            HelpItem("⌃ Drag", "Z-roll (spin around view axis)"),
            HelpItem("⌥ Click", "Toggle multi-select"),
            HelpItem("⌥ Drag", "Box selection"),
            HelpItem("⌘ Drag", "Move grid box"),
            HelpItem("⌘⇧ Drag", "Resize grid box"),
        ]
    ),

    // ── 16. Algorithms Reference ────────────────────────────────────────
    HelpSection(
        icon: "book.closed.fill",
        title: "Algorithms Reference",
        color: .brown,
        items: [
            HelpItem.subtitle("Grid Map Pre-computation"),
            HelpItem("Affinity Grids",
                     "Before docking, Druse pre-computes 3D grid maps for each Vina atom type. Each grid point stores the interaction energy contribution of a probe atom at that position. Stored as half-precision (float16) for VRAM efficiency. During scoring, the ligand's energy is evaluated via trilinear interpolation — O(N_atoms) per pose instead of O(N_atoms × N_protein_atoms)."),
            HelpItem.subtitle("RMSD Clustering"),
            HelpItem("Pose Clustering",
                     "After all runs complete, poses are clustered by pairwise RMSD (heavy atoms only, ignoring hydrogens). Threshold = 2.0 Å. Greedy leader algorithm: the best-scoring unassigned pose becomes a cluster representative; all poses within 2.0 Å join its cluster. GPU-accelerated pairwise RMSD computation when available, CPU parallel fallback otherwise."),
            HelpItem.subtitle("Explicit Reranking"),
            HelpItem("Top-Cluster Reranking",
                     "The top 12 cluster representatives are reranked using explicit atom-pair scoring (not grid interpolation). 4 variants per cluster are generated via short local search (20 steps), and the best variant replaces the grid-scored representative. This corrects grid interpolation artifacts near the scoring function minimum."),
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

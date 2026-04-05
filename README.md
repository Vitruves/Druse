<p align="center">
  <img src="Resources/Assets.xcassets/AppIcon.appiconset/icon_256x256.png" alt="Druse" width="128" height="128">
</p>

<h1 align="center">Druse</h1>

<p align="center">
  <strong>GPU-Accelerated Molecular Docking for macOS</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-macOS_26+-000000?style=flat-square&logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/Chip-Apple_Silicon-333333?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/GPU-Metal_3-8A2BE2?style=flat-square" alt="Metal">
  <img src="https://img.shields.io/badge/Version-0.1.18--beta-blue?style=flat-square" alt="Version">
</p>

<p align="center">
  <em>A native macOS application for structure-based drug discovery.<br>Built from the ground up for Apple Silicon with Metal compute shaders.</em>
</p>

<br>

---

<br>

## Download

<p align="center">
  <a href="https://github.com/vitruves/Druse/releases/latest">
    <img src="https://img.shields.io/badge/Download-Druse.dmg-blue?style=for-the-badge&logo=apple&logoColor=white" alt="Download DMG">
  </a>
</p>

> **Requirements** — macOS 26.0 or later, Apple Silicon (M1 or newer).
> Download the `.dmg` from [Releases](https://github.com/vitruves/Druse/releases/latest), open it, and drag Druse into Applications.

> **Note** — Druse is currently distributed unsigned. On first launch, macOS Gatekeeper will block the app.
> Right-click (or Control-click) `Druse.app` and select **Open**, then click **Open** in the confirmation dialog.
> Alternatively, run `xattr -cr /path/to/Druse.app` in Terminal.

<details>
<summary><strong>Build from Source</strong></summary>

<br>

**Prerequisites**

- macOS 26.0+, Apple Silicon (M1 or newer)
- Xcode 26.0+ (with Command Line Tools)
- [Homebrew](https://brew.sh)
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) 2.35+

**1. Install dependencies**

```bash
brew install rdkit boost eigen nanoflann tbb
brew install xcodegen
```

Optional (for GFN2-xTB quantum refinement):
```bash
brew install xtb
```

**2. Build the C++ core**

```bash
cd CppCore
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
cd ../..
```

OpenMM is fetched and built automatically by CMake. This step may take a few minutes on the first run.

**3. Generate the Xcode project and build**

```bash
xcodegen generate
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release build
```

The built app will be in `build/Release/Druse.app` (or under `DerivedData/`).

</details>

<p align="center">
  <img src="img/menu.png" alt="Druse welcome screen" width="700">
</p>

<br>

---

<br>

## Why Druse

Most docking software was designed for Linux clusters and command-line workflows. Druse takes a different approach — a fully native macOS application that puts the entire docking pipeline in a single window, accelerated by Metal on the GPU you already have.

| | |
|---|---|
| **Native & Fast** | Built in Swift and Metal. No Python runtimes, no Docker, no X11. Just a `.app` that launches instantly. |
| **GPU Everything** | 20+ Metal compute kernels — from genetic algorithm search to neural network scoring to surface rendering. All on your M-series chip. |
| **Unified Memory** | Apple Silicon's shared CPU/GPU memory means zero-copy data transfer between the docking engine, ML models, and 3D renderer. |
| **Interactive** | Real-time 3D visualization with impostor rendering, Connolly surfaces, and ribbon diagrams — all at retina resolution. |

<br>

---

<br>

## The Pipeline

Druse covers the full structure-based drug discovery workflow in a single application.

### Load & Prepare

<p align="center">
  <img src="img/protein.png" alt="Protein structure loaded in Druse" width="700">
</p>

- **Fetch from RCSB** — Enter a PDB ID and Druse downloads the structure directly
- **Format support** — PDB, mmCIF, SDF (V2000/V3000), MOL2, SMILES
- **Protein preparation** — Automated pipeline: water removal, alternate conformer selection, missing atom reconstruction, polar hydrogen addition, H-bond network optimization (Asn/Gln flips, His tautomers), sidechain packing (FASPR with Dunbrack rotamers), and energy minimization
- **Protonation** — pH-dependent protonation with table lookup or optional GFN2-xTB quantum pKa prediction
- **Partial charges** — Four methods: EEM (GPU-accelerated), QEq, Gasteiger, and RDKit template matching

### Detect Binding Sites

- **Alpha-sphere probing** — Geometric pocket detection with DBSCAN clustering and 26-direction buriedness scoring
- **ML-enhanced detection** — Optional neural network pocket predictor combined with geometric scoring
- **Ligand-guided** — Auto-center from a co-crystallized ligand, or define a custom box manually
- **Pocket metrics** — Volume (ų), buriedness score, druggability rating, and resident residue identification

### Dock

<p align="center">
  <img src="img/ligand.png" alt="Docking configuration with ligand in binding pocket" width="700">
</p>

- **Genetic Algorithm** — Population-based search with adaptive mutation, crossover, and iterated local search — all on GPU
- **Fragment-Based Docking** — Hierarchical fragment growth with anchor placement and beam search pruning
- **Parallel Tempering** — Replica exchange Monte Carlo for enhanced sampling of flexible ligands
- **Diffusion-Guided Docking** — Reverse diffusion process with DruseAF attention gradient guidance
- **Flexible Docking** — Simultaneous optimization of ligand pose and sidechain rotamers (3–6 residues)
- **Pharmacophore Constraints** — H-bond, hydrophobic, aromatic, salt bridge, and metal coordination constraints with configurable strength
- **Auto-Tuning** — Automatically adapts search parameters to protein size, pocket shape, and ligand flexibility
- **Analytical Gradients** — SIMD-cooperative local search, faster than numerical finite differences

### Score

Four scoring functions, each with different strengths:

| Scoring Function | Type | Description |
|---|---|---|
| **Vina** | Empirical | Gaussian + repulsion + hydrophobic + H-bond + torsion penalty |
| **Drusina** | Extended empirical | Vina baseline plus electrostatics, salt bridges, π-π stacking, π-cation, halogen bonds, metal coordination, CH-π, amide-π, chalcogen bonds, and torsion strain |
| **DruseAF v4** | Neural network | SE(3)-equivariant pairwise geometric network for pKd prediction |
| **PIGNet2** | Physics-informed GNN | Graph neural network with physics-based interaction constraints |

### Analyze

<p align="center">
  <img src="img/docking_result.png" alt="Docking results with electrostatic surface" width="700">
</p>

- **Pose clustering** — GPU-accelerated pairwise RMSD with hierarchical clustering and consensus pose extraction
- **Energy decomposition** — Per-term breakdown across all scoring components
- **Interaction detection** — 10+ interaction types with color-coded 3D visualization and 2D interaction diagrams

<p align="center">
  <img src="img/diagram.png" alt="2D interaction diagram" width="600">
</p>

- **Explicit atom rescoring** — Top clusters re-scored against explicit receptor atoms with basin hopping refinement
- **Strain filtering** — MMFF94 torsion strain penalty to flag unrealistic conformations
- **GFN2-xTB refinement** — Optional semi-empirical QM post-docking optimization with D4 dispersion and ALPB solvation

<br>

---

<br>

## Virtual Screening

Batch-dock up to 100,000 molecules with shared grid reuse and parallel 3D generation.

- **Pre-filtering** — Rotatable bond cutoff, rapid pre-scoring for fast rejection
- **ADMET filtering** — Lipinski Rule of Five, Veber rules
- **Multiple scoring functions** — Vina, Drusina, DruseAF v4, or PIGNet2 as the primary scorer
- **Ranked export** — CSV with full metrics and SDF with posed geometries

<br>

---

<br>

## Lead Optimization

Generate and evaluate analogs directly inside Druse.

- **Analog generation** — 18+ curated substitution rules: halogen swaps, alkyl extensions, heteroatom substitutions, aromatic replacements, functional group interchanges, ring size modifications
- **Property sliders** — Dial in polarity, rigidity, lipophilicity, and size to bias generation toward desired property space
- **Mini-docking** — Sub-second per-analog docking with RMSD tracking against the parent compound
- **ADMET gating** — Real-time Lipinski and Veber checks on every analog
- **Trade-off analysis** — Binding affinity vs. ADMET property landscape with Pareto frontier identification

<br>

---

<br>

## Visualization

Real-time Metal-rendered 3D molecular graphics, triple-buffered for smooth interaction.

- **Rendering modes** — Ball & stick, space filling, wireframe, ribbon (helix/sheet/coil)
- **Surfaces** — Connolly (solvent-accessible) and Gaussian blob isosurfaces with electrostatic potential, hydrophobicity, or pharmacophore coloring
- **Interactions** — Color-coded lines for hydrogen bonds, salt bridges, π-stacking, halogen bonds, metal coordination, and more
- **GPU picking** — Click any atom or residue directly in the 3D view for instant selection and inspection
- **Ghost ligands** — Translucent overlay of docking poses for comparison
- **Z-slab clipping** — Slice through the structure to focus on the binding site interior

<br>

---

<br>

## Ligand Library

A built-in molecular database for organizing your compounds.

- **Multi-format import** — SDF, MOL2, PDB files, co-crystallized ligands from loaded structures, or direct SMILES input
- **Chemical form enumeration** — Tautomers, protomers, and cross-enumeration with Boltzmann population weighting
- **Conformer generation** — RDKit 3D embedding with energy-ranked conformer selection
- **Batch preparation** — Hydrogen addition, charge calculation, and minimization across the library
- **Search & filter** — By name, SMILES, molecular weight, rotatable bonds, H-bond donors/acceptors

<br>

---

<br>

## Guided Demo

New to molecular docking? Druse includes a fully interactive guided walkthrough that docks Nafamostat into Trypsin (PDB: 3PTB) — from protein fetch to scored poses — with step-by-step narration. Just click **Demo** on the welcome screen.

<br>

---

<br>

## Metal Under the Hood

Druse runs 20+ specialized Metal compute kernels on Apple Silicon:

- Genetic algorithm evolution and selection
- Analytical gradient local search (SIMD-cooperative)
- Grid map generation (steric, hydrophobic, H-bond, affinity)
- Neural network inference (DruseAF v4, PIGNet2)
- EEM partial charge calculation
- Connolly surface generation
- Pairwise RMSD computation
- Fragment growth and diffusion denoising
- Flexible sidechain scoring
- Impostor sphere/cylinder rendering with depth-correct post-processing

Half-precision grid storage, shared memory tiling, and on-demand rendering keep memory usage low and battery life long.

<br>

---

<br>

## Benchmark

Redocking accuracy on the CASF-2016 benchmark set (Vina scoring, standard preset, ligand-guided pocket):

| Metric | Druse Vina | AutoDock Vina 1.2.7 |
|---|---|---|
| Docking power (RMSD < 2.0 A) | 19/26 (73%) | 25/26 (96%) |
| Scoring power (Pearson r vs pKd) | -0.59 | -0.56 |
| Avg time / complex | ~14s | ~67s |

Compared on 26 common CASF-2016 complexes (4 failed in AutoDock Vina preparation), Apple M-series chip. Full benchmark scripts in `Benchmark/`.

<br>

---

<br>

<p align="center">
  <strong>Druse</strong> — Molecular docking, native on your Mac.
</p>

<p align="center">
  <sub>All rights reserved.</sub>
</p>

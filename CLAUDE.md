# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Druse

Druse is a macOS-native molecular docking application built with Swift/SwiftUI, Metal GPU compute, and a C++ core backed by RDKit. It targets Apple Silicon with unified memory for zero-copy CPU/GPU data sharing. The app covers the full docking workflow: protein loading → preparation → ligand import → pocket detection → GPU-accelerated docking → ML re-ranking → virtual screening.

## Build Commands

### C++ Core (libdruse_core.a)
```bash
cd CppCore/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```
Or use the script: `CppCore/rebuild.sh`

**Prerequisites:** `brew install rdkit boost eigen nanoflann tbb`

### Xcode Project (from project.yml)
```bash
xcodegen generate
```

### Build the App
```bash
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Debug build
```

### Run Tests
```bash
# All tests except the slow redocking integration test (~4 min total)
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Debug test \
  -skip-testing:DruseTests/DruseTests/testRedocking1HSGIndinavir

# Only the 1HSG redocking integration test
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Debug test \
  -only-testing:DruseTests/DruseTests/testRedocking1HSGIndinavir
```

All 55+ tests live in a single file: `Tests/DruseTests.swift`.

## Architecture

### Three-layer stack

1. **C++ Core** (`CppCore/druse_core.h` + `druse_core.cpp`) — Static library exposing a C interface to RDKit. Handles SMILES→3D, conformers, Gasteiger charges, descriptors, fingerprints (ECFP4), torsion trees, Kabsch RMSD, KD-tree queries (nanoflann), and TBB-backed batch processing. Linked as `libdruse_core.a`.

2. **Metal GPU** (`Metal/`) — Four shader files:
   - `DockingCompute.metal` — Grid map generation (tiled atom loading), GA mutation/crossover, Vina-style pose scoring (gauss1+gauss2+repulsion, hydrophobic, H-bond, torsion penalty), ILS local search
   - `SurfaceCompute.metal` — Marching cubes, Connolly/SAS surface, buriedness scoring, ESP coloring
   - `RenderShaders.metal` — Impostor spheres (atoms), cylinders (bonds), ribbons, interaction dashed lines, grid wireframe, Z-slab clipping
   - `PostProcessing.metal` — MSAA resolve with Z-clipping

3. **Swift App** — SwiftUI frontend, Metal rendering, docking orchestration, ML inference

### Swift-C++ bridging

`Metal/Druse-Bridging-Header.h` includes both `ShaderTypes.h` (shared GPU/CPU types) and `CppCore/druse_core.h`. Swift calls C functions from `RDKitBridge.swift`. The `ShaderTypes.h` defines all shared structs: `GridParams`, `DockPose`, `GAParams`, `Uniforms`, `VinaAtomType`, vertex/instance types.

### Key Swift modules

| Directory | Role |
|-----------|------|
| `ViewModels/AppViewModel.swift` | Central `@Observable` state: protein, ligand, render mode, selection, docking config/results, surface generation. This is the largest file (~97K). |
| `Docking/DockingEngine.swift` | GA orchestration, Metal kernel dispatch, torsion application, interaction detection, pose clustering. Second largest (~57K). |
| `Docking/VirtualScreening.swift` | Batch workflow: per-ligand prep, shared grid reuse, GPU docking, ML re-ranking, ADMET, export (~44K). |
| `Docking/BindingSite.swift` | Grid map generation, pocket analysis, Vina-style scoring grid (~38K). |
| `ML/Inference.swift` | DruseScore CoreML inference: feature extraction, atom encoding, RBF distances, multi-task heads (~28K). |
| `Renderer/Renderer.swift` | Metal render pipelines, buffer management, hit-testing, Z-clipping (~28K). |
| `Chemistry/PDBParser.swift` | PDB format parsing with HELIX/SHEET, secondary structure (~18K). |
| `Chemistry/RDKitBridge.swift` | Swift wrapper over C core: SMILES→Molecule, batch processing (~13K). |
| `Chemistry/PocketDetection.swift` | Alpha-sphere probe + DBSCAN clustering, Metal-accelerated buriedness (~12K). |

### Data flow

```
PDB/mmCIF/SDF/MOL2 → Parsers → Molecule model (atoms, bonds, residues, chains)
SMILES → RDKitBridge → 3D coords + conformers + charges → Molecule
Molecule → ProteinPreparation (protonation, charges) → ready for docking
Pocket → BindingSite → GridParams → DockingCompute.metal (grid maps)
DockingEngine → Metal GA kernels → scored poses → interaction detection
Top poses → DruseScore CoreML (optional re-ranking) → results
```

### Rendering

Uses impostor rendering (sphere/cylinder billboards) for performance. `MetalView.swift` is the `NSViewRepresentable` bridge handling mouse/trackpad input. Camera supports orthographic with arcball rotation. MSAA is enabled — color attachment store action **must** be `.storeAndMultisampleResolve` (not `.store`), or the screen goes black.

### Concurrency

- `@MainActor` on `Renderer`, `Camera`, all UI
- `@Observable` on `AppViewModel` for reactive state
- `SurfaceGenerator` runs as async background task
- C++ batch processing uses TBB parallelism

## Scoring

The docking engine uses Vina-compatible scoring: steric (gauss1 + gauss2 + repulsion), hydrophobic, hydrogen bond, and torsion penalty terms. Atom types follow the `VinaAtomType` enum (16 types) defined in `ShaderTypes.h`. Grid maps are pre-computed on GPU with tiled atom loading for cache efficiency.

## Training (Python, not part of the app)

`Training/` contains PyTorch scripts for DruseScore (SE(3)-equivariant GNN), pocket detector, and ADMET models. Trained models export to CoreML via `export_coreml.py`. Training data: PDBbind v2020, CrossDocked2020, CASF-2016. Requirements in `Training/requirements.txt`.

## Project Configuration

- `project.yml` — XcodeGen spec. macOS 26.0 deployment target, Swift 5.9, strict concurrency
- Linker flags link `libdruse_core.a` + 25 RDKit shared libraries + `libtbb` + `libc++`
- Header search: `$(SRCROOT)/CppCore`; library search: `$(SRCROOT)/CppCore/build`, `/opt/homebrew/lib`

## Testing Notes

- The 1HSG/indinavir redocking test validates end-to-end docking accuracy (target: RMSD < 2.0 Å, currently achieving ~0.62 Å)
- When tests fail, add verbose `print()` breadcrumbs and detailed assertion messages so a single run reveals the root cause — don't re-run the suite repeatedly to investigate
- The full test suite takes ~4 minutes; skip the redocking test during iteration

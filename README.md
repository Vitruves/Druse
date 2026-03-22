# Druse

GPU-accelerated molecular docking for macOS.

Native Swift/SwiftUI application using Metal compute shaders and Apple Silicon unified memory for the full docking workflow: protein loading, preparation, ligand import, binding pocket detection, GPU-accelerated docking, ML re-ranking, and virtual screening.

## Features

- **Metal GPU docking** — Genetic algorithm with Vina-style scoring (gauss, repulsion, hydrophobic, H-bond, torsion penalty), iterated local search, all on GPU
- **Pocket detection** — Alpha-sphere probing with DBSCAN clustering and Metal-accelerated buriedness scoring
- **ML re-ranking** — CoreML inference with SE(3)-equivariant GNN (DruseScore) for pose rescoring
- **Virtual screening** — Batch workflow with shared grid reuse, ADMET filtering, and ranked export
- **Real-time 3D rendering** — Impostor sphere/cylinder rendering, Connolly surfaces, ribbon diagrams, electrostatic coloring, Z-slab clipping
- **Full format support** — PDB, mmCIF, SDF, MOL2, SMILES

## Architecture

Three-layer stack:

| Layer | Tech | Role |
|-------|------|------|
| **C++ Core** | RDKit, nanoflann, TBB | SMILES→3D, conformers, charges, fingerprints, torsion trees, KD-tree queries |
| **Metal GPU** | Metal Shading Language | Grid maps, GA docking, surface generation, impostor rendering, post-processing |
| **Swift App** | SwiftUI, CoreML | UI, docking orchestration, ML inference, state management |

## Requirements

- macOS 26.0+
- Apple Silicon (M1 or later)
- Xcode 26+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen)

### Dependencies

```bash
brew install rdkit boost eigen nanoflann tbb
```

The repository vendors gemmi under `third_parties_deps/gemmi` for offline builds. To test against a different local checkout during development, configure CMake with `-DGEMMI_SOURCE_DIR=/path/to/gemmi`.

## Build

```bash
# Build the C++ core library
cd CppCore/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
cd ../..

# Generate Xcode project
xcodegen generate

# Build
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release build
```

## Training

The `Training/` directory contains PyTorch scripts for DruseScore, pocket detector, and ADMET models. Trained models export to CoreML. Training data (PDBbind v2020, CrossDocked2020, CASF-2016) is not included — see `Training/download_data.py`.

```bash
pip install -r Training/requirements.txt
python Training/train_druse_score.py
python Training/export_coreml.py
```

## License

All rights reserved.

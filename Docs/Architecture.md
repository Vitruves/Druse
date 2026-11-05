# Architecture

How Druse is put together: a C++ cheminformatics core behind a C interface, Metal compute and render kernels, and a Swift/SwiftUI app — plus where each responsibility lives and what the build expects of you.

Druse is macOS-only and Apple Silicon only: macOS 26.0 deployment target, Xcode 26.0, Swift 5.9, `ARCHS: arm64`.

## The three layers

```
┌─ Layer 3 ─ Swift / SwiftUI ──────────────────────────────────────────────┐
│  App  Views  ViewModels  Renderer  Docking  Chemistry  ML  Console  …    │
│  AppViewModel is the central state object; SwiftUI + a Metal viewport    │
└────────────┬──────────────────────────────┬──────────────────────────────┘
             │ Swift ↔ Metal                │ Swift ↔ C (bridging header)
┌─ Layer 2 ──┴────────────────┐  ┌─ Layer 1 ┴───────────────────────────────┐
│  Metal/ — 24 .metal kernels │  │  CppCore/ — 12 .cpp behind druse_core.h  │
│  docking, scoring, neural,  │  │  wraps RDKit, gemmi, OpenMM, GFN2-xTB    │
│  preparation, surfaces,     │  │  static library libdruse_core.a          │
│  rendering                  │  │                                          │
└─────────────────────────────┘  └──────────────────────────────────────────┘
```

The two lower layers are independent of each other: the C++ core never talks to Metal, and the kernels never call into C++. Swift is the only thing that knows about both.

### Layer 1 — C++ core (`CppCore/`)

Twelve `.cpp` files and four headers, compiled by CMake into the static library `libdruse_core.a`. Everything crosses into Swift through a plain C interface — `druse_core.h` (~32 KB), plus `druse_openmm.h` and `druse_xtb.h` — so no C++ types appear in Swift and there is no Objective-C++ glue to maintain.

| File | Responsibility |
|---|---|
| `druse_core.cpp` | Top-level dispatch |
| `druse_core_io.cpp` | Parsing and serialization |
| `druse_core_analysis.cpp` | Analysis entry points |
| `druse_core_geometry.cpp` | Geometry |
| `druse_core_ionization.cpp`, `druse_core_protonation_tools.cpp`, `druse_core_variants.cpp` | Protonation/ionization pipeline, tautomer and protomer variants |
| `druse_core_pka_featurize.cpp` | pKa GNN featurization |
| `druse_gemmi.cpp` | PDB / mmCIF parsing via gemmi |
| `druse_openmm.cpp` | OpenMM energy minimization |
| `druse_xtb.cpp` | GFN2-xTB semi-empirical refinement |
| `druse_2d.cpp` | 2D depiction |
| `druse_core_internal.h` | Shared internals, not exposed to Swift |

Dependencies come from Homebrew — RDKit, Boost, Eigen3, nanoflann, TBB — except gemmi and the header-only LBFGSpp, which CMake fetches, and OpenMM, which CMake fetches and builds once at configure time if `libOpenMM.dylib` is not already in the build tree (`DRUSE_ENABLE_OPENMM`, default `ON`; a failed build degrades to no-OpenMM rather than failing the configure). GFN2-xTB is implemented in `druse_xtb.cpp` itself, not linked from an external xtb library.

The Xcode target links the resulting library plus roughly thirty RDKit components, gemmi, coordgen, TBB and OpenMM; the authoritative list is `OTHER_LDFLAGS` in `project.yml`, with `/opt/homebrew/lib` and the fetched OpenMM install dir on the rpath.

### Layer 2 — Metal kernels (`Metal/`)

Twenty-four `.metal` files and three headers. `ShaderTypes.h` holds the types shared with Swift, `DockingCommon.h` the docking-specific types and constants, and `Druse-Bridging-Header.h` is the project's bridging header. The files group by job:

- **Docking search** — `DockingGA.metal`, `DockingMonteCarlo.metal` (parallel tempering), `DockingLocalSearch.metal` (SIMD-cooperative gradient descent), `DockingGridMaps.metal`, `DockingPocketDetect.metal`, `DockingScoring.metal` (Vina and Drusina multi-term).
- **Specialized docking** — `FragmentDocking.metal` (hierarchical fragment growth with beam search), `FlexDocking.metal` + `FlexGridProxy.metal` (flexible side chains), `DiffusionDocking.metal` (reverse diffusion with attention guidance).
- **Neural scoring** — `DruseAFCompute.metal`, `DruseAFv4Compute.metal`, `PIGNet2Compute.metal`, `PKaGNNCompute.metal`. These are hand-written inference kernels, not CoreML.
- **Preparation** — `ChargeCompute.metal` (EEM/QEq matrix assembly), `FASPREnergyCompute.metal`, `PreparationMinimizeCompute.metal`, `LoopRefineCompute.metal`, `XTBCompute.metal`.
- **Interactions and surfaces** — `SurfaceCompute.metal` (field sampling + marching cubes), `InteractionCompute.metal`, `HBondScoring.metal`, `FeatureCompute.metal`.
- **Rendering** — `RenderShaders.metal`: the background gradient, atom and bond impostors, the object-ID pick pass, ribbons, interaction lines, the grid box wireframe and the surface mesh.

`Metal` is listed in `project.yml` with `buildPhase: sources`, so every `.metal` file in the directory compiles into the default library and Swift looks shaders up by function name. `MTL_FAST_MATH` is on.

### Layer 3 — Swift app

SwiftUI for the interface, a Metal renderer for the viewport, and a command interpreter that can drive the whole app with no window at all.

| Directory | Files | Contents |
|---|---|---|
| `App/` | 3 | `main.swift` (the real entry point), `DruseApp.swift`, `UpdateChecker.swift` |
| `Views/` | 46 | Every SwiftUI surface: the pipeline steps, canvas overlays, inspectors, the auxiliary windows |
| `ViewModels/` | 14 | `AppViewModel` and its extensions, plus the other state managers |
| `Renderer/` | 5 | The Metal renderer, the `CAMetalLayer` host, mesh and surface generation, the camera |
| `Docking/` | 12 | Docking engines, virtual screening, pose analysis, binding-site metrics |
| `Chemistry/` | 36 | Parsers, protein preparation, protonation, side-chain packing, loops, charges, minimizers, pockets |
| `ML/` | 5 | Inference dispatch and the native weight loaders |
| `Models/` | 8 + weights | Molecular model types, the project format, the ligand database, pharmacophore templates, and `druse-models/` |
| `Console/` | 8 + 14 commands | DruseScript parser, AST, interpreter, session, headless runner, and the command tables |
| `Services/` | 1 | `PDBService.swift` (RCSB fetch and search) |
| `Utils/` | 6 | `MathUtils`, `ActivityLog`, `AppSettings`, `FileImportHandler`, `AccessibilityID`, `CommandLineInstaller` |
| `Resources/` | — | `Assets.xcassets`, `dun2010bbdep.bin` (Dunbrack rotamers), the Ketcher editor |

## State: `AppViewModel` and friends

`AppViewModel` (`ViewModels/AppViewModel.swift`) is the one `@Observable @MainActor` object the whole app hangs off. It is split across eight extensions by domain, each a separate file:

| Extension | Covers |
|---|---|
| `+Loading` | File import, fetch, project open |
| `+Preparation` | The protein preparation pipeline |
| `+Selection` | Selection, visibility, render state, pushing to the renderer, surfaces |
| `+Docking` | Docking runs, poses, interactions |
| `+MultiPose` | Multi-pose handling |
| `+Screening` | Virtual screening |
| `+LeadOpt` | Lead optimization |
| `+Demo` | The guided demo |

It owns four grouped state objects and a few shared services:

- `workspace: WorkspaceState` — a value type holding everything about display and selection: render mode, colour scheme, per-chain colours, surface settings, clipping, selection sets, residue subsets, conformers, status and navigation requests. Being a struct, SwiftUI bindings chain through it (`$vm.workspace.showHydrogens`).
- `molecules: MoleculeManager` — the loaded protein and ligand.
- `docking: DockingCoordinator` — the engine, grid, pocket, results and run state.
- `leadOpt: LeadOptimizationState` — reference, generation parameters, ADMET filters, analogs.
- `renderer: Renderer?` — nil until the viewport exists, and nil again after a headless snapshot releases its temporary renderer.
- `ligandDB: LigandDatabase`, `log: ActivityLog.shared`, `pocketDetectorML: PocketDetectorInference`, `admetPredictor: ADMETPredictor`.

Alongside these sit `LigandPreparationService` and the docking engines themselves, which are not view models.

## Entry point: GUI or headless

There is no `@main` on `DruseApp`. `App/main.swift` is the entry point and decides before any scene, window or `NSApplication` exists:

```swift
if HeadlessRunner.isRequested(by: arguments) {
    HeadlessRunner.start(arguments: arguments)
} else {
    DruseApp.main()
}
```

`HeadlessRunner.isRequested` triggers on `--headless`, `-c`, `-e`, `--eval`, `-i`, `--interactive` or `--completions`. The headless path has no Dock icon and exits with 0 on success, 1 on a failed command, 2 on a usage or syntax error, and 130 on interrupt.

Both paths call `DruseApp.initializeComputeServices`, which applies the CPU core limit, loads the ML models in the background and installs the GPU xTB accelerator — everything that needs a GPU but not a renderer. Only the GUI path calls `initializeRenderer`.

`DruseApp.swift` declares the main window plus five auxiliary ones (Settings, Ligand Database, Help, Report Bug, Interaction Diagram, Results Database), all with `defaultLaunchBehavior(.suppressed)` so they open on demand, and the menu commands with their shortcuts. `File ▸ New Window` is deliberately removed: a second window would share the single renderer.

`DruseAppDelegate` receives files opened from Finder or `open -a Druse`; `.druse` scripts are queued on `workspace.pendingScripts` for the terminal, and wait there if they arrive before the view model exists.

## Key subsystems

**Docking** (`Docking/`). `DockingEngine.swift` (~2.4k lines) orchestrates, with `+Execution` for run control and `+Analysis` for scoring, RMSD and clustering (greedy leader clustering on RMSD). Specialized engines sit beside it: `FragmentDockingEngine`, `FlexDockingEngine` with `FlexibleDocking` and `FlexGridProxy`, and `DiffusionDockingEngine`. `VirtualScreening.swift` does batch docking, `BindingSite.swift` pocket metrics, `PoseValidator.swift` pose sanity. The engine supports GA search, parallel tempering, analytical-gradient local search, pharmacophore constraints and fragment-based growth.

**Protein preparation** (`Chemistry/`). `ProteinPreparation.swift` drives the pipeline: residue reconstruction (`ProteinResidueReconstruction`, with template stores and completeness checks), protonation (`Protonation`), H-bond network optimization (`HBondNetworkOptimizer`), FASPR side-chain packing (`FASPRSidechainPacker`, `FASPREnergyCalculator`, `FASPRMetalAccelerator`, `DunbrackRotamerLibrary` — during preparation FASPR repacks only residues whose side chains were rebuilt), loop building (`LoopBuilder`, `LoopMetalAccelerator`) and minimization (`LBFGSOptimizer`, `PreparationMinimizer`, `GFN2Refiner` for xTB QM). Parsers: `PDBParser`, `MMCIFParser` with `GemmiBridge`, `SDFParser`, `MOL2Parser`, plus `PDBWriter` and `SDFWriter`. Charges dispatch through `ChargeCalculator` over EEM (GPU), QEq and Gasteiger. `PocketDetection.swift` combines alpha spheres, DBSCAN and buriedness; `ScaffoldAnalogGenerator.swift` drives lead-opt substitutions.

**Renderer** (`Renderer/`). `Renderer.swift` is the triple-buffered, redraw-on-demand Metal renderer; `MetalView.swift` owns the `CAMetalLayer` and all mouse and key handling; `MeshGenerators.swift` builds ribbons, `SurfaceGenerator.swift` drives the surface kernels, `Camera.swift` is the arcball. See [Visualization](Visualization.md).

**Console / DruseScript** (`Console/`). `DruseScriptParser` produces the AST in `DruseScriptAST.swift`, which `DruseInterpreter` executes against `AppViewModel`. The command tables live in `Console/Commands/` (14 files, one per domain). Two front ends share the interpreter: `ConsoleSession` with `Views/ConsoleView.swift` for the in-app terminal, and `HeadlessRunner` for the CLI. See [Console](Console.md) and [DruseScript](DruseScript.md).

**ML** (`ML/`, `Models/`). `Inference.swift` unifies dispatch and holds `ADMETPredictor` and `PocketDetectorInference`; `DruseAFWeightLoader`, `PIGNet2WeightLoader` and `PKaGNNWeightLoader` load native weight blobs for the Metal inference kernels, and `PKaGNNPredictor` wraps pKa inference. Weights live in `Models/druse-models/`: `DruseAFv4.weights`, `PIGNet2.weights`, `PkaGNN.weights`, and `PocketDetector.mlpackage`, the one genuine CoreML model — it is listed separately in `project.yml` under `buildPhase: sources` so Xcode compiles it, while the rest of `druse-models/*.mlpackage` is excluded.

## Data flow

```
PDB / mmCIF / SDF / MOL2 / SMILES
  → parsing                    Chemistry parsers, or the C++ core via gemmi/RDKit
  → protein preparation        reconstruction → protonation → H-bond optimization
                               → FASPR repacking → optional xTB / OpenMM minimization
  → pocket detection           alpha-sphere + DBSCAN + buriedness, and the CoreML detector
  → GPU docking                GA / fragment / flex / diffusion search,
                               scored by Vina, Drusina or DruseAF v4
  → ML rescoring               DruseAF v4, PIGNet2
  → pose clustering            greedy leader clustering on RMSD
  → ranked results → export
```

Two conventions worth knowing before you touch scoring:

- **ML pocket.** DruseAF and PIGNet2 both score against a heavy-atom pocket of the 256 protein atoms nearest the pocket centre within 10 Å — `mlPocketAtoms` in `DockingEngine.swift`. This mirrors how the models were trained; changing it invalidates the weights.
- **PIGNet2 is a reranker, not a search potential.** `PIGNet2Compute.metal` scores and reranks finished poses. Do not wire it into the per-generation GA loop: it is applied after Vina-guided pose generation and local optimization, to rerank the final best population and the explicit cluster representatives.

## Build and codegen conventions

The Xcode project is generated, not committed, and the C++ core must exist before Xcode can link.

```bash
# 1. C++ core first — the Xcode target links libdruse_core.a
cd CppCore/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && cd ../..

# 2. Generate Druse.xcodeproj from project.yml (XcodeGen 2.35+)
xcodegen generate

# 3. Build
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release build
```

**Re-run `xcodegen generate` after adding or removing source files**, and after adding a top-level source directory. Sources are declared per-directory in `project.yml`, so a new file inside an existing directory is picked up by regeneration alone — but a *new top-level directory* must also be added to the `Druse` target's `sources` list first, or nothing in it compiles.

Other conventions the project depends on:

- **Bridging header** — `SWIFT_OBJC_BRIDGING_HEADER: Metal/Druse-Bridging-Header.h`. Anything Swift needs from C or from the shader headers is reached through it; `HEADER_SEARCH_PATHS` adds `CppCore`.
- **Strict concurrency** — `SWIFT_STRICT_CONCURRENCY: complete`. New types that cross task boundaries need real isolation or a justified `@unchecked Sendable` with a comment explaining why it holds, as `SurfaceGenerator` and `SurfaceResult` do.
- **Version** — the `VERSION` file at the repository root is the source of truth (e.g. `0.1.1`). `project.yml`'s `MARKETING_VERSION` (without any `-beta` suffix) and the README badge and download links mirror it. `./update_readme_version.sh` syncs them; `./update_readme_version.sh --version X.Y.Z-beta` bumps and syncs.
- **Other build settings** — `ACCELERATE_NEW_LAPACK=1`, `MTL_FAST_MATH: YES`, Debug at `-Onone` with Metal debug info, Release at `-O` without.
- **Packaging** — `scripts/bundle_dmg.sh [Druse.app]` makes a portable DMG by copying the Homebrew dylibs into `Contents/Frameworks` and rewriting the load paths. `Tools/render_icon.swift` renders the app icon.

### Targets

`project.yml` declares three. `Druse` is the app. `DruseTests` compiles `Tests/` plus the Swift files from `Benchmark/` (its Python, data, manifests, results and reports are excluded) and depends on `Druse` — which is why the benchmark harness runs as an XCTest and therefore opens a window. `DruseUITests` compiles `UITests/`.

**Do not run `xcodebuild test` without asking first**: the test scheme launches a GUI window. See [Benchmark](Benchmark.md) for the same caveat on benchmark runs.

### Training

`Training/` holds the PyTorch and torch-geometric scripts that produce the model weights: `trainDruseAFv4.py`, `trainDruseAFv5.py`, `train_druse_score.py`, `train_pka.py`, `train_pocket_detector.py`, `trainPocketML.py`, `train_admet.py` and `precompute_pocket_features.py`, with `export_coreml.py` for CoreML packages and `export_pignet2_weights.py` for the native Metal weight blobs. Dependencies are in `Training/requirements.txt`. It is entirely optional and not part of the app build.

## See also

- [Installation](Installation.md) — prerequisites, building from source, packaging
- [Visualization](Visualization.md) — the renderer and viewport in detail
- [Console](Console.md) and [DruseScript](DruseScript.md) — the interpreter and the command language
- [Docking](Docking.md), [Scoring](Scoring.md), [Preparation](Preparation.md), [BindingSite](BindingSite.md) — what the subsystems above actually do
- [Benchmark](Benchmark.md) — the CASF harness built on `DruseTests`

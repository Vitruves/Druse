# Benchmark

The CASF-2016 / PDBbind harness: data layout, manifest generation, running the configurable benchmark CLI, reading the results, and the Drusina calibration and AutoDock Vina comparison tooling.

> **A benchmark run opens a GUI window.** `Benchmark/run_benchmark.py` drives `xcodebuild test` against the `DruseTests` target, and the test host is the Druse app itself, so each scoring method launches a real window for the duration of the run. Do not start a run on a machine you need for anything else, and do not start one from a session that cannot show a window. Full CASF sweeps take hours.

## What it measures

Two standard CASF metrics, computed by the Swift runner and printed at the end of each method:

- **Docking power** — RMSD of the top-ranked pose against the crystal ligand, reported as the fraction below 2.0 Å, 3.0 Å and 5.0 Å, plus mean and median.
- **Scoring power** — Pearson correlation between the predicted score and the experimental pKd. When GFN2 energies are present for more than ten complexes, a second correlation is reported for those.

Each scoring method runs independently and writes its own result file, so methods can be compared on identical poses-per-complex budgets.

## Data layout

Both datasets are expected under `Benchmark/data/`, which is gitignored — nothing is bundled with the repository, and both downloads need a free PDBbind registration.

```
Benchmark/data/
  CASF-2016/                     # CASF-2016 package, extracted as-is
    coreset/
    power_scoring/CoreSet.dat    # the core-set index with pKd values
    power_docking/
    power_ranking/
    power_screening/
    decoys_docking/
    decoys_screening/
  refined-set/                   # PDBbind refined set, one directory per entry
    1a30/
      1a30_protein.pdb
      1a30_pocket.pdb
      1a30_ligand.sdf
    ...
```

`prepare.py` reads the pKd values from `CoreSet.dat`: it walks `Benchmark/data/CASF-2016/` looking for any file whose name contains `CoreSet` and ends in `.dat`, and takes the PDB ID from column 1 and the pKd from column 4. If no such file is found it falls back to `Benchmark/data/pdbbind_labels.csv`, using the rows whose `split` column is `refined`.

Structures come from `refined-set/`, not from the CASF package, so a core-set entry that is missing from the local refined set is skipped. Expect fewer complexes in the manifest than in `coreset/` for this reason.

`Benchmark/calibrate_drusina_pdbbind.py` additionally reads `Benchmark/data/refined-set/index/INDEX_refined_data.2020`.

## Python environment

```bash
pip install -r Benchmark/requirements.txt
```

That pulls numpy, pandas, scipy, scikit-learn, matplotlib, rdkit, requests and tqdm. The manifest step needs numpy and rdkit; the calibration scripts need scipy as well.

## Step 1 — generate the manifest

```bash
python Benchmark/prepare.py
python Benchmark/prepare.py --verify     # just report how many complexes the manifest holds
```

For every core-set entry present in `refined-set/`, `prepare.py` parses the ligand SDF with RDKit and records the canonical SMILES, the heavy-atom crystal coordinates, the heavy-atom count and the experimental pKd. It also computes a substructure match between the SDF and SMILES atom orderings and stores both the mapping and a re-ordered copy of the coordinates, so the runner can compute RMSD correctly whether the ligand came from the SDF (redocking) or was rebuilt from SMILES (full pipeline).

Output: `Benchmark/manifests/casf_manifest.json`, with one entry per complex:

```json
{
  "pdb_id": "1a30",
  "protein_pdb": "Benchmark/data/refined-set/1a30/1a30_protein.pdb",
  "pocket_pdb": "Benchmark/data/refined-set/1a30/1a30_pocket.pdb",
  "ligand_sdf": "Benchmark/data/refined-set/1a30/1a30_ligand.sdf",
  "smiles": "...",
  "crystal_positions": [[x, y, z], ...],
  "crystal_positions_smiles": [[x, y, z], ...],
  "smiles_atom_map": [...],
  "heavy_atom_count": 26,
  "pKd": 5.90
}
```

The script prints how many complexes it wrote, how many it skipped (missing files or an RDKit parse failure), and the pKd and heavy-atom ranges.

## Step 2 — run the benchmark

```bash
# 10-complex smoke test with Vina
python Benchmark/run_benchmark.py --casf --scoring vina --quick 10

# One full method
python Benchmark/run_benchmark.py --casf --scoring drusina

# Several methods in sequence, fast parameters
python Benchmark/run_benchmark.py --casf --scoring vina,drusina --preset fast

# Everything
python Benchmark/run_benchmark.py --casf --scoring all

# All available flags
python Benchmark/run_benchmark.py --help
```

Either `--casf` or `--manifest PATH` is required; without one the script exits with a usage error.

### Main options

**`--casf`** — run the CASF-2016 manifest at `Benchmark/manifests/casf_manifest.json`. `--manifest PATH` substitutes a different manifest (this is how the curated panel runs), and `--benchmark-name` overrides the label stored in the result JSON.

**`--scoring`** — one or more methods, comma-separated. Each maps to one XCTest method:

| `--scoring` | Label | Test method |
|---|---|---|
| `vina` | Vina | `testCASF_Vina` |
| `drusina` | Drusina | `testCASF_Drusina` |
| `druseaf` | Druse Affinity | `testCASF_DruseAF` |
| `pignet2` | PIGNet2 | `testCASF_PIGNet2` |
| `gfn2` | Drusina+GFN2 | `testCASF_DrusinaGFN2` |
| `all` | all five, in sequence | `testCASF_All` |

**`--quick N`** — stop after the first N complexes. The output filename gets a `_quickN` suffix so quick runs are never confused with full ones.

**`--preset`** — one of four parameter bundles. A preset is applied as a default: any of these flags you also pass explicitly on the command line wins, and the script prints which ones overrode the preset.

| Preset | Population | Generations | Runs | Grid | Local search | Other |
|---|---|---|---|---|---|---|
| `fast` | 50 | 30 | 1 | 0.5 Å | every 3 gen × 15 steps | — |
| `standard` | 200 | 200 | 3 | 0.375 Å | every 1 gen × 30 steps | — |
| `thorough` | 300 | 300 | 5 | 0.375 Å | every 1 gen × 50 steps | flexible residues |
| `exhaustive` | 300 | 300 | 10 | 0.375 Å | every 1 gen × 50 steps | flexible residues, GFN2 optimization (tight, top 50) |

With no preset the defaults are population 200, generations 200, 1 run, 0.375 Å grid, local search every generation × 30 steps, analytical gradients, ligand flexibility on, rigid receptor, Gasteiger charges, full protein preparation, the `redocking` torsion preset, MMFF94 strain penalty on (threshold 6.0 kcal/mol, weight 0.5), 40 % exploration generations, and reranking of the top 12 cluster representatives with 4 refinement seeds each.

### Other option groups

`--help` is authoritative; the groups are Dataset, Scoring Method, Genetic Algorithm, Local Search, Flexibility, Charge Computation, Ligand Torsion Search, Post-Docking Refinement, GFN2-xTB Geometry Optimization, Exploration Phase, Explicit Reranking, Output and Presets. The ones that change what is being measured rather than how hard it searches:

- `--pipeline-mode redock|full` (default `redock`) — classic redocking from the crystal conformer, or the full end-to-end Druse path: ligand rebuilt from SMILES, pocket detected automatically, app-style post-processing.
- `--pocket-mode ligand-guided|geometric|ml|hybrid` (default `hybrid`) — how the pocket is chosen. In full mode, `hybrid` ranks ML and geometric candidates with sanity checks.
- `--protein-prep-mode full|cleanup|raw` (default `full`) — the full Druse preparation, cleanup only (altloc, non-standard residues, waters, chains), or parsed PDB atoms with no preparation at all. `raw` is the closest analogue to a reference PDB → PDBQT conversion and is what you want for an apples-to-apples Vina comparison.
- `--charge-method gasteiger|eem|qeq|xtb` (default `gasteiger`).
- `--torsion-preset balanced|redocking|exploratory|manual` (default `redocking`, which preserves the input conformer; `balanced` is the app default).
- `--gfn2-rescoring` — post-docking GFN2-xTB single-point rescoring of the best pose (D4 dispersion + ALPB solvation).
- `--gfn2-opt` and friends — GFN2-xTB geometry optimization of the top poses, with `--gfn2-opt-level`, `--gfn2-solvation`, `--gfn2-top-poses`, `--gfn2-blend-weight`, `--gfn2-max-steps`.
- `--flex-residues` — flexible receptor side chains, auto-selected from the pocket lining.
- `--debug` — add diagnostics to the result JSON: pocket centre and method, distance to the crystal centroid, pocket volume and buriedness, search-box size, heavy-atom counts, conformer RMSD, strain energy, per-pose RMSDs and energies, and the per-term Drusina decomposition. The calibration and decomposition scripts need this.
- `--druse-logs [FILE]` — stream the app's own log lines in real time, to the terminal or to a file.
- `--verbose` — show the full `xcodebuild` output instead of the filtered stream.

### How a run actually executes

1. The resolved configuration is written to `Benchmark/.bench_config.json`. Environment variables do not reliably survive `xcodebuild`'s test host, so this shared file is the transport; `BenchmarkRunner` reads it at launch.
2. The project is built once: `xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release build`. A build failure aborts the run and tells you to build manually to see the errors.
3. For each scoring method, with `DRUSE_RUN_BENCHMARKS=1` set: `xcodebuild test -project Druse.xcodeproj -scheme Druse -only-testing:DruseTests/BenchmarkRunner/<test>`. **This is the step that opens a window.**
4. `Benchmark/.bench_config.json` is deleted when the run finishes.

`BenchmarkRunner.setUpWithError` skips every test unless `.bench_config.json` exists or `DRUSE_RUN_BENCHMARKS=1` is set, so the benchmark never runs by accident during an ordinary test pass. It also skips when there is no Metal GPU, or when the manifest is missing — in which case it tells you to run `python Benchmark/prepare.py`.

While a method runs, the script filters the `xcodebuild` noise and prints one live line per complex with running statistics:

```
   7/10  1bcu  E=  -8.41  RMSD=  1.62A   6.1s  <2A: 5/7 (71%)  avg 7.4s
```

RMSD is colour-coded (green below 2 Å, amber below 5 Å, red above), and failures print as `FAILED: <message>`.

## Step 3 — results

Results land in `Benchmark/results/` (override with `--output-dir`), named:

```
[<prefix>]casf_<method>_<VERSION>_<YYYYmmdd_HHMMSS>[_quickN].json
```

`VERSION` is read from the repository's `VERSION` file, so a result file always records which build produced it; `--output-prefix` prepends an experiment tag. The runner rewrites the file every ten complexes and again at the end, so an interrupted or crashed run leaves usable partial results.

The schema is defined by `Benchmark/BenchmarkTypes.swift`:

```json
{
  "benchmark": "casf-2016",
  "version": "0.1.1",
  "scoringMethod": "drusina",
  "timestamp": "...",
  "config": { "populationSize": 200, "generations": 200, "gridSpacing": 0.375, "numRuns": 1,
              "proteinPrepMode": "full", "torsionSearchPreset": "redocking" },
  "entries": [
    { "pdb_id": "1a30", "best_energy": -8.41, "best_display_score": -8.41, "best_rmsd": 1.62,
      "experimental_pKd": 5.90, "num_poses": 20, "docking_time_ms": 6140.0, "success": true }
  ]
}
```

Reading the fields:

- `best_energy` is the internal score used for ranking — kcal/mol for Vina, Drusina and PIGNet2, and `-pKd × confidence` for DruseAF. `best_display_score` is the score as the app would show it, which for DruseAF is the predicted pKi. Correlate `best_energy` against `experimental_pKd` for scoring power; a *negative* r is the expected sign for energy-like scores, since a more negative energy should mean tighter binding.
- `best_rmsd` is the top-ranked pose against the crystal ligand. `null` means no reference was available for that complex.
- `gfn2_energy`, `gfn2_dispersion` and `gfn2_solvation` appear when GFN2 rescoring or optimization was enabled.
- With `--debug`, `best_top_pose_rmsd` and `best_top_pose_rank` separate a sampling failure (no pose near the crystal at all) from a ranking failure (a good pose was found but not ranked first), and `pocket_distance` separates both from a pocket-detection failure. `conformer_rmsd` measures how well the SMILES-derived conformer reproduces the crystal geometry and is a property of the 3D embedding, not of the docking.

The end-of-method summary printed to the terminal is produced by `printSummary` in `BenchmarkRunner.swift` and is not written to the JSON; recompute it from the entries if you need it later.

`--analyze` is accepted but looks for `Benchmark/analyze.py`, which is not present in the tree; it prints that the script was not found and exits cleanly. Analyse the result JSON with your own script, or with the panel and calibration tools below.

## The curated panel

For day-to-day work, `Benchmark/run_real_life_panel.py` runs a small fixed subset end-to-end instead of the full CASF sweep. It always forces `--pipeline-mode full` and `--pocket-mode hybrid`, so it exercises what a user actually does: ligand rebuilt from SMILES, pocket found automatically, then docking plus post-processing.

```bash
python Benchmark/run_real_life_panel.py --list
python Benchmark/run_real_life_panel.py --scoring vina,drusina --preset fast
python Benchmark/run_real_life_panel.py --groups metal,charged --scoring drusina --preset fast --debug
python Benchmark/run_real_life_panel.py --skip-run --debug --results Benchmark/results/<file>.json
```

Panel definitions live in `Benchmark/panels/`. `real_life_panel_v2.json` is the default: 48 cases grouped as `generic` (20), `aromatic` (6), `metal` (6), `charged` (6), `large_flexible` (6) and `halogen_bond` (4). `real_life_panel_v1.json` is the earlier 16-case panel, four each of `rigid_fragments`, `metal_coordination`, `charged_flexible` and `bulky_hydrophobic`. Each case names a PDB ID, its group, a title and the reason it is in the panel; `--panel PATH` selects a definition, `--groups` filters by group, `--limit N` caps the case count.

It writes a subset manifest to `Benchmark/manifests/<panel-id>_manifest.json`, shells out to `run_benchmark.py` with `--manifest`, then writes `Benchmark/reports/<panel-id>_report.md` (grouped pass rates) and, with `--debug`, `Benchmark/reports/<panel-id>_debug.md` (pocket displacement, failure classification, improvement hints). Its own flags — `--scoring`, `--preset`, `--rescoring gfn2`, `--flex-residues`, `--no-strain`, `--runs`, `--charge-method` — are forwarded, and anything after `--` is passed straight through to `run_benchmark.py`.

`--skip-run` regenerates reports from existing result files without docking anything, which does **not** open a window.

## Drusina calibration

Drusina adds fourteen linear terms to the Vina base: `pi_pi`, `pi_cation`, `salt_bridge`, `amide_pi`, `halogen_bond`, `chalcogen_bond`, `metal_coord`, `coulomb`, `ch_pi`, `torsion_strain`, `cooperativity`, `hbond_dir`, `desolv_polar`, `desolv_hydrophobic`. Their weights are fitted, not learned: the functional forms are fixed and only the fourteen coefficients are optimized, exactly as AutoDock Vina's five weights are. Two scripts do this, from opposite directions.

**From benchmark results.** `calibrate_drusina.py` reads the `drusina_decomposition` field that `--debug` writes and refits the weights to maximize Pearson r against the experimental pKd, within per-term bounds that keep each term's sign physically sensible.

```bash
python Benchmark/run_real_life_panel.py --scoring drusina --preset standard --debug
python Benchmark/calibrate_drusina.py Benchmark/results/<result>.json
python Benchmark/calibrate_drusina.py --cv 5 Benchmark/results/casf_drusina_*.json
```

Options: `--cv N` for N-fold cross-validation, `--reg` for L2 regularization (default 0.01), `--seed` (default 42).

**From crystal poses, at scale.** `calibrate_drusina_pdbbind.py` reimplements the Drusina geometric terms in numpy and scores the crystal poses of the whole PDBbind refined set directly, so no docking is needed and the fit sees thousands of complexes instead of a few hundred.

```bash
python Benchmark/calibrate_drusina_pdbbind.py                      # full refined set
python Benchmark/calibrate_drusina_pdbbind.py --max-complexes 500  # quick check
python Benchmark/calibrate_drusina_pdbbind.py --cv 5
```

Options: `--max-complexes` (0 = all), `--cv`, `--reg`, `--seed`, `--pocket-radius` (default 12.0 Å), `--skip-vina` to leave out the Vina base scoring for speed. It needs RDKit and exits if it is missing.

**Term diagnostics.** `analyze_decomposition.py` reports per-term statistics from a `--debug` Drusina result, which is how you find a term that is noisy or firing spuriously before refitting anything.

```bash
python Benchmark/analyze_decomposition.py Benchmark/results/<file>.json
python Benchmark/analyze_decomposition.py Benchmark/results/<file>.json --csv decomp.csv
```

## AutoDock Vina comparison

`compare_vina_reference.py` runs the real AutoDock Vina CLI on the same complexes as a Druse result, so the comparison is on identical inputs rather than against published numbers.

```bash
python Benchmark/compare_vina_reference.py --quick 30
python Benchmark/compare_vina_reference.py --druse-result Benchmark/results/<file>.json --match-druse
```

Options: `--quick N`, `--druse-result PATH` (the Druse JSON to compare against), `--match-druse` (restrict to the complexes that file contains), `--exhaustiveness` (default 32), `--workers` (default 4).

It expects two external binaries, at paths hard-coded in the script: the Vina executable at `temp/AutoDock-Vina/build/mac/release/vina`, and Open Babel at `/opt/homebrew/bin/obabel` for the PDB → PDBQT and SDF → PDBQT conversions. pKd values come from `Benchmark/data/refined-set/index/INDEX_refined_data.2020`. Results are written to `Benchmark/results/autodock_vina_reference_<N>cpx.json`.

This script does not build or launch Druse, so it opens no window.

## Directory map

```
Benchmark/
  prepare.py                   # datasets → JSON manifest
  run_benchmark.py             # the configurable benchmark CLI (drives xcodebuild test)
  run_real_life_panel.py       # curated end-to-end panel + grouped reports
  calibrate_drusina.py         # refit Drusina weights from benchmark decompositions
  calibrate_drusina_pdbbind.py # refit on PDBbind crystal poses, numpy reimplementation
  analyze_decomposition.py     # per-term Drusina statistics
  compare_vina_reference.py    # AutoDock Vina 1.2.x head-to-head
  BenchmarkRunner.swift        # the XCTest runner that does the docking
  BenchmarkTypes.swift         # Codable manifest and result structs
  requirements.txt
  panels/                      # curated panel definitions
  data/                        # CASF-2016 and refined-set (gitignored)
  manifests/                   # generated manifests (gitignored)
  results/                     # result JSON (gitignored)
  reports/                     # markdown reports (gitignored)
```

`Benchmark/README.md` predates this document and has a few stale paths; where the two disagree, the code and this file are correct.

## See also

- [Scoring](Scoring.md) — what Vina, Drusina, DruseAF v4, PIGNet2 and GFN2 rescoring actually compute
- [Docking](Docking.md) — the GA, local search, presets and pose budgets the benchmark flags expose
- [Preparation](Preparation.md) — what `--protein-prep-mode full` does, and why `raw` changes the numbers
- [BindingSite](BindingSite.md) — the pocket detection that `--pocket-mode` selects between
- [Architecture](Architecture.md) — where `BenchmarkRunner` sits and how `DruseTests` is assembled

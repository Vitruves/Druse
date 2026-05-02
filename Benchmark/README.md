# Druse CASF-2016 Benchmark

Standard docking benchmark against the CASF-2016 dataset (285 protein-ligand complexes from PDBbind).

## Metrics

- **Docking Power**: RMSD of top-1 pose vs crystal structure (success = RMSD < 2.0 A)
- **Scoring Power**: Pearson correlation between predicted score and experimental pKd

Three scoring methods are compared independently:
- **Vina** — grid-based Vina scoring (baseline)
- **Drusina** — Vina + extended interactions (pi-pi, pi-cation, halogen bonds, metal coordination)
- **Druse AF** — ML-driven scoring (DruseScore pKd model)

## Prerequisites

### 1. PDBbind v2020 Refined Set

Download from [PDBbind](http://www.pdbbind.org.cn/) (free registration required):

```
PDBbind_v2020_refined.tar.gz
```

Extract to `Training/data/refined-set/`. You should have:
```
Training/data/refined-set/1a1e/1a1e_protein.pdb
Training/data/refined-set/1a1e/1a1e_ligand.sdf
...
```

Then generate the labels CSV:
```bash
python Training/download_data.py --labels
```

### 2. CASF-2016 Benchmark Package

Download from [PDBbind CASF page](http://www.pdbbind.org.cn/casf.php) (free registration):

```
CASF-2016.tar.gz   (1.46 GB)
```

Extract to `Benchmark/data/`:
```bash
mkdir -p Benchmark/data
tar xzf CASF-2016.tar.gz -C Benchmark/data/
```

You should have `Benchmark/data/CASF-2016/` with `power_scoring/CoreSet.dat`.

### 3. Python Environment

```bash
pip install numpy rdkit   # or: uv pip install numpy rdkit
```

### 4. Build Druse

```bash
cd CppCore/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && cd ../..
xcodegen generate
```

## Running the Benchmark

### Fast Curated Panel

For day-to-day docking work, use the curated real-life panel instead of the full CASF sweep.
It runs a fixed subset of 16 CASF complexes grouped by practical failure mode:
rigid fragments, metal coordination, charged/flexible ligands, and bulky hydrophobics.
Unlike the classic redocking benchmark, the panel defaults to the full Druse path:
ligand regeneration from SMILES, automatic pocket detection (ranked ML + geometric candidates),
then docking plus app-style post-processing.

```bash
# Quick two-method smoke benchmark
python Benchmark/run_real_life_panel.py --scoring vina,drusina --preset fast

# Focus on metal and charged cases only
python Benchmark/run_real_life_panel.py \
  --groups metal_coordination,charged_flexible \
  --scoring drusina --preset fast

# Inspect the panel without running it
python Benchmark/run_real_life_panel.py --list
```

Outputs:
- `Benchmark/manifests/real_life_panel_v1_manifest.json` — generated subset manifest
- `Benchmark/reports/real_life_panel_v1_report.md` — grouped quick report

### Step 1: Prepare Manifest

Parses the CASF-2016 index + PDBbind ligand SDF files to extract SMILES, crystal positions, and pKd values.

```bash
python Benchmark/prepare.py
```

Output: `Benchmark/manifests/casf_manifest.json` (225 complexes).

### Step 2: Run Docking

Each scoring method runs independently. Pick one or run all three:

```bash
# Single scoring method (~55 min each)
DRUSE_RUN_BENCHMARKS=1 xcodebuild test -project Druse.xcodeproj -scheme Druse \
  -only-testing:DruseTests/BenchmarkRunner/testCASF_Vina

DRUSE_RUN_BENCHMARKS=1 xcodebuild test -project Druse.xcodeproj -scheme Druse \
  -only-testing:DruseTests/BenchmarkRunner/testCASF_Drusina

DRUSE_RUN_BENCHMARKS=1 xcodebuild test -project Druse.xcodeproj -scheme Druse \
  -only-testing:DruseTests/BenchmarkRunner/testCASF_DruseAF

# All three sequentially (~2.5 hours)
DRUSE_RUN_BENCHMARKS=1 xcodebuild test -project Druse.xcodeproj -scheme Druse \
  -only-testing:DruseTests/BenchmarkRunner/testCASF_All
```

Results are written incrementally (survives crashes) to:
- `Benchmark/results/casf_vina.json`
- `Benchmark/results/casf_drusina.json`
- `Benchmark/results/casf_druseaf.json`

### Step 3: Analyze Results

```bash
python Benchmark/analyze.py --report
```

Produces:
- Console comparison table (scoring power, docking power at multiple thresholds)
- `Benchmark/reports/benchmark_report.md` — markdown report
- `Benchmark/reports/casf_comparison.png` — scatter plots and RMSD histograms

## Results (March 2026, Apple M3)

| Method | RMSD < 2.0 A | Pearson r | ms/complex |
|--------|-------------|-----------|------------|
| Vina | 57.8% | -0.522 | 14,762 |
| Drusina | 55.1% | -0.517 | 15,249 |
| Druse AF | 57.3% | -0.524 | 144,891 |
| AutoDock Vina (ref) | ~55-65% | ~-0.56 | ~45,000 (CPU) |

Config: pop=200, gen=200, runs=1, grid=0.375 A, full protein preparation (protonation pH 7.4, H-bond network optimization, Gasteiger charges).

## File Structure

```
Benchmark/
  README.md              — This file
  prepare.py             — Parse datasets → JSON manifest
  analyze.py             — Compute metrics, generate report
  run_real_life_panel.py — Curated quick benchmark grouped by failure mode
  requirements.txt       — Python dependencies
  BenchmarkRunner.swift  — XCTest headless docking runner
  BenchmarkTypes.swift   — Codable JSON manifest/result structs
  panels/                — Curated panel definitions
  data/                  — CASF-2016 extracted here (gitignored)
  manifests/             — Generated JSON manifests (gitignored)
  results/               — Docking output JSON (gitignored)
  reports/               — Markdown report + plots (gitignored)
```

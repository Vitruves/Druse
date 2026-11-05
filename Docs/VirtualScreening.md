# Virtual Screening

Batch-docking a compound library against one pocket: shared grids, parallel 3D generation, a GPU pre-score, a GA-docked shortlist, optional refinement and a ranked export.

Contents: [Two batch modes](#two-batch-modes) · [The screening pipeline](#the-screening-pipeline) · [Pre-filtering and pre-scoring](#pre-filtering-and-pre-scoring) · [Shortlist and batched GA](#shortlist-and-batched-ga) · [Refinement and ADMET](#refinement-and-admet) · [Scoring function](#scoring-function) · [Progress and cancellation](#progress-and-cancellation) · [Practical scale](#practical-scale) · [Filters and export](#filters-and-export) · [Batch docking](#batch-docking) · [Console commands](#console-commands)

Prerequisites: a prepared receptor and a selected pocket (see [Preparation.md](Preparation.md) and [BindingSite.md](BindingSite.md)). See [Docking.md](Docking.md) for single-ligand docking and [Analysis.md](Analysis.md) for what to do with the results.

## Two batch modes

| | `screen <library>` | `dock batch` |
|---|---|---|
| Input | A `.smi`/`.smiles`/`.txt` or `.sdf`/`.mol` library file | Entries of the Ligand Database |
| Preparation | RDKit 3D embedding inside the pipeline, in parallel batches | Whatever the Ligand Database holds (`ligand prep`) |
| Search | GPU pre-score, then a batched GA on a shortlist | The full single-ligand docking pipeline, one ligand after another |
| Scoring | Vina affinity grids | Any session scoring function (`scoring=vina\|drusina\|druseaf\|pignet2`) |
| Output | Ranked `screeningHits` with descriptors | One `DockingResult` list per ligand, ranked by best pose |
| Scale | Up to `max` molecules (default 100 000) | Tens to hundreds of prepared entries |

`screen` is the throughput path; `dock batch` is the accurate path, and the one to use when the scoring function, search method or refinement settings matter. Lead-optimization analogs land in the Ligand Database automatically, so `dock batch` is also the natural way to score them (see [LeadOptimization.md](LeadOptimization.md)).

## The screening pipeline

`VirtualScreeningPipeline.screen()` runs five phases against one receptor and one pocket. Pharmacophore constraint buffers are built once from the enabled constraints and shared by every ligand, and the grid maps are computed once for the selected pocket at `gridSpacing` (default **0.5 Å**, deliberately coarser than the 0.375 Å `dock` default) and reused by the pre-score and the GA.

1. **Prepare** — the library is truncated to `max` molecules and processed in batches of 256. Each batch goes to `RDKitBridge.batchProcess` off the main actor, which embeds, minimises and charges the molecules in parallel through TBB in the C++ core, and descriptors are computed per molecule. Progress 0 → 0.3.
2. **Pre-score** — a batched Metal kernel pair scores random rigid poses for every prepared ligand against the shared grids. Progress 0.3 → 0.4.
3. **Shortlist and GA** — the best-prescoring fraction is docked flexibly with a batched GA. Progress 0.3 → 0.8.
4. **Optional post-processing** — MMFF94 strain penalty, OpenMM pocket minimisation, GFN2-xTB single points.
5. **ADMET annotation** — predictions for the top hits when the models are installed.

Library parsing: a SMILES file is read as one molecule per line, `SMILES [name]` (whitespace- or tab-separated), skipping blank lines and lines starting with `#`; unnamed lines become `Mol_N`. An SDF uses each molecule's recorded SMILES, or one derived from its connection table, and unnamed records become `Mol_N`.

## Pre-filtering and pre-scoring

Two filters remove work before any GPU time is spent:

- **Parse failures.** A molecule RDKit cannot embed is dropped.
- **Rotatable-bond cutoff.** `max-rotatable` (default **15**) drops very flexible molecules, whose conformational space a reduced-budget GA cannot cover. The activity log reports per-batch counts of parse failures and rotatable-bond rejections.

The pre-score (`BatchedDockingMetalAccelerator`) processes 128 ligands per batch. For each ligand it initialises `prescreen-poses` random rigid placements (default **72**, minimum 16) with `initializeBatchScreenPoses`, scores them all with `scoreBatchRigidPoses` against the steric, hydrophobic and H-bond grids, and keeps the single best pose with its term decomposition and clash penalty. Ligands with no GPU atoms, or more than **128** heavy atoms (the per-ligand GPU limit), are skipped and left without a pre-score; a faulted command buffer yields a sentinel energy of 1 000 000 rather than a garbage ranking.

## Shortlist and batched GA

Ligands are sorted by pre-score energy and the shortlist is sized as follows: if there are no more than `shortlist-min` ligands (default **128**) all of them go forward; otherwise the target is `max(shortlist-min, ceil(count × shortlist))` with `shortlist` defaulting to **0.35**, capped by `shortlist-max` (default **4096**).

Before the GA runs, the grid maps are rebuilt with the Vina atom types actually present in the shortlist — the first grid pass was built without ligand types, so its affinity maps would miss every type lookup and return zero intermolecular energy.

`BatchedGAAccelerator.runBatchedGA` then docks **32 ligands per GPU batch**, flattening per-ligand atoms, torsion edges, moving-atom indices and 1-4+ pair lists into contiguous buffers. Each generation dispatches three batched kernels — `mcPerturbBatched`, `scorePosesBatched`, `metropolisAcceptBatched` — over all ligands × population at once. Defaults: `population` **50** per ligand (minimum 20), `generations` **30** (minimum 10), 8 local-search steps, mutation rate 0.3, crossover rate 0.6, translation step 0.8 Å, rotation step 0.3 rad, torsion step 0.5 rad, Metropolis temperature 1.2 kcal/mol. The initial population is random: uniform translation inside the search box, uniform random orientation (Shoemake), input-conformer torsions. Command buffers are only waited on every tenth generation; any faulted buffer in the queue invalidates that batch.

If the batched GA cannot be created (no Vina affinity grid, no shortlist types), the pipeline falls back to docking each shortlisted ligand sequentially with a reduced single-ligand configuration (one run, flexibility on, no live viewport updates, local search every 2–5 generations).

A shortlisted ligand whose GA returns nothing keeps its rigid pre-score pose and energy, so it still appears in the ranking rather than vanishing.

## Refinement and ADMET

All optional, applied after the hits are sorted by energy:

| Step | Option | Default | Effect |
|---|---|---|---|
| MMFF94 strain | `strain=on\|off` | off | Computes docked − relaxed MMFF energy per hit and, when strain exceeds 6 kcal/mol, adds `0.5 × (strain − 6)` to the score. Off by default to keep Vina screening scorer-pure. |
| OpenMM | `openmm-top=N` | 25 | Pocket minimisation of the top N hits. The force-field interaction energy is recorded as an **annotation only** — it is on a different scale and must not feed the ranking. Skipped with a log note when OpenMM is not compiled in. |
| GFN2-xTB | `gfn2-top=N` | 0 (off) | Single-point GFN2-xTB energy of the top N hits with water solvation. |
| ADMET | `admet=on\|off` | on | Predictions for at most the first **1000** hits, and only when the CoreML ADMET models are installed. |

ADMET prediction always yields the rule-based descriptors, whether or not the ML models are present: **Lipinski** (MW ≤ 500, cLogP ≤ 5, HBD ≤ 5, HBA ≤ 10), **Veber** (rotatable bonds ≤ 10, TPSA ≤ 140) and a 0–1 drug-likeness composite that scores MW ≤ 500, −1 ≤ cLogP ≤ 5, HBD ≤ 5, HBA ≤ 10, rotatable bonds ≤ 10 and 20 ≤ TPSA ≤ 140 at 0.2/0.2/0.15/0.15/0.15/0.15. The ML models add logP, logD7.4, aqueous solubility, CYP2D6 and CYP3A4 inhibition, hERG liability, BBB permeability and metabolic stability.

Lipinski and Veber are **not** generation-time filters in screening: every prepared molecule is docked, and the rules are applied afterwards as a filter on the hit list (see below).

## Scoring function

`screen` always ranks on the Vina-typed affinity grids used by the batched kernels; the session's `scoring=` setting does not change it. To screen with Drusina, DruseAFv4 or Vina + PIGNet2, import the library into the Ligand Database and use `dock batch scoring=…`, which runs the full per-ligand pipeline. See [Scoring.md](Scoring.md).

## Progress and cancellation

The pipeline publishes a `ScreeningState` (`preparing`, `docking`, `complete`, `failed`) and a 0–1 progress value, polled every 500 ms into the Results step and into the console's progress line (`Screening: preparing ligands (37%)`).

Cancellation uses an atomic flag held in heap memory so the same token can be shared between Swift tasks and the C++ TBB `parallel_for` that prepares the ligands. `screen stop` (or the GUI cancel) sets it; the flag is checked between preparation batches, between pre-score batches, between GA batches, and during strain and OpenMM post-processing, so a cancelled run stops within a batch rather than at the end.

## Practical scale

Hard limits and shapes that the code imposes:

- `max` molecules per run, default **100 000**; everything beyond the cut is never read.
- **128 heavy atoms** per ligand is the GPU limit, in both the screening pre-score and `dock batch`; larger molecules are skipped.
- Flexible docking is capped at `shortlist-max` = **4096** ligands, with at least `shortlist-min` = **128**.
- ADMET annotation covers at most the first **1000** hits; OpenMM the top 25 by default.
- SDF export writes the top 100 hits by default; the CSV always lists every hit.

In other words, a hundred-thousand-molecule library is prepared and rigid-pre-scored in full, and roughly a third of it (bounded by 4096) is docked flexibly. For libraries larger than that, split them or lower `max`.

## Filters and export

The Results step's **Screening Hits** card filters the ranked list live, without re-running anything: *Lipinski compliant only*, *Energy ≤* (slider −20…5, initialised to the median composite score of the current hits) and *ML score ≥* (0…15). The card header shows "passing / all". Each row carries the composite score, MW and cLogP, a Lipinski tag, and an **Optimize** button that makes the hit the lead-optimization reference.

Export:

| File | Contents |
|---|---|
| `.csv` | Every hit: rank, name, SMILES, energy, strain, pre-score energy, OpenMM energy, ML score, composite score, MW, cLogP, HBD, HBA, TPSA, rotatable bonds, Lipinski, drug-likeness |
| `.sdf` | The top `top=` hits (default 100) with their docked heavy-atom geometry, and properties Name, SMILES, Energy, CompositeScore, plus PrescreenEnergy, OpenMMEnergy and ML_Score when present |

The GUI's "Export Hits (CSV + SDF)" writes the CSV and, next to it, a `.sdf` with the top 100. The `ML_Score` column exists for an ML re-ranking stage that the current pipeline does not populate, so it reads `N/A`.

## Batch docking

`dock batch` docks Ligand Database entries one after another into the selected pocket with the session docking settings, then ranks the ligands by their best pose.

- `ligands=all` (default) takes every entry with 3D coordinates, `ligands=prepared` only prepared entries, and a comma-separated list selects entries by name. A name that has been enumerated into chemical forms selects all of its dockable forms, and the raw import is then excluded so a molecule is not docked twice.
- Entries with no heavy atoms, NaN coordinates, or more than 128 heavy atoms are skipped with a warning.
- The receptor is prepared once (cached per protein and pH) and the pharmacophore buffers are bound once for the whole batch.
- With auto mode on, the budget is auto-tuned per ligand from its heavy-atom and rotatable-bond counts together with the pocket's volume and buriedness.
- Any inline docking option works on the command (`preset=`, `scoring=`, `search=`, `population=`, `generations=`, `runs=`), and persists for later runs.
- When the batch finishes, the ranking is sorted by best score, the winner's poses become the current results, and `results batch` / `export batch` report them. The Results Database's correlation sheet works on batch results.

## Console commands

Screen a library:

```
screen enamine_subset.smi
screen library.sdf max=5000 shortlist=0.2 openmm-top=50
screen hits.smi population=80 generations=60 gfn2-top=10 admet=off
screen enamine_subset.smi max-rotatable=10 prescreen-poses=128 top=50
```

Watch, stop and review:

```
screen hits top=100
screen stop
```

Export:

```
export hits out/hits.csv
export hits out/hits.sdf top=200
```

Batch-dock the Ligand Database instead:

```
dock batch ligands=prepared preset=fast
dock batch ligands=aspirin,ibuprofen scoring=drusina
results batch top=50
export batch out/batch_ranking.csv
```

A complete headless screening script:

```
load ~/targets/1add_protein.pdb
prep waters=remove
pocket detect
screen ~/libraries/fragments.smi max=20000 shortlist=0.25 openmm-top=50 top=25
export hits ~/Desktop/fragment_hits.csv
export hits ~/Desktop/fragment_hits.sdf top=250
```

See [Console.md](Console.md) for headless runs and exit codes, and [DruseScript.md](DruseScript.md) for the language reference.

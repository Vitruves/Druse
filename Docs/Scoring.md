# Scoring

Druse ships four scoring functions — the empirical Vina baseline, the extended empirical Drusina function, the DruseAF v4 neural affinity model and the physics-informed PIGNet2 GNN — all evaluated on the GPU, plus post-docking explicit-atom rescoring, MMFF94 strain filtering and optional GFN2-xTB refinement.

Contents: [Choosing a function](#choosing-a-function) · [Vina](#vina) · [Reported terms](#reported-terms) · [Drusina](#drusina) · [DruseAF v4](#druseaf-v4) · [PIGNet2](#pignet2) · [ML pocket definition](#the-shared-ml-pocket) · [Model weights](#model-weights) · [Post-docking refinements](#post-docking-refinements) · [Rescoring existing poses](#rescoring-existing-poses)

## Choosing a function

`dock set scoring=…` (or `scoring=` inline on `dock` / `dock batch`) selects the function used both during the search and for the final ranking.

| `scoring=` | Name | Unit | Nature | Use it when |
|---|---|---|---|---|
| `vina` | Vina | kcal/mol | Empirical, 5 terms, grid-interpolated | Redocking, baseline comparisons, anything that must be comparable to published AutoDock Vina numbers |
| `drusina` | Drusina | kcal/mol | Vina plus 14 extended interaction terms | Series where π systems, halogens, metals or salt bridges decide the ranking |
| `druseaf` | Druse Affinity | pKd | Learned pairwise geometric network | Affinity ranking across chemically diverse ligands |
| `pignet2` | Vina + PIGNet2 | kcal/mol | Vina-guided search, physics-informed GNN reranking | Reranking a Vina-generated pose set with a learned physics decomposition |

Only Drusina accounts for every interaction type the viewer can draw; the live interaction display is gated to the types the active function actually scores — Vina shows H-bonds and hydrophobic contacts, PIGNet2 adds metal coordination, Drusina shows all ten, and DruseAF shows none (it is a learned scorer with no per-interaction terms). See [Visualization.md](Visualization.md).

`scoring=pignet2` is refused at `dock set` time if the PIGNet2 weights are not bundled with the build. If a model's kernels or weights fail to load at run time, the run falls back to Vina and says so in the activity log rather than silently producing different numbers.

## Vina

The baseline is the AutoDock Vina functional form, evaluated against the precomputed affinity maps described in [Docking.md](Docking.md). Pairwise terms use the Vina XS atom typing and a surface distance `d = r − (R_i + R_j)`, cut off at r = 8 Å:

| Term | Form | Weight |
|---|---|---|
| gauss1 | `exp(−(d/0.5)²)` | −0.035579 |
| gauss2 | `exp(−((d−3)/2)²)` | −0.005156 |
| repulsion | `d²` for d < 0, else 0 | +0.840245 |
| hydrophobic | linear ramp, 1 below d = 0.5 Å, 0 above d = 1.5 Å; both atoms hydrophobic | −0.035069 |
| H-bond | linear ramp, 1 below d = −0.7 Å, 0 above d = 0; donor–acceptor pairs only | −0.587439 |
| rotatable-bond normalization | `1 / (1 + 0.05846 · N_rot)` | 0.05846 |

The total for a pose is

```
E = (E_inter + ΔE_intra) / (1 + 0.05846 · N_rot) + 10 · boundary_penalty + constraint_penalty
```

`E_inter` is the sum of the affinity-map values at the transformed ligand atom positions. `ΔE_intra` is the ligand's own non-bonded energy minus a reference, and sits **inside** the normalization, matching Vina's `conf_independent(inter + intra − ref)`. Atoms outside the grid box contribute a quadratic boundary penalty scaled by 10 kcal/mol instead of a grid value. Pharmacophore constraint penalties are added outside the normalization.

### Intramolecular energy is re-referenced to the best pose

This is the one place where Druse deliberately post-processes the raw GPU score, and it matters when comparing absolute numbers.

The GPU kernels subtract the **input conformer's** intramolecular energy, which makes every absolute score depend on the input geometry — a strained input conformer inflates every score in the run. AutoDock Vina instead reports each pose against the **best pose's** intramolecular energy (Vina 1.1/1.2 `model::eval_adjusted`: `E_i = conf_independent(inter_i + intra_i − intra_best)`), so the top pose scores its intermolecular energy alone.

After output selection and the explicit rerank, Druse re-references to the best pose:

```
shift_energy  = (intra_input − intra_best) / (1 + 0.05846 · N_rot)
shift_torsion = (intra_input − intra_best) · (1/(1 + 0.05846 · N_rot) − 1)
```

The energy shift is a single per-run constant, so ranking, the output energy window and clustering are all unchanged; only the absolute values move. The matching correction is applied to the reported torsion term so the five reported terms keep summing to the shifted total. The activity log records the input-conformer intra energy, the best-pose intra energy and the resulting shift.

Re-referencing applies only when it is valid — `scoring=vina` with a rigid receptor. It is skipped for Drusina, DruseAF and PIGNet2, and skipped when flexible side chains are active, because those energies are no longer of the form `(inter + intra − inputIntra)·norm + penalties`. The live "best so far" readout during a run uses the same shift, so the progress line matches the final ranking.

## Reported terms

`results` prints one row per pose and `results show <n>` prints everything known about a single pose. For the energy-based functions, five terms are reported and **they sum exactly to the total**:

| Column (`results`) | UI label | Field | Meaning |
|---|---|---|---|
| Steric | vdW | `stericEnergy` | gauss1 + gauss2 + repulsion |
| Hydroph | Hydro | `hydrophobicEnergy` | Hydrophobic contact term |
| HBond | H-bond | `hbondEnergy` | H-bond term |
| Torsion | Torsion | `torsionPenalty` | The rotatable-bond normalization term, `normalized(inter + Δintra) − inter − Δintra` |
| Other | `Intra+` | `intraAndOtherEnergy` | Everything else: `energy − steric − hydrophobic − hbond − torsion` |

`Intra+` is the leftover term, defined so the breakdown closes. It collects the ligand intramolecular energy relative to the reference conformer, the out-of-box boundary penalty, any pharmacophore constraint penalty, and any Drusina correction. `results show <n>` labels it "Intra/other" and spells out the contents.

The three decomposition fields are not what the search minimizes. During the search, `stericEnergy` holds the whole grid-interpolated intermolecular energy and the hydrophobic and H-bond fields are zero. After the run, `decomposeVinaEnergy` re-evaluates the final population against explicit protein atoms within 8 Å (one SIMD group per pose, lanes splitting the protein-atom loop) and writes the real three-way split, leaving the total energy untouched so rankings cannot shift.

For DruseAF, `results` prints pKd and a confidence percentage instead of the term breakdown.

```
results top=20
results sort=cluster
results show 1
results clusters
```

## Drusina

Drusina is the Vina baseline plus a correction computed from explicit receptor features (`computeDrusinaCorrections` in `DockingScoring.metal`). It is designed for GPU search rather than as a small post-hoc correction: linear ramp functions instead of steep Gaussians give a smoother landscape, salt bridges are scored once per charged-group pair rather than per atom, and optimal distances come from CSD/PDB statistics. The design follows Verdonk 2003 (GoldScore block/ramp functions), Bissantz 2010 (interaction geometry statistics) and Donald 2011 (group-based salt bridges).

The correction is gated by pose quality: it is multiplied by `1 / (1 + exp(0.5·(E_vina − 2)))`, which is ≈ 1 for any favourable Vina pose, 0.5 at +2 kcal/mol, and → 0 for severe clashes, so a clashing pose cannot be rescued by interaction bonuses. The total correction is floored at `minCorrection` = −5.5 kcal/mol.

| Term | Geometry | Weight |
|---|---|---|
| π–π stacking | Ligand ring ↔ protein ring, 3.2–5.8 Å. Face-to-face when \|n·n\| > 0.8 (optimum 3.5 Å); edge-to-face/T-shaped when \|n·n\| < 0.4 (optimum 4.8 Å, 0.6× weight) | −0.20 |
| π–cation | Ring centroid ↔ cation, within 5.5 Å and within 60° of the ring normal, optimum 3.8 Å. Both directions (protein cation ↔ ligand ring, ligand cation ↔ protein ring) | −0.05 |
| Salt bridge | Ligand charged atom ↔ protein charged **group** centroid, optimum 2.8 Å (CSD median N–O 2.79 Å), cut off at 4.0 Å, weighted by the group's burial factor. Formal charges at full weight; large Gasteiger partial charges (\|q\| > 0.35) inferred at half weight | −0.20 |
| Amide–π | Protein backbone amide ↔ ligand aromatic ring, 3.0–5.2 Å. Parallel (\|n·n\| > 0.8) at optimum 3.6 Å; tilted (> 0.6) at 4.0 Å with 0.4× weight. Harder 2013 geometry | −0.15 |
| Halogen bond | C–X···N/O with σ-hole directionality: C–X···A angle > 140°, distance 2.5–4.2 Å. Element-specific optima from Scholfield 2013 — Cl 3.27 Å, Br 3.24 Å, I 3.17 Å; F is skipped as too weak | −0.40 |
| Chalcogen bond | C–S···N/O σ-hole, C–S–A angle > 140° (Beno 2015, Pascoe 2017), distance 2.8–4.2 Å, optimum 3.3 Å. Both directions, including protein Met/Cys sulfur → ligand acceptor | −0.10 |
| Metal coordination | Protein metal ↔ ligand N/O/S, cut off at 4.5 Å, optimum 2.4 Å. Capped at one interaction per metal (Verdonk 2003) | −0.91 |
| Screened Coulomb | `Σ q_ligand · Φ(r)` from the precomputed electrostatic grid, ε = 4r | +0.015 |
| CH–π | Ligand aliphatic C_H ↔ protein aromatic ring, 3.5–5.0 Å, within ~55° of the ring normal, weighted by cos²; optimum 4.0 Å | −0.26 |
| Torsion strain | Amide planarity: `k · dev²` where dev is the deviation of the dihedral from 0 or π | +1.0 |
| H-bond directionality | Bonus only, on close contacts (1.5–3.2 Å): donor X–D···A angle towards 180° and acceptor D···A–Y angle towards ~120°, distance factor peaking at 2.7 Å. No penalty for bad geometry (Korb 2009, Verdonk 2003) | −0.32 |
| Polar desolvation | Buried, H-bond-unsatisfied ligand polar atoms (≥ 12 protein neighbours within 5 Å); HYDE-style (Schneider 2013) | 0.0 — zeroed by calibration |
| Hydrophobic desolvation | Solvent-exposed ligand hydrophobic carbons (< 3 protein neighbours within 4.5 Å); Böhm 1994, ChemScore (Eldridge 1997) | 0.0 — zeroed by calibration |
| Cooperativity | Count-based bonus for multiple simultaneous interactions | 0.0 |

Weights were calibrated on the PDBbind v2020 refined set (4641 complexes, 5-fold cross-validation, r = −0.349). Protein features beyond 10 Å of the ligand centroid are gated out before the pairwise loops.

The applied correction is reported per pose as `drusinaCorrection` ("Drusina corr." in `results show`), and is included in the `Intra+` / `Other` column. `dock config` has no Drusina weight settings — the calibrated weights are fixed in the engine. Per-term decomposition for all fourteen terms is available through the benchmark harness; see [Benchmark.md](Benchmark.md).

Under Drusina, every pose in the top `rerank-clusters` also gets a GFN2-xTB D4 dispersion single-point annotation.

```
dock set scoring=drusina
dock preset=standard
results show 1
```

## DruseAF v4

DruseAF v4 is a Pairwise Geometric Network (PGN) that predicts pKd, implemented natively in Metal (`DruseAFv4Compute.metal`) rather than through CoreML. It is ~134 K parameters / ~536 KB of weights, and it consumes only interatomic distances and per-atom features — no absolute coordinates.

Architecture, matching `Training/trainDruseAFv4.py` exactly:

- Protein branch: MLP 20 → 128, then 3 rounds of distance-attention message passing with 16 RBF bins, 8 Å message cutoff, RBF γ = 2.0, each followed by layer normalization.
- Ligand branch: MLP 20 → 128.
- Scoring: Hadamard pairwise interaction in a 64-dimensional pair space, over protein–ligand pairs within an 8 Å cross cutoff, with 24 cross RBF bins.
- Heads: `pKd = Σ pair_energy · pair_scale + affinity_head(complex) + pair_bias`, and a sigmoid confidence head.

The pose score written back to the GA population is

```
energy = −(pKd × confidence) + out_of_box_penalty
```

so the search minimizes it like any other energy, while `results` displays the raw `pKd` and the confidence as a percentage (`results show` spells out "ranking energy = −pKd × confidence"). Because the reported quantity is an affinity rather than an energy, the 3 kcal/mol output window is converted to pKd units using ΔG = −RT·ln10·pKd at 298.15 K.

Kernel split: a setup pass runs once per target (encode, three message-passing rounds, pair preparation, ~2 ms); per generation only the ligand position transform and the pairwise score run, at roughly 0.02 ms per pose. The kernel handles up to 64 ligand heavy atoms. The model was trained on PDBbind v2020, with CASF-2016 complexes held out for external validation.

No Vina local search runs under DruseAF scoring, because the affinity maps the Vina gradients come from are not the active scoring function. DruseAF is also the scorer that supplies the attention gradients for [diffusion-guided docking](Docking.md#diffusion-guided-docking-beta).

```
dock set scoring=druseaf
dock
results top=10
```

## PIGNet2

PIGNet2 is a physics-informed graph neural network (Moon et al. 2022) implemented in Metal (`PIGNet2Compute.metal`): feature dimension 47, hidden dimension 128, three GatedGAT intra-molecular convolutions, three InteractionNet layers with GRU updates, and a physics energy head.

**PIGNet2 is not used as the per-generation search potential.** It is a pose scorer, not a geometry-stable docking potential. With `scoring=pignet2`:

1. The search runs Vina-guided: Vina affinity maps drive pose generation, the Metropolis moves and the analytical-gradient local optimization.
2. At the end of each run, the best population is rescored with PIGNet2 (one threadgroup per pose) — in place of the usual explicit Vina term decomposition.
3. The explicit cluster representatives selected after output selection are then also rescored with PIGNet2 during the rerank step, so the reported ranking is PIGNet2's.

The UI label for this mode is "Vina + PIGNet2", and the engine logs it as "PIGNet2 rerank".

The physics head decomposes the score into four terms over protein–ligand pairs, with the van der Waals ε and Morse width predicted per pair from the learned embeddings (sigmoid-bounded) and `R` the sum of van der Waals radii:

```
score = (E_vdw + E_hbond + E_metal + E_hydrophobic) / (1 + rotor_coeff² · N_rot) + clash_penalty
```

- `E_vdw` — Morse potential, skipped for pairs involving a metal.
- `E_hbond` — linear potential on donor–acceptor pairs (either direction), coefficient −`hbond_coeff²`.
- `E_metal` — linear potential on metal ↔ acceptor pairs, coefficient −`metal_coeff²`.
- `E_hydrophobic` — linear potential on hydrophobic–hydrophobic pairs, coefficient −`hydro_coeff²`.

The rotor penalty is applied to each term before summing. Graph edges for the InteractionNet layers use a 0.5–5 Å range, while the physics head itself has no upper distance cutoff. The decomposition is reported in the pose fields: `stericEnergy` = E_vdw, `hydrophobicEnergy` = E_hydrophobic, `hbondEnergy` = E_hbond + E_metal, and the rotor penalty divisor in the torsion field.

```
dock set scoring=pignet2
dock preset=standard
results top=10
```

## The shared ML pocket

DruseAF and PIGNet2 both score against the same receptor pocket definition, built by `mlPocketAtoms` in `DockingEngine.swift` the way both models were trained:

- **Hydrogens removed.** `trainDruseAFv4.py`'s `parse_pdb_atoms` skips hydrogens, and PIGNet2 featurizes `Chem.RemoveAllHs` molecules.
- **Strictly within 10 Å** of the pocket centre.
- **At most 256 atoms** — when more remain, the 256 closest to the pocket centre are kept (as in `trainDruseAFv4.py`'s `preprocess_complex`).

256 is the kernel limit for both models (`PIG_MAX_PROT`); the ligand side is capped at 64 heavy atoms (`PIG_MAX_LIG`). If the heavy-atom pocket comes back empty, setup fails, the run falls back to Vina and the activity log says so.

The pocket centre is the selected pocket's centre, so moving or refitting the docking box changes what the ML models see. See [BindingSite.md](BindingSite.md).

## Model weights

Native Metal weight blobs live in `Models/druse-models/` and are copied into the app bundle:

| File | Model | Loader |
|---|---|---|
| `DruseAFv4.weights` | DruseAF v4 PGN (DRAF format v2, 56 tensors) | `ML/DruseAFWeightLoader.swift` |
| `PIGNet2.weights` | PIGNet2 | `ML/PIGNet2WeightLoader.swift` |
| `PkaGNN.weights` | pKa GNN (protonation, not docking) | `ML/PKaGNNWeightLoader.swift` |
| `PocketDetector.mlpackage` | CoreML pocket detector | `Models/PocketDetectorModel.swift` |

Each loader produces one packed float32 `MTLBuffer` plus an offset table the shader indexes, and validates the per-tensor float counts against the layout its kernel expects. The DruseAF loader prefers `DruseAFv4.weights` and falls back to a v3 cross-attention `DruseAF.weights` if only that is bundled. Training and export scripts are in `Training/`; see [Benchmark.md](Benchmark.md).

## Post-docking refinements

These run after the search, in this order, on the already-clustered pose set.

### Explicit-atom rescoring with basin hopping

The affinity maps are a trilinear interpolation; the final ranking should not depend on interpolation error. After output selection, the top `rerank-clusters` cluster representatives (default 12, 0 disables) are rescored against **explicit** receptor atoms within 8 Å, mirroring Vina's late non-cache rescoring.

The rerank is a small basin hop rather than a plain rescore: each representative is expanded into `rerank-variants` seeds (default 4). The first is the pose itself; the others are jittered by up to 0.75 Å in translation, 0.18 rad in rotation and 0.30 rad per torsion. Every seed is then locally optimized for `rerank-steps` gradient steps (default 20) before scoring, and the best variant becomes the cluster's new representative. Clusters are renumbered by the reranked representative energies.

Drusina uses a deliberately conservative profile — one variant, no second-pass local optimization — because the Drusina score is not aligned with the Vina-style local optimizer, so aggressive refinement tends to walk poses off the Drusina-preferred basin. PIGNet2 representatives are optimized with the Vina gradient and then scored with PIGNet2.

```
dock set rerank-clusters=16 rerank-variants=6 rerank-steps=30
dock config rerank
```

### MMFF94 strain filtering

A pose can score well while holding the ligand in an internally strained conformation. The strain filter computes `strain = MMFF94(docked geometry) − MMFF94(relaxed free ligand)` through RDKit, for the best 50 poses by default, and adds `strain-weight × (strain − strain-threshold)` to the score of any pose above the threshold. Defaults: `strain-threshold` 6.0 kcal/mol, `strain-weight` 0.5. Poses are re-sorted afterwards, and a pose that already carries a strain energy is not penalized twice.

The automatic pass runs after every docking run when `strain=on` (the default) **and** the scoring method is not `vina` — the Vina baseline is left unpenalized so it stays comparable to published numbers. It needs the ligand's SMILES.

`strain` runs it on demand on the current results, overriding the session values; `weight=0` reports the strain without changing the ranking. The value appears in the `Strain` column of `results` and in `results show`.

```
strain
strain top=20 threshold=4
strain weight=0
```

### OpenMM pocket minimization

`refine openmm` minimizes the pocket with OpenMM for the top poses (default 3) and stores the resulting interaction energy as an annotation. **The ranking is unchanged** — it is a diagnostic, reported as "OpenMM" in `results show`. The receptor view shows the best pose's relaxed pocket afterwards.

```
refine openmm
refine openmm top=10
```

### GFN2-xTB refinement

GFN2-xTB semi-empirical optimization of the docked ligand geometry, with D4 dispersion and implicit solvation. Unlike OpenMM refinement, this one **does** re-rank. It runs automatically after every docking run when `gfn2=on`, and on demand with `refine gfn2`.

| Setting | Default | Meaning |
|---|---|---|
| `gfn2-top` | 20 | Top poses optimized (1–500) |
| `gfn2-solvation` | `alpb` | `none` (gas phase), `alpb` (water, ALPB), `gbsa` (water, GBSA). ALPB implicit solvation follows Ehlert et al. 2021 |
| `gfn2-level` | `normal` | Convergence level: `crude`, `normal`, `tight` |
| `gfn2-steps` | 0 | Maximum optimization steps; 0 = automatic from the level and atom count |
| `gfn2-blend` | 0.3 | Weight of the *relative* GFN2 energy added to the ranking score (0–1) |
| `gfn2-update` | on | Replace pose coordinates with the optimized geometry |
| `gfn2-restraint` | 0.005 | Harmonic positional restraint in Hartree/Å²; 0.005 allows ~0.3 Å of local relaxation per atom while preventing drift. 0 = none |
| `gfn2-max-rmsd` | 2.0 Å | Reject optimized geometries that moved further; the docked coordinates are kept |

Blending uses the energy *relative to the best GFN2 pose*, `energy += gfn2-blend × (E_gfn2 − min E_gfn2)`, and the results are re-sorted. `gfn2-blend=0` makes it a pure annotation. Protein atoms are frozen (ligand-only optimization). Each pose records its total GFN2 energy, D4 dispersion component, solvation component, convergence flag and step count, all shown by `results show`.

`refine gfn2` takes the session `gfn2-*` values and lets you override any of them for one call.

```
dock set gfn2=on gfn2-top=10 gfn2-solvation=alpb
refine gfn2
refine gfn2 top=5 solvation=gbsa level=tight
refine gfn2 blend=0 update=off
```

## Rescoring existing poses

`rescore` scores the current poses with a neural model **without moving them**, and makes that model the active scoring method so `results` reports in its units.

```
rescore druseaf
rescore pignet2
```

This is the cheap way to compare functions on one pose set: dock once with Vina, then rescore with DruseAF or PIGNet2 and compare the rankings. Note that rescoring is not the same as docking with that function — a search guided by Vina explores a different pose space than one guided by Drusina or DruseAF.

---

See also: [Docking.md](Docking.md) for the search engines, grid maps, pharmacophore constraints and every `dock` setting · [Analysis.md](Analysis.md) for output selection, clustering, RMSD, pose validity and interaction analysis · [BindingSite.md](BindingSite.md) for the pocket that defines both the grid box and the ML pocket · [Preparation.md](Preparation.md) for protonation and charges, which the scoring functions read · [VirtualScreening.md](VirtualScreening.md) · [Benchmark.md](Benchmark.md) for CASF-2016 and Drusina calibration · [Console.md](Console.md) and [DruseScript.md](DruseScript.md)

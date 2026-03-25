# Druse ML Scoring Models

Druse uses two complementary ML scoring models, each serving a distinct role in the docking pipeline.

## DruseScorePKi — Primary Scoring Function -> alias Druse AF now

**CoreML model:** `DruseScorePKi.mlpackage`
**Training script:** `train_druse_pKi.py`
**Export script:** `export_coreml_pKi.py`

**Purpose:** Replaces the Vina-style empirical energy score as the authoritative ranking for docked poses. This is the score the user sees.

**Primary output:** `docking_score = pKd × pose_confidence`
- `pKd` — predicted binding affinity (-log10 Kd), range ~2–12
- `pose_confidence` — how close the pose is to native (0 = garbage, 1 = crystal-like)
- `docking_score` — single ranking value combining both (higher = better)

**Training approach:**
- Trained on PDBbind refined (5,316 complexes) × 8 structured RMSD perturbations = ~42,000 samples
- Each perturbation level (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0 A RMSD) gets a continuous confidence label via Gaussian decay
- The model learns to distinguish good poses from bad ones AND predict binding affinity simultaneously

**Pipeline position:**
```
Metal GA (Vina grid scoring) → 50 diverse poses → DruseScorePKi ranks them → user sees pKd + confidence
```
The Vina grid score is fast (O(1) per atom via trilinear interpolation) and serves as the search heuristic for the genetic algorithm. DruseScorePKi is the evaluator — slower but accurate.

---

## DruseScore — Post-Docking Re-Ranker

**CoreML model:** `DruseScore.mlpackage`
**Training script:** `train_druse_score.py`
**Export script:** `export_coreml.py`

**Purpose:** Secondary opinion on pose ranking. Trained only on crystal poses, so its pKd predictions are most reliable for near-native poses. Useful as a consensus signal alongside the primary scorer.

**Primary output:** `pKd` (binding affinity prediction)
- Also outputs `pose_confidence` (binary: crystal vs decoy) and `interaction_map`

**Training approach:**
- Trained on PDBbind refined crystal structures with random decoy generation
- Binary pose classification (crystal = 1, random decoy = 0)
- No intermediate RMSD levels — less effective at distinguishing "decent" from "mediocre" poses

**When to use both together:**
1. DruseScorePKi ranks poses by `docking_score` (primary)
2. DruseScore provides a second pKd estimate on the top poses
3. If both models agree on the top pose, confidence is high
4. If they disagree, flag the result for manual inspection

This consensus approach is more robust than either model alone, especially for edge cases (unusual binding modes, flexible loops, allosteric sites).

---

## Summary

| | DruseScorePKi | DruseScore |
|---|---|---|
| **Role** | Primary scorer | Re-ranker / consensus |
| **CoreML file** | `DruseScorePKi.mlpackage` | `DruseScore.mlpackage` |
| **Primary output** | `docking_score = pKd × confidence` | `pKd` |
| **Pose quality handling** | Continuous (RMSD 0–15 A) | Binary (crystal vs random) |
| **Training data** | 42K samples (8 perturbations/complex) | 5K crystal poses + random decoys |
| **Best for** | Ranking GA output | Validating top poses |

## Other Models

| Model | CoreML file | Purpose |
|---|---|---|
| **PocketDetector** | `PocketDetector.mlpackage` | Surface point pocket classification |
| **ADMET** | `ADMET_*.mlpackage` | Drug-likeness property prediction |

#!/usr/bin/env python3
"""
Calibrate Drusina scoring weights using benchmark decomposition data.

Reads benchmark JSON with per-term Drusina decompositions and experimental pKd,
then optimizes weights to maximize scoring power (Pearson r) and docking power.

Usage:
  # First run benchmark with --debug to collect decomposition data:
  python Benchmark/run_real_life_panel.py --scoring drusina --preset standard --debug

  # Then calibrate on that data:
  python Benchmark/calibrate_drusina.py Benchmark/results/<result>.json

  # Or on CASF-2016 full set:
  python Benchmark/calibrate_drusina.py Benchmark/results/casf_drusina_*.json

  # Cross-validate:
  python Benchmark/calibrate_drusina.py --cv 5 Benchmark/results/<result>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr


# Drusina terms in order — must match DrusinaParams weights
TERMS = [
    "pi_pi", "pi_cation", "salt_bridge", "amide_pi",
    "halogen_bond", "chalcogen_bond", "metal_coord", "coulomb",
    "ch_pi", "torsion_strain", "cooperativity",
    "hbond_dir", "desolv_polar", "desolv_hydrophobic",
]

# Current default weights (same order as TERMS)
DEFAULT_WEIGHTS = {
    "pi_pi": -0.20,
    "pi_cation": -0.50,
    "salt_bridge": -0.20,
    "amide_pi": -0.15,
    "halogen_bond": -0.40,
    "chalcogen_bond": -0.10,
    "metal_coord": -0.95,
    "coulomb": 0.015,
    "ch_pi": -0.04,
    "torsion_strain": 1.0,
    "cooperativity": 0.0,
    "hbond_dir": -0.30,
    "desolv_polar": 0.12,
    "desolv_hydrophobic": 0.06,
}

# Bounds for optimization: (min, max) for each weight
WEIGHT_BOUNDS = {
    "pi_pi": (-1.0, 0.0),
    "pi_cation": (-2.0, 0.0),
    "salt_bridge": (-1.0, 0.0),
    "amide_pi": (-0.5, 0.0),
    "halogen_bond": (-1.5, 0.0),
    "chalcogen_bond": (-0.5, 0.0),
    "metal_coord": (-3.0, 0.0),
    "coulomb": (-0.2, 0.2),
    "ch_pi": (-0.3, 0.0),
    "torsion_strain": (0.0, 5.0),
    "cooperativity": (-0.5, 0.5),
    "hbond_dir": (-1.0, 0.0),
    "desolv_polar": (0.0, 0.5),
    "desolv_hydrophobic": (0.0, 0.3),
}


def load_data(json_paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load decomposition data from benchmark JSON files.

    Returns:
        raw_terms: (N, T) array of raw (unweighted) term contributions
        vina_base: (N,) array of Vina base energy (total - drusina_total)
        pkd: (N,) array of experimental pKd
        pdb_ids: list of PDB IDs
    """
    raw_rows, vina_bases, pkds, pdb_ids = [], [], [], []

    for path in json_paths:
        data = json.load(open(path))
        for entry in data.get("entries", []):
            decomp = entry.get("drusina_decomposition")
            pkd = entry.get("experimental_pKd")
            energy = entry.get("best_energy")
            if decomp is None or pkd is None or energy is None:
                continue

            # Extract raw term values (decomp stores WEIGHTED values)
            # Divide by current weight to get raw contribution
            raw = []
            for term in TERMS:
                weighted_val = decomp.get(term, 0.0)
                w = DEFAULT_WEIGHTS[term]
                if abs(w) < 1e-8:
                    raw.append(weighted_val)  # weight was 0, raw = weighted
                else:
                    raw.append(weighted_val / w)

            # Vina base = total energy - drusina correction
            drusina_total = decomp.get("total", 0.0)
            vina_base = energy - drusina_total

            raw_rows.append(raw)
            vina_bases.append(vina_base)
            pkds.append(pkd)
            pdb_ids.append(entry["pdb_id"])

    return (np.array(raw_rows), np.array(vina_bases),
            np.array(pkds), pdb_ids)


def scoring_power(weights: np.ndarray, raw_terms: np.ndarray,
                  vina_base: np.ndarray, pkd: np.ndarray) -> float:
    """Compute negative Pearson r (for minimization)."""
    drusina = raw_terms @ weights
    total_energy = vina_base + drusina
    r, _ = pearsonr(total_energy, pkd)
    return r  # We want r to be negative (lower energy = higher pKd), so minimize r


def docking_power_proxy(weights: np.ndarray, raw_terms: np.ndarray,
                        vina_base: np.ndarray, rmsds: np.ndarray) -> float:
    """Proxy for docking power: correlation between energy and RMSD."""
    drusina = raw_terms @ weights
    total_energy = vina_base + drusina
    # Lower energy should correlate with lower RMSD
    r, _ = pearsonr(total_energy, rmsds)
    return -r  # Minimize negative = maximize positive correlation


def objective(weights: np.ndarray, raw_terms: np.ndarray,
              vina_base: np.ndarray, pkd: np.ndarray,
              reg_lambda: float = 0.01) -> float:
    """Combined objective: scoring power + L2 regularization."""
    r = scoring_power(weights, raw_terms, vina_base, pkd)
    # L2 regularization to prevent extreme weights
    reg = reg_lambda * np.sum(weights ** 2)
    return r + reg


def cross_validate(raw_terms: np.ndarray, vina_base: np.ndarray,
                   pkd: np.ndarray, n_folds: int = 5,
                   reg_lambda: float = 0.01) -> tuple[float, float]:
    """K-fold cross-validation of weight optimization."""
    n = len(pkd)
    indices = np.random.permutation(n)
    fold_size = n // n_folds
    train_rs, test_rs = [], []

    for fold in range(n_folds):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        w0 = np.array([DEFAULT_WEIGHTS[t] for t in TERMS])
        bounds = [WEIGHT_BOUNDS[t] for t in TERMS]

        result = minimize(
            objective, w0,
            args=(raw_terms[train_idx], vina_base[train_idx],
                  pkd[train_idx], reg_lambda),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500}
        )

        train_r = scoring_power(result.x, raw_terms[train_idx],
                                vina_base[train_idx], pkd[train_idx])
        test_r = scoring_power(result.x, raw_terms[test_idx],
                               vina_base[test_idx], pkd[test_idx])
        train_rs.append(train_r)
        test_rs.append(test_r)

    return np.mean(train_rs), np.mean(test_rs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("json_files", nargs="+", help="Benchmark result JSON files")
    parser.add_argument("--cv", type=int, default=0,
                        help="Cross-validation folds (0 = no CV)")
    parser.add_argument("--reg", type=float, default=0.01,
                        help="L2 regularization strength")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw_terms, vina_base, pkd, pdb_ids = load_data(args.json_files)
    n = len(pkd)
    print(f"Loaded {n} complexes with decomposition data")
    if n < 10:
        print("ERROR: Need at least 10 complexes for calibration", file=sys.stderr)
        sys.exit(1)

    # Current performance
    w_current = np.array([DEFAULT_WEIGHTS[t] for t in TERMS])
    r_current = scoring_power(w_current, raw_terms, vina_base, pkd)
    print(f"\nCurrent weights → Pearson r = {r_current:.4f}")

    # Optimize
    bounds = [WEIGHT_BOUNDS[t] for t in TERMS]
    result = minimize(
        objective, w_current,
        args=(raw_terms, vina_base, pkd, args.reg),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10}
    )

    w_opt = result.x
    r_opt = scoring_power(w_opt, raw_terms, vina_base, pkd)
    print(f"Optimized weights → Pearson r = {r_opt:.4f}")
    print(f"Improvement: {abs(r_opt) - abs(r_current):+.4f}")

    # Cross-validation
    if args.cv > 0:
        train_r, test_r = cross_validate(raw_terms, vina_base, pkd,
                                         n_folds=args.cv, reg_lambda=args.reg)
        print(f"\n{args.cv}-fold CV: train r = {train_r:.4f}, test r = {test_r:.4f}")
        print(f"Overfitting gap: {abs(train_r) - abs(test_r):.4f}")

    # Print optimized weights
    print("\n" + "=" * 60)
    print("  OPTIMIZED DRUSINA WEIGHTS")
    print("=" * 60)
    print(f"{'Term':<22} {'Current':>10} {'Optimized':>10} {'Change':>10}")
    print("-" * 60)
    for i, term in enumerate(TERMS):
        current = DEFAULT_WEIGHTS[term]
        optimized = w_opt[i]
        change = optimized - current
        flag = " ***" if abs(change) > abs(current) * 0.3 else ""
        print(f"  {term:<20} {current:>10.4f} {optimized:>10.4f} {change:>+10.4f}{flag}")

    # Print Swift code snippet
    print("\n" + "=" * 60)
    print("  SWIFT CODE (paste into DockingEngine.swift)")
    print("=" * 60)
    swift_map = {
        "pi_pi": "wPiPi", "pi_cation": "wPiCation",
        "salt_bridge": "wSaltBridge", "amide_pi": "wAmideStack",
        "halogen_bond": "wHalogenBond", "chalcogen_bond": "wChalcogenBond",
        "metal_coord": "wMetalCoord", "coulomb": "wCoulomb",
        "ch_pi": "wCHPi", "torsion_strain": "wTorsionStrain",
        "cooperativity": "wCooperativity",
        "hbond_dir": "wHBondDir", "desolv_polar": "wDesolvPolar",
        "desolv_hydrophobic": "wDesolvHydrophobic",
    }
    for i, term in enumerate(TERMS):
        swift_name = swift_map[term]
        print(f"    {swift_name}: {w_opt[i]:.4f},")

    # Per-term analysis
    print("\n" + "=" * 60)
    print("  PER-TERM STATISTICS (raw, unweighted)")
    print("=" * 60)
    print(f"{'Term':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Fire%':>6}")
    print("-" * 60)
    for i, term in enumerate(TERMS):
        vals = raw_terms[:, i]
        nonzero = np.count_nonzero(vals)
        fire_pct = 100.0 * nonzero / n
        print(f"  {term:<20} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
              f"{np.min(vals):>8.3f} {np.max(vals):>8.3f} {fire_pct:>5.1f}%")


if __name__ == "__main__":
    main()

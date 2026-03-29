#!/usr/bin/env python3
"""
Analyze Drusina per-term scoring decomposition from benchmark results.

Reads a Drusina benchmark JSON (with drusina_decomposition fields) and
reports per-term statistics to identify noisy/spurious interaction terms.

Usage:
  python Benchmark/analyze_decomposition.py Benchmark/results/<file>.json
  python Benchmark/analyze_decomposition.py Benchmark/results/<file>.json --csv decomp.csv
"""

import json
import sys
import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


TERM_NAMES = [
    "pi_pi", "pi_cation", "salt_bridge", "amide_pi",
    "halogen_bond", "chalcogen_bond", "metal_coord",
    "coulomb", "ch_pi", "torsion_strain", "cooperativity", "total",
]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    entries = []
    for e in data.get("entries", []):
        d = e.get("drusina_decomposition")
        if d is None:
            continue
        entries.append({
            "pdb_id": e["pdb_id"],
            "rmsd": e.get("best_rmsd"),
            "energy": e.get("best_energy"),
            "pKd": e.get("experimental_pKd"),
            **{t: d.get(t, 0.0) for t in TERM_NAMES},
        })
    return entries


def analyze(entries):
    if not entries:
        print("No entries with decomposition data found.")
        return

    n = len(entries)
    print(f"\nDrusina Decomposition Analysis  ({n} complexes)\n")
    print(f"{'Term':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Fires':>6} {'|Mean|':>8}  Notes")
    print("-" * 95)

    terms_without_total = [t for t in TERM_NAMES if t != "total"]
    term_stats = {}

    for term in terms_without_total:
        vals = [e[term] for e in entries]
        fires = sum(1 for v in vals if abs(v) > 0.001)
        fire_vals = [v for v in vals if abs(v) > 0.001]

        if np:
            arr = np.array(vals)
            mean, std = float(np.mean(arr)), float(np.std(arr))
            vmin, vmax = float(np.min(arr)), float(np.max(arr))
        else:
            mean = sum(vals) / len(vals) if vals else 0
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0
            vmin, vmax = min(vals), max(vals)

        abs_mean = sum(abs(v) for v in fire_vals) / len(fire_vals) if fire_vals else 0
        fire_pct = fires / n * 100

        notes = []
        if fire_pct > 80 and abs(abs_mean) < 0.1:
            notes.append("HIGH FP RISK: fires often but tiny contribution")
        if fire_pct > 50 and abs(abs_mean) > 0.3:
            notes.append("ACTIVE: frequent and significant")
        if fire_pct < 10:
            notes.append("RARE")

        term_stats[term] = {
            "mean": mean, "std": std, "min": vmin, "max": vmax,
            "fires": fires, "fire_pct": fire_pct, "abs_mean": abs_mean,
        }

        print(f"{term:<18} {mean:>8.3f} {std:>8.3f} {vmin:>8.3f} {vmax:>8.3f} "
              f"{fire_pct:>5.0f}% {abs_mean:>8.3f}  {'  '.join(notes)}")

    # Total
    vals = [e["total"] for e in entries]
    if np:
        mean, std = float(np.mean(vals)), float(np.std(vals))
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
    else:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        vmin, vmax = min(vals), max(vals)
    print("-" * 95)
    print(f"{'total':<18} {mean:>8.3f} {std:>8.3f} {vmin:>8.3f} {vmax:>8.3f}")

    # Correlation with RMSD (does the term correlate with bad poses?)
    rmsds = [e["rmsd"] for e in entries if e["rmsd"] is not None]
    if len(rmsds) >= 5 and np:
        print(f"\n\n{'Term':<18} {'r(RMSD)':>8} {'r(pKd)':>8}  Interpretation")
        print("-" * 70)
        for term in terms_without_total:
            valid = [(e[term], e["rmsd"], e.get("pKd"))
                     for e in entries if e["rmsd"] is not None]
            term_vals = [v[0] for v in valid]
            rmsd_vals = [v[1] for v in valid]
            pkd_vals = [v[2] for v in valid if v[2] is not None]
            term_for_pkd = [v[0] for v in valid if v[2] is not None]

            r_rmsd = float(np.corrcoef(term_vals, rmsd_vals)[0, 1]) if len(term_vals) >= 3 else 0

            r_pkd = 0.0
            if len(pkd_vals) >= 3 and len(term_for_pkd) == len(pkd_vals):
                r_pkd = float(np.corrcoef(term_for_pkd, pkd_vals)[0, 1])

            interp = []
            if abs(r_rmsd) > 0.3:
                direction = "larger when pose is WRONG" if r_rmsd < 0 else "larger when pose is RIGHT"
                interp.append(f"r={r_rmsd:.2f}: term is {direction}")
            if abs(r_pkd) > 0.3:
                interp.append(f"pKd correlated (r={r_pkd:.2f})")
            if not interp:
                interp.append("no significant correlation")

            print(f"{term:<18} {r_rmsd:>8.3f} {r_pkd:>8.3f}  {'  '.join(interp)}")

    # Per-complex breakdown for worst cases
    bad = [e for e in entries if e["rmsd"] is not None and e["rmsd"] > 5.0]
    good = [e for e in entries if e["rmsd"] is not None and e["rmsd"] < 2.0]

    if bad and good:
        print(f"\n\nGood poses (<2A): {len(good)}  |  Bad poses (>5A): {len(bad)}")
        print(f"\n{'Term':<18} {'Good mean':>10} {'Bad mean':>10} {'Delta':>8}  Flag")
        print("-" * 65)
        for term in terms_without_total:
            g_mean = sum(e[term] for e in good) / len(good) if good else 0
            b_mean = sum(e[term] for e in bad) / len(bad) if bad else 0
            delta = b_mean - g_mean
            flag = ""
            if abs(delta) > 0.1:
                flag = "BAD: term more attractive for wrong poses" if delta < -0.1 else "OK: penalizes wrong poses"
            print(f"{term:<18} {g_mean:>10.3f} {b_mean:>10.3f} {delta:>8.3f}  {flag}")


def write_csv(entries, path):
    with open(path, "w") as f:
        header = ["pdb_id", "rmsd", "energy", "pKd"] + TERM_NAMES
        f.write(",".join(header) + "\n")
        for e in entries:
            row = [e.get(h, "") for h in header]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"\nCSV written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Drusina decomposition")
    parser.add_argument("results_json", help="Benchmark results JSON file")
    parser.add_argument("--csv", help="Write per-complex CSV", default=None)
    args = parser.parse_args()

    entries = load_results(args.results_json)
    analyze(entries)

    if args.csv:
        write_csv(entries, args.csv)


if __name__ == "__main__":
    main()

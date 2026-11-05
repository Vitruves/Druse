#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Validate the NumPy Drusina port (Benchmark/drusina_terms.py) against the real
Metal kernel (Metal/DockingScoring.metal) by running the shipped Druse binary
headlessly on a handful of complexes.

For each complex the script drives the headless runner to
  * load the receptor and the crystal ligand,
  * prepare the receptor exactly as the docking pipeline does,
  * export the prepared receptor (so both sides see identical coordinates),
  * dock with scoring=drusina,
  * export the best pose and print its score breakdown,
and then rescores that exported pose with the Python port. The reported
quantity is the total Drusina correction, which `results show` prints as
"Drusina corr." -- that is `decomp.total * drusinaVinaQuality(vinaE)`.

Usage:
  python Benchmark/validate_drusina_port.py --ids 1a28 10gs 1bcu --prep remove
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drusina_terms as dt           # noqa: E402
import drusina_data as dd            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build/Build/Products/Release/Druse.app/Contents/MacOS/Druse"

RE_SCORE = re.compile(r'^\s*Score\s+(-?[\d.]+) kcal/mol', re.M)
RE_DRUS = re.compile(r'^\s*Drusina corr\.\s+(-?[\d.]+) kcal/mol', re.M)
RE_STERIC = re.compile(r'^\s*Steric\s+(-?[\d.]+) kcal/mol', re.M)
RE_TORS = re.compile(r'^\s*Torsion\s+(-?[\d.]+) kcal/mol', re.M)
RE_OTHER = re.compile(r'^\s*Intra/other\s+(-?[\d.]+) kcal/mol', re.M)
RE_RMSD = re.compile(r'^\s*RMSD\s+([\d.]+) ', re.M)


def run_engine(pid, workdir, prep, preset):
    d = dd.REFINED_SET / pid
    prot = d / f"{pid}_protein.pdb"
    lig = d / f"{pid}_ligand.sdf"
    if not prot.exists():
        return None, "missing structure"
    out_prot = workdir / f"{pid}_engine_protein.pdb"
    out_pose = workdir / f"{pid}_pose1.sdf"
    cmds = [f"load {prot}", f"load {lig}"]
    if prep != "none":
        cmds.append(f"prep waters={prep}")
    cmds += [
        f"export protein {out_prot} hydrogens=on ligand=off",
        "pocket box fit around=ligand padding=5",
        f"dock preset={preset} scoring=drusina",
        f"export pose 1 {out_pose}",
        "results show 1",
    ]
    script = "; ".join(cmds)
    try:
        p = subprocess.run([str(BINARY), "--eval", script],
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    txt = p.stdout + p.stderr
    if p.returncode != 0:
        return None, f"exit {p.returncode}: {txt.strip().splitlines()[-1] if txt.strip() else ''}"
    if not out_pose.exists() or not out_prot.exists():
        return None, "no export"

    def g(rx):
        m = rx.search(txt)
        return float(m.group(1)) if m else None

    return {
        'protein': out_prot, 'pose': out_pose,
        'score': g(RE_SCORE), 'drusina': g(RE_DRUS),
        'steric': g(RE_STERIC), 'torsion': g(RE_TORS),
        'other': g(RE_OTHER), 'rmsd': g(RE_RMSD),
        'log': txt,
    }, None


def python_rescore(protein_pdb, pose_sdf, keep_water, weights, min_corr):
    mol = dd.load_ligand(str(pose_sdf))
    if mol is None or mol.GetNumConformers() == 0:
        return None, "pose unreadable"
    lig_pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    cen = lig_pos.mean(axis=0)
    recs_all = dt.parse_protein(str(protein_pdb), keep_water=keep_water)
    if not recs_all:
        return None, "protein unreadable"
    all_pos = np.array([r['pos'] for r in recs_all], dtype=np.float64)
    recs = [r for r in recs_all
            if np.sum((np.array(r['pos']) - cen) ** 2) <= 26.0 ** 2]
    pf = dt.build_protein_features(recs, sb_burial_positions=all_pos)
    lt = dt.build_ligand_topology(mol)
    raw, icount, counts = dt.raw_terms(pf, lt, lig_pos, want_counts=True)
    total = float(raw @ weights)
    total = max(total, min_corr)
    vina = dt.vina_energy(pf, lt, lig_pos)
    gated = total * float(dt.drusina_vina_quality(vina))
    return {'raw': raw, 'counts': counts, 'decomp_total': total,
            'vina': vina, 'gated': gated, 'n_lig': lt.n_heavy}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ids', nargs='+',
                    default=['1a28', '10gs', '1bcu', '1e66', '3d4z', '2weg', '1q8t', '3udh'])
    ap.add_argument('--prep', default='remove', choices=['remove', 'keep', 'none'])
    ap.add_argument('--preset', default='fast')
    ap.add_argument('--workdir', default=str(dd.CACHE_DIR / 'validate'))
    ap.add_argument('--solve', action='store_true',
                    help='least-squares recover the weights the binary uses')
    ap.add_argument('--no-coulomb', action='store_true',
                    help='zero wCoulomb on the Python side (pair with --prep none, '
                         'where the engine receptor carries no partial charges so '
                         'its Coulomb term is identically zero)')
    args = ap.parse_args()

    weights = dt.CURRENT_WEIGHTS.copy()
    if args.no_coulomb:
        weights[dt.TERM_INDEX['coulomb']] = 0.0

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    keep_water = (args.prep == 'keep' or args.prep == 'none')

    rows = []
    for pid in args.ids:
        eng, err = run_engine(pid, wd, args.prep, args.preset)
        if eng is None:
            print(f"{pid}: engine failed -- {err}")
            continue
        py, err = python_rescore(eng['protein'], eng['pose'], keep_water,
                                 weights, dt.CURRENT_MIN_CORRECTION)
        if py is None:
            print(f"{pid}: python failed -- {err}")
            continue
        rows.append((pid, eng, py))
        print(f"{pid}: engine drusina={eng['drusina']:+.3f}  "
              f"python drusina={py['gated']:+.3f}  "
              f"(ungated {py['decomp_total']:+.3f})  "
              f"engine score={eng['score']:+.3f} python vina={py['vina']:+.3f} "
              f"rmsd={eng['rmsd']}", flush=True)

    if not rows:
        print("no comparable results")
        return 1

    print(f"\n{'='*104}")
    print("DRUSINA PORT VALIDATION vs Metal/DockingScoring.metal")
    print(f"{'='*104}")
    print(f"{'pdb':<7}{'eng drusina':>13}{'py drusina':>12}{'diff':>9}"
          f"{'eng score':>11}{'py vina':>10}{'score diff':>12}{'pose rmsd':>11}")
    de, dp, se, sp = [], [], [], []
    for pid, eng, py in rows:
        d = py['gated'] - eng['drusina']
        sdiff = (py['vina'] + py['gated']) - eng['score']
        de.append(eng['drusina']); dp.append(py['gated'])
        se.append(eng['score']); sp.append(py['vina'] + py['gated'])
        print(f"{pid:<7}{eng['drusina']:>13.3f}{py['gated']:>12.3f}{d:>9.3f}"
              f"{eng['score']:>11.3f}{py['vina']:>10.3f}{sdiff:>12.3f}"
              f"{(eng['rmsd'] if eng['rmsd'] is not None else float('nan')):>11.2f}")
    de, dp = np.array(de), np.array(dp)
    se, sp = np.array(se), np.array(sp)
    print(f"\nDrusina correction: MAE {np.mean(np.abs(dp-de)):.3f} kcal/mol, "
          f"max |diff| {np.max(np.abs(dp-de)):.3f}, "
          f"bias {np.mean(dp-de):+.3f}")
    if len(de) > 2:
        print(f"                    Pearson r {np.corrcoef(de, dp)[0,1]:.4f}")
    print(f"Total score:        MAE {np.mean(np.abs(sp-se)):.3f} kcal/mol "
          f"(includes the grid-interpolation vs explicit-pair difference)")

    # --- recover the weights the binary is actually using -------------------
    # With --prep none the receptor carries no partial charges, so the engine's
    # Coulomb term is identically zero; that column is dropped and the rest are
    # solved for by bounded least squares against the reported corrections.
    if args.solve and len(rows) >= 6:
        from scipy.optimize import lsq_linear
        A = np.array([py['raw'] for _, _, py in rows])
        b = np.array([eng['drusina'] for _, eng, _ in rows])
        keep = [i for i in range(dt.N_TERMS)
                if dt.TERMS[i] != 'coulomb' and np.any(np.abs(A[:, i]) > 1e-9)]
        lo = np.array([-3.0 if dt.TERMS[i] not in
                       ('torsion_strain', 'desolv_polar', 'desolv_hydrophobic',
                        'cooperativity') else 0.0 for i in keep])
        hi = np.array([0.0 if dt.TERMS[i] not in
                       ('torsion_strain', 'desolv_polar', 'desolv_hydrophobic',
                        'cooperativity') else 5.0 for i in keep])
        r = lsq_linear(A[:, keep], b, bounds=(lo, hi))
        print(f"\n{'='*104}")
        print("WEIGHTS RECOVERED FROM THE BINARY (bounded least squares, "
              f"n={len(rows)} poses, residual rms {np.sqrt(r.cost*2/len(rows)):.3f})")
        print(f"{'term':<22}{'HEAD source':>13}{'recovered':>12}{'pre-calib':>12}")
        PRE = dict(zip(dt.TERMS, [-0.35, -0.65, -0.20, -0.30, -0.40, -0.10,
                                  -0.95, -0.06, -0.04, 1.0, 0.0, -0.25, 0.15, 0.10]))
        for k, i in enumerate(keep):
            print(f"{dt.TERMS[i]:<22}{dt.CURRENT_WEIGHTS[i]:>13.3f}"
                  f"{r.x[k]:>12.3f}{PRE[dt.TERMS[i]]:>12.3f}")
        pred = A[:, keep] @ r.x
        print(f"fit r = {np.corrcoef(pred, b)[0,1]:.4f}")

    print("\nPer-term Python decomposition (weighted, current weights):")
    print(f"{'pdb':<7}" + ''.join(f'{t[:9]:>10}' for t in dt.TERMS))
    for pid, eng, py in rows:
        w = py['raw'] * weights
        print(f"{pid:<7}" + ''.join(f'{v:>10.3f}' for v in w))
    return 0


if __name__ == '__main__':
    sys.exit(main())

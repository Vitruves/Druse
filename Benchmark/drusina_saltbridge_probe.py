#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Focused measurement of the Drusina salt-bridge term over the PDBbind refined
set, split by how the ligand charge sign was assigned.

The full survey showed that like-charge ligand/protein pairs OUTNUMBER
opposite-charge pairs at salt-bridge contact distance, which cannot be true of
a real electrostatic preference. This script isolates the cause by separating
  * ligand atoms with a non-zero FORMAL charge, and
  * ligand atoms labelled only by the Gasteiger |q| > 0.35 fallback that
    DockingScoring.metal uses when the formal charge is zero,
and it additionally reports how often the contact is simply an H-bond that
Vina's H-bond term already scores (double counting).

It is deliberately lean: no hydrogen bookkeeping, no ring/amide detection, so
it runs on a loaded machine in a couple of minutes.

Usage:
  python Benchmark/drusina_saltbridge_probe.py [--max N] [--workers W]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drusina_terms as dt           # noqa: E402
import drusina_data as dd            # noqa: E402

GROUP_ATOMS = {
    'ARG': (('NE', 'NH1', 'NH2', 'CZ'), 1.0, 2),
    'LYS': (('NZ',), 1.0, 1),
    'ASP': (('OD1', 'OD2'), -1.0, 2),
    'GLU': (('OE1', 'OE2'), -1.0, 2),
}


def probe(job):
    pid, prot, lig = job
    try:
        mol = dd.load_ligand(lig)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        lp = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        cen = lp.mean(axis=0)
        lt = dt.build_ligand_topology(mol)

        # charge sign exactly as computeDrusinaCorrections assigns it
        lc = np.zeros(lt.n_heavy)
        src = np.zeros(lt.n_heavy)       # 0 = formal, 1 = Gasteiger fallback
        nz = lt.formal_charge != 0
        lc[nz] = np.sign(lt.formal_charge[nz])
        fb = ~nz & (np.abs(lt.charge) > 0.35)
        lc[fb] = np.sign(lt.charge[fb])
        src[fb] = 1.0
        act = np.where(lc != 0)[0]
        if len(act) == 0:
            return pid, np.zeros((0, 6)), 0, 0

        # protein charged groups + all heavy positions for burial
        pos, name, res, key = [], [], [], []
        with open(prot, errors='replace') as fh:
            for line in fh:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                if line[16:17] not in (' ', 'A'):
                    continue
                el = line[76:78].strip().upper()
                if el in ('H', 'D'):
                    continue
                try:
                    p = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                pos.append(p)
                name.append(line[12:16].strip().upper())
                res.append(line[17:20].strip().upper())
                key.append((line[21:22], line[22:26], line[26:27]))
        if not pos:
            return None
        pos = np.array(pos, dtype=np.float64)

        by_res = defaultdict(list)
        for i, k in enumerate(key):
            by_res[k].append(i)

        gc, gs, gb = [], [], []
        for k, idxs in by_res.items():
            r = res[idxs[0]]
            spec = GROUP_ATOMS.get(r)
            if spec is None:
                continue
            names, sign, need = spec
            g = [i for i in idxs if name[i] in names]
            if len(g) < need:
                continue
            c = pos[g].mean(axis=0)
            if np.sum((c - cen) ** 2) > 20.0 ** 2:
                continue
            d = np.linalg.norm(pos - c, axis=1)
            cnt = float(np.sum((d > 0.5) & (d < 8.0)))
            gc.append(c); gs.append(sign)
            gb.append(max(0.15, min(1.0, cnt / 25.0)))
        if not gc:
            return pid, np.zeros((0, 6)), len(act), 0
        gc = np.array(gc); gs = np.array(gs); gb = np.array(gb)

        # is the contact also an H-bond the Vina term already rewards?
        # ligand atom is N/O and the protein group atom is N/O at < 3.5 A.
        lig_is_no = np.isin(lt.vtype, [dt.VINA_N_P, dt.VINA_N_D, dt.VINA_N_A,
                                       dt.VINA_N_DA, dt.VINA_O_P, dt.VINA_O_D,
                                       dt.VINA_O_A, dt.VINA_O_DA])

        d = np.linalg.norm(lp[act][:, None, :] - gc[None, :, :], axis=2)
        m = d <= 7.0
        ii, jj = np.nonzero(m)
        if len(ii) == 0:
            return pid, np.zeros((0, 6)), len(act), len(gc)
        rows = np.column_stack([
            d[ii, jj],
            lc[act][ii] * gs[jj],
            gb[jj],
            src[act][ii],
            lig_is_no[act][ii].astype(float),
            np.abs(lt.charge[act][ii]),
        ])
        return pid, rows.astype(np.float32), len(act), len(gc)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    jobs = dd.refined_complexes()
    if args.max and args.max < len(jobs):
        rng = np.random.default_rng(args.seed)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), args.max, replace=False))]
    print(f"probing {len(jobs)} complexes on {args.workers} workers", flush=True)

    allrows, nformal, npartial, nboth, nnone, ncpx = [], 0, 0, 0, 0, 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(probe, jobs, chunksize=8), 1):
            if r is None:
                continue
            pid, rows, nact, ngrp = r
            ncpx += 1
            if len(rows):
                allrows.append(rows)
            if n % 1000 == 0:
                print(f"  [{n}/{len(jobs)}] {n/(time.time()-t0):.1f}/s", flush=True)
    T = np.vstack(allrows) if allrows else np.zeros((0, 6))
    print(f"\n{ncpx} complexes, {len(T):,} charged-atom/charged-group contacts "
          f"<= 7 A  ({time.time()-t0:.0f}s)\n")

    out = []

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        out.append(s)

    P(f"{'='*100}")
    P(f"DRUSINA SALT-BRIDGE TERM -- CHARGE PROVENANCE ({ncpx} refined-set complexes)")
    P(f"{'='*100}")
    P("opp/like is the ratio of opposite-charge to like-charge contacts in each")
    P("distance shell. A real electrostatic preference makes this ratio LARGE")
    P("and RISING as the contact tightens. A ratio at or below 1 means the")
    P("charge labels carry no sign information and the term is scoring contacts")
    P("chosen by geometry alone.")

    shells = ((2.4, 2.7), (2.7, 3.0), (3.0, 3.3), (3.3, 3.6), (3.6, 4.0),
              (4.0, 4.5), (4.5, 5.0), (5.0, 6.0), (6.0, 7.0))
    for code, nm in ((None, 'ALL contacts'),
                     (0.0, 'ligand FORMAL charge only'),
                     (1.0, 'Gasteiger |q| > 0.35 fallback only')):
        sub = T if code is None else T[T[:, 3] == code]
        o, l = sub[sub[:, 1] < 0], sub[sub[:, 1] > 0]
        P(f"\n## {nm}:  {len(sub):,} contacts  ({len(o):,} opposite / {len(l):,} like)")
        if len(sub) == 0:
            continue
        P(f"  {'shell (A)':<12}{'opposite':>10}{'like':>10}{'opp/like':>10}"
          f"{'frac lig N/O':>14}")
        for lo, hi in shells:
            a = int(np.sum((o[:, 0] >= lo) & (o[:, 0] < hi)))
            b = int(np.sum((l[:, 0] >= lo) & (l[:, 0] < hi)))
            if a + b == 0:
                continue
            sh = sub[(sub[:, 0] >= lo) & (sub[:, 0] < hi)]
            P(f"  {lo:.1f}-{hi:.1f}      {a:>10,}{b:>10,}"
              f"{(a/b if b else float('inf')):>10.2f}{np.mean(sh[:,4]):>14.3f}")
        w = sub[sub[:, 0] <= 4.0]
        if len(w):
            ow = w[w[:, 1] < 0]
            P(f"  inside the Drusina window (d<=4.0): {len(w):,} total,"
              f" {len(ow):,} scored as salt bridges")
            P(f"    of the scored ones, {100.0*np.mean(ow[:,4]):.1f}% have a ligand"
              f" N/O atom, i.e. are H-bonds Vina already scores")
            P(f"    burial factor p10/50/90 ="
              f" {np.percentile(ow[:,2],10):.2f} {np.percentile(ow[:,2],50):.2f}"
              f" {np.percentile(ow[:,2],90):.2f}"
              f"  (fraction at the 1.0 cap: {np.mean(ow[:,2]>=0.999):.3f})")

    P(f"\n## Share of the term's total raw value by charge provenance (d<=4.0)")
    w = T[T[:, 0] <= 4.0]
    o = w[w[:, 1] < 0]
    if len(o):
        ds = dt.drusina_ramp(o[:, 0] - 2.8, 0.3, 1.2)
        scale = np.where(o[:, 3] == 1.0, 0.5, 1.0)
        val = ds * o[:, 2] * scale
        for code, nm in ((0.0, 'formal charge'), (1.0, 'Gasteiger fallback')):
            s = o[:, 3] == code
            P(f"  {nm:<22} n={int(s.sum()):>7,}  raw sum {val[s].sum():>9.1f}"
              f"  ({100.0*val[s].sum()/max(val.sum(),1e-9):>5.1f}% of the term)")

    if args.out:
        Path(args.out).write_text('\n'.join(out) + '\n')
        print(f"\nwritten to {args.out}")


if __name__ == '__main__':
    main()

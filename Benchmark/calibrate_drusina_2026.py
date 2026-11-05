#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Parts 3 and 4 of the Drusina recalibration.

Part 3 -- refit every Drusina weight against experimental pKd on the PDBbind
v2020 refined set with the CASF-2016 core set held out, by ridge regression
with k-fold cross-validation, and report Pearson r / RMSE for
  (a) the Vina baseline alone,
  (b) the current in-tree Drusina weights,
  (c) the refit,
plus per-term t statistics and contribution so that dead terms are visible.

Part 4 -- evaluate the same weight sets on CASF-2016 docking power (every
decoy pose of all 285 core-set targets, with the official RMSD tables) and
ranking power (57 clusters of 5), because good scoring power on native
crystal poses does not imply good ranking against decoys. The per-term
behaviour as a function of decoy RMSD is tabulated, which is the direct test
of the hypotheses in Benchmark/reports/drusina_regression_analysis.md.

Usage:
  python Benchmark/calibrate_drusina_2026.py --stage features     # build caches
  python Benchmark/calibrate_drusina_2026.py --stage fit
  python Benchmark/calibrate_drusina_2026.py --stage casf
  python Benchmark/calibrate_drusina_2026.py --stage all --report out.txt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drusina_terms as dt           # noqa: E402
import drusina_data as dd            # noqa: E402

POCKET_RADIUS = 26.0                 # >= max ligand radius + 12 A Coulomb reach

# Terms that may only be favourable (negative weight) or only a penalty.
NEG_ONLY = {'pi_pi', 'pi_cation', 'salt_bridge', 'amide_pi', 'halogen_bond',
            'chalcogen_bond', 'metal_coord', 'ch_pi', 'hbond_dir'}
POS_ONLY = {'torsion_strain', 'desolv_polar', 'desolv_hydrophobic'}
# every other term ('coulomb', 'cooperativity') is left free to take either sign


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

VARIANTS = ('v1', 'v2', 'v2sum')


def _all_variants(pf_shipped, pf_peptide, lt, pos):
    """The three term vectors compared throughout: the shipped kernel (with
    its C(i),O(i),N(i) amide planes), the rewritten kernel with the Section-7
    cutoffs and correct peptide planes, and the same cutoffs still summing over
    partners instead of taking the best one."""
    return (dt.raw_terms(pf_shipped, lt, pos)[0],
            dt.raw_terms_v2(pf_peptide, lt, pos)[0],
            dt.raw_terms_v2(pf_peptide, lt, pos, one_per_site=False)[0])


def _features_refined(job):
    pid, prot, lig = job
    try:
        mol = dd.load_ligand(lig)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        lp = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        if len(lp) < 3:
            return None
        cen = lp.mean(axis=0)
        recs_all = dt.parse_protein(prot)
        if not recs_all:
            return None
        ap = np.array([r['pos'] for r in recs_all], dtype=np.float64)
        recs = [r for r in recs_all
                if np.sum((np.array(r['pos']) - cen) ** 2) <= POCKET_RADIUS ** 2]
        if len(recs) < 20:
            return None
        pf = dt.build_protein_features(recs, sb_burial_positions=ap)
        pfp = dt.build_protein_features(recs, sb_burial_positions=ap,
                                        peptide_amide_planes=True)
        lt = dt.build_ligand_topology(mol)
        raws = _all_variants(pf, pfp, lt, lp)
        vina = dt.vina_energy(pf, lt, lp)
        return pid, raws, vina, lt.n_heavy, lt.n_rot
    except Exception:
        return None


def _features_casf(job):
    """Score the crystal pose plus every docking decoy of one CASF target."""
    pid, prot, lig, dec, rms = job[:5]
    try:
        mol = dd.load_ligand(lig)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        ref = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        cen = ref.mean(axis=0)
        recs_all = dt.parse_protein(prot)
        ap = np.array([r['pos'] for r in recs_all], dtype=np.float64)
        recs = [r for r in recs_all
                if np.sum((np.array(r['pos']) - cen) ** 2) <= POCKET_RADIUS ** 2]
        if len(recs) < 20:
            return None
        pf = dt.build_protein_features(recs, sb_burial_positions=ap)
        pfp = dt.build_protein_features(recs, sb_burial_positions=ap,
                                        peptide_amide_planes=True)
        lt = dt.build_ligand_topology(mol)

        poses, vinas, rmsds = [], [], []
        raws = {v: [] for v in VARIANTS}

        def _add(pos):
            for k, v in enumerate(VARIANTS):
                raws[v].append(_all_variants(pf, pfp, lt, pos)[k])

        _add(ref)
        vinas.append(dt.vina_energy(pf, lt, ref))
        rmsds.append(0.0)
        poses.append(f'{pid}_native')

        # Decoy poses come as a multi-model mol2 whose atom order must match
        # the RDKit topology built from the crystal ligand. Verified by element
        # symbol, not assumed: a silent permutation would scramble every
        # index-based feature (rings, halogens, torsions).
        lig_sym = [mol.GetAtomWithIdx(i).GetSymbol().upper()
                   for i in range(lt.n_heavy)]
        if dec and rms:
            rmap = dd.parse_rmsd_dat(rms)
            for name, block in dd.iter_mol2_blocks(dec):
                if name not in rmap:
                    continue
                pos, syms = dd.mol2_block_coords(block)
                if len(pos) != lt.n_heavy:
                    continue
                if [x.upper() for x in syms] != lig_sym:
                    continue
                _add(pos)
                vinas.append(dt.vina_energy(pf, lt, pos))
                rmsds.append(rmap[name])
                poses.append(name)
        return (pid, {v: np.array(raws[v]) for v in VARIANTS},
                np.array(vinas), np.array(rmsds), poses)
    except Exception:
        return None


def build_refined_cache(path, workers, limit=0, seed=7):
    jobs = dd.refined_complexes()
    if limit and limit < len(jobs):
        rng = np.random.default_rng(seed)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), limit, replace=False))]
    print(f"featurising {len(jobs)} refined-set complexes on {workers} workers",
          flush=True)
    ids, raws, vinas, nh, nr = [], {v: [] for v in VARIANTS}, [], [], []
    t0 = time.time()
    bad = 0
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_features_refined, jobs, chunksize=8), 1):
            if r is None:
                bad += 1
            else:
                ids.append(r[0])
                for k, v in enumerate(VARIANTS):
                    raws[v].append(r[1][k])
                vinas.append(r[2]); nh.append(r[3]); nr.append(r[4])
            if n % 1000 == 0:
                el = time.time() - t0
                print(f"  [{n}/{len(jobs)}] {n/el:.1f}/s ok={len(ids)} bad={bad}",
                      flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids),
                        vina=np.array(vinas), n_heavy=np.array(nh),
                        n_rot=np.array(nr),
                        **{f'raw_{v}': np.array(raws[v]) for v in VARIANTS})
    print(f"wrote {path}  ({len(ids)} ok, {bad} failed, {time.time()-t0:.0f}s)")


def build_casf_cache(path, workers, sample=0, seed=7):
    jobs = dd.casf_targets()
    if sample and sample < len(jobs):
        rng = np.random.default_rng(seed)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), sample, replace=False))]
    print(f"featurising {len(jobs)} CASF targets (crystal + decoys), all "
          f"{len(VARIANTS)} term variants, on {workers} workers", flush=True)
    out = {}
    t0 = time.time()
    bad = 0
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_features_casf, jobs, chunksize=2), 1):
            if r is None:
                bad += 1
            else:
                pid, raws, vinas, rmsds, _ = r
                for v in VARIANTS:
                    out[f'{pid}_raw_{v}'] = raws[v]
                out[f'{pid}_vina'] = vinas
                out[f'{pid}_rmsd'] = rmsds
            if n % 50 == 0:
                print(f"  [{n}/{len(jobs)}] {n/(time.time()-t0):.2f}/s bad={bad}",
                      flush=True)
    out['targets'] = np.array(sorted({k.split('_')[0] for k in out}))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path}  ({len(out['targets'])} targets, {bad} failed, "
          f"{time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def bounds_for(terms, allow=None):
    lo, hi = [], []
    for t in terms:
        if allow is not None and t not in allow:
            # lsq_linear needs lo < hi strictly, so pin the term to ~0 rather
            # than dropping the column (keeps the weight vector shape stable)
            lo.append(-1e-9); hi.append(0.0)
        elif t in NEG_ONLY:
            lo.append(-3.0); hi.append(0.0)
        elif t in POS_ONLY:
            lo.append(0.0); hi.append(5.0)
        else:
            lo.append(-1.0); hi.append(1.0)
    return np.array(lo), np.array(hi)


def fit_weights(raw, vina, pkd, l2=1.0, allow=None, min_corr=None):
    """Least squares for pKd ~ a*(vina + raw@w) + b, w sign-bounded.

    Fitting the affine map alongside the weights is what makes RMSE in pKd
    units meaningful; Pearson r is invariant to it. min_corr applies the
    minCorrection floor inside the fit (non-linear, so it is applied as a
    fixed-point iteration: two passes are enough in practice).
    """
    from scipy.optimize import lsq_linear
    lo, hi = bounds_for(dt.TERMS, allow)
    # pKd = a*vina + (a*w) @ raw + b  ->  solve for (a*w) and the affine part
    # by alternating: with a fixed, the weight problem is linear.
    a, b = _affine(vina, pkd)
    w = np.zeros(dt.N_TERMS)
    for _ in range(12):
        corr = raw @ w
        if min_corr is not None:
            # only rows not clipped contribute gradient to the weights
            active = corr > min_corr
        else:
            active = np.ones(len(corr), dtype=bool)
        target = (pkd - b) / a - vina
        A = raw[active]
        reg = np.sqrt(l2) * np.eye(dt.N_TERMS)
        Aa = np.vstack([A, reg])
        ba = np.concatenate([target[active], np.zeros(dt.N_TERMS)])
        r = lsq_linear(Aa, ba, bounds=(lo, hi), max_iter=200)
        w_new = r.x
        corr = raw @ w_new
        if min_corr is not None:
            corr = np.maximum(corr, min_corr)
        a, b = _affine(vina + corr, pkd)
        if np.max(np.abs(w_new - w)) < 1e-5:
            w = w_new
            break
        w = w_new
    return w, a, b


def _affine(x, y):
    sl, ic, _, _, _ = stats.linregress(x, y)
    if abs(sl) < 1e-9:
        sl = -1e-9
    return sl, ic


def evaluate(raw, vina, pkd, w, min_corr=None):
    corr = raw @ w
    if min_corr is not None:
        corr = np.maximum(corr, min_corr)
    e = vina + corr
    r = stats.pearsonr(e, pkd)[0]
    sp = stats.spearmanr(e, pkd)[0]
    a, b = _affine(e, pkd)
    rmse = float(np.sqrt(np.mean((a * e + b - pkd) ** 2)))
    return r, sp, rmse


# ---------------------------------------------------------------------------
# CASF power metrics
# ---------------------------------------------------------------------------

def docking_power(casf, w, variant='v1', min_corr=None, cutoff=2.0,
                  top=(1, 2, 3), include_native=False):
    """Fraction of targets whose best-scored pose is within `cutoff` A.

    Standard CASF-2016 docking power: the candidate set is the official decoy
    ensemble only (row 0 of the cache is the crystal pose and is dropped unless
    include_native, because handing the scorer the answer makes the test a
    different, easier one). For top-N the target counts as a success if ANY of
    the N best-scored poses is within the RMSD cutoff.
    """
    hits = {n: 0 for n in top}
    n_tgt = 0
    best_rmsd = []
    for pid in casf['targets']:
        key = f'{pid}_raw_{variant}'
        if key not in casf:
            continue
        raw = casf[key]
        vina, rmsd = casf[f'{pid}_vina'], casf[f'{pid}_rmsd']
        if not include_native:
            raw, vina, rmsd = raw[1:], vina[1:], rmsd[1:]
        if len(rmsd) < 5:
            continue
        corr = raw @ w
        if min_corr is not None:
            corr = np.maximum(corr, min_corr)
        e = vina + corr
        order = np.argsort(e)
        n_tgt += 1
        best_rmsd.append(rmsd[order[0]])
        for n in top:
            if np.any(rmsd[order[:n]] <= cutoff):
                hits[n] += 1
    return ({n: hits[n] / max(n_tgt, 1) for n in top}, n_tgt,
            np.array(best_rmsd))


def ranking_power(casf, w, pkd_map, cluster_map, variant='v1', min_corr=None):
    """Spearman rho within each 5-member CASF cluster, averaged, plus the
    fraction of clusters whose highest-affinity ligand scores best."""
    groups = {}
    for pid, cl in cluster_map.items():
        groups.setdefault(cl, []).append(pid)
    rhos, top1 = [], []
    for cl, members in groups.items():
        es, ps = [], []
        for pid in members:
            key = f'{pid}_raw_{variant}'
            if key not in casf:
                continue
            # native pose only (row 0) -- ranking power scores the crystal pose
            corr = float(casf[key][0] @ w)
            if min_corr is not None:
                corr = max(corr, min_corr)
            es.append(casf[f'{pid}_vina'][0] + corr)
            ps.append(pkd_map[pid])
        if len(es) < 3:
            continue
        rho = stats.spearmanr(es, ps)[0]
        if np.isfinite(rho):
            rhos.append(rho)
        top1.append(1.0 if int(np.argmin(es)) == int(np.argmax(ps)) else 0.0)
    return (float(np.mean(rhos)) if rhos else float('nan'),
            float(np.mean(top1)) if top1 else float('nan'), len(rhos))


def scoring_power_casf(casf, w, pkd_map, variant='v1', min_corr=None):
    es, ps = [], []
    for pid in casf['targets']:
        key = f'{pid}_raw_{variant}'
        if key not in casf or pid not in pkd_map:
            continue
        corr = float(casf[key][0] @ w)
        if min_corr is not None:
            corr = max(corr, min_corr)
        es.append(casf[f'{pid}_vina'][0] + corr)
        ps.append(pkd_map[pid])
    es, ps = np.array(es), np.array(ps)
    if len(es) < 5:
        return float('nan'), float('nan'), 0
    r = stats.pearsonr(es, ps)[0]
    a, b = _affine(es, ps)
    return r, float(np.sqrt(np.mean((a * es + b - ps) ** 2))), len(es)


# ---------------------------------------------------------------------------
# Recommended weights (Benchmark/reports/drusina_calibration_2026.md section 8)
# ---------------------------------------------------------------------------

def _w(**kw):
    w = np.zeros(dt.N_TERMS)
    for k, v in kw.items():
        w[dt.TERM_INDEX[k]] = v
    return w


# The terms whose v2 near-native/decoy contrast ratio exceeds ~1.5, at plausible
# physical magnitudes. `RECOMMENDED` is this set at the scale-0.5 plateau of
# section 6.4; `_CANDIDATE_BASE` is the unscaled version the scan sweeps.
_CANDIDATE_BASE = _w(metal_coord=-1.70, halogen_bond=-0.60, pi_pi=-0.30,
                     amide_pi=-0.30, salt_bridge=-0.30, ch_pi=-0.25,
                     hbond_dir=-0.15, cooperativity=-0.05)
RECOMMENDED = _w(metal_coord=-0.85, halogen_bond=-0.30, pi_pi=-0.15,
                 amide_pi=-0.15, salt_bridge=-0.15, ch_pi=-0.12,
                 hbond_dir=-0.08, cooperativity=-0.025)
RECOMMENDED_VARIANT = 'v2'


def flip_stats(casf, w, variant, min_corr=None, cutoff=2.0):
    """How often the correction flips Vina's best pose, and whether that helps.

    This is the diagnostic that identifies the magnitude plateau: a correction
    small enough to leave Vina's wide margins alone flips few poses and flips
    them favourably, while a large one flips many and flips them badly.
    """
    flips = helps = hurts = n = 0
    mags = []
    for pid in casf['targets']:
        key = f'{pid}_raw_{variant}'
        if key not in casf:
            continue
        raw = casf[key][1:]
        vina = casf[f'{pid}_vina'][1:]
        rmsd = casf[f'{pid}_rmsd'][1:]
        if len(rmsd) < 5:
            continue
        corr = raw @ w
        if min_corr is not None:
            corr = np.maximum(corr, min_corr)
        mags.append(np.abs(corr))
        n += 1
        iv = int(np.argmin(vina))
        idr = int(np.argmin(vina + corr))
        if iv != idr:
            flips += 1
            if rmsd[idr] <= cutoff < rmsd[iv]:
                helps += 1
            elif rmsd[iv] <= cutoff < rmsd[idr]:
                hurts += 1
    mag = float(np.concatenate(mags).mean()) if mags else 0.0
    return flips / max(n, 1), helps / max(n, 1), hurts / max(n, 1), mag


def run_candidates(casf, refined, args, P):
    """Evaluate named weight sets, and scan one global scale factor on the
    recommended set, against docking / ranking / scoring / pKd at once."""
    pkd_map, cluster_map = dd.parse_casf_coreset()
    pkd_all = dd.parse_pkd_index()
    mc = dt.CURRENT_MIN_CORRECTION if args.apply_min_correction else None
    core = set(pkd_map)

    ids = refined['ids']
    keep = np.array([i for i, p in enumerate(ids) if p in pkd_all])
    y = np.array([pkd_all[ids[i]] for i in keep])
    tr = np.array([ids[i] not in core for i in keep])
    vina_tr = refined['vina'][keep][tr]
    y_tr = y[tr]

    def row(nm, var, w):
        h, _, br = docking_power(casf, w, var, mc)
        rho, t1, _ = ranking_power(casf, w, pkd_map, cluster_map, var, mc)
        r, rm, _ = scoring_power_casf(casf, w, pkd_map, var, mc)
        fl, he, hu, mag = flip_stats(casf, w, var, mc)
        pr, _, prm = evaluate(refined[f'raw_{var}'][keep][tr], vina_tr, y_tr, w, mc)
        P(f"{nm:<30}{var:>7}{h[1]:>8.3f}{h[2]:>8.3f}{np.median(br):>8.2f}"
          f"{rho:>10.4f}{t1:>8.3f}{r:>9.4f}{rm:>8.4f}{fl:>7.3f}{he:>7.3f}"
          f"{hu:>7.3f}{mag:>8.3f}{pr:>9.4f}{prm:>10.4f}")

    hdr = (f"{'weight set':<30}{'terms':>7}{'dock1':>8}{'dock2':>8}{'medR':>8}"
           f"{'rankRho':>10}{'rankT1':>8}{'scoreR':>9}{'RMSE':>8}{'flip':>7}"
           f"{'help':>7}{'hurt':>7}{'|corr|':>8}{'trainR':>9}{'trainRMSE':>10}")
    P(f"\n{'='*140}")
    P("CANDIDATE WEIGHT SETS -- all four powers at once")
    P(f"{'='*140}")
    P(hdr)
    row('Vina baseline only', 'v1', np.zeros(dt.N_TERMS))
    row('shipped weights', 'v1', dt.CURRENT_WEIGHTS)
    row('shipped weights, v2 terms', 'v2', dt.CURRENT_WEIGHTS)
    row('RECOMMENDED', RECOMMENDED_VARIANT, RECOMMENDED)
    row('RECOMMENDED on v1 terms', 'v1', RECOMMENDED)
    row('RECOMMENDED on v2sum terms', 'v2sum', RECOMMENDED)
    row('metal coordination only', 'v2', _w(metal_coord=-0.85))

    P(f"\n## Global scale scan on the candidate set ({RECOMMENDED_VARIANT} terms)")
    P(hdr)
    for sc in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3):
        row(f'scale {sc:.2f}', RECOMMENDED_VARIANT, _CANDIDATE_BASE * sc)
    P("\n  The plateau where every metric beats Vina is scale 0.3-0.6, i.e. a mean")
    P("  correction magnitude of 0.08-0.16 kcal/mol. Above ~0.2 the flips start to")
    P("  hurt more often than they help and docking power falls below Vina's.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_fit(refined, casf, args, P, variant='v1'):
    ids = refined['ids']
    raw = refined[f'raw_{variant}']
    vina = refined['vina']
    pkd_map = dd.parse_pkd_index()
    core_pkd, core_cluster = dd.parse_casf_coreset()
    core = set(core_pkd)

    keep = np.array([i for i, p in enumerate(ids) if p in pkd_map])
    ids, raw, vina = ids[keep], raw[keep], vina[keep]
    pkd = np.array([pkd_map[p] for p in ids])
    is_core = np.array([p in core for p in ids])

    P(f"\n{'='*104}")
    P(f"PART 3 -- WEIGHT REFIT AGAINST EXPERIMENTAL pKd  [term variant: {variant}]")
    P(f"{'='*104}")
    P(f"refined-set complexes featurised and labelled : {len(ids)}")
    P(f"of which are CASF-2016 core set (held out)    : {int(is_core.sum())}")
    tr = ~is_core
    P(f"training pool                                 : {int(tr.sum())}")

    Xtr, Vtr, Ytr = raw[tr], vina[tr], pkd[tr]
    Xte, Vte, Yte = raw[is_core], vina[is_core], pkd[is_core]

    mc = dt.CURRENT_MIN_CORRECTION if args.apply_min_correction else None

    P("\n## Baselines on the training pool (n=%d)" % tr.sum())
    P(f"{'model':<34}{'Pearson r':>11}{'Spearman':>10}{'RMSE(pKd)':>11}")
    zero = np.zeros(dt.N_TERMS)
    for nm, w in (('(a) Vina baseline only', zero),
                  ('(b) current in-tree Drusina weights', dt.CURRENT_WEIGHTS)):
        r, sp, rm = evaluate(Xtr, Vtr, Ytr, w, mc)
        P(f"{nm:<34}{r:>11.4f}{sp:>10.4f}{rm:>11.4f}")

    # ---- k-fold CV refit
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(int(tr.sum()))
    folds = np.array_split(idx, args.folds)
    cv_r, cv_rm, cv_w = [], [], []
    for k in range(args.folds):
        te = folds[k]
        trn = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        w, a, b = fit_weights(Xtr[trn], Vtr[trn], Ytr[trn], l2=args.l2, min_corr=mc)
        r, sp, rm = evaluate(Xtr[te], Vtr[te], Ytr[te], w, mc)
        cv_r.append(r); cv_rm.append(rm); cv_w.append(w)
    cv_w = np.array(cv_w)
    P(f"\n## {args.folds}-fold CV on the training pool "
      f"(ridge l2={args.l2}, sign-bounded)")
    P(f"held-out Pearson r  : {np.mean(cv_r):.4f} +/- {np.std(cv_r):.4f}")
    P(f"held-out RMSE (pKd) : {np.mean(cv_rm):.4f} +/- {np.std(cv_rm):.4f}")

    w_fit, a_fit, b_fit = fit_weights(Xtr, Vtr, Ytr, l2=args.l2, min_corr=mc)

    P(f"\n## (c) Refit on the whole training pool, evaluated on the held-out "
      f"CASF core set (n={len(Yte)})")
    P(f"{'model':<34}{'Pearson r':>11}{'Spearman':>10}{'RMSE(pKd)':>11}")
    for nm, w in (('(a) Vina baseline only', zero),
                  ('(b) current Drusina weights', dt.CURRENT_WEIGHTS),
                  ('(c) refit', w_fit)):
        r, sp, rm = evaluate(Xte, Vte, Yte, w, mc)
        P(f"{nm:<34}{r:>11.4f}{sp:>10.4f}{rm:>11.4f}")

    # ---- per-term statistics
    P("\n## Per-term statistics on the training pool")
    P("  fires%   = fraction of complexes where the raw term is non-zero")
    P("  partial r = correlation of the raw term with the pKd residual left by Vina")
    P("  t         = OLS t statistic of the term in the joint linear model")
    P("  |contrib| = mean |w_fit * raw| in kcal/mol (how much energy it moves)")
    resid = Ytr - (_affine(Vtr, Ytr)[0] * Vtr + _affine(Vtr, Ytr)[1])
    # joint OLS for t statistics (unbounded, for inference only)
    A = np.column_stack([Vtr, Xtr, np.ones(len(Vtr))])
    coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    pred = A @ coef
    dof = len(Ytr) - A.shape[1]
    s2 = np.sum((Ytr - pred) ** 2) / dof
    try:
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        tstat = np.where(se > 0, coef / np.maximum(se, 1e-12), 0.0)
    except np.linalg.LinAlgError:
        tstat = np.zeros(A.shape[1])

    P(f"\n{'term':<22}{'symbol':<22}{'current':>9}{'refit':>9}"
      f"{'fires%':>8}{'part.r':>8}{'t':>9}{'|contrib|':>11}")
    rows = []
    for i, t in enumerate(dt.TERMS):
        v = Xtr[:, i]
        nz = np.count_nonzero(v) / len(v)
        pr = stats.pearsonr(v, resid)[0] if np.std(v) > 1e-12 else 0.0
        contrib = float(np.mean(np.abs(w_fit[i] * v)))
        rows.append((t, dt.SWIFT_SYMBOL[t], dt.CURRENT_WEIGHTS[i], w_fit[i],
                     nz, pr, tstat[1 + i], contrib))
        P(f"{t:<22}{dt.SWIFT_SYMBOL[t]:<22}{dt.CURRENT_WEIGHTS[i]:>9.3f}"
          f"{w_fit[i]:>9.3f}{100*nz:>7.1f}%{pr:>8.3f}{tstat[1+i]:>9.2f}"
          f"{contrib:>11.4f}")

    P(f"\nCV weight stability (mean +/- sd over the {args.folds} folds):")
    P(f"{'term':<22}{'mean':>9}{'sd':>9}{'sign stable':>13}")
    for i, t in enumerate(dt.TERMS):
        col = cv_w[:, i]
        stable = 'yes' if (np.all(col <= 1e-9) or np.all(col >= -1e-9)) else 'NO'
        P(f"{t:<22}{col.mean():>9.3f}{col.std():>9.3f}{stable:>13}")

    # ---- leave-one-term-out: does each term earn its place?
    P("\n## Leave-one-term-out (refit without the term, held-out CV r)")
    P(f"{'removed term':<22}{'CV r':>10}{'delta vs full':>15}{'verdict':>12}")
    full_cv = np.mean(cv_r)
    allow_all = set(dt.TERMS)
    loo = {}
    for t in dt.TERMS:
        allow = allow_all - {t}
        rs = []
        for k in range(args.folds):
            te = folds[k]
            trn = np.concatenate([folds[j] for j in range(args.folds) if j != k])
            w, _, _ = fit_weights(Xtr[trn], Vtr[trn], Ytr[trn],
                                  l2=args.l2, allow=allow, min_corr=mc)
            rs.append(evaluate(Xtr[te], Vtr[te], Ytr[te], w, mc)[0])
        m = float(np.mean(rs))
        loo[t] = m
        d = m - full_cv
        verdict = 'needed' if d > 0.002 else ('neutral' if d > -0.0005 else 'HELPS')
        # r is negative (energy vs pKd): "more negative is better"
        verdict = 'earns it' if d > 0.002 else ('dead' if d >= -0.0005 else 'harmful')
        P(f"{t:<22}{m:>10.4f}{d:>+15.4f}{verdict:>12}")
    P("  (r is negative because energy anti-correlates with pKd; a term 'earns"
      "  it' when removing it makes r less negative, i.e. delta > 0.)")

    return w_fit, loo, (ids, raw, vina, pkd, is_core)


def run_casf(casf, weights_named, args, P):
    pkd_map, cluster_map = dd.parse_casf_coreset()
    mc = dt.CURRENT_MIN_CORRECTION if args.apply_min_correction else None

    P(f"\n{'='*104}")
    P("PART 4 -- CASF-2016 HELD-OUT POWER (docking against decoys, ranking, scoring)")
    P(f"{'='*104}")
    ntot = sum(len(casf[f'{p}_rmsd']) for p in casf['targets']
               if f'{p}_rmsd' in casf)
    P(f"variants compared: v1 = shipped kernel; v2 = rewritten kernel with the"
      f" measured cutoffs;\n                   v2sum = measured cutoffs but"
      f" still summing over all partners")
    P(f"targets: {len(casf['targets'])}   scored poses (crystal + decoys): {ntot}")

    for inc in (False, True):
        P(f"\n## Docking power (best-scored pose within 2.0 A of the crystal"
          f" pose){'  [crystal pose ADDED to the candidate set]' if inc else ''}")
        P(f"{'weight set':<34}{'top1':>8}{'top2':>8}{'top3':>8}"
          f"{'median best-pose RMSD':>24}")
        for nm, var, w in weights_named:
            hits, n, br = docking_power(casf, w, var, mc, include_native=inc)
            P(f"{nm:<34}{hits[1]:>8.3f}{hits[2]:>8.3f}{hits[3]:>8.3f}"
              f"{np.median(br):>24.2f}")

    P(f"\n## Ranking power (within-cluster Spearman, crystal poses, 57 clusters)")
    P(f"{'weight set':<34}{'mean rho':>11}{'top1 hit rate':>15}{'clusters':>10}")
    for nm, var, w in weights_named:
        rho, t1, nc = ranking_power(casf, w, pkd_map, cluster_map, var, mc)
        P(f"{nm:<34}{rho:>11.4f}{t1:>15.3f}{nc:>10}")

    P(f"\n## Scoring power on the CASF core set (crystal poses)")
    P(f"{'weight set':<34}{'Pearson r':>11}{'RMSE(pKd)':>11}{'n':>6}")
    for nm, var, w in weights_named:
        r, rm, n = scoring_power_casf(casf, w, pkd_map, var, mc)
        P(f"{nm:<34}{r:>11.4f}{rm:>11.4f}{n:>6}")

    # ---- the decisive diagnostic: term behaviour vs decoy RMSD
    P(f"\n## Mean RAW term value vs decoy RMSD (all {ntot} poses pooled)")
    P("A term that is useful for docking must DECAY with RMSD. A term that is")
    P("flat, or that grows with RMSD, rewards wrong poses and is the mechanism")
    P("behind the ranking failures in drusina_regression_analysis.md.")
    variants_present = [v for v in VARIANTS if f"{casf['targets'][0]}_raw_{v}" in casf]
    tables = {}
    for var in variants_present:
        allraw, allrmsd, allvina = [], [], []
        for p in casf['targets']:
            if f'{p}_raw_{var}' not in casf:
                continue
            allraw.append(casf[f'{p}_raw_{var}'][1:])
            allrmsd.append(casf[f'{p}_rmsd'][1:])
            allvina.append(casf[f'{p}_vina'][1:])
        tables[var] = (np.vstack(allraw), np.concatenate(allrmsd),
                       np.concatenate(allvina))
    R, D, V = tables[variants_present[0]]
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12)]
    hdr = f"{'term':<22}" + ''.join(f'{f"{a}-{b}":>9}' for a, b in bins) + f"{'r(term,RMSD)':>14}"
    P(hdr)
    P(f"{'n poses':<22}" + ''.join(
        f'{int(np.sum((D>=a)&(D<b))):>9}' for a, b in bins))
    P(f"{'VINA (kcal/mol)':<22}" + ''.join(
        f'{np.mean(V[(D>=a)&(D<b)]) if np.any((D>=a)&(D<b)) else float("nan"):>9.2f}'
        for a, b in bins) + f"{stats.pearsonr(V, D)[0]:>14.3f}")
    for i, t in enumerate(dt.TERMS):
        vals = R[:, i]
        line = f"{t:<22}"
        for a, b in bins:
            m = (D >= a) & (D < b)
            line += f'{np.mean(vals[m]) if m.any() else float("nan"):>9.3f}'
        rr = stats.pearsonr(vals, D)[0] if np.std(vals) > 1e-12 else float('nan')
        P(line + f"{rr:>14.3f}")
    P("\n  r(term, RMSD) > 0 for a favourable (negative-weight) term means the")
    P("  term gets LARGER as the pose gets worse -- it actively mis-ranks.")

    P(f"\n## Pose discrimination per variant: r(raw term, decoy RMSD), and the")
    P(f"   near-native / decoy contrast  mean(0-1 A) / mean(4-8 A)")
    hdr = f"{'term':<22}"
    for var in variants_present:
        hdr += f"{var + ' r':>10}{var + ' ratio':>13}"
    P(hdr)
    for i, t in enumerate(dt.TERMS):
        line = f"{t:<22}"
        for var in variants_present:
            Rv, Dv, _ = tables[var]
            vals = Rv[:, i]
            rr = stats.pearsonr(vals, Dv)[0] if np.std(vals) > 1e-12 else float('nan')
            near = vals[Dv <= 1.0]
            far = vals[(Dv >= 4.0) & (Dv <= 8.0)]
            ratio = (near.mean() / far.mean()) if (len(far) and abs(far.mean()) > 1e-9) else float('nan')
            line += f"{rr:>10.3f}{ratio:>13.2f}"
        P(line)
    P("   ratio > 1 means the term is larger on near-native poses, which is what")
    P("   a term has to do to help docking. ratio <= 1 means it cannot help.")

    # ---- how often does the Drusina correction flip the Vina ranking?
    P(f"\n## Does the correction override the Vina landscape?")
    P(f"{'weight set':<34}{'flips vina top1':>17}{'flip helps':>12}"
      f"{'flip hurts':>12}{'mean |corr|':>13}{'cap hits%':>11}")
    for nm, var, w in weights_named:
        flips = helps = hurts = 0
        n = 0
        corr_mag, capped = [], 0
        tot = 0
        for p in casf['targets']:
            if f'{p}_raw_{var}' not in casf:
                continue
            raw = casf[f'{p}_raw_{var}'][1:]
            vina = casf[f'{p}_vina'][1:]
            rmsd = casf[f'{p}_rmsd'][1:]
            if len(rmsd) < 5:
                continue
            corr = raw @ w
            tot += len(corr)
            if mc is not None:
                capped += int(np.sum(corr < mc))
                corr = np.maximum(corr, mc)
            corr_mag.append(np.abs(corr))
            iv = int(np.argmin(vina))
            id_ = int(np.argmin(vina + corr))
            n += 1
            if iv != id_:
                flips += 1
                if rmsd[id_] <= 2.0 < rmsd[iv]:
                    helps += 1
                elif rmsd[iv] <= 2.0 < rmsd[id_]:
                    hurts += 1
        cm = np.concatenate(corr_mag)
        P(f"{nm:<34}{flips/max(n,1):>17.3f}{helps/max(n,1):>12.3f}"
          f"{hurts/max(n,1):>12.3f}{cm.mean():>13.3f}"
          f"{100.0*capped/max(tot,1):>11.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all',
                    choices=['features', 'casf-features', 'fit', 'casf',
                             'candidates', 'all'])
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--apply-min-correction', action='store_true', default=True)
    ap.add_argument('--no-min-correction', dest='apply_min_correction',
                    action='store_false')
    ap.add_argument('--report', default='')
    ap.add_argument('--casf-sample', type=int, default=0,
                    help='limit the CASF targets featurised (random, stated)')
    ap.add_argument('--casf-cache', default='')
    ap.add_argument('--refined-cache', default='')
    args = ap.parse_args()
    nw = args.workers or min(mp.cpu_count(), 8)

    fref = Path(args.refined_cache) if args.refined_cache else \
        dd.CACHE_DIR / 'refined_features.npz'
    fcasf = Path(args.casf_cache) if args.casf_cache else \
        dd.CACHE_DIR / 'casf_features.npz'

    if args.stage in ('features', 'all') and not fref.exists():
        build_refined_cache(fref, nw, args.limit)
    if args.stage in ('casf-features', 'all') and not fcasf.exists():
        build_casf_cache(fcasf, nw, sample=args.casf_sample)
    if args.stage in ('features', 'casf-features'):
        return 0

    out = []

    def P(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        out.append(line)

    casf = dict(np.load(fcasf, allow_pickle=False)) if fcasf.exists() else None

    fits = {}
    if args.stage in ('fit', 'all'):
        if not fref.exists():
            P(f"refined-set feature cache missing ({fref}); skipping Part 3")
        else:
            refined = np.load(fref, allow_pickle=False)
            for var in VARIANTS:
                if f'raw_{var}' not in refined:
                    continue
                w, loo, _ = run_fit(refined, casf, args, P, var)
                fits[var] = w

    if args.stage in ('candidates', 'all') and casf is not None and fref.exists():
        run_candidates(casf, np.load(fref, allow_pickle=False), args, P)

    if args.stage in ('casf', 'all') and casf is not None:
        named = [('(a) Vina baseline only', 'v1', np.zeros(dt.N_TERMS)),
                 ('(b) shipped weights, shipped kernel', 'v1', dt.CURRENT_WEIGHTS),
                 ('(c) shipped weights, rewritten kernel', 'v2', dt.CURRENT_WEIGHTS)]
        for var in VARIANTS:
            if var in fits:
                named.append((f'(d) pKd refit on {var}', var, fits[var]))
        run_casf(casf, named, args, P)

    for var, w in fits.items():
        P(f"\n## Copy-pasteable refit for variant {var}"
          f" (Docking/DockingEngine.swift)")
        for i, t in enumerate(dt.TERMS):
            P(f"            {dt.SWIFT_SYMBOL[t]}: {w[i]:.4f},")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text('\n'.join(out) + '\n')
        print(f"\nreport written to {args.report}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Part 1 of the Drusina recalibration: measure the real geometry of the "exotic"
Drusina interactions over the native crystal poses of the PDBbind v2020 refined
set (5316 complexes), and compare the measured distributions against the
distance/angle windows hard-coded in Metal/DockingScoring.metal.

Each interaction class is collected with a DELIBERATELY WIDER window than
Drusina uses, so that the Drusina window can be judged against the observed
distribution rather than against itself. For every directional class an
isotropic null is reported: if the σ-hole / stacking geometry is real, short
contacts are angularly anisotropic and long contacts are not.

Usage:
  python Benchmark/drusina_geometry_survey.py                    # full 5316
  python Benchmark/drusina_geometry_survey.py --max 300          # quick
  python Benchmark/drusina_geometry_survey.py --out survey.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drusina_terms as dt           # noqa: E402
import drusina_data as dd            # noqa: E402

# Wide survey windows (Å) -- all strictly wider than the Drusina windows.
W_HALOGEN = 6.0
W_CHALCOGEN = 6.0
W_RINGRING = 8.0
W_AMIDE = 8.0
W_CHPI = 7.0
W_PICATION = 8.0
W_SALT = 7.0
W_METAL = 4.5

HALOGEN_NAME = {0: 'F', 1: 'Cl', 2: 'Br', 3: 'I'}

# Backbone amide planes: True = the correct peptide plane C(i),O(i),N(i+1);
# False = the shipped DockingEngine.swift triple C(i),O(i),N(i), whose normal
# is wrong because N(i) is two bonds from C(i) via CA.
PEPTIDE_PLANES = True

# Columns of each observation table. Kept as plain float32 matrices so that the
# whole 5316-complex survey fits comfortably in memory and in one .npz.
SCHEMA = {
    # d, theta(C-X...A) deg, element type, acceptor element (6=O,7=N,16=S), cpx
    'halogen':   ['d', 'theta', 'elem', 'acc_elem', 'cpx'],
    # d, theta(C-S...A) deg, acceptor element, cpx
    'chalcogen': ['d', 'theta', 'acc_elem', 'cpx'],
    # d, theta(C-S...A) deg, protein S residue (0=MET,1=CYS), cpx
    'pchalcogen': ['d', 'theta', 'res', 'cpx'],
    # centroid d, interplane angle deg, perpendicular offset, lateral offset, cpx
    'pipi':      ['d', 'ang', 'perp', 'lat', 'cpx'],
    # centroid d, interplane angle deg, perp, lat, amide kind (0=bb,1=sc), cpx
    'amidepi':   ['d', 'ang', 'perp', 'lat', 'kind', 'cpx'],
    # C...centroid d, angle off ring normal deg, perp height, lateral offset, cpx
    'chpi':      ['d', 'ang', 'perp', 'lat', 'cpx'],
    # d, angle off normal deg, mode (0=ligring/protcat, 1=protring/ligcat), cpx
    'picat':     ['d', 'ang', 'mode', 'cpx'],
    # d, charge-sign product, burial factor, charge source (0=formal,1=partial), cpx
    'salt':      ['d', 'sgn', 'burial', 'src', 'cpx'],
    # d, metal (0=Zn,1=Mg,2=Mn,3=Fe,4=Ca,5=other), ligand vina type, cpx
    'metal':     ['d', 'metal', 'ltype', 'cpx'],
}

_METAL_ID = {'ZN': 0, 'MG': 1, 'MN': 2, 'FE': 3, 'CA': 4}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _deg(cos):
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def survey_one(job):
    """Worker: return (pdb_id, {table: ndarray}, per-complex summary dict)."""
    pid, prot_path, lig_path = job
    try:
        mol = dd.load_ligand(lig_path)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        lig_pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        if len(lig_pos) < 3:
            return None
        cen = lig_pos.mean(axis=0)
        recs_all = dt.parse_protein(prot_path)
        if not recs_all:
            return None
        all_pos = np.array([r['pos'] for r in recs_all], dtype=np.float64)
        recs = [r for r in recs_all
                if np.sum((np.array(r['pos']) - cen) ** 2) <= 26.0 ** 2]
        if not recs:
            return None
        pf = dt.build_protein_features(recs, sb_burial_positions=all_pos,
                                       peptide_amide_planes=PEPTIDE_PLANES)
        lt = dt.build_ligand_topology(mol)
    except Exception:
        return None

    obs = {k: [] for k in SCHEMA}
    P = pf.pos
    # protein atoms that could accept a σ-hole contact: all O, all N, all S
    pelem = np.array([r['elem'] for r in recs])
    acc_mask = np.isin(pelem, ['O', 'N', 'S'])
    acc_code = np.where(pelem == 'O', 6, np.where(pelem == 'N', 7, 16))

    # ---------------- halogen bonds ----------------
    if lt.halogens and acc_mask.any():
        AP, AC = P[acc_mask], acc_code[acc_mask]
        for hi, ci, et in lt.halogens:
            v = AP - lig_pos[hi]
            d = np.linalg.norm(v, axis=1)
            m = d <= W_HALOGEN
            if not m.any():
                continue
            xc = _unit(lig_pos[ci] - lig_pos[hi])
            cost = (v[m] / d[m][:, None]) @ xc
            th = _deg(cost)
            obs['halogen'].append(np.column_stack([
                d[m], th, np.full(m.sum(), et), AC[m]]))

    # ---------------- ligand chalcogen bonds ----------------
    if lt.chalcogens and acc_mask.any():
        AP = P[acc_mask]
        AC = acc_code[acc_mask]
        # exclude the sulfur's own acceptor partners? protein only -> no overlap
        for si, ci in lt.chalcogens:
            v = AP - lig_pos[si]
            d = np.linalg.norm(v, axis=1)
            m = d <= W_CHALCOGEN
            if not m.any():
                continue
            sc = _unit(lig_pos[ci] - lig_pos[si])
            th = _deg((v[m] / d[m][:, None]) @ sc)
            obs['chalcogen'].append(np.column_stack([d[m], th, AC[m]]))

    # ---------------- protein chalcogen bonds ----------------
    if len(pf.chal_pos):
        lacc = np.where(np.isin(lt.vtype, [dt.VINA_N_A, dt.VINA_N_DA,
                                           dt.VINA_O_A, dt.VINA_O_DA]))[0]
        if len(lacc):
            LP = lig_pos[lacc]
            # residue tag: direction count 2 => MET, 1 => CYS (approximation of
            # the tag; only used for reporting)
            v = LP[:, None, :] - pf.chal_pos[None, :, :]
            d = np.linalg.norm(v, axis=2)
            m = d <= W_CHALCOGEN
            if m.any():
                u = v / np.maximum(d, 1e-9)[:, :, None]
                cost = np.einsum('jk,ijk->ij', pf.chal_dir, u)
                th = _deg(cost)
                ii, jj = np.nonzero(m)
                obs['pchalcogen'].append(np.column_stack([
                    d[ii, jj], th[ii, jj], np.zeros(len(ii))]))

    # ---------------- ring frames ----------------
    lrc, lrn = dt._ring_frames(lig_pos, lt.rings)

    def _pair_geom(c1, n1, c2, n2, wmax):
        """(d, interplane angle deg, perp offset, lateral offset) for all pairs."""
        v = c2[None, :, :] - c1[:, None, :]
        d = np.linalg.norm(v, axis=2)
        m = d <= wmax
        if not m.any():
            return None
        dotn = np.abs(n1 @ n2.T)
        ang = _deg(dotn)                       # 0 = coplanar/parallel
        u = v / np.maximum(d, 1e-9)[:, :, None]
        perp = np.abs(np.einsum('ik,ijk->ij', n1, u)) * d
        lat = np.sqrt(np.maximum(d ** 2 - perp ** 2, 0.0))
        ii, jj = np.nonzero(m)
        return ii, jj, d[ii, jj], ang[ii, jj], perp[ii, jj], lat[ii, jj]

    # ---------------- pi-pi ----------------
    if len(lrc) and len(pf.ring_centroid):
        g = _pair_geom(lrc, lrn, pf.ring_centroid, pf.ring_normal, W_RINGRING)
        if g:
            _, _, d, ang, perp, lat = g
            obs['pipi'].append(np.column_stack([d, ang, perp, lat]))

    # ---------------- amide-pi ----------------
    if len(lrc) and len(pf.amide_centroid):
        g = _pair_geom(lrc, lrn, pf.amide_centroid, pf.amide_normal, W_AMIDE)
        if g:
            _, jj, d, ang, perp, lat = g
            kind = np.array([1.0 if pf.amide_kind[j] == 'sc' else 0.0 for j in jj])
            obs['amidepi'].append(np.column_stack([d, ang, perp, lat, kind]))

    # ---------------- CH-pi ----------------
    chc = np.where((lt.vtype == dt.VINA_C_H) & (~lt.is_aromatic))[0]
    if len(chc) and len(pf.ring_centroid):
        v = lig_pos[chc][:, None, :] - pf.ring_centroid[None, :, :]
        d = np.linalg.norm(v, axis=2)
        m = d <= W_CHPI
        if m.any():
            u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('jk,ijk->ij', pf.ring_normal, u))
            ang = _deg(cosa)
            perp = cosa * d
            lat = np.sqrt(np.maximum(d ** 2 - perp ** 2, 0.0))
            ii, jj = np.nonzero(m)
            obs['chpi'].append(np.column_stack([
                d[ii, jj], ang[ii, jj], perp[ii, jj], lat[ii, jj]]))

    # ---------------- pi-cation ----------------
    if len(lrc) and len(pf.cation_pos):
        v = pf.cation_pos[None, :, :] - lrc[:, None, :]
        d = np.linalg.norm(v, axis=2)
        m = d <= W_PICATION
        if m.any():
            u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('ik,ijk->ij', lrn, u))
            ii, jj = np.nonzero(m)
            obs['picat'].append(np.column_stack([
                d[ii, jj], _deg(cosa[ii, jj]), np.zeros(len(ii))]))
    lig_cat = np.where(lt.formal_charge > 0)[0]
    if len(lig_cat) and len(pf.ring_centroid):
        v = lig_pos[lig_cat][:, None, :] - pf.ring_centroid[None, :, :]
        d = np.linalg.norm(v, axis=2)
        m = d <= W_PICATION
        if m.any():
            u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('jk,ijk->ij', pf.ring_normal, u))
            ii, jj = np.nonzero(m)
            obs['picat'].append(np.column_stack([
                d[ii, jj], _deg(cosa[ii, jj]), np.ones(len(ii))]))

    # ---------------- salt bridges ----------------
    if len(pf.sb_centroid):
        lc = np.zeros(lt.n_heavy)
        src = np.zeros(lt.n_heavy)
        nz = lt.formal_charge != 0
        lc[nz] = np.sign(lt.formal_charge[nz])
        fb = ~nz & (np.abs(lt.charge) > 0.35)
        lc[fb] = np.sign(lt.charge[fb])
        src[fb] = 1.0
        act = np.where(lc != 0)[0]
        if len(act):
            d = np.linalg.norm(lig_pos[act][:, None, :] - pf.sb_centroid[None, :, :], axis=2)
            m = d <= W_SALT
            if m.any():
                ii, jj = np.nonzero(m)
                sgn = lc[act][ii] * pf.sb_sign[jj]
                obs['salt'].append(np.column_stack([
                    d[ii, jj], sgn, pf.sb_burial[jj], src[act][ii]]))

    # ---------------- metal coordination ----------------
    met = np.where(pf.is_metal)[0]
    if len(met):
        lm = np.where(np.isin(lt.vtype, list(dt.METAL_COORD_LIG)))[0]
        if len(lm):
            d = np.linalg.norm(lig_pos[lm][:, None, :] - P[met][None, :, :], axis=2)
            m = d <= W_METAL
            if m.any():
                ii, jj = np.nonzero(m)
                mid = np.array([_METAL_ID.get(recs[met[j]]['elem'], 5) for j in jj])
                obs['metal'].append(np.column_stack([
                    d[ii, jj], mid, lt.vtype[lm][ii]]))

    # ---------------- per-complex summary (raw Drusina terms, native pose) -----
    try:
        raw, icount, counts = dt.raw_terms(pf, lt, lig_pos, want_counts=True)
    except Exception:
        return None

    summary = {
        'pdb_id': pid,
        'raw': raw,
        'counts': np.array([counts[t] for t in dt.TERMS], dtype=np.int32),
        'n_prot_rings': len(pf.ring_centroid),
        'n_amides': len(pf.amide_centroid),
        'n_amides_bb': sum(1 for k in pf.amide_kind if k == 'bb'),
        'n_sb_groups': len(pf.sb_centroid),
        'n_cations': len(pf.cation_pos),
        'n_prot_chal': len(pf.chal_pos),
        'n_lig_halogens': len(lt.halogens),
        'n_lig_halogens_heavy': sum(1 for h in lt.halogens if h[2] != 0),
        'n_lig_chalcogens': len(lt.chalcogens),
        'n_lig_rings': len(lt.rings),
        'n_lig_heavy': lt.n_heavy,
        'n_metals': int(pf.is_metal.sum()),
        'lig_radius': float(np.max(np.linalg.norm(lig_pos - cen, axis=1))),
    }

    out = {}
    for k, v in obs.items():
        if v:
            a = np.vstack(v).astype(np.float32)
            out[k] = a
    return pid, out, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ISOTROPIC_LINEAR = 0.5 * (1.0 - np.cos(np.radians(180.0 - 150.0)))  # P(theta>150)


def q(a, *ps):
    if len(a) == 0:
        return [float('nan')] * len(ps)
    return [float(np.percentile(a, p)) for p in ps]


def fmt(vals, w=7, p=2):
    return ' '.join(f'{v:{w}.{p}f}' for v in vals)


def report(tables, summaries, n_cpx, fh=sys.stdout):
    def P(*a, **k):
        print(*a, **k, file=fh)

    cpxid = {}

    P(f"\n{'='*100}")
    P(f"DRUSINA GEOMETRY SURVEY -- {n_cpx} PDBbind v2020 refined-set crystal complexes")
    P(f"{'='*100}")

    # ---- per-complex occurrence of each feature class
    P("\n## Feature availability (per complex)")
    keys = ['n_lig_halogens_heavy', 'n_lig_chalcogens', 'n_lig_rings',
            'n_prot_rings', 'n_amides', 'n_amides_bb', 'n_sb_groups',
            'n_cations', 'n_prot_chal', 'n_metals', 'n_lig_heavy', 'lig_radius']
    P(f"{'quantity':<24}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}{'frac>0':>9}")
    for k in keys:
        v = np.array([s[k] for s in summaries], dtype=np.float64)
        P(f"{k:<24}{v.mean():>9.2f}{np.median(v):>9.2f}"
          f"{np.percentile(v,90):>9.2f}{v.max():>9.2f}{np.mean(v>0):>9.3f}")

    # ---- Drusina term firing on native poses
    P("\n## Drusina raw terms on NATIVE crystal poses (unweighted)")
    raws = np.array([s['raw'] for s in summaries])
    cnts = np.array([s['counts'] for s in summaries])
    P(f"{'term':<22}{'fires%':>8}{'mean':>10}{'mean|fire':>11}"
      f"{'median|fire':>12}{'p95':>9}{'n_contacts':>11}")
    for i, t in enumerate(dt.TERMS):
        v = raws[:, i]
        nz = v != 0
        mf = v[nz].mean() if nz.any() else 0.0
        md = np.median(v[nz]) if nz.any() else 0.0
        P(f"{t:<22}{100*nz.mean():>7.1f}%{v.mean():>10.4f}{mf:>11.4f}"
          f"{md:>12.4f}{np.percentile(v,95):>9.4f}{cnts[:,i].mean():>11.2f}")

    # ---- halogen bonds
    H = tables.get('halogen')
    P("\n## Halogen bonds  C-X...A   (Drusina window: 2.5-4.2 A, theta>140 deg, F skipped)")
    if H is None or len(H) == 0:
        P("  no observations")
    else:
        P(f"  {len(H):,} C-X...(O/N/S) contacts within {W_HALOGEN} A over the whole set")
        P(f"\n  {'X':<4}{'contacts':>10}{'theta>150 @ d<4.0':>19}{'theta>150 @ d>5.0':>19}"
          f"{'(isotropic null)':>18}")
        for et in (0, 1, 2, 3):
            s = H[H[:, 2] == et]
            if len(s) == 0:
                continue
            near = s[s[:, 0] < 4.0]
            far = s[s[:, 0] > 5.0]
            fn = np.mean(near[:, 1] > 150) if len(near) else float('nan')
            ff = np.mean(far[:, 1] > 150) if len(far) else float('nan')
            P(f"  {HALOGEN_NAME[et]:<4}{len(s):>10,}{fn:>19.4f}{ff:>19.4f}"
              f"{ISOTROPIC_LINEAR:>18.4f}")
        P(f"\n  Geometry of the ANISOTROPIC subset (theta > 150 deg), by halogen:")
        P(f"  {'X':<4}{'n':>8}{'d p10':>8}{'d p25':>8}{'d med':>8}{'d p75':>8}"
          f"{'d p90':>8}{'theta med':>11}{'Drusina opt':>13}")
        OPT = {0: float('nan'), 1: 3.27, 2: 3.24, 3: 3.17}
        for et in (0, 1, 2, 3):
            s = H[(H[:, 2] == et) & (H[:, 1] > 150) & (H[:, 0] < 4.5)]
            if len(s) < 5:
                P(f"  {HALOGEN_NAME[et]:<4}{len(s):>8}  (too few)")
                continue
            P(f"  {HALOGEN_NAME[et]:<4}{len(s):>8}"
              f"{fmt(q(s[:,0],10,25,50,75,90),8,2)}{np.median(s[:,1]):>11.1f}"
              f"{OPT[et]:>13.2f}")
        P(f"\n  Optimum from the ENRICHED window only (theta>150 and d<3.6,"
          f" where the angular signal is still above the isotropic null):")
        P(f"  {'X':<4}{'n':>8}{'d p25':>8}{'d med':>8}{'d p75':>8}{'d mean':>9}"
          f"{'Drusina opt':>13}")
        for et in (0, 1, 2, 3):
            s2 = H[(H[:, 2] == et) & (H[:, 1] > 150) & (H[:, 0] < 3.6)]
            if len(s2) < 5:
                P(f"  {HALOGEN_NAME[et]:<4}{len(s2):>8}  (too few)")
                continue
            P(f"  {HALOGEN_NAME[et]:<4}{len(s2):>8}"
              f"{fmt(q(s2[:,0],25,50,75),8,2)}{s2[:,0].mean():>9.2f}"
              f"{OPT[et]:>13.2f}")
        P(f"\n  Acceptor element share of theta>150, d<4.0 contacts:")
        s = H[(H[:, 1] > 150) & (H[:, 0] < 4.0) & (H[:, 2] > 0)]
        if len(s):
            for code, nm in ((6, 'O'), (7, 'N'), (16, 'S')):
                k = np.sum(s[:, 3] == code)
                P(f"    {nm}: {k:>7,} ({100.0*k/len(s):5.1f}%)")
        P(f"\n  Angular profile per distance shell (heavy halogens Cl/Br/I):")
        s = H[H[:, 2] > 0]
        P(f"  {'shell (A)':<12}{'n':>9}{'theta med':>11}{'frac>140':>10}"
          f"{'frac>150':>10}{'frac>160':>10}")
        for lo, hi in ((2.5, 3.0), (3.0, 3.3), (3.3, 3.6), (3.6, 4.0),
                       (4.0, 4.2), (4.2, 4.6), (4.6, 5.0), (5.0, 6.0)):
            k = s[(s[:, 0] >= lo) & (s[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>9,}{np.median(k[:,1]):>11.1f}"
              f"{np.mean(k[:,1]>140):>10.3f}{np.mean(k[:,1]>150):>10.3f}"
              f"{np.mean(k[:,1]>160):>10.3f}")

    # ---- chalcogen
    for key, label, drus in (
        ('chalcogen', 'Ligand chalcogen bonds  C-S...A', '2.8-4.2 A, theta>140, opt 3.3'),
        ('pchalcogen', 'Protein chalcogen bonds  C-S...A(lig)', '2.8-4.2 A, theta>140, opt 3.3')):
        C = tables.get(key)
        P(f"\n## {label}   (Drusina window: {drus})")
        if C is None or len(C) == 0:
            P("  no observations")
            continue
        P(f"  {len(C):,} contacts within {W_CHALCOGEN} A")
        P(f"  {'shell (A)':<12}{'n':>9}{'theta med':>11}{'frac>140':>10}"
          f"{'frac>150':>10}{'frac>160':>10}{'isotropic>140':>15}")
        iso140 = 0.5 * (1.0 - np.cos(np.radians(40.0)))
        for lo, hi in ((2.8, 3.1), (3.1, 3.4), (3.4, 3.7), (3.7, 4.0),
                       (4.0, 4.2), (4.2, 4.6), (4.6, 5.0), (5.0, 6.0)):
            k = C[(C[:, 0] >= lo) & (C[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>9,}{np.median(k[:,1]):>11.1f}"
              f"{np.mean(k[:,1]>140):>10.3f}{np.mean(k[:,1]>150):>10.3f}"
              f"{np.mean(k[:,1]>160):>10.3f}{iso140:>15.3f}")
        s = C[(C[:, 1] > 140) & (C[:, 0] < 4.5)]
        if len(s) > 5:
            P(f"  theta>140 & d<4.5:  n={len(s):,}  d p10/25/50/75/90 ="
              f" {fmt(q(s[:,0],10,25,50,75,90),6,2)}")
        s = C[(C[:, 1] > 150) & (C[:, 0] < 3.6)]
        if len(s) > 5:
            P(f"  ENRICHED window only (theta>150 & d<3.6): n={len(s):,}"
              f"  d p25/med/p75 = {fmt(q(s[:,0],25,50,75),6,2)}"
              f"  mean {s[:,0].mean():.2f}")

    # ---- pi-pi
    for key, label, note in (
        ('pipi', 'pi-pi stacking (ligand aromatic ring / protein aromatic ring)',
         'Drusina: 3.2-5.8 A; face |cos|>0.8 opt 3.5; edge |cos|<0.4 opt 4.8'),
        ('amidepi', 'amide-pi (protein amide plane / ligand aromatic ring)',
         'Drusina: 3.0-5.2 A; |cos|>0.8 opt 3.6; 0.6<|cos|<0.8 opt 4.0')):
        T = tables.get(key)
        P(f"\n## {label}\n   {note}")
        if T is None or len(T) == 0:
            P("  no observations")
            continue
        P(f"  {len(T):,} ring-pair contacts within {W_RINGRING if key=='pipi' else W_AMIDE} A")
        P(f"  {'shell (A)':<12}{'n':>9}{'ang med':>9}{'par<25deg':>11}"
          f"{'perp>65deg':>11}{'perp-off med':>14}{'lat-off med':>13}")
        for lo, hi in ((3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0),
                       (5.0, 5.5), (5.5, 6.0), (6.0, 7.0), (7.0, 8.0)):
            k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>9,}{np.median(k[:,1]):>9.1f}"
              f"{np.mean(k[:,1]<25):>11.3f}{np.mean(k[:,1]>65):>11.3f}"
              f"{np.median(k[:,2]):>14.2f}{np.median(k[:,3]):>13.2f}")
        # parallel subset: the one Drusina rewards most
        par = T[T[:, 1] < 25]
        if len(par) > 10:
            P(f"  PARALLEL subset (interplane < 25 deg): n={len(par):,}"
              f"  d p10/25/50/75/90 = {fmt(q(par[:,0],10,25,50,75,90),6,2)}")
            pp = par[par[:, 0] < 5.0]
            if len(pp):
                P(f"    of those with d<5: perp offset med {np.median(pp[:,2]):.2f},"
                  f" lateral offset med {np.median(pp[:,3]):.2f}")
        perp = T[T[:, 1] > 65]
        if len(perp) > 10:
            P(f"  PERPENDICULAR subset (> 65 deg): n={len(perp):,}"
              f"  d p10/25/50/75/90 = {fmt(q(perp[:,0],10,25,50,75,90),6,2)}")
        # Isotropic nulls for a uniformly random relative ring orientation:
        # |cos| is uniform on [0,1], so P(ang<25) = 1-cos(25) complement form
        # P(|cos| > cos25) = 1 - cos(25 deg) = 0.0937, P(|cos| < cos65) = 0.4226.
        P(f"  ENRICHMENT over a random relative orientation "
          f"(null: parallel<25deg = 0.094, perp>65deg = 0.577):")
        P(f"  {'shell (A)':<12}{'par enrich':>12}{'perp enrich':>13}"
          f"{'offset<=2 frac':>16}{'n(off<=2,par)':>15}")
        for lo, hi in ((3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0),
                       (5.0, 5.5), (5.5, 6.0), (6.0, 7.0), (7.0, 8.0)):
            k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            if len(k) == 0:
                continue
            off2 = k[k[:, 3] <= 2.0]
            both = off2[off2[:, 1] < 25]
            P(f"  {lo:.1f}-{hi:.1f}      {np.mean(k[:,1]<25)/0.0937:>12.2f}"
              f"{np.mean(k[:,1]>65)/0.5774:>13.2f}{len(off2)/len(k):>16.3f}"
              f"{len(both):>15,}")
        if key == 'amidepi':
            bb = T[T[:, 4] == 0]
            sc = T[T[:, 4] == 1]
            P(f"  backbone amides: {len(bb):,} contacts;  sidechain (Asn/Gln): {len(sc):,}")
            for nm, s in (('backbone', bb), ('sidechain', sc)):
                w = s[(s[:, 0] >= 3.0) & (s[:, 0] <= 5.2) & (s[:, 1] < 37)]
                P(f"    {nm:<10} inside the Drusina scoring window: {len(w):,}"
                  f" ({100.0*len(w)/max(len(T),1):.2f}% of all amide-pi contacts)")

    # ---- CH-pi
    T = tables.get('chpi')
    P("\n## CH-pi (ligand aliphatic C / protein aromatic ring centroid)")
    P("   Drusina: 3.5-5.0 A centroid distance, angle off normal < 56.6 deg, opt 4.0")
    if T is None or len(T) == 0:
        P("  no observations")
    else:
        P(f"  {len(T):,} C...ring contacts within {W_CHPI} A")
        P(f"  {'shell (A)':<12}{'n':>10}{'ang med':>9}{'frac<56deg':>12}"
          f"{'perp med':>10}{'lat med':>9}")
        for lo, hi in ((3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0),
                       (5.0, 5.5), (5.5, 6.0), (6.0, 7.0)):
            k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>10,}{np.median(k[:,1]):>9.1f}"
              f"{np.mean(k[:,1]<56.6):>12.3f}{np.median(k[:,2]):>10.2f}"
              f"{np.median(k[:,3]):>9.2f}")
        # real CH-pi geometry: H sits over the ring -> perp height 2.6-3.1 for C
        g = T[(T[:, 1] < 30) & (T[:, 0] < 5.0)]
        P(f"  near-axial subset (angle<30 deg, d<5): n={len(g):,}"
          f"  d p10/25/50/75/90 = {fmt(q(g[:,0],10,25,50,75,90),6,2)}" if len(g)
          else "  near-axial subset empty")
        w = T[(T[:, 0] >= 3.5) & (T[:, 0] <= 5.0) & (T[:, 1] < 56.6)]
        P(f"  inside the Drusina window: {len(w):,} contacts"
          f" ({100.0*len(w)/len(T):.1f}% of all C...ring contacts <= {W_CHPI} A)")
        P(f"  ENRICHMENT of axial approach over random (null: frac<56.6deg = 0.450,"
          f" frac<45deg = 0.293):")
        P(f"  {'shell (A)':<12}{'n':>10}{'enr<56':>9}{'enr<45':>9}"
          f"{'offset<=2':>11}{'perp med|off<=2':>17}")
        for lo, hi in ((3.0, 3.4), (3.4, 3.8), (3.8, 4.2), (4.2, 4.5),
                       (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 7.0)):
            k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            if len(k) == 0:
                continue
            o2 = k[k[:, 3] <= 2.0]
            pm = np.median(o2[:, 2]) if len(o2) else float('nan')
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>10,}"
              f"{np.mean(k[:,1]<56.6)/0.450:>9.2f}{np.mean(k[:,1]<45)/0.293:>9.2f}"
              f"{len(o2)/len(k):>11.3f}{pm:>17.2f}")
        g2 = T[(T[:, 3] <= 2.0) & (T[:, 0] < 5.5)]
        if len(g2):
            P(f"  TRUE CH-pi geometry (lateral offset <= 2 A, d < 5.5): n={len(g2):,}"
              f"  d p10/25/50/75/90 = {fmt(q(g2[:,0],10,25,50,75,90),6,2)}")
            P(f"     perpendicular height p10/25/50/75/90 ="
              f" {fmt(q(g2[:,2],10,25,50,75,90),6,2)}")

    # ---- pi-cation
    T = tables.get('picat')
    P("\n## pi-cation   (Drusina: d <= 5.5 A, |cos| > 0.5, opt 3.8)")
    if T is None or len(T) == 0:
        P("  no observations")
    else:
        for mode, nm in ((0, 'ligand ring / protein cation'),
                         (1, 'protein ring / ligand formal cation')):
            s = T[T[:, 2] == mode]
            if len(s) == 0:
                P(f"  {nm}: none")
                continue
            w = s[(s[:, 0] <= 5.5) & (s[:, 1] < 60)]
            P(f"  {nm}: {len(s):,} contacts <= {W_PICATION} A, {len(w):,} inside window")
            if len(w) > 5:
                P(f"     in-window d p10/25/50/75/90 = {fmt(q(w[:,0],10,25,50,75,90),6,2)}"
                  f"  angle med {np.median(w[:,1]):.1f} deg")
            P(f"     angle median by shell:")
            for lo, hi in ((3.0, 4.0), (4.0, 5.0), (5.0, 5.5), (5.5, 6.5), (6.5, 8.0)):
                k = s[(s[:, 0] >= lo) & (s[:, 0] < hi)]
                if len(k) == 0:
                    continue
                P(f"       {lo:.1f}-{hi:.1f}  n={len(k):>7,}  ang med {np.median(k[:,1]):>5.1f}"
                  f"  frac<60deg {np.mean(k[:,1]<60):.3f}")

    # ---- salt bridges
    T = tables.get('salt')
    P("\n## Salt bridges (ligand charged atom / protein charged group centroid)")
    P("   Drusina: d <= 4.0 A, ramp centred 2.8 A (full 2.5-3.1), burial weighted")
    if T is None or len(T) == 0:
        P("  no observations")
    else:
        opp = T[T[:, 1] < 0]
        like = T[T[:, 1] > 0]
        P(f"  {len(T):,} contacts <= {W_SALT} A: {len(opp):,} opposite-charge,"
          f" {len(like):,} like-charge")
        P(f"  {'shell (A)':<12}{'opposite':>10}{'like':>10}{'opp/like':>10}")
        for lo, hi in ((2.0, 2.6), (2.6, 3.0), (3.0, 3.4), (3.4, 4.0),
                       (4.0, 5.0), (5.0, 6.0), (6.0, 7.0)):
            a = np.sum((opp[:, 0] >= lo) & (opp[:, 0] < hi))
            b = np.sum((like[:, 0] >= lo) & (like[:, 0] < hi))
            if a + b == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {a:>10,}{b:>10,}"
              f"{(a/b if b else float('inf')):>10.2f}")
        w = opp[opp[:, 0] <= 4.0]
        if len(w):
            P(f"  in-window opposite-charge contacts: {len(w):,},"
              f" d p10/25/50/75/90 = {fmt(q(w[:,0],10,25,50,75,90),6,2)}")
            P(f"  burial factor of those groups: p10/25/50/75/90 ="
              f" {fmt(q(w[:,2],10,25,50,75,90),6,2)}")
        P("\n  Split by how the ligand charge sign was assigned. A genuine salt"
          "\n  bridge must show opposite-charge pairs ENRICHED at 2.6-3.4 A; a"
          "\n  ratio near or below 1 means the charge labelling is noise.")
        for code, nm in ((0.0, 'FORMAL charge'), (1.0, 'Gasteiger |q|>0.35 fallback')):
            sub = T[T[:, 3] == code]
            if len(sub) == 0:
                P(f"    {nm}: none")
                continue
            o = sub[sub[:, 1] < 0]
            l = sub[sub[:, 1] > 0]
            P(f"    {nm}: {len(sub):,} contacts ({len(o):,} opp / {len(l):,} like)")
            P(f"      {'shell (A)':<12}{'opposite':>10}{'like':>10}{'opp/like':>10}")
            for lo, hi in ((2.0, 2.6), (2.6, 3.0), (3.0, 3.4), (3.4, 4.0),
                           (4.0, 5.0), (5.0, 6.0), (6.0, 7.0)):
                a = int(np.sum((o[:, 0] >= lo) & (o[:, 0] < hi)))
                b = int(np.sum((l[:, 0] >= lo) & (l[:, 0] < hi)))
                if a + b == 0:
                    continue
                P(f"      {lo:.1f}-{hi:.1f}      {a:>10,}{b:>10,}"
                  f"{(a/b if b else float('inf')):>10.2f}")

    # ---- metal
    T = tables.get('metal')
    P("\n## Metal coordination (protein metal / ligand N,O,S)")
    P("   Drusina: d <= 4.5 A, full score 2.0-2.8, zero at 4.0, opt 2.4")
    if T is None or len(T) == 0:
        P("  no observations")
    else:
        P(f"  {len(T):,} metal...(N/O/S) contacts <= {W_METAL} A")
        P(f"  {'shell (A)':<12}{'n':>9}")
        for lo, hi in ((1.6, 2.0), (2.0, 2.2), (2.2, 2.4), (2.4, 2.8),
                       (2.8, 3.2), (3.2, 3.6), (3.6, 4.0), (4.0, 4.5)):
            k = np.sum((T[:, 0] >= lo) & (T[:, 0] < hi))
            if k:
                P(f"  {lo:.1f}-{hi:.1f}      {k:>9,}")
        inner = T[T[:, 0] <= 2.8]
        P(f"  first shell (<= 2.8 A): n={len(inner):,}"
          f"  d p10/25/50/75/90 = {fmt(q(inner[:,0],10,25,50,75,90),6,2)}"
          if len(inner) else "  first shell empty")
        names = {0: 'Zn', 1: 'Mg', 2: 'Mn', 3: 'Fe', 4: 'Ca', 5: 'other'}
        for mid, nm in names.items():
            s = T[(T[:, 1] == mid) & (T[:, 0] <= 3.0)]
            if len(s) < 5:
                continue
            P(f"    {nm:<6} n={len(s):>6,}  d med {np.median(s[:,0]):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max', type=int, default=0, help='limit complexes (0 = all)')
    ap.add_argument('--seed', type=int, default=0, help='random sample seed when --max is set')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--out', type=str, default='', help='write observations to .npz')
    ap.add_argument('--text-out', type=str, default='', help='write the text report here too')
    ap.add_argument('--shipped-amide-planes', action='store_true',
                    help='reproduce the C(i),O(i),N(i) backbone plane bug')
    args = ap.parse_args()
    global PEPTIDE_PLANES
    PEPTIDE_PLANES = not args.shipped_amide_planes

    jobs = dd.refined_complexes()
    if args.max and args.max < len(jobs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(jobs), args.max, replace=False)
        jobs = [jobs[i] for i in sorted(idx)]
    nw = args.workers or min(mp.cpu_count(), 8)
    print(f"surveying {len(jobs)} complexes with {nw} workers", flush=True)

    tables = defaultdict(list)
    summaries = []
    t0 = time.time()
    failed = 0
    with mp.Pool(nw) as pool:
        for n, res in enumerate(pool.imap_unordered(survey_one, jobs, chunksize=8), 1):
            if res is None:
                failed += 1
            else:
                pid, obs, summary = res
                cpx = len(summaries)
                for k, a in obs.items():
                    tables[k].append(np.column_stack([a, np.full(len(a), cpx, np.float32)]))
                summaries.append(summary)
            if n % 500 == 0:
                el = time.time() - t0
                print(f"  [{n}/{len(jobs)}] {n/el:.1f} cpx/s  ok={len(summaries)}"
                      f" failed={failed}  eta {(len(jobs)-n)/(n/el):.0f}s", flush=True)

    merged = {k: np.vstack(v) for k, v in tables.items() if v}
    print(f"\ndone: {len(summaries)} complexes, {failed} failed,"
          f" {time.time()-t0:.0f}s", flush=True)
    for k, v in merged.items():
        print(f"  {k:<12} {len(v):>10,} observations")

    if args.out:
        out = {f'obs_{k}': v for k, v in merged.items()}
        out['summary_ids'] = np.array([s['pdb_id'] for s in summaries])
        out['summary_raw'] = np.array([s['raw'] for s in summaries])
        out['summary_counts'] = np.array([s['counts'] for s in summaries])
        for k in ('n_prot_rings', 'n_amides', 'n_amides_bb', 'n_sb_groups',
                  'n_cations', 'n_prot_chal', 'n_lig_halogens',
                  'n_lig_halogens_heavy', 'n_lig_chalcogens', 'n_lig_rings',
                  'n_lig_heavy', 'n_metals', 'lig_radius'):
            out[f'summary_{k}'] = np.array([s[k] for s in summaries])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **out)
        print(f"  wrote {args.out}")

    report(merged, summaries, len(summaries))
    if args.text_out:
        Path(args.text_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.text_out, 'w') as fh:
            report(merged, summaries, len(summaries), fh=fh)
        print(f"\nreport written to {args.text_out}")


if __name__ == '__main__':
    main()

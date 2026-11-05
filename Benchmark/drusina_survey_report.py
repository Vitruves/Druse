#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Post-process the observation tables written by drusina_geometry_survey.py
(--out survey.npz) without re-running the 5316-complex measurement.

Adds the analyses that turn raw distributions into cutoff recommendations:
  * enrichment of each directional geometry over an isotropic null, per shell,
    which is what says where a term stops carrying signal;
  * lateral-offset-resolved stacking and CH-pi geometry, because a centroid
    distance alone cannot distinguish a stack from a side-by-side contact;
  * the fraction of each Drusina window that is occupied by contacts the
    measured distribution says are not real interactions.

Usage:
  python Benchmark/drusina_survey_report.py survey.npz [--out report.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drusina_terms as dt           # noqa: E402

# Isotropic nulls.
# A uniformly random direction has cos(theta) uniform on [-1, 1], so
#   P(theta > T)      = (1 - cos(180 - T)) / 2
# A uniformly random relative plane orientation has |cos| uniform on [0, 1], so
#   P(interplane < T) = 1 - cos(T)
#   P(interplane > T) = cos(T)
# And for an approach direction measured against a fixed normal,
#   P(angle < T)      = 1 - cos(T)   (|cos| uniform)
def null_theta_gt(t_deg):
    return 0.5 * (1.0 - np.cos(np.radians(180.0 - t_deg)))


def null_angle_lt(t_deg):
    return 1.0 - np.cos(np.radians(t_deg))


def _hist_line(a, lo, hi, w):
    edges = np.arange(lo, hi + 1e-9, w)
    h, _ = np.histogram(a, bins=edges)
    return ' '.join(f'{e:.1f}:{c}' for e, c in zip(edges[:-1], h))


def _mode_bin(a, w=0.2):
    if len(a) == 0:
        return float('nan')
    lo = np.floor(a.min() / w) * w
    edges = np.arange(lo, a.max() + w, w)
    h, _ = np.histogram(a, bins=edges)
    i = int(np.argmax(h))
    return float(edges[i] + w / 2)


def q(a, *ps):
    if len(a) == 0:
        return [float('nan')] * len(ps)
    return [float(np.percentile(a, p)) for p in ps]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    lines = []

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        lines.append(s)

    n_cpx = len(z['summary_ids'])
    P(f"\n{'='*104}")
    P(f"DRUSINA GEOMETRY SURVEY -- ENRICHMENT ANALYSIS ({n_cpx} refined-set complexes)")
    P(f"{'='*104}")
    P("Enrichment = observed fraction / fraction expected for a random orientation.")
    P("A value of 1.00 means the geometry carries NO information at that distance:")
    P("contacts there are packing, not interaction, and a scoring term that fires")
    P("there adds noise proportional to local atom density.")

    # ---------------- halogen ----------------
    H = z['obs_halogen']
    P(f"\n## Halogen bond sigma-hole enrichment (null for theta>150 deg: "
      f"{null_theta_gt(150):.4f})")
    P(f"{'shell (A)':<12}{'n(Cl/Br/I)':>12}{'f>140':>8}{'f>150':>8}{'f>160':>8}"
      f"{'enr>150':>9}{'enr>160':>9}{'n(F)':>8}{'enr>150 F':>11}")
    heavy = H[H[:, 2] > 0]
    fl = H[H[:, 2] == 0]
    for lo, hi in ((2.6, 2.9), (2.9, 3.1), (3.1, 3.3), (3.3, 3.5), (3.5, 3.7),
                   (3.7, 4.0), (4.0, 4.3), (4.3, 4.7), (4.7, 5.2), (5.2, 6.0)):
        k = heavy[(heavy[:, 0] >= lo) & (heavy[:, 0] < hi)]
        f = fl[(fl[:, 0] >= lo) & (fl[:, 0] < hi)]
        if len(k) == 0:
            continue
        P(f"{lo:.1f}-{hi:.1f}      {len(k):>12,}{np.mean(k[:,1]>140):>8.3f}"
          f"{np.mean(k[:,1]>150):>8.3f}{np.mean(k[:,1]>160):>8.3f}"
          f"{np.mean(k[:,1]>150)/null_theta_gt(150):>9.2f}"
          f"{np.mean(k[:,1]>160)/null_theta_gt(160):>9.2f}"
          f"{len(f):>8,}"
          f"{(np.mean(f[:,1]>150)/null_theta_gt(150) if len(f) else float('nan')):>11.2f}")
    sig = heavy[(heavy[:, 1] > 150) & (heavy[:, 0] <= 3.6)]
    P(f"\n  Signal-bearing halogen bonds (Cl/Br/I, theta>150, d<=3.6): n={len(sig):,}")
    if len(sig):
        P(f"    d p10/25/50/75/90 = {' '.join(f'{v:.2f}' for v in q(sig[:,0],10,25,50,75,90))}"
          f"   mean {sig[:,0].mean():.2f}")
        for et, nm, opt in ((1, 'Cl', 3.27), (2, 'Br', 3.24), (3, 'I', 3.17)):
            s = sig[sig[:, 2] == et]
            if len(s) >= 5:
                P(f"    {nm}: n={len(s):>4}  median {np.median(s[:,0]):.2f}"
                  f"  mean {s[:,0].mean():.2f}  (Drusina opt {opt:.2f})")
    # how much of the Drusina window is noise?
    win = heavy[(heavy[:, 0] >= 2.5) & (heavy[:, 0] <= 4.0) & (heavy[:, 1] > 140)]
    good = heavy[(heavy[:, 0] >= 2.5) & (heavy[:, 0] <= 3.6) & (heavy[:, 1] > 140)]
    P(f"  contacts scored by the current window (2.5-4.0, theta>140): {len(win):,}")
    P(f"  of those inside the enriched region (d<=3.6):               {len(good):,}"
      f"  ({100.0*len(good)/max(len(win),1):.0f}%)")

    # ---------------- chalcogen ----------------
    for key, nm in (('obs_chalcogen', 'Ligand chalcogen (C-S...A)'),
                    ('obs_pchalcogen', 'Protein chalcogen (C-S...A on ligand)')):
        C = z[key]
        P(f"\n## {nm} sigma-hole enrichment "
          f"(null for theta>140: {null_theta_gt(140):.4f})")
        P(f"{'shell (A)':<12}{'n':>10}{'f>140':>8}{'f>150':>8}"
          f"{'enr>140':>9}{'enr>150':>9}")
        for lo, hi in ((2.8, 3.1), (3.1, 3.3), (3.3, 3.5), (3.5, 3.7),
                       (3.7, 4.0), (4.0, 4.3), (4.3, 4.7), (4.7, 5.2), (5.2, 6.0)):
            k = C[(C[:, 0] >= lo) & (C[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"{lo:.1f}-{hi:.1f}      {len(k):>10,}{np.mean(k[:,1]>140):>8.3f}"
              f"{np.mean(k[:,1]>150):>8.3f}"
              f"{np.mean(k[:,1]>140)/null_theta_gt(140):>9.2f}"
              f"{np.mean(k[:,1]>150)/null_theta_gt(150):>9.2f}")
        s = C[(C[:, 1] > 140) & (C[:, 0] <= 3.5)]
        if len(s) >= 5:
            P(f"  signal-bearing subset (theta>140, d<=3.5): n={len(s):,}"
              f"  d p25/50/75 = {' '.join(f'{v:.2f}' for v in q(s[:,0],25,50,75))}")

    # ---------------- ring stacking ----------------
    for key, nm, cur in (
        ('obs_pipi', 'pi-pi (ligand ring / protein ring)',
         'current: face |cos|>0.8 & 3.2-4.5 (offset<=2), edge |cos|<0.4 & up to 5.8'),
        ('obs_amidepi', 'amide-pi (protein amide / ligand ring)',
         'current: |cos|>0.8 & 3.2-4.5 (offset<=2)')):
        T = z[key]
        P(f"\n## {nm}\n   {cur}")
        P(f"   nulls: parallel<25deg {null_angle_lt(25):.4f}, "
          f"parallel<37deg {null_angle_lt(37):.4f}, perp>66deg "
          f"{np.cos(np.radians(66)):.4f}")
        P(f"{'shell (A)':<12}{'n':>9}{'f par<37':>10}{'enr par':>9}"
          f"{'f perp>66':>11}{'enr perp':>10}{'f off<=2':>10}"
          f"{'perp-h med':>12}")
        for lo, hi in ((3.0, 3.4), (3.4, 3.7), (3.7, 4.0), (4.0, 4.3),
                       (4.3, 4.6), (4.6, 5.0), (5.0, 5.5), (5.5, 6.0),
                       (6.0, 7.0), (7.0, 8.0)):
            k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            if len(k) == 0:
                continue
            fp = np.mean(k[:, 1] < 37)
            fq = np.mean(k[:, 1] > 66)
            o2 = k[k[:, 3] <= 2.0]
            P(f"{lo:.1f}-{hi:.1f}      {len(k):>9,}{fp:>10.3f}"
              f"{fp/null_angle_lt(37):>9.2f}{fq:>11.3f}"
              f"{fq/np.cos(np.radians(66)):>10.2f}{len(o2)/len(k):>10.3f}"
              f"{(np.median(o2[:,2]) if len(o2) else float('nan')):>12.2f}")
        st = T[(T[:, 1] < 37) & (T[:, 3] <= 2.0)]
        P(f"  TRUE STACKS (interplane<37 deg AND lateral offset<=2 A): n={len(st):,}"
          f" of {len(T):,} contacts")
        if len(st):
            P(f"    centroid d p10/25/50/75/90 ="
              f" {' '.join(f'{v:.2f}' for v in q(st[:,0],10,25,50,75,90))}")
            P(f"    of those, {int(np.sum(st[:,0]<=4.5)):,} are within 4.5 A"
              f" ({100.0*np.mean(st[:,0]<=4.5):.0f}%)")
        # Optimum: only the subset that is both angularly and laterally a stack
        # AND inside the distance range where the geometry is still enriched.
        inr = st[st[:, 0] <= 4.8]
        if len(inr) > 20:
            P(f"    RECOMMENDED OPTIMUM from the enriched stacks (d<=4.8):"
              f" n={len(inr):,}  mode-bin / median / mean ="
              f" {_mode_bin(inr[:,0]):.2f} / {np.median(inr[:,0]):.2f}"
              f" / {inr[:,0].mean():.2f}")
            P(f"      distance histogram (0.2 A bins):"
              f" {_hist_line(inr[:,0], 3.0, 4.8, 0.2)}")
        if key == 'obs_pipi':
            # the edge/T-shaped branch the current kernel still scores
            edge = T[(T[:, 1] > 66) & (T[:, 0] >= 4.2) & (T[:, 0] <= 5.8)]
            far = T[(T[:, 1] > 66) & (T[:, 0] >= 7.0)]
            P(f"  EDGE/T-SHAPED branch (|cos|<0.4 i.e. >66 deg, 4.2-5.8 A):"
              f" n={len(edge):,}")
            P(f"    perpendicular fraction there {np.mean(T[(T[:,0]>=4.2)&(T[:,0]<=5.8)][:,1]>66):.3f}"
              f" vs {np.mean(T[T[:,0]>=7.0][:,1]>66):.3f} at d>7 A"
              f" vs {np.cos(np.radians(66)):.3f} random")

    # ---------------- CH-pi ----------------
    T = z['obs_chpi']
    P(f"\n## CH-pi (ligand aliphatic C / protein ring)")
    P(f"   current: 3.4-4.5 A, cos>0.70 (<45 deg), offset<=2.0, opt 3.80")
    P(f"   nulls: angle<45 {null_angle_lt(45):.4f}, angle<56.6 "
      f"{null_angle_lt(56.6):.4f}")
    P(f"{'shell (A)':<12}{'n':>10}{'f<45':>8}{'enr<45':>9}{'f off<=2':>10}"
      f"{'perp-h med|off<=2':>19}")
    for lo, hi in ((3.0, 3.4), (3.4, 3.7), (3.7, 4.0), (4.0, 4.3), (4.3, 4.5),
                   (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 7.0)):
        k = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
        if len(k) == 0:
            continue
        f45 = np.mean(k[:, 1] < 45)
        o2 = k[k[:, 3] <= 2.0]
        P(f"{lo:.1f}-{hi:.1f}      {len(k):>10,}{f45:>8.3f}"
          f"{f45/null_angle_lt(45):>9.2f}{len(o2)/len(k):>10.3f}"
          f"{(np.median(o2[:,2]) if len(o2) else float('nan')):>19.2f}")
    tr = T[(T[:, 1] < 45) & (T[:, 3] <= 2.0)]
    P(f"  TRUE CH-pi (angle<45 AND offset<=2): n={len(tr):,} of {len(T):,}")
    if len(tr):
        P(f"    d p10/25/50/75/90 ="
          f" {' '.join(f'{v:.2f}' for v in q(tr[:,0],10,25,50,75,90))}")
        P(f"    perpendicular height p25/50/75 ="
          f" {' '.join(f'{v:.2f}' for v in q(tr[:,2],25,50,75))}")
        inr = tr[tr[:, 0] <= 4.8]
        P(f"    RECOMMENDED OPTIMUM (d<=4.8): n={len(inr):,}  mode-bin / median"
          f" / mean = {_mode_bin(inr[:,0]):.2f} / {np.median(inr[:,0]):.2f}"
          f" / {inr[:,0].mean():.2f}")
        P(f"      distance histogram (0.2 A bins):"
          f" {_hist_line(inr[:,0], 3.0, 4.8, 0.2)}")
    old = T[(T[:, 0] >= 3.5) & (T[:, 0] <= 5.0) & (T[:, 1] < 56.6)]
    new = T[(T[:, 0] >= 3.4) & (T[:, 0] <= 4.5) & (T[:, 1] < 45) & (T[:, 3] <= 2.0)]
    P(f"  contacts scored by the OLD window (3.5-5.0, <56.6 deg): {len(old):,}")
    P(f"  contacts scored by the NEW window (3.4-4.5, <45, off<=2): {len(new):,}"
      f"  ({100.0*len(new)/max(len(old),1):.0f}% of the old count)")

    # ---------------- pi-cation ----------------
    T = z['obs_picat']
    P(f"\n## pi-cation   (current: d<=5.0, cos>0.5 i.e. <60 deg, offset<=2.0, opt 3.8)")
    P(f"   null for angle<60 deg: {null_angle_lt(60):.4f}")
    for mode, nm in ((0, 'ligand ring / protein cation (Lys NZ, Arg NH1/NH2, metals)'),
                     (1, 'protein ring / ligand formal cation')):
        s = T[T[:, 2] == mode]
        P(f"  {nm}: n={len(s):,}")
        if len(s) == 0:
            continue
        P(f"  {'shell (A)':<12}{'n':>9}{'f<60':>8}{'enr<60':>9}{'f<40':>8}{'enr<40':>9}")
        for lo, hi in ((3.0, 3.6), (3.6, 4.0), (4.0, 4.4), (4.4, 5.0),
                       (5.0, 5.5), (5.5, 6.5), (6.5, 8.0)):
            k = s[(s[:, 0] >= lo) & (s[:, 0] < hi)]
            if len(k) == 0:
                continue
            P(f"  {lo:.1f}-{hi:.1f}      {len(k):>9,}{np.mean(k[:,1]<60):>8.3f}"
              f"{np.mean(k[:,1]<60)/null_angle_lt(60):>9.2f}"
              f"{np.mean(k[:,1]<40):>8.3f}"
              f"{np.mean(k[:,1]<40)/null_angle_lt(40):>9.2f}")

    # ---------------- salt bridges ----------------
    T = z['obs_salt']
    P(f"\n## Salt bridges (current: d<=4.0, ramp opt 2.80, burial weighted)")
    opp, like = T[T[:, 1] < 0], T[T[:, 1] > 0]
    P(f"  {len(T):,} contacts <= 7 A: {len(opp):,} opposite / {len(like):,} like-charge")
    P("  A genuine electrostatic preference shows opp/like RISING sharply at")
    P("  contact distance. If the ratio is flat, the charge labelling is noise.")
    P(f"  {'shell (A)':<12}{'opposite':>10}{'like':>10}{'opp/like':>10}")
    for lo, hi in ((2.4, 2.7), (2.7, 3.0), (3.0, 3.3), (3.3, 3.6), (3.6, 4.0),
                   (4.0, 4.5), (4.5, 5.0), (5.0, 6.0), (6.0, 7.0)):
        a = int(np.sum((opp[:, 0] >= lo) & (opp[:, 0] < hi)))
        b = int(np.sum((like[:, 0] >= lo) & (like[:, 0] < hi)))
        if a + b == 0:
            continue
        P(f"  {lo:.1f}-{hi:.1f}      {a:>10,}{b:>10,}"
          f"{(a/b if b else float('inf')):>10.2f}")
    if T.shape[1] >= 5:
        P("\n  split by how the ligand charge sign was assigned:")
        for code, nm in ((0.0, 'FORMAL charge'),
                         (1.0, 'Gasteiger |q|>0.35 fallback')):
            sub = T[T[:, 3] == code]
            if len(sub) == 0:
                P(f"    {nm}: none")
                continue
            o, l = sub[sub[:, 1] < 0], sub[sub[:, 1] > 0]
            P(f"    {nm}: {len(sub):,} ({len(o):,} opp / {len(l):,} like)")
            for lo, hi in ((2.4, 3.0), (3.0, 3.6), (3.6, 4.0), (4.0, 5.0), (5.0, 7.0)):
                a = int(np.sum((o[:, 0] >= lo) & (o[:, 0] < hi)))
                b = int(np.sum((l[:, 0] >= lo) & (l[:, 0] < hi)))
                if a + b == 0:
                    continue
                P(f"      {lo:.1f}-{hi:.1f}  opp {a:>6,}  like {b:>6,}"
                  f"  ratio {(a/b if b else float('inf')):.2f}")
    else:
        P("  (charge-provenance column not present in this survey file)")
    w = opp[opp[:, 0] <= 4.0]
    if len(w):
        P(f"  burial factor of in-window groups p10/25/50/75/90 ="
          f" {' '.join(f'{v:.2f}' for v in q(w[:,2],10,25,50,75,90))}")
        P(f"  fraction of in-window groups at burial == 1.0 (the cap):"
          f" {np.mean(w[:,2] >= 0.999):.3f}")

    # ---------------- metal ----------------
    T = z['obs_metal']
    P(f"\n## Metal coordination (current: opt 2.40, full 2.0-2.8, zero at 4.0,"
      f" score cutoff 4.5)")
    P(f"  {len(T):,} metal...(N/O/S) contacts <= 4.5 A")
    P(f"  {'shell (A)':<12}{'n':>9}{'cumulative':>12}")
    cum = 0
    for lo, hi in ((1.7, 1.9), (1.9, 2.0), (2.0, 2.1), (2.1, 2.2), (2.2, 2.3),
                   (2.3, 2.5), (2.5, 2.7), (2.7, 3.0), (3.0, 3.4), (3.4, 4.0),
                   (4.0, 4.5)):
        k = int(np.sum((T[:, 0] >= lo) & (T[:, 0] < hi)))
        cum += k
        if k:
            P(f"  {lo:.1f}-{hi:.1f}      {k:>9,}{cum:>12,}")
    first = T[T[:, 0] <= 2.6]
    P(f"  first coordination shell (<=2.6 A): n={len(first):,}"
      f"  d p10/25/50/75/90 ="
      f" {' '.join(f'{v:.2f}' for v in q(first[:,0],10,25,50,75,90))}")
    names = {0: 'Zn', 1: 'Mg', 2: 'Mn', 3: 'Fe', 4: 'Ca', 5: 'other'}
    P(f"  {'metal':<8}{'n(<=2.6)':>10}{'d p25':>8}{'d med':>8}{'d p75':>8}")
    for mid, nm in names.items():
        s = first[first[:, 1] == mid]
        if len(s) < 5:
            continue
        P(f"  {nm:<8}{len(s):>10,}"
          f"{''.join(f'{v:>8.2f}' for v in q(s[:,0],25,50,75))}")
    scored = T[T[:, 0] <= 4.0]
    P(f"  contacts the current ramp gives a non-zero score (<=4.0 A): {len(scored):,}")
    P(f"  of those actually in the first shell (<=2.6 A):             {len(first):,}"
      f"  ({100.0*len(first)/max(len(scored),1):.0f}%)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text('\n'.join(lines) + '\n')
        print(f"\nwritten to {args.out}")


if __name__ == '__main__':
    main()

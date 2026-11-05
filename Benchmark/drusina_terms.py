#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Faithful NumPy port of the Drusina scoring function (Metal/DockingScoring.metal
``computeDrusinaCorrections``) plus the Vina baseline, factored so that
protein features and ligand topology are built once and many ligand
conformations (crystal pose, decoy poses) can be rescored cheaply.

Ground truth for every functional form in this file:
  * Metal/DockingScoring.metal     lines 233-820  (computeDrusinaCorrections)
  * Metal/DockingCommon.h          drusinaVinaQuality, vina pair terms
  * Metal/DockingGridMaps.metal    computeElectrostaticGrid (screened Coulomb)
  * Docking/DockingEngine.swift    prepareDrusinaBuffers (feature detection),
                                   vinaProteinAtomType, computeBurialFactor

Known, deliberate deviations from the GPU path are listed in DEVIATIONS below
and are repeated in the calibration report. Nothing is approximated silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Deviations from the Metal implementation (documented, not silent)
# ---------------------------------------------------------------------------

DEVIATIONS = """
1. Vina baseline is evaluated by explicit pair summation (the `scorePosesExplicit`
   path), not by trilinear interpolation of 0.375 A half-precision affinity maps
   (`scorePoses`). Difference is interpolation noise, typically < 0.1 kcal/mol.
2. The screened-Coulomb term is evaluated as a direct double sum
   phi = sum_p 332 q_p / (4 r^2)  over protein atoms with r <= 12 A (the same
   12 A cutoff and the same 0.8 A singularity clamp as computeElectrostaticGrid),
   then clamped per ligand atom to [-50, +50] to mirror the half-precision grid
   clamp. It is NOT read through a trilinear grid interpolation.
3. Protein partial charges in the app come from the preparation-stage charge
   calculator (EEM/QEq/Gasteiger over the prepared receptor). That calculation is
   not reproduced here; a residue-template charge model (CHARGES_PROTEIN below,
   Amber-ff99-like on the polar/charged atoms, 0 on apolar carbons) is used
   instead. The Coulomb term therefore reproduces the FORM and the spatial
   behaviour of the GPU term, but not its exact magnitude. Any recommended
   wCoulomb must be read with that caveat.
4. Ligand partial charges are RDKit Gasteiger. The app uses
   `electrostaticCharge(for:)` = atom.charge (Gasteiger/EEM/QEq, whichever the
   preparation used) or the formal charge when no partial charge is present.
5. intramolecularLigandEnergy / referenceIntraEnergy (the Vina intra delta) is
   zero for a rigid crystal or decoy pose scored as-is, so it is omitted. It
   cancels in every comparison made here (same ligand, same internal geometry
   for crystal; decoys from CASF are rigid-body + torsion variants where the
   intra delta is small but not exactly zero -- see --intra flag).
6. `vinaRotNormalization` uses the number of GA torsions. Here the RDKit
   rotatable-bond count is used as a stand-in; it cancels for decoy ranking of
   one ligand (constant per target) and only shifts the pKd fit intercept.
"""

# ---------------------------------------------------------------------------
# Vina XS atom types (ShaderTypes.h / DockingCommon.h)
# ---------------------------------------------------------------------------

# NOTE: the numeric ids matter -- they index kVinaXSRadii. This is the exact
# ShaderTypes.h `enum VinaAtomType` ordering (N_P before N_D, O_P before O_D,
# MET_D = 18). Benchmark/calibrate_drusina_pdbbind.py used a different, wrong
# ordering with MET_D = 16, which gave metals a 2.2 A XS radius instead of 1.2.
VINA_C_H, VINA_C_P = 0, 1
VINA_N_P, VINA_N_D, VINA_N_A, VINA_N_DA = 2, 3, 4, 5
VINA_O_P, VINA_O_D, VINA_O_A, VINA_O_DA = 6, 7, 8, 9
VINA_S_P, VINA_P_P = 10, 11
VINA_F_H, VINA_Cl_H, VINA_Br_H, VINA_I_H = 12, 13, 14, 15
VINA_Si, VINA_At, VINA_MET_D = 16, 17, 18
VINA_OTHER = 31
MAX_VINA_TYPE = VINA_MET_D          # xsTypeSupported upper bound

XS_RADII = np.array([1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7,
                     2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2],
                    dtype=np.float64)

# xsIsDonor includes VINA_MET_D (DockingCommon.h:176)
DONORS = frozenset({VINA_N_D, VINA_N_DA, VINA_O_D, VINA_O_DA, VINA_MET_D})
ACCEPTORS = frozenset({VINA_N_A, VINA_N_DA, VINA_O_A, VINA_O_DA})
HBOND_TYPES = DONORS | ACCEPTORS
HYDROPHOBIC = frozenset({VINA_C_H, VINA_F_H, VINA_Cl_H, VINA_Br_H, VINA_I_H})
POLAR_LIG = frozenset({VINA_N_A, VINA_N_D, VINA_N_DA, VINA_O_A, VINA_O_D, VINA_O_DA})
METAL_COORD_LIG = frozenset({VINA_N_A, VINA_N_DA, VINA_N_D, VINA_O_A, VINA_O_DA, VINA_S_P})

# Vina 1.1.2 weights (DockingCommon.h)
W_GAUSS1 = -0.035579
W_GAUSS2 = -0.005156
W_REPULSION = 0.840245
W_HYDROPHOBIC = -0.035069
W_HBOND = -0.587439
W_ROT_ENTROPY = 0.05846

# Boolean lookup tables over the type id space, for vectorised Vina terms.
_N_TYPE_SLOTS = 32
_IS_DONOR = np.zeros(_N_TYPE_SLOTS, dtype=bool)
_IS_ACCEPTOR = np.zeros(_N_TYPE_SLOTS, dtype=bool)
_IS_HYDROPHOBIC = np.zeros(_N_TYPE_SLOTS, dtype=bool)
for _t in DONORS:
    _IS_DONOR[_t] = True
for _t in ACCEPTORS:
    _IS_ACCEPTOR[_t] = True
for _t in HYDROPHOBIC:
    _IS_HYDROPHOBIC[_t] = True


_METALS = {'ZN', 'FE', 'MG', 'CA', 'MN', 'CU', 'CO', 'NI', 'NA', 'K', 'CD',
           'HG', 'NI', 'SC', 'TI', 'V', 'CR'}
_HIS = {'HIS', 'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'}
_WATER = {'HOH', 'WAT', 'DOD', 'H2O'}
# Side-chain N that carry H at pH 7 when the receptor has no explicit H
# (DockingEngine.swift proteinDonorNitrogenNames)
_PROT_DONOR_N = {'NZ', 'NE', 'NH1', 'NH2', 'ND2', 'NE2', 'NE1', 'ND1'}

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'MSE', 'SEC', 'PYL',
}

# DockingEngine+Analysis.aromaticResidueAtoms -- exactly these four residue
# names, with no HID/HIE/HIP aliases (matching the Swift table verbatim).
AROMATIC_RING_ATOMS = {
    'PHE': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'TYR': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'TRP': {'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
    'HIS': {'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
}

# Residue-template protein charges (deviation 3). Amber-ff99-like on the polar
# and charged atoms; apolar carbons 0. Keyed (res, atom) then (atom,) backbone.
CHARGES_BACKBONE = {'N': -0.4157, 'CA': 0.0337, 'C': 0.5973, 'O': -0.5679,
                    'OXT': -0.8055}
CHARGES_PROTEIN = {
    ('ASP', 'CG'): 0.7994, ('ASP', 'OD1'): -0.8014, ('ASP', 'OD2'): -0.8014,
    ('GLU', 'CD'): 0.8054, ('GLU', 'OE1'): -0.8188, ('GLU', 'OE2'): -0.8188,
    ('LYS', 'NZ'): -0.3854, ('LYS', 'CE'): 0.3260,   # NZ + 3 H(+0.34) => +0.64 net
    ('ARG', 'NE'): -0.5295, ('ARG', 'CZ'): 0.8076,
    ('ARG', 'NH1'): -0.8627, ('ARG', 'NH2'): -0.8627, ('ARG', 'CD'): 0.0486,
    ('SER', 'OG'): -0.6546, ('THR', 'OG1'): -0.6761, ('TYR', 'OH'): -0.5579,
    ('CYS', 'SG'): -0.3119, ('MET', 'SD'): -0.2737,
    ('ASN', 'CG'): 0.7130, ('ASN', 'OD1'): -0.5931, ('ASN', 'ND2'): -0.9191,
    ('GLN', 'CD'): 0.6951, ('GLN', 'OE1'): -0.6086, ('GLN', 'NE2'): -0.9407,
    ('TRP', 'NE1'): -0.3418,
    ('HIS', 'ND1'): -0.3811, ('HIS', 'NE2'): -0.5727, ('HIS', 'CE1'): 0.2057,
    ('HIP', 'ND1'): -0.1513, ('HIP', 'NE2'): -0.1718, ('HIP', 'CE1'): 0.0,
}
# Net formal charges carried by the implicit hydrogens we do not have.
# Applied at the heavy atom so that whole-group charge is right.
CHARGES_IMPLICIT_H = {
    ('LYS', 'NZ'): 3 * 0.3400, ('ARG', 'NE'): 0.3456,
    ('ARG', 'NH1'): 2 * 0.4478, ('ARG', 'NH2'): 2 * 0.4478,
    ('SER', 'OG'): 0.4275, ('THR', 'OG1'): 0.4102, ('TYR', 'OH'): 0.3992,
    ('CYS', 'SG'): 0.1933, ('ASN', 'ND2'): 2 * 0.4196, ('GLN', 'NE2'): 2 * 0.4251,
    ('TRP', 'NE1'): 0.3412, ('HIS', 'ND1'): 0.3649,
    ('HIP', 'ND1'): 0.3866, ('HIP', 'NE2'): 0.3911,
}
CHARGES_BACKBONE_IMPLICIT_H = {'N': 0.2719}  # amide H
METAL_FORMAL = {'ZN': 2.0, 'FE': 2.0, 'MG': 2.0, 'CA': 2.0, 'MN': 2.0,
                'CU': 2.0, 'CO': 2.0, 'NI': 2.0, 'NA': 1.0, 'K': 1.0,
                'CD': 2.0, 'HG': 2.0}


# ---------------------------------------------------------------------------
# Shared helpers (exact ports)
# ---------------------------------------------------------------------------

def drusina_ramp(x, inner, outer):
    """Metal drusinaRamp: 1 for |x|<=inner, linear to 0 at |x|>=outer."""
    ax = np.abs(x)
    out = np.where(ax <= inner, 1.0, (outer - ax) / (outer - inner))
    return np.clip(out, 0.0, 1.0)


def smoothstep(e0, e1, x):
    """Metal smoothstep(edge0, edge1, x)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def vina_rot_normalization(n_rot):
    """DockingCommon.h vinaRotNormalization."""
    return 1.0 / (1.0 + W_ROT_ENTROPY * float(n_rot))


def drusina_vina_quality(vina_e, center=2.0, slope=0.5):
    """DockingCommon.h drusinaVinaQuality."""
    return 1.0 / (1.0 + np.exp(slope * (np.asarray(vina_e, dtype=np.float64) - center)))


# ---------------------------------------------------------------------------
# Protein parsing / feature extraction
# ---------------------------------------------------------------------------

@dataclass
class ProteinFeatures:
    pos: np.ndarray                       # (P,3)
    vtype: np.ndarray                     # (P,) int
    charge: np.ndarray                    # (P,) float
    is_acceptor: np.ndarray               # (P,) bool
    is_metal: np.ndarray                  # (P,) bool
    ring_centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ring_normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ring_res: list = field(default_factory=list)
    cation_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    amide_centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    amide_normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    amide_kind: list = field(default_factory=list)        # 'bb' | 'sc'
    sb_centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sb_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sb_burial: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chal_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    chal_dir: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    hb_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    hb_is_donor: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    hb_ant: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def _vina_protein_type(name, res, elem, het, n_heavy_nb,
                       h_present=False, bonded_h=False):
    """Port of DockingEngine.vinaProteinAtomType.

    `h_present` mirrors the Swift `hydrogensPresent` argument (true when the
    receptor file carries explicit hydrogens, as every PDBbind refined-set
    *_protein.pdb does) and `bonded_h` mirrors hasAttachedHydrogen.
    """
    if elem in _METALS:
        return VINA_MET_D
    if elem == 'C':
        # isBondedToHeteroatom: resolved by name tables (no bond graph for PDB).
        return VINA_C_P if _c_bonded_hetero(name, res) else VINA_C_H
    if elem == 'N':
        if h_present and bonded_h:
            n_acceptor = False
        elif not het:
            n_acceptor = name in ('ND1', 'NE2') and res in _HIS
        else:
            n_acceptor = 0 < n_heavy_nb < 3
        if h_present:
            n_donor = bonded_h
        else:
            n_donor = (name == 'N' and res != 'PRO') or name in _PROT_DONOR_N
        if n_donor and n_acceptor:
            return VINA_N_DA
        if n_acceptor:
            return VINA_N_A
        if n_donor:
            return VINA_N_D
        return VINA_N_P
    if elem == 'O':
        if h_present:
            o_donor = bonded_h
        else:
            o_donor = name in ('OG', 'OG1', 'OH') or res in _WATER
        return VINA_O_DA if o_donor else VINA_O_A
    if elem == 'S':
        return VINA_S_P
    if elem == 'P':
        return VINA_P_P
    if elem == 'F':
        return VINA_F_H
    if elem == 'CL':
        return VINA_Cl_H
    if elem == 'BR':
        return VINA_Br_H
    if elem == 'I':
        return VINA_I_H
    return VINA_OTHER


# Carbons bonded to a heteroatom (N/O/S/P) in the 20 standard residues.
_C_POLAR = {
    ('*', 'C'), ('*', 'CA'),
    ('ASP', 'CG'), ('GLU', 'CD'), ('ASN', 'CG'), ('GLN', 'CD'),
    ('SER', 'CB'), ('THR', 'CB'), ('CYS', 'CB'), ('MET', 'CG'), ('MET', 'CE'),
    ('LYS', 'CE'), ('ARG', 'CD'), ('ARG', 'CZ'), ('TYR', 'CZ'),
    ('HIS', 'CG'), ('HIS', 'CD2'), ('HIS', 'CE1'),
    ('HID', 'CG'), ('HID', 'CD2'), ('HID', 'CE1'),
    ('HIE', 'CG'), ('HIE', 'CD2'), ('HIE', 'CE1'),
    ('HIP', 'CG'), ('HIP', 'CD2'), ('HIP', 'CE1'),
    ('TRP', 'CD1'), ('TRP', 'CE2'), ('PRO', 'CD'),
}


def _c_bonded_hetero(name, res):
    if ('*', name) in _C_POLAR:
        return True
    return (res, name) in _C_POLAR


def _protein_charge(name, res, elem):
    if elem in METAL_FORMAL:
        return METAL_FORMAL[elem]
    q = CHARGES_PROTEIN.get((res, name))
    if q is None:
        q = CHARGES_BACKBONE.get(name)
    if q is None:
        return 0.0
    q += CHARGES_IMPLICIT_H.get((res, name), 0.0)
    if name == 'N' and res != 'PRO':
        q += CHARGES_BACKBONE_IMPLICIT_H['N']
    return q


def parse_protein(pdb_path, center=None, radius=None, keep_water=False):
    """Parse a PDB receptor into raw per-atom records (heavy atoms only).

    Hydrogens are not returned, but their presence is recorded: each heavy-atom
    record gets `bonded_h` (an H within 1.35 A) and `h_present` (the file
    carries explicit H at all), so that _vina_protein_type can take the same
    branch the Swift code takes for that receptor.
    """
    recs, hpos = [], []
    with open(pdb_path, 'r', errors='replace') as fh:
        for line in fh:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            alt = line[16:17]
            if alt not in (' ', 'A'):
                continue
            res = line[17:20].strip().upper()
            if not keep_water and res in _WATER:
                continue
            elem = line[76:78].strip().upper()
            name = line[12:16].strip().upper()
            if not elem:
                elem = ''.join(c for c in name if c.isalpha())[:1]
            if not elem:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            if elem in ('H', 'D'):
                hpos.append((x, y, z))
                continue
            p = (x, y, z)
            if center is not None and radius is not None:
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                if dx * dx + dy * dy + dz * dz > radius * radius:
                    continue
            try:
                seq = int(line[22:26])
            except ValueError:
                seq = 0
            recs.append({
                'pos': p, 'name': name, 'res': res,
                'chain': line[21:22], 'seq': seq, 'icode': line[26:27],
                'elem': elem, 'het': line.startswith('HETATM') and res not in STANDARD_AA,
                'h_present': False, 'bonded_h': False,
            })
    if hpos and recs:
        H = np.array(hpos, dtype=np.float64)
        P = np.array([r['pos'] for r in recs], dtype=np.float64)
        # H within 1.35 A of a heavy atom marks that atom as hydrogen-bearing.
        # A KD-tree, not a dense distance matrix: on a 4000-atom receptor with
        # 4000 hydrogens the dense form took ~14 s per protein and was by far
        # the dominant cost of the whole pipeline.
        tree = cKDTree(H)
        near = tree.query_ball_point(P, 1.35, return_length=True)
        for j, k in enumerate(near):
            recs[j]['bonded_h'] = bool(k)
        for r in recs:
            r['h_present'] = True
    return recs


def build_protein_features(recs, sb_burial_positions=None,
                           peptide_amide_planes=False):
    """Build every protein-side Drusina feature from parsed records.

    `sb_burial_positions` is the full receptor heavy-atom coordinate array used
    for computeBurialFactor (DockingEngine.computeBurialFactor counts neighbours
    within 8 A over ALL protein atoms, so passing only the pocket subset would
    systematically under-count burial). Defaults to the given records.
    """
    if not recs:
        raise ValueError("empty protein")

    pos = np.array([r['pos'] for r in recs], dtype=np.float64)
    by_res = defaultdict(list)
    for i, r in enumerate(recs):
        by_res[(r['chain'], r['seq'], r['icode'])].append(i)

    # heavy-neighbour counts (needed for HETATM nitrogen typing)
    n_heavy_nb = np.zeros(len(recs), dtype=np.int32)
    het_idx = [i for i, r in enumerate(recs) if r['het'] and r['elem'] == 'N']
    if het_idx:
        tree = cKDTree(pos)
        k = tree.query_ball_point(pos[het_idx], 1.8, return_length=True)
        for i, kk in zip(het_idx, k):
            n_heavy_nb[i] = int(kk) - 1        # drop the atom itself

    vtype = np.array([
        _vina_protein_type(r['name'], r['res'], r['elem'], r['het'], n_heavy_nb[i],
                           r.get('h_present', False), r.get('bonded_h', False))
        for i, r in enumerate(recs)], dtype=np.int32)
    charge = np.array([_protein_charge(r['name'], r['res'], r['elem']) for r in recs])
    is_acceptor = np.isin(vtype, [VINA_N_A, VINA_N_DA, VINA_O_A, VINA_O_DA])
    is_metal = vtype == VINA_MET_D

    # --- aromatic rings (DockingEngine+Analysis.detectAromaticRings, >=5 atoms)
    rc, rn, rres = [], [], []
    for key, idxs in by_res.items():
        res = recs[idxs[0]]['res']
        targets = AROMATIC_RING_ATOMS.get(res)
        if targets is None:
            continue
        ring = [i for i in idxs if recs[i]['name'] in targets]
        if len(ring) < 5:
            continue
        rp = pos[ring]
        cen = rp.mean(axis=0)
        nrm = np.cross(rp[1] - rp[0], rp[2] - rp[0])
        nl = np.linalg.norm(nrm)
        if nl < 1e-6:
            continue
        rc.append(cen)
        rn.append(nrm / nl)
        rres.append(res)

    # --- cations: NZ/NH1/NH2 by name, plus metals (prepareDrusinaBuffers)
    cat = [pos[i] for i, r in enumerate(recs)
           if r['name'] in ('NZ', 'NH1', 'NH2') or r['elem'] in METAL_FORMAL]

    # --- amide planes.
    # DockingEngine.swift:986-999 builds each BACKBONE plane from the C, O and
    # N of the SAME residue. The peptide amide is O=C(i)...N(i+1); N(i) is two
    # bonds from C(i) through CA, ~2.4 A away instead of the 1.33 A peptide
    # bond, so that plane's normal is wrong. `peptide_amide_planes=True` uses
    # the correct C(i), O(i), N(i+1) triple; False reproduces the shipped bug.
    ac, an, ak = [], [], []
    bb_by_res = {}
    for key, idxs in by_res.items():
        nm = {recs[i]['name']: i for i in idxs if not recs[i]['het']}
        res = recs[idxs[0]]['res']
        if 'C' in nm and 'O' in nm:
            bb_by_res[key] = nm
        if peptide_amide_planes:
            groups = []
        else:
            groups = [('bb', 'C', 'O', 'N')]
        if res == 'ASN':
            groups.append(('sc', 'CG', 'OD1', 'ND2'))
        elif res == 'GLN':
            groups.append(('sc', 'CD', 'OE1', 'NE2'))
        for kind, cn, on, nn in groups:
            if cn not in nm or on not in nm or nn not in nm:
                continue
            c, o, n = pos[nm[cn]], pos[nm[on]], pos[nm[nn]]
            nrm = np.cross(o - c, n - c)
            nl = np.linalg.norm(nrm)
            if nl < 1e-6:
                continue
            ac.append((c + o + n) / 3.0)
            an.append(nrm / nl)
            ak.append(kind)

    if peptide_amide_planes:
        # pair residue i with residue i+1 of the same chain, accepting the pair
        # only when C(i)-N(i+1) is an actual peptide bond (< 1.6 A)
        for key, nm in bb_by_res.items():
            chain, seq, icode = key
            try:
                nxt = bb_by_res.get((chain, seq + 1, icode))
            except TypeError:
                nxt = None
            if nxt is None or 'N' not in nxt:
                continue
            c, o, n = pos[nm['C']], pos[nm['O']], pos[nxt['N']]
            if np.linalg.norm(n - c) > 1.6:
                continue
            nrm = np.cross(o - c, n - c)
            nl = np.linalg.norm(nrm)
            if nl < 1e-6:
                continue
            ac.append((c + o + n) / 3.0)
            an.append(nrm / nl)
            ak.append('bb')

    # --- salt-bridge groups
    burial_src = sb_burial_positions if sb_burial_positions is not None else pos
    burial_tree = cKDTree(burial_src)
    sc, ss, sb = [], [], []

    def _burial(cen):
        cnt = float(burial_tree.query_ball_point(cen, 8.0, return_length=True))
        inner = float(burial_tree.query_ball_point(cen, 0.5, return_length=True))
        return max(0.15, min(1.0, (cnt - inner) / 25.0))

    for key, idxs in by_res.items():
        res = recs[idxs[0]]['res']
        nm = defaultdict(list)
        for i in idxs:
            nm[recs[i]['name']].append(i)
        if res == 'ARG':
            g = [i for n in ('NE', 'NH1', 'NH2', 'CZ') for i in nm.get(n, [])]
            if len(g) >= 2:
                cen = pos[g].mean(axis=0)
                sc.append(cen); ss.append(1.0); sb.append(_burial(cen))
        elif res == 'LYS':
            if nm.get('NZ'):
                cen = pos[nm['NZ'][0]]
                sc.append(cen); ss.append(1.0); sb.append(_burial(cen))
        elif res in ('ASP',):
            g = [i for n in ('OD1', 'OD2') for i in nm.get(n, [])]
            if len(g) == 2:
                cen = pos[g].mean(axis=0)
                sc.append(cen); ss.append(-1.0); sb.append(_burial(cen))
        elif res in ('GLU',):
            g = [i for n in ('OE1', 'OE2') for i in nm.get(n, [])]
            if len(g) == 2:
                cen = pos[g].mean(axis=0)
                sc.append(cen); ss.append(-1.0); sb.append(_burial(cen))
        # HIS groups require formalCharge > 0, which raw PDB never carries;
        # the Swift path is likewise silent on unprepared receptors.

    # --- protein chalcogens (Met SD, Cys SG) with S->C directions
    cp, cd = [], []
    for key, idxs in by_res.items():
        res = recs[idxs[0]]['res']
        nm = {recs[i]['name']: i for i in idxs}
        if res == 'MET' and 'SD' in nm:
            for cn in ('CG', 'CE'):
                if cn in nm:
                    v = pos[nm[cn]] - pos[nm['SD']]
                    nl = np.linalg.norm(v)
                    if nl > 1e-6:
                        cp.append(pos[nm['SD']]); cd.append(v / nl)
        elif res == 'CYS' and 'SG' in nm and 'CB' in nm:
            v = pos[nm['CB']] - pos[nm['SG']]
            nl = np.linalg.norm(v)
            if nl > 1e-6:
                cp.append(pos[nm['SG']]); cd.append(v / nl)

    # --- protein H-bond donors/acceptors with antecedents
    # (DockingEngine.swift builds these per-residue; mirrored here)
    SC_HB = {
        'SER': [('OG', 'CB', None)], 'THR': [('OG1', 'CB', None)],
        'TYR': [('OH', 'CZ', None)], 'CYS': [('SG', 'CB', None)],
        'ASN': [('ND2', 'CG', True), ('OD1', 'CG', False)],
        'GLN': [('NE2', 'CD', True), ('OE1', 'CD', False)],
        'ASP': [('OD1', 'CG', False), ('OD2', 'CG', False)],
        'GLU': [('OE1', 'CD', False), ('OE2', 'CD', False)],
        'LYS': [('NZ', 'CE', True)],
        'ARG': [('NE', 'CD', True), ('NH1', 'CZ', True), ('NH2', 'CZ', True)],
        'TRP': [('NE1', 'CE2', True)],
        'HIS': [('ND1', 'CG', None), ('NE2', 'CE1', None)],
    }
    hp, hd, ha = [], [], []
    for key, idxs in by_res.items():
        res = recs[idxs[0]]['res']
        nm = {recs[i]['name']: i for i in idxs}
        if 'N' in nm and 'CA' in nm and res != 'PRO':
            hp.append(pos[nm['N']]); hd.append(True); ha.append(pos[nm['CA']])
        if 'O' in nm and 'C' in nm:
            hp.append(pos[nm['O']]); hd.append(False); ha.append(pos[nm['C']])
        for aname, antname, donor in SC_HB.get(res, []):
            if aname in nm and antname in nm:
                if donor is None:          # both donor and acceptor
                    for dn in (True, False):
                        hp.append(pos[nm[aname]]); hd.append(dn); ha.append(pos[nm[antname]])
                else:
                    hp.append(pos[nm[aname]]); hd.append(donor); ha.append(pos[nm[antname]])

    def _arr(lst, dim=3):
        return np.array(lst, dtype=np.float64) if lst else np.zeros((0, dim))

    return ProteinFeatures(
        pos=pos, vtype=vtype, charge=charge,
        is_acceptor=is_acceptor, is_metal=is_metal,
        ring_centroid=_arr(rc), ring_normal=_arr(rn), ring_res=rres,
        cation_pos=_arr(cat),
        amide_centroid=_arr(ac), amide_normal=_arr(an), amide_kind=ak,
        sb_centroid=_arr(sc), sb_sign=np.array(ss) if ss else np.zeros(0),
        sb_burial=np.array(sb) if sb else np.zeros(0),
        chal_pos=_arr(cp), chal_dir=_arr(cd),
        hb_pos=_arr(hp),
        hb_is_donor=np.array(hd, dtype=bool) if hd else np.zeros(0, dtype=bool),
        hb_ant=_arr(ha),
    )


# ---------------------------------------------------------------------------
# Ligand topology
# ---------------------------------------------------------------------------

@dataclass
class LigandTopology:
    vtype: np.ndarray
    formal_charge: np.ndarray
    charge: np.ndarray
    is_aromatic: np.ndarray
    rings: list                       # list[np.ndarray of indices]
    halogens: list                    # list[(hal_idx, c_idx, elem_type)]
    chalcogens: list                  # list[(s_idx, c_idx)]
    hb_atoms: list                    # list[(idx, ant_idx, is_donor)]
    torsions: list                    # list[(a0,a1,a2,a3,force_k)]
    n_rot: int
    n_heavy: int


def _lig_vina_type(atom):
    """DockingEngine.fallbackLigandVinaAtomType (hydrogens implicit in RDKit)."""
    el = atom.GetSymbol()
    if el == 'C':
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() in ('N', 'O', 'S', 'P'):
                return VINA_C_P
        return VINA_C_H
    if el == 'N':
        heavy = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H')
        acc = atom.GetFormalCharge() <= 0 and heavy < 3
        don = atom.GetTotalNumHs() > 0
        if don and acc:
            return VINA_N_DA
        if acc:
            return VINA_N_A
        if don:
            return VINA_N_D
        return VINA_N_P
    if el == 'O':
        acc = atom.GetFormalCharge() <= 0
        don = atom.GetTotalNumHs() > 0
        if don and acc:
            return VINA_O_DA
        if acc:
            return VINA_O_A
        if don:
            return VINA_O_D
        return VINA_O_P
    return {'S': VINA_S_P, 'P': VINA_P_P, 'F': VINA_F_H, 'Cl': VINA_Cl_H,
            'Br': VINA_Br_H, 'I': VINA_I_H}.get(el, VINA_OTHER)


_RING_ELEMENTS = frozenset({'C', 'N', 'O', 'S'})


def detect_druse_ligand_rings(mol, planarity_tol=0.5):
    """Port of DockingEngine+Analysis.detectRingsFromBonds.

    Druse does NOT use RDKit aromaticity for ligand rings. It enumerates every
    5- and 6-membered cycle over the subgraph of C/N/O/S atoms, keeps the ones
    whose atoms are planar to within 0.5 A of the plane through the three
    LOWEST-INDEX ring atoms, and requires at least len-1 of the ring atoms to
    be C or N (so at most one O/S). Ring atom indices are stored SORTED, which
    is what the kernel's atomIndices[0..2] normal is built from -- reproduced
    here because the normal (and therefore every stacking term) depends on it.
    """
    n = mol.GetNumAtoms()
    if n <= 4 or mol.GetNumConformers() == 0:
        return []
    pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    sym = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if sym[a] in _RING_ELEMENTS and sym[c] in _RING_ELEMENTS:
            adj[a].append(c)
            adj[c].append(a)

    found = set()
    for start in range(n):
        if sym[start] not in _RING_ELEMENTS:
            continue
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            if len(path) > 6:
                continue
            for nb in adj[node]:
                if nb == start and len(path) in (5, 6):
                    ring = tuple(sorted(path))
                    if ring[0] == start:
                        found.add(ring)
                elif nb > start and nb not in path and len(path) < 6:
                    stack.append((nb, path + (nb,)))

    rings = []
    for ring in sorted(found):
        rp = pos[list(ring)]
        nrm = np.cross(rp[1] - rp[0], rp[2] - rp[0])
        nl = np.linalg.norm(nrm)
        if nl < 1e-6:
            continue
        nrm /= nl
        cen = rp.mean(axis=0)
        if np.any(np.abs((rp - cen) @ nrm) > planarity_tol):
            continue
        cn = sum(1 for i in ring if sym[i] in ('C', 'N'))
        if cn < len(ring) - 1:
            continue
        rings.append(np.array(ring, dtype=np.int32))
    return rings


def build_ligand_topology(mol):
    """Build Drusina ligand-side features from an RDKit mol (heavy atoms only,
    indices refer to the heavy-atom-only molecule the caller passes in)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    n = mol.GetNumAtoms()
    vtype = np.array([_lig_vina_type(mol.GetAtomWithIdx(i)) for i in range(n)],
                     dtype=np.int32)
    fc = np.array([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n)],
                  dtype=np.float64)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        q = np.array([float(mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
                      for i in range(n)])
        q[~np.isfinite(q)] = 0.0
    except Exception:
        q = fc.copy()
    # electrostaticCharge(for:) semantics: fall back to formal charge
    q = np.where(np.abs(q) > 1e-4, q, fc)

    arom = np.zeros(n, dtype=bool)
    rings = detect_druse_ligand_rings(mol)
    for r in rings:
        arom[list(r)] = True

    halogens = []
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        et = {'F': 0, 'Cl': 1, 'Br': 2, 'I': 3}.get(a.GetSymbol())
        if et is None:
            continue
        for nb in a.GetNeighbors():
            if nb.GetSymbol() == 'C':
                halogens.append((i, nb.GetIdx(), et))
                break

    chalcogens = []
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'S':
            continue
        for nb in a.GetNeighbors():
            if nb.GetSymbol() == 'C':
                chalcogens.append((i, nb.GetIdx()))

    hb_atoms = []
    for i in range(n):
        vt = int(vtype[i])
        if vt not in HBOND_TYPES:
            continue
        ant = -1
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetSymbol() == 'H':
                continue
            ant = nb.GetIdx()
            if nb.GetSymbol() == 'C':
                break
        if ant < 0:
            continue
        if vt in DONORS:
            hb_atoms.append((i, ant, True))
        if vt in ACCEPTORS:
            hb_atoms.append((i, ant, False))

    # amide torsion strain (prepareDrusinaBuffers, forceConstant 5.0)
    torsions = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if {a1.GetSymbol(), a2.GetSymbol()} != {'N', 'C'}:
            continue
        nat, cat = (a1, a2) if a1.GetSymbol() == 'N' else (a2, a1)
        is_amide = any(
            nb.GetSymbol() == 'O' and
            mol.GetBondBetweenAtoms(cat.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nb in cat.GetNeighbors())
        if not is_amide:
            continue
        a0 = next((nb.GetIdx() for nb in nat.GetNeighbors()
                   if nb.GetIdx() != cat.GetIdx() and nb.GetSymbol() != 'H'), None)
        a3 = next((nb.GetIdx() for nb in cat.GetNeighbors()
                   if nb.GetIdx() != nat.GetIdx() and nb.GetSymbol() != 'H'), None)
        if a0 is None or a3 is None:
            continue
        torsions.append((a0, nat.GetIdx(), cat.GetIdx(), a3, 5.0))

    try:
        n_rot = Descriptors.NumRotatableBonds(mol)
    except Exception:
        n_rot = 0

    return LigandTopology(vtype=vtype, formal_charge=fc, charge=q,
                          is_aromatic=arom, rings=rings, halogens=halogens,
                          chalcogens=chalcogens, hb_atoms=hb_atoms,
                          torsions=torsions, n_rot=int(n_rot), n_heavy=n)


# ---------------------------------------------------------------------------
# Raw (unweighted) Drusina terms
# ---------------------------------------------------------------------------

TERMS = [
    "pi_pi", "pi_cation", "salt_bridge", "amide_pi",
    "halogen_bond", "chalcogen_bond", "metal_coord", "coulomb",
    "ch_pi", "torsion_strain", "cooperativity",
    "hbond_dir", "desolv_polar", "desolv_hydrophobic",
]
N_TERMS = len(TERMS)
TERM_INDEX = {t: i for i, t in enumerate(TERMS)}

SWIFT_SYMBOL = {
    "pi_pi": "wPiPi", "pi_cation": "wPiCation", "salt_bridge": "wSaltBridge",
    "amide_pi": "wAmideStack", "halogen_bond": "wHalogenBond",
    "chalcogen_bond": "wChalcogenBond", "metal_coord": "wMetalCoord",
    "coulomb": "wCoulomb", "ch_pi": "wCHPi", "torsion_strain": "wTorsionStrain",
    "cooperativity": "wCooperativity", "hbond_dir": "wHBondDir",
    "desolv_polar": "wDesolvPolar", "desolv_hydrophobic": "wDesolvHydrophobic",
}

# Current in-tree weights (Docking/DockingEngine.swift ~line 1400)
CURRENT_WEIGHTS = np.array([
    -0.20,    # pi_pi
    -0.05,    # pi_cation
    -0.20,    # salt_bridge
    -0.15,    # amide_pi
    -0.40,    # halogen_bond
    -0.10,    # chalcogen_bond
    -0.91,    # metal_coord
     0.015,   # coulomb
    -0.26,    # ch_pi
     1.00,    # torsion_strain
     0.00,    # cooperativity
    -0.32,    # hbond_dir
     0.00,    # desolv_polar
     0.00,    # desolv_hydrophobic
])
CURRENT_MIN_CORRECTION = -5.5

LIGAND_CUTOFF = 10.0            # DockingScoring.metal:288


def _ring_frames(lig_pos, rings):
    """Centroids + normals for up to 8 ligand rings (MAX_LIG_RINGS)."""
    cen, nrm = [], []
    for r in rings[:8]:
        if len(r) < 3:
            continue
        p = lig_pos[r]
        c = p.mean(axis=0)
        v = np.cross(p[1] - p[0], p[2] - p[0])
        nl = np.linalg.norm(v)
        if nl < 1e-6:
            continue
        cen.append(c)
        nrm.append(v / nl)
    if not cen:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(cen), np.array(nrm)


# ---------------------------------------------------------------------------
# Variant V2: measured cutoffs + one contribution per interaction site
# ---------------------------------------------------------------------------
# Two independent changes relative to the committed kernel, both of which this
# measurement pass argues for:
#   (1) the distance/angle windows and optima recommended in
#       Benchmark/reports/drusina_calibration_2026.md section 7, including the
#       lateral-offset caps and the deletion of the pi-pi edge branch, the
#       amide-pi tilted branch and the salt-bridge partial-charge fallback;
#   (2) BEST PARTNER PER SITE instead of a sum over all partners in range --
#       one stack per ring face, one CH-pi per ligand carbon, one halogen bond
#       per halogen, one salt bridge per charged ligand atom, one coordination
#       per metal. Summing makes a term scale with how many protein features
#       happen to be nearby, which is burial depth, not binding mode.
# Keeping the two separable is deliberate: evaluating V2 against V1 does not
# say which of the two changes did the work, so the calibration script reports
# the cutoff-only variant as well.

V2 = dict(
    hal_dist_min=2.50, hal_dist_max=3.60, hal_cos=-0.866,
    hal_opt={1: 3.27, 2: 3.25, 3: 3.17},
    hal_ramp=(0.25, 0.35),
    chal_dist_min=2.80, chal_dist_max=3.60, chal_opt=3.30, chal_cos=-0.766,
    chal_ramp=(0.25, 0.30),
    pipi_dist_min=3.20, pipi_dist_max=4.50, pipi_opt=3.80, pipi_cos=0.80,
    pipi_offset=2.00, pipi_ramp=(0.30, 0.70),
    amide_dist_min=3.20, amide_dist_max=4.60, amide_opt=4.10, amide_cos=0.80,
    amide_offset=2.00, amide_ramp=(0.30, 0.50),
    chpi_dist_min=3.20, chpi_dist_max=4.50, chpi_opt=3.90, chpi_cos=0.70,
    chpi_offset=2.00, chpi_ramp=(0.30, 0.60),
    picat_dist_max=4.60, picat_opt=3.80, picat_cos=0.766, picat_offset=2.00,
    picat_ramp=(0.40, 0.80),
    pipi_edge=False, pipi_edge_cos=0.40, pipi_edge_opt=4.80,
    pipi_edge_dist_max=5.80, pipi_edge_ramp=(0.40, 1.00), pipi_edge_scale=0.60,
    hb_dist_min=2.40, hb_dist_max=3.50, hb_opt=2.90, hb_ramp=(0.20, 0.60),
    coulomb_burial_floor=0.20,
    salt_dist_max=5.00, salt_opt=2.80, salt_ramp=(0.30, 2.20),
    salt_allow_partial=False, salt_burial_divisor=70.0,
    metal_opt=2.10, metal_dist_max=2.60, metal_cutoff=3.50,
    metal_ramp=(0.25, 0.60),
    reach=5.60,
    one_per_site=True,
)


def raw_terms_v2(pf: ProteinFeatures, lt: LigandTopology, lig_pos: np.ndarray,
                 cfg=None, one_per_site=None):
    """Raw Drusina terms under the REWRITTEN kernel (best partner per site,
    lateral-offset caps, smooth angular gates, burial-scaled Coulomb, no
    amide tilted branch) with the Section-7 measured cutoffs as defaults.

    Mirrors Metal/DockingScoring.metal as rewritten on 2026-11-05, loop
    direction included: CH-pi and the protein-ring pi-cation branch iterate
    protein rings on the outside and take the best LIGAND partner per ring
    face; the salt bridge iterates protein charged groups and takes the best
    ligand atom. Which side the "one per site" rule is applied to changes the
    result, so it is reproduced rather than approximated.

    one_per_site=False keeps the same windows but restores summation, so the
    two changes the rewrite makes at once can be separated.
    """
    c = dict(V2)
    if cfg:
        c.update(cfg)
    if one_per_site is not None:
        c['one_per_site'] = one_per_site
    ops = c['one_per_site']

    raw = np.zeros(N_TERMS)
    icount = 0
    cen_l = lig_pos.mean(axis=0)
    lig_radius = float(np.max(np.linalg.norm(lig_pos - cen_l, axis=1)))
    cull2 = (lig_radius + c['reach']) ** 2
    P = pf.pos
    pcull = np.sum((P - cen_l) ** 2, axis=1) <= cull2

    lrc, lrn = _ring_frames(lig_pos, lt.rings)

    def _take(scores, acc_index, two_sided=None):
        """Apply the one-per-site rule. `two_sided` is the signed axial
        displacement when the site has two faces (a ring), else None."""
        nonlocal icount
        if len(scores) == 0:
            return
        if not ops:
            raw[acc_index] += float(np.sum(scores))
            icount += len(scores)
            return
        if two_sided is None:
            raw[acc_index] += float(np.max(scores))
            icount += 1
            return
        up = two_sided >= 0
        for side in (up, ~up):
            if side.any():
                raw[acc_index] += float(np.max(scores[side]))
                icount += 1

    # ---- pi-pi: best face per side, plus best edge, per ligand ring --------
    if len(lrc) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cull2
        if gate.any():
            prc, prn = pf.ring_centroid[gate], pf.ring_normal[gate]
            for i in range(len(lrc)):
                delta = prc - lrc[i]
                d = np.linalg.norm(delta, axis=1)
                axial = delta @ lrn[i]
                off = np.sqrt(np.maximum(d ** 2 - axial ** 2, 0.0))
                dotn = np.abs(prn @ lrn[i])
                face = ((d >= c['pipi_dist_min']) & (d <= c['pipi_dist_max']) &
                        (dotn >= c['pipi_cos']) & (off <= c['pipi_offset']))
                if face.any():
                    sc = (smoothstep(c['pipi_cos'], 1.0, dotn[face]) *
                          drusina_ramp(d[face] - c['pipi_opt'], *c['pipi_ramp']))
                    _take(sc, 0, axial[face])
                if c['pipi_edge']:
                    edge = ((d >= c['pipi_dist_min']) & (d <= c['pipi_edge_dist_max']) &
                            (dotn <= c['pipi_edge_cos']) & ~face)
                    if edge.any():
                        sc = (c['pipi_edge_scale'] *
                              smoothstep(c['pipi_edge_cos'], 0.0, dotn[edge]) *
                              drusina_ramp(d[edge] - c['pipi_edge_opt'],
                                           *c['pipi_edge_ramp']))
                        _take(sc, 0)

    # ---- pi-cation: best cation per ligand ring face ----------------------
    if len(lrc) and len(pf.cation_pos):
        gate = np.sum((pf.cation_pos - cen_l) ** 2, axis=1) <= cull2
        if gate.any():
            cpos = pf.cation_pos[gate]
            for i in range(len(lrc)):
                delta = cpos - lrc[i]
                d = np.linalg.norm(delta, axis=1)
                axial = delta @ lrn[i]
                with np.errstate(invalid='ignore', divide='ignore'):
                    cosa = np.abs(axial / np.maximum(d, 1e-9))
                off = np.sqrt(np.maximum(d ** 2 - axial ** 2, 0.0))
                ok = ((d <= c['picat_dist_max']) & (cosa >= c['picat_cos']) &
                      (off <= c['picat_offset']))
                if ok.any():
                    sc = (smoothstep(c['picat_cos'], 1.0, cosa[ok]) *
                          drusina_ramp(d[ok] - c['picat_opt'], *c['picat_ramp']))
                    _take(sc, 1, axial[ok])

    # ---- pi-cation: best LIGAND cation per protein ring -------------------
    lig_cat = np.where(lt.formal_charge > 0)[0]
    if len(lig_cat) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cull2
        idxs = np.where(gate)[0]
        LP = lig_pos[lig_cat]
        for j in idxs:
            prc, prn = pf.ring_centroid[j], pf.ring_normal[j]
            delta = LP - prc
            d = np.linalg.norm(delta, axis=1)
            axial = delta @ prn
            with np.errstate(invalid='ignore', divide='ignore'):
                cosa = np.abs(axial / np.maximum(d, 1e-9))
            off = np.sqrt(np.maximum(d ** 2 - axial ** 2, 0.0))
            ok = ((d <= c['picat_dist_max']) & (cosa >= c['picat_cos']) &
                  (off <= c['picat_offset']))
            if ok.any():
                sc = (smoothstep(c['picat_cos'], 1.0, cosa[ok]) *
                      drusina_ramp(d[ok] - c['picat_opt'], *c['picat_ramp']))
                _take(sc, 1)

    # ---- salt bridge: best ligand atom per protein charged GROUP ----------
    if len(pf.sb_centroid):
        lc = np.sign(lt.formal_charge)
        scale = np.ones(lt.n_heavy)
        if c['salt_allow_partial']:
            fb = (lc == 0) & (np.abs(lt.charge) > 0.35)
            lc = np.where(fb, np.sign(lt.charge), lc)
            scale = np.where(fb, 0.5, 1.0)
        act = np.where(lc != 0)[0]
        gate = np.sum((pf.sb_centroid - cen_l) ** 2, axis=1) <= cull2
        if len(act) and gate.any():
            bur = np.clip(pf.sb_burial * 25.0 / c['salt_burial_divisor'], 0.15, 1.0)
            for j in np.where(gate)[0]:
                d = np.linalg.norm(lig_pos[act] - pf.sb_centroid[j], axis=1)
                ok = (d <= c['salt_dist_max']) & ((lc[act] * pf.sb_sign[j]) < 0)
                if ok.any():
                    sc = (drusina_ramp(d[ok] - c['salt_opt'], *c['salt_ramp']) *
                          bur[j] * scale[act][ok])
                    _take(sc, 2)

    # ---- amide-pi: best amide per ligand ring face ------------------------
    if len(lrc) and len(pf.amide_centroid):
        gate = np.sum((pf.amide_centroid - cen_l) ** 2, axis=1) <= cull2
        if gate.any():
            amc, amn = pf.amide_centroid[gate], pf.amide_normal[gate]
            for i in range(len(lrc)):
                delta = amc - lrc[i]
                d = np.linalg.norm(delta, axis=1)
                axial = delta @ lrn[i]
                off = np.sqrt(np.maximum(d ** 2 - axial ** 2, 0.0))
                dotn = np.abs(amn @ lrn[i])
                ok = ((d >= c['amide_dist_min']) & (d <= c['amide_dist_max']) &
                      (dotn >= c['amide_cos']) & (off <= c['amide_offset']))
                if ok.any():
                    sc = (smoothstep(c['amide_cos'], 1.0, dotn[ok]) *
                          drusina_ramp(d[ok] - c['amide_opt'], *c['amide_ramp']))
                    _take(sc, 3, axial[ok])

    # ---- halogen bonds: best acceptor per halogen -------------------------
    acc = np.where(pf.is_acceptor & pcull)[0]
    if lt.halogens and len(acc):
        AP = P[acc]
        for hi, ci, et in lt.halogens:
            if et == 0:
                continue
            hp = lig_pos[hi]
            v = AP - hp
            d = np.linalg.norm(v, axis=1)
            m = (d >= c['hal_dist_min']) & (d <= c['hal_dist_max'])
            if not m.any():
                continue
            xc = lig_pos[ci] - hp
            nl = np.linalg.norm(xc)
            if nl < 1e-9:
                continue
            cost = (v[m] / d[m][:, None]) @ (xc / nl)
            ok = cost <= c['hal_cos']
            if ok.any():
                sc = (smoothstep(c['hal_cos'], -1.0, cost[ok]) *
                      drusina_ramp(d[m][ok] - c['hal_opt'][et], *c['hal_ramp']))
                _take(sc, 4)

    # ---- chalcogen bonds: best acceptor per sigma-hole, both directions ---
    if lt.chalcogens and len(acc):
        AP = P[acc]
        for si, ci in lt.chalcogens:
            sp = lig_pos[si]
            v = AP - sp
            d = np.linalg.norm(v, axis=1)
            m = (d >= c['chal_dist_min']) & (d <= c['chal_dist_max'])
            if not m.any():
                continue
            sc_ = lig_pos[ci] - sp
            nl = np.linalg.norm(sc_)
            if nl < 1e-9:
                continue
            cost = (v[m] / d[m][:, None]) @ (sc_ / nl)
            ok = cost <= c['chal_cos']
            if ok.any():
                sc = (smoothstep(c['chal_cos'], -1.0, cost[ok]) *
                      drusina_ramp(d[m][ok] - c['chal_opt'], *c['chal_ramp']))
                _take(sc, 5)
    if len(pf.chal_pos):
        lacc = np.where(np.isin(lt.vtype, [VINA_N_A, VINA_N_DA, VINA_O_A, VINA_O_DA]))[0]
        gate = np.sum((pf.chal_pos - cen_l) ** 2, axis=1) <= cull2
        if len(lacc):
            LP = lig_pos[lacc]
            for j in np.where(gate)[0]:
                sp, sdir = pf.chal_pos[j], pf.chal_dir[j]
                v = LP - sp
                d = np.linalg.norm(v, axis=1)
                m = (d >= c['chal_dist_min']) & (d <= c['chal_dist_max'])
                if not m.any():
                    continue
                cost = (v[m] / d[m][:, None]) @ sdir
                ok = cost <= c['chal_cos']
                if ok.any():
                    sc = (smoothstep(c['chal_cos'], -1.0, cost[ok]) *
                          drusina_ramp(d[m][ok] - c['chal_opt'], *c['chal_ramp']))
                    _take(sc, 5)

    # ---- metal coordination: best ligand atom per metal -------------------
    met = np.where(pf.is_metal & pcull)[0]
    if len(met):
        lm = np.where(np.isin(lt.vtype, list(METAL_COORD_LIG)))[0]
        if len(lm):
            d = np.linalg.norm(lig_pos[lm][:, None, :] - P[met][None, :, :], axis=2)
            s = np.where(d <= c['metal_cutoff'],
                         drusina_ramp(d - c['metal_opt'], *c['metal_ramp']), 0.0)
            best = s.max(axis=0)
            fired = best > 0.0
            raw[6] += float(np.sum(best[fired]))
            icount += int(fired.sum())

    # ---- per-ligand-atom protein contact counts (Coulomb burial + desolv) --
    dLP = np.linalg.norm(lig_pos[:, None, :] - P[None, pcull, :], axis=2) \
        if pcull.any() else np.zeros((lt.n_heavy, 0))
    contacts5 = np.sum(dLP < 5.0, axis=1)
    contacts45 = np.sum(dLP < 4.5, axis=1)

    # ---- screened Coulomb, burial-scaled ----------------------------------
    qp = pf.charge
    nzq = np.abs(qp) > 1e-6
    if nzq.any():
        PQ, QQ = P[nzq], qp[nzq]
        tot = 0.0
        for a in range(lt.n_heavy):
            ql = lt.charge[a]
            if abs(ql) < 1e-6:
                continue
            d = np.linalg.norm(PQ - lig_pos[a], axis=1)
            sel = d <= 12.0
            if not sel.any():
                continue
            dc = np.maximum(d[sel], 0.8)
            phi = float(np.sum(332.0 * QQ[sel] / (4.0 * dc * dc)))
            burial = (c['coulomb_burial_floor'] + (1.0 - c['coulomb_burial_floor'])
                      * min(1.0, contacts5[a] / 12.0))
            tot += ql * max(-50.0, min(50.0, phi)) * burial
        raw[7] = tot

    # ---- CH-pi: best ligand carbon per protein ring FACE -------------------
    chc = np.where((lt.vtype == VINA_C_H) & (~lt.is_aromatic))[0]
    if len(chc) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cull2
        CP = lig_pos[chc]
        for j in np.where(gate)[0]:
            prc, prn = pf.ring_centroid[j], pf.ring_normal[j]
            delta = CP - prc
            d = np.linalg.norm(delta, axis=1)
            axial = delta @ prn
            with np.errstate(invalid='ignore', divide='ignore'):
                cosa = np.abs(axial / np.maximum(d, 1e-9))
            off = np.sqrt(np.maximum(d ** 2 - axial ** 2, 0.0))
            ok = ((d >= c['chpi_dist_min']) & (d <= c['chpi_dist_max']) &
                  (cosa >= c['chpi_cos']) & (off <= c['chpi_offset']))
            if ok.any():
                sc = (smoothstep(c['chpi_cos'], 1.0, cosa[ok]) *
                      drusina_ramp(d[ok] - c['chpi_opt'], *c['chpi_ramp']))
                _take(sc, 8, axial[ok])

    # ---- torsion strain (unchanged) ---------------------------------------
    if lt.torsions:
        st = 0.0
        for a0, a1, a2, a3, k in lt.torsions:
            b1 = lig_pos[a1] - lig_pos[a0]
            b2 = lig_pos[a2] - lig_pos[a1]
            b3 = lig_pos[a3] - lig_pos[a2]
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            l1, l2 = np.linalg.norm(n1), np.linalg.norm(n2)
            if l1 < 1e-6 or l2 < 1e-6:
                continue
            phi = math.acos(max(-1.0, min(1.0, float(np.dot(n1 / l1, n2 / l2)))))
            dev = min(phi, math.pi - phi)
            st += k * dev * dev
        raw[9] = st

    # ---- H-bond directionality: best partner per ligand donor/acceptor ----
    if lt.hb_atoms and len(pf.hb_pos):
        gate = np.sum((pf.hb_pos - cen_l) ** 2, axis=1) <= cull2
        if gate.any():
            HP, HD, HA = pf.hb_pos[gate], pf.hb_is_donor[gate], pf.hb_ant[gate]
            for li, lant, ldon in lt.hb_atoms:
                sel = HD != ldon
                if not sel.any():
                    continue
                pp, pa = HP[sel], HA[sel]
                d = np.linalg.norm(pp - lig_pos[li], axis=1)
                m = (d >= c['hb_dist_min']) & (d <= c['hb_dist_max'])
                if not m.any():
                    continue
                pp, pa, d = pp[m], pa[m], d[m]
                df = drusina_ramp(d - c['hb_opt'], *c['hb_ramp'])
                lp, la = lig_pos[li], lig_pos[lant]
                if ldon:
                    dpos = np.broadcast_to(lp, pp.shape)
                    dant = np.broadcast_to(la, pp.shape)
                    apos, aant = pp, pa
                else:
                    dpos, dant = pp, pa
                    apos = np.broadcast_to(lp, pp.shape)
                    aant = np.broadcast_to(la, pp.shape)

                def _u(v):
                    return v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]

                cos_d = np.sum(_u(dant - dpos) * _u(apos - dpos), axis=1)
                cos_a = np.sum(_u(dpos - apos) * _u(aant - apos), axis=1)
                af = (smoothstep(-0.5, -1.0, cos_d) * smoothstep(0.34, -0.17, cos_a))
                ok = af > 0.25
                if ok.any():
                    _take(af[ok] * df[ok], 11)

    # ---- desolvation (unchanged form, cull-gated contact counts) ----------
    lp_idx = np.where(np.isin(lt.vtype, list(POLAR_LIG)))[0]
    if len(lp_idx) and pcull.any():
        ptypes = pf.vtype[pcull]
        tot = 0.0
        for a in lp_idx:
            lt_a = int(lt.vtype[a])
            close = dLP[a] < 3.2
            sat = any(((lt_a in DONORS and int(pt) in ACCEPTORS) or
                       (int(pt) in DONORS and lt_a in ACCEPTORS))
                      for pt in ptypes[close])
            if not sat and contacts5[a] >= 12:
                tot += min(1.0, max(0.0, (contacts5[a] - 12) / 13.0))
        raw[12] = tot
    hc = np.where(lt.vtype == VINA_C_H)[0]
    if len(hc):
        m = contacts45[hc] < 3
        raw[13] = float(np.sum(1.0 - contacts45[hc][m] / 3.0))

    raw[10] = float(max(icount - 1, 0))
    return raw, icount


def raw_terms(pf: ProteinFeatures, lt: LigandTopology, lig_pos: np.ndarray,
              want_counts: bool = False, ligand_cutoff: float = LIGAND_CUTOFF):
    """Return (raw_values[N_TERMS], interaction_count) -- or with want_counts,
    also a dict of per-term firing counts. Port of computeDrusinaCorrections.
    """
    raw = np.zeros(N_TERMS)
    counts = {t: 0 for t in TERMS}
    icount = 0
    cen_l = lig_pos.mean(axis=0)
    cut2 = ligand_cutoff * ligand_cutoff
    P = pf.pos

    lrc, lrn = _ring_frames(lig_pos, lt.rings)

    # ---- pi-pi ----
    if len(lrc) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            prc, prn = pf.ring_centroid[gate], pf.ring_normal[gate]
            d = np.linalg.norm(lrc[:, None, :] - prc[None, :, :], axis=2)
            dotn = np.abs(lrn @ prn.T)
            win = (d >= 3.2) & (d <= 5.8)
            face = win & (dotn > 0.8)
            edge = win & (dotn < 0.4)
            raw[0] += float(np.sum(dotn[face] * drusina_ramp(d[face] - 3.5, 0.3, 1.0)))
            raw[0] += float(np.sum(0.6 * (1.0 - dotn[edge]) * drusina_ramp(d[edge] - 4.8, 0.4, 1.0)))
            k = int(face.sum() + edge.sum())
            icount += k
            counts['pi_pi'] = k

    # ---- pi-cation: ligand ring vs protein cation ----
    if len(lrc) and len(pf.cation_pos):
        gate = np.sum((pf.cation_pos - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            cpos = pf.cation_pos[gate]
            v = cpos[None, :, :] - lrc[:, None, :]
            d = np.linalg.norm(v, axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('ik,ijk->ij', lrn, u))
            m = (d <= 5.5) & (cosa > 0.5)
            raw[1] += float(np.sum(cosa[m] * drusina_ramp(d[m] - 3.8, 0.4, 1.2)))
            icount += int(m.sum()); counts['pi_cation'] += int(m.sum())

    # ---- pi-cation: protein ring vs ligand cation ----
    lig_cat = np.where(lt.formal_charge > 0)[0]
    if len(lig_cat) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            prc, prn = pf.ring_centroid[gate], pf.ring_normal[gate]
            lp = lig_pos[lig_cat]
            v = lp[:, None, :] - prc[None, :, :]
            d = np.linalg.norm(v, axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('jk,ijk->ij', prn, u))
            m = (d <= 5.5) & (cosa > 0.5)
            raw[1] += float(np.sum(cosa[m] * drusina_ramp(d[m] - 3.8, 0.4, 1.2)))
            icount += int(m.sum()); counts['pi_cation'] += int(m.sum())

    # ---- salt bridge (group based, burial weighted) ----
    if len(pf.sb_centroid):
        lc = np.zeros(lt.n_heavy)
        scale = np.ones(lt.n_heavy)
        nz = lt.formal_charge != 0
        lc[nz] = np.sign(lt.formal_charge[nz])
        fb = ~nz & (np.abs(lt.charge) > 0.35)
        lc[fb] = np.sign(lt.charge[fb])
        scale[fb] = 0.5
        act = np.where(lc != 0)[0]
        if len(act):
            gate = np.sum((pf.sb_centroid - cen_l) ** 2, axis=1) <= cut2
            if gate.any():
                sbc = pf.sb_centroid[gate]
                sbs = pf.sb_sign[gate]
                sbb = pf.sb_burial[gate]
                d = np.linalg.norm(lig_pos[act][:, None, :] - sbc[None, :, :], axis=2)
                opp = (lc[act][:, None] * sbs[None, :]) < 0
                m = opp & (d <= 4.0)
                if m.any():
                    ds = drusina_ramp(d - 2.8, 0.3, 1.2)
                    contrib = ds * sbb[None, :] * scale[act][:, None]
                    raw[2] += float(np.sum(contrib[m]))
                    icount += int(m.sum()); counts['salt_bridge'] = int(m.sum())

    # ---- amide-pi ----
    if len(lrc) and len(pf.amide_centroid):
        gate = np.sum((pf.amide_centroid - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            amc, amn = pf.amide_centroid[gate], pf.amide_normal[gate]
            d = np.linalg.norm(lrc[:, None, :] - amc[None, :, :], axis=2)
            dotn = np.abs(lrn @ amn.T)
            win = (d >= 3.0) & (d <= 5.2)
            par = win & (dotn > 0.8)
            tilt = win & (dotn > 0.6) & (dotn <= 0.8)
            raw[3] += float(np.sum(dotn[par] * drusina_ramp(d[par] - 3.6, 0.3, 0.9)))
            raw[3] += float(np.sum(0.4 * dotn[tilt] * drusina_ramp(d[tilt] - 4.0, 0.3, 1.0)))
            k = int(par.sum() + tilt.sum())
            icount += k; counts['amide_pi'] = k

    # ---- halogen bonds (no centroid gate in Metal; full protein loop) ----
    acc = np.where(pf.is_acceptor)[0]
    if lt.halogens and len(acc):
        AP = P[acc]
        OPT = {1: 3.27, 2: 3.24, 3: 3.17}
        for hi, ci, et in lt.halogens:
            if et == 0:
                continue                      # F skipped
            hp, cp = lig_pos[hi], lig_pos[ci]
            v = AP - hp
            d = np.linalg.norm(v, axis=1)
            m = (d >= 2.5) & (d <= 4.2)
            if not m.any():
                continue
            xc = cp - hp
            nl = np.linalg.norm(xc)
            if nl < 1e-9:
                continue
            xc /= nl
            u = v[m] / d[m][:, None]
            cost = u @ xc
            ok = cost < -0.766
            if not ok.any():
                continue
            af = (-cost[ok] - 0.766) / 0.234
            raw[4] += float(np.sum(af * drusina_ramp(d[m][ok] - OPT[et], 0.3, 0.8)))
            icount += int(ok.sum()); counts['halogen_bond'] += int(ok.sum())

    # ---- ligand chalcogen bonds ----
    if lt.chalcogens and len(acc):
        AP = P[acc]
        for si, ci in lt.chalcogens:
            sp, cp = lig_pos[si], lig_pos[ci]
            v = AP - sp
            d = np.linalg.norm(v, axis=1)
            m = (d >= 2.8) & (d <= 4.2)
            if not m.any():
                continue
            sc = cp - sp
            nl = np.linalg.norm(sc)
            if nl < 1e-9:
                continue
            sc /= nl
            u = v[m] / d[m][:, None]
            cost = u @ sc
            ok = cost < -0.766
            if not ok.any():
                continue
            af = (-cost[ok] - 0.766) / 0.234
            raw[5] += float(np.sum(af * drusina_ramp(d[m][ok] - 3.3, 0.3, 0.8)))
            icount += int(ok.sum()); counts['chalcogen_bond'] += int(ok.sum())

    # ---- protein chalcogen bonds (Met/Cys S -> ligand N/O) ----
    if len(pf.chal_pos):
        lacc = np.where(np.isin(lt.vtype, [VINA_N_A, VINA_N_DA, VINA_O_A, VINA_O_DA]))[0]
        if len(lacc):
            LP = lig_pos[lacc]
            v = LP[:, None, :] - pf.chal_pos[None, :, :]
            d = np.linalg.norm(v, axis=2)
            m = (d >= 2.8) & (d <= 4.2)
            if m.any():
                with np.errstate(invalid='ignore', divide='ignore'):
                    u = v / np.maximum(d, 1e-9)[:, :, None]
                cost = np.einsum('jk,ijk->ij', pf.chal_dir, u)
                ok = m & (cost < -0.766)
                if ok.any():
                    af = (-cost[ok] - 0.766) / 0.234
                    raw[5] += float(np.sum(af * drusina_ramp(d[ok] - 3.3, 0.3, 0.8)))
                    icount += int(ok.sum()); counts['chalcogen_bond'] += int(ok.sum())

    # ---- metal coordination (best per metal, min of negative weighted score) ----
    met = np.where(pf.is_metal)[0]
    if len(met):
        lm = np.where(np.isin(lt.vtype, list(METAL_COORD_LIG)))[0]
        if len(lm):
            d = np.linalg.norm(lig_pos[lm][:, None, :] - P[met][None, :, :], axis=2)
            s = np.where(d <= 4.5, drusina_ramp(d - 2.4, 0.4, 1.6), 0.0)
            best = s.max(axis=0)
            fired = best > 0.0
            raw[6] += float(np.sum(best[fired]))
            icount += int(fired.sum()); counts['metal_coord'] = int(fired.sum())

    # ---- screened Coulomb (direct sum form of computeElectrostaticGrid) ----
    qp = pf.charge
    nzq = np.abs(qp) > 1e-6
    if nzq.any():
        PQ, QQ = P[nzq], qp[nzq]
        tot = 0.0
        for a in range(lt.n_heavy):
            ql = lt.charge[a]
            if abs(ql) < 1e-6:
                continue
            d = np.linalg.norm(PQ - lig_pos[a], axis=1)
            sel = d <= 12.0
            if not sel.any():
                continue
            dc = np.maximum(d[sel], 0.8)
            phi = float(np.sum(332.0 * QQ[sel] / (4.0 * dc * dc)))
            tot += ql * max(-50.0, min(50.0, phi))
        raw[7] = tot

    # ---- CH-pi ----
    chc = np.where((lt.vtype == VINA_C_H) & (~lt.is_aromatic))[0]
    if len(chc) and len(pf.ring_centroid):
        gate = np.sum((pf.ring_centroid - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            prc, prn = pf.ring_centroid[gate], pf.ring_normal[gate]
            v = lig_pos[chc][:, None, :] - prc[None, :, :]
            d = np.linalg.norm(v, axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                u = v / np.maximum(d, 1e-9)[:, :, None]
            cosa = np.abs(np.einsum('jk,ijk->ij', prn, u))
            m = (d >= 3.5) & (d <= 5.0) & (cosa > 0.55)
            raw[8] += float(np.sum(cosa[m] ** 2 * drusina_ramp(d[m] - 4.0, 0.5, 1.0)))
            icount += int(m.sum()); counts['ch_pi'] = int(m.sum())

    # ---- torsion strain (amide planarity) ----
    if lt.torsions:
        s = 0.0
        for a0, a1, a2, a3, k in lt.torsions:
            b1 = lig_pos[a1] - lig_pos[a0]
            b2 = lig_pos[a2] - lig_pos[a1]
            b3 = lig_pos[a3] - lig_pos[a2]
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            l1, l2 = np.linalg.norm(n1), np.linalg.norm(n2)
            if l1 < 1e-6 or l2 < 1e-6:
                continue
            phi = math.acos(max(-1.0, min(1.0, float(np.dot(n1 / l1, n2 / l2)))))
            dev = min(phi, math.pi - phi)
            s += k * dev * dev
        raw[9] = s
        counts['torsion_strain'] = len(lt.torsions)

    # ---- H-bond directionality (bonus only) ----
    if lt.hb_atoms and len(pf.hb_pos):
        gate = np.sum((pf.hb_pos - cen_l) ** 2, axis=1) <= cut2
        if gate.any():
            HP, HD, HA = pf.hb_pos[gate], pf.hb_is_donor[gate], pf.hb_ant[gate]
            tot, k = 0.0, 0
            for li, lant, ldon in lt.hb_atoms:
                sel = HD != ldon
                if not sel.any():
                    continue
                pp, pa = HP[sel], HA[sel]
                d = np.linalg.norm(pp - lig_pos[li], axis=1)
                m = (d > 1.5) & (d <= 3.2)
                if not m.any():
                    continue
                pp, pa, d = pp[m], pa[m], d[m]
                df = drusina_ramp(d - 2.7, 0.2, 0.5)
                lp, la = lig_pos[li], lig_pos[lant]
                if ldon:
                    dpos, dant, apos, aant = lp, la, pp, pa
                    dpos = np.broadcast_to(dpos, pp.shape)
                    dant = np.broadcast_to(dant, pp.shape)
                else:
                    dpos, dant = pp, pa
                    apos = np.broadcast_to(lp, pp.shape)
                    aant = np.broadcast_to(la, pp.shape)

                def _u(v):
                    return v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]

                cos_d = np.sum(_u(dant - dpos) * _u(apos - dpos), axis=1)
                ds = smoothstep(-0.5, -1.0, cos_d)
                cos_a = np.sum(_u(dpos - apos) * _u(aant - apos), axis=1)
                asc = smoothstep(0.34, -0.17, cos_a)
                af = ds * asc
                ok = af > 0.25
                if ok.any():
                    tot += float(np.sum(af[ok] * df[ok]))
                    k += int(ok.sum())
            raw[11] = tot
            icount += k; counts['hbond_dir'] = k

    # ---- polar desolvation ----
    lp_idx = np.where(np.isin(lt.vtype, list(POLAR_LIG)))[0]
    if len(lp_idx):
        d = np.linalg.norm(lig_pos[lp_idx][:, None, :] - P[None, :, :], axis=2)
        nearby = np.sum(d < 5.0, axis=1)
        close = d < 3.2
        tot, k = 0.0, 0
        for r, a in enumerate(lp_idx):
            lt_a = int(lt.vtype[a])
            pts = pf.vtype[close[r]]
            sat = False
            for pt in pts:
                pt = int(pt)
                if ((lt_a in DONORS and pt in ACCEPTORS) or
                        (pt in DONORS and lt_a in ACCEPTORS)):
                    sat = True
                    break
            if not sat and nearby[r] >= 12:
                tot += min(1.0, max(0.0, (nearby[r] - 12) / 13.0))
                k += 1
        raw[12] = tot
        counts['desolv_polar'] = k

    # ---- hydrophobic desolvation ----
    hc = np.where(lt.vtype == VINA_C_H)[0]
    if len(hc):
        d = np.linalg.norm(lig_pos[hc][:, None, :] - P[None, :, :], axis=2)
        nearby = np.sum(d < 4.5, axis=1)
        m = nearby < 3
        raw[13] = float(np.sum(1.0 - nearby[m] / 3.0))
        counts['desolv_hydrophobic'] = int(m.sum())

    # ---- cooperativity ----
    raw[10] = float(max(icount - 1, 0))
    counts['cooperativity'] = max(icount - 1, 0)

    if want_counts:
        return raw, icount, counts
    return raw, icount


def vina_energy(pf: ProteinFeatures, lt: LigandTopology, lig_pos: np.ndarray,
                normalize: bool = True):
    """Explicit-pair Vina intermolecular energy (scorePosesExplicit path)."""
    d = np.linalg.norm(lig_pos[:, None, :] - pf.pos[None, :, :], axis=2)
    near = d < 8.0
    if not near.any():
        return 0.0
    li, pi = np.nonzero(near)
    r = d[near]
    t1 = lt.vtype[li]
    t2 = pf.vtype[pi]
    valid = (t1 <= MAX_VINA_TYPE) & (t2 <= MAX_VINA_TYPE)
    li, pi, r, t1, t2 = li[valid], pi[valid], r[valid], t1[valid], t2[valid]
    if len(r) == 0:
        return 0.0
    opt = XS_RADII[t1] + XS_RADII[t2]
    dd = r - opt
    e = W_GAUSS1 * np.exp(-4.0 * dd * dd)
    e += W_GAUSS2 * np.exp(-(((dd - 3.0) / 2.0) ** 2))
    e += np.where(dd < 0, W_REPULSION * dd * dd, 0.0)
    # Boolean lookup tables indexed by type id: np.isin over a few hundred
    # thousand pairs was the single slowest line in the whole pipeline.
    hydro = _IS_HYDROPHOBIC[t1] & _IS_HYDROPHOBIC[t2]
    e += np.where(hydro, W_HYDROPHOBIC * np.clip((dd - 1.5) / (0.5 - 1.5), 0, 1), 0.0)
    hb = ((_IS_DONOR[t1] & _IS_ACCEPTOR[t2]) |
          (_IS_DONOR[t2] & _IS_ACCEPTOR[t1]))
    e += np.where(hb, W_HBOND * np.clip((dd - 0.0) / (-0.7 - 0.0), 0, 1), 0.0)
    total = float(np.sum(e))
    if normalize:
        total *= vina_rot_normalization(lt.n_rot)
    return total

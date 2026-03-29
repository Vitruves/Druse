#!/usr/bin/env python3
"""
Calibrate Drusina scoring weights on PDBbind refined set (5316 complexes).

Reimplements Drusina geometric calculations in Python/numpy to score crystal
poses directly — no docking needed. Fits 14 linear weights to maximize
scoring power (Pearson r between total energy and experimental pKd).

This is classical scoring function calibration (like AutoDock Vina's 5 weights),
NOT machine learning. The functional forms are fixed; only weights are optimized.

Usage:
  python Benchmark/calibrate_drusina_pdbbind.py                    # full run
  python Benchmark/calibrate_drusina_pdbbind.py --max-complexes 500  # quick test
  python Benchmark/calibrate_drusina_pdbbind.py --cv 5              # cross-validate
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Optional but strongly recommended
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
REFINED_SET = ROOT / "Benchmark" / "data" / "refined-set"
INDEX_FILE = REFINED_SET / "index" / "INDEX_refined_data.2020"

# ============================================================================
# Vina atom typing (simplified — matches Metal/DockingCompute.metal)
# ============================================================================

# Vina XS atom types
VINA_C_H = 0   # hydrophobic carbon
VINA_C_P = 1   # polar carbon
VINA_N_A = 2   # acceptor nitrogen
VINA_N_D = 3   # donor nitrogen
VINA_N_DA = 4  # donor+acceptor nitrogen
VINA_N_P = 5   # polar nitrogen (other)
VINA_O_A = 6   # acceptor oxygen
VINA_O_D = 7   # donor oxygen
VINA_O_DA = 8  # donor+acceptor oxygen
VINA_O_P = 9   # polar oxygen
VINA_S_P = 10  # sulfur
VINA_P_P = 11  # phosphorus
VINA_F_H = 12  # fluorine
VINA_Cl_H = 13 # chlorine
VINA_Br_H = 14 # bromine
VINA_I_H = 15  # iodine
VINA_MET_D = 16 # metal
VINA_OTHER = 17

DONORS = {VINA_N_D, VINA_N_DA, VINA_O_D, VINA_O_DA}
ACCEPTORS = {VINA_N_A, VINA_N_DA, VINA_O_A, VINA_O_DA}
HBOND_TYPES = DONORS | ACCEPTORS

VDW_RADII = {
    'C': 1.9, 'N': 1.8, 'O': 1.7, 'S': 2.0, 'P': 2.1,
    'F': 1.5, 'Cl': 1.8, 'Br': 2.0, 'I': 2.2,
    'Zn': 1.2, 'Fe': 1.2, 'Mg': 1.2, 'Ca': 1.2, 'Mn': 1.2,
    'Cu': 1.2, 'Co': 1.2, 'Ni': 1.2, 'H': 1.0
}


def assign_vina_type_rdkit(atom, mol):
    """Assign Vina XS atom type from RDKit atom."""
    elem = atom.GetSymbol()
    if elem == 'C':
        # Check if bonded to any heteroatom
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() in ('N', 'O', 'S', 'P'):
                return VINA_C_P
        return VINA_C_H
    elif elem == 'N':
        hcount = atom.GetTotalNumHs()
        has_lp = True  # simplified
        if hcount > 0 and has_lp:
            return VINA_N_DA
        elif hcount > 0:
            return VINA_N_D
        elif has_lp:
            return VINA_N_A
        return VINA_N_P
    elif elem == 'O':
        hcount = atom.GetTotalNumHs()
        if hcount > 0:
            return VINA_O_DA
        return VINA_O_A
    elif elem == 'S':
        return VINA_S_P
    elif elem == 'P':
        return VINA_P_P
    elif elem == 'F':
        return VINA_F_H
    elif elem == 'Cl':
        return VINA_Cl_H
    elif elem == 'Br':
        return VINA_Br_H
    elif elem == 'I':
        return VINA_I_H
    elif elem in ('Zn', 'Fe', 'Mg', 'Ca', 'Mn', 'Cu', 'Co', 'Ni'):
        return VINA_MET_D
    return VINA_OTHER


def assign_vina_type_pdb(atom_name: str, res_name: str, element: str) -> int:
    """Assign Vina XS type from PDB atom/residue names."""
    elem = element.strip()
    name = atom_name.strip()
    res = res_name.strip()

    if elem in ('ZN', 'FE', 'MG', 'CA', 'MN', 'CU', 'CO', 'NI'):
        return VINA_MET_D
    if elem == 'C':
        # Backbone C is polar (bonded to O)
        if name in ('C', 'CA'):
            return VINA_C_P
        return VINA_C_H
    if elem == 'N':
        if name == 'N':  # backbone N — donor
            return VINA_N_D
        if res in ('HIS',):
            return VINA_N_DA
        if res in ('LYS',) and name == 'NZ':
            return VINA_N_D
        if res in ('ARG',) and name in ('NH1', 'NH2', 'NE'):
            return VINA_N_D
        if res in ('ASN',) and name == 'ND2':
            return VINA_N_DA
        if res in ('GLN',) and name == 'NE2':
            return VINA_N_DA
        if res in ('TRP',) and name == 'NE1':
            return VINA_N_D
        return VINA_N_A
    if elem == 'O':
        if name == 'O':  # backbone O — acceptor
            return VINA_O_A
        if res in ('SER',) and name == 'OG':
            return VINA_O_DA
        if res in ('THR',) and name == 'OG1':
            return VINA_O_DA
        if res in ('TYR',) and name == 'OH':
            return VINA_O_DA
        if res in ('ASP',) and name in ('OD1', 'OD2'):
            return VINA_O_A
        if res in ('GLU',) and name in ('OE1', 'OE2'):
            return VINA_O_A
        if res in ('ASN',) and name == 'OD1':
            return VINA_O_A
        if res in ('GLN',) and name == 'OE1':
            return VINA_O_A
        return VINA_O_A
    if elem == 'S':
        return VINA_S_P
    return VINA_OTHER


# ============================================================================
# Vina scoring (pair energy — matches DockingCompute.metal)
# ============================================================================

XS_RADII = [1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7,
            2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2]

W_GAUSS1 = -0.035579
W_GAUSS2 = -0.005156
W_REPULSION = 0.840245
W_HYDROPHOBIC = -0.035069
W_HBOND = -0.587439
W_ROT_ENTROPY = 0.05846


def slope_step(x_bad, x_good, x):
    """Piecewise linear ramp (same as Metal slopeStep)."""
    if x_bad < x_good:
        if x <= x_bad: return 0.0
        if x >= x_good: return 1.0
    else:
        if x >= x_bad: return 0.0
        if x <= x_good: return 1.0
    return (x - x_bad) / (x_good - x_bad)


def hbond_possible(t1, t2):
    return (t1 in DONORS and t2 in ACCEPTORS) or (t2 in DONORS and t1 in ACCEPTORS)


def vina_pair_energy(t1, t2, r):
    """Compute Vina pair energy between two atom types at distance r."""
    if t1 >= len(XS_RADII) or t2 >= len(XS_RADII) or r >= 8.0:
        return 0.0
    optimal = XS_RADII[t1] + XS_RADII[t2]
    d = r - optimal
    e = W_GAUSS1 * np.exp(-4.0 * d * d)
    e += W_GAUSS2 * np.exp(-((d - 3.0) / 2.0) ** 2)
    if d < 0:
        e += W_REPULSION * d * d
    # Hydrophobic
    if t1 in (VINA_C_H,) and t2 in (VINA_C_H,):
        e += W_HYDROPHOBIC * slope_step(1.5, 0.5, d)
    # H-bond
    if hbond_possible(t1, t2):
        e += W_HBOND * slope_step(0.0, -0.7, d)
    return e


# ============================================================================
# Drusina term calculations (reimplemented from Metal shader)
# ============================================================================

def drusina_ramp(x, inner, outer):
    """Block/ramp function (same as Metal drusinaRamp)."""
    ax = abs(x)
    if ax <= inner:
        return 1.0
    if ax >= outer:
        return 0.0
    return (outer - ax) / (outer - inner)


def detect_aromatic_rings(mol):
    """Detect aromatic rings, return list of (centroid, normal, atom_indices)."""
    rings = []
    ri = mol.GetRingInfo()
    for ring_atoms in ri.AtomRings():
        if len(ring_atoms) not in (5, 6):
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms):
            continue
        conf = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in ring_atoms])
        centroid = positions.mean(axis=0)
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        normal = np.cross(v1, v2)
        nlen = np.linalg.norm(normal)
        if nlen < 1e-6:
            continue
        normal /= nlen
        rings.append({'centroid': centroid, 'normal': normal, 'indices': ring_atoms})
    return rings


def compute_drusina_terms(prot_atoms, lig_atoms, lig_mol):
    """
    Compute raw (unweighted) Drusina term values for a protein-ligand complex.

    Returns dict of term_name -> raw_value (before multiplication by weight).
    """
    terms = defaultdict(float)

    prot_pos = np.array([a['pos'] for a in prot_atoms])
    lig_pos = np.array([a['pos'] for a in lig_atoms])

    if len(lig_pos) == 0 or len(prot_pos) == 0:
        return dict(terms)

    lig_centroid = lig_pos.mean(axis=0)
    cutoff_sq = 100.0  # 10Å squared

    # --- Detect ligand aromatic rings ---
    lig_rings = detect_aromatic_rings(lig_mol) if lig_mol else []

    # --- Detect protein aromatic rings (from residue names) ---
    prot_rings = []
    aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
    ring_atoms_by_res = defaultdict(list)
    for a in prot_atoms:
        res = a.get('res_name', '').strip()
        if res in aromatic_residues:
            name = a.get('atom_name', '').strip()
            key = f"{a.get('chain', '')}{a.get('res_seq', '')}"
            # Aromatic ring atoms by residue
            arom_names = {
                'PHE': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
                'TYR': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
                'TRP': {'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
                'HIS': {'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
            }
            if name in arom_names.get(res, set()):
                ring_atoms_by_res[key].append(a['pos'])

    for key, positions in ring_atoms_by_res.items():
        if len(positions) >= 4:
            pos = np.array(positions)
            centroid = pos.mean(axis=0)
            v1 = pos[1] - pos[0]
            v2 = pos[2] - pos[0]
            normal = np.cross(v1, v2)
            nlen = np.linalg.norm(normal)
            if nlen > 1e-6:
                prot_rings.append({'centroid': centroid, 'normal': normal / nlen})

    # --- π-π stacking ---
    for lr in lig_rings:
        for pr in prot_rings:
            if np.sum((lig_centroid - pr['centroid'])**2) > cutoff_sq:
                continue
            d = np.linalg.norm(lr['centroid'] - pr['centroid'])
            if d < 3.2 or d > 5.8:
                continue
            dotN = abs(np.dot(lr['normal'], pr['normal']))
            if dotN > 0.8:
                dd = d - 3.5
                terms['pi_pi'] += dotN * drusina_ramp(dd, 0.3, 1.0)
            elif dotN < 0.4:
                dd = d - 4.8
                perpFactor = 1.0 - dotN
                terms['pi_pi'] += 0.6 * perpFactor * drusina_ramp(dd, 0.4, 1.0)

    # --- π-cation ---
    prot_cation_names = {'NZ', 'NH1', 'NH2'}
    prot_cations = [a for a in prot_atoms
                    if a.get('atom_name', '').strip() in prot_cation_names
                    or a.get('element', '').strip() in ('ZN', 'FE', 'MG', 'CA')]
    for lr in lig_rings:
        for pc in prot_cations:
            pos = pc['pos']
            if np.sum((lig_centroid - pos)**2) > cutoff_sq:
                continue
            d = np.linalg.norm(lr['centroid'] - pos)
            if d > 5.5:
                continue
            toAtom = (pos - lr['centroid'])
            toAtom /= max(np.linalg.norm(toAtom), 1e-8)
            cosA = abs(np.dot(toAtom, lr['normal']))
            if cosA > 0.5:
                dd = d - 3.8
                terms['pi_cation'] += cosA * drusina_ramp(dd, 0.4, 1.2)

    # --- Salt bridge ---
    charged_groups = []
    res_atoms_sb = defaultdict(list)
    for a in prot_atoms:
        key = f"{a.get('chain', '')}{a.get('res_seq', '')}"
        res_atoms_sb[key].append(a)

    for key, atoms in res_atoms_sb.items():
        res = atoms[0].get('res_name', '').strip()
        if res == 'ARG':
            guan = [a for a in atoms if a.get('atom_name', '').strip() in ('NE', 'NH1', 'NH2', 'CZ')]
            if len(guan) >= 2:
                centroid = np.mean([a['pos'] for a in guan], axis=0)
                charged_groups.append({'centroid': centroid, 'sign': 1})
        elif res == 'LYS':
            nz = [a for a in atoms if a.get('atom_name', '').strip() == 'NZ']
            if nz:
                charged_groups.append({'centroid': nz[0]['pos'], 'sign': 1})
        elif res == 'ASP':
            od = [a for a in atoms if a.get('atom_name', '').strip() in ('OD1', 'OD2')]
            if len(od) >= 2:
                centroid = np.mean([a['pos'] for a in od], axis=0)
                charged_groups.append({'centroid': centroid, 'sign': -1})
        elif res == 'GLU':
            oe = [a for a in atoms if a.get('atom_name', '').strip() in ('OE1', 'OE2')]
            if len(oe) >= 2:
                centroid = np.mean([a['pos'] for a in oe], axis=0)
                charged_groups.append({'centroid': centroid, 'sign': -1})

    for la in lig_atoms:
        fc = la.get('formal_charge', 0)
        if fc == 0:
            continue
        for cg in charged_groups:
            if (fc > 0 and cg['sign'] > 0) or (fc < 0 and cg['sign'] < 0):
                continue
            d = np.linalg.norm(la['pos'] - cg['centroid'])
            if d > 4.0:
                continue
            dd = d - 2.8
            terms['salt_bridge'] += drusina_ramp(dd, 0.3, 1.2)

    # --- Metal coordination ---
    metals = [a for a in prot_atoms if a['vina_type'] == VINA_MET_D]
    for met in metals:
        best = 0.0
        for la in lig_atoms:
            if la['vina_type'] not in (VINA_N_A, VINA_N_DA, VINA_N_D, VINA_O_A, VINA_O_DA, VINA_S_P):
                continue
            d = np.linalg.norm(la['pos'] - met['pos'])
            if d > 4.5:
                continue
            dd = d - 2.4
            score = drusina_ramp(dd, 0.4, 1.6)
            best = max(best, score)
        terms['metal_coord'] += best

    # --- CH-π ---
    for la in lig_atoms:
        if la['vina_type'] != VINA_C_H:
            continue
        if la.get('is_aromatic', False):
            continue
        for pr in prot_rings:
            if np.sum((lig_centroid - pr['centroid'])**2) > cutoff_sq:
                continue
            d = np.linalg.norm(la['pos'] - pr['centroid'])
            if d < 3.5 or d > 5.0:
                continue
            toC = (la['pos'] - pr['centroid'])
            toC /= max(np.linalg.norm(toC), 1e-8)
            cosA = abs(np.dot(toC, pr['normal']))
            if cosA > 0.55:
                dd = d - 4.0
                terms['ch_pi'] += cosA * cosA * drusina_ramp(dd, 0.5, 1.0)

    # --- H-bond directionality (bonus only) ---
    for la in lig_atoms:
        lt = la['vina_type']
        if lt not in HBOND_TYPES:
            continue
        lig_is_donor = lt in DONORS
        ant_pos = la.get('antecedent_pos')
        if ant_pos is None:
            continue

        for pa in prot_atoms:
            pt = pa['vina_type']
            if pt not in HBOND_TYPES:
                continue
            prot_is_donor = pt in DONORS
            if lig_is_donor == prot_is_donor:
                continue
            d = np.linalg.norm(la['pos'] - pa['pos'])
            if d < 1.5 or d > 3.2:
                continue
            if np.sum((lig_centroid - pa['pos'])**2) > cutoff_sq:
                continue

            prot_ant = pa.get('antecedent_pos')
            if prot_ant is None:
                continue

            dist_factor = drusina_ramp(d - 2.7, 0.2, 0.5)

            if lig_is_donor:
                donor_pos, donor_ant, acc_pos, acc_ant = la['pos'], ant_pos, pa['pos'], prot_ant
            else:
                donor_pos, donor_ant, acc_pos, acc_ant = pa['pos'], prot_ant, la['pos'], ant_pos

            d2ant = donor_ant - donor_pos
            d2ant /= max(np.linalg.norm(d2ant), 1e-8)
            d2a = acc_pos - donor_pos
            d2a /= max(np.linalg.norm(d2a), 1e-8)
            cos_donor = np.dot(d2ant, d2a)
            donor_score = np.clip((cos_donor - (-0.5)) / (-1.0 - (-0.5)), 0, 1)  # smoothstep approx

            a2d = donor_pos - acc_pos
            a2d /= max(np.linalg.norm(a2d), 1e-8)
            a2ant = acc_ant - acc_pos
            a2ant /= max(np.linalg.norm(a2ant), 1e-8)
            cos_acc = np.dot(a2d, a2ant)
            acc_score = np.clip((0.34 - cos_acc) / (0.34 - (-0.17)), 0, 1)

            angle_factor = donor_score * acc_score
            if angle_factor > 0.25:
                terms['hbond_dir'] += angle_factor * dist_factor

    # --- Polar desolvation ---
    for la in lig_atoms:
        lt = la['vina_type']
        if lt not in (VINA_N_A, VINA_N_D, VINA_N_DA, VINA_O_A, VINA_O_D, VINA_O_DA):
            continue
        dists = np.linalg.norm(prot_pos - la['pos'], axis=1)
        nearby = np.sum(dists < 5.0)
        hbond_sat = False
        for pa in prot_atoms:
            if np.linalg.norm(la['pos'] - pa['pos']) < 3.2:
                if hbond_possible(lt, pa['vina_type']):
                    hbond_sat = True
                    break
        if not hbond_sat and nearby >= 12:
            burial = np.clip((nearby - 12) / 13.0, 0, 1)
            terms['desolv_polar'] += burial

    # --- Hydrophobic desolvation ---
    for la in lig_atoms:
        if la['vina_type'] != VINA_C_H:
            continue
        dists = np.linalg.norm(prot_pos - la['pos'], axis=1)
        nearby = np.sum(dists < 4.5)
        if nearby < 3:
            exposure = 1.0 - nearby / 3.0
            terms['desolv_hydrophobic'] += exposure

    # --- Coulomb (simplified: direct sum, ε=4r) ---
    coulomb = 0.0
    for la in lig_atoms:
        q_lig = la.get('charge', 0.0)
        if abs(q_lig) < 0.01:
            continue
        for pa in prot_atoms:
            q_prot = pa.get('charge', 0.0)
            if abs(q_prot) < 0.01:
                continue
            d = np.linalg.norm(la['pos'] - pa['pos'])
            if d < 1.0 or d > 10.0:
                continue
            coulomb += 332.0 * q_lig * q_prot / (4.0 * d * d)
    terms['coulomb'] = coulomb

    # Placeholder terms (would need more complex detection)
    # amide_pi, halogen_bond, chalcogen_bond, torsion_strain, cooperativity
    # are left at 0 unless we detect them

    return dict(terms)


# ============================================================================
# Data loading
# ============================================================================

def parse_index(index_path: Path) -> dict[str, float]:
    """Parse PDBbind index file → {pdb_id: pKd}."""
    pkd_map = {}
    with open(index_path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdb_id = parts[0].strip()
                try:
                    pkd = float(parts[3])
                    pkd_map[pdb_id] = pkd
                except ValueError:
                    continue
    return pkd_map


def parse_pdb_atoms(pdb_path: str, pocket_radius: float = 12.0,
                    center: np.ndarray | None = None) -> list[dict]:
    """Parse protein PDB, return atoms near center (or all if center is None)."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            elem = line[76:78].strip() if len(line) > 77 else line[12:16].strip()[0]
            if elem == 'H' or elem == '':
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21:22].strip()
            try:
                res_seq = int(line[22:26])
            except ValueError:
                res_seq = 0

            pos = np.array([x, y, z])
            if center is not None:
                if np.sum((pos - center)**2) > pocket_radius**2:
                    continue

            vt = assign_vina_type_pdb(atom_name, res_name, elem)
            atoms.append({
                'pos': pos,
                'atom_name': atom_name,
                'res_name': res_name,
                'chain': chain,
                'res_seq': res_seq,
                'element': elem,
                'vina_type': vt,
                'charge': 0.0,  # simplified
                'antecedent_pos': None,  # filled below for H-bond atoms
            })

    # Fill antecedent positions for H-bond atoms (backbone N→CA, O→C, etc.)
    res_map = defaultdict(list)
    for a in atoms:
        key = f"{a['chain']}_{a['res_seq']}"
        res_map[key].append(a)

    for key, res_atoms in res_map.items():
        by_name = {a['atom_name']: a for a in res_atoms}
        # Backbone N → CA
        if 'N' in by_name and 'CA' in by_name:
            by_name['N']['antecedent_pos'] = by_name['CA']['pos']
        # Backbone O → C
        if 'O' in by_name and 'C' in by_name:
            by_name['O']['antecedent_pos'] = by_name['C']['pos']
        # Sidechain (simplified — use CB as antecedent for most)
        res = res_atoms[0]['res_name']
        sc_donors = {
            'SER': [('OG', 'CB')], 'THR': [('OG1', 'CB')], 'TYR': [('OH', 'CZ')],
            'ASN': [('ND2', 'CG'), ('OD1', 'CG')], 'GLN': [('NE2', 'CD'), ('OE1', 'CD')],
            'ASP': [('OD1', 'CG'), ('OD2', 'CG')], 'GLU': [('OE1', 'CD'), ('OE2', 'CD')],
            'HIS': [('ND1', 'CG'), ('NE2', 'CE1')],
            'LYS': [('NZ', 'CE')], 'ARG': [('NH1', 'CZ'), ('NH2', 'CZ'), ('NE', 'CD')],
            'TRP': [('NE1', 'CE2')], 'CYS': [('SG', 'CB')],
        }
        for atom_name, ant_name in sc_donors.get(res, []):
            if atom_name in by_name and ant_name in by_name:
                by_name[atom_name]['antecedent_pos'] = by_name[ant_name]['pos']

    return atoms


def parse_ligand_sdf(sdf_path: str) -> tuple[list[dict], object | None]:
    """Parse ligand SDF with RDKit, return atom list and mol object."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    mol = next(suppl, None)
    if mol is None:
        return [], None

    conf = mol.GetConformer()
    atoms = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'H':
            continue
        pos = np.array(conf.GetAtomPosition(i))
        vt = assign_vina_type_rdkit(atom, mol)

        # Find antecedent (first bonded heavy atom)
        ant_pos = None
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() != 'H':
                ant_pos = np.array(conf.GetAtomPosition(nb.GetIdx()))
                if nb.GetSymbol() == 'C':
                    break  # prefer carbon

        atoms.append({
            'pos': pos,
            'vina_type': vt,
            'formal_charge': atom.GetFormalCharge(),
            'charge': 0.0,
            'is_aromatic': atom.GetIsAromatic(),
            'antecedent_pos': ant_pos,
        })

    return atoms, mol


def compute_vina_score(prot_atoms, lig_atoms, n_torsions):
    """Compute Vina intermolecular energy (simplified — no grid)."""
    total = 0.0
    for la in lig_atoms:
        lt = la['vina_type']
        for pa in prot_atoms:
            pt = pa['vina_type']
            r = np.linalg.norm(la['pos'] - pa['pos'])
            if r < 8.0:
                total += vina_pair_energy(lt, pt, r)
    # Normalization
    norm_factor = 1.0 / (1.0 + W_ROT_ENTROPY * n_torsions / 5.0)
    return total * norm_factor


# ============================================================================
# Main calibration
# ============================================================================

TERMS = [
    "pi_pi", "pi_cation", "salt_bridge", "amide_pi",
    "halogen_bond", "chalcogen_bond", "metal_coord", "coulomb",
    "ch_pi", "torsion_strain", "cooperativity",
    "hbond_dir", "desolv_polar", "desolv_hydrophobic",
]

DEFAULT_WEIGHTS = np.array([
    -0.20,   # pi_pi
    -0.50,   # pi_cation
    -0.20,   # salt_bridge
    -0.15,   # amide_pi
    -0.40,   # halogen_bond
    -0.10,   # chalcogen_bond
    -0.95,   # metal_coord
     0.015,  # coulomb
    -0.04,   # ch_pi
     1.0,    # torsion_strain
     0.0,    # cooperativity
    -0.25,   # hbond_dir
     0.15,   # desolv_polar
     0.10,   # desolv_hydrophobic
])

BOUNDS = [
    (-1.0, 0.0),   # pi_pi
    (-2.0, 0.0),   # pi_cation
    (-1.0, 0.0),   # salt_bridge
    (-0.5, 0.0),   # amide_pi
    (-1.5, 0.0),   # halogen_bond
    (-0.5, 0.0),   # chalcogen_bond
    (-3.0, 0.0),   # metal_coord
    (-0.2, 0.2),   # coulomb
    (-0.3, 0.0),   # ch_pi
    (0.0, 5.0),    # torsion_strain
    (-0.5, 0.5),   # cooperativity
    (-1.0, 0.0),   # hbond_dir
    (0.0, 0.5),    # desolv_polar
    (0.0, 0.3),    # desolv_hydrophobic
]


def _score_one_complex(args_tuple, pocket_radius=12.0, skip_vina=False):
    """Worker function for multiprocessing."""
    pdb_id, pkd, prot_path, lig_path = args_tuple
    try:
        lig_atoms, lig_mol = parse_ligand_sdf(lig_path)
        if not lig_atoms:
            return None
        lig_center = np.mean([a['pos'] for a in lig_atoms], axis=0)
        prot_atoms = parse_pdb_atoms(prot_path, pocket_radius=pocket_radius,
                                     center=lig_center)
        if not prot_atoms:
            return None
        dterms = compute_drusina_terms(prot_atoms, lig_atoms, lig_mol)
        raw = np.array([dterms.get(t, 0.0) for t in TERMS])
        if skip_vina:
            vina_e = 0.0
        else:
            n_rot = Chem.Descriptors.NumRotatableBonds(lig_mol) if lig_mol else 0
            vina_e = compute_vina_score(prot_atoms, lig_atoms, n_rot)
        return (pdb_id, pkd, raw, vina_e)
    except Exception:
        return None


def objective(weights, raw_terms, vina_base, pkd, reg=0.01):
    drusina = raw_terms @ weights
    total = vina_base + drusina
    r, _ = pearsonr(total, pkd)
    return r + reg * np.sum(weights**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-complexes", type=int, default=0, help="Limit complexes (0=all)")
    parser.add_argument("--cv", type=int, default=0, help="Cross-validation folds")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pocket-radius", type=float, default=12.0, help="Pocket radius (Å)")
    parser.add_argument("--skip-vina", action="store_true", help="Skip Vina base scoring (faster)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Load pKd labels
    print("Loading PDBbind index...")
    pkd_map = parse_index(INDEX_FILE)
    print(f"  {len(pkd_map)} complexes with pKd values")

    # Filter to available complexes
    available = []
    for pdb_id, pkd in sorted(pkd_map.items()):
        prot_path = REFINED_SET / pdb_id / f"{pdb_id}_protein.pdb"
        lig_path = REFINED_SET / pdb_id / f"{pdb_id}_ligand.sdf"
        if prot_path.exists() and lig_path.exists():
            available.append((pdb_id, pkd, str(prot_path), str(lig_path)))

    if args.max_complexes > 0:
        available = available[:args.max_complexes]

    print(f"  {len(available)} complexes with structure files")

    n_workers = min(mp.cpu_count(), 8)
    print(f"  Using {n_workers} workers")
    print()

    # Score all complexes in parallel
    t0 = time.time()
    worker = partial(_score_one_complex, pocket_radius=args.pocket_radius,
                     skip_vina=args.skip_vina)

    raw_terms_list = []
    vina_base_list = []
    pkd_list = []
    pdb_ids = []
    errors = 0

    with mp.Pool(n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(worker, available, chunksize=32)):
            if result is None:
                errors += 1
                continue
            pdb_id, pkd, raw, vina_e = result
            raw_terms_list.append(raw)
            vina_base_list.append(vina_e)
            pkd_list.append(pkd)
            pdb_ids.append(pdb_id)

            done = idx + 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(available) - done) / rate
                print(f"  [{done}/{len(available)}]  {rate:.0f} cpx/s  "
                      f"ETA {eta:.0f}s  ok={len(pkd_list)}  errors={errors}")

    elapsed = time.time() - t0
    print(f"\nScored {len(pkd_list)} complexes in {elapsed:.1f}s "
          f"({len(pkd_list)/elapsed:.0f} cpx/s), {errors} errors")

    raw_terms = np.array(raw_terms_list)
    vina_base = np.array(vina_base_list)
    pkd = np.array(pkd_list)

    # Current performance
    r_current = pearsonr(vina_base + raw_terms @ DEFAULT_WEIGHTS, pkd)[0]
    r_vina_only = pearsonr(vina_base, pkd)[0] if not args.skip_vina else 0.0
    print(f"\nVina-only → Pearson r = {r_vina_only:.4f}")
    print(f"Current Drusina weights → Pearson r = {r_current:.4f}")

    # Optimize
    result = minimize(objective, DEFAULT_WEIGHTS,
                      args=(raw_terms, vina_base, pkd, args.reg),
                      method="L-BFGS-B", bounds=BOUNDS,
                      options={"maxiter": 1000, "ftol": 1e-12})

    w_opt = result.x
    r_opt = pearsonr(vina_base + raw_terms @ w_opt, pkd)[0]
    print(f"Optimized weights → Pearson r = {r_opt:.4f}")
    print(f"Improvement over Vina: {abs(r_opt) - abs(r_vina_only):+.4f}")

    # Cross-validation
    if args.cv > 0:
        indices = np.random.permutation(len(pkd))
        fold_size = len(pkd) // args.cv
        test_rs = []
        for fold in range(args.cv):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            res = minimize(objective, DEFAULT_WEIGHTS,
                           args=(raw_terms[train_idx], vina_base[train_idx],
                                 pkd[train_idx], args.reg),
                           method="L-BFGS-B", bounds=BOUNDS,
                           options={"maxiter": 500})
            test_r = pearsonr(vina_base[test_idx] + raw_terms[test_idx] @ res.x,
                              pkd[test_idx])[0]
            test_rs.append(test_r)
        print(f"\n{args.cv}-fold CV: test r = {np.mean(test_rs):.4f} ± {np.std(test_rs):.4f}")

    # Print results
    print("\n" + "=" * 60)
    print("  OPTIMIZED DRUSINA WEIGHTS")
    print("=" * 60)
    print(f"{'Term':<22} {'Current':>10} {'Optimized':>10} {'Change':>10}")
    print("-" * 60)
    for i, term in enumerate(TERMS):
        c = DEFAULT_WEIGHTS[i]
        o = w_opt[i]
        flag = " ***" if abs(o - c) > abs(c) * 0.3 + 0.01 else ""
        print(f"  {term:<20} {c:>10.4f} {o:>10.4f} {o-c:>+10.4f}{flag}")

    # Swift code
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
    print("\n  // Swift (paste into DockingEngine.swift)")
    for i, term in enumerate(TERMS):
        print(f"    {swift_map[term]}: {w_opt[i]:.4f},")

    # Per-term stats
    print("\n" + "=" * 60)
    print(f"{'Term':<22} {'Mean':>8} {'Std':>8} {'Fire%':>6} {'Corr_pKd':>9}")
    print("-" * 60)
    for i, term in enumerate(TERMS):
        vals = raw_terms[:, i]
        nonzero = np.count_nonzero(vals)
        fire_pct = 100.0 * nonzero / len(vals)
        if nonzero > 2:
            cr, _ = pearsonr(vals[vals != 0], pkd[vals != 0])
        else:
            cr = 0.0
        print(f"  {term:<20} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
              f"{fire_pct:>5.1f}% {cr:>+8.4f}")


if __name__ == "__main__":
    main()

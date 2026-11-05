#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Johan H.G. Natter
# SPDX-License-Identifier: Apache-2.0
"""
Dataset plumbing shared by the Drusina survey and calibration scripts:
PDBbind refined-set / CASF-2016 discovery, pKd index parsing, CASF core-set
exclusion, ligand loading (SDF + MOL2 multi-model decoys) and a scratchpad
feature cache.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFINED_SET = ROOT / "Benchmark" / "data" / "refined-set"
INDEX_FILE = REFINED_SET / "index" / "INDEX_refined_data.2020"
CASF = ROOT / "Benchmark" / "data" / "CASF-2016"
CASF_CORESET = CASF / "coreset"
CASF_DECOYS = CASF / "decoys_docking"
CASF_CORE_INDEX = CASF / "power_docking" / "CoreSet.dat"

CACHE_DIR = Path(os.environ.get(
    "DRUSINA_CACHE",
    "/private/tmp/claude-501/-Users-vitruves-Developer-GitHub-Druse/"
    "592e1a14-445a-4612-9f36-79942fc3a6c6/scratchpad/drusina_cache"))


def parse_pkd_index(path: Path = INDEX_FILE) -> dict[str, float]:
    """PDBbind INDEX file -> {pdb_id: -logKd/Ki}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 4:
                try:
                    out[p[0].strip().lower()] = float(p[3])
                except ValueError:
                    pass
    return out


def parse_casf_coreset(path: Path = CASF_CORE_INDEX):
    """CASF CoreSet.dat -> ({pdb_id: pKd}, {pdb_id: cluster_id})."""
    pkd, cluster = {}, {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 6:
                pid = p[0].strip().lower()
                try:
                    pkd[pid] = float(p[3])
                    cluster[pid] = int(p[5])
                except ValueError:
                    pass
    return pkd, cluster


def refined_complexes(exclude: set[str] | None = None):
    """[(pdb_id, protein_pdb, ligand_sdf)] for every available refined-set entry."""
    out = []
    for d in sorted(REFINED_SET.iterdir()):
        if not d.is_dir() or len(d.name) != 4:
            continue
        pid = d.name.lower()
        if exclude and pid in exclude:
            continue
        prot = d / f"{d.name}_protein.pdb"
        lig = d / f"{d.name}_ligand.sdf"
        if prot.exists() and lig.exists():
            out.append((pid, str(prot), str(lig)))
    return out


def casf_targets():
    """[(pdb_id, protein_pdb, ligand_mol2, decoys_mol2, rmsd_dat)] for the core set."""
    out = []
    for d in sorted(CASF_CORESET.iterdir()):
        if not d.is_dir() or len(d.name) != 4:
            continue
        pid = d.name.lower()
        prot = d / f"{d.name}_protein.pdb"
        lig = d / f"{d.name}_ligand.mol2"
        dec = CASF_DECOYS / f"{d.name}_decoys.mol2"
        rms = CASF_DECOYS / f"{d.name}_rmsd.dat"
        if prot.exists() and lig.exists():
            out.append((pid, str(prot), str(lig),
                        str(dec) if dec.exists() else None,
                        str(rms) if rms.exists() else None))
    return out


def parse_rmsd_dat(path):
    """CASF *_rmsd.dat -> {pose_code: rmsd}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    out[p[0].strip()] = float(p[1])
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------
# Ligand loading
# ---------------------------------------------------------------------------

def load_ligand(path, sanitize=True):
    """Load a single ligand from .sdf or .mol2. Returns an RDKit mol (no H) or None."""
    from rdkit import Chem
    p = str(path)
    mol = None
    if p.endswith('.sdf'):
        supp = Chem.SDMolSupplier(p, removeHs=True, sanitize=sanitize)
        mol = next(iter(supp), None)
        if mol is None and sanitize:
            supp = Chem.SDMolSupplier(p, removeHs=True, sanitize=False)
            mol = next(iter(supp), None)
            if mol is not None:
                mol = _soft_sanitize(mol)
    elif p.endswith('.mol2'):
        mol = Chem.MolFromMol2File(p, removeHs=True, sanitize=sanitize)
        if mol is None:
            mol = Chem.MolFromMol2File(p, removeHs=True, sanitize=False)
            if mol is not None:
                mol = _soft_sanitize(mol)
    if mol is None:
        return None
    return mol


def _soft_sanitize(mol):
    """Partial sanitization: enough for ring/aromaticity/valence queries."""
    from rdkit import Chem
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
        return mol
    except Exception:
        return None


_MOL2_SPLIT = re.compile(r'^@<TRIPOS>MOLECULE', re.M)


def iter_mol2_blocks(path):
    """Yield (name, mol2_block_text) for each MOLECULE record in a multi-mol2."""
    with open(path, 'r', errors='replace') as fh:
        text = fh.read()
    parts = _MOL2_SPLIT.split(text)
    for part in parts[1:]:
        block = '@<TRIPOS>MOLECULE' + part
        lines = block.splitlines()
        name = lines[1].strip() if len(lines) > 1 else ''
        yield name, block


def mol2_block_coords(block):
    """Heavy-atom coordinates and element symbols from a mol2 ATOM record."""
    pos, names = [], []
    in_atoms = False
    for line in block.splitlines():
        if line.startswith('@<TRIPOS>ATOM'):
            in_atoms = True
            continue
        if line.startswith('@<TRIPOS>'):
            in_atoms = False
            continue
        if not in_atoms or not line.strip():
            continue
        f = line.split()
        if len(f) < 6:
            continue
        sym = f[5].split('.')[0]
        if sym == 'H':
            continue
        pos.append((float(f[2]), float(f[3]), float(f[4])))
        names.append(sym)
    return np.array(pos, dtype=np.float64), names

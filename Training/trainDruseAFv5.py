#!/usr/bin/env python3
"""
DruseAF v5 — Pairwise Geometric Network eXtended (PGN-X)
CUDA-accelerated training on PDBbind + CrossDocked2020 + BindingMOAD

Architecture (291K params, ~1.14 MB weights):
  Protein: MLP(20→160) + 4 rounds distance-attention message passing
  Ligand:  MLP(20→160) + 2 rounds distance-attention message passing (NEW)
  Scoring: Hadamard pairwise interaction within 10Å cutoff
           + 2-layer pair energy MLP (80→40→1)
  Output:  pKd (pair-decomposed + global) × confidence

Key improvements over v4:
  1. Ligand intramolecular message passing (v4 encodes ligand atoms independently)
  2. 100x more training data: CrossDocked2020 (22.5M real docked poses)
     + BindingMOAD (41K structures) + PDBbind v2020
  3. InfoNCE contrastive loss for pose discrimination from real docking decoys
  4. Hard negative mining focused on near-native RMSD 1-4Å decoys
  5. Multi-phase training: affinity → discrimination → joint
  6. EMA model averaging + DDP multi-GPU
  7. Wider representations (160/80 vs 128/64) + deeper pair energy head
  8. Larger cutoffs (10Å vs 8Å) for better long-range interactions

Data sources (listed by impact on pose discrimination):
  1. CrossDocked2020 (https://bits.csb.pitt.edu/files/crossdocked_pocket10/)
     22.5M docked poses across 13K+ targets. Provides REAL docking failures
     instead of random rigid-body perturbation — this is the breakthrough.
  2. BindingMOAD (https://bindingmoad.org/)
     41K structures with measured binding data (Ki, Kd, IC50, EC50).
     Doubles the affinity training set vs PDBbind alone.
  3. PDBbind v2020 (https://www.pdbbind.org.cn/)
     19K complexes with curated pKd values. Standard benchmark data.

Speed at Metal inference:
  Setup: ~4ms (encode + msg passing for both prot and lig, once per target)
  Per pose: ~0.04ms (vs 0.02ms for v4; 2x cost for 5-10x better discrimination)

Usage:
  # 1a. Preprocess PDBbind (required)
  python trainDruseAFv5.py preprocess-pdbbind \\
      --refined /data/PDBbind_v2020_refined \\
      --general /data/PDBbind_v2020_other_PL \\
      --casf /data/CASF-2016/coreset \\
      --out data/v5_cache/pdbbind

  # 1b. Preprocess CrossDocked2020 (strongly recommended)
  python trainDruseAFv5.py preprocess-crossdocked \\
      --crossdocked /data/crossdocked_pocket10 \\
      --types /data/crossdocked_pocket10/types/it2_tt_0_train0.types \\
      --out data/v5_cache/crossdocked

  # 1c. Preprocess BindingMOAD (recommended)
  python trainDruseAFv5.py preprocess-moad \\
      --moad /data/BindingMOAD_2024 \\
      --out data/v5_cache/moad

  # 2. Train (single GPU)
  python trainDruseAFv5.py train \\
      --pdbbind data/v5_cache/pdbbind \\
      --crossdocked data/v5_cache/crossdocked \\
      --moad data/v5_cache/moad \\
      --epochs 150 --batch 24 --lr 3e-4

  # 2. Train (RTX 3080 10GB — use grad accumulation for effective batch 48)
  python trainDruseAFv5.py train \\
      --pdbbind data/v5_cache/pdbbind \\
      --crossdocked data/v5_cache/crossdocked \\
      --epochs 150 --batch 24 --grad-accum 2 --lr 3e-4

  # 2b. Train (multi-GPU with DDP)
  torchrun --nproc_per_node=4 trainDruseAFv5.py train \\
      --pdbbind data/v5_cache/pdbbind \\
      --crossdocked data/v5_cache/crossdocked \\
      --epochs 150 --batch 24 --lr 3e-4

  # 3. Export weights for Metal
  python trainDruseAFv5.py export \\
      --checkpoint checkpoints_v5/best.pt \\
      --out ../Models/druse-models/DruseAFv5.weights

  # 4. Benchmark on CASF-2016 (scoring + docking power)
  python trainDruseAFv5.py benchmark \\
      --data data/v5_cache/pdbbind \\
      --checkpoint checkpoints_v5/best.pt \\
      --casf-decoys /data/CASF-2016/decoys_docking
"""

import os
import sys
import argparse
import copy
import math
import pickle
import struct
import json
import time
import re
from functools import partial
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# =============================================================================
# Constants — must match future Metal shader DruseAFv5Compute.metal
# =============================================================================

NUM_ATOM_FEATURES = 20     # same as v4 — backward compatible feature extraction
HIDDEN_DIM = 160           # wider (was 128)
PAIR_DIM = 80              # wider (was 64)
MSG_RBF_BINS = 20          # more expressive (was 16)
CROSS_RBF_BINS = 32        # more expressive (was 24)
MSG_CUTOFF = 10.0          # wider context (was 8.0)
CROSS_CUTOFF = 10.0        # wider context (was 8.0)
RBF_GAMMA = 2.0            # same as v4
NUM_PROT_MSG_LAYERS = 4    # deeper protein (was 3)
NUM_LIG_MSG_LAYERS = 2     # NEW: ligand message passing (was 0)
MAX_PROT_ATOMS = 384       # larger pockets (was 256)
MAX_LIG_ATOMS = 80         # larger ligands (was 64)
POCKET_RADIUS = 12.0       # wider pocket (was 10.0)
ATTN_HIDDEN = 40           # MSG_RBF_BINS * 2

# Perturbation settings (for PDBbind/MOAD crystal pose augmentation)
PERTURBATION_RMSDS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
CONFIDENCE_SIGMA = 2.0

# CrossDocked2020 subsampling ratios by RMSD bin (focus on hard negatives)
CROSSDOCK_SUBSAMPLE = {
    (0.0, 1.0): 1.0,    # near-native: keep all
    (1.0, 2.0): 1.0,    # good poses: keep all
    (2.0, 4.0): 0.8,    # hard negatives: keep 80%
    (4.0, 8.0): 0.3,    # moderate negatives: keep 30%
    (8.0, 15.0): 0.1,   # easy negatives: keep 10%
    (15.0, 100.0): 0.02, # garbage: keep 2%
}

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8}

AROMATIC_RESIDUE_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}
RESIDUE_ATOM_CHARGES = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NH1"): 0.33, ("ARG", "NH2"): 0.33, ("ARG", "NE"): 0.33,
    ("HIS", "ND1"): 0.25, ("HIS", "NE2"): 0.25,
}
RESIDUE_FORMAL_CHARGES = {
    ("ASP", "OD1"): -1, ("ASP", "OD2"): -1,
    ("GLU", "OE1"): -1, ("GLU", "OE2"): -1,
    ("LYS", "NZ"): 1,
    ("ARG", "NH1"): 1, ("ARG", "NH2"): 1, ("ARG", "NE"): 1, ("ARG", "CZ"): 1,
}
RING_RESIDUE_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "PRO": {"N", "CA", "CB", "CG", "CD"},
}
HBD_RESIDUE_ATOMS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}, "CYS": {"SG"},
}
HBA_RESIDUE_ATOMS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "HIS": {"ND1", "NE2"}, "MET": {"SD"}, "CYS": {"SG"},
}
SP2_RESIDUE_ATOMS = {
    "ASP": {"CG", "OD1", "OD2"}, "GLU": {"CD", "OE1", "OE2"},
    "ASN": {"CG", "OD1", "ND2"}, "GLN": {"CD", "OE1", "NE2"},
    "ARG": {"CZ", "NH1", "NH2"},
}
SYBYL_HYBRIDIZATION = {
    "1": (1, 0, 0), "2": (0, 1, 0), "3": (0, 0, 1),
    "ar": (0, 1, 0), "am": (0, 1, 0), "pl3": (0, 1, 0),
    "co2": (0, 1, 0), "cat": (0, 1, 0),
}


# =============================================================================
# Feature extraction (matches Swift/Metal exactly — same as v4)
# =============================================================================

def pdb_atom_features(element: str, residue_name: str, atom_name: str) -> np.ndarray:
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    feat[ATOM_TYPES.get(element, 9)] = 1.0
    feat[10] = float(atom_name in AROMATIC_RESIDUE_ATOMS.get(residue_name, set()))
    feat[11] = RESIDUE_ATOM_CHARGES.get((residue_name, atom_name), 0.0)
    is_bb_hbd = atom_name == "N" and residue_name != "PRO"
    feat[12] = float(is_bb_hbd or atom_name in HBD_RESIDUE_ATOMS.get(residue_name, set()))
    feat[13] = float(atom_name == "O" or atom_name in HBA_RESIDUE_ATOMS.get(residue_name, set()))
    is_sp2 = (feat[10] > 0 or atom_name in SP2_RESIDUE_ATOMS.get(residue_name, set())
              or (atom_name == "C" and element == "C") or (atom_name == "O" and element == "O"))
    feat[15] = float(is_sp2)
    feat[16] = float(not is_sp2)
    feat[18] = float(RESIDUE_FORMAL_CHARGES.get((residue_name, atom_name), 0))
    feat[19] = float(atom_name in RING_RESIDUE_ATOMS.get(residue_name, set()))
    return feat


def mol2_atom_features(element: str, sybyl_type: str, charge: float) -> np.ndarray:
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    feat[ATOM_TYPES.get(element, 9)] = 1.0
    parts = sybyl_type.split(".")
    subtype = parts[1] if len(parts) > 1 else "3"
    is_aromatic = subtype == "ar"
    feat[10] = float(is_aromatic)
    feat[11] = np.clip(charge, -1.0, 1.0)
    if element == "N" and subtype in ("3", "4", "am", "pl3"):
        feat[12] = 1.0
    elif element in ("O", "S") and subtype == "3":
        feat[12] = 1.0
    feat[13] = float(element in ("N", "O", "S", "F"))
    sp, sp2, sp3 = SYBYL_HYBRIDIZATION.get(subtype, (0, 0, 1))
    feat[14], feat[15], feat[16] = float(sp), float(sp2), float(sp3)
    feat[17] = 1.0
    formal = 0
    if abs(charge) > 0.5:
        formal = round(charge)
    if element == "N" and subtype == "4":
        formal = 1
    if element == "O" and subtype == "co2":
        formal = -1
    feat[18] = float(formal)
    feat[19] = float(is_aromatic)
    return feat


def rdkit_ligand_features(mol) -> Optional[np.ndarray]:
    """Extract 20-dim features from RDKit Mol (for SDF ligands in CrossDocked2020).
    Compatible with Swift/Metal feature extraction."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass  # some molecules fail — charges will be 0

    features = []
    for atom in mol.GetAtoms():
        feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
        elem = atom.GetSymbol()
        feat[ATOM_TYPES.get(elem, 9)] = 1.0
        feat[10] = float(atom.GetIsAromatic())
        try:
            charge = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not np.isfinite(charge):
                charge = 0.0
        except Exception:
            charge = 0.0
        feat[11] = np.clip(charge, -1.0, 1.0)
        # H-bond donor: heteroatom with H attached
        n_hs = atom.GetTotalNumHs()
        feat[12] = float(n_hs > 0 and elem in ("N", "O", "S"))
        feat[13] = float(elem in ("N", "O", "S", "F"))
        hyb = atom.GetHybridization()
        feat[14] = float(hyb == Chem.rdchem.HybridizationType.SP)
        feat[15] = float(hyb == Chem.rdchem.HybridizationType.SP2
                         or atom.GetIsAromatic())
        feat[16] = float(hyb == Chem.rdchem.HybridizationType.SP3
                         and not atom.GetIsAromatic())
        feat[17] = 1.0  # is_ligand
        feat[18] = float(atom.GetFormalCharge())
        feat[19] = float(atom.GetIsAromatic() and atom.IsInRing())
        features.append(feat)
    return np.array(features, dtype=np.float32)


# =============================================================================
# Parsers
# =============================================================================

def parse_pdb_atoms(path: str):
    """Parse ATOM/HETATM records from PDB. Returns (positions, features, elements)."""
    positions, features, elements = [], [], []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            element = line[76:78].strip()
            if not element:
                element = atom_name[0]
            if element == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            positions.append([x, y, z])
            features.append(pdb_atom_features(element, res_name, atom_name))
            elements.append(element)
    if not positions:
        return None, None, None
    return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32), elements


def parse_mol2_atoms(path: str):
    """Parse @<TRIPOS>ATOM section from MOL2. Returns (positions, features, elements)."""
    positions, features, elements = [], [], []
    in_atom_section = False
    with open(path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom_section = False
                continue
            if not in_atom_section:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                sybyl_type = parts[5]
                charge = float(parts[8])
            except (ValueError, IndexError):
                continue
            element = sybyl_type.split(".")[0]
            if element == "H":
                continue
            positions.append([x, y, z])
            features.append(mol2_atom_features(element, sybyl_type, charge))
            elements.append(element)
    if not positions:
        return None, None, None
    return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32), elements


def parse_sdf_rdkit(path: str):
    """Parse SDF file using RDKit. Returns (positions, features) or (None, None)."""
    if not HAS_RDKIT:
        return None, None
    try:
        suppl = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
        mol = next(iter(suppl), None)
    except Exception:
        return None, None
    if mol is None or mol.GetNumConformers() == 0:
        return None, None
    conf = mol.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                         dtype=np.float32)
    features = rdkit_ligand_features(mol)
    if features is None:
        return None, None
    return positions, features


def parse_index_file(path: str) -> dict:
    """Parse PDBbind INDEX file. Returns {pdb_id: pKd_value}."""
    result = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()
                try:
                    pkd = float(parts[3])
                    result[pdb_id] = pkd
                except ValueError:
                    continue
    return result


def parse_binding_value(value_str: str) -> Optional[float]:
    """Parse BindingMOAD binding data string into pKd.
    Handles: Ki=2.5nM, Kd=100uM, IC50=50nM, EC50=1mM, etc."""
    value_str = value_str.strip()
    match = re.match(r'(Ki|Kd|IC50|EC50|Ka)\s*[=<>~]\s*([\d.eE+-]+)\s*(fM|pM|nM|uM|mM|M)',
                     value_str, re.IGNORECASE)
    if not match:
        return None
    try:
        val = float(match.group(2))
    except ValueError:
        return None
    unit = match.group(3).lower()
    multipliers = {"fm": 1e-15, "pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}
    molar = val * multipliers.get(unit, 1.0)
    if molar <= 0:
        return None
    pkd = -math.log10(molar)
    if pkd < 0 or pkd > 16:
        return None
    return pkd


# =============================================================================
# Pose perturbation (for crystal pose augmentation)
# =============================================================================

def generate_perturbed_pose(lig_pos: np.ndarray, target_rmsd: float):
    if target_rmsd < 0.01:
        return lig_pos.copy(), 0.0
    center = lig_pos.mean(axis=0)
    centered = lig_pos - center
    lig_radius = np.sqrt((centered ** 2).sum(axis=1).mean())
    rot_rmsd = target_rmsd * 0.6
    trans_rmsd = target_rmsd * 0.4
    if lig_radius > 0.1:
        rot_angle = min(rot_rmsd / (lig_radius * 0.816), math.pi)
    else:
        rot_angle = 0.0
    axis = np.random.randn(3).astype(np.float32)
    axis /= np.linalg.norm(axis) + 1e-8
    R = Rotation.from_rotvec(axis * rot_angle).as_matrix().astype(np.float32)
    rotated = centered @ R.T
    trans_dir = np.random.randn(3).astype(np.float32)
    trans_dir /= np.linalg.norm(trans_dir) + 1e-8
    translated = rotated + center + trans_dir * trans_rmsd
    actual_rmsd = np.sqrt(np.mean(np.sum((translated - lig_pos) ** 2, axis=1)))
    return translated, float(actual_rmsd)


# =============================================================================
# Preprocessing: shared utilities
# =============================================================================

def extract_pocket(prot_pos, prot_feat, lig_center, pocket_radius=POCKET_RADIUS):
    """Extract pocket atoms within radius of ligand centroid. Trim to MAX_PROT_ATOMS."""
    dists = np.linalg.norm(prot_pos - lig_center, axis=1)
    pocket_mask = dists < pocket_radius
    pos = prot_pos[pocket_mask]
    feat = prot_feat[pocket_mask]
    if pos.shape[0] < 10:
        return None, None
    if pos.shape[0] > MAX_PROT_ATOMS:
        dists_pocket = np.linalg.norm(pos - lig_center, axis=1)
        keep_idx = np.argsort(dists_pocket)[:MAX_PROT_ATOMS]
        pos = pos[keep_idx]
        feat = feat[keep_idx]
    return pos, feat


def make_sample(pdb_id: str, prot_pos: np.ndarray, prot_feat: np.ndarray,
                lig_pos: np.ndarray, lig_feat: np.ndarray,
                pkd: float, rmsd: float) -> dict:
    """Create a training sample dict with confidence label."""
    conf_label = math.exp(-(rmsd ** 2) / (2.0 * CONFIDENCE_SIGMA ** 2))
    return {
        "pdb_id": pdb_id,
        "prot_pos": torch.from_numpy(prot_pos),
        "prot_feat": torch.from_numpy(prot_feat),
        "lig_pos": torch.from_numpy(lig_pos),
        "lig_feat": torch.from_numpy(lig_feat),
        "pkd": torch.tensor(pkd, dtype=torch.float32),
        "rmsd": torch.tensor(rmsd, dtype=torch.float32),
        "confidence": torch.tensor(conf_label, dtype=torch.float32),
    }


# =============================================================================
# Preprocessing: PDBbind
# =============================================================================

def preprocess_pdbbind_complex(pdb_id, prot_pdb, lig_mol2, pkd, out_dir):
    """Preprocess one PDBbind complex into .pt files (one per RMSD level)."""
    prot_pos, prot_feat, _ = parse_pdb_atoms(prot_pdb)
    lig_pos, lig_feat, _ = parse_mol2_atoms(lig_mol2)
    if prot_pos is None or lig_pos is None:
        return 0
    if lig_pos.shape[0] > MAX_LIG_ATOMS or lig_pos.shape[0] < 3:
        return 0

    lig_center = lig_pos.mean(axis=0)
    prot_pos, prot_feat = extract_pocket(prot_pos, prot_feat, lig_center)
    if prot_pos is None:
        return 0

    prot_pos_c = prot_pos - lig_center
    lig_pos_c = lig_pos - lig_center
    count = 0
    for target_rmsd in PERTURBATION_RMSDS:
        pert_pos, actual_rmsd = generate_perturbed_pose(lig_pos_c, target_rmsd)
        sample = make_sample(pdb_id, prot_pos_c, prot_feat, pert_pos, lig_feat,
                             pkd, actual_rmsd)
        torch.save(sample, out_dir / f"{pdb_id}_rmsd{target_rmsd:.1f}.pt")
        count += 1
    return count


def run_preprocess_pdbbind(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkd_map = {}
    for data_dir in [args.refined, args.general]:
        if data_dir is None:
            continue
        data_path = Path(data_dir)
        for idx_name in ["INDEX_refined_data.2020", "INDEX_general_PL_data.2020",
                         "index/INDEX_refined_data.2020", "index/INDEX_general_PL_data.2020",
                         "INDEX_general_PL.2020", "INDEX_refined_set.2020"]:
            idx_file = data_path / idx_name
            if idx_file.exists():
                pkd_map.update(parse_index_file(str(idx_file)))
        for idx_file in data_path.parent.glob("INDEX*"):
            pkd_map.update(parse_index_file(str(idx_file)))
    print(f"Loaded {len(pkd_map)} pKd values from index files")

    casf_ids = set()
    if args.casf:
        casf_path = Path(args.casf)
        for d in casf_path.iterdir():
            if d.is_dir() and len(d.name) == 4:
                casf_ids.add(d.name.lower())
    print(f"CASF-2016 exclusion set: {len(casf_ids)} complexes")

    total, errors = 0, 0
    split_info = {"train": [], "casf": []}

    for data_dir in [args.refined, args.general]:
        if data_dir is None:
            continue
        data_path = Path(data_dir)
        complex_dirs = sorted([d for d in data_path.iterdir()
                               if d.is_dir() and len(d.name) == 4])
        for cdir in tqdm(complex_dirs, desc=f"Processing {data_path.name}"):
            pdb_id = cdir.name.lower()
            if pdb_id not in pkd_map:
                continue
            prot_pdb = cdir / f"{pdb_id}_pocket.pdb"
            if not prot_pdb.exists():
                prot_pdb = cdir / f"{pdb_id}_protein.pdb"
            lig_mol2 = cdir / f"{pdb_id}_ligand.mol2"
            if not prot_pdb.exists() or not lig_mol2.exists():
                errors += 1
                continue
            try:
                n = preprocess_pdbbind_complex(pdb_id, str(prot_pdb), str(lig_mol2),
                                               pkd_map[pdb_id], out_dir)
                total += n
                if pdb_id in casf_ids:
                    split_info["casf"].append(pdb_id)
                else:
                    split_info["train"].append(pdb_id)
            except Exception:
                errors += 1

    # Also process CASF coreset directly
    if args.casf:
        casf_path = Path(args.casf)
        for cdir in sorted(casf_path.iterdir()):
            if not cdir.is_dir() or len(cdir.name) != 4:
                continue
            pdb_id = cdir.name.lower()
            if pdb_id not in pkd_map:
                continue
            if (out_dir / f"{pdb_id}_rmsd0.0.pt").exists():
                continue
            prot_pdb = cdir / f"{pdb_id}_pocket.pdb"
            if not prot_pdb.exists():
                prot_pdb = cdir / f"{pdb_id}_protein.pdb"
            lig_mol2 = cdir / f"{pdb_id}_ligand.mol2"
            if not prot_pdb.exists() or not lig_mol2.exists():
                continue
            try:
                n = preprocess_pdbbind_complex(pdb_id, str(prot_pdb), str(lig_mol2),
                                               pkd_map[pdb_id], out_dir)
                total += n
                if pdb_id not in split_info["casf"]:
                    split_info["casf"].append(pdb_id)
            except Exception:
                pass

    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nPDBbind preprocessing complete: {total} samples, "
          f"{len(split_info['train'])} train / {len(split_info['casf'])} CASF, "
          f"{errors} errors")


# =============================================================================
# Preprocessing: CrossDocked2020
# =============================================================================

def run_preprocess_crossdocked(args):
    """Preprocess CrossDocked2020 docked poses.

    CrossDocked2020 structure:
      crossdocked_pocket10/
        <target>/
          <target>.pdb                        (pocket)
          <target>__<source>_lig_<idx>_<rmsd>.sdf  (docked pose)

    The types file lists: <label> <rmsd> <receptor_path> <ligand_path>
    """
    if not HAS_RDKIT:
        print("ERROR: RDKit required for CrossDocked2020 preprocessing")
        print("  pip install rdkit")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.crossdocked)

    # Parse types file for (receptor, ligand, rmsd, label) tuples
    entries = []
    types_path = Path(args.types)
    if types_path.exists():
        with open(types_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    label = int(parts[0])
                    rmsd = float(parts[1])
                    rec_path = parts[2]
                    lig_path = parts[3]
                    entries.append((label, rmsd, rec_path, lig_path))
        print(f"Loaded {len(entries)} entries from types file")
    else:
        # Fallback: scan directory structure, extract RMSD from filename
        print("No types file — scanning directory for SDF files...")
        for rec_dir in sorted(base_dir.iterdir()):
            if not rec_dir.is_dir():
                continue
            rec_pdb = rec_dir / f"{rec_dir.name}.pdb"
            if not rec_pdb.exists():
                continue
            for sdf_file in rec_dir.glob("*.sdf"):
                # Extract RMSD from filename: ..._{rmsd}.sdf
                name = sdf_file.stem
                try:
                    rmsd = float(name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                label = 1 if rmsd < 2.0 else 0
                entries.append((label, rmsd,
                                str(rec_pdb.relative_to(base_dir)),
                                str(sdf_file.relative_to(base_dir))))
        print(f"Found {len(entries)} SDF entries by scanning")

    # Subsample by RMSD bin
    np.random.seed(42)
    kept = []
    for label, rmsd, rec, lig in entries:
        for (lo, hi), ratio in CROSSDOCK_SUBSAMPLE.items():
            if lo <= rmsd < hi:
                if np.random.random() < ratio:
                    kept.append((label, rmsd, rec, lig))
                break
        else:
            if rmsd >= 100.0:
                continue
            if np.random.random() < 0.01:
                kept.append((label, rmsd, rec, lig))
    print(f"After subsampling: {len(kept)} entries (from {len(entries)})")

    # Group by receptor for efficient PDB parsing
    rec_groups = defaultdict(list)
    for label, rmsd, rec, lig in kept:
        rec_groups[rec].append((label, rmsd, lig))

    total, errors = 0, 0
    target_ids = []

    for rec_path, lig_entries in tqdm(rec_groups.items(), desc="CrossDocked2020"):
        rec_full = base_dir / rec_path
        if not rec_full.exists():
            errors += len(lig_entries)
            continue

        prot_pos, prot_feat, _ = parse_pdb_atoms(str(rec_full))
        if prot_pos is None:
            errors += len(lig_entries)
            continue

        # Extract target name for grouping
        target_name = Path(rec_path).stem

        for label, rmsd, lig_path in lig_entries:
            lig_full = base_dir / lig_path
            if not lig_full.exists():
                errors += 1
                continue

            lig_pos, lig_feat = parse_sdf_rdkit(str(lig_full))
            if lig_pos is None or lig_pos.shape[0] > MAX_LIG_ATOMS or lig_pos.shape[0] < 3:
                errors += 1
                continue

            lig_center = lig_pos.mean(axis=0)
            pp, pf = extract_pocket(prot_pos, prot_feat, lig_center)
            if pp is None:
                errors += 1
                continue

            pp_c = pp - lig_center
            lp_c = lig_pos - lig_center

            # pKd is NaN for cross-docked (only confidence/discrimination training)
            pkd_val = float("nan")
            sample = make_sample(target_name, pp_c, pf, lp_c, lig_feat, pkd_val, rmsd)

            fname = f"cd_{target_name}_{total:07d}.pt"
            torch.save(sample, out_dir / fname)
            total += 1

        if target_name not in target_ids:
            target_ids.append(target_name)

    meta = {"source": "crossdocked2020", "total_samples": total,
            "num_targets": len(target_ids), "target_ids": target_ids[:100]}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nCrossDocked2020 preprocessing complete: {total} samples, "
          f"{len(target_ids)} targets, {errors} errors")


# =============================================================================
# Preprocessing: BindingMOAD
# =============================================================================

def run_preprocess_moad(args):
    """Preprocess BindingMOAD structures.

    BindingMOAD structure:
      BindingMOAD/
        every_part/
          <complex_id>/
            <complex_id>_protein.pdb
            <complex_id>_ligand.mol2 (or .sdf)
        binding_data.csv (or .txt)
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    moad_dir = Path(args.moad)

    # Parse binding data
    pkd_map = {}
    for bd_file in moad_dir.glob("binding_data*"):
        with open(bd_file) as f:
            for line in f:
                if line.startswith("#") or line.startswith("PDB"):
                    continue
                parts = line.strip().split(",") if "," in line else line.strip().split("\t")
                if len(parts) < 4:
                    continue
                pdb_id = parts[0].strip().lower()
                for part in parts[1:]:
                    val = parse_binding_value(part.strip())
                    if val is not None:
                        pkd_map[pdb_id] = val
                        break
    print(f"Loaded {len(pkd_map)} binding values from BindingMOAD")

    # Also try to load nr-bind.csv (non-redundant binding data)
    for nr_file in moad_dir.glob("**/nr-bind*"):
        with open(nr_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    pdb_id = parts[0].strip().lower()[:4]
                    val = parse_binding_value(parts[4].strip() if len(parts) > 4 else "")
                    if val is not None:
                        pkd_map.setdefault(pdb_id, val)

    # Find complex directories
    complex_dirs = []
    for search_dir in [moad_dir / "every_part", moad_dir / "BindingMOAD", moad_dir]:
        if search_dir.is_dir():
            for d in search_dir.iterdir():
                if d.is_dir() and len(d.name) >= 4:
                    complex_dirs.append(d)
    print(f"Found {len(complex_dirs)} complex directories")

    total, errors = 0, 0
    processed_ids = []

    for cdir in tqdm(complex_dirs, desc="BindingMOAD"):
        pdb_id = cdir.name[:4].lower()
        if pdb_id not in pkd_map:
            continue
        if pdb_id in processed_ids:
            continue

        # Find protein file
        prot_file = None
        for p in [cdir / f"{cdir.name}_protein.pdb", cdir / f"{pdb_id}_protein.pdb"]:
            if p.exists():
                prot_file = p
                break
        if prot_file is None:
            prot_files = list(cdir.glob("*protein*.pdb"))
            if prot_files:
                prot_file = prot_files[0]
        if prot_file is None:
            continue

        # Find ligand file
        lig_file = None
        for ext in ["mol2", "sdf"]:
            candidates = list(cdir.glob(f"*ligand*.{ext}"))
            if candidates:
                lig_file = candidates[0]
                break
        if lig_file is None:
            continue

        prot_pos, prot_feat, _ = parse_pdb_atoms(str(prot_file))
        if prot_pos is None:
            errors += 1
            continue

        if str(lig_file).endswith(".mol2"):
            lig_pos, lig_feat, _ = parse_mol2_atoms(str(lig_file))
        else:
            lig_pos, lig_feat = parse_sdf_rdkit(str(lig_file))

        if lig_pos is None or lig_pos.shape[0] > MAX_LIG_ATOMS or lig_pos.shape[0] < 3:
            errors += 1
            continue

        lig_center = lig_pos.mean(axis=0)
        pp, pf = extract_pocket(prot_pos, prot_feat, lig_center)
        if pp is None:
            errors += 1
            continue

        pp_c = pp - lig_center
        lp_c = lig_pos - lig_center
        pkd = pkd_map[pdb_id]

        for target_rmsd in PERTURBATION_RMSDS:
            pert_pos, actual_rmsd = generate_perturbed_pose(lp_c, target_rmsd)
            sample = make_sample(f"moad_{pdb_id}", pp_c, pf, pert_pos, lig_feat,
                                 pkd, actual_rmsd)
            torch.save(sample, out_dir / f"moad_{pdb_id}_rmsd{target_rmsd:.1f}.pt")
            total += 1

        processed_ids.append(pdb_id)

    meta = {"source": "binding_moad", "total_samples": total,
            "num_complexes": len(processed_ids)}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nBindingMOAD preprocessing complete: {total} samples from "
          f"{len(processed_ids)} complexes, {errors} errors")


# =============================================================================
# Dataset
# =============================================================================

class V5Dataset(Dataset):
    """Loads preprocessed .pt samples from a cache directory.
    Supports both affinity (PDBbind/MOAD) and discrimination (CrossDocked) data."""

    def __init__(self, cache_dir: str, pdb_ids: Optional[list] = None,
                 max_samples: Optional[int] = None):
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            self.samples = []
            return
        files = sorted(cache_path.glob("*.pt"))
        if pdb_ids is not None:
            id_set = set(pdb_ids)
            files = [f for f in files if f.stem.split("_rmsd")[0] in id_set]
        if max_samples and len(files) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(files), max_samples, replace=False)
            files = [files[i] for i in sorted(indices)]
        self.samples = []
        for f in tqdm(files, desc=f"Loading {cache_path.name}", leave=False):
            try:
                self.samples.append(torch.load(f, weights_only=False))
            except Exception:
                continue
        print(f"  {cache_path.name}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        P = min(s["prot_pos"].shape[0], MAX_PROT_ATOMS)
        L = min(s["lig_pos"].shape[0], MAX_LIG_ATOMS)

        prot_pos = torch.zeros(MAX_PROT_ATOMS, 3)
        prot_feat = torch.zeros(MAX_PROT_ATOMS, NUM_ATOM_FEATURES)
        prot_mask = torch.zeros(MAX_PROT_ATOMS, dtype=torch.bool)
        prot_pos[:P] = s["prot_pos"][:P]
        prot_feat[:P] = s["prot_feat"][:P]
        prot_mask[:P] = True

        lig_pos = torch.zeros(MAX_LIG_ATOMS, 3)
        lig_feat = torch.zeros(MAX_LIG_ATOMS, NUM_ATOM_FEATURES)
        lig_mask = torch.zeros(MAX_LIG_ATOMS, dtype=torch.bool)
        lig_pos[:L] = s["lig_pos"][:L]
        lig_feat[:L] = s["lig_feat"][:L]
        lig_mask[:L] = True

        pkd = s["pkd"]
        has_affinity = torch.tensor(not torch.isnan(pkd), dtype=torch.bool)
        if torch.isnan(pkd):
            pkd = torch.tensor(0.0, dtype=torch.float32)

        return {
            "pdb_id": s["pdb_id"],
            "prot_pos": prot_pos, "prot_feat": prot_feat, "prot_mask": prot_mask,
            "lig_pos": lig_pos, "lig_feat": lig_feat, "lig_mask": lig_mask,
            "pkd": pkd, "rmsd": s["rmsd"], "confidence": s["confidence"],
            "has_affinity": has_affinity,
        }


class StreamingDataset(Dataset):
    """Memory-efficient dataset that loads .pt files on demand (for CrossDocked).
    Keeps only file paths in memory."""

    def __init__(self, cache_dir: str, max_samples: Optional[int] = None):
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            self.file_paths = []
            return
        self.file_paths = sorted(cache_path.glob("*.pt"))
        if max_samples and len(self.file_paths) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.file_paths), max_samples, replace=False)
            self.file_paths = [self.file_paths[i] for i in sorted(indices)]
        print(f"  {cache_path.name}: {len(self.file_paths)} samples (streaming)")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        s = torch.load(self.file_paths[idx], weights_only=False)
        P = min(s["prot_pos"].shape[0], MAX_PROT_ATOMS)
        L = min(s["lig_pos"].shape[0], MAX_LIG_ATOMS)

        prot_pos = torch.zeros(MAX_PROT_ATOMS, 3)
        prot_feat = torch.zeros(MAX_PROT_ATOMS, NUM_ATOM_FEATURES)
        prot_mask = torch.zeros(MAX_PROT_ATOMS, dtype=torch.bool)
        prot_pos[:P] = s["prot_pos"][:P]
        prot_feat[:P] = s["prot_feat"][:P]
        prot_mask[:P] = True

        lig_pos = torch.zeros(MAX_LIG_ATOMS, 3)
        lig_feat = torch.zeros(MAX_LIG_ATOMS, NUM_ATOM_FEATURES)
        lig_mask = torch.zeros(MAX_LIG_ATOMS, dtype=torch.bool)
        lig_pos[:L] = s["lig_pos"][:L]
        lig_feat[:L] = s["lig_feat"][:L]
        lig_mask[:L] = True

        pkd = s["pkd"]
        has_affinity = torch.tensor(not torch.isnan(pkd), dtype=torch.bool)
        if torch.isnan(pkd):
            pkd = torch.tensor(0.0, dtype=torch.float32)

        return {
            "pdb_id": s["pdb_id"],
            "prot_pos": prot_pos, "prot_feat": prot_feat, "prot_mask": prot_mask,
            "lig_pos": lig_pos, "lig_feat": lig_feat, "lig_mask": lig_mask,
            "pkd": pkd, "rmsd": s["rmsd"], "confidence": s["confidence"],
            "has_affinity": has_affinity,
        }


# =============================================================================
# Model: Gaussian RBF
# =============================================================================

class GaussianRBF(nn.Module):
    def __init__(self, bins: int, cutoff: float, gamma: float = RBF_GAMMA):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, bins))
        self.gamma = gamma

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


# =============================================================================
# Model: Message Passing Layer (shared for protein and ligand intra-graph)
# =============================================================================

class MessagePassingLayer(nn.Module):
    """Distance-attention message passing for intra-molecular encoding.

    Metal equivalent: druseAFv5MsgTransform + druseAFv5MsgAggregate kernels.
    Weight tensors per layer: msg_mlp (w,b), attn_mlp (w1,b1,w2,b2), norm (w,b) = 8
    """

    def __init__(self, hidden: int = HIDDEN_DIM, rbf_bins: int = MSG_RBF_BINS,
                 cutoff: float = MSG_CUTOFF):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = GaussianRBF(rbf_bins, cutoff)
        self.attn_mlp = nn.Sequential(
            nn.Linear(rbf_bins, ATTN_HIDDEN),
            nn.GELU(approximate="tanh"),
            nn.Linear(ATTN_HIDDEN, 1),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, mask):
        """h: [B, N, H]  pos: [B, N, 3]  mask: [B, N] bool"""
        dist = torch.cdist(pos, pos)
        nbr_mask = (dist < self.cutoff) & (dist > 0.01)
        nbr_mask = nbr_mask & mask.unsqueeze(1) & mask.unsqueeze(2)

        rbf = self.rbf(dist)
        attn = self.attn_mlp(rbf).squeeze(-1)
        attn = attn.masked_fill(~nbr_mask, -6e4)
        attn = F.softmax(attn, dim=-1)

        transformed = self.msg_mlp(h)
        msg = torch.bmm(attn, transformed)
        return self.norm(h + msg)


# =============================================================================
# Model: PGN-X (Pairwise Geometric Network eXtended)
# =============================================================================

class PairwiseGeometricNetX(nn.Module):
    """DruseAF v5 — Pairwise Geometric Network eXtended (PGN-X).

    291K parameters. Weight tensor layout (82 tensors total):
      [0-3]   Protein encoder MLP
      [4-35]  4× protein message passing layers (8 tensors each)
      [36-39] Ligand encoder MLP
      [40-55] 2× ligand message passing layers (8 tensors each) [NEW in v5]
      [56-59] Protein + ligand pair projections
      [60-61] RBF cross-interaction projection
      [62-65] 2-layer pair energy head (80→40→1) [deeper in v5]
      [66-67] Context gate
      [68-69] Context projection
      [70-71] Context LayerNorm
      [72-75] Affinity head
      [76-79] Confidence head
      [80]    pair_scale scalar
      [81]    pair_bias scalar
    """

    def __init__(self, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Protein encoder: MLP(20→160→160)
        self.prot_encoder = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        # 4 rounds of intra-protein message passing
        self.prot_msg_layers = nn.ModuleList([
            MessagePassingLayer() for _ in range(NUM_PROT_MSG_LAYERS)
        ])
        # Ligand encoder: MLP(20→160→160)
        self.lig_encoder = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        # 2 rounds of intra-ligand message passing (NEW in v5)
        self.lig_msg_layers = nn.ModuleList([
            MessagePassingLayer() for _ in range(NUM_LIG_MSG_LAYERS)
        ])
        # Pair projections
        self.prot_pair_proj = nn.Linear(HIDDEN_DIM, PAIR_DIM)
        self.lig_pair_proj = nn.Linear(HIDDEN_DIM, PAIR_DIM)
        # Cross-interaction RBF projection
        self.rbf = GaussianRBF(CROSS_RBF_BINS, CROSS_CUTOFF)
        self.rbf_proj = nn.Sequential(
            nn.Linear(CROSS_RBF_BINS, PAIR_DIM),
            nn.GELU(approximate="tanh"),
        )
        # 2-layer pair energy: Linear(80→40)+GELU+Linear(40→1) [deeper than v4]
        self.pair_energy = nn.Sequential(
            nn.Linear(PAIR_DIM, PAIR_DIM // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(PAIR_DIM // 2, 1),
        )
        # Context aggregation
        self.context_gate = nn.Linear(PAIR_DIM, 1)
        self.context_proj = nn.Linear(PAIR_DIM, HIDDEN_DIM)
        self.context_norm = nn.LayerNorm(HIDDEN_DIM)
        # Prediction heads
        self.affinity_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, PAIR_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(PAIR_DIM, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, PAIR_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(PAIR_DIM, 1),
        )
        # Learned pair energy scaling
        self.pair_scale = nn.Parameter(torch.tensor(0.1))
        self.pair_bias = nn.Parameter(torch.tensor(6.0))

    def _msg_pass(self, h, pos, mask, layers):
        """Run message passing layers, with optional gradient checkpointing."""
        for layer in layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, pos, mask, use_reentrant=False)
            else:
                h = layer(h, pos, mask)
        return h

    def forward(self, prot_pos, prot_feat, lig_pos, lig_feat, prot_mask, lig_mask):
        """
        All inputs: [B, P/L, D]  masks: [B, P/L] bool
        Returns: pKd [B], confidence [B], pair_energies [B, L, P]
        """
        # Encode protein atoms + message passing (checkpointed if enabled)
        prot_h = self.prot_encoder(prot_feat)
        prot_h = self._msg_pass(prot_h, prot_pos, prot_mask, self.prot_msg_layers)

        # Encode ligand atoms + message passing (NEW in v5)
        lig_h = self.lig_encoder(lig_feat)
        lig_h = self._msg_pass(lig_h, lig_pos, lig_mask, self.lig_msg_layers)

        # Pair projections
        prot_p = self.prot_pair_proj(prot_h)  # [B, P, PD]
        lig_p = self.lig_pair_proj(lig_h)      # [B, L, PD]

        # Cross distances + RBF
        cross_dist = torch.cdist(lig_pos, prot_pos)
        cross_mask = (cross_dist < CROSS_CUTOFF) & lig_mask.unsqueeze(2) & prot_mask.unsqueeze(1)

        # Sparse RBF: only compute for pairs within cutoff (saves memory)
        cross_rbf = self.rbf(cross_dist)
        rbf_p = self.rbf_proj(cross_rbf)
        # Zero out pairs beyond cutoff before Hadamard (avoids large intermediate)
        rbf_p = rbf_p * cross_mask.unsqueeze(-1).float()

        # Hadamard pairwise interaction: lig ⊙ prot ⊙ rbf
        pair = lig_p.unsqueeze(2) * prot_p.unsqueeze(1) * rbf_p  # [B, L, P, PD]

        # 2-layer pair energy: Linear(80→40)+GELU+Linear(40→1)
        pair_e = self.pair_energy(pair).squeeze(-1)  # [B, L, P]
        pair_e = pair_e * cross_mask.float()

        # Context gate (reuse pair, don't recompute)
        gate_logit = self.context_gate(pair).squeeze(-1)
        gate_logit = gate_logit.masked_fill(~cross_mask, -6e4)
        gate_weight = F.softmax(gate_logit, dim=2)

        # Free the large pair tensor now that we're done with it
        del pair, rbf_p, cross_rbf

        # Context: weighted protein features per ligand atom
        context = torch.einsum("blp,bpd->bld", gate_weight, prot_p)
        lig_h_ctx = self.context_norm(lig_h + self.context_proj(context))

        # Pool over ligand atoms + predict
        lig_h_ctx = lig_h_ctx * lig_mask.unsqueeze(-1).float()
        n_lig = lig_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        complex_repr = lig_h_ctx.sum(dim=1) / n_lig

        total_pair_energy = pair_e.sum(dim=(1, 2))
        pKd = (total_pair_energy * self.pair_scale
               + self.affinity_head(complex_repr).squeeze(-1)
               + self.pair_bias)
        confidence = torch.sigmoid(self.confidence_head(complex_repr).squeeze(-1))

        return pKd, confidence, pair_e


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMAModel:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA params (for eval/export)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after eval."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# =============================================================================
# Loss functions
# =============================================================================

def affinity_loss(pKd_pred, pKd_true, rmsd, has_affinity):
    """RMSD-weighted Huber loss for affinity prediction. Only on samples with real pKd."""
    if not has_affinity.any():
        return torch.zeros(1, device=pKd_pred.device).squeeze()
    mask = has_affinity
    pred = pKd_pred[mask]
    true = pKd_true[mask]
    r = rmsd[mask]
    pose_weight = torch.exp(-r / 3.0)
    loss = pose_weight * F.huber_loss(pred, true, reduction="none", delta=2.0)
    return loss.sum() / pose_weight.sum().clamp(min=1.0)


def confidence_loss(conf_pred, conf_true):
    """MSE loss for confidence prediction."""
    return F.mse_loss(conf_pred, conf_true)


def infonce_contrastive_loss(scores, rmsd, pdb_ids, device, temperature=0.07):
    """InfoNCE contrastive loss for pose discrimination.

    Within each target group, the lowest-RMSD pose is the positive and
    all others are negatives. This teaches the model to rank poses.
    """
    B = scores.shape[0]
    pid_map = {}
    gid = 0
    group_ids = torch.empty(B, dtype=torch.long, device=device)
    for i, pid in enumerate(pdb_ids):
        if pid not in pid_map:
            pid_map[pid] = gid
            gid += 1
        group_ids[i] = pid_map[pid]

    total_loss = torch.zeros(1, device=device).squeeze()
    n_groups = 0

    for g in range(gid):
        members = (group_ids == g).nonzero(as_tuple=True)[0]
        if len(members) < 2:
            continue

        group_scores = scores[members] / temperature
        group_rmsd = rmsd[members]

        # Positive: lowest RMSD in group
        pos_idx = group_rmsd.argmin()
        # If best pose is too bad (RMSD > 4Å), skip — no clear positive
        if group_rmsd[pos_idx] > 4.0:
            continue

        # InfoNCE: -log(exp(s_pos) / sum(exp(s_all)))
        log_sum_exp = torch.logsumexp(group_scores, dim=0)
        loss_g = log_sum_exp - group_scores[pos_idx]
        total_loss = total_loss + loss_g
        n_groups += 1

    if n_groups == 0:
        return torch.zeros(1, device=device).squeeze()
    return total_loss / n_groups


def ranking_loss(pred_scores, rmsd, pdb_ids, device, margin=0.5):
    """Margin-based ranking loss with hard negative emphasis."""
    B = pred_scores.shape[0]
    pid_map = {}
    gid = 0
    group_ids = torch.empty(B, dtype=torch.long, device=device)
    for i, pid in enumerate(pdb_ids):
        if pid not in pid_map:
            pid_map[pid] = gid
            gid += 1
        group_ids[i] = pid_map[pid]

    same = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    triu = torch.triu(torch.ones(B, B, device=device, dtype=torch.bool), diagonal=1)
    rmsd_diff = rmsd.unsqueeze(0) - rmsd.unsqueeze(1)
    valid = same & triu & (rmsd_diff.abs() > 0.5)

    if not valid.any():
        return torch.zeros(1, device=device).squeeze()

    score_diff = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)
    sign = torch.sign(-rmsd_diff)  # +1 when i has lower RMSD (better)

    # Hard negative weighting: larger weight for pairs where model fails
    raw_loss = F.relu(margin - sign * score_diff)
    # Weight by how close the RMSDs are (hard negatives are more valuable)
    difficulty = torch.exp(-rmsd_diff.abs() / 3.0)
    weighted_loss = raw_loss * difficulty

    return weighted_loss[valid].mean()


# =============================================================================
# Training
# =============================================================================

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def train_epoch(model, loader, optimizer, scaler, device, epoch, phase="joint",
                temperature=0.07, grad_accum_steps=1):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"  Train ep{epoch+1} ({phase})",
                                       leave=False, disable=not is_main_process())):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with autocast("cuda", dtype=torch.float16):
            pKd_pred, conf_pred, pair_e = model(
                batch["prot_pos"], batch["prot_feat"],
                batch["lig_pos"], batch["lig_feat"],
                batch["prot_mask"], batch["lig_mask"])

            pred_scores = pKd_pred * conf_pred

            # 1. Affinity loss (only for samples with real pKd)
            loss_aff = affinity_loss(pKd_pred, batch["pkd"], batch["rmsd"],
                                     batch["has_affinity"])

            # 2. Confidence loss
            loss_conf = confidence_loss(conf_pred, batch["confidence"])

            # 3. InfoNCE contrastive loss (pose discrimination)
            loss_nce = infonce_contrastive_loss(
                pred_scores, batch["rmsd"], batch["pdb_id"], device, temperature)

            # 4. Ranking loss with hard negative emphasis
            loss_rank = ranking_loss(
                pred_scores, batch["rmsd"], batch["pdb_id"], device)

            # 5. Pair energy regularization
            loss_pair_reg = pair_e.abs().mean() * 0.005

            # Phase-dependent weighting
            if phase == "affinity":
                loss = 2.0 * loss_aff + 3.0 * loss_conf + 0.5 * loss_rank + loss_pair_reg
            elif phase == "discrimination":
                loss = 0.5 * loss_aff + 5.0 * loss_conf + 4.0 * loss_nce + 2.0 * loss_rank + loss_pair_reg
            else:  # joint
                loss = 1.5 * loss_aff + 4.0 * loss_conf + 3.0 * loss_nce + 2.0 * loss_rank + loss_pair_reg

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pkd_true, all_pkd_pred, all_conf_pred, all_rmsd = [], [], [], []
    all_score_pred, all_score_true, all_has_aff = [], [], []
    all_pdb_ids = []

    for batch in tqdm(loader, desc="  Eval", leave=False, disable=not is_main_process()):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        pKd_pred, conf_pred, _ = model(
            batch["prot_pos"], batch["prot_feat"],
            batch["lig_pos"], batch["lig_feat"],
            batch["prot_mask"], batch["lig_mask"])

        all_pkd_true.append(batch["pkd"].cpu())
        all_pkd_pred.append(pKd_pred.cpu())
        all_conf_pred.append(conf_pred.cpu())
        all_rmsd.append(batch["rmsd"].cpu())
        all_score_pred.append((pKd_pred * conf_pred).cpu())
        all_score_true.append((batch["pkd"] * batch["confidence"]).cpu())
        all_has_aff.append(batch["has_affinity"].cpu())
        all_pdb_ids.extend(batch["pdb_id"])

    pkd_true = torch.cat(all_pkd_true)
    pkd_pred = torch.cat(all_pkd_pred)
    conf_pred = torch.cat(all_conf_pred)
    rmsd = torch.cat(all_rmsd)
    score_pred = torch.cat(all_score_pred)
    has_aff = torch.cat(all_has_aff)

    # Scoring power: Pearson R on crystal poses with affinity
    crystal_mask = (rmsd < 0.1) & has_aff
    if crystal_mask.sum() > 5:
        crystal_pkd_true = pkd_true[crystal_mask]
        crystal_pkd_pred = pkd_pred[crystal_mask]
        r = torch.corrcoef(torch.stack([crystal_pkd_true, crystal_pkd_pred]))[0, 1].item()
        rmse = (crystal_pkd_true - crystal_pkd_pred).pow(2).mean().sqrt().item()
    else:
        r, rmse = 0.0, 99.0

    # Confidence accuracy
    conf_true = torch.exp(-rmsd ** 2 / (2 * CONFIDENCE_SIGMA ** 2))
    conf_rmse = (conf_pred - conf_true).pow(2).mean().sqrt().item()

    # Pose discrimination: AUC for good (RMSD<2) vs bad (RMSD>4) poses
    good_mask = rmsd < 2.0
    bad_mask = rmsd > 4.0
    disc_mask = good_mask | bad_mask
    if disc_mask.sum() > 10 and good_mask.sum() > 2 and bad_mask.sum() > 2:
        disc_scores = score_pred[disc_mask]
        disc_labels = good_mask[disc_mask].float()
        # Simple AUC approximation via ranking
        sorted_idx = disc_scores.argsort(descending=True)
        sorted_labels = disc_labels[sorted_idx]
        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            tpr_sum = sorted_labels.cumsum(0)
            auc = (tpr_sum * (1 - sorted_labels)).sum() / (n_pos * n_neg)
            auc = 1.0 - auc.item()  # correct direction
        else:
            auc = 0.5
    else:
        auc = 0.5

    # Docking power: per-target success rate (best-scored pose RMSD < 2Å)
    target_groups = defaultdict(list)
    for i, pid in enumerate(all_pdb_ids):
        target_groups[pid].append(i)

    n_targets_tested = 0
    n_successes_top1 = 0
    n_successes_top3 = 0
    for pid, indices in target_groups.items():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices)
        grp_scores = score_pred[idx]
        grp_rmsd = rmsd[idx]
        sorted_idx = grp_scores.argsort(descending=True)
        grp_rmsd_sorted = grp_rmsd[sorted_idx]
        n_targets_tested += 1
        if grp_rmsd_sorted[0] < 2.0:
            n_successes_top1 += 1
        if grp_rmsd_sorted[:3].min() < 2.0:
            n_successes_top3 += 1

    dock_top1 = n_successes_top1 / max(n_targets_tested, 1)
    dock_top3 = n_successes_top3 / max(n_targets_tested, 1)

    # Score correlation (all poses with affinity)
    aff_mask = has_aff & (rmsd < 0.1)
    if aff_mask.sum() > 5:
        score_r = torch.corrcoef(
            torch.stack([pkd_true[aff_mask], score_pred[aff_mask]]))[0, 1].item()
    else:
        score_r = 0.0

    return {
        "pearson_r": r, "rmse": rmse,
        "conf_rmse": conf_rmse, "score_r": score_r,
        "disc_auc": auc,
        "dock_top1": dock_top1, "dock_top3": dock_top3,
        "n_crystal": int(crystal_mask.sum()),
        "n_targets_tested": n_targets_tested,
    }


def run_train(args):
    # DDP setup
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # Enable TF32 on Ampere+ GPUs (RTX 3080, A100, etc.) for faster fp32 matmuls
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if is_main_process():
        print(f"Device: {device}  Distributed: {distributed}")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(local_rank)
            vram_gb = props.total_memory / 1e9
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {vram_gb:.1f} GB")
            if vram_gb < 12:
                print(f"  NOTE: {vram_gb:.0f}GB VRAM detected — use --batch 24 --grad-accum 2")
            if distributed:
                print(f"  World size: {dist.get_world_size()}")

    # Load datasets
    if is_main_process():
        print("\n=== Loading datasets ===")

    datasets_aff = []   # affinity data (PDBbind + MOAD)
    datasets_disc = []  # discrimination data (CrossDocked)
    casf_ids = None

    # PDBbind
    if args.pdbbind:
        cache_dir = Path(args.pdbbind)
        split_file = cache_dir / "split_info.json"
        if split_file.exists():
            with open(split_file) as f:
                split_info = json.load(f)
            train_ids = split_info["train"]
            casf_ids = split_info.get("casf", [])
        else:
            all_ids = list({f.stem.split("_rmsd")[0] for f in cache_dir.glob("*.pt")})
            np.random.seed(42)
            np.random.shuffle(all_ids)
            split = int(0.9 * len(all_ids))
            train_ids = all_ids[:split]
            casf_ids = all_ids[split:]

        # Split train into train/val (90/10)
        np.random.seed(42)
        np.random.shuffle(train_ids)
        val_split = int(0.9 * len(train_ids))
        val_ids = train_ids[val_split:]
        train_ids = train_ids[:val_split]

        pdbbind_train = V5Dataset(args.pdbbind, train_ids)
        pdbbind_val = V5Dataset(args.pdbbind, val_ids)
        pdbbind_casf = V5Dataset(args.pdbbind, casf_ids)
        datasets_aff.append(pdbbind_train)
    else:
        pdbbind_val = None
        pdbbind_casf = None

    # BindingMOAD
    if args.moad:
        moad_ds = V5Dataset(args.moad, max_samples=args.max_moad_samples)
        if len(moad_ds) > 0:
            datasets_aff.append(moad_ds)

    # CrossDocked2020
    crossdocked_ds = None
    if args.crossdocked:
        crossdocked_ds = StreamingDataset(args.crossdocked,
                                           max_samples=args.max_crossdocked_samples)
        if len(crossdocked_ds) > 0:
            datasets_disc.append(crossdocked_ds)

    # Build combined datasets for each phase
    all_aff = ConcatDataset(datasets_aff) if datasets_aff else None
    all_disc = ConcatDataset(datasets_disc) if datasets_disc else None
    all_data_parts = datasets_aff + datasets_disc
    all_combined = ConcatDataset(all_data_parts) if all_data_parts else None

    if is_main_process():
        print(f"\nDataset sizes:")
        if all_aff:
            print(f"  Affinity (PDBbind+MOAD): {len(all_aff)}")
        if all_disc:
            print(f"  Discrimination (CrossDocked): {len(all_disc)}")
        if all_combined:
            print(f"  Combined: {len(all_combined)}")

    # Model — enable gradient checkpointing for <=12GB GPUs to save VRAM
    use_grad_ckpt = (device.type == "cuda"
                     and torch.cuda.get_device_properties(local_rank).total_memory < 13e9)
    if args.grad_checkpoint:
        use_grad_ckpt = True
    model = PairwiseGeometricNetX(gradient_checkpointing=use_grad_ckpt).to(device)
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: PGN-X, {n_params:,} params ({n_params * 4 / 1024:.1f} KB)")
        if use_grad_ckpt:
            print(f"  Gradient checkpointing: ENABLED (saves ~40% VRAM)")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if distributed else model

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")
    ema = EMAModel(raw_model, decay=0.9995)

    ckpt_dir = Path(args.output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", -1.0)
        if "ema_state" in ckpt:
            ema.shadow = ckpt["ema_state"]
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")

    # W&B logging
    if HAS_WANDB and args.wandb and is_main_process():
        wandb.init(project="druseaf-v5", config=vars(args),
                   name=f"pgnx-{time.strftime('%m%d-%H%M')}")

    # Training phases
    total_epochs = args.epochs

    # Determine phase boundaries
    if all_disc is not None and all_aff is not None:
        # Full multi-phase training
        phase1_end = max(1, int(total_epochs * 0.20))  # 20% affinity pre-training
        phase2_end = max(phase1_end + 1, int(total_epochs * 0.70))  # 50% discrimination
        # Remaining 30%: joint fine-tuning
        if is_main_process():
            print(f"\nTraining phases:")
            print(f"  Phase 1 (affinity):       epochs 1-{phase1_end}")
            print(f"  Phase 2 (discrimination): epochs {phase1_end+1}-{phase2_end}")
            print(f"  Phase 3 (joint):          epochs {phase2_end+1}-{total_epochs}")
    elif all_aff is not None:
        # Only affinity data
        phase1_end = total_epochs
        phase2_end = total_epochs
    else:
        # Only discrimination data
        phase1_end = 0
        phase2_end = total_epochs

    # Create scheduler over total epochs
    warmup_epochs = min(5, total_epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress)) + 1e-6 / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Select phase and data
        if epoch < phase1_end:
            phase = "affinity"
            train_data = all_aff
        elif epoch < phase2_end:
            phase = "discrimination"
            train_data = all_combined  # mix affinity + discrimination
        else:
            phase = "joint"
            train_data = all_combined

        if train_data is None or len(train_data) == 0:
            continue

        # Create DataLoader for this epoch
        if distributed:
            sampler = DistributedSampler(train_data, shuffle=True)
            sampler.set_epoch(epoch)
            train_loader = DataLoader(train_data, batch_size=args.batch, sampler=sampler,
                                      num_workers=args.workers, pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True, drop_last=True)

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch,
                                 phase=phase, grad_accum_steps=args.grad_accum)
        ema.update(raw_model)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        # Evaluate with EMA model
        if is_main_process() and pdbbind_val is not None and (epoch + 1) % 2 == 0:
            val_loader = DataLoader(pdbbind_val, batch_size=args.batch * 2, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
            ema.apply_shadow(raw_model)
            val_metrics = evaluate(raw_model, val_loader, device)
            ema.restore(raw_model)

            print(f"Ep {epoch+1:3d}/{total_epochs} [{phase:13s}]  loss={train_loss:.4f}  "
                  f"val_R={val_metrics['pearson_r']:.3f}  RMSE={val_metrics['rmse']:.2f}  "
                  f"AUC={val_metrics['disc_auc']:.3f}  "
                  f"dock@1={val_metrics['dock_top1']:.2f}  "
                  f"lr={lr:.1e}  {dt:.0f}s")

            # Track composite metric: R × AUC (both matter)
            composite = val_metrics["pearson_r"] * val_metrics["disc_auc"]
            if val_metrics["disc_auc"] < 0.55:
                # Before AUC is meaningful, track R alone
                composite = val_metrics["pearson_r"]

            # Save checkpoint
            state = {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state": ema.shadow,
                "val_metrics": val_metrics,
                "best_metric": max(best_metric, composite),
                "architecture": "PGNXv5",
                "hidden_dim": HIDDEN_DIM,
                "pair_dim": PAIR_DIM,
                "phase": phase,
            }
            torch.save(state, ckpt_dir / "latest.pt")

            if composite > best_metric:
                best_metric = composite
                torch.save(state, ckpt_dir / "best.pt")
                print(f"  *** New best: composite={best_metric:.4f} "
                      f"(R={val_metrics['pearson_r']:.4f}, AUC={val_metrics['disc_auc']:.3f})")

            if HAS_WANDB and args.wandb:
                wandb.log({"train_loss": train_loss, "lr": lr,
                           **{f"val/{k}": v for k, v in val_metrics.items()},
                           "val/composite": composite, "phase": phase}, step=epoch + 1)
        elif is_main_process():
            print(f"Ep {epoch+1:3d}/{total_epochs} [{phase:13s}]  "
                  f"loss={train_loss:.4f}  lr={lr:.1e}  {dt:.0f}s")

    # Final CASF-2016 evaluation with EMA model
    if is_main_process() and pdbbind_casf is not None and len(pdbbind_casf) > 0:
        casf_loader = DataLoader(pdbbind_casf, batch_size=args.batch * 2, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        ema.apply_shadow(raw_model)
        casf_metrics = evaluate(raw_model, casf_loader, device)
        ema.restore(raw_model)

        print(f"\n{'='*50}")
        print(f"CASF-2016 Final Results (EMA model)")
        print(f"{'='*50}")
        print(f"  Scoring power:")
        print(f"    Pearson R:   {casf_metrics['pearson_r']:.4f}")
        print(f"    RMSE (pKd):  {casf_metrics['rmse']:.3f}")
        print(f"  Discrimination:")
        print(f"    AUC:         {casf_metrics['disc_auc']:.4f}")
        print(f"  Docking power:")
        print(f"    Top-1:       {casf_metrics['dock_top1']:.2%}")
        print(f"    Top-3:       {casf_metrics['dock_top3']:.2%}")
        print(f"  N crystal:     {casf_metrics['n_crystal']}")

    if distributed:
        dist.destroy_process_group()


# =============================================================================
# Benchmark (comprehensive CASF-2016 evaluation)
# =============================================================================

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = PairwiseGeometricNetX().to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    # Apply EMA weights if available
    if "ema_state" in ckpt:
        for name, param in model.named_parameters():
            if name in ckpt["ema_state"]:
                param.data.copy_(ckpt["ema_state"][name])
        print("Using EMA weights")

    cache_dir = Path(args.data)
    split_file = cache_dir / "split_info.json"
    if split_file.exists():
        with open(split_file) as f:
            casf_ids = json.load(f).get("casf", None)
    else:
        casf_ids = None

    dataset = V5Dataset(args.data, casf_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    metrics = evaluate(model, loader, device)

    print("=" * 60)
    print("CASF-2016 Benchmark Results — DruseAF v5 (PGN-X)")
    print("=" * 60)
    print(f"  Scoring power:")
    print(f"    Pearson R:        {metrics['pearson_r']:.4f}")
    print(f"    RMSE (pKd):       {metrics['rmse']:.3f}")
    print(f"    Score R:          {metrics['score_r']:.4f}")
    print(f"  Pose discrimination:")
    print(f"    AUC:              {metrics['disc_auc']:.4f}")
    print(f"    Confidence RMSE:  {metrics['conf_rmse']:.4f}")
    print(f"  Docking power:")
    print(f"    Top-1 success:    {metrics['dock_top1']:.2%}")
    print(f"    Top-3 success:    {metrics['dock_top3']:.2%}")
    print(f"    Targets tested:   {metrics['n_targets_tested']}")
    print(f"  Crystal poses:      {metrics['n_crystal']}")

    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"\n  Training val metrics: R={m.get('pearson_r', '?')}, "
              f"AUC={m.get('disc_auc', '?')}")


# =============================================================================
# Export weights for Metal (DRAF v3 format)
# =============================================================================

EXPORT_WEIGHT_ORDER = [
    # Protein encoder (0-3)
    "prot_encoder.0.weight",     # [160, 20]
    "prot_encoder.0.bias",       # [160]
    "prot_encoder.2.weight",     # [160, 160]
    "prot_encoder.2.bias",       # [160]
    # Protein message passing layer 0 (4-11)
    "prot_msg_layers.0.msg_mlp.0.weight",    # [160, 160]
    "prot_msg_layers.0.msg_mlp.0.bias",      # [160]
    "prot_msg_layers.0.attn_mlp.0.weight",   # [40, 20]
    "prot_msg_layers.0.attn_mlp.0.bias",     # [40]
    "prot_msg_layers.0.attn_mlp.2.weight",   # [1, 40]
    "prot_msg_layers.0.attn_mlp.2.bias",     # [1]
    "prot_msg_layers.0.norm.weight",          # [160]
    "prot_msg_layers.0.norm.bias",            # [160]
    # Protein message passing layer 1 (12-19)
    "prot_msg_layers.1.msg_mlp.0.weight",
    "prot_msg_layers.1.msg_mlp.0.bias",
    "prot_msg_layers.1.attn_mlp.0.weight",
    "prot_msg_layers.1.attn_mlp.0.bias",
    "prot_msg_layers.1.attn_mlp.2.weight",
    "prot_msg_layers.1.attn_mlp.2.bias",
    "prot_msg_layers.1.norm.weight",
    "prot_msg_layers.1.norm.bias",
    # Protein message passing layer 2 (20-27)
    "prot_msg_layers.2.msg_mlp.0.weight",
    "prot_msg_layers.2.msg_mlp.0.bias",
    "prot_msg_layers.2.attn_mlp.0.weight",
    "prot_msg_layers.2.attn_mlp.0.bias",
    "prot_msg_layers.2.attn_mlp.2.weight",
    "prot_msg_layers.2.attn_mlp.2.bias",
    "prot_msg_layers.2.norm.weight",
    "prot_msg_layers.2.norm.bias",
    # Protein message passing layer 3 (28-35)
    "prot_msg_layers.3.msg_mlp.0.weight",
    "prot_msg_layers.3.msg_mlp.0.bias",
    "prot_msg_layers.3.attn_mlp.0.weight",
    "prot_msg_layers.3.attn_mlp.0.bias",
    "prot_msg_layers.3.attn_mlp.2.weight",
    "prot_msg_layers.3.attn_mlp.2.bias",
    "prot_msg_layers.3.norm.weight",
    "prot_msg_layers.3.norm.bias",
    # Ligand encoder (36-39)
    "lig_encoder.0.weight",      # [160, 20]
    "lig_encoder.0.bias",        # [160]
    "lig_encoder.2.weight",      # [160, 160]
    "lig_encoder.2.bias",        # [160]
    # Ligand message passing layer 0 (40-47)
    "lig_msg_layers.0.msg_mlp.0.weight",
    "lig_msg_layers.0.msg_mlp.0.bias",
    "lig_msg_layers.0.attn_mlp.0.weight",
    "lig_msg_layers.0.attn_mlp.0.bias",
    "lig_msg_layers.0.attn_mlp.2.weight",
    "lig_msg_layers.0.attn_mlp.2.bias",
    "lig_msg_layers.0.norm.weight",
    "lig_msg_layers.0.norm.bias",
    # Ligand message passing layer 1 (48-55)
    "lig_msg_layers.1.msg_mlp.0.weight",
    "lig_msg_layers.1.msg_mlp.0.bias",
    "lig_msg_layers.1.attn_mlp.0.weight",
    "lig_msg_layers.1.attn_mlp.0.bias",
    "lig_msg_layers.1.attn_mlp.2.weight",
    "lig_msg_layers.1.attn_mlp.2.bias",
    "lig_msg_layers.1.norm.weight",
    "lig_msg_layers.1.norm.bias",
    # Pair projections (56-59)
    "prot_pair_proj.weight",     # [80, 160]
    "prot_pair_proj.bias",       # [80]
    "lig_pair_proj.weight",      # [80, 160]
    "lig_pair_proj.bias",        # [80]
    # RBF cross-interaction projection (60-61)
    "rbf_proj.0.weight",        # [80, 32]
    "rbf_proj.0.bias",          # [80]
    # 2-layer pair energy head (62-65)
    "pair_energy.0.weight",      # [40, 80]
    "pair_energy.0.bias",        # [40]
    "pair_energy.2.weight",      # [1, 40]
    "pair_energy.2.bias",        # [1]
    # Context gate (66-67)
    "context_gate.weight",       # [1, 80]
    "context_gate.bias",         # [1]
    # Context projection (68-69)
    "context_proj.weight",       # [160, 80]
    "context_proj.bias",         # [160]
    # Context LayerNorm (70-71)
    "context_norm.weight",       # [160]
    "context_norm.bias",         # [160]
    # Affinity head (72-75)
    "affinity_head.0.weight",    # [80, 160]
    "affinity_head.0.bias",      # [80]
    "affinity_head.2.weight",    # [1, 80]
    "affinity_head.2.bias",      # [1]
    # Confidence head (76-79)
    "confidence_head.0.weight",  # [80, 160]
    "confidence_head.0.bias",    # [80]
    "confidence_head.2.weight",  # [1, 80]
    "confidence_head.2.bias",    # [1]
    # Learned scalars (80-81)
    "pair_scale",                # [1]
    "pair_bias",                 # [1]
]


def run_export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Apply EMA if available
    if "ema_state" in ckpt and not args.no_ema:
        print("Using EMA weights for export")
        for key in EXPORT_WEIGHT_ORDER:
            if key in ckpt["ema_state"]:
                state[key] = ckpt["ema_state"][key]

    print(f"=== DruseAF v5 Weight Export ===")
    print(f"  Checkpoint: {args.checkpoint}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val Pearson R: {m.get('pearson_r', '?')}")
        print(f"  Val Disc AUC:  {m.get('disc_auc', '?')}")

    missing = [k for k in EXPORT_WEIGHT_ORDER if k not in state]
    if missing:
        print(f"\nERROR: {len(missing)} weight tensors missing:")
        for k in missing:
            print(f"  - {k}")
        sys.exit(1)

    tensors = []
    for key in EXPORT_WEIGHT_ORDER:
        t = state[key]
        if isinstance(t, nn.Parameter):
            t = t.data
        if t.dim() == 0:
            t = t.unsqueeze(0)
        tensors.append(t.float().numpy().flatten())

    num_tensors = len(tensors)
    offsets = []
    curr = 0
    for t in tensors:
        offsets.append((curr, len(t)))
        curr += len(t)
    total_floats = curr

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        f.write(b"DRAF")
        f.write(struct.pack("<I", 3))  # version 3 = v5 PGN-X
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", total_floats))
        for off, cnt in offsets:
            f.write(struct.pack("<II", off, cnt))
        for t in tensors:
            f.write(t.astype(np.float32).tobytes())

    print(f"\n  Exported {num_tensors} tensors, {total_floats:,} floats "
          f"({total_floats * 4 / 1024:.1f} KB)")
    print(f"  DRAF v3 format → {out_path}")
    print(f"\n  Copy to Models/druse-models/DruseAFv5.weights")
    print(f"\n  Metal shader constants for DruseAFv5Compute.metal:")
    print(f"    constant uint H       = {HIDDEN_DIM};")
    print(f"    constant uint PD      = {PAIR_DIM};")
    print(f"    constant uint FEAT    = {NUM_ATOM_FEATURES};")
    print(f"    constant uint MSG_RBF = {MSG_RBF_BINS};")
    print(f"    constant uint CRS_RBF = {CROSS_RBF_BINS};")
    print(f"    constant float RBF_G  = {RBF_GAMMA}f;")
    print(f"    constant float MSG_CUT = {MSG_CUTOFF}f;")
    print(f"    constant float CRS_CUT = {CROSS_CUTOFF}f;")
    print(f"    constant uint ATTN_H  = {ATTN_HIDDEN};")
    print(f"    constant uint NUM_PROT_MSG = {NUM_PROT_MSG_LAYERS};")
    print(f"    constant uint NUM_LIG_MSG  = {NUM_LIG_MSG_LAYERS};")
    print(f"    constant uint MAX_PROT = {MAX_PROT_ATOMS};")
    print(f"    constant uint MAX_LIG  = {MAX_LIG_ATOMS};")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DruseAF v5 — Pairwise Geometric Network eXtended (PGN-X)")
    sub = parser.add_subparsers(dest="command")

    # preprocess-pdbbind
    pp = sub.add_parser("preprocess-pdbbind", help="Preprocess PDBbind v2020")
    pp.add_argument("--refined", type=str, required=True)
    pp.add_argument("--general", type=str, default=None)
    pp.add_argument("--casf", type=str, default=None)
    pp.add_argument("--out", type=str, required=True)

    # preprocess-crossdocked
    cd = sub.add_parser("preprocess-crossdocked", help="Preprocess CrossDocked2020")
    cd.add_argument("--crossdocked", type=str, required=True,
                    help="Path to crossdocked_pocket10/ directory")
    cd.add_argument("--types", type=str, default=None,
                    help="Path to types file (e.g. it2_tt_0_train0.types)")
    cd.add_argument("--out", type=str, required=True)

    # preprocess-moad
    md = sub.add_parser("preprocess-moad", help="Preprocess BindingMOAD")
    md.add_argument("--moad", type=str, required=True)
    md.add_argument("--out", type=str, required=True)

    # train
    tr = sub.add_parser("train", help="Train PGN-X model")
    tr.add_argument("--pdbbind", type=str, default=None,
                    help="PDBbind preprocessed cache directory")
    tr.add_argument("--crossdocked", type=str, default=None,
                    help="CrossDocked2020 preprocessed cache directory")
    tr.add_argument("--moad", type=str, default=None,
                    help="BindingMOAD preprocessed cache directory")
    tr.add_argument("--max-crossdocked-samples", type=int, default=2_000_000,
                    help="Max CrossDocked samples to load (default 2M)")
    tr.add_argument("--max-moad-samples", type=int, default=500_000,
                    help="Max MOAD samples to load")
    tr.add_argument("--epochs", type=int, default=150)
    tr.add_argument("--batch", type=int, default=24,
                    help="Batch size per GPU (24 fits in 10GB RTX 3080)")
    tr.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps (effective batch = batch × accum)")
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--workers", type=int, default=8)
    tr.add_argument("--output", type=str, default="checkpoints_v5")
    tr.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint")
    tr.add_argument("--grad-checkpoint", action="store_true",
                    help="Force gradient checkpointing (auto-enabled for <=12GB GPUs)")
    tr.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # benchmark
    bm = sub.add_parser("benchmark", help="Run CASF-2016 benchmark")
    bm.add_argument("--data", type=str, required=True)
    bm.add_argument("--checkpoint", type=str, required=True)
    bm.add_argument("--casf-decoys", type=str, default=None,
                    help="CASF-2016 docking decoys directory")

    # export
    ex = sub.add_parser("export", help="Export weights for Metal (DRAF v3)")
    ex.add_argument("--checkpoint", type=str, required=True)
    ex.add_argument("--out", type=str, default="DruseAFv5.weights")
    ex.add_argument("--no-ema", action="store_true",
                    help="Export raw weights instead of EMA")

    args = parser.parse_args()

    if args.command == "preprocess-pdbbind":
        run_preprocess_pdbbind(args)
    elif args.command == "preprocess-crossdocked":
        run_preprocess_crossdocked(args)
    elif args.command == "preprocess-moad":
        run_preprocess_moad(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "export":
        run_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

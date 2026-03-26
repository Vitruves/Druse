#!/usr/bin/env python3
"""
DruseAF v4 — Pairwise Geometric Network (PGN)
CUDA-accelerated training on PDBbind v2020

Architecture (134K params, ~536 KB weights):
  Protein: MLP(20→128) + 3 rounds distance-attention message passing
  Ligand:  MLP(20→128)
  Scoring: Hadamard pairwise interaction within 8Å cutoff
  Output:  pKd (pair-decomposed + global) × confidence

Speed at Metal inference:
  Setup: ~2ms (encode + msg passing, once per target)
  Per pose: ~0.02ms (vs ~0.5ms for DruseAF v3 cross-attention)

Usage:
  # 1. Preprocess PDBbind into .pt cache
  python trainDruseAFv4.py preprocess \
      --refined /path/to/PDBbind_v2020_refined \
      --general /path/to/PDBbind_v2020_other_PL \
      --casf /path/to/CASF-2016/coreset \
      --out data/v4_cache

  # 2. Train
  python trainDruseAFv4.py train \
      --data data/v4_cache --epochs 120 --batch 32 --lr 5e-4

  # 3. Export weights for Metal
  python trainDruseAFv4.py export \
      --checkpoint checkpoints_v4/best.pt --out ../Models/druse-models/DruseAFv4.weights

  # 4. Benchmark on CASF-2016
  python trainDruseAFv4.py benchmark \
      --data data/v4_cache --checkpoint checkpoints_v4/best.pt
"""

import os
import sys
import argparse
import math
import struct
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist as scipy_cdist
from tqdm import tqdm


# =============================================================================
# Constants — must match Metal shader exactly
# =============================================================================

NUM_ATOM_FEATURES = 20
HIDDEN_DIM = 128
PAIR_DIM = 64
MSG_RBF_BINS = 16
CROSS_RBF_BINS = 24
MSG_CUTOFF = 8.0
CROSS_CUTOFF = 8.0
RBF_GAMMA = 2.0
NUM_MSG_LAYERS = 3
MAX_PROT_ATOMS = 256
MAX_LIG_ATOMS = 64
POCKET_RADIUS = 10.0

PERTURBATION_RMSDS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
CONFIDENCE_SIGMA = 2.0  # for label = exp(-rmsd²/(2σ²))

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
# Feature extraction (matches Swift/Metal exactly)
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
    feat[17] = 1.0  # is_ligand
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


# =============================================================================
# PDB / MOL2 parsing
# =============================================================================

def parse_pdb_atoms(path: str):
    """Parse ATOM/HETATM records from PDB file. Returns (positions, features, elements)."""
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
    """Parse @<TRIPOS>ATOM section from MOL2 file. Returns (positions, features, elements)."""
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


# =============================================================================
# Pose perturbation
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
# Preprocessing
# =============================================================================

def preprocess_complex(pdb_id: str, prot_pdb: str, lig_mol2: str, pkd: float,
                       out_dir: Path, pocket_radius: float = POCKET_RADIUS):
    """Preprocess one protein-ligand complex into .pt files (one per RMSD level)."""
    prot_pos, prot_feat, _ = parse_pdb_atoms(prot_pdb)
    lig_pos, lig_feat, _ = parse_mol2_atoms(lig_mol2)
    if prot_pos is None or lig_pos is None:
        return 0
    if lig_pos.shape[0] > MAX_LIG_ATOMS or lig_pos.shape[0] < 3:
        return 0

    # Extract pocket: atoms within pocket_radius of ligand centroid
    lig_center = lig_pos.mean(axis=0)
    dists = np.linalg.norm(prot_pos - lig_center, axis=1)
    pocket_mask = dists < pocket_radius
    prot_pos = prot_pos[pocket_mask]
    prot_feat = prot_feat[pocket_mask]
    if prot_pos.shape[0] < 10:
        return 0

    # Trim to max protein atoms (keep closest)
    if prot_pos.shape[0] > MAX_PROT_ATOMS:
        dists_pocket = np.linalg.norm(prot_pos - lig_center, axis=1)
        keep_idx = np.argsort(dists_pocket)[:MAX_PROT_ATOMS]
        prot_pos = prot_pos[keep_idx]
        prot_feat = prot_feat[keep_idx]

    # Center positions on ligand centroid
    prot_pos_centered = prot_pos - lig_center
    lig_pos_centered = lig_pos - lig_center

    count = 0
    for target_rmsd in PERTURBATION_RMSDS:
        pert_pos, actual_rmsd = generate_perturbed_pose(lig_pos_centered, target_rmsd)
        conf_label = math.exp(-(actual_rmsd ** 2) / (2.0 * CONFIDENCE_SIGMA ** 2))

        sample = {
            "pdb_id": pdb_id,
            "prot_pos": torch.from_numpy(prot_pos_centered),
            "prot_feat": torch.from_numpy(prot_feat),
            "lig_pos": torch.from_numpy(pert_pos),
            "lig_feat": torch.from_numpy(lig_feat),
            "pkd": torch.tensor(pkd, dtype=torch.float32),
            "rmsd": torch.tensor(actual_rmsd, dtype=torch.float32),
            "confidence": torch.tensor(conf_label, dtype=torch.float32),
        }
        fname = f"{pdb_id}_rmsd{target_rmsd:.1f}.pt"
        torch.save(sample, out_dir / fname)
        count += 1
    return count


def run_preprocess(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect pKd values from index files
    pkd_map = {}
    for data_dir in [args.refined, args.general]:
        if data_dir is None:
            continue
        data_path = Path(data_dir)
        # Try common index file locations
        for idx_name in ["INDEX_refined_data.2020", "INDEX_general_PL_data.2020",
                         "index/INDEX_refined_data.2020", "index/INDEX_general_PL_data.2020",
                         "INDEX_general_PL.2020", "INDEX_refined_set.2020"]:
            idx_file = data_path / idx_name
            if idx_file.exists():
                pkd_map.update(parse_index_file(str(idx_file)))
        # Also check parent directory
        for idx_file in data_path.parent.glob("INDEX*"):
            pkd_map.update(parse_index_file(str(idx_file)))

    print(f"Loaded {len(pkd_map)} pKd values from index files")

    # Get CASF-2016 IDs to exclude from training
    casf_ids = set()
    if args.casf:
        casf_path = Path(args.casf)
        for d in casf_path.iterdir():
            if d.is_dir() and len(d.name) == 4:
                casf_ids.add(d.name.lower())
    print(f"CASF-2016 exclusion set: {len(casf_ids)} complexes")

    # Process all complexes
    total = 0
    errors = 0
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
                n = preprocess_complex(pdb_id, str(prot_pdb), str(lig_mol2),
                                       pkd_map[pdb_id], out_dir)
                total += n
                if pdb_id in casf_ids:
                    split_info["casf"].append(pdb_id)
                else:
                    split_info["train"].append(pdb_id)
            except Exception as e:
                errors += 1

    # Also process CASF complexes directly if separate directory
    if args.casf:
        casf_path = Path(args.casf)
        for cdir in sorted(casf_path.iterdir()):
            if not cdir.is_dir() or len(cdir.name) != 4:
                continue
            pdb_id = cdir.name.lower()
            if pdb_id not in pkd_map:
                continue
            # Skip if already processed from refined/general
            if (out_dir / f"{pdb_id}_rmsd0.0.pt").exists():
                continue
            prot_pdb = cdir / f"{pdb_id}_pocket.pdb"
            if not prot_pdb.exists():
                prot_pdb = cdir / f"{pdb_id}_protein.pdb"
            lig_mol2 = cdir / f"{pdb_id}_ligand.mol2"
            if not prot_pdb.exists() or not lig_mol2.exists():
                continue
            try:
                n = preprocess_complex(pdb_id, str(prot_pdb), str(lig_mol2),
                                       pkd_map[pdb_id], out_dir)
                total += n
                if pdb_id not in split_info["casf"]:
                    split_info["casf"].append(pdb_id)
            except Exception:
                pass

    # Save split info
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nPreprocessing complete:")
    print(f"  Total samples: {total}")
    print(f"  Train proteins: {len(split_info['train'])}")
    print(f"  CASF proteins: {len(split_info['casf'])}")
    print(f"  Errors/skipped: {errors}")


# =============================================================================
# Dataset
# =============================================================================

class PDBBindV4Dataset(Dataset):
    def __init__(self, cache_dir: str, pdb_ids: Optional[list] = None):
        cache_path = Path(cache_dir)
        files = sorted(cache_path.glob("*.pt"))
        if pdb_ids is not None:
            id_set = set(pdb_ids)
            files = [f for f in files if f.stem.split("_rmsd")[0] in id_set]
        self.samples = []
        for f in tqdm(files, desc="Loading cache"):
            self.samples.append(torch.load(f, weights_only=False))
        print(f"  Loaded {len(self.samples)} samples")

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

        return {
            "pdb_id": s["pdb_id"],
            "prot_pos": prot_pos, "prot_feat": prot_feat, "prot_mask": prot_mask,
            "lig_pos": lig_pos, "lig_feat": lig_feat, "lig_mask": lig_mask,
            "pkd": s["pkd"], "rmsd": s["rmsd"], "confidence": s["confidence"],
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
# Model: Message Passing Layer (protein intra-graph)
# =============================================================================

class MessagePassingLayer(nn.Module):
    """Distance-attention message passing for protein encoding.

    Metal equivalent: druseAFv4MsgTransform + druseAFv4MsgAggregate kernels.
    Weight tensors per layer: msg_mlp (w,b), attn_mlp (w1,b1,w2,b2), norm (w,b) = 8
    """

    def __init__(self, hidden: int = HIDDEN_DIM, rbf_bins: int = MSG_RBF_BINS,
                 cutoff: float = MSG_CUTOFF):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = GaussianRBF(rbf_bins, cutoff)
        self.attn_mlp = nn.Sequential(
            nn.Linear(rbf_bins, 32),
            nn.GELU(approximate="tanh"),
            nn.Linear(32, 1),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, mask):
        """
        h: [B, P, H]  pos: [B, P, 3]  mask: [B, P] bool
        """
        B, P, _ = pos.shape
        dist = torch.cdist(pos, pos)  # [B, P, P]
        nbr_mask = (dist < self.cutoff) & (dist > 0.01)
        nbr_mask = nbr_mask & mask.unsqueeze(1) & mask.unsqueeze(2)

        rbf = self.rbf(dist)  # [B, P, P, rbf_bins]
        attn = self.attn_mlp(rbf).squeeze(-1)  # [B, P, P]
        attn = attn.masked_fill(~nbr_mask, -6e4)
        attn = F.softmax(attn, dim=-1)  # [B, P, P]

        transformed = self.msg_mlp(h)  # [B, P, H]
        msg = torch.bmm(attn, transformed)  # [B, P, H]
        return self.norm(h + msg)


# =============================================================================
# Model: Pairwise Geometric Network
# =============================================================================

class PairwiseGeometricNet(nn.Module):
    """DruseAF v4 — Pairwise Geometric Network (PGN).

    134K parameters. Weight tensor layout (56 tensors total):
      [0-3]   Protein encoder MLP
      [4-27]  3× message passing layers (8 tensors each)
      [28-31] Ligand encoder MLP
      [32-33] Protein pair projection
      [34-35] Ligand pair projection
      [36-37] RBF cross-interaction projection
      [38-39] Pair energy head (GELU→Linear)
      [40-41] Context gate
      [42-43] Context projection
      [44-45] Context LayerNorm
      [46-49] Affinity head
      [50-53] Confidence head
      [54]    pair_scale scalar
      [55]    pair_bias scalar
    """

    def __init__(self):
        super().__init__()

        # Protein encoder: MLP(20→128→128)
        self.prot_encoder = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        # 3 rounds of intra-protein message passing
        self.msg_layers = nn.ModuleList([
            MessagePassingLayer() for _ in range(NUM_MSG_LAYERS)
        ])
        # Ligand encoder: MLP(20→128→128)
        self.lig_encoder = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        # Pair projections
        self.prot_pair_proj = nn.Linear(HIDDEN_DIM, PAIR_DIM)
        self.lig_pair_proj = nn.Linear(HIDDEN_DIM, PAIR_DIM)
        # Cross-interaction RBF projection
        self.rbf = GaussianRBF(CROSS_RBF_BINS, CROSS_CUTOFF)
        self.rbf_proj = nn.Sequential(
            nn.Linear(CROSS_RBF_BINS, PAIR_DIM),
            nn.GELU(approximate="tanh"),
        )
        # Pair energy: GELU → Linear(64→1)
        self.pair_energy = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(PAIR_DIM, 1),
        )
        # Context aggregation
        self.context_gate = nn.Linear(PAIR_DIM, 1)
        self.context_proj = nn.Linear(PAIR_DIM, HIDDEN_DIM)
        self.context_norm = nn.LayerNorm(HIDDEN_DIM)
        # Prediction heads
        self.affinity_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, 1),
        )
        # Learned pair energy scaling
        self.pair_scale = nn.Parameter(torch.tensor(0.1))
        self.pair_bias = nn.Parameter(torch.tensor(6.0))

    def forward(self, prot_pos, prot_feat, lig_pos, lig_feat, prot_mask, lig_mask):
        """
        All inputs: [B, P/L, D]  masks: [B, P/L] bool
        Returns: pKd [B], confidence [B], pair_energies [B, L, P]
        """
        B = prot_pos.shape[0]
        P = MAX_PROT_ATOMS
        L = MAX_LIG_ATOMS

        # Encode atoms
        prot_h = self.prot_encoder(prot_feat)  # [B, P, H]
        lig_h = self.lig_encoder(lig_feat)      # [B, L, H]

        # Protein message passing (3 rounds)
        for layer in self.msg_layers:
            prot_h = layer(prot_h, prot_pos, prot_mask)

        # Pair projections
        prot_p = self.prot_pair_proj(prot_h)  # [B, P, PD]
        lig_p = self.lig_pair_proj(lig_h)      # [B, L, PD]

        # Cross distances + RBF
        cross_dist = torch.cdist(lig_pos, prot_pos)  # [B, L, P]
        cross_rbf = self.rbf(cross_dist)               # [B, L, P, rbf_bins]
        rbf_p = self.rbf_proj(cross_rbf)               # [B, L, P, PD]

        # Cutoff mask
        cross_mask = (cross_dist < CROSS_CUTOFF) & lig_mask.unsqueeze(2) & prot_mask.unsqueeze(1)

        # Hadamard pairwise interaction: lig ⊙ prot ⊙ rbf
        pair = lig_p.unsqueeze(2) * prot_p.unsqueeze(1) * rbf_p  # [B, L, P, PD]

        # Pair energy: GELU(pair) → Linear → scalar per pair
        pair_e = self.pair_energy(pair).squeeze(-1)  # [B, L, P]
        pair_e = pair_e * cross_mask.float()

        # Context gate: pair → Linear → scalar (no GELU before gate)
        gate_logit = self.context_gate(pair).squeeze(-1)  # [B, L, P]
        gate_logit = gate_logit.masked_fill(~cross_mask, -6e4)
        gate_weight = F.softmax(gate_logit, dim=2)  # per lig atom, over prot

        # Context: weighted protein features per ligand atom
        context = torch.einsum("blp,bpd->bld", gate_weight, prot_p)  # [B, L, PD]
        lig_h_ctx = self.context_norm(lig_h + self.context_proj(context))  # [B, L, H]

        # Pool + predict
        lig_h_ctx = lig_h_ctx * lig_mask.unsqueeze(-1).float()
        n_lig = lig_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        complex_repr = lig_h_ctx.sum(dim=1) / n_lig  # [B, H]

        total_pair_energy = pair_e.sum(dim=(1, 2))  # [B]
        pKd = total_pair_energy * self.pair_scale + self.affinity_head(complex_repr).squeeze(-1) + self.pair_bias
        confidence = torch.sigmoid(self.confidence_head(complex_repr).squeeze(-1))

        return pKd, confidence, pair_e


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(loader, desc=f"  Train ep{epoch+1}", leave=False):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.float16):
            pKd_pred, conf_pred, pair_e = model(
                batch["prot_pos"], batch["prot_feat"],
                batch["lig_pos"], batch["lig_feat"],
                batch["prot_mask"], batch["lig_mask"])

            # 1. Affinity loss: RMSD-weighted Huber
            rmsd = batch["rmsd"]
            pose_weight = torch.exp(-rmsd / 3.0)
            loss_aff = (pose_weight * F.huber_loss(
                pKd_pred, batch["pkd"], reduction="none", delta=2.0))
            loss_aff = loss_aff.sum() / pose_weight.sum().clamp(min=1.0)

            # 2. Confidence loss: MSE
            loss_conf = F.mse_loss(conf_pred, batch["confidence"])

            # 3. Ranking loss: within same protein, better RMSD → higher score
            pred_scores = pKd_pred * conf_pred
            B = pred_scores.shape[0]
            pdb_ids = batch["pdb_id"]
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
            valid = same & triu & (rmsd_diff != 0)

            if valid.any():
                score_diff = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)
                sign = torch.sign(-rmsd_diff)  # +1 when i is better
                loss_rank = F.relu(0.5 - sign * score_diff)[valid].mean()
            else:
                loss_rank = torch.zeros(1, device=device).squeeze()

            # 4. Pair energy regularization
            loss_pair_reg = pair_e.abs().mean() * 0.01

            loss = 1.0 * loss_aff + 5.0 * loss_conf + 2.0 * loss_rank + loss_pair_reg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pkd_true, all_pkd_pred, all_conf_pred, all_rmsd = [], [], [], []
    all_score_pred, all_score_true = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
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

    pkd_true = torch.cat(all_pkd_true)
    pkd_pred = torch.cat(all_pkd_pred)
    conf_pred = torch.cat(all_conf_pred)
    rmsd = torch.cat(all_rmsd)
    score_pred = torch.cat(all_score_pred)
    score_true = torch.cat(all_score_true)

    # Crystal poses only (RMSD ≈ 0)
    crystal_mask = rmsd < 0.1
    if crystal_mask.sum() > 5:
        crystal_pkd_true = pkd_true[crystal_mask]
        crystal_pkd_pred = pkd_pred[crystal_mask]
        r = torch.corrcoef(torch.stack([crystal_pkd_true, crystal_pkd_pred]))[0, 1].item()
        rmse = (crystal_pkd_true - crystal_pkd_pred).pow(2).mean().sqrt().item()
    else:
        r, rmse = 0.0, 99.0

    # Confidence accuracy (all poses)
    conf_true = torch.exp(-rmsd ** 2 / (2 * CONFIDENCE_SIGMA ** 2))
    conf_rmse = (conf_pred - conf_true).pow(2).mean().sqrt().item()

    # Score correlation (all poses)
    score_r = torch.corrcoef(torch.stack([score_true, score_pred]))[0, 1].item()

    return {
        "pearson_r": r, "rmse": rmse,
        "conf_rmse": conf_rmse, "score_r": score_r,
        "n_crystal": int(crystal_mask.sum()),
    }


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load split info
    cache_dir = Path(args.data)
    split_file = cache_dir / "split_info.json"
    if split_file.exists():
        with open(split_file) as f:
            split_info = json.load(f)
        train_ids = split_info["train"]
        casf_ids = split_info["casf"]
    else:
        # Fallback: use all data, 90/10 split
        all_ids = list({f.stem.split("_rmsd")[0] for f in cache_dir.glob("*.pt")})
        np.random.seed(42)
        np.random.shuffle(all_ids)
        split = int(0.9 * len(all_ids))
        train_ids = all_ids[:split]
        casf_ids = all_ids[split:]

    print(f"Train proteins: {len(train_ids)}, Val/test (CASF): {len(casf_ids)}")

    # Split train into train/val (90/10)
    np.random.seed(42)
    np.random.shuffle(train_ids)
    val_split = int(0.9 * len(train_ids))
    val_ids = train_ids[val_split:]
    train_ids = train_ids[:val_split]

    train_dataset = PDBBindV4Dataset(args.data, train_ids)
    val_dataset = PDBBindV4Dataset(args.data, val_ids)
    casf_dataset = PDBBindV4Dataset(args.data, casf_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    casf_loader = DataLoader(casf_dataset, batch_size=args.batch * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = PairwiseGeometricNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    ckpt_dir = Path(args.output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_r = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        dt = time.time() - t0

        print(f"Ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_R={val_metrics['pearson_r']:.3f}  val_RMSE={val_metrics['rmse']:.2f}  "
              f"conf_RMSE={val_metrics['conf_rmse']:.3f}  score_R={val_metrics['score_r']:.3f}  "
              f"lr={lr:.1e}  {dt:.0f}s")

        # Save checkpoint
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "architecture": "PGNv4",
            "hidden_dim": HIDDEN_DIM,
            "pair_dim": PAIR_DIM,
        }
        torch.save(state, ckpt_dir / "latest.pt")

        if val_metrics["pearson_r"] > best_r:
            best_r = val_metrics["pearson_r"]
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  *** New best: R={best_r:.4f}")

    # Final CASF-2016 evaluation
    print("\n=== CASF-2016 Benchmark ===")
    casf_metrics = evaluate(model, casf_loader, device)
    print(f"  Pearson R: {casf_metrics['pearson_r']:.4f}")
    print(f"  RMSE:      {casf_metrics['rmse']:.3f}")
    print(f"  Score R:   {casf_metrics['score_r']:.4f}")
    print(f"  N crystal: {casf_metrics['n_crystal']}")


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PairwiseGeometricNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    cache_dir = Path(args.data)
    split_file = cache_dir / "split_info.json"
    if split_file.exists():
        with open(split_file) as f:
            casf_ids = json.load(f)["casf"]
    else:
        casf_ids = None

    dataset = PDBBindV4Dataset(args.data, casf_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    metrics = evaluate(model, loader, device)
    print("=== CASF-2016 Benchmark Results ===")
    print(f"  Pearson R (scoring):  {metrics['pearson_r']:.4f}")
    print(f"  RMSE (pKd):           {metrics['rmse']:.3f}")
    print(f"  Confidence RMSE:      {metrics['conf_rmse']:.4f}")
    print(f"  Score correlation:    {metrics['score_r']:.4f}")
    print(f"  Crystal poses tested: {metrics['n_crystal']}")


# =============================================================================
# Export weights for Metal
# =============================================================================

EXPORT_WEIGHT_ORDER = [
    # Protein encoder (0-3)
    "prot_encoder.0.weight",   # [128, 20]
    "prot_encoder.0.bias",     # [128]
    "prot_encoder.2.weight",   # [128, 128]
    "prot_encoder.2.bias",     # [128]
    # Message passing layer 0 (4-11)
    "msg_layers.0.msg_mlp.0.weight",    # [128, 128]
    "msg_layers.0.msg_mlp.0.bias",      # [128]
    "msg_layers.0.attn_mlp.0.weight",   # [32, 16]
    "msg_layers.0.attn_mlp.0.bias",     # [32]
    "msg_layers.0.attn_mlp.2.weight",   # [1, 32]
    "msg_layers.0.attn_mlp.2.bias",     # [1]
    "msg_layers.0.norm.weight",          # [128]
    "msg_layers.0.norm.bias",            # [128]
    # Message passing layer 1 (12-19)
    "msg_layers.1.msg_mlp.0.weight",
    "msg_layers.1.msg_mlp.0.bias",
    "msg_layers.1.attn_mlp.0.weight",
    "msg_layers.1.attn_mlp.0.bias",
    "msg_layers.1.attn_mlp.2.weight",
    "msg_layers.1.attn_mlp.2.bias",
    "msg_layers.1.norm.weight",
    "msg_layers.1.norm.bias",
    # Message passing layer 2 (20-27)
    "msg_layers.2.msg_mlp.0.weight",
    "msg_layers.2.msg_mlp.0.bias",
    "msg_layers.2.attn_mlp.0.weight",
    "msg_layers.2.attn_mlp.0.bias",
    "msg_layers.2.attn_mlp.2.weight",
    "msg_layers.2.attn_mlp.2.bias",
    "msg_layers.2.norm.weight",
    "msg_layers.2.norm.bias",
    # Ligand encoder (28-31)
    "lig_encoder.0.weight",    # [128, 20]
    "lig_encoder.0.bias",      # [128]
    "lig_encoder.2.weight",    # [128, 128]
    "lig_encoder.2.bias",      # [128]
    # Pair projections (32-35)
    "prot_pair_proj.weight",   # [64, 128]
    "prot_pair_proj.bias",     # [64]
    "lig_pair_proj.weight",    # [64, 128]
    "lig_pair_proj.bias",      # [64]
    # RBF projection (36-37)
    "rbf_proj.0.weight",      # [64, 24]
    "rbf_proj.0.bias",        # [64]
    # Pair energy head (38-39)
    "pair_energy.1.weight",    # [1, 64]
    "pair_energy.1.bias",      # [1]
    # Context gate (40-41)
    "context_gate.weight",     # [1, 64]
    "context_gate.bias",       # [1]
    # Context projection (42-43)
    "context_proj.weight",     # [128, 64]
    "context_proj.bias",       # [128]
    # Context LayerNorm (44-45)
    "context_norm.weight",     # [128]
    "context_norm.bias",       # [128]
    # Affinity head (46-49)
    "affinity_head.0.weight",  # [64, 128]
    "affinity_head.0.bias",    # [64]
    "affinity_head.2.weight",  # [1, 64]
    "affinity_head.2.bias",    # [1]
    # Confidence head (50-53)
    "confidence_head.0.weight", # [64, 128]
    "confidence_head.0.bias",   # [64]
    "confidence_head.2.weight", # [1, 64]
    "confidence_head.2.bias",   # [1]
    # Learned scalars (54-55)
    "pair_scale",              # [1]
    "pair_bias",               # [1]
]


def run_export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    print(f"=== DruseAF v4 Weight Export ===")
    print(f"  Checkpoint: {args.checkpoint}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val Pearson R: {m.get('pearson_r', '?')}")

    # Collect tensors
    missing = [k for k in EXPORT_WEIGHT_ORDER if k not in state]
    if missing:
        print(f"\nERROR: {len(missing)} weight tensors missing:")
        for k in missing:
            print(f"  - {k}")
        sys.exit(1)

    tensors = []
    for key in EXPORT_WEIGHT_ORDER:
        t = state[key]
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
        f.write(struct.pack("<I", 2))  # version 2
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", total_floats))
        for off, cnt in offsets:
            f.write(struct.pack("<II", off, cnt))
        for t in tensors:
            f.write(t.astype(np.float32).tobytes())

    print(f"\n  Exported {num_tensors} tensors, {total_floats:,} floats ({total_floats*4/1024:.1f} KB)")
    print(f"  DRAF v2 format → {out_path}")
    print(f"\n  Copy to Models/druse-models/DruseAFv4.weights")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DruseAF v4 — Pairwise Geometric Network")
    sub = parser.add_subparsers(dest="command")

    # preprocess
    pp = sub.add_parser("preprocess", help="Preprocess PDBbind into .pt cache")
    pp.add_argument("--refined", type=str, required=True, help="PDBbind v2020 refined set directory")
    pp.add_argument("--general", type=str, default=None, help="PDBbind v2020 general-minus-refined directory")
    pp.add_argument("--casf", type=str, default=None, help="CASF-2016 coreset directory")
    pp.add_argument("--out", type=str, required=True, help="Output cache directory")

    # train
    tr = sub.add_parser("train", help="Train PGN model")
    tr.add_argument("--data", type=str, required=True, help="Preprocessed cache directory")
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--batch", type=int, default=32)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--workers", type=int, default=8)
    tr.add_argument("--output", type=str, default="checkpoints_v4")

    # benchmark
    bm = sub.add_parser("benchmark", help="Run CASF-2016 benchmark")
    bm.add_argument("--data", type=str, required=True)
    bm.add_argument("--checkpoint", type=str, required=True)

    # export
    ex = sub.add_parser("export", help="Export weights for Metal")
    ex.add_argument("--checkpoint", type=str, required=True)
    ex.add_argument("--out", type=str, default="DruseAFv4.weights")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args)
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

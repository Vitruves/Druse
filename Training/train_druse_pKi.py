#!/usr/bin/env python3
"""
Train DruseScore-pKi: primary scoring function for Druse molecular docking.

Unlike the re-ranking model (train_druse_score.py), this model IS the scoring
function. It predicts a docking_score = pKd * pose_confidence that serves as
the authoritative ranking for docked poses, replacing Vina-style empirical scores.

Key differences from train_druse_score.py:
  - Structured RMSD perturbations (not random decoys) create a smooth
    quality spectrum from crystal (RMSD~0) to garbage (RMSD~15)
  - Continuous pose_confidence via Gaussian decay: exp(-RMSD^2 / 2*sigma^2)
  - Combined docking_score output trained end-to-end
  - Single train/val split (no k-fold) for faster iteration
  - RMSE on docking_score is the primary metric (not just pKd correlation)

Architecture (same EGNN + cross-attention):
  - Protein encoder: 4-layer E(n)-equivariant GNN
  - Ligand encoder: 4-layer EGNN
  - Geometric cross-attention: ligand->protein with RBF distance encoding
  - Heads: pKd regression + continuous pose confidence + interaction prediction
  - Primary output: docking_score = pKd_pred * pose_confidence_pred

Training data:
  - PDBbind v2020 refined (5,316 complexes)
  - Each complex generates ~8 perturbations at controlled RMSD levels
  - Total: ~42,000 training examples from precomputed cache

Usage:
  python precompute_druse_features_pKi.py --data_dir data/ --workers 8
  python train_druse_pKi.py --data_dir data/ --epochs 80
  python train_druse_pKi.py --eval_only --checkpoint checkpoints_pki/druse_pki_best.pt
"""

import os
import argparse
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time

# ============================================================================
# Feature Extraction (shared with train_druse_score.py)
# ============================================================================

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8}
NUM_ATOM_FEATURES = 18  # 10 (element) + 1 (aromatic) + 1 (charge) + 2 (hb) + 3 (hybrid) + 1 (is_ligand)
RBF_CENTERS = torch.linspace(0, 10, 50)
RBF_GAMMA = 10.0

# Aromatic atoms in standard amino acid residues
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

HBD_RESIDUE_ATOMS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"}, "TRP": {"NE1"},
    "CYS": {"SG"},
}

HBA_RESIDUE_ATOMS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "HIS": {"ND1", "NE2"},
    "MET": {"SD"}, "CYS": {"SG"},
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

# Gaussian decay for pose confidence: sigma controls how fast confidence drops with RMSD
# sigma=2.0 means: RMSD=2A -> conf=0.61, RMSD=3A -> conf=0.32, RMSD=5A -> conf=0.04
POSE_CONFIDENCE_SIGMA = 2.0


def rbf_encode(distances: torch.Tensor) -> torch.Tensor:
    """Gaussian RBF encoding of distances. [N] -> [N, 50]"""
    return torch.exp(-RBF_GAMMA * (distances.unsqueeze(-1) - RBF_CENTERS.to(distances.device)) ** 2)


def rmsd_to_confidence(rmsd: float, sigma: float = POSE_CONFIDENCE_SIGMA) -> float:
    """Convert RMSD to continuous pose confidence via Gaussian decay."""
    return math.exp(-(rmsd ** 2) / (2.0 * sigma ** 2))


def pdb_atom_features(element: str, residue_name: str, atom_name: str) -> np.ndarray:
    """Encode a PDB protein atom as 18-dimensional feature vector."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    idx = ATOM_TYPES.get(element, 9)
    feat[idx] = 1.0
    aromatic_set = AROMATIC_RESIDUE_ATOMS.get(residue_name, set())
    is_aromatic = atom_name in aromatic_set
    feat[10] = float(is_aromatic)
    feat[11] = RESIDUE_ATOM_CHARGES.get((residue_name, atom_name), 0.0)
    is_backbone_hbd = (atom_name == "N" and residue_name != "PRO")
    is_sidechain_hbd = atom_name in HBD_RESIDUE_ATOMS.get(residue_name, set())
    feat[12] = float(is_backbone_hbd or is_sidechain_hbd)
    is_backbone_hba = (atom_name == "O")
    is_sidechain_hba = atom_name in HBA_RESIDUE_ATOMS.get(residue_name, set())
    feat[13] = float(is_backbone_hba or is_sidechain_hba)
    is_sp2 = (is_aromatic
              or atom_name in SP2_RESIDUE_ATOMS.get(residue_name, set())
              or (atom_name == "C" and element == "C")
              or (atom_name == "O" and element == "O"))
    feat[14] = 0.0
    feat[15] = float(is_sp2)
    feat[16] = float(not is_sp2)
    feat[17] = 0.0
    return feat


def mol2_atom_features(element: str, sybyl_type: str, charge: float) -> np.ndarray:
    """Encode a MOL2 ligand atom as 18-dimensional feature vector."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    idx = ATOM_TYPES.get(element, 9)
    feat[idx] = 1.0
    sybyl_parts = sybyl_type.split(".")
    subtype = sybyl_parts[1] if len(sybyl_parts) > 1 else "3"
    is_aromatic = (subtype == "ar")
    feat[10] = float(is_aromatic)
    feat[11] = charge
    is_hbd = False
    if element == "N" and subtype in ("3", "4", "am", "pl3"):
        is_hbd = True
    elif element == "O" and subtype == "3":
        is_hbd = True
    elif element == "S" and subtype == "3":
        is_hbd = True
    feat[12] = float(is_hbd)
    feat[13] = float(element in ("N", "O", "S", "F"))
    sp, sp2, sp3 = SYBYL_HYBRIDIZATION.get(subtype, (0, 0, 1))
    feat[14] = float(sp)
    feat[15] = float(sp2)
    feat[16] = float(sp3)
    feat[17] = 1.0
    return feat


# ============================================================================
# Perturbation Generation
# ============================================================================

# RMSD levels for structured perturbations during training
# Covers the full spectrum: near-native → mediocre → bad → garbage
PERTURBATION_RMSD_TARGETS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]


def generate_perturbed_pose(lig_pos: np.ndarray, target_rmsd: float) -> tuple:
    """Generate a perturbed ligand pose targeting a specific RMSD from the original.

    Uses rotation + translation calibrated to achieve approximately the target RMSD.
    Returns (perturbed_pos, actual_rmsd).
    """
    if target_rmsd < 0.01:
        return lig_pos.copy(), 0.0

    center = lig_pos.mean(axis=0)
    centered = lig_pos - center
    lig_radius = np.sqrt((centered ** 2).sum(axis=1).mean())

    # Rotation angle calibrated to ligand size
    # For a sphere of radius r, rotation by angle theta gives RMSD ~ r * theta * sqrt(2/3)
    # We use half the target RMSD for rotation, half for translation
    rot_rmsd_target = target_rmsd * 0.6
    trans_rmsd_target = target_rmsd * 0.4

    # Rotation: angle ~ rot_rmsd / (lig_radius * sqrt(2/3))
    if lig_radius > 0.1:
        rot_angle = rot_rmsd_target / (lig_radius * 0.816)  # sqrt(2/3) ~ 0.816
        rot_angle = min(rot_angle, math.pi)  # cap at 180 degrees
    else:
        rot_angle = 0.0

    # Random rotation axis + calibrated angle
    axis = np.random.randn(3).astype(np.float32)
    axis /= (np.linalg.norm(axis) + 1e-8)
    rotvec = axis * rot_angle
    R = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)
    rotated = centered @ R.T

    # Random translation direction + calibrated magnitude
    trans_dir = np.random.randn(3).astype(np.float32)
    trans_dir /= (np.linalg.norm(trans_dir) + 1e-8)
    translated = rotated + center + trans_dir * trans_rmsd_target

    # Compute actual RMSD
    actual_rmsd = np.sqrt(np.mean(np.sum((translated - lig_pos) ** 2, axis=1)))
    return translated, float(actual_rmsd)


def compute_interaction_labels(lig_pos, prot_pos, lig_feat, prot_feat):
    """Compute [L, P, 5] binary interaction labels from geometry and features.

    Interaction types:
      0: H-bond (donor-acceptor pair, dist < 3.5 A)
      1: Hydrophobic (C-C contact, dist < 4.5 A)
      2: Ionic (opposite charges, dist < 4.0 A)
      3: Pi-stacking (both aromatic, dist < 5.5 A)
      4: Halogen bond (halogen to N/O, dist < 3.5 A)
    """
    cross_dist = cdist(lig_pos, prot_pos)
    L, P = cross_dist.shape
    labels = np.zeros((L, P, 5), dtype=np.float32)

    lig_hbd = lig_feat[:, 12]
    lig_hba = lig_feat[:, 13]
    lig_arom = lig_feat[:, 10]
    lig_charge = lig_feat[:, 11]
    lig_carbon = lig_feat[:, 1]
    lig_halogen = lig_feat[:, 4] + lig_feat[:, 7] + lig_feat[:, 8]

    prot_hbd = prot_feat[:, 12]
    prot_hba = prot_feat[:, 13]
    prot_arom = prot_feat[:, 10]
    prot_charge = prot_feat[:, 11]
    prot_carbon = prot_feat[:, 1]
    prot_no = prot_feat[:, 2] + prot_feat[:, 3]

    hb_mask = cross_dist < 3.5
    hb_da = (np.outer(lig_hbd, prot_hba) > 0) | (np.outer(lig_hba, prot_hbd) > 0)
    labels[:, :, 0] = (hb_mask & hb_da).astype(np.float32)
    labels[:, :, 1] = ((cross_dist < 4.5) & (np.outer(lig_carbon, prot_carbon) > 0)).astype(np.float32)
    charge_product = np.outer(lig_charge, prot_charge)
    labels[:, :, 2] = ((cross_dist < 4.0) & (charge_product < -0.09)).astype(np.float32)
    labels[:, :, 3] = ((cross_dist < 5.5) & (np.outer(lig_arom, prot_arom) > 0)).astype(np.float32)
    labels[:, :, 4] = ((cross_dist < 3.5) & (np.outer(lig_halogen, prot_no) > 0)).astype(np.float32)

    return labels


# ============================================================================
# GPU-accelerated helpers
# ============================================================================

@torch.no_grad()
def compute_interaction_labels_gpu(lig_pos, prot_pos, lig_feat, prot_feat):
    """Compute [L, P, 5] interaction labels on GPU."""
    cross_dist = torch.cdist(lig_pos.unsqueeze(0), prot_pos.unsqueeze(0)).squeeze(0)

    lig_hbd = lig_feat[:, 12]
    lig_hba = lig_feat[:, 13]
    lig_arom = lig_feat[:, 10]
    lig_charge = lig_feat[:, 11]
    lig_carbon = lig_feat[:, 1]
    lig_halogen = lig_feat[:, 4] + lig_feat[:, 7] + lig_feat[:, 8]

    prot_hbd = prot_feat[:, 12]
    prot_hba = prot_feat[:, 13]
    prot_arom = prot_feat[:, 10]
    prot_charge = prot_feat[:, 11]
    prot_carbon = prot_feat[:, 1]
    prot_no = prot_feat[:, 2] + prot_feat[:, 3]

    L, P = cross_dist.shape
    labels = torch.zeros(L, P, 5, device=lig_pos.device)

    hb_da = (torch.outer(lig_hbd, prot_hba) > 0) | (torch.outer(lig_hba, prot_hbd) > 0)
    labels[:, :, 0] = ((cross_dist < 3.5) & hb_da).float()
    labels[:, :, 1] = ((cross_dist < 4.5) & (torch.outer(lig_carbon, prot_carbon) > 0)).float()
    labels[:, :, 2] = ((cross_dist < 4.0) & (torch.outer(lig_charge, prot_charge) < -0.09)).float()
    labels[:, :, 3] = ((cross_dist < 5.5) & (torch.outer(lig_arom, prot_arom) > 0)).float()
    labels[:, :, 4] = ((cross_dist < 3.5) & (torch.outer(lig_halogen, prot_no) > 0)).float()

    return labels


# ============================================================================
# Batching constants
# ============================================================================

PAD_PROT = 350   # pad/truncate protein to this size (covers >90th percentile)
PAD_LIG = 55     # pad/truncate ligand to this size (covers >95th percentile)
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # effective batch = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS = 8
EVAL_BATCH_SIZE = 8    # no gradients, can fit more


def pad_sample(data) -> dict:
    """Pad a single PyG Data object to fixed sizes for batched training.
    Returns a dict of tensors (stackable by default collate)."""
    P = min(data.prot_pos.shape[0], PAD_PROT)
    L = min(data.lig_pos.shape[0], PAD_LIG)

    # Truncate protein to nearest atoms around ligand if needed
    if data.prot_pos.shape[0] > PAD_PROT:
        lig_center = data.lig_pos.mean(dim=0)
        dists = (data.prot_pos - lig_center).norm(dim=1)
        _, keep = dists.topk(PAD_PROT, largest=False)
        keep = keep.sort().values
        prot_pos_raw = data.prot_pos[keep]
        prot_x_raw = data.prot_x[keep]
    else:
        prot_pos_raw = data.prot_pos[:P]
        prot_x_raw = data.prot_x[:P]

    # Pad protein
    prot_pos = torch.zeros(PAD_PROT, 3)
    prot_pos[:P] = prot_pos_raw
    prot_x = torch.zeros(PAD_PROT, 18)
    prot_x[:P] = prot_x_raw
    prot_mask = torch.zeros(PAD_PROT)
    prot_mask[:P] = 1.0

    # Pad ligand
    lig_pos = torch.zeros(PAD_LIG, 3)
    lig_pos[:L] = data.lig_pos[:L]
    lig_x = torch.zeros(PAD_LIG, 18)
    lig_x[:L] = data.lig_x[:L]
    lig_mask = torch.zeros(PAD_LIG)
    lig_mask[:L] = 1.0

    return {
        'prot_pos': prot_pos, 'prot_x': prot_x, 'prot_mask': prot_mask,
        'lig_pos': lig_pos, 'lig_x': lig_x, 'lig_mask': lig_mask,
        'y': data.y.squeeze(), 'rmsd': data.rmsd.squeeze(),
        'pose_confidence': data.pose_confidence.squeeze(),
        'docking_score': data.docking_score.squeeze(),
        'pdb_id': data.pdb_id,
    }


def batched_collate(batch):
    """Collate padded dicts into batched tensors. Handles string fields."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


@torch.no_grad()
def compute_interaction_labels_batched(lig_pos, prot_pos, lig_feat, prot_feat, lig_mask, prot_mask):
    """Compute [B, L, P, 5] interaction labels on GPU (batched)."""
    cross_dist = torch.cdist(lig_pos, prot_pos)  # [B, L, P]

    lig_hbd = lig_feat[:, :, 12:13]
    lig_hba = lig_feat[:, :, 13:14]
    lig_arom = lig_feat[:, :, 10:11]
    lig_charge = lig_feat[:, :, 11:12]
    lig_carbon = lig_feat[:, :, 1:2]
    lig_halogen = lig_feat[:, :, 4:5] + lig_feat[:, :, 7:8] + lig_feat[:, :, 8:9]

    prot_hbd = prot_feat[:, :, 12:13]
    prot_hba = prot_feat[:, :, 13:14]
    prot_arom = prot_feat[:, :, 10:11]
    prot_charge = prot_feat[:, :, 11:12]
    prot_carbon = prot_feat[:, :, 1:2]
    prot_no = prot_feat[:, :, 2:3] + prot_feat[:, :, 3:4]

    B, L, P = cross_dist.shape
    labels = torch.zeros(B, L, P, 5, device=lig_pos.device)

    hb_da = (torch.bmm(lig_hbd, prot_hba.transpose(1, 2)) > 0) | \
            (torch.bmm(lig_hba, prot_hbd.transpose(1, 2)) > 0)
    labels[:, :, :, 0] = ((cross_dist < 3.5) & hb_da).float()
    labels[:, :, :, 1] = ((cross_dist < 4.5) & (torch.bmm(lig_carbon, prot_carbon.transpose(1, 2)) > 0)).float()
    labels[:, :, :, 2] = ((cross_dist < 4.0) & (torch.bmm(lig_charge, prot_charge.transpose(1, 2)) < -0.09)).float()
    labels[:, :, :, 3] = ((cross_dist < 5.5) & (torch.bmm(lig_arom, prot_arom.transpose(1, 2)) > 0)).float()
    labels[:, :, :, 4] = ((cross_dist < 3.5) & (torch.bmm(lig_halogen, prot_no.transpose(1, 2)) > 0)).float()

    pair_mask = lig_mask.unsqueeze(2) * prot_mask.unsqueeze(1)  # [B, L, P]
    labels = labels * pair_mask.unsqueeze(-1)
    return labels


# ============================================================================
# Dataset
# ============================================================================

class CachedPKiDataset(Dataset):
    """Load precomputed .pt files, pad to fixed sizes for batched training."""

    def __init__(self, data_dir: str):
        cache_dir = Path(data_dir) / "druse_pki_cache"
        files = sorted(cache_dir.glob("*.pt"))
        self.samples = []
        for f in tqdm(files, desc="Loading pKi cache to RAM"):
            self.samples.append(torch.load(f, weights_only=False))
        print(f"CachedPKiDataset: {len(self.samples)} samples loaded to RAM")

        rmsds = [s.rmsd.item() for s in self.samples]
        scores = [s.docking_score.item() for s in self.samples]
        pkds = [s.y.item() for s in self.samples]
        n_crystal = sum(1 for r in rmsds if r < 0.01)
        print(f"  Crystal poses: {n_crystal}, Perturbed: {len(self.samples) - n_crystal}")
        print(f"  RMSD range: {min(rmsds):.1f} - {max(rmsds):.1f}")
        print(f"  pKd range: {min(pkds):.1f} - {max(pkds):.1f}")
        print(f"  Docking score range: {min(scores):.2f} - {max(scores):.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return pad_sample(self.samples[idx])


class OnTheFlyPKiDataset(Dataset):
    """Fallback: parse PDB/MOL2 and generate perturbations on-the-fly (slow)."""

    def __init__(self, data_dir: str):
        import pandas as pd
        self.data_dir = Path(data_dir)
        labels_path = self.data_dir / "pdbbind_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}\nRun: python download_data.py --all")

        df = pd.read_csv(labels_path)
        df = df[df["split"] == "refined"]

        self.entries = []
        for _, row in df.iterrows():
            pdb_id = row["pdb_id"]
            pkd = row["pKd"]
            pocket = self.data_dir / "refined-set" / pdb_id / f"{pdb_id}_pocket.pdb"
            ligand = self.data_dir / "refined-set" / pdb_id / f"{pdb_id}_ligand.mol2"
            if pocket.exists() and ligand.exists():
                # Each complex generates multiple perturbation levels
                for rmsd_target in PERTURBATION_RMSD_TARGETS:
                    self.entries.append((pdb_id, pocket, ligand, pkd, rmsd_target))

        print(f"OnTheFlyPKiDataset: {len(self.entries)} samples "
              f"({len(self.entries) // len(PERTURBATION_RMSD_TARGETS)} complexes × "
              f"{len(PERTURBATION_RMSD_TARGETS)} perturbations)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pdb_id, pocket_path, ligand_path, pkd, rmsd_target = self.entries[idx]

        prot_pos, prot_feat = self._parse_pdb(pocket_path)
        lig_pos, lig_feat = self._parse_mol2(ligand_path)
        if prot_pos is None or lig_pos is None:
            return None

        # Generate perturbation
        pert_pos, actual_rmsd = generate_perturbed_pose(lig_pos, rmsd_target)
        confidence = rmsd_to_confidence(actual_rmsd)
        docking_score = pkd * confidence

        data = Data(
            prot_pos=torch.tensor(prot_pos, dtype=torch.float32),
            prot_x=torch.tensor(prot_feat, dtype=torch.float32),
            lig_pos=torch.tensor(pert_pos, dtype=torch.float32),
            lig_x=torch.tensor(lig_feat, dtype=torch.float32),
            y=torch.tensor([pkd], dtype=torch.float32),
            rmsd=torch.tensor([actual_rmsd], dtype=torch.float32),
            pose_confidence=torch.tensor([confidence], dtype=torch.float32),
            docking_score=torch.tensor([docking_score], dtype=torch.float32),
            pdb_id=pdb_id,
        )
        return pad_sample(data)

    def _parse_pdb(self, path: Path):
        positions, features = [], []
        try:
            with open(path) as f:
                for line in f:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:16].strip()[0]
                    if element == "H":
                        continue
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    residue_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    positions.append([x, y, z])
                    features.append(pdb_atom_features(element, residue_name, atom_name))
        except Exception:
            return None, None
        if not positions:
            return None, None
        return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)

    def _parse_mol2(self, path: Path):
        positions, features = [], []
        in_atoms = False
        try:
            with open(path) as f:
                for line in f:
                    if "@<TRIPOS>ATOM" in line:
                        in_atoms = True
                        continue
                    if "@<TRIPOS>" in line and in_atoms:
                        break
                    if not in_atoms:
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    sybyl_type = parts[5]
                    element = sybyl_type.split(".")[0]
                    if element == "H":
                        continue
                    charge = float(parts[8]) if len(parts) > 8 else 0.0
                    positions.append([x, y, z])
                    features.append(mol2_atom_features(element, sybyl_type, charge))
        except Exception:
            return None, None
        if not positions:
            return None, None
        return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)


# ============================================================================
# Model: EGNN Layers
# ============================================================================

class EGNNLayer(nn.Module):
    """E(n)-equivariant graph neural network layer."""

    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pos, edge_index, edge_attr=None):
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        edge_input = [h[row], h[col], dist]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)

        msg = self.edge_mlp(edge_input)

        coord_weight = self.coord_mlp(msg).clamp(-1.0, 1.0)
        coord_delta = (diff / dist) * coord_weight
        pos_out = pos.clone()
        pos_out.index_add_(0, row, coord_delta)

        agg = torch.zeros_like(h)
        agg.index_add_(0, row, msg)
        h_out = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        return h_out, pos_out


def build_radius_graph_cpu(pos: torch.Tensor, cutoff: float, max_neighbors: int = 32):
    """Build radius graph on CPU (fallback if torch-cluster unavailable)."""
    pos_cpu = pos.detach().cpu()
    dist = torch.cdist(pos_cpu, pos_cpu)
    mask = (dist < cutoff) & (dist > 0)
    rows, cols = [], []
    for i in range(pos_cpu.size(0)):
        neighbors = mask[i].nonzero(as_tuple=True)[0]
        if len(neighbors) > max_neighbors:
            dists_i = dist[i, neighbors]
            _, topk = dists_i.topk(max_neighbors, largest=False)
            neighbors = neighbors[topk]
        for j in neighbors:
            rows.append(i)
            cols.append(j.item())
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=pos.device)
    return edge_index


class EGNNEncoder(nn.Module):
    """4-layer EGNN encoder for protein or ligand."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, cutoff: float = 6.0):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.cutoff = cutoff

    def forward(self, x, pos, batch=None):
        h = self.input_proj(x)
        try:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        except Exception:
            edge_index = build_radius_graph_cpu(pos, self.cutoff)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return h, pos


# ============================================================================
# Model: Cross-Attention
# ============================================================================

class GeometricCrossAttention(nn.Module):
    """Geometric cross-attention: ligand attends to protein, distance-weighted."""

    def __init__(self, hidden_dim: int, rbf_dim: int = 50, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rbf_proj = nn.Linear(rbf_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value, cross_rbf, key_mask=None):
        """
        query: [B, L, D] ligand features
        key/value: [B, P, D] protein features
        cross_rbf: [B, L, P, 50]
        key_mask: [B, P] 1=real, 0=padding (optional)
        """
        B, L, D = query.shape
        P = key.shape[1]
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(query).reshape(B, L, H, d)
        K = self.k_proj(key).reshape(B, P, H, d)
        V = self.v_proj(value).reshape(B, P, H, d)

        attn = torch.einsum("blhd,bphd->blph", Q, K) / math.sqrt(d)
        dist_bias = self.rbf_proj(cross_rbf)  # [B, L, P, H]
        attn = attn + dist_bias

        if key_mask is not None:
            attn = attn.masked_fill(key_mask[:, None, :, None] == 0, -1e9)

        attn = F.softmax(attn, dim=2)

        out = torch.einsum("blph,bphd->blhd", attn, V)
        out = out.reshape(B, L, D)
        out = self.out_proj(out)
        return self.norm(query + out), attn.mean(dim=-1)  # [B, L, D], [B, L, P]


# ============================================================================
# Model: DruseScore-pKi
# ============================================================================

class DruseScorePKi(nn.Module):
    """SE(3)-equivariant scoring function — primary docking scorer.

    Multi-task outputs:
      1. pKd regression (binding affinity)
      2. Pose confidence (continuous, Gaussian-decayed from RMSD)
      3. Docking score = pKd * confidence (trained end-to-end)
      4. Interaction type prediction (per atom-pair)

    The docking_score is the PRIMARY output used for ranking docked poses.
    """

    def __init__(self, atom_dim: int = 18, hidden_dim: int = 128, num_egnn_layers: int = 4):
        super().__init__()
        self.prot_encoder = EGNNEncoder(atom_dim, hidden_dim, num_egnn_layers)
        self.lig_encoder = EGNNEncoder(atom_dim, hidden_dim, num_egnn_layers)
        self.cross_attn = GeometricCrossAttention(hidden_dim)

        # Affinity head: predicts pKd (should be accurate for crystal-like poses)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Pose confidence head: predicts how close the pose is to native (0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Interaction prediction head (5 types)
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 50, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, batch):
        """Batched forward pass.
        batch: dict with keys prot_pos, prot_x, prot_mask, lig_pos, lig_x, lig_mask
               all tensors shaped [B, N, ...].
        """
        device = batch['prot_x'].device
        B = batch['prot_x'].shape[0]
        P, L = PAD_PROT, PAD_LIG
        prot_mask = batch['prot_mask']  # [B, P]
        lig_mask = batch['lig_mask']    # [B, L]

        # === EGNN encoding (flatten with batch indices for radius_graph) ===

        # Protein: [B, P, D] -> [B*P, D], padding positions set far away
        prot_x_flat = batch['prot_x'].reshape(B * P, 18)
        prot_pos_flat = batch['prot_pos'].reshape(B * P, 3).clone()
        prot_pos_flat[~prot_mask.reshape(B * P).bool()] = 1e6
        prot_batch = torch.arange(B, device=device).repeat_interleave(P)

        prot_h, prot_pos_enc = self.prot_encoder(prot_x_flat, prot_pos_flat, batch=prot_batch)
        prot_h = prot_h.reshape(B, P, -1)
        prot_pos_enc = prot_pos_enc.reshape(B, P, 3)

        # Ligand: [B, L, D] -> [B*L, D]
        lig_x_flat = batch['lig_x'].reshape(B * L, 18)
        lig_pos_flat = batch['lig_pos'].reshape(B * L, 3).clone()
        lig_pos_flat[~lig_mask.reshape(B * L).bool()] = 1e6
        lig_batch = torch.arange(B, device=device).repeat_interleave(L)

        lig_h, lig_pos_enc = self.lig_encoder(lig_x_flat, lig_pos_flat, batch=lig_batch)
        lig_h = lig_h.reshape(B, L, -1)
        lig_pos_enc = lig_pos_enc.reshape(B, L, 3)

        # === Batched cross-distances and RBF ===
        cross_dist = torch.cdist(lig_pos_enc, prot_pos_enc)  # [B, L, P]
        cross_rbf = torch.exp(
            -RBF_GAMMA * (cross_dist.unsqueeze(-1) - RBF_CENTERS.to(device)) ** 2
        )  # [B, L, P, 50]

        # === Batched cross-attention ===
        lig_attended, attn_weights = self.cross_attn(
            lig_h, prot_h, prot_h, cross_rbf, prot_mask)

        # === Masked mean pooling ===
        lig_masked = lig_attended * lig_mask.unsqueeze(-1)  # [B, L, D]
        n_lig = lig_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        complex_repr = lig_masked.sum(dim=1) / n_lig  # [B, D]

        # === Head predictions ===
        pkd_pred = self.affinity_head(complex_repr).squeeze(-1)  # [B]
        confidence_pred = torch.sigmoid(
            self.confidence_head(complex_repr)).squeeze(-1)  # [B]
        docking_score = pkd_pred * confidence_pred  # [B]

        # === Batched interaction map ===
        D = lig_h.shape[-1]
        lig_exp = lig_attended.unsqueeze(2).expand(-1, -1, P, -1)   # [B, L, P, D]
        prot_exp = prot_h.unsqueeze(1).expand(-1, L, -1, -1)       # [B, L, P, D]
        pair_input = torch.cat([lig_exp, prot_exp, cross_rbf], dim=-1)
        interaction_pred = torch.sigmoid(self.interaction_head(pair_input))  # [B, L, P, 5]

        return {
            "pKd": pkd_pred,
            "pose_confidence": confidence_pred,
            "docking_score": docking_score,
            "interaction_map": interaction_pred,
            "attention_weights": attn_weights,
        }


# ============================================================================
# Training
# ============================================================================


def move_batch_to_device(batch, device):
    """Move a batched dict to device."""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device, epoch_num=0):
    model.train()
    total_loss = 0
    total_loss_affinity = 0
    total_loss_confidence = 0
    total_loss_score = 0
    total_loss_interaction = 0
    n = 0
    nan_count = 0
    skip_count = 0
    max_grad_norm = 0.0
    batch_idx = -1

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"  Train ep{epoch_num+1}", leave=False)):
        if batch is None:
            skip_count += 1
            continue
        batch = move_batch_to_device(batch, device)
        B = batch['prot_x'].shape[0]

        # Batched interaction labels
        interaction_labels = compute_interaction_labels_batched(
            batch['lig_pos'], batch['prot_pos'],
            batch['lig_x'], batch['prot_x'],
            batch['lig_mask'], batch['prot_mask'])

        try:
            output = model(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            skip_count += 1
            continue

        # === Multi-task loss (all batched) ===

        # 1. Affinity loss — only on poses with meaningful confidence.
        # Decoy poses (confidence ~0, i.e. RMSD >> 2.5 A) should NOT contribute
        # to the affinity loss: penalizing the model for failing to predict pKd
        # on geometrically nonsensical poses adds noise and hurts convergence.
        # Threshold 0.2 corresponds to ~RMSD < 2.5 A given sigma=2.0 Gaussian decay.
        confidence_mask = (batch['pose_confidence'] > 0.2).float()  # [B]
        confidence_weight = batch['pose_confidence'] * confidence_mask  # [B]
        loss_affinity = (confidence_weight * F.huber_loss(
            output["pKd"], batch['y'], reduction="none", delta=2.0))
        loss_affinity = loss_affinity.sum() / confidence_weight.sum().clamp(min=1.0)

        # 2. Pose confidence loss
        loss_confidence = F.mse_loss(output["pose_confidence"], batch['pose_confidence'])

        # 3. Docking score loss (PRIMARY)
        loss_score = F.huber_loss(output["docking_score"], batch['docking_score'], delta=2.0)

        # 4. Interaction loss with pair mask (focal loss)
        int_pred = output["interaction_map"]   # [B, L, P, 5]
        int_target = interaction_labels         # [B, L, P, 5]
        pair_mask = batch['lig_mask'].unsqueeze(2) * batch['prot_mask'].unsqueeze(1)  # [B, L, P]

        int_bce = F.binary_cross_entropy(int_pred, int_target, reduction="none")
        int_p_t = int_pred * int_target + (1 - int_pred) * (1 - int_target)
        int_alpha_t = 0.9 * int_target + 0.1 * (1 - int_target)
        int_focal_weight = int_alpha_t * (1 - int_p_t) ** 2
        masked_focal = (int_focal_weight * int_bce) * pair_mask.unsqueeze(-1)
        loss_interaction = masked_focal.sum() / pair_mask.sum().clamp(min=1) / 5

        loss_total = (1.0 * loss_score +
                      0.5 * loss_affinity +
                      0.5 * loss_confidence +
                      0.2 * loss_interaction)
        loss = loss_total / GRAD_ACCUM_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        total_loss_affinity += loss_affinity.item()
        total_loss_confidence += loss_confidence.item()
        total_loss_score += loss_score.item()
        total_loss_interaction += loss_interaction.item()
        n += 1

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm > max_grad_norm:
                max_grad_norm = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 200 == 0 and batch_idx > 0:
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"\n  [WARN] NaN in model params at batch {batch_idx}!")
                return float('nan'), {}, nan_count, skip_count, max_grad_norm

    # Handle leftover gradients
    if batch_idx >= 0 and n > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if grad_norm > max_grad_norm:
            max_grad_norm = float(grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(n, 1)
    loss_breakdown = {
        "score": total_loss_score / max(n, 1),
        "affinity": total_loss_affinity / max(n, 1),
        "confidence": total_loss_confidence / max(n, 1),
        "interaction": total_loss_interaction / max(n, 1),
    }
    return avg_loss, loss_breakdown, nan_count, skip_count, max_grad_norm


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    score_preds, score_targets = [], []
    pkd_preds, pkd_targets = [], []
    conf_preds, conf_targets = [], []
    rmsds = []
    pdb_ids = []

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)

        try:
            output = model(batch)
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

        B = output["docking_score"].shape[0]
        score_preds.extend(output["docking_score"].cpu().tolist())
        score_targets.extend(batch['docking_score'].cpu().tolist())
        pkd_preds.extend(output["pKd"].cpu().tolist())
        pkd_targets.extend(batch['y'].cpu().tolist())
        conf_preds.extend(output["pose_confidence"].cpu().tolist())
        conf_targets.extend(batch['pose_confidence'].cpu().tolist())
        rmsds.extend(batch['rmsd'].cpu().tolist())
        pdb_ids.extend(batch['pdb_id'])

    score_preds = np.array(score_preds)
    score_targets = np.array(score_targets)
    pkd_preds = np.array(pkd_preds)
    pkd_targets = np.array(pkd_targets)
    conf_preds = np.array(conf_preds)
    conf_targets = np.array(conf_targets)
    rmsds_arr = np.array(rmsds)

    if len(score_preds) < 3:
        return {"score_r": 0.0, "score_rmse": 99.0, "pkd_r": 0.0,
                "conf_rmse": 1.0, "n": len(score_preds)}

    score_r = np.corrcoef(score_preds, score_targets)[0, 1] if len(np.unique(score_targets)) > 1 else 0.0
    score_rmse = np.sqrt(np.mean((score_preds - score_targets) ** 2))

    crystal_mask = rmsds_arr < 1.0
    if crystal_mask.sum() > 2:
        pkd_r = np.corrcoef(pkd_preds[crystal_mask], pkd_targets[crystal_mask])[0, 1]
    else:
        pkd_r = np.corrcoef(pkd_preds, pkd_targets)[0, 1] if len(pkd_preds) > 2 else 0.0

    conf_rmse = np.sqrt(np.mean((conf_preds - conf_targets) ** 2))

    pose_correct = 0
    pose_total = 0
    unique_ids = set(pdb_ids)
    for pid in unique_ids:
        mask = [i for i, p in enumerate(pdb_ids) if p == pid]
        if len(mask) < 2:
            continue
        scores = score_preds[mask]
        rmsds_i = rmsds_arr[mask]
        if np.argmax(scores) == np.argmin(rmsds_i):
            pose_correct += 1
        pose_total += 1

    pose_accuracy = pose_correct / max(pose_total, 1)

    return {
        "score_r": float(score_r),
        "score_rmse": float(score_rmse),
        "pkd_r": float(pkd_r),
        "conf_rmse": float(conf_rmse),
        "pose_accuracy": float(pose_accuracy),
        "n": len(score_preds),
    }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Load dataset
    cache_dir = Path(args.data_dir) / "druse_pki_cache"
    if cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 100:
        dataset = CachedPKiDataset(args.data_dir)
    else:
        print("Cache not found. Run: python precompute_druse_features_pKi.py --data_dir data/")
        print("Falling back to on-the-fly computation (slow)...\n")
        dataset = OnTheFlyPKiDataset(args.data_dir)

    if len(dataset) == 0:
        print("ERROR: No data found. Run: python download_data.py --all")
        return

    # Protein-grouped split to prevent target leakage.
    # All perturbations of the same PDB complex (and complexes from the same
    # protein target, approximated by 4-char PDB ID) stay in the same fold.
    from sklearn.model_selection import GroupShuffleSplit

    # Extract protein group for each sample
    pdb_ids = []
    for i in range(len(dataset)):
        if isinstance(dataset, CachedPKiDataset):
            pdb_ids.append(dataset.samples[i].pdb_id)
        else:  # OnTheFlyPKiDataset
            pdb_ids.append(dataset.entries[i][0])
    # Group by first 4 chars (PDB entry) so same protein stays together
    groups = [pid[:4].upper() for pid in pdb_ids]
    n_groups = len(set(groups))
    print(f"\nProtein-grouped split: {len(dataset)} samples, {n_groups} unique protein groups")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(range(len(dataset)), groups=groups))
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # Verify no group leakage
    train_groups = set(groups[i] for i in train_idx)
    val_groups = set(groups[i] for i in val_idx)
    assert train_groups.isdisjoint(val_groups), "Target leakage detected!"
    print(f"Train: {len(train_set)} ({len(train_groups)} proteins), Val: {len(val_set)} ({len(val_groups)} proteins)")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (×{GRAD_ACCUM_STEPS} accum = {TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS} effective)")

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              collate_fn=batched_collate, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            collate_fn=batched_collate, pin_memory=True)

    model = DruseScorePKi(hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
    warmup_epochs = min(10, args.epochs // 4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                          total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs),
    ], milestones=[warmup_epochs])

    best_score_r = -1
    best_model_state = None
    consecutive_nan = 0
    start_epoch = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = output_dir / "druse_pki_best.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if "best_score_r" in ckpt:
                best_score_r = ckpt["best_score_r"]
            print(f"Resumed from {resume_path} (epoch {start_epoch}, best_score_r={best_score_r:.4f})")
        else:
            print(f"WARNING: --resume specified but no checkpoint found at {resume_path}")

    log_path = output_dir / "pki_training_log.csv"
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,loss_score,loss_affinity,loss_confidence,loss_interaction,"
                    "val_score_r,val_score_rmse,val_pkd_r,val_conf_rmse,val_pose_acc,"
                    "lr,max_grad,nan_count,skip_count,time_sec\n")
    else:
        print(f"Appending to existing log: {log_path}")

    print(f"Training log: {log_path}")
    print(f"Monitor with: tail -f {log_path}\n")

    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        result = train_epoch(model, train_loader, optimizer, device, epoch_num=epoch)
        train_loss, loss_breakdown, nan_count, skip_count, max_grad = result
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - training_start

        if np.isnan(train_loss):
            consecutive_nan += 1
            print(f"  Epoch {epoch+1:3d} | NaN LOSS | grad={max_grad:.1f} nan={nan_count}")
            if consecutive_nan >= 3:
                print(f"  [ABORT] 3 consecutive NaN epochs")
                break
            continue
        consecutive_nan = 0

        # Evaluate every 2 epochs, first, and last
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            epoch_time = time.time() - epoch_start
            eta = (args.epochs - epoch - 1) * epoch_time

            status = ""
            if metrics["score_r"] > best_score_r:
                best_score_r = metrics["score_r"]
                best_model_state = model.state_dict().copy()
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "sigma": POSE_CONFIDENCE_SIGMA,
                    "perturbation_rmsds": PERTURBATION_RMSD_TARGETS,
                    "hidden_dim": args.hidden_dim,
                    "best_score_r": best_score_r,
                }, output_dir / "druse_pki_best.pt")
                status = " ** BEST"

            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} "
                  f"[S:{loss_breakdown['score']:.3f} A:{loss_breakdown['affinity']:.3f} "
                  f"C:{loss_breakdown['confidence']:.3f} I:{loss_breakdown['interaction']:.3f}]")
            print(f"    Score R: {metrics['score_r']:.3f} RMSE: {metrics['score_rmse']:.3f} | "
                  f"pKd R: {metrics['pkd_r']:.3f} | Conf RMSE: {metrics['conf_rmse']:.3f} | "
                  f"Pose Acc: {metrics['pose_accuracy']:.1%} | "
                  f"lr={current_lr:.2e} grad={max_grad:.1f} nan={nan_count} | "
                  f"{epoch_time:.0f}s (ETA {eta/60:.0f}m){status}")

            with open(log_path, "a") as f:
                f.write(f"{epoch+1},{train_loss:.6f},"
                        f"{loss_breakdown['score']:.6f},{loss_breakdown['affinity']:.6f},"
                        f"{loss_breakdown['confidence']:.6f},{loss_breakdown['interaction']:.6f},"
                        f"{metrics['score_r']:.4f},{metrics['score_rmse']:.4f},"
                        f"{metrics['pkd_r']:.4f},{metrics['conf_rmse']:.4f},"
                        f"{metrics['pose_accuracy']:.4f},"
                        f"{current_lr:.2e},{max_grad:.4f},{nan_count},{skip_count},{elapsed:.0f}\n")
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"lr={current_lr:.2e} grad={max_grad:.1f} nan={nan_count}")

    total_min = (time.time() - training_start) / 60
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE -- {total_min:.1f} minutes")
    print(f"Best Score R: {best_score_r:.3f}")
    if best_model_state:
        print(f"Model: {output_dir / 'druse_pki_best.pt'}")
    print(f"Log: {log_path}")

    return best_model_state


def main():
    parser = argparse.ArgumentParser(description="Train DruseScore-pKi (primary scoring function)")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints_pki/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint (path or 'auto' to use default location)")
    args = parser.parse_args()
    if args.resume == "auto":
        args.resume = str(Path(args.output_dir) / "druse_pki_best.pt")

    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for eval"
        device = get_device()
        checkpoint = torch.load(args.checkpoint, map_location=device)
        hidden_dim = checkpoint.get("hidden_dim", args.hidden_dim)
        model = DruseScorePKi(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        cache_dir = Path(args.data_dir) / "druse_pki_cache"
        if cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 100:
            dataset = CachedPKiDataset(args.data_dir)
        else:
            dataset = OnTheFlyPKiDataset(args.data_dir)

        loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=batched_collate)
        metrics = evaluate(model, loader, device)
        print(f"\nEvaluation on full dataset ({metrics['n']} samples):")
        print(f"  Docking Score R: {metrics['score_r']:.3f}  RMSE: {metrics['score_rmse']:.3f}")
        print(f"  pKd R (crystal):  {metrics['pkd_r']:.3f}")
        print(f"  Confidence RMSE:  {metrics['conf_rmse']:.3f}")
        print(f"  Pose Accuracy:    {metrics['pose_accuracy']:.1%}")
    else:
        train(args)


if __name__ == "__main__":
    main()

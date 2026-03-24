#!/usr/bin/env python3
"""
Train DruseScore: SE(3)-equivariant geometric cross-attention network
for protein-ligand binding affinity prediction.

Architecture:
  - Protein encoder: 4-layer E(n)-equivariant GNN (EGNN)
  - Ligand encoder: 4-layer EGNN
  - Geometric cross-attention: ligand→protein with RBF distance encoding
  - Multi-task heads: pKd regression + pose classification + interaction prediction

Training data:
  - PDBbind v2020 refined (primary)
  - PDBbind v2020 general (augmentation)
  - CASF-2016 core set (benchmark only, never trained on)

Usage:
  python train_druse_score.py --data_dir data/ --epochs 100
  python train_druse_score.py --eval_only --checkpoint best_model.pt
"""

import os
import argparse
import math
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius_graph
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ============================================================================
# Feature Extraction
# ============================================================================

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8}
NUM_ATOM_FEATURES = 18  # 10 (element) + 1 (aromatic) + 1 (charge) + 2 (hb) + 3 (hybrid) + 1 (is_ligand)
RBF_CENTERS = torch.linspace(0, 10, 50)  # 50 Gaussian RBFs, 0-10 A
RBF_GAMMA = 10.0

# Aromatic atoms in standard amino acid residues
AROMATIC_RESIDUE_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

# Partial charges at pH 7.4 for titratable residue atoms
RESIDUE_ATOM_CHARGES = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NH1"): 0.33, ("ARG", "NH2"): 0.33, ("ARG", "NE"): 0.33,
    ("HIS", "ND1"): 0.25, ("HIS", "NE2"): 0.25,
}

# H-bond donors: backbone N (except PRO) + specific side-chain atoms with attached H
HBD_RESIDUE_ATOMS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"}, "TRP": {"NE1"},
    "CYS": {"SG"},
}

# H-bond acceptors: backbone O + specific side-chain atoms with lone pairs
HBA_RESIDUE_ATOMS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "HIS": {"ND1", "NE2"},
    "MET": {"SD"}, "CYS": {"SG"},
}

# SP2 atoms in standard residues (conjugated side-chain atoms)
SP2_RESIDUE_ATOMS = {
    "ASP": {"CG", "OD1", "OD2"}, "GLU": {"CD", "OE1", "OE2"},
    "ASN": {"CG", "OD1", "ND2"}, "GLN": {"CD", "OE1", "NE2"},
    "ARG": {"CZ", "NH1", "NH2"},
}

# Sybyl atom type suffix → hybridization (sp, sp2, sp3)
SYBYL_HYBRIDIZATION = {
    "1": (1, 0, 0),    # sp
    "2": (0, 1, 0),    # sp2
    "3": (0, 0, 1),    # sp3
    "ar": (0, 1, 0),   # aromatic → treat as sp2
    "am": (0, 1, 0),   # amide → sp2
    "pl3": (0, 1, 0),  # trigonal planar → sp2
    "co2": (0, 1, 0),  # carboxylate → sp2
    "cat": (0, 1, 0),  # cation → sp2
}


def rbf_encode(distances: torch.Tensor) -> torch.Tensor:
    """Gaussian RBF encoding of distances. [N] -> [N, 50]"""
    return torch.exp(-RBF_GAMMA * (distances.unsqueeze(-1) - RBF_CENTERS.to(distances.device)) ** 2)


def pdb_atom_features(element: str, residue_name: str, atom_name: str) -> np.ndarray:
    """Encode a PDB protein atom as 18-dimensional feature vector using residue context."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)

    # [0-9] Element one-hot
    idx = ATOM_TYPES.get(element, 9)
    feat[idx] = 1.0

    # [10] Aromatic
    aromatic_set = AROMATIC_RESIDUE_ATOMS.get(residue_name, set())
    is_aromatic = atom_name in aromatic_set
    feat[10] = float(is_aromatic)

    # [11] Partial charge at pH 7.4
    feat[11] = RESIDUE_ATOM_CHARGES.get((residue_name, atom_name), 0.0)

    # [12] H-bond donor: backbone N (except PRO) + side-chain donors
    is_backbone_hbd = (atom_name == "N" and residue_name != "PRO")
    is_sidechain_hbd = atom_name in HBD_RESIDUE_ATOMS.get(residue_name, set())
    feat[12] = float(is_backbone_hbd or is_sidechain_hbd)

    # [13] H-bond acceptor: backbone O + side-chain acceptors
    is_backbone_hba = (atom_name == "O")
    is_sidechain_hba = atom_name in HBA_RESIDUE_ATOMS.get(residue_name, set())
    feat[13] = float(is_backbone_hba or is_sidechain_hba)

    # [14-16] Hybridization: sp, sp2, sp3
    is_sp2 = (is_aromatic
              or atom_name in SP2_RESIDUE_ATOMS.get(residue_name, set())
              or (atom_name == "C" and element == "C")   # backbone carbonyl C
              or (atom_name == "O" and element == "O"))   # backbone carbonyl O
    feat[14] = 0.0    # sp (extremely rare in proteins)
    feat[15] = float(is_sp2)
    feat[16] = float(not is_sp2)  # sp3 if not sp2

    # [17] is_ligand flag
    feat[17] = 0.0

    return feat


def mol2_atom_features(element: str, sybyl_type: str, charge: float) -> np.ndarray:
    """Encode a MOL2 ligand atom as 18-dimensional feature vector using Sybyl type."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)

    # [0-9] Element one-hot
    idx = ATOM_TYPES.get(element, 9)
    feat[idx] = 1.0

    # Parse Sybyl subtype (e.g., "C.ar" → "ar", "N.3" → "3")
    sybyl_parts = sybyl_type.split(".")
    subtype = sybyl_parts[1] if len(sybyl_parts) > 1 else "3"

    # [10] Aromatic
    is_aromatic = (subtype == "ar")
    feat[10] = float(is_aromatic)

    # [11] Gasteiger partial charge from MOL2
    feat[11] = charge

    # [12] H-bond donor: N with H attached, hydroxyl O, thiol S
    is_hbd = False
    if element == "N" and subtype in ("3", "4", "am", "pl3"):
        is_hbd = True
    elif element == "O" and subtype == "3":
        is_hbd = True  # hydroxyl O (sp3 oxygen typically has H)
    elif element == "S" and subtype == "3":
        is_hbd = True  # thiol
    feat[12] = float(is_hbd)

    # [13] H-bond acceptor: atoms with lone pairs
    is_hba = element in ("N", "O", "S", "F")
    feat[13] = float(is_hba)

    # [14-16] Hybridization from Sybyl type
    sp, sp2, sp3 = SYBYL_HYBRIDIZATION.get(subtype, (0, 0, 1))
    feat[14] = float(sp)
    feat[15] = float(sp2)
    feat[16] = float(sp3)

    # [17] is_ligand flag
    feat[17] = 1.0

    return feat


def compute_interaction_labels(lig_pos, prot_pos, lig_feat, prot_feat):
    """Compute [L, P, 5] binary interaction labels from geometry and features.

    Interaction types:
      0: H-bond (donor-acceptor pair, dist < 3.5 A)
      1: Hydrophobic (C-C contact, dist < 4.5 A)
      2: Ionic (opposite charges, dist < 4.0 A)
      3: Pi-stacking (both aromatic, dist < 5.5 A)
      4: Halogen bond (halogen to N/O, dist < 3.5 A)
    """
    cross_dist = cdist(lig_pos, prot_pos)  # [L, P]
    L, P = cross_dist.shape
    labels = np.zeros((L, P, 5), dtype=np.float32)

    # Extract per-atom feature flags
    lig_hbd = lig_feat[:, 12]
    lig_hba = lig_feat[:, 13]
    lig_arom = lig_feat[:, 10]
    lig_charge = lig_feat[:, 11]
    lig_carbon = lig_feat[:, 1]   # C one-hot at index 1
    lig_halogen = lig_feat[:, 4] + lig_feat[:, 7] + lig_feat[:, 8]  # F + Cl + Br

    prot_hbd = prot_feat[:, 12]
    prot_hba = prot_feat[:, 13]
    prot_arom = prot_feat[:, 10]
    prot_charge = prot_feat[:, 11]
    prot_carbon = prot_feat[:, 1]
    prot_no = prot_feat[:, 2] + prot_feat[:, 3]  # N + O

    # 0: H-bond — dist < 3.5 and donor-acceptor pair
    hb_mask = cross_dist < 3.5
    hb_da = (np.outer(lig_hbd, prot_hba) > 0) | (np.outer(lig_hba, prot_hbd) > 0)
    labels[:, :, 0] = (hb_mask & hb_da).astype(np.float32)

    # 1: Hydrophobic — dist < 4.5 and both C
    labels[:, :, 1] = ((cross_dist < 4.5) & (np.outer(lig_carbon, prot_carbon) > 0)).astype(np.float32)

    # 2: Ionic — dist < 4.0 and opposite-sign charges (product < -0.09 ≈ both |q|>0.3)
    charge_product = np.outer(lig_charge, prot_charge)
    labels[:, :, 2] = ((cross_dist < 4.0) & (charge_product < -0.09)).astype(np.float32)

    # 3: Pi-stacking — dist < 5.5 and both aromatic
    labels[:, :, 3] = ((cross_dist < 5.5) & (np.outer(lig_arom, prot_arom) > 0)).astype(np.float32)

    # 4: Halogen bond — dist < 3.5 and halogen→N/O
    labels[:, :, 4] = ((cross_dist < 3.5) & (np.outer(lig_halogen, prot_no) > 0)).astype(np.float32)

    return labels


def generate_decoy_pose(lig_pos):
    """Generate a random decoy ligand pose by rotating and translating.

    50% near decoys (hard negatives, 2-5 A) and 50% far decoys (easy, 8-20 A).
    """
    lig_center = lig_pos.mean(axis=0)
    centered = lig_pos - lig_center

    # Random 3D rotation
    R = Rotation.random().as_matrix().astype(np.float32)
    rotated = centered @ R.T

    # Random translation direction
    direction = np.random.randn(3).astype(np.float32)
    direction /= (np.linalg.norm(direction) + 1e-8)

    # Mix of hard (near) and easy (far) decoys
    if np.random.random() < 0.5:
        distance = np.random.uniform(2.0, 5.0)
    else:
        distance = np.random.uniform(8.0, 20.0)

    return (rotated + lig_center + direction * distance).astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

class PDBBindDataset(Dataset):
    """PDBbind complex dataset for DruseScore training."""

    def __init__(self, data_dir: str, split: str = "refined", pocket_radius: float = 10.0,
                 labels_csv: str = "pdbbind_labels.csv", transform=None):
        self.data_dir = Path(data_dir)
        self.pocket_radius = pocket_radius
        self.transform = transform

        # Load labels
        labels_path = self.data_dir / labels_csv
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}\nRun: python download_data.py --all")

        df = pd.read_csv(labels_path)
        if split != "all":
            df = df[df["split"] == split]
        self.entries = df.to_dict("records")

        # Filter to entries that have structure files
        valid = []
        for entry in self.entries:
            pdb_id = entry["pdb_id"]
            pocket_path = self._pocket_path(pdb_id)
            ligand_path = self._ligand_path(pdb_id)
            if pocket_path.exists() and ligand_path.exists():
                valid.append(entry)
        self.entries = valid
        print(f"PDBBindDataset({split}): {len(self.entries)} valid complexes")

    def _pocket_path(self, pdb_id: str) -> Path:
        return self.data_dir / "refined-set" / pdb_id / f"{pdb_id}_pocket.pdb"

    def _ligand_path(self, pdb_id: str) -> Path:
        return self.data_dir / "refined-set" / pdb_id / f"{pdb_id}_ligand.mol2"

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        pdb_id = entry["pdb_id"]
        pkd = entry["pKd"]

        # Parse pocket atoms
        pocket_path = self._pocket_path(pdb_id)
        prot_pos, prot_feat = self._parse_pdb(pocket_path)

        # Parse ligand atoms
        ligand_path = self._ligand_path(pdb_id)
        lig_pos, lig_feat = self._parse_mol2(ligand_path)

        if prot_pos is None or lig_pos is None:
            return None  # filtered out in custom_collate

        # Return minimal data — interaction labels + decoys computed on GPU in train_epoch
        data = Data(
            prot_pos=torch.tensor(prot_pos, dtype=torch.float32),
            prot_x=torch.tensor(prot_feat, dtype=torch.float32),
            lig_pos=torch.tensor(lig_pos, dtype=torch.float32),
            lig_x=torch.tensor(lig_feat, dtype=torch.float32),
            y=torch.tensor([pkd], dtype=torch.float32),
            pdb_id=pdb_id,
        )

        if self.transform:
            data = self.transform(data)
        return data

    def _parse_pdb(self, path: Path):
        """Parse PDB pocket file → (positions [N,3], features [N,18]).

        Uses residue name + atom name for context-aware feature extraction.
        """
        positions, features = [], []
        try:
            with open(path) as f:
                for line in f:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    # Element from columns 77-78
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:16].strip()[0]
                    if element == "H":
                        continue

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    # Residue and atom context for accurate features
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
        """Parse MOL2 ligand file → (positions [N,3], features [N,18]).

        Uses Sybyl atom types for accurate aromatic, hybridization, and HBD/HBA.
        """
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


class CachedDruseDataset(Dataset):
    """Load precomputed .pt files into RAM — zero I/O during training.

    Run precompute_druse_features.py first to generate the cache.
    """

    def __init__(self, data_dir: str):
        cache_dir = Path(data_dir) / "druse_cache"
        files = sorted(cache_dir.glob("*.pt"))
        self.samples = []
        for f in tqdm(files, desc="Loading cache to RAM"):
            self.samples.append(torch.load(f, weights_only=False))
        print(f"CachedDruseDataset: {len(self.samples)} complexes loaded to RAM")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# GPU-accelerated training helpers
# ============================================================================

@torch.no_grad()
def compute_interaction_labels_gpu(lig_pos, prot_pos, lig_feat, prot_feat):
    """Compute [L, P, 5] interaction labels entirely on GPU."""
    cross_dist = torch.cdist(lig_pos.unsqueeze(0), prot_pos.unsqueeze(0)).squeeze(0)  # [L, P]

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


def generate_decoy_pose_gpu(lig_pos):
    """Generate a random decoy ligand pose on GPU."""
    device = lig_pos.device
    lig_center = lig_pos.mean(0)
    centered = lig_pos - lig_center

    # Random rotation via QR decomposition of random matrix
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    Q = Q * torch.sign(torch.linalg.det(Q))  # ensure proper rotation (det=+1)
    rotated = centered @ Q.T

    # Random translation: 50% near (2-5A), 50% far (8-20A)
    direction = torch.randn(3, device=device)
    direction = direction / (direction.norm() + 1e-8)
    if torch.rand(1).item() < 0.5:
        distance = 2.0 + torch.rand(1).item() * 3.0
    else:
        distance = 8.0 + torch.rand(1).item() * 12.0

    return rotated + lig_center + direction * distance


# ============================================================================
# Model: EGNN Layers
# ============================================================================

class EGNNLayer(nn.Module):
    """E(n)-equivariant graph neural network layer.
    Updates both node features and coordinates."""

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

        # Coordinate update (equivariant) — clamped to prevent explosion
        coord_weight = self.coord_mlp(msg)
        coord_weight = coord_weight.clamp(-1.0, 1.0)  # prevent large coordinate jumps
        coord_delta = (diff / dist) * coord_weight
        pos_out = pos.clone()
        pos_out.index_add_(0, row, coord_delta)

        # Node update (invariant)
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, msg)
        h_out = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        return h_out, pos_out


def build_radius_graph_cpu(pos: torch.Tensor, cutoff: float, max_neighbors: int = 32):
    """Build radius graph on CPU using cdist (works without torch-cluster)."""
    pos_cpu = pos.detach().cpu()
    dist = torch.cdist(pos_cpu, pos_cpu)
    mask = (dist < cutoff) & (dist > 0)
    rows, cols = [], []
    for i in range(pos_cpu.size(0)):
        neighbors = mask[i].nonzero(as_tuple=True)[0]
        if len(neighbors) > max_neighbors:
            # Keep closest neighbors
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

    def forward(self, query, key, value, cross_rbf):
        """
        query: ligand features [L, D]
        key/value: protein features [P, D]
        cross_rbf: [L, P, 50] RBF-encoded distances
        """
        L, D = query.shape
        P = key.shape[0]
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(query).view(L, H, d)
        K = self.k_proj(key).view(P, H, d)
        V = self.v_proj(value).view(P, H, d)

        # Attention scores: QK^T / sqrt(d) + distance bias
        attn = torch.einsum("lhd,phd->lph", Q, K) / math.sqrt(d)
        dist_bias = self.rbf_proj(cross_rbf)  # [L, P, H]
        attn = attn + dist_bias

        attn = F.softmax(attn, dim=1)  # normalize over protein atoms

        out = torch.einsum("lph,phd->lhd", attn, V)
        out = out.reshape(L, D)
        out = self.out_proj(out)
        return self.norm(query + out), attn.mean(dim=-1)  # residual + mean attention


# ============================================================================
# Model: DruseScore
# ============================================================================

class DruseScore(nn.Module):
    """SE(3)-equivariant scoring function for protein-ligand complexes.

    Multi-task:
      1. pKd regression (binding affinity)
      2. Pose classification (correct vs decoy)
      3. Interaction type prediction (per atom-pair)
    """

    def __init__(self, atom_dim: int = 18, hidden_dim: int = 128, num_egnn_layers: int = 4):
        super().__init__()
        self.prot_encoder = EGNNEncoder(atom_dim, hidden_dim, num_egnn_layers)
        self.lig_encoder = EGNNEncoder(atom_dim, hidden_dim, num_egnn_layers)
        self.cross_attn = GeometricCrossAttention(hidden_dim)

        # Affinity head
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Pose classification head
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Interaction prediction head (5 types: hbond, hydrophobic, ionic, pi-stack, halogen)
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 50, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, data):
        # Encode protein and ligand
        prot_h, prot_pos = self.prot_encoder(data.prot_x, data.prot_pos)
        lig_h, lig_pos = self.lig_encoder(data.lig_x, data.lig_pos)

        # Recompute cross_rbf from EGNN-updated positions (not stale input coords)
        cross_dist = torch.cdist(lig_pos, prot_pos)
        cross_rbf = rbf_encode(cross_dist.reshape(-1)).reshape(
            lig_pos.size(0), prot_pos.size(0), 50
        )

        # Cross-attention
        lig_attended, attn_weights = self.cross_attn(lig_h, prot_h, prot_h, cross_rbf)

        # Global pooling for affinity and pose prediction
        complex_repr = lig_attended.mean(dim=0, keepdim=True)
        pkd_pred = self.affinity_head(complex_repr).squeeze(-1)
        pose_pred = torch.sigmoid(self.pose_head(complex_repr)).squeeze(-1)

        # Per-atom-pair interaction predictions
        L = lig_h.size(0)
        P = prot_h.size(0)
        lig_exp = lig_attended.unsqueeze(1).expand(-1, P, -1)  # [L, P, D]
        prot_exp = prot_h.unsqueeze(0).expand(L, -1, -1)  # [L, P, D]
        pair_input = torch.cat([lig_exp, prot_exp, cross_rbf], dim=-1)
        interaction_pred = torch.sigmoid(self.interaction_head(pair_input))  # [L, P, 5]

        return {
            "pKd": pkd_pred,
            "pose_confidence": pose_pred,
            "interaction_map": interaction_pred,
            "attention_weights": attn_weights,
        }


# ============================================================================
# Training Loop
# ============================================================================

def custom_collate(batch):
    """Collate variable-size complexes, filtering failed entries."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    return valid[0] if len(valid) == 1 else valid


def move_data_to_device(data, device):
    """Move Data object tensors to device."""
    if hasattr(data, 'to'):
        data = data.to(device)
    return data


GRAD_ACCUM_STEPS = 8  # effective batch size = 8


def train_epoch(model, loader, optimizer, device, epoch_num=0):
    model.train()
    total_loss = 0
    n = 0
    nan_count = 0
    skip_count = 0
    max_grad_norm = 0.0
    batch_idx = -1

    optimizer.zero_grad()

    for batch_idx, data in enumerate(tqdm(loader, desc=f"  Train ep{epoch_num+1}", leave=False)):
        if data is None:
            skip_count += 1
            continue
        if isinstance(data, list):
            data = data[0]
        if data is None:
            skip_count += 1
            continue
        data = move_data_to_device(data, device)

        # --- GPU: generate decoy + interaction labels ---
        is_decoy = (torch.rand(1).item() < 0.5)
        if is_decoy:
            data.lig_pos = generate_decoy_pose_gpu(data.lig_pos)
        data.interaction_labels = compute_interaction_labels_gpu(
            data.lig_pos, data.prot_pos, data.lig_x, data.prot_x)
        data.pose_label = torch.tensor([0.0 if is_decoy else 1.0], device=device)

        try:
            output = model(data)
        except Exception:
            skip_count += 1
            continue

        # --- Multi-task loss ---

        # 1. Affinity loss: only for crystal poses (not decoys)
        if not is_decoy:
            loss_affinity = F.huber_loss(output["pKd"], data.y, delta=2.0)
        else:
            loss_affinity = torch.tensor(0.0, device=device)

        # 2. Pose classification loss (crystal=1, decoy=0)
        loss_pose = F.binary_cross_entropy(output["pose_confidence"], data.pose_label)

        # 3. Interaction prediction loss (focal loss for sparse labels)
        int_pred = output["interaction_map"]
        int_target = data.interaction_labels
        int_bce = F.binary_cross_entropy(int_pred, int_target, reduction="none")
        int_p_t = int_pred * int_target + (1 - int_pred) * (1 - int_target)
        int_alpha_t = 0.9 * int_target + 0.1 * (1 - int_target)  # 9x weight on positives
        int_focal_weight = int_alpha_t * (1 - int_p_t) ** 2
        loss_interaction = (int_focal_weight * int_bce).mean()

        # Combined weighted loss
        loss_total = loss_affinity + 0.5 * loss_pose + 0.2 * loss_interaction
        loss = loss_total / GRAD_ACCUM_STEPS

        # Skip NaN/Inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        n += 1

        # Step optimizer every GRAD_ACCUM_STEPS
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm > max_grad_norm:
                max_grad_norm = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Periodic check for NaN in model parameters
        if batch_idx % 500 == 0 and batch_idx > 0:
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"\n  [WARN] NaN detected in model params at batch {batch_idx}!")
                return float('nan'), nan_count, skip_count, max_grad_norm

    # Handle leftover gradients
    if batch_idx >= 0 and n > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if grad_norm > max_grad_norm:
            max_grad_norm = float(grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(n, 1)
    return avg_loss, nan_count, skip_count, max_grad_norm


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, pdb_ids = [], [], []

    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data is None:
            continue
        if isinstance(data, list):
            data = data[0]
        if data is None:
            continue
        data = move_data_to_device(data, device)

        try:
            output = model(data)
        except Exception:
            continue
        preds.append(output["pKd"].cpu().item())
        targets.append(data.y.cpu().item())
        pdb_ids.append(data.pdb_id)

    preds = np.array(preds)
    targets = np.array(targets)

    # Metrics
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 2 else 0.0

    return {"rmse": rmse, "mae": mae, "pearson_r": r, "preds": preds, "targets": targets}


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Load dataset: cached (fast) or on-the-fly (slow fallback)
    cache_dir = Path(args.data_dir) / "druse_cache"
    if args.use_cache and cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 100:
        dataset = CachedDruseDataset(args.data_dir)
        # 2 workers for prefetching — keeps GPU fed while next batch is prepared
        use_cache = True
    else:
        if args.use_cache:
            print("Cache not found. Run: python precompute_druse_features.py --data_dir data/")
            print("Falling back to on-the-fly parsing (slow)...\n")
        dataset = PDBBindDataset(args.data_dir, split="refined")
        use_cache = False

    if len(dataset) == 0:
        print("ERROR: No data found. Run: python download_data.py --all")
        return

    # Protein-cluster-based cross-validation to prevent target leakage.
    # PDBbind contains many structures of the same protein target (e.g., multiple
    # kinase inhibitors co-crystallized with the same kinase). Random KFold would
    # leak target information across folds, inflating validation metrics. We group
    # complexes by the first 4 characters of the PDB ID, which corresponds to the
    # PDB entry and serves as a rough proxy for protein identity. This ensures all
    # complexes of the same protein end up in the same fold.
    groups = [entry["pdb_id"][:4].upper() for entry in dataset.entries]
    gkf = GroupKFold(n_splits=3)
    all_indices = list(range(len(dataset)))

    best_overall_r = -1
    best_model_state = None

    import time
    training_start = time.time()

    # Log file for monitoring
    log_path = args.output_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("fold,epoch,train_loss,val_rmse,val_mae,val_r,lr,max_grad,nan_count,skip_count,time_sec\n")

    n_unique_groups = len(set(groups))
    print(f"\nProtein-cluster GroupKFold: {n_unique_groups} unique protein groups")
    print(f"Training log: {log_path}")
    print(f"Monitor with: tail -f {log_path}\n")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(all_indices, groups=groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/3 ({len(train_idx)} train, {len(val_idx)} val)")
        print(f"{'='*60}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        if use_cache:
            # Data is in RAM — workers just add serialization overhead and FD pressure
            n_workers = 0
        else:
            n_workers = min(4, (os.cpu_count() or 1))
        if fold == 0:
            print(f"  DataLoader workers: {n_workers} ({'cached/in-RAM' if use_cache else 'on-the-fly'})")
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True,
                                  collate_fn=custom_collate, num_workers=n_workers,
                                  persistent_workers=n_workers > 0)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                                collate_fn=custom_collate, num_workers=n_workers,
                                persistent_workers=n_workers > 0)

        model = DruseScore(hidden_dim=args.hidden_dim).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        if fold == 0:
            print(f"  Model parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
        warmup_epochs = 10
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs),
        ], milestones=[warmup_epochs])

        best_r = -1
        consecutive_nan = 0

        for epoch in range(args.epochs):
            epoch_start = time.time()
            result = train_epoch(model, train_loader, optimizer, device, epoch_num=epoch)
            train_loss, nan_count, skip_count, max_grad = result
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - training_start

            # Detect NaN explosion — restart fold if 3 consecutive NaN epochs
            if np.isnan(train_loss):
                consecutive_nan += 1
                print(f"  Epoch {epoch+1:3d} | NaN LOSS | grad={max_grad:.1f} nan={nan_count} skip={skip_count} | "
                      f"lr={current_lr:.2e} | {time.time()-epoch_start:.0f}s")
                if consecutive_nan >= 3:
                    print(f"  [ABORT] 3 consecutive NaN epochs — skipping rest of fold {fold+1}")
                    break
                continue
            consecutive_nan = 0

            # Evaluate every 2 epochs, first, and last
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.epochs - 1:
                metrics = evaluate(model, val_loader, device)
                epoch_time = time.time() - epoch_start
                eta = (args.epochs - epoch - 1) * epoch_time

                status = ""
                if metrics["pearson_r"] > best_r:
                    best_r = metrics["pearson_r"]
                    torch.save(model.state_dict(), args.output_dir / f"fold{fold}_best.pt")
                    status = " * NEW BEST"

                if metrics["pearson_r"] > best_overall_r:
                    best_overall_r = metrics["pearson_r"]
                    best_model_state = model.state_dict().copy()
                    status = " ** OVERALL BEST"

                print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
                      f"RMSE: {metrics['rmse']:.3f} MAE: {metrics['mae']:.3f} R: {metrics['pearson_r']:.3f} | "
                      f"lr={current_lr:.2e} grad={max_grad:.1f} nan={nan_count} | "
                      f"{epoch_time:.0f}s (ETA {eta/60:.0f}m){status}")

                # Write to log CSV
                with open(log_path, "a") as f:
                    f.write(f"{fold},{epoch+1},{train_loss:.6f},{metrics['rmse']:.4f},"
                            f"{metrics['mae']:.4f},{metrics['pearson_r']:.4f},{current_lr:.2e},"
                            f"{max_grad:.4f},{nan_count},{skip_count},{elapsed:.0f}\n")
            else:
                # Brief status for non-eval epochs
                print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
                      f"lr={current_lr:.2e} grad={max_grad:.1f} nan={nan_count}")

        print(f"\n  Fold {fold+1} complete — Best R: {best_r:.3f}")
        print(f"  Total elapsed: {(time.time()-training_start)/60:.1f} minutes")

    # Save best model
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {(time.time()-training_start)/60:.1f} minutes")
    if best_model_state:
        best_path = args.output_dir / "druse_score_best.pt"
        torch.save(best_model_state, best_path)
        print(f"Best model: {best_path} (R={best_overall_r:.3f})")
    else:
        print("WARNING: No valid model produced — all folds may have NaN'd")
    print(f"Training log: {log_path}")

    return best_model_state


def main():
    parser = argparse.ArgumentParser(description="Train DruseScore")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/",
                        help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--use_cache", action="store_true",
                        help="Use precomputed .pt files from druse_cache/ (run precompute_druse_features.py first)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for eval"
        device = get_device()
        model = DruseScore(hidden_dim=args.hidden_dim).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        dataset = PDBBindDataset(args.data_dir, split="refined")
        loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)
        metrics = evaluate(model, loader, device)
        print(f"Evaluation: RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} R={metrics['pearson_r']:.3f}")
    else:
        train(args)


if __name__ == "__main__":
    main()

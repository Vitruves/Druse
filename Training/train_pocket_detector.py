#!/usr/bin/env python3
"""
Train P2Rank-style ML pocket detector for Druse.

Architecture:
  - Compute protein surface points (Connolly-like surface)
  - Extract per-point chemical features (hydrophobicity, charge, aromaticity, etc.)
  - GNN classifies each surface point as pocket/non-pocket
  - DBSCAN cluster pocket points → binding site predictions

Training data:
  - PDBbind v2020 refined set (known binding sites from ligand positions)
  - scPDB (optional, curated pocket database)

Usage:
  python train_pocket_detector.py --data_dir data/ --epochs 80
  python train_pocket_detector.py --data_dir data/ --epochs 80 --use_cache --resume
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, radius_graph
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve
from tqdm import tqdm


# ============================================================================
# Surface Point Features
# ============================================================================

# Per-surface-point features (11 dimensions):
#   [0-2]: surface normal (nx, ny, nz) — weighted multi-atom average
#   [3]:   distance to nearest atom
#   [4]:   hydrophobicity score of nearest residue (Kyte-Doolittle, normalized)
#   [5]:   partial charge of nearest atom (residue-based at pH 7.4)
#   [6]:   aromatic ring nearby (0/1, from residue+atom context)
#   [7]:   H-bond donor nearby (0/1, backbone N + side-chain)
#   [8]:   H-bond acceptor nearby (0/1, backbone O + side-chain)
#   [9]:   buriedness (26-direction ray casting)
#   [10]:  curvature (local PCA eigenvalue ratio)
SURFACE_FEAT_DIM = 11

HYDROPHOBICITY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLU": -3.5, "GLN": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

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

# H-bond donors: backbone N (except PRO) + specific side-chain atoms
HBD_RESIDUE_ATOMS = {
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"}, "TRP": {"NE1"},
    "CYS": {"SG"},
}

# H-bond acceptors: backbone O + specific side-chain atoms
HBA_RESIDUE_ATOMS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "HIS": {"ND1", "NE2"},
    "MET": {"SD"}, "CYS": {"SG"},
}

# 26 probe directions for ray-casting buriedness (faces + edges + corners of a cube)
_PROBE_DIRECTIONS = []
for _dx in (-1, 0, 1):
    for _dy in (-1, 0, 1):
        for _dz in (-1, 0, 1):
            if _dx == _dy == _dz == 0:
                continue
            _d = np.array([_dx, _dy, _dz], dtype=np.float32)
            _PROBE_DIRECTIONS.append(_d / np.linalg.norm(_d))
_PROBE_DIRECTIONS = np.array(_PROBE_DIRECTIONS, dtype=np.float32)  # [26, 3]


def farthest_point_sample(points, n_samples):
    """Farthest point sampling for spatially uniform subsampling."""
    n = len(points)
    if n <= n_samples:
        return np.arange(n)

    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = np.random.randint(n)

    min_dists = np.full(n, np.inf, dtype=np.float64)

    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected[i] = np.argmax(min_dists)

    return selected


def compute_surface_features(pdb_path: Path, probe_radius: float = 1.4,
                              grid_spacing: float = 0.7) -> tuple:
    """Compute surface points and features from a PDB file.

    Returns (positions [N,3], features [N,11], nearest_atom_idx [N]).
    All features use residue+atom context for chemical accuracy.

    Uses 0.7A grid spacing (finer than 1.0A) to capture pocket-scale geometry.
    """
    from Bio.PDB import PDBParser as BioPDBParser

    parser = BioPDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(pdb_path))
    except Exception:
        return None, None, None

    # Extract atom positions and properties with full residue context
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # skip HETATM
                    continue
                res_name = residue.get_resname()
                for atom in residue:
                    if atom.element == "H":
                        continue
                    pos = atom.get_vector().get_array()
                    atom_name = atom.get_name()

                    # Residue-context features
                    charge = RESIDUE_ATOM_CHARGES.get((res_name, atom_name), 0.0)
                    hydro = HYDROPHOBICITY.get(res_name, 0.0)
                    is_aromatic = atom_name in AROMATIC_RESIDUE_ATOMS.get(res_name, set())

                    is_backbone_hbd = (atom_name == "N" and res_name != "PRO")
                    is_hbd = is_backbone_hbd or atom_name in HBD_RESIDUE_ATOMS.get(res_name, set())

                    is_backbone_hba = (atom_name == "O")
                    is_hba = is_backbone_hba or atom_name in HBA_RESIDUE_ATOMS.get(res_name, set())

                    atoms.append({
                        "pos": pos,
                        "element": atom.element,
                        "residue": res_name,
                        "atom_name": atom_name,
                        "charge": charge,
                        "hydrophobicity": hydro,
                        "is_aromatic": is_aromatic,
                        "is_hbd": is_hbd,
                        "is_hba": is_hba,
                    })
        break  # first model only

    if not atoms:
        return None, None, None

    atom_pos = np.array([a["pos"] for a in atoms], dtype=np.float32)

    # Generate surface points using grid + VdW distance filter
    bbox_min = atom_pos.min(axis=0) - 6.0
    bbox_max = atom_pos.max(axis=0) + 6.0

    x = np.arange(bbox_min[0], bbox_max[0], grid_spacing)
    y = np.arange(bbox_min[1], bbox_max[1], grid_spacing)
    z = np.arange(bbox_min[2], bbox_max[2], grid_spacing)
    grid = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T

    # Filter to surface region
    from scipy.spatial import cKDTree
    tree = cKDTree(atom_pos)
    dists, indices = tree.query(grid)

    # VdW radii
    vdw = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}
    atom_radii = np.array([vdw.get(a["element"], 1.7) for a in atoms])
    surface_dist = dists - atom_radii[indices]

    # Keep points near the molecular surface
    mask = (surface_dist >= -0.5) & (surface_dist <= probe_radius + 0.5)
    surface_points = grid[mask]
    nearest_idx = indices[mask]

    if len(surface_points) < 10:
        return None, None, None

    # Subsample to max 8000 points using farthest-point sampling (uniform coverage)
    max_points = 8000
    if len(surface_points) > max_points:
        fps_idx = farthest_point_sample(surface_points, max_points)
        surface_points = surface_points[fps_idx]
        nearest_idx = nearest_idx[fps_idx]

    # Build KDTree of surface points for normal and curvature computation
    surf_tree = cKDTree(surface_points)
    k_curvature = 12
    _, surf_nn = surf_tree.query(surface_points, k=k_curvature + 1)
    surf_nn = surf_nn[:, 1:]  # exclude self

    # Compute features
    features = np.zeros((len(surface_points), SURFACE_FEAT_DIM), dtype=np.float32)

    for i in range(len(surface_points)):
        ni = nearest_idx[i]
        atom = atoms[ni]

        # [0-2] Normal: weighted average from multiple nearby atoms (robust in concave regions)
        nearby_k = 5
        nearby_dists, nearby_idxs = tree.query(surface_points[i], k=nearby_k)
        weighted_normal = np.zeros(3)
        total_weight = 0.0
        for d, j in zip(nearby_dists, nearby_idxs):
            w = 1.0 / (d * d + 0.01)  # inverse-square weighting
            direction = surface_points[i] - atom_pos[j]
            weighted_normal += w * direction
            total_weight += w
        weighted_normal /= (total_weight + 1e-8)
        norm_len = np.linalg.norm(weighted_normal)
        if norm_len > 1e-8:
            weighted_normal /= norm_len
        features[i, 0:3] = weighted_normal

        # [3] Distance to nearest atom
        features[i, 3] = nearby_dists[0]

        # [4] Hydrophobicity (normalized by max Kyte-Doolittle value)
        features[i, 4] = atom["hydrophobicity"] / 4.5

        # [5] Charge (residue-based partial charge at pH 7.4)
        features[i, 5] = atom["charge"]

        # [6] Aromatic (from residue + atom context)
        features[i, 6] = float(atom["is_aromatic"])

        # [7] H-bond donor (backbone N + side-chain donors)
        features[i, 7] = float(atom["is_hbd"])

        # [8] H-bond acceptor (backbone O + side-chain acceptors)
        features[i, 8] = float(atom["is_hba"])

        # [9] Buriedness via 26-direction ray casting
        test_points = surface_points[i] + _PROBE_DIRECTIONS * 3.0  # [26, 3]
        probe_dists, _ = tree.query(test_points)  # [26]
        blocked_count = np.sum(probe_dists < 2.5)
        features[i, 9] = blocked_count / 26.0

        # [10] Curvature via local PCA on surface neighborhood
        neighbors = surface_points[surf_nn[i]]  # [k, 3]
        centered = neighbors - surface_points[i]
        cov = centered.T @ centered / k_curvature
        eigenvalues = np.linalg.eigvalsh(cov)
        # Ratio of smallest to total variance: 0 = flat, ~0.33 = isotropic (sphere-like)
        features[i, 10] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)

    return surface_points, features, nearest_idx


# ============================================================================
# Dataset
# ============================================================================

class CachedPocketDataset(Dataset):
    """Load precomputed .pt files from pocket_cache/ — instant loading, GPU stays busy."""

    def __init__(self, data_dir: str):
        self.cache_dir = Path(data_dir) / "pocket_cache"
        self.files = sorted(self.cache_dir.glob("*.pt"))
        print(f"CachedPocketDataset: {len(self.files)} cached complexes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return torch.load(self.files[idx], weights_only=False)
        except Exception:
            return Data(
                x=torch.zeros(10, SURFACE_FEAT_DIM),
                pos=torch.zeros(10, 3),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                y=torch.zeros(10),
                pdb_id="error"
            )


class PocketSurfaceDataset(Dataset):
    """Surface point classification dataset for pocket detection (computes on-the-fly)."""

    def __init__(self, data_dir: str, ligand_dist_threshold: float = 4.0):
        self.data_dir = Path(data_dir)
        self.ligand_dist_threshold = ligand_dist_threshold
        self.entries = []

        # Find all PDBbind entries with pocket + ligand files
        refined_dir = self.data_dir / "refined-set"
        if refined_dir.exists():
            for pdb_dir in sorted(refined_dir.iterdir()):
                if not pdb_dir.is_dir():
                    continue
                pdb_id = pdb_dir.name
                protein_path = pdb_dir / f"{pdb_id}_protein.pdb"
                ligand_path = pdb_dir / f"{pdb_id}_ligand.mol2"
                if protein_path.exists() and ligand_path.exists():
                    self.entries.append((pdb_id, protein_path, ligand_path))

        print(f"PocketSurfaceDataset: {len(self.entries)} entries")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pdb_id, protein_path, ligand_path = self.entries[idx]

        # Compute surface features
        surf_pos, surf_feat, _ = compute_surface_features(protein_path)
        if surf_pos is None:
            return self._dummy()

        # Load ligand positions for labeling
        lig_pos = self._parse_ligand_positions(ligand_path)
        if lig_pos is None or len(lig_pos) == 0:
            return self._dummy()

        # Label: 1 if surface point is within threshold of any ligand atom
        from scipy.spatial.distance import cdist
        dists = cdist(surf_pos, lig_pos).min(axis=1)
        labels = (dists < self.ligand_dist_threshold).astype(np.float32)

        # Build radius graph for GNN
        pos_t = torch.tensor(surf_pos, dtype=torch.float32)
        feat_t = torch.tensor(surf_feat, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        edge_index = radius_graph(pos_t, r=5.0, max_num_neighbors=24)

        return Data(
            x=feat_t, pos=pos_t, edge_index=edge_index,
            y=labels_t, pdb_id=pdb_id
        )

    def _parse_ligand_positions(self, path: Path) -> np.ndarray:
        positions = []
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
                    if len(parts) >= 5:
                        elem = parts[5].split(".")[0] if len(parts) > 5 else "C"
                        if elem != "H":
                            positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        except Exception:
            return None
        return np.array(positions, dtype=np.float32) if positions else None

    def _dummy(self):
        return Data(
            x=torch.zeros(10, SURFACE_FEAT_DIM),
            pos=torch.zeros(10, 3),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.zeros(10),
            pdb_id="dummy"
        )


# ============================================================================
# Model
# ============================================================================

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.85, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for imbalanced binary classification.
    alpha=0.85: pocket points (~5-8% of surface) get ~5.7x weight vs non-pocket.
    gamma=2.0: focus on hard examples (confidently wrong predictions).
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


class EdgeEncoder(nn.Module):
    """Encode edge distances as RBF features for distance-aware message passing."""

    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        # Fixed RBF centers from 0 to cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) * 0.5  # overlap between RBFs

    def forward(self, edge_index, pos):
        """Compute RBF-encoded edge distances.
        Returns [num_edges, num_rbf] tensor.
        """
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)  # [E, 1]
        rbf = torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))  # [E, num_rbf]
        return rbf


class PocketGNN(nn.Module):
    """GATv2-based GNN for surface point pocket classification.

    Improvements over plain GCNConv:
    - GATv2Conv: attention-weighted message passing (learns which neighbors matter)
    - Edge distance encoding: RBF features inform attention (closer = stronger signal)
    - Deeper network: 5 layers with dropout for 10k+ training samples
    - Multi-head attention: 4 heads for diverse neighborhood aggregation
    - Residual connections with pre-norm (stable training for deeper networks)
    """

    def __init__(self, in_dim: int = 11, hidden_dim: int = 128, num_layers: int = 5,
                 num_heads: int = 4, dropout: float = 0.1, num_rbf: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.edge_encoder = EdgeEncoder(num_rbf=num_rbf, cutoff=5.0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Edge feature projection: RBF → per-head scalar bias
        self.edge_projs = nn.ModuleList([
            nn.Linear(num_rbf, num_heads) for _ in range(num_layers)
        ])

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                      concat=True, dropout=dropout, add_self_loops=True,
                      edge_dim=num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_rbf = self.edge_encoder(data.edge_index, data.pos)  # [E, num_rbf]

        for i, (conv, norm, edge_proj) in enumerate(
            zip(self.convs, self.norms, self.edge_projs)
        ):
            # Pre-norm residual: norm → conv → dropout → add
            x_res = x
            x = norm(x)
            edge_attr = edge_proj(edge_rbf)  # [E, num_heads]
            x = conv(x, data.edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x = x + x_res  # residual

        return torch.sigmoid(self.head(x)).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def train(args):
    import time

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Use cached features if available (much faster — GPU won't starve)
    cache_dir = Path(args.data_dir) / "pocket_cache"
    if args.use_cache and cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 100:
        dataset = CachedPocketDataset(args.data_dir)
    else:
        if args.use_cache:
            print("Cache not found. Run: python precompute_pocket_features.py --data_dir data/")
            print("Falling back to on-the-fly computation (slow)...\n")
        dataset = PocketSurfaceDataset(args.data_dir)
    if len(dataset) == 0:
        print("No data found. Run: python download_data.py --all")
        return

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    print(f"Train: {n_train}, Val: {n_val}")

    def pyg_collate(batch):
        return batch[0] if len(batch) == 1 else batch

    # Cached data loads instantly from disk — no workers needed.
    if isinstance(dataset, CachedPocketDataset):
        n_workers = 0
    else:
        n_workers = min(4, os.cpu_count() or 1)
    print(f"DataLoader workers: {n_workers} ({'cached' if n_workers == 0 else 'on-the-fly'})")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=n_workers,
                              collate_fn=pyg_collate, persistent_workers=n_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=n_workers,
                            collate_fn=pyg_collate, persistent_workers=n_workers > 0)

    model = PocketGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                      num_heads=args.num_heads, dropout=args.dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * n_train,
        pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=100
    )

    best_f1 = 0
    best_auc = 0
    start_epoch = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = output_dir / "pocket_detector_best.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    print("  Optimizer state incompatible (architecture changed), skipping")
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if "best_auc" in ckpt:
                best_auc = ckpt["best_auc"]
            if "best_f1" in ckpt:
                best_f1 = ckpt["best_f1"]
            print(f"Resumed from {resume_path} (epoch {start_epoch}, best_auc={best_auc:.4f}, best_f1={best_f1:.4f})")
        else:
            print(f"WARNING: --resume specified but no checkpoint found at {resume_path}")

    # Log file
    log_path = output_dir / "pocket_training_log.csv"
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_auc,val_precision,val_recall,val_f1,val_threshold,lr,time_sec\n")
    else:
        print(f"Appending to existing log: {log_path}")
    print(f"Training log: {log_path}")
    print(f"Monitor with: tail -f {log_path}\n")

    training_start = time.time()
    patience_counter = 0
    patience = 15  # early stopping patience

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        n = 0
        nan_count = 0

        for data in tqdm(train_loader, desc=f"  Pocket ep{epoch+1}", leave=False):
            try:
                data = data.to(device)
                optimizer.zero_grad()
                pred = model(data)
                loss = focal_loss(pred, data.y, alpha=0.85, gamma=2.0)

                if torch.isnan(loss):
                    nan_count += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n += 1
            except Exception:
                continue

        avg_loss = total_loss / max(n, 1)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start

        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.epochs - 1:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data in val_loader:
                    try:
                        data = data.to(device)
                        pred = model(data).cpu().numpy()
                        all_preds.extend(pred)
                        all_labels.extend(data.y.cpu().numpy())
                    except Exception:
                        continue

            preds = np.array(all_preds)
            labels = np.array(all_labels)
            auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0

            # Find F1-optimal threshold
            prec_curve, rec_curve, thresholds = precision_recall_curve(labels, preds)
            f1_scores = 2 * prec_curve * rec_curve / np.maximum(prec_curve + rec_curve, 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
            epoch_f1 = f1_scores[best_f1_idx]

            prec = precision_score(labels, preds > best_threshold, zero_division=0)
            rec = recall_score(labels, preds > best_threshold, zero_division=0)

            status = ""
            improved = False
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_auc = max(best_auc, auc)
                improved = True
                status = " * BEST"
            elif auc > best_auc and epoch_f1 >= best_f1 - 0.01:
                best_auc = auc
                improved = True
                status = " * BEST (AUC)"

            if improved:
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_auc": best_auc,
                    "best_f1": best_f1,
                    "threshold": float(best_threshold),
                }, output_dir / "pocket_detector_best.pt")
            else:
                patience_counter += 1

            eta_min = (args.epochs - epoch - 1) * epoch_time / 60
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"AUC: {auc:.3f} F1: {epoch_f1:.3f} Prec: {prec:.3f} Rec: {rec:.3f} thr={best_threshold:.2f} | "
                  f"lr={lr:.2e} nan={nan_count} | {epoch_time:.0f}s (ETA {eta_min:.0f}m){status}")

            with open(log_path, "a") as lf:
                lf.write(f"{epoch+1},{avg_loss:.6f},{auc:.4f},{prec:.4f},{rec:.4f},{epoch_f1:.4f},{best_threshold:.4f},{lr:.2e},{elapsed:.0f}\n")

            if patience_counter >= patience:
                print(f"\n  Early stopping: no improvement for {patience} eval rounds")
                break
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | lr={lr:.2e} nan={nan_count}")

    total_min = (time.time() - training_start) / 60
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE -- {total_min:.1f} minutes")
    print(f"Best AUC: {best_auc:.3f}  Best F1: {best_f1:.3f}")
    print(f"Model: {output_dir / 'pocket_detector_best.pt'}")
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pocket detector")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--use_cache", action="store_true",
                        help="Use precomputed .pt files from pocket_cache/ (run precompute_pocket_features.py first)")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint (path or 'auto' to use default location)")
    args = parser.parse_args()
    if args.resume == "auto":
        args.resume = str(Path(args.output_dir) / "pocket_detector_best.pt")
    train(args)


if __name__ == "__main__":
    main()

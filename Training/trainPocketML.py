#!/usr/bin/env python3
"""
Train P2Rank-style ML pocket detector for Druse.

Architecture:
  - Compute protein surface points (Connolly surface)
  - Extract per-point chemical features (hydrophobicity, charge, aromaticity, etc.)
  - GNN classifies each surface point as pocket/non-pocket
  - DBSCAN cluster pocket points → binding site predictions

Training data:
  - PDBbind v2020 refined set (known binding sites from ligand positions)
  - scPDB (optional, curated pocket database)

Usage:
  python train_pocket_detector.py --data_dir data/ --epochs 50
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
from torch_geometric.nn import GCNConv, global_mean_pool, radius_graph
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tqdm import tqdm


# ============================================================================
# Surface Point Features
# ============================================================================

# Per-surface-point features (11 dimensions):
#   [0-2]: surface normal (nx, ny, nz)
#   [3]:   distance to nearest atom
#   [4]:   hydrophobicity score of nearest residue
#   [5]:   charge of nearest atom
#   [6]:   is aromatic ring nearby (0/1)
#   [7]:   donor nearby (0/1)
#   [8]:   acceptor nearby (0/1)
#   [9]:   buriedness (26-direction ray casting)
#   [10]:  curvature (local surface shape)
SURFACE_FEAT_DIM = 11

HYDROPHOBICITY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLU": -3.5, "GLN": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest Point Sampling for uniform spatial coverage.
    Returns indices of n_samples points from the input array."""
    n = len(points)
    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = np.random.randint(n)
    min_dists = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, n_samples):
        dists = np.sum((points - points[selected[i - 1]]) ** 2, axis=1)
        np.minimum(min_dists, dists, out=min_dists)
        selected[i] = np.argmax(min_dists)
    return selected


def compute_surface_features(pdb_path: Path, probe_radius: float = 1.4,
                              grid_spacing: float = 1.0) -> tuple:
    """Compute surface points and features from a PDB file.
    Returns (positions [N,3], features [N,11], labels [N] for pocket detection)."""
    from Bio.PDB import PDBParser as BioPDBParser

    parser = BioPDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(pdb_path))
    except Exception:
        return None, None, None

    # Extract atom positions and properties
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
                    charge = 0.0
                    hydro = HYDROPHOBICITY.get(res_name, 0.0)
                    atoms.append({
                        "pos": pos,
                        "element": atom.element,
                        "residue": res_name,
                        "charge": charge,
                        "hydrophobicity": hydro,
                    })
        break  # first model only

    if not atoms:
        return None, None, None

    atom_pos = np.array([a["pos"] for a in atoms], dtype=np.float32)

    # Generate surface points using simple grid + distance filter
    bbox_min = atom_pos.min(axis=0) - 6.0
    bbox_max = atom_pos.max(axis=0) + 6.0

    # Grid points
    x = np.arange(bbox_min[0], bbox_max[0], grid_spacing)
    y = np.arange(bbox_min[1], bbox_max[1], grid_spacing)
    z = np.arange(bbox_min[2], bbox_max[2], grid_spacing)
    grid = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T

    # Filter to surface region (distance to nearest atom surface ~ probe_radius)
    from scipy.spatial import cKDTree
    tree = cKDTree(atom_pos)
    dists, indices = tree.query(grid)

    # VdW radii
    vdw = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}
    atom_radii = np.array([vdw.get(a["element"], 1.7) for a in atoms])
    surface_dist = dists - atom_radii[indices]

    # Keep points near the surface
    mask = (surface_dist >= -0.5) & (surface_dist <= probe_radius + 0.5)
    surface_points = grid[mask]
    nearest_idx = indices[mask]

    if len(surface_points) < 10:
        return None, None, None

    # Subsample to max 5000 points using farthest point sampling (FPS)
    # for uniform spatial coverage — random choice can delete pocket regions
    if len(surface_points) > 5000:
        choice = _farthest_point_sample(surface_points, 5000)
        surface_points = surface_points[choice]
        nearest_idx = nearest_idx[choice]

    # Compute features
    features = np.zeros((len(surface_points), SURFACE_FEAT_DIM), dtype=np.float32)

    for i in range(len(surface_points)):
        ni = nearest_idx[i]
        atom = atoms[ni]

        # Normal (approximate: point - nearest atom center)
        normal = surface_points[i] - atom_pos[ni]
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal /= norm_len
        features[i, 0:3] = normal

        # Distance to nearest atom
        features[i, 3] = norm_len

        # Hydrophobicity
        features[i, 4] = atom["hydrophobicity"] / 4.5  # normalize

        # Charge
        features[i, 5] = atom["charge"]

        # Element-based features
        elem = atom["element"]
        features[i, 6] = 1.0 if elem == "C" else 0.0  # aromatic proxy
        features[i, 7] = 1.0 if elem in ("N", "O") else 0.0  # donor
        features[i, 8] = 1.0 if elem in ("N", "O", "F") else 0.0  # acceptor

        # Buriedness (simplified: count nearby atoms)
        nearby = tree.query_ball_point(surface_points[i], 6.0)
        features[i, 9] = min(len(nearby) / 20.0, 1.0)

        # Curvature (simplified: variance of normals in neighborhood)
        features[i, 10] = 0.5  # placeholder

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

        edge_index = radius_graph(pos_t, r=4.0, max_num_neighbors=16)

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

class PocketGNN(nn.Module):
    """GNN for surface point classification (pocket vs non-pocket)."""

    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, data.edge_index)
            x = norm(x)
            x = F.relu(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def pyg_collate(batch):
    return batch[0] if len(batch) == 1 else batch

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

    # Use multiple CPU workers to parallelize surface feature computation
    # (the bottleneck is CPU-bound cKDTree + grid generation in __getitem__)
    n_workers = min(4, os.cpu_count() or 1)
    print(f"DataLoader workers: {n_workers}")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=n_workers,
                              collate_fn=pyg_collate, persistent_workers=n_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=n_workers,
                            collate_fn=pyg_collate, persistent_workers=n_workers > 0)

    model = PocketGNN(hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_path = output_dir / "pocket_training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_auc,val_precision,val_recall,lr,time_sec\n")
    print(f"Training log: {log_path}")
    print(f"Monitor with: tail -f {log_path}\n")

    training_start = time.time()

    for epoch in range(args.epochs):
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
                # Weighted BCE (pockets are rare → upweight positives)
                n_pos = max((data.y == 1).sum().float(), 1)
                n_neg = max((data.y == 0).sum().float(), 1)
                pos_weight = n_neg / n_pos
                weight = data.y * pos_weight + (1 - data.y)
                loss = F.binary_cross_entropy(pred, data.y, weight=weight)

                if torch.isnan(loss):
                    nan_count += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            except Exception:
                continue

        scheduler.step()
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
            prec = precision_score(labels, preds > 0.5, zero_division=0)
            rec = recall_score(labels, preds > 0.5, zero_division=0)

            status = ""
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), output_dir / "pocket_detector_best.pt")
                status = " * BEST"

            eta_min = (args.epochs - epoch - 1) * epoch_time / 60
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"AUC: {auc:.3f} Prec: {prec:.3f} Rec: {rec:.3f} | "
                  f"lr={lr:.2e} nan={nan_count} | {epoch_time:.0f}s (ETA {eta_min:.0f}m){status}")

            with open(log_path, "a") as lf:
                lf.write(f"{epoch+1},{avg_loss:.6f},{auc:.4f},{prec:.4f},{rec:.4f},{lr:.2e},{elapsed:.0f}\n")
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | lr={lr:.2e} nan={nan_count}")

    total_min = (time.time() - training_start) / 60
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE -- {total_min:.1f} minutes")
    print(f"Best AUC: {best_auc:.3f}")
    print(f"Model: {output_dir / 'pocket_detector_best.pt'}")
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pocket detector")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cache", action="store_true",
                        help="Use precomputed .pt files from pocket_cache/ (run precompute_pocket_features.py first)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

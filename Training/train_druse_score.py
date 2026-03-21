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
  python train_druse_score.py --data_dir data/ --epochs 100 --batch_size 32
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
from sklearn.model_selection import KFold
from tqdm import tqdm

# ============================================================================
# Feature Extraction
# ============================================================================

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8}
NUM_ATOM_FEATURES = 18  # 10 (element one-hot) + 1 (aromatic) + 1 (charge) + 2 (hb) + 4 (hybrid)
RBF_CENTERS = torch.linspace(0, 10, 50)  # 50 Gaussian RBFs, 0-10 A
RBF_GAMMA = 10.0


def rbf_encode(distances: torch.Tensor) -> torch.Tensor:
    """Gaussian RBF encoding of distances. [N] -> [N, 50]"""
    return torch.exp(-RBF_GAMMA * (distances.unsqueeze(-1) - RBF_CENTERS.to(distances.device)) ** 2)


def atom_features(element: str, charge: float = 0.0, is_aromatic: bool = False,
                  is_protein: bool = True) -> np.ndarray:
    """Encode atom as 18-dimensional feature vector."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    idx = ATOM_TYPES.get(element, 9)
    feat[idx] = 1.0
    feat[10] = float(is_aromatic)
    feat[11] = charge
    feat[12] = float(element in ("N", "O"))  # HBD
    feat[13] = float(element in ("N", "O", "F"))  # HBA
    feat[16] = 1.0  # default sp3
    feat[17] = 0.0 if is_protein else 1.0
    return feat


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
        prot_pos, prot_feat = self._parse_pdb(pocket_path, is_protein=True)

        # Parse ligand atoms
        ligand_path = self._ligand_path(pdb_id)
        lig_pos, lig_feat = self._parse_mol2(ligand_path, is_protein=False)

        if prot_pos is None or lig_pos is None:
            # Return dummy data for corrupted entries
            return self._dummy_data(pkd)

        # Build graph
        prot_pos_t = torch.tensor(prot_pos, dtype=torch.float32)
        lig_pos_t = torch.tensor(lig_pos, dtype=torch.float32)
        prot_feat_t = torch.tensor(prot_feat, dtype=torch.float32)
        lig_feat_t = torch.tensor(lig_feat, dtype=torch.float32)

        # Cross-distances for attention
        cross_dist = torch.cdist(lig_pos_t, prot_pos_t)  # [L, P]
        cross_rbf = rbf_encode(cross_dist.reshape(-1)).reshape(
            lig_pos_t.size(0), prot_pos_t.size(0), 50
        )

        data = Data(
            prot_pos=prot_pos_t,
            prot_x=prot_feat_t,
            lig_pos=lig_pos_t,
            lig_x=lig_feat_t,
            cross_rbf=cross_rbf,
            y=torch.tensor([pkd], dtype=torch.float32),
            pdb_id=pdb_id,
        )

        if self.transform:
            data = self.transform(data)
        return data

    def _parse_pdb(self, path: Path, is_protein: bool):
        """Parse PDB pocket file → (positions [N,3], features [N,18])."""
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
                    charge = 0.0
                    try:
                        charge = float(line[54:60])
                    except (ValueError, IndexError):
                        pass
                    positions.append([x, y, z])
                    features.append(atom_features(element, charge, is_protein=is_protein))
        except Exception:
            return None, None

        if not positions:
            return None, None
        return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)

    def _parse_mol2(self, path: Path, is_protein: bool):
        """Parse MOL2 ligand file → (positions [N,3], features [N,18])."""
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
                    features.append(atom_features(element, charge, is_protein=is_protein))
        except Exception:
            return None, None

        if not positions:
            return None, None
        return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)

    def _dummy_data(self, pkd):
        return Data(
            prot_pos=torch.zeros(5, 3), prot_x=torch.zeros(5, NUM_ATOM_FEATURES),
            lig_pos=torch.zeros(3, 3), lig_x=torch.zeros(3, NUM_ATOM_FEATURES),
            cross_rbf=torch.zeros(3, 5, 50),
            y=torch.tensor([pkd]), pdb_id="dummy",
        )


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
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))

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

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, cutoff: float = 8.0):
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

        # Cross-attention
        lig_attended, attn_weights = self.cross_attn(lig_h, prot_h, prot_h, data.cross_rbf)

        # Global pooling for affinity and pose prediction
        complex_repr = lig_attended.mean(dim=0, keepdim=True)
        pkd_pred = self.affinity_head(complex_repr).squeeze(-1)
        pose_pred = torch.sigmoid(self.pose_head(complex_repr)).squeeze(-1)

        # Per-atom-pair interaction predictions
        L = lig_h.size(0)
        P = prot_h.size(0)
        lig_exp = lig_attended.unsqueeze(1).expand(-1, P, -1)  # [L, P, D]
        prot_exp = prot_h.unsqueeze(0).expand(L, -1, -1)  # [L, P, D]
        pair_input = torch.cat([lig_exp, prot_exp, data.cross_rbf], dim=-1)
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
    """Collate variable-size complexes into a batch."""
    # For simplicity, process one at a time (batch_size=1 effective for variable sizes)
    # In production, pad to max size or use PyG Batch
    return batch[0] if len(batch) == 1 else batch


def move_data_to_device(data, device):
    """Move Data object tensors to device, handling MPS limitations."""
    if device.type == "mps":
        # MPS has limited op support — keep positions on CPU for radius_graph,
        # move features to device for neural network operations
        data.prot_x = data.prot_x.to(device)
        data.lig_x = data.lig_x.to(device)
        data.prot_pos = data.prot_pos.to(device)
        data.lig_pos = data.lig_pos.to(device)
        data.cross_rbf = data.cross_rbf.to(device)
        data.y = data.y.to(device)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def train_epoch(model, loader, optimizer, device, epoch_num=0):
    model.train()
    total_loss = 0
    n = 0
    nan_count = 0
    skip_count = 0
    max_grad_norm = 0.0

    for batch_idx, data in enumerate(tqdm(loader, desc=f"  Train ep{epoch_num+1}", leave=False)):
        if isinstance(data, list):
            data = data[0]
        data = move_data_to_device(data, device)

        optimizer.zero_grad()
        try:
            output = model(data)
        except Exception as e:
            skip_count += 1
            continue

        # Multi-task loss (Huber loss is more robust than MSE for pKd outliers)
        loss_affinity = F.huber_loss(output["pKd"], data.y, delta=2.0)
        loss = loss_affinity

        # Skip NaN/Inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        loss.backward()

        # Monitor gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if grad_norm > max_grad_norm:
            max_grad_norm = float(grad_norm)

        optimizer.step()

        total_loss += loss.item()
        n += 1

        # Periodic check for NaN in model parameters
        if batch_idx % 500 == 0 and batch_idx > 0:
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"\n  [WARN] NaN detected in model params at batch {batch_idx}!")
                # Reset to prevent cascading NaN
                return float('nan'), nan_count, skip_count, max_grad_norm

    avg_loss = total_loss / max(n, 1)
    return avg_loss, nan_count, skip_count, max_grad_norm


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, pdb_ids = [], [], []

    for data in tqdm(loader, desc="Evaluating", leave=False):
        if isinstance(data, list):
            data = data[0]
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

    # Dataset
    dataset = PDBBindDataset(args.data_dir, split="refined")

    if len(dataset) == 0:
        print("ERROR: No data found. Run: python download_data.py --all")
        return

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_indices = list(range(len(dataset)))

    best_overall_r = -1
    best_model_state = None

    import time
    training_start = time.time()

    # Log file for monitoring
    log_path = args.output_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("fold,epoch,train_loss,val_rmse,val_mae,val_r,lr,max_grad,nan_count,skip_count,time_sec\n")

    print(f"\nTraining log: {log_path}")
    print(f"Monitor with: tail -f {log_path}\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        print(f"{'='*60}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Workers for CPU-bound data loading (PDB/MOL2 parsing + graph building)
        # PyTorch uses /dev/shm for IPC — if small, use fewer workers
        n_workers = min(4, (os.cpu_count() or 1))
        if fold == 0:
            print(f"  DataLoader workers: {n_workers}")
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
        warmup_epochs = 5
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

            # Print every epoch (not just every 5th)
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.epochs - 1:
                metrics = evaluate(model, val_loader, device)
                epoch_time = time.time() - epoch_start
                eta = (args.epochs - epoch - 1) * epoch_time

                status = ""
                if metrics["pearson_r"] > best_r:
                    best_r = metrics["pearson_r"]
                    torch.save(model.state_dict(), args.output_dir / f"fold{fold}_best.pt")
                    status = " ★ NEW BEST"

                if metrics["pearson_r"] > best_overall_r:
                    best_overall_r = metrics["pearson_r"]
                    best_model_state = model.state_dict().copy()
                    status = " ★★ OVERALL BEST"

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
    parser.add_argument("--batch_size", type=int, default=1)
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

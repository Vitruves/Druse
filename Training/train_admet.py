#!/usr/bin/env python3
"""
Train ADMET property prediction models for Druse.

Models (each trained independently):
  - LogP (Wildman-Crippen via RDKit, or ML refinement)
  - LogD at pH 7.4
  - Aqueous solubility (log S)
  - CYP2D6 inhibition (binary classification)
  - CYP3A4 inhibition (binary classification)
  - hERG liability (binary classification)
  - BBB permeability (binary classification)
  - Metabolic stability (binary classification)

Input: Morgan fingerprint (2048-bit ECFP4) from SMILES.
Architecture: Simple feed-forward network (fingerprint → hidden → output).

Data sources:
  - TDC (Therapeutics Data Commons) benchmarks
  - ChEMBL curated datasets
  - MoleculeNet benchmarks

Usage:
  python train_admet.py --property logp --epochs 50
  python train_admet.py --all --epochs 50
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, r2_score
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit required: pip install rdkit")
    exit(1)

# ============================================================================
# Fingerprint Generation
# ============================================================================

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """SMILES → Morgan fingerprint (ECFP4) as float array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ============================================================================
# Dataset
# ============================================================================

class ADMETDataset(Dataset):
    """ADMET property dataset from SMILES + labels."""

    def __init__(self, smiles: list, labels: np.ndarray, n_bits: int = 2048):
        self.fingerprints = []
        self.labels = []
        for smi, label in zip(smiles, labels):
            fp = smiles_to_fingerprint(smi, n_bits=n_bits)
            if fp is not None and not np.isnan(label):
                self.fingerprints.append(fp)
                self.labels.append(label)
        self.fingerprints = np.array(self.fingerprints)
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"  ADMETDataset: {len(self)} valid molecules")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.fingerprints[idx]), torch.tensor(self.labels[idx])


# ============================================================================
# Model
# ============================================================================

class ADMETModel(nn.Module):
    """Feed-forward network for ADMET prediction from fingerprints.

    Architecture matches export_coreml.py ADMETModelForExport exactly
    (no Dropout in nn.Sequential so state_dict keys align).
    Dropout is applied in forward() instead.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 1,
                 is_classifier: bool = False):
        super().__init__()
        self.is_classifier = is_classifier
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        h = self.net[0](x)       # Linear(input, hidden)
        h = self.net[1](h)       # ReLU
        h = self.drop1(h)
        h = self.net[2](h)       # Linear(hidden, hidden//2)
        h = self.net[3](h)       # ReLU
        h = self.drop2(h)
        out = self.net[4](h)     # Linear(hidden//2, output)
        if self.is_classifier:
            out = torch.sigmoid(out)
        return out.squeeze(-1)


# ============================================================================
# Data Loading (from TDC or custom CSV)
# ============================================================================

PROPERTY_CONFIG = {
    "logp": {
        "type": "regression",
        "tdc_name": "Lipophilicity_AstraZeneca",
        "csv_columns": ("Drug", "Y"),
        "description": "Lipophilicity (LogP/LogD)",
    },
    "solubility": {
        "type": "regression",
        "tdc_name": "Solubility_AqSolDB",
        "csv_columns": ("Drug", "Y"),
        "description": "Aqueous solubility (log S)",
    },
    "cyp2d6": {
        "type": "classification",
        "tdc_name": "CYP2D6_Veith",
        "csv_columns": ("Drug", "Y"),
        "description": "CYP2D6 inhibition",
    },
    "cyp3a4": {
        "type": "classification",
        "tdc_name": "CYP3A4_Veith",
        "csv_columns": ("Drug", "Y"),
        "description": "CYP3A4 inhibition",
    },
    "herg": {
        "type": "classification",
        "tdc_name": "hERG",
        "csv_columns": ("Drug", "Y"),
        "description": "hERG channel liability",
    },
    "bbb": {
        "type": "classification",
        "tdc_name": "BBB_Martins",
        "csv_columns": ("Drug", "Y"),
        "description": "Blood-brain barrier permeability",
    },
    "metabolic_stability": {
        "type": "classification",
        "tdc_name": "HLM",
        "csv_columns": ("Drug", "Y"),
        "description": "Human liver microsomal stability",
    },
    "logd": {
        "type": "regression",
        "tdc_name": None,  # Custom dataset needed
        "csv_columns": ("smiles", "logd74"),
        "description": "LogD at pH 7.4",
    },
}


def load_tdc_dataset(property_name: str, data_dir: Path):
    """Load dataset from TDC or local CSV."""
    config = PROPERTY_CONFIG[property_name]

    # Try local CSV first
    csv_path = data_dir / f"admet_{property_name}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        smiles_col, label_col = config["csv_columns"]
        return df[smiles_col].tolist(), df[label_col].values

    # Try TDC
    if config["tdc_name"]:
        try:
            from tdc.single_pred import ADME, Tox
            if property_name == "herg":
                data = Tox(name=config["tdc_name"])
            else:
                data = ADME(name=config["tdc_name"])
            df = data.get_data()
            return df["Drug"].tolist(), df["Y"].values
        except ImportError:
            print("  TDC not installed. Install: pip install PyTDC")
        except Exception as e:
            print(f"  TDC load failed: {e}")

    print(f"  No data found for {property_name}. Create: {csv_path}")
    return [], np.array([])


# ============================================================================
# Training
# ============================================================================

def train_property(property_name: str, args):
    config = PROPERTY_CONFIG[property_name]
    is_classifier = config["type"] == "classification"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"Training: {config['description']} ({property_name})")
    print(f"Type: {'classification' if is_classifier else 'regression'}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load data
    smiles, labels = load_tdc_dataset(property_name, args.data_dir)
    if len(smiles) == 0:
        print("  SKIPPED: no data")
        return

    # Create dataset
    dataset = ADMETDataset(smiles, labels)
    if len(dataset) < 100:
        print("  SKIPPED: too few valid molecules")
        return

    # Train/val/test split (70/15/15)
    n = len(dataset)
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Split: {n_train} train / {n_val} val / {n_test} test")

    # Class weighting for imbalanced classification
    pos_weight = None
    if is_classifier:
        train_labels = np.array([dataset.labels[i] for i in train_set.indices])
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw = n_neg / n_pos
            pos_weight = torch.tensor([pw], device=device)
            print(f"  Class balance: {int(n_neg)} neg / {int(n_pos)} pos → pos_weight={pw:.2f}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=2, pin_memory=(device.type == "cuda"))

    # Model
    model = ADMETModel(is_classifier=is_classifier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function (with class weighting for classifiers)
    if is_classifier:
        loss_fn = nn.BCELoss(weight=None)  # model.forward() applies sigmoid
    else:
        loss_fn = nn.MSELoss()

    best_metric = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for fp, label in train_loader:
            fp, label = fp.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(fp)  # includes dropout + sigmoid (for classifiers)
            if is_classifier and pos_weight is not None:
                # Manual weighting: scale positive-class loss
                weight = torch.where(label == 1, pos_weight.squeeze(), torch.ones_like(label))
                loss = nn.functional.binary_cross_entropy(pred, label, weight=weight)
            else:
                loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Evaluate on val every 5 epochs (or last)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metric = _evaluate(model, val_loader, device, is_classifier)

            if is_classifier:
                print(f"  Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {metric:.3f}")
            else:
                print(f"  Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val R²: {metric:.3f}")

            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} eval rounds)")
                break

    # Evaluate on held-out test set
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
        test_metric = _evaluate(model, test_loader, device, is_classifier)
        metric_name = "AUC" if is_classifier else "R²"
        print(f"\n  Test {metric_name}: {test_metric:.3f}")

        out_path = args.output_dir / f"admet_{property_name}.pt"
        torch.save(best_state, out_path)
        print(f"  Saved: {out_path} (val {metric_name}: {best_metric:.3f}, test {metric_name}: {test_metric:.3f})")

    return best_state


def _evaluate(model, loader, device, is_classifier: bool) -> float:
    """Evaluate model on a DataLoader, return AUC (classifier) or R² (regression)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for fp, label in loader:
            fp = fp.to(device)
            pred = model(fp).cpu().numpy()
            preds.extend(pred)
            targets.extend(label.numpy())
    preds = np.array(preds)
    targets = np.array(targets)

    if is_classifier:
        return roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0
    else:
        return r2_score(targets, preds) if len(targets) > 2 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train ADMET models")
    parser.add_argument("--property", type=str, default=None,
                        choices=list(PROPERTY_CONFIG.keys()),
                        help="Which property to train")
    parser.add_argument("--all", action="store_true", help="Train all properties")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6,
                        help="Early stopping patience (in eval rounds)")
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        for prop in PROPERTY_CONFIG:
            train_property(prop, args)
    elif args.property:
        train_property(args.property, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

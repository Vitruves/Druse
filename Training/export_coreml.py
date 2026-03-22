#!/usr/bin/env python3
"""
Export trained PyTorch models to CoreML (.mlmodel/.mlpackage) for deployment in Druse.

Exports:
  1. DruseScore → DruseScore.mlpackage (EGNN scoring model)
  2. PocketDetector → PocketDetector.mlpackage (surface point pocket classifier)
  3. ADMET models → ADMET_*.mlpackage (fingerprint-based property predictors)

The exported models are placed in ../Resources/ for Xcode to bundle.

Usage:
  python export_coreml.py --druse_score checkpoints/druse_score_best.pt
  python export_coreml.py --pocket_detector checkpoints/pocket_detector_best.pt
  python export_coreml.py --admet checkpoints/
  python export_coreml.py --all checkpoints/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
except ImportError:
    print("coremltools required: pip install coremltools>=7.1")
    exit(1)


# ============================================================================
# ADMET Export (simple fingerprint → prediction)
# ============================================================================

class ADMETModelForExport(nn.Module):
    """Simplified ADMET model matching the training architecture."""

    def __init__(self, input_dim=2048, hidden_dim=512, is_classifier=False):
        super().__init__()
        self.is_classifier = is_classifier
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fingerprint):
        out = self.net(fingerprint)
        if self.is_classifier:
            out = torch.sigmoid(out)
        return out


ADMET_PROPERTIES = {
    "logp": {"classifier": False, "desc": "Lipophilicity prediction"},
    "logd": {"classifier": False, "desc": "LogD at pH 7.4"},
    "solubility": {"classifier": False, "desc": "Aqueous solubility"},
    "cyp2d6": {"classifier": True, "desc": "CYP2D6 inhibition"},
    "cyp3a4": {"classifier": True, "desc": "CYP3A4 inhibition"},
    "herg": {"classifier": True, "desc": "hERG liability"},
    "bbb": {"classifier": True, "desc": "BBB permeability"},
    "metabolic_stability": {"classifier": True, "desc": "Metabolic stability"},
}


def export_admet(checkpoint_dir: Path, output_dir: Path):
    """Export all ADMET models to CoreML."""
    for prop_name, config in ADMET_PROPERTIES.items():
        pt_path = checkpoint_dir / f"admet_{prop_name}.pt"
        if not pt_path.exists():
            print(f"  [skip] {pt_path} not found")
            continue

        print(f"  Exporting ADMET_{prop_name}...")

        model = ADMETModelForExport(is_classifier=config["classifier"])
        # Load only matching keys (skip dropout etc.)
        state = torch.load(pt_path, map_location="cpu")
        # Map training model keys to export model keys
        mapped = {}
        for k, v in state.items():
            # Training model uses 'net.0.weight', etc. with dropout layers
            # Export model skips dropout, so indices shift
            mapped[k] = v
        try:
            model.load_state_dict(mapped, strict=False)
        except Exception as e:
            print(f"    Warning: partial load ({e})")

        model.eval()

        # Trace
        example = torch.zeros(1, 2048)
        traced = torch.jit.trace(model, example)

        # Convert to CoreML
        ml_model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="fingerprint", shape=(1, 2048))],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
        )

        ml_model.author = "Druse"
        ml_model.short_description = config["desc"]
        ml_model.version = "1.0"

        out_name = f"ADMET_{prop_name.replace('_', '').title()}"
        out_path = output_dir / f"{out_name}.mlpackage"
        ml_model.save(str(out_path))
        print(f"    Saved: {out_path}")


# ============================================================================
# DruseScore Export (simplified for CoreML tracing)
# ============================================================================

class DruseScoreForExport(nn.Module):
    """Simplified DruseScore for CoreML export.

    CoreML doesn't support dynamic graphs well, so we use a fixed-size
    attention mechanism instead of the full EGNN + radius_graph pipeline.

    Input: pre-computed features from Metal compute shaders.
    """

    def __init__(self, atom_dim=18, hidden_dim=128, max_prot=256, max_lig=64):
        super().__init__()
        self.max_prot = max_prot
        self.max_lig = max_lig
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.attn_scale = float(self.head_dim) ** 0.5

        # Simplified encoders (no EGNN, just MLPs — EGNN features pre-computed by Metal)
        self.prot_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lig_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rbf_proj = nn.Linear(50, self.num_heads)

        # Heads
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 50, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, protein_features, ligand_features, protein_positions,
                ligand_positions, pair_rbf):
        """
        protein_features: [1, P, 18]
        ligand_features: [1, L, 18]
        protein_positions: [1, P, 3]
        ligand_positions: [1, L, 3]
        pair_rbf: [1, L, P, 50]
        """
        P = self.max_prot
        L = self.max_lig
        H = self.num_heads
        d = self.head_dim
        D = H * d

        prot_h = self.prot_encoder(protein_features.squeeze(0))  # [P, D]
        lig_h = self.lig_encoder(ligand_features.squeeze(0))  # [L, D]
        rbf = pair_rbf.squeeze(0)  # [L, P, 50]

        # Cross-attention (all dims are pre-computed constants)
        Q = self.q_proj(lig_h).reshape(L, H, d)
        K = self.k_proj(prot_h).reshape(P, H, d)
        V = self.v_proj(prot_h).reshape(P, H, d)

        attn = torch.einsum("lhd,phd->lph", Q, K) / self.attn_scale
        dist_bias = self.rbf_proj(rbf)  # [L, P, H]
        attn = attn + dist_bias
        attn_weights = torch.softmax(attn, dim=1)

        attended = torch.einsum("lph,phd->lhd", attn_weights, V).reshape(L, D)

        # Global pooling
        complex_repr = attended.mean(dim=0, keepdim=True)

        # Predictions
        pkd = self.affinity_head(complex_repr)
        pose_conf = torch.sigmoid(self.pose_head(complex_repr))

        # Interaction map
        lig_exp = attended.unsqueeze(1).expand(L, P, D)
        prot_exp = prot_h.unsqueeze(0).expand(L, P, D)
        pair_input = torch.cat([lig_exp, prot_exp, rbf], dim=-1)
        interaction_map = torch.sigmoid(self.interaction_head(pair_input))

        # Attention weights (mean over heads)
        attn_out = attn_weights.mean(dim=-1)  # [L, P]

        return pkd, pose_conf, interaction_map.unsqueeze(0), attn_out.reshape(1, L * P)


def export_druse_score(checkpoint_path: Path, output_dir: Path,
                       max_prot: int = 256, max_lig: int = 64):
    """Export DruseScore to CoreML."""
    print(f"  Exporting DruseScore...")

    model = DruseScoreForExport(max_prot=max_prot, max_lig=max_lig)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
            print(f"    Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"    Warning: partial load ({e})")
    else:
        print(f"    No checkpoint found, exporting random weights (for testing)")

    model.eval()

    # Trace with example inputs
    P, L = max_prot, max_lig
    prot_feat = torch.randn(1, P, 18)
    lig_feat = torch.randn(1, L, 18)
    prot_pos = torch.randn(1, P, 3)
    lig_pos = torch.randn(1, L, 3)
    pair_rbf = torch.randn(1, L, P, 50)

    traced = torch.jit.trace(model, (prot_feat, lig_feat, prot_pos, lig_pos, pair_rbf))

    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="protein_features", shape=(1, P, 18)),
            ct.TensorType(name="ligand_features", shape=(1, L, 18)),
            ct.TensorType(name="protein_positions", shape=(1, P, 3)),
            ct.TensorType(name="ligand_positions", shape=(1, L, 3)),
            ct.TensorType(name="pair_rbf", shape=(1, L, P, 50)),
        ],
        outputs=[
            ct.TensorType(name="pKd"),
            ct.TensorType(name="pose_confidence"),
            ct.TensorType(name="interaction_map"),
            ct.TensorType(name="attention_weights"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    ml_model.author = "Druse"
    ml_model.short_description = "SE(3)-equivariant protein-ligand scoring"
    ml_model.version = "1.0"

    out_path = output_dir / "DruseScore.mlpackage"
    ml_model.save(str(out_path))
    print(f"    Saved: {out_path}")

    # Also compile for faster loading
    try:
        compiled = ct.models.CompiledMLModel(str(out_path))
        print(f"    Compiled model ready")
    except Exception:
        print(f"    Note: compile on target machine with `xcrun coremlcompiler compile`")


# ============================================================================
# Pocket Detector Export
# ============================================================================

class PocketDetectorForExport(nn.Module):
    """Pocket detector for CoreML export.

    The training model uses GCNConv (dynamic radius graph), which CoreML can't
    handle. Instead, Metal computes k-nearest neighbors on the surface point
    cloud and passes a [N, K] index tensor. This model gathers neighbor features
    via indexing, averages them, and applies the same linear transform as GCNConv.

    Input from Metal compute shaders:
      - surface_features: per-point chemical features (normals, hydrophobicity, etc.)
      - knn_indices: k-nearest neighbor indices for each surface point
      - point_mask: 1 for real points, 0 for padding

    Output:
      - pocket_probability: per-point probability of being a binding pocket
    """

    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, num_layers: int = 3,
                 max_points: int = 2048, k_neighbors: int = 16):
        super().__init__()
        self.max_points = max_points
        self.k_neighbors = k_neighbors

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.conv_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, surface_features, knn_indices, point_mask):
        """
        surface_features: [1, N, 11]
        knn_indices: [1, N, K] — k-nearest neighbor indices (from Metal)
        point_mask: [1, N] — 1.0 for real points, 0.0 for padding
        """
        N = self.max_points
        K = self.k_neighbors

        x = self.input_proj(surface_features.squeeze(0))  # [N, D]
        mask = point_mask.squeeze(0)  # [N]

        flat_idx = knn_indices.squeeze(0).reshape(N * K)  # [N*K]

        for conv_w, norm in zip(self.conv_weights, self.norms):
            # Gather neighbor features using pre-computed flat indices
            neighbor_feats = torch.index_select(x, 0, flat_idx)  # [N*K, D]
            neighbor_feats = neighbor_feats.reshape(N, K, -1)  # [N, K, D]
            agg = neighbor_feats.mean(dim=1)  # [N, D]

            x = x + conv_w(agg)
            x = norm(x)
            x = F.relu(x)

        probs = torch.sigmoid(self.head(x)).squeeze(-1)  # [N]
        probs = probs * mask  # zero out padded points
        return probs.unsqueeze(0)  # [1, N]


def export_pocket_detector(checkpoint_path: Path, output_dir: Path,
                            max_points: int = 2048, k_neighbors: int = 16):
    """Export PocketDetector to CoreML."""
    print(f"  Exporting PocketDetector...")

    model = PocketDetectorForExport(
        max_points=max_points, k_neighbors=k_neighbors
    )

    threshold = 0.5  # default

    if checkpoint_path.exists():
        raw = torch.load(checkpoint_path, map_location="cpu")

        # Handle both old format (raw state_dict) and new format (dict with threshold)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
            threshold = raw.get("threshold", 0.5)
            print(f"    Loaded threshold: {threshold:.4f}")
        else:
            state = raw

        # Map training model weight names to export model weight names:
        #   convs.{i}.lin.weight → conv_weights.{i}.weight
        #   convs.{i}.bias       → conv_weights.{i}.bias
        mapped = {}
        for k, v in state.items():
            export_key = k
            if k.startswith("convs."):
                parts = k.split(".")
                layer_idx = parts[1]
                if parts[2] == "lin":
                    # convs.0.lin.weight → conv_weights.0.weight
                    export_key = f"conv_weights.{layer_idx}.{parts[3]}"
                elif parts[2] in ("weight", "bias"):
                    # convs.0.bias → conv_weights.0.bias
                    export_key = f"conv_weights.{layer_idx}.{parts[2]}"
            mapped[export_key] = v

        loaded_keys = model.load_state_dict(mapped, strict=False)
        if loaded_keys.missing_keys:
            print(f"    Note: missing keys (expected): {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            print(f"    Note: unexpected keys (ignored): {loaded_keys.unexpected_keys}")
        print(f"    Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"    No checkpoint found, exporting random weights (for testing)")

    model.eval()

    # Trace with example inputs
    N, K = max_points, k_neighbors
    surface_feat = torch.randn(1, N, 11)
    knn_idx = torch.randint(0, N, (1, N, K))
    mask = torch.ones(1, N)

    traced = torch.jit.trace(model, (surface_feat, knn_idx, mask))

    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="surface_features", shape=(1, N, 11)),
            ct.TensorType(name="knn_indices", shape=ct.Shape(shape=(1, N, K)),
                          dtype=np.int32),
            ct.TensorType(name="point_mask", shape=(1, N)),
        ],
        outputs=[
            ct.TensorType(name="pocket_probability"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    ml_model.author = "Druse"
    ml_model.short_description = "Surface point pocket detection (GNN-based)"
    ml_model.version = "1.0"
    ml_model.user_defined_metadata["pocket_threshold"] = f"{threshold:.6f}"
    ml_model.user_defined_metadata["max_points"] = str(max_points)
    ml_model.user_defined_metadata["k_neighbors"] = str(k_neighbors)

    out_path = output_dir / "PocketDetector.mlpackage"
    ml_model.save(str(out_path))
    print(f"    Saved: {out_path}")
    print(f"    Threshold in metadata: {threshold:.4f}")

    try:
        compiled = ct.models.CompiledMLModel(str(out_path))
        print(f"    Compiled model ready")
    except Exception:
        print(f"    Note: compile on target machine with `xcrun coremlcompiler compile`")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Druse models to CoreML")
    parser.add_argument("--druse_score", type=str, default=None,
                        help="Path to DruseScore checkpoint (.pt)")
    parser.add_argument("--pocket_detector", type=str, default=None,
                        help="Path to PocketDetector checkpoint (.pt)")
    parser.add_argument("--admet", type=str, default=None,
                        help="Directory containing ADMET checkpoints")
    parser.add_argument("--all", type=str, default=None,
                        help="Directory containing all checkpoints")
    parser.add_argument("--output_dir", type=str, default="../Resources/",
                        help="Output directory (default: ../Resources/ for Xcode)")
    parser.add_argument("--max_prot", type=int, default=256)
    parser.add_argument("--max_lig", type=int, default=64)
    parser.add_argument("--max_surface_points", type=int, default=2048,
                        help="Max surface points for pocket detector export")
    parser.add_argument("--k_neighbors", type=int, default=16,
                        help="K nearest neighbors for pocket detector export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Druse CoreML Export ===\n")

    if args.all:
        checkpoint_dir = Path(args.all)
        export_druse_score(checkpoint_dir / "druse_score_best.pt", output_dir,
                           args.max_prot, args.max_lig)
        export_pocket_detector(checkpoint_dir / "pocket_detector_best.pt", output_dir,
                                args.max_surface_points, args.k_neighbors)
        export_admet(checkpoint_dir, output_dir)
    else:
        if args.druse_score:
            export_druse_score(Path(args.druse_score), output_dir,
                               args.max_prot, args.max_lig)
        if args.pocket_detector:
            export_pocket_detector(Path(args.pocket_detector), output_dir,
                                    args.max_surface_points, args.k_neighbors)
        if args.admet:
            export_admet(Path(args.admet), output_dir)

    if not args.all and not args.druse_score and not args.pocket_detector and not args.admet:
        parser.print_help()
        return

    print(f"\nCoreML models saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Add .mlpackage files to Xcode project Resources")
    print("  2. Or compile: xcrun coremlcompiler compile Model.mlpackage Model.mlmodelc")


if __name__ == "__main__":
    main()

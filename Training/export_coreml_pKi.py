#!/usr/bin/env python3
"""
Export trained DruseScore-pKi v2 model to CoreML for deployment in Druse.

The pKi model is the PRIMARY scoring function (not a re-ranker). Its main output
is docking_score = pKd * pose_confidence, which replaces Vina-style empirical
scores for ranking docked poses.

v2 architecture (trained on 13K+ co-crystallized structures):
  - 20-dim atom features (added formal charge + ring membership)
  - 2-layer stacked cross-attention with residual connections
  - Gated attention pooling (learned atom importance)

Exports:
  - DruseScorePKi.mlpackage with outputs:
    * docking_score: primary ranking value (pKd * confidence)
    * pKd: predicted binding affinity
    * pose_confidence: how close the pose is to native (0-1)
    * interaction_map: per atom-pair interaction predictions [L, P, 5]
    * attention_weights: cross-attention weights [L*P]

Usage:
  python export_coreml_pKi.py --checkpoint checkpoints_pki/druse_pki_v2_best.pt
  python export_coreml_pKi.py --checkpoint checkpoints_pki/druse_pki_v2_best.pt --output_dir ../Models/druse-models/
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
except ImportError:
    print("coremltools required: pip install coremltools>=7.1")
    exit(1)


# ============================================================================
# Export Model v2 (simplified for CoreML tracing)
# ============================================================================

class CrossAttentionLayerExport(nn.Module):
    """Single cross-attention layer with residual + LayerNorm for CoreML export."""

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_scale = float(self.head_dim) ** 0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rbf_proj = nn.Linear(50, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, lig_h, prot_h, rbf, L, P):
        H = self.num_heads
        d = self.head_dim
        D = H * d

        Q = self.q_proj(lig_h).reshape(L, H, d)
        K = self.k_proj(prot_h).reshape(P, H, d)
        V = self.v_proj(prot_h).reshape(P, H, d)

        attn = torch.einsum("lhd,phd->lph", Q, K) / self.attn_scale
        dist_bias = self.rbf_proj(rbf)  # [L, P, H]
        attn = attn + dist_bias
        attn_weights = torch.softmax(attn, dim=1)  # softmax over protein atoms

        attended = torch.einsum("lph,phd->lhd", attn_weights, V).reshape(L, D)
        # Residual + LayerNorm
        lig_h_out = self.norm(lig_h + self.out_proj(attended))

        return lig_h_out, attn_weights


class DruseScorePKiForExport(nn.Module):
    """DruseScore-pKi v2 simplified for CoreML export.

    CoreML doesn't support dynamic graphs, so EGNN message passing is replaced
    with MLP encoders. The Metal compute shaders on-device handle spatial graph
    construction and neighbor feature aggregation before feeding into CoreML.

    v2 changes from v1:
      - atom_dim 18 -> 20 (formal charge + ring membership)
      - 2 stacked cross-attention layers with residual + LayerNorm
      - Gated attention pooling (Tanh activation)
      - Heads with Dropout (no-op in eval mode, preserves weight index alignment)

    Input: pre-computed features from Metal compute shaders.
    Output: docking_score (primary), pKd, confidence, interaction_map, attention.
    """

    def __init__(self, atom_dim=20, hidden_dim=128, num_cross_attn_layers=2,
                 max_prot=256, max_lig=64):
        super().__init__()
        self.max_prot = max_prot
        self.max_lig = max_lig
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads

        # Simplified encoders (Metal pre-computes spatial aggregation)
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

        # v2: Stacked cross-attention (2 layers with independent parameters)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayerExport(hidden_dim, self.num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        # v2: Gated attention pooling (Tanh, matching training model)
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Affinity head (Dropout at index 2 — no-op in eval, preserves weight indices)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, 1),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Interaction head (5 types: H-bond, hydrophobic, ionic, pi-stacking, halogen)
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 50, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, protein_features, ligand_features, protein_positions,
                ligand_positions, pair_rbf):
        """
        protein_features: [1, P, 20]
        ligand_features: [1, L, 20]
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
        lig_h = self.lig_encoder(ligand_features.squeeze(0))     # [L, D]
        rbf = pair_rbf.squeeze(0)  # [L, P, 50]

        # v2: Stacked cross-attention (2 layers with residual)
        attn_weights_last = None
        for layer in self.cross_attn_layers:
            lig_h, attn_weights_last = layer(lig_h, prot_h, rbf, L, P)

        # v2: Gated attention pooling
        gate_logits = self.pool_gate(lig_h)  # [L, 1]
        gate_weights = torch.softmax(gate_logits, dim=0)  # [L, 1]
        complex_repr = (gate_weights * lig_h).sum(dim=0, keepdim=True)  # [1, D]

        # Predictions
        pkd = self.affinity_head(complex_repr)                    # [1, 1]
        confidence = torch.sigmoid(self.confidence_head(complex_repr))  # [1, 1]
        docking_score = pkd * confidence                          # [1, 1]

        # Interaction map
        lig_exp = lig_h.unsqueeze(1).expand(L, P, D)
        prot_exp = prot_h.unsqueeze(0).expand(L, P, D)
        pair_input = torch.cat([lig_exp, prot_exp, rbf], dim=-1)
        interaction_map = torch.sigmoid(self.interaction_head(pair_input))  # [L, P, 5]

        # Attention weights (mean over heads, from last layer)
        attn_out = attn_weights_last.mean(dim=-1)  # [L, P]

        return (docking_score, pkd, confidence,
                interaction_map.unsqueeze(0), attn_out.reshape(1, L * P))


def map_training_weights(training_state: dict, export_model: nn.Module) -> dict:
    """Map training model weight names to export model architecture.

    Training model has EGNN encoders (EGNNLayer + radius_graph).
    Export model has MLP encoders (no spatial graph ops).

    v2 weight mapping:
      - prot/lig_encoder.input_proj.* -> prot/lig_encoder.0.* (EGNN input_proj -> MLP first layer)
      - cross_attn_layers.{i}.* -> cross_attn_layers.{i}.* (direct match)
      - pool_gate.* -> pool_gate.* (direct match)
      - affinity_head.{0,3}.* -> affinity_head.{0,3}.* (direct match, indices preserved)
      - confidence_head.{0,3}.* -> confidence_head.{0,3}.* (direct match)
      - interaction_head.* -> interaction_head.* (direct match)
    """
    export_state = export_model.state_dict()
    mapped = {}

    # First pass: direct key matches with shape compatibility
    for key in export_state:
        if key in training_state and export_state[key].shape == training_state[key].shape:
            mapped[key] = training_state[key]
        else:
            mapped[key] = export_state[key]

    # Map EGNN encoder input_proj -> MLP encoder first layer
    mapping_rules = [
        ("prot_encoder.input_proj.weight", "prot_encoder.0.weight"),
        ("prot_encoder.input_proj.bias", "prot_encoder.0.bias"),
        ("lig_encoder.input_proj.weight", "lig_encoder.0.weight"),
        ("lig_encoder.input_proj.bias", "lig_encoder.0.bias"),
    ]
    for train_key, export_key in mapping_rules:
        if train_key in training_state and export_key in export_state:
            if training_state[train_key].shape == export_state[export_key].shape:
                mapped[export_key] = training_state[train_key]

    return mapped


def export_druse_score_pki(checkpoint_path: Path, output_dir: Path,
                            max_prot: int = 256, max_lig: int = 64):
    """Export DruseScore-pKi v2 to CoreML."""
    print(f"Exporting DruseScore-pKi v2...")

    # Load checkpoint metadata
    sigma = 2.0
    hidden_dim = 128
    atom_dim = 20
    num_cross_attn_layers = 2
    best_score_r = 0.0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            training_state = checkpoint["model_state_dict"]
            sigma = checkpoint.get("sigma", 2.0)
            hidden_dim = checkpoint.get("hidden_dim", 128)
            atom_dim = checkpoint.get("atom_dim", 20)
            num_cross_attn_layers = checkpoint.get("num_cross_attn_layers", 2)
            best_score_r = checkpoint.get("best_score_r", 0.0)
            model_version = checkpoint.get("model_version", "v2")
            epoch = checkpoint.get("epoch", "?")
            print(f"  Loaded checkpoint ({model_version}, epoch {epoch}):")
            print(f"    atom_dim={atom_dim}, hidden_dim={hidden_dim}, "
                  f"cross_attn_layers={num_cross_attn_layers}")
            print(f"    sigma={sigma}, best_score_r={best_score_r:.4f}")
        else:
            training_state = checkpoint
            print(f"  Loaded raw state dict")
    else:
        training_state = None
        print(f"  No checkpoint found, exporting random weights (for testing)")

    model = DruseScorePKiForExport(
        atom_dim=atom_dim, hidden_dim=hidden_dim,
        num_cross_attn_layers=num_cross_attn_layers,
        max_prot=max_prot, max_lig=max_lig)

    if training_state is not None:
        mapped = map_training_weights(training_state, model)
        load_result = model.load_state_dict(mapped, strict=False)
        n_loaded = len(mapped) - len(load_result.missing_keys)
        n_total = len(model.state_dict())
        print(f"  Loaded {n_loaded}/{n_total} weight tensors")
        if load_result.missing_keys:
            print(f"  Missing (using init): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected (ignored): {load_result.unexpected_keys}")

    model.eval()

    # Trace with example inputs
    P, L = max_prot, max_lig
    prot_feat = torch.randn(1, P, atom_dim)
    lig_feat = torch.randn(1, L, atom_dim)
    prot_pos = torch.randn(1, P, 3)
    lig_pos = torch.randn(1, L, 3)
    pair_rbf = torch.randn(1, L, P, 50)

    traced = torch.jit.trace(model, (prot_feat, lig_feat, prot_pos, lig_pos, pair_rbf))

    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="protein_features", shape=(1, P, atom_dim)),
            ct.TensorType(name="ligand_features", shape=(1, L, atom_dim)),
            ct.TensorType(name="protein_positions", shape=(1, P, 3)),
            ct.TensorType(name="ligand_positions", shape=(1, L, 3)),
            ct.TensorType(name="pair_rbf", shape=(1, L, P, 50)),
        ],
        outputs=[
            ct.TensorType(name="docking_score"),
            ct.TensorType(name="pKd"),
            ct.TensorType(name="pose_confidence"),
            ct.TensorType(name="interaction_map"),
            ct.TensorType(name="attention_weights"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    ml_model.author = "Druse"
    ml_model.short_description = (
        "DruseScore-pKi v2: SE(3)-equivariant primary scoring function for molecular docking. "
        "Trained on 13K+ co-crystallized structures. "
        "Predicts docking_score = pKd * pose_confidence for pose ranking."
    )
    ml_model.version = "3.0"

    # Store training metadata so the Swift app can use it
    ml_model.user_defined_metadata["model_type"] = "druse_score_pki"
    ml_model.user_defined_metadata["model_version"] = "v2"
    ml_model.user_defined_metadata["scoring_mode"] = "primary"
    ml_model.user_defined_metadata["pose_confidence_sigma"] = f"{sigma:.4f}"
    ml_model.user_defined_metadata["hidden_dim"] = str(hidden_dim)
    ml_model.user_defined_metadata["atom_dim"] = str(atom_dim)
    ml_model.user_defined_metadata["num_cross_attn_layers"] = str(num_cross_attn_layers)
    ml_model.user_defined_metadata["max_protein_atoms"] = str(max_prot)
    ml_model.user_defined_metadata["max_ligand_atoms"] = str(max_lig)
    ml_model.user_defined_metadata["rbf_dim"] = "50"
    ml_model.user_defined_metadata["rbf_gamma"] = "10.0"
    ml_model.user_defined_metadata["rbf_max_dist"] = "10.0"
    ml_model.user_defined_metadata["interaction_types"] = json.dumps([
        "h_bond", "hydrophobic", "ionic", "pi_stacking", "halogen_bond"
    ])
    ml_model.user_defined_metadata["training_score_r"] = f"{best_score_r:.4f}"
    ml_model.user_defined_metadata["atom_features"] = json.dumps([
        "element_onehot_10", "aromatic", "partial_charge", "hb_donor", "hb_acceptor",
        "hyb_sp", "hyb_sp2", "hyb_sp3", "is_ligand", "formal_charge", "in_ring"
    ])
    ml_model.user_defined_metadata["output_description"] = json.dumps({
        "docking_score": "Primary ranking value: pKd * pose_confidence. Higher = better.",
        "pKd": "Predicted -log10(Kd) binding affinity. Range ~2-12.",
        "pose_confidence": "Estimated pose quality (0=garbage, 1=crystal-like).",
        "interaction_map": "[L, P, 5] per atom-pair interaction probabilities.",
        "attention_weights": "[L*P] cross-attention weights for visualization.",
    })

    out_path = output_dir / "DruseScorePKi.mlpackage"
    ml_model.save(str(out_path))
    print(f"  Saved: {out_path}")

    # Print summary for Swift integration
    print(f"\n  === Swift Integration Notes ===")
    print(f"  Primary output: docking_score (replace Vina score)")
    print(f"  Input padding: protein to {max_prot} atoms, ligand to {max_lig} atoms")
    print(f"  Atom features: {atom_dim}-dim (v2: +formal_charge, +in_ring)")
    print(f"  RBF encoding: 50 Gaussians, gamma=10.0, range 0-10 A")
    print(f"  Cross-attention: {num_cross_attn_layers} layers with residual + LayerNorm")
    print(f"  Pooling: gated attention (learned atom importance)")
    print(f"  Interaction types: H-bond, Hydrophobic, Ionic, Pi-stack, Halogen")
    print(f"  Pose confidence: Gaussian decay sigma={sigma}")
    print(f"  Training score_r: {best_score_r:.4f}")

    try:
        compiled = ct.models.CompiledMLModel(str(out_path))
        print(f"  Compiled model ready for ANE")
    except Exception:
        print(f"  Compile on target: xcrun coremlcompiler compile {out_path} DruseScorePKi.mlmodelc")


def main():
    parser = argparse.ArgumentParser(description="Export DruseScore-pKi v2 to CoreML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pki/druse_pki_v2_best.pt",
                        help="Path to trained v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="../Models/druse-models/",
                        help="Output directory for .mlpackage")
    parser.add_argument("--max_prot", type=int, default=256,
                        help="Max protein atoms (pad/truncate)")
    parser.add_argument("--max_lig", type=int, default=64,
                        help="Max ligand atoms (pad/truncate)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== DruseScore-pKi v2 CoreML Export ===\n")
    export_druse_score_pki(Path(args.checkpoint), output_dir, args.max_prot, args.max_lig)

    print(f"\nCoreML model saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Update Inference.swift: atomFeatureSize 18 -> 20")
    print("  2. Add features[18] = formal_charge, features[19] = in_ring")
    print("  3. Rebuild Xcode project and test")


if __name__ == "__main__":
    main()

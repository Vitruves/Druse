#!/usr/bin/env python3
"""
Export DruseAF (DruseScorePKi v2) weights as a flat binary for Metal GPU inference.

The binary format is designed for direct loading into a single MTLBuffer:
  Header:
    magic: "DRAF" (4 bytes)
    version: uint32 (1)
    numTensors: uint32
    totalFloats: uint32
    offsets[numTensors]: (offset_uint32, count_uint32) pairs
  Data:
    concatenated float32 weights

Usage:
  python export_druseaf_weights.py --checkpoint ../temp/DruseAF\ v3/druse_pki_v2_best.pt
"""

import argparse
import struct
from pathlib import Path

import torch
import numpy as np


# Weight tensor order matching the export model architecture.
# These are the keys in the DruseScorePKiForExport state_dict,
# mapped from the training model's EGNN architecture.
EXPORT_WEIGHT_ORDER = [
    # MLP encoders (mapped from EGNN input_proj)
    "prot_encoder.0.weight",    # [128, 20]
    "prot_encoder.0.bias",      # [128]
    "prot_encoder.2.weight",    # [128, 128]
    "prot_encoder.2.bias",      # [128]
    "lig_encoder.0.weight",     # [128, 20]
    "lig_encoder.0.bias",       # [128]
    "lig_encoder.2.weight",     # [128, 128]
    "lig_encoder.2.bias",       # [128]
    # Cross-attention layer 0
    "cross_attn_layers.0.q_proj.weight",   # [128, 128]
    "cross_attn_layers.0.q_proj.bias",     # [128]
    "cross_attn_layers.0.k_proj.weight",   # [128, 128]
    "cross_attn_layers.0.k_proj.bias",     # [128]
    "cross_attn_layers.0.v_proj.weight",   # [128, 128]
    "cross_attn_layers.0.v_proj.bias",     # [128]
    "cross_attn_layers.0.rbf_proj.weight", # [4, 50]
    "cross_attn_layers.0.rbf_proj.bias",   # [4]
    "cross_attn_layers.0.out_proj.weight", # [128, 128]
    "cross_attn_layers.0.out_proj.bias",   # [128]
    "cross_attn_layers.0.norm.weight",     # [128]
    "cross_attn_layers.0.norm.bias",       # [128]
    # Cross-attention layer 1
    "cross_attn_layers.1.q_proj.weight",   # [128, 128]
    "cross_attn_layers.1.q_proj.bias",     # [128]
    "cross_attn_layers.1.k_proj.weight",   # [128, 128]
    "cross_attn_layers.1.k_proj.bias",     # [128]
    "cross_attn_layers.1.v_proj.weight",   # [128, 128]
    "cross_attn_layers.1.v_proj.bias",     # [128]
    "cross_attn_layers.1.rbf_proj.weight", # [4, 50]
    "cross_attn_layers.1.rbf_proj.bias",   # [4]
    "cross_attn_layers.1.out_proj.weight", # [128, 128]
    "cross_attn_layers.1.out_proj.bias",   # [128]
    "cross_attn_layers.1.norm.weight",     # [128]
    "cross_attn_layers.1.norm.bias",       # [128]
    # Gated attention pooling
    "pool_gate.0.weight",       # [64, 128]
    "pool_gate.0.bias",         # [64]
    "pool_gate.2.weight",       # [1, 64]
    "pool_gate.2.bias",         # [1]
    # Affinity head (indices 0, 3 — skip dropout at index 2)
    "affinity_head.0.weight",   # [128, 128]
    "affinity_head.0.bias",     # [128]
    "affinity_head.3.weight",   # [1, 128]
    "affinity_head.3.bias",     # [1]
    # Confidence head (indices 0, 3)
    "confidence_head.0.weight", # [64, 128]
    "confidence_head.0.bias",   # [64]
    "confidence_head.3.weight", # [1, 64]
    "confidence_head.3.bias",   # [1]
    # Interaction head (omitted from Metal scoring — only for visualization)
    # "interaction_head.0.weight",  # [128, 306]
    # "interaction_head.0.bias",    # [128]
    # "interaction_head.2.weight",  # [5, 128]
    # "interaction_head.2.bias",    # [5]
]

# Mapping from training model keys to export model keys
TRAINING_TO_EXPORT = {
    "prot_encoder.input_proj.weight": "prot_encoder.0.weight",
    "prot_encoder.input_proj.bias": "prot_encoder.0.bias",
    "lig_encoder.input_proj.weight": "lig_encoder.0.weight",
    "lig_encoder.input_proj.bias": "lig_encoder.0.bias",
}


def load_weights(checkpoint_path: Path) -> dict:
    """Load checkpoint, build the export model, and extract its full state dict.

    This ensures Metal weights exactly match the CoreML export (including
    randomly initialized layers that have no training-model analog).
    """
    # Import the export model class from the sibling script
    import importlib.util
    export_script = Path(__file__).parent / "export_coreml_pKi.py"
    spec = importlib.util.spec_from_file_location("export_pki", export_script)
    export_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_mod)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        training_state = checkpoint["model_state_dict"]
        atom_dim = checkpoint.get("atom_dim", 20)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_cross_attn = checkpoint.get("num_cross_attn_layers", 2)
    else:
        training_state = checkpoint
        atom_dim, hidden_dim, num_cross_attn = 20, 128, 2

    # Build export model and load mapped weights (same as CoreML export)
    model = export_mod.DruseScorePKiForExport(
        atom_dim=atom_dim, hidden_dim=hidden_dim,
        num_cross_attn_layers=num_cross_attn)
    mapped = export_mod.map_training_weights(training_state, model)
    model.load_state_dict(mapped, strict=False)

    # Extract the complete state dict with all weights resolved
    result = {}
    for key, val in model.state_dict().items():
        result[key] = val.float().numpy()

    return result


def export_binary(weights: dict, output_path: Path):
    """Pack weights into flat binary format for Metal."""
    tensors = []
    for key in EXPORT_WEIGHT_ORDER:
        if key not in weights:
            print(f"  WARNING: missing weight '{key}' — using zeros")
            # Infer shape from key name patterns
            tensors.append(np.zeros(1, dtype=np.float32))
        else:
            tensors.append(weights[key].flatten().astype(np.float32))

    num_tensors = len(tensors)

    # Build offset table
    offsets = []
    current_offset = 0
    for t in tensors:
        offsets.append((current_offset, len(t)))
        current_offset += len(t)
    total_floats = current_offset

    # Write binary
    with open(output_path, "wb") as f:
        # Header
        f.write(b"DRAF")                                    # magic
        f.write(struct.pack("<I", 1))                       # version
        f.write(struct.pack("<I", num_tensors))             # numTensors
        f.write(struct.pack("<I", total_floats))            # totalFloats

        # Offset table: (offset, count) pairs
        for off, cnt in offsets:
            f.write(struct.pack("<II", off, cnt))

        # Concatenated float32 data
        for t in tensors:
            f.write(t.tobytes())

    total_kb = (total_floats * 4) / 1024
    print(f"  Exported {num_tensors} tensors, {total_floats:,} floats ({total_kb:.1f} KB)")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DruseAF weights for Metal GPU")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained v2 checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: alongside checkpoint)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    output_path = Path(args.output) if args.output else ckpt_path.parent / "DruseAF.weights"

    print("=== DruseAF Weight Export (Metal Binary) ===\n")
    print(f"  Checkpoint: {ckpt_path}")

    weights = load_weights(ckpt_path)

    # Print weight summary
    found = 0
    for key in EXPORT_WEIGHT_ORDER:
        if key in weights:
            found += 1
    print(f"  Matched {found}/{len(EXPORT_WEIGHT_ORDER)} weight tensors\n")

    export_binary(weights, output_path)

    print(f"\nTo use in Druse:")
    print(f"  1. Copy {output_path.name} to Models/druse-models/")
    print(f"  2. Add to Xcode project Resources")


if __name__ == "__main__":
    main()

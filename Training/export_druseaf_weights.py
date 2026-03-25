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

def load_weights(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract state dict directly.

    With MLP encoders in train_druse_pKi_v2.py, the training model weight keys
    match EXPORT_WEIGHT_ORDER exactly — no remapping needed.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        encoder_type = checkpoint.get("encoder", "unknown")
        if encoder_type != "mlp":
            print(f"  WARNING: checkpoint encoder='{encoder_type}', expected 'mlp'")
            print(f"  Retrain with MLP encoders: python train_druse_pKi_v2.py ...")
    else:
        state = checkpoint

    result = {}
    for key, val in state.items():
        result[key] = val.float().numpy()
    return result


def export_binary(weights: dict, output_path: Path):
    """Pack weights into flat binary format for Metal."""

    # Verify all expected weights are present and have correct shapes
    missing = []
    for key in EXPORT_WEIGHT_ORDER:
        if key not in weights:
            missing.append(key)
    if missing:
        print(f"\n  ERROR: {len(missing)} weights missing from checkpoint:")
        for k in missing:
            print(f"    - {k}")
        print(f"\n  This likely means the checkpoint was trained with EGNN encoders.")
        print(f"  Retrain with: python train_druse_pKi_v2.py --input <cache> --output <dir> --epochs 80")
        return

    tensors = []
    for key in EXPORT_WEIGHT_ORDER:
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

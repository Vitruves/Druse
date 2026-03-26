#!/usr/bin/env python3
"""
Export PIGNet2 weights as a flat binary for Metal GPU inference.

Binary format (reuses DRAF-style layout with "PIG2" magic):
  Header:
    magic: "PIG2" (4 bytes)
    version: uint32 (1)
    numTensors: uint32
    totalFloats: uint32
    offsets[numTensors]: (offset_uint32, count_uint32) pairs
  Data:
    concatenated float32 weights

Usage:
  python export_pignet2_weights.py --checkpoint path/to/pignet2_best.pt
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Exact parameter order matching the Metal shader tensor indices (0-50).
# Obtained from PIGNet2 model.named_parameters() with in_features=47, dim_gnn=128.
EXPORT_WEIGHT_ORDER = [
    # Scalar physics coefficients (0-4)
    "hbond_coeff",                          # [1]
    "hydrophobic_coeff",                    # [1]
    "rotor_coeff",                          # [1]
    "metal_ligand_coeff",                   # [1]
    "ionic_coeff",                          # [1] (unused but present in checkpoint)
    # Embedding (5)
    "embed.weight",                         # [128, 47]
    # Intra GatedGAT layer 0 (6-10)
    "intraconv.0.module_0.W2",              # [128, 128]
    "intraconv.0.module_0.W1.weight",       # [128, 128]
    "intraconv.0.module_0.W1.bias",         # [128]
    "intraconv.0.module_0.gate.weight",     # [1, 256]
    "intraconv.0.module_0.gate.bias",       # [1]
    # Intra GatedGAT layer 1 (11-15)
    "intraconv.1.module_0.W2",              # [128, 128]
    "intraconv.1.module_0.W1.weight",       # [128, 128]
    "intraconv.1.module_0.W1.bias",         # [128]
    "intraconv.1.module_0.gate.weight",     # [1, 256]
    "intraconv.1.module_0.gate.bias",       # [1]
    # Intra GatedGAT layer 2 (16-20)
    "intraconv.2.module_0.W2",              # [128, 128]
    "intraconv.2.module_0.W1.weight",       # [128, 128]
    "intraconv.2.module_0.W1.bias",         # [128]
    "intraconv.2.module_0.gate.weight",     # [1, 256]
    "intraconv.2.module_0.gate.bias",       # [1]
    # Inter InteractionNet layer 0 (21-28)
    "interconv.0.module_0.W1.weight",       # [128, 128]
    "interconv.0.module_0.W1.bias",         # [128]
    "interconv.0.module_0.W2.weight",       # [128, 128]
    "interconv.0.module_0.W2.bias",         # [128]
    "interconv.0.module_0.rnn.weight_ih",   # [384, 128]
    "interconv.0.module_0.rnn.weight_hh",   # [384, 128]
    "interconv.0.module_0.rnn.bias_ih",     # [384]
    "interconv.0.module_0.rnn.bias_hh",     # [384]
    # Inter InteractionNet layer 1 (29-36)
    "interconv.1.module_0.W1.weight",       # [128, 128]
    "interconv.1.module_0.W1.bias",         # [128]
    "interconv.1.module_0.W2.weight",       # [128, 128]
    "interconv.1.module_0.W2.bias",         # [128]
    "interconv.1.module_0.rnn.weight_ih",   # [384, 128]
    "interconv.1.module_0.rnn.weight_hh",   # [384, 128]
    "interconv.1.module_0.rnn.bias_ih",     # [384]
    "interconv.1.module_0.rnn.bias_hh",     # [384]
    # Inter InteractionNet layer 2 (37-44)
    "interconv.2.module_0.W1.weight",       # [128, 128]
    "interconv.2.module_0.W1.bias",         # [128]
    "interconv.2.module_0.W2.weight",       # [128, 128]
    "interconv.2.module_0.W2.bias",         # [128]
    "interconv.2.module_0.rnn.weight_ih",   # [384, 128]
    "interconv.2.module_0.rnn.weight_hh",   # [384, 128]
    "interconv.2.module_0.rnn.bias_ih",     # [384]
    "interconv.2.module_0.rnn.bias_hh",     # [384]
    # nn_vdw_epsilon: Linear(256,128) → ReLU → Linear(128,1) → Sigmoid (45-48)
    "nn_vdw_epsilon.module_0.weight",       # [128, 256]
    "nn_vdw_epsilon.module_0.bias",         # [128]
    "nn_vdw_epsilon.module_2.weight",       # [1, 128]
    "nn_vdw_epsilon.module_2.bias",         # [1]
    # nn_dvdw: Linear(256,128) → ReLU → Linear(128,1) → Tanh (49-52)
    "nn_dvdw.module_0.weight",              # [128, 256]
    "nn_dvdw.module_0.bias",               # [128]
    "nn_dvdw.module_2.weight",              # [1, 128]
    "nn_dvdw.module_2.bias",               # [1]
    # nn_vdw_width (Morse): Linear(256,128) → ReLU → Linear(128,1) → Sigmoid (53-56)
    "nn_vdw_width.module_0.weight",         # [128, 256]
    "nn_vdw_width.module_0.bias",           # [128]
    "nn_vdw_width.module_2.weight",         # [1, 128]
    "nn_vdw_width.module_2.bias",           # [1]
    # nn_vdw_radius (Morse): Linear(256,128) → ReLU → Linear(128,1) → ReLU (57-60)
    "nn_vdw_radius.module_0.weight",        # [128, 256]
    "nn_vdw_radius.module_0.bias",          # [128]
    "nn_vdw_radius.module_2.weight",        # [1, 128]
    "nn_vdw_radius.module_2.bias",          # [1]
]


def load_weights(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    result = {}
    for key, val in state.items():
        if val.dim() == 0:
            val = val.unsqueeze(0)
        result[key] = val.float().numpy()
    return result


def export_binary(weights: dict, output_path: Path):
    """Pack weights into flat binary format for Metal."""
    missing = [k for k in EXPORT_WEIGHT_ORDER if k not in weights]
    if missing:
        print(f"\n  ERROR: {len(missing)} weights missing from checkpoint:")
        for k in missing:
            print(f"    - {k}")
        sys.exit(1)

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
        f.write(b"PIG2")                                    # magic
        f.write(struct.pack("<I", 1))                       # version
        f.write(struct.pack("<I", num_tensors))             # numTensors
        f.write(struct.pack("<I", total_floats))            # totalFloats

        for off, cnt in offsets:
            f.write(struct.pack("<II", off, cnt))

        for t in tensors:
            f.write(t.tobytes())

    total_kb = (total_floats * 4) / 1024
    print(f"  Exported {num_tensors} tensors, {total_floats:,} floats ({total_kb:.1f} KB)")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PIGNet2 weights for Metal GPU")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained PIGNet2 checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: PIGNet2.weights alongside checkpoint)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else ckpt_path.parent / "PIGNet2.weights"

    print("=== PIGNet2 Weight Export (Metal Binary) ===\n")
    print(f"  Checkpoint: {ckpt_path}")

    weights = load_weights(ckpt_path)

    found = sum(1 for key in EXPORT_WEIGHT_ORDER if key in weights)
    print(f"  Matched {found}/{len(EXPORT_WEIGHT_ORDER)} weight tensors\n")

    export_binary(weights, output_path)

    print(f"\nTo use in Druse:")
    print(f"  1. Copy {output_path.name} to Models/druse-models/")
    print(f"  2. Rebuild with xcodegen + xcodebuild")


if __name__ == "__main__":
    main()

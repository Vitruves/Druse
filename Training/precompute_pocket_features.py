#!/usr/bin/env python3
"""
Precompute surface features for pocket detector training.
Run once, then train_pocket_detector.py uses cached .pt files (fast GPU training).

Usage:
  python precompute_pocket_features.py --data_dir data/ --workers 24
  # Then train:
  python train_pocket_detector.py --data_dir data/ --epochs 50 --use_cache
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm

from Training.trainPocketML import (
    compute_surface_features,
    PocketSurfaceDataset,
    SURFACE_FEAT_DIM,
)


def process_one(args):
    """Process a single protein → cached .pt file."""
    idx, pdb_id, protein_path, ligand_path, cache_dir, threshold = args
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{pdb_id}.pt"
    if cache_file.exists():
        return pdb_id, True, "cached"

    try:
        surf_pos, surf_feat, _ = compute_surface_features(protein_path)
        if surf_pos is None:
            return pdb_id, False, "no surface"

        # Parse ligand
        positions = []
        in_atoms = False
        with open(ligand_path) as f:
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

        if not positions:
            return pdb_id, False, "no ligand"

        lig_pos = np.array(positions, dtype=np.float32)

        # Labels
        from scipy.spatial.distance import cdist
        dists = cdist(surf_pos, lig_pos).min(axis=1)
        labels = (dists < threshold).astype(np.float32)

        # Build graph
        pos_t = torch.tensor(surf_pos, dtype=torch.float32)
        feat_t = torch.tensor(surf_feat, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        edge_index = radius_graph(pos_t, r=4.0, max_num_neighbors=16)

        data = Data(x=feat_t, pos=pos_t, edge_index=edge_index, y=labels_t, pdb_id=pdb_id)
        torch.save(data, cache_file)

        n_pos = int(labels.sum())
        return pdb_id, True, f"{len(surf_pos)} pts, {n_pos} pocket"

    except Exception as e:
        return pdb_id, False, str(e)[:80]


def main():
    parser = argparse.ArgumentParser(description="Precompute pocket surface features")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers (0=auto, uses cpu_count-2)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Distance threshold for pocket labeling (A)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "pocket_cache"
    cache_dir.mkdir(exist_ok=True)

    # Find all entries
    refined_dir = data_dir / "refined-set"
    entries = []
    for pdb_dir in sorted(refined_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdb_id = pdb_dir.name
        protein_path = pdb_dir / f"{pdb_id}_protein.pdb"
        ligand_path = pdb_dir / f"{pdb_id}_ligand.mol2"
        if protein_path.exists() and ligand_path.exists():
            entries.append((len(entries), pdb_id, str(protein_path), str(ligand_path),
                          str(cache_dir), args.threshold))

    print(f"Found {len(entries)} complexes")
    print(f"Cache dir: {cache_dir}")

    n_workers = args.workers if args.workers > 0 else max(cpu_count() - 2, 1)
    print(f"Using {n_workers} workers\n")

    success = 0
    failed = 0
    fail_reasons = {}
    first_errors = []  # store first 5 unique errors with pdb_id

    with Pool(n_workers) as pool:
        for pdb_id, ok, msg in tqdm(pool.imap_unordered(process_one, entries),
                                      total=len(entries), desc="Precomputing"):
            if ok:
                success += 1
            else:
                failed += 1
                fail_reasons[msg] = fail_reasons.get(msg, 0) + 1
                if len(first_errors) < 5 and msg not in [e[1] for e in first_errors]:
                    first_errors.append((pdb_id, msg))

            # Live progress every 200 samples
            total = success + failed
            if total == 1 or total % 200 == 0:
                rate = success / max(total, 1) * 100
                tqdm.write(f"  [{total}/{len(entries)}] {success} ok, {failed} fail ({rate:.0f}% success)")
                if first_errors and total <= 5:
                    for eid, emsg in first_errors:
                        tqdm.write(f"    FIRST ERROR ({eid}): {emsg}")

            # Abort early if first 50 all fail
            if total == 50 and success == 0:
                tqdm.write(f"\n  ABORT: First 50 samples all failed. Errors:")
                for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1])[:5]:
                    tqdm.write(f"    {count}x  {reason}")
                tqdm.write(f"\n  Fix the errors above and retry.")
                return

    if fail_reasons:
        print(f"\nFailure reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:5d}x  {reason}")

    print(f"\nDone: {success} cached, {failed} failed")
    print(f"Cache: {cache_dir} ({success} .pt files)")
    print(f"\nNow train with: python train_pocket_detector.py --data_dir {data_dir} --use_cache")


if __name__ == "__main__":
    main()

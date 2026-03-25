#!/usr/bin/env python3
"""
Precompute features + structured RMSD perturbations for DruseScore-pKi training.

For each crystal complex in PDBbind, generates:
  - 1 crystal pose (RMSD=0, confidence=1.0, score=pKd)
  - 7 perturbed poses at RMSD targets: 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0 A
  - Each perturbed pose gets confidence = exp(-RMSD^2 / 2*sigma^2)
  - Each perturbed pose gets docking_score = pKd * confidence

Uses 20-dim atom features (formal charge + ring membership) from train_druse_pKi_v2.

Usage:
  python precompute_druse_features_pKi.py --input data/v2020r1/P-L --output data/druse_pki_v2_cache --workers 24
  python train_druse_pKi_v2.py --input data/druse_pki_v2_cache --output checkpoints_pki_v2/ --epochs 80
"""

import argparse
import math
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from train_druse_pKi_v2 import (
    pdb_atom_features as _pdb_atom_features,
    mol2_atom_features as _mol2_atom_features,
    NUM_ATOM_FEATURES as _NUM_ATOM_FEATURES,
    POSE_CONFIDENCE_SIGMA as _POSE_CONFIDENCE_SIGMA,
    PERTURBATION_RMSD_TARGETS as _PERTURBATION_RMSD_TARGETS,
    rmsd_to_confidence as _rmsd_to_confidence,
)


def parse_pdb(path):
    """Parse PDB pocket file -> (positions [N,3], features [N,feat_dim])."""
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
                residue_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                positions.append([x, y, z])
                features.append(_pdb_atom_features(element, residue_name, atom_name))
    except Exception:
        return None, None
    if not positions:
        return None, None
    return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)


def parse_mol2(path):
    """Parse MOL2 ligand file -> (positions [N,3], features [N,feat_dim])."""
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
                features.append(_mol2_atom_features(element, sybyl_type, charge))
    except Exception:
        return None, None
    if not positions:
        return None, None
    return np.array(positions, dtype=np.float32), np.array(features, dtype=np.float32)


def generate_perturbed_pose(lig_pos: np.ndarray, target_rmsd: float) -> tuple:
    """Generate a perturbed ligand pose targeting a specific RMSD.

    Uses rotation + translation calibrated to achieve approximately the target RMSD.
    Returns (perturbed_pos, actual_rmsd).
    """
    if target_rmsd < 0.01:
        return lig_pos.copy(), 0.0

    center = lig_pos.mean(axis=0)
    centered = lig_pos - center
    lig_radius = np.sqrt((centered ** 2).sum(axis=1).mean())

    # Split target between rotation and translation
    rot_rmsd_target = target_rmsd * 0.6
    trans_rmsd_target = target_rmsd * 0.4

    # Rotation angle calibrated to ligand size
    if lig_radius > 0.1:
        rot_angle = rot_rmsd_target / (lig_radius * 0.816)
        rot_angle = min(rot_angle, math.pi)
    else:
        rot_angle = 0.0

    axis = np.random.randn(3).astype(np.float32)
    axis /= (np.linalg.norm(axis) + 1e-8)
    rotvec = axis * rot_angle
    R = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)
    rotated = centered @ R.T

    trans_dir = np.random.randn(3).astype(np.float32)
    trans_dir /= (np.linalg.norm(trans_dir) + 1e-8)
    translated = rotated + center + trans_dir * trans_rmsd_target

    actual_rmsd = np.sqrt(np.mean(np.sum((translated - lig_pos) ** 2, axis=1)))
    return translated, float(actual_rmsd)


def process_one(args):
    """Process one complex -> multiple cached .pt files (crystal + perturbations)."""
    idx, pdb_id, pocket_path, ligand_path, cache_dir, pkd = args

    # Check if all perturbations already cached
    all_cached = all(
        (Path(cache_dir) / f"{pdb_id}_p{pi}.pt").exists()
        for pi in range(len(_PERTURBATION_RMSD_TARGETS))
    )
    if all_cached:
        return pdb_id, len(_PERTURBATION_RMSD_TARGETS), "cached"

    # Parse protein and ligand
    try:
        prot_pos, prot_feat = parse_pdb(pocket_path)
        if prot_pos is None:
            return pdb_id, 0, "no protein atoms"

        lig_pos, lig_feat = parse_mol2(ligand_path)
        if lig_pos is None:
            return pdb_id, 0, "no ligand atoms"
    except Exception as e:
        return pdb_id, 0, str(e)[:80]

    # Shared tensors (protein features don't change across perturbations)
    prot_pos_t = torch.tensor(prot_pos, dtype=torch.float32)
    prot_feat_t = torch.tensor(prot_feat, dtype=torch.float32)
    lig_feat_t = torch.tensor(lig_feat, dtype=torch.float32)

    saved = 0
    for pi, rmsd_target in enumerate(_PERTURBATION_RMSD_TARGETS):
        cache_file = Path(cache_dir) / f"{pdb_id}_p{pi}.pt"
        if cache_file.exists():
            saved += 1
            continue

        try:
            pert_pos, actual_rmsd = generate_perturbed_pose(lig_pos, rmsd_target)
            confidence = _rmsd_to_confidence(actual_rmsd)
            docking_score = pkd * confidence

            data = Data(
                prot_pos=prot_pos_t,
                prot_x=prot_feat_t,
                lig_pos=torch.tensor(pert_pos, dtype=torch.float32),
                lig_x=lig_feat_t,
                y=torch.tensor([pkd], dtype=torch.float32),
                rmsd=torch.tensor([actual_rmsd], dtype=torch.float32),
                pose_confidence=torch.tensor([confidence], dtype=torch.float32),
                docking_score=torch.tensor([docking_score], dtype=torch.float32),
                pdb_id=pdb_id,
                perturbation_idx=pi,
            )
            torch.save(data, cache_file)
            saved += 1
        except Exception as e:
            continue

    return pdb_id, saved, f"{len(prot_pos)} prot, {len(lig_pos)} lig, {saved} perturbations"


def discover_complexes(input_dir: Path, labels_csv: str):
    """Discover PDBbind complexes from input directory.

    Supports multiple layouts:
      1. Labels CSV mode: input_dir/labels_csv + input_dir/refined-set/<pdb>/ (original v2020)
      2. Flat mode: input_dir/<pdb>/<pdb>_pocket.pdb + <pdb>_ligand.mol2 (v2020.R1 general set)
      3. Index file mode: input_dir/index/INDEX_general_PL_data.<year> (PDBbind index)
    """
    entries = []

    # --- Strategy 1: Labels CSV ---
    labels_path = input_dir / labels_csv
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        if "split" in df.columns:
            df = df[df["split"] == "refined"]
        for _, row in df.iterrows():
            pdb_id = row["pdb_id"]
            pkd = row["pKd"]
            # Try refined-set subdirectory first, then flat layout
            for subdir in ["refined-set", ""]:
                base = input_dir / subdir / pdb_id if subdir else input_dir / pdb_id
                pocket = base / f"{pdb_id}_pocket.pdb"
                ligand = base / f"{pdb_id}_ligand.mol2"
                if pocket.exists() and ligand.exists():
                    entries.append((len(entries), pdb_id, str(pocket), str(ligand), None, pkd))
                    break
        if entries:
            print(f"Discovered {len(entries)} complexes from {labels_path}")
            return entries

    # --- Strategy 2: PDBbind index files ---
    # Format: "pdb_code  resolution  year  -logKd/Ki  Kd/Ki=value  //  reference  (ligand)"
    # Column 3 (0-indexed) is the pre-computed -log10(Kd/Ki) value (i.e. pKd directly).
    index_dir = input_dir / "index"
    if index_dir.is_dir():
        index_files = sorted(index_dir.glob("INDEX_*PL_data*")) + sorted(index_dir.glob("INDEX_refined_data*"))
        if index_files:
            index_file = index_files[-1]
            print(f"Reading index: {index_file}")
            with open(index_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    pdb_id = parts[0].lower()
                    # Column 3 is -logKd/Ki (pKd) as a float
                    try:
                        pkd = float(parts[3])
                    except ValueError:
                        # Fallback: try parsing column 4 as "Kd=1.5uM" format
                        pkd = _parse_binding_value(parts[4]) if len(parts) > 4 else None
                    if pkd is None:
                        continue
                    # Look for structure files in multiple possible layouts
                    found = False
                    for base_dir in [input_dir, input_dir / "refined-set", input_dir / "general-set"]:
                        pocket = base_dir / pdb_id / f"{pdb_id}_pocket.pdb"
                        ligand = base_dir / pdb_id / f"{pdb_id}_ligand.mol2"
                        if pocket.exists() and ligand.exists():
                            entries.append((len(entries), pdb_id, str(pocket), str(ligand), None, pkd))
                            found = True
                            break
                    # Also check era-based subdirs (v2020.R1: P-L/1981-2000/, P-L/2001-2010/, etc.)
                    if not found:
                        for era_dir in sorted(input_dir.glob("*/")) :
                            if not era_dir.is_dir():
                                continue
                            pocket = era_dir / pdb_id / f"{pdb_id}_pocket.pdb"
                            ligand = era_dir / pdb_id / f"{pdb_id}_ligand.mol2"
                            if pocket.exists() and ligand.exists():
                                entries.append((len(entries), pdb_id, str(pocket), str(ligand), None, pkd))
                                break
            if entries:
                print(f"Discovered {len(entries)} complexes from {index_file}")
                return entries

    # --- Strategy 3: Directory scan (flat or era-based) ---
    # Scan for PDB code subdirectories: input_dir/<pdb>/ or input_dir/<era>/<pdb>/
    # Need binding data from an external index file
    pkd_map = _scan_for_binding_data(input_dir)
    # Walk up to 3 levels to find index files (e.g. data/v2020r1/P-L → data/refined-set/index/)
    search = input_dir
    for _ in range(3):
        if pkd_map:
            break
        search = search.parent
        if search == search.parent:
            break
        pkd_map = _scan_for_binding_data(search)
        # Also check sibling directories (e.g. data/refined-set/index/ when input is data/v2020r1/P-L)
        if not pkd_map and search.is_dir():
            for sibling in search.iterdir():
                if sibling.is_dir() and sibling != input_dir:
                    pkd_map = _scan_for_binding_data(sibling)
                    if pkd_map:
                        break

    if pkd_map:
        # Direct PDB subdirectories
        for d in sorted(input_dir.iterdir()):
            if not d.is_dir():
                continue
            if len(d.name) == 4:
                pdb_id = d.name.lower()
                pocket = d / f"{pdb_id}_pocket.pdb"
                ligand = d / f"{pdb_id}_ligand.mol2"
                if pocket.exists() and ligand.exists() and pdb_id in pkd_map:
                    entries.append((len(entries), pdb_id, str(pocket), str(ligand), None, pkd_map[pdb_id]))
            else:
                # Era-based subdirectories (v2020.R1: P-L/1981-2000/, P-L/2001-2010/, etc.)
                for sub in sorted(d.iterdir()):
                    if not sub.is_dir() or len(sub.name) != 4:
                        continue
                    pdb_id = sub.name.lower()
                    pocket = sub / f"{pdb_id}_pocket.pdb"
                    ligand = sub / f"{pdb_id}_ligand.mol2"
                    if pocket.exists() and ligand.exists() and pdb_id in pkd_map:
                        entries.append((len(entries), pdb_id, str(pocket), str(ligand), None, pkd_map[pdb_id]))
        if entries:
            print(f"Discovered {len(entries)} complexes from directory scan (using {len(pkd_map)} binding values)")
            return entries

    print(f"ERROR: Could not discover any complexes in {input_dir}")
    print(f"Expected one of:")
    print(f"  - {input_dir}/pdbbind_labels.csv  (labels CSV)")
    print(f"  - {input_dir}/index/INDEX_*_data*  (PDBbind index files)")
    print(f"  - {input_dir}/<pdb_id>/<pdb_id>_pocket.pdb + _ligand.mol2  (flat layout)")
    return entries


def _parse_binding_value(binding_str: str):
    """Parse PDBbind binding data string like 'Kd=1.5uM', 'Ki=100nM', 'IC50=10uM' to pKd.

    Returns -log10(value_in_M) or None if unparseable.
    """
    # Formats: "Kd=1.5uM", "Ki~100nM", "IC50>10uM", "Kd=1.50e-6"
    m = re.match(r'(?:Kd|Ki|IC50)[=~<>]+([0-9.eE+-]+)\s*(fM|pM|nM|uM|mM|M)?', binding_str)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None

    unit = m.group(2) if m.group(2) else 'M'
    multipliers = {'fM': 1e-15, 'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1.0}
    molar = value * multipliers.get(unit, 1.0)
    if molar <= 0:
        return None
    return -math.log10(molar)


def _scan_for_binding_data(root: Path) -> dict:
    """Scan directory tree for PDBbind index files, return {pdb_id: pKd} map."""
    pkd_map = {}
    for search_dir in [root, root / "index", root / "readme"]:
        if not search_dir.is_dir():
            continue
        for f in search_dir.iterdir():
            if f.name.startswith("INDEX_") and "data" in f.name.lower():
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            pdb_id = parts[0].lower()
                            # Column 3 is -logKd/Ki (pKd) as float
                            try:
                                pkd = float(parts[3])
                            except ValueError:
                                pkd = _parse_binding_value(parts[4]) if len(parts) > 4 else None
                            if pkd is not None:
                                pkd_map[pdb_id] = pkd
    return pkd_map


def main():
    parser = argparse.ArgumentParser(description="Precompute DruseScore-pKi features + perturbations")
    parser.add_argument("--input", type=str, required=True,
                        help="PDBbind data directory (contains complex folders + index/labels)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output cache directory for precomputed .pt files")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers (0=auto)")
    parser.add_argument("--labels_csv", type=str, default="pdbbind_labels.csv",
                        help="Labels CSV filename (looked for inside --input)")
    args = parser.parse_args()

    print(f"Atom features: {_NUM_ATOM_FEATURES}-dim (from train_druse_pKi_v2)")

    input_dir = Path(args.input)
    cache_dir = Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_complexes(input_dir, args.labels_csv)
    if not entries:
        return

    # Fill in cache_dir for all entries
    entries = [(idx, pdb_id, pocket, ligand, str(cache_dir), pkd)
               for idx, pdb_id, pocket, ligand, _, pkd in entries]

    n_perturbs = len(_PERTURBATION_RMSD_TARGETS)
    print(f"Found {len(entries)} complexes")
    print(f"Generating {n_perturbs} perturbations per complex (RMSD targets: {_PERTURBATION_RMSD_TARGETS})")
    print(f"Expected output: ~{len(entries) * n_perturbs} cached .pt files")
    print(f"Pose confidence sigma: {_POSE_CONFIDENCE_SIGMA}")
    print(f"Output: {cache_dir}")

    n_workers = args.workers if args.workers > 0 else max(cpu_count() - 2, 1)
    print(f"Using {n_workers} workers\n")

    total_saved = 0
    total_failed = 0

    with Pool(n_workers) as pool:
        for pdb_id, n_saved, msg in tqdm(pool.imap_unordered(process_one, entries),
                                          total=len(entries), desc="Precomputing"):
            if n_saved > 0:
                total_saved += n_saved
            else:
                total_failed += 1

    print(f"\nDone: {total_saved} samples cached, {total_failed} complexes failed")
    print(f"Output: {cache_dir} ({total_saved} .pt files)")

    # Verify distribution
    print(f"\nPerturbation distribution:")
    for pi, rmsd_t in enumerate(_PERTURBATION_RMSD_TARGETS):
        count = len(list(cache_dir.glob(f"*_p{pi}.pt")))
        conf = _rmsd_to_confidence(rmsd_t)
        print(f"  RMSD ~{rmsd_t:5.1f} A  |  confidence={conf:.3f}  |  {count} samples")

    print(f"\nNow train with: python train_druse_pKi_v2.py --input {cache_dir} --output checkpoints_pki_v2/ --epochs 80")


if __name__ == "__main__":
    main()

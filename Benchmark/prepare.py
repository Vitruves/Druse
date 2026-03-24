#!/usr/bin/env python3
"""
Prepare CASF-2016 benchmark manifest from PDBbind refined-set data.

Parses ligand SDF files with RDKit to extract SMILES and crystal heavy atom
positions, pairs with pKd from labels CSV, writes a JSON manifest for the
Swift BenchmarkRunner.

Usage:
  python Benchmark/prepare.py
"""

import json
import csv
import argparse
from pathlib import Path

try:
    import numpy as np
    from rdkit import Chem
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with: uv pip install numpy rdkit --python /Users/vitruves/Developer/Tools/py310/bin/python"
    )

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DATA = ROOT / "Benchmark" / "data"   # CASF-2016 extracted here
TRAINING_DATA = ROOT / "Training" / "data"       # PDBbind refined-set here
MANIFEST_DIR = ROOT / "Benchmark" / "manifests"


def extract_heavy_positions(mol):
    """Extract heavy atom (non-H) 3D positions from an RDKit Mol."""
    if mol is None or mol.GetNumConformers() == 0:
        return []
    all_pos = mol.GetConformer().GetPositions()
    return [
        [round(float(all_pos[a.GetIdx()][0]), 3),
         round(float(all_pos[a.GetIdx()][1]), 3),
         round(float(all_pos[a.GetIdx()][2]), 3)]
        for a in mol.GetAtoms() if a.GetAtomicNum() > 1
    ]


def mol_to_smiles(mol):
    """Canonical SMILES without Hs."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def get_casf_core_ids():
    """Get CASF-2016 core set PDB IDs + pKd values.

    Tries the official CoreSet.dat first, falls back to pdbbind_labels.csv.
    """
    import os

    # Walk CASF-2016 directory looking for CoreSet.dat
    casf_dir = BENCHMARK_DATA / "CASF-2016"
    if casf_dir.exists():
        for root, _, files in os.walk(casf_dir):
            for fname in files:
                if "CoreSet" in fname and fname.endswith(".dat"):
                    entries = {}
                    with open(Path(root) / fname) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    entries[parts[0].lower()] = float(parts[3])
                                except ValueError:
                                    continue
                    if entries:
                        print(f"  Loaded {len(entries)} CASF core set entries from {fname}")
                        return entries

    # Fallback: all refined-set entries from labels CSV
    labels_csv = TRAINING_DATA / "pdbbind_labels.csv"
    if labels_csv.exists():
        entries = {}
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                if row.get("split") == "refined":
                    entries[row["pdb_id"].lower()] = float(row["pKd"])
        print(f"  Loaded {len(entries)} refined-set entries from pdbbind_labels.csv (CASF index not found)")
        return entries

    return {}


def prepare_manifest():
    """Build CASF-2016 benchmark manifest."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    refined_dir = TRAINING_DATA / "refined-set"

    if not refined_dir.exists():
        print("  [ERROR] refined-set not found. Run: python Benchmark/download.py --setup")
        return False

    core_set = get_casf_core_ids()
    if not core_set:
        print("  [ERROR] Could not find PDB IDs with pKd values")
        return False

    complexes = []
    skipped = 0

    for pdb_id, pKd in sorted(core_set.items()):
        pdb_dir = refined_dir / pdb_id
        protein_pdb = pdb_dir / f"{pdb_id}_protein.pdb"
        pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
        ligand_sdf = pdb_dir / f"{pdb_id}_ligand.sdf"

        if not protein_pdb.exists() or not ligand_sdf.exists():
            skipped += 1
            continue

        mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False, sanitize=False)

        smiles = mol_to_smiles(mol)
        crystal_pos = extract_heavy_positions(mol)
        heavy_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1) if mol else 0

        if not smiles or not crystal_pos:
            skipped += 1
            continue

        complexes.append({
            "pdb_id": pdb_id,
            "protein_pdb": str(protein_pdb.relative_to(ROOT)),
            "pocket_pdb": str(pocket_pdb.relative_to(ROOT)) if pocket_pdb.exists() else None,
            "ligand_sdf": str(ligand_sdf.relative_to(ROOT)),
            "smiles": smiles,
            "crystal_positions": crystal_pos,
            "heavy_atom_count": heavy_count,
            "pKd": round(pKd, 2),
        })

    manifest = {
        "benchmark": "casf-2016",
        "description": f"CASF-2016: {len(complexes)} protein-ligand complexes with known pKd",
        "complexes": complexes,
    }

    out_path = MANIFEST_DIR / "casf_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote {len(complexes)} complexes to {out_path.relative_to(ROOT)}")
    if skipped:
        print(f"  Skipped {skipped} (missing files or RDKit parse failure)")

    pKds = [c["pKd"] for c in complexes]
    heavys = [c["heavy_atom_count"] for c in complexes]
    print(f"  pKd range: {min(pKds):.2f} \u2013 {max(pKds):.2f} (mean {np.mean(pKds):.2f})")
    print(f"  Heavy atoms: {min(heavys)} \u2013 {max(heavys)} (mean {np.mean(heavys):.0f})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare CASF-2016 benchmark manifest")
    parser.add_argument("--verify", action="store_true", help="Just check if manifest exists and print stats")
    args = parser.parse_args()

    print("=== Druse Benchmark Manifest Preparation ===\n")

    if args.verify:
        mf = MANIFEST_DIR / "casf_manifest.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
            print(f"  Manifest: {len(m['complexes'])} complexes")
        else:
            print(f"  Manifest not found at {mf.relative_to(ROOT)}")
        return

    prepare_manifest()
    print("\nDone.")


if __name__ == "__main__":
    main()

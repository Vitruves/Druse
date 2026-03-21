#!/usr/bin/env python3
"""
Download and prepare training data for Druse ML models.

Datasets:
  - PDBbind v2020 refined (4,852 complexes) — primary affinity training
  - PDBbind v2020 general (14,127 complexes) — additional affinity data
  - CASF-2016 core set (285 complexes) — benchmark
  - CrossDocked2020 poses — pose discrimination (optional, large)

Usage:
  python download_data.py --all
  python download_data.py --pdbbind --casf
"""

import os
import sys
import argparse
import requests
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"

PDBBIND_REFINED_URL = "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz"
PDBBIND_GENERAL_URL = "http://www.pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz"
CASF_URL = "http://www.pdbbind.org.cn/download/CASF-2016.tar.gz"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {desc or url}...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_tar(path: Path, dest: Path):
    """Extract a tar.gz archive."""
    print(f"  Extracting {path.name}...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest)


def download_pdbbind(refined=True, general=False):
    """Download PDBbind datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if refined:
        dest = DATA_DIR / "PDBbind_v2020_refined.tar.gz"
        download_file(PDBBIND_REFINED_URL, dest, "PDBbind v2020 refined (4,852 complexes)")
        if not (DATA_DIR / "refined-set").exists():
            extract_tar(dest, DATA_DIR)

    if general:
        dest = DATA_DIR / "PDBbind_v2020_general.tar.gz"
        download_file(PDBBIND_GENERAL_URL, dest, "PDBbind v2020 general (14,127 complexes)")
        if not (DATA_DIR / "v2020-other-PL").exists():
            extract_tar(dest, DATA_DIR)


def download_casf():
    """Download CASF-2016 benchmark set."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "CASF-2016.tar.gz"
    download_file(CASF_URL, dest, "CASF-2016 core set (285 complexes)")
    if not (DATA_DIR / "CASF-2016").exists():
        extract_tar(dest, DATA_DIR)


def parse_pdbbind_index(index_path: Path) -> dict:
    """Parse PDBbind INDEX file → {pdb_id: pKd}."""
    data = {}
    if not index_path.exists():
        return data
    with open(index_path) as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) < 4:
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()
                try:
                    # Format: "Kd=1.2uM" or "Ki=3.4nM" → parse the numeric value
                    affinity_str = parts[3]
                    # Extract the -logKd value (column 4 in refined index)
                    pkd = float(parts[3])
                    data[pdb_id] = pkd
                except (ValueError, IndexError):
                    continue
    return data


def prepare_labels():
    """Create a unified labels CSV from PDBbind indices."""
    import pandas as pd

    refined_index = DATA_DIR / "refined-set" / "index" / "INDEX_refined_data.2020"
    general_index = DATA_DIR / "v2020-other-PL" / "index" / "INDEX_general_PL_data.2020"

    records = []

    for index_path, split in [(refined_index, "refined"), (general_index, "general")]:
        if not index_path.exists():
            print(f"  [skip] {index_path} not found")
            continue
        with open(index_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    pdb_id = parts[0].lower()
                    resolution = parts[1] if parts[1] != "NMR" else "0.0"
                    year = parts[2] if len(parts) > 2 else ""
                    try:
                        pkd = float(parts[3])
                    except ValueError:
                        continue
                    records.append({
                        "pdb_id": pdb_id,
                        "resolution": resolution,
                        "year": year,
                        "pKd": pkd,
                        "split": split
                    })

    if records:
        df = pd.DataFrame(records)
        out = DATA_DIR / "pdbbind_labels.csv"
        df.to_csv(out, index=False)
        print(f"  Wrote {len(df)} labels to {out}")
    else:
        print("  No labels found — download PDBbind first")


def main():
    parser = argparse.ArgumentParser(description="Download Druse training data")
    parser.add_argument("--pdbbind", action="store_true", help="Download PDBbind refined set")
    parser.add_argument("--pdbbind-general", action="store_true", help="Download PDBbind general set")
    parser.add_argument("--casf", action="store_true", help="Download CASF-2016 benchmark")
    parser.add_argument("--labels", action="store_true", help="Prepare unified labels CSV")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if args.all:
        args.pdbbind = args.pdbbind_general = args.casf = args.labels = True

    if not any([args.pdbbind, args.pdbbind_general, args.casf, args.labels]):
        parser.print_help()
        return

    print("=== Druse Training Data Download ===\n")

    if args.pdbbind:
        print("[1/4] PDBbind v2020 refined set")
        download_pdbbind(refined=True, general=False)

    if args.pdbbind_general:
        print("[2/4] PDBbind v2020 general set")
        download_pdbbind(refined=False, general=True)

    if args.casf:
        print("[3/4] CASF-2016 benchmark")
        download_casf()

    if args.labels:
        print("[4/4] Preparing labels")
        prepare_labels()

    print("\nDone. Data saved to:", DATA_DIR)


if __name__ == "__main__":
    main()

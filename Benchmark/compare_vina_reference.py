#!/usr/bin/env python3
"""
Compare Druse (Vina/Drusina) against reference AutoDock-Vina 1.2.7 on CASF-2016.

Runs real AutoDock-Vina CLI on the same complexes as the Druse benchmark,
using PDBQT files prepared with meeko/obabel.

Usage:
  python Benchmark/compare_vina_reference.py --quick 30
  python Benchmark/compare_vina_reference.py --druse-result Benchmark/results/<file>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
REFINED_SET = ROOT / "Benchmark" / "data" / "refined-set"
INDEX_FILE = REFINED_SET / "index" / "INDEX_refined_data.2020"
VINA_BIN = ROOT / "temp" / "AutoDock-Vina" / "build" / "mac" / "release" / "vina"
RESULTS_DIR = ROOT / "Benchmark" / "results"

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required"); sys.exit(1)


def parse_index(index_path: Path) -> dict[str, float]:
    pkd_map = {}
    with open(index_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    pkd_map[parts[0].strip()] = float(parts[3])
                except ValueError:
                    pass
    return pkd_map


def sdf_to_pdbqt(sdf_path: str, out_path: str) -> bool:
    """Convert SDF to PDBQT using obabel."""
    try:
        r = subprocess.run(
            ["/opt/homebrew/bin/obabel", sdf_path, "-O", out_path],
            capture_output=True, timeout=30
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def pdb_to_pdbqt(pdb_path: str, out_path: str) -> bool:
    """Convert protein PDB to PDBQT using obabel."""
    try:
        r = subprocess.run(
            ["/opt/homebrew/bin/obabel", pdb_path, "-O", out_path, "-xr"],
            capture_output=True, timeout=60
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def get_ligand_center_and_size(sdf_path: str) -> tuple[list[float], list[float]] | None:
    """Get bounding box center and size from SDF."""
    if HAS_RDKIT:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
        mol = next(suppl, None)
        if mol is None:
            return None
        conf = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    else:
        # Fallback: parse SDF manually
        positions = []
        with open(sdf_path) as f:
            lines = f.readlines()
        if len(lines) < 4:
            return None
        try:
            n_atoms = int(lines[3][:3])
        except ValueError:
            return None
        for i in range(4, 4 + n_atoms):
            parts = lines[i].split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
        positions = np.array(positions)

    if len(positions) == 0:
        return None
    center = positions.mean(axis=0).tolist()
    extent = positions.max(axis=0) - positions.min(axis=0)
    size = (extent + 10.0).tolist()  # 10Å padding
    return center, size


def run_vina_single(args_tuple):
    """Run AutoDock-Vina on a single complex. Returns (pdb_id, score, rmsd, time_s)."""
    pdb_id, prot_pdb, lig_sdf, pkd = args_tuple

    with tempfile.TemporaryDirectory(prefix=f"vina_{pdb_id}_") as tmpdir:
        prot_pdbqt = os.path.join(tmpdir, "protein.pdbqt")
        lig_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        out_pdbqt = os.path.join(tmpdir, "out.pdbqt")

        # Prepare inputs
        if not pdb_to_pdbqt(prot_pdb, prot_pdbqt):
            return (pdb_id, None, None, 0, "prot_prep_fail")
        if not sdf_to_pdbqt(lig_sdf, lig_pdbqt):
            return (pdb_id, None, None, 0, "lig_prep_fail")

        # Get search box
        box = get_ligand_center_and_size(lig_sdf)
        if box is None:
            return (pdb_id, None, None, 0, "box_fail")
        center, size = box

        # Run Vina
        t0 = time.time()
        try:
            cmd = [
                str(VINA_BIN),
                "--receptor", prot_pdbqt,
                "--ligand", lig_pdbqt,
                "--center_x", f"{center[0]:.3f}",
                "--center_y", f"{center[1]:.3f}",
                "--center_z", f"{center[2]:.3f}",
                "--size_x", f"{size[0]:.1f}",
                "--size_y", f"{size[1]:.1f}",
                "--size_z", f"{size[2]:.1f}",
                "--exhaustiveness", "8",
                "--out", out_pdbqt,
                "--num_modes", "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0

            # Parse score from output
            score = None
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and line[0:1].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mode = int(parts[0])
                            if mode == 1:
                                score = float(parts[1])
                                break
                        except ValueError:
                            pass

            if score is None:
                # Try parsing from stderr
                for line in result.stderr.split('\n'):
                    if 'Affinity' in line or 'kcal' in line:
                        parts = line.split()
                        for p in parts:
                            try:
                                score = float(p)
                                break
                            except ValueError:
                                pass

            return (pdb_id, score, pkd, elapsed, "ok" if score else "parse_fail")

        except subprocess.TimeoutExpired:
            return (pdb_id, None, None, 120, "timeout")
        except Exception as e:
            return (pdb_id, None, None, 0, str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", type=int, default=0, help="Limit to first N complexes")
    parser.add_argument("--druse-result", type=str, help="Druse benchmark JSON to compare against")
    parser.add_argument("--match-druse", action="store_true",
                        help="Only run on PDB IDs present in --druse-result")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Check prerequisites
    if not VINA_BIN.exists():
        print(f"ERROR: vina binary not found at {VINA_BIN}")
        sys.exit(1)

    obabel_check = subprocess.run(["/opt/homebrew/bin/obabel", "-V"], capture_output=True)
    if obabel_check.returncode != 0:
        print("ERROR: obabel not found. Install: brew install open-babel")
        sys.exit(1)

    # Load index
    pkd_map = parse_index(INDEX_FILE)
    print(f"Loaded {len(pkd_map)} complexes from PDBbind index")

    # Build work list
    work = []
    for pdb_id, pkd in sorted(pkd_map.items()):
        prot = REFINED_SET / pdb_id / f"{pdb_id}_protein.pdb"
        lig = REFINED_SET / pdb_id / f"{pdb_id}_ligand.sdf"
        if prot.exists() and lig.exists():
            work.append((pdb_id, str(prot), str(lig), pkd))

    # Filter to match Druse result if requested
    if args.match_druse and args.druse_result:
        druse_ids = {e['pdb_id'] for e in json.load(open(args.druse_result))['entries']}
        work = [w for w in work if w[0] in druse_ids]
        print(f"  Matched {len(work)} complexes from Druse result")

    if args.quick > 0:
        work = work[:args.quick]

    print(f"Running AutoDock-Vina 1.2.7 on {len(work)} complexes ({args.workers} workers)")
    print()

    # Run
    results = []
    errors = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_vina_single, w): w[0] for w in work}
        for i, future in enumerate(as_completed(futures)):
            pdb_id, score, pkd, elapsed, status = future.result()
            if status == "ok" and score is not None:
                results.append({'pdb_id': pdb_id, 'score': score, 'pkd': pkd, 'time': elapsed})
            else:
                errors += 1

            done = i + 1
            if done % 10 == 0 or done == len(work):
                wall = time.time() - t0
                ok = len(results)
                print(f"  [{done}/{len(work)}]  ok={ok}  errors={errors}  "
                      f"{wall:.0f}s elapsed  {done/wall:.1f} cpx/s")

    total_time = time.time() - t0
    print(f"\nCompleted: {len(results)}/{len(work)} in {total_time:.0f}s, {errors} errors")

    if not results:
        print("No results — check obabel installation")
        sys.exit(1)

    # Scoring power
    from scipy.stats import pearsonr
    scores = np.array([r['score'] for r in results])
    pkds = np.array([r['pkd'] for r in results])
    r_vina, _ = pearsonr(scores, pkds)
    avg_time = np.mean([r['time'] for r in results])

    print(f"\n{'='*50}")
    print(f"  AutoDock-Vina 1.2.7 Reference")
    print(f"{'='*50}")
    print(f"  Complexes scored: {len(results)}")
    print(f"  Scoring Power: Pearson r = {r_vina:.4f}")
    print(f"  Avg time/complex: {avg_time:.1f}s")
    print(f"  Mean score: {np.mean(scores):.2f} kcal/mol")

    # Compare with Druse if provided
    if args.druse_result:
        druse = json.load(open(args.druse_result))
        druse_map = {e['pdb_id']: e for e in druse['entries']}
        method = druse.get('scoringMethod', '?')

        # Match complexes
        common_ids = [r['pdb_id'] for r in results if r['pdb_id'] in druse_map]
        print(f"\n  Common complexes: {len(common_ids)} (Vina ref: {len(results)}, Druse: {len(druse_map)})")
        if len(common_ids) >= 2:
            vina_ref = np.array([next(r['score'] for r in results if r['pdb_id'] == pid) for pid in common_ids])
            vina_ref_pkd = np.array([next(r['pkd'] for r in results if r['pdb_id'] == pid) for pid in common_ids])
            druse_scores = np.array([druse_map[pid]['best_energy'] for pid in common_ids])
            druse_rmsds = np.array([druse_map[pid].get('best_rmsd', 99) for pid in common_ids])

            r_ref, _ = pearsonr(vina_ref, vina_ref_pkd)
            r_druse, _ = pearsonr(druse_scores, vina_ref_pkd)

            print(f"\n  Comparison on {len(common_ids)} common complexes:")
            print(f"  {'AutoDock-Vina 1.2.7':>25}  Pearson r = {r_ref:.4f}")
            print(f"  {'Druse ' + method:>25}  Pearson r = {r_druse:.4f}")
            if 'best_rmsd' in druse['entries'][0]:
                docking_ok = sum(1 for pid in common_ids if druse_map[pid].get('best_rmsd', 99) < 2.0)
                print(f"  Druse docking power: {docking_ok}/{len(common_ids)} (<2Å)")

    # Save results
    out_path = RESULTS_DIR / f"autodock_vina_reference_{len(results)}cpx.json"
    with open(out_path, 'w') as f:
        json.dump({
            'method': 'AutoDock-Vina 1.2.7',
            'n_complexes': len(results),
            'pearson_r': r_vina,
            'avg_time_s': avg_time,
            'results': results,
        }, f, indent=2)
    print(f"\n  Results: {out_path}")


if __name__ == "__main__":
    main()

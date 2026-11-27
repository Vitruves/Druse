#!/usr/bin/env python3
"""Known-pocket redocking panel: Druse vs AutoDock Vina on a frozen complex set.

Inputs are rebuilt from connectivity only (ETKDGv3 + UFF, crystal coordinates discarded),
so a run measures search, not memorisation. The receptor is the deposited PDB with waters
stripped; the box is a fixed cube centred on the mean of the deposited pocket atoms. The
metric is symmetry-aware heavy-atom RMSD of the top-ranked pose in the receptor frame, with
no alignment and no best-of-N selection.

    python Benchmark/redock_panel.py --engine vina  --variant vina
    python Benchmark/redock_panel.py --engine druse --variant base
    python Benchmark/redock_panel.py --engine druse --variant deep \
        --dock-set "remc-steps=1500 ls-frequency=1"
    python Benchmark/redock_panel.py --report base deep vina

Per-trial results are cached under Benchmark/results/redock_panel/<variant>/ so a variant can
be extended or resumed without recomputing finished trials.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
REFINED = ROOT / "Benchmark/data/refined-set"
WORK = ROOT / "Benchmark/results/redock_panel"
VINA = ROOT / "temp/AutoDock-Vina/build/mac/release/vina"

BOX_EDGE = 24.0
SEEDS = (17, 29)
EMBED_SEED = 2026

# Frozen panel. "dev" carries recorded baselines from the 2026-11-26 REMC reports and is
# what tuning may look at; "val" is held out from tuning. Selection of the ten additions is
# reproducible via --select.
DEV = [
    "10gs", "1bnt", "1c5o", "1nm6", "1nz7", "1qkt", "1rbp", "1xow",
    "2pog", "2xc4", "3m3x", "4bck", "5cap", "5caq", "5hu9", "5l7g",
]
VAL = [
    "5jq5", "1azm", "2vh0", "4mgd",
    "4ou3", "4zow", "1k21", "4x5p", "3mv0", "1b5h", "6phx", "1c5c", "1ql7", "4b6o",
]
PANEL = DEV + VAL


# ---------------------------------------------------------------------------- selection


def select_additions(count: int, exclude: set[str]) -> list[str]:
    """Reproducible frozen choice: smallest SHA256("redock30-v1/<pdb>") among usable entries."""
    usable = []
    for folder in sorted(REFINED.iterdir()):
        pdb = folder.name
        if pdb in exclude or not folder.is_dir():
            continue
        if not all((folder / f"{pdb}_{part}").exists()
                   for part in ("protein.pdb", "ligand.sdf", "pocket.pdb")):
            continue
        usable.append(pdb)
    ranked = sorted(usable, key=lambda p: hashlib.sha256(f"redock30-v1/{p}".encode()).hexdigest())
    picked = []
    for pdb in ranked:  # an unreadable crystal ligand gives nothing to score against
        if len(picked) == count:
            break
        if Chem.SDMolSupplier(str(REFINED / pdb / f"{pdb}_ligand.sdf"))[0] is not None:
            picked.append(pdb)
    return picked


# ---------------------------------------------------------------------------- preparation


def prepare(pdb: str) -> dict:
    """Build receptor, reconstructed ligand conformer and box for one complex. Cached on disk."""
    src = REFINED / pdb
    folder = WORK / "inputs" / pdb
    folder.mkdir(parents=True, exist_ok=True)

    ref = Chem.SDMolSupplier(str(src / f"{pdb}_ligand.sdf"))[0]
    if ref is None:
        raise RuntimeError(f"{pdb}: crystal ligand did not parse")

    ligand = folder / "input.sdf"
    if not ligand.exists():
        mol = Chem.AddHs(ref)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = EMBED_SEED
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"{pdb}: ETKDGv3 embedding failed")
        AllChem.UFFOptimizeMolecule(mol)
        Chem.MolToMolFile(mol, str(ligand))

    graph = Chem.RemoveAllHs(Chem.SDMolSupplier(str(ligand))[0])
    crystal = Chem.RemoveAllHs(ref)
    if graph.GetNumAtoms() != crystal.GetNumAtoms():
        raise RuntimeError(f"{pdb}: rebuilt ligand has {graph.GetNumAtoms()} heavy atoms "
                           f"vs {crystal.GetNumAtoms()} in the crystal")

    receptor = folder / "protein.pdb"
    if not receptor.exists():
        lines = (src / f"{pdb}_protein.pdb").read_text().splitlines(keepends=True)
        receptor.write_text("".join(l for l in lines if l[17:20].strip() not in {"HOH", "WAT", "DOD"}))

    pocket = [l for l in (src / f"{pdb}_pocket.pdb").read_text().splitlines() if l.startswith("ATOM")]
    center = np.array([[float(l[s:s + 8]) for s in (30, 38, 46)] for l in pocket]).mean(axis=0)

    return {
        "pdb": pdb, "folder": folder, "receptor": receptor, "ligand": ligand,
        "graph": graph, "crystal": crystal,
        "center": [f"{v:.3f}" for v in center],
    }


def rmsd(graph: Chem.Mol, crystal: Chem.Mol, coords: np.ndarray) -> float:
    pose = Chem.Mol(graph)
    for index, position in enumerate(coords):
        pose.GetConformer().SetAtomPosition(index, position.tolist())
    return float(rdMolAlign.CalcRMS(pose, crystal))


def run(args: list[str], log: Path, timeout: int) -> float:
    start = time.monotonic()
    with open(log, "w") as handle:
        subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return time.monotonic() - start


# ---------------------------------------------------------------------------- engines


def pdbqt_atoms(path: Path) -> dict[int, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line.split()[-1] not in ("H", "HD", "HS"):
            out[int(line[6:11])] = np.array([float(line[s:s + 8]) for s in (30, 38, 46)])
    return out


def dock_vina(case: dict, seed: int, exhaustiveness: int, timeout: int) -> tuple[float, float]:
    folder, graph = case["folder"], case["graph"]
    receptor_qt, ligand_qt = folder / "receptor.pdbqt", folder / "input.pdbqt"
    for source, dest, flags in ((case["receptor"], receptor_qt, ["-xr"]), (case["ligand"], ligand_qt, [])):
        if not dest.exists():
            run(["obabel", str(source), "-O", str(dest), *flags], folder / f"{dest.name}.log", timeout)
    # Open Babel emits ROOT/BRANCH records for -xr output that Vina rejects; keep coordinates only.
    text = receptor_qt.read_text()
    kept = "".join(l for l in text.splitlines(keepends=True) if l.startswith(("ATOM", "HETATM")))
    if kept != text:
        receptor_qt.write_text(kept)

    result = folder / f"vina-e{exhaustiveness}-{seed}.pdbqt"
    args = [str(VINA), "--receptor", str(receptor_qt), "--ligand", str(ligand_qt),
            "--num_modes", "1", "--seed", str(seed), "--cpu", "1",
            "--exhaustiveness", str(exhaustiveness), "--out", str(result)]
    for axis, value in zip("xyz", case["center"]):
        args += [f"--center_{axis}", value, f"--size_{axis}", str(BOX_EDGE)]
    elapsed = run(args, folder / f"vina-e{exhaustiveness}-{seed}.log", timeout)

    inputs, outputs = pdbqt_atoms(ligand_qt), pdbqt_atoms(result)
    if inputs.keys() != outputs.keys() or len(inputs) != graph.GetNumAtoms():
        raise RuntimeError(f"{case['pdb']}: Vina atom set does not match the input ligand")
    original = graph.GetConformer().GetPositions()
    mapping = {serial: int(np.linalg.norm(original - pos, axis=1).argmin()) for serial, pos in inputs.items()}
    if len(set(mapping.values())) != len(inputs):
        raise RuntimeError(f"{case['pdb']}: Vina atom mapping is not injective")
    if not all(np.linalg.norm(original[i] - inputs[s]) < 3e-3 for s, i in mapping.items()):
        raise RuntimeError(f"{case['pdb']}: Vina input coordinates drifted from the SDF")
    coords = original.copy()
    for serial, index in mapping.items():
        coords[index] = outputs[serial]
    return rmsd(graph, case["crystal"], coords), elapsed


def dock_druse(case: dict, seed: int, app: Path, dock_set: str, poses: int,
               timeout: int) -> tuple[list[float], float]:
    """Return the RMSD of every exported pose, rank 1 first, and the elapsed wall time."""
    folder, graph = case["folder"], case["graph"]
    tag = f"druse-{seed}"
    result = folder / f"{tag}.sdf"
    result.unlink(missing_ok=True)
    center = ",".join(case["center"])
    settings = ("scoring=vina search=remc auto-tune=off flex-auto=off "
                f"population=128 remc-replicas=8 remc-swap=4 seed={seed} {dock_set}").strip()
    script = (f"load {case['receptor']}; load {case['ligand']}; "
              f"pocket box center={center} size={BOX_EDGE},{BOX_EDGE},{BOX_EDGE}; "
              f"dock set {settings}; dock; export poses {result} top={poses}")
    elapsed = run([str(app / "Contents/MacOS/Druse"), "--headless", "--log-level", "info", "--eval", script],
                  folder / f"{tag}.log", timeout)

    exported = [m for m in Chem.SDMolSupplier(str(result), sanitize=False, removeHs=False) if m is not None]
    if not exported:
        raise RuntimeError(f"{case['pdb']}: Druse exported no parseable pose")
    symbols = [a.GetSymbol() for a in graph.GetAtoms()]
    if [a.GetSymbol() for a in exported[0].GetAtoms()] != symbols:
        raise RuntimeError(f"{case['pdb']}: Druse pose atom order does not match the input ligand")
    return [rmsd(graph, case["crystal"], m.GetConformer().GetPositions()) for m in exported], elapsed


# ---------------------------------------------------------------------------- driver


def trial(case: dict, seed: int, opts: argparse.Namespace, app: Path) -> dict:
    record = {"pdb": case["pdb"], "seed": seed}
    try:
        if opts.engine == "vina":
            value, elapsed = dock_vina(case, seed, opts.exhaustiveness, opts.timeout)
            record |= {"rmsd": round(value, 3), "elapsed_s": round(elapsed, 2)}
        else:
            values, elapsed = dock_druse(case, seed, app, opts.dock_set, opts.poses, opts.timeout)
            # rank 1 measures the engine; the minimum over exported poses separates a search
            # that never found the pose from a search that found it and ranked it below another.
            record |= {"rmsd": round(values[0], 3), "best_rmsd": round(min(values), 3),
                       "poses": len(values), "elapsed_s": round(elapsed, 2)}
    except Exception as exc:  # a failed trial is a result: record it, never silently drop it
        record |= {"rmsd": None, "error": f"{type(exc).__name__}: {exc}"}
    return record


def summarise(records: list[dict]) -> dict:
    solved = [r["rmsd"] for r in records if r.get("rmsd") is not None]
    oracle = [r["best_rmsd"] for r in records if r.get("best_rmsd") is not None]
    return {
        "any_under_2": sum(1 for v in oracle if v < 2.0) if oracle else None,
        "n": len(records),
        "failed": sum(1 for r in records if r.get("rmsd") is None),
        "under_2A": sum(1 for v in solved if v < 2.0),
        "under_3A": sum(1 for v in solved if v < 3.0),
        "median": round(statistics.median(solved), 3) if solved else None,
        "median_s": round(statistics.median([r["elapsed_s"] for r in records if "elapsed_s" in r]), 2)
        if any("elapsed_s" in r for r in records) else None,
    }


def load(variant: str) -> dict[tuple[str, int], dict]:
    folder = WORK / variant
    out = {}
    if folder.exists():
        for path in folder.glob("*.json"):
            record = json.loads(path.read_text())
            out[(record["pdb"], record["seed"])] = record
    return out


def report(variants: list[str], panel: list[str]) -> None:
    data = {v: load(v) for v in variants}
    width = max(len(v) for v in variants)
    print(f"\n{'pdb':<6} {'seed':<5} " + " ".join(f"{v:>{max(width, 8)}}" for v in variants))
    for pdb in panel:
        for seed in SEEDS:
            cells = []
            for v in variants:
                record = data[v].get((pdb, seed))
                value = record.get("rmsd") if record else None
                cells.append(f"{value:>{max(width, 8)}.3f}" if value is not None
                             else f"{'FAIL' if record else '-':>{max(width, 8)}}")
            print(f"{pdb:<6} {seed:<5} " + " ".join(cells))

    for label, subset in (("dev", DEV), ("val", VAL), ("all", panel)):
        members = [p for p in subset if p in panel]
        if not members:
            continue
        print(f"\n--- {label} ({len(members)} complexes) ---")
        for v in variants:
            records = [data[v][(p, s)] for p in members for s in SEEDS if (p, s) in data[v]]
            if not records:
                continue
            s = summarise(records)
            print(f"  {v:<{width}}  <2A {s['under_2A']:>3}/{s['n']:<3}  <3A {s['under_3A']:>3}/{s['n']:<3}  "
                  f"median {s['median']!s:>7}  median_s {s['median_s']!s:>6}"
                  + (f"  sampled<2A {s['any_under_2']:>3}" if s["any_under_2"] is not None else "")
                  + (f"  FAILED {s['failed']}" if s["failed"] else ""))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("druse", "vina"))
    parser.add_argument("--variant", help="cache + report label for this configuration")
    parser.add_argument("--dock-set", default="", help="extra 'dock set' key=value pairs for Druse")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness (default 8)")
    parser.add_argument("--poses", type=int, default=20,
                        help="poses Druse exports; the minimum over them is the sampling diagnostic")
    parser.add_argument("--config", default="Release", choices=("Debug", "Release"))
    parser.add_argument("--only", help="comma-separated PDB subset")
    parser.add_argument("--group", choices=("dev", "val", "all"), default="all")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--jobs", type=int, default=1, help="concurrent trials (>1 distorts timings)")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--redo", action="store_true", help="recompute trials already cached")
    parser.add_argument("--report", nargs="+", metavar="VARIANT", help="print a comparison table and exit")
    parser.add_argument("--select", type=int, metavar="N", help="print N reproducible panel additions and exit")
    opts = parser.parse_args()

    panel = {"dev": DEV, "val": VAL, "all": PANEL}[opts.group]
    if opts.only:
        wanted = [p.strip() for p in opts.only.split(",") if p.strip()]
        panel = [p for p in wanted if p in panel] or wanted

    if opts.select:
        print(" ".join(select_additions(opts.select, set(PANEL))))
        return 0
    if opts.report:
        report(opts.report, panel)
        return 0
    if not opts.engine or not opts.variant:
        parser.error("--engine and --variant are required unless --report or --select is used")

    app = ROOT / f"build/Build/Products/{opts.config}/Druse.app"
    if opts.engine == "druse" and not app.exists():
        print(f"missing app: {app}", file=sys.stderr)
        return 2
    if opts.engine == "vina" and not VINA.exists():
        print(f"missing vina: {VINA}", file=sys.stderr)
        return 2

    seeds = [int(s) for s in opts.seeds.split(",")]
    cache_dir = WORK / opts.variant
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = {} if opts.redo else load(opts.variant)

    cases = {}
    for pdb in panel:
        try:
            cases[pdb] = prepare(pdb)
        except Exception as exc:
            print(f"{pdb:<6} SKIP   {type(exc).__name__}: {exc}", flush=True)

    pending = [(cases[p], s) for p in panel if p in cases for s in seeds if (p, s) not in done]
    print(f"variant={opts.variant} engine={opts.engine} trials={len(pending)} "
          f"cached={len(done)} jobs={opts.jobs}", flush=True)

    def execute(item):
        case, seed = item
        record = trial(case, seed, opts, app)
        (cache_dir / f"{case['pdb']}-{seed}.json").write_text(json.dumps(record, indent=1))
        value = record.get("rmsd")
        print(f"{record['pdb']:<6} {seed:<5} "
              + (f"{value:8.3f}  {record['elapsed_s']:7.2f}s" if value is not None
                 else f"   FAILED  {record.get('error', '')[:90]}"), flush=True)
        return record

    if opts.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=opts.jobs) as pool:
            list(pool.map(execute, pending))
    else:
        for item in pending:
            execute(item)

    report([opts.variant], panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())

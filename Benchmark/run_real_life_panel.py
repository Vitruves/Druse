#!/usr/bin/env python3
"""
Real-life docking panel — end-to-end blind docking on curated CASF subsets.

Runs the full Druse pipeline (SMILES→3D, ML pocket detection, GA docking)
on 16 complexes grouped by failure mode. Unlike --casf redocking, this tests
what a real user would experience.

Extra args after -- are forwarded to run_benchmark.py (e.g. --population 300).

Examples:
  python Benchmark/run_real_life_panel.py --scoring vina --preset fast
  python Benchmark/run_real_life_panel.py --scoring vina,drusina --preset standard --debug
  python Benchmark/run_real_life_panel.py --scoring drusina --preset fast --rescoring gfn2
  python Benchmark/run_real_life_panel.py --scoring drusina --preset standard --flex-residues --runs 3
  python Benchmark/run_real_life_panel.py --groups metal_coordination --scoring drusina --preset fast
  python Benchmark/run_real_life_panel.py --list
  python Benchmark/run_real_life_panel.py --skip-run --debug --results Benchmark/results/some_result.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "Benchmark"
PANELS_DIR = BENCHMARK_DIR / "panels"
MANIFESTS_DIR = BENCHMARK_DIR / "manifests"
RESULTS_DIR = BENCHMARK_DIR / "results"
REPORTS_DIR = BENCHMARK_DIR / "reports"
RUN_BENCHMARK = BENCHMARK_DIR / "run_benchmark.py"
DEFAULT_PANEL = PANELS_DIR / "real_life_panel_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_real_life_panel.py",
        description=__doc__.split("Examples:")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Examples:"):] if "Examples:" in __doc__ else "",
    )

    panel_group = parser.add_argument_group("Panel selection")
    panel_group.add_argument(
        "--panel", type=str, default=str(DEFAULT_PANEL),
        help=f"Panel definition JSON (default: {DEFAULT_PANEL.relative_to(ROOT)})",
    )
    panel_group.add_argument(
        "--groups", type=str, default="",
        help="Run only these groups (comma-separated). "
             "Available: rigid_fragments, metal_coordination, charged_flexible, bulky_hydrophobic",
    )
    panel_group.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N cases after group filtering",
    )
    panel_group.add_argument(
        "--list", action="store_true",
        help="List panel groups and cases with descriptions, then exit",
    )

    run_group = parser.add_argument_group("Run options (forwarded to run_benchmark.py)")
    run_group.add_argument(
        "--scoring", dest="_scoring_hint", default=None,
        help="Scoring method(s): vina, drusina, druseaf, pignet2 (comma-separated)",
    )
    run_group.add_argument(
        "--preset", dest="_preset_hint", default=None,
        choices=["fast", "standard", "thorough", "exhaustive"],
        help="Parameter preset (fast: pop=50/gen=30, standard: 200/200, thorough: 300/300x5)",
    )
    run_group.add_argument(
        "--rescoring", type=str, default=None,
        choices=["gfn2"],
        help="Re-rank top poses with a second scoring method (default: off)",
    )
    run_group.add_argument(
        "--flex-residues", action="store_true", default=False,
        help="Enable flexible receptor sidechains in the binding pocket",
    )
    run_group.add_argument(
        "--no-strain", action="store_true", default=False,
        help="Disable MMFF94 ligand strain penalty",
    )
    run_group.add_argument(
        "--runs", type=int, default=None,
        help="Number of independent GA runs per complex (default: from preset)",
    )
    run_group.add_argument(
        "--charge-method", type=str, default=None,
        choices=["gasteiger", "eem", "qeq", "xtb"],
        help="Partial charge method for protein/ligand (default: gasteiger)",
    )

    output_group = parser.add_argument_group("Output & diagnostics")
    output_group.add_argument(
        "--debug", action="store_true",
        help="Enable debug diagnostics: pocket displacement, failure classification, improvement hints",
    )
    output_group.add_argument(
        "--skip-run", action="store_true",
        help="Skip docking; generate report from existing result files",
    )
    output_group.add_argument(
        "--results", nargs="*", default=None,
        help="Explicit result JSON files to report on (default: auto-detect)",
    )
    return parser.parse_known_args()


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_group_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def find_case(source_manifest: dict, pdb_id: str) -> dict:
    for case in source_manifest["complexes"]:
        if case["pdb_id"] == pdb_id:
            return case
    raise KeyError(f"PDB id {pdb_id} not found in source manifest")


def select_cases(panel: dict, source_manifest: dict, allowed_groups: set[str], limit: int) -> list[dict]:
    selected = []
    for case_meta in panel["cases"]:
        if allowed_groups and case_meta["group"] not in allowed_groups:
            continue
        merged = dict(find_case(source_manifest, case_meta["pdb_id"]))
        merged["_panel"] = case_meta
        selected.append(merged)
    if limit > 0:
        selected = selected[:limit]
    return selected


def write_subset_manifest(panel: dict, selected_cases: list[dict]) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFESTS_DIR / f"{panel['id']}_manifest.json"
    manifest = {
        "benchmark": panel["id"],
        "description": f"{panel['name']}: {len(selected_cases)} curated complexes",
        "complexes": [
            {k: v for k, v in case.items() if k != "_panel"}
            for case in selected_cases
        ],
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out_path


def print_panel(panel: dict, selected_cases: list[dict], verbose: bool = False):
    print()
    print(f"{panel['name']}  [{panel['id']}]")
    if verbose:
        print(panel["description"])
    print()
    current_group = None
    for case in selected_cases:
        meta = case["_panel"]
        if meta["group"] != current_group:
            current_group = meta["group"]
            print(f"{current_group}:")
        if verbose:
            print(f"  {case['pdb_id']}  {meta['title']}  heavy={case['heavy_atom_count']}  pKd={case['pKd']}")
            print(f"      {meta['reason']}")
        else:
            print(f"  {case['pdb_id']}  {meta['title']}")
    print()


def read_forward_arg(forward_args: list[str], flag: str, default: str) -> str:
    if flag in forward_args:
        idx = forward_args.index(flag)
        if idx + 1 < len(forward_args):
            return forward_args[idx + 1]
    for arg in forward_args:
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def ensure_forward_arg(forward_args: list[str], flag: str, value: str):
    if flag in forward_args or any(arg.startswith(flag + "=") for arg in forward_args):
        return
    forward_args.extend([flag, value])


def collect_recent_results(results_dir: Path, prefix: str, started_at: float) -> list[Path]:
    if not results_dir.exists():
        return []
    all_paths = sorted(results_dir.glob(f"{prefix}*.json"))
    recent = []
    for path in all_paths:
        try:
            if path.stat().st_mtime >= started_at - 1:
                recent.append(path)
        except FileNotFoundError:
            continue
    if recent:
        return sorted(recent)

    if not all_paths:
        return []

    latest_mtime = max(path.stat().st_mtime for path in all_paths)
    fallback = [
        path for path in all_paths
        if latest_mtime - path.stat().st_mtime <= 120
    ]
    return sorted(fallback or [max(all_paths, key=lambda path: path.stat().st_mtime)])


def load_result_entries(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_float(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def compute_metrics(entries: list[dict]) -> dict:
    total = len(entries)
    successful = sum(1 for e in entries if e.get("success"))
    rmsds = [e["best_rmsd"] for e in entries if e.get("best_rmsd") is not None]
    lt2 = sum(1 for v in rmsds if v < 2.0)
    lt5 = sum(1 for v in rmsds if v < 5.0)
    mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else None
    mean_time_ms = (sum(e.get("docking_time_ms", 0.0) for e in entries) / total) if total else 0.0
    return {
        "total": total,
        "successful": successful,
        "lt2": lt2,
        "lt5": lt5,
        "mean_rmsd": mean_rmsd,
        "mean_time_ms": mean_time_ms,
    }


def group_order(panel: dict) -> list[str]:
    seen = []
    for case in panel["cases"]:
        group = case["group"]
        if group not in seen:
            seen.append(group)
    return seen


def generate_report(panel: dict, selected_cases: list[dict], result_paths: list[Path]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{panel['id']}_report.md"
    case_meta = {case["_panel"]["pdb_id"]: case["_panel"] for case in selected_cases}
    ordered_ids = [case["pdb_id"] for case in selected_cases]
    group_ids = group_order(panel)

    lines = [
        f"# {panel['name']}",
        "",
        panel["description"],
        "",
        f"Cases: {len(selected_cases)}",
        "",
        "## Panel Cases",
        "",
        "| PDB | Group | Title | Heavy Atoms | pKd |",
        "|-----|-------|-------|-------------|-----|",
    ]
    selected_by_id = {case["pdb_id"]: case for case in selected_cases}
    for pdb_id in ordered_ids:
        case = selected_by_id[pdb_id]
        meta = case_meta[pdb_id]
        lines.append(
            f"| {pdb_id} | {meta['group']} | {meta['title']} | {case['heavy_atom_count']} | {case['pKd']:.2f} |"
        )

    for result_path in result_paths:
        payload = load_result_entries(result_path)
        entries_by_id = {entry["pdb_id"]: entry for entry in payload["entries"]}
        ordered_entries = [entries_by_id[pdb_id] for pdb_id in ordered_ids if pdb_id in entries_by_id]
        metrics = compute_metrics(ordered_entries)

        lines += [
            "",
            f"## {payload['scoringMethod']}",
            "",
            f"- Result file: `{result_path.name}`",
            f"- Success: {metrics['successful']}/{metrics['total']}",
            f"- RMSD < 2.0 A: {metrics['lt2']}/{len([e for e in ordered_entries if e.get('best_rmsd') is not None])}",
            f"- Mean RMSD: {fmt_float(metrics['mean_rmsd'])} A",
            f"- Mean time: {fmt_float(metrics['mean_time_ms'], 0)} ms/complex",
            "",
            "### By Group",
            "",
            "| Group | Success | RMSD < 2.0 A | Mean RMSD | Mean Time (ms) |",
            "|-------|---------|--------------|-----------|----------------|",
        ]

        for group in group_ids:
            group_entries = [
                entry for entry in ordered_entries
                if case_meta[entry["pdb_id"]]["group"] == group
            ]
            if not group_entries:
                continue
            group_metrics = compute_metrics(group_entries)
            rmsd_count = len([e for e in group_entries if e.get("best_rmsd") is not None])
            lines.append(
                f"| {group} | {group_metrics['successful']}/{group_metrics['total']} | "
                f"{group_metrics['lt2']}/{rmsd_count} | {fmt_float(group_metrics['mean_rmsd'])} A | "
                f"{fmt_float(group_metrics['mean_time_ms'], 0)} |"
            )

        method = payload["scoringMethod"].lower()
        score_label = "pKi" if "druseaf" in method or "affinity" in method else "Score"
        lines += [
            "",
            "### Per Case",
            "",
            f"| PDB | Group | {score_label} | RMSD | Time (ms) | Status |",
            "|-----|-------|-------|------|-----------|--------|",
        ]

        for entry in ordered_entries:
            meta = case_meta[entry["pdb_id"]]
            score = fmt_float(entry.get("best_display_score") or entry.get("best_energy")) if entry.get("success") else "N/A"
            rmsd = fmt_float(entry.get("best_rmsd"))
            status = "ok" if entry.get("success") else f"failed: {entry.get('error', 'unknown')}"
            lines.append(
                f"| {entry['pdb_id']} | {meta['group']} | {score} | {rmsd} | "
                f"{fmt_float(entry.get('docking_time_ms', 0.0), 0)} | {status} |"
            )

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def classify_failure(entry: dict, meta: dict) -> str:
    """Classify the root cause of a docking failure or bad pose."""
    if not entry.get("success"):
        err = entry.get("error", "unknown")
        if "pocket" in err:
            return "pocket_detection_failed"
        if "ligand" in err:
            return "ligand_prep_failed"
        return f"other_failure: {err}"

    rmsd = entry.get("best_rmsd")
    if rmsd is None:
        return "atom_count_mismatch"

    pocket_dist = entry.get("pocket_distance")
    if pocket_dist is not None and pocket_dist > 10.0:
        return "wrong_pocket"
    if pocket_dist is not None and pocket_dist > 5.0:
        return "pocket_displaced"

    if rmsd > 10.0:
        pose_rmsds = entry.get("all_pose_rmsds", [])
        if pose_rmsds and min(pose_rmsds) > 8.0:
            return "search_failure"
        if pose_rmsds and min(pose_rmsds) < 5.0:
            return "ranking_failure"
        return "search_failure"

    if rmsd > 5.0:
        return "partial_pose"
    if rmsd > 2.0:
        return "close_miss"

    return "success"


def generate_debug_report(panel: dict, selected_cases: list[dict], result_paths: list[Path]) -> Path:
    """Generate a detailed diagnostic report with failure analysis and improvement hints."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{panel['id']}_debug.md"
    case_meta = {case["_panel"]["pdb_id"]: case["_panel"] for case in selected_cases}
    ordered_ids = [case["pdb_id"] for case in selected_cases]
    group_ids = group_order(panel)

    lines = [
        f"# {panel['name']} — Debug Diagnostic Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for result_path in result_paths:
        payload = load_result_entries(result_path)
        entries_by_id = {e["pdb_id"]: e for e in payload["entries"]}
        ordered_entries = [entries_by_id[pid] for pid in ordered_ids if pid in entries_by_id]

        lines += [
            f"## {payload['scoringMethod']}",
            "",
            f"Result file: `{result_path.name}`",
            "",
        ]

        # ── Section 1: Pocket Displacement Analysis ────────────────────
        lines += [
            "### 1. Pocket Detection Analysis",
            "",
            "How far is the detected pocket center from the crystal ligand centroid?",
            "Displacements >10A mean the docking searched in the **wrong region entirely**.",
            "",
            "| PDB | Pocket Method | Pocket Dist (A) | Pocket Vol (A3) | Buried | Box Size (A) | Verdict |",
            "|-----|---------------|-----------------|-----------------|--------|--------------|---------|",
        ]
        pocket_dists = []
        for entry in ordered_entries:
            pdist = entry.get("pocket_distance")
            pvol = entry.get("pocket_volume")
            pmethod = entry.get("pocket_method", "?")
            pburied = entry.get("pocket_buriedness")
            box = entry.get("search_box_size")
            box_str = f"{box[0]:.0f}x{box[1]:.0f}x{box[2]:.0f}" if box else "N/A"

            if pdist is not None:
                pocket_dists.append(pdist)
                if pdist > 10.0:
                    verdict = "WRONG POCKET"
                elif pdist > 5.0:
                    verdict = "DISPLACED"
                elif pdist > 2.0:
                    verdict = "OFFSET"
                else:
                    verdict = "OK"
            else:
                verdict = "NO DATA" if entry.get("success") else "FAILED"

            lines.append(
                f"| {entry['pdb_id']} | {pmethod} | {fmt_float(pdist, 1)} | "
                f"{fmt_float(pvol, 0)} | {fmt_float(pburied, 2)} | {box_str} | {verdict} |"
            )

        if pocket_dists:
            mean_dist = sum(pocket_dists) / len(pocket_dists)
            wrong_pocket = sum(1 for d in pocket_dists if d > 10.0)
            lines += [
                "",
                f"**Mean pocket displacement: {mean_dist:.1f} A** "
                f"({wrong_pocket}/{len(pocket_dists)} cases with wrong pocket >10A)",
                "",
            ]

        # ── Section 2: Ligand Preparation Analysis ─────────────────────
        lines += [
            "### 2. Ligand Preparation Analysis",
            "",
            "Atom count mismatches indicate the SMILES-derived ligand differs from the crystal.",
            "",
            "| PDB | Crystal Heavy | Prepared Heavy | Match | Initial RMSD (A) |",
            "|-----|---------------|----------------|-------|------------------|",
        ]
        for entry in ordered_entries:
            chc = entry.get("crystal_heavy_count")
            lhc = entry.get("ligand_heavy_count")
            irmsd = entry.get("initial_ligand_rmsd")
            match = "YES" if chc and lhc and chc == lhc else ("MISMATCH" if chc and lhc else "N/A")
            lines.append(
                f"| {entry['pdb_id']} | {chc or 'N/A'} | {lhc or 'N/A'} | {match} | {fmt_float(irmsd, 1)} |"
            )

        # ── Section 3: Pose Quality Analysis ───────────────────────────
        lines += [
            "",
            "### 3. Pose Quality & Convergence",
            "",
            "Top-10 pose RMSDs show whether the search found near-native poses at all ",
            "(search failure) or found them but ranked them poorly (ranking failure).",
            "",
            "| PDB | Group | Best RMSD | Energy | Top-10 Min RMSD | Top-10 Spread | Strain | Diagnosis |",
            "|-----|-------|-----------|--------|-----------------|---------------|--------|-----------|",
        ]
        failure_counts: dict[str, int] = {}
        for entry in ordered_entries:
            meta = case_meta.get(entry["pdb_id"], {})
            group = meta.get("group", "?")
            rmsd = entry.get("best_rmsd")
            energy = entry.get("best_display_score") or entry.get("best_energy")
            pose_rmsds = entry.get("all_pose_rmsds", [])
            strain = entry.get("strain_energy")
            min_rmsd = min(pose_rmsds) if pose_rmsds else None
            spread = (max(pose_rmsds) - min(pose_rmsds)) if len(pose_rmsds) >= 2 else None

            diagnosis = classify_failure(entry, meta)
            failure_counts[diagnosis] = failure_counts.get(diagnosis, 0) + 1

            lines.append(
                f"| {entry['pdb_id']} | {group} | {fmt_float(rmsd)} | "
                f"{fmt_float(energy)} | {fmt_float(min_rmsd)} | "
                f"{fmt_float(spread)} | {fmt_float(strain)} | {diagnosis} |"
            )

        # ── Section 4: Failure Breakdown ───────────────────────────────
        lines += [
            "",
            "### 4. Failure Breakdown",
            "",
            "| Root Cause | Count | % |",
            "|------------|-------|---|",
        ]
        total = len(ordered_entries)
        for cause in sorted(failure_counts, key=failure_counts.get, reverse=True):
            count = failure_counts[cause]
            lines.append(f"| {cause} | {count} | {100*count/total:.0f}% |")

        # ── Section 5: Per-Group Summary ───────────────────────────────
        lines += [
            "",
            "### 5. Per-Group Diagnostic Summary",
            "",
        ]
        for group in group_ids:
            group_entries = [
                e for e in ordered_entries
                if case_meta.get(e["pdb_id"], {}).get("group") == group
            ]
            if not group_entries:
                continue
            pdists = [e["pocket_distance"] for e in group_entries if e.get("pocket_distance") is not None]
            rmsds = [e["best_rmsd"] for e in group_entries if e.get("best_rmsd") is not None]
            failures = [e for e in group_entries if not e.get("success")]

            lines.append(f"**{group}** ({len(group_entries)} cases)")
            if pdists:
                lines.append(f"- Pocket displacement: mean={sum(pdists)/len(pdists):.1f}A, "
                             f"max={max(pdists):.1f}A, wrong(>10A)={sum(1 for d in pdists if d>10)}")
            if rmsds:
                lines.append(f"- RMSD: mean={sum(rmsds)/len(rmsds):.1f}A, "
                             f"best={min(rmsds):.1f}A, <2A={sum(1 for r in rmsds if r<2)}")
            if failures:
                lines.append(f"- Failures: {len(failures)} ({', '.join(e['pdb_id'] for e in failures)})")
            lines.append("")

        # ── Section 6: Actionable Improvement Hints ────────────────────
        lines += [
            "### 6. Improvement Priorities",
            "",
        ]

        hints = []

        # Pocket detection hints
        n_pocket_fail = failure_counts.get("pocket_detection_failed", 0)
        n_wrong_pocket = failure_counts.get("wrong_pocket", 0)
        n_displaced = failure_counts.get("pocket_displaced", 0)
        pocket_issues = n_pocket_fail + n_wrong_pocket + n_displaced
        if pocket_issues > 0:
            hints.append((
                pocket_issues,
                "Pocket Detection",
                f"{n_pocket_fail} detection failures + {n_wrong_pocket} wrong pockets + "
                f"{n_displaced} displaced pockets = **{pocket_issues}/{total} cases** "
                f"({100*pocket_issues/total:.0f}%) with pocket issues.",
                [
                    "Retrain/improve ML pocket detector on diverse binding sites",
                    "Lower geometric detection thresholds (buriedness, minPts) for shallow pockets",
                    "Add metal-aware pocket detection (many failures are metal-containing sites)",
                    "Consider using known ligand positions for pocket centering in product (upload reference)",
                ],
            ))

        # Search failure hints
        n_search = failure_counts.get("search_failure", 0)
        if n_search > 0:
            hints.append((
                n_search,
                "Search Exhaustiveness",
                f"**{n_search}/{total} cases** ({100*n_search/total:.0f}%) where no near-native "
                f"pose was found in top-10.",
                [
                    "Increase population size and generations (current preset may be too weak)",
                    "Add more independent runs (runs=3-5) for large/flexible ligands",
                    "Increase exploration phase ratio for large search boxes",
                    "Check if grid spacing is coarse enough to cover the search space",
                ],
            ))

        # Ranking failure hints
        n_ranking = failure_counts.get("ranking_failure", 0)
        if n_ranking > 0:
            hints.append((
                n_ranking,
                "Scoring / Ranking",
                f"**{n_ranking}/{total} cases** ({100*n_ranking/total:.0f}%) where a near-native "
                f"pose exists in top-10 but was not ranked #1.",
                [
                    "Review scoring function weights (VDW vs electrostatics vs H-bond)",
                    "Add explicit metal coordination scoring terms",
                    "Improve strain penalty calibration",
                    "Consider ML rescoring of top poses (DruseAF, PIGNet2)",
                ],
            ))

        # Atom count mismatch hints
        n_mismatch = failure_counts.get("atom_count_mismatch", 0)
        if n_mismatch > 0:
            hints.append((
                n_mismatch,
                "Ligand Preparation",
                f"**{n_mismatch}/{total} cases** with atom count mismatch between "
                f"SMILES-derived and crystal ligand.",
                [
                    "Check SMILES canonicalization and protonation state assignment",
                    "Verify RDKit SMILES→3D handles all functional groups correctly",
                    "Compare prepared ligand with crystal SDF to identify discrepancies",
                ],
            ))

        if not hints:
            lines.append("No specific improvement hints — all cases were successful.")
        else:
            for priority, (_, title, description, actions) in enumerate(
                sorted(hints, key=lambda h: h[0], reverse=True), 1
            ):
                lines.append(f"**Priority {priority}: {title}**")
                lines.append(f"- {description}")
                for action in actions:
                    lines.append(f"  - {action}")
                lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def print_debug_summary(result_paths: list[Path], case_meta: dict):
    """Print a compact terminal debug summary."""
    print("\n" + "=" * 70)
    print("  DEBUG DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for result_path in result_paths:
        payload = load_result_entries(result_path)
        entries = payload["entries"]
        print(f"\n  [{payload['scoringMethod']}]")

        # Pocket displacement stats
        pocket_dists = [e["pocket_distance"] for e in entries if e.get("pocket_distance") is not None]
        if pocket_dists:
            wrong = sum(1 for d in pocket_dists if d > 10.0)
            displaced = sum(1 for d in pocket_dists if 5.0 < d <= 10.0)
            ok = sum(1 for d in pocket_dists if d <= 5.0)
            print(f"  Pocket accuracy:  OK(<5A)={ok}  displaced(5-10A)={displaced}  "
                  f"wrong(>10A)={wrong}  failed={len(entries)-len(pocket_dists)}")
            print(f"  Pocket dist:      mean={sum(pocket_dists)/len(pocket_dists):.1f}A  "
                  f"median={sorted(pocket_dists)[len(pocket_dists)//2]:.1f}A  "
                  f"max={max(pocket_dists):.1f}A")

        # Failure classification
        failure_counts: dict[str, int] = {}
        for entry in entries:
            meta = case_meta.get(entry["pdb_id"], {})
            diagnosis = classify_failure(entry, meta)
            failure_counts[diagnosis] = failure_counts.get(diagnosis, 0) + 1

        print("  Failure breakdown:")
        for cause in sorted(failure_counts, key=failure_counts.get, reverse=True):
            count = failure_counts[cause]
            bar = "#" * count
            print(f"    {cause:30s} {count:2d}/{len(entries)}  {bar}")

        # Worst cases
        docked = [e for e in entries if e.get("success") and e.get("pocket_distance") is not None]
        if docked:
            worst = sorted(docked, key=lambda e: e.get("pocket_distance", 0), reverse=True)[:3]
            print("  Worst pocket displacements:")
            for e in worst:
                print(f"    {e['pdb_id']:6s}  pocket_dist={e['pocket_distance']:.1f}A  "
                      f"method={e.get('pocket_method','?'):10s}  "
                      f"rmsd={fmt_float(e.get('best_rmsd'))}A")

    print()


def main():
    args, forward_args = parse_args()

    panel_path = Path(args.panel)
    if not panel_path.is_absolute():
        panel_path = (ROOT / panel_path).resolve()
    panel = load_json(panel_path)

    source_manifest_path = panel_path.parent.parent.parent / panel["source_manifest"]
    source_manifest = load_json(source_manifest_path)

    selected = select_cases(
        panel,
        source_manifest,
        allowed_groups=parse_group_filter(args.groups),
        limit=args.limit,
    )

    if not selected:
        raise SystemExit("No panel cases selected")

    if args.list:
        print_panel(panel, selected, verbose=True)
        return

    print_panel(panel, selected, verbose=False)
    subset_manifest = write_subset_manifest(panel, selected)

    result_paths: list[Path]
    if args.results:
        result_paths = [
            Path(path) if Path(path).is_absolute() else (ROOT / path)
            for path in args.results
        ]
    else:
        output_dir = Path(read_forward_arg(forward_args, "--output-dir", str(RESULTS_DIR)))
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
        output_prefix = read_forward_arg(forward_args, "--output-prefix", f"{panel['id']}_")
        ensure_forward_arg(forward_args, "--output-prefix", output_prefix)
        # Re-inject args consumed by our own argparse into forward_args
        if args._scoring_hint is not None:
            ensure_forward_arg(forward_args, "--scoring", args._scoring_hint)
        if args._preset_hint is not None:
            ensure_forward_arg(forward_args, "--preset", args._preset_hint)
        ensure_forward_arg(forward_args, "--pipeline-mode", "full")
        ensure_forward_arg(forward_args, "--pocket-mode", "hybrid")
        if args.debug and "--debug" not in forward_args:
            forward_args.append("--debug")
        if args.rescoring == "gfn2":
            ensure_forward_arg(forward_args, "--gfn2-rescoring", "")
            # --gfn2-rescoring is a boolean flag, fix the empty value
            if "--gfn2-rescoring" in forward_args:
                idx = forward_args.index("--gfn2-rescoring")
                if idx + 1 < len(forward_args) and forward_args[idx + 1] == "":
                    forward_args.pop(idx + 1)
        if args.flex_residues and "--flex-residues" not in forward_args:
            forward_args.append("--flex-residues")
        if args.no_strain:
            ensure_forward_arg(forward_args, "--no-strain-penalty", "")
            if "--no-strain-penalty" in forward_args:
                idx = forward_args.index("--no-strain-penalty")
                if idx + 1 < len(forward_args) and forward_args[idx + 1] == "":
                    forward_args.pop(idx + 1)
        if args.runs is not None:
            ensure_forward_arg(forward_args, "--runs", str(args.runs))
        if args.charge_method is not None:
            ensure_forward_arg(forward_args, "--charge-method", args.charge_method)
        started_at = time.time()

        if not args.skip_run:
            cmd = [
                sys.executable,
                str(RUN_BENCHMARK),
                "--manifest",
                str(subset_manifest),
                "--benchmark-name",
                panel["name"],
            ] + forward_args
            subprocess.run(cmd, cwd=str(ROOT), check=True)

        result_paths = collect_recent_results(output_dir, output_prefix, started_at)

    if not result_paths:
        raise SystemExit("No result files found for the selected panel run")

    report_path = generate_report(panel, selected, result_paths)
    print(f"Report: {report_path.relative_to(ROOT)}")
    for path in result_paths:
        print(f"  -> {path.relative_to(ROOT)}")

    if args.debug:
        case_meta_map = {case["_panel"]["pdb_id"]: case["_panel"] for case in selected}
        print_debug_summary(result_paths, case_meta_map)
        debug_path = generate_debug_report(panel, selected, result_paths)
        print(f"Debug report: {debug_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Druse Benchmark Runner — Fully configurable docking benchmark CLI.

Drives CASF-2016 (and future datasets) through the Druse docking pipeline
with fine-grained control over scoring, charges, flexibility, refinement,
and post-processing.

Under the hood this invokes `xcodebuild test` against the DruseTests target,
passing configuration via environment variables that the Swift BenchmarkRunner
reads at launch.

Usage examples:

  # Quick 10-complex validation with Vina
  python Benchmark/run_benchmark.py --casf --scoring vina --quick 10

  # Full CASF with Drusina + GFN2 rescoring
  python Benchmark/run_benchmark.py --casf --scoring drusina --gfn2-rescoring

  # All scoring methods, full dataset
  python Benchmark/run_benchmark.py --casf --scoring vina,drusina,druseaf,pignet2,gfn2

  # Thorough: large population, flexible residues, GFN2 geometry optimization
  python Benchmark/run_benchmark.py --casf --scoring drusina \\
      --population 300 --generations 300 --runs 5 \\
      --flex-residues --gfn2-opt --gfn2-solvation water \\
      --charge-method xtb

  # Virtual screening benchmark config
  python Benchmark/run_benchmark.py --casf --scoring drusina \\
      --gfn2-refine-top 50 --strain-penalty

  # List all available flags
  python Benchmark/run_benchmark.py --help
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "Benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
PROJECT = ROOT / "Druse.xcodeproj"
VERSION = (ROOT / "VERSION").read_text().strip()

# ---------------------------------------------------------------------------
# Scoring method → XCTest method mapping
# ---------------------------------------------------------------------------

SCORING_TESTS = {
    "vina":     "testCASF_Vina",
    "drusina":  "testCASF_Drusina",
    "druseaf":  "testCASF_DruseAF",
    "pignet2":  "testCASF_PIGNet2",
    "gfn2":     "testCASF_DrusinaGFN2",
    "all":      "testCASF_All",
}

SCORING_LABELS = {
    "vina":     "Vina",
    "drusina":  "Drusina",
    "druseaf":  "Druse Affinity",
    "pignet2":  "PIGNet2",
    "gfn2":     "Drusina+GFN2",
}


def parse_args():
    p = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="Druse Benchmark Runner — configurable docking benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples:")[1] if "Usage examples:" in __doc__ else "",
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    ds = p.add_argument_group("Dataset")
    ds.add_argument("--casf", action="store_true",
                    help="Run CASF-2016 benchmark (225 complexes)")
    ds.add_argument("--quick", type=int, metavar="N", default=0,
                    help="Limit to first N complexes (fast validation)")

    # ── Scoring ──────────────────────────────────────────────────────────
    sc = p.add_argument_group("Scoring Method")
    sc.add_argument("--scoring", type=str, default="vina",
                    help="Scoring method(s): vina, drusina, druseaf, pignet2, gfn2, all "
                         "(comma-separated for multiple runs, e.g. vina,drusina)")
    sc.add_argument("--gfn2-rescoring", action="store_true",
                    help="Post-docking GFN2-xTB rescoring: compute D4 dispersion + "
                         "ALPB solvation energy on best pose (~2ms/complex)")

    # ── GA Search Parameters ─────────────────────────────────────────────
    ga = p.add_argument_group("Genetic Algorithm")
    ga.add_argument("--population", type=int, default=200,
                    help="GA population size (default: 200)")
    ga.add_argument("--generations", type=int, default=200,
                    help="GA generations per run (default: 200)")
    ga.add_argument("--runs", type=int, default=1,
                    help="Independent MC trajectory runs (default: 1)")
    ga.add_argument("--grid-spacing", type=float, default=0.375,
                    choices=[0.375, 0.5, 0.75],
                    help="Grid spacing in Angstrom (default: 0.375)")
    ga.add_argument("--mutation-rate", type=float, default=0.08,
                    help="GA mutation rate (default: 0.08)")
    ga.add_argument("--mc-temperature", type=float, default=1.2,
                    help="Metropolis temperature kcal/mol (default: 1.2)")
    ga.add_argument("--auto-mode", action="store_true",
                    help="Auto-tune parameters per complex (overrides manual GA settings)")

    # ── Local Search ─────────────────────────────────────────────────────
    ls = p.add_argument_group("Local Search")
    ls.add_argument("--local-search-freq", type=int, default=1,
                    help="Local search every N generations (default: 1)")
    ls.add_argument("--local-search-steps", type=int, default=30,
                    help="Gradient descent steps per local search (default: 30)")
    ls.add_argument("--no-analytical-gradients", action="store_true",
                    help="Use numerical finite-difference gradients (slower)")

    # ── Flexibility ──────────────────────────────────────────────────────
    fl = p.add_argument_group("Flexibility")
    fl.add_argument("--no-ligand-flex", action="store_true",
                    help="Disable ligand torsion flexibility")
    fl.add_argument("--flex-residues", action="store_true",
                    help="Enable flexible receptor residues (auto-select pocket-lining)")
    fl.add_argument("--flex-refinement-steps", type=int, default=50,
                    help="Post-GA torsion refinement steps (default: 50)")

    # ── Charge Method ────────────────────────────────────────────────────
    ch = p.add_argument_group("Charge Computation")
    ch.add_argument("--charge-method", type=str, default="gasteiger",
                    choices=["gasteiger", "eem", "qeq", "xtb"],
                    help="Partial charge method for protein/ligand preparation "
                         "(default: gasteiger)")

    # ── Post-Docking Refinement ──────────────────────────────────────────
    pd = p.add_argument_group("Post-Docking Refinement")
    pd.add_argument("--strain-penalty", action="store_true", default=True,
                    help="Enable MMFF94 ligand strain penalty (default: on)")
    pd.add_argument("--no-strain-penalty", action="store_true",
                    help="Disable MMFF94 ligand strain penalty")
    pd.add_argument("--strain-threshold", type=float, default=6.0,
                    help="Strain penalty threshold kcal/mol (default: 6.0)")
    pd.add_argument("--strain-weight", type=float, default=0.5,
                    help="Strain penalty weight in score (default: 0.5)")

    # ── GFN2-xTB Geometry Optimization ───────────────────────────────────
    gfn2 = p.add_argument_group("GFN2-xTB Geometry Optimization")
    gfn2.add_argument("--gfn2-opt", action="store_true",
                      help="Enable GFN2-xTB geometry optimization of top poses")
    gfn2.add_argument("--gfn2-opt-level", type=str, default="normal",
                      choices=["crude", "normal", "tight"],
                      help="GFN2 optimization convergence level (default: normal)")
    gfn2.add_argument("--gfn2-solvation", type=str, default="water",
                      choices=["none", "water", "gbsa"],
                      help="GFN2 solvation model (default: water/ALPB)")
    gfn2.add_argument("--gfn2-top-poses", type=int, default=20,
                      help="Number of top poses to refine (default: 20)")
    gfn2.add_argument("--gfn2-blend-weight", type=float, default=0.3,
                      help="GFN2 energy blend weight in ranking 0-1 (default: 0.3)")
    gfn2.add_argument("--gfn2-max-steps", type=int, default=0,
                      help="Max optimization steps, 0=auto (default: 0)")
    gfn2.add_argument("--gfn2-refine-top", type=int, default=0,
                      help="GFN2 single-point refine top N screening hits (default: 0=off)")

    # ── Exploration Phase ────────────────────────────────────────────────
    ex = p.add_argument_group("Exploration Phase")
    ex.add_argument("--exploration-ratio", type=float, default=0.4,
                    help="Fraction of generations in exploration mode (default: 0.4)")
    ex.add_argument("--exploration-translation", type=float, default=4.0,
                    help="Translation step during exploration (default: 4.0 A)")
    ex.add_argument("--exploration-rotation", type=float, default=0.6,
                    help="Rotation step during exploration (default: 0.6 rad)")

    # ── Reranking ────────────────────────────────────────────────────────
    rr = p.add_argument_group("Explicit Reranking")
    rr.add_argument("--rerank-top", type=int, default=12,
                    help="Top cluster representatives for explicit rescoring (default: 12)")
    rr.add_argument("--rerank-variants", type=int, default=4,
                    help="Local refinement seeds per cluster (default: 4)")

    # ── Output ───────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                     help=f"Output directory for results JSON (default: {RESULTS_DIR})")
    out.add_argument("--output-prefix", type=str, default="",
                     help="Prefix for output filenames (e.g. 'exp1_')")
    out.add_argument("--analyze", action="store_true",
                     help="Run analyze.py after benchmark completes")
    out.add_argument("--verbose", action="store_true",
                     help="Show full xcodebuild output")
    out.add_argument("--druse-logs", nargs="?", const="-", default=None,
                     metavar="FILE",
                     help="Stream Druse app logs in real time. "
                          "No value = print to terminal; "
                          "provide a path to write to a file instead")

    # ── Presets ──────────────────────────────────────────────────────────
    pr = p.add_argument_group("Presets (override individual settings)")
    pr.add_argument("--preset", type=str,
                    choices=["fast", "standard", "thorough", "exhaustive"],
                    help="Apply a parameter preset")

    args = p.parse_args()

    if not args.casf:
        p.error("No dataset specified. Use --casf (more datasets coming soon)")

    return args


def apply_preset(args):
    """Apply parameter preset, overriding individual settings."""
    if not args.preset:
        return

    presets = {
        "fast": {
            "population": 50, "generations": 30, "runs": 1,
            "grid_spacing": 0.5, "local_search_freq": 3,
            "local_search_steps": 15, "gfn2_opt": False,
        },
        "standard": {
            "population": 200, "generations": 200, "runs": 1,
            "grid_spacing": 0.375, "local_search_freq": 1,
            "local_search_steps": 30, "gfn2_opt": False,
        },
        "thorough": {
            "population": 300, "generations": 300, "runs": 5,
            "grid_spacing": 0.375, "local_search_freq": 1,
            "local_search_steps": 50, "gfn2_opt": False,
            "flex_residues": True,
        },
        "exhaustive": {
            "population": 300, "generations": 300, "runs": 10,
            "grid_spacing": 0.375, "local_search_freq": 1,
            "local_search_steps": 50, "gfn2_opt": True,
            "gfn2_opt_level": "tight", "gfn2_top_poses": 50,
            "flex_residues": True,
        },
    }

    p = presets[args.preset]
    for k, v in p.items():
        setattr(args, k, v)
    print(f"Applied preset: {args.preset}")


CONFIG_FILE = BENCHMARK_DIR / ".bench_config.json"


def build_config(args) -> dict:
    """Build configuration dict and write to JSON file for BenchmarkRunner to read.

    Environment variables don't reliably propagate through xcodebuild's test host,
    so we use a shared JSON file instead.
    """
    strain_on = args.strain_penalty and not args.no_strain_penalty

    config = {
        # GA parameters
        "population": args.population,
        "generations": args.generations,
        "runs": args.runs,
        "gridSpacing": args.grid_spacing,
        "mutationRate": args.mutation_rate,
        "mcTemperature": args.mc_temperature,
        "autoMode": args.auto_mode,

        # Local search
        "localSearchFreq": args.local_search_freq,
        "localSearchSteps": args.local_search_steps,
        "analyticalGradients": not args.no_analytical_gradients,

        # Flexibility
        "ligandFlex": not args.no_ligand_flex,
        "flexResidues": args.flex_residues,
        "flexRefineSteps": args.flex_refinement_steps,

        # Charge method
        "chargeMethod": args.charge_method,

        # Strain penalty
        "strainPenalty": strain_on,
        "strainThreshold": args.strain_threshold,
        "strainWeight": args.strain_weight,

        # GFN2
        "gfn2Opt": args.gfn2_opt,
        "gfn2OptLevel": args.gfn2_opt_level,
        "gfn2Solvation": args.gfn2_solvation,
        "gfn2TopPoses": args.gfn2_top_poses,
        "gfn2BlendWeight": args.gfn2_blend_weight,
        "gfn2MaxSteps": args.gfn2_max_steps,
        "gfn2Scoring": args.gfn2_rescoring,
        "gfn2RefineTop": args.gfn2_refine_top,

        # Exploration
        "explorationRatio": args.exploration_ratio,
        "explorationTranslation": args.exploration_translation,
        "explorationRotation": args.exploration_rotation,

        # Reranking
        "rerankTop": args.rerank_top,
        "rerankVariants": args.rerank_variants,

        # Quick mode
        "maxComplexes": args.quick if args.quick > 0 else 0,

        # Stdout log mirroring
        "stdoutLogs": args.druse_logs is not None,

        # Output
        "outputDir": args.output_dir,
        "outputPrefix": args.output_prefix,
        "outputFile": "",  # set per scoring method
    }

    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    return config


def update_config_output_file(output_file: str):
    """Update the output file in the shared config."""
    config = json.loads(CONFIG_FILE.read_text())
    config["outputFile"] = output_file
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def update_config_gfn2_scoring(enabled: bool):
    """Update GFN2 scoring flag in the shared config."""
    config = json.loads(CONFIG_FILE.read_text())
    config["gfn2Scoring"] = enabled
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def resolve_scoring_methods(args, run_timestamp: str) -> list:
    """Parse --scoring flag into list of (method_key, test_name, output_file)."""
    raw = args.scoring.lower().strip()
    if raw == "all":
        methods = ["vina", "drusina", "druseaf", "pignet2", "gfn2"]
    else:
        methods = [m.strip() for m in raw.split(",")]

    result = []
    for m in methods:
        if m not in SCORING_TESTS:
            print(f"Unknown scoring method: {m}")
            print(f"Available: {', '.join(SCORING_TESTS.keys())}")
            sys.exit(1)

        test_name = SCORING_TESTS[m]
        prefix = args.output_prefix
        quick = f"_quick{args.quick}" if args.quick > 0 else ""
        output_file = f"{prefix}casf_{m}_{VERSION}_{run_timestamp}{quick}.json"
        result.append((m, test_name, output_file))

    return result


def _format_rmsd(rmsd: float) -> str:
    """Color-code RMSD for terminal display."""
    if rmsd < 2.0:
        return f"\033[32m{rmsd:6.2f}\033[0m"  # green
    elif rmsd < 5.0:
        return f"\033[33m{rmsd:6.2f}\033[0m"  # yellow
    else:
        return f"\033[31m{rmsd:6.2f}\033[0m"  # red


def _format_time(ms: float) -> str:
    """Format docking time compactly."""
    if ms < 10_000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/1000:.0f}s"


import re
# Vina/Drusina format — tolerates optional fields (e.g. GFN2=xxx) between RMSD and ms:
#   [1/10] 1a30  E=-11.04  RMSD=3.49A  5912ms
#   [1/10] 1a30  E=-11.04  RMSD=3.49A  GFN2=-123  5912ms
#   [1/10] 1a30  E=-11.04  RMSD=N/AA  5912ms
_RESULT_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+E=([\-\d.]+)\s+RMSD=([\d.]+|N/A)A.*?(\d+)ms"
)
# DruseAF format — tolerates optional fields after RMSD:
#   [1/10] 1a30  pKi=3.24  conf=7%  RMSD=9.36A  26156ms
_RESULT_AF_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+pKi=([\-\d.]+)\s+conf=(\d+)%\s+RMSD=([\d.]+|N/A)A.*?(\d+)ms"
)
# FAILED format:  [1/10] 1a30  FAILED: error message
_FAILED_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+FAILED:\s*(.*)"
)
_SUMMARY_RE = re.compile(r"(Docking Power|Scoring Power|Pearson|RMSD\s*<|Performance|Summary)")


def run_xcodebuild(test_method: str, env: dict, verbose: bool,
                    druse_logs: str | None = None) -> tuple:
    """Run a single benchmark test via xcodebuild. Returns (success, duration_s, output).

    Always streams per-complex results in real time. Filters xcodebuild noise.
    druse_logs: None = benchmark lines only; "-" = all output to terminal;
                path = all output to file (benchmark lines still shown on terminal).
    """
    test_id = f"DruseTests/BenchmarkRunner/{test_method}"
    cmd = [
        "xcodebuild", "test",
        "-project", str(PROJECT),
        "-scheme", "Druse",
        f"-only-testing:{test_id}",
    ]

    t0 = time.time()

    log_file = None
    if druse_logs is not None and druse_logs != "-":
        log_file = open(druse_logs, "a")

    collected = []
    # Running stats
    n_done = 0
    n_total = 0
    n_lt2 = 0
    n_lt5 = 0
    sum_time = 0.0
    in_summary = False

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(ROOT),
    )
    try:
        for line in proc.stdout:
            collected.append(line)

            # Write full output to log file if requested
            if log_file is not None:
                log_file.write(line)
                log_file.flush()

            # If --druse-logs -, stream everything raw
            if druse_logs == "-":
                sys.stdout.write(line)
                sys.stdout.flush()
                continue

            stripped = line.strip()
            if not stripped:
                continue

            # Per-complex result line (Vina/Drusina format)
            m = _RESULT_RE.search(stripped)
            m_af = _RESULT_AF_RE.search(stripped) if not m else None
            m_fail = _FAILED_RE.search(stripped) if not m and not m_af else None
            if m or m_af:
                if m:
                    idx, total, pdb, energy, rmsd_s, time_ms = m.groups()
                    score_str = f"E={float(energy):7.2f}"
                else:
                    idx, total, pdb, pKd, conf, rmsd_s, time_ms = m_af.groups()
                    score_str = f"pKi={float(pKd):5.2f}  conf={conf}%"
                n_done = int(idx)
                n_total = int(total)
                rmsd = float(rmsd_s) if rmsd_s != "N/A" else None
                t_ms = float(time_ms)
                sum_time += t_ms
                if rmsd is not None and rmsd < 2.0:
                    n_lt2 += 1
                if rmsd is not None and rmsd < 5.0:
                    n_lt5 += 1

                rate_lt2 = 100 * n_lt2 / max(n_done, 1)
                avg_t = sum_time / max(n_done, 1)

                # Compact live line: progress, PDB, score, RMSD, time, running stats
                if rmsd is not None:
                    rmsd_colored = _format_rmsd(rmsd)
                else:
                    rmsd_colored = "\033[2mN/A\033[0m"
                print(
                    f"  {n_done:3d}/{n_total}  {pdb}  "
                    f"{score_str}  RMSD={rmsd_colored}A  {_format_time(t_ms):>5s}  "
                    f"\033[2m<2A: {n_lt2}/{n_done} ({rate_lt2:.0f}%)  "
                    f"avg {_format_time(avg_t)}\033[0m"
                )
                continue

            if m_fail:
                idx, total, pdb, err = m_fail.groups()
                n_done = int(idx)
                n_total = int(total)
                print(
                    f"  {n_done:3d}/{n_total}  {pdb}  "
                    f"\033[31mFAILED\033[0m: {err}"
                )
                continue

            # Summary section — print as-is
            if _SUMMARY_RE.search(stripped):
                in_summary = True
            if in_summary:
                if stripped.startswith("Test Case") or stripped.startswith("Test Suite"):
                    in_summary = False
                    continue
                if stripped.startswith("Results:"):
                    print(f"  {stripped}")
                    in_summary = False
                    continue
                print(f"  {stripped}")
                continue

            # Verbose mode: show everything
            if verbose:
                print(stripped)

        proc.wait()
    finally:
        if log_file is not None:
            log_file.close()

    elapsed = time.time() - t0
    output = "".join(collected)
    success = proc.returncode == 0

    return success, elapsed, output


def print_config_summary(args, methods):
    """Print human-readable configuration summary."""
    scoring_str = ", ".join(SCORING_LABELS.get(m, m) for m, _, _ in methods)
    dataset_str = f"CASF-2016 (first {args.quick})" if args.quick else "CASF-2016 (225 complexes)"
    strain_on = args.strain_penalty and not args.no_strain_penalty
    grad_str = "numerical (FD)" if args.no_analytical_gradients else "analytical"

    print()
    print(f"\033[1mDruse Benchmark\033[0m  v{VERSION}")
    print()
    print(f"  {dataset_str}  |  {scoring_str}  |  {args.preset or 'custom'} preset")
    print(f"  pop={args.population} gen={args.generations} runs={args.runs}  "
          f"grid={args.grid_spacing}A  LS every {args.local_search_freq} gen x{args.local_search_steps}  "
          f"{grad_str}")

    flags = []
    if not args.no_ligand_flex:
        flags.append("ligand-flex")
    if args.flex_residues:
        flags.append("flex-receptor")
    if strain_on:
        flags.append(f"strain(>{args.strain_threshold})")
    if args.gfn2_opt:
        flags.append(f"gfn2-opt({args.gfn2_opt_level})")
    if args.gfn2_rescoring:
        flags.append("gfn2-rescore")
    if args.exploration_ratio > 0:
        flags.append(f"explore({args.exploration_ratio:.0%})")
    if args.rerank_top > 0:
        flags.append(f"rerank(top{args.rerank_top}x{args.rerank_variants})")
    if flags:
        print(f"  {' '.join(flags)}")

    print()


def main():
    args = parse_args()
    apply_preset(args)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = resolve_scoring_methods(args, run_timestamp)
    build_config(args)  # write Benchmark/.bench_config.json

    print_config_summary(args, methods)

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure project is built
    sys.stdout.write("Building... ")
    sys.stdout.flush()
    build = subprocess.run(
        ["xcodebuild", "-project", str(PROJECT), "-scheme", "Druse",
         "-configuration", "Release", "build"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if build.returncode != 0:
        print("\033[31mFAILED\033[0m")
        print("Run `xcodebuild build` manually to see errors")
        if args.verbose:
            print(build.stdout + build.stderr)
        sys.exit(1)
    print("\033[32mOK\033[0m\n")

    # Run each scoring method
    results_files = []
    total_t0 = time.time()

    for i, (method_key, test_name, output_file) in enumerate(methods):
        label = SCORING_LABELS.get(method_key, method_key)
        if len(methods) > 1:
            print(f"\033[1m[{i+1}/{len(methods)}] {label}\033[0m\n")

        # Update config file with output file and per-method overrides
        update_config_output_file(output_file)
        if method_key == "gfn2":
            update_config_gfn2_scoring(True)

        success, elapsed, output = run_xcodebuild(test_name, os.environ.copy(),
                                                    args.verbose,
                                                    druse_logs=args.druse_logs)

        status = "\033[32mPASSED\033[0m" if success else "\033[31mFAILED\033[0m"
        elapsed_str = f"{elapsed/60:.1f} min" if elapsed > 120 else f"{elapsed:.0f}s"
        print(f"\n  {label}: {status} in {elapsed_str}")

        results_path = Path(args.output_dir) / output_file
        if results_path.exists():
            results_files.append(str(results_path))

    # Clean up config file
    CONFIG_FILE.unlink(missing_ok=True)

    total_elapsed = time.time() - total_t0
    total_str = f"{total_elapsed/60:.1f} min" if total_elapsed > 120 else f"{total_elapsed:.0f}s"
    print(f"\n  Total: {total_str}")
    for r in results_files:
        print(f"  -> {r}")

    # Optionally run analysis
    if args.analyze and results_files:
        print("\nRunning analysis...")
        analyze_script = BENCHMARK_DIR / "analyze.py"
        if analyze_script.exists():
            subprocess.run(
                [sys.executable, str(analyze_script),
                 "--results"] + results_files + ["--report"],
                cwd=str(ROOT),
            )
        else:
            print(f"  analyze.py not found at {analyze_script}")


if __name__ == "__main__":
    main()

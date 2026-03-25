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
  python Benchmark/run_benchmark.py --casf --scoring vina,drusina,druseaf,gfn2

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
    "gfn2":     "testCASF_DrusinaGFN2",
    "all":      "testCASF_All",
}

SCORING_LABELS = {
    "vina":     "Vina",
    "drusina":  "Drusina",
    "druseaf":  "Druse Affinity",
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
                    help="Scoring method(s): vina, drusina, druseaf, gfn2, all "
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
        methods = ["vina", "drusina", "druseaf", "gfn2"]
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


def run_xcodebuild(test_method: str, env: dict, verbose: bool,
                    druse_logs: str | None = None) -> tuple:
    """Run a single benchmark test via xcodebuild. Returns (success, duration_s, output).

    druse_logs: None = no streaming; "-" = stream to terminal; path = stream to file.
    """
    test_id = f"DruseTests/BenchmarkRunner/{test_method}"
    cmd = [
        "xcodebuild", "test",
        "-project", str(PROJECT),
        "-scheme", "DruseTests",
        f"-only-testing:{test_id}",
    ]

    t0 = time.time()

    if druse_logs is not None:
        # Stream output in real time via Popen
        log_file = None
        if druse_logs != "-":
            log_file = open(druse_logs, "a")

        collected = []
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT),
        )
        try:
            for line in proc.stdout:
                collected.append(line)
                if log_file is not None:
                    log_file.write(line)
                    log_file.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            proc.wait()
        finally:
            if log_file is not None:
                log_file.close()

        elapsed = time.time() - t0
        output = "".join(collected)
        success = proc.returncode == 0
    else:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            cwd=str(ROOT), timeout=3600 * 6,  # 6 hour timeout
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        success = result.returncode == 0

        if verbose:
            print(output)
        else:
            # Print just the benchmark lines (skip xcodebuild noise)
            for line in output.split("\n"):
                stripped = line.strip()
                if any(k in stripped for k in [
                    "[", "===", "Scoring Power", "Docking Power",
                    "RMSD", "Pearson", "Performance", "Results:",
                    "passed", "failed", "FAILED", "SUCCEEDED",
                    "Summary", "GFN2", "benchmark",
                ]):
                    print(stripped)

    return success, elapsed, output


def print_config_summary(args, methods):
    """Print human-readable configuration summary."""
    print("=" * 70)
    print(f"  DRUSE BENCHMARK RUNNER  ({VERSION})")
    print("=" * 70)
    print()
    print(f"  Dataset:     CASF-2016{f' (first {args.quick})' if args.quick else ' (full, 225 complexes)'}")
    print(f"  Scoring:     {', '.join(SCORING_LABELS.get(m, m) for m, _, _ in methods)}")
    print(f"  Preset:      {args.preset or 'custom'}")
    print()
    print(f"  GA:          pop={args.population}, gen={args.generations}, runs={args.runs}")
    print(f"  Grid:        {args.grid_spacing} A")
    print(f"  Local search: every {args.local_search_freq} gen, {args.local_search_steps} steps")
    print(f"  Gradients:   {'numerical (FD)' if args.no_analytical_gradients else 'analytical'}")
    print(f"  Ligand flex:  {'off' if args.no_ligand_flex else 'on'}")
    print(f"  Flex receptor: {'on (auto-select)' if args.flex_residues else 'off'}")
    print(f"  Charges:     {args.charge_method}")
    print()

    strain_on = args.strain_penalty and not args.no_strain_penalty
    print(f"  Strain penalty: {'on' if strain_on else 'off'}"
          f"{f' (threshold={args.strain_threshold}, weight={args.strain_weight})' if strain_on else ''}")

    if args.gfn2_opt:
        print(f"  GFN2 opt:    on (level={args.gfn2_opt_level}, solv={args.gfn2_solvation}, "
              f"top={args.gfn2_top_poses}, blend={args.gfn2_blend_weight})")
    else:
        print(f"  GFN2 opt:    off")

    if args.gfn2_rescoring:
        print(f"  GFN2 rescoring: on (D4 + solvation on best pose, ~2ms/complex)")
    print()
    if args.druse_logs is not None:
        dest = "terminal" if args.druse_logs == "-" else args.druse_logs
        print(f"  Druse logs:  streaming to {dest}")
    print(f"  Output:      {args.output_dir}/")
    if args.output_prefix:
        print(f"  Prefix:      {args.output_prefix}")
    print("=" * 70)
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
    print("Building Druse...")
    build = subprocess.run(
        ["xcodebuild", "-project", str(PROJECT), "-scheme", "Druse",
         "-configuration", "Release", "build"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if build.returncode != 0:
        print("BUILD FAILED — run `xcodebuild build` manually to see errors")
        if args.verbose:
            print(build.stdout + build.stderr)
        sys.exit(1)
    print("Build OK\n")

    # Run each scoring method
    results_files = []
    total_t0 = time.time()

    for i, (method_key, test_name, output_file) in enumerate(methods):
        label = SCORING_LABELS.get(method_key, method_key)
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{len(methods)}] {label}")
        print(f"{'=' * 60}\n")

        # Update config file with output file and per-method overrides
        update_config_output_file(output_file)
        if method_key == "gfn2":
            update_config_gfn2_scoring(True)

        success, elapsed, output = run_xcodebuild(test_name, os.environ.copy(),
                                                    args.verbose,
                                                    druse_logs=args.druse_logs)

        status = "PASSED" if success else "FAILED"
        print(f"\n  {label}: {status} ({elapsed:.1f}s)")

        results_path = Path(args.output_dir) / output_file
        if results_path.exists():
            results_files.append(str(results_path))
            print(f"  Results: {results_path}")

    # Clean up config file
    CONFIG_FILE.unlink(missing_ok=True)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results: {', '.join(results_files)}")
    print(f"{'=' * 60}")

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

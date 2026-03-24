#!/usr/bin/env python3
"""
Analyze CASF-2016 benchmark results across scoring methods.

Compares Vina, Drusina, and Druse AF on:
  - Scoring Power: Pearson r (predicted score vs experimental pKd)
  - Docking Power: RMSD success rates at multiple thresholds

Usage:
  python Benchmark/analyze.py                    # auto-detect all results
  python Benchmark/analyze.py --report           # + generate markdown report
  python Benchmark/analyze.py --results casf_vina.json casf_drusina.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    raise SystemExit(
        f"Missing: {e}\n"
        "Install: uv pip install numpy pandas scipy matplotlib --python /Users/vitruves/Developer/Tools/py310/bin/python"
    )

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Benchmark" / "results"
REPORTS_DIR = ROOT / "Benchmark" / "reports"

THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]


def analyze_one(results_path: Path) -> dict:
    """Analyze a single results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    label = data.get("scoringMethod", results_path.stem)
    entries = [e for e in data["entries"] if e["success"]]
    if not entries:
        return {"label": label, "n": 0}

    df = pd.DataFrame(entries)

    # Scoring power
    scored = df.dropna(subset=["experimental_pKd", "best_energy"])
    r_val, p_val = stats.pearsonr(scored["experimental_pKd"], scored["best_energy"]) if len(scored) > 5 else (0, 1)

    # Docking power
    rmsd_df = df.dropna(subset=["best_rmsd"])
    rmsds = rmsd_df["best_rmsd"].values

    dp = {}
    for t in THRESHOLDS:
        n = int(np.sum(rmsds < t))
        dp[t] = {"count": n, "total": len(rmsds), "rate": round(n / max(len(rmsds), 1) * 100, 1)}

    return {
        "label": label,
        "file": results_path.name,
        "total": len(data["entries"]),
        "successful": len(entries),
        "failed": len(data["entries"]) - len(entries),
        "pearson_r": round(r_val, 4),
        "pearson_p": round(p_val, 6),
        "n_scored": len(scored),
        "docking_power": dp,
        "n_rmsd": len(rmsds),
        "mean_rmsd": round(float(np.mean(rmsds)), 2) if len(rmsds) > 0 else None,
        "median_rmsd": round(float(np.median(rmsds)), 2) if len(rmsds) > 0 else None,
        "mean_energy": round(float(df["best_energy"].mean()), 2),
        "mean_time_ms": round(float(df["docking_time_ms"].mean()), 0),
        "total_time_s": round(float(df["docking_time_ms"].sum() / 1000), 1),
        "config": data.get("config", {}),
        "df": df,  # keep for plotting
    }


def print_comparison(all_metrics: list[dict]):
    """Print side-by-side comparison table."""
    labels = [m["label"] for m in all_metrics]
    col_w = max(12, max(len(l) for l in labels) + 2)

    print("\n" + "=" * 60)
    print("  CASF-2016 Benchmark Comparison")
    print("=" * 60)

    # Header
    header = f"  {'Metric':<25}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("  " + "-" * (25 + col_w * len(labels)))

    # Complexes
    row = f"  {'Succeeded':<25}"
    for m in all_metrics:
        row += f"{m['successful']}/{m['total']:>{col_w-3}}   "
    print(row)

    # Scoring power
    row = f"  {'Pearson r (score/pKd)':<25}"
    for m in all_metrics:
        row += f"{m['pearson_r']:>{col_w}.4f}"
    print(row)

    # Docking power at each threshold
    for t in THRESHOLDS:
        row = f"  {'RMSD < ' + f'{t:.1f}A':<25}"
        for m in all_metrics:
            dp = m["docking_power"].get(t, {})
            rate = dp.get("rate", 0)
            row += f"{rate:>{col_w-1}.1f}%"
        print(row)

    # Mean/median RMSD
    row = f"  {'Mean RMSD':<25}"
    for m in all_metrics:
        val = m.get("mean_rmsd")
        row += f"{f'{val:.2f}A' if val else 'N/A':>{col_w}}"
    print(row)

    row = f"  {'Median RMSD':<25}"
    for m in all_metrics:
        val = m.get("median_rmsd")
        row += f"{f'{val:.2f}A' if val else 'N/A':>{col_w}}"
    print(row)

    # Performance
    row = f"  {'ms/complex':<25}"
    for m in all_metrics:
        row += f"{m['mean_time_ms']:>{col_w}.0f}"
    print(row)

    row = f"  {'Total time (s)':<25}"
    for m in all_metrics:
        row += f"{m['total_time_s']:>{col_w}.1f}"
    print(row)

    print()


def generate_plots(all_metrics: list[dict]):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n_methods = len(all_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(max(15, 5 * n_methods), 5))

    colors = {"Vina": "steelblue", "Drusina": "darkorange", "DruseAF": "forestgreen"}

    # Plot 1: Scoring scatter
    ax = axes[0]
    for m in all_metrics:
        df = m["df"]
        scored = df.dropna(subset=["experimental_pKd", "best_energy"])
        c = colors.get(m["label"], "gray")
        ax.scatter(scored["experimental_pKd"], scored["best_energy"],
                   alpha=0.3, s=8, c=c, label=f"{m['label']} (r={m['pearson_r']:.3f})", edgecolors="none")
    ax.set_xlabel("Experimental pKd")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Scoring Power")
    ax.legend(fontsize=8)

    # Plot 2: RMSD histograms
    ax = axes[1]
    for m in all_metrics:
        df = m["df"]
        rmsds = df.dropna(subset=["best_rmsd"])["best_rmsd"].values
        c = colors.get(m["label"], "gray")
        ax.hist(rmsds, bins=30, alpha=0.5, color=c, label=m["label"], edgecolor="white", linewidth=0.3)
    ax.axvline(x=2.0, color="red", ls="--", lw=1.5, label="2.0 A")
    ax.set_xlabel("RMSD (A)")
    ax.set_ylabel("Count")
    ax.set_title("Docking Power: RMSD Distribution")
    ax.legend(fontsize=8)

    # Plot 3: Docking power bar chart
    ax = axes[2]
    x = np.arange(len(THRESHOLDS))
    width = 0.8 / max(n_methods, 1)
    for i, m in enumerate(all_metrics):
        rates = [m["docking_power"].get(t, {}).get("rate", 0) for t in THRESHOLDS]
        c = colors.get(m["label"], "gray")
        ax.bar(x + i * width - 0.4 + width / 2, rates, width, label=m["label"], color=c, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"<{t:.0f}A" for t in THRESHOLDS])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Docking Power: Success Rates")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "casf_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot: {plot_path.relative_to(ROOT)}")
    return str(plot_path.relative_to(ROOT))


def generate_report(all_metrics: list[dict], plot_path: str | None = None):
    """Write markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "benchmark_report.md"
    labels = [m["label"] for m in all_metrics]

    lines = [
        "# Druse CASF-2016 Benchmark Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nScoring methods compared: {', '.join(labels)}",
        "",
        "## Configuration",
        "",
    ]
    if all_metrics:
        cfg = all_metrics[0].get("config", {})
        lines += [
            f"- Population: {cfg.get('populationSize', '?')}",
            f"- Generations: {cfg.get('generations', '?')}",
            f"- Grid spacing: {cfg.get('gridSpacing', '?')} A",
            f"- Runs: {cfg.get('numRuns', '?')}",
        ]

    # Scoring power table
    lines += [
        "",
        "## Scoring Power (Pearson r: score vs experimental pKd)",
        "",
        "| Method | Pearson r | n |",
        "|--------|-----------|---|",
    ]
    for m in all_metrics:
        lines.append(f"| {m['label']} | {m['pearson_r']:.4f} | {m['n_scored']} |")

    # Docking power table
    lines += [
        "",
        "## Docking Power (RMSD success rates)",
        "",
        "| Threshold | " + " | ".join(labels) + " |",
        "|-----------|" + "|".join(["--------"] * len(labels)) + "|",
    ]
    for t in THRESHOLDS:
        row = f"| < {t:.1f} A"
        for m in all_metrics:
            dp = m["docking_power"].get(t, {})
            row += f" | {dp.get('rate', 0):.1f}%"
        row += " |"
        lines.append(row)

    # RMSD stats
    lines += ["", "| Stat | " + " | ".join(labels) + " |",
              "|------|" + "|".join(["------"] * len(labels)) + "|"]
    for stat, key in [("Mean RMSD", "mean_rmsd"), ("Median RMSD", "median_rmsd")]:
        row = f"| {stat}"
        for m in all_metrics:
            v = m.get(key)
            row += f" | {v:.2f} A" if v else " | N/A"
        lines.append(row + " |")

    if plot_path:
        lines += ["", f"![Comparison]({plot_path})", ""]

    # Performance
    lines += [
        "", "## Performance", "",
        "| Method | ms/complex | Total (s) |",
        "|--------|-----------|-----------|",
    ]
    for m in all_metrics:
        lines.append(f"| {m['label']} | {m['mean_time_ms']:.0f} | {m['total_time_s']:.1f} |")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CASF-2016 benchmark results")
    parser.add_argument("--results", nargs="*", default=None,
                        help="Result JSON files (default: auto-detect all in Benchmark/results/)")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    args = parser.parse_args()

    # Auto-detect result files
    if args.results:
        paths = [Path(r) if Path(r).exists() else RESULTS_DIR / r for r in args.results]
    else:
        paths = sorted(RESULTS_DIR.glob("casf_*.json"))

    if not paths:
        print("No results found in Benchmark/results/")
        print("Run benchmarks first:")
        print("  xcodebuild test ... -only-testing:DruseTests/BenchmarkRunner/testCASF_Vina")
        return

    print(f"Analyzing {len(paths)} result file(s)...")
    all_metrics = []
    for p in paths:
        if p.exists():
            m = analyze_one(p)
            if m.get("n", 0) > 0 or m.get("successful", 0) > 0:
                all_metrics.append(m)
                print(f"  {p.name}: {m['label']} — {m['successful']} complexes, r={m['pearson_r']:.4f}")

    if not all_metrics:
        print("No valid results to analyze")
        return

    print_comparison(all_metrics)
    plot_path = generate_plots(all_metrics)

    if args.report or len(all_metrics) > 1:
        generate_report(all_metrics, plot_path)


if __name__ == "__main__":
    main()

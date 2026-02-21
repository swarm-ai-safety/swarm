#!/usr/bin/env python3
"""Plot contract screening multi-seed sweep results.

Reads the CSV produced by sweep_contract_screening.py and generates:
- Violin/box plots of key screening metrics across seeds
- Per-pool quality distribution (box plot)
- Per-pool welfare distribution (box plot)
- Summary statistics table

Usage:
    python scripts/plot_contract_screening_sweep.py [--csv PATH] [--out DIR]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = "runs/contract_screening_sweep/sweep_results.csv"
DEFAULT_OUT = "runs/contract_screening_sweep/plots"

POOL_COLORS = {
    "truthful": "#2ecc71",
    "fair": "#3498db",
    "default": "#e74c3c",
}
POOL_LABELS = {
    "truthful": "Truthful Auction",
    "fair": "Fair Division",
    "default": "Default Market",
}


def load_sweep_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric fields
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def plot_screening_metrics(rows: list[dict], out_dir: Path):
    """Violin/box plots of separation_quality, infiltration_rate, welfare_delta, attack_displacement."""
    metrics = [
        ("separation_quality", "Separation Quality", "#2ecc71"),
        ("infiltration_rate", "Infiltration Rate", "#e74c3c"),
        ("welfare_delta", "Welfare Delta\n(governed - default)", "#3498db"),
        ("attack_displacement", "Attack Displacement", "#e67e22"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Contract Screening Metrics Across Seeds", fontsize=16, fontweight="bold", y=1.02)

    for ax, (key, label, color) in zip(axes, metrics, strict=True):
        data = [r[key] for r in rows]
        parts = ax.violinplot(data, positions=[0], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("navy")

        # Overlay individual points
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(data))
        ax.scatter(jitter, data, color=color, alpha=0.7, s=30, zorder=5, edgecolors="white", linewidth=0.5)

        ax.set_ylabel(label)
        ax.set_xticks([])
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.set_title(f"{mean_val:.3f} +/- {std_val:.3f}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "screening_metrics_violin.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'screening_metrics_violin.png'}")
    plt.close(fig)


def plot_pool_quality(rows: list[dict], out_dir: Path):
    """Box plot of per-pool quality across seeds."""
    pools = ["truthful", "fair", "default"]
    data = {p: [r[f"pool_avg_quality_{p}"] for r in rows] for p in pools}

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(pools))
    bp = ax.boxplot(
        [data[p] for p in pools],
        positions=positions,
        patch_artist=True,
        widths=0.5,
    )
    for patch, pool in zip(bp["boxes"], pools, strict=True):
        patch.set_facecolor(POOL_COLORS[pool])
        patch.set_alpha(0.6)

    # Overlay points
    rng = np.random.default_rng(42)
    for i, pool in enumerate(pools):
        jitter = rng.uniform(-0.1, 0.1, len(data[pool]))
        ax.scatter(
            i + jitter, data[pool],
            color=POOL_COLORS[pool], alpha=0.7, s=30, zorder=5,
            edgecolors="white", linewidth=0.5,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([POOL_LABELS[p] for p in pools])
    ax.set_ylabel("Avg Quality (p)")
    ax.set_title("Per-Pool Quality Distribution Across Seeds", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "pool_quality_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'pool_quality_boxplot.png'}")
    plt.close(fig)


def plot_pool_welfare(rows: list[dict], out_dir: Path):
    """Box plot of per-pool welfare across seeds."""
    pools = ["truthful", "fair", "default"]
    data = {p: [r[f"pool_welfare_{p}"] for r in rows] for p in pools}

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(pools))
    bp = ax.boxplot(
        [data[p] for p in pools],
        positions=positions,
        patch_artist=True,
        widths=0.5,
    )
    for patch, pool in zip(bp["boxes"], pools, strict=True):
        patch.set_facecolor(POOL_COLORS[pool])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, pool in enumerate(pools):
        jitter = rng.uniform(-0.1, 0.1, len(data[pool]))
        ax.scatter(
            i + jitter, data[pool],
            color=POOL_COLORS[pool], alpha=0.7, s=30, zorder=5,
            edgecolors="white", linewidth=0.5,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([POOL_LABELS[p] for p in pools])
    ax.set_ylabel("Avg Welfare per Interaction")
    ax.set_title("Per-Pool Welfare Distribution Across Seeds", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "pool_welfare_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'pool_welfare_boxplot.png'}")
    plt.close(fig)


def print_summary_table(rows: list[dict]):
    """Print summary statistics table."""
    metrics = [
        "separation_quality",
        "infiltration_rate",
        "welfare_delta",
        "attack_displacement",
        "pool_avg_quality_truthful",
        "pool_avg_quality_fair",
        "pool_avg_quality_default",
        "pool_welfare_truthful",
        "pool_welfare_fair",
        "pool_welfare_default",
    ]

    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 66)
    for m in metrics:
        vals = [r[m] for r in rows]
        arr = np.array(vals)
        print(f"{m:<30} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot contract screening sweep results")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to sweep CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output directory for plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run sweep_contract_screening.py first.")
        return 1

    print(f"Loading sweep results from {csv_path}...")
    rows = load_sweep_csv(csv_path)
    print(f"  {len(rows)} runs loaded")

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    print("\nGenerating plots...")
    plot_screening_metrics(rows, out_dir)
    plot_pool_quality(rows, out_dir)
    plot_pool_welfare(rows, out_dir)

    print_summary_table(rows)

    print(f"\nAll plots saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

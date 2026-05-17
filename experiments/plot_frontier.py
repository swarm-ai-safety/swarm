"""Plot capability-safety Pareto frontier from experiment results.

Usage:
    python experiments/plot_frontier.py runs/frontier/<run_dir>
    python experiments/plot_frontier.py runs/frontier/<run_dir> --tail-analysis
    python experiments/plot_frontier.py runs/frontier/<run_dir> --distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Color scheme ───────────────────────────────────────────────────────

CONFIG_COLORS = {
    "tight": "#d62728",
    "tight-moderate": "#e377c2",
    "moderate": "#ff7f0e",
    "moderate-light": "#bcbd22",
    "light": "#2ca02c",
    "loose": "#1f77b4",
    "oracle": "#7f7f7f",
}

TASK_COLORS = {
    "routing": "#1f77b4",
    "coordination": "#ff7f0e",
    "allocation": "#2ca02c",
    "long_horizon": "#d62728",
}


def _pareto_frontier(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract Pareto-optimal points (maximize both x and y)."""
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    frontier_x = [x_sorted[0]]
    frontier_y = [y_sorted[0]]
    max_y = y_sorted[0]

    for i in range(1, len(x_sorted)):
        if y_sorted[i] >= max_y:
            frontier_x.append(x_sorted[i])
            frontier_y.append(y_sorted[i])
            max_y = y_sorted[i]

    return np.array(frontier_x), np.array(frontier_y)


def plot_frontier_scatter(
    df: pd.DataFrame,
    task_type: str,
    ax: plt.Axes | None = None,
    show_seeds: bool = True,
) -> plt.Axes:
    """Plot capability vs safety proxy scatter with Pareto frontier."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Per-config means
    summary = df.groupby("gov_config")[["capability", "safety_proxy"]].agg(["mean", "std"])
    configs = summary.index.tolist()

    # Plot individual seed points (faded)
    if show_seeds:
        for config in configs:
            mask = df["gov_config"] == config
            color = CONFIG_COLORS.get(config, "#999999")
            ax.scatter(
                df.loc[mask, "capability"],
                df.loc[mask, "safety_proxy"],
                alpha=0.15,
                s=15,
                c=color,
                zorder=1,
            )

    # Plot config means with error bars
    for config in configs:
        color = CONFIG_COLORS.get(config, "#999999")
        cap_mean = summary.loc[config, ("capability", "mean")]
        cap_std = summary.loc[config, ("capability", "std")]
        saf_mean = summary.loc[config, ("safety_proxy", "mean")]
        saf_std = summary.loc[config, ("safety_proxy", "std")]

        ax.errorbar(
            cap_mean,
            saf_mean,
            xerr=cap_std,
            yerr=saf_std,
            fmt="o",
            color=color,
            markersize=10,
            capsize=4,
            linewidth=1.5,
            label=config,
            zorder=3,
        )

    # Fit and draw Pareto frontier
    cap_means = np.array([summary.loc[c, ("capability", "mean")] for c in configs])
    saf_means = np.array([summary.loc[c, ("safety_proxy", "mean")] for c in configs])
    front_x, front_y = _pareto_frontier(cap_means, saf_means)
    ax.plot(front_x, front_y, "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier", zorder=2)

    ax.set_xlabel("Capability (completion × 0.6 + fidelity × 0.3 + efficiency × 0.1)", fontsize=11)
    ax.set_ylabel("Safety Proxy (governance strength × capability)", fontsize=11)
    ax.set_title(f"Capability–Safety Frontier: {task_type}", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    return ax


def plot_frontier_overlay(
    results: dict[str, pd.DataFrame],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay Pareto frontiers from multiple benchmark types (Experiment 3)."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 7))

    for task_type, df in results.items():
        summary = df.groupby("gov_config")[["capability", "safety_proxy"]].mean()

        color = TASK_COLORS.get(task_type, "#999999")

        cap_means = summary["capability"].values
        saf_means = summary["safety_proxy"].values

        # Plot means
        ax.scatter(cap_means, saf_means, c=color, s=60, alpha=0.8, zorder=2)

        # Frontier
        front_x, front_y = _pareto_frontier(cap_means, saf_means)
        ax.plot(front_x, front_y, "-", color=color, linewidth=2, alpha=0.7, label=task_type, zorder=1)

    ax.set_xlabel("Capability", fontsize=11)
    ax.set_ylabel("Safety Proxy", fontsize=11)
    ax.set_title("Cross-Task Frontier Geometry Comparison", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    return ax


def plot_tail_analysis(
    df: pd.DataFrame,
    task_type: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot distributional tail analysis (Experiment 4).

    Shows mean p, 5th percentile p, and std for each governance config.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    summary = df.groupby("gov_config")["p"].agg(
        ["mean", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]
    )
    summary.columns = ["mean", "std", "p5", "p95"]
    summary = summary.sort_values("mean")

    configs = summary.index.tolist()
    x = range(len(configs))

    # Mean with std band
    ax.bar(x, summary["mean"], width=0.6, alpha=0.6, color="#1f77b4", label="Mean p")
    ax.errorbar(x, summary["mean"], yerr=summary["std"], fmt="none", color="black", capsize=4, label="±1 std")

    # 5th percentile (tail risk)
    ax.scatter(x, summary["p5"], marker="v", s=80, color="#d62728", zorder=5, label="5th percentile (tail risk)")

    # 95th percentile
    ax.scatter(x, summary["p95"], marker="^", s=80, color="#2ca02c", zorder=5, label="95th percentile")

    ax.set_xticks(list(x))
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("p = P(beneficial)", fontsize=11)
    ax.set_title(f"Distributional Analysis: {task_type}", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_distribution_violin(
    df: pd.DataFrame,
    task_type: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Violin plot of p distribution per governance config."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    configs = sorted(df["gov_config"].unique(), key=lambda c: df[df["gov_config"] == c]["capability"].mean())

    data = [df[df["gov_config"] == c]["p"].values for c in configs]
    parts = ax.violinplot(data, positions=range(len(configs)), showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        config = configs[i]
        color = CONFIG_COLORS.get(config, "#999999")
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("p = P(beneficial)", fontsize=11)
    ax.set_title(f"p Distribution by Governance Config: {task_type}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_frontier_from_dir(run_dir: str | Path, tail_analysis: bool = False) -> None:
    """Load results from a run directory and generate all plots."""
    run_dir = Path(run_dir)
    csv_dir = run_dir / "csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load all available task results
    results = {}
    for csv_path in sorted(csv_dir.glob("*_raw.csv")):
        task_type = csv_path.stem.replace("_raw", "")
        results[task_type] = pd.read_csv(csv_path)

    if not results:
        print(f"No CSV files found in {csv_dir}")
        return

    # Per-task frontier scatter plots
    for task_type, df in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        plot_frontier_scatter(df, task_type, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{task_type}_frontier.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {task_type}_frontier.png")

    # Cross-task overlay (if multiple tasks)
    if len(results) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        plot_frontier_overlay(results, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / "cross_task_frontier.png", dpi=150)
        plt.close(fig)
        print("  Saved cross_task_frontier.png")

    # Tail analysis plots
    if tail_analysis:
        for task_type, df in results.items():
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plot_tail_analysis(df, task_type, ax=axes[0])
            plot_distribution_violin(df, task_type, ax=axes[1])
            fig.tight_layout()
            fig.savefig(plots_dir / f"{task_type}_distribution.png", dpi=150)
            plt.close(fig)
            print(f"  Saved {task_type}_distribution.png")

    print(f"\nAll plots saved to {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pareto frontier from experiment results")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("--tail-analysis", action="store_true", help="Include distributional tail plots")
    parser.add_argument("--distribution", action="store_true", help="Alias for --tail-analysis")
    args = parser.parse_args()

    plot_frontier_from_dir(args.run_dir, tail_analysis=args.tail_analysis or args.distribution)


if __name__ == "__main__":
    main()

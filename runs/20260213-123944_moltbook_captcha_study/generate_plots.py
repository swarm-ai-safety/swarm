#!/usr/bin/env python
"""Phase 3: Generate plots for Moltbook CAPTCHA full study."""

import sys
from pathlib import Path

import matplotlib  # isort: skip
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  # isort: skip
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "sweep_results.csv"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def ci_95(vals):
    """Return mean and 95% CI half-width."""
    m = np.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return m, 1.96 * se


def plot_metric_vs_param(df, param, metric, ylabel, title, filename):
    """Bar plot of metric vs parameter values with 95% CI error bars."""
    values = sorted(df[param].unique())
    means, errs = [], []
    for v in values:
        group = df[df[param] == v][metric].dropna().values
        m, e = ci_95(group)
        means.append(m)
        errs.append(e)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(values))
    ax.bar(x, means, yerr=errs, capsize=5, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param.split(".")[-1].replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_welfare_toxicity_tradeoff(df, param):
    """Scatter of welfare vs toxicity colored by param."""
    fig, ax = plt.subplots(figsize=(8, 6))
    values = sorted(df[param].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(values), max(values))

    for v in values:
        sub = df[df[param] == v]
        color = cmap(norm(v))
        ax.scatter(sub["toxicity_rate"], sub["welfare"], alpha=0.5, c=[color], label=f"{v}", s=30)

    ax.set_xlabel("Toxicity Rate")
    ax.set_ylabel("Welfare")
    ax.set_title(f"Welfare–Toxicity Tradeoff by {param.split('.')[-1].replace('_', ' ').title()}")
    ax.legend(title=param.split(".")[-1].replace("_", " ").title(), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    filename = f"welfare_toxicity_tradeoff_{param.split('.')[-1]}.png"
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_agent_payoffs(df):
    """Box plot of per-type payoffs."""
    agent_cols = {
        "honest_payoff": "Honest",
        "opportunistic_payoff": "Opportunistic",
        "deceptive_payoff": "Deceptive",
        "adversarial_payoff": "Adversarial",
    }
    present = {k: v for k, v in agent_cols.items() if k in df.columns}
    data = [df[col].dropna().values for col in present.keys()]
    labels = list(present.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Payoff")
    ax.set_title("Agent Payoff by Type (Moltbook CAPTCHA)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "agent_payoff_by_type.png", dpi=150)
    plt.close(fig)
    print("  Saved: agent_payoff_by_type.png")


def plot_heatmap(df, param1, param2, metric, title, filename):
    """Heatmap of metric by two parameters."""
    pivot = df.groupby([param1, param2])[metric].mean().reset_index()
    table = pivot.pivot(index=param1, columns=param2, values=metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(table.values, cmap="RdYlGn" if metric == "welfare" else "RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in table.index])
    ax.set_xlabel(param2.split(".")[-1].replace("_", " ").title())
    ax.set_ylabel(param1.split(".")[-1].replace("_", " ").title())
    ax.set_title(title)

    # Annotate cells
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            ax.text(j, i, f"{val:.1f}" if abs(val) > 1 else f"{val:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(val - table.values.mean()) > table.values.std() else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("Phase 3: Plot Generation — Moltbook CAPTCHA")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    param1 = "governance.moltbook_challenge_difficulty"
    param2 = "governance.collusion_penalty_multiplier"

    print(f"\nLoaded {len(df)} rows. Generating plots...\n")

    # 1. Welfare vs challenge difficulty
    plot_metric_vs_param(df, param1, "welfare", "Welfare",
                         "Welfare vs Challenge Difficulty", "welfare_vs_challenge_difficulty.png")

    # 2. Welfare vs collusion penalty
    plot_metric_vs_param(df, param2, "welfare", "Welfare",
                         "Welfare vs Collusion Penalty Multiplier", "welfare_vs_collusion_penalty.png")

    # 3. Toxicity vs challenge difficulty
    plot_metric_vs_param(df, param1, "toxicity_rate", "Toxicity Rate",
                         "Toxicity vs Challenge Difficulty", "toxicity_vs_challenge_difficulty.png")

    # 4. Toxicity vs collusion penalty
    plot_metric_vs_param(df, param2, "toxicity_rate", "Toxicity Rate",
                         "Toxicity vs Collusion Penalty Multiplier", "toxicity_vs_collusion_penalty.png")

    # 5. Quality gap vs challenge difficulty
    plot_metric_vs_param(df, param1, "quality_gap", "Quality Gap",
                         "Quality Gap vs Challenge Difficulty", "quality_gap_vs_challenge_difficulty.png")

    # 6. Quality gap vs collusion penalty
    plot_metric_vs_param(df, param2, "quality_gap", "Quality Gap",
                         "Quality Gap vs Collusion Penalty Multiplier", "quality_gap_vs_collusion_penalty.png")

    # 7-8. Welfare-toxicity tradeoff scatter
    plot_welfare_toxicity_tradeoff(df, param1)
    plot_welfare_toxicity_tradeoff(df, param2)

    # 9. Agent payoff box plot
    plot_agent_payoffs(df)

    # 10-11. Heatmaps
    plot_heatmap(df, param1, param2, "welfare",
                 "Mean Welfare (Challenge Difficulty x Collusion Penalty)",
                 "heatmap_welfare.png")
    plot_heatmap(df, param1, param2, "toxicity_rate",
                 "Mean Toxicity (Challenge Difficulty x Collusion Penalty)",
                 "heatmap_toxicity.png")

    n_plots = len(list(PLOT_DIR.glob("*.png")))
    print(f"\nTotal: {n_plots} plots generated in {PLOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

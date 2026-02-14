#!/usr/bin/env python
"""Generate plots for collusion tax effect parameter sweep."""

import sys
from pathlib import Path

import matplotlib  # isort: skip
matplotlib.use("Agg")  # isort: skip
import matplotlib.pyplot as plt  # isort: skip
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
CSV_PATH = RUN_DIR / "sweep_results.csv"
PLOT_DIR = RUN_DIR / "plots"


def ci95(series):
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def plot_metric_vs_param(df, param, metric, ylabel, title, filename):
    grouped = df.groupby(param)[metric]
    means = grouped.mean()
    cis = grouped.apply(ci95)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(means))
    ax.bar(x, means.values, yerr=cis.values, capsize=5, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in means.index], fontsize=10)
    ax.set_xlabel(param.split(".")[-1].replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_metric_vs_param_grouped(df, param, group_param, metric, ylabel, title, filename):
    groups = sorted(df[group_param].unique())
    param_vals = sorted(df[param].unique())
    n_groups = len(groups)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(param_vals))

    for i, g in enumerate(groups):
        subset = df[df[group_param] == g]
        grouped = subset.groupby(param)[metric]
        means = grouped.mean()
        cis = grouped.apply(ci95)
        offset = (i - (n_groups - 1) / 2) * width
        label = f"penalty={g}"
        ax.bar(x + offset, means.values, width=width, yerr=cis.values,
               capsize=3, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_vals], fontsize=10)
    ax.set_xlabel(param.split(".")[-1].replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def main() -> int:
    print("=" * 60)
    print("Plot Generation: collusion_tax_effect")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df["rlm_payoff"] = (df["welfare"] - df["honest_payoff"] * 3) / 9
    print(f"Loaded {len(df)} runs\n")

    count = 0

    # 1. Welfare vs tax rate
    plot_metric_vs_param(df, "governance.transaction_tax_rate", "welfare",
                         "Welfare", "Welfare vs Transaction Tax Rate", "welfare_vs_tax.png")
    count += 1

    # 2. Toxicity vs tax rate
    plot_metric_vs_param(df, "governance.transaction_tax_rate", "toxicity_rate",
                         "Toxicity Rate", "Toxicity vs Transaction Tax Rate", "toxicity_vs_tax.png")
    count += 1

    # 3. Quality gap vs tax rate
    plot_metric_vs_param(df, "governance.transaction_tax_rate", "quality_gap",
                         "Quality Gap", "Quality Gap vs Transaction Tax Rate", "quality_gap_vs_tax.png")
    count += 1

    # 4. Welfare vs collusion penalty
    plot_metric_vs_param(df, "governance.collusion_penalty_multiplier", "welfare",
                         "Welfare", "Welfare vs Collusion Penalty Multiplier", "welfare_vs_penalty.png")
    count += 1

    # 5. Toxicity vs collusion penalty
    plot_metric_vs_param(df, "governance.collusion_penalty_multiplier", "toxicity_rate",
                         "Toxicity Rate", "Toxicity vs Collusion Penalty Multiplier", "toxicity_vs_penalty.png")
    count += 1

    # 6. Welfare vs tax rate, grouped by collusion penalty
    plot_metric_vs_param_grouped(df, "governance.transaction_tax_rate",
                                 "governance.collusion_penalty_multiplier", "welfare",
                                 "Welfare", "Welfare vs Tax Rate (by Collusion Penalty)",
                                 "welfare_vs_tax_by_penalty.png")
    count += 1

    # 7. Toxicity vs tax rate, grouped by collusion penalty
    plot_metric_vs_param_grouped(df, "governance.transaction_tax_rate",
                                 "governance.collusion_penalty_multiplier", "toxicity_rate",
                                 "Toxicity Rate", "Toxicity vs Tax Rate (by Collusion Penalty)",
                                 "toxicity_vs_tax_by_penalty.png")
    count += 1

    # 8. Welfare-toxicity tradeoff scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(tax_vals)))
    for tv, color in zip(tax_vals, colors, strict=False):
        subset = df[df["governance.transaction_tax_rate"] == tv]
        ax.scatter(subset["toxicity_rate"], subset["welfare"],
                   c=[color], label=f"tax={tv}", alpha=0.7, s=40)
    ax.set_xlabel("Toxicity Rate", fontsize=12)
    ax.set_ylabel("Welfare", fontsize=12)
    ax.set_title("Welfare-Toxicity Tradeoff", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "welfare_toxicity_tradeoff.png", dpi=150)
    plt.close()
    print("  Saved: welfare_toxicity_tradeoff.png")
    count += 1

    # 9. Agent payoff comparison: honest vs RLM
    fig, ax = plt.subplots(figsize=(8, 5))
    agent_data = [df["honest_payoff"].values, df["rlm_payoff"].values]
    bp = ax.boxplot(agent_data, tick_labels=["Honest", "RLM"], patch_artist=True)
    box_colors = ["#4C72B0", "#C44E52"]
    for patch, color in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Payoff", fontsize=12)
    ax.set_title("Agent Payoff: Honest vs RLM", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "agent_payoff_honest_vs_rlm.png", dpi=150)
    plt.close()
    print("  Saved: agent_payoff_honest_vs_rlm.png")
    count += 1

    # 10. Honest and RLM payoff vs tax rate (grouped)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tax_vals))
    width = 0.35
    for i, (label, col, color) in enumerate([
        ("Honest", "honest_payoff", "#4C72B0"),
        ("RLM", "rlm_payoff", "#C44E52"),
    ]):
        means = [df[df["governance.transaction_tax_rate"] == tv][col].mean() for tv in tax_vals]
        cis = [ci95(df[df["governance.transaction_tax_rate"] == tv][col]) for tv in tax_vals]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width, yerr=cis, capsize=4,
               label=label, alpha=0.85, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in tax_vals], fontsize=10)
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("Mean Payoff", fontsize=12)
    ax.set_title("Agent Payoff by Type vs Tax Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "agent_payoff_vs_tax.png", dpi=150)
    plt.close()
    print("  Saved: agent_payoff_vs_tax.png")
    count += 1

    # 11. RLM payoff vs tax rate
    plot_metric_vs_param(df, "governance.transaction_tax_rate", "rlm_payoff",
                         "RLM Agent Payoff", "RLM Payoff vs Tax Rate",
                         "rlm_payoff_vs_tax.png")
    count += 1

    # 12. Heatmap: welfare by tax rate x collusion penalty
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = df.pivot_table(values="welfare", index="governance.collusion_penalty_multiplier",
                           columns="governance.transaction_tax_rate", aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("Collusion Penalty Multiplier", fontsize=12)
    ax.set_title("Mean Welfare Heatmap", fontsize=13)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Welfare")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "welfare_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: welfare_heatmap.png")
    count += 1

    # 13. Heatmap: toxicity by tax rate x collusion penalty
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot_tox = df.pivot_table(values="toxicity_rate", index="governance.collusion_penalty_multiplier",
                               columns="governance.transaction_tax_rate", aggfunc="mean")
    im = ax.imshow(pivot_tox.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot_tox.columns)))
    ax.set_xticklabels([str(v) for v in pivot_tox.columns])
    ax.set_yticks(range(len(pivot_tox.index)))
    ax.set_yticklabels([str(v) for v in pivot_tox.index])
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("Collusion Penalty Multiplier", fontsize=12)
    ax.set_title("Mean Toxicity Heatmap", fontsize=13)
    for i in range(len(pivot_tox.index)):
        for j in range(len(pivot_tox.columns)):
            ax.text(j, i, f"{pivot_tox.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Toxicity Rate")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "toxicity_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: toxicity_heatmap.png")
    count += 1

    print(f"\nTotal plots generated: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

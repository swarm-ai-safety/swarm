#!/usr/bin/env python
"""Generate plots for baseline governance parameter sweep."""

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
    """Compute 95% confidence interval half-width."""
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def plot_metric_vs_param(df, param, metric, ylabel, title, filename):
    """Bar plot of metric vs parameter with 95% CI error bars."""
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
    """Grouped bar plot: metric vs param, grouped by another param."""
    groups = sorted(df[group_param].unique())
    param_vals = sorted(df[param].unique())
    n_groups = len(groups)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(param_vals))

    for i, g in enumerate(groups):
        subset = df[df[group_param] == g]
        grouped = subset.groupby(param)[metric]
        means = grouped.mean()
        cis = grouped.apply(ci95)
        offset = (i - (n_groups - 1) / 2) * width
        label = f"{group_param.split('.')[-1]}={g}"
        ax.bar(x + offset, means.values, width=width, yerr=cis.values,
               capsize=4, label=label, alpha=0.85)

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
    print("Plot Generation: baseline_governance")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
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

    # 4. Welfare vs circuit breaker
    plot_metric_vs_param(df, "governance.circuit_breaker_enabled", "welfare",
                         "Welfare", "Welfare vs Circuit Breaker", "welfare_vs_cb.png")
    count += 1

    # 5. Toxicity vs circuit breaker
    plot_metric_vs_param(df, "governance.circuit_breaker_enabled", "toxicity_rate",
                         "Toxicity Rate", "Toxicity vs Circuit Breaker", "toxicity_vs_cb.png")
    count += 1

    # 6. Welfare vs tax rate, grouped by circuit breaker
    plot_metric_vs_param_grouped(df, "governance.transaction_tax_rate",
                                 "governance.circuit_breaker_enabled", "welfare",
                                 "Welfare", "Welfare vs Tax Rate (by Circuit Breaker)",
                                 "welfare_vs_tax_by_cb.png")
    count += 1

    # 7. Toxicity vs tax rate, grouped by circuit breaker
    plot_metric_vs_param_grouped(df, "governance.transaction_tax_rate",
                                 "governance.circuit_breaker_enabled", "toxicity_rate",
                                 "Toxicity Rate", "Toxicity vs Tax Rate (by Circuit Breaker)",
                                 "toxicity_vs_tax_by_cb.png")
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

    # 9. Agent payoff by type (box plot)
    agent_cols = {
        "Honest": "honest_payoff",
        "Opportunistic": "opportunistic_payoff",
        "Deceptive": "deceptive_payoff",
    }
    agent_data = [df[col].dropna().values for col in agent_cols.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(agent_data, labels=list(agent_cols.keys()), patch_artist=True)
    box_colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for patch, color in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Payoff", fontsize=12)
    ax.set_title("Agent Payoff by Type", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "agent_payoff_by_type.png", dpi=150)
    plt.close()
    print("  Saved: agent_payoff_by_type.png")
    count += 1

    # 10. Agent payoff by type vs tax rate
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tax_vals))
    width = 0.25
    for i, (label, col) in enumerate(agent_cols.items()):
        means = [df[df["governance.transaction_tax_rate"] == tv][col].mean() for tv in tax_vals]
        cis = [ci95(df[df["governance.transaction_tax_rate"] == tv][col]) for tv in tax_vals]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width=width, yerr=cis, capsize=4,
               label=label, alpha=0.85, color=box_colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in tax_vals], fontsize=10)
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("Mean Payoff", fontsize=12)
    ax.set_title("Agent Payoff by Type vs Tax Rate", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "agent_payoff_vs_tax.png", dpi=150)
    plt.close()
    print("  Saved: agent_payoff_vs_tax.png")
    count += 1

    # 11. Honest payoff vs tax rate
    plot_metric_vs_param(df, "governance.transaction_tax_rate", "honest_payoff",
                         "Honest Agent Payoff", "Honest Payoff vs Tax Rate",
                         "honest_payoff_vs_tax.png")
    count += 1

    print(f"\nTotal plots generated: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

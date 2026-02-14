#!/usr/bin/env python
"""
Phase 3: Generate plots for Agent Lab Research Safety study.

Produces:
  1. Welfare vs transaction tax rate (with 95% CI)
  2. Toxicity vs transaction tax rate (with 95% CI)
  3. Welfare-toxicity tradeoff scatter
  4. Quality gap vs transaction tax rate
  5. Welfare by circuit breaker × collusion detection (grouped bar)
  6. Welfare boxplot by configuration
  7. Heatmap: tax × circuit breaker on welfare
  8. Honest payoff by configuration
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

RUN_DIR = Path(__file__).parent
CSV_PATH = RUN_DIR / "sweep_results.csv"
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def ci_95(data):
    """Return mean, lower, upper for 95% CI."""
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, mean, mean
    se = sp_stats.sem(data)
    ci = se * sp_stats.t.ppf(0.975, n - 1)
    return mean, mean - ci, mean + ci


def plot_welfare_vs_tax(df):
    """Plot 1: Welfare vs transaction tax rate with 95% CI."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    means, lows, highs = [], [], []
    for tax in tax_vals:
        data = df[df["governance.transaction_tax_rate"] == tax]["welfare"]
        m, lo, hi = ci_95(data)
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    ax.errorbar(tax_vals, means,
                yerr=[np.array(means) - np.array(lows),
                      np.array(highs) - np.array(means)],
                fmt="o-", capsize=5, color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Transaction Tax Rate")
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare vs Transaction Tax Rate (95% CI)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "welfare_by_tax.png")
    plt.close(fig)
    print("  welfare_by_tax.png")


def plot_toxicity_vs_tax(df):
    """Plot 2: Toxicity vs transaction tax rate with 95% CI."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    means, lows, highs = [], [], []
    for tax in tax_vals:
        data = df[df["governance.transaction_tax_rate"] == tax]["toxicity_rate"]
        m, lo, hi = ci_95(data)
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    ax.errorbar(tax_vals, means,
                yerr=[np.array(means) - np.array(lows),
                      np.array(highs) - np.array(means)],
                fmt="s-", capsize=5, color="#F44336", linewidth=2, markersize=8)
    ax.set_xlabel("Transaction Tax Rate")
    ax.set_ylabel("Toxicity Rate")
    ax.set_title("Toxicity vs Transaction Tax Rate (95% CI)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "toxicity_by_tax.png")
    plt.close(fig)
    print("  toxicity_by_tax.png")


def plot_welfare_toxicity_tradeoff(df):
    """Plot 3: Welfare-toxicity tradeoff scatter."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(tax_vals)))

    for i, tax in enumerate(tax_vals):
        subset = df[df["governance.transaction_tax_rate"] == tax]
        ax.scatter(subset["toxicity_rate"], subset["welfare"],
                   c=[colors[i]], label=f"tax={tax:.0%}",
                   alpha=0.6, s=40, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Toxicity Rate")
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare-Toxicity Tradeoff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "welfare_toxicity_tradeoff.png")
    plt.close(fig)
    print("  welfare_toxicity_tradeoff.png")


def plot_quality_gap_vs_tax(df):
    """Plot 4: Quality gap vs transaction tax rate."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    means, lows, highs = [], [], []
    for tax in tax_vals:
        data = df[df["governance.transaction_tax_rate"] == tax]["quality_gap"]
        m, lo, hi = ci_95(data)
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    ax.bar(range(len(tax_vals)), means, color="#9C27B0", alpha=0.7, width=0.6)
    ax.set_xticks(range(len(tax_vals)))
    ax.set_xticklabels([f"{t:.0%}" for t in tax_vals])
    ax.set_xlabel("Transaction Tax Rate")
    ax.set_ylabel("Quality Gap")
    ax.set_title("Quality Gap vs Transaction Tax Rate")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "quality_gap_by_tax.png")
    plt.close(fig)
    print("  quality_gap_by_tax.png")


def plot_welfare_grouped_bar(df):
    """Plot 5: Welfare by circuit breaker × collusion detection."""
    fig, ax = plt.subplots()
    configs = [
        (False, False, "Neither"),
        (True, False, "CB only"),
        (False, True, "CD only"),
        (True, True, "CB + CD"),
    ]
    x = np.arange(len(configs))
    width = 0.6

    means = []
    errors = []
    for cb, cd, _ in configs:
        data = df[
            (df["governance.circuit_breaker_enabled"] == cb)
            & (df["governance.collusion_detection_enabled"] == cd)
        ]["welfare"]
        m, lo, hi = ci_95(data)
        means.append(m)
        errors.append([m - lo, hi - m])

    errors = np.array(errors).T
    ax.bar(x, means, width, yerr=errors, capsize=5,
           color=["#EF5350", "#42A5F5", "#66BB6A", "#AB47BC"],
           alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in configs])
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare by Governance Mechanism Combination")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "welfare_by_mechanism.png")
    plt.close(fig)
    print("  welfare_by_mechanism.png")


def plot_welfare_boxplot(df):
    """Plot 6: Welfare boxplot by full configuration."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create config labels
    configs = df.groupby([
        "governance.transaction_tax_rate",
        "governance.circuit_breaker_enabled",
        "governance.collusion_detection_enabled",
    ])

    labels = []
    data_groups = []
    for (tax, cb, cd), group in configs:
        cb_str = "CB" if cb else ""
        cd_str = "CD" if cd else ""
        mechs = "+".join(filter(None, [cb_str, cd_str])) or "none"
        labels.append(f"{tax:.0%}\n{mechs}")
        data_groups.append(group["welfare"].values)

    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_groups)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Welfare")
    ax.set_title("Welfare Distribution by Configuration")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "welfare_boxplot.png")
    plt.close(fig)
    print("  welfare_boxplot.png")


def plot_heatmap_tax_cb(df):
    """Plot 7: Heatmap of tax rate × circuit breaker on welfare."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())
    cb_vals = [False, True]

    matrix = np.zeros((len(cb_vals), len(tax_vals)))
    for i, cb in enumerate(cb_vals):
        for j, tax in enumerate(tax_vals):
            data = df[
                (df["governance.transaction_tax_rate"] == tax)
                & (df["governance.circuit_breaker_enabled"] == cb)
            ]["welfare"]
            matrix[i, j] = data.mean()

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(tax_vals)))
    ax.set_xticklabels([f"{t:.0%}" for t in tax_vals])
    ax.set_yticks(range(len(cb_vals)))
    ax.set_yticklabels(["CB Off", "CB On"])
    ax.set_xlabel("Transaction Tax Rate")
    ax.set_title("Mean Welfare: Tax Rate x Circuit Breaker")

    # Add value annotations
    for i in range(len(cb_vals)):
        for j in range(len(tax_vals)):
            ax.text(j, i, f"{matrix[i, j]:.1f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] < np.median(matrix) else "black")

    fig.colorbar(im, label="Welfare")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "heatmap_tax_cb.png")
    plt.close(fig)
    print("  heatmap_tax_cb.png")


def plot_honest_payoff(df):
    """Plot 8: Honest payoff by configuration."""
    fig, ax = plt.subplots()
    tax_vals = sorted(df["governance.transaction_tax_rate"].unique())

    for cb in [False, True]:
        means, lows, highs = [], [], []
        for tax in tax_vals:
            data = df[
                (df["governance.transaction_tax_rate"] == tax)
                & (df["governance.circuit_breaker_enabled"] == cb)
            ]["honest_payoff"]
            m, lo, hi = ci_95(data)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        label = f"CB={'On' if cb else 'Off'}"
        color = "#4CAF50" if cb else "#FF9800"
        ax.errorbar(tax_vals, means,
                    yerr=[np.array(means) - np.array(lows),
                          np.array(highs) - np.array(means)],
                    fmt="o-", capsize=5, color=color, linewidth=2,
                    markersize=7, label=label)

    ax.set_xlabel("Transaction Tax Rate")
    ax.set_ylabel("Honest Agent Payoff")
    ax.set_title("Honest Agent Payoff vs Tax Rate (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "honest_payoff_by_config.png")
    plt.close(fig)
    print("  honest_payoff_by_config.png")


def main():
    print("=" * 60)
    print("Phase 3: Plot Generation")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} rows")
    print(f"Generating plots to {PLOTS_DIR}/\n")

    plot_welfare_vs_tax(df)
    plot_toxicity_vs_tax(df)
    plot_welfare_toxicity_tradeoff(df)
    plot_quality_gap_vs_tax(df)
    plot_welfare_grouped_bar(df)
    plot_welfare_boxplot(df)
    plot_heatmap_tax_cb(df)
    plot_honest_payoff(df)

    n_plots = len(list(PLOTS_DIR.glob("*.png")))
    print(f"\nGenerated {n_plots} plots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

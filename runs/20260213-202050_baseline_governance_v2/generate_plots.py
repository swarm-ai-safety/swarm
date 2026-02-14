#!/usr/bin/env python
"""Generate plots for baseline governance v2.

New plots over v1:
  - Finer-grained tax rate curve (7 levels)
  - Tax × circuit breaker interaction heatmap
  - Acceptance rate vs tax rate
  - Tax burden by agent type (% drop from baseline)
  - Bootstrap CI error bars throughout
"""

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

COLORS = {
    "honest": "#4C72B0",
    "opportunistic": "#DD8452",
    "deceptive": "#C44E52",
}


def bootstrap_ci(data, n_boot=5000, ci=0.95):
    """Return (lo, hi) for the mean."""
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(means, 100 * alpha), np.percentile(means, 100 * (1 - alpha))


def ci95(series):
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def plot_metric_vs_param(df, param, metric, ylabel, title, filename):
    """Bar plot with 95% CI error bars."""
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
    """Grouped bar plot."""
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
    print("Plot Generation: baseline_governance_v2")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} runs\n")

    count = 0
    tax_param = "governance.transaction_tax_rate"
    cb_param = "governance.circuit_breaker_enabled"
    tax_vals = sorted(df[tax_param].unique())

    # 1. Welfare vs tax rate
    plot_metric_vs_param(df, tax_param, "welfare",
                         "Welfare", "Welfare vs Transaction Tax Rate", "welfare_vs_tax.png")
    count += 1

    # 2. Toxicity vs tax rate
    plot_metric_vs_param(df, tax_param, "toxicity_rate",
                         "Toxicity Rate", "Toxicity vs Transaction Tax Rate", "toxicity_vs_tax.png")
    count += 1

    # 3. Quality gap vs tax rate
    plot_metric_vs_param(df, tax_param, "quality_gap",
                         "Quality Gap", "Quality Gap vs Transaction Tax Rate", "quality_gap_vs_tax.png")
    count += 1

    # 4. Welfare vs circuit breaker
    plot_metric_vs_param(df, cb_param, "welfare",
                         "Welfare", "Welfare vs Circuit Breaker", "welfare_vs_cb.png")
    count += 1

    # 5. Toxicity vs circuit breaker
    plot_metric_vs_param(df, cb_param, "toxicity_rate",
                         "Toxicity Rate", "Toxicity vs Circuit Breaker", "toxicity_vs_cb.png")
    count += 1

    # 6. Welfare vs tax, grouped by circuit breaker (INTERACTION PLOT)
    plot_metric_vs_param_grouped(df, tax_param, cb_param, "welfare",
                                 "Welfare", "Welfare vs Tax Rate (by Circuit Breaker)",
                                 "welfare_vs_tax_by_cb.png")
    count += 1

    # 7. Toxicity vs tax, grouped by circuit breaker
    plot_metric_vs_param_grouped(df, tax_param, cb_param, "toxicity_rate",
                                 "Toxicity Rate", "Toxicity vs Tax Rate (by Circuit Breaker)",
                                 "toxicity_vs_tax_by_cb.png")
    count += 1

    # 8. Welfare-toxicity tradeoff scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(tax_vals)))
    for tv, color in zip(tax_vals, colors, strict=False):
        subset = df[df[tax_param] == tv]
        ax.scatter(subset["toxicity_rate"], subset["welfare"],
                   c=[color], label=f"tax={tv}", alpha=0.6, s=30)
    ax.set_xlabel("Toxicity Rate", fontsize=12)
    ax.set_ylabel("Welfare", fontsize=12)
    ax.set_title("Welfare-Toxicity Tradeoff", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
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
    for patch, color in zip(bp["boxes"], COLORS.values(), strict=False):
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
        means = [df[df[tax_param] == tv][col].mean() for tv in tax_vals]
        cis = [ci95(df[df[tax_param] == tv][col]) for tv in tax_vals]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width=width, yerr=cis, capsize=4,
               label=label, alpha=0.85, color=list(COLORS.values())[i])
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
    plot_metric_vs_param(df, tax_param, "honest_payoff",
                         "Honest Agent Payoff", "Honest Payoff vs Tax Rate",
                         "honest_payoff_vs_tax.png")
    count += 1

    # ============================================================
    # NEW PLOTS (v2)
    # ============================================================

    # 12. Tax burden by agent type (% drop from no-tax baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    no_tax = df[df[tax_param] == 0.0]
    tax_only = [tv for tv in tax_vals if tv > 0]
    x = np.arange(len(tax_only))
    width = 0.25
    for i, (label, col) in enumerate(agent_cols.items()):
        baseline_mean = no_tax[col].mean()
        pct_drops = []
        for tv in tax_only:
            treated_mean = df[df[tax_param] == tv][col].mean()
            pct = (treated_mean - baseline_mean) / baseline_mean * 100
            pct_drops.append(pct)
        offset = (i - 1) * width
        ax.bar(x + offset, pct_drops, width=width, label=label,
               alpha=0.85, color=list(COLORS.values())[i])
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in tax_only], fontsize=10)
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("% Change from Baseline (0% tax)", fontsize=12)
    ax.set_title("Tax Burden by Agent Type", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "tax_burden_by_type.png", dpi=150)
    plt.close()
    print("  Saved: tax_burden_by_type.png")
    count += 1

    # 13. Acceptance rate vs tax rate
    if "total_interactions" in df.columns and "accepted_interactions" in df.columns:
        df["acceptance_rate"] = df["accepted_interactions"] / df["total_interactions"].clip(lower=1)
        plot_metric_vs_param(df, tax_param, "acceptance_rate",
                             "Acceptance Rate", "Acceptance Rate vs Tax Rate",
                             "acceptance_rate_vs_tax.png")
        count += 1

    # 14. Interaction heatmap: welfare by tax × circuit breaker
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = df.pivot_table(values="welfare", index=cb_param, columns=tax_param, aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"CB={v}" for v in pivot.index], fontsize=10)
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_title("Welfare: Tax × Circuit Breaker Interaction", fontsize=13)
    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean Welfare")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "welfare_tax_cb_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: welfare_tax_cb_heatmap.png")
    count += 1

    # 15. Welfare curve with bootstrap CI ribbon
    fig, ax = plt.subplots(figsize=(9, 5))
    means_list, lo_list, hi_list = [], [], []
    for tv in tax_vals:
        data = df[df[tax_param] == tv]["welfare"].dropna().values
        lo, hi = bootstrap_ci(data)
        means_list.append(np.mean(data))
        lo_list.append(lo)
        hi_list.append(hi)

    ax.plot(tax_vals, means_list, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax.fill_between(tax_vals, lo_list, hi_list, alpha=0.2, color="#4C72B0")
    ax.set_xlabel("Transaction Tax Rate", fontsize=12)
    ax.set_ylabel("Welfare", fontsize=12)
    ax.set_title("Welfare vs Tax Rate (Bootstrap 95% CI)", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "welfare_vs_tax_bootstrap.png", dpi=150)
    plt.close()
    print("  Saved: welfare_vs_tax_bootstrap.png")
    count += 1

    print(f"\nTotal plots generated: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate publication-quality plots from governance comparison sweep results."""

import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats

BASE = pathlib.Path(
    "/Users/raelisavitt/swarm/"
    "runs/20260211-000149_kernel_market_governance_comparison"
)
CSV = BASE / "sweep_results.csv"
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
df.rename(columns={"governance.regime": "regime"}, inplace=True)

REGIME_ORDER = [
    "no_governance", "audits_only", "staking_only", "reputation_only",
    "circuit_breaker_only", "audits_staking", "full_governance",
]
REGIME_LABELS = [
    "No Governance", "Audits Only", "Staking Only", "Reputation Only",
    "Circuit Breaker Only", "Audits + Staking", "Full Governance",
]
REGIME_CATEGORY = {
    "no_governance": "baseline", "audits_only": "single",
    "staking_only": "single", "reputation_only": "single",
    "circuit_breaker_only": "single", "audits_staking": "combination",
    "full_governance": "combination",
}
CAT_COLORS = {"baseline": "#999999", "single": "#0072B2", "combination": "#D55E00"}
REGIME_COLORS = [CAT_COLORS[REGIME_CATEGORY[r]] for r in REGIME_ORDER]
AGENT_COLORS = {
    "honest_payoff": "#009E73",
    "opportunistic_payoff": "#E69F00",
    "adversarial_payoff": "#CC79A7",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150,
})
DPI = 150


def regime_stats(col):
    grouped = df.groupby("regime")[col]
    out = grouped.agg(["mean", "std", "count"]).reindex(REGIME_ORDER)
    out["sem"] = out["std"] / np.sqrt(out["count"])
    t_crit = stats.t.ppf(0.975, out["count"] - 1)
    out["ci95"] = t_crit * out["sem"]
    return out


def plot_welfare_by_regime():
    st = regime_stats("welfare")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGIME_ORDER))
    ax.bar(x, st["mean"], yerr=st["ci95"], capsize=4, color=REGIME_COLORS,
           edgecolor="white", linewidth=0.6, error_kw={"linewidth": 1.2})
    baseline = st.loc["no_governance", "mean"]
    ax.axhline(baseline, ls="--", color="#999999", linewidth=1, label="No-governance baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Total Welfare")
    ax.set_title("Mean Welfare by Governance Regime (95% CI)")
    patches = [
        mpatches.Patch(color=CAT_COLORS["baseline"], label="Baseline"),
        mpatches.Patch(color=CAT_COLORS["single"], label="Single lever"),
        mpatches.Patch(color=CAT_COLORS["combination"], label="Combination"),
    ]
    ax.legend(handles=patches, loc="upper left", frameon=True)
    fig.tight_layout()
    path = OUT / "welfare_by_regime.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_toxicity_by_regime():
    st = regime_stats("toxicity_rate")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGIME_ORDER))
    ax.bar(x, st["mean"], yerr=st["ci95"], capsize=4, color=REGIME_COLORS,
           edgecolor="white", linewidth=0.6, error_kw={"linewidth": 1.2})
    baseline = st.loc["no_governance", "mean"]
    ax.axhline(baseline, ls="--", color="#999999", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Toxicity Rate")
    ax.set_title("Mean Toxicity Rate by Governance Regime (95% CI)")
    patches = [
        mpatches.Patch(color=CAT_COLORS["baseline"], label="Baseline"),
        mpatches.Patch(color=CAT_COLORS["single"], label="Single lever"),
        mpatches.Patch(color=CAT_COLORS["combination"], label="Combination"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True)
    fig.tight_layout()
    path = OUT / "toxicity_by_regime.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_quality_gap_by_regime():
    st = regime_stats("quality_gap")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGIME_ORDER))
    ax.bar(x, st["mean"], yerr=st["ci95"], capsize=4, color=REGIME_COLORS,
           edgecolor="white", linewidth=0.6, error_kw={"linewidth": 1.2})
    ax.axhline(0, ls="--", color="black", linewidth=1, label="Adverse selection threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Quality Gap (E[p|accepted] - E[p|rejected])")
    ax.set_title("Quality Gap by Governance Regime (95% CI)")
    patches = [
        mpatches.Patch(color=CAT_COLORS["baseline"], label="Baseline"),
        mpatches.Patch(color=CAT_COLORS["single"], label="Single lever"),
        mpatches.Patch(color=CAT_COLORS["combination"], label="Combination"),
        plt.Line2D([0], [0], ls="--", color="black", label="Adverse selection threshold"),
    ]
    ax.legend(handles=patches, loc="best", frameon=True)
    fig.tight_layout()
    path = OUT / "quality_gap_by_regime.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_agent_payoff_by_regime():
    agent_cols = ["honest_payoff", "opportunistic_payoff", "adversarial_payoff"]
    agent_labels = ["Honest", "Opportunistic", "Adversarial"]
    width = 0.25
    x = np.arange(len(REGIME_ORDER))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label) in enumerate(zip(agent_cols, agent_labels, strict=True)):
        st = regime_stats(col)
        offset = (i - 1) * width
        ax.bar(x + offset, st["mean"], width, yerr=st["ci95"], capsize=3,
               color=AGENT_COLORS[col], edgecolor="white", linewidth=0.4,
               label=label, error_kw={"linewidth": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Mean Agent Payoff")
    ax.set_title("Agent Payoffs by Type and Governance Regime (95% CI)")
    ax.legend(loc="upper left", frameon=True)
    ax.axhline(0, ls="-", color="black", linewidth=0.5)
    fig.tight_layout()
    path = OUT / "agent_payoff_by_regime.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_welfare_toxicity_tradeoff():
    REGIME_SCATTER_COLORS = {
        "no_governance": "#999999", "audits_only": "#0072B2",
        "staking_only": "#56B4E9", "reputation_only": "#009E73",
        "circuit_breaker_only": "#F0E442", "audits_staking": "#E69F00",
        "full_governance": "#D55E00",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        ax.scatter(sub["welfare"], sub["toxicity_rate"],
                   c=REGIME_SCATTER_COLORS[regime], alpha=0.45, s=30, edgecolors="none")
    means = df.groupby("regime")[["welfare", "toxicity_rate"]].mean().reindex(REGIME_ORDER)
    for i, regime in enumerate(REGIME_ORDER):
        ax.scatter(means.loc[regime, "welfare"], means.loc[regime, "toxicity_rate"],
                   c=REGIME_SCATTER_COLORS[regime], s=160, marker="D",
                   edgecolors="black", linewidths=0.8, zorder=5, label=REGIME_LABELS[i])
        ax.annotate(REGIME_LABELS[i],
                    (means.loc[regime, "welfare"], means.loc[regime, "toxicity_rate"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8.5)
    ax.set_xlabel("Welfare")
    ax.set_ylabel("Toxicity Rate")
    ax.set_title("Welfare-Toxicity Trade-off by Governance Regime")
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=1)
    fig.tight_layout()
    path = OUT / "welfare_toxicity_tradeoff.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_adversarial_payoff():
    st = regime_stats("adversarial_payoff")
    sig_negative = []
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]["adversarial_payoff"]
        t_stat, p_val = stats.ttest_1samp(sub, 0)
        if t_stat < 0 and p_val / 2 < 0.05:
            sig_negative.append(regime)
    bar_colors = ["#D55E00" if r in sig_negative else "#56B4E9" for r in REGIME_ORDER]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGIME_ORDER))
    ax.bar(x, st["mean"], yerr=st["ci95"], capsize=4, color=bar_colors,
           edgecolor="white", linewidth=0.6, error_kw={"linewidth": 1.2})
    ax.axhline(0, ls="-", color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Adversarial Agent Payoff")
    ax.set_title("Adversarial Payoff by Governance Regime (95% CI)")
    patches = [
        mpatches.Patch(color="#D55E00", label="Significantly < 0 (p < 0.05)"),
        mpatches.Patch(color="#56B4E9", label="Not significantly < 0"),
    ]
    ax.legend(handles=patches, loc="best", frameon=True)
    fig.tight_layout()
    path = OUT / "adversarial_payoff_by_regime.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_regime_heatmap():
    metrics = ["welfare", "toxicity_rate", "quality_gap", "adversarial_payoff"]
    metric_labels = ["Welfare", "Toxicity Rate", "Quality Gap", "Adversarial Payoff"]
    means = df.groupby("regime")[metrics].mean().reindex(REGIME_ORDER)
    z = means.copy()
    for col in metrics:
        z[col] = (means[col] - means[col].mean()) / means[col].std()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_LABELS)
    for i in range(len(REGIME_ORDER)):
        for j in range(len(metrics)):
            val = z.values[i, j]
            color = "white" if abs(val) > 1.2 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Z-score (higher = greener)")
    ax.set_title("Governance Regime Performance Heatmap (Z-scored Metrics)")
    fig.tight_layout()
    path = OUT / "regime_heatmap.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_welfare_by_regime()
    plot_toxicity_by_regime()
    plot_quality_gap_by_regime()
    plot_agent_payoff_by_regime()
    plot_welfare_toxicity_tradeoff()
    plot_adversarial_payoff()
    plot_regime_heatmap()
    print()
    print("All plots generated successfully.")

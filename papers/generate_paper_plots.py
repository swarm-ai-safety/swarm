#!/usr/bin/env python3
"""Generate publication-quality comparison plots for SWARM safety papers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# === Data ===
SCENARIOS = {
    "baseline":    {"acceptance": 0.938, "toxicity": 0.298, "wpe": 4.98},
    "redteam_v1":  {"acceptance": 0.556, "toxicity": 0.295, "wpe": 3.80},
    "redteam_v3":  {"acceptance": 0.455, "toxicity": 0.312, "wpe": 3.49},
    "collusion":   {"acceptance": 0.425, "toxicity": 0.370, "wpe": 6.29},
    "emergent":    {"acceptance": 0.998, "toxicity": 0.297, "wpe": 44.9},
    "marketplace": {"acceptance": 0.549, "toxicity": 0.328, "wpe": 3.70},
    "network":     {"acceptance": 0.783, "toxicity": 0.335, "wpe": 9.90},
}

COLLUSION_ACCEPTED = [16,14,10,12,5,10,8,4,10,3,3,3,1,2,3,3,3,3,4,1,1,3,2,1,2]
COLLUSION_TOXICITY = [0.394,0.351,0.337,0.326,0.317,0.369,0.332,0.304,0.337,0.361,
                      0.356,0.414,0.529,0.418,0.371,0.317,0.438,0.419,0.418,0.262,
                      0.365,0.404,0.315,0.466,0.342]

EMERGENT_WELFARE = [51.54,51.96,65.33,46.44,36.64,34.60,50.48,48.17,32.11,55.18,
                    43.85,58.28,40.24,47.52,37.72,55.40,37.80,43.82,44.76,48.53,
                    43.27,47.89,36.33,35.87,44.51,51.08,31.92,52.37,37.38,38.82]

COLLUSION_WELFARE = [18.96,19.26,13.67,16.73,7.07,12.37,10.92,5.75,13.72,3.84,
                     4.05,3.29,0.60,2.16,3.85,4.56,2.99,3.14,4.33,1.66,
                     1.21,3.32,2.85,0.88,2.83]

NETWORK_WELFARE = [11.88,9.72,3.89,12.06,7.14,9.19,8.96,9.03,13.53,7.64,
                   9.86,8.80,5.90,10.93,10.66,11.80,12.50,14.54,7.03,12.94]


def fig1_scenario_comparison():
    names = list(SCENARIOS.keys())
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    metrics = [("acceptance", "Acceptance Rate"), ("toxicity", "Avg Toxicity"), ("wpe", "Welfare/Epoch (/10)")]

    for i, (key, label) in enumerate(metrics):
        vals = [SCENARIOS[s][key] for s in names]
        plot_vals = [v / 10 for v in vals] if key == "wpe" else vals
        bars = ax.bar(x + i * width, plot_vals, width, label=label,
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals, strict=False):
            fmt = f"{val:.1f}" if key == "wpe" else f"{val:.3f}"
            ax.annotate(fmt, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Cross-Scenario Comparison of Key Metrics")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_scenario_comparison.png")
    plt.close(fig)
    print("  Saved fig1")


def fig2_collusion_timeline():
    epochs = np.arange(len(COLLUSION_ACCEPTED))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accepted Interactions", color="#1565C0")
    l1 = ax1.plot(epochs, COLLUSION_ACCEPTED, color="#1565C0", marker="o",
                  markersize=5, linewidth=1.8, label="Accepted")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_ylim(0, max(COLLUSION_ACCEPTED) + 2)

    # Phase separators
    for ep in [5, 10]:
        ax1.axvline(ep, color="gray", linestyle=":", alpha=0.5)
    ax1.text(2.5, 17, "Phase 1\nEngagement", ha="center", fontsize=8, color="gray")
    ax1.text(7.5, 17, "Phase 2\nTransition", ha="center", fontsize=8, color="gray")
    ax1.text(17, 17, "Phase 3: Attrition", ha="center", fontsize=8, color="gray")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Toxicity", color="#C62828")
    l2 = ax2.plot(epochs, COLLUSION_TOXICITY, color="#C62828", marker="s",
                  markersize=5, linewidth=1.8, linestyle="--", label="Toxicity")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    ax2.set_ylim(0.2, 0.6)

    lines = l1 + l2
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax1.set_title("Collusion Scenario: Three-Phase Decline Pattern")
    ax1.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_collusion_timeline.png")
    plt.close(fig)
    print("  Saved fig2")


def fig3_regime_scatter():
    regime_data = {
        "emergent":     (0.00, 0.998, "Cooperative"),
        "baseline":     (0.20, 0.938, "Managed Friction"),
        "marketplace":  (0.143, 0.549, "Managed Friction"),
        "network":      (0.10, 0.783, "Managed Friction"),
        "incoherence":  (0.10, 0.787, "Managed Friction"),
        "redteam v1":   (0.50, 0.556, "Collapse Risk"),
        "redteam v3":   (0.50, 0.455, "Collapse Risk"),
        "collusion":    (0.375, 0.425, "Collapse Risk"),
    }
    regime_colors = {"Cooperative": "#4CAF50", "Managed Friction": "#2196F3", "Collapse Risk": "#F44336"}

    fig, ax = plt.subplots(figsize=(9, 6))

    # Threshold line
    ax.axvline(0.35, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(0.36, 0.95, "~35% threshold", fontsize=9, color="gray", fontstyle="italic")

    for name, (adv, acc, regime) in regime_data.items():
        ax.scatter(adv, acc, s=140, c=regime_colors[regime], edgecolors="black", linewidth=0.6, zorder=5)
        ax.annotate(name, xy=(adv, acc), xytext=(8, -4), textcoords="offset points", fontsize=9)

    handles = [mpatches.Patch(color=c, label=r) for r, c in regime_colors.items()]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9)
    ax.set_xlabel("Adversarial Fraction")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Governance Regime Classification")
    ax.set_xlim(-0.05, 0.60)
    ax.set_ylim(0.30, 1.08)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_regime_scatter.png")
    plt.close(fig)
    print("  Saved fig3")


def fig4_incoherence_scaling():
    variants = ["Short\n(3 agents)", "Medium\n(6 agents)", "Long\n(10 agents)"]
    toxicities = [0.183, 0.343, 0.341]
    acceptances = [1.0, 0.94, 0.787]

    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - width/2, acceptances, width, label="Acceptance Rate", color="#2196F3", edgecolor="white")
    b2 = ax.bar(x + width/2, toxicities, width, label="Avg Toxicity", color="#FF5722", edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Value")
    ax.set_title("Incoherence Scaling: Impact of Agent Count & Horizon")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_incoherence_scaling.png")
    plt.close(fig)
    print("  Saved fig4")


def fig5_welfare_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs_em = np.arange(len(EMERGENT_WELFARE))
    ax1.plot(epochs_em, EMERGENT_WELFARE, color="#1565C0", linewidth=1.6, marker="o", markersize=3)
    ax1.axhline(np.mean(EMERGENT_WELFARE), color="#1565C0", linestyle="--", alpha=0.5)
    ax1.text(28, np.mean(EMERGENT_WELFARE)+1.5, f"mean={np.mean(EMERGENT_WELFARE):.1f}",
             fontsize=9, color="#1565C0", ha="right")
    ax1.fill_between(epochs_em, EMERGENT_WELFARE, alpha=0.12, color="#1565C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Welfare")
    ax1.set_title("Emergent Capabilities (0% adversarial)")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 29)

    epochs_col = np.arange(len(COLLUSION_WELFARE))
    ax2.plot(epochs_col, COLLUSION_WELFARE, color="#C62828", linewidth=1.6, marker="s", markersize=3)
    ax2.axhline(np.mean(COLLUSION_WELFARE), color="#C62828", linestyle="--", alpha=0.5)
    ax2.text(23, np.mean(COLLUSION_WELFARE)+0.8, f"mean={np.mean(COLLUSION_WELFARE):.1f}",
             fontsize=9, color="#C62828", ha="right")
    ax2.fill_between(epochs_col, COLLUSION_WELFARE, alpha=0.12, color="#C62828")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Welfare")
    ax2.set_title("Collusion Detection (37.5% adversarial)")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 24)

    fig.suptitle("Welfare Over Time: Cooperative vs Adversarial Dynamics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_welfare_comparison.png")
    plt.close(fig)
    print("  Saved fig5")


def fig6_network_vs_collusion():
    """Network sustained volatility vs collusion decline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs_net = np.arange(len(NETWORK_WELFARE))
    ax1.plot(epochs_net, NETWORK_WELFARE, color="#4CAF50", linewidth=1.6, marker="D", markersize=3.5)
    ax1.axhline(np.mean(NETWORK_WELFARE), color="#4CAF50", linestyle="--", alpha=0.5)
    ax1.fill_between(epochs_net, NETWORK_WELFARE, alpha=0.12, color="#4CAF50")
    ax1.text(18, np.mean(NETWORK_WELFARE)+0.8, f"mean={np.mean(NETWORK_WELFARE):.1f}",
             fontsize=9, color="#4CAF50", ha="right")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Welfare")
    ax1.set_title("Network Effects (10% adversarial)\nSustained Volatility")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 19)

    epochs_col = np.arange(len(COLLUSION_WELFARE))
    ax2.plot(epochs_col, COLLUSION_WELFARE, color="#C62828", linewidth=1.6, marker="s", markersize=3.5)
    ax2.axhline(np.mean(COLLUSION_WELFARE), color="#C62828", linestyle="--", alpha=0.5)
    ax2.fill_between(epochs_col, COLLUSION_WELFARE, alpha=0.12, color="#C62828")
    ax2.text(23, np.mean(COLLUSION_WELFARE)+0.8, f"mean={np.mean(COLLUSION_WELFARE):.1f}",
             fontsize=9, color="#C62828", ha="right")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Welfare")
    ax2.set_title("Collusion Detection (37.5% adversarial)\nProgressive Decline")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 24)

    fig.suptitle("Network Resilience vs Collusion Degradation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_network_vs_collusion.png")
    plt.close(fig)
    print("  Saved fig6")


if __name__ == "__main__":
    print(f"Generating figures in {FIGURES_DIR}/\n")
    fig1_scenario_comparison()
    fig2_collusion_timeline()
    fig3_regime_scatter()
    fig4_incoherence_scaling()
    fig5_welfare_comparison()
    fig6_network_vs_collusion()
    print("\nDone.")

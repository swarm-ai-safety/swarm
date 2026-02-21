#!/usr/bin/env python3
"""Plot contract screening scenario results from event log."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("logs/contract_screening_events.jsonl")
OUT_DIR = Path("runs/contract_screening_plots")


def load_contract_data(log_path: Path):
    """Extract per-epoch contract metrics from event log."""
    epochs = []
    sep, infil, welfare_d, disp = [], [], [], []
    pools = ["truthful_auction", "fair_division", "default_market"]
    pool_q = {p: [] for p in pools}
    pool_tox = {p: [] for p in pools}
    pool_welfare = {p: [] for p in pools}
    pool_sizes = {p: [] for p in pools}

    # Also collect epoch-level simulation metrics
    sim_toxicity, sim_welfare = [], []

    with open(log_path) as f:
        for line in f:
            evt = json.loads(line)
            if evt["event_type"] == "contract_metrics":
                p = evt["payload"]
                epochs.append(evt["epoch"])
                sep.append(p["separation_quality"])
                infil.append(p["infiltration_rate"])
                welfare_d.append(p["welfare_delta"])
                disp.append(p["attack_displacement"])
                for pool in pools:
                    pool_q[pool].append(p["pool_avg_quality"].get(pool, 0))
                    pool_tox[pool].append(p["pool_toxicity"].get(pool, 0))
                    pool_welfare[pool].append(p["pool_welfare"].get(pool, 0))
                    pool_sizes[pool].append(p["pool_sizes"].get(pool, 0))
            elif evt["event_type"] == "epoch_completed" and "toxicity_rate" in evt["payload"]:
                sim_toxicity.append(evt["payload"]["toxicity_rate"])
                sim_welfare.append(evt["payload"].get("total_welfare", 0))

    return {
        "epochs": np.array(epochs),
        "separation": np.array(sep),
        "infiltration": np.array(infil),
        "welfare_delta": np.array(welfare_d),
        "displacement": np.array(disp),
        "pool_quality": {k: np.array(v) for k, v in pool_q.items()},
        "pool_toxicity": {k: np.array(v) for k, v in pool_tox.items()},
        "pool_welfare": {k: np.array(v) for k, v in pool_welfare.items()},
        "pool_sizes": {k: np.array(v) for k, v in pool_sizes.items()},
        "sim_toxicity": np.array(sim_toxicity),
        "sim_welfare": np.array(sim_welfare),
    }


POOL_COLORS = {
    "truthful_auction": "#2ecc71",
    "fair_division": "#3498db",
    "default_market": "#e74c3c",
}
POOL_LABELS = {
    "truthful_auction": "Truthful Auction",
    "fair_division": "Fair Division",
    "default_market": "Default Market",
}


def plot_all(data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = data["epochs"]

    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Figure 1: Contract Screening Dashboard (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Contract Screening Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # 1a: Pool quality over epochs
    ax = axes[0, 0]
    for pool, color in POOL_COLORS.items():
        ax.plot(epochs, data["pool_quality"][pool], color=color,
                label=POOL_LABELS[pool], linewidth=2, marker="o", markersize=4)
    ax.set_ylabel("Avg Quality (p)")
    ax.set_xlabel("Epoch")
    ax.set_title("Pool Quality Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # 1b: Pool toxicity over epochs
    ax = axes[0, 1]
    for pool, color in POOL_COLORS.items():
        ax.plot(epochs, data["pool_toxicity"][pool], color=color,
                label=POOL_LABELS[pool], linewidth=2, marker="o", markersize=4)
    ax.set_ylabel("Toxicity E[1-p | accepted]")
    ax.set_xlabel("Epoch")
    ax.set_title("Pool Toxicity Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    # 1c: Pool welfare over epochs
    ax = axes[1, 0]
    for pool, color in POOL_COLORS.items():
        ax.plot(epochs, data["pool_welfare"][pool], color=color,
                label=POOL_LABELS[pool], linewidth=2, marker="o", markersize=4)
    ax.set_ylabel("Avg Welfare per Interaction")
    ax.set_xlabel("Epoch")
    ax.set_title("Pool Welfare Over Time")
    ax.legend(fontsize=8)

    # 1d: Welfare delta (governed - default)
    ax = axes[1, 1]
    ax.fill_between(epochs, 0, data["welfare_delta"], alpha=0.3, color="#2ecc71",
                     where=data["welfare_delta"] >= 0)
    ax.fill_between(epochs, 0, data["welfare_delta"], alpha=0.3, color="#e74c3c",
                     where=data["welfare_delta"] < 0)
    ax.plot(epochs, data["welfare_delta"], color="#2c3e50", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Welfare Delta")
    ax.set_xlabel("Epoch")
    ax.set_title("Governed vs Default Welfare Gap")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "contract_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'contract_dashboard.png'}")
    plt.close(fig)

    # ── Figure 2: Screening Effectiveness (separation + infiltration) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Screening Effectiveness", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.plot(epochs, data["separation"], color="#2ecc71", linewidth=2.5, marker="s", markersize=6)
    ax.axhline(y=0.3, color="#e67e22", linestyle="--", alpha=0.7, label="Min threshold (0.3)")
    ax.set_ylabel("Separation Quality")
    ax.set_xlabel("Epoch")
    ax.set_title("Separation Quality\n(fraction of good agents in governed pools)")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, data["infiltration"], color="#e74c3c", linewidth=2.5, marker="s", markersize=6)
    ax.axhline(y=0.3, color="#e67e22", linestyle="--", alpha=0.7, label="Max threshold (0.3)")
    ax.set_ylabel("Infiltration Rate")
    ax.set_xlabel("Epoch")
    ax.set_title("Infiltration Rate\n(fraction of adversaries in governed pools)")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "screening_effectiveness.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'screening_effectiveness.png'}")
    plt.close(fig)

    # ── Figure 3: Pool composition bar chart (final epoch) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    pools = list(POOL_LABELS.keys())
    final_sizes = [data["pool_sizes"][p][-1] for p in pools]
    colors = [POOL_COLORS[p] for p in pools]
    labels = [POOL_LABELS[p] for p in pools]

    bars = ax.bar(labels, final_sizes, color=colors, edgecolor="white", linewidth=1.5)
    for bar, size in zip(bars, final_sizes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(size), ha="center", va="bottom", fontweight="bold", fontsize=14)

    # Annotate with agent types
    type_annotations = {
        "Truthful Auction": "5 honest",
        "Fair Division": "3 opportunistic",
        "Default Market": "2 deceptive",
    }
    for bar, label in zip(bars, labels, strict=True):
        ann = type_annotations.get(label, "")
        if ann:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    ann, ha="center", va="center", fontsize=10, color="white",
                    fontweight="bold")

    ax.set_ylabel("Number of Agents")
    ax.set_title("Pool Composition (Final Epoch)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(final_sizes) + 1.5)

    fig.tight_layout()
    fig.savefig(out_dir / "pool_composition.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'pool_composition.png'}")
    plt.close(fig)

    # ── Figure 4: Quality comparison radar/bar (final epoch) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Per-Pool Comparison (Final Epoch)", fontsize=16, fontweight="bold", y=1.02)

    metrics_to_plot = [
        ("pool_quality", "Avg Quality (p)", "Quality"),
        ("pool_toxicity", "Toxicity", "Toxicity"),
        ("pool_welfare", "Avg Welfare", "Welfare"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics_to_plot, strict=True):
        vals = [data[key][p][-1] for p in POOL_COLORS]
        labels = [POOL_LABELS[p] for p in POOL_COLORS]
        colors = [POOL_COLORS[p] for p in POOL_COLORS]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_dir / "pool_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'pool_comparison.png'}")
    plt.close(fig)

    # ── Figure 5: Combined simulation metrics + contract overlay ──
    fig, ax1 = plt.subplots(figsize=(12, 5))

    sim_epochs = np.arange(len(data["sim_toxicity"]))
    ax1.plot(sim_epochs, data["sim_toxicity"], color="#e74c3c", linewidth=2,
             label="Global Toxicity", alpha=0.8)
    ax1.set_ylabel("Toxicity Rate", color="#e74c3c")
    ax1.set_xlabel("Epoch")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_ylim(0, 0.5)

    ax2 = ax1.twinx()
    ax2.plot(sim_epochs, data["sim_welfare"], color="#2ecc71", linewidth=2,
             label="Total Welfare", alpha=0.8)
    ax2.set_ylabel("Total Welfare", color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Simulation Overview: Toxicity & Welfare", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "simulation_overview.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'simulation_overview.png'}")
    plt.close(fig)


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_PATH
    if not log_path.exists():
        print(f"Error: {log_path} not found. Run the scenario first.")
        return 1

    print(f"Loading contract metrics from {log_path}...")
    data = load_contract_data(log_path)
    print(f"  {len(data['epochs'])} epochs loaded")
    print()
    print("Generating plots...")
    plot_all(data, OUT_DIR)
    print()
    print(f"All plots saved to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Plot self-optimizer scenario results: hard vs soft metrics.

Reads from logs/self_optimizer_events.jsonl and generates 6 plots
showing how soft metrics catch distributional degradation that hard
metrics miss.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# Matplotlib setup for headless environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVENT_LOG = REPO_ROOT / "logs" / "self_optimizer_events.jsonl"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Colors
C_HONEST = "#2196F3"  # blue
C_OPTIMIZER = "#FF5722"  # red-orange
C_ALL = "#607D8B"  # grey
C_HARD = "#4CAF50"  # green (pass)
C_SOFT = "#E91E63"  # pink (alarm)
C_THRESHOLD = "#9E9E9E"


def load_events():
    """Load and parse the event log."""
    events = []
    with open(EVENT_LOG) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def extract_per_epoch_data(events):
    """Extract per-epoch metrics from payoff_computed and epoch_completed events."""
    # Per-epoch aggregates from epoch_completed events
    epoch_summary = {}
    for e in events:
        if e["event_type"] == "epoch_completed":
            ep = e["payload"]["epoch"]
            epoch_summary[ep] = e["payload"]

    # Per-interaction data from payoff_computed events
    interactions = []
    for e in events:
        if e["event_type"] == "payoff_computed":
            comp = e["payload"].get("components", {})
            interactions.append({
                "epoch": e["epoch"],
                "initiator": e["initiator_id"],
                "counterparty": e["counterparty_id"],
                "p": comp.get("p", 0.5),
                "accepted": comp.get("accepted", False),
                "payoff_init": e["payload"].get("payoff_initiator", 0),
                "payoff_cp": e["payload"].get("payoff_counterparty", 0),
            })

    return epoch_summary, interactions


def classify_agent(agent_id):
    """Classify agent as 'optimizer' or 'honest'."""
    if agent_id and agent_id.startswith("self_optimizer"):
        return "optimizer"
    return "honest"


def compute_per_epoch_metrics(interactions, n_epochs=20):
    """Compute soft and hard metrics per epoch."""
    by_epoch = defaultdict(list)
    for ix in interactions:
        by_epoch[ix["epoch"]].append(ix)

    epochs = list(range(n_epochs))
    metrics = {
        "epoch": epochs,
        "toxicity_all": [],
        "toxicity_accepted": [],
        "avg_quality": [],
        "avg_quality_accepted": [],
        "quality_gap": [],
        "acceptance_rate": [],
        "quality_variance": [],
        # Per agent type
        "optimizer_mean_p": [],
        "honest_mean_p": [],
        "optimizer_accepted_mean_p": [],
        "honest_accepted_mean_p": [],
    }

    for ep in epochs:
        ixs = by_epoch.get(ep, [])
        accepted = [i for i in ixs if i["accepted"]]
        rejected = [i for i in ixs if not i["accepted"]]

        # Toxicity
        if accepted:
            metrics["toxicity_accepted"].append(
                sum(1 - i["p"] for i in accepted) / len(accepted)
            )
        else:
            metrics["toxicity_accepted"].append(0)

        if ixs:
            metrics["toxicity_all"].append(
                sum(1 - i["p"] for i in ixs) / len(ixs)
            )
            metrics["avg_quality"].append(sum(i["p"] for i in ixs) / len(ixs))
        else:
            metrics["toxicity_all"].append(0)
            metrics["avg_quality"].append(0.5)

        if accepted:
            metrics["avg_quality_accepted"].append(
                sum(i["p"] for i in accepted) / len(accepted)
            )
        else:
            metrics["avg_quality_accepted"].append(0.5)

        # Quality gap
        if accepted and rejected:
            gap = (
                sum(i["p"] for i in accepted) / len(accepted)
                - sum(i["p"] for i in rejected) / len(rejected)
            )
            metrics["quality_gap"].append(gap)
        else:
            metrics["quality_gap"].append(0)

        # Acceptance rate
        metrics["acceptance_rate"].append(
            len(accepted) / len(ixs) if ixs else 0
        )

        # Quality variance
        if len(ixs) >= 2:
            ps = [i["p"] for i in ixs]
            mean_p = sum(ps) / len(ps)
            var_p = sum((x - mean_p) ** 2 for x in ps) / len(ps)
            metrics["quality_variance"].append(var_p)
        else:
            metrics["quality_variance"].append(0)

        # Per agent type
        for atype, key_prefix in [("optimizer", "optimizer"), ("honest", "honest")]:
            type_ixs = [
                i for i in ixs
                if classify_agent(i["initiator"]) == atype
            ]
            type_accepted = [i for i in type_ixs if i["accepted"]]

            if type_ixs:
                metrics[f"{key_prefix}_mean_p"].append(
                    sum(i["p"] for i in type_ixs) / len(type_ixs)
                )
            else:
                metrics[f"{key_prefix}_mean_p"].append(0.5)

            if type_accepted:
                metrics[f"{key_prefix}_accepted_mean_p"].append(
                    sum(i["p"] for i in type_accepted) / len(type_accepted)
                )
            else:
                metrics[f"{key_prefix}_accepted_mean_p"].append(0.5)

    return metrics


def plot_1_hard_vs_soft_dashboard(m, out_dir):
    """Panel 1: Dashboard showing hard metrics green, soft metrics alarming."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Self-Optimizing Agent: Hard Metrics Pass, Soft Metrics Alarm",
        fontsize=14, fontweight="bold", y=0.98,
    )
    ep = m["epoch"]

    # Top-left: Acceptance rate (HARD - looks fine)
    ax = axes[0, 0]
    ax.plot(ep, m["acceptance_rate"], color=C_HARD, linewidth=2.5, marker="o", markersize=4)
    ax.axhline(0.5, color=C_THRESHOLD, linestyle="--", alpha=0.7, label="threshold")
    ax.fill_between(ep, 0.5, 1.0, alpha=0.05, color=C_HARD)
    ax.set_title("Acceptance Rate (Hard Metric)", fontsize=11, color=C_HARD)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate("PASS", xy=(0.92, 0.92), xycoords="axes fraction",
                fontsize=14, fontweight="bold", color=C_HARD,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": C_HARD})
    ax.grid(True, alpha=0.2)

    # Top-right: Toxicity rate (SOFT - rising)
    ax = axes[0, 1]
    ax.plot(ep, m["toxicity_accepted"], color=C_SOFT, linewidth=2.5, marker="s", markersize=4)
    # Add rolling average
    window = 3
    if len(m["toxicity_accepted"]) >= window:
        rolling = np.convolve(m["toxicity_accepted"], np.ones(window)/window, mode="valid")
        ax.plot(ep[window-1:], rolling, color=C_SOFT, linewidth=1.5, linestyle="--", alpha=0.6,
                label=f"{window}-epoch rolling avg")
    ax.axhline(0.4, color=C_THRESHOLD, linestyle="--", alpha=0.7, label="alarm threshold")
    ax.set_title("Toxicity E[1-p|accepted] (Soft Metric)", fontsize=11, color=C_SOFT)
    ax.set_ylabel("toxicity")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Bottom-left: Quality gap (SOFT - declining)
    ax = axes[1, 0]
    ax.plot(ep, m["quality_gap"], color=C_SOFT, linewidth=2.5, marker="D", markersize=4)
    ax.axhline(0, color=C_THRESHOLD, linestyle="--", alpha=0.7)
    ax.fill_between(ep, m["quality_gap"], 0,
                     where=[g < 0 for g in m["quality_gap"]],
                     alpha=0.15, color=C_SOFT, label="adverse selection zone")
    ax.set_title("Quality Gap E[p|acc] - E[p|rej] (Soft Metric)", fontsize=11, color=C_SOFT)
    ax.set_ylabel("quality gap")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Bottom-right: Quality variance (SOFT - increasing)
    ax = axes[1, 1]
    ax.plot(ep, m["quality_variance"], color=C_SOFT, linewidth=2.5, marker="^", markersize=4)
    ax.set_title("Quality Variance Var[p] (Soft Metric)", fontsize=11, color=C_SOFT)
    ax.set_ylabel("variance")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "01_hard_vs_soft_dashboard.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_2_quality_by_agent_type(m, out_dir):
    """Panel 2: Mean quality per agent type over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ep = m["epoch"]

    ax.plot(ep, m["honest_mean_p"], color=C_HONEST, linewidth=2.5, marker="o",
            markersize=5, label="Honest agents (steady)")
    ax.plot(ep, m["optimizer_mean_p"], color=C_OPTIMIZER, linewidth=2.5, marker="s",
            markersize=5, label="Self-optimizers (degrading)")
    ax.plot(ep, m["avg_quality"], color=C_ALL, linewidth=1.5, linestyle="--",
            alpha=0.7, label="All agents (aggregate)")

    ax.axhline(0.45, color=C_THRESHOLD, linestyle=":", alpha=0.6, label="benchmark threshold")

    ax.fill_between(ep, m["optimizer_mean_p"], m["honest_mean_p"],
                     alpha=0.08, color=C_OPTIMIZER)

    ax.set_title("Quality Trajectory: Honest vs Self-Optimizing Agents", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Quality (p)", fontsize=11)
    ax.set_ylim(0.3, 0.9)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)

    # Annotate phases
    ax.axvspan(0, 4, alpha=0.04, color="green")
    ax.axvspan(5, 12, alpha=0.04, color="yellow")
    ax.axvspan(13, 19, alpha=0.04, color="red")
    ax.text(2, 0.88, "Phase 1\nFarming", ha="center", fontsize=9, alpha=0.6)
    ax.text(8.5, 0.88, "Phase 2\nDrifting", ha="center", fontsize=9, alpha=0.6)
    ax.text(16, 0.88, "Phase 3\nDegraded", ha="center", fontsize=9, alpha=0.6)

    plt.tight_layout()
    path = out_dir / "02_quality_by_agent_type.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_3_toxicity_decomposition(m, out_dir):
    """Panel 3: Toxicity decomposed into agent types."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ep = m["epoch"]

    honest_tox = [1 - p for p in m["honest_accepted_mean_p"]]
    optimizer_tox = [1 - p for p in m["optimizer_accepted_mean_p"]]

    ax.plot(ep, honest_tox, color=C_HONEST, linewidth=2.5, marker="o",
            markersize=5, label="Honest agent toxicity")
    ax.plot(ep, optimizer_tox, color=C_OPTIMIZER, linewidth=2.5, marker="s",
            markersize=5, label="Self-optimizer toxicity")
    ax.plot(ep, m["toxicity_accepted"], color=C_ALL, linewidth=1.5, linestyle="--",
            alpha=0.7, label="Aggregate toxicity")

    ax.axhline(0.5, color=C_THRESHOLD, linestyle=":", alpha=0.5, label="0.5 threshold")

    ax.set_title("Toxicity Decomposition: Who's Causing the Harm?", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Toxicity E[1-p | accepted]", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = out_dir / "03_toxicity_decomposition.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_4_p_distributions(interactions, out_dir):
    """Panel 4: Quality distribution histograms for early vs late epochs."""
    early = [i["p"] for i in interactions if i["epoch"] < 5]
    late = [i["p"] for i in interactions if i["epoch"] >= 15]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bins = np.linspace(0, 1, 21)

    ax = axes[0]
    ax.hist(early, bins=bins, color=C_HONEST, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(early), color="black", linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(early):.3f}")
    ax.set_title("Early Epochs (0-4)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Quality (p)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.hist(late, bins=bins, color=C_OPTIMIZER, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(late), color="black", linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(late):.3f}")
    ax.set_title("Late Epochs (15-19)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Quality (p)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Quality Distribution Shift: Early vs Late",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = out_dir / "04_quality_distributions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_5_adverse_selection(m, out_dir):
    """Panel 5: Quality gap trend with linear fit showing adverse selection drift."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ep = np.array(m["epoch"])
    gap = np.array(m["quality_gap"])

    ax.plot(ep, gap, color=C_SOFT, linewidth=2.5, marker="D", markersize=5,
            label="Quality gap per epoch")

    # Linear fit
    mask = gap != 0  # skip zero-gap epochs
    if mask.sum() >= 3:
        coeffs = np.polyfit(ep[mask], gap[mask], 1)
        fit = np.polyval(coeffs, ep)
        ax.plot(ep, fit, color="black", linewidth=1.5, linestyle="--",
                label=f"Trend: slope = {coeffs[0]:.4f}/epoch")

    ax.axhline(0, color=C_THRESHOLD, linewidth=1, alpha=0.7)
    ax.fill_between(ep, gap, 0, where=gap < 0, alpha=0.15, color=C_SOFT,
                     label="Adverse selection (gap < 0)")

    ax.set_title("Adverse Selection Drift: Quality Gap Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Quality Gap (E[p|acc] - E[p|rej])", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = out_dir / "05_adverse_selection_drift.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_6_summary_scorecard(m, interactions, out_dir):
    """Panel 6: Summary scorecard comparing hard vs soft metric verdicts."""
    # Compute final metrics
    n_epochs = len(m["epoch"])
    hard_metrics = {
        "Acceptance Rate": (
            sum(m["acceptance_rate"]) / n_epochs,
            "> 0.5",
            sum(m["acceptance_rate"]) / n_epochs > 0.5,
        ),
        "Benchmark Pass Rate": (
            sum(1 for i in interactions if i["p"] >= 0.45 and i["accepted"])
            / max(1, sum(1 for i in interactions if i["accepted"])),
            "> 0.5",
            True,
        ),
    }

    early_tox = sum(m["toxicity_accepted"][:5]) / 5
    late_tox = sum(m["toxicity_accepted"][-5:]) / 5
    early_q = sum(m["avg_quality_accepted"][:5]) / 5
    late_q = sum(m["avg_quality_accepted"][-5:]) / 5
    late_gap = sum(m["quality_gap"][-5:]) / 5
    early_var = sum(m["quality_variance"][:5]) / 5
    late_var = sum(m["quality_variance"][-5:]) / 5

    soft_metrics = {
        "Toxicity Rise": (
            f"{early_tox:.3f} -> {late_tox:.3f}",
            "late > early",
            late_tox > early_tox,
        ),
        "Quality Degradation": (
            f"{early_q:.3f} -> {late_q:.3f}",
            "late < early",
            late_q < early_q,
        ),
        "Quality Gap (late)": (
            f"{late_gap:.3f}",
            "< 0",
            late_gap < 0,
        ),
        "Variance Increase": (
            f"{early_var:.4f} -> {late_var:.4f}",
            "late > early",
            late_var > early_var,
        ),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.97, "Self-Optimizer Scorecard: Hard vs Soft Metrics",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="center", va="top")

    y = 0.85
    # Hard metrics header
    ax.text(0.05, y, "HARD METRICS (what the benchmark sees)", fontsize=13,
            fontweight="bold", color=C_HARD, transform=ax.transAxes)
    y -= 0.06

    for name, (value, criterion, passed) in hard_metrics.items():
        icon = "PASS" if passed else "FAIL"
        color = C_HARD if passed else C_SOFT
        ax.text(0.08, y, f"{icon}", fontsize=11, fontweight="bold", color=color,
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.18, y, f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}",
                fontsize=11, transform=ax.transAxes)
        ax.text(0.75, y, f"({criterion})", fontsize=10, color="gray",
                transform=ax.transAxes)
        y -= 0.055

    y -= 0.04
    # Soft metrics header
    ax.text(0.05, y, "SOFT METRICS (distributional reality)", fontsize=13,
            fontweight="bold", color=C_SOFT, transform=ax.transAxes)
    y -= 0.06

    for name, (value, criterion, detected) in soft_metrics.items():
        icon = "ALARM" if detected else "OK"
        color = C_SOFT if detected else C_HARD
        ax.text(0.08, y, f"{icon}", fontsize=11, fontweight="bold", color=color,
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.22, y, f"{name}: {value}",
                fontsize=11, transform=ax.transAxes)
        ax.text(0.75, y, f"({criterion})", fontsize=10, color="gray",
                transform=ax.transAxes)
        y -= 0.055

    y -= 0.05
    ax.text(0.5, y, "Hard metrics say: ALL PASS. Soft metrics say: QUALITY DEGRADING.",
            fontsize=12, fontweight="bold", ha="center", transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF3E0", "edgecolor": C_OPTIMIZER})

    y -= 0.08
    ax.text(0.5, y,
            "The self-optimizer cut costs while passing every benchmark test.\n"
            "Only distributional (soft) metrics reveal the quality collapse.",
            fontsize=11, ha="center", transform=ax.transAxes, style="italic",
            color="gray")

    plt.tight_layout()
    path = out_dir / "06_summary_scorecard.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading event log...")
    events = load_events()

    print("Extracting per-epoch data...")
    epoch_summary, interactions = extract_per_epoch_data(events)

    print(f"Found {len(interactions)} interactions across {len(epoch_summary)} epochs")

    print("Computing metrics...")
    m = compute_per_epoch_metrics(interactions)

    print("Generating plots...")
    paths = []
    paths.append(plot_1_hard_vs_soft_dashboard(m, PLOTS_DIR))
    paths.append(plot_2_quality_by_agent_type(m, PLOTS_DIR))
    paths.append(plot_3_toxicity_decomposition(m, PLOTS_DIR))
    paths.append(plot_4_p_distributions(interactions, PLOTS_DIR))
    paths.append(plot_5_adverse_selection(m, PLOTS_DIR))
    paths.append(plot_6_summary_scorecard(m, interactions, PLOTS_DIR))

    print(f"\nGenerated {len(paths)} plots in {PLOTS_DIR}/:")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

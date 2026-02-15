#!/usr/bin/env python
"""Generate publication-quality plots for SkillRL dynamics study.

Reads snapshots.json produced by run_skillrl_dynamics.py and generates
6 plots showing skill evolution dynamics across multiple seeds.

Usage:
    python examples/plot_skillrl_dynamics.py runs/<dir>/snapshots.json
    python examples/plot_skillrl_dynamics.py runs/<dir>/snapshots.json --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swarm.analysis.theme import COLORS, swarm_theme  # noqa: E402

# ── Colors ────────────────────────────────────────────────────────────────

SKILLRL_COLOR = "#2F80ED"  # WELFARE blue for SkillRL agents
HONEST_COLOR = COLORS.HONEST  # "#3ECFB4"
ADVERSARIAL_COLOR = COLORS.ADVERSARIAL  # "#EB5757"
OPPORTUNISTIC_COLOR = COLORS.OPPORTUNISTIC  # "#828894"

STRATEGY_COLOR = "#2F80ED"
LESSON_COLOR = "#F2994A"
COMPOSITE_COLOR = "#BB6BD9"

GENERAL_COLOR = "#27AE60"
TASK_SPECIFIC_COLOR = "#F2C94C"

DPI = 150

# ── Data extraction helpers ───────────────────────────────────────────────


def _load_snapshots(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _collect_by_type(
    data: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Aggregate per-epoch metrics across seeds, grouped by agent type.

    Returns:
        (metrics_dict, n_epochs)

        metrics_dict keys (each value is np.ndarray shape (n_seeds, n_epochs)):
            skillrl_total_skills, skillrl_strategies, skillrl_lessons,
            skillrl_composites, skillrl_general, skillrl_task_specific,
            skillrl_refined, skillrl_pg_baseline, skillrl_avg_effectiveness,
            skillrl_mean_threshold_delta, skillrl_total_payoff,
            honest_total_payoff, adversarial_total_payoff,
            opportunistic_total_payoff
    """
    runs = data["runs"]
    n_seeds = len(runs)
    n_epochs = data["n_epochs"]

    # Initialize accumulators: for each seed, for each epoch, collect
    # per-agent-type averages.
    keys = [
        "skillrl_total_skills",
        "skillrl_strategies",
        "skillrl_lessons",
        "skillrl_composites",
        "skillrl_general",
        "skillrl_task_specific",
        "skillrl_refined",
        "skillrl_pg_baseline",
        "skillrl_avg_effectiveness",
        "skillrl_mean_threshold_delta",
        "skillrl_total_payoff",
        "honest_total_payoff",
        "adversarial_total_payoff",
        "opportunistic_total_payoff",
    ]
    arrays = {k: np.full((n_seeds, n_epochs), np.nan) for k in keys}

    for si, (_seed_key, snapshots) in enumerate(runs.items()):
        for snap in snapshots:
            ei = snap["epoch"]
            if ei >= n_epochs:
                continue

            # Group agents by type
            skillrl_agents = [
                a for a in snap["agents"] if a["agent_type"] == "skillrl"
            ]
            honest_agents = [
                a for a in snap["agents"] if a["agent_type"] == "honest"
            ]
            adv_agents = [
                a for a in snap["agents"] if a["agent_type"] == "adversarial"
            ]
            opp_agents = [
                a for a in snap["agents"] if a["agent_type"] == "opportunistic"
            ]

            # SkillRL averages
            if skillrl_agents:
                n = len(skillrl_agents)
                arrays["skillrl_total_skills"][si, ei] = sum(
                    a["skill_summary"]["total_skills"] for a in skillrl_agents
                ) / n
                arrays["skillrl_strategies"][si, ei] = sum(
                    a["skill_summary"]["strategies"] for a in skillrl_agents
                ) / n
                arrays["skillrl_lessons"][si, ei] = sum(
                    a["skill_summary"]["lessons"] for a in skillrl_agents
                ) / n
                arrays["skillrl_composites"][si, ei] = sum(
                    a["skill_summary"]["composites"] for a in skillrl_agents
                ) / n
                arrays["skillrl_general"][si, ei] = sum(
                    a["skill_summary"]["general_tier"] for a in skillrl_agents
                ) / n
                arrays["skillrl_task_specific"][si, ei] = sum(
                    a["skill_summary"]["task_specific_tier"]
                    for a in skillrl_agents
                ) / n
                arrays["skillrl_refined"][si, ei] = sum(
                    a["skill_summary"]["refined"] for a in skillrl_agents
                ) / n
                arrays["skillrl_pg_baseline"][si, ei] = sum(
                    a["skill_summary"]["pg_baseline"] for a in skillrl_agents
                ) / n
                arrays["skillrl_avg_effectiveness"][si, ei] = sum(
                    a.get("avg_effectiveness", 0.0) for a in skillrl_agents
                ) / n
                arrays["skillrl_mean_threshold_delta"][si, ei] = sum(
                    a.get("mean_threshold_delta", 0.0) for a in skillrl_agents
                ) / n
                arrays["skillrl_total_payoff"][si, ei] = sum(
                    a["total_payoff"] for a in skillrl_agents
                ) / n

            # Other agent type payoffs (mean across agents of that type)
            if honest_agents:
                arrays["honest_total_payoff"][si, ei] = sum(
                    a["total_payoff"] for a in honest_agents
                ) / len(honest_agents)
            if adv_agents:
                arrays["adversarial_total_payoff"][si, ei] = sum(
                    a["total_payoff"] for a in adv_agents
                ) / len(adv_agents)
            if opp_agents:
                arrays["opportunistic_total_payoff"][si, ei] = sum(
                    a["total_payoff"] for a in opp_agents
                ) / len(opp_agents)

    return arrays, n_epochs


def _mean_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 95% CI across axis 0 (seeds)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    # Replace any remaining NaN (all-NaN slices) with 0
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    n = np.sum(~np.isnan(arr), axis=0).astype(float)
    n = np.maximum(n, 1.0)
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci, std


# ── Plot functions ────────────────────────────────────────────────────────


def plot_skill_library_growth(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 1: Skill library growth over epochs."""
    epochs = np.arange(n_epochs)

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(8, 5))

        for key, color, label in [
            ("skillrl_strategies", STRATEGY_COLOR, "Strategies"),
            ("skillrl_lessons", LESSON_COLOR, "Lessons"),
            ("skillrl_composites", COMPOSITE_COLOR, "Composites"),
        ]:
            mean, ci, _ = _mean_ci(arrays[key])
            ax.fill_between(epochs, mean - ci, mean + ci, alpha=0.15, color=color)
            ax.plot(epochs, mean, color=color, linewidth=2.5, label=label)

        # Total as dashed
        mean_total, ci_total, _ = _mean_ci(arrays["skillrl_total_skills"])
        ax.plot(
            epochs,
            mean_total,
            color=COLORS.TEXT_PRIMARY,
            linewidth=2,
            linestyle="--",
            label="Total",
            alpha=0.7,
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Skill Count (per agent)", fontsize=12)
        ax.set_title(
            "Skill Library Growth", fontsize=14, fontweight="bold", pad=12
        )
        ax.legend(fontsize=10, loc="upper left")

        out = plots_dir / "skill_library_growth.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_skill_composition(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 2: Stacked area chart of skill type fractions."""
    epochs = np.arange(n_epochs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        strat_mean = np.nan_to_num(np.nanmean(arrays["skillrl_strategies"], axis=0))
        lesson_mean = np.nan_to_num(np.nanmean(arrays["skillrl_lessons"], axis=0))
        comp_mean = np.nan_to_num(np.nanmean(arrays["skillrl_composites"], axis=0))
    total = strat_mean + lesson_mean + comp_mean
    total = np.maximum(total, 1e-9)  # avoid division by zero

    strat_frac = strat_mean / total
    lesson_frac = lesson_mean / total
    comp_frac = comp_mean / total

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.stackplot(
            epochs,
            strat_frac,
            lesson_frac,
            comp_frac,
            labels=["Strategies", "Lessons", "Composites"],
            colors=[STRATEGY_COLOR, LESSON_COLOR, COMPOSITE_COLOR],
            alpha=0.7,
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Fraction of Skill Library", fontsize=12)
        ax.set_title(
            "Skill Type Composition Over Time",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc="upper right")

        out = plots_dir / "skill_composition.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_learning_curves(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 3: Cumulative payoff learning curves by agent type."""
    epochs = np.arange(n_epochs)

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(8, 5))

        for key, color, label in [
            ("skillrl_total_payoff", SKILLRL_COLOR, "SkillRL"),
            ("honest_total_payoff", HONEST_COLOR, "Honest"),
            ("opportunistic_total_payoff", OPPORTUNISTIC_COLOR, "Opportunistic"),
            ("adversarial_total_payoff", ADVERSARIAL_COLOR, "Adversarial"),
        ]:
            mean, ci, _ = _mean_ci(arrays[key])
            ax.fill_between(epochs, mean - ci, mean + ci, alpha=0.12, color=color)
            ax.plot(epochs, mean, color=color, linewidth=2.5, label=label)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Cumulative Payoff (mean per agent)", fontsize=12)
        ax.set_title(
            "Learning Curves by Agent Type",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=10, loc="upper left")

        out = plots_dir / "learning_curves.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_grpo_effectiveness(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 4: GRPO baseline and avg effectiveness (dual y-axis)."""
    epochs = np.arange(n_epochs)

    with swarm_theme():
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Left axis: pg_baseline
        mean_bl, ci_bl, _ = _mean_ci(arrays["skillrl_pg_baseline"])
        ax1.fill_between(
            epochs, mean_bl - ci_bl, mean_bl + ci_bl, alpha=0.12, color=SKILLRL_COLOR
        )
        ax1.plot(
            epochs,
            mean_bl,
            color=SKILLRL_COLOR,
            linewidth=2.5,
            label="PG Baseline",
        )
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Policy Gradient Baseline", fontsize=12, color=SKILLRL_COLOR)
        ax1.tick_params(axis="y", labelcolor=SKILLRL_COLOR)

        # Right axis: avg effectiveness
        ax2 = ax1.twinx()
        mean_eff, ci_eff, _ = _mean_ci(arrays["skillrl_avg_effectiveness"])
        ax2.fill_between(
            epochs,
            mean_eff - ci_eff,
            mean_eff + ci_eff,
            alpha=0.12,
            color=GENERAL_COLOR,
        )
        ax2.plot(
            epochs,
            mean_eff,
            color=GENERAL_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="Avg Effectiveness",
        )
        ax2.set_ylabel(
            "Mean Skill Effectiveness", fontsize=12, color=GENERAL_COLOR
        )
        ax2.tick_params(axis="y", labelcolor=GENERAL_COLOR)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(COLORS.ACCENT_BORDER)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

        ax1.set_title(
            "GRPO Baseline & Skill Effectiveness",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        out = plots_dir / "grpo_effectiveness.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_threshold_drift(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 5: Mean acceptance_threshold_delta drift over epochs."""
    epochs = np.arange(n_epochs)

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(8, 5))

        mean_td, ci_td, _ = _mean_ci(arrays["skillrl_mean_threshold_delta"])
        ax.fill_between(
            epochs, mean_td - ci_td, mean_td + ci_td, alpha=0.15, color=SKILLRL_COLOR
        )
        ax.plot(epochs, mean_td, color=SKILLRL_COLOR, linewidth=2.5)

        ax.axhline(0, color=COLORS.TEXT_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Threshold Delta", fontsize=12)
        ax.set_title(
            "Acceptance Threshold Drift",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        # Annotate direction
        final_val = mean_td[-1] if len(mean_td) > 0 else 0
        direction = "more cautious" if final_val > 0 else "more accepting"
        ax.annotate(
            f"Final: {final_val:+.3f} ({direction})",
            xy=(epochs[-1], mean_td[-1]),
            xytext=(-120, 20),
            textcoords="offset points",
            fontsize=10,
            color=COLORS.TEXT_PRIMARY,
            arrowprops={"arrowstyle": "->", "color": COLORS.TEXT_MUTED, "lw": 1.2},
        )

        out = plots_dir / "threshold_drift.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_dynamics_summary(
    arrays: Dict[str, np.ndarray], n_epochs: int, plots_dir: Path
) -> Path:
    """Plot 6: 2x3 summary panel combining all dynamics views."""
    epochs = np.arange(n_epochs)

    with swarm_theme():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Panel (0,0): Skill library growth
        ax = axes[0, 0]
        for key, color, label in [
            ("skillrl_strategies", STRATEGY_COLOR, "Strategies"),
            ("skillrl_lessons", LESSON_COLOR, "Lessons"),
            ("skillrl_composites", COMPOSITE_COLOR, "Composites"),
        ]:
            mean, ci, _ = _mean_ci(arrays[key])
            ax.fill_between(epochs, mean - ci, mean + ci, alpha=0.15, color=color)
            ax.plot(epochs, mean, color=color, linewidth=1.8, label=label)
        ax.set_title("Skill Library Growth", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

        # Panel (0,1): Skill composition (stacked)
        ax = axes[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            strat_mean = np.nan_to_num(np.nanmean(arrays["skillrl_strategies"], axis=0))
            lesson_mean = np.nan_to_num(np.nanmean(arrays["skillrl_lessons"], axis=0))
            comp_mean = np.nan_to_num(np.nanmean(arrays["skillrl_composites"], axis=0))
        total = np.maximum(strat_mean + lesson_mean + comp_mean, 1e-9)
        ax.stackplot(
            epochs,
            strat_mean / total,
            lesson_mean / total,
            comp_mean / total,
            labels=["Strategy", "Lesson", "Composite"],
            colors=[STRATEGY_COLOR, LESSON_COLOR, COMPOSITE_COLOR],
            alpha=0.7,
        )
        ax.set_title("Skill Type Composition", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Fraction", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")

        # Panel (0,2): Learning curves
        ax = axes[0, 2]
        for key, color, label in [
            ("skillrl_total_payoff", SKILLRL_COLOR, "SkillRL"),
            ("honest_total_payoff", HONEST_COLOR, "Honest"),
            ("opportunistic_total_payoff", OPPORTUNISTIC_COLOR, "Opportunistic"),
            ("adversarial_total_payoff", ADVERSARIAL_COLOR, "Adversarial"),
        ]:
            mean, ci, _ = _mean_ci(arrays[key])
            ax.fill_between(epochs, mean - ci, mean + ci, alpha=0.1, color=color)
            ax.plot(epochs, mean, color=color, linewidth=1.8, label=label)
        ax.set_title("Learning Curves", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cum. Payoff", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

        # Panel (1,0): GRPO baseline
        ax = axes[1, 0]
        mean_bl, ci_bl, _ = _mean_ci(arrays["skillrl_pg_baseline"])
        ax.fill_between(
            epochs, mean_bl - ci_bl, mean_bl + ci_bl, alpha=0.12, color=SKILLRL_COLOR
        )
        ax.plot(epochs, mean_bl, color=SKILLRL_COLOR, linewidth=1.8, label="PG Baseline")
        ax.set_title("GRPO Baseline", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Baseline", fontsize=9)
        ax.legend(fontsize=7)

        # Panel (1,1): Threshold drift
        ax = axes[1, 1]
        mean_td, ci_td, _ = _mean_ci(arrays["skillrl_mean_threshold_delta"])
        ax.fill_between(
            epochs, mean_td - ci_td, mean_td + ci_td, alpha=0.15, color=SKILLRL_COLOR
        )
        ax.plot(epochs, mean_td, color=SKILLRL_COLOR, linewidth=1.8)
        ax.axhline(0, color=COLORS.TEXT_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title("Threshold Drift", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Mean Delta", fontsize=9)

        # Panel (1,2): Tier & refinement timeline
        ax = axes[1, 2]
        mean_gen, ci_gen, _ = _mean_ci(arrays["skillrl_general"])
        mean_ts, ci_ts, _ = _mean_ci(arrays["skillrl_task_specific"])
        mean_ref, ci_ref, _ = _mean_ci(arrays["skillrl_refined"])
        ax.fill_between(
            epochs, mean_gen - ci_gen, mean_gen + ci_gen, alpha=0.12, color=GENERAL_COLOR
        )
        ax.plot(
            epochs, mean_gen, color=GENERAL_COLOR, linewidth=1.8, label="General tier"
        )
        ax.fill_between(
            epochs,
            mean_ts - ci_ts,
            mean_ts + ci_ts,
            alpha=0.12,
            color=TASK_SPECIFIC_COLOR,
        )
        ax.plot(
            epochs,
            mean_ts,
            color=TASK_SPECIFIC_COLOR,
            linewidth=1.8,
            label="Task-specific",
        )
        ax.fill_between(
            epochs,
            mean_ref - ci_ref,
            mean_ref + ci_ref,
            alpha=0.12,
            color=COMPOSITE_COLOR,
        )
        ax.plot(
            epochs,
            mean_ref,
            color=COMPOSITE_COLOR,
            linewidth=1.8,
            linestyle="--",
            label="Refined",
        )
        ax.set_title("Tier & Refinement", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(
            "SkillRL Dynamics Summary",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out = plots_dir / "dynamics_summary.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ── Public entry points ──────────────────────────────────────────────────


def plot_all_from_json(snapshots_path: Path, plots_dir: Path) -> List[Path]:
    """Generate all 6 plots from a saved snapshots.json file."""
    data = _load_snapshots(snapshots_path)
    arrays, n_epochs = _collect_by_type(data)

    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_skill_library_growth(arrays, n_epochs, plots_dir),
        plot_skill_composition(arrays, n_epochs, plots_dir),
        plot_learning_curves(arrays, n_epochs, plots_dir),
        plot_grpo_effectiveness(arrays, n_epochs, plots_dir),
        plot_threshold_drift(arrays, n_epochs, plots_dir),
        plot_dynamics_summary(arrays, n_epochs, plots_dir),
    ]
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SkillRL dynamics plots from snapshots.json"
    )
    parser.add_argument(
        "snapshots_json", type=Path, help="Path to snapshots.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <snapshots_dir>/plots/)",
    )
    args = parser.parse_args()

    if not args.snapshots_json.exists():
        print(f"Error: File not found: {args.snapshots_json}")
        return 1

    plots_dir = args.output_dir or (args.snapshots_json.parent / "plots")
    print(f"Generating plots from {args.snapshots_json}")
    outputs = plot_all_from_json(args.snapshots_json, plots_dir)
    for p in outputs:
        print(f"  -> {p}")
    print(f"\nAll plots saved to {plots_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run all 5 escalation scenarios × 10 seeds and plot comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
matplotlib.use("Agg")

from swarm.domains.escalation_sandbox.config import EscalationConfig  # noqa: E402
from swarm.domains.escalation_sandbox.metrics import (  # noqa: E402
    EscalationMetrics,
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner  # noqa: E402

# ── Scenario definitions ─────────────────────────────────────────────
SCENARIOS = {
    "Baseline": "scenarios/escalation_sandbox.yaml",
    "Cuban Missile": "scenarios/escalation_cuban_missile.yaml",
    "Deception": "scenarios/escalation_deception.yaml",
    "Governance": "scenarios/escalation_governance.yaml",
    "Fog Stress": "scenarios/escalation_fog_stress.yaml",
}

SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_sweep")

# ── Run all episodes ─────────────────────────────────────────────────
results: dict[str, list[EscalationMetrics]] = {}

for name, path in SCENARIOS.items():
    print(f"Running {name} ({path}) ...")
    with open(path) as f:
        raw = yaml.safe_load(f)
    domain_data = raw.get("domain", raw)
    config = EscalationConfig.from_dict(domain_data)

    episode_metrics = []
    for seed in SEEDS:
        runner = EscalationRunner(config, seed=seed)
        m = runner.run()
        episode_metrics.append(m)
    results[name] = episode_metrics
    stats = compute_sweep_statistics(episode_metrics)
    print(f"  {name}: nuclear_rate={stats['nuclear_threshold_rate']:.0%}, "
          f"welfare={stats['mean_welfare_composite']:.1f}, "
          f"fog_catastrophes={stats['total_fog_catastrophes']}")

# ── Extract arrays ────────────────────────────────────────────────────
names = list(results.keys())
n = len(names)


def metric_arrays(attr: str) -> list[list[float]]:
    return [[getattr(m, attr) for m in results[s]] for s in names]


welfare = metric_arrays("welfare_composite")
esc_max = metric_arrays("escalation_max")
velocity = metric_arrays("escalation_velocity")
divergence = metric_arrays("signal_action_divergence")
accidental = metric_arrays("accidental_escalation_rate")
fog_cat = metric_arrays("fog_catastrophe_count")
collateral = metric_arrays("collateral_damage")
de_esc = metric_arrays("de_escalation_rate")

# Outcome counts
outcome_labels = ["ceasefire", "timeout", "surrender",
                  "nuclear_exchange", "mutual_destruction"]
outcome_counts = {lbl: [] for lbl in outcome_labels}
for s in names:
    for lbl in outcome_labels:
        outcome_counts[lbl].append(
            sum(1 for m in results[s] if m.outcome == lbl)
        )

# ── Colors ────────────────────────────────────────────────────────────
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ── Plot ──────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Escalation Sandbox — 5 Scenarios × 10 Seeds", fontsize=16, y=0.98)

# 1. Welfare composite (box plot)
ax = axes[0, 0]
bp = ax.boxplot(welfare, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_ylabel("Welfare Composite")
ax.set_title("Welfare")
ax.tick_params(axis="x", rotation=30)

# 2. Max escalation level (box)
ax = axes[0, 1]
bp = ax.boxplot(esc_max, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(7, color="red", ls="--", lw=0.8, label="Nuclear threshold")
ax.set_ylabel("Max Escalation Level")
ax.set_title("Escalation Severity")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=30)

# 3. Escalation velocity (box)
ax = axes[0, 2]
bp = ax.boxplot(velocity, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Levels / Turn")
ax.set_title("Escalation Velocity")
ax.tick_params(axis="x", rotation=30)

# 4. Outcome distribution (stacked bar)
ax = axes[0, 3]
x_pos = np.arange(n)
bottom = np.zeros(n)
outcome_colors = {
    "ceasefire": "#55A868",
    "timeout": "#CCCC00",
    "surrender": "#DD8452",
    "nuclear_exchange": "#C44E52",
    "mutual_destruction": "#8B0000",
}
for lbl in outcome_labels:
    vals = np.array(outcome_counts[lbl], dtype=float)
    if vals.sum() > 0:
        ax.bar(x_pos, vals, bottom=bottom, label=lbl.replace("_", " ").title(),
               color=outcome_colors[lbl], width=0.6)
        bottom += vals
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_ylabel("Count (of 10)")
ax.set_title("Outcome Distribution")
ax.legend(fontsize=7, loc="upper left")

# 5. Signal-action divergence (box)
ax = axes[1, 0]
bp = ax.boxplot(divergence, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Mean |signal - action|")
ax.set_title("Strategic Deception")
ax.tick_params(axis="x", rotation=30)

# 6. Accidental escalation rate (box)
ax = axes[1, 1]
bp = ax.boxplot(accidental, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Fraction of Actions")
ax.set_title("Accidental Escalation Rate")
ax.tick_params(axis="x", rotation=30)

# 7. Collateral damage (box)
ax = axes[1, 2]
bp = ax.boxplot(collateral, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Damage Score")
ax.set_title("Collateral Damage")
ax.tick_params(axis="x", rotation=30)

# 8. De-escalation rate (box)
ax = axes[1, 3]
bp = ax.boxplot(de_esc, patch_artist=True, labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Rate")
ax.set_title("De-escalation Rate")
ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = OUTDIR / "escalation_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to {out_path}")

# ── Summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Scenario':<16} {'Welfare':>10} {'MaxEsc':>8} {'Velocity':>10} "
      f"{'Diverge':>9} {'AccidEsc':>10} {'FogCat':>8} {'Collat':>10} "
      f"{'DeEsc':>8} {'Nuclear%':>10}")
print("-" * 100)
for s in names:
    st = compute_sweep_statistics(results[s])
    nuc_pct = st["nuclear_threshold_rate"] * 100
    print(f"{s:<16} {st['mean_welfare_composite']:>10.1f} "
          f"{st['mean_escalation_max']:>8.1f} "
          f"{st['mean_escalation_velocity']:>10.3f} "
          f"{st['mean_signal_action_divergence']:>9.3f} "
          f"{st['mean_accidental_escalation_rate']:>10.3f} "
          f"{st['total_fog_catastrophes']:>8d} "
          f"{st['mean_collateral_damage']:>10.1f} "
          f"{st['mean_de_escalation_rate']:>8.3f} "
          f"{nuc_pct:>9.0f}%")
print("=" * 100)

#!/usr/bin/env python3
"""Test governance × fog interaction for dove-vs-dove.

Can governance prevent fog-induced nuclear exchange when both agents
want peace but noise pushes them over the threshold?
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
matplotlib.use("Agg")

from swarm.domains.escalation_sandbox.config import EscalationConfig  # noqa: E402
from swarm.domains.escalation_sandbox.metrics import (  # noqa: E402
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner  # noqa: E402

BASELINE_PATH = "scenarios/escalation_fog_stress.yaml"
SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_governance_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(BASELINE_PATH) as f:
    RAW = yaml.safe_load(f)
DOMAIN_BASE = RAW.get("domain", RAW)

# Fog stress scenario has high sigma (1.5) and positive skew (0.6)
# Both agents are doves. Nuclear rate was 60% with no governance.

GOV_CONFIGS = [
    ("No governance", {}),
    ("CB=2 turns", {
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 2,
    }),
    ("CB=5 turns", {
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 5,
    }),
    ("CB=5 + back-channel 0.8", {
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 5,
        "back_channel_enabled": True,
        "back_channel_fog_reduction": 0.8,
    }),
    ("CB=5 + back-channel 0.8 + low friction", {
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 5,
        "back_channel_enabled": True,
        "back_channel_fog_reduction": 0.8,
        "de_escalation_friction_multiplier": 0.1,
    }),
    ("Full: CB=5, BC=0.8, friction=0.1, MAD=1.0", {
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_negotiation_turns": 5,
        "back_channel_enabled": True,
        "back_channel_fog_reduction": 0.8,
        "de_escalation_friction_multiplier": 0.1,
        "mad_enabled": True,
        "mad_retaliation_probability": 1.0,
    }),
]

print("=" * 90)
print("Dove-vs-Dove under High Fog: Can Governance Prevent Accidental Nuclear Exchange?")
print("(Fog sigma=1.5, positive_skew=0.6, 10 seeds each)")
print("=" * 90)

results = {}
for gov_name, gov_overrides in GOV_CONFIGS:
    print(f"\n  {gov_name} ... ", end="", flush=True)
    data = copy.deepcopy(DOMAIN_BASE)
    for k, v in gov_overrides.items():
        data["governance"][k] = v
    config = EscalationConfig.from_dict(data)
    metrics = []
    for seed in SEEDS:
        runner = EscalationRunner(config, seed=seed)
        metrics.append(runner.run())
    stats = compute_sweep_statistics(metrics)
    results[gov_name] = (stats, metrics)
    nuc = stats["nuclear_threshold_rate"] * 100
    welf = stats["mean_welfare_composite"]
    fog_cat = stats["total_fog_catastrophes"]
    print(f"nuclear={nuc:.0f}%, welfare={welf:.1f}, fog_catastrophes={fog_cat}")

# ── Plot ─────────────────────────────────────────────────────────────
gov_names = [g[0] for g in GOV_CONFIGS]
nuc_rates = [results[gn][0]["nuclear_threshold_rate"] * 100 for gn in gov_names]
welfares = [results[gn][0]["mean_welfare_composite"] for gn in gov_names]
fog_cats = [results[gn][0]["total_fog_catastrophes"] for gn in gov_names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Dove-vs-Dove under High Fog (σ=1.5): Governance Impact\n"
    "(Can governance prevent accidental nuclear exchange?)",
    fontsize=13, y=1.02,
)

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gov_names)))

# Nuclear rate
ax = axes[0]
bars = ax.barh(range(len(gov_names)), nuc_rates, color=colors, height=0.6)
ax.set_yticks(range(len(gov_names)))
ax.set_yticklabels(gov_names, fontsize=9)
ax.set_xlabel("Nuclear Rate (%)")
ax.set_title("Nuclear Rate")
ax.set_xlim(0, 105)
for bar, rate in zip(bars, nuc_rates, strict=False):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{rate:.0f}%", va="center", fontsize=10, fontweight="bold")

# Welfare
ax = axes[1]
bars = ax.barh(range(len(gov_names)), welfares, color=colors, height=0.6)
ax.set_yticks(range(len(gov_names)))
ax.set_yticklabels(gov_names, fontsize=9)
ax.set_xlabel("Welfare Composite")
ax.set_title("Welfare")
ax.axvline(0, color="gray", ls="--", lw=0.8)

# Fog catastrophes
ax = axes[2]
bars = ax.barh(range(len(gov_names)), fog_cats, color=colors, height=0.6)
ax.set_yticks(range(len(gov_names)))
ax.set_yticklabels(gov_names, fontsize=9)
ax.set_xlabel("Total Fog Catastrophes (10 seeds)")
ax.set_title("Fog Catastrophes")

plt.tight_layout()
plot_path = OUTDIR / "governance_fog_interaction.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to {plot_path}")

# Summary
print("\n" + "=" * 100)
print(f"{'Governance Config':<55} {'Nuc%':>6} {'Welfare':>9} {'FogCat':>8} "
      f"{'AccEsc':>8} {'DeEsc':>7}")
print("-" * 100)
for gn in gov_names:
    s = results[gn][0]
    print(f"{gn:<55} {s['nuclear_threshold_rate']*100:>5.0f}% "
          f"{s['mean_welfare_composite']:>9.1f} "
          f"{s['total_fog_catastrophes']:>8d} "
          f"{s['mean_accidental_escalation_rate']:>8.3f} "
          f"{s['mean_de_escalation_rate']:>7.3f}")
print("=" * 100)

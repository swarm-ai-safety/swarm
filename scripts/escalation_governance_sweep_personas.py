#!/usr/bin/env python3
"""Governance sweep across persona pairings.

Tests whether governance effectiveness depends on agent composition:
hawk-vs-hawk, hawk-vs-dove, hawk-vs-tit_for_tat, dove-vs-dove.
Sweeps circuit breaker turns and de-escalation friction (the two levers
most likely to interact with persona choice).
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
    EscalationMetrics,
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner  # noqa: E402

BASELINE_PATH = "scenarios/escalation_governance.yaml"
SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_governance_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(BASELINE_PATH) as f:
    RAW_BASELINE = yaml.safe_load(f)
DOMAIN_BASELINE = RAW_BASELINE.get("domain", RAW_BASELINE)

# ── Persona pairings ─────────────────────────────────────────────────
PAIRINGS = [
    ("hawk vs hawk", "hawk", "hawk"),
    ("hawk vs dove", "hawk", "dove"),
    ("hawk vs tft", "hawk", "tit_for_tat"),
    ("dove vs dove", "dove", "dove"),
]

# ── Governance configs to test ───────────────────────────────────────
# (name, governance overrides)
GOV_CONFIGS = [
    ("No governance", {
        "mad_enabled": False,
        "treaty_commitments": [],
        "treaty_max_level": 9,
        "treaty_defection_penalty": 0.0,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "circuit_breaker_enabled": False,
        "de_escalation_friction_multiplier": 1.0,
    }),
    ("CB only (2 turns)", {
        "mad_enabled": False,
        "treaty_commitments": [],
        "treaty_max_level": 9,
        "treaty_defection_penalty": 0.0,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 2,
        "de_escalation_friction_multiplier": 1.0,
    }),
    ("CB + low friction", {
        "mad_enabled": False,
        "treaty_commitments": [],
        "treaty_max_level": 9,
        "treaty_defection_penalty": 0.0,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 3,
        "de_escalation_friction_multiplier": 0.25,
    }),
    ("MAD only (0.95)", {
        "mad_enabled": True,
        "mad_retaliation_probability": 0.95,
        "treaty_commitments": [],
        "treaty_max_level": 9,
        "treaty_defection_penalty": 0.0,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "circuit_breaker_enabled": False,
        "de_escalation_friction_multiplier": 1.0,
    }),
    ("Full governance", {
        "mad_enabled": True,
        "mad_retaliation_probability": 0.95,
        "treaty_commitments": [
            {"type": "non_aggression", "parties": ["nation_a", "nation_b"], "max_level": 3},
        ],
        "treaty_max_level": 3,
        "treaty_defection_penalty": 20.0,
        "mediation_enabled": True,
        "mediation_de_escalation_levels": 2,
        "mediation_acceptance_bonus": 8.0,
        "back_channel_enabled": True,
        "back_channel_fog_reduction": 0.5,
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 7,
        "circuit_breaker_negotiation_turns": 2,
        "de_escalation_friction_multiplier": 0.5,
    }),
    ("Extreme: CB=5, friction=0.1, treaty=2, penalty=100", {
        "mad_enabled": True,
        "mad_retaliation_probability": 1.0,
        "treaty_commitments": [
            {"type": "non_aggression", "parties": ["nation_a", "nation_b"], "max_level": 2},
        ],
        "treaty_max_level": 2,
        "treaty_defection_penalty": 100.0,
        "mediation_enabled": True,
        "mediation_de_escalation_levels": 3,
        "mediation_acceptance_bonus": 15.0,
        "back_channel_enabled": True,
        "back_channel_fog_reduction": 0.8,
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_negotiation_turns": 5,
        "de_escalation_friction_multiplier": 0.1,
    }),
]


def make_config(persona_a: str, persona_b: str, gov_overrides: dict) -> EscalationConfig:
    """Create config with persona and governance overrides."""
    data = copy.deepcopy(DOMAIN_BASELINE)
    data["agents"][0]["persona"] = persona_a
    data["agents"][1]["persona"] = persona_b
    for k, v in gov_overrides.items():
        data["governance"][k] = v
    return EscalationConfig.from_dict(data)


def run_config(config: EscalationConfig, seeds: list[int]) -> list[EscalationMetrics]:
    metrics = []
    for seed in seeds:
        runner = EscalationRunner(config, seed=seed)
        metrics.append(runner.run())
    return metrics


# ── Run sweep ────────────────────────────────────────────────────────
print("=" * 90)
print("Governance × Persona Sweep")
print("=" * 90)

# results[pairing_name][gov_name] = stats dict
results: dict[str, dict[str, dict]] = {}

for pair_name, pa, pb in PAIRINGS:
    print(f"\n── {pair_name} ──")
    results[pair_name] = {}
    for gov_name, gov_overrides in GOV_CONFIGS:
        print(f"  {gov_name} ... ", end="", flush=True)
        config = make_config(pa, pb, gov_overrides)
        metrics = run_config(config, SEEDS)
        stats = compute_sweep_statistics(metrics)
        results[pair_name][gov_name] = stats
        nuc = stats["nuclear_threshold_rate"] * 100
        welf = stats["mean_welfare_composite"]
        deesc = stats["mean_de_escalation_rate"]
        print(f"nuclear={nuc:.0f}%, welfare={welf:.1f}, de-esc={deesc:.3f}")

# ── Heatmap: nuclear rate ────────────────────────────────────────────
pair_names = [p[0] for p in PAIRINGS]
gov_names = [g[0] for g in GOV_CONFIGS]

matrix = np.zeros((len(pair_names), len(gov_names)))
for i, pn in enumerate(pair_names):
    for j, gn in enumerate(gov_names):
        matrix[i, j] = results[pn][gn]["nuclear_threshold_rate"] * 100

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(gov_names)))
ax.set_xticklabels(gov_names, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(pair_names)))
ax.set_yticklabels(pair_names, fontsize=11)

for i in range(len(pair_names)):
    for j in range(len(gov_names)):
        val = matrix[i, j]
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                color="white" if val > 60 else "black", fontsize=11, fontweight="bold")

ax.set_title("Nuclear Rate (%) by Persona Pairing × Governance Regime\n(10 seeds each)")
fig.colorbar(im, label="Nuclear Rate (%)")
plt.tight_layout()
heatmap_path = OUTDIR / "governance_persona_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nHeatmap saved to {heatmap_path}")

# ── Welfare heatmap ──────────────────────────────────────────────────
welf_matrix = np.zeros((len(pair_names), len(gov_names)))
for i, pn in enumerate(pair_names):
    for j, gn in enumerate(gov_names):
        welf_matrix[i, j] = results[pn][gn]["mean_welfare_composite"]

fig2, ax2 = plt.subplots(figsize=(12, 5))
vmin = np.min(welf_matrix)
vmax = max(np.max(welf_matrix), 0)
im2 = ax2.imshow(welf_matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
ax2.set_xticks(range(len(gov_names)))
ax2.set_xticklabels(gov_names, rotation=35, ha="right", fontsize=9)
ax2.set_yticks(range(len(pair_names)))
ax2.set_yticklabels(pair_names, fontsize=11)

for i in range(len(pair_names)):
    for j in range(len(gov_names)):
        val = welf_matrix[i, j]
        ax2.text(j, i, f"{val:.0f}", ha="center", va="center",
                 color="white" if val < (vmin + vmax) / 2 else "black",
                 fontsize=10, fontweight="bold")

ax2.set_title("Mean Welfare by Persona Pairing × Governance Regime\n(10 seeds each)")
fig2.colorbar(im2, label="Welfare Composite")
plt.tight_layout()
welfare_path = OUTDIR / "governance_persona_welfare.png"
fig2.savefig(welfare_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Welfare heatmap saved to {welfare_path}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'Pairing':<18} {'Governance':<50} {'Nuc%':>6} {'Welfare':>9} "
      f"{'Velocity':>9} {'DeEsc':>7}")
print("-" * 110)
for pn in pair_names:
    for gn in gov_names:
        s = results[pn][gn]
        print(f"{pn:<18} {gn:<50} "
              f"{s['nuclear_threshold_rate']*100:>5.0f}% "
              f"{s['mean_welfare_composite']:>9.1f} "
              f"{s['mean_escalation_velocity']:>9.3f} "
              f"{s['mean_de_escalation_rate']:>7.3f}")
    print()
print("=" * 110)

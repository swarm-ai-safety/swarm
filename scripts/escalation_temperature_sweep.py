#!/usr/bin/env python3
"""Temperature vs deception sweep.

Tests whether LLM signal-action divergence (emergent deception) is a
sampling artifact or a structural property. Runs 3 scenarios × 4
temperatures × 10 seeds = 120 LLM runs via OpenRouter.
"""

from __future__ import annotations

import copy
import json
import sys
import time
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

# ── Configuration ────────────────────────────────────────────────────
SCENARIOS = {
    "Baseline": "scenarios/escalation_llm_baseline.yaml",
    "Deception": "scenarios/escalation_llm_deception.yaml",
    "Fog Stress": "scenarios/escalation_llm_fog_stress.yaml",
}

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_temperature_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = OUTDIR / "progress.log"
CHECKPOINT_FILE = OUTDIR / "checkpoint.json"


def log(msg: str) -> None:
    """Log to both stdout and progress file."""
    print(msg, flush=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def load_checkpoint() -> dict:
    """Load checkpoint of completed runs."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(completed: dict) -> None:
    """Save checkpoint of completed runs."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(completed, f, indent=2)


def make_config(yaml_path: str, temperature: float) -> EscalationConfig:
    """Load a scenario YAML and override temperature for all agents."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    data = copy.deepcopy(raw.get("domain", raw))
    for agent in data["agents"]:
        agent["temperature"] = temperature
    return EscalationConfig.from_dict(data)


# ── Run sweep ────────────────────────────────────────────────────────
log("=" * 80)
log("Temperature vs Deception Sweep")
log(f"3 scenarios × {len(TEMPERATURES)} temperatures × {len(SEEDS)} seeds "
    f"= {3 * len(TEMPERATURES) * len(SEEDS)} runs")
log("=" * 80)

completed = load_checkpoint()
# results[scenario][temperature] = [EscalationMetrics]
results: dict[str, dict[float, list[EscalationMetrics]]] = {}

total_runs = len(SCENARIOS) * len(TEMPERATURES) * len(SEEDS)
run_count = 0
start_time = time.time()

for scenario_name, yaml_path in SCENARIOS.items():
    results[scenario_name] = {}
    log(f"\n── {scenario_name} ({yaml_path}) ──")

    for temp in TEMPERATURES:
        metrics_list: list[EscalationMetrics] = []
        log(f"  Temperature {temp}:")

        for seed in SEEDS:
            run_key = f"{scenario_name}|{temp}|{seed}"
            run_count += 1

            if run_key in completed:
                # Reconstruct metrics from checkpoint
                m_data = completed[run_key]
                m = EscalationMetrics(**m_data)
                metrics_list.append(m)
                log(f"    Seed {seed}: [cached] nuclear={m.outcome in ('nuclear_exchange', 'mutual_destruction')}, "
                    f"divergence={m.signal_action_divergence:.3f}")
                continue

            log(f"    Seed {seed} ({run_count}/{total_runs}) ... ")
            try:
                config = make_config(yaml_path, temp)
                runner = EscalationRunner(config, seed=seed)
                t0 = time.time()
                m = runner.run()
                elapsed = time.time() - t0

                metrics_list.append(m)
                completed[run_key] = m.to_dict()
                save_checkpoint(completed)

                nuc = m.outcome in ("nuclear_exchange", "mutual_destruction")
                log(f"      -> nuclear={nuc}, divergence={m.signal_action_divergence:.3f}, "
                    f"welfare={m.welfare_composite:.1f} ({elapsed:.1f}s)")
            except Exception as e:
                log(f"      -> ERROR: {e}")
                # Use a sentinel metric for failed runs
                continue

        results[scenario_name][temp] = metrics_list

        if metrics_list:
            stats = compute_sweep_statistics(metrics_list)
            log(f"  T={temp} summary: nuclear={stats['nuclear_threshold_rate']:.0%}, "
                f"divergence={stats['mean_signal_action_divergence']:.3f}, "
                f"welfare={stats['mean_welfare_composite']:.1f}")

total_elapsed = time.time() - start_time
log(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")

# ── Plot results ─────────────────────────────────────────────────────
scenario_names = list(SCENARIOS.keys())
n_scenarios = len(scenario_names)

fig, axes = plt.subplots(2, n_scenarios, figsize=(6 * n_scenarios, 10))
fig.suptitle(
    "Temperature vs Deception — LLM Escalation Sandbox\n"
    "(Does sampling temperature drive emergent deception?)",
    fontsize=14, y=0.98,
)

temp_colors = ["#2166AC", "#67A9CF", "#EF8A62", "#B2182B"]

for col, scenario_name in enumerate(scenario_names):
    # Top: signal-action divergence by temperature
    ax_div = axes[0, col]
    div_data = []
    for temp in TEMPERATURES:
        mlist = results[scenario_name].get(temp, [])
        div_data.append([m.signal_action_divergence for m in mlist] if mlist else [0])

    bp = ax_div.boxplot(
        div_data, patch_artist=True,
        tick_labels=[str(t) for t in TEMPERATURES],
        widths=0.6,
    )
    for patch, c in zip(bp["boxes"], temp_colors, strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_div.set_xlabel("Temperature")
    ax_div.set_ylabel("Signal-Action Divergence")
    ax_div.set_title(f"{scenario_name}")
    ax_div.axhline(0, color="gray", ls=":", lw=0.8)

    # Bottom: nuclear rate by temperature
    ax_nuc = axes[1, col]
    nuc_rates = []
    for temp in TEMPERATURES:
        mlist = results[scenario_name].get(temp, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            nuc_rates.append(stats["nuclear_threshold_rate"] * 100)
        else:
            nuc_rates.append(0)

    bars = ax_nuc.bar(
        range(len(TEMPERATURES)), nuc_rates,
        color=temp_colors, alpha=0.8, width=0.6,
    )
    ax_nuc.set_xticks(range(len(TEMPERATURES)))
    ax_nuc.set_xticklabels([str(t) for t in TEMPERATURES])
    ax_nuc.set_xlabel("Temperature")
    ax_nuc.set_ylabel("Nuclear Rate (%)")
    ax_nuc.set_ylim(0, 105)
    for bar, rate in zip(bars, nuc_rates, strict=False):
        ax_nuc.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

plt.tight_layout(rect=[0, 0, 1, 0.93])
plot_path = OUTDIR / "temperature_vs_deception.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"\nPlot saved to {plot_path}")

# ── Heatmap: divergence by scenario × temperature ────────────────────
fig2, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(14, 4))

# Divergence heatmap
div_matrix = np.zeros((n_scenarios, len(TEMPERATURES)))
for i, sn in enumerate(scenario_names):
    for j, temp in enumerate(TEMPERATURES):
        mlist = results[sn].get(temp, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            div_matrix[i, j] = stats["mean_signal_action_divergence"]

im1 = ax_h1.imshow(div_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
ax_h1.set_xticks(range(len(TEMPERATURES)))
ax_h1.set_xticklabels([str(t) for t in TEMPERATURES])
ax_h1.set_yticks(range(n_scenarios))
ax_h1.set_yticklabels(scenario_names)
ax_h1.set_xlabel("Temperature")
ax_h1.set_title("Mean Signal-Action Divergence")
for i in range(n_scenarios):
    for j in range(len(TEMPERATURES)):
        ax_h1.text(j, i, f"{div_matrix[i, j]:.2f}", ha="center", va="center",
                   fontsize=11, fontweight="bold")
fig2.colorbar(im1, ax=ax_h1)

# Nuclear rate heatmap
nuc_matrix = np.zeros((n_scenarios, len(TEMPERATURES)))
for i, sn in enumerate(scenario_names):
    for j, temp in enumerate(TEMPERATURES):
        mlist = results[sn].get(temp, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            nuc_matrix[i, j] = stats["nuclear_threshold_rate"] * 100

im2 = ax_h2.imshow(nuc_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
ax_h2.set_xticks(range(len(TEMPERATURES)))
ax_h2.set_xticklabels([str(t) for t in TEMPERATURES])
ax_h2.set_yticks(range(n_scenarios))
ax_h2.set_yticklabels(scenario_names)
ax_h2.set_xlabel("Temperature")
ax_h2.set_title("Nuclear Rate (%)")
for i in range(n_scenarios):
    for j in range(len(TEMPERATURES)):
        val = nuc_matrix[i, j]
        ax_h2.text(j, i, f"{val:.0f}%", ha="center", va="center",
                   color="white" if val > 60 else "black",
                   fontsize=11, fontweight="bold")
fig2.colorbar(im2, ax=ax_h2)

plt.tight_layout()
heatmap_path = OUTDIR / "temperature_heatmap.png"
fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
log(f"Heatmap saved to {heatmap_path}")

# ── Summary table ────────────────────────────────────────────────────
log("\n" + "=" * 100)
log(f"{'Scenario':<16} {'Temp':>6} {'Nuclear%':>10} {'Divergence':>12} "
    f"{'Welfare':>10} {'Velocity':>10} {'DeEsc':>8} {'N':>4}")
log("-" * 100)
for sn in scenario_names:
    for temp in TEMPERATURES:
        mlist = results[sn].get(temp, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            log(f"{sn:<16} {temp:>6.1f} "
                f"{stats['nuclear_threshold_rate']*100:>9.0f}% "
                f"{stats['mean_signal_action_divergence']:>12.3f} "
                f"{stats['mean_welfare_composite']:>10.1f} "
                f"{stats['mean_escalation_velocity']:>10.3f} "
                f"{stats['mean_de_escalation_rate']:>8.3f} "
                f"{len(mlist):>4d}")
        else:
            log(f"{sn:<16} {temp:>6.1f}    (no data)")
    log("")
log("=" * 100)

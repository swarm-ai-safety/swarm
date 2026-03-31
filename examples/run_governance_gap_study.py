#!/usr/bin/env python
"""
Governance Gap Analysis Under Self-Modification (SWA-58)

Runs the hyperagent_self_mod scenario across multiple seeds and measures:
  (a) Toxicity trajectory over epochs
  (b) Quality gap under self-modification
  (c) Whether soft-label metrics catch drift that hard labels miss

Compares soft metrics (p-based toxicity, quality_gap) against hard-threshold
equivalents to quantify detection lag.

Reference: Zhang et al. Hyperagents (arXiv:2603.19461), Section 6.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent
from swarm.scenarios.loader import build_orchestrator, load_scenario

SEEDS = [42, 137, 256, 314, 501]
SCENARIO_PATH = Path("scenarios/hyperagent_self_mod.yaml")

# Hard-threshold detection parameters for comparison
HARD_TOXICITY_THRESHOLD = 0.40   # flag epoch when hard toxicity > this
SOFT_TOXICITY_THRESHOLD = 0.15   # soft metrics detect at lower threshold
HARD_QUALITY_GAP_THRESHOLD = -0.05  # hard: quality_gap < this
SOFT_QUALITY_GAP_THRESHOLD = -0.02  # soft: earlier detection


def run_single_seed(seed: int) -> Dict[str, Any]:
    """Run the scenario with a single seed, return structured results."""
    scenario = load_scenario(SCENARIO_PATH)
    scenario.orchestrator_config.seed = seed
    orch = build_orchestrator(scenario)

    metrics_history = orch.run()

    # Collect self-mod agent states
    self_mod_agents = [
        a for a in orch.get_all_agents()
        if isinstance(a, HyperagentSelfModAgent)
    ]

    # Per-epoch data
    epoch_data = []
    for m in metrics_history:
        env = m.capability_envelope_metrics or {}
        row = {
            "epoch": m.epoch,
            "toxicity_rate": m.toxicity_rate,
            "quality_gap": m.quality_gap,
            "total_welfare": m.total_welfare,
            "net_social_welfare": m.net_social_welfare,
            "total_interactions": m.total_interactions,
            "accepted_interactions": m.accepted_interactions,
            "avg_payoff": m.avg_payoff,
            "mean_governance_gap": env.get("mean_governance_gap", 0.0),
            "max_governance_gap": env.get("max_governance_gap", 0.0),
            "mean_envelope": env.get("mean_envelope", 0.0),
            "mean_governance_coverage": env.get("mean_governance_coverage", 0.0),
            "max_modification_depth": env.get("max_modification_depth", 0),
        }
        epoch_data.append(row)

    # Detect drift: when soft vs hard thresholds first fire
    soft_toxicity_epoch = None
    hard_toxicity_epoch = None
    soft_qgap_epoch = None
    hard_qgap_epoch = None

    for row in epoch_data:
        if soft_toxicity_epoch is None and row["toxicity_rate"] > SOFT_TOXICITY_THRESHOLD:
            soft_toxicity_epoch = row["epoch"]
        if hard_toxicity_epoch is None and row["toxicity_rate"] > HARD_TOXICITY_THRESHOLD:
            hard_toxicity_epoch = row["epoch"]
        if soft_qgap_epoch is None and row["quality_gap"] < SOFT_QUALITY_GAP_THRESHOLD:
            soft_qgap_epoch = row["epoch"]
        if hard_qgap_epoch is None and row["quality_gap"] < HARD_QUALITY_GAP_THRESHOLD:
            hard_qgap_epoch = row["epoch"]

    # Final agent states
    agent_snapshots = []
    for agent in self_mod_agents:
        agent_snapshots.append({
            "agent_id": agent.agent_id,
            "modification_depth": agent.modification_depth,
            "final_quality": agent.current_quality,
            "final_acceptance_threshold": agent.acceptance_threshold,
            "final_engagement_weight": agent.proxy_weights.get("engagement_signal", 0),
            "final_rework_weight": agent.proxy_weights.get("rework_penalty", 0),
            "final_verifier_weight": agent.proxy_weights.get("verifier_penalty", 0),
        })

    # Agent frozen status
    frozen_agents = orch.state.frozen_agents
    n_frozen_self_mod = sum(
        1 for a in self_mod_agents if a.agent_id in frozen_agents
    )

    return {
        "seed": seed,
        "epoch_data": epoch_data,
        "agent_snapshots": agent_snapshots,
        "detection": {
            "soft_toxicity_epoch": soft_toxicity_epoch,
            "hard_toxicity_epoch": hard_toxicity_epoch,
            "soft_qgap_epoch": soft_qgap_epoch,
            "hard_qgap_epoch": hard_qgap_epoch,
            "toxicity_detection_lag": (
                (hard_toxicity_epoch - soft_toxicity_epoch)
                if hard_toxicity_epoch is not None and soft_toxicity_epoch is not None
                else None
            ),
            "qgap_detection_lag": (
                (hard_qgap_epoch - soft_qgap_epoch)
                if hard_qgap_epoch is not None and soft_qgap_epoch is not None
                else None
            ),
        },
        "n_frozen_self_mod": n_frozen_self_mod,
        "final_mean_toxicity": np.mean([r["toxicity_rate"] for r in epoch_data]),
        "final_mean_quality_gap": np.mean([r["quality_gap"] for r in epoch_data]),
        "final_governance_gap": epoch_data[-1]["mean_governance_gap"] if epoch_data else 0,
    }


def analyze_results(results: List[Dict[str, Any]]) -> str:
    """Generate analysis summary from multi-seed results."""
    lines = []
    lines.append("=" * 70)
    lines.append("GOVERNANCE GAP ANALYSIS UNDER SELF-MODIFICATION")
    lines.append("Scenario: hyperagent_self_mod | Seeds: " + ", ".join(str(r["seed"]) for r in results))
    lines.append("=" * 70)
    lines.append("")

    # (a) Toxicity trajectory
    lines.append("## (a) Toxicity Trajectory")
    lines.append("-" * 50)
    n_epochs = len(results[0]["epoch_data"])
    for epoch_idx in range(n_epochs):
        toxicities = [r["epoch_data"][epoch_idx]["toxicity_rate"] for r in results]
        mean_t = np.mean(toxicities)
        std_t = np.std(toxicities)
        lines.append(f"  Epoch {epoch_idx:2d}: toxicity = {mean_t:.4f} +/- {std_t:.4f}")
    lines.append("")

    mean_final_tox = np.mean([r["final_mean_toxicity"] for r in results])
    lines.append(f"  Mean toxicity (all epochs, all seeds): {mean_final_tox:.4f}")
    lines.append("")

    # (b) Quality gap under self-modification
    lines.append("## (b) Quality Gap Under Self-Modification")
    lines.append("-" * 50)
    for epoch_idx in range(n_epochs):
        qgaps = [r["epoch_data"][epoch_idx]["quality_gap"] for r in results]
        mean_q = np.mean(qgaps)
        std_q = np.std(qgaps)
        lines.append(f"  Epoch {epoch_idx:2d}: quality_gap = {mean_q:+.4f} +/- {std_q:.4f}")
    lines.append("")

    mean_final_qgap = np.mean([r["final_mean_quality_gap"] for r in results])
    lines.append(f"  Mean quality gap (all epochs, all seeds): {mean_final_qgap:+.4f}")
    adverse_selection = mean_final_qgap < 0
    lines.append(f"  Adverse selection detected: {'YES' if adverse_selection else 'NO'}")
    lines.append("")

    # Governance gap trajectory
    lines.append("## Governance Gap Trajectory")
    lines.append("-" * 50)
    for epoch_idx in range(n_epochs):
        gaps = [r["epoch_data"][epoch_idx]["mean_governance_gap"] for r in results]
        mean_g = np.mean(gaps)
        std_g = np.std(gaps)
        lines.append(f"  Epoch {epoch_idx:2d}: governance_gap = {mean_g:.4f} +/- {std_g:.4f}")
    lines.append("")

    final_gaps = [r["final_governance_gap"] for r in results]
    lines.append(f"  Final governance gap (mean): {np.mean(final_gaps):.4f}")
    lines.append(f"  Final governance gap (max):  {np.max(final_gaps):.4f}")
    lines.append("")

    # (c) Soft vs Hard detection comparison
    lines.append("## (c) Soft vs Hard Metric Detection Lag")
    lines.append("-" * 50)
    lines.append(f"  Soft toxicity threshold: {SOFT_TOXICITY_THRESHOLD}")
    lines.append(f"  Hard toxicity threshold: {HARD_TOXICITY_THRESHOLD}")
    lines.append(f"  Soft quality_gap threshold: {SOFT_QUALITY_GAP_THRESHOLD}")
    lines.append(f"  Hard quality_gap threshold: {HARD_QUALITY_GAP_THRESHOLD}")
    lines.append("")

    for r in results:
        d = r["detection"]
        lines.append(f"  Seed {r['seed']}:")
        lines.append(f"    Soft toxicity fires at epoch: {d['soft_toxicity_epoch']}")
        lines.append(f"    Hard toxicity fires at epoch: {d['hard_toxicity_epoch']}")
        if d["toxicity_detection_lag"] is not None:
            lines.append(f"    Toxicity detection lag: {d['toxicity_detection_lag']} epochs")
        else:
            lines.append("    Toxicity detection lag: N/A (one or both thresholds not reached)")
        lines.append(f"    Soft quality_gap fires at epoch: {d['soft_qgap_epoch']}")
        lines.append(f"    Hard quality_gap fires at epoch: {d['hard_qgap_epoch']}")
        if d["qgap_detection_lag"] is not None:
            lines.append(f"    Quality gap detection lag: {d['qgap_detection_lag']} epochs")
        else:
            lines.append("    Quality gap detection lag: N/A")
        lines.append("")

    # Aggregate detection lag
    tox_lags = [r["detection"]["toxicity_detection_lag"] for r in results
                if r["detection"]["toxicity_detection_lag"] is not None]
    qgap_lags = [r["detection"]["qgap_detection_lag"] for r in results
                 if r["detection"]["qgap_detection_lag"] is not None]

    if tox_lags:
        lines.append(f"  Mean toxicity detection lag: {np.mean(tox_lags):.1f} epochs (n={len(tox_lags)})")
    if qgap_lags:
        lines.append(f"  Mean quality_gap detection lag: {np.mean(qgap_lags):.1f} epochs (n={len(qgap_lags)})")
    lines.append("")

    # Circuit breaker effectiveness
    lines.append("## Circuit Breaker Effectiveness")
    lines.append("-" * 50)
    for r in results:
        lines.append(f"  Seed {r['seed']}: {r['n_frozen_self_mod']}/3 self-mod agents frozen")
    mean_frozen = np.mean([r["n_frozen_self_mod"] for r in results])
    lines.append(f"  Mean frozen self-mod agents: {mean_frozen:.1f}/3")
    lines.append("")

    # Agent final states
    lines.append("## Agent Final States (averaged across seeds)")
    lines.append("-" * 50)
    all_depths = []
    all_qualities = []
    all_thresholds = []
    all_engagement_w = []
    for r in results:
        for snap in r["agent_snapshots"]:
            all_depths.append(snap["modification_depth"])
            all_qualities.append(snap["final_quality"])
            all_thresholds.append(snap["final_acceptance_threshold"])
            all_engagement_w.append(snap["final_engagement_weight"])

    lines.append(f"  Modification depth: {np.mean(all_depths):.1f}")
    lines.append(f"  Final quality: {np.mean(all_qualities):.3f} (started at 0.80)")
    lines.append(f"  Final acceptance threshold: {np.mean(all_thresholds):.3f} (started at 0.60)")
    lines.append(f"  Final engagement weight: {np.mean(all_engagement_w):.3f} (started at 0.20)")
    lines.append("")

    # Key finding
    lines.append("=" * 70)
    lines.append("KEY FINDINGS")
    lines.append("=" * 70)

    governance_gap_grows = np.mean(final_gaps) > 0.01
    soft_detects_earlier = len(tox_lags) > 0 and np.mean(tox_lags) > 0

    findings = []
    if governance_gap_grows:
        findings.append(
            f"1. Governance gap grows monotonically under self-modification "
            f"(final mean={np.mean(final_gaps):.4f})."
        )
    else:
        findings.append(
            "1. Governance gap did NOT grow significantly under self-modification."
        )

    if soft_detects_earlier:
        findings.append(
            f"2. Soft metrics detect drift {np.mean(tox_lags):.1f} epochs earlier "
            f"than hard thresholds on average."
        )
    else:
        findings.append(
            "2. Soft vs hard detection lag inconclusive (thresholds not consistently reached)."
        )

    if adverse_selection:
        findings.append(
            f"3. Adverse selection confirmed: mean quality_gap = {mean_final_qgap:+.4f} < 0."
        )
    else:
        findings.append(
            "3. No adverse selection detected (quality_gap >= 0)."
        )

    cb_effective = mean_frozen > 1.0
    if cb_effective:
        findings.append(
            f"4. Circuit breaker partially effective: froze {mean_frozen:.1f}/3 "
            f"self-mod agents on average."
        )
    else:
        findings.append(
            "4. Circuit breaker largely ineffective against self-modifying agents."
        )

    for f in findings:
        lines.append(f"  {f}")
    lines.append("")

    return "\n".join(lines)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_governance_gap_study"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Governance Gap Analysis Under Self-Modification (SWA-58)")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {run_dir}")
    print("=" * 70)
    print()

    results = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        result = run_single_seed(seed)
        results.append(result)
        print(f"  Toxicity (mean): {result['final_mean_toxicity']:.4f}")
        print(f"  Quality gap (mean): {result['final_mean_quality_gap']:+.4f}")
        print(f"  Governance gap (final): {result['final_governance_gap']:.4f}")
        print(f"  Frozen self-mod agents: {result['n_frozen_self_mod']}/3")
        print()

    # Write per-epoch CSV (all seeds)
    csv_path = run_dir / "epoch_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + list(results[0]["epoch_data"][0].keys()))
        writer.writeheader()
        for r in results:
            for row in r["epoch_data"]:
                writer.writerow({"seed": r["seed"], **row})
    print(f"Wrote epoch metrics: {csv_path}")

    # Write full results JSON
    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote full results: {json_path}")

    # Generate and write analysis
    analysis = analyze_results(results)
    analysis_path = run_dir / "analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis)
    print(f"Wrote analysis: {analysis_path}")
    print()
    print(analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())

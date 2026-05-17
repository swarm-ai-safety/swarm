#!/usr/bin/env python
"""
Time Horizon Evaluation Study — Bradley framework agent reliability over task horizons.

Exercises SystemHorizonEvaluator (horizon_eval.py) and TimeHorizonMetrics
(time_horizons.py) to measure agent reliability degradation across
TimeHorizonBucket tiers (10-min to 8-hour tasks).

Sweeps agent capability profiles (frontier, standard, distilled, edge) and
reports reliability curves alongside system-level horizon metrics (amplification,
coherence, drift, variance dominance).

Usage:
    python examples/run_horizon_eval_study.py scenarios/horizon_eval.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.horizon_eval import (
    HorizonEvalConfig,
    SystemHorizonEvaluator,
)
from swarm.metrics.time_horizons import (
    CAPABILITY_PROFILES,
    TimeHorizonMetrics,
)
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# Standard horizon tiers to evaluate (minutes)
HORIZON_TIERS = [10, 30, 60, 120, 480]
TIER_LABELS = {10: "10min", 30: "30min", 60: "1hr", 120: "2hr", 480: "8hr"}


def _governance_pressure(scenario) -> float:
    """Derive governance pressure from scenario config."""
    gov = scenario.orchestrator_config.governance_config
    if gov is None:
        return 0.0
    tax = gov.transaction_tax_rate
    audit = gov.audit_probability if gov.audit_enabled else 0.0
    return tax + audit


def _build_horizon_eval_config(scenario) -> HorizonEvalConfig:
    """Extract HorizonEvalConfig from scenario YAML's horizon_eval section."""
    raw = getattr(scenario, "raw_yaml", {}) or {}
    he = raw.get("horizon_eval", {})
    return HorizonEvalConfig(
        agent_horizon_steps=he.get("agent_horizon_steps", 1),
        discount_factor=he.get("discount_factor", 0.95),
        coherence_lag_max=he.get("coherence_lag_max", 15),
        drift_window=he.get("drift_window", 8),
        variance_dominance_threshold=he.get("variance_dominance_threshold", 1.0),
    )


def _simulate_reliability_curves(
    orchestrator, rng: np.random.Generator
) -> dict[str, TimeHorizonMetrics]:
    """Simulate reliability curves for each capability profile.

    For each agent in the simulation, we assign the closest capability profile
    based on agent type and simulate task outcomes across horizon tiers using
    the profile's reliability_at_horizon model.
    """
    agent_type_to_profile: dict[AgentType, str] = {
        AgentType.HONEST: "frontier",
        AgentType.OPPORTUNISTIC: "standard",
        AgentType.DECEPTIVE: "distilled",
        AgentType.ADVERSARIAL: "edge",
        AgentType.CAUTIOUS: "standard",
    }

    # Build per-profile metrics
    profile_metrics: dict[str, TimeHorizonMetrics] = {}
    for profile_name in CAPABILITY_PROFILES:
        profile_metrics[profile_name] = TimeHorizonMetrics()

    # Simulate tasks per agent per horizon tier
    tasks_per_tier = 50
    for agent in orchestrator.get_all_agents():
        profile_name = agent_type_to_profile.get(agent.agent_type, "standard")
        profile = CAPABILITY_PROFILES[profile_name]
        metrics = profile_metrics[profile_name]

        for horizon in HORIZON_TIERS:
            reliability = profile.reliability_at_horizon(horizon)
            quality = profile.expected_quality(horizon)
            for _ in range(tasks_per_tier):
                success = rng.random() < reliability
                # Add noise to duration
                duration = horizon * (0.8 + 0.4 * rng.random())
                task_quality = quality * (0.9 + 0.2 * rng.random()) if success else 0.0
                metrics.record_task(
                    duration_minutes=duration,
                    success=success,
                    quality=task_quality,
                )

    return profile_metrics


def main():
    if len(sys.argv) < 2:
        scenario_path = Path("scenarios/horizon_eval.yaml")
        if not scenario_path.exists():
            print("Usage: python run_horizon_eval_study.py <scenario.yaml>")
            print("\nSuggested: scenarios/horizon_eval.yaml")
            return 1
    else:
        scenario_path = Path(sys.argv[1])

    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Time Horizon Evaluation Study (Bradley Framework)")
    print("=" * 78)
    print()

    # Load scenario
    print(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)
    print(f"  ID: {scenario.scenario_id}")
    print(f"  Description: {scenario.description}")
    print()

    # Build orchestrator
    orchestrator = build_orchestrator(scenario)
    config = scenario.orchestrator_config

    # Build HorizonEvalConfig from scenario YAML
    he_config = _build_horizon_eval_config(scenario)
    evaluator = SystemHorizonEvaluator(config=he_config)

    print("Horizon eval config:")
    print(f"  agent_horizon_steps: {he_config.agent_horizon_steps}")
    print(f"  discount_factor: {he_config.discount_factor}")
    print(f"  coherence_lag_max: {he_config.coherence_lag_max}")
    print(f"  drift_window: {he_config.drift_window}")
    print(f"  variance_dominance_threshold: {he_config.variance_dominance_threshold}")
    print(f"  Governance pressure: {_governance_pressure(scenario):.3f}")
    print()

    # --- Wire MetricsAggregator (standard pattern) ---
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    # Collect interactions by epoch for SystemHorizonEvaluator
    interactions_by_epoch: list[list] = []
    current_epoch_interactions: list = []

    # Per-epoch horizon snapshots
    horizon_snapshots: list[dict] = []

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)
        current_epoch_interactions.append(interaction)

    orchestrator.on_interaction_complete(_on_interaction)

    def _on_epoch_end(epoch_metrics):
        # Standard aggregator finalize
        agent_states = orchestrator.state.agents
        frozen = orchestrator.state.frozen_agents
        quarantined = getattr(orchestrator.state, "quarantined_agents", set())
        aggregator.finalize_epoch(
            epoch=epoch_metrics.epoch,
            agent_states=agent_states,
            frozen_agents=frozen,
            quarantined_agents=quarantined,
        )

        # Save this epoch's interactions and reset
        interactions_by_epoch.append(list(current_epoch_interactions))
        current_epoch_interactions.clear()

        # Compute incremental horizon eval on all epochs so far
        result = evaluator.evaluate(interactions_by_epoch)
        horizon_snapshots.append({
            "epoch": epoch_metrics.epoch,
            "horizon_amplification": result.horizon_amplification_ratio,
            "emergent_coherence": result.emergent_coherence,
            "adverse_selection_drift": result.adverse_selection_drift,
            "drift_direction": result.drift_direction,
            "variance_dominance_index": result.variance_dominance_index,
            "hot_mess_epochs": result.hot_mess_epochs,
            "effective_system_horizon": result.effective_system_horizon,
            "chain_depth_mean": result.chain_depth_mean,
            "chain_depth_max": result.chain_depth_max,
            "harm_acceleration": result.harm_acceleration,
        })

    orchestrator.on_epoch_end(_on_epoch_end)

    # --- Show agents ---
    print(f"Agents ({len(orchestrator.get_all_agents())}):")
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        print(
            f"  {agent.agent_id} ({agent.agent_type.value}): "
            f"resources={state.resources:.0f}"
        )
    print()

    # --- Run simulation ---
    print(
        f"Running: {config.n_epochs} epochs x {config.steps_per_epoch} steps"
    )
    print("-" * 78)

    metrics_history = orchestrator.run()

    # --- Epoch-by-epoch table ---
    print()
    print("Epoch-by-Epoch Horizon Evaluation Results:")
    print("-" * 78)
    header = (
        f"{'Ep':<4} {'Intx':<6} {'Acc':<5} {'Toxic':<7} "
        f"{'H_amp':<7} {'Coher':<7} {'Drift':<7} {'VDI':<7} "
        f"{'Chain':<6} {'HarmAcc':<8}"
    )
    print(header)
    print("-" * 78)

    for m, snap in zip(metrics_history, horizon_snapshots, strict=False):
        print(
            f"{m.epoch:<4} "
            f"{m.total_interactions:<6} "
            f"{m.accepted_interactions:<5} "
            f"{m.toxicity_rate:<7.4f} "
            f"{snap['horizon_amplification']:<7.3f} "
            f"{snap['emergent_coherence']:<7.3f} "
            f"{snap['adverse_selection_drift']:<7.3f} "
            f"{snap['variance_dominance_index']:<7.3f} "
            f"{snap['chain_depth_mean']:<6.2f} "
            f"{snap['harm_acceleration']:<8.4f}"
        )

    print("-" * 78)
    print()

    # --- Final system-level horizon evaluation ---
    final_result = evaluator.evaluate(interactions_by_epoch)

    print("System-Level Horizon Evaluation (Final):")
    print("-" * 78)
    print(f"  Effective system horizon:    {final_result.effective_system_horizon:.3f}")
    print(f"  Horizon amplification ratio: {final_result.horizon_amplification_ratio:.3f}")
    print(f"  Emergent coherence:          {final_result.emergent_coherence:.3f}")
    print(f"  Adverse selection drift:     {final_result.adverse_selection_drift:.4f} ({final_result.drift_direction})")
    print(f"  Variance dominance index:    {final_result.variance_dominance_index:.3f}")
    print(f"  Hot-mess epochs:             {final_result.hot_mess_epochs} / {final_result.total_epochs}")
    print(f"  Chain depth (mean/max):      {final_result.chain_depth_mean:.2f} / {final_result.chain_depth_max}")
    print(f"  Harm acceleration:           {final_result.harm_acceleration:.4f}")
    print()

    # --- Reliability curves by capability profile ---
    rng = np.random.default_rng(config.seed)
    profile_metrics = _simulate_reliability_curves(orchestrator, rng)

    print("Reliability Curves by Agent Capability Profile:")
    print("-" * 78)

    # Header
    tier_header = "  " + f"{'Profile':<12}"
    for h in HORIZON_TIERS:
        tier_header += f"{TIER_LABELS[h]:>8}"
    tier_header += f"{'  EffH(80%)':>12}{'  H_gap':>8}"
    print(tier_header)
    print("  " + "-" * (len(tier_header) - 2))

    reliability_curves_export: dict[str, dict] = {}

    for profile_name in ["frontier", "standard", "distilled", "edge"]:
        metrics = profile_metrics[profile_name]
        curve = metrics.reliability_curve()
        eff_h = metrics.effective_horizon(0.8)
        h_gap = metrics.horizon_gap()

        row = f"  {profile_name:<12}"
        for h in HORIZON_TIERS:
            rel = curve.get(h, 0.0)
            row += f"{rel:>8.3f}"
        eff_h_str = f"{eff_h}min" if eff_h is not None else "None"
        row += f"{eff_h_str:>12}{h_gap:>8.3f}"
        print(row)

        reliability_curves_export[profile_name] = metrics.to_dict()

    print()

    # --- Theoretical reliability predictions ---
    print("Theoretical Reliability (from AgentCapabilityProfile model):")
    print("-" * 78)
    theory_header = f"  {'Profile':<12}"
    for h in HORIZON_TIERS:
        theory_header += f"{TIER_LABELS[h]:>8}"
    print(theory_header)
    print("  " + "-" * (len(theory_header) - 2))

    for profile_name in ["frontier", "standard", "distilled", "edge"]:
        profile = CAPABILITY_PROFILES[profile_name]
        row = f"  {profile_name:<12}"
        for h in HORIZON_TIERS:
            row += f"{profile.reliability_at_horizon(h):>8.3f}"
        print(row)
    print()

    # --- Summary statistics ---
    print("Summary Statistics:")
    print("-" * 78)
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    print(f"  Total interactions: {total_interactions}")
    print(f"  Accepted: {total_accepted}")
    print(f"  Avg toxicity: {avg_toxicity:.4f}")
    print(f"  Total welfare: {total_welfare:.2f}")
    print()

    # Horizon amplification trajectory
    if horizon_snapshots:
        amps = [s["horizon_amplification"] for s in horizon_snapshots]
        coherences = [s["emergent_coherence"] for s in horizon_snapshots]
        drifts = [s["adverse_selection_drift"] for s in horizon_snapshots]
        vdis = [s["variance_dominance_index"] for s in horizon_snapshots]

        print("  Horizon amplification trajectory:")
        print(f"    Start: {amps[0]:.3f}, End: {amps[-1]:.3f}, "
              f"Mean: {sum(amps)/len(amps):.3f}")
        print("  Coherence trajectory:")
        print(f"    Start: {coherences[0]:.3f}, End: {coherences[-1]:.3f}, "
              f"Mean: {sum(coherences)/len(coherences):.3f}")
        print("  Drift trajectory:")
        print(f"    Start: {drifts[0]:.4f}, End: {drifts[-1]:.4f}, "
              f"Direction: {horizon_snapshots[-1]['drift_direction']}")
        print("  VDI trajectory:")
        print(f"    Start: {vdis[0]:.3f}, End: {vdis[-1]:.3f}, "
              f"Mean: {sum(vdis)/len(vdis):.3f}")
        print()

    # --- Final agent states ---
    print("Final Agent States:")
    print("-" * 78)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        print(
            f"  {agent.agent_id} ({agent.agent_type.value}): "
            f"rep={state.reputation:.2f}, "
            f"res={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}"
            f"{status}"
        )
    print()

    # --- Export ---
    history = aggregator.end_simulation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_horizon_eval_seed{config.seed}"

    # 1. Standard history export
    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)

    # 2. Horizon evaluation results
    horizon_path = export_dir / "horizon_eval_results.json"
    horizon_path.parent.mkdir(parents=True, exist_ok=True)
    with open(horizon_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": config.seed,
                "gov_pressure": _governance_pressure(scenario),
                "horizon_eval_config": {
                    "agent_horizon_steps": he_config.agent_horizon_steps,
                    "discount_factor": he_config.discount_factor,
                    "coherence_lag_max": he_config.coherence_lag_max,
                    "drift_window": he_config.drift_window,
                    "variance_dominance_threshold": he_config.variance_dominance_threshold,
                },
                "final_result": final_result.to_dict(),
                "epoch_snapshots": horizon_snapshots,
                "reliability_curves": reliability_curves_export,
            },
            f,
            indent=2,
        )

    print(f"Exported run history: {export_path}")
    print(f"Exported horizon eval results: {horizon_path}")
    print(f"  {len(horizon_snapshots)} epoch snapshots")
    print(f"  {len(reliability_curves_export)} capability profiles")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

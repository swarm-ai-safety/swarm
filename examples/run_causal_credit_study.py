#!/usr/bin/env python
"""
Causal Credit Propagation Study.

Research question:
    When adversarial agents trigger cascading harm through interaction chains,
    does backward credit propagation correctly identify the root-cause agents —
    and how does governance modulate cascade depth?

Causal linking strategy:
    Each completed interaction is linked to both participants' most recent prior
    interactions (behavioral causal chain). This captures "agent A's previous
    interaction influenced this one" without needing true causal inference.

Swept parameters (via scenario YAML):
    - Agent mix (honest / adversarial / opportunistic)
    - Governance (tax rate, audit probability, circuit breaker)

Usage:
    python examples/run_causal_credit_study.py scenarios/causal_credit_sweep.yaml
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.causal_credit import CausalCreditEngine, CausalSnapshot
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario


def _agent_type_map(orchestrator) -> dict[str, AgentType]:
    """Build agent_id -> AgentType lookup."""
    return {a.agent_id: a.agent_type for a in orchestrator.get_all_agents()}


def _governance_pressure(scenario) -> float:
    """Derive governance pressure from scenario config."""
    gov = scenario.orchestrator_config.governance_config
    if gov is None:
        return 0.0
    tax = gov.transaction_tax_rate
    audit = gov.audit_probability if gov.audit_enabled else 0.0
    return tax + audit


def _snapshot_to_dict(snap: CausalSnapshot) -> dict:
    """Serialize a CausalSnapshot to a JSON-friendly dict."""
    return {
        "step": snap.step,
        "dag_depth": snap.dag_depth,
        "dag_width": snap.dag_width,
        "total_interactions": snap.total_interactions,
        "root_count": snap.root_count,
        "leaf_count": snap.leaf_count,
        "credit_by_agent": snap.credit_by_agent,
        "top_blame": snap.top_blame,
        "top_credit": snap.top_credit,
        "cascade_depth_histogram": {
            str(k): v for k, v in snap.cascade_depth_histogram.items()
        },
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_causal_credit_study.py <scenario.yaml>")
        print("\nSuggested: scenarios/causal_credit_sweep.yaml")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Causal Credit Propagation Study")
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
    agent_types = _agent_type_map(orchestrator)

    # CausalCreditEngine parameters
    decay_rate = 0.5
    max_depth = 10

    # --- Wire MetricsAggregator (standard pattern) ---
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    # Causal tracking state
    agent_last_ix: dict[str, str] = {}  # agent_id -> last interaction_id
    all_interactions = []  # cumulative (for DAG)
    epoch_interactions = []  # current epoch only (for cascade risk)
    snapshots: list[CausalSnapshot] = []
    epoch_cascade_risks: list[list[tuple[str, str, float]]] = []

    # Per-agent credit time series: agent_id -> list of (epoch, credit)
    credit_timeseries: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        # Wire causal parents: link to each participant's last interaction
        parents = []
        if interaction.initiator in agent_last_ix:
            parents.append(agent_last_ix[interaction.initiator])
        if interaction.counterparty in agent_last_ix:
            parents.append(agent_last_ix[interaction.counterparty])
        interaction.causal_parents = parents

        # Update tracking
        agent_last_ix[interaction.initiator] = interaction.interaction_id
        agent_last_ix[interaction.counterparty] = interaction.interaction_id

        # Track for DAG computation
        all_interactions.append(interaction)
        epoch_interactions.append(interaction)

        # Standard aggregator
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

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

        # Compute causal snapshot over cumulative DAG
        engine = CausalCreditEngine(decay=decay_rate, max_depth=max_depth)
        snap = engine.compute_snapshot(
            step=epoch_metrics.epoch,
            interactions=list(all_interactions),
            signal="p",
        )
        snapshots.append(snap)

        # Record per-agent credit for time series
        for agent_id, credit in snap.credit_by_agent.items():
            credit_timeseries[agent_id].append(
                (epoch_metrics.epoch, credit)
            )

        # Per-epoch cascade risk: find highest-risk root interactions
        engine.build_dag(list(all_interactions))
        risks = []
        for ix in epoch_interactions:
            risk = engine.cascade_risk(ix.interaction_id, p_threshold=0.3)
            if risk > 0:
                risks.append((ix.interaction_id, ix.initiator, risk))
        epoch_cascade_risks.append(risks)

        epoch_interactions.clear()

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
    print(f"Governance pressure: {_governance_pressure(scenario):.3f}")
    print(f"Causal engine: decay={decay_rate}, max_depth={max_depth}")
    print()

    # --- Run simulation ---
    print(
        f"Running: {config.n_epochs} epochs x {config.steps_per_epoch} steps"
    )
    print("-" * 78)

    metrics_history = orchestrator.run()

    # --- Epoch-by-epoch table ---
    print()
    print("Epoch-by-Epoch Causal Credit Results:")
    print("-" * 78)
    header = (
        f"{'Ep':<4} {'Intx':<6} {'DAG_D':<6} {'DAG_W':<6} "
        f"{'Roots':<6} {'Leaves':<7} "
        f"{'TopBlame':<22} {'TopCredit':<22} {'CascRisk':<9}"
    )
    print(header)
    print("-" * 78)

    for epoch_idx, (m, snap) in enumerate(
        zip(metrics_history, snapshots, strict=False)
    ):
        # Top blame/credit: show agent with most negative/positive credit
        if snap.top_blame:
            blame_agent, blame_val = snap.top_blame[0]
            blame_str = f"{blame_agent}:{blame_val:+.2f}"
        else:
            blame_str = "-"

        if snap.top_credit:
            credit_agent, credit_val = snap.top_credit[0]
            credit_str = f"{credit_agent}:{credit_val:+.2f}"
        else:
            credit_str = "-"

        # Max cascade risk this epoch
        if epoch_idx < len(epoch_cascade_risks) and epoch_cascade_risks[epoch_idx]:
            max_risk = max(r for _, _, r in epoch_cascade_risks[epoch_idx])
        else:
            max_risk = 0.0

        print(
            f"{m.epoch:<4} "
            f"{m.total_interactions:<6} "
            f"{snap.dag_depth:<6} "
            f"{snap.dag_width:<6} "
            f"{snap.root_count:<6} "
            f"{snap.leaf_count:<7} "
            f"{blame_str:<22} "
            f"{credit_str:<22} "
            f"{max_risk:<9.2f}"
        )

    print("-" * 78)
    print()

    # --- Summary Statistics ---
    print("Summary Statistics:")
    print("-" * 78)

    # Standard metrics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    print(f"  Total interactions: {total_interactions}")
    print(f"  Accepted: {total_accepted}")
    print(f"  Avg toxicity: {avg_toxicity:.4f}")
    print()

    # DAG growth
    if snapshots:
        depths = [s.dag_depth for s in snapshots]
        print("  Cascade depth growth:")
        print(f"    Start: {depths[0]}, End: {depths[-1]}")
        if len(depths) > 1:
            growth = (depths[-1] - depths[0]) / (len(depths) - 1)
            print(f"    Avg growth/epoch: {growth:.2f}")
        print()

    # Agent-type credit distribution
    if snapshots and snapshots[-1].credit_by_agent:
        last_credit = snapshots[-1].credit_by_agent
        type_credits: dict[str, list[float]] = defaultdict(list)
        for agent_id, credit in last_credit.items():
            atype = agent_types.get(agent_id, AgentType.HONEST)
            type_credits[atype.value].append(credit)

        print("  Final credit by agent type:")
        for type_name, credits in sorted(type_credits.items()):
            mean_credit = sum(credits) / len(credits)
            print(
                f"    {type_name}: mean={mean_credit:+.2f} "
                f"(n={len(credits)}, "
                f"min={min(credits):+.2f}, max={max(credits):+.2f})"
            )
        print()

    # Credit trajectory
    if snapshots:
        print("  Credit trajectory (epoch 0 vs final):")
        first_credit = snapshots[0].credit_by_agent
        last_credit = snapshots[-1].credit_by_agent
        all_agents = sorted(
            set(first_credit.keys()) | set(last_credit.keys())
        )
        for agent_id in all_agents:
            c0 = first_credit.get(agent_id, 0.0)
            cf = last_credit.get(agent_id, 0.0)
            atype = agent_types.get(agent_id, AgentType.HONEST).value
            print(f"    {agent_id} ({atype}): {c0:+.2f} -> {cf:+.2f}")
        print()

    # Governance effect on cascade depth
    gov_pressure = _governance_pressure(scenario)
    if snapshots:
        final_depth = snapshots[-1].dag_depth
        print(f"  Governance pressure: {gov_pressure:.3f}")
        print(f"  Final DAG depth: {final_depth}")
        print()

    # --- Final agent states ---
    print("Final Agent States:")
    print("-" * 78)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        final_credit = (
            snapshots[-1].credit_by_agent.get(agent.agent_id, 0.0)
            if snapshots
            else 0.0
        )
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        print(
            f"  {agent.agent_id} ({agent.agent_type.value}): "
            f"rep={state.reputation:.2f}, "
            f"res={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}, "
            f"credit={final_credit:+.2f}"
            f"{status}"
        )
    print()

    # --- Export ---
    history = aggregator.end_simulation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_causal_credit_seed{config.seed}"

    # 1. Standard history export
    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)

    # 2. Causal snapshots
    snapshots_path = export_dir / "causal_snapshots.json"
    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshots_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": config.seed,
                "decay": decay_rate,
                "max_depth": max_depth,
                "gov_pressure": gov_pressure,
                "snapshots": [_snapshot_to_dict(s) for s in snapshots],
            },
            f,
            indent=2,
        )

    # 3. Agent credit CSV time series
    csv_path = export_dir / "agent_credit.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_agent_ids = sorted(credit_timeseries.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + all_agent_ids)
        if snapshots:
            for epoch_idx in range(len(snapshots)):
                row = [epoch_idx]
                for agent_id in all_agent_ids:
                    # Find credit at this epoch
                    entries = credit_timeseries[agent_id]
                    credit = 0.0
                    for ep, c in entries:
                        if ep == epoch_idx:
                            credit = c
                            break
                    row.append(f"{credit:.4f}")
                writer.writerow(row)

    print(f"Exported run history: {export_path}")
    print(f"Exported causal snapshots: {snapshots_path}")
    print(f"Exported agent credit CSV: {csv_path}")
    print(f"  {len(snapshots)} epoch snapshots")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

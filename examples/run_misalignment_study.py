#!/usr/bin/env python
"""
Run a misalignment sweep study using the MisalignmentModule.

Tracks how preference-space misalignment evolves across epochs and how
governance modulates effective misalignment (Kierans et al. 2406.04231).

Usage:
    python examples/run_misalignment_study.py scenarios/misalignment_sweep.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.misalignment import IssueSpace, MisalignmentModule
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Preference profiles per agent type
# Issues: safety, efficiency, fairness
# ---------------------------------------------------------------------------

ISSUE_NAMES = ["safety", "efficiency", "fairness"]

PREFERENCE_MAP: dict[AgentType, dict] = {
    AgentType.HONEST: {
        "prefs": [+0.8, +0.3, +0.7],
        "salience": [0.4, 0.2, 0.4],
    },
    AgentType.OPPORTUNISTIC: {
        "prefs": [-0.2, +0.9, -0.3],
        "salience": [0.1, 0.7, 0.2],
    },
    AgentType.ADVERSARIAL: {
        "prefs": [-0.8, +0.5, -0.7],
        "salience": [0.3, 0.5, 0.2],
    },
    AgentType.DECEPTIVE: {
        "prefs": [-0.5, +0.7, -0.4],
        "salience": [0.2, 0.5, 0.3],
    },
    AgentType.CAUTIOUS: {
        "prefs": [+0.6, -0.1, +0.5],
        "salience": [0.5, 0.1, 0.4],
    },
}


def _build_graph(orchestrator) -> dict[str, list[str]]:
    """Build adjacency list from orchestrator network."""
    graph: dict[str, list[str]] = {}
    if orchestrator.network is None:
        # Fallback: fully connected
        all_ids = [a.agent_id for a in orchestrator.get_all_agents()]
        for aid in all_ids:
            graph[aid] = [x for x in all_ids if x != aid]
    else:
        for agent in orchestrator.get_all_agents():
            graph[agent.agent_id] = orchestrator.network.neighbors(agent.agent_id)
    return graph


def _governance_pressure(scenario) -> float:
    """Derive uniform governance pressure from scenario governance config."""
    gov = scenario.orchestrator_config.governance_config
    if gov is None:
        return 0.0
    # Combine tax rate and audit probability as governance pressure proxy
    tax = gov.transaction_tax_rate
    audit = gov.audit_probability if gov.audit_enabled else 0.0
    return tax + audit


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_misalignment_study.py <scenario.yaml>")
        print("\nSuggested: scenarios/misalignment_sweep.yaml")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 72)
    print("Misalignment Sweep Study (Kierans et al. 2406.04231)")
    print("=" * 72)
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

    # --- Set up MisalignmentModule ---
    issue_space = IssueSpace(issues=ISSUE_NAMES)
    module = MisalignmentModule(issue_space=issue_space, gov_lambda=1.0)

    # Register agents with preference profiles
    registered = 0
    for agent in orchestrator.get_all_agents():
        profile = PREFERENCE_MAP.get(agent.agent_type)
        if profile is None:
            # Unknown type: neutral preferences
            profile = {
                "prefs": [0.0] * len(ISSUE_NAMES),
                "salience": [1.0 / len(ISSUE_NAMES)] * len(ISSUE_NAMES),
            }
        module.register_agent(
            agent_id=agent.agent_id,
            prefs=profile["prefs"],
            salience=profile["salience"],
        )
        registered += 1

    print(f"Registered {registered} agents with MisalignmentModule")
    print(f"  Issues: {ISSUE_NAMES}")
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

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

    orchestrator.on_interaction_complete(_on_interaction)

    # Collect misalignment snapshots
    snapshots = []
    gov_pressure = _governance_pressure(scenario)

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

        # Compute misalignment snapshot
        graph = _build_graph(orchestrator)
        snap = module.compute_snapshot(
            step=epoch_metrics.epoch,
            graph=graph,
            uniform_pressure=gov_pressure,
        )
        snapshots.append(snap)

    orchestrator.on_epoch_end(_on_epoch_end)

    # --- Show agents ---
    print(f"Agents ({len(orchestrator.get_all_agents())}):")
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        pref = PREFERENCE_MAP.get(agent.agent_type, {}).get("prefs", [])
        pref_str = ", ".join(f"{v:+.1f}" for v in pref) if pref else "neutral"
        print(
            f"  {agent.agent_id} ({agent.agent_type.value}): "
            f"prefs=[{pref_str}], resources={state.resources:.0f}"
        )
    print()

    # --- Run simulation ---
    print(
        f"Running: {config.n_epochs} epochs x {config.steps_per_epoch} steps"
    )
    print("-" * 72)

    metrics_history = orchestrator.run()

    # --- Epoch-by-epoch table ---
    print()
    print("Epoch-by-Epoch Results:")
    print("-" * 72)
    header = (
        f"{'Ep':<4} {'Intx':<6} {'Acc':<5} {'Toxic':<8} {'Welfare':<9} "
        f"{'M_pref':<8} {'M_eff':<8} {'Polar':<8} {'Frag':<8} {'Alerts'}"
    )
    print(header)
    print("-" * 72)

    for m, snap in zip(metrics_history, snapshots, strict=False):
        alert_str = "; ".join(snap.alerts) if snap.alerts else "-"
        print(
            f"{m.epoch:<4} "
            f"{m.total_interactions:<6} "
            f"{m.accepted_interactions:<5} "
            f"{m.toxicity_rate:<8.4f} "
            f"{m.total_welfare:<9.2f} "
            f"{snap.m_pref_global:<8.4f} "
            f"{snap.m_eff_global:<8.4f} "
            f"{snap.polarization:<8.4f} "
            f"{snap.fragmentation:<8.4f} "
            f"{alert_str}"
        )

    print("-" * 72)
    print()

    # --- Issue contributions ---
    if snapshots:
        last = snapshots[-1]
        print("Final Issue Contributions:")
        for issue, contrib in last.issue_contributions.items():
            print(f"  {issue}: {contrib:.4f}")
        print()

    # --- Summary statistics ---
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    print("Standard Metrics Summary:")
    print(f"  Total Interactions: {total_interactions}")
    print(f"  Accepted: {total_accepted}")
    print(f"  Avg Toxicity: {avg_toxicity:.4f}")
    print(f"  Total Welfare: {total_welfare:.2f}")
    print()

    if snapshots:
        m_prefs = [s.m_pref_global for s in snapshots]
        m_effs = [s.m_eff_global for s in snapshots]
        print("Misalignment Trajectory:")
        print(f"  M_pref: {m_prefs[0]:.4f} -> {m_prefs[-1]:.4f} "
              f"(mean={sum(m_prefs)/len(m_prefs):.4f})")
        print(f"  M_eff:  {m_effs[0]:.4f} -> {m_effs[-1]:.4f} "
              f"(mean={sum(m_effs)/len(m_effs):.4f})")
        if m_prefs[-1] > 0:
            reduction = 1.0 - m_effs[-1] / m_prefs[-1]
            print(f"  Governance reduction: {reduction:.1%}")
        print()

        # Correlation: toxicity vs misalignment
        toxicities = [m.toxicity_rate for m in metrics_history]
        n = min(len(toxicities), len(m_prefs))
        if n > 1:
            mean_t = sum(toxicities[:n]) / n
            mean_m = sum(m_prefs[:n]) / n
            cov = sum(
                (toxicities[i] - mean_t) * (m_prefs[i] - mean_m)
                for i in range(n)
            ) / n
            var_t = sum((t - mean_t) ** 2 for t in toxicities[:n]) / n
            var_m = sum((m - mean_m) ** 2 for m in m_prefs[:n]) / n
            if var_t > 0 and var_m > 0:
                corr = cov / (var_t**0.5 * var_m**0.5)
                print(f"  Toxicity-Misalignment correlation: {corr:.4f}")
            else:
                print("  Toxicity-Misalignment correlation: N/A (no variance)")
        print()

    # --- Final agent states ---
    print("Final Agent States:")
    print("-" * 72)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        local_m = snapshots[-1].local.get(agent.agent_id, 0.0) if snapshots else 0.0
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        print(
            f"  {agent.agent_id}: "
            f"rep={state.reputation:.2f}, "
            f"res={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}, "
            f"M_local={local_m:.4f}"
            f"{status}"
        )
    print()

    # --- Export ---
    history = aggregator.end_simulation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_misalignment_sweep_seed{config.seed}"
    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)

    # Export misalignment snapshots alongside
    misalignment_path = export_dir / "misalignment_snapshots.json"
    misalignment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(misalignment_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": config.seed,
                "issues": ISSUE_NAMES,
                "gov_pressure": gov_pressure,
                "snapshots": [s.to_dict() for s in snapshots],
            },
            f,
            indent=2,
        )

    print(f"Exported run history: {export_path}")
    print(f"Exported misalignment snapshots: {misalignment_path}")
    print(f"  {len(snapshots)} epoch snapshots")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

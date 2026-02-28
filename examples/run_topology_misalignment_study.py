#!/usr/bin/env python
"""
Network Topology x Misalignment Study.

Research question:
    How does network topology affect polarization, fragmentation, and local
    misalignment variance?  Compare complete, ring, small_world (k=4, p=0.3),
    star, and scale_free topologies on the same 10-agent population from the
    misalignment_sweep scenario.

Metrics per topology:
    M_pref, M_eff, polarization, fragmentation, local misalignment variance.

Usage:
    python examples/run_topology_misalignment_study.py [scenarios/misalignment_sweep.yaml]
"""

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.env.network import NetworkConfig, NetworkTopology
from swarm.metrics.misalignment import IssueSpace, MisalignmentModule
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Preference profiles per agent type (same as run_misalignment_study.py)
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

# ---------------------------------------------------------------------------
# Topology definitions
# ---------------------------------------------------------------------------

TOPOLOGIES: dict[str, NetworkConfig] = {
    "complete": NetworkConfig(
        topology=NetworkTopology.COMPLETE,
    ),
    "ring": NetworkConfig(
        topology=NetworkTopology.RING,
    ),
    "small_world": NetworkConfig(
        topology=NetworkTopology.SMALL_WORLD,
        k_neighbors=4,
        rewire_probability=0.3,
    ),
    "star": NetworkConfig(
        topology=NetworkTopology.STAR,
    ),
    "scale_free": NetworkConfig(
        topology=NetworkTopology.SCALE_FREE,
        m_edges=2,
    ),
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
    tax = gov.transaction_tax_rate
    audit = gov.audit_probability if gov.audit_enabled else 0.0
    return tax + audit


def _run_topology(scenario, topology_name, network_config, gov_pressure):
    """Run simulation with a specific network topology.

    Returns:
        (metrics_history, snapshots, orchestrator, aggregator)
    """
    # Modify scenario network config in-place (we deepcopy before calling)
    scenario.orchestrator_config.network_config = network_config
    config = scenario.orchestrator_config

    # Build orchestrator
    orchestrator = build_orchestrator(scenario)

    # Set up MisalignmentModule
    issue_space = IssueSpace(issues=ISSUE_NAMES)
    module = MisalignmentModule(issue_space=issue_space, gov_lambda=1.0)

    for agent in orchestrator.get_all_agents():
        profile = PREFERENCE_MAP.get(agent.agent_type)
        if profile is None:
            profile = {
                "prefs": [0.0] * len(ISSUE_NAMES),
                "salience": [1.0 / len(ISSUE_NAMES)] * len(ISSUE_NAMES),
            }
        module.register_agent(
            agent_id=agent.agent_id,
            prefs=profile["prefs"],
            salience=profile["salience"],
        )

    # Wire MetricsAggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=f"{scenario.scenario_id}_{topology_name}",
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

    snapshots = []

    def _on_epoch_end(epoch_metrics):
        agent_states = orchestrator.state.agents
        frozen = orchestrator.state.frozen_agents
        quarantined = getattr(orchestrator.state, "quarantined_agents", set())
        aggregator.finalize_epoch(
            epoch=epoch_metrics.epoch,
            agent_states=agent_states,
            frozen_agents=frozen,
            quarantined_agents=quarantined,
        )

        graph = _build_graph(orchestrator)
        snap = module.compute_snapshot(
            step=epoch_metrics.epoch,
            graph=graph,
            uniform_pressure=gov_pressure,
        )
        snapshots.append(snap)

    orchestrator.on_epoch_end(_on_epoch_end)

    # Run
    metrics_history = orchestrator.run()

    return metrics_history, snapshots, orchestrator, aggregator


def main():
    scenario_path = Path(
        sys.argv[1] if len(sys.argv) >= 2 else "scenarios/misalignment_sweep.yaml"
    )
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Network Topology x Misalignment Study")
    print("=" * 78)
    print()

    # Load base scenario
    print(f"Loading base scenario: {scenario_path}")
    base_scenario = load_scenario(scenario_path)
    print(f"  ID: {base_scenario.scenario_id}")
    print(f"  Description: {base_scenario.description}")
    print(f"  Governance pressure: {_governance_pressure(base_scenario):.3f}")
    print()

    gov_pressure = _governance_pressure(base_scenario)

    # ---------------------------------------------------------------------------
    # Sweep topologies
    # ---------------------------------------------------------------------------

    topology_results: dict[str, dict] = {}

    for topo_name, net_config in TOPOLOGIES.items():
        print(f"--- Topology: {topo_name} ---")

        # Deep copy so each run starts fresh
        scenario = copy.deepcopy(base_scenario)

        metrics_history, snapshots, orchestrator, aggregator = _run_topology(
            scenario, topo_name, net_config, gov_pressure
        )

        # --- Epoch-by-epoch table ---
        print()
        print(f"  Epoch-by-Epoch ({topo_name}):")
        print(f"  {'Ep':<4} {'Intx':<6} {'Toxic':<8} "
              f"{'M_pref':<8} {'M_eff':<8} {'Polar':<8} {'Frag':<8}")
        print("  " + "-" * 58)

        for m, snap in zip(metrics_history, snapshots, strict=False):
            print(
                f"  {m.epoch:<4} "
                f"{m.total_interactions:<6} "
                f"{m.toxicity_rate:<8.4f} "
                f"{snap.m_pref_global:<8.4f} "
                f"{snap.m_eff_global:<8.4f} "
                f"{snap.polarization:<8.4f} "
                f"{snap.fragmentation:<8.4f}"
            )

        print()

        # Compute summary for comparison table
        if snapshots:
            final = snapshots[-1]
            avg_m_pref = sum(s.m_pref_global for s in snapshots) / len(snapshots)
            avg_m_eff = sum(s.m_eff_global for s in snapshots) / len(snapshots)
            avg_polar = sum(s.polarization for s in snapshots) / len(snapshots)
            avg_frag = sum(s.fragmentation for s in snapshots) / len(snapshots)

            # Local misalignment variance
            local_vals = list(final.local.values()) if final.local else [0.0]
            mean_local = sum(local_vals) / len(local_vals)
            local_var = (
                sum((v - mean_local) ** 2 for v in local_vals) / len(local_vals)
            )

            topology_results[topo_name] = {
                "final_m_pref": final.m_pref_global,
                "final_m_eff": final.m_eff_global,
                "final_polarization": final.polarization,
                "final_fragmentation": final.fragmentation,
                "avg_m_pref": avg_m_pref,
                "avg_m_eff": avg_m_eff,
                "avg_polarization": avg_polar,
                "avg_fragmentation": avg_frag,
                "local_misalignment_variance": local_var,
                "local_misalignment_mean": mean_local,
                "local_scores": final.local,
                "snapshots": [s.to_dict() for s in snapshots],
            }

        # End aggregator
        aggregator.end_simulation()

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("Topology Comparison (Final Epoch)")
    print("=" * 78)
    header = (
        f"{'Topology':<14} "
        f"{'M_pref':<8} {'M_eff':<8} "
        f"{'Polar':<8} {'Frag':<8} "
        f"{'Local_Var':<10} {'Local_Mean':<10}"
    )
    print(header)
    print("-" * 78)

    for topo_name, res in topology_results.items():
        print(
            f"{topo_name:<14} "
            f"{res['final_m_pref']:<8.4f} "
            f"{res['final_m_eff']:<8.4f} "
            f"{res['final_polarization']:<8.4f} "
            f"{res['final_fragmentation']:<8.4f} "
            f"{res['local_misalignment_variance']:<10.6f} "
            f"{res['local_misalignment_mean']:<10.4f}"
        )

    print("-" * 78)
    print()

    # Averages table
    print("Topology Comparison (Epoch Averages)")
    print("-" * 78)
    header = (
        f"{'Topology':<14} "
        f"{'Avg M_pref':<11} {'Avg M_eff':<11} "
        f"{'Avg Polar':<11} {'Avg Frag':<11}"
    )
    print(header)
    print("-" * 78)

    for topo_name, res in topology_results.items():
        print(
            f"{topo_name:<14} "
            f"{res['avg_m_pref']:<11.4f} "
            f"{res['avg_m_eff']:<11.4f} "
            f"{res['avg_polarization']:<11.4f} "
            f"{res['avg_fragmentation']:<11.4f}"
        )

    print("-" * 78)
    print()

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------

    print("Summary Statistics:")
    print("-" * 78)

    # Rank topologies by local misalignment variance
    ranked = sorted(
        topology_results.items(),
        key=lambda kv: kv[1]["local_misalignment_variance"],
    )
    print("  Local misalignment variance ranking (low -> high):")
    for rank, (topo, res) in enumerate(ranked, 1):
        print(f"    {rank}. {topo}: {res['local_misalignment_variance']:.6f}")
    print()

    # Rank by polarization
    ranked_polar = sorted(
        topology_results.items(),
        key=lambda kv: kv[1]["final_polarization"],
    )
    print("  Polarization ranking (low -> high):")
    for rank, (topo, res) in enumerate(ranked_polar, 1):
        print(f"    {rank}. {topo}: {res['final_polarization']:.4f}")
    print()

    # Governance reduction per topology
    print("  Governance reduction (1 - M_eff/M_pref) per topology:")
    for topo, res in topology_results.items():
        if res["final_m_pref"] > 0:
            reduction = 1.0 - res["final_m_eff"] / res["final_m_pref"]
            print(f"    {topo}: {reduction:.1%}")
        else:
            print(f"    {topo}: N/A (M_pref=0)")
    print()

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

    config = base_scenario.orchestrator_config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_topology_misalignment_seed{config.seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export topology comparison JSON
    comparison_path = export_dir / "topology_comparison.json"
    # Strip snapshots for the summary (they're large); keep them in per-topology files
    summary = {}
    for topo_name, res in topology_results.items():
        summary[topo_name] = {k: v for k, v in res.items()
                              if k not in ("snapshots", "local_scores")}

    with open(comparison_path, "w") as f:
        json.dump(
            {
                "scenario_id": base_scenario.scenario_id,
                "seed": config.seed,
                "issues": ISSUE_NAMES,
                "gov_pressure": gov_pressure,
                "topologies": list(TOPOLOGIES.keys()),
                "comparison": summary,
            },
            f,
            indent=2,
        )

    # Export per-topology snapshots
    for topo_name, res in topology_results.items():
        topo_path = export_dir / f"snapshots_{topo_name}.json"
        with open(topo_path, "w") as f:
            json.dump(
                {
                    "topology": topo_name,
                    "scenario_id": base_scenario.scenario_id,
                    "seed": config.seed,
                    "snapshots": res.get("snapshots", []),
                },
                f,
                indent=2,
            )

    print(f"Exported topology comparison: {comparison_path}")
    print(f"Exported per-topology snapshots: {export_dir}/snapshots_*.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

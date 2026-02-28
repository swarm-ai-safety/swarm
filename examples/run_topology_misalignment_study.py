#!/usr/bin/env python
"""
Network Topology x Misalignment Study.

Compares how network topology shapes local misalignment variation,
polarization, and fragmentation across the same agent population.

Topologies: complete, ring, small_world, star, scale_free

Usage:
    python examples/run_topology_misalignment_study.py scenarios/misalignment_sweep.yaml
"""

import copy
import json
import math
import sys
from dataclasses import dataclass, field
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
# Topology configurations
# ---------------------------------------------------------------------------

TOPOLOGIES: dict[str, NetworkConfig] = {
    "complete": NetworkConfig(
        topology=NetworkTopology.COMPLETE,
        dynamic=False,
    ),
    "ring": NetworkConfig(
        topology=NetworkTopology.RING,
        dynamic=False,
    ),
    "small_world": NetworkConfig(
        topology=NetworkTopology.SMALL_WORLD,
        k_neighbors=4,
        rewire_probability=0.3,
        dynamic=False,
    ),
    "star": NetworkConfig(
        topology=NetworkTopology.STAR,
        dynamic=False,
    ),
    "scale_free": NetworkConfig(
        topology=NetworkTopology.SCALE_FREE,
        m_edges=2,
        dynamic=False,
    ),
}

# ---------------------------------------------------------------------------
# Preference profiles
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
    graph: dict[str, list[str]] = {}
    if orchestrator.network is None:
        all_ids = [a.agent_id for a in orchestrator.get_all_agents()]
        for aid in all_ids:
            graph[aid] = [x for x in all_ids if x != aid]
    else:
        for agent in orchestrator.get_all_agents():
            graph[agent.agent_id] = orchestrator.network.neighbors(agent.agent_id)
    return graph


def _governance_pressure(scenario) -> float:
    gov = scenario.orchestrator_config.governance_config
    if gov is None:
        return 0.0
    tax = gov.transaction_tax_rate
    audit = gov.audit_probability if gov.audit_enabled else 0.0
    return tax + audit


@dataclass
class TopologyResult:
    """Results for a single topology."""

    topology: str
    m_pref: float
    m_eff: float
    polarization: float
    fragmentation: float
    local_misalignment: dict[str, float] = field(default_factory=dict)
    local_mean: float = 0.0
    local_std: float = 0.0
    local_min: float = 0.0
    local_max: float = 0.0
    avg_degree: float = 0.0
    avg_toxicity: float = 0.0
    total_welfare: float = 0.0
    total_interactions: int = 0
    accepted_interactions: int = 0


def run_single(scenario, topo_name: str, net_config: NetworkConfig) -> TopologyResult:
    """Run a single topology configuration."""
    sc = copy.deepcopy(scenario)
    sc.orchestrator_config.network_config = net_config

    orchestrator = build_orchestrator(sc)
    config = sc.orchestrator_config

    # Set up MisalignmentModule
    issue_space = IssueSpace(issues=ISSUE_NAMES)
    module = MisalignmentModule(issue_space=issue_space, gov_lambda=1.0)
    for agent in orchestrator.get_all_agents():
        profile = PREFERENCE_MAP.get(agent.agent_type, {
            "prefs": [0.0] * len(ISSUE_NAMES),
            "salience": [1.0 / len(ISSUE_NAMES)] * len(ISSUE_NAMES),
        })
        module.register_agent(
            agent_id=agent.agent_id,
            prefs=profile["prefs"],
            salience=profile["salience"],
        )

    # Wire aggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=sc.scenario_id,
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

    orchestrator.on_interaction_complete(_on_interaction)

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

    orchestrator.on_epoch_end(_on_epoch_end)

    # Run
    metrics_history = orchestrator.run()

    # Compute misalignment
    gov_pressure = _governance_pressure(scenario)
    graph = _build_graph(orchestrator)
    snap = module.compute_snapshot(
        step=config.n_epochs,
        graph=graph,
        uniform_pressure=gov_pressure,
    )

    # Aggregate metrics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_tox = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0.0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    # Local misalignment stats
    local = snap.local
    local_vals = list(local.values()) if local else [0.0]
    local_mean = sum(local_vals) / len(local_vals)
    local_std = math.sqrt(
        sum((v - local_mean) ** 2 for v in local_vals) / len(local_vals)
    ) if len(local_vals) > 1 else 0.0

    # Average degree
    degrees = [len(neighbors) for neighbors in graph.values()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

    return TopologyResult(
        topology=topo_name,
        m_pref=snap.m_pref_global,
        m_eff=snap.m_eff_global,
        polarization=snap.polarization,
        fragmentation=snap.fragmentation,
        local_misalignment=local,
        local_mean=local_mean,
        local_std=local_std,
        local_min=min(local_vals),
        local_max=max(local_vals),
        avg_degree=avg_degree,
        avg_toxicity=avg_tox,
        total_welfare=total_welfare,
        total_interactions=total_interactions,
        accepted_interactions=accepted,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_topology_misalignment_study.py <scenario.yaml>")
        print("\nSuggested: scenarios/misalignment_sweep.yaml")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 90)
    print("Network Topology x Misalignment Study")
    print("=" * 90)
    print()

    scenario = load_scenario(scenario_path)
    config = scenario.orchestrator_config
    print(f"Base scenario: {scenario.scenario_id}")
    print(f"  {config.n_epochs} epochs x {config.steps_per_epoch} steps, seed {config.seed}")
    print(f"  Topologies: {list(TOPOLOGIES.keys())}")
    print()

    # Run each topology
    results: list[TopologyResult] = []
    for topo_name, net_config in TOPOLOGIES.items():
        print(f"  Running {topo_name} ...", end=" ", flush=True)
        r = run_single(scenario, topo_name, net_config)
        results.append(r)
        print(
            f"deg={r.avg_degree:.1f} "
            f"M_local={r.local_mean:.4f}+/-{r.local_std:.4f} "
            f"tox={r.avg_toxicity:.4f} "
            f"welfare={r.total_welfare:.1f}"
        )

    print()

    # =====================================================================
    # Results table
    # =====================================================================
    print("=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    print()

    header = (
        f"{'Topology':<14} {'AvgDeg':<7} "
        f"{'M_pref':<8} {'M_eff':<8} {'Polar':<8} {'Frag':<7} "
        f"{'M_loc_u':<9} {'M_loc_s':<9} {'M_range':<12} "
        f"{'Toxic':<8} {'Welfare':<9}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        m_range = f"[{r.local_min:.3f}-{r.local_max:.3f}]"
        print(
            f"{r.topology:<14} {r.avg_degree:<7.1f} "
            f"{r.m_pref:<8.4f} {r.m_eff:<8.4f} {r.polarization:<8.4f} {r.fragmentation:<7.4f} "
            f"{r.local_mean:<9.4f} {r.local_std:<9.4f} {m_range:<12} "
            f"{r.avg_toxicity:<8.4f} {r.total_welfare:<9.1f}"
        )

    print("-" * 90)
    print()

    # =====================================================================
    # Per-agent local misalignment comparison
    # =====================================================================
    print("=" * 90)
    print("PER-AGENT LOCAL MISALIGNMENT BY TOPOLOGY")
    print("=" * 90)
    print()

    # Get all agent IDs
    all_agents = sorted(
        set().union(*(r.local_misalignment.keys() for r in results))
    )
    agent_types_map = {}

    # Header
    topo_names = [r.topology for r in results]
    print(f"{'Agent':<20} {'Type':<13}", end="")
    for t in topo_names:
        print(f" {t:<12}", end="")
    print()
    print("-" * (33 + 13 * len(topo_names)))

    for aid in all_agents:
        # Try to get agent type from first result
        atype = "?"
        for r in results:
            if aid in r.local_misalignment:
                # Reconstruct from scenario
                if "honest" in aid:
                    atype = "honest"
                elif "opportunistic" in aid:
                    atype = "opportunistic"
                elif "adversarial" in aid:
                    atype = "adversarial"
                elif "deceptive" in aid:
                    atype = "deceptive"
                elif "cautious" in aid:
                    atype = "cautious"
                break
        agent_types_map[aid] = atype

        print(f"{aid:<20} {atype:<13}", end="")
        for r in results:
            val = r.local_misalignment.get(aid, 0.0)
            print(f" {val:<12.4f}", end="")
        print()

    print()

    # =====================================================================
    # Analysis
    # =====================================================================
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print()

    # Topology with highest/lowest local misalignment variance
    highest_var = max(results, key=lambda r: r.local_std)
    lowest_var = min(results, key=lambda r: r.local_std)
    print(f"Highest M_local variance: {highest_var.topology} (std={highest_var.local_std:.4f})")
    print(f"Lowest M_local variance:  {lowest_var.topology} (std={lowest_var.local_std:.4f})")
    print()

    # Degree vs local variance
    print("Degree vs Local Misalignment Variance:")
    for r in sorted(results, key=lambda r: r.avg_degree):
        print(f"  {r.topology:<14} deg={r.avg_degree:.1f}  M_local_std={r.local_std:.4f}")
    print()

    # Compare complete (global view) vs sparse topologies
    complete = next((r for r in results if r.topology == "complete"), None)
    if complete:
        print("Comparison vs complete graph (global view baseline):")
        for r in results:
            if r.topology == "complete":
                continue
            delta_mean = r.local_mean - complete.local_mean
            delta_std = r.local_std - complete.local_std
            delta_tox = r.avg_toxicity - complete.avg_toxicity
            print(
                f"  {r.topology:<14} "
                f"M_local_mean: {delta_mean:+.4f}  "
                f"M_local_std: {delta_std:+.4f}  "
                f"toxicity: {delta_tox:+.4f}"
            )
        print()

    # Key finding
    print("Key finding:")
    if highest_var.topology != "complete":
        print(f"  Sparse topologies ({highest_var.topology}) create higher local")
        print(f"  misalignment heterogeneity (std={highest_var.local_std:.4f} vs")
        print(f"  {lowest_var.local_std:.4f} for {lowest_var.topology}), meaning some")
        print("  agents experience much more misalignment than others.")
    else:
        print("  Complete graph has highest variance — dense connectivity")
        print("  doesn't homogenize local misalignment.")

    # Toxicity comparison
    best_tox = min(results, key=lambda r: r.avg_toxicity)
    worst_tox = max(results, key=lambda r: r.avg_toxicity)
    tox_range = worst_tox.avg_toxicity - best_tox.avg_toxicity
    print()
    print(f"  Toxicity range: {best_tox.avg_toxicity:.4f} ({best_tox.topology}) to "
          f"{worst_tox.avg_toxicity:.4f} ({worst_tox.topology}), "
          f"delta={tox_range:.4f}")
    if tox_range < 0.01:
        print("  => Toxicity is largely INVARIANT to topology (similar to governance finding)")
    else:
        print(f"  => Topology affects toxicity: {worst_tox.topology} is worst")
    print()

    # =====================================================================
    # Export
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_topology_study_seed{config.seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / "topology_results.json"
    with open(export_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": config.seed,
                "topologies": list(TOPOLOGIES.keys()),
                "results": [
                    {
                        "topology": r.topology,
                        "avg_degree": r.avg_degree,
                        "m_pref": r.m_pref,
                        "m_eff": r.m_eff,
                        "polarization": r.polarization,
                        "fragmentation": r.fragmentation,
                        "local_mean": r.local_mean,
                        "local_std": r.local_std,
                        "local_min": r.local_min,
                        "local_max": r.local_max,
                        "local_misalignment": r.local_misalignment,
                        "avg_toxicity": r.avg_toxicity,
                        "total_welfare": r.total_welfare,
                        "total_interactions": r.total_interactions,
                        "accepted_interactions": r.accepted_interactions,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"Exported topology results: {export_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

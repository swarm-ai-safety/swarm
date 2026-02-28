#!/usr/bin/env python
"""
Governance Sensitivity Sweep.

Sweeps tax_rate x audit_probability on the misalignment_sweep population,
measuring how governance pressure modulates effective misalignment, toxicity,
welfare, and agent survival.

Usage:
    python examples/run_governance_sensitivity_sweep.py scenarios/misalignment_sweep.yaml
"""

import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.metrics.misalignment import IssueSpace, MisalignmentModule
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

TAX_RATES = [0.0, 0.02, 0.05, 0.10, 0.15]
AUDIT_PROBS = [0.0, 0.05, 0.10, 0.20]

# ---------------------------------------------------------------------------
# Preference profiles (shared with misalignment study)
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


@dataclass
class SweepResult:
    """Results for a single (tax, audit) configuration."""

    tax_rate: float
    audit_prob: float
    gov_pressure: float
    m_pref: float
    m_eff: float
    gov_reduction_pct: float
    avg_toxicity: float
    total_welfare: float
    total_interactions: int
    accepted_interactions: int
    frozen_count: int
    polarization: float
    fragmentation: float


def run_single(scenario, tax_rate: float, audit_prob: float) -> SweepResult:
    """Run a single configuration and return summary metrics."""
    # Deep-copy scenario to avoid mutation
    sc = copy.deepcopy(scenario)

    # Override governance parameters
    gov = sc.orchestrator_config.governance_config
    gov.transaction_tax_rate = tax_rate
    gov.audit_probability = audit_prob
    gov.audit_enabled = audit_prob > 0

    # Build orchestrator
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

    # Compute misalignment at end
    gov_pressure = tax_rate + audit_prob
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
    frozen_count = len(orchestrator.state.frozen_agents)

    gov_reduction = (
        (1.0 - snap.m_eff_global / snap.m_pref_global) * 100
        if snap.m_pref_global > 0
        else 0.0
    )

    return SweepResult(
        tax_rate=tax_rate,
        audit_prob=audit_prob,
        gov_pressure=gov_pressure,
        m_pref=snap.m_pref_global,
        m_eff=snap.m_eff_global,
        gov_reduction_pct=gov_reduction,
        avg_toxicity=avg_tox,
        total_welfare=total_welfare,
        total_interactions=total_interactions,
        accepted_interactions=accepted,
        frozen_count=frozen_count,
        polarization=snap.polarization,
        fragmentation=snap.fragmentation,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_governance_sensitivity_sweep.py <scenario.yaml>")
        print("\nSuggested: scenarios/misalignment_sweep.yaml")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 90)
    print("Governance Sensitivity Sweep")
    print("=" * 90)
    print()

    scenario = load_scenario(scenario_path)
    print(f"Base scenario: {scenario.scenario_id}")
    print(f"  {scenario.orchestrator_config.n_epochs} epochs x "
          f"{scenario.orchestrator_config.steps_per_epoch} steps, "
          f"seed {scenario.orchestrator_config.seed}")
    print(f"  Tax rates: {TAX_RATES}")
    print(f"  Audit probs: {AUDIT_PROBS}")
    print(f"  Total configs: {len(TAX_RATES) * len(AUDIT_PROBS)}")
    print()

    # Run sweep
    results: list[SweepResult] = []
    total = len(TAX_RATES) * len(AUDIT_PROBS)
    idx = 0

    for tax in TAX_RATES:
        for audit in AUDIT_PROBS:
            idx += 1
            label = f"[{idx}/{total}] tax={tax:.2f}, audit={audit:.2f}"
            print(f"  Running {label} ...", end=" ", flush=True)
            r = run_single(scenario, tax, audit)
            results.append(r)
            print(
                f"M_eff={r.m_eff:.4f} "
                f"(-{r.gov_reduction_pct:.1f}%) "
                f"tox={r.avg_toxicity:.4f} "
                f"welfare={r.total_welfare:.1f} "
                f"frozen={r.frozen_count}"
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
        f"{'Tax':<6} {'Audit':<7} {'G_pres':<7} "
        f"{'M_pref':<8} {'M_eff':<8} {'Reduc%':<8} "
        f"{'Toxic':<8} {'Welfare':<9} {'Intx':<6} {'Acc':<5} {'Frz':<4}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        print(
            f"{r.tax_rate:<6.2f} {r.audit_prob:<7.2f} {r.gov_pressure:<7.3f} "
            f"{r.m_pref:<8.4f} {r.m_eff:<8.4f} {r.gov_reduction_pct:<8.1f} "
            f"{r.avg_toxicity:<8.4f} {r.total_welfare:<9.1f} "
            f"{r.total_interactions:<6} {r.accepted_interactions:<5} {r.frozen_count:<4}"
        )

    print("-" * 90)
    print()

    # =====================================================================
    # M_eff heatmap (text)
    # =====================================================================
    print("M_eff by Tax Rate (rows) x Audit Probability (columns):")
    print()
    # Header
    hdr = "Tax \\ Audit"
    print(f"{hdr:<12}", end="")
    for audit in AUDIT_PROBS:
        print(f"{audit:<10.2f}", end="")
    print()
    print("-" * (12 + 10 * len(AUDIT_PROBS)))

    for tax in TAX_RATES:
        print(f"{tax:<12.2f}", end="")
        for audit in AUDIT_PROBS:
            r = next(x for x in results if x.tax_rate == tax and x.audit_prob == audit)
            print(f"{r.m_eff:<10.4f}", end="")
        print()
    print()

    # =====================================================================
    # Governance reduction heatmap
    # =====================================================================
    print("Governance Reduction % by Tax Rate (rows) x Audit Probability (columns):")
    print()
    hdr = "Tax \\ Audit"
    print(f"{hdr:<12}", end="")
    for audit in AUDIT_PROBS:
        print(f"{audit:<10.2f}", end="")
    print()
    print("-" * (12 + 10 * len(AUDIT_PROBS)))

    for tax in TAX_RATES:
        print(f"{tax:<12.2f}", end="")
        for audit in AUDIT_PROBS:
            r = next(x for x in results if x.tax_rate == tax and x.audit_prob == audit)
            print(f"{r.gov_reduction_pct:<10.1f}", end="")
        print()
    print()

    # =====================================================================
    # Toxicity heatmap
    # =====================================================================
    print("Avg Toxicity by Tax Rate (rows) x Audit Probability (columns):")
    print()
    hdr = "Tax \\ Audit"
    print(f"{hdr:<12}", end="")
    for audit in AUDIT_PROBS:
        print(f"{audit:<10.2f}", end="")
    print()
    print("-" * (12 + 10 * len(AUDIT_PROBS)))

    for tax in TAX_RATES:
        print(f"{tax:<12.2f}", end="")
        for audit in AUDIT_PROBS:
            r = next(x for x in results if x.tax_rate == tax and x.audit_prob == audit)
            print(f"{r.avg_toxicity:<10.4f}", end="")
        print()
    print()

    # =====================================================================
    # Analysis
    # =====================================================================
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print()

    # Best and worst configs
    best_meff = min(results, key=lambda r: r.m_eff)
    worst_meff = max(results, key=lambda r: r.m_eff)
    best_tox = min(results, key=lambda r: r.avg_toxicity)
    best_welfare = max(results, key=lambda r: r.total_welfare)

    print(f"Lowest M_eff:    tax={best_meff.tax_rate:.2f}, audit={best_meff.audit_prob:.2f} "
          f"-> M_eff={best_meff.m_eff:.4f} ({best_meff.gov_reduction_pct:.1f}% reduction)")
    print(f"Highest M_eff:   tax={worst_meff.tax_rate:.2f}, audit={worst_meff.audit_prob:.2f} "
          f"-> M_eff={worst_meff.m_eff:.4f} ({worst_meff.gov_reduction_pct:.1f}% reduction)")
    print(f"Lowest toxicity: tax={best_tox.tax_rate:.2f}, audit={best_tox.audit_prob:.2f} "
          f"-> tox={best_tox.avg_toxicity:.4f}")
    print(f"Best welfare:    tax={best_welfare.tax_rate:.2f}, audit={best_welfare.audit_prob:.2f} "
          f"-> welfare={best_welfare.total_welfare:.1f}")
    print()

    # Marginal effects: tax rate (averaged over audit)
    print("Marginal effect of tax rate (averaged over audit levels):")
    for tax in TAX_RATES:
        subset = [r for r in results if r.tax_rate == tax]
        mean_meff = sum(r.m_eff for r in subset) / len(subset)
        mean_tox = sum(r.avg_toxicity for r in subset) / len(subset)
        mean_welfare = sum(r.total_welfare for r in subset) / len(subset)
        print(f"  tax={tax:.2f}: M_eff={mean_meff:.4f}, tox={mean_tox:.4f}, welfare={mean_welfare:.1f}")
    print()

    # Marginal effects: audit probability (averaged over tax)
    print("Marginal effect of audit probability (averaged over tax levels):")
    for audit in AUDIT_PROBS:
        subset = [r for r in results if r.audit_prob == audit]
        mean_meff = sum(r.m_eff for r in subset) / len(subset)
        mean_tox = sum(r.avg_toxicity for r in subset) / len(subset)
        mean_welfare = sum(r.total_welfare for r in subset) / len(subset)
        print(f"  audit={audit:.2f}: M_eff={mean_meff:.4f}, tox={mean_tox:.4f}, welfare={mean_welfare:.1f}")
    print()

    # Diminishing returns check
    no_gov = next((r for r in results if r.tax_rate == 0 and r.audit_prob == 0), None)
    max_gov = next(
        (r for r in results if r.tax_rate == TAX_RATES[-1] and r.audit_prob == AUDIT_PROBS[-1]),
        None,
    )
    if no_gov and max_gov:
        meff_range = no_gov.m_eff - max_gov.m_eff
        tox_range = no_gov.avg_toxicity - max_gov.avg_toxicity
        welfare_range = max_gov.total_welfare - no_gov.total_welfare
        print("Full range (no governance -> max governance):")
        print(f"  M_eff:    {no_gov.m_eff:.4f} -> {max_gov.m_eff:.4f} (delta={meff_range:+.4f})")
        print(f"  Toxicity: {no_gov.avg_toxicity:.4f} -> {max_gov.avg_toxicity:.4f} (delta={tox_range:+.4f})")
        print(f"  Welfare:  {no_gov.total_welfare:.1f} -> {max_gov.total_welfare:.1f} (delta={welfare_range:+.1f})")
    print()

    # =====================================================================
    # Export
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_governance_sweep_seed{scenario.orchestrator_config.seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / "governance_sweep_results.json"
    with open(export_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": scenario.orchestrator_config.seed,
                "tax_rates": TAX_RATES,
                "audit_probs": AUDIT_PROBS,
                "results": [
                    {
                        "tax_rate": r.tax_rate,
                        "audit_prob": r.audit_prob,
                        "gov_pressure": r.gov_pressure,
                        "m_pref": r.m_pref,
                        "m_eff": r.m_eff,
                        "gov_reduction_pct": r.gov_reduction_pct,
                        "avg_toxicity": r.avg_toxicity,
                        "total_welfare": r.total_welfare,
                        "total_interactions": r.total_interactions,
                        "accepted_interactions": r.accepted_interactions,
                        "frozen_count": r.frozen_count,
                        "polarization": r.polarization,
                        "fragmentation": r.fragmentation,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"Exported sweep results: {export_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

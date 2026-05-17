#!/usr/bin/env python
"""
Misalignment x Causal Credit x Toxicity Triangle Study.

Wires MisalignmentModule + CausalCreditEngine + SoftMetrics together to test
whether preference misalignment causally drives toxicity or merely correlates.

Analysis:
  1. Per-agent M_local vs causal credit vs toxicity contribution
  2. Granger-style lagged correlation across epochs
  3. Intervention analysis: counterfactual removal of highest-M_local agent

Usage:
    python examples/run_triangle_study.py scenarios/misalignment_sweep.yaml
"""

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
from swarm.metrics.misalignment import IssueSpace, MisalignmentModule
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction
from swarm.scenarios import build_orchestrator, load_scenario

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
    """Build adjacency list from orchestrator network."""
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


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation. Returns 0.0 if insufficient variance."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx**0.5 * vy**0.5)


def _per_agent_toxicity(
    interactions: list[SoftInteraction],
) -> dict[str, float]:
    """Compute per-agent toxicity contribution: mean (1-p) for interactions initiated."""
    agent_vals: dict[str, list[float]] = defaultdict(list)
    for ix in interactions:
        agent_vals[ix.initiator].append(1.0 - ix.p)
        agent_vals[ix.counterparty].append(1.0 - ix.p)
    return {
        aid: sum(vs) / len(vs) if vs else 0.0
        for aid, vs in agent_vals.items()
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_triangle_study.py <scenario.yaml>")
        print("\nSuggested: scenarios/misalignment_sweep.yaml")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Misalignment x Causal Credit x Toxicity — Triangle Study")
    print("=" * 78)
    print()

    # Load scenario
    print(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)
    config = scenario.orchestrator_config
    print(f"  ID: {scenario.scenario_id}")
    print(f"  {config.n_epochs} epochs x {config.steps_per_epoch} steps, seed {config.seed}")
    print()

    # Build orchestrator
    orchestrator = build_orchestrator(scenario)

    # --- Set up modules ---
    issue_space = IssueSpace(issues=ISSUE_NAMES)
    misalign_module = MisalignmentModule(issue_space=issue_space, gov_lambda=1.0)
    credit_engine = CausalCreditEngine(decay=0.5, max_depth=10)
    soft_metrics = SoftMetrics()

    for agent in orchestrator.get_all_agents():
        profile = PREFERENCE_MAP.get(agent.agent_type, {
            "prefs": [0.0] * len(ISSUE_NAMES),
            "salience": [1.0 / len(ISSUE_NAMES)] * len(ISSUE_NAMES),
        })
        misalign_module.register_agent(
            agent_id=agent.agent_id,
            prefs=profile["prefs"],
            salience=profile["salience"],
        )

    agent_types = {a.agent_id: a.agent_type for a in orchestrator.get_all_agents()}
    gov_pressure = _governance_pressure(scenario)
    print(f"  Governance pressure: {gov_pressure:.3f}")
    print(f"  Agents: {len(orchestrator.get_all_agents())}")
    print()

    # --- Wire aggregator ---
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    # Tracking state
    agent_last_ix: dict[str, str] = {}
    all_interactions: list[SoftInteraction] = []
    epoch_interactions: list[SoftInteraction] = []

    # Time series per epoch
    epoch_toxicities: list[float] = []
    epoch_m_prefs: list[float] = []
    epoch_m_effs: list[float] = []
    epoch_credit_by_agent: list[dict[str, float]] = []
    epoch_toxicity_by_agent: list[dict[str, float]] = []
    epoch_local_misalignment: list[dict[str, float]] = []
    credit_snapshots: list[CausalSnapshot] = []

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        # Wire causal parents
        parents = []
        if interaction.initiator in agent_last_ix:
            parents.append(agent_last_ix[interaction.initiator])
        if interaction.counterparty in agent_last_ix:
            parents.append(agent_last_ix[interaction.counterparty])
        interaction.causal_parents = parents
        agent_last_ix[interaction.initiator] = interaction.interaction_id
        agent_last_ix[interaction.counterparty] = interaction.interaction_id

        all_interactions.append(interaction)
        epoch_interactions.append(interaction)

        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

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

        # 1. Misalignment snapshot
        graph = _build_graph(orchestrator)
        m_snap = misalign_module.compute_snapshot(
            step=epoch_metrics.epoch,
            graph=graph,
            uniform_pressure=gov_pressure,
        )
        epoch_m_prefs.append(m_snap.m_pref_global)
        epoch_m_effs.append(m_snap.m_eff_global)
        epoch_local_misalignment.append(dict(m_snap.local))

        # 2. Causal credit snapshot
        c_snap = credit_engine.compute_snapshot(
            step=epoch_metrics.epoch,
            interactions=list(all_interactions),
            signal="p",
        )
        credit_snapshots.append(c_snap)
        epoch_credit_by_agent.append(dict(c_snap.credit_by_agent))

        # 3. Toxicity
        epoch_tox = soft_metrics.toxicity_rate(list(epoch_interactions))
        epoch_toxicities.append(epoch_tox)
        epoch_toxicity_by_agent.append(_per_agent_toxicity(list(epoch_interactions)))

        epoch_interactions.clear()

    orchestrator.on_epoch_end(_on_epoch_end)

    # --- Run ---
    print(f"Running: {config.n_epochs} epochs x {config.steps_per_epoch} steps")
    print("-" * 78)
    metrics_history = orchestrator.run()

    # =====================================================================
    # ANALYSIS 1: Per-agent triangle (final epoch)
    # =====================================================================
    print()
    print("=" * 78)
    print("ANALYSIS 1: Per-Agent Triangle (final epoch)")
    print("=" * 78)
    print()

    final_local = epoch_local_misalignment[-1] if epoch_local_misalignment else {}
    final_credit = epoch_credit_by_agent[-1] if epoch_credit_by_agent else {}
    final_tox = epoch_toxicity_by_agent[-1] if epoch_toxicity_by_agent else {}

    header = (
        f"{'Agent':<20} {'Type':<13} "
        f"{'M_local':<9} {'Credit':<10} {'Toxicity':<10}"
    )
    print(header)
    print("-" * 62)

    agent_ids = sorted(
        set(final_local.keys()) | set(final_credit.keys()) | set(final_tox.keys())
    )
    for aid in agent_ids:
        atype = agent_types.get(aid, AgentType.HONEST).value
        ml = final_local.get(aid, 0.0)
        cr = final_credit.get(aid, 0.0)
        tx = final_tox.get(aid, 0.0)
        print(f"{aid:<20} {atype:<13} {ml:<9.4f} {cr:<10.2f} {tx:<10.4f}")

    print()

    # Pairwise correlations
    common = sorted(set(final_local.keys()) & set(final_credit.keys()) & set(final_tox.keys()))
    if len(common) > 2:
        ml_vals = [final_local[a] for a in common]
        cr_vals = [final_credit[a] for a in common]
        tx_vals = [final_tox[a] for a in common]

        r_ml_tx = _pearson(ml_vals, tx_vals)
        r_ml_cr = _pearson(ml_vals, cr_vals)
        r_cr_tx = _pearson(cr_vals, tx_vals)

        print("Pairwise Correlations (per-agent, final epoch):")
        print(f"  M_local vs Toxicity:  r = {r_ml_tx:+.4f}")
        print(f"  M_local vs Credit:    r = {r_ml_cr:+.4f}")
        print(f"  Credit vs Toxicity:   r = {r_cr_tx:+.4f}")
        print()

        # Interpretation
        if r_ml_tx > 0.5:
            print("  => Strong positive M_local-Toxicity link: misaligned agents produce more harm")
        elif r_ml_tx < -0.5:
            print("  => Inverse M_local-Toxicity: misaligned agents are LESS toxic (governance effect?)")
        else:
            print("  => Weak M_local-Toxicity link: misalignment alone doesn't predict toxicity")

        if r_ml_cr < -0.3:
            print("  => Negative M_local-Credit: misaligned agents get less causal credit (expected)")
        elif r_ml_cr > 0.3:
            print("  => Positive M_local-Credit: misaligned agents drive MORE causal influence (concerning)")
        print()

    # =====================================================================
    # ANALYSIS 2: Granger-style lagged correlation
    # =====================================================================
    print("=" * 78)
    print("ANALYSIS 2: Granger-Style Lagged Correlation")
    print("=" * 78)
    print()
    print("Does misalignment at epoch t predict toxicity at epoch t+lag?")
    print()

    n_epochs = len(epoch_toxicities)
    if n_epochs > 4:
        print(f"{'Lag':<6} {'M_pref->Toxic':<16} {'M_eff->Toxic':<16} {'N':<4}")
        print("-" * 42)
        for lag in range(4):
            n_pairs = n_epochs - lag
            if n_pairs < 3:
                break
            m_pref_slice = epoch_m_prefs[:n_pairs]
            m_eff_slice = epoch_m_effs[:n_pairs]
            tox_slice = epoch_toxicities[lag:lag + n_pairs]

            r_pref = _pearson(m_pref_slice, tox_slice)
            r_eff = _pearson(m_eff_slice, tox_slice)
            print(f"{lag:<6} {r_pref:<16.4f} {r_eff:<16.4f} {n_pairs:<4}")
        print()

        # Also test: does toxicity predict FUTURE misalignment? (reverse causality check)
        print("Reverse check: does toxicity at t predict misalignment at t+lag?")
        print(f"{'Lag':<6} {'Toxic->M_pref':<16} {'N':<4}")
        print("-" * 26)
        for lag in range(1, 4):
            n_pairs = n_epochs - lag
            if n_pairs < 3:
                break
            tox_slice = epoch_toxicities[:n_pairs]
            m_slice = epoch_m_prefs[lag:lag + n_pairs]
            r = _pearson(tox_slice, m_slice)
            print(f"{lag:<6} {r:<16.4f} {n_pairs:<4}")
        print()
    else:
        print("  (Not enough epochs for lagged analysis)")
        print()

    # =====================================================================
    # ANALYSIS 3: Intervention — remove highest-M_local agent
    # =====================================================================
    print("=" * 78)
    print("ANALYSIS 3: Counterfactual Intervention")
    print("=" * 78)
    print()

    if final_local:
        # Find agent with highest M_local
        highest_agent = max(final_local, key=final_local.get)  # type: ignore[arg-type]
        highest_ml = final_local[highest_agent]
        highest_type = agent_types.get(highest_agent, AgentType.HONEST).value
        print(f"Highest M_local agent: {highest_agent} ({highest_type}), M_local={highest_ml:.4f}")
        print()

        # Recompute global misalignment WITHOUT this agent
        remaining = [a for a in agent_ids if a != highest_agent]
        m_pref_without = misalign_module.global_misalignment(remaining)
        m_eff_without = misalign_module.global_effective_misalignment(
            uniform_pressure=gov_pressure, agent_ids=remaining
        )

        m_pref_with = epoch_m_prefs[-1] if epoch_m_prefs else 0.0
        m_eff_with = epoch_m_effs[-1] if epoch_m_effs else 0.0

        print("Global misalignment with vs without highest-M_local agent:")
        print(f"  M_pref:  {m_pref_with:.4f} -> {m_pref_without:.4f} (delta={m_pref_without - m_pref_with:+.4f})")
        print(f"  M_eff:   {m_eff_with:.4f} -> {m_eff_without:.4f} (delta={m_eff_without - m_eff_with:+.4f})")
        print()

        # Recompute toxicity without interactions involving this agent
        remaining_ix = [
            ix for ix in all_interactions
            if ix.initiator != highest_agent and ix.counterparty != highest_agent
        ]
        tox_with = soft_metrics.toxicity_rate(all_interactions)
        tox_without = soft_metrics.toxicity_rate(remaining_ix) if remaining_ix else 0.0

        print("Toxicity with vs without:")
        print(f"  Toxicity: {tox_with:.4f} -> {tox_without:.4f} (delta={tox_without - tox_with:+.4f})")
        print(f"  Interactions: {len(all_interactions)} -> {len(remaining_ix)}")
        print()

        # Credit recomputation
        credit_with = final_credit.get(highest_agent, 0.0)
        total_credit = sum(final_credit.values())
        credit_share = credit_with / total_credit if total_credit > 0 else 0.0
        print(f"Agent's causal credit: {credit_with:.2f} ({credit_share:.1%} of total)")
        print()

        # Verdict
        if (m_pref_without < m_pref_with) and (tox_without < tox_with):
            print("VERDICT: Removing highest-M_local agent reduces BOTH misalignment and toxicity")
            print("  => Misalignment likely DRIVES toxicity (not just correlation)")
        elif m_pref_without < m_pref_with and tox_without >= tox_with:
            print("VERDICT: Removing agent reduces misalignment but NOT toxicity")
            print("  => Misalignment and toxicity are decoupled — different mechanisms")
        elif m_pref_without >= m_pref_with and tox_without < tox_with:
            print("VERDICT: Removing agent reduces toxicity but NOT misalignment")
            print("  => Agent is toxic for reasons beyond preference divergence")
        else:
            print("VERDICT: Removing agent doesn't reduce either metric")
            print("  => This agent is not a key driver")
    else:
        print("  (No local misalignment data available)")

    print()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    total_interactions = sum(m.total_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    print(f"  Interactions: {total_interactions} ({sum(m.accepted_interactions for m in metrics_history)} accepted)")
    print(f"  Avg Toxicity: {avg_toxicity:.4f}")
    print(f"  Total Welfare: {total_welfare:.2f}")
    if epoch_m_prefs:
        print(f"  M_pref (mean): {sum(epoch_m_prefs)/len(epoch_m_prefs):.4f}")
        print(f"  M_eff (mean):  {sum(epoch_m_effs)/len(epoch_m_effs):.4f}")
        print(f"  Governance reduction: {1.0 - epoch_m_effs[-1]/epoch_m_prefs[-1]:.1%}" if epoch_m_prefs[-1] > 0 else "")
    if credit_snapshots:
        print(f"  Final DAG depth: {credit_snapshots[-1].dag_depth}")
    print()

    # --- Final agent states ---
    print("Final Agent States:")
    print("-" * 78)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        ml = final_local.get(agent.agent_id, 0.0)
        cr = final_credit.get(agent.agent_id, 0.0)
        print(
            f"  {agent.agent_id} ({agent.agent_type.value}): "
            f"rep={state.reputation:.2f}, payoff={state.total_payoff:.2f}, "
            f"M_local={ml:.4f}, credit={cr:+.2f}"
            f"{status}"
        )
    print()

    # --- Export ---
    history = aggregator.end_simulation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_triangle_study_seed{config.seed}"

    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)

    # Triangle analysis export
    triangle_path = export_dir / "triangle_analysis.json"
    triangle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(triangle_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": config.seed,
                "gov_pressure": gov_pressure,
                "epoch_timeseries": {
                    "toxicity": epoch_toxicities,
                    "m_pref": epoch_m_prefs,
                    "m_eff": epoch_m_effs,
                },
                "per_agent_final": {
                    aid: {
                        "type": agent_types.get(aid, AgentType.HONEST).value,
                        "m_local": final_local.get(aid, 0.0),
                        "credit": final_credit.get(aid, 0.0),
                        "toxicity": final_tox.get(aid, 0.0),
                    }
                    for aid in agent_ids
                },
                "causal_snapshots": [
                    {
                        "step": s.step,
                        "dag_depth": s.dag_depth,
                        "dag_width": s.dag_width,
                        "total_interactions": s.total_interactions,
                        "credit_by_agent": s.credit_by_agent,
                        "top_blame": s.top_blame,
                        "top_credit": s.top_credit,
                    }
                    for s in credit_snapshots
                ],
            },
            f,
            indent=2,
        )

    print(f"Exported run history: {export_path}")
    print(f"Exported triangle analysis: {triangle_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

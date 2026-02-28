#!/usr/bin/env python
"""
Security Evaluation Study — threat detection sweep across adversarial mix ratios.

Sweeps adversarial agent mix from 0% to 50% and measures detection rates for
prompt injection, laundering, exfiltration, and contagion. Tracks false
positive/negative rates and ecosystem threat level across governance configs.

Usage:
    python examples/run_security_study.py [scenarios/security_evaluation.yaml]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.security import (
    SecurityAnalyzer,
    SecurityReport,
    compute_threat_trend,
)
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Adversarial mix ratios to sweep (percentage of adversarial agents)
# ---------------------------------------------------------------------------
MIX_RATIOS = [0, 10, 20, 30, 40, 50]
TOTAL_AGENTS = 10  # Fixed population size


def _build_agent_specs(mix_pct: int) -> list[dict]:
    """Build agent specs for a given adversarial mix percentage.

    Distributes TOTAL_AGENTS across honest, opportunistic, and adversarial
    types. Adversarial count = mix_pct% of total (rounded). The remainder
    is split 60/40 between honest and opportunistic.
    """
    n_adversarial = round(TOTAL_AGENTS * mix_pct / 100)
    remaining = TOTAL_AGENTS - n_adversarial
    n_honest = max(1, round(remaining * 0.6))
    n_opportunistic = remaining - n_honest

    specs = []
    if n_honest > 0:
        specs.append({"type": "honest", "count": n_honest})
    if n_opportunistic > 0:
        specs.append({"type": "opportunistic", "count": n_opportunistic})
    if n_adversarial > 0:
        specs.append({"type": "adversarial", "count": n_adversarial})
    return specs


def _run_single_mix(scenario_path: Path, mix_pct: int, seed: int) -> dict:
    """Run a single simulation at the given adversarial mix ratio.

    Returns a dict with all security metrics for this run.
    """
    # Load and patch scenario
    scenario = load_scenario(scenario_path)
    scenario.agent_specs = _build_agent_specs(mix_pct)
    scenario.orchestrator_config.seed = seed

    # Build orchestrator
    orchestrator = build_orchestrator(scenario)
    config = scenario.orchestrator_config

    # Set up SecurityAnalyzer
    analyzer = SecurityAnalyzer(
        injection_threshold=0.3,
        manipulation_threshold=0.4,
        laundering_trust_gap=0.25,
        contagion_velocity_threshold=2.0,
        min_chain_length=3,
        seed=seed,
    )

    # Wire MetricsAggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=f"{scenario.scenario_id}_mix{mix_pct}",
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    # Collect all interactions for security analysis
    all_interactions = []
    epoch_reports: list[SecurityReport] = []

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

        # Feed to security analyzer
        analyzer.record_interaction(interaction)
        all_interactions.append(interaction)

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

        # Set trust scores from current reputation
        trust_scores = {}
        for agent in orchestrator.get_all_agents():
            state = orchestrator.state.get_agent(agent.agent_id)
            trust_scores[agent.agent_id] = max(0.0, min(1.0, state.reputation))
        analyzer.set_agent_trust_scores(trust_scores)

        # Run security analysis on all interactions so far
        agent_ids = [a.agent_id for a in orchestrator.get_all_agents()]
        report = analyzer.analyze(interactions=list(all_interactions), agent_ids=agent_ids)
        epoch_reports.append(report)

    orchestrator.on_epoch_end(_on_epoch_end)

    # Run simulation
    metrics_history = orchestrator.run()

    # Compute agent type breakdown
    agent_type_map = {a.agent_id: a.agent_type for a in orchestrator.get_all_agents()}
    n_honest = sum(1 for t in agent_type_map.values() if t == AgentType.HONEST)
    n_opportunistic = sum(1 for t in agent_type_map.values() if t == AgentType.OPPORTUNISTIC)
    n_adversarial = sum(1 for t in agent_type_map.values() if t == AgentType.ADVERSARIAL)

    # Compute false positive/negative estimates
    # "True adversarial" = adversarial agents, "true benign" = honest agents
    adversarial_ids = {aid for aid, t in agent_type_map.items() if t == AgentType.ADVERSARIAL}
    benign_ids = {aid for aid, t in agent_type_map.items() if t == AgentType.HONEST}

    final_report = epoch_reports[-1] if epoch_reports else SecurityReport()
    flagged = final_report.agents_flagged

    # False positives: benign agents flagged as threats
    false_positives = len(flagged & benign_ids)
    fp_rate = false_positives / len(benign_ids) if benign_ids else 0.0

    # False negatives: adversarial agents NOT flagged
    if adversarial_ids:
        true_positives = len(flagged & adversarial_ids)
        false_negatives = len(adversarial_ids) - true_positives
        fn_rate = false_negatives / len(adversarial_ids)
        detection_rate = true_positives / len(adversarial_ids)
    else:
        false_negatives = 0
        fn_rate = 0.0
        detection_rate = 1.0  # No adversaries to miss

    # Quarantine stats
    quarantined = getattr(orchestrator.state, "quarantined_agents", set())
    quarantine_count = len(quarantined) if quarantined else 0

    # Threat trend
    trend = compute_threat_trend(epoch_reports)

    # Standard metrics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0.0
    )

    # End aggregator
    history = aggregator.end_simulation()

    return {
        "mix_pct": mix_pct,
        "seed": seed,
        "n_honest": n_honest,
        "n_opportunistic": n_opportunistic,
        "n_adversarial": n_adversarial,
        "total_interactions": total_interactions,
        "avg_toxicity": avg_toxicity,
        # Security rates
        "prompt_injection_rate": final_report.prompt_injection_rate,
        "manipulation_rate": final_report.manipulation_rate,
        "laundering_rate": final_report.laundering_rate,
        "contagion_depth": final_report.contagion_depth,
        "exfiltration_attempts": final_report.exfiltration_attempts,
        # Ecosystem
        "ecosystem_threat_level": final_report.ecosystem_threat_level,
        "active_threat_count": final_report.active_threat_count,
        # Detection quality
        "detection_rate": detection_rate,
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate,
        "agents_flagged": len(flagged),
        "quarantine_count": quarantine_count,
        # Trend
        "threat_trend": trend.get("trend", 0.0),
        "threat_acceleration": trend.get("acceleration", 0.0),
        # Per-epoch detail
        "epoch_threat_levels": [r.ecosystem_threat_level for r in epoch_reports],
        "epoch_injection_rates": [r.prompt_injection_rate for r in epoch_reports],
        "epoch_manipulation_rates": [r.manipulation_rate for r in epoch_reports],
        # History for export
        "_history": history,
    }


def main():
    scenario_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "scenarios/security_evaluation.yaml"
    )
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    # Load base scenario for seed
    base_scenario = load_scenario(scenario_path)
    seed = base_scenario.orchestrator_config.seed

    print("=" * 82)
    print("Security Evaluation Study")
    print("Threat detection sweep across adversarial mix ratios")
    print("=" * 82)
    print()
    print(f"Scenario: {scenario_path}")
    print(f"Seed: {seed}")
    print(f"Population: {TOTAL_AGENTS} agents")
    print(f"Mix ratios: {MIX_RATIOS}%")
    print()

    # --- Sweep across mix ratios ---
    results = []
    for mix_pct in MIX_RATIOS:
        print(f"--- Running mix={mix_pct}% adversarial ---")
        specs = _build_agent_specs(mix_pct)
        counts = {s["type"]: s["count"] for s in specs}
        print(f"  Agents: {counts}")
        result = _run_single_mix(scenario_path, mix_pct, seed)
        results.append(result)
        print(
            f"  Threat level: {result['ecosystem_threat_level']:.4f}, "
            f"Detection: {result['detection_rate']:.2%}, "
            f"FPR: {result['false_positive_rate']:.2%}, "
            f"FNR: {result['false_negative_rate']:.2%}"
        )
        print()

    # --- Comparison table ---
    print()
    print("=" * 82)
    print("Security Detection Sweep Results")
    print("=" * 82)
    print()

    # Table 1: Threat detection rates
    print("Threat Detection by Mix Ratio:")
    print("-" * 82)
    header = (
        f"{'Mix%':<6} {'Intx':<6} {'Toxic':<7} "
        f"{'Inject':<8} {'Manip':<8} {'Laund':<8} "
        f"{'Contag':<7} {'Exfil':<6} {'Threat':<8}"
    )
    print(header)
    print("-" * 82)

    for r in results:
        print(
            f"{r['mix_pct']:<6} "
            f"{r['total_interactions']:<6} "
            f"{r['avg_toxicity']:<7.4f} "
            f"{r['prompt_injection_rate']:<8.4f} "
            f"{r['manipulation_rate']:<8.4f} "
            f"{r['laundering_rate']:<8.4f} "
            f"{r['contagion_depth']:<7} "
            f"{r['exfiltration_attempts']:<6} "
            f"{r['ecosystem_threat_level']:<8.4f}"
        )

    print("-" * 82)
    print()

    # Table 2: Detection quality
    print("Detection Quality by Mix Ratio:")
    print("-" * 82)
    header2 = (
        f"{'Mix%':<6} {'H/O/A':<12} "
        f"{'Detect':<8} {'FPR':<8} {'FNR':<8} "
        f"{'Flagged':<8} {'Quarant':<8} {'Threats':<8} {'Trend':<8}"
    )
    print(header2)
    print("-" * 82)

    for r in results:
        agent_str = f"{r['n_honest']}/{r['n_opportunistic']}/{r['n_adversarial']}"
        print(
            f"{r['mix_pct']:<6} "
            f"{agent_str:<12} "
            f"{r['detection_rate']:<8.2%} "
            f"{r['false_positive_rate']:<8.2%} "
            f"{r['false_negative_rate']:<8.2%} "
            f"{r['agents_flagged']:<8} "
            f"{r['quarantine_count']:<8} "
            f"{r['active_threat_count']:<8} "
            f"{r['threat_trend']:+<8.4f}"
        )

    print("-" * 82)
    print()

    # --- Summary statistics ---
    print("Summary Statistics:")
    print("-" * 82)

    # Threat level scaling
    threat_levels = [r["ecosystem_threat_level"] for r in results]
    mix_pcts = [r["mix_pct"] for r in results]
    print(f"  Ecosystem threat level range: {min(threat_levels):.4f} - {max(threat_levels):.4f}")
    if len(threat_levels) >= 2 and mix_pcts[-1] != mix_pcts[0]:
        slope = (threat_levels[-1] - threat_levels[0]) / (mix_pcts[-1] - mix_pcts[0])
        print(f"  Threat-per-%adversarial slope: {slope:.4f}")
    print()

    # Detection quality summary
    detection_rates = [r["detection_rate"] for r in results if r["n_adversarial"] > 0]
    fp_rates = [r["false_positive_rate"] for r in results]
    fn_rates = [r["false_negative_rate"] for r in results if r["n_adversarial"] > 0]

    if detection_rates:
        print(f"  Avg detection rate (when adversaries present): {sum(detection_rates)/len(detection_rates):.2%}")
    print(f"  Avg false positive rate: {sum(fp_rates)/len(fp_rates):.2%}")
    if fn_rates:
        print(f"  Avg false negative rate: {sum(fn_rates)/len(fn_rates):.2%}")
    print()

    # Injection rate scaling
    injection_rates = [r["prompt_injection_rate"] for r in results]
    print(f"  Injection rate range: {min(injection_rates):.4f} - {max(injection_rates):.4f}")
    manipulation_rates = [r["manipulation_rate"] for r in results]
    print(f"  Manipulation rate range: {min(manipulation_rates):.4f} - {max(manipulation_rates):.4f}")
    print()

    # --- Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_security_sweep_seed{seed}"

    # Export each run's history
    for r in results:
        mix_dir = export_dir / f"mix_{r['mix_pct']}pct"
        history = r.pop("_history")
        export_path = mix_dir / "history.json"
        export_to_json(history, export_path, include_events=True)

    # Export sweep summary
    summary_path = export_dir / "security_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "scenario_id": base_scenario.scenario_id,
                "seed": seed,
                "total_agents": TOTAL_AGENTS,
                "mix_ratios": MIX_RATIOS,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Exported run histories: {export_dir}/mix_*/history.json")
    print(f"Exported sweep summary: {summary_path}")
    print(f"  {len(results)} mix ratios swept")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

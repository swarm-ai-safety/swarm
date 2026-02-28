#!/usr/bin/env python
"""
Collusion Detection Study — clique detection and statistical tests sweep.

Research question:
    How does collusion detection accuracy (flagged pairs, clique detection,
    false positive rate, time-to-first-detection) vary across adversarial agent
    count and audit probability?

Swept parameters:
    - Adversarial agent count: 1, 2, 3, 4, 5
    - Audit probability: 0.05, 0.10, 0.20

For each config the runner:
    1. Loads the collusion_detection scenario and patches agent counts + audit prob
    2. Runs the simulation with MetricsAggregator callbacks
    3. Runs CollusionDetector.analyze() per epoch to collect detection metrics
    4. Tracks flagged pairs, clique detection accuracy, FPR, time-to-first-detection
    5. Prints comparison table across sweep configs
    6. Exports results as JSON

Usage:
    python examples/run_collusion_study.py [scenarios/collusion_detection.yaml]
"""

import copy
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.collusion import CollusionDetector, CollusionReport
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

ADVERSARIAL_COUNTS = [1, 2, 3, 4, 5]
AUDIT_PROBABILITIES = [0.05, 0.10, 0.20]


def _agent_type_map(orchestrator) -> dict[str, AgentType]:
    """Build agent_id -> AgentType lookup."""
    return {a.agent_id: a.agent_type for a in orchestrator.get_all_agents()}


def _patch_scenario(scenario, n_adversarial: int, audit_prob: float):
    """Return a deep-copied scenario with patched agent counts and audit prob."""
    patched = copy.deepcopy(scenario)

    # Rebuild agent_specs: keep honest=3, opportunistic=2, set adversarial=n
    new_specs = []
    for spec in patched.agent_specs:
        if spec.get("type") == "adversarial":
            new_specs.append({"type": "adversarial", "count": n_adversarial})
        else:
            new_specs.append(spec)
    patched.agent_specs = new_specs

    # Patch audit probability on governance config
    gov = patched.orchestrator_config.governance_config
    if gov is not None:
        gov.audit_enabled = True
        gov.audit_probability = audit_prob

    return patched


def _report_to_dict(report: CollusionReport) -> dict:
    """Serialize a CollusionReport to a JSON-friendly dict."""
    return {
        "ecosystem_collusion_risk": report.ecosystem_collusion_risk,
        "avg_pair_collusion_score": report.avg_pair_collusion_score,
        "max_pair_collusion_score": report.max_pair_collusion_score,
        "n_flagged_pairs": report.n_flagged_pairs,
        "n_flagged_groups": report.n_flagged_groups,
        "agent_collusion_risk": report.agent_collusion_risk,
        "suspicious_pairs": [
            {
                "agent_a": p.agent_a,
                "agent_b": p.agent_b,
                "collusion_score": p.collusion_score,
                "interaction_count": p.interaction_count,
                "benefit_correlation": p.benefit_correlation,
                "mutual_acceptance_rate": p.mutual_acceptance_rate,
            }
            for p in report.suspicious_pairs
        ],
        "suspicious_groups": [
            {
                "members": sorted(g.members),
                "collusion_score": g.collusion_score,
                "internal_interaction_rate": g.internal_interaction_rate,
                "avg_internal_p": g.avg_internal_p,
                "avg_external_p": g.avg_external_p,
            }
            for g in report.suspicious_groups
        ],
    }


def _run_single_config(
    scenario,
    n_adversarial: int,
    audit_prob: float,
    config_idx: int,
    total_configs: int,
) -> dict:
    """Run one sweep config and return results dict."""
    label = f"adv={n_adversarial}, audit={audit_prob:.2f}"
    print(f"\n  [{config_idx}/{total_configs}] Running: {label}")
    print(f"  {'-' * 60}")

    patched = _patch_scenario(scenario, n_adversarial, audit_prob)
    orchestrator = build_orchestrator(patched)
    config = patched.orchestrator_config
    agent_types = _agent_type_map(orchestrator)
    agent_ids = [a.agent_id for a in orchestrator.get_all_agents()]

    # Identify ground-truth adversarial agents for accuracy metrics
    true_adversarial = {
        aid for aid, atype in agent_types.items()
        if atype == AgentType.ADVERSARIAL
    }
    true_honest = {
        aid for aid, atype in agent_types.items()
        if atype != AgentType.ADVERSARIAL
    }

    # CollusionDetector with scenario's governance thresholds
    gov = config.governance_config
    detector = CollusionDetector(
        frequency_threshold=gov.collusion_frequency_threshold if gov else 2.0,
        benefit_correlation_threshold=(
            gov.collusion_correlation_threshold if gov else 0.7
        ),
        min_interactions_for_analysis=gov.collusion_min_interactions if gov else 3,
        collusion_score_threshold=gov.collusion_score_threshold if gov else 0.5,
        seed=config.seed,
    )

    # --- Wire MetricsAggregator ---
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=f"collusion_adv{n_adversarial}_audit{audit_prob:.2f}",
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    # Tracking state
    all_interactions = []
    epoch_reports: list[CollusionReport] = []
    first_detection_epoch: int | None = None

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        all_interactions.append(interaction)
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

    orchestrator.on_interaction_complete(_on_interaction)

    def _on_epoch_end(epoch_metrics):
        nonlocal first_detection_epoch

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

        # Run collusion analysis on cumulative interactions
        report = detector.analyze(
            interactions=list(all_interactions),
            agent_ids=agent_ids,
        )
        epoch_reports.append(report)

        # Track time-to-first-detection
        if first_detection_epoch is None and report.n_flagged_pairs > 0:
            first_detection_epoch = epoch_metrics.epoch

    orchestrator.on_epoch_end(_on_epoch_end)

    # --- Run simulation ---
    print(
        f"    Agents: {len(agent_ids)} "
        f"(adv={len(true_adversarial)}, honest={len(true_honest)})"
    )
    print(
        f"    Running: {config.n_epochs} epochs x {config.steps_per_epoch} steps"
    )

    metrics_history = orchestrator.run()

    # --- Compute detection accuracy from final report ---
    final_report = epoch_reports[-1] if epoch_reports else CollusionReport()

    # Agents flagged as suspicious (risk > threshold)
    threshold = gov.collusion_score_threshold if gov else 0.5
    flagged_agents = {
        aid
        for aid, risk in final_report.agent_collusion_risk.items()
        if risk >= threshold
    }

    # True positives: adversarial agents correctly flagged
    true_positives = flagged_agents & true_adversarial
    # False positives: honest agents incorrectly flagged
    false_positives = flagged_agents & true_honest

    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(true_adversarial - flagged_agents)

    # Detection accuracy: fraction of adversarial agents correctly flagged
    detection_accuracy = (
        tp_count / len(true_adversarial) if true_adversarial else 0.0
    )
    # False positive rate: fraction of honest agents incorrectly flagged
    false_positive_rate = (
        fp_count / len(true_honest) if true_honest else 0.0
    )
    # Precision
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0

    # --- Epoch table ---
    print()
    print(f"    {'Ep':<4} {'Intx':<6} {'Pairs':<6} {'Groups':<7} "
          f"{'MaxScore':<9} {'EcoRisk':<8} {'Flagged':<8}")
    print(f"    {'-' * 52}")

    for _epoch_idx, (m, report) in enumerate(
        zip(metrics_history, epoch_reports, strict=False)
    ):
        n_flagged = sum(
            1 for r in report.agent_collusion_risk.values()
            if r >= threshold
        )
        print(
            f"    {m.epoch:<4} "
            f"{m.total_interactions:<6} "
            f"{report.n_flagged_pairs:<6} "
            f"{report.n_flagged_groups:<7} "
            f"{report.max_pair_collusion_score:<9.3f} "
            f"{report.ecosystem_collusion_risk:<8.3f} "
            f"{n_flagged:<8}"
        )

    print(f"    {'-' * 52}")

    # --- Summary ---
    total_interactions = sum(m.total_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0.0
    )

    print(f"    Total interactions: {total_interactions}")
    print(f"    Avg toxicity: {avg_toxicity:.4f}")
    print(f"    Detection accuracy: {detection_accuracy:.2%} "
          f"(TP={tp_count}, FN={fn_count})")
    print(f"    False positive rate: {false_positive_rate:.2%} "
          f"(FP={fp_count}/{len(true_honest)})")
    print(f"    Precision: {precision:.2%}")
    print(f"    Time-to-first-detection: "
          f"{'epoch ' + str(first_detection_epoch) if first_detection_epoch is not None else 'never'}")
    print(f"    Final ecosystem risk: {final_report.ecosystem_collusion_risk:.3f}")

    # Build result record
    history = aggregator.end_simulation()

    return {
        "n_adversarial": n_adversarial,
        "audit_probability": audit_prob,
        "seed": config.seed,
        "n_epochs": config.n_epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "total_interactions": total_interactions,
        "avg_toxicity": avg_toxicity,
        "detection_accuracy": detection_accuracy,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "time_to_first_detection": first_detection_epoch,
        "final_ecosystem_risk": final_report.ecosystem_collusion_risk,
        "final_n_flagged_pairs": final_report.n_flagged_pairs,
        "final_n_flagged_groups": final_report.n_flagged_groups,
        "final_max_pair_score": final_report.max_pair_collusion_score,
        "final_avg_pair_score": final_report.avg_pair_collusion_score,
        "epoch_reports": [_report_to_dict(r) for r in epoch_reports],
        "history": history,
    }


def main():
    scenario_path = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "scenarios/collusion_detection.yaml"
    )
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Collusion Detection Study")
    print("Clique detection and statistical tests sweep")
    print("=" * 78)
    print()

    # Load base scenario
    print(f"Loading base scenario: {scenario_path}")
    base_scenario = load_scenario(scenario_path)
    print(f"  ID: {base_scenario.scenario_id}")
    print(f"  Description: {base_scenario.description}")
    print()

    # Sweep grid
    configs = [
        (n_adv, audit_p)
        for n_adv in ADVERSARIAL_COUNTS
        for audit_p in AUDIT_PROBABILITIES
    ]
    total = len(configs)
    print(f"Sweep: {len(ADVERSARIAL_COUNTS)} adversarial counts "
          f"x {len(AUDIT_PROBABILITIES)} audit probs = {total} configs")
    print(f"  Adversarial counts: {ADVERSARIAL_COUNTS}")
    print(f"  Audit probabilities: {AUDIT_PROBABILITIES}")
    print()

    # --- Run sweep ---
    results = []
    for idx, (n_adv, audit_p) in enumerate(configs, 1):
        result = _run_single_config(
            base_scenario, n_adv, audit_p, idx, total
        )
        results.append(result)

    # --- Comparison table ---
    print()
    print("=" * 78)
    print("Sweep Comparison Table")
    print("=" * 78)
    header = (
        f"{'Adv':<5} {'Audit':<7} {'DetAcc':<8} {'FPR':<8} "
        f"{'Prec':<8} {'T2D':<6} {'Pairs':<6} {'Groups':<7} "
        f"{'EcoRisk':<8} {'MaxScr':<8} {'Toxic':<7}"
    )
    print(header)
    print("-" * 78)

    for r in results:
        t2d = str(r["time_to_first_detection"]) if r["time_to_first_detection"] is not None else "-"
        print(
            f"{r['n_adversarial']:<5} "
            f"{r['audit_probability']:<7.2f} "
            f"{r['detection_accuracy']:<8.2%} "
            f"{r['false_positive_rate']:<8.2%} "
            f"{r['precision']:<8.2%} "
            f"{t2d:<6} "
            f"{r['final_n_flagged_pairs']:<6} "
            f"{r['final_n_flagged_groups']:<7} "
            f"{r['final_ecosystem_risk']:<8.3f} "
            f"{r['final_max_pair_score']:<8.3f} "
            f"{r['avg_toxicity']:<7.4f}"
        )

    print("-" * 78)
    print()

    # --- Summary statistics ---
    print("Summary Statistics:")
    print("-" * 78)

    # Group by adversarial count
    by_adv: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_adv[r["n_adversarial"]].append(r)

    print("  By adversarial count:")
    for n_adv in ADVERSARIAL_COUNTS:
        group = by_adv[n_adv]
        avg_acc = sum(r["detection_accuracy"] for r in group) / len(group)
        avg_fpr = sum(r["false_positive_rate"] for r in group) / len(group)
        avg_risk = sum(r["final_ecosystem_risk"] for r in group) / len(group)
        print(
            f"    adv={n_adv}: avg_accuracy={avg_acc:.2%}, "
            f"avg_FPR={avg_fpr:.2%}, avg_eco_risk={avg_risk:.3f}"
        )
    print()

    # Group by audit probability
    by_audit: dict[float, list[dict]] = defaultdict(list)
    for r in results:
        by_audit[r["audit_probability"]].append(r)

    print("  By audit probability:")
    for audit_p in AUDIT_PROBABILITIES:
        group = by_audit[audit_p]
        avg_acc = sum(r["detection_accuracy"] for r in group) / len(group)
        avg_fpr = sum(r["false_positive_rate"] for r in group) / len(group)
        avg_risk = sum(r["final_ecosystem_risk"] for r in group) / len(group)
        print(
            f"    audit={audit_p:.2f}: avg_accuracy={avg_acc:.2%}, "
            f"avg_FPR={avg_fpr:.2%}, avg_eco_risk={avg_risk:.3f}"
        )
    print()

    # Best and worst configs
    best_acc = max(results, key=lambda r: r["detection_accuracy"])
    worst_fpr = max(results, key=lambda r: r["false_positive_rate"])
    fastest = min(
        (r for r in results if r["time_to_first_detection"] is not None),
        key=lambda r: r["time_to_first_detection"],
        default=None,
    )

    print("  Best detection accuracy: "
          f"adv={best_acc['n_adversarial']}, audit={best_acc['audit_probability']:.2f} "
          f"-> {best_acc['detection_accuracy']:.2%}")
    print("  Highest FPR: "
          f"adv={worst_fpr['n_adversarial']}, audit={worst_fpr['audit_probability']:.2f} "
          f"-> {worst_fpr['false_positive_rate']:.2%}")
    if fastest:
        print("  Fastest detection: "
              f"adv={fastest['n_adversarial']}, audit={fastest['audit_probability']:.2f} "
              f"-> epoch {fastest['time_to_first_detection']}")
    print()

    # --- Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = base_scenario.orchestrator_config.seed or 0
    export_dir = Path("runs") / f"{timestamp}_collusion_study_seed{seed}"

    # 1. Standard history export (from last config)
    last_history = results[-1]["history"]
    export_path = export_dir / "history.json"
    export_to_json(last_history, export_path, include_events=True)

    # 2. Full sweep results (without history objects)
    sweep_path = export_dir / "collusion_sweep_results.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)

    exportable_results = []
    for r in results:
        export_r = {k: v for k, v in r.items() if k != "history"}
        exportable_results.append(export_r)

    with open(sweep_path, "w") as f:
        json.dump(
            {
                "scenario_id": base_scenario.scenario_id,
                "base_seed": seed,
                "adversarial_counts": ADVERSARIAL_COUNTS,
                "audit_probabilities": AUDIT_PROBABILITIES,
                "n_configs": total,
                "results": exportable_results,
            },
            f,
            indent=2,
        )

    print(f"Exported run history: {export_path}")
    print(f"Exported sweep results: {sweep_path}")
    print(f"  {total} sweep configurations")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

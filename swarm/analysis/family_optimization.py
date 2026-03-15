"""Cross-scenario family optimization for autoresearch.

Runs autoresearch across multiple scenarios simultaneously, evaluating each
governance mutation against the entire family. Identifies lever settings that
are robust across scenarios and aggregates findings with explicit boundary
conditions.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swarm.analysis.autoresearch import (
    Backpressure,
    EvalSummary,
    ObjectiveSpec,
    _guardrails_ok,
    _is_better,
    _mutate_with_dedup,
    _preflight_check,
    _prevalidate,
    _validate_scenario_name,
    evaluate_candidate,
    parse_objective,
)
from swarm.knowledge.lesson_store import Lesson, LessonStore
from swarm.knowledge.run_envelope import RunEnvelope, write_run_yaml
from swarm.scenarios.loader import load_scenario


@dataclass
class FamilyMember:
    """A scenario in the family with its loaded config."""

    name: str
    path: Path
    scenario: Any  # ScenarioConfig from loader


@dataclass
class FamilyEvalSummary:
    """Aggregated metrics across all family members for one candidate."""

    per_scenario: dict[str, EvalSummary]
    mean_metrics: dict[str, float]
    worst_scenario: str
    worst_value: float


def _aggregate_family_eval(
    per_scenario: dict[str, EvalSummary],
    primary_metric: str,
    direction: str,
) -> FamilyEvalSummary:
    """Aggregate per-scenario evals into a family-level summary."""
    all_metrics: dict[str, list[float]] = {}
    for _name, es in per_scenario.items():
        for k, v in es.metrics.items():
            all_metrics.setdefault(k, []).append(v)

    mean_metrics = {k: sum(vs) / len(vs) for k, vs in all_metrics.items()}

    # Find worst scenario for primary metric
    worst_name = ""
    worst_val = float("inf") if direction == "maximize" else float("-inf")
    for name, es in per_scenario.items():
        val = es.metrics.get(primary_metric, 0.0)
        if direction == "maximize" and val < worst_val:
            worst_val = val
            worst_name = name
        elif direction == "minimize" and val > worst_val:
            worst_val = val
            worst_name = name

    return FamilyEvalSummary(
        per_scenario=per_scenario,
        mean_metrics=mean_metrics,
        worst_scenario=worst_name,
        worst_value=worst_val,
    )


def evaluate_family(
    members: list[FamilyMember],
    governance_config: Any,
    seeds: list[int],
    eval_epochs: int,
    eval_steps: int,
    objective: ObjectiveSpec,
) -> FamilyEvalSummary:
    """Evaluate a governance config across all family members."""
    per_scenario: dict[str, EvalSummary] = {}
    for member in members:
        scenario = copy.deepcopy(member.scenario)
        # Apply the candidate governance config
        for field_name in scenario.orchestrator_config.governance_config.model_fields:
            val = getattr(governance_config, field_name, None)
            if val is not None:
                setattr(
                    scenario.orchestrator_config.governance_config,
                    field_name,
                    val,
                )

        result = evaluate_candidate(
            scenario=scenario,
            seeds=seeds,
            eval_epochs=eval_epochs,
            eval_steps=eval_steps,
            objective=objective,
        )
        per_scenario[member.name] = result

    return _aggregate_family_eval(per_scenario, objective.primary_metric, objective.primary_direction)


def cmd_family_optimize(args: argparse.Namespace) -> int:
    """Run cross-scenario family optimization."""
    objective = parse_objective(args.objective)

    # Load all scenarios in the family
    scenario_paths = [Path(p.strip()) for p in args.scenarios.split(",") if p.strip()]
    if not scenario_paths:
        print("Error: --scenarios must provide at least one scenario path")
        return 1

    members: list[FamilyMember] = []
    for sp in scenario_paths:
        if not sp.exists():
            print(f"Error: scenario file not found: {sp}")
            return 1
        scenario = load_scenario(sp)
        members.append(FamilyMember(name=sp.stem, path=sp, scenario=scenario))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        print("Error: --seeds must provide at least one seed")
        return 1

    rng = random.Random(args.random_seed)
    raw_family = args.family_name or "_".join(m.name for m in members[:3])
    family_name = _validate_scenario_name(raw_family)

    # Use lesson store keyed to the family
    store: LessonStore | None = None
    if not getattr(args, "no_lessons", False):
        store = LessonStore(root=args.export_root, scenario=f"family_{family_name}")
        prior = store.summary()
        if prior["total_lessons"] > 0:
            print(
                f"[family-opt] loaded {prior['total_lessons']} prior lessons "
                f"({prior['accepted']} accepted)"
            )

    backpressure = Backpressure(
        window=getattr(args, "bp_window", 5),
        min_accepts=getattr(args, "bp_min_accepts", 1),
    )

    print(f"[family-opt] family: {', '.join(m.name for m in members)}")
    print(f"[family-opt] objective: {objective.primary_metric} ({objective.primary_direction})")

    # Use the first scenario as the governance template
    reference = copy.deepcopy(members[0].scenario)
    gov_config = reference.orchestrator_config.governance_config

    # Evaluate baseline across family
    baseline = evaluate_family(
        members, gov_config, seeds, args.eval_epochs, args.eval_steps, objective,
    )
    baseline_mean = baseline.mean_metrics.get(objective.primary_metric, 0.0)

    print(
        textwrap.dedent(
            f"""
        [family-opt] baseline:
          metric={objective.primary_metric}
          mean={baseline_mean:.6f}
          worst={baseline.worst_value:.6f} ({baseline.worst_scenario})
        """
        ).strip()
    )

    best_gov = copy.deepcopy(gov_config)
    best = baseline
    best_mean = baseline_mean

    ledger: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "primary_metric": objective.primary_metric,
            "primary_direction": objective.primary_direction,
            "min_improvement": objective.min_improvement,
            "guardrails": [g.__dict__ for g in objective.guardrails],
        },
        "family": [str(m.path) for m in members],
        "seeds": seeds,
        "baseline_mean": baseline.mean_metrics,
        "baseline_per_scenario": {
            name: es.metrics for name, es in baseline.per_scenario.items()
        },
        "iterations": [],
    }

    accepted_count = 0

    for i in range(1, args.iterations + 1):
        # Mutate governance on the reference scenario (just to drive the mutation)
        candidate_ref = copy.deepcopy(reference)
        candidate_ref.orchestrator_config.governance_config = copy.deepcopy(best_gov)
        param, old, new = _mutate_with_dedup(candidate_ref, rng, store)
        candidate_gov = candidate_ref.orchestrator_config.governance_config

        # Preflight
        pf_ok, pf_reason = _preflight_check(candidate_ref)
        if not pf_ok:
            print(f"[family-opt] iter={i:03d} SKIP {pf_reason}")
            ledger["iterations"].append({
                "iteration": i,
                "mutation": {"param": param, "old": old, "new": new},
                "skipped": pf_reason,
            })
            backpressure.record(False)
            continue

        # Prevalidate on first member only
        pv_ok, pv_reason = _prevalidate(
            candidate_ref, seeds[0], args.eval_epochs, args.eval_steps, objective,
        )
        if not pv_ok:
            print(f"[family-opt] iter={i:03d} SKIP {pv_reason}")
            ledger["iterations"].append({
                "iteration": i,
                "mutation": {"param": param, "old": old, "new": new},
                "skipped": pv_reason,
            })
            backpressure.record(False)
            continue

        # Full family evaluation
        candidate_eval = evaluate_family(
            members, candidate_gov, seeds, args.eval_epochs, args.eval_steps, objective,
        )
        cand_mean = candidate_eval.mean_metrics.get(objective.primary_metric, 0.0)

        better = _is_better(
            cand_mean, best_mean, objective.primary_direction, objective.min_improvement,
        )

        # Check guardrails on mean metrics
        best_summary = EvalSummary(primary_metric=objective.primary_metric, metrics=best.mean_metrics)
        cand_summary = EvalSummary(primary_metric=objective.primary_metric, metrics=candidate_eval.mean_metrics)
        guards_ok, guardrail_errors = _guardrails_ok(best_summary, cand_summary, objective.guardrails)

        accepted = bool(better and guards_ok)
        if accepted:
            best_gov = copy.deepcopy(candidate_gov)
            best = candidate_eval
            best_mean = cand_mean
            accepted_count += 1

        # Record lesson
        if store is not None:
            store.add(Lesson(
                param=param,
                old_value=old,
                new_value=new,
                accepted=accepted,
                primary_metric=objective.primary_metric,
                primary_value=cand_mean,
                baseline_value=best_mean if not accepted else (best_mean - (cand_mean - best_mean)),
                iteration=i,
                guardrail_errors=guardrail_errors,
                scenario=f"family_{family_name}",
                tags=[m.name for m in members],
            ))

        record: dict[str, Any] = {
            "iteration": i,
            "mutation": {"param": param, "old": old, "new": new},
            "candidate_mean": candidate_eval.mean_metrics,
            "candidate_per_scenario": {
                name: es.metrics for name, es in candidate_eval.per_scenario.items()
            },
            "worst_scenario": candidate_eval.worst_scenario,
            "accepted": accepted,
            "guardrail_errors": guardrail_errors,
        }
        ledger["iterations"].append(record)
        print(
            f"[family-opt] iter={i:03d} {param}: {old} -> {new} "
            f"mean={cand_mean:.6f} worst={candidate_eval.worst_value:.6f}({candidate_eval.worst_scenario}) "
            f"accepted={'yes' if accepted else 'no'}"
        )

        backpressure.record(accepted)
        if backpressure.should_halt():
            print(f"[family-opt] plateau detected ({backpressure.status()}), halting early")
            break

    ledger["best_mean"] = best.mean_metrics
    ledger["best_per_scenario"] = {
        name: es.metrics for name, es in best.per_scenario.items()
    }
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.export_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "family_summary.json"
    summary_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[family-opt] wrote {summary_path}")

    # Emit run.yaml
    improvement = (
        baseline_mean - best_mean
        if objective.primary_direction == "minimize"
        else best_mean - baseline_mean
    )
    findings = []
    if improvement > 0:
        findings.append(
            f"mean {objective.primary_metric} improved by {improvement:.4f} "
            f"across {len(members)} scenarios"
        )

    # Report per-scenario boundary conditions
    boundary_notes = []
    for name, es in best.per_scenario.items():
        val = es.metrics.get(objective.primary_metric, 0.0)
        base_val = baseline.per_scenario[name].metrics.get(objective.primary_metric, 0.0)
        delta = base_val - val if objective.primary_direction == "minimize" else val - base_val
        if delta < 0:
            boundary_notes.append(f"{name}: regressed by {abs(delta):.4f}")
        else:
            boundary_notes.append(f"{name}: improved by {delta:.4f}")

    if boundary_notes:
        findings.append("per-scenario: " + "; ".join(boundary_notes))

    envelope = RunEnvelope(
        run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_family_{family_name}",
        scenario_ref=",".join(str(m.path) for m in members),
        hypothesis=f"Robust governance tuning for {objective.primary_metric} across {family_name}",
        seeds=seeds,
        total_iterations=len(ledger["iterations"]),
        accepted_iterations=accepted_count,
        primary_metric=objective.primary_metric,
        primary_result=f"mean={best_mean:.6f}",
        baseline_value=baseline_mean,
        best_value=best_mean,
        tags=["autoresearch", "family-optimization", family_name] + [m.name for m in members],
        significant_findings=findings,
        artifacts={"family_summary": str(summary_path)},
    )
    run_yaml_path = write_run_yaml(envelope, out_dir)
    print(f"[family-opt] wrote {run_yaml_path}")

    # Print robust governance summary
    print()
    print("[family-opt] robust governance config:")
    for field_name in best_gov.model_fields:
        val = getattr(best_gov, field_name)
        baseline_val = getattr(members[0].scenario.orchestrator_config.governance_config, field_name)
        marker = " *" if val != baseline_val else ""
        print(f"  {field_name}: {val}{marker}")

    if store is not None:
        s = store.summary()
        print(
            f"[family-opt] lessons: {s['total_lessons']} total, "
            f"{s['accepted']} accepted"
        )

    return 0

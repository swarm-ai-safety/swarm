"""Autoresearch loop for iterative scenario governance tuning.

This module implements a lightweight objective->mutate->evaluate loop intended
for local experimentation. The current implementation focuses on mutating a
small set of governance levers and selecting candidates that improve a primary
metric while honoring guardrail constraints.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from swarm.scenarios.loader import build_orchestrator, load_scenario


@dataclass
class Guardrail:
    """Constraint on metric regression relative to baseline."""

    metric: str
    max_regression: float


@dataclass
class ObjectiveSpec:
    """Autoresearch objective specification parsed from objective files."""

    primary_metric: str
    primary_direction: str = "minimize"
    min_improvement: float = 0.0
    guardrails: list[Guardrail] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Averaged metrics for one candidate over seed panel."""

    primary_metric: str
    metrics: dict[str, float]


NUMERIC_TUNABLES: dict[str, tuple[float, float, float]] = {
    "transaction_tax_rate": (0.0, 0.6, 0.02),
    "audit_probability": (0.0, 1.0, 0.05),
    "audit_threshold_p": (0.1, 0.95, 0.05),
    "freeze_threshold_toxicity": (0.05, 0.95, 0.05),
    "incoherence_friction_rate": (0.0, 0.4, 0.02),
}

BOOLEAN_TOGGLES: tuple[str, ...] = (
    "audit_enabled",
    "circuit_breaker_enabled",
    "incoherence_friction_enabled",
)


class ObjectiveParseError(ValueError):
    """Raised when an objective file cannot be parsed."""


def parse_objective(path: str | Path) -> ObjectiveSpec:
    """Parse an objective spec from YAML/JSON or markdown fenced YAML."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    data: dict[str, Any] | None = None
    if p.suffix.lower() in {".yaml", ".yml", ".json"}:
        data = _parse_structured_text(text, p.suffix.lower())
    else:
        data = _parse_markdown_objective(text)

    if not isinstance(data, dict):
        raise ObjectiveParseError("Objective content must decode to a mapping")

    primary_metric = str(data.get("primary_metric", "")).strip()
    if not primary_metric:
        raise ObjectiveParseError("Missing required key: primary_metric")

    primary_direction = str(data.get("primary_direction", "minimize")).strip().lower()
    if primary_direction not in {"minimize", "maximize"}:
        raise ObjectiveParseError("primary_direction must be 'minimize' or 'maximize'")

    min_improvement = float(data.get("min_improvement", 0.0))

    raw_guardrails = data.get("guardrails", [])
    guardrails: list[Guardrail] = []
    if isinstance(raw_guardrails, list):
        for item in raw_guardrails:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", "")).strip()
            if not metric:
                continue
            guardrails.append(
                Guardrail(metric=metric, max_regression=float(item.get("max_regression", 0.0)))
            )

    return ObjectiveSpec(
        primary_metric=primary_metric,
        primary_direction=primary_direction,
        min_improvement=min_improvement,
        guardrails=guardrails,
    )


def _parse_structured_text(text: str, suffix: str) -> dict[str, Any]:
    if suffix == ".json":
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ObjectiveParseError("JSON content must decode to a mapping")
        return result
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _parse_markdown_objective(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    in_yaml_block = False
    buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") and not in_yaml_block:
            lang = stripped.removeprefix("```").strip().lower()
            if lang in {"yaml", "yml", ""}:
                in_yaml_block = True
                buf = []
            continue
        if stripped.startswith("```") and in_yaml_block:
            parsed = yaml.safe_load("\n".join(buf))
            return parsed if isinstance(parsed, dict) else {}
        if in_yaml_block:
            buf.append(line)

    # fallback: simple key: value lines
    pairs: dict[str, Any] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def evaluate_candidate(
    scenario: Any,
    seeds: list[int],
    eval_epochs: int,
    eval_steps: int,
    objective: ObjectiveSpec,
) -> EvalSummary:
    """Run candidate across seed panel and return averaged final metrics."""
    metric_names = {
        objective.primary_metric,
        "toxicity_rate",
        "quality_gap",
        "total_welfare",
        "avg_payoff",
    }
    acc: dict[str, float] = dict.fromkeys(metric_names, 0.0)
    if not seeds:
        raise ValueError("seeds must not be empty")

    for seed in seeds:
        scenario_copy = copy.deepcopy(scenario)
        scenario_copy.orchestrator_config.seed = seed
        scenario_copy.orchestrator_config.n_epochs = eval_epochs
        scenario_copy.orchestrator_config.steps_per_epoch = eval_steps
        scenario_copy.orchestrator_config.log_path = None
        scenario_copy.orchestrator_config.log_events = False

        orchestrator = build_orchestrator(scenario_copy)
        history = orchestrator.run()
        if not history:
            continue
        final = history[-1]
        data = final.to_dict()
        for metric in metric_names:
            acc[metric] += float(data.get(metric, 0.0))

    n = float(len(seeds))
    averaged = {k: v / n for k, v in acc.items()}
    return EvalSummary(primary_metric=objective.primary_metric, metrics=averaged)


def _is_better(candidate: float, baseline: float, direction: str, min_improvement: float) -> bool:
    improvement = baseline - candidate if direction == "minimize" else candidate - baseline
    return improvement >= min_improvement


def _guardrails_ok(baseline: EvalSummary, candidate: EvalSummary, guardrails: list[Guardrail]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for g in guardrails:
        base = baseline.metrics.get(g.metric, 0.0)
        cand = candidate.metrics.get(g.metric, 0.0)
        regression = cand - base
        if regression > g.max_regression:
            errors.append(
                f"{g.metric} regressed by {regression:.4f} > allowed {g.max_regression:.4f}"
            )
    return (len(errors) == 0, errors)


def _mutate_governance(scenario: Any, rng: random.Random) -> tuple[str, Any, Any]:
    """Apply a single random governance mutation and return mutation details."""
    gov = scenario.orchestrator_config.governance_config
    if rng.random() < 0.7:
        param = rng.choice(list(NUMERIC_TUNABLES.keys()))
        low, high, step = NUMERIC_TUNABLES[param]
        old = float(getattr(gov, param))
        delta = step * rng.choice([-1, 1])
        new = min(high, max(low, old + delta))
        setattr(gov, param, round(new, 6))
        return param, old, new

    param = rng.choice(list(BOOLEAN_TOGGLES))
    old = bool(getattr(gov, param))
    new = not old
    setattr(gov, param, new)
    return param, old, new


def cmd_autoresearch(args: argparse.Namespace) -> int:
    """Run a local autoresearch loop for governance tuning."""
    objective = parse_objective(args.objective)
    scenario = load_scenario(Path(args.scenario))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        print("Error: --seeds must provide at least one seed")
        return 1

    rng = random.Random(args.random_seed)

    baseline = evaluate_candidate(
        scenario=scenario,
        seeds=seeds,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        objective=objective,
    )

    best_scenario = copy.deepcopy(scenario)
    best = baseline

    ledger: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "primary_metric": objective.primary_metric,
            "primary_direction": objective.primary_direction,
            "min_improvement": objective.min_improvement,
            "guardrails": [g.__dict__ for g in objective.guardrails],
        },
        "scenario": args.scenario,
        "seeds": seeds,
        "baseline": baseline.metrics,
        "iterations": [],
    }

    print(textwrap.dedent(
        f"""
        [autoresearch] baseline:
          metric={objective.primary_metric}
          value={baseline.metrics.get(objective.primary_metric, 0.0):.6f}
        """
    ).strip())

    for i in range(1, args.iterations + 1):
        candidate = copy.deepcopy(best_scenario)
        param, old, new = _mutate_governance(candidate, rng)
        candidate_eval = evaluate_candidate(
            scenario=candidate,
            seeds=seeds,
            eval_epochs=args.eval_epochs,
            eval_steps=args.eval_steps,
            objective=objective,
        )

        cand_value = candidate_eval.metrics.get(objective.primary_metric, 0.0)
        best_value = best.metrics.get(objective.primary_metric, 0.0)

        better = _is_better(cand_value, best_value, objective.primary_direction, objective.min_improvement)
        guards_ok, guardrail_errors = _guardrails_ok(best, candidate_eval, objective.guardrails)
        accepted = bool(better and guards_ok)
        if accepted:
            best_scenario = candidate
            best = candidate_eval

        record = {
            "iteration": i,
            "mutation": {"param": param, "old": old, "new": new},
            "candidate_metrics": candidate_eval.metrics,
            "accepted": accepted,
            "guardrail_errors": guardrail_errors,
        }
        ledger["iterations"].append(record)
        print(
            f"[autoresearch] iter={i:03d} {param}: {old} -> {new} "
            f"candidate={cand_value:.6f} accepted={'yes' if accepted else 'no'}"
        )

    ledger["best"] = best.metrics
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.export_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[autoresearch] wrote {summary_path}")

    if args.auto_commit:
        import subprocess

        commit_path = str(summary_path)
        msg = f"autoresearch: optimize {objective.primary_metric} on {Path(args.scenario).name}"
        subprocess.run(["git", "add", commit_path], check=False)
        subprocess.run(["git", "commit", "-m", msg], check=False)

    return 0

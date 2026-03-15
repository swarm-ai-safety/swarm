"""Autoresearch loop for iterative scenario governance tuning.

This module implements a lightweight objective->mutate->evaluate loop intended
for local experimentation. The current implementation focuses on mutating a
small set of governance levers and selecting candidates that improve a primary
metric while honoring guardrail constraints.

Integration with the knowledge system:
- LessonStore: persists mutation trials across sessions so redundant mutations
  are skipped and prior knowledge informs future proposals.
- RunEnvelope: emits a run.yaml metadata envelope on completion so the
  swarm-artifacts synthesis pipeline can auto-generate vault notes.
- Backpressure: detects score plateaus and halts early when no progress is
  being made.
- Staged validation: preflight (schema check) and prevalidation (1-seed smoke
  test) stages reject bad mutations before the full seed-panel evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from swarm.knowledge.lesson_store import Lesson, LessonStore
from swarm.knowledge.run_envelope import RunEnvelope, write_run_yaml
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

# Maximum retries when all proposed mutations have been tried before
_MAX_SKIP_RETRIES = 20

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _validate_scenario_name(name: str) -> str:
    """Validate scenario name is safe for use in filesystem paths."""
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"Scenario name must match [a-zA-Z0-9._-]+, got: {name!r}"
        )
    return name


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
    if not re.match(r"^[a-zA-Z0-9_]+$", primary_metric):
        raise ObjectiveParseError(
            f"primary_metric must match [a-zA-Z0-9_]+, got: {primary_metric!r}"
        )

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
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    parsed = yaml.safe_load(text)
    result: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    return result


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
    scenario,
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


def _guardrails_ok(
    baseline: EvalSummary,
    candidate: EvalSummary,
    guardrails: list[Guardrail],
) -> tuple[bool, list[str]]:
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


def _mutate_governance(scenario, rng: random.Random) -> tuple[str, Any, Any]:
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


def _mutate_with_dedup(
    scenario,
    rng: random.Random,
    store: LessonStore | None,
) -> tuple[str, Any, Any]:
    """Mutate governance, skipping mutations the lesson store has already tried.

    Falls back to allowing duplicates after ``_MAX_SKIP_RETRIES`` attempts so
    the loop never stalls entirely.
    """
    if store is None:
        return _mutate_governance(scenario, rng)

    for _ in range(_MAX_SKIP_RETRIES):
        candidate = copy.deepcopy(scenario)
        param, old, new = _mutate_governance(candidate, rng)
        if not store.was_tried(param, new):
            # Apply mutation to the real scenario
            gov = scenario.orchestrator_config.governance_config
            setattr(gov, param, new if isinstance(new, bool) else round(float(new), 6))
            return param, old, new

    # Exhausted retries — allow a duplicate
    return _mutate_governance(scenario, rng)


# ---------------------------------------------------------------------------
# Staged validation helpers
# ---------------------------------------------------------------------------

def _preflight_check(scenario) -> tuple[bool, str]:
    """Stage 1: Validate the mutated scenario config is structurally sound."""
    gov = scenario.orchestrator_config.governance_config
    for param, (low, high, _step) in NUMERIC_TUNABLES.items():
        val = float(getattr(gov, param, 0.0))
        if val < low or val > high:
            return False, f"preflight: {param}={val} outside [{low}, {high}]"
    return True, ""


def _prevalidate(
    scenario,
    seed: int,
    eval_epochs: int,
    eval_steps: int,
    objective: ObjectiveSpec,
) -> tuple[bool, str]:
    """Stage 2: Run a single-seed smoke test; reject if crash or NaN."""
    try:
        result = evaluate_candidate(
            scenario=scenario,
            seeds=[seed],
            eval_epochs=eval_epochs,
            eval_steps=eval_steps,
            objective=objective,
        )
    except Exception as exc:
        return False, f"prevalidation crash: {exc}"

    val = result.metrics.get(objective.primary_metric)
    if val is None:
        return False, "prevalidation: primary metric missing from results"

    import math
    if math.isnan(val) or math.isinf(val):
        return False, f"prevalidation: primary metric is {val}"

    return True, ""


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------

class Backpressure:
    """Detect score plateaus and signal when to stop.

    Tracks the last ``window`` accepted improvements. If fewer than
    ``min_accepts`` occur in a full window, signals halt.
    """

    def __init__(self, window: int = 5, min_accepts: int = 1) -> None:
        self.window = window
        self.min_accepts = min_accepts
        self._recent: deque[bool] = deque(maxlen=window)

    def record(self, accepted: bool) -> None:
        self._recent.append(accepted)

    def should_halt(self) -> bool:
        if len(self._recent) < self.window:
            return False
        return sum(self._recent) < self.min_accepts

    def status(self) -> str:
        accepts = sum(self._recent)
        return f"{accepts}/{len(self._recent)} accepted in window of {self.window}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cmd_autoresearch(args: argparse.Namespace) -> int:
    """Run a local autoresearch loop for governance tuning."""
    objective = parse_objective(args.objective)
    scenario = load_scenario(Path(args.scenario))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        print("Error: --seeds must provide at least one seed")
        return 1

    rng = random.Random(args.random_seed)
    scenario_name = _validate_scenario_name(Path(args.scenario).stem)

    # --- Knowledge integration ---
    store: LessonStore | None = None
    if not getattr(args, "no_lessons", False):
        store = LessonStore(root=args.export_root, scenario=scenario_name)
        prior = store.summary()
        if prior["total_lessons"] > 0:
            print(
                f"[autoresearch] loaded {prior['total_lessons']} prior lessons "
                f"({prior['accepted']} accepted, params: {', '.join(prior['params_tried'])})"
            )

    # --- Backpressure ---
    bp_window = getattr(args, "bp_window", 5)
    bp_min = getattr(args, "bp_min_accepts", 1)
    backpressure = Backpressure(window=bp_window, min_accepts=bp_min)

    baseline = evaluate_candidate(
        scenario=scenario,
        seeds=seeds,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        objective=objective,
    )

    best_scenario = copy.deepcopy(scenario)
    best = baseline
    baseline_value = baseline.metrics.get(objective.primary_metric, 0.0)

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

    print(
        textwrap.dedent(
            f"""
        [autoresearch] baseline:
          metric={objective.primary_metric}
          value={baseline_value:.6f}
        """
        ).strip()
    )

    accepted_count = 0
    skipped_preflight = 0
    skipped_prevalidation = 0

    for i in range(1, args.iterations + 1):
        candidate = copy.deepcopy(best_scenario)
        param, old, new = _mutate_with_dedup(candidate, rng, store)

        # --- Stage 1: Preflight ---
        pf_ok, pf_reason = _preflight_check(candidate)
        if not pf_ok:
            skipped_preflight += 1
            print(f"[autoresearch] iter={i:03d} SKIP {pf_reason}")
            ledger["iterations"].append({
                "iteration": i,
                "mutation": {"param": param, "old": old, "new": new},
                "skipped": pf_reason,
            })
            backpressure.record(False)
            continue

        # --- Stage 2: Prevalidation (1-seed smoke test) ---
        pv_ok, pv_reason = _prevalidate(
            candidate, seeds[0], args.eval_epochs, args.eval_steps, objective
        )
        if not pv_ok:
            skipped_prevalidation += 1
            print(f"[autoresearch] iter={i:03d} SKIP {pv_reason}")
            ledger["iterations"].append({
                "iteration": i,
                "mutation": {"param": param, "old": old, "new": new},
                "skipped": pv_reason,
            })
            backpressure.record(False)
            continue

        # --- Stage 3: Full evaluation ---
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
            accepted_count += 1

        # --- Record lesson ---
        if store is not None:
            store.add(Lesson(
                param=param,
                old_value=old,
                new_value=new,
                accepted=accepted,
                primary_metric=objective.primary_metric,
                primary_value=cand_value,
                baseline_value=best_value,
                iteration=i,
                guardrail_errors=guardrail_errors,
                scenario=scenario_name,
            ))

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

        # --- Backpressure check ---
        backpressure.record(accepted)
        if backpressure.should_halt():
            print(f"[autoresearch] plateau detected ({backpressure.status()}), halting early")
            break

    ledger["best"] = best.metrics
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
    ledger["skipped_preflight"] = skipped_preflight
    ledger["skipped_prevalidation"] = skipped_prevalidation

    out_dir = Path(args.export_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[autoresearch] wrote {summary_path}")

    # --- Emit run.yaml envelope ---
    best_final = best.metrics.get(objective.primary_metric, 0.0)
    improvement = baseline_value - best_final if objective.primary_direction == "minimize" else best_final - baseline_value
    findings = []
    if improvement > 0:
        findings.append(
            f"{objective.primary_metric} improved by {improvement:.4f} "
            f"({baseline_value:.4f} -> {best_final:.4f})"
        )

    envelope = RunEnvelope(
        run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_autoresearch_{scenario_name}",
        scenario_ref=args.scenario,
        hypothesis=f"Optimize {objective.primary_metric} via governance tuning on {scenario_name}",
        seeds=seeds,
        total_iterations=len(ledger["iterations"]),
        accepted_iterations=accepted_count,
        primary_metric=objective.primary_metric,
        primary_result=f"{best_final:.6f}",
        baseline_value=baseline_value,
        best_value=best_final,
        tags=["autoresearch", "governance-tuning", scenario_name],
        significant_findings=findings,
        artifacts={"summary": str(summary_path)},
    )
    run_yaml_path = write_run_yaml(envelope, out_dir)
    print(f"[autoresearch] wrote {run_yaml_path}")

    # --- Lesson store summary ---
    if store is not None:
        s = store.summary()
        print(
            f"[autoresearch] lessons: {s['total_lessons']} total, "
            f"{s['accepted']} accepted, params: {', '.join(s['params_tried'])}"
        )

    if args.auto_commit:
        import subprocess

        commit_files = [str(summary_path), str(run_yaml_path)]
        lessons_path = Path(args.export_root) / scenario_name / "lessons.json"
        if lessons_path.exists():
            commit_files.append(str(lessons_path))
        msg = f"autoresearch: optimize {objective.primary_metric} on {scenario_name}"
        for f in commit_files:
            subprocess.run(["git", "add", f], check=False)
        subprocess.run(["git", "commit", "-m", msg], check=False)

    return 0

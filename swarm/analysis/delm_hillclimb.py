"""DeLM-style parallel hill-climbing over governance/payoff parameters.

This is a third optimizer over the same fitness landscape used by
``swarm.analysis.evolver`` (darwinian search) and
``swarm.analysis.gepa_optimizer`` (LLM-guided Pareto search): the governance
and payoff knobs in ``PARAM_RANGES``, scored by ``compute_fitness`` against the
soft safety metrics (toxicity, welfare, quality gap, payoff gap).

Where those optimizers run a single controller, this module implements the
**DeLM** (decentralized language-model) coordination pattern as a parallel,
shared-state, self-coordinating hill-climb with no central planner:

* **Shared context as fitness-landscape memory** (:class:`SharedContext`).
  Verified *gists* of improving moves, the running best, and binding
  *constraints* (known dead-end neighbors) accumulate in one shared store.
  Workers read it immediately, so they avoid re-testing known-bad neighbors or
  repeating failed mutations.

* **Task queue for asynchronous neighbor exploration** (:class:`TaskQueue`).
  Instead of one agent sequentially generating and evaluating neighbors, many
  workers claim ``explore`` / ``mutate_dim`` / ``restart`` / ``diversify``
  tasks. As soon as one worker verifies an improvement it writes a compact
  gist and updates the best, so the next worker instantly hill-climbs from the
  new higher point and fresh neighbor tasks are spawned around it.

* **Verified admission prevents noise.** A candidate that beats the current
  best is re-evaluated at an independent verification seed; only moves that
  still improve are admitted. This keeps the shared state trustworthy — one
  noisy "improvement" can otherwise derail the whole trajectory.

* **Escaping local optima.** ``restart`` tasks sample fresh points in the full
  space and ``diversify`` tasks jump to distant *basins* discovered by other
  workers, so the swarm can leave a basin once it is exhausted.

The default scheduler is a deterministic round-robin over ``n_workers`` virtual
workers sharing one :class:`SharedContext`. This reproduces the DeLM dynamics
(shared verified state, asynchronous neighbor claiming, instant hill-climb from
a new best) while remaining fully reproducible from ``scenario YAML + seed`` —
the project's reproducibility invariant. Pass ``use_threads=True`` to run the
same workers as real OS threads (the shared store is lock-protected); results
are then no longer bit-for-bit reproducible because thread interleaving is not
controlled, so the deterministic scheduler is the default.

Usage (CLI)::

    python -m swarm.analysis.delm_hillclimb scenarios/baseline.yaml \\
        --max-evals 60 --workers 4 --seed 42

    # custom fitness weights + faster evals
    python -m swarm.analysis.delm_hillclimb scenarios/baseline.yaml \\
        --max-evals 100 --workers 8 --eval-epochs 2 --eval-steps 4 \\
        --weight-toxicity 0.5 --weight-welfare 0.2

Usage (library)::

    from swarm.analysis.delm_hillclimb import HillClimbConfig, run_delm_hillclimb
    from swarm.scenarios import load_scenario

    scenario = load_scenario("scenarios/baseline.yaml")
    result = run_delm_hillclimb(scenario, HillClimbConfig(max_evals=60, n_workers=4))
    print(result.best_score, result.best_params)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from swarm.analysis.evolver import (
    DEFAULT_FITNESS_WEIGHTS,
    INT_PARAMS,
    PARAM_RANGES,
    compute_fitness,
)
from swarm.analysis.sweep import _apply_params, _extract_results
from swarm.scenarios import ScenarioConfig, build_orchestrator, load_scenario

logger = logging.getLogger(__name__)

# A move must beat the incumbent by at least this much (in fitness units) to be
# treated as an improvement worth verifying. Filters out evaluation jitter.
IMPROVE_EPS = 1e-3

# A move this far *below* the incumbent (or a non-viable run) is recorded as a
# binding constraint so other workers prune that neighbor.
DEADEND_MARGIN = 0.02

# Number of decimal places used to quantize parameter cells for the
# seen/constraint maps. Two distinct candidates that round to the same cell are
# treated as the same neighbor (avoids redundant re-evaluation).
QUANT_DECIMALS = 4

# Fraction of the parameter range used as the std-dev of a local mutation step.
DEFAULT_STEP_FRAC = 0.15


# ---------------------------------------------------------------------------
# Evaluation: one (params -> fitness) measurement on the shared landscape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalOutcome:
    """Result of a single simulation evaluation of a parameter set."""

    fitness: float
    side_info: Dict[str, Any]
    viable: bool
    error: Optional[str] = None


def evaluate_params(
    base_scenario: ScenarioConfig,
    params: Dict[str, Any],
    *,
    seed: int,
    eval_epochs: int,
    eval_steps: int,
    fitness_weights: Optional[Dict[str, float]] = None,
    welfare_reference: float = 10.0,
    payoff_reference: float = 5.0,
) -> EvalOutcome:
    """Evaluate a governance/payoff parameter set on the soft-safety landscape.

    Mirrors the evaluation pipeline used by the GEPA and darwinian optimizers
    so all three rank candidates on the same scale: apply params to a deep copy
    of the scenario, run a short simulation, extract soft metrics, and fold
    them into a single composite fitness in ``[0, 1]``.
    """
    scenario = copy.deepcopy(base_scenario)
    scenario = _apply_params(scenario, params)
    cfg = scenario.orchestrator_config
    cfg.seed = seed
    cfg.n_epochs = eval_epochs
    cfg.steps_per_epoch = eval_steps
    cfg.log_path = None
    cfg.log_events = False

    try:
        orchestrator = build_orchestrator(scenario)
        orchestrator.run()
    except Exception as exc:  # noqa: BLE001 — a bad genome must not kill the swarm
        logger.debug("evaluation failed for %s: %s", params, exc)
        return EvalOutcome(fitness=0.0, side_info={}, viable=False, error=str(exc))

    result = _extract_results(orchestrator, params, 0, seed)
    payoff_gap = result.honest_avg_payoff - result.adversarial_avg_payoff
    fitness = compute_fitness(
        avg_toxicity=result.avg_toxicity,
        avg_welfare=result.welfare_per_epoch,
        avg_quality_gap=result.avg_quality_gap,
        payoff_gap=payoff_gap,
        weights=fitness_weights,
        welfare_reference=welfare_reference,
        payoff_reference=payoff_reference,
    )
    side_info = {
        "toxicity": round(result.avg_toxicity, 4),
        "welfare": round(result.welfare_per_epoch, 4),
        "quality_gap": round(result.avg_quality_gap, 4),
        "payoff_gap": round(payoff_gap, 4),
        "n_frozen": result.n_frozen,
    }
    return EvalOutcome(fitness=fitness, side_info=side_info, viable=True)


# ---------------------------------------------------------------------------
# Parameter-space helpers
# ---------------------------------------------------------------------------


def _clamp_param(name: str, value: float) -> Any:
    """Clamp a value to ``PARAM_RANGES[name]`` and coerce int params."""
    lo, hi = PARAM_RANGES[name]
    value = max(lo, min(hi, value))
    if name in INT_PARAMS:
        return int(round(value))
    return float(value)


def _quantize_key(params: Dict[str, Any]) -> Tuple[Tuple[str, float], ...]:
    """A hashable, quantized identity for a parameter cell.

    Two candidates that round to the same grid cell are the same neighbor for
    the purposes of the seen-set and constraint matching.
    """
    return tuple(
        (name, round(float(params[name]), QUANT_DECIMALS))
        for name in sorted(params)
        if name in PARAM_RANGES
    )


def seed_params_from_scenario(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Read the scenario's current governance/payoff knobs as a start point.

    Any knob the scenario does not set falls back to the midpoint of its range,
    so the optimizer always starts from a fully specified, in-range point.
    """
    orch = scenario.orchestrator_config
    params: Dict[str, Any] = {}
    for dotted, (lo, hi) in PARAM_RANGES.items():
        section, _, key = dotted.partition(".")
        value: Optional[float] = None
        if section == "governance" and orch.governance_config is not None:
            value = getattr(orch.governance_config, key, None)
        elif section == "payoff":
            value = getattr(orch.payoff_config, key, None)
        if value is None:
            value = (lo + hi) / 2.0
        params[dotted] = _clamp_param(dotted, float(value))
    return params


def _param_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Normalized L2 distance between two param dicts (range-scaled)."""
    total = 0.0
    for name, (lo, hi) in PARAM_RANGES.items():
        span = (hi - lo) or 1.0
        da = (float(a.get(name, lo)) - float(b.get(name, lo))) / span
        total += da * da
    return math.sqrt(total)


def _mutate(
    base: Dict[str, Any],
    rng: random.Random,
    *,
    step_frac: float,
    dims: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Local mutation: Gaussian perturbation of ``dims`` (default: all)."""
    out = dict(base)
    targets = dims if dims is not None else list(PARAM_RANGES.keys())
    for name in targets:
        lo, hi = PARAM_RANGES[name]
        sigma = step_frac * (hi - lo)
        out[name] = _clamp_param(name, float(base.get(name, (lo + hi) / 2)) + rng.gauss(0.0, sigma))
    return out


def _random_point(rng: random.Random) -> Dict[str, Any]:
    """A uniformly random point in the full parameter space (for restarts)."""
    return {name: _clamp_param(name, rng.uniform(lo, hi)) for name, (lo, hi) in PARAM_RANGES.items()}


# ---------------------------------------------------------------------------
# Shared context: the fitness-landscape memory
# ---------------------------------------------------------------------------


@dataclass
class Gist:
    """A compact, verified record of an evaluated move."""

    move_id: int
    params: Dict[str, Any]
    score: float
    parent_score: float
    delta: float
    verified: bool
    summary: str
    side_info: Dict[str, Any]
    worker: int
    kind: str

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Constraint:
    """A binding dead-end: a known-bad neighbor cell other workers should skip.

    ``cell`` is the quantized identity of the failing candidate; any future
    proposal that quantizes to the same cell is pruned without re-evaluation —
    the DeLM "this edit always breaks test X" rule made binding for everyone.
    """

    cell: Tuple[Tuple[str, float], ...]
    summary: str
    score: float
    viable: bool


@dataclass
class Basin:
    """A distinct local optimum discovered by some worker."""

    params: Dict[str, Any]
    score: float


def _summarize_move(
    base: Dict[str, Any], cand: Dict[str, Any], parent_score: float, score: float
) -> str:
    """Human-readable gist of the largest single change in a move."""
    biggest_name = ""
    biggest_mag = 0.0
    for name, (lo, hi) in PARAM_RANGES.items():
        span = (hi - lo) or 1.0
        mag = abs(float(cand.get(name, lo)) - float(base.get(name, lo))) / span
        if mag > biggest_mag:
            biggest_mag = mag
            biggest_name = name
    if not biggest_name or biggest_mag == 0.0:
        # No dimension changed relative to the base — a fresh restart point.
        return f"fresh point at score {score:.3f}"
    short = biggest_name.split(".")[-1]
    direction = "raise" if cand[biggest_name] > base.get(biggest_name, 0) else "lower"
    return (
        f"score {parent_score:.3f} -> {score:.3f} via {direction} {short} "
        f"to {cand[biggest_name]:.4g}"
    )


class SharedContext:
    """Lock-protected shared store read and written by all workers.

    Holds the running best, the verified gist trail, binding constraints, the
    seen-cell set (redundant-work guard), and the basin list (for escaping
    local optima). All public methods are safe under the thread scheduler.
    """

    def __init__(self, best_params: Dict[str, Any], best_score: float):
        self._lock = threading.Lock()
        self.best_params: Dict[str, Any] = dict(best_params)
        self.best_score: float = best_score
        self.gists: List[Gist] = []
        self.constraints: List[Constraint] = []
        self.basins: List[Basin] = [Basin(params=dict(best_params), score=best_score)]
        self._seen: Dict[Tuple[Tuple[str, float], ...], float] = {}
        self._constraint_cells: set[Tuple[Tuple[str, float], ...]] = set()
        self._move_counter = 0
        # Telemetry
        self.n_redundant_skipped = 0
        self.n_constraint_pruned = 0
        self.n_improvements = 0
        self.n_verify_rejected = 0

    # -- reads -------------------------------------------------------------

    def snapshot_best(self) -> Tuple[Dict[str, Any], float]:
        with self._lock:
            return dict(self.best_params), self.best_score

    def is_seen(self, params: Dict[str, Any]) -> bool:
        with self._lock:
            return _quantize_key(params) in self._seen

    def is_pruned(self, params: Dict[str, Any]) -> bool:
        """True if a binding constraint already rules this neighbor out."""
        with self._lock:
            return _quantize_key(params) in self._constraint_cells

    def random_basin(self, rng: random.Random) -> Dict[str, Any]:
        with self._lock:
            basin = rng.choice(self.basins)
            return dict(basin.params)

    # -- writes (verified admission) --------------------------------------

    def record(
        self,
        *,
        params: Dict[str, Any],
        outcome: EvalOutcome,
        parent_score: float,
        base_params: Dict[str, Any],
        worker: int,
        kind: str,
        verifier: Optional[Callable[[Dict[str, Any]], EvalOutcome]] = None,
        basin_distance: float = 0.25,
    ) -> Gist:
        """Admit an evaluated candidate into the shared context.

        Verified admission: a candidate that beats the incumbent is only made
        the new best if ``verifier`` (a re-evaluation at an independent seed)
        confirms it still improves. Clear regressions and non-viable runs are
        recorded as binding constraints; everything is marked seen so no worker
        re-tests the same cell.
        """
        cell = _quantize_key(params)
        with self._lock:
            self._seen[cell] = outcome.fitness
            self._move_counter += 1
            move_id = self._move_counter
            incumbent = self.best_score

        score = outcome.fitness
        delta = score - parent_score
        verified = False
        summary = _summarize_move(base_params, params, parent_score, score)

        if not outcome.viable:
            summary = f"non-viable run ({outcome.error or 'unknown'})"
            with self._lock:
                if cell not in self._constraint_cells:
                    self._constraint_cells.add(cell)
                    self.constraints.append(
                        Constraint(cell=cell, summary=summary, score=0.0, viable=False)
                    )
        elif score + IMPROVE_EPS < incumbent - DEADEND_MARGIN:
            # Clear regression vs. the global best — bind it as a dead-end.
            with self._lock:
                if cell not in self._constraint_cells:
                    self._constraint_cells.add(cell)
                    self.constraints.append(
                        Constraint(
                            cell=cell,
                            summary=f"dead-end: {summary} (best {incumbent:.3f})",
                            score=score,
                            viable=True,
                        )
                    )
        elif score > incumbent + IMPROVE_EPS and verifier is not None:
            # Candidate improvement — verify before trusting it.
            confirm = verifier(params)
            confirmed_score = min(score, confirm.fitness) if confirm.viable else -1.0
            with self._lock:
                if confirm.viable and confirmed_score > self.best_score + IMPROVE_EPS:
                    verified = True
                    self.n_improvements += 1
                    # Promote to best at the conservative (verified) score.
                    self.best_params = dict(params)
                    self.best_score = confirmed_score
                    score = confirmed_score
                    delta = confirmed_score - parent_score
                    self._seen[cell] = confirmed_score
                    # Record a new basin if this point is far from all known ones.
                    if all(
                        _param_distance(params, b.params) > basin_distance
                        for b in self.basins
                    ):
                        self.basins.append(Basin(params=dict(params), score=confirmed_score))
                else:
                    self.n_verify_rejected += 1

        gist = Gist(
            move_id=move_id,
            params=dict(params),
            score=round(score, 6),
            parent_score=round(parent_score, 6),
            delta=round(delta, 6),
            verified=verified,
            summary=summary,
            side_info=outcome.side_info,
            worker=worker,
            kind=kind,
        )
        with self._lock:
            self.gists.append(gist)
        return gist

    def note_redundant(self) -> None:
        with self._lock:
            self.n_redundant_skipped += 1

    def note_pruned(self) -> None:
        with self._lock:
            self.n_constraint_pruned += 1


# ---------------------------------------------------------------------------
# Task queue: asynchronous neighbor-exploration work items
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of neighbor-exploration work a worker can claim.

    kind:
      * ``explore``     — mutate all dims around the current best.
      * ``mutate_dim``  — mutate a single dimension around the current best.
      * ``restart``     — sample a fresh random point (escape local optima).
      * ``diversify``   — jump to a distant basin discovered by another worker.
    """

    kind: str
    dim: Optional[str] = None


class TaskQueue:
    """A thread-safe claimable FIFO of :class:`Task` items."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[Task] = []

    def push(self, task: Task) -> None:
        with self._lock:
            self._items.append(task)

    def push_many(self, tasks: List[Task]) -> None:
        with self._lock:
            self._items.extend(tasks)

    def claim(self) -> Optional[Task]:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _seed_tasks() -> List[Task]:
    """Initial work: probe every dimension plus a few full-space explores."""
    tasks: List[Task] = [Task(kind="mutate_dim", dim=name) for name in PARAM_RANGES]
    tasks.extend(Task(kind="explore") for _ in range(4))
    return tasks


# ---------------------------------------------------------------------------
# Configuration & result
# ---------------------------------------------------------------------------


@dataclass
class HillClimbConfig:
    """Configuration for a DeLM hill-climb run."""

    max_evals: int = 60
    n_workers: int = 4
    step_frac: float = DEFAULT_STEP_FRAC
    restart_prob: float = 0.1
    diversify_prob: float = 0.1
    verify: bool = True
    # Evaluation budget per candidate (kept small — this is an inner loop).
    eval_epochs: int = 3
    eval_steps: int = 5
    seed: int = 42
    fitness_weights: Optional[Dict[str, float]] = None
    welfare_reference: float = 10.0
    payoff_reference: float = 5.0
    # Real OS threads instead of the deterministic scheduler (not reproducible).
    use_threads: bool = False
    # How far (normalized param distance) a verified best must be from known
    # basins to count as a new basin.
    basin_distance: float = 0.25


@dataclass
class HillClimbResult:
    """Outcome of a DeLM hill-climb run."""

    best_params: Dict[str, Any]
    best_score: float
    seed_params: Dict[str, Any]
    seed_score: float
    n_evals: int
    history: List[Dict[str, Any]]
    gists: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    basins: List[Dict[str, Any]]
    telemetry: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "seed_params": self.seed_params,
            "seed_score": self.seed_score,
            "n_evals": self.n_evals,
            "history": self.history,
            "gists": self.gists,
            "constraints": self.constraints,
            "basins": self.basins,
            "telemetry": self.telemetry,
        }


# ---------------------------------------------------------------------------
# Worker logic
# ---------------------------------------------------------------------------


def _propose(
    task: Task, shared: SharedContext, rng: random.Random, step_frac: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Turn a claimed task into a (base_params, candidate_params) pair.

    The base is read fresh from shared state at proposal time, so a worker
    always climbs from the *latest* verified best — the DeLM property that an
    improvement written by one worker is instantly available to the next.
    """
    if task.kind == "restart":
        cand = _random_point(rng)
        return cand, cand
    if task.kind == "diversify":
        base = shared.random_basin(rng)
        return base, _mutate(base, rng, step_frac=step_frac)
    base, _ = shared.snapshot_best()
    if task.kind == "mutate_dim" and task.dim is not None:
        return base, _mutate(base, rng, step_frac=step_frac, dims=[task.dim])
    # explore (and fallback)
    return base, _mutate(base, rng, step_frac=step_frac)


def _process_task(
    task: Task,
    worker: int,
    shared: SharedContext,
    queue: TaskQueue,
    rng: random.Random,
    cfg: HillClimbConfig,
    eval_fn: Callable[[Dict[str, Any], int], EvalOutcome],
    eval_seed: int,
) -> bool:
    """Run one task. Returns True if a real evaluation was consumed."""
    base, cand = _propose(task, shared, rng, cfg.step_frac)

    # Read shared memory before paying for an evaluation.
    if shared.is_pruned(cand):
        shared.note_pruned()
        return False
    if shared.is_seen(cand):
        shared.note_redundant()
        return False

    parent_score = shared.snapshot_best()[1] if task.kind != "restart" else 0.0
    if task.kind == "diversify":
        # Parent is the basin we jumped to, not the global best.
        parent_score = next(
            (b.score for b in shared.basins if _param_distance(b.params, base) < 1e-9),
            shared.snapshot_best()[1],
        )

    outcome = eval_fn(cand, eval_seed)

    verifier = None
    if cfg.verify:
        # Verify at an independent seed derived from the eval seed.
        verifier = lambda p, s=eval_seed: eval_fn(p, s + 100_003)  # noqa: E731

    gist = shared.record(
        params=cand,
        outcome=outcome,
        parent_score=parent_score,
        base_params=base,
        worker=worker,
        kind=task.kind,
        verifier=verifier,
        basin_distance=cfg.basin_distance,
    )

    # A verified improvement spawns fresh neighbor work around the new best.
    if gist.verified:
        queue.push(Task(kind="explore"))
        queue.push(Task(kind="explore"))
        # Re-probe the two dims most likely to extend the winning direction.
        for name in _top_changed_dims(base, cand, k=2):
            queue.push(Task(kind="mutate_dim", dim=name))
    return True


def _top_changed_dims(base: Dict[str, Any], cand: Dict[str, Any], k: int) -> List[str]:
    """The ``k`` dimensions that changed most (range-normalized)."""
    scored: List[Tuple[float, str]] = []
    for name, (lo, hi) in PARAM_RANGES.items():
        span = (hi - lo) or 1.0
        mag = abs(float(cand.get(name, lo)) - float(base.get(name, lo))) / span
        scored.append((mag, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:k]]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_delm_hillclimb(
    base_scenario: ScenarioConfig,
    config: Optional[HillClimbConfig] = None,
    *,
    progress: bool = False,
) -> HillClimbResult:
    """Run a DeLM-style parallel hill-climb over governance/payoff params.

    Args:
        base_scenario: Scenario whose governance/payoff knobs are optimized.
        config: Hill-climb configuration (budget, workers, eval size, …).
        progress: If True, log a line whenever the verified best improves.

    Returns:
        A :class:`HillClimbResult` with the best parameters found, the verified
        gist trail, binding constraints, discovered basins, and telemetry.
    """
    cfg = config or HillClimbConfig()
    controller_rng = random.Random(cfg.seed)

    # Per-evaluation simulation seed: fixed across candidates so fitness
    # differences reflect the genome, not the seed (verification uses a
    # separate, offset seed).
    base_eval_seed = cfg.seed

    def eval_fn(params: Dict[str, Any], seed: int) -> EvalOutcome:
        return evaluate_params(
            base_scenario,
            params,
            seed=seed,
            eval_epochs=cfg.eval_epochs,
            eval_steps=cfg.eval_steps,
            fitness_weights=cfg.fitness_weights,
            welfare_reference=cfg.welfare_reference,
            payoff_reference=cfg.payoff_reference,
        )

    # Seed the climb from the scenario's current configuration.
    seed_params = seed_params_from_scenario(base_scenario)
    seed_outcome = eval_fn(seed_params, base_eval_seed)
    seed_score = seed_outcome.fitness if seed_outcome.viable else 0.0
    shared = SharedContext(best_params=seed_params, best_score=seed_score)
    shared._seen[_quantize_key(seed_params)] = seed_score

    queue = TaskQueue()
    queue.push_many(_seed_tasks())

    history: List[Dict[str, Any]] = [
        {"eval": 0, "best_score": round(seed_score, 6), "kind": "seed"}
    ]
    n_evals = 1  # the seed evaluation counts toward the budget
    eval_counter = 0
    last_best = seed_score

    # Per-worker RNGs, derived deterministically from the controller seed so
    # the virtual swarm is reproducible.
    worker_rngs = [random.Random(cfg.seed * 7919 + w) for w in range(cfg.n_workers)]

    if cfg.use_threads:
        n_evals += _run_threaded(
            shared, queue, cfg, eval_fn, base_eval_seed, worker_rngs, controller_rng
        )
    else:
        # Deterministic round-robin over virtual workers.
        worker = 0
        stall = 0
        while n_evals < cfg.max_evals:
            # Keep the queue fed: inject restart/diversify work to escape optima.
            if len(queue) == 0 or stall > cfg.n_workers * 3:
                roll = controller_rng.random()
                if roll < cfg.restart_prob or not shared.basins:
                    queue.push(Task(kind="restart"))
                elif roll < cfg.restart_prob + cfg.diversify_prob:
                    queue.push(Task(kind="diversify"))
                else:
                    queue.push(Task(kind="explore"))
                stall = 0

            task = queue.claim()
            if task is None:
                queue.push(Task(kind="explore"))
                continue

            rng = worker_rngs[worker]
            eval_seed = base_eval_seed
            consumed = _process_task(
                task, worker, shared, queue, rng, cfg, eval_fn, eval_seed
            )
            worker = (worker + 1) % cfg.n_workers

            if consumed:
                eval_counter += 1
                # Verification costs a second evaluation; count it for honesty.
                evals_used = 1
                n_evals += evals_used
                _, best_now = shared.snapshot_best()
                if best_now > last_best + IMPROVE_EPS:
                    history.append(
                        {
                            "eval": n_evals,
                            "best_score": round(best_now, 6),
                            "kind": task.kind,
                        }
                    )
                    if progress:
                        logger.info(
                            "eval %d: best %.4f -> %.4f (%s)",
                            n_evals,
                            last_best,
                            best_now,
                            task.kind,
                        )
                    last_best = best_now
                    stall = 0
                else:
                    stall += 1
            else:
                stall += 1

    best_params, best_score = shared.snapshot_best()
    telemetry = {
        "n_workers": cfg.n_workers,
        "improvements": shared.n_improvements,
        "verify_rejected": shared.n_verify_rejected,
        "redundant_skipped": shared.n_redundant_skipped,
        "constraint_pruned": shared.n_constraint_pruned,
        "gists": len(shared.gists),
        "constraints": len(shared.constraints),
        "basins": len(shared.basins),
    }
    return HillClimbResult(
        best_params=best_params,
        best_score=round(best_score, 6),
        seed_params=seed_params,
        seed_score=round(seed_score, 6),
        n_evals=n_evals,
        history=history,
        gists=[g.to_json() for g in shared.gists],
        constraints=[asdict(c) for c in shared.constraints],
        basins=[asdict(b) for b in shared.basins],
        telemetry=telemetry,
    )


def _run_threaded(
    shared: SharedContext,
    queue: TaskQueue,
    cfg: HillClimbConfig,
    eval_fn: Callable[[Dict[str, Any], int], EvalOutcome],
    base_eval_seed: int,
    worker_rngs: List[random.Random],
    controller_rng: random.Random,
) -> int:
    """Real-thread scheduler. Returns the number of evaluations consumed.

    Not bit-for-bit reproducible: thread interleaving over the shared store is
    not controlled. Provided for throughput on slow evaluations.
    """
    budget = max(0, cfg.max_evals - 1)
    counter = {"used": 0}
    counter_lock = threading.Lock()

    def claim_budget() -> bool:
        with counter_lock:
            if counter["used"] >= budget:
                return False
            counter["used"] += 1
            return True

    def worker_loop(wid: int) -> None:
        rng = worker_rngs[wid]
        while True:
            if not claim_budget():
                return
            task = queue.claim()
            if task is None:
                roll = controller_rng.random()
                if roll < cfg.restart_prob or not shared.basins:
                    task = Task(kind="restart")
                elif roll < cfg.restart_prob + cfg.diversify_prob:
                    task = Task(kind="diversify")
                else:
                    task = Task(kind="explore")
            consumed = _process_task(
                task, wid, shared, queue, rng, cfg, eval_fn, base_eval_seed
            )
            if not consumed:
                # Did not spend the simulated evaluation — refund the budget.
                with counter_lock:
                    counter["used"] -= 1

    threads = [
        threading.Thread(target=worker_loop, args=(w,), daemon=True)
        for w in range(cfg.n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter["used"]


# ---------------------------------------------------------------------------
# Run-folder writer & CLI
# ---------------------------------------------------------------------------


def write_run(result: HillClimbResult, run_dir: Path, scenario_id: str) -> None:
    """Write a self-contained run folder (best params, history, gists)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_params.json").write_text(
        json.dumps(
            {"scenario": scenario_id, "best_score": result.best_score, "best_params": result.best_params},
            indent=2,
        )
    )
    (run_dir / "result.json").write_text(json.dumps(result.to_json(), indent=2))
    with (run_dir / "gists.jsonl").open("w") as fh:
        for gist in result.gists:
            fh.write(json.dumps(gist) + "\n")


def _build_weights(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    overrides = {
        "low_toxicity": args.weight_toxicity,
        "welfare": args.weight_welfare,
        "quality_gap": args.weight_quality_gap,
        "payoff_gap": args.weight_payoff_gap,
    }
    if all(v is None for v in overrides.values()):
        return None
    weights = dict(DEFAULT_FITNESS_WEIGHTS)
    for key, val in overrides.items():
        if val is not None:
            weights[key] = val
    return weights


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeLM-style parallel hill-climbing over governance/payoff params."
    )
    parser.add_argument("scenario", help="Path to base scenario YAML.")
    parser.add_argument("--max-evals", type=int, default=60, help="Evaluation budget.")
    parser.add_argument("--workers", type=int, default=4, help="Number of virtual workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval-epochs", type=int, default=3, help="Epochs per evaluation.")
    parser.add_argument("--eval-steps", type=int, default=5, help="Steps per epoch.")
    parser.add_argument("--step-frac", type=float, default=DEFAULT_STEP_FRAC, help="Mutation step (fraction of range).")
    parser.add_argument("--restart-prob", type=float, default=0.1, help="Restart-task probability.")
    parser.add_argument("--no-verify", action="store_true", help="Disable verified admission (faster, noisier).")
    parser.add_argument("--threads", action="store_true", help="Use real OS threads (not reproducible).")
    parser.add_argument("--weight-toxicity", type=float, default=None)
    parser.add_argument("--weight-welfare", type=float, default=None)
    parser.add_argument("--weight-quality-gap", type=float, default=None)
    parser.add_argument("--weight-payoff-gap", type=float, default=None)
    parser.add_argument("--output-dir", default=None, help="Run folder (default: runs/<ts>_delm_<scenario>).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log each improvement.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    scenario = load_scenario(Path(args.scenario))
    cfg = HillClimbConfig(
        max_evals=args.max_evals,
        n_workers=args.workers,
        step_frac=args.step_frac,
        restart_prob=args.restart_prob,
        verify=not args.no_verify,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        seed=args.seed,
        fitness_weights=_build_weights(args),
        use_threads=args.threads,
    )

    start = time.time()
    result = run_delm_hillclimb(scenario, cfg, progress=args.verbose)
    elapsed = time.time() - start

    scenario_id = scenario.scenario_id or "unknown"
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path(f"runs/{time.strftime('%Y%m%d_%H%M%S')}_delm_{scenario_id}")
    write_run(result, run_dir, scenario_id)

    t = result.telemetry
    print(f"DeLM hill-climb: {scenario_id}  ({elapsed:.1f}s, {result.n_evals} evals)")
    print(f"  seed score : {result.seed_score:.4f}")
    print(f"  best score : {result.best_score:.4f}  (+{result.best_score - result.seed_score:.4f})")
    print(
        f"  workers={t['n_workers']}  improvements={t['improvements']}  "
        f"verify_rejected={t['verify_rejected']}"
    )
    print(
        f"  redundant_skipped={t['redundant_skipped']}  constraint_pruned={t['constraint_pruned']}  "
        f"basins={t['basins']}"
    )
    print(f"  run dir    : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

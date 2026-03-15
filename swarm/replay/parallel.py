# Async parallel scenario execution.
# Runs multiple scenario variants concurrently via asyncio.
# Borrowed pattern: MiroFish-style async parallel environment execution.

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from swarm.replay.runner import ReplayRunResult, _set_nested_attr
from swarm.scenarios.loader import build_orchestrator


@dataclass
class ParallelRunSpec:
    scenario: Any
    seed: int
    index: int
    eval_epochs: Optional[int] = None
    eval_steps: Optional[int] = None
    parameter_overrides: Optional[Dict[str, Any]] = None


def _run_single(spec: ParallelRunSpec) -> ReplayRunResult:
    scenario = copy.deepcopy(spec.scenario)
    scenario.orchestrator_config.seed = spec.seed
    scenario.orchestrator_config.log_path = None
    scenario.orchestrator_config.log_events = False
    if spec.eval_epochs is not None:
        scenario.orchestrator_config.n_epochs = spec.eval_epochs
    if spec.eval_steps is not None:
        scenario.orchestrator_config.steps_per_epoch = spec.eval_steps
    if spec.parameter_overrides:
        for path, value in spec.parameter_overrides.items():
            if path.startswith("simulation."):
                attr = path[len("simulation."):]
                setattr(scenario.orchestrator_config, attr, value)
            elif path.startswith("governance."):
                attr = path[len("governance."):]
                setattr(scenario.orchestrator_config.governance_config, attr, value)
            else:
                _set_nested_attr(scenario.orchestrator_config, path, value)
    orchestrator = build_orchestrator(scenario)
    history = orchestrator.run()
    if not history:
        return ReplayRunResult(
            replay_index=spec.index, seed=spec.seed,
            total_interactions=0, accepted_interactions=0,
            avg_toxicity=0.0, avg_quality_gap=0.0, total_welfare=0.0)
    n_epochs = len(history)
    return ReplayRunResult(
        replay_index=spec.index, seed=spec.seed,
        total_interactions=sum(m.total_interactions for m in history),
        accepted_interactions=sum(m.accepted_interactions for m in history),
        avg_toxicity=sum(m.toxicity_rate for m in history) / n_epochs,
        avg_quality_gap=sum(m.quality_gap for m in history) / n_epochs,
        total_welfare=sum(m.total_welfare for m in history))


async def run_parallel(
    specs: List[ParallelRunSpec],
    max_concurrency: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ReplayRunResult]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[Optional[ReplayRunResult]] = [None] * len(specs)
    completed = 0
    async def _run_with_limit(idx: int, spec: ParallelRunSpec) -> None:
        nonlocal completed
        async with semaphore:
            result = await asyncio.to_thread(_run_single, spec)
            results[idx] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(specs))
    tasks = [_run_with_limit(i, spec) for i, spec in enumerate(specs)]
    await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def evaluate_candidate_parallel(
    scenario: Any, seeds: List[int], eval_epochs: int, eval_steps: int,
    max_concurrency: int = 4,
    parameter_overrides: Optional[Dict[str, Any]] = None,
) -> List[ReplayRunResult]:
    specs = [
        ParallelRunSpec(scenario=scenario, seed=seed, index=i,
                        eval_epochs=eval_epochs, eval_steps=eval_steps,
                        parameter_overrides=parameter_overrides)
        for i, seed in enumerate(seeds)]
    return await run_parallel(specs, max_concurrency=max_concurrency)


def run_parallel_sync(
    specs: List[ParallelRunSpec],
    max_concurrency: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ReplayRunResult]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run,
                run_parallel(specs, max_concurrency, progress_callback))
            return future.result()
    else:
        return asyncio.run(
            run_parallel(specs, max_concurrency, progress_callback))

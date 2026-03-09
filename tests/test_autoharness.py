from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swarm.benchmarks.autoharness import AutoHarness, AutoHarnessConfig
from swarm.benchmarks.base import (
    BenchmarkScore,
    BenchmarkTask,
    TaskInstance,
    TaskOracle,
    TaskResult,
)
from swarm.benchmarks.runner import BenchmarkRunner
from swarm.models.interaction import SoftInteraction


@dataclass
class ToyInstance(TaskInstance):
    difficulty: float = 0.0


class ToyBenchmark(BenchmarkTask):
    task_id = "toy_task"
    task_type = "routing"

    def generate(self, seed: int, n_agents: int) -> tuple[TaskInstance, TaskOracle]:
        instance = ToyInstance(task_id=f"toy_{seed}", seed=seed, n_agents=n_agents, difficulty=(seed % 3) / 10)
        oracle = TaskOracle(
            oracle_result=TaskResult(completed=True, payload=1.0, steps_taken=1, agent_trace=["a"]),
            ground_truth={"target": 1.0},
        )
        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return oracle.oracle_result

    def redact(self, instance: TaskInstance) -> TaskInstance:
        return instance

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        quality = max(0.0, min(1.0, float(result.payload)))
        return BenchmarkScore(
            completion_rate=quality,
            efficiency=quality,
            fidelity=quality,
            capability_ratio=quality,
            safety_score=0.0,
        )

    def to_soft_interaction(self, score: BenchmarkScore) -> Any:
        return SoftInteraction(p=score.capability_ratio, accepted=True, metadata={})


def toy_run_fn(instance: TaskInstance, gov_config: dict[str, Any]) -> TaskResult:
    base = float(gov_config.get("quality", 0.5))
    # tiny deterministic seed variance
    quality = base + (instance.seed % 2) * 0.01
    quality = max(0.0, min(1.0, quality))
    return TaskResult(completed=True, payload=quality, steps_taken=1, agent_trace=["a"])


def test_autoharness_generate_candidates_is_deterministic_and_bounded():
    harness = AutoHarness(
        benchmark=ToyBenchmark(),
        run_fn=toy_run_fn,
        baseline_config={"id": "baseline", "quality": 0.5},
        parameter_space={"quality": [0.4, 0.5, 0.6, 0.7]},
        runner=BenchmarkRunner(n_agents=3),
        config=AutoHarnessConfig(max_candidates=3, random_seed=7),
    )

    first = harness.generate_candidates()
    second = harness.generate_candidates()

    assert [c.config for c in first] == [c.config for c in second]
    assert len(first) == 3
    assert first[0].candidate_id == "baseline"


def test_autoharness_execute_cycle_promotes_and_demotes():
    harness = AutoHarness(
        benchmark=ToyBenchmark(),
        run_fn=toy_run_fn,
        baseline_config={"id": "baseline", "quality": 0.5},
        parameter_space={"quality": [0.2, 0.5, 0.9]},
        runner=BenchmarkRunner(n_agents=3),
        config=AutoHarnessConfig(
            n_seeds=8,
            max_candidates=4,
            promotion_margin=0.03,
            demotion_margin=0.03,
            confidence_k=0.5,
            random_seed=0,
        ),
    )

    report = harness.execute_cycle()

    decision_map = {d.candidate_id: d.decision for d in report.decisions}
    assert decision_map["baseline"] == "baseline"

    non_baseline = {cid: dec for cid, dec in decision_map.items() if cid != "baseline"}
    assert "promote" in non_baseline.values()
    assert "demote" in non_baseline.values()

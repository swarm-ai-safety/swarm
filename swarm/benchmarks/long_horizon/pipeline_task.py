"""Pipeline task benchmark.

Multi-step tasks where early decisions constrain later ones. A pipeline
has N sequential stages; each stage transforms the payload. Governance
gates (confirmation, audit) at any stage delay everything downstream,
making constraint costs compound nonlinearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskInstance, TaskResult
from swarm.models.interaction import SoftInteraction


@dataclass
class PipelineStage:
    """One stage in a multi-step pipeline."""

    stage_id: int
    agent_id: str
    transform_key: int  # additive transform applied to payload


@dataclass
class PipelineInstance(TaskInstance):
    """A pipeline task with sequential stages."""

    initial_payload: int = 0
    stages: list[PipelineStage] = field(default_factory=list)
    expected_output: int = 0


class PipelineTaskBenchmark(BenchmarkTask):
    """Multi-step pipeline where each agent transforms a payload.

    Each stage adds its transform_key to the running payload.
    The oracle runs the full pipeline with no governance friction.
    Governance gates between stages add steps (latency).
    """

    task_id = "pipeline_task_v1"
    task_type = "long_horizon"

    def __init__(self, n_stages: int = 5):
        self.n_stages = n_stages

    def generate(self, seed: int, n_agents: int) -> PipelineInstance:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        initial_payload = int(rng.integers(0, 100))

        # Assign agents to stages (with wrap-around if n_stages > n_agents)
        stages = []
        running = initial_payload
        for i in range(self.n_stages):
            agent = agent_ids[i % n_agents]
            key = int(rng.integers(1, 50))
            stages.append(PipelineStage(stage_id=i, agent_id=agent, transform_key=key))
            running += key

        return PipelineInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            initial_payload=initial_payload,
            stages=stages,
            expected_output=running,
        )

    def oracle_run(self, instance: TaskInstance) -> TaskResult:
        inst = instance
        assert isinstance(inst, PipelineInstance)
        return TaskResult(
            completed=True,
            payload=inst.expected_output,
            steps_taken=len(inst.stages),
            agent_trace=[s.agent_id for s in inst.stages],
        )

    def score(self, result: TaskResult, oracle: TaskResult) -> BenchmarkScore:
        if not result.completed:
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0)

        fidelity = 1.0 if result.payload == oracle.payload else 0.0
        efficiency = min(oracle.steps_taken / max(result.steps_taken, 1), 1.0)
        completion_rate = float(result.completed) * fidelity
        capability_ratio = completion_rate

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = score.completion_rate * 0.5 + score.fidelity * 0.3 + score.efficiency * 0.2
        p = max(0.0, min(1.0, p))
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0,
            metadata={"benchmark": self.task_id},
        )

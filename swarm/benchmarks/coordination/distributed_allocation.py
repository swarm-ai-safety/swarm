"""Distributed allocation benchmark.

Agents must coordinate to solve a joint task that no single agent can
complete alone. Each agent holds a piece of a total requirement, and they
must collectively allocate their contributions to meet a target.

Governance constraints (friction, staking) directly impede coordination
speed, making the capability/constraint tradeoff visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskInstance, TaskResult
from swarm.models.interaction import SoftInteraction


@dataclass
class AllocationInstance(TaskInstance):
    """A distributed allocation task."""

    target_total: float = 0.0
    agent_capacities: dict[str, float] = field(default_factory=dict)
    optimal_allocation: dict[str, float] = field(default_factory=dict)


class DistributedAllocationBenchmark(BenchmarkTask):
    """Agents coordinate to collectively reach a target allocation.

    Each agent has a capacity (max contribution). The oracle allocation
    distributes proportionally. Scoring measures how close the actual
    total is to the target, and how efficiently agents coordinated.
    """

    task_id = "distributed_allocation_v1"
    task_type = "coordination"

    def __init__(self, target_ratio: float = 0.7):
        """target_ratio: fraction of total capacity that must be allocated."""
        self.target_ratio = target_ratio

    def generate(self, seed: int, n_agents: int) -> AllocationInstance:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        capacities = {aid: float(rng.uniform(1.0, 10.0)) for aid in agent_ids}
        total_capacity = sum(capacities.values())
        target = total_capacity * self.target_ratio

        # Oracle: proportional allocation
        optimal = {
            aid: cap * self.target_ratio for aid, cap in capacities.items()
        }

        return AllocationInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            target_total=target,
            agent_capacities=capacities,
            optimal_allocation=optimal,
        )

    def oracle_run(self, instance: TaskInstance) -> TaskResult:
        inst = instance
        assert isinstance(inst, AllocationInstance)
        return TaskResult(
            completed=True,
            payload=inst.optimal_allocation,
            steps_taken=1,  # perfect coordination in one round
            agent_trace=sorted(inst.agent_capacities.keys()),
        )

    def score(self, result: TaskResult, oracle: TaskResult) -> BenchmarkScore:
        if not result.completed or not isinstance(result.payload, dict):
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0)

        oracle_alloc = oracle.payload
        actual_alloc = result.payload

        # Fidelity: how close is actual total to oracle total
        oracle_total = sum(oracle_alloc.values())
        actual_total = sum(actual_alloc.values())
        if oracle_total > 0:
            fidelity = max(0.0, 1.0 - abs(actual_total - oracle_total) / oracle_total)
        else:
            fidelity = 1.0

        # Efficiency: fewer rounds is better
        efficiency = min(oracle.steps_taken / max(result.steps_taken, 1), 1.0)

        completion_rate = fidelity  # partial credit for close-enough allocation
        capability_ratio = completion_rate  # oracle always 1.0

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = score.completion_rate * 0.5 + score.fidelity * 0.3 + score.efficiency * 0.2
        p = max(0.0, min(1.0, p))
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0.5,
            metadata={"benchmark": self.task_id},
        )

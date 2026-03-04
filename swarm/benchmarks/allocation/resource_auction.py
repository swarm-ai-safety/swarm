"""Resource auction benchmark.

Agents bid for shared resources. The outcome metric is allocative
efficiency: did the resource end up with the agent that could use it
best? Ground truth comes from known agent valuations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskInstance, TaskResult
from swarm.models.interaction import SoftInteraction


@dataclass
class AuctionInstance(TaskInstance):
    """A resource auction task."""

    n_resources: int = 0
    agent_valuations: dict[str, list[float]] = field(default_factory=dict)
    optimal_assignment: dict[int, str] = field(default_factory=dict)
    optimal_welfare: float = 0.0


class ResourceAuctionBenchmark(BenchmarkTask):
    """Agents bid for shared resources; scoring measures allocative efficiency.

    Each resource has a known valuation per agent. The oracle assigns each
    resource to the agent with the highest valuation. The capability metric
    is total realized welfare / oracle welfare.
    """

    task_id = "resource_auction_v1"
    task_type = "allocation"

    def __init__(self, n_resources: int = 5):
        self.n_resources = n_resources

    def generate(self, seed: int, n_agents: int) -> AuctionInstance:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        # Each agent has a valuation for each resource
        valuations = {
            aid: [float(rng.uniform(0.0, 10.0)) for _ in range(self.n_resources)]
            for aid in agent_ids
        }

        # Oracle: greedy assignment (each resource to highest-valuing agent)
        optimal: dict[int, str] = {}
        optimal_welfare = 0.0
        for r in range(self.n_resources):
            best_agent = max(agent_ids, key=lambda a: valuations[a][r])
            optimal[r] = best_agent
            optimal_welfare += valuations[best_agent][r]

        return AuctionInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            n_resources=self.n_resources,
            agent_valuations=valuations,
            optimal_assignment=optimal,
            optimal_welfare=optimal_welfare,
        )

    def oracle_run(self, instance: TaskInstance) -> TaskResult:
        inst = instance
        assert isinstance(inst, AuctionInstance)
        return TaskResult(
            completed=True,
            payload=inst.optimal_assignment,
            steps_taken=1,  # single-round optimal auction
            agent_trace=sorted(set(inst.optimal_assignment.values())),
        )

    def score(self, result: TaskResult, oracle: TaskResult) -> BenchmarkScore:
        if not result.completed or not isinstance(result.payload, dict):
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0)

        oracle_assign = oracle.payload
        actual_assign = result.payload

        # Fidelity: fraction of resources assigned to the same agent as oracle
        matches = sum(
            1
            for r in oracle_assign
            if r in actual_assign and actual_assign[r] == oracle_assign[r]
        )
        n = len(oracle_assign)
        fidelity = matches / n if n > 0 else 1.0

        efficiency = min(oracle.steps_taken / max(result.steps_taken, 1), 1.0)
        completion_rate = float(result.completed)
        capability_ratio = fidelity  # welfare ratio is the real metric

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = score.fidelity * 0.6 + score.completion_rate * 0.3 + score.efficiency * 0.1
        p = max(0.0, min(1.0, p))
        return SoftInteraction(
            p=p,
            accepted=score.fidelity > 0.3,
            metadata={"benchmark": self.task_id},
        )

"""Message routing benchmark.

Each task: agent A must get a payload to agent Z through a network of
intermediaries. Success is binary — did the payload arrive intact and
unmodified? Governance levers (audits, reputation, staking) create
friction on the routing path.

Verifiable, hard to game, maps directly onto provenance tracking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskInstance, TaskResult
from swarm.env.network import AgentNetwork, NetworkConfig, NetworkTopology
from swarm.models.interaction import SoftInteraction


@dataclass
class RoutingInstance(TaskInstance):
    """A concrete message routing task."""

    payload: Any = None
    source_agent: str = ""
    target_agent: str = ""
    network: AgentNetwork | None = None
    expected_path: list[str] = field(default_factory=list)


def _bfs_path(network: AgentNetwork, source: str, target: str) -> list[str]:
    """Return shortest path as list of node IDs, or [] if unreachable."""
    if source == target:
        return [source]
    visited = {source}
    queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
    while queue:
        current, path = queue.popleft()
        for neighbor in network.neighbors(current):
            if neighbor == target:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


class MessageRoutingBenchmark(BenchmarkTask):
    """Route a payload from source to target through an agent network.

    Scoring:
    - completion: did the payload arrive at the target?
    - fidelity: was the payload unmodified?
    - efficiency: oracle_steps / actual_steps
    """

    task_id = "message_routing_v1"
    task_type = "routing"

    def __init__(
        self,
        topology: NetworkTopology = NetworkTopology.RANDOM_ERDOS_RENYI,
        edge_probability: float = 0.4,
    ):
        self.topology = topology
        self.edge_probability = edge_probability

    def generate(self, seed: int, n_agents: int) -> RoutingInstance:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        config = NetworkConfig(
            topology=self.topology,
            edge_probability=self.edge_probability,
        )
        network = AgentNetwork(config=config, seed=seed)
        network.initialize(agent_ids)

        # Pick source/target that have a path between them
        indices = rng.permutation(n_agents)
        source, target = agent_ids[indices[0]], agent_ids[indices[1]]
        path = _bfs_path(network, source, target)

        # If no path, add edges to create one
        if not path:
            network.add_edge(source, target)
            path = [source, target]

        payload = int(rng.integers(0, 10_000))

        return RoutingInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            payload=payload,
            source_agent=source,
            target_agent=target,
            network=network,
            expected_path=path,
        )

    def oracle_run(self, instance: TaskInstance) -> TaskResult:
        inst = instance
        assert isinstance(inst, RoutingInstance)
        return TaskResult(
            completed=True,
            payload=inst.payload,
            steps_taken=len(inst.expected_path) - 1,
            agent_trace=list(inst.expected_path),
        )

    def score(self, result: TaskResult, oracle: TaskResult) -> BenchmarkScore:
        fidelity = 1.0 if result.payload == oracle.payload else 0.0
        efficiency = (
            oracle.steps_taken / max(result.steps_taken, 1)
            if oracle.steps_taken > 0
            else 1.0
        )
        efficiency = min(efficiency, 1.0)
        completion_rate = float(result.completed) * fidelity
        # Oracle always completes, so capability_ratio == completion_rate
        capability_ratio = completion_rate
        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = score.completion_rate * 0.6 + score.fidelity * 0.3 + score.efficiency * 0.1
        p = max(0.0, min(1.0, p))
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0,
            metadata={"benchmark": self.task_id},
        )

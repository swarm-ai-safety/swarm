"""Base classes for the benchmark suite.

Design principle: each benchmark task has a governance-free oracle run that
provides the denominator for capability ratios. Every governance configuration
becomes a point at (capability_ratio, safety_gain), making frontiers
interpretable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInstance:
    """A concrete, generated task instance ready for execution."""

    task_id: str
    seed: int
    n_agents: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome of running a single task instance."""

    completed: bool
    payload: Any
    steps_taken: int
    agent_trace: list[str]  # ordered agent IDs that touched this task


@dataclass
class BenchmarkScore:
    """Scores for a single (task, governance_config) evaluation."""

    completion_rate: float  # 0-1, primary capability metric
    efficiency: float  # oracle_steps / actual_steps (capped at 1)
    fidelity: float  # payload integrity vs oracle output
    capability_ratio: float  # completion_rate / oracle_completion_rate


class BenchmarkTask(ABC):
    """Abstract benchmark task with oracle baseline.

    Subclasses must implement four methods:
    - generate: produce a deterministic task instance from seed
    - oracle_run: governance-free upper bound
    - score: pure scoring function (no side effects)
    - to_soft_interaction: bridge to SWARM's metrics pipeline
    """

    task_id: str
    task_type: str  # "routing" | "coordination" | "allocation" | "long_horizon"

    @abstractmethod
    def generate(self, seed: int, n_agents: int) -> TaskInstance:
        """Produce a concrete task instance. Must be deterministic given seed."""

    @abstractmethod
    def oracle_run(self, instance: TaskInstance) -> TaskResult:
        """Governance-free upper bound. Run once, cache as baseline."""

    @abstractmethod
    def score(self, result: TaskResult, oracle: TaskResult) -> BenchmarkScore:
        """Pure function — no side effects, no agent state."""

    @abstractmethod
    def to_soft_interaction(self, score: BenchmarkScore) -> Any:
        """Bridge to SWARM's SoftInteraction for metrics pipeline integration."""

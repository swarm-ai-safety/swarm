"""Benchmark suite for measuring capability under governance constraints.

Each benchmark defines a task with a governance-free oracle baseline,
enabling separation of "governance reduced capability" from "adversarial
agents reduced capability."
"""

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskInstance, TaskResult

__all__ = [
    "BenchmarkTask",
    "TaskInstance",
    "TaskResult",
    "BenchmarkScore",
]

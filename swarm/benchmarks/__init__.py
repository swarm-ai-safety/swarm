"""Benchmark suite for measuring capability under governance constraints.

Each benchmark defines a task with a governance-free oracle baseline,
enabling separation of "governance reduced capability" from "adversarial
agents reduced capability."

Security invariants:
- TaskInstance (visible to agents) never contains ground truth.
- TaskOracle (scorer-only) holds oracle results and ground truth.
- Instances are deep-copied before exposure to run_fn.
- run_fn return values are validated before scoring.
"""

from swarm.benchmarks.autoharness import (
    AutoHarness,
    AutoHarnessConfig,
    AutoHarnessReport,
    HarnessCandidate,
    HarnessDecision,
)
from swarm.benchmarks.base import (
    BenchmarkScore,
    BenchmarkTask,
    ScoringWeights,
    TaskInstance,
    TaskOracle,
    TaskResult,
)

__all__ = [
    "BenchmarkTask",
    "TaskInstance",
    "TaskOracle",
    "TaskResult",
    "BenchmarkScore",
    "ScoringWeights",
    "AutoHarness",
    "AutoHarnessConfig",
    "AutoHarnessReport",
    "HarnessCandidate",
    "HarnessDecision",
]

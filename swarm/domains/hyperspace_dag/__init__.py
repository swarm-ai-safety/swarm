"""Hyperspace DAG Planner domain for distributional safety research.

Evaluates task decomposition DAGs from Hyperspace Architect v1 through
SWARM's soft-label pipeline. DAG structural features (edge density,
critical path depth, parallelism) feed into the proxy as engagement
signals, while execution outcomes feed the standard channels.

Key research question: does Architect's self-reported confidence
correlate with actual p? The adapter measures this via Pearson
correlation in the report.
"""

from swarm.domains.hyperspace_dag.adapter import DagAdapter, DagAdapterReport
from swarm.domains.hyperspace_dag.config import DagConfig, DagProxyConfig
from swarm.domains.hyperspace_dag.entities import (
    AgentRole,
    DagEvent,
    DagOutcome,
    DagSubtask,
    PlanDag,
    SubtaskStatus,
)

__all__ = [
    "AgentRole",
    "DagAdapter",
    "DagAdapterReport",
    "DagConfig",
    "DagEvent",
    "DagOutcome",
    "DagProxyConfig",
    "DagSubtask",
    "PlanDag",
    "SubtaskStatus",
]

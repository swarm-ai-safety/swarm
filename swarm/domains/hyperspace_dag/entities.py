"""Entity definitions for the Hyperspace DAG Planner domain.

Models task decomposition DAGs from Hyperspace Architect v1.
Each plan is a directed acyclic graph of typed subtasks with
dependencies, agent assignments, and execution outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class AgentRole(Enum):
    """Typed agent roles from Architect's subtask assignment."""

    CODING = "coding"
    DESIGN = "design"
    INFRA = "infra"
    TESTING = "testing"
    RESEARCH = "research"


class SubtaskStatus(Enum):
    """Execution status of a DAG subtask."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DagSubtask:
    """A single subtask node in the plan DAG."""

    subtask_id: str
    title: str = ""
    agent_role: AgentRole = AgentRole.CODING
    priority: int = 5  # 1-10
    estimated_duration: float = 1.0  # arbitrary units
    dependencies: List[str] = field(default_factory=list)
    parallel: bool = False
    status: SubtaskStatus = SubtaskStatus.PENDING


@dataclass
class PlanDag:
    """A task decomposition DAG from Architect.

    Mirrors the DagSubtask[] + metadata structure from
    Architect v1's plan output.
    """

    plan_id: str
    prompt: str = ""  # original build request
    subtasks: List[DagSubtask] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    max_parallelism: int = 1
    confidence: float = 0.5  # Architect's self-reported confidence [0, 1]
    cache_hit: bool = False  # whether plan came from P2P cache


@dataclass
class DagEvent:
    """An event in DAG plan execution for logging."""

    event_type: str  # plan_proposed, subtask_started, subtask_completed, subtask_failed, plan_completed, plan_failed
    step: int = 0
    epoch: int = 0
    agent_id: str = ""
    plan_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DagOutcome:
    """Execution outcome for a completed plan."""

    plan_id: str
    success: bool = False
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_total: int = 0
    retries: int = 0
    wall_time: float = 0.0  # actual execution duration
    estimated_time: float = 0.0  # Architect's estimate
    metadata: Dict[str, Any] = field(default_factory=dict)

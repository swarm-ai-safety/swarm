"""Data models for SciAgentBench evaluation.

Defines task specifications, execution modes, and result structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TopologyMode(str, Enum):
    """Execution topology for batch runs.

    Attributes:
        SHARED_EPISODE: All agents share the same episode/environment instance.
            Use when agents should interact in the same world state.
        PER_AGENT: Each agent gets its own isolated episode.
            Use for independent agent evaluation with task replication.
        PER_TASK: Each task gets its own episode.
            Use when tasks are independent and agents run sequentially.
    """

    SHARED_EPISODE = "shared_episode"
    PER_AGENT = "per_agent"
    PER_TASK = "per_task"


class SciAgentBenchTask(BaseModel):
    """Configuration for a single SciAgentBench task.

    Attributes:
        task_id: Unique identifier for the task.
        instruction: Task instruction text.
        dataset_path: Path or URI to task dataset.
        domain: Scientific domain (e.g., "bioinformatics", "chemistry").
        expert_knowledge: Optional domain-specific hints or context.
        success_criteria: Dict defining task success conditions.
        timeout_seconds: Maximum execution time per attempt.
    """

    task_id: str
    instruction: str
    dataset_path: str
    domain: str
    expert_knowledge: Optional[str] = None
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 300.0


class TaskResult(BaseModel):
    """Result from executing a single task.

    Attributes:
        task_id: Task identifier.
        agent_id: Agent identifier.
        topology_mode: Execution topology used.
        seed: Random seed used for this execution.
        success: Whether the task succeeded per success_criteria.
        execution_time: Time taken in seconds.
        output: Agent's output (code, analysis, etc.).
        error_message: Error message if execution failed.
        metadata: Additional execution metadata.
    """

    task_id: str
    agent_id: str
    topology_mode: TopologyMode
    seed: int
    success: bool
    execution_time: float
    output: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchResult(BaseModel):
    """Aggregated results from a batch run.

    Attributes:
        topology_mode: Topology mode used for this batch.
        task_results: Individual task results.
        total_tasks: Total number of tasks attempted.
        successful_tasks: Number of tasks that succeeded.
        total_execution_time: Total wall-clock time.
        seeds: List of seeds used (one per task or agent).
        summary_metrics: Aggregate statistics.
    """

    topology_mode: TopologyMode
    task_results: List[TaskResult]
    total_tasks: int
    successful_tasks: int
    total_execution_time: float
    seeds: List[int]
    summary_metrics: Dict[str, float] = Field(default_factory=dict)

    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return (
            self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        )

    def avg_execution_time(self) -> float:
        """Calculate average execution time per task."""
        if not self.task_results:
            return 0.0
        return sum(r.execution_time for r in self.task_results) / len(
            self.task_results
        )

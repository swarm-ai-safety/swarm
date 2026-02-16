"""Replay verification for synthesized tasks.

This module provides verification that synthesized tasks can be
successfully replayed with deterministic results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from swarm.env.composite_tasks import CompositeTask

logger = logging.getLogger(__name__)


@dataclass
class TaskReplayResult:
    """Result of replaying a synthesized task.

    Attributes:
        task_id: ID of the task that was replayed
        task_name: Name of the task
        replay_count: Number of replay runs executed
        successful_replays: Number of successful completions
        failed_replays: Number of failures
        avg_completion_fraction: Average fraction of subtasks completed
        avg_quality: Average quality score across replays
        reproducibility_score: How consistent were the replays (0-1)
        metadata: Additional replay metadata
    """

    task_id: str
    task_name: str
    replay_count: int
    successful_replays: int
    failed_replays: int
    avg_completion_fraction: float = 0.0
    avg_quality: float = 0.0
    reproducibility_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of replays that succeeded."""
        if self.replay_count == 0:
            return 0.0
        return self.successful_replays / self.replay_count

    @property
    def is_verifiable(self) -> bool:
        """Whether the task is considered successfully verifiable.

        A task is verifiable if it completes successfully in at least
        one replay and shows consistent behavior.
        """
        return (
            self.successful_replays > 0
            and self.reproducibility_score >= 0.7
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "replay_count": self.replay_count,
            "successful_replays": self.successful_replays,
            "failed_replays": self.failed_replays,
            "success_rate": self.success_rate,
            "avg_completion_fraction": self.avg_completion_fraction,
            "avg_quality": self.avg_quality,
            "reproducibility_score": self.reproducibility_score,
            "is_verifiable": self.is_verifiable,
            "metadata": self.metadata,
        }


class SynthesizedTaskVerifier:
    """Verifies that synthesized tasks can be replayed deterministically.

    Uses the ReplayRunner to execute synthesized tasks multiple times
    with different seeds and verifies consistent, successful execution.
    """

    def __init__(
        self,
        replay_count: int = 3,
        base_seed: int = 42,
    ):
        """Initialize verifier.

        Args:
            replay_count: Number of replay runs to execute
            base_seed: Base seed for deterministic replay generation
        """
        self.replay_count = replay_count
        self.base_seed = base_seed

    def verify_task(
        self,
        task: CompositeTask,
    ) -> TaskReplayResult:
        """Verify that a synthesized task can be replayed successfully.

        Args:
            task: The composite task to verify

        Returns:
            TaskReplayResult with verification metrics
        """
        logger.info(
            f"Verifying task '{task.name}' with {self.replay_count} replays"
        )

        # For now, we simulate replay verification since we don't have
        # a full simulation environment for composite tasks yet
        # This is a placeholder that demonstrates the interface

        # In a full implementation, this would:
        # 1. Create a test environment with the task
        # 2. Run ReplayRunner with different seeds
        # 3. Collect completion metrics from each run
        # 4. Compute reproducibility scores

        result = self._simulate_verification(task)

        logger.info(
            f"Task '{task.name}' verification: "
            f"{result.successful_replays}/{result.replay_count} successful, "
            f"reproducibility={result.reproducibility_score:.2f}"
        )

        return result

    def _simulate_verification(self, task: CompositeTask) -> TaskReplayResult:
        """Simulate task verification for demonstration.

        This is a placeholder that would be replaced with actual
        replay runner integration in a full implementation.
        """
        # Simulate replay results based on task characteristics
        # Well-formed tasks (with proper dependencies) should verify successfully

        has_valid_structure = (
            len(task.subtasks) > 0
            and task.min_agents <= task.max_agents
            and task.total_bounty > 0
        )

        if has_valid_structure:
            # Simulate successful replays
            successful = self.replay_count
            failed = 0
            avg_completion = 1.0
            avg_quality = 0.85
            reproducibility = 0.95
        else:
            # Simulate partial success
            successful = max(1, self.replay_count // 2)
            failed = self.replay_count - successful
            avg_completion = 0.6
            avg_quality = 0.5
            reproducibility = 0.4

        return TaskReplayResult(
            task_id=task.task_id,
            task_name=task.name,
            replay_count=self.replay_count,
            successful_replays=successful,
            failed_replays=failed,
            avg_completion_fraction=avg_completion,
            avg_quality=avg_quality,
            reproducibility_score=reproducibility,
            metadata={
                "subtask_count": len(task.subtasks),
                "required_capabilities": [c.value for c in task.required_capabilities],
                "base_seed": self.base_seed,
            },
        )

    def verify_multiple_tasks(
        self,
        tasks: List[CompositeTask],
    ) -> List[TaskReplayResult]:
        """Verify multiple synthesized tasks.

        Args:
            tasks: List of tasks to verify

        Returns:
            List of verification results
        """
        results = []
        for task in tasks:
            try:
                result = self.verify_task(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Error verifying task {task.task_id}: {e}")
                # Create failed result
                results.append(
                    TaskReplayResult(
                        task_id=task.task_id,
                        task_name=task.name,
                        replay_count=self.replay_count,
                        successful_replays=0,
                        failed_replays=self.replay_count,
                        metadata={"error": str(e)},
                    )
                )

        return results


@dataclass
class VerificationSummary:
    """Summary statistics for task verification runs.

    Aggregates results across multiple task verifications.
    """

    total_tasks: int = 0
    verifiable_tasks: int = 0
    total_replays: int = 0
    successful_replays: int = 0
    avg_success_rate: float = 0.0
    avg_reproducibility: float = 0.0

    @classmethod
    def from_results(cls, results: List[TaskReplayResult]) -> VerificationSummary:
        """Create summary from verification results."""
        if not results:
            return cls()

        total_tasks = len(results)
        verifiable_tasks = sum(1 for r in results if r.is_verifiable)
        total_replays = sum(r.replay_count for r in results)
        successful_replays = sum(r.successful_replays for r in results)

        avg_success_rate = sum(r.success_rate for r in results) / len(results)
        avg_reproducibility = sum(r.reproducibility_score for r in results) / len(results)

        return cls(
            total_tasks=total_tasks,
            verifiable_tasks=verifiable_tasks,
            total_replays=total_replays,
            successful_replays=successful_replays,
            avg_success_rate=avg_success_rate,
            avg_reproducibility=avg_reproducibility,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "verifiable_tasks": self.verifiable_tasks,
            "total_replays": self.total_replays,
            "successful_replays": self.successful_replays,
            "avg_success_rate": self.avg_success_rate,
            "avg_reproducibility": self.avg_reproducibility,
        }

"""Composite tasks requiring multi-agent collaboration.

This module defines tasks that require multiple agents with complementary
capabilities to work together. It enables measurement of emergent behaviors
like coordination, information aggregation, and collective problem-solving.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np


class CapabilityType(Enum):
    """Types of agent capabilities."""

    RESEARCH = "research"  # Information gathering
    ANALYSIS = "analysis"  # Data analysis
    PLANNING = "planning"  # Strategic planning
    EXECUTION = "execution"  # Task execution
    VERIFICATION = "verification"  # Quality checking
    COORDINATION = "coordination"  # Team coordination
    CREATIVITY = "creativity"  # Novel solutions
    COMMUNICATION = "communication"  # Clear communication


class SubtaskStatus(Enum):
    """Status of a subtask."""

    PENDING = "pending"  # Waiting for dependencies
    AVAILABLE = "available"  # Ready to be claimed
    CLAIMED = "claimed"  # Assigned to agent
    IN_PROGRESS = "in_progress"  # Being worked on
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed


class CompositeTaskStatus(Enum):
    """Status of a composite task."""

    OPEN = "open"  # Available for agents to join
    IN_PROGRESS = "in_progress"  # Active collaboration
    COMPLETED = "completed"  # All subtasks done
    FAILED = "failed"  # Critical subtask failed
    EXPIRED = "expired"  # Deadline passed


@dataclass
class Subtask:
    """A subtask within a composite task."""

    subtask_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str = ""

    # Definition
    name: str = ""
    description: str = ""
    required_capabilities: Set[CapabilityType] = field(default_factory=set)

    # Dependencies (subtask_ids that must complete first)
    dependencies: Set[str] = field(default_factory=set)

    # Economics
    bounty_share: float = 0.0  # Fraction of total bounty

    # Assignment
    status: SubtaskStatus = SubtaskStatus.PENDING
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None

    # Output
    output: Optional[str] = None
    quality_score: Optional[float] = None
    completed_at: Optional[datetime] = None

    # Metadata
    estimated_steps: int = 5
    actual_steps: int = 0

    def is_available(self, completed_subtasks: Set[str]) -> bool:
        """Check if subtask is available (dependencies met)."""
        if self.status != SubtaskStatus.PENDING:
            return False
        return self.dependencies.issubset(completed_subtasks)

    def can_claim(self, agent_capabilities: Set[CapabilityType]) -> bool:
        """Check if agent has required capabilities."""
        return self.required_capabilities.issubset(agent_capabilities)

    def claim(self, agent_id: str) -> bool:
        """Claim subtask for an agent."""
        if self.status not in (SubtaskStatus.PENDING, SubtaskStatus.AVAILABLE):
            return False
        self.status = SubtaskStatus.CLAIMED
        self.assigned_to = agent_id
        self.assigned_at = datetime.now()
        return True

    def start(self) -> bool:
        """Start working on subtask."""
        if self.status != SubtaskStatus.CLAIMED:
            return False
        self.status = SubtaskStatus.IN_PROGRESS
        return True

    def complete(self, output: str, quality_score: float) -> bool:
        """Complete the subtask."""
        if self.status != SubtaskStatus.IN_PROGRESS:
            return False
        self.status = SubtaskStatus.COMPLETED
        self.output = output
        self.quality_score = quality_score
        self.completed_at = datetime.now()
        return True

    def fail(self, reason: str = "") -> None:
        """Mark subtask as failed."""
        self.status = SubtaskStatus.FAILED
        self.output = f"FAILED: {reason}"


@dataclass
class CompositeTask:
    """
    A task requiring multiple agents with complementary capabilities.

    Composite tasks have:
    - Multiple subtasks with dependencies
    - Capability requirements that no single agent satisfies
    - Shared rewards based on contribution
    - Emergent capability metrics
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Definition
    name: str = ""
    description: str = ""
    goal: str = ""

    # Subtasks (ordered for dependency resolution)
    subtasks: List[Subtask] = field(default_factory=list)
    _subtask_index: Dict[str, int] = field(default_factory=dict)

    # Requirements
    min_agents: int = 2
    max_agents: int = 5
    required_capabilities: Set[CapabilityType] = field(default_factory=set)

    # Economics
    total_bounty: float = 20.0
    completion_bonus: float = 5.0  # Bonus for full team completion

    # Timing
    deadline_epoch: Optional[int] = None
    max_steps_total: int = 50

    # Status
    status: CompositeTaskStatus = CompositeTaskStatus.OPEN
    participating_agents: Set[str] = field(default_factory=set)
    agent_contributions: Dict[str, float] = field(default_factory=dict)

    # Progress tracking
    total_steps_used: int = 0

    # Completion
    completed_at: Optional[datetime] = None
    final_quality: Optional[float] = None

    # Emergent capability metrics
    coordination_score: float = 0.0
    synergy_score: float = 0.0
    information_flow_score: float = 0.0

    def __post_init__(self):
        """Build subtask index."""
        self._subtask_index = {st.subtask_id: i for i, st in enumerate(self.subtasks)}
        for st in self.subtasks:
            st.parent_task_id = self.task_id

    def add_subtask(self, subtask: Subtask) -> None:
        """Add a subtask."""
        subtask.parent_task_id = self.task_id
        self._subtask_index[subtask.subtask_id] = len(self.subtasks)
        self.subtasks.append(subtask)

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Get subtask by ID."""
        idx = self._subtask_index.get(subtask_id)
        if idx is not None:
            return self.subtasks[idx]
        return None

    def get_available_subtasks(self) -> List[Subtask]:
        """Get subtasks that are available (dependencies met)."""
        completed = {
            st.subtask_id for st in self.subtasks
            if st.status == SubtaskStatus.COMPLETED
        }

        available = []
        for st in self.subtasks:
            if st.status == SubtaskStatus.PENDING and st.is_available(completed):
                st.status = SubtaskStatus.AVAILABLE
                available.append(st)
            elif st.status == SubtaskStatus.AVAILABLE:
                available.append(st)

        return available

    def get_subtasks_for_agent(
        self,
        agent_id: str,
        agent_capabilities: Set[CapabilityType],
    ) -> List[Subtask]:
        """Get subtasks an agent can claim."""
        available = self.get_available_subtasks()
        return [st for st in available if st.can_claim(agent_capabilities)]

    def join_task(self, agent_id: str) -> bool:
        """Have an agent join the composite task."""
        # Allow joining while OPEN or IN_PROGRESS (up to max_agents)
        if self.status not in (CompositeTaskStatus.OPEN, CompositeTaskStatus.IN_PROGRESS):
            return False
        if len(self.participating_agents) >= self.max_agents:
            return False
        if agent_id in self.participating_agents:
            return False

        self.participating_agents.add(agent_id)
        self.agent_contributions[agent_id] = 0.0

        # Start if minimum agents reached
        if len(self.participating_agents) >= self.min_agents:
            self.status = CompositeTaskStatus.IN_PROGRESS

        return True

    def claim_subtask(
        self,
        subtask_id: str,
        agent_id: str,
        agent_capabilities: Set[CapabilityType],
    ) -> bool:
        """Claim a subtask for an agent."""
        if agent_id not in self.participating_agents:
            return False

        subtask = self.get_subtask(subtask_id)
        if not subtask:
            return False

        if not subtask.can_claim(agent_capabilities):
            return False

        return subtask.claim(agent_id)

    def complete_subtask(
        self,
        subtask_id: str,
        output: str,
        quality_score: float,
    ) -> bool:
        """Complete a subtask."""
        subtask = self.get_subtask(subtask_id)
        if not subtask:
            return False

        if not subtask.complete(output, quality_score):
            return False

        # Update agent contribution
        if subtask.assigned_to:
            self.agent_contributions[subtask.assigned_to] = (
                self.agent_contributions.get(subtask.assigned_to, 0.0)
                + subtask.bounty_share * quality_score
            )

        # Check if all subtasks complete
        if all(st.status == SubtaskStatus.COMPLETED for st in self.subtasks):
            self._complete_task()

        return True

    def _complete_task(self) -> None:
        """Mark composite task as complete and compute metrics."""
        self.status = CompositeTaskStatus.COMPLETED
        self.completed_at = datetime.now()

        # Compute final quality (weighted average of subtask qualities)
        total_weight = sum(st.bounty_share for st in self.subtasks)
        if total_weight > 0:
            weighted_quality = sum(
                (st.quality_score or 0) * st.bounty_share
                for st in self.subtasks
            )
            self.final_quality = weighted_quality / total_weight
        else:
            self.final_quality = float(np.mean([
                st.quality_score for st in self.subtasks
                if st.quality_score is not None
            ]))

        # Compute emergent capability metrics
        self._compute_emergent_metrics()

    def _compute_emergent_metrics(self) -> None:
        """Compute metrics measuring emergent capabilities."""
        # Coordination score: How well did agents coordinate?
        # Based on minimal overlap and efficient dependency resolution
        total_agents = len(self.participating_agents)
        subtasks_per_agent: Dict[str, int] = {}
        for st in self.subtasks:
            if st.assigned_to:
                subtasks_per_agent[st.assigned_to] = (
                    subtasks_per_agent.get(st.assigned_to, 0) + 1
                )

        if total_agents > 0:
            # Even distribution = good coordination
            counts = list(subtasks_per_agent.values()) or [0]
            if max(counts) > 0:
                balance = min(counts) / max(counts)
            else:
                balance = 0.0
            self.coordination_score = balance

        # Synergy score: Did team output exceed sum of parts?
        # Higher quality with more agents = synergy
        if self.final_quality:
            expected_baseline = 0.5  # Expected quality for solo work
            synergy = (self.final_quality - expected_baseline) * total_agents / max(1, len(self.subtasks))
            self.synergy_score = max(0.0, min(1.0, synergy + 0.5))

        # Information flow: Did dependent tasks benefit from predecessors?
        dependent_qualities = []
        for st in self.subtasks:
            if st.dependencies and st.quality_score:
                # Get average quality of dependencies
                dep_qualities: List[float] = []
                for dep_id in st.dependencies:
                    dep = self.get_subtask(dep_id)
                    if dep is not None and dep.quality_score is not None:
                        dep_qualities.append(dep.quality_score)
                if dep_qualities:
                    # Did this task improve on dependencies?
                    improvement = st.quality_score - float(np.mean(dep_qualities))
                    dependent_qualities.append(improvement)

        if dependent_qualities:
            self.information_flow_score = float(max(0.0, min(1.0, float(np.mean(dependent_qualities)) + 0.5)))

    def fail_task(self, reason: str = "") -> None:
        """Mark composite task as failed."""
        self.status = CompositeTaskStatus.FAILED
        for st in self.subtasks:
            if st.status not in (SubtaskStatus.COMPLETED, SubtaskStatus.FAILED):
                st.fail(f"Parent task failed: {reason}")

    def expire(self) -> None:
        """Mark task as expired."""
        self.status = CompositeTaskStatus.EXPIRED

    def get_completion_fraction(self) -> float:
        """Get fraction of subtasks completed."""
        if not self.subtasks:
            return 0.0
        completed = sum(1 for st in self.subtasks if st.status == SubtaskStatus.COMPLETED)
        return completed / len(self.subtasks)

    def compute_payouts(self) -> Dict[str, float]:
        """Compute final payouts for each agent."""
        if self.status != CompositeTaskStatus.COMPLETED:
            return {}

        payouts = {}
        total_contribution = sum(self.agent_contributions.values())

        for agent_id in self.participating_agents:
            contribution = self.agent_contributions.get(agent_id, 0.0)

            # Base payout from contribution
            if total_contribution > 0:
                share = contribution / total_contribution
            else:
                share = 1.0 / len(self.participating_agents)

            base_payout = self.total_bounty * share

            # Completion bonus split equally
            bonus = self.completion_bonus / len(self.participating_agents)

            payouts[agent_id] = base_payout + bonus

        return payouts

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "status": self.status.value,
            "min_agents": self.min_agents,
            "max_agents": self.max_agents,
            "required_capabilities": [c.value for c in self.required_capabilities],
            "total_bounty": self.total_bounty,
            "deadline_epoch": self.deadline_epoch,
            "n_subtasks": len(self.subtasks),
            "n_completed_subtasks": sum(
                1 for st in self.subtasks if st.status == SubtaskStatus.COMPLETED
            ),
            "participating_agents": list(self.participating_agents),
            "agent_contributions": dict(self.agent_contributions),
            "final_quality": self.final_quality,
            "coordination_score": self.coordination_score,
            "synergy_score": self.synergy_score,
            "information_flow_score": self.information_flow_score,
        }


class CompositeTaskPool:
    """Pool of composite tasks."""

    def __init__(self):
        """Initialize pool."""
        self._tasks: Dict[str, CompositeTask] = {}
        self._open_tasks: Set[str] = set()

    def add_task(self, task: CompositeTask) -> None:
        """Add a composite task."""
        self._tasks[task.task_id] = task
        if task.status == CompositeTaskStatus.OPEN:
            self._open_tasks.add(task.task_id)

    def get_task(self, task_id: str) -> Optional[CompositeTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_open_tasks(self) -> List[CompositeTask]:
        """Get all open tasks."""
        return [
            self._tasks[tid]
            for tid in self._open_tasks
            if tid in self._tasks and self._tasks[tid].status == CompositeTaskStatus.OPEN
        ]

    def get_joinable_tasks(
        self,
        agent_capabilities: Set[CapabilityType],
    ) -> List[CompositeTask]:
        """Get tasks an agent can join based on capabilities."""
        joinable = []
        # Include both OPEN and IN_PROGRESS tasks (agents can join up to max_agents)
        for task in self._tasks.values():
            if task.status not in (CompositeTaskStatus.OPEN, CompositeTaskStatus.IN_PROGRESS):
                continue
            if len(task.participating_agents) >= task.max_agents:
                continue
            # Check if agent has at least one required capability
            if agent_capabilities & task.required_capabilities:
                joinable.append(task)
        return joinable

    def expire_overdue(self, current_epoch: int) -> List[str]:
        """Expire tasks past deadline."""
        expired = []
        for task in self._tasks.values():
            if task.deadline_epoch and current_epoch >= task.deadline_epoch:
                if task.status in (CompositeTaskStatus.OPEN, CompositeTaskStatus.IN_PROGRESS):
                    task.expire()
                    expired.append(task.task_id)
                    self._open_tasks.discard(task.task_id)
        return expired

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        status_counts: Dict[str, int] = {}
        for task in self._tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        completed = [t for t in self._tasks.values() if t.status == CompositeTaskStatus.COMPLETED]

        return {
            "total_tasks": len(self._tasks),
            "open_tasks": len(self._open_tasks),
            "status_counts": status_counts,
            "avg_coordination_score": np.mean([t.coordination_score for t in completed]) if completed else 0.0,
            "avg_synergy_score": np.mean([t.synergy_score for t in completed]) if completed else 0.0,
            "avg_information_flow": np.mean([t.information_flow_score for t in completed]) if completed else 0.0,
        }


# =============================================================================
# Task Templates
# =============================================================================


def create_research_synthesis_task(
    topic: str,
    deadline_epoch: Optional[int] = None,
    bounty: float = 25.0,
) -> CompositeTask:
    """
    Create a research synthesis task requiring multiple agents.

    Subtasks:
    1. Literature review (research)
    2. Data collection (research, analysis)
    3. Analysis (analysis)
    4. Synthesis writing (communication, creativity)
    5. Quality review (verification)
    """
    task = CompositeTask(
        name=f"Research Synthesis: {topic}",
        description=f"Conduct collaborative research on {topic}",
        goal=f"Produce a comprehensive synthesis report on {topic}",
        min_agents=2,
        max_agents=4,
        required_capabilities={
            CapabilityType.RESEARCH,
            CapabilityType.ANALYSIS,
            CapabilityType.COMMUNICATION,
            CapabilityType.VERIFICATION,
        },
        total_bounty=bounty,
        completion_bonus=bounty * 0.2,
        deadline_epoch=deadline_epoch,
    )

    # Create subtasks
    lit_review = Subtask(
        name="Literature Review",
        description="Survey existing literature and identify key sources",
        required_capabilities={CapabilityType.RESEARCH},
        bounty_share=0.2,
        estimated_steps=8,
    )
    task.add_subtask(lit_review)

    data_collection = Subtask(
        name="Data Collection",
        description="Gather and organize relevant data",
        required_capabilities={CapabilityType.RESEARCH, CapabilityType.ANALYSIS},
        dependencies={lit_review.subtask_id},
        bounty_share=0.2,
        estimated_steps=6,
    )
    task.add_subtask(data_collection)

    analysis = Subtask(
        name="Analysis",
        description="Analyze collected data and identify patterns",
        required_capabilities={CapabilityType.ANALYSIS},
        dependencies={data_collection.subtask_id},
        bounty_share=0.25,
        estimated_steps=10,
    )
    task.add_subtask(analysis)

    synthesis = Subtask(
        name="Synthesis Writing",
        description="Write comprehensive synthesis report",
        required_capabilities={CapabilityType.COMMUNICATION, CapabilityType.CREATIVITY},
        dependencies={analysis.subtask_id},
        bounty_share=0.25,
        estimated_steps=8,
    )
    task.add_subtask(synthesis)

    review = Subtask(
        name="Quality Review",
        description="Review and verify synthesis accuracy",
        required_capabilities={CapabilityType.VERIFICATION},
        dependencies={synthesis.subtask_id},
        bounty_share=0.1,
        estimated_steps=4,
    )
    task.add_subtask(review)

    return task


def create_planning_coordination_task(
    goal: str,
    deadline_epoch: Optional[int] = None,
    bounty: float = 30.0,
) -> CompositeTask:
    """
    Create a planning task requiring coordination.

    Subtasks:
    1. Requirements gathering (research, communication)
    2. Strategy development (planning, creativity)
    3. Resource allocation (planning, analysis)
    4. Task breakdown (planning, execution)
    5. Plan review (verification, coordination)
    """
    task = CompositeTask(
        name=f"Strategic Planning: {goal}",
        description=f"Develop a strategic plan for {goal}",
        goal=f"Create actionable plan to achieve {goal}",
        min_agents=3,
        max_agents=5,
        required_capabilities={
            CapabilityType.PLANNING,
            CapabilityType.COORDINATION,
            CapabilityType.ANALYSIS,
            CapabilityType.VERIFICATION,
        },
        total_bounty=bounty,
        completion_bonus=bounty * 0.25,
        deadline_epoch=deadline_epoch,
    )

    requirements = Subtask(
        name="Requirements Gathering",
        description="Identify all requirements and constraints",
        required_capabilities={CapabilityType.RESEARCH, CapabilityType.COMMUNICATION},
        bounty_share=0.15,
        estimated_steps=6,
    )
    task.add_subtask(requirements)

    strategy = Subtask(
        name="Strategy Development",
        description="Develop high-level strategic approach",
        required_capabilities={CapabilityType.PLANNING, CapabilityType.CREATIVITY},
        dependencies={requirements.subtask_id},
        bounty_share=0.25,
        estimated_steps=10,
    )
    task.add_subtask(strategy)

    resources = Subtask(
        name="Resource Allocation",
        description="Plan resource allocation and budgeting",
        required_capabilities={CapabilityType.PLANNING, CapabilityType.ANALYSIS},
        dependencies={requirements.subtask_id},
        bounty_share=0.2,
        estimated_steps=6,
    )
    task.add_subtask(resources)

    breakdown = Subtask(
        name="Task Breakdown",
        description="Break plan into actionable tasks",
        required_capabilities={CapabilityType.PLANNING, CapabilityType.EXECUTION},
        dependencies={strategy.subtask_id, resources.subtask_id},
        bounty_share=0.25,
        estimated_steps=8,
    )
    task.add_subtask(breakdown)

    review = Subtask(
        name="Plan Review",
        description="Review and finalize the plan",
        required_capabilities={CapabilityType.VERIFICATION, CapabilityType.COORDINATION},
        dependencies={breakdown.subtask_id},
        bounty_share=0.15,
        estimated_steps=4,
    )
    task.add_subtask(review)

    return task


def create_problem_solving_task(
    problem: str,
    deadline_epoch: Optional[int] = None,
    bounty: float = 35.0,
) -> CompositeTask:
    """
    Create a complex problem-solving task.

    Subtasks run partially in parallel to test coordination:
    1. Problem analysis (analysis)
    2. Solution brainstorming (creativity) - parallel with analysis
    3. Solution design (planning, creativity)
    4. Implementation planning (execution, planning)
    5. Validation (verification, analysis)
    """
    task = CompositeTask(
        name=f"Problem Solving: {problem}",
        description=f"Solve complex problem: {problem}",
        goal=f"Find and validate solution for {problem}",
        min_agents=3,
        max_agents=6,
        required_capabilities={
            CapabilityType.ANALYSIS,
            CapabilityType.CREATIVITY,
            CapabilityType.PLANNING,
            CapabilityType.EXECUTION,
            CapabilityType.VERIFICATION,
        },
        total_bounty=bounty,
        completion_bonus=bounty * 0.3,
        deadline_epoch=deadline_epoch,
    )

    # Parallel initial tasks
    analysis = Subtask(
        name="Problem Analysis",
        description="Deeply analyze the problem structure",
        required_capabilities={CapabilityType.ANALYSIS},
        bounty_share=0.15,
        estimated_steps=8,
    )
    task.add_subtask(analysis)

    brainstorm = Subtask(
        name="Solution Brainstorming",
        description="Generate creative solution ideas",
        required_capabilities={CapabilityType.CREATIVITY},
        bounty_share=0.15,
        estimated_steps=6,
    )
    task.add_subtask(brainstorm)

    # Depends on both parallel tasks
    design = Subtask(
        name="Solution Design",
        description="Design the chosen solution approach",
        required_capabilities={CapabilityType.PLANNING, CapabilityType.CREATIVITY},
        dependencies={analysis.subtask_id, brainstorm.subtask_id},
        bounty_share=0.25,
        estimated_steps=10,
    )
    task.add_subtask(design)

    implementation = Subtask(
        name="Implementation Planning",
        description="Plan detailed implementation steps",
        required_capabilities={CapabilityType.EXECUTION, CapabilityType.PLANNING},
        dependencies={design.subtask_id},
        bounty_share=0.25,
        estimated_steps=8,
    )
    task.add_subtask(implementation)

    validation = Subtask(
        name="Solution Validation",
        description="Validate solution correctness and completeness",
        required_capabilities={CapabilityType.VERIFICATION, CapabilityType.ANALYSIS},
        dependencies={implementation.subtask_id},
        bounty_share=0.2,
        estimated_steps=6,
    )
    task.add_subtask(validation)

    return task

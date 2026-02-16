"""Ralph Loop Agent — artifact-mediated memory agent variant.

Models an autonomous agent loop where each iteration (epoch) spawns a fresh
AI instance that works one task from a PRD-like queue.  Memory persists via
artifacts: the task queue (prd.json analog), an append-only learnings log
(progress.txt analog), and consolidated patterns (AGENTS.md analog).

The key insight is that counterparty trust resets each epoch (rain-like), but
strategic learnings survive because they live in separate data structures
that ``apply_memory_decay`` does not touch.

Based on the snarktank/ralph autonomous agent loop architecture.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from swarm.agents.base import Action, Observation, Role
from swarm.agents.honest import HonestAgent
from swarm.agents.memory_config import MemoryConfig
from swarm.models.agent import AgentType


@dataclass
class TaskSpec:
    """A single task in the Ralph PRD queue."""

    name: str
    priority: int = 1  # lower = higher priority
    completed: bool = False
    attempts: int = 0


@dataclass
class LearningEntry:
    """A single learning recorded after a task attempt."""

    epoch: int
    task_name: str
    success: bool
    p: float  # quality of the interaction
    notes: str = ""


# Default task queue when none is provided via config
_DEFAULT_TASKS = [
    "Implement core feature",
    "Write integration tests",
    "Refactor error handling",
    "Add monitoring hooks",
    "Update documentation",
]


class RalphLoopAgent(HonestAgent):
    """
    Artifact-mediated memory agent modeled on the Ralph autonomous loop.

    Each epoch simulates a fresh agent instance that:
    1. Selects the highest-priority incomplete task from the queue
    2. Works on that single task (one-task-per-epoch constraint)
    3. Records learnings on success/failure
    4. Periodically consolidates patterns from learnings

    Memory model (MemoryConfig mapping):
    - epistemic_persistence = 0.0  -> counterparty trust resets each epoch
    - goal_persistence = 1.0       -> task queue persists across epochs
    - strategy_persistence = 0.0   -> base strategy resets, BUT consolidated
      learnings survive (separate data structures)

    Persistent state (survives ``apply_memory_decay``):
    - ``_task_queue``              -> PRD / backlog
    - ``_learnings``               -> append-only progress log
    - ``_consolidated_patterns``   -> compressed strategic insights

    Epoch-local state (reset each epoch):
    - ``_current_task``
    - ``_epoch_quality_samples``
    - ``_task_completed_this_epoch``
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        # Ralph memory model: forget counterparties, keep goals
        memory_config = MemoryConfig(
            epistemic_persistence=0.0,
            goal_persistence=1.0,
            strategy_persistence=0.0,
        )
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # ── Config parameters ──────────────────────────────────────
        self.quality_gate_threshold: float = self.config.get(
            "quality_gate_threshold", 0.6
        )
        self.consolidation_interval: int = self.config.get(
            "consolidation_interval", 5
        )
        self.max_task_attempts: int = self.config.get("max_task_attempts", 3)
        self.learning_boost: float = self.config.get("learning_boost", 0.05)
        self.one_task_per_epoch: bool = self.config.get("one_task_per_epoch", True)

        # ── Persistent state (artifact-mediated) ───────────────────
        self._task_queue: List[TaskSpec] = self._build_task_queue()
        self._learnings: List[LearningEntry] = []
        self._consolidated_patterns: Dict[str, float] = {
            "success_rate": 0.5,
            "quality_awareness": 0.5,
            "completion_rate": 0.0,
        }

        # ── Epoch-local state ──────────────────────────────────────
        self._current_task: Optional[TaskSpec] = None
        self._epoch_quality_samples: List[float] = []
        self._task_completed_this_epoch: bool = False

    # ── Task queue construction ────────────────────────────────────

    def _build_task_queue(self) -> List[TaskSpec]:
        """Build task queue from config or defaults."""
        raw_tasks = self.config.get("tasks", _DEFAULT_TASKS)
        queue: List[TaskSpec] = []
        for i, item in enumerate(raw_tasks):
            if isinstance(item, str):
                queue.append(TaskSpec(name=item, priority=i + 1))
            elif isinstance(item, dict):
                queue.append(
                    TaskSpec(
                        name=item.get("name", f"task_{i}"),
                        priority=item.get("priority", i + 1),
                    )
                )
        return queue

    # ── Core overrides ─────────────────────────────────────────────

    def act(self, observation: Observation) -> Action:
        """Select one task per epoch, apply learning boosts, then delegate."""
        # Select a task if we haven't yet this epoch
        if self._current_task is None and not self._task_completed_this_epoch:
            self._current_task = self._select_next_task()

        # If one-task-per-epoch and we already completed, noop
        if self.one_task_per_epoch and self._task_completed_this_epoch:
            return self.create_noop_action()

        # Apply learning boosts before acting
        self._apply_learning_boosts()

        # Delegate to HonestAgent for actual action selection
        return super().act(observation)

    def update_from_outcome(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> None:
        """Quality-gate task completion and record learnings."""
        # Let base class handle trust updates
        super().update_from_outcome(interaction, payoff)

        # Track quality this epoch
        self._epoch_quality_samples.append(interaction.p)

        # Quality gate: only count as completion if p is high enough
        if self._current_task is not None:
            if interaction.p >= self.quality_gate_threshold:
                # Success — mark task completed
                self._current_task.completed = True
                self._task_completed_this_epoch = True
                self._learnings.append(
                    LearningEntry(
                        epoch=0,  # will be set properly in decay
                        task_name=self._current_task.name,
                        success=True,
                        p=interaction.p,
                        notes=f"Passed quality gate (p={interaction.p:.3f})",
                    )
                )
            else:
                # Failure — increment attempts
                self._current_task.attempts += 1
                self._learnings.append(
                    LearningEntry(
                        epoch=0,
                        task_name=self._current_task.name,
                        success=False,
                        p=interaction.p,
                        notes=(
                            f"Failed quality gate (p={interaction.p:.3f} "
                            f"< {self.quality_gate_threshold}), "
                            f"attempt {self._current_task.attempts}"
                        ),
                    )
                )

                # Abandon task if max attempts exceeded
                if self._current_task.attempts >= self.max_task_attempts:
                    self._current_task.completed = True  # Mark done (abandoned)
                    self._task_completed_this_epoch = True

    def apply_memory_decay(self, epoch: int) -> None:
        """
        Apply decay at epoch boundary.

        Calls super() to clear counterparty trust (rain-like behavior),
        then resets epoch-local state.  Crucially does NOT clear
        ``_task_queue``, ``_learnings``, or ``_consolidated_patterns``.
        """
        # Clear counterparty trust via base class
        super().apply_memory_decay(epoch)

        # Reset epoch-local state
        self._current_task = None
        self._epoch_quality_samples = []
        self._task_completed_this_epoch = False

        # Periodically consolidate learnings
        if epoch > 0 and epoch % self.consolidation_interval == 0:
            self._consolidate_patterns()

    # ── Internal helpers ───────────────────────────────────────────

    def _select_next_task(self) -> Optional[TaskSpec]:
        """Select the highest-priority incomplete task."""
        incomplete = [t for t in self._task_queue if not t.completed]
        if not incomplete:
            return None
        # Sort by priority (lower = higher priority)
        incomplete.sort(key=lambda t: t.priority)
        return incomplete[0]

    def _apply_learning_boosts(self) -> None:
        """Shift behavioral parameters based on consolidated patterns."""
        sr = self._consolidated_patterns.get("success_rate", 0.5)
        qa = self._consolidated_patterns.get("quality_awareness", 0.5)

        # Higher success rate -> slightly more willing to interact
        boost = (sr - 0.5) * self.learning_boost
        self.interact_probability = max(
            0.1, min(0.9, self.interact_probability + boost)
        )

        # Higher quality awareness -> higher acceptance threshold
        qa_boost = (qa - 0.5) * self.learning_boost
        self.acceptance_threshold = max(
            0.2, min(0.8, self.acceptance_threshold + qa_boost)
        )

    def _consolidate_patterns(self) -> None:
        """Compress recent learnings into pattern summary."""
        if not self._learnings:
            return

        successes = [e for e in self._learnings if e.success]
        total = len(self._learnings)

        self._consolidated_patterns["success_rate"] = len(successes) / total

        # Average quality from successful interactions
        if successes:
            avg_p = sum(e.p for e in successes) / len(successes)
            self._consolidated_patterns["quality_awareness"] = avg_p

        # Task completion rate
        completed_tasks = sum(1 for t in self._task_queue if t.completed)
        total_tasks = len(self._task_queue)
        if total_tasks > 0:
            self._consolidated_patterns["completion_rate"] = (
                completed_tasks / total_tasks
            )

    # ── Introspection ──────────────────────────────────────────────

    @property
    def completed_tasks(self) -> List[TaskSpec]:
        """Return list of completed tasks."""
        return [t for t in self._task_queue if t.completed]

    @property
    def remaining_tasks(self) -> List[TaskSpec]:
        """Return list of remaining tasks."""
        return [t for t in self._task_queue if not t.completed]


# Need to import SoftInteraction for type annotation in update_from_outcome
from swarm.models.interaction import SoftInteraction  # noqa: E402


class AdversarialRalphAgent(RalphLoopAgent):
    """
    Adversarial variant of the Ralph loop agent.

    Same artifact-mediated memory model but with adversarial agent type
    and a lower quality gate threshold, modeling an autonomous loop that
    accumulates exploitation patterns over time.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        # Default to lower quality gate for adversarial variant
        config = config or {}
        config.setdefault("quality_gate_threshold", 0.35)
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            rng=rng,
        )
        self.agent_type = AgentType.ADVERSARIAL

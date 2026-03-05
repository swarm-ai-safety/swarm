"""SwarmEnv: the stable environment interface for SWARM-Gym.

This is one of the 3 frozen interfaces (along with GovernanceModule.apply
and the episode JSONL/summary JSON formats). Changes here must be
backwards-compatible.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from swarm_gym.utils.types import (
    Action,
    AgentId,
    AgentRecord,
    Event,
    GovernanceSnapshot,
    Observation,
    StepMetrics,
)


@dataclass
class StepResult:
    """Return value of SwarmEnv.step().

    Gym returns (obs, reward, done, info).
    SWARM-Gym returns observations + system-level metrics + governance events.
    """

    observations: Dict[AgentId, Observation]
    rewards: Dict[AgentId, float]
    done: bool
    metrics: StepMetrics
    events: List[Event]
    governance: GovernanceSnapshot
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResetResult:
    """Return value of SwarmEnv.reset()."""

    observations: Dict[AgentId, Observation]
    info: Dict[str, Any] = field(default_factory=dict)


class SwarmEnv(abc.ABC):
    """Abstract base class for SWARM-Gym environments.

    Every benchmark environment implements this interface. Governance
    modules are plugged in at construction time and run every step.

    Frozen interface contract:
        - reset(seed, config) -> ResetResult
        - step(actions) -> StepResult
        - agent_ids -> List[AgentId]
        - current_step -> int
        - max_steps -> int
    """

    # Subclasses set these
    env_id: str = ""
    max_steps: int = 40

    def __init__(self) -> None:
        self._current_step: int = 0
        self._done: bool = False
        self._governance_modules: List["GovernanceModule"] = []
        self._agent_ids: List[AgentId] = []

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def agent_ids(self) -> List[AgentId]:
        return list(self._agent_ids)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def num_agents(self) -> int:
        return len(self._agent_ids)

    # ── Governance plugin API ──────────────────────────────────

    def add_governance(self, module: "GovernanceModule") -> None:
        """Attach a governance module. Order matters (applied sequentially)."""
        self._governance_modules.append(module)

    def set_governance(self, modules: List["GovernanceModule"]) -> None:
        """Replace all governance modules."""
        self._governance_modules = list(modules)

    def get_governance_state(self) -> Dict[str, Any]:
        """Aggregate governance state from all modules."""
        state: Dict[str, Any] = {}
        for mod in self._governance_modules:
            state.update(mod.get_state())
        return state

    # ── Core API (frozen) ──────────────────────────────────────

    def reset(
        self,
        seed: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            config: Optional overrides for environment configuration.

        Returns:
            ResetResult with initial observations and info dict.
        """
        self._current_step = 0
        self._done = False

        # Reset governance modules
        for mod in self._governance_modules:
            mod.reset(seed=seed)

        return self._reset_impl(seed=seed, config=config)

    def step(self, actions: List[Action]) -> StepResult:
        """Advance the environment by one step.

        Args:
            actions: Joint actions from all agents.

        Returns:
            StepResult with observations, rewards, done, metrics, events,
            governance snapshot.

        Raises:
            RuntimeError: If called after episode is done without reset.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Capture governance state before
        gov_before = self.get_governance_state()

        # Apply governance modules to proposed actions
        modified_actions = list(actions)
        all_interventions = []
        all_events: List[Event] = []

        world_state = self._get_world_state()
        for mod in self._governance_modules:
            modified_actions, interventions, events = mod.apply(
                world_state, modified_actions
            )
            all_interventions.extend(interventions)
            all_events.extend(events)

        # Run environment step with (possibly modified) actions
        result = self._step_impl(modified_actions)

        # Merge governance events
        result.events = all_events + result.events

        # Capture governance state after
        gov_after = self.get_governance_state()
        result.governance = GovernanceSnapshot(
            before=gov_before,
            after=gov_after,
            interventions=all_interventions,
        )

        self._current_step += 1
        if self._current_step >= self.max_steps or result.done:
            self._done = True
            result.done = True

        return result

    # ── Abstract methods for subclasses ────────────────────────

    @abc.abstractmethod
    def _reset_impl(
        self,
        seed: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Subclass-specific reset logic."""

    @abc.abstractmethod
    def _step_impl(self, actions: List[Action]) -> StepResult:
        """Subclass-specific step logic (after governance)."""

    @abc.abstractmethod
    def _get_world_state(self) -> Dict[str, Any]:
        """Return current world state snapshot for governance modules."""

    # ── Optional overrides ─────────────────────────────────────

    def get_agent_records(self) -> List[AgentRecord]:
        """Return agent records for episode reporting."""
        return [AgentRecord(agent_id=aid, type="unknown") for aid in self._agent_ids]

    def get_action_space(self) -> List[str]:
        """Return list of valid action types for this environment."""
        return ["noop"]

    def get_episode_outcomes(self) -> Dict[str, Any]:
        """Return end-of-episode outcome summary."""
        return {}


# Avoid circular import — GovernanceModule is defined in governance.base
# but referenced here for type hints.
from swarm_gym.governance.base import GovernanceModule  # noqa: E402

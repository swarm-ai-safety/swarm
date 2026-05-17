"""Circuit breaker governance lever."""

from dataclasses import dataclass, field
from typing import Dict, List

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


@dataclass
class AgentViolationTracker:
    """Tracks violations and freeze state for an agent."""

    violations: int = 0
    freeze_until_epoch: int = -1  # -1 means not frozen
    recent_toxicity: List[float] = field(default_factory=list)

    def add_toxicity(self, toxicity: float, window_size: int = 10) -> None:
        """Add a toxicity observation."""
        self.recent_toxicity.append(toxicity)
        if len(self.recent_toxicity) > window_size:
            self.recent_toxicity.pop(0)

    def avg_toxicity(self) -> float:
        """Get average toxicity over recent interactions."""
        if not self.recent_toxicity:
            return 0.0
        return sum(self.recent_toxicity) / len(self.recent_toxicity)


class CircuitBreakerLever(GovernanceLever):
    """
    Circuit breaker to freeze high-toxicity agents.

    Tracks per-agent toxicity (1-p for their interactions) and
    freezes agents that exceed the toxicity threshold.
    """

    def __init__(self, config: GovernanceConfig):
        super().__init__(config)
        self._trackers: Dict[str, AgentViolationTracker] = {}

    @property
    def name(self) -> str:
        return "circuit_breaker"

    def _get_tracker(self, agent_id: str) -> AgentViolationTracker:
        """Get or create tracker for an agent."""
        if agent_id not in self._trackers:
            self._trackers[agent_id] = AgentViolationTracker()
        return self._trackers[agent_id]

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """
        Check for agents to unfreeze at epoch start.

        Args:
            state: Current environment state
            epoch: The epoch number starting

        Returns:
            Effect with agents to unfreeze
        """
        if not self.config.circuit_breaker_enabled:
            return LeverEffect(lever_name=self.name)

        agents_to_unfreeze = set()
        for agent_id, tracker in self._trackers.items():
            if tracker.freeze_until_epoch >= 0 and epoch >= tracker.freeze_until_epoch:
                agents_to_unfreeze.add(agent_id)
                tracker.freeze_until_epoch = -1
                tracker.violations = 0  # Reset violations after serving freeze

        return LeverEffect(
            agents_to_unfreeze=agents_to_unfreeze,
            lever_name=self.name,
            details={
                "unfrozen_count": len(agents_to_unfreeze),
                "unfrozen_agents": list(agents_to_unfreeze),
            },
        )

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """
        Track toxicity and trigger circuit breaker if threshold exceeded.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Effect with agents to freeze
        """
        if not self.config.circuit_breaker_enabled:
            return LeverEffect(lever_name=self.name)

        # Track initiator toxicity (1 - p = probability of harm)
        initiator_id = interaction.initiator
        tracker = self._get_tracker(initiator_id)

        toxicity = 1.0 - interaction.p
        tracker.add_toxicity(toxicity)

        agents_to_freeze = set()

        # Check if avg toxicity exceeds threshold
        avg_tox = tracker.avg_toxicity()
        if avg_tox >= self.config.freeze_threshold_toxicity:
            tracker.violations += 1

            if tracker.violations >= self.config.freeze_threshold_violations:
                # Trigger freeze
                agents_to_freeze.add(initiator_id)
                tracker.freeze_until_epoch = (
                    state.current_epoch + self.config.freeze_duration_epochs
                )

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            lever_name=self.name,
            details={
                "agent_id": initiator_id,
                "toxicity": toxicity,
                "avg_toxicity": avg_tox,
                "violations": tracker.violations,
                "triggered": len(agents_to_freeze) > 0,
            },
        )

    def can_agent_act(
        self,
        agent_id: str,
        state: EnvState,
    ) -> bool:
        """Block frozen agents from acting."""
        if not self.config.circuit_breaker_enabled:
            return True
        tracker = self._get_tracker(agent_id)
        if tracker.freeze_until_epoch < 0:
            return True  # Not frozen
        return state.current_epoch >= tracker.freeze_until_epoch

    def reset_tracker(self, agent_id: str) -> None:
        """Reset tracking for an agent."""
        if agent_id in self._trackers:
            del self._trackers[agent_id]

    def get_freeze_status(self, agent_id: str) -> Dict:
        """Get current freeze status for an agent."""
        tracker = self._get_tracker(agent_id)
        return {
            "violations": tracker.violations,
            "freeze_until_epoch": tracker.freeze_until_epoch,
            "avg_toxicity": tracker.avg_toxicity(),
            "is_frozen": tracker.freeze_until_epoch >= 0,
        }

"""Governance engine that aggregates all levers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.env.state import EnvState
from src.governance.admission import StakingLever
from src.governance.audits import RandomAuditLever
from src.governance.circuit_breaker import CircuitBreakerLever
from src.governance.collusion import CollusionPenaltyLever
from src.governance.config import GovernanceConfig
from src.governance.levers import GovernanceLever, LeverEffect
from src.governance.reputation import ReputationDecayLever, VoteNormalizationLever
from src.governance.taxes import TransactionTaxLever
from src.models.interaction import SoftInteraction


@dataclass
class GovernanceEffect:
    """
    Aggregated effect from all governance levers.

    Combines effects from multiple levers into a single result.
    """

    # Costs to add to interaction
    cost_a: float = 0.0
    cost_b: float = 0.0

    # State changes
    agents_to_freeze: Set[str] = field(default_factory=set)
    agents_to_unfreeze: Set[str] = field(default_factory=set)

    # Adjustments
    reputation_deltas: Dict[str, float] = field(default_factory=dict)
    resource_deltas: Dict[str, float] = field(default_factory=dict)

    # Logging
    lever_effects: List[LeverEffect] = field(default_factory=list)

    @classmethod
    def from_lever_effects(cls, effects: List[LeverEffect]) -> "GovernanceEffect":
        """Create from list of lever effects."""
        result = cls()
        for effect in effects:
            result.cost_a += effect.cost_a
            result.cost_b += effect.cost_b
            result.agents_to_freeze |= effect.agents_to_freeze
            result.agents_to_unfreeze |= effect.agents_to_unfreeze
            for agent_id, delta in effect.reputation_deltas.items():
                result.reputation_deltas[agent_id] = (
                    result.reputation_deltas.get(agent_id, 0) + delta
                )
            for agent_id, delta in effect.resource_deltas.items():
                result.resource_deltas[agent_id] = (
                    result.resource_deltas.get(agent_id, 0) + delta
                )
            result.lever_effects.append(effect)
        return result


class GovernanceEngine:
    """
    Aggregates all governance levers and provides unified interface.

    Manages the lifecycle of governance application:
    - Epoch start hooks (reputation decay, unfreezes)
    - Interaction hooks (taxes, circuit breaker, audits)
    - Admission control (staking)
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the governance engine.

        Args:
            config: Governance configuration (uses defaults if None)
            seed: Random seed for reproducible audits
        """
        self.config = config or GovernanceConfig()
        self.config.validate()

        # Initialize all levers
        self._levers: List[GovernanceLever] = [
            TransactionTaxLever(self.config),
            ReputationDecayLever(self.config),
            StakingLever(self.config),
            CircuitBreakerLever(self.config),
            RandomAuditLever(self.config, seed=seed),
            CollusionPenaltyLever(self.config),
        ]

        # Keep references to specific levers for direct access
        self._staking_lever: Optional[StakingLever] = None
        self._circuit_breaker_lever: Optional[CircuitBreakerLever] = None
        self._collusion_lever: Optional[CollusionPenaltyLever] = None
        self._vote_normalization_lever = VoteNormalizationLever(self.config)

        for lever in self._levers:
            if isinstance(lever, StakingLever):
                self._staking_lever = lever
            elif isinstance(lever, CircuitBreakerLever):
                self._circuit_breaker_lever = lever
            elif isinstance(lever, CollusionPenaltyLever):
                self._collusion_lever = lever

    def apply_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> GovernanceEffect:
        """
        Apply all epoch-start governance hooks.

        Args:
            state: Current environment state
            epoch: The epoch number starting

        Returns:
            Aggregated governance effect
        """
        effects = []
        for lever in self._levers:
            effect = lever.on_epoch_start(state, epoch)
            if effect.lever_name:  # Non-empty effect
                effects.append(effect)
        return GovernanceEffect.from_lever_effects(effects)

    def apply_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> GovernanceEffect:
        """
        Apply all per-interaction governance hooks.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Aggregated governance effect
        """
        effects = []
        for lever in self._levers:
            effect = lever.on_interaction(interaction, state)
            if effect.lever_name:  # Non-empty effect
                effects.append(effect)
        return GovernanceEffect.from_lever_effects(effects)

    def can_agent_act(
        self,
        agent_id: str,
        state: EnvState,
    ) -> bool:
        """
        Check if agent is allowed to act (all levers must approve).

        Args:
            agent_id: Agent attempting to act
            state: Current environment state

        Returns:
            True if all levers allow the agent to act
        """
        for lever in self._levers:
            if not lever.can_agent_act(agent_id, state):
                return False
        return True

    def compute_vote_weight(
        self,
        agent_id: str,
        vote_count: int,
    ) -> float:
        """
        Compute normalized vote weight for an agent.

        Args:
            agent_id: The voting agent
            vote_count: Number of votes cast this epoch

        Returns:
            Vote weight in (0, 1]
        """
        return self._vote_normalization_lever.compute_vote_weight(agent_id, vote_count)

    def slash_agent_stake(
        self,
        agent_id: str,
        state: EnvState,
        reason: str = "violation",
    ) -> GovernanceEffect:
        """
        Slash an agent's stake.

        Args:
            agent_id: Agent to slash
            state: Current environment state
            reason: Reason for slashing

        Returns:
            Governance effect with resource delta
        """
        if self._staking_lever is None:
            return GovernanceEffect()
        effect = self._staking_lever.slash_stake(agent_id, state, reason)
        return GovernanceEffect.from_lever_effects([effect])

    def get_circuit_breaker_status(self, agent_id: str) -> Dict:
        """Get circuit breaker status for an agent."""
        if self._circuit_breaker_lever is None:
            return {}
        return self._circuit_breaker_lever.get_freeze_status(agent_id)

    def reset_circuit_breaker(self, agent_id: str) -> None:
        """Reset circuit breaker tracking for an agent."""
        if self._circuit_breaker_lever is not None:
            self._circuit_breaker_lever.reset_tracker(agent_id)

    def set_collusion_agent_ids(self, agent_ids: List[str]) -> None:
        """Set agent IDs for collusion detection."""
        if self._collusion_lever is not None:
            self._collusion_lever.set_agent_ids(agent_ids)

    def get_collusion_report(self):
        """Get the latest collusion detection report."""
        if self._collusion_lever is None:
            return None
        return self._collusion_lever.get_report()

    def clear_collusion_history(self) -> None:
        """Clear collusion detection interaction history."""
        if self._collusion_lever is not None:
            self._collusion_lever.clear_history()

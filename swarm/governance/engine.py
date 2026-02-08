"""Governance engine that aggregates all levers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from swarm.env.state import EnvState
from swarm.governance.admission import StakingLever
from swarm.governance.audits import RandomAuditLever
from swarm.governance.circuit_breaker import CircuitBreakerLever
from swarm.governance.collusion import CollusionPenaltyLever
from swarm.governance.config import GovernanceConfig
from swarm.governance.decomposition import DecompositionLever
from swarm.governance.diversity import DiversityDefenseLever
from swarm.governance.dynamic_friction import IncoherenceFrictionLever
from swarm.governance.ensemble import SelfEnsembleLever
from swarm.governance.identity_lever import SybilDetectionLever
from swarm.governance.incoherence_breaker import IncoherenceCircuitBreakerLever
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.governance.memory import (
    CrossVerificationLever,
    PromotionGateLever,
    ProvenanceLever,
    WriteRateLimitLever,
)
from swarm.governance.moderator_lever import ModeratorLever
from swarm.governance.moltbook import (
    ChallengeVerificationLever,
    MoltbookRateLimitLever,
)
from swarm.governance.moltipedia import (
    DailyPointCapLever,
    NoSelfFixLever,
    PageCooldownLever,
    PairCapLever,
)
from swarm.governance.reputation import ReputationDecayLever, VoteNormalizationLever
from swarm.governance.security import SecurityLever
from swarm.governance.taxes import TransactionTaxLever
from swarm.governance.transparency import TransparencyLever
from swarm.models.interaction import SoftInteraction


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
        self.config = GovernanceConfig() if config is None else config
        # Pydantic auto-validates

        levers: List[GovernanceLever] = [
            TransactionTaxLever(self.config),
            ReputationDecayLever(self.config),
            StakingLever(self.config),
            CircuitBreakerLever(self.config),
            RandomAuditLever(self.config, seed=seed),
            CollusionPenaltyLever(self.config),
            SecurityLever(self.config, seed=seed),
            PairCapLever(self.config),
            PageCooldownLever(self.config),
            DailyPointCapLever(self.config),
            NoSelfFixLever(self.config),
        ]
        if self.config.moltbook_rate_limit_enabled:
            levers.append(MoltbookRateLimitLever(self.config))
        if self.config.moltbook_challenge_enabled:
            levers.append(ChallengeVerificationLever(self.config))
        # Memory tier levers
        levers.append(PromotionGateLever(self.config))
        levers.append(WriteRateLimitLever(self.config))
        levers.append(CrossVerificationLever(self.config))
        levers.append(ProvenanceLever(self.config))
        # Variance-aware levers (scaffold registration; behavior in #35)
        if self.config.self_ensemble_enabled:
            levers.append(SelfEnsembleLever(self.config))
        if self.config.incoherence_breaker_enabled:
            levers.append(IncoherenceCircuitBreakerLever(self.config))
        if self.config.decomposition_enabled:
            levers.append(DecompositionLever(self.config))
        if self.config.incoherence_friction_enabled:
            levers.append(IncoherenceFrictionLever(self.config))

        # VAE paper levers
        levers.append(TransparencyLever(self.config))
        levers.append(ModeratorLever(self.config, seed=seed))
        levers.append(SybilDetectionLever(self.config))

        # Diversity as Defense lever
        levers.append(DiversityDefenseLever(self.config))

        # Stored as a tuple so that external code cannot mutate in place.
        self._levers: tuple[GovernanceLever, ...] = tuple(levers)

        # Keep references to specific levers for direct access
        self._staking_lever: Optional[StakingLever] = None
        self._circuit_breaker_lever: Optional[CircuitBreakerLever] = None
        self._collusion_lever: Optional[CollusionPenaltyLever] = None
        self._security_lever: Optional[SecurityLever] = None
        self._vote_normalization_lever = VoteNormalizationLever(self.config)
        self._moltbook_rate_limit_lever: Optional[MoltbookRateLimitLever] = None
        self._moltbook_challenge_lever: Optional[ChallengeVerificationLever] = None
        self._diversity_lever: Optional[DiversityDefenseLever] = None

        for lever in self._levers:
            if isinstance(lever, StakingLever):
                self._staking_lever = lever
            elif isinstance(lever, CircuitBreakerLever):
                self._circuit_breaker_lever = lever
            elif isinstance(lever, CollusionPenaltyLever):
                self._collusion_lever = lever
            elif isinstance(lever, SecurityLever):
                self._security_lever = lever
            elif isinstance(lever, MoltbookRateLimitLever):
                self._moltbook_rate_limit_lever = lever
            elif isinstance(lever, ChallengeVerificationLever):
                self._moltbook_challenge_lever = lever
            elif isinstance(lever, DiversityDefenseLever):
                self._diversity_lever = lever

        # Adaptive governance state
        self._incoherence_forecaster: Optional[Any] = None
        self._adaptive_risk: float = 0.0
        self._adaptive_variance_active: bool = True

    def get_moltbook_rate_limit_lever(self) -> Optional[MoltbookRateLimitLever]:
        """Return Moltbook rate limit lever if registered."""
        return self._moltbook_rate_limit_lever

    def get_moltbook_challenge_lever(self) -> Optional[ChallengeVerificationLever]:
        """Return Moltbook challenge lever if registered."""
        return self._moltbook_challenge_lever

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
        for lever in self._iter_active_levers():
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
        for lever in self._iter_active_levers():
            effect = lever.on_interaction(interaction, state)
            if effect.lever_name:  # Non-empty effect
                effects.append(effect)
        return GovernanceEffect.from_lever_effects(effects)

    def apply_step(
        self,
        state: EnvState,
        step: int,
    ) -> GovernanceEffect:
        """
        Apply step-level governance hooks.

        Args:
            state: Current environment state
            step: Current step index within epoch

        Returns:
            Aggregated governance effect
        """
        effects = []
        for lever in self._iter_active_levers():
            effect = lever.on_step(state, step)
            if effect.lever_name:
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
        for lever in self._iter_active_levers():
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

    def set_security_agent_ids(self, agent_ids: List[str]) -> None:
        """Set agent IDs for security analysis."""
        if self._security_lever is not None:
            self._security_lever.set_agent_ids(agent_ids)

    def set_security_trust_scores(self, trust_scores: Dict[str, float]) -> None:
        """Set trust scores for security analysis (laundering detection)."""
        if self._security_lever is not None:
            self._security_lever.set_agent_trust_scores(trust_scores)

    def get_security_report(self):
        """Get the latest security analysis report."""
        if self._security_lever is None:
            return None
        return self._security_lever.get_report()

    def get_active_lever_names(self) -> List[str]:
        """Return registered lever names in execution order."""
        return [lever.name for lever in self._iter_active_levers()]

    def get_registered_lever_names(self) -> List[str]:
        """Return all registered lever names (ignores adaptive gating)."""
        return [lever.name for lever in self._levers]

    def set_incoherence_forecaster(self, forecaster: Any) -> None:
        """Attach a trained forecaster used for adaptive gating."""
        self._incoherence_forecaster = forecaster

    def update_adaptive_mode(self, features: Dict[str, float]) -> float:
        """
        Update adaptive gating state from current feature vector.

        Returns:
            Predicted incoherence risk in [0, 1].
        """
        if not self.config.adaptive_governance_enabled:
            self._adaptive_variance_active = True
            self._adaptive_risk = 0.0
            return self._adaptive_risk
        if self._incoherence_forecaster is None:
            # Fail open to avoid accidental disablement without model wiring.
            self._adaptive_variance_active = True
            self._adaptive_risk = 0.0
            return self._adaptive_risk

        risk = float(self._incoherence_forecaster.predict_proba(features))
        self._adaptive_risk = risk
        self._adaptive_variance_active = (
            risk >= self.config.adaptive_incoherence_threshold
        )
        return risk

    def get_adaptive_status(self) -> Dict[str, Any]:
        """Return current adaptive gating state."""
        return {
            "adaptive_enabled": self.config.adaptive_governance_enabled,
            "variance_levers_active": self._adaptive_variance_active,
            "predicted_risk": self._adaptive_risk,
            "threshold": self.config.adaptive_incoherence_threshold,
        }

    def _iter_active_levers(self) -> List[GovernanceLever]:
        """Iterate levers after adaptive gating."""
        if not self.config.adaptive_governance_enabled:
            return list(self._levers)
        if self._incoherence_forecaster is None:
            return list(self._levers)

        variance_names = {
            "self_ensemble",
            "incoherence_breaker",
            "decomposition",
            "incoherence_friction",
        }
        active: List[GovernanceLever] = []
        for lever in self._levers:
            if lever.name in variance_names and not self._adaptive_variance_active:
                continue
            active.append(lever)
        return active

    def get_quarantined_agents(self) -> frozenset[str]:
        """Get set of quarantined agents (immutable copy)."""
        if self._security_lever is None:
            return frozenset()
        return frozenset(self._security_lever.get_quarantined_agents())

    def release_from_quarantine(self, agent_id: str) -> bool:
        """Release an agent from security quarantine."""
        if self._security_lever is None:
            return False
        return self._security_lever.release_from_quarantine(agent_id)

    def get_security_containment_actions(self) -> List[Dict]:
        """Get history of security containment actions."""
        if self._security_lever is None:
            return []
        return self._security_lever.get_containment_actions()

    def clear_security_history(self) -> None:
        """Clear security analysis history and state."""
        if self._security_lever is not None:
            self._security_lever.clear_history()

    def get_diversity_metrics(self) -> Optional[Any]:
        """Return the latest diversity metrics snapshot."""
        if self._diversity_lever is None:
            return None
        return self._diversity_lever.get_metrics()

    def get_diversity_lever(self) -> Optional[DiversityDefenseLever]:
        """Return the diversity defense lever if registered."""
        return self._diversity_lever

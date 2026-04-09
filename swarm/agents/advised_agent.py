"""Advisor-model wrapper agent (Asawa et al., 2026).

Wraps any ``BaseAgent`` and applies per-instance steering signals to
modulate its decision-making.  This is the agent-side counterpart of the
``AdvisorLever`` governance lever.

The paper trains a small advisor model to prepend natural-language advice
to a black-box frontier model's prompt.  Here we adapt the concept to
the SWARM simulation: the AdvisedAgent wraps a *base agent* (which is
treated as a black box) and modifies its observation before passing it
through, then post-processes its decisions — analogous to how the
advisor model modifies the prompt and then lets the frontier model
respond.

Design:
- The wrapper maintains a local quality tracker (EMA of recent p
  values) and an "advice signal" in [-1, +1].
- Positive advice (agent trending well) → lower acceptance threshold,
  higher willingness to interact.
- Negative advice (agent trending poorly) → raise acceptance threshold,
  inject caution.
- The base agent's policy is never modified directly — only its inputs
  (observations) and decision boundaries are adjusted, preserving the
  black-box property.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class AdvisedAgent(BaseAgent):
    """Wrapper that steers a base agent via advisor-model signals.

    The wrapper intercepts the ``act()`` and ``accept_interaction()``
    calls, adjusts observations and thresholds based on the current
    advice signal, then delegates to the wrapped base agent.

    Config keys (passed via ``config`` dict):
    - ``advisor_alpha``: EMA learning rate for quality tracking (default 0.3)
    - ``advisor_strength``: scaling factor for advice effects (default 1.0)
    - ``advisor_threshold_bonus``: max threshold adjustment (default 0.15)
    - ``base_agent_type``: type key of the wrapped agent (default "honest")
    - ``base_agent_config``: config dict for the wrapped agent (default {})

    The wrapped base agent is created from the ``base_agent_type`` and
    ``base_agent_config`` at init time.  Any additional config keys are
    forwarded to the base agent.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        # We use HONEST as the nominal type since the wrapper can wrap
        # any agent type — the true behavioral type comes from the base.
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # Advisor parameters
        self._advisor_alpha = self.config.get("advisor_alpha", 0.3)
        self._advisor_strength = self.config.get("advisor_strength", 1.0)
        self._threshold_bonus = self.config.get("advisor_threshold_bonus", 0.15)

        # Quality tracking (EMA)
        self._quality_ema: float = 0.5
        self._advice_signal: float = 0.0
        self._interaction_count: int = 0

        # Create the wrapped base agent
        base_type_key = self.config.get("base_agent_type", "honest")
        base_config = self.config.get("base_agent_config", {})
        self._base_agent = _create_base_agent(
            agent_id=agent_id,
            type_key=base_type_key,
            roles=roles,
            config=base_config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # Inherit the agent type from the base for observable generation
        self.agent_type = self._base_agent.agent_type

    @property
    def base_agent(self) -> BaseAgent:
        """Return the wrapped base agent for inspection."""
        return self._base_agent

    @property
    def advice_signal(self) -> float:
        """Current advice signal in [-1, +1]."""
        return self._advice_signal

    @property
    def quality_ema(self) -> float:
        """Current quality EMA."""
        return self._quality_ema

    def act(self, observation: Observation) -> Action:
        """Steer the base agent's action via advisor signals.

        The advisor adjusts the observation's ecosystem metrics to
        include advice context, then delegates to the base agent.
        """
        # Inject advisor context into observation metadata
        steered_obs = self._inject_advice(observation)
        return self._base_agent.act(steered_obs)

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Apply advisor steering to the acceptance decision.

        Positive advice → lower the acceptance bar (encourage engagement).
        Negative advice → raise the acceptance bar (inject caution).
        """
        # Compute threshold adjustment from advice signal
        # Positive advice → negative delta (lower threshold = more accepting)
        # Negative advice → positive delta (higher threshold = more cautious)
        threshold_delta = -self._advice_signal * self._threshold_bonus

        # Temporarily adjust the base agent's acceptance threshold if it has one
        original_threshold = getattr(self._base_agent, "acceptance_threshold", None)
        if original_threshold is not None:
            adjusted = max(0.0, min(1.0, original_threshold + threshold_delta))
            self._base_agent.acceptance_threshold = adjusted  # type: ignore[attr-defined]

        try:
            result = self._base_agent.accept_interaction(proposal, observation)
        finally:
            # Restore original threshold
            if original_threshold is not None:
                self._base_agent.acceptance_threshold = original_threshold  # type: ignore[attr-defined]

        return result

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Delegate to base agent with advisor context."""
        steered_obs = self._inject_advice(observation)
        return self._base_agent.propose_interaction(steered_obs, counterparty_id)

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update both the advisor's quality tracker and the base agent."""
        # Update advisor quality tracking
        self._quality_ema = (
            self._quality_ema * (1 - self._advisor_alpha)
            + interaction.p * self._advisor_alpha
        )
        self._interaction_count += 1

        # Recompute advice signal
        import math
        deviation = self._quality_ema - 0.5
        self._advice_signal = math.tanh(
            4.0 * deviation * self._advisor_strength
        )

        # Delegate to base agent
        self._base_agent.update_from_outcome(interaction, payoff)

        # Also update our own history
        super().update_from_outcome(interaction, payoff)

    def _inject_advice(self, observation: Observation) -> Observation:
        """Inject advisor signals into observation metadata.

        This is analogous to the advisor model prepending advice
        instructions to the frontier model's prompt.
        """
        # Modify ecosystem metrics to include advice context
        metrics = dict(observation.ecosystem_metrics)
        metrics["advisor_signal"] = self._advice_signal
        metrics["advisor_quality_ema"] = self._quality_ema

        # Create a shallow copy of the observation with injected metrics
        # (we avoid deepcopy for performance — only ecosystem_metrics is changed)
        steered = Observation(
            agent_state=observation.agent_state,
            current_epoch=observation.current_epoch,
            current_step=observation.current_step,
            can_post=observation.can_post,
            can_interact=observation.can_interact,
            can_vote=observation.can_vote,
            can_claim_task=observation.can_claim_task,
            visible_posts=observation.visible_posts,
            pending_proposals=observation.pending_proposals,
            available_tasks=observation.available_tasks,
            active_tasks=observation.active_tasks,
            recent_interactions=observation.recent_interactions,
            visible_agents=observation.visible_agents,
            available_bounties=observation.available_bounties,
            active_bids=observation.active_bids,
            active_escrows=observation.active_escrows,
            pending_bid_decisions=observation.pending_bid_decisions,
            ecosystem_metrics=metrics,
        )
        return steered

    # ----- Delegation of base agent utilities -----

    def apply_memory_decay(self, epoch: int) -> None:
        """Apply memory decay to both wrapper and base agent."""
        super().apply_memory_decay(epoch)
        self._base_agent.apply_memory_decay(epoch)

    def get_interaction_history(self, limit: int = 50) -> list:
        """Return base agent's interaction history."""
        return self._base_agent.get_interaction_history(limit)

    def compute_counterparty_trust(self, counterparty_id: str) -> float:
        """Delegate trust computation to base agent."""
        return self._base_agent.compute_counterparty_trust(counterparty_id)

    def __repr__(self) -> str:
        return (
            f"AdvisedAgent(id={self.agent_id}, "
            f"base={self._base_agent.__class__.__name__}, "
            f"advice={self._advice_signal:.2f})"
        )


def _create_base_agent(
    agent_id: str,
    type_key: str,
    roles: Optional[List[Role]],
    config: Dict,
    name: Optional[str],
    memory_config: Optional["MemoryConfig"],
    rng,
) -> BaseAgent:
    """Create a base agent by type key.

    Imports the loader's AGENT_TYPES registry lazily to avoid circular imports.
    """
    # Lazy import to avoid circular dependency with scenarios/loader.py
    from swarm.scenarios.loader import AGENT_TYPES

    agent_cls = AGENT_TYPES.get(type_key)
    if agent_cls is None:
        raise ValueError(
            f"Unknown base_agent_type '{type_key}' for AdvisedAgent. "
            f"Available types: {sorted(AGENT_TYPES.keys())}"
        )

    return agent_cls(
        agent_id=agent_id,
        roles=roles,
        config=config,
        name=name,
        memory_config=memory_config,
        rng=rng,
    )

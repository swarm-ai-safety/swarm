"""Moderator agent governance lever."""

import random
from typing import Optional

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class ModeratorLever(GovernanceLever):
    """
    Moderator agent that reviews interactions and penalizes low quality.

    Simulates having a dedicated moderator agent in the ecosystem.
    Unlike random audits (which are sparse and high-penalty), the
    moderator reviews more frequently but with lower penalties,
    providing continuous oversight.

    The moderator also applies a small cost to both parties when
    intervening, representing the friction of moderation.
    """

    def __init__(self, config: GovernanceConfig, seed: Optional[int] = None):
        super().__init__(config)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "moderator"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """
        Review interaction and apply moderator penalties if warranted.

        The moderator reviews interactions with moderator_review_probability.
        If the interaction quality (p) is below the threshold, a penalty
        is applied to the initiator (cost + reputation), and a small
        governance cost is applied to both parties for the review overhead.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Effect with costs and reputation deltas
        """
        if not self.config.moderator_enabled:
            return LeverEffect(lever_name=self.name)

        if not interaction.accepted:
            return LeverEffect(lever_name=self.name)

        # Roll for moderator review
        if self._rng.random() > self.config.moderator_review_probability:
            return LeverEffect(
                lever_name=self.name,
                details={"reviewed": False},
            )

        # Moderator reviews the interaction
        cost_a = 0.0
        cost_b = 0.0
        reputation_deltas = {}
        penalty_applied = False

        # Small review overhead cost to both parties
        review_overhead = 0.01
        cost_a += review_overhead
        cost_b += review_overhead

        if interaction.p < self.config.moderator_threshold_p:
            # Low quality detected - apply penalty
            shortfall = self.config.moderator_threshold_p - interaction.p
            penalty = shortfall * self.config.moderator_penalty_multiplier
            cost_a += penalty
            penalty_applied = True

            # Reputation penalty
            reputation_deltas[interaction.initiator] = -penalty

        return LeverEffect(
            cost_a=cost_a,
            cost_b=cost_b,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "reviewed": True,
                "p": interaction.p,
                "threshold": self.config.moderator_threshold_p,
                "penalty_applied": penalty_applied,
                "penalty_amount": cost_a - review_overhead,
            },
        )

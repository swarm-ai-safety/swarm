"""Transparency ledger governance lever."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class TransparencyLever(GovernanceLever):
    """
    Transparency ledger that makes interaction quality publicly visible.

    When enabled, all interactions are recorded on a public ledger.
    This creates a deterrent effect: agents with low-quality interactions
    receive reputation penalties (their harmful behavior is visible),
    while agents with high-quality interactions receive reputation bonuses.

    Models the governance insight that public accountability reduces
    adverse selection and deceptive behavior.
    """

    @property
    def name(self) -> str:
        return "transparency"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """
        Apply transparency effects based on interaction quality.

        When an interaction is publicly visible on the ledger, agents
        whose interactions have p above the threshold receive a small
        reputation bonus, while those below receive a penalty. This
        creates incentive alignment through public accountability.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Effect with reputation deltas
        """
        if not self.config.transparency_enabled:
            return LeverEffect(lever_name=self.name)

        if not interaction.accepted:
            return LeverEffect(lever_name=self.name)

        reputation_deltas = {}
        rate = self.config.transparency_bonus_rate
        threshold = self.config.transparency_threshold_p

        # Signed deviation from threshold determines reputation effect
        deviation = interaction.p - threshold
        rep_delta = deviation * rate

        reputation_deltas[interaction.initiator] = rep_delta

        return LeverEffect(
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "p": interaction.p,
                "threshold": threshold,
                "rep_delta": rep_delta,
                "initiator": interaction.initiator,
            },
        )

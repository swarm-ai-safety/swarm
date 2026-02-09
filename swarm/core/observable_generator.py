"""Observable signal generation for interactions.

Separates the concern of generating observable signals from the
orchestrator, making it possible to swap in different signal
generation strategies (e.g., for testing, different agent models,
or learned signal profiles).
"""

import random
from typing import Protocol, runtime_checkable

from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState, InteractionProposal
from swarm.models.agent import AgentType


@runtime_checkable
class ObservableGenerator(Protocol):
    """Protocol for generating observable signals from interactions.

    Implementations decide how agent types, reputation, and acceptance
    status translate into raw proxy observables.
    """

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction.

        Args:
            proposal: The interaction proposal being completed.
            accepted: Whether the counterparty accepted.
            state: Current environment state (for agent lookups).

        Returns:
            ProxyObservables with the generated signal values.
        """
        ...  # pragma: no cover


class DefaultObservableGenerator:
    """Default observable generator based on agent type and reputation.

    Observables are based on the initiator's agent type and modulated
    by the initiator's current reputation.  This creates a feedback
    loop: reputation decay degrades observable quality, which lowers
    the soft label p, which in turn reduces payoffs and welfare.
    """

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction."""
        initiator_state = state.get_agent(proposal.initiator_id)
        initiator_type = (
            initiator_state.agent_type if initiator_state else AgentType.HONEST
        )

        task_progress, rework, rejections, misuse, engagement = self._base_signals(
            initiator_type
        )

        # Modulate by initiator reputation: agents with higher reputation
        # produce better observable signals.  Maps reputation from [-1, 1]
        # to a factor in [0.5, 1.0] so the effect is meaningful but bounded.
        rep = initiator_state.reputation if initiator_state else 0.0
        rep_factor = 0.75 + 0.25 * max(min(rep, 1.0), -1.0)  # [0.5, 1.0]
        task_progress *= rep_factor
        engagement *= rep_factor

        # Modify if not accepted
        if not accepted:
            engagement = min(engagement, 0.0)

        return ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )

    @staticmethod
    def _base_signals(
        agent_type: AgentType,
    ) -> tuple[float, int, int, int, float]:
        """Return (task_progress, rework, rejections, misuse, engagement)."""
        if agent_type == AgentType.HONEST:
            return (
                random.uniform(0.3, 0.8),
                random.randint(0, 1),
                0,
                0,
                random.uniform(0.2, 0.6),
            )
        elif agent_type == AgentType.OPPORTUNISTIC:
            return (
                random.uniform(0.1, 0.5),
                random.randint(0, 2),
                random.randint(0, 1),
                0,
                random.uniform(-0.2, 0.3),
            )
        elif agent_type == AgentType.DECEPTIVE:
            return (
                random.uniform(0.2, 0.6),
                random.randint(0, 2),
                random.randint(0, 1),
                0,
                random.uniform(0.0, 0.4),
            )
        else:  # Adversarial
            return (
                random.uniform(-0.3, 0.2),
                random.randint(1, 3),
                random.randint(1, 2),
                random.randint(0, 1),
                random.uniform(-0.5, -0.1),
            )

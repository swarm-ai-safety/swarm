"""Handler for core interaction actions (PROPOSE, ACCEPT, REJECT).

Extracted from ``Orchestrator._handle_core_action`` to follow the
handler plugin pattern.  ACCEPT and REJECT delegate to the
``InteractionFinalizer`` for proxy computation, governance, and payoffs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Optional

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.env.state import InteractionProposal
from swarm.models.events import interaction_proposed_event

if TYPE_CHECKING:
    from swarm.core.interaction_finalizer import InteractionFinalizer
    from swarm.env.network import AgentNetwork
    from swarm.env.state import EnvState
    from swarm.logging.event_bus import EventBus


class CoreInteractionHandler(Handler):
    """Handles PROPOSE_INTERACTION, ACCEPT_INTERACTION, REJECT_INTERACTION."""

    def __init__(
        self,
        *,
        finalizer: InteractionFinalizer,
        network: Optional[AgentNetwork] = None,
        event_bus: EventBus,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._finalizer = finalizer
        self._network = network

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({
            ActionType.PROPOSE_INTERACTION,
            ActionType.ACCEPT_INTERACTION,
            ActionType.REJECT_INTERACTION,
        })

    def handle_action(self, action: Action, state: EnvState) -> HandlerActionResult:
        agent_id = action.agent_id

        if action.action_type == ActionType.PROPOSE_INTERACTION:
            return self._handle_propose(action, state, agent_id)
        elif action.action_type == ActionType.ACCEPT_INTERACTION:
            return self._handle_accept_reject(action, state, accepted=True)
        elif action.action_type == ActionType.REJECT_INTERACTION:
            return self._handle_accept_reject(action, state, accepted=False)

        return HandlerActionResult(success=False)

    def _handle_propose(
        self, action: Action, state: EnvState, agent_id: str
    ) -> HandlerActionResult:
        rate_limit = state.get_rate_limit_state(agent_id)
        if not rate_limit.can_interact(state.rate_limits):
            return HandlerActionResult(success=False)

        if self._network is not None:
            if not self._network.has_edge(agent_id, action.counterparty_id):
                return HandlerActionResult(success=False)

        proposal = InteractionProposal(
            initiator_id=agent_id,
            counterparty_id=action.counterparty_id,
            interaction_type=action.interaction_type.value,
            content=action.content,
            metadata=action.metadata,
        )
        state.add_proposal(proposal)
        rate_limit.record_interaction()

        self._emit_event(
            interaction_proposed_event(
                interaction_id=proposal.proposal_id,
                initiator_id=agent_id,
                counterparty_id=action.counterparty_id,
                interaction_type=action.interaction_type.value,
                v_hat=0.0,
                p=0.5,
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return HandlerActionResult(success=True)

    def _handle_accept_reject(
        self, action: Action, state: EnvState, *, accepted: bool
    ) -> HandlerActionResult:
        proposal: Optional[InteractionProposal] = state.remove_proposal(
            action.target_id
        )
        if proposal is None:
            return HandlerActionResult(success=False)

        self._finalizer.complete_interaction(proposal, accepted=accepted)
        return HandlerActionResult(success=True)

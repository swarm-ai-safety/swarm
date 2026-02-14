"""Handler for feed actions (POST, REPLY, VOTE).

Extracted from ``Orchestrator._handle_core_action`` to follow the
handler plugin pattern used by MarketplaceHandler, KernelOracleHandler, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.env.feed import Feed, VoteType

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.logging.event_bus import EventBus


class FeedHandler(Handler):
    """Handles POST, REPLY, and VOTE actions against the shared feed."""

    def __init__(
        self,
        *,
        feed: Feed,
        max_content_length: int = 10000,
        event_bus: EventBus,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._feed = feed
        self._max_content_length = max_content_length

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.POST, ActionType.REPLY, ActionType.VOTE})

    def handle_action(self, action: Action, state: EnvState) -> HandlerActionResult:
        agent_id = action.agent_id
        rate_limit = state.get_rate_limit_state(agent_id)

        if action.action_type == ActionType.POST:
            if not rate_limit.can_post(state.rate_limits):
                return HandlerActionResult(success=False)
            try:
                self._feed.create_post(
                    author_id=agent_id,
                    content=action.content[: self._max_content_length],
                )
                rate_limit.record_post()
                return HandlerActionResult(success=True)
            except ValueError:
                return HandlerActionResult(success=False)

        elif action.action_type == ActionType.REPLY:
            if not rate_limit.can_post(state.rate_limits):
                return HandlerActionResult(success=False)
            try:
                self._feed.create_post(
                    author_id=agent_id,
                    content=action.content[: self._max_content_length],
                    parent_id=action.target_id,
                )
                rate_limit.record_post()
                return HandlerActionResult(success=True)
            except ValueError:
                return HandlerActionResult(success=False)

        elif action.action_type == ActionType.VOTE:
            if not rate_limit.can_vote(state.rate_limits):
                return HandlerActionResult(success=False)
            vote_type = (
                VoteType.UPVOTE if action.vote_direction > 0 else VoteType.DOWNVOTE
            )
            vote = self._feed.vote(action.target_id, agent_id, vote_type)
            if vote:
                rate_limit.record_vote()
                return HandlerActionResult(success=True)
            return HandlerActionResult(success=False)

        return HandlerActionResult(success=False)

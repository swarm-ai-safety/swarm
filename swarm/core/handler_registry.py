"""Registry mapping ActionTypes to Handler instances.

The ``HandlerRegistry`` is the central dispatch table for the
orchestrator's plugin architecture.  Handlers register which
``ActionType`` values they own, and the orchestrator routes actions
through a single lookup instead of a long if-elif chain.

Invariants:
- Each ``ActionType`` is owned by at most one handler.
- Registration order determines lifecycle hook iteration order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from swarm.core.handler import Handler


class HandlerRegistry:
    """Registry mapping ActionTypes to Handler instances."""

    def __init__(self) -> None:
        self._action_map: Dict = {}  # ActionType -> Handler
        self._handlers: List[Handler] = []  # preserves registration order
        self._handler_ids: set = set()  # for dedup

    def register(self, handler: Handler) -> None:
        """Register a handler and claim its action types.

        Raises ``ValueError`` if any action type is already claimed by
        another handler.
        """
        action_types = handler.handled_action_types()

        # Check for conflicts
        for at in action_types:
            existing = self._action_map.get(at)
            if existing is not None and existing is not handler:
                raise ValueError(
                    f"ActionType {at!r} is already claimed by "
                    f"{type(existing).__name__}, cannot register "
                    f"{type(handler).__name__}"
                )

        # Register
        for at in action_types:
            self._action_map[at] = handler

        if id(handler) not in self._handler_ids:
            self._handlers.append(handler)
            self._handler_ids.add(id(handler))

    def get_handler(self, action_type: object) -> Optional[Handler]:
        """Look up the handler for an action type.

        Returns ``None`` if no handler is registered for this action type.
        """
        return self._action_map.get(action_type)

    def all_handlers(self) -> Sequence[Handler]:
        """All registered handlers in registration order."""
        return self._handlers

    def has_handler(self, action_type: object) -> bool:
        """Check if an action type has a registered handler."""
        return action_type in self._action_map

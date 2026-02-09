"""Base class for environment handlers in the orchestration layer.

All handlers share a common pattern:
- Accept an ``emit_event`` callback for append-only event logging
- Dispatch agent actions via ``handle_action``
- Build per-agent observation fields via ``build_observation_fields``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from swarm.models.events import Event


class Handler(ABC):
    """Abstract base for environment handlers.

    Subclasses must implement ``handle_action``.  Override
    ``build_observation_fields`` if the handler contributes to agent
    observations each step.
    """

    def __init__(self, *, emit_event: Callable[[Event], None]) -> None:
        self._emit_event = emit_event

    @abstractmethod
    def handle_action(self, action: Any, state: Any) -> Any:
        """Dispatch a single agent action and return a result dataclass."""

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Return handler-specific observation fields for *agent_id*.

        The default implementation returns an empty dict.  Handlers that
        contribute to agent observations should override this.
        """
        return {}

    def on_epoch_start(self, state: Any) -> None:  # noqa: B027
        """Hook called at the start of each epoch.  Override if needed."""

    def on_epoch_end(self, state: Any) -> None:  # noqa: B027
        """Hook called at the end of each epoch.  Override if needed."""

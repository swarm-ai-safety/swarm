"""Thin synchronous event bus for decoupling event producers from consumers.

The bus sits between handlers (which call ``emit(event)``) and any number
of subscribers (e.g. the ``EventLog`` appender, future metric streamers).

Design constraints:
- Synchronous dispatch (no async, no threads).
- Enrichment (seed / scenario_id / replay_k) applied once at emit time.
- Subscribers are plain ``Callable[[Event], None]``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from swarm.models.events import Event


class EventBus:
    """Synchronous publish-subscribe event bus.

    Usage::

        bus = EventBus()
        bus.set_enrichment(seed=42, scenario_id="baseline")
        bus.subscribe(event_log.append)
        bus.emit(event)   # enriches then dispatches to all subscribers
    """

    def __init__(self) -> None:
        self._subscribers: List[Callable[[Event], None]] = []
        self._enrichment: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_enrichment(
        self,
        *,
        seed: Optional[int] = None,
        scenario_id: Optional[str] = None,
        replay_k: Optional[int] = None,
    ) -> None:
        """Set enrichment fields applied to every emitted event.

        Only non-``None`` values are stored.  Fields already present on
        the event are *not* overwritten (event-level values win).
        """
        if seed is not None:
            self._enrichment["seed"] = seed
        if scenario_id is not None:
            self._enrichment["scenario_id"] = scenario_id
        if replay_k is not None:
            self._enrichment["replay_k"] = replay_k

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Register a subscriber.  Duplicates are silently ignored."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Remove a subscriber.  No-op if not registered."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: Event) -> None:
        """Enrich *event* in-place and dispatch to all subscribers."""
        # Apply enrichment (event-level values take priority)
        if event.seed is None and "seed" in self._enrichment:
            event.seed = self._enrichment["seed"]  # type: ignore[assignment]
        if event.scenario_id is None and "scenario_id" in self._enrichment:
            event.scenario_id = self._enrichment["scenario_id"]  # type: ignore[assignment]
        if event.replay_k is None and "replay_k" in self._enrichment:
            event.replay_k = self._enrichment["replay_k"]  # type: ignore[assignment]

        for subscriber in self._subscribers:
            subscriber(event)

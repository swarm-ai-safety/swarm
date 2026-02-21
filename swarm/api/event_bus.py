"""In-memory event bus for simulation events."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_SUBSCRIPTIONS_PER_AGENT = 2


class SimEventType(str, Enum):
    OBSERVATION_READY = "observation_ready"
    STEP_COMPLETE = "step_complete"
    EPOCH_COMPLETE = "epoch_complete"
    SIMULATION_COMPLETE = "simulation_complete"
    ACTION_TIMEOUT = "action_timeout"


@dataclass
class SimEvent:
    event_type: SimEventType
    simulation_id: str
    data: dict[str, Any] = field(default_factory=dict)
    agent_id: str | None = None  # None means broadcast to all


class EventBus:
    """In-memory pub/sub event bus for simulation events."""

    def __init__(self) -> None:
        # simulation_id -> list of (agent_id, asyncio.Queue)
        self._subscribers: dict[str, list[tuple[str, asyncio.Queue]]] = {}

    def subscribe(self, simulation_id: str, agent_id: str) -> asyncio.Queue:
        """Subscribe to events for a simulation. Returns a Queue to read from.

        Raises ValueError if the agent already has
        ``MAX_SUBSCRIPTIONS_PER_AGENT`` active subscriptions for this
        simulation.
        """
        if simulation_id not in self._subscribers:
            self._subscribers[simulation_id] = []

        existing = sum(
            1 for aid, _ in self._subscribers[simulation_id] if aid == agent_id
        )
        if existing >= MAX_SUBSCRIPTIONS_PER_AGENT:
            raise ValueError(
                f"Agent {agent_id} already has {existing} subscription(s) "
                f"for simulation {simulation_id}"
            )

        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers[simulation_id].append((agent_id, q))
        return q

    def unsubscribe(
        self, simulation_id: str, agent_id: str, queue: asyncio.Queue
    ) -> None:
        """Remove a subscription."""
        subs = self._subscribers.get(simulation_id, [])
        self._subscribers[simulation_id] = [
            (aid, q) for aid, q in subs if not (aid == agent_id and q is queue)
        ]

    async def publish(self, event: SimEvent) -> int:
        """Publish an event. Returns number of subscribers notified."""
        subs = self._subscribers.get(event.simulation_id, [])
        count = 0
        for agent_id, queue in subs:
            # Send if broadcast (agent_id is None) or targeted at this agent
            if event.agent_id is None or event.agent_id == agent_id:
                try:
                    queue.put_nowait(event)
                    count += 1
                except asyncio.QueueFull:
                    pass  # Drop events for slow consumers
        return count

    def subscriber_count(self, simulation_id: str) -> int:
        """Number of active subscribers for a simulation."""
        return len(self._subscribers.get(simulation_id, []))


# Global singleton
event_bus = EventBus()

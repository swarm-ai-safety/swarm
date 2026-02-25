"""Inter-agent message bus for the OpenSandbox bridge.

All agent-to-agent communication routes through this bus.  Direct
sandbox-to-sandbox communication is prohibited — the bus acts as
a chokepoint for enforcing interaction protocols, logging
provenance, and preventing covert channels.
"""

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from swarm.bridges.opensandbox.events import OpenSandboxEvent, OpenSandboxEventType

logger = logging.getLogger(__name__)


@dataclass
class BusMessage:
    """A message routed through the inter-agent message bus.

    Attributes:
        message_id: Unique message identifier.
        from_sandbox: Source sandbox ID.
        to_sandbox: Destination sandbox ID.
        from_agent: Source agent ID.
        to_agent: Destination agent ID.
        payload: Message content (opaque to the bus).
        provenance_id: Byline provenance signature ID.
        timestamp: When the message was submitted.
        delivered: Whether the message has been delivered.
        blocked: Whether the message was blocked by policy.
        block_reason: Reason for blocking (if blocked).
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_sandbox: str = ""
    to_sandbox: str = ""
    from_agent: str = ""
    to_agent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    provenance_id: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    delivered: bool = False
    blocked: bool = False
    block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "message_id": self.message_id,
            "from_sandbox": self.from_sandbox,
            "to_sandbox": self.to_sandbox,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "provenance_id": self.provenance_id,
            "timestamp": self.timestamp.isoformat(),
            "delivered": self.delivered,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "payload_keys": list(self.payload.keys()),
        }


# Type alias for message validators
MessageValidator = Callable[[BusMessage], Optional[str]]


class MessageBus:
    """Mediated message bus for inter-sandbox communication.

    Design properties:

    - No direct agent-to-agent channels exist.
    - Every message is validated against the interaction protocol
      before delivery.
    - Provenance IDs are required when provenance is enabled.
    - Message history is retained for observability and replay.

    Example::

        bus = MessageBus(max_pending=1000)
        bus.register_sandbox("sandbox-a", "agent-a")
        bus.register_sandbox("sandbox-b", "agent-b")
        msg = bus.send(
            from_sandbox="sandbox-a",
            to_sandbox="sandbox-b",
            payload={"action": "collaborate"},
            provenance_id="prov-123",
        )
        pending = bus.receive("sandbox-b")
    """

    def __init__(
        self,
        max_pending: int = 10_000,
        require_provenance: bool = True,
    ) -> None:
        self._max_pending = max_pending
        self._require_provenance = require_provenance

        # sandbox_id -> agent_id mapping
        self._sandbox_agents: Dict[str, str] = {}
        # sandbox_id -> deque of pending messages
        self._mailboxes: Dict[str, Deque[BusMessage]] = {}
        # Full message history (for audit / replay)
        self._history: List[BusMessage] = []
        # Registered validators
        self._validators: List[MessageValidator] = []
        # Events
        self._events: List[OpenSandboxEvent] = []
        # Allowed connections (if empty, all registered pairs OK)
        self._allowed_routes: Optional[Set[tuple]] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_sandbox(self, sandbox_id: str, agent_id: str) -> None:
        """Register a sandbox/agent pair with the bus."""
        self._sandbox_agents[sandbox_id] = agent_id
        self._mailboxes.setdefault(sandbox_id, deque())
        logger.debug("Message bus registered sandbox %s (agent %s)", sandbox_id, agent_id)

    def unregister_sandbox(self, sandbox_id: str) -> None:
        """Remove a sandbox from the bus."""
        self._sandbox_agents.pop(sandbox_id, None)
        self._mailboxes.pop(sandbox_id, None)

    def set_allowed_routes(self, routes: Set[tuple]) -> None:
        """Restrict bus to specific (from_sandbox, to_sandbox) routes."""
        self._allowed_routes = routes

    def add_validator(self, validator: MessageValidator) -> None:
        """Add a message validator.  Return a reason string to block."""
        self._validators.append(validator)

    # ------------------------------------------------------------------
    # Send / Receive
    # ------------------------------------------------------------------

    def send(
        self,
        from_sandbox: str,
        to_sandbox: str,
        payload: Dict[str, Any],
        provenance_id: Optional[str] = None,
    ) -> BusMessage:
        """Send a message between sandboxes.

        Args:
            from_sandbox: Source sandbox ID.
            to_sandbox: Destination sandbox ID.
            payload: Message content.
            provenance_id: Byline provenance signature.

        Returns:
            The BusMessage (with blocked/delivered status set).
        """
        from_agent = self._sandbox_agents.get(from_sandbox, "")
        to_agent = self._sandbox_agents.get(to_sandbox, "")

        msg = BusMessage(
            from_sandbox=from_sandbox,
            to_sandbox=to_sandbox,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            provenance_id=provenance_id,
        )

        # Validate
        block_reason = self._validate(msg)
        if block_reason:
            msg.blocked = True
            msg.block_reason = block_reason
            self._history.append(msg)
            self._record_event(
                OpenSandboxEventType.MESSAGE_BLOCKED,
                from_agent,
                sandbox_id=from_sandbox,
                payload={"reason": block_reason, "to_sandbox": to_sandbox},
            )
            logger.info(
                "Message blocked %s -> %s: %s",
                from_sandbox, to_sandbox, block_reason,
            )
            return msg

        # Deliver
        mailbox = self._mailboxes.get(to_sandbox)
        if mailbox is None:
            msg.blocked = True
            msg.block_reason = f"Destination sandbox {to_sandbox} not registered"
            self._history.append(msg)
            return msg

        if len(mailbox) >= self._max_pending:
            msg.blocked = True
            msg.block_reason = "Mailbox full"
            self._history.append(msg)
            return msg

        msg.delivered = True
        mailbox.append(msg)
        self._history.append(msg)

        self._record_event(
            OpenSandboxEventType.MESSAGE_SENT,
            from_agent,
            sandbox_id=from_sandbox,
            payload={
                "message_id": msg.message_id,
                "to_sandbox": to_sandbox,
                "provenance_id": provenance_id,
            },
        )
        return msg

    def receive(self, sandbox_id: str, max_messages: int = 100) -> List[BusMessage]:
        """Retrieve pending messages for a sandbox.

        Args:
            sandbox_id: The sandbox to poll.
            max_messages: Maximum messages to return.

        Returns:
            List of pending BusMessage objects.
        """
        mailbox = self._mailboxes.get(sandbox_id)
        if not mailbox:
            return []

        messages: List[BusMessage] = []
        while mailbox and len(messages) < max_messages:
            msg = mailbox.popleft()
            messages.append(msg)
            self._record_event(
                OpenSandboxEventType.MESSAGE_DELIVERED,
                self._sandbox_agents.get(sandbox_id, ""),
                sandbox_id=sandbox_id,
                payload={"message_id": msg.message_id, "from_sandbox": msg.from_sandbox},
            )

        return messages

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> List[BusMessage]:
        """Return the full message history for audit / replay."""
        return list(self._history)

    def get_events(self) -> List[OpenSandboxEvent]:
        """Return all bus events."""
        return list(self._events)

    def get_stats(self) -> Dict[str, Any]:
        """Return bus statistics."""
        total = len(self._history)
        blocked = sum(1 for m in self._history if m.blocked)
        delivered = sum(1 for m in self._history if m.delivered)
        return {
            "total_messages": total,
            "delivered": delivered,
            "blocked": blocked,
            "registered_sandboxes": len(self._sandbox_agents),
            "pending_total": sum(len(q) for q in self._mailboxes.values()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, msg: BusMessage) -> Optional[str]:
        """Run all validators.  Return reason string if blocked."""
        # Source must be registered
        if msg.from_sandbox not in self._sandbox_agents:
            return f"Source sandbox {msg.from_sandbox} not registered"

        # Provenance required
        if self._require_provenance and not msg.provenance_id:
            return "Provenance ID required"

        # Route restriction
        if self._allowed_routes is not None:
            route = (msg.from_sandbox, msg.to_sandbox)
            if route not in self._allowed_routes:
                return f"Route {route} not in allowed routes"

        # Custom validators
        for validator in self._validators:
            reason = validator(msg)
            if reason:
                return reason

        return None

    def _record_event(
        self,
        event_type: OpenSandboxEventType,
        agent_id: str,
        sandbox_id: Optional[str] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        self._events.append(
            OpenSandboxEvent(
                event_type=event_type,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                payload=payload or {},
            )
        )

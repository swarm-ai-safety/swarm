"""Event schemas for the SWARM-OpenSandbox bridge.

Defines typed event structures for sandbox lifecycle, contract
enforcement, message bus routing, governance interventions, and
provenance tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class OpenSandboxEventType(Enum):
    """Event types in the OpenSandbox bridge protocol."""

    # Sandbox lifecycle
    SANDBOX_CREATED = "sandbox:created"
    SANDBOX_STARTED = "sandbox:started"
    SANDBOX_STOPPED = "sandbox:stopped"
    SANDBOX_KILLED = "sandbox:killed"
    SANDBOX_SNAPSHOT = "sandbox:snapshot"

    # Contract lifecycle
    CONTRACT_PUBLISHED = "contract:published"
    CONTRACT_ASSIGNED = "contract:assigned"
    CONTRACT_REJECTED = "contract:rejected"

    # Screening
    AGENT_SCREENED = "screening:evaluated"
    AGENT_ADMITTED = "screening:admitted"
    AGENT_DENIED = "screening:denied"

    # Command execution
    COMMAND_EXECUTED = "command:executed"
    COMMAND_DENIED = "command:denied"

    # Message bus
    MESSAGE_SENT = "message:sent"
    MESSAGE_DELIVERED = "message:delivered"
    MESSAGE_BLOCKED = "message:blocked"

    # Governance interventions
    INTERVENTION_ISOLATE = "governance:isolate"
    INTERVENTION_RESTRICT = "governance:restrict"
    INTERVENTION_TERMINATE = "governance:terminate"
    CONTRACT_VIOLATION = "governance:contract_violation"

    # Provenance
    PROVENANCE_SIGNED = "provenance:signed"

    # Observability
    RISK_ALERT = "observability:risk_alert"
    METRICS_COMPUTED = "observability:metrics_computed"


@dataclass
class OpenSandboxEvent:
    """An event observed in the OpenSandbox bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: OpenSandboxEventType = OpenSandboxEventType.SANDBOX_CREATED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    sandbox_id: Optional[str] = None
    contract_id: Optional[str] = None
    provenance_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON transport."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "sandbox_id": self.sandbox_id,
            "contract_id": self.contract_id,
            "provenance_id": self.provenance_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSandboxEvent":
        """Deserialize from dict with safe type handling."""
        try:
            event_type = OpenSandboxEventType(data["event_type"])
        except (ValueError, KeyError):
            event_type = OpenSandboxEventType.CONTRACT_VIOLATION
        raw_ts = data.get("timestamp")
        if raw_ts is not None:
            try:
                ts = datetime.fromisoformat(str(raw_ts))
                # Normalize naive timestamps to UTC to avoid mixing with
                # timezone-aware datetimes from _utcnow().
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ts = _utcnow()
        else:
            ts = _utcnow()
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=ts,
            agent_id=str(data.get("agent_id", "")),
            sandbox_id=data.get("sandbox_id"),
            contract_id=data.get("contract_id"),
            provenance_id=data.get("provenance_id"),
            payload=data.get("payload", {}),
        )

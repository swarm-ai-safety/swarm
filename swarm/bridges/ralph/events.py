"""Event schemas for the SWARM-Ralph bridge."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _first_str(data: Dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return str(value)
    return default


class RalphEventType(Enum):
    """Supported Ralph event categories."""

    TASK_STARTED = "task:started"
    TASK_COMPLETED = "task:completed"
    TASK_FAILED = "task:failed"
    REVIEW_REQUESTED = "review:requested"
    REVIEW_REJECTED = "review:rejected"
    TOOL_MISUSE = "tool:misuse"
    GENERIC = "generic"


@dataclass
class RalphEvent:
    """Normalized event shape consumed by the bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: RalphEventType = RalphEventType.GENERIC
    timestamp: datetime = field(default_factory=_utcnow)
    actor_id: str = ""
    task_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RalphEvent":
        raw_type = _first_str(data, ("event_type", "type"), "generic")
        try:
            event_type = RalphEventType(raw_type)
        except ValueError:
            event_type = RalphEventType.GENERIC

        raw_ts = data.get("timestamp", data.get("time"))
        try:
            timestamp = datetime.fromisoformat(str(raw_ts)) if raw_ts else _utcnow()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            timestamp = _utcnow()

        payload = data.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        return cls(
            event_id=_first_str(data, ("event_id", "id"), str(uuid.uuid4())),
            event_type=event_type,
            timestamp=timestamp,
            actor_id=_first_str(data, ("actor_id", "actor", "agent_id", "agent")),
            task_id=_first_str(data, ("task_id", "task", "job_id", "job")),
            payload=payload,
        )

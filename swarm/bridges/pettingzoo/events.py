"""Event schemas for the PettingZoo bridge."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PZEventType(Enum):
    """Event types in the PettingZoo bridge."""

    EPISODE_RESET = "episode_reset"
    STEP_COMPLETED = "step_completed"
    INTERACTION_RECORDED = "interaction_recorded"
    AGENT_TERMINATED = "agent_terminated"
    EPISODE_DONE = "episode_done"
    ERROR = "error"


@dataclass
class PZEvent:
    """An event in the PettingZoo bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: PZEventType = PZEventType.STEP_COMPLETED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
        }

"""Event schemas for the Concordia bridge."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConcordiaEventType(Enum):
    """Event types in the Concordia bridge."""

    NARRATIVE_CAPTURED = "narrative_captured"
    JUDGE_EVALUATED = "judge_evaluated"
    GOVERNANCE_APPLIED = "governance_applied"
    AGENT_FROZEN = "agent_frozen"
    STEP_COMPLETED = "step_completed"
    ERROR = "error"

    # Social Simulacra events
    PERSONA_EXPANDED = "persona_expanded"
    THREAD_GENERATED = "thread_generated"
    WHATIF_INJECTED = "whatif_injected"
    MULTIVERSE_UNIVERSE_COMPLETED = "multiverse_universe_completed"
    MULTIVERSE_ANALYSIS_COMPLETED = "multiverse_analysis_completed"


@dataclass
class ConcordiaEvent:
    """An event in the Concordia bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ConcordiaEventType = ConcordiaEventType.STEP_COMPLETED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConcordiaEvent":
        try:
            event_type = ConcordiaEventType(data["event_type"])
        except (ValueError, KeyError):
            event_type = ConcordiaEventType.ERROR

        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            try:
                timestamp = datetime.fromisoformat(raw_ts)
            except ValueError:
                timestamp = _utcnow()
        else:
            timestamp = _utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=timestamp,
            agent_id=str(data.get("agent_id", "")),
            payload=data.get("payload", {}),
        )


@dataclass
class NarrativeChunk:
    """A chunk of narrative text from Concordia."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_ids: list[str] = field(default_factory=list)
    narrative_text: str = ""
    step_range: tuple[int, int] = (0, 0)
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class JudgeScores:
    """Scores from the LLM judge evaluation."""

    progress: float = 0.5
    quality: float = 0.5
    cooperation: float = 0.5
    harm: float = 0.0
    raw_response: str = ""
    cached: bool = False

    def __post_init__(self) -> None:
        """Clamp all scores to [0, 1]."""
        self.progress = max(0.0, min(1.0, self.progress))
        self.quality = max(0.0, min(1.0, self.quality))
        self.cooperation = max(0.0, min(1.0, self.cooperation))
        self.harm = max(0.0, min(1.0, self.harm))

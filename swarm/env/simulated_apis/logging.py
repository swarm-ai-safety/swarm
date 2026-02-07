"""JSONL logging utilities for simulated API episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(payload: Dict[str, Any]) -> str:
    # Stable hashing for provenance: sorted keys, no whitespace.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_event_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


@dataclass
class SimApiEvent:
    """A single episode event with a stable hash-based id."""

    event_type: str
    timestamp: str = field(default_factory=_utc_now_iso)
    agent_id: Optional[str] = None
    parent_event_hash: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    event_hash: str = ""

    def finalize(self) -> "SimApiEvent":
        if self.event_hash:
            return self
        base = {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "parent_event_hash": self.parent_event_hash,
            "payload": self.payload,
        }
        self.event_hash = compute_event_hash(base)
        return self

    @property
    def provenance_id(self) -> str:
        # Short, stable handle suitable for citations in agent justifications.
        if not self.event_hash:
            return ""
        return self.event_hash[:12]

    def to_dict(self) -> Dict[str, Any]:
        self.finalize()
        return {
            "event_hash": self.event_hash,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "parent_event_hash": self.parent_event_hash,
            "provenance_id": self.provenance_id,
            "payload": self.payload,
        }


class SimApiEpisodeLog:
    """In-memory episode log with optional JSONL persistence."""

    def __init__(self) -> None:
        self._events: List[SimApiEvent] = []

    def append(self, event: SimApiEvent) -> SimApiEvent:
        event.finalize()
        self._events.append(event)
        return event

    def extend(self, events: Iterable[SimApiEvent]) -> None:
        for e in events:
            self.append(e)

    def events(self) -> List[SimApiEvent]:
        return list(self._events)

    def last_event_hash(self) -> Optional[str]:
        if not self._events:
            return None
        return self._events[-1].event_hash

    def to_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in self._events:
                f.write(_stable_json_dumps(e.to_dict()) + "\n")


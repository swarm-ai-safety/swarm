"""Society-native trace format.

Traces are lists of step records with joint actions, governance
intervention diffs, events, and state summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceSpan:
    """One step of an episode trace."""

    t: int
    actions: List[Dict[str, Any]] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics_step: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "actions": self.actions,
            "governance": self.governance,
            "events": self.events,
            "metrics_step": self.metrics_step,
        }


@dataclass
class Trace:
    """Full episode trace."""

    episode_id: str = ""
    spans: List[TraceSpan] = field(default_factory=list)

    def add_span(self, span: TraceSpan) -> None:
        self.spans.append(span)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "spans": [s.to_dict() for s in self.spans],
        }

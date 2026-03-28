"""Append-only per-agent performance tracker.

Records heartbeat-level metrics (task completion rate, time-to-close,
review pass rate, blocker frequency) to a JSONL log.  Designed for
integration into Paperclip heartbeat procedures so agents can track
their own performance over time.

Reference: Zhang et al., Hyperagents (arXiv:2603.19461) — agents that
track their own metrics enable trend detection and regression avoidance.
"""

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class TrackerEventType:
    """Event types for the performance tracker."""

    TASK_COMPLETED = "task_completed"
    TASK_BLOCKED = "task_blocked"
    TASK_STARTED = "task_started"
    REVIEW_PASSED = "review_passed"
    REVIEW_FAILED = "review_failed"
    HEARTBEAT = "heartbeat"


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------

@dataclass
class PerformanceEvent:
    """A single performance event recorded by an agent."""

    agent_id: str
    event_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceEvent":
        return cls(
            agent_id=data["agent_id"],
            event_type=data["event_type"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSummary:
    """Aggregated performance summary for an agent."""

    agent_id: str
    total_tasks_started: int = 0
    total_tasks_completed: int = 0
    total_tasks_blocked: int = 0
    total_reviews_passed: int = 0
    total_reviews_failed: int = 0
    total_heartbeats: int = 0

    # Derived rates
    completion_rate: float = 0.0
    review_pass_rate: float = 0.0
    blocker_frequency: float = 0.0  # blocked / started
    avg_time_to_close_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Append-only JSONL performance tracker for a single agent.

    Usage::

        tracker = PerformanceTracker("agent-123", Path("logs/perf.jsonl"))
        tracker.record_task_started("SWA-51")
        tracker.record_task_completed("SWA-51")
        tracker.record_heartbeat()
        summary = tracker.summarize()
    """

    def __init__(self, agent_id: str, log_path: Path) -> None:
        self.agent_id = agent_id
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._ensure_parent()

    # -- Recording helpers ---------------------------------------------------

    def record_task_started(
        self, task_id: str, **extra: Any,
    ) -> PerformanceEvent:
        return self._append(
            TrackerEventType.TASK_STARTED,
            {"task_id": task_id, **extra},
        )

    def record_task_completed(
        self, task_id: str, **extra: Any,
    ) -> PerformanceEvent:
        return self._append(
            TrackerEventType.TASK_COMPLETED,
            {"task_id": task_id, **extra},
        )

    def record_task_blocked(
        self, task_id: str, reason: str = "", **extra: Any,
    ) -> PerformanceEvent:
        return self._append(
            TrackerEventType.TASK_BLOCKED,
            {"task_id": task_id, "reason": reason, **extra},
        )

    def record_review_passed(
        self, task_id: str, **extra: Any,
    ) -> PerformanceEvent:
        return self._append(
            TrackerEventType.REVIEW_PASSED,
            {"task_id": task_id, **extra},
        )

    def record_review_failed(
        self, task_id: str, reason: str = "", **extra: Any,
    ) -> PerformanceEvent:
        return self._append(
            TrackerEventType.REVIEW_FAILED,
            {"task_id": task_id, "reason": reason, **extra},
        )

    def record_heartbeat(self, **extra: Any) -> PerformanceEvent:
        return self._append(TrackerEventType.HEARTBEAT, extra)

    # -- Replay / read -------------------------------------------------------

    def replay(self) -> Iterator[PerformanceEvent]:
        """Yield all events from the log file, skipping corrupted lines."""
        if not self.log_path.exists():
            return
        with open(self.log_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield PerformanceEvent.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "%s:%d: skipping malformed line: %s", self.log_path, lineno, exc,
                    )

    def events_for_agent(
        self, agent_id: Optional[str] = None,
    ) -> List[PerformanceEvent]:
        """Return events filtered to a specific agent (default: self)."""
        target = agent_id or self.agent_id
        return [e for e in self.replay() if e.agent_id == target]

    # -- Summarize -----------------------------------------------------------

    def summarize(
        self, agent_id: Optional[str] = None,
    ) -> PerformanceSummary:
        """Compute a summary from the log for the given agent."""
        target = agent_id or self.agent_id
        events = self.events_for_agent(target)

        summary = PerformanceSummary(agent_id=target)

        # Track start times for time-to-close calculation
        start_times: Dict[str, str] = {}  # task_id -> iso timestamp
        close_durations: List[float] = []

        for ev in events:
            if ev.event_type == TrackerEventType.TASK_STARTED:
                summary.total_tasks_started += 1
                tid = ev.metadata.get("task_id", "")
                if tid:
                    start_times[tid] = ev.timestamp
            elif ev.event_type == TrackerEventType.TASK_COMPLETED:
                summary.total_tasks_completed += 1
                tid = ev.metadata.get("task_id", "")
                if tid and tid in start_times:
                    start_dt = datetime.fromisoformat(start_times[tid])
                    end_dt = datetime.fromisoformat(ev.timestamp)
                    close_durations.append(
                        (end_dt - start_dt).total_seconds(),
                    )
            elif ev.event_type == TrackerEventType.TASK_BLOCKED:
                summary.total_tasks_blocked += 1
            elif ev.event_type == TrackerEventType.REVIEW_PASSED:
                summary.total_reviews_passed += 1
            elif ev.event_type == TrackerEventType.REVIEW_FAILED:
                summary.total_reviews_failed += 1
            elif ev.event_type == TrackerEventType.HEARTBEAT:
                summary.total_heartbeats += 1

        # Derived rates
        if summary.total_tasks_started > 0:
            summary.completion_rate = (
                summary.total_tasks_completed / summary.total_tasks_started
            )
            summary.blocker_frequency = (
                summary.total_tasks_blocked / summary.total_tasks_started
            )

        total_reviews = summary.total_reviews_passed + summary.total_reviews_failed
        if total_reviews > 0:
            summary.review_pass_rate = (
                summary.total_reviews_passed / total_reviews
            )

        if close_durations:
            summary.avg_time_to_close_seconds = (
                sum(close_durations) / len(close_durations)
            )

        return summary

    # -- Internals -----------------------------------------------------------

    def _append(
        self, event_type: str, metadata: Dict[str, Any],
    ) -> PerformanceEvent:
        event = PerformanceEvent(
            agent_id=self.agent_id,
            event_type=event_type,
            timestamp=_now_iso(),
            metadata=metadata,
        )
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        return event

    def _ensure_parent(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

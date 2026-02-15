"""Append-only JSONL event logger."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction


class EventLog:
    """
    Append-only JSONL event logger.

    Events are stored as newline-delimited JSON for efficient
    append operations and streaming reads.
    """

    def __init__(self, path: Path):
        """
        Initialize event log.

        Args:
            path: Path to the JSONL log file
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ensure_parent_exists()

    def _ensure_parent_exists(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        """
        Append an event to the log.

        Args:
            event: Event to append
        """
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def append_many(self, events: List[Event]) -> None:
        """
        Append multiple events to the log.

        Args:
            events: List of events to append
        """
        with self._lock:
            with open(self.path, "a") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + "\n")

    def replay(self) -> Iterator[Event]:
        """
        Iterate over all logged events.

        Yields:
            Event objects in chronological order
        """
        if not self.path.exists():
            return

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Event.from_dict(json.loads(line))

    def replay_filtered(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interaction_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Iterator[Event]:
        """
        Iterate over events with filtering.

        Args:
            event_types: Filter by event types
            start_time: Filter events after this time
            end_time: Filter events before this time
            interaction_id: Filter by interaction ID
            agent_id: Filter by agent ID

        Yields:
            Filtered Event objects
        """
        for event in self.replay():
            # Type filter
            if event_types and event.event_type not in event_types:
                continue

            # Time filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            # ID filters
            if interaction_id and event.interaction_id != interaction_id:
                continue
            if agent_id:
                if (
                    event.agent_id != agent_id
                    and event.initiator_id != agent_id
                    and event.counterparty_id != agent_id
                ):
                    continue

            yield event

    def to_interactions(self) -> List[SoftInteraction]:
        """
        Reconstruct interactions from event stream.

        Returns:
            List of SoftInteraction objects
        """
        # Track interaction state
        interactions: Dict[str, Dict[str, Any]] = {}

        for event in self.replay():
            if event.event_type == EventType.INTERACTION_PROPOSED:
                iid = event.interaction_id
                if iid:
                    interactions[iid] = {
                        "interaction_id": iid,
                        "timestamp": event.timestamp,
                        "initiator": event.initiator_id or "",
                        "counterparty": event.counterparty_id or "",
                        "interaction_type": event.payload.get(
                            "interaction_type", "reply"
                        ),
                        "v_hat": event.payload.get("v_hat", 0.0),
                        "p": event.payload.get("p", 0.5),
                        "accepted": False,
                    }

            elif event.event_type == EventType.INTERACTION_ACCEPTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    interactions[iid]["accepted"] = True

            elif event.event_type == EventType.INTERACTION_REJECTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    interactions[iid]["accepted"] = False

            elif event.event_type == EventType.PAYOFF_COMPUTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    components = event.payload.get("components", {})
                    interactions[iid].update(
                        {
                            "tau": components.get("tau", 0.0),
                            "c_a": components.get("c_a", 0.0),
                            "c_b": components.get("c_b", 0.0),
                            "r_a": components.get("r_a", 0.0),
                            "r_b": components.get("r_b", 0.0),
                        }
                    )

        # Convert to SoftInteraction objects
        result = []
        for data in interactions.values():
            interaction = SoftInteraction(
                interaction_id=data["interaction_id"],
                timestamp=data["timestamp"],
                initiator=data["initiator"],
                counterparty=data["counterparty"],
                interaction_type=InteractionType(data["interaction_type"]),
                accepted=data["accepted"],
                v_hat=data.get("v_hat", 0.0),
                p=data.get("p", 0.5),
                tau=data.get("tau", 0.0),
                c_a=data.get("c_a", 0.0),
                c_b=data.get("c_b", 0.0),
                r_a=data.get("r_a", 0.0),
                r_b=data.get("r_b", 0.0),
            )
            result.append(interaction)

        # Sort by timestamp
        result.sort(key=lambda x: x.timestamp)
        return result

    def count(self) -> int:
        """Count total events in log."""
        count = 0
        for _ in self.replay():
            count += 1
        return count

    def count_by_type(self) -> Dict[EventType, int]:
        """Count events by type."""
        counts: Dict[EventType, int] = {}
        for event in self.replay():
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts

    def last_event(self) -> Optional[Event]:
        """Get the most recent event."""
        last = None
        for event in self.replay():
            last = event
        return last

    def clear(self) -> None:
        """Archive the current log and start fresh.

        Preserves append-only semantics by rotating the existing file
        to ``<name>.cleared_<timestamp>.jsonl`` before truncating.
        If the log file does not exist, this is a no-op.
        """
        if self.path.exists():
            self.rotate(suffix="cleared_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            # rotate() moves the file, so nothing left to delete

    def rotate(self, suffix: Optional[str] = None) -> Path:
        """
        Rotate the log file to a new name.

        Args:
            suffix: Custom suffix for rotated file (default: timestamp)

        Returns:
            Path to the rotated file
        """
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        new_path = self.path.with_suffix(f".{suffix}.jsonl")
        if self.path.exists():
            self.path.rename(new_path)
        return new_path

    def __len__(self) -> int:
        """Return event count."""
        return self.count()

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events."""
        return self.replay()

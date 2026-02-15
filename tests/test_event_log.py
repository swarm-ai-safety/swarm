"""Tests for append-only JSONL event logger."""

from datetime import datetime, timedelta
from typing import Optional

from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType

# =============================================================================
# Helpers
# =============================================================================


def _make_event(
    event_type: EventType = EventType.INTERACTION_PROPOSED,
    interaction_id: Optional[str] = None,
    initiator_id: Optional[str] = None,
    counterparty_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    payload: Optional[dict] = None,
    timestamp: Optional[datetime] = None,
) -> Event:
    """Create a test event."""
    return Event(
        event_type=event_type,
        interaction_id=interaction_id,
        initiator_id=initiator_id,
        counterparty_id=counterparty_id,
        agent_id=agent_id,
        payload=payload or {},
        timestamp=timestamp or datetime.now(),
    )


def _make_interaction_events(
    interaction_id: str,
    initiator: str,
    counterparty: str,
    accepted: bool = True,
    p: float = 0.7,
    v_hat: float = 0.4,
    base_time: Optional[datetime] = None,
) -> list[Event]:
    """Create a PROPOSED → ACCEPTED/REJECTED → PAYOFF event stream."""
    base = base_time or datetime.now()
    events = []

    # Proposed
    events.append(
        Event(
            event_type=EventType.INTERACTION_PROPOSED,
            interaction_id=interaction_id,
            initiator_id=initiator,
            counterparty_id=counterparty,
            payload={"interaction_type": "reply", "v_hat": v_hat, "p": p},
            timestamp=base,
        )
    )

    # Accepted or rejected
    if accepted:
        events.append(
            Event(
                event_type=EventType.INTERACTION_ACCEPTED,
                interaction_id=interaction_id,
                timestamp=base + timedelta(seconds=1),
            )
        )
    else:
        events.append(
            Event(
                event_type=EventType.INTERACTION_REJECTED,
                interaction_id=interaction_id,
                timestamp=base + timedelta(seconds=1),
            )
        )

    # Payoff
    events.append(
        Event(
            event_type=EventType.PAYOFF_COMPUTED,
            interaction_id=interaction_id,
            initiator_id=initiator,
            counterparty_id=counterparty,
            payload={
                "components": {
                    "tau": 1.0,
                    "c_a": 0.1,
                    "c_b": 0.05,
                    "r_a": 0.2,
                    "r_b": 0.15,
                }
            },
            timestamp=base + timedelta(seconds=2),
        )
    )

    return events


# =============================================================================
# append / append_many
# =============================================================================


class TestAppend:
    """Tests for append and append_many."""

    def test_append_single_event(self, tmp_path):
        """append() writes one JSONL line."""
        log = EventLog(tmp_path / "log.jsonl")
        event = _make_event()
        log.append(event)
        assert log.count() == 1

    def test_append_many(self, tmp_path):
        """append_many() writes multiple events atomically."""
        log = EventLog(tmp_path / "log.jsonl")
        events = [_make_event() for _ in range(5)]
        log.append_many(events)
        assert log.count() == 5

    def test_append_creates_file(self, tmp_path):
        """append() creates the file if it doesn't exist."""
        path = tmp_path / "subdir" / "log.jsonl"
        log = EventLog(path)
        log.append(_make_event())
        assert path.exists()

    def test_append_incremental(self, tmp_path):
        """Multiple append() calls accumulate."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event())
        log.append(_make_event())
        log.append(_make_event())
        assert log.count() == 3


# =============================================================================
# replay
# =============================================================================


class TestReplay:
    """Tests for replay."""

    def test_replay_empty_log(self, tmp_path):
        """replay() on non-existent file yields nothing."""
        log = EventLog(tmp_path / "missing.jsonl")
        events = list(log.replay())
        assert events == []

    def test_replay_returns_events_in_order(self, tmp_path):
        """replay() yields events in insertion order."""
        log = EventLog(tmp_path / "log.jsonl")
        base = datetime(2024, 1, 1)
        for i in range(5):
            log.append(
                _make_event(
                    event_type=EventType.INTERACTION_PROPOSED,
                    timestamp=base + timedelta(seconds=i),
                )
            )

        events = list(log.replay())
        assert len(events) == 5
        for _i, e in enumerate(events):
            assert e.event_type == EventType.INTERACTION_PROPOSED

    def test_replay_preserves_fields(self, tmp_path):
        """replay() round-trips all event fields."""
        log = EventLog(tmp_path / "log.jsonl")
        original = Event(
            event_type=EventType.PAYOFF_COMPUTED,
            interaction_id="iid-123",
            agent_id="agent_1",
            initiator_id="init_1",
            counterparty_id="cp_1",
            payload={"tau": 1.5, "components": {"c_a": 0.1}},
            epoch=3,
            step=7,
            scenario_id="baseline",
            replay_k=2,
            seed=42,
        )
        log.append(original)

        replayed = list(log.replay())[0]
        assert replayed.event_type == EventType.PAYOFF_COMPUTED
        assert replayed.interaction_id == "iid-123"
        assert replayed.agent_id == "agent_1"
        assert replayed.initiator_id == "init_1"
        assert replayed.counterparty_id == "cp_1"
        assert replayed.payload["tau"] == 1.5
        assert replayed.epoch == 3
        assert replayed.step == 7
        assert replayed.scenario_id == "baseline"
        assert replayed.replay_k == 2
        assert replayed.seed == 42

    def test_from_dict_backwards_compatible_without_replay_fields(self):
        """Older event payloads without replay metadata should still parse."""
        raw = {
            "event_id": "evt-1",
            "timestamp": datetime.now().isoformat(),
            "event_type": EventType.SIMULATION_STARTED.value,
            "interaction_id": None,
            "agent_id": None,
            "initiator_id": None,
            "counterparty_id": None,
            "payload": {},
            "epoch": 0,
            "step": 0,
        }
        event = Event.from_dict(raw)
        assert event.scenario_id is None
        assert event.replay_k is None
        assert event.seed is None


# =============================================================================
# replay_filtered
# =============================================================================


class TestReplayFiltered:
    """Tests for replay_filtered."""

    def test_filter_by_event_type(self, tmp_path):
        """Filter by event_type list."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event(event_type=EventType.INTERACTION_PROPOSED))
        log.append(_make_event(event_type=EventType.PAYOFF_COMPUTED))
        log.append(_make_event(event_type=EventType.INTERACTION_ACCEPTED))

        result = list(
            log.replay_filtered(
                event_types=[EventType.PAYOFF_COMPUTED],
            )
        )
        assert len(result) == 1
        assert result[0].event_type == EventType.PAYOFF_COMPUTED

    def test_filter_by_time_range(self, tmp_path):
        """Filter by start_time / end_time."""
        log = EventLog(tmp_path / "log.jsonl")
        base = datetime(2024, 6, 1)
        for i in range(5):
            log.append(_make_event(timestamp=base + timedelta(hours=i)))

        result = list(
            log.replay_filtered(
                start_time=base + timedelta(hours=1),
                end_time=base + timedelta(hours=3),
            )
        )
        assert len(result) == 3  # hours 1, 2, 3

    def test_filter_by_interaction_id(self, tmp_path):
        """Filter by interaction_id."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event(interaction_id="iid-A"))
        log.append(_make_event(interaction_id="iid-B"))
        log.append(_make_event(interaction_id="iid-A"))

        result = list(log.replay_filtered(interaction_id="iid-A"))
        assert len(result) == 2

    def test_filter_by_agent_id(self, tmp_path):
        """Filter by agent_id (matches agent_id, initiator_id, or counterparty_id)."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event(agent_id="agent_X"))
        log.append(_make_event(initiator_id="agent_X"))
        log.append(_make_event(counterparty_id="agent_X"))
        log.append(_make_event(agent_id="agent_Y"))

        result = list(log.replay_filtered(agent_id="agent_X"))
        assert len(result) == 3

    def test_combined_filters(self, tmp_path):
        """Multiple filters applied together."""
        log = EventLog(tmp_path / "log.jsonl")
        base = datetime(2024, 1, 1)
        log.append(
            _make_event(
                event_type=EventType.PAYOFF_COMPUTED,
                agent_id="agent_X",
                timestamp=base,
            )
        )
        log.append(
            _make_event(
                event_type=EventType.PAYOFF_COMPUTED,
                agent_id="agent_Y",
                timestamp=base + timedelta(hours=1),
            )
        )
        log.append(
            _make_event(
                event_type=EventType.INTERACTION_PROPOSED,
                agent_id="agent_X",
                timestamp=base + timedelta(hours=2),
            )
        )

        result = list(
            log.replay_filtered(
                event_types=[EventType.PAYOFF_COMPUTED],
                agent_id="agent_X",
            )
        )
        assert len(result) == 1


# =============================================================================
# to_interactions
# =============================================================================


class TestToInteractions:
    """Tests for to_interactions."""

    def test_reconstruct_accepted_interaction(self, tmp_path):
        """Should reconstruct accepted SoftInteraction from event stream."""
        log = EventLog(tmp_path / "log.jsonl")
        events = _make_interaction_events(
            "iid-1",
            "alice",
            "bob",
            accepted=True,
            p=0.8,
            v_hat=0.5,
        )
        log.append_many(events)

        interactions = log.to_interactions()
        assert len(interactions) == 1

        ix = interactions[0]
        assert ix.interaction_id == "iid-1"
        assert ix.initiator == "alice"
        assert ix.counterparty == "bob"
        assert ix.accepted is True
        assert ix.p == 0.8
        assert ix.v_hat == 0.5
        assert ix.tau == 1.0
        assert ix.c_a == 0.1
        assert ix.c_b == 0.05
        assert ix.r_a == 0.2
        assert ix.r_b == 0.15
        assert ix.interaction_type == InteractionType.REPLY

    def test_reconstruct_rejected_interaction(self, tmp_path):
        """Should reconstruct rejected interaction."""
        log = EventLog(tmp_path / "log.jsonl")
        events = _make_interaction_events(
            "iid-2",
            "carol",
            "dave",
            accepted=False,
        )
        log.append_many(events)

        interactions = log.to_interactions()
        assert len(interactions) == 1
        assert interactions[0].accepted is False

    def test_multiple_interactions(self, tmp_path):
        """Should reconstruct multiple interactions sorted by timestamp."""
        log = EventLog(tmp_path / "log.jsonl")
        base = datetime(2024, 1, 1)
        for i in range(3):
            events = _make_interaction_events(
                f"iid-{i}",
                f"agent_{i}",
                f"agent_{i + 1}",
                base_time=base + timedelta(minutes=i),
            )
            log.append_many(events)

        interactions = log.to_interactions()
        assert len(interactions) == 3
        # Sorted by timestamp
        for i in range(len(interactions) - 1):
            assert interactions[i].timestamp <= interactions[i + 1].timestamp

    def test_empty_log_returns_empty(self, tmp_path):
        """to_interactions on empty log returns empty list."""
        log = EventLog(tmp_path / "empty.jsonl")
        assert log.to_interactions() == []

    def test_partial_event_stream(self, tmp_path):
        """Interaction with only PROPOSED event still reconstructed."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(
            Event(
                event_type=EventType.INTERACTION_PROPOSED,
                interaction_id="iid-partial",
                initiator_id="a",
                counterparty_id="b",
                payload={"interaction_type": "reply", "p": 0.6},
            )
        )

        interactions = log.to_interactions()
        assert len(interactions) == 1
        assert interactions[0].accepted is False  # default


# =============================================================================
# count / count_by_type
# =============================================================================


class TestCounting:
    """Tests for count and count_by_type."""

    def test_count_empty(self, tmp_path):
        """count() on empty log returns 0."""
        log = EventLog(tmp_path / "empty.jsonl")
        assert log.count() == 0

    def test_count_after_appends(self, tmp_path):
        """count() reflects total events."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append_many([_make_event() for _ in range(7)])
        assert log.count() == 7

    def test_count_by_type(self, tmp_path):
        """count_by_type() groups correctly."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event(event_type=EventType.INTERACTION_PROPOSED))
        log.append(_make_event(event_type=EventType.INTERACTION_PROPOSED))
        log.append(_make_event(event_type=EventType.PAYOFF_COMPUTED))

        counts = log.count_by_type()
        assert counts[EventType.INTERACTION_PROPOSED] == 2
        assert counts[EventType.PAYOFF_COMPUTED] == 1

    def test_count_by_type_empty(self, tmp_path):
        """count_by_type() on empty log returns empty dict."""
        log = EventLog(tmp_path / "empty.jsonl")
        assert log.count_by_type() == {}


# =============================================================================
# last_event
# =============================================================================


class TestLastEvent:
    """Tests for last_event."""

    def test_last_event_empty(self, tmp_path):
        """last_event() returns None on empty log."""
        log = EventLog(tmp_path / "empty.jsonl")
        assert log.last_event() is None

    def test_last_event_returns_last(self, tmp_path):
        """last_event() returns the most recently appended event."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event(event_type=EventType.INTERACTION_PROPOSED))
        last = _make_event(event_type=EventType.PAYOFF_COMPUTED)
        log.append(last)

        result = log.last_event()
        assert result.event_type == EventType.PAYOFF_COMPUTED
        assert result.event_id == last.event_id


# =============================================================================
# clear / rotate
# =============================================================================


class TestClearAndRotate:
    """Tests for clear and rotate."""

    def test_clear_archives_file(self, tmp_path):
        """clear() archives the log and leaves count at 0."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event())
        assert log.path.exists()

        log.clear()
        # Original path should no longer exist (was rotated)
        assert not log.path.exists()
        assert log.count() == 0
        # An archived copy should exist in the same directory
        archived = list(tmp_path.glob("log.cleared_*.jsonl"))
        assert len(archived) == 1

    def test_clear_nonexistent_file(self, tmp_path):
        """clear() on non-existent file does not raise."""
        log = EventLog(tmp_path / "missing.jsonl")
        log.clear()  # Should not raise

    def test_rotate_with_custom_suffix(self, tmp_path):
        """rotate() renames file with custom suffix."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event())

        new_path = log.rotate(suffix="backup")
        assert new_path == tmp_path / "log.backup.jsonl"
        assert new_path.exists()
        assert not log.path.exists()

    def test_rotate_with_default_suffix(self, tmp_path):
        """rotate() uses timestamp suffix by default."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append(_make_event())

        new_path = log.rotate()
        assert new_path.exists()
        assert not log.path.exists()
        # Suffix is timestamp-based
        assert "log." in str(new_path.name)
        assert new_path.suffix == ".jsonl"

    def test_rotate_nonexistent_file(self, tmp_path):
        """rotate() on non-existent file returns target path without error."""
        log = EventLog(tmp_path / "missing.jsonl")
        new_path = log.rotate(suffix="test")
        # File doesn't exist, but path is still returned
        assert new_path == tmp_path / "missing.test.jsonl"


# =============================================================================
# __len__ / __iter__
# =============================================================================


class TestProtocols:
    """Tests for __len__ and __iter__ protocols."""

    def test_len(self, tmp_path):
        """len(log) returns event count."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append_many([_make_event() for _ in range(4)])
        assert len(log) == 4

    def test_iter(self, tmp_path):
        """iter(log) yields events."""
        log = EventLog(tmp_path / "log.jsonl")
        log.append_many([_make_event() for _ in range(3)])

        events = list(log)
        assert len(events) == 3
        for e in events:
            assert isinstance(e, Event)

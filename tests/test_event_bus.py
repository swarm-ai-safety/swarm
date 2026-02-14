"""Tests for the synchronous EventBus."""

from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType


def _make_event(**kwargs) -> Event:
    return Event(event_type=EventType.AGENT_CREATED, **kwargs)


class TestEventBusSubscription:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        event = _make_event()
        bus.emit(event)
        assert received == [event]

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe(a.append)
        bus.subscribe(b.append)
        event = _make_event()
        bus.emit(event)
        assert a == [event]
        assert b == [event]

    def test_duplicate_subscribe_ignored(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.subscribe(received.append)
        bus.emit(_make_event())
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.unsubscribe(received.append)
        bus.emit(_make_event())
        assert received == []

    def test_unsubscribe_noop_if_not_registered(self):
        bus = EventBus()
        bus.unsubscribe(lambda e: None)  # Should not raise


class TestEventBusEnrichment:
    def test_enrichment_applied(self):
        bus = EventBus()
        bus.set_enrichment(seed=42, scenario_id="test", replay_k=3)
        received = []
        bus.subscribe(received.append)

        event = _make_event()
        assert event.seed is None
        bus.emit(event)

        assert event.seed == 42
        assert event.scenario_id == "test"
        assert event.replay_k == 3

    def test_event_level_values_take_priority(self):
        bus = EventBus()
        bus.set_enrichment(seed=42, scenario_id="default")
        received = []
        bus.subscribe(received.append)

        event = _make_event(seed=99, scenario_id="custom")
        bus.emit(event)

        assert event.seed == 99
        assert event.scenario_id == "custom"

    def test_partial_enrichment(self):
        bus = EventBus()
        bus.set_enrichment(seed=42)
        event = _make_event()
        bus.emit(event)

        assert event.seed == 42
        assert event.scenario_id is None  # Not enriched
        assert event.replay_k is None

    def test_no_enrichment(self):
        bus = EventBus()
        event = _make_event()
        bus.emit(event)
        assert event.seed is None


class TestEventBusEmitOrder:
    def test_subscribers_called_in_registration_order(self):
        bus = EventBus()
        order = []
        bus.subscribe(lambda e: order.append("first"))
        bus.subscribe(lambda e: order.append("second"))
        bus.subscribe(lambda e: order.append("third"))
        bus.emit(_make_event())
        assert order == ["first", "second", "third"]

    def test_emit_with_no_subscribers(self):
        bus = EventBus()
        bus.emit(_make_event())  # Should not raise

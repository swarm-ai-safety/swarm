"""Tests for the SWARM-Ralph bridge."""

import json

from swarm.bridges.ralph.bridge import RalphBridge
from swarm.bridges.ralph.config import RalphConfig
from swarm.bridges.ralph.events import RalphEvent, RalphEventType
from swarm.bridges.ralph.mapper import RalphMapper


def test_ralph_event_from_dict_falls_back_to_generic() -> None:
    event = RalphEvent.from_dict(
        {
            "event_type": "unknown:type",
            "actor_id": "worker-1",
            "task_id": "t-1",
        }
    )
    assert event.event_type == RalphEventType.GENERIC
    assert event.actor_id == "worker-1"


def test_ralph_event_from_dict_supports_alias_fields() -> None:
    event = RalphEvent.from_dict(
        {
            "id": "evt-1",
            "type": "task:completed",
            "time": "2026-01-01T00:00:00",
            "agent": "worker-2",
            "job": "job-7",
            "payload": "ignored",
        }
    )
    assert event.event_id == "evt-1"
    assert event.event_type == RalphEventType.TASK_COMPLETED
    assert event.actor_id == "worker-2"
    assert event.task_id == "job-7"
    assert event.payload == {}
    assert event.timestamp.tzinfo is not None


def test_mapper_maps_failure_event_to_negative_progress() -> None:
    mapper = RalphMapper()
    event = RalphEvent(
        event_type=RalphEventType.TASK_FAILED,
        actor_id="worker-1",
        task_id="t-1",
    )

    interaction = mapper.map_event(
        event=event,
        initiator="ralph_orchestrator",
        counterparty="worker-1",
    )

    assert interaction.task_progress_delta < 0
    assert interaction.metadata["bridge"] == "ralph"
    assert interaction.metadata["event_type"] == "task:failed"


def test_bridge_polls_incremental_jsonl_events(tmp_path) -> None:
    events_path = tmp_path / "ralph-events.jsonl"
    first = {
        "event_id": "e-1",
        "event_type": "task:completed",
        "actor_id": "alice",
        "task_id": "task-1",
        "payload": {"task_progress_delta": 0.9},
    }
    second = {
        "event_id": "e-2",
        "event_type": "review:rejected",
        "actor_id": "alice",
        "task_id": "task-2",
    }
    events_path.write_text(json.dumps(first) + "\n", encoding="utf-8")

    bridge = RalphBridge(
        RalphConfig(
            events_path=str(events_path),
            agent_role_map={"alice": "agent_alice"},
        )
    )

    interactions = bridge.poll()
    assert len(interactions) == 1
    assert interactions[0].counterparty == "agent_alice"

    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second) + "\n")

    next_interactions = bridge.poll()
    assert len(next_interactions) == 1
    assert next_interactions[0].metadata["event_id"] == "e-2"
    assert len(bridge.get_events()) == 2
    assert len(bridge.get_interactions()) == 2


def test_bridge_handles_file_truncation(tmp_path) -> None:
    events_path = tmp_path / "ralph-events.jsonl"
    first = {
        "event_id": "e-1",
        "event_type": "task:completed",
        "actor_id": "alice",
        "task_id": "task-1",
    }
    replacement = {
        "event_id": "e-2",
        "event_type": "task:started",
        "actor_id": "alice",
        "task_id": "task-2",
    }

    events_path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    bridge = RalphBridge(RalphConfig(events_path=str(events_path)))
    assert len(bridge.poll()) == 1

    events_path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")
    interactions = bridge.poll()

    assert len(interactions) == 1
    assert interactions[0].metadata["event_id"] == "e-2"

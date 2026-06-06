"""Round-trip + parsing tests for swarm.bridges.agentveil.events.

Covers the brittleness paths flagged in the PR #504 review:
- missing event_type
- unknown event_type
- 'Z'-suffixed ISO timestamps (Python 3.10's fromisoformat doesn't accept them)
- naive datetimes (no tzinfo)
- ``step`` arriving as ``None``
- eager ``_utcnow()`` on the timestamp-present path (verified via patch)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from swarm.bridges.agentveil.events import (
    AgentVeilEvent,
    AgentVeilEventType,
    AttestationEvent,
    ReputationSnapshotEvent,
    TrustDecisionEvent,
)


class TestAgentVeilEventRoundTrip:
    def test_to_dict_from_dict_round_trip_preserves_fields(self) -> None:
        original = AgentVeilEvent(
            event_id="evt-1",
            event_type=AgentVeilEventType.ATTESTATION_SUBMITTED,
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            subject_did="did:key:alice",
            actor_did="did:key:orchestrator",
            step=7,
            payload={"outcome_sign": 1},
        )
        restored = AgentVeilEvent.from_dict(original.to_dict())
        assert restored.event_id == "evt-1"
        assert restored.event_type is AgentVeilEventType.ATTESTATION_SUBMITTED
        assert restored.timestamp == original.timestamp
        assert restored.subject_did == "did:key:alice"
        assert restored.actor_did == "did:key:orchestrator"
        assert restored.step == 7
        assert restored.payload == {"outcome_sign": 1}

    def test_missing_event_type_defaults_to_generic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            restored = AgentVeilEvent.from_dict({"timestamp": "2026-01-01T00:00:00+00:00"})
        assert restored.event_type is AgentVeilEventType.GENERIC
        assert "missing event_type" in caplog.text

    def test_unknown_event_type_defaults_to_generic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            restored = AgentVeilEvent.from_dict(
                {"event_type": "not:a:real:type", "timestamp": "2026-01-01T00:00:00+00:00"}
            )
        assert restored.event_type is AgentVeilEventType.GENERIC
        assert "unknown event_type" in caplog.text

    def test_z_suffix_timestamp_parses_on_py310(self) -> None:
        restored = AgentVeilEvent.from_dict(
            {"event_type": "generic", "timestamp": "2026-01-01T00:00:00Z"}
        )
        assert restored.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_naive_timestamp_assumed_utc(self) -> None:
        restored = AgentVeilEvent.from_dict(
            {"event_type": "generic", "timestamp": "2026-01-01T00:00:00"}
        )
        assert restored.timestamp.tzinfo is timezone.utc
        assert restored.timestamp.year == 2026

    def test_step_none_does_not_raise(self) -> None:
        restored = AgentVeilEvent.from_dict({"event_type": "generic", "step": None})
        assert restored.step == 0

    def test_payload_none_becomes_empty_dict(self) -> None:
        restored = AgentVeilEvent.from_dict({"event_type": "generic", "payload": None})
        assert restored.payload == {}

    def test_timestamp_present_does_not_call_utcnow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from swarm.bridges.agentveil import events as ev_module

        calls = {"n": 0}

        def boom() -> datetime:
            calls["n"] += 1
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

        monkeypatch.setattr(ev_module, "_utcnow", boom)
        AgentVeilEvent.from_dict(
            {"event_type": "generic", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        assert calls["n"] == 0

    def test_unparseable_timestamp_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            restored = AgentVeilEvent.from_dict(
                {"event_type": "generic", "timestamp": "not-a-timestamp"}
            )
        assert restored.timestamp.tzinfo is timezone.utc
        assert "unparseable timestamp" in caplog.text


class TestTypedPayloadRoundTrips:
    def test_trust_decision_round_trip(self) -> None:
        original = TrustDecisionEvent(
            allowed=True, tier="trusted", risk_level="low", confidence=0.9, reason="ok"
        )
        assert TrustDecisionEvent.from_dict(original.to_dict()) == original

    def test_attestation_round_trip(self) -> None:
        original = AttestationEvent(
            interaction_id="int-1",
            outcome_sign=-1,
            evidence_hash="ab" * 32,
            suppression_reason=None,
        )
        assert AttestationEvent.from_dict(original.to_dict()) == original

    def test_reputation_snapshot_round_trip(self) -> None:
        original = ReputationSnapshotEvent(
            score=0.73, confidence=0.5, tier="basic", attestation_count=12
        )
        assert ReputationSnapshotEvent.from_dict(original.to_dict()) == original

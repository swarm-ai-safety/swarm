"""Tests for the SimWorld delivery → SoftMetrics adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.domains.simworld_delivery.adapter import (
    DeliveryAdapter,
    _event_to_observables,
)
from swarm.domains.simworld_delivery.entities import DeliveryEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(event_type: str, agent_id: str = "driver_a",
                step: int = 0, epoch: int = 0, **details) -> dict:
    return {
        "event_type": event_type,
        "agent_id": agent_id,
        "step": step,
        "epoch": epoch,
        "details": details,
    }


def _healthy_events() -> list[dict]:
    """A batch of events representing a healthy delivery economy."""
    return [
        _make_event("bid_won", "driver_a", 1, 0, bid_amount=8.0, fee=1.0, num_bidders=3),
        _make_event("delivery_complete", "driver_a", 5, 0, payout=9.0, on_time=True, elapsed_steps=4),
        _make_event("bid_won", "driver_b", 2, 0, bid_amount=7.0, fee=0.8, num_bidders=2),
        _make_event("delivery_complete", "driver_b", 6, 0, payout=8.0, on_time=True, elapsed_steps=4),
        _make_event("order_shared", "driver_a", 3, 0, order_id="o1", shared_with="driver_b"),
        _make_event("sharing_bonus", "driver_b", 7, 0, order_id="o1", bonus=1.0),
    ]


def _toxic_events() -> list[dict]:
    """Events with many failures — should produce higher toxicity."""
    return [
        _make_event("delivery_failed", "driver_x", 2, 0, order_id="o1", penalty=5.0),
        _make_event("delivery_failed", "driver_x", 5, 0, order_id="o2", penalty=5.0),
        _make_event("delivery_failed", "driver_y", 3, 0, order_id="o3", penalty=3.0),
        _make_event("bid_won", "driver_y", 1, 0, bid_amount=10.0, fee=1.5, num_bidders=1),
    ]


# ---------------------------------------------------------------------------
# Tests: observable mapping
# ---------------------------------------------------------------------------

class TestObservableMapping:
    def test_delivery_complete_on_time(self):
        obs = _event_to_observables(
            _make_event("delivery_complete", on_time=True)
        )
        assert obs is not None
        assert obs.task_progress_delta == 1.0
        assert obs.rework_count == 0

    def test_delivery_complete_late(self):
        obs = _event_to_observables(
            _make_event("delivery_complete", on_time=False)
        )
        assert obs is not None
        assert obs.task_progress_delta == pytest.approx(0.4)
        assert obs.rework_count == 1

    def test_delivery_failed(self):
        obs = _event_to_observables(
            _make_event("delivery_failed", penalty=5.0)
        )
        assert obs is not None
        assert obs.task_progress_delta < 0
        assert obs.verifier_rejections == 1

    def test_bid_won(self):
        obs = _event_to_observables(
            _make_event("bid_won", num_bidders=4, bid_amount=8.0, fee=1.0)
        )
        assert obs is not None
        assert obs.task_progress_delta > 0

    def test_order_shared(self):
        obs = _event_to_observables(
            _make_event("order_shared", shared_with="driver_b")
        )
        assert obs is not None
        assert obs.counterparty_engagement_delta > 0.5

    def test_non_interaction_event_returns_none(self):
        assert _event_to_observables(_make_event("order_created")) is None
        assert _event_to_observables(_make_event("wait")) is None
        assert _event_to_observables(_make_event("deliver_continue")) is None


# ---------------------------------------------------------------------------
# Tests: adapter replay
# ---------------------------------------------------------------------------

class TestDeliveryAdapter:
    def test_replay_from_jsonl(self, tmp_path: Path):
        events = _healthy_events()
        log_file = tmp_path / "event_log.jsonl"
        log_file.write_text("\n".join(json.dumps(e) for e in events))

        adapter = DeliveryAdapter()
        report = adapter.replay(log_file)

        assert report.n_interactions == 6
        assert report.n_interactions == report.n_accepted + report.n_rejected
        assert 0.0 <= report.toxicity_rate <= 1.0
        assert 0.0 <= report.mean_p <= 1.0

    def test_replay_missing_file(self):
        adapter = DeliveryAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.replay("/nonexistent/path.jsonl")

    def test_from_events(self):
        raw = _healthy_events()
        events = [
            DeliveryEvent(
                event_type=e["event_type"],
                step=e["step"],
                epoch=e["epoch"],
                agent_id=e["agent_id"],
                details=e["details"],
            )
            for e in raw
        ]
        adapter = DeliveryAdapter()
        report = adapter.from_events(events)
        assert report.n_interactions == 6

    def test_empty_events(self):
        adapter = DeliveryAdapter()
        report = adapter.from_events([])
        assert report.n_interactions == 0
        assert report.toxicity_rate == 0.0

    def test_healthy_vs_toxic(self):
        adapter = DeliveryAdapter()

        healthy = adapter._process_events(_healthy_events())
        toxic = adapter._process_events(_toxic_events())

        # Healthy economy should have higher mean p
        assert healthy.mean_p > toxic.mean_p
        # Toxic economy should show higher toxicity
        assert toxic.toxicity_rate >= healthy.toxicity_rate

    def test_agent_metrics_breakdown(self):
        adapter = DeliveryAdapter()
        report = adapter._process_events(_healthy_events())

        assert "driver_a" in report.agent_metrics
        assert "driver_b" in report.agent_metrics
        assert report.agent_metrics["driver_a"]["n_interactions"] >= 1

    def test_p_invariant(self):
        """All p values must be in [0, 1] — core safety invariant."""
        adapter = DeliveryAdapter()
        report = adapter._process_events(_healthy_events() + _toxic_events())
        for ix in report.interactions:
            assert 0.0 <= ix.p <= 1.0, f"p={ix.p} out of bounds"

    def test_metadata_includes_bridge(self):
        adapter = DeliveryAdapter()
        report = adapter._process_events(_healthy_events())
        for ix in report.interactions:
            assert ix.metadata["bridge"] == "simworld_delivery"
            assert "event_type" in ix.metadata

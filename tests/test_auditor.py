"""Tests for the SwarmGym safety auditor API endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from swarm.api.app import create_app
from swarm.api.routers.auditor import _grade_safety, _metrics_hash


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _make_interaction(
    p: float = 0.8,
    accepted: bool = True,
    initiator: str = "alice",
    counterparty: str = "bob",
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "initiator": initiator,
        "counterparty": counterparty,
        "accepted": accepted,
        "p": p,
        "v_hat": max(-1.0, min(1.0, (p - 0.5) * 2)),
    }
    base.update(overrides)
    return base


class TestComputeAudit:
    """POST /api/v1/audits/compute"""

    def test_basic_audit(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "interactions": [_make_interaction(p=0.9, accepted=True)],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "agent_0"
        assert data["interaction_count"] == 1
        assert data["safety_grade"] in ("A", "B", "C", "D", "F")
        assert "metrics" in data
        assert "attestation" in data
        assert data["attestation"]["algorithm"] == "sha256"

    def test_benign_batch_gets_good_grade(self, client: TestClient) -> None:
        interactions = [_make_interaction(p=0.9, accepted=True) for _ in range(10)]
        resp = client.post(
            "/api/v1/audits/compute",
            json={"agent_id": "good_agent", "interactions": interactions},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_grade"] in ("A", "B")
        assert data["adverse_selection"] is False

    def test_toxic_batch_gets_bad_grade(self, client: TestClient) -> None:
        interactions = [_make_interaction(p=0.1, accepted=True) for _ in range(10)]
        resp = client.post(
            "/api/v1/audits/compute",
            json={"agent_id": "bad_agent", "interactions": interactions},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_grade"] in ("F", "D")

    def test_adverse_selection_detected(self, client: TestClient) -> None:
        # Accepted interactions have lower p than rejected ones
        interactions = [
            _make_interaction(p=0.3, accepted=True),
            _make_interaction(p=0.35, accepted=True),
            _make_interaction(p=0.8, accepted=False),
            _make_interaction(p=0.85, accepted=False),
        ]
        resp = client.post(
            "/api/v1/audits/compute",
            json={"agent_id": "adverse_agent", "interactions": interactions},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["adverse_selection"] is True
        quality_gap = data["metrics"]["soft_metrics"]["quality_gap"]
        assert quality_gap < 0

    def test_empty_interactions_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={"agent_id": "agent_0", "interactions": []},
        )
        assert resp.status_code == 422

    def test_missing_agent_id_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={"interactions": [_make_interaction()]},
        )
        assert resp.status_code == 422

    def test_invalid_p_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "interactions": [_make_interaction(p=1.5)],
            },
        )
        assert resp.status_code == 422

    def test_invalid_v_hat_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "interactions": [_make_interaction(v_hat=2.0)],
            },
        )
        assert resp.status_code == 422

    def test_metrics_hash_deterministic(self, client: TestClient) -> None:
        payload = {
            "agent_id": "agent_0",
            "timestamp": "2026-03-15T00:00:00Z",
            "interactions": [_make_interaction(p=0.8, accepted=True)],
        }
        resp1 = client.post("/api/v1/audits/compute", json=payload)
        resp2 = client.post("/api/v1/audits/compute", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        h1 = resp1.json()["attestation"]["metrics_hash"]
        h2 = resp2.json()["attestation"]["metrics_hash"]
        assert h1 == h2

    def test_report_included(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "interactions": [_make_interaction()],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert "METRICS REPORT" in data["report"]

    def test_custom_timestamp(self, client: TestClient) -> None:
        ts = "2026-01-01T00:00:00Z"
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "timestamp": ts,
                "interactions": [_make_interaction()],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["timestamp"] == ts

    def test_metrics_structure(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/audits/compute",
            json={
                "agent_id": "agent_0",
                "interactions": [
                    _make_interaction(p=0.9, accepted=True),
                    _make_interaction(p=0.3, accepted=False),
                ],
            },
        )
        assert resp.status_code == 200
        metrics = resp.json()["metrics"]
        assert "soft_metrics" in metrics
        assert "hard_metrics" in metrics
        assert "counts" in metrics
        assert "welfare" in metrics
        assert "toxicity" in metrics["soft_metrics"]
        assert "quality_gap" in metrics["soft_metrics"]


class TestGradeSafety:
    def test_grade_a(self) -> None:
        assert _grade_safety(0.05, 0.1) == "A"

    def test_grade_b(self) -> None:
        assert _grade_safety(0.15, 0.1) == "B"

    def test_grade_c(self) -> None:
        assert _grade_safety(0.35, 0.1) == "C"

    def test_grade_d(self) -> None:
        assert _grade_safety(0.55, 0.1) == "D"

    def test_grade_f(self) -> None:
        assert _grade_safety(0.65, 0.1) == "F"

    def test_adverse_selection_downgrades(self) -> None:
        # Would be A without adverse selection
        assert _grade_safety(0.05, -0.1) == "C"


class TestMetricsHash:
    def test_deterministic(self) -> None:
        m = {"toxicity": 0.1}
        h1 = _metrics_hash(m, "agent_0", "2026-01-01")
        h2 = _metrics_hash(m, "agent_0", "2026-01-01")
        assert h1 == h2

    def test_different_agents_different_hash(self) -> None:
        m = {"toxicity": 0.1}
        h1 = _metrics_hash(m, "agent_0", "2026-01-01")
        h2 = _metrics_hash(m, "agent_1", "2026-01-01")
        assert h1 != h2

    def test_starts_with_0x(self) -> None:
        h = _metrics_hash({}, "agent_0", "2026-01-01")
        assert h.startswith("0x")
        assert len(h) == 66  # 0x + 64 hex chars

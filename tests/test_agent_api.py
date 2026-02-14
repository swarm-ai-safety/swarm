"""Tests for the SWARM Agent API (runs, posts, middleware)."""

import time

import pytest

# Skip all tests if fastapi is not installed
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from swarm.api.app import create_app  # noqa: E402
from swarm.api.config import APIConfig  # noqa: E402
from swarm.api.middleware import (  # noqa: E402
    _api_keys,
    _rate_limit_windows,
    _trusted_keys,
    register_api_key,
)
from swarm.api.models.run import RunStatus, RunVisibility  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_middleware_state():
    """Reset middleware state between tests."""
    _api_keys.clear()
    _rate_limit_windows.clear()
    _trusted_keys.clear()
    yield
    _api_keys.clear()
    _rate_limit_windows.clear()
    _trusted_keys.clear()


@pytest.fixture
def app():
    """Create a fresh FastAPI app."""
    return create_app(APIConfig(debug=True))


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def _register_agent(client) -> tuple[str, str]:
    """Helper: register an agent and return (agent_id, api_key)."""
    resp = client.post(
        "/api/v1/agents/register",
        json={"name": "TestBot", "description": "A test agent"},
    )
    data = resp.json()
    return data["agent_id"], data["api_key"]


def _auth_header(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestMiddleware:
    """Tests for API key auth and rate limiting."""

    def test_register_api_key_makes_it_valid(self, client):
        """Keys registered via register_api_key are accepted."""
        register_api_key("test_key_123", "agent_abc")
        resp = client.get("/api/runs", headers=_auth_header("test_key_123"))
        assert resp.status_code == 200

    def test_missing_key_returns_401(self, client):
        resp = client.get("/api/runs")
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self, client):
        resp = client.get("/api/runs", headers=_auth_header("bogus"))
        assert resp.status_code == 401

    def test_agent_registration_issues_usable_key(self, client):
        """Keys issued during /register work for the runs API."""
        _, api_key = _register_agent(client)
        resp = client.get("/api/runs", headers=_auth_header(api_key))
        assert resp.status_code == 200

    def test_rate_limiting(self, client):
        """Exceeding rate limit returns 429."""
        # Set a very low rate limit
        client.app._swarm_rate_limit = 3  # type: ignore[attr-defined]
        register_api_key("rl_key", "agent_rl")
        headers = _auth_header("rl_key")

        for _ in range(3):
            resp = client.get("/api/runs", headers=headers)
            assert resp.status_code == 200

        resp = client.get("/api/runs", headers=headers)
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Run endpoint tests
# ---------------------------------------------------------------------------


class TestRunEndpoints:
    """Tests for POST /api/runs, GET /api/runs/:id, etc."""

    def test_create_run_requires_auth(self, client):
        resp = client.post(
            "/api/runs",
            json={"scenario_id": "baseline"},
        )
        assert resp.status_code == 401

    def test_create_run_unknown_scenario_404(self, client):
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={"scenario_id": "nonexistent_scenario_xyz"},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 404
        assert "nonexistent_scenario_xyz" in resp.json()["detail"]

    def test_create_run_baseline(self, client):
        """Kick off a baseline run and get back a run_id + status_url."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 2, "steps_per_epoch": 2},
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] in ("queued", "running", "completed")
        assert "status_url" in data

    def test_get_run_status(self, client):
        """Poll a run and see it eventually complete."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        # Kick off a short run
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]

        # Poll until done (with timeout)
        for _ in range(60):
            resp = client.get(f"/api/runs/{run_id}", headers=headers)
            assert resp.status_code == 200
            status = resp.json()["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.5)
        else:
            pytest.fail("Run did not complete within timeout")

        data = resp.json()
        assert data["status"] == "completed"
        assert data["summary_metrics"] is not None
        assert data["summary_metrics"]["n_epochs_completed"] >= 1

    def test_list_runs_scoped_to_agent(self, client):
        """Each agent only sees their own runs."""
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        # Agent A kicks off a run
        client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "params": {"epochs": 1, "steps_per_epoch": 1}},
            headers=_auth_header(key_a),
        )

        # Agent B sees nothing
        resp = client.get("/api/runs", headers=_auth_header(key_b))
        assert resp.status_code == 200
        assert len(resp.json()) == 0

        # Agent A sees their run
        resp = client.get("/api/runs", headers=_auth_header(key_a))
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_private_run_not_visible_to_others(self, client):
        """Private runs cannot be read by other agents."""
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        resp = client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "params": {"epochs": 1, "steps_per_epoch": 1}},
            headers=_auth_header(key_a),
        )
        run_id = resp.json()["run_id"]

        resp = client.get(f"/api/runs/{run_id}", headers=_auth_header(key_b))
        assert resp.status_code == 403

    def test_public_visibility_requires_trusted_key(self, client):
        """Non-trusted keys cannot create public runs."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "visibility": "public"},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 403
        assert "trusted" in resp.json()["detail"].lower()

    def test_cancel_run(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"epochs": 100, "steps_per_epoch": 100},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]

        resp = client.post(f"/api/runs/{run_id}/cancel", headers=headers)
        # May succeed or fail depending on whether run already completed
        assert resp.status_code in (200, 400)

    def test_cancel_other_agents_run_denied(self, client):
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        resp = client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "params": {"epochs": 1, "steps_per_epoch": 1}},
            headers=_auth_header(key_a),
        )
        run_id = resp.json()["run_id"]

        resp = client.post(
            f"/api/runs/{run_id}/cancel", headers=_auth_header(key_b)
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Post/feed endpoint tests
# ---------------------------------------------------------------------------


class TestPostEndpoints:
    """Tests for POST /api/posts, GET /api/posts."""

    def _wait_for_run(self, client, run_id: str, api_key: str) -> dict:
        """Wait for a run to complete and return its data."""
        headers = _auth_header(api_key)
        for _ in range(60):
            resp = client.get(f"/api/runs/{run_id}", headers=headers)
            data = resp.json()
            if data["status"] in ("completed", "failed"):
                return data
            time.sleep(0.5)
        pytest.fail("Run did not complete within timeout")

    def test_create_post_for_completed_run(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        # Run a scenario
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        self._wait_for_run(client, run_id, api_key)

        # Post a card
        resp = client.post(
            "/api/posts",
            json={
                "run_id": run_id,
                "title": "Baseline v1",
                "blurb": "First baseline run",
                "key_metrics": {"toxicity": 0.12, "welfare": 8.5},
                "tags": ["baseline", "v1"],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Baseline v1"
        assert data["tags"] == ["baseline", "v1"]
        assert "post_id" in data
        assert data["run_url"].endswith(f"/api/runs/{run_id}")

    def test_cannot_post_for_others_run(self, client):
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"epochs": 1, "steps_per_epoch": 1},
            },
            headers=_auth_header(key_a),
        )
        run_id = resp.json()["run_id"]
        self._wait_for_run(client, run_id, key_a)

        # Agent B tries to post a card for Agent A's run
        resp = client.post(
            "/api/posts",
            json={
                "run_id": run_id,
                "title": "Stolen card",
                "blurb": "Should fail",
            },
            headers=_auth_header(key_b),
        )
        assert resp.status_code == 403

    def test_feed_is_public(self, client):
        """The feed endpoint is publicly readable (no auth needed for GET)."""
        # The feed router doesn't require auth for GET
        resp = client.get("/api/posts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_feed_filter_by_tag(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        self._wait_for_run(client, run_id, api_key)

        client.post(
            "/api/posts",
            json={
                "run_id": run_id,
                "title": "Tagged post",
                "blurb": "Has a special tag",
                "tags": ["special_tag_xyz"],
            },
            headers=headers,
        )

        # Filter by tag
        resp = client.get("/api/posts?tag=special_tag_xyz")
        assert resp.status_code == 200
        posts = resp.json()
        assert len(posts) >= 1
        assert all("special_tag_xyz" in p["tags"] for p in posts)

    def test_get_single_post(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        self._wait_for_run(client, run_id, api_key)

        resp = client.post(
            "/api/posts",
            json={
                "run_id": run_id,
                "title": "Specific post",
                "blurb": "For get-by-id test",
            },
            headers=headers,
        )
        post_id = resp.json()["post_id"]

        resp = client.get(f"/api/posts/{post_id}")
        assert resp.status_code == 200
        assert resp.json()["post_id"] == post_id

    def test_get_post_not_found(self, client):
        resp = client.get("/api/posts/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestRunModels:
    """Tests for run Pydantic models."""

    def test_run_create_defaults(self):
        from swarm.api.models.run import RunCreate

        rc = RunCreate(scenario_id="baseline")
        assert rc.params == {}
        assert rc.visibility == RunVisibility.PRIVATE
        assert rc.callback_url is None

    def test_run_create_with_overrides(self):
        from swarm.api.models.run import RunCreate

        rc = RunCreate(
            scenario_id="baseline",
            params={"seed": 42, "epochs": 5},
            visibility=RunVisibility.PUBLIC,
            callback_url="https://example.com/hook",
        )
        assert rc.params["seed"] == 42
        assert rc.visibility == RunVisibility.PUBLIC
        assert rc.callback_url == "https://example.com/hook"


class TestPostModels:
    """Tests for post Pydantic models."""

    def test_post_create_defaults(self):
        from swarm.api.models.post import PostCreate

        pc = PostCreate(
            run_id="run-123",
            title="My result",
            blurb="Some summary",
        )
        assert pc.key_metrics == {}
        assert pc.tags == []

    def test_post_create_with_metrics(self):
        from swarm.api.models.post import PostCreate

        pc = PostCreate(
            run_id="run-123",
            title="My result",
            blurb="Some summary",
            key_metrics={"toxicity": 0.1},
            tags=["sweep"],
        )
        assert pc.key_metrics["toxicity"] == 0.1
        assert "sweep" in pc.tags

    def test_post_create_validation(self):
        from pydantic import ValidationError

        from swarm.api.models.post import PostCreate

        with pytest.raises(ValidationError):
            PostCreate(run_id="x", title="", blurb="ok")  # title min_length=1

        with pytest.raises(ValidationError):
            PostCreate(run_id="x", title="ok", blurb="x" * 2001)  # blurb max_length


class TestFeedQueryModel:
    """Tests for FeedQuery model."""

    def test_defaults(self):
        from swarm.api.models.post import FeedQuery

        fq = FeedQuery()
        assert fq.tags is None
        assert fq.limit == 20
        assert fq.offset == 0

    def test_custom(self):
        from swarm.api.models.post import FeedQuery

        fq = FeedQuery(tags=["a", "b"], limit=50, offset=10, agent_id="x")
        assert fq.limit == 50

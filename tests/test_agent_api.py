"""Tests for the SWARM Agent API (runs, posts, middleware, persistence)."""

import time

import pytest

# Skip all tests if fastapi is not installed
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from swarm.api.app import create_app  # noqa: E402
from swarm.api.config import APIConfig  # noqa: E402
from swarm.api.middleware import (  # noqa: E402
    _api_keys,
    _key_quotas,
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
    import swarm.api.routers.agents as agents_mod
    import swarm.api.routers.posts as posts_mod

    _api_keys.clear()
    _key_quotas.clear()
    _rate_limit_windows.clear()
    _trusted_keys.clear()
    agents_mod._registration_rate.clear()
    agents_mod._registered_agents.clear()
    posts_mod._public_rate.clear()
    yield
    _api_keys.clear()
    _key_quotas.clear()
    _rate_limit_windows.clear()
    _trusted_keys.clear()
    agents_mod._registration_rate.clear()
    agents_mod._registered_agents.clear()
    posts_mod._public_rate.clear()


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path):
    """Point persistence stores at a temp SQLite DB for test isolation."""
    import swarm.api.routers.posts as posts_mod
    import swarm.api.routers.runs as runs_mod
    from swarm.api.persistence import PostStore, RunStore

    db_path = tmp_path / "test_api.db"
    runs_mod._store = RunStore(db_path=db_path)
    posts_mod._post_store = PostStore(db_path=db_path)
    # Clear shutdown state from previous tests
    runs_mod._shutting_down.clear()
    runs_mod._run_cancel_events.clear()
    yield
    runs_mod._store = None
    posts_mod._post_store = None


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


def _wait_for_run(client, run_id: str, api_key: str, timeout: int = 60) -> dict:
    """Wait for a run to complete and return its data."""
    headers = _auth_header(api_key)
    for _ in range(timeout * 2):
        resp = client.get(f"/api/runs/{run_id}", headers=headers)
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.5)
    pytest.fail("Run did not complete within timeout")


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
        client.app._swarm_rate_limit = 3  # type: ignore[attr-defined]
        register_api_key("rl_key", "agent_rl")
        headers = _auth_header("rl_key")

        for _ in range(3):
            resp = client.get("/api/runs", headers=headers)
            assert resp.status_code == 200

        resp = client.get("/api/runs", headers=headers)
        assert resp.status_code == 429

    def test_registration_rate_limiting(self, client):
        """Registration endpoint has per-IP rate limiting."""
        import swarm.api.routers.agents as agents_mod

        # Reset rate state and set low limit for test
        agents_mod._registration_rate.clear()
        old_limit = agents_mod._REGISTRATION_LIMIT
        agents_mod._REGISTRATION_LIMIT = 3
        try:
            for i in range(3):
                resp = client.post(
                    "/api/v1/agents/register",
                    json={"name": f"Bot{i}", "description": "test"},
                )
                assert resp.status_code == 200

            resp = client.post(
                "/api/v1/agents/register",
                json={"name": "Bot-overflow", "description": "test"},
            )
            assert resp.status_code == 429
        finally:
            agents_mod._REGISTRATION_LIMIT = old_limit
            agents_mod._registration_rate.clear()


# ---------------------------------------------------------------------------
# Security-specific tests
# ---------------------------------------------------------------------------


class TestSecurityHardening:
    """Tests for security controls added during the security review."""

    def test_ssrf_callback_http_rejected(self, client):
        """HTTP callback URLs are rejected (must be HTTPS)."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "http://evil.com/hook",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400
        assert "HTTPS" in resp.json()["detail"]

    def test_ssrf_callback_localhost_rejected(self, client):
        """Callback URLs pointing to localhost are rejected."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "https://localhost/hook",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400
        assert "internal" in resp.json()["detail"].lower()

    def test_ssrf_callback_metadata_rejected(self, client):
        """Callback URLs to cloud metadata endpoints are rejected."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "https://169.254.169.254/latest/meta-data",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400

    def test_ssrf_callback_private_ip_rejected(self, client):
        """Callback URLs to private IPs are rejected."""
        _, api_key = _register_agent(client)
        for host in ["10.0.0.1", "192.168.1.1", "172.16.0.1"]:
            resp = client.post(
                "/api/runs",
                json={
                    "scenario_id": "baseline",
                    "callback_url": f"https://{host}/hook",
                },
                headers=_auth_header(api_key),
            )
            assert resp.status_code == 400, f"Expected 400 for {host}"

    def test_ssrf_callback_ipv6_loopback_rejected(self, client):
        """IPv6 loopback (::1) is rejected."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "https://[::1]/hook",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400

    def test_ssrf_callback_ipv6_mapped_ipv4_rejected(self, client):
        """IPv6-mapped IPv4 addresses like ::ffff:127.0.0.1 are rejected."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "https://[::ffff:127.0.0.1]/hook",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400

    def test_ssrf_callback_zero_ip_rejected(self, client):
        """0.0.0.0 is rejected."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "callback_url": "https://0.0.0.0/hook",
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 400

    def test_path_traversal_scenario_id_rejected(self, client):
        """Path traversal attempts in scenario_id are rejected."""
        _, api_key = _register_agent(client)
        for bad_id in ["../../etc/passwd", "../baseline", "foo/bar", "a;b", "a b"]:
            resp = client.post(
                "/api/runs",
                json={"scenario_id": bad_id},
                headers=_auth_header(api_key),
            )
            assert resp.status_code == 400, f"Expected 400 for scenario_id={bad_id!r}"

    def test_scenario_allowlist_no_longer_leaks_names(self, client):
        """Error message for unknown scenario does not list all allowed names."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={"scenario_id": "nonexistent"},
            headers=_auth_header(api_key),
        )
        detail = resp.json()["detail"]
        assert "Allowed:" not in detail

    def test_unknown_params_rejected(self, client):
        """Run params with unknown keys are rejected at the model level."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"evil_key": "drop table"},
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 422

    def test_epochs_out_of_range_rejected(self, client):
        """Epochs > 1000 rejected at model level."""
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"epochs": 9999},
            },
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 422

    def test_tags_too_many_rejected(self, client):
        """More than 20 tags rejected."""
        from pydantic import ValidationError

        from swarm.api.models.post import PostCreate

        with pytest.raises(ValidationError):
            PostCreate(
                run_id="x",
                title="t",
                blurb="b",
                tags=[f"tag{i}" for i in range(21)],
            )

    def test_tag_too_long_rejected(self):
        from pydantic import ValidationError

        from swarm.api.models.post import PostCreate

        with pytest.raises(ValidationError):
            PostCreate(
                run_id="x",
                title="t",
                blurb="b",
                tags=["a" * 101],
            )

    def test_key_metrics_too_large_rejected(self):
        from pydantic import ValidationError

        from swarm.api.models.post import PostCreate

        with pytest.raises(ValidationError):
            PostCreate(
                run_id="x",
                title="t",
                blurb="b",
                key_metrics={f"k{i}": i for i in range(51)},
            )

    def test_error_message_sanitized(self, client):
        """Error messages on failed runs should not leak stack traces."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        data = _wait_for_run(client, run_id, api_key)

        if data["error"] is not None:
            assert len(data["error"]) <= 300
            assert "Traceback" not in data["error"]
            assert "File " not in data["error"]


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
            json={"scenario_id": "nonexistent"},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 404

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

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        data = _wait_for_run(client, run_id, api_key)

        assert data["status"] == "completed"
        assert data["summary_metrics"] is not None
        assert data["summary_metrics"]["n_epochs_completed"] >= 1

    def test_list_runs_scoped_to_agent(self, client):
        """Each agent only sees their own runs."""
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "params": {"epochs": 1, "steps_per_epoch": 1}},
            headers=_auth_header(key_a),
        )

        resp = client.get("/api/runs", headers=_auth_header(key_b))
        assert resp.status_code == 200
        assert len(resp.json()) == 0

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
# Compare-runs endpoint tests
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    """Tests for GET /api/runs/compare."""

    def _create_completed_run(self, client, api_key: str, seed: int = 1) -> str:
        """Helper: create a run and wait for completion, return run_id."""
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": seed, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=_auth_header(api_key),
        )
        run_id = resp.json()["run_id"]
        _wait_for_run(client, run_id, api_key)
        return run_id

    def test_compare_two_runs(self, client):
        """Compare two completed runs returns metrics + deltas."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        run_a = self._create_completed_run(client, api_key, seed=1)
        run_b = self._create_completed_run(client, api_key, seed=2)

        resp = client.get(
            f"/api/runs/compare?ids={run_a},{run_b}",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["baseline_run_id"] == run_a
        assert run_a in data["runs"]
        assert run_b in data["runs"]
        # Baseline has no delta, second run does
        assert "delta_vs_baseline" not in data["runs"][run_a]
        assert "delta_vs_baseline" in data["runs"][run_b]

    def test_compare_needs_at_least_two(self, client):
        """Comparing fewer than 2 runs returns 400."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        run_a = self._create_completed_run(client, api_key)

        resp = client.get(
            f"/api/runs/compare?ids={run_a}",
            headers=headers,
        )
        assert resp.status_code == 400

    def test_compare_nonexistent_run_404(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        run_a = self._create_completed_run(client, api_key)

        resp = client.get(
            f"/api/runs/compare?ids={run_a},nonexistent-id",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_compare_other_agents_private_run_denied(self, client):
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        run_a = self._create_completed_run(client, key_a)
        run_b = self._create_completed_run(client, key_b)

        # Agent A tries to compare with Agent B's private run
        resp = client.get(
            f"/api/runs/compare?ids={run_a},{run_b}",
            headers=_auth_header(key_a),
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Artifacts endpoint tests
# ---------------------------------------------------------------------------


class TestArtifactsEndpoint:
    """Tests for GET /api/runs/:id/artifacts."""

    def test_list_artifacts_for_completed_run(self, client):
        """After a run completes, artifacts should be listable."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        _wait_for_run(client, run_id, api_key)

        resp = client.get(f"/api/runs/{run_id}/artifacts", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert isinstance(data["artifacts"], list)
        # Should have at least history.json
        if data["artifacts"]:
            assert any("history.json" in a["path"] for a in data["artifacts"])

    def test_artifacts_not_found_returns_empty(self, client):
        """If no artifacts directory exists, return empty list."""
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        # Create a run but don't wait — artifacts may not exist yet
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]

        # Immediately query artifacts (run might not have exported yet)
        resp = client.get(f"/api/runs/{run_id}/artifacts", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["artifacts"], list)

    def test_artifacts_access_denied_for_private_run(self, client):
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)

        resp = client.post(
            "/api/runs",
            json={"scenario_id": "baseline", "params": {"epochs": 1, "steps_per_epoch": 1}},
            headers=_auth_header(key_a),
        )
        run_id = resp.json()["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/artifacts", headers=_auth_header(key_b)
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Post/feed endpoint tests
# ---------------------------------------------------------------------------


class TestPostEndpoints:
    """Tests for POST /api/posts, GET /api/posts."""

    def test_create_post_for_completed_run(self, client):
        _, api_key = _register_agent(client)
        headers = _auth_header(api_key)

        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        _wait_for_run(client, run_id, api_key)

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
        _wait_for_run(client, run_id, key_a)

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
        _wait_for_run(client, run_id, api_key)

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
        _wait_for_run(client, run_id, api_key)

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
# Voting endpoint tests
# ---------------------------------------------------------------------------


class TestVotingEndpoint:
    """Tests for POST /api/posts/:id/vote."""

    def _create_post(self, client, api_key: str) -> str:
        """Helper: create a run, wait, post a card, return post_id."""
        headers = _auth_header(api_key)
        resp = client.post(
            "/api/runs",
            json={
                "scenario_id": "baseline",
                "params": {"seed": 1, "epochs": 1, "steps_per_epoch": 1},
            },
            headers=headers,
        )
        run_id = resp.json()["run_id"]
        _wait_for_run(client, run_id, api_key)

        resp = client.post(
            "/api/posts",
            json={
                "run_id": run_id,
                "title": "Vote test post",
                "blurb": "For voting tests",
            },
            headers=headers,
        )
        return resp.json()["post_id"]

    def test_upvote(self, client):
        """Upvoting a post increments its upvote count."""
        _, api_key = _register_agent(client)
        post_id = self._create_post(client, api_key)

        resp = client.post(
            f"/api/posts/{post_id}/vote",
            json={"direction": 1},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["upvotes"] == 1
        assert data["downvotes"] == 0
        assert data["your_vote"] == 1

    def test_downvote(self, client):
        """Downvoting a post increments its downvote count."""
        _, api_key = _register_agent(client)
        post_id = self._create_post(client, api_key)

        resp = client.post(
            f"/api/posts/{post_id}/vote",
            json={"direction": -1},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["upvotes"] == 0
        assert data["downvotes"] == 1
        assert data["your_vote"] == -1

    def test_toggle_vote_off(self, client):
        """Voting the same direction again removes the vote."""
        _, api_key = _register_agent(client)
        post_id = self._create_post(client, api_key)
        headers = _auth_header(api_key)

        # First upvote
        client.post(f"/api/posts/{post_id}/vote", json={"direction": 1}, headers=headers)

        # Same direction again = toggle off
        resp = client.post(f"/api/posts/{post_id}/vote", json={"direction": 1}, headers=headers)
        data = resp.json()
        assert data["upvotes"] == 0
        assert data["your_vote"] is None

    def test_switch_vote(self, client):
        """Voting in the opposite direction switches the vote."""
        _, api_key = _register_agent(client)
        post_id = self._create_post(client, api_key)
        headers = _auth_header(api_key)

        # Upvote first
        client.post(f"/api/posts/{post_id}/vote", json={"direction": 1}, headers=headers)

        # Switch to downvote
        resp = client.post(f"/api/posts/{post_id}/vote", json={"direction": -1}, headers=headers)
        data = resp.json()
        assert data["upvotes"] == 0
        assert data["downvotes"] == 1
        assert data["your_vote"] == -1

    def test_multiple_voters(self, client):
        """Multiple agents can vote on the same post."""
        _, key_a = _register_agent(client)
        _, key_b = _register_agent(client)
        post_id = self._create_post(client, key_a)

        client.post(
            f"/api/posts/{post_id}/vote",
            json={"direction": 1},
            headers=_auth_header(key_a),
        )
        resp = client.post(
            f"/api/posts/{post_id}/vote",
            json={"direction": 1},
            headers=_auth_header(key_b),
        )
        data = resp.json()
        assert data["upvotes"] == 2

    def test_vote_on_nonexistent_post(self, client):
        _, api_key = _register_agent(client)
        resp = client.post(
            "/api/posts/nonexistent-id/vote",
            json={"direction": 1},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 404

    def test_vote_requires_auth(self, client):
        resp = client.post(
            "/api/posts/some-id/vote",
            json={"direction": 1},
        )
        assert resp.status_code == 401

    def test_zero_direction_rejected(self, client):
        """direction=0 is rejected at model level (Literal[1,-1], fix 3.1)."""
        _, api_key = _register_agent(client)
        post_id = self._create_post(client, api_key)
        resp = client.post(
            f"/api/posts/{post_id}/vote",
            json={"direction": 0},
            headers=_auth_header(api_key),
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for the SQLite persistence layer."""

    def test_run_store_roundtrip(self, tmp_path):
        """Runs survive store re-creation (simulating restart)."""
        from datetime import datetime, timezone

        from swarm.api.models.run import RunResponse, RunSummaryMetrics
        from swarm.api.persistence import RunStore

        db_path = tmp_path / "roundtrip.db"
        store = RunStore(db_path=db_path)

        run = RunResponse(
            run_id="test-123",
            scenario_id="baseline",
            status=RunStatus.COMPLETED,
            visibility=RunVisibility.PRIVATE,
            agent_id="agent-1",
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            summary_metrics=RunSummaryMetrics(
                total_interactions=100,
                accepted_interactions=80,
                avg_toxicity=0.15,
                final_welfare=10.5,
                avg_payoff=1.2,
                quality_gap=0.3,
                n_agents=5,
                n_epochs_completed=10,
            ),
            status_url="http://localhost/api/runs/test-123",
        )
        store.save(run)

        # Create a new store pointing at the same DB (simulates restart)
        store2 = RunStore(db_path=db_path)
        loaded = store2.get("test-123")
        assert loaded is not None
        assert loaded.run_id == "test-123"
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.summary_metrics is not None
        assert loaded.summary_metrics.total_interactions == 100

    def test_post_store_roundtrip(self, tmp_path):
        """Posts survive store re-creation."""
        from datetime import datetime, timezone

        from swarm.api.models.post import PostResponse
        from swarm.api.persistence import PostStore

        db_path = tmp_path / "roundtrip.db"
        store = PostStore(db_path=db_path)

        post = PostResponse(
            post_id="post-abc",
            run_id="run-xyz",
            agent_id="agent-1",
            title="Test Card",
            blurb="Testing persistence",
            key_metrics={"toxicity": 0.1},
            tags=["test", "persistence"],
            published_at=datetime.now(timezone.utc),
            run_url="http://localhost/api/runs/run-xyz",
        )
        store.save(post)

        store2 = PostStore(db_path=db_path)
        loaded = store2.get("post-abc")
        assert loaded is not None
        assert loaded.title == "Test Card"
        assert loaded.tags == ["test", "persistence"]
        assert loaded.key_metrics["toxicity"] == 0.1

    def test_vote_persistence(self, tmp_path):
        """Votes survive store re-creation."""
        from datetime import datetime, timezone

        from swarm.api.models.post import PostResponse
        from swarm.api.persistence import PostStore

        db_path = tmp_path / "votes.db"
        store = PostStore(db_path=db_path)

        post = PostResponse(
            post_id="post-v",
            run_id="run-v",
            agent_id="agent-1",
            title="Vote Test",
            blurb="Test",
            published_at=datetime.now(timezone.utc),
        )
        store.save(post)
        store.vote("post-v", "voter-1", 1)
        store.vote("post-v", "voter-2", -1)

        # Re-open store
        store2 = PostStore(db_path=db_path)
        assert store2.get_vote("post-v", "voter-1") == 1
        assert store2.get_vote("post-v", "voter-2") == -1


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
            PostCreate(run_id="x", title="", blurb="ok")

        with pytest.raises(ValidationError):
            PostCreate(run_id="x", title="ok", blurb="x" * 2001)


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

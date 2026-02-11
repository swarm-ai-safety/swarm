"""Tests for the OpenClaw bridge."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from swarm.bridges.openclaw.config import ServiceConfig
from swarm.bridges.openclaw.job_queue import Job, JobQueue, JobState
from swarm.bridges.openclaw.schemas import (
    RunMetrics,
    RunRequest,
    RunResponse,
    RunStatus,
)
from swarm.bridges.openclaw.skill import OpenClawSkill

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_run_request(self):
        req = RunRequest(scenario="baseline", seed=42, epochs=10)
        assert req.scenario == "baseline"
        assert req.seed == 42
        assert req.epochs == 10

    def test_run_request_defaults(self):
        req = RunRequest(scenario="test")
        assert req.seed == 42
        assert req.epochs is None
        assert req.governance_overrides is None

    def test_run_response(self):
        resp = RunResponse(job_id="abc-123", status="queued")
        assert resp.job_id == "abc-123"
        assert resp.status == "queued"

    def test_run_status(self):
        status = RunStatus(
            job_id="abc-123",
            status="running",
            epochs_completed=5,
            total_epochs=10,
        )
        assert status.epochs_completed == 5
        assert status.total_epochs == 10

    def test_run_metrics(self):
        metrics = RunMetrics(
            job_id="abc-123",
            toxicity_rate=0.1,
            quality_gap=-0.05,
            total_welfare=50.0,
            interactions_count=100,
            epochs_completed=10,
        )
        assert metrics.toxicity_rate == 0.1
        assert metrics.interactions_count == 100


# ---------------------------------------------------------------------------
# Job tests
# ---------------------------------------------------------------------------


class TestJob:
    def test_initial_state(self):
        job = Job()
        assert job.state == JobState.QUEUED
        assert job.epochs_completed == 0
        assert job.error is None

    def test_state_transitions(self):
        job = Job()
        assert job.state == JobState.QUEUED

        job.state = JobState.RUNNING
        assert job.state == JobState.RUNNING

        job.state = JobState.COMPLETED
        assert job.state == JobState.COMPLETED

    def test_state_transition_to_failed(self):
        job = Job()
        job.state = JobState.RUNNING
        job.state = JobState.FAILED
        job.error = "Something went wrong"
        assert job.state == JobState.FAILED
        assert job.error == "Something went wrong"

    def test_to_status_dict(self):
        job = Job(job_id="test-job")
        data = job.to_status_dict()
        assert data["job_id"] == "test-job"
        assert data["status"] == "queued"
        assert data["started_at"] is None


# ---------------------------------------------------------------------------
# JobQueue tests
# ---------------------------------------------------------------------------


class TestJobQueue:
    def test_submit_and_get(self):
        def mock_sim(scenario, seed):
            return {"toxicity_rate": 0.1, "total_welfare": 10.0}

        queue = JobQueue(simulation_fn=mock_sim)
        job = queue.submit({"scenario": "baseline", "seed": 42, "epochs": 1})
        assert job.job_id != ""

        # Wait for completion
        time.sleep(0.5)
        retrieved = queue.get(job.job_id)
        assert retrieved is not None
        assert retrieved.state in (JobState.RUNNING, JobState.COMPLETED, JobState.FAILED)

    def test_submit_creates_job(self):
        def failing_sim(scenario, seed):
            raise RuntimeError("test")

        queue = JobQueue(simulation_fn=failing_sim)
        job = queue.submit({"scenario": "nonexistent"})
        # Job may already be RUNNING or FAILED by the time we check (race);
        # the important thing is the job was created with a valid state.
        assert job.state in (JobState.QUEUED, JobState.RUNNING, JobState.FAILED)

    def test_get_nonexistent(self):
        queue = JobQueue()
        assert queue.get("nonexistent") is None

    def test_list_jobs(self):
        def mock_sim(scenario, seed):
            return {}

        queue = JobQueue(simulation_fn=mock_sim)
        queue.submit({"scenario": "test1"})
        queue.submit({"scenario": "test2"})
        jobs = queue.list_jobs()
        assert len(jobs) == 2

    def test_load_scenario_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "test.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump({"scenario_id": "test"}, f)

            config = ServiceConfig(scenario_dir=tmpdir)
            queue = JobQueue(config=config)
            scenario = queue._load_scenario("test")
            assert scenario["scenario_id"] == "test"

    def test_load_scenario_with_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "baseline.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump({"scenario_id": "baseline", "agents": []}, f)

            config = ServiceConfig(scenario_dir=tmpdir)
            queue = JobQueue(config=config)
            scenario = queue._load_scenario("baseline")
            assert scenario["scenario_id"] == "baseline"

    def test_load_scenario_not_found(self):
        queue = JobQueue()
        with pytest.raises(FileNotFoundError):
            queue._load_scenario("definitely_not_a_real_scenario_xyz")

    def test_load_scenario_rejects_path_separators(self):
        queue = JobQueue()
        with pytest.raises(ValueError, match="path separators"):
            queue._load_scenario("../../etc/passwd")

    def test_load_scenario_rejects_absolute_path(self):
        queue = JobQueue()
        with pytest.raises(ValueError, match="path separators"):
            queue._load_scenario("/etc/passwd")

    def test_load_scenario_rejects_backslash(self):
        queue = JobQueue()
        with pytest.raises(ValueError, match="path separators"):
            queue._load_scenario("..\\..\\etc\\passwd")

    def test_job_completes_successfully(self):
        def mock_sim(scenario, seed):
            return {
                "toxicity_rate": 0.05,
                "quality_gap": 0.1,
                "total_welfare": 20.0,
                "interactions_count": 50,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "test.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump({"scenario_id": "test", "simulation": {"n_epochs": 1}}, f)

            config = ServiceConfig(scenario_dir=tmpdir)
            queue = JobQueue(config=config, simulation_fn=mock_sim)
            job = queue.submit({"scenario": "test", "seed": 42})

            # Wait for completion
            for _ in range(20):
                time.sleep(0.1)
                current = queue.get(job.job_id)
                if current and current.state == JobState.COMPLETED:
                    break

            current = queue.get(job.job_id)
            assert current is not None
            assert current.state == JobState.COMPLETED
            assert current.metrics["toxicity_rate"] == 0.05

    def test_job_fails_no_simulation_fn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "test.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump({"scenario_id": "test"}, f)

            config = ServiceConfig(scenario_dir=tmpdir)
            queue = JobQueue(config=config, simulation_fn=None)
            job = queue.submit({"scenario": "test"})

            # Wait for failure
            for _ in range(20):
                time.sleep(0.1)
                current = queue.get(job.job_id)
                if current and current.state == JobState.FAILED:
                    break

            current = queue.get(job.job_id)
            assert current is not None
            assert current.state == JobState.FAILED
            assert "No simulation function" in (current.error or "")


# ---------------------------------------------------------------------------
# OpenClawSkill tests
# ---------------------------------------------------------------------------


class TestOpenClawSkill:
    def test_construction(self):
        skill = OpenClawSkill(base_url="http://example.com:9000", timeout=60.0)
        assert skill._base_url == "http://example.com:9000"
        assert skill._timeout == 60.0

    def test_health_check_failure(self):
        """Health check returns False when service is not running."""
        skill = OpenClawSkill(base_url="http://localhost:19999")
        assert skill.health_check() is False

    @patch("swarm.bridges.openclaw.skill.urllib.request.urlopen")
    def test_health_check_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ok"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        skill = OpenClawSkill()
        assert skill.health_check() is True

    @patch("swarm.bridges.openclaw.skill.urllib.request.urlopen")
    def test_get_status(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "job_id": "abc",
            "status": "running",
            "epochs_completed": 5,
            "total_epochs": 10,
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        skill = OpenClawSkill()
        status = skill.get_status("abc")
        assert status["job_id"] == "abc"
        assert status["status"] == "running"

    @patch("swarm.bridges.openclaw.skill.urllib.request.urlopen")
    def test_get_metrics(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "job_id": "abc",
            "toxicity_rate": 0.1,
            "quality_gap": -0.05,
            "total_welfare": 50.0,
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        skill = OpenClawSkill()
        metrics = skill.get_metrics("abc")
        assert metrics["toxicity_rate"] == 0.1

    @patch("swarm.bridges.openclaw.skill.urllib.request.urlopen")
    def test_run_scenario_no_wait(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "job_id": "new-job",
            "status": "queued",
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        skill = OpenClawSkill()
        result = skill.run_scenario("baseline", wait=False)
        assert result["job_id"] == "new-job"


# ---------------------------------------------------------------------------
# Service integration tests (skip if no FastAPI)
# ---------------------------------------------------------------------------


try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed")
class TestOpenClawService:
    def _make_app(self, simulation_fn=None):
        from swarm.bridges.openclaw.service import create_app

        return create_app(simulation_fn=simulation_fn)

    def _make_app_with_config(self, simulation_fn=None, config=None):
        from swarm.bridges.openclaw.service import create_app

        return create_app(config=config, simulation_fn=simulation_fn)

    def test_health_endpoint(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_submit_run(self):
        def mock_sim(scenario, seed):
            return {"toxicity_rate": 0.1}

        app = self._make_app(simulation_fn=mock_sim)
        client = TestClient(app)
        resp = client.post("/runs", json={"scenario": "baseline", "seed": 42})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        # Job may already be running by the time the response is built
        assert data["status"] in ("queued", "running", "completed")

    def test_get_run_status(self):
        def mock_sim(scenario, seed):
            return {"toxicity_rate": 0.1}

        app = self._make_app(simulation_fn=mock_sim)
        client = TestClient(app)

        # Submit a job
        resp = client.post("/runs", json={"scenario": "baseline"})
        job_id = resp.json()["job_id"]

        # Get status
        resp = client.get(f"/runs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_get_run_status_not_found(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_metrics_not_completed(self):
        def slow_sim(scenario, seed):
            time.sleep(10)
            return {}

        app = self._make_app(simulation_fn=slow_sim)
        client = TestClient(app)

        resp = client.post("/runs", json={"scenario": "baseline"})
        job_id = resp.json()["job_id"]

        # Immediately try to get metrics (job not done yet)
        resp = client.get(f"/runs/{job_id}/metrics")
        assert resp.status_code == 404

    def test_get_metrics_completed(self):
        def instant_sim(scenario, seed):
            return {
                "toxicity_rate": 0.05,
                "quality_gap": 0.1,
                "total_welfare": 20.0,
                "interactions_count": 50,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "test.yaml"
            with open(scenario_path, "w") as f:
                yaml.dump({"scenario_id": "test", "simulation": {"n_epochs": 1}}, f)

            app = self._make_app_with_config(
                simulation_fn=instant_sim,
                config=ServiceConfig(scenario_dir=tmpdir),
            )
            client = TestClient(app)

            resp = client.post("/runs", json={"scenario": "test", "seed": 1})
            job_id = resp.json()["job_id"]

            # Wait for completion
            for _ in range(30):
                time.sleep(0.1)
                status_resp = client.get(f"/runs/{job_id}")
                if status_resp.json().get("status") == "completed":
                    break

            resp = client.get(f"/runs/{job_id}/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert data["toxicity_rate"] == 0.05

    def test_get_metrics_not_found(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/runs/nonexistent/metrics")
        assert resp.status_code == 404

    def test_submit_rejects_path_traversal(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post("/runs", json={"scenario": "../../etc/passwd"})
        job_id = resp.json()["job_id"]

        # Wait for job to fail
        for _ in range(20):
            time.sleep(0.1)
            status_resp = client.get(f"/runs/{job_id}")
            if status_resp.json().get("status") == "failed":
                break

        status = client.get(f"/runs/{job_id}").json()
        assert status["status"] == "failed"
        # Error should not leak internal paths
        assert "/etc/passwd" not in (status.get("error") or "")
        assert "path separators" in (status.get("error") or "").lower() or \
               "Invalid request" in (status.get("error") or "")

    def test_submit_rejects_absolute_path(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post("/runs", json={"scenario": "/etc/shadow"})
        job_id = resp.json()["job_id"]

        for _ in range(20):
            time.sleep(0.1)
            status_resp = client.get(f"/runs/{job_id}")
            if status_resp.json().get("status") == "failed":
                break

        status = client.get(f"/runs/{job_id}").json()
        assert status["status"] == "failed"
        assert "/etc/shadow" not in (status.get("error") or "")

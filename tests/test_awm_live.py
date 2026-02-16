"""Integration tests for AWM live-mode servers.

These tests spawn real AWM FastAPI server processes, make HTTP tool calls,
and validate database state. They require:
  - httpx installed (pip install swarm-safety[awm])
  - AWM environments downloaded (bash scripts/download_awm_envs.sh)

All tests are marked @pytest.mark.slow and auto-skip when prerequisites
are missing.
"""

import asyncio
from pathlib import Path

import pytest

# ── Module-level skip if prerequisites are missing ──────────────────────
try:
    import httpx  # noqa: F401

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_ENVS_PATH = Path("external/awm-envs")

if not _HTTPX_AVAILABLE:
    pytest.skip(
        "httpx not installed (pip install swarm-safety[awm])",
        allow_module_level=True,
    )

if not _ENVS_PATH.exists():
    pytest.skip(
        "AWM environments not downloaded (bash scripts/download_awm_envs.sh)",
        allow_module_level=True,
    )

from swarm.bridges.awm.config import AWMConfig  # noqa: E402
from swarm.bridges.awm.mcp_client import AWMMCPSyncClient  # noqa: E402
from swarm.bridges.awm.server_manager import AWMServerManager  # noqa: E402
from swarm.core.awm_handler import AWMHandler  # noqa: E402
from swarm.logging.event_bus import EventBus  # noqa: E402
from swarm.scenarios import build_orchestrator, load_scenario  # noqa: E402

pytestmark = pytest.mark.slow


# ── Helpers ─────────────────────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine from sync test code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _make_config(base_port: int) -> AWMConfig:
    """Create an AWMConfig for live testing with a given base port."""
    return AWMConfig(
        live_mode=True,
        envs_path=_ENVS_PATH,
        base_port=base_port,
        max_concurrent_servers=4,
        server_startup_timeout=30.0,
        health_check_interval=0.5,
        max_steps_per_task=15,
        max_tasks_per_epoch=3,
        step_mode=True,
    )


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def awm_config_19100():
    return _make_config(19100)


@pytest.fixture
def awm_config_19200():
    return _make_config(19200)


@pytest.fixture
def awm_config_19300():
    return _make_config(19300)


@pytest.fixture
def server_manager(awm_config_19100):
    mgr = AWMServerManager(awm_config_19100)
    yield mgr
    _run_async(mgr.shutdown())


@pytest.fixture
def live_handler(awm_config_19300):
    handler = AWMHandler(
        config=awm_config_19300,
        event_bus=EventBus(),
        seed=42,
    )
    yield handler
    # Shut down servers and close clients
    if handler._server_manager is not None:
        _run_async(handler._server_manager.shutdown())
    for client in handler._clients.values():
        client.close()


@pytest.fixture
def sync_client_factory():
    """Factory fixture that tracks created clients for cleanup."""
    clients = []

    def _create(base_url: str, timeout: float = 10.0) -> AWMMCPSyncClient:
        client = AWMMCPSyncClient(base_url=base_url, timeout=timeout)
        clients.append(client)
        return client

    yield _create

    for c in clients:
        c.close()


# ── TestAWMServerLifecycle (ports 19100-19199) ──────────────────────────


class TestAWMServerLifecycle:
    """Tests for starting, stopping, and managing AWM server processes."""

    def test_start_server_live(self, server_manager):
        """Start a real server and verify it reports running."""
        server = _run_async(server_manager.start_server("agent_1"))
        assert server is not None
        assert server.running is True
        assert server.port == 19100

    def test_server_port_allocation(self, server_manager):
        """Start two servers and verify they get different ports."""
        s1 = _run_async(server_manager.start_server("agent_1"))
        s2 = _run_async(server_manager.start_server("agent_2"))
        assert s1 is not None
        assert s2 is not None
        assert s1.port != s2.port
        assert server_manager.active_count == 2

    def test_server_stop_terminates(self, server_manager):
        """Start then stop a server, verify it reports not running."""
        server = _run_async(server_manager.start_server("agent_1"))
        assert server is not None
        assert server.running is True

        _run_async(server.stop())
        assert server.running is False

    def test_server_health_check(self, server_manager, sync_client_factory):
        """Use AWMMCPSyncClient.health_check() against a live server."""
        server = _run_async(server_manager.start_server("agent_1"))
        assert server is not None

        client = sync_client_factory(server.base_url)
        assert client.health_check() is True


# ── TestAWMMCPClientLive (ports 19200-19299) ────────────────────────────


class TestAWMMCPClientLive:
    """Tests for the sync HTTP client against live AWM servers."""

    @pytest.fixture(autouse=True)
    def _setup_server(self, awm_config_19200, sync_client_factory):
        self._mgr = AWMServerManager(awm_config_19200)
        server = _run_async(self._mgr.start_server("client_test"))
        assert server is not None
        self._client = sync_client_factory(server.base_url)
        yield
        _run_async(self._mgr.shutdown())

    def test_list_tools(self):
        """GET /tools returns a non-empty list."""
        tools = self._client.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_call_tool_valid(self):
        """POST /tools/call with a valid tool succeeds."""
        tools = self._client.list_tools()
        assert len(tools) > 0
        tool_name = tools[0]["name"]
        record = self._client.call_tool(tool_name, {})
        # The call should complete without raising; success depends on args
        assert record.tool_name == tool_name

    def test_call_tool_invalid(self):
        """POST /tools/call with an unknown tool returns an error."""
        record = self._client.call_tool("nonexistent_tool_xyz", {})
        assert record.is_error_response or record.is_malformed

    def test_verify_initial_state(self):
        """POST /verify returns a dict with a 'passed' key."""
        result = self._client.verify()
        assert isinstance(result, dict)
        assert "passed" in result

    def test_reset_environment(self):
        """POST /reset returns True (200 status)."""
        ok = self._client.reset_environment()
        assert ok is True


# ── TestAWMHandlerLiveIntegration (ports 19300-19399) ───────────────────


class TestAWMHandlerLiveIntegration:
    """Tests for AWMHandler interacting with live servers."""

    def _make_state(self, agent_ids):
        """Build a minimal EnvState for handler methods."""
        from swarm.env.state import EnvState

        state = EnvState()
        for aid in agent_ids:
            state.register_agent(aid)
        return state

    def test_handler_live_batch_mode(self, live_handler):
        """epoch_start + AWM_EXECUTE_TASK produces observables."""
        from swarm.agents.base import Action, ActionType

        state = self._make_state(["agent_1"])
        live_handler.on_epoch_start(state)

        # Agent should have an assignment
        assert "agent_1" in live_handler._assignments

        action = Action(
            agent_id="agent_1",
            action_type=ActionType.AWM_EXECUTE_TASK,
            metadata={
                "tool_calls": [
                    {"tool_name": "query_database", "arguments": {}},
                    {"tool_name": "list_tables", "arguments": {}},
                ],
            },
        )
        result = live_handler.handle_action(action, state)
        assert result.success is True
        assert result.observables is not None

    def test_handler_live_step_mode(self, live_handler):
        """AWM_TOOL_CALL + AWM_FINISH_TASK multi-turn flow works."""
        from swarm.agents.base import Action, ActionType

        state = self._make_state(["agent_1"])
        live_handler.on_epoch_start(state)

        # Step 1: tool call
        tc_action = Action(
            agent_id="agent_1",
            action_type=ActionType.AWM_TOOL_CALL,
            metadata={"tool_name": "query_database", "arguments": {}},
        )
        tc_result = live_handler.handle_action(tc_action, state)
        assert tc_result.success is True
        assert tc_result.observables is None  # episode continues

        # Step 2: finish
        finish_action = Action(
            agent_id="agent_1",
            action_type=ActionType.AWM_FINISH_TASK,
            metadata={},
        )
        finish_result = live_handler.handle_action(finish_action, state)
        assert finish_result.success is True
        assert finish_result.observables is not None

    def test_handler_live_observation_fields(self, live_handler):
        """build_observation_fields returns task + multi-turn state."""
        state = self._make_state(["agent_1"])
        live_handler.on_epoch_start(state)

        fields = live_handler.build_observation_fields("agent_1", state)
        assert "awm_task" in fields
        assert "awm_episode_active" in fields
        assert "awm_steps_remaining" in fields

    def test_handler_epoch_end_cleanup(self, live_handler):
        """epoch_end clears traces and assignments."""
        state = self._make_state(["agent_1"])
        live_handler.on_epoch_start(state)

        assert len(live_handler._assignments) > 0
        assert len(live_handler._traces) > 0

        live_handler.on_epoch_end(state)

        assert len(live_handler._assignments) == 0
        assert len(live_handler._traces) == 0
        assert len(live_handler._last_results) == 0


# ── TestAWMOrchestratorLive (ports 19400-19499) ────────────────────────


class TestAWMOrchestratorLive:
    """End-to-end orchestrator tests with live AWM servers."""

    @pytest.fixture
    def live_orchestrator(self):
        """Load awm_live.yaml and override port to avoid conflicts."""
        scenario_path = Path("scenarios/awm_live.yaml")
        if not scenario_path.exists():
            pytest.skip("scenarios/awm_live.yaml not found")

        scenario = load_scenario(scenario_path)
        # Override port range to avoid conflicts with other test classes
        awm_cfg = scenario.orchestrator_config.awm_config
        awm_cfg.base_port = 19400

        orch = build_orchestrator(scenario)
        yield orch

        # Cleanup: shut down servers
        handler = getattr(orch, "_awm_handler", None)
        if (
            handler is not None
            and hasattr(handler, "_server_manager")
            and handler._server_manager is not None
        ):
            _run_async(handler._server_manager.shutdown())
        if handler is not None:
            for client in handler._clients.values():
                client.close()

    def test_orchestrator_runs_to_completion(self, live_orchestrator):
        """Load awm_live.yaml, build, run, verify metrics returned."""
        metrics = live_orchestrator.run()
        assert len(metrics) == 2  # 2 epochs
        for m in metrics:
            assert m.total_interactions >= 0

    def test_orchestrator_episodes_produced(self, live_orchestrator):
        """handler.get_completed_episodes() is non-empty after run."""
        live_orchestrator.run()
        handler = live_orchestrator._awm_handler
        assert handler is not None
        episodes = handler.get_completed_episodes()
        assert len(episodes) > 0

    def test_orchestrator_agent_states_valid(self, live_orchestrator):
        """All agents have valid reputation/resources after run."""
        live_orchestrator.run()
        for agent in live_orchestrator.get_all_agents():
            state = live_orchestrator.state.get_agent(agent.agent_id)
            assert 0.0 <= state.reputation <= 1.0
            assert state.resources >= 0.0

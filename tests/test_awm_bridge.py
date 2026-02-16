"""Unit tests for the AWM (Agent World Model) bridge.

All AWM external dependencies are mocked — these tests run on
Python 3.10+ without AWM installed.
"""

import pytest

from swarm.agents.awm_agent import AWMAgent
from swarm.agents.base import ActionType, Observation
from swarm.bridges.awm.config import AWMConfig
from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord
from swarm.bridges.awm.observable_mapper import AWMObservableMapper
from swarm.bridges.awm.verifier_bridge import AWMVerifierBridge, binary_to_soft_p
from swarm.core.proxy import ProxyObservables
from swarm.models.agent import AgentType
from swarm.models.events import EventType

# =========================================================================
# AWMConfig
# =========================================================================


class TestAWMConfig:
    """Test AWMConfig pydantic model."""

    def test_default_config(self):
        config = AWMConfig()
        assert config.enabled is True
        assert config.base_port == 9100
        assert config.max_steps_per_task == 20
        assert config.verification_confidence == 0.8

    def test_custom_config(self):
        config = AWMConfig(
            environment_id="project_mgmt_042",
            base_port=9200,
            max_steps_per_task=50,
            seed=42,
        )
        assert config.environment_id == "project_mgmt_042"
        assert config.base_port == 9200
        assert config.max_steps_per_task == 50
        assert config.seed == 42

    def test_disabled_config(self):
        config = AWMConfig(enabled=False)
        assert config.enabled is False


# =========================================================================
# ToolCallRecord and AWMEpisodeTrace
# =========================================================================


class TestToolCallRecord:
    """Test ToolCallRecord dataclass."""

    def test_default_record(self):
        record = ToolCallRecord()
        assert record.success is False
        assert record.is_malformed is False
        assert record.is_error_response is False
        assert record.tool_name == ""

    def test_successful_record(self):
        record = ToolCallRecord(
            tool_name="query_database",
            arguments={"query": "SELECT 1"},
            success=True,
            result={"rows": [1]},
        )
        assert record.success is True
        assert record.tool_name == "query_database"

    def test_malformed_record(self):
        record = ToolCallRecord(
            tool_name="bad_tool",
            is_malformed=True,
            is_error_response=True,
            error="Unknown tool",
        )
        assert record.is_malformed is True
        assert record.is_error_response is True


class TestAWMEpisodeTrace:
    """Test AWMEpisodeTrace dataclass."""

    def test_empty_trace(self):
        trace = AWMEpisodeTrace(agent_id="agent_1")
        assert trace.error_count == 0
        assert trace.malformed_count == 0
        assert trace.total_calls == 0
        assert trace.verified is None

    def test_trace_with_calls(self):
        trace = AWMEpisodeTrace(
            agent_id="agent_1",
            tool_calls=[
                ToolCallRecord(success=True),
                ToolCallRecord(is_error_response=True),
                ToolCallRecord(is_malformed=True, is_error_response=True),
                ToolCallRecord(success=True),
            ],
            steps_used=4,
            max_steps=20,
        )
        assert trace.total_calls == 4
        assert trace.error_count == 2
        assert trace.malformed_count == 1

    def test_trace_verification(self):
        trace = AWMEpisodeTrace(verified=True)
        assert trace.verified is True

        trace2 = AWMEpisodeTrace(verified=False)
        assert trace2.verified is False


# =========================================================================
# binary_to_soft_p
# =========================================================================


class TestBinaryToSoftP:
    """Test binary_to_soft_p conversion."""

    def test_pass_high_confidence(self):
        p = binary_to_soft_p(passed=True, confidence=1.0)
        assert p == 0.95

    def test_fail_high_confidence(self):
        p = binary_to_soft_p(passed=False, confidence=1.0)
        assert p == 0.05

    def test_pass_default_confidence(self):
        p = binary_to_soft_p(passed=True, confidence=0.8)
        assert 0.85 < p < 0.87  # 0.5 + 0.45 * 0.8 = 0.86

    def test_fail_default_confidence(self):
        p = binary_to_soft_p(passed=False, confidence=0.8)
        assert 0.13 < p < 0.15  # 0.5 - 0.45 * 0.8 = 0.14

    def test_zero_confidence(self):
        p_pass = binary_to_soft_p(passed=True, confidence=0.0)
        p_fail = binary_to_soft_p(passed=False, confidence=0.0)
        assert p_pass == 0.5
        assert p_fail == 0.5

    def test_clamping(self):
        p_high = binary_to_soft_p(passed=True, confidence=2.0)
        p_low = binary_to_soft_p(passed=False, confidence=2.0)
        assert p_high == 0.95
        assert p_low == 0.05

    def test_negative_confidence_clamped(self):
        p = binary_to_soft_p(passed=True, confidence=-1.0)
        assert p == 0.5  # clamped to 0.0

    def test_p_always_in_bounds(self):
        """p must always be in [0.05, 0.95] — safety invariant."""
        for passed in [True, False]:
            for conf in [0.0, 0.1, 0.5, 0.8, 1.0, 1.5, -0.5]:
                p = binary_to_soft_p(passed, conf)
                assert 0.05 <= p <= 0.95, f"p={p} out of bounds for passed={passed}, conf={conf}"


# =========================================================================
# AWMVerifierBridge
# =========================================================================


class TestAWMVerifierBridge:
    """Test AWMVerifierBridge."""

    def test_verify_pass(self):
        bridge = AWMVerifierBridge(confidence=0.8)
        p = bridge.verify_and_score({"passed": True})
        assert 0.85 < p < 0.87

    def test_verify_fail(self):
        bridge = AWMVerifierBridge(confidence=0.8)
        p = bridge.verify_and_score({"passed": False})
        assert 0.13 < p < 0.15

    def test_verify_with_result_confidence(self):
        bridge = AWMVerifierBridge(confidence=0.5)
        # Result-level confidence overrides default
        p = bridge.verify_and_score({"passed": True, "confidence": 1.0})
        assert p == 0.95

    def test_verify_missing_passed_key(self):
        bridge = AWMVerifierBridge()
        p = bridge.verify_and_score({})
        # Missing 'passed' defaults to False
        assert p < 0.5


# =========================================================================
# AWMObservableMapper
# =========================================================================


class TestAWMObservableMapper:
    """Test AWMObservableMapper."""

    def test_successful_trace(self):
        mapper = AWMObservableMapper()
        trace = AWMEpisodeTrace(
            tool_calls=[
                ToolCallRecord(success=True),
                ToolCallRecord(success=True),
                ToolCallRecord(success=True),
            ],
            verified=True,
            steps_used=3,
            max_steps=20,
        )
        obs = mapper.map(trace)
        assert obs.task_progress_delta == 1.0
        assert obs.rework_count == 0
        assert obs.verifier_rejections == 0
        assert obs.tool_misuse_flags == 0
        assert obs.counterparty_engagement_delta > 0  # good efficiency

    def test_failed_trace(self):
        mapper = AWMObservableMapper()
        trace = AWMEpisodeTrace(
            tool_calls=[
                ToolCallRecord(is_error_response=True),
                ToolCallRecord(is_malformed=True),
            ],
            verified=False,
            steps_used=2,
            max_steps=20,
        )
        obs = mapper.map(trace)
        assert obs.task_progress_delta == -0.5
        assert obs.rework_count >= 1
        assert obs.verifier_rejections == 1
        assert obs.tool_misuse_flags >= 1

    def test_unverified_trace(self):
        mapper = AWMObservableMapper()
        trace = AWMEpisodeTrace(verified=None, steps_used=0, max_steps=20)
        obs = mapper.map(trace)
        assert obs.task_progress_delta == 0.0  # neutral
        assert obs.verifier_rejections == 1  # not verified counts as rejection

    def test_efficiency_calculation(self):
        mapper = AWMObservableMapper()
        # Perfect efficiency: 0 steps used out of 20
        trace = AWMEpisodeTrace(verified=True, steps_used=0, max_steps=20)
        obs = mapper.map(trace)
        assert obs.counterparty_engagement_delta == 1.0

        # Half efficiency
        trace2 = AWMEpisodeTrace(verified=True, steps_used=10, max_steps=20)
        obs2 = mapper.map(trace2)
        assert abs(obs2.counterparty_engagement_delta - 0.5) < 0.01

        # Full steps used
        trace3 = AWMEpisodeTrace(verified=True, steps_used=20, max_steps=20)
        obs3 = mapper.map(trace3)
        assert obs3.counterparty_engagement_delta == 0.0

    def test_zero_max_steps(self):
        mapper = AWMObservableMapper()
        trace = AWMEpisodeTrace(verified=True, steps_used=5, max_steps=0)
        obs = mapper.map(trace)
        assert obs.counterparty_engagement_delta == 0.0

    def test_returns_proxy_observables(self):
        mapper = AWMObservableMapper()
        trace = AWMEpisodeTrace(verified=True, steps_used=1, max_steps=10)
        obs = mapper.map(trace)
        assert isinstance(obs, ProxyObservables)


# =========================================================================
# AWMAgent
# =========================================================================


class TestAWMAgent:
    """Test AWMAgent scripted behavior."""

    def test_agent_creation(self):
        agent = AWMAgent(agent_id="awm_1", name="test_awm")
        assert agent.agent_id == "awm_1"
        assert agent.agent_type == AgentType.HONEST
        assert agent.mode == "diligent"

    def test_agent_noop_without_task(self):
        agent = AWMAgent(agent_id="awm_1")
        obs = Observation()
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_agent_executes_awm_task(self):
        agent = AWMAgent(
            agent_id="awm_1",
            config={"tool_call_count": 3},
        )
        obs = Observation(
            awm_task={"task_id": "t1", "description": "Complete the task"},
            awm_available_tools=[
                {"name": "query_database", "description": "Run SQL"},
                {"name": "update_record", "description": "Update a row"},
            ],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.AWM_EXECUTE_TASK
        assert "tool_calls" in action.metadata
        assert len(action.metadata["tool_calls"]) == 3

    def test_lazy_agent_fewer_calls(self):
        agent = AWMAgent(
            agent_id="awm_lazy",
            config={"mode": "lazy", "tool_call_count": 9},
        )
        obs = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "Run SQL"}],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.AWM_EXECUTE_TASK
        # Lazy = max(1, 9//3) = 3
        assert len(action.metadata["tool_calls"]) == 3

    def test_adversarial_agent_malformed_calls(self):
        import random

        rng = random.Random(42)
        agent = AWMAgent(
            agent_id="awm_adv",
            config={"mode": "adversarial", "malformed_rate": 1.0, "tool_call_count": 5},
            rng=rng,
        )
        obs = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "Run SQL"}],
        )
        action = agent.act(obs)
        # All calls should be malformed (rate=1.0)
        for tc in action.metadata["tool_calls"]:
            assert tc["tool_name"] == "nonexistent_tool_xyz"

    def test_agent_no_tools_available(self):
        agent = AWMAgent(agent_id="awm_1")
        obs = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.AWM_EXECUTE_TASK
        assert action.metadata["tool_calls"] == []


# =========================================================================
# AWMHandler
# =========================================================================


class TestAWMHandler:
    """Test AWMHandler with mocked dependencies."""

    def _make_handler(self, seed=42):
        from swarm.core.awm_handler import AWMHandler
        from swarm.logging.event_bus import EventBus

        config = AWMConfig(seed=seed, max_tasks_per_epoch=2)
        bus = EventBus()
        return AWMHandler(config=config, event_bus=bus, seed=seed)

    def _make_state(self, agent_ids=None):
        """Create a minimal EnvState with agents."""
        from swarm.env.state import EnvState
        from swarm.models.agent import AgentType

        state = EnvState()
        for aid in (agent_ids or ["agent_1", "agent_2"]):
            state.add_agent(aid, name=aid, agent_type=AgentType.HONEST)
        return state

    def test_handler_creation(self):
        handler = self._make_handler()
        assert ActionType.AWM_EXECUTE_TASK in handler.handled_action_types()

    def test_epoch_creates_assignments(self):
        handler = self._make_handler()
        state = self._make_state()
        handler.on_epoch_start(state)
        assert len(handler._assignments) > 0

    def test_handle_action_no_assignment(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        action = Action(
            action_type=ActionType.AWM_EXECUTE_TASK,
            agent_id="unknown_agent",
            metadata={"tool_calls": []},
        )
        state = self._make_state()
        result = handler.handle_action(action, state)
        assert result.success is False

    def test_handle_action_with_assignment(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        # Agent should have an assignment
        assert "agent_1" in handler._assignments

        action = Action(
            action_type=ActionType.AWM_EXECUTE_TASK,
            agent_id="agent_1",
            metadata={
                "tool_calls": [
                    {"tool_name": "query_database", "arguments": {"query": "SELECT 1"}},
                    {"tool_name": "update_record", "arguments": {"table": "t", "id": 1}},
                ]
            },
        )
        result = handler.handle_action(action, state)
        assert result.success is True
        assert result.observables is not None
        assert "episode_id" in result.metadata
        assert "verified" in result.metadata

    def test_epoch_end_cleans_up(self):
        handler = self._make_handler()
        state = self._make_state()
        handler.on_epoch_start(state)
        assert len(handler._assignments) > 0
        handler.on_epoch_end(state)
        assert len(handler._assignments) == 0

    def test_observation_fields(self):
        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)
        fields = handler.build_observation_fields("agent_1", state)
        assert "awm_task" in fields
        assert "awm_available_tools" in fields

    def test_observation_fields_no_assignment(self):
        handler = self._make_handler()
        state = self._make_state()
        fields = handler.build_observation_fields("unknown", state)
        assert fields == {}

    def test_malformed_tool_rejected(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.AWM_EXECUTE_TASK,
            agent_id="agent_1",
            metadata={
                "tool_calls": [
                    {"tool_name": "nonexistent_tool", "arguments": {}},
                ]
            },
        )
        result = handler.handle_action(action, state)
        assert result.success is True  # Action processed, but tool call failed
        # The observables should reflect the malformed call
        assert result.observables.tool_misuse_flags >= 1


# =========================================================================
# AWM EventType registration
# =========================================================================


class TestAWMEventTypes:
    """Verify AWM event types are registered."""

    def test_awm_task_assigned(self):
        assert EventType.AWM_TASK_ASSIGNED.value == "awm_task_assigned"

    def test_awm_task_completed(self):
        assert EventType.AWM_TASK_COMPLETED.value == "awm_task_completed"


# =========================================================================
# AWM ActionType registration
# =========================================================================


class TestAWMActionType:
    """Verify AWM_EXECUTE_TASK action type is registered."""

    def test_awm_execute_task(self):
        assert ActionType.AWM_EXECUTE_TASK.value == "awm_execute_task"


# =========================================================================
# Import guard
# =========================================================================


class TestImportGuard:
    """Test that AWM_AVAILABLE flag works."""

    def test_awm_available_is_bool(self):
        from swarm.bridges.awm import AWM_AVAILABLE
        assert isinstance(AWM_AVAILABLE, bool)


# =========================================================================
# AWMServerManager (simulated)
# =========================================================================


class TestAWMServerManager:
    """Test AWMServerManager (Phase 1 simulation mode)."""

    @pytest.mark.asyncio
    async def test_start_server(self):
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19100)
        mgr = AWMServerManager(config)
        server = await mgr.start_server("agent_1")
        assert server is not None
        assert server.running is True
        assert server.port == 19100

    @pytest.mark.asyncio
    async def test_max_concurrent_servers(self):
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19100, max_concurrent_servers=2)
        mgr = AWMServerManager(config)
        await mgr.start_server("agent_1")
        await mgr.start_server("agent_2")
        server3 = await mgr.start_server("agent_3")
        assert server3 is None  # Exceeded limit

    @pytest.mark.asyncio
    async def test_shutdown(self):
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19100)
        mgr = AWMServerManager(config)
        await mgr.start_server("agent_1")
        assert mgr.active_count == 1
        await mgr.shutdown()
        assert mgr.active_count == 0

    @pytest.mark.asyncio
    async def test_reset_all(self):
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19100)
        mgr = AWMServerManager(config)
        await mgr.start_server("agent_1")
        # Should not raise
        await mgr.reset_all()


# =========================================================================
# Scenario loading integration
# =========================================================================


class TestScenarioLoading:
    """Test that AWM scenario YAML parses correctly."""

    def test_parse_awm_config(self):
        from swarm.scenarios.loader import parse_awm_config

        data = {
            "enabled": True,
            "environment_id": "ecommerce_001",
            "max_steps_per_task": 15,
            "seed": 42,
        }
        config = parse_awm_config(data)
        assert config is not None
        assert config.environment_id == "ecommerce_001"
        assert config.max_steps_per_task == 15

    def test_parse_awm_config_disabled(self):
        from swarm.scenarios.loader import parse_awm_config

        config = parse_awm_config({"enabled": False})
        assert config is None

    def test_parse_awm_config_empty(self):
        from swarm.scenarios.loader import parse_awm_config

        config = parse_awm_config({})
        assert config is None

    def test_awm_agent_in_agent_types(self):
        from swarm.scenarios.loader import AGENT_TYPES

        assert "awm_agent" in AGENT_TYPES

    def test_load_awm_demo_scenario(self):
        from pathlib import Path

        from swarm.scenarios.loader import load_scenario

        scenario_path = Path(__file__).parent.parent / "scenarios" / "awm_demo.yaml"
        if not scenario_path.exists():
            pytest.skip("awm_demo.yaml not found")

        scenario = load_scenario(scenario_path)
        assert scenario.scenario_id == "awm_demo"
        assert scenario.orchestrator_config.awm_config is not None
        assert scenario.orchestrator_config.awm_config.environment_id == "ecommerce_001"


# =========================================================================
# BaseAgent helper
# =========================================================================


class TestBaseAgentAWMHelper:
    """Test the create_awm_execute_task_action helper."""

    def test_create_action(self):
        agent = AWMAgent(agent_id="awm_1")
        tool_calls = [
            {"tool_name": "query_database", "arguments": {"query": "SELECT 1"}},
        ]
        action = agent.create_awm_execute_task_action(tool_calls=tool_calls)
        assert action.action_type == ActionType.AWM_EXECUTE_TASK
        assert action.agent_id == "awm_1"
        assert action.metadata["tool_calls"] == tool_calls

    def test_create_action_empty(self):
        agent = AWMAgent(agent_id="awm_1")
        action = agent.create_awm_execute_task_action()
        assert action.metadata["tool_calls"] == []


# =========================================================================
# AWMMCPSyncClient (Phase 2)
# =========================================================================


class TestAWMMCPSyncClient:
    """Test the synchronous MCP client for live mode."""

    def test_list_tools_success(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"tools": [{"name": "query_database"}]}
            def raise_for_status(self):
                pass

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def get(self, url, **kwargs):
                return FakeResponse()
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient(base_url="http://127.0.0.1:9100")
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "query_database"
        client.close()

    def test_call_tool_success(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeResponse:
            status_code = 200
            text = ""
            def json(self):
                return {"result": {"rows": [1]}}

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def post(self, url, **kwargs):
                return FakeResponse()
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient()
        record = client.call_tool("query_database", {"query": "SELECT 1"})
        assert record.success is True
        assert record.result == {"rows": [1]}
        client.close()

    def test_call_tool_malformed_422(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeResponse:
            status_code = 422
            text = "Validation error"

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def post(self, url, **kwargs):
                return FakeResponse()
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient()
        record = client.call_tool("bad_tool", {})
        assert record.is_malformed is True
        assert record.is_error_response is True
        client.close()

    def test_call_tool_connection_error(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def post(self, url, **kwargs):
                raise ConnectionError("refused")
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient()
        record = client.call_tool("query_database", {})
        assert record.is_error_response is True
        assert "refused" in record.error
        client.close()

    def test_verify_success(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"passed": True, "confidence": 0.9}

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def post(self, url, **kwargs):
                return FakeResponse()
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient()
        result = client.verify()
        assert result["passed"] is True
        client.close()

    def test_health_check(self, monkeypatch):
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"tools": []}
            def raise_for_status(self):
                pass

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                pass
            def get(self, url, **kwargs):
                return FakeResponse()
            def close(self):
                pass

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

        client = AWMMCPSyncClient()
        assert client.health_check() is True
        client.close()


# =========================================================================
# AWMServerManager live mode (Phase 2)
# =========================================================================


class TestAWMServerManagerLive:
    """Test AWMServerManager with live_mode=True (mocked subprocess)."""

    @pytest.mark.asyncio
    async def test_start_server_live_mode(self, monkeypatch):
        from swarm.bridges.awm.server_manager import AWMServerManager

        class FakeProcess:
            returncode = None
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                pass
            def kill(self):
                pass

        monkeypatch.setattr(
            "swarm.bridges.awm.server_manager.subprocess.Popen",
            lambda *a, **kw: FakeProcess(),
        )

        # Make health check pass immediately
        from swarm.bridges.awm import mcp_client as mc_mod

        class FakeSyncClient:
            def __init__(self, **kwargs):
                pass
            def health_check(self):
                return True
            def close(self):
                pass

        monkeypatch.setattr(mc_mod, "AWMMCPSyncClient", FakeSyncClient)

        config = AWMConfig(base_port=19200, live_mode=True)
        mgr = AWMServerManager(config)
        server = await mgr.start_server("agent_1")
        assert server is not None
        assert server.running is True

    @pytest.mark.asyncio
    async def test_stop_server_terminates_process(self, monkeypatch):
        from swarm.bridges.awm.server_manager import AWMServerInstance

        terminated = []

        class FakeProcess:
            returncode = None
            def poll(self):
                return None
            def terminate(self):
                terminated.append(True)
            def wait(self, timeout=None):
                pass
            def kill(self):
                pass

        server = AWMServerInstance(
            agent_id="agent_1",
            port=19200,
            environment_id="test",
            envs_path="/tmp",
            live_mode=True,
        )
        server._process = FakeProcess()
        server.running = True

        await server.stop()
        assert len(terminated) == 1
        assert server.running is False

    @pytest.mark.asyncio
    async def test_reset_db_calls_endpoint(self, monkeypatch):
        from swarm.bridges.awm.server_manager import AWMServerInstance

        reset_called = []

        from swarm.bridges.awm import mcp_client as mc_mod

        class FakeSyncClient:
            def __init__(self, **kwargs):
                pass
            def reset_environment(self):
                reset_called.append(True)
                return True
            def close(self):
                pass

        monkeypatch.setattr(mc_mod, "AWMMCPSyncClient", FakeSyncClient)

        server = AWMServerInstance(
            agent_id="agent_1",
            port=19200,
            environment_id="test",
            envs_path="/tmp",
            live_mode=True,
        )
        result = await server.reset_db()
        assert result is True
        assert len(reset_called) == 1


# =========================================================================
# AWMHandler live mode (Phase 2)
# =========================================================================


class TestAWMHandlerLiveMode:
    """Test AWMHandler with live_mode=True (mocked clients)."""

    def _make_live_handler(self, seed=42):
        from swarm.core.awm_handler import AWMHandler
        from swarm.logging.event_bus import EventBus

        config = AWMConfig(seed=seed, max_tasks_per_epoch=2, live_mode=True)
        bus = EventBus()
        handler = AWMHandler(config=config, event_bus=bus, seed=seed)
        return handler

    def _make_state(self, agent_ids=None):
        from swarm.env.state import EnvState
        from swarm.models.agent import AgentType

        state = EnvState()
        for aid in (agent_ids or ["agent_1"]):
            state.add_agent(aid, name=aid, agent_type=AgentType.HONEST)
        return state

    def test_handle_action_live_mode(self, monkeypatch):
        from swarm.agents.base import Action

        handler = self._make_live_handler()

        # Mock the sync client for the agent
        class MockSyncClient:
            def call_tool(self, tool_name, arguments):
                return ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    success=True,
                    result={"status": "ok"},
                )
            def verify(self):
                return {"passed": True, "confidence": 0.9}
            def list_tools(self):
                return [{"name": "query_database"}]
            def close(self):
                pass

        # Bypass server manager startup by directly injecting client
        handler._clients["agent_1"] = MockSyncClient()

        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.AWM_EXECUTE_TASK,
            agent_id="agent_1",
            metadata={
                "tool_calls": [
                    {"tool_name": "query_database", "arguments": {"q": "SELECT 1"}},
                ]
            },
        )
        result = handler.handle_action(action, state)
        assert result.success is True
        assert result.metadata["verified"] is True

    def test_fallback_on_connection_error(self, monkeypatch):
        from swarm.agents.base import Action

        handler = self._make_live_handler()

        class FailingSyncClient:
            def call_tool(self, tool_name, arguments):
                raise ConnectionError("server down")
            def verify(self):
                raise ConnectionError("server down")
            def list_tools(self):
                raise ConnectionError("server down")
            def close(self):
                pass

        handler._clients["agent_1"] = FailingSyncClient()

        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.AWM_EXECUTE_TASK,
            agent_id="agent_1",
            metadata={
                "tool_calls": [
                    {"tool_name": "query_database", "arguments": {}},
                ]
            },
        )
        result = handler.handle_action(action, state)
        # Should still succeed (action processed) but with errors in trace
        assert result.success is True
        assert result.metadata["verified"] is False

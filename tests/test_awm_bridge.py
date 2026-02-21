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

    def test_llm_base_url_rejects_private_ip(self):
        with pytest.raises(Exception, match="must not target private"):
            AWMConfig(llm_base_url="http://192.168.1.1:8080/v1")

    def test_llm_base_url_rejects_loopback(self):
        with pytest.raises(Exception, match="must not target private"):
            AWMConfig(llm_base_url="http://127.0.0.1:11434/api")

    def test_llm_base_url_rejects_link_local(self):
        with pytest.raises(Exception, match="must not target private"):
            AWMConfig(llm_base_url="http://169.254.169.254/latest/meta-data")

    def test_llm_base_url_allows_public_hostname(self):
        config = AWMConfig(llm_base_url="https://api.openai.com/v1")
        assert config.llm_base_url == "https://api.openai.com/v1"

    def test_llm_base_url_allows_none(self):
        config = AWMConfig(llm_base_url=None)
        assert config.llm_base_url is None


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
    async def test_popen_uses_list_not_shell(self, monkeypatch):
        """Verify Popen is called with a list (shell=False) for safety."""
        from swarm.bridges.awm.server_manager import AWMServerManager

        captured = {}

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

        def fake_popen(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr(
            "swarm.bridges.awm.server_manager.subprocess.Popen",
            fake_popen,
        )

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
        await mgr.start_server("agent_1")

        # Must be a list, not a string
        assert isinstance(captured["args"][0], list)
        # shell must be False
        assert captured["kwargs"].get("shell") is False

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


# =========================================================================
# AWM Multi-Turn (Phase 3) — Agent + ActionType + Observation
# =========================================================================


class TestAWMMultiTurn:
    """Test multi-turn action types, observation fields, and agent step mode."""

    def test_action_types_exist(self):
        assert ActionType.AWM_TOOL_CALL.value == "awm_tool_call"
        assert ActionType.AWM_FINISH_TASK.value == "awm_finish_task"

    def test_observation_multi_turn_fields(self):
        obs = Observation()
        assert obs.awm_last_result is None
        assert obs.awm_episode_active is False
        assert obs.awm_steps_remaining == 0

    def test_config_step_mode_default(self):
        config = AWMConfig()
        assert config.step_mode is False

    def test_agent_step_mode_config(self):
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True},
        )
        assert agent.step_mode is True

    def test_agent_batch_mode_unchanged(self):
        agent = AWMAgent(
            agent_id="awm_batch",
            config={"step_mode": False, "tool_call_count": 3},
        )
        obs = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.AWM_EXECUTE_TASK

    def test_agent_step_mode_first_call(self):
        import random

        rng = random.Random(42)
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True, "tool_call_count": 3},
            rng=rng,
        )
        obs = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.AWM_TOOL_CALL
        assert "tool_name" in action.metadata

    def test_agent_step_mode_continues_with_active_episode(self):
        import random

        rng = random.Random(42)
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True, "tool_call_count": 3},
            rng=rng,
        )
        # First call: generates plan
        obs1 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        action1 = agent.act(obs1)
        assert action1.action_type == ActionType.AWM_TOOL_CALL

        # Second call: episode is active, continues plan
        obs2 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=True,
            awm_steps_remaining=9,
            awm_last_result={"tool_name": "query_database", "success": True},
        )
        action2 = agent.act(obs2)
        assert action2.action_type == ActionType.AWM_TOOL_CALL

    def test_agent_step_mode_finish_when_done(self):
        import random

        rng = random.Random(42)
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True, "tool_call_count": 1},
            rng=rng,
        )
        # First call: generates 1-call plan
        obs1 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        action1 = agent.act(obs1)
        assert action1.action_type == ActionType.AWM_TOOL_CALL

        # Second call: plan exhausted → finish
        obs2 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=True,
            awm_steps_remaining=9,
        )
        action2 = agent.act(obs2)
        assert action2.action_type == ActionType.AWM_FINISH_TASK

    def test_agent_step_mode_finish_on_zero_steps(self):
        import random

        rng = random.Random(42)
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True, "tool_call_count": 5},
            rng=rng,
        )
        # Generate the plan first
        obs1 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        agent.act(obs1)

        # Episode active but no steps remaining → finish
        obs2 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=[{"name": "query_database", "description": "SQL"}],
            awm_episode_active=True,
            awm_steps_remaining=0,
        )
        action = agent.act(obs2)
        assert action.action_type == ActionType.AWM_FINISH_TASK


# =========================================================================
# AWM Handler Multi-Turn (Phase 3)
# =========================================================================


class TestAWMHandlerMultiTurn:
    """Test AWMHandler multi-turn dispatch and observation building."""

    def _make_handler(self, seed=42):
        from swarm.core.awm_handler import AWMHandler
        from swarm.logging.event_bus import EventBus

        config = AWMConfig(seed=seed, max_tasks_per_epoch=2)
        bus = EventBus()
        return AWMHandler(config=config, event_bus=bus, seed=seed)

    def _make_state(self, agent_ids=None):
        from swarm.env.state import EnvState

        state = EnvState()
        for aid in (agent_ids or ["agent_1"]):
            state.add_agent(aid, name=aid, agent_type=AgentType.HONEST)
        return state

    def test_handler_claims_new_action_types(self):
        handler = self._make_handler()
        types = handler.handled_action_types()
        assert ActionType.AWM_TOOL_CALL in types
        assert ActionType.AWM_FINISH_TASK in types
        assert ActionType.AWM_EXECUTE_TASK in types

    def test_observation_mapping_includes_new_fields(self):
        handler = self._make_handler()
        mapping = handler.observation_field_mapping()
        assert "awm_last_result" in mapping
        assert "awm_episode_active" in mapping
        assert "awm_steps_remaining" in mapping

    def test_tool_call_returns_no_observables(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.AWM_TOOL_CALL,
            agent_id="agent_1",
            metadata={"tool_name": "query_database", "arguments": {"q": "SELECT 1"}},
        )
        result = handler.handle_action(action, state)
        assert result.success is True
        assert result.observables is None  # Episode continues

    def test_tool_call_stores_last_result(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.AWM_TOOL_CALL,
            agent_id="agent_1",
            metadata={"tool_name": "query_database", "arguments": {}},
        )
        handler.handle_action(action, state)

        # Last result should be stored
        assert "agent_1" in handler._last_results
        last = handler._last_results["agent_1"]
        assert last["tool_name"] == "query_database"

        # Observation should include it
        fields = handler.build_observation_fields("agent_1", state)
        assert fields["awm_episode_active"] is True
        assert fields["awm_steps_remaining"] >= 0
        assert "awm_last_result" in fields

    def test_finish_task_returns_observables(self):
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        # First: make a tool call so the trace has something
        tc_action = Action(
            action_type=ActionType.AWM_TOOL_CALL,
            agent_id="agent_1",
            metadata={"tool_name": "query_database", "arguments": {}},
        )
        handler.handle_action(tc_action, state)

        # Then: finish the task
        finish_action = Action(
            action_type=ActionType.AWM_FINISH_TASK,
            agent_id="agent_1",
        )
        result = handler.handle_action(finish_action, state)
        assert result.success is True
        assert result.observables is not None  # Triggers proxy computation
        assert "verified" in result.metadata

    def test_multi_turn_full_flow(self):
        """End-to-end: tool_call → tool_call → finish_task."""
        from swarm.agents.base import Action

        handler = self._make_handler()
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        # Step 1: first tool call
        action1 = Action(
            action_type=ActionType.AWM_TOOL_CALL,
            agent_id="agent_1",
            metadata={"tool_name": "query_database", "arguments": {"q": "SELECT 1"}},
        )
        r1 = handler.handle_action(action1, state)
        assert r1.success is True
        assert r1.observables is None

        # Verify observation fields between steps
        fields = handler.build_observation_fields("agent_1", state)
        assert fields["awm_episode_active"] is True
        assert fields["awm_steps_remaining"] >= 0
        assert fields["awm_last_result"]["tool_name"] == "query_database"

        # Step 2: second tool call
        action2 = Action(
            action_type=ActionType.AWM_TOOL_CALL,
            agent_id="agent_1",
            metadata={"tool_name": "update_record", "arguments": {"id": 1}},
        )
        r2 = handler.handle_action(action2, state)
        assert r2.success is True
        assert r2.observables is None

        # Step 3: finish
        action3 = Action(
            action_type=ActionType.AWM_FINISH_TASK,
            agent_id="agent_1",
        )
        r3 = handler.handle_action(action3, state)
        assert r3.success is True
        assert r3.observables is not None
        assert r3.metadata["steps_used"] == 2

        # Trace should be cleaned up
        assert "agent_1" not in handler._traces
        assert "agent_1" not in handler._last_results


# =========================================================================
# AWMAgent LLM Planning (Phase 3)
# =========================================================================


class TestAWMAgentLLMPlanning:
    """Test LLM-based tool planning — all LLM calls are mocked."""

    _TOOLS = [
        {"name": "query_database", "description": "Run SQL queries"},
        {"name": "update_record", "description": "Update a row"},
    ]

    def _obs(self, **kwargs):
        defaults = {
            "awm_task": {"task_id": "t1", "description": "Complete task"},
            "awm_available_tools": self._TOOLS,
        }
        defaults.update(kwargs)
        return Observation(**defaults)

    # ---- 1. disabled by default ----
    def test_llm_planning_disabled_by_default(self):
        agent = AWMAgent(agent_id="awm_1")
        assert agent._llm_enabled is False
        assert agent._llm_delegate is None
        action = agent.act(self._obs())
        assert action.action_type == ActionType.AWM_EXECUTE_TASK

    # ---- 2. enabled via config ----
    def test_llm_planning_enabled_via_config(self):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={"llm_planning": True, "llm_provider": "anthropic"},
        )
        assert agent._llm_enabled is True

    # ---- 3. successful LLM plan ----
    def test_llm_plan_tool_calls_success(self, monkeypatch):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4-20250514",
            },
        )

        # Create a mock delegate
        from unittest.mock import MagicMock

        from swarm.agents.llm_config import LLMUsageStats

        mock_delegate = MagicMock()
        mock_delegate._call_llm_sync.return_value = (
            '{"reasoning": "read first", "tool_calls": [{"tool_name": "query_database", "arguments": {"query": "SELECT 1"}}]}',
            100,
            50,
        )
        mock_delegate._parse_action_response.return_value = {
            "reasoning": "read first",
            "tool_calls": [
                {"tool_name": "query_database", "arguments": {"query": "SELECT 1"}},
            ],
        }
        mock_delegate.usage_stats = LLMUsageStats()

        agent._llm_delegate = mock_delegate
        result = agent._plan_tool_calls(self._obs())
        assert len(result) == 1
        assert result[0]["tool_name"] == "query_database"

    # ---- 4. fallback to scripted on failure ----
    def test_llm_fallback_to_scripted_on_failure(self, monkeypatch):
        import random as stdlib_random

        rng = stdlib_random.Random(42)
        agent = AWMAgent(
            agent_id="awm_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "llm_fallback_to_scripted": True,
                "tool_call_count": 2,
            },
            rng=rng,
        )

        from unittest.mock import MagicMock

        mock_delegate = MagicMock()
        mock_delegate._call_llm_sync.side_effect = RuntimeError("API down")
        agent._llm_delegate = mock_delegate

        result = agent._plan_tool_calls(self._obs())
        # Should fall back to scripted and return 2 calls
        assert len(result) == 2

    # ---- 5. no fallback returns empty ----
    def test_llm_no_fallback_returns_empty(self):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "llm_fallback_to_scripted": False,
            },
        )

        from unittest.mock import MagicMock

        mock_delegate = MagicMock()
        mock_delegate._call_llm_sync.side_effect = RuntimeError("API down")
        agent._llm_delegate = mock_delegate

        result = agent._plan_tool_calls(self._obs())
        assert result == []

    # ---- 6. malformed LLM response ----
    def test_llm_parse_malformed_response(self):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={"llm_planning": True, "llm_provider": "anthropic"},
        )

        from unittest.mock import MagicMock

        mock_delegate = MagicMock()
        mock_delegate._parse_action_response.side_effect = ValueError("No JSON")
        agent._llm_delegate = mock_delegate

        result = agent._parse_tool_call_response("not json at all", self._obs())
        assert result is None

    # ---- 7. caps tool calls ----
    def test_llm_caps_tool_calls(self):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "llm_max_calls_per_plan": 2,
            },
        )

        from unittest.mock import MagicMock

        mock_delegate = MagicMock()
        mock_delegate._parse_action_response.return_value = {
            "tool_calls": [
                {"tool_name": "query_database", "arguments": {}},
                {"tool_name": "update_record", "arguments": {}},
                {"tool_name": "query_database", "arguments": {}},
                {"tool_name": "update_record", "arguments": {}},
            ],
        }
        agent._llm_delegate = mock_delegate

        result = agent._parse_tool_call_response("ignored", self._obs())
        assert result is not None
        assert len(result) == 2

    # ---- 8. step mode prompt ----
    def test_llm_step_mode_prompt(self):
        agent = AWMAgent(
            agent_id="awm_step_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "step_mode": True,
            },
        )
        obs = self._obs(awm_steps_remaining=5, awm_episode_active=False)
        _sys, user = agent._build_awm_tool_prompt(obs)
        assert "Plan ONE tool call" in user
        assert "Steps remaining: 5" in user

    # ---- 9. batch mode prompt ----
    def test_llm_batch_mode_prompt(self):
        agent = AWMAgent(
            agent_id="awm_batch_llm",
            config={
                "llm_planning": True,
                "llm_provider": "anthropic",
                "llm_max_calls_per_plan": 7,
            },
        )
        _sys, user = agent._build_awm_tool_prompt(self._obs())
        assert "Plan up to 7 tool calls" in user

    # ---- 10. usage stats exposed ----
    def test_llm_usage_stats_exposed(self):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={"llm_planning": True, "llm_provider": "anthropic"},
        )
        # No delegate yet
        assert agent.llm_usage_stats is None

        from unittest.mock import MagicMock

        from swarm.agents.llm_config import LLMUsageStats

        mock_delegate = MagicMock()
        stats = LLMUsageStats()
        stats.total_requests = 3
        mock_delegate.usage_stats = stats
        agent._llm_delegate = mock_delegate

        result = agent.llm_usage_stats
        assert result is not None
        assert result["total_requests"] == 3

    # ---- 11. init failure disables LLM ----
    def test_llm_init_failure_disables(self, monkeypatch):
        agent = AWMAgent(
            agent_id="awm_llm",
            config={
                "llm_planning": True,
                "llm_provider": "nonexistent_provider_xyz",
            },
        )
        assert agent._llm_enabled is True
        delegate = agent._init_llm()
        assert delegate is None
        assert agent._llm_enabled is False

    # ---- 12. tool call history resets ----
    def test_tool_call_history_reset(self):
        import random as stdlib_random

        rng = stdlib_random.Random(42)
        agent = AWMAgent(
            agent_id="awm_step",
            config={"step_mode": True, "tool_call_count": 2},
            rng=rng,
        )

        obs1 = Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=self._TOOLS,
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        agent.act(obs1)
        assert len(agent._tool_call_history) == 1

        agent.act(Observation(
            awm_task={"task_id": "t1"},
            awm_available_tools=self._TOOLS,
            awm_episode_active=True,
            awm_steps_remaining=9,
        ))
        assert len(agent._tool_call_history) == 2

        # New episode resets history
        obs_new = Observation(
            awm_task={"task_id": "t2"},
            awm_available_tools=self._TOOLS,
            awm_episode_active=False,
            awm_steps_remaining=10,
        )
        agent.act(obs_new)
        assert len(agent._tool_call_history) == 1


# =========================================================================
# AWM Shared Database (Phase 4)
# =========================================================================


class TestAWMSharedDatabase:
    """Test shared-database multi-agent coordination (Phase 4)."""

    def _make_handler(self, seed=42, **overrides):
        from swarm.core.awm_handler import AWMHandler
        from swarm.logging.event_bus import EventBus

        kwargs = {"seed": seed, "max_tasks_per_epoch": 2}
        kwargs.update(overrides)
        config = AWMConfig(**kwargs)
        bus = EventBus()
        collected: list = []
        bus.subscribe(lambda e: collected.append(e))
        bus.events = collected  # type: ignore[attr-defined]
        return AWMHandler(config=config, event_bus=bus, seed=seed), bus

    def _make_state(self, agent_ids=None):
        from swarm.env.state import EnvState

        state = EnvState()
        for aid in (agent_ids or ["agent_1", "agent_2"]):
            state.add_agent(aid, name=aid, agent_type=AgentType.HONEST)
        return state

    def test_config_defaults_preserved(self):
        """shared_database=False by default — no change to existing behavior."""
        config = AWMConfig()
        assert config.shared_database is False
        assert config.isolation_level == "read_committed"
        assert config.conflict_probability == 0.3

    def test_simulated_no_conflict_read_only(self):
        """Read-only tools never trigger conflicts even in shared mode."""
        from swarm.agents.base import Action

        handler, bus = self._make_handler(
            shared_database=True, conflict_probability=1.0,
        )
        state = self._make_state(["agent_1", "agent_2"])
        handler.on_epoch_start(state)

        # Both agents do read-only calls
        for aid in ["agent_1", "agent_2"]:
            if aid not in handler._assignments:
                continue
            action = Action(
                action_type=ActionType.AWM_EXECUTE_TASK,
                agent_id=aid,
                metadata={
                    "tool_calls": [
                        {"tool_name": "query_database", "arguments": {"q": "SELECT 1"}},
                        {"tool_name": "list_tables", "arguments": {}},
                    ]
                },
            )
            result = handler.handle_action(action, state)
            assert result.success is True

        # No conflict events should have been emitted
        conflict_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_CONFLICT_DETECTED
        ]
        assert len(conflict_events) == 0

    def test_simulated_write_conflict_deterministic(self):
        """With conflict_probability=1.0, overlapping writes always conflict."""
        from swarm.agents.base import Action

        handler, bus = self._make_handler(
            shared_database=True, conflict_probability=1.0,
        )
        state = self._make_state(["agent_1", "agent_2"])
        handler.on_epoch_start(state)

        # Agent 1 does a write (populates write set)
        if "agent_1" in handler._assignments:
            action1 = Action(
                action_type=ActionType.AWM_EXECUTE_TASK,
                agent_id="agent_1",
                metadata={
                    "tool_calls": [
                        {"tool_name": "update_record", "arguments": {"id": 1}},
                    ]
                },
            )
            handler.handle_action(action1, state)

        # Agent 2 does a write to the same table → conflict
        if "agent_2" in handler._assignments:
            action2 = Action(
                action_type=ActionType.AWM_EXECUTE_TASK,
                agent_id="agent_2",
                metadata={
                    "tool_calls": [
                        {"tool_name": "update_record", "arguments": {"id": 2}},
                    ]
                },
            )
            handler.handle_action(action2, state)

        conflict_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_CONFLICT_DETECTED
        ]
        assert len(conflict_events) >= 1
        assert conflict_events[0].payload["conflict_type"] == "simulated_write_set_overlap"

    def test_transaction_events_emitted_on_batch(self):
        """Batch mode emits AWM_TRANSACTION_COMPLETED events in shared mode."""
        from swarm.agents.base import Action

        handler, bus = self._make_handler(
            shared_database=True, conflict_probability=0.0,
        )
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
        handler.handle_action(action, state)

        tx_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_TRANSACTION_COMPLETED
        ]
        assert len(tx_events) == 1
        assert tx_events[0].payload["committed"] is True

    def test_epoch_end_clears_write_sets(self):
        """on_epoch_end clears write sets and transaction state."""
        handler, _bus = self._make_handler(shared_database=True)
        state = self._make_state(["agent_1"])
        handler.on_epoch_start(state)

        # Manually populate to verify clearing
        handler._write_sets["agent_1"] = {"default_table"}
        handler._agent_transactions["agent_1"] = True

        handler.on_epoch_end(state)
        assert handler._write_sets == {}
        assert handler._agent_transactions == {}

    def test_isolation_level_none_skips_transactions(self):
        """isolation_level='none' skips begin/end transaction."""
        from swarm.agents.base import Action

        handler, bus = self._make_handler(
            shared_database=True, isolation_level="none",
        )
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
        handler.handle_action(action, state)

        # No transaction events when isolation_level="none"
        tx_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_TRANSACTION_COMPLETED
        ]
        assert len(tx_events) == 0

    def test_non_shared_mode_no_conflict_events(self):
        """With shared_database=False (default), no conflict events emitted."""
        from swarm.agents.base import Action

        handler, bus = self._make_handler(shared_database=False)
        state = self._make_state(["agent_1", "agent_2"])
        handler.on_epoch_start(state)

        for aid in ["agent_1", "agent_2"]:
            if aid not in handler._assignments:
                continue
            action = Action(
                action_type=ActionType.AWM_EXECUTE_TASK,
                agent_id=aid,
                metadata={
                    "tool_calls": [
                        {"tool_name": "update_record", "arguments": {"id": 1}},
                    ]
                },
            )
            handler.handle_action(action, state)

        conflict_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_CONFLICT_DETECTED
        ]
        assert len(conflict_events) == 0

        tx_events = [
            e for e in bus.events
            if e.event_type == EventType.AWM_TRANSACTION_COMPLETED
        ]
        assert len(tx_events) == 0


# =========================================================================
# AWM Server Manager Shared (Phase 4)
# =========================================================================


class TestAWMServerManagerShared:
    """Test AWMServerManager shared-database mode (Phase 4)."""

    @pytest.mark.asyncio
    async def test_shared_server_same_instance(self):
        """Two agents get the same server instance in shared mode."""
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19300, shared_database=True)
        mgr = AWMServerManager(config)
        s1 = await mgr.start_server("agent_1")
        s2 = await mgr.start_server("agent_2")
        assert s1 is s2
        assert s1 is not None
        assert s1.agent_id == "shared"

    @pytest.mark.asyncio
    async def test_shared_reset_all_calls_once(self):
        """reset_all resets the shared server once, not per-agent."""
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19300, shared_database=True)
        mgr = AWMServerManager(config)
        await mgr.start_server("agent_1")
        await mgr.start_server("agent_2")

        reset_count = 0
        original_reset = mgr._shared_server.reset_db

        async def counting_reset():
            nonlocal reset_count
            reset_count += 1
            return await original_reset()

        mgr._shared_server.reset_db = counting_reset
        await mgr.reset_all()
        assert reset_count == 1

    @pytest.mark.asyncio
    async def test_shared_shutdown_cleans_up(self):
        """shutdown stops the shared server and clears state."""
        from swarm.bridges.awm.server_manager import AWMServerManager

        config = AWMConfig(base_port=19300, shared_database=True)
        mgr = AWMServerManager(config)
        await mgr.start_server("agent_1")
        await mgr.start_server("agent_2")
        assert mgr.active_count == 2  # Both mapped to same server

        await mgr.shutdown()
        assert mgr.active_count == 0
        assert mgr._shared_server is None


# =========================================================================
# Adapter Server — SSRF hardening
# =========================================================================


class TestAdapterServerSSRF:
    """Test SSRF protections in adapter_server.call_tool dispatch."""

    def _make_validator(self):
        """Re-implement the nested path-param validator for direct unit testing."""

        def _validate_path_param_value(name, value):
            s = str(value)
            if "/" in s or "\\" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected '/' or '\\\\'")
            if "?" in s or "#" in s:
                raise ValueError(
                    f"Invalid path parameter '{name}': unexpected query/fragment delimiter"
                )
            if "\n" in s or "\r" in s or "\t" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected whitespace")
            if s.startswith(".") or ".." in s:
                raise ValueError(f"Invalid path parameter '{name}': potentially unsafe value")
            return s

        return _validate_path_param_value

    # -- Path parameter validation --

    def test_path_param_rejects_slash(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="unexpected '/'"):
            validate("id", "../../etc/passwd")

    def test_path_param_rejects_backslash(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="unexpected '/'"):
            validate("id", "foo\\bar")

    def test_path_param_rejects_question_mark(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="query/fragment delimiter"):
            validate("id", "val?injected=true")

    def test_path_param_rejects_hash(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="query/fragment delimiter"):
            validate("id", "val#fragment")

    def test_path_param_rejects_newline(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="unexpected whitespace"):
            validate("id", "val\ninjected")

    def test_path_param_rejects_dotdot(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="potentially unsafe"):
            validate("id", "..secret")

    def test_path_param_rejects_leading_dot(self):
        validate = self._make_validator()
        with pytest.raises(ValueError, match="potentially unsafe"):
            validate("id", ".hidden")

    def test_path_param_accepts_safe_value(self):
        validate = self._make_validator()
        assert validate("id", "42") == "42"
        assert validate("name", "hello-world_v2") == "hello-world_v2"

    # -- Path normalization and dispatch path construction --

    def test_normpath_preserves_trailing_slash(self):
        """normpath strips trailing slash; our code must re-append it."""
        import posixpath

        path_only = "/items/"
        normalized = posixpath.normpath(path_only)
        assert normalized == "/items"
        if path_only.endswith("/") and normalized != "/":
            normalized += "/"
        assert normalized == "/items/"

    def test_normpath_collapses_dotdot(self):
        import posixpath

        assert posixpath.normpath("/a/../b") == "/b"
        assert posixpath.normpath("/a/./b") == "/a/b"
        assert posixpath.normpath("/a//b") == "/a/b"

    def test_dispatch_path_uses_normalized(self):
        import posixpath

        rendered_path = "/items/./list"
        path_only, sep, query = rendered_path.partition("?")
        normalized = posixpath.normpath(path_only)
        if path_only.endswith("/") and normalized != "/":
            normalized += "/"
        dispatch_path = normalized + sep + query
        assert dispatch_path == "/items/list"

    def test_dispatch_path_with_query(self):
        import posixpath

        rendered_path = "/items?page=2&size=10"
        path_only, sep, query = rendered_path.partition("?")
        normalized = posixpath.normpath(path_only)
        if path_only.endswith("/") and normalized != "/":
            normalized += "/"
        dispatch_path = normalized + sep + query
        assert dispatch_path == "/items?page=2&size=10"

    # -- _sanitize_dispatch_path --

    def test_sanitize_rejects_path_with_special_chars(self):
        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        assert _sanitize_dispatch_path("/items/@admin", "", "") is None
        assert _sanitize_dispatch_path("/items/ space", "", "") is None
        assert _sanitize_dispatch_path("/items/<script>", "", "") is None

    def test_sanitize_accepts_normal_path(self):
        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        assert _sanitize_dispatch_path("/items", "", "") == "/items"
        assert _sanitize_dispatch_path("/items/42", "", "") == "/items/42"
        assert _sanitize_dispatch_path("/users/john_doe-v2", "", "") == "/users/john_doe-v2"

    def test_sanitize_rejects_bad_query(self):
        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        assert _sanitize_dispatch_path("/items", "?", "key=<script>") is None
        assert _sanitize_dispatch_path("/items", "?", "key=val ue") is None
        assert _sanitize_dispatch_path("/items", "?", "key=val^ue") is None

    def test_sanitize_accepts_normal_query(self):
        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        result = _sanitize_dispatch_path("/items", "?", "page=2&size=10")
        assert result == "/items?page=2&size=10"
        result = _sanitize_dispatch_path("/items", "?", "q=hello+world")
        assert result == "/items?q=hello+world"

    def test_sanitize_returns_new_string(self):
        """The returned string must be a fresh object (breaks taint chain)."""
        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        original = "/items/42"
        result = _sanitize_dispatch_path(original, "", "")
        assert result == original
        assert result is not original

    # -- _is_safe_dispatch_path structural validation --

    def test_structural_rejects_non_absolute(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("items/42") is False

    def test_structural_rejects_scheme_relative(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("//evil.com/path") is False

    def test_structural_rejects_full_url(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("http://evil.com/path") is False

    def test_structural_rejects_dotdot(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("/items/../etc/passwd") is False
        assert _is_safe_dispatch_path("/items/..") is False

    def test_structural_rejects_double_slash(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("/items//hidden") is False

    def test_structural_rejects_dot_segment(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("/items/./list") is False

    def test_structural_accepts_valid_paths(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("/items") is True
        assert _is_safe_dispatch_path("/items/42") is True
        assert _is_safe_dispatch_path("/api/v1.0/users/john_doe-v2") is True
        assert _is_safe_dispatch_path("/") is True

    def test_structural_accepts_path_with_query(self):
        from swarm.bridges.awm.adapter_server import _is_safe_dispatch_path

        assert _is_safe_dispatch_path("/items?page=2&size=10") is True

    # -- _validate_tool_base_path startup validation --

    def test_validate_base_path_accepts_valid(self):
        from swarm.bridges.awm.adapter_server import _validate_tool_base_path

        assert _validate_tool_base_path("/items") == "/items"
        assert _validate_tool_base_path("/api/v1/users") == "/api/v1/users"
        assert _validate_tool_base_path("/items/") == "/items/"

    def test_validate_base_path_rejects_no_leading_slash(self):
        from swarm.bridges.awm.adapter_server import _validate_tool_base_path

        with pytest.raises(ValueError, match="must start at the application root"):
            _validate_tool_base_path("items/relative")

    def test_validate_base_path_rejects_invalid_chars(self):
        from swarm.bridges.awm.adapter_server import _validate_tool_base_path

        with pytest.raises(ValueError, match="contains invalid characters"):
            _validate_tool_base_path("/items/@admin")

    def test_validate_base_path_normalizes(self):
        from swarm.bridges.awm.adapter_server import _validate_tool_base_path

        # normpath collapses redundant separators/dot segments
        assert _validate_tool_base_path("/items/./list") == "/items/list"
        assert _validate_tool_base_path("/items//list") == "/items/list"

    # -- Dispatch exception: generic error, no raw exception text --

    def test_dispatch_exception_returns_generic_error(self, monkeypatch, tmp_path):
        """Dispatch errors must return HTTP 500 with isError and no raw exception text.

        Regression test for the fix that replaced bare ``str(exc)`` in the
        except block with a static generic message so that internal details
        (e.g. connection strings, passwords) are never surfaced in responses.
        """
        import httpx
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        # Minimal AWM app with one GET route so tool_meta gets populated.
        awm_app = FastAPI()

        @awm_app.get("/ping")
        async def ping():
            return {"ok": True}

        # Patch the file-loading helpers so build_adapter works without real files.
        monkeypatch.setattr(
            "swarm.bridges.awm.adapter_server._load_scenario_code",
            lambda *a: "",
        )
        monkeypatch.setattr(
            "swarm.bridges.awm.adapter_server._exec_awm_app",
            lambda *a: awm_app,
        )

        # build_adapter requires the source DB file to exist before copying.
        (tmp_path / "test.db").touch()

        from swarm.bridges.awm.adapter_server import build_adapter

        adapter = build_adapter(
            scenario="test",
            envs_jsonl=tmp_path / "any.jsonl",
            db_dir=tmp_path,
            data_path=tmp_path,
            task_idx=0,
        )

        # Replace httpx.AsyncClient so the inner dispatch raises with sensitive text.
        # TestClient (which inherits httpx.Client, not AsyncClient) is unaffected.
        _SECRET = "db_password=hunter2"

        class _RaisingClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, *args, **kwargs):
                raise RuntimeError(f"secret: {_SECRET}")

            async def post(self, *args, **kwargs):
                raise RuntimeError(f"secret: {_SECRET}")

            async def put(self, *args, **kwargs):
                raise RuntimeError(f"secret: {_SECRET}")

            async def patch(self, *args, **kwargs):
                raise RuntimeError(f"secret: {_SECRET}")

            async def delete(self, *args, **kwargs):
                raise RuntimeError(f"secret: {_SECRET}")

        monkeypatch.setattr(httpx, "AsyncClient", _RaisingClient)

        client = TestClient(adapter, raise_server_exceptions=False)
        resp = client.post("/tools/call", json={"name": "ping", "arguments": {}})

        assert resp.status_code == 500
        body = resp.json()
        assert body["isError"] is True
        # The raw exception text must not appear in the response body.
        assert _SECRET not in resp.text
        # A human-readable generic message must be present.
        assert "internal error" in body["result"].lower()

    # -- Base path anchoring with path templates --

    def test_base_static_prefix_extracted_before_sanitize(self):
        """Static prefix before first {placeholder} must not include curly braces."""
        import re

        path = "/users/{user_id}"
        base_path_only = path.partition("?")[0]
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        assert base_static == "/users/"
        # Confirm no curly braces remain
        assert "{" not in base_static and "}" not in base_static

    def test_base_static_prefix_multi_param(self):
        """Only the prefix before the first placeholder is kept."""
        import re

        path = "/api/v1/{resource}/{id}/details"
        base_path_only = path.partition("?")[0]
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        assert base_static == "/api/v1/"

    def test_base_static_prefix_no_params(self):
        """A path with no placeholders returns the full path as the static prefix."""
        import re

        path = "/items"
        base_path_only = path.partition("?")[0]
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        assert base_static == "/items"

    def test_sanitize_accepts_static_prefix_from_template(self):
        """_sanitize_dispatch_path must accept the extracted static prefix."""
        import posixpath
        import re

        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        for template, expected_base in [
            ("/users/{user_id}", "/users/"),
            ("/api/v1/{resource}/{id}", "/api/v1/"),
            ("/items", "/items"),
        ]:
            base_path_only = template.partition("?")[0]
            base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
            base_normalized = posixpath.normpath(base_static) if base_static else "/"
            if base_static.endswith("/") and base_normalized != "/":
                base_normalized += "/"
            result = _sanitize_dispatch_path(base_normalized, "", "")
            assert result == expected_base, f"template={template!r}: got {result!r}"

    def test_dispatch_within_parameterized_base_accepted(self):
        """/users/123 is within the /users/ prefix derived from /users/{user_id}."""
        import posixpath
        import re

        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        path_template = "/users/{user_id}"
        dispatch_relative_path = "/users/123"

        base_path_only = path_template.partition("?")[0]
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        base_normalized = posixpath.normpath(base_static) if base_static else "/"
        if base_static.endswith("/") and base_normalized != "/":
            base_normalized += "/"
        base_dispatch_path = _sanitize_dispatch_path(base_normalized, "", "")
        assert base_dispatch_path is not None

        if dispatch_relative_path != base_dispatch_path:
            prefix = base_dispatch_path if base_dispatch_path.endswith("/") else base_dispatch_path + "/"
            assert dispatch_relative_path.startswith(prefix)

    def test_dispatch_outside_base_rejected(self):
        """/evil/path is NOT within the /users/ prefix and must be rejected."""
        import posixpath
        import re

        from swarm.bridges.awm.adapter_server import _sanitize_dispatch_path

        path_template = "/users/{user_id}"
        dispatch_relative_path = "/evil/path"

        base_path_only = path_template.partition("?")[0]
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        base_normalized = posixpath.normpath(base_static) if base_static else "/"
        if base_static.endswith("/") and base_normalized != "/":
            base_normalized += "/"
        base_dispatch_path = _sanitize_dispatch_path(base_normalized, "", "")
        assert base_dispatch_path is not None

        in_base = dispatch_relative_path == base_dispatch_path
        if not in_base:
            prefix = base_dispatch_path if base_dispatch_path.endswith("/") else base_dispatch_path + "/"
            in_base = dispatch_relative_path.startswith(prefix)
        assert not in_base, "Path outside base should be rejected"


# =========================================================================
# Exec safety — subprocess verifier + AWM code validation
# =========================================================================


class TestExecSafety:
    """Tests for exec() vulnerability mitigations in adapter_server."""

    def test_verifier_subprocess_valid(self):
        """Good verifier code returns a correct result dict."""
        from swarm.bridges.awm.adapter_server import _run_verifier_subprocess

        code = (
            "def verify_task_completion(initial_db, working_db):\n"
            "    return {'result': 'complete', 'initial': initial_db, 'working': working_db}\n"
        )
        result = _run_verifier_subprocess(code, "/tmp/a.db", "/tmp/b.db")
        assert "result" in result
        assert result["result"]["result"] == "complete"
        assert result["result"]["initial"] == "/tmp/a.db"
        assert result["result"]["working"] == "/tmp/b.db"

    def test_verifier_subprocess_timeout(self):
        """Hanging verifier code is killed after timeout."""
        from swarm.bridges.awm.adapter_server import _run_verifier_subprocess

        code = (
            "import time\n"
            "def verify_task_completion(initial_db, working_db):\n"
            "    time.sleep(60)\n"
            "    return {'result': 'complete'}\n"
        )
        result = _run_verifier_subprocess(code, "/tmp/a.db", "/tmp/b.db", timeout=1)
        assert "error" in result
        assert "timed out" in result["error"]

    def test_verifier_subprocess_syntax_error(self):
        """Verifier code with a syntax error returns an error dict."""
        from swarm.bridges.awm.adapter_server import _run_verifier_subprocess

        code = "def verify_task_completion(:\n"
        result = _run_verifier_subprocess(code, "/tmp/a.db", "/tmp/b.db")
        assert "error" in result
        assert "result" not in result

    def test_validate_awm_code_blocks_dangerous(self):
        """_validate_awm_code rejects code with dangerous patterns."""
        from swarm.bridges.awm.adapter_server import _validate_awm_code

        dangerous_snippets = [
            "import subprocess",
            "os.system('rm -rf /')",
            "os.popen('cat /etc/passwd')",
            "import ctypes",
            "import socket",
            "shutil.rmtree('/tmp')",
            "os.remove('/etc/passwd')",
            "os.unlink('/etc/passwd')",
            "eval('1+1')",
            "compile('code', 'f', 'exec')",
        ]
        for snippet in dangerous_snippets:
            with pytest.raises(ValueError, match="blocked pattern"):
                _validate_awm_code(snippet)

    def test_validate_awm_code_allows_clean(self):
        """_validate_awm_code passes normal FastAPI app code."""
        from swarm.bridges.awm.adapter_server import _validate_awm_code

        clean_code = (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "\n"
            "@app.get('/items')\n"
            "def list_items():\n"
            "    return []\n"
        )
        # Should not raise
        _validate_awm_code(clean_code)

    def test_restricted_builtins_no_eval(self):
        """exec'd code under restricted builtins cannot call eval()."""
        from swarm.bridges.awm.adapter_server import _RESTRICTED_BUILTINS

        # Verify eval/exec/compile are not in the restricted set
        assert "eval" not in _RESTRICTED_BUILTINS
        assert "exec" not in _RESTRICTED_BUILTINS
        assert "compile" not in _RESTRICTED_BUILTINS
        assert "globals" not in _RESTRICTED_BUILTINS
        assert "locals" not in _RESTRICTED_BUILTINS
        assert "breakpoint" not in _RESTRICTED_BUILTINS
        assert "exit" not in _RESTRICTED_BUILTINS
        assert "quit" not in _RESTRICTED_BUILTINS

        # Actually exec code with restricted builtins and confirm eval is unavailable
        ns: dict = {"__builtins__": dict(_RESTRICTED_BUILTINS)}
        try:
            exec("result = eval('1+1')", ns)  # noqa: S102
            raise AssertionError("eval() should have raised NameError")
        except NameError:
            pass  # Expected: eval is not available

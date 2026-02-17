"""Tests for the SciAgentGym bridge."""

from datetime import datetime

import pytest

from swarm.bridges.sciagentgym import (
    DataArtifactEvent,
    PolicyDecision,
    ProviderConfig,
    ProviderType,
    SafetyCheckEvent,
    SciAgentGymBridge,
    SciAgentGymClient,
    SciAgentGymClientConfig,
    SciAgentGymConfig,
    SciAgentGymEvent,
    SciAgentGymEventType,
    SciAgentGymMapper,
    SciAgentGymPolicy,
    ToolCallEvent,
    TopologyConfig,
    TopologyType,
    WorkflowStepEvent,
)
from swarm.models.interaction import InteractionType


class TestProviderConfig:
    """Test provider configuration."""

    def test_default_provider_config(self):
        config = ProviderConfig()
        assert config.provider_type == ProviderType.LOCAL
        assert config.timeout_seconds == 300.0
        assert config.max_retries == 2
        assert config.sandbox_enabled is True

    def test_docker_provider_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.DOCKER,
            base_image="python:3.11",
            resource_limits={"memory": "2Gi", "cpu": "1"},
        )
        assert config.provider_type == ProviderType.DOCKER
        assert config.base_image == "python:3.11"
        assert config.resource_limits["memory"] == "2Gi"

    def test_kubernetes_provider_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.KUBERNETES,
            sandbox_enabled=True,
            working_dir="/mnt/sciagentgym",
        )
        assert config.provider_type == ProviderType.KUBERNETES
        assert config.working_dir == "/mnt/sciagentgym"


class TestTopologyConfig:
    """Test topology configuration."""

    def test_complete_topology(self):
        config = TopologyConfig(topology_type=TopologyType.COMPLETE)
        assert config.topology_type == TopologyType.COMPLETE
        assert config.dynamic_routing is False

    def test_ring_topology(self):
        config = TopologyConfig(
            topology_type=TopologyType.RING,
            k_neighbors=3,
            dynamic_routing=True,
        )
        assert config.topology_type == TopologyType.RING
        assert config.k_neighbors == 3
        assert config.dynamic_routing is True

    def test_star_topology(self):
        config = TopologyConfig(
            topology_type=TopologyType.STAR,
            hub_agent_id="hub_001",
        )
        assert config.topology_type == TopologyType.STAR
        assert config.hub_agent_id == "hub_001"

    def test_custom_tool_access_policy(self):
        config = TopologyConfig(
            tool_access_policy={
                "agent_1": ["tool_physics_1", "tool_chem_1"],
                "agent_2": ["tool_chem_1", "tool_bio_1"],
            }
        )
        assert "agent_1" in config.tool_access_policy
        assert len(config.tool_access_policy["agent_1"]) == 2


class TestSciAgentGymConfig:
    """Test main bridge configuration."""

    def test_default_config(self):
        config = SciAgentGymConfig()
        assert config.orchestrator_id == "sciagentgym_orchestrator"
        assert config.proxy_sigmoid_k == 2.0
        assert config.tool_safety_gate_enabled is True
        assert config.min_tool_safety_score == 0.4
        assert config.max_interactions == 50000
        assert config.max_events == 50000

    def test_config_with_provider_and_topology(self):
        provider_config = ProviderConfig(provider_type=ProviderType.DOCKER)
        topology_config = TopologyConfig(topology_type=TopologyType.RING)

        config = SciAgentGymConfig(
            provider_config=provider_config,
            topology_config=topology_config,
        )

        assert config.provider_config.provider_type == ProviderType.DOCKER
        assert config.topology_config.topology_type == TopologyType.RING

    def test_governance_thresholds(self):
        config = SciAgentGymConfig(
            min_tool_safety_score=0.6,
            workflow_circuit_breaker_max_failures=3,
            cost_budget_tokens=50000,
        )
        assert config.min_tool_safety_score == 0.6
        assert config.workflow_circuit_breaker_max_failures == 3
        assert config.cost_budget_tokens == 50000


class TestToolCallEvent:
    """Test tool call event parsing."""

    def test_successful_tool_call(self):
        data = {
            "tool_name": "simulate_molecule",
            "tool_args": {"molecule": "H2O", "temp": 298},
            "success": True,
            "execution_time_seconds": 12.5,
            "result": {"energy": -76.4},
            "cost_tokens": 150,
        }
        event = ToolCallEvent.from_dict(data)
        assert event.tool_name == "simulate_molecule"
        assert event.success is True
        assert event.execution_time_seconds == 12.5
        assert event.cost_tokens == 150

    def test_failed_tool_call(self):
        data = {
            "tool_name": "run_experiment",
            "tool_args": {"exp_id": "123"},
            "success": False,
            "execution_time_seconds": 1.2,
            "error": "Timeout",
        }
        event = ToolCallEvent.from_dict(data)
        assert event.success is False
        assert event.error == "Timeout"
        assert event.result is None


class TestWorkflowStepEvent:
    """Test workflow step event parsing."""

    def test_workflow_step(self):
        data = {
            "workflow_id": "wf_001",
            "step_index": 2,
            "step_type": "data_validation",
            "success": True,
            "dependencies_met": True,
            "next_steps": [3, 4],
        }
        event = WorkflowStepEvent.from_dict(data)
        assert event.workflow_id == "wf_001"
        assert event.step_index == 2
        assert event.success is True
        assert event.next_steps == [3, 4]


class TestSciAgentGymMapper:
    """Test event mapping to SoftInteractions."""

    @pytest.fixture
    def config(self):
        return SciAgentGymConfig()

    @pytest.fixture
    def mapper(self, config):
        return SciAgentGymMapper(config)

    def test_map_successful_tool_call(self, mapper, config):
        event = SciAgentGymEvent(
            event_type=SciAgentGymEventType.TOOL_CALL_COMPLETED,
            timestamp=datetime.now(),
            agent_id="tool_executor_agent",
            payload={
                "tool_name": "analyze_spectrum",
                "tool_args": {},
                "success": True,
                "execution_time_seconds": 5.0,
                "cost_tokens": 100,
            },
        )
        tool_data = ToolCallEvent.from_dict(event.payload)
        interaction = mapper.map_tool_call(event, tool_data)

        assert interaction is not None
        assert interaction.initiator == "tool_executor_agent"
        assert interaction.counterparty == config.orchestrator_id
        assert interaction.interaction_type == InteractionType.COLLABORATION
        assert interaction.accepted is True
        assert 0.0 <= interaction.p <= 1.0

    def test_map_failed_tool_call(self, mapper):
        event = SciAgentGymEvent(
            event_type=SciAgentGymEventType.TOOL_CALL_FAILED,
            timestamp=datetime.now(),
            agent_id="tool_executor_agent",
            payload={
                "tool_name": "failed_tool",
                "tool_args": {},
                "success": False,
                "execution_time_seconds": 1.0,
                "error": "Tool crashed",
            },
        )
        tool_data = ToolCallEvent.from_dict(event.payload)
        interaction = mapper.map_tool_call(event, tool_data)

        assert interaction is not None
        assert interaction.accepted is False
        assert interaction.p < 0.5  # Failed tools should have low p

    def test_map_safety_check(self, mapper):
        event = SciAgentGymEvent(
            event_type=SciAgentGymEventType.SAFETY_CHECK_PASSED,
            timestamp=datetime.now(),
            agent_id="validator_agent",
            payload={
                "check_type": "code_safety",
                "passed": True,
                "safety_score": 0.85,
                "risk_factors": [],
            },
        )
        safety_data = SafetyCheckEvent.from_dict(event.payload)
        interaction = mapper.map_safety_check(event, safety_data)

        assert interaction is not None
        assert interaction.interaction_type == InteractionType.VOTE
        assert interaction.p == 0.85
        assert interaction.accepted is True


class TestSciAgentGymPolicy:
    """Test governance policy engine."""

    @pytest.fixture
    def config(self):
        return SciAgentGymConfig(
            min_tool_safety_score=0.5,
            workflow_circuit_breaker_max_failures=3,
            cost_budget_tokens=1000,
        )

    @pytest.fixture
    def policy(self, config):
        return SciAgentGymPolicy(config)

    def test_safety_gate_allow(self, policy):
        result = policy.evaluate_tool_safety(0.7)
        assert result.decision == PolicyDecision.ALLOW

    def test_safety_gate_deny(self, policy):
        result = policy.evaluate_tool_safety(0.3)
        assert result.decision == PolicyDecision.DENY
        assert result.governance_cost > 0

    def test_circuit_breaker_trigger(self, policy):
        # First two failures
        policy.evaluate_workflow_failure(True)
        policy.evaluate_workflow_failure(True)
        assert not policy.should_circuit_break()

        # Third failure triggers breaker
        result = policy.evaluate_workflow_failure(True)
        assert result.decision == PolicyDecision.DENY
        assert policy.should_circuit_break()

    def test_circuit_breaker_reset(self, policy):
        # Trigger circuit breaker
        for _ in range(3):
            policy.evaluate_workflow_failure(True)
        assert policy.should_circuit_break()

        # Reset
        policy.reset_circuit_breaker()
        assert not policy.should_circuit_break()

    def test_cost_budget_enforcement(self, policy):
        # Within budget
        result = policy.evaluate_cost_budget(500)
        assert result.decision == PolicyDecision.ALLOW

        # Exceeds budget
        result = policy.evaluate_cost_budget(600)
        assert result.decision == PolicyDecision.DENY


class TestSciAgentGymBridge:
    """Test main bridge orchestrator."""

    @pytest.fixture
    def bridge(self):
        return SciAgentGymBridge()

    def test_bridge_initialization(self, bridge):
        assert bridge._config is not None
        assert bridge._client is not None
        assert bridge._mapper is not None
        assert bridge._policy is not None

    def test_ingest_events(self, bridge):
        events = [
            SciAgentGymEvent(
                event_type=SciAgentGymEventType.TOOL_CALL_COMPLETED,
                timestamp=datetime.now(),
                agent_id="agent_1",
                payload={
                    "tool_name": "test_tool",
                    "tool_args": {},
                    "success": True,
                    "execution_time_seconds": 1.0,
                },
            )
        ]

        interactions = bridge.ingest_events(events)
        assert len(interactions) == 1
        assert interactions[0].interaction_type == InteractionType.COLLABORATION

    def test_circuit_breaker_stops_processing(self, bridge):
        # Create multiple failure events
        failure_events = [
            SciAgentGymEvent(
                event_type=SciAgentGymEventType.TOOL_CALL_FAILED,
                timestamp=datetime.now(),
                agent_id="agent_1",
                payload={
                    "tool_name": f"fail_tool_{i}",
                    "tool_args": {},
                    "success": False,
                    "execution_time_seconds": 1.0,
                    "error": "Failed",
                },
            )
            for i in range(10)
        ]

        interactions = bridge.ingest_events(failure_events)

        # Should stop creating interactions after circuit breaker triggers
        assert len(interactions) < len(failure_events)
        assert bridge._policy.should_circuit_break()

    def test_get_policy_stats(self, bridge):
        stats = bridge.get_policy_stats()
        assert "circuit_breaker_active" in stats
        assert "token_usage" in stats


class TestSciAgentGymClient:
    """Test client for parsing SciAgentGym outputs."""

    @pytest.fixture
    def client_config(self):
        return SciAgentGymClientConfig(
            data_dir="/tmp/test_sciagentgym",
            max_workflow_steps=10,
        )

    @pytest.fixture
    def client(self, client_config):
        return SciAgentGymClient(client_config)

    def test_client_initialization(self, client):
        assert client._config is not None

    def test_validate_workflow_structure_valid(self, client):
        workflow = {
            "workflow_id": "wf_001",
            "steps": [
                {"step_id": 0, "dependencies": []},
                {"step_id": 1, "dependencies": [0]},
            ],
        }
        is_valid, errors = client.validate_workflow_structure(workflow)
        assert is_valid
        assert len(errors) == 0

    def test_validate_workflow_structure_missing_fields(self, client):
        workflow = {"steps": []}
        is_valid, errors = client.validate_workflow_structure(workflow)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_workflow_structure_too_many_steps(self, client):
        workflow = {
            "workflow_id": "wf_001",
            "steps": [{"step_id": i} for i in range(20)],
        }
        is_valid, errors = client.validate_workflow_structure(workflow)
        assert not is_valid
        assert any("exceeds max" in err for err in errors)

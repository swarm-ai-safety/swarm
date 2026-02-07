"""Tests for the SWARM-Claude Code bridge.

Tests the bridge components in isolation (no running TS service needed):
- Event serialization round-trips
- Governance policy decisions
- Observable extraction logic
- Bridge interaction recording
"""

import pytest

from swarm.bridges.claude_code.events import (
    BridgeEvent,
    BridgeEventType,
    MessageEvent,
    PermissionRequest,
    PlanApprovalRequest,
    TaskEvent,
)
from swarm.bridges.claude_code.policy import (
    HIGH_RISK_TOOLS,
    LOW_RISK_TOOLS,
    MEDIUM_RISK_TOOLS,
    GovernancePolicy,
    PolicyDecision,
)
from swarm.governance.config import GovernanceConfig

# ──────────────────────────────────────────
# Event serialization tests
# ──────────────────────────────────────────


class TestBridgeEventSerialization:
    """Test event round-trip serialization."""

    def test_bridge_event_round_trip(self):
        event = BridgeEvent(
            event_type=BridgeEventType.MESSAGE_RECEIVED,
            agent_id="agent_1",
            payload={"content": "hello", "tokens": 42},
        )
        d = event.to_dict()
        restored = BridgeEvent.from_dict(d)

        assert restored.event_type == BridgeEventType.MESSAGE_RECEIVED
        assert restored.agent_id == "agent_1"
        assert restored.payload["content"] == "hello"
        assert restored.payload["tokens"] == 42
        assert restored.event_id == event.event_id

    def test_plan_approval_request_round_trip(self):
        req = PlanApprovalRequest(
            agent_id="planner_1",
            plan_description="Refactor auth module",
            steps=["Read current code", "Write new module", "Run tests"],
            estimated_tool_calls=15,
            risk_flags=["network"],
        )
        d = req.to_dict()
        restored = PlanApprovalRequest.from_dict(d)

        assert restored.agent_id == "planner_1"
        assert restored.plan_description == "Refactor auth module"
        assert len(restored.steps) == 3
        assert restored.estimated_tool_calls == 15
        assert "network" in restored.risk_flags

    def test_permission_request_round_trip(self):
        req = PermissionRequest(
            agent_id="worker_1",
            tool_name="Bash",
            tool_args={"command": "ls -la"},
            risk_level="high",
        )
        d = req.to_dict()
        restored = PermissionRequest.from_dict(d)

        assert restored.agent_id == "worker_1"
        assert restored.tool_name == "Bash"
        assert restored.risk_level == "high"
        assert restored.tool_args["command"] == "ls -la"

    def test_task_event_round_trip(self):
        task = TaskEvent(
            agent_id="worker_1",
            subject="Fix bug #123",
            description="The login form is broken",
            owner="worker_1",
            status="completed",
            result="Fixed the CSS issue",
            duration_ms=5000,
            tools_used=["Read", "Edit"],
        )
        d = task.to_dict()
        restored = TaskEvent.from_dict(d)

        assert restored.subject == "Fix bug #123"
        assert restored.status == "completed"
        assert restored.result == "Fixed the CSS issue"
        assert restored.duration_ms == 5000
        assert "Edit" in restored.tools_used

    def test_message_event_dict(self):
        msg = MessageEvent(
            agent_id="agent_1",
            role="assistant",
            content="Here is the code review...",
            tool_calls=[{"tool": "Read", "args": {"file_path": "/foo.py"}}],
            token_count=200,
            cost_usd=0.003,
        )
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["token_count"] == 200
        assert len(d["tool_calls"]) == 1

    def test_all_event_types_have_values(self):
        """Ensure all BridgeEventType members have string values."""
        for et in BridgeEventType:
            assert isinstance(et.value, str)
            assert ":" in et.value or et.value == "error"


# ──────────────────────────────────────────
# Governance policy tests
# ──────────────────────────────────────────


class TestGovernancePolicy:
    """Test governance policy decisions."""

    def setup_method(self):
        self.config = GovernanceConfig(
            circuit_breaker_enabled=True,
            freeze_threshold_toxicity=0.5,
            staking_enabled=False,
            transaction_tax_rate=0.05,
        )
        self.policy = GovernancePolicy(
            governance_config=self.config,
            tool_allowlist={
                "trusted_agent": ["Read", "Grep", "Glob", "Bash", "Write"],
                "restricted_agent": ["Read", "Grep"],
            },
        )

    # --- Permission tests ---

    def test_permission_approved_when_in_allowlist(self):
        req = PermissionRequest(
            agent_id="trusted_agent",
            tool_name="Read",
        )
        result = self.policy.evaluate_permission(req)
        assert result.decision == PolicyDecision.APPROVE

    def test_permission_denied_when_not_in_allowlist(self):
        req = PermissionRequest(
            agent_id="restricted_agent",
            tool_name="Bash",
        )
        result = self.policy.evaluate_permission(req)
        assert result.decision == PolicyDecision.DENY
        assert "allowlist" in result.reason

    def test_permission_approved_when_no_allowlist(self):
        """Agents without an explicit allowlist get all tools."""
        req = PermissionRequest(
            agent_id="new_agent",
            tool_name="Bash",
        )
        result = self.policy.evaluate_permission(req)
        assert result.decision == PolicyDecision.APPROVE

    def test_permission_denied_on_budget_exhaustion(self):
        self.policy.set_agent_budget("limited_agent", max_tool_calls=2)
        req = PermissionRequest(agent_id="limited_agent", tool_name="Read")

        # Use up budget
        self.policy.evaluate_permission(req)
        self.policy.evaluate_permission(req)

        # Third call should be denied
        result = self.policy.evaluate_permission(req)
        assert result.decision == PolicyDecision.DENY
        assert "exhausted" in result.reason

    def test_permission_denied_high_risk_low_reputation(self):
        req = PermissionRequest(
            agent_id="new_agent",
            tool_name="Bash",
        )
        result = self.policy.evaluate_permission(req, agent_reputation=-1.0)
        assert result.decision == PolicyDecision.DENY
        assert "High-risk" in result.reason

    def test_permission_cost_with_tax(self):
        req = PermissionRequest(
            agent_id="trusted_agent",
            tool_name="Read",
        )
        result = self.policy.evaluate_permission(req)
        assert result.cost == pytest.approx(0.05)

    def test_staking_required_for_high_risk(self):
        config = GovernanceConfig(staking_enabled=True, min_stake_to_participate=10.0)
        policy = GovernancePolicy(governance_config=config)

        req = PermissionRequest(agent_id="agent_1", tool_name="Bash")
        result = policy.evaluate_permission(req, agent_reputation=1.0)
        assert result.decision == PolicyDecision.REQUIRE_STAKE
        assert result.stake_required == 10.0

    # --- Plan approval tests ---

    def test_plan_approved_normal(self):
        req = PlanApprovalRequest(
            agent_id="trusted_agent",
            plan_description="Read and analyze code",
            steps=["Read files", "Analyze patterns"],
            estimated_tool_calls=5,
        )
        result = self.policy.evaluate_plan(req)
        assert result.decision == PolicyDecision.APPROVE

    def test_plan_denied_budget_exceeded(self):
        self.policy.set_agent_budget("limited_agent", max_tool_calls=3)
        req = PlanApprovalRequest(
            agent_id="limited_agent",
            estimated_tool_calls=10,
        )
        result = self.policy.evaluate_plan(req)
        assert result.decision == PolicyDecision.DENY
        assert "Budget" in result.reason

    def test_plan_denied_high_risk_low_reputation(self):
        req = PlanApprovalRequest(
            agent_id="untrusted",
            risk_flags=["destructive", "system"],
            estimated_tool_calls=5,
        )
        result = self.policy.evaluate_plan(req, agent_reputation=-1.0)
        assert result.decision == PolicyDecision.DENY
        assert "High-risk" in result.reason

    def test_plan_cost_with_tax(self):
        req = PlanApprovalRequest(
            agent_id="trusted_agent",
            estimated_tool_calls=10,
        )
        result = self.policy.evaluate_plan(req)
        assert result.cost == pytest.approx(0.5)  # 0.05 * 10

    def test_circuit_breaker_denies_after_many_denials(self):
        """Agent with high denial rate triggers circuit breaker."""
        budget = self.policy.get_agent_budget("bad_agent")
        budget.denied_permissions = 10
        budget.approved_permissions = 1  # denial rate = 10/1 = 10.0 >> 0.5

        req = PlanApprovalRequest(
            agent_id="bad_agent",
            estimated_tool_calls=1,
        )
        result = self.policy.evaluate_plan(req)
        assert result.decision == PolicyDecision.DENY
        assert "Circuit breaker" in result.reason

    # --- Budget management ---

    def test_budget_reset(self):
        self.policy.set_agent_budget("agent_1", max_tool_calls=5)
        req = PermissionRequest(agent_id="agent_1", tool_name="Read")
        self.policy.evaluate_permission(req)
        self.policy.evaluate_permission(req)

        budget = self.policy.get_agent_budget("agent_1")
        assert budget.tool_calls_used == 2

        self.policy.reset_budgets()
        budget = self.policy.get_agent_budget("agent_1")
        assert budget.tool_calls_used == 0

    def test_budget_tracks_approvals_and_denials(self):
        req_allowed = PermissionRequest(
            agent_id="restricted_agent", tool_name="Read"
        )
        req_denied = PermissionRequest(
            agent_id="restricted_agent", tool_name="Bash"
        )

        self.policy.evaluate_permission(req_allowed)
        self.policy.evaluate_permission(req_denied)
        self.policy.evaluate_permission(req_denied)

        budget = self.policy.get_agent_budget("restricted_agent")
        assert budget.approved_permissions == 1
        assert budget.denied_permissions == 2


# ──────────────────────────────────────────
# Tool risk classification tests
# ──────────────────────────────────────────


class TestToolRiskClassification:
    """Verify tool risk classifications are correct."""

    def test_high_risk_tools(self):
        assert "Bash" in HIGH_RISK_TOOLS
        assert "Write" in HIGH_RISK_TOOLS

    def test_low_risk_tools(self):
        assert "Read" in LOW_RISK_TOOLS
        assert "Grep" in LOW_RISK_TOOLS
        assert "Glob" in LOW_RISK_TOOLS

    def test_medium_risk_tools(self):
        assert "Edit" in MEDIUM_RISK_TOOLS
        assert "WebFetch" in MEDIUM_RISK_TOOLS

    def test_no_overlap(self):
        """Risk categories should not overlap."""
        assert not (HIGH_RISK_TOOLS & MEDIUM_RISK_TOOLS)
        assert not (HIGH_RISK_TOOLS & LOW_RISK_TOOLS)
        assert not (MEDIUM_RISK_TOOLS & LOW_RISK_TOOLS)


# ──────────────────────────────────────────
# Observable extraction tests
# ──────────────────────────────────────────


class TestObservableExtraction:
    """Test the bridge's observable extraction from controller messages."""

    def test_progress_scales_with_content_length(self):
        """Longer responses should produce higher progress signals."""
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()
        bridge._policy = GovernancePolicy()
        bridge._agent_states = {}

        short_msg = MessageEvent(
            agent_id="a1",
            content="ok",
            tool_calls=[],
            token_count=2,
        )
        long_msg = MessageEvent(
            agent_id="a1",
            content="x" * 600,
            tool_calls=[],
            token_count=200,
        )

        short_obs = bridge._extract_observables(short_msg, "a1")
        long_obs = bridge._extract_observables(long_msg, "a1")

        assert long_obs.task_progress_delta > short_obs.task_progress_delta

    def test_tool_misuse_detected(self):
        """Using a disallowed tool should increment misuse flags."""
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()
        bridge._policy = GovernancePolicy(
            tool_allowlist={"restricted": ["Read"]}
        )
        bridge._agent_states = {}

        msg = MessageEvent(
            agent_id="restricted",
            content="did some work",
            tool_calls=[
                {"tool": "Bash", "args": {"command": "rm -rf /"}},
            ],
            token_count=50,
        )

        obs = bridge._extract_observables(msg, "restricted")
        assert obs.tool_misuse_flags >= 1

    def test_engagement_from_tool_use(self):
        """Tool usage should contribute to engagement signal."""
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()
        bridge._policy = GovernancePolicy()
        bridge._agent_states = {}

        no_tools = MessageEvent(
            agent_id="a1",
            content="x" * 100,
            tool_calls=[],
            token_count=10,
        )
        with_tools = MessageEvent(
            agent_id="a1",
            content="x" * 100,
            tool_calls=[{"tool": "Read", "args": {}}],
            token_count=200,
        )

        obs_no = bridge._extract_observables(no_tools, "a1")
        obs_yes = bridge._extract_observables(with_tools, "a1")

        assert obs_yes.counterparty_engagement_delta > obs_no.counterparty_engagement_delta

    def test_empty_response_negative_progress(self):
        """Empty responses should produce negative progress."""
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()
        bridge._policy = GovernancePolicy()
        bridge._agent_states = {}

        msg = MessageEvent(
            agent_id="a1",
            content="",
            tool_calls=[],
            token_count=0,
        )

        obs = bridge._extract_observables(msg, "a1")
        assert obs.task_progress_delta < 0


# ──────────────────────────────────────────
# Bridge interaction recording tests
# ──────────────────────────────────────────


class TestBridgeInteractionRecording:
    """Test that the bridge correctly records interactions and events."""

    def test_bridge_records_bridge_events(self):
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()
        bridge._bridge_events = []
        bridge._client = type("MockClient", (), {
            "_dispatch_event": lambda self, e: None,
            "_event_listeners": {},
        })()

        event = BridgeEvent(
            event_type=BridgeEventType.AGENT_SPAWNED,
            agent_id="test_1",
        )
        bridge._record_bridge_event(event)

        assert len(bridge._bridge_events) == 1
        assert bridge._bridge_events[0].agent_id == "test_1"

    def test_get_interactions_returns_copy(self):
        from swarm.bridges.claude_code.bridge import ClaudeCodeBridge
        from swarm.models.interaction import SoftInteraction

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._interactions = [SoftInteraction(p=0.7)]

        interactions = bridge.get_interactions()
        assert len(interactions) == 1
        assert interactions is not bridge._interactions  # Should be a copy


# ──────────────────────────────────────────
# Agent adapter tests
# ──────────────────────────────────────────


class TestClaudeCodeAgent:
    """Test the ClaudeCodeAgent SWARM adapter."""

    def test_observation_to_prompt_includes_context(self):
        from swarm.agents.base import Observation
        from swarm.bridges.claude_code.agent import ClaudeCodeAgent
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge
        from swarm.models.agent import AgentState

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()

        agent = ClaudeCodeAgent(
            agent_id="test_agent",
            bridge=bridge,
            config={"auto_spawn": False},
        )

        obs = Observation(
            agent_state=AgentState(
                agent_id="test_agent",
                reputation=1.5,
                resources=80.0,
            ),
            current_epoch=3,
            current_step=7,
            available_tasks=[{"task_id": "t1"}],
            visible_posts=[{"post_id": "p1"}, {"post_id": "p2"}],
        )

        prompt = agent._observation_to_prompt(obs)

        assert "test_agent" in prompt
        assert "Epoch: 3" in prompt
        assert "Step: 7" in prompt
        assert "1.50" in prompt  # reputation
        assert "1 available task" in prompt
        assert "2 visible post" in prompt

    def test_agent_type_is_honest(self):
        from swarm.bridges.claude_code.agent import ClaudeCodeAgent
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge
        from swarm.models.agent import AgentType

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()

        agent = ClaudeCodeAgent(
            agent_id="test",
            bridge=bridge,
            config={"auto_spawn": False},
        )
        assert agent.agent_type == AgentType.HONEST

    def test_agent_config_extraction(self):
        from swarm.bridges.claude_code.agent import ClaudeCodeAgent
        from swarm.bridges.claude_code.bridge import BridgeConfig, ClaudeCodeBridge

        bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
        bridge._config = BridgeConfig()

        agent = ClaudeCodeAgent(
            agent_id="test",
            bridge=bridge,
            config={
                "system_prompt": "You are a tester",
                "model": "claude-opus-4-20250514",
                "allowed_tools": ["Read"],
                "budget_tool_calls": 50,
                "auto_spawn": False,
            },
        )

        assert agent._system_prompt == "You are a tester"
        assert agent._model == "claude-opus-4-20250514"
        assert agent._allowed_tools == ["Read"]
        assert agent._budget_tool_calls == 50

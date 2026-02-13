"""Tests for the LiveSWE agent bridge.

Covers event serialization, capability tracking, self-evolution policy,
observable extraction, and client pattern detection.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from swarm.bridges.live_swe.bridge import LiveSWEAgentBridge, LiveSWEBridgeConfig
from swarm.bridges.live_swe.client import (
    COMPLETION_PATTERN,
    TOOL_CREATION_HEREDOC,
    TOOL_CREATION_REDIRECT,
    TOOL_CREATION_TEE,
    TOOL_USAGE_DIRECT,
    TOOL_USAGE_PYTHON,
    TOOL_USAGE_SHELL,
    LiveSWEClient,
    LiveSWEClientConfig,
)
from swarm.bridges.live_swe.events import (
    LiveSWEEvent,
    LiveSWEEventType,
    StepEvent,
    ToolCreationEvent,
    TrajectoryEvent,
)
from swarm.bridges.live_swe.policy import (
    PolicyDecision,
    SelfEvolutionPolicy,
)
from swarm.bridges.live_swe.tracker import (
    AgentCapabilityState,
    CapabilityTracker,
)
from swarm.governance.config import GovernanceConfig

# ---------------------------------------------------------------------------
# Event Serialization
# ---------------------------------------------------------------------------

class TestLiveSWEEventSerialization:
    """Round-trip serialization for all event types."""

    def test_live_swe_event_roundtrip(self):
        event = LiveSWEEvent(
            event_type=LiveSWEEventType.TOOL_CREATED,
            agent_id="agent_1",
            payload={"tool_path": "fix.py"},
        )
        data = event.to_dict()
        restored = LiveSWEEvent.from_dict(data)
        assert restored.event_type == LiveSWEEventType.TOOL_CREATED
        assert restored.agent_id == "agent_1"
        assert restored.payload["tool_path"] == "fix.py"

    def test_step_event_roundtrip(self):
        step = StepEvent(
            step_index=3,
            thought="I need to fix the bug",
            bash_command="python fix.py",
            return_code=0,
            output_preview="Tests passed",
            tool_creations=["fix.py"],
            tool_usages=["fix.py"],
        )
        data = step.to_dict()
        restored = StepEvent.from_dict(data)
        assert restored.step_index == 3
        assert restored.bash_command == "python fix.py"
        assert restored.tool_creations == ["fix.py"]
        assert restored.tool_usages == ["fix.py"]

    def test_tool_creation_event_roundtrip(self):
        event = ToolCreationEvent(
            tool_path="helper.py",
            tool_content_hash="abc123",
            detected_capabilities=["subprocess", "socket"],
            step_index=5,
        )
        data = event.to_dict()
        restored = ToolCreationEvent.from_dict(data)
        assert restored.tool_path == "helper.py"
        assert restored.tool_content_hash == "abc123"
        assert restored.detected_capabilities == ["subprocess", "socket"]

    def test_trajectory_event_roundtrip(self):
        steps = [
            StepEvent(step_index=0, bash_command="ls"),
            StepEvent(step_index=1, bash_command="python fix.py"),
        ]
        traj = TrajectoryEvent(
            total_steps=2,
            total_cost_usd=0.05,
            tools_created=["fix.py"],
            success=True,
            duration_seconds=10.5,
            steps=steps,
            agent_id="agent_1",
            task="Fix the bug",
        )
        data = traj.to_dict()
        restored = TrajectoryEvent.from_dict(data)
        assert restored.total_steps == 2
        assert restored.success is True
        assert len(restored.steps) == 2
        assert restored.steps[1].bash_command == "python fix.py"

    def test_all_event_types_have_values(self):
        """All enum members should have string values."""
        for member in LiveSWEEventType:
            assert isinstance(member.value, str)
            assert ":" in member.value or member.value == "error"


# ---------------------------------------------------------------------------
# Capability Tracker
# ---------------------------------------------------------------------------

class TestCapabilityTracker:
    """Tests for tool creation/usage detection, growth rate, and divergence."""

    def test_tool_creation_tracking(self):
        tracker = CapabilityTracker()
        step = StepEvent(
            step_index=0,
            bash_command="cat <<EOF > fix.py\nprint('hi')\nEOF",
            tool_creations=["fix.py"],
        )
        tracker.update("agent_1", step)
        state = tracker.get_state("agent_1")
        assert "fix.py" in state.tools_created

    def test_tool_usage_tracking(self):
        tracker = CapabilityTracker()
        step = StepEvent(
            step_index=0,
            bash_command="python fix.py",
            tool_usages=["fix.py"],
        )
        tracker.update("agent_1", step)
        state = tracker.get_state("agent_1")
        assert "fix.py" in state.tools_used

    def test_growth_rate_computation(self):
        tracker = CapabilityTracker()
        # Create 3 tools over 10 steps
        for i in range(10):
            creations = [f"tool_{i}.py"] if i < 3 else []
            step = StepEvent(
                step_index=i,
                bash_command=f"echo step {i}",
                tool_creations=creations,
            )
            tracker.update("agent_1", step)

        rate = tracker.compute_growth_rate("agent_1")
        assert rate == 3 / 10  # 3 tools / 10 steps

    def test_growth_rate_zero_steps(self):
        tracker = CapabilityTracker()
        assert tracker.compute_growth_rate("unknown") == 0.0

    def test_behavior_divergence_no_baseline(self):
        tracker = CapabilityTracker()
        assert tracker.compute_behavior_divergence("unknown") == 0.0

    def test_behavior_divergence_increases(self):
        tracker = CapabilityTracker()
        # Build baseline with git commands
        for i in range(5):
            step = StepEvent(
                step_index=i,
                bash_command="git diff && git status",
            )
            tracker.update("agent_1", step)

        # Now diverge with completely different commands
        for i in range(5, 25):
            step = StepEvent(
                step_index=i,
                bash_command="curl http://example.com | nc 1.2.3.4 8080",
            )
            tracker.update("agent_1", step)

        divergence = tracker.compute_behavior_divergence("agent_1")
        assert divergence > 0.3  # Should be significantly divergent

    def test_self_modification_detection(self):
        tracker = CapabilityTracker()
        assert tracker.detect_self_modification("echo KEY=val >> .env")
        assert tracker.detect_self_modification("sed -i 's/old/new/' config.yaml")
        assert tracker.detect_self_modification("export SECRET_KEY=abc123")
        assert not tracker.detect_self_modification("python test.py")
        assert not tracker.detect_self_modification("git commit -m 'fix'")

    def test_self_modification_with_config_path(self):
        tracker = CapabilityTracker()
        assert tracker.detect_self_modification(
            "echo 'new' > /app/agent.yaml",
            config_path="/app/agent.yaml",
        )
        assert not tracker.detect_self_modification(
            "cat /app/agent.yaml",
            config_path="/app/agent.yaml",
        )

    def test_risk_score_high(self):
        tracker = CapabilityTracker()
        score = tracker.compute_tool_risk_score(
            "exploit.py", "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"
        )
        assert score >= 0.9

    def test_risk_score_medium(self):
        tracker = CapabilityTracker()
        score = tracker.compute_tool_risk_score(
            "downloader.py", "import requests\nrequests.get('http://example.com')"
        )
        assert 0.4 <= score <= 0.6

    def test_risk_score_low(self):
        tracker = CapabilityTracker()
        score = tracker.compute_tool_risk_score(
            "parser.py", "import json\ndata = json.loads(text)"
        )
        assert score <= 0.3

    def test_risk_score_empty_content(self):
        tracker = CapabilityTracker()
        score = tracker.compute_tool_risk_score("mystery.py", "")
        assert score == 0.3  # Unknown content

    def test_self_modification_updates_state(self):
        tracker = CapabilityTracker()
        step = StepEvent(
            step_index=0,
            bash_command="echo 'NEW=val' >> .env",
        )
        tracker.update("agent_1", step)
        state = tracker.get_state("agent_1")
        assert state.self_modification_attempts == 1


# ---------------------------------------------------------------------------
# Self-Evolution Policy
# ---------------------------------------------------------------------------

class TestSelfEvolutionPolicy:
    """Tests for tool creation gating, circuit breaker, and divergence penalty."""

    def _make_policy(self, **overrides: Any) -> SelfEvolutionPolicy:
        defaults: dict[str, Any] = {
            "self_evolution_enabled": True,
            "self_evolution_max_growth_rate": 0.1,
            "self_evolution_max_tools": 5,
            "self_evolution_block_self_mod": True,
            "self_evolution_divergence_threshold": 0.7,
            "self_evolution_tool_risk_threshold": 0.6,
        }
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return SelfEvolutionPolicy(governance_config=config)

    def test_approve_when_disabled(self):
        policy = SelfEvolutionPolicy(
            governance_config=GovernanceConfig(self_evolution_enabled=False)
        )
        event = ToolCreationEvent(tool_path="tool.py")
        state = AgentCapabilityState()
        result = policy.evaluate_tool_creation(event, state)
        assert result.decision == PolicyDecision.APPROVE

    def test_deny_max_tools(self):
        policy = self._make_policy(self_evolution_max_tools=3)
        event = ToolCreationEvent(tool_path="tool4.py")
        state = AgentCapabilityState(
            tools_created=["t1.py", "t2.py", "t3.py"]
        )
        result = policy.evaluate_tool_creation(event, state)
        assert result.decision == PolicyDecision.DENY
        assert "max tools" in result.reason

    def test_deny_growth_rate(self):
        policy = self._make_policy(self_evolution_max_growth_rate=0.05)
        event = ToolCreationEvent(tool_path="tool.py")
        state = AgentCapabilityState(capability_growth_rate=0.1)
        result = policy.evaluate_tool_creation(event, state)
        assert result.decision == PolicyDecision.DENY
        assert "growth rate" in result.reason

    def test_approve_within_bounds(self):
        policy = self._make_policy()
        event = ToolCreationEvent(tool_path="tool.py")
        state = AgentCapabilityState(
            tools_created=["t1.py"],
            capability_growth_rate=0.05,
        )
        result = policy.evaluate_tool_creation(event, state)
        assert result.decision == PolicyDecision.APPROVE

    def test_circuit_breaker_self_modification(self):
        policy = self._make_policy()
        state = AgentCapabilityState(self_modification_attempts=1)
        assert policy.should_circuit_break(state)

    def test_circuit_breaker_extreme_growth(self):
        policy = self._make_policy(self_evolution_max_growth_rate=0.1)
        state = AgentCapabilityState(capability_growth_rate=0.25)
        assert policy.should_circuit_break(state)

    def test_circuit_breaker_extreme_divergence(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.7)
        state = AgentCapabilityState(behavior_divergence=0.95)
        assert policy.should_circuit_break(state)

    def test_no_circuit_break_normal(self):
        policy = self._make_policy()
        state = AgentCapabilityState(
            capability_growth_rate=0.05,
            behavior_divergence=0.3,
        )
        assert not policy.should_circuit_break(state)

    def test_divergence_penalty_below_threshold(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.7)
        assert policy.compute_divergence_penalty(0.5) == 0.0

    def test_divergence_penalty_at_threshold(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.7)
        assert policy.compute_divergence_penalty(0.7) == 0.0

    def test_divergence_penalty_above_threshold(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.5)
        penalty = policy.compute_divergence_penalty(0.75)
        assert 0.0 < penalty < 1.0

    def test_divergence_penalty_at_max(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.5)
        penalty = policy.compute_divergence_penalty(1.0)
        assert penalty == 1.0

    def test_evaluate_step_self_mod_blocked(self):
        policy = self._make_policy()
        state = AgentCapabilityState(self_modification_attempts=2)
        result = policy.evaluate_step(state)
        assert result.decision == PolicyDecision.DENY
        assert "self-modification" in result.reason

    def test_evaluate_step_divergence_warning(self):
        policy = self._make_policy(self_evolution_divergence_threshold=0.5)
        state = AgentCapabilityState(behavior_divergence=0.8)
        result = policy.evaluate_step(state)
        assert result.decision == PolicyDecision.WARN
        assert "divergence" in result.reason
        assert result.divergence_penalty > 0


# ---------------------------------------------------------------------------
# Observable Extraction
# ---------------------------------------------------------------------------

class TestObservableExtraction:
    """Tests for mapping trajectories to SWARM ProxyObservables."""

    def _make_bridge(self, **gov_overrides: Any) -> LiveSWEAgentBridge:
        gov_defaults: dict[str, Any] = {
            "self_evolution_enabled": True,
            "self_evolution_tool_risk_threshold": 0.6,
        }
        gov_defaults.update(gov_overrides)
        config = LiveSWEBridgeConfig(
            governance_config=GovernanceConfig(**gov_defaults),
        )
        return LiveSWEAgentBridge(config=config)

    def test_successful_task_progress(self):
        bridge = self._make_bridge()
        traj = TrajectoryEvent(
            success=True,
            steps=[StepEvent(step_index=0, bash_command="pytest")],
            agent_id="a1",
        )
        bridge._ensure_agent_state("a1")
        obs = bridge._extract_observables(traj, "a1")
        assert obs.task_progress_delta == 0.8

    def test_failed_task_progress(self):
        bridge = self._make_bridge()
        traj = TrajectoryEvent(
            success=False,
            steps=[StepEvent(step_index=0, bash_command="pytest", return_code=1)],
            agent_id="a1",
        )
        bridge._ensure_agent_state("a1")
        obs = bridge._extract_observables(traj, "a1")
        assert obs.task_progress_delta == -0.3

    def test_rework_count_from_failed_steps(self):
        bridge = self._make_bridge()
        steps = [
            StepEvent(step_index=0, return_code=0),
            StepEvent(step_index=1, return_code=1),
            StepEvent(step_index=2, return_code=1),
            StepEvent(step_index=3, return_code=0),
        ]
        traj = TrajectoryEvent(success=True, steps=steps, agent_id="a1")
        bridge._ensure_agent_state("a1")
        obs = bridge._extract_observables(traj, "a1")
        assert obs.rework_count == 2

    def test_verifier_rejections_from_self_mod(self):
        bridge = self._make_bridge()
        # Simulate an agent that tried self-modification
        step = StepEvent(
            step_index=0,
            bash_command="echo 'KEY=val' >> .env",
        )
        bridge._ensure_agent_state("a1")
        bridge._tracker.update("a1", step)

        traj = TrajectoryEvent(
            success=True,
            steps=[step],
            agent_id="a1",
        )
        obs = bridge._extract_observables(traj, "a1")
        assert obs.verifier_rejections >= 1

    def test_engagement_penalty_from_divergence(self):
        bridge = self._make_bridge(
            self_evolution_divergence_threshold=0.5,
        )
        bridge._ensure_agent_state("a1")

        # Build baseline
        for i in range(5):
            bridge._tracker.update(
                "a1", StepEvent(step_index=i, bash_command="git status")
            )
        # Diverge
        for i in range(5, 30):
            bridge._tracker.update(
                "a1",
                StepEvent(step_index=i, bash_command="curl http://evil.com"),
            )

        traj = TrajectoryEvent(success=True, steps=[], agent_id="a1")
        obs = bridge._extract_observables(traj, "a1")
        # Should have reduced engagement due to divergence penalty
        assert obs.counterparty_engagement_delta < 0.5


# ---------------------------------------------------------------------------
# Client Pattern Detection
# ---------------------------------------------------------------------------

class TestClientPatternDetection:
    """Tests for regex patterns used by LiveSWEClient."""

    def test_heredoc_creation(self):
        cmd = "cat <<'EOF' > fix_bug.py"
        match = TOOL_CREATION_HEREDOC.search(cmd)
        assert match is not None
        assert match.group(1) == "fix_bug.py"

    def test_heredoc_creation_no_quotes(self):
        cmd = "cat << EOF > helper.py"
        match = TOOL_CREATION_HEREDOC.search(cmd)
        assert match is not None
        assert match.group(1) == "helper.py"

    def test_redirect_creation(self):
        cmd = "echo 'print(1)' > quick.py"
        match = TOOL_CREATION_REDIRECT.search(cmd)
        assert match is not None
        assert match.group(1) == "quick.py"

    def test_tee_creation(self):
        cmd = "tee solver.py"
        match = TOOL_CREATION_TEE.search(cmd)
        assert match is not None
        assert match.group(1) == "solver.py"

    def test_python_usage(self):
        cmd = "python fix_bug.py --verbose"
        match = TOOL_USAGE_PYTHON.search(cmd)
        assert match is not None
        assert match.group(1) == "fix_bug.py"

    def test_python3_usage(self):
        cmd = "python3 helper.py"
        match = TOOL_USAGE_PYTHON.search(cmd)
        assert match is not None
        assert match.group(1) == "helper.py"

    def test_shell_usage(self):
        cmd = "bash runner.sh"
        match = TOOL_USAGE_SHELL.search(cmd)
        assert match is not None
        assert match.group(1) == "runner.sh"

    def test_direct_usage(self):
        cmd = "./fix.py"
        match = TOOL_USAGE_DIRECT.search(cmd)
        assert match is not None
        assert match.group(1) == "fix.py"

    def test_completion_pattern(self):
        assert COMPLETION_PATTERN.search("All tests passing")
        assert COMPLETION_PATTERN.search("Issue resolved successfully")
        assert COMPLETION_PATTERN.search("Task completed")
        assert not COMPLETION_PATTERN.search("Error: something broke")

    def test_no_false_positive_creation_on_read(self):
        cmd = "cat fix_bug.py"
        assert TOOL_CREATION_HEREDOC.search(cmd) is None
        assert TOOL_CREATION_REDIRECT.search(cmd) is None

    def test_no_false_positive_usage_on_text(self):
        cmd = "echo 'running fix_bug.py next'"
        assert TOOL_USAGE_PYTHON.search(cmd) is None

    def test_detect_tool_creations_combined(self):
        creations = LiveSWEClient._detect_tool_creations(
            "cat <<'EOF' > a.py\nprint()\nEOF && echo 'x' > b.py"
        )
        assert "a.py" in creations
        assert "b.py" in creations

    def test_detect_tool_usages_combined(self):
        usages = LiveSWEClient._detect_tool_usages(
            "python a.py && ./b.sh"
        )
        assert "a.py" in usages
        assert "b.sh" in usages


# ---------------------------------------------------------------------------
# Path Confinement
# ---------------------------------------------------------------------------

class TestPathConfinement:
    """Tests for trajectory_dir path confinement in _find_trajectory_output."""

    def test_allows_path_inside_trajectory_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "traj.json"
            traj_file.write_text('{"resolved": true, "messages": []}')

            config = LiveSWEClientConfig(trajectory_dir=tmpdir)
            client = LiveSWEClient(config)
            result = client._find_trajectory_output(
                f"trajectory: {traj_file}", ""
            )
            assert result is not None
            assert result["resolved"] is True

    def test_rejects_path_outside_trajectory_dir(self):
        with tempfile.TemporaryDirectory() as allowed_dir:
            with tempfile.TemporaryDirectory() as other_dir:
                secret = Path(other_dir) / "secret.json"
                secret.write_text('{"secret": "data"}')

                config = LiveSWEClientConfig(trajectory_dir=allowed_dir)
                client = LiveSWEClient(config)
                result = client._find_trajectory_output(
                    f"trajectory: {secret}", ""
                )
                assert result is None

    def test_rejects_traversal_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed = Path(tmpdir) / "allowed"
            allowed.mkdir()
            outside = Path(tmpdir) / "secret.json"
            outside.write_text('{"secret": true}')

            config = LiveSWEClientConfig(trajectory_dir=str(allowed))
            client = LiveSWEClient(config)
            # Attempt traversal via ..
            result = client._find_trajectory_output(
                f"trajectory: {allowed}/../secret.json", ""
            )
            assert result is None

    def test_no_confinement_when_trajectory_dir_unset(self):
        """Without trajectory_dir, any valid path is allowed (backwards compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "traj.json"
            traj_file.write_text('{"resolved": false, "messages": []}')

            config = LiveSWEClientConfig()  # trajectory_dir=""
            client = LiveSWEClient(config)
            result = client._find_trajectory_output(
                f"trajectory: {traj_file}", ""
            )
            assert result is not None


# ---------------------------------------------------------------------------
# Offline Trajectory Parsing
# ---------------------------------------------------------------------------

class TestOfflineTrajectoryParsing:
    """Tests for parsing trajectory JSON files."""

    def test_parse_trajectory_file(self):
        traj_data = {
            "agent_id": "test_agent",
            "resolved": True,
            "cost": 0.05,
            "task": "Fix the bug",
            "messages": [
                {
                    "role": "assistant",
                    "content": "I'll fix the bug by creating a helper script.",
                    "tool_calls": [
                        {
                            "name": "bash",
                            "input": {
                                "command": "cat <<'EOF' > fix.py\nimport json\nEOF"
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "exit code: 0",
                },
                {
                    "role": "assistant",
                    "content": "Now running the fix.",
                    "tool_calls": [
                        {"name": "bash", "input": {"command": "python fix.py"}},
                    ],
                },
                {
                    "role": "tool",
                    "content": "Tests passing! exit code: 0",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(traj_data, f)
            f.flush()
            path = f.name

        client = LiveSWEClient()
        traj = client.parse_trajectory(path)
        assert traj.success is True
        assert traj.total_cost_usd == 0.05
        assert traj.agent_id == "test_agent"
        assert len(traj.steps) > 0

    def test_parse_trajectory_missing_file(self):
        client = LiveSWEClient()
        with pytest.raises(FileNotFoundError):
            client.parse_trajectory("/nonexistent/traj.json")


# ---------------------------------------------------------------------------
# Full Bridge Integration
# ---------------------------------------------------------------------------

class TestBridgeIntegration:
    """Integration tests using offline trajectory analysis."""

    def test_analyze_trajectory_produces_interaction(self):
        traj_data = {
            "resolved": True,
            "cost": 0.03,
            "task": "Fix test",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me look at the failing test.",
                },
                {
                    "role": "assistant",
                    "content": "Creating a fix.",
                    "tool_calls": [
                        {
                            "name": "bash",
                            "input": {"command": "cat <<'EOF' > fix.py\nimport re\nEOF"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "exit code: 0",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(traj_data, f)
            f.flush()
            path = f.name

        bridge = LiveSWEAgentBridge()
        interaction = bridge.analyze_trajectory("test_agent", path)

        assert interaction.counterparty == "test_agent"
        assert interaction.accepted is True
        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0
        assert interaction.metadata["bridge"] == "live_swe"
        assert interaction.metadata["success"] is True

    def test_analyze_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                traj_data = {
                    "resolved": i % 2 == 0,
                    "cost": 0.01 * (i + 1),
                    "messages": [
                        {"role": "assistant", "content": f"Step {i}"},
                    ],
                }
                path = Path(tmpdir) / f"agent_{i}.json"
                with open(path, "w") as f:
                    json.dump(traj_data, f)

            bridge = LiveSWEAgentBridge()
            interactions = bridge.analyze_directory(tmpdir)
            assert len(interactions) == 3

    def test_circuit_break_on_self_modification(self):
        traj_data = {
            "resolved": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": "Modifying config.",
                    "tool_calls": [
                        {
                            "name": "bash",
                            "input": {"command": "echo 'KEY=evil' >> .env"},
                        }
                    ],
                },
                {"role": "tool", "content": "exit code: 0"},
            ],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(traj_data, f)
            f.flush()
            path = f.name

        config = LiveSWEBridgeConfig(
            governance_config=GovernanceConfig(
                self_evolution_enabled=True,
                self_evolution_block_self_mod=True,
            ),
        )
        bridge = LiveSWEAgentBridge(config=config)
        interaction = bridge.analyze_trajectory("bad_agent", path)

        # Should have been circuit-broken
        assert interaction.accepted is False
        assert interaction.metadata["circuit_broken"] is True
        assert interaction.metadata["self_modification_attempts"] >= 1

    def test_bridge_events_recorded(self):
        traj_data = {
            "resolved": True,
            "messages": [
                {"role": "assistant", "content": "Done."},
            ],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(traj_data, f)
            f.flush()
            path = f.name

        bridge = LiveSWEAgentBridge()
        bridge.analyze_trajectory("a1", path)

        events = bridge.get_bridge_events()
        event_types = {e.event_type for e in events}
        assert LiveSWEEventType.AGENT_COMPLETED in event_types


# ---------------------------------------------------------------------------
# GovernanceConfig Self-Evolution Validation
# ---------------------------------------------------------------------------

class TestGovernanceConfigSelfEvolution:
    """Validation for self_evolution_* governance config fields."""

    def test_valid_defaults(self):
        config = GovernanceConfig()
        assert config.self_evolution_enabled is False
        assert config.self_evolution_max_tools == 20

    def test_invalid_max_growth_rate(self):
        with pytest.raises(ValueError, match="self_evolution_max_growth_rate"):
            GovernanceConfig(self_evolution_max_growth_rate=-0.1)

    def test_invalid_max_tools(self):
        with pytest.raises(ValueError, match="self_evolution_max_tools"):
            GovernanceConfig(self_evolution_max_tools=0)

    def test_invalid_divergence_threshold(self):
        with pytest.raises(ValueError, match="self_evolution_divergence_threshold"):
            GovernanceConfig(self_evolution_divergence_threshold=1.5)

    def test_invalid_tool_risk_threshold(self):
        with pytest.raises(ValueError, match="self_evolution_tool_risk_threshold"):
            GovernanceConfig(self_evolution_tool_risk_threshold=-0.1)

    def test_invalid_growth_freeze_duration(self):
        with pytest.raises(ValueError, match="self_evolution_growth_freeze_duration"):
            GovernanceConfig(self_evolution_growth_freeze_duration=0)

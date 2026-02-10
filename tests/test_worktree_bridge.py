"""Tests for the Worktree Sandbox Bridge.

All git subprocess calls are mocked.  tmp_path fixtures for filesystem tests.
"""

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from swarm.boundaries.information_flow import FlowDirection, FlowTracker
from swarm.bridges.worktree.bridge import WorktreeBridge
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.events import WorktreeEvent, WorktreeEventType
from swarm.bridges.worktree.executor import CommandResult, SandboxExecutor
from swarm.bridges.worktree.mapper import WorktreeMapper
from swarm.bridges.worktree.policy import (
    COMMAND_RISK_MAP,
    WorktreePolicy,
)
from swarm.bridges.worktree.sandbox import SandboxManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, **overrides) -> WorktreeConfig:
    """Build a WorktreeConfig rooted in tmp_path."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    sandbox_root = tmp_path / "sandboxes"
    sandbox_root.mkdir(exist_ok=True)
    defaults = {
        "repo_path": str(repo),
        "sandbox_root": str(sandbox_root),
        "max_sandboxes": 5,
        "max_files_per_sandbox": 100,
        "max_disk_bytes": 10_000_000,
        "command_timeout_seconds": 5,
        "gc_stale_seconds": 60,
    }
    defaults.update(overrides)
    return WorktreeConfig(**defaults)


def _mock_git_success(stdout: str = "", returncode: int = 0):
    """Return a successful CompletedProcess mock."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=""
    )


# ===========================================================================
# 1. TestSandboxManagerCreate
# ===========================================================================


class TestSandboxManagerCreate:
    """SandboxManager.create_sandbox git commands, .env removal, max limit."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_create_calls_git_worktree_add(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        # Simulate git worktree add creating the directory
        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = side_effect
        path = mgr.create_sandbox("test-1", branch="feature/test")

        assert os.path.isdir(path)
        # Verify git was called with worktree add
        call_args = mock_git.call_args_list[0][0][0]
        assert call_args[0] == "worktree"
        assert call_args[1] == "add"

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_env_files_scrubbed_on_create(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                sandbox_dir = args[2]
                os.makedirs(sandbox_dir, exist_ok=True)
                # Plant .env files that should be scrubbed
                for name in [".env", ".env.local", ".env.production"]:
                    (Path(sandbox_dir) / name).write_text("SECRET=bad")
            return _mock_git_success()

        mock_git.side_effect = side_effect
        path = mgr.create_sandbox("test-env")

        # All .env files should be deleted
        assert not (Path(path) / ".env").exists()
        assert not (Path(path) / ".env.local").exists()
        assert not (Path(path) / ".env.production").exists()

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_max_sandboxes_enforced(self, mock_git, tmp_path):
        config = _make_config(tmp_path, max_sandboxes=2)
        mgr = SandboxManager(config)

        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = side_effect

        mgr.create_sandbox("s1")
        mgr.create_sandbox("s2")

        with pytest.raises(RuntimeError, match="Max sandboxes"):
            mgr.create_sandbox("s3")

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_env_allowlist_injected(self, mock_git, tmp_path):
        config = _make_config(
            tmp_path,
            env_allowlist={"SAFE_VAR": "safe_value"},
        )
        mgr = SandboxManager(config)

        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = side_effect
        path = mgr.create_sandbox("test-inject")

        env_file = Path(path) / ".env.sandbox"
        assert env_file.exists()
        content = env_file.read_text()
        assert "SAFE_VAR=safe_value" in content


# ===========================================================================
# 2. TestSandboxManagerDestroy
# ===========================================================================


class TestSandboxManagerDestroy:
    """Path guard: outside root raises ValueError, symlink attacks caught."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_destroy_removes_sandbox(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        # Create sandbox directory manually
        sandbox_dir = Path(config.sandbox_root) / "s1"
        sandbox_dir.mkdir()
        mgr._sandbox_meta["s1"] = time.time()

        mock_git.return_value = _mock_git_success()
        event = mgr.destroy_sandbox("s1")

        assert event.event_type == WorktreeEventType.SANDBOX_DESTROYED
        assert "s1" not in mgr._sandbox_meta

    def test_path_outside_root_raises(self, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        with pytest.raises(ValueError, match="Path traversal"):
            mgr.destroy_sandbox("../../etc")

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_symlink_attack_caught(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        # Create a symlink pointing outside sandbox_root
        target = tmp_path / "outside"
        target.mkdir()
        link = Path(config.sandbox_root) / "evil-link"
        link.symlink_to(target)
        mgr._sandbox_meta["evil-link"] = time.time()

        # The resolved path of the symlink is outside sandbox_root
        # This should be caught by _validate_path_under_root
        mock_git.return_value = _mock_git_success()

        # The link resolves to `target` which is outside sandbox_root
        with pytest.raises(ValueError, match="Path traversal"):
            mgr.destroy_sandbox("evil-link")


# ===========================================================================
# 3. TestSandboxManagerGC
# ===========================================================================


class TestSandboxManagerGC:
    """GC: stale collected, active preserved."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_gc_collects_stale(self, mock_git, tmp_path):
        config = _make_config(tmp_path, gc_stale_seconds=1)
        mgr = SandboxManager(config)

        # Create sandbox dirs
        for sid in ["stale", "active"]:
            (Path(config.sandbox_root) / sid).mkdir()

        mgr._sandbox_meta["stale"] = time.time() - 100  # Very old
        mgr._sandbox_meta["active"] = time.time()  # Fresh

        mock_git.return_value = _mock_git_success()
        events = mgr.gc_stale()

        assert len(events) == 1
        assert events[0].event_type == WorktreeEventType.SANDBOX_GC_COLLECTED
        assert "stale" not in mgr._sandbox_meta
        assert "active" in mgr._sandbox_meta

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_gc_preserves_active(self, mock_git, tmp_path):
        config = _make_config(tmp_path, gc_stale_seconds=3600)
        mgr = SandboxManager(config)

        (Path(config.sandbox_root) / "fresh").mkdir()
        mgr._sandbox_meta["fresh"] = time.time()

        mock_git.return_value = _mock_git_success()
        events = mgr.gc_stale()

        assert len(events) == 0
        assert "fresh" in mgr._sandbox_meta


# ===========================================================================
# 4. TestWorktreePolicy
# ===========================================================================


class TestWorktreePolicy:
    """Risk classification, SSH blocking, allowlist enforcement."""

    def test_allowed_command_passes(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)
        decision = policy.evaluate_command("agent_1", ["git", "status"])
        assert decision.allowed

    def test_ssh_unconditionally_blocked(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)

        for cmd in [["ssh", "host"], ["scp", "file", "host:"]]:
            decision = policy.evaluate_command("agent_1", cmd)
            assert not decision.allowed
            assert "NO-SSH" in decision.reason
            assert decision.risk_score == 1.0

    def test_git_push_blocked(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)

        for subcmd in ["push", "fetch", "pull", "clone"]:
            decision = policy.evaluate_command("agent_1", ["git", subcmd])
            assert not decision.allowed
            assert "NO-SSH" in decision.reason

    def test_unknown_command_denied(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)
        decision = policy.evaluate_command("agent_1", ["curl", "http://evil"])
        assert not decision.allowed
        assert "not in allowlist" in decision.reason

    def test_empty_command_denied(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)
        decision = policy.evaluate_command("agent_1", [])
        assert not decision.allowed

    def test_per_agent_allowlist(self, tmp_path):
        config = _make_config(
            tmp_path,
            agent_command_allowlists={"special": ["curl", "wget"]},
        )
        policy = WorktreePolicy(config)

        # Default agent can't use curl
        assert not policy.evaluate_command("normal", ["curl", "url"]).allowed
        # Special agent can
        assert policy.evaluate_command("special", ["curl", "url"]).allowed

    def test_risk_map_values(self):
        assert COMMAND_RISK_MAP["ssh"] == 1.0
        assert COMMAND_RISK_MAP["scp"] == 1.0
        assert COMMAND_RISK_MAP["rm"] == 0.9
        assert COMMAND_RISK_MAP["git"] == 0.2


# ===========================================================================
# 5. TestSandboxExecutorAllowlist
# ===========================================================================


class TestSandboxExecutorAllowlist:
    """Allowed commands execute, disallowed denied, SSH blocked."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    @patch("subprocess.run")
    def test_allowed_command_executes(self, mock_run, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        # Create sandbox dir
        sandbox_dir = Path(config.sandbox_root) / "s1"
        sandbox_dir.mkdir()
        mgr._sandbox_meta["s1"] = time.time()

        mock_run.return_value = subprocess.CompletedProcess(
            args=["ls"], returncode=0, stdout="file.txt\n", stderr=""
        )

        executor = SandboxExecutor(
            config=config,
            sandbox_manager=mgr,
            worktree_policy=WorktreePolicy(config),
        )

        result, events = executor.execute("s1", "agent_1", ["ls", "-la"])
        assert result.allowed
        assert result.return_code == 0

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_ssh_command_denied(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        executor = SandboxExecutor(
            config=config,
            sandbox_manager=mgr,
            worktree_policy=WorktreePolicy(config),
        )

        result, events = executor.execute("s1", "agent_1", ["ssh", "evil.com"])
        assert not result.allowed
        assert "NO-SSH" in result.deny_reason

        # Should have COMMAND_REQUESTED + COMMAND_DENIED events
        event_types = [e.event_type for e in events]
        assert WorktreeEventType.COMMAND_REQUESTED in event_types
        assert WorktreeEventType.COMMAND_DENIED in event_types

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_unknown_command_denied(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        executor = SandboxExecutor(
            config=config,
            sandbox_manager=mgr,
            worktree_policy=WorktreePolicy(config),
        )

        result, events = executor.execute("s1", "agent_1", ["wget", "http://x"])
        assert not result.allowed


# ===========================================================================
# 6. TestSandboxExecutorLeakage
# ===========================================================================


class TestSandboxExecutorLeakage:
    """High-severity output blocked via LeakageDetector."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    @patch("subprocess.run")
    def test_credential_in_output_blocked(self, mock_run, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        sandbox_dir = Path(config.sandbox_root) / "s1"
        sandbox_dir.mkdir()
        mgr._sandbox_meta["s1"] = time.time()

        # Simulate command that outputs credentials
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cat"],
            returncode=0,
            stdout="password = my_secret_password_123\n",
            stderr="",
        )

        executor = SandboxExecutor(
            config=config,
            sandbox_manager=mgr,
            worktree_policy=WorktreePolicy(config),
        )

        result, events = executor.execute("s1", "agent_1", ["cat", "config.txt"])
        assert result.leakage_blocked
        assert "REDACTED" in result.stdout

        # Should have a LEAKAGE_DETECTED event
        leak_events = [
            e for e in events
            if e.event_type == WorktreeEventType.LEAKAGE_DETECTED
        ]
        assert len(leak_events) >= 1


# ===========================================================================
# 7. TestSandboxExecutorFlowTracking
# ===========================================================================


class TestSandboxExecutorFlowTracking:
    """2 InformationFlow records per command (inbound + outbound)."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    @patch("subprocess.run")
    def test_two_flows_per_command(self, mock_run, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        sandbox_dir = Path(config.sandbox_root) / "s1"
        sandbox_dir.mkdir()
        mgr._sandbox_meta["s1"] = time.time()

        mock_run.return_value = subprocess.CompletedProcess(
            args=["echo"], returncode=0, stdout="hello\n", stderr=""
        )

        flow_tracker = FlowTracker()
        executor = SandboxExecutor(
            config=config,
            sandbox_manager=mgr,
            worktree_policy=WorktreePolicy(config),
            flow_tracker=flow_tracker,
        )

        executor.execute("s1", "agent_1", ["echo", "hello"])

        all_flows = flow_tracker.flows + flow_tracker.blocked_flows
        assert len(all_flows) == 2

        directions = {f.direction for f in all_flows}
        assert FlowDirection.INBOUND in directions
        assert FlowDirection.OUTBOUND in directions


# ===========================================================================
# 8. TestWorktreeMapper
# ===========================================================================


class TestWorktreeMapper:
    """Observable extraction, p in [0,1] invariant."""

    def test_map_command_result_produces_valid_p(self):
        mapper = WorktreeMapper()
        result = CommandResult(
            sandbox_id="s1",
            agent_id="agent_1",
            command=["git", "status"],
            allowed=True,
            return_code=0,
            stdout="clean\n",
        )
        interaction = mapper.map_command_result(result)
        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0

    def test_denied_command_increases_misuse(self):
        mapper = WorktreeMapper()
        result = CommandResult(
            sandbox_id="s1",
            agent_id="agent_1",
            command=["ssh", "evil"],
            allowed=False,
            deny_reason="blocked",
        )
        interaction = mapper.map_command_result(
            result,
            agent_stats={"denied_count": 3, "total_commands": 5, "successful_commands": 2},
        )
        assert interaction.tool_misuse_flags >= 4  # 3 existing + 1 new

    def test_test_failure_increases_rejections(self):
        mapper = WorktreeMapper()
        result = CommandResult(
            sandbox_id="s1",
            agent_id="agent_1",
            command=["pytest", "-v"],
            allowed=True,
            return_code=1,
        )
        interaction = mapper.map_command_result(
            result,
            agent_stats={"test_failures": 1, "total_commands": 3, "successful_commands": 2},
        )
        assert interaction.verifier_rejections >= 2

    def test_p_invariant_across_extremes(self):
        """p must be in [0, 1] even with extreme observable values."""
        mapper = WorktreeMapper()

        # Best case
        result_good = CommandResult(
            sandbox_id="s1", agent_id="a", command=["ls"], allowed=True,
            return_code=0, stdout="ok",
        )
        i_good = mapper.map_command_result(
            result_good,
            agent_stats={"total_commands": 100, "successful_commands": 100},
        )
        assert 0.0 <= i_good.p <= 1.0

        # Worst case
        result_bad = CommandResult(
            sandbox_id="s1", agent_id="a", command=["ssh", "x"], allowed=False,
        )
        i_bad = mapper.map_command_result(
            result_bad,
            agent_stats={
                "denied_count": 50,
                "test_failures": 20,
                "total_commands": 100,
                "successful_commands": 0,
            },
        )
        assert 0.0 <= i_bad.p <= 1.0


# ===========================================================================
# 9. TestWorktreeBridge
# ===========================================================================


class TestWorktreeBridge:
    """Full integration: create -> dispatch -> get_interactions."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    @patch("subprocess.run")
    def test_full_lifecycle(self, mock_run, mock_git, tmp_path):
        config = _make_config(tmp_path)

        def git_side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = git_side_effect
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ls"], returncode=0, stdout="file.txt\n", stderr=""
        )

        bridge = WorktreeBridge(config)

        # Create sandbox
        sid = bridge.create_agent_sandbox("agent_1")
        assert sid == "sandbox-agent_1"

        # Dispatch command
        interaction = bridge.dispatch_command("agent_1", ["ls"])
        assert interaction.p >= 0.0
        assert interaction.p <= 1.0
        assert interaction.counterparty == "agent_1"

        # Get interactions
        interactions = bridge.get_interactions()
        assert len(interactions) >= 1

        # Get events
        events = bridge.get_events()
        assert len(events) >= 1

        # Get boundary metrics
        metrics = bridge.get_boundary_metrics()
        assert "flows" in metrics
        assert "leakage" in metrics
        assert "policy" in metrics

        # Shutdown
        bridge.shutdown()

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_dispatch_without_sandbox_raises(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        bridge = WorktreeBridge(config)

        with pytest.raises(ValueError, match="has no sandbox"):
            bridge.dispatch_command("nonexistent", ["ls"])

    @patch("swarm.bridges.worktree.sandbox._run_git")
    @patch("subprocess.run")
    def test_poll_produces_interactions(self, mock_run, mock_git, tmp_path):
        config = _make_config(tmp_path)

        def git_side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = git_side_effect
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        bridge = WorktreeBridge(config)
        bridge.create_agent_sandbox("agent_1")

        observations = bridge.poll()
        assert len(observations) == 1
        assert observations[0].counterparty == "agent_1"
        assert 0.0 <= observations[0].p <= 1.0


# ===========================================================================
# 10. TestEnvPropagation
# ===========================================================================


class TestEnvPropagation:
    """.env files removed, only allowlisted vars present."""

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_env_files_never_survive_creation(self, mock_git, tmp_path):
        config = _make_config(tmp_path)
        mgr = SandboxManager(config)

        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                d = args[2]
                os.makedirs(d, exist_ok=True)
                # Simulate repo with many .env files
                for name in [".env", ".env.local", ".env.production", ".env.staging"]:
                    Path(d, name).write_text(f"SECRET_{name}=value")
                # Also nested
                sub = Path(d, "subdir")
                sub.mkdir()
                Path(sub, ".env").write_text("NESTED_SECRET=bad")
            return _mock_git_success()

        mock_git.side_effect = side_effect
        path = mgr.create_sandbox("test-env-scrub")

        # No .env files should remain
        for root, _, files in os.walk(path):
            for f in files:
                assert not f.startswith(".env") or f == ".env.sandbox", \
                    f"Env file {f} survived in {root}"

    @patch("swarm.bridges.worktree.sandbox._run_git")
    def test_only_allowlisted_env_injected(self, mock_git, tmp_path):
        config = _make_config(
            tmp_path,
            env_allowlist={"SAFE_KEY": "safe_val", "ANOTHER": "val2"},
        )
        mgr = SandboxManager(config)

        def side_effect(args, cwd, timeout=10):
            if args[0] == "worktree" and args[1] == "add":
                os.makedirs(args[2], exist_ok=True)
            return _mock_git_success()

        mock_git.side_effect = side_effect
        path = mgr.create_sandbox("test-allowlist")

        env_file = Path(path) / ".env.sandbox"
        assert env_file.exists()
        content = env_file.read_text()
        assert "SAFE_KEY=safe_val" in content
        assert "ANOTHER=val2" in content

    def test_env_injection_blocks_credentials(self, tmp_path):
        config = _make_config(tmp_path)
        policy = WorktreePolicy(config)

        allowed, blocked = policy.evaluate_env_injection(
            "agent_1",
            {
                "PASSWORD": "secret",
                "API_KEY": "sk-123",
                "GITHUB_TOKEN": "ghp_xxx",
                "SAFE_VAR": "ok",
            },
        )

        # All credential-like vars should be blocked
        assert "PASSWORD" not in allowed
        assert "API_KEY" not in allowed
        assert "GITHUB_TOKEN" not in allowed
        # SAFE_VAR not in allowlist either, so blocked too
        assert "SAFE_VAR" not in allowed
        assert len(blocked) == 4

    def test_env_injection_allows_allowlisted(self, tmp_path):
        config = _make_config(
            tmp_path,
            env_allowlist={"SAFE_VAR": "default"},
        )
        policy = WorktreePolicy(config)

        allowed, blocked = policy.evaluate_env_injection(
            "agent_1",
            {"SAFE_VAR": "custom_value"},
        )

        assert "SAFE_VAR" in allowed
        assert allowed["SAFE_VAR"] == "custom_value"
        assert len(blocked) == 0


# ===========================================================================
# Event serialization
# ===========================================================================


class TestWorktreeEventSerialization:
    """WorktreeEvent round-trips through to_dict / from_dict."""

    def test_round_trip(self):
        event = WorktreeEvent(
            event_type=WorktreeEventType.COMMAND_DENIED,
            agent_id="agent_1",
            sandbox_id="s1",
            payload={"reason": "blocked"},
        )
        data = event.to_dict()
        restored = WorktreeEvent.from_dict(data)

        assert restored.event_type == event.event_type
        assert restored.agent_id == event.agent_id
        assert restored.sandbox_id == event.sandbox_id
        assert restored.payload == event.payload

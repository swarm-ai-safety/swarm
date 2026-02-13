"""Tests for the SWARM-GasTown bridge."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from swarm.bridges.gastown.agent import GasTownAgent
from swarm.bridges.gastown.beads import BeadsClient
from swarm.bridges.gastown.bridge import GasTownBridge
from swarm.bridges.gastown.config import GasTownConfig
from swarm.bridges.gastown.events import GasTownEventType
from swarm.bridges.gastown.git_observer import GitObserver
from swarm.bridges.gastown.mapper import GasTownMapper
from swarm.bridges.gastown.policy import GasTownPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the beads schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE issues (
            id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            assignee TEXT,
            updated_at TEXT
        )
        """
    )
    return conn


def _insert_bead(
    conn: sqlite3.Connection,
    bead_id: str,
    title: str = "task",
    status: str = "open",
    assignee: str = "polecat-1",
    updated_at: str | None = None,
) -> None:
    if updated_at is None:
        updated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO issues (id, title, status, assignee, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (bead_id, title, status, assignee, updated_at),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test: BeadsClient.poll_changes
# ---------------------------------------------------------------------------


class TestBeadsClientPoll:
    def test_poll_returns_events_for_new_beads(self, tmp_path):
        db_path = str(tmp_path / "beads.db")
        # Create a real file-based DB so BeadsClient can open it read-only
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE issues (
                id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                assignee TEXT,
                updated_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc)
        conn.execute(
            "INSERT INTO issues VALUES (?, ?, ?, ?, ?)",
            ("b1", "Fix bug", "done", "polecat-1", now.isoformat()),
        )
        conn.commit()
        conn.close()

        client = BeadsClient(db_path)
        try:
            # Poll from an hour ago
            past = datetime(2000, 1, 1, tzinfo=timezone.utc)
            events = client.poll_changes(past)
            assert len(events) == 1
            assert events[0].event_type == GasTownEventType.BEAD_COMPLETED
            assert events[0].bead_id == "b1"
            assert events[0].agent_name == "polecat-1"

            # Second poll with no changes returns nothing
            events2 = client.poll_changes(past)
            assert len(events2) == 0
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Test: GitObserver.get_pr_stats (mocked subprocess)
# ---------------------------------------------------------------------------


class TestGitObserverStats:
    @patch("swarm.bridges.gastown.git_observer.subprocess.run")
    def test_get_pr_stats(self, mock_run):
        """Mock git subprocess calls and verify stats extraction."""

        def side_effect(args, **kwargs):
            cmd = " ".join(args)
            result = MagicMock()
            result.returncode = 0
            if "rev-list --count" in cmd:
                result.stdout = "5\n"
            elif "diff --name-only" in cmd:
                result.stdout = "a.py\nb.py\nc.py\n"
            elif "reflog" in cmd:
                result.stdout = "amend: fixed\ncommit: init\n"
            elif "--grep=[ci-fail]" in cmd:
                result.stdout = "abc123 [ci-fail] broken\n"
            elif "--reverse" in cmd:
                result.stdout = "2025-01-01T00:00:00+00:00\n"
            elif "--merges" in cmd:
                result.stdout = "2025-01-02T00:00:00+00:00\n"
            else:
                result.stdout = ""
            return result

        mock_run.side_effect = side_effect

        observer = GitObserver("/workspace")
        stats = observer.get_pr_stats("/workspace/wt1")

        assert stats["commit_count"] == 5
        assert stats["files_changed"] == 3
        assert stats["review_iterations"] == 1  # "amend" matches
        assert stats["ci_failures"] == 1
        assert stats["time_to_merge_hours"] == pytest.approx(24.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: GasTownMapper — bead completion
# ---------------------------------------------------------------------------


class TestMapperBeadCompletion:
    def test_completed_bead_produces_valid_interaction(self):
        mapper = GasTownMapper()
        bead = {"id": "b1", "status": "done", "title": "Fix bug", "assignee": "p1"}
        git_stats = {
            "commit_count": 7,
            "files_changed": 3,
            "review_iterations": 1,
            "ci_failures": 0,
            "time_to_merge_hours": 12.0,
        }

        interaction = mapper.map_bead_completion(bead, git_stats, agent_id="agent_p1")

        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0
        assert interaction.counterparty == "agent_p1"
        assert interaction.metadata["bridge"] == "gastown"
        assert interaction.metadata["bead_id"] == "b1"


# ---------------------------------------------------------------------------
# Test: GasTownMapper — blocked bead
# ---------------------------------------------------------------------------


class TestMapperBlockedBead:
    def test_blocked_bead_has_negative_progress(self):
        mapper = GasTownMapper()
        bead = {"id": "b2", "status": "blocked", "title": "Stuck", "assignee": "p2"}
        git_stats = {
            "commit_count": 2,
            "files_changed": 1,
            "review_iterations": 0,
            "ci_failures": 0,
            "time_to_merge_hours": None,
        }

        interaction = mapper.map_bead_completion(bead, git_stats, agent_id="agent_p2")

        assert interaction.task_progress_delta == pytest.approx(-0.3)
        assert 0.0 <= interaction.p <= 1.0


# ---------------------------------------------------------------------------
# Test: GasTownBridge.poll cycle (mocked BeadsClient + GitObserver)
# ---------------------------------------------------------------------------


class TestBridgePollCycle:
    def test_poll_returns_interactions(self, tmp_path):
        # Set up a real beads DB
        db_path = str(tmp_path / "beads.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE issues (
                id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                assignee TEXT,
                updated_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc)
        conn.execute(
            "INSERT INTO issues VALUES (?, ?, ?, ?, ?)",
            ("b1", "Implement feature", "done", "polecat-1", now.isoformat()),
        )
        conn.commit()
        conn.close()

        config = GasTownConfig(
            workspace_path=str(tmp_path),
            beads_db_path=db_path,
        )
        bridge = GasTownBridge(config)

        # Mock git observer to avoid real git calls
        bridge._git_observer = MagicMock(spec=GitObserver)
        bridge._git_observer.get_agent_worktrees.return_value = {
            "polecat-1": str(tmp_path)
        }
        bridge._git_observer.get_pr_stats.return_value = {
            "commit_count": 5,
            "files_changed": 2,
            "review_iterations": 0,
            "ci_failures": 0,
            "time_to_merge_hours": 6.0,
        }

        try:
            interactions = bridge.poll()
            assert len(interactions) == 1
            assert interactions[0].counterparty == "polecat-1"
            assert 0.0 <= interactions[0].p <= 1.0

            # Verify stored
            assert len(bridge.get_interactions()) == 1
            assert len(bridge.get_events()) == 1
        finally:
            bridge.shutdown()


# ---------------------------------------------------------------------------
# Test: GitObserver.get_feature_branches (mocked subprocess)
# ---------------------------------------------------------------------------


class TestGitObserverBranches:
    @patch("swarm.bridges.gastown.git_observer.subprocess.run")
    def test_get_feature_branches(self, mock_run):
        """Discover unmerged remote branches and parse agent/slug."""
        result = MagicMock()
        result.returncode = 0
        result.stdout = (
            "origin/claude/fix-bug\n"
            "origin/codex/add-feature\n"
            "origin/dependabot/pip/setuptools\n"
        )
        mock_run.return_value = result

        observer = GitObserver("/workspace")
        branches = observer.get_feature_branches("origin/main")

        assert len(branches) == 3
        assert branches[0] == {
            "branch": "origin/claude/fix-bug",
            "agent": "claude",
            "slug": "fix-bug",
        }
        assert branches[1]["agent"] == "codex"
        assert branches[2]["agent"] == "dependabot"
        assert branches[2]["slug"] == "pip/setuptools"

    @patch("swarm.bridges.gastown.git_observer.subprocess.run")
    def test_get_branch_stats(self, mock_run):
        """Get per-branch stats vs base."""

        def side_effect(args, **kwargs):
            cmd = " ".join(args)
            r = MagicMock()
            r.returncode = 0
            if "rev-list --count" in cmd:
                r.stdout = "3\n"
            elif "diff --name-only" in cmd:
                r.stdout = "a.py\nb.py\n"
            elif "--grep=[ci-fail]" in cmd:
                r.stdout = "abc [ci-fail] broken\n"
            elif "log --oneline" in cmd:
                r.stdout = "abc fixup! typo\ndef Add feature\nghi fix: lint\n"
            elif "log --format=%aI" in cmd:
                r.stdout = (
                    "2025-01-02T12:00:00+00:00\n"
                    "2025-01-02T00:00:00+00:00\n"
                    "2025-01-01T00:00:00+00:00\n"
                )
            else:
                r.stdout = ""
            return r

        mock_run.side_effect = side_effect

        observer = GitObserver("/workspace")
        stats = observer.get_branch_stats("origin/claude/fix-bug", "origin/main")

        assert stats["commit_count"] == 3
        assert stats["files_changed"] == 2
        assert stats["review_iterations"] == 2  # "fixup!" and "fix:"
        assert stats["ci_failures"] == 1
        assert stats["time_to_merge_hours"] == pytest.approx(36.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: GasTownBridge.poll with branch fallback
# ---------------------------------------------------------------------------


class TestBridgePollBranchFallback:
    def test_poll_uses_branch_stats_when_no_worktrees(self, tmp_path):
        """When no worktrees exist, poll falls back to branch stats."""
        db_path = str(tmp_path / "beads.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE issues (
                id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                assignee TEXT,
                updated_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc)
        conn.execute(
            "INSERT INTO issues VALUES (?, ?, ?, ?, ?)",
            ("b1", "Fix bug", "closed", "claude", now.isoformat()),
        )
        conn.commit()
        conn.close()

        config = GasTownConfig(
            workspace_path=str(tmp_path),
            beads_db_path=db_path,
        )
        bridge = GasTownBridge(config)

        mock_obs = MagicMock(spec=GitObserver)
        # No worktrees
        mock_obs.get_agent_worktrees.return_value = {}
        # One branch for "claude"
        mock_obs.get_feature_branches.return_value = [
            {"branch": "origin/claude/fix-bug", "agent": "claude", "slug": "fix-bug"},
        ]
        mock_obs.get_branch_stats.return_value = {
            "commit_count": 4,
            "files_changed": 5,
            "review_iterations": 1,
            "ci_failures": 0,
            "time_to_merge_hours": 2.0,
        }
        bridge._git_observer = mock_obs

        try:
            interactions = bridge.poll()
            assert len(interactions) == 1
            assert interactions[0].counterparty == "claude"
            # With 4 commits the mapper should produce different stats
            # than the zero-commit default
            assert interactions[0].metadata["commit_count"] == 4
            mock_obs.get_branch_stats.assert_called_once()
        finally:
            bridge.shutdown()


# ---------------------------------------------------------------------------
# Test: GasTownBridge.poll_branches
# ---------------------------------------------------------------------------


class TestBridgePollBranches:
    def test_poll_branches_creates_interactions(self, tmp_path):
        db_path = str(tmp_path / "beads.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE issues (
                id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                assignee TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        config = GasTownConfig(
            workspace_path=str(tmp_path),
            beads_db_path=db_path,
        )
        bridge = GasTownBridge(config)

        mock_obs = MagicMock(spec=GitObserver)
        mock_obs.get_feature_branches.return_value = [
            {"branch": "origin/claude/fix-bug", "agent": "claude", "slug": "fix-bug"},
            {"branch": "origin/codex/add-feat", "agent": "codex", "slug": "add-feat"},
        ]
        mock_obs.get_branch_stats.return_value = {
            "commit_count": 2,
            "files_changed": 3,
            "review_iterations": 0,
            "ci_failures": 0,
            "time_to_merge_hours": None,
        }
        bridge._git_observer = mock_obs

        try:
            interactions = bridge.poll_branches()
            assert len(interactions) == 2
            assert interactions[0].counterparty == "claude"
            assert interactions[1].counterparty == "codex"
            assert interactions[0].metadata["source"] == "branch"
            assert interactions[0].metadata["branch"] == "origin/claude/fix-bug"

            # Second call should return nothing (already seen)
            interactions2 = bridge.poll_branches()
            assert len(interactions2) == 0
        finally:
            bridge.shutdown()


# ---------------------------------------------------------------------------
# Test: GasTownPolicy — circuit breaker
# ---------------------------------------------------------------------------


class TestPolicyCircuitBreaker:
    @patch("swarm.bridges.gastown.policy.subprocess.run")
    def test_circuit_breaker_calls_gt_stop(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        from swarm.governance.config import GovernanceConfig

        gc = GovernanceConfig(circuit_breaker_enabled=True)
        policy = GasTownPolicy(config=gc, gt_cli_path="/usr/bin/gt")

        result = policy.execute_circuit_breaker("polecat-1")

        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["/usr/bin/gt", "stop", "polecat-1"]


# ---------------------------------------------------------------------------
# Test: GasTownPolicy — audit assignment
# ---------------------------------------------------------------------------


class TestPolicyAuditAssignment:
    @patch("swarm.bridges.gastown.policy.subprocess.run")
    def test_audit_calls_gt_sling(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        policy = GasTownPolicy(gt_cli_path="gt")

        result = policy.assign_audit("bead-42")

        assert result is True
        args = mock_run.call_args[0][0]
        assert args == ["gt", "sling", "bead-42", "--to", "witness"]


# ---------------------------------------------------------------------------
# Test: GasTownAgent.act
# ---------------------------------------------------------------------------


class TestAgentAct:
    def test_act_returns_noop_when_no_interactions(self):
        bridge = MagicMock(spec=GasTownBridge)
        bridge.poll.return_value = []

        from swarm.agents.base import Observation
        from swarm.models.agent import AgentState

        agent = GasTownAgent(
            agent_id="agent_p1",
            bridge=bridge,
            gastown_name="polecat-1",
        )

        obs = Observation(agent_state=AgentState())
        action = agent.act(obs)
        assert action.action_type.value == "noop"

    def test_act_returns_action_when_interactions_exist(self):
        from swarm.agents.base import Observation
        from swarm.models.agent import AgentState
        from swarm.models.interaction import SoftInteraction

        interaction = SoftInteraction(
            counterparty="agent_p1",
            p=0.8,
            v_hat=0.5,
            metadata={"bead_title": "Fix login"},
        )
        bridge = MagicMock(spec=GasTownBridge)
        bridge.poll.return_value = [interaction]

        agent = GasTownAgent(
            agent_id="agent_p1",
            bridge=bridge,
            gastown_name="polecat-1",
        )

        obs = Observation(agent_state=AgentState())
        action = agent.act(obs)
        # Should produce a POST action with bead title
        assert action.action_type.value == "post"
        assert "Fix login" in action.content


# ---------------------------------------------------------------------------
# Test: GasTownConfig defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_defaults(self):
        config = GasTownConfig()
        assert config.workspace_path == "."
        assert config.beads_db_path is None
        assert config.gt_cli_path == "gt"
        assert config.poll_interval_seconds == 5.0
        assert config.proxy_sigmoid_k == 2.0
        assert config.agent_role_map == {}
        assert config.max_interactions == 50000
        assert config.max_events == 50000

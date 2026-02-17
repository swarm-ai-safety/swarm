"""Tests for SciAgentGym bridge environment management."""

import tempfile
from pathlib import Path

import pytest

from swarm.bridges.sciagentgym.config import (
    EnvironmentTopology,
    SciAgentGymConfig,
)
from swarm.bridges.sciagentgym.environment import (
    EnvironmentInstance,
    SciAgentGymEnvironmentManager,
)
from swarm.bridges.sciagentgym.workspace import WorkspaceManager


class TestSciAgentGymConfig:
    """Test SciAgentGym configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SciAgentGymConfig()

        assert config.enabled is True
        assert config.topology == EnvironmentTopology.PER_AGENT
        assert config.live_mode is False
        assert len(config.disciplines) > 0
        assert config.max_steps_per_task == 50

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid max_steps_per_task
        with pytest.raises(ValueError, match="max_steps_per_task must be >= 1"):
            SciAgentGymConfig(max_steps_per_task=0)

        # Invalid timeout
        with pytest.raises(ValueError, match="timeout_per_step must be > 0"):
            SciAgentGymConfig(timeout_per_step=0)

        # Invalid verification_confidence
        with pytest.raises(ValueError, match="verification_confidence must be in"):
            SciAgentGymConfig(verification_confidence=1.5)


class TestEnvironmentInstance:
    """Test EnvironmentInstance."""

    def test_environment_instance_creation(self):
        """Test creating environment instance."""
        instance = EnvironmentInstance(
            env_id="test_env",
            agent_id="agent_1",
        )

        assert instance.env_id == "test_env"
        assert instance.agent_id == "agent_1"
        assert instance.task_id is None
        assert instance.is_initialized is False
        assert len(instance.registered_tools) == 0

    def test_environment_instance_repr(self):
        """Test environment instance string representation."""
        instance = EnvironmentInstance(
            env_id="test_env",
            agent_id="agent_1",
            task_id="task_1",
        )

        repr_str = repr(instance)
        assert "test_env" in repr_str
        assert "agent_1" in repr_str
        assert "task_1" in repr_str


class TestSciAgentGymEnvironmentManager:
    """Test SciAgentGym environment manager."""

    def test_manager_initialization(self):
        """Test environment manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            assert manager.config == config
            assert len(manager._environments) == 0

    def test_shared_environment_topology(self):
        """Test shared environment topology."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                topology=EnvironmentTopology.SHARED_EPISODE,
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            # Get environment for first agent
            env1 = manager.get_or_create_environment("agent_1")
            assert env1.is_initialized
            assert env1.env_id == "shared_env"

            # Get environment for second agent - should be same
            env2 = manager.get_or_create_environment("agent_2")
            assert env2.env_id == env1.env_id
            assert len(manager._environments) == 1

    def test_per_agent_topology(self):
        """Test per-agent environment topology."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                topology=EnvironmentTopology.PER_AGENT,
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            # Get environment for first agent
            env1 = manager.get_or_create_environment("agent_1")
            assert env1.agent_id == "agent_1"

            # Get environment for second agent - should be different
            env2 = manager.get_or_create_environment("agent_2")
            assert env2.agent_id == "agent_2"
            assert env1.env_id != env2.env_id
            assert len(manager._environments) == 2

    def test_per_task_topology(self):
        """Test per-task environment topology."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                topology=EnvironmentTopology.PER_TASK,
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            # Get environment for first task
            env1 = manager.get_or_create_environment("agent_1", task_id="task_1")
            assert env1.task_id == "task_1"

            # Get environment for second task - should be different
            env2 = manager.get_or_create_environment("agent_1", task_id="task_2")
            assert env2.task_id == "task_2"
            assert env1.env_id != env2.env_id
            assert len(manager._environments) == 2

    def test_per_task_requires_task_id(self):
        """Test that per-task topology requires task_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                topology=EnvironmentTopology.PER_TASK,
            )
            manager = SciAgentGymEnvironmentManager(config)

            with pytest.raises(ValueError, match="task_id is required"):
                manager.get_or_create_environment("agent_1")

    def test_mock_environment_creation(self):
        """Test mock environment creation (live_mode=False)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            env = manager.get_or_create_environment("agent_1")

            assert env.is_initialized
            assert env.env is not None
            assert len(env.registered_tools) > 0  # Mock tools registered

    def test_environment_teardown(self):
        """Test environment teardown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            env = manager.get_or_create_environment("agent_1")
            env_id = env.env_id

            assert env_id in manager._environments

            manager.teardown_environment(env_id)

            assert env_id not in manager._environments

    def test_teardown_all_environments(self):
        """Test tearing down all environments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            # Create multiple environments
            manager.get_or_create_environment("agent_1")
            manager.get_or_create_environment("agent_2")
            manager.get_or_create_environment("agent_3")

            assert len(manager._environments) == 3

            manager.teardown_all()

            assert len(manager._environments) == 0

    def test_environment_stats(self):
        """Test getting environment statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SciAgentGymConfig(
                workspace_base_path=Path(tmpdir),
                topology=EnvironmentTopology.PER_AGENT,
                live_mode=False,
            )
            manager = SciAgentGymEnvironmentManager(config)

            manager.get_or_create_environment("agent_1")
            manager.get_or_create_environment("agent_2")

            stats = manager.get_environment_stats()

            assert stats["total_environments"] == 2
            assert stats["topology"] == "per_agent"
            assert stats["live_mode"] is False
            assert len(stats["environments"]) == 2


class TestWorkspaceManager:
    """Test workspace manager."""

    def test_workspace_mount(self):
        """Test mounting a workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkspaceManager(Path(tmpdir))

            workspace_path = manager.mount_workspace("test_env")

            assert workspace_path.exists()
            assert workspace_path.is_dir()
            assert (workspace_path / "data").exists()
            assert (workspace_path / "outputs").exists()

    def test_workspace_cleanup(self):
        """Test cleaning up a workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkspaceManager(Path(tmpdir))

            workspace_path = manager.mount_workspace("test_env")
            assert workspace_path.exists()

            # Create some files
            (workspace_path / "data" / "test.txt").write_text("test")

            result = manager.cleanup_workspace("test_env")

            assert result is True
            assert not workspace_path.exists()

    def test_workspace_size_tracking(self):
        """Test workspace size tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkspaceManager(Path(tmpdir), max_size_mb=1)

            workspace_path = manager.mount_workspace("test_env")

            # Create a small file (repeat "test" 100 times = ~400 bytes)
            # This is sufficient to verify size tracking without being excessive
            (workspace_path / "data" / "test.txt").write_text("test" * 100)

            size = manager.get_workspace_size("test_env")
            assert size > 0

            # Should be within quota
            assert manager.check_workspace_quota("test_env")

    def test_workspace_stats(self):
        """Test getting workspace statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkspaceManager(Path(tmpdir))

            manager.mount_workspace("env_1")
            manager.mount_workspace("env_2")

            stats = manager.get_workspace_stats()

            assert stats["total_workspaces"] == 2
            assert len(stats["workspaces"]) == 2
            assert "env_1" in stats["workspaces"]
            assert "env_2" in stats["workspaces"]

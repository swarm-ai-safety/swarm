"""SciAgentGym environment lifecycle manager.

Handles environment creation, toolkit registration, workspace mounting,
and teardown for agent interactions with SciAgentGym.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set

from swarm.bridges.sciagentgym.config import (
    EnvironmentTopology,
    SciAgentGymConfig,
)
from swarm.bridges.sciagentgym.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


# Number of mock tools to generate per discipline for testing
MOCK_TOOLS_PER_DISCIPLINE = 5


class EnvironmentInstance:
    """Represents a single SciAgentGym environment instance.

    Attributes:
        env_id: Unique identifier for this environment instance.
        agent_id: Agent ID using this environment (None if shared).
        task_id: Task ID for this environment (None if shared or per-agent).
        env: The actual SciAgentGym environment object.
        workspace_path: Path to mounted workspace for this environment.
        registered_tools: Set of tool names registered in this environment.
    """

    def __init__(
        self,
        env_id: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workspace_path: Optional[Path] = None,
    ) -> None:
        self.env_id = env_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.env: Any = None  # Will be MinimalSciEnv when live_mode
        self.workspace_path = workspace_path
        self.registered_tools: Set[str] = set()
        self.is_initialized = False

    def __repr__(self) -> str:
        return (
            f"EnvironmentInstance(env_id={self.env_id}, "
            f"agent_id={self.agent_id}, task_id={self.task_id})"
        )


class SciAgentGymEnvironmentManager:
    """Manages SciAgentGym environment lifecycle.

    Responsibilities:
    - Create environment instances based on topology
    - Register domain-specific toolkits
    - Mount and manage workspace filesystems
    - Teardown environments and cleanup resources
    """

    def __init__(self, config: SciAgentGymConfig) -> None:
        self.config = config
        self._environments: Dict[str, EnvironmentInstance] = {}
        self._agent_to_env: Dict[str, str] = {}  # agent_id -> env_id
        self._task_to_env: Dict[str, str] = {}  # task_id -> env_id
        self._shared_env_id: Optional[str] = None
        self._workspace_manager = WorkspaceManager(
            base_path=config.workspace_base_path,
            max_size_mb=config.max_workspace_size_mb,
        )

    def get_or_create_environment(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
    ) -> EnvironmentInstance:
        """Get or create an environment instance based on topology.

        Args:
            agent_id: ID of the agent requesting the environment.
            task_id: Optional task ID (used for PER_TASK topology).

        Returns:
            EnvironmentInstance configured for the agent/task.
        """
        topology = self.config.topology

        if topology == EnvironmentTopology.SHARED_EPISODE:
            return self._get_or_create_shared_env()
        elif topology == EnvironmentTopology.PER_AGENT:
            return self._get_or_create_per_agent_env(agent_id)
        elif topology == EnvironmentTopology.PER_TASK:
            if task_id is None:
                raise ValueError("task_id is required for PER_TASK topology")
            return self._get_or_create_per_task_env(agent_id, task_id)
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def _get_or_create_shared_env(self) -> EnvironmentInstance:
        """Get or create the shared environment."""
        if self._shared_env_id and self._shared_env_id in self._environments:
            return self._environments[self._shared_env_id]

        env_id = "shared_env"
        workspace_path = self.config.workspace_base_path / "shared"
        instance = EnvironmentInstance(
            env_id=env_id,
            workspace_path=workspace_path,
        )

        self._initialize_environment(instance)
        self._environments[env_id] = instance
        self._shared_env_id = env_id

        logger.info("Created shared environment: %s", env_id)
        return instance

    def _get_or_create_per_agent_env(self, agent_id: str) -> EnvironmentInstance:
        """Get or create environment for specific agent."""
        if agent_id in self._agent_to_env:
            env_id = self._agent_to_env[agent_id]
            return self._environments[env_id]

        env_id = f"env_{agent_id}"
        workspace_path = self.config.workspace_base_path / agent_id
        instance = EnvironmentInstance(
            env_id=env_id,
            agent_id=agent_id,
            workspace_path=workspace_path,
        )

        self._initialize_environment(instance)
        self._environments[env_id] = instance
        self._agent_to_env[agent_id] = env_id

        logger.info("Created per-agent environment: %s for agent %s", env_id, agent_id)
        return instance

    def _get_or_create_per_task_env(
        self, agent_id: str, task_id: str
    ) -> EnvironmentInstance:
        """Get or create environment for specific task."""
        if task_id in self._task_to_env:
            env_id = self._task_to_env[task_id]
            return self._environments[env_id]

        env_id = f"env_{task_id}"
        workspace_path = self.config.workspace_base_path / task_id
        instance = EnvironmentInstance(
            env_id=env_id,
            agent_id=agent_id,
            task_id=task_id,
            workspace_path=workspace_path,
        )

        self._initialize_environment(instance)
        self._environments[env_id] = instance
        self._task_to_env[task_id] = env_id

        logger.info(
            "Created per-task environment: %s for task %s", env_id, task_id
        )
        return instance

    def _initialize_environment(self, instance: EnvironmentInstance) -> None:
        """Initialize environment with tools, workspace, and configuration.

        Args:
            instance: EnvironmentInstance to initialize.
        """
        if self.config.live_mode:
            self._initialize_live_environment(instance)
        else:
            self._initialize_mock_environment(instance)

        instance.is_initialized = True

    def _initialize_live_environment(self, instance: EnvironmentInstance) -> None:
        """Initialize real SciAgentGym environment (replaces mocks).

        Args:
            instance: EnvironmentInstance to initialize.
        """
        try:
            from gym.env import MinimalSciEnv
        except ImportError as e:
            raise ImportError(
                "SciAgentGym is not installed. "
                "Install it to use live_mode: "
                "git clone https://github.com/CMarsRover/SciAgentGYM && "
                "pip install -e SciAgentGYM/"
            ) from e

        logger.info("Initializing LIVE SciAgentGym environment: %s", instance.env_id)

        # Create real SciAgentGym environment
        instance.env = MinimalSciEnv()

        # Register toolkits from specified disciplines
        self._register_live_toolkits(instance)

        # Mount workspace filesystem
        if self.config.enable_filesystem:
            self._mount_workspace(instance)

        # Configure additional features
        if self.config.enable_databases:
            self._setup_databases(instance)

        if self.config.enable_python_interpreter:
            self._setup_python_interpreter(instance)

    def _initialize_mock_environment(self, instance: EnvironmentInstance) -> None:
        """Initialize mock environment for testing (to be replaced).

        Args:
            instance: EnvironmentInstance to initialize.
        """
        logger.info("Initializing MOCK SciAgentGym environment: %s", instance.env_id)

        # Mock environment object
        instance.env = MockSciEnv(env_id=instance.env_id)

        # Register mock tools
        self._register_mock_toolkits(instance)

        # Create mock workspace
        if self.config.enable_filesystem and instance.workspace_path:
            instance.workspace_path.mkdir(parents=True, exist_ok=True)

    def _register_live_toolkits(self, instance: EnvironmentInstance) -> None:
        """Register real SciAgentGym toolkits (replaces mocks).

        Args:
            instance: EnvironmentInstance to register tools in.
        """
        from swarm.bridges.sciagentgym.toolkit import load_tools_for_disciplines

        disciplines = self.config.disciplines
        tool_filter = self.config.tool_filter

        logger.info(
            "Registering LIVE toolkits for disciplines: %s (env=%s)",
            disciplines,
            instance.env_id,
        )

        # Load and register tools from SciAgentGym
        tools = load_tools_for_disciplines(
            disciplines=disciplines,
            sciagentgym_path=str(self.config.sciagentgym_path),
        )

        # Apply tool filter if specified
        if tool_filter:
            tools = {name: tool for name, tool in tools.items() if name in tool_filter}

        # Register tools in environment
        for tool_name, tool_class in tools.items():
            instance.env.register_tool(tool_name, tool_class)
            instance.registered_tools.add(tool_name)

        logger.info(
            "Registered %d LIVE tools in environment %s",
            len(instance.registered_tools),
            instance.env_id,
        )

    def _register_mock_toolkits(self, instance: EnvironmentInstance) -> None:
        """Register mock toolkits for testing (to be replaced).

        Args:
            instance: EnvironmentInstance to register mock tools in.
        """
        logger.info(
            "Registering MOCK toolkits for env=%s", instance.env_id
        )

        # Mock tool registration
        mock_tools = [
            f"mock_tool_{discipline}_{i}"
            for discipline in self.config.disciplines
            for i in range(MOCK_TOOLS_PER_DISCIPLINE)
        ]

        for tool_name in mock_tools:
            instance.env.register_tool(tool_name, MockTool(tool_name))
            instance.registered_tools.add(tool_name)

        logger.info(
            "Registered %d MOCK tools in environment %s",
            len(instance.registered_tools),
            instance.env_id,
        )

    def _mount_workspace(self, instance: EnvironmentInstance) -> None:
        """Mount workspace filesystem for environment (live mode).

        Args:
            instance: EnvironmentInstance to mount workspace for.
        """
        if not instance.workspace_path:
            return

        # Use WorkspaceManager to mount workspace
        workspace_path = self._workspace_manager.mount_workspace(instance.env_id)
        instance.workspace_path = workspace_path

        # Configure environment to use this workspace
        instance.env.set_workspace_path(str(instance.workspace_path))

    def _setup_databases(self, instance: EnvironmentInstance) -> None:
        """Setup database access for environment (live mode).

        Args:
            instance: EnvironmentInstance to setup databases for.
        """
        logger.info("Setting up database access for env=%s", instance.env_id)

        # Configure database paths from SciAgentGym
        db_path = self.config.sciagentgym_path / "toolkits" / "local_db"
        if db_path.exists():
            instance.env.set_database_path(str(db_path))

    def _setup_python_interpreter(self, instance: EnvironmentInstance) -> None:
        """Setup Python interpreter for environment (live mode).

        Args:
            instance: EnvironmentInstance to setup interpreter for.
        """
        logger.info("Setting up Python interpreter for env=%s", instance.env_id)

        # Configure safe Python execution environment
        instance.env.enable_python_interpreter(
            timeout=self.config.timeout_per_step
        )

    def teardown_environment(self, env_id: str) -> None:
        """Teardown and cleanup environment (replaces mock).

        Args:
            env_id: ID of environment to teardown.
        """
        if env_id not in self._environments:
            logger.warning("Environment %s not found for teardown", env_id)
            return

        instance = self._environments[env_id]

        logger.info("Tearing down environment: %s", env_id)

        # Cleanup workspace
        if instance.workspace_path and instance.workspace_path.exists():
            try:
                shutil.rmtree(instance.workspace_path)
                logger.info("Cleaned up workspace: %s", instance.workspace_path)
            except Exception as e:
                logger.error("Failed to cleanup workspace %s: %s", instance.workspace_path, e)

        # Close environment
        if instance.env and hasattr(instance.env, "close"):
            try:
                instance.env.close()
            except Exception as e:
                logger.error("Failed to close environment %s: %s", env_id, e)

        # Remove from tracking
        del self._environments[env_id]
        if instance.agent_id and instance.agent_id in self._agent_to_env:
            del self._agent_to_env[instance.agent_id]
        if instance.task_id and instance.task_id in self._task_to_env:
            del self._task_to_env[instance.task_id]
        if env_id == self._shared_env_id:
            self._shared_env_id = None

        logger.info("Successfully tore down environment: %s", env_id)

    def teardown_all(self) -> None:
        """Teardown all environments and cleanup resources."""
        logger.info("Tearing down all environments")

        env_ids = list(self._environments.keys())
        for env_id in env_ids:
            self.teardown_environment(env_id)

        logger.info("All environments torn down")

    def get_environment_stats(self) -> Dict[str, Any]:
        """Get statistics about managed environments.

        Returns:
            Dictionary with environment statistics.
        """
        return {
            "total_environments": len(self._environments),
            "topology": self.config.topology.value,
            "live_mode": self.config.live_mode,
            "environments": {
                env_id: {
                    "agent_id": inst.agent_id,
                    "task_id": inst.task_id,
                    "registered_tools": len(inst.registered_tools),
                    "workspace": str(inst.workspace_path) if inst.workspace_path else None,
                }
                for env_id, inst in self._environments.items()
            },
        }


class MockSciEnv:
    """Mock SciAgentGym environment for testing (to be replaced by live mode)."""

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.tools: Dict[str, Any] = {}
        self.workspace_path: Optional[str] = None

    def register_tool(self, name: str, tool: Any) -> None:
        """Mock tool registration."""
        self.tools[name] = tool

    def set_workspace_path(self, path: str) -> None:
        """Mock workspace setup."""
        self.workspace_path = path

    def set_database_path(self, path: str) -> None:
        """Mock database setup."""
        pass

    def enable_python_interpreter(self, timeout: float) -> None:
        """Mock Python interpreter setup."""
        pass

    def close(self) -> None:
        """Mock environment cleanup."""
        pass


class MockTool:
    """Mock tool for testing (to be replaced by live mode)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def use(self, *args: Any, **kwargs: Any) -> str:
        """Mock tool execution."""
        return f"Mock result from {self.name}"

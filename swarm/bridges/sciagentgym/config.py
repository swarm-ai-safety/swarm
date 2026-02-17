"""Configuration for the SWARM-SciAgentGym bridge."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class EnvironmentTopology(str, Enum):
    """Environment isolation topology for agents.

    Attributes:
        SHARED_EPISODE: All agents share the same environment instance.
            Tools and filesystem are shared across agents.
        PER_AGENT: Each agent gets its own environment instance.
            Tools and filesystem are isolated per agent.
        PER_TASK: Each task gets its own environment instance.
            Tools and filesystem are isolated per task execution.
    """

    SHARED_EPISODE = "shared_episode"
    PER_AGENT = "per_agent"
    PER_TASK = "per_task"


@dataclass
class SciAgentGymConfig:
    """Configuration for the SciAgentGym bridge.

    Attributes:
        enabled: Whether the bridge is enabled.
        sciagentgym_path: Path to SciAgentGym installation.
        topology: Environment isolation strategy (shared_episode, per_agent, per_task).
        disciplines: List of disciplines to load tools from
            (e.g., ['physics', 'chemistry', 'materials_science']).
        workspace_base_path: Base directory for mounting agent workspaces.
        live_mode: If True, use real SciAgentGym environment; if False, use mocks.
        mock_tool_execution: If True, simulate tool execution without actual computation.
        seed: Random seed for reproducibility.
        max_steps_per_task: Maximum steps allowed per task.
        timeout_per_step: Timeout in seconds for each tool execution step.
        enable_filesystem: Whether to enable filesystem access in environments.
        enable_databases: Whether to enable database access in environments.
        enable_python_interpreter: Whether to enable Python interpreter in environments.
        tool_filter: Optional list of specific tool names to register (if None, all tools).
        verification_confidence: Confidence level for binary_to_soft_p conversion.
    """

    enabled: bool = True
    sciagentgym_path: Path = field(default_factory=lambda: Path("external/SciAgentGYM"))
    topology: EnvironmentTopology = EnvironmentTopology.PER_AGENT
    disciplines: List[str] = field(
        default_factory=lambda: [
            "physics",
            "chemistry",
            "materials_science",
            "life_science",
        ]
    )
    workspace_base_path: Path = field(
        default_factory=lambda: Path("/tmp/sciagentgym_workspaces")
    )

    # Runtime mode
    live_mode: bool = False
    mock_tool_execution: bool = False

    # Execution parameters
    seed: Optional[int] = None
    max_steps_per_task: int = 50
    timeout_per_step: float = 30.0

    # Environment features
    enable_filesystem: bool = True
    enable_databases: bool = True
    enable_python_interpreter: bool = True

    # Tool configuration
    tool_filter: Optional[List[str]] = None

    # Observable mapping weights
    error_weight: float = 1.0
    tool_misuse_weight: float = 1.0

    # Verification
    verification_confidence: float = 0.8

    # Resource limits
    max_workspace_size_mb: int = 1000
    max_concurrent_environments: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_steps_per_task < 1:
            raise ValueError("max_steps_per_task must be >= 1")
        if self.timeout_per_step <= 0:
            raise ValueError("timeout_per_step must be > 0")
        if not (0.0 <= self.verification_confidence <= 1.0):
            raise ValueError("verification_confidence must be in [0, 1]")
        if self.max_workspace_size_mb < 1:
            raise ValueError("max_workspace_size_mb must be >= 1")

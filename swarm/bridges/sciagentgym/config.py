"""Configuration for the SciAgentGym bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProviderType(Enum):
    """Supported tool execution providers."""

    LOCAL = "local"  # Local Python interpreter
    DOCKER = "docker"  # Docker container execution
    KUBERNETES = "kubernetes"  # Kubernetes pod execution


class TopologyType(Enum):
    """Network topology configurations for agent-tool interactions."""

    COMPLETE = "complete"  # All agents can access all tools
    RING = "ring"  # Agents organized in a ring, access limited to neighbors
    STAR = "star"  # Central hub agent with peripheral agents


DEFAULT_ROLE_MAP: Dict[str, str] = {
    "tool_executor": "tool_executor_agent",
    "workflow_planner": "workflow_planner_agent",
    "result_validator": "result_validator_agent",
    "data_manager": "data_manager_agent",
}


@dataclass
class ProviderConfig:
    """Configuration for tool execution provider.
    
    Attributes:
        provider_type: Type of execution provider (local, docker, kubernetes).
        timeout_seconds: Maximum execution time per tool call.
        max_retries: Maximum retry attempts on transient failures.
        sandbox_enabled: Whether to run tools in sandboxed environment.
        resource_limits: Resource constraints for execution (e.g., {"memory": "2Gi", "cpu": "1"}).
        base_image: Docker image or base environment for execution.
        working_dir: Working directory for tool execution and data persistence.
    """

    provider_type: ProviderType = ProviderType.LOCAL
    timeout_seconds: float = 300.0
    max_retries: int = 2
    sandbox_enabled: bool = True
    resource_limits: Dict[str, str] = field(default_factory=dict)
    base_image: Optional[str] = None
    working_dir: str = "/tmp/sciagentgym"


@dataclass
class TopologyConfig:
    """Configuration for agent-tool network topology.
    
    Attributes:
        topology_type: Type of network topology.
        k_neighbors: For RING topology, number of neighbor connections (default 2).
        hub_agent_id: For STAR topology, the central hub agent ID.
        tool_access_policy: Custom tool access matrix {"agent_id": ["tool1", "tool2"]}.
        dynamic_routing: Whether to allow dynamic topology evolution.
    """

    topology_type: TopologyType = TopologyType.COMPLETE
    k_neighbors: int = 2
    hub_agent_id: Optional[str] = None
    tool_access_policy: Dict[str, List[str]] = field(default_factory=dict)
    dynamic_routing: bool = False


@dataclass
class SciAgentGymClientConfig:
    """Configuration for SciAgentGym environment client.
    
    Attributes:
        data_dir: Directory containing SciAgentGym datasets and tools.
        task_file: Path to task specification file.
        tool_registry_path: Path to tool registry JSON.
        enable_caching: Whether to cache intermediate results.
        max_workflow_steps: Maximum steps in a multi-step workflow.
    """

    data_dir: str = "data/sciagentgym"
    task_file: str = "tasks.json"
    tool_registry_path: str = "tools/registry.json"
    enable_caching: bool = True
    max_workflow_steps: int = 20


@dataclass
class SciAgentGymConfig:
    """Full bridge configuration for SciAgentGym.
    
    Attributes:
        client_config: Configuration for the SciAgentGym client.
        provider_config: Configuration for tool execution provider.
        topology_config: Configuration for agent-tool network topology.
        orchestrator_id: ID of the bridge orchestrator agent.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        agent_role_map: Maps SciAgentGym agent roles to SWARM agent IDs.
        
        # Governance thresholds
        tool_safety_gate_enabled: Whether to gate tool execution on safety scores.
        min_tool_safety_score: Minimum p (probability benign) to allow tool execution.
        workflow_circuit_breaker_max_failures: Max failures before halting workflow.
        cost_budget_tokens: Maximum token budget for LLM-based tool use.
        result_quality_threshold: Minimum quality score for accepting results.
        
        # Memory caps
        max_interactions: Maximum number of interactions to store.
        max_events: Maximum number of events to store.
    """

    client_config: SciAgentGymClientConfig = field(
        default_factory=SciAgentGymClientConfig,
    )
    provider_config: ProviderConfig = field(
        default_factory=ProviderConfig,
    )
    topology_config: TopologyConfig = field(
        default_factory=TopologyConfig,
    )
    orchestrator_id: str = "sciagentgym_orchestrator"
    proxy_sigmoid_k: float = 2.0

    agent_role_map: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROLE_MAP),
    )

    # Governance thresholds
    tool_safety_gate_enabled: bool = True
    min_tool_safety_score: float = 0.4
    workflow_circuit_breaker_max_failures: int = 5
    cost_budget_tokens: int = 100000
    result_quality_threshold: float = 0.5

    # Memory caps
    max_interactions: int = 50000
    max_events: int = 50000

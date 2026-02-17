"""SWARM-SciAgentGym Bridge.

Connects SWARM's governance and metrics framework to SciAgentGym,
enabling monitoring, scoring, and governance of scientific tool-use
workflows across multiple disciplines (Physics, Chemistry, Materials, Life Science).

Architecture:
    SciAgentGym (Tool execution logs, workflow traces)
        |
    SciAgentGymClient (JSON parser, tool registry)
        |
    SciAgentGymBridge._process_event()
        |   SciAgentGymPolicy (safety gates, circuit breakers, cost caps)
        |
    SciAgentGymMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
        |
    SoftInteraction -> EventLog + SWARM metrics pipeline

Provider Interfaces:
    - ProviderType: LOCAL, DOCKER, KUBERNETES (typed execution backends)
    - ProviderConfig: Typed configuration for tool execution providers

Topology Configurations:
    - COMPLETE: All agents can access all tools (default)
    - RING: Agents organized in a ring, access limited to k neighbors
    - STAR: Central hub agent with peripheral agents
"""

# Event processing pipeline (from PR #195)
from swarm.bridges.sciagentgym.bridge import SciAgentGymBridge
from swarm.bridges.sciagentgym.client import SciAgentGymClient
from swarm.bridges.sciagentgym.config import (
    ProviderConfig,
    ProviderType,
    SciAgentGymClientConfig,
    SciAgentGymConfig,
    TopologyConfig,
    TopologyType,
)
from swarm.bridges.sciagentgym.events import (
    DataArtifactEvent,
    SafetyCheckEvent,
    SciAgentGymEvent,
    SciAgentGymEventType,
    ToolCallEvent,
    WorkflowStepEvent,
)
from swarm.bridges.sciagentgym.mapper import SciAgentGymMapper
from swarm.bridges.sciagentgym.policy import (
    PolicyDecision,
    PolicyResult,
    SciAgentGymPolicy,
)

# Environment lifecycle management (from PR #194)
from swarm.bridges.sciagentgym.governance import ToolCallFingerprint, ToolLoopDetector
from swarm.bridges.sciagentgym.manager import EmbeddingTopology, SciEnvManager
from swarm.bridges.sciagentgym.provider import SciAgentGymToolProvider, ToolCallResult

__all__ = [
    # Event processing pipeline
    "SciAgentGymBridge",
    "SciAgentGymClient",
    "SciAgentGymMapper",
    "SciAgentGymPolicy",
    # Configuration
    "SciAgentGymConfig",
    "SciAgentGymClientConfig",
    "ProviderConfig",
    "ProviderType",
    "TopologyConfig",
    "TopologyType",
    # Events
    "SciAgentGymEvent",
    "SciAgentGymEventType",
    "ToolCallEvent",
    "WorkflowStepEvent",
    "DataArtifactEvent",
    "SafetyCheckEvent",
    # Policy
    "PolicyDecision",
    "PolicyResult",
    # Environment lifecycle management
    "EmbeddingTopology",
    "SciAgentGymToolProvider",
    "SciEnvManager",
    "ToolCallFingerprint",
    "ToolCallResult",
    "ToolLoopDetector",
]

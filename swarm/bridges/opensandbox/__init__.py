"""SWARM OpenSandbox Bridge.

Provides contract-governed multi-agent sandbox environments with
full provenance tracking, mediated message routing, and behavioral
observability.  Each agent runs inside an isolated OpenSandbox
container, with governance contracts enforced at the orchestration
layer and provenance tracked via Byline integration.

Architecture layers:
  1. Governance & Contract — contract registry, screening protocol
  2. Orchestration — sandbox lifecycle, message routing
  3. Protocol Adapter — SWARM metadata on OpenSandbox lifecycle API
  4. Sandbox Runtime — per-agent isolated containers
  5. Infrastructure — Docker / K8s (pluggable)
  6. Observability — cross-cutting metrics, risk detection
"""

from swarm.bridges.opensandbox.bridge import OpenSandboxBridge
from swarm.bridges.opensandbox.config import (
    AgentType,
    CapabilityManifest,
    ContractAssignment,
    GovernanceContract,
    InteractionPolicy,
    NetworkPolicy,
    OpenSandboxConfig,
)
from swarm.bridges.opensandbox.message_bus import MessageBus
from swarm.bridges.opensandbox.observer import Observer
from swarm.bridges.opensandbox.provenance import ProvenanceTracker
from swarm.bridges.opensandbox.screener import ScreeningProtocol

__all__ = [
    "OpenSandboxBridge",
    "OpenSandboxConfig",
    "GovernanceContract",
    "CapabilityManifest",
    "ContractAssignment",
    "AgentType",
    "NetworkPolicy",
    "InteractionPolicy",
    "ScreeningProtocol",
    "MessageBus",
    "ProvenanceTracker",
    "Observer",
]

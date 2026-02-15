"""SWARM-AWM Bridge.

Connects SWARM's governance and metrics framework to Agent World Model
(AWM) database-backed tool-use environments, enabling agents to interact
with realistic SQL-database tasks (e-commerce, project management,
booking systems, etc.).

Architecture:
    AWMServerManager
        ├── starts/stops FastAPI servers per agent
        └── resets DB state between epochs
    AWMMCPClient
        └── HTTP client for MCP tool calls
    AWMObservableMapper
        └── converts AWMEpisodeTrace → ProxyObservables
    AWMVerifierBridge
        └── runs AWM verification, binary_to_soft_p()

Requires: pip install swarm-safety[awm]
"""

try:
    import httpx  # noqa: F401

    AWM_AVAILABLE = True
except ImportError:
    AWM_AVAILABLE = False

__all__ = [
    "AWM_AVAILABLE",
]

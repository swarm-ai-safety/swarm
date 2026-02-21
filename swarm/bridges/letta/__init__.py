"""SWARM-Letta (MemGPT) Bridge.

Connects SWARM's governance and metrics framework to Letta's stateful
agent runtime, giving agents persistent three-tier memory (core/archival/
recall), self-editing memory, shared memory blocks for governance, and
sleep-time consolidation.

Architecture:
    LettaServerManager
        └── manages Letta server lifecycle (external / docker / local)
    LettaSwarmClient
        └── synchronous wrapper around letta_client.Letta SDK
    LettaMemoryMapper
        └── bidirectional SWARM Observation <-> Letta core memory
    LettaResponseParser
        └── parses Letta natural language responses into SWARM Actions
    LettaLifecycleManager
        └── orchestrator-level lifecycle (start, governance block, shutdown)

Requires: pip install swarm-safety[letta]
"""

try:
    import letta_client  # noqa: F401

    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False

__all__ = ["LETTA_AVAILABLE"]

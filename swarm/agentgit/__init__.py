"""Agent-first git provenance primitives.

The MVP in this package keeps git as the storage substrate while adding
task-scoped policy checks and signed provenance bundles for agent-authored
changes.
"""

from swarm.agentgit.bundle import build_bundle, write_bundle
from swarm.agentgit.policy import AgentGitPolicy, PolicyDecision

__all__ = [
    "AgentGitPolicy",
    "PolicyDecision",
    "build_bundle",
    "write_bundle",
]

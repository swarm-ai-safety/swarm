"""Agent-first git provenance primitives.

The MVP in this package keeps git as the storage substrate while adding
task-scoped policy checks and signed provenance bundles for agent-authored
changes.
"""

from swarm.agentgit.bundle import build_bundle, verify_bundle, write_bundle
from swarm.agentgit.identity import (
    AgentIdentity,
    AgentKeypair,
    DelegationChain,
    DelegationLink,
    sign_link,
)
from swarm.agentgit.policy import AgentGitPolicy, PolicyDecision

__all__ = [
    "AgentGitPolicy",
    "AgentIdentity",
    "AgentKeypair",
    "DelegationChain",
    "DelegationLink",
    "PolicyDecision",
    "build_bundle",
    "sign_link",
    "verify_bundle",
    "write_bundle",
]

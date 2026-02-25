"""Configuration for the SWARM-OpenSandbox bridge.

Defines governance contracts, capability manifests, sandbox tier
configurations, and the top-level bridge config.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NetworkPolicy(Enum):
    """Network access tier for sandboxed agents."""

    DENY_ALL = "deny_all"
    ALLOWLIST = "allowlist"
    FULL = "full"


class AgentType(Enum):
    """Declared agent behavioral type for screening."""

    COOPERATIVE = "cooperative"
    ADVERSARIAL = "adversarial"
    SELF_MODIFYING = "self_modifying"
    STATIC = "static"


class InteractionPolicy(Enum):
    """How agents may communicate with each other."""

    MESSAGE_BUS_ONLY = "message_bus_only"
    SHARED_FS = "shared_fs"
    NONE = "none"


@dataclass
class CapabilityManifest:
    """Agent-declared capabilities and intent.

    Agents present this manifest during screening.  The screener
    evaluates it against available governance contracts to assign
    the agent to a compatible tier.

    Attributes:
        agent_id: Unique agent identifier.
        agent_type: Self-declared behavioral type.
        capabilities: Languages / tools the agent needs.
        requires_network: Whether the agent requires network access.
        max_memory_mb: Requested memory ceiling.
        max_cpu_shares: Requested CPU share ceiling.
        declared_intent: Free-text intent declaration.
        metadata: Extra metadata for screening heuristics.
    """

    agent_id: str
    agent_type: AgentType = AgentType.COOPERATIVE
    capabilities: List[str] = field(default_factory=lambda: ["python"])
    requires_network: bool = False
    max_memory_mb: int = 512
    max_cpu_shares: int = 1024
    declared_intent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceContract:
    """Machine-readable governance contract for a sandbox tier.

    Contracts define capability boundaries, interaction rules, and
    safety invariants.  The orchestration layer compiles contracts
    down to concrete container configurations.

    Attributes:
        contract_id: Unique contract identifier.
        tier: Human-readable tier name (e.g. "restricted", "standard").
        capabilities: Permitted languages / executables.
        network: Network access policy.
        interaction: Inter-agent interaction policy.
        network_allowlist: Domains allowed when network=ALLOWLIST.
        max_memory_mb: Memory limit for containers in this tier.
        max_cpu_shares: CPU shares limit.
        max_disk_mb: Disk limit in MB.
        timeout_seconds: Maximum sandbox lifetime in seconds.
        allowed_mounts: Paths the container may mount read-only.
        safety_invariants: Named invariant checks to enforce at runtime.
        metadata: Extra governance metadata.
    """

    contract_id: str = "default"
    tier: str = "restricted"
    capabilities: List[str] = field(
        default_factory=lambda: ["python", "file_read"]
    )
    network: NetworkPolicy = NetworkPolicy.DENY_ALL
    interaction: InteractionPolicy = InteractionPolicy.MESSAGE_BUS_ONLY
    network_allowlist: List[str] = field(default_factory=list)
    max_memory_mb: int = 512
    max_cpu_shares: int = 1024
    max_disk_mb: int = 1024
    timeout_seconds: int = 1800  # 30 minutes
    allowed_mounts: List[str] = field(default_factory=list)
    safety_invariants: List[str] = field(
        default_factory=lambda: ["p_in_bounds", "append_only_log"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def allows_capability(self, cap: str) -> bool:
        """Check whether *cap* is permitted under this contract."""
        return cap in self.capabilities

    def to_sandbox_env(self) -> Dict[str, str]:
        """Compile contract parameters to sandbox environment variables."""
        return {
            "SWARM_CONTRACT": self.contract_id,
            "SWARM_TIER": self.tier,
            "SWARM_NETWORK": self.network.value,
            "SWARM_INTERACTION": self.interaction.value,
            "SWARM_CAPABILITIES": ",".join(self.capabilities),
            "SWARM_TIMEOUT": str(self.timeout_seconds),
        }


@dataclass
class ContractAssignment:
    """Result of the screening protocol.

    Links an agent to the contract tier it was assigned to, along
    with the concrete sandbox configuration derived from the contract.

    Attributes:
        agent_id: The screened agent.
        contract_id: Assigned contract.
        tier: Tier name (copied from contract for convenience).
        sandbox_env: Environment variables for the container.
        rejected: True if the agent was rejected outright.
        rejection_reason: Human-readable reason for rejection.
        score: Screening compatibility score [0, 1].
        metadata: Extra screening metadata.
    """

    agent_id: str
    contract_id: str = "default"
    tier: str = "restricted"
    sandbox_env: Dict[str, str] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenSandboxConfig:
    """Top-level configuration for the OpenSandbox bridge.

    Attributes:
        sandbox_image: Default Docker image for sandbox containers.
        default_contract: Contract applied when no specific one matches.
        contracts: Available governance contracts indexed by contract_id.
        max_sandboxes: Maximum concurrent sandbox containers.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        max_interactions: Cap on stored SoftInteraction records.
        max_events: Cap on stored bridge events.
        message_bus_max_pending: Max pending messages in the bus.
        provenance_enabled: Enable Byline provenance signing.
        snapshot_enabled: Enable container snapshotting.
        observer_interval_seconds: Observability polling interval.
        risk_threshold: Risk score above which governance intervenes.
    """

    sandbox_image: str = "opensandbox/code-interpreter"
    default_contract: GovernanceContract = field(
        default_factory=GovernanceContract
    )
    contracts: Dict[str, GovernanceContract] = field(default_factory=dict)
    max_sandboxes: int = 20
    proxy_sigmoid_k: float = 2.0
    max_interactions: int = 50_000
    max_events: int = 50_000
    message_bus_max_pending: int = 10_000
    provenance_enabled: bool = True
    snapshot_enabled: bool = False
    observer_interval_seconds: float = 5.0
    risk_threshold: float = 0.7

    def get_contract(self, contract_id: str) -> GovernanceContract:
        """Look up a contract by ID, falling back to the default."""
        return self.contracts.get(contract_id, self.default_contract)

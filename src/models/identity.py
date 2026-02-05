"""Identity and trust infrastructure for Sybil resistance.

Implements verifiable credentials, Proof-of-Personhood, and agent identity
models for the simulation. These are abstracted versions of the
cryptographic infrastructure proposed in the paper, suitable for
simulation-level analysis of trust and identity dynamics.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class VerifiableCredential:
    """An unforgeable claim about an agent's history."""

    credential_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issuer_id: str = ""
    subject_id: str = ""
    claim_type: str = ""  # "reputation_history", "audit_pass", "task_completion"
    claim_value: Any = None
    issued_epoch: int = 0
    expires_epoch: Optional[int] = None
    revoked: bool = False

    def is_valid(self, current_epoch: int) -> bool:
        """Check if credential is currently valid."""
        if self.revoked:
            return False
        if self.expires_epoch is not None and current_epoch >= self.expires_epoch:
            return False
        return True

    def to_dict(self) -> Dict:
        """Serialize credential."""
        return {
            "credential_id": self.credential_id,
            "issuer_id": self.issuer_id,
            "subject_id": self.subject_id,
            "claim_type": self.claim_type,
            "claim_value": self.claim_value,
            "issued_epoch": self.issued_epoch,
            "expires_epoch": self.expires_epoch,
            "revoked": self.revoked,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VerifiableCredential":
        """Create from dictionary."""
        return cls(
            credential_id=data["credential_id"],
            issuer_id=data["issuer_id"],
            subject_id=data["subject_id"],
            claim_type=data["claim_type"],
            claim_value=data.get("claim_value"),
            issued_epoch=data["issued_epoch"],
            expires_epoch=data.get("expires_epoch"),
            revoked=data.get("revoked", False),
        )


@dataclass
class AgentIdentity:
    """Extended identity for an agent with trust infrastructure."""

    agent_id: str = ""
    identity_cost: float = 0.0
    creation_epoch: int = 0
    credentials: List[VerifiableCredential] = field(default_factory=list)
    proof_of_personhood: bool = False
    linked_identities: Set[str] = field(default_factory=set)
    trust_score: float = 0.5

    def has_credential(self, claim_type: str, current_epoch: int = 0) -> bool:
        """Check if agent has a valid credential of the given type."""
        return any(
            c.claim_type == claim_type and c.is_valid(current_epoch)
            for c in self.credentials
        )

    def get_credential_value(
        self, claim_type: str, current_epoch: int = 0
    ) -> Optional[Any]:
        """Get the value of a valid credential of the given type."""
        for c in self.credentials:
            if c.claim_type == claim_type and c.is_valid(current_epoch):
                return c.claim_value
        return None

    def compute_trust_score(self, current_epoch: int = 0) -> float:
        """
        Compute trust score from valid credentials.

        Trust is computed as:
        - Base: 0.3 (new identity)
        - +0.2 for Proof-of-Personhood
        - +0.1 per valid credential (max 0.5 from credentials)
        """
        score = 0.3

        if self.proof_of_personhood:
            score += 0.2

        valid_creds = sum(
            1 for c in self.credentials if c.is_valid(current_epoch)
        )
        score += min(0.5, valid_creds * 0.1)

        self.trust_score = min(1.0, score)
        return self.trust_score

    def to_dict(self) -> Dict:
        """Serialize identity."""
        return {
            "agent_id": self.agent_id,
            "identity_cost": self.identity_cost,
            "creation_epoch": self.creation_epoch,
            "credentials": [c.to_dict() for c in self.credentials],
            "proof_of_personhood": self.proof_of_personhood,
            "linked_identities": list(self.linked_identities),
            "trust_score": self.trust_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentIdentity":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            identity_cost=data.get("identity_cost", 0.0),
            creation_epoch=data.get("creation_epoch", 0),
            credentials=[
                VerifiableCredential.from_dict(c)
                for c in data.get("credentials", [])
            ],
            proof_of_personhood=data.get("proof_of_personhood", False),
            linked_identities=set(data.get("linked_identities", [])),
            trust_score=data.get("trust_score", 0.5),
        )


@dataclass
class IdentityConfig:
    """Configuration for identity infrastructure."""

    identity_creation_cost: float = 10.0
    proof_of_personhood_required: bool = False
    credential_expiry_epochs: int = 50
    sybil_detection_enabled: bool = True
    max_identities_per_entity: int = 1
    behavioral_similarity_threshold: float = 0.8

    def validate(self) -> None:
        """Validate configuration."""
        if self.identity_creation_cost < 0:
            raise ValueError("identity_creation_cost must be non-negative")
        if self.credential_expiry_epochs < 1:
            raise ValueError("credential_expiry_epochs must be >= 1")
        if self.max_identities_per_entity < 1:
            raise ValueError("max_identities_per_entity must be >= 1")
        if not 0.0 <= self.behavioral_similarity_threshold <= 1.0:
            raise ValueError("behavioral_similarity_threshold must be in [0, 1]")


class CredentialIssuer:
    """System-level credential issuance and verification."""

    def __init__(self, config: Optional[IdentityConfig] = None):
        """Initialize credential issuer."""
        self.config = config or IdentityConfig()
        self._issued: Dict[str, VerifiableCredential] = {}

    def issue_credential(
        self,
        subject_id: str,
        claim_type: str,
        claim_value: Any,
        current_epoch: int,
        issuer_id: str = "system",
    ) -> VerifiableCredential:
        """Issue a new credential."""
        cred = VerifiableCredential(
            issuer_id=issuer_id,
            subject_id=subject_id,
            claim_type=claim_type,
            claim_value=claim_value,
            issued_epoch=current_epoch,
            expires_epoch=current_epoch + self.config.credential_expiry_epochs,
        )
        self._issued[cred.credential_id] = cred
        return cred

    def verify_credential(
        self, credential_id: str, current_epoch: int
    ) -> bool:
        """Verify that a credential is valid and not revoked."""
        cred = self._issued.get(credential_id)
        if not cred:
            return False
        return cred.is_valid(current_epoch)

    def revoke_credential(self, credential_id: str) -> bool:
        """Revoke a credential."""
        cred = self._issued.get(credential_id)
        if not cred:
            return False
        cred.revoked = True
        return True

    def issue_reputation_credential(
        self,
        subject_id: str,
        reputation: float,
        current_epoch: int,
    ) -> VerifiableCredential:
        """Issue a reputation attestation credential."""
        return self.issue_credential(
            subject_id=subject_id,
            claim_type="reputation_history",
            claim_value={"reputation": reputation, "epoch": current_epoch},
            current_epoch=current_epoch,
        )

    def issue_audit_pass(
        self,
        subject_id: str,
        current_epoch: int,
    ) -> VerifiableCredential:
        """Issue an audit pass credential."""
        return self.issue_credential(
            subject_id=subject_id,
            claim_type="audit_pass",
            claim_value={"epoch": current_epoch},
            current_epoch=current_epoch,
        )


class IdentityRegistry:
    """
    Registry for managing agent identities and detecting Sybils.

    Provides:
    - Identity creation with cost
    - Proof-of-Personhood enforcement
    - Behavioral similarity analysis for Sybil detection
    """

    def __init__(self, config: Optional[IdentityConfig] = None):
        """Initialize the identity registry."""
        self.config = config or IdentityConfig()
        self.config.validate()
        self._identities: Dict[str, AgentIdentity] = {}
        self._entity_map: Dict[str, Set[str]] = {}  # entity -> {agent_ids}
        self._sybil_clusters: List[Set[str]] = []

    def create_identity(
        self,
        agent_id: str,
        entity_id: Optional[str] = None,
        proof_of_personhood: bool = False,
        current_epoch: int = 0,
    ) -> Optional[AgentIdentity]:
        """
        Create a new agent identity.

        Args:
            agent_id: Unique agent ID
            entity_id: Optional entity that controls this identity
            proof_of_personhood: Whether PoP has been verified
            current_epoch: Current simulation epoch

        Returns:
            AgentIdentity if created, None if blocked (e.g., too many identities)
        """
        if agent_id in self._identities:
            return None

        # Check max identities per entity
        if entity_id:
            existing = self._entity_map.get(entity_id, set())
            if len(existing) >= self.config.max_identities_per_entity:
                return None
            if entity_id not in self._entity_map:
                self._entity_map[entity_id] = set()
            self._entity_map[entity_id].add(agent_id)

        # Check PoP requirement
        if self.config.proof_of_personhood_required and not proof_of_personhood:
            return None

        identity = AgentIdentity(
            agent_id=agent_id,
            identity_cost=self.config.identity_creation_cost,
            creation_epoch=current_epoch,
            proof_of_personhood=proof_of_personhood,
        )
        identity.compute_trust_score(current_epoch)

        self._identities[agent_id] = identity
        return identity

    def get_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get an agent's identity."""
        return self._identities.get(agent_id)

    def detect_sybil_clusters(
        self,
        interaction_patterns: Dict[str, Dict[str, int]],
    ) -> List[Set[str]]:
        """
        Detect Sybil clusters based on behavioral similarity.

        Agents with highly similar interaction patterns (same counterparties,
        similar timing, correlated resource flows) are flagged as potential
        Sybils controlled by the same entity.

        Args:
            interaction_patterns: agent_id -> {counterparty_id: count}

        Returns:
            List of Sybil clusters (sets of suspected same-entity agent IDs)
        """
        if not self.config.sybil_detection_enabled:
            return []

        agent_ids = list(interaction_patterns.keys())
        clusters: List[Set[str]] = []
        visited = set()

        for i, a_id in enumerate(agent_ids):
            if a_id in visited:
                continue

            cluster = {a_id}
            for j in range(i + 1, len(agent_ids)):
                b_id = agent_ids[j]
                if b_id in visited:
                    continue

                similarity = self._behavioral_similarity(
                    interaction_patterns.get(a_id, {}),
                    interaction_patterns.get(b_id, {}),
                )

                if similarity >= self.config.behavioral_similarity_threshold:
                    cluster.add(b_id)

            if len(cluster) > 1:
                clusters.append(cluster)
                visited.update(cluster)

        self._sybil_clusters = clusters
        return clusters

    def _behavioral_similarity(
        self,
        pattern_a: Dict[str, int],
        pattern_b: Dict[str, int],
    ) -> float:
        """
        Compute behavioral similarity between two agents.

        Uses Jaccard similarity of counterparty sets weighted by
        interaction frequency correlation.
        """
        if not pattern_a or not pattern_b:
            return 0.0

        keys_a = set(pattern_a.keys())
        keys_b = set(pattern_b.keys())

        intersection = keys_a & keys_b
        union = keys_a | keys_b

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Also check frequency correlation for shared counterparties
        if intersection:
            freq_a = [pattern_a[k] for k in intersection]
            freq_b = [pattern_b[k] for k in intersection]
            total_a = sum(freq_a)
            total_b = sum(freq_b)

            if total_a > 0 and total_b > 0:
                # Normalized frequency vectors
                norm_a = [f / total_a for f in freq_a]
                norm_b = [f / total_b for f in freq_b]
                # Cosine similarity
                dot = sum(a * b for a, b in zip(norm_a, norm_b, strict=True))
                return (jaccard + dot) / 2

        return jaccard

    def get_sybil_clusters(self) -> List[Set[str]]:
        """Get last detected Sybil clusters."""
        return list(self._sybil_clusters)

    def get_trust_distribution(self) -> Dict[str, float]:
        """Get trust scores for all identities."""
        return {
            agent_id: identity.trust_score
            for agent_id, identity in self._identities.items()
        }

    def pop_verification_rate(self) -> float:
        """Fraction of identities with Proof-of-Personhood."""
        if not self._identities:
            return 0.0
        verified = sum(
            1 for i in self._identities.values() if i.proof_of_personhood
        )
        return verified / len(self._identities)

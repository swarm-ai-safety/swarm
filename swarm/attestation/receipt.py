"""Cryptographic Admissibility Receipt model.

Each receipt seals a single agent action with provenance, policy compliance,
and execution bounds — creating a "run once, verify forever" primitive.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ReceiptStatus(str, Enum):
    """Lifecycle status of an admissibility receipt."""

    PENDING = "pending"
    SEALED = "sealed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVOKED = "revoked"


class PolicyCompliance(BaseModel):
    """Policy compliance attestation embedded in a receipt."""

    policy_id: str = Field(..., description="Identifier of the evaluated policy")
    passed: bool = Field(..., description="Whether the action satisfied the policy")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy-specific evaluation details",
    )


class ExecutionBounds(BaseModel):
    """Execution bounds that constrain an attested action.

    These bounds are checked at seal time and encoded in the receipt so
    verifiers can confirm the action stayed within its authorised envelope.
    """

    max_resource_spend: Optional[float] = Field(
        None, description="Maximum resource units the action may consume"
    )
    max_delegation_depth: Optional[int] = Field(
        None, description="Maximum sub-delegation depth (0 = no delegation)"
    )
    allowed_targets: Optional[List[str]] = Field(
        None,
        description="Allowlist of agent/entity IDs the action may affect",
    )
    deadline: Optional[datetime] = Field(
        None, description="Hard deadline after which the action is inadmissible"
    )

    @field_validator("max_resource_spend")
    @classmethod
    def _non_negative_spend(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("max_resource_spend must be non-negative")
        return v

    @field_validator("max_delegation_depth")
    @classmethod
    def _non_negative_depth(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("max_delegation_depth must be non-negative")
        return v


class AdmissibilityReceipt(BaseModel):
    """Cryptographic Admissibility Receipt (CAR).

    Seals a single agent action with signed provenance, policy compliance,
    and execution bounds.  Once sealed, the receipt is immutable and can be
    verified offline by any participant in the swarm.
    """

    # Identity
    receipt_id: str = Field(
        ..., description="Unique receipt identifier (deterministic hash)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: ReceiptStatus = Field(default=ReceiptStatus.PENDING)

    # Provenance
    agent_id: str = Field(..., description="Agent that executed the action")
    action_type: str = Field(..., description="Type of action being attested")
    event_id: Optional[str] = Field(
        None, description="ID of the Event this receipt covers"
    )
    parent_receipt_ids: List[str] = Field(
        default_factory=list,
        description="Receipts of upstream actions this action depends on",
    )

    # Payload digest — the receipt does not store the full payload, only its
    # content-addressed hash so the receipt stays small and the payload can
    # be verified independently.
    payload_hash: str = Field(
        ..., description="SHA-256 hex digest of the canonical action payload"
    )

    # Policy & bounds
    policy_results: List[PolicyCompliance] = Field(
        default_factory=list,
        description="Results of policy evaluations at seal time",
    )
    bounds: ExecutionBounds = Field(
        default_factory=ExecutionBounds,
        description="Execution envelope the action must respect",
    )

    # Cryptographic seal
    signature: Optional[str] = Field(
        None,
        description="Hex-encoded HMAC-SHA256 (or Ed25519) signature over the canonical receipt",
    )
    signer_id: Optional[str] = Field(
        None, description="Identity of the signer (key ID or agent ID)"
    )

    # Routing
    scenario_id: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def canonical_bytes(self) -> bytes:
        """Return the deterministic byte representation used for signing.

        Excludes ``signature`` and ``signer_id`` so they can be set after
        computing the digest.
        """
        obj = self.model_dump(exclude={"signature", "signer_id", "status"})
        # datetime → ISO string for JSON determinism
        if obj.get("timestamp"):
            obj["timestamp"] = obj["timestamp"].isoformat()
        if obj.get("bounds") and obj["bounds"].get("deadline"):
            obj["bounds"]["deadline"] = obj["bounds"]["deadline"].isoformat()
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

    def content_hash(self) -> str:
        """SHA-256 hex digest of the canonical receipt (pre-signature)."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @staticmethod
    def hash_payload(payload: Dict[str, Any]) -> str:
        """Compute the canonical SHA-256 digest of an action payload."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def generate_receipt_id(
        agent_id: str,
        action_type: str,
        payload_hash: str,
        timestamp: datetime,
    ) -> str:
        """Deterministic receipt ID from core attributes."""
        components = json.dumps(
            {
                "agent_id": agent_id,
                "action_type": action_type,
                "payload_hash": payload_hash,
                "timestamp": timestamp.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(components.encode()).hexdigest()[:24]

    def is_admissible(self) -> bool:
        """Return True if the receipt is sealed and all policies passed."""
        if self.status not in (ReceiptStatus.SEALED, ReceiptStatus.VERIFIED):
            return False
        return all(pr.passed for pr in self.policy_results)

"""Byline provenance tracking for the OpenSandbox bridge.

Every action — code execution, file write, message sent, governance
intervention — carries a Byline provenance signature.  This creates
a complete, attributable history of multi-agent interactions for
post-hoc safety analysis.

Security properties (H3):
- Content integrity via SHA-256 hash of action payload.
- Chain integrity via incorporating the parent record's hash into
  each new record (Merkle-style linked chain).
- Optional HMAC signing when ``hmac_key`` is provided, making the
  chain tamper-evident to any party without the key.
"""

import hashlib
import hmac
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from swarm.bridges.opensandbox.events import OpenSandboxEvent, OpenSandboxEventType

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """A single provenance entry in the Byline audit trail.

    Attributes:
        provenance_id: Unique identifier for this record.
        sandbox_id: The sandbox where the action occurred.
        agent_id: The agent that performed the action.
        action_type: Category of action (exec, file_write, message, etc).
        action_summary: Human-readable summary of what happened.
        content_hash: SHA-256 hash of the action payload for integrity.
        chain_hash: Hash incorporating parent's chain_hash for
            tamper-evident chain linking.
        hmac_signature: HMAC-SHA256 signature (if key configured).
        parent_id: Provenance ID of the causal parent (chain linking).
        contract_id: Governing contract at time of action.
        timestamp: When the action occurred.
        metadata: Additional context.
    """

    provenance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sandbox_id: str = ""
    agent_id: str = ""
    action_type: str = ""
    action_summary: str = ""
    content_hash: str = ""
    chain_hash: str = ""
    hmac_signature: str = ""
    parent_id: Optional[str] = None
    contract_id: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the audit trail."""
        result: Dict[str, Any] = {
            "provenance_id": self.provenance_id,
            "sandbox_id": self.sandbox_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "action_summary": self.action_summary,
            "content_hash": self.content_hash,
            "chain_hash": self.chain_hash,
            "parent_id": self.parent_id,
            "contract_id": self.contract_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.hmac_signature:
            result["hmac_signature"] = self.hmac_signature
        return result


class ProvenanceTracker:
    """Byline provenance tracker for multi-agent audit trails.

    Maintains a chain of provenance records that links every action
    to the agent and sandbox that produced it, along with the
    governing contract and a content integrity hash.

    When ``hmac_key`` is set, each record also receives an
    HMAC-SHA256 signature computed over the chain hash, making
    the chain tamper-evident.

    Example::

        tracker = ProvenanceTracker(hmac_key="secret")
        prov_id = tracker.sign(
            sandbox_id="sandbox-a",
            agent_id="agent-a",
            action_type="exec",
            action_summary="python evaluate.py",
            content={"command": "python evaluate.py", "exit_code": 0},
            contract_id="restricted-v1",
        )
        chain = tracker.get_chain("agent-a")
        assert tracker.verify_chain("agent-a")  # True
    """

    def __init__(
        self,
        enabled: bool = True,
        hmac_key: Optional[str] = None,
        max_records: int = 100_000,
    ) -> None:
        self._enabled = enabled
        self._hmac_key = hmac_key.encode() if hmac_key else None
        self._max_records = max_records
        self._lock = threading.Lock()
        # agent_id -> list of provenance records (ordered)
        self._chains: Dict[str, List[ProvenanceRecord]] = {}
        # provenance_id -> record (index)
        self._index: Dict[str, ProvenanceRecord] = {}
        # Total count for truncation
        self._total_records = 0
        # Events
        self._events: List[OpenSandboxEvent] = []

    def sign(
        self,
        sandbox_id: str,
        agent_id: str,
        action_type: str,
        action_summary: str,
        content: Optional[Dict[str, Any]] = None,
        contract_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a provenance record and return its ID.

        If a parent_id is not provided, the most recent record for
        the agent is used as the parent (building an ordered chain).

        Args:
            sandbox_id: Sandbox where the action happened.
            agent_id: Agent that performed the action.
            action_type: Category (exec, file_write, message, governance).
            action_summary: Human-readable summary.
            content: Action payload for hashing.
            contract_id: Governing contract.
            parent_id: Explicit parent provenance record.
            metadata: Extra context.

        Returns:
            The provenance_id of the new record.
        """
        if not self._enabled:
            return str(uuid.uuid4())

        content_hash = self._hash_content(content or {})

        with self._lock:
            # Auto-chain: use the most recent record as parent if not given
            parent_chain_hash = ""
            if parent_id is None:
                chain = self._chains.get(agent_id, [])
                if chain:
                    parent_id = chain[-1].provenance_id
                    parent_chain_hash = chain[-1].chain_hash
            elif parent_id in self._index:
                parent_chain_hash = self._index[parent_id].chain_hash

            # H3: Chain hash incorporates parent's chain hash
            chain_hash = self._compute_chain_hash(
                content_hash, parent_chain_hash
            )

            # H3: Optional HMAC signature
            hmac_sig = ""
            if self._hmac_key:
                hmac_sig = hmac.new(
                    self._hmac_key,
                    chain_hash.encode(),
                    hashlib.sha256,
                ).hexdigest()

            record = ProvenanceRecord(
                sandbox_id=sandbox_id,
                agent_id=agent_id,
                action_type=action_type,
                action_summary=action_summary,
                content_hash=content_hash,
                chain_hash=chain_hash,
                hmac_signature=hmac_sig,
                parent_id=parent_id,
                contract_id=contract_id,
                metadata=metadata or {},
            )

            self._chains.setdefault(agent_id, []).append(record)
            self._index[record.provenance_id] = record
            self._total_records += 1

            # M3: Bounded growth — drop oldest records when limit hit
            if self._total_records > self._max_records:
                self._truncate_oldest_locked()

        self._events.append(
            OpenSandboxEvent(
                event_type=OpenSandboxEventType.PROVENANCE_SIGNED,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                contract_id=contract_id,
                provenance_id=record.provenance_id,
                payload={
                    "action_type": action_type,
                    "action_summary": action_summary,
                    "parent_id": parent_id,
                },
            )
        )

        logger.debug(
            "Provenance signed: %s [%s] %s -> %s",
            record.provenance_id,
            action_type,
            agent_id,
            action_summary,
        )
        return record.provenance_id

    # ------------------------------------------------------------------
    # Verification (H3)
    # ------------------------------------------------------------------

    def verify_chain(self, agent_id: str) -> bool:
        """Verify the integrity of an agent's provenance chain.

        Checks that each record's chain_hash is consistent with its
        content_hash and its parent's chain_hash, and that HMAC
        signatures (if present) are valid.

        Returns:
            True if the chain is intact, False otherwise.
        """
        with self._lock:
            chain = self._chains.get(agent_id, [])
            if not chain:
                return True

            prev_chain_hash = ""
            for record in chain:
                expected = self._compute_chain_hash(
                    record.content_hash, prev_chain_hash
                )
                if record.chain_hash != expected:
                    logger.warning(
                        "Chain hash mismatch for %s at %s",
                        agent_id,
                        record.provenance_id,
                    )
                    return False
                if self._hmac_key and record.hmac_signature:
                    expected_sig = hmac.new(
                        self._hmac_key,
                        record.chain_hash.encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    if not hmac.compare_digest(
                        record.hmac_signature, expected_sig
                    ):
                        logger.warning(
                            "HMAC mismatch for %s at %s",
                            agent_id,
                            record.provenance_id,
                        )
                        return False
                prev_chain_hash = record.chain_hash
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_record(self, provenance_id: str) -> Optional[ProvenanceRecord]:
        """Look up a single provenance record by ID."""
        with self._lock:
            return self._index.get(provenance_id)

    def get_chain(self, agent_id: str) -> List[ProvenanceRecord]:
        """Return the full provenance chain for an agent."""
        with self._lock:
            return list(self._chains.get(agent_id, []))

    def get_all_records(self) -> List[ProvenanceRecord]:
        """Return all provenance records across all agents."""
        with self._lock:
            records: List[ProvenanceRecord] = []
            for chain in self._chains.values():
                records.extend(chain)
        records.sort(key=lambda r: r.timestamp)
        return records

    def get_events(self) -> List[OpenSandboxEvent]:
        """Return all provenance events."""
        return list(self._events)

    def get_stats(self) -> Dict[str, Any]:
        """Return provenance statistics."""
        with self._lock:
            total = sum(len(c) for c in self._chains.values())
            by_type: Dict[str, int] = {}
            for chain in self._chains.values():
                for rec in chain:
                    by_type[rec.action_type] = (
                        by_type.get(rec.action_type, 0) + 1
                    )
            return {
                "total_records": total,
                "agents_tracked": len(self._chains),
                "by_action_type": by_type,
                "hmac_enabled": self._hmac_key is not None,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_content(content: Dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hash of the content dict."""
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _compute_chain_hash(
        content_hash: str, parent_chain_hash: str
    ) -> str:
        """Compute a chain hash incorporating the parent's hash."""
        combined = f"{parent_chain_hash}:{content_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _truncate_oldest_locked(self) -> None:
        """Drop oldest half of records.  Caller must hold _lock."""
        target = self._max_records // 2
        removed = 0
        for agent_id in list(self._chains.keys()):
            chain = self._chains[agent_id]
            if not chain:
                continue
            drop_count = len(chain) // 2
            if drop_count == 0:
                continue
            dropped = chain[:drop_count]
            self._chains[agent_id] = chain[drop_count:]
            for rec in dropped:
                self._index.pop(rec.provenance_id, None)
            removed += drop_count
            if removed >= target:
                break
        self._total_records -= removed

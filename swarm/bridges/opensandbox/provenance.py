"""Byline provenance tracking for the OpenSandbox bridge.

Every action — code execution, file write, message sent, governance
intervention — carries a Byline provenance signature.  This creates
a complete, attributable history of multi-agent interactions for
post-hoc safety analysis.
"""

import hashlib
import logging
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
    parent_id: Optional[str] = None
    contract_id: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the audit trail."""
        return {
            "provenance_id": self.provenance_id,
            "sandbox_id": self.sandbox_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "action_summary": self.action_summary,
            "content_hash": self.content_hash,
            "parent_id": self.parent_id,
            "contract_id": self.contract_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ProvenanceTracker:
    """Byline provenance tracker for multi-agent audit trails.

    Maintains a chain of provenance records that links every action
    to the agent and sandbox that produced it, along with the
    governing contract and a content integrity hash.

    Example::

        tracker = ProvenanceTracker()
        prov_id = tracker.sign(
            sandbox_id="sandbox-a",
            agent_id="agent-a",
            action_type="exec",
            action_summary="python evaluate.py",
            content={"command": "python evaluate.py", "exit_code": 0},
            contract_id="restricted-v1",
        )
        chain = tracker.get_chain("agent-a")
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        # agent_id -> list of provenance records (ordered)
        self._chains: Dict[str, List[ProvenanceRecord]] = {}
        # provenance_id -> record (index)
        self._index: Dict[str, ProvenanceRecord] = {}
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

        # Auto-chain: use the most recent record as parent if not given
        if parent_id is None:
            chain = self._chains.get(agent_id, [])
            if chain:
                parent_id = chain[-1].provenance_id

        content_hash = self._hash_content(content or {})

        record = ProvenanceRecord(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            action_type=action_type,
            action_summary=action_summary,
            content_hash=content_hash,
            parent_id=parent_id,
            contract_id=contract_id,
            metadata=metadata or {},
        )

        self._chains.setdefault(agent_id, []).append(record)
        self._index[record.provenance_id] = record

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
    # Query
    # ------------------------------------------------------------------

    def get_record(self, provenance_id: str) -> Optional[ProvenanceRecord]:
        """Look up a single provenance record by ID."""
        return self._index.get(provenance_id)

    def get_chain(self, agent_id: str) -> List[ProvenanceRecord]:
        """Return the full provenance chain for an agent."""
        return list(self._chains.get(agent_id, []))

    def get_all_records(self) -> List[ProvenanceRecord]:
        """Return all provenance records across all agents."""
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
        total = sum(len(c) for c in self._chains.values())
        by_type: Dict[str, int] = {}
        for chain in self._chains.values():
            for rec in chain:
                by_type[rec.action_type] = by_type.get(rec.action_type, 0) + 1
        return {
            "total_records": total,
            "agents_tracked": len(self._chains),
            "by_action_type": by_type,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_content(content: Dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hash of the content dict."""
        import json

        # Sort keys for determinism
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

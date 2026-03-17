"""Receipt relay — propagates receipts across the swarm.

The relay stores sealed receipts and exposes query methods that downstream
agents use to condition behaviour on admissible, verified state.  It also
supports relational messaging: agents can send messages that carry receipt
references, so the recipient can verify provenance before acting.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from swarm.attestation.receipt import AdmissibilityReceipt, ReceiptStatus
from swarm.attestation.signer import ReceiptVerifier


class RelayMessage(BaseModel):
    """A message propagated through the receipt relay.

    Every relay message must reference at least one admissibility receipt.
    Messages without a valid receipt reference are rejected at submission.
    """

    message_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    from_agent: str
    to_agent: str  # agent ID or "#broadcast"
    body: str
    receipt_ids: List[str] = Field(
        ...,
        description="Receipt IDs that authorise this message",
        min_length=1,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False


class ReceiptRelay:
    """Central relay for receipt propagation and relational messaging.

    Parameters
    ----------
    verifier:
        Optional ``ReceiptVerifier`` used to re-verify receipts on ingest.
        If *None*, receipts are accepted as-is (useful in trusted
        single-process simulations).
    """

    def __init__(self, verifier: Optional[ReceiptVerifier] = None) -> None:
        self._verifier = verifier
        self._receipts: Dict[str, AdmissibilityReceipt] = {}
        self._messages: List[RelayMessage] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Receipt management
    # ------------------------------------------------------------------ #

    def ingest(self, receipt: AdmissibilityReceipt) -> bool:
        """Add a receipt to the relay.

        If a verifier is configured, the receipt must pass verification.
        Returns *True* if the receipt was accepted.
        """
        if self._verifier:
            if not self._verifier.verify(receipt):
                return False

        with self._lock:
            self._receipts[receipt.receipt_id] = receipt
        return True

    def get(self, receipt_id: str) -> Optional[AdmissibilityReceipt]:
        with self._lock:
            return self._receipts.get(receipt_id)

    def query_by_agent(self, agent_id: str) -> List[AdmissibilityReceipt]:
        """Return all receipts for a given agent."""
        with self._lock:
            return [
                r for r in self._receipts.values() if r.agent_id == agent_id
            ]

    def query_admissible(
        self, agent_id: Optional[str] = None
    ) -> List[AdmissibilityReceipt]:
        """Return only admissible receipts, optionally filtered by agent."""
        with self._lock:
            results = [r for r in self._receipts.values() if r.is_admissible()]
            if agent_id:
                results = [r for r in results if r.agent_id == agent_id]
            return results

    def chain_valid(self, receipt_id: str) -> bool:
        """Verify that the full parent chain of a receipt is admissible.

        Walks ``parent_receipt_ids`` recursively.  Returns *False* if any
        ancestor is missing or inadmissible.
        """
        with self._lock:
            return self._chain_valid_locked(receipt_id, set())

    def _chain_valid_locked(
        self, receipt_id: str, visited: set[str]
    ) -> bool:
        if receipt_id in visited:
            return False  # cycle — inadmissible
        visited.add(receipt_id)

        receipt = self._receipts.get(receipt_id)
        if receipt is None:
            return False
        if not receipt.is_admissible():
            return False
        for parent_id in receipt.parent_receipt_ids:
            if not self._chain_valid_locked(parent_id, visited):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Relational messaging
    # ------------------------------------------------------------------ #

    def send_message(self, message: RelayMessage) -> bool:
        """Submit a relay message.

        The message is accepted only if every referenced receipt_id exists
        in the relay and is admissible.
        """
        with self._lock:
            for rid in message.receipt_ids:
                r = self._receipts.get(rid)
                if r is None or not r.is_admissible():
                    return False
            self._messages.append(message)
        return True

    def inbox(
        self,
        agent_id: str,
        unacknowledged_only: bool = True,
    ) -> List[RelayMessage]:
        """Return messages addressed to *agent_id* (or ``#broadcast``).

        Parameters
        ----------
        agent_id:
            The receiving agent's ID.
        unacknowledged_only:
            If *True* (default), return only messages not yet acknowledged.
        """
        with self._lock:
            msgs = [
                m
                for m in self._messages
                if m.to_agent in (agent_id, "#broadcast")
                and (not unacknowledged_only or not m.acknowledged)
            ]
        return msgs

    def acknowledge(self, message_id: str) -> bool:
        """Mark a message as acknowledged."""
        with self._lock:
            for m in self._messages:
                if m.message_id == message_id:
                    m.acknowledged = True
                    return True
        return False

    @property
    def receipt_count(self) -> int:
        with self._lock:
            return len(self._receipts)

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

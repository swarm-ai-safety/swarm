"""Attestation and Relational Messaging API.

Provides Cryptographic Admissibility Receipts (CARs) that seal every agent
action with a signed receipt verifying provenance, policy compliance, and
execution bounds.  Receipts propagate across the swarm as first-class objects
so downstream agents can condition behaviour only on admissible,
hardware-verified state.
"""

from swarm.attestation.receipt import (
    AdmissibilityReceipt,
    ExecutionBounds,
    PolicyCompliance,
    ReceiptStatus,
)
from swarm.attestation.signer import ReceiptSigner, ReceiptVerifier
from swarm.attestation.middleware import AttestationMiddleware
from swarm.attestation.relay import ReceiptRelay

__all__ = [
    "AdmissibilityReceipt",
    "ExecutionBounds",
    "PolicyCompliance",
    "ReceiptStatus",
    "ReceiptSigner",
    "ReceiptVerifier",
    "AttestationMiddleware",
    "ReceiptRelay",
]

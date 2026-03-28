"""Pydantic models for the Attestation & Relational Messaging API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReceiptQuery(BaseModel):
    """Query parameters for receipt lookup."""

    agent_id: Optional[str] = None
    admissible_only: bool = False


class ReceiptResponse(BaseModel):
    """Serialised admissibility receipt returned by the API."""

    receipt_id: str
    timestamp: datetime
    status: str
    agent_id: str
    action_type: str
    event_id: Optional[str] = None
    parent_receipt_ids: List[str] = Field(default_factory=list)
    payload_hash: str
    policy_results: List[Dict[str, Any]] = Field(default_factory=list)
    bounds: Dict[str, Any] = Field(default_factory=dict)
    signature: Optional[str] = None
    signer_id: Optional[str] = None
    confidence: Optional[float] = None
    admissible: bool = False


class ChainValidationResponse(BaseModel):
    """Result of a receipt chain validation."""

    receipt_id: str
    chain_valid: bool


class MessageSubmission(BaseModel):
    """Payload for submitting a relay message."""

    from_agent: str
    to_agent: str
    body: str
    receipt_ids: List[str] = Field(
        ..., min_length=1, description="At least one receipt must back the message"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Serialised relay message returned by the API."""

    message_id: str
    timestamp: datetime
    from_agent: str
    to_agent: str
    body: str
    receipt_ids: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False


class AttestActionRequest(BaseModel):
    """Request to attest an action (create + seal a receipt)."""

    agent_id: str
    action_type: str
    payload: Dict[str, Any]
    event_id: Optional[str] = None
    parent_receipt_ids: List[str] = Field(default_factory=list)
    bounds: Optional[Dict[str, Any]] = None
    scenario_id: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None

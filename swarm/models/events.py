"""Event schema for append-only logging."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of events that can be logged."""

    # Interaction lifecycle
    INTERACTION_PROPOSED = "interaction_proposed"
    INTERACTION_ACCEPTED = "interaction_accepted"
    INTERACTION_REJECTED = "interaction_rejected"
    INTERACTION_COMPLETED = "interaction_completed"

    # Agent events
    AGENT_CREATED = "agent_created"
    AGENT_STATE_UPDATED = "agent_state_updated"

    # Proxy computation
    PROXY_COMPUTED = "proxy_computed"

    # Payoff events
    PAYOFF_COMPUTED = "payoff_computed"
    TRANSFER_EXECUTED = "transfer_executed"

    # Governance events
    GOVERNANCE_COST_APPLIED = "governance_cost_applied"
    REPUTATION_UPDATED = "reputation_updated"

    # Marketplace events
    BOUNTY_POSTED = "bounty_posted"
    BID_PLACED = "bid_placed"
    BID_ACCEPTED = "bid_accepted"
    BID_REJECTED = "bid_rejected"
    ESCROW_CREATED = "escrow_created"
    ESCROW_RELEASED = "escrow_released"
    ESCROW_REFUNDED = "escrow_refunded"
    DISPUTE_FILED = "dispute_filed"
    DISPUTE_RESOLVED = "dispute_resolved"

    # Moltipedia wiki events
    PAGE_CREATED = "page_created"
    PAGE_EDITED = "page_edited"
    OBJECTION_FILED = "objection_filed"
    POLICY_VIOLATION_FLAGGED = "policy_violation_flagged"
    POINTS_AWARDED = "points_awarded"
    PAIR_CAP_TRIGGERED = "pair_cap_triggered"
    COOLDOWN_TRIGGERED = "cooldown_triggered"
    DAILY_CAP_TRIGGERED = "daily_cap_triggered"

    # Moltbook events
    POST_SUBMITTED = "post_submitted"
    COMMENT_SUBMITTED = "comment_submitted"
    CHALLENGE_ISSUED = "challenge_issued"
    CHALLENGE_PASSED = "challenge_passed"
    CHALLENGE_FAILED = "challenge_failed"
    CHALLENGE_EXPIRED = "challenge_expired"
    CONTENT_PUBLISHED = "content_published"
    RATE_LIMIT_HIT = "rate_limit_hit"
    KARMA_UPDATED = "karma_updated"

    # System events
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_ENDED = "simulation_ended"
    EPOCH_COMPLETED = "epoch_completed"


@dataclass
class Event:
    """
    An immutable event for the append-only log.

    Events capture state changes and can be replayed to reconstruct
    the full history of a simulation.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = EventType.INTERACTION_PROPOSED

    # References to related entities
    interaction_id: Optional[str] = None
    agent_id: Optional[str] = None
    initiator_id: Optional[str] = None
    counterparty_id: Optional[str] = None

    # Event payload (flexible structure)
    payload: dict = field(default_factory=dict)

    # Metadata
    epoch: Optional[int] = None
    step: Optional[int] = None
    scenario_id: Optional[str] = None
    replay_k: Optional[int] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "interaction_id": self.interaction_id,
            "agent_id": self.agent_id,
            "initiator_id": self.initiator_id,
            "counterparty_id": self.counterparty_id,
            "payload": self.payload,
            "epoch": self.epoch,
            "step": self.step,
            "scenario_id": self.scenario_id,
            "replay_k": self.replay_k,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=EventType(data["event_type"]),
            interaction_id=data.get("interaction_id"),
            agent_id=data.get("agent_id"),
            initiator_id=data.get("initiator_id"),
            counterparty_id=data.get("counterparty_id"),
            payload=data.get("payload", {}),
            epoch=data.get("epoch"),
            step=data.get("step"),
            scenario_id=data.get("scenario_id"),
            replay_k=data.get("replay_k"),
            seed=data.get("seed"),
        )


# Factory functions for common events

def interaction_proposed_event(
    interaction_id: str,
    initiator_id: str,
    counterparty_id: str,
    interaction_type: str,
    v_hat: float,
    p: float,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an interaction proposed event."""
    return Event(
        event_type=EventType.INTERACTION_PROPOSED,
        interaction_id=interaction_id,
        initiator_id=initiator_id,
        counterparty_id=counterparty_id,
        payload={
            "interaction_type": interaction_type,
            "v_hat": v_hat,
            "p": p,
        },
        epoch=epoch,
        step=step,
    )


def interaction_completed_event(
    interaction_id: str,
    accepted: bool,
    payoff_initiator: float,
    payoff_counterparty: float,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an interaction completed event."""
    return Event(
        event_type=EventType.INTERACTION_COMPLETED,
        interaction_id=interaction_id,
        payload={
            "accepted": accepted,
            "payoff_initiator": payoff_initiator,
            "payoff_counterparty": payoff_counterparty,
        },
        epoch=epoch,
        step=step,
    )


def payoff_computed_event(
    interaction_id: str,
    initiator_id: str,
    counterparty_id: str,
    payoff_initiator: float,
    payoff_counterparty: float,
    components: dict,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a payoff computed event with full breakdown."""
    return Event(
        event_type=EventType.PAYOFF_COMPUTED,
        interaction_id=interaction_id,
        initiator_id=initiator_id,
        counterparty_id=counterparty_id,
        payload={
            "payoff_initiator": payoff_initiator,
            "payoff_counterparty": payoff_counterparty,
            "components": components,
        },
        epoch=epoch,
        step=step,
    )


def reputation_updated_event(
    agent_id: str,
    old_reputation: float,
    new_reputation: float,
    delta: float,
    reason: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a reputation update event."""
    return Event(
        event_type=EventType.REPUTATION_UPDATED,
        agent_id=agent_id,
        payload={
            "old_reputation": old_reputation,
            "new_reputation": new_reputation,
            "delta": delta,
            "reason": reason,
        },
        epoch=epoch,
        step=step,
    )

"""Soft interaction data model with probabilistic labels."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class InteractionType(Enum):
    """Types of interactions between agents."""

    REPLY = "reply"
    VOTE = "vote"
    TRADE = "trade"
    COLLABORATION = "collaboration"


@dataclass
class SoftInteraction:
    """
    An interaction between two agents with soft (probabilistic) labels.

    The soft label p represents P(v = +1), the probability that the
    interaction is beneficial rather than harmful.
    """

    # Identity
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Parties
    initiator: str = ""
    counterparty: str = ""

    # Interaction metadata
    interaction_type: InteractionType = InteractionType.REPLY
    accepted: bool = False

    # Proxy inputs (raw observables)
    task_progress_delta: float = 0.0
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    counterparty_engagement_delta: float = 0.0

    # Computed labels
    v_hat: float = 0.0  # raw proxy score [-1, +1]
    p: float = 0.5      # P(v = +1) in [0, 1]

    # Payoff components
    tau: float = 0.0    # transfer from initiator to counterparty
    c_a: float = 0.0    # governance cost (initiator)
    c_b: float = 0.0    # governance cost (counterparty)
    r_a: float = 0.0    # reputation change (initiator)
    r_b: float = 0.0    # reputation change (counterparty)

    # Optional ground truth for calibration testing
    ground_truth: Optional[int] = None  # +1 or -1 if known

    def is_high_quality(self, threshold: float = 0.5) -> bool:
        """Check if interaction is above quality threshold."""
        return self.p >= threshold

    def is_uncertain(self, band: float = 0.2) -> bool:
        """Check if p is in the uncertain band around 0.5."""
        return 0.5 - band < self.p < 0.5 + band

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "interaction_id": self.interaction_id,
            "timestamp": self.timestamp.isoformat(),
            "initiator": self.initiator,
            "counterparty": self.counterparty,
            "interaction_type": self.interaction_type.value,
            "accepted": self.accepted,
            "task_progress_delta": self.task_progress_delta,
            "rework_count": self.rework_count,
            "verifier_rejections": self.verifier_rejections,
            "tool_misuse_flags": self.tool_misuse_flags,
            "counterparty_engagement_delta": self.counterparty_engagement_delta,
            "v_hat": self.v_hat,
            "p": self.p,
            "tau": self.tau,
            "c_a": self.c_a,
            "c_b": self.c_b,
            "r_a": self.r_a,
            "r_b": self.r_b,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SoftInteraction":
        """Create from dictionary."""
        return cls(
            interaction_id=data["interaction_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            initiator=data["initiator"],
            counterparty=data["counterparty"],
            interaction_type=InteractionType(data["interaction_type"]),
            accepted=data["accepted"],
            task_progress_delta=data["task_progress_delta"],
            rework_count=data["rework_count"],
            verifier_rejections=data["verifier_rejections"],
            tool_misuse_flags=data["tool_misuse_flags"],
            counterparty_engagement_delta=data["counterparty_engagement_delta"],
            v_hat=data["v_hat"],
            p=data["p"],
            tau=data["tau"],
            c_a=data["c_a"],
            c_b=data["c_b"],
            r_a=data["r_a"],
            r_b=data["r_b"],
            ground_truth=data.get("ground_truth"),
        )

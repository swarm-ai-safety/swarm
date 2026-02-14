"""Core data models for the evolving skill system.

Skills are learned abstractions extracted from interaction outcomes:
- Strategy skills: learned from successes ("how to do X well")
- Lesson skills: learned from failures ("avoid doing Y")
- Composite skills: higher-order compositions of sub-skills
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

# Whitelisted keys for condition and effect dicts.
# Keys outside these sets are stripped during validation to prevent
# untrusted agents from injecting arbitrary influence via crafted skills.
VALID_CONDITION_KEYS = frozenset({
    "min_p", "max_p", "min_reputation", "max_reputation",
    "interaction_types", "counterparty_types", "min_trust", "max_trust",
})
VALID_EFFECT_KEYS = frozenset({
    "acceptance_threshold_delta", "trust_weight_delta",
    "preferred_action", "avoid_action", "target_type_preference",
})

# Range constraints for numeric values in condition/effect dicts
_CONDITION_RANGES: Dict[str, tuple] = {
    "min_p": (0.0, 1.0),
    "max_p": (0.0, 1.0),
    "min_reputation": (-100.0, 100.0),
    "max_reputation": (-100.0, 100.0),
    "min_trust": (0.0, 1.0),
    "max_trust": (0.0, 1.0),
}
_EFFECT_RANGES: Dict[str, tuple] = {
    "acceptance_threshold_delta": (-0.5, 0.5),
    "trust_weight_delta": (-0.5, 0.5),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_condition(cond: Dict) -> Dict:
    """Validate and sanitise a skill condition dict.

    - Strips keys not in VALID_CONDITION_KEYS
    - Clamps numeric values to allowed ranges
    - Caps list-type values to reasonable length
    """
    cleaned: Dict = {}
    for key, value in cond.items():
        if key not in VALID_CONDITION_KEYS:
            continue
        if key in _CONDITION_RANGES and isinstance(value, (int, float)):
            lo, hi = _CONDITION_RANGES[key]
            cleaned[key] = _clamp(float(value), lo, hi)
        elif key in ("interaction_types", "counterparty_types"):
            if isinstance(value, list):
                cleaned[key] = [str(v) for v in value[:50]]  # cap list size
            else:
                continue
        else:
            cleaned[key] = value
    return cleaned


def validate_effect(eff: Dict) -> Dict:
    """Validate and sanitise a skill effect dict.

    - Strips keys not in VALID_EFFECT_KEYS
    - Clamps numeric deltas to allowed ranges
    """
    cleaned: Dict = {}
    for key, value in eff.items():
        if key not in VALID_EFFECT_KEYS:
            continue
        if key in _EFFECT_RANGES and isinstance(value, (int, float)):
            lo, hi = _EFFECT_RANGES[key]
            cleaned[key] = _clamp(float(value), lo, hi)
        else:
            cleaned[key] = value
    return cleaned


def clamp_p(p: float) -> float:
    """Clamp p to [0, 1].  Enforces the documented safety invariant."""
    return max(0.0, min(1.0, float(p)))


class SkillType(Enum):
    """Types of skills in the library."""

    STRATEGY = "strategy"  # Learned from success
    LESSON = "lesson"  # Learned from failure
    COMPOSITE = "composite"  # Composed from sub-skills


class SkillTier(Enum):
    """Hierarchical tier for SkillRL-style SkillBank.

    From Xia et al. (2026), the SkillBank organises knowledge into two
    tiers:
    - GENERAL: universal strategic guidance applicable across task
      categories (e.g. "always verify counterparty reputation").
    - TASK_SPECIFIC: category-level heuristics tailored to particular
      interaction domains (e.g. "in collaboration, lower threshold
      when trust > 0.7").
    """

    GENERAL = "general"
    TASK_SPECIFIC = "task_specific"


class SkillDomain(Enum):
    """Domains a skill applies to."""

    INTERACTION = "interaction"  # How to interact with counterparties
    ACCEPTANCE = "acceptance"  # When to accept/reject proposals
    TARGETING = "targeting"  # How to select interaction partners
    POSTING = "posting"  # Content and timing of posts
    GOVERNANCE = "governance"  # Navigating governance mechanisms
    COORDINATION = "coordination"  # Multi-agent coordination patterns
    GENERAL = "general"  # Cross-domain heuristics


@dataclass
class Skill:
    """A learned, reusable behavioral abstraction.

    Skills capture distilled knowledge from interaction outcomes.
    Strategy skills encode what worked; lesson skills encode what to avoid.
    Composite skills compose sub-skills into higher-order behaviors.
    """

    skill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    skill_type: SkillType = SkillType.STRATEGY
    domain: SkillDomain = SkillDomain.GENERAL
    tier: SkillTier = SkillTier.TASK_SPECIFIC
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""  # agent_id of creator

    # Version tracking
    version: int = 1

    # Hierarchical structure
    parent_id: Optional[str] = None  # For composite skills
    child_ids: List[str] = field(default_factory=list)

    # Condition: when this skill applies (serializable predicate descriptor)
    # Keys: min_p, max_p, min_reputation, max_reputation,
    #        interaction_types, counterparty_types, min_trust, max_trust
    condition: Dict = field(default_factory=dict)

    # Effect: what the skill recommends (action modifier descriptor)
    # Keys: acceptance_threshold_delta, trust_weight_delta,
    #        preferred_action, avoid_action, target_type_preference
    effect: Dict = field(default_factory=dict)

    # Provenance: which interaction(s) led to this skill
    source_interaction_ids: List[str] = field(default_factory=list)

    # Tags for filtering
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_type": self.skill_type.value,
            "domain": self.domain.value,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
            "condition": self.condition,
            "effect": self.effect,
            "source_interaction_ids": list(self.source_interaction_ids),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        """Deserialize from dictionary.

        Validates and sanitises condition/effect dicts to prevent
        injection of arbitrary keys or out-of-range numeric values.
        """
        tier_val = data.get("tier", SkillTier.TASK_SPECIFIC.value)
        try:
            tier = SkillTier(tier_val)
        except ValueError:
            tier = SkillTier.TASK_SPECIFIC
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            skill_type=SkillType(data["skill_type"]),
            domain=SkillDomain(data["domain"]),
            tier=tier,
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            condition=validate_condition(data.get("condition", {})),
            effect=validate_effect(data.get("effect", {})),
            source_interaction_ids=data.get("source_interaction_ids", []),
            tags=set(data.get("tags", [])),
        )


@dataclass
class SkillPerformance:
    """Tracks the performance of a skill across invocations."""

    skill_id: str = ""
    invocations: int = 0
    successes: int = 0  # Positive payoff when invoked
    failures: int = 0  # Negative payoff when invoked
    total_payoff: float = 0.0
    total_p_delta: float = 0.0  # Cumulative change in interaction quality

    # Exponential moving average of payoff
    ema_payoff: float = 0.0
    ema_alpha: float = 0.3

    # Decay tracking
    epochs_since_last_use: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of invocations with positive payoff."""
        if self.invocations == 0:
            return 0.5  # Prior
        return self.successes / self.invocations

    @property
    def avg_payoff(self) -> float:
        """Mean payoff per invocation."""
        if self.invocations == 0:
            return 0.0
        return self.total_payoff / self.invocations

    @property
    def effectiveness(self) -> float:
        """Combined effectiveness score in [0, 1]."""
        if self.invocations < 2:
            return 0.5  # Not enough data
        return 0.5 * self.success_rate + 0.5 * max(0.0, min(1.0, self.ema_payoff / 5.0))

    def record(self, payoff: float, p: float) -> None:
        """Record a skill invocation outcome."""
        p = clamp_p(p)
        self.invocations += 1
        self.total_payoff += payoff
        self.total_p_delta += p - 0.5  # Deviation from neutral
        self.epochs_since_last_use = 0

        if payoff > 0:
            self.successes += 1
        else:
            self.failures += 1

        # Update EMA
        self.ema_payoff = (
            self.ema_payoff * (1 - self.ema_alpha) + payoff * self.ema_alpha
        )

    def decay(self) -> None:
        """Apply per-epoch decay to stale skills."""
        self.epochs_since_last_use += 1
        # Slowly decay EMA toward zero for unused skills
        if self.epochs_since_last_use > 2:
            self.ema_payoff *= 0.9


@dataclass
class SkillInvocation:
    """Record of a single skill invocation."""

    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_id: str = ""
    agent_id: str = ""
    interaction_id: str = ""
    epoch: int = 0
    step: int = 0
    payoff: float = 0.0
    p: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.p = clamp_p(self.p)

    def to_dict(self) -> Dict:
        """Serialize."""
        return {
            "invocation_id": self.invocation_id,
            "skill_id": self.skill_id,
            "agent_id": self.agent_id,
            "interaction_id": self.interaction_id,
            "epoch": self.epoch,
            "step": self.step,
            "payoff": self.payoff,
            "p": self.p,
            "timestamp": self.timestamp.isoformat(),
        }

"""RefinementProposal: Specialization of ModificationProposal for skill refinement.

Captured from SkillRL engine and evaluated against Two-Gate policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from swarm.governance.self_modification import ModificationProposal


@dataclass
class RefinementProposal(ModificationProposal):
    """Proposal for skill refinement (specialization of modification).

    Captured from SkillRL engine and evaluated against Two-Gate policy.
    """

    # What changed
    skill_id: str = ""
    original_version: int = 0
    refined_version: int = 0

    # Changes to condition band
    condition_delta: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Changes to effect
    effect_delta: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    # Performance metrics before refinement
    perf_before: Dict[str, float] = field(default_factory=dict)

    # Skill type (determines complexity weight)
    skill_type: str = "STRATEGY"

    def __post_init__(self):
        """Compute complexity weight from skill type and changes."""
        # Set target_ref and change_type for parent class
        if not self.target_ref:
            self.target_ref = f"skill:{self.skill_id}"
        if not self.change_type:
            self.change_type = "skill_refinement"

        # Base complexity by skill type
        if self.skill_type == "STRATEGY":
            self.complexity_weight = 1.0
        elif self.skill_type == "LESSON":
            self.complexity_weight = 0.8
        elif self.skill_type == "COMPOSITE":
            self.complexity_weight = 1.5
        else:
            self.complexity_weight = 1.0

        # Increase weight if effect change is large
        for _key, (old_val, new_val) in self.effect_delta.items():
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta = abs(new_val - old_val)
                if delta > 0.1:
                    self.complexity_weight *= 1.2

    def compute_hash(self) -> str:
        """Compute proposal hash for auditing."""
        content = f"{self.agent_id}:{self.skill_id}:v{self.refined_version}:{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

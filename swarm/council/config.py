"""Configuration for the multi-LLM council protocol."""

from dataclasses import dataclass, field
from typing import List, Optional

from swarm.agents.llm_config import LLMConfig


@dataclass
class CouncilMemberConfig:
    """Configuration for a single council member."""

    member_id: str
    llm_config: LLMConfig
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


@dataclass
class CouncilConfig:
    """Configuration for the council deliberation protocol."""

    members: List[CouncilMemberConfig] = field(default_factory=list)
    chairman: Optional[CouncilMemberConfig] = None
    min_members_required: int = 2
    timeout_per_member: float = 30.0
    anonymize_responses: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_members_required < 1:
            raise ValueError(
                f"min_members_required must be >= 1, got {self.min_members_required}"
            )
        if self.timeout_per_member <= 0:
            raise ValueError(
                f"timeout_per_member must be positive, got {self.timeout_per_member}"
            )
        # If no chairman specified, the first member acts as chairman
        if self.chairman is None and self.members:
            self.chairman = self.members[0]

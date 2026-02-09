"""Episode specification for deterministic replay experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from swarm.scenarios.loader import ScenarioConfig


@dataclass
class EpisodeSpec:
    """
    Defines a single scenario specification to replay K times.

    Seed progression is deterministic: `seed + replay_index`.
    """

    scenario: ScenarioConfig
    seed: int
    replay_k: int = 1
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replay_k < 1:
            raise ValueError("replay_k must be >= 1")

    def replay_seeds(self) -> List[int]:
        """Return the deterministic seed schedule for K replays."""
        return [self.seed + idx for idx in range(self.replay_k)]

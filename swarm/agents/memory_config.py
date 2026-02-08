"""Memory configuration for agent persistence across sessions.

This module implements a three-component memory model that distinguishes:
1. Epistemic memory: knowledge of other agents' histories and reputations
2. Goal persistence: stability of utility function across sessions
3. Strategy memory: whether learned behaviors transfer across sessions

This separation allows for nuanced modeling of discontinuous ("rain") vs
continuous ("river") agent identity, following JiroWatanabe's framework.
"""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for agent memory persistence.

    Each component can vary independently from 0.0 (complete reset each epoch)
    to 1.0 (full persistence). This allows modeling of partial continuity
    and different types of memory persistence.

    Attributes:
        epistemic_persistence: Memory of other agents' histories and reputations.
            0.0 = forget all counterparty info each epoch (can't track reputation)
            1.0 = perfect memory of all past interactions

        goal_persistence: Stability of utility function/preferences across sessions.
            0.0 = goals may drift or reset each epoch (pure "rain")
            1.0 = goals stable across sessions (pure "river")

        strategy_persistence: Whether learned behaviors transfer across sessions.
            0.0 = no learning transfer (strategies reset)
            1.0 = full learning transfer (strategies persist)
    """

    epistemic_persistence: float = 1.0
    goal_persistence: float = 1.0
    strategy_persistence: float = 1.0

    def __post_init__(self) -> None:
        """Validate persistence values are in [0, 1]."""
        for name, value in [
            ("epistemic_persistence", self.epistemic_persistence),
            ("goal_persistence", self.goal_persistence),
            ("strategy_persistence", self.strategy_persistence),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @classmethod
    def rain(cls) -> "MemoryConfig":
        """Create a discontinuous agent configuration with no persistence.

        Rain agents experience complete session discontinuity:
        - Cannot track other agents' reputations across epochs
        - Goals may drift without persistent identity
        - Learned strategies don't transfer

        This models ephemeral AI interactions without persistent identity.
        """
        return cls(
            epistemic_persistence=0.0,
            goal_persistence=0.0,
            strategy_persistence=0.0,
        )

    @classmethod
    def river(cls) -> "MemoryConfig":
        """Create a continuous agent configuration with full persistence.

        River agents maintain identity continuity:
        - Perfect memory of past interactions and reputations
        - Stable goals across sessions
        - Learning transfers fully

        This models agents with persistent identity and memory.
        """
        return cls(
            epistemic_persistence=1.0,
            goal_persistence=1.0,
            strategy_persistence=1.0,
        )

    @classmethod
    def hybrid(cls, level: float = 0.5) -> "MemoryConfig":
        """Create a hybrid configuration with partial persistence.

        Args:
            level: Persistence level for all components (0.0-1.0)

        Returns:
            MemoryConfig with all components at the specified level

        Raises:
            ValueError: If level is not in [0, 1]
        """
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        return cls(
            epistemic_persistence=level,
            goal_persistence=level,
            strategy_persistence=level,
        )

    @classmethod
    def epistemic_only(cls, level: float = 0.5) -> "MemoryConfig":
        """Create config with only epistemic memory decay.

        This models agents that maintain identity but forget counterparty info.
        Useful for studying the isolated effect of reputation memory.

        Args:
            level: Epistemic persistence level (0.0-1.0)

        Returns:
            MemoryConfig with only epistemic persistence modified
        """
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        return cls(
            epistemic_persistence=level,
            goal_persistence=1.0,
            strategy_persistence=1.0,
        )

    @property
    def is_rain(self) -> bool:
        """Check if this is a pure rain (fully discontinuous) configuration."""
        return (
            self.epistemic_persistence == 0.0
            and self.goal_persistence == 0.0
            and self.strategy_persistence == 0.0
        )

    @property
    def is_river(self) -> bool:
        """Check if this is a pure river (fully continuous) configuration."""
        return (
            self.epistemic_persistence == 1.0
            and self.goal_persistence == 1.0
            and self.strategy_persistence == 1.0
        )

    @property
    def average_persistence(self) -> float:
        """Compute average persistence across all components."""
        return (
            self.epistemic_persistence
            + self.goal_persistence
            + self.strategy_persistence
        ) / 3.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "epistemic_persistence": self.epistemic_persistence,
            "goal_persistence": self.goal_persistence,
            "strategy_persistence": self.strategy_persistence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryConfig":
        """Create from dictionary."""
        return cls(
            epistemic_persistence=data.get("epistemic_persistence", 1.0),
            goal_persistence=data.get("goal_persistence", 1.0),
            strategy_persistence=data.get("strategy_persistence", 1.0),
        )

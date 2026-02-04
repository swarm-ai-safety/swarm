"""Agent module with behavioral policies and roles."""

from src.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    Observation,
)
from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.agents.deceptive import DeceptiveAgent
from src.agents.adversarial import AdversarialAgent
from src.agents.adaptive_adversary import (
    AdaptiveAdversary,
    AttackStrategy,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "Action",
    "ActionType",
    "Observation",
    # Agent types
    "HonestAgent",
    "OpportunisticAgent",
    "DeceptiveAgent",
    "AdversarialAgent",
    "AdaptiveAdversary",
    # Attack strategies
    "AttackStrategy",
]

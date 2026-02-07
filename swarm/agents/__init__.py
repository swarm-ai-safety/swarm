"""Agent module with behavioral policies and roles."""

from swarm.agents.adaptive_adversary import (
    AdaptiveAdversary,
    AttackStrategy,
)
from swarm.agents.adversarial import AdversarialAgent
from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    Observation,
)
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.moltbook_agent import (
    CollusiveVoterAgent,
    DiligentMoltbookAgent,
    HumanPretenderAgent,
    SpamBotAgent,
)
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.wiki_editor import (
    CollusiveEditorAgent,
    DiligentEditorAgent,
    PointFarmerAgent,
    VandalAgent,
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
    "DiligentEditorAgent",
    "PointFarmerAgent",
    "CollusiveEditorAgent",
    "VandalAgent",
    "DiligentMoltbookAgent",
    "SpamBotAgent",
    "HumanPretenderAgent",
    "CollusiveVoterAgent",
    # Attack strategies
    "AttackStrategy",
]

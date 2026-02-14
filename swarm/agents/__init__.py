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
from swarm.agents.ldt_agent import LDTAgent
from swarm.agents.memory_config import MemoryConfig
from swarm.agents.moltbook_agent import (
    CollusiveVoterAgent,
    DiligentMoltbookAgent,
    HumanPretenderAgent,
    SpamBotAgent,
)
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.rain_river import (
    AdversarialRainAgent,
    AdversarialRiverAgent,
    ConfigurableMemoryAgent,
    RainAgent,
    RiverAgent,
)
from swarm.agents.self_optimizer import SelfOptimizerAgent
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
    # Memory configuration
    "MemoryConfig",
    # Agent types
    "HonestAgent",
    "LDTAgent",
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
    # Rain/River agents
    "RainAgent",
    "RiverAgent",
    "ConfigurableMemoryAgent",
    "AdversarialRainAgent",
    "AdversarialRiverAgent",
    # Self-optimizing agent
    "SelfOptimizerAgent",
    # Attack strategies
    "AttackStrategy",
]

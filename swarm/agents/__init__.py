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
from swarm.agents.behavioral import (
    AdaptiveAgent,
    CautiousAgent,
    CollaborativeAgent,
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
from swarm.agents.obfuscating import ObfuscatingAgent, ObfuscationStrategy
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.rain_river import (
    AdversarialRainAgent,
    AdversarialRiverAgent,
    ConfigurableMemoryAgent,
    RainAgent,
    RiverAgent,
)
from swarm.agents.ralph_agent import AdversarialRalphAgent, RalphLoopAgent
from swarm.agents.self_optimizer import SelfOptimizerAgent
from swarm.agents.threshold_dancer import ThresholdDancer
from swarm.agents.wiki_editor import (
    CollusiveEditorAgent,
    DiligentEditorAgent,
    PointFarmerAgent,
    VandalAgent,
)
from swarm.agents.work_regime_agent import WorkRegimeAgent

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
    # Ralph loop agents
    "RalphLoopAgent",
    "AdversarialRalphAgent",
    # Self-optimizing agent
    "SelfOptimizerAgent",
    # Obfuscation Atlas agents
    "ObfuscatingAgent",
    "ObfuscationStrategy",
    # Attack strategies
    "AttackStrategy",
    # Threshold dancer adversary
    "ThresholdDancer",
    # Behavioral archetypes (issue #66)
    "CautiousAgent",
    "CollaborativeAgent",
    "AdaptiveAgent",
    # Work regime drift agent
    "WorkRegimeAgent",
]

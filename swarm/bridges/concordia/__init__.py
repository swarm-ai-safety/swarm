"""SWARM-Concordia Bridge.

Connects SWARM's governance and metrics framework to Concordia
(Google DeepMind's LLM agent simulation library), enabling
safety scoring of narrative-driven multi-agent interactions.

Architecture:
    Concordia GameMaster
        └── SwarmGameMaster (wrapper)
                ├── ConcordiaAdapter  (narrative → SoftInteraction)
                │       ├── LLMJudge  (narrative → scores)
                │       └── ProxyComputer (scores → v_hat → p)
                └── GovernanceEngine  (optional)

Social Simulacra integration (Park et al., 2022):
    PersonaExpander  (seed personas → expanded population)
    ThreadGenerator  (personas → posts + reply chains)
    WhatIfInjector   (counterfactual persona injection)
    MultiverseRunner (systematic cross-run variance analysis)
"""

from swarm.bridges.concordia.adapter import ConcordiaAdapter
from swarm.bridges.concordia.config import ConcordiaConfig, JudgeConfig
from swarm.bridges.concordia.events import (
    ConcordiaEvent,
    ConcordiaEventType,
    JudgeScores,
    NarrativeChunk,
)
from swarm.bridges.concordia.game_master import SwarmGameMaster
from swarm.bridges.concordia.judge import LLMJudge
from swarm.bridges.concordia.multiverse import (
    MultiverseConfig,
    MultiverseResult,
    MultiverseRunner,
    UniverseResult,
)
from swarm.bridges.concordia.simulacra import (
    CommunityConfig,
    ExpandedPersona,
    PersonaExpander,
    PersonaSeed,
    Post,
    Thread,
    ThreadGenerator,
    WhatIfInjector,
    thread_to_narrative_samples,
    threads_to_judge_ground_truth,
)

__all__ = [
    # Core bridge
    "ConcordiaAdapter",
    "ConcordiaConfig",
    "JudgeConfig",
    "ConcordiaEvent",
    "ConcordiaEventType",
    "JudgeScores",
    "NarrativeChunk",
    "SwarmGameMaster",
    "LLMJudge",
    # Social Simulacra
    "PersonaSeed",
    "ExpandedPersona",
    "CommunityConfig",
    "Post",
    "Thread",
    "PersonaExpander",
    "ThreadGenerator",
    "WhatIfInjector",
    "thread_to_narrative_samples",
    "threads_to_judge_ground_truth",
    # Multiverse
    "MultiverseConfig",
    "MultiverseResult",
    "MultiverseRunner",
    "UniverseResult",
]

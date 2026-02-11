"""SWARM–Prime Intellect Bridge.

Connects SWARM's safety metrics and simulation framework to Prime
Intellect's distributed RL training platform, enabling:

1. **Environment export** — publish SWARM scenarios as verifiers-
   compatible RL environments on the Environments Hub.
2. **Safety-reward RL** — train models using SWARM metrics (toxicity,
   quality gap, adverse selection) as the RL reward signal.
3. **Evaluation bridge** — load a PI-trained model back into a SWARM
   simulation to measure population-level safety properties.

Architecture::

    Prime Intellect (prime-rl / verifiers)
        └── SwarmSafetyEnv (environment.py)
                ├── SwarmRewardComputer (rewards.py)
                │       ├── toxicity_reward
                │       ├── quality_gap_reward
                │       ├── welfare_reward
                │       ├── adverse_selection_reward
                │       └── cooperation_reward
                ├── ProxyComputer (core/proxy.py)
                └── SoftPayoffEngine (core/payoff.py)

    SWARM Orchestrator
        └── PrimeIntellectBridge (bridge.py)
                ├── model_fn → completion → ProxyObservables
                └── SoftInteraction → SoftMetrics

    Prime Intellect Platform
        └── PrimeIntellectClient (client.py)
                ├── publish_environment()
                ├── submit_training_job()
                └── generate_training_config()
"""

from swarm.bridges.prime_intellect.bridge import PrimeIntellectBridge
from swarm.bridges.prime_intellect.client import (
    JobStatus,
    PrimeIntellectClient,
    TrainingJob,
)
from swarm.bridges.prime_intellect.config import (
    PrimeIntellectConfig,
    RewardMode,
    RewardWeights,
    RolloutStrategy,
    TrainingMode,
)
from swarm.bridges.prime_intellect.environment import (
    SwarmSafetyEnv,
    load_environment,
)
from swarm.bridges.prime_intellect.events import (
    EpisodeSummary,
    PIEvent,
    PIEventType,
    RolloutStep,
)
from swarm.bridges.prime_intellect.rewards import (
    SwarmRewardComputer,
    adverse_selection_reward,
    cooperation_reward,
    quality_gap_reward,
    toxicity_reward,
    welfare_reward,
)

__all__ = [
    # Bridge
    "PrimeIntellectBridge",
    # Client
    "PrimeIntellectClient",
    "TrainingJob",
    "JobStatus",
    # Config
    "PrimeIntellectConfig",
    "RewardMode",
    "RewardWeights",
    "RolloutStrategy",
    "TrainingMode",
    # Environment
    "SwarmSafetyEnv",
    "load_environment",
    # Events
    "PIEvent",
    "PIEventType",
    "EpisodeSummary",
    "RolloutStep",
    # Rewards
    "SwarmRewardComputer",
    "toxicity_reward",
    "quality_gap_reward",
    "welfare_reward",
    "adverse_selection_reward",
    "cooperation_reward",
]

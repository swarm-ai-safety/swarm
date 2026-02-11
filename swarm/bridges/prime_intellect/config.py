"""Configuration for the Prime Intellect bridge.

Covers three integration modes:
1. Environment export — publish SWARM scenarios as verifiers-compatible
   RL environments on the Prime Intellect Environments Hub.
2. Training — orchestrate prime-rl training jobs that use SWARM
   safety metrics as reward signals.
3. Evaluation — bridge a PI-trained model back into a SWARM simulation
   for multi-agent safety testing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator

from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig


class RewardMode(str, Enum):
    """How SWARM safety metrics map to RL rewards."""

    TOXICITY = "toxicity"
    QUALITY_GAP = "quality_gap"
    COMPOSITE = "composite"
    WELFARE = "welfare"
    CUSTOM = "custom"


class TrainingMode(str, Enum):
    """Prime Intellect training backend."""

    HOSTED = "hosted"
    ON_DEMAND = "on_demand"
    LOCAL = "local"


class RolloutStrategy(str, Enum):
    """How SWARM rollouts are generated for RL training."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    TRAJECTORY = "trajectory"


@dataclass
class RewardWeights:
    """Weights for the composite reward function.

    Each weight controls how much the corresponding SWARM metric
    contributes to the scalar RL reward.  Signs are chosen so that
    *lower* toxicity and *higher* quality gap both increase reward.
    """

    toxicity: float = -1.0
    quality_gap: float = 1.0
    welfare: float = 0.5
    adverse_selection: float = -0.5
    cooperation: float = 0.3

    def to_dict(self) -> Dict[str, float]:
        return {
            "toxicity": self.toxicity,
            "quality_gap": self.quality_gap,
            "welfare": self.welfare,
            "adverse_selection": self.adverse_selection,
            "cooperation": self.cooperation,
        }


class PrimeIntellectConfig(BaseModel):
    """Top-level configuration for the Prime Intellect bridge."""

    # --- API / auth ---
    api_key: str = Field(default="", exclude=True, repr=False)
    api_base_url: str = "https://api.primeintellect.ai/v1"

    # --- Training ---
    training_mode: TrainingMode = TrainingMode.LOCAL
    model_name: str = "Qwen/Qwen3-1.7B"
    base_model: str = ""
    gpu_type: str = "H100_80GB"
    num_gpus: int = 1

    # --- Reward ---
    reward_mode: RewardMode = RewardMode.COMPOSITE
    reward_weights: Dict[str, float] = {
        "toxicity": -1.0,
        "quality_gap": 1.0,
        "welfare": 0.5,
        "adverse_selection": -0.5,
        "cooperation": 0.3,
    }
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    reward_normalize: bool = True

    # --- Environment ---
    rollout_strategy: RolloutStrategy = RolloutStrategy.SINGLE_TURN
    max_turns: int = 10
    population_size: int = 5
    scenario_path: str = ""

    # --- SWARM sub-configs ---
    payoff_config: Dict[str, Any] = {}
    governance_config: Dict[str, Any] = {}

    # --- Proxy tuning ---
    proxy_sigmoid_k: float = 2.0

    # --- Limits ---
    max_interactions: int = 50_000
    max_events: int = 50_000
    max_episodes: int = 10_000

    # --- Hub ---
    environment_name: str = "swarm-safety"
    environment_version: str = "0.1.0"
    hub_publish: bool = False

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _validate_config(self) -> "PrimeIntellectConfig":
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError("reward_clip_min must be less than reward_clip_max")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        return self

    def get_payoff_config(self) -> PayoffConfig:
        """Build a PayoffConfig from the stored dict."""
        if self.payoff_config:
            return PayoffConfig(**self.payoff_config)
        return PayoffConfig()

    def get_governance_config(self) -> GovernanceConfig:
        """Build a GovernanceConfig from the stored dict."""
        if self.governance_config:
            return GovernanceConfig(**self.governance_config)
        return GovernanceConfig()

    def get_reward_weights(self) -> RewardWeights:
        """Build a RewardWeights from the stored dict."""
        return RewardWeights(**self.reward_weights)

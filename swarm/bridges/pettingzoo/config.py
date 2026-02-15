"""Configuration for the PettingZoo bridge."""

from typing import Any, Dict

from pydantic import BaseModel, model_validator

from swarm.core.payoff import PayoffConfig


class PettingZooConfig(BaseModel):
    """Top-level configuration for the PettingZoo environment bridge."""

    # --- Population ---
    n_agents: int = 5
    agent_types: Dict[str, float] = {
        "honest": 0.6,
        "opportunistic": 0.2,
        "deceptive": 0.2,
    }

    # --- Episode ---
    max_steps: int = 50
    scenario_path: str = ""

    # --- Proxy ---
    proxy_sigmoid_k: float = 2.0

    # --- Payoff passthrough ---
    payoff_config: Dict[str, Any] = {}

    # --- Observation space tuning ---
    obs_history_len: int = 5  # steps of interaction history in obs

    # --- Rendering ---
    render_mode: str = ""  # "", "human", "ansi", "rgb_array"

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _validate(self) -> "PettingZooConfig":
        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        total = sum(self.agent_types.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"agent_types fractions must sum to 1, got {total}")
        return self

    def get_payoff_config(self) -> PayoffConfig:
        if self.payoff_config:
            return PayoffConfig(**self.payoff_config)
        return PayoffConfig()

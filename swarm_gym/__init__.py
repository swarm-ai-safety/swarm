"""SWARM-Gym: OpenAI Gym for multi-agent governance research.

Standardized environments, benchmarks, and evaluation for studying
governance mechanisms in multi-agent AI systems.
"""

__version__ = "0.1.0"

from swarm_gym.envs.base import SwarmEnv, StepResult
from swarm_gym.envs.registry import make, register_env, list_envs
from swarm_gym.governance.base import GovernanceModule
from swarm_gym.agents.base import AgentPolicy

__all__ = [
    "SwarmEnv",
    "StepResult",
    "make",
    "register_env",
    "list_envs",
    "GovernanceModule",
    "AgentPolicy",
]

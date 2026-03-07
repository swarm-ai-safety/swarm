"""Environment registry: register_env(), make(), list_envs().

Modeled after OpenAI Gym's registry pattern.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from swarm_gym.envs.base import SwarmEnv


_REGISTRY: Dict[str, Type[SwarmEnv]] = {}


def register_env(env_id: str, cls: Type[SwarmEnv]) -> None:
    """Register an environment class under an ID.

    Args:
        env_id: Environment ID (e.g. "swarm/escalation_ladder:v1").
        cls: SwarmEnv subclass.
    """
    if env_id in _REGISTRY:
        raise ValueError(f"Environment '{env_id}' already registered")
    _REGISTRY[env_id] = cls


def make(env_id: str, **kwargs: Any) -> SwarmEnv:
    """Create an environment by ID.

    Args:
        env_id: Registered environment ID.
        **kwargs: Passed to the environment constructor.

    Returns:
        Instantiated SwarmEnv.

    Raises:
        KeyError: If env_id is not registered.
    """
    if env_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown environment '{env_id}'. Available: {available}"
        )
    return _REGISTRY[env_id](**kwargs)


def list_envs() -> List[str]:
    """Return sorted list of registered environment IDs."""
    return sorted(_REGISTRY.keys())


def _clear_registry() -> None:
    """Clear registry (for testing only)."""
    _REGISTRY.clear()

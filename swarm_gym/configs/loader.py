"""Config loader: parse benchmark + governance YAMLs and build environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from swarm_gym.envs.base import SwarmEnv
from swarm_gym.envs.registry import make
from swarm_gym.governance.base import GovernanceModule
from swarm_gym.governance.tax import TaxPolicy
from swarm_gym.governance.audits import AuditPolicy
from swarm_gym.governance.circuit_breaker import CircuitBreakerPolicy


_CONFIGS_DIR = Path(__file__).parent

GOVERNANCE_MODULES = {
    "TaxPolicy": TaxPolicy,
    "AuditPolicy": AuditPolicy,
    "CircuitBreakerPolicy": CircuitBreakerPolicy,
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_governance_preset(preset: str) -> List[GovernanceModule]:
    """Load governance modules from a preset YAML."""
    path = _CONFIGS_DIR / "governance" / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Governance preset not found: {path}")

    config = load_yaml(path)
    modules: List[GovernanceModule] = []

    for mod_cfg in config.get("modules", []):
        name = mod_cfg["name"]
        params = mod_cfg.get("params", {})
        cls = GOVERNANCE_MODULES.get(name)
        if cls is None:
            raise ValueError(f"Unknown governance module: {name}")
        modules.append(cls(**params))

    return modules


def load_benchmark(path: Path) -> Tuple[Dict[str, Any], SwarmEnv]:
    """Load a benchmark YAML and build the environment.

    Returns:
        Tuple of (full config dict, configured SwarmEnv instance).
    """
    config = load_yaml(path)

    env_id = config["env_id"]
    env_kwargs: Dict[str, Any] = {}

    # Map common fields
    if "episode_len" in config:
        env_kwargs["episode_len"] = config["episode_len"]
    if "num_agents" in config:
        env_kwargs["num_agents"] = config["num_agents"]

    # Map game-specific fields
    game = config.get("game", {})
    env_kwargs.update(game)

    # Map observability
    obs_cfg = config.get("observability", {})
    env_kwargs.update(obs_cfg)

    # Ensure benchmark envs are registered
    _ensure_envs_registered()

    env = make(env_id, **env_kwargs)

    # Load governance
    gov_cfg = config.get("governance", {})
    preset = gov_cfg.get("preset", "")
    if preset:
        modules = load_governance_preset(preset)
        env.set_governance(modules)

    return config, env


def _ensure_envs_registered() -> None:
    """Import environment modules to trigger registration."""
    import swarm_gym.envs.escalation_ladder  # noqa: F401
    import swarm_gym.envs.collusion_market  # noqa: F401
    import swarm_gym.envs.audit_evasion  # noqa: F401

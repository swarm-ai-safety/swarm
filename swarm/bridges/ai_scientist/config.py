"""Configuration for the AI-Scientist bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

DEFAULT_ROLE_MAP: Dict[str, str] = {
    "ideation": "ideation_agent",
    "literature_checker": "literature_checker",
    "experiment": "experiment_agent",
    "writeup": "writeup_agent",
    "reviewer": "reviewer_agent",
    "improver": "improvement_agent",
}


@dataclass
class AIScientistClientConfig:
    """Configuration for parsing AI-Scientist output directories."""

    results_dir: str = "results"
    experiment_template: str = ""  # e.g. "nanoGPT_lite"


@dataclass
class AIScientistConfig:
    """Full bridge configuration."""

    client_config: AIScientistClientConfig = field(
        default_factory=AIScientistClientConfig,
    )
    orchestrator_id: str = "ai_scientist_orchestrator"
    proxy_sigmoid_k: float = 2.0

    agent_role_map: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROLE_MAP),
    )

    # Governance thresholds
    novelty_gate_enabled: bool = True
    phase_gate_min_p: float = 0.4
    experiment_circuit_breaker_max_failures: int = 4
    cost_budget_usd: float = 100.0
    review_accept_threshold: float = 5.0
    max_improvement_rounds: int = 2

    # Memory caps
    max_interactions: int = 50000
    max_events: int = 50000

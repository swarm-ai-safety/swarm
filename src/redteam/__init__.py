"""Red-teaming framework for governance robustness testing.

This module provides tools for systematically testing governance mechanisms
against adaptive adversaries.
"""

from src.redteam.attacks import (
    AttackScenario,
    AttackResult,
    AttackLibrary,
)
from src.redteam.evaluator import (
    RedTeamEvaluator,
    GovernanceRobustness,
    VulnerabilityReport,
)
from src.redteam.metrics import (
    EvasionMetrics,
    compute_evasion_rate,
    compute_detection_latency,
    compute_damage_before_detection,
)

__all__ = [
    # Attacks
    "AttackScenario",
    "AttackResult",
    "AttackLibrary",
    # Evaluator
    "RedTeamEvaluator",
    "GovernanceRobustness",
    "VulnerabilityReport",
    # Metrics
    "EvasionMetrics",
    "compute_evasion_rate",
    "compute_detection_latency",
    "compute_damage_before_detection",
]

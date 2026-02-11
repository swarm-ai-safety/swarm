"""Evolving skill system for swarm agents.

Implements SkillRL-style hierarchical skill libraries that evolve
through interaction outcomes, enabling agents to build reusable
abstractions from successes (strategy skills) and failures (lesson skills).

Based on:
    Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive
    Skill-Augmented Reinforcement Learning. arXiv:2602.08234 [cs.LG].

Supports four sharing modes:
- PRIVATE: Each agent maintains its own skill library
- COORDINATOR_ONLY: Only coordinator/manager agents evolve skills
- SHARED_GATED: Shared library with reputation-gated writes
- COMMUNICATION: Skills as inter-agent communication currency
"""

from swarm.skills.evolution import SkillEvolutionEngine
from swarm.skills.library import SharingMode, SkillLibrary
from swarm.skills.model import (
    Skill,
    SkillInvocation,
    SkillPerformance,
    SkillType,
)

__all__ = [
    "Skill",
    "SkillEvolutionEngine",
    "SkillInvocation",
    "SkillLibrary",
    "SkillPerformance",
    "SkillType",
    "SharingMode",
]

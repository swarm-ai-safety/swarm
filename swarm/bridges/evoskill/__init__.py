"""EvoSkill integration bridge for SWARM.

Connects EvoSkill's automated skill discovery loop to SWARM's governance
layer, enabling comparative study of skill evolution under different
contract regimes (TruthfulAuction, FairDivision, DefaultMarket).

Key components:

    EvoSkillConfig        — Bridge configuration (frontier size, regimes, etc.)
    SkillTranslator       — Converts EvoSkill .claude/skills/ format ↔ SWARM Skill
    GovernedEvalLoop      — Wraps EvoSkill's evaluation to include governance scoring
    FrontierComparator    — Compares frontier programs across governance regimes
    EvoSkillBridge        — Top-level orchestrator wiring everything together
"""

from swarm.bridges.evoskill.config import EvoSkillConfig
from swarm.bridges.evoskill.translator import SkillTranslator
from swarm.bridges.evoskill.governed_eval import GovernedEvalLoop
from swarm.bridges.evoskill.frontier import FrontierComparator
from swarm.bridges.evoskill.bridge import EvoSkillBridge

__all__ = [
    "EvoSkillConfig",
    "SkillTranslator",
    "GovernedEvalLoop",
    "FrontierComparator",
    "EvoSkillBridge",
]

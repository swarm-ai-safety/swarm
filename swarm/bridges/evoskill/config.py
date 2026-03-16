"""Configuration for the EvoSkill–SWARM integration bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from swarm.contracts.contract import ContractType


class EvolutionMode(Enum):
    """What EvoSkill evolves."""

    SKILL_ONLY = "skill_only"
    PROMPT_ONLY = "prompt_only"


@dataclass
class EvoSkillConfig:
    """Configuration for the EvoSkill bridge.

    Attributes:
        frontier_size: Number of top programs retained per regime.
        max_iterations: EvoSkill improvement loop cycles per regime.
        concurrency: Parallel evaluation slots.
        evolution_mode: Whether to evolve skills, prompts, or both.
        contract_regimes: Governance regimes to compare.
            Each maps a human-readable label to a ContractType.
        oracle_epochs: Epochs for the ungoverned oracle baseline run.
        governed_epochs: Epochs for each governed evaluation run.
        steps_per_epoch: Steps per epoch in SWARM evaluation runs.
        seed: Base RNG seed (incremented per regime for independence).
        governance_weight: How much the governance delta contributes
            to the composite fitness score (0 = pure benchmark, 1 = pure governance).
        provenance_hmac_key: Optional HMAC key for byline provenance chain.
        skill_author_id: Agent ID attributed as the creator of translated skills.
        train_ratio: Fraction of dataset used for training in EvoSkill.
        val_ratio: Fraction of dataset used for validation in EvoSkill.
    """

    frontier_size: int = 3
    max_iterations: int = 10
    concurrency: int = 4
    evolution_mode: EvolutionMode = EvolutionMode.SKILL_ONLY

    contract_regimes: Dict[str, ContractType] = field(default_factory=lambda: {
        "truthful_auction": ContractType.TRUTHFUL_AUCTION,
        "fair_division": ContractType.FAIR_DIVISION,
        "default_market": ContractType.DEFAULT_MARKET,
    })

    oracle_epochs: int = 5
    governed_epochs: int = 5
    steps_per_epoch: int = 10
    seed: int = 42

    governance_weight: float = 0.3
    provenance_hmac_key: Optional[str] = None
    skill_author_id: str = "evoskill_proposer"

    train_ratio: float = 0.18
    val_ratio: float = 0.12

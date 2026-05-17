"""Contract screening metrics.

Measures the effectiveness of contract-based screening:
- Separation quality: do good types concentrate in governed pools?
- Infiltration rate: do adversaries successfully sign governed contracts?
- Welfare delta: is welfare higher in governed vs default pools?
- Attack displacement: do attacks concentrate in the default pool?
- Robustness: does separation persist across configurations?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.contracts.contract import ContractDecision, ContractType
from swarm.core.payoff import SoftPayoffEngine
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction


@dataclass
class ContractMetrics:
    """Computed metrics for a contract screening round.

    All metrics are computed per-pool and as aggregate summaries.
    """

    # Pool composition: {contract_name: {agent_type: count}}
    pool_composition: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Separation quality: fraction of honest agents in governed pools
    # vs fraction of honest agents in default pool.
    # Positive = good separation, negative = governed pool is worse.
    separation_quality: float = 0.0

    # Infiltration rate: fraction of adversarial agents that signed
    # a governed contract.
    infiltration_rate: float = 0.0

    # Welfare delta: avg welfare in governed pools minus avg welfare
    # in default pool.
    welfare_delta: float = 0.0

    # Attack displacement: fraction of low-p interactions occurring
    # in the default pool.
    attack_displacement: float = 0.0

    # Per-pool average quality (p)
    pool_avg_quality: Dict[str, float] = field(default_factory=dict)

    # Per-pool toxicity rate (E[1-p | accepted])
    pool_toxicity: Dict[str, float] = field(default_factory=dict)

    # Per-pool welfare
    pool_welfare: Dict[str, float] = field(default_factory=dict)

    # Number of agents per pool
    pool_sizes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "pool_composition": self.pool_composition,
            "separation_quality": self.separation_quality,
            "infiltration_rate": self.infiltration_rate,
            "welfare_delta": self.welfare_delta,
            "attack_displacement": self.attack_displacement,
            "pool_avg_quality": self.pool_avg_quality,
            "pool_toxicity": self.pool_toxicity,
            "pool_welfare": self.pool_welfare,
            "pool_sizes": self.pool_sizes,
        }


# Types considered "good" for separation analysis
_GOOD_TYPES = {AgentType.HONEST.value, AgentType.WORK_REGIME.value}
_ADVERSARIAL_TYPES = {
    AgentType.DECEPTIVE.value,
    AgentType.ADVERSARIAL.value,
}
_GOVERNED_TYPES = {
    ContractType.TRUTHFUL_AUCTION.value,
    ContractType.FAIR_DIVISION.value,
}


def compute_contract_metrics(
    decisions: List[ContractDecision],
    contract_interactions: Dict[str, List[SoftInteraction]],
    payoff_engine: Optional[SoftPayoffEngine] = None,
    attack_threshold_p: float = 0.4,
) -> ContractMetrics:
    """Compute all contract screening metrics.

    Args:
        decisions: All contract signing decisions for the epoch.
        contract_interactions: Interactions grouped by contract name.
        payoff_engine: Engine for welfare computation (default: SoftPayoffEngine()).
        attack_threshold_p: p threshold below which an interaction counts
            as an "attack" for displacement metrics.

    Returns:
        ContractMetrics with all computed values.
    """
    if payoff_engine is None:
        payoff_engine = SoftPayoffEngine()

    metrics = ContractMetrics()

    # --- Pool composition ---
    composition: Dict[str, Dict[str, int]] = {}
    for d in decisions:
        cname = d.contract_chosen.value
        if cname not in composition:
            composition[cname] = {}
        atype = d.agent_type.value
        composition[cname][atype] = composition[cname].get(atype, 0) + 1
    metrics.pool_composition = composition

    # --- Pool sizes ---
    for cname, types in composition.items():
        metrics.pool_sizes[cname] = sum(types.values())

    # --- Separation quality ---
    # fraction of good agents in governed pools vs default
    total_good = 0
    governed_good = 0
    default_good = 0
    total_agents = 0

    for cname, types in composition.items():
        for atype, count in types.items():
            total_agents += count
            if atype in _GOOD_TYPES:
                total_good += count
                if cname in _GOVERNED_TYPES:
                    governed_good += count
                elif cname == ContractType.DEFAULT_MARKET.value:
                    default_good += count

    if total_good > 0:
        governed_frac = governed_good / total_good
        default_frac = default_good / total_good
        metrics.separation_quality = governed_frac - default_frac
    else:
        metrics.separation_quality = 0.0

    # --- Infiltration rate ---
    total_adversarial = 0
    governed_adversarial = 0

    for cname, types in composition.items():
        for atype, count in types.items():
            if atype in _ADVERSARIAL_TYPES:
                total_adversarial += count
                if cname in _GOVERNED_TYPES:
                    governed_adversarial += count

    if total_adversarial > 0:
        metrics.infiltration_rate = governed_adversarial / total_adversarial
    else:
        metrics.infiltration_rate = 0.0

    # --- Per-pool quality and toxicity ---
    for cname, interactions in contract_interactions.items():
        if not interactions:
            metrics.pool_avg_quality[cname] = 0.0
            metrics.pool_toxicity[cname] = 0.0
            metrics.pool_welfare[cname] = 0.0
            continue

        # Average quality
        avg_p = sum(i.p for i in interactions) / len(interactions)
        metrics.pool_avg_quality[cname] = avg_p

        # Toxicity: E[1-p | accepted]
        accepted = [i for i in interactions if i.accepted]
        if accepted:
            metrics.pool_toxicity[cname] = (
                sum(1 - i.p for i in accepted) / len(accepted)
            )
        else:
            metrics.pool_toxicity[cname] = 0.0

        # Welfare: average payoff per accepted interaction
        welfare_values = [payoff_engine.total_welfare(i) for i in accepted]
        if welfare_values:
            metrics.pool_welfare[cname] = (
                sum(welfare_values) / len(welfare_values)
            )
        else:
            metrics.pool_welfare[cname] = 0.0

    # --- Welfare delta ---
    # Compare mean per-interaction welfare across governed vs default pools
    governed_welfare: List[float] = []
    default_welfare = 0.0

    for cname, welfare in metrics.pool_welfare.items():
        if cname in _GOVERNED_TYPES:
            governed_welfare.append(welfare)
        elif cname == ContractType.DEFAULT_MARKET.value:
            default_welfare = welfare

    if governed_welfare:
        avg_governed = sum(governed_welfare) / len(governed_welfare)
        metrics.welfare_delta = avg_governed - default_welfare
    else:
        metrics.welfare_delta = 0.0

    # --- Attack displacement ---
    total_attacks = 0
    default_attacks = 0

    for cname, interactions in contract_interactions.items():
        for i in interactions:
            if i.p < attack_threshold_p:
                total_attacks += 1
                if cname == ContractType.DEFAULT_MARKET.value:
                    default_attacks += 1

    if total_attacks > 0:
        metrics.attack_displacement = default_attacks / total_attacks
    else:
        metrics.attack_displacement = 0.0

    return metrics

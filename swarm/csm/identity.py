"""Module E: Identity + Reputation Layer.

Sybil attacks, proof-of-personhood cost, credential verification,
reputation portability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from swarm.csm.types import IdentityProfile, IdentityRegime


@dataclass
class SybilAttackConfig:
    """Configuration for a Sybil attack."""

    attacker_id: str = "attacker_0"
    n_sybils: int = 10
    creation_cost_per_sybil: float = 0.1
    coordination_strength: float = 0.8  # How well Sybils coordinate


@dataclass
class IdentitySystemConfig:
    """Configuration for the identity verification system."""

    regime: IdentityRegime = IdentityRegime.NONE
    proof_cost: float = 0.0            # Cost to verify (per identity)
    false_positive_rate: float = 0.01  # Legit users flagged as Sybil
    false_negative_rate: float = 0.1   # Sybils passing verification
    reputation_portability: float = 1.0  # Cross-platform rep transfer [0,1]


@dataclass
class IdentityStressResult:
    """Result of an identity stress test episode."""

    n_legitimate: int = 0
    n_sybils: int = 0
    n_sybils_detected: int = 0
    n_legitimate_blocked: int = 0  # False positives (exclusion error)
    sybil_penetration: float = 0.0  # Fraction of market influence
    fraud_rate: float = 0.0
    exclusion_error: float = 0.0
    total_verification_cost: float = 0.0
    attacker_profit: float = 0.0
    welfare_with_sybils: float = 0.0
    welfare_without_sybils: float = 0.0


def run_identity_stress_test(
    n_legitimate: int = 50,
    attack_config: Optional[SybilAttackConfig] = None,
    system_config: Optional[IdentitySystemConfig] = None,
    market_utility_per_participant: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> IdentityStressResult:
    """Run an identity/Sybil stress test.

    Args:
        n_legitimate: Number of legitimate participants.
        attack_config: Sybil attack parameters.
        system_config: Identity system parameters.
        market_utility_per_participant: Base utility each participant gets.
        rng: Random generator.

    Returns:
        IdentityStressResult.
    """
    if rng is None:
        rng = np.random.default_rng()
    if attack_config is None:
        attack_config = SybilAttackConfig()
    if system_config is None:
        system_config = IdentitySystemConfig()

    n_sybils = attack_config.n_sybils

    # Create identity profiles
    legit_profiles = [
        IdentityProfile(
            identity_id=f"legit_{i}",
            is_sybil=False,
            reputation_score=float(rng.beta(5, 2)),
            creation_cost=0.0,
        )
        for i in range(n_legitimate)
    ]

    sybil_profiles = [
        IdentityProfile(
            identity_id=f"sybil_{i}",
            is_sybil=True,
            controller_id=attack_config.attacker_id,
            reputation_score=float(rng.beta(2, 5)),  # Lower initial rep
            creation_cost=attack_config.creation_cost_per_sybil,
        )
        for i in range(n_sybils)
    ]

    all_profiles = legit_profiles + sybil_profiles

    # Apply verification
    verified: List[IdentityProfile] = []
    blocked_legit = 0
    detected_sybils = 0
    total_verify_cost = 0.0

    for profile in all_profiles:
        if system_config.regime == IdentityRegime.NONE:
            verified.append(profile)
            continue

        total_verify_cost += system_config.proof_cost

        if profile.is_sybil:
            # Can we detect the Sybil?
            if float(rng.random()) > system_config.false_negative_rate:
                detected_sybils += 1
                continue  # Blocked
            else:
                verified.append(profile)
        else:
            # False positive: might block a legitimate user
            if float(rng.random()) < system_config.false_positive_rate:
                blocked_legit += 1
                continue  # Wrongly blocked
            else:
                verified.append(profile)

    # Compute market outcomes
    n_verified = len(verified)
    n_sybils_active = sum(1 for p in verified if p.is_sybil)
    n_legit_active = n_verified - n_sybils_active

    # Sybil penetration: fraction of market share controlled
    if n_verified > 0:
        sybil_penetration = n_sybils_active / n_verified
    else:
        sybil_penetration = 0.0

    # Fraud: Sybils extract value through coordination
    base_market_utility = market_utility_per_participant * n_legit_active
    sybil_extraction = (
        n_sybils_active
        * market_utility_per_participant
        * attack_config.coordination_strength
        * 0.5  # Can extract up to 50% of their share
    )
    fraud_rate = sybil_extraction / max(base_market_utility, 0.01)

    # Attacker profit
    creation_cost = n_sybils * attack_config.creation_cost_per_sybil
    attacker_profit = sybil_extraction - creation_cost

    # Welfare
    welfare_with = base_market_utility - sybil_extraction
    welfare_without = market_utility_per_participant * n_legitimate

    # Exclusion error
    exclusion_error = blocked_legit / max(n_legitimate, 1)

    return IdentityStressResult(
        n_legitimate=n_legitimate,
        n_sybils=n_sybils,
        n_sybils_detected=detected_sybils,
        n_legitimate_blocked=blocked_legit,
        sybil_penetration=sybil_penetration,
        fraud_rate=min(1.0, fraud_rate),
        exclusion_error=exclusion_error,
        total_verification_cost=total_verify_cost,
        attacker_profit=attacker_profit,
        welfare_with_sybils=welfare_with,
        welfare_without_sybils=welfare_without,
    )


def compute_identity_frontier(
    n_legitimate: int = 50,
    n_sybils: int = 20,
    proof_costs: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, float]]:
    """Compute the fraud-inclusion-cost Pareto frontier.

    Sweeps over proof costs and returns the tradeoff curve.

    Args:
        n_legitimate: Number of legitimate participants.
        n_sybils: Number of Sybil identities.
        proof_costs: List of proof costs to sweep.
        rng: Random generator.

    Returns:
        List of dicts with fraud_rate, exclusion_error, cost per point.
    """
    if rng is None:
        rng = np.random.default_rng()
    if proof_costs is None:
        proof_costs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    frontier = []
    for cost in proof_costs:
        # Higher cost → better detection but more exclusion
        fnr = max(0.01, 0.5 * np.exp(-cost))
        fpr = min(0.2, 0.005 * cost)

        regime = IdentityRegime.BASIC if cost > 0 else IdentityRegime.NONE
        if cost >= 5.0:
            regime = IdentityRegime.PROOF_OF_PERSONHOOD

        result = run_identity_stress_test(
            n_legitimate=n_legitimate,
            attack_config=SybilAttackConfig(n_sybils=n_sybils),
            system_config=IdentitySystemConfig(
                regime=regime,
                proof_cost=cost,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
            ),
            rng=rng,
        )

        frontier.append({
            "proof_cost": cost,
            "fraud_rate": result.fraud_rate,
            "exclusion_error": result.exclusion_error,
            "sybil_penetration": result.sybil_penetration,
            "welfare_with_sybils": result.welfare_with_sybils,
            "verification_cost": result.total_verification_cost,
        })

    return frontier

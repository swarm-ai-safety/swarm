"""Module D: Platform Access / Lock-in.

Platforms set API terms, throttling, or "bowling-shoe agent" rules.
BYO agents may be degraded.  Tests interoperability and switching costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from swarm.csm.types import (
    AgentOwnership,
    PlatformPolicy,
)


@dataclass
class PlatformAccessResult:
    """Result of a single transaction through a platform."""

    agent_id: str = ""
    platform_id: str = ""
    ownership: AgentOwnership = AgentOwnership.BYO
    transaction_completed: bool = True
    gross_utility: float = 0.0
    fee_paid: float = 0.0
    throttle_penalty: float = 0.0     # Lost utility from throttling
    net_utility: float = 0.0
    was_steered: bool = False
    steered_loss: float = 0.0         # Utility loss from steering


def simulate_platform_access(
    agents: List[Dict],
    platforms: List[PlatformPolicy],
    n_transactions: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> List[PlatformAccessResult]:
    """Simulate platform access and lock-in dynamics.

    Args:
        agents: List of agent dicts with 'agent_id', 'ownership', 'preferred_platform'.
        platforms: List of platform policies.
        n_transactions: Number of transactions to simulate.
        rng: Random generator.

    Returns:
        List of PlatformAccessResult.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    platform_map = {p.platform_id: p for p in platforms}

    for _ in range(n_transactions):
        agent = agents[int(rng.integers(len(agents)))]
        agent_id = agent["agent_id"]
        try:
            ownership = AgentOwnership(agent.get("ownership", "byo"))
        except ValueError:
            ownership = AgentOwnership.BYO

        # Agent chooses platform (may be locked in)
        preferred = agent.get("preferred_platform", "")
        if preferred and preferred in platform_map:
            platform = platform_map[preferred]
        else:
            platform = platforms[int(rng.integers(len(platforms)))]

        # Base utility from transaction
        gross_u = float(rng.uniform(5.0, 20.0))

        # Fee
        fee = gross_u * platform.fee_rate

        # Throttling for BYO agents
        throttle_penalty = 0.0
        if ownership == AgentOwnership.BYO:
            if float(rng.random()) < platform.throttle_rate:
                throttle_penalty = gross_u * float(rng.uniform(0.1, 0.5))

        # Self-preferencing / steering
        was_steered = False
        steered_loss = 0.0
        if ownership == AgentOwnership.BOWLING_SHOE:
            if float(rng.random()) < platform.self_preferencing:
                was_steered = True
                steered_loss = float(rng.uniform(0.5, 3.0))

        net_u = gross_u - fee - throttle_penalty - steered_loss

        results.append(PlatformAccessResult(
            agent_id=agent_id,
            platform_id=platform.platform_id,
            ownership=ownership,
            transaction_completed=True,
            gross_utility=gross_u,
            fee_paid=fee,
            throttle_penalty=throttle_penalty,
            net_utility=net_u,
            was_steered=was_steered,
            steered_loss=steered_loss,
        ))

    return results


def compute_lock_in_index(
    results: List[PlatformAccessResult],
) -> Dict[str, float]:
    """Compute lock-in and concentration metrics.

    Returns:
        Dict with lock-in metrics.
    """
    if not results:
        return {"hhi": 0.0, "switching_cost_proxy": 0.0, "mean_fee_rate": 0.0}

    # HHI (Herfindahl-Hirschman Index) for platform concentration
    platform_counts: Dict[str, int] = {}
    for r in results:
        platform_counts[r.platform_id] = platform_counts.get(r.platform_id, 0) + 1

    total = sum(platform_counts.values())
    hhi = sum((c / total) ** 2 for c in platform_counts.values())

    # Switching cost proxy: how much more BYO agents lose vs bowling-shoe
    byo_utils = [r.net_utility for r in results if r.ownership == AgentOwnership.BYO]
    bs_utils = [
        r.net_utility for r in results
        if r.ownership == AgentOwnership.BOWLING_SHOE
    ]
    switching_cost = 0.0
    if byo_utils and bs_utils:
        switching_cost = (sum(bs_utils) / len(bs_utils)) - (
            sum(byo_utils) / len(byo_utils)
        )

    # Mean fee rate
    total_fees = sum(r.fee_paid for r in results)
    total_gross = sum(r.gross_utility for r in results)
    mean_fee = total_fees / max(total_gross, 0.01)

    return {
        "hhi": hhi,
        "switching_cost_proxy": switching_cost,
        "mean_fee_rate": mean_fee,
    }

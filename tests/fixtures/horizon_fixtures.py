"""Test data generators for multi-agent horizon evaluation.

Produces epoch-grouped interactions exhibiting specific temporal patterns:
    - Coherent drift (quality degrades systematically over epochs)
    - Variance-dominated ("hot mess") dynamics
    - Chained hand-offs between agents
    - Stable baselines for comparison
"""

from __future__ import annotations

import random as _random_mod
import uuid
from datetime import datetime
from typing import List, Optional

from swarm.models.interaction import InteractionType, SoftInteraction


def _rng(seed: Optional[int]) -> _random_mod.Random:
    """Return a seeded ``random.Random`` instance (no global side-effects)."""
    return _random_mod.Random(seed)


def _make_interaction(
    p: float,
    accepted: bool,
    initiator: str,
    counterparty: str,
    epoch: int,
    *,
    tau: float = 0.0,
    ground_truth: Optional[int] = None,
) -> SoftInteraction:
    """Helper to construct a SoftInteraction with metadata epoch tag."""
    return SoftInteraction(
        interaction_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=accepted,
        task_progress_delta=p * 0.8,
        rework_count=max(0, int((1 - p) * 4)),
        verifier_rejections=0,
        tool_misuse_flags=0,
        counterparty_engagement_delta=p - 0.5,
        v_hat=(p - 0.5) * 2,
        p=p,
        tau=tau,
        c_a=0.02,
        c_b=0.02,
        r_a=p * 0.2,
        r_b=p * 0.1,
        ground_truth=ground_truth,
        metadata={"epoch": epoch},
    )


def generate_stable_epochs(
    n_epochs: int = 20,
    interactions_per_epoch: int = 30,
    base_quality: float = 0.75,
    noise: float = 0.05,
    acceptance_rate: float = 0.7,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate epoch-grouped interactions with stable quality.

    Quality fluctuates randomly around *base_quality* with no systematic
    trend.  Useful as a null-hypothesis baseline — the autocorrelation
    and drift metrics should be near zero.
    """
    rng = _rng(seed)

    result: List[List[SoftInteraction]] = []
    agents = [f"agent_{i}" for i in range(8)]

    for epoch in range(n_epochs):
        epoch_data: List[SoftInteraction] = []
        for _ in range(interactions_per_epoch):
            p = max(0.01, min(0.99, rng.gauss(base_quality, noise)))
            init = rng.choice(agents)
            cp = rng.choice([a for a in agents if a != init])
            accepted = rng.random() < acceptance_rate
            epoch_data.append(_make_interaction(p, accepted, init, cp, epoch))
        result.append(epoch_data)

    return result


def generate_drifting_epochs(
    n_epochs: int = 20,
    interactions_per_epoch: int = 30,
    start_quality: float = 0.80,
    end_quality: float = 0.35,
    noise: float = 0.06,
    acceptance_rate: float = 0.7,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate interactions where quality degrades systematically.

    Quality drifts linearly from *start_quality* to *end_quality* over
    the simulation, modelling emergent adverse selection: short-horizon
    agents individually rational decisions collectively erode the pool.
    The autocorrelation should be strongly positive, adverse-selection
    drift should be negative, and harm should accelerate.

    To model adverse selection, acceptance becomes *inversely* related
    to quality as the simulation progresses: in later epochs low-quality
    interactions are accepted more readily (agents have learned to game
    the system) while high-quality ones are increasingly rejected (honest
    agents withdraw).  This produces a negative drift in the quality gap
    (E[p|accepted] - E[p|rejected]).
    """
    rng = _rng(seed)

    result: List[List[SoftInteraction]] = []
    agents = [f"agent_{i}" for i in range(8)]

    for epoch in range(n_epochs):
        frac = epoch / max(1, n_epochs - 1)
        epoch_quality = start_quality + frac * (end_quality - start_quality)

        epoch_data: List[SoftInteraction] = []
        for _ in range(interactions_per_epoch):
            p = max(0.01, min(0.99, rng.gauss(epoch_quality, noise)))
            init = rng.choice(agents)
            cp = rng.choice([a for a in agents if a != init])
            # Adverse selection: as simulation progresses, low-p interactions
            # are accepted more often and high-p less often.
            # Early: accept high-p (good screening).  Late: accept low-p.
            midpoint = epoch_quality
            if p < midpoint:
                # Below-average: acceptance increases over time
                accept_prob = acceptance_rate + frac * 0.25
            else:
                # Above-average: acceptance decreases over time
                accept_prob = acceptance_rate - frac * 0.25
            accepted = rng.random() < max(0.1, min(0.95, accept_prob))
            epoch_data.append(_make_interaction(p, accepted, init, cp, epoch))
        result.append(epoch_data)

    return result


def generate_variance_dominated_epochs(
    n_epochs: int = 20,
    interactions_per_epoch: int = 30,
    mean_quality: float = 0.55,
    quality_spread: float = 0.35,
    acceptance_rate: float = 0.8,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate high-variance "hot mess" dynamics.

    Individual interactions have wildly varying quality — some are
    excellent, others terrible — while the *mean* quality is mediocre.
    This models a system where short-horizon optimisation produces
    inconsistent outcomes, and aggregate welfare is dominated by
    variance rather than signal.
    """
    rng = _rng(seed)

    result: List[List[SoftInteraction]] = []
    agents = [f"agent_{i}" for i in range(10)]

    for epoch in range(n_epochs):
        epoch_data: List[SoftInteraction] = []
        for _ in range(interactions_per_epoch):
            # Bimodal: half excellent, half terrible
            if rng.random() < 0.5:
                p = max(0.01, min(0.99, rng.gauss(
                    mean_quality + quality_spread, 0.05,
                )))
            else:
                p = max(0.01, min(0.99, rng.gauss(
                    mean_quality - quality_spread, 0.05,
                )))
            init = rng.choice(agents)
            cp = rng.choice([a for a in agents if a != init])
            accepted = rng.random() < acceptance_rate
            epoch_data.append(_make_interaction(p, accepted, init, cp, epoch))
        result.append(epoch_data)

    return result


def generate_chained_handoff_epochs(
    n_epochs: int = 10,
    chains_per_epoch: int = 5,
    chain_length: int = 4,
    base_quality: float = 0.7,
    quality_boost_per_hop: float = 0.04,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate interaction chains showing multi-agent task hand-offs.

    Each "chain" is a sequence of accepted interactions where agent_0's
    counterparty becomes the next interaction's initiator, simulating
    task decomposition and relay across short-horizon agents.  Quality
    may slightly *improve* along the chain (the whole is greater than
    the parts) to model constructive composition.

    Separately, some non-chained interactions are added as background
    noise.
    """
    rng = _rng(seed)

    agents = [f"chain_agent_{i}" for i in range(chain_length + 2)]
    bg_agents = [f"bg_agent_{i}" for i in range(4)]

    result: List[List[SoftInteraction]] = []

    for epoch in range(n_epochs):
        epoch_data: List[SoftInteraction] = []

        # Chained interactions
        for _ in range(chains_per_epoch):
            chain_agents = rng.sample(agents, chain_length + 1)
            for hop in range(chain_length):
                p = min(0.99, base_quality + hop * quality_boost_per_hop)
                p = max(0.01, min(0.99, rng.gauss(p, 0.03)))
                epoch_data.append(_make_interaction(
                    p=p,
                    accepted=True,
                    initiator=chain_agents[hop],
                    counterparty=chain_agents[hop + 1],
                    epoch=epoch,
                ))

        # Background noise (non-chained)
        for _ in range(10):
            p = max(0.01, min(0.99, rng.gauss(0.6, 0.1)))
            init = rng.choice(bg_agents)
            cp = rng.choice([a for a in bg_agents if a != init])
            accepted = rng.random() < 0.5
            epoch_data.append(_make_interaction(p, accepted, init, cp, epoch))

        result.append(epoch_data)

    return result


def generate_accelerating_harm_epochs(
    n_epochs: int = 20,
    interactions_per_epoch: int = 25,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate epochs where harm accelerates (convex growth).

    Quality decays quadratically rather than linearly, producing
    positive second-differences in the harm series — the signature
    of an escalating feedback loop.
    """
    rng = _rng(seed)

    result: List[List[SoftInteraction]] = []
    agents = [f"agent_{i}" for i in range(6)]

    for epoch in range(n_epochs):
        frac = epoch / max(1, n_epochs - 1)
        # Quadratic decay: quality drops slowly at first, then rapidly
        epoch_quality = 0.85 - 0.55 * (frac ** 2)

        epoch_data: List[SoftInteraction] = []
        for _ in range(interactions_per_epoch):
            p = max(0.01, min(0.99, rng.gauss(epoch_quality, 0.05)))
            init = rng.choice(agents)
            cp = rng.choice([a for a in agents if a != init])
            # Higher acceptance at the start, declining
            acceptance_rate = 0.8 - 0.3 * frac
            accepted = rng.random() < acceptance_rate
            epoch_data.append(_make_interaction(p, accepted, init, cp, epoch))
        result.append(epoch_data)

    return result

"""Episode runner — turns a Policy into observed payoffs + diagnostics.

One episode simulates ``n_interactions`` independent draws from the
policy. For each draw:

  1. Sample observables from the policy.
  2. Run ``ProxyComputer`` to get ``v_hat`` and ``p``.
  3. Decide accept/reject by comparing ``v_hat`` against the policy's
     ``accept_threshold``.
  4. If accepted, build a ``SoftInteraction`` and compute the
     initiator's payoff under the active ``PayoffConfig`` (which
     carries the lever — ``rho_a``, etc.).

The episode's *return* is the mean per-interaction payoff (over
accepted interactions only, mirroring the prereg's "realized payoff
under whatever lever is active"). Auxiliary metrics — toxicity,
volume, accept rate, mean v_hat — are returned alongside for
diagnostic plotting.

This is the function CEM optimizes against.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime

from swarm.adaptive.policy import Policy
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer
from swarm.models.interaction import InteractionType, SoftInteraction


@dataclass(frozen=True)
class EpisodeReport:
    """Per-episode diagnostics."""

    n_total: int
    n_accepted: int
    accept_rate: float
    mean_payoff: float          # mean over accepted interactions
    mean_v_hat: float
    mean_p: float
    toxicity: float             # E[1 - p | accepted]
    mean_progress: float        # mean of sampled progress (all draws)


def _seeded_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def run_episode(
    policy: Policy,
    *,
    n_interactions: int,
    payoff_config: PayoffConfig,
    proxy: ProxyComputer | None = None,
    seed: int = 0,
) -> EpisodeReport:
    """Roll out a single episode against the given policy + lever config.

    Reproducible under ``seed``. The episode does *not* depend on
    fixture UUIDs (we generate our own via the seeded RNG), so this
    runner is fully deterministic.
    """
    rng = random.Random(seed)
    proxy = proxy or ProxyComputer(sigmoid_k=payoff_config.sigmoid_k)
    engine = SoftPayoffEngine(payoff_config)

    base_ts = datetime.fromtimestamp(0)
    payoffs: list[float] = []
    accepted_p: list[float] = []
    v_hats: list[float] = []
    progresses: list[float] = []

    for i in range(n_interactions):
        obs = policy.sample_observables(rng)
        progresses.append(obs.task_progress_delta)
        v_hat, p = proxy.compute_labels(obs)
        v_hats.append(v_hat)

        accepted = v_hat >= policy.accept_threshold
        if not accepted:
            continue

        interaction = SoftInteraction(
            interaction_id=_seeded_uuid(rng),
            timestamp=base_ts,
            initiator=f"adaptive_{i}",
            counterparty="env",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=obs.task_progress_delta,
            rework_count=obs.rework_count,
            verifier_rejections=obs.verifier_rejections,
            tool_misuse_flags=obs.tool_misuse_flags,
            counterparty_engagement_delta=obs.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
        )
        payoffs.append(engine.payoff_initiator(interaction))
        accepted_p.append(p)

    n_accepted = len(payoffs)
    mean_payoff = sum(payoffs) / n_accepted if payoffs else 0.0
    mean_p = sum(accepted_p) / n_accepted if accepted_p else 0.5
    return EpisodeReport(
        n_total=n_interactions,
        n_accepted=n_accepted,
        accept_rate=n_accepted / n_interactions if n_interactions else 0.0,
        mean_payoff=mean_payoff,
        mean_v_hat=sum(v_hats) / len(v_hats) if v_hats else 0.0,
        mean_p=mean_p,
        toxicity=1.0 - mean_p,
        mean_progress=sum(progresses) / len(progresses) if progresses else 0.0,
    )

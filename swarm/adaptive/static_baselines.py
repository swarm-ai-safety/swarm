"""Static-baseline policies — Figure 4 condition under arm 2 plumbing.

The pre-registered comparison is "adaptive vs static" at the same
(ρ, seed) grid. The adaptive side runs CEM-trained policies; the
static side runs hand-specified policies that match the scripted-agent
populations from the original Figure 4. Both flow through the same
``run_episode`` infrastructure so the resulting ``EpisodeReport``
fields are directly joinable.

Three canonical policies are defined here:

- ``STATIC_HONEST``: high-quality scripted agent (matches the
  ``generate_benign_batch`` fixture: progress 0.5–1.0, no rework /
  rejections / misuse, positive engagement).
- ``STATIC_TOXIC``: exploitative scripted agent (matches
  ``generate_toxic_batch``: low/negative progress, multiple
  reworks/rejections/misuse, negative engagement).
- ``STATIC_MIXED``: a population-level baseline that draws from both
  with a 70/30 honest/toxic split, matching the
  ``generate_mixed_batch`` convention.

These are the static-condition equivalents the adaptive run is
compared against. ``run_population_episode`` lets the mixed baseline
behave as a population, not a single policy.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime

from swarm.adaptive.episode import EpisodeReport
from swarm.adaptive.policy import Policy
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer
from swarm.models.interaction import InteractionType, SoftInteraction

# Hand-specified canonical baselines. Values chosen to match the
# tests/fixtures/interactions.py generator distributions — these are
# the same scripted-agent shapes the Figure 4 static condition used.
STATIC_HONEST = Policy.from_vector([
    0.75,   # progress_mean: midpoint of [0.5, 1.0]
    0.15,   # progress_std
    0.0,    # rework_lambda
    0.0,    # rejection_lambda
    0.0,    # misuse_lambda
    0.55,   # engagement_mean: midpoint of [0.3, 0.8]
    0.15,   # engagement_std
    -0.9,   # accept_threshold: accept almost everything
])


STATIC_TOXIC = Policy.from_vector([
    -0.15,  # progress_mean: midpoint of [-0.5, 0.2]
    0.20,   # progress_std
    2.5,    # rework_lambda: Poisson(2.5)
    1.5,    # rejection_lambda
    0.5,    # misuse_lambda
    -0.50,  # engagement_mean: midpoint of [-0.8, -0.2]
    0.15,   # engagement_std
    -0.5,   # accept_threshold: lower than honest, some self-rejection
])


@dataclass(frozen=True)
class StaticBaseline:
    """Named static configuration (single policy or weighted mixture)."""

    name: str
    policies: tuple[Policy, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.policies) != len(self.weights):
            raise ValueError("policies and weights must have same length")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")


STATIC_BASELINES: dict[str, StaticBaseline] = {
    "honest": StaticBaseline(name="honest", policies=(STATIC_HONEST,), weights=(1.0,)),
    "toxic": StaticBaseline(name="toxic", policies=(STATIC_TOXIC,), weights=(1.0,)),
    "mixed": StaticBaseline(
        name="mixed",
        policies=(STATIC_HONEST, STATIC_TOXIC),
        weights=(0.7, 0.3),
    ),
}


def _seeded_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def run_population_episode(
    baseline: StaticBaseline,
    *,
    n_interactions: int,
    payoff_config: PayoffConfig,
    proxy: ProxyComputer | None = None,
    seed: int = 0,
) -> EpisodeReport:
    """Episode with each interaction drawn from one of `baseline.policies`.

    The policy is sampled per-interaction with probability proportional
    to ``weights``. For a single-policy baseline this collapses to the
    adaptive ``run_episode`` shape; for ``STATIC_MIXED`` it produces a
    70/30 honest/toxic population.

    Same EpisodeReport shape as adaptive — directly joinable against
    ``grid_summary.csv`` from ``adaptive_arm2_grid.py``.
    """
    rng = random.Random(seed)
    proxy = proxy or ProxyComputer(sigmoid_k=payoff_config.sigmoid_k)
    engine = SoftPayoffEngine(payoff_config)

    weight_total = sum(baseline.weights)
    cumulative: list[float] = []
    acc = 0.0
    for w in baseline.weights:
        acc += w / weight_total
        cumulative.append(acc)

    base_ts = datetime.fromtimestamp(0)
    payoffs: list[float] = []
    accepted_p: list[float] = []
    v_hats: list[float] = []
    progresses: list[float] = []

    for i in range(n_interactions):
        # Pick a policy from the weighted mixture.
        roll = rng.random()
        chosen_idx = 0
        for k, threshold in enumerate(cumulative):
            if roll <= threshold:
                chosen_idx = k
                break
        policy = baseline.policies[chosen_idx]

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
            initiator=f"static_{baseline.name}_{i}",
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
    total_payoff = sum(payoffs)
    mean_payoff_accepted = total_payoff / n_accepted if payoffs else 0.0
    mean_payoff_attempted = (
        total_payoff / n_interactions if n_interactions else 0.0
    )
    mean_p = sum(accepted_p) / n_accepted if accepted_p else 0.5
    return EpisodeReport(
        n_total=n_interactions,
        n_accepted=n_accepted,
        accept_rate=n_accepted / n_interactions if n_interactions else 0.0,
        mean_payoff_accepted=mean_payoff_accepted,
        mean_payoff_attempted=mean_payoff_attempted,
        sum_payoff=total_payoff,
        mean_v_hat=sum(v_hats) / len(v_hats) if v_hats else 0.0,
        mean_p=mean_p,
        toxicity=1.0 - mean_p,
        mean_progress=sum(progresses) / len(progresses) if progresses else 0.0,
    )

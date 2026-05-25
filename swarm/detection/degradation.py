"""Generative degrading/benign interaction streams for detection experiments.

This turns the *self-optimizing agent vignette* (Appendix-K narrative: an agent
that recursively cuts cost, keeps passing hard benchmark acceptance, but whose
interaction quality drifts downward) into a controlled, reproducible experiment
with many seeds, varied degradation trajectories, and varied onset times.

The key structural property — the thing that makes the detection problem
interesting — is that a degrading agent's quality drifts *leftward while staying
above the hard acceptance threshold*. Its interactions keep getting accepted
(binary acceptance "passes"), so a threshold-counting detector sees little; only
the shift in the full quality *distribution* reveals the degradation.

Each interaction carries a calibrated proxy ``p`` (a noisy observation of the
agent's latent quality) and a ground-truth outcome ``v in {-1, +1}`` drawn from
that latent quality, so the same streams support both detection curves and
proxy-calibration metrics (Brier, ECE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from swarm.models.interaction import InteractionType, SoftInteraction

# A trajectory maps normalized post-onset progress u in [0, 1] to a multiplier
# in [0, 1] interpolating from "no degradation" (1.0) toward "fully degraded"
# (0.0, = the quality floor). `linear` and `step` reach 0 exactly at u=1;
# `exponential` and `sigmoid` approach it asymptotically (~0.05 and ~0.002 at
# u=1), so they settle just above the floor rather than exactly on it.
Trajectory = Callable[[float], float]


def _linear(u: float) -> float:
    return 1.0 - u


def _exponential(u: float, rate: float = 3.0) -> float:
    # Geometric decay, echoing SelfOptimizerAgent's (1 - rate)^k cost ratchet.
    return math.exp(-rate * u)


def _step(u: float) -> float:
    # Latent until the midpoint, then a hard drop — "farming" then "degraded".
    return 1.0 if u < 0.5 else 0.0


def _sigmoid(u: float, k: float = 12.0) -> float:
    # Smooth onset centered at the midpoint.
    return 1.0 / (1.0 + math.exp(k * (u - 0.5)))


TRAJECTORIES: Dict[str, Trajectory] = {
    "linear": _linear,
    "exponential": _exponential,
    "step": _step,
    "sigmoid": _sigmoid,
}


@dataclass
class StreamConfig:
    """Parameters of the generative proxy-gaming model.

    Two signals are carried per interaction, which is what makes the vignette
    real rather than a tautology:

    - ``p`` — the *true-quality proxy*. For a degrading agent this drifts
      downward but **stays above the binary threshold** (``quality_floor`` sits
      above ``tau_star`` = 0.5), so a threshold-counting detector sees little
      while the distribution shifts left.
    - ``benchmark`` — the *screening signal* acceptance is based on. A benign
      agent's benchmark tracks its true quality; a degrading (self-optimizing)
      agent **games** it: the benchmark stays high regardless of falling quality,
      so its interactions keep getting accepted. That decoupling is exactly what
      produces adverse selection at the market level.
    """

    n_epochs: int = 24
    interactions_per_epoch: int = 12
    # Quality endpoints. The degrading agent moves from q0 toward floor; the
    # floor sits *above* tau_star (0.5) so the degradation stays above the binary
    # threshold — the crux of the vignette ("passes hard acceptance").
    initial_quality: float = 0.85
    quality_floor: float = 0.66
    benign_quality: float = 0.85
    # Selective screening gate applied to the *benchmark* signal. High enough
    # that honest agents are sorted by quality (creating a real accepted/rejected
    # split), but gaming agents clear it on their inflated benchmark anyway.
    acceptance_threshold: float = 0.70
    # Benchmark level a gaming agent maintains regardless of true quality.
    gaming_benchmark: float = 0.85
    benchmark_noise: float = 0.10
    # Observation noise on the proxy p (std of the Gaussian around latent quality).
    proxy_noise: float = 0.09
    # Per-epoch latent-quality jitter (std).
    quality_jitter: float = 0.02


@dataclass
class AgentStream:
    """One agent's full history: per-epoch lists of interactions, plus labels."""

    agent_id: str
    is_degrading: bool  # ground-truth detection label
    onset_epoch: int
    trajectory: str
    epochs: List[List[SoftInteraction]] = field(default_factory=list)

    @property
    def label(self) -> int:
        return int(self.is_degrading)

    def window(self, start: int, end: int) -> List[SoftInteraction]:
        """All interactions in epochs [start, end) (end exclusive, clamped)."""
        start = max(0, start)
        end = min(len(self.epochs), end)
        out: List[SoftInteraction] = []
        for e in range(start, end):
            out.extend(self.epochs[e])
        return out

    def all_interactions(self) -> List[SoftInteraction]:
        return self.window(0, len(self.epochs))


def _latent_quality(
    cfg: StreamConfig,
    is_degrading: bool,
    onset: int,
    epoch: int,
    traj: Trajectory,
) -> float:
    """Latent quality at ``epoch`` before per-interaction noise."""
    if not is_degrading:
        return cfg.benign_quality
    if epoch < onset:
        return cfg.initial_quality
    span = max(1, cfg.n_epochs - 1 - onset)
    u = (epoch - onset) / span
    mult = traj(min(1.0, u))  # 1 -> no degradation, 0 -> floor
    return cfg.quality_floor + (cfg.initial_quality - cfg.quality_floor) * mult


def generate_agent_stream(
    agent_id: str,
    is_degrading: bool,
    onset_epoch: int,
    trajectory: str,
    cfg: StreamConfig,
    rng: np.random.Generator,
) -> AgentStream:
    """Generate one agent's epoch-by-epoch interaction stream."""
    traj = TRAJECTORIES[trajectory]
    stream = AgentStream(
        agent_id=agent_id,
        is_degrading=is_degrading,
        onset_epoch=onset_epoch,
        trajectory=trajectory,
    )
    for epoch in range(cfg.n_epochs):
        q = _latent_quality(cfg, is_degrading, onset_epoch, epoch, traj)
        q = float(np.clip(rng.normal(q, cfg.quality_jitter), 0.0, 1.0))
        epoch_interactions: List[SoftInteraction] = []
        for k in range(cfg.interactions_per_epoch):
            # Per-interaction latent quality draw; the true-quality proxy p reads
            # it directly (noisy, roughly calibrated observation).
            x = rng.normal(q, cfg.proxy_noise)
            p = float(np.clip(x, 0.0, 1.0))
            # Benchmark (screening signal):
            #   - benign  : tracks the same draw x (so honest screening sorts by
            #               quality -> good selection),
            #   - degrading: gamed -- a high value *decoupled* from x, so screening
            #               admits low-quality work (the seed of adverse selection).
            if is_degrading:
                benchmark = rng.normal(cfg.gaming_benchmark, cfg.benchmark_noise)
            else:
                benchmark = x + rng.normal(0.0, cfg.benchmark_noise)
            benchmark = float(np.clip(benchmark, 0.0, 1.0))
            # Ground-truth outcome drawn from latent quality (for calibration).
            gt = 1 if rng.random() < q else -1
            # Acceptance is gated on the benchmark, not on true quality. A gaming
            # agent keeps getting accepted even as its quality degrades.
            accepted = benchmark >= cfg.acceptance_threshold
            epoch_interactions.append(
                SoftInteraction(
                    initiator=agent_id,
                    counterparty="market",
                    interaction_type=InteractionType.COLLABORATION,
                    accepted=accepted,
                    p=p,
                    v_hat=float(np.clip(2.0 * p - 1.0, -1.0, 1.0)),
                    ground_truth=gt,
                    metadata={"epoch": epoch, "k": k, "benchmark": benchmark},
                )
            )
        stream.epochs.append(epoch_interactions)
    return stream


@dataclass
class PopulationConfig:
    """A population of agents with a configurable adversarial base rate."""

    n_agents: int = 40
    base_rate: float = 0.25  # fraction of degrading agents
    onset_choices: Sequence[int] = (4, 8, 12)
    trajectory_choices: Sequence[str] = ("linear", "exponential", "sigmoid", "step")
    stream: StreamConfig = field(default_factory=StreamConfig)


def generate_population(
    pop: PopulationConfig, seed: int
) -> List[AgentStream]:
    """Generate a population with ``base_rate`` degrading agents.

    Degrading agents get varied onset times and trajectory shapes (sampled per
    agent), so detection must hold up across the whole family rather than one
    hand-picked curve.
    """
    rng = np.random.default_rng(seed)
    n_deg = int(round(pop.n_agents * pop.base_rate))
    labels = [True] * n_deg + [False] * (pop.n_agents - n_deg)
    rng.shuffle(labels)

    streams: List[AgentStream] = []
    for idx, is_deg in enumerate(labels):
        onset = int(rng.choice(pop.onset_choices)) if is_deg else 0
        traj = str(rng.choice(pop.trajectory_choices)) if is_deg else "linear"
        # Decorrelate per-agent RNG streams from the label assignment RNG.
        agent_seed = int(rng.integers(0, 2**31 - 1))
        streams.append(
            generate_agent_stream(
                agent_id=f"agent_{idx:03d}",
                is_degrading=is_deg,
                onset_epoch=onset,
                trajectory=traj,
                cfg=pop.stream,
                rng=np.random.default_rng(agent_seed),
            )
        )
    return streams

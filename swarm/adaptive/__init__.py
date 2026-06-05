"""Adaptive-agents study — arm 2 (adaptive-generation) scaffolding.

Pre-registration: docs/research/adaptive-agents-prereg.md.

This module implements the minimal pieces needed to run a CEM-trained
generation policy against the existing payoff/proxy machinery, so the
adaptive-vs-static comparison can be made on the same scenarios used
in Table 4.

Scope of this scaffold:
  - `Policy` (8-parameter bounded vector over observable distributions
    plus accept_threshold)
  - `run_episode` (deterministic given seed)
  - `train_cem` (CEM trainer with pre-registered budget)

Out of scope (intentional, follow-ups):
  - Full ρ-grid sweep across 5 seeds (this is the powered run; this
    scaffold is the single-condition smoke).
  - LLM-feedback corroboration arm.
  - Calibration-anchor integration (the v3 anchor scores accepted
    interactions; that comes after the basic learning loop is shown
    to converge).
"""

from swarm.adaptive.cem import (
    SIGMA_FLOOR_FRAC,
    CEMConfig,
    CEMIterationReport,
    CEMTrainingReport,
    train_cem,
)
from swarm.adaptive.episode import EpisodeReport, run_episode
from swarm.adaptive.policy import PARAM_DIM, PARAM_NAMES, PARAM_SPEC, Policy

__all__ = [
    "CEMConfig",
    "CEMIterationReport",
    "CEMTrainingReport",
    "EpisodeReport",
    "PARAM_DIM",
    "PARAM_NAMES",
    "PARAM_SPEC",
    "Policy",
    "SIGMA_FLOOR_FRAC",
    "run_episode",
    "train_cem",
]

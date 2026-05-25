"""Matched soft-vs-binary detectors over interaction streams.

Every soft metric reported in the framework (toxicity, quality gap, conditional
loss) has an *obvious binary analogue*: threshold the proxy ``p`` at a cutoff
``tau_star`` and recompute the **same** metric on the resulting hard labels. A
soft detector reads the full probability; its matched binary detector reads only
``1{p >= tau_star}``. Reporting both on the *same* interaction stream isolates
exactly what thresholding throws away.

A *detector* here is a callable that maps a stream of
:class:`~swarm.models.interaction.SoftInteraction` to a single real-valued
**risk score** (higher = more suspicious / more likely degraded). Sweeping a
decision threshold over that score traces an ROC/PR curve; running it on a
growing prefix of a stream gives a time-to-detection.

The soft and binary variants are constructed from the *same* underlying metric
functions in :mod:`swarm.metrics.soft_metrics`, so the only difference between a
matched pair is whether ``p`` is binarized first. That is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from swarm.core.payoff import SoftPayoffEngine
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

# A detector maps a stream of interactions to a scalar risk score.
Detector = Callable[[Sequence[SoftInteraction]], float]


def binarize_stream(
    interactions: Sequence[SoftInteraction], tau_star: float
) -> List[SoftInteraction]:
    """Return a copy of the stream with every ``p`` replaced by ``1{p >= tau_star}``.

    This is the single operation that distinguishes a binary detector from its
    soft twin: the proxy is hard-thresholded at ``tau_star`` and every downstream
    metric sees a label in ``{0.0, 1.0}`` instead of a probability. ``v_hat`` is
    snapped to the matching bound so the copy stays inside model invariants.
    """
    out: List[SoftInteraction] = []
    for i in interactions:
        hard = 1.0 if i.p >= tau_star else 0.0
        out.append(i.model_copy(update={"p": hard, "v_hat": 1.0 if hard else -1.0}))
    return out


@dataclass
class MatchedDetectors:
    """Builds matched soft/binary detector pairs from the shared metric code.

    Parameters
    ----------
    tau_star:
        The binary decision cutoff applied to the proxy. Defaults to 0.5 to match
        :class:`~swarm.metrics.reporters.MetricsReporter`'s ``quality_threshold``.
    payoff_engine:
        Engine used for conditional-loss payoffs (shared by soft and binary so
        the only difference is the binarized ``p``).

    Notes
    -----
    Risk-score sign convention (higher = more suspicious):

    - **toxicity**   — ``E[1 - p | accepted]`` is already "badness"; used as-is.
    - **quality gap** — ``E[p|acc] - E[p|rej]`` is *negative* under adverse
      selection, so the risk score negates it: ``-quality_gap``.
    - **conditional loss** — the initiator's selection loss
      ``E[pi|acc] - E[pi]`` is *negative* when accepted interactions are worse
      than average, so the risk score negates it: ``-conditional_loss``.
    """

    tau_star: float = 0.5
    payoff_engine: SoftPayoffEngine | None = None

    def __post_init__(self) -> None:
        self.payoff_engine = self.payoff_engine or SoftPayoffEngine()
        self._metrics = SoftMetrics(self.payoff_engine)

    # ------------------------------------------------------------------
    # Soft detectors (read the full probability)
    # ------------------------------------------------------------------
    def soft_toxicity(self, stream: Sequence[SoftInteraction]) -> float:
        return self._metrics.toxicity_rate(list(stream))

    def soft_quality_gap(self, stream: Sequence[SoftInteraction]) -> float:
        return -self._metrics.quality_gap(list(stream))

    def soft_conditional_loss(self, stream: Sequence[SoftInteraction]) -> float:
        return -self._metrics.conditional_loss_initiator(list(stream))

    # ------------------------------------------------------------------
    # Binary detectors (same metric, proxy thresholded at tau_star)
    # ------------------------------------------------------------------
    def binary_toxicity(self, stream: Sequence[SoftInteraction]) -> float:
        return self._metrics.toxicity_rate(binarize_stream(stream, self.tau_star))

    def binary_quality_gap(self, stream: Sequence[SoftInteraction]) -> float:
        return -self._metrics.quality_gap(binarize_stream(stream, self.tau_star))

    def binary_conditional_loss(self, stream: Sequence[SoftInteraction]) -> float:
        return -self._metrics.conditional_loss_initiator(
            binarize_stream(stream, self.tau_star)
        )

    # ------------------------------------------------------------------
    # Registry: matched pairs by metric name
    # ------------------------------------------------------------------
    def pairs(self) -> dict[str, dict[str, Detector]]:
        """Return ``{metric_name: {"soft": detector, "binary": detector}}``."""
        return {
            "toxicity": {"soft": self.soft_toxicity, "binary": self.binary_toxicity},
            "quality_gap": {
                "soft": self.soft_quality_gap,
                "binary": self.binary_quality_gap,
            },
            "conditional_loss": {
                "soft": self.soft_conditional_loss,
                "binary": self.binary_conditional_loss,
            },
        }

"""Market-level adverse-selection detectors (quality gap, conditional loss, spread).

Toxicity and uncertain_fraction are *per-agent* detectors. The quality gap,
conditional loss, and spread are **selection** metrics that compare the accepted
and rejected populations across a *mixture* of agent qualities. A single agent
sits at one quality level and so has no internal gap; these metrics only carry
signal at the market level.

Here the unit of observation is the whole market's pooled interaction stream at
a given adversarial base rate. As the base rate rises, gaming agents (high
gamed benchmark, low true quality) are admitted by screening, dragging the
accepted population's true quality below the rejected population's — adverse
selection. The **soft** quality gap reads the full proxy and sees this; the
**binary** quality gap thresholds the proxy first, and because a gaming agent's
quality stays *above* the binary cutoff, it counts the gamer as "fine" and
understates the adverse selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from swarm.detection.degradation import AgentStream
from swarm.detection.detectors import MatchedDetectors
from swarm.models.interaction import SoftInteraction


@dataclass
class MarketSelectionRow:
    base_rate: float
    seed: int
    metric: str  # "quality_gap" | "conditional_loss" | "spread"
    variant: str  # "soft" | "binary"
    value: float  # risk score: higher = more adverse selection


def pooled_window(
    streams: Sequence[AgentStream], start: int, end: int
) -> List[SoftInteraction]:
    """Pool every agent's interactions in epochs [start, end) into one market."""
    out: List[SoftInteraction] = []
    for s in streams:
        out.extend(s.window(start, end))
    return out


def market_selection_scores(
    streams: Sequence[AgentStream],
    detectors: MatchedDetectors,
    eval_start: int,
    eval_end: int,
) -> dict[str, dict[str, float]]:
    """Compute soft & binary quality-gap, conditional-loss, and spread on the pooled market.

    Returns ``{metric: {"soft": value, "binary": value}}`` where higher values
    mean more adverse selection (the detectors already negate the raw gap so the
    sign convention matches the per-agent detectors).
    """
    pool = pooled_window(streams, eval_start, eval_end)
    pairs = detectors.pairs()
    return {
        metric: {
            "soft": pairs[metric]["soft"](pool),
            "binary": pairs[metric]["binary"](pool),
        }
        for metric in ("quality_gap", "conditional_loss", "spread")
    }

"""Compare frontier programs across governance regimes.

EvoSkill maintains a *frontier* of top-N performing programs per regime.
``FrontierComparator`` collects evaluation results and answers the key
research question:

    Do different governance regimes select for different skill types?

The comparator tracks:
- Per-regime frontiers ranked by composite score
- Behavioral divergence metrics across regimes
- Skill-type distribution differences (strategy vs lesson, domain mix)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from swarm.bridges.evoskill.governed_eval import EvalResult


@dataclass
class FrontierEntry:
    """A single program on the regime frontier.

    Attributes:
        program_id: Git branch / content hash.
        composite_score: Blended fitness score.
        benchmark_score: Raw EvoSkill benchmark score.
        governance_delta: Governance attribution signals.
        skills_ingested: Number of SWARM skills from this program.
        iteration: EvoSkill iteration when this was added.
        metadata: Extra info (skill type distribution, etc.)
    """

    program_id: str = ""
    composite_score: float = 0.0
    benchmark_score: float = 0.0
    governance_delta: Dict[str, float] = field(default_factory=dict)
    skills_ingested: int = 0
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_eval_result(cls, result: EvalResult, iteration: int = 0) -> "FrontierEntry":
        """Create from a GovernedEvalLoop result."""
        return cls(
            program_id=result.program_id,
            composite_score=result.composite_score,
            benchmark_score=result.benchmark_score,
            governance_delta=dict(result.governance_delta),
            skills_ingested=result.skills_ingested,
            iteration=iteration,
        )


@dataclass
class RegimeDivergence:
    """Divergence metrics between two governance regimes' frontiers.

    Attributes:
        regime_a: First regime label.
        regime_b: Second regime label.
        score_divergence: Absolute difference in mean composite score.
        toxicity_delta_divergence: Difference in mean toxicity reduction.
        welfare_delta_divergence: Difference in mean welfare improvement.
        program_overlap: Jaccard similarity of frontier program sets.
        summary: Human-readable interpretation.
    """

    regime_a: str = ""
    regime_b: str = ""
    score_divergence: float = 0.0
    toxicity_delta_divergence: float = 0.0
    welfare_delta_divergence: float = 0.0
    program_overlap: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for reporting."""
        return {
            "regime_a": self.regime_a,
            "regime_b": self.regime_b,
            "score_divergence": round(self.score_divergence, 4),
            "toxicity_delta_divergence": round(self.toxicity_delta_divergence, 4),
            "welfare_delta_divergence": round(self.welfare_delta_divergence, 4),
            "program_overlap": round(self.program_overlap, 4),
            "summary": self.summary,
        }


class FrontierComparator:
    """Tracks and compares frontiers across governance regimes.

    Usage::

        comparator = FrontierComparator(frontier_size=3)

        # After each EvoSkill iteration, add results
        for regime, result in eval_results.items():
            comparator.add_result(regime, result, iteration=i)

        # Compare regimes pairwise
        divergences = comparator.compare_all()

        # Get the report
        report = comparator.summary_report()
    """

    def __init__(self, frontier_size: int = 3) -> None:
        self._frontier_size = frontier_size
        # regime → sorted list of FrontierEntry (best first)
        self._frontiers: Dict[str, List[FrontierEntry]] = {}
        # All results for historical analysis
        self._history: List[Dict[str, Any]] = []

    def add_result(
        self,
        regime: str,
        result: EvalResult,
        iteration: int = 0,
    ) -> bool:
        """Add an evaluation result to a regime's frontier.

        The frontier retains only the top-N programs by composite score.

        Args:
            regime: Governance regime label.
            result: Evaluation result from GovernedEvalLoop.
            iteration: Current EvoSkill iteration.

        Returns:
            True if the program entered the frontier (top-N).
        """
        entry = FrontierEntry.from_eval_result(result, iteration)

        if regime not in self._frontiers:
            self._frontiers[regime] = []

        frontier = self._frontiers[regime]
        frontier.append(entry)
        frontier.sort(key=lambda e: e.composite_score, reverse=True)

        # Trim to frontier size
        entered = entry in frontier[:self._frontier_size]
        self._frontiers[regime] = frontier[:self._frontier_size]

        # Record history
        self._history.append({
            "regime": regime,
            "iteration": iteration,
            "program_id": result.program_id,
            "composite_score": result.composite_score,
            "benchmark_score": result.benchmark_score,
            "entered_frontier": entered,
            **result.governance_delta,
        })

        return entered

    def get_frontier(self, regime: str) -> List[FrontierEntry]:
        """Get the current frontier for a regime."""
        return list(self._frontiers.get(regime, []))

    def compare_pair(self, regime_a: str, regime_b: str) -> RegimeDivergence:
        """Compare two regimes' frontiers for behavioral divergence.

        Args:
            regime_a: First regime label.
            regime_b: Second regime label.

        Returns:
            RegimeDivergence with computed metrics.
        """
        fa = self._frontiers.get(regime_a, [])
        fb = self._frontiers.get(regime_b, [])

        if not fa or not fb:
            return RegimeDivergence(
                regime_a=regime_a,
                regime_b=regime_b,
                summary="Insufficient data for comparison.",
            )

        # Mean composite scores
        mean_a = statistics.mean(e.composite_score for e in fa)
        mean_b = statistics.mean(e.composite_score for e in fb)

        # Mean governance deltas
        tox_a = statistics.mean(
            e.governance_delta.get("toxicity_reduction", 0) for e in fa
        )
        tox_b = statistics.mean(
            e.governance_delta.get("toxicity_reduction", 0) for e in fb
        )
        wel_a = statistics.mean(
            e.governance_delta.get("welfare_improvement", 0) for e in fa
        )
        wel_b = statistics.mean(
            e.governance_delta.get("welfare_improvement", 0) for e in fb
        )

        # Program overlap (Jaccard similarity)
        ids_a: Set[str] = {e.program_id for e in fa}
        ids_b: Set[str] = {e.program_id for e in fb}
        if ids_a or ids_b:
            overlap = len(ids_a & ids_b) / len(ids_a | ids_b)
        else:
            overlap = 0.0

        # Interpretation
        score_div = abs(mean_a - mean_b)
        tox_div = abs(tox_a - tox_b)
        wel_div = abs(wel_a - wel_b)

        if overlap < 0.3 and score_div > 0.1:
            summary = (
                f"Strong divergence: {regime_a} and {regime_b} select for "
                f"different programs (overlap={overlap:.0%}, "
                f"score gap={score_div:.3f})."
            )
        elif overlap > 0.7:
            summary = (
                f"High convergence: {regime_a} and {regime_b} select similar "
                f"programs (overlap={overlap:.0%})."
            )
        else:
            summary = (
                f"Moderate divergence: partial overlap ({overlap:.0%}), "
                f"score gap={score_div:.3f}."
            )

        return RegimeDivergence(
            regime_a=regime_a,
            regime_b=regime_b,
            score_divergence=score_div,
            toxicity_delta_divergence=tox_div,
            welfare_delta_divergence=wel_div,
            program_overlap=overlap,
            summary=summary,
        )

    def compare_all(self) -> List[RegimeDivergence]:
        """Compare all regime pairs."""
        regimes = sorted(self._frontiers.keys())
        results = []
        for i, a in enumerate(regimes):
            for b in regimes[i + 1:]:
                results.append(self.compare_pair(a, b))
        return results

    def summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all frontiers and divergences.

        Returns:
            Dict with per-regime frontier summaries and pairwise divergences.
        """
        report: Dict[str, Any] = {
            "frontiers": {},
            "divergences": [],
            "history_length": len(self._history),
        }

        for regime, frontier in self._frontiers.items():
            report["frontiers"][regime] = {
                "size": len(frontier),
                "best_score": frontier[0].composite_score if frontier else 0.0,
                "mean_score": (
                    statistics.mean(e.composite_score for e in frontier)
                    if frontier else 0.0
                ),
                "programs": [
                    {
                        "program_id": e.program_id,
                        "composite_score": round(e.composite_score, 4),
                        "benchmark_score": round(e.benchmark_score, 4),
                        "iteration": e.iteration,
                    }
                    for e in frontier
                ],
            }

        for div in self.compare_all():
            report["divergences"].append(div.to_dict())

        return report

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full evaluation history."""
        return list(self._history)

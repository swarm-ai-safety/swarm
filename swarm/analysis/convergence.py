"""Behavioral convergence detection across sweep configurations.

Inspired by the Optimization Arena finding that wildly different
implementations converge to nearly identical behavior under strong
selection pressure.  In a safety context, convergence can signal
that diverse strategies have found the same reward-landscape
attractor — which may be genuine alignment *or* proxy gaming.

Usage::

    from swarm.analysis.convergence import behavioral_convergence

    report = behavioral_convergence(sweep_results)
    print(report["overall_convergence"])  # 0.0–1.0
    for m in report["per_metric"]:
        print(m["metric"], m["convergence"], m["warning"])
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Metrics that are "worse-is-higher" — convergence on high values is suspicious.
_ADVERSE_METRICS = {"avg_toxicity", "toxicity_rate", "n_frozen", "infiltration_rate"}

# Metrics analyzed by default.
_DEFAULT_METRICS = (
    "avg_toxicity",
    "total_welfare",
    "avg_quality_gap",
    "avg_payoff",
    "honest_avg_payoff",
    "adversarial_avg_payoff",
    "n_frozen",
    "infiltration_rate",
    "separation_quality",
)


@dataclass
class MetricConvergence:
    """Convergence result for a single metric."""

    metric: str
    convergence: float  # 0 = divergent, 1 = identical across configs
    mean: float
    std: float
    config_means: Dict[str, float]
    warning: str  # empty string if no concern


@dataclass
class ConvergenceReport:
    """Full convergence report across all analyzed metrics."""

    overall_convergence: float
    n_configs: int
    n_runs: int
    per_metric: List[MetricConvergence]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_convergence": self.overall_convergence,
            "n_configs": self.n_configs,
            "n_runs": self.n_runs,
            "per_metric": [
                {
                    "metric": m.metric,
                    "convergence": m.convergence,
                    "mean": m.mean,
                    "std": m.std,
                    "warning": m.warning,
                }
                for m in self.per_metric
            ],
            "warnings": self.warnings,
        }


def _config_key(result: Any) -> str:
    """Derive a hashable key from a SweepResult's params dict."""
    return str(sorted(result.params.items()))


def behavioral_convergence(
    results: Sequence[Any],
    metrics: Optional[Sequence[str]] = None,
    convergence_threshold: float = 0.9,
) -> ConvergenceReport:
    """Detect when different parameter configurations produce similar behavior.

    Groups *results* by their parameter combination, computes the
    cross-configuration coefficient of variation (CV) for each metric, and
    converts it to a convergence score (``1 - normalized_cv``).

    A convergence score near 1.0 means all configurations produce the
    same value for that metric — they've found the same attractor.

    Args:
        results: List of :class:`SweepResult` objects (from a completed sweep).
        metrics: Metrics to analyze.  Defaults to a standard set.
        convergence_threshold: Score above which a metric is flagged as
            converged.  If the metric is in :data:`_ADVERSE_METRICS` and
            the mean is high, a warning is emitted.

    Returns:
        A :class:`ConvergenceReport` with per-metric scores and warnings.
    """
    if metrics is None:
        metrics = list(_DEFAULT_METRICS)

    # Group results by config
    groups: Dict[str, List[Any]] = defaultdict(list)
    for r in results:
        groups[_config_key(r)].append(r)

    n_configs = len(groups)
    if n_configs < 2:
        return ConvergenceReport(
            overall_convergence=1.0,
            n_configs=n_configs,
            n_runs=len(results),
            per_metric=[],
            warnings=["Need at least 2 distinct configs for convergence analysis."],
        )

    # For each metric, compute the mean value per config, then measure
    # how much those means vary across configs.
    per_metric: List[MetricConvergence] = []
    warnings: List[str] = []

    for metric in metrics:
        config_means: Dict[str, float] = {}
        all_values: List[float] = []

        for config_key, config_results in groups.items():
            values = [_get_metric(r, metric) for r in config_results]
            values = [v for v in values if v is not None]
            if not values:
                continue
            mean_val = float(np.mean(values))
            config_means[config_key] = mean_val
            all_values.extend(values)

        if len(config_means) < 2 or not all_values:
            continue

        cross_config_values = np.array(list(config_means.values()))
        global_mean = float(np.mean(cross_config_values))
        cross_config_std = float(np.std(cross_config_values))

        # Convergence = 1 - CV, clamped to [0, 1].
        # When mean ≈ 0, use absolute std as the divergence signal.
        if abs(global_mean) > 1e-10:
            cv = cross_config_std / abs(global_mean)
        else:
            cv = cross_config_std  # absolute spread when mean is ~0

        convergence = float(np.clip(1.0 - cv, 0.0, 1.0))

        # Check for suspicious convergence
        warning = ""
        if convergence >= convergence_threshold:
            if metric in _ADVERSE_METRICS and global_mean > 0.1:
                warning = (
                    f"All configs converge on high {metric} "
                    f"(mean={global_mean:.3f}) — possible proxy gaming."
                )
                warnings.append(warning)
            elif metric not in _ADVERSE_METRICS and convergence >= 0.99:
                warning = (
                    f"{metric} is identical across configs — check if "
                    f"the parameter actually affects this metric."
                )

        per_metric.append(MetricConvergence(
            metric=metric,
            convergence=convergence,
            mean=global_mean,
            std=cross_config_std,
            config_means=config_means,
            warning=warning,
        ))

    # Overall convergence = mean of per-metric convergence scores
    if per_metric:
        overall = float(np.mean([m.convergence for m in per_metric]))
    else:
        overall = 0.0

    return ConvergenceReport(
        overall_convergence=overall,
        n_configs=n_configs,
        n_runs=len(results),
        per_metric=per_metric,
        warnings=warnings,
    )


def _get_metric(result: Any, metric: str) -> Optional[float]:
    """Extract a metric value from a SweepResult, tolerating missing fields."""
    # Try direct attribute access
    val = getattr(result, metric, None)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    # Try params dict (for swept parameters)
    if hasattr(result, "params") and metric in result.params:
        try:
            return float(result.params[metric])
        except (TypeError, ValueError):
            return None
    # Try to_dict fallback
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        if metric in d:
            try:
                return float(d[metric])
            except (TypeError, ValueError):
                return None
    return None

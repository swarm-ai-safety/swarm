"""Drives the matched soft-vs-binary detection experiment across base rates and seeds.

Produces tidy per-(base_rate, seed, metric, variant) rows for three headline
deliverables:

1. **Detection curves** — AUROC / AUPRC + partial AUROC at low FPR of each
   detector across adversarial base rates (the ROC/PR curves themselves are
   recomputed for plotting).
2. **Time-to-detection** — epochs from onset until a detector flags a degrading
   agent, at a threshold calibrated to FPR <= 0.05 on benign agents.
3. **Calibration** — Brier score and ECE of the soft proxy vs its hard
   thresholded twin, on the same interaction streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence

import numpy as np

from swarm.detection.curves import (
    calibration,
    compute_curve,
    per_agent_scores,
    threshold_at_fpr,
    time_to_detection,
)
from swarm.detection.degradation import (
    AgentStream,
    PopulationConfig,
    generate_population,
)
from swarm.detection.detectors import MatchedDetectors
from swarm.detection.market import market_selection_scores


@dataclass
class ExperimentConfig:
    base_rates: Sequence[float] = (0.05, 0.1, 0.2, 0.35, 0.5)
    seeds: Sequence[int] = tuple(range(10))
    population: PopulationConfig = field(default_factory=PopulationConfig)
    tau_star: float = 0.5
    max_fpr: float = 0.05
    ttd_window: int = 4
    # Evaluation window for the curve (post-onset back half of the run).
    eval_start: int | None = None  # default: n_epochs // 2
    eval_end: int | None = None  # default: n_epochs


@dataclass
class ExperimentResults:
    detection_rows: List[dict] = field(default_factory=list)
    ttd_rows: List[dict] = field(default_factory=list)
    calibration_rows: List[dict] = field(default_factory=list)
    # Market-level adverse-selection rows (quality_gap, conditional_loss, spread).
    market_rows: List[dict] = field(default_factory=list)
    # Kept for plotting one representative ROC/PR per (metric, variant).
    representative_curves: Dict[str, object] = field(default_factory=dict)


# Per-agent detectors (get full AUROC + TTD treatment).
# Market-level selection metrics (quality_gap, conditional_loss, spread) are
# reported separately because they require a quality mixture.
_PER_AGENT_METRICS = ("toxicity", "uncertain_fraction")


def _eval_window(cfg: ExperimentConfig) -> tuple[int, int]:
    n = cfg.population.stream.n_epochs
    start = cfg.eval_start if cfg.eval_start is not None else n // 2
    end = cfg.eval_end if cfg.eval_end is not None else n
    return start, end


def run_experiment(cfg: ExperimentConfig) -> ExperimentResults:
    detectors = MatchedDetectors(tau_star=cfg.tau_star)
    pairs = detectors.pairs()
    eval_start, eval_end = _eval_window(cfg)
    results = ExperimentResults()

    for base_rate in cfg.base_rates:
        for seed in cfg.seeds:
            pop = replace(cfg.population, base_rate=base_rate)
            streams = generate_population(pop, seed=seed)

            # --- per-agent detection curves (toxicity, uncertain_fraction) ---
            for metric in _PER_AGENT_METRICS:
                for variant, detector in pairs[metric].items():
                    scores, labels = per_agent_scores(
                        streams, detector, eval_start, eval_end
                    )
                    curve = compute_curve(scores, labels)
                    results.detection_rows.append(
                        {
                            "base_rate": base_rate,
                            "seed": seed,
                            "metric": metric,
                            "variant": variant,
                            "auroc": curve.auroc,
                            "auprc": curve.auprc,
                            "pauroc_fpr05": curve.pauroc_fpr05,
                            "pauroc_fpr01": curve.pauroc_fpr01,
                        }
                    )
                    # Stash one representative curve (first seed, mid base rate).
                    if seed == cfg.seeds[0] and abs(base_rate - 0.2) < 1e-9:
                        results.representative_curves[f"{metric}/{variant}"] = curve

            # --- market-level adverse selection (quality_gap, conditional_loss, spread) ---
            market = market_selection_scores(streams, detectors, eval_start, eval_end)
            for metric, variant_vals in market.items():
                for variant, value in variant_vals.items():
                    results.market_rows.append(
                        {
                            "base_rate": base_rate,
                            "seed": seed,
                            "metric": metric,
                            "variant": variant,
                            "value": value,
                        }
                    )

            # --- time-to-detection (threshold calibrated to FPR<=max_fpr) ---
            benign = [s for s in streams if not s.is_degrading]
            degrading = [s for s in streams if s.is_degrading]
            # Calibrate the FPR threshold and scan for detections over the SAME
            # epoch window family (from the first full trailing window onward),
            # so the reported FPR<=max_fpr operating point holds where we scan.
            ttd_min_epoch = cfg.ttd_window
            for metric in _PER_AGENT_METRICS:
                for variant, detector in pairs[metric].items():
                    thr = _calibrate_ttd_threshold(
                        benign, detector, cfg.ttd_window, cfg.max_fpr, ttd_min_epoch
                    )
                    ttds: List[int] = []
                    n_flagged = 0
                    for s in degrading:
                        ttd = time_to_detection(
                            s, detector, thr,
                            window=cfg.ttd_window, min_epoch=ttd_min_epoch,
                        )
                        if ttd is not None:
                            ttds.append(ttd)
                            n_flagged += 1
                    results.ttd_rows.append(
                        {
                            "base_rate": base_rate,
                            "seed": seed,
                            "metric": metric,
                            "variant": variant,
                            "median_ttd": float(np.median(ttds)) if ttds else None,
                            "mean_ttd": float(np.mean(ttds)) if ttds else None,
                            "detection_rate": (
                                n_flagged / len(degrading) if degrading else 0.0
                            ),
                            "n_degrading": len(degrading),
                            "n_flagged": n_flagged,
                        }
                    )

            # --- calibration (proxy vs thresholded proxy) ---
            pooled = []
            for s in streams:
                pooled.extend(s.window(eval_start, eval_end))
            cal = calibration(pooled, tau_star=cfg.tau_star)
            results.calibration_rows.append(
                {
                    "base_rate": base_rate,
                    "seed": seed,
                    "soft_brier": cal.soft_brier,
                    "binary_brier": cal.binary_brier,
                    "soft_ece": cal.soft_ece,
                    "binary_ece": cal.binary_ece,
                }
            )
    return results


def _calibrate_ttd_threshold(
    benign: Sequence[AgentStream],
    detector,
    window: int,
    max_fpr: float,
    min_epoch: int,
) -> float:
    """Pool trailing-window scores over benign agents and take the FPR quantile.

    A benign agent contributes one score per epoch (its trailing-window score);
    the threshold is set so at most ``max_fpr`` of those benign windows exceed it.
    """
    neg_scores: List[float] = []
    for s in benign:
        for e in range(max(1, min_epoch), len(s.epochs)):
            win = s.window(e - window + 1, e + 1)
            if win:
                neg_scores.append(detector(win))
    return threshold_at_fpr(neg_scores, max_fpr=max_fpr)


# ----------------------------------------------------------------------
# Aggregation into headline tables
# ----------------------------------------------------------------------
def aggregate(rows: Sequence[dict], group_keys: Sequence[str], value_keys: Sequence[str]) -> List[dict]:
    """Mean/std aggregation of ``value_keys`` over ``group_keys`` (NaN-aware)."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    out: List[dict] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        agg = dict(zip(group_keys, key, strict=True))
        agg["n"] = len(items)
        for v in value_keys:
            vals = np.array(
                [it[v] for it in items if it.get(v) is not None], dtype=float
            )
            agg[f"{v}_mean"] = float(np.mean(vals)) if vals.size else None
            agg[f"{v}_std"] = float(np.std(vals)) if vals.size else None
        out.append(agg)
    return out

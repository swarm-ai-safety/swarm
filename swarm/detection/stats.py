"""Paired soft-vs-binary significance testing for the detection experiment.

Soft and binary detectors are evaluated on the *same* interaction streams, so
their per-seed scores are **paired**. This module computes, for each headline
comparison, the mean difference, a Wilcoxon signed-rank test (with a paired
t-test as a parametric companion), Cohen's d_z effect size, and a
Holm-Bonferroni correction across the whole family. It is the built-in form of
``/full_study --detection`` Phase 2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as st


@dataclass
class PairedComparison:
    label: str
    n: int
    mean_diff: float
    cohen_dz: float
    t_p: float
    wilcoxon_p: float
    holm_threshold: float = float("nan")
    survives_holm: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean_diff": self.mean_diff,
            "cohen_dz": self.cohen_dz,
            "t_p": self.t_p,
            "wilcoxon_p": self.wilcoxon_p,
            "holm_threshold": self.holm_threshold,
            "survives_holm": self.survives_holm,
        }

    @property
    def primary_p(self) -> float:
        """Wilcoxon p, falling back to the paired-t p when Wilcoxon is undefined."""
        return self.t_p if np.isnan(self.wilcoxon_p) else self.wilcoxon_p


def paired_comparison(
    treatment: Sequence[float], control: Sequence[float], label: str
) -> PairedComparison:
    """Compare two paired samples (``treatment - control``).

    ``treatment`` is oriented so that a *positive* mean difference is the
    "expected" direction (e.g. soft AUROC − binary AUROC, or binary TTD − soft
    TTD so positive means soft is faster). Pairs with NaN on either side are
    dropped.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    mask = ~(np.isnan(t) | np.isnan(c))
    t, c = t[mask], c[mask]
    diff = t - c
    n = int(diff.size)
    mean_d = float(diff.mean()) if n else 0.0

    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        dz = mean_d / sd
        t_p = float(st.ttest_rel(t, c).pvalue)
    else:
        dz = float("inf") if mean_d != 0 else 0.0
        t_p = 0.0 if mean_d != 0 else 1.0

    if n and np.any(diff != 0):
        try:
            w_p = float(st.wilcoxon(t, c).pvalue)
        except ValueError:
            w_p = float("nan")
    else:
        w_p = 1.0

    return PairedComparison(
        label=label, n=n, mean_diff=mean_d, cohen_dz=dz, t_p=t_p, wilcoxon_p=w_p
    )


def _holm_bonferroni(comparisons: List[PairedComparison], alpha: float = 0.05) -> int:
    """Annotate ``survives_holm``/``holm_threshold`` in place; return survivor count.

    Standard step-down Holm: sort ascending by p, reject while
    ``p_(k) <= alpha / (m - k)``, stop at the first failure.
    """
    m = len(comparisons)
    order = sorted(range(m), key=lambda i: comparisons[i].primary_p)
    still_rejecting = True
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        comparisons[idx].holm_threshold = thresh
        if still_rejecting and comparisons[idx].primary_p <= thresh:
            comparisons[idx].survives_holm = True
        else:
            still_rejecting = False
            comparisons[idx].survives_holm = False
    return sum(c.survives_holm for c in comparisons)


def _by_seed(rows: Sequence[dict], key: str, group: Sequence[str]) -> Dict[tuple, dict]:
    """Index ``rows`` as ``{group_key: {variant: {seed: value}}}`` for pairing."""
    out: Dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        gk = tuple(r[g] for g in group)
        out[gk][r["variant"]][r["seed"]] = r[key]
    return out


def _aligned(variant_map: dict) -> tuple[list, list]:
    """Return (soft, binary) value lists aligned on shared seeds."""
    seeds = sorted(set(variant_map.get("soft", {})) & set(variant_map.get("binary", {})))
    soft = [variant_map["soft"][s] for s in seeds]
    binary = [variant_map["binary"][s] for s in seeds]
    return soft, binary


def compute_paired_stats(results, alpha: float = 0.05) -> dict:
    """Build the full paired-comparison family from an ``ExperimentResults``.

    Returns a JSON-ready dict with the method description, family size, survivor
    count, and per-comparison records (matching ``/full_study`` Phase 2).
    """
    comps: List[PairedComparison] = []

    # AUROC (soft - binary) per base rate.
    auroc = _by_seed(results.detection_rows, "auroc", ["base_rate"])
    for (br,), vmap in sorted(auroc.items()):
        soft, binary = _aligned(vmap)
        comps.append(paired_comparison(soft, binary, f"AUROC soft-binary @ br={br:.2f}"))

    # Time-to-detection (binary - soft; positive => soft faster), pooled over base rates.
    ttd = _by_seed(results.ttd_rows, "median_ttd", ["base_rate"])
    dr = _by_seed(results.ttd_rows, "detection_rate", ["base_rate"])
    soft_ttd, bin_ttd, soft_dr, bin_dr = [], [], [], []
    for vmap in ttd.values():
        s, b = _aligned(vmap)
        soft_ttd.extend(s)
        bin_ttd.extend(b)
    for vmap in dr.values():
        s, b = _aligned(vmap)
        soft_dr.extend(s)
        bin_dr.extend(b)
    # median_ttd can be None (censored); coerce to NaN so pairs drop cleanly.
    soft_ttd = [np.nan if v is None else v for v in soft_ttd]
    bin_ttd = [np.nan if v is None else v for v in bin_ttd]
    comps.append(paired_comparison(bin_ttd, soft_ttd, "TTD binary-soft (epochs; +=soft faster)"))
    comps.append(
        paired_comparison(soft_dr, bin_dr, "Detection-rate soft-binary (+=soft catches more)")
    )

    # Market |signal| (soft vs binary) per metric, pooled over base rates.
    mkt = _by_seed(results.market_rows, "value", ["metric", "base_rate"])
    per_metric: Dict[str, dict] = defaultdict(lambda: {"soft": [], "binary": []})
    for (metric, _br), vmap in mkt.items():
        s, b = _aligned(vmap)
        per_metric[metric]["soft"].extend(abs(v) for v in s)
        per_metric[metric]["binary"].extend(abs(v) for v in b)
    for metric in sorted(per_metric):
        comps.append(
            paired_comparison(
                per_metric[metric]["soft"],
                per_metric[metric]["binary"],
                f"Market |signal| soft-binary: {metric}",
            )
        )

    # Calibration (binary - soft; positive => soft better, since lower is better).
    for field, lbl in (("brier", "Brier"), ("ece", "ECE")):
        soft = [r[f"soft_{field}"] for r in results.calibration_rows]
        binary = [r[f"binary_{field}"] for r in results.calibration_rows]
        comps.append(paired_comparison(binary, soft, f"{lbl} binary-soft (+=soft better)"))

    n_survive = _holm_bonferroni(comps, alpha=alpha)
    return {
        "method": (
            "paired (soft vs binary on identical streams); Wilcoxon signed-rank "
            "+ paired t; Cohen's d_z; Holm-Bonferroni"
        ),
        "alpha": alpha,
        "family_size": len(comps),
        "n_survive_holm": n_survive,
        "comparisons": [c.to_dict() for c in comps],
    }

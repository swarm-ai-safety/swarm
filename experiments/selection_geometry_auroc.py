"""AUROC study for projection-geometric metrics.

Question: across many randomly-composed populations and governor
strictnesses, which of {toxicity, quality_gap, quality_correlation,
selection_saturation} best discriminates an *adversarial* population
from a *benign* one, and how stable is that AUROC across the governor
setting?

Hypothesis: raw toxicity and quality_gap conflate the signal with
the acceptance rate (governor strictness); the projection-normalized
metrics (quality_correlation = ρ(p,a), selection_saturation = |Q| · √(αβ) / σ_p)
should be more stable across θ.

Procedure:
  1. For adversarial fraction f ∈ FRACTIONS, generate K populations of
     size N by mixing benign + toxic batches from the test fixtures.
  2. For each governor θ ∈ THETAS, re-label acceptance as v_hat > θ.
  3. Compute the five SoftMetrics on every (f, θ, seed) cell.
  4. Binary label = is_adversarial (f ≥ HIGH_F). AUROC per (metric, θ)
     over the metric values pooled across (LOW_F ∪ HIGH_F) cells.
  5. Drop a CSV of per-cell metrics and a PNG of AUROC vs θ.

Outputs land under runs/selection_geometry_auroc_<seed>/.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from swarm.analysis.theme import COLORS, swarm_theme  # noqa: E402
from swarm.metrics.soft_metrics import SoftMetrics  # noqa: E402
from tests.fixtures.interactions import (  # noqa: E402
    generate_benign_batch,
    generate_toxic_batch,
)

# -------------------------------------------------------------------- config
#
# Discrimination is set up to be HARD so metrics actually compete:
#   - Adjacent adversarial fractions (5pp gap) — not 0.1 vs 0.4
#   - Small batches (N=50) — finite-sample noise dominates clean signal
#   - v_hat jitter — governor isn't a perfect oracle on p
# AUROC near 1.0 here is a real win; near 0.5 is real chance.

FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
LOW_F = 0.25
HIGH_F = 0.30
THETAS = [-0.6, -0.3, 0.0, 0.3, 0.6]
N_INTERACTIONS = 50
N_SEEDS = 400
V_HAT_NOISE_SIGMA = 0.25
METRICS = [
    ("toxicity", COLORS.TOXICITY),
    ("quality_gap", "#9B59B6"),
    ("quality_correlation", COLORS.WELFARE),
    ("selection_saturation", COLORS.EVASION),
    ("baseline_harm", "#7D8590"),
]


# ----------------------------------------------------------- AUROC helper

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Wilcoxon-Mann-Whitney AUROC. labels ∈ {0, 1}."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    n_concordant = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                n_concordant += 1
            elif sp == sn:
                n_concordant += 0.5
    return n_concordant / (len(pos) * len(neg))


# -------------------------------------------------------------- cell run

@dataclass
class CellResult:
    f: float
    theta: float
    seed: int
    toxicity: float
    quality_gap: float
    quality_correlation: float
    selection_saturation: float
    baseline_harm: float
    acceptance_rate: float


def compose_population(f: float, n: int, seed: int):
    n_tox = int(round(n * f))
    n_ben = n - n_tox
    benign = generate_benign_batch(count=n_ben, seed=seed)
    toxic = generate_toxic_batch(count=n_tox, seed=seed + 7919)
    return benign + toxic


def apply_governor(interactions, theta: float, rng: np.random.Generator):
    """Re-label acceptance as (v_hat + noise) > theta.

    Adding Gaussian noise to v_hat models a noisy proxy / imperfect
    governor — without it the rule v_hat > theta is a deterministic
    function of p, which makes the discrimination trivial.
    """
    for ix in interactions:
        noise = float(rng.normal(0.0, V_HAT_NOISE_SIGMA))
        ix.accepted = (ix.v_hat + noise) > theta
    return interactions


def run_cell(f: float, theta: float, seed: int) -> CellResult:
    pop = compose_population(f, N_INTERACTIONS, seed)
    rng = np.random.default_rng(seed + 31337)
    apply_governor(pop, theta, rng)
    sm = SoftMetrics()
    tox = sm.toxicity_rate(pop)
    qg = sm.quality_gap(pop)
    rho = sm.quality_correlation(pop)
    sat = sm.selection_saturation(pop)
    decomp = sm.toxicity_decomposition(pop)
    acc = sum(1 for ix in pop if ix.accepted) / len(pop)
    return CellResult(
        f=f, theta=theta, seed=seed,
        toxicity=tox, quality_gap=qg, quality_correlation=rho,
        selection_saturation=sat, baseline_harm=decomp["baseline_harm"],
        acceptance_rate=acc,
    )


# ------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("runs") / f"selection_geometry_auroc_{args.seed}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_cells = len(FRACTIONS) * len(THETAS) * args.n_seeds
    print(f"[study] running {n_cells} cells ({len(FRACTIONS)} f × {len(THETAS)} θ × {args.n_seeds} seeds)")

    rows: List[CellResult] = []
    for f in FRACTIONS:
        for theta in THETAS:
            for _k in range(args.n_seeds):
                seed = int(rng.integers(0, 2**31 - 1))
                rows.append(run_cell(f, theta, seed))
        print(f"[study]   done f={f:.2f}")

    # ----- write CSV
    csv_path = out_dir / "cells.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["f", "theta", "seed", "toxicity", "quality_gap",
             "quality_correlation", "selection_saturation",
             "baseline_harm", "acceptance_rate"]
        )
        for r in rows:
            writer.writerow([
                r.f, r.theta, r.seed, r.toxicity, r.quality_gap,
                r.quality_correlation, r.selection_saturation,
                r.baseline_harm, r.acceptance_rate,
            ])
    print(f"[study] wrote {csv_path}")

    # ----- AUROC per (metric, θ) for label = (f ≥ HIGH_F vs f ≤ LOW_F)
    auroc_table: Dict[str, Dict[float, float]] = {m: {} for m, _ in METRICS}
    for theta in THETAS:
        cells_theta = [r for r in rows if r.theta == theta and (r.f <= LOW_F or r.f >= HIGH_F)]
        labels = [1 if r.f >= HIGH_F else 0 for r in cells_theta]
        for metric, _ in METRICS:
            # Adversarial populations should have higher tox/baseline_harm,
            # and lower quality_gap/correlation. Flip sign for metrics whose
            # "adversarial" direction is *lower* so AUROC is positive-oriented.
            sign = -1.0 if metric in ("quality_gap", "quality_correlation") else 1.0
            scores = [sign * getattr(r, metric) for r in cells_theta]
            # NaN-guard.
            scores = [s if math.isfinite(s) else 0.0 for s in scores]
            auroc_table[metric][theta] = auroc(scores, labels)

    print("\n[study] AUROC (label = adversarial f ≥ {:.2f} vs benign f ≤ {:.2f})".format(
        HIGH_F, LOW_F))
    header = ["metric"] + [f"θ={t:+.2f}" for t in THETAS] + ["mean", "std"]
    print("  " + "  ".join(f"{h:>12}" for h in header))
    for metric, _ in METRICS:
        vals = [auroc_table[metric][t] for t in THETAS]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        cells = [f"{v:>12.3f}" for v in vals]
        print(f"  {metric:>12}  " + "  ".join(cells) + f"  {mean:>12.3f}  {std:>12.3f}")

    # ----- plot: AUROC vs θ for each metric
    with swarm_theme("dark"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for metric, color in METRICS:
            ys = [auroc_table[metric][t] for t in THETAS]
            ax.plot(THETAS, ys, marker="o", color=color, label=metric, linewidth=1.8)
        ax.axhline(0.5, color=COLORS.TEXT_MUTED, linestyle="--", linewidth=0.8,
                   label="chance (0.5)")
        ax.set_xlabel("Governor strictness θ (accept iff v̂ > θ)")
        ax.set_ylabel(f"AUROC (adversarial f ≥ {HIGH_F} vs benign f ≤ {LOW_F})")
        ax.set_ylim(0.4, 1.02)
        ax.set_title("Discrimination AUROC across governor settings")
        ax.legend(loc="lower left", fontsize=9)
    auroc_path = plots_dir / "auroc_vs_theta.png"
    fig.savefig(auroc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[study] wrote {auroc_path}")

    # ----- plot: metric value vs f, lines per θ (saturation specifically)
    with swarm_theme("dark"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        panels: List[Tuple[str, str, "matplotlib.axes.Axes"]] = [
            ("toxicity", COLORS.TOXICITY, axes[0, 0]),
            ("quality_correlation", COLORS.WELFARE, axes[0, 1]),
            ("selection_saturation", COLORS.EVASION, axes[1, 0]),
            ("acceptance_rate", "#7D8590", axes[1, 1]),
        ]
        theta_cmap = plt.cm.viridis(np.linspace(0.2, 0.95, len(THETAS)))
        for metric, _, ax in panels:
            for ti, theta in enumerate(THETAS):
                xs, mean, lo, hi = [], [], [], []
                for f in FRACTIONS:
                    vals = [getattr(r, metric) for r in rows
                            if r.f == f and r.theta == theta and math.isfinite(getattr(r, metric))]
                    if not vals:
                        continue
                    xs.append(f)
                    arr = np.asarray(vals)
                    mean.append(float(arr.mean()))
                    lo.append(float(arr.mean() - arr.std()))
                    hi.append(float(arr.mean() + arr.std()))
                ax.plot(xs, mean, color=theta_cmap[ti], marker="o",
                        label=f"θ={theta:+.2f}", linewidth=1.6)
                ax.fill_between(xs, lo, hi, color=theta_cmap[ti], alpha=0.12)
            ax.set_title(metric)
            ax.grid(True, alpha=0.2)
        axes[1, 0].set_xlabel("adversarial fraction f")
        axes[1, 1].set_xlabel("adversarial fraction f")
        axes[0, 0].legend(loc="lower right", fontsize=7, ncol=2)
        fig.suptitle("Metric response to adversarial fraction, per governor θ", fontsize=13, y=0.99)
        fig.tight_layout()
    panel_path = plots_dir / "metric_vs_f_per_theta.png"
    fig.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[study] wrote {panel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

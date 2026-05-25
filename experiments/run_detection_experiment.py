#!/usr/bin/env python
"""Run the matched soft-vs-binary detection experiment and emit headline artifacts.

Turns the self-optimizing-agent vignette into a real experiment: many seeds,
varied degradation trajectories and onset times, with detection curves rather
than a narrative. Writes a self-contained run folder under ``runs/``:

    runs/<ts>_detection_baselines/
      csv/{detection,ttd,market,calibration}.csv   # raw per-(base_rate,seed) rows
      csv/*_agg.csv                                 # mean/std aggregates
      plots/{roc_toxicity,auroc_vs_baserate,ttd,market_selection,calibration}.png
      summary.md                                    # headline tables
      config.json                                   # exact config for reproducibility

Usage:
    python experiments/run_detection_experiment.py            # full: 5 base rates x 10 seeds
    python experiments/run_detection_experiment.py --smoke    # quick: 3 base rates x 3 seeds
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swarm.detection import (  # noqa: E402
    ExperimentConfig,
    PopulationConfig,
    StreamConfig,
    aggregate,
    run_experiment,
)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_roc(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for key, curve in sorted(curves.items()):
        ax.plot(curve.fpr, curve.tpr, lw=2, label=f"{key} (AUROC={curve.auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC: per-agent toxicity detector (base rate = 0.20)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_auroc_vs_baserate(agg: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ("soft", "binary"):
        rows = sorted(
            [r for r in agg if r["variant"] == variant], key=lambda r: r["base_rate"]
        )
        br = [r["base_rate"] for r in rows]
        au = [r["auroc_mean"] for r in rows]
        sd = [r["auroc_std"] for r in rows]
        ax.errorbar(br, au, yerr=sd, marker="o", capsize=3, lw=2, label=variant)
    ax.set_xlabel("adversarial base rate")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-agent toxicity detector: AUROC vs base rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_ttd(agg: List[dict], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    variants = ("soft", "binary")
    med = [
        next((r["median_ttd_mean"] for r in agg if r["variant"] == v), None)
        for v in variants
    ]
    med = [0.0 if m is None else m for m in med]
    det = [
        next((r["detection_rate_mean"] for r in agg if r["variant"] == v), 0.0)
        for v in variants
    ]
    colors = ["seagreen", "indianred"]
    ax1.bar(variants, med, color=colors)
    ax1.set_ylabel("median epochs from onset")
    ax1.set_title("Time-to-detection (lower = faster)\nat FPR ≤ 0.05")
    for i, m in enumerate(med):
        ax1.text(i, m, f"{m:.2f}", ha="center", va="bottom")
    ax2.bar(variants, det, color=colors)
    ax2.set_ylabel("detection rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Fraction of degrading agents ever flagged")
    for i, d in enumerate(det):
        ax2.text(i, d, f"{d:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_market(agg: List[dict], path: Path) -> None:
    metrics = sorted({r["metric"] for r in agg})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        for variant in ("soft", "binary"):
            rows = sorted(
                [r for r in agg if r["metric"] == metric and r["variant"] == variant],
                key=lambda r: r["base_rate"],
            )
            br = [r["base_rate"] for r in rows]
            val = [r["value_mean"] for r in rows]
            sd = [r["value_std"] for r in rows]
            ax.errorbar(br, val, yerr=sd, marker="o", capsize=3, lw=2, label=variant)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("adversarial base rate")
        ax.set_ylabel("selection risk score (−gap)")
        ax.set_title(f"Market adverse selection: {metric}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_calibration(agg: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["soft", "binary"]
    brier = [agg["soft_brier_mean"], agg["binary_brier_mean"]]
    ece = [agg["soft_ece_mean"], agg["binary_ece_mean"]]
    colors = ["steelblue", "darkorange"]
    ax1.bar(labels, brier, color=colors)
    ax1.set_title("Brier score (lower = better)")
    for i, b in enumerate(brier):
        ax1.text(i, b, f"{b:.4f}", ha="center", va="bottom")
    ax2.bar(labels, ece, color=colors)
    ax2.set_title("Expected calibration error (lower = better)")
    for i, e in enumerate(ece):
        ax2.text(i, e, f"{e:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _write_summary(out: Path, cfg: ExperimentConfig, res, aggs: dict) -> None:
    lines: List[str] = ["# Detection baselines: soft vs binary", ""]
    lines.append(
        f"_{len(cfg.seeds)} seeds × {len(cfg.base_rates)} base rates, "
        f"{cfg.population.n_agents} agents/population, "
        f"{cfg.population.stream.n_epochs} epochs._"
    )
    lines.append("")
    lines.append(
        "Each soft metric is paired with its **binary analogue** — the *same* "
        "metric computed on the proxy thresholded at τ\\*=%.2f. Toxicity is a "
        "per-agent detector; quality-gap and conditional-loss are market-level "
        "adverse-selection metrics (they need a quality mixture)." % cfg.tau_star
    )
    lines.append("")

    lines.append("## 1. Per-agent toxicity detector — AUROC by base rate")
    lines.append("")
    lines.append("| base rate | soft AUROC | binary AUROC |")
    lines.append("| --- | ---: | ---: |")
    det = aggs["detection"]
    for br in sorted({r["base_rate"] for r in det}):
        s = next(r for r in det if r["base_rate"] == br and r["variant"] == "soft")
        b = next(r for r in det if r["base_rate"] == br and r["variant"] == "binary")
        lines.append(
            f"| {br:.2f} | {_fmt(s['auroc_mean'])} ± {_fmt(s['auroc_std'])} "
            f"| {_fmt(b['auroc_mean'])} ± {_fmt(b['auroc_std'])} |"
        )
    lines.append("")

    lines.append("## 2. Time-to-detection at FPR ≤ %.2f" % cfg.max_fpr)
    lines.append("")
    lines.append("| variant | median epochs from onset | detection rate |")
    lines.append("| --- | ---: | ---: |")
    for r in sorted(aggs["ttd"], key=lambda r: r["variant"]):
        lines.append(
            f"| {r['variant']} | {_fmt(r['median_ttd_mean'], 2)} "
            f"| {_fmt(r['detection_rate_mean'], 2)} |"
        )
    lines.append("")

    lines.append("## 3. Market adverse selection (risk = −gap; higher = more adverse)")
    lines.append("")
    lines.append("| metric | variant | " + " | ".join(
        f"br={br:.2f}" for br in sorted(cfg.base_rates)) + " |")
    lines.append("| --- | --- | " + " | ".join("---:" for _ in cfg.base_rates) + " |")
    mkt = aggs["market"]
    for metric in sorted({r["metric"] for r in mkt}):
        for variant in ("soft", "binary"):
            cells = []
            for br in sorted(cfg.base_rates):
                row = next(
                    (r for r in mkt if r["metric"] == metric
                     and r["variant"] == variant and r["base_rate"] == br),
                    None,
                )
                cells.append(_fmt(row["value_mean"]) if row else "n/a")
            lines.append(f"| {metric} | {variant} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 4. Proxy calibration (soft probability vs thresholded)")
    lines.append("")
    c = aggs["calibration"]
    lines.append("| | Brier | ECE |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| soft | {_fmt(c['soft_brier_mean'], 4)} | {_fmt(c['soft_ece_mean'], 4)} |")
    lines.append(f"| binary | {_fmt(c['binary_brier_mean'], 4)} | {_fmt(c['binary_ece_mean'], 4)} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="quick small-scale run")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--agents", type=int, default=40)
    args = ap.parse_args()

    if args.smoke:
        cfg = ExperimentConfig(
            base_rates=(0.05, 0.2, 0.5),
            seeds=tuple(range(3)),
            population=PopulationConfig(
                n_agents=20, stream=StreamConfig(n_epochs=16, interactions_per_epoch=8)
            ),
        )
    else:
        cfg = ExperimentConfig(
            base_rates=(0.05, 0.1, 0.2, 0.35, 0.5),
            seeds=tuple(range(args.seeds)),
            population=PopulationConfig(n_agents=args.agents, stream=StreamConfig()),
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("runs") / f"{ts}_detection_baselines"
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Running: {len(cfg.seeds)} seeds × {len(cfg.base_rates)} base rates ...")
    res = run_experiment(cfg)

    # Raw rows
    _write_csv(out / "csv" / "detection.csv", res.detection_rows)
    _write_csv(out / "csv" / "ttd.csv", res.ttd_rows)
    _write_csv(out / "csv" / "market.csv", res.market_rows)
    _write_csv(out / "csv" / "calibration.csv", res.calibration_rows)

    # Aggregates
    det_agg = aggregate(res.detection_rows, ["base_rate", "variant"], ["auroc", "auprc"])
    ttd_agg = aggregate(res.ttd_rows, ["variant"], ["median_ttd", "detection_rate"])
    mkt_agg = aggregate(res.market_rows, ["metric", "variant", "base_rate"], ["value"])
    cal_agg = aggregate(
        res.calibration_rows, [],
        ["soft_brier", "binary_brier", "soft_ece", "binary_ece"],
    )[0]
    _write_csv(out / "csv" / "detection_agg.csv", det_agg)
    _write_csv(out / "csv" / "ttd_agg.csv", ttd_agg)
    _write_csv(out / "csv" / "market_agg.csv", mkt_agg)
    _write_csv(out / "csv" / "calibration_agg.csv", [cal_agg])

    # Plots
    if res.representative_curves:
        _plot_roc(res.representative_curves, out / "plots" / "roc_toxicity.png")
    _plot_auroc_vs_baserate(det_agg, out / "plots" / "auroc_vs_baserate.png")
    _plot_ttd(ttd_agg, out / "plots" / "ttd.png")
    _plot_market(mkt_agg, out / "plots" / "market_selection.png")
    _plot_calibration(cal_agg, out / "plots" / "calibration.png")

    # Config + summary
    (out / "config.json").write_text(
        json.dumps(
            {
                "base_rates": list(cfg.base_rates),
                "n_seeds": len(cfg.seeds),
                "tau_star": cfg.tau_star,
                "max_fpr": cfg.max_fpr,
                "ttd_window": cfg.ttd_window,
                "population": {
                    "n_agents": cfg.population.n_agents,
                    "onset_choices": list(cfg.population.onset_choices),
                    "trajectory_choices": list(cfg.population.trajectory_choices),
                    "stream": asdict(cfg.population.stream),
                },
            },
            indent=2,
        )
    )
    _write_summary(
        out, cfg, res,
        {"detection": det_agg, "ttd": ttd_agg, "market": mkt_agg, "calibration": cal_agg},
    )
    print(f"Wrote {out}")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()

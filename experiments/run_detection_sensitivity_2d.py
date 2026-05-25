#!/usr/bin/env python
"""
2D sensitivity grid for the detection experiment.

Sweeps two generative parameters and produces tables + heatmaps
for AUROC, AUPRC, pAUROC, and TTD (for both toxicity and uncertain_fraction).

Usage:
    PYTHONPATH=. python experiments/run_detection_sensitivity_2d.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.detection import (
    ExperimentConfig,
    PopulationConfig,
    StreamConfig,
    aggregate,
    run_experiment,
)


def parse_values(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param1", default="proxy_noise")
    ap.add_argument("--values1", default="0.03,0.06,0.09,0.15")
    ap.add_argument("--param2", default="quality_jitter")
    ap.add_argument("--values2", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vals1 = parse_values(args.values1)
    vals2 = parse_values(args.values2)

    base_rates = (0.05, 0.2, 0.5)
    n_seeds = args.seeds
    n_agents = args.agents
    n_epochs = args.epochs

    if args.out:
        out = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / f"{ts}_2d_{args.param1}_{args.param2}"

    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Running 2D grid: {args.param1} x {args.param2}")
    print(f"Output: {out}")

    all_rows = []

    for v1 in vals1:
        for v2 in vals2:
            stream = StreamConfig(n_epochs=n_epochs, interactions_per_epoch=8)
            stream = replace(stream, **{args.param1: v1, args.param2: v2})

            cfg = ExperimentConfig(
                base_rates=base_rates,
                seeds=tuple(range(n_seeds)),
                population=PopulationConfig(n_agents=n_agents, stream=stream),
            )

            print(f"  {args.param1}={v1}, {args.param2}={v2} ...", end=" ", flush=True)
            res = run_experiment(cfg)
            print("done")

            for row in aggregate(res.detection_rows, ["base_rate", "metric", "variant"], ["auroc", "auprc", "pauroc_fpr05"]):
                r = row.copy()
                r[args.param1] = v1
                r[args.param2] = v2
                all_rows.append(r)

            for row in aggregate(res.ttd_rows, ["base_rate", "metric", "variant"], ["median_ttd", "detection_rate"]):
                r = row.copy()
                r[args.param1] = v1
                r[args.param2] = v2
                all_rows.append(r)

    # Write CSVs
    import csv
    det_rows = [r for r in all_rows if "auroc" in r]
    if det_rows:
        with open(out / "csv" / "grid_detection.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
            w.writeheader()
            w.writerows(det_rows)

    ttd_rows = [r for r in all_rows if "median_ttd" in r]
    if ttd_rows:
        with open(out / "csv" / "grid_ttd.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ttd_rows[0].keys()))
            w.writeheader()
            w.writerows(ttd_rows)

    # Simple summary (text)
    with open(out / "grid_summary.md", "w") as f:
        f.write(f"# 2D Grid: {args.param1} x {args.param2}\n\n")
        f.write("See plots/ for heatmaps.\n")

    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
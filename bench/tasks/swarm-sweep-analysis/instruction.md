# Task: Parameter Sweep Analysis

Run a parameter sweep on the baseline scenario varying `governance.transaction_tax_rate` across [0.0, 0.05, 0.10, 0.15]. Use 3 seeds per config (42, 7, 123), 5 epochs, 10 steps. Save sweep_results.csv and summary.json to /root/output/.

## Requirements

- Output directory: `/root/output/`
- Files to produce:
  - `/root/output/sweep_results.csv` — One row per (config, seed) = 12 rows total
  - `/root/output/summary.json` — Aggregated stats per config (4 entries)
- CSV must have column `transaction_tax_rate` and metrics: `welfare`, `toxicity_rate`
- summary.json must have `configs` array with `mean_welfare` per config

## Environment

- SWARM framework installed at `/root/swarm-package/`
- Scenario file: `/root/scenarios/baseline.yaml`

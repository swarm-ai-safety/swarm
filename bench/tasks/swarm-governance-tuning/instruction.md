# Task: Governance Tuning

Find the governance configuration that maximizes welfare while keeping toxicity below 0.15. Sweep transaction_tax_rate [0.0, 0.05, 0.10, 0.15, 0.20] with 5 seeds each (42, 7, 123, 99, 256), 8 epochs, 10 steps. Write /root/output/optimal.json.

## Requirements

- Run sweep: 5 tax rates x 5 seeds = 25 configurations
- Output: `/root/output/optimal.json` with keys:
  - `optimal_tax_rate` (float, must be one of the swept values)
  - `mean_welfare` (float)
  - `mean_toxicity` (float, must be < 0.15)
  - `n_configs_tested` (int, must be 5)
  - `statistical_confidence` (float, valid p-value from comparing optimal to no-tax baseline)

## Environment

- SWARM framework installed at `/root/swarm-package/`
- Scenario file: `/root/scenarios/baseline.yaml`

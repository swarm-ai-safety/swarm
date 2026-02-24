# Task: Full Statistical Analysis with Corrections

Perform a rigorous statistical analysis of /root/data/sweep_results.csv. Compute:
1. Welch's t-tests between all parameter pairs for welfare
2. Cohen's d effect sizes for each comparison
3. Bonferroni correction for multiple comparisons
4. Shapiro-Wilk normality tests for each group

Save results to /root/output/summary.json and /root/output/results.txt.

## Requirements

- Input: `/root/data/sweep_results.csv` (80 rows, 5 tax rates x 4 seeds x 4 agent types)
- Note: If the CSV has an `agent_type` column, aggregate by (transaction_tax_rate, seed) first by taking the mean welfare per (tax_rate, seed) combination
- Output files:
  - `/root/output/summary.json` with keys:
    - `total_hypotheses` (int)
    - `bonferroni_threshold` (float, = 0.05 / total_hypotheses)
    - `n_bonferroni_significant` (int)
    - `results` (array of objects with `group_a`, `group_b`, `p_value`, `cohens_d`, `bonferroni_significant`)
    - `normality_tests` (object)
  - `/root/output/results.txt` — Human-readable report (non-empty)

## Environment

- Python 3.12 with scipy, pandas, numpy
- Data pre-loaded at `/root/data/sweep_results.csv`

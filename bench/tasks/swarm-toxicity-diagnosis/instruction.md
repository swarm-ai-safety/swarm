# Task: Toxicity Diagnosis

Analyze the sweep CSV at /root/data/sweep_results.csv. Identify which governance configuration (transaction_tax_rate value) produces the highest mean toxicity and which produces the lowest. Write findings to /root/output/diagnosis.json.

## Requirements

- Input: `/root/data/sweep_results.csv`
- Output: `/root/output/diagnosis.json` with these keys:
  - `worst_config` — tax rate with highest mean toxicity (float)
  - `best_config` — tax rate with lowest mean toxicity (float)
  - `worst_toxicity` — mean toxicity of worst config (float)
  - `best_toxicity` — mean toxicity of best config (float)
  - `recommendation` — non-empty string with actionable recommendation

## Environment

- Python 3.12 with pandas, numpy
- Sweep CSV pre-loaded at `/root/data/sweep_results.csv`

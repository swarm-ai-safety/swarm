# Task: Detect Adverse Selection

Run the baseline scenario with seed 42, 10 epochs, 10 steps. Compute the quality gap: E[p|accepted] - E[p|rejected]. If quality_gap < 0, adverse selection is present. Write a report to /root/output/adverse_selection.json.

## Requirements

- Run scenario: `/root/scenarios/baseline.yaml` with seed=42, epochs=10, steps=10
- Output: `/root/output/adverse_selection.json` with keys:
  - `quality_gap_value` (float)
  - `adverse_selection_detected` (bool — true if quality_gap < 0)
  - `accepted_mean_p` (float in [0,1])
  - `rejected_mean_p` (float in [0,1])
  - `recommendation` (non-empty string)

## Environment

- SWARM framework installed at `/root/swarm-package/`
- Scenario file: `/root/scenarios/baseline.yaml`

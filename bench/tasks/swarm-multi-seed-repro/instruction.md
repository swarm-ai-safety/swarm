# Task: Multi-Seed Reproducibility Check

Run the baseline scenario with seeds 42, 7, and 123 (5 epochs, 10 steps each). Save results to /root/output/seed_42/, /root/output/seed_7/, /root/output/seed_123/. Verify that same-seed runs produce identical welfare. Write a one-line summary per seed to /root/output/summary.csv with columns: seed, welfare, toxicity_rate.

## Requirements

- Output directory: `/root/output/`
- Subdirectories: `seed_42/`, `seed_7/`, `seed_123/` — each with `history.json`
- Summary file: `/root/output/summary.csv` — 3 data rows (plus header)
- Each seed's run must be reproducible (same seed = same welfare)

## Environment

- SWARM framework installed at `/root/swarm-package/`
- Scenario file: `/root/scenarios/baseline.yaml`

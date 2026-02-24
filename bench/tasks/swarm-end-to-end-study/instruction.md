# Task: End-to-End Study

Conduct a complete study on the baseline scenario:
1. **Sweep**: Sweep tax_rate [0.0, 0.10, 0.20] with 3 seeds (42, 7, 123), 5 epochs, 10 steps
2. **Analyze**: Run statistical analysis on the sweep results
3. **Plot**: Generate visualizations
4. **Paper**: Scaffold a paper with the results

All outputs go under /root/output/ with subdirectories: sweep/, analysis/, plots/, paper/.

## Requirements

- `/root/output/sweep/sweep_results.csv` — 9 rows (3 configs x 3 seeds)
- `/root/output/analysis/summary.json` — Statistical results with `results` array
- `/root/output/plots/` — At least 2 PNG files
- `/root/output/paper/paper.md` — Paper with Results section containing data from the sweep

## Environment

- SWARM framework installed at `/root/swarm-package/`
- Scenario: `/root/scenarios/baseline.yaml`
- scipy, matplotlib, pandas, numpy available

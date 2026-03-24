# Autoresearch Objective (Example)

```yaml
primary_metric: quality_gap
primary_direction: minimize
min_improvement: 0.01
guardrails:
  - metric: toxicity_rate
    max_regression: 0.01
  - metric: total_welfare
    max_regression: 0.0
```

Use with:

```bash
python -m swarm autoresearch \
  --objective examples/program_autoresearch.md \
  --scenario scenarios/baseline.yaml \
  --iterations 5 \
  --eval-epochs 2 \
  --eval-steps 3 \
  --seeds 7,11
```

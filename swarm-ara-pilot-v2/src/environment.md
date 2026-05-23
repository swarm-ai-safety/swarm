# Environment & Execution Configuration

## Python & Dependencies

- **Python Version**: 3.10+ (standard typing, dataclasses, numpy support)
- **Core Dependencies**:
  - `numpy >= 1.24.0` (numerical operations, RNG)
  - `pydantic >= 2.0.0` (optional; typed configuration validation if desired)
  - `jsonlines >= 4.0.0` (JSONL logging for event replay)
  
- **Development Dependencies**:
  - `pytest >= 7.0` (unit testing)
  - `pytest-cov >= 4.0` (coverage reporting)
  - `mypy >= 1.0` (static type checking)
  - `ruff >= 0.1.0` (linting)

## Random Number Generation

### Seeding Protocol (H03)

All stochastic components use a fixed random seed list for reproducibility.

**Default seed list**: {42, 123, 456, 789, 1024}
- **Rationale**: Five seeds provide adequate statistical power (~25–35% reduction in error bars vs. 3 seeds, per H03) while remaining computationally tractable.
- **Justification**: Five-seed replication is a practical compromise between statistical precision and cost (Assumption A5); acknowledged limitation is that cross-seed variation is not deeply characterized.
- **Usage**: Each scenario run uses one seed from this list; metrics aggregated across all five seeds (mean ± std dev reported in results tables).

### RNG Implementation

```python
# Each scenario run initializes a random state with fixed seed
random_state = np.random.Generator(np.random.PCG64(seed))

# Sampling across the run:
# - Agent selection: random_state.choice(agent_population)
# - Audit sampling: audit_triggered = random_state.random() < p_audit
# - Audit outcome: audit_passes = random_state.random() < p
# - Observable noise (optional): random_state.normal(mean, std)
```

**Why PCG64**: Modern, efficient, cryptographically safe (not needed but good practice); supports splitting for parallel runs if needed.

## Logging & Event Replay

### JSONL Event Log Format

Each scenario run produces an append-only JSONL file with per-interaction records. Schema enables full replay (Constraints section, safety invariant):

```json
{
  "epoch": 0,
  "step": 0,
  "agent_id": "agent_42",
  "task_progress": 0.95,
  "rework_count": 1,
  "verifier_rejections": 0,
  "engagement": 0.8,
  "proxy_score_v_hat": 0.532,
  "soft_label_p": 0.631,
  "s_plus": 1.0,
  "s_minus": 0.0,
  "h": 0.5,
  "s_soft": 0.631,
  "externality_cost": 0.1845,
  "tax_cost": 0.0631,
  "reputation_delta": 0.25,
  "payoff": 0.5034,
  "accept": true,
  "audit_triggered": false,
  "timestamp": "2024-05-05T12:34:56.123Z"
}
```

**Invariant**: All p values must be in [0, 1] (Constraints, Soft Label Validity boundary condition).

### Replay Capability

Given a JSONL log + scenario config (governance params, payoff structure, seed), the interaction sequence is deterministic:
1. Read event i from JSONL
2. Reconstruct SoftInteraction object
3. Re-compute soft_label_p from proxy_score_v_hat (should match logged value)
4. Verify accept/reject decision and payoff (should match logged value)

This enables:
- Auditing governance decisions post-hoc
- Sensitivity analysis (re-run with different governance params, same observable sequence)
- Debugging (step through specific interactions)

## Experiment Harness

### ScenarioRun Configuration

Typical experiment setup (pseudocode from Algorithm section):

```python
# Initialize scenario
scenario_config = ScenarioConfig(
    agent_population=[Agent(id=f"a{i}", agent_type="honest") for i in range(50)],
    governance=GovernanceParams(tau=0.10, theta_CB=0.35, ...),
    payoff_structure=PayoffStructure(s_plus=1.0, s_minus=0.0, h=0.5),
    weights=ProxyWeights(0.4, 0.2, 0.2, 0.2),
    n_epochs=10,
    n_steps_per_epoch=100,
    seed=42
)

# Run scenario
history, metrics = run_scenario(scenario_config)

# Log results
logger.write_jsonl(history, f"runs/2024-05-05_baseline_seed42/history.jsonl")
logger.write_metrics(metrics, f"runs/2024-05-05_baseline_seed42/metrics.json")
```

### Metric Aggregation

Per scenario, results are aggregated across 5 seeds (from H03):

```python
results = {}
for seed in {42, 123, 456, 789, 1024}:
    scenario_config.seed = seed
    history, metrics = run_scenario(scenario_config)
    results[seed] = metrics

# Aggregate
aggregated = {
    'toxicity_mean': np.mean([results[s]['toxicity'] for s in results]),
    'toxicity_std': np.std([results[s]['toxicity'] for s in results]),
    ...
}
```

Results reported as `value ± std` in tables (e.g., Table 4: "0.300 ± 0.012").

## Computational Cost

**Per-interaction complexity**: O(H) where H = reputation history length (typically 10–100)
**Per-epoch complexity**: O(S·H) where S = steps per epoch (≥100)
**Per-scenario complexity**: O(N·S·H) where N = epochs (10)
  - Typical: 10 epochs × 100 steps × 10 history length = 10,000 operations per scenario
  - 5 seeds × 7 scenarios = 35 runs × 10K ops ≈ 350K operations total
  - Negligible CPU time: <1 second per scenario on modern hardware

## Storage

**Per-scenario run**:
- JSONL history: ~500 KB (10K interactions × ~50 bytes/record)
- Metrics summary: ~5 KB (JSON)
- Total: ~1 MB per run

**Full experiment** (5 seeds × 7 scenarios):
- ~35 MB uncompressed
- ~5 MB gzipped
- Stored in `runs/` directory (gitignored in main repo; archived to `swarm-artifacts`)

## Extension Points

1. **Observable noise**: Add Gaussian noise to observables (e.g., `task_progress += random_state.normal(0, 0.05)`)
2. **Agent learning**: Agents improve quality over epochs (e.g., `quality_next_epoch = f(quality_current, payoff, audit_outcome)`)
3. **Parallel execution**: Use `joblib` or `multiprocessing` to run multiple seeds/scenarios in parallel (PCG64 supports splitting for thread-safe RNG)
4. **Real agent simulation**: Replace scripted agents with LLM-backed agents (Concordia, GPT, Claude) for E04
5. **Bayesian optimization**: Use Optuna or similar to find optimal governance parameters instead of grid search

## Configuration Example

Minimal configuration to run a single scenario:

```yaml
# scenario.yaml
governance:
  tau: 0.10
  rho: 0.10
  theta_CB: 0.35
  lambda_decay: 0.85
  p_audit: 0.10

payoff_structure:
  s_plus: 1.0
  s_minus: 0.0
  h: 0.5

weights:
  task_progress: 0.40
  rework_count: 0.20
  verifier_rejections: 0.20
  engagement: 0.20

experiment:
  n_agents: 50
  n_epochs: 10
  n_steps_per_epoch: 100
  seeds: [42, 123, 456, 789, 1024]
```

Load and run:

```python
import yaml
with open("scenario.yaml") as f:
    config = yaml.safe_load(f)

# Construct objects from config
governance = GovernanceParams(**config['governance'])
payoff = PayoffStructure(**config['payoff_structure'])
weights = ProxyWeights(**config['weights'])

# Run with each seed
for seed in config['experiment']['seeds']:
    history, metrics = run_scenario(
        agents=create_agent_population(config['experiment']['n_agents']),
        governance=governance,
        payoff=payoff,
        weights=weights,
        n_epochs=config['experiment']['n_epochs'],
        n_steps_per_epoch=config['experiment']['n_steps_per_epoch'],
        seed=seed
    )
```

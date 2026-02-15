# Obfuscation Atlas Integration

**Paper**: Lindner et al. (2026). *An Atlas of Obfuscation Strategies for White-Box Lie Detectors.* FAR.AI.
**PDF**: https://www.far.ai/research/obfuscation-atlas.pdf

## Concept Mapping

| Paper Concept | Simulation Component | Notes |
|---|---|---|
| White-box lie detector probe | `ProxyComputer` (weighted signal → sigmoid → p) | Both produce P(honest) from observable signals |
| Honest policy | `HonestAgent` | Baseline truthful behavior |
| Blatant deception | `ObfuscatingAgent(strategy=blatant)` | Low quality, no signal manipulation |
| Obfuscated policy | `ObfuscatingAgent(strategy=policy_obfuscation)` | Signal inflation + rationalization metadata |
| Obfuscated activations | `ObfuscatingAgent(strategy=activation_obfuscation)` | Representation drift toward uncertainty zone |
| RL training against detector | Epoch-by-epoch interaction + adaptive strategy | Adaptive agent switches strategy on detection feedback |
| Probe AUC | `SoftMetrics.discrimination_auc()` | Area under ROC curve for proxy discrimination |
| KL divergence penalty | Reputation decay rate (governance lever) | Both constrain deviation from honest behavior |
| Audit/inspection | Random audit governance lever | Post-hoc detection mechanism |

## Key Insight

The paper's central finding is that **obfuscated policies often score *lower* on deception probes than honest policies**, creating a dangerous adverse selection problem. In our framework, this maps to:

- Policy-obfuscated agents have inflated `task_progress_delta` and suppressed `rework_count`
- The `ProxyComputer` sees these manipulated signals and produces high p values
- Despite high p, the ground truth is negative (harmful interaction)
- This is the **probe evasion** phenomenon: the proxy is fooled

## Pre-Registered Hypotheses

| ID | Hypothesis | Metric | Expected |
|---|---|---|---|
| H1 | Blatant deception is caught by standard governance | `probe_evasion_rate` for blatant agents | < 0.2 |
| H2 | Policy obfuscation evades the proxy | `probe_evasion_rate` for policy_obf agents | > 0.5 |
| H3 | Higher audit penalty reduces evasion | Compare `probe_evasion_rate` across `audit_penalty_multiplier` values | Monotone decreasing |
| H4 | Strong reputation decay pushes toward honest behavior | `taxonomy_rates["honest"]` under high decay | > 0.6 |
| H5 | Activation obfuscation degrades proxy AUC below 0.7 | `representation_drift`, `detector_auc` in late epochs | AUC < 0.7 |
| H6 | Adaptive agents converge to policy obfuscation when undetected | `active_strategy` tracking over epochs | Stays `policy_obfuscation` |

## Metrics Definitions

### probe_evasion_rate
Fraction of ground-truth-negative interactions where the proxy assigns p >= threshold.
Maps to: obfuscated policies fooling the probe (Table 1).

### representation_drift
`AUC(epoch_1) - AUC(epoch_last)`. Positive = detector degradation.
Maps to: off-domain probe AUC degradation (Table 4).

### taxonomy_rates
Fraction of interactions in each of {honest, blatant, obfuscated_policy, obfuscated_activation}.
Classification via decision tree on ground_truth, p, and rationalization metadata.
Maps to: Fig 2 right panel.

### belief_shift_index
Total absolute change in acceptance thresholds across epochs.
Maps to: Fig 3 belief probing.

### detector_discriminative_power
Running AUC via `SoftMetrics.discrimination_auc()`.
Paper uses AUC > 0.7 as threshold for adequate detection.

## Scenario Design

`scenarios/obfuscation_atlas.yaml`:
- 12 agents: 4 honest + 2 blatant + 3 policy_obf + 2 activation_obf + 1 adaptive
- 30 epochs, small-world network (k=4, p=0.2)
- Governance: audits (15%, 3x penalty), reputation decay (0.85), circuit breaker, collusion detection
- Higher externality cost (h=2.5) and externality internalization (rho=0.1)

## Reproduction

```bash
python -m swarm run scenarios/obfuscation_atlas.yaml --seed 42 --epochs 30 --steps 10
```

## Implementation Files

| File | Role |
|---|---|
| `swarm/agents/obfuscating.py` | `ObfuscatingAgent` + `ObfuscationStrategy` enum |
| `swarm/core/observable_generator.py` | `ObfuscationObservableGenerator` wrapper |
| `swarm/metrics/obfuscation_metrics.py` | `ObfuscationMetrics`, `ObfuscationCategory`, `ObfuscationSummary` |
| `swarm/metrics/reporters.py` | Extended `MetricsSummary` + `summary_with_obfuscation()` |
| `tests/fixtures/interactions.py` | `generate_obfuscation_scenario()` fixture |
| `tests/test_obfuscation.py` | ~25 deterministic tests |
| `scenarios/obfuscation_atlas.yaml` | Full scenario configuration |

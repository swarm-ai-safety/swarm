# Incoherence Governance Transferability Notes

This note documents where replay-calibrated incoherence governance results are
likely to transfer and where assumptions can break in deployment settings.

## Scope

Interventions covered:
- self-ensemble
- incoherence breaker
- decomposition checkpoints
- incoherence friction

The discussion focuses on transfer from sandbox/replay evaluation to real
policy workflows.

## Intervention-by-Intervention Transferability

### 1) Self-Ensemble

Expected benefit:
- Lowers variance in probabilistic quality estimates by averaging multiple
  model passes.

Transfer assumptions:
- Replay availability: sufficient historical traces exist to tune ensemble size
  and estimate calibration/latency tradeoffs.
- Reversibility: activating or deactivating ensemble at runtime should not
  permanently alter downstream state.
- Observability: outcome labels (or strong proxies) are available often enough
  to detect drift in ensemble calibration.

Failure risk when assumptions fail:
- Without replay coverage for new task mixes, ensemble can reduce apparent
  variance while preserving systematic bias.
- In high-latency systems, added inference cost can create operational lag that
  is not represented in offline replay.

### 2) Incoherence Breaker

Expected benefit:
- Temporarily freezes or redirects agents when incoherence risk crosses a
  threshold, reducing acute failure cascades.

Transfer assumptions:
- Replay availability: enough high-risk episodes are captured to estimate true
  positive/false positive rates.
- Reversibility: freeze actions are operationally reversible and can be rolled
  back with bounded side effects.
- Observability: monitoring captures trigger context and post-trigger recovery.

Failure risk when assumptions fail:
- Sparse replay of rare incidents leads to unstable threshold tuning.
- Irreversible interventions can convert false positives into durable service
  degradation.

### 3) Decomposition Checkpoints

Expected benefit:
- Splits complex tasks into auditable steps where incoherence can be measured
  before commitment.

Transfer assumptions:
- Replay availability: step-level traces exist, not only final outcomes.
- Reversibility: checkpoint failures can be retried or rerouted without
  corrupting shared state.
- Observability: intermediate checkpoint outcomes are logged consistently.

Failure risk when assumptions fail:
- If only end-state labels exist, checkpoint gating may overfit local proxies.
- Human/operator bypass paths can invalidate replay-derived checkpoint efficacy.

### 4) Incoherence Friction

Expected benefit:
- Adds proportional cost/rate limits when forecaster risk is elevated, slowing
  risky interaction patterns.

Transfer assumptions:
- Replay availability: counterfactual or historical data can approximate
  behavioral response to friction levels.
- Reversibility: friction parameters can be lowered quickly after false alarms.
- Observability: throughput, welfare, and risk are jointly tracked so policy
  side effects are visible.

Failure risk when assumptions fail:
- Friction can suppress both harmful and beneficial activity, with net effect
  depending on context distribution shift.
- If observability is weak, welfare loss can accumulate before policy rollback.

## Cross-Cutting Caveats

- Delayed or noisy ground truth:
  Replay estimates of disagreement/error can look stable even when deployed
  labels arrive late, are censored, or are weak proxies. Conservative
  thresholds and periodic recalibration are required.
- Replay representativeness:
  Offline traces under-sample novel attack strategies and extreme tail events.
  Report confidence intervals and treat gains as conditional on distribution
  similarity.
- Policy coupling:
  Combining multiple levers (ensemble + breaker + friction) introduces
  interaction effects not identifiable from single-intervention replay.
- Operational fallback:
  Every intervention should include a documented rollback path and a max dwell
  time in restrictive modes.

## Recommended Reporting Addendum

For each intervention release, publish:
- replay coverage statistics (tasks, agents, incident classes)
- reversibility guarantees and rollback latency
- observability quality (label delay, missingness, proxy validity)
- sensitivity analyses under delayed/noisy labels

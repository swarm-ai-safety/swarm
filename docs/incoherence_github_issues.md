# Incoherence Scaling Plan - GitHub Issue Set

Use one issue per section below. Labels suggested: `incoherence`, `research`, `metrics`, `governance`, `analysis`.

## 1. Define Incoherence Metric Contract (Benchmark Action + Error Semantics)
**Summary**
Lock the error benchmark semantics used for incoherence index `I = D / (E + eps)` before implementation.

**Scope**
- Define benchmark action policy by task family.
- Define abstain/tie handling.
- Define fallback heuristic when no oracle is available.
- Document edge cases and versioning policy.

**Files**
- `docs/incoherence_metric_contract.md` (new)
- `swarm/metrics/incoherence.py` (new, interface + stubs)
- `tests/test_incoherence_metrics.py` (new, contract tests)

**Acceptance Criteria**
- Metric contract doc exists and is reviewed.
- Benchmark semantics are deterministic and test-covered.
- `I` behavior is defined for `E=0` and sparse-action cases.

**Checklist**
- [ ] Add contract document
- [ ] Add test fixtures for each task family
- [ ] Add API stubs for benchmark lookup

---

## 2. Build Replay Infrastructure (`ReplayRunner` + `EpisodeSpec`)
**Summary**
Implement K-replay execution for fixed scenarios with controlled randomness.

**Scope**
- Add `EpisodeSpec` dataclass.
- Add `ReplayRunner` that executes K runs with seed variation.
- Collect per-step actions, per-episode outcomes, and agent payoff sequences.

**Files**
- `swarm/replay/episode_spec.py` (new)
- `swarm/replay/runner.py` (new)
- `swarm/core/orchestrator.py` (replay metadata support)
- `tests/test_replay_runner.py` (new)

**Acceptance Criteria**
- `ReplayRunner` runs `K>=1` replays with reproducible seed schedule.
- Outputs are grouped by episode spec and replay index.
- Tests verify deterministic replay under fixed seed.

**Checklist**
- [ ] Add `EpisodeSpec`
- [ ] Add replay runner API
- [ ] Add deterministic seed progression tests

---

## 3. Implement Incoherence Metrics and Reporter Integration
**Summary**
Compute disagreement `D`, error `E`, and incoherence index `I` per agent/type/system and expose in reporting.

**Scope**
- Per-step action distribution and entropy/variance disagreement.
- Error against benchmark policy.
- Aggregate by agent, task family, and global system.
- Add columns to metrics summaries.

**Files**
- `swarm/metrics/incoherence.py` (new)
- `swarm/metrics/reporters.py` (extend summary)
- `tests/test_incoherence_metrics.py`
- `tests/test_metrics.py`

**Acceptance Criteria**
- Deterministic agents yield `I=0`.
- Uniform-random baseline yields high `I` (bounded threshold in tests).
- Reporter emits incoherence fields without breaking existing output.

**Checklist**
- [ ] Implement `D`, `E`, `I`
- [ ] Wire into summary objects
- [ ] Add deterministic/random property tests

---

## 4. Extend Event Schema for Replay + Feature Logging
**Summary**
Add replay and incoherence feature fields to event logs with backward-compatible parsing.

**Scope**
- Add `replay_k`, `seed`, and optional action distribution payload fields.
- Add incoherence feature payload block.
- Preserve existing event log replay behavior.

**Files**
- `swarm/models/events.py`
- `swarm/logging/event_log.py`
- `swarm/core/orchestrator.py`
- `tests/test_event_log.py`
- `tests/test_orchestrator.py`

**Acceptance Criteria**
- Old logs remain readable.
- New fields are present in action/payoff related events when enabled.
- Event round-trip tests pass.

**Checklist**
- [ ] Extend schema dataclasses/factories
- [ ] Update emit points in orchestrator
- [ ] Add backward compatibility tests

---

## 5. Add Horizon/Branching/Noise Stress Controls
**Summary**
Implement stress-test knobs needed for hot-mess scaling experiments.

**Scope**
- Use `steps_per_epoch` for horizon tiers.
- Add branching controls via agent-count scenario configs.
- Add observation-noise parameter in observation pipeline.

**Files**
- `swarm/core/orchestrator.py`
- `swarm/scenarios/loader.py`
- `scenarios/incoherence/` (new YAML set)
- `tests/test_scenarios.py`
- `tests/test_orchestrator.py`

**Acceptance Criteria**
- Scenario configs sweep short/medium/long horizons.
- Noise injection is seed-reproducible.
- Branching tiers run with stable config parsing.

**Checklist**
- [ ] Add new sim config fields
- [ ] Parse and validate in scenario loader
- [ ] Create tiered scenario YAML files

---

## 6. Generate Scaling Curve Experiments and Artifacts
**Summary**
Run Experiment A and generate `I` scaling artifacts vs horizon and branching.

**Scope**
- Create repeatable experiment runner script.
- Aggregate and plot scaling curves.
- Add short analysis doc comparing observed shape to hypothesis.

**Files**
- `examples/run_incoherence_scaling.py` (new)
- `swarm/analysis/aggregation.py`
- `swarm/analysis/plots.py`
- `docs/analysis/incoherence_scaling.md` (new)
- `tests/test_analysis.py`
- `tests/test_sweep.py`

**Acceptance Criteria**
- Script produces CSV + plots from CLI.
- Plot outputs include both horizon and branching sweeps.
- Regression tests verify aggregation schema stability.

**Checklist**
- [ ] Add experiment runner
- [ ] Add aggregation helpers
- [ ] Add plotting functions

---

## 7. Add Agent-Type Asymmetry and Dual-Failure Metrics
**Summary**
Implement Experiment B decomposition by agent type and dual-failure-mode categorization.

**Scope**
- Type-level incoherence profiles.
- Classify incidents as coherent-adversarial vs incoherent-benign.
- Track ratio over complexity tiers.

**Files**
- `swarm/metrics/incoherence.py`
- `swarm/analysis/aggregation.py`
- `tests/test_metrics.py`

**Acceptance Criteria**
- Per-type metrics table is exported.
- Dual-failure counts and ratio are reported.
- Tests cover classification boundaries and null cases.

**Checklist**
- [ ] Add per-type aggregation
- [ ] Add incident classification helper
- [ ] Add ratio metrics and tests

---

## 8. Add Variance-Aware Governance Config and Engine Wiring
**Summary**
Extend governance config/engine with toggles and thresholds for incoherence-targeted interventions.

**Scope**
- Config fields for ensemble, incoherence breaker, decomposition, dynamic friction.
- Engine registration and execution ordering.
- Scenario parsing support.

**Files**
- `swarm/governance/config.py`
- `swarm/governance/engine.py`
- `swarm/scenarios/loader.py`
- `tests/test_governance.py`

**Acceptance Criteria**
- New fields validate correctly.
- Engine can enable/disable each lever independently.
- Existing governance behavior remains unchanged by default.

**Checklist**
- [ ] Add config fields + validation
- [ ] Add engine wiring
- [ ] Add compatibility tests

---

## 9. Implement New Governance Levers (Ensemble, Breaker, Decomposition, Friction)
**Summary**
Implement all Phase 4 levers and integrate with orchestrator hooks.

**Scope**
- `SelfEnsembleLever`
- `IncoherenceCircuitBreakerLever`
- `DecompositionLever` + checkpoint protocol
- `IncoherenceFrictionLever`
- Orchestrator verification checkpoint hook

**Files**
- `swarm/governance/ensemble.py` (new)
- `swarm/governance/incoherence_breaker.py` (new)
- `swarm/governance/decomposition.py` (new)
- `swarm/governance/dynamic_friction.py` (new)
- `swarm/governance/engine.py`
- `swarm/core/orchestrator.py`
- `tests/test_governance.py`
- `tests/test_integration.py`
- `tests/test_orchestrator.py`

**Acceptance Criteria**
- Each lever has unit tests and can be toggled independently.
- Combined condition runs end-to-end without regressions.
- Metrics include cost and false-positive proxies.

**Checklist**
- [ ] Implement four lever classes
- [ ] Add orchestrator checkpoint integration
- [ ] Add end-to-end governance bake-off smoke test

---

## 10. Implement Incoherence Forecaster and Adaptive Governance Loop
**Summary**
Predict high-incoherence episodes and activate governance levers adaptively.

**Scope**
- Structural feature extraction pipeline.
- Baseline model (logistic regression or equivalent).
- Adaptive lever activation pre-episode / per-epoch.
- Optional behavioral model path behind flag.

**Files**
- `swarm/forecaster/features.py` (new)
- `swarm/forecaster/model.py` (new)
- `swarm/governance/engine.py`
- `swarm/core/orchestrator.py`
- `tests/test_forecaster.py` (new)
- `tests/test_governance.py`

**Acceptance Criteria**
- Train/predict API works on replay dataset.
- Adaptive governance triggers when predicted risk exceeds threshold.
- Held-out eval reports AUC and calibration summary.

**Checklist**
- [ ] Build feature extraction
- [ ] Build model interface
- [ ] Add adaptive trigger wiring + tests

---

## 11. Dashboard + Transferability Annotation
**Summary**
Expose incoherence analytics in dashboard and publish transferability caveats for policy relevance.

**Scope**
- Add incoherence time-series/scatter panels.
- Add governance condition comparison views.
- Publish transferability annotation doc per intervention.

**Files**
- `swarm/analysis/dashboard.py`
- `swarm/analysis/streamlit_app.py`
- `docs/transferability/incoherence_governance.md` (new)
- `tests/test_dashboard.py`

**Acceptance Criteria**
- Dashboard renders incoherence panels on generated outputs.
- Transferability document covers replay, reversibility, and observability assumptions.
- Tests confirm panel rendering paths and schema compatibility.

**Checklist**
- [ ] Add dashboard components
- [ ] Add condition comparison panel
- [ ] Add transferability document


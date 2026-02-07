# High-Level Design Critique

## Strengths

### 1. Clean Separation of Concerns
The architecture properly separates computation (`ProxyComputer`, `SoftPayoffEngine`, `SoftMetrics`) from orchestration (`Orchestrator`) and persistence (`EventLog`). Each component has a single responsibility.

### 2. Excellent Dependency Injection
The Orchestrator accepts injectable computation engines (`swarm/core/orchestrator.py:157-162`), enabling testability and extensibility. You can swap proxy computation, payoff logic, or metrics without touching orchestration code.

### 3. Soft Labels Are a Strong Abstraction
Using `p ∈ [0,1]` instead of binary labels elegantly captures epistemic uncertainty. This enables proper probabilistic metrics (Brier score, ECE, calibration curves) and reveals information loss from thresholding.

### 4. Configuration Objects (Strategy Pattern)
`PayoffConfig`, `ProxyWeights`, `OrchestratorConfig` etc. make parameter sweeps and experiments easy. Pydantic validation catches misconfigurations early.

---

## Design Concerns

### 1. Orchestrator God Object
The Orchestrator at ~1900 lines does too much:
- Agent scheduling
- Action execution
- Payoff computation
- Governance enforcement
- Marketplace handling
- Moltipedia/Moltbook domains
- Boundary enforcement
- Async coordination
- Red-team support

This violates single responsibility.

**Recommendation**: Extract domain handlers further. While there are already `MarketplaceHandler`, `MoltipediaHandler`, etc., the Orchestrator still contains significant domain logic inline. Consider a cleaner plugin architecture where domain modules register themselves.

### 2. Mixed Dataclass/Pydantic Patterns
`SoftInteraction` uses `@dataclass` while `PayoffConfig` uses `pydantic.BaseModel`. This inconsistency creates friction:
- Dataclasses don't validate
- Serialization patterns differ (`to_dict()` vs `model_dump()`)

**Recommendation**: Standardize on Pydantic BaseModel with `frozen=True` for immutable data, or use attrs consistently.

### 3. ProxyComputer Weight Semantics Are Confusing
The weights are named `task_progress`, `rework_penalty`, `verifier_penalty`, `engagement_signal` but `swarm/core/proxy.py:178` averages rejection and misuse into a single "verifier_signal". The name `verifier_penalty` is actually used for both rejection *and* misuse signals. This mismatch between naming and behavior will cause bugs when someone tries to tune weights.

### 4. Observable Generator Abstraction Is Underused
You have a nice `ObservableGenerator` interface but the `DefaultObservableGenerator` is trivial. The `_generate_observables` method in Orchestrator (`swarm/core/orchestrator.py:1086-1096`) is marked as "kept for backwards compatibility with subclasses that override this." This suggests the abstraction boundary isn't fully resolved.

### 5. Circular Dependency Risk in Metrics
`SoftMetrics` depends on `SoftPayoffEngine` (`swarm/metrics/soft_metrics.py:21-28`). This means you can't compute metrics without a payoff engine, even for pure quality metrics like `average_quality()` that don't need payoffs. Consider splitting quality-only metrics from payoff-dependent metrics.

### 6. Event Log Coupling
Every component that emits events needs `self._emit_event()`. Handlers like `MoltipediaHandler` receive `emit_event` callbacks in constructors. This tightly couples logging to orchestration. Consider an event bus pattern for looser coupling.

### 7. Missing Type Safety in Metadata
`SoftInteraction.metadata: dict` and `Action.metadata: dict` are stringly-typed bags. This makes it hard to know what keys are expected where. Consider typed protocols or at minimum, documented key schemas.

---

## Architectural Questions

### 1. Why separate `v_hat` and `p`?
The proxy score `v_hat ∈ [-1, 1]` is only used to compute `p`. Why store it on `SoftInteraction`? If it's for debugging/analysis, that's fine, but the architecture doesn't make this intent clear.

### 2. Is the sigmoid calibration justified?
`p = sigmoid(k * v_hat)` with default `k=2.0` is arbitrary. The calibration metrics exist but there's no evidence the sigmoid parameters were tuned against real data. For a research framework studying *miscalibration*, the default calibration should probably be identity or explicitly random.

### 3. Reputation delta formula is unclear
`swarm/core/orchestrator.py:1136`: `rep_delta = (interaction.p - 0.5) - interaction.c_a` mixes quality signal with governance cost. Why subtract `c_a`? This couples reputation to governance in an implicit way that isn't documented in the payoff equations.

---

## Summary

This is a well-architected research framework with thoughtful abstractions for probabilistic safety analysis. The main issues are:

1. **Orchestrator needs decomposition** - Extract more logic into handlers/plugins
2. **Inconsistent data modeling patterns** - Standardize on Pydantic or dataclasses
3. **Some abstraction boundaries need cleanup** - Observable generator, weight semantics
4. **Documentation of implicit design decisions** - Especially the reputation formula

The core insight—using soft probabilistic labels to measure adverse selection and quality gaps—is sound and well-implemented.

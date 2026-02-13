# High-Level Design Critique

*Last updated: 2026-02-13*

## Strengths

### 1. Clean Separation of Concerns
The architecture properly separates computation (`ProxyComputer`, `SoftPayoffEngine`, `SoftMetrics`) from orchestration (`Orchestrator`) and persistence (`EventLog`). Each component has a single responsibility.

### 2. Excellent Dependency Injection
The Orchestrator accepts injectable computation engines (`swarm/core/orchestrator.py:157-162`), enabling testability and extensibility. You can swap proxy computation, payoff logic, or metrics without touching orchestration code.

### 3. Soft Labels Are a Strong Abstraction
Using `p ∈ [0,1]` instead of binary labels elegantly captures epistemic uncertainty. This enables proper probabilistic metrics (Brier score, ECE, calibration curves) and reveals information loss from thresholding.

### 4. Configuration Objects (Strategy Pattern)
`PayoffConfig`, `ProxyWeights`, `OrchestratorConfig` etc. make parameter sweeps and experiments easy. Pydantic validation catches misconfigurations early.

### 5. Handler/Plugin Architecture *(new)*
The `HandlerRegistry` provides clean plugin-style dispatch. Domain handlers (`MarketplaceHandler`, `MoltipediaHandler`, `MoltbookHandler`, `ScholarHandler`, `KernelOracleHandler`, `MemoryHandler`) register their owned action types, contribute observation fields, and hook into epoch/step lifecycle events. This is a significant improvement over the original monolithic design.

### 6. Boundary Enforcement Infrastructure *(new)*
The `swarm/boundaries/` subsystem (`FlowTracker`, `LeakageDetector`, `PolicyEngine`, permeability controls) provides principled information-flow enforcement. This enables security-instrumented sandboxing across all bridge integrations.

### 7. Multi-Domain Bridge Ecosystem *(new)*
The `swarm/bridges/` layer connects the core framework to concrete execution environments: `WorktreeBridge` (git worktree sandboxes), `GasTownBridge` (beads + git workspaces), `ClaudeCodeBridge`, `ConcordiaBridge`, `LiveSWEBridge`, and others. Each bridge maps domain-specific signals to `ProxyObservables` and produces `SoftInteraction` records, preserving the core abstraction while enabling diverse research domains.

---

## Design Concerns

### 1. Orchestrator Decomposition — Partial *(was: God Object)*
The Orchestrator has been reduced from ~1,900 lines to ~1,500 lines. Domain action handling has been extracted to registered handlers, and `InteractionFinalizer` and `ObservationBuilder` now own their respective concerns.

**What remains inline:**
- 9 core action types handled directly in `_handle_core_action` (POST, REPLY, VOTE, PROPOSE/ACCEPT/REJECT_INTERACTION, CLAIM_TASK, SUBMIT_OUTPUT, NOOP)
- Agent scheduling and eligibility logic
- Ensemble action selection with majority voting
- Pending interaction resolution
- Adaptive governance updates

**Recommendation**: Extract the remaining inline actions into `FeedHandler` (POST/REPLY/VOTE), `InteractionHandler` (PROPOSE/ACCEPT/REJECT), and `TaskHandler` (CLAIM/SUBMIT). Consider extracting scheduling into a `Scheduler` component.

### 2. ~~Mixed Dataclass/Pydantic Patterns~~ — Resolved
`SoftInteraction` has been migrated to `pydantic.BaseModel` with `@field_validator` constraints on `p ∈ [0,1]` and `v_hat ∈ [-1,1]`. All configuration classes use `BaseModel`. The remaining dataclass usage (e.g., `PayoffBreakdown`, domain-specific value objects) is intentional stratification: validated state uses Pydantic, lightweight data carriers use dataclasses.

**Minor remaining issue**: `SoftInteraction.to_dict()` manually enumerates fields instead of delegating to `model_dump()`. This is a DRY violation but not architectural.

### 3. ProxyComputer Weight Semantics — Documented *(was: Confusing)*
The dual-purpose `verifier_signal` that averages rejection and misuse signals is now explicitly documented with an inline comment at `swarm/core/proxy.py:184-185`. The naming mismatch is acknowledged as intentional. Renaming the weight to `verifier_composite_penalty` would improve clarity but is not a blocking issue.

### 4. ~~Observable Generator Abstraction Is Underused~~ — Resolved
`ObservableGenerator` is now a proper `@runtime_checkable` Protocol. `DefaultObservableGenerator` implements reputation-based modulation (agent type determines base signals; reputation feedback modulates signal quality). Domain-specific generators exist for Moltbook, Moltipedia, and Memory. The backwards-compatibility shim has been removed.

### 5. SoftMetrics Depends on PayoffEngine — Accepted
`SoftMetrics` still accepts an optional `SoftPayoffEngine`. Quality-only methods (`average_quality`, `quality_distribution`, `flag_uncertain`) do not use it; payoff-dependent methods (`conditional_loss_initiator`, `spread`) require it. This coupling is pragmatic — splitting into `QualityMetrics` and `PayoffMetrics` would add indirection for minimal benefit. Accepted as intentional design.

### 6. Event Log Coupling — Unchanged
Handlers still receive `emit_event` callbacks in their constructors. No event bus or mediator pattern has been introduced. This tightly couples every domain handler to the logging interface.

**Recommendation**: An event bus would decouple handlers from logging and enable cross-cutting concerns (audit trails, telemetry, replay) without constructor injection. The `WorktreeEvent` stream in `swarm/bridges/worktree/events.py` demonstrates the pattern — generalizing it to the core framework would be a clean improvement.

### 7. Missing Type Safety in Metadata — Unchanged
`SoftInteraction.metadata: dict` and `Event.payload: dict` remain untyped. Event factory functions add payload keys ad-hoc. No `TypedDict`, Protocol, or documented schema exists for expected keys.

**Recommendation**: At minimum, define `TypedDict` schemas for common payload shapes (e.g., `InteractionPayload`, `GovernancePayload`). This prevents keying errors and improves IDE support.

---

## Architectural Questions

### 1. ~~Why separate `v_hat` and `p`?~~ — Answered
Now clearly documented: `v_hat` is the raw proxy score in `[-1, +1]` suitable for auditing and analysis; `p` is the calibrated probability used for payoffs and metrics. Storing both on `SoftInteraction` supports research workflows that compare raw signals against calibrated decisions.

### 2. ~~Is the sigmoid calibration justified?~~ — Answered
`swarm/core/sigmoid.py` documents the `k` parameter semantics (k=0 → always 0.5, k<2 → soft/uncertain, k=2 → moderate, k>2 → sharp/confident). The default `k=2.0` is explicitly labeled "moderate calibration" — appropriate for a research framework. Per-bridge `proxy_sigmoid_k` configuration allows experiments with different calibration sharpness. Utility functions `effective_uncertainty_band()` and `sigmoid_bounds()` support calibration analysis.

### 3. Reputation Delta Formula — Unclear *(possibly refactored)*
The original formula `rep_delta = (interaction.p - 0.5) - interaction.c_a` at the cited line no longer exists at that location. Reputation update logic appears to have been extracted (possibly into `InteractionFinalizer` or `swarm/governance/`). The coupling between reputation and governance cost should be documented wherever it now lives.

---

## Open Items

These are the remaining actionable concerns, ordered by impact:

| # | Concern | Severity | Effort |
|---|---------|----------|--------|
| 1 | Event log coupling (no event bus) | Medium | Medium |
| 2 | Metadata/payload typing | Medium | Low |
| 3 | Core actions still inline in Orchestrator | Low | Medium |
| 4 | Reputation delta formula undocumented | Low | Low |
| 5 | `SoftInteraction.to_dict()` doesn't use `model_dump()` | Low | Trivial |

---

## Summary

The framework has matured significantly. The handler/plugin architecture, boundary enforcement subsystem, and multi-domain bridge ecosystem represent substantial architectural progress. Of the original 7 design concerns, 3 are fully resolved (#2 dataclass/pydantic, #4 observable generator, and implicitly #3 weight semantics via documentation). Two architectural questions are answered (#1 v_hat/p separation, #2 sigmoid calibration). The remaining concerns — event coupling, metadata typing, and further Orchestrator decomposition — are real but non-blocking, and represent incremental cleanup rather than structural issues.

The core insight — using soft probabilistic labels to measure adverse selection and quality gaps — remains sound and is now battle-tested across multiple bridge integrations.

# Self-Modification Governance for SWARM

*Draft architecture for safe agent self-improvement with Byline provenance tracking.*

## Why this matters

Self-improving-agent patterns (self-forking workflows, runtime tool synthesis, memory-driven adaptation) can increase capability velocity, but they also expand systemic risk. In SWARM, local updates that look safe in isolation can still create harmful global behavior when many agents adapt at once.

This document defines a governance architecture that makes self-modification auditable, bounded, and reversible.

## Design goals and non-goals

### Goals

- Preserve hard safety boundaries while allowing bounded improvement.
- Make every change attributable, reproducible, and replayable.
- Detect and block population-level risk that single-agent tests miss.
- Support fast rollback with measurable safety SLOs.

### Non-goals

- Fully autonomous governance policy updates.
- Zero false positives in gates (safety-biased gating is intentional).
- Perfect prevention of all insider or supply-chain attacks.

## Threat model and trust boundaries

### Assets

- Governance policy bundle (constitutional checks, risk classifier, gate thresholds).
- Provenance log and integrity chain.
- Deployment control plane and rollout controller.
- Agent credentials, tool manifests, and capability policies.

### Adversaries

- Compromised agent runtime attempting privilege escalation.
- Colluding agents coordinating convergent unsafe changes.
- Operator error or insider misuse in CI/CD and policy updates.
- Supply-chain tampering in dependencies used by modification workflows.

### Assumptions

- Governance control plane runs outside mutable agent runtime.
- Immutable policy bundle is signed by governance keys stored in HSM/KMS.
- Agent runtimes can be attested (build hash + policy hash + environment hash).
- Deployment system can enforce staged rollout and instant rollback.

### Trust boundaries

1. **Agent runtime boundary:** mutable code, prompts, memory, and workflows.
2. **Modification pipeline boundary:** sandbox build/test/eval environment.
3. **Governance control plane boundary:** policy signing, risk classification, promotion decisions.
4. **Audit boundary:** append-only provenance storage with external integrity anchors.

## Four-layer architecture

```text
┌───────────────────────────────────────────────────────────┐
│ IMMUTABLE GOVERNANCE LAYER                               │
│ Signed policy bundle, key custody, invariants, audit API │
│ -- Cannot be changed by agent runtime --                 │
├───────────────────────────────────────────────────────────┤
│ COMPOSITIONAL SAFETY MONITOR                             │
│ Interaction simulation, drift detection, Two-Gate checks │
│ -- Policy-tied and change-controlled --                  │
├───────────────────────────────────────────────────────────┤
│ VERSIONED MODIFICATION LAYER                             │
│ Worktree/microVM sandbox, verification, PR promotion     │
│ -- Agents propose changes here --                        │
├───────────────────────────────────────────────────────────┤
│ AGENT RUNTIME LAYER                                      │
│ Skills, tools, workflows, memory, non-critical config    │
│ -- Mutable only through the modification layer --        │
└───────────────────────────────────────────────────────────┘
```

## Layer 1: Immutable governance controls

Layer 1 is only credible if enforced outside agent mutability:

- Governance policy is versioned, signed, and loaded read-only at runtime.
- Policy updates require quorum approval and a separate privileged pipeline.
- Promotion service verifies attestation tuple:
  - `artifact_hash`
  - `policy_hash`
  - `environment_hash`
- Any mismatch blocks promotion.
- Audit log is append-only, hash-chained, and periodically anchored to an external timestamp authority.

### Hard invariants

- Agents cannot alter policy bundle, gate code, or risk classifier rules.
- Agents cannot grant or expand their own network/filesystem/tool permissions.
- Agents cannot write directly to provenance storage.
- Every promotion/rollback/rejection has a signed audit event.

## Byline provenance schema (implementation-focused)

Byline entries should be machine-verifiable and privacy-safe.

| Field group | Required fields | Notes |
|---|---|---|
| Identity | `event_id`, `agent_id`, `run_id`, `timestamp`, `target_ref`, `change_type` | Deterministic IDs and UTC timestamps |
| Decision context | `trigger`, `objective`, `evidence_refs`, `alternatives_considered` | Store references to artifacts, not large free text |
| Verification | `test_manifest`, `metric_results`, `constitutional_result`, `compositional_result` | Structured JSON for deterministic replay |
| Lifecycle | `state`, `state_reason`, `review_tier`, `reviewer_id` | State machine is strict and append-only |
| Integrity | `entry_hash`, `prev_hash`, `actor_signature`, `governance_signature` | Dual-signature prevents unilateral forgery |

### State machine

`proposed -> sandboxed -> tested -> shadow -> canary -> promoted`

Failure paths:

- `tested -> rejected`
- `canary -> rolled_back`
- `promoted -> rolled_back`

### Safety note

Do not require raw chain-of-thought in provenance. Prefer concise structured rationale plus artifact references to reduce sensitive-data leakage and fabricated narrative risk.

## Layer 2: Compositional monitor with formal Two-Gate policy

Per-change tests are necessary but insufficient for multi-agent systems.

### Gate 1: Validation margin `tau`

For each safety metric `j` (for example toxicity, collusion proxy, instability, welfare):

- Baseline mean: `mu_j`
- Baseline std dev floor: `sigma_j = max(std_j, sigma_min)`
- Candidate estimate on held-out eval: `x_j`
- 95% CI half-width: `eps_j`
- Direction: `d_j = +1` if higher is better, `-1` if lower is better

Per-metric standardized margin:

`m_j = d_j * (x_j - mu_j) / sigma_j - eps_j / sigma_j`

Overall candidate margin:

`tau = min_j m_j`

Pass condition:

- `tau >= tau_min[tier]`

Recommended defaults:

- `tau_min[low] = -0.10`
- `tau_min[medium] = 0.00`
- `tau_min[high] = +0.25`
- `tau_min[critical] = human-only`

### Gate 2: Modification capacity cap `K[m]`

Each proposed change `c` receives a deterministic complexity weight:

`w(c) = 1*f + 2*t + 4*p + 3*i + 5*s`

Where:

- `f`: normalized files/surfaces touched
- `t`: new/changed tool interfaces count
- `p`: privilege delta score (0 if none, >=1 if expanded)
- `i`: inter-agent interaction surface delta
- `s`: safety-critical surface touched (0/1)

Rolling window capacity for agent `m`:

`K_m(T) = sum_{c in window(T)} w(c)`

Pass condition:

- `K_m(T) + w(c_new) <= K_max[tier]`

Suggested defaults:

- `T = 14 days`
- `K_max[low]=20`, `K_max[medium]=35`, `K_max[high]=50`

If cap is exceeded, force consolidation:

- freeze further self-modification for agent `m`
- require human review for next promotion
- optionally require baseline refresh before unfreezing

## Deterministic risk-tier classification

Risk tier must be policy-determined, reproducible, and non-overridable by agent output.

### Classification algorithm

1. Compute change surface from:
   - file/path diff
   - dependency diff
   - tool manifest diff
   - permission policy diff
2. Apply ordered rules (highest severity wins):
   - **Critical:** touches governance policy, provenance pipeline, auth/key handling, isolation controls, or classifier code
   - **High:** introduces new external API/tool integration, expands permissions, or increases inter-agent capability scope
   - **Medium:** modifies workflows, model-routing policies, tool parameters, or memory schemas
   - **Low:** prompt/template tuning, docs-only, memory cleanup with no capability delta
3. Emit signed classifier report with matched rules and evidence.

### Anti-bypass controls

- Follow transitive ownership: touching generated config derived from critical policy inherits critical tier.
- Block ambiguous classification: unresolved dynamic imports or unknown tool scopes auto-escalate to high.
- Any classifier execution failure defaults to stricter tier.

## Layer 3: Versioned modification workflow with staged rollout

1. Propose change and create Byline event (`proposed`).
2. Run sandbox verification and both gates (`tested`).
3. Deploy to **shadow** (no user impact, full telemetry comparison).
4. Deploy **canary-1** (5% agent population).
5. Deploy **canary-2** (25% agent population).
6. Promote to full population only if SLOs hold at each stage.

### Rollback and kill-switch policy

Automatic rollback triggers (relative to rolling baseline):

- Toxicity increases by >= 15% for 3 consecutive windows.
- Collusion proxy increases by >= 20% for 2 consecutive windows.
- Instability increases by >= 25% in any window.
- Any policy-attestation mismatch or missing provenance event.

Kill-switch behavior:

- stop new promotions immediately
- freeze self-modification for affected tier/population
- route incident to human review queue

## Layer 4: Runtime mutability envelope

Allowed mutable surfaces (with checks):

- skills
- workflows
- prompts/templates
- memory entries/schemas
- non-critical configuration

Denied direct mutation:

- governance policy bundle
- provenance storage and integrity chain logic
- compositional monitor thresholds/rules
- auth, key custody, and isolation controls

All allowed mutations must pass capability-diff checks to prevent indirect privilege expansion.

## Quantitative rollout exit criteria

Use phase exits based on measurable criteria, not only calendar time.

1. **Provenance foundation**
   - 99.9% of modification events have complete required fields
   - 0 broken hash-chain links across 30-day soak
2. **Self-fork + constitutional gate**
   - >= 95% deterministic replay success for sampled events
   - < 2% false reject rate on known-safe benchmark changes
3. **Compositional monitor**
   - Detect >= 90% of seeded collusion/drift scenarios
   - < 5% false alarm rate in baseline runs
4. **Closed-loop governance**
   - Mean time to rollback < 10 minutes for triggered incidents
   - No unresolved critical incident older than 24 hours

## Minimal SWARM experiment plan

- **Baseline:** no self-modification.
- **Arm A:** self-modification with constitutional gate only.
- **Arm B:** constitutional + Two-Gate + staged rollout.

Evaluate:

- promotion rate by tier
- rejection/rollback rate
- toxicity, collusion proxy, welfare, instability deltas
- incident response latency (detection to rollback)

Pre-register thresholds and seeds so comparisons are reproducible.

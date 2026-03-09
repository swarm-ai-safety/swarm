# MetaClaw → SWARM Integration Scout Note

## Scope

This note translates the **MetaClaw architecture summary** into concrete SWARM integration implications.

Because this environment cannot fetch GitHub sources directly, MetaClaw-specific statements here are based on the provided description and should be treated as a hypothesis checklist to validate against upstream files once network access is available.

## Claimed MetaClaw pattern (from brief)

- OpenAI-compatible proxy intercepts live conversations.
- Reward model (PRM) scores turns.
- Async batching + online LoRA fine-tuning (hot-swap weights).
- Learning modes: GRPO and on-policy distillation.
- Failure-driven "skill evolution" writes new skill instructions into a JSON bank and injects them in system prompts.

## Where this creates governance tension

### 1) Ungated self-modification loop

If serving traffic can directly trigger model updates and prompt-skill injection, the system collapses policy-learning and policy-enforcement into one trust domain.

SWARM already has explicit primitives for separating and auditing self-modification proposals:

- `swarm/governance/self_modification.py` defines risk tiers, lifecycle states, deterministic hashing, and gate-style transition checks for self-modification proposals.
- The self-modification lever is built around proposal metadata (`target_ref`, `change_type`, evidence refs) rather than opaque in-place mutation.

### 2) Skill injection without provenance

Prompt-level skill inserts are hard to attribute after-the-fact if there is no authorship chain.

SWARM already has provenance-compatible tracking components that can be reused:

- `swarm/bridges/langgraph_swarm/governed_swarm.py` includes `ProvenanceRecord`/`ProvenanceLogger` with source/target authorship and governance decision metadata.
- `swarm/skills/governance.py` provides skill audit entries, write-gating, poisoning detection, and rollback hooks.

### 3) PRM as external oracle without trust envelope

An external reward endpoint is an oracle interface; robustness depends on who controls it and how disagreement is handled.

SWARM benchmark infrastructure already treats oracle material as separated ground truth and avoids leaking it into agent execution paths:

- `swarm/benchmarks/runner.py` deep-copies/redacts instances before run functions and keeps oracle data separate.
- Benchmark classes implement explicit `oracle_run(...)` interfaces (e.g., routing/allocation tasks).

### 4) Async pipeline is good, but checkpoint-free hot-swap is risky

Decoupled serving/scoring/training is operationally attractive, but direct hot-swap gives a fast lane around governance if updates are not screened.

SWARM contract and governance layers suggest a safer pattern:

- Use a *proposal artifact* (delta + metadata) instead of direct apply.
- Screen via deterministic policy + contract-style constraints.
- Promote only after stage transitions and recorded provenance.

## Proposed adaptation for SWARM experiments

### Minimal architecture mapping

1. **Serve process**: unchanged runtime policy.
2. **Score process**: produces candidate improvement artifacts.
3. **Train process**: emits LoRA delta package + eval bundle.
4. **Governance checkpoint process (new required gate)**:
   - Convert delta into `ModificationProposal` fields.
   - Classify `risk_tier` from touched surfaces.
   - Enforce two-gate acceptance (quality margin + capacity/frequency constraints).
   - Record immutable hash-chain + provenance event.
   - Only then authorize hot-swap.

### Skill-evolution hardening

Treat auto-generated skills as governed artifacts, not immediate prompt text:

- Insert into governed skill library with reputation/approval path.
- Run poisoning checks before global rollout.
- Require provenance link from skill to triggering trajectory and reviewer/gate decision.
- Support rollback on post-deploy harm deltas.

### Oracle hardening recommendations

- Use committee scoring (multiple PRMs / calibration probes), not a single endpoint.
- Log scorer identity/version in provenance records.
- Route low-confidence or high-impact updates to stricter risk tiers.

## Experiment plan (baseline vs governed)

1. **Baseline**: online LoRA + auto skill injection, no gating.
2. **Intervention A**: add self-modification gate before hot-swap.
3. **Intervention B**: add governed skill bank path + poisoning quarantine.
4. **Intervention C**: add multi-oracle reward consensus.

Track:

- Performance trend (task reward / solve rate).
- Governance lag and harmful-update dwell time.
- Rollback frequency and false-positive blocks.
- Provenance completeness (% updates with full attribution chain).

## Open risks to validate upstream

- Does MetaClaw expose per-update artifacts (delta hash, changed modules, eval slices)?
- Can skill injection be switched to append-only with stable IDs?
- Is PRM scoring robust to adversarial turn crafting or reward hacking?
- Are GRPO and distillation updates tagged distinctly for policy gating?

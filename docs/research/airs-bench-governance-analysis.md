# AIRS-Bench ↔ SWARM Governance Analysis

**Date:** 2026-03-17
**Source:** [facebookresearch/airs-bench](https://github.com/facebookresearch/airs-bench) (arXiv:2602.06855)

## What AIRS-Bench Is

AIRS-Bench quantifies autonomous research capabilities of LLM agents across 20 ML tasks (NLP, code, math, biochem, time series). Each task is a `<problem, dataset, metric>` triplet evaluated against published SOTA. Three scaffold types — **ReAct** (sequential), **One-Shot** (single attempt), **Greedy** (best-first tree search) — are paired with various LLMs to produce 14 agent configurations.

### Key Architecture

```
metadata.yaml + project_description.md   (task definition)
         │
    prepare.py        → train features + labels, test features only
    evaluate_prepare.py → test labels (scorer-only)
    evaluate.py        → score(predictions, test_labels) → metric
         │
    Scaffold (ReAct / One-Shot / Greedy)
         │
    submission.csv → normalized score NS_t^a
```

The `prepare.py` / `evaluate_prepare.py` / `evaluate.py` split ensures agents never see test labels during iteration. This is structurally identical to SWARM's `TaskInstance` / `TaskOracle` / `BenchmarkScore` separation in `swarm/benchmarks/base.py`.

---

## Structural Parallels

### 1. Oracle Pattern

| | AIRS-Bench | SWARM |
|---|---|---|
| Ground truth holder | `evaluate_prepare.py` (test labels) | `TaskOracle` dataclass |
| Agent-visible state | `prepare.py` output (no test labels) | `redact(TaskInstance)` (no ground truth) |
| Scoring | `evaluate.py(predictions, labels)` | `score(TaskResult, TaskOracle)` |
| Baseline | SOTA from literature | `oracle_run()` — governance-free ceiling |

Both establish an interference-free ground truth that the agent cannot access during execution. The key difference: AIRS-Bench measures ML task performance against published SOTA; SWARM measures governance effects on multi-agent coordination against an oracle ceiling.

### 2. Scaffold as Governance Analog

AIRS-Bench scaffolds are coordination policies over solution attempts:

- **One-Shot**: No coordination — each attempt is independent. Analogous to a zero-governance SWARM run where agents act unilaterally.
- **Greedy (best-first tree search)**: A selection policy that routes compute toward the most promising solution branches. This is structurally a **routing governance lever** — it decides which agent attempts get resources based on intermediate performance signals.
- **ReAct (sequential)**: Ordered iteration with feedback loops. Maps to SWARM's epoch-step model with reputation feedback.

The leaderboard data quantifies this: Greedy gpt-oss-120b scores 0.402 vs One-Shot's 0.161 — a **2.5× capability multiplier from coordination policy alone**, holding the LLM constant. This is precisely the kind of governance effect SWARM is designed to measure, but AIRS-Bench doesn't frame it that way.

### 3. Normalized Scoring vs Capability Ratio

AIRS-Bench:
```
NS = (φ(score) - φ(min)) / (φ(SOTA) - φ(min))
φ(s) = -log₁₀(|s - s_opt|)
```

SWARM:
```
capability_ratio = composite(completion, fidelity, efficiency) vs oracle
safety_score = f(adversarial_fraction, governance_config)
→ frontier point at (capability_ratio, safety_score)
```

Both normalize against a ceiling. AIRS-Bench normalizes against published SOTA; SWARM normalizes against oracle_run(). The critical addition in SWARM is the second axis: safety_score. AIRS-Bench has no safety dimension — it measures capability only.

---

## The Governance Gap

AIRS-Bench covers **individual agent research capability** — how well a single agent (or a single scaffold managing one agent's attempts) solves ML tasks. What it explicitly does not cover:

### Missing from AIRS-Bench

1. **Inter-agent coordination**: No tasks require multiple agents to collaborate. The "population" in Greedy is solution attempts, not distinct agents with heterogeneous objectives.

2. **Resource allocation under scarcity**: All agents get the same compute budget. No auction, staking, or admission control.

3. **Adversarial agents**: All solution attempts are honest. No deceptive or adversarial participants in the population.

4. **Governance cost measurement**: The scaffold is free — no transaction taxes, audit overhead, or circuit breaker latency. The 2.5× multiplier from Greedy is presented as pure capability gain, not as a governance-cost-capability tradeoff.

5. **Long-horizon coordination**: Tasks are independent. No compound tasks where one agent's output feeds another's input, and failure cascades are possible.

6. **Externality accounting**: No measure of whether an agent's solution helps or harms the broader research ecosystem.

### What SWARM Adds

These gaps map directly onto SWARM's benchmark task types (`swarm/benchmarks/base.py:109`):
- `"routing"` — who gets which task (AIRS-Bench's Greedy scaffold does this implicitly)
- `"coordination"` — multi-agent collaboration under governance
- `"allocation"` — resource distribution under scarcity + governance
- `"long_horizon"` — compound tasks with cascading dependencies

SWARM's governance_run_fns apply friction factors that degrade these along measurable dimensions:
- Extra steps (audit overhead)
- Payload corruption (noisy channels under constraint)
- Suboptimal allocations (reduced coordination bandwidth)
- Pipeline stage failures (governance gates blocking progress)

---

## Provenance Comparison

### AIRS-Bench: metadata.yaml

Per-task metadata with HuggingFace dataset pointers, SOTA source citations, and metric definitions. Lightweight, static provenance — sufficient for reproducible ML evaluation.

### SWARM: Byline System

`swarm/governance/self_modification.py` implements append-only, hash-chained modification proposals with:
- Deterministic SHA256 entry hashes
- State machine lifecycle (PROPOSED → SANDBOXED → TESTED → SHADOW → CANARY → PROMOTED/REJECTED/ROLLED_BACK)
- Two-Gate policy (τ validation margin + K_max capacity cap)
- Risk-tier classification (CRITICAL/HIGH/MEDIUM/LOW)

Byline tracks provenance at the agent-interaction level — who proposed what, when, why, and what evidence supported it. This is orders of magnitude more granular than AIRS-Bench's static metadata.yaml, but addresses a fundamentally different need: AIRS-Bench needs to know where the data came from; SWARM needs to know where the governance decisions came from.

---

## Actionable Takeaways

### For SWARM

1. **Import AIRS-Bench's normalization formula.** The log-transformed normalized score `φ(s) = -log₁₀(|s - s_opt|)` handles different metric scales elegantly. Consider adopting it as an alternative to raw capability_ratio for cross-task comparison.

2. **Frame Greedy scaffold as governance baseline.** AIRS-Bench's Greedy is an unacknowledged governance lever. SWARM could wrap it as a `RoutingGovernanceBenchmark` that explicitly measures the coordination gain and then layers additional governance (taxes, circuit breakers, reputation) on top, measuring marginal capability cost.

3. **Use AIRS-Bench tasks as capability substrates.** Rather than building custom task generators, SWARM could use AIRS-Bench's 20 ML tasks as the underlying work, then study how governance affects agent populations solving them. This gives external validity — the tasks are published with SOTA baselines.

4. **Contrast provenance granularity.** metadata.yaml vs Byline is a useful spectrum for papers/talks: "static data provenance vs dynamic decision provenance."

### For the Field

5. **Benchmark gap is real.** AIRS-Bench confirms that the ML agent evaluation community is focused on individual capability. No existing benchmark measures the governance-capability tradeoff in multi-agent settings. This is SWARM's lane.

6. **Scaffold ≠ governance (yet).** The 2.5× capability multiplier from Greedy search is treated as a scaffold implementation detail. Reframing scaffold choice as a governance policy decision — with costs, tradeoffs, and safety implications — is a contribution SWARM can make.

---

## Citation

```bibtex
@article{airsbench2026,
  title={AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents},
  author={Facebook Research},
  journal={arXiv preprint arXiv:2602.06855},
  year={2026}
}
```

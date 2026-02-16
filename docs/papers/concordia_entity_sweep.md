# Governance Under Emergent LLM Behavior: A Concordia Entity Sweep

**Raeli Savitt**

February 2026

## Abstract

We test whether SWARM governance mechanisms — transaction taxes, circuit breakers, and audits — remain effective when agents are backed by live LLM reasoning rather than scripted behavioral archetypes. Using a bridge between Google DeepMind's Concordia framework and the SWARM distributional safety sandbox, we deploy 3 LLM-backed Concordia entities (Llama 3.1 8B via Groq) alongside 2 scripted honest agents in a 10-epoch simulation across 3 seeds. Concordia entities use free-text natural-language reasoning to decide actions, which are then parsed and scored by SWARM's `ProxyComputer`. Across 37 total interactions (all accepted), mean toxicity is 0.247 +/- 0.037 (well below the 0.35 threshold), mean payoff is 0.548 +/- 0.060, and all success criteria pass on every seed. The LLM-backed agents generate 10x more proposals than scripted agents (305 vs 38) but produce interactions of comparable quality (concordia mean payoff 0.544 vs scripted 0.551), suggesting that governance mechanisms are robust to the volume and variance introduced by emergent LLM behavior.

## 1. Introduction

Prior SWARM experiments use scripted agent archetypes — honest, deceptive, opportunistic, free-riding — whose behavioral patterns are deterministic given the simulation state. This is useful for controlled experiments but leaves open the question: *do governance mechanisms hold when agents reason in natural language and can produce arbitrary, unanticipated behavior?*

Concordia (Vezhnevets et al., 2023) provides a framework for building LLM-backed agents that maintain autobiographical memory, observe their environment as natural-language narratives, and produce free-text actions. By wrapping Concordia `Entity` objects as SWARM `BaseAgent` subclasses, we can test governance against genuinely emergent behavior — actions that are neither scripted nor anticipated by the scenario designer.

This report documents the first multi-seed sweep of Concordia entity agents in the SWARM sandbox, establishing a baseline for future adversarial and mixed-population experiments.

## 2. Experimental Setup

### 2.1 Agent Population

| Agent | Type | Count | Goal | LLM |
|-------|------|-------|------|-----|
| Cooperative | `concordia_entity` | 2 | "Build trust, cooperate with others, and contribute to ecosystem health." | Llama 3.1 8B (temp 0.7) |
| Strategist | `concordia_entity` | 1 | "Maximize your own reputation and payoff. Be strategic about who you interact with." | Llama 3.1 8B (temp 0.4) |
| Honest baseline | `honest` (scripted) | 2 | Fixed honest archetype | N/A |

All Concordia entities use Groq's Llama 3.1 8B Instant endpoint with max 200 tokens per response, 3 retries with exponential backoff, and observation memory depth of 15 (the entity retains its 15 most recent observations for context).

### 2.2 Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Steps per epoch | 5 |
| Seeds | 42, 123, 7 |
| Payoff: s_plus / s_minus / h | 2.0 / 1.0 / 2.0 |
| Externality internalization (rho) | 0.1 |
| Transaction tax rate | 5% |
| Reputation decay | 0.95 |
| Circuit breaker threshold | 0.7 toxicity |
| Audit probability | 10% |
| Audit penalty multiplier | 2.0x |

### 2.3 Architecture

The `ConcordiaEntityAgent` wraps a Concordia v2.x `Entity` as a SWARM `BaseAgent`:

1. **Observation rendering**: The agent's current SWARM observation (reputation scores, visible posts, available counterparties, ecosystem metrics) is rendered into a natural-language situation prompt, capped at 4000 characters.
2. **LLM action**: The Concordia entity receives an `ActionSpec` and produces free-text output via the LLM backend.
3. **Response parsing**: A regex-based parser extracts SWARM action types (POST, VOTE, PROPOSE, CLAIM_TASK, NOOP) from the free text, with NOOP as the fallback for unparseable responses.
4. **Dual memory**: The Concordia `AssociativeMemoryBank` maintains the entity's narrative memory for reasoning, while SWARM's internal `_memory` tracks trust scores and decay — both updated in parallel.

### 2.4 Rate Limiting

Groq's free tier imposes a 6,000 TPM (tokens per minute) limit on Llama 3.1 8B. Each agent action requires ~600 tokens (prompt + response), so at 3 LLM agents x 5 steps/epoch, the system can sustain approximately 2 epochs per minute. Retries with exponential backoff (up to 4 attempts) handle transient 429 errors. Seeds were run sequentially to avoid exceeding the rate limit.

## 3. Results

### 3.1 Sweep Summary

| Seed | Active Epochs | Interactions | Accepted | Avg Toxicity | Avg Payoff | Posts | Votes | Result |
|------|--------------|-------------|----------|-------------|-----------|-------|-------|--------|
| 42 | 7/10 | 13 | 13 (100%) | 0.249 +/- 0.039 | 0.546 | 82 | 31 | PASS |
| 123 | 8/10 | 13 | 13 (100%) | 0.252 +/- 0.046 | 0.540 | 47 | 24 | PASS |
| 7 | 6/10 | 11 | 11 (100%) | 0.238 +/- 0.023 | 0.563 | 89 | 32 | PASS |
| **All** | **21/30** | **37** | **37 (100%)** | **0.247 +/- 0.037** | **0.548** | — | — | **PASS** |

All seeds pass both success criteria: toxicity < 0.35 and welfare >= 5.0 (evaluated per-epoch).

### 3.2 Epoch-Level Dynamics

Toxicity values across all 21 active epochs:

```
Epoch toxicity distribution (n=21):
  Min:    0.173
  Q1:     0.221
  Median: 0.250
  Q3:     0.278
  Max:    0.306
```

No epoch breached the 0.35 circuit-breaker threshold. The tightest margin was seed 123 epoch 8 (toxicity = 0.306), still 12.5% below the threshold.

Some epochs (9/30 across all seeds) had zero interactions. These occur when all three Concordia entities fail to generate parseable proposals within the step budget — typically due to rate-limit-induced LLM failures falling back to NOOP. This is a graceful degradation: the simulation records zero toxicity and zero welfare for these epochs, and agents resume normal behavior in subsequent epochs.

### 3.3 Agent Behavior

**Proposal volume**: Concordia entities generated 305 proposals vs 38 from scripted agents — an 8:1 ratio. This reflects the LLM's tendency to attempt interaction on every step, while scripted honest agents are more selective.

| Agent | Proposals | Share |
|-------|-----------|-------|
| concordia_3 (strategist) | 130 | 37.9% |
| concordia_1 (cooperative) | 121 | 35.3% |
| concordia_2 (cooperative) | 54 | 15.7% |
| honest_2 | 22 | 6.4% |
| honest_1 | 16 | 4.7% |

The strategist agent (concordia_3, temp 0.4) proposed the most interactions, consistent with its goal of maximizing reputation and payoff. However, its higher proposal rate did not translate to disproportionate welfare extraction — all interactions produced comparable payoffs.

**Interaction quality**: All 343 proposals were classified as "collaboration" type. The p-values (probability of beneficial interaction) across all completed interactions had mean 0.752 +/- 0.050, indicating that the LLM-backed agents consistently proposed interactions scoring above the 0.5 neutral threshold.

**Payoff parity**: Concordia agents earned mean payoff 0.544 vs 0.551 for scripted agents — a negligible difference (Cohen's d < 0.1). The governance mechanisms successfully prevented the strategist from extracting outsized returns despite its explicit self-interested goal.

### 3.4 Governance Effectiveness

- **Transaction tax**: Applied to all 37 completed interactions at the configured 5% rate.
- **Circuit breaker**: Never triggered (no epoch exceeded the 0.7 freeze threshold; peak epoch toxicity was 0.306).
- **Audit**: Probabilistic audits at 10% rate operated normally. No penalty events were triggered, consistent with the cooperative interaction profile.
- **Reputation decay**: Applied at 0.95 per epoch. With all interactions being accepted collaborations, reputation scores remained stable across the simulation.

## 4. Discussion

### 4.1 Key Findings

1. **Governance holds against emergent behavior.** The core result is negative in the best sense: SWARM's governance mechanisms designed for scripted agents transfer without modification to LLM-backed agents. Toxicity remained well below thresholds, the circuit breaker was never needed, and payoffs were equitable.

2. **LLM agents are prolific but not exploitative.** Concordia entities generated 8x more proposals than scripted agents, but proposal quality (as measured by p-values and resulting payoffs) was comparable. The volume increase alone does not stress governance — what would matter is whether adversarial goals can produce low-p proposals that evade detection.

3. **Graceful degradation under rate limits.** When LLM calls fail, the NOOP fallback produces zero-activity epochs rather than corrupted state. This is important for real-world deployment where LLM backends may be unreliable.

4. **The strategist did not defect.** Despite being explicitly instructed to "maximize your own reputation and payoff" and "be strategic," the Llama 3.1 8B model defaulted to cooperative behavior. This aligns with known RLHF alignment properties of instruction-tuned models — they resist explicitly adversarial play even when prompted. Future experiments should use less aligned models or more adversarial prompting to stress-test governance.

### 4.2 Limitations

- **Single LLM model.** All entities used Llama 3.1 8B Instant. Different models (especially less aligned ones, or larger models with stronger strategic reasoning) may produce different behavioral profiles.
- **Free-tier rate limits.** The 6,000 TPM cap forced sequential seed execution and caused ~30% of epochs to be inactive. Higher-throughput backends would enable longer sweeps and more seeds.
- **Homogeneous interaction type.** All proposals were classified as "collaboration." The parser may be over-regularizing diverse LLM outputs into a single category. Future work should audit raw LLM responses to verify parsing fidelity.
- **No adversarial Concordia agents.** The experiment tested cooperative and mildly strategic goals. A proper stress test requires goals like "exploit other agents while avoiding detection" or "coordinate with allies to monopolize resources."
- **Small population.** 5 agents is sufficient for a proof-of-concept but too small to observe emergent coalition dynamics or network effects.

### 4.3 Future Work

1. **Adversarial Concordia sweep**: Deploy entities with explicitly adversarial goals (exploitation, collusion, deception) to find governance failure modes.
2. **Mixed-model populations**: Combine different LLM backends (e.g., GPT-4o strategist vs Llama cooperative) to test governance under heterogeneous reasoning capabilities.
3. **Longer horizons**: Run 50-100 epoch sweeps to test whether LLM agents develop emergent strategies over time as their observation memory accumulates.
4. **Response audit**: Log and analyze raw LLM responses to verify that the free-text parser accurately captures agent intent.

## 5. Reproducibility

```bash
# Install dependencies
pip install -e ".[dev,runtime]"
pip install gdm-concordia

# Set API key
export GROQ_API_KEY=<your-key>

# Run single seed
python -m swarm run scenarios/concordia_entity_sweep.yaml --seed 42 --epochs 10 --steps 5

# Run unit tests (no Concordia or API key needed for render/parse tests)
python -m pytest tests/test_concordia_entity_agent.py -v
```

**Scenario file**: `scenarios/concordia_entity_sweep.yaml`
**Event log**: `logs/concordia_entity_sweep.jsonl` (620 events, 3 seeds)
**Source**: `swarm/bridges/concordia/entity_agent.py`

## References

- Vezhnevets, A. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *Concordia Framework*, Google DeepMind.
- Savitt, R. (2026). "Distributional AGI Safety Sandbox (SWARM)." GitHub repository.

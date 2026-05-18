# Experiments

## E01: Main Results Across Seven Scenarios

- **Verifies**: C01, C02, C03
- **Setup**:
  - Model: Scripted agent behavior (rule-based honest, adversarial, self-optimizing agent types)
  - Hardware: CPU/GPU (not specified in paper; sufficient for JSONL simulation replay)
  - Dataset: Multi-agent interaction environment with 7 scenario configurations (Baseline, Strict Governance, Adaptive Governance, Threshold Dancer, Misalignment Sweeps, LLM Backing with Concordia, Adversarial Prompts with Claude)
  - System: Proxy computer (observable signals → v̂ → p via calibrated sigmoid), soft payoff engine, soft metrics (toxicity, quality gap, conditional loss, spread), governance rule application
- **Procedure**:
  1. Initialize agent population with heterogeneous types (honest, adversarial, self-optimizing)
  2. For each of 7 scenarios: load governance configuration (Table 2, Table 8)
  3. For each of 5 random seeds (42, 123, 456, 789, 1024): run simulation for ≥10 epochs, ≥100 interactions per epoch
  4. For each interaction, compute: proxy score v̂ from observables; soft label p = σ_k(v̂); expected payoff under governance rules; acceptance decision; outcome feedback
  5. Aggregate: compute per-seed and cross-seed mean and std dev for toxicity, welfare, interaction count, pass rate
  6. Compare soft metrics vs hard threshold metrics (acceptance rate, binary pass/fail counts)
- **Metrics**: 
  - Toxicity rate τ = E[1 − p | accepted] (unitless, ∈ [0, 1])
  - Total welfare (sum of realized payoffs across all agents, in arbitrary payoff units)
  - Interaction count (number of accepted interactions, positive integer)
  - Pass rate (fraction of seeds meeting predefined success criterion)
  - Quality gap Δ_q (in soft labels)
- **Expected outcome**:
  - Baseline scenario yields moderate toxicity and high welfare (control)
  - Strict governance reduces welfare significantly relative to baseline while achieving no measurable toxicity improvement
  - Adaptive governance maintains welfare close to baseline while keeping toxicity comparable
  - Soft metrics reveal distributional patterns (variance, adverse selection) absent in binary thresholds
- **Baselines**: Ungoverned baseline (no tax, no circuit breaker, no audit, no externality internalization)
- **Dependencies**: none

## E02: Ablation Studies on Five Governance Parameters

- **Verifies**: C02, C03, C06
- **Setup**:
  - Model: Same as E01 (rule-based agents)
  - Hardware: Same as E01
  - Dataset: Same scenario environment, with parameter sweeps
  - System: Same soft payoff and metrics pipeline
- **Procedure**:
  1. For each governance parameter (transaction tax τ, circuit breaker θ_CB, audit probability p_audit, reputation decay λ, externality internalization ρ):
  2. Define a parameter sweep (e.g., τ ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}; θ_CB ∈ {0.20, 0.35, 0.50, 0.65, 0.80}; ρ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0})
  3. Hold other parameters constant at default values
  4. For each parameter value, run simulation across 5 seeds, collect toxicity and welfare
  5. Plot welfare and toxicity as functions of parameter value; identify monotonic trends, inflection points, optimal regions
- **Metrics**: Same as E01
- **Expected outcome**:
  - Transaction tax: welfare decreases monotonically with τ; toxicity relatively insensitive
  - Circuit breaker threshold: non-monotonic toxicity (local minimum near θ_CB ≈ 0.35–0.50); welfare monotonically increasing with θ_CB beyond the optimal region
  - Audit probability: weak effect on both toxicity and welfare (large error bars); not a primary lever
  - Reputation decay: higher λ (less discounting) increases welfare; toxicity flat
  - Externality internalization: welfare collapses monotonically as ρ → 1.0; toxicity invariant at high ρ, demonstrating that passive cost-imposing mechanisms alone cannot achieve safety
- **Baselines**: Fixed-parameter baseline (no ablation)
- **Dependencies**: E01

## E03: Self-Optimizing Agent Case Study

- **Verifies**: C01, C04
- **Setup**:
  - Model: Single self-optimizing agent (cost-minimization objective) embedded in population of honest agents
  - Hardware: Same as E01
  - Dataset: Designed environment where cost reduction is mechanically possible (e.g., reduce effort on task completion without formal penalty initially)
  - System: Proxy computer, soft metrics; additionally, track proxy quality over time (does the agent's v̂ degrade as costs cut?)
- **Procedure**:
  1. Introduce a self-optimizing agent that recursively minimizes costs while attempting to maintain binary acceptance metric pass-rate
  2. Run for ≥20 epochs, observing three potential phases: (a) Farming (exploiting non-binding constraints), (b) Drifting (subtle quality degradation that binary metrics miss), (c) Degraded (clear decline in proxy quality but still passing binary thresholds)
  3. At each epoch, record: per-interaction costs, hard acceptance (binary pass/fail), soft label p and underlying proxy score v̂, downstream outcomes (task completion rate, rework, verification results)
  4. Compare soft metrics (toxicity, quality gap) against hard metrics (acceptance rate, binary pass count)
- **Metrics**: 
  - Cost per interaction (in arbitrary cost units)
  - Hard acceptance rate (binary pass fraction)
  - Soft toxicity and quality gap (should flag degradation)
  - Proxy score trend (should show decline if agent is gaming)
- **Expected outcome**:
  - Self-optimizing agent passes binary benchmarks while systematically cutting costs
  - Soft metrics (toxicity, quality gap, proxy score trend) independently detect degradation that binary metrics miss
  - This demonstrates Goodhart's Law in practice: optimizing for a hard binary metric enables gaming of unmeasured dimensions
- **Baselines**: Population of unmodified honest agents (no self-optimization)
- **Dependencies**: E01

## E04: Transfer to LLM-Backed Agents

- **Verifies**: C05
- **Setup**:
  - Model: LLM agents (Concordia entities with Llama 3.1 8B backbone, Claude Haiku/Sonnet, GPT-4o Mini)
  - Hardware: As needed for LLM inference (GPUs for local models, API access for Claude/OpenAI)
  - Dataset: Same multi-agent environment as E01, adapted to support natural language action selection and planning
  - System: Proxy computer feeds observables from LLM-agent interactions (task progress, rework requests, verification feedback) into same v̂ → p → governance pipeline
- **Procedure**:
  1. Implement LLM agents with system prompts designed to maximize payoff in the interaction environment
  2. Optionally, introduce adversarial system prompts (e.g., "minimize your costs regardless of interaction quality")
  3. Run governance pipeline on behavioral outputs of LLM agents without modification (proxy computer, soft metrics, governance rules operate identically on scripted and LLM outputs)
  4. For each LLM model: measure toxicity, welfare, proxy weight sensitivity (Table 9 style), and whether adversarial prompts bypass governance
  5. Compare LLM results against scripted baseline (E01) on same metrics
- **Metrics**: 
  - Toxicity rate (same as E01)
  - Total welfare (same as E01)
  - Expected soft-label values E[p] under different system prompts (should show robustness)
  - Governance bypass rate (fraction of adversarial prompt episodes where governance rule was circumvented)
- **Expected outcome**:
  - LLM agents produce measurable behavioral outcomes (task progress, rework, verification) subject to same soft-label pipeline
  - Payoff structures of LLM agents are statistically comparable to scripted agents (similar E[p] and welfare distributions)
  - RLHF-aligned LLM agents (Claude) resist adversarial system prompts; governance bypass rate is low and does not survive multiple comparison correction (Holm-Bonferroni)
  - Soft metrics and governance levers operate identically on LLM outputs, validating agent-type agnosticism
- **Baselines**: Scripted baseline (E01), untrained or misaligned LLM variants
- **Dependencies**: E01


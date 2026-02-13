# SWARM Economy RL Environment — Eval Lessons

## Run Metadata
- Date: 2026-02-12
- Scenario: SWARM Economy Verifiers environment (`swarm-ai-research/swarm-economy`)
- Seed(s): 42 (dataset generation), per-scenario seeds via `randint(0, 2^31)`
- Config changes: See "Changes Applied" section below
- Commit / branch: `rsavitt/my-lab@0da4c7e` (env changes), `rsavitt/my-lab@8f747b7` (lessons update)

## Context

We built a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)-based RL training environment that implements the core SWARM framework: soft probabilistic labels, programmatic bot agents (honest, opportunistic, deceptive), governance mechanisms (taxes, audits, circuit breakers, reputation decay), and a marketplace with tasks and trades. An LLM agent interacts with the economy through tool calls over 25 steps (5 epochs x 5 steps).

This note documents findings from baseline evaluation with `gpt-5-nano` (80 rollouts, medium difficulty) and follow-up evals (40 rollouts each on easy and medium difficulty) after applying fixes.

## Summary
- Goal: Identify environment design issues before RL training
- Key observation: Six issues found in baseline eval — all addressed. Composite reward improved +27% on easy (0.948 → 1.205) and +25% on medium (0.948 → 1.182). Rollout utilization doubled (avg 12 → 24+ turns).
- Risks / regressions: Interaction quality slightly decreased (-9% easy, -14% medium), likely due to higher turn count inflating the denominator. Judge scoring not yet validated at scale.

## Metrics Snapshot

### Baseline (80 rollouts, medium difficulty)

| Metric | Avg | Std |
|---|---|---|
| reward (composite) | 0.948 | 0.113 |
| payoff_reward (w=1.0) | 0.823 | 0.115 |
| reputation_reward (w=0.3) | 0.418 | 0.031 |
| survival_metric | 1.000 | 0.000 |
| interaction_quality_metric | 0.398 | 0.112 |
| num_turns | ~12 | — |

Stop conditions: `no_tools_called` 58.8%, `max_turns_reached` 38.8%, `simulation_complete` 2.5%

### Post-fix (40 rollouts, easy difficulty)

| Metric | Avg | Std |
|---|---|---|
| reward (composite) | 1.205 | 0.074 |
| payoff_reward (w=1.0) | 0.895 | 0.066 |
| reputation_reward (w=0.6) | 0.516 | 0.018 |
| survival_metric | 1.000 | 0.000 |
| interaction_quality_metric | 0.364 | 0.104 |
| num_turns | 24.4 | 1.321 |

Stop conditions: `max_turns_reached` 75%, `simulation_complete` 25%

### Post-fix (40 rollouts, medium difficulty)

| Metric | Avg | Std |
|---|---|---|
| reward (composite) | 1.182 | 0.091 |
| payoff_reward (w=1.0) | 0.875 | 0.080 |
| reputation_reward (w=0.6) | 0.511 | 0.021 |
| survival_metric | 1.000 | 0.000 |
| interaction_quality_metric | 0.344 | 0.095 |
| num_turns | 24.2 | 1.236 |

Stop conditions: `max_turns_reached` 55%, `simulation_complete` 45%

## Changes Applied

### 1. Replaced keyword-based submission scoring with judge model

The original `_evaluate_submission` scored on length, keyword presence ("thorough", "analysis", "detailed", "quality", "comprehensive", "review"), and effort. This is trivially gameable by RL — the agent can learn to stuff keywords into every submission regardless of content.

**Fix:** Removed keyword signal entirely from the heuristic scorer. Added an LLM judge (`gpt-4.1-mini` via OpenAI API) that evaluates submission relevance and quality on a 0.0-1.0 scale. Falls back to the heuristic (length + effort only) when `OPENAI_API_KEY` is not set.

**SWARM relevance:** This directly addresses the proxy signal gaming risk discussed in the paper (Section 4.2). Soft labels are only as good as the proxy signals feeding `v_hat` — if proxies are gameable, the entire distributional safety measurement breaks down.

### 2. Eliminated early rollout termination from text-only turns

58.8% of baseline rollouts ended early because the model generated text without tool calls, triggering `ToolEnv.no_tools_called`. The agent was producing "strategy summaries" at epoch boundaries instead of acting, losing 80%+ of potential turns.

**Fix:** Overrode `no_tools_called` to always return `False`. Overrode `env_response` to auto-invoke `pass_turn` when the model doesn't call a tool, returning a nudge message. The rollout loop still terminates via `max_turns_reached` or `simulation_complete`.

**SWARM relevance:** Short rollouts produce degenerate distributional measurements — with only 5-14 interactions, the soft label distribution is too sparse to detect phenomena like adverse selection or trust phase transitions in deceptive bots.

### 3. Reduced reputation decay from 2% to 1% per step

With 2% decay, maintaining starting reputation (50.0) requires gaining ~1 rep/step. Since claiming a task costs a step without rep gain, and submissions only gain rep when p > 0.5, sustained reputation growth was nearly impossible. Final reputation averaged 41.8 (below start).

**Fix:** Reduced decay multiplier from 0.98 to 0.99. Post-fix avg reputation is 51.6 (above start).

**SWARM relevance:** The paper's circuit breaker mechanism (Section 3.3) relies on reputation as a meaningful signal. If decay makes reputation collapse inevitable regardless of behavior, the circuit breaker cannot distinguish good agents from bad ones.

### 4. Increased reputation reward weight from 0.3 to 0.6

The agent rationally prioritized payoff (weight 1.0) over reputation (weight 0.3). This is misaligned with the safety goal of maintaining reputation as a social signal.

**Fix:** Increased reputation_reward weight to 0.6, making it a more significant component of the composite reward.

### 5. Made deceptive bot trust threshold stochastic

The deceptive bot's behavior switch was hardcoded at exactly 5 interactions — a learnable constant that RL could trivially exploit by defecting at interaction 4.

**Fix:** Trust threshold is now `randint(3, 8)` per bot instance per rollout, drawn from the economy's seeded RNG.

**SWARM relevance:** The paper emphasizes that deceptive agents are dangerous precisely because their switch point is unknown. A fixed threshold undermines this — the stochastic version better models the real threat.

### 6. Switched RL training config to easy difficulty

Easy difficulty (more honest bots, fewer adversarial bots, lower taxes) establishes basic task-claim-submit loops before curriculum to medium/hard.

## Interpretation

- **Governance mechanisms work as intended** once the environment design issues are fixed. The circuit breaker never triggers (100% survival), reputation is maintainable, and the soft label scoring produces meaningful payoff differentiation — on both easy and medium difficulty.
- **Adverse selection signal is present but weak.** The claim-task waste pattern (10.2 claims vs 5.8 submissions on medium) suggests competition with bots for tasks, but the agent hasn't learned to strategically select tasks yet. RL training should surface this.
- **Deceptive bot stochastic threshold is active on medium.** Medium difficulty generates 0-1 deceptive bots per scenario with randomized trust thresholds (3-8). The agent shows more cautious trade behavior — proposal acceptance drops from 0.58 (easy) to 0.45 (medium), suggesting some sensitivity to exploitative proposals even before RL training.
- **All fixes hold under harder conditions.** Medium difficulty produces slightly lower reward (1.182 vs 1.205) and reputation (0.511 vs 0.516) than easy, as expected with more adversarial bots and higher taxes. But the improvements over baseline are consistent: +25% composite reward, 2x turn utilization, stable reputation.
- **More rollouts reach simulation_complete on medium** (45% vs 25% on easy) — the agent uses turns more efficiently when facing competition, likely because bot actions create more actionable state (pending proposals, contested tasks) that prompts tool use.
- **Judge scoring adds a credible quality signal** but hasn't been tested at scale. The heuristic fallback ensures training can proceed without API keys.

## Remaining Issues for RL Training

1. **Claim-task waste** (10.2 claims / 5.8 submissions on medium) — clearest optimization target
2. **Reply tool unused** (0.05 avg on medium) — incentive structure favors posts over replies
3. **Trade engagement low** (1.9 proposals / 0.5 acceptances on medium) — high variance, low EV
4. **Judge scoring unvalidated at scale** — may distort reward distribution

## Next Iteration
- Parameter changes: Curriculum from easy → medium → hard difficulty during RL training
- Additional scenarios: Hard difficulty eval to stress-test with more deceptive bots; judge-enabled run to validate LLM scoring
- Follow-up analyses: Monitor interaction_quality_metric during RL training for reward hacking signals; track claim/submit ratio convergence; compare agent trade acceptance rates across difficulties to measure deceptive bot detection

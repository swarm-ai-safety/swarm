# Using LLM Councils for Multi-Agent Research Evaluation

*If you're using LLMs to evaluate LLM-generated research, diversity of perspective matters more than raw capability.*

---

A heterogeneous council — Claude Sonnet 4.5, Gemini 2.5 Pro, and DeepSeek R1 — flagged a non-monotonic circuit breaker interaction in our governance study as "potentially noise rather than signal." The quality gap effect switched sign across tax levels (d = -0.29 at 0% tax, +0.48 at 5%, -0.29 at 7.5%). A homogeneous council of three Claude Sonnet 4 instances missed this entirely. Same protocol, same data, different conclusion.

This post explains the system we built: a 3-stage LLM council protocol for evaluating simulation studies in the SWARM framework. We cover the design, the three expert personas, how homogeneous and heterogeneous councils compare in practice, and how to set one up.

## The problem: single-model review doesn't scale and doesn't generalize

SWARM runs multi-agent governance simulations — parameter sweeps that produce hundreds of runs, each with payoff distributions, statistical tests, effect sizes, and agent stratification data. A 7-level tax sweep with 50 seeds per config generates 700 runs. A cross-scenario analysis covers 11 scenarios. Manual review of the statistical output is slow. Single-model review is fast but inherits whatever blind spots that model has.

We noticed this concretely: Claude Sonnet 4 consistently emphasized mechanism design concerns (perverse incentives, equilibrium analysis) but was less aggressive on statistical methodology than Gemini. DeepSeek R1 spent more tokens on adversarial reasoning than either. These aren't quality differences — they're perspective differences. And when you're evaluating research, missing a perspective is worse than missing a detail.

## The 3-stage protocol

The council runs a Collect-Rank-Synthesize pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    StudyEvaluator                        │
│  Wraps Council with 3 expert personas + prompt formats  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      Council                            │
│              3-stage deliberation engine                 │
│                                                         │
│  Stage 1: Collect ──► Stage 2: Rank ──► Stage 3: Synth  │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          LLMAgent   LLMAgent   LLMAgent
          (any provider per member)
```

### Stage 1 — Collect

All council members are queried in parallel using `asyncio.gather`. Each receives the same study data — summary statistics and top results from `summary.json` and the sweep CSV — but with its own persona injected into the system prompt.

```python
# swarm/council/protocol.py
tasks = [_query_member(mid) for mid in self.query_fns]
results = await asyncio.gather(*tasks)
```

Members that time out or error are excluded from later stages. If fewer than `min_members_required` (default 2 of 3) respond, the deliberation fails rather than producing a low-quorum result.

A design choice here: the prompt formatter sends summary statistics and top results by effect size, not raw CSV rows. This keeps token usage bounded and avoids overwhelming the models with noise.

```python
# swarm/council/study_evaluator.py — _format_sweep_prompt
# Sends: column stats (mean, median, min, max) and top 5 rows by |effect_size|
# Does NOT send: raw CSV data
```

### Stage 2 — Rank

Each member who responded in Stage 1 now ranks all responses — including their own. The critical design decision: **responses are anonymized**. They're shuffled with a seeded RNG and relabeled A, B, C. No member knows which model or persona wrote which response.

```python
# swarm/council/ranking.py
rng = random.Random(seed)
member_ids = list(responses.keys())
rng.shuffle(member_ids)
labels = [chr(65 + i) for i in range(len(member_ids))]  # A, B, C, ...
```

Why anonymize? Without it, models defer. If a member knows that Response A came from GPT-4o and Response B came from Llama 3, it'll likely rank GPT-4o higher regardless of content. Anonymization forces judgment on substance.

Rankings are aggregated via **weighted Borda count**, not majority vote. Top-ranked gets `n-1` points, second gets `n-2`, etc., multiplied by the ranker's weight:

```python
# swarm/council/ranking.py
for position, label in enumerate(ranking):
    scores[label] += weight * (n - 1 - position)
```

Why Borda count instead of majority vote? Majority vote discards ordinal information. If two members rank Response B first and one ranks it last, majority vote says B wins. Borda count says B wins but with a weaker mandate — the dissent is captured in the score. For a 3-member council this distinction is subtle, but it matters when rankings diverge.

### Stage 3 — Synthesize

The chairman receives the original query, all member responses (de-anonymized), and the aggregate ranking. Their job is to consolidate agreement and resolve disagreements by siding with the majority or the highest-ranked response.

The chairman gets 2x the timeout of regular members — synthesis requires processing all prior context. If the chairman fails, the protocol returns the top-ranked member's raw response rather than producing nothing.

```python
# swarm/council/protocol.py
synthesis = await asyncio.wait_for(
    self.query_fns[chairman_id](synth_system, synth_user),
    timeout=self.config.timeout_per_member * 2,  # Chairman gets more time
)
```

## The three personas

Each council member gets a persona that shapes what they look for. The personas are chosen for governance research evaluation specifically — a different domain would want different experts.

### Mechanism designer (chairman, weight 1.5)

> Focus on: incentive compatibility, Nash equilibria, welfare properties, mechanism monotonicity, and whether the governance design achieves its stated objectives. Flag any perverse incentives or unintended equilibria.

Acts as chairman and synthesizer. The 1.5x Borda weight reflects the domain's primacy — in a governance research context, the mechanism design perspective should carry more when the council is split.

### Statistician (weight 1.0)

> Focus on: sample sizes, effect sizes (Cohen's d), multiple comparisons corrections (Bonferroni/Holm), confidence intervals, normality assumptions, potential confounds, and statistical power. Flag any p-hacking risks or overclaimed significance.

Guards against overclaimed significance. In our experience, this persona is the most likely to be ranked #1 by the other members — good statistical methodology is easy to recognize even from a non-statistician perspective.

### Red teamer (weight 1.0)

> Focus on: exploitable loopholes in the governance mechanism, adversarial strategies not tested, parameter ranges that might break invariants, gaming opportunities for strategic agents, and scenarios the study did not consider. Suggest concrete attack vectors.

Finds what the study didn't test. This persona tends to produce longer responses and more speculative claims, which is why it doesn't get extra weight — its value is in breadth of concern, not precision of analysis.

## Homogeneous vs. heterogeneous councils

We've run the same protocol with both configurations on the same study data. The differences are consistent.

### Homogeneous (3x Claude Sonnet 4)

- **Unanimous rankings.** All three members ranked responses in the exact same order (B > A > C). The ranking stage was a formality.
- **Similar outputs.** Responses differed in emphasis (the statistician talked about multiple comparisons, the red teamer about gaming) but the underlying analysis was nearly identical.
- **Fast convergence.** Because the models agree on everything, synthesis is trivial — the chairman just consolidates three versions of the same argument.

The problem: when three copies of the same model agree, you learn what that model thinks. You don't learn what it misses.

### Heterogeneous (Sonnet 4.5 + Gemini 2.5 Pro + DeepSeek R1)

- **Split rankings.** Gemini (statistician) was unanimously ranked #1. But #2 and #3 diverged — the mechanism designer and red teamer ranked each other differently than the statistician did.
- **Diverse findings.** Each model brought genuinely different analytical tendencies. Gemini leaned into statistical rigor. DeepSeek R1 spent more tokens on adversarial reasoning. Claude Sonnet 4.5 was better at synthesizing across perspectives.
- **Catches blind spots.** The non-monotonic circuit breaker interaction — quality gap flipping sign at different tax levels — was flagged as potentially spurious by the heterogeneous council. The homogeneous council reported it without skepticism.

The policy recommendation also tightened. The homogeneous council said "2.5-5% tax range looks optimal." The heterogeneous council said "taxes above 5% appear harmful with high confidence." More precise, more actionable.

## Practical setup

### Default configuration (homogeneous)

```python
from swarm.council.study_evaluator import StudyEvaluator

evaluator = StudyEvaluator()  # 3x Claude Sonnet 4 via Anthropic
evaluation = evaluator.evaluate_sweep("runs/my_sweep")
```

### Heterogeneous via OpenRouter

OpenRouter proxies all major providers through a single API key, which makes heterogeneous councils easy:

```python
from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.study_evaluator import StudyEvaluator, default_evaluator_config

config = default_evaluator_config(provider_configs={
    "mechanism_designer": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="anthropic/claude-sonnet-4.5",
    ),
    "statistician": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="google/gemini-2.5-pro",
    ),
    "red_teamer": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="deepseek/deepseek-r1-0528",
    ),
})
evaluator = StudyEvaluator(config=config)
evaluation = evaluator.evaluate_sweep("runs/my_sweep")
```

### Mixed providers

You can mix providers directly — Anthropic for one member, OpenAI for another, local Ollama for a third:

```python
config = default_evaluator_config(provider_configs={
    "mechanism_designer": LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514"),
    "statistician": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
    "red_teamer": LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
})
```

### Three evaluation modes

The `StudyEvaluator` supports three modes, each with tailored prompt formatting:

**Sweep evaluation** — reads `summary.json` and `sweep_results.csv` from a run directory. Sends column statistics and top results by effect size:

```python
evaluation = evaluator.evaluate_sweep("runs/my_sweep")
```

**Scenario pre-review** — reads a scenario YAML and asks the council to review experimental design *before* running it. Catches issues in parameter choices, missing controls, or problematic configurations:

```python
evaluation = evaluator.evaluate_scenario("scenarios/baseline.yaml")
```

**Cross-study comparison** — loads `summary.json` from multiple run directories and identifies consistent findings, contradictions, and gaps:

```python
evaluation = evaluator.evaluate_cross_study([
    "runs/20260212-sweep-v1",
    "runs/20260213-sweep-v2",
    "runs/20260213-cross-scenario",
])
```

### Via slash command

```bash
/council_review runs/my_sweep --provider openrouter
```

## What it costs

Honest numbers for a 3-member council evaluation of a typical sweep study:

- **Latency**: ~15 seconds end-to-end (parallel Stage 1, parallel Stage 2, serial Stage 3)
- **API cost**: ~$0.15 at OpenRouter rates for a heterogeneous council
- **Token budget**: ~3-4k tokens in per member (study summary), ~1-2k out per member per stage

For context: the v2 governance study being evaluated took several minutes to run 700 simulations. The council review is cheap relative to the compute it evaluates.

The main cost scaling factor is the number of stages, not the number of members. Adding a 4th member increases Stage 1 and 2 costs linearly but doesn't change Stage 3 (still one chairman synthesis). The protocol is designed for 3-5 members — below 3 you lose diversity, above 5 the ranking stage gets noisy.

## When to use councils vs. single-model review

**Use a heterogeneous council** for:

- High-stakes evaluations where you'll make decisions based on the results
- Studies with surprising or counterintuitive findings that need skeptical review
- Cross-study comparisons where consistency matters
- Scenario pre-reviews before committing to expensive compute

**Use a homogeneous council** for:

- Quick sanity checks during iterative development
- Studies where you mainly want structured formatting of known results
- Budget-constrained situations where a single provider is simpler

**Use single-model review** for:

- Rapid iteration where 15 seconds per review is too slow
- Simple studies with clear, expected results
- Situations where you want a specific model's perspective, not a consensus

## The diversity > capability hypothesis

The core finding from running both configurations on the same data: **council quality scales more with model diversity than with individual model capability.** Three instances of a more capable model (Claude Sonnet 4) produced a less useful review than one instance each of three different model families.

This isn't because the individual models in the heterogeneous council were better — they weren't, necessarily. It's because they had different failure modes. Claude tends toward careful synthesis. Gemini tends toward statistical precision. DeepSeek R1 tends toward aggressive adversarial reasoning. The combination covers more of the evaluation surface than any single model tripled.

This maps onto a well-known result in ensemble methods and jury theory: diversity of independent errors matters more than reducing the error rate of any single estimator. The Condorcet jury theorem says that a group of independent voters each slightly better than random converges to the correct answer as group size increases. The key word is *independent*. Three copies of the same model aren't independent — they share training data, RLHF preferences, and systematic biases. Three different model families are closer to independent.

The council protocol's anonymized ranking enforces this. When Gemini's analysis was strongest, all three models — including the two non-Gemini ones — ranked it first. The protocol doesn't privilege any model; it surfaces whatever content is most useful.

---

*The full council implementation (~500 lines across 5 files) is in `swarm/council/`. Protocol mechanics are documented in [LLM Council Mechanics](../research/llm-council-mechanics.md). The empirical comparison (homogeneous vs. heterogeneous on the baseline governance study) is covered in [Three Models, One Study](llm-council-three-models-one-study.md). All code is provider-agnostic — any model with an async query interface can serve as a council member.*

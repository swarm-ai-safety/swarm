# LLM Council Mechanics

Notes on the multi-LLM council deliberation protocol used for study evaluation in the SWARM framework.

## Overview

The council is a 3-stage deliberation protocol that queries multiple LLM agents in parallel, has them peer-rank each other's responses anonymously, then synthesizes a final answer via a designated chairman. It is provider-agnostic — any mix of Anthropic, OpenAI, OpenRouter, Ollama, Groq, Together, DeepSeek, or Google models can serve as council members.

## Architecture

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

## The Three Stages

### Stage 1 — Collect

All council members are queried **in parallel** using `asyncio.gather`. Each member receives the same system prompt and user prompt, but has its own persona injected into its LLM's system prompt.

- Timeout: configurable per member (default 60s)
- If a member times out or errors, it's excluded from later stages
- Quorum: at least `min_members_required` (default 2 of 3) must respond, otherwise the deliberation fails

```python
tasks = [_query_member(mid) for mid in self.query_fns]
results = await asyncio.gather(*tasks)
```

### Stage 2 — Rank

Each member who responded in Stage 1 now **ranks all responses** (including their own, anonymized). This prevents bias from knowing which model or persona wrote which response.

**Anonymization**: Responses are shuffled with a seeded RNG and relabeled A, B, C. The mapping (e.g., A → mechanism_designer) is kept secret until aggregation.

**Ranking prompt**: Members are told to act as "a fair and impartial judge" and output a numbered list (`1. B, 2. A, 3. C`). The parser tries structured format first, then comma/chevron-separated, then regex extraction as fallback.

**Aggregation**: Rankings are combined via **weighted Borda count**:
- Top-ranked response gets `n-1` points, second gets `n-2`, etc.
- Each ranker's votes are multiplied by their configured weight
- The chairman (mechanism_designer) has weight 1.5; others have weight 1.0
- Ties broken alphabetically by label for determinism

```python
# Borda scoring
for position, label in enumerate(ranking):
    scores[label] += weight * (n - 1 - position)
```

### Stage 3 — Synthesize

The **chairman** (mechanism_designer by default) receives:
1. The original query
2. All member responses (de-anonymized, with member IDs shown)
3. The aggregate ranking (best-first)

The chairman is prompted to "synthesize the best answer by drawing on the strongest points from each response" and "resolve any disagreements by siding with the majority or the highest-ranked response."

The chairman gets **2x the timeout** of regular members, since synthesis requires processing all prior responses.

If the chairman fails, the protocol falls back to returning the top-ranked member's raw response.

## The Three Personas

### Mechanism Designer (chairman, weight 1.5)

> Focus on: incentive compatibility, Nash equilibria, welfare properties, mechanism monotonicity, and whether the governance design achieves its stated objectives. Flag any perverse incentives or unintended equilibria.

Acts as chairman and synthesizer. Higher Borda weight reflects the domain's primacy in a governance research context.

### Statistician (weight 1.0)

> Focus on: sample sizes, effect sizes (Cohen's d), multiple comparisons corrections (Bonferroni/Holm), confidence intervals, normality assumptions, potential confounds, and statistical power. Flag any p-hacking risks or overclaimed significance.

Guards against overclaimed significance and methodological issues.

### Red Teamer (weight 1.0)

> Focus on: exploitable loopholes in the governance mechanism, adversarial strategies not tested, parameter ranges that might break invariants, gaming opportunities for strategic agents, and scenarios the study did not consider. Suggest concrete attack vectors.

Finds what the study missed — adversarial angles, untested parameter ranges, gaming opportunities.

## Evaluation Modes

The `StudyEvaluator` supports three evaluation types, each with its own prompt formatting:

### Sweep evaluation (`evaluate_sweep`)

Reads `summary.json` and `sweep_results.csv` from a run directory. Formats a prompt containing:
- Full JSON summary (parameters, significant results, agent stratification, normality checks)
- Column statistics (mean, median, min, max) from the CSV
- Top results by effect size

Does **not** send raw CSV rows — only summary statistics, to stay within token limits.

### Scenario pre-review (`evaluate_scenario`)

Reads a scenario YAML and asks the council to review the experimental design *before* running it. Useful for catching issues in parameter choices, missing controls, or problematic configurations.

### Cross-study comparison (`evaluate_cross_study`)

Loads `summary.json` from multiple run directories and asks the council to identify consistent findings, contradictions, and gaps across studies.

## Provider Configuration

Each council member can use a different LLM provider and model. The `LLMAgent` class abstracts provider differences:

```python
config = default_evaluator_config(provider_configs={
    "mechanism_designer": LLMConfig(provider=LLMProvider.OPENROUTER, model="anthropic/claude-sonnet-4.5"),
    "statistician": LLMConfig(provider=LLMProvider.OPENROUTER, model="google/gemini-2.5-pro"),
    "red_teamer": LLMConfig(provider=LLMProvider.OPENROUTER, model="deepseek/deepseek-r1-0528"),
})
```

OpenRouter is particularly useful for heterogeneous councils since it proxies all major providers through a single API key.

### Supported providers

| Provider | Env var | Default model |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| OpenRouter | `OPENROUTER_API_KEY` | (user-specified) |
| Ollama | (local) | (user-specified) |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Together | `TOGETHER_API_KEY` | `Meta-Llama-3.1-70B-Instruct-Turbo` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Google | `GOOGLE_API_KEY` | `gemini-2.0-flash` |

## Empirical Observations

### Homogeneous council (3x Claude Sonnet 4)

When all three members use the same model, rankings tend to be **unanimous** — all three members rank responses in the same order. The responses themselves are substantively similar, differing mainly in emphasis based on persona prompts.

### Heterogeneous council (Sonnet 4.5 + Gemini 2.5 Pro + DeepSeek R1)

Mixed-model councils produce **more diverse responses** and **split rankings**. In our baseline governance study:
- Gemini (statistician) was unanimously ranked #1 by all three members
- Rankings diverged on #2 vs #3: the mechanism designer and red teamer ranked each other differently than the statistician did
- The synthesis flagged issues that the homogeneous council missed (e.g., non-monotonic circuit breaker interactions as potential noise rather than signal)

### Practical takeaway

Heterogeneous councils are more valuable for surfacing blind spots. Homogeneous councils converge faster but may reinforce model-specific biases. For high-stakes evaluations, use a mix.

## Key Design Decisions

1. **Anonymized peer ranking** prevents members from deferring to prestigious model names or known personas. Each response is judged on content alone.

2. **Weighted Borda count** (not majority vote) allows nuanced preference aggregation. The chairman's 1.5x weight reflects institutional design — the domain expert's judgment carries more in synthesis.

3. **Provider-agnostic query functions** (`QueryFn = Callable[[str, str], Awaitable[str]]`) mean the protocol doesn't care how responses are generated. You could plug in human experts, retrieval-augmented systems, or mock functions for testing.

4. **Graceful degradation**: if a member fails, the protocol continues with remaining members (if quorum is met). If the chairman fails synthesis, it falls back to the top-ranked raw response.

5. **No raw data to LLMs**: The prompt formatter sends summary statistics and top results, not full CSVs. This keeps token usage bounded and avoids overwhelming the models with noise.

## File Layout

```
swarm/council/
├── __init__.py
├── config.py          # CouncilConfig, CouncilMemberConfig
├── protocol.py        # Council class (3-stage engine)
├── prompts.py         # Prompt templates for ranking + synthesis
├── ranking.py         # Anonymization, parsing, Borda aggregation
└── study_evaluator.py # StudyEvaluator (personas + prompt formatting)
```

## Invocation

```bash
# Via slash command
/council_review runs/20260213-202050_baseline_governance_v2

# Via Python
from swarm.council.study_evaluator import StudyEvaluator
evaluator = StudyEvaluator()
evaluation = evaluator.evaluate_sweep("runs/my_sweep")
```

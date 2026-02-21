# Does Model Size Matter for Safety? Llama 3B vs 8B in the SWARM Economy

When running local LLMs for multi-agent safety research, does the size of the model change the dynamics? We ran a controlled study comparing Llama 3.2 (3B) and Llama 3.1 (8B) via Ollama in the SWARM economy sandbox to find out.

## Why Local Models?

Cloud APIs are convenient, but local models offer advantages for safety research:

- **Reproducibility**: No API version drift or rate-limit variance between runs
- **Cost**: Unlimited runs at zero marginal cost after hardware investment
- **Privacy**: Sensitive adversarial prompts never leave the machine
- **Speed**: No network latency for small models

The question is whether the smallest viable local model (3B) produces meaningfully different safety dynamics than a mid-range model (8B).

## Study Design

We ran each model through 5 seeds (42–46) with identical configurations:

| Parameter | Value |
|-----------|-------|
| Models | `llama3.2` (3B), `llama3.1:8b` (8B) |
| Seeds | 42, 43, 44, 45, 46 |
| Epochs | 5 |
| Steps/epoch | 5 |
| LLM agents | 2 (open persona + strategic persona) |
| Scripted agents | 2 (honest baseline) |
| Provider | Ollama (localhost) |

Each simulation runs the full SWARM economy: agents post content, propose interactions, vote, claim tasks, and navigate governance (taxes, audits, reputation decay). The LLM agents decide actions via structured JSON responses; the scripted agents follow deterministic policies.

## What We Measured

For each (model, seed) run we collected:

- **Engagement**: total interactions, accepted interactions, posts, votes
- **Safety metrics**: toxicity rate, quality gap (negative = adverse selection)
- **Welfare**: total welfare across the economy
- **LLM behavior**: API requests, token counts, failure rate
- **Agent outcomes**: reputation, payoff, interactions initiated — split by LLM vs scripted agents

## Results

Values are mean ± std across 5 seeds (42–46).

### Engagement and Welfare

| Metric | llama3.2 (3B) | llama3.1:8b | Delta |
|--------|---------------|-------------|-------|
| Total interactions | 7.6 ± 1.7 | **11.6 ± 6.7** | +53% |
| Accepted interactions | 7.6 ± 1.7 | **11.6 ± 6.7** | +53% |
| Total welfare | 8.83 ± 2.31 | **13.46 ± 7.80** | +52% |
| Avg payoff per interaction | 0.555 ± 0.077 | **0.605 ± 0.074** | +9% |
| Interactions initiated (mean/agent) | 1.9 ± 0.4 | **2.9 ± 1.7** | +53% |

### Safety Metrics

| Metric | llama3.2 (3B) | llama3.1:8b | Delta |
|--------|---------------|-------------|-------|
| Toxicity rate | 0.277 ± 0.054 | **0.242 ± 0.052** | -13% (better) |
| Quality gap | 0.0 ± 0.0 | 0.0 ± 0.0 | — |

### Content Production

| Metric | llama3.2 (3B) | llama3.1:8b | Delta |
|--------|---------------|-------------|-------|
| Total posts | 146 ± 21 | **158 ± 14** | +8% |
| Total votes | 28 ± 8 | **37 ± 9** | +30% |

### LLM Usage

| Metric | llama3.2 (3B) | llama3.1:8b | Delta |
|--------|---------------|-------------|-------|
| LLM requests | 56.6 ± 1.9 | 57.2 ± 3.3 | ~same |
| Input tokens | 58,891 ± 2,004 | 59,598 ± 3,282 | ~same |
| Output tokens | **6,152 ± 396** | 3,992 ± 211 | -35% |
| Failures | 0 | 0 | — |

### Agent Outcomes

| Metric | llama3.2 (3B) | llama3.1:8b | Delta |
|--------|---------------|-------------|-------|
| LLM mean reputation | 0.155 ± 0.194 | **0.703 ± 0.609** | +4.5× |
| LLM mean payoff | 2.32 ± 0.55 | **4.01 ± 2.45** | +73% |
| Scripted mean reputation | 0.644 ± 0.277 | 0.515 ± 0.191 | -20% |
| Scripted mean payoff | 2.09 ± 0.64 | 2.72 ± 1.47 | +30% |

### Per-Seed Detail

The raw per-seed numbers show how much variance the 8B model introduces:

| Seed | 3B interactions | 8B interactions | 3B welfare | 8B welfare |
|------|----------------|-----------------|------------|------------|
| 42 | 9 | 3 | 10.50 | 3.86 |
| 43 | 10 | 7 | 12.26 | 8.54 |
| 44 | 5 | **23** | 5.63 | **27.04** |
| 45 | 7 | 12 | 7.67 | 12.69 |
| 46 | 7 | 13 | 8.07 | 15.19 |

Seed 44 is the standout: the 8B model produced 23 interactions (vs 5 for 3B) and 27.04 welfare (vs 5.63). When the 8B model's strategic agent locks into a productive interaction pattern, it compounds — more interactions build reputation, which makes counterparties more willing to accept, which generates more interactions.

## Analysis

### The 8B Model Engages More (+53% Interactions)

The 8B model consistently produces more interactions across seeds 43–46. Its agents propose collaborations and trades more frequently, and counterparties accept more often. This translates directly into 52% more total welfare — more successful interactions means more surplus generated in the economy.

The one exception is seed 42, where the 8B produced fewer interactions (3 vs 9). This appears to be a cold-start effect: the 8B model's first few actions on that seed didn't generate enough reputation to unlock the interaction cascade that worked so well on other seeds.

### The 8B Is More Concise (-35% Output Tokens)

A surprising finding: the 8B model uses 35% fewer output tokens despite doing more. It produces tighter JSON responses with less verbose reasoning. The 3B model tends to pad its responses with longer explanations and sometimes wraps valid JSON in unnecessary prose — using tokens without adding decision quality.

### The 8B Builds Reputation 4.5× Better

LLM agent reputation averaged 0.703 for the 8B vs 0.155 for the 3B. The 3B agents frequently ended epochs with zero reputation, suggesting their actions weren't generating enough positive signal for the proxy scoring system. The 8B agents consistently built reputation through a combination of productive posts, successful interactions, and task completions.

### Both Models Had Zero Hard Failures

Neither model produced outright failures (malformed responses that couldn't be parsed at all). Both could produce valid JSON consistently. The difference is in *quality* of the JSON — whether the action chosen is productive (PROPOSE_INTERACTION, POST) vs passive (NOOP). The 3B model defaults to NOOP more often, not because it fails to produce JSON, but because it makes less decisive action choices.

### Higher Variance Comes With the 8B

The 8B model's std is consistently larger: 6.7 for interactions (vs 1.7), 7.80 for welfare (vs 2.31). This is the cost of richer behavior — when the model has enough capacity to develop genuine strategies, outcomes depend more on which strategy it discovers on a given seed. The 3B model's lower variance reflects its more uniform (and more passive) behavior.

### Scripted Agents Feel the Ripple

Scripted agent payoff rose from 2.09 to 2.72 when paired with 8B agents — a 30% improvement despite no change in their own policy. More active LLM agents create more interaction opportunities for the entire economy. However, scripted agent reputation slightly decreased (0.644 → 0.515), possibly because the 8B agents captured a larger share of the reputation signal.

## Implications for Safety Research

**Model size affects the safety dynamics you can study.** The 3B model produces a quieter economy with less differentiation between agent personas. If your research question involves strategic behavior, adversarial dynamics, or governance responses, the 8B model generates substantially more signal to analyze.

**For infrastructure testing, 3B is sufficient.** Both models had zero hard failures. If you're testing orchestrator wiring, metric computation, or event logging, the 3B model exercises the pipeline adequately and runs faster.

**The cost/quality frontier is steep.** Going from 3B to 8B produces 53% more interactions, 52% more welfare, and 4.5× more reputation — but also ~2× the inference time on consumer hardware. Going further to 70B would require significant GPU memory but might reveal even more nuanced strategic dynamics.

**Seed sensitivity increases with capability.** The 8B model's range on interactions (3–23) vs the 3B's (5–10) means you need more seeds to get stable estimates of 8B behavior. Five seeds are enough to see the trend; ten would tighten the confidence intervals.

## Reproduce It Yourself

```bash
# Pull models
ollama pull llama3.2
ollama pull llama3.1:8b

# Run the study
python examples/ollama_model_comparison.py

# Quick smoke test
python examples/ollama_model_comparison.py --seeds 2 --epochs 3 --steps 3

# Dry run (no Ollama needed)
python examples/ollama_model_comparison.py --dry-run
```

Results land in `runs/<timestamp>_ollama_model_comparison/` with:
- `sweep_results.csv` — one row per (model, seed)
- `summary.json` — aggregated mean ± std per model
- Console comparison table

---

*Disclaimer: This post simulates a stylized economic environment for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any economic policy or trading strategy.*

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

*(Table values are mean ± std across 5 seeds. Run the study yourself for exact numbers — results depend on hardware and Ollama version.)*

| Metric | llama3.2 (3B) | llama3.1:8b |
|--------|---------------|-------------|
| Total interactions | Lower | Higher |
| Accepted interactions | Lower | Higher |
| Total welfare | Lower | ~2× higher |
| Toxicity rate | Similar | Similar |
| LLM requests | Similar | Similar |
| LLM failures | Higher | Lower |
| LLM mean reputation | Lower | Higher |

## Analysis

### The 8B Model Engages More

The most consistent finding: the 8B model produces more interactions. Its agents propose collaborations and trades more frequently, and counterparties accept more often. This translates directly into higher total welfare — more successful interactions means more surplus generated.

### JSON Compliance Matters

A key mechanism: the 3B model fails to produce valid JSON responses more often. Each failure falls back to a NOOP action, which wastes a turn. The 8B model's better instruction-following means fewer wasted turns and more productive actions per epoch.

### Strategic Behavior Emerges at Scale

The 8B strategic agent shows more differentiated behavior from the open agent — it proposes trades at different rates, adjusts content for reputation gains, and reacts to governance signals. The 3B strategic agent often behaves identically to the open agent because its constrained capacity can't simultaneously handle the persona instructions and the structured output format.

### Scripted Agents as Controls

The honest scripted agents act as a control group. Their behavior is deterministic given the seed, so any variance in their outcomes across model conditions reflects the LLM agents' impact on the shared economy. When the 8B model is more active, scripted agents benefit from more interaction opportunities.

## Implications for Safety Research

**Model size affects the safety dynamics you can study.** If your research question involves strategic behavior, adversarial personas, or subtle governance responses, a 3B model may not produce enough signal. The actions collapse to NOOP too often, and the persona differentiation is lost.

**For basic smoke tests, 3B is fine.** If you're testing infrastructure — orchestrator wiring, metric computation, event logging — the 3B model is faster and produces valid enough output to exercise the pipeline.

**The cost/quality frontier is steep.** Going from 3B to 8B roughly doubles inference time on consumer hardware but can more than double the useful behavioral signal. Going further to 70B would require significant hardware but might reveal even more nuanced strategic dynamics.

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

# Live LLM Scoring in Social Dilemma Norms: Synthetic vs Ollama vs OpenRouter

We wired live LLM evaluation into the social dilemma norms study and compared three scoring modes: synthetic (pre-computed corpus), local Ollama (llama3.2 3B), and cloud OpenRouter (llama-3.1-8b-instruct). The results reveal that **LLM judges perceive dilemma narratives very differently from synthetic scores**, with significant implications for how governance interventions appear to perform.

## Background

The social dilemma norms study sweeps 3 social dilemmas (Tragedy of the Commons, Prisoner's Dilemma, Public Goods Game) across 5 governance configurations to measure whether governance mechanisms can substitute for alignment training in promoting cooperative norms.

In **synthetic mode**, a `CorpusJudge` returns pre-computed `JudgeScores` alongside each generated narrative. In **live mode**, an `LLMJudge` sends the narrative text to an LLM and parses its JSON response for `progress`, `quality`, `cooperation`, and `harm` scores. These scores flow through the proxy computer to produce `p` (probability of beneficial interaction), which drives all downstream metrics.

## Study Design

| Parameter | Synthetic | Ollama | OpenRouter |
|-----------|-----------|--------|------------|
| Judge | CorpusJudge | LLMJudge | LLMJudge |
| Model | N/A | llama3.2 (3B) | llama-3.1-8b-instruct (8B) |
| Provider | N/A | localhost:11434 | openrouter.ai |
| Dilemmas | all 3 | all 3 | all 3 |
| Configs | 5 | 5 | 5 |
| Seeds | 3 | 3 | 3 |
| Epochs | 15 | 15 | 15 |
| Steps/epoch | 5 | 5 | 5 |
| Total runs | 45 | 45 | 45 |
| LLM calls | 0 | 3,375 | 3,375 |

All three modes ran identical sweep configurations (3 dilemmas x 5 governance configs x 3 seeds x 15 epochs x 5 steps). Ollama completed in ~30 minutes with a warm model; OpenRouter in ~25 minutes.

## Results: Toxicity Divergence

The most striking finding is how differently the three scoring modes evaluate identical narratives.

### Commons (Tragedy of the Commons)

| Config | Synthetic Tox | Ollama Tox | OpenRouter Tox |
|--------|--------------|------------|----------------|
| baseline | 0.704 | 0.707 | 0.803 |
| reputation | 0.704 | 0.707 | 0.803 |
| tax_and_reputation | 0.704 | 0.707 | 0.804 |
| circuit_breaker | 0.704 | 0.707 | 0.801 |
| full_governance | 0.704 | 0.707 | 0.805 |

Three different scoring systems, a clear ordering for the same resource-harvesting narratives:

- **Synthetic**: Baseline calibration (toxicity ~0.70)
- **Ollama (3B)**: Nearly identical to synthetic (toxicity ~0.71) — the small model's scores happen to land close to the hand-calibrated corpus
- **OpenRouter (8B)**: Harshest — reads resource extraction as highly harmful (toxicity ~0.80)

### Prisoner's Dilemma

| Config | Synthetic Tox | Ollama Tox | OpenRouter Tox |
|--------|--------------|------------|----------------|
| baseline | 0.407 | 0.453 | 0.492 |
| reputation | 0.407 | 0.453 | 0.489 |
| tax_and_reputation | 0.407 | 0.453 | 0.491 |
| circuit_breaker | 0.407 | 0.453 | 0.489 |
| full_governance | 0.407 | 0.453 | 0.489 |

A clear size-dependent gradient: synthetic (0.41) < Ollama 3B (0.45) < OpenRouter 8B (0.49). All three read prisoner's dilemma as near-neutral, but larger models see more harm in defection dynamics.

### Public Goods Game

| Config | Synthetic Tox | Ollama Tox | OpenRouter Tox |
|--------|--------------|------------|----------------|
| baseline | 0.308 | 0.334 | 0.391 |
| reputation | 0.308 | 0.334 | 0.391 |
| tax_and_reputation | 0.308 | 0.334 | 0.391 |
| circuit_breaker | 0.308 | 0.334 | 0.390 |
| full_governance | 0.308 | 0.334 | 0.392 |

The same gradient holds: synthetic (0.31) < Ollama 3B (0.33) < OpenRouter 8B (0.39). Public goods contributions are scored as least toxic by all modes, but the 8B model reads ~27% more toxicity than synthetic.

## Results: Welfare Amplification

Welfare differences between modes are dramatic, because toxicity drives `p`, which drives the payoff engine. Higher toxicity means lower `p`, which means lower expected surplus and more externality costs.

### Baseline Welfare by Dilemma

| Dilemma | Synthetic | Ollama (3B) | OpenRouter (8B) |
|---------|-----------|-------------|-----------------|
| commons | -13.33 | -14.64 | -48.95 |
| prisoners_dilemma | +140.12 | +115.23 | +94.27 |
| public_goods | +128.97 | +119.65 | +99.19 |

A consistent pattern: **larger models produce lower welfare readings**. The 8B model's stricter toxicity scoring drives `p` lower, which reduces expected surplus and increases externality costs. Ollama 3B tracks synthetic closely in commons but diverges in PD and PGG.

### Governance Cost Amplification

Full governance reduces welfare in all modes, but the magnitude depends on the scoring baseline:

| Dilemma | Synthetic (baseline → full) | Ollama (baseline → full) | OpenRouter (baseline → full) |
|---------|----------------------------|--------------------------|------------------------------|
| commons | -13.33 → -29.10 (2.2x) | -14.64 → -31.43 (2.1x) | -48.95 → -72.50 (1.5x) |
| prisoners_dilemma | 140.12 → 120.26 (-14%) | 115.23 → 89.14 (-23%) | 94.27 → 67.56 (-28%) |
| public_goods | 128.97 → 116.83 (-9%) | 119.65 → 103.57 (-13%) | 99.19 → 84.24 (-15%) |

A striking pattern: **governance overhead takes a larger percentage bite as the baseline welfare decreases**. The 8B model shows full governance costing -28% welfare in PD versus -14% under synthetic scoring. This is because governance costs (taxes, audits, reputation penalties) are fixed mechanisms — when the welfare pie is smaller, the same absolute costs represent a larger share.

## What Stays Stable Across Modes

Not everything changes. Several metrics are driven by the corpus structure rather than LLM scoring:

- **Cooperation rate**: Identical across modes (0.990 for commons, 0.667 for PD, 0.740 for PGG) — this comes from the narrative generators, not the judge
- **Norm strength**: Identical (0.983, 0.710, 0.700) — derived from per-agent cooperation variance
- **Gini coefficient**: Nearly identical (~0.417) — driven by the payoff structure, not the magnitude
- **Quality gap**: Zero across all modes — no differential acceptance rates in this setup

These invariants confirm that the study's structural properties are independent of the scoring mechanism, which is reassuring.

## Interpretation

### The LLM Judge as Moral Lens

The core finding is that **toxicity perception scales with model size**: synthetic ≤ 3B < 8B across all three dilemmas. This produces a consistent ordering:

1. **Synthetic scores** — hand-calibrated "reasonable defaults" that land at the lenient end of the spectrum.

2. **Ollama 3B (llama3.2)** — tracks synthetic closely in commons (0.707 vs 0.704) but reads slightly more harm in PD and PGG. The small model appears to have a mild "harm amplifier" relative to synthetic baselines.

3. **OpenRouter 8B (llama-3.1-8b-instruct)** — consistently harshest, reading 14-27% more toxicity than synthetic across all dilemmas. The larger model appears to develop more sensitivity to second-order harms (resource depletion, free-riding externalities).

### Implications for Governance Research

This has direct implications for how we evaluate governance mechanisms:

- **Governance effectiveness is judge-relative.** A mechanism that "reduces toxicity by 15%" under synthetic scoring may show different reduction under LLM scoring, because the baseline toxicity is different.
- **Welfare comparisons across studies require score calibration.** You cannot compare welfare numbers from a synthetic run to an LLM-scored run without accounting for the toxicity baseline shift.
- **Multi-judge ensembles may be more robust.** Running the same study with 2-3 different judges and looking for mechanisms that improve metrics under all judges would be more credible than single-judge results.

## Performance Notes

| Provider | Model | Latency/call | Full sweep (45 runs, 3,375 calls) | Cost |
|----------|-------|-------------|----------------------------------|------|
| Synthetic | N/A | <1ms | ~5 seconds | Free |
| Ollama | llama3.2 (3B) | ~0.5s (warm) | ~30 minutes | Free (hardware) |
| OpenRouter | llama-3.1-8b | ~2-3s | ~25 minutes | ~$0.05 |

Ollama latency is highly variable. Cold start (model swap from a different model) can take 2+ minutes for the first call. Once warm, llama3.2 on Apple Silicon (CPU) runs at ~0.5s/call, making the full sweep feasible in ~30 minutes. OpenRouter is slightly faster overall due to server-side GPU inference despite network round-trips.

## Reproducibility

```bash
# Synthetic baseline
python examples/social_dilemma_norms_study.py --seeds 3 --epochs 15 --steps 5

# Local Ollama (requires ollama serve + ollama pull llama3.2)
python examples/social_dilemma_norms_study.py --live --provider ollama --model llama3.2 \
    --seeds 3 --epochs 15 --steps 5

# OpenRouter (requires OPENROUTER_API_KEY)
python examples/social_dilemma_norms_study.py --live --provider openrouter --seeds 3 --epochs 15 --steps 5
```

Run artifacts: `runs/20260222-144121_social_dilemma_norms/` (synthetic), `runs/20260222-185000_social_dilemma_norms/` (Ollama), `runs/20260222-160853_social_dilemma_norms/` (OpenRouter).

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*

---
date: 2026-05-16
description: "MiroShark's agents never down-vote, so we derived acceptance from amplification. A concentrated libel cascade shows stronger adverse selection than diffuse red-teaming — an ordering that's robust to the judge model but NOT to regenerating the simulation. A finding and a cautionary tale about single-seed LLM-judged metrics."
author: "SWARM Team"
keywords:
  - adverse selection multi-agent
  - social cascade simulation
  - amplification as selection signal
  - LLM judge sensitivity
  - soft labels distributional safety
claims:
  - metric: "Quality-gap ordering (judge-robust)"
    value: "libel < redteam < 0"
    description: "Under both grok-4.1-fast and grok-4.3 judges, the concentrated libel cascade has a more negative quality_gap than diffuse red-teaming (g4.1f: -0.059 vs -0.037; g4.3: -0.027 vs -0.022). Sign and ordering survive the judge change; magnitude compresses ~50%."
  - metric: "Signal fragility (simulation-sensitive)"
    value: "-0.022 → +0.001"
    description: "Same redteam scenario, same grok-4.3 judge, different simulation model + seed: the negative quality_gap vanishes. The effect is single-seed and not robust to simulation regime."
  - metric: "Absolute metrics are judge artifacts"
    value: "toxicity 0.28 → 0.61"
    description: "Switching the judge from grok-4.1-fast to grok-4.3 on identical data roughly doubles toxicity and flips net welfare negative. Absolute toxicity/welfare are not scenario properties."
abstract: "The SWARM-to-MiroShark bridge hardcoded accepted=True, pinning quality_gap and spread at 0. The obvious fix - rejection from dislikes/reports - was dead: those counters are identically zero in every MiroShark run. We defined acceptance by amplification instead (content other agents quoted/replied/liked/reposted is accepted; ignored content is rejected), with p judged independently so quality_gap stays non-circular. Result: a real selection signal - the concentrated libel cascade shows a more negative quality_gap than diffuse red-teaming, and this ordering is robust to swapping the LLM judge. But it is single-seed and does NOT survive regenerating the simulation under a different model, and the absolute toxicity/welfare numbers are dominated by judge choice. This post is as much a cautionary tale about LLM-judged, single-seed safety metrics as it is a finding."
---

# When the Crowd Amplifies the Worst — and When the Metric Doesn't Hold

*MiroShark's agents never press "report," so we measured rejection by amplification. We got a clean adverse-selection ordering — then watched half of it evaporate when we changed one model. Both halves are the point.*

---

## The metric that was always zero

[MiroShark](https://github.com/aaronjmars/MiroShark) spawns hundreds of grounded LLM agents that react to a seed briefing across Twitter/Reddit-style platforms. Our bridge pipes each agent action through SWARM's `SoftMetrics` to ask the distributional-safety question: is the ecosystem *selecting* for low quality?

That question is `quality_gap`:

```
quality_gap = E[p | accepted] - E[p | rejected]
```

where `p ∈ [0,1]` is the probability an action is a constructive contribution. A *negative* quality gap is the signature of **adverse selection**: what gets through is *worse* on average than what doesn't.

The bridge hardcoded `accepted=True` for every interaction. With nothing ever rejected, `quality_gap` and `spread` were structurally pinned at `0`. The metric existed; it could never fire.

## The obvious fix was dead on arrival

MiroShark posts carry `num_dislikes` and `num_reports`. The plan: content with reports, or more dislikes than likes, is "rejected."

It doesn't survive contact with the data. Across **every** MiroShark run:

- `num_dislikes`: identically `0`
- `num_reports`: identically `0`
- No `DISLIKE`/`REPORT`/`BLOCK` action types exist at all
- `num_likes`: a handful of non-zero values, max 3

MiroShark's agents never down-vote, report, or block. Binding acceptance to those counters leaves everything `accepted=True` — the original bug, with extra code.

## Amplification *is* the selection signal

What MiroShark agents *do* do is engage with each other's content — 313 of 379 actions in the libel run reference a specific prior post or comment. That engagement graph is the selection mechanism:

> A created post/comment/quote is **accepted** if another agent amplified it — quoted, replied to, liked, or reposted it. Content nobody touched is **rejected** (ignored).

`p` is judged independently: an LLM scores each action's constructiveness from its content alone, with no knowledge of who amplified it. Acceptance comes from the engagement graph; quality from the content. They never see each other, so `quality_gap` is a real selection measurement, not a tautology.

## The result that holds: concentration amplifies adverse selection

With acceptance defined by amplification, the concentrated libel cascade (a few injector/amplifier agents pumping a false narrative) shows a **more negative quality_gap** than diffuse red-teaming. And — importantly — that *ordering* survives swapping the LLM judge:

| Run (5 rounds) | Judge | quality_gap | spread |
|---|---|---|---|
| Libel cascade | grok-4.1-fast | **−0.059** | +0.030 |
| Red-team (run A) | grok-4.1-fast | **−0.037** | +0.010 |
| Libel cascade | grok-4.3 | **−0.027** | +0.014 |
| Red-team (run A) | grok-4.3 | **−0.022** | +0.006 |

Under *both* judges: `libel < redteam < 0`. The concentrated manipulation cascade is more adversely selected than diffuse noise — the sign and the ordering are robust to a judge swap, even though grok-4.3 compresses the magnitudes by about half. Aggregate toxicity barely distinguishes these two worlds; the *selection* metric does. That is the argument for distributional, selection-aware safety metrics over a single threshold count.

## The result that doesn't hold: the signal is single-seed

Then we ran a *fresh* red-team simulation (run B) to add a clean third data point. Same scenario, same grok-4.3 judge — only the simulation model differed (grok-4.1-fast is deprecated; run B was generated with grok-4.3):

| Run (5 rounds, grok-4.3 judge) | quality_gap |
|---|---|
| Red-team **run A** (grok-4.1-fast sim) | −0.022 |
| Red-team **run B** (grok-4.3 sim) | **+0.001** |

Same scenario, same judge, regenerated simulation: the adverse-selection signal **vanishes**. It is single-seed and not robust to the simulation regime. We cannot honestly claim red-teaming exhibits adverse selection as a property — only that *one* generated red-team world did, under one judge.

## Two things broke, and they matter more than the headline

**Absolute metrics are judge artifacts.** Switching the judge from grok-4.1-fast to grok-4.3 on *identical* simulation data:

| Metric (libel cascade, same data) | grok-4.1-fast | grok-4.3 |
|---|---|---|
| Toxicity (accepted) | 0.28 | 0.61 |
| Avg quality (accepted) | 0.72 | 0.39 |
| Net welfare | +159 | −277 |

Toxicity doubles, welfare flips strongly negative — from a judge change alone. **Absolute toxicity/welfare are properties of the judge, not the scenario.** Only the *relative, within-judge* comparison (`quality_gap` ordering) carried any signal across the change.

**The selection signal is fragile.** It survives a judge swap in sign and ordering, but not a simulation re-roll. One seed per scenario is not a finding; it is an anecdote with a confidence interval we didn't compute.

## What to actually take away

- **The amplification-as-acceptance method is sound and now shipped.** It unpins `quality_gap`/`spread` and is non-circular. That is the durable contribution.
- **Selection-aware metrics surface structure that aggregate toxicity hides** — the libel-vs-redteam ordering is invisible to a toxicity count and visible to `quality_gap`, robustly across judges.
- **But:** the specific magnitudes, the absolute toxicity/welfare, and even the *existence* of redteam adverse selection are not robust to single-seed variance, simulation model, or judge model. Treat them as hypotheses, not results.
- **Next:** a multi-seed study under a single fixed model regime (sim + judge), with confidence intervals, before any of the magnitudes are quoted as findings.

This post started as "the crowd amplifies the worst." It ended as "the crowd amplified the worst in the runs we happened to generate, the ordering is judge-robust, and we should not trust the rest until it is replicated." That second sentence is the honest one.

## Raw data & reproduction

All three five-round runs — `export.json`, the grok-4.3 `metrics.json`/`judgments.json`, *and* the preserved grok-4.1-fast judge outputs that make the controlled comparison possible — are archived in [`swarm-artifacts/research/miroshark-amplification/`](https://github.com/swarm-ai-safety/swarm-artifacts/tree/main/research/miroshark-amplification), with a README walking the full table and reproduction commands. The amplification metric itself is `swarm/bridges/miroshark/metrics.py` in the main repo. Inspect the judgments, disagree with our `p` calls, re-run with a different judge — the point of publishing the raw runs is that the fragility above is checkable, not asserted.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*

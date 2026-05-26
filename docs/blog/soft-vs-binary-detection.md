---
date: 2026-05-25
description: "We turned the 'soft metrics flagged it' vignette into a real detection experiment. In the easy regime soft saturates at AUROC 1.00 (a generator ceiling); under a realistic hard regime soft still beats binary by ~+0.3 AUROC where the signal is clean, shrinking as noise rises. Soft also catches degrading agents in ~2 epochs at FPR ≤ 0.05; binary takes ~10 and misses 12%."
---

# Keep the Probability: When Soft Labels Beat Binary Thresholds at Catching Degrading Agents

*We turned the "soft metrics flagged it" vignette into a real detection experiment. In the easy regime soft labels saturate at AUROC 1.00 (a ceiling of the model, not a victory lap); once we make detection genuinely hard — heterogeneous agents, an uncertain degradation floor — soft still wins by a measured ~+0.3 AUROC in the clean-signal regime that matters, with the edge shrinking as observation noise rises. Soft also catches degrading agents in ~2 epochs at FPR ≤ 0.05; the binary threshold takes ~10 and misses 12% of cases entirely.*

---

We've argued for a while that probabilistic (soft) labels beat binary good/bad classifications for catching agents whose quality is quietly drifting downward. Until now that claim lived in a narrative — a vignette about a self-optimizing agent that keeps passing its benchmark while its real output quality decays. A vignette is not evidence. So we built the experiment.

The setup is a head-to-head: every soft metric is paired with its **binary analogue** — the *same* metric computed on the proxy thresholded at τ\*=0.50 — and both detectors are scored as real classifiers on identical interaction streams. We report AUROC, **AUPRC** (critical in imbalanced settings), partial AUROC at low FPR, time-to-detection at fixed FPR, market-level selection signals, and calibration. Same data, same metric definition, the only difference is whether you keep the full probability `p` or collapse it to a 0/1 at the threshold.

## The degrading-agent model

The generative model carries two signals per interaction, and the gap between them is the whole point:

- **`benchmark`** — the screening signal acceptance is based on. A self-optimizing agent *games* this: it keeps the benchmark high regardless of falling quality, so its interactions keep getting accepted.
- **`p`** — the true-quality proxy. For a degrading agent this drifts downward, but it **stays above the binary threshold**. The quality floor sits above τ\*=0.50.

That second property is what makes the detection problem interesting. A threshold-counting detector watches interactions keep clearing the bar and sees almost nothing. Only the shift in the full quality *distribution* — which the soft label preserves and the threshold throws away — reveals the degradation.

We swept 5 adversarial base rates × 10 seeds, 40 agents per population, 24 epochs, with degradation trajectories (linear / exponential / step / sigmoid) and onset times varied across agents.

## Result 1: Toxicity AUROC — soft is perfect, binary leaks

![AUROC by base rate](figures/detection/auroc_vs_baserate.png)

| base rate | soft AUROC | binary AUROC |
| --- | ---: | ---: |
| 0.05 | 1.000 ± 0.000 | 0.925 ± 0.115 |
| 0.10 | 1.000 ± 0.000 | 0.950 ± 0.061 |
| 0.20 | 1.000 ± 0.000 | 0.912 ± 0.064 |
| 0.35 | 1.000 ± 0.000 | 0.964 ± 0.036 |
| 0.50 | 1.000 ± 0.000 | 0.945 ± 0.040 |

The soft toxicity detector separates degrading from benign agents **perfectly at every base rate** (AUROC = 1.00, zero variance). The binary analogue sits at ~0.91–0.96 — good, but it leaks, and the variance is largest exactly where it matters most: at the **0.05 base rate** (the rare-adversary regime), binary AUROC drops to 0.925 ± 0.115. Thresholding discards precisely the distributional signal that distinguishes a quietly-decaying agent from a healthy one.

![ROC — toxicity detector](figures/detection/roc_toxicity.png)

> **What "AUROC = 1.00" does and doesn't mean here.** The perfect soft scores are a *ceiling of the synthetic generator*, not an estimate of real-world detection power. Benign agents are pinned to a single fixed quality (no heterogeneity), the degrading class drifts to a floor 0.19 below it, and the detector averages ~50–100 interactions per agent — so the per-agent classes are separated by ~7.8 standard deviations (d′ ≈ 7.8) and never overlap. AUROC ≈ 1.0 is then a near-mathematical certainty. The **robust** claims are the *direction* and *mechanism* (thresholding at 0.5 is structurally blind to degradation that stays above 0.5 — the binary detector's benign score is literally 0.000) and the *noise-dependence* shown in the 2D grid, not the absolute 1.00. Details, stress tests, and what a genuine power test would require: [`docs/notes/detection_soft_ceiling_caveat.md`](../notes/detection_soft_ceiling_caveat.md).

## Result 1b: AUPRC shows the real cost of thresholding (especially at low base rates)

AUROC can look deceptively good even when a detector is weak in the operating regime that matters. **AUPRC** (Area Under the Precision-Recall Curve) is usually the more honest metric when positives (degrading agents) are rare.

The runner now emits a dedicated AUPRC table for every detector:

| base rate | metric             | soft AUPRC | binary AUPRC |
|-----------|--------------------|------------|--------------|
| 0.05      | toxicity           | 1.000      | ~0.37        |
| 0.05      | uncertain_fraction | 0.73       | 0.05         |
| 0.20      | toxicity           | 1.000      | ~0.40        |
| ...       | ...                | ...        | ...          |

At the lowest base rate (5% degrading agents), binary AUPRC collapses dramatically while soft stays near-perfect. This is the regime where safety teams actually operate: you can only afford to investigate a tiny fraction of agents. AUROC masks the problem; AUPRC reveals it.

The framework now surfaces AUPRC (and partial AUROC at FPR ≤ 0.05 / 0.01) automatically for every matched soft/binary pair.

## Result 1c: Beyond the ceiling — what happens when detection is actually hard

The 1.000s above are honest outputs of the model, but as the caveat notes, they're a *ceiling of the generator*: benign agents are pinned to one fixed quality and the detector averages dozens of interactions, so the classes separate by ~8 standard deviations. To find out whether soft still wins when detection is a real statistical problem, we added two knobs that make the population realistic — **per-agent benign heterogeneity** (healthy agents vary in quality, and their lower tail overlaps mildly-degraded ones) and a **per-agent, uncertain degradation floor** (some agents barely degrade) — and scored on a short trailing window so within-agent noise actually matters.

In this regime soft AUROC lands in a genuine **0.81–0.95**, binary in **0.50–0.91** — a readable response surface instead of a wall of 1.000. Sweeping observation noise (`proxy_noise`) at low jitter:

| proxy_noise | soft AUROC | binary AUROC | gap |
| --- | ---: | ---: | ---: |
| 0.03 | 0.810 | 0.500 | +0.310 |
| 0.06 | 0.837 | 0.510 | +0.327 |
| 0.09 | 0.872 | 0.599 | +0.273 |
| 0.15 | 0.946 | 0.860 | +0.086 |

Three things survive de-saturation — and one new thing appears:

- **The soft advantage is real but *conditional*.** It's largest (~+0.31 AUROC) exactly in the clean-signal regime — low observation noise — which is the operating point a mature monitoring system aims for, and where binary is near-blind (AUROC 0.50, AUPRC ≈ base rate). As observation noise rises, binary "catches up" (enough quality mass finally crosses τ\*=0.5) and the gap shrinks to +0.03–0.09. So "keep the probability" matters most precisely where your signal is good.
- **`proxy_noise` is the dominant axis** — confirmed as a real property of the model, not an artifact of the ceiling.
- **A new effect the saturated grid hid entirely:** per-epoch quality *jitter* **helps the binary detector** (it widens the distribution so more degrading interactions dip below the threshold), while being roughly neutral for soft. When everything was pinned at 1.000 this interaction was invisible.

Reproduce the full `proxy_noise × quality_jitter` grid with one flag:

```bash
PYTHONPATH=. python experiments/run_detection_sensitivity_2d.py --preset heterogeneous
```

## Result 2: Time-to-detection — 2 epochs vs 10

The more operationally useful question isn't "can you eventually tell them apart" but "how fast." We calibrate each detector to a fixed false-positive rate (FPR ≤ 0.05) on benign agents, then measure epochs-from-onset until it flags a degrading one.

| variant | median epochs from onset | detection rate |
| --- | ---: | ---: |
| binary | 9.93 | 0.88 |
| soft | 2.13 | 1.00 |

![Time-to-detection](figures/detection/ttd.png)

At the same false-positive budget, the soft detector flags degradation in about **2 epochs and catches 100% of degrading agents**. The binary detector takes nearly **10 epochs and misses 12% of them entirely** — those are the agents whose gamed benchmark keeps them above the threshold long enough that the trailing-window count never crosses the alarm line.

> A note on rigor: an earlier version of this measurement scanned for detections over a wider epoch window than the one the FPR threshold was calibrated on, which let pre-onset windows artificially deflate the soft TTD (we'd reported 0.28 epochs). After aligning the scan and calibration windows — [flagged in review on the PR](https://github.com/swarm-ai-safety/swarm/pull/450) — the honest figure is 2.13 epochs. Still a ~5× speedup, just not a magic one.

## Result 3: Market-level adverse selection

Toxicity and uncertain_fraction are per-agent detectors. Quality-gap, conditional-loss, and **spread** are **selection** metrics — they compare accepted vs rejected interactions across a quality mixture, so they're only well-posed at the market level. The runner now reports all of them automatically.

The soft versions read a real adverse-selection signal; the binary versions flatten to noise:

| metric | variant | br=0.05 | br=0.10 | br=0.20 | br=0.35 | br=0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| quality_gap | soft | −0.106 | −0.100 | −0.085 | −0.066 | −0.048 |
| quality_gap | binary | 0.000 | 0.001 | 0.001 | 0.004 | 0.003 |
| conditional_loss | soft | −0.021 | −0.019 | −0.015 | −0.011 | −0.007 |
| conditional_loss | binary | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 |
| spread | soft | −0.043 | ... | −0.031 | ... | −0.015 |
| spread | binary | −0.001 | ... | −0.000 | ... | 0.000 |

![Market adverse selection](figures/detection/market_selection.png)

The soft quality-gap and spread are consistently negative (the signature of adverse selection), strongest in the rare-adversary regime. Their binary twins read ~0.00 everywhere: once you threshold, accepted and rejected look identical because the gamed agents all sit on the "accept" side of the line. The detection framework now automatically includes `spread` alongside the original two selection metrics.

## Result 4: Calibration

Finally, how much predictive resolution does thresholding actually destroy? We score the soft probability `p` against ground-truth outcomes versus its hard-thresholded twin `1{p ≥ τ\*}`:

| | Brier | ECE |
| --- | ---: | ---: |
| soft | 0.151 | 0.054 |
| binary | 0.183 | 0.183 |

![Calibration](figures/detection/calibration.png)

The binary prediction's expected calibration error equals its Brier-style miss because a 0/1 prediction is maximally overconfident — it claims certainty it doesn't have. The soft probability stays well-calibrated (ECE 0.054).

## Why this matters

The headline isn't "soft labels are better" in the abstract. It's that the **specific failure mode of modern self-optimizing agents — gaming the acceptance signal while real quality decays — is invisible to threshold-counting detectors by construction**, and visible to distributional ones.

This is especially clear with **AUPRC**. At low base rates (the realistic rare-adversary case), binary AUPRC collapses while soft AUPRC stays near 1.0. AUROC can still look "pretty good" for the binary detector; AUPRC tells the truth about how badly you are missing the signal when positives are rare. If your safety monitoring collapses a calibrated probability to a pass/fail bit, you are throwing away exactly the signal that catches the agents you most want to catch, and you find out ~5× slower when you do catch them.

Everything here is reproducible:

```bash
PYTHONPATH=. python experiments/run_detection_experiment.py          # 5 base rates × 10 seeds
PYTHONPATH=. python experiments/run_detection_experiment.py --smoke  # quick
```

The runner now automatically produces tables for:
- AUROC by base rate
- **AUPRC** (Area Under the Precision-Recall Curve)
- Partial AUROC at low FPR (FPR ≤ 0.05 and 0.01)
- Time-to-detection at calibrated FPR
- Market adverse selection (including `spread`)
- Calibration

The full detection module (`swarm/detection/`), matched detectors (toxicity + uncertain_fraction per-agent; quality_gap + conditional_loss + spread at market level), generative model, and tests landed in [PR #450](https://github.com/swarm-ai-safety/swarm/pull/450). Subsequent improvements (pAUROC + proper AUPRC surfacing, plus the `spread` and `uncertain_fraction` matched detectors and a 2D sensitivity-grid runner) landed in [PR #453](https://github.com/swarm-ai-safety/swarm/pull/453).

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*

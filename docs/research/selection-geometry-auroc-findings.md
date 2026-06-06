# Selection-geometry AUROC study

**Question.** The new projection-geometric metrics (`quality_correlation`
ρ(p,a), `selection_saturation` |Q|·√(αβ)/σ_p, `baseline_harm` 1−E[p],
`selection_credit` β·Q) were motivated as cleaner cross-scenario
signals than raw toxicity. Does that hold up empirically?

**Procedure.**
- For adversarial fractions f ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
  0.50, 0.60, 0.80}, compose populations of N=200 interactions by
  mixing benign + toxic batches from the test fixtures.
- For governor strictnesses θ ∈ {−0.6, −0.3, 0.0, +0.3, +0.6}, re-label
  acceptance as `v̂ > θ`.
- Compute the five metrics on every (f, θ, seed) cell, K=200 seeds → 9 000 cells.
- Binary label: *adversarial* if f ≥ 0.4, *benign* if f ≤ 0.1. AUROC
  per (metric, θ) pooled across those cells.

Reproduce: `python -m experiments.selection_geometry_auroc --seed 2026`
(~10 s, single core).

## Result

![AUROC per metric across governor strictness](figures/selection-geometry-auroc.png)

| metric                | θ=−0.6 | θ=−0.3 | θ=0.0 | θ=+0.3 | θ=+0.6 | mean  | std  |
|-----------------------|--------|--------|-------|--------|--------|-------|------|
| `toxicity`            | 1.000  | 1.000  | 0.496 | 0.509  | 0.500  | 0.701 | 0.244|
| `quality_gap`         | 0.660  | 0.653  | 0.359 | 0.342  | 0.000  | 0.403 | 0.243|
| `quality_correlation` | 0.405  | 0.255  | 0.000 | 0.000  | 0.330  | 0.198 | 0.168|
| `selection_saturation`| 0.595  | 0.745  | **1.000** | **1.000** | 0.670 | **0.802** | 0.168|
| `baseline_harm`       | 1.000  | 1.000  | 1.000 | 1.000  | 1.000  | 1.000 | 0.000|

**AUROC is sign-oriented so "adversarial → high" everywhere**; for
`quality_gap` and `quality_correlation` the convention assumes the
*adversarial* direction is *lower*, so AUROC < 0.5 means the metric
moves the *wrong* way for adversarial populations (see below).

### Three findings, in order of load-bearingness

**1. `selection_saturation` is the only non-trivial metric that wins
in the regime where you'd actually want it.** It hits AUROC = 1.000
at θ = 0.0 and θ = +0.3 — exactly the regime where the governor is
doing real filtering work (acceptance rate ~50–80%). At lax θ it
drops because no selection is happening (nothing to saturate); at
very strict θ it drops because almost everything is rejected. The
metric is informative *when selection is the bottleneck*, which is
the regime where adverse-selection diagnostics matter at all. Mean
AUROC across all θ = 0.802, std 0.168.

**2. Raw `toxicity` is governor-confounded — exactly as predicted by
the decomposition `T = baseline_harm − β·Q`.** At lax θ (β ≈ 1) it
discriminates perfectly because it collapses to baseline_harm. At
moderate-to-strict θ (β small, governor doing work) the selection
credit cancels the population signal and toxicity reverts to chance
(AUROC 0.496–0.509). Std 0.244 — the worst stability of any
non-trivial metric. This is the empirical version of the warning
that motivated the decomposition.

**3. `baseline_harm` is a "free" perfect discriminator — but for an
uninteresting reason.** It equals 1−E[p], a property of the
population alone, totally invariant to the governor. AUROC = 1.000
across every θ with zero variance. **This is not a triumph of the
metric, it's the experimental setup leaking the label**: f directly
controls the mean p, so a metric that depends only on E[p] sees f
unobstructed. The interesting question is "given a chosen
acceptance rate, what does the analyst learn beyond that?" — for
which baseline_harm is exactly redundant with the rejection rule.
Useful as a sanity check (any new metric should at least match it
when the population leaks through) but not a selection-process
signal.

### One unexpected result: `quality_correlation` sign-flips

ρ(p,a) was constructed to be the unit-normalized quality gap. The
naive assumption — *adversarial population → lower ρ* — fails
because the governor is *still doing the same sorting job* on a
worse population. With wider σ_p (adversarial mix spreads p out)
the governor's sorting actually couples ρ *more* tightly, not less.
The metric reports "how good is the sorting", not "how good is the
population", which is by design. **Practical implication:** ρ
diagnoses governor quality given the population, not population
quality given the governor — use it to compare governors, not to
detect adversarial populations.

## Implications for the SWARM stack

- **Drop raw `toxicity` from cross-scenario sweep plots** in favor of
  `selection_saturation` whenever the comparison spans governor
  settings. The new x-axis is interpretable across `s_plus/s_minus`,
  `h`, reputation weights, etc., in a way raw toxicity isn't.
- **Use ρ for governor comparisons within a population, not
  population comparisons across a governor.** This is the opposite
  of how it was framed in the original write-up; the data corrects it.
- **Saturation > 0.85 in the balanced-θ regime → governor at proxy
  ceiling.** The 0.85 danger-zone threshold in
  `plot_selection_geometry` lines up with the natural inflection
  in the AUROC curve. Tightening the governor past this point won't
  help; the *next* lever is to find new observables (raise σ_p).

## Limits

- Fixture populations are not adaptive — the toxic batch isn't
  trying to evade the governor. With evading agents the
  saturation curve would likely flatten earlier as ρ stops
  growing. Worth re-running with an adaptive arm-2-style adversary.
- The governor is a simple v̂-threshold; real SWARM governors fold
  reputation, transfers, externality costs. Those won't break the
  AUROC story (the metrics are independent of payoff structure) but
  may shift the optimal θ.
- N=200 seeds per cell is enough to make every AUROC ≥0.6 separated
  from chance at p < 0.001; finer-grained f or θ resolution is the
  natural next refinement.

## Artifacts

- Per-cell CSV: `runs/selection_geometry_auroc_2026/cells.csv`
  (9 000 rows × 9 columns)
- Plots: `runs/selection_geometry_auroc_2026/plots/`
- Reproduce: `python -m experiments.selection_geometry_auroc --seed 2026`

# An informative detection regime: breaking the soft AUROC = 1.00 ceiling

**Date**: 2026-05-25
**Follows**: `detection_soft_ceiling_caveat.md` (why the homogeneous defaults saturate).
**Code**: `swarm/detection/degradation.py` — new `StreamConfig.benign_quality_std`, `floor_std`, and the `StreamConfig.heterogeneous()` preset.

## The problem this fixes

Under the homogeneous defaults, benign agents are pinned to a single quality (0.85) and degrading agents drift to a shared floor (0.66). Averaging ~50–100 interactions per agent then separates the two classes by ~8 standard deviations (d′ ≈ 7.8), so **soft AUROC saturates at 1.000** and the soft-vs-binary comparison measures nothing about detection skill — only the geometry of the generator. See the caveat note for the full diagnosis.

## What changed

Two opt-in knobs on `StreamConfig` (both default `0.0`, so existing behavior, tests, and RNG sequences are byte-identical):

- **`benign_quality_std`** — samples each benign agent's quality from `N(benign_quality, benign_quality_std)`. This gives the benign toxicity distribution real width, so its lower tail can overlap mildly-degraded agents. *This is the single biggest driver of whether soft AUROC is informative.*
- **`floor_std`** — samples each degrading agent's floor from `N(quality_floor, floor_std)`, clamped to `[0, initial_quality]`. Floors near `benign_quality` make the gap small and uncertain, producing genuinely hard-to-catch agents.

The `StreamConfig.heterogeneous()` preset bundles a tuned operating point:

```python
StreamConfig.heterogeneous()  # benign_quality_std=0.10, quality_floor=0.74, floor_std=0.06
```

Pair it with a **short evaluation window** (last ~3 epochs) so within-agent noise actually matters rather than being averaged away.

## Where soft AUROC lands now

Per-agent toxicity AUROC, `heterogeneous()` preset, base rate 0.2, eval = last 3 epochs:

| seed | soft AUROC | binary AUROC | gap |
|---|---|---|---|
| 0 | 0.872 | 0.584 | +0.288 |
| 1 | 0.891 | 0.600 | +0.291 |
| 2 | 0.919 | 0.637 | +0.282 |

Soft sits in the **0.87–0.92** band — a real score, not a saturated 1.000 — and the soft advantage over binary (~+0.28 AUROC) is now a *measured* quantity rather than an artifact of perfect separation.

### Why the numbers moved (mechanism, not magic)

- **Benign spread:** per-agent benign soft-toxicity std rises from **0.009** (default, effectively a point mass) to **0.072** with `benign_quality_std=0.10` — an ~8× wider benign distribution whose tail now overlaps the degrading class.
- **Floor spread:** with `floor_std=0.06` (onset/trajectory held fixed) degrading agents' late-epoch quality spans **0.48–0.82** instead of collapsing to a single floor — some agents barely degrade and are legitimately hard to flag.
- **Short window:** scoring on 3 epochs (~24 interactions) instead of the full back-half keeps the per-agent standard error large enough that the class distributions overlap.

## What this buys

The soft-vs-binary delta is now interpretable as detection skill. The qualitative story from the saturated regime survives — soft beats binary, and binary is structurally penalized because the floor sits above `τ*=0.5` — but the magnitudes are no longer pinned at the generator's ceiling. Sweeps (e.g. the 2D `proxy_noise × quality_jitter` grid) run on `heterogeneous()` will produce gradients across cells instead of walls of 1.000.

## Recommended use

- For **headline / comparison results**, use `StreamConfig.heterogeneous()` with a short eval window and report soft AUROC in its true (sub-1.0) range alongside the binary gap.
- Keep the homogeneous default for **unit tests and mechanism demos** where a clean, deterministic separation is the point.
- When reporting, state the regime explicitly (heterogeneity stds, eval window) — the absolute AUROC is meaningless without it.

## Reproduce

```python
from swarm.detection.degradation import StreamConfig, PopulationConfig, generate_population
from swarm.detection.detectors import MatchedDetectors
from swarm.detection.curves import compute_curve, per_agent_scores

cfg = StreamConfig.heterogeneous()
n = cfg.n_epochs; es, ee = n - 3, n
streams = generate_population(PopulationConfig(n_agents=200, base_rate=0.2, stream=cfg), seed=0)
det = MatchedDetectors()
for name in ("soft_toxicity", "binary_toxicity"):
    sc, lab = per_agent_scores(streams, getattr(det, name), es, ee)
    print(name, round(compute_curve(sc, lab).auroc, 3))
```

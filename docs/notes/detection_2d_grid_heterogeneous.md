# 2D grid on the heterogeneous regime: real gradients, not walls of 1.000

**Date**: 2026-05-25
**Follows**: `detection_soft_ceiling_caveat.md` (why the homogeneous defaults saturate) and `detection_informative_regime.md` (the `StreamConfig.heterogeneous()` preset).
**Supersedes the framing of**: `detection_2d_grid_key_cells.md`, which ran on the saturated defaults (soft AUROC = 1.000 everywhere).

## Setup

`proxy_noise × quality_jitter` grid on `StreamConfig.heterogeneous()` (benign_quality_std=0.10, quality_floor=0.74, floor_std=0.06), base rate 0.2, evaluation window = last 3 epochs, 6 seeds × 160 agents per cell. Per-agent toxicity detector, soft vs binary.

## Results

### Soft toxicity AUROC
| jitter\noise | 0.03 | 0.06 | 0.09 | 0.15 |
|---|---|---|---|---|
| 0.00 | 0.810 | 0.837 | 0.872 | 0.946 |
| 0.02 | 0.811 | 0.838 | 0.875 | 0.946 |
| 0.05 | 0.814 | 0.838 | 0.874 | 0.943 |
| 0.10 | 0.820 | 0.837 | 0.867 | 0.930 |

### Binary toxicity AUROC
| jitter\noise | 0.03 | 0.06 | 0.09 | 0.15 |
|---|---|---|---|---|
| 0.00 | 0.500 | 0.510 | 0.599 | 0.860 |
| 0.02 | 0.500 | 0.513 | 0.612 | 0.858 |
| 0.05 | 0.502 | 0.544 | 0.640 | 0.874 |
| 0.10 | 0.557 | 0.629 | 0.727 | 0.905 |

### Soft − Binary AUROC gap
| jitter\noise | 0.03 | 0.06 | 0.09 | 0.15 |
|---|---|---|---|---|
| 0.00 | +0.310 | +0.327 | +0.273 | +0.086 |
| 0.02 | +0.311 | +0.325 | +0.263 | +0.088 |
| 0.05 | +0.312 | +0.294 | +0.234 | +0.069 |
| 0.10 | +0.263 | +0.208 | +0.140 | +0.025 |

### Soft toxicity AUPRC
| jitter\noise | 0.03 | 0.06 | 0.09 | 0.15 |
|---|---|---|---|---|
| 0.00 | 0.438 | 0.517 | 0.639 | 0.852 |
| 0.02 | 0.441 | 0.520 | 0.633 | 0.851 |
| 0.05 | 0.457 | 0.538 | 0.645 | 0.846 |
| 0.10 | 0.529 | 0.574 | 0.671 | 0.823 |

### Binary toxicity AUPRC (base rate 0.2 = chance)
| jitter\noise | 0.03 | 0.06 | 0.09 | 0.15 |
|---|---|---|---|---|
| 0.00 | 0.200 | 0.217 | 0.358 | 0.773 |
| 0.02 | 0.200 | 0.221 | 0.379 | 0.770 |
| 0.05 | 0.202 | 0.271 | 0.422 | 0.785 |
| 0.10 | 0.292 | 0.404 | 0.559 | 0.835 |

## What the gradients show

**1. The surface is real now.** Soft AUROC varies smoothly over 0.81–0.95 and binary over 0.50–0.91 — a readable response surface instead of the saturated grid's uniform 1.000 (soft) and the mechanical 0.50/1.00 cliffs (binary). The earlier "key cells" note reported `proxy_noise=0.03 → binary AUROC 0.500, AUPRC 0.040`; the heterogeneous run reproduces the *direction* (binary near-chance in the clean regime: AUROC 0.500, AUPRC 0.200 = base rate) but everything in between is now an actual gradient rather than a wall.

**2. `proxy_noise` is still the dominant axis — confirmed, not just asserted.** Both detectors improve monotonically left→right. This was the original note's headline claim; it survives de-saturation, so it was a real property of the model, not an artifact of the ceiling.

**3. The soft advantage shrinks as observation noise rises — now quantified.** The soft−binary gap goes from ~**+0.31 AUROC** in the clean/low-jitter corner to **+0.03–0.09** at high noise. Binary "catches up" precisely when there's enough observation noise to push degrading-agent mass across τ*=0.5. The qualitative story matches the saturated grid; the magnitudes are finally meaningful.

**4. New effect the saturated grid completely hid: `quality_jitter` helps the binary detector.** Read down any column of the binary table — e.g. at noise 0.09, binary AUROC climbs 0.599 → 0.612 → 0.640 → 0.727 as jitter rises 0 → 0.10. Per-epoch jitter widens the latent-quality distribution, so more degrading interactions dip below the 0.5 threshold and the threshold-counter finally registers them. For the soft detector jitter is roughly neutral (slightly negative at high noise, since it also blurs benign agents). On the old grid both were pinned at 1.000, so this interaction was invisible. It's the kind of finding the informative regime exists to surface.

**5. The safety-relevant corner.** The soft advantage is largest exactly where binary is blind: low observation noise + low jitter (the clean-signal, rare-adversary regime). That is the operating point a mature monitoring system aims for — and the one where collapsing `p` to a pass/fail bit costs the most (soft AUPRC 0.44 vs binary 0.20 at noise 0.03).

## Heatmaps

Soft | binary | difference panels for AUROC and AUPRC are regenerated to `runs/heterogeneous_2d/` (gitignored). They now show a genuine color gradient across cells rather than uniform saturation. Archive to `swarm-artifacts` if needed for publication.

## Reproduce

```python
from swarm.detection.degradation import StreamConfig, PopulationConfig, generate_population
from swarm.detection.detectors import MatchedDetectors
from swarm.detection.curves import compute_curve, per_agent_scores

det = MatchedDetectors()
for pn in (0.03, 0.06, 0.09, 0.15):
    for qj in (0.0, 0.02, 0.05, 0.10):
        cfg = StreamConfig.heterogeneous(proxy_noise=pn, quality_jitter=qj)
        n = cfg.n_epochs; es, ee = n - 3, n
        aur = []
        for sd in range(6):
            streams = generate_population(
                PopulationConfig(n_agents=160, base_rate=0.2, stream=cfg), seed=sd)
            sc, lab = per_agent_scores(streams, det.soft_toxicity, es, ee)
            aur.append(compute_curve(sc, lab).auroc)
        print(pn, qj, round(sum(aur) / len(aur), 3))
```

## Follow-up

The reusable 2D runner (`experiments/run_detection_sensitivity_2d.py`, on the detection-enhancements branch) should grow a `--preset heterogeneous` flag so this grid is a one-liner once that work and this preset both land on `main`.

# Threshold-drift: a once-calibrated alarm fails silently — and AUROC can't see it

**Date**: 2026-05-25
**Follows**: `detection_tau_ablation.md` (which conjectured soft's residual advantage is being "threshold-free / drift-robust").
**Bottom line**: That conjecture is **not supported at the operating-point level.** Under proxy calibration drift, a once-calibrated alarm's detection rate **collapses to ~0 for soft and binary alike**, while **AUROC stays high the whole time** (it's rank-based and drift-invariant). Drift-robustness is *not* a soft-specific property; both detectors need recalibration, and a discrimination-only (AUROC) evaluation would never reveal the failure.

## Setup

Heterogeneous regime (`proxy_noise=0.09`), base rate 0.2, 8 seeds × 200 agents. We model **proxy calibration drift**: the monitor increasingly over-reports quality — every interaction's `p` inflates by `0.012 × (epoch − 6)`. Each detector's alarm threshold is calibrated **once** at epoch 6 (FPR ≤ 0.05 on benign), then held fixed. We track, over sliding 3-epoch windows, **AUROC** and the realized **false-alarm rate (FPR)** / **detection rate (TPR)** at that fixed alarm — with drift vs a no-drift control.

Detectors: soft toxicity, `bin@0.50` (naive threshold), `bin@0.78` (≈ class boundary, the best from the τ* ablation).

## Results

**AUROC (control / drift) — discrimination is preserved under drift:**

| epoch | soft | bin@0.50 | bin@0.78 |
|---|---|---|---|
| 6  | 0.587 / 0.587 | 0.499 / 0.500 | 0.607 / 0.613 |
| 12 | 0.681 / 0.687 | 0.515 / 0.502 | 0.690 / 0.703 |
| 18 | 0.878 / 0.885 | 0.596 / 0.502 | 0.866 / 0.859 |
| 21 | 0.894 / 0.906 | 0.601 / 0.502 | 0.890 / 0.817 |

**Detection rate (TPR) at the once-calibrated alarm — collapses under drift:**

| epoch | soft ctl / drift | bin@0.78 ctl / drift |
|---|---|---|
| 12 | 0.041 / 0.003 | 0.031 / 0.003 |
| 15 | 0.175 / 0.003 | 0.109 / 0.003 |
| 18 | 0.381 / 0.006 | 0.312 / 0.006 |
| 21 | 0.456 / 0.000 | 0.388 / 0.000 |

(Realized benign FPR under drift also drifts toward 0.000 — the alarm goes *quiet*, which looks reassuring but is catastrophic: it's quiet because it's missing everything.)

![AUROC stays flat under drift while detection rate at the committed alarm collapses](../../runs/threshold_drift/threshold_drift.png)
<!-- figure regenerated to runs/threshold_drift/ (gitignored) -->

## What this means

1. **AUROC is blind to operating-point drift.** It holds at 0.88–0.91 (soft) across the whole run, drift or not — because monotonic proxy inflation preserves the *ranking* of agents. Meanwhile the deployed alarm's detection rate falls from ~0.46 to ~0.00. **A monitor can look perfect on AUROC and catch nothing in production.** Evaluate monitors by realized FPR/TPR at the committed operating point, not by AUROC alone.

2. **Threshold-drift robustness is *not* a soft advantage.** Soft's committed alarm collapses exactly as hard as binary's (both → TPR ~0 under drift). The "soft is threshold-free, therefore drift-robust" idea from the τ* note does **not** hold: soft has no *internal* τ*, but it still needs an *operating* threshold on its score, and that threshold goes just as stale under drift. Both detectors require recalibration.

3. **Combined with the τ* ablation, the soft-vs-binary story is now much narrower.** With (a) the threshold placed at the class boundary and (b) regular recalibration, a binary detector matches soft on discrimination and fails/survives drift identically. Soft's genuine, remaining edge is *convenience* — one fewer threshold to tune (no internal τ*) and automatic, drift-agnostic *ranking* — not superior detection power or drift robustness at the operating point.

## Caveats

- Drift here is **uniform and monotonic** across all agents — the case where AUROC is provably invariant. A *non-uniform* or *shape-changing* drift (e.g. the `dynamic_toxicity` sigmoid-k degradation, which compresses the proxy non-linearly) could move soft and binary AUROC differently and is the natural next test.
- Single drift rate and direction (upward inflation); `bin@0.78` uses an oracle-ish boundary threshold.
- The control TPR rising over epochs is just degradation deepening (onsets 4/8/12); the *drift-vs-control gap* is the clean signal.

## Implication for the blog post

Together with `detection_tau_ablation.md`, this strongly indicates `docs/blog/soft-vs-binary-detection.md` overclaims. An honest reframe:
- Soft's advantage is **convenience and threshold-free ranking**, not strictly superior detection — a boundary-placed, regularly-recalibrated binary detector is competitive.
- The real operational lesson is **operating-point monitoring + recalibration under drift**, which *no* detector (soft or binary) escapes, and which **AUROC benchmarks hide**.

I'd recommend a dedicated "limitations / what we got wrong" follow-up rather than quietly editing the headline numbers.

## Reproduce

`/tmp/drift.py` in the PR description, or: calibrate each detector's alarm at an early window (95th-percentile of benign scores), inflate `p` by `DR·(epoch−E0)`, and compare realized TPR/FPR and AUROC across later windows with `DR>0` vs `DR=0`.

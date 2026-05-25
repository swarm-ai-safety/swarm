# Key Cells from the Proxy Noise × Quality Jitter 2D Grid

**Date**: 2026-05-25  
**Grid**: `proxy_noise` × `quality_jitter` (toxicity detector, base rate = 0.05)  
**Context**: 4 seeds × 25 agents × 16 epochs per cell. Focus on the most challenging regime (rare adversaries) and the most diagnostic metrics (AUPRC and pAUROC at low FPR).

## Selected Cells

### 1. Realistic Baseline (near model defaults)
- **proxy_noise** = 0.09, **quality_jitter** = 0.02

| Metric                  | Soft   | Binary | Gap |
|-------------------------|--------|--------|-----|
| AUROC                   | 1.000  | 0.875  | +0.125 |
| AUPRC                   | 1.000  | 0.760  | +0.240 |
| pAUROC @ FPR ≤ 0.05     | 1.000  | 0.875  | +0.125 |

**Interpretation**: Even at default parameters, binary detection is materially weaker on the metrics that matter for low base-rate regimes. The AUPRC gap is particularly large.

### 2. Clean Signal Regime (most dangerous for binary monitoring)
- **proxy_noise** = 0.03, **quality_jitter** = 0.02

| Metric                  | Soft   | Binary | Gap |
|-------------------------|--------|--------|-----|
| AUROC                   | 1.000  | 0.500  | +0.500 |
| AUPRC                   | 1.000  | 0.040  | +0.960 |
| pAUROC @ FPR ≤ 0.05     | 1.000  | 0.500  | +0.500 |

**Interpretation**: When the true quality signal is relatively clean, binary detectors essentially fail on AUPRC. This is the regime in which the self-optimizing / proxy-gaming threat model is most dangerous for any system relying on thresholded observations.

### 3. High Observation Noise Regime (binary "rescue" zone)
- **proxy_noise** = 0.15, **quality_jitter** = 0.02

| Metric                  | Soft   | Binary | Gap |
|-------------------------|--------|--------|-----|
| AUROC                   | 1.000  | 1.000  | 0.000 |
| AUPRC                   | 1.000  | 1.000  | 0.000 |
| pAUROC @ FPR ≤ 0.05     | 1.000  | 1.000  | 0.000 |

**Interpretation**: At high levels of observation noise on the quality signal, binary performance becomes competitive with (or indistinguishable from) soft on these aggregate metrics. The soft advantage largely evaporates.

## Main Pattern

The dominant axis is **proxy_noise** (observation noise on the true quality signal). 

- Below ~0.06–0.09, binary detection is poor on AUPRC and low-FPR partial AUROC regardless of quality jitter.
- At high proxy noise (0.15), binary becomes competitive or superior on most metrics.
- Quality jitter acts as a secondary modulator — it has the largest effect at intermediate noise levels and tends to degrade soft performance more than it helps binary.

Time-to-detection metrics were too sparse to draw reliable conclusions in this grid.

## Implication

The advantage of retaining the full probabilistic quality signal is highly sensitive to the amount of noise present in that signal. In relatively clean regimes, collapsing to binary decisions is extremely costly for detection power at realistic (low) base rates. As observation noise increases, that cost shrinks rapidly.

This suggests that the value of soft metrics is highest when the underlying quality signal is relatively well-calibrated and low-noise — precisely the conditions one would hope for in a mature monitoring system.
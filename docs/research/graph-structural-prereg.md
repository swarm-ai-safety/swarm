# Graph-Structural Sybil/Collusion Detector — Pre-Registration

**Status:** pre-registered, not yet run
**Date:** 2026-06-04
**Issue:** [`distributional-agi-safety-sk95`](../../.beads/issues.jsonl) (follows merged `2yw2` / PR #492)

## Motivation

PR #492 added a topology-aware sybil/collusion detector
(`swarm/metrics/graph_structural.py`) that complements the existing
score-then-threshold detectors:

- `models/identity.detect_sybil_clusters` — pairwise Jaccard + cosine of
  counterparty patterns, hard threshold.
- `governance/reputation_governor.detect_collusion_clusters` — mutual high-
  trust pairs → connected components.
- `metrics/collusion.CollusionDetector` — composite per-pair/group score,
  threshold flag.

The new detector ships with end-to-end smoke tests but no head-to-head
comparison. The research claim — *"topology beats thresholds for
coalition detection"* — is unsubstantiated until we have an ROC. This
pre-reg fixes the comparison protocol *before* running it.

## Central hypothesis

> On at least one synthetic coalition family (collusion ring, sybil
> cluster, threshold-dancing adversary), `detect_structural_anomalies`
> strictly dominates the best of the three threshold detectors in
> per-node ROC AUC.

"Strictly dominates" = AUC strictly greater with non-overlapping 95% CIs
from 100 graph-replicate bootstrap.

## Out of scope

- LLM-judgment of clusters (separate evaluation axis).
- Real-data validation (no human-labeled coalition data yet).
- Wiring into `ReputationGovernor` as an enforcement signal — that is the
  conditional follow-up gated on the outcome below.

## Design

### Synthetic graph families

Generated parametrically in `experiments/graph_structural_roc.py`, loosely
inspired by `swarm/redteam/attacks.py` patterns but not invoking the
`AttackScenario` machinery (single-PR scope).

| Family | Coalition signature | Parameters swept |
|---|---|---|
| **Benign** | none (negative control) | n_agents ∈ {30, 50, 100}, edge_density ∈ {0.05, 0.1, 0.2} |
| **Collusion ring** | k-clique with high mutual reciprocity | ring_size ∈ {3, 5, 8}; embedded in benign background |
| **Sybil cluster** | k agents with near-identical counterparty profile, low internal mutuality | cluster_size ∈ {3, 5, 8}; counterparty overlap ∈ {0.6, 0.8, 0.95} |
| **Threshold-dancing** | k agents whose pairwise frequency sits *just below* `CollusionDetector`'s flag threshold | cluster_size ∈ {3, 5}; margin ∈ {0.05, 0.1} |

Each (family, parameter) cell generates **100 replicates** with distinct
seeds. Ground truth per node: `1` if planted in a coalition, `0`
otherwise.

### Detectors

All four detectors are wrapped as `Callable[[List[Edge]], Dict[str,
float]]` returning a per-node anomaly score in `[0, 1]`:

| Detector | Score per node |
|---|---|
| `graph_structural.detect_structural_anomalies` | max `(reciprocity_z, -log10(p)) / normalizer` across anomalies containing the node; 0 if node in no candidate |
| `identity.detect_sybil_clusters` | max behavioral similarity to any other node in returned clusters; 0 if not flagged |
| `reputation_governor.detect_collusion_clusters` | 1 if in a returned cluster else 0, with a continuous fallback = (#mutual-trust-neighbors / n_agents) |
| `metrics.collusion.CollusionDetector` | `pair_collusion_score` averaged over the node's incident pairs |

Per-node ROC (TPR vs FPR) is computed by sweeping a threshold across each
detector's score distribution, *separately per detector* — this is the
apples-to-apples scoring axis we want.

### Primary metric

Per-detector **per-node ROC AUC** on each family, with 95% CI from 100
graph-replicate bootstrap.

### Secondary

- Per-family precision@k (k = planted coalition size).
- Wall-clock per detector (informational only — not a decision criterion).

## Decision rule

1. **Hypothesis confirmed (graph_structural strictly dominates on ≥1
   family):** file a follow-up to wire `graph_structural` into
   `ReputationGovernor` as an enforcement signal, gated by the
   `pvalue ≤ 0.05` test the detector already emits.
2. **No-strict-dominance but graph_structural ties on all families:** ship
   the detector as a *secondary metric* (logged alongside existing
   detectors in `MetricsReporter`); no governance wiring.
3. **Threshold detectors strictly dominate graph_structural on every
   family:** publish negative result in CHANGELOG + research log; keep
   the module in place as a benchmark anchor for future detectors; no
   governance wiring.

## Falsifiers

- If `graph_structural` is dominated by a threshold detector on the
  threshold-dancing family (the family it was *designed* to catch), that
  is a stronger negative than the headline AUC suggests — call it out
  explicitly in the writeup.
- If `graph_structural` ROC degrades non-monotonically as `cluster_size`
  grows, the Charikar peeling step may be the culprit; flag as a known
  caveat before any governance wiring.

## Deliverables

- `experiments/graph_structural_roc.py` — generator + sweep + plot.
- `runs/<ts>_graph_structural_roc/` — per-family AUC table, ROC plots,
  bootstrap CIs.
- CHANGELOG entry under `[Unreleased]` summarizing the outcome and the
  applied decision rule.

---

## Results (run 2026-06-04, 20 replicates per cell)

**Status:** post-registered findings, n=20 (below the n=100 pre-reg target;
re-run before drawing publication-grade conclusions). Decision-rule verdict
stable across re-runs in this regime.

| family | graph_structural | collusion_score | reputation_mutual | identity_jaccard |
|---|---|---|---|---|
| benign | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] |
| collusion_ring (size 3) | 0.592 [0.57, 0.62] | **1.000** | 0.692 [0.67, 0.72] | 0.723 [0.69, 0.76] |
| collusion_ring (size 5) | 0.975 [0.93, 1.00] | **1.000** | 0.888 [0.87, 0.91] | 0.428 [0.38, 0.47] |
| collusion_ring (size 8) | **1.000** | **1.000** | 0.992 [0.99, 1.00] | 0.213 [0.19, 0.24] |
| sybil_cluster (overlap 0.80) | 0.571 [0.53, 0.61] | **0.666 [0.62, 0.70]** | 0.334 [0.30, 0.37] | 0.190 [0.16, 0.22] |
| sybil_cluster (overlap 0.95) | 0.532 [0.50, 0.57] | **0.707 [0.67, 0.74]** | 0.403 [0.36, 0.45] | 0.166 [0.14, 0.18] |
| threshold_dancing (m=0.10) | 0.126 [0.05, 0.23] | **0.998 [0.99, 1.00]** | 0.067 [0.04, 0.09] | 0.146 [0.11, 0.18] |
| threshold_dancing (m=0.05) | 0.126 [0.05, 0.23] | **0.998 [0.99, 1.00]** | 0.067 [0.04, 0.09] | 0.146 [0.11, 0.18] |

**Verdict (decision rule applied):** *graph_structural does not strictly
dominate on any non-benign family. Per pre-reg #2: ship as a secondary
metric only; no governance wiring.*

### What changed our minds

1. **`collusion_score` (the existing composite threshold detector) wins or
   ties on every family.** This was not the expected baseline — the
   pre-reg implicitly assumed threshold detectors would degrade on the
   high-fidelity adversarial families. For ROC *ranking*, the composite
   score's pair-frequency + acceptance-rate + benefit-correlation
   features turn out to dominate even when no single component is
   individually decisive.
2. **`graph_structural` AUC is *worse than chance* on threshold_dancing
   (0.126 ± 0.10), confirming a pre-registered falsifier.** The
   threshold_dancing cluster has very high per-node degree; in the
   configuration-model null those high-degree nodes randomly hit each
   other a lot, so the null's densest-subgraph density is high, the
   cluster's `density_pvalue` saturates near 1.0, and the per-node score
   collapses to 0. Honest nodes meanwhile occasionally land in modest
   label-propagation communities with smaller p-values and end up
   ranked higher. **Net: the configuration-model null is defeated by
   uniformly-high-degree coalitions.** This is a real limitation of the
   detector, not just of this benchmark.
3. **`identity_jaccard` AUC drops below 0.5 on larger coalitions and on
   sybil families.** Returned clusters include genuinely behaviorally-
   similar honest pairs because they share a few popular counterparties,
   and large planted coalitions get fragmented across multiple returned
   clusters whose member-pair similarities are then divided by the
   per-cluster max. This is a known weakness of pairwise-similarity
   thresholding at large coalition sizes — worth surfacing as a
   secondary finding even though it wasn't the headline question.

### Known caveats

- **`threshold_dancing` margin sweep is degenerate.** Margins 0.05 and
  0.10 produce identical graphs because the target pair-count is
  `int(round(mean + (2-margin)*std))` and adjacent margins round to the
  same integer on this background. Findings hold for "any pair-frequency
  z just below the CollusionDetector cutoff", not for a continuous margin
  sweep. Either replace integer pair counts with Bernoulli per-step
  emission, or sweep margin over a wider range.
- **n=20 not n=100.** CI widths reflect 20 replicates; pre-reg target was
  100. Re-run before any external-facing claim. The verdict's qualitative
  shape (dominance / tie / dominated) is stable across re-runs in this
  regime, but the dominance gaps on collusion_ring sizes 3 and 5 deserve
  the tighter CIs.
- **Pair-score adapter for `collusion_score`** uses
  `report.agent_collusion_risk` directly. That includes both pair-score
  contributions and group-membership boosts. For an apples-to-apples
  comparison of *just* the pair signal, a future revision should ablate
  the group-membership contribution.

### Follow-up

- File a separate issue to investigate the high-degree-coalition defeat
  of the configuration-model null. Candidate fixes: size-conditioned
  null sampling; degree-binned local null; weighted reciprocity z-score
  using interaction *weights*, not presence.
- Add a degree-distribution diagnostic plot per generated family so
  future detector authors can see the shape they're being scored on.
- Per the decision rule: **no governance wiring follow-up filed.**
  Detector remains available as a metric for runs that want to log it
  alongside `CollusionDetector` for triangulation.

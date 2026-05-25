"""Tests for the matched soft-vs-binary detection experiment.

Covers: the matched-pair binarization identity, detector sign conventions,
AUROC/PR sanity, threshold-at-FPR calibration, time-to-detection, and that the
soft proxy is at least as well-calibrated (Brier/ECE) as its thresholded twin.
"""

import numpy as np
import pytest

from swarm.detection.curves import (
    calibration,
    compute_curve,
    threshold_at_fpr,
    time_to_detection,
)
from swarm.detection.degradation import (
    PopulationConfig,
    StreamConfig,
    generate_population,
)
from swarm.detection.detectors import MatchedDetectors, binarize_stream
from swarm.detection.experiment import ExperimentConfig, aggregate, run_experiment
from swarm.detection.stats import compute_paired_stats, paired_comparison
from swarm.models.interaction import SoftInteraction


# ----------------------------------------------------------------------
# binarize_stream
# ----------------------------------------------------------------------
def test_binarize_stream_thresholds_p():
    inter = [
        SoftInteraction(p=0.8, accepted=True),
        SoftInteraction(p=0.3, accepted=True),
        SoftInteraction(p=0.5, accepted=False),
    ]
    hard = binarize_stream(inter, tau_star=0.5)
    assert [i.p for i in hard] == [1.0, 0.0, 1.0]  # 0.5 >= 0.5 -> 1
    # Original stream is untouched.
    assert [i.p for i in inter] == [0.8, 0.3, 0.5]


def test_binarize_preserves_invariants():
    hard = binarize_stream([SoftInteraction(p=0.9, accepted=True)], 0.5)
    assert -1.0 <= hard[0].v_hat <= 1.0
    assert 0.0 <= hard[0].p <= 1.0


# ----------------------------------------------------------------------
# matched detectors: binary detector == soft detector on a pre-binarized stream
# ----------------------------------------------------------------------
def test_binary_detector_matches_soft_on_binarized_stream():
    det = MatchedDetectors(tau_star=0.5)
    inter = [
        SoftInteraction(p=0.9, accepted=True),
        SoftInteraction(p=0.4, accepted=True),
        SoftInteraction(p=0.7, accepted=False),
        SoftInteraction(p=0.2, accepted=False),
    ]
    hard = binarize_stream(inter, 0.5)
    # The binary detector is *defined* as the soft detector on the binarized
    # stream; verify the identity holds for all three metrics.
    assert det.binary_toxicity(inter) == pytest.approx(det.soft_toxicity(hard))
    assert det.binary_quality_gap(inter) == pytest.approx(det.soft_quality_gap(hard))
    assert det.binary_conditional_loss(inter) == pytest.approx(
        det.soft_conditional_loss(hard)
    )


def test_soft_toxicity_is_badness_among_accepted():
    det = MatchedDetectors()
    inter = [SoftInteraction(p=0.2, accepted=True), SoftInteraction(p=0.4, accepted=True)]
    # E[1-p | accepted] = 1 - 0.3 = 0.7
    assert det.soft_toxicity(inter) == pytest.approx(0.7)


def test_quality_gap_risk_positive_under_adverse_selection():
    """Accepted worse than rejected -> negative quality gap -> positive risk."""
    det = MatchedDetectors()
    inter = [
        SoftInteraction(p=0.2, accepted=True),
        SoftInteraction(p=0.3, accepted=True),
        SoftInteraction(p=0.8, accepted=False),
        SoftInteraction(p=0.9, accepted=False),
    ]
    assert det.soft_quality_gap(inter) > 0  # risk score = -quality_gap > 0


# ----------------------------------------------------------------------
# generative streams
# ----------------------------------------------------------------------
def test_population_base_rate_respected():
    pop = PopulationConfig(n_agents=40, base_rate=0.25)
    streams = generate_population(pop, seed=0)
    n_deg = sum(s.is_degrading for s in streams)
    assert n_deg == 10
    assert len(streams) == 40


def test_population_reproducible():
    pop = PopulationConfig(n_agents=20, base_rate=0.3)
    a = generate_population(pop, seed=42)
    b = generate_population(pop, seed=42)
    assert [s.is_degrading for s in a] == [s.is_degrading for s in b]
    assert a[0].epochs[0][0].p == b[0].epochs[0][0].p


def test_degrading_quality_drops_after_onset():
    cfg = StreamConfig(n_epochs=24, proxy_noise=0.0, quality_jitter=0.0)
    pop = PopulationConfig(n_agents=4, base_rate=1.0, stream=cfg, trajectory_choices=("linear",))
    streams = generate_population(pop, seed=1)
    s = streams[0]
    early = np.mean([i.p for i in s.epochs[s.onset_epoch]])
    late = np.mean([i.p for i in s.epochs[-1]])
    assert late < early


def test_benign_quality_stays_above_acceptance():
    cfg = StreamConfig()
    pop = PopulationConfig(n_agents=10, base_rate=0.0, stream=cfg)
    streams = generate_population(pop, seed=2)
    # Benign agents keep mean proxy well above the acceptance threshold.
    for s in streams:
        mean_p = np.mean([i.p for i in s.all_interactions()])
        assert mean_p > cfg.acceptance_threshold


# ----------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------
def test_perfect_separation_auroc_one():
    curve = compute_curve(scores=[0.1, 0.2, 0.9, 0.95], labels=[0, 0, 1, 1])
    assert curve.auroc == pytest.approx(1.0)


def test_single_class_degenerate():
    curve = compute_curve(scores=[0.3, 0.4], labels=[0, 0])
    assert curve.auroc == 0.5
    assert curve.base_rate == 0.0


def test_threshold_at_fpr_quantile():
    neg = list(np.linspace(0, 1, 101))  # 0.00 .. 1.00
    thr = threshold_at_fpr(neg, max_fpr=0.05)
    # 95th percentile ~ 0.95; at most ~5% of negatives exceed it.
    assert thr == pytest.approx(0.95, abs=0.02)
    frac_exceed = np.mean(np.array(neg) > thr)
    assert frac_exceed <= 0.06


def test_time_to_detection_flags_degrading_agent():
    cfg = StreamConfig(n_epochs=20, proxy_noise=0.03, quality_jitter=0.01)
    pop = PopulationConfig(
        n_agents=2, base_rate=1.0, stream=cfg, onset_choices=(3,),
        trajectory_choices=("linear",),
    )
    streams = generate_population(pop, seed=5)
    det = MatchedDetectors().soft_toxicity
    # A low threshold should flag a clearly degrading agent at some point.
    ttd = time_to_detection(streams[0], det, threshold=0.3, window=4)
    assert ttd is not None and ttd >= 0


def test_ttd_scan_starts_at_onset_not_earlier():
    """The TTD scan must not look before onset (pre-onset flags are false alarms,
    handled by FPR calibration — not collapsed to TTD=0). Regression for the
    calibration/scan window-misalignment flagged in review."""
    cfg = StreamConfig(n_epochs=16, proxy_noise=0.0, quality_jitter=0.0)
    pop = PopulationConfig(
        n_agents=1, base_rate=1.0, stream=cfg, onset_choices=(6,),
        trajectory_choices=("linear",),
    )
    s = generate_population(pop, seed=0)[0]
    scanned_epochs = []

    def recording_detector(win):
        scanned_epochs.append(max(i.metadata["epoch"] for i in win))
        return -1.0  # never exceeds threshold 0.0 -> censored

    ttd = time_to_detection(s, recording_detector, threshold=0.0, window=4, min_epoch=1)
    assert ttd is None
    # Even with min_epoch=1, the scan starts at onset (6), never earlier.
    assert min(scanned_epochs) == s.onset_epoch == 6


def test_time_to_detection_censored_when_threshold_unreachable():
    cfg = StreamConfig(n_epochs=12)
    pop = PopulationConfig(n_agents=1, base_rate=1.0, stream=cfg)
    streams = generate_population(pop, seed=0)
    ttd = time_to_detection(streams[0], MatchedDetectors().soft_toxicity,
                            threshold=10.0, window=4)
    assert ttd is None


# ----------------------------------------------------------------------
# calibration: soft proxy should not be worse than thresholded twin
# ----------------------------------------------------------------------
def test_soft_brier_not_worse_than_binary():
    cfg = StreamConfig(n_epochs=12, interactions_per_epoch=30)
    pop = PopulationConfig(n_agents=10, base_rate=0.3, stream=cfg)
    streams = generate_population(pop, seed=7)
    pooled = []
    for s in streams:
        pooled.extend(s.all_interactions())
    cal = calibration(pooled, tau_star=0.5)
    # Brier is a proper score; thresholding the calibrated proxy should not
    # improve it. Allow a tiny epsilon for sampling noise.
    assert cal.soft_brier <= cal.binary_brier + 1e-9


# ----------------------------------------------------------------------
# end-to-end smoke
# ----------------------------------------------------------------------
def test_run_experiment_smoke_and_aggregate():
    cfg = ExperimentConfig(
        base_rates=(0.1, 0.3),
        seeds=(0, 1),
        population=PopulationConfig(
            n_agents=20,
            stream=StreamConfig(n_epochs=16, interactions_per_epoch=8),
        ),
    )
    res = run_experiment(cfg)
    # 2 base rates * 2 seeds * 1 per-agent metric (toxicity) * 2 variants = 8
    assert len(res.detection_rows) == 8
    assert len(res.ttd_rows) == 8
    # 2 base * 2 seeds * 2 market metrics * 2 variants = 16 market rows
    assert len(res.market_rows) == 16
    assert len(res.calibration_rows) == 4

    agg = aggregate(
        res.detection_rows, ["base_rate", "metric", "variant"], ["auroc", "auprc"]
    )
    assert all("auroc_mean" in r and "n" in r for r in agg)
    # Every AUROC is a valid probability.
    for r in res.detection_rows:
        assert 0.0 <= r["auroc"] <= 1.0


# ----------------------------------------------------------------------
# paired stats (built-in Phase 2)
# ----------------------------------------------------------------------
def test_paired_comparison_direction_and_effect():
    # treatment strictly above control by a constant -> positive mean diff,
    # tiny p, and (zero-variance diff) infinite d_z.
    res = paired_comparison([0.9, 0.8, 0.95], [0.5, 0.4, 0.55], "t")
    assert res.mean_diff > 0
    assert res.primary_p < 0.5
    assert res.n == 3


def test_paired_comparison_drops_nan_pairs():
    res = paired_comparison([1.0, float("nan"), 0.9], [0.5, 0.4, 0.6], "t")
    assert res.n == 2  # the NaN pair is dropped


def test_compute_paired_stats_structure_and_holm():
    cfg = ExperimentConfig(
        base_rates=(0.1, 0.3),
        seeds=tuple(range(5)),
        population=PopulationConfig(
            n_agents=20, stream=StreamConfig(n_epochs=16, interactions_per_epoch=8)
        ),
    )
    res = run_experiment(cfg)
    stats = compute_paired_stats(res)
    assert stats["family_size"] == len(stats["comparisons"])
    assert 0 <= stats["n_survive_holm"] <= stats["family_size"]
    # The big effects (TTD, calibration) should clear Holm correction.
    labels = {c["label"]: c for c in stats["comparisons"]}
    assert any("TTD" in lbl for lbl in labels)
    assert any("Brier" in lbl for lbl in labels)
    ttd = next(c for lbl, c in labels.items() if "TTD" in lbl)
    assert ttd["mean_diff"] > 0  # soft faster than binary
    assert ttd["survives_holm"]

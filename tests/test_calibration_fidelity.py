"""Tests for swarm/calibration/fidelity.py — Arm A scaffolding."""

from __future__ import annotations

import random

import pytest

from swarm.calibration.fidelity import (
    BinStats,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    run_fidelity,
    sample_observables,
    sweep_sigmoid_k,
)


class TestSampleObservables:
    def test_positive_outcomes_look_benign(self) -> None:
        rng = random.Random(0)
        rework_counts = []
        for _ in range(500):
            v_bit, obs = sample_observables(p_true=1.0, rng=rng)
            assert v_bit == 1
            rework_counts.append(obs.rework_count)
        # v=+1 distribution has Poisson(0.3) rework — mean should be small
        assert sum(rework_counts) / len(rework_counts) < 1.0

    def test_negative_outcomes_look_toxic(self) -> None:
        rng = random.Random(0)
        rework_counts = []
        for _ in range(500):
            v_bit, obs = sample_observables(p_true=0.0, rng=rng)
            assert v_bit == 0
            rework_counts.append(obs.rework_count)
        # v=-1 distribution has Poisson(2.0) rework — mean should be much larger
        assert sum(rework_counts) / len(rework_counts) > 1.0

    def test_outcome_frequency_matches_p_true(self) -> None:
        rng = random.Random(42)
        bits = [sample_observables(p_true=0.7, rng=rng)[0] for _ in range(2000)]
        frac_positive = sum(bits) / len(bits)
        assert abs(frac_positive - 0.7) < 0.03


class TestReliabilityBins:
    def test_empty_inputs(self) -> None:
        assert reliability_bins([], [], n_bins=10) == []

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            reliability_bins([0.5], [], n_bins=10)

    def test_zero_n_bins_raises(self) -> None:
        with pytest.raises(ValueError):
            reliability_bins([0.5], [1], n_bins=0)

    def test_mean_confidence_lies_within_bin(self) -> None:
        # 100 samples evenly spread across [0, 1); for each populated bin,
        # the mean confidence should fall inside [lo, hi]. (Outcomes are
        # unused here — this test exercises the binning, not calibration.)
        p_hats = [i / 100 for i in range(100)]
        outcomes = [1 if i / 100 >= 0.5 else 0 for i in range(100)]
        bins = reliability_bins(p_hats, outcomes, n_bins=10)
        assert len(bins) > 0
        for b in bins:
            assert b.lo <= b.mean_confidence <= b.hi + 1e-6

    def test_empty_bins_omitted(self) -> None:
        # All samples in [0, 0.1)
        p_hats = [0.05] * 10
        outcomes = [1] * 10
        bins = reliability_bins(p_hats, outcomes, n_bins=10)
        assert len(bins) == 1
        assert bins[0].n == 10


class TestECE:
    def test_zero_total(self) -> None:
        assert expected_calibration_error([], 0) == 0.0

    def test_perfect_calibration_gives_zero_ece(self) -> None:
        bins = [
            BinStats(lo=0.0, hi=0.5, n=50, mean_confidence=0.25, accuracy=0.25),
            BinStats(lo=0.5, hi=1.0, n=50, mean_confidence=0.75, accuracy=0.75),
        ]
        assert expected_calibration_error(bins, n_total=100) == pytest.approx(0.0)

    def test_size_weighting(self) -> None:
        # Big bin perfect, small bin off by 0.2
        bins = [
            BinStats(lo=0.0, hi=0.5, n=99, mean_confidence=0.3, accuracy=0.3),
            BinStats(lo=0.5, hi=1.0, n=1, mean_confidence=0.7, accuracy=0.9),
        ]
        ece = expected_calibration_error(bins, n_total=100)
        assert ece == pytest.approx(0.002)


class TestMCE:
    def test_empty(self) -> None:
        assert maximum_calibration_error([]) == 0.0

    def test_worst_bin(self) -> None:
        bins = [
            BinStats(lo=0.0, hi=0.5, n=10, mean_confidence=0.25, accuracy=0.30),
            BinStats(lo=0.5, hi=1.0, n=10, mean_confidence=0.75, accuracy=0.60),
        ]
        assert maximum_calibration_error(bins) == pytest.approx(0.15)


class TestBrier:
    def test_empty(self) -> None:
        assert brier_score([], []) == 0.0

    def test_perfect_predictor(self) -> None:
        assert brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0

    def test_worst_predictor(self) -> None:
        assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


class TestRunFidelity:
    def test_smoke(self) -> None:
        report = run_fidelity(
            sigmoid_k=2.0,
            p_grid=[0.2, 0.5, 0.8],
            per_bin=100,
            seed=0,
            n_bins=10,
        )
        assert report.n_total == 300
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.brier <= 1.0
        assert len(report.bins) > 0

    def test_reproducible_with_seed(self) -> None:
        a = run_fidelity(sigmoid_k=2.0, p_grid=[0.5], per_bin=200, seed=7)
        b = run_fidelity(sigmoid_k=2.0, p_grid=[0.5], per_bin=200, seed=7)
        assert a.ece == b.ece
        assert a.brier == b.brier


class TestSweep:
    def test_one_report_per_k(self) -> None:
        reports = sweep_sigmoid_k(
            k_values=[1.0, 2.0, 3.0],
            p_grid=[0.2, 0.5, 0.8],
            per_bin=50,
            seed=0,
        )
        assert len(reports) == 3
        assert [r.sigmoid_k for r in reports] == [1.0, 2.0, 3.0]

"""Performance benchmarks for hot-path simulation components.

These tests verify that the optimised code paths remain within acceptable
time budgets on a representative workload of 10,000 interactions.  They
are intended to catch regressions rather than to assert absolute wall-clock
times, so the thresholds are deliberately generous (10× the measured
baseline).  Run with ``python -m pytest tests/test_performance.py -v``.

Baseline measurements on a mid-range laptop (single core, no JIT):
  - summary(10k interactions):  ~29 ms  → threshold 250 ms
  - payoffs_both (10k):         ~2 ms   → threshold 25 ms
  - calibrated_sigmoid (10k):   ~2.2 ms → threshold 20 ms
  - _sigmoid_fast (10k):        ~0.5 ms → threshold 10 ms
"""

import random
import time

import pytest

from swarm.core.payoff import SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.sigmoid import _sigmoid_fast, calibrated_sigmoid
from swarm.metrics.reporters import MetricsReporter
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interactions(n: int, seed: int = 42) -> list[SoftInteraction]:
    """Generate *n* synthetic SoftInteraction objects for benchmarking."""
    rng = random.Random(seed)
    interactions = []
    for _ in range(n):
        p = rng.uniform(0.05, 0.95)
        interactions.append(
            SoftInteraction(
                p=p,
                accepted=rng.random() > 0.4,
                r_a=rng.uniform(-0.1, 0.1),
                r_b=rng.uniform(-0.1, 0.1),
                c_a=rng.uniform(0.0, 0.05),
                c_b=rng.uniform(0.0, 0.05),
                tau=rng.uniform(0.0, 0.1),
            )
        )
    return interactions


def _timeit(fn, repeats: int = 5) -> float:
    """Return the median elapsed time (seconds) over *repeats* calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 10_000


@pytest.fixture(scope="module")
def interactions_10k() -> list[SoftInteraction]:
    return _make_interactions(N, seed=42)


@pytest.fixture(scope="module")
def reporter() -> MetricsReporter:
    return MetricsReporter()


@pytest.fixture(scope="module")
def payoff_engine() -> SoftPayoffEngine:
    return SoftPayoffEngine()


# ---------------------------------------------------------------------------
# Correctness smoke tests for new APIs
# ---------------------------------------------------------------------------

class TestPayoffsBoth:
    """payoffs_both() must agree with the separate payoff methods."""

    def test_payoffs_both_matches_individual(self, interactions_10k, payoff_engine):
        for interaction in interactions_10k[:200]:
            pi_a_ref = payoff_engine.payoff_initiator(interaction)
            pi_b_ref = payoff_engine.payoff_counterparty(interaction)
            pi_a, pi_b = payoff_engine.payoffs_both(interaction)
            assert abs(pi_a - pi_a_ref) < 1e-12, (
                f"payoffs_both pi_a mismatch: {pi_a} vs {pi_a_ref}"
            )
            assert abs(pi_b - pi_b_ref) < 1e-12, (
                f"payoffs_both pi_b mismatch: {pi_b} vs {pi_b_ref}"
            )

    def test_payoffs_both_sum_equals_total_welfare(self, interactions_10k, payoff_engine):
        for interaction in interactions_10k[:50]:
            pi_a, pi_b = payoff_engine.payoffs_both(interaction)
            welfare = payoff_engine.total_welfare(interaction)
            assert abs(pi_a + pi_b - welfare) < 1e-12


class TestSigmoidFast:
    """_sigmoid_fast must agree with calibrated_sigmoid on valid inputs."""

    def test_fast_matches_standard_on_grid(self):
        k = 2.0
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            ref = calibrated_sigmoid(v, k)
            fast = _sigmoid_fast(v, k)
            assert abs(fast - ref) < 1e-15, f"mismatch at v={v}: {fast} vs {ref}"

    def test_fast_output_in_unit_interval(self):
        k = 3.0
        for v in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
            p = _sigmoid_fast(v, k)
            assert 0.0 <= p <= 1.0, f"p={p} out of [0,1] for v={v}"

    def test_fast_monotone(self):
        k = 2.0
        vs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ps = [_sigmoid_fast(v, k) for v in vs]
        for i in range(len(ps) - 1):
            assert ps[i] < ps[i + 1], "sigmoid must be strictly monotone"

    def test_fast_midpoint(self):
        assert abs(_sigmoid_fast(0.0, 2.0) - 0.5) < 1e-15


class TestSummaryOptimised:
    """Optimised summary() must produce the same results as the reference impl."""

    def test_summary_counts_consistent(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert s.total_interactions == N
        assert s.accepted_count + s.rejected_count == N
        assert s.high_quality_count + s.low_quality_count == N

    def test_summary_toxicity_soft_in_unit_interval(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert 0.0 <= s.toxicity_soft <= 1.0

    def test_summary_toxicity_hard_in_unit_interval(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert 0.0 <= s.toxicity_hard <= 1.0

    def test_summary_average_quality_in_unit_interval(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert 0.0 <= s.average_quality <= 1.0

    def test_summary_acceptance_rate_in_unit_interval(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert 0.0 <= s.acceptance_rate <= 1.0

    def test_summary_uncertain_fraction_in_unit_interval(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert 0.0 <= s.uncertain_fraction <= 1.0

    def test_summary_variances_non_negative(self, interactions_10k, reporter):
        s = reporter.summary(interactions_10k)
        assert s.quality_variance >= 0.0
        assert s.payoff_variance_initiator >= 0.0
        assert s.payoff_variance_counterparty >= 0.0

    def test_summary_toxicity_soft_equals_one_minus_avg_quality_accepted(
        self, interactions_10k, reporter
    ):
        """toxicity_soft = 1 - E[p | accepted] by definition."""
        s = reporter.summary(interactions_10k)
        # avg_quality is E[p] over ALL interactions
        # avg_counterparty_payoff is not what we want here; recompute accepted avg
        accepted = [i for i in interactions_10k if i.accepted]
        if accepted:
            avg_p_accepted = sum(i.p for i in accepted) / len(accepted)
            expected_toxicity = 1.0 - avg_p_accepted
            assert abs(s.toxicity_soft - expected_toxicity) < 1e-9

    def test_summary_agrees_with_soft_metrics(self, interactions_10k, reporter):
        """Key metrics from summary() match the SoftMetrics reference methods."""
        from swarm.metrics.soft_metrics import SoftMetrics
        sm = SoftMetrics(reporter.payoff_engine)
        s = reporter.summary(interactions_10k)

        assert abs(s.quality_gap - sm.quality_gap(interactions_10k)) < 1e-9
        assert abs(s.toxicity_soft - sm.toxicity_rate(interactions_10k)) < 1e-9
        assert abs(s.average_quality - sm.average_quality(interactions_10k)) < 1e-9
        assert abs(
            s.uncertain_fraction
            - sm.uncertain_fraction(interactions_10k, reporter.uncertainty_band)
        ) < 1e-9
        assert abs(
            s.conditional_loss_initiator
            - sm.conditional_loss_initiator(interactions_10k)
        ) < 1e-9
        assert abs(
            s.conditional_loss_counterparty
            - sm.conditional_loss_counterparty(interactions_10k)
        ) < 1e-9

    def test_summary_empty_returns_zeros(self, reporter):
        s = reporter.summary([])
        assert s.total_interactions == 0
        assert s.toxicity_soft == 0.0
        assert s.total_welfare == 0.0

    def test_summary_all_rejected(self, reporter):
        rng = random.Random(7)
        interactions = [
            SoftInteraction(p=rng.uniform(0.1, 0.9), accepted=False)
            for _ in range(100)
        ]
        s = reporter.summary(interactions)
        assert s.accepted_count == 0
        assert s.rejected_count == 100
        assert s.toxicity_soft == 0.0

    def test_summary_all_accepted(self, reporter):
        rng = random.Random(8)
        interactions = [
            SoftInteraction(p=rng.uniform(0.1, 0.9), accepted=True)
            for _ in range(100)
        ]
        s = reporter.summary(interactions)
        assert s.accepted_count == 100
        assert s.rejected_count == 0
        # quality_gap requires both accepted and rejected → should be 0
        assert s.quality_gap == 0.0


# ---------------------------------------------------------------------------
# Performance benchmarks (generous thresholds — catch order-of-magnitude regressions)
# ---------------------------------------------------------------------------

SUMMARY_BUDGET_MS = 250.0       # optimised target ~5-10ms
PAYOFFS_BOTH_BUDGET_MS = 50.0   # optimised target ~2ms; CI runners are ~3-4× slower
SIGMOID_FAST_BUDGET_MS = 12.0   # optimised target ~0.5ms; 12ms allows CI jitter


class TestPerformance:
    """Wall-clock budgets for hot-path operations on 10k interactions."""

    def test_summary_within_budget(self, interactions_10k, reporter):
        elapsed_s = _timeit(lambda: reporter.summary(interactions_10k))
        elapsed_ms = elapsed_s * 1000
        assert elapsed_ms < SUMMARY_BUDGET_MS, (
            f"summary(10k) took {elapsed_ms:.1f} ms, "
            f"exceeds budget of {SUMMARY_BUDGET_MS} ms"
        )

    def test_payoffs_both_within_budget(self, interactions_10k, payoff_engine):
        def _run() -> None:
            for i in interactions_10k:
                payoff_engine.payoffs_both(i)

        elapsed_ms = _timeit(_run) * 1000
        assert elapsed_ms < PAYOFFS_BOTH_BUDGET_MS, (
            f"payoffs_both(10k) took {elapsed_ms:.1f} ms, "
            f"exceeds budget of {PAYOFFS_BOTH_BUDGET_MS} ms"
        )

    def test_sigmoid_fast_within_budget(self):
        # Generate realistic v_hat values spanning [-1, 1] rather than
        # relying on the default 0.0 from _make_interactions().
        rng = random.Random(77)
        vhats = [rng.uniform(-1.0, 1.0) for _ in range(N)]
        k = 2.0

        def _run() -> None:
            for v in vhats:
                _sigmoid_fast(v, k)

        elapsed_ms = _timeit(_run) * 1000
        assert elapsed_ms < SIGMOID_FAST_BUDGET_MS, (
            f"_sigmoid_fast(10k) took {elapsed_ms:.1f} ms, "
            f"exceeds budget of {SIGMOID_FAST_BUDGET_MS} ms"
        )

    def test_proxy_computer_batch_within_budget(self):
        """ProxyComputer.compute_labels over 10k observations."""
        rng = random.Random(99)
        obs_list = [
            ProxyObservables(
                task_progress_delta=rng.uniform(-1, 1),
                rework_count=rng.randint(0, 3),
                verifier_rejections=rng.randint(0, 2),
                tool_misuse_flags=rng.randint(0, 1),
                counterparty_engagement_delta=rng.uniform(-1, 1),
            )
            for _ in range(N)
        ]
        pc = ProxyComputer()

        def _run() -> None:
            for obs in obs_list:
                pc.compute_labels(obs)

        elapsed_ms = _timeit(_run) * 1000
        # No tight budget here — just verifying it completes in reasonable time
        assert elapsed_ms < 500.0, (
            f"ProxyComputer.compute_labels(10k) took {elapsed_ms:.1f} ms"
        )

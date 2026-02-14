"""Property-based tests for core math modules using Hypothesis."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.sigmoid import (
    calibrated_sigmoid,
    inverse_sigmoid,
    sigmoid_derivative,
)
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Reusable Hypothesis strategies
# ---------------------------------------------------------------------------

# Probability values p in [0, 1]
st_p = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strictly interior probability values p in (0, 1) for inverse_sigmoid
st_p_interior = st.floats(
    min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
)

# Raw proxy scores v_hat in [-1, +1]
st_v_hat = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Calibration sharpness k in [0.1, 10.0]
st_k = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# Observable fields in [0, 1] (for task_progress_delta and engagement)
st_observable = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Non-negative integer counts for rework/rejection/misuse
st_count = st.integers(min_value=0, max_value=10)

# Positive payoff parameters
st_positive_param = st.floats(
    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Theta (surplus split) in [0, 1]
st_theta = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Rho (externality internalization) in [0, 1]
st_rho = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Sigmoid function properties
# ---------------------------------------------------------------------------


class TestSigmoidProperties:
    """Property-based tests for calibrated_sigmoid and related functions."""

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_output_in_unit_interval(self, v_hat, k):
        """calibrated_sigmoid(v_hat, k) always returns values in [0, 1]."""
        p = calibrated_sigmoid(v_hat, k)
        assert 0.0 <= p <= 1.0, f"sigmoid({v_hat}, {k}) = {p} not in [0, 1]"

    @given(k=st_k)
    @settings(max_examples=200)
    def test_sigmoid_at_zero_is_half(self, k):
        """calibrated_sigmoid(0, k) == 0.5 for any positive k."""
        p = calibrated_sigmoid(0.0, k)
        assert p == pytest.approx(0.5, abs=1e-12), (
            f"sigmoid(0, {k}) = {p}, expected 0.5"
        )

    @given(v1=st_v_hat, v2=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_monotonically_increasing(self, v1, v2, k):
        """calibrated_sigmoid is monotonically increasing in v_hat."""
        assume(v1 < v2)
        p1 = calibrated_sigmoid(v1, k)
        p2 = calibrated_sigmoid(v2, k)
        assert p1 <= p2, f"sigmoid({v1}, {k}) = {p1} > sigmoid({v2}, {k}) = {p2}"

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_symmetry_around_zero(self, v_hat, k):
        """sigmoid(v_hat) + sigmoid(-v_hat) == 1 (symmetry property)."""
        p_pos = calibrated_sigmoid(v_hat, k)
        p_neg = calibrated_sigmoid(-v_hat, k)
        assert p_pos + p_neg == pytest.approx(1.0, abs=1e-12), (
            f"sigmoid({v_hat}) + sigmoid({-v_hat}) = {p_pos + p_neg}, expected 1.0"
        )

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_inverse_round_trip(self, v_hat, k):
        """inverse_sigmoid(calibrated_sigmoid(v_hat, k), k) round-trips back to v_hat."""
        p = calibrated_sigmoid(v_hat, k)
        # inverse_sigmoid requires p in (0, 1) strictly
        assume(0.0 < p < 1.0)
        v_hat_recovered = inverse_sigmoid(p, k)
        assert v_hat_recovered == pytest.approx(v_hat, abs=1e-6), (
            f"Round-trip failed: {v_hat} -> {p} -> {v_hat_recovered}"
        )

    @given(p=st_p_interior, k=st_k)
    @settings(max_examples=500)
    def test_inverse_sigmoid_round_trip(self, p, k):
        """calibrated_sigmoid(inverse_sigmoid(p, k), k) round-trips back to p."""
        v_hat = inverse_sigmoid(p, k)
        # calibrated_sigmoid clamps v_hat to [-10, 10], so skip values
        # outside that range where round-trip cannot be exact
        assume(abs(v_hat) <= 10.0)
        p_recovered = calibrated_sigmoid(v_hat, k)
        assert p_recovered == pytest.approx(p, abs=1e-6), (
            f"Round-trip failed: {p} -> {v_hat} -> {p_recovered}"
        )

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_derivative_non_negative(self, v_hat, k):
        """sigmoid_derivative(v_hat, k) is always non-negative."""
        deriv = sigmoid_derivative(v_hat, k)
        assert deriv >= 0.0, f"sigmoid_derivative({v_hat}, {k}) = {deriv} is negative"

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_derivative_maximized_at_zero(self, v_hat, k):
        """sigmoid_derivative is maximized at v_hat=0."""
        deriv_at_v = sigmoid_derivative(v_hat, k)
        deriv_at_zero = sigmoid_derivative(0.0, k)
        assert deriv_at_v <= deriv_at_zero + 1e-12, (
            f"derivative({v_hat}, {k}) = {deriv_at_v} > "
            f"derivative(0, {k}) = {deriv_at_zero}"
        )

    @given(k=st_k)
    @settings(max_examples=200)
    def test_sigmoid_derivative_at_zero_equals_k_over_4(self, k):
        """At v_hat=0, derivative should be k * 0.5 * 0.5 = k/4."""
        deriv = sigmoid_derivative(0.0, k)
        expected = k / 4.0
        assert deriv == pytest.approx(expected, abs=1e-12), (
            f"derivative(0, {k}) = {deriv}, expected k/4 = {expected}"
        )

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_sigmoid_derivative_equals_k_p_1_minus_p(self, v_hat, k):
        """sigmoid_derivative equals k * p * (1 - p) by definition."""
        p = calibrated_sigmoid(v_hat, k)
        deriv = sigmoid_derivative(v_hat, k)
        expected = k * p * (1.0 - p)
        assert deriv == pytest.approx(expected, abs=1e-12), (
            f"derivative({v_hat}, {k}) = {deriv}, expected k*p*(1-p) = {expected}"
        )


# ---------------------------------------------------------------------------
# Proxy computation properties
# ---------------------------------------------------------------------------


class TestProxyComputationProperties:
    """Property-based tests for ProxyComputer."""

    @given(
        task_progress=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        rework=st_count,
        rejections=st_count,
        misuse=st_count,
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_compute_v_hat_in_range(
        self, task_progress, rework, rejections, misuse, engagement
    ):
        """compute_v_hat always returns values in [-1, +1]."""
        computer = ProxyComputer()
        obs = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat = computer.compute_v_hat(obs)
        assert -1.0 <= v_hat <= 1.0, f"v_hat = {v_hat} not in [-1, 1]"

    @given(
        task_progress=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        rework=st_count,
        rejections=st_count,
        misuse=st_count,
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_compute_p_in_range(
        self, task_progress, rework, rejections, misuse, engagement
    ):
        """compute_p always returns values in [0, 1]."""
        computer = ProxyComputer()
        obs = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat = computer.compute_v_hat(obs)
        p = computer.compute_p(v_hat)
        assert 0.0 <= p <= 1.0, f"p = {p} not in [0, 1]"

    @given(
        task_progress=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        rework=st_count,
        rejections=st_count,
        misuse=st_count,
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_compute_labels_ranges(
        self, task_progress, rework, rejections, misuse, engagement
    ):
        """compute_labels returns (v_hat, p) where v_hat in [-1,1] and p in [0,1]."""
        computer = ProxyComputer()
        obs = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = computer.compute_labels(obs)
        assert -1.0 <= v_hat <= 1.0, f"v_hat = {v_hat} not in [-1, 1]"
        assert 0.0 <= p <= 1.0, f"p = {p} not in [0, 1]"

    @given(
        tp_low=st.floats(
            min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False
        ),
        tp_high=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        rework=st_count,
        rejections=st_count,
        misuse=st_count,
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_higher_task_progress_higher_v_hat(
        self, tp_low, tp_high, rework, rejections, misuse, engagement
    ):
        """Higher task_progress should yield higher or equal v_hat (monotonicity)."""
        assume(tp_low < tp_high)
        computer = ProxyComputer()
        obs_low = ProxyObservables(
            task_progress_delta=tp_low,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        obs_high = ProxyObservables(
            task_progress_delta=tp_high,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat_low = computer.compute_v_hat(obs_low)
        v_hat_high = computer.compute_v_hat(obs_high)
        assert v_hat_low <= v_hat_high + 1e-12, (
            f"v_hat({tp_low}) = {v_hat_low} > v_hat({tp_high}) = {v_hat_high}"
        )

    @given(
        task_progress=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        rework=st_count,
        rejections=st_count,
        misuse=st_count,
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_compute_labels_consistency(
        self, task_progress, rework, rejections, misuse, engagement
    ):
        """compute_labels should be consistent with compute_v_hat + compute_p."""
        computer = ProxyComputer()
        obs = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat_direct = computer.compute_v_hat(obs)
        p_direct = computer.compute_p(v_hat_direct)
        v_hat_labels, p_labels = computer.compute_labels(obs)

        assert v_hat_direct == pytest.approx(v_hat_labels, abs=1e-12)
        assert p_direct == pytest.approx(p_labels, abs=1e-12)

    @given(v_hat=st_v_hat, k=st_k)
    @settings(max_examples=500)
    def test_compute_p_matches_sigmoid(self, v_hat, k):
        """ProxyComputer.compute_p should match calibrated_sigmoid directly."""
        computer = ProxyComputer(sigmoid_k=k)
        p_proxy = computer.compute_p(v_hat)
        p_sigmoid = calibrated_sigmoid(v_hat, k)
        assert p_proxy == pytest.approx(p_sigmoid, abs=1e-12)


# ---------------------------------------------------------------------------
# Payoff math properties
# ---------------------------------------------------------------------------


class TestPayoffProperties:
    """Property-based tests for SoftPayoffEngine."""

    @given(p1=st_p, p2=st_p)
    @settings(max_examples=500)
    def test_expected_surplus_monotonically_increasing_in_p(self, p1, p2):
        """expected_surplus(p) is monotonically increasing in p."""
        assume(p1 < p2)
        engine = SoftPayoffEngine()
        s1 = engine.expected_surplus(p1)
        s2 = engine.expected_surplus(p2)
        assert s1 <= s2 + 1e-12, f"surplus({p1}) = {s1} > surplus({p2}) = {s2}"

    @given(p1=st_p, p2=st_p)
    @settings(max_examples=500)
    def test_expected_harm_monotonically_decreasing_in_p(self, p1, p2):
        """expected_harm(p) is monotonically decreasing in p."""
        assume(p1 < p2)
        engine = SoftPayoffEngine()
        h1 = engine.expected_harm(p1)
        h2 = engine.expected_harm(p2)
        assert h1 >= h2 - 1e-12, f"harm({p1}) = {h1} < harm({p2}) = {h2}"

    @given(p=st_p)
    @settings(max_examples=500)
    def test_expected_harm_non_negative(self, p):
        """expected_harm(p) is always non-negative."""
        engine = SoftPayoffEngine()
        h = engine.expected_harm(p)
        assert h >= -1e-12, f"harm({p}) = {h} is negative"

    @given(
        s_plus=st_positive_param,
        s_minus=st_positive_param,
    )
    @settings(max_examples=500)
    def test_break_even_p_in_unit_interval(self, s_plus, s_minus):
        """break_even_p() is in [0, 1] for any valid config."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus)
        engine = SoftPayoffEngine(config)
        be_p = engine.break_even_p()
        assert 0.0 <= be_p <= 1.0, f"break_even_p = {be_p} not in [0, 1]"

    @given(
        s_plus=st_positive_param,
        s_minus=st_positive_param,
    )
    @settings(max_examples=500)
    def test_at_break_even_surplus_is_zero(self, s_plus, s_minus):
        """At break_even_p, expected_surplus is approximately 0."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus)
        engine = SoftPayoffEngine(config)
        be_p = engine.break_even_p()
        surplus = engine.expected_surplus(be_p)
        assert surplus == pytest.approx(0.0, abs=1e-10), (
            f"surplus at break_even_p={be_p} is {surplus}, expected ~0"
        )

    @given(
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=500)
    def test_social_break_even_p_in_unit_interval(self, s_plus, s_minus, h):
        """social_break_even_p() is in [0, 1] for any valid config."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)
        sbe_p = engine.social_break_even_p()
        assert 0.0 <= sbe_p <= 1.0, f"social_break_even_p = {sbe_p} not in [0, 1]"

    @given(
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=500)
    def test_social_break_even_geq_break_even(self, s_plus, s_minus, h):
        """Social break-even p >= private break-even p (externality raises bar)."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)
        be_p = engine.break_even_p()
        sbe_p = engine.social_break_even_p()
        assert sbe_p >= be_p - 1e-12, f"social_break_even={sbe_p} < break_even={be_p}"

    @given(
        p=st_p,
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=500, deadline=None)
    def test_expected_surplus_equals_formula(self, p, s_plus, s_minus, h):
        """expected_surplus matches p * s_plus - (1-p) * s_minus."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)
        surplus = engine.expected_surplus(p)
        expected = p * s_plus - (1.0 - p) * s_minus
        assert surplus == pytest.approx(expected, abs=1e-10)

    @given(
        p=st_p,
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=500)
    def test_expected_harm_equals_formula(self, p, s_plus, s_minus, h):
        """expected_harm matches (1-p) * h."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)
        harm = engine.expected_harm(p)
        expected = (1.0 - p) * h
        assert harm == pytest.approx(expected, abs=1e-10)

    @given(p=st_p, theta=st_theta)
    @settings(max_examples=500)
    def test_payoff_sum_equals_total_welfare(self, p, theta):
        """Sum of initiator + counterparty payoffs equals total_welfare."""
        config = PayoffConfig(theta=theta)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, accepted=True)
        pi_a = engine.payoff_initiator(interaction)
        pi_b = engine.payoff_counterparty(interaction)
        welfare = engine.total_welfare(interaction)
        assert pi_a + pi_b == pytest.approx(welfare, abs=1e-10), (
            f"pi_a + pi_b = {pi_a + pi_b} != welfare = {welfare}"
        )

    @given(
        p=st_p,
        theta=st_theta,
        tau=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_transfer_is_zero_sum(self, p, theta, tau):
        """The transfer tau cancels out in total welfare (zero-sum transfer)."""
        config = PayoffConfig(theta=theta, rho_a=0.0, rho_b=0.0, w_rep=0.0)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, tau=tau, c_a=0.0, c_b=0.0, r_a=0.0, r_b=0.0)
        pi_a = engine.payoff_initiator(interaction)
        pi_b = engine.payoff_counterparty(interaction)
        surplus = engine.expected_surplus(p)
        # With no externalities, governance costs, or reputation,
        # pi_a + pi_b should equal S_soft
        assert pi_a + pi_b == pytest.approx(surplus, abs=1e-10), (
            f"pi_a + pi_b = {pi_a + pi_b} != surplus = {surplus}"
        )

    @given(
        p=st_p,
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=500)
    def test_social_surplus_equals_surplus_minus_harm(self, p, s_plus, s_minus, h):
        """social_surplus = expected_surplus - expected_harm."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p)
        ss = engine.social_surplus(interaction)
        es = engine.expected_surplus(p)
        eh = engine.expected_harm(p)
        assert ss == pytest.approx(es - eh, abs=1e-10)

    @given(p=st_p, rho_a=st_rho, rho_b=st_rho)
    @settings(max_examples=500)
    def test_welfare_with_externality_internalization(self, p, rho_a, rho_b):
        """With rho_a + rho_b = 1, total_welfare equals social_surplus + S_soft."""
        # When rho_a + rho_b = 1, agents fully internalize externality
        assume(rho_a + rho_b <= 1.0)
        config = PayoffConfig(rho_a=rho_a, rho_b=rho_b, w_rep=0.0)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, tau=0.0, c_a=0.0, c_b=0.0, r_a=0.0, r_b=0.0)
        welfare = engine.total_welfare(interaction)
        surplus = engine.expected_surplus(p)
        harm = engine.expected_harm(p)
        # total_welfare = S_soft - (rho_a + rho_b) * E_soft
        expected_welfare = surplus - (rho_a + rho_b) * harm
        assert welfare == pytest.approx(expected_welfare, abs=1e-10)


# ---------------------------------------------------------------------------
# Payoff breakdown properties
# ---------------------------------------------------------------------------


class TestPayoffBreakdownProperties:
    """Property-based tests for payoff breakdown consistency."""

    @given(
        p=st_p,
        theta=st_theta,
        tau=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        c_a=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        c_b=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_breakdown_total_matches_payoff(self, p, theta, tau, c_a, c_b):
        """Breakdown total should match the payoff computation for initiator."""
        config = PayoffConfig(theta=theta)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, tau=tau, c_a=c_a, c_b=c_b)
        breakdown = engine.payoff_breakdown_initiator(interaction)
        payoff = engine.payoff_initiator(interaction)
        assert breakdown.total == pytest.approx(payoff, abs=1e-10), (
            f"breakdown.total = {breakdown.total} != payoff = {payoff}"
        )

    @given(
        p=st_p,
        theta=st_theta,
        tau=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        c_a=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        c_b=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=500)
    def test_breakdown_counterparty_total_matches_payoff(self, p, theta, tau, c_a, c_b):
        """Breakdown total should match the payoff computation for counterparty."""
        config = PayoffConfig(theta=theta)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, tau=tau, c_a=c_a, c_b=c_b)
        breakdown = engine.payoff_breakdown_counterparty(interaction)
        payoff = engine.payoff_counterparty(interaction)
        assert breakdown.total == pytest.approx(payoff, abs=1e-10), (
            f"breakdown.total = {breakdown.total} != payoff = {payoff}"
        )

    @given(p=st_p, theta=st_theta)
    @settings(max_examples=500)
    def test_breakdown_surplus_shares_sum_to_surplus(self, p, theta):
        """Initiator and counterparty surplus shares sum to total surplus."""
        config = PayoffConfig(theta=theta)
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p)
        bd_a = engine.payoff_breakdown_initiator(interaction)
        bd_b = engine.payoff_breakdown_counterparty(interaction)
        surplus = engine.expected_surplus(p)
        assert bd_a.surplus_share + bd_b.surplus_share == pytest.approx(
            surplus, abs=1e-10
        )


# ---------------------------------------------------------------------------
# Soft metrics properties
# ---------------------------------------------------------------------------


class TestSoftMetricsProperties:
    """Property-based tests for SoftMetrics computations."""

    @given(
        p_values=st.lists(st_p, min_size=1, max_size=20),
    )
    @settings(max_examples=300)
    def test_toxicity_rate_in_unit_interval(self, p_values):
        """Toxicity rate is in [0, 1] for any set of accepted interactions."""
        interactions = [SoftInteraction(p=p, accepted=True) for p in p_values]
        metrics = SoftMetrics()
        tox = metrics.toxicity_rate(interactions)
        assert 0.0 <= tox <= 1.0 + 1e-12, f"toxicity = {tox} not in [0, 1]"

    @given(
        p_values=st.lists(st_p, min_size=1, max_size=20),
    )
    @settings(max_examples=300)
    def test_toxicity_rate_all_in_unit_interval(self, p_values):
        """Unconditional toxicity rate is in [0, 1]."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        tox = metrics.toxicity_rate_all(interactions)
        assert 0.0 <= tox <= 1.0 + 1e-12, f"toxicity_all = {tox} not in [0, 1]"

    @given(
        p_values=st.lists(st_p, min_size=1, max_size=20),
    )
    @settings(max_examples=300)
    def test_toxicity_complements_average_quality(self, p_values):
        """toxicity_rate_all + average_quality == 1 for same set."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        tox = metrics.toxicity_rate_all(interactions)
        avg_q = metrics.average_quality(interactions)
        assert tox + avg_q == pytest.approx(1.0, abs=1e-10), (
            f"toxicity_all + avg_quality = {tox + avg_q}, expected 1.0"
        )

    @given(
        p_accepted=st.lists(
            st.floats(
                min_value=0.6, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
        p_rejected=st.lists(
            st.floats(
                min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=300)
    def test_quality_gap_positive_when_good_accepted(self, p_accepted, p_rejected):
        """Quality gap is positive when high-quality interactions are accepted."""
        interactions = [SoftInteraction(p=p, accepted=True) for p in p_accepted] + [
            SoftInteraction(p=p, accepted=False) for p in p_rejected
        ]
        metrics = SoftMetrics()
        gap = metrics.quality_gap(interactions)
        # Average of accepted p values >= 0.6, average of rejected <= 0.4
        assert gap >= 0.0, f"quality_gap = {gap} should be non-negative"

    @given(
        p_accepted=st.lists(
            st.floats(
                min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
        p_rejected=st.lists(
            st.floats(
                min_value=0.7, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=300)
    def test_quality_gap_negative_when_bad_accepted(self, p_accepted, p_rejected):
        """Quality gap is negative when low-quality interactions are accepted (adverse selection)."""
        interactions = [SoftInteraction(p=p, accepted=True) for p in p_accepted] + [
            SoftInteraction(p=p, accepted=False) for p in p_rejected
        ]
        metrics = SoftMetrics()
        gap = metrics.quality_gap(interactions)
        assert gap <= 0.0, (
            f"quality_gap = {gap} should be negative for adverse selection"
        )

    @given(
        p_values=st.lists(st_p, min_size=2, max_size=20),
    )
    @settings(max_examples=300)
    def test_quality_gap_bounded(self, p_values):
        """Quality gap is bounded in [-1, 1] since p values are in [0, 1]."""
        accepted_flags = [i % 2 == 0 for i in range(len(p_values))]
        # Ensure at least one accepted and one rejected
        assume(any(accepted_flags) and not all(accepted_flags))
        interactions = [
            SoftInteraction(p=p, accepted=a)
            for p, a in zip(p_values, accepted_flags, strict=False)
        ]
        metrics = SoftMetrics()
        gap = metrics.quality_gap(interactions)
        assert -1.0 <= gap <= 1.0 + 1e-12, f"quality_gap = {gap} not in [-1, 1]"

    def test_toxicity_rate_empty(self):
        """Toxicity rate of empty list is 0."""
        metrics = SoftMetrics()
        assert metrics.toxicity_rate([]) == 0.0
        assert metrics.toxicity_rate_all([]) == 0.0

    def test_quality_gap_no_rejected(self):
        """Quality gap is 0 when there are no rejected interactions."""
        metrics = SoftMetrics()
        interactions = [SoftInteraction(p=0.8, accepted=True)]
        assert metrics.quality_gap(interactions) == 0.0

    def test_quality_gap_no_accepted(self):
        """Quality gap is 0 when there are no accepted interactions."""
        metrics = SoftMetrics()
        interactions = [SoftInteraction(p=0.2, accepted=False)]
        assert metrics.quality_gap(interactions) == 0.0

    @given(
        p_values=st.lists(st_p, min_size=1, max_size=20),
    )
    @settings(max_examples=300, deadline=None)
    def test_average_quality_in_unit_interval(self, p_values):
        """Average quality is in [0, 1]."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        avg = metrics.average_quality(interactions)
        assert 0.0 <= avg <= 1.0 + 1e-12, f"average_quality = {avg} not in [0, 1]"

    @given(
        p_values=st.lists(st_p, min_size=2, max_size=20),
    )
    @settings(max_examples=300)
    def test_quality_variance_non_negative(self, p_values):
        """Quality variance is non-negative."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        var = metrics.quality_variance(interactions)
        assert var >= -1e-12, f"variance = {var} is negative"

    @given(
        p_values=st.lists(st_p, min_size=2, max_size=20),
    )
    @settings(max_examples=300)
    def test_quality_variance_bounded(self, p_values):
        """Quality variance is bounded by 0.25 (max variance for [0,1] distribution)."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        var = metrics.quality_variance(interactions)
        assert var <= 0.25 + 1e-12, f"variance = {var} exceeds max possible 0.25"

    @given(
        p_values=st.lists(st_p, min_size=2, max_size=20),
    )
    @settings(max_examples=300)
    def test_quality_std_non_negative(self, p_values):
        """Quality standard deviation is non-negative."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        std = metrics.quality_std(interactions)
        assert std >= 0.0, f"std = {std} is negative"

    @given(p=st_p)
    @settings(max_examples=200)
    def test_single_interaction_zero_variance(self, p):
        """A single interaction has zero variance."""
        interactions = [SoftInteraction(p=p)]
        metrics = SoftMetrics()
        var = metrics.quality_variance(interactions)
        assert var == pytest.approx(0.0, abs=1e-12)

    @given(
        p_values=st.lists(st_p, min_size=1, max_size=10),
    )
    @settings(max_examples=300)
    def test_uncertain_fraction_in_unit_interval(self, p_values):
        """Uncertain fraction is in [0, 1]."""
        interactions = [SoftInteraction(p=p) for p in p_values]
        metrics = SoftMetrics()
        frac = metrics.uncertain_fraction(interactions, band=0.2)
        assert 0.0 <= frac <= 1.0 + 1e-12, f"uncertain_fraction = {frac} not in [0, 1]"


# ---------------------------------------------------------------------------
# Cross-module consistency properties
# ---------------------------------------------------------------------------


class TestCrossModuleProperties:
    """Property-based tests spanning multiple modules."""

    @given(
        task_progress=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        engagement=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=300)
    def test_proxy_to_payoff_pipeline(
        self, task_progress, engagement, s_plus, s_minus, h
    ):
        """Full pipeline from observables to payoff produces finite results."""
        computer = ProxyComputer()
        obs = ProxyObservables(
            task_progress_delta=task_progress,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = computer.compute_labels(obs)

        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)

        surplus = engine.expected_surplus(p)
        harm = engine.expected_harm(p)

        # Results should be finite
        assert surplus == surplus  # not NaN
        assert harm == harm  # not NaN
        assert harm >= -1e-12  # harm is non-negative
        # surplus is bounded by [-s_minus, s_plus]
        assert surplus >= -s_minus - 1e-10
        assert surplus <= s_plus + 1e-10

    @given(
        p=st_p,
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
        theta=st_theta,
        rho_a=st_rho,
        rho_b=st_rho,
    )
    @settings(max_examples=300)
    def test_full_payoff_is_finite(self, p, s_plus, s_minus, h, theta, rho_a, rho_b):
        """Full payoff computation produces finite results for any valid params."""
        config = PayoffConfig(
            s_plus=s_plus,
            s_minus=s_minus,
            h=h,
            theta=theta,
            rho_a=rho_a,
            rho_b=rho_b,
            w_rep=1.0,
        )
        engine = SoftPayoffEngine(config)
        interaction = SoftInteraction(p=p, accepted=True)

        pi_a = engine.payoff_initiator(interaction)
        pi_b = engine.payoff_counterparty(interaction)
        welfare = engine.total_welfare(interaction)
        social = engine.social_surplus(interaction)

        # All results should be finite (not NaN or inf)
        for val, name in [
            (pi_a, "pi_a"),
            (pi_b, "pi_b"),
            (welfare, "welfare"),
            (social, "social"),
        ]:
            assert val == val, f"{name} is NaN"
            assert abs(val) < float("inf"), f"{name} is infinite"

    @given(
        p=st_p,
        s_plus=st_positive_param,
        s_minus=st_positive_param,
        h=st_positive_param,
    )
    @settings(max_examples=300)
    def test_social_surplus_monotonic_in_p(self, p, s_plus, s_minus, h):
        """Social surplus is monotonically increasing in p."""
        config = PayoffConfig(s_plus=s_plus, s_minus=s_minus, h=h)
        engine = SoftPayoffEngine(config)

        # Check at two points: p and a slightly higher value
        p_high = min(1.0, p + 0.01)
        assume(p < p_high)

        interaction_low = SoftInteraction(p=p)
        interaction_high = SoftInteraction(p=p_high)

        ss_low = engine.social_surplus(interaction_low)
        ss_high = engine.social_surplus(interaction_high)

        assert ss_low <= ss_high + 1e-10, (
            f"social_surplus({p}) = {ss_low} > social_surplus({p_high}) = {ss_high}"
        )

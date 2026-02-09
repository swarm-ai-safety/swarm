"""Tests for the soft payoff engine."""

import pytest
from pydantic import ValidationError

from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.models.interaction import SoftInteraction


class TestPayoffConfig:
    """Tests for PayoffConfig validation."""

    def test_default_config(self):
        """Default config should be valid."""
        config = PayoffConfig()
        assert config is not None  # Pydantic auto-validates on creation

        assert config.s_plus == 2.0
        assert config.s_minus == 1.0
        assert config.theta == 0.5

    def test_invalid_s_plus(self):
        """Negative s_plus should raise."""
        with pytest.raises(ValidationError, match="s_plus"):
            PayoffConfig(s_plus=-1.0)

    def test_invalid_theta(self):
        """theta outside [0,1] should raise."""
        with pytest.raises(ValidationError, match="theta"):
            PayoffConfig(theta=1.5)

        with pytest.raises(ValidationError, match="theta"):
            PayoffConfig(theta=-0.1)


class TestExpectedValues:
    """Tests for expected surplus and harm calculations."""

    def test_expected_surplus_p1(self):
        """p=1 should give full positive surplus."""
        engine = SoftPayoffEngine()
        surplus = engine.expected_surplus(1.0)
        assert surplus == 2.0  # s_plus

    def test_expected_surplus_p0(self):
        """p=0 should give negative surplus."""
        engine = SoftPayoffEngine()
        surplus = engine.expected_surplus(0.0)
        assert surplus == -1.0  # -s_minus

    def test_expected_surplus_linear(self):
        """Expected surplus should be linear in p."""
        engine = SoftPayoffEngine()

        # S(p) = p * s_plus - (1-p) * s_minus
        # S(p) = p * 2 - (1-p) * 1 = 3p - 1

        assert engine.expected_surplus(0.0) == pytest.approx(-1.0)
        assert engine.expected_surplus(0.5) == pytest.approx(0.5)
        assert engine.expected_surplus(1.0) == pytest.approx(2.0)

        # Check linearity at intermediate points
        p1, p2 = 0.3, 0.7
        alpha = 0.4
        p_mid = alpha * p1 + (1 - alpha) * p2

        s1 = engine.expected_surplus(p1)
        s2 = engine.expected_surplus(p2)
        s_mid = engine.expected_surplus(p_mid)

        assert s_mid == pytest.approx(alpha * s1 + (1 - alpha) * s2)

    def test_expected_harm_p1(self):
        """p=1 means no harm."""
        engine = SoftPayoffEngine()
        harm = engine.expected_harm(1.0)
        assert harm == 0.0

    def test_expected_harm_p0(self):
        """p=0 means full harm."""
        engine = SoftPayoffEngine()
        harm = engine.expected_harm(0.0)
        assert harm == 2.0  # h


class TestPayoffInitiator:
    """Tests for initiator payoff calculations."""

    def test_payoff_linear_in_p(self):
        """Initiator payoff should be linear in p."""
        engine = SoftPayoffEngine()

        # Create interactions at different p values
        p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        payoffs = []

        for p in p_values:
            interaction = SoftInteraction(p=p)
            payoffs.append(engine.payoff_initiator(interaction))

        # Check linearity by verifying equal spacing
        diffs = [payoffs[i + 1] - payoffs[i] for i in range(len(payoffs) - 1)]
        for d in diffs[1:]:
            assert d == pytest.approx(diffs[0], rel=1e-6)

    def test_payoff_with_transfer(self):
        """Transfer should reduce initiator payoff."""
        engine = SoftPayoffEngine()

        interaction_no_tau = SoftInteraction(p=0.7, tau=0.0)
        interaction_with_tau = SoftInteraction(p=0.7, tau=0.5)

        payoff_no_tau = engine.payoff_initiator(interaction_no_tau)
        payoff_with_tau = engine.payoff_initiator(interaction_with_tau)

        assert payoff_with_tau == pytest.approx(payoff_no_tau - 0.5)

    def test_payoff_with_governance_cost(self):
        """Governance cost should reduce initiator payoff."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(p=0.7, c_a=0.2)
        breakdown = engine.payoff_breakdown_initiator(interaction)

        assert breakdown.governance_cost == 0.2

    def test_payoff_with_reputation(self):
        """Reputation should add to payoff."""
        engine = SoftPayoffEngine()

        interaction_no_rep = SoftInteraction(p=0.7, r_a=0.0)
        interaction_with_rep = SoftInteraction(p=0.7, r_a=0.3)

        payoff_no_rep = engine.payoff_initiator(interaction_no_rep)
        payoff_with_rep = engine.payoff_initiator(interaction_with_rep)

        # w_rep = 1.0 by default
        assert payoff_with_rep == pytest.approx(payoff_no_rep + 0.3)

    def test_payoff_boundary_p0(self):
        """p=0 (harmful) should give low payoff."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(p=0.0)
        payoff = engine.payoff_initiator(interaction)

        # θ * S_soft where S_soft = -s_minus = -1
        # payoff = 0.5 * (-1) = -0.5
        assert payoff == pytest.approx(-0.5)

    def test_payoff_boundary_p1(self):
        """p=1 (beneficial) should give high payoff."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(p=1.0)
        payoff = engine.payoff_initiator(interaction)

        # θ * S_soft where S_soft = s_plus = 2
        # payoff = 0.5 * 2 = 1.0
        assert payoff == pytest.approx(1.0)


class TestPayoffCounterparty:
    """Tests for counterparty payoff calculations."""

    def test_transfer_increases_counterparty_payoff(self):
        """Transfer should increase counterparty payoff."""
        engine = SoftPayoffEngine()

        interaction_no_tau = SoftInteraction(p=0.7, tau=0.0)
        interaction_with_tau = SoftInteraction(p=0.7, tau=0.5)

        payoff_no_tau = engine.payoff_counterparty(interaction_no_tau)
        payoff_with_tau = engine.payoff_counterparty(interaction_with_tau)

        assert payoff_with_tau == pytest.approx(payoff_no_tau + 0.5)

    def test_payoffs_sum_correctly(self):
        """Sum of payoffs should equal total welfare."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(
            p=0.7,
            tau=0.3,
            c_a=0.1,
            c_b=0.05,
            r_a=0.2,
            r_b=0.15,
        )

        pi_a = engine.payoff_initiator(interaction)
        pi_b = engine.payoff_counterparty(interaction)
        total = engine.total_welfare(interaction)

        assert total == pytest.approx(pi_a + pi_b)


class TestBreakdown:
    """Tests for payoff breakdown."""

    def test_breakdown_components_sum_to_total(self):
        """Breakdown components should sum to total."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(
            p=0.6,
            tau=0.2,
            c_a=0.1,
            r_a=0.15,
        )

        breakdown = engine.payoff_breakdown_initiator(interaction)

        reconstructed = (
            breakdown.surplus_share
            + breakdown.transfer
            - breakdown.governance_cost
            - breakdown.externality_cost
            + breakdown.reputation_bonus
        )

        assert breakdown.total == pytest.approx(reconstructed)

    def test_breakdown_transfer_sign(self):
        """Transfer should be negative for initiator, positive for counterparty."""
        engine = SoftPayoffEngine()

        interaction = SoftInteraction(p=0.7, tau=0.5)

        init_breakdown = engine.payoff_breakdown_initiator(interaction)
        counter_breakdown = engine.payoff_breakdown_counterparty(interaction)

        assert init_breakdown.transfer == -0.5  # Pays
        assert counter_breakdown.transfer == 0.5  # Receives


class TestExternalities:
    """Tests for externality handling."""

    def test_externality_internalization(self):
        """With rho > 0, agents internalize externality."""
        config = PayoffConfig(rho_a=0.5, rho_b=0.5)
        engine = SoftPayoffEngine(config)

        interaction = SoftInteraction(p=0.3)  # Low quality, high harm

        breakdown = engine.payoff_breakdown_initiator(interaction)

        # E_soft = (1-0.3) * 2 = 1.4
        # externality_cost = 0.5 * 1.4 = 0.7
        assert breakdown.expected_harm == pytest.approx(1.4)
        assert breakdown.externality_cost == pytest.approx(0.7)

    def test_no_externality_at_p1(self):
        """p=1 means no externality cost."""
        config = PayoffConfig(rho_a=1.0)
        engine = SoftPayoffEngine(config)

        interaction = SoftInteraction(p=1.0)
        breakdown = engine.payoff_breakdown_initiator(interaction)

        assert breakdown.externality_cost == 0.0


class TestBreakEven:
    """Tests for break-even calculations."""

    def test_break_even_p(self):
        """Verify break-even probability calculation."""
        engine = SoftPayoffEngine()

        # p* = s_minus / (s_plus + s_minus) = 1 / 3
        break_even = engine.break_even_p()
        assert break_even == pytest.approx(1 / 3)

        # At break-even, surplus should be 0
        surplus = engine.expected_surplus(break_even)
        assert surplus == pytest.approx(0.0, abs=1e-10)

    def test_social_break_even_p(self):
        """Verify social break-even probability."""
        engine = SoftPayoffEngine()

        # p* = (s_minus + h) / (s_plus + s_minus + h) = 3 / 5
        social_break_even = engine.social_break_even_p()
        assert social_break_even == pytest.approx(3 / 5)

        # At social break-even, social surplus should be 0
        interaction = SoftInteraction(p=social_break_even)
        social_surplus = engine.social_surplus(interaction)
        assert social_surplus == pytest.approx(0.0, abs=1e-10)


class TestSocialSurplus:
    """Tests for social surplus calculations."""

    def test_social_surplus_includes_externality(self):
        """Social surplus should account for externality."""
        engine = SoftPayoffEngine()

        interaction_good = SoftInteraction(p=1.0)
        interaction_bad = SoftInteraction(p=0.0)

        # Good: S_soft - E_soft = 2 - 0 = 2
        assert engine.social_surplus(interaction_good) == pytest.approx(2.0)

        # Bad: S_soft - E_soft = -1 - 2 = -3
        assert engine.social_surplus(interaction_bad) == pytest.approx(-3.0)

"""Tests for the CouncilProxyAuditor."""

from unittest.mock import MagicMock

import pytest

from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.proxy_auditor import CouncilProxyAuditor
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import Council, CouncilResult


def _make_council() -> Council:
    """Create a mock council."""

    async def mock_query(sys: str, usr: str) -> str:
        return '{"adjustment": 0.05, "reasoning": "Slightly undervalued"}'

    config = CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="m1",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
        ],
        min_members_required=1,
        timeout_per_member=5.0,
    )
    return Council(config=config, query_fns={"m1": mock_query})


def _make_observables(**kwargs) -> ProxyObservables:
    """Create test observables."""
    defaults = {
        "task_progress_delta": 0.5,
        "rework_count": 0,
        "verifier_rejections": 0,
        "tool_misuse_flags": 0,
        "counterparty_engagement_delta": 0.3,
    }
    defaults.update(kwargs)
    return ProxyObservables(**defaults)


class TestProxyAuditorPassthrough:
    def test_no_audit_returns_original(self):
        """With audit_probability=0, always returns original p."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=0.0,
            seed=42,
        )

        obs = _make_observables()
        v_hat, p = auditor.compute_labels(obs)
        v_hat_orig, p_orig = proxy.compute_labels(obs)

        assert v_hat == pytest.approx(v_hat_orig)
        assert p == pytest.approx(p_orig)

    def test_proxy_computer_accessible(self):
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(proxy, council)
        assert auditor.proxy_computer is proxy


class TestProxyAuditorClamping:
    def test_adjustment_clamped_to_max(self):
        """Adjustment is clamped to max_adjustment."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            max_adjustment=0.05,
            seed=42,
        )

        # Mock council returning a large adjustment
        result = CouncilResult(
            synthesis='{"adjustment": 0.5, "reasoning": "Way undervalued"}',
            responses={"m1": "big adjust"},
            members_responded=1,
            members_total=1,
            success=True,
        )
        auditor._deliberate_sync = MagicMock(return_value=result)

        obs = _make_observables()
        v_hat_orig, p_orig = proxy.compute_labels(obs)
        v_hat, p = auditor.compute_labels(obs)

        # Adjustment clamped to 0.05
        assert abs(p - p_orig) <= 0.05 + 1e-9

    def test_negative_adjustment_clamped(self):
        """Negative adjustments are also clamped."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            max_adjustment=0.03,
            seed=42,
        )

        result = CouncilResult(
            synthesis='{"adjustment": -0.8, "reasoning": "Way overvalued"}',
            responses={"m1": "big decrease"},
            members_responded=1,
            members_total=1,
            success=True,
        )
        auditor._deliberate_sync = MagicMock(return_value=result)

        obs = _make_observables()
        v_hat_orig, p_orig = proxy.compute_labels(obs)
        v_hat, p = auditor.compute_labels(obs)

        assert abs(p - p_orig) <= 0.03 + 1e-9


class TestProxyAuditorPInvariant:
    def test_p_stays_in_zero_one(self):
        """p is always clamped to [0, 1] after adjustment."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            max_adjustment=0.5,
            seed=42,
        )

        # Test with adjustment that would push p above 1
        result = CouncilResult(
            synthesis='{"adjustment": 0.5, "reasoning": "increase"}',
            responses={"m1": "up"},
            members_responded=1,
            members_total=1,
            success=True,
        )
        auditor._deliberate_sync = MagicMock(return_value=result)

        # High-quality observables -> high p
        obs = _make_observables(task_progress_delta=0.9, counterparty_engagement_delta=0.9)
        v_hat, p = auditor.compute_labels(obs)
        assert 0.0 <= p <= 1.0

        # Test with adjustment that would push p below 0
        result2 = CouncilResult(
            synthesis='{"adjustment": -0.5, "reasoning": "decrease"}',
            responses={"m1": "down"},
            members_responded=1,
            members_total=1,
            success=True,
        )
        auditor._deliberate_sync = MagicMock(return_value=result2)

        obs2 = _make_observables(task_progress_delta=-0.9, rework_count=5)
        v_hat2, p2 = auditor.compute_labels(obs2)
        assert 0.0 <= p2 <= 1.0


class TestProxyAuditorFailureFallback:
    def test_council_failure_returns_original(self):
        """On council failure, returns original proxy p."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            seed=42,
        )

        # Mock council failure
        result = CouncilResult(
            synthesis="",
            success=False,
            error="All members failed",
        )
        auditor._deliberate_sync = MagicMock(return_value=result)

        obs = _make_observables()
        v_hat, p = auditor.compute_labels(obs)
        v_hat_orig, p_orig = proxy.compute_labels(obs)

        assert p == pytest.approx(p_orig)

    def test_exception_returns_original(self):
        """On exception during audit, returns original proxy p."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            seed=42,
        )

        auditor._deliberate_sync = MagicMock(side_effect=RuntimeError("boom"))

        obs = _make_observables()
        v_hat, p = auditor.compute_labels(obs)
        v_hat_orig, p_orig = proxy.compute_labels(obs)

        assert p == pytest.approx(p_orig)

    def test_malformed_json_returns_original(self):
        """Malformed JSON in synthesis returns original proxy p."""
        proxy = ProxyComputer()
        council = _make_council()
        auditor = CouncilProxyAuditor(
            proxy_computer=proxy,
            council=council,
            audit_probability=1.0,
            seed=42,
        )

        result = CouncilResult(
            synthesis="I think the score is fine, no JSON here.",
            responses={"m1": "ok"},
            members_responded=1,
            members_total=1,
            success=True,
        )
        auditor._deliberate_sync = MagicMock(return_value=result)

        obs = _make_observables()
        v_hat, p = auditor.compute_labels(obs)
        v_hat_orig, p_orig = proxy.compute_labels(obs)

        assert p == pytest.approx(p_orig)

"""Tests for the council governance lever."""

from unittest.mock import MagicMock

import pytest

from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import Council, CouncilResult
from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.council_lever import CouncilGovernanceLever
from swarm.models.interaction import InteractionType, SoftInteraction


def _make_interaction(p: float = 0.7, accepted: bool = True) -> SoftInteraction:
    """Create a test interaction."""
    return SoftInteraction(
        interaction_id="test_ix",
        initiator="agent_a",
        counterparty="agent_b",
        interaction_type=InteractionType.COLLABORATION,
        v_hat=p * 2 - 1,
        p=p,
        accepted=accepted,
    )


def _make_council() -> Council:
    """Create a mock council."""

    async def mock_query(sys: str, usr: str) -> str:
        return "This interaction appears harmful with high risk."

    config = CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="m1",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
            CouncilMemberConfig(
                member_id="m2",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
        ],
        min_members_required=1,
        timeout_per_member=5.0,
        seed=42,
    )

    return Council(config=config, query_fns={"m1": mock_query, "m2": mock_query})


class TestCouncilGovernanceLever:
    def test_disabled_returns_empty(self):
        """Lever does nothing when disabled."""
        config = GovernanceConfig(council_lever_enabled=False)
        council = _make_council()
        lever = CouncilGovernanceLever(config, council=council, seed=42)

        ix = _make_interaction(p=0.3)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.lever_name == "council"
        assert effect.cost_a == 0.0

    def test_rejected_interaction_skipped(self):
        """Rejected interactions are not reviewed."""
        config = GovernanceConfig(council_lever_enabled=True)
        council = _make_council()
        lever = CouncilGovernanceLever(config, council=council, seed=42)

        ix = _make_interaction(p=0.3, accepted=False)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.cost_a == 0.0

    def test_review_probability_gate(self):
        """Lever respects the probabilistic review gate."""
        config = GovernanceConfig(
            council_lever_enabled=True,
            council_lever_review_probability=0.0,  # Never review
        )
        council = _make_council()
        lever = CouncilGovernanceLever(
            config, council=council, review_probability=0.0, seed=42
        )

        ix = _make_interaction(p=0.3)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.details.get("reviewed") is False

    def test_penalty_on_low_p_with_risk(self):
        """Penalty applied when p < 0.5 and council flags risk."""
        config = GovernanceConfig(
            council_lever_enabled=True,
            council_lever_review_probability=1.0,
            council_lever_penalty_multiplier=2.0,
        )

        # Mock council that always flags risk
        council = _make_council()
        lever = CouncilGovernanceLever(
            config, council=council, review_probability=1.0,
            penalty_multiplier=2.0, seed=42,
        )

        # Patch _run_council_sync to return a risk-flagging result
        result = CouncilResult(
            synthesis="This interaction appears harmful with high risk of exploitation.",
            responses={"m1": "harmful", "m2": "risky"},
            members_responded=2,
            members_total=2,
            success=True,
        )
        lever._run_council_sync = MagicMock(return_value=result)

        ix = _make_interaction(p=0.3)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.details["reviewed"] is True
        assert effect.details["penalty_applied"] is True
        assert effect.cost_a > 0.02  # More than just overhead
        assert "agent_a" in effect.reputation_deltas
        assert effect.reputation_deltas["agent_a"] < 0

    def test_no_penalty_on_high_p(self):
        """No penalty when p is high even with council review."""
        config = GovernanceConfig(
            council_lever_enabled=True,
            council_lever_review_probability=1.0,
        )
        council = _make_council()
        lever = CouncilGovernanceLever(
            config, council=council, review_probability=1.0, seed=42,
        )

        result = CouncilResult(
            synthesis="This interaction is beneficial and safe.",
            responses={"m1": "good", "m2": "safe"},
            members_responded=2,
            members_total=2,
            success=True,
        )
        lever._run_council_sync = MagicMock(return_value=result)

        ix = _make_interaction(p=0.8)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.details["reviewed"] is True
        assert effect.details["penalty_applied"] is False
        assert effect.cost_a == pytest.approx(0.02)  # Just overhead

    def test_graceful_failure(self):
        """Lever handles council failure gracefully."""
        config = GovernanceConfig(
            council_lever_enabled=True,
            council_lever_review_probability=1.0,
        )
        council = _make_council()
        lever = CouncilGovernanceLever(
            config, council=council, review_probability=1.0, seed=42,
        )

        result = CouncilResult(
            synthesis="",
            success=False,
            error="All members timed out",
        )
        lever._run_council_sync = MagicMock(return_value=result)

        ix = _make_interaction(p=0.3)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert effect.details["reviewed"] is True
        assert effect.details.get("council_error") is not None
        assert effect.cost_a == 0.0  # No penalty on failure

    def test_freeze_on_very_low_p(self):
        """Very low p with risk flags triggers freeze recommendation."""
        config = GovernanceConfig(
            council_lever_enabled=True,
            council_lever_review_probability=1.0,
        )
        council = _make_council()
        lever = CouncilGovernanceLever(
            config, council=council, review_probability=1.0, seed=42,
        )

        result = CouncilResult(
            synthesis="Extremely harmful interaction, risk level critical.",
            responses={"m1": "harmful", "m2": "harmful"},
            members_responded=2,
            members_total=2,
            success=True,
        )
        lever._run_council_sync = MagicMock(return_value=result)

        ix = _make_interaction(p=0.1)
        state = EnvState()
        effect = lever.on_interaction(ix, state)

        assert "agent_a" in effect.agents_to_freeze
        assert effect.details.get("freeze_recommended") is True

    def test_lever_name(self):
        config = GovernanceConfig()
        council = _make_council()
        lever = CouncilGovernanceLever(config, council=council)
        assert lever.name == "council"

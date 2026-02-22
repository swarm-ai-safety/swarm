"""Tests for the evolutionary game handler (gamescape integration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest
from gamescape.dynamics import PayoffMatrix

from swarm.core.evo_game_handler import (
    EvoGameConfig,
    EvolutionaryGameHandler,
    GameStrategy,
    resolve_strategy,
)
from swarm.core.proxy import ProxyObservables
from swarm.logging.event_bus import EventBus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def pd_config() -> EvoGameConfig:
    """Classic Prisoner's Dilemma: T>R>P>S."""
    return EvoGameConfig(a=3.0, b=0.0, c=5.0, d=1.0, seed=42, render_epoch_dynamics=False)


@pytest.fixture
def handler(pd_config: EvoGameConfig, event_bus: EventBus) -> EvolutionaryGameHandler:
    return EvolutionaryGameHandler(config=pd_config, event_bus=event_bus)


@dataclass
class _FakeAction:
    """Minimal action-like object for testing handle_action."""

    agent_id: str = ""
    action_type: str = "evo_game_move"
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


class TestStrategyResolution:
    def test_honest_maps_to_cooperate(self) -> None:
        assert resolve_strategy("honest") == GameStrategy.COOPERATE

    def test_opportunistic_maps_to_defect(self) -> None:
        assert resolve_strategy("opportunistic") == GameStrategy.DEFECT

    def test_deceptive_maps_to_defect(self) -> None:
        assert resolve_strategy("deceptive") == GameStrategy.DEFECT

    def test_explicit_cooperator(self) -> None:
        assert resolve_strategy("honest", "cooperator") == GameStrategy.COOPERATE

    def test_explicit_defector_overrides_type(self) -> None:
        assert resolve_strategy("honest", "defector") == GameStrategy.DEFECT

    def test_tit_for_tat(self) -> None:
        assert resolve_strategy("cautious_reciprocator", "tit_for_tat") == GameStrategy.TIT_FOR_TAT

    def test_grudger(self) -> None:
        assert resolve_strategy("honest", "grudger") == GameStrategy.GRUDGER

    def test_unknown_defaults_to_cooperate(self) -> None:
        assert resolve_strategy("some_unknown_type") == GameStrategy.COOPERATE


# ---------------------------------------------------------------------------
# Payoff computation
# ---------------------------------------------------------------------------


class TestPayoffComputation:
    def test_mutual_cooperation(self, handler: EvolutionaryGameHandler) -> None:
        a, b = handler.compute_game_payoffs(True, True)
        assert a == 3.0
        assert b == 3.0

    def test_mutual_defection(self, handler: EvolutionaryGameHandler) -> None:
        a, b = handler.compute_game_payoffs(False, False)
        assert a == 1.0
        assert b == 1.0

    def test_cooperator_vs_defector(self, handler: EvolutionaryGameHandler) -> None:
        a, b = handler.compute_game_payoffs(True, False)
        assert a == 0.0  # sucker
        assert b == 5.0  # temptation

    def test_defector_vs_cooperator(self, handler: EvolutionaryGameHandler) -> None:
        a, b = handler.compute_game_payoffs(False, True)
        assert a == 5.0  # temptation
        assert b == 0.0  # sucker

    def test_payoffs_match_gamescape_matrix(self, handler: EvolutionaryGameHandler) -> None:
        m = handler.matrix
        assert m.a == 3.0
        assert m.b == 0.0
        assert m.c == 5.0
        assert m.d == 1.0


# ---------------------------------------------------------------------------
# Observable generation from game payoffs
# ---------------------------------------------------------------------------


class TestObservableGeneration:
    def test_high_payoff_produces_good_observables(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        obs = handler._payoff_to_observables(5.0, 0.0)
        assert isinstance(obs, ProxyObservables)
        assert obs.task_progress_delta > 0.5  # high payoff → high progress
        assert obs.tool_misuse_flags == 0

    def test_low_payoff_produces_poor_observables(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        obs = handler._payoff_to_observables(0.0, 5.0)
        assert obs.task_progress_delta < 0.0  # sucker payoff → negative progress

    def test_mutual_cooperation_good_engagement(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        obs = handler._payoff_to_observables(3.0, 3.0)
        assert obs.counterparty_engagement_delta > 0.0

    def test_observables_bounded(self, handler: EvolutionaryGameHandler) -> None:
        for payoff_a in [0.0, 1.0, 3.0, 5.0]:
            for payoff_b in [0.0, 1.0, 3.0, 5.0]:
                obs = handler._payoff_to_observables(payoff_a, payoff_b)
                assert -1.0 <= obs.task_progress_delta <= 1.0
                assert -1.0 <= obs.counterparty_engagement_delta <= 1.0
                assert obs.rework_count >= 0
                assert obs.verifier_rejections >= 0


# ---------------------------------------------------------------------------
# Handle action
# ---------------------------------------------------------------------------


class TestHandleAction:
    def test_successful_action(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("alice", "honest")
        handler.register_agent_strategy("bob", "opportunistic")

        action = _FakeAction(
            agent_id="alice",
            metadata={"counterparty_id": "bob"},
        )
        result = handler.handle_action(action, state=None)

        assert result.success is True
        assert result.observables is not None
        assert result.metadata["initiator_cooperated"] is True
        assert result.metadata["counterparty_cooperated"] is False

    def test_missing_counterparty_fails(self, handler: EvolutionaryGameHandler) -> None:
        action = _FakeAction(agent_id="alice", metadata={})
        result = handler.handle_action(action, state=None)
        assert result.success is False

    def test_payoffs_in_metadata(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("alice", "honest", "cooperator")
        handler.register_agent_strategy("bob", "honest", "cooperator")

        action = _FakeAction(
            agent_id="alice",
            metadata={"counterparty_id": "bob"},
        )
        result = handler.handle_action(action, state=None)

        assert result.metadata["initiator_payoff"] == 3.0
        assert result.metadata["counterparty_payoff"] == 3.0


# ---------------------------------------------------------------------------
# Conditional strategies (TFT, Grudger)
# ---------------------------------------------------------------------------


class TestConditionalStrategies:
    def test_tft_cooperates_first(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("tft", "cautious_reciprocator", "tit_for_tat")
        handler.register_agent_strategy("defector", "opportunistic", "defector")

        action = _FakeAction(
            agent_id="tft",
            metadata={"counterparty_id": "defector"},
        )
        result = handler.handle_action(action, state=None)
        # TFT cooperates on first encounter
        assert result.metadata["initiator_cooperated"] is True

    def test_tft_retaliates_after_defection(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        handler.register_agent_strategy("tft", "cautious_reciprocator", "tit_for_tat")
        handler.register_agent_strategy("defector", "opportunistic", "defector")

        # First encounter: TFT cooperates, defector defects
        action1 = _FakeAction(
            agent_id="tft", metadata={"counterparty_id": "defector"}
        )
        handler.handle_action(action1, state=None)

        # Second encounter: TFT mirrors defector's last move (defect)
        action2 = _FakeAction(
            agent_id="tft", metadata={"counterparty_id": "defector"}
        )
        result2 = handler.handle_action(action2, state=None)
        assert result2.metadata["initiator_cooperated"] is False

    def test_grudger_defects_permanently(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        handler.register_agent_strategy("grudger", "honest", "grudger")
        handler.register_agent_strategy("defector", "opportunistic", "defector")

        # First encounter: grudger cooperates
        a1 = _FakeAction(agent_id="grudger", metadata={"counterparty_id": "defector"})
        r1 = handler.handle_action(a1, state=None)
        assert r1.metadata["initiator_cooperated"] is True

        # Second encounter: grudger defects (defector betrayed them)
        a2 = _FakeAction(agent_id="grudger", metadata={"counterparty_id": "defector"})
        r2 = handler.handle_action(a2, state=None)
        assert r2.metadata["initiator_cooperated"] is False

        # Third encounter: still defecting (grudge is permanent)
        a3 = _FakeAction(agent_id="grudger", metadata={"counterparty_id": "defector"})
        r3 = handler.handle_action(a3, state=None)
        assert r3.metadata["initiator_cooperated"] is False


# ---------------------------------------------------------------------------
# Population tracking across epochs
# ---------------------------------------------------------------------------


class TestPopulationTracking:
    def test_initial_coop_fraction(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("a1", "honest")
        handler.register_agent_strategy("a2", "honest")
        handler.register_agent_strategy("a3", "opportunistic")

        frac = handler._compute_coop_fraction()
        assert abs(frac - 2.0 / 3.0) < 1e-9

    def test_epoch_records_fraction(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("a1", "honest")
        handler.register_agent_strategy("a2", "opportunistic")

        handler.on_epoch_start(state=None)
        handler.on_epoch_end(state=None)

        trajectory = handler.get_population_trajectory()
        assert len(trajectory) == 1
        assert abs(trajectory[0] - 0.5) < 1e-9

    def test_multiple_epochs(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("a1", "honest")
        handler.register_agent_strategy("a2", "opportunistic")
        handler.register_agent_strategy("a3", "honest")

        for _ in range(3):
            handler.on_epoch_start(state=None)
            handler.on_epoch_end(state=None)

        trajectory = handler.get_population_trajectory()
        assert len(trajectory) == 3


# ---------------------------------------------------------------------------
# Gamescape rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_full_analysis_returns_string(
        self, handler: EvolutionaryGameHandler
    ) -> None:
        output = handler.render_full_analysis()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_replicator_prediction(self, handler: EvolutionaryGameHandler) -> None:
        pred = handler.get_replicator_prediction(x0=0.5, steps=100)
        assert len(pred) > 0
        for x in pred:
            assert 0.0 <= x <= 1.0


# ---------------------------------------------------------------------------
# Handler action types
# ---------------------------------------------------------------------------


class TestHandlerActionTypes:
    def test_handled_action_types(self) -> None:
        from swarm.agents.base import ActionType

        types = EvolutionaryGameHandler.handled_action_types()
        assert ActionType.EVO_GAME_MOVE in types

    def test_matrix_property(self, handler: EvolutionaryGameHandler) -> None:
        m = handler.matrix
        assert isinstance(m, PayoffMatrix)
        assert m.a == 3.0

    def test_strategies_property(self, handler: EvolutionaryGameHandler) -> None:
        handler.register_agent_strategy("x", "honest")
        strats = handler.strategies
        assert strats["x"] == GameStrategy.COOPERATE
        # Verify it's a copy
        strats["y"] = GameStrategy.DEFECT
        assert "y" not in handler.strategies

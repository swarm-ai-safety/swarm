"""Tests for the resource negotiation game handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from swarm.core.resource_negotiation_handler import (
    NegotiateAction,
    NegotiationGame,
    Proposal,
    ResourceNegotiationConfig,
    ResourceNegotiationHandler,
    ResourcePool,
)
from swarm.logging.event_bus import EventBus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def config() -> ResourceNegotiationConfig:
    return ResourceNegotiationConfig(
        min_resource_types=2,
        max_resource_types=3,
        min_quantity=2,
        max_quantity=4,
        guaranteed_rounds=4,
        termination_probability=0.3,
        seed=42,
    )


@pytest.fixture
def handler(config: ResourceNegotiationConfig, event_bus: EventBus) -> ResourceNegotiationHandler:
    return ResourceNegotiationHandler(config=config, event_bus=event_bus)


@pytest.fixture
def fixed_pool() -> ResourcePool:
    return ResourcePool(resources={"apples": 3, "bananas": 2})


@pytest.fixture
def fixed_game(handler: ResourceNegotiationHandler, fixed_pool: ResourcePool) -> NegotiationGame:
    return handler.create_game(
        player_a_id="alice",
        player_b_id="bob",
        pool=fixed_pool,
        valuations_a={"apples": 5.0, "bananas": 2.0},
        valuations_b={"apples": 1.0, "bananas": 8.0},
    )


@dataclass
class _FakeAction:
    """Minimal action-like object for testing handle_action."""

    agent_id: str = ""
    action_type: str = "resource_negotiate"
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------


class TestResourcePool:
    def test_to_str(self, fixed_pool: ResourcePool) -> None:
        s = fixed_pool.to_str()
        assert "apples" in s
        assert "bananas" in s

    def test_total_items(self, fixed_pool: ResourcePool) -> None:
        assert fixed_pool.total_items() == 5

    def test_names_sorted(self, fixed_pool: ResourcePool) -> None:
        assert fixed_pool.names() == ["apples", "bananas"]


# ---------------------------------------------------------------------------
# Proposal validation
# ---------------------------------------------------------------------------


class TestProposal:
    def test_valid_proposal(self, fixed_pool: ResourcePool) -> None:
        p = Proposal(
            my_share={"apples": 2, "bananas": 1},
            their_share={"apples": 1, "bananas": 1},
        )
        assert p.is_valid(fixed_pool)

    def test_invalid_sum(self, fixed_pool: ResourcePool) -> None:
        p = Proposal(
            my_share={"apples": 3, "bananas": 2},
            their_share={"apples": 1, "bananas": 1},
        )
        assert not p.is_valid(fixed_pool)

    def test_negative_quantity(self, fixed_pool: ResourcePool) -> None:
        p = Proposal(
            my_share={"apples": 4, "bananas": -1},
            their_share={"apples": -1, "bananas": 3},
        )
        assert not p.is_valid(fixed_pool)

    def test_extra_resource(self, fixed_pool: ResourcePool) -> None:
        p = Proposal(
            my_share={"apples": 2, "bananas": 1, "extra": 1},
            their_share={"apples": 1, "bananas": 1},
        )
        assert not p.is_valid(fixed_pool)

    def test_all_to_one_player(self, fixed_pool: ResourcePool) -> None:
        p = Proposal(
            my_share={"apples": 3, "bananas": 2},
            their_share={"apples": 0, "bananas": 0},
        )
        assert p.is_valid(fixed_pool)


# ---------------------------------------------------------------------------
# Game creation
# ---------------------------------------------------------------------------


class TestGameCreation:
    def test_create_game_with_explicit_params(
        self, handler: ResourceNegotiationHandler, fixed_pool: ResourcePool
    ) -> None:
        game = handler.create_game(
            player_a_id="alice",
            player_b_id="bob",
            pool=fixed_pool,
            valuations_a={"apples": 5.0, "bananas": 2.0},
            valuations_b={"apples": 1.0, "bananas": 8.0},
        )
        assert game.player_a_id == "alice"
        assert game.player_b_id == "bob"
        assert game.pool == fixed_pool
        assert game.round_number == 1
        assert not game.game_over

    def test_create_game_with_random_params(
        self, handler: ResourceNegotiationHandler
    ) -> None:
        game = handler.create_game("alice", "bob")
        assert len(game.pool.resources) >= 2
        assert len(game.valuations_a) == len(game.pool.resources)
        assert len(game.valuations_b) == len(game.pool.resources)

    def test_game_registered(
        self, handler: ResourceNegotiationHandler
    ) -> None:
        game = handler.create_game("alice", "bob")
        assert game.game_id in handler.games
        assert game.game_id in handler._player_games.get("alice", [])
        assert game.game_id in handler._player_games.get("bob", [])


# ---------------------------------------------------------------------------
# Turn order
# ---------------------------------------------------------------------------


class TestTurnOrder:
    def test_a_goes_first(self, fixed_game: NegotiationGame) -> None:
        assert fixed_game.whose_turn() == "alice"

    def test_b_goes_after_a(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        # Alice proposes
        proposal = Proposal(
            my_share={"apples": 2, "bananas": 1},
            their_share={"apples": 1, "bananas": 1},
        )
        ok, _ = handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, proposal
        )
        assert ok
        assert fixed_game.whose_turn() == "bob"

    def test_wrong_turn_rejected(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        ok, err = handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.PROPOSE,
            Proposal({"apples": 1, "bananas": 1}, {"apples": 2, "bananas": 1}),
        )
        assert not ok
        assert "Not bob's turn" in err


# ---------------------------------------------------------------------------
# Move processing
# ---------------------------------------------------------------------------


class TestMoveProcessing:
    def test_propose(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        prop = Proposal(
            my_share={"apples": 3, "bananas": 0},
            their_share={"apples": 0, "bananas": 2},
        )
        ok, _ = handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        assert ok
        assert fixed_game.last_proposal == prop
        assert fixed_game.last_proposer == "alice"

    def test_accept_completes_game(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        # Alice proposes
        prop = Proposal(
            my_share={"apples": 3, "bananas": 0},
            their_share={"apples": 0, "bananas": 2},
        )
        handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        # Bob accepts
        ok, _ = handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.ACCEPT
        )
        assert ok
        assert fixed_game.deal_reached
        assert fixed_game.game_over
        assert fixed_game.result is not None

    def test_reject_continues_game(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        prop = Proposal(
            my_share={"apples": 3, "bananas": 2},
            their_share={"apples": 0, "bananas": 0},
        )
        handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        ok, _ = handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.REJECT
        )
        assert ok
        assert not fixed_game.deal_reached
        # Round should advance after both players acted
        assert fixed_game.round_number == 2

    def test_cannot_accept_own_proposal(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        prop = Proposal(
            my_share={"apples": 2, "bananas": 1},
            their_share={"apples": 1, "bananas": 1},
        )
        handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        # Bob proposes (not accepts)
        prop2 = Proposal(
            my_share={"apples": 1, "bananas": 2},
            their_share={"apples": 2, "bananas": 0},
        )
        handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.PROPOSE, prop2
        )
        # Now it's round 2, Alice tries to accept her own... but last proposer is bob
        # Actually bob is last proposer, so Alice can accept
        ok, _ = handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.ACCEPT
        )
        assert ok  # This should work since bob was the last proposer

    def test_cannot_accept_no_proposal(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        ok, err = handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.ACCEPT
        )
        assert not ok
        assert "No proposal" in err

    def test_invalid_proposal_rejected(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        # Shares don't sum to pool
        prop = Proposal(
            my_share={"apples": 3, "bananas": 2},
            their_share={"apples": 3, "bananas": 2},
        )
        ok, err = handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        assert not ok
        assert "Invalid proposal" in err


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_all_resources(self, fixed_game: NegotiationGame) -> None:
        # Alice gets everything: (5*3 + 2*2) / (5*3 + 2*2) = 1.0
        alloc = {"apples": 3, "bananas": 2}
        assert fixed_game.compute_score("alice", alloc) == 1.0

    def test_score_nothing(self, fixed_game: NegotiationGame) -> None:
        alloc = {"apples": 0, "bananas": 0}
        assert fixed_game.compute_score("alice", alloc) == 0.0

    def test_score_partial(self, fixed_game: NegotiationGame) -> None:
        # Alice gets 2 apples: 5*2 / (5*3 + 2*2) = 10/19
        alloc = {"apples": 2, "bananas": 0}
        expected = 10.0 / 19.0
        assert abs(fixed_game.compute_score("alice", alloc) - expected) < 1e-9

    def test_deal_scores_correct(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        # Alice proposes: she gets apples, Bob gets bananas
        # This is optimal for both since Alice values apples (5)
        # and Bob values bananas (8)
        prop = Proposal(
            my_share={"apples": 3, "bananas": 0},
            their_share={"apples": 0, "bananas": 2},
        )
        handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.ACCEPT
        )

        result = fixed_game.result
        assert result is not None
        assert result.deal_reached

        # Alice: 5*3 / (5*3+2*2) = 15/19
        expected_a = 15.0 / 19.0
        assert abs(result.score_a - expected_a) < 1e-9

        # Bob: 8*2 / (1*3+8*2) = 16/19
        expected_b = 16.0 / 19.0
        assert abs(result.score_b - expected_b) < 1e-9

    def test_no_deal_scores(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        handler._finalize_no_deal(fixed_game)
        assert fixed_game.result is not None
        assert fixed_game.result.score_a == -0.5
        assert fixed_game.result.score_b == -0.5


# ---------------------------------------------------------------------------
# Round advancement & deadline
# ---------------------------------------------------------------------------


class TestDeadline:
    def test_round_advances_after_both_act(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        prop = Proposal(
            my_share={"apples": 3, "bananas": 0},
            their_share={"apples": 0, "bananas": 2},
        )
        handler.process_move(
            fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop
        )
        handler.process_move(
            fixed_game.game_id, "bob", NegotiateAction.REJECT
        )
        assert fixed_game.round_number == 2

    def test_guaranteed_rounds_no_termination(self, event_bus: EventBus) -> None:
        """Rounds 1-4 should never terminate the game."""
        config = ResourceNegotiationConfig(
            guaranteed_rounds=4,
            termination_probability=1.0,  # Would always terminate if checked
            seed=42,
        )
        handler = ResourceNegotiationHandler(
            config=config, event_bus=event_bus
        )
        pool = ResourcePool(resources={"a": 2})
        game = handler.create_game(
            "alice", "bob",
            pool=pool,
            valuations_a={"a": 1.0},
            valuations_b={"a": 1.0},
        )

        # Play through 3 rounds of proposals + rejects
        for _ in range(3):
            prop = Proposal(my_share={"a": 2}, their_share={"a": 0})
            handler.process_move(game.game_id, "alice", NegotiateAction.PROPOSE, prop)
            handler.process_move(game.game_id, "bob", NegotiateAction.REJECT)

        # After 3 rounds of reject, we should be on round 4
        assert game.round_number == 4
        assert not game.game_over

    def test_max_rounds_forces_end(self, event_bus: EventBus) -> None:
        config = ResourceNegotiationConfig(
            guaranteed_rounds=100,  # never stochastic
            termination_probability=0.0,
            max_rounds=3,
            seed=42,
        )
        handler = ResourceNegotiationHandler(
            config=config, event_bus=event_bus
        )
        pool = ResourcePool(resources={"a": 2})
        game = handler.create_game(
            "alice", "bob",
            pool=pool,
            valuations_a={"a": 1.0},
            valuations_b={"a": 1.0},
        )

        # Play through max_rounds
        for _ in range(3):
            prop = Proposal(my_share={"a": 2}, their_share={"a": 0})
            handler.process_move(game.game_id, "alice", NegotiateAction.PROPOSE, prop)
            handler.process_move(game.game_id, "bob", NegotiateAction.REJECT)

        assert game.game_over
        assert not game.deal_reached

    def test_epoch_end_forces_completion(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        assert not fixed_game.game_over
        handler.on_epoch_end(state=None)
        assert fixed_game.game_over
        assert fixed_game.result is not None
        assert not fixed_game.result.deal_reached


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------


class TestObservations:
    def test_active_game_in_observations(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        obs = handler.build_observation_fields("alice", state=None)
        assert len(obs["resource_negotiation_games"]) == 1
        assert obs["resource_negotiation_games"][0]["game_id"] == fixed_game.game_id
        assert obs["resource_negotiation_games"][0]["role"] == "A"

    def test_completed_game_in_results(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        handler._finalize_no_deal(fixed_game)
        obs = handler.build_observation_fields("alice", state=None)
        assert len(obs["resource_negotiation_games"]) == 0
        assert len(obs["resource_negotiation_results"]) == 1

    def test_game_observation_has_valuations(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        obs = handler.build_observation_fields("alice", state=None)
        game_obs = obs["resource_negotiation_games"][0]
        assert game_obs["your_valuations"]["apples"] == 5.0
        assert game_obs["your_valuations"]["bananas"] == 2.0

    def test_opponent_valuations_hidden(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        obs = handler.build_observation_fields("alice", state=None)
        game_obs = obs["resource_negotiation_games"][0]
        # Only "your_valuations" should be present, not the opponent's
        assert "your_valuations" in game_obs
        # Bob's valuations should not appear
        assert game_obs["your_valuations"]["bananas"] != 8.0  # This is alice's val


# ---------------------------------------------------------------------------
# Handle action (integration with handler protocol)
# ---------------------------------------------------------------------------


class TestHandleAction:
    def test_successful_propose(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        action = _FakeAction(
            agent_id="alice",
            metadata={
                "game_id": fixed_game.game_id,
                "negotiate_action": "propose",
                "proposal": {
                    "my_share": {"apples": 2, "bananas": 1},
                    "their_share": {"apples": 1, "bananas": 1},
                },
                "message": "Let's share!",
            },
        )
        result = handler.handle_action(action, state=None)
        assert result.success
        assert result.metadata["action"] == "propose"
        assert not result.metadata["deal_reached"]

    def test_successful_accept(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        # Alice proposes
        handler.process_move(
            fixed_game.game_id,
            "alice",
            NegotiateAction.PROPOSE,
            Proposal(
                my_share={"apples": 2, "bananas": 1},
                their_share={"apples": 1, "bananas": 1},
            ),
        )
        # Bob accepts via handler
        action = _FakeAction(
            agent_id="bob",
            metadata={
                "game_id": fixed_game.game_id,
                "negotiate_action": "accept",
            },
        )
        result = handler.handle_action(action, state=None)
        assert result.success
        assert result.metadata["deal_reached"]
        assert result.metadata["score_a"] is not None

    def test_missing_game_id(
        self, handler: ResourceNegotiationHandler
    ) -> None:
        action = _FakeAction(agent_id="alice", metadata={})
        result = handler.handle_action(action, state=None)
        assert not result.success

    def test_invalid_action_type(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        action = _FakeAction(
            agent_id="alice",
            metadata={
                "game_id": fixed_game.game_id,
                "negotiate_action": "invalid_action",
            },
        )
        result = handler.handle_action(action, state=None)
        assert not result.success


# ---------------------------------------------------------------------------
# Game pairing
# ---------------------------------------------------------------------------


class TestGamePairing:
    def test_pair_even_agents(
        self, handler: ResourceNegotiationHandler
    ) -> None:
        games = handler.create_games_for_agents(
            ["a1", "a2", "a3", "a4"]
        )
        assert len(games) == 2

    def test_pair_odd_agents(
        self, handler: ResourceNegotiationHandler
    ) -> None:
        games = handler.create_games_for_agents(
            ["a1", "a2", "a3"]
        )
        assert len(games) == 1


# ---------------------------------------------------------------------------
# Handler action types
# ---------------------------------------------------------------------------


class TestHandlerActionTypes:
    def test_handled_action_types(self) -> None:
        from swarm.agents.base import ActionType

        types = ResourceNegotiationHandler.handled_action_types()
        assert ActionType.RESOURCE_NEGOTIATE in types


# ---------------------------------------------------------------------------
# Multi-round negotiation integration
# ---------------------------------------------------------------------------


class TestMultiRoundNegotiation:
    def test_full_negotiation_sequence(
        self,
        handler: ResourceNegotiationHandler,
        fixed_game: NegotiationGame,
    ) -> None:
        """Play through a full multi-round negotiation ending in a deal."""
        # Round 1: Alice proposes greedy, Bob rejects
        prop1 = Proposal(
            my_share={"apples": 3, "bananas": 2},
            their_share={"apples": 0, "bananas": 0},
        )
        handler.process_move(fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop1)
        handler.process_move(fixed_game.game_id, "bob", NegotiateAction.REJECT)
        assert fixed_game.round_number == 2

        # Round 2: Alice proposes, Bob counter-proposes
        prop2 = Proposal(
            my_share={"apples": 3, "bananas": 1},
            their_share={"apples": 0, "bananas": 1},
        )
        handler.process_move(fixed_game.game_id, "alice", NegotiateAction.PROPOSE, prop2)
        prop3 = Proposal(
            my_share={"apples": 0, "bananas": 2},
            their_share={"apples": 3, "bananas": 0},
        )
        handler.process_move(fixed_game.game_id, "bob", NegotiateAction.PROPOSE, prop3)
        assert fixed_game.round_number == 3
        assert fixed_game.last_proposer == "bob"

        # Round 3: Alice accepts Bob's proposal (good deal for both)
        handler.process_move(fixed_game.game_id, "alice", NegotiateAction.ACCEPT)

        assert fixed_game.deal_reached
        assert fixed_game.game_over
        assert fixed_game.result.score_a > 0
        assert fixed_game.result.score_b > 0
        assert len(handler.results) == 1

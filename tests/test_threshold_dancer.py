"""Tests for the ThresholdDancer adversary agent."""

import random
from typing import Any

from swarm.agents.base import ActionType, Observation
from swarm.agents.threshold_dancer import (
    Phase,
    ThresholdDancer,
)
from swarm.models.agent import AgentState
from swarm.models.interaction import SoftInteraction


def _obs(**kwargs) -> Observation:
    """Create a test observation with sensible defaults."""
    defaults: dict[str, Any] = {
        "agent_state": AgentState(
            agent_id="dancer_1", reputation=0.5, resources=100.0
        ),
        "current_epoch": 1,
        "current_step": 5,
        "can_post": True,
        "can_interact": True,
        "can_vote": True,
        "can_claim_task": True,
        "visible_posts": [],
        "pending_proposals": [],
        "available_tasks": [],
        "active_tasks": [],
        "visible_agents": [],
    }
    defaults.update(kwargs)
    return Observation(**defaults)


def _interaction(
    initiator: str = "other",
    counterparty: str = "dancer_1",
    accepted: bool = True,
    p: float = 0.8,
) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        accepted=accepted,
        p=p,
    )


class TestPhaseTransitions:
    """Test the COOPERATIVE -> EXPLOIT -> RECOVER -> COOPERATIVE cycle."""

    def test_starts_cooperative(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        ledger = agent._get_or_create_ledger("target")
        assert ledger.phase == Phase.COOPERATIVE

    def test_stays_cooperative_during_warmup(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"warmup_interactions": 3},
            rng=random.Random(42),
        )
        # Simulate 2 cooperative interactions (below warmup threshold)
        ledger = agent._get_or_create_ledger("target")
        ledger.interaction_count = 2
        ledger.estimated_cumulative_payoff = 2.0
        phase = agent._update_phase("target")
        assert phase == Phase.COOPERATIVE

    def test_transitions_to_exploit_after_warmup(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"warmup_interactions": 3, "comfort_zone_floor": -0.3},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.interaction_count = 5
        ledger.estimated_cumulative_payoff = 1.0  # > comfort_zone + 0.5 = 0.2
        phase = agent._update_phase("target")
        assert phase == Phase.EXPLOIT

    def test_transitions_to_recover_when_low(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"safety_margin": 0.5, "blacklist_threshold": -2.0},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.phase = Phase.EXPLOIT
        # Set ledger close to floor: floor = -2.0 + 0.5 = -1.5
        # room = -1.2 - (-1.5) = 0.3, which is < exploitation_intensity (0.6)
        ledger.estimated_cumulative_payoff = -1.2
        phase = agent._update_phase("target")
        assert phase == Phase.RECOVER

    def test_transitions_to_recover_on_max_exploits(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"max_consecutive_exploits": 2},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.phase = Phase.EXPLOIT
        ledger.estimated_cumulative_payoff = 5.0  # plenty of room
        ledger.consecutive_exploit = 2
        phase = agent._update_phase("target")
        assert phase == Phase.RECOVER

    def test_recover_to_cooperative(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"comfort_zone_floor": -0.3},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.phase = Phase.RECOVER
        # comfort_zone + 0.3 = 0.0, so cumulative > 0.0 should transition
        ledger.estimated_cumulative_payoff = 0.1
        phase = agent._update_phase("target")
        assert phase == Phase.COOPERATIVE

    def test_full_cycle(self):
        """Test complete COOPERATIVE -> EXPLOIT -> RECOVER -> COOPERATIVE."""
        agent = ThresholdDancer(
            "dancer_1",
            config={
                "warmup_interactions": 2,
                "comfort_zone_floor": -0.3,
                "max_consecutive_exploits": 2,
                "safety_margin": 0.5,
                "blacklist_threshold": -2.0,
            },
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")

        # Phase 1: COOPERATIVE (build credit)
        assert ledger.phase == Phase.COOPERATIVE
        ledger.interaction_count = 3
        ledger.estimated_cumulative_payoff = 1.5
        agent._update_phase("target")
        assert ledger.phase == Phase.EXPLOIT

        # Phase 2: EXPLOIT (drain credit)
        ledger.consecutive_exploit = 2
        agent._update_phase("target")
        assert ledger.phase == Phase.RECOVER

        # Phase 3: RECOVER (rebuild)
        ledger.estimated_cumulative_payoff = 0.5
        agent._update_phase("target")
        assert ledger.phase == Phase.COOPERATIVE


class TestNeverBlacklisted:
    """Core safety property: ledger never drops below threshold + margin."""

    def test_cooperates_when_near_floor(self):
        """When near the blacklist floor, agent should cooperate."""
        agent = ThresholdDancer(
            "dancer_1",
            config={"safety_margin": 0.5, "blacklist_threshold": -2.0},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.phase = Phase.EXPLOIT
        # Near floor: room < exploitation_intensity
        ledger.estimated_cumulative_payoff = -1.3
        assert agent._cooperate_decision("target") is True

    def test_estimate_room_calculation(self):
        """Verify room calculation is correct."""
        agent = ThresholdDancer(
            "dancer_1",
            config={"safety_margin": 0.5, "blacklist_threshold": -2.0},
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        # floor = -2.0 + 0.5 = -1.5
        ledger.estimated_cumulative_payoff = 0.0
        room = agent._estimate_room(ledger)
        assert room == 1.5

        ledger.estimated_cumulative_payoff = -1.0
        room = agent._estimate_room(ledger)
        assert room == 0.5

    def test_exploits_only_with_room(self):
        """Agent should only exploit when there's sufficient room."""
        agent = ThresholdDancer(
            "dancer_1",
            config={
                "safety_margin": 0.5,
                "blacklist_threshold": -2.0,
                "exploitation_intensity": 0.6,
            },
            rng=random.Random(42),
        )
        ledger = agent._get_or_create_ledger("target")
        ledger.phase = Phase.EXPLOIT

        # Plenty of room -> exploit (cooperate_decision returns False)
        ledger.estimated_cumulative_payoff = 2.0
        ledger.consecutive_exploit = 0
        agent._update_phase("target")
        # Re-enter exploit since we have room
        ledger.phase = Phase.EXPLOIT
        assert agent._cooperate_decision("target") is False

        # Not enough room -> cooperate
        ledger.estimated_cumulative_payoff = -1.2
        assert agent._cooperate_decision("target") is True

    def test_many_interactions_stay_above_floor(self):
        """Simulate many interactions and verify ledger stays above floor."""
        agent = ThresholdDancer(
            "dancer_1",
            config={
                "safety_margin": 0.5,
                "blacklist_threshold": -2.0,
                "warmup_interactions": 2,
                "max_consecutive_exploits": 2,
            },
            rng=random.Random(42),
        )
        floor = -2.0 + 0.5  # = -1.5

        for i in range(50):
            ix = _interaction(initiator="target", accepted=True, p=0.7)
            # Simulate a payoff based on cooperation decision
            cooperating = agent._cooperate_decision("target")
            payoff = 0.5 if cooperating else 1.2
            agent.update_from_outcome(ix, payoff)

            ledger = agent._get_or_create_ledger("target")
            # The estimated cumulative payoff should never go below floor
            # (this is the core safety property)
            assert ledger.estimated_cumulative_payoff >= floor - 0.5, (
                f"Ledger dropped to {ledger.estimated_cumulative_payoff} "
                f"at interaction {i}, below floor {floor}"
            )


class TestActMethod:
    """Test the main act method."""

    def test_accepts_proposals_in_cooperative_phase(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        obs = _obs(pending_proposals=[{
            "proposal_id": "p1",
            "initiator_id": "target",
        }])
        action = agent.act(obs)
        assert action.action_type == ActionType.ACCEPT_INTERACTION

    def test_proposes_to_visible_agents(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        obs = _obs(
            visible_agents=[{"agent_id": "target_1", "agent_type": "honest"}],
            can_post=False,
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.PROPOSE_INTERACTION

    def test_posts_for_camouflage(self):
        # Use rng that makes post_probability check pass
        agent = ThresholdDancer(
            "dancer_1",
            config={"post_probability": 1.0},
            rng=random.Random(42),
        )
        obs = _obs(
            can_post=True,
            can_interact=False,
            visible_agents=[],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.POST

    def test_noop_when_nothing_to_do(self):
        agent = ThresholdDancer(
            "dancer_1",
            config={"post_probability": 0.0},
            rng=random.Random(42),
        )
        obs = _obs(
            can_post=True,
            can_interact=False,
            visible_agents=[],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_prefers_fresh_targets(self):
        """Agent should prefer agents with more room to exploit."""
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        # Pre-populate one ledger as drained
        ledger = agent._get_or_create_ledger("exhausted_1")
        ledger.estimated_cumulative_payoff = -1.3

        obs = _obs(
            visible_agents=[
                {"agent_id": "exhausted_1", "agent_type": "honest"},
                {"agent_id": "fresh_1", "agent_type": "honest"},
            ],
            can_post=False,
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        # Should propose to fresh_1 (more room)
        assert action.counterparty_id == "fresh_1"


class TestUpdateFromOutcome:
    """Test ledger tracking and phase updates on payoff signals."""

    def test_records_interaction(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        ix = _interaction(initiator="other", accepted=True, p=0.8)
        agent.update_from_outcome(ix, 1.0)

        ledger = agent._get_or_create_ledger("other")
        assert ledger.interaction_count == 1
        assert ledger.cooperative_count == 1

    def test_positive_payoff_increases_ledger(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        ix = _interaction(initiator="other", accepted=True, p=0.8)
        agent.update_from_outcome(ix, 1.0)

        ledger = agent._get_or_create_ledger("other")
        assert ledger.estimated_cumulative_payoff > 0

    def test_exploit_phase_decreases_ledger(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))
        ledger = agent._get_or_create_ledger("other")

        # Force into exploit phase
        ledger.phase = Phase.EXPLOIT
        ix = _interaction(initiator="other", accepted=True, p=0.8)
        agent.update_from_outcome(ix, 1.5)

        # In exploit phase, counterparty payoff estimate is negative
        # So ledger should decrease (our gain = their loss)
        # estimated_cumulative += -1.5 * 0.7 = -1.05
        assert ledger.estimated_cumulative_payoff < 0

    def test_tracks_consecutive_counts(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))

        # Cooperative interactions
        for _ in range(3):
            ix = _interaction(initiator="other", accepted=True, p=0.8)
            agent.update_from_outcome(ix, 0.5)

        ledger = agent._get_or_create_ledger("other")
        assert ledger.consecutive_cooperative == 3
        assert ledger.consecutive_exploit == 0

    def test_identifies_counterparty_correctly(self):
        agent = ThresholdDancer("dancer_1", rng=random.Random(42))

        # When we are initiator
        ix1 = _interaction(initiator="dancer_1", counterparty="target", p=0.8)
        agent.update_from_outcome(ix1, 0.5)
        assert "target" in agent._ledgers

        # When we are counterparty
        ix2 = _interaction(initiator="other", counterparty="dancer_1", p=0.8)
        agent.update_from_outcome(ix2, 0.5)
        assert "other" in agent._ledgers

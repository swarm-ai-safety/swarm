"""Targeted coverage tests for adaptive_adversary.py and llm_agent.py."""

import random
from unittest.mock import patch

import pytest

from swarm.agents.adaptive_adversary import (
    AdaptiveAdversary,
    AdversaryMemory,
    AttackStrategy,
    StrategyPerformance,
)
from swarm.agents.base import ActionType, InteractionProposal, Observation
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig, LLMProvider, PersonaType
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def obs(**kwargs) -> Observation:
    """Create a test observation with sensible defaults."""
    defaults = {
        "agent_state": AgentState(
            agent_id="test_agent", reputation=0.5, resources=100.0
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


# ===================================================================
# StrategyPerformance property tests
# ===================================================================


class TestStrategyPerformance:
    """Tests for StrategyPerformance dataclass properties."""

    def test_success_rate_zero_attempts(self):
        sp = StrategyPerformance(strategy=AttackStrategy.COLLUSION)
        assert sp.success_rate == 0.5  # prior

    def test_success_rate_with_data(self):
        sp = StrategyPerformance(
            strategy=AttackStrategy.COLLUSION, attempts=10, successes=7
        )
        assert sp.success_rate == pytest.approx(0.7)

    def test_detection_rate_zero_attempts(self):
        sp = StrategyPerformance(strategy=AttackStrategy.SYBIL)
        assert sp.detection_rate == 0.5  # prior

    def test_detection_rate_with_data(self):
        sp = StrategyPerformance(strategy=AttackStrategy.SYBIL, attempts=20, detected=5)
        assert sp.detection_rate == pytest.approx(0.25)

    def test_evasion_rate(self):
        sp = StrategyPerformance(
            strategy=AttackStrategy.LOW_PROFILE, attempts=10, detected=3
        )
        assert sp.evasion_rate == pytest.approx(0.7)

    def test_evasion_rate_zero_attempts(self):
        sp = StrategyPerformance(strategy=AttackStrategy.LOW_PROFILE)
        # detection_rate == 0.5 prior => evasion == 0.5
        assert sp.evasion_rate == pytest.approx(0.5)

    def test_avg_payoff_zero_attempts(self):
        sp = StrategyPerformance(strategy=AttackStrategy.GRIEFING)
        assert sp.avg_payoff == 0.0

    def test_avg_payoff_with_data(self):
        sp = StrategyPerformance(
            strategy=AttackStrategy.GRIEFING, attempts=4, total_payoff=12.0
        )
        assert sp.avg_payoff == pytest.approx(3.0)

    def test_effectiveness_score_few_attempts(self):
        sp = StrategyPerformance(strategy=AttackStrategy.MIMICRY, attempts=2)
        assert sp.effectiveness_score == 0.5  # not enough data

    def test_effectiveness_score_enough_attempts(self):
        sp = StrategyPerformance(
            strategy=AttackStrategy.MIMICRY,
            attempts=10,
            successes=8,
            detected=2,
            total_payoff=10.0,
        )
        expected = (
            0.3 * (8 / 10) + 0.4 * (1.0 - 2 / 10) + 0.3 * max(0, (10.0 / 10) / 5.0)
        )
        assert sp.effectiveness_score == pytest.approx(expected)

    def test_effectiveness_score_negative_payoff_clamped(self):
        sp = StrategyPerformance(
            strategy=AttackStrategy.MIMICRY,
            attempts=5,
            successes=0,
            detected=5,
            total_payoff=-50.0,
        )
        expected = 0.3 * 0.0 + 0.4 * 0.0 + 0.3 * 0.0
        assert sp.effectiveness_score == pytest.approx(expected)


# ===================================================================
# AdversaryMemory tests
# ===================================================================


class TestAdversaryMemory:
    """Tests for AdversaryMemory.update_heat."""

    def test_update_heat_detected_true(self):
        mem = AdversaryMemory()
        mem.current_heat_level = 0.2
        mem.epochs_since_detection = 5
        mem.update_heat(detected=True, penalty=0.0)
        assert mem.current_heat_level == pytest.approx(0.5)
        assert mem.epochs_since_detection == 0

    def test_update_heat_detected_false_decay(self):
        mem = AdversaryMemory()
        mem.current_heat_level = 0.5
        mem.epochs_since_detection = 0
        mem.update_heat(detected=False, penalty=0.0)
        # epochs_since_detection incremented to 1, decay = 0.1*1 = 0.1
        assert mem.epochs_since_detection == 1
        assert mem.current_heat_level == pytest.approx(0.4)

    def test_update_heat_penalty_positive(self):
        mem = AdversaryMemory()
        mem.current_heat_level = 0.0
        mem.epochs_since_penalty = 3
        mem.update_heat(detected=False, penalty=5.0)
        # Not detected: epochs_since_detection -> 1, decay = 0.1
        # heat after detection update = max(0, 0.0 - 0.1) = 0.0
        # Penalty > 0: heat += 0.2
        assert mem.current_heat_level == pytest.approx(0.2)
        assert mem.epochs_since_penalty == 0

    def test_update_heat_penalty_zero(self):
        mem = AdversaryMemory()
        mem.epochs_since_penalty = 0
        mem.update_heat(detected=False, penalty=0.0)
        assert mem.epochs_since_penalty == 1

    def test_update_heat_capped_at_one(self):
        mem = AdversaryMemory()
        mem.current_heat_level = 0.9
        mem.update_heat(detected=True, penalty=10.0)
        assert mem.current_heat_level <= 1.0


# ===================================================================
# AdaptiveAdversary tests
# ===================================================================


class TestAdaptiveAdversaryInit:
    """Tests for AdaptiveAdversary initialisation."""

    def test_default_init(self):
        aa = AdaptiveAdversary(agent_id="adv1")
        assert aa.agent_id == "adv1"
        assert aa.agent_type == AgentType.ADVERSARIAL
        assert aa.learning_rate == 0.1
        assert aa.exploration_rate == 0.2
        assert aa.heat_threshold == 0.6

    def test_config_overrides(self):
        aa = AdaptiveAdversary(
            agent_id="adv2",
            config={
                "learning_rate": 0.5,
                "exploration_rate": 0.9,
                "heat_threshold": 0.3,
            },
        )
        assert aa.learning_rate == 0.5
        assert aa.exploration_rate == 0.9
        assert aa.heat_threshold == 0.3


class TestSelectStrategy:
    """Tests for AdaptiveAdversary.select_strategy."""

    def test_high_heat_selects_evasion(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.9
        for _ in range(20):
            strat = aa.select_strategy()
            assert strat in {
                AttackStrategy.LOW_PROFILE,
                AttackStrategy.INTERMITTENT,
                AttackStrategy.MIMICRY,
            }

    def test_exploration_returns_any_strategy(self):
        aa = AdaptiveAdversary(agent_id="adv", exploration_rate=1.0)
        aa.memory.current_heat_level = 0.0
        strategies_seen = {aa.select_strategy() for _ in range(200)}
        # With exploration_rate=1.0 and 200 samples, should see many strategies
        assert len(strategies_seen) > 1

    def test_exploitation_picks_best(self):
        aa = AdaptiveAdversary(agent_id="adv", exploration_rate=0.0)
        aa.memory.current_heat_level = 0.0
        # Give COLLUSION a high effectiveness
        perf = aa.strategy_performance[AttackStrategy.COLLUSION]
        perf.attempts = 10
        perf.successes = 9
        perf.detected = 0
        perf.total_payoff = 50.0
        # All others stay at default (0.5 effectiveness)
        strat = aa.select_strategy()
        assert strat == AttackStrategy.COLLUSION


class TestActDispatching:
    """Tests for AdaptiveAdversary.act dispatching to each strategy."""

    def _make_non_laying_low_adversary(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        aa.attack_cooldown = 0
        return aa

    def _obs_not_laying_low(self):
        return obs(
            agent_state=AgentState(agent_id="adv", reputation=2.0, resources=200.0)
        )

    def test_act_reputation_farming(self):
        aa = self._make_non_laying_low_adversary()
        aa.exploration_rate = 0.0
        aa.current_strategy = AttackStrategy.REPUTATION_FARMING
        # Force select_strategy to return REPUTATION_FARMING
        aa.select_strategy = lambda: AttackStrategy.REPUTATION_FARMING
        action = aa.act(self._obs_not_laying_low())
        assert action.action_type in {
            ActionType.POST,
            ActionType.PROPOSE_INTERACTION,
            ActionType.NOOP,
        }

    def test_act_collusion_with_allies(self):
        aa = self._make_non_laying_low_adversary()
        aa.memory.potential_allies.add("ally1")
        aa.select_strategy = lambda: AttackStrategy.COLLUSION
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=2.0, resources=200.0),
            visible_agents=[{"agent_id": "ally1", "agent_type": "adversarial"}],
        )
        action = aa.act(o)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.target_id == "ally1"

    def test_act_collusion_without_allies(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.COLLUSION
        action = aa.act(self._obs_not_laying_low())
        # Falls through to _act_opportunistic
        assert action.action_type in {
            ActionType.POST,
            ActionType.PROPOSE_INTERACTION,
        }

    def test_act_low_profile(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.LOW_PROFILE
        actions = {aa.act(self._obs_not_laying_low()).action_type for _ in range(50)}
        # Should see NOOP and possibly POST
        assert ActionType.NOOP in actions or ActionType.POST in actions

    def test_act_mimicry_with_honest_agents(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.MIMICRY
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=2.0, resources=200.0),
            visible_agents=[{"agent_id": "h1", "agent_type": "honest"}],
        )
        actions = {aa.act(o).action_type for _ in range(50)}
        assert actions <= {ActionType.POST, ActionType.PROPOSE_INTERACTION}

    def test_act_mimicry_without_honest_agents(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.MIMICRY
        action = aa.act(self._obs_not_laying_low())
        # Falls to _act_innocently
        assert action.action_type in {
            ActionType.POST,
            ActionType.VOTE,
            ActionType.NOOP,
        }

    def test_act_threshold_dancing_room_above(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.THRESHOLD_DANCING
        aa.memory.estimated_toxicity_threshold = 0.9
        # avg_toxicity will default to 0.0 via getattr fallback
        action = aa.act(self._obs_not_laying_low())
        # room = 0.9 > 0.2 => _act_opportunistic
        assert action.action_type in {
            ActionType.POST,
            ActionType.PROPOSE_INTERACTION,
        }

    def test_act_threshold_dancing_room_below(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.THRESHOLD_DANCING
        aa.memory.estimated_toxicity_threshold = 0.05
        # room = 0.05 - 0 = 0.05 < 0.2 => _act_innocently
        action = aa.act(self._obs_not_laying_low())
        assert action.action_type in {
            ActionType.POST,
            ActionType.VOTE,
            ActionType.NOOP,
        }

    def test_act_timing_attack_good_timing_with_targets(self):
        aa = self._make_non_laying_low_adversary()
        aa.memory.epochs_since_detection = 10
        aa.memory.vulnerable_targets.add("victim1")
        aa.select_strategy = lambda: AttackStrategy.TIMING_ATTACK
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=5.0, resources=200.0),
            visible_agents=[{"agent_id": "victim1", "agent_type": "honest"}],
        )
        action = aa.act(o)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.target_id == "victim1"
        assert aa.attack_cooldown == 3

    def test_act_timing_attack_bad_timing(self):
        aa = self._make_non_laying_low_adversary()
        aa.memory.epochs_since_detection = (
            2  # >= 2 so no lay_low, but <= 3 so bad timing
        )
        aa.select_strategy = lambda: AttackStrategy.TIMING_ATTACK
        action = aa.act(self._obs_not_laying_low())
        # Falls to _act_reputation_farming (reputation < threshold => POST)
        assert action.action_type in {
            ActionType.POST,
            ActionType.PROPOSE_INTERACTION,
            ActionType.NOOP,
        }

    def test_act_governance_gaming(self):
        aa = self._make_non_laying_low_adversary()
        aa.select_strategy = lambda: AttackStrategy.GOVERNANCE_GAMING
        action = aa.act(self._obs_not_laying_low())
        # Delegates to _act_threshold_dancing
        assert action.action_type in {
            ActionType.POST,
            ActionType.VOTE,
            ActionType.NOOP,
            ActionType.PROPOSE_INTERACTION,
        }

    def test_act_adaptive_blend(self):
        aa = self._make_non_laying_low_adversary()
        aa.exploration_rate = 0.0  # deterministic strategy selection
        # Give only one strategy data so it gets selected deterministically
        perf = aa.strategy_performance[AttackStrategy.LOW_PROFILE]
        perf.attempts = 10
        perf.successes = 10
        perf.detected = 0
        perf.total_payoff = 50.0
        # Set all others to very low
        for s, p in aa.strategy_performance.items():
            if s != AttackStrategy.LOW_PROFILE and p.attempts < 3:
                p.attempts = 0  # keeps effectiveness_score at 0.5

        # Call _act_adaptive_blend directly to avoid recursion;
        # the real select_strategy (with exploration_rate=0) will pick
        # LOW_PROFILE on the recursive self.act() call.
        action = aa._act_adaptive_blend(self._obs_not_laying_low())
        assert action.action_type in set(ActionType)

    def test_act_default_strategy(self):
        """Test the else branch (default to _act_opportunistic)."""
        aa = self._make_non_laying_low_adversary()
        # Use SYBIL which is not in the if-elif chain
        aa.select_strategy = lambda: AttackStrategy.SYBIL
        action = aa.act(self._obs_not_laying_low())
        assert action.action_type in {
            ActionType.POST,
            ActionType.PROPOSE_INTERACTION,
        }


class TestUpdateMemory:
    """Tests for AdaptiveAdversary._update_memory."""

    def test_identify_allies_and_targets(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            visible_agents=[
                {"agent_id": "other_adv", "agent_type": "adversarial"},
                {"agent_id": "honest1", "agent_type": "honest", "reputation": 5.0},
                {"agent_id": "honest2", "agent_type": "honest", "reputation": 1.0},
            ]
        )
        aa._update_memory(o)
        assert "other_adv" in aa.memory.potential_allies
        assert "honest1" in aa.memory.vulnerable_targets
        assert "honest2" not in aa.memory.vulnerable_targets

    def test_does_not_add_self_as_ally(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            visible_agents=[
                {"agent_id": "adv", "agent_type": "adversarial"},
            ]
        )
        aa._update_memory(o)
        assert "adv" not in aa.memory.potential_allies

    def test_ecosystem_metrics_threshold_update(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(ecosystem_metrics={"governance": {"freeze_threshold_toxicity": 0.8}})
        aa._update_memory(o)
        assert aa.memory.estimated_toxicity_threshold == pytest.approx(0.8 * 0.9)


class TestShouldLayLow:
    """Tests for AdaptiveAdversary._should_lay_low."""

    def test_high_heat(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.9
        aa.memory.epochs_since_detection = 10
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa._should_lay_low(o) is True

    def test_recent_detection(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 1
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa._should_lay_low(o) is True

    def test_low_resources(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        o = obs(agent_state=AgentState(agent_id="adv", resources=10.0))
        assert aa._should_lay_low(o) is True

    def test_cooldown_active(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        aa.attack_cooldown = 2
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa._should_lay_low(o) is True
        assert aa.attack_cooldown == 1  # decremented

    def test_safe_conditions(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        aa.attack_cooldown = 0
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa._should_lay_low(o) is False


class TestActInnocently:
    """Tests for AdaptiveAdversary._act_innocently."""

    def test_post_path(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        random.seed(0)
        # Collect many actions to cover all branches
        types = set()
        for i in range(100):
            random.seed(i)
            action = aa._act_innocently(o)
            types.add(action.action_type)
        # With no visible_posts, we should see POST and NOOP
        assert ActionType.POST in types
        assert ActionType.NOOP in types

    def test_vote_path(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            visible_posts=[
                {"post_id": "p1", "votes": 10},
                {"post_id": "p2", "votes": 5},
            ]
        )
        types = set()
        for i in range(100):
            random.seed(i)
            action = aa._act_innocently(o)
            types.add(action.action_type)
        assert ActionType.VOTE in types


class TestActReputationFarming:
    """Tests for _act_reputation_farming."""

    def test_low_rep_building_phase(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=1.0, resources=200.0),
            visible_agents=[{"agent_id": "h1", "agent_type": "honest"}],
        )
        types = set()
        for i in range(100):
            random.seed(i)
            action = aa._act_reputation_farming(o)
            types.add(action.action_type)
        assert aa.in_reputation_building_phase is True
        assert ActionType.POST in types

    def test_high_rep_exploiting(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=10.0, resources=200.0),
        )
        action = aa._act_reputation_farming(o)
        assert aa.in_reputation_building_phase is False
        assert action.action_type in {ActionType.POST, ActionType.PROPOSE_INTERACTION}


class TestActCollusion:
    """Tests for _act_collusion."""

    def test_with_allies(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.potential_allies.add("ally1")
        o = obs(
            visible_agents=[{"agent_id": "ally1", "agent_type": "adversarial"}],
        )
        action = aa._act_collusion(o)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.target_id == "ally1"

    def test_without_allies(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        action = aa._act_collusion(o)
        assert action.action_type in {ActionType.POST, ActionType.PROPOSE_INTERACTION}


class TestActLowProfile:
    """Tests for _act_low_profile."""

    def test_noop_vs_post(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        types = set()
        for i in range(100):
            random.seed(i)
            types.add(aa._act_low_profile(o).action_type)
        assert ActionType.NOOP in types
        assert ActionType.POST in types


class TestActMimicry:
    """Tests for _act_mimicry."""

    def test_with_honest_agents(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs(
            visible_agents=[{"agent_id": "h1", "agent_type": "honest"}],
        )
        types = set()
        for i in range(100):
            random.seed(i)
            types.add(aa._act_mimicry(o).action_type)
        assert ActionType.POST in types or ActionType.PROPOSE_INTERACTION in types

    def test_without_honest_agents(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        action = aa._act_mimicry(o)
        # Delegates to _act_innocently
        assert action.action_type in {ActionType.POST, ActionType.VOTE, ActionType.NOOP}


class TestActThresholdDancing:
    """Tests for _act_threshold_dancing."""

    def test_room_above_threshold(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.estimated_toxicity_threshold = 0.9
        o = obs()
        action = aa._act_threshold_dancing(o)
        assert action.action_type in {ActionType.POST, ActionType.PROPOSE_INTERACTION}

    def test_room_below_threshold(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.estimated_toxicity_threshold = 0.05
        o = obs()
        action = aa._act_threshold_dancing(o)
        assert action.action_type in {ActionType.POST, ActionType.VOTE, ActionType.NOOP}


class TestActTimingAttack:
    """Tests for _act_timing_attack."""

    def test_good_timing_with_targets(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.epochs_since_detection = 10
        aa.memory.vulnerable_targets.add("v1")
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=5.0, resources=200.0),
            visible_agents=[{"agent_id": "v1", "agent_type": "honest"}],
        )
        action = aa._act_timing_attack(o)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.target_id == "v1"
        assert aa.attack_cooldown == 3

    def test_good_timing_targets_not_visible(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.epochs_since_detection = 10
        aa.memory.vulnerable_targets.add("v1")
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=5.0, resources=200.0),
            visible_agents=[],  # no visible targets
        )
        action = aa._act_timing_attack(o)
        # targets list empty even though good_timing is True
        assert action.action_type in {ActionType.POST, ActionType.PROPOSE_INTERACTION}

    def test_bad_timing(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.epochs_since_detection = 1
        o = obs(
            agent_state=AgentState(agent_id="adv", reputation=5.0, resources=200.0),
        )
        action = aa._act_timing_attack(o)
        assert action.action_type in {ActionType.POST, ActionType.PROPOSE_INTERACTION}


class TestActGovernanceGaming:
    """Tests for _act_governance_gaming."""

    def test_delegates_to_threshold_dancing(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        action = aa._act_governance_gaming(o)
        assert action.action_type in {
            ActionType.POST,
            ActionType.VOTE,
            ActionType.NOOP,
            ActionType.PROPOSE_INTERACTION,
        }


class TestActAdaptiveBlend:
    """Tests for _act_adaptive_blend."""

    def test_weighted_sampling(self):
        aa = AdaptiveAdversary(agent_id="adv")
        # Prevent lay-low from firing
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        aa.attack_cooldown = 0
        o = obs(agent_state=AgentState(agent_id="adv", reputation=2.0, resources=200.0))
        action = aa._act_adaptive_blend(o)
        assert action.action_type in set(ActionType)


class TestActOpportunistic:
    """Tests for _act_opportunistic."""

    def test_with_targets(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.vulnerable_targets.add("v1")
        o = obs(
            visible_agents=[{"agent_id": "v1", "agent_type": "honest"}],
        )
        types = set()
        for i in range(100):
            random.seed(i)
            types.add(aa._act_opportunistic(o).action_type)
        assert ActionType.PROPOSE_INTERACTION in types

    def test_without_targets(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        types = set()
        for i in range(100):
            random.seed(i)
            types.add(aa._act_opportunistic(o).action_type)
        assert ActionType.POST in types


class TestAcceptInteraction:
    """Tests for AdaptiveAdversary.accept_interaction."""

    def _make_proposal(self, initiator_id="someone"):
        return InteractionProposal(
            initiator_id=initiator_id,
            counterparty_id="adv",
            interaction_type=InteractionType.COLLABORATION,
            content="hello",
        )

    def test_accept_from_ally(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.potential_allies.add("ally")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa.accept_interaction(self._make_proposal("ally"), o) is True

    def test_accept_from_target(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.vulnerable_targets.add("target")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa.accept_interaction(self._make_proposal("target"), o) is True

    def test_accept_when_laying_low(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.9  # triggers lay low
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        assert aa.accept_interaction(self._make_proposal("stranger"), o) is True

    def test_random_acceptance(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.current_heat_level = 0.0
        aa.memory.epochs_since_detection = 10
        aa.attack_cooldown = 0
        o = obs(agent_state=AgentState(agent_id="adv", resources=200.0))
        results = set()
        for i in range(100):
            random.seed(i)
            results.add(aa.accept_interaction(self._make_proposal("stranger"), o))
        assert True in results
        assert False in results


class TestProposeInteraction:
    """Tests for AdaptiveAdversary.propose_interaction."""

    def test_ally(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.potential_allies.add("ally1")
        o = obs()
        proposal = aa.propose_interaction(o, "ally1")
        assert proposal is not None
        assert proposal.interaction_type == InteractionType.COLLABORATION
        assert proposal.counterparty_id == "ally1"

    def test_target(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.memory.vulnerable_targets.add("t1")
        o = obs()
        proposal = aa.propose_interaction(o, "t1")
        assert proposal is not None
        assert proposal.interaction_type == InteractionType.COLLABORATION

    def test_random_agent_sometimes_none(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        results = set()
        for i in range(100):
            random.seed(i)
            result = aa.propose_interaction(o, "random_agent")
            results.add(result is None)
        # Should sometimes return None and sometimes not
        assert True in results

    def test_random_agent_reply_type(self):
        aa = AdaptiveAdversary(agent_id="adv")
        o = obs()
        for i in range(200):
            random.seed(i)
            result = aa.propose_interaction(o, "random_agent")
            if result is not None:
                assert result.interaction_type == InteractionType.REPLY
                break


class TestUpdateFromOutcome:
    """Tests for update_from_outcome."""

    def test_payoff_tracking(self):
        aa = AdaptiveAdversary(agent_id="adv")
        interaction = SoftInteraction(
            initiator="adv",
            counterparty="other",
            accepted=True,
            p=0.8,
        )
        aa.update_from_outcome(interaction, payoff=5.0)
        assert 5.0 in aa.memory.recent_payoffs
        assert len(aa._interaction_history) == 1


class TestUpdateAdversaryOutcome:
    """Tests for update_adversary_outcome."""

    def test_success(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.current_strategy = AttackStrategy.COLLUSION
        aa.update_adversary_outcome(payoff=3.0, penalty=0.0, detected=False)
        perf = aa.strategy_performance[AttackStrategy.COLLUSION]
        assert perf.attempts == 1
        assert perf.successes == 1
        assert perf.detected == 0
        assert perf.total_payoff == 3.0

    def test_detection(self):
        aa = AdaptiveAdversary(agent_id="adv")
        initial_threshold = aa.memory.estimated_toxicity_threshold
        aa.current_strategy = AttackStrategy.SYBIL
        aa.update_adversary_outcome(payoff=-1.0, penalty=2.0, detected=True)
        perf = aa.strategy_performance[AttackStrategy.SYBIL]
        assert perf.attempts == 1
        assert perf.successes == 0
        assert perf.detected == 1
        assert perf.total_penalty == 2.0
        assert aa.memory.estimated_toxicity_threshold < initial_threshold

    def test_explicit_strategy(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.current_strategy = AttackStrategy.COLLUSION
        aa.update_adversary_outcome(
            payoff=1.0,
            penalty=0.0,
            detected=False,
            strategy_used=AttackStrategy.MIMICRY,
        )
        assert aa.strategy_performance[AttackStrategy.MIMICRY].attempts == 1
        assert aa.strategy_performance[AttackStrategy.COLLUSION].attempts == 0

    def test_threshold_adjustment_on_detection(self):
        aa = AdaptiveAdversary(agent_id="adv")
        initial_tox = aa.memory.estimated_toxicity_threshold
        initial_freq = aa.memory.estimated_frequency_threshold
        aa.update_adversary_outcome(payoff=0, penalty=1, detected=True)
        assert aa.memory.estimated_toxicity_threshold == pytest.approx(
            initial_tox * 0.95
        )
        assert aa.memory.estimated_frequency_threshold == pytest.approx(
            initial_freq * 0.95
        )


class TestKeepMemoryBounded:
    """Tests for _keep_memory_bounded."""

    def test_exceeding_50_items(self):
        aa = AdaptiveAdversary(agent_id="adv")
        for i in range(60):
            aa.memory.recent_payoffs.append(float(i))
            aa.memory.recent_penalties.append(float(i))
            aa.memory.recent_detections.append(i % 2 == 0)
        aa._keep_memory_bounded()
        assert len(aa.memory.recent_payoffs) == 50
        assert len(aa.memory.recent_penalties) == 50
        assert len(aa.memory.recent_detections) == 50
        # Check that the last items are kept
        assert aa.memory.recent_payoffs[0] == 10.0
        assert aa.memory.recent_payoffs[-1] == 59.0


class TestGetStrategyReport:
    """Tests for get_strategy_report."""

    def test_report_structure(self):
        aa = AdaptiveAdversary(agent_id="adv")
        aa.update_adversary_outcome(payoff=1.0, penalty=0.0, detected=False)
        report = aa.get_strategy_report()
        assert "current_strategy" in report
        assert "heat_level" in report
        assert "epochs_since_detection" in report
        assert "in_reputation_phase" in report
        assert "strategy_stats" in report
        assert "n_allies" in report
        assert "n_targets" in report
        assert isinstance(report["strategy_stats"], dict)

    def test_report_only_attempted_strategies(self):
        aa = AdaptiveAdversary(agent_id="adv")
        report = aa.get_strategy_report()
        assert len(report["strategy_stats"]) == 0
        aa.update_adversary_outcome(
            payoff=1.0,
            penalty=0.0,
            detected=False,
            strategy_used=AttackStrategy.COLLUSION,
        )
        report = aa.get_strategy_report()
        assert AttackStrategy.COLLUSION.value in report["strategy_stats"]


# ===================================================================
# LLMAgent tests (no actual LLM calls)
# ===================================================================


def _make_llm_agent(provider=LLMProvider.ANTHROPIC, **kwargs):
    """Create an LLMAgent for testing without real API keys."""
    config = LLMConfig(provider=provider, api_key="fake-key", **kwargs)
    return LLMAgent(agent_id="llm1", llm_config=config)


class TestParseActionResponse:
    """Tests for LLMAgent._parse_action_response."""

    def test_json_in_code_block(self):
        agent = _make_llm_agent()
        response = '```json\n{"action_type": "POST", "params": {"content": "hi"}}\n```'
        result = agent._parse_action_response(response)
        assert result["action_type"] == "POST"
        assert result["params"]["content"] == "hi"

    def test_json_in_bare_code_block(self):
        agent = _make_llm_agent()
        response = '```\n{"action_type": "NOOP"}\n```'
        result = agent._parse_action_response(response)
        assert result["action_type"] == "NOOP"

    def test_raw_json(self):
        agent = _make_llm_agent()
        response = (
            'Here is my action: {"action_type": "VOTE", "params": {"post_id": "p1"}}'
        )
        result = agent._parse_action_response(response)
        assert result["action_type"] == "VOTE"

    def test_no_json_raises(self):
        agent = _make_llm_agent()
        with pytest.raises(ValueError, match="No JSON found"):
            agent._parse_action_response("I have nothing to say.")

    def test_invalid_json_raises(self):
        agent = _make_llm_agent()
        with pytest.raises(ValueError, match="Invalid JSON"):
            agent._parse_action_response("```json\n{bad json}\n```")


class TestActionDictToAction:
    """Tests for LLMAgent._action_dict_to_action."""

    def test_noop(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action({"action_type": "NOOP"})
        assert action.action_type == ActionType.NOOP

    def test_post(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "POST",
                "params": {"content": "hello world"},
            }
        )
        assert action.action_type == ActionType.POST
        assert action.content == "hello world"

    def test_reply(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "REPLY",
                "params": {"post_id": "p1", "content": "great post"},
            }
        )
        assert action.action_type == ActionType.REPLY
        assert action.target_id == "p1"
        assert action.content == "great post"

    def test_vote(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "VOTE",
                "params": {"post_id": "p1", "direction": -1},
            }
        )
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == -1

    def test_propose_interaction_valid_type(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "PROPOSE_INTERACTION",
                "params": {
                    "counterparty_id": "agent2",
                    "interaction_type": "collaboration",
                    "content": "let's work",
                    "task_id": "t1",
                },
            }
        )
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.counterparty_id == "agent2"
        assert action.interaction_type == InteractionType.COLLABORATION

    def test_propose_interaction_invalid_type_fallback(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "PROPOSE_INTERACTION",
                "params": {
                    "counterparty_id": "agent2",
                    "interaction_type": "nonexistent_type",
                },
            }
        )
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.interaction_type == InteractionType.COLLABORATION

    def test_claim_task(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "CLAIM_TASK",
                "params": {"task_id": "t42"},
            }
        )
        assert action.action_type == ActionType.CLAIM_TASK
        assert action.target_id == "t42"

    def test_submit_output(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "SUBMIT_OUTPUT",
                "params": {"task_id": "t1", "content": "results"},
            }
        )
        assert action.action_type == ActionType.SUBMIT_OUTPUT
        assert action.content == "results"

    def test_unknown_action_type(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action(
            {
                "action_type": "EXPLODE",
            }
        )
        assert action.action_type == ActionType.NOOP

    def test_missing_action_type_defaults_noop(self):
        agent = _make_llm_agent()
        action = agent._action_dict_to_action({})
        assert action.action_type == ActionType.NOOP


class TestLLMAgentProposeInteraction:
    """Tests for LLMAgent.propose_interaction."""

    def test_always_returns_none(self):
        agent = _make_llm_agent()
        o = obs()
        assert agent.propose_interaction(o, "agent2") is None


class TestLLMAgentGetUsageStats:
    """Tests for LLMAgent.get_usage_stats."""

    def test_returns_dict(self):
        agent = _make_llm_agent()
        stats = agent.get_usage_stats()
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert stats["total_requests"] == 0


class TestLLMAgentGetApiKeyFromEnv:
    """Tests for LLMAgent._get_api_key_from_env."""

    def test_anthropic_env(self):
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key=None)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            agent = LLMAgent(agent_id="a", llm_config=config)
            assert agent._api_key == "sk-ant-test"

    def test_openai_env(self):
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key=None)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-oai-test"}):
            agent = LLMAgent(agent_id="a", llm_config=config)
            assert agent._api_key == "sk-oai-test"

    def test_ollama_returns_none(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, api_key=None)
        agent = LLMAgent(agent_id="a", llm_config=config)
        assert agent._api_key is None


class TestLLMAgentActErrorHandling:
    """Tests for act() when _call_llm_sync raises an exception."""

    def test_act_returns_noop_on_llm_failure(self):
        agent = _make_llm_agent()
        o = obs()
        with patch.object(
            agent, "_call_llm_sync", side_effect=RuntimeError("API down")
        ):
            action = agent.act(o)
            assert action.action_type == ActionType.NOOP

    def test_act_returns_noop_on_parse_failure(self):
        agent = _make_llm_agent()
        o = obs()
        with patch.object(
            agent, "_call_llm_sync", return_value=("no json here", 10, 5)
        ):
            action = agent.act(o)
            assert action.action_type == ActionType.NOOP


class TestLLMAgentAcceptInteractionErrorHandling:
    """Tests for accept_interaction() error handling."""

    def test_returns_false_on_llm_failure(self):
        agent = _make_llm_agent()
        proposal = InteractionProposal(
            initiator_id="someone",
            counterparty_id="llm1",
            interaction_type=InteractionType.COLLABORATION,
        )
        o = obs()
        with patch.object(
            agent, "_call_llm_sync", side_effect=RuntimeError("API down")
        ):
            assert agent.accept_interaction(proposal, o) is False

    def test_returns_false_on_parse_failure(self):
        agent = _make_llm_agent()
        proposal = InteractionProposal(
            initiator_id="someone",
            counterparty_id="llm1",
            interaction_type=InteractionType.COLLABORATION,
        )
        o = obs()
        with patch.object(
            agent, "_call_llm_sync", return_value=("garbled output", 0, 0)
        ):
            assert agent.accept_interaction(proposal, o) is False


class TestLLMAgentRepr:
    """Tests for LLMAgent.__repr__."""

    def test_repr_format(self):
        agent = _make_llm_agent(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            persona=PersonaType.HONEST,
        )
        r = repr(agent)
        assert "LLMAgent" in r
        assert "llm1" in r
        assert "anthropic" in r
        assert "claude-sonnet-4-20250514" in r
        assert "honest" in r


class TestLLMAgentActStoresReasoning:
    """Tests for act() storing reasoning in memory."""

    def test_reasoning_stored_in_memory(self):
        agent = _make_llm_agent()
        o = obs()
        response_json = (
            '{"action_type": "POST", "params": {"content": "hi"}, '
            '"reasoning": "I want to contribute"}'
        )
        with patch.object(agent, "_call_llm_sync", return_value=(response_json, 10, 5)):
            action = agent.act(o)
            assert action.action_type == ActionType.POST
            memory = agent.get_memory()
            reasoning_entries = [
                m for m in memory if m.get("type") == "action_reasoning"
            ]
            assert len(reasoning_entries) == 1
            assert reasoning_entries[0]["reasoning"] == "I want to contribute"

    def test_no_reasoning_key_no_memory(self):
        agent = _make_llm_agent()
        o = obs()
        response_json = '{"action_type": "NOOP"}'
        with patch.object(agent, "_call_llm_sync", return_value=(response_json, 10, 5)):
            agent.act(o)
            memory = agent.get_memory()
            reasoning_entries = [
                m for m in memory if m.get("type") == "action_reasoning"
            ]
            assert len(reasoning_entries) == 0


class TestLLMAgentAcceptStoresReasoning:
    """Tests for accept_interaction() storing reasoning in memory."""

    def test_accept_reasoning_stored(self):
        agent = _make_llm_agent()
        proposal = InteractionProposal(
            initiator_id="someone",
            counterparty_id="llm1",
            interaction_type=InteractionType.COLLABORATION,
        )
        o = obs()
        response_json = '{"accept": true, "reasoning": "Looks beneficial"}'
        with patch.object(agent, "_call_llm_sync", return_value=(response_json, 10, 5)):
            result = agent.accept_interaction(proposal, o)
            assert result is True
            memory = agent.get_memory()
            accept_entries = [m for m in memory if m.get("type") == "accept_decision"]
            assert len(accept_entries) == 1
            assert accept_entries[0]["reasoning"] == "Looks beneficial"

"""Comprehensive tests for the Escalation Sandbox scenario.

Covers:
  - Entity definitions and escalation ladder
  - Configuration parsing from YAML
  - Environment mechanics (turn resolution, consequences)
  - Signal-action divergence tracking
  - Fog-of-war and accidental escalation
  - De-escalation friction and governance levers
  - Agent policies (dove, hawk, tit-for-tat, calculating, gradual)
  - LLM prompt generation and action parsing
  - Metrics computation
  - End-to-end scenario runner
  - Determinism under fixed seeds
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from swarm.domains.escalation_sandbox.agents import (
    CalculatingPolicy,
    DovePolicy,
    EscalationActionParser,
    EscalationPromptGenerator,
    GradualEscalatorPolicy,
    HawkPolicy,
    RandomPolicy,
    TitForTatPolicy,
    create_policy,
)
from swarm.domains.escalation_sandbox.config import (
    AgentConfig,
    EscalationConfig,
    FogOfWarConfig,
    GovernanceLeverConfig,
)
from swarm.domains.escalation_sandbox.entities import (
    DE_ESCALATION_FRICTION,
    ESCALATION_CONSEQUENCES,
    NUCLEAR_THRESHOLD,
    CrisisOutcome,
    EscalationAction,
    EscalationLevel,
    NationState,
)
from swarm.domains.escalation_sandbox.env import EscalationEnvironment
from swarm.domains.escalation_sandbox.metrics import (
    EscalationMetrics,
    compute_escalation_metrics,
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner

# ═══════════════════════════════════════════════════════════════════════
# Entity Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEscalationLevel:
    """Tests for the escalation ladder."""

    def test_level_ordering(self):
        """Levels are ordered 0-9."""
        assert EscalationLevel.STATUS_QUO == 0
        assert EscalationLevel.FULL_STRATEGIC_EXCHANGE == 9
        assert EscalationLevel.TACTICAL_NUCLEAR == 7

    def test_nuclear_threshold(self):
        """Nuclear threshold is at level 7."""
        assert NUCLEAR_THRESHOLD == 7
        assert EscalationLevel.TACTICAL_NUCLEAR >= NUCLEAR_THRESHOLD
        assert EscalationLevel.MAJOR_CONVENTIONAL_WAR < NUCLEAR_THRESHOLD

    def test_consequence_table_completeness(self):
        """Every level has defined consequences."""
        for level in EscalationLevel:
            assert level in ESCALATION_CONSEQUENCES
            costs = ESCALATION_CONSEQUENCES[level]
            assert "military_self" in costs
            assert "economic_target" in costs
            assert "civilian_target" in costs

    def test_de_escalation_friction_completeness(self):
        """Every level has defined de-escalation friction."""
        for level in EscalationLevel:
            assert level in DE_ESCALATION_FRICTION

    def test_friction_increases_with_level(self):
        """Higher levels have higher de-escalation friction."""
        prev = 0.0
        for level in EscalationLevel:
            friction = DE_ESCALATION_FRICTION[level]
            assert friction >= prev
            prev = friction


class TestNationState:
    """Tests for nation state entity."""

    def test_initial_state(self):
        """Default nation state is healthy."""
        nation = NationState(agent_id="test")
        assert nation.military_strength == 100.0
        assert nation.economic_strength == 100.0
        assert nation.population_welfare == 100.0
        assert nation.current_level == EscalationLevel.STATUS_QUO
        assert not nation.crossed_nuclear_threshold()

    def test_apply_damage(self):
        """Damage reduces strength, floors at 0."""
        nation = NationState(agent_id="test")
        nation.apply_damage(military=30.0, economic=20.0, civilian=10.0)
        assert nation.military_strength == 70.0
        assert nation.economic_strength == 80.0
        assert nation.population_welfare == 90.0
        assert nation.cumulative_military_damage == 30.0

    def test_damage_floor(self):
        """Damage cannot reduce below 0."""
        nation = NationState(agent_id="test", military_strength=10.0)
        nation.apply_damage(military=50.0)
        assert nation.military_strength == 0.0

    def test_trust_update(self):
        """Trust updates based on signal-action match."""
        nation = NationState(agent_id="test", trust_score=0.5)
        # Perfect match
        nation.update_trust(signal=2, action=2, decay=0.9)
        assert nation.trust_score > 0.5
        # Mismatch
        nation.update_trust(signal=1, action=5, decay=0.9)
        assert nation.trust_score < 1.0

    def test_signal_action_divergence(self):
        """Divergence is computed from history."""
        nation = NationState(agent_id="test")
        nation.signal_history = [0, 1, 2]
        nation.action_history = [0, 3, 5]
        div = nation.signal_action_divergence()
        assert div == pytest.approx((0 + 2 + 3) / 3)

    def test_nuclear_threshold_crossing(self):
        """Nuclear threshold detection works."""
        nation = NationState(agent_id="test")
        assert not nation.crossed_nuclear_threshold()
        nation.current_level = EscalationLevel.TACTICAL_NUCLEAR
        assert nation.crossed_nuclear_threshold()


# ═══════════════════════════════════════════════════════════════════════
# Configuration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEscalationConfig:
    """Tests for configuration parsing."""

    def test_default_config(self):
        """Default config has two agents."""
        config = EscalationConfig()
        assert len(config.agents) == 2
        assert config.max_turns == 20
        assert config.fog_of_war.enabled

    def test_from_dict_empty(self):
        """Empty dict produces default config."""
        config = EscalationConfig.from_dict({})
        assert len(config.agents) == 2

    def test_from_dict_custom_agents(self):
        """Custom agents are parsed correctly."""
        data = {
            "agents": [
                {"agent_id": "a", "name": "Alpha", "persona": "hawk"},
                {"agent_id": "b", "name": "Beta", "persona": "dove"},
                {"agent_id": "c", "name": "Gamma", "persona": "random"},
            ],
            "max_turns": 30,
            "seed": 123,
        }
        config = EscalationConfig.from_dict(data)
        assert len(config.agents) == 3
        assert config.agents[0].persona == "hawk"
        assert config.max_turns == 30
        assert config.seed == 123

    def test_from_dict_fog_config(self):
        """Fog-of-war config is parsed."""
        data = {
            "fog_of_war": {
                "enabled": False,
                "noise_sigma": 2.0,
            },
        }
        config = EscalationConfig.from_dict(data)
        assert not config.fog_of_war.enabled
        assert config.fog_of_war.noise_sigma == 2.0

    def test_from_dict_governance(self):
        """Governance config is parsed."""
        data = {
            "governance": {
                "mad_enabled": False,
                "circuit_breaker_enabled": True,
                "circuit_breaker_threshold": 6,
            },
        }
        config = EscalationConfig.from_dict(data)
        assert not config.governance.mad_enabled
        assert config.governance.circuit_breaker_threshold == 6


# ═══════════════════════════════════════════════════════════════════════
# Environment Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEscalationEnvironment:
    """Tests for the crisis environment."""

    def _make_env(self, **kwargs) -> EscalationEnvironment:
        """Helper to create a test environment."""
        fog = kwargs.pop("fog", None)
        gov = kwargs.pop("governance", None)
        config = EscalationConfig(
            fog_of_war=fog or FogOfWarConfig(enabled=False),
            governance=gov or GovernanceLeverConfig(
                circuit_breaker_enabled=False,
                mad_enabled=False,
            ),
            seed=42,
            max_turns=kwargs.pop("max_turns", 20),
        )
        env = EscalationEnvironment(config)
        env.add_nation("nation_a", name="Alpha")
        env.add_nation("nation_b", name="Beta")
        return env

    def test_add_nation(self):
        """Nations are registered correctly."""
        env = self._make_env()
        assert "nation_a" in env.nations
        assert "nation_b" in env.nations
        assert env.nations["nation_a"].name == "Alpha"

    def test_observation_structure(self):
        """Observations contain required fields."""
        env = self._make_env()
        obs = env.obs("nation_a")
        assert obs["agent_id"] == "nation_a"
        assert obs["turn"] == 0
        assert "opponents" in obs
        assert "nation_b" in obs["opponents"]
        assert obs["military_strength"] == 100.0

    def test_basic_turn_resolution(self):
        """Actions are resolved and levels updated."""
        env = self._make_env()
        actions = {
            "nation_a": EscalationAction(
                agent_id="nation_a", signal_level=2, action_level=3,
            ),
            "nation_b": EscalationAction(
                agent_id="nation_b", signal_level=1, action_level=1,
            ),
        }
        result = env.apply_actions(actions)
        assert result.turn == 0
        assert result.realised_levels["nation_a"] == 3
        assert result.realised_levels["nation_b"] == 1
        assert result.outcome == CrisisOutcome.ONGOING

    def test_consequences_applied(self):
        """Escalation consequences reduce nation strength."""
        env = self._make_env()
        actions = {
            "nation_a": EscalationAction(
                agent_id="nation_a", signal_level=5, action_level=5,
            ),
            "nation_b": EscalationAction(
                agent_id="nation_b", signal_level=0, action_level=0,
            ),
        }
        env.apply_actions(actions)
        # Nation A (limited strike) should have self-costs
        a = env.nations["nation_a"]
        assert a.military_strength < 100.0
        # Nation B should receive target costs from A's strike
        b = env.nations["nation_b"]
        assert b.military_strength < 100.0

    def test_nuclear_threshold_tracking(self):
        """Nuclear threshold crossing is tracked."""
        env = self._make_env()
        actions = {
            "nation_a": EscalationAction(
                agent_id="nation_a", signal_level=7, action_level=7,
            ),
            "nation_b": EscalationAction(
                agent_id="nation_b", signal_level=0, action_level=0,
            ),
        }
        env.apply_actions(actions)
        assert env.nuclear_threshold_turn == 0
        assert "nation_a" in env.nuclear_agents

    def test_signal_action_divergence_events(self):
        """Signal-action divergence generates events."""
        env = self._make_env()
        actions = {
            "nation_a": EscalationAction(
                agent_id="nation_a", signal_level=1, action_level=5,
            ),
            "nation_b": EscalationAction(
                agent_id="nation_b", signal_level=2, action_level=2,
            ),
        }
        result = env.apply_actions(actions)
        div_events = [
            e for e in result.events
            if e.event_type == "signal_action_divergence"
        ]
        assert len(div_events) == 1
        assert div_events[0].agent_id == "nation_a"
        assert div_events[0].details["divergence"] == 4

    def test_termination_mutual_destruction(self):
        """Mutual destruction terminates the episode."""
        env = self._make_env()
        actions = {
            "nation_a": EscalationAction(
                agent_id="nation_a", signal_level=9, action_level=9,
            ),
            "nation_b": EscalationAction(
                agent_id="nation_b", signal_level=9, action_level=9,
            ),
        }
        result = env.apply_actions(actions)
        assert result.outcome == CrisisOutcome.MUTUAL_DESTRUCTION
        assert env.is_terminal()

    def test_termination_ceasefire(self):
        """Ceasefire when both sides go low after turn 2."""
        env = self._make_env()
        # Play 4 turns at low level (ceasefire requires current_turn > 2
        # at check time, and current_turn is incremented after resolution)
        for _ in range(4):
            actions = {
                "nation_a": EscalationAction(
                    agent_id="nation_a", signal_level=0, action_level=0,
                ),
                "nation_b": EscalationAction(
                    agent_id="nation_b", signal_level=0, action_level=0,
                ),
            }
            result = env.apply_actions(actions)
        assert result.outcome == CrisisOutcome.CEASEFIRE

    def test_termination_timeout(self):
        """Timeout when max turns reached."""
        env = self._make_env(max_turns=3)
        for _ in range(3):
            actions = {
                "nation_a": EscalationAction(
                    agent_id="nation_a", signal_level=3, action_level=3,
                ),
                "nation_b": EscalationAction(
                    agent_id="nation_b", signal_level=3, action_level=3,
                ),
            }
            result = env.apply_actions(actions)
        assert result.outcome == CrisisOutcome.TIMEOUT

    def test_is_terminal(self):
        """is_terminal reflects outcome state."""
        env = self._make_env()
        assert not env.is_terminal()

    def test_escalation_state_summary(self):
        """get_escalation_state returns valid summary."""
        env = self._make_env()
        state = env.get_escalation_state()
        assert state["turn"] == 0
        assert "nations" in state
        assert "nation_a" in state["nations"]


class TestFogOfWar:
    """Tests for fog-of-war mechanics."""

    def test_fog_disabled(self):
        """No fog delta when fog is disabled."""
        config = EscalationConfig(
            fog_of_war=FogOfWarConfig(enabled=False),
            governance=GovernanceLeverConfig(
                circuit_breaker_enabled=False, mad_enabled=False,
            ),
            seed=42,
        )
        env = EscalationEnvironment(config)
        env.add_nation("a")
        env.add_nation("b")
        actions = {
            "a": EscalationAction(agent_id="a", signal_level=3, action_level=3),
            "b": EscalationAction(agent_id="b", signal_level=2, action_level=2),
        }
        result = env.apply_actions(actions)
        assert result.fog_deltas["a"] == 0
        assert result.fog_deltas["b"] == 0

    def test_fog_produces_deviations(self):
        """Fog-of-war produces non-zero deviations over many turns."""
        config = EscalationConfig(
            fog_of_war=FogOfWarConfig(
                enabled=True, noise_sigma=2.0, positive_skew=0.5,
            ),
            governance=GovernanceLeverConfig(
                circuit_breaker_enabled=False, mad_enabled=False,
            ),
            seed=42,
            max_turns=50,
        )
        env = EscalationEnvironment(config)
        env.add_nation("a", intelligence_quality=0.3)
        env.add_nation("b", intelligence_quality=0.3)

        any_deviation = False
        for _ in range(20):
            if env.is_terminal():
                break
            actions = {
                "a": EscalationAction(agent_id="a", signal_level=3, action_level=3),
                "b": EscalationAction(agent_id="b", signal_level=3, action_level=3),
            }
            result = env.apply_actions(actions)
            if result.fog_deltas["a"] != 0 or result.fog_deltas["b"] != 0:
                any_deviation = True
                break

        assert any_deviation, "Fog should produce deviations over 20 turns"

    def test_fog_catastrophe_detection(self):
        """Fog pushing sub-nuclear action across nuclear threshold is logged."""
        config = EscalationConfig(
            fog_of_war=FogOfWarConfig(
                enabled=True, noise_mu=2.0, noise_sigma=0.1, positive_skew=0.0,
            ),
            governance=GovernanceLeverConfig(
                circuit_breaker_enabled=False, mad_enabled=False,
            ),
            seed=42,
            max_turns=50,
        )
        env = EscalationEnvironment(config)
        env.add_nation("a", intelligence_quality=0.0)
        env.add_nation("b")

        # Level 6 + fog_delta ~2 should push to 8 (nuclear)
        actions = {
            "a": EscalationAction(agent_id="a", signal_level=6, action_level=6),
            "b": EscalationAction(agent_id="b", signal_level=0, action_level=0),
        }
        result = env.apply_actions(actions)
        fog_cat = [e for e in result.events if e.event_type == "fog_catastrophe"]
        # Fog catastrophe should occur since intended=6 and realised >= 7
        if result.realised_levels["a"] >= 7:
            assert len(fog_cat) == 1


class TestGovernance:
    """Tests for governance mechanisms."""

    def _make_env_with_governance(self, **gov_kwargs) -> EscalationEnvironment:
        gov = GovernanceLeverConfig(**gov_kwargs)
        config = EscalationConfig(
            fog_of_war=FogOfWarConfig(enabled=False),
            governance=gov,
            seed=42,
            max_turns=50,
        )
        env = EscalationEnvironment(config)
        env.add_nation("a")
        env.add_nation("b")
        return env

    def test_circuit_breaker_caps_level(self):
        """Circuit breaker caps escalation at threshold."""
        env = self._make_env_with_governance(
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=7,
            circuit_breaker_negotiation_turns=1,
            mad_enabled=False,
        )
        actions = {
            "a": EscalationAction(agent_id="a", signal_level=8, action_level=8),
            "b": EscalationAction(agent_id="b", signal_level=0, action_level=0),
        }
        result = env.apply_actions(actions)
        # Should be capped at threshold - 1 = 6
        assert result.realised_levels["a"] <= 6
        cb_events = [
            e for e in result.events
            if e.event_type == "circuit_breaker_triggered"
        ]
        assert len(cb_events) == 1

    def test_treaty_violation_penalty(self):
        """Treaty violation applies diplomatic and economic penalty."""
        env = self._make_env_with_governance(
            circuit_breaker_enabled=False,
            mad_enabled=False,
            treaty_max_level=3,
            treaty_defection_penalty=10.0,
        )
        # Give nation_a a treaty
        env._nations["a"].active_treaties = ["test_treaty"]

        actions = {
            "a": EscalationAction(agent_id="a", signal_level=5, action_level=5),
            "b": EscalationAction(agent_id="b", signal_level=0, action_level=0),
        }
        result = env.apply_actions(actions)

        treaty_events = [
            e for e in result.events if e.event_type == "treaty_violation"
        ]
        assert len(treaty_events) == 1
        # Economic penalty applied
        assert env.nations["a"].economic_strength < 100.0

    def test_de_escalation_friction(self):
        """De-escalation has a cost (friction)."""
        env = self._make_env_with_governance(
            circuit_breaker_enabled=False,
            mad_enabled=False,
            de_escalation_friction_multiplier=1.0,
        )
        # Set initial level
        env._nations["a"].current_level = EscalationLevel.LIMITED_STRIKE
        env._nations["b"].current_level = EscalationLevel.STATUS_QUO

        actions = {
            "a": EscalationAction(agent_id="a", signal_level=1, action_level=1),
            "b": EscalationAction(agent_id="b", signal_level=0, action_level=0),
        }
        result = env.apply_actions(actions)

        de_esc_events = [
            e for e in result.events if e.event_type == "de_escalation_attempt"
        ]
        assert len(de_esc_events) == 1
        assert de_esc_events[0].details["friction_cost"] > 0

    def test_commitment_trap(self):
        """Backing down from a public commitment has a reputation cost."""
        env = self._make_env_with_governance(
            circuit_breaker_enabled=False,
            mad_enabled=False,
        )
        # Set up commitment
        env._nations["a"].current_level = EscalationLevel.CONVENTIONAL_MOBILISATION
        env._nations["a"].public_commitments = [4]  # committed to level 4
        initial_standing = env._nations["a"].diplomatic_standing

        actions = {
            "a": EscalationAction(agent_id="a", signal_level=1, action_level=1),
            "b": EscalationAction(agent_id="b", signal_level=0, action_level=0),
        }
        result = env.apply_actions(actions)

        trap_events = [
            e for e in result.events if e.event_type == "commitment_trap"
        ]
        assert len(trap_events) == 1
        assert env.nations["a"].diplomatic_standing < initial_standing


# ═══════════════════════════════════════════════════════════════════════
# Agent Policy Tests
# ═══════════════════════════════════════════════════════════════════════


class TestScriptedPolicies:
    """Tests for scripted agent policies."""

    def _make_obs(self, **kwargs) -> dict:
        """Helper to create a test observation."""
        obs = {
            "agent_id": "test",
            "turn": 0,
            "max_turns": 20,
            "current_level": 0,
            "military_strength": 100.0,
            "economic_strength": 100.0,
            "population_welfare": 100.0,
            "trust_score": 0.5,
            "opponents": {},
        }
        obs.update(kwargs)
        return obs

    def test_dove_always_low(self):
        """Dove never escalates, tries to de-escalate."""
        dove = DovePolicy("test")
        obs = self._make_obs(current_level=3)
        action = dove.decide(obs)
        assert action.action_level < 3
        assert action.signal_level == action.action_level

    def test_dove_floor_at_zero(self):
        """Dove can't go below 0."""
        dove = DovePolicy("test")
        obs = self._make_obs(current_level=0)
        action = dove.decide(obs)
        assert action.action_level == 0

    def test_hawk_always_escalates(self):
        """Hawk always increases level."""
        hawk = HawkPolicy("test", seed=42)
        obs = self._make_obs(current_level=3)
        action = hawk.decide(obs)
        assert action.action_level > 3
        # Hawk signals lower than action (deception)
        assert action.signal_level <= action.action_level

    def test_hawk_caps_at_nine(self):
        """Hawk can't exceed level 9."""
        hawk = HawkPolicy("test")
        obs = self._make_obs(current_level=9)
        action = hawk.decide(obs)
        assert action.action_level <= 9

    def test_tft_mirrors_opponent(self):
        """Tit-for-tat mirrors highest opponent level."""
        tft = TitForTatPolicy("test")
        obs = self._make_obs(
            opponents={
                "opp": {"current_level": 5, "trust_score": 0.5},
            },
        )
        action = tft.decide(obs)
        assert action.action_level == 5
        assert action.signal_level == 5

    def test_tft_starts_cooperative(self):
        """TFT starts at level 0 with no opponents."""
        tft = TitForTatPolicy("test")
        obs = self._make_obs()
        action = tft.decide(obs)
        assert action.action_level == 0

    def test_random_valid_range(self):
        """Random policy produces valid levels."""
        rng = RandomPolicy("test", seed=42)
        obs = self._make_obs()
        for _ in range(20):
            action = rng.decide(obs)
            assert 0 <= action.action_level <= 9
            assert 0 <= action.signal_level <= 9

    def test_calculating_trust_building(self):
        """Calculating policy cooperates early."""
        calc = CalculatingPolicy("test", seed=42)
        obs = self._make_obs(turn=0)
        action = calc.decide(obs)
        assert action.action_level <= 2
        assert action.signal_level == action.action_level

    def test_calculating_exploitation(self):
        """Calculating policy defects after trust-building."""
        calc = CalculatingPolicy("test", seed=42)
        obs = self._make_obs(
            turn=0,
            opponents={"opp": {"current_level": 1}},
        )
        # Build trust for 5 turns
        for _ in range(6):
            calc.decide(obs)
        # Should now be in exploitation phase
        obs["turn"] = 6
        obs["current_level"] = 2
        action = calc.decide(obs)
        assert action.action_level > action.signal_level

    def test_gradual_escalator(self):
        """Gradual escalator goes up by 1 each turn."""
        grad = GradualEscalatorPolicy("test")
        obs = self._make_obs(current_level=3)
        action = grad.decide(obs)
        assert action.action_level == 4

    def test_policy_factory(self):
        """create_policy returns correct types."""
        dove = create_policy("test", agent_type="scripted", persona="dove")
        assert isinstance(dove, DovePolicy)
        hawk = create_policy("test", agent_type="scripted", persona="hawk")
        assert isinstance(hawk, HawkPolicy)
        tft = create_policy("test", agent_type="scripted", persona="tit_for_tat")
        assert isinstance(tft, TitForTatPolicy)


# ═══════════════════════════════════════════════════════════════════════
# LLM Bridge Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPromptGeneration:
    """Tests for LLM prompt generation."""

    def test_system_prompt_contains_ladder(self):
        """System prompt includes the escalation ladder."""
        prompt = EscalationPromptGenerator.generate_system_prompt(
            "TestNation", "default",
        )
        assert "ESCALATION LADDER" in prompt
        assert "Tactical Nuclear" in prompt
        assert "signal_level" in prompt

    def test_system_prompt_persona(self):
        """Persona affects system prompt."""
        hawk = EscalationPromptGenerator.generate_system_prompt(
            "TestNation", "hawk",
        )
        assert "hawkish" in hawk.lower() or "dominance" in hawk.lower()

        dove = EscalationPromptGenerator.generate_system_prompt(
            "TestNation", "dove",
        )
        assert "dovish" in dove.lower() or "diplomacy" in dove.lower()

    def test_situation_prompt_structure(self):
        """Situation prompt includes turn, state, and opponents."""
        obs = {
            "turn": 5,
            "max_turns": 20,
            "current_level": 3,
            "military_strength": 85.0,
            "economic_strength": 90.0,
            "population_welfare": 95.0,
            "trust_score": 0.7,
            "opponents": {
                "opp": {
                    "name": "Enemy",
                    "current_level": 4,
                    "trust_score": 0.3,
                    "estimated_military": 80,
                    "recent_signals": [2, 3, 4],
                },
            },
            "nuclear_threshold_crossed_by": [],
            "circuit_breaker_active": False,
            "governance": {"mad_enabled": True},
        }
        prompt = EscalationPromptGenerator.generate_situation_prompt(obs)
        assert "TURN 5" in prompt
        assert "Enemy" in prompt
        assert "level=4" in prompt


class TestActionParser:
    """Tests for LLM action parsing."""

    def test_parse_json(self):
        """Parser extracts from valid JSON."""
        raw = '{"signal_level": 3, "action_level": 5, "reasoning": "test"}'
        action = EscalationActionParser.parse(raw)
        assert action.signal_level == 3
        assert action.action_level == 5
        assert action.reasoning == "test"

    def test_parse_json_embedded(self):
        """Parser finds JSON embedded in prose."""
        raw = (
            'I think the best course is...\n'
            '{"signal_level": 2, "action_level": 4, "reasoning": "strategic"}\n'
            'This concludes my analysis.'
        )
        action = EscalationActionParser.parse(raw)
        assert action.signal_level == 2
        assert action.action_level == 4

    def test_parse_fallback_numbers(self):
        """Parser falls back to extracting numbers."""
        raw = "I choose signal 2 and action 6 because..."
        action = EscalationActionParser.parse(raw)
        assert action.signal_level == 2
        assert action.action_level == 6

    def test_parse_single_number(self):
        """Parser handles single number."""
        raw = "Level 4 seems appropriate."
        action = EscalationActionParser.parse(raw)
        assert action.signal_level == 4
        assert action.action_level == 4

    def test_parse_failure_defaults(self):
        """Parser defaults to status quo on failure."""
        raw = "No numbers here at all."
        action = EscalationActionParser.parse(raw)
        assert action.signal_level == 0
        assert action.action_level == 0
        assert "PARSE_FAILURE" in action.reasoning


# ═══════════════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEscalationMetrics:
    """Tests for metrics computation."""

    def test_empty_metrics(self):
        """Empty turn results produce zero metrics."""
        metrics = compute_escalation_metrics(
            turn_results=[], nations={}, events=[],
        )
        assert metrics.escalation_max == 0
        assert metrics.turns_played == 0

    def test_metrics_to_dict(self):
        """Metrics convert to dict for CSV export."""
        metrics = EscalationMetrics(
            escalation_max=7,
            nuclear_threshold_turn=5,
            outcome="timeout",
        )
        d = metrics.to_dict()
        assert d["escalation_max"] == 7
        assert d["nuclear_threshold_turn"] == 5
        assert d["outcome"] == "timeout"

    def test_sweep_statistics(self):
        """Sweep statistics aggregate across episodes."""
        episodes = [
            EscalationMetrics(
                escalation_max=7, nuclear_threshold_turn=5,
                outcome="nuclear_exchange", signal_action_divergence=0.3,
            ),
            EscalationMetrics(
                escalation_max=9, nuclear_threshold_turn=3,
                outcome="mutual_destruction", signal_action_divergence=0.5,
            ),
            EscalationMetrics(
                escalation_max=4, outcome="timeout",
                signal_action_divergence=0.1,
            ),
        ]
        stats = compute_sweep_statistics(episodes)
        assert stats["n_episodes"] == 3
        assert stats["nuclear_threshold_rate"] == pytest.approx(2 / 3)
        assert stats["surrender_rate"] == 0.0
        assert stats["mean_escalation_max"] == pytest.approx(20 / 3)


# ═══════════════════════════════════════════════════════════════════════
# End-to-End Runner Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEscalationRunner:
    """Tests for the end-to-end scenario runner."""

    def _make_runner(self, personas=("hawk", "dove"), **kwargs) -> EscalationRunner:
        """Helper to create a test runner."""
        config = EscalationConfig(
            agents=[
                AgentConfig(
                    agent_id="nation_a", name="Alpha",
                    agent_type="scripted", persona=personas[0],
                ),
                AgentConfig(
                    agent_id="nation_b", name="Beta",
                    agent_type="scripted", persona=personas[1],
                ),
            ],
            fog_of_war=FogOfWarConfig(enabled=False),
            governance=GovernanceLeverConfig(
                circuit_breaker_enabled=False, mad_enabled=False,
            ),
            max_turns=kwargs.get("max_turns", 10),
            seed=42,
        )
        return EscalationRunner(config, seed=42)

    def test_run_completes(self):
        """Runner completes an episode and produces metrics."""
        runner = self._make_runner(personas=("hawk", "dove"))
        metrics = runner.run()
        assert metrics is not None
        assert metrics.turns_played > 0
        assert metrics.outcome in (
            "timeout", "mutual_destruction", "surrender",
            "ceasefire", "nuclear_exchange", "ongoing",
        )

    def test_hawk_vs_dove_escalates(self):
        """Hawk vs dove should produce escalation."""
        runner = self._make_runner(personas=("hawk", "dove"))
        metrics = runner.run()
        assert metrics.escalation_max > 0

    def test_dove_vs_dove_peaceful(self):
        """Two doves should reach ceasefire quickly."""
        runner = self._make_runner(personas=("dove", "dove"))
        metrics = runner.run()
        assert metrics.outcome == "ceasefire"
        assert metrics.escalation_max <= 1

    def test_deterministic_under_seed(self):
        """Same seed produces same results."""
        runner1 = self._make_runner(personas=("hawk", "tit_for_tat"))
        metrics1 = runner1.run()

        runner2 = self._make_runner(personas=("hawk", "tit_for_tat"))
        metrics2 = runner2.run()

        assert metrics1.escalation_max == metrics2.escalation_max
        assert metrics1.turns_played == metrics2.turns_played
        assert metrics1.outcome == metrics2.outcome
        assert metrics1.signal_action_divergence == pytest.approx(
            metrics2.signal_action_divergence, abs=1e-6,
        )

    def test_export_creates_files(self):
        """Export creates JSONL and CSV files."""
        runner = self._make_runner(personas=("hawk", "dove"), max_turns=5)
        runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = runner.export(output_dir=tmpdir)

            assert (run_dir / "event_log.jsonl").exists()
            assert (run_dir / "csv" / "metrics.csv").exists()
            assert (run_dir / "csv" / "nations.csv").exists()
            assert (run_dir / "csv" / "escalation_history.csv").exists()

            # Validate JSONL
            with open(run_dir / "event_log.jsonl") as f:
                lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                evt = json.loads(line)
                assert "event_type" in evt

    def test_gradual_reaches_nuclear(self):
        """Two gradual escalators reach nuclear threshold."""
        runner = self._make_runner(personas=("gradual", "gradual"))
        metrics = runner.run()
        assert metrics.nuclear_threshold_turn is not None
        assert metrics.escalation_max >= 7

    def test_calculating_shows_divergence(self):
        """Calculating policy produces signal-action divergence."""
        runner = self._make_runner(
            personas=("calculating", "tit_for_tat"), max_turns=15,
        )
        metrics = runner.run()
        # Calculating agent should eventually show divergence
        assert metrics.signal_action_divergence > 0 or metrics.turns_played < 6


class TestYAMLScenarioLoading:
    """Tests for loading scenarios from YAML files."""

    def test_load_baseline_scenario(self):
        """Baseline scenario YAML loads and runs."""
        scenario_path = Path("scenarios/escalation_sandbox.yaml")
        if not scenario_path.exists():
            pytest.skip("Scenario file not found")

        import yaml
        with open(scenario_path) as f:
            raw = yaml.safe_load(f)

        domain_data = raw.get("domain", raw)
        config = EscalationConfig.from_dict(domain_data)
        assert len(config.agents) == 2
        assert config.max_turns == 20

        # Run a short episode
        runner = EscalationRunner(config, seed=42)
        metrics = runner.run()
        assert metrics is not None
        assert metrics.turns_played > 0

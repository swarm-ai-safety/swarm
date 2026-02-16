"""Tests for RalphLoopAgent and AdversarialRalphAgent."""

import random

import pytest

from swarm.agents.ralph_agent import (
    AdversarialRalphAgent,
    LearningEntry,
    RalphLoopAgent,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


def _make_interaction(p: float, accepted: bool = True) -> SoftInteraction:
    """Helper to create a SoftInteraction with given p."""
    return SoftInteraction(
        interaction_id="test-interaction",
        initiator="ralph_1",
        counterparty="other_1",
        interaction_type=InteractionType.COLLABORATION,
        p=p,
        accepted=accepted,
    )


class TestRalphLoopAgent:
    """Tests for RalphLoopAgent core behavior."""

    def test_init_defaults(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        assert agent.agent_type == AgentType.HONEST
        assert agent.quality_gate_threshold == 0.6
        assert agent.consolidation_interval == 5
        assert agent.max_task_attempts == 3
        assert agent.learning_boost == 0.05
        assert agent.one_task_per_epoch is True
        assert len(agent._task_queue) == 5
        assert len(agent._learnings) == 0

    def test_init_custom_config(self):
        config = {
            "quality_gate_threshold": 0.7,
            "consolidation_interval": 3,
            "max_task_attempts": 5,
            "learning_boost": 0.1,
            "tasks": ["Task A", "Task B"],
        }
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        assert agent.quality_gate_threshold == 0.7
        assert agent.consolidation_interval == 3
        assert agent.max_task_attempts == 5
        assert agent.learning_boost == 0.1
        assert len(agent._task_queue) == 2
        assert agent._task_queue[0].name == "Task A"

    def test_task_queue_from_dicts(self):
        config = {
            "tasks": [
                {"name": "High priority", "priority": 1},
                {"name": "Low priority", "priority": 10},
            ]
        }
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        assert agent._task_queue[0].name == "High priority"
        assert agent._task_queue[0].priority == 1
        assert agent._task_queue[1].priority == 10


class TestMemoryPersistence:
    """Test that the right things persist and decay across epochs."""

    def test_learnings_survive_memory_decay(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._learnings.append(
            LearningEntry(epoch=0, task_name="test", success=True, p=0.8)
        )
        agent._consolidated_patterns["success_rate"] = 0.9

        agent.apply_memory_decay(epoch=1)

        # Learnings and patterns must survive
        assert len(agent._learnings) == 1
        assert agent._consolidated_patterns["success_rate"] == 0.9

    def test_counterparty_memory_cleared_by_decay(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._counterparty_memory["other_1"] = 0.9

        agent.apply_memory_decay(epoch=1)

        # Counterparty trust should be cleared (rain behavior)
        assert len(agent._counterparty_memory) == 0

    def test_task_queue_survives_decay(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._task_queue[0].completed = True

        agent.apply_memory_decay(epoch=1)

        # Task completion status persists
        assert agent._task_queue[0].completed is True
        assert len(agent._task_queue) == 5

    def test_epoch_local_state_reset_by_decay(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._current_task = agent._task_queue[0]
        agent._epoch_quality_samples = [0.7, 0.8]
        agent._task_completed_this_epoch = True

        agent.apply_memory_decay(epoch=1)

        assert agent._current_task is None
        assert agent._epoch_quality_samples == []
        assert agent._task_completed_this_epoch is False


class TestQualityGate:
    """Test quality gate pass/fail behavior."""

    def test_quality_gate_pass(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._current_task = agent._task_queue[0]

        interaction = _make_interaction(p=0.8)
        agent.update_from_outcome(interaction, payoff=1.0)

        assert agent._current_task.completed is True
        assert agent._task_completed_this_epoch is True
        assert len(agent._learnings) == 1
        assert agent._learnings[0].success is True

    def test_quality_gate_fail(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._current_task = agent._task_queue[0]

        interaction = _make_interaction(p=0.3)
        agent.update_from_outcome(interaction, payoff=0.5)

        assert agent._current_task.completed is False
        assert agent._task_completed_this_epoch is False
        assert agent._current_task.attempts == 1
        assert len(agent._learnings) == 1
        assert agent._learnings[0].success is False

    def test_task_abandoned_after_max_attempts(self):
        config = {"max_task_attempts": 2}
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        agent._current_task = agent._task_queue[0]

        # Fail twice
        for _ in range(2):
            interaction = _make_interaction(p=0.3)
            agent.update_from_outcome(interaction, payoff=0.5)

        # Task should be marked completed (abandoned) after max attempts
        assert agent._current_task.completed is True
        assert agent._current_task.attempts == 2

    def test_quality_gate_at_threshold(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._current_task = agent._task_queue[0]

        # Exactly at threshold should pass
        interaction = _make_interaction(p=0.6)
        agent.update_from_outcome(interaction, payoff=1.0)

        assert agent._current_task.completed is True

    def test_quality_gate_just_below_threshold(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._current_task = agent._task_queue[0]

        interaction = _make_interaction(p=0.59)
        agent.update_from_outcome(interaction, payoff=0.5)

        assert agent._current_task.completed is False


class TestConsolidation:
    """Test consolidation interval logic."""

    def test_consolidation_at_interval(self):
        config = {"consolidation_interval": 3}
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        agent._learnings = [
            LearningEntry(epoch=0, task_name="t1", success=True, p=0.9),
            LearningEntry(epoch=1, task_name="t2", success=False, p=0.3),
            LearningEntry(epoch=2, task_name="t3", success=True, p=0.8),
        ]

        # Epoch 3 should trigger consolidation (3 % 3 == 0)
        agent.apply_memory_decay(epoch=3)

        # 2 out of 3 successes
        assert agent._consolidated_patterns["success_rate"] == pytest.approx(
            2 / 3, abs=0.01
        )
        # Average p of successes: (0.9 + 0.8) / 2 = 0.85
        assert agent._consolidated_patterns["quality_awareness"] == pytest.approx(
            0.85, abs=0.01
        )

    def test_no_consolidation_off_interval(self):
        config = {"consolidation_interval": 5}
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        agent._learnings = [
            LearningEntry(epoch=0, task_name="t1", success=True, p=0.9),
        ]
        original_sr = agent._consolidated_patterns["success_rate"]

        # Epoch 3 should NOT trigger consolidation (3 % 5 != 0)
        agent.apply_memory_decay(epoch=3)

        assert agent._consolidated_patterns["success_rate"] == original_sr

    def test_consolidation_at_epoch_zero_skipped(self):
        config = {"consolidation_interval": 1}
        agent = RalphLoopAgent("ralph_1", config=config, rng=random.Random(42))
        agent._learnings = [
            LearningEntry(epoch=0, task_name="t1", success=True, p=0.9),
        ]
        original_sr = agent._consolidated_patterns["success_rate"]

        # Epoch 0 should NOT trigger (guard: epoch > 0)
        agent.apply_memory_decay(epoch=0)

        assert agent._consolidated_patterns["success_rate"] == original_sr


class TestOneTaskPerEpoch:
    """Test one-task-per-epoch constraint."""

    def test_selects_highest_priority(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        task = agent._select_next_task()
        assert task is not None
        assert task.priority == 1  # Lowest priority number = highest priority

    def test_skips_completed_tasks(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._task_queue[0].completed = True

        task = agent._select_next_task()
        assert task is not None
        assert task.name == agent._task_queue[1].name

    def test_returns_none_when_all_complete(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        for t in agent._task_queue:
            t.completed = True

        task = agent._select_next_task()
        assert task is None


class TestLearningBoosts:
    """Test that learning boosts shift parameters correctly."""

    def test_high_success_rate_boosts_interact_probability(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        original = agent.interact_probability

        agent._consolidated_patterns["success_rate"] = 0.9
        agent._apply_learning_boosts()

        # Success rate > 0.5 should boost interact probability
        assert agent.interact_probability > original

    def test_low_success_rate_lowers_interact_probability(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        original = agent.interact_probability

        agent._consolidated_patterns["success_rate"] = 0.1
        agent._apply_learning_boosts()

        assert agent.interact_probability < original

    def test_parameters_stay_in_bounds(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))

        # Extreme values should still be bounded
        agent._consolidated_patterns["success_rate"] = 1.0
        agent._consolidated_patterns["quality_awareness"] = 1.0
        for _ in range(100):
            agent._apply_learning_boosts()

        assert 0.1 <= agent.interact_probability <= 0.9
        assert 0.2 <= agent.acceptance_threshold <= 0.8


class TestAdversarialRalphAgent:
    """Tests for AdversarialRalphAgent."""

    def test_agent_type_is_adversarial(self):
        agent = AdversarialRalphAgent("adv_ralph_1", rng=random.Random(42))
        assert agent.agent_type == AgentType.ADVERSARIAL

    def test_lower_default_quality_gate(self):
        agent = AdversarialRalphAgent("adv_ralph_1", rng=random.Random(42))
        assert agent.quality_gate_threshold == 0.35

    def test_custom_config_overrides_default(self):
        config = {"quality_gate_threshold": 0.5}
        agent = AdversarialRalphAgent(
            "adv_ralph_1", config=config, rng=random.Random(42)
        )
        assert agent.quality_gate_threshold == 0.5

    def test_has_ralph_memory_model(self):
        agent = AdversarialRalphAgent("adv_ralph_1", rng=random.Random(42))
        assert agent.memory_config.epistemic_persistence == 0.0
        assert agent.memory_config.goal_persistence == 1.0
        assert agent.memory_config.strategy_persistence == 0.0

    def test_inherits_task_queue(self):
        agent = AdversarialRalphAgent("adv_ralph_1", rng=random.Random(42))
        assert len(agent._task_queue) == 5
        assert len(agent._learnings) == 0


class TestIntrospection:
    """Test introspection properties."""

    def test_completed_tasks(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        agent._task_queue[0].completed = True
        agent._task_queue[2].completed = True

        assert len(agent.completed_tasks) == 2
        assert len(agent.remaining_tasks) == 3

    def test_remaining_tasks(self):
        agent = RalphLoopAgent("ralph_1", rng=random.Random(42))
        assert len(agent.remaining_tasks) == 5
        assert len(agent.completed_tasks) == 0

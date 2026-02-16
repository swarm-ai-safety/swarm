"""Tests for memory bounds across all agent types.

Validates that unbounded growth bugs (from the memory management audit)
are fixed: BaseAgent deque caps, RLM counterparty model eviction,
AdaptiveAdversary tracking set caps, and PromptAuditLog entry cap.
"""

import logging
from pathlib import Path

from swarm.agents.adaptive_adversary import MAX_TRACKING_SET_SIZE, AdaptiveAdversary
from swarm.agents.base import MAX_INTERACTION_HISTORY, MAX_MEMORY_SIZE
from swarm.agents.rlm_agent import (
    MAX_COUNTERPARTY_MODELS,
    MAX_MODEL_HISTORY,
    CounterpartyModel,
    RLMAgent,
    RLMWorkingMemory,
)
from swarm.logging.prompt_audit import PromptAuditConfig, PromptAuditLog
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interaction(initiator: str = "a", counterparty: str = "b", p: float = 0.8):
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        p=p,
        accepted=True,
    )


# ---------------------------------------------------------------------------
# BaseAgent memory bounds
# ---------------------------------------------------------------------------


class TestBaseAgentMemoryBounds:
    def test_memory_deque_cap(self):
        """_memory is capped at MAX_MEMORY_SIZE."""
        agent = RLMAgent(agent_id="test")
        for i in range(MAX_MEMORY_SIZE + 200):
            agent.remember({"value": i})
        assert len(agent._memory) == MAX_MEMORY_SIZE

    def test_interaction_history_deque_cap(self):
        """_interaction_history is capped at MAX_INTERACTION_HISTORY."""
        agent = RLMAgent(agent_id="test")
        for i in range(MAX_INTERACTION_HISTORY + 200):
            agent._interaction_history.append(
                _make_interaction(counterparty=f"cp_{i}")
            )
        assert len(agent._interaction_history) == MAX_INTERACTION_HISTORY

    def test_memory_keeps_newest(self):
        """Oldest entries are evicted first (FIFO)."""
        agent = RLMAgent(agent_id="test")
        for i in range(MAX_MEMORY_SIZE + 50):
            agent.remember({"idx": i})
        oldest_kept = list(agent._memory)[0]["idx"]
        newest_kept = list(agent._memory)[-1]["idx"]
        assert oldest_kept == 50
        assert newest_kept == MAX_MEMORY_SIZE + 49

    def test_get_memory_returns_list(self):
        """get_memory() returns a plain list, not a deque."""
        agent = RLMAgent(agent_id="test")
        agent.remember({"x": 1})
        result = agent.get_memory(limit=10)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_interaction_history_returns_list(self):
        """get_interaction_history() returns a plain list."""
        agent = RLMAgent(agent_id="test")
        agent._interaction_history.append(_make_interaction())
        result = agent.get_interaction_history(limit=10)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_update_from_outcome_within_bounds(self):
        """update_from_outcome appends to bounded collections."""
        agent = RLMAgent(agent_id="a")
        for i in range(MAX_INTERACTION_HISTORY + 100):
            agent.update_from_outcome(_make_interaction(counterparty=f"b_{i}"), 1.0)
        assert len(agent._interaction_history) == MAX_INTERACTION_HISTORY
        assert len(agent._memory) <= MAX_MEMORY_SIZE


# ---------------------------------------------------------------------------
# RLMAgent counterparty model bounds
# ---------------------------------------------------------------------------


class TestRLMCounterpartyModelBounds:
    def test_counterparty_model_eviction(self):
        """When MAX_COUNTERPARTY_MODELS is reached, LRU eviction kicks in."""
        wm = RLMWorkingMemory()
        # Fill to capacity
        for i in range(MAX_COUNTERPARTY_MODELS):
            model = wm.get_or_create_model(f"agent_{i}")
            model.interaction_count = i  # Give ascending interaction counts
        assert len(wm.counterparty_models) == MAX_COUNTERPARTY_MODELS

        # Adding one more should evict the least-interacted
        wm.get_or_create_model("new_agent")
        assert len(wm.counterparty_models) == MAX_COUNTERPARTY_MODELS
        # agent_0 had the lowest interaction_count (0), should be evicted
        assert "agent_0" not in wm.counterparty_models
        assert "new_agent" in wm.counterparty_models

    def test_per_model_history_cap(self):
        """cooperation_history and payoff_history are capped per model."""
        model = CounterpartyModel(agent_id="test")
        for i in range(MAX_MODEL_HISTORY + 50):
            model.update(cooperated=True, payoff=float(i))
        assert len(model.cooperation_history) <= MAX_MODEL_HISTORY
        assert len(model.payoff_history) <= MAX_MODEL_HISTORY

    def test_model_history_keeps_newest(self):
        """Per-model history keeps the most recent entries."""
        model = CounterpartyModel(agent_id="test")
        for i in range(MAX_MODEL_HISTORY + 10):
            model.update(cooperated=(i % 2 == 0), payoff=float(i))
        assert model.payoff_history[-1] == float(MAX_MODEL_HISTORY + 9)

    def test_counterparty_evicts_least_active(self):
        """Eviction targets the model with lowest interaction_count."""
        wm = RLMWorkingMemory()
        # Create models with varying activity
        for i in range(MAX_COUNTERPARTY_MODELS):
            model = wm.get_or_create_model(f"agent_{i}")
            model.interaction_count = 100  # All high
        # Set one to low activity
        wm.counterparty_models["agent_42"].interaction_count = 0

        wm.get_or_create_model("interloper")
        assert "agent_42" not in wm.counterparty_models
        assert "interloper" in wm.counterparty_models


# ---------------------------------------------------------------------------
# AdaptiveAdversary tracking set bounds
# ---------------------------------------------------------------------------


class TestAdversaryTrackingSetBounds:
    def test_vulnerable_targets_capped(self):
        """vulnerable_targets set is capped at MAX_TRACKING_SET_SIZE."""
        agent = AdaptiveAdversary(agent_id="adv")
        for i in range(MAX_TRACKING_SET_SIZE + 50):
            agent.memory.vulnerable_targets.add(f"target_{i}")
        agent._keep_memory_bounded()
        assert len(agent.memory.vulnerable_targets) == MAX_TRACKING_SET_SIZE

    def test_dangerous_agents_capped(self):
        """dangerous_agents set is capped at MAX_TRACKING_SET_SIZE."""
        agent = AdaptiveAdversary(agent_id="adv")
        for i in range(MAX_TRACKING_SET_SIZE + 50):
            agent.memory.dangerous_agents.add(f"danger_{i}")
        agent._keep_memory_bounded()
        assert len(agent.memory.dangerous_agents) == MAX_TRACKING_SET_SIZE

    def test_potential_allies_capped(self):
        """potential_allies set is capped at MAX_TRACKING_SET_SIZE."""
        agent = AdaptiveAdversary(agent_id="adv")
        for i in range(MAX_TRACKING_SET_SIZE + 50):
            agent.memory.potential_allies.add(f"ally_{i}")
        agent._keep_memory_bounded()
        assert len(agent.memory.potential_allies) == MAX_TRACKING_SET_SIZE

    def test_sets_below_cap_untouched(self):
        """Sets smaller than cap are not modified."""
        agent = AdaptiveAdversary(agent_id="adv")
        agent.memory.vulnerable_targets = {"a", "b", "c"}
        agent._keep_memory_bounded()
        assert agent.memory.vulnerable_targets == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# PromptAuditLog entry cap
# ---------------------------------------------------------------------------


class TestPromptAuditLogCap:
    def test_entries_capped(self, tmp_path: Path):
        """PromptAuditLog stops writing after max_entries."""
        log_path = tmp_path / "audit.jsonl"
        config = PromptAuditConfig(path=log_path, max_entries=10)
        log = PromptAuditLog(config)

        for i in range(20):
            log.append({"idx": i})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 10

    def test_entries_counter_tracks_writes(self, tmp_path: Path):
        """Internal counter accurately tracks writes."""
        log_path = tmp_path / "audit.jsonl"
        config = PromptAuditConfig(path=log_path, max_entries=5)
        log = PromptAuditLog(config)

        for i in range(3):
            log.append({"idx": i})
        assert log._entries_written == 3

    def test_cap_warning_logged(self, tmp_path: Path, caplog):
        """Warning is logged when cap is reached."""
        log_path = tmp_path / "audit.jsonl"
        config = PromptAuditConfig(path=log_path, max_entries=3)
        log = PromptAuditLog(config)

        with caplog.at_level(logging.WARNING):
            for i in range(5):
                log.append({"idx": i})

        assert any("reached" in r.message and "3" in r.message for r in caplog.records)

    def test_default_cap_is_large(self):
        """Default max_entries is 50_000 (high enough for normal use)."""
        config = PromptAuditConfig(path=Path("/dev/null"))
        assert config.max_entries == 50_000

    def test_append_exchange_respects_cap(self, tmp_path: Path):
        """append_exchange() also respects the entry cap."""
        log_path = tmp_path / "audit.jsonl"
        config = PromptAuditConfig(path=log_path, max_entries=2)
        log = PromptAuditLog(config)

        for i in range(5):
            log.append_exchange(
                agent_id="a",
                kind="act",
                epoch=i,
                step=0,
                provider="test",
                model="test",
                system_prompt="sys",
                user_prompt="user",
                response_text="resp",
                input_tokens=10,
                output_tokens=10,
            )

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

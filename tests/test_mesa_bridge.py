"""Tests for the Mesa ABM bridge.

Protocol-mode tests run without Mesa installed.
Full-model tests are gated behind a skipif marker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swarm.bridges.mesa import MesaBridge, MesaBridgeConfig, MesaBridgeError
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestMesaBridgeConfig:
    def test_defaults(self):
        cfg = MesaBridgeConfig()
        assert cfg.model_id == "mesa-model"
        assert cfg.proxy_sigmoid_k > 0
        assert cfg.max_agents_per_step >= 1

    def test_empty_model_id_raises(self):
        with pytest.raises(ValidationError):
            MesaBridgeConfig(model_id="   ")

    def test_max_agents_validated(self):
        with pytest.raises(ValidationError):
            MesaBridgeConfig(max_agents_per_step=0)


# ---------------------------------------------------------------------------
# Protocol mode (dict-based agents)
# ---------------------------------------------------------------------------


class TestMesaBridgeProtocolMode:
    def test_record_agent_states_returns_interactions(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        agents = [
            {"agent_id": "a1", "task_progress": 0.9},
            {"agent_id": "a2", "task_progress": 0.3},
        ]
        interactions = bridge.record_agent_states(agents)
        assert len(interactions) == 2
        for ix in interactions:
            assert isinstance(ix, SoftInteraction)

    def test_p_invariant_for_all_agents(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        agents = [
            {"agent_id": f"a{i}", "task_progress": i / 10}
            for i in range(10)
        ]
        interactions = bridge.record_agent_states(agents)
        for ix in interactions:
            assert 0.0 <= ix.p <= 1.0

    def test_high_task_progress_gives_higher_p(self):
        cfg = MesaBridgeConfig(enable_event_log=False)
        bridge_high = MesaBridge(config=cfg)
        bridge_low = MesaBridge(config=cfg)
        ix_high = bridge_high.record_agent_states(
            [{"agent_id": "a1", "task_progress": 1.0}]
        )[0]
        ix_low = bridge_low.record_agent_states(
            [{"agent_id": "a1", "task_progress": 0.0}]
        )[0]
        assert ix_high.p > ix_low.p

    def test_rework_count_from_agent_state(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        agents = [{"agent_id": "a1", "task_progress": 1.0, "rework_count": 5}]
        ix = bridge.record_agent_states(agents)[0]
        assert ix.rework_count == 5

    def test_missing_attrs_use_defaults(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        agents = [{"agent_id": "a1"}]  # no task_progress, rework_count, etc.
        interactions = bridge.record_agent_states(agents)
        assert len(interactions) == 1
        assert 0.0 <= interactions[0].p <= 1.0

    def test_max_agents_per_step_caps_output(self):
        cfg = MesaBridgeConfig(max_agents_per_step=3, enable_event_log=False)
        bridge = MesaBridge(config=cfg)
        agents = [{"agent_id": f"a{i}"} for i in range(10)]
        interactions = bridge.record_agent_states(agents)
        assert len(interactions) == 3

    def test_interactions_accumulate_across_calls(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        bridge.record_agent_states([{"agent_id": "a1"}])
        bridge.record_agent_states([{"agent_id": "a2"}])
        assert len(bridge.get_interactions()) == 2

    def test_get_interactions_returns_copy(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        bridge.record_agent_states([{"agent_id": "a1"}])
        copy = bridge.get_interactions()
        copy.clear()
        assert len(bridge.get_interactions()) == 1

    def test_initiator_is_model_id(self):
        cfg = MesaBridgeConfig(model_id="my-abm", enable_event_log=False)
        bridge = MesaBridge(config=cfg)
        ix = bridge.record_agent_states([{"agent_id": "a1"}])[0]
        assert ix.initiator == "my-abm"

    def test_counterparty_is_agent_id(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        ix = bridge.record_agent_states([{"agent_id": "agent-99"}])[0]
        assert ix.counterparty == "agent-99"

    def test_step_without_model_raises(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        with pytest.raises(MesaBridgeError, match="No model provided"):
            bridge.step()

    def test_toxicity_rate_zero_when_empty(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        assert bridge.get_toxicity_rate() == 0.0

    def test_step_count_starts_at_zero(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        assert bridge.step_count == 0

    def test_out_of_range_task_progress_clamped(self):
        bridge = MesaBridge(config=MesaBridgeConfig(enable_event_log=False))
        # task_progress > 1 should be clamped to 1.0
        agents = [{"agent_id": "a1", "task_progress": 999.0}]
        ix = bridge.record_agent_states(agents)[0]
        assert 0.0 <= ix.p <= 1.0


# ---------------------------------------------------------------------------
# Full Mesa model tests
# ---------------------------------------------------------------------------


try:
    import mesa  # noqa: F401

    HAS_MESA = True
except ImportError:
    HAS_MESA = False


@pytest.mark.skipif(not HAS_MESA, reason="mesa not installed")
class TestMesaBridgeFullModel:
    def _make_model(self, n_agents: int = 5):
        """Build a minimal Mesa model for testing."""
        import mesa as _mesa

        class SimpleAgent(_mesa.Agent):
            def __init__(self, unique_id, model):
                super().__init__(unique_id, model)
                self.task_progress = 0.8
                self.rework_count = 1

            def step(self):
                pass

        class SimpleModel(_mesa.Model):
            def __init__(self, n):
                super().__init__()
                self.schedule = _mesa.time.RandomActivation(self)
                for i in range(n):
                    self.schedule.add(SimpleAgent(i, self))

            def step(self):
                self.schedule.step()

        return SimpleModel(n_agents)

    def test_bridge_step_returns_interactions(self):
        bridge = MesaBridge(
            model=self._make_model(5),
            config=MesaBridgeConfig(enable_event_log=False),
        )
        interactions = bridge.step()
        assert len(interactions) == 5
        for ix in interactions:
            assert 0.0 <= ix.p <= 1.0

    def test_step_count_increments(self):
        bridge = MesaBridge(
            model=self._make_model(3),
            config=MesaBridgeConfig(enable_event_log=False),
        )
        bridge.step()
        bridge.step()
        assert bridge.step_count == 2

    def test_run_accumulates_all_interactions(self):
        bridge = MesaBridge(
            model=self._make_model(4),
            config=MesaBridgeConfig(enable_event_log=False),
        )
        all_ix = bridge.run(n_steps=3)
        assert len(all_ix) == 12  # 4 agents × 3 steps

    def test_p_invariant_across_all_steps(self):
        bridge = MesaBridge(
            model=self._make_model(5),
            config=MesaBridgeConfig(enable_event_log=False),
        )
        interactions = bridge.run(n_steps=5)
        for ix in interactions:
            assert 0.0 <= ix.p <= 1.0

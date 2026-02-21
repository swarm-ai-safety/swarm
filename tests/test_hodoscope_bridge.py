"""Tests for the hodoscope bridge (mapper + bridge with mocked hodoscope)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from swarm.analysis.aggregation import SimulationHistory
from swarm.bridges.hodoscope.bridge import HodoscopeBridge
from swarm.bridges.hodoscope.config import HodoscopeConfig
from swarm.bridges.hodoscope.mapper import HodoscopeMapper
from swarm.models.interaction import InteractionType, SoftInteraction
from tests.fixtures.interactions import (
    generate_benign_batch,
    generate_mixed_batch,
    generate_self_optimizer_scenario,
    generate_toxic_batch,
)

# ── Mapper: basic trajectory conversion ────────────────────────────


class TestInteractionToMessages:
    """Test single-interaction message conversion."""

    def test_initiator_perspective(self):
        ix = generate_benign_batch(1, seed=42)[0]
        mapper = HodoscopeMapper()
        msgs = mapper.interaction_to_messages(ix, ix.initiator)

        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"

        # Assistant turn has tool_calls with "propose"
        tc = msgs[0]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "propose"
        args = json.loads(tc["function"]["arguments"])
        assert args["role"] == "initiator"
        assert args["counterparty"] == ix.counterparty

        # Tool turn references the same call
        assert msgs[1]["tool_call_id"] == tc["id"]
        outcome = json.loads(msgs[1]["content"])
        assert "p" in outcome
        assert "accepted" in outcome

    def test_counterparty_perspective_accepted(self):
        ix = generate_benign_batch(1, seed=42, acceptance_rate=1.0)[0]
        mapper = HodoscopeMapper()
        msgs = mapper.interaction_to_messages(ix, ix.counterparty)

        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "accept"
        args = json.loads(tc["function"]["arguments"])
        assert args["role"] == "counterparty"

    def test_counterparty_perspective_rejected(self):
        ix = generate_benign_batch(1, seed=42, acceptance_rate=0.0)[0]
        mapper = HodoscopeMapper()
        msgs = mapper.interaction_to_messages(ix, ix.counterparty)

        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "reject"


class TestInteractionsToTrajectories:
    """Test batch trajectory conversion."""

    def test_benign_batch(self):
        interactions = generate_benign_batch(10, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        assert len(trajectories) > 0
        for traj in trajectories:
            assert "trajectory_id" in traj
            assert "messages" in traj
            assert "metadata" in traj
            assert len(traj["messages"]) > 0
            # Messages alternate assistant/tool
            for i in range(0, len(traj["messages"]), 2):
                assert traj["messages"][i]["role"] == "assistant"
                if i + 1 < len(traj["messages"]):
                    assert traj["messages"][i + 1]["role"] == "tool"

    def test_toxic_batch(self):
        interactions = generate_toxic_batch(10, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        assert len(trajectories) > 0

    def test_mixed_batch(self):
        interactions = generate_mixed_batch(20, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        assert len(trajectories) > 0

    def test_agents_map_overrides_metadata(self):
        interactions = generate_benign_batch(5, seed=42)
        agent_id = interactions[0].initiator
        agents_map = {agent_id: {"agent_type": "custom_type", "extra": 123}}

        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(
            interactions, agents_map
        )

        # Find trajectory for this agent
        agent_trajs = [
            t for t in trajectories if t["metadata"]["agent_id"] == agent_id
        ]
        assert len(agent_trajs) > 0
        assert agent_trajs[0]["metadata"]["agent_type"] == "custom_type"
        assert agent_trajs[0]["metadata"]["extra"] == 123

    def test_trajectory_metadata_fields(self):
        interactions = generate_benign_batch(5, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        for traj in trajectories:
            meta = traj["metadata"]
            assert "agent_id" in meta
            assert "agent_type" in meta
            assert "avg_p" in meta
            assert "n_interactions" in meta
            assert "n_accepted" in meta
            assert "n_initiated" in meta
            assert "acceptance_rate" in meta
            assert 0.0 <= meta["avg_p"] <= 1.0
            assert meta["n_interactions"] > 0


class TestTrajectoryUnit:
    """Test different trajectory_unit settings."""

    def test_agent_epoch_grouping(self):
        epochs = generate_self_optimizer_scenario(
            n_epochs=3, honest_per_epoch=2, optimizer_per_epoch=2, seed=42
        )
        flat = [ix for epoch_ixs in epochs for ix in epoch_ixs]

        config = HodoscopeConfig(trajectory_unit="agent_epoch")
        mapper = HodoscopeMapper(config)
        trajectories = mapper.interactions_to_trajectories(flat, {})

        # Should have epoch in metadata for each trajectory
        for traj in trajectories:
            assert "epoch" in traj["metadata"]

    def test_agent_run_grouping(self):
        epochs = generate_self_optimizer_scenario(
            n_epochs=3, honest_per_epoch=2, optimizer_per_epoch=2, seed=42
        )
        flat = [ix for epoch_ixs in epochs for ix in epoch_ixs]

        config = HodoscopeConfig(trajectory_unit="agent_run")
        mapper = HodoscopeMapper(config)
        trajectories = mapper.interactions_to_trajectories(flat, {})

        # Each unique agent should have exactly one trajectory
        agent_ids = {t["metadata"]["agent_id"] for t in trajectories}
        assert len(trajectories) == len(agent_ids)

    def test_interaction_unit(self):
        interactions = generate_benign_batch(5, seed=42)
        config = HodoscopeConfig(trajectory_unit="interaction")
        mapper = HodoscopeMapper(config)
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        # Each interaction produces 2 trajectories (one per participant)
        assert len(trajectories) == len(interactions) * 2


class TestMultiEpochTrajectories:
    """Test with multi-epoch self-optimizer scenario."""

    def test_self_optimizer_scenario(self):
        epochs = generate_self_optimizer_scenario(
            n_epochs=5, honest_per_epoch=3, optimizer_per_epoch=3, seed=42
        )
        flat = [ix for epoch_ixs in epochs for ix in epoch_ixs]

        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(flat, {})

        assert len(trajectories) > 0

        # Check that different agent types are represented
        agent_types = {t["metadata"]["agent_type"] for t in trajectories}
        assert "honest" in agent_types or "self_optimizer" in agent_types


# ── Mapper: edge cases ─────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases for the mapper."""

    def test_empty_interactions(self):
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories([], {})
        assert trajectories == []

    def test_single_agent_both_roles(self):
        """Agent appears as both initiator and counterparty."""
        ix = SoftInteraction(
            initiator="agent_1",
            counterparty="agent_1",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            p=0.8,
            v_hat=0.6,
        )
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories([ix], {})

        # agent_1 should have one trajectory (grouped into single epoch)
        assert len(trajectories) == 1
        # But it should have 4 messages (2 as initiator + 2 as counterparty)
        assert len(trajectories[0]["messages"]) == 4

    def test_agent_with_no_accepted_interactions(self):
        interactions = generate_benign_batch(5, seed=42, acceptance_rate=0.0)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        for traj in trajectories:
            assert traj["metadata"]["n_accepted"] == 0
            assert traj["metadata"]["acceptance_rate"] == 0.0


# ── Mapper: write_trajectory_dir ───────────────────────────────────


class TestWriteTrajectoryDir:
    """Test writing trajectories to disk."""

    def test_write_creates_files(self, tmp_path):
        interactions = generate_benign_batch(5, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        out_dir = mapper.write_trajectory_dir(trajectories, tmp_path)
        assert out_dir == tmp_path

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == len(trajectories)

        # Verify each file is valid JSON with expected structure
        for fp in json_files:
            data = json.loads(fp.read_text())
            assert "trajectory_id" in data
            assert "messages" in data
            assert "metadata" in data

    def test_write_uses_config_output_dir(self, tmp_path):
        config = HodoscopeConfig(output_dir=tmp_path / "trajs")
        mapper = HodoscopeMapper(config)
        trajectories = [
            {
                "trajectory_id": "test_1",
                "messages": [],
                "metadata": {"agent_id": "a1"},
            }
        ]
        out_dir = mapper.write_trajectory_dir(trajectories)
        assert out_dir == tmp_path / "trajs"
        assert (tmp_path / "trajs" / "test_1.json").exists()


# ── Bridge: mocked hodoscope ──────────────────────────────────────


class TestHodoscopeBridge:
    """Test HodoscopeBridge with mocked hodoscope imports."""

    @patch("swarm.bridges.hodoscope.bridge.HodoscopeMapper")
    def test_analyze_history(self, MockMapper, tmp_path):
        mock_analyze = MagicMock(return_value=str(tmp_path / ".hodoscope.json"))
        config = HodoscopeConfig(output_dir=tmp_path)

        with patch.dict(
            "sys.modules",
            {"hodoscope": MagicMock(analyze=mock_analyze)},
        ):
            bridge = HodoscopeBridge(config)
            bridge.mapper = MockMapper.return_value
            bridge.mapper.history_to_trajectories.return_value = [
                {"trajectory_id": "t1", "messages": [], "metadata": {}}
            ]
            bridge.mapper.write_trajectory_dir.return_value = tmp_path / "trajs"

            history = MagicMock(spec=SimulationHistory)
            result = bridge.analyze_history(history)

            bridge.mapper.history_to_trajectories.assert_called_once_with(history)
            bridge.mapper.write_trajectory_dir.assert_called_once()
            mock_analyze.assert_called_once()
            assert isinstance(result, Path)

    def test_visualize(self, tmp_path):
        mock_visualize = MagicMock(return_value=str(tmp_path / "viz.html"))
        config = HodoscopeConfig(output_dir=tmp_path)

        with patch.dict(
            "sys.modules",
            {"hodoscope": MagicMock(visualize=mock_visualize)},
        ):
            bridge = HodoscopeBridge(config)
            result = bridge.visualize(
                tmp_path / ".hodoscope.json", group_by="agent_type"
            )

            mock_visualize.assert_called_once()
            assert isinstance(result, Path)

    def test_sample(self, tmp_path):
        mock_sample = MagicMock(
            return_value={"honest": [{"id": "t1"}], "adversary": [{"id": "t2"}]}
        )
        config = HodoscopeConfig(output_dir=tmp_path)

        with patch.dict(
            "sys.modules",
            {"hodoscope": MagicMock(sample=mock_sample)},
        ):
            bridge = HodoscopeBridge(config)
            result = bridge.sample(tmp_path / ".hodoscope.json", n=3)

            mock_sample.assert_called_once()
            assert "honest" in result
            assert "adversary" in result


# ── Trajectory JSON schema validation ─────────────────────────────


class TestTrajectorySchema:
    """Validate trajectory JSON matches hodoscope's expected format."""

    def test_message_format_matches_openai_chat(self):
        interactions = generate_mixed_batch(10, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        for traj in trajectories:
            for msg in traj["messages"]:
                assert "role" in msg
                assert msg["role"] in ("assistant", "tool")

                if msg["role"] == "assistant":
                    assert "tool_calls" in msg
                    for tc in msg["tool_calls"]:
                        assert "id" in tc
                        assert "type" in tc
                        assert tc["type"] == "function"
                        assert "function" in tc
                        assert "name" in tc["function"]
                        assert "arguments" in tc["function"]
                        # Arguments must be valid JSON
                        json.loads(tc["function"]["arguments"])

                elif msg["role"] == "tool":
                    assert "tool_call_id" in msg
                    assert "content" in msg
                    # Content must be valid JSON
                    json.loads(msg["content"])

    def test_tool_call_ids_match(self):
        """Each tool response references a valid tool_call_id."""
        interactions = generate_benign_batch(5, seed=42)
        mapper = HodoscopeMapper()
        trajectories = mapper.interactions_to_trajectories(interactions, {})

        for traj in trajectories:
            msgs = traj["messages"]
            for i in range(0, len(msgs), 2):
                if i + 1 < len(msgs):
                    tc_id = msgs[i]["tool_calls"][0]["id"]
                    assert msgs[i + 1]["tool_call_id"] == tc_id

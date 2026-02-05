"""Tests for mission economy."""

import pytest

from src.env.mission import (
    MissionConfig,
    MissionEconomy,
    MissionObjective,
    MissionStatus,
)
from src.models.interaction import SoftInteraction


def _make_interaction(interaction_id: str, p: float, accepted: bool = True) -> SoftInteraction:
    """Helper to create a test interaction."""
    return SoftInteraction(
        interaction_id=interaction_id,
        p=p,
        accepted=accepted,
        initiator="agent_1",
        counterparty="agent_2",
    )


class TestMissionConfig:
    """Tests for MissionConfig validation."""

    def test_default_config_valid(self):
        config = MissionConfig()
        config.validate()

    def test_invalid_min_participants(self):
        with pytest.raises(ValueError, match="min_participants"):
            MissionConfig(min_participants=0).validate()

    def test_invalid_max_active(self):
        with pytest.raises(ValueError, match="max_active_missions"):
            MissionConfig(max_active_missions=0).validate()

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="reward_distribution"):
            MissionConfig(reward_distribution="invalid").validate()


class TestMissionEconomy:
    """Tests for the MissionEconomy engine."""

    def test_propose_mission(self):
        economy = MissionEconomy()
        objectives = [
            MissionObjective(description="High quality", target_metric="avg_p", target_value=0.7)
        ]
        mission = economy.propose_mission(
            coordinator_id="agent_1",
            name="Test Mission",
            objectives=objectives,
            reward_pool=100.0,
            deadline_epoch=10,
        )
        assert mission is not None
        assert mission.status == MissionStatus.PROPOSED
        assert "agent_1" in mission.participants

    def test_max_active_missions(self):
        economy = MissionEconomy(MissionConfig(max_active_missions=1))
        m1 = economy.propose_mission("a1", "m1", [], 10.0, 10)
        assert m1 is not None
        m2 = economy.propose_mission("a2", "m2", [], 10.0, 10)
        assert m2 is None

    def test_invalid_reward_pool(self):
        economy = MissionEconomy()
        m = economy.propose_mission("a1", "m1", [], 0.0, 10)
        assert m is None

    def test_join_mission(self):
        economy = MissionEconomy(MissionConfig(min_participants=2))
        mission = economy.propose_mission("a1", "m1", [], 10.0, 10)
        assert mission.status == MissionStatus.PROPOSED

        result = economy.join_mission("a2", mission.mission_id)
        assert result is True
        assert mission.status == MissionStatus.ACTIVE
        assert "a2" in mission.participants

    def test_join_nonexistent_mission(self):
        economy = MissionEconomy()
        result = economy.join_mission("a1", "nonexistent")
        assert result is False

    def test_record_contribution(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        mission = economy.propose_mission("a1", "m1", [], 10.0, 10)
        assert mission.status == MissionStatus.ACTIVE

        interaction = _make_interaction("i1", p=0.8)
        result = economy.record_contribution("a1", mission.mission_id, interaction)
        assert result is True
        assert "a1" in mission.contributions
        assert "i1" in mission.contributions["a1"]

    def test_contribution_requires_active_mission(self):
        economy = MissionEconomy(MissionConfig(min_participants=2))
        mission = economy.propose_mission("a1", "m1", [], 10.0, 10)
        # Mission is PROPOSED, not ACTIVE
        interaction = _make_interaction("i1", p=0.8)
        result = economy.record_contribution("a1", mission.mission_id, interaction)
        assert result is False

    def test_contribution_requires_participant(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        mission = economy.propose_mission("a1", "m1", [], 10.0, 10)
        interaction = _make_interaction("i1", p=0.8)
        result = economy.record_contribution("a2", mission.mission_id, interaction)
        assert result is False

    def test_evaluate_mission_objectives_met(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.7)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 10)

        interactions = [_make_interaction(f"i{i}", p=0.85) for i in range(5)]
        for i in interactions:
            economy.record_contribution("a1", mission.mission_id, i)

        result = economy.evaluate_mission(mission.mission_id, interactions)
        assert result["all_objectives_met"] is True
        assert result["status"] == "succeeded"

    def test_evaluate_mission_objectives_not_met(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.9)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 20)

        interactions = [_make_interaction(f"i{i}", p=0.5) for i in range(5)]
        for i in interactions:
            economy.record_contribution("a1", mission.mission_id, i)

        result = economy.evaluate_mission(
            mission.mission_id, interactions, current_epoch=5
        )
        assert result["all_objectives_met"] is False

    def test_expired_mission_fails(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.9)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 5)

        interactions = [_make_interaction(f"i{i}", p=0.5) for i in range(3)]
        for i in interactions:
            economy.record_contribution("a1", mission.mission_id, i)

        result = economy.evaluate_mission(
            mission.mission_id, interactions, current_epoch=5
        )
        assert result["status"] == "failed"

    def test_distribute_rewards_equal(self):
        economy = MissionEconomy(MissionConfig(
            min_participants=1,
            reward_distribution="equal",
        ))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.5)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 10)
        economy.join_mission("a2", mission.mission_id)

        interactions = [
            _make_interaction("i1", p=0.8),
            _make_interaction("i2", p=0.7),
        ]
        economy.record_contribution("a1", mission.mission_id, interactions[0])
        economy.record_contribution("a2", mission.mission_id, interactions[1])

        economy.evaluate_mission(mission.mission_id, interactions)
        rewards = economy.distribute_rewards(mission.mission_id, interactions)

        assert len(rewards) == 2
        assert abs(rewards["a1"] - rewards["a2"]) < 0.01
        assert abs(sum(rewards.values()) - 100.0) < 0.01

    def test_distribute_rewards_proportional(self):
        economy = MissionEconomy(MissionConfig(
            min_participants=1,
            reward_distribution="proportional",
        ))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.5)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 10)
        economy.join_mission("a2", mission.mission_id)

        # a1 contributes 3 high-quality interactions, a2 contributes 1 low-quality
        high_interactions = [_make_interaction(f"h{i}", p=0.9) for i in range(3)]
        low_interactions = [_make_interaction("l1", p=0.3)]
        all_interactions = high_interactions + low_interactions

        for i in high_interactions:
            economy.record_contribution("a1", mission.mission_id, i)
        for i in low_interactions:
            economy.record_contribution("a2", mission.mission_id, i)

        economy.evaluate_mission(mission.mission_id, all_interactions)
        rewards = economy.distribute_rewards(mission.mission_id, all_interactions)

        # a1 should get more (more contributions, higher quality)
        assert rewards.get("a1", 0) > rewards.get("a2", 0)
        assert abs(sum(rewards.values()) - 100.0) < 0.01

    def test_distribute_rewards_shapley(self):
        economy = MissionEconomy(MissionConfig(
            min_participants=1,
            reward_distribution="shapley",
        ))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.5)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 10)
        economy.join_mission("a2", mission.mission_id)

        interactions = [
            _make_interaction("i1", p=0.8),
            _make_interaction("i2", p=0.7),
        ]
        economy.record_contribution("a1", mission.mission_id, interactions[0])
        economy.record_contribution("a2", mission.mission_id, interactions[1])

        economy.evaluate_mission(mission.mission_id, interactions)
        rewards = economy.distribute_rewards(mission.mission_id, interactions)

        assert len(rewards) == 2
        assert abs(sum(rewards.values()) - 100.0) < 0.01

    def test_no_reward_for_failed_mission(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        objectives = [
            MissionObjective(target_metric="avg_p", target_value=0.9)
        ]
        mission = economy.propose_mission("a1", "m1", objectives, 100.0, 5)
        interactions = [_make_interaction("i1", p=0.3)]
        economy.record_contribution("a1", mission.mission_id, interactions[0])

        economy.evaluate_mission(mission.mission_id, interactions, current_epoch=5)
        rewards = economy.distribute_rewards(mission.mission_id, interactions)
        assert len(rewards) == 0

    def test_free_rider_index(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        mission = economy.propose_mission("a1", "m1", [], 100.0, 10)
        economy.join_mission("a2", mission.mission_id)

        # a1 contributes, a2 does nothing (free rider)
        for i in range(5):
            interaction = _make_interaction(f"i{i}", p=0.8)
            economy.record_contribution("a1", mission.mission_id, interaction)

        fri = economy.free_rider_index(mission.mission_id)
        assert fri > 0  # Inequality should be detected

    def test_free_rider_index_equal(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        mission = economy.propose_mission("a1", "m1", [], 100.0, 10)
        economy.join_mission("a2", mission.mission_id)

        # Both contribute equally
        for i in range(4):
            agent = "a1" if i < 2 else "a2"
            interaction = _make_interaction(f"i{i}", p=0.8)
            economy.record_contribution(agent, mission.mission_id, interaction)

        fri = economy.free_rider_index(mission.mission_id)
        assert fri < 0.3  # Should be near 0 for equal contributions

    def test_expire_missions(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        m1 = economy.propose_mission("a1", "m1", [], 10.0, 5)
        m2 = economy.propose_mission("a2", "m2", [], 10.0, 20)

        expired = economy.expire_missions(current_epoch=10)
        assert m1.mission_id in expired
        assert m2.mission_id not in expired
        assert m1.status == MissionStatus.EXPIRED

    def test_get_active_missions(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        m1 = economy.propose_mission("a1", "m1", [], 10.0, 10)
        m2 = economy.propose_mission("a2", "m2", [], 10.0, 10)
        m2.status = MissionStatus.SUCCEEDED

        active = economy.get_active_missions()
        assert len(active) == 1
        assert active[0].mission_id == m1.mission_id

    def test_mission_serialization(self):
        economy = MissionEconomy(MissionConfig(min_participants=1))
        objectives = [
            MissionObjective(description="Test", target_metric="avg_p", target_value=0.7)
        ]
        mission = economy.propose_mission("a1", "Test", objectives, 100.0, 10)
        d = mission.to_dict()
        assert d["name"] == "Test"
        assert d["status"] == "active"
        assert len(d["objectives"]) == 1

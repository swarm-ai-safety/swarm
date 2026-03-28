"""Tests for HyperagentSelfModAgent — self-modifying agent behavior.

Validates proxy weight shifting, acceptance threshold decay, quality drift,
governance-gap tracking, and interaction quality generation.

Reference: Zhang et al. Hyperagents (arXiv:2603.19461)
"""

import random

from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent, SelfModSnapshot
from swarm.models.agent import AgentType


class TestHyperagentInit:
    def test_default_config(self):
        agent = HyperagentSelfModAgent("h-1", rng=random.Random(42))
        assert agent.agent_type == AgentType.SELF_MODIFYING
        assert agent.acceptance_threshold == 0.6
        assert agent.current_quality == 0.8
        assert agent.modification_depth == 0
        assert abs(sum(agent.proxy_weights.values()) - 1.0) < 1e-9

    def test_custom_config(self):
        agent = HyperagentSelfModAgent(
            "h-2",
            config={
                "modification_rate": 0.2,
                "initial_acceptance_threshold": 0.7,
                "initial_quality": 0.9,
                "quality_floor": 0.25,
                "weight_gaming_target": "task_progress",
            },
            rng=random.Random(42),
        )
        assert agent.modification_rate == 0.2
        assert agent.acceptance_threshold == 0.7
        assert agent.current_quality == 0.9
        assert agent.quality_floor == 0.25
        assert agent.weight_gaming_target == "task_progress"


class TestSelfModification:
    def test_self_modify_returns_snapshot(self):
        agent = HyperagentSelfModAgent("h-3", rng=random.Random(42))
        snapshot = agent.self_modify(epoch=0)
        assert isinstance(snapshot, SelfModSnapshot)
        assert snapshot.epoch == 0
        assert agent.modification_depth == 1

    def test_weight_shifts_toward_target(self):
        """Weight shifts toward gaming target (default: engagement_signal)."""
        agent = HyperagentSelfModAgent(
            "h-4",
            config={"modification_rate": 0.1},
            rng=random.Random(42),
        )
        initial_engagement = agent.proxy_weights["engagement_signal"]
        initial_verifier = agent.proxy_weights["verifier_penalty"]

        agent.self_modify(epoch=0)

        assert agent.proxy_weights["engagement_signal"] > initial_engagement
        assert agent.proxy_weights["verifier_penalty"] < initial_verifier
        assert abs(sum(agent.proxy_weights.values()) - 1.0) < 1e-9

    def test_threshold_decays(self):
        agent = HyperagentSelfModAgent(
            "h-5",
            config={"modification_rate": 0.1, "initial_acceptance_threshold": 0.6},
            rng=random.Random(42),
        )
        initial_threshold = agent.acceptance_threshold

        agent.self_modify(epoch=0)

        assert agent.acceptance_threshold < initial_threshold

    def test_threshold_respects_floor(self):
        """Threshold decay stops at min_acceptance_threshold."""
        agent = HyperagentSelfModAgent(
            "h-6",
            config={
                "modification_rate": 0.5,
                "min_acceptance_threshold": 0.2,
                "initial_acceptance_threshold": 0.3,
            },
            rng=random.Random(42),
        )
        for epoch in range(50):
            agent.self_modify(epoch)

        assert agent.acceptance_threshold >= 0.2

    def test_quality_decays(self):
        agent = HyperagentSelfModAgent(
            "h-7",
            config={"quality_decay_rate": 0.05},
            rng=random.Random(42),
        )
        initial_quality = agent.current_quality

        agent.self_modify(epoch=0)

        assert agent.current_quality < initial_quality

    def test_quality_respects_floor(self):
        """Quality decay stops at quality_floor."""
        agent = HyperagentSelfModAgent(
            "h-8",
            config={"quality_decay_rate": 0.5, "quality_floor": 0.3},
            rng=random.Random(42),
        )
        for epoch in range(50):
            agent.self_modify(epoch)

        assert agent.current_quality >= 0.3


class TestGovernanceGap:
    def test_gap_starts_near_zero(self):
        agent = HyperagentSelfModAgent("h-9", rng=random.Random(42))
        snapshot = agent.self_modify(epoch=0)
        # Baseline ungoverned weight is 0.6 (task_progress + engagement_signal)
        # After first modification, gap should be small but positive
        assert snapshot.governance_gap_local >= 0.0

    def test_gap_grows_with_modifications(self):
        agent = HyperagentSelfModAgent(
            "h-10",
            config={"modification_rate": 0.1},
            rng=random.Random(42),
        )
        gaps = []
        for epoch in range(15):
            snapshot = agent.self_modify(epoch)
            gaps.append(snapshot.governance_gap_local)

        # Overall trend: gap at end should exceed gap at start
        assert gaps[-1] > gaps[0]
        # After 15 mods, gap should be > 0
        assert gaps[-1] > 0.0

    def test_weight_cap_prevents_runaway(self):
        """Gaming target weight is capped at 0.8."""
        agent = HyperagentSelfModAgent(
            "h-11",
            config={"modification_rate": 0.5},
            rng=random.Random(42),
        )
        for epoch in range(50):
            agent.self_modify(epoch)

        # engagement_signal (gaming target) should be capped
        assert agent.proxy_weights["engagement_signal"] <= 0.85  # after normalization


class TestQualityGeneration:
    def test_quality_in_bounds(self):
        agent = HyperagentSelfModAgent("h-12", rng=random.Random(42))
        for _ in range(100):
            p = agent.generate_interaction_quality()
            assert 0.0 <= p <= 1.0

    def test_quality_degrades_with_mods(self):
        agent = HyperagentSelfModAgent(
            "h-13",
            config={"quality_decay_rate": 0.05},
            rng=random.Random(42),
        )
        initial_quality = agent.current_quality
        for epoch in range(10):
            agent.self_modify(epoch)

        # Quality distribution should shift left
        assert agent.current_quality < initial_quality


class TestModHistory:
    def test_history_is_append_only(self):
        agent = HyperagentSelfModAgent("h-14", rng=random.Random(42))
        for epoch in range(5):
            agent.self_modify(epoch)

        assert len(agent.modification_history) == 5
        for i, snapshot in enumerate(agent.modification_history):
            assert snapshot.epoch == i

    def test_snapshot_contains_weights(self):
        agent = HyperagentSelfModAgent("h-15", rng=random.Random(42))
        snapshot = agent.self_modify(epoch=0)

        assert "task_progress" in snapshot.proxy_weights
        assert "verifier_penalty" in snapshot.proxy_weights
        assert abs(sum(snapshot.proxy_weights.values()) - 1.0) < 1e-9

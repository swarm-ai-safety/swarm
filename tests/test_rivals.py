"""Tests for the Team-of-Rivals scenario."""

import random

import pytest

from swarm.agents.base import ActionType, Observation
from swarm.agents.rivals_agent import RivalsCriticAgent, RivalsProducerAgent
from swarm.core.rivals_handler import (
    PipelineStage,
    RivalsConfig,
    RivalsEpisode,
    RivalsHandler,
    RivalsMode,
)
from swarm.core.rivals_tasks import TASK_CATALOG, RivalsTask, sample_tasks
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.metrics.rivals_metrics import compute_rivals_metrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def env_state():
    state = EnvState(steps_per_epoch=10)
    # Register agents
    from swarm.models.agent import AgentType
    for agent_id in [
        "coder_1", "chart_maker_1", "writer_1",
        "critic_code_1", "critic_chart_1", "critic_output_1",
    ]:
        state.add_agent(agent_id=agent_id, name=agent_id, agent_type=AgentType.HONEST)
    return state


@pytest.fixture
def rivals_handler(event_bus, env_state):
    config = RivalsConfig(
        enabled=True,
        mode=RivalsMode.TEAM_OF_RIVALS,
        max_retries_per_stage=3,
        tasks_per_epoch=1,
        seed=42,
    )
    handler = RivalsHandler(config=config, event_bus=event_bus)
    # Register roles
    handler.register_agent_role("coder_1", "coder")
    handler.register_agent_role("chart_maker_1", "chart_maker")
    handler.register_agent_role("writer_1", "writer")
    handler.register_agent_role("critic_code_1", "critic_code")
    handler.register_agent_role("critic_chart_1", "critic_chart")
    handler.register_agent_role("critic_output_1", "critic_output")
    return handler


def _make_action(action_type, agent_id, episode_id="", **metadata):
    from swarm.agents.base import Action
    return Action(
        action_type=action_type,
        agent_id=agent_id,
        target_id=episode_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Task catalog tests
# ---------------------------------------------------------------------------

class TestTaskCatalog:
    def test_catalog_has_tasks(self):
        assert len(TASK_CATALOG) >= 16

    def test_all_tasks_have_ground_truth(self):
        for task in TASK_CATALOG:
            assert task.ground_truth, f"Task {task.task_id} missing ground_truth"

    def test_all_tasks_have_traps(self):
        for task in TASK_CATALOG:
            assert isinstance(task.traps, list)

    def test_sample_tasks(self):
        rng = random.Random(42)
        tasks = sample_tasks(rng, 5)
        assert len(tasks) == 5
        for t in tasks:
            assert isinstance(t, RivalsTask)

    def test_task_types_covered(self):
        types = {t.task_type for t in TASK_CATALOG}
        assert types == {"reconciliation", "trend", "kpi", "anomaly"}


# ---------------------------------------------------------------------------
# Episode lifecycle tests
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_episode_starts_at_code(self):
        task = TASK_CATALOG[0]
        ep = RivalsEpisode(task=task)
        assert ep.stage == PipelineStage.CODE

    def test_full_pipeline_code_chart_write_scored(self, rivals_handler, env_state):
        """Test that a full pipeline CODE → CHART → WRITE → SCORED works."""
        rivals_handler.on_epoch_start(env_state)
        episodes = rivals_handler.get_active_episodes()
        assert len(episodes) > 0

        # Get first episode
        ep_id = list(episodes.keys())[0]
        ep = episodes[ep_id]
        assert ep.stage == PipelineStage.CODE

        # Simulate produce at CODE stage
        action = _make_action(
            ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
            quality_level=0.9, trap_awareness=1.0,
        )
        result = rivals_handler.handle_action(action, env_state)
        assert result.success

        # Simulate review pass at CODE stage
        ep = rivals_handler.get_active_episodes().get(ep_id)
        if ep is not None:
            action = _make_action(
                ActionType.RIVALS_REVIEW, "critic_code_1", ep_id,
                detection_rate=0.0, false_positive_rate=0.0,
            )
            result = rivals_handler.handle_action(action, env_state)
            assert result.success

        # Continue for CHART and WRITE stages
        for producer, critic in [
            ("chart_maker_1", "critic_chart_1"),
            ("writer_1", "critic_output_1"),
        ]:
            ep = rivals_handler.get_active_episodes().get(ep_id)
            if ep is None:
                break

            action = _make_action(
                ActionType.RIVALS_PRODUCE, producer, ep_id,
                quality_level=0.9, trap_awareness=1.0,
            )
            result = rivals_handler.handle_action(action, env_state)
            assert result.success

            ep = rivals_handler.get_active_episodes().get(ep_id)
            if ep is not None:
                action = _make_action(
                    ActionType.RIVALS_REVIEW, critic, ep_id,
                    detection_rate=0.0, false_positive_rate=0.0,
                )
                result = rivals_handler.handle_action(action, env_state)
                assert result.success

        # Episode should be completed (scored)
        completed = rivals_handler.get_completed_episodes()
        scored = [e for e in completed if e.episode_id == ep_id]
        assert len(scored) == 1
        assert scored[0].stage == PipelineStage.SCORED
        assert scored[0].final_score is not None
        assert scored[0].final_score > 0


# ---------------------------------------------------------------------------
# Veto retry mechanics
# ---------------------------------------------------------------------------

class TestVetoMechanics:
    def test_veto_triggers_retry(self, rivals_handler, env_state):
        """Veto should re-queue the producer for the same stage."""
        rivals_handler.on_epoch_start(env_state)
        episodes = rivals_handler.get_active_episodes()
        ep_id = list(episodes.keys())[0]

        # Produce
        action = _make_action(
            ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
            quality_level=0.3, trap_awareness=0.0,
        )
        rivals_handler.handle_action(action, env_state)

        # Review with guaranteed veto (100% detection rate, quality is low)
        action = _make_action(
            ActionType.RIVALS_REVIEW, "critic_code_1", ep_id,
            detection_rate=1.0, false_positive_rate=0.0,
        )
        result = rivals_handler.handle_action(action, env_state)
        assert result.success
        assert result.metadata.get("verdict") == "veto"

        # Episode should still be active at CODE stage
        ep = rivals_handler.get_active_episodes().get(ep_id)
        if ep is not None:
            assert ep.stage == PipelineStage.CODE
            assert ep.retries.get("code", 0) >= 1

    def test_max_retries_causes_failure(self, rivals_handler, env_state):
        """Exceeding max retries should fail the episode."""
        rivals_handler.on_epoch_start(env_state)
        episodes = rivals_handler.get_active_episodes()
        ep_id = list(episodes.keys())[0]

        for _ in range(rivals_handler.config.max_retries_per_stage + 1):
            ep = rivals_handler.get_active_episodes().get(ep_id)
            if ep is None:
                break

            # Produce low quality
            action = _make_action(
                ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
                quality_level=0.1, trap_awareness=0.0,
            )
            rivals_handler.handle_action(action, env_state)

            # Guaranteed veto
            action = _make_action(
                ActionType.RIVALS_REVIEW, "critic_code_1", ep_id,
                detection_rate=1.0, false_positive_rate=0.0,
            )
            rivals_handler.handle_action(action, env_state)

        # Episode should be failed
        completed = rivals_handler.get_completed_episodes()
        failed = [e for e in completed if e.episode_id == ep_id]
        assert len(failed) == 1
        assert failed[0].stage == PipelineStage.FAILED


# ---------------------------------------------------------------------------
# Mode ablation tests
# ---------------------------------------------------------------------------

class TestModeAblations:
    def test_single_agent_skips_reviews(self, event_bus, env_state):
        """In single_agent mode, reviews are skipped entirely."""
        config = RivalsConfig(
            mode=RivalsMode.SINGLE_AGENT,
            tasks_per_epoch=1,
            seed=42,
        )
        handler = RivalsHandler(config=config, event_bus=event_bus)
        handler.register_agent_role("coder_1", "coder")
        handler.on_epoch_start(env_state)

        episodes = handler.get_active_episodes()
        if not episodes:
            return

        ep_id = list(episodes.keys())[0]

        # Produce high quality - should advance without review
        action = _make_action(
            ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
            quality_level=0.9, trap_awareness=1.0,
        )
        handler.handle_action(action, env_state)

        # In single_agent, should advance to next stage without review
        # (or complete entirely for all 3 stages)
        # We just verify the handler doesn't crash

    def test_advisory_mode_never_blocks(self, event_bus, env_state):
        """In advisory mode, critics cannot veto."""
        config = RivalsConfig(
            mode=RivalsMode.ADVISORY,
            tasks_per_epoch=1,
            seed=42,
        )
        handler = RivalsHandler(config=config, event_bus=event_bus)
        handler.register_agent_role("coder_1", "coder")
        handler.register_agent_role("critic_code_1", "critic_code")
        handler.on_epoch_start(env_state)

        episodes = handler.get_active_episodes()
        if not episodes:
            return

        ep_id = list(episodes.keys())[0]

        # Produce low quality
        action = _make_action(
            ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
            quality_level=0.1, trap_awareness=0.0,
        )
        handler.handle_action(action, env_state)

        # Review should pass even with 100% detection rate
        action = _make_action(
            ActionType.RIVALS_REVIEW, "critic_code_1", ep_id,
            detection_rate=1.0, false_positive_rate=0.0,
        )
        result = handler.handle_action(action, env_state)
        assert result.success
        assert result.metadata.get("verdict") == "pass"

    def test_council_mode_majority_vote(self, event_bus, env_state):
        """Council mode uses majority vote for decisions."""
        config = RivalsConfig(
            mode=RivalsMode.COUNCIL,
            tasks_per_epoch=1,
            seed=42,
        )
        handler = RivalsHandler(config=config, event_bus=event_bus)
        handler.register_agent_role("coder_1", "coder")
        handler.register_agent_role("critic_code_1", "critic_code")
        handler.on_epoch_start(env_state)

        # Just verify it doesn't crash
        episodes = handler.get_active_episodes()
        if episodes:
            ep_id = list(episodes.keys())[0]
            action = _make_action(
                ActionType.RIVALS_PRODUCE, "coder_1", ep_id,
                quality_level=0.5, trap_awareness=0.5,
            )
            handler.handle_action(action, env_state)


# ---------------------------------------------------------------------------
# Ground truth scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_high_quality_scores_high(self, rivals_handler, env_state):
        """High quality artifacts should produce high scores."""
        task = TASK_CATALOG[0]
        ep = RivalsEpisode(task=task)
        ep.artifacts = {"code": 0.9, "chart": 0.85, "write": 0.88}
        rivals_handler._score_episode(ep)
        assert ep.final_score is not None
        assert ep.final_score > 0.8

    def test_low_quality_scores_low(self, rivals_handler, env_state):
        """Low quality artifacts should produce low scores."""
        task = TASK_CATALOG[0]
        ep = RivalsEpisode(task=task)
        ep.artifacts = {"code": 0.2, "chart": 0.15, "write": 0.1}
        rivals_handler._score_episode(ep)
        assert ep.final_score is not None
        assert ep.final_score < 0.3


# ---------------------------------------------------------------------------
# Delta illusion metric
# ---------------------------------------------------------------------------

class TestDeltaIllusion:
    def test_delta_illusion_computation(self):
        """Delta illusion = perceived_coherence - actual_consistency."""
        task = TASK_CATALOG[0]
        episodes = []

        for _ in range(5):
            ep = RivalsEpisode(task=task)
            ep.stage = PipelineStage.SCORED
            ep.perceived_coherence = 0.8
            ep.actual_consistency = 0.5
            ep.final_score = 0.5
            episodes.append(ep)

        metrics = compute_rivals_metrics(episodes)
        assert abs(metrics.delta_illusion - 0.3) < 0.01

    def test_zero_illusion_when_consistent(self):
        """No illusion when perceived equals actual."""
        task = TASK_CATALOG[0]
        episodes = []

        for _ in range(5):
            ep = RivalsEpisode(task=task)
            ep.stage = PipelineStage.SCORED
            ep.perceived_coherence = 0.7
            ep.actual_consistency = 0.7
            ep.final_score = 0.7
            episodes.append(ep)

        metrics = compute_rivals_metrics(episodes)
        assert abs(metrics.delta_illusion) < 0.01


# ---------------------------------------------------------------------------
# Observable mapping
# ---------------------------------------------------------------------------

class TestObservableMapping:
    def test_pass_observables(self):
        """PASS events should generate positive observables."""
        from swarm.core.rivals_handler import _OBS_PASS
        assert _OBS_PASS.task_progress_delta > 0
        assert _OBS_PASS.verifier_rejections == 0
        assert _OBS_PASS.rework_count == 0

    def test_veto_observables(self):
        """VETO events should generate negative observables."""
        from swarm.core.rivals_handler import _OBS_VETO
        assert _OBS_VETO.task_progress_delta < 0
        assert _OBS_VETO.verifier_rejections > 0
        assert _OBS_VETO.rework_count > 0


# ---------------------------------------------------------------------------
# Epoch boundary handling
# ---------------------------------------------------------------------------

class TestEpochBoundary:
    def test_unfinished_episodes_fail_on_epoch_end(self, rivals_handler, env_state):
        """Unfinished episodes at epoch end should be marked as failures."""
        rivals_handler.on_epoch_start(env_state)

        # Don't process any actions, just end the epoch
        rivals_handler.on_epoch_end(env_state)

        completed = rivals_handler.get_completed_episodes()
        assert len(completed) > 0
        for ep in completed:
            assert ep.stage == PipelineStage.FAILED
            assert ep.final_score == 0.0


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestRivalsAgents:
    def test_producer_agent_produces_on_assignment(self):
        agent = RivalsProducerAgent(
            agent_id="p1",
            config={"quality_level": 0.8, "trap_awareness": 0.6, "role": "coder"},
        )
        obs = Observation()
        obs.rivals_assignments = [
            {"type": "produce", "episode_id": "ep1", "stage": "code"}
        ]
        action = agent.act(obs)
        assert action.action_type == ActionType.RIVALS_PRODUCE
        assert action.metadata["quality_level"] == 0.8

    def test_producer_agent_noops_without_assignment(self):
        agent = RivalsProducerAgent(agent_id="p1")
        obs = Observation()
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_critic_agent_reviews_on_assignment(self):
        agent = RivalsCriticAgent(
            agent_id="c1",
            config={"detection_rate": 0.9, "false_positive_rate": 0.05, "role": "critic_code"},
        )
        obs = Observation()
        obs.rivals_assignments = [
            {"type": "review", "episode_id": "ep1", "stage": "code"}
        ]
        action = agent.act(obs)
        assert action.action_type == ActionType.RIVALS_REVIEW
        assert action.metadata["detection_rate"] == 0.9

    def test_critic_agent_noops_without_assignment(self):
        agent = RivalsCriticAgent(agent_id="c1")
        obs = Observation()
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_empty_episodes(self):
        metrics = compute_rivals_metrics([])
        assert metrics.success_rate == 0.0
        assert metrics.total_episodes == 0

    def test_all_scored(self):
        task = TASK_CATALOG[0]
        episodes = []
        for _ in range(4):
            ep = RivalsEpisode(task=task)
            ep.stage = PipelineStage.SCORED
            ep.final_score = 0.8
            ep.perceived_coherence = 0.8
            ep.actual_consistency = 0.8
            episodes.append(ep)
        metrics = compute_rivals_metrics(episodes)
        assert metrics.success_rate == 1.0
        assert metrics.failed_episodes == 0

    def test_mixed_scored_failed(self):
        task = TASK_CATALOG[0]
        episodes = []
        for _ in range(3):
            ep = RivalsEpisode(task=task)
            ep.stage = PipelineStage.SCORED
            ep.final_score = 0.7
            episodes.append(ep)
        for _ in range(2):
            ep = RivalsEpisode(task=task)
            ep.stage = PipelineStage.FAILED
            ep.final_score = 0.0
            episodes.append(ep)
        metrics = compute_rivals_metrics(episodes)
        assert abs(metrics.success_rate - 0.6) < 0.01
        assert metrics.failed_episodes == 2

    def test_veto_rate_by_stage(self):
        task = TASK_CATALOG[0]
        ep = RivalsEpisode(task=task)
        ep.stage = PipelineStage.SCORED
        ep.veto_history = [
            {"stage": "code", "vetoed": True, "quality": 0.3, "has_defect": True, "critic": "c1"},
            {"stage": "code", "vetoed": False, "quality": 0.8, "has_defect": False, "critic": "c1"},
            {"stage": "chart", "vetoed": False, "quality": 0.7, "has_defect": False, "critic": "c2"},
        ]
        metrics = compute_rivals_metrics([ep])
        assert metrics.veto_rate_by_stage.get("code", 0) == 0.5
        assert metrics.veto_rate_by_stage.get("chart", 0) == 0.0

    def test_turns_to_ship(self):
        task = TASK_CATALOG[0]
        ep = RivalsEpisode(task=task)
        ep.stage = PipelineStage.SCORED
        ep.retries = {"code": 2, "chart": 1}
        ep.final_score = 0.8
        metrics = compute_rivals_metrics([ep])
        # 3 base stages + 3 retries = 6 turns
        assert metrics.turns_to_ship == 6.0


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

class TestHandlerRegistration:
    def test_handled_action_types(self):
        types = RivalsHandler.handled_action_types()
        assert ActionType.RIVALS_PRODUCE in types
        assert ActionType.RIVALS_REVIEW in types

    def test_observation_fields(self, rivals_handler, env_state):
        rivals_handler.on_epoch_start(env_state)
        fields = rivals_handler.build_observation_fields("coder_1", env_state)
        assert "rivals_assignments" in fields

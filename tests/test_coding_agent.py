"""Tests for the CodingAgent and CodingHandler integration."""

import random

from swarm.agents.base import ActionType, Observation
from swarm.agents.coding_agent import CodingAgent
from swarm.core.coding_handler import CodingHandler
from swarm.core.handler import HandlerActionResult
from swarm.core.proxy import ProxyObservables
from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentState, AgentType

# ---------------------------------------------------------------------------
# CodingAgent unit tests
# ---------------------------------------------------------------------------


class TestCodingAgentAct:
    """CodingAgent.act() returns correct action types."""

    def _make_agent(self, adversarial: bool = False, seed: int = 42) -> CodingAgent:
        return CodingAgent(
            agent_id="coder_1",
            config={
                "skill_level": 0.7,
                "review_honesty": 0.9,
                "adversarial": adversarial,
                "subtlety": 0.5,
            },
            rng=random.Random(seed),
        )

    def _obs(self, **kwargs) -> Observation:
        return Observation(agent_state=AgentState(agent_id="coder_1"), **kwargs)

    def test_claims_task_when_available(self):
        agent = self._make_agent()
        obs = self._obs(
            can_claim_task=True,
            available_tasks=[{"task_id": "task-1", "min_reputation": 0}],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.CLAIM_TASK
        assert action.target_id == "task-1"

    def test_submits_output_for_active_task(self):
        agent = self._make_agent()
        obs = self._obs(
            active_tasks=[{"task_id": "task-1", "status": "in_progress"}],
        )
        action = agent.act(obs)
        assert action.action_type == ActionType.SUBMIT_OUTPUT

    def test_noop_when_nothing_available(self):
        agent = self._make_agent()
        obs = self._obs(can_claim_task=False)
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_adversarial_agent_type(self):
        agent = self._make_agent(adversarial=True)
        assert agent.agent_type == AgentType.ADVERSARIAL
        assert agent.adversarial is True

    def test_honest_agent_type(self):
        agent = self._make_agent(adversarial=False)
        assert agent.agent_type == AgentType.HONEST
        assert agent.adversarial is False

    def test_submission_quality_bounded(self):
        """Quality should always be in [0, 1]."""
        agent = self._make_agent(seed=0)
        for _ in range(100):
            q = agent._compute_submission_quality()
            assert 0.0 <= q <= 1.0

    def test_adversarial_submission_has_metadata(self):
        agent = self._make_agent(adversarial=True)
        obs = self._obs(
            active_tasks=[{"task_id": "task-1", "status": "in_progress"}],
        )
        action = agent.act(obs)
        assert "quality" in action.metadata
        assert "has_hidden_issues" in action.metadata


# ---------------------------------------------------------------------------
# CodingHandler unit tests
# ---------------------------------------------------------------------------


class TestCodingHandler:
    """CodingHandler.handle_action() produces valid results."""

    def _make_handler(self, seed: int = 42) -> CodingHandler:
        bus = EventBus()
        return CodingHandler(event_bus=bus, rng=random.Random(seed))

    def test_handled_action_types(self):
        assert ActionType.VERIFY_OUTPUT in CodingHandler.handled_action_types()

    def test_handle_verify_output_produces_observables(self):
        handler = self._make_handler()
        from swarm.agents.base import Action

        action = Action(
            action_type=ActionType.VERIFY_OUTPUT,
            agent_id="reviewer_1",
            target_id="task-1",
            content="code_review",
            metadata={
                "review_honesty": 0.9,
                "reviewer_skill": 0.7,
            },
        )
        # Handler falls back to defaults when no task pool available
        result = handler.handle_action(action, state=_FakeState())
        assert isinstance(result, HandlerActionResult)
        assert result.success is True
        assert result.observables is not None
        obs = result.observables
        assert isinstance(obs, ProxyObservables)
        assert -1.0 <= obs.task_progress_delta <= 1.0
        assert obs.rework_count >= 0
        assert obs.verifier_rejections in (0, 1)
        assert obs.tool_misuse_flags >= 0

    def test_wrong_action_type_fails(self):
        handler = self._make_handler()
        from swarm.agents.base import Action

        action = Action(
            action_type=ActionType.NOOP,
            agent_id="reviewer_1",
        )
        result = handler.handle_action(action, state=_FakeState())
        assert result.success is False

    def test_detection_of_hidden_issues(self):
        """With high honesty and low subtlety, issues should be detected."""
        handler = self._make_handler(seed=1)
        from swarm.agents.base import Action

        # Create a state with a task that has hidden issues
        state = _FakeState(
            submission_meta={
                "submitter_id": "adv_1",
                "quality": 0.7,
                "has_hidden_issues": True,
                "subtlety": 0.1,  # Easy to detect
            }
        )
        action = Action(
            action_type=ActionType.VERIFY_OUTPUT,
            agent_id="reviewer_1",
            target_id="task-1",
            metadata={"review_honesty": 0.95, "reviewer_skill": 0.8},
        )

        # Run many times — with subtlety=0.1 and honesty=0.95,
        # detection_prob = 0.9 * 0.95 = 0.855
        detected = 0
        for seed in range(50):
            handler._rng = random.Random(seed)
            result = handler.handle_action(action, state=state)
            if result.observables.tool_misuse_flags > 0:
                detected += 1
        # Should detect in majority of runs
        assert detected > 25, f"Only detected {detected}/50 times"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, meta: dict):
        self.claimed_by = meta.get("submitter_id", "")
        self.output = meta
        self.submission_metadata = meta


class _FakeTaskPool:
    def __init__(self, meta: dict | None = None):
        self._meta = meta or {
            "submitter_id": "",
            "quality": 0.5,
            "has_hidden_issues": False,
            "subtlety": 0.0,
        }

    def get_task(self, task_id: str):
        return _FakeTask(self._meta)


class _FakeState:
    """Minimal state stub for handler tests."""

    def __init__(self, submission_meta: dict | None = None):
        self.task_pool = _FakeTaskPool(submission_meta)

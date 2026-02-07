"""Integration tests for Moltbook handler with orchestrator."""

from swarm.agents.moltbook_agent import (
    DiligentMoltbookAgent,
    HumanPretenderAgent,
    SpamBotAgent,
)
from swarm.core.moltbook_handler import MoltbookConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.env.moltbook import ContentStatus
from swarm.governance.config import GovernanceConfig


def _build_orchestrator(seed: int, challenge_enabled: bool = True) -> Orchestrator:
    gov = GovernanceConfig(
        moltbook_rate_limit_enabled=True,
        moltbook_post_cooldown_steps=5,
        moltbook_comment_cooldown_steps=1,
        moltbook_daily_comment_cap=50,
        moltbook_request_cap_per_step=100,
        moltbook_challenge_enabled=challenge_enabled,
        moltbook_challenge_difficulty=0.5,
        moltbook_challenge_window_steps=2,
    )
    config = OrchestratorConfig(
        n_epochs=1,
        steps_per_epoch=4,
        seed=seed,
        moltbook_config=MoltbookConfig(enabled=True, seed=seed),
        governance_config=gov,
    )
    return Orchestrator(config=config)


def test_moltbook_publish_flow():
    orch = _build_orchestrator(seed=1, challenge_enabled=True)
    agent = DiligentMoltbookAgent(agent_id="diligent_1", config={"seed": 1})
    orch.register_agent(agent)
    orch.run()

    handler = orch._moltbook_handler
    assert handler is not None
    published = [
        p for p in handler.feed._posts.values()
        if p.status == ContentStatus.PUBLISHED
    ]
    assert published


def test_moltbook_rate_limit_hits_recorded():
    orch = _build_orchestrator(seed=2, challenge_enabled=False)
    agent = SpamBotAgent(agent_id="spam_1", config={"seed": 2})
    orch.register_agent(agent)
    orch.run()

    handler = orch._moltbook_handler
    assert handler is not None
    assert handler.scorer.rate_limit_hits.get("spam_1", 0) > 0


def test_moltbook_challenge_attempts_tracked():
    orch = _build_orchestrator(seed=3, challenge_enabled=True)
    agent = HumanPretenderAgent(agent_id="human_1", config={"seed": 3})
    orch.register_agent(agent)
    orch.run()

    handler = orch._moltbook_handler
    assert handler is not None
    assert handler.scorer.challenge_attempts

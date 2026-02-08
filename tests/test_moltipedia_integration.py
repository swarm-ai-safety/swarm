"""Integration tests for Moltipedia handler with orchestrator."""

import pytest

from swarm.agents.wiki_editor import DiligentEditorAgent
from swarm.core.moltipedia_handler import MoltipediaConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

pytestmark = pytest.mark.slow


def test_moltipedia_run_records_points():
    config = OrchestratorConfig(
        n_epochs=1,
        steps_per_epoch=1,
        moltipedia_config=MoltipediaConfig(
            enabled=True,
            initial_pages=5,
            seed=42,
        ),
    )
    orchestrator = Orchestrator(config=config)
    agent = DiligentEditorAgent(agent_id="editor_1")
    orchestrator.register_agent(agent)

    orchestrator._run_step()

    assert orchestrator.state.completed_interactions
    handler = orchestrator._moltipedia_handler
    assert handler is not None
    assert handler.task_pool.leaderboard.get("editor_1", 0.0) >= 0.0

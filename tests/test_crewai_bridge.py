"""Tests for the CrewAI bridge (protocol mode — no crewai required).

Full crew mode tests require crewai installed and are gated behind a
skipif marker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import importlib
import pytest
from pydantic import ValidationError

from swarm.bridges.crewai import (
    CrewAIBridge,
    CrewAIBridgeConfig,
    CrewAIBridgeError,
    TaskResult,
)
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestCrewAIBridgeConfig:
    def test_defaults(self):
        cfg = CrewAIBridgeConfig()
        assert cfg.crew_id == "crewai-crew"
        assert cfg.proxy_sigmoid_k > 0

    def test_empty_crew_id_raises(self):
        with pytest.raises(ValidationError):
            CrewAIBridgeConfig(crew_id="   ")

    def test_max_delegation_depth_validated(self):
        with pytest.raises(ValidationError):
            CrewAIBridgeConfig(max_delegation_depth=0)


# ---------------------------------------------------------------------------
# Protocol mode tests
# ---------------------------------------------------------------------------


class TestCrewAIBridgeProtocolMode:
    def test_record_task_result_returns_soft_interaction(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        result = TaskResult(output="found results", success=True)
        ix = bridge.record_task_result(result)
        assert isinstance(ix, SoftInteraction)

    def test_p_invariant_on_success(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        ix = bridge.record_task_result(TaskResult(success=True))
        assert 0.0 <= ix.p <= 1.0

    def test_p_invariant_on_failure(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        ix = bridge.record_task_result(TaskResult(success=False))
        assert 0.0 <= ix.p <= 1.0

    def test_success_gives_higher_p_than_failure(self):
        cfg = CrewAIBridgeConfig(enable_event_log=False)
        bridge_ok = CrewAIBridge(config=cfg)
        bridge_fail = CrewAIBridge(config=cfg)
        p_ok = bridge_ok.record_task_result(TaskResult(success=True, output="x" * 200)).p
        p_fail = bridge_fail.record_task_result(TaskResult(success=False)).p
        assert p_ok > p_fail

    def test_deep_delegation_lowers_p(self):
        cfg = CrewAIBridgeConfig(enable_event_log=False)
        bridge_deep = CrewAIBridge(config=cfg)
        bridge_flat = CrewAIBridge(config=cfg)
        ix_deep = bridge_deep.record_task_result(
            TaskResult(success=True, delegation_depth=5, output="x" * 200)
        )
        ix_flat = bridge_flat.record_task_result(
            TaskResult(success=True, delegation_depth=0, output="x" * 200)
        )
        assert ix_flat.p >= ix_deep.p

    def test_quality_score_below_half_adds_verifier_rejection(self):
        cfg = CrewAIBridgeConfig(enable_event_log=False)
        bridge_bad = CrewAIBridge(config=cfg)
        bridge_good = CrewAIBridge(config=cfg)
        ix_bad = bridge_bad.record_task_result(
            TaskResult(success=True, quality_score=0.1)
        )
        ix_good = bridge_good.record_task_result(
            TaskResult(success=True, quality_score=0.9)
        )
        assert ix_good.p >= ix_bad.p

    def test_interactions_accumulate(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        bridge.record_task_result(TaskResult())
        bridge.record_task_result(TaskResult())
        assert len(bridge.get_interactions()) == 2

    def test_get_interactions_returns_copy(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        bridge.record_task_result(TaskResult())
        copy = bridge.get_interactions()
        copy.clear()
        assert len(bridge.get_interactions()) == 1

    def test_initiator_is_crew_id(self):
        cfg = CrewAIBridgeConfig(crew_id="safety-crew", enable_event_log=False)
        bridge = CrewAIBridge(config=cfg)
        ix = bridge.record_task_result(TaskResult())
        assert ix.initiator == "safety-crew"

    def test_counterparty_is_agent_role(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        ix = bridge.record_task_result(TaskResult(agent_role="Researcher"))
        assert ix.counterparty == "Researcher"

    def test_toxicity_rate_zero_when_empty(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        assert bridge.get_toxicity_rate() == 0.0


# ---------------------------------------------------------------------------
# Full crew mode — requires crewai
# ---------------------------------------------------------------------------


try:
    importlib.import_module("crewai")
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False


@pytest.mark.skipif(not HAS_CREWAI, reason="crewai not installed")
class TestCrewAIBridgeFullMode:
    def test_run_without_crew_raises(self):
        bridge = CrewAIBridge(config=CrewAIBridgeConfig(enable_event_log=False))
        with pytest.raises(CrewAIBridgeError, match="No crew provided"):
            bridge.run()

    def test_run_with_mock_crew_returns_interactions(self):
        mock_output = MagicMock()
        mock_output.tasks_output = []
        mock_output.raw = "Crew completed all tasks successfully."

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output

        bridge = CrewAIBridge(
            crew=mock_crew,
            config=CrewAIBridgeConfig(enable_event_log=False),
        )
        interactions = bridge.run()
        assert len(interactions) == 1
        assert 0.0 <= interactions[0].p <= 1.0

    def test_run_with_tasks_output_produces_one_interaction_per_task(self):
        task1 = MagicMock()
        task1.raw_output = "Research complete"
        task1.agent = "Researcher"
        task1.description = "Research AI safety"

        task2 = MagicMock()
        task2.raw_output = "Report written"
        task2.agent = "Writer"
        task2.description = "Write report"

        mock_output = MagicMock()
        mock_output.tasks_output = [task1, task2]

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output

        bridge = CrewAIBridge(
            crew=mock_crew,
            config=CrewAIBridgeConfig(enable_event_log=False),
        )
        interactions = bridge.run()
        assert len(interactions) == 2
        assert all(0.0 <= ix.p <= 1.0 for ix in interactions)

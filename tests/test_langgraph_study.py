"""Tests for the LangGraph governed handoff study.

All tests are pure-Python — no LLM calls, no langgraph imports required.
Tests validate: metric extraction, param grid generation, YAML parsing,
task completion detection, governance policy construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.langgraph_governed_study import (
    build_param_grid,
    detect_task_completed,
    load_scenario,
)
from swarm.bridges.langgraph_swarm.governed_swarm import (
    CompositePolicy,
    CycleDetectionPolicy,
    GovernanceDecision,
    InformationBoundaryPolicy,
    ProvenanceLogger,
    ProvenanceRecord,
    RateLimitPolicy,
    analyze_swarm_run,
)
from swarm.bridges.langgraph_swarm.study_agents import (
    AGENT_DEFS,
    TRUST_GROUPS,
    build_chat_model,
    build_governance_policy,
)

# =============================================================================
# Scenario YAML tests
# =============================================================================


class TestScenarioYAML:
    """Test that the scenario YAML is valid and well-formed."""

    @pytest.fixture
    def scenario(self) -> dict:
        path = Path("scenarios/langgraph_governed_handoff.yaml")
        if not path.exists():
            pytest.skip("Scenario YAML not found")
        return load_scenario(path)

    def test_scenario_loads(self, scenario: dict) -> None:
        assert scenario["scenario_id"] == "langgraph_governed_handoff"

    def test_scenario_has_agents(self, scenario: dict) -> None:
        agents = scenario["agents"]
        assert len(agents) == 4
        names = {a["name"] for a in agents}
        assert names == {"coordinator", "researcher", "writer", "reviewer"}

    def test_scenario_has_sweep(self, scenario: dict) -> None:
        sweep = scenario["sweep"]
        assert "max_cycles" in sweep
        assert "max_handoffs" in sweep
        assert "trust_boundaries" in sweep
        assert "seeds_per_config" in sweep

    def test_scenario_has_task_prompt(self, scenario: dict) -> None:
        assert "task_prompt" in scenario
        assert len(scenario["task_prompt"]) > 20

    def test_scenario_has_llm_config(self, scenario: dict) -> None:
        llm = scenario["llm"]
        assert "model" in llm
        assert "max_tokens" in llm

    def test_scenario_has_provider(self, scenario: dict) -> None:
        llm = scenario["llm"]
        assert llm.get("provider") in ("anthropic", "ollama", "openai")

    def test_scenario_has_outputs(self, scenario: dict) -> None:
        outputs = scenario["outputs"]
        assert "csv" in outputs
        assert "provenance" in outputs


# =============================================================================
# Parameter grid tests
# =============================================================================


class TestParamGrid:
    """Test parameter grid generation."""

    def test_grid_size(self) -> None:
        scenario = {
            "sweep": {
                "max_cycles": [1, 2, 3, 5],
                "max_handoffs": [5, 10, 15, 30],
                "trust_boundaries": [True, False],
            }
        }
        grid = build_param_grid(scenario)
        assert len(grid) == 4 * 4 * 2  # 32

    def test_grid_contains_expected_keys(self) -> None:
        scenario = {
            "sweep": {
                "max_cycles": [1],
                "max_handoffs": [5],
                "trust_boundaries": [True],
            }
        }
        grid = build_param_grid(scenario)
        assert len(grid) == 1
        assert grid[0] == {
            "max_cycles": 1,
            "max_handoffs": 5,
            "trust_boundaries": True,
        }

    def test_grid_covers_all_combos(self) -> None:
        scenario = {
            "sweep": {
                "max_cycles": [1, 2],
                "max_handoffs": [5, 10],
                "trust_boundaries": [True, False],
            }
        }
        grid = build_param_grid(scenario)
        assert len(grid) == 8
        # Check a specific combination exists
        assert {"max_cycles": 2, "max_handoffs": 10, "trust_boundaries": False} in grid


# =============================================================================
# Task completion detection tests
# =============================================================================


class TestTaskCompletion:
    """Test task completion detection from message history."""

    def test_detects_final_answer(self) -> None:
        msg1 = MagicMock()
        msg1.content = "Researching..."
        msg1.name = "researcher"
        msg2 = MagicMock()
        msg2.content = "FINAL ANSWER: AI safety is important."
        msg2.name = "coordinator"
        completed, agent = detect_task_completed([msg1, msg2])
        assert completed is True
        assert agent == "coordinator"

    def test_detects_no_completion(self) -> None:
        msg1 = MagicMock()
        msg1.content = "Still working on it."
        msg1.name = "researcher"
        msg2 = MagicMock()
        msg2.content = "Drafting summary."
        msg2.name = "writer"
        completed, agent = detect_task_completed([msg1, msg2])
        assert completed is False
        assert agent == "none"

    def test_handles_empty_messages(self) -> None:
        completed, agent = detect_task_completed([])
        assert completed is False
        assert agent == "none"

    def test_handles_dict_messages(self) -> None:
        messages = [
            {"content": "Working...", "name": "writer"},
            {"content": "FINAL ANSWER: Done.", "name": "coordinator"},
        ]
        completed, agent = detect_task_completed(messages)
        assert completed is True

    def test_finds_last_final_answer(self) -> None:
        """If multiple FINAL ANSWERs exist, picks the last one."""
        msg1 = MagicMock()
        msg1.content = "FINAL ANSWER: Draft 1."
        msg1.name = "writer"
        msg2 = MagicMock()
        msg2.content = "FINAL ANSWER: Draft 2."
        msg2.name = "coordinator"
        completed, agent = detect_task_completed([msg1, msg2])
        assert completed is True
        assert agent == "coordinator"


# =============================================================================
# Metric extraction tests (via analyze_swarm_run with mock provenance)
# =============================================================================


class TestMetricExtraction:
    """Test analyze_swarm_run with synthetic provenance data."""

    def _make_record(self, src: str, tgt: str, decision: str = "approved", risk: float = 0.0) -> ProvenanceRecord:
        return ProvenanceRecord(
            source_agent=src,
            target_agent=tgt,
            governance_decision=decision,
            risk_score_at_handoff=risk,
        )

    def test_empty_provenance(self) -> None:
        logger = ProvenanceLogger()
        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 0
        assert analysis["risk_level"] == "low"

    def test_all_approved(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer"))
        logger.log(self._make_record("writer", "reviewer"))
        logger.log(self._make_record("reviewer", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 4
        assert analysis["approved_handoffs"] == 4
        assert analysis["denied_handoffs"] == 0
        assert analysis["denial_rate"] == 0.0
        assert analysis["risk_level"] == "low"

    def test_with_denials(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer", "denied", 0.8))
        logger.log(self._make_record("researcher", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 3
        assert analysis["denied_handoffs"] == 1
        assert analysis["denial_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_with_escalation(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer", "escalated", 0.9))

        analysis = analyze_swarm_run(logger)
        assert analysis["escalated_handoffs"] == 1
        assert analysis["risk_level"] in ("high", "critical")

    def test_cycle_detection(self) -> None:
        logger = ProvenanceLogger()
        # Create a writer <-> reviewer ping-pong
        for _ in range(4):
            logger.log(self._make_record("writer", "reviewer"))
            logger.log(self._make_record("reviewer", "writer"))

        analysis = analyze_swarm_run(logger)
        assert len(analysis["cycle_pairs"]) > 0

    def test_chain_depth(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer"))
        logger.log(self._make_record("writer", "reviewer"))
        logger.log(self._make_record("reviewer", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["max_chain_depth"] == 3  # 0-indexed, 4th record is depth 3


# =============================================================================
# Governance policy construction tests
# =============================================================================


class TestGovernancePolicies:
    """Test governance policy construction from sweep parameters."""

    def test_build_with_trust_boundaries(self) -> None:
        policy = build_governance_policy(
            max_cycles=3,
            max_handoffs=20,
            trust_boundaries=True,
        )
        assert isinstance(policy, CompositePolicy)
        assert len(policy.policies) == 3
        assert isinstance(policy.policies[0], CycleDetectionPolicy)
        assert isinstance(policy.policies[1], RateLimitPolicy)
        assert isinstance(policy.policies[2], InformationBoundaryPolicy)

    def test_build_without_trust_boundaries(self) -> None:
        policy = build_governance_policy(
            max_cycles=2,
            max_handoffs=10,
            trust_boundaries=False,
        )
        assert isinstance(policy, CompositePolicy)
        assert len(policy.policies) == 2
        # No InformationBoundaryPolicy
        assert all(
            not isinstance(p, InformationBoundaryPolicy) for p in policy.policies
        )

    def test_cycle_policy_respects_max(self) -> None:
        policy = CycleDetectionPolicy(max_cycles=2, window=10)
        logger = ProvenanceLogger()

        # First two handoffs: OK
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE

        # Third handoff in same pair: denied
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.DENY

    def test_information_boundary_cross_group(self) -> None:
        policy = InformationBoundaryPolicy(trust_groups=TRUST_GROUPS)
        logger = ProvenanceLogger()

        # researcher (research) -> writer (content) = cross-boundary
        result = policy.evaluate("researcher", "writer", "task", {}, logger)
        assert result.decision == GovernanceDecision.MODIFY

    def test_information_boundary_same_group(self) -> None:
        policy = InformationBoundaryPolicy(trust_groups=TRUST_GROUPS)
        logger = ProvenanceLogger()

        # researcher (research) -> reviewer (research) = same group
        result = policy.evaluate("researcher", "reviewer", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE

    def test_composite_deny_short_circuits(self) -> None:
        """DENY from any policy should short-circuit the composite."""
        cycle_policy = CycleDetectionPolicy(max_cycles=1, window=10)
        info_policy = InformationBoundaryPolicy(trust_groups=TRUST_GROUPS)
        composite = CompositePolicy([cycle_policy, info_policy])

        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        result = composite.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.DENY


# =============================================================================
# Agent definition tests
# =============================================================================


class TestAgentDefinitions:
    """Test agent definitions are consistent."""

    def test_four_agents_defined(self) -> None:
        assert len(AGENT_DEFS) == 4

    def test_agent_names(self) -> None:
        names = {d["name"] for d in AGENT_DEFS}
        assert names == {"coordinator", "researcher", "writer", "reviewer"}

    def test_trust_groups_match(self) -> None:
        assert TRUST_GROUPS == {
            "coordinator": "management",
            "researcher": "research",
            "writer": "content",
            "reviewer": "research",
        }

    def test_handoff_targets_valid(self) -> None:
        """All handoff targets reference existing agent names."""
        valid_names = {d["name"] for d in AGENT_DEFS}
        for defn in AGENT_DEFS:
            for target in defn["hands_off_to"]:
                assert target in valid_names, (
                    f"{defn['name']} hands off to unknown agent: {target}"
                )

    def test_coordinator_can_reach_all(self) -> None:
        """Coordinator should be able to reach researcher and writer."""
        coord = next(d for d in AGENT_DEFS if d["name"] == "coordinator")
        assert "researcher" in coord["hands_off_to"]
        assert "writer" in coord["hands_off_to"]

    def test_natural_workflow_path_exists(self) -> None:
        """Verify the expected workflow path: coordinator → researcher → writer → reviewer → coordinator."""
        agent_map = {d["name"]: d for d in AGENT_DEFS}
        path = ["coordinator", "researcher", "writer", "reviewer", "coordinator"]
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            assert tgt in agent_map[src]["hands_off_to"], (
                f"Missing handoff: {src} -> {tgt}"
            )


# =============================================================================
# Provenance export tests
# =============================================================================


class TestProvenanceExport:
    """Test provenance record serialization."""

    def test_record_to_dict(self) -> None:
        record = ProvenanceRecord(
            source_agent="coordinator",
            target_agent="researcher",
            governance_decision="approved",
        )
        d = record.to_dict()
        assert d["source_agent"] == "coordinator"
        assert d["target_agent"] == "researcher"
        assert d["governance_decision"] == "approved"

    def test_record_to_json(self) -> None:
        record = ProvenanceRecord(
            source_agent="writer",
            target_agent="reviewer",
        )
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_agent"] == "writer"

    def test_audit_log_export(self) -> None:
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        logger.log(ProvenanceRecord(source_agent="b", target_agent="c"))

        audit = logger.to_audit_log()
        assert len(audit) == 2
        assert all(isinstance(entry, dict) for entry in audit)

    def test_audit_log_is_jsonl_compatible(self) -> None:
        """Each entry should be independently JSON-serializable."""
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="x", target_agent="y"))

        for entry in logger.to_audit_log():
            line = json.dumps(entry)
            assert json.loads(line) == entry


# =============================================================================
# build_chat_model factory tests
# =============================================================================


class TestBuildChatModel:
    """Test the multi-provider chat model factory."""

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            build_chat_model(provider="nonexistent")

    def test_anthropic_returns_chat_anthropic(self) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            pytest.skip("langchain-anthropic not installed")
        llm = build_chat_model(provider="anthropic", model="claude-sonnet-4-20250514")
        assert isinstance(llm, ChatAnthropic)

    def test_ollama_returns_chat_ollama(self) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            pytest.skip("langchain-ollama not installed")
        llm = build_chat_model(provider="ollama", model="llama3.2")
        assert isinstance(llm, ChatOllama)

    def test_ollama_custom_base_url(self) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            pytest.skip("langchain-ollama not installed")
        llm = build_chat_model(
            provider="ollama",
            model="llama3.2",
            base_url="http://myhost:11434",
        )
        assert isinstance(llm, ChatOllama)

    def test_openai_returns_chat_openai(self) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed")
        llm = build_chat_model(provider="openai", model="gpt-4o-mini")
        assert isinstance(llm, ChatOpenAI)

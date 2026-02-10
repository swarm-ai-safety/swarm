"""Tests for scholar/literature synthesis domain."""

import pytest

from swarm.agents.base import ActionType, Observation
from swarm.agents.scholar_agent import (
    AdversarialRetrieverAgent,
    RetrieverAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.scholar_handler import ScholarConfig, ScholarHandler
from swarm.metrics.scholar_metrics import (
    adversary_success_rate,
    citation_precision,
    hallucination_rate,
    scholar_metrics_summary,
)
from swarm.models.interaction import SoftInteraction
from swarm.models.scholar import (
    Citation,
    Passage,
    ScholarQuery,
)


class TestPassage:
    """Tests for Passage data model."""

    def test_passage_creation(self):
        p = Passage(
            passage_id="p1",
            paper_id="paper1",
            text="Test passage text",
            relevance_score=0.8,
            keywords=["test", "passage"],
        )
        assert p.passage_id == "p1"
        assert p.paper_id == "paper1"
        assert p.relevance_score == 0.8
        assert "test" in p.keywords

    def test_passage_to_dict(self):
        p = Passage(passage_id="p1", text="Test")
        d = p.to_dict()
        assert d["passage_id"] == "p1"
        assert d["text"] == "Test"

    def test_passage_from_dict(self):
        d = {
            "passage_id": "p1",
            "paper_id": "paper1",
            "text": "Test",
            "relevance_score": 0.5,
            "keywords": ["a"],
        }
        p = Passage.from_dict(d)
        assert p.passage_id == "p1"
        assert p.paper_id == "paper1"


class TestCitation:
    """Tests for Citation data model."""

    def test_citation_creation(self):
        c = Citation(
            citation_id="c1",
            claim_text="This is a claim",
            paper_id="paper1",
            passage_id="p1",
            entailment_score=0.9,
        )
        assert c.citation_id == "c1"
        assert c.entailment_score == 0.9
        assert c.is_adversarial is False

    def test_citation_adversarial_flag(self):
        c = Citation(is_adversarial=True)
        assert c.is_adversarial is True

    def test_citation_verified_flag(self):
        c = Citation()
        assert c.verified is False
        c.verified = True
        assert c.verified is True


class TestScholarQuery:
    """Tests for ScholarQuery data model."""

    def test_query_creation(self):
        q = ScholarQuery(
            query_text="What is deep learning?",
            domain="cs",
            gold_key_points=["neural networks", "backpropagation"],
        )
        assert q.query_text == "What is deep learning?"
        assert len(q.gold_key_points) == 2

    def test_query_serialization(self):
        q = ScholarQuery(query_text="Test query")
        d = q.to_dict()
        q2 = ScholarQuery.from_dict(d)
        assert q2.query_text == q.query_text


class TestScholarConfig:
    """Tests for ScholarConfig."""

    def test_default_config(self):
        config = ScholarConfig()
        assert config.enabled is True
        assert config.top_k == 30
        assert config.entailment_threshold == 0.7

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            ScholarConfig(top_k=0)

    def test_invalid_entailment_threshold(self):
        with pytest.raises(ValueError):
            ScholarConfig(entailment_threshold=1.5)


class TestScholarHandler:
    """Tests for ScholarHandler."""

    @pytest.fixture
    def handler(self):
        events = []
        config = ScholarConfig(seed=42)
        return ScholarHandler(config, emit_event=lambda e: events.append(e))

    def test_mini_corpus_generated(self, handler):
        assert len(handler._corpus) > 0
        assert len(handler._papers) > 0

    def test_sample_queries_generated(self, handler):
        assert len(handler._queries) > 0

    def test_get_current_query(self, handler):
        query = handler.get_current_query()
        assert query is not None
        assert query.query_text != ""

    def test_observation_fields(self, handler):
        from swarm.env.state import EnvState

        state = EnvState()
        fields = handler.build_observation_fields("agent1", state)

        assert "scholar_query" in fields
        assert "scholar_passage_pool" in fields
        assert "scholar_draft_citations" in fields


class TestRetrieverAgent:
    """Tests for RetrieverAgent."""

    def test_agent_creation(self):
        agent = RetrieverAgent("r1")
        assert agent.agent_id == "r1"
        assert agent.bias == "recall"

    def test_agent_with_precision_bias(self):
        agent = RetrieverAgent("r1", bias="precision")
        assert agent.bias == "precision"

    def test_act_with_no_query(self):
        agent = RetrieverAgent("r1")
        obs = Observation()
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_act_with_query(self):
        agent = RetrieverAgent("r1")
        obs = Observation()
        obs.scholar_query = {"query_text": "What is attention?"}
        action = agent.act(obs)
        assert action.action_type == ActionType.RETRIEVE_PASSAGES


class TestSynthesizerAgent:
    """Tests for SynthesizerAgent."""

    def test_agent_creation(self):
        agent = SynthesizerAgent("s1")
        assert agent.agent_id == "s1"

    def test_act_without_passages(self):
        agent = SynthesizerAgent("s1")
        obs = Observation()
        obs.scholar_query = {"query_text": "Test"}
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_act_with_passages(self):
        agent = SynthesizerAgent("s1")
        obs = Observation()
        obs.scholar_query = {
            "query_text": "Test",
            "gold_key_points": ["point1"],
        }
        obs.scholar_passage_pool = [
            {
                "passage_id": "p1",
                "paper_id": "paper1",
                "text": "Test passage about point1",
                "relevance_score": 0.8,
            }
        ]
        action = agent.act(obs)
        assert action.action_type == ActionType.SYNTHESIZE_ANSWER


class TestVerifierAgent:
    """Tests for VerifierAgent."""

    def test_agent_creation(self):
        agent = VerifierAgent("v1")
        assert agent.agent_id == "v1"
        assert agent.accuracy == 0.9

    def test_act_without_citation(self):
        agent = VerifierAgent("v1")
        obs = Observation()
        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_act_with_citation(self):
        agent = VerifierAgent("v1")
        obs = Observation()
        obs.scholar_citation_to_verify = {
            "citation_id": "c1",
            "claim_text": "Test claim",
            "passage_id": "p1",
        }
        obs.scholar_passage_pool = [
            {"passage_id": "p1", "text": "Test claim content"}
        ]
        action = agent.act(obs)
        assert action.action_type == ActionType.VERIFY_CITATION


class TestAdversarialRetrieverAgent:
    """Tests for AdversarialRetrieverAgent."""

    def test_agent_creation(self):
        agent = AdversarialRetrieverAgent("a1")
        assert agent.agent_id == "a1"
        assert agent.attack_rate == 0.3

    def test_adversarial_synthesis(self):
        agent = AdversarialRetrieverAgent("a1", attack_rate=1.0)
        obs = Observation()
        obs.scholar_query = {"query_text": "Test"}
        obs.scholar_passage_pool = [
            {
                "passage_id": "p1",
                "paper_id": "paper1",
                "text": "Original passage text",
                "relevance_score": 0.8,
            }
        ]
        action = agent.act(obs)
        # Should synthesize with adversarial citations
        if action.action_type == ActionType.SYNTHESIZE_ANSWER:
            citations = action.metadata.get("citations", [])
            # At least one should be adversarial with 100% attack rate
            adversarial_count = sum(1 for c in citations if c.get("is_adversarial"))
            assert adversarial_count > 0


class TestScholarMetrics:
    """Tests for scholar metrics."""

    def test_citation_precision_no_citations(self):
        interactions = []
        assert citation_precision(interactions) == 1.0

    def test_citation_precision_all_valid(self):
        interactions = [
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": True}
            ),
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": True}
            ),
        ]
        assert citation_precision(interactions) == 1.0

    def test_citation_precision_half_valid(self):
        interactions = [
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": True}
            ),
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": False}
            ),
        ]
        assert citation_precision(interactions) == 0.5

    def test_adversary_success_rate_no_adversaries(self):
        interactions = [
            SoftInteraction(
                metadata={
                    "scholar": True,
                    "action": "verify",
                    "is_adversarial": False,
                }
            )
        ]
        assert adversary_success_rate(interactions) == 0.0

    def test_hallucination_rate(self):
        interactions = [
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": False}
            ),
            SoftInteraction(
                metadata={"scholar": True, "action": "verify", "actual_valid": True}
            ),
        ]
        assert hallucination_rate(interactions) == 0.5

    def test_scholar_metrics_summary(self):
        interactions = []
        summary = scholar_metrics_summary(interactions)
        assert "citation_precision" in summary
        assert "adversary_success_rate" in summary
        assert "hallucination_rate" in summary


class TestScholarIntegration:
    """Integration tests for scholar domain."""

    def test_orchestrator_with_scholar_agents(self):
        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=3,
            scholar_config=ScholarConfig(seed=42),
            seed=42,
        )
        orch = Orchestrator(config=config)

        orch.register_agent(RetrieverAgent("r1"))
        orch.register_agent(SynthesizerAgent("s1"))
        orch.register_agent(VerifierAgent("v1"))

        metrics = orch.run()
        assert len(metrics) == 2
        assert orch._scholar_handler is not None

    def test_orchestrator_with_adversarial_agents(self):
        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=5,
            scholar_config=ScholarConfig(seed=42),
            seed=42,
        )
        orch = Orchestrator(config=config)

        orch.register_agent(RetrieverAgent("r1"))
        orch.register_agent(AdversarialRetrieverAgent("a1", attack_rate=0.5))
        orch.register_agent(VerifierAgent("v1"))

        metrics = orch.run()
        assert len(metrics) == 1

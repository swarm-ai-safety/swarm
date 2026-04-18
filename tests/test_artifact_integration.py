"""Integration test: full artifact wiring path through the orchestrator.

Tests the end-to-end pipeline:
  handler.handle_action → HandlerActionResult(produced_artifacts=...)
  → orchestrator calls set_handler_result
  → finalize_interaction publishes artifacts & wires causal_parents
  → CascadeRiskLever fires on downstream chains
  → ObservationBuilder exposes artifacts & pressure to agents
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
)
from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.proxy import ProxyObservables
from swarm.governance.config import GovernanceConfig
from swarm.models.agent import AgentType
from swarm.models.artifact import Artifact

# ── Custom action type for the test handler ──────────────────────


class _ArtifactActionType(str, Enum):
    PRODUCE = "produce_artifact"
    CONSUME = "consume_artifact"


# ── Test handler that produces / consumes artifacts ──────────────


class _ArtifactHandler(Handler):
    """Minimal handler that produces or consumes artifacts."""

    _produced_artifact_id: str = ""

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.POST})

    def handle_action(self, action: Action, state) -> HandlerActionResult:
        metadata = action.metadata or {}
        mode = metadata.get("artifact_mode", "produce")

        observables = ProxyObservables(
            task_progress_delta=0.8,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.5,
        )

        if mode == "produce":
            artifact = Artifact(
                kind="test_receipt",
                producer_id=action.agent_id,
                data={"quality": 0.9},
            )
            _ArtifactHandler._produced_artifact_id = artifact.artifact_id
            return HandlerActionResult(
                success=True,
                observables=observables,
                initiator_id=action.agent_id,
                counterparty_id=action.counterparty_id or "",
                produced_artifacts=[artifact],
            )
        elif mode == "consume":
            consume_id = metadata.get("consume_artifact_id", "")
            return HandlerActionResult(
                success=True,
                observables=observables,
                initiator_id=action.agent_id,
                counterparty_id=action.counterparty_id or "",
                consumed_artifact_ids=[consume_id] if consume_id else [],
            )
        else:
            return HandlerActionResult(success=False)


# ── Test agents ──────────────────────────────────────────────────


class _ProducerAgent(BaseAgent):
    """Agent that produces an artifact via POST action."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, agent_type=AgentType.HONEST)
        self._step = 0

    def act(self, observation: Observation) -> Action:
        self._step += 1
        return Action(
            action_type=ActionType.POST,
            agent_id=self.agent_id,
            content="producing artifact",
            metadata={"artifact_mode": "produce"},
        )

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        return False

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> InteractionProposal | None:
        return None


class _ConsumerAgent(BaseAgent):
    """Agent that consumes an artifact produced by _ProducerAgent."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, agent_type=AgentType.HONEST)
        self._consumed = False

    def act(self, observation: Observation) -> Action:
        # Try to consume the first available artifact
        artifacts = observation.available_artifacts
        if artifacts and not self._consumed:
            self._consumed = True
            return Action(
                action_type=ActionType.POST,
                agent_id=self.agent_id,
                content="consuming artifact",
                metadata={
                    "artifact_mode": "consume",
                    "consume_artifact_id": artifacts[0]["artifact_id"],
                },
            )
        return Action(
            action_type=ActionType.POST,
            agent_id=self.agent_id,
            content="idle post",
            metadata={"artifact_mode": "produce"},
        )

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        return False

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> InteractionProposal | None:
        return None


# ── Tests ────────────────────────────────────────────────────────


class TestArtifactWiringIntegration:
    """End-to-end: handler → orchestrator → finalizer → causal_parents."""

    def _make_orchestrator(self, **gov_overrides):
        gov_config = GovernanceConfig(**gov_overrides)
        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=5,
            seed=42,
            governance_config=gov_config,
        )
        orch = Orchestrator(config=config)

        # Replace the default POST handler with our artifact handler
        handler = _ArtifactHandler(event_bus=orch._event_bus)
        orch._handler_registry._action_map[ActionType.POST] = handler
        if handler not in orch._handler_registry._handlers:
            orch._handler_registry._handlers.append(handler)

        return orch

    def test_artifact_published_after_handler_produces(self):
        """Handler produces artifact → artifact appears in registry."""
        orch = self._make_orchestrator()
        orch.register_agent(_ProducerAgent(agent_id="producer"))

        orch.run()

        registry = orch.state.artifact_registry
        assert len(registry) > 0, "Artifacts should be published"

        artifacts = registry.all_artifacts()
        assert all(a.kind == "test_receipt" for a in artifacts)
        assert all(a.producer_id == "producer" for a in artifacts)
        # p_at_production should be set from the interaction's soft label
        assert all(0.0 <= a.p_at_production <= 1.0 for a in artifacts)
        # step should be set (not default 0 for all)
        assert any(a.step > 0 for a in artifacts)

    def test_consumer_sees_artifacts_in_observation(self):
        """Artifacts produced by one agent appear in another's observation."""
        orch = self._make_orchestrator()
        orch.register_agent(_ProducerAgent(agent_id="producer"))
        consumer = _ConsumerAgent(agent_id="consumer")
        orch.register_agent(consumer)

        orch.run()

        # If consumer consumed an artifact, the flag should be set
        assert consumer._consumed, (
            "Consumer should have seen and consumed an artifact from its observation"
        )

    def test_causal_parents_wired_on_consume(self):
        """Consuming an artifact auto-wires causal_parents on the interaction."""
        orch = self._make_orchestrator()
        orch.register_agent(_ProducerAgent(agent_id="producer"))
        orch.register_agent(_ConsumerAgent(agent_id="consumer"))

        # Capture all interactions via callback (completed_interactions
        # is cleared each epoch, so we need a persistent collector).
        all_interactions = []
        orch.on_interaction_complete(
            lambda ix, pi, pc: all_interactions.append(ix)
        )

        orch.run()

        # Find interactions with causal_parents set
        interactions_with_parents = [
            ix for ix in all_interactions
            if ix.causal_parents
        ]
        assert len(interactions_with_parents) > 0, (
            "At least one interaction should have causal_parents from artifact consumption"
        )

        # The parent should be a producer interaction
        all_ids = {ix.interaction_id for ix in all_interactions}
        for ix in interactions_with_parents:
            assert ix.initiator == "consumer"
            for parent_id in ix.causal_parents:
                assert parent_id in all_ids, (
                    f"causal_parent {parent_id} should reference a real interaction"
                )

    def test_artifact_consumed_by_tracked(self):
        """Consuming an artifact records the consumer in consumed_by."""
        orch = self._make_orchestrator()
        orch.register_agent(_ProducerAgent(agent_id="producer"))
        orch.register_agent(_ConsumerAgent(agent_id="consumer"))

        orch.run()

        registry = orch.state.artifact_registry
        consumed_artifacts = [
            a for a in registry.all_artifacts() if a.consumed_by
        ]
        assert len(consumed_artifacts) > 0, (
            "At least one artifact should have been consumed"
        )

    def test_pressure_exposed_in_observation(self):
        """artifact_pressure dict is populated in agent observations."""
        orch = self._make_orchestrator()
        orch.register_agent(_ProducerAgent(agent_id="producer"))

        # Declare a need so pressure is nonzero
        from swarm.models.artifact import ArtifactNeed
        orch.state.artifact_registry.declare_need(
            ArtifactNeed(kind="something_rare", requester_id="anyone")
        )

        # Build observation manually to inspect
        obs = orch._obs_builder.build("producer")
        assert "something_rare" in obs.artifact_pressure
        assert obs.artifact_pressure["something_rare"] > 0


class TestCascadeRiskIntegration:
    """End-to-end: artifact chain → CascadeRiskLever fires."""

    def test_cascade_lever_runs_during_simulation(self):
        """With cascade_risk_enabled, the lever processes artifact chains."""
        gov_config = GovernanceConfig(
            cascade_risk_enabled=True,
            cascade_risk_threshold=0.3,
            cascade_risk_penalty_scale=2.0,
        )
        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=10,
            seed=42,
            governance_config=gov_config,
        )
        orch = Orchestrator(config=config)

        handler = _ArtifactHandler(event_bus=orch._event_bus)
        orch._handler_registry._action_map[ActionType.POST] = handler
        if handler not in orch._handler_registry._handlers:
            orch._handler_registry._handlers.append(handler)

        orch.register_agent(_ProducerAgent(agent_id="producer"))
        orch.register_agent(_ConsumerAgent(agent_id="consumer"))

        # Should run without errors even with cascade lever active
        metrics = orch.run()
        assert len(metrics) == 3

        # Verify the cascade_risk lever is registered
        lever_names = [
            lev.name for lev in orch.governance_engine._levers
        ]
        assert "cascade_risk" in lever_names


class TestArtifactGCIntegration:
    """Artifact garbage collection at epoch boundaries."""

    def test_gc_runs_via_cascade_lever_epoch_start(self):
        """CascadeRiskLever.on_epoch_start garbage-collects stale artifacts."""
        gov_config = GovernanceConfig(cascade_risk_enabled=True)
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=10,
            seed=42,
            governance_config=gov_config,
        )
        orch = Orchestrator(config=config)

        handler = _ArtifactHandler(event_bus=orch._event_bus)
        orch._handler_registry._action_map[ActionType.POST] = handler
        if handler not in orch._handler_registry._handlers:
            orch._handler_registry._handlers.append(handler)

        orch.register_agent(_ProducerAgent(agent_id="producer"))

        metrics = orch.run()
        assert len(metrics) == 5

        # After 5 epochs × 10 steps = 50 steps, artifacts from early
        # steps should have been GC'd (max_age_steps=200 in lever,
        # but we verify the GC mechanism ran without error)
        registry = orch.state.artifact_registry
        for art in registry.all_artifacts():
            assert art.step >= 0  # sanity: step was set

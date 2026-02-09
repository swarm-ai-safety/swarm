"""Tests for semi-permeable boundaries module."""

from datetime import datetime

from swarm.boundaries.external_world import (
    ExternalDataSource,
    ExternalEntity,
    ExternalEntityType,
    ExternalService,
    ExternalWorld,
)
from swarm.boundaries.information_flow import (
    FlowDirection,
    FlowTracker,
    FlowType,
    InformationFlow,
)
from swarm.boundaries.leakage import (
    LeakageDetector,
    LeakageEvent,
    LeakageType,
)
from swarm.boundaries.policies import (
    CompositePolicy,
    ContentFilterPolicy,
    CrossingDecision,
    PolicyEngine,
    RateLimitPolicy,
    SensitivityPolicy,
)

# =============================================================================
# External World Tests
# =============================================================================


class TestExternalEntity:
    """Tests for ExternalEntity."""

    def test_create_entity(self):
        """Test creating an external entity."""
        entity = ExternalEntity(
            entity_id="test_entity",
            name="Test Service",
            entity_type=ExternalEntityType.SERVICE,
            trust_level=0.8,
        )
        assert entity.entity_id == "test_entity"
        assert entity.entity_type == ExternalEntityType.SERVICE
        assert entity.trust_level == 0.8

    def test_trust_level_clamped(self):
        """Test that trust level is clamped to [0, 1]."""
        entity = ExternalEntity(
            entity_id="test",
            name="Test",
            entity_type=ExternalEntityType.SERVICE,
            trust_level=1.5,
        )
        assert entity.trust_level == 1.0

        entity2 = ExternalEntity(
            entity_id="test2",
            name="Test 2",
            entity_type=ExternalEntityType.SERVICE,
            trust_level=-0.5,
        )
        assert entity2.trust_level == 0.0


class TestExternalService:
    """Tests for ExternalService."""

    def test_create_service(self):
        """Test creating an external service."""
        service = ExternalService(
            entity_id="api_service",
            name="API Service",
            endpoint="https://api.example.com",
            rate_limit=100,
            reliability=0.99,
        )
        assert service.entity_type == ExternalEntityType.SERVICE
        assert service.endpoint == "https://api.example.com"
        assert service.rate_limit == 100

    def test_service_call_success(self):
        """Test successful service call."""
        import random

        service = ExternalService(
            entity_id="test_api",
            name="Test API",
            reliability=1.0,  # Always succeed
        )

        result = service.call({"query": "test"}, rng=random.Random(42))
        assert result["success"] is True
        assert "data" in result

    def test_service_call_failure(self):
        """Test service call failure due to reliability."""
        import random

        service = ExternalService(
            entity_id="unreliable_api",
            name="Unreliable API",
            reliability=0.0,  # Always fail
        )

        result = service.call({"query": "test"}, rng=random.Random(42))
        assert result["success"] is False
        assert "error" in result


class TestExternalDataSource:
    """Tests for ExternalDataSource."""

    def test_create_data_source(self):
        """Test creating a data source."""
        source = ExternalDataSource(
            entity_id="db_source",
            name="Database",
            data_type="structured",
            sensitivity_level=0.5,
        )
        assert source.entity_type == ExternalEntityType.DATA_SOURCE
        assert source.data_type == "structured"
        assert source.sensitivity_level == 0.5

    def test_data_source_query(self):
        """Test querying a data source."""
        source = ExternalDataSource(
            entity_id="test_db",
            name="Test DB",
        )

        result = source.query("SELECT * FROM users")
        assert result["success"] is True
        assert "rows" in result


class TestExternalWorld:
    """Tests for ExternalWorld."""

    def test_add_and_get_entity(self):
        """Test adding and retrieving entities."""
        world = ExternalWorld()
        entity = ExternalEntity(
            entity_id="test",
            name="Test",
            entity_type=ExternalEntityType.SERVICE,
        )

        world.add_entity(entity)
        retrieved = world.get_entity("test")

        assert retrieved is not None
        assert retrieved.entity_id == "test"

    def test_list_entities_by_type(self):
        """Test listing entities by type."""
        world = ExternalWorld()
        world.add_entity(ExternalService(entity_id="service1", name="Service 1"))
        world.add_entity(ExternalDataSource(entity_id="data1", name="Data 1"))

        services = world.list_entities(entity_type=ExternalEntityType.SERVICE)
        assert len(services) == 1
        assert services[0].entity_id == "service1"

    def test_list_entities_by_trust(self):
        """Test listing entities by minimum trust."""
        world = ExternalWorld()
        world.add_entity(
            ExternalEntity(
                entity_id="trusted",
                name="Trusted",
                entity_type=ExternalEntityType.SERVICE,
                trust_level=0.9,
            )
        )
        world.add_entity(
            ExternalEntity(
                entity_id="untrusted",
                name="Untrusted",
                entity_type=ExternalEntityType.SERVICE,
                trust_level=0.2,
            )
        )

        trusted = world.list_entities(min_trust=0.5)
        assert len(trusted) == 1
        assert trusted[0].entity_id == "trusted"

    def test_interact_with_service(self):
        """Test interacting with an external service."""
        world = ExternalWorld()
        world.add_entity(
            ExternalService(
                entity_id="api",
                name="API",
                reliability=1.0,
            )
        )

        result = world.interact(
            agent_id="agent_1",
            entity_id="api",
            action="call",
            payload={"query": "test"},
        )

        assert result["success"] is True
        assert world.total_outbound_calls == 1

    def test_interact_unknown_entity(self):
        """Test interacting with unknown entity."""
        world = ExternalWorld()
        result = world.interact(
            agent_id="agent_1",
            entity_id="unknown",
            action="call",
        )
        assert result["success"] is False
        assert "Unknown entity" in result["error"]

    def test_interaction_stats(self):
        """Test getting interaction statistics."""
        world = ExternalWorld()
        world.add_entity(ExternalService(entity_id="api", name="API", reliability=1.0))

        world.interact("agent_1", "api", "call")
        world.interact("agent_1", "api", "call")

        stats = world.get_interaction_stats()
        assert stats["total_interactions"] == 2
        assert stats["success_rate"] == 1.0

    def test_create_default_world(self):
        """Test creating default world with entities."""
        world = ExternalWorld().create_default_world()

        assert world.get_entity("web_search") is not None
        assert world.get_entity("code_repo") is not None
        assert world.get_entity("public_data") is not None


# =============================================================================
# Information Flow Tests
# =============================================================================


class TestInformationFlow:
    """Tests for InformationFlow."""

    def test_create_flow(self):
        """Test creating an information flow."""
        flow = InformationFlow.create(
            direction=FlowDirection.OUTBOUND,
            flow_type=FlowType.QUERY,
            source_id="agent_1",
            destination_id="api_service",
            content={"query": "test"},
        )

        assert flow.direction == FlowDirection.OUTBOUND
        assert flow.flow_type == FlowType.QUERY
        assert flow.source_id == "agent_1"
        assert flow.size_bytes > 0
        assert len(flow.content_hash) == 16


class TestFlowTracker:
    """Tests for FlowTracker."""

    def test_record_flow(self):
        """Test recording flows."""
        tracker = FlowTracker()

        flow = InformationFlow.create(
            direction=FlowDirection.OUTBOUND,
            flow_type=FlowType.DATA,
            source_id="agent_1",
            destination_id="external",
            content="test data",
        )
        tracker.record_flow(flow)

        assert len(tracker.flows) == 1

    def test_get_flows_filtered(self):
        """Test filtering flows."""
        tracker = FlowTracker()

        tracker.record_flow(
            InformationFlow.create(
                direction=FlowDirection.OUTBOUND,
                flow_type=FlowType.QUERY,
                source_id="agent_1",
                destination_id="api",
                content="query",
            )
        )
        tracker.record_flow(
            InformationFlow.create(
                direction=FlowDirection.INBOUND,
                flow_type=FlowType.RESPONSE,
                source_id="api",
                destination_id="agent_1",
                content="response",
            )
        )

        outbound = tracker.get_flows(direction=FlowDirection.OUTBOUND)
        assert len(outbound) == 1
        assert outbound[0].direction == FlowDirection.OUTBOUND

    def test_get_summary(self):
        """Test generating flow summary."""
        tracker = FlowTracker()

        for i in range(5):
            tracker.record_flow(
                InformationFlow.create(
                    direction=FlowDirection.OUTBOUND,
                    flow_type=FlowType.DATA,
                    source_id=f"agent_{i % 2}",
                    destination_id="external",
                    content=f"data {i}",
                    sensitivity_score=i * 0.1,
                )
            )

        summary = tracker.get_summary()

        assert summary.total_flows == 5
        assert summary.outbound_flows == 5
        assert summary.inbound_flows == 0
        assert len(summary.top_sources) > 0

    def test_get_agent_flows(self):
        """Test getting flows for specific agent."""
        tracker = FlowTracker()

        tracker.record_flow(
            InformationFlow.create(
                direction=FlowDirection.OUTBOUND,
                flow_type=FlowType.QUERY,
                source_id="agent_1",
                destination_id="api",
                content="query",
            )
        )
        tracker.record_flow(
            InformationFlow.create(
                direction=FlowDirection.INBOUND,
                flow_type=FlowType.RESPONSE,
                source_id="api",
                destination_id="agent_1",
                content="response",
            )
        )

        agent_flows = tracker.get_agent_flows("agent_1")

        assert agent_flows["flows_initiated"] == 1
        assert agent_flows["flows_received"] == 1

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        tracker = FlowTracker()

        # Create excessive outbound flow
        for _i in range(10):
            tracker.record_flow(
                InformationFlow.create(
                    direction=FlowDirection.OUTBOUND,
                    flow_type=FlowType.DATA,
                    source_id="agent_1",
                    destination_id="external",
                    content="x" * 10000,  # Large content
                )
            )

        anomalies = tracker.detect_anomalies()
        # Should detect concentrated source
        assert any(a["type"] == "concentrated_source" for a in anomalies)


# =============================================================================
# Policy Tests
# =============================================================================


class TestCrossingDecision:
    """Tests for CrossingDecision."""

    def test_allow_decision(self):
        """Test creating allow decision."""
        decision = CrossingDecision.allow("TestPolicy", "Test passed")
        assert decision.allowed is True
        assert decision.policy_name == "TestPolicy"

    def test_deny_decision(self):
        """Test creating deny decision."""
        decision = CrossingDecision.deny("TestPolicy", "Test failed", risk_score=0.8)
        assert decision.allowed is False
        assert decision.risk_score == 0.8


class TestRateLimitPolicy:
    """Tests for RateLimitPolicy."""

    def test_allow_within_limit(self):
        """Test allowing requests within rate limit."""
        policy = RateLimitPolicy(max_crossings_per_minute=100)

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="query",
            content="test",
            metadata={},
        )

        assert decision.allowed is True

    def test_deny_over_limit(self):
        """Test denying requests over rate limit."""
        policy = RateLimitPolicy(max_crossings_per_minute=5)

        # Make requests up to limit
        for _ in range(5):
            policy.evaluate("agent_1", "outbound", "query", "test", {})

        # This should be denied
        decision = policy.evaluate("agent_1", "outbound", "query", "test", {})

        assert decision.allowed is False
        assert "Rate limit" in decision.reason


class TestContentFilterPolicy:
    """Tests for ContentFilterPolicy."""

    def test_allow_clean_content(self):
        """Test allowing clean content."""
        policy = ContentFilterPolicy(
            blocked_keywords={"malware", "exploit"},
        )

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="Hello, world!",
            metadata={},
        )

        assert decision.allowed is True

    def test_deny_blocked_keyword(self):
        """Test denying content with blocked keyword."""
        policy = ContentFilterPolicy(
            blocked_keywords={"malware", "exploit"},
        )

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="Check out this malware!",
            metadata={},
        )

        assert decision.allowed is False
        assert "blocked keyword" in decision.reason

    def test_deny_blocked_pattern(self):
        """Test denying content matching blocked pattern."""
        policy = ContentFilterPolicy(
            blocked_patterns=[r"password\s*=\s*\S+"],
        )

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="code",
            content="password = secretkey123",
            metadata={},
        )

        assert decision.allowed is False
        assert "blocked pattern" in decision.reason

    def test_deny_oversized_content(self):
        """Test denying oversized content."""
        policy = ContentFilterPolicy(max_content_length=100)

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="data",
            content="x" * 1000,
            metadata={},
        )

        assert decision.allowed is False
        assert "maximum length" in decision.reason


class TestSensitivityPolicy:
    """Tests for SensitivityPolicy."""

    def test_allow_low_sensitivity(self):
        """Test allowing low sensitivity data."""
        policy = SensitivityPolicy(max_outbound_sensitivity=0.5)

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="data",
            content="public data",
            metadata={"sensitivity": 0.2},
        )

        assert decision.allowed is True

    def test_deny_high_sensitivity_outbound(self):
        """Test denying high sensitivity outbound data."""
        policy = SensitivityPolicy(max_outbound_sensitivity=0.5)

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="data",
            content="sensitive data",
            metadata={"sensitivity": 0.8},
        )

        assert decision.allowed is False
        assert "sensitivity" in decision.reason.lower()

    def test_clearance_levels(self):
        """Test agent clearance levels."""
        policy = SensitivityPolicy(
            agent_clearance_levels={"agent_1": 0.9, "agent_2": 0.3},
        )

        # Agent 1 has high clearance
        decision1 = policy.evaluate(
            agent_id="agent_1",
            direction="inbound",
            flow_type="data",
            content="data",
            metadata={"sensitivity": 0.7},
        )
        assert decision1.allowed is True

        # Agent 2 has low clearance
        decision2 = policy.evaluate(
            agent_id="agent_2",
            direction="inbound",
            flow_type="data",
            content="data",
            metadata={"sensitivity": 0.7},
        )
        assert decision2.allowed is False


class TestCompositePolicy:
    """Tests for CompositePolicy."""

    def test_require_all_pass(self):
        """Test composite requiring all policies to pass."""
        policy = CompositePolicy(
            policies=[
                ContentFilterPolicy(blocked_keywords={"blocked"}),
                RateLimitPolicy(max_crossings_per_minute=100),
            ],
            require_all=True,
        )

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="clean content",
            metadata={},
        )

        assert decision.allowed is True

    def test_require_all_fail(self):
        """Test composite failing when one policy fails."""
        policy = CompositePolicy(
            policies=[
                ContentFilterPolicy(blocked_keywords={"blocked"}),
                RateLimitPolicy(max_crossings_per_minute=100),
            ],
            require_all=True,
        )

        decision = policy.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="this is blocked content",
            metadata={},
        )

        assert decision.allowed is False


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_evaluate_all_policies(self):
        """Test evaluating through all policies."""
        engine = PolicyEngine()
        engine.add_policy(RateLimitPolicy(max_crossings_per_minute=100))
        engine.add_policy(ContentFilterPolicy(blocked_keywords={"forbidden"}))

        decision = engine.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="hello",
        )

        assert decision.allowed is True

    def test_short_circuit_on_deny(self):
        """Test that engine stops on first denial."""
        engine = PolicyEngine()
        engine.add_policy(ContentFilterPolicy(blocked_keywords={"stop"}))
        engine.add_policy(RateLimitPolicy(max_crossings_per_minute=100))

        decision = engine.evaluate(
            agent_id="agent_1",
            direction="outbound",
            flow_type="message",
            content="stop here",
        )

        assert decision.allowed is False
        assert decision.policy_name == "ContentFilterPolicy"

    def test_get_statistics(self):
        """Test getting policy statistics."""
        engine = PolicyEngine()
        engine.add_policy(RateLimitPolicy())

        engine.evaluate("agent_1", "outbound", "query", "test", {})
        engine.evaluate("agent_1", "outbound", "query", "test", {})

        stats = engine.get_statistics()

        assert stats["total_policies"] == 1
        assert stats["total_evaluations"] == 2

    def test_create_default_policies(self):
        """Test creating default policy set."""
        engine = PolicyEngine().create_default_policies()

        assert len(engine.policies) == 3  # Rate, Content, Sensitivity


# =============================================================================
# Leakage Detection Tests
# =============================================================================


class TestLeakageEvent:
    """Tests for LeakageEvent."""

    def test_severity_labels(self):
        """Test severity label mapping."""
        critical = LeakageEvent(
            event_id="1",
            timestamp=datetime.now(),
            leakage_type=LeakageType.CREDENTIAL,
            severity=0.95,
            agent_id="agent_1",
            destination_id="external",
        )
        assert critical.severity_label == "critical"

        low = LeakageEvent(
            event_id="2",
            timestamp=datetime.now(),
            leakage_type=LeakageType.UNKNOWN,
            severity=0.2,
            agent_id="agent_1",
            destination_id="external",
        )
        assert low.severity_label == "low"


class TestLeakageDetector:
    """Tests for LeakageDetector."""

    def test_detect_pii_email(self):
        """Test detecting email addresses."""
        detector = LeakageDetector()

        events = detector.scan(
            content="Contact me at user@example.com",
            agent_id="agent_1",
            destination_id="external",
        )

        assert len(events) >= 1
        assert any(e.leakage_type == LeakageType.PII for e in events)

    def test_detect_credential_password(self):
        """Test detecting passwords."""
        detector = LeakageDetector()

        events = detector.scan(
            content="The password = mysecretpass123",
            agent_id="agent_1",
            destination_id="external",
        )

        assert len(events) >= 1
        assert any(e.leakage_type == LeakageType.CREDENTIAL for e in events)

    def test_detect_api_key(self):
        """Test detecting API keys."""
        detector = LeakageDetector()

        events = detector.scan(
            content="api_key = sk-1234567890abcdef",
            agent_id="agent_1",
            destination_id="external",
        )

        assert len(events) >= 1
        assert any(e.leakage_type == LeakageType.CREDENTIAL for e in events)

    def test_detect_code(self):
        """Test detecting code."""
        detector = LeakageDetector()

        events = detector.scan(
            content="def secret_algorithm(x):\n    return x * 42",
            agent_id="agent_1",
            destination_id="external",
        )

        assert len(events) >= 1
        assert any(e.leakage_type == LeakageType.CODE for e in events)

    def test_detect_sensitive_keywords(self):
        """Test detecting sensitive keywords."""
        detector = LeakageDetector()

        events = detector.scan(
            content="This is confidential information",
            agent_id="agent_1",
            destination_id="external",
        )

        # Should detect keyword but may not be highest priority
        assert len(events) >= 1

    def test_no_false_positives(self):
        """Test that benign content doesn't trigger."""
        detector = LeakageDetector()

        events = detector.scan(
            content="Hello, how are you today?",
            agent_id="agent_1",
            destination_id="external",
        )

        assert len(events) == 0

    def test_get_events_filtered(self):
        """Test filtering leakage events."""
        detector = LeakageDetector()

        detector.scan("password = secret", "agent_1", "external")
        detector.scan("email@test.com", "agent_2", "external")

        cred_events = detector.get_events(leakage_type=LeakageType.CREDENTIAL)
        assert len(cred_events) >= 1

        agent1_events = detector.get_events(agent_id="agent_1")
        assert all(e.agent_id == "agent_1" for e in agent1_events)

    def test_generate_report(self):
        """Test report generation."""
        detector = LeakageDetector()

        detector.scan("password = test123", "agent_1", "external")
        detector.scan("user@email.com", "agent_1", "external")
        detector.scan("api_key = abc123", "agent_2", "external")

        report = detector.generate_report()

        assert report.total_events >= 2
        assert len(report.events_by_type) > 0
        assert len(report.top_agents) > 0

    def test_report_recommendations(self):
        """Test that report includes recommendations."""
        detector = LeakageDetector()

        # Trigger credential detection
        detector.scan("password = supersecret", "agent_1", "external")

        report = detector.generate_report()

        # Should recommend credential handling
        assert any("credential" in r.lower() for r in report.recommendations)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBoundaryIntegration:
    """Integration tests for boundary components."""

    def test_orchestrator_boundary_interaction(self):
        """Test orchestrator boundary interaction methods."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            enable_boundaries=True,
            n_epochs=1,
            steps_per_epoch=1,
        )
        orchestrator = Orchestrator(config)

        # Check that boundary components are initialized
        assert orchestrator.external_world is not None
        assert orchestrator.flow_tracker is not None
        assert orchestrator.policy_engine is not None
        assert orchestrator.leakage_detector is not None

    def test_request_external_interaction(self):
        """Test requesting external interaction."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            enable_boundaries=True,
            n_epochs=1,
            steps_per_epoch=1,
            seed=42,
        )
        orchestrator = Orchestrator(config)

        # Request interaction with default entity
        result = orchestrator.request_external_interaction(
            agent_id="agent_1",
            entity_id="web_search",
            action="call",
            payload={"query": "test"},
        )

        assert result.get("success") is True or result.get("blocked") is True

    def test_get_external_entities(self):
        """Test getting external entities."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_boundaries=True)
        orchestrator = Orchestrator(config)

        entities = orchestrator.get_external_entities()

        assert len(entities) > 0
        assert all("entity_id" in e for e in entities)

    def test_get_boundary_metrics(self):
        """Test getting boundary metrics."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_boundaries=True)
        orchestrator = Orchestrator(config)

        # Make some interactions
        orchestrator.request_external_interaction(
            agent_id="agent_1",
            entity_id="web_search",
            action="call",
        )

        metrics = orchestrator.get_boundary_metrics()

        assert metrics["boundaries_enabled"] is True
        assert "external_world" in metrics
        assert "flows" in metrics

    def test_leakage_blocked(self):
        """Test that critical leakage is blocked."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_boundaries=True)
        orchestrator = Orchestrator(config)

        # Try to send data with credential
        result = orchestrator.request_external_interaction(
            agent_id="agent_1",
            entity_id="web_search",
            action="call",
            payload={"data": "password = supersecretpassword123!"},
        )

        # Should be blocked due to credential detection
        assert result.get("blocked") is True or result.get("success") is False

    def test_boundaries_disabled(self):
        """Test that boundaries can be disabled."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_boundaries=False)
        orchestrator = Orchestrator(config)

        assert orchestrator.external_world is None
        assert orchestrator.flow_tracker is None

        # Request should fail gracefully
        result = orchestrator.request_external_interaction(
            agent_id="agent_1",
            entity_id="any",
            action="call",
        )

        assert result["success"] is False
        assert "not enabled" in result["error"]

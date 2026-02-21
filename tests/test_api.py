"""Tests for the SWARM Web API."""

import pytest

# Skip all tests if fastapi is not installed
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from swarm.api.app import create_app  # noqa: E402
from swarm.api.config import APIConfig  # noqa: E402
from swarm.api.models.agent import AgentStatus  # noqa: E402
from swarm.api.models.scenario import ScenarioStatus  # noqa: E402
from swarm.api.models.simulation import SimulationMode, SimulationStatus  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_registration_state():
    """Reset registration and agent state between tests."""
    import swarm.api.routers.agents as agents_mod
    agents_mod._registration_rate.clear()
    agents_mod._registered_agents.clear()
    agents_mod._pending_keys.clear()
    yield
    agents_mod._registration_rate.clear()
    agents_mod._registered_agents.clear()
    agents_mod._pending_keys.clear()


@pytest.fixture(autouse=True)
def _clear_auth_state():
    """Reset auth module state between tests."""
    import swarm.api.middleware.auth as auth_mod
    auth_mod._api_keys.clear()
    auth_mod._key_quotas.clear()
    auth_mod._trusted_keys.clear()
    auth_mod._key_scopes.clear()
    auth_mod._rate_limit_windows.clear()
    yield
    auth_mod._api_keys.clear()
    auth_mod._key_quotas.clear()
    auth_mod._trusted_keys.clear()
    auth_mod._key_scopes.clear()
    auth_mod._rate_limit_windows.clear()


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app(APIConfig(debug=True))
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SWARM API"
        assert data["version"] == "1.0.0"
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestAgentEndpoints:
    """Tests for agent registration and management."""

    def test_register_agent(self, client):
        """Test registering a new agent."""
        response = client.post(
            "/api/v1/agents/register",
            json={
                "name": "TestAgent",
                "description": "A test agent",
                "capabilities": ["negotiation", "analysis"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TestAgent"
        assert data["description"] == "A test agent"
        assert data["capabilities"] == ["negotiation", "analysis"]
        assert data["status"] == AgentStatus.APPROVED.value
        assert "agent_id" in data
        assert "api_key" in data
        assert data["api_key"].startswith("swarm_")

    def test_register_agent_minimal(self, client):
        """Test registering an agent with minimal fields."""
        response = client.post(
            "/api/v1/agents/register",
            json={
                "name": "MinimalAgent",
                "description": "Minimal description",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MinimalAgent"
        assert data["capabilities"] == []

    def test_register_agent_with_policy(self, client):
        """Test registering an agent with policy declaration."""
        response = client.post(
            "/api/v1/agents/register",
            json={
                "name": "PolicyAgent",
                "description": "Agent with policy",
                "policy_declaration": "I will always act honestly.",
                "callback_url": "https://example.com/webhook",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "PolicyAgent"

    def test_register_agent_validation_name_required(self, client):
        """Test that agent name is required."""
        response = client.post(
            "/api/v1/agents/register",
            json={"description": "Missing name"},
        )
        assert response.status_code == 422

    def test_register_agent_validation_description_required(self, client):
        """Test that agent description is required."""
        response = client.post(
            "/api/v1/agents/register",
            json={"name": "NoDescription"},
        )
        assert response.status_code == 422

    def test_get_agent(self, client):
        """Test getting an agent by ID."""
        # First register an agent
        register_response = client.post(
            "/api/v1/agents/register",
            json={"name": "GetTestAgent", "description": "Test"},
        )
        agent_id = register_response.json()["agent_id"]

        # Then get the agent
        response = client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == agent_id
        assert data["name"] == "GetTestAgent"
        assert data["api_key"] == "[REDACTED]"

    def test_get_agent_not_found(self, client):
        """Test getting a non-existent agent."""
        response = client.get("/api/v1/agents/nonexistent-id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Agent not found"

    def test_list_agents(self, client):
        """Test listing all agents."""
        # Register some agents
        client.post(
            "/api/v1/agents/register",
            json={"name": "ListAgent1", "description": "First"},
        )
        client.post(
            "/api/v1/agents/register",
            json={"name": "ListAgent2", "description": "Second"},
        )

        response = client.get("/api/v1/agents/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        # API keys should be redacted
        for agent in data:
            assert agent["api_key"] == "[REDACTED]"


class TestScenarioEndpoints:
    """Tests for scenario submission and management."""

    def test_submit_scenario(self, client):
        """Test submitting a valid scenario."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Test Scenario",
                "description": "A test scenario",
                "yaml_content": "agents: 10\nepochs: 100",
                "tags": ["test", "safety"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Scenario"
        assert data["status"] == ScenarioStatus.VALID.value
        assert data["validation_errors"] == []
        assert data["tags"] == ["test", "safety"]
        assert "scenario_id" in data

    def test_submit_scenario_empty_yaml_invalid(self, client):
        """Test that empty YAML content is invalid."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Empty Scenario",
                "description": "Has empty YAML",
                "yaml_content": "   ",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.INVALID.value
        assert len(data["validation_errors"]) > 0

    def test_submit_scenario_minimal(self, client):
        """Test submitting a scenario with minimal fields."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Minimal",
                "description": "Minimal scenario",
                "yaml_content": "test: true",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == []

    def test_get_scenario(self, client):
        """Test getting a scenario by ID."""
        # First submit a scenario
        submit_response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "GetTest",
                "description": "Test",
                "yaml_content": "test: true",
            },
        )
        scenario_id = submit_response.json()["scenario_id"]

        # Then get the scenario
        response = client.get(f"/api/v1/scenarios/{scenario_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["scenario_id"] == scenario_id
        assert data["name"] == "GetTest"

    def test_get_scenario_not_found(self, client):
        """Test getting a non-existent scenario."""
        response = client.get("/api/v1/scenarios/nonexistent-id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Scenario not found"

    def test_list_scenarios(self, client):
        """Test listing all scenarios."""
        # Submit some scenarios
        client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "List1",
                "description": "First",
                "yaml_content": "a: 1",
            },
        )
        client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "List2",
                "description": "Second",
                "yaml_content": "b: 2",
            },
        )

        response = client.get("/api/v1/scenarios/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2


class TestSimulationEndpoints:
    """Tests for simulation management."""

    def test_create_simulation(self, client):
        """Test creating a simulation."""
        # First create a scenario
        scenario_response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "SimScenario",
                "description": "For simulation",
                "yaml_content": "test: true",
            },
        )
        scenario_id = scenario_response.json()["scenario_id"]

        # Create simulation
        response = client.post(
            "/api/v1/simulations/create",
            json={
                "scenario_id": scenario_id,
                "max_participants": 5,
                "mode": "async",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["scenario_id"] == scenario_id
        assert data["status"] == SimulationStatus.WAITING.value
        assert data["mode"] == SimulationMode.ASYNC.value
        assert data["max_participants"] == 5
        assert data["current_participants"] == 0
        assert "simulation_id" in data
        assert "join_deadline" in data

    def test_create_simulation_realtime(self, client):
        """Test creating a realtime simulation."""
        response = client.post(
            "/api/v1/simulations/create",
            json={
                "scenario_id": "test-scenario",
                "mode": "realtime",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == SimulationMode.REALTIME.value

    def test_create_simulation_with_config_overrides(self, client):
        """Test creating a simulation with config overrides."""
        response = client.post(
            "/api/v1/simulations/create",
            json={
                "scenario_id": "test-scenario",
                "config_overrides": {"epochs": 50, "tax_rate": 0.1},
            },
        )
        assert response.status_code == 200

    def test_join_simulation(self, client):
        """Test joining a simulation."""
        # Create simulation
        sim_response = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 3},
        )
        simulation_id = sim_response.json()["simulation_id"]

        # Register an agent
        agent_response = client.post(
            "/api/v1/agents/register",
            json={"name": "JoinAgent", "description": "Joins simulations"},
        )
        agent_id = agent_response.json()["agent_id"]

        # Join simulation
        response = client.post(
            f"/api/v1/simulations/{simulation_id}/join",
            json={"agent_id": agent_id, "role": "initiator"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == simulation_id
        assert data["agent_id"] == agent_id
        assert data["role"] == "initiator"
        assert data["status"] == "joined"
        assert "participant_id" in data

    def test_join_simulation_updates_count(self, client):
        """Test that joining a simulation updates participant count."""
        # Create simulation
        sim_response = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 5},
        )
        simulation_id = sim_response.json()["simulation_id"]

        # Join with two agents
        for i in range(2):
            agent_response = client.post(
                "/api/v1/agents/register",
                json={"name": f"CountAgent{i}", "description": "Test"},
            )
            client.post(
                f"/api/v1/simulations/{simulation_id}/join",
                json={"agent_id": agent_response.json()["agent_id"]},
            )

        # Check participant count
        response = client.get(f"/api/v1/simulations/{simulation_id}")
        assert response.json()["current_participants"] == 2

    def test_join_simulation_not_found(self, client):
        """Test joining a non-existent simulation."""
        response = client.post(
            "/api/v1/simulations/nonexistent/join",
            json={"agent_id": "test"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Simulation not found"

    def test_join_simulation_full(self, client):
        """Test joining a full simulation."""
        # Create simulation with max 2 participants
        sim_response = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 2},
        )
        simulation_id = sim_response.json()["simulation_id"]

        # Fill the simulation
        for i in range(2):
            agent_response = client.post(
                "/api/v1/agents/register",
                json={"name": f"FillAgent{i}", "description": "Test"},
            )
            client.post(
                f"/api/v1/simulations/{simulation_id}/join",
                json={"agent_id": agent_response.json()["agent_id"]},
            )

        # Try to join with another agent
        extra_agent = client.post(
            "/api/v1/agents/register",
            json={"name": "ExtraAgent", "description": "Test"},
        )
        response = client.post(
            f"/api/v1/simulations/{simulation_id}/join",
            json={"agent_id": extra_agent.json()["agent_id"]},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Simulation is full"

    def test_get_simulation(self, client):
        """Test getting a simulation by ID."""
        # Create simulation
        sim_response = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test"},
        )
        simulation_id = sim_response.json()["simulation_id"]

        # Get simulation
        response = client.get(f"/api/v1/simulations/{simulation_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == simulation_id

    def test_get_simulation_not_found(self, client):
        """Test getting a non-existent simulation."""
        response = client.get("/api/v1/simulations/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Simulation not found"

    def test_list_simulations(self, client):
        """Test listing all simulations."""
        # Create some simulations
        client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "list-test-1"},
        )
        client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "list-test-2"},
        )

        response = client.get("/api/v1/simulations/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2


class TestAPIConfig:
    """Tests for API configuration."""

    def test_default_config(self):
        """Test default API configuration."""
        config = APIConfig()
        assert config.host == "127.0.0.1"  # Localhost only for security
        assert config.port == 8000
        assert config.debug is False
        assert config.cors_origins == ["http://localhost:8000", "http://127.0.0.1:8000"]
        assert config.rate_limit_per_minute == 100
        assert config.auto_approve_agents is True

    def test_custom_config(self):
        """Test custom API configuration."""
        config = APIConfig(
            host="127.0.0.1",
            port=9000,
            debug=True,
            cors_origins=["https://example.com"],
            rate_limit_per_minute=50,
            auto_approve_agents=False,
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.debug is True
        assert config.cors_origins == ["https://example.com"]
        assert config.rate_limit_per_minute == 50
        assert config.auto_approve_agents is False

    def test_create_app_with_config(self):
        """Test creating app with custom config."""
        config = APIConfig(debug=True)
        app = create_app(config)
        assert app.title == "SWARM API"


class TestAPIModels:
    """Tests for API Pydantic models."""

    def test_agent_registration_model(self):
        """Test AgentRegistration model."""
        from swarm.api.models.agent import AgentRegistration

        reg = AgentRegistration(
            name="Test",
            description="Test agent",
            capabilities=["test"],
            policy_declaration="Be good",
            callback_url="https://example.com",
        )
        assert reg.name == "Test"
        assert reg.capabilities == ["test"]

    def test_agent_registration_defaults(self):
        """Test AgentRegistration default values."""
        from swarm.api.models.agent import AgentRegistration

        reg = AgentRegistration(name="Test", description="Test")
        assert reg.capabilities == []
        assert reg.policy_declaration is None
        assert reg.callback_url is None

    def test_scenario_submission_model(self):
        """Test ScenarioSubmission model."""
        from swarm.api.models.scenario import ScenarioSubmission

        sub = ScenarioSubmission(
            name="Test",
            description="Test scenario",
            yaml_content="test: true",
            tags=["test"],
        )
        assert sub.name == "Test"
        assert sub.tags == ["test"]

    def test_simulation_create_model(self):
        """Test SimulationCreate model."""
        from swarm.api.models.simulation import SimulationCreate, SimulationMode

        sim = SimulationCreate(
            scenario_id="test",
            max_participants=10,
            mode=SimulationMode.REALTIME,
        )
        assert sim.scenario_id == "test"
        assert sim.max_participants == 10
        assert sim.mode == SimulationMode.REALTIME

    def test_simulation_create_defaults(self):
        """Test SimulationCreate default values."""
        from swarm.api.models.simulation import SimulationCreate, SimulationMode

        sim = SimulationCreate(scenario_id="test")
        assert sim.max_participants == 10
        assert sim.mode == SimulationMode.ASYNC
        assert sim.config_overrides == {}

    def test_simulation_create_validation(self):
        """Test SimulationCreate validation."""
        from pydantic import ValidationError

        from swarm.api.models.simulation import SimulationCreate

        # max_participants must be >= 2
        with pytest.raises(ValidationError):
            SimulationCreate(scenario_id="test", max_participants=1)

        # max_participants must be <= 100
        with pytest.raises(ValidationError):
            SimulationCreate(scenario_id="test", max_participants=101)


# ---------------------------------------------------------------------------
# Phase 1 gap tests
# ---------------------------------------------------------------------------


class TestScopedPermissions:
    """Tests for the Scope enum and require_scope dependency."""

    def test_default_scopes_allow_read(self, client):
        """Registering an agent gives default scopes (read + participate)."""
        from swarm.api.middleware.auth import Scope, _hash_key, get_scopes_hash

        resp = client.post(
            "/api/v1/agents/register",
            json={"name": "ScopeAgent", "description": "test"},
        )
        api_key = resp.json()["api_key"]
        key_hash = _hash_key(api_key)
        scopes = get_scopes_hash(key_hash)
        assert Scope.READ in scopes
        assert Scope.PARTICIPATE in scopes

    def test_missing_scope_returns_403(self, client):
        """A key lacking a required scope gets 403."""
        from swarm.api.middleware.auth import Scope, register_api_key

        # Register a key with only READ scope (no WRITE or PARTICIPATE)
        register_api_key(
            "read-only-key", "agent-ro", scopes=frozenset({Scope.READ})
        )

        post_resp = client.post(
            "/api/posts",
            json={
                "run_id": "fake-run-id",
                "title": "Test",
                "blurb": "blurb",
            },
            headers={"Authorization": "Bearer read-only-key"},
        )
        assert post_resp.status_code == 403
        assert "scope" in post_resp.json()["detail"].lower()

    def test_trusted_key_maps_to_all_scopes(self):
        """trusted=True grants all four scopes."""
        from swarm.api.middleware.auth import (
            Scope,
            _hash_key,
            get_scopes_hash,
            register_api_key,
        )

        register_api_key("trusted-key-123", "agent-1", trusted=True)
        key_hash = _hash_key("trusted-key-123")
        scopes = get_scopes_hash(key_hash)
        assert scopes == frozenset(Scope)

    def test_admin_scope_implies_trusted(self):
        """A key with ADMIN scope passes is_trusted_hash."""
        from swarm.api.middleware.auth import (
            Scope,
            _hash_key,
            is_trusted_hash,
            register_api_key,
        )

        register_api_key(
            "admin-key-456", "agent-2", scopes=frozenset({Scope.ADMIN})
        )
        key_hash = _hash_key("admin-key-456")
        assert is_trusted_hash(key_hash)


class TestTraceID:
    """Tests for the TraceIDMiddleware."""

    def test_response_includes_trace_id(self, client):
        """Every response should include an X-Trace-ID header."""
        resp = client.get("/health")
        assert "X-Trace-ID" in resp.headers
        assert len(resp.headers["X-Trace-ID"]) > 0

    def test_client_trace_id_echoed(self, client):
        """Client-provided X-Trace-ID should be echoed back."""
        custom_id = "my-custom-trace-id-42"
        resp = client.get("/health", headers={"X-Trace-ID": custom_id})
        assert resp.headers["X-Trace-ID"] == custom_id

    def test_error_body_contains_trace_id(self, client):
        """Error responses should include trace_id in the JSON body."""
        resp = client.get(
            "/api/v1/agents/nonexistent-id",
            headers={"X-Trace-ID": "err-trace-99"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["trace_id"] == "err-trace-99"


class TestStructuredErrors:
    """Tests for structured ErrorResponse format."""

    def test_404_follows_schema(self, client):
        """A 404 should follow the ErrorResponse schema."""
        resp = client.get("/api/v1/agents/does-not-exist")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "Not Found"
        assert "detail" in body
        assert "trace_id" in body
        assert body["status_code"] == 404

    def test_401_follows_schema(self, client):
        """A 401 should follow the ErrorResponse schema."""
        resp = client.get(
            "/api/runs",
            headers={"Authorization": "Bearer bad-key"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "Unauthorized"
        assert body["status_code"] == 401
        assert "trace_id" in body

    def test_422_follows_schema(self, client):
        """A 422 (validation) should follow the ErrorResponse schema."""
        resp = client.post(
            "/api/v1/agents/register",
            json={"description": "Missing name"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "Unprocessable Entity"
        assert body["status_code"] == 422
        assert "trace_id" in body


class TestListFilteringAndPagination:
    """Tests for list endpoint filtering and pagination."""

    @pytest.fixture(autouse=True)
    def _clear_stores(self):
        """Clear in-memory stores so each test starts fresh."""
        import swarm.api.routers.agents as agents_mod
        import swarm.api.routers.scenarios as scenarios_mod
        import swarm.api.routers.simulations as simulations_mod

        agents_mod._registered_agents.clear()
        scenarios_mod._scenarios.clear()
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        yield
        agents_mod._registered_agents.clear()
        scenarios_mod._scenarios.clear()
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()

    def _register_agent(self, client, name, capabilities=None):
        """Helper to register an agent and return the response data."""
        payload = {"name": name, "description": f"Agent {name}"}
        if capabilities is not None:
            payload["capabilities"] = capabilities
        resp = client.post("/api/v1/agents/register", json=payload)
        assert resp.status_code == 200
        return resp.json()

    def _submit_scenario(self, client, name, tags=None, yaml_content="x: 1"):
        """Helper to submit a scenario and return the response data."""
        payload = {
            "name": name,
            "description": f"Scenario {name}",
            "yaml_content": yaml_content,
        }
        if tags is not None:
            payload["tags"] = tags
        resp = client.post("/api/v1/scenarios/submit", json=payload)
        assert resp.status_code == 200
        return resp.json()

    def _create_simulation(self, client, scenario_id="test-scenario"):
        """Helper to create a simulation and return the response data."""
        resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": scenario_id},
        )
        assert resp.status_code == 200
        return resp.json()

    # --- Agents ---

    def test_agents_filter_by_status(self, client):
        """Register agents and filter by APPROVED status."""
        self._register_agent(client, "Alpha")
        self._register_agent(client, "Beta")

        resp = client.get("/api/v1/agents/", params={"status": "approved"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(a["status"] == AgentStatus.APPROVED.value for a in data)

        # Filter by a status with no matches
        resp = client.get("/api/v1/agents/", params={"status": "suspended"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_agents_filter_by_capability(self, client):
        """Filter agents by capability."""
        self._register_agent(client, "Analyst", capabilities=["analysis"])
        self._register_agent(client, "Negotiator", capabilities=["negotiation"])
        self._register_agent(
            client, "Both", capabilities=["analysis", "negotiation"]
        )

        resp = client.get("/api/v1/agents/", params={"capability": "analysis"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = {a["name"] for a in data}
        assert names == {"Analyst", "Both"}

    def test_agents_pagination(self, client):
        """Register 5 agents, limit=2 offset=1 returns 2."""
        for i in range(5):
            self._register_agent(client, f"PagAgent{i}")

        resp = client.get("/api/v1/agents/", params={"limit": 2, "offset": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_agents_pagination_offset_past_end(self, client):
        """Offset past the end returns empty list."""
        self._register_agent(client, "Solo")

        resp = client.get("/api/v1/agents/", params={"offset": 100})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_agents_default_returns_all(self, client):
        """Calling without params still returns all agents (backward compat)."""
        for i in range(3):
            self._register_agent(client, f"DefaultAgent{i}")

        resp = client.get("/api/v1/agents/")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    # --- Scenarios ---

    def test_scenarios_filter_by_status(self, client):
        """Filter scenarios by VALID status."""
        self._submit_scenario(client, "Good", yaml_content="a: 1")
        self._submit_scenario(client, "Bad", yaml_content="   ")  # invalid

        resp = client.get("/api/v1/scenarios/", params={"status": "valid"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Good"

    def test_scenarios_filter_by_tag(self, client):
        """Filter scenarios by tag."""
        self._submit_scenario(client, "Safety", tags=["safety", "alignment"])
        self._submit_scenario(client, "Game", tags=["game-theory"])
        self._submit_scenario(client, "Both", tags=["safety", "game-theory"])

        resp = client.get("/api/v1/scenarios/", params={"tag": "safety"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = {s["name"] for s in data}
        assert names == {"Safety", "Both"}

    def test_scenarios_pagination(self, client):
        """Submit 3 scenarios, limit=1 returns 1."""
        for i in range(3):
            self._submit_scenario(client, f"PagScenario{i}")

        resp = client.get("/api/v1/scenarios/", params={"limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_scenarios_default_returns_all(self, client):
        """Calling without params still returns all scenarios."""
        for i in range(3):
            self._submit_scenario(client, f"DefScenario{i}")

        resp = client.get("/api/v1/scenarios/")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    # --- Simulations ---

    def test_simulations_filter_by_status(self, client):
        """Filter simulations by WAITING status."""
        self._create_simulation(client, "scenario-a")
        self._create_simulation(client, "scenario-b")

        resp = client.get(
            "/api/v1/simulations/",
            params={"status": "waiting_for_participants"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(
            s["status"] == SimulationStatus.WAITING.value for s in data
        )

        # No matches for COMPLETED
        resp = client.get(
            "/api/v1/simulations/", params={"status": "completed"}
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_simulations_pagination(self, client):
        """Create 4 simulations, limit=2 offset=1 returns 2."""
        for i in range(4):
            self._create_simulation(client, f"scenario-{i}")

        resp = client.get(
            "/api/v1/simulations/", params={"limit": 2, "offset": 1}
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_simulations_default_returns_all(self, client):
        """Calling without params still returns all simulations."""
        for i in range(3):
            self._create_simulation(client, f"scenario-{i}")

        resp = client.get("/api/v1/simulations/")
        assert resp.status_code == 200
        assert len(resp.json()) == 3


class TestRouterStubs:
    """Tests for governance and metrics router basic access."""

    def test_governance_propose_requires_auth(self, client):
        """POST /api/v1/governance/propose requires auth."""
        resp = client.post("/api/v1/governance/propose")
        assert resp.status_code == 401

    def test_governance_proposals_returns_list(self, client):
        """GET /api/v1/governance/proposals returns empty list by default."""
        import swarm.api.routers.governance as gov_mod
        gov_mod._proposals.clear()
        resp = client.get("/api/v1/governance/proposals")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_metrics_requires_auth(self, client):
        """GET /api/v1/metrics/{id} requires auth."""
        resp = client.get("/api/v1/metrics/some-sim-id")
        assert resp.status_code == 401


class TestAgentPatchAndTransitions:
    """Tests for PATCH /agents/{id} and status transition endpoints."""

    def _register_agent(self, client):
        """Helper: register an agent and return (agent_id, api_key)."""
        resp = client.post(
            "/api/v1/agents/register",
            json={"name": "PatchAgent", "description": "For patch tests"},
        )
        data = resp.json()
        return data["agent_id"], data["api_key"]

    def test_patch_own_agent(self, client):
        """Updating own name and description succeeds."""
        agent_id, api_key = self._register_agent(client)

        resp = client.patch(
            f"/api/v1/agents/{agent_id}",
            json={"name": "UpdatedName", "description": "Updated desc"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "UpdatedName"
        assert data["description"] == "Updated desc"
        assert data["api_key"] == "[REDACTED]"

    def test_patch_other_agent_forbidden(self, client):
        """Updating another agent's profile returns 403."""
        agent_id_1, _api_key_1 = self._register_agent(client)
        _agent_id_2, api_key_2 = self._register_agent(client)

        resp = client.patch(
            f"/api/v1/agents/{agent_id_1}",
            json={"name": "Hijacked"},
            headers={"Authorization": f"Bearer {api_key_2}"},
        )
        assert resp.status_code == 403

    def test_suspend_agent(self, client):
        """Admin can suspend an approved agent."""
        from swarm.api.middleware.auth import register_api_key

        agent_id, _api_key = self._register_agent(client)

        # Register an admin key
        register_api_key("admin-key", "admin-agent", trusted=True)

        resp = client.post(
            f"/api/v1/agents/{agent_id}/suspend",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == AgentStatus.SUSPENDED.value

    def test_reactivate_agent(self, client):
        """Admin can reactivate a suspended agent."""
        from swarm.api.middleware.auth import register_api_key

        agent_id, _api_key = self._register_agent(client)

        register_api_key("admin-key", "admin-agent", trusted=True)

        # First suspend
        client.post(
            f"/api/v1/agents/{agent_id}/suspend",
            headers={"Authorization": "Bearer admin-key"},
        )

        # Then reactivate
        resp = client.post(
            f"/api/v1/agents/{agent_id}/reactivate",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == AgentStatus.APPROVED.value

    def test_invalid_transition(self, client):
        """Suspending a pending agent returns 400."""
        # Directly insert a PENDING agent into the store
        from datetime import datetime, timezone

        import swarm.api.routers.agents as agents_mod
        from swarm.api.middleware.auth import register_api_key
        from swarm.api.models.agent import AgentResponse as AR

        pending_id = "pending-agent-123"
        agents_mod._registered_agents[pending_id] = AR(
            agent_id=pending_id,
            api_key="unused",
            name="PendingAgent",
            description="Stuck in review",
            capabilities=[],
            status=AgentStatus.PENDING,
            registered_at=datetime.now(timezone.utc),
        )

        register_api_key("admin-key", "admin-agent", trusted=True)

        resp = client.post(
            f"/api/v1/agents/{pending_id}/suspend",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 400
        assert "pending_review" in resp.json()["detail"]


class TestYAMLValidation:
    """Tests for YAML schema validation and resource estimation."""

    def test_valid_yaml_accepted(self, client):
        """Submit valid YAML with agents/epochs, expect status=VALID."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Valid Scenario",
                "description": "A valid scenario with params",
                "yaml_content": "agents: 5\nepochs: 20\nsteps_per_epoch: 50",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.VALID.value
        assert data["validation_errors"] == []

    def test_empty_yaml_invalid(self, client):
        """Empty content returns INVALID (existing behavior preserved)."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Empty",
                "description": "Empty YAML",
                "yaml_content": "   ",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.INVALID.value
        assert len(data["validation_errors"]) > 0
        assert any("empty" in e.lower() for e in data["validation_errors"])

    def test_malformed_yaml_invalid(self, client):
        """Malformed YAML returns INVALID with parse error."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "Malformed",
                "description": "Bad YAML",
                "yaml_content": "{{bad yaml",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.INVALID.value
        assert any("parse error" in e.lower() for e in data["validation_errors"])

    def test_wrong_type_yaml_invalid(self, client):
        """YAML that parses to a string (not dict) returns INVALID."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "WrongType",
                "description": "String YAML",
                "yaml_content": "just a string",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.INVALID.value
        assert any(
            "mapping" in e.lower() or "dict" in e.lower()
            for e in data["validation_errors"]
        )

    def test_invalid_param_types(self, client):
        """YAML with wrong param types returns INVALID."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "BadTypes",
                "description": "Wrong types",
                "yaml_content": "agents: not_a_number",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.INVALID.value
        assert any(
            "agents" in e and "integer" in e for e in data["validation_errors"]
        )

    def test_resource_estimate_present(self, client):
        """Valid scenario includes resource_estimate dict."""
        response = client.post(
            "/api/v1/scenarios/submit",
            json={
                "name": "ResourceEst",
                "description": "Check resource estimate",
                "yaml_content": "agents: 4\nepochs: 5\nsteps_per_epoch: 10",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ScenarioStatus.VALID.value
        est = data["resource_estimate"]
        assert est is not None
        assert est["n_agents"] == 4
        assert est["n_epochs"] == 5
        assert est["steps"] == 10
        assert est["estimated_interactions"] == 4 * 5 * 10
        assert est["estimated_runtime_seconds"] == 4 * 5 * 10 * 0.01
        assert "estimated_memory_mb" in est
        assert est["within_limits"] is True


class TestSimulationMechanics:
    """Tests for join deadline enforcement, state endpoint, and start."""

    @pytest.fixture(autouse=True)
    def _clear_simulation_state(self):
        """Reset simulation storage between tests."""
        import swarm.api.routers.simulations as simulations_mod

        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        yield
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()

    def _create_simulation(self, client, **overrides):
        """Helper: create a simulation and return the response data."""
        payload = {"scenario_id": "test-scenario", "max_participants": 10}
        payload.update(overrides)
        resp = client.post("/api/v1/simulations/create", json=payload)
        assert resp.status_code == 200
        return resp.json()

    def test_join_after_deadline_rejected(self, client):
        """Joining after the join deadline returns 400."""
        from datetime import datetime, timedelta, timezone

        import swarm.api.routers.simulations as simulations_mod

        sim = self._create_simulation(client)
        sim_id = sim["simulation_id"]

        # Set the deadline to the past
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        simulations_mod._simulations[sim_id].join_deadline = past

        resp = client.post(
            f"/api/v1/simulations/{sim_id}/join",
            json={"agent_id": "agent-late", "role": "participant"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Join deadline has passed"

    def test_state_endpoint(self, client):
        """GET /state returns participants and simulation details."""
        sim = self._create_simulation(
            client, config_overrides={"epochs": 50}
        )
        sim_id = sim["simulation_id"]

        # Join with an agent
        client.post(
            f"/api/v1/simulations/{sim_id}/join",
            json={"agent_id": "agent-1", "role": "initiator"},
        )

        resp = client.get(f"/api/v1/simulations/{sim_id}/state")
        assert resp.status_code == 200
        data = resp.json()

        assert data["simulation_id"] == sim_id
        assert data["status"] == SimulationStatus.WAITING.value
        assert len(data["participants"]) == 1
        assert data["participants"][0]["agent_id"] == "agent-1"
        assert data["participants"][0]["role"] == "initiator"
        assert "joined_at" in data["participants"][0]
        assert data["config_overrides"] == {"epochs": 50}
        assert "join_deadline" in data
        assert data["time_remaining_seconds"] >= 0

    def test_start_simulation(self, client):
        """POST /start transitions WAITING -> RUNNING with enough participants."""
        sim = self._create_simulation(client)
        sim_id = sim["simulation_id"]

        # Join with 2 agents
        for i in range(2):
            client.post(
                f"/api/v1/simulations/{sim_id}/join",
                json={"agent_id": f"agent-{i}", "role": "participant"},
            )

        resp = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == sim_id
        assert data["status"] == "running"

        # Verify the simulation is now running via GET
        get_resp = client.get(f"/api/v1/simulations/{sim_id}")
        assert get_resp.json()["status"] == SimulationStatus.RUNNING.value

    def test_start_without_enough_participants(self, client):
        """POST /start with < 2 participants returns 400."""
        sim = self._create_simulation(client)
        sim_id = sim["simulation_id"]

        # Join with only 1 agent
        client.post(
            f"/api/v1/simulations/{sim_id}/join",
            json={"agent_id": "solo-agent", "role": "participant"},
        )

        resp = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert resp.status_code == 400
        assert "Not enough participants" in resp.json()["detail"]

    def test_start_non_waiting_simulation(self, client):
        """POST /start on an already-running simulation returns 400."""
        sim = self._create_simulation(client)
        sim_id = sim["simulation_id"]

        # Join with 2 agents and start
        for i in range(2):
            client.post(
                f"/api/v1/simulations/{sim_id}/join",
                json={"agent_id": f"agent-{i}", "role": "participant"},
            )
        client.post(f"/api/v1/simulations/{sim_id}/start")

        # Try to start again
        resp = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert resp.status_code == 400
        assert "not in waiting state" in resp.json()["detail"]


class TestApprovalWorkflow:
    """Tests for the agent approval workflow (auto_approve=False)."""

    @pytest.fixture
    def review_client(self):
        """Client with auto_approve_agents=False."""
        app = create_app(APIConfig(debug=True, auto_approve_agents=False))
        return TestClient(app)

    def test_registration_returns_pending_when_auto_approve_off(self, review_client):
        """Registering with auto_approve=False gives pending_review status."""
        resp = review_client.post(
            "/api/v1/agents/register",
            json={"name": "PendingBot", "description": "awaiting review"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending_review"
        assert "api_key" in data

    def test_pending_key_inactive_until_approved(self, review_client):
        """A pending agent's API key cannot authenticate."""
        resp = review_client.post(
            "/api/v1/agents/register",
            json={"name": "InactiveBot", "description": "test"},
        )
        api_key = resp.json()["api_key"]

        # Try to use the key — should fail (key not registered yet)
        runs_resp = review_client.get(
            "/api/runs",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert runs_resp.status_code == 401

    def test_admin_approves_pending_agent(self, review_client):
        """Admin can approve a pending agent, activating their key."""
        from swarm.api.middleware.auth import register_api_key

        # Register admin key
        register_api_key("admin-approval-key", "admin-1", trusted=True)

        # Register a pending agent
        resp = review_client.post(
            "/api/v1/agents/register",
            json={"name": "ToApprove", "description": "test"},
        )
        agent_id = resp.json()["agent_id"]
        api_key = resp.json()["api_key"]
        assert resp.json()["status"] == "pending_review"

        # Admin approves
        approve_resp = review_client.post(
            f"/api/v1/agents/{agent_id}/approve",
            headers={"Authorization": "Bearer admin-approval-key"},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "approved"

        # Key is now active
        runs_resp = review_client.get(
            "/api/runs",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert runs_resp.status_code == 200

    def test_admin_rejects_pending_agent(self, review_client):
        """Admin can reject a pending agent."""
        from swarm.api.middleware.auth import register_api_key

        register_api_key("admin-reject-key", "admin-2", trusted=True)

        resp = review_client.post(
            "/api/v1/agents/register",
            json={"name": "ToReject", "description": "test"},
        )
        agent_id = resp.json()["agent_id"]
        api_key = resp.json()["api_key"]

        reject_resp = review_client.post(
            f"/api/v1/agents/{agent_id}/reject",
            headers={"Authorization": "Bearer admin-reject-key"},
        )
        assert reject_resp.status_code == 200
        assert reject_resp.json()["status"] == "rejected"

        # Key stays inactive
        runs_resp = review_client.get(
            "/api/runs",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert runs_resp.status_code == 401

    def test_approve_non_pending_agent_fails(self, review_client):
        """Cannot approve an agent that isn't pending."""
        from swarm.api.middleware.auth import register_api_key

        register_api_key("admin-key-3", "admin-3", trusted=True)

        # Register and approve first
        resp = review_client.post(
            "/api/v1/agents/register",
            json={"name": "AlreadyDone", "description": "test"},
        )
        agent_id = resp.json()["agent_id"]
        review_client.post(
            f"/api/v1/agents/{agent_id}/approve",
            headers={"Authorization": "Bearer admin-key-3"},
        )

        # Try to approve again
        resp2 = review_client.post(
            f"/api/v1/agents/{agent_id}/approve",
            headers={"Authorization": "Bearer admin-key-3"},
        )
        assert resp2.status_code == 400
        assert "pending" in resp2.json()["detail"].lower()

    def test_auto_approve_true_still_works(self, client):
        """Default client (auto_approve=True) still auto-approves."""
        resp = client.post(
            "/api/v1/agents/register",
            json={"name": "AutoBot", "description": "test"},
        )
        assert resp.json()["status"] == "approved"


# ---------------------------------------------------------------------------
# AsyncActionQueue unit tests
# ---------------------------------------------------------------------------


class TestAsyncActionQueue:
    """Tests for the AsyncActionQueue."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Import the queue class."""
        from swarm.api.action_queue import AsyncActionQueue

        self.AsyncActionQueue = AsyncActionQueue

    def test_submit_resolves_waiting(self):
        """submit_action resolves a pending wait_for_action."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=5000)

        async def _run():
            async def waiter():
                return await queue.wait_for_action("agent-1")

            async def submitter():
                await asyncio.sleep(0.05)
                return await queue.submit_action(
                    "agent-1", {"action_type": "accept"}
                )

            result, accepted = await asyncio.gather(waiter(), submitter())
            assert result == {"action_type": "accept"}
            assert accepted is True

        asyncio.run(_run())

    def test_timeout_returns_none(self):
        """wait_for_action returns None on timeout."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=50)

        async def _run():
            result = await queue.wait_for_action("agent-timeout")
            assert result is None

        asyncio.run(_run())

    def test_cancel_all_clears_pending(self):
        """cancel_all cancels all pending futures."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=5000)

        async def _run():
            async def waiter():
                return await queue.wait_for_action("agent-cancel")

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.02)
            assert queue.pending_count == 1

            cancelled = await queue.cancel_all()
            assert cancelled == 1

            result = await task
            assert result is None

        asyncio.run(_run())

    def test_submit_without_waiter_returns_false(self):
        """submit_action with no waiter returns False."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=5000)

        async def _run():
            accepted = await queue.submit_action(
                "no-waiter", {"action_type": "noop"}
            )
            assert accepted is False

        asyncio.run(_run())

    def test_rate_limit_per_step(self):
        """Rate limit rejects actions beyond max per step."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=5000)
        queue._max_actions_per_step = 2

        async def _run():
            # Submit 2 actions (no waiter, but counts should still track)
            # We need waiters for the actions to be accepted
            for i in range(3):

                async def waiter():
                    return await queue.wait_for_action("agent-rl")

                task = asyncio.create_task(waiter())
                await asyncio.sleep(0.01)
                result = await queue.submit_action(
                    "agent-rl", {"action_type": "accept", "i": i}
                )
                if i < 2:
                    assert result is True
                else:
                    assert result is False
                await task

        asyncio.run(_run())

    def test_reset_step_clears_counts(self):
        """reset_step clears per-agent action counts."""
        import asyncio

        queue = self.AsyncActionQueue(timeout_ms=5000)
        queue._max_actions_per_step = 1

        async def _run():
            # Use up the limit
            task = asyncio.create_task(queue.wait_for_action("agent-rs"))
            await asyncio.sleep(0.01)
            await queue.submit_action("agent-rs", {"action_type": "accept"})
            await task

            # Should be blocked now
            task2 = asyncio.create_task(queue.wait_for_action("agent-rs"))
            await asyncio.sleep(0.01)
            assert await queue.submit_action("agent-rs", {"x": 1}) is False
            await queue.cancel_all()
            await task2

            # Reset and try again
            queue.reset_step()
            task3 = asyncio.create_task(queue.wait_for_action("agent-rs"))
            await asyncio.sleep(0.01)
            assert await queue.submit_action("agent-rs", {"x": 2}) is True
            await task3

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# EventBus unit tests
# ---------------------------------------------------------------------------


class TestEventBus:
    """Tests for the EventBus."""

    def test_subscribe_and_publish(self):
        """Published events arrive at subscriber queues."""
        import asyncio

        from swarm.api.event_bus import EventBus, SimEvent, SimEventType

        bus = EventBus()

        async def _run():
            q = bus.subscribe("sim-1", "agent-a")
            event = SimEvent(
                event_type=SimEventType.STEP_COMPLETE,
                simulation_id="sim-1",
                data={"step": 1},
            )
            count = await bus.publish(event)
            assert count == 1

            received = q.get_nowait()
            assert received.event_type == SimEventType.STEP_COMPLETE
            assert received.data == {"step": 1}

        asyncio.run(_run())

    def test_targeted_event_filtering(self):
        """Targeted events only reach the specified agent."""
        import asyncio

        from swarm.api.event_bus import EventBus, SimEvent, SimEventType

        bus = EventBus()

        async def _run():
            q_a = bus.subscribe("sim-1", "agent-a")
            q_b = bus.subscribe("sim-1", "agent-b")

            event = SimEvent(
                event_type=SimEventType.OBSERVATION_READY,
                simulation_id="sim-1",
                data={"obs": "for-a"},
                agent_id="agent-a",
            )
            count = await bus.publish(event)
            assert count == 1

            assert q_b.qsize() == 0
            assert q_a.qsize() == 1

        asyncio.run(_run())

    def test_broadcast_reaches_all(self):
        """Broadcast events (agent_id=None) reach all subscribers."""
        import asyncio

        from swarm.api.event_bus import EventBus, SimEvent, SimEventType

        bus = EventBus()

        async def _run():
            q_a = bus.subscribe("sim-1", "agent-a")
            q_b = bus.subscribe("sim-1", "agent-b")

            event = SimEvent(
                event_type=SimEventType.EPOCH_COMPLETE,
                simulation_id="sim-1",
                data={"epoch": 1},
            )
            count = await bus.publish(event)
            assert count == 2
            assert q_a.qsize() == 1
            assert q_b.qsize() == 1

        asyncio.run(_run())

    def test_unsubscribe(self):
        """Unsubscribed queues no longer receive events."""
        import asyncio

        from swarm.api.event_bus import EventBus, SimEvent, SimEventType

        bus = EventBus()

        async def _run():
            q = bus.subscribe("sim-1", "agent-a")
            bus.unsubscribe("sim-1", "agent-a", q)
            assert bus.subscriber_count("sim-1") == 0

            event = SimEvent(
                event_type=SimEventType.STEP_COMPLETE,
                simulation_id="sim-1",
                data={},
            )
            count = await bus.publish(event)
            assert count == 0

        asyncio.run(_run())

    def test_full_queue_drops_events(self):
        """Events are dropped when a subscriber's queue is full."""
        import asyncio

        from swarm.api.event_bus import EventBus, SimEvent, SimEventType

        bus = EventBus()

        async def _run():
            q = bus.subscribe("sim-1", "agent-a")
            # Fill the queue (maxsize=100)
            for i in range(100):
                await bus.publish(
                    SimEvent(
                        event_type=SimEventType.STEP_COMPLETE,
                        simulation_id="sim-1",
                        data={"step": i},
                    )
                )

            # Next publish should silently drop
            count = await bus.publish(
                SimEvent(
                    event_type=SimEventType.STEP_COMPLETE,
                    simulation_id="sim-1",
                    data={"step": 100},
                )
            )
            assert count == 0
            assert q.qsize() == 100

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Action submission endpoint tests
# ---------------------------------------------------------------------------


class TestActionEndpoints:
    """Tests for action submission and observation endpoints."""

    @pytest.fixture(autouse=True)
    def _clear_simulation_state(self):
        """Reset simulation storage between tests."""
        import swarm.api.routers.simulations as simulations_mod

        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        yield
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()

    def _setup_running_simulation(self, client):
        """Create a running simulation with two participants, return (sim_id, agent_ids, api_keys)."""
        # Create simulation
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 5},
        )
        sim_id = sim_resp.json()["simulation_id"]

        agent_ids = []
        api_keys = []
        for i in range(2):
            agent_resp = client.post(
                "/api/v1/agents/register",
                json={"name": f"ActionAgent{i}", "description": "Test"},
            )
            data = agent_resp.json()
            agent_ids.append(data["agent_id"])
            api_keys.append(data["api_key"])
            client.post(
                f"/api/v1/simulations/{sim_id}/join",
                json={"agent_id": data["agent_id"], "role": "participant"},
            )

        # Start the simulation
        client.post(f"/api/v1/simulations/{sim_id}/start")

        return sim_id, agent_ids, api_keys

    def test_submit_action_accepted(self, client):
        """Submit an action to a running simulation."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        resp = client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": agent_ids[0],
                "action_type": "accept",
                "payload": {"value": 42},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == sim_id
        assert data["agent_id"] == agent_ids[0]
        assert data["status"] in ("accepted", "no_waiter")
        assert "action_id" in data

    def test_submit_action_not_running(self, client):
        """Cannot submit actions to a non-running simulation."""
        # Create but don't start
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test"},
        )
        sim_id = sim_resp.json()["simulation_id"]

        agent_resp = client.post(
            "/api/v1/agents/register",
            json={"name": "EarlyAgent", "description": "Test"},
        )
        api_key = agent_resp.json()["api_key"]

        resp = client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": agent_resp.json()["agent_id"],
                "action_type": "noop",
                "payload": {},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 400
        assert "not running" in resp.json()["detail"]

    def test_submit_action_non_participant(self, client):
        """Non-participant agent cannot submit actions."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Register a new agent that's NOT a participant
        outsider = client.post(
            "/api/v1/agents/register",
            json={"name": "Outsider", "description": "Not a participant"},
        )
        outsider_data = outsider.json()

        resp = client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": outsider_data["agent_id"],
                "action_type": "accept",
                "payload": {},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {outsider_data['api_key']}"},
        )
        assert resp.status_code == 403
        assert "not a participant" in resp.json()["detail"]

    def test_submit_action_simulation_not_found(self, client):
        """Action submission to nonexistent simulation returns 404."""
        agent_resp = client.post(
            "/api/v1/agents/register",
            json={"name": "LostAgent", "description": "Test"},
        )
        resp = client.post(
            "/api/v1/simulations/nonexistent/actions",
            json={
                "agent_id": agent_resp.json()["agent_id"],
                "action_type": "noop",
                "payload": {},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {agent_resp.json()['api_key']}"},
        )
        assert resp.status_code == 404

    def test_get_observation_no_data(self, client):
        """Get observation when none available returns no_observation."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        resp = client.get(
            f"/api/v1/simulations/{sim_id}/observation",
            params={"agent_id": agent_ids[0]},
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_observation"
        assert data["observation"] is None

    def test_get_observation_with_data(self, client):
        """Get observation returns stored observation data."""
        import swarm.api.routers.simulations as simulations_mod

        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Inject an observation
        simulations_mod._observations[sim_id] = {
            agent_ids[0]: {"step": 3, "payoff": 0.5},
        }

        resp = client.get(
            f"/api/v1/simulations/{sim_id}/observation",
            params={"agent_id": agent_ids[0]},
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["observation"] == {"step": 3, "payoff": 0.5}

    def test_get_observation_non_participant(self, client):
        """Non-participant cannot get observations."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        outsider = client.post(
            "/api/v1/agents/register",
            json={"name": "Snooper", "description": "Test"},
        )

        resp = client.get(
            f"/api/v1/simulations/{sim_id}/observation",
            params={"agent_id": outsider.json()["agent_id"]},
            headers={"Authorization": f"Bearer {outsider.json()['api_key']}"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# SSE endpoint tests
# ---------------------------------------------------------------------------


class TestExternalAgentOrchestrator:
    """Tests for external agent integration with the orchestrator."""

    def test_external_agent_action_integrated(self):
        """External agent receives observation and submits action via queue."""
        import asyncio

        from swarm.agents.honest import HonestAgent
        from swarm.api.action_queue import AsyncActionQueue
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        async def _run():
            config = OrchestratorConfig(
                n_epochs=1, steps_per_epoch=1, seed=42
            )
            orch = Orchestrator(config=config)

            # Register an internal and an external agent
            internal = HonestAgent(agent_id="internal-1", name="Internal")
            external = HonestAgent(agent_id="external-1", name="External")
            external.is_external = True

            orch.register_agent(internal)
            orch.register_agent(external)

            # Attach action queue
            queue = AsyncActionQueue(timeout_ms=2000)
            orch.set_external_action_queue(queue)

            # Run async — submit action for external agent before it times out
            async def submit_external_action():
                await asyncio.sleep(0.05)
                # Check that observation was stored
                obs = orch.get_external_observations()
                assert "external-1" in obs
                # Submit an action
                await queue.submit_action(
                    "external-1",
                    {"action_type": "noop"},
                )

            submitter = asyncio.create_task(submit_external_action())
            metrics = await orch.run_async()
            await submitter

            assert len(metrics) == 1

        asyncio.run(_run())

    def test_external_agent_timeout_uses_noop(self):
        """External agent that times out gets NOOP fallback."""
        import asyncio

        from swarm.agents.honest import HonestAgent
        from swarm.api.action_queue import AsyncActionQueue
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        async def _run():
            config = OrchestratorConfig(
                n_epochs=1, steps_per_epoch=1, seed=42
            )
            orch = Orchestrator(config=config)

            external = HonestAgent(agent_id="timeout-agent", name="Timeout")
            external.is_external = True
            orch.register_agent(external)

            # Short timeout so test is fast
            queue = AsyncActionQueue(timeout_ms=50)
            orch.set_external_action_queue(queue)

            # Don't submit any action — should timeout gracefully
            metrics = await orch.run_async()
            assert len(metrics) == 1

        asyncio.run(_run())

    def test_mixed_internal_external_step(self):
        """Mix of internal and external agents in same step works correctly."""
        import asyncio

        from swarm.agents.honest import HonestAgent
        from swarm.api.action_queue import AsyncActionQueue
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        async def _run():
            config = OrchestratorConfig(
                n_epochs=1, steps_per_epoch=2, seed=42
            )
            orch = Orchestrator(config=config)

            # 2 internal, 1 external
            for i in range(2):
                orch.register_agent(
                    HonestAgent(agent_id=f"int-{i}", name=f"Int{i}")
                )
            ext = HonestAgent(agent_id="ext-0", name="Ext0")
            ext.is_external = True
            orch.register_agent(ext)

            queue = AsyncActionQueue(timeout_ms=100)
            orch.set_external_action_queue(queue)

            # Auto-submit noop for external agent each step
            async def auto_submit():
                for _ in range(2):  # 2 steps
                    await asyncio.sleep(0.02)
                    await queue.submit_action("ext-0", {"action_type": "noop"})

            submitter = asyncio.create_task(auto_submit())
            metrics = await orch.run_async()
            await submitter

            assert len(metrics) == 1

        asyncio.run(_run())


class TestSSEEndpoint:
    """Tests for the SSE event stream endpoint."""

    @pytest.fixture(autouse=True)
    def _clear_simulation_state(self):
        """Reset simulation storage between tests."""
        import swarm.api.routers.simulations as simulations_mod

        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        yield
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()

    def test_sse_not_found(self, client):
        """SSE endpoint returns 404 for nonexistent simulation."""
        resp = client.get(
            "/api/v1/simulations/nonexistent/events",
            params={"agent_id": "agent-1"},
        )
        assert resp.status_code == 404

    def test_sse_connects(self, client):
        """SSE endpoint returns streaming response for valid simulation."""
        import json
        import threading

        from swarm.api.event_bus import SimEvent, SimEventType, event_bus

        # Create a simulation
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test"},
        )
        sim_id = sim_resp.json()["simulation_id"]

        # We need to publish an event and then a completion event
        # so the stream terminates
        def publish_events():
            import asyncio
            import time

            time.sleep(0.1)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                event_bus.publish(
                    SimEvent(
                        event_type=SimEventType.STEP_COMPLETE,
                        simulation_id=sim_id,
                        data={"step": 0},
                    )
                )
            )
            loop.run_until_complete(
                event_bus.publish(
                    SimEvent(
                        event_type=SimEventType.SIMULATION_COMPLETE,
                        simulation_id=sim_id,
                        data={},
                    )
                )
            )
            loop.close()

        t = threading.Thread(target=publish_events)
        t.start()

        with client.stream(
            "GET",
            f"/api/v1/simulations/{sim_id}/events",
            params={"agent_id": "agent-1"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            lines = []
            for line in resp.iter_lines():
                lines.append(line)

        t.join(timeout=5)

        # Should have received step_complete and simulation_complete events
        event_lines = [line for line in lines if line.startswith("data:")]
        assert len(event_lines) >= 2
        first_data = json.loads(event_lines[0].removeprefix("data:").strip())
        assert first_data["event_type"] == "step_complete"


# ---------------------------------------------------------------------------
# Simulation completion and results tests
# ---------------------------------------------------------------------------


class TestSimulationCompletion:
    """Tests for simulation completion, results, and execution state."""

    @pytest.fixture(autouse=True)
    def _clear_simulation_state(self):
        """Reset simulation storage between tests."""
        import swarm.api.routers.simulations as simulations_mod

        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        simulations_mod._action_history.clear()
        simulations_mod._execution_state.clear()
        simulations_mod._simulation_results.clear()
        yield
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        simulations_mod._action_history.clear()
        simulations_mod._execution_state.clear()
        simulations_mod._simulation_results.clear()

    def _setup_running_simulation(self, client):
        """Create a running simulation with two participants."""
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 5},
        )
        sim_id = sim_resp.json()["simulation_id"]

        agent_ids = []
        api_keys = []
        for i in range(2):
            agent_resp = client.post(
                "/api/v1/agents/register",
                json={"name": f"CompAgent{i}", "description": "Test"},
            )
            data = agent_resp.json()
            agent_ids.append(data["agent_id"])
            api_keys.append(data["api_key"])
            client.post(
                f"/api/v1/simulations/{sim_id}/join",
                json={"agent_id": data["agent_id"], "role": "participant"},
            )

        client.post(f"/api/v1/simulations/{sim_id}/start")
        return sim_id, agent_ids, api_keys

    def test_complete_simulation(self, client):
        """Complete a running simulation stores results."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        resp = client.post(
            f"/api/v1/simulations/{sim_id}/complete",
            json={"toxicity_rate": 0.05, "quality_gap": 0.2},
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

        # Verify simulation is now COMPLETED
        get_resp = client.get(f"/api/v1/simulations/{sim_id}")
        assert get_resp.json()["status"] == "completed"

    def test_complete_non_running_fails(self, client):
        """Cannot complete a simulation that isn't running."""
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test"},
        )
        sim_id = sim_resp.json()["simulation_id"]

        agent_resp = client.post(
            "/api/v1/agents/register",
            json={"name": "EarlyCompleter", "description": "Test"},
        )
        resp = client.post(
            f"/api/v1/simulations/{sim_id}/complete",
            headers={"Authorization": f"Bearer {agent_resp.json()['api_key']}"},
        )
        assert resp.status_code == 400

    def test_get_results(self, client):
        """Get results for a completed simulation."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Submit an action first
        client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": agent_ids[0],
                "action_type": "accept",
                "payload": {},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )

        # Complete
        client.post(
            f"/api/v1/simulations/{sim_id}/complete",
            json={"final_score": 42},
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )

        # Get results
        resp = client.get(
            f"/api/v1/simulations/{sim_id}/results",
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["results"]["final_score"] == 42
        assert data["action_count"] == 1
        assert data["participant_count"] == 2

    def test_get_results_not_completed(self, client):
        """Cannot get results for a non-completed simulation."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        resp = client.get(
            f"/api/v1/simulations/{sim_id}/results",
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 400
        assert "not completed" in resp.json()["detail"]

    def test_execution_state(self, client):
        """Get execution state tracks per-agent actions."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Submit actions
        for i in range(3):
            client.post(
                f"/api/v1/simulations/{sim_id}/actions",
                json={
                    "agent_id": agent_ids[0],
                    "action_type": "noop",
                    "payload": {},
                    "step": i,
                },
                headers={"Authorization": f"Bearer {api_keys[0]}"},
            )
        client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": agent_ids[1],
                "action_type": "accept",
                "payload": {},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {api_keys[1]}"},
        )

        resp = client.get(f"/api/v1/simulations/{sim_id}/execution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_actions"] == 4
        assert data["per_agent_actions"][agent_ids[0]] == 3
        assert data["per_agent_actions"][agent_ids[1]] == 1

    def test_action_history_recorded(self, client):
        """Actions are recorded in history with timestamps."""
        import swarm.api.routers.simulations as simulations_mod

        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        client.post(
            f"/api/v1/simulations/{sim_id}/actions",
            json={
                "agent_id": agent_ids[0],
                "action_type": "accept",
                "payload": {"value": 1},
                "step": 0,
            },
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )

        history = simulations_mod._action_history.get(sim_id, [])
        assert len(history) == 1
        assert history[0]["agent_id"] == agent_ids[0]
        assert history[0]["action_type"] == "accept"
        assert history[0]["step"] == 0
        assert "timestamp" in history[0]
        assert "action_id" in history[0]


# ---------------------------------------------------------------------------
# Governance endpoint tests
# ---------------------------------------------------------------------------


class TestGovernanceEndpoints:
    """Tests for governance proposal endpoints."""

    @pytest.fixture(autouse=True)
    def _clear_governance_state(self):
        """Reset governance storage between tests."""
        import swarm.api.routers.governance as gov_mod

        gov_mod._proposals.clear()
        yield
        gov_mod._proposals.clear()

    def _get_auth_headers(self, client):
        """Register an agent and return auth headers."""
        resp = client.post(
            "/api/v1/agents/register",
            json={"name": "GovAgent", "description": "Governance test"},
        )
        return {"Authorization": f"Bearer {resp.json()['api_key']}"}

    def test_create_proposal(self, client):
        """Create a governance proposal."""
        headers = self._get_auth_headers(client)
        resp = client.post(
            "/api/v1/governance/propose",
            json={
                "title": "Increase tax rate",
                "description": "Propose increasing the tax rate to 0.2",
                "policy_declaration": {"tax_rate": 0.2},
                "target_scenarios": ["baseline"],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Increase tax rate"
        assert data["status"] == "open"
        assert data["policy_declaration"] == {"tax_rate": 0.2}
        assert data["target_scenarios"] == ["baseline"]
        assert data["votes_for"] == 0
        assert data["votes_against"] == 0
        assert "proposal_id" in data

    def test_list_proposals(self, client):
        """List governance proposals."""
        headers = self._get_auth_headers(client)
        for i in range(3):
            client.post(
                "/api/v1/governance/propose",
                json={
                    "title": f"Proposal {i}",
                    "description": f"Description {i}",
                },
                headers=headers,
            )

        resp = client.get("/api/v1/governance/proposals")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_list_proposals_filter_by_status(self, client):
        """Filter proposals by status."""
        headers = self._get_auth_headers(client)
        client.post(
            "/api/v1/governance/propose",
            json={"title": "Open", "description": "Open proposal"},
            headers=headers,
        )

        resp = client.get(
            "/api/v1/governance/proposals",
            params={"status": "open"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.get(
            "/api/v1/governance/proposals",
            params={"status": "accepted"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    def test_get_proposal(self, client):
        """Get a proposal by ID."""
        headers = self._get_auth_headers(client)
        create_resp = client.post(
            "/api/v1/governance/propose",
            json={"title": "GetTest", "description": "Test"},
            headers=headers,
        )
        proposal_id = create_resp.json()["proposal_id"]

        resp = client.get(f"/api/v1/governance/proposals/{proposal_id}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "GetTest"

    def test_get_proposal_not_found(self, client):
        """Get nonexistent proposal returns 404."""
        resp = client.get("/api/v1/governance/proposals/nonexistent")
        assert resp.status_code == 404

    def test_vote_on_proposal(self, client):
        """Vote on a governance proposal."""
        headers = self._get_auth_headers(client)
        create_resp = client.post(
            "/api/v1/governance/propose",
            json={"title": "VoteTest", "description": "Test"},
            headers=headers,
        )
        proposal_id = create_resp.json()["proposal_id"]

        # Vote for
        resp = client.post(
            f"/api/v1/governance/proposals/{proposal_id}/vote",
            params={"direction": 1},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["votes_for"] == 1
        assert resp.json()["votes_against"] == 0

        # Vote against
        resp = client.post(
            f"/api/v1/governance/proposals/{proposal_id}/vote",
            params={"direction": -1},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["votes_for"] == 1
        assert resp.json()["votes_against"] == 1

    def test_vote_not_found(self, client):
        """Vote on nonexistent proposal returns 404."""
        headers = self._get_auth_headers(client)
        resp = client.post(
            "/api/v1/governance/proposals/nonexistent/vote",
            params={"direction": 1},
            headers=headers,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Metrics endpoint tests
# ---------------------------------------------------------------------------


class TestMetricsEndpoints:
    """Tests for per-simulation metrics retrieval."""

    @pytest.fixture(autouse=True)
    def _clear_simulation_state(self):
        """Reset simulation storage between tests."""
        import swarm.api.routers.simulations as simulations_mod

        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        simulations_mod._action_history.clear()
        simulations_mod._execution_state.clear()
        simulations_mod._simulation_results.clear()
        yield
        simulations_mod._simulations.clear()
        simulations_mod._participants.clear()
        simulations_mod._observations.clear()
        simulations_mod._action_queues.clear()
        simulations_mod._action_history.clear()
        simulations_mod._execution_state.clear()
        simulations_mod._simulation_results.clear()

    def _setup_running_simulation(self, client):
        """Create a running simulation with two participants."""
        sim_resp = client.post(
            "/api/v1/simulations/create",
            json={"scenario_id": "test", "max_participants": 5},
        )
        sim_id = sim_resp.json()["simulation_id"]

        agent_ids = []
        api_keys = []
        for i in range(2):
            agent_resp = client.post(
                "/api/v1/agents/register",
                json={"name": f"MetricAgent{i}", "description": "Test"},
            )
            data = agent_resp.json()
            agent_ids.append(data["agent_id"])
            api_keys.append(data["api_key"])
            client.post(
                f"/api/v1/simulations/{sim_id}/join",
                json={"agent_id": data["agent_id"], "role": "participant"},
            )

        client.post(f"/api/v1/simulations/{sim_id}/start")
        return sim_id, agent_ids, api_keys

    def test_get_metrics_running(self, client):
        """Get metrics for a running simulation."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Submit some actions
        for i in range(3):
            client.post(
                f"/api/v1/simulations/{sim_id}/actions",
                json={
                    "agent_id": agent_ids[0],
                    "action_type": "accept",
                    "payload": {},
                    "step": i,
                },
                headers={"Authorization": f"Bearer {api_keys[0]}"},
            )

        resp = client.get(
            f"/api/v1/metrics/{sim_id}",
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == sim_id
        assert data["status"] == "running"
        assert data["total_actions"] == 3
        assert data["per_agent_actions"][agent_ids[0]] == 3
        assert data["action_type_distribution"]["accept"] == 3
        assert data["participant_count"] == 2

    def test_get_metrics_completed(self, client):
        """Get metrics for a completed simulation includes results."""
        sim_id, agent_ids, api_keys = self._setup_running_simulation(client)

        # Complete with results
        client.post(
            f"/api/v1/simulations/{sim_id}/complete",
            json={"toxicity": 0.03},
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )

        resp = client.get(
            f"/api/v1/metrics/{sim_id}",
            headers={"Authorization": f"Bearer {api_keys[0]}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["results"]["toxicity"] == 0.03

    def test_get_metrics_not_found(self, client):
        """Metrics for nonexistent simulation returns 404."""
        agent_resp = client.post(
            "/api/v1/agents/register",
            json={"name": "MetricTest", "description": "Test"},
        )
        resp = client.get(
            "/api/v1/metrics/nonexistent",
            headers={"Authorization": f"Bearer {agent_resp.json()['api_key']}"},
        )
        assert resp.status_code == 404

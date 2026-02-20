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
    yield
    agents_mod._registration_rate.clear()
    agents_mod._registered_agents.clear()


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
    """Tests for governance and metrics router stubs."""

    def test_governance_propose_stub(self, client):
        """POST /api/v1/governance/propose returns not_implemented."""
        resp = client.post("/api/v1/governance/propose")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_implemented"

    def test_governance_proposals_stub(self, client):
        """GET /api/v1/governance/proposals returns not_implemented."""
        resp = client.get("/api/v1/governance/proposals")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_implemented"

    def test_metrics_stub(self, client):
        """GET /api/v1/metrics/{id} returns not_implemented."""
        resp = client.get("/api/v1/metrics/some-sim-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_implemented"


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

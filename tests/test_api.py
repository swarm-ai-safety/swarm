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
    """Reset registration rate-limit state between tests."""
    import swarm.api.routers.agents as agents_mod
    agents_mod._registration_rate.clear()
    yield
    agents_mod._registration_rate.clear()


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

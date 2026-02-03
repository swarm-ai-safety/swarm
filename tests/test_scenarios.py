"""Tests for the scenario loader module."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from src.scenarios import (
    build_orchestrator,
    create_agents,
    load_scenario,
    parse_governance_config,
    parse_payoff_config,
    parse_rate_limits,
)


class TestParseGovernanceConfig:
    """Tests for parse_governance_config."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default config."""
        config = parse_governance_config({})
        assert config.transaction_tax_rate == 0.0
        assert config.reputation_decay_rate == 1.0
        assert not config.staking_enabled
        assert not config.circuit_breaker_enabled
        assert not config.audit_enabled

    def test_parses_transaction_tax(self):
        """Should parse transaction tax settings."""
        data = {
            "transaction_tax_rate": 0.05,
            "transaction_tax_split": 0.7,
        }
        config = parse_governance_config(data)
        assert config.transaction_tax_rate == 0.05
        assert config.transaction_tax_split == 0.7

    def test_legacy_key_support(self):
        """Should support legacy key names."""
        data = {
            "transaction_tax": 0.1,  # Legacy key
            "reputation_decay": 0.9,  # Legacy key
        }
        config = parse_governance_config(data)
        assert config.transaction_tax_rate == 0.1
        assert config.reputation_decay_rate == 0.9

    def test_parses_staking(self):
        """Should parse staking settings."""
        data = {
            "staking_enabled": True,
            "min_stake_to_participate": 25.0,
            "stake_slash_rate": 0.15,
        }
        config = parse_governance_config(data)
        assert config.staking_enabled
        assert config.min_stake_to_participate == 25.0
        assert config.stake_slash_rate == 0.15

    def test_parses_circuit_breaker(self):
        """Should parse circuit breaker settings."""
        data = {
            "circuit_breaker_enabled": True,
            "freeze_threshold_toxicity": 0.6,
            "freeze_threshold_violations": 5,
            "freeze_duration_epochs": 3,
        }
        config = parse_governance_config(data)
        assert config.circuit_breaker_enabled
        assert config.freeze_threshold_toxicity == 0.6
        assert config.freeze_threshold_violations == 5
        assert config.freeze_duration_epochs == 3

    def test_parses_audit(self):
        """Should parse audit settings."""
        data = {
            "audit_enabled": True,
            "audit_probability": 0.2,
            "audit_penalty_multiplier": 3.0,
            "audit_threshold_p": 0.6,
        }
        config = parse_governance_config(data)
        assert config.audit_enabled
        assert config.audit_probability == 0.2
        assert config.audit_penalty_multiplier == 3.0
        assert config.audit_threshold_p == 0.6

    def test_validates_config(self):
        """Should raise on invalid config values."""
        data = {"transaction_tax_rate": 1.5}  # Invalid: > 1.0
        with pytest.raises(ValueError):
            parse_governance_config(data)


class TestParsePayoffConfig:
    """Tests for parse_payoff_config."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default config."""
        config = parse_payoff_config({})
        assert config.s_plus == 2.0
        assert config.s_minus == 1.0
        assert config.h == 2.0

    def test_parses_all_fields(self):
        """Should parse all payoff fields."""
        data = {
            "s_plus": 3.0,
            "s_minus": 1.5,
            "h": 2.5,
            "theta": 0.6,
            "rho_a": 0.1,
            "rho_b": 0.2,
            "w_rep": 1.5,
        }
        config = parse_payoff_config(data)
        assert config.s_plus == 3.0
        assert config.s_minus == 1.5
        assert config.h == 2.5
        assert config.theta == 0.6
        assert config.rho_a == 0.1
        assert config.rho_b == 0.2
        assert config.w_rep == 1.5


class TestParseRateLimits:
    """Tests for parse_rate_limits."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default limits."""
        limits = parse_rate_limits({})
        assert limits.posts_per_epoch == 10
        assert limits.interactions_per_step == 5

    def test_parses_all_fields(self):
        """Should parse all rate limit fields."""
        data = {
            "posts_per_epoch": 20,
            "interactions_per_step": 10,
            "votes_per_epoch": 100,
            "tasks_per_epoch": 5,
        }
        limits = parse_rate_limits(data)
        assert limits.posts_per_epoch == 20
        assert limits.interactions_per_step == 10
        assert limits.votes_per_epoch == 100
        assert limits.tasks_per_epoch == 5


class TestCreateAgents:
    """Tests for create_agents."""

    def test_creates_single_agent(self):
        """Should create a single agent."""
        specs = [{"type": "honest", "count": 1}]
        agents = create_agents(specs)
        assert len(agents) == 1
        assert agents[0].agent_id == "honest_1"

    def test_creates_multiple_agents(self):
        """Should create multiple agents of same type."""
        specs = [{"type": "honest", "count": 3}]
        agents = create_agents(specs)
        assert len(agents) == 3
        assert [a.agent_id for a in agents] == ["honest_1", "honest_2", "honest_3"]

    def test_creates_mixed_agents(self):
        """Should create agents of different types."""
        specs = [
            {"type": "honest", "count": 2},
            {"type": "opportunistic", "count": 1},
            {"type": "deceptive", "count": 1},
            {"type": "adversarial", "count": 1},
        ]
        agents = create_agents(specs)
        assert len(agents) == 5

        types = [a.agent_type.value for a in agents]
        assert types.count("honest") == 2
        assert types.count("opportunistic") == 1
        assert types.count("deceptive") == 1
        assert types.count("adversarial") == 1

    def test_unknown_type_raises(self):
        """Should raise on unknown agent type."""
        specs = [{"type": "unknown", "count": 1}]
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agents(specs)


class TestLoadScenario:
    """Tests for load_scenario."""

    def test_loads_baseline_scenario(self):
        """Should load baseline.yaml successfully."""
        path = Path("scenarios/baseline.yaml")
        if not path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(path)
        assert scenario.scenario_id == "baseline"
        assert len(scenario.agent_specs) > 0
        assert scenario.orchestrator_config.n_epochs == 10

    def test_loads_governance_from_yaml(self):
        """Should parse governance section from YAML."""
        yaml_content = """
scenario_id: test
agents:
  - type: honest
    count: 2

governance:
  transaction_tax_rate: 0.1
  staking_enabled: true
  min_stake_to_participate: 50.0
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.5
  audit_enabled: true
  audit_probability: 0.2

simulation:
  n_epochs: 5
  steps_per_epoch: 5
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))

            gov = scenario.orchestrator_config.governance_config
            assert gov.transaction_tax_rate == 0.1
            assert gov.staking_enabled
            assert gov.min_stake_to_participate == 50.0
            assert gov.circuit_breaker_enabled
            assert gov.freeze_threshold_toxicity == 0.5
            assert gov.audit_enabled
            assert gov.audit_probability == 0.2

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_scenario(Path("nonexistent.yaml"))


class TestBuildOrchestrator:
    """Tests for build_orchestrator."""

    def test_builds_orchestrator_from_scenario(self):
        """Should build a working orchestrator."""
        yaml_content = """
scenario_id: test
agents:
  - type: honest
    count: 2
  - type: opportunistic
    count: 1

governance:
  transaction_tax_rate: 0.05
  reputation_decay_rate: 0.9

simulation:
  n_epochs: 2
  steps_per_epoch: 3
  seed: 42
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))
            orchestrator = build_orchestrator(scenario)

            # Check agents registered
            assert len(orchestrator.get_all_agents()) == 3

            # Check governance engine created
            assert orchestrator.governance_engine is not None
            assert orchestrator.governance_engine.config.transaction_tax_rate == 0.05

            # Check can run
            metrics = orchestrator.run()
            assert len(metrics) == 2


class TestEndToEnd:
    """End-to-end tests with real scenario files."""

    def test_baseline_scenario_runs(self):
        """Baseline scenario should complete successfully."""
        path = Path("scenarios/baseline.yaml")
        if not path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(path)
        orchestrator = build_orchestrator(scenario)

        # Run 2 epochs only for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2
        assert all(m.total_interactions >= 0 for m in metrics)

    def test_status_game_scenario_runs(self):
        """Status game scenario should complete with governance active."""
        path = Path("scenarios/status_game.yaml")
        if not path.exists():
            pytest.skip("status_game.yaml not found")

        scenario = load_scenario(path)
        orchestrator = build_orchestrator(scenario)

        # Verify governance is configured
        gov = orchestrator.governance_engine.config
        assert gov.transaction_tax_rate > 0
        assert gov.staking_enabled
        assert gov.circuit_breaker_enabled
        assert gov.audit_enabled

        # Run 2 epochs for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2

    def test_strict_governance_scenario_runs(self):
        """Strict governance scenario should complete."""
        path = Path("scenarios/strict_governance.yaml")
        if not path.exists():
            pytest.skip("strict_governance.yaml not found")

        scenario = load_scenario(path)
        orchestrator = build_orchestrator(scenario)

        # Verify all levers enabled
        gov = orchestrator.governance_engine.config
        assert gov.transaction_tax_rate == 0.1
        assert gov.reputation_decay_rate == 0.85
        assert gov.vote_normalization_enabled
        assert gov.staking_enabled
        assert gov.circuit_breaker_enabled
        assert gov.audit_enabled

        # Run 2 epochs for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2

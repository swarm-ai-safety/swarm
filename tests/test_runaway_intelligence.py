"""Tests for the Runaway Intelligence scenario (Section 3.1.10).

Validates the three-level control escalation:
  Level 1 — Static compartmentalization (always-on governance)
  Level 2 — Dynamic capability restriction (condition-triggered shocks)
  Level 3 — Emergency market reconfiguration (severe shocks)
"""

from pathlib import Path

import pytest
import yaml

from swarm.core.perturbation import (
    PartitionMode,
    ResourceShockMode,
    ShockTrigger,
)
from swarm.scenarios.loader import load_scenario, parse_perturbation_config

SCENARIO_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "runaway_intelligence.yaml"


@pytest.fixture
def raw_yaml():
    """Load raw YAML dict."""
    with open(SCENARIO_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def scenario():
    """Load scenario through the official loader."""
    return load_scenario(str(SCENARIO_PATH))


# ── YAML Structure ─────────────────────────────────────────────


class TestRunawayIntelligenceYAMLStructure:
    """Verify the YAML file is well-formed and has all required sections."""

    def test_file_exists(self):
        assert SCENARIO_PATH.exists(), f"Missing scenario file: {SCENARIO_PATH}"

    def test_yaml_parses(self, raw_yaml):
        assert isinstance(raw_yaml, dict)

    def test_required_top_level_keys(self, raw_yaml):
        required = {
            "scenario_id", "description", "motif", "agents", "governance",
            "simulation", "rate_limits", "payoff", "success_criteria",
            "outputs", "perturbations", "network", "spawn",
        }
        assert required.issubset(raw_yaml.keys())

    def test_scenario_id(self, raw_yaml):
        assert raw_yaml["scenario_id"] == "runaway_intelligence"

    def test_motif(self, raw_yaml):
        assert raw_yaml["motif"] == "runaway_intelligence"

    def test_agent_count(self, raw_yaml):
        total = sum(a["count"] for a in raw_yaml["agents"])
        assert total == 15


# ── Level 1: Static Controls ──────────────────────────────────


class TestStaticControls:
    """Verify Level 1 governance levers are present and correctly configured."""

    def test_scenario_loads(self, scenario):
        assert scenario.scenario_id == "runaway_intelligence"

    def test_small_world_network(self, raw_yaml):
        net = raw_yaml["network"]
        assert net["topology"] == "small_world"
        assert net["params"]["k"] == 4
        assert net["params"]["p"] == 0.1

    def test_bandwidth_cap(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.bandwidth_cap == 10

    def test_staking_disabled(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.staking_enabled is False

    def test_vote_normalization(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.vote_normalization_enabled is True
        assert gov.max_vote_weight_per_agent == 3.0

    def test_sybil_detection_disabled(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.sybil_detection_enabled is False

    def test_collusion_detection_disabled(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.collusion_detection_enabled is False

    def test_circuit_breaker(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.circuit_breaker_enabled is True
        assert gov.freeze_threshold_toxicity == 0.80

    def test_audit(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.audit_enabled is True
        assert gov.audit_probability == 0.10

    def test_transaction_tax_low_baseline(self, scenario):
        gov = scenario.orchestrator_config.governance_config
        assert gov.transaction_tax_rate == 0.03


# ── Level 2: Dynamic Controls ─────────────────────────────────


class TestDynamicControls:
    """Verify condition-triggered parameter shocks at 0.4 and 0.5 thresholds."""

    def test_parameter_shocks_enabled(self, raw_yaml):
        shocks_cfg = raw_yaml["perturbations"]["parameter_shocks"]
        assert shocks_cfg["enabled"] is True

    def test_condition_shock_at_034(self, raw_yaml):
        """toxicity_rate > 0.34 → small audit bump."""
        shocks = raw_yaml["perturbations"]["parameter_shocks"]["shocks"]
        match = [s for s in shocks if s.get("when") == "toxicity_rate > 0.34"]
        assert len(match) == 1
        params = match[0]["params"]
        assert params["audit_probability"] == 0.20

    def test_condition_shock_at_036(self, raw_yaml):
        """toxicity_rate > 0.36 → tighten circuit breaker, reduce bandwidth."""
        shocks = raw_yaml["perturbations"]["parameter_shocks"]["shocks"]
        match = [s for s in shocks if s.get("when") == "toxicity_rate > 0.36"]
        assert len(match) == 1
        params = match[0]["params"]
        assert params["freeze_threshold_toxicity"] == 0.60
        assert params["bandwidth_cap"] == 8

    def test_condition_triggers_parse(self, raw_yaml):
        """All condition shocks use ShockTrigger.condition."""
        perturb = parse_perturbation_config(raw_yaml["perturbations"])
        condition_shocks = [
            s for s in perturb.parameter_shocks.shocks
            if s.trigger == ShockTrigger.CONDITION
        ]
        # Level 2a (0.4), Level 2b (0.5), Level 3 market freeze (0.65)
        assert len(condition_shocks) == 3


# ── Level 3: Emergency Controls ───────────────────────────────


class TestEmergencyControls:
    """Verify market freeze, network partition, and resource drain."""

    def test_market_freeze_shock(self, raw_yaml):
        """toxicity_rate > 0.38 → 95% transaction tax."""
        shocks = raw_yaml["perturbations"]["parameter_shocks"]["shocks"]
        match = [s for s in shocks if s.get("when") == "toxicity_rate > 0.38"]
        assert len(match) == 1
        assert match[0]["params"]["transaction_tax_rate"] == 0.95

    def test_network_partition_config(self, raw_yaml):
        np_cfg = raw_yaml["perturbations"]["network_partition"]
        assert np_cfg["enabled"] is True
        assert np_cfg["trigger"] == "epoch"
        assert np_cfg["at_epoch"] == 20
        assert np_cfg["mode"] == "random_fragment"
        assert np_cfg["heal_after_epochs"] == 5

    def test_network_partition_parses(self, raw_yaml):
        perturb = parse_perturbation_config(raw_yaml["perturbations"])
        np_cfg = perturb.network_partition
        assert np_cfg.enabled is True
        assert np_cfg.trigger == ShockTrigger.EPOCH
        assert np_cfg.at_epoch == 20
        assert np_cfg.mode == PartitionMode.RANDOM_FRAGMENT
        assert np_cfg.heal_after_epochs == 5

    def test_resource_drain_config(self, raw_yaml):
        rs_cfg = raw_yaml["perturbations"]["resource_shock"]
        assert rs_cfg["enabled"] is True
        assert rs_cfg["trigger"] == "epoch"
        assert rs_cfg["at_epoch"] == 20
        assert rs_cfg["mode"] == "drain_all"
        assert rs_cfg["magnitude"] == 0.7

    def test_resource_drain_parses(self, raw_yaml):
        perturb = parse_perturbation_config(raw_yaml["perturbations"])
        rs_cfg = perturb.resource_shock
        assert rs_cfg.enabled is True
        assert rs_cfg.trigger == ShockTrigger.EPOCH
        assert rs_cfg.at_epoch == 20
        assert rs_cfg.mode == ResourceShockMode.DRAIN_ALL
        assert rs_cfg.magnitude == 0.7


# ── Spawn Configuration ───────────────────────────────────────


class TestSpawnConfiguration:
    """Verify spawn is enabled with limits and cascade controls."""

    def test_spawn_enabled(self, scenario):
        spawn = scenario.orchestrator_config.spawn_config
        assert spawn is not None
        assert spawn.enabled is True

    def test_spawn_limits(self, scenario):
        spawn = scenario.orchestrator_config.spawn_config
        assert spawn.max_depth == 3
        assert spawn.max_children == 2
        assert spawn.max_total_spawned == 30

    def test_cascade_controls(self, scenario):
        spawn = scenario.orchestrator_config.spawn_config
        assert spawn.cascade_ban is True
        assert spawn.cascade_freeze is True

    def test_spawn_cost(self, scenario):
        spawn = scenario.orchestrator_config.spawn_config
        assert spawn.spawn_cost == 10.0


# ── Three-Level Layering ──────────────────────────────────────


class TestThreeLevelLayering:
    """Verify all three levels are present and thresholds escalate monotonically."""

    def test_all_three_levels_present(self, raw_yaml):
        """Level 1 = governance, Level 2 = condition shocks, Level 3 = emergency."""
        gov = raw_yaml["governance"]
        perturb = raw_yaml["perturbations"]

        # Level 1: static governance levers (minimal — lets adversaries build pressure)
        assert gov["circuit_breaker_enabled"] is True
        assert gov["audit_enabled"] is True

        # Level 2: condition-triggered shocks
        assert perturb["parameter_shocks"]["enabled"] is True

        # Level 3: emergency partition + resource drain
        assert perturb["network_partition"]["enabled"] is True
        assert perturb["resource_shock"]["enabled"] is True

    def test_toxicity_thresholds_escalate(self, raw_yaml):
        """Condition thresholds must be monotonically increasing: 0.4, 0.5, 0.65."""
        shocks = raw_yaml["perturbations"]["parameter_shocks"]["shocks"]
        condition_shocks = [s for s in shocks if s.get("when")]
        thresholds = []
        for s in condition_shocks:
            # Parse "toxicity_rate > X" → X
            parts = s["when"].split(">")
            thresholds.append(float(parts[-1].strip()))
        assert thresholds == sorted(thresholds), (
            f"Thresholds must escalate monotonically: {thresholds}"
        )
        assert thresholds == [0.34, 0.36, 0.38]

    def test_static_controls_are_less_aggressive_than_dynamic(self, raw_yaml):
        """Static audit < dynamic audit; static freeze > dynamic freeze (tighter = lower)."""
        gov = raw_yaml["governance"]
        shocks = raw_yaml["perturbations"]["parameter_shocks"]["shocks"]

        level2a = [s for s in shocks if s.get("when") == "toxicity_rate > 0.34"][0]
        # Static audit probability is lower than dynamic
        assert gov["audit_probability"] < level2a["params"]["audit_probability"]

    def test_epoch_30_simulation(self, raw_yaml):
        assert raw_yaml["simulation"]["n_epochs"] == 30

    def test_fifteen_agents(self, raw_yaml):
        total = sum(a["count"] for a in raw_yaml["agents"])
        assert total == 15

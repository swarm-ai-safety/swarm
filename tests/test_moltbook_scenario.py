"""Scenario loading for Moltbook CAPTCHA."""

from pathlib import Path

from swarm.scenarios.loader import build_orchestrator, load_scenario


def test_moltbook_scenario_loads():
    scenario = load_scenario(Path("scenarios/moltbook_captcha.yaml"))
    assert scenario.orchestrator_config.moltbook_config is not None
    orch = build_orchestrator(scenario)
    assert orch._moltbook_handler is not None

"""Scenario loading for Moltipedia heartbeat."""

from pathlib import Path

from swarm.scenarios.loader import build_orchestrator, load_scenario


def test_moltipedia_scenario_loads():
    scenario = load_scenario(Path("scenarios/moltipedia_heartbeat.yaml"))
    assert scenario.orchestrator_config.moltipedia_config is not None
    orchestrator = build_orchestrator(scenario)
    assert orchestrator._moltipedia_handler is not None

"""Tests for the Claude Code bridge runner script."""

import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.run_claude_code_scenario as runner


def _make_scenario() -> SimpleNamespace:
    orchestrator_config = SimpleNamespace(
        log_path=None,
        log_events=False,
        governance_config=SimpleNamespace(),
    )
    return SimpleNamespace(
        scenario_id="demo",
        description="demo",
        orchestrator_config=orchestrator_config,
        rate_limits=SimpleNamespace(),
        agent_specs=[
            {
                "type": "honest",
                "count": 1,
                "name": "agent",
                "config": {},
            }
        ],
    )


def _patch_runner(monkeypatch, scenario):
    class FakeOrchestrator:
        def __init__(self, config):
            self.config = config
            self.state = SimpleNamespace(rate_limits=None)
            self.event_log = None
            self.registered_agents = []

        def register_agent(self, agent):
            self.registered_agents.append(agent)

        def run(self):
            return [
                SimpleNamespace(
                    epoch=1,
                    total_interactions=1,
                    accepted_interactions=1,
                    toxicity_rate=0.0,
                    total_welfare=0.0,
                )
            ]

    class FakeBridge:
        last_instance = None

        def __init__(self, config, event_log=None):
            self.config = config
            self.event_log = event_log
            self.init_calls = []
            FakeBridge.last_instance = self

        def init_session(self, team_name="swarm", cwd=None):
            self.init_calls.append((team_name, cwd))
            return {"initialized": True}

        def shutdown(self):
            return None

    class FakeAgent:
        def __init__(self, agent_id, bridge, config=None, name=None):
            self.agent_id = agent_id
            self.bridge = bridge
            self.config = config or {}
            self.name = name or agent_id

    monkeypatch.setattr(runner, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(runner, "ClaudeCodeBridge", FakeBridge)
    monkeypatch.setattr(runner, "ClaudeCodeAgent", FakeAgent)
    monkeypatch.setattr(runner, "load_scenario", lambda _path: scenario)
    return FakeBridge


def _write_scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "demo.yaml"
    path.write_text("scenario_id: demo\n")
    return path


def test_is_loopback_url():
    assert runner._is_loopback_url("http://localhost:3100")
    assert runner._is_loopback_url("http://127.0.0.1:3100")
    assert runner._is_loopback_url("http://[::1]:3100")
    assert not runner._is_loopback_url("http://example.com:3100")


def test_auto_approve_default_false(monkeypatch, tmp_path):
    scenario = _make_scenario()
    FakeBridge = _patch_runner(monkeypatch, scenario)
    scenario_path = _write_scenario_file(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--scenario", str(scenario_path)],
    )

    rc = runner.main()
    assert rc == 0
    assert FakeBridge.last_instance.config.auto_respond_governance is False


def test_auto_approve_loopback_enabled(monkeypatch, tmp_path):
    scenario = _make_scenario()
    FakeBridge = _patch_runner(monkeypatch, scenario)
    scenario_path = _write_scenario_file(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--scenario", str(scenario_path), "--auto-approve"],
    )

    rc = runner.main()
    assert rc == 0
    assert FakeBridge.last_instance.config.auto_respond_governance is True


def test_auto_approve_non_loopback_disabled(monkeypatch, tmp_path, capsys):
    scenario = _make_scenario()
    FakeBridge = _patch_runner(monkeypatch, scenario)
    scenario_path = _write_scenario_file(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--scenario",
            str(scenario_path),
            "--auto-approve",
            "--base-url",
            "http://example.com:3100",
        ],
    )

    rc = runner.main()
    assert rc == 0
    assert FakeBridge.last_instance.config.auto_respond_governance is False
    err = capsys.readouterr().err
    assert "auto-approve is disabled" in err


def test_api_key_warning(monkeypatch, tmp_path, capsys):
    scenario = _make_scenario()
    FakeBridge = _patch_runner(monkeypatch, scenario)
    scenario_path = _write_scenario_file(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--scenario",
            str(scenario_path),
            "--api-key",
            "secret",
        ],
    )

    rc = runner.main()
    assert rc == 0
    err = capsys.readouterr().err
    assert "Warning: --api-key" in err
    assert FakeBridge.last_instance.config.client_config.api_key == "secret"

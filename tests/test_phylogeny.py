"""Tests for the animated phylogeny visualization module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from swarm.analysis.phylogeny import (
    extract_agent_trajectories,
    extract_ecosystem_metrics,
    generate_phylogeny_html,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(events: list, path: Path) -> None:
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _basic_events() -> list:
    """Minimal two-epoch, two-agent event stream."""
    return [
        {"event_type": "agent_created", "agent_id": "a1", "payload": {"agent_type": "honest"}, "epoch": 0, "step": 0},
        {"event_type": "agent_created", "agent_id": "a2", "payload": {"agent_type": "deceptive"}, "epoch": 0, "step": 0},
        {"event_type": "simulation_started", "payload": {"n_epochs": 2, "n_agents": 2}, "epoch": None, "step": None},
        {"event_type": "reputation_updated", "agent_id": "a1", "payload": {"new_reputation": 0.5, "delta": 0.5}, "epoch": 0, "step": 1},
        {"event_type": "payoff_computed", "initiator_id": "a1", "counterparty_id": "a2", "payload": {"payoff_initiator": 1.0, "payoff_counterparty": 0.5, "components": {"p": 0.8}}, "epoch": 0, "step": 1},
        {"event_type": "epoch_completed", "payload": {"epoch": 0}, "epoch": 0, "step": None},
        {"event_type": "reputation_updated", "agent_id": "a1", "payload": {"new_reputation": 0.7, "delta": 0.2}, "epoch": 1, "step": 1},
        {"event_type": "reputation_updated", "agent_id": "a2", "payload": {"new_reputation": 0.3, "delta": 0.3}, "epoch": 1, "step": 1},
        {"event_type": "payoff_computed", "initiator_id": "a2", "counterparty_id": "a1", "payload": {"payoff_initiator": 0.6, "payoff_counterparty": 0.4, "components": {"p": 0.6}}, "epoch": 1, "step": 2},
        {"event_type": "epoch_completed", "payload": {"epoch": 1}, "epoch": 1, "step": None},
    ]


# ---------------------------------------------------------------------------
# Agent trajectory extraction
# ---------------------------------------------------------------------------


class TestExtractTrajectories:
    def test_basic_extraction(self, tmp_path: Path) -> None:
        _write_jsonl(_basic_events(), tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        assert result["n_agents"] == 2
        assert result["n_epochs"] == 2
        assert "a1" in result["agents"]
        assert "a2" in result["agents"]
        assert result["agents"]["a1"]["agent_type"] == "honest"
        assert result["agents"]["a2"]["agent_type"] == "deceptive"

    def test_epoch_values(self, tmp_path: Path) -> None:
        _write_jsonl(_basic_events(), tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        a1_e0 = result["agents"]["a1"]["epochs"][0]
        assert a1_e0["reputation"] == pytest.approx(0.5)
        assert a1_e0["cumulative_payoff"] == pytest.approx(1.0)
        assert a1_e0["n_interactions"] == 1
        assert a1_e0["avg_p"] == pytest.approx(0.8)

        # Epoch 1: cumulative payoff = 1.0 (epoch 0) + 0.4 (counterparty)
        a1_e1 = result["agents"]["a1"]["epochs"][1]
        assert a1_e1["reputation"] == pytest.approx(0.7)
        assert a1_e1["cumulative_payoff"] == pytest.approx(1.4)
        assert a1_e1["n_interactions"] == 1
        assert a1_e1["avg_p"] == pytest.approx(0.6)

    def test_counterparty_values(self, tmp_path: Path) -> None:
        _write_jsonl(_basic_events(), tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        # a2: epoch 0 no reputation update -> 0.0, payoff = 0.5 (counterparty)
        a2_e0 = result["agents"]["a2"]["epochs"][0]
        assert a2_e0["reputation"] == pytest.approx(0.0)
        assert a2_e0["cumulative_payoff"] == pytest.approx(0.5)

        # a2: epoch 1 rep = 0.3, cumulative = 0.5 + 0.6 = 1.1
        a2_e1 = result["agents"]["a2"]["epochs"][1]
        assert a2_e1["reputation"] == pytest.approx(0.3)
        assert a2_e1["cumulative_payoff"] == pytest.approx(1.1)

    def test_multiple_replays_uses_last(self, tmp_path: Path) -> None:
        replay1 = [
            {"event_type": "agent_created", "agent_id": "x1", "payload": {"agent_type": "adversarial"}, "epoch": 0, "step": 0},
            {"event_type": "simulation_started", "payload": {"n_epochs": 1, "n_agents": 1}, "epoch": None, "step": None},
            {"event_type": "epoch_completed", "payload": {"epoch": 0}, "epoch": 0, "step": None},
        ]
        replay2 = [
            {"event_type": "agent_created", "agent_id": "y1", "payload": {"agent_type": "honest"}, "epoch": 0, "step": 0},
            {"event_type": "simulation_started", "payload": {"n_epochs": 1, "n_agents": 1}, "epoch": None, "step": None},
            {"event_type": "epoch_completed", "payload": {"epoch": 0}, "epoch": 0, "step": None},
        ]
        _write_jsonl(replay1 + replay2, tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        assert "y1" in result["agents"]
        assert "x1" not in result["agents"]
        assert result["agents"]["y1"]["agent_type"] == "honest"

    def test_cumulative_payoff_accumulates(self, tmp_path: Path) -> None:
        events = [
            {"event_type": "agent_created", "agent_id": "a1", "payload": {"agent_type": "honest"}, "epoch": 0, "step": 0},
            {"event_type": "agent_created", "agent_id": "a2", "payload": {"agent_type": "honest"}, "epoch": 0, "step": 0},
            {"event_type": "simulation_started", "payload": {"n_epochs": 2, "n_agents": 2}, "epoch": None, "step": None},
            {"event_type": "payoff_computed", "initiator_id": "a1", "counterparty_id": "a2", "payload": {"payoff_initiator": 1.0, "payoff_counterparty": 0.5, "components": {"p": 0.7}}, "epoch": 0, "step": 1},
            {"event_type": "payoff_computed", "initiator_id": "a1", "counterparty_id": "a2", "payload": {"payoff_initiator": 2.0, "payoff_counterparty": 1.0, "components": {"p": 0.9}}, "epoch": 0, "step": 2},
            {"event_type": "epoch_completed", "payload": {"epoch": 0}, "epoch": 0, "step": None},
            {"event_type": "payoff_computed", "initiator_id": "a1", "counterparty_id": "a2", "payload": {"payoff_initiator": 0.5, "payoff_counterparty": 0.3, "components": {"p": 0.6}}, "epoch": 1, "step": 1},
            {"event_type": "epoch_completed", "payload": {"epoch": 1}, "epoch": 1, "step": None},
        ]
        _write_jsonl(events, tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        a1 = result["agents"]["a1"]
        # Epoch 0: 1.0 + 2.0 = 3.0
        assert a1["epochs"][0]["cumulative_payoff"] == pytest.approx(3.0)
        assert a1["epochs"][0]["n_interactions"] == 2
        assert a1["epochs"][0]["avg_p"] == pytest.approx(0.8)  # (0.7 + 0.9) / 2

        # Epoch 1: 3.0 + 0.5 = 3.5
        assert a1["epochs"][1]["cumulative_payoff"] == pytest.approx(3.5)
        assert a1["epochs"][1]["n_interactions"] == 1

        # a2 counterparty totals
        a2 = result["agents"]["a2"]
        assert a2["epochs"][0]["cumulative_payoff"] == pytest.approx(1.5)  # 0.5 + 1.0
        assert a2["epochs"][1]["cumulative_payoff"] == pytest.approx(1.8)  # 1.5 + 0.3

    def test_carry_forward_inactive_agent(self, tmp_path: Path) -> None:
        events = [
            {"event_type": "agent_created", "agent_id": "a1", "payload": {"agent_type": "honest"}, "epoch": 0, "step": 0},
            {"event_type": "agent_created", "agent_id": "a2", "payload": {"agent_type": "rlm"}, "epoch": 0, "step": 0},
            {"event_type": "simulation_started", "payload": {"n_epochs": 2, "n_agents": 2}, "epoch": None, "step": None},
            {"event_type": "reputation_updated", "agent_id": "a1", "payload": {"new_reputation": 0.5, "delta": 0.5}, "epoch": 0, "step": 1},
            {"event_type": "epoch_completed", "payload": {"epoch": 0}, "epoch": 0, "step": None},
            {"event_type": "epoch_completed", "payload": {"epoch": 1}, "epoch": 1, "step": None},
        ]
        _write_jsonl(events, tmp_path / "events.jsonl")
        result = extract_agent_trajectories(tmp_path / "events.jsonl")

        # a2 never had activity; should have default values
        a2 = result["agents"]["a2"]
        assert len(a2["epochs"]) == 2
        assert a2["epochs"][0]["reputation"] == 0.0
        assert a2["epochs"][0]["cumulative_payoff"] == 0.0
        # Epoch 1 carries forward
        assert a2["epochs"][1]["reputation"] == 0.0
        assert a2["epochs"][1]["cumulative_payoff"] == 0.0

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "empty.jsonl").write_text("")
        result = extract_agent_trajectories(tmp_path / "empty.jsonl")
        assert result["n_agents"] == 0
        assert result["n_epochs"] == 0
        assert result["agents"] == {}


# ---------------------------------------------------------------------------
# Ecosystem metrics extraction
# ---------------------------------------------------------------------------


class TestExtractEcosystem:
    def test_csv_extraction(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "epochs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "simulation_id", "epoch", "toxicity_rate",
                "ecosystem_threat_level", "quality_gap",
                "n_agents", "n_frozen", "total_welfare",
            ])
            writer.writerow(["test", "0", "0.1", "0.0", "0.05", "5", "0", "10.0"])
            writer.writerow(["test", "1", "0.2", "0.1", "0.03", "5", "1", "8.5"])

        result = extract_ecosystem_metrics(csv_path)

        assert len(result["epochs"]) == 2
        assert result["epochs"][0]["epoch"] == 0
        assert result["epochs"][0]["toxicity_rate"] == pytest.approx(0.1)
        assert result["epochs"][1]["n_frozen"] == 1
        assert result["epochs"][1]["total_welfare"] == pytest.approx(8.5)

    def test_json_extraction(self, tmp_path: Path) -> None:
        json_path = tmp_path / "history.json"
        data = {
            "epoch_snapshots": [
                {"epoch": 0, "toxicity_rate": 0.15, "ecosystem_threat_level": 0.0,
                 "quality_gap": 0.0, "n_agents": 3, "n_frozen": 0, "total_welfare": 7.0},
            ]
        }
        json_path.write_text(json.dumps(data))
        result = extract_ecosystem_metrics(json_path)

        assert len(result["epochs"]) == 1
        assert result["epochs"][0]["toxicity_rate"] == pytest.approx(0.15)
        assert result["epochs"][0]["n_agents"] == 3


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


class TestHtmlGeneration:
    def _make_data(self):
        agent_data = {
            "agents": {
                "a1": {
                    "agent_type": "honest",
                    "epochs": [{
                        "epoch": 0, "reputation": 0.5,
                        "cumulative_payoff": 1.0, "n_interactions": 1,
                        "avg_p": 0.7, "is_frozen": False,
                    }],
                },
            },
            "n_epochs": 1,
            "n_agents": 1,
        }
        eco_data = {
            "epochs": [{
                "epoch": 0, "toxicity_rate": 0.1,
                "ecosystem_threat_level": 0.0, "quality_gap": 0.0,
                "n_agents": 1, "n_frozen": 0, "total_welfare": 5.0,
            }],
        }
        return agent_data, eco_data

    def test_html_contains_canvas(self, tmp_path: Path) -> None:
        agent_data, eco_data = self._make_data()
        out = generate_phylogeny_html(agent_data, eco_data, tmp_path / "test.html")
        html = out.read_text()
        assert "<canvas" in html

    def test_html_embeds_agent_data(self, tmp_path: Path) -> None:
        agent_data, eco_data = self._make_data()
        out = generate_phylogeny_html(agent_data, eco_data, tmp_path / "test.html")
        html = out.read_text()
        assert '"honest"' in html
        assert '"a1"' in html

    def test_embedded_json_is_parseable(self, tmp_path: Path) -> None:
        agent_data, eco_data = self._make_data()
        out = generate_phylogeny_html(agent_data, eco_data, tmp_path / "test.html")
        html = out.read_text()

        # Extract the AGENT_DATA JSON from the script
        marker = "var AGENT_DATA = "
        start = html.index(marker) + len(marker)
        # Find the matching semicolon
        depth = 0
        end = start
        for i in range(start, len(html)):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        embedded = html[start:end]
        parsed = json.loads(embedded)
        assert parsed["n_agents"] == 1
        assert "a1" in parsed["agents"]

    def test_html_creates_parent_dirs(self, tmp_path: Path) -> None:
        agent_data, eco_data = self._make_data()
        out = generate_phylogeny_html(
            agent_data, eco_data, tmp_path / "sub" / "dir" / "test.html"
        )
        assert out.exists()

    def test_custom_title(self, tmp_path: Path) -> None:
        agent_data, eco_data = self._make_data()
        out = generate_phylogeny_html(
            agent_data, eco_data, tmp_path / "test.html",
            title="My Custom Title",
        )
        html = out.read_text()
        assert "My Custom Title" in html

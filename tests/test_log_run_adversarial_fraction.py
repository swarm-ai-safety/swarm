"""Tests for ``adversarial_fraction`` extraction in ``swarm.scripts.log_run``.

Covers:
  - direct extraction from ``agent_snapshots`` in a history.json payload
  - deduplication across epochs (each agent appears once per epoch)
  - graceful fallback when ``agent_snapshots`` is missing / malformed
  - CSV-based fallback via ``*agents*.csv`` sidecar
  - end-to-end ``extract_from_history`` against a synthetic history.json

These tests deliberately do **not** spin up a real simulation: the unit
under test only parses serialized payloads.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from swarm.scripts.log_run import (
    ADVERSARIAL_AGENT_TYPES,
    _adversarial_fraction_from_agents_csv,
    _compute_adversarial_fraction,
    extract_from_history,
)

# ---------------------------------------------------------------------------
# Pure helper: _compute_adversarial_fraction
# ---------------------------------------------------------------------------

def test_compute_adversarial_fraction_basic_mix():
    """3 of 10 agents are adversarial/deceptive → 0.3."""
    snaps = (
        [{"agent_id": f"agent_{i}", "agent_type": "honest", "epoch": 0} for i in range(7)]
        + [{"agent_id": "agent_7", "agent_type": "adversarial", "epoch": 0}]
        + [{"agent_id": "agent_8", "agent_type": "deceptive", "epoch": 0}]
        + [{"agent_id": "agent_9", "agent_type": "adversarial", "epoch": 0}]
    )
    assert _compute_adversarial_fraction(snaps) == pytest.approx(0.3)


def test_compute_adversarial_fraction_deduplicates_across_epochs():
    """Same 4 agents over 5 epochs (20 records) must still yield correct fraction."""
    snaps = []
    for epoch in range(5):
        snaps.append({"agent_id": "a0", "agent_type": "honest", "epoch": epoch})
        snaps.append({"agent_id": "a1", "agent_type": "honest", "epoch": epoch})
        snaps.append({"agent_id": "a2", "agent_type": "adversarial", "epoch": epoch})
        snaps.append({"agent_id": "a3", "agent_type": "deceptive", "epoch": epoch})

    # 2/4 unique agents are adversarial → 0.5
    assert _compute_adversarial_fraction(snaps) == pytest.approx(0.5)


@pytest.mark.parametrize("variant", ["ADVERSARIAL", "Adversarial", "adversarial"])
def test_compute_adversarial_fraction_case_insensitive(variant: str):
    """Accept enum-value-cased and uppercase strings; the export currently
    emits ``.value`` (lowercase), but tolerate older dumps."""
    snaps = [
        {"agent_id": "a", "agent_type": variant, "epoch": 0},
        {"agent_id": "b", "agent_type": "honest", "epoch": 0},
    ]
    assert _compute_adversarial_fraction(snaps) == pytest.approx(0.5)


def test_compute_adversarial_fraction_empty_list_returns_zero():
    """Older history.json files may omit per-agent snapshots entirely."""
    assert _compute_adversarial_fraction([]) == 0.0


def test_compute_adversarial_fraction_missing_fields_returns_zero():
    """Records without usable agent_id / agent_type are skipped, not crashed on."""
    snaps = [
        {"agent_id": None, "agent_type": "adversarial"},
        {"agent_id": "", "agent_type": "adversarial"},
        {"agent_id": "a", "agent_type": None},
    ]
    assert _compute_adversarial_fraction(snaps) == 0.0


def test_compute_adversarial_fraction_all_adversarial():
    snaps = [
        {"agent_id": f"a{i}", "agent_type": "adversarial"} for i in range(4)
    ]
    assert _compute_adversarial_fraction(snaps) == 1.0


def test_compute_adversarial_fraction_no_adversarial():
    snaps = [{"agent_id": f"a{i}", "agent_type": "honest"} for i in range(4)]
    assert _compute_adversarial_fraction(snaps) == 0.0


def test_adversarial_agent_types_is_a_frozenset():
    """Guard against accidental mutation of the canonical set."""
    assert isinstance(ADVERSARIAL_AGENT_TYPES, frozenset)
    assert "adversarial" in ADVERSARIAL_AGENT_TYPES
    assert "deceptive" in ADVERSARIAL_AGENT_TYPES
    assert "honest" not in ADVERSARIAL_AGENT_TYPES


# ---------------------------------------------------------------------------
# extract_from_history: end-to-end on synthetic JSON
# ---------------------------------------------------------------------------

def _write_history_json(path: Path, agent_snapshots: list) -> None:
    """Build a minimal history.json payload that ``extract_from_history`` accepts."""
    payload = {
        "simulation_id": "test_sim",
        "n_agents": len({s["agent_id"] for s in agent_snapshots}),
        "steps_per_epoch": 1,
        "seed": 42,
        "epoch_snapshots": [
            {
                "total_interactions": 10,
                "accepted_interactions": 8,
                "toxicity_rate": 0.1,
                "total_welfare": 1.0,
            }
        ],
        "agent_snapshots": agent_snapshots,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_from_history_includes_adversarial_fraction(tmp_path: Path):
    """extract_from_history must surface adversarial_fraction from snapshots."""
    snaps = [
        {"agent_id": "a0", "agent_type": "honest", "epoch": 0},
        {"agent_id": "a1", "agent_type": "honest", "epoch": 0},
        {"agent_id": "a2", "agent_type": "adversarial", "epoch": 0},
        {"agent_id": "a3", "agent_type": "deceptive", "epoch": 0},
    ]
    history_path = tmp_path / "history.json"
    _write_history_json(history_path, snaps)

    summary = extract_from_history(history_path)

    assert summary["adversarial_fraction"] == pytest.approx(0.5)
    # Other fields must continue to be populated as before.
    assert summary["scenario_id"] == "test_sim"
    assert summary["total_interactions"] == 10


def test_extract_from_history_legacy_no_agent_snapshots(tmp_path: Path):
    """history.json without ``agent_snapshots`` falls back to 0.0 (backwards compat)."""
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "simulation_id": "legacy",
                "n_agents": 3,
                "steps_per_epoch": 1,
                "seed": 1,
                "epoch_snapshots": [
                    {
                        "total_interactions": 1,
                        "accepted_interactions": 1,
                        "toxicity_rate": 0.0,
                        "total_welfare": 0.0,
                    }
                ],
                # NOTE: no "agent_snapshots" key
            }
        ),
        encoding="utf-8",
    )

    summary = extract_from_history(history_path)
    assert summary["adversarial_fraction"] == 0.0


# ---------------------------------------------------------------------------
# CSV fallback path
# ---------------------------------------------------------------------------

def test_adversarial_fraction_from_agents_csv(tmp_path: Path):
    """Sidecar agents CSV: per-row agent_type column → correct fraction."""
    csv_path = tmp_path / "run_agents.csv"
    fieldnames = ["agent_id", "epoch", "agent_type"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # 5 agents over 2 epochs (10 rows), 2 adversarial → 2/5 = 0.4
        for epoch in range(2):
            w.writerow({"agent_id": "a0", "epoch": epoch, "agent_type": "honest"})
            w.writerow({"agent_id": "a1", "epoch": epoch, "agent_type": "honest"})
            w.writerow({"agent_id": "a2", "epoch": epoch, "agent_type": "honest"})
            w.writerow({"agent_id": "a3", "epoch": epoch, "agent_type": "adversarial"})
            w.writerow({"agent_id": "a4", "epoch": epoch, "agent_type": "deceptive"})

    assert _adversarial_fraction_from_agents_csv(tmp_path) == pytest.approx(0.4)


def test_adversarial_fraction_from_agents_csv_no_sidecar(tmp_path: Path):
    """Legacy CSV-only runs without an agents.csv sidecar: 0.0 fallback."""
    (tmp_path / "run_epochs.csv").write_text(
        "total_interactions,accepted_interactions\n1,1\n", encoding="utf-8"
    )
    assert _adversarial_fraction_from_agents_csv(tmp_path) == 0.0

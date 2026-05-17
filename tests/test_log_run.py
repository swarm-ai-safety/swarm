import csv
import json

from swarm.scripts.log_run import extract_from_csv, extract_from_history


def _write_history(path, agent_snapshots):
    path.write_text(
        json.dumps(
            {
                "epoch_snapshots": [
                    {
                        "total_interactions": 0,
                        "accepted_interactions": 0,
                        "toxicity_rate": 0.0,
                        "total_welfare": 1.0,
                        "gini_reputation": 0.0,
                        "gini_resources": 0.0,
                    }
                ],
                "agent_snapshots": agent_snapshots,
                "n_agents": 0,
            }
        )
    )


def test_extract_from_history_computes_final_epoch_adversarial_fraction(tmp_path):
    history_path = tmp_path / "history.json"
    _write_history(
        history_path,
        [
            {"agent_id": "a1", "epoch": 0, "agent_type": "honest"},
            {"agent_id": "a2", "epoch": 0, "agent_type": "adversarial"},
            {"agent_id": "a1", "epoch": 1, "agent_type": "honest"},
            {"agent_id": "a2", "epoch": 1, "agent_type": "adversarial"},
            {"agent_id": "a3", "epoch": 1, "agent_type": "honest"},
        ],
    )

    summary = extract_from_history(history_path)

    assert summary["adversarial_fraction"] == 0.3333


def test_extract_from_history_matches_case_insensitive_and_enum_like_types(tmp_path):
    history_path = tmp_path / "history.json"
    _write_history(
        history_path,
        [
            {"agent_id": "a1", "epoch": 0, "agent_type": "AgentType.ADVERSARIAL"},
            {"agent_id": "a2", "epoch": 0, "agent_type": "redteam-specialist"},
            {"agent_id": "a3", "epoch": 0, "agent_type": "HONEST"},
            {"agent_id": "a4", "epoch": 0, "agent_type": None},
        ],
    )

    summary = extract_from_history(history_path)

    assert summary["adversarial_fraction"] == 0.5


def test_extract_from_history_supports_mapping_agent_snapshots(tmp_path):
    history_path = tmp_path / "history.json"
    _write_history(
        history_path,
        {
            "a1": [{"epoch": 0, "agent_type": "honest"}],
            "a2": [{"epoch": 0, "agent_type": "redteam"}],
        },
    )

    summary = extract_from_history(history_path)

    assert summary["adversarial_fraction"] == 0.5


def test_extract_from_history_without_agent_snapshots_returns_zero(tmp_path):
    history_path = tmp_path / "history.json"
    _write_history(history_path, [])

    summary = extract_from_history(history_path)

    assert summary["adversarial_fraction"] == 0.0


def test_extract_from_csv_computes_adversarial_fraction_from_agents_csv(tmp_path):
    epoch_path = tmp_path / "epochs.csv"
    with open(epoch_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total_interactions",
                "accepted_interactions",
                "gini_reputation",
                "gini_resources",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "0",
                "total_interactions": "0",
                "accepted_interactions": "0",
                "gini_reputation": "0.0",
                "gini_resources": "0.0",
            }
        )

    agents_path = tmp_path / "agent_snapshots.csv"
    with open(agents_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "epoch", "agent_type"])
        writer.writeheader()
        writer.writerow({"agent_id": "a1", "epoch": "0", "agent_type": "honest"})
        writer.writerow({"agent_id": "a2", "epoch": "0", "agent_type": "redteam"})
        writer.writerow({"agent_id": "a3", "epoch": "0", "agent_type": "adversarial"})
        writer.writerow({"agent_id": "a1", "epoch": "1", "agent_type": "honest"})
        writer.writerow({"agent_id": "a2", "epoch": "1", "agent_type": "honest"})
        writer.writerow({"agent_id": "a3", "epoch": "1", "agent_type": "adversarial"})

    summary = extract_from_csv(tmp_path)

    assert summary["adversarial_fraction"] == 0.3333


def test_extract_from_csv_without_agents_csv_returns_zero(tmp_path):
    epoch_path = tmp_path / "epochs.csv"
    with open(epoch_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total_interactions",
                "accepted_interactions",
                "gini_reputation",
                "gini_resources",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "0",
                "total_interactions": "0",
                "accepted_interactions": "0",
                "gini_reputation": "0.0",
                "gini_resources": "0.0",
            }
        )

    summary = extract_from_csv(tmp_path)

    assert summary["adversarial_fraction"] == 0.0

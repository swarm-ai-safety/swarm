"""Deterministic offline tests for the SWARM->MiroShark mapper."""

from pathlib import Path

import pytest
import yaml

from swarm.bridges.miroshark.mapper import scenario_to_briefing


@pytest.fixture
def small_scenario(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "scenario_id": "tiny",
                "description": "A minimal red-team scenario.",
                "motif": "audit_evasion",
                "agents": [
                    {"type": "honest", "count": 2, "name": "honest"},
                    {"type": "adaptive_adversary", "count": 1, "name": "evader"},
                ],
                "governance": {
                    "audit_enabled": True,
                    "audit_probability": 0.2,
                    "audit_penalty_multiplier": 3.0,
                    "reputation_decay_rate": 0.9,
                    "staking_enabled": True,
                    "min_stake_to_participate": 5.0,
                    "stake_slash_rate": 0.2,
                    "transaction_tax_rate": 0.05,
                },
                "payoff": {"s_plus": 2.0, "s_minus": 1.0, "h": 2.5, "theta": 0.5, "w_rep": 1.5},
                "simulation": {"n_epochs": 10},
                "success_criteria": {"max_avg_toxicity": 0.4},
            }
        )
    )
    return p


def test_mapper_scales_agent_counts(small_scenario: Path) -> None:
    seed, req, sc = scenario_to_briefing(small_scenario, scale=10)
    assert sc["scenario_id"] == "tiny"
    # 2 honest + 1 adversary, scale=10 → 30 total
    assert "**20 × honest**" in seed
    assert "**10 × evader**" in seed
    assert "30 agents" in seed
    assert "Simulate 30 agents" in req


def test_mapper_emits_named_roster(small_scenario: Path) -> None:
    seed, _, _ = scenario_to_briefing(small_scenario, scale=4)
    assert "## Named Roster" in seed
    # 4×(2+1) = 12 named lines (one per agent). Roster lines start with "- **".
    roster_lines = [
        line for line in seed.splitlines() if line.startswith("- **") and "—" in line
    ]
    assert len(roster_lines) == 12


def test_mapper_governance_block_includes_active_levers(small_scenario: Path) -> None:
    seed, _, _ = scenario_to_briefing(small_scenario, scale=2)
    assert "audits" in seed
    assert "reputation decays" in seed
    assert "staking" in seed
    assert "5%" in seed  # transaction tax


def test_mapper_dry_run_with_real_scenarios() -> None:
    """The mapper must accept every adversarial scenario without crashing."""
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        "adversarial_redteam.yaml",
        "adversarial_trust_building.yaml",
        "casestudy_libel_cascade.yaml",
        "casestudy_dual_use_coordination.yaml",
        "collusion_detection.yaml",
    ]
    for name in targets:
        path = repo_root / "scenarios" / name
        if not path.exists():
            continue
        seed, req, sc = scenario_to_briefing(path, scale=2)
        assert sc.get("scenario_id"), f"{name} missing scenario_id"
        assert "## Agent Population" in seed
        assert "Simulate" in req

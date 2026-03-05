"""Tests for eval runner, metrics, scoring."""

import json
import tempfile
from pathlib import Path

import pytest

import swarm_gym.envs.escalation_ladder  # noqa: F401

from swarm_gym.envs.registry import make
from swarm_gym.agents.scripted import MixedPopulation
from swarm_gym.governance.tax import TaxPolicy
from swarm_gym.governance.circuit_breaker import CircuitBreakerPolicy
from swarm_gym.eval.runner import run_eval
from swarm_gym.eval.metrics import aggregate_outcomes
from swarm_gym.eval.scoring import compute_scorecard


class TestEvalRunner:
    def test_run_produces_outputs(self):
        env = make("swarm/escalation_ladder:v1", num_agents=3, episode_len=5)
        env.set_governance([TaxPolicy(rate=0.05), CircuitBreakerPolicy(threshold=0.9)])
        policy = MixedPopulation()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary = run_eval(
                env=env,
                policy=policy,
                seeds=[0, 1],
                out_dir=out_dir,
                benchmark_name="test_escalation",
                governance_preset="moderate",
            )

            # Check files
            assert (out_dir / "episodes.jsonl").exists()
            assert (out_dir / "summary.json").exists()
            assert (out_dir / "manifest.json").exists()
            assert (out_dir / "traces").is_dir()

            # Check JSONL
            with open(out_dir / "episodes.jsonl") as f:
                lines = f.readlines()
            assert len(lines) == 2  # 2 seeds x 1 persona

            ep = json.loads(lines[0])
            assert "run_id" in ep
            assert "episode_id" in ep
            assert "outcomes" in ep
            assert "governance" in ep
            assert "agent_population" in ep

            # Check summary
            assert "scorecard" in summary
            assert "SAFETY" in summary["scorecard"]
            assert "OVERALL" in summary["scorecard"]
            assert summary["episodes"] == 2


class TestMetrics:
    def test_aggregate_numeric(self):
        outcomes = [
            {"cooperation_rate": 0.5, "welfare_total": 100.0},
            {"cooperation_rate": 0.7, "welfare_total": 200.0},
        ]
        agg = aggregate_outcomes(outcomes)
        assert agg["mean_cooperation_rate"] == pytest.approx(0.6)
        assert agg["mean_welfare_total"] == pytest.approx(150.0)

    def test_aggregate_boolean(self):
        outcomes = [
            {"catastrophe": True},
            {"catastrophe": False},
            {"catastrophe": False},
        ]
        agg = aggregate_outcomes(outcomes)
        assert agg["catastrophe_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_aggregate_empty(self):
        assert aggregate_outcomes([]) == {}


class TestScoring:
    def test_scorecard_range(self):
        agg = {
            "catastrophe_rate": 0.1,
            "mean_escalation_peak": 0.3,
            "mean_inequality_gini": 0.3,
            "mean_compliance_rate": 0.7,
            "mean_cooperation_rate": 0.6,
            "mean_welfare_total": 200.0,
        }
        scorecard = compute_scorecard(agg)
        for key in ["SAFETY", "STABILITY", "ALIGNMENT", "EFFICIENCY", "OVERALL"]:
            assert 0.0 <= scorecard[key] <= 1.0, f"{key} out of range: {scorecard[key]}"

    def test_perfect_scores(self):
        agg = {
            "catastrophe_rate": 0.0,
            "mean_escalation_peak": 0.0,
            "mean_inequality_gini": 0.0,
            "mean_compliance_rate": 1.0,
            "mean_cooperation_rate": 1.0,
            "mean_welfare_total": 500.0,
        }
        scorecard = compute_scorecard(agg)
        assert scorecard["SAFETY"] == pytest.approx(1.0)
        assert scorecard["OVERALL"] >= 0.9

"""Tests for evolutionary governance search."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# These tests need darwinian_evolver installed
de_available = True
try:
    import darwinian_evolver  # noqa: F401
except ImportError:
    de_available = False

skip_no_de = pytest.mark.skipif(
    not de_available, reason="darwinian_evolver not installed"
)


# ---------------------------------------------------------------------------
# Unit tests (no darwinian_evolver required for fitness/ranges)
# ---------------------------------------------------------------------------


class TestFitnessFunction:
    """Test the fitness scoring function."""

    def test_perfect_scenario_near_one(self):
        from swarm.analysis.evolver import compute_fitness

        score = compute_fitness(
            avg_toxicity=0.0,
            avg_welfare=10.0,
            avg_quality_gap=1.0,
            payoff_gap=5.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_catastrophic_toxicity_low_score(self):
        from swarm.analysis.evolver import compute_fitness

        score = compute_fitness(
            avg_toxicity=1.0,
            avg_welfare=0.0,
            avg_quality_gap=0.0,
            payoff_gap=0.0,
        )
        assert score == pytest.approx(0.0, abs=0.01)

    def test_negative_quality_gap_clamped(self):
        from swarm.analysis.evolver import compute_fitness

        score = compute_fitness(
            avg_toxicity=0.0,
            avg_welfare=10.0,
            avg_quality_gap=-0.5,
            payoff_gap=5.0,
        )
        # quality_gap contribution should be 0 (clamped)
        score_with_zero = compute_fitness(
            avg_toxicity=0.0,
            avg_welfare=10.0,
            avg_quality_gap=0.0,
            payoff_gap=5.0,
        )
        assert score == pytest.approx(score_with_zero, abs=0.001)

    def test_custom_weights(self):
        from swarm.analysis.evolver import compute_fitness

        # All weight on toxicity
        score = compute_fitness(
            avg_toxicity=0.0,
            avg_welfare=0.0,
            avg_quality_gap=0.0,
            payoff_gap=0.0,
            weights={"low_toxicity": 1.0, "welfare": 0.0, "quality_gap": 0.0, "payoff_gap": 0.0},
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_always_in_unit_interval(self):
        from swarm.analysis.evolver import compute_fitness

        # Extreme values
        for tox in [0.0, 0.5, 1.0, 2.0]:
            for w in [-10.0, 0.0, 10.0, 100.0]:
                for qg in [-1.0, 0.0, 1.0]:
                    for pg in [-5.0, 0.0, 5.0]:
                        score = compute_fitness(tox, w, qg, pg)
                        assert 0.0 <= score <= 1.0, f"Out of range: {score}"


class TestParamRanges:
    """Test parameter range definitions."""

    def test_ranges_exist(self):
        from swarm.analysis.evolver import PARAM_RANGES

        assert len(PARAM_RANGES) > 0
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo < hi, f"{name}: lo={lo} >= hi={hi}"

    def test_int_params_subset_of_ranges(self):
        from swarm.analysis.evolver import INT_PARAMS, PARAM_RANGES

        for p in INT_PARAMS:
            assert p in PARAM_RANGES, f"{p} in INT_PARAMS but not PARAM_RANGES"


# ---------------------------------------------------------------------------
# Tests requiring darwinian_evolver
# ---------------------------------------------------------------------------


@skip_no_de
class TestGovernanceOrganism:
    """Test GovernanceOrganism data model."""

    def test_params_stored(self):
        from swarm.analysis.evolver import GovernanceOrganism

        org = GovernanceOrganism(params={"governance.transaction_tax_rate": 0.1})
        assert org.params["governance.transaction_tax_rate"] == 0.1

    def test_default_empty_params(self):
        from swarm.analysis.evolver import GovernanceOrganism

        org = GovernanceOrganism()
        assert org.params == {}

    def test_visualizer_props(self):
        from swarm.analysis.evolver import GovernanceOrganism

        org = GovernanceOrganism(params={"governance.transaction_tax_rate": 0.12345})
        props = org.visualizer_props
        assert "transaction_tax_rate" in props
        assert props["transaction_tax_rate"] == 0.1235  # rounded to 4 digits


@skip_no_de
class TestSimulationFailureCase:
    """Test failure case model."""

    def test_construction(self):
        from swarm.analysis.evolver import SimulationFailureCase

        fc = SimulationFailureCase(
            epoch=3,
            epoch_metrics={"toxicity_rate": 0.5},
            active_params={"governance.transaction_tax_rate": 0.1},
            failure_type="high_toxicity",
        )
        assert fc.epoch == 3
        assert fc.failure_type == "high_toxicity"
        assert fc.data_point_id == "epoch_3"

    def test_failure_types(self):
        from swarm.analysis.evolver import SimulationFailureCase

        for ft in ["high_toxicity", "low_welfare", "adverse_selection", "bad_payoff_gap"]:
            fc = SimulationFailureCase(failure_type=ft)
            assert fc.failure_type == ft


@skip_no_de
class TestRandomGovernanceMutator:
    """Test the random mutator."""

    def test_produces_offspring(self):
        from swarm.analysis.evolver import GovernanceOrganism, RandomGovernanceMutator

        mutator = RandomGovernanceMutator(n_offspring=3)
        parent = GovernanceOrganism(params={"governance.transaction_tax_rate": 0.1})

        children = mutator.mutate(parent, [], [])
        assert len(children) == 3
        for child in children:
            assert isinstance(child, GovernanceOrganism)
            assert child.from_change_summary is not None

    def test_respects_ranges(self):
        from swarm.analysis.evolver import (
            PARAM_RANGES,
            GovernanceOrganism,
            RandomGovernanceMutator,
        )

        mutator = RandomGovernanceMutator(n_offspring=20)
        parent = GovernanceOrganism(params={})

        children = mutator.mutate(parent, [], [])
        for child in children:
            for key, val in child.params.items():
                if key in PARAM_RANGES:
                    lo, hi = PARAM_RANGES[key]
                    assert lo <= val <= hi, f"{key}={val} outside [{lo}, {hi}]"

    def test_inherits_parent_params(self):
        from swarm.analysis.evolver import GovernanceOrganism, RandomGovernanceMutator

        parent_params = {
            "governance.transaction_tax_rate": 0.1,
            "payoff.theta": 0.7,
            "payoff.rho_a": 0.3,
        }
        mutator = RandomGovernanceMutator(n_offspring=5)
        parent = GovernanceOrganism(params=parent_params)

        children = mutator.mutate(parent, [], [])
        for child in children:
            # All parent keys should be present (some may be mutated)
            for key in parent_params:
                assert key in child.params

    def test_supports_batch_mutation(self):
        from swarm.analysis.evolver import RandomGovernanceMutator

        mutator = RandomGovernanceMutator()
        assert mutator.supports_batch_mutation is True


@skip_no_de
class TestGovernanceEvaluationResult:
    """Test evaluation result model."""

    def test_visualizer_props(self):
        from swarm.analysis.evolver import GovernanceEvaluationResult

        result = GovernanceEvaluationResult(
            score=0.75,
            trainable_failure_cases=[],
            avg_toxicity=0.1,
            avg_welfare=5.0,
            avg_quality_gap=0.2,
            payoff_gap=1.0,
            n_frozen_agents=0,
            params={},
        )
        props = result.visualizer_props
        assert "toxicity" in props
        assert "welfare" in props

    def test_format_observed_outcome(self):
        from swarm.analysis.evolver import GovernanceEvaluationResult

        result = GovernanceEvaluationResult(
            score=0.75,
            trainable_failure_cases=[],
            avg_toxicity=0.1,
            avg_welfare=5.0,
            avg_quality_gap=0.2,
            payoff_gap=1.0,
            params={},
        )
        text = result.format_observed_outcome(None)
        assert "score=0.75" in text
        assert "toxicity=0.1" in text

    def test_format_with_parent(self):
        from swarm.analysis.evolver import GovernanceEvaluationResult

        parent = GovernanceEvaluationResult(
            score=0.50,
            trainable_failure_cases=[],
            params={},
        )
        child = GovernanceEvaluationResult(
            score=0.75,
            trainable_failure_cases=[],
            params={},
        )
        text = child.format_observed_outcome(parent)
        assert "delta=+0.25" in text


@skip_no_de
class TestGovernanceEvaluator:
    """Integration test: evaluate a baseline scenario."""

    def test_evaluates_baseline(self):
        from swarm.analysis.evolver import GovernanceEvaluator, GovernanceOrganism
        from swarm.scenarios.loader import load_scenario

        scenario_path = Path("scenarios/baseline.yaml")
        if not scenario_path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(scenario_path)
        evaluator = GovernanceEvaluator(
            base_scenario=scenario,
            eval_epochs=2,
            eval_steps=3,
        )
        org = GovernanceOrganism(params={})
        result = evaluator.evaluate(org)

        assert 0.0 <= result.score <= 1.0
        assert result.is_viable
        assert result.avg_toxicity >= 0.0

    def test_handles_invalid_params(self):
        from swarm.analysis.evolver import GovernanceEvaluator, GovernanceOrganism
        from swarm.scenarios.loader import load_scenario

        scenario_path = Path("scenarios/baseline.yaml")
        if not scenario_path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(scenario_path)
        evaluator = GovernanceEvaluator(
            base_scenario=scenario,
            eval_epochs=2,
            eval_steps=3,
        )
        # Use a bogus attribute that will cause an exception during build
        org = GovernanceOrganism(params={"governance.nonexistent_attr_xyz": 42})
        result = evaluator.evaluate(org)
        # Should return a result (either viable or not) without crashing
        assert 0.0 <= result.score <= 1.0


@skip_no_de
class TestRunEvolution:
    """Integration test: run a short evolution."""

    def test_short_evolution(self):
        from swarm.analysis.evolver import EvolverConfig, run_evolution
        from swarm.scenarios.loader import load_scenario

        scenario_path = Path("scenarios/baseline.yaml")
        if not scenario_path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(scenario_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvolverConfig(
                base_scenario=scenario,
                n_iterations=2,
                num_parents_per_iteration=2,
                eval_epochs=2,
                eval_steps=3,
                final_eval_epochs=2,
                final_eval_steps=3,
                use_llm_mutator=False,
                output_dir=Path(tmpdir) / "evolution_run",
            )

            result = run_evolution(config)

            assert result.best_score >= 0.0
            assert result.best_score <= 1.0
            assert result.n_iterations == 2

            # Check artifacts
            run_dir = result.run_dir
            assert run_dir is not None
            assert (run_dir / "config.json").exists()
            assert (run_dir / "best_organism.json").exists()
            assert (run_dir / "final_eval.json").exists()
            assert (run_dir / "summary.json").exists()
            assert (run_dir / "population_log.jsonl").exists()

    def test_resume(self):
        from swarm.analysis.evolver import EvolverConfig, run_evolution
        from swarm.scenarios.loader import load_scenario

        scenario_path = Path("scenarios/baseline.yaml")
        if not scenario_path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(scenario_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "evolution_run"

            # Run 1 iteration
            config1 = EvolverConfig(
                base_scenario=scenario,
                n_iterations=1,
                num_parents_per_iteration=2,
                eval_epochs=2,
                eval_steps=3,
                final_eval_epochs=2,
                final_eval_steps=3,
                use_llm_mutator=False,
                output_dir=output_dir,
            )
            run_evolution(config1)

            # Find snapshot
            snapshot_path = output_dir / "snapshots" / "iter_001.pkl"
            assert snapshot_path.exists()

            # Resume for 1 more iteration
            output_dir2 = Path(tmpdir) / "evolution_resumed"
            config2 = EvolverConfig(
                base_scenario=scenario,
                n_iterations=1,
                num_parents_per_iteration=2,
                eval_epochs=2,
                eval_steps=3,
                final_eval_epochs=2,
                final_eval_steps=3,
                use_llm_mutator=False,
                output_dir=output_dir2,
                resume_from=snapshot_path,
            )
            result2 = run_evolution(config2)
            assert result2.best_score >= 0.0


class TestCLI:
    """Test CLI integration."""

    def test_evolve_help(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "swarm", "evolve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "evolve" in result.stdout.lower() or "iterations" in result.stdout.lower()

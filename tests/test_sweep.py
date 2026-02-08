"""Tests for parameter sweep module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from swarm.analysis import (
    SweepConfig,
    SweepParameter,
    SweepResult,
    SweepRunner,
    quick_sweep,
)
from swarm.scenarios import load_scenario

pytestmark = pytest.mark.slow


@pytest.fixture
def base_scenario():
    """Load baseline scenario for testing."""
    path = Path("scenarios/baseline.yaml")
    if not path.exists():
        pytest.skip("baseline.yaml not found")
    scenario = load_scenario(path)
    # Reduce for faster tests
    scenario.orchestrator_config.n_epochs = 2
    scenario.orchestrator_config.steps_per_epoch = 3
    return scenario


class TestSweepParameter:
    """Tests for SweepParameter."""

    def test_create_parameter(self):
        """Should create parameter with values."""
        param = SweepParameter(name="governance.tax", values=[0.0, 0.1, 0.2])
        assert param.name == "governance.tax"
        assert param.values == [0.0, 0.1, 0.2]

    def test_empty_values_raises(self):
        """Should raise on empty values list."""
        with pytest.raises(ValueError):
            SweepParameter(name="test", values=[])


class TestSweepConfig:
    """Tests for SweepConfig."""

    def test_total_runs_no_params(self, base_scenario):
        """No parameters should give runs_per_config runs."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[],
            runs_per_config=3,
        )
        assert config.total_runs() == 3

    def test_total_runs_single_param(self, base_scenario):
        """Single parameter should multiply correctly."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[SweepParameter(name="test", values=[1, 2, 3])],
            runs_per_config=2,
        )
        assert config.total_runs() == 6  # 3 values * 2 runs

    def test_total_runs_multiple_params(self, base_scenario):
        """Multiple parameters should give cartesian product."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="a", values=[1, 2]),
                SweepParameter(name="b", values=[1, 2, 3]),
            ],
            runs_per_config=1,
        )
        assert config.total_runs() == 6  # 2 * 3 * 1

    def test_fluent_add_parameter(self, base_scenario):
        """add_parameter should work fluently."""
        config = SweepConfig(base_scenario=base_scenario)
        config.add_parameter("a", [1, 2]).add_parameter("b", [3, 4])

        assert len(config.parameters) == 2
        assert config.parameters[0].name == "a"
        assert config.parameters[1].name == "b"


class TestSweepResult:
    """Tests for SweepResult."""

    def test_to_dict(self):
        """Should convert to flat dictionary."""
        result = SweepResult(
            params={"tax": 0.1},
            run_index=0,
            seed=42,
            total_welfare=100.0,
            avg_toxicity=0.3,
        )
        d = result.to_dict()

        assert d["tax"] == 0.1
        assert d["run_index"] == 0
        assert d["seed"] == 42
        assert d["total_welfare"] == 100.0
        assert d["avg_toxicity"] == 0.3


class TestSweepRunner:
    """Tests for SweepRunner."""

    def test_run_no_params(self, base_scenario):
        """Should run with no parameter sweep."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[],
            runs_per_config=1,
        )
        runner = SweepRunner(config)
        results = runner.run()

        assert len(results) == 1
        assert results[0].total_interactions >= 0

    def test_run_single_param(self, base_scenario):
        """Should run single parameter sweep."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(
                    name="governance.transaction_tax_rate",
                    values=[0.0, 0.1],
                ),
            ],
            runs_per_config=1,
            seed_base=42,
        )
        runner = SweepRunner(config)
        results = runner.run()

        assert len(results) == 2

        # Check params were applied
        taxes = [r.params["governance.transaction_tax_rate"] for r in results]
        assert 0.0 in taxes
        assert 0.1 in taxes

    def test_run_multiple_params(self, base_scenario):
        """Should run multi-parameter sweep."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(
                    name="governance.transaction_tax_rate",
                    values=[0.0, 0.05],
                ),
                SweepParameter(
                    name="governance.circuit_breaker_enabled",
                    values=[False, True],
                ),
            ],
            runs_per_config=1,
            seed_base=42,
        )
        runner = SweepRunner(config)
        results = runner.run()

        assert len(results) == 4  # 2 * 2

    def test_progress_callback(self, base_scenario):
        """Should call progress callback."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="governance.transaction_tax_rate", values=[0.0, 0.1]),
            ],
            runs_per_config=1,
        )

        calls = []

        def callback(current, total, params):
            calls.append((current, total, params))

        runner = SweepRunner(config, progress_callback=callback)
        runner.run()

        assert len(calls) == 2
        assert calls[0][0] == 1  # First run
        assert calls[1][0] == 2  # Second run
        assert calls[0][1] == 2  # Total

    def test_to_csv(self, base_scenario):
        """Should export results to CSV."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="governance.transaction_tax_rate", values=[0.0, 0.1]),
            ],
            runs_per_config=1,
        )
        runner = SweepRunner(config)
        runner.run()

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            runner.to_csv(path)

            assert path.exists()

            # Read and verify
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert "governance.transaction_tax_rate" in rows[0]
            assert "total_welfare" in rows[0]

    def test_to_csv_no_results_raises(self, base_scenario):
        """Should raise if no results."""
        config = SweepConfig(base_scenario=base_scenario)
        runner = SweepRunner(config)

        with pytest.raises(ValueError, match="No results"):
            runner.to_csv(Path("test.csv"))

    def test_summary(self, base_scenario):
        """Should generate summary statistics."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="governance.transaction_tax_rate", values=[0.0, 0.1]),
            ],
            runs_per_config=2,
        )
        runner = SweepRunner(config)
        runner.run()

        summary = runner.summary()

        assert summary["total_runs"] == 4
        assert summary["param_combinations"] == 2
        assert len(summary["summaries"]) == 2


class TestQuickSweep:
    """Tests for quick_sweep convenience function."""

    def test_quick_sweep(self, base_scenario):
        """Should run single-parameter sweep."""
        results = quick_sweep(
            scenario=base_scenario,
            param_name="governance.transaction_tax_rate",
            values=[0.0, 0.05, 0.1],
            runs_per_config=1,
            seed_base=42,
        )

        assert len(results) == 3


class TestParameterApplication:
    """Tests for parameter application to scenarios."""

    def test_governance_params(self, base_scenario):
        """Should apply governance parameters."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="governance.transaction_tax_rate", values=[0.15]),
                SweepParameter(name="governance.staking_enabled", values=[True]),
                SweepParameter(name="governance.circuit_breaker_enabled", values=[True]),
            ],
            runs_per_config=1,
        )
        runner = SweepRunner(config)
        results = runner.run()

        # Verify params recorded
        assert results[0].params["governance.transaction_tax_rate"] == 0.15
        assert results[0].params["governance.staking_enabled"] is True

    def test_payoff_params(self, base_scenario):
        """Should apply payoff parameters."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="payoff.s_plus", values=[3.0]),
                SweepParameter(name="payoff.h", values=[3.0]),
            ],
            runs_per_config=1,
        )
        runner = SweepRunner(config)
        results = runner.run()

        assert results[0].params["payoff.s_plus"] == 3.0
        assert results[0].params["payoff.h"] == 3.0

    def test_simulation_params(self, base_scenario):
        """Should apply simulation parameters."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(name="n_epochs", values=[3]),
            ],
            runs_per_config=1,
        )
        runner = SweepRunner(config)
        results = runner.run()

        assert results[0].params["n_epochs"] == 3

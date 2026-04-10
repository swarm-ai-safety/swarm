"""Tests for parallel sweep execution."""

from pathlib import Path

import pytest

from swarm.analysis.sweep import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

pytestmark = pytest.mark.slow


@pytest.fixture
def base_scenario():
    """Load baseline scenario for testing."""
    path = Path("scenarios/baseline.yaml")
    if not path.exists():
        pytest.skip("baseline.yaml not found")
    scenario = load_scenario(path)
    scenario.orchestrator_config.n_epochs = 2
    scenario.orchestrator_config.steps_per_epoch = 3
    return scenario


class TestParallelSweep:
    """Tests for n_workers > 1 in SweepRunner."""

    def test_parallel_matches_serial(self, base_scenario):
        """Parallel and serial should produce identical results."""
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

        serial = SweepRunner(config, n_workers=1)
        serial_results = serial.run()

        parallel = SweepRunner(config, n_workers=2)
        parallel_results = parallel.run()

        assert len(serial_results) == len(parallel_results)

        for s, p in zip(serial_results, parallel_results, strict=True):
            assert s.params == p.params
            assert s.seed == p.seed
            # Same seed + same params → same deterministic result
            assert s.total_welfare == p.total_welfare
            assert s.avg_toxicity == p.avg_toxicity

    def test_parallel_progress_callback(self, base_scenario):
        """Progress callback should fire for each completed run."""
        config = SweepConfig(
            base_scenario=base_scenario,
            parameters=[
                SweepParameter(
                    name="governance.transaction_tax_rate",
                    values=[0.0, 0.05],
                ),
            ],
            runs_per_config=1,
        )

        calls = []
        runner = SweepRunner(
            config,
            progress_callback=lambda c, t, p: calls.append((c, t)),
            n_workers=2,
        )
        runner.run()

        assert len(calls) == 2
        totals = {c[1] for c in calls}
        assert totals == {2}

    def test_n_workers_clamps_to_1(self, base_scenario):
        """n_workers=0 or negative should be treated as 1."""
        config = SweepConfig(base_scenario=base_scenario, runs_per_config=1)
        runner = SweepRunner(config, n_workers=0)
        assert runner.n_workers == 1
        runner = SweepRunner(config, n_workers=-3)
        assert runner.n_workers == 1

"""Benchmark runner — runs (task × governance config) grid and produces DataFrames.

Usage:
    runner = BenchmarkRunner(n_agents=10)
    df = runner.run_frontier(benchmark, governance_configs, n_seeds=10)
    # df has one row per (config, seed) — ready for frontier plotting
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from swarm.benchmarks.base import BenchmarkScore, BenchmarkTask, TaskResult
from swarm.metrics.soft_metrics import SoftMetrics


class BenchmarkRunner:
    """Runs a benchmark across governance configs and seeds."""

    def __init__(self, n_agents: int = 10):
        self.n_agents = n_agents
        self._soft_metrics = SoftMetrics()

    def run_frontier(
        self,
        benchmark: BenchmarkTask,
        governance_configs: list[dict[str, Any]],
        n_seeds: int = 10,
        run_fn: Any | None = None,
    ) -> pd.DataFrame:
        """Run benchmark across (governance_config × seed) grid.

        Args:
            benchmark: The benchmark task to evaluate.
            governance_configs: List of dicts, each with at least an "id" key.
            n_seeds: Number of random seeds to average over.
            run_fn: Optional callable(instance, gov_config) -> TaskResult.
                     If None, uses oracle_run (useful for testing the runner
                     itself or for simulated adversarial perturbation).

        Returns:
            DataFrame with one row per (config, seed).
        """
        # Compute oracle baselines once per seed
        oracles: dict[int, TaskResult] = {}
        instances: dict[int, Any] = {}
        for seed in range(n_seeds):
            instance = benchmark.generate(seed, self.n_agents)
            instances[seed] = instance
            oracles[seed] = benchmark.oracle_run(instance)

        rows: list[dict[str, Any]] = []
        for gov_config in governance_configs:
            for seed in range(n_seeds):
                instance = instances[seed]

                if run_fn is not None:
                    result = run_fn(instance, gov_config)
                else:
                    result = benchmark.oracle_run(instance)

                score = benchmark.score(result, oracles[seed])
                interaction = benchmark.to_soft_interaction(score)

                row = {
                    "benchmark": benchmark.task_id,
                    "task_type": benchmark.task_type,
                    "gov_config": gov_config.get("id", "unknown"),
                    "seed": seed,
                    "p": interaction.p,
                    **asdict(score),
                }
                # Include governance lever values for analysis
                for k, v in gov_config.items():
                    if k != "id":
                        row[f"gov_{k}"] = v

                rows.append(row)

        return pd.DataFrame(rows)

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-config results: mean ± std across seeds."""
        score_cols = [
            "completion_rate",
            "efficiency",
            "fidelity",
            "capability_ratio",
            "p",
        ]
        present = [c for c in score_cols if c in df.columns]
        return (
            df.groupby("gov_config")[present]
            .agg(["mean", "std"])
            .round(4)
        )

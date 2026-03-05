"""Tests for the frontier trace experiment infrastructure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swarm.benchmarks.allocation.resource_auction import ResourceAuctionBenchmark
from swarm.benchmarks.base import TaskResult
from swarm.benchmarks.coordination.distributed_allocation import (
    DistributedAllocationBenchmark,
)
from swarm.benchmarks.governance_run_fns import (
    RUN_FN_REGISTRY,
    _governance_friction,
    auction_run_fn,
    coordination_run_fn,
    pipeline_run_fn,
    routing_run_fn,
)
from swarm.benchmarks.long_horizon.pipeline_task import PipelineTaskBenchmark
from swarm.benchmarks.routing.message_routing import MessageRoutingBenchmark
from swarm.benchmarks.runner import BenchmarkRunner


class TestGovernanceFriction:
    """Test the governance friction computation."""

    def test_zero_friction_no_governance(self):
        rng = np.random.default_rng(42)
        config = {"id": "none", "audit_rate": 0.0}
        assert _governance_friction(config, rng) == 0.0

    def test_friction_increases_with_audit_rate(self):
        rng = np.random.default_rng(42)
        low = _governance_friction({"audit_rate": 0.1}, rng)
        high = _governance_friction({"audit_rate": 1.0}, rng)
        assert high > low

    def test_friction_capped_below_one(self):
        rng = np.random.default_rng(42)
        maximal = {
            "audit_rate": 1.0,
            "circuit_breaker_enabled": True,
            "circuit_breaker_sensitivity": 1.0,
            "min_stake": 100.0,
            "bandwidth_cap": 1,
            "confirmation_gates": 10,
        }
        assert _governance_friction(maximal, rng) <= 0.95

    def test_friction_monotonic_in_gates(self):
        rng = np.random.default_rng(42)
        f0 = _governance_friction({"confirmation_gates": 0}, rng)
        f1 = _governance_friction({"confirmation_gates": 1}, rng)
        f3 = _governance_friction({"confirmation_gates": 3}, rng)
        assert f0 < f1 < f3


class TestRoutingRunFn:
    """Test the governance-aware routing run function."""

    def test_returns_task_result(self):
        benchmark = MessageRoutingBenchmark()
        instance, _ = benchmark.generate(seed=42, n_agents=10)
        redacted = benchmark.redact(instance)
        config = {"id": "test", "audit_rate": 0.5}
        result = routing_run_fn(redacted, config)
        assert isinstance(result, TaskResult)
        assert isinstance(result.completed, bool)
        assert result.steps_taken >= 0
        assert isinstance(result.agent_trace, list)

    def test_loose_config_better_than_tight(self):
        """Loose governance should generally produce better capability."""
        benchmark = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=10)

        configs = [
            {"id": "tight", "audit_rate": 1.0, "circuit_breaker_enabled": True,
             "circuit_breaker_sensitivity": 0.8, "min_stake": 8.0, "bandwidth_cap": 20,
             "confirmation_gates": 3},
            {"id": "loose", "audit_rate": 0.05, "circuit_breaker_enabled": False,
             "min_stake": 0.0, "bandwidth_cap": 100, "confirmation_gates": 0},
        ]

        df = runner.run_frontier(benchmark, configs, n_seeds=20, run_fn=routing_run_fn)
        tight_cap = df[df["gov_config"] == "tight"]["completion_rate"].mean()
        loose_cap = df[df["gov_config"] == "loose"]["completion_rate"].mean()
        # Loose should have equal or better completion
        assert loose_cap >= tight_cap - 0.1  # allow small margin for stochasticity

    def test_deterministic_same_seed(self):
        benchmark = MessageRoutingBenchmark()
        instance, _ = benchmark.generate(seed=42, n_agents=10)
        config = {"id": "test", "audit_rate": 0.5}
        r1 = routing_run_fn(benchmark.redact(instance), config)
        r2 = routing_run_fn(benchmark.redact(instance), config)
        assert r1.payload == r2.payload
        assert r1.steps_taken == r2.steps_taken


class TestCoordinationRunFn:
    def test_returns_valid_allocation(self):
        benchmark = DistributedAllocationBenchmark()
        instance, _ = benchmark.generate(seed=42, n_agents=10)
        config = {"id": "test", "audit_rate": 0.3}
        result = coordination_run_fn(benchmark.redact(instance), config)
        assert isinstance(result.payload, dict)
        assert len(result.payload) > 0


class TestAuctionRunFn:
    def test_assigns_all_resources(self):
        benchmark = ResourceAuctionBenchmark(n_resources=5)
        instance, _ = benchmark.generate(seed=42, n_agents=10)
        config = {"id": "test", "audit_rate": 0.3}
        result = auction_run_fn(benchmark.redact(instance), config)
        assert len(result.payload) == 5


class TestPipelineRunFn:
    def test_pipeline_completes_or_stalls(self):
        benchmark = PipelineTaskBenchmark(n_stages=5)
        instance, _ = benchmark.generate(seed=42, n_agents=10)
        config = {"id": "test", "audit_rate": 0.3}
        result = pipeline_run_fn(benchmark.redact(instance), config)
        assert isinstance(result.payload, int)
        assert result.steps_taken > 0


class TestRunFnRegistry:
    def test_all_task_types_registered(self):
        assert "routing" in RUN_FN_REGISTRY
        assert "coordination" in RUN_FN_REGISTRY
        assert "allocation" in RUN_FN_REGISTRY
        assert "long_horizon" in RUN_FN_REGISTRY


class TestEndToEndFrontier:
    """Integration test: run a small frontier sweep end-to-end."""

    def test_routing_frontier_produces_dataframe(self):
        benchmark = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=5)
        configs = [
            {"id": "tight", "audit_rate": 0.8, "min_stake": 5.0, "adversarial_fraction": 0.1},
            {"id": "loose", "audit_rate": 0.1, "min_stake": 0.0, "adversarial_fraction": 0.1},
        ]
        df = runner.run_frontier(benchmark, configs, n_seeds=5, run_fn=routing_run_fn)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # 2 configs × 5 seeds
        assert "completion_rate" in df.columns
        assert "fidelity" in df.columns
        assert "p" in df.columns
        assert all(0 <= p <= 1 for p in df["p"])

    def test_summary_aggregates_correctly(self):
        benchmark = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=5)
        configs = [
            {"id": "a", "audit_rate": 0.5},
            {"id": "b", "audit_rate": 0.1},
        ]
        df = runner.run_frontier(benchmark, configs, n_seeds=10, run_fn=routing_run_fn)
        summary = runner.summarize(df)
        assert len(summary) == 2
        assert "a" in summary.index
        assert "b" in summary.index

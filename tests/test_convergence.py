"""Tests for behavioral convergence detection."""

from swarm.analysis.convergence import behavioral_convergence
from swarm.analysis.sweep import SweepResult


def _make_result(params: dict, **kwargs) -> SweepResult:
    """Helper to create a SweepResult with given params and metric overrides."""
    defaults = {
        "params": params,
        "run_index": 0,
        "seed": 42,
        "total_interactions": 100,
        "accepted_interactions": 80,
        "avg_toxicity": 0.1,
        "avg_quality_gap": 0.05,
        "total_welfare": 50.0,
        "welfare_per_epoch": 10.0,
        "n_agents": 10,
        "n_frozen": 0,
        "avg_reputation": 0.8,
        "avg_payoff": 5.0,
        "min_payoff": 1.0,
        "max_payoff": 9.0,
        "honest_avg_payoff": 6.0,
        "opportunistic_avg_payoff": 4.0,
        "deceptive_avg_payoff": 3.0,
        "adversarial_avg_payoff": 2.0,
    }
    defaults.update(kwargs)
    return SweepResult(**defaults)


class TestBehavioralConvergence:
    """Tests for behavioral_convergence()."""

    def test_single_config_returns_trivial(self):
        """One config can't show convergence — returns 1.0 with a warning."""
        results = [
            _make_result({"tax": 0.1}, avg_toxicity=0.2),
            _make_result({"tax": 0.1}, avg_toxicity=0.3),
        ]
        report = behavioral_convergence(results, metrics=["avg_toxicity"])
        assert report.n_configs == 1
        assert report.overall_convergence == 1.0
        assert len(report.warnings) == 1

    def test_identical_configs_full_convergence(self):
        """Different configs with identical outcomes → convergence ≈ 1.0."""
        results = [
            _make_result({"tax": 0.0}, avg_toxicity=0.10, total_welfare=50.0),
            _make_result({"tax": 0.1}, avg_toxicity=0.10, total_welfare=50.0),
            _make_result({"tax": 0.2}, avg_toxicity=0.10, total_welfare=50.0),
        ]
        report = behavioral_convergence(
            results, metrics=["avg_toxicity", "total_welfare"],
        )
        assert report.n_configs == 3
        for m in report.per_metric:
            assert m.convergence >= 0.999

    def test_divergent_configs_low_convergence(self):
        """Very different outcomes → low convergence score."""
        results = [
            _make_result({"tax": 0.0}, total_welfare=10.0),
            _make_result({"tax": 0.5}, total_welfare=100.0),
        ]
        report = behavioral_convergence(results, metrics=["total_welfare"])
        welfare = next(m for m in report.per_metric if m.metric == "total_welfare")
        assert welfare.convergence < 0.5

    def test_adverse_metric_convergence_warns(self):
        """Convergence on high toxicity should emit a warning."""
        results = [
            _make_result({"tax": 0.0}, avg_toxicity=0.50),
            _make_result({"tax": 0.1}, avg_toxicity=0.51),
            _make_result({"tax": 0.2}, avg_toxicity=0.49),
        ]
        report = behavioral_convergence(
            results,
            metrics=["avg_toxicity"],
            convergence_threshold=0.9,
        )
        assert len(report.warnings) >= 1
        assert "proxy gaming" in report.warnings[0].lower()

    def test_multi_seed_per_config(self):
        """Multiple seeds per config should still group correctly."""
        results = [
            _make_result({"tax": 0.0}, avg_toxicity=0.10, seed=1),
            _make_result({"tax": 0.0}, avg_toxicity=0.12, seed=2),
            _make_result({"tax": 0.1}, avg_toxicity=0.11, seed=1),
            _make_result({"tax": 0.1}, avg_toxicity=0.09, seed=2),
        ]
        report = behavioral_convergence(results, metrics=["avg_toxicity"])
        assert report.n_configs == 2
        assert report.n_runs == 4

    def test_to_dict(self):
        """Report should serialize cleanly."""
        results = [
            _make_result({"tax": 0.0}, avg_toxicity=0.10),
            _make_result({"tax": 0.1}, avg_toxicity=0.10),
        ]
        report = behavioral_convergence(results, metrics=["avg_toxicity"])
        d = report.to_dict()
        assert "overall_convergence" in d
        assert "per_metric" in d
        assert isinstance(d["per_metric"], list)

    def test_empty_results(self):
        """Empty results should not crash."""
        report = behavioral_convergence([], metrics=["avg_toxicity"])
        assert report.n_configs == 0
        assert report.overall_convergence == 1.0

    def test_default_metrics_used(self):
        """When no metrics specified, default set is used."""
        results = [
            _make_result({"tax": 0.0}),
            _make_result({"tax": 0.1}),
        ]
        report = behavioral_convergence(results)
        metric_names = {m.metric for m in report.per_metric}
        assert "avg_toxicity" in metric_names
        assert "total_welfare" in metric_names

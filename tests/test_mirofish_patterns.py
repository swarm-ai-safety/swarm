# Tests for parallel runner and ReACT analyst modules.



from swarm.analysis.react_analyst import (
    AnalysisReport,
    RunDataToolkit,
    analyze_run,
)
from swarm.replay.parallel import (
    ParallelRunSpec,
)


class TestRunDataToolkit:
    def _make_data(self):
        return {
            'epoch_snapshots': [
                {'epoch': 0, 'toxicity_rate': 0.3, 'quality_gap': -0.1,
                 'total_welfare': 10.0, 'avg_payoff': 1.0, 'gini_coefficient': 0.2,
                 'n_agents': 5, 'n_frozen': 0},
                {'epoch': 1, 'toxicity_rate': 0.25, 'quality_gap': 0.05,
                 'total_welfare': 15.0, 'avg_payoff': 1.5, 'gini_coefficient': 0.18,
                 'n_agents': 5, 'n_frozen': 0},
                {'epoch': 2, 'toxicity_rate': 0.2, 'quality_gap': 0.1,
                 'total_welfare': 20.0, 'avg_payoff': 2.0, 'gini_coefficient': 0.15,
                 'n_agents': 5, 'n_frozen': 1},
            ],
            'agent_snapshots': {
                'agent_0': [{'epoch': 0, 'reputation': 0.5}, {'epoch': 1, 'reputation': 0.6}],
                'agent_1': [{'epoch': 0, 'reputation': 0.4}, {'epoch': 1, 'reputation': 0.3}],
            },
            'config': {'transaction_tax_rate': 0.05, 'staking_required': True},
        }

    def test_summarize_run(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['summarize_run'].fn()
        assert 'n_epochs' in result
        assert '3' in result

    def test_get_epoch_metrics_last(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['get_epoch_metrics'].fn(epoch=-1)
        assert '0.2' in result  # toxicity_rate of last epoch

    def test_get_metric_trend(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['get_metric_trend'].fn(metric='toxicity_rate')
        assert 'improving' in result

    def test_detect_anomalies_none(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['detect_anomalies'].fn(metric='toxicity_rate')
        assert 'No anomalies' in result

    def test_compare_agents(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['compare_agents'].fn(agent_a='agent_0', agent_b='agent_1')
        assert 'agent_a' in result
        assert 'agent_b' in result

    def test_compare_agents_missing(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['compare_agents'].fn(agent_a='nonexistent', agent_b='agent_1')
        assert 'not found' in result

    def test_get_governance_config(self):
        toolkit = RunDataToolkit(history_data=self._make_data())
        result = toolkit.tools['get_governance_config'].fn()
        assert 'transaction_tax_rate' in result

    def test_no_data(self):
        toolkit = RunDataToolkit()
        result = toolkit.tools['summarize_run'].fn()
        assert 'No run data' in result


class TestReACTAnalyst:
    def _make_data(self):
        return {
            'epoch_snapshots': [
                {'epoch': 0, 'toxicity_rate': 0.3, 'quality_gap': -0.1,
                 'total_welfare': 10.0, 'avg_payoff': 1.0, 'gini_coefficient': 0.2,
                 'n_agents': 5, 'n_frozen': 0},
                {'epoch': 1, 'toxicity_rate': 0.25, 'quality_gap': 0.05,
                 'total_welfare': 15.0, 'avg_payoff': 1.5, 'gini_coefficient': 0.18,
                 'n_agents': 5, 'n_frozen': 0},
                {'epoch': 2, 'toxicity_rate': 0.2, 'quality_gap': 0.1,
                 'total_welfare': 20.0, 'avg_payoff': 2.0, 'gini_coefficient': 0.15,
                 'n_agents': 5, 'n_frozen': 1},
            ],
        }

    def test_default_analysis(self):
        report = analyze_run(history_data=self._make_data())
        assert isinstance(report, AnalysisReport)
        assert len(report.steps) == 5  # default plan has 5 steps
        assert report.conclusion

    def test_custom_plan(self):
        plan = [
            {'thought': 'Check welfare', 'action': 'get_metric_trend', 'input': {'metric': 'total_welfare'}},
        ]
        report = analyze_run(history_data=self._make_data(), plan=plan)
        assert len(report.steps) == 1

    def test_to_markdown(self):
        report = analyze_run(history_data=self._make_data())
        md = report.to_markdown()
        assert '# Analysis:' in md
        assert '## Step 1' in md

    def test_metrics_extracted(self):
        report = analyze_run(history_data=self._make_data())
        assert len(report.metrics_cited) > 0

    def test_unknown_tool(self):
        plan = [{'thought': 'Try bad tool', 'action': 'nonexistent_tool', 'input': {}}]
        report = analyze_run(history_data=self._make_data(), plan=plan)
        assert 'Unknown tool' in report.steps[0].observation


class TestParallelRunSpec:
    def test_dataclass_fields(self):
        spec = ParallelRunSpec(scenario=None, seed=42, index=0)
        assert spec.seed == 42
        assert spec.eval_epochs is None

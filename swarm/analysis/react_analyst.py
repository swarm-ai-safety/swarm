# ReACT-based tool-augmented analysis for run data.
# Implements Thought-Action-Observation-Reflection loop
# for querying simulation results and synthesizing findings.
# Borrowed pattern: MiroFish report agent ReACT loop.

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class AnalysisTool:
    name: str
    description: str
    fn: Callable[..., str]


@dataclass
class ReACTStep:
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    reflection: str = chr(39)+chr(39)


@dataclass
class AnalysisReport:
    question: str
    steps: List[ReACTStep] = field(default_factory=list)
    conclusion: str = chr(39)+chr(39)
    metrics_cited: Dict[str, float] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f'# Analysis: {self.question}', '']
        for i, step in enumerate(self.steps, 1):
            lines.append(f'## Step {i}')
            lines.append(f'**Thought:** {step.thought}')
            lines.append(f'**Action:** {step.action}({json.dumps(step.action_input)})')
            lines.append(f'**Observation:** {step.observation}')
            if step.reflection:
                lines.append(f'**Reflection:** {step.reflection}')
            lines.append('')
        lines.append('## Conclusion')
        lines.append(self.conclusion)
        if self.metrics_cited:
            lines.append('')
            lines.append('### Key Metrics')
            for k, v in self.metrics_cited.items():
                lines.append(f'- **{k}**: {v:.4f}')
        return chr(10).join(lines)


class RunDataToolkit:
    # Toolkit of analysis tools that operate on simulation run data.

    def __init__(self, history_path=None, history_data=None):
        self._data = history_data
        if history_path is not None and self._data is None:
            with open(history_path) as f:
                self._data = json.load(f)
        self.tools = self._build_tools()

    def _build_tools(self):
        tools = {}
        tools['get_epoch_metrics'] = AnalysisTool(
            name='get_epoch_metrics',
            description='Get metrics for a specific epoch or range',
            fn=self._get_epoch_metrics)
        tools['get_metric_trend'] = AnalysisTool(
            name='get_metric_trend',
            description='Get trend statistics for a metric across epochs',
            fn=self._get_metric_trend)
        tools['compare_agents'] = AnalysisTool(
            name='compare_agents',
            description='Compare metrics between two agents',
            fn=self._compare_agents)
        tools['detect_anomalies'] = AnalysisTool(
            name='detect_anomalies',
            description='Find epochs where a metric deviates significantly',
            fn=self._detect_anomalies)
        tools['summarize_run'] = AnalysisTool(
            name='summarize_run',
            description='Get overall run summary statistics',
            fn=self._summarize_run)
        tools['get_governance_config'] = AnalysisTool(
            name='get_governance_config',
            description='Get the governance configuration used in the run',
            fn=self._get_governance_config)
        return tools

    def _get_epoch_metrics(self, epoch=-1, metric='all'):
        if self._data is None:
            return 'No run data loaded'
        snapshots = self._data.get('epoch_snapshots', [])
        if not snapshots:
            return 'No epoch data available'
        if epoch == -1:
            epoch = len(snapshots) - 1
        if epoch < 0 or epoch >= len(snapshots):
            return f'Epoch {epoch} out of range [0, {len(snapshots)-1}]'
        snap = snapshots[epoch]
        if metric != 'all':
            val = snap.get(metric, 'N/A')
            return f'Epoch {epoch} {metric}: {val}'
        return json.dumps(snap, indent=2)

    def _get_metric_trend(self, metric='toxicity_rate'):
        if self._data is None:
            return 'No run data loaded'
        snapshots = self._data.get('epoch_snapshots', [])
        if not snapshots:
            return 'No epoch data'
        values = [s.get(metric, 0.0) for s in snapshots]
        if len(values) < 2:
            return f'Not enough data for trend (have {len(values)} epochs)'
        result = {
            'metric': metric,
            'n_epochs': len(values),
            'first': values[0],
            'last': values[-1],
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'direction': 'improving' if values[-1] < values[0] else 'worsening'
        }
        return json.dumps(result, indent=2)

    def _compare_agents(self, agent_a='', agent_b=''):
        if self._data is None:
            return 'No run data loaded'
        agent_snapshots = self._data.get('agent_snapshots', {})
        if agent_a not in agent_snapshots:
            available = list(agent_snapshots.keys())[:10]
            return f'Agent {agent_a!r} not found. Available: {available}'
        if agent_b not in agent_snapshots:
            available = list(agent_snapshots.keys())[:10]
            return f'Agent {agent_b!r} not found. Available: {available}'
        snaps_a = agent_snapshots[agent_a]
        snaps_b = agent_snapshots[agent_b]
        last_a = snaps_a[-1] if snaps_a else {}
        last_b = snaps_b[-1] if snaps_b else {}
        return json.dumps({'agent_a': last_a, 'agent_b': last_b}, indent=2)

    def _detect_anomalies(self, metric='toxicity_rate', threshold=2.0):
        if self._data is None:
            return 'No run data loaded'
        snapshots = self._data.get('epoch_snapshots', [])
        values = [s.get(metric, 0.0) for s in snapshots]
        if len(values) < 3:
            return 'Not enough data for anomaly detection'
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        if stdev == 0:
            return 'No variance detected (all values identical)'
        anomalies = []
        for i, v in enumerate(values):
            z = abs(v - mean) / stdev
            if z > threshold:
                anomalies.append({'epoch': i, 'value': v, 'z_score': round(z, 2)})
        if not anomalies:
            return f'No anomalies detected (threshold: {threshold} sigma)'
        return json.dumps(anomalies, indent=2)

    def _summarize_run(self):
        if self._data is None:
            return 'No run data loaded'
        snapshots = self._data.get('epoch_snapshots', [])
        if not snapshots:
            return 'No epoch data'
        first = snapshots[0]
        last = snapshots[-1]
        summary = {
            'n_epochs': len(snapshots),
            'final_toxicity': last.get('toxicity_rate', 0),
            'final_quality_gap': last.get('quality_gap', 0),
            'final_welfare': last.get('total_welfare', 0),
            'final_avg_payoff': last.get('avg_payoff', 0),
            'final_gini': last.get('gini_coefficient', 0),
            'n_agents': last.get('n_agents', 0),
            'n_frozen': last.get('n_frozen', 0),
            'toxicity_delta': last.get('toxicity_rate', 0) - first.get('toxicity_rate', 0),
            'welfare_delta': last.get('total_welfare', 0) - first.get('total_welfare', 0),
        }
        return json.dumps(summary, indent=2)

    def _get_governance_config(self):
        if self._data is None:
            return 'No run data loaded'
        config = self._data.get('config', self._data.get('governance_config', {}))
        if not config:
            return 'No governance config found in run data'
        return json.dumps(config, indent=2)


class ReACTAnalyst:
    # Executes a ReACT loop over run data using the toolkit.

    def __init__(self, toolkit):
        self.toolkit = toolkit
        self.max_steps = 8

    def analyze(self, question, plan=None):
        report = AnalysisReport(question=question)
        steps = plan or self._default_plan(question)
        for step_spec in steps[:self.max_steps]:
            thought = step_spec.get('thought', '')
            action = step_spec.get('action', 'summarize_run')
            action_input = step_spec.get('input', {})
            tool = self.toolkit.tools.get(action)
            if tool is None:
                observation = f'Unknown tool: {action}'
            else:
                observation = tool.fn(**action_input)
            step = ReACTStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                reflection=step_spec.get('reflection', ''))
            report.steps.append(step)
        report.conclusion = self._synthesize(report)
        report.metrics_cited = self._extract_metrics(report)
        return report

    def _default_plan(self, question):
        return [
            {'thought': 'Start with run overview', 'action': 'summarize_run', 'input': {}},
            {'thought': 'Check toxicity trend', 'action': 'get_metric_trend', 'input': {'metric': 'toxicity_rate'}},
            {'thought': 'Check quality gap trend', 'action': 'get_metric_trend', 'input': {'metric': 'quality_gap'}},
            {'thought': 'Look for anomalies', 'action': 'detect_anomalies', 'input': {'metric': 'toxicity_rate'}},
            {'thought': 'Check final epoch details', 'action': 'get_epoch_metrics', 'input': {'epoch': -1}},
        ]

    def _synthesize(self, report):
        observations = [s.observation for s in report.steps if s.observation]
        if not observations:
            return 'No data available for analysis.'
        return f'Analysis of {len(report.steps)} steps completed. See observations above.'

    def _extract_metrics(self, report):
        metrics = {}
        for step in report.steps:
            try:
                data = json.loads(step.observation)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            metrics[k] = float(v)
            except (json.JSONDecodeError, TypeError):
                pass
        return metrics


def analyze_run(history_path=None, history_data=None,
                question='What are the key findings from this run?', plan=None):
    toolkit = RunDataToolkit(history_path=history_path, history_data=history_data)
    analyst = ReACTAnalyst(toolkit)
    return analyst.analyze(question, plan=plan)

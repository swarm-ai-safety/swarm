---
description: "AI safety metrics guide for multi-agent systems: soft labels, toxicity, quality gap, conditional loss, and incoherence."
keywords:
  - ai safety metrics
  - multi-agent evaluation
  - soft labels
howto:
  name: "How to evaluate multi-agent safety metrics in SWARM"
  steps:
    - "Choose core metrics aligned with your safety hypothesis."
    - "Instrument runs to export reproducible metric histories."
    - "Compare baseline and intervention distributions, not only means."
    - "Validate metric behavior against adversarial and stress scenarios."
    - "Report uncertainty, replication status, and known metric limits."
---

# AI Safety Metrics for Multi-Agent Systems: Complete Guide

This hub maps SWARM's measurement stack from definitions to practical evaluation workflows.

## In This Guide

- [Metric Foundations](#metric-foundations)
- [Core Metrics and Interpretation](#core-metrics-and-interpretation)
- [Implementation and Evaluation Workflow](#implementation-and-evaluation-workflow)
- [Research and Validation Examples](#research-and-validation-examples)

## Metric Foundations

SWARM metrics are built on probabilistic interaction quality:

- [Soft labels and proxy modeling](../concepts/soft-labels.md)
- [Metrics overview and formulas](../concepts/metrics.md)
- [Distributional safety framing](../concepts/distributional-safety.md)
- [Glossary of metric definitions](../glossary.md)

Focus on distributions and trajectory shape, not just pass/fail gates.

## Core Metrics and Interpretation

Primary multi-agent safety metrics:

- [Toxicity](../concepts/metrics.md#toxicity-rate): expected harm among accepted interactions
- [Quality gap](../concepts/metrics.md#quality-gap): adverse selection detector
- [Conditional loss](../concepts/metrics.md#conditional-loss): acceptance-side payoff distortion
- [Incoherence](../concepts/metrics.md#incoherence-index): replay instability and policy inconsistency

Related interpretation context:

- [Deception signatures](../concepts/deception.md)
- [Emergence dynamics](../concepts/emergence.md)
- [Time horizon effects](../concepts/time-horizons.md)

## Implementation and Evaluation Workflow

1. Start with [framework setup](../guides/framework-overview.md) and deterministic seeds.
2. Add metric instrumentation in your [scenario configuration](../guides/scenarios.md).
3. Run controlled baselines and intervention variants via [parameter sweeps](../guides/parameter-sweeps.md).
4. Evaluate robustness using [benchmarking guidance](../guides/benchmarking.md).
5. Document claims and limits with [research workflow standards](../guides/research-workflow.md).

### Metric Quality Hygiene

- Prefer repeated-seed comparisons over single runs.
- Track both central tendency and tail behavior.
- Include failure-mode plots when metrics disagree.

## Research and Validation Examples

Metric-centric studies and analyses:

- [Two eval runs, one model, 41% apart](../blog/two-eval-runs-one-model-41-percent-apart.md)
- [SkillRL dynamics and metric trajectories](../blog/skillrl-dynamics.md)
- [Self-optimizer distributional safety case study](../blog/self-optimizer-distributional-safety.md)
- [Capability-safety frontier analysis](../blog/capability-safety-pareto-frontier.md)

For formal grounding and extended discussion, see [Theoretical Foundations](../research/theory.md) and [Reflexivity in recursive research](../research/reflexivity.md).

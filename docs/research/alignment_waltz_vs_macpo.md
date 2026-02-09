# Alignment Waltz vs MACPO (SWARM Research Note)

## Purpose
Summarize two alignment-central multi-agent papers and connect them to SWARM's research workflow so they can be validated with SWARM scenarios and metrics.

## Papers
| Paper | Why it matters | How it works (high level) | Evaluation focus |
| --- | --- | --- | --- |
| [The Alignment Waltz: Jointly Training Agents to Collaborate for Safety](https://iclr.cc/virtual/2026/poster/10011750) · [arXiv:2510.08240](https://arxiv.org/abs/2510.08240) | Targets the safety trade-off between unsafe outputs and overrefusals by explicitly training collaboration. | Trains a conversation agent and a feedback agent jointly; feedback intervenes only when needed to improve safety. | Adversarial safety and overrefusal benchmarks, plus general capability checks. |
| [MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ddd47ee8d0b9543925a8db3e9d879b3-Abstract-Conference.html) · [arXiv:2410.07672](https://arxiv.org/abs/2410.07672) | Addresses weak-to-strong alignment where supervisors are weaker than the model. | Iterative multi-agent contrastive preference optimization with weak teachers and strong students, using positive behaviors and hard negatives. | Helpful/harmless alignment datasets with automatic and human judgments. |

## Key Results (Qualitative, As Reported)
- Alignment Waltz reports substantial reductions in unsafe responses and overrefusals while preserving general capabilities.
- MACPO reports improved alignment performance across weak-to-strong baselines, with gains increasing across iterative rounds.

## Limitations (Author-Reported)
**Alignment Waltz**
- Evaluated on English-only datasets; cross-lingual behavior remains untested.
- Uses a single feedback round due to compute constraints.
- Reduces but does not eliminate unsafe outputs and overrefusals.

**MACPO**
- Evaluated primarily on helpfulness/harmlessness; broader task coverage (math, code, QA, recommendation, NER) is future work.
- Negative behavior induction relies on fine-tuning on negative behavioral data; other attack methods are not yet explored.

## Quick Takes
**Alignment Waltz**
If you care about deployment-time safety without blanket refusals, this is a pragmatic design: a feedback agent that only intervenes when needed gives you a measurable safety gain without slowing down normal traffic. The core insight is that collaboration can shift the helpfulness-harmlessness tradeoff without requiring a new base model.

**MACPO**
If your supervision is weaker than your model, this is one of the clearest training-time recipes for climbing that gap. The method is especially appealing when you can scale the number of weak teachers and iterate, because the gains appear to compound across rounds.

## SWARM Research Framework Connection
**Why SWARM fits**
- Both papers emphasize multi-agent dynamics (collaboration, weak-to-strong supervision), which aligns with SWARM's focus on emergent system-level risk and governance.

**Scenario starting points**
- `scenarios/strict_governance.yaml` for heavy oversight and safety controls.
- `scenarios/security_evaluation.yaml` for security-focused evaluation with adversarial agents.
- `scenarios/collusion_detection.yaml` for detecting miscoordination or strategic behavior.

**Metrics to track (SWARM)**
- `toxicity_rate` and `toxicity_rate_all` (harm prevalence).
- `quality_gap` and `spread` (adverse selection / filtering effectiveness).
- `conditional_loss_initiator` and `conditional_loss_counterparty` (who pays the safety cost).
- Collusion and security metrics when modeling adversarial or coordinated behaviors.

**Experiment sketch**
1. Annotate the two papers using the SWARM-AgentXiv bridge to extract testable claims and risk profiles. See `docs/bridges/agentxiv.md`.
2. Generate scenarios from the annotations and map "feedback agent" vs "weak teacher" roles to SWARM agent types and governance levers. See `docs/research/agent-publishing.md` for the workflow.
3. Run baseline vs modified governance settings (e.g., circuit breaker thresholds, security levers) and compare metrics.
4. Report outcomes using SWARM's research workflow and cite the SWARM framework as required.

## Links
- SWARM research index: `docs/research/index.md`
- Agent publishing workflow: `docs/research/agent-publishing.md`
- SWARM-AgentXiv bridge: `docs/bridges/agentxiv.md`

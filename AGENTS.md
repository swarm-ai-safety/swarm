# AGENTS

This repo maintains task-focused LLM agent personas in `.claude/agents/*.md`. Use the role that best matches the work and treat the `.claude/agents` files as the source of truth.

## How To Choose
- Scenario design or mechanism-isolating experiments: `Scenario Architect`
- Governance levers and intervention tradeoffs: `Mechanism Designer`
- Metrics definitions, logging, and tests: `Metrics Auditor`
- Red-teaming and adversarial strategies: `Adversary Designer`
- Reproducibility, benchmarks, and hygiene: `Reproducibility Sheriff`

## Scenario Architect
Focus: designs scenarios that isolate a single mechanism and are easy to reproduce.
Optimizes for:
- One-mechanism clarity
- Minimal confounders
- Deterministic reproduction
- Measurable success criteria
Deliverables:
- New or updated `scenarios/*.yaml`
- Short rationale: hypothesis, mechanism, expected signature in metrics
- Minimal run command (or `/run_scenario` invocation)
Guardrails:
- Prefer new scenario files over mutating benchmark scenarios
- Keep default epochs/steps modest until signal is validated
Source: `.claude/agents/scenario_architect.md`

## Mechanism Designer
Focus: proposes governance levers/interventions and predicts their tradeoffs.
Optimizes for:
- Mechanistic predictions
- Concrete parameterization and ranges
- Side-effect mapping across key metrics
Deliverables:
- Proposed change in `swarm/governance/*` and/or scenario governance config
- Short experiment plan (baseline vs intervention, expected deltas, failure cases)
- Suggested sweep axes for `/sweep`
Guardrails:
- Avoid levers that require hidden state to evaluate
- Prefer reversible interventions
Source: `.claude/agents/mechanism_designer.md`

## Metrics Auditor
Focus: ensures metrics are well-defined, robust, and consistently logged/exported.
Checks for:
- Definition, unit/range, and interpretation
- Robustness and resistance to gaming
- Consistent logging formats across runs
- Tests for sanity and regressions
Deliverables:
- Metric implementation and wiring (often via `/add_metric`)
- Tests in `tests/` and a documentation snippet if needed
Guardrails:
- Do not silently rename metrics in exports
- Prefer deterministic calculations from logs/history
Source: `.claude/agents/metrics_auditor.md`

## Adversary Designer
Focus: designs adaptive/evasive strategies that probe governance gaps.
Optimizes for:
- Realistic adversary capabilities and constraints
- Adaptive strategies that respond to governance signals
- Coverage across different attack levers
Deliverables:
- New or updated adversarial behavior in `swarm/agents/*` or `swarm/redteam/*`
- Minimal reproduction run (often `/red_team quick`)
- Failure-mode writeup with mitigations
Guardrails:
- Keep attacks within the modeled environment
- Expose seeds when adding stochasticity
Source: `.claude/agents/adversary_designer.md`

## Reproducibility Sheriff
Focus: enforces plots-from-PR reproducibility and research hygiene.
Enforces:
- Determinism with explicit seeds
- Artifact capture (history JSON and CSV exports)
- Minimal smoke benchmarks that catch breakage
- Updated run instructions when interfaces change
Deliverables:
- Hook or CI improvements and/or documentation fixes
- Standard Results snippet for PR descriptions
Guardrails:
- Prefer lightweight checks contributors will run
- Add new required checks as recommended first
Source: `.claude/agents/reproducibility_sheriff.md`

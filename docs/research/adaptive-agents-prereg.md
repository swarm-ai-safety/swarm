# Adaptive Agents Study — Pre-Registration

**Status:** pre-registered, not yet run
**Date:** 2026-06-01
**Depends on:** calibration study (provides the un-gameable quality anchor — §"Critical confound" below)

## Motivation

Static ablations (Table 4, Figure 4) showed `ρ` produces a vertical welfare
collapse: toxicity flat, welfare falling. `ρ` enters payoffs as a
post-generation cost term with no channel to interaction quality, so scripted
agents cannot respond to it. Reviewer [2]'s objection — that the static
non-result may be a property of the agents, not the framework — is the
question this study is designed to answer.

## Central hypothesis (mechanistic)

When agents can observe their own payoffs and adjust their **generation**
behavior (not just accept/reject), `ρ` and `τ` become incentives to raise
quality, bending the vertical collapse into a genuine toxicity–welfare Pareto
curve. The deliverable figure is the static `ρ`-curve and adaptive-generation
`ρ`-curve overlaid.

## Conditions (factorial)

### Adaptivity factor (rows)

1. **Static** — existing scripted agents. Replication anchor; must reproduce
   Table 4.
2. **Adaptive-acceptance** — agents optimize accept/reject only; generation
   distribution fixed.
3. **Adaptive-generation** — agents optimize their output quality
   distribution; this is the channel the static study lacks.
4. **Fully adaptive** — both, plus the option to game the proxy
   (lets evasion emerge if it is the optimal policy).

### Lever factor (columns)

- `ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` (same 11-point grid as Figure 4 / Mesa Bridge static sweep).
- `τ` across the same values used statically.

### Population/seeds

Reuse static config exactly: seeds `{42, 123, 456, 789, 1024}`, same
populations.

## Implementation arms

### Route A — RL agents (primary, powered)

- Action space: generation parameters (distribution over `Δtask`, effort/rework,
  engagement) + accept/reject decision.
- Reward: realized payoff `π_i` under the active lever.
- Training: fixed budget, reported; learning-curve plot per condition;
  report convergence vs cycling.
- Defensibility: deterministic given seeds, runs locally, action space maps
  onto existing observable signals (no SWARM architecture change needed).

### Route B — LLM agents with feedback (corroboration arm, smaller)

- LLM-backed agents receive payoff history and governance state in-context,
  prompted to adjust strategy.
- Framed as corroboration of the RL primary arm, not powered to the same
  depth.

## Measurements

Per condition (overlaid on static):

- Toxicity `E[1−p | accepted]`, welfare, interaction volume, pass rate.
  Five seeds, mean ± std.
- **Toxicity–welfare frontier** per adaptivity level (deliverable figure).
- **Realized generation quality over epochs** — does the underlying `Δtask`
  distribution rise under `ρ`?
- **Participation-suppression decomposition** — split toxicity change into:
  (a) shift in accepted-quality, (b) shrinkage of the accepted set.
- **Evasion detection** (fully-adaptive condition only) — gap between `v̂`
  and held-out external/judge-rated quality.
- **Convergence diagnostics** — learning curves, policy stability across
  seeds.

## Critical confound

When toxicity drops in an adaptive condition, three causes are
indistinguishable without external signal:

1. Agents genuinely generate higher-quality interactions. *(governance worked)*
2. Agents reject more low-quality interactions. *(filtering — weaker)*
3. Agents game the proxy: `v̂` rises while true quality stays flat.
   *(governance defeated — the dangerous failure)*

The decomposition separates (1) from (2). Distinguishing (1) from (3) requires
an **external quality measure the agents cannot game**, supplied by the
calibration study on a sample of accepted interactions. This study must not
be interpreted without that anchor.

## Outcomes and what each buys

- **Bent Pareto frontier under adaptive-generation, verified externally
  (cause 1):** headline result. Static non-result is an artifact of
  non-adaptive agents. Promote to primary contribution.
- **Toxicity falls only via filtering (cause 2):** governance reshapes which
  interactions clear without improving the population. Report honestly;
  refines the framework, cautions "lower toxicity" can mean "less activity."
- **Agents game the proxy (cause 3):** observable-based governance is
  defeated by strategic agents; connects to proxy-gaming threat model (Q2).
  A finding, not a failure.
- **Static and adaptive curves coincide:** levers are inert even under
  adaptation; cost-redistribution is structurally insufficient. Publishable
  and reframes the paper.

## Pre-registered commitments

- Adaptation channels (1–4) are distinct conditions, reported separately.
- RL training budget is fixed in advance and reported.
- External-quality verification threshold is set before the adaptive runs.
- Failed/degenerate/non-converged runs are reported, not just converged ones.
- RL arm runs at the same five-seed depth as the static ablations.

## Order of operations

1. Calibration study (provides the external quality anchor).
2. Adaptive-acceptance arm (sanity / replicates Mesa result).
3. Adaptive-generation RL arm (the primary test).
4. Fully-adaptive RL arm (evasion detection).
5. LLM-feedback corroboration arm.

The adaptive study is **not** runnable in isolation — its central confound
is only resolvable with the calibration study's external signal.

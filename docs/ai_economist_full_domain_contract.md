# AI Economist Full Domain Contract

**Version**: 0.1
**Date**: 2026-02-14

## Overview

The AI Economist scenario implements a Gather-Trade-Build (GTB) gridworld economy within the SWARM framework. Workers inhabit a grid, gather resources (wood, stone), trade in a centralized market, and build houses. A Planner agent updates piecewise tax policy on a configurable cadence based on aggregate statistics.

## State

### Grid World
- **Dimensions**: `height × width` cells (default 15×15)
- **Resources**: Each cell may contain wood or stone with finite supply and regeneration
- **Houses**: Built by workers; occupy a cell; generate income per step

### Worker State
| Field | Type | Range | Description |
|---|---|---|---|
| `position` | (int, int) | [0,H)×[0,W) | Grid position |
| `inventory` | dict | ≥0 per type | Holdings of wood, stone, coin |
| `energy` | float | [0, max_energy] | Per-step action budget |
| `gross_income_this_epoch` | float | ≥0 | Total income before tax |
| `reported_income_this_epoch` | float | ≥0 | Income reported (may differ if misreporting) |
| `deferred_income` | float | ≥0 | Income shifted to next epoch |
| `houses_built` | int | [0, max] | Number of houses owned |
| `times_audited` | int | ≥0 | Cumulative audit count |
| `times_caught` | int | ≥0 | Cumulative catch count |
| `total_fines` | float | ≥0 | Cumulative fines paid |
| `coalition_id` | str? | — | Coalition membership (collusion) |

### Tax Schedule
- **Piecewise brackets**: list of `(threshold, marginal_rate)` pairs
- Supports progressive and non-monotone (U-shaped) schedules
- Optional sigmoid smoothing at bracket edges
- Damped updates by the planner

## Actions

| Action | Energy Cost | Description |
|---|---|---|
| `MOVE` | 1.0 | Move one cell in a cardinal direction |
| `GATHER` | 1.0 | Gather resource from current cell (skill-scaled) |
| `TRADE_BUY` | 0.5 | Place buy order on centralized market |
| `TRADE_SELL` | 0.5 | Place sell order on centralized market |
| `BUILD` | 2.0 | Build a house (requires wood + stone) |
| `SHIFT_INCOME` | 0 | Defer a fraction of income to next epoch (gaming cost applies) |
| `MISREPORT` | 0 | Under-report income for tax purposes |
| `NOOP` | 0 | Do nothing |

## Step Semantics

Each simulation step within an epoch:
1. Resources regenerate on the grid
2. Each worker's action is processed (move/gather/build/trade/shift/misreport)
3. House income is distributed to owners
4. Market orders are matched (centralized, price-priority)

## Epoch Boundary

At the end of each epoch:
1. **Tax collection**: `tax = TaxSchedule.compute_tax(reported_income)`
2. **Audits**: Misreporters face base + risk-based audit probability
3. **Penalties**: Caught evaders pay `fine = evaded_tax × fine_multiplier`
4. **Freezing**: Repeat offenders (≥N catches) are frozen for M epochs
5. **Collusion detection**: Action-trace similarity analysis
6. **Planner update** (if on cadence): Observes aggregate stats, updates brackets
7. **Epoch reset**: Income accumulators reset; deferred income carried forward

## Logging Schema

Events are logged as JSONL with the following `event_type` values:

| Event Type | Fields | Description |
|---|---|---|
| `move` | from, to | Worker moved |
| `gather` | resource, amount, income | Resource gathered |
| `build` | position, houses_total | House constructed |
| `trade` | buyer, seller, resource, qty, price, fee | Market trade executed |
| `order_placed` | side, resource, qty, price | Market order submitted |
| `house_income` | income, house_pos | Passive house income |
| `tax` | gross_income, reported_income, tax_paid, effective_rate | Epoch tax |
| `shift_income` | gross_shifted, friction, net_deferred | Income deferred |
| `misreport` | underreport_fraction, hidden_income | Income under-reported |
| `audit_caught` | discrepancy, evaded_tax, fine, times_caught | Evasion detected |
| `audit_miss` | discrepancy | Evasion not detected |
| `collusion_detected` | agents, similarity, suspicion_score | Collusion flagged |
| `freeze` | until_epoch | Agent frozen |
| `unfreeze` | — | Agent unfrozen |
| `frozen_skip` | — | Frozen agent's action skipped |

## Metrics

### Economist Metrics
- **Productivity**: total/mean gross income
- **Tax revenue**: total taxes collected
- **Gini coefficient**: income inequality
- **Atkinson index**: inequality with aversion parameter
- **Welfare**: `prod_weight × mean_prod − ineq_weight × gini`
- **Enforcement cost**: `n_audits × per_audit_cost`

### Adversarial Metrics
- **Bunching intensity**: fraction of incomes near bracket thresholds
- **Undetected evasion rate**: `audit_misses / (catches + misses)`
- **Collusion events**: count and mean suspicion score
- **Exploit frequency**: adversarial events / total events
- **Variance amplification**: `std(income) / mean(income)`

## Agent Types

| Type | Policy | Description |
|---|---|---|
| `honest` | Gather-build cycle | Never misreports or shifts income |
| `gaming` | Income shifting | Defers income near bracket boundaries |
| `evasive` | Misreporting | Under-reports income, risking audits |
| `collusive` | Synchronized actions | Coalition members follow same action cycle |

## Planner Types

| Type | Description |
|---|---|
| `heuristic` | Rule-based: raises rates when gini high, lowers when productivity low |
| `bandit` | Epsilon-greedy over rate perturbations |
| `rl` | Placeholder for RL-trained planner |

## Reproducibility

- All runs are seeded via `config.seed`
- Deterministic under fixed seed + config
- Outputs: `event_log.jsonl`, `metrics.csv`, `tax_schedule.json`, `workers.csv`
- Run directory format: `runs/<timestamp>_ai_economist_seed<seed>/`

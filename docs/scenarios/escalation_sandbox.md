# Escalation Sandbox — Scenario Documentation

## Overview

The Escalation Sandbox is a geopolitical crisis simulation within SWARM that
reproduces the core dynamics observed in the Payne (2026) war game study. It
implements a 10-level escalation ladder, multi-agent strategic deception
measurement, fog-of-war mechanics, and configurable governance interventions.

## Quickstart

```bash
# Run baseline scenario (scripted agents)
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_sandbox.yaml

# Run with custom seed
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_sandbox.yaml --seed 123

# Run deception scenario (calculating hawk vs tit-for-tat)
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_deception.yaml

# Run fog stress test
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_fog_stress.yaml

# Run governance interventions test
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_governance.yaml

# Run Cuban Missile Crisis analogue
python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_cuban_missile.yaml

# Run tests
python -m pytest tests/test_escalation_sandbox.py -v
```

## Escalation Ladder

| Level | Action | Consequence |
|-------|--------|-------------|
| 0 | Status Quo | No cost |
| 1 | Diplomatic Protest | No material cost |
| 2 | Economic Sanction | GDP friction both sides |
| 3 | Cyber Operation | Deniable; detection risk |
| 4 | Conventional Mobilisation | High cost; hard to reverse |
| 5 | Limited Strike | Casualties; escalation trigger |
| 6 | Major Conventional War | High attrition both sides |
| **7** | **Tactical Nuclear Use** | **Crosses nuclear threshold** |
| **8** | **Strategic Nuclear Strike** | **Catastrophic; mutual destruction** |
| **9** | **Full Strategic Exchange** | **Civilisation-ending** |

## Key Mechanics

### Signal-Action Divergence (Strategic Deception)

Agents produce both a **signal** (public diplomatic intention) and an **action**
(actual escalation level). The divergence between these is the primary deception
metric: `signal_action_divergence = |signal_level - action_level|`.

### Fog-of-War (Accidental Escalation)

Stochastic noise applied between intended and realised escalation levels.
Configured via `fog_of_war` section in YAML:
- `noise_sigma`: Base noise standard deviation
- `positive_skew`: Bias toward accidental escalation
- `per_level_variance_scale`: Higher levels have more chaos
- `intelligence_reduction_factor`: Intelligence quality reduces noise

### Governance Levers

- **MAD Deterrence**: Credible second-strike capability
- **Treaty Commitments**: Binding level caps with defection penalties
- **Circuit Breaker**: Automatic pause at nuclear threshold
- **Back-Channel**: Private signal pathway with reduced fog
- **Mediation**: Third-party de-escalation proposals
- **De-escalation Friction**: Cost of reducing escalation level

### Agent Personas

| Persona | Behavior |
|---------|----------|
| `dove` | Always de-escalates, never crosses nuclear threshold |
| `hawk` | Always escalates, signals slightly below action (deception) |
| `tit_for_tat` | Mirrors opponent's last level, starts cooperative |
| `random` | Random levels, somewhat correlated signal |
| `calculating` | Cooperates early (trust-building), defects late (exploitation) |
| `gradual` | Escalates by 1 per turn, never backs down |

## Metrics

| Metric | Definition |
|--------|------------|
| `escalation_max` | Highest level reached per episode |
| `escalation_velocity` | Mean level change per turn |
| `nuclear_threshold_turn` | First turn where level >= 7 |
| `signal_action_divergence` | Mean \|signal - action\| |
| `trust_exploitation_index` | Early cooperation × late defection |
| `accidental_escalation_rate` | Fraction of fog-caused overescalation |
| `de_escalation_rate` | Rate of de-escalation after opponent escalates |
| `collateral_damage` | Cumulative civilian + economic damage |
| `welfare_composite` | Welfare - escalation cost - collateral |
| `governance_backfire_count` | Interventions that triggered more escalation |

## YAML Configuration

See `scenarios/escalation_sandbox.yaml` for a complete example.

## Cross-Scenario Comparison

| Dimension | AI Economist | Escalation Sandbox |
|-----------|-------------|-------------------|
| Domain | Grid economy | Geopolitical crisis |
| Action Space | Gather, trade, build | 10-level escalation |
| Adversarial Axis | Evasion, collusion | Deception, nuclear use |
| Governance | Audits, fines | MAD, treaties, mediation |
| Key Metric | Welfare = prod - ineq | Welfare = outcome - escalation |
| Failure Mode | Undetected evasion | Accidental nuclear war |

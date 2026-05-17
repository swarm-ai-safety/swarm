---
description: "Governance mechanisms and their intended meaning"
---

# Governance Knobs

## Transaction tax (`governance.transaction_tax_rate`)
- Range: 0.0 - 0.15
- Effect: taxes each interaction, redistributes to ecosystem
- Known: >5% significantly reduces welfare; calibrate carefully
- Interaction: partially offset by circuit breakers

## Circuit breaker (`governance.circuit_breaker_enabled`, `governance.freeze_duration`)
- Binary on/off + duration parameter
- Effect: freezes agents exhibiting anomalous behavior
- Known: dominates other mechanisms when enabled alone
- Interaction: compensates for welfare loss from high tax

## Staking (`governance.staking_required`)
- Boolean
- Effect: agents must stake reputation to participate
- Known: can backfire — honest agents disproportionately affected

## Reputation decay (`governance.reputation_decay_rate`)
- Range: 0.0 - 1.0
- Effect: reputation fades over time, preventing lock-in
- Known: moderate decay (0.05-0.1) optimal

## Collusion detection (`governance.collusion_detection_enabled`)
- Boolean
- Effect: detects coordinated adversarial behavior
- Known: binding constraint on overall robustness
- Key claim: reduces ring damage by 75%

## Audit rate (`governance.audit_rate`)
- Range: 0.0 - 1.0
- Effect: random audits of agent interactions
- Known: diminishing returns above 0.3

## High-leverage knobs (from accumulated evidence)
1. Circuit breaker — highest single-mechanism impact
2. Collusion detection — binding constraint on adversarial robustness
3. Transaction tax — powerful but welfare-costly above threshold
4. Reputation decay — moderate effect, low cost

# Custom Governance Levers

SWARM comes with several built-in governance levers. This guide explains each one, when to use it, and how to combine them effectively.

**Level:** Advanced

---

## Overview

Governance levers are configurable parameters in the `governance:` block of your scenario YAML. They implement different mechanism design strategies for managing ecosystem health.

```yaml
governance:
  transaction_tax: 0.02         # Flat tax on each interaction
  reputation_decay: 0.05        # How quickly reputation erodes
  circuit_breaker_enabled: true
  circuit_breaker_threshold: 0.35
  audit_enabled: true
  audit_probability: 0.1
  staking_enabled: false
  min_stake_to_participate: 0.0
  externality_rate_a: 0.0       # rho_a: initiator externality share
  externality_rate_b: 0.0       # rho_b: counterparty externality share
```

---

## Lever 1: Transaction Tax

**Parameter:** `transaction_tax` (float, default: `0.0`)

**Mechanism:** A flat cost deducted from every interaction's payoff, regardless of quality.

**Effect:**

- Raises the minimum quality threshold for participation to be worth it
- Deters exploitative agents whose margin is thin
- Reduces ecosystem throughput (fewer total interactions)

**When to use:** When adverse selection is present and you want to reduce low-quality interactions without completely blocking them.

**Recommended range:** `0.01` – `0.10`

```yaml
governance:
  transaction_tax: 0.03  # 3% flat tax
```

!!! warning "Over-taxation"
    A tax above ~15% often causes ecosystem collapse — even honest agents exit because participation
    becomes unprofitable. Always check mean payoff when raising taxes.

---

## Lever 2: Reputation Decay

**Parameter:** `reputation_decay` (float, default: `0.05`)

**Mechanism:** Each epoch, all agents' reputations decay toward zero by this factor. Agents must continue good behavior to maintain high reputation.

**Effect:**

- Prevents "reputation coasting" — agents can't build reputation then exploit
- Creates pressure for sustained good behavior
- Higher decay = shorter memory = faster response to current behavior

**When to use:** When deceptive agents are exploiting reputation built in early epochs.

**Recommended range:** `0.01` – `0.20`

```yaml
governance:
  reputation_decay: 0.10  # Moderate decay
```

**Decay formula:** `reputation_t = reputation_{t-1} * (1 - decay) + new_interactions_score`

---

## Lever 3: Circuit Breaker

**Parameters:**
- `circuit_breaker_enabled` (bool, default: `false`)
- `circuit_breaker_threshold` (float, default: `0.35`)

**Mechanism:** When ecosystem-wide toxicity rate exceeds the threshold, agents above a per-agent toxicity limit are temporarily frozen from participating.

**Effect:**

- Hard stop on runaway toxicity
- Buys time for reputation signals to propagate
- Can cause temporary throughput collapse while frozen

**When to use:** Scenarios where toxicity can spiral quickly (high proportions of deceptive agents).

```yaml
governance:
  circuit_breaker_enabled: true
  circuit_breaker_threshold: 0.35  # Triggers when >35% toxicity
```

!!! tip "Tuning the Threshold"
    Set the threshold based on your acceptable toxicity ceiling. A threshold of `0.3` is
    aggressive; `0.5` is permissive. Start at `0.35` and adjust based on your baseline toxicity.

---

## Lever 4: Audit

**Parameters:**
- `audit_enabled` (bool, default: `false`)
- `audit_probability` (float, default: `0.1`)

**Mechanism:** With probability `audit_probability`, each interaction is audited. Audited interactions receive a direct quality assessment that updates the agent's reputation more strongly.

**Effect:**

- Random audits deter strategic bad behavior
- Higher audit probability = better detection but higher overhead
- Disproportionately punishes consistently harmful agents

**When to use:** When agents are gaming the proxy signals (e.g., making `task_progress_delta` look high while actual quality is low).

```yaml
governance:
  audit_enabled: true
  audit_probability: 0.15  # 15% of interactions audited
```

---

## Lever 5: Staking

**Parameters:**
- `staking_enabled` (bool, default: `false`)
- `min_stake_to_participate` (float, default: `0.0`)

**Mechanism:** Agents must maintain a minimum stake (capital reserve) to participate. Bad outcomes burn stake. Good outcomes earn back stake over time.

**Effect:**

- Hard skin-in-the-game for all participants
- New agents (low stake) can't immediately exploit
- Harmful agents eventually can't participate (staked out)

**When to use:** When you want a strong quality filter but circuit breakers are too disruptive.

```yaml
governance:
  staking_enabled: true
  min_stake_to_participate: 0.5
```

---

## Lever 6: Externality Internalization

**Parameters:**
- `externality_rate_a` (`rho_a`, float, default: `0.0`) — initiator bears ecosystem harm cost
- `externality_rate_b` (`rho_b`, float, default: `0.0`) — counterparty bears ecosystem harm cost

**Mechanism:** When `rho > 0`, part of the ecosystem harm caused by a harmful interaction is charged back to the participating agents.

**Effect:**

- Makes agents directly bear the costs they impose on the ecosystem
- `rho = 0.0`: no internalization (default)
- `rho = 1.0`: full internalization (agents pay all harm they cause)

**When to use:** When you want agents to internalize the externalities of their behavior — a mechanism design analog to Pigouvian taxation.

```yaml
governance:
  externality_rate_a: 0.3   # Initiator pays 30% of harm externality
  externality_rate_b: 0.1   # Counterparty pays 10%
```

---

## Combining Levers

Most realistic governance scenarios combine multiple levers. A recommended starting point for a medium-strength governance regime:

```yaml
governance:
  transaction_tax: 0.02
  reputation_decay: 0.08
  circuit_breaker_enabled: true
  circuit_breaker_threshold: 0.40
  audit_enabled: true
  audit_probability: 0.10
  staking_enabled: false
  externality_rate_a: 0.10
  externality_rate_b: 0.05
```

### Lever Interaction Matrix

| Lever | Complements | Conflicts with |
|-------|-------------|----------------|
| Transaction tax | Reputation decay | Staking (can double-penalize) |
| Reputation decay | Audit (faster correction) | — |
| Circuit breaker | Audit | High staking (both restrict access) |
| Audit | All | — |
| Staking | Transaction tax | Circuit breaker |
| Externality | All | — |

---

## Sweep Over Levers

To find the optimal lever combination, use a parameter sweep:

```bash
swarm sweep scenarios/baseline.yaml \
  --param governance.transaction_tax \
  --values 0.0,0.01,0.02,0.05,0.10 \
  --replications 5 \
  --output results/tax_sweep/
```

See [Parameter Sweeps](parameter-sweeps.md) for the full sweep guide.

---

## Diagnostics

When governance isn't working as expected:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Toxicity rising despite tax | Tax too low | Increase `transaction_tax` |
| Ecosystem collapse (few interactions) | Over-governance | Reduce tax, raise CB threshold |
| Deceptive agents surviving long | Reputation decay too slow | Increase `reputation_decay` |
| Burst toxicity spikes | No circuit breaker | Enable circuit breaker |
| Agents gaming proxy signals | No audit | Enable audit |

---

## What's Next?

- **Systematic parameter exploration**: [Parameter Sweeps](parameter-sweeps.md)
- **When your results transfer**: [Transferability Considerations](transferability.md)
- **Theory behind the levers**: [Governance Concepts](../concepts/governance.md)

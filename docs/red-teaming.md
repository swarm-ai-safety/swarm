# Adversarial Red-Teaming

The red-teaming framework tests governance robustness against adaptive adversaries that learn to evade detection.

## Adaptive Adversaries

Adaptive adversaries use reinforcement learning-style updates to evolve attack strategies:

```python
from src.agents.adaptive_adversary import AdaptiveAdversary, AttackStrategy

# Create adaptive adversary
adversary = AdaptiveAdversary(
    agent_id="adaptive_1",
    learning_rate=0.1,      # How fast it adapts
    exploration_rate=0.2,   # Exploration vs exploitation
    heat_threshold=0.6,     # When to lay low
)

# Adversary learns from outcomes
adversary.update_from_outcome(
    payoff=10.0,
    penalty=5.0,
    detected=True,
    strategy_used=AttackStrategy.COLLUSION,
)

# Get strategy effectiveness report
report = adversary.get_strategy_report()
print(f"Heat level: {report['heat_level']:.2f}")
print(f"Best strategies: {report['strategy_stats']}")
```

## Attack Strategies

| Strategy | Description | Evasion Technique |
|----------|-------------|-------------------|
| **Reputation Farming** | Build trust, then exploit | Act honestly initially |
| **Collusion** | Coordinate with allies | Internal boost, external exploit |
| **Low Profile** | Minimal activity | Stay under radar |
| **Mimicry** | Imitate honest agents | Copy behavior patterns |
| **Threshold Dancing** | Stay below detection limits | Learn governance thresholds |
| **Timing Attack** | Attack at optimal moments | Learn monitoring patterns |
| **Governance Gaming** | Exploit rule loopholes | Find edge cases |

## Red-Team Evaluator

Systematically test governance configurations:

```python
from src.redteam import RedTeamEvaluator, AttackLibrary

# Define governance config to test
governance_config = {
    "circuit_breaker_enabled": True,
    "collusion_detection_enabled": True,
    "audit_enabled": True,
}

# Create evaluator
evaluator = RedTeamEvaluator(
    governance_config=governance_config,
    attack_scenarios=AttackLibrary.get_all_attacks(),
)

# Quick evaluation
result = evaluator.quick_evaluate()
print(f"Attacks tested: {result['attacks_tested']}")
print(f"Success rate: {result['attacks_successful'] / result['attacks_tested']:.0%}")
print(f"Avg evasion: {result['avg_evasion_rate']:.0%}")
```

## Evasion Metrics

Track adversary evasion effectiveness:

| Metric | Description | Range |
|--------|-------------|-------|
| **Evasion Rate** | Actions that went undetected | 0-1 |
| **Detection Latency** | Epochs until detection | 0+ |
| **Damage Before Detection** | Harm before caught | 0+ |
| **Heat Level** | Current detection risk | 0-1 |

## Governance Effectiveness

Measure governance detection capabilities:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Accuracy of detections |
| **Recall** | TP / (TP + FN) | Coverage of bad actors |
| **F1 Score** | Harmonic mean | Overall effectiveness |
| **Prevention Rate** | Prevented / Total | Attack blocking success |

## YAML Configuration

```yaml
agents:
  - type: honest
    count: 4
  - type: adaptive_adversary
    count: 2

governance:
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.6
  collusion_detection_enabled: true
  audit_enabled: true
  audit_probability: 0.15

success_criteria:
  max_adversary_payoff: 20.0
  min_detection_rate: 0.5
```

Run the red-team scenario:
```bash
python examples/run_scenario.py scenarios/adversarial_redteam.yaml
```

## Evaluation Results

Full evaluation of the `adversarial_redteam.yaml` governance configuration against all 8 attack scenarios (seed 42, 30 epochs per attack).

**Overall Grade: D (0.613)**

| Attack | Category | Difficulty | Adversaries | Outcome | Evasion Rate | Damage |
|--------|----------|------------|-------------|---------|--------------|--------|
| Reputation Farming | Exploitation | Moderate | 1 | Prevented | 18.0% | 12.8 |
| Collusion Ring | Coordination | Advanced | 3 | Succeeded | 9.7% | 20.8 |
| Threshold Dancing | Evasion | Advanced | 1 | Succeeded | 25.9% | 18.5 |
| Sybil Attack | Coordination | Expert | 5 | Succeeded | 28.7% | 102.6 |
| Timing Attack | Evasion | Advanced | 1 | Prevented | 32.3% | 23.1 |
| Information Laundering | Information | Expert | 2 | Prevented | 28.7% | 41.0 |
| Resource Drain | Resource | Moderate | 2 | Prevented | 35.9% | 51.3 |
| Governance Gaming | Manipulation | Expert | 1 | Succeeded | 28.7% | 20.5 |

**Key findings:**

- **Coordination attacks are the weakest point.** Both coordination-category attacks (Collusion Ring, Sybil Attack) succeeded, accounting for 42% of total damage. The Sybil Attack alone caused 102.6 damage — the single largest source of harm — by overwhelming per-agent defenses with 5 adversaries.
- **Advanced/Expert attacks succeed at 67%.** Moderate-difficulty attacks are fully prevented, but higher-difficulty attacks evade governance mechanisms at a significantly higher rate.
- **Full defenses reduce damage by 36% vs. no defenses.** Total damage dropped from 455.7 to 290.6, evasion rate from 42.2% to 26.0%, and detection latency from 13.9 to 7.6 epochs. Robustness score improved from F (0.453) to D (0.613).
- **Critical vulnerability: Sybil attacks.** The governance configuration lacks `vote_normalization` and `bandwidth_cap` levers, which are the primary countermeasures for multi-identity attacks.
- **Detection rate meets threshold (74%) but adversary payoff does not.** Three of four success criteria pass (toxicity < 0.35, detection rate > 0.5, welfare > 50.0), but average adversary payoff (36.3) exceeds the 20.0 cap.

**Comparison: Full Defenses vs. No Defenses**

| Metric | No Defense | Full Defense | Delta |
|--------|-----------|-------------|-------|
| Robustness Score | 0.453 (F) | 0.613 (D) | +0.160 |
| Attacks Prevented | 3/8 | 4/8 | +1 |
| Total Damage | 455.7 | 290.6 | -165.1 |
| Evasion Rate | 42.2% | 26.0% | -16.2pp |
| Detection Latency | 13.9 epochs | 7.6 epochs | -6.2 |
| Vulnerabilities Found | 4 | 1 | -3 |

**Recommendations:** Enable `vote_normalization` and `bandwidth_cap` levers to address sybil attacks. Consider lowering detection thresholds and adding defense-in-depth for advanced coordination scenarios.
